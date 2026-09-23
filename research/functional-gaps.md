# NovaCron functional-gap census (2026-09-22, tree @ ac73ea45)

Read-only audit of the code that actually runs (`backend/cmd/api-server` binary +
`backend/core/vm`, `backend/core/storage`, `backend/core/orchestration`). Every claim
below names the file/symbol where the capability lives or should live. Classifications:
**TRUE GAP** = absent where a paying tenant hits it; **PARTIAL** = mechanism exists but
unreachable/incomplete; **ABSENT** = deliberately not built, decision needed.

```json
{
  "gaps": [
    {
      "gap": "multi_tenancy_enforcement",
      "rank": 1,
      "currentState": "PARTIAL (read-filtered list only; all other endpoints unscoped)",
      "claim": {
        "where": "backend/cmd/api-server/main.go",
        "evidence": [
          "LIST /vms (main.go:832-853) IS org-filtered but SOFT by its own comment ('still soft ... RLS is a separate bead novacron-ok7') and has two exits: orgID==\"\" (JWT missing tenant claim) falls into the UNFILTERED query, so a token with no tenant claim sees every tenant's VMs; a legacy non-uuid claim ('default') is compared against the uuid column organization_id -> Postgres invalid-uuid cast -> 500 for that tenant.",
          "GET /vms/{id} (main.go:898) reads the row and returns organization_id with NO ownership check — cross-tenant read.",
          "DELETE /vms/{id} (main.go:936) kills qemu + drops the row with no org check — cross-tenant DESTRUCTIVE delete by any authenticated user.",
          "POST /vms/{id}/start|stop (registerVMPowerRoute, main.go:1327) — no org check.",
          "POST /vms/{id}/migrate (main.go:1646) and /migrate/async + GET /migrate/jobs/{id} (migrate_jobs.go:245, :300) — no org check; any tenant can move/kill another tenant's VM.",
          "GET /vms/{id}/metrics (main.go:976), GET /monitoring/vms (main.go:1002, lists ALL vms id/name/state) — no scoping.",
          "Interfaces /vms/{vm_id}/interfaces GET/POST/PUT/DELETE (main.go:1075-1253) — no org check.",
          "POST /transfers (fabric_transfers.go:372) — any authed user can migrate any vm_id to any registered peer.",
          "Fabric jobs: fabric_jobs table has NO org column; listFabricJobs/getFabricJob (fabric_jobs.go:594,612) SELECT with no org filter — cross-tenant job list/get/cancel.",
          "NULL-org exit path: createVMLocal (cluster.go:352) stamps org via scalar subquery; an org id this node's organizations table can't resolve is silently stamped NULL, and legacy non-uuid labels are dropped to NULL (orgLabelForVM). NULL-org VMs are invisible to their own tenant's filtered LIST and visible only to admins — a silent tenant-griefing / lost-ownership path.",
          "Spoofing: organization_id rides the /internal/vms/create wire payload (clusterCreateSpec.OrganizationID, cluster.go:228) and is trusted by the peer; anyone holding a node credential stamps any org -> billing attribution forgery (usageOrgForVM bills the VM row's org). There is no uploader_id concept at all; nothing binds a process VM to its submitter beyond this forgeable field."
        ]
      },
      "minimalFix": "One ownership predicate vmVisibleTo(ctx, db, vmID) (org match OR admin; NULL-org => admin-only) called by every per-VM handler above; add organization_id to fabric_jobs stamped at submit; treat missing/non-uuid tenant claims as 'no listable VMs' instead of unfiltered; add owner-subject binding (owner_id from user_id claim) to createVMLocal wire contract. Smallest test: sqlmock+token test — tenant-B token DELETEs/GETs tenant-A's VM id, must return 403/404; today returns 200 (delete) / 200 (get)."
    },
    {
      "gap": "gpu_pci_passthrough",
      "rank": 5,
      "currentState": "TRUE GAP",
      "claim": {
        "where": "backend/core/vm/driver_kvm_enhanced.go",
        "evidence": [
          "SupportsGPUPassthrough() returns false, comment 'Not implemented yet' (driver_kvm_enhanced.go:1375-1378); GetCapabilities reports GPUDevices: [] and SupportsGPUPassthrough:false (line ~1432).",
          "buildQEMUArgs (lines ~838-935) emits virtio-net/blk/balloon/rng and pcie-root-ports, NO hostdev/vfio-pci args; VMConfig has no PCI/GPU device list consumed at launch.",
          "hardware_virtualization.go has GPUDevice/PCIAddress/GPUModePassthrough inventory structs incl. VMConfig.PCIDevices — an inventory model nothing in the launch path consumes; the Libvirt driver that could generate <hostdev> is drivers/kvm/libvirt_driver.go.disabled.",
          "No resize spec exists at the API at all: no REST route calls HotPlugDevice/ConfigureCPUPinning/ConfigureNUMA (grep over backend/cmd + backend/api finds zero callers); qemu-img resize is used only inside initial disk creation (driver_kvm_enhanced.go:1085)."
        ]
      },
      "minimalFix": "Jenkins-stage-style: add Devices []DeviceConfig to VMConfig persisted in config.json; buildQEMUArgs appends '-device vfio-pci,host=<pci>' per device; RequiresRestart. Smallest functional test: same pattern as TestBuildQEMUArgsIOThreadOptIn — construct driver, set Config.Devices, assert vfio-pci arg; then real-qemu test with a fake/unbound PCI addr asserting qemu rejects at launch (exercises wiring, not hardware)."
    },
    {
      "gap": "crash_resilience",
      "rank": 3,
      "currentState": "TRUE GAP (detection exists, reaction does not)",
      "claim": {
        "where": "backend/core/vm/driver_kvm_enhanced.go monitorVM (line 1269)",
        "evidence": [
          "monitorVM's cmd.Wait() goroutine is the ONLY crash watcher: on error exit it log.Printf's 'exited with error' and sets in-memory StateFailed; on clean exit StateStopped. No restart policy, no backoff, no event/alert emission.",
          "The vms DB row is NOT updated on crash — it keeps reading 'running' until the next api-server boot reconcile (reconcileVMState only runs at startup, main.go:92). GET /vms/{id} paper-reads live state via liveVMState, so DB and API disagree after a crash.",
          "Per-VM diagnostics: serial console captured to console.log (driver_kvm_enhanced.go:858) — exists — but the exit reason/stderr tail/exit code is not recorded per-VM anywhere queryable; nothing lives in-process UI-wise (no /vms/{id}/events endpoint, vm_monitor.go is an unused richer alerting model).",
          "StateFailed/StateRestarting exist as constants (vm.go:63) — vocabulary present, mechanism absent. Process driver (fabric jobs) has re-adoption but not crash-restart."
        ]
      },
      "minimalFix": "In monitorVM: persist state + exit detail (exit code, stderr tail, timestamp) to vms row on exit; optional restart_policy field ('no'|'on-failure', max N, exponential backoff) re-invoking Start. Smallest test: launch real qemu, SIGKILL it, assert (a) DB row updated to failed with reason without api-server restart, (b) with policy=on-failure a new pidfile/pid appears within backoff window — both fail today."
    },
    {
      "gap": "volume_hot_ops",
      "rank": 6,
      "currentState": "PARTIAL",
      "claim": {
        "where": "backend/api/graphql/schema.volume.graphql + backend/core/storage/storage_manager.go",
        "evidence": [
          "Exposed volume surface = createVolume/changeVolumeTier/list only (schema.volume.graphql:6-12). No resize, no attach, no detach mutation.",
          "StorageManager.ResizeVolume (storage_manager.go:299) exists but is unrouted; it resizes the backing FILE asynchronously (goroutine, revert-on-error) and tells neither qemu (no QMP block_resize) nor the guest (no '/proc' notification; zero occurrences of resize2fs/growpart/xfs_growfs anywhere in backend/) — running guests never see the new size.",
          "Disk hot-attach/detach machinery DOES exist at driver level: HotPlugDevice/HotUnplugDevice (driver_kvm_enhanced.go:1462, :1652, QMP blockdev-add + device_add virtio-blk / device_del) with real-qemu tests — but no HTTP route calls them, so a tenant cannot attach a created volume.",
          "Volumes carry no org/tenant field — same unscoped-write problem as VMs if routed."
        ]
      },
      "minimalFix": "Route three mutations: resizeVolume (StorageManager.ResizeVolume + QMP 'block_resize' on the running VM's block node + documented guest-side grow step or a guest-agent hook), attachVolume/detachVolume calling HotPlugDevice('disk')/HotUnplugDevice. Smallest test: create volume via GraphQL, hot-attach to a running test VM, assert query-block shows the new blockdev (test pattern already exists in driver_kvm_hotplug_test.go); resize attached volume and assert query-block size grew — resize half fails today (no block_resize)."
    },
    {
      "gap": "fabric_admission_control_and_dispatch_retry",
      "rank": 2,
      "currentState": "TRUE GAP",
      "claim": {
        "where": "backend/cmd/api-server/fabric_jobs.go submitFabricJob / cluster.go dispatchCreateToPeerAs",
        "evidence": [
          "Submit is synchronous place->create->start with no queue, no concurrency cap, no backpressure: every POST spawns a process VM. Rate limiting exists ONLY on the login endpoint (middleware.go loginRateLimiter, main.go:551) — write APIs are unbounded.",
          "Placement reads a capacity SNAPSHOT (localNodeCapacity, cluster.go:88: MemAllocated from vmManager.ClusterUsage) with no atomic reservation: 1000 concurrent submitters each see the same free-memory snapshot and all pass placement before any VM is recorded — classic oversubscribe. errNoNodeFits exists but fires only against the stale snapshot.",
          "dispatchCreateToPeerAs (cluster.go:388): one 60s-timeout POST, NO retry — a transient blip fails the whole job submit with 502. (Retry+backoff exists in exactly one RPC: migration abort, vm_operations.go:1168.)",
          "Durability asymmetry is documented but real: fabric_jobs rows survive restart; transfers deliberately do not (fabric_transfers.go:14-17)."
        ]
      },
      "minimalFix": "(a) bounded admission: per-org inflight-job semaphore + 429 with Retry-After on /compute/jobs and /vms POST; (b) atomic reservation: increment a reservations counter keyed by node BEFORE placement returns, released on start failure; (c) wrap dispatchCreateToPeerAs in the same 3-attempt backoff pattern already used in vm_operations.go:1168. Smallest test: httptest race — N=50 concurrent submits against a node with memory for 2 jobs; assert <=2 reach 'running' and the rest get 429/503 deterministically; plus a peer-server-that-fails-once test asserting dispatch succeeds on retry (fails today)."
    },
    {
      "gap": "node_evacuation_drain",
      "rank": 4,
      "currentState": "PARTIAL (exists in a binary that is not the one running)",
      "claim": {
        "where": "backend/core/orchestration/engine.go EvacuateNode (line 135) + backend/orchestration/adapters.go",
        "evidence": [
          "A real evacuation engine exists (DefaultOrchestrationEngine + NewDefaultEvacuationHandler: list-VMs-by-node -> select-target -> Migrate per VM), and core-server wires it (cmd/core-server/main.go:41-42, POST /nodes/{id}/evacuate, operator-gated).",
          "The RUNNING api-server binary never calls SetEvacuationHandler — the only caller in cmd/ is core-server. api/vm/router.go:297 (/cluster/nodes/{node_id}/drain) and api/vm/migration_handlers_enhanced.go:97 (/api/v2/migrations/evacuate) are registered inside the api/vm package, which cmd/api-server/main.go does not mount (its route list main.go:2370+ contains no drain/evacuate).",
          "So on the production surface the only option is issuing one-shot POST /vms/{id}/migrate per VM by hand — no capacity check, no ordering, no progress tracking, no node cordon (placement keeps landing new work on the draining node)."
        ]
      },
      "minimalFix": "Register one route in cmd/api-server: POST /cluster/nodes/{id}/drain that (1) marks the node draining in node profiles so placeFabricJob/clusteredCreate skip it, (2) enumerates vms WHERE node_id=$1, (3) runs MigrateVM sequentially (serial = safe default) with a drain-jobs table mirroring migrate_jobs for progress. Smallest test: two fake peers, 2 VMs on node A; drain A; assert both migrated to B, new creates land on B only, and drain job reports completion."
    },
    {
      "gap": "shared_disk_state_sharing_qcow2_over_nfs",
      "rank": 7,
      "currentState": "ABSENT (assumption-only); recommend NO for now",
      "claim": {
        "where": "backend/core/vm/driver_kvm_enhanced.go — KVMVMInfo.SharedStorage / lock=off (lines ~48-56, :839-842)",
        "evidence": [
          "Shared-storage ASSUMPTION exists for migration: SharedStorage makes source+dest open the same disk path with file.locking=off. But nothing in NovaCron makes that path actually shared — no NFS server/client management anywhere; core/storage NFS paths are log-stubs ('In a real implementation...' volume_operations.go:41-45,161,270,303); vm_storage.go has StorageTypeNFS constant only.",
          "No blockdev-mirror / active block replication for a network-backed image; the block-migrate path copies the disk between node-local stores instead (driver_kvm_migrate.go), which is the working design today.",
          "qcow2-over-NFS for concurrent multi-writer is also UNSAFE generally (qcow2 has no cluster locking); even single-writer it adds read latency to every guest block IO."
        ]
      },
      "minimalFix": "Decision, not code: document that shared-disk migration requires the OPERATOR to pre-mount a shared export at the same path on both nodes (already what SharedStorage assumes), and defer blockdev-mirror-over-NFS state-sharing as out of scope — the existing block-migrate path covers portability. Smallest test that proves the current contract: two mounts of one NFS export at vmBase on both peers, shared-storage migration round-trip — driver_kvm_block_migrate_* tests already pin the non-shared path."
    },
    {
      "gap": "soft_realtime_cpu_numa_cgroups",
      "rank": 8,
      "currentState": "PARTIAL (driver primitives exist, no tenant/operator surface)",
      "claim": {
        "where": "backend/core/vm/driver_kvm_enhanced.go ConfigureCPUPinning (line 1790) / ConfigureNUMA (line 1926)",
        "evidence": [
          "Real, tested primitives: ConfigureCPUPinning maps guest vCPUs to host threads via QMP query-cpus-fast and pins with sched_setaffinity (vcpus + emulator + iothreads); ConfigureNUMA emits -object memory-backend-ram + -numa node at launch, persisted in config.json (driver_kvm_numa_persist_test.go proves restart survival). IOThreads dataplane pinning exists (iothread0).",
          "Zero HTTP/API exposure: no handler in backend/cmd or backend/api calls either function — a tenant cannot request pinning or NUMA placement; there is no /vms/{id}/pinning route.",
          "No cgroup integration at all (no cpuset.cpus/cpu.max writes — grep 'cgroup' in the driver matches only comments); no host-topology-aware placement: placeFabricJob/placement cost model uses memory + link RTT only (cluster.go localNodeCapacity, fabric_jobs.go placeFabricJob); nothing consults hardware_virtualization.go's NUMA/host inventory when scheduling; no isolated-core reservation so pinned sets can overlap across VMs."
        ]
      },
      "minimalFix": "Expose POST /vms/{id}/cpu-pinning and put-if-match NUMA declarative spec at create time (Config.NUMA field already wired end-to-end in the driver); add a host-core allocator (reserved-cores bitmap per node, checked at pin time to reject overlap). Smallest tests exist in kernel already (driver_kvm_cpupinning_test.go / numa_persist); add one API-level sqlmock+token test asserting POST /vms/{id}/cpu-pinning reaches the driver and 404s->403s correctly — route is missing today."
    }
  ]
}
```

## Summary

The dominant hole is trust, not features: org scoping landed only on the VM *list* read and even there it silently degrades (empty claim → unfiltered list; unresolvable org → NULL-stamped VM invisible to its own tenant; org travels forgeably on the node-to-node create wire and directly drives billing attribution). Every write and read path a tenant can hit — get/delete/start/stop/migrate/transfers/interfaces/metrics/jobs — checks authentication but never ownership, so one authenticated tenant can delete or exfiltrate another tenant's VMs; locking that down with one shared ownership predicate is the single highest-leverage change. Close behind is the fabric job plane: submit has no admission control, placement races a capacity snapshot while granting no reservation, and the peer dispatch RPC has no retry, so saturation means oversubscription and blips mean failed jobs. Crash handling detects qemu exits but neither records why in the DB nor restarts the guest. GPU passthrough, hot volume resize/attach, NUMA/pinning, and node drain all have real, tested driver-level primitives (or a working engine in the case of evacuation) that simply have no HTTP surface in the api-server — wiring routes is most of the work — while shared-disk qcow2-over-NFS is the one area where the honest answer is a documented operator-provided shared mount, not new code.
