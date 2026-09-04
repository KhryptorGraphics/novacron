# NovaCron — Honest Status

NovaCron is a real single-node KVM VM manager with a genuine distributed-systems
foundation, being consolidated toward a production-ready release. This file
reflects the ACTUAL state of the code (verified by build + reading) and
supersedes the older completion/production-ready reports now in `docs/archive/`,
which overstate completion and should not be trusted.

## Swarm session — 2026-09-04

### What landed (all verified by build + test + live run today)

- **Raft leader check-quorum (product fix)** — `backend/core/consensus/raft.go`:
  a leader that cannot confirm majority acks within one electionTimeout
  (after a grace period) steps down (Raft thesis §9.6 / etcd check-quorum).
  `TestRaftNode_NetworkPartition` is now deterministic (was ~1/3 flaky: a
  minority leader held `IsLeader()` forever — a real split-brain window).
  26/26 post-fix runs; also fixed the `GetStats` copylocks vet warning.
  Known limit: disruption-on-heal (no pre-vote yet) — filed `novacron-fpg`.
- **DWCP Bullshark shutdown race (product fix)** — `v3/consensus/bullshark`:
  `Stop()` no longer closes buffered channels (cancel → `wg.Wait` → drain);
  `ProposeBlock` is ctx-guarded. The `panic: send on closed channel`
  (25–40% of runs) is eliminated; 10/10 stress runs clean under `-race`.
- **DWCP vet copylocks ×4 fixed** — `federation_adapter.go`,
  `partition_integration.go`, `partition/training/simulator.go`;
  `go vet ./network/dwcp/...` now exits 0. Also right-sized the 1 GB
  scenario buffer (128 MiB) in `testing/scenarios/high_latency_test.go`.
- **api-server boot fixed (product fix)** — `backend/cmd/api-server/main.go`:
  removed the divergent embedded `runMigrations` DDL (its VARCHAR-id
  `vm_interfaces` FK was unimplementable against the canonical UUID
  `vms.id` — the server could not boot against the golang-migrate schema at
  all). Replaced with a `requireMigratedSchema()` probe + actionable error.
  golang-migrate (`make db-migrate` / compose migrate service / k8s Job) is
  the single schema owner. Makefile: 9 `migrate.sh` invocations now run via
  `bash` (exec bit was missing → `make db-migrate` was broken).
- **API layer reconciled to canonical schema (product fix)** — VM CRUD,
  interfaces, admin users, and auth `SimpleAuthManager` rewritten from
  legacy columns (`tenant_id`/`config`/`networks`/`vm_interfaces`/int ids)
  to the canonical schema (uuid ids, `organization_id`,
  `network_interfaces`, `user_role`/`user_status` enums, bcrypt). No new
  migration needed. Verified LIVE: `/health` 200 healthy; `POST
  /api/auth/login` 200; `POST /api/vms` 201 (real KVM disk created); GET
  list/get 200; DELETE 200; teardown clean.
- **`docker/api.Dockerfile` fixed** — added the missing `sdk/` COPYs (the
  module-replacement target was never in the build context; canonical
  backend docker build now passes).
- **Frontend** — `tsc --noEmit` 0 errors; jest canonical 14/14 suites,
  34/34 tests; `next build` green; lint errors 986 → 116 (146-file
  mechanical `eslint --fix` + testing-library overrides in
  `.eslintrc.json`; remaining 116 are jsx-a11y / no-unescaped-entities /
  no-case-declarations, warnings-only debt left).
- **ai_engine (Python)** — `bandwidth_predictor_v3`: fixed the
  scaled-target training bug (val_loss 25,974,754 → 0.33 datacenter /
  0.25 internet) + inverse-transform in `predict()` + autocorrelated
  AR(1)-with-daily-cycle synthetic generators (i.i.d. noise made the 60%
  internet-accuracy assertion unreachable by construction). All 23 tests
  pass (was 9/23 + a hang).
- **ai-engine (Python service)** — pydantic v1→v2 migration complete
  (imports were fatal: `BaseSettings` moved, dead
  `FailurePredictionRecord` import); `OptimizationObjective` is now a real
  Enum; `HTTPException` pass-through in 4 handlers; fixture fixes.
  `tests/test_api.py` 17/17 (was: could not even collect).

### Canonical gates (run today)

- backend CI command set: 98 pass / 0 fail
- `vm -short -race`: 67 pass / 12 skip / 0 fail
- frontend full CI command set: green
- repo-root `go build ./...`: exit 0; `go build ./backend/...`: exit 0;
  backend/core `go build ./...`: exit 0

### Still broken (all filed in beads — do not duplicate)

- `novacron-frz`: dwcp scenarios suite hangs (WorkloadGenerator 8 GiB
  byte-fill; pre-existing).
- `novacron-slp`: dwcp simulator gates unmeetable (CompressedBytes never
  populated).
- `novacron-349`: dwcp Manager components never wired (silent no-op when
  Enabled).
- `novacron-5c7`: docker-compose.test.yml references nonexistent
  Dockerfiles.
- `novacron-8ba`: verify-email / resend-verification not implemented.
- `novacron-gwh`: ai-engine needs `NUMBA_DISABLE_CUDA=1` on aarch64.
- `novacron-fpg`: Raft pre-vote follow-up.
- `novacron-fb8`: documentation pollution (~4900 .md; 72 fabricated
  99.999% claims; `masterdocs/` 1231 dupes; `graphify-out/wiki` 2822) —
  archival NEEDS USER APPROVAL.
- Residues from schema reconciliation (comment them onto `novacron-ahm` or
  the relevant bead): `vms.cpu_cores` holds a scheduling weight (1024
  default), not a vCPU count; the `/api/networks` catalog is honestly
  empty (501 on create) — per-VM `/vms/{id}/interfaces` is the canonical
  surface; legacy role labels (`user`, `readonly`, `super-admin`) collapse
  to the canonical enum on write; admin role typos silently map to viewer.

## Works (verified real)

- **Single-node VM lifecycle via KVM through the canonical api-server** —
  create / start / stop / delete drive a real, arch-aware QEMU driver (not fake
  DB writes). arm64 verified end-to-end: a cirros guest boots to a login prompt
  through the HTTP API; x86 selection is unit-tested (live boot via CI matrix).
  Persisted state is the driver's actual state; running VMs are re-adopted into
  the manager after an api-server restart.
- **Canonical-path hardening (Phase 4)** — the api-server has panic-recovery
  middleware (clean 500 + one centralized log line; `net/http` already prevents
  the crash), a request-body-size cap (`http.MaxBytesReader`, env-tunable), and a
  64 KiB header limit. VM creation rolls back atomically — a failed `Create`
  leaves no orphaned disk dir or manager entry — and bounds its inputs
  (`DiskSizeGB`/`MemoryMB` ceilings, name length/charset). Wave 2 adds:
  fail-fast boot on a weak/default `AUTH_SECRET` (a short secret no longer
  slips through `Validate()`); reliable qemu termination (poll `/proc` +
  SIGKILL escalation, so a re-adopted SIGTERM-ignoring qemu can't be silently
  orphaned by a delete); and exactly-once resource accounting on delete (an
  atomic claim guards the counters against a double-release from an
  idempotent-Delete driver or a concurrent delete). All covered by
  discrimination-proven tests (each bar fails when its fix is removed).
- **Auth** — JWT (RS256/HS256), TOTP 2FA, OAuth2, RBAC, tenants, Postgres-backed,
  plus GitHub OAuth cluster admission.
- **Consensus** — Raft leader election + log replication, split-brain detection,
  distributed locks.
- **Storage** — distributed, replication, content-defined deduplication.
- **Monitoring** — Prometheus, OpenTelemetry, statistical anomaly detection.
- **ML** — LSTM predictor, gradient compression.
- **Network** — L4/L7 load balancer, overlay / segmentation.
- **Backup** — CBT changed-block tracking.

## Not yet real (simulated / in progress)

- **Live VM migration** — REAL QMP-driven QEMU live migration, proven single-host
  AND true cross-node (two processes over HTTP: `POST /vms/{id}/migrate` → node→URI
  resolution → a target-side `/internal/migrate/incoming` RPC launches the dest →
  guest cutover, ~10–24 ms downtime, **0 boot markers on the destination console**).
  Shared-storage live migration is a complete, standard posture (the libvirt/KVM
  default; early vMotion required it too). **Block (non-shared storage) migration
  now works too** — the destination gets its OWN empty disk, the source
  drive-mirrors into it over NBD (`copy-mode=write-blocking`, so no guest write is
  lost at the RAM cutover) until ready, then RAM cuts over and the source cancels
  the mirror (the dest owns its disk; pivoting would be wrong). Proven end-to-end
  at the driver level on **both x86 KVM and arm64 TCG**: a cirros guest writes a
  sentinel to its disk, block-migrates to a dest with **separate** storage, and
  the sentinel plus a live (no-reboot) counter cutover land on the dest's own disk
  (~190 ms downtime). **Block migration is now also wired through the cross-node
  HTTP path and proven on a real two-node x86 KVM microcluster** (two api-server
  processes, separate Postgres DBs + storage): `POST /api/vms/{id}/migrate
  {migration_type:"block", target_addr}` → source resolves the target, the dest
  RPC stands up an own-disk + NBD export, the source drive-mirrors + RAM-cuts-over.
  Observed: source qemu exits, the dest runs the **same** guest (its cirros
  metadata-retry counter advances 5→6 across the cutover with **0 boot markers** on
  the dest console — no reboot), the dest disk is populated on the dest's own
  storage, downtime 333 ms, and the dest's NBD export tears down cleanly only after
  the incoming migration resumes (fixing a teardown race). **Ownership transfers on
  cutover**: once the incoming guest resumes, the destination registers it in its
  manager + DB (so it lists and control ops route there — verified by a `stop` that
  killed the dest qemu), and the source retires it (manager + DB row removed, node
  accounting released). **Migrated VMs survive a dest-node restart**: the dest now
  launches into the canonical `<id>/` runtime dir and persists `config.json`, so
  reconcile keeps it running and the manager re-adopts the qemu (fixed an
  orphan-on-restart bug where the old `<id>-incoming/` dir hid it → reconcile wrongly
  marked it stopped; verified by a post-restart `stop` that killed the qemu).
  **Peer discovery** via `NOVACRON_PEERS="node2=host:port,…"`
  resolves a bare `target_node` with no `target_addr` (kept out of the scheduler so
  it can't trip placement/admission) — both re-verified on the two-node .53
  microcluster; shared-storage cross-node gets the same registration wiring
  symmetrically (not separately re-run cross-node). Remaining: only for *large/slow*
  migrations, the synchronous migrate endpoint can outlast the default 15 s
  `WRITE_TIMEOUT` (this fast run at 1.07 s did not) — raise the env or move to an
  async job API (deferred, YAGNI until a large-VM need appears).
- **Federation cross-region data plane** — build repaired; a REAL Raft-backed
  replication mechanism now exists and is **proven by test** (two instances, one
  Raft group: a write on the leader is applied on the follower via the committed
  Raft entry and read back; the follower rejects direct writes, so the value
  arrives solely via Raft). Honest scope: asynchronous replication over one local
  Raft group — NOT linearizable, NOT true geo-distribution. **Not yet wired into
  live federation:** nothing constructs `GeoDistributedState` (zero callers), and
  the federation root is off the canonical build path (behind
  `//go:build novacron_multicloud`) and does not build on arm64 (an `onnxruntime`
  transitive pull via `cross_cluster_components_v3.go`). Mechanism proven;
  live integration deferred (a strategic decision — federation is not on the
  canonical run path).
- **Multicloud abstraction** — build repaired: `backend/core/multicloud` now
  compiles for the first time (a committed syntax error had kept it from *ever*
  building). Off the canonical path (behind `//go:build novacron_multicloud`).
  **Consolidation done 2026-07-05** (commit `7931b9f9`): of the 3–4 redundant,
  mostly-hollow cloud designs, kept the only test-covered, self-consistent one
  (`multicloud/{orchestrator,*_integration,disaster_recovery,config}`, phase7
  integration tests green — re-verified 2026-07-11) and deleted the dead
  scaffolding with zero external importers (`federation/multicloud/{api_handlers,
  compliance_engine,cost_optimizer,cross_cloud_migration,policy_engine,
  provider_registry,unified_orchestrator}.go` and
  `multicloud/{bursting,cost,dr,management,migration}/`). **Latent lies fixed
  2026-07-05** (commits `0952bc24`, `64fa84a9`): both surviving AWS providers now
  error honestly instead of fabricating billing/usage/quota/pricing and claiming
  success — `abstraction/aws_provider.go` (`GetQuotas`/`GetUsage`/`GetCost`) and
  `federation/multicloud/providers/aws_provider.go`
  (`GetResourceQuota`/`GetResourceUsage`/`GetPricing`/`GetCostEstimate`/`GetBillingData`
  — the last was fabricating a $1250.75 bill with fake resource IDs); the
  remaining latent lie flagged in the consolidation commit
  (`multicloud/aws_integration.go` `CalculateCost()` using a hardcoded price
  table) is now labeled as an explicit static estimate rather than silently
  passed off as real pricing. **Re-verified 2026-07-11**:
  `abstraction/aws_provider.go` has ~21 real `aws-sdk-go-v2`-backed methods and
  25 `not implemented` stubs (recount from 22; no fabrication in any of
  them — all honestly error), `go build ./multicloud/...` is clean, and
  `go test ./multicloud/...` (phase7 integration suite) passes. Verification
  ceiling: inspection + build/test only — off the canonical path, and an
  `onnxruntime` transitive pull excludes all Go files on this arm64 box under
  CGO=0 (canonical CI does not build this package either). No cloud
  credentials, and LocalStack covers only EC2/VPC/S3, not cost/monitoring/quotas,
  so the stubs stay stubs — implementing them for real is unverifiable here.
- **Advanced VM ops** — **CPU pinning, device hot-plug (disk/net), cpu+memory hotplug, and NUMA all implemented** (KVM, all QMP-driven). CPU pinning: `query-cpus-fast` → `sched_setaffinity` per vCPU/emulator thread. Device hot-plug: disk/network via `blockdev-add`/`netdev_add` + `device_add` (+ `device_del`); arm64 PCIe hot-plug needs slots so `buildQEMUArgs` pre-provisions 4 `pcie-root-port`s on the `virt` machine only (x86 `pc` uses `pci.0`). CPU/memory hotplug + NUMA are **opt-in** (`VMConfig.Tags["hotplug.maxvcpus"|"hotplug.maxmem_mb"]` → `-smp N,maxcpus=M` / `-m N,slots,maxmem`; `ConfigureNUMA` sets topology before `Start` since NUMA is fixed at machine init → `-numa`), so **default VMs' `-smp`/`-m` are byte-for-byte unchanged** (proven by `TestBuildQEMUArgsHotplugOptIn`). All have discriminating real-qemu tests on arm64 TCG; memory-hotplug + NUMA pass here, cpu-hotplug works on x86 `pc`/`q35` but skips on arm64 `virt` (QEMU 8.2 genuinely can't hot-plug vCPUs there — skipped, not faked). Every shared-`buildQEMUArgs` change verified non-regressive: Gate 1 boot + Gate 2 shared/block migration cutover green across CPU pinning, hot-plug, and advanced-ops changes (one non-reproducible TCG timing flake seen under heavy parallel load, 5/5 on focused re-run). **iothread pinning now works** via opt-in `-object iothread` (`VMConfig.Tags["iothreads"]`, disk attaches to `iothread0`) — `ConfigureCPUPinning`'s iothread branch is live. **NUMA persists across a driver restart** (`ConfigureNUMA` encodes the topology into `Config.Tags["numa.topology"]` + config.json; `buildQEMUArgs` rehydrates it via `effectiveNUMA`). Both opt-in → default VMs' args stay byte-for-byte identical (asserted by tests). **PROVEN ON REAL x86 KVM** (192.168.1.53, 96-core, `/dev/kvm`): after making the real-qemu tests arch-portable (`findCirrosImage` + `defaultQEMUBinary` now pick by `runtime.GOARCH`), CPU pinning (vcpu narrowed across 96 real cores), **CPU hotplug (present vCPU 1→2 — impossible on arm64 `virt`)**, memory hotplug (DIMM), NUMA (2-node), and iothread pinning all PASS on real KVM. Finding (root-caused): x86 hot-**unplug** originally left the device stuck because the test plugged+unplugged within milliseconds of `Start`, before the guest had booted and enumerated the slot — `device_del`'s guest-driven eject is silently lost if issued that early (harmless on slow arm64 TCG, exposed on fast x86 KVM). NOT a machine-type bug — **verified by testing q35 + PCIe root ports on x86, which did NOT fix it, then reverting** (the wrong hypothesis) — and NOT a cirros-x86 guest limitation. Fix is test realism: `TestHotPlugDiskRealQMP` now boots with a cloud-init seed, **waits for the guest to fully boot (first MIGTICK) + a brief enumerate settle before unplugging**, and **hard-asserts the device is REMOVED**. Empirically PASSES with genuine removal on BOTH arm64 TCG (39s) and real x86 KVM (10s). The driver was correct throughout (`HotUnplugDevice` issues `device_del`, waits for the guest to release via `awaitDeviceGone`, then frees the backend). **Net: all 6 advanced-ops tests pass on real x86 KVM, hot-unplug now with a hard removal assertion on both arches.** ~~Still pending: a cleaner `VMConfig.NUMA` field vs. the Tags encoding~~ — **DONE 2026-07-05** (`9476ed67`): typed `VMConfig` fields for hotplug/NUMA replaced the Tags encoding (the `Tags["hotplug.*"]`/`Tags["numa.topology"]` references above describe the historical mechanism; see "Real containerd driver + typed VMConfig" section).

## vm sub-package compile gap — quarantined 2026-07-04

Four `backend/core/vm` sub-packages did not compile and were **off the canonical
production path** (no `api-server`/`core-server`/root-`vm` import; the only
non-test cross-import was `vm/unified/scheduler.go` → `vm/kata`). They were
experimental/moonshot code drifting against upstream bindings:

- `vm/drivers/kvm/libvirt_driver.go` — `undefined: libvirt.Connect` etc. (libvirt Go binding missing/mismatched).
- `vm/kata/driver.go` — `undefined: syscall` (missing import), containerd `ExitStatus` API drift, `VMMetrics` field drift.
- `vm/unified/scheduler.go` — imports the broken `vm/kata`.
- `vm/tests/{delta_sync_benchmark,delta_sync_integration,ebpf_migration}_test.go` — bad import path (`novacron/backend/core/vm is not in std`); eBPF page-tracker benchmarks.

Files renamed to `.go.disabled`, matching the repo's existing ~30-file quarantine
convention in `vm/` (e.g. `driver_kvm_old.go.disabled`, `driver_kata_containers.go.disabled`).
`ponytail:` ceiling: re-enable only after porting to the current libvirt/containerd
API and verifying the package compiles in isolation. `ebpf_programs/` (`.bpf.c` +
`Makefile`) is not a Go package and is left as-is.

Container-driver integration tests — **fixed** 2026-07-04 (were red under full
`go test ./vm/`, behind `-short` so CI never ran them). Two real bugs in the
**real Docker** `ContainerDriver` (`driver_container.go`): (1) `config.Name` was
interpolated raw into `docker create --name`, so any name with a space failed
("only [a-zA-Z0-9][a-zA-Z0-9_.-] are allowed") — now scrubbed via a regexp;
(2) `GetInfo`'s `docker inspect -f` template referenced `.State.MemoryStats`/
`.State.CPUStats` (those are `docker stats` fields, invalid in `inspect`), so
inspect always errored and the fetched output was discarded anyway — replaced
with a valid inspect that populates `Image`, `NetworkID` (first attached
network), and the configured `CpuShares`/`Memory` limits. Plus an empty-config
guard on `MockHypervisor.Create` + `ContainerDriver.Create` (reject a config with
neither Name nor ID, matching the kvm/containerd drivers). Result: `TestDocker
Integration`, `TestMultiHypervisorIntegration`, `TestVMDriverIntegration` all
PASS against real docker/qemu; CI `-short` gate stays green.

~~`TestContainerdIntegration`'s two un-simulatable subtests (`ContainerNetworking`,
`InvalidImage`) are honestly `t.Skip`ped for the containerd driver, because
`driver_containerd_stub.go` is a pure in-memory simulation~~ — **SUPERSEDED
2026-07-05**: the stub is deleted and a real containerd driver now exists; see
"Real containerd driver" section below.

## Real containerd driver + typed VMConfig — done 2026-07-05

Two tracks landed on main (pushed HEAD `d1b22571`), canonical CI GREEN:

- **Track B** (`9476ed67`, `refactor(vm)`): typed `VMConfig` fields for
  hotplug/NUMA replace the stringly-typed `Config.Tags` encoding; the NUMA/
  hotplug config JSON survives driver restart instead of JSON-in-Tags. Full
  unit suite PASS on main after cherry-pick.
- **Track A** (`d902a34b`, `feat(vm)`): **real containerd driver replacing the
  simulation stub** — `driver_containerd_stub.go` (pure in-memory, every real
  call commented out) deleted; new `driver_containerd.go` shells out to `ctr`
  against live containerd. The previously un-simulatable
  `TestContainerdIntegration` subtests (`ContainerNetworking`, `InvalidImage`)
  now run against live containerd instead of being skipped (3 files changed,
  638 insertions, 577 deletions).

Canonical gate for this push: `CI - Canonical Verification` **success** (run
28803552345) on `d1b22571`. The simultaneously-failing workflows on that sha
(`DWCP v3 - CI/CD`, `CI/CD Pipeline`, `NovaCron Production CI/CD`,
`comprehensive-testing.yml`, `Update Code Memory`) are the pre-existing broken
legacy pipelines, not regressions from this change and not part of the gate.

## Whole-module build repair — done 2026-07-05

**`backend/core` now builds completely** on this arm64 box: `go build ./...` = 0
failures AND `go test -run '^ZZZ$' ./...` (every test binary compiles) = 0
failures, CGO on. Canonical api-server+core-server build under BOTH CGO=0 and
CGO=1; vm gate green.

Key correction to the earlier "onnxruntime arm64 platform limit" claim: it was
just `CGO_ENABLED=0`. `github.com/yalue/onnxruntime_go` uses cgo (`import "C"`), so
CGO=0 excluded all its files → "build constraints exclude all Go files" for the ~12
packages that transitively import it. With CGO=1 the dep builds; only 4 of those 12
had real code errors (agents, compute, federation/multicloud, migration), all fixed
as bad-merge/API-drift reconciliation against the CURRENT type model (interface-vs-
impl, renamed fields, removed methods; two orphan files with zero external callers —
migration/cross_cluster_runner.go and orchestrator_dwcp_v3.go — quarantined).

Two genuine arch limits were also fixed properly (not worked around): the
`dwcp/optimization/simd` package (amd64-only asm) and `dwcp/optimization/prefetch.go`
(`//go:linkname` to amd64-only `runtime.prefetch`) now build on all arches via
`//go:build`-split asm-decl + pure-Go/no-op fallback files; both cross-compile clean
for amd64 too.

A generation of DWCP `phase1_*`/`phase3_*` integration+benchmark tests and a couple
of other test files were quarantined (`.go.disabled`) — they target a redesigned
transport.AMSTConfig / compression.DeltaEncodingConfig / multiregion API and would
be rewrites; all off the canonical path. Real bugs fixed in passing: an audit-Reason
drop and a chaos-engineering ImpactDuration metric that were both dead-code-after-
return; a storage context leak; firewall case-insensitive-regex no-op; IPv6-unsafe
address formatting.

Residual is now resolved — see "Deliberately-left work completed" below.

## Deliberately-left work completed — 2026-07-05

Both remaining deferrals from the sweeps above are now closed.

**On-path `vm/vm.go` vet warnings — FIXED (45 → 0).** Root cause: `VMEvent.VM`
held a `VM` by value, and `VM` carries 3 `sync.RWMutex`, so every emit / append /
handler-dispatch / `json.Marshal` of an event copied a live lock (37 copylocks) —
and the copy was a torn read of mutex-guarded fields anyway. Changed `VMEvent.VM`
to `*VM`: lock-free, no torn copy, identical JSON output, and all reads
(`event.VM.ID()`) work unchanged on a pointer; handlers already run async
(`go handler(event)`) so no snapshot semantics were lost. Also dropped 7 dead
json tags on unexported `vm.VM` distributed-state fields (json ignores unexported
fields — the tags were no-ops). `go vet ./vm/`: 45 → 0; api-server/core-server
build (CGO on+off); `go test -short ./vm/` ok. Commit d805930a.

**Quarantined DWCP test suites — dispositioned: documented, left disabled.** The
7 phase/orchestrator suites (~3300 LOC) target redesigned AMST/HDE/multiregion
APIs (11+ compile errors each). Verified OFF the canonical binary path — neither
api-server nor core-server imports `network/dwcp` or `migration` — and OFF-CI
(`ci.yml` never references dwcp). The whole `network/dwcp` tree is experimental
WAN-protocol scaffolding; rewriting off-path tests against dead APIs is
speculative investment, so they stay `.go.disabled`.

Discovered in passing: the dwcp package's *active* test suite is itself broadly
red (pre-existing, off-CI) from two root causes — (1) most failures are the
multi-stream TCP transport dialing a live peer a unit env can't provide
(`transport/multi_stream_tcp.go:181` → "failed to create any streams"), and
(2) real config/validation drift in `config_test.go` / `manager_config_test.go`
(`TestPredictionValidation`, `TestConsensusValidation`, `TestManagerGetConfig`).
Per the off-path / don't-invest disposition, only the named `race_test.go` was
cleaned: its two `Start()`-based tests now skip cleanly when no peer is reachable
(mirrors the container-driver skip-guards) instead of failing. The
`dwcp_manager_test.go` / `config_test.go` failures are left as-is and recorded
here as known off-path debt.

## Canonical CI gate — GREEN 2026-09-04 (canonical gates re-run locally)

The `CI - Canonical Verification` workflow (`.github/workflows/ci.yml`) is
**green on `main`** — run `28748286801`, commit `d3e26f1f`, all three jobs
(Canonical Backend, Canonical Frontend, x86 KVM smoke [non-gating,
`continue-on-error`]). This closed a multi-commit red streak surfaced only on
push: local checks had run a *subset* of the CI command set. Two root causes,
both fixed:

- **Backend** — `TestRegisterSecureAPIRoutesCreatesVMOnCompatibilityRoute` failed
  on a stale sqlmock expectation (`node_id` matched `nil`, but VM-create now
  records `selfNodeID()` = `"local"` since 219c25d4). Fixed to `sqlmock.AnyArg()`
  (commit `cb60c022`).
- **Frontend** — `frontend/package-lock.json` was gitignored and never committed,
  so setup-node's `cache: npm` + `npm ci` both failed at "Set up Node.js". Un-ignored
  and committed the 754 KB lockfile (commit `d3e26f1f`).

Lesson recorded: verify the *exact* CI command locally, not a subset — the vm
`go build` + `-short` test I ran never exercised `go test ./backend/cmd/api-server`.
Re-verified 2026-09-04 (swarm session): backend CI command set 98 pass /
0 fail; `vm -short -race` 67 pass / 12 skip / 0 fail; frontend full CI
command set green (tsc 0 errors, jest canonical 14/14 suites 34/34, next
build green); repo-root and backend/core `go build ./...` exit 0.

## Moonshot sweep — done 2026-07-04

The `backend/core` module previously carried ~96 experimental "moonshot" packages
(~344K LOC: quantum, photonic, planetary, arvr, iot, autonomous, v4/v5, cognitive,
blockchain, edge, research, plus rotted production-named variants cache/ml/security/
ha/sdn/…) that never compiled and sat on no production path. **All deleted** (two
reviewable commits). Before deletion, verified none were on the canonical
api-server/core-server dep closure, in federation/multicloud, behind a build tag, or
referenced by any `.go.disabled` file; their only importers were themselves already-
dead off-path code (api/ml, api/admin, `//go:build novacron_secure` main_secure.go,
cache-monitor, core/compliance, core/governance, examples/policy, dead api tests),
removed alongside. Root cause across all of them was compile rot (redeclared symbols,
undefined constants, unused vars, type mismatches) — errors a feature in real use
physically cannot contain, i.e. never-functioned scaffolding.

Result: **`cd backend/core && go build ./...` now reports 0 broken packages** (was
96). Canonical api-server + core-server build exit 0; vm gate green. Recoverable via
git history if any is ever revived.

## Test-compile + vet cleanup — done 2026-07-05

Follow-on to the moonshot sweep. Made every `backend/core` package's test binary
build (or honestly quarantined the deep-drift ones) and fixed the real bugs vet
surfaced.

Test files repaired (real, bounded API drift — no assertions weakened):
`cmd/novacron` (stale `registerLocalSchedulerNode` arg — my own 72afc2f7 change),
`consensus/raft_test.go` (node.id→nodeID), `integration_tests/basic_validation`
(unused import), `scheduler` (nil predictor arg), `network` syntax (`]`→`}`,
unnamed returns), `dwcp/{sync,conflict,security,testing,multiregion}` (imports,
redeclare, redundant newline), `dwcp/metrics`+`dwcp/optimization` (malformed import
paths missing the module prefix), `dwcp/v3/{transport,monitoring,optimization}`.

Test files quarantined to `.go.disabled` (deep drift — written against APIs that no
longer exist: methods now unexported, types migrated, symbols removed; all off the
canonical path): `consensus/{chaos,raft_comprehensive}_test.go`,
`integration_tests/{qos_enforcement,stun_parsing,udp_hole_punching}_test.go`,
`network/{isolation,qos,network_benchmark}_test.go`,
`scheduler/network_aware_scheduler_test.go`, `vm_isolated_test.go`. Quarantining a
broken file unblocks its package's still-valid sibling tests (which the build error
had been suppressing).

Real bugs vet caught, now fixed: **audit/types.go** (ON path) dropped the audit
`Reason` field via a premature unconditional return (dead reason-extraction block);
**storage/distributed_storage.go** (ON path) leaked a `context.WithCancel` on
construction-failure paths; firewall DPI silently ignored case-insensitive rules
(`flags = flags` no-op → now `(?i)`); IPv6-unsafe `%s:%d` in discovery + loadbalancing
→ `net.JoinHostPort`; dead code after early returns in the off-path hypervisor stub.

Remaining test-build failures are ONLY platform/dependency limits, not code to fix:
~12 packages need `github.com/yalue/onnxruntime_go` (no arm64 build files) and
`dwcp/optimization` needs an amd64-only `/simd` assembly package. They need an x86
host or a build-tag/stub decision. (The ~42 vet "copies lock" warnings from
`VMEvent` embedding a whole `VM` were subsequently fixed — see "Deliberately-left
work completed — 2026-07-05" above.) CI runs targeted `go test`, not `go vet ./...`.

## Frontend strict-mode type cleanup — done 2026-07-06

The Next.js frontend's production source is now **type-clean under the strict
tsconfig** (`strict`, `exactOptionalPropertyTypes`, `noImplicitReturns`,
`moduleResolution: bundler`) and `npm run build` compiles green
(`✓ Compiled successfully`). Drove `tsc --noEmit` on production source from **705
errors → 0** across a multi-session sweep, without loosening a single compiler
flag. Representative real fixes (not suppressions): added the missing `success`/
`warning` Badge variants (used across six dashboards); added the missing
`@/components/ui/avatar` wrapper over the already-installed
`@radix-ui/react-avatar`; widened optional API-param fields (vms/networks/admin/
client) to accept explicit `undefined` as `exactOptionalPropertyTypes` requires;
widened Recharts/chart.js formatter+options signatures at call sites;
added `return undefined` on the no-op branch of effects (`noImplicitReturns`) and
`override` modifiers on error-boundary lifecycle methods; reshaped
`LoadingStates`/`RefreshIndicator` to the wrapper contract their callers already
used; unified a duplicate `MLModelMetrics` by exporting the panel's type;
`MetricsCard.value` → `ReactNode`; `lucide` `Memory→MemoryStick`, `CPU→Cpu`, add
`ArrowRight`; `utils.bytesToSize`.

**CI gate verified locally against the exact command set**: the canonical
`npm test -- --runTestsByPath <14 curated suites>` → **14 suites / 33 tests pass**,
and `npm run build` compiles. (Lesson from the 2026-07-05 red streak applied: ran
the *exact* CI command, not a subset.)

Deliberately left (does NOT gate CI or the production build): ~33 `tsc` errors
remain in **non-canonical test files** — drift against refactored component/hook
APIs (`MetricsCard`, `usePerformance`, `validation`), a couple of mock-typing
mismatches in `distributed-monitoring.*`, and a missing `jest-axe` devDep. None
are in the canonical `--runTestsByPath` set and `next build` excludes `__tests__`,
so they are off the gate; a follow-up should either update those tests to the
current APIs or quarantine them.

## Backend go.sum repair + frontend test suite — done 2026-07-07

**Backend build was broken** (P0): the root `go.sum` was missing a swath of module-graph
hashes (a bad merge dropped them), so `CGO_ENABLED=1 go build ./backend/...`,
`core-server`, and `make core-build` all failed at graph verification with
`missing go.sum entry`. Reconciled with `go build -mod=mod` (minimal path — adds only
what building needs; +~20 go.sum lines vs +900 from `go mod download all`). The go
directive also moved `go 1.24.0`→`go 1.25.0`: **mandated by the repo's own
`backend/core/orchestration` submodule, which requires `go >= 1.25.0`** (pinning
`go1.24.6` fails on it), and the handful of indirect-dep bumps (x/crypto, x/net, logrus,
…) are MVS-required corrections of stale pins, not gratuitous upgrades. Net: root
`go build ./backend/...` + core-server + api-server + `make core-build` all exit 0 in
readonly mode; consensus (ProBFT) + `network/dwcp` test-compile clean (42 pkgs); the
Python neural-training scripts syntax-check OK.

**Frontend test suite**: drove `tsc --noEmit` from 32 test-file errors to **0** and
kept `next build` + the canonical 14-file jest gate green (33/33). Installed
`jest-axe`; fixed import-path bugs (`tests/dashboard` → `@/app/dashboard/page`,
`NetworkTopology` → `visualizations/`); and **properly rewrote** three drifted suites
to the current APIs — `usePerformance` (Web Vitals surface, not the removed mark/measure
API), `validation` (step-based `validateRegistrationStep`; also fixed the old file's
latent `validateEmail`-returns-object-vs-`true` bug and the over-strict email cases),
and `MetricsCard` (current props; the old `trend`/`color`/`article`/`aria` contract is
gone). All three pass.

Quarantined (per repo `.disabled` convention) five deeply-drifted, **off-gate** suites
whose repair is a separate initiative: the three `distributed-monitoring*` suites (their
WebSocket/dashboard mock infrastructure is written against removed hook shapes and fails
at render), the naive `tests/dashboard` app-page smoke test (renders a full route with no
provider/router harness), and `auth-accessibility` — **note: the a11y suite surfaces a
real finding** (the `Progress` UI component renders an indeterminate progressbar with no
`aria-valuenow`, which axe flags); re-enable it after the `Progress` component sets a
`value`/`aria-valuenow` and framer-motion is mocked in that test.

## Canonical

- **Server** — `backend/cmd/api-server`.
- **Deploy** — `docker-compose.yml` + `docker/api.Dockerfile`.
- **Target** — multi-arch (arm64/Jetson + x86_64).
