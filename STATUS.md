# NovaCron — Honest Status

NovaCron is a real single-node KVM VM manager with a genuine distributed-systems
foundation, being consolidated toward a production-ready release. This file
reflects the ACTUAL state of the code (verified by build + reading) and
supersedes the older completion/production-ready reports now in `docs/archive/`,
which overstate completion and should not be trusted.

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
  The subsystem holds 3–4 **redundant, mostly-hollow** cloud designs;
  `abstraction/aws_provider.go` has ~21 real `aws-sdk-go-v2` methods but 22
  `not implemented` stubs, and `GetQuotas`/`GetUsage` return fabricated/empty
  data (latent lies, flagged not yet fixed). Needs a **consolidation decision**
  (which design survives) before any implementation; the honest verification
  ceiling on this box is low — no cloud credentials, and LocalStack covers only
  EC2/VPC/S3, not cost/monitoring/quotas.
- **Advanced VM ops** (hot-plug, CPU pinning, NUMA) — not implemented.

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

`TestContainerdIntegration`'s two un-simulatable subtests (`ContainerNetworking`,
`InvalidImage`) are **honestly `t.Skip`ped** for the containerd driver, because
`driver_containerd_stub.go` is a pure in-memory simulation (every real containerd
call commented out) — a stub can't attach a real network or fail on a bad image
pull. They PASS for the real Docker driver. Making the stub echo fake data to
pass would be the exact "simulated coat" this effort removes, so it was
deliberately not done; the honest remaining work is a real containerd driver.

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

## Canonical

- **Server** — `backend/cmd/api-server`.
- **Deploy** — `docker-compose.yml` + `docker/api.Dockerfile`.
- **Target** — multi-arch (arm64/Jetson + x86_64).
