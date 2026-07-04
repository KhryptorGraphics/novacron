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
  (~190 ms downtime). Remaining: wire it into the cross-node HTTP path (the
  `/internal/migrate/incoming` RPC + `migrateVM`) for a true two-node cluster;
  multi-node peer-address discovery (arrives with federation, Phase 3).
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

## Known-broken / out of scope

The `backend/core` module also contains ~105 experimental "moonshot" packages
(quantum, photonic, planetary, arvr, iot, autonomous, v4/v5, cognitive, …) that do
not compile and sit on no production path. They are intentionally excluded from the
build gate — CI builds the canonical `backend/cmd/api-server`, which transitively
compiles the real `backend/core` packages. Isolate into a separate module or delete;
tracked as a follow-up, not a release blocker.

## Canonical

- **Server** — `backend/cmd/api-server`.
- **Deploy** — `docker-compose.yml` + `docker/api.Dockerfile`.
- **Target** — multi-arch (arm64/Jetson + x86_64).
