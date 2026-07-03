# NovaCron — Honest Status

NovaCron is a real single-node KVM VM manager with a genuine distributed-systems
foundation, being consolidated toward a production-ready release. This file
reflects the ACTUAL state of the code (verified by build + reading) and
supersedes the older completion/production-ready reports now in `docs/archive/`,
which overstate completion and should not be trusted.

## Works (verified real)

- **Single-node VM lifecycle via KVM** — QEMU-process driver: create / start /
  stop / delete / snapshot.
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

- **Live VM migration** — simulated today; real QEMU QMP + dirty-bitmap planned
  (Phase 2).
- **Federation cross-region data plane** — simulated (Phase 3).
- **Multicloud abstraction** — partially stubbed; real AWS SDK integration exists
  but is not wired behind the common interface (Phase 3).
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
