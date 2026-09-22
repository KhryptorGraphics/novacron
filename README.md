# NovaCron

NovaCron is a KVM hypervisor manager with a peer-to-peer fabric. One
`api-server` process owns VM lifecycle on a node and speaks to its peers:
signed cluster join (HMAC-SHA256 over `node_id|addr|ts`, with a reachability
callback before admission), a 30 s heartbeat that measures each link's
throughput and RTT, bandwidth-aware placement and job dispatch (a job is a
Process-driver VM placed on one node, chosen by pin > locality > measured-link
cost), live and block (non-shared-storage) VM migration with a compression
decision taken from measured link state, and persisted usage metering behind an
organization-scoped billing API. Most other subsystems in this tree are libraries
the canonical binary never imports; see [Limitations](#limitations).

## Quickstart — one node

Prerequisites: Go (root `go.mod` requires `go 1.25.0`), a reachable PostgreSQL,
and the arch-matched `qemu-system-*` binary on `PATH` for real VMs
(`defaultQEMUBinary()` in `backend/core/vm/`: `qemu-system-x86_64` on amd64,
`qemu-system-aarch64` on arm64).

```bash
# 1. Build the canonical server (entrypoint: backend/cmd/api-server)
CGO_ENABLED=1 go build -o bin/novacron-api ./backend/cmd/api-server

# 2. Create the schema
cd database && DB_URL="postgres://postgres:postgres@127.0.0.1:5432/novacron?sslmode=disable" \
  bash ./scripts/migrate.sh up && cd ..

# 3. Run
export DB_URL="postgres://postgres:postgres@127.0.0.1:5432/novacron?sslmode=disable"
export AUTH_SECRET="$(openssl rand -hex 32)"   # required; the default and secrets <16 chars are rejected
export STORAGE_PATH=/srv/novacron/vms          # VM runtime/disk root
export NOVACRON_MIGRATION_SECRET="$(openssl rand -hex 32)"   # node-to-node RPC + cluster join; unset = fail closed
export NOVACRON_NODE_ID=node-a
./bin/novacron-api   # HTTP on API_PORT (8090), WebSocket on WS_PORT (8091)
```

`DB_URL`, `AUTH_SECRET` and `STORAGE_PATH` are read by `backend/pkg/config`
(`config.go`). `NOVACRON_MIGRATION_SECRET`, `NOVACRON_NODE_ID`,
`NOVACRON_JOIN_PEERS` / `NOVACRON_PEERS` and the `NOVACRON_RATE_*` rate card are
read directly by `backend/cmd/api-server` (`cluster_join.go`,
`migration_auth.go`, `billing_usage.go`).

For the fabric rather than a lone node: `bash scripts/fabric/two-node-fabric-test.sh`
provisions two real api-servers (separate databases, storage and ports; node B
inside a network namespace behind a `tc`-shaped veth), joins them through the
signed join protocol, and asserts membership plus a measured link profile, a
job dispatched to the peer with its stdout fetched back, a second transfer to a
busy link queued with an ETA, and a cross-node migration. `FABRIC_REQUIRE_ALL=1`
turns SKIP into failure — that is how CI runs it (`.github/workflows/ci.yml`).

## Architecture

- **Fabric control plane** — `backend/cmd/api-server/`: `cluster_join.go`
  (signed join, peer-map poisoning rejected, link profiles persisted in
  `cluster_peers`), `fabric_jobs.go` (placement order: pin > locality >
  measured-link cost > default), `fabric_transfers.go` (one transfer per link,
  later ones queued with an ETA; compression is enabled only when the link is
  below 500 Mbps and the sampled payload ratio exceeds 1.3), `migration_auth.go`
  (internal RPC secret, fail-closed).
  Metering lives here too: `billing_usage.go` writes `usage_events` and serves
  `GET /api/billing/usage` and `/api/billing/usage/summary`.
- **VM lifecycle and migration** — `backend/core/vm/`: KVM driver over QMP,
  shared-storage and block live migration, `-incoming defer` for every
  destination, boot-time reconciliation of interrupted jobs and orphaned
  migration destinations.
- **Auth and access control** — `backend/core/auth/` (JWT, bcrypt, TOTP 2FA)
  and `backend/api/security/` (Postgres-backed RBAC).
- **Schema** — `database/migrations/` (through `000010_usage_events`), applied
  with `database/scripts/migrate.sh`.
- **User surfaces** — `cli/` (`novacron fabric nodes|jobs|transfers|usage`),
  `sdk/typescript/` (`FabricClient`), `frontend/` (the `/fabric` page).

## Development

Canonical gates (run from the repo root unless noted):

```bash
cd backend/core && go build ./... && go vet ./...
cd backend/core && go test -short -race ./vm/    # KVM driver + migration logic
go test -short ./backend/cmd/api-server/ ./backend/api/graphql/ ./backend/api/security/ \
  ./backend/api/websocket/ ./backend/pkg/config/   # canonical root test set
bash scripts/fabric/two-node-fabric-test.sh        # live two-node acceptance
```

The two-node harness needs root for namespace/veth work and real qemu; run it
when nothing else on the host is using it. `cd backend/core && go build ./...`
plus the root set is what `STATUS.md` calls the G0 gate. Last recorded runs
(STATUS.md, 2026-09-20/21): the harness passed 12/12 with 0 SKIP, including
usage metering from real cross-node transfers; on a `tc tbf 50mbit` link,
compression took one RAM-dominant migration from 57.9 s / 345 MiB to 26.7 s /
156 MiB — STATUS.md does not generalize that to small or incompressible guests.

## Limitations

Not generally available; read `STATUS.md` first — it is the authoritative dated
state record and supersedes `docs/archive/**` (fabricated figures; do not cite).

- **GPU passthrough: absent.** Every driver that compiles into the canonical
  binary returns `false` from `SupportsGPUPassthrough()`; no `vfio-pci` or
  `hostdev` code is on the canonical path (`research/profitability/technical-audit.md`).
- **Multi-tenancy: partially wired** (as of 2026-09). Users carry
  `organization_id` and the JWT `tenant_id` claim, and `GET /api/vms` filters by
  organization for non-admins — but `vms.organization_id` is still NULL on
  create and there is no per-tenant query isolation or RLS yet (tracked as
  `novacron-ok7` in STATUS.md).
- **Billing measures, it does not charge.** Rates come from
  `NOVACRON_RATE_PER_GB_EGRESS` / `_PER_VCPU_HOUR` / `_PER_JOB_SECOND` /
  `_PER_MIGRATION` and default to 0 (unpriced, not free); invoicing, dunning and
  a payment gateway are unbuilt, and `backend/enterprise/billing/` is unwired.
- **Unwired libraries.** Backup/DR (`backend/core/backup`, `backend/core/dr`),
  the general monitoring/alerting engine (`backend/core/monitoring`) and the
  multi-cloud/federation code are real code with tests, but no import path
  reaches them from `api-server`; the fabric's own link telemetry
  (`GET /api/cluster/links`) is the live observability.
- **Marketplace: absent.** `marketplace/` holds a single text listing, no code.
- Anything here older than 2026-09-20 is unverified by the current sessions
  (check STATUS.md).