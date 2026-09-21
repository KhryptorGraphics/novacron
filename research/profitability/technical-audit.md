# NovaCron Technical Differentiation Audit — What's Actually Real

**Audit date:** 2026-09-21
**Scope:** `/home/kp/thordrive/novacron`, canonical binary `backend/cmd/api-server`
**Method:** Static code reading, grep-based evidence gathering, and cross-reference against `STATUS.md` (the repo's own honest state record, last updated 2026-09-20 — one day before this audit).

**Constraint disclosure:** This audit was performed under a read-only mandate — no `go build`, `go test`, or other execution was run by the auditor. Task 1 and Task 3's numeric claims ("compiles", "N tests pass") are therefore **not independently re-verified here**; they are cited from `STATUS.md`'s own build/test log, which is dated one day before this audit, was itself produced by running the exact commands quoted, and is corroborated at the source level everywhere this audit could check it (e.g., the exact import lists, build tags, and file wiring described below match what `STATUS.md` claims). Everywhere a claim is NOT directly grounded in code read during this audit, it is marked `[STATUS.md]`.

---

## 1. What compiles and is wired into the canonical binary

**Canonical entrypoint:** `backend/cmd/api-server/main.go`, gated by:
```go
//go:build !novacron_enhanced && !novacron_improved && !novacron_multicloud && !novacron_production && !novacron_real_backend && !novacron_secure && !novacron_working && !novacron_simple_api
```
This means a plain `go build ./cmd/api-server` compiles `main.go` plus every **untagged** sibling file in that directory (`cluster.go`, `cluster_join.go`, `fabric_jobs.go`, `fabric_transfers.go`, `migration_auth.go`, etc. — confirmed by reading their headers, none carry a `//go:build` line). The six `main_enhanced.go` / `main_improved.go` / `main_multicloud.go` / `main_production.go` / `main_real_backend.go` / `main_working.go` files are **entirely separate, mutually-exclusive alternate binaries** selected only by an opt-in build tag — they never run in the canonical build and are effectively legacy/abandoned prototypes left in the same directory.

**Canonical `main.go`'s full internal import list** (i.e., everything the running server actually links against):
```
backend/api/graphql
backend/api/security
backend/api/websocket
backend/core/audit
backend/core/auth
backend/core/storage
backend/core/vm
backend/pkg/config
backend/pkg/logger
```
That's it. Grep confirms **zero** imports of `core/backup`, `core/dr`, `core/monitoring`, `core/quotas`, `core/consensus`, `core/network/dwcp`, `core/multicloud`, `core/federation`, `core/ml`, `enterprise/billing`, or any of `backend/business|corporate|competitive|partners|ecosystem|sustainability|sales`.

`[STATUS.md]`: `cd backend/core && go build ./... && go vet ./...` exits 0 (all ~50+ core sub-packages compile as standalone libraries — this is a build-cleanliness claim, not a wiring claim), `go build ./cmd/api-server` and `./cmd/core-server` succeed under both `CGO_ENABLED=0` and `=1`, and a dedicated two-node fabric acceptance harness (`scripts/fabric/two-node-fabric-test.sh`) passes 10/10 against the live canonical binary.

**Practical reading:** the repo is really two things overlaid — (a) a small, actively-maintained canonical server (`main.go` + `core/vm` + `core/auth` + `core/storage` + `core/audit` + three API-facade packages) that is genuinely built, tested, and run live, and (b) a very large body of standalone Go/Python packages under `backend/core/*`, `backend/enterprise/*`, and `backend/{business,corporate,competitive,partners,ecosystem,sustainability,sales}/*` that compile in isolation but have **no caller anywhere** in the canonical dependency graph.

---

## 2. Feature reality check

### Multi-tenancy (organizations, tenants, isolation) — **STUB**
- The Postgres schema does model tenancy for real: `database/migrations/000001_init_schema.up.sql:19` creates `organizations` (id, name, slug), and `:29-42` gives `users.organization_id UUID REFERENCES organizations(id)`.
- But the canonical auth manager wired into `main.go:81` (`auth.NewSimpleAuthManager`) never touches that column: `backend/core/auth/simple_auth_manager.go` `CreateUser`'s `INSERT INTO users (username, email, password_hash, role, status) ...` (lines ~72-79) omits `organization_id` entirely, and `scanUser`'s `SELECT id, username, email, password_hash, role, status, created_at, updated_at ...` (lines ~120-124, ~135-138) never selects it either. Every user created through the canonical `/api/auth/register` path has `organization_id = NULL` — permanently.
- The VM list/get endpoints in `main.go:822` and `:876` do `SELECT ... organization_id ... FROM vms` with **no `WHERE organization_id = ...` filter** — even if the column were populated, there is no query-level tenant isolation in the canonical binary.
- A real, more sophisticated multi-tenant engine does exist — `backend/core/auth/postgres_tenant_store.go`, `tenant.go`, and `auth_manager.go`'s tenant-scoped `HasPermissionInTenant`/`AuthorizeRequest` — with its own tests. It is simply never instantiated by `main.go`, which uses `SimpleAuthManager` exclusively.
- **Verdict: STUB.** Schema-level scaffolding is real; enforcement is absent from the running server.

### RBAC / auth (roles, permissions) — **REAL** (coarse role-gating + 2FA), **PARTIAL** (fine-grained permission engine unwired)
- JWT (HS256) issuance/verification, bcrypt password hashing, and route middleware are real and wired: `backend/core/auth/simple_auth_manager.go` (`Authenticate`, `generateJWTToken`), `main.go:341` (`requireAuth`).
- TOTP 2FA is real and wired: `auth.TwoFactorService` constructed at `main.go:1825`, routes registered at `main.go:1852-1864`.
- Role-based route gating is real and wired: `requireAnyRoleMiddleware("admin","super-admin")` guards `/api/admin/*` (`main.go:1867-1870`) and the security route set (`main.go:2232-2234`).
- Postgres-backed role/permission CRUD is real and wired: `backend/api/security/rbac_store.go` (`PostgresRBACStore`) reads/writes real `roles`/`permissions` tables with JSONB permission arrays, constructed at `main.go:1826` (`securityapi.NewPostgresRBACStore(db)`). Note its own comment (line ~275): a user carries exactly **one** role column — no multi-role composition.
- A separate, considerably richer resource:action permission engine exists in `backend/core/auth/auth_manager.go` + `role.go` + `postgres_role_store.go` (tenant-scoped roles, wildcard resource/action matching, permission caching) with real tests (`auth_test.go`, `security_test.go`). It is **not** what `main.go` uses — `main.go`'s gating is a flat string match against the `user_role` enum (`admin`/`operator`/`viewer`), not this engine.
- **Verdict:** JWT + bcrypt + 2FA + coarse role gating = **REAL** and running. The granular multi-tenant permission engine is **REAL code, unwired** (dead in production).

### Billing / metering / quotas (`backend/enterprise/billing/advanced_billing.go`) — **STUB, unwired, marketing-labeled**
- The file is substantial (1,160+ lines): `EnterpriseAccount`, `Subscription`, `Invoice`, `Commitment`, `Discount`, `PaymentMethod`, tiered/volume/usage pricing models, a `RevenueRecognitionEngine`, `CurrencyManager`, `PaymentGateway` — all fully-typed Go structs.
- Its package header reads: `// Package billing provides advanced enterprise billing and revenue management` / `// Supporting $100M+ ARR with 40%+ margins through sophisticated pricing models`.
- Storage is entirely **in-memory**: `AdvancedBillingEngine{ accounts map[string]*EnterpriseAccount, subscriptions map[string]*Subscription, invoices map[string]*Invoice, payments map[string]*Payment, ... mu sync.RWMutex }`. No database table, no persistence, no migration references it.
- Grep across the entire `backend/` tree for any import of `enterprise/billing` returns **zero results**. Nothing constructs an `AdvancedBillingEngine`; it is reachable from no binary.
- `backend/core/quotas` exists as a separate package but is likewise absent from `cmd/api-server`'s import graph.
- **Verdict: STUB.** Extensive types, zero callers, zero persistence, zero reachability from the running server.

### Marketplace (`marketplace/`) — **ABSENT (code); doc-only**
- The entire directory contains exactly one file: `marketplace/listings/aws/product-description.md` — AWS-Marketplace-style sales copy (pricing tiers, "SOC 2 Type II certified", "ISO 27001 certified", "HIPAA compliant", "FedRAMP authorized (GovCloud)", "99.99% availability SLA").
- No code, no listing schema, no integration with any cloud marketplace API. These compliance/certification claims are asserted in prose with no supporting evidence anywhere in the repo (no compliance program, no audit artifacts).
- **Verdict: ABSENT.** A single unsupported marketing document; no functionality.

### GPU / PCI passthrough — **ABSENT** in compiling code; present only in dead scaffolding
- The `HypervisorDriver` interface declares `SupportsGPUPassthrough() bool` (`backend/core/vm/vm_types_minimal.go:52`).
- Every driver that actually compiles into the canonical path hardcodes `false`:
  - `KVMDriverEnhanced` (**the canonical KVM driver** — confirmed: `main.go:1381`, `:1466` type-assert `driver.(*core_vm.KVMDriverEnhanced)`), `driver_kvm_enhanced.go:1376-1378`: `// Not implemented yet` / `return false`. Its `GetCapabilities()` also statically reports `IOMMUEnabled: false, GPUDevices: []GPUDevice{}` (`:1429-1433`).
  - `ContainerDriver`, `ContainerdDriver`, `ProcessDriver`, `MockHypervisor`, `CoreStubDriver` — all `return false`.
- The only code claiming `SupportsGPUPassthrough() bool { return true }` plus real `vfio`/PCI-hostdev XML types (`libvirt_driver.go.disabled:364-476,875-878`) is the **quarantined, non-compiling** libvirt driver (`.go.disabled`, `undefined: libvirt.Connect` — off the canonical path per `STATUS.md`'s 2026-07-04 quarantine).
- Real, verified device hot-plug exists (disk/net via QMP `blockdev-add`/`netdev_add`+`device_add` `[STATUS.md]`) but that is disjoint from PCI/VFIO GPU passthrough — no `vfio-pci` or `hostdev` string appears anywhere in the compiling codebase.
- **Verdict: ABSENT.**

### Backup / disaster recovery — **REAL as a library, completely UNWIRED**
- `backend/core/backup` is a real, substantial, independently-tested module (own `go.mod`): `snapshot_manager.go`, `cbt_tracker.go` (changed-block tracking — the one piece `STATUS.md`'s "Works" list explicitly names as real), `dedup.go`, `incremental_engine.go`, `replication_system.go`, `retention.go`, `restore.go`, `disaster_recovery.go`, `multicloud_storage.go`, plus `providers/{local,remote,encryption,compression,incremental}_provider.go` and their own tests.
- `backend/api/backup/handlers.go` wraps it in a full REST API (`BackupAPIServer`, `BackupCreateRequest/Response`, etc.) — but grep for `"backend/api/backup"` across the entire repo returns **zero importers**. This is a complete, tested HTTP layer mounted on no router anywhere.
- `backend/core/dr` (own `go.mod`, own README, own tests `dr_test.go`): `orchestrator.go`, `regional_failover.go`, `split_brain.go`, `health_monitor.go`, `restore_system.go`, `integrity_checker.go` — same story: real code, zero references from `cmd/api-server` or any other reachable binary.
- **Verdict: REAL implementation, UNWIRED.** This is genuine, tested engineering sitting completely outside the running product.

### Monitoring / alerting — **REAL as a library, UNWIRED from the canonical binary**
- `backend/core/monitoring`: `alert.go`, `analytics_engine.go`, `collectors.go`, `metric_aggregator.go`, `notification.go`, `vm_telemetry_collector.go`, `distributed_metric_collector.go` — real code with tests.
- `backend/api/monitoring/handlers.go` (WebSocket + REST `MonitoringHandlers`) is imported **only** by `main_enhanced.go` and `main_multicloud.go` (both gated behind `novacron_enhanced`/`novacron_multicloud` — never the canonical build). `main.go` has zero import of `core/monitoring` or `api/monitoring`.
- The fabric session did add lightweight, purpose-built observability directly in `cmd/api-server` (`GET /api/cluster/links` exposing `{rtt_ms, throughput_bps, measured_at, stale}` `[STATUS.md]`) — real but narrowly scoped to fabric link health, not the general-purpose Prometheus/alerting engine in `core/monitoring`.
- **Verdict: REAL as a library, largely UNWIRED.** A small, purpose-built subset (fabric link telemetry) is genuinely live in the canonical binary; the general alerting/analytics engine is not.

### The P2P fabric (cluster join, jobs, transfers, migration) — **REAL, confirmed**
- Confirmed unconditionally compiled into the canonical binary (no `//go:build` tag on any of these files):
  - `backend/cmd/api-server/cluster_join.go` — signed join: `POST /internal/cluster/join` carries `{node_id, addr, ts}` + HMAC-SHA256 `X-Join-Signature`, constant-time verified, rejects clock skew ≥60s, and requires a reachability callback to the joiner before admission (prevents peer-map poisoning). Membership persists in `cluster_peers` (migration `000006_cluster_peers`), refreshed by a 30s heartbeat that also measures throughput/RTT.
  - `backend/cmd/api-server/fabric_jobs.go` — a fabric job is literally a Process-driver VM; placement priority is pin > locality > measured-link cost > default; job truth comes from driver-observed process liveness + exit code, not a stale cache.
  - `backend/cmd/api-server/fabric_transfers.go` — one in-flight transfer per link, second transfer queued with a computed ETA; compression decision rule (link <500 Mbps AND sampled ratio >1.3 ⇒ zstd-multifd) applied via QMP `migrate-set-capabilities`/`migrate-set-parameters`.
  - `backend/core/vm/driver_kvm_enhanced.go` + `driver_kvm_migrate.go` — real QMP-driven live migration, including non-shared-storage block migration via NBD drive-mirror + RAM cutover, with dozens of dedicated regression tests (`driver_kvm_block_migrate_*_test.go`, `driver_kvm_migrate_*_test.go`, `driver_kvm_migration_cpu_hints_fallback_test.go`).
- `[STATUS.md]` cites live, non-simulated measurements corroborating this: 21.78 Gbps unshaped / 186.1 Mbps under `tc tbf 200mbit` / 47.9 Mbps under `50mbit` link throughput; a repo-owned two-node acceptance harness (`scripts/fabric/two-node-fabric-test.sh`) provisioning two real `api-server` processes + a network-namespaced, `tc`-shaped peer, passing 10/10 across a full join→job→queued-transfer→cross-node-migration assertion chain, wired into CI; and a documented, honestly-caveated compression benefit (2.17x wall-time, 2.2x fewer wire bytes for a RAM-dominant guest — explicitly *not* generalized to small/incompressible guests).
- **Verdict: REAL.** This is the one subsystem in the repo with live, adversarially-debugged, numerically-measured, CI-gated proof — not aspirational documentation.

---

## 3. Test reality

Not independently re-executed (read-only audit). `STATUS.md`'s self-reported, dated log (most recent entry 2026-09-20, one day before this audit) states:
- `cd backend/core && go build ./... && go vet ./...` → exit 0.
- `go test -short -race ./vm/` → green, ~148-157s across repeated runs (this is the package containing the KVM driver + migration logic).
- The "canonical root test set" (`api-server`, `api/graphql`, `api/security`, `api/websocket`, `pkg/config` — i.e., exactly the packages `main.go` actually imports) — green.
- The `backend/core` `-short` census (all ~50+ sub-packages under `go test -short`) — green, 0 FAIL, versus a documented 2026-09-04 baseline of 7 FAIL packages before that session's fixes.
- Frontend: `tsc --noEmit` 0 errors, `npm run lint` 0 errors, canonical jest 14 suites / 34 tests green, `next build` green.
- The two-node fabric acceptance harness — 10/10 PASS, twice consecutively, with confirmed clean teardown.
- Explicitly **not** green / off-CI, by `STATUS.md`'s own admission: most of `network/dwcp`'s active test suite (multi-stream TCP transport dialing a live peer the unit env can't provide; real config/validation drift in a few tests) — pre-existing, off the canonical path, left as documented debt rather than fixed.

This is a credible, execution-grounded log (it names exact commands, timings, and prior failure counts, and is corroborated by everything this audit could check independently — e.g. the exact import list, build tags, and file wiring all match what the log describes) — but it is the project's own self-report, not a rerun performed for this audit.

---

## 4. Dead/fabricated surface area

**Marketing-style revenue-claim comments found via grep** (none of these constructs are reachable from the canonical binary):

| File | Claim |
|---|---|
| `backend/enterprise/billing/advanced_billing.go:2` | "Supporting $100M+ ARR with 40%+ margins through sophisticated pricing models" |
| `backend/business/pricing/pricing_optimization.py:4` | "sophisticated pricing strategies to maximize $1B ARR achievement while maintaining 42% margins" |
| `backend/business/revenue/acceleration_engine.go:1-2,15-18` | "Revenue Acceleration Engine for achieving $1B ARR milestone through 10x growth automation"; `CurrentARR float64 // Current $120M ARR` |
| `backend/business/revenue/billion_arr_tracker.go:1,13` | "$1B ARR milestone tracking and revenue acceleration" |
| `backend/business/revenue/revenue_coordinator.go:1,343,356` | "$1B ARR achievement coordination"; `"🎉 $1B ARR MILESTONE ACHIEVED!"` |
| `backend/business/validation/metrics_validator.go:1-2` | "Validates $1B ARR milestone tracking, 50%+ market share achievement" |
| `backend/business/verticals/vertical_domination.py:648` | "100M+ Subscriber Carrier" reference architecture |
| `marketplace/listings/aws/product-description.md` | "99.99% availability SLA", "SOC 2 Type II certified", "ISO 27001 certified", "HIPAA compliant", "PCI DSS compliant", "FedRAMP authorized" — none corroborated anywhere else in the repo |

**Scope of the speculative surface:** a cluster of top-level directories exists purely as narrative business simulation, entirely disjoint from the canonical import graph (confirmed: `cmd/api-server` imports none of them): `backend/business/{expansion,fortune500,rev_ops,revenue,validation,verticals,pricing}`, `backend/corporate/{ecosystem,integration,ma,strategy}`, `backend/competitive/{displacement,dominance,leadership}` (files named `market_share_tracker.go`, `displacement_engine.go`, `market_leadership.go`), `backend/partners/{strategic}`, `backend/ecosystem/`, `backend/sustainability/` (`esg_leadership.py`), and `backend/sales/` (`enterprise_intelligence.py`, `intelligence/sales_forecasting.py`). That's roughly two dozen files of pure fictional-business-metrics code (file names alone — "Fortune 500 acceleration," "market displacement engine," "ESG leadership," "billion-ARR tracker" — describe a company roleplay, not software) sitting alongside `backend/core`'s several hundred real, tested Go files. It is a small fraction of total file count but a disproportionately large fraction of the repo's *aspirational* prose.

**Practical read:** the fabrication is concentrated and identifiable — it lives in clearly-named, clearly-isolated directories (`business/`, `corporate/`, `competitive/`, `sales/`, `partners/`, `ecosystem/`, `sustainability/`, `enterprise/billing`) that share no code path with `backend/core` or `backend/cmd/api-server`. Deleting all of it would not break the canonical build (confirmed: zero cross-imports found). `STATUS.md` itself already flags a related but distinct problem — `docs/archive/fabricated-claims/` and a ~4,900-file documentation-pollution debt (`novacron-fb8`, "72 fabricated 99.999% claims") — as a known, unresolved, user-approval-gated cleanup item.

---

## 5. What is genuinely differentiated — ranked by difficulty to copy

1. **Bandwidth-aware transfer admission + adaptive compression decision on measured (not assumed) link state.** `backend/cmd/api-server/fabric_transfers.go` + the KVM driver's QMP compression wiring. The decision rule (queue-with-ETA on a busy measured link; zstd-multifd only when the link is slow *and* the sampled payload is actually compressible) requires (a) a live throughput/RTT probe feeding a decaying link profile, (b) QMP capability negotiation that degrades gracefully on older QEMU (`query-migrate-parameters` probed before use), and (c) end-to-end proof that the decision changes real wire bytes, not just a flag. `[STATUS.md]` documents this proven live with a measured 2.17x wall-time and 2.2x wire-byte improvement for the case it targets, and — just as importantly — an honest negative result (no win for small/incompressible VMs) rather than a blanket claim. Competitors that fake "intelligent" placement/compression with static thresholds cannot produce this kind of measured, conditional, occasionally-negative result; building the real version requires deep QEMU internals plus a live-traffic feedback loop.

2. **Cross-node, non-shared-storage (block) live migration with ownership handoff and crash-safe re-adoption.** `backend/core/vm/driver_kvm_migrate.go`, `driver_kvm_enhanced.go`, and the dozen dedicated regression tests covering drive-mirror-to-a-fresh-disk, deferred `-incoming` (to close the "stray connection steals the migration stream" class of bug), source-exit-before-dest-registration races, and boot-time reconciliation of orphaned migration destinations. This is the hardest class of hypervisor bug to get right (concurrent QMP state machines across two independent processes, crash windows on both sides) and it comes with a working two-node acceptance harness that exercises it against real QEMU, not mocks. Most "distributed VM platforms" at this scale either don't attempt non-shared-storage migration or don't have it working across process/node restarts.

3. **Signed, reachability-verified cluster admission feeding a decaying, per-link scheduling signal.** `backend/cmd/api-server/cluster_join.go`: HMAC-SHA256-signed join with clock-skew rejection, constant-time comparison, and — the part that's easy to skip and hard to bolt on later — a mandatory callback-reachability check of the joiner *before* it is admitted, closing the peer-map-poisoning hole that a naive "trust the join request" design leaves open. Combined with the heartbeat-driven link-profile decay that placement/admission actually consumes, this is a coherent security-plus-scheduling primitive, not a decorative auth check. It is straightforward to fake (a static peer list, or join-without-verification) and comparatively hard to build correctly with real NAT/veth-shaped-link testing behind it, which `[STATUS.md]` documents was actually done (a netns/MASQUERADE/DNAT node join was verified live).

**What's notably *not* on this list despite being present in the codebase:** the RBAC/2FA/JWT stack, backup/DR engines, and monitoring/alerting engine are real, non-trivial engineering — but none of them is differentiated *for this product*: they are the same kind of feature every VM/orchestration platform has (or claims to have), several of them aren't even wired into the running server today, and none requires the same category of distributed-systems-under-real-network-conditions engineering that the fabric does.
