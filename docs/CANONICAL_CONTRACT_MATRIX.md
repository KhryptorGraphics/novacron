# NovaCron Canonical Contract Matrix

This document is the current source of truth for the shipped control-plane surface.
Treat older README sections, feature reports, and alternate entrypoints as historical context unless they match this matrix.

## Environment

| Key | Status | Notes |
| --- | --- | --- |
| `NEXT_PUBLIC_API_URL` | live | Single supported frontend origin input. Example: `http://localhost:8090`. |
| `NEXT_PUBLIC_WS_URL` | compat | Deprecated override still honored by `frontend/src/lib/api/origin.ts`; no deployment sets it. WebSocket origins derive from `NEXT_PUBLIC_API_URL`. |
| `NEXT_PUBLIC_API_BASE_URL` | retired | Do not use for new code. `/api/v1` is derived from `NEXT_PUBLIC_API_URL`. |
| `AUTH_SECRET` | live | Required by the canonical Go API server. |
| `DB_URL` | live | Required by the canonical Go API server. |
| `STORAGE_PATH` | live | Required by the canonical Go API server. |
| `API_PORT` | live | Single listener (default `8090`) serving HTTP, GraphQL and every WebSocket route. There is no separate WebSocket port; `WS_PORT` is retired. |
| `CORS_ALLOWED_ORIGINS` | live | Comma-separated browser origins for CORS and the WebSocket `Origin` check (no-Origin and same-origin requests are always allowed; `*` allows all). |
| `NOVACRON_TRUSTED_PROXIES` | live | Comma-separated IPs/CIDRs of reverse proxies. Only a peer in this list may supply `X-Forwarded-For`/`X-Real-IP`; the client IP is the first untrusted hop walking `X-Forwarded-For` right to left. Unset = trust nobody (the peer address is the client). MUST be set wherever a proxy fronts the API, or every client shares the proxy's login rate-limit bucket. |
| `NOVACRON_LOGIN_RATE_LIMIT` / `NOVACRON_LOGIN_RATE_WINDOW_S` | live | Per-client login attempts per window (default 10 per 300 s; `0` disables). |
| `NOVACRON_MIGRATION_PORT_RANGE` | live | Inclusive LO-HI TCP range for incoming QEMU migration and NBD block-migration listeners; default 49152-49215 (libvirt's default range). Keep it a power-of-two-aligned block so deploy/p2pnet QoS classifies it with one mask. |

## HTTP Surface

| Route | Status | Notes |
| --- | --- | --- |
| `GET /health` | live | Canonical health endpoint. |
| `GET /api/info` | live | Canonical service metadata endpoint. `auth.providers` lists enabled login providers (`["password"]` on this server); the frontend shows provider buttons only for advertised providers. |
| `POST /api/auth/login` | live | Canonical login route. Returns the frontend `AuthResponse`: `token` (HS256 access token carrying a `sid` session claim), `refreshToken` (opaque; only its SHA-256 is stored), `expiresAt`, `user`, `admission`, `memberships`, `selectedCluster`, `session`. The local fabric is the single admitted cluster. |
| `POST /api/auth/register` | live | Canonical registration route. |
| `GET /api/auth/check-email` | live | Canonical email availability route. |
| `POST /api/auth/2fa/verify-login` | live | Completes pending 2FA login challenge; returns the same `AuthResponse` as login. Pending-2FA tokens are rejected by every other authenticated route. |
| `GET /api/auth/me` | live | Authenticated. Current user, memberships, selected cluster and session; requires a token with a live `sid` session. |
| `POST /api/auth/refresh` | live | Public (called when the access token is missing/expired). Body `{refreshToken}`. Rotates the refresh token (compare-and-swap); reusing a rotated token revokes the session. Re-reads the user row: inactive users or credentials changed after the session started → 401. |
| `POST /api/auth/logout` | live | Public. Revokes the session identified by `{refreshToken}`, falling back to the bearer token's `sid`, so logout works with an expired access token. |
| `GET /api/auth/sessions` | live | Authenticated. The caller's unrevoked, unexpired sessions. |
| `GET /api/cluster/admissions` | live | Authenticated. Cluster admissions for the caller (the local fabric). |
| `POST /api/cluster/admissions/select` | live | Authenticated. Selects the admitted cluster for the session. |
| `POST /api/auth/2fa/setup` | live | Authenticated route. |
| `GET /api/auth/2fa/qr` | live | Authenticated route. |
| `POST /api/auth/2fa/verify` | live | Authenticated route. |
| `POST /api/auth/2fa/enable` | live | Authenticated route. |
| `POST /api/auth/2fa/disable` | live | Authenticated route. |
| `GET /api/auth/2fa/status` | live | Authenticated route. |
| `GET/POST /api/auth/2fa/backup-codes` | live | Authenticated route. |
| `POST /auth/login` | compat | Legacy alias retained during gradual cutover. |
| `POST /auth/register` | compat | Legacy alias retained during gradual cutover. |
| `POST /api/auth/forgot-password` | live | Issues a single-use `password_reset` token (sha256-hashed in `auth_tokens`), emails a reset link; always returns a generic success message. 503 when SMTP is unconfigured. |
| `POST /api/auth/reset-password` | live | Consumes a live `password_reset` token: rotates the password hash, revokes sessions, marks the token used. |
| `POST /api/auth/verify-email` | live | Consumes a live `email_verification` token: sets `email_verified = TRUE` and promotes `pending` users to `active`. |
| `POST /api/auth/resend-verification` | live | Re-issues a verification email for unverified accounts; always returns `{"success":true}`. 503 when SMTP is unconfigured. |
| `GET/POST /api/v1/vms` | live | Canonical VM list/create route set. |
| `GET/DELETE /api/v1/vms/{id}` | live | Canonical VM detail/delete route set. |
| `POST /api/v1/vms/{id}/start` | live | Canonical VM action route. |
| `POST /api/v1/vms/{id}/stop` | live | Canonical VM action route. |
| `POST /api/v1/vms/{id}/pause` | live | Canonical VM action route. |
| `POST /api/v1/vms/{id}/resume` | live | Canonical VM action route. |
| `POST /api/v1/vms/{id}/restart` | live | Canonical VM action route. |
| `GET /api/v1/vms/{id}/metrics` | live | Canonical VM metrics route. |
| `GET /api/v1/monitoring/metrics` | live | Canonical monitoring summary route used by the routed monitoring dashboard. |
| `GET /api/v1/monitoring/vms` | live | Canonical monitoring VM summary route used by the routed monitoring dashboard. |
| `GET /api/v1/monitoring/alerts` | live | Recent alerts from the in-process alert store (VM errors and healing events), the same events pushed on `/api/ws/alerts`. |
| `POST /api/v1/monitoring/alerts/{id}/acknowledge` | deferred | Frontend should present this as unavailable until the canonical server exposes it. |
| `GET /api/v1/networks` | live | Networks catalog (migration `000018_networks`): NovaCron-managed host bridges of this node. `vm_count` is org-scoped for non-admins. |
| `POST /api/v1/networks` | live | Admin/super-admin. Body `{name, cidr, gateway?, vlan_id?, mtu?}`; inserts the row and provisions the bridge (`ncbr-<id prefix>`, optional 802.1Q port on `NOVACRON_NETWORK_UPLINK`, qemu-bridge-helper ACL) in one transaction. 409 on duplicate name / overlapping CIDR / host conflicts. |
| `GET/DELETE /api/v1/networks/{id}` | live | Delete is admin-only and refused (409, `vm_ids`) while any VM has `network_id` set to it; otherwise removes the bridge and the row together. |
| `POST /api/v1/vms` `network_id` | live | Bridges the KVM guest's primary NIC onto that catalog network (`-netdev bridge`); persisted in `vms.network_id`. Forces local placement (the catalog is node-local). |
| `GET/POST /api/v1/vms/{vm_id}/interfaces` | live | Canonical VM interface list/attach route set. |
| `GET/PUT/DELETE /api/v1/vms/{vm_id}/interfaces/{id}` | live | Canonical VM interface detail/update/delete route set. |
| `/api/vms*` and `/api/monitoring/*` | compat | Legacy secure aliases retained during gradual cutover. |
| `/api/networks*` and `/api/vms/{vm_id}/interfaces*` | compat | Legacy secure aliases retained during gradual cutover. |
| `/api/security/*` | live | Canonical admin/security surface. Requires auth and admin/super-admin roles. Includes event acknowledgement, compliance recheck/export, manual incidents, audit export, and RBAC assignment. |
| `/api/admin/security/*` | live | Canonical alias for admin/security UI. Requires auth and admin/super-admin roles and mirrors `/api/security/*`. |
| `/api/admin/users*` | live | Canonical admin-only user management surface. Supports list/create/update/delete plus role assignment. |
| `POST /graphql` | live | Public release GraphQL surface is storage-backed volume operations only: `volumes`, `createVolume`, and `changeVolumeTier`. VM/cluster resolvers, subscriptions and `schema.graphql` were removed. |
| `/api/orchestration/*` | live | Admin/super-admin only (autoscaling, policies, placement, healing). Healing: only VM targets are healed: restart goes through the VM restart supervisor (honors user stops, restart policy `no`, and exhausted retries); migrate queues a fabric transfer to a reachable peer. Service/node/cluster targets, scaling and failover return an explicit unsupported error. |

## WebSocket Surface

WebSocket routes are served on `API_PORT`. Authentication: an `Authorization: Bearer <token>` header, or, for browsers (which cannot set handshake headers), the subprotocol pair `Sec-WebSocket-Protocol: bearer, <token>`; the server echoes `bearer`. Tokens in query strings are not accepted. Same fail-closed rules as HTTP (expired/revoked/pending-2FA tokens → 401 before upgrade). The `Origin` header must be absent, same-origin, or listed in `CORS_ALLOWED_ORIGINS`.

| Route | Status | Notes |
| --- | --- | --- |
| `GET /api/ws/console/{vmId}` | live | Canonical console channel. |
| `GET /api/ws/metrics` | live | Canonical metrics stream: `{type:"metric", source, metrics, timestamp}` per client at `?interval=` (1–300 s), filtered by `?sources=`. Host metrics are sampled by one shared sampler. |
| `GET /api/ws/alerts` | live | Canonical alert stream: `{type, data, timestamp}` with `type` `security_alert` (VM errors, healing events, heartbeat-detected node unreachable/reachable after 3 missed/1 good probe, triggered autoscaler scale-up/down) or `vm_status` (`data.id/status/previous_status`). Fanned out to every subscriber; slow clients are disconnected. |
| `GET /api/ws/logs` | live | Canonical log stream: api-server log entries at or above `LOG_STREAM_LEVEL` (default `info`) as `{type:"log", source:"system", level, message, timestamp, component:"api-server", vm_id?, labels}`; filters `?level=`, `?components=`, `?vm_id=`. Best-effort: entries are dropped when the broadcast queue is full. |
| `GET /api/ws/logs/{source}` | live | Canonical source-scoped log stream. |
| `GET /api/ws/security/events` | live | Canonical security event stream. |
| `/ws/console/*`, `/ws/metrics`, `/ws/alerts`, `/ws/logs*` | compat | Legacy websocket aliases retained during gradual cutover. |
| `GET /api/security/events/stream` | compat | Legacy security websocket alias retained during gradual cutover. |
| `GET /api/ws/admin` | deferred | Frontend code still references this channel, but the canonical server does not expose it. |
| `GET /ws/events/v1` | deferred | Legacy generic event stream is not part of the canonical server. |
| `GET /api/ws/vms*` | deferred | Frontend assumptions exist; canonical server does not expose these channels yet. |
| `GET /api/ws/network*` | deferred | Frontend assumptions exist; canonical server does not expose these channels yet. |
| `GET /api/ws/storage*` | deferred | Frontend assumptions exist; canonical server does not expose these channels yet. |
| `GET /api/ws/jobs*` | deferred | Frontend assumptions exist; canonical server does not expose these channels yet. |
| `GET /api/ws/ai/*` | deferred | Frontend assumptions exist; canonical server does not expose these channels yet. |

## Notes for Implementation

- New frontend work should build URLs through `frontend/src/lib/api/origin.ts`.
- Frontend realtime helpers should fail closed for `deferred` websocket channels instead of opening speculative connections.
- The routed dashboard should expose only canonical VM, monitoring, storage, and security surfaces. Experimental fabric, AI, topology, mobile, and deferred realtime views are not part of the release path.
- The routed admin surface is restricted to canonical `security`, `roles & permissions`, and `audit` tabs. `/admin/users`, `/admin/analytics`, and `/admin/config` redirect back to `/admin`.
- The routed `/users` page is admin-only and backed exclusively by `/api/admin/users*`.
- The routed `/network` page is narrowed to live inventory and interface attachment operations only. Topology, QoS, and traffic analytics remain deferred.
- The routed `/analytics` page is read-only and composed from live dashboard domains; it must not synthesize historical trends or mock charts.
- The routed `/settings` page is limited to account and security controls, including the canonical 2FA flow.
- `/core/vms` is a compatibility route that intentionally renders the same canonical implementation as `/vms`.
- The routed storage surface is volume-only. Pools, snapshots, backups, deletion, and storage realtime channels are intentionally out of scope for the release candidate.
- New backend work should extend canonical paths first and add compat aliases only when required by the gradual cutover plan.
- If a route or channel is not marked `live` or `compat` here, treat it as unsupported until it is explicitly implemented and promoted.
- The API server refuses to boot unless golang-migrate's `schema_migrations` reports a clean (non-dirty) version at or above the migration it requires (currently `000018_networks`). Apply `database/migrations` first (the `novacron/migrate` image / `database/migrate.go`).
- Catalog networks are re-provisioned at api-server boot (`reconcileNetworks`), since host bridges do not survive a reboot. Provisioning needs `CAP_NET_ADMIN`; the qemu-bridge-helper allow list is kept under `NOVACRON_QEMU_BRIDGE_ACL_DIR` (default `/etc/qemu`, `none` to manage `bridge.conf` by hand).
- `backend/cmd/api-server` is the only control-plane server binary. `backend/cmd/core-server` and its `backend/api/vm` handlers were retired; `backend/core/cmd/novacron` is the hypervisor node agent (`-listen`, health at `/healthz`).
