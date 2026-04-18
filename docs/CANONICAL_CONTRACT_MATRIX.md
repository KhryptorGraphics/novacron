# NovaCron Canonical Contract Matrix

This document is the current source of truth for the shipped control-plane surface.
Treat older README sections, feature reports, and alternate entrypoints as historical context unless they match this matrix.

## Environment

| Key | Status | Notes |
| --- | --- | --- |
| `NEXT_PUBLIC_API_URL` | live | Single supported frontend origin input. Example: `http://localhost:8090`. |
| `NEXT_PUBLIC_WS_URL` | compat | Temporary override only. The frontend should derive websocket origins from `NEXT_PUBLIC_API_URL` by default. |
| `NEXT_PUBLIC_API_BASE_URL` | retired | Do not use for new code. `/api/v1` is derived from `NEXT_PUBLIC_API_URL`. |
| `AUTH_SECRET` | live | Required by the canonical Go API server. |
| `DB_URL` | live | Required by the canonical Go API server. |
| `STORAGE_PATH` | live | Required by the canonical Go API server. |

## HTTP Surface

| Route | Status | Notes |
| --- | --- | --- |
| `GET /health` | live | Canonical health endpoint. |
| `GET /api/info` | live | Canonical service metadata endpoint. |
| `POST /api/auth/login` | live | Canonical login route. |
| `POST /api/auth/register` | live | Canonical registration route. |
| `GET /api/auth/check-email` | live | Canonical email availability route. |
| `POST /api/auth/2fa/verify-login` | live | Completes pending 2FA login challenge. |
| `POST /api/auth/2fa/setup` | live | Authenticated route. |
| `GET /api/auth/2fa/qr` | live | Authenticated route. |
| `POST /api/auth/2fa/verify` | live | Authenticated route. |
| `POST /api/auth/2fa/enable` | live | Authenticated route. |
| `POST /api/auth/2fa/disable` | live | Authenticated route. |
| `GET /api/auth/2fa/status` | live | Authenticated route. |
| `GET/POST /api/auth/2fa/backup-codes` | live | Authenticated route. |
| `POST /auth/login` | compat | Legacy alias retained during gradual cutover. |
| `POST /auth/register` | compat | Legacy alias retained during gradual cutover. |
| `POST /api/auth/forgot-password` | live | Canonical password-reset request route. Returns a generic success message. |
| `POST /api/auth/reset-password` | live | Canonical password-reset completion route. Returns a generic success message. |
| `POST /api/auth/verify-email` | deferred | Canonical server still returns not implemented; the routed registration flow does not depend on it. |
| `POST /api/auth/resend-verification` | deferred | Canonical server still returns not implemented; the routed registration flow does not depend on it. |
| `GET/POST /api/v1/vms` | live | Canonical VM list/create route set. |
| `GET/DELETE /api/v1/vms/{id}` | live | Canonical VM detail/delete route set. |
| `POST /api/v1/vms/{id}/start` | live | Canonical VM action route. |
| `POST /api/v1/vms/{id}/stop` | live | Canonical VM action route. |
| `GET /api/v1/vms/{id}/metrics` | live | Canonical VM metrics route. |
| `GET /api/v1/monitoring/metrics` | live | Canonical monitoring summary route used by the routed monitoring dashboard. |
| `GET /api/v1/monitoring/vms` | live | Canonical monitoring VM summary route used by the routed monitoring dashboard. |
| `GET /api/v1/monitoring/alerts` | live | Canonical monitoring alert route used by the routed monitoring dashboard. |
| `POST /api/v1/monitoring/alerts/{id}/acknowledge` | deferred | Frontend should present this as unavailable until the canonical server exposes it. |
| `/api/vms*` and `/api/monitoring/*` | compat | Legacy secure aliases retained during gradual cutover. |
| `/api/security/*` | live | Canonical admin/security surface. Requires auth and admin/super-admin roles. Includes event acknowledgement, compliance recheck/export, manual incidents, audit export, and RBAC assignment. |
| `/api/admin/security/*` | live | Canonical alias for admin/security UI. Requires auth and admin/super-admin roles and mirrors `/api/security/*`. |
| `POST /graphql` | live | Public release GraphQL surface is storage-backed volume operations only: `volumes`, `createVolume`, and `changeVolumeTier`. |

## WebSocket Surface

| Route | Status | Notes |
| --- | --- | --- |
| `GET /api/ws/console/{vmId}` | live | Canonical console channel. |
| `GET /api/ws/metrics` | live | Canonical metrics stream. |
| `GET /api/ws/alerts` | live | Canonical alert stream. |
| `GET /api/ws/logs` | live | Canonical log stream. |
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
- The routed storage surface is volume-only. Pools, snapshots, backups, deletion, and storage realtime channels are intentionally out of scope for the release candidate.
- New backend work should extend canonical paths first and add compat aliases only when required by the gradual cutover plan.
- If a route or channel is not marked `live` or `compat` here, treat it as unsupported until it is explicitly implemented and promoted.
