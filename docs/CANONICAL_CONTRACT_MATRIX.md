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
| `POST /api/auth/forgot-password` | deferred | Canonical server returns not implemented. |
| `POST /api/auth/reset-password` | deferred | Canonical server returns not implemented. |
| `POST /api/auth/verify-email` | deferred | Canonical server returns not implemented. |
| `POST /api/auth/resend-verification` | deferred | Canonical server returns not implemented. |
| `GET/POST /api/v1/vms` | live | Canonical VM list/create route set. |
| `GET/DELETE /api/v1/vms/{id}` | live | Canonical VM detail/delete route set. |
| `POST /api/v1/vms/{id}/start` | live | Canonical VM action route. |
| `POST /api/v1/vms/{id}/stop` | live | Canonical VM action route. |
| `GET /api/v1/vms/{id}/metrics` | live | Canonical VM metrics route. |
| `GET /api/v1/monitoring/metrics` | compat | Still exposed by the canonical server, but not the preferred long-term surface. |
| `GET /api/v1/monitoring/vms` | compat | Still exposed by the canonical server, but not the preferred long-term surface. |
| `GET /api/v1/monitoring/alerts` | compat | Still exposed by the canonical server, but not the preferred long-term surface. |
| `/api/vms*` and `/api/monitoring/*` | compat | Legacy secure aliases retained during gradual cutover. |
| `/api/security/*` | live | Canonical admin/security surface. Requires auth and admin/super-admin roles. |
| `/api/admin/security/*` | live | Canonical alias for admin/security UI. Requires auth and admin/super-admin roles. |
| `POST /graphql` | live | Supported GraphQL surface is storage-backed volume operations only. |

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
- New backend work should extend canonical paths first and add compat aliases only when required by the gradual cutover plan.
- If a route or channel is not marked `live` or `compat` here, treat it as unsupported until it is explicitly implemented and promoted.
