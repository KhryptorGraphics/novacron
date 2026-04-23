# Phase 1 Sprint 2 Follow-up: Runtime Monitoring Read Bridge

## Objective

Deliver the first guarded runtime-backed canonical read without creating split-brain behavior across the wider API surface.

## Scope

- Add one internal daemon endpoint: `GET /internal/runtime/v1/monitoring/metrics`
- Keep the response shape identical to the canonical `GET /api/v1/monitoring/metrics` payload
- Add one opt-in proxy path in `backend/cmd/api-server/main.go` for:
  - `GET /api/v1/monitoring/metrics`
  - `GET /api/monitoring/metrics`
- Preserve the current synthetic fallback when the runtime bridge is disabled or unavailable
- Add source observability with `X-NovaCron-Read-Source: runtime|sql-fallback`

## Guards

- The runtime bridge is off by default.
- The canonical API server only enables the bridge when:
  - `CANONICAL_RUNTIME_MONITORING_READS=true`
  - `CANONICAL_RUNTIME_BASE_URL` points at the daemon runtime API base URL
- The canonical API server does not forward end-user auth headers to the internal daemon route.
- VM, monitoring inventory, and network read paths remain on their existing sources until source-of-truth convergence is completed.

## Runtime Data Model

- CPU, memory, and disk usage are aggregated from `vmManager.ListSchedulerNodes()`
- Network usage is deterministic `0` until the daemon exposes a live network metric in the runtime inventory
- The daemon returns deterministic `timeLabels`, `cpuAnalysis`, and `memoryAnalysis` strings so tests and dashboards receive a stable contract

## Verification

- `GOTOOLCHAIN=go1.24.6 go test ./cmd/novacron` in `backend/core`
- `GOTOOLCHAIN=go1.24.6 go test ./backend/cmd/api-server`

## Deferred

- Runtime-backed `GET /api/v1/vms`
- Runtime-backed `GET /api/v1/vms/{id}`
- Runtime-backed `GET /api/v1/vms/{id}/metrics`
- Runtime-backed `GET /api/v1/monitoring/vms`
- Runtime-backed network inventory reads

These remain blocked on `novacron-0ex`, which tracks source-of-truth convergence before canonical read cutover.
