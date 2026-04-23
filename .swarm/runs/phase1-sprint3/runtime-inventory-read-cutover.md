# Runtime Inventory Read Cutover

## Problem

The canonical API server already owned VM and network writes through Postgres tables, while the Phase 1 daemon owned a separate in-memory VM and network runtime. Moving canonical reads directly onto the in-memory managers would have created split-brain behavior after restart and whenever SQL-backed writes were not mirrored into the daemon process.

## Decision

Phase 1 converges VM and network inventory reads on the persisted inventory tables already used by the canonical API:

- `vms`
- `vm_metrics`
- `networks`
- `vm_interfaces`

The daemon now exposes internal read endpoints backed by those tables:

- `GET /internal/runtime/v1/vms`
- `GET /internal/runtime/v1/vms/{id}`
- `GET /internal/runtime/v1/vms/{id}/metrics`
- `GET /internal/runtime/v1/monitoring/vms`
- `GET /internal/runtime/v1/networks`
- `GET /internal/runtime/v1/networks/{id}`
- `GET /internal/runtime/v1/vms/{vm_id}/interfaces`
- `GET /internal/runtime/v1/vms/{vm_id}/interfaces/{id}`

The canonical API now proxies the matching external GET routes to those daemon endpoints when `CANONICAL_RUNTIME_INVENTORY_READS=true` and `CANONICAL_RUNTIME_BASE_URL` are set. If the daemon route is unavailable, the API falls back to its existing SQL handler and marks the response with `X-NovaCron-Read-Source: sql-fallback`.

## Why This Closes Phase 1

- The canonical runtime entrypoint remains `backend/core/cmd/novacron/main.go`.
- Canonical read traffic can now flow through the daemon without inventing a second write path.
- The converged read source of truth is restart-stable and matches the current write authority.
- The cutover keeps wire compatibility for current UI and client contracts.

## Explicit Non-Goals

- No VM or network write-through to the in-memory daemon managers.
- No claim that `VMManager` or `NetworkManager` is the canonical persisted inventory authority yet.
- No tenant-scope tightening in the inventory queries.
- No replacement of the synthetic `/monitoring/vms` envelope shape in this sprint.

## Rollback

Rollback is flag-only:

- Set `CANONICAL_RUNTIME_INVENTORY_READS=false`, or
- unset `CANONICAL_RUNTIME_BASE_URL`.

The canonical API then returns to its direct SQL handlers without code reversion.

## Follow-On Work

Phase 2 and later can decide whether to:

- keep SQL-backed inventory as the durable source of truth and push more reads through the daemon, or
- add real write-through and hydration so the in-memory runtime managers can become authoritative without lying about restart behavior.
