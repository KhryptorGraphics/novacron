# Phase 3 Sprint 6: Mobility, Storage, And Resilience

## Status

In progress.

## Implemented Slice

- Added the canonical VM mobility policy contract in `backend/core/vm`.
- Normalized VM create requests onto the mobility policy contract.
- Added canonical daemon reporting at `/internal/runtime/v1/mobility/policy`.
- Kept live migration gated by default; `runtime.migration_mode: live` falls back to checkpoint as the default executable mode until latency, storage, and driver gates are explicit.
- Added explicit backup service reporting in `/internal/runtime/v1/services`.

## Current Boundary

- Cold and checkpoint policy are represented and validated, but the daemon does not yet execute cross-node migration from the new policy endpoint.
- The richer `backend/core/backup` and `backend/core/dr` subsystems remain unwired to `backend/core/cmd/novacron`.
- Advanced live migration remains behind existing optional/build-tagged paths.

## Validation

- `GOTOOLCHAIN=go1.24.6 go test ./cmd/novacron` from `backend/core`
- `GOTOOLCHAIN=go1.24.6 go test ./vm/migration_backup_policy.go ./vm/migration_backup_policy_test.go` from `backend/core`
- `GOTOOLCHAIN=go1.24.6 go test ./backend/pkg/services` from repo root
