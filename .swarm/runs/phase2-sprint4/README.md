# Phase 2 Sprint 4: Federation and Discovery Baseline

## Status

In progress.

## Implemented Slice

- Added the canonical signed node-inventory contract in `backend/core/federation`.
- Extended the shared runtime manifest with optional `runtime.discovery_seeds`.
- Mirrored discovery seeds through the API-server runtime-manifest summary and canonical daemon service report.

## Current Boundary

- Discovery seeds are declarative only in this slice.
- The daemon does not yet start a federation/discovery manager from the manifest.
- DWCP internet discovery remains experimental and is not the canonical Phase 2 path.

## Validation

- `GOTOOLCHAIN=go1.24.6 go test ./federation` from `backend/core`
- `GOTOOLCHAIN=go1.24.6 go test ./config` from `backend/core/initialization`
- `GOTOOLCHAIN=go1.24.6 go test ./backend/pkg/config` from repo root
- `GOTOOLCHAIN=go1.24.6 go test ./cmd/novacron` from `backend/core`
