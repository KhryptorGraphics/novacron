# Phase 1 Sprint 2: Shared Runtime Manifest Baseline

## Objective

Introduce the first converged runtime-manifest contract and make both active entrypoints aware of it without breaking the existing canonical API server or the legacy daemon-only YAML path.

## Deliverables

- `manifest-baseline.md`
- `state.yaml`

## Scope Of This Slice

- Extend `backend/core/initialization/config/loader.go` with a versioned runtime-manifest section that declares deployment profile, discovery mode, federation mode, migration mode, auth mode, storage classes, and enabled services.
- Teach `backend/core/cmd/novacron/main.go` to load the shared manifest format first and fall back to the existing daemon-only YAML when the file is still in the legacy shape.
- Teach `backend/pkg/config/config.go` to load an optional shared runtime manifest summary so `backend/cmd/api-server/main.go` can report the same contract on `/api/info` and `/health`.

## Explicit Non-Goals

- No API write-path convergence yet.
- No replacement of the canonical API server with the core daemon.
- No service gating beyond the runtime-auth toggle that can already be expressed by the daemon config.
- No federation, discovery, migration, backup, edge, or ML wiring changes beyond manifest declaration.

## Exit Status

- Shared runtime-manifest contract: complete
- Canonical daemon manifest awareness with legacy fallback: complete
- Canonical API manifest awareness: complete
- Canonical API runtime-backed reads: deferred
- One startup path / one shutdown path: deferred to later Phase 1 work
