# Phase 0 Sprint 1: Toolchain And CI Normalization

## Objective

Establish one supported Go toolchain source, one supported Node toolchain source, one default frontend package-manager path, and one canonical automatic CI hierarchy for the core backend, frontend, integration, and smoke lanes.

## Deliverables

- `toolchain-baseline.md`
- `state.yaml`

## Canonical Defaults

- Go defaults to the root `go.mod` declaration (`go 1.24.0`, `toolchain go1.24.6`) for the core platform lanes.
- Node defaults to `.nvmrc` (`20`) for the root Playwright lane and the frontend workspace.
- npm with the existing root and `frontend/` lockfiles is the only supported default JavaScript package-manager path for the convergence baseline.
- `.github/workflows/ci.yml` is the canonical automatic verification workflow, with `.github/workflows/integration-tests.yml` and `.github/workflows/e2e-tests.yml` retained as targeted automatic lanes.
- The canonical frontend lane is smoke-test plus production build. Repo-wide lint remains available as an explicit developer command but is not part of the Sprint 1 gate because the workspace still carries legacy lint debt outside the shipping slice.

## Isolation Decisions

- Legacy general CI/CD workflows were moved to `workflow_dispatch` so they no longer compete with the canonical path.
- DWCP v3 CI/CD and visual-regression automation were also moved to `workflow_dispatch` because they are specialized or prototype lanes with unresolved script and toolchain drift.
- Scheduled or path-filtered specialized workflows remain outside the core Sprint 1 baseline unless they overlap the default platform path.

## Exit Status

- Canonical Go toolchain source: complete
- Canonical Node toolchain source: complete
- Canonical frontend smoke/build entry: complete
- Canonical automatic CI hierarchy: complete
- Canonical backend compile drift fixed during validation
- Canonical E2E auth bootstrap fixture fixed during validation
- Optional specialized workflow normalization: deferred
