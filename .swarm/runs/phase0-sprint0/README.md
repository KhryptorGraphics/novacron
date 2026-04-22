# Phase 0 Sprint 0: Truth Reset

## Objective

Establish one authoritative repo-truth baseline for the convergence program, bootstrap `.swarm/runs/` conventions, and seed one Beads epic per phase so the remaining work can move through explicit phase ownership instead of ad hoc status documents.

## Deliverables

- `repo-truth-matrix.md`
- `architecture-map.md`
- `state.yaml`
- `phase-epics.md`
- `../../validation-matrix.yaml`

## Key Findings

- The planned canonical runtime exists at `backend/core/cmd/novacron/main.go`, but the root development workflow still starts `backend/cmd/api-server/main.go`.
- The repository currently exposes 31 Go modules, multiple Go build-tag variants, and more than one Node toolchain expectation.
- CI is duplicated across canonical, DWCP-specific, onboarding, integration, E2E, and production workflows with no single authoritative hierarchy.
- Deployment assets are split across `deployment/`, `deployments/`, `docker/`, root compose files, `k8s/`, `backend/deployments/k8s/`, `systemd/`, and `terraform/`.
- The local Beads rig is degraded because `bd` expects a Dolt-backed database that is not available in this checkout; phase epics were seeded directly into `.beads/issues.jsonl` as a compatibility fallback.

## Exit Status

- Repo truth matrix: complete
- Initial architecture map: complete
- Shared validation matrix: complete
- Phase epics: seeded
- Toolchain normalization: deferred to Sprint 1

## Next Sprint Entry Criteria

- Choose one supported Go toolchain and one supported Node toolchain.
- Collapse the CI hierarchy into one canonical path plus optional specialized lanes.
- Decide which entrypoints stay active, which become wrappers, and which move to archive or examples.
- Repair the local Beads rig with `bd init --prefix novacron --from-jsonl` once `dolt` is available.
