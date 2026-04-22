# NovaCron Run Memory

This directory is the file-backed control plane for the NovaCron convergence program. If a sprint artifact here disagrees with a status or completion report elsewhere in the repository, trust the run memory until the sprint explicitly promotes a replacement.

## Required Layout

- `README.md`: sprint brief, objective, scope, exits, and next sprint handoff.
- `state.yaml`: owners, blockers, gate status, and the artifact manifest.
- `repo-truth-matrix.md`: current repo truth for entrypoints, modules, workflows, deployments, tests, and authority rules.
- `architecture-map.md`: GitNexus-backed execution map and convergence notes.
- Optional subdirectories:
  - `research/`: source-backed memos for discovery gaps.
  - `manifests/`: generated inventories, snapshots, and scope ledgers.
  - `validation/`: sprint-specific evidence when the shared validation matrix is not enough.

## Authority Rules

- Code and active build files beat prose.
- `.swarm/runs/<phase-sprint>/repo-truth-matrix.md` beats root-level completion and status reports.
- `.swarm/validation-matrix.yaml` is the shared gate contract unless a sprint explicitly overrides it.
- Root documents matching `*_SUMMARY.md`, `*_REPORT.md`, `*_COMPLETE*.md`, `FINAL*.md`, `PHASE*.md`, `WEEKS_*.md`, `*_PROMPT.md`, `PROJECT_*`, and similar celebratory status files are non-authoritative unless a current sprint cites them as evidence.
- Generated artifacts such as `frontend/.next`, `frontend/coverage`, `coverage`, `logs`, `output`, and `node_modules` are never planning authority.

## Naming

- Sprint directories use `phaseN-sprintM`, for example `phase0-sprint0` and `phase2-sprint5`.
- Shared cross-sprint artifacts live directly under `.swarm/`.

## Workflow

1. Refresh repo truth before implementation.
2. Record blockers and scope in `state.yaml`.
3. Seed or update Beads issues for remaining work.
4. Run gate checks from `.swarm/validation-matrix.yaml`.
5. Hand off with explicit next-sprint entry criteria.
