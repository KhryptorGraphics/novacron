# Convergence Phase Epics

These phase epics were seeded directly into `.beads/issues.jsonl` because the local `bd` CLI cannot currently open a working Dolt-backed rig in this checkout.

| Issue | Phase | Scope |
| --- | --- | --- |
| `NC-p0a` | Phase 0 | Truth reset, repo classification, toolchain normalization, and CI hierarchy cleanup |
| `NC-p1a` | Phase 1 | Canonical runtime convergence around `backend/core/cmd/novacron/main.go` |
| `NC-p2a` | Phase 2 | Discovery and federation convergence into one trusted control path |
| `NC-p3a` | Phase 3 | Mobility, storage, replication, backup, rollback, and recovery hardening |
| `NC-p4a` | Phase 4 | Frontend and API consolidation onto one typed client and real contracts |
| `NC-p5a` | Phase 5 | Deployment profile alignment, release hardening, and release-candidate gates |

## Follow-Up

- Repair the local Beads rig before the next sprint if issue mutation through `bd` is required.
- Treat these epics as the remaining-work tracker for the rest of the convergence program.
