# Architecture Map

This map captures the highest-signal execution facts needed to start runtime convergence.

## Canonical Target Versus Current Startup Path

- The convergence target is `backend/core/cmd/novacron/main.go`.
- The current root development workflow still uses `package.json` `start:api`, which points at `backend/cmd/api-server/main.go`.
- `cmd/novacron/main.go` remains a third launcher with direct VM wiring and static frontend serving.
- Root `main.go` is not a runtime entrypoint and should be treated as archive residue.

## GitNexus-Backed Runtime Seams

| Symbol | Evidence | Interpretation |
| --- | --- | --- |
| `backend/core/cmd/novacron/main.go:main` | GitNexus context shows calls to `initializeStorage`, `initializeVMManager`, `initializeMigrationManager`, `initializeNetwork`, `initializeHypervisor`, and `initializeAPI`; it participates in at least 11 indexed processes, including `Main -> StorageManager`, `Main -> VMManager`, `Main -> Scheduler`, and `Main -> Start` | This is the closest thing to the full WAN hypervisor fabric composition root |
| `backend/cmd/api-server/main.go:main` | GitNexus context shows `config.Load`, `initDatabase`, `NewSimpleAuthManager`, `initializeCanonicalServices`, and `buildCanonicalServer`; it participates in at least 8 indexed processes around config and database setup | This is a standalone API server that still boots outside the canonical daemon path |
| `backend/cmd/api-server/main_multicloud.go:main` | Build tag `novacron_multicloud` and direct multicloud or federation wiring indicate a separate runtime shape | Federation work is still expressed as a parallel binary instead of a canonical service seam |

## Entrypoint Competition

| Surface | Current role | Convergence action |
| --- | --- | --- |
| `backend/core/cmd/novacron/main.go` | Full-system daemon entrypoint | Keep and harden |
| `backend/cmd/api-server/main.go` | Default API runtime | Rewire under canonical runtime or wrap |
| `backend/cmd/api-server/main_*.go` | Tagged runtime variants | Convert to wrappers or archive |
| `cmd/novacron/main.go` | Legacy monolith | Archive or reduce to thin wrapper |
| `backend/cmd/core-server/main.go` | Auxiliary backend entrypoint | Keep only if it remains a distinct supported profile |
| `backend/core/edge/agent/main.go` | Edge agent | Keep as a separate supported role if required by product scope |
| `cli/cmd/novacron/main.go` | CLI | Keep as operator tooling |
| `k8s-operator/cmd/manager/main.go` | Operator manager | Keep as deployment tooling |
| `main.go` | Completion message | Archive immediately |

## Toolchain And Workflow Drift

- Root `go.mod` pins Go 1.24.0 with toolchain 1.24.6, while the inspected CI workflows still pin Go 1.21.
- Root `package.json` allows Node 18+, but the frontend uses Next 16.0.2 and the workflows split between Node 18 and Node 20.
- The repository presents npm at the root, npm in `frontend/`, and a `tools/indexer` package surface that carries `yarn.lock` plus pnpm metadata.
- CI is duplicated across canonical, production, onboarding, DWCP-specific, integration, E2E, and visual-regression lanes without one declared owner hierarchy.

## Static-Analysis Noise To Account For

- Codemap reports `masterdocs/testing.md` and `.claude/commands/swarm/testing.md` as top hub files with 366 importers each.
- That result is implausible for real runtime dependency flow and indicates the repo includes enough generated or doc-adjacent references to pollute broad import graphs.
- Convergence work should therefore use GitNexus context for symbol-level flow, direct repo inspection for build and ops truth, and a tighter exclusion list for future static-analysis passes.

## Immediate Convergence Priorities

1. Make `backend/core/cmd/novacron/main.go` the only authoritative startup and shutdown path for the shipping fabric.
2. Express API, federation, migration, discovery, backup, edge, and ML as services behind a shared runtime manifest instead of parallel binaries.
3. Collapse tagged API variants into wrappers or non-default profiles with explicit support status.
4. Remove or quarantine legacy launchers and celebratory completion artifacts that imply the repo is already finished.
5. Normalize CI, deployment, and package-manager paths before shared runtime work expands the blast radius further.
