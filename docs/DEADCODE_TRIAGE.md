# Dead-code triage

Inventory of the current working tree (Go 1.27.1, linux/arm64, default build tags). Counts are unreachable-function reports, not total functions. The root run was `go run golang.org/x/tools/cmd/deadcode@latest -filter 'novacron/backend/' ./backend/cmd/api-server ./backend/cmd/auth-test ./backend/cmd/loadtest`; the core run was the same tool with `-filter 'novacron/backend/core' ./cmd/...` from `backend/core`. Core/vm and Networks2-owned `backend/core/network/provision` and `backend/cmd/api-server` are excluded from this table.

No Go files or packages were deleted in this pass. A provisional source scan identified zero-importer package candidates, but standalone/test entry points, nested modules, generated/API surfaces, active work, and impact evidence were not consistently resolved; those candidates are kept. Whole-file candidates were checked for dual-run unreachable functions and cross-file references, but none was cleared for deletion after accounting for references/tests and the required GitNexus impact review. Existing unrelated working-tree changes were preserved.

| Package | Root unreachable | Core unreachable | Action | Reason |
|---|---:|---:|---|---|
| `backend/api/graphql` | 1 | 0 | Keep | Single unreachable method; no whole-file deletion proof. |
| `backend/api/security` | 1 | 0 | Keep | Single unreachable method; no whole-file deletion proof. |
| `backend/core/audit` | 39 | 0 | Keep | Partial dead functions; remaining declarations/API use not cleared. |
| `backend/core/auth` | 374 | 273 | Keep | Partial dead functions; test and declaration references remain. |
| `backend/core/cmd/novacron` | 0 | 1 | Keep | Executable entry point. |
| `backend/core/hypervisor` | 55 | 49 | Keep | Partial dead functions; declarations/API use not cleared. |
| `backend/core/monitoring` | 214 | 0 | Keep | Partial dead functions; tests and shared types reference candidate files. |
| `backend/core/monitoring/dashboard` | 30 | 0 | Keep | Partial dead functions; declarations/API use not cleared. |
| `backend/core/monitoring/ml_anomaly` | 71 | 0 | Keep | Partial dead functions; declarations/API use not cleared. |
| `backend/core/monitoring/prometheus` | 40 | 0 | Keep | Partial dead functions; declarations/API use not cleared. |
| `backend/core/monitoring/tracing` | 32 | 0 | Keep | Partial dead functions; declarations/API use not cleared. |
| `backend/core/network` | 0 | 135 | Keep | Networking subsystem/API; preserve, including active concurrent work. |
| `backend/core/network/ovs` | 0 | 59 | Keep | Networking subsystem/API; package-level deletion not approved by importer evidence alone. |
| `backend/core/network/topology` | 0 | 42 | Keep | Networking subsystem/API; preserve. |
| `backend/core/orchestration` | 6 | 0 | Keep | Mixed live package and active edits; partial dead functions only. |
| `backend/core/orchestration/autoscaling` | 14 | 0 | Keep | Partial dead functions; package has live consumers. |
| `backend/core/orchestration/events` | 21 | 0 | Keep | Live consumers in orchestration and api-server. |
| `backend/core/orchestration/healing` | 22 | 0 | Keep | Partial dead functions; tests/consumers remain. |
| `backend/core/scheduler` | 0 | 183 | Keep | Live scheduler package; partial dead functions only. |
| `backend/core/scheduler/migration` | 0 | 46 | Keep | Package/API has unresolved consumers; do not remove from deadcode output alone. |
| `backend/core/scheduler/workload` | 0 | 59 | Keep | The simulated `vm_metrics_collector.go` was deleted after reachability proof; remaining dead functions are partial package members, not proven safe to remove. |
| `backend/core/storage` | 131 | 160 | Keep | Partial dead functions; tests and declaration references remain. |
| `backend/core/storage/compression` | 7 | 7 | Keep | Tests reference candidate declarations. |
| `backend/core/storage/deduplication` | 14 | 14 | Keep | Tests reference candidate declarations. |
| `backend/core/storage/encryption` | 15 | 15 | Keep | Tests and sibling package code reference candidate declarations. |
| `backend/pkg/database` | 21 | 0 | Keep | Partial dead functions; package declarations remain in use. |
| `backend/pkg/logger` | 6 | 0 | Keep | Partial dead functions; package is shared infrastructure. |
| `backend/pkg/middleware` | 2 | 0 | Keep | Partial dead functions; shared middleware package. |
| `backend/pkg/services` | 51 | 0 | Keep | Partial dead functions; declarations/API use not cleared. |

The two runs reported 1,167 root and 1,043 core unreachable functions across the packages above after the stated exclusions. Deadcode is a call-graph signal; an unreachable function does not by itself establish that its file or package can be removed.