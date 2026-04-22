# Sprint 2 Shared Runtime Manifest Baseline

## Canonical Contract

The shared manifest now lives in `backend/core/initialization/config/loader.go` under the top-level `runtime:` block.

```yaml
runtime:
  version: v1alpha1
  deployment_profile: single-node
  discovery_mode: disabled
  federation_mode: disabled
  migration_mode: disabled
  auth_mode: runtime
  storage_classes:
    - default
  enabled_services:
    - api
    - auth
    - hypervisor
    - network
    - scheduler
    - storage
    - vm
```

## EntryPoint Behavior

- `backend/core/cmd/novacron/main.go` now detects the shared manifest shape via `system:` or `runtime:` and adapts it into the existing daemon runtime config.
- The same daemon still accepts the previous daemon-only YAML shape and preserves the existing tests and config semantics for that path.
- `backend/pkg/config/config.go` now loads an optional shared manifest summary when `NOVACRON_RUNTIME_MANIFEST_PATH` is set.
- `backend/cmd/api-server/main.go` exposes that summary on `/api/info` and marks the runtime-manifest check on `/health`.

## Compatibility Rules

- Existing env-only API-server callers do not need to set `NOVACRON_RUNTIME_MANIFEST_PATH`.
- Existing daemon configs with only `storage`, `hypervisor`, `vm_manager`, `scheduler`, and `auth` still load through the legacy fallback.
- Requiring the manifest is opt-in through `NOVACRON_REQUIRE_RUNTIME_MANIFEST=true`.

## Deferred Convergence

- Runtime-backed `/api/v1/vms*` reads
- Read/write-path ownership reconciliation between SQL-backed handlers and live runtime managers
- Real service gating beyond runtime auth
- Public port consolidation between the core daemon and the canonical API server
