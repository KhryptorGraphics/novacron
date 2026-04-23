# Phase 1 Sprint 3 Slice: Manifest-Backed Daemon Service State

## Problem

The canonical daemon consumed the shared runtime manifest but mostly ignored `runtime.enabled_services`. Aside from runtime auth, optional subsystems still initialized unconditionally and there was no operator-visible report showing which services were actually active.

## What Changed

- `backend/core/cmd/novacron/main.go` now keeps a runtime manifest summary in the daemon config
- Optional subsystem startup is skipped when the manifest disables:
  - `storage`
  - `vm`
  - `scheduler`
  - `migration`
  - `network`
  - `hypervisor`
- `api` is treated as required for the canonical daemon entrypoint
- The runtime API exposes `GET /internal/runtime/v1/services`
- The service report returns:
  - overall `status`
  - manifest metadata
  - per-service `enabled`
  - per-service `state`
  - `disabled_services`

## State Model

- `running`: the service is enabled and a concrete runtime instance was initialized
- `disabled`: the service is off because of the runtime manifest or auth-mode settings
- `unavailable`: the service was expected to be enabled but no runtime instance is available

## Guardrails

- The daemon no longer requires a VM manager just to expose internal service state
- Cluster node and leader routes now degrade cleanly when `vm` is disabled instead of panicking on nil access
- Runtime auth tests keep using the same API flow; only the daemon startup metadata changed

## Validation

- `GOTOOLCHAIN=go1.24.6 go test ./cmd/novacron`
- `GOTOOLCHAIN=go1.24.6 go test ./cmd/novacron -run 'TestInitializeAPIReportsDisabledServicesWhenVMServiceIsOff|TestInitializeAPIExposesClusterLocalEndpoints|TestInitializeAPIExposesInternalRuntimeMonitoringMetrics|TestRuntimeAuth(RegisterLoginAndAdmissionFlow|TestRuntimeAuthRefreshLogoutAndClusterSelectionFlow|TestRuntimeAuthGitHubAuthorizationURLRequiresConfiguration|TestRuntimeAuthGitHubAuthorizationURLConfigured)' -count=1`

## Deferred

- Canonical API read cutover for VM, monitoring inventory, and network routes
- Deeper dependency enforcement between all manifest-declared aspirational services
- Federation, backup, edge, and ML runtime service-state wiring
