# Phase 1 Sprint 3: Daemon Service-State Reporting

## Objective

Make the canonical daemon report which core services are running, disabled, or unavailable based on the shared runtime manifest and actual startup results.

## Deliverables

- `state.yaml`
- `service-state-reporting.md`

## Scope Of This Slice

- Track the daemon runtime manifest summary inside `backend/core/cmd/novacron/main.go`
- Gate optional startup for `storage`, `vm`, `scheduler`, `migration`, `network`, and `hypervisor` when `runtime.enabled_services` disables them
- Keep `api` as a required service for the canonical daemon entrypoint
- Expose `GET /internal/runtime/v1/services` with manifest metadata plus per-service `enabled` and `state`
- Allow the runtime API to boot with `vm` disabled and report explicit disabled-service state instead of failing at startup

## Explicit Non-Goals

- No VM or network source-of-truth convergence for canonical read cutover
- No change to canonical API-server write paths
- No deep dependency orchestration across every manifest-declared aspirational service
- No federation, backup, edge, or ML startup wiring beyond status reporting

## Exit Status

- Manifest-backed service-state reporting: complete
- Optional daemon service gating for current core subsystems: complete
- Explicit disabled-service reporting: complete
- VM and network canonical read source-of-truth convergence: deferred to `novacron-0ex`
