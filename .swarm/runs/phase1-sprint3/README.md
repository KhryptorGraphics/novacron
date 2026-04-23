# Phase 1 Sprint 3: Runtime Read Convergence

## Objective

Complete Phase 1 by finishing the last unsafe read split: converge canonical VM and network inventory reads on the canonical daemon path without pretending the in-memory managers are restart-stable sources of truth.

## Deliverables

- `state.yaml`
- `service-state-reporting.md`
- `runtime-inventory-read-cutover.md`

## Scope Of This Slice

- Keep the daemon service-state reporting work from the earlier Sprint 3 slice
- Add internal daemon VM, VM metrics, monitoring inventory, network, and VM interface read endpoints backed by the persisted inventory tables
- Cut canonical API GET routes for VMs, monitoring inventory, networks, and interface reads over to the daemon behind an env gate with SQL fallback
- Preserve existing response shapes and compatibility paths while making the daemon the canonical read entrypoint

## Explicit Non-Goals

- No VM or network write-through from the canonical API into the in-memory daemon managers
- No change to canonical API-server write paths
- No claim that the in-memory runtime managers are the durable source of truth after restart
- No federation, backup, edge, or ML startup wiring beyond the existing service-state work

## Exit Status

- Manifest-backed service-state reporting: complete
- Optional daemon service gating for current core subsystems: complete
- Explicit disabled-service reporting: complete
- VM and network canonical read source-of-truth convergence: complete
- Phase 1 exit criteria: complete
