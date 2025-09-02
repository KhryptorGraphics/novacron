# Legacy Hypervisor Integration - Brownfield Enhancement Epic

## Epic Goal

Enable NovaCron to manage legacy hypervisor infrastructure (VMware vSphere, Hyper-V, XenServer) alongside modern KVM/libvirt deployments, providing unified management across heterogeneous virtualization environments.

## Epic Description

### Existing System Context

- **Current relevant functionality**: KVM/libvirt primary hypervisor support via go-libvirt
- **Technology stack**: Go 1.23.0, factory pattern for drivers, event-driven architecture
- **Integration points**: `backend/core/hypervisor/`, `VMDriverFactory` interface pattern

### Enhancement Details

- **What's being added**: Multi-hypervisor support for VMware, Hyper-V, XenServer
- **How it integrates**: New driver implementations following existing factory pattern
- **Success criteria**: 
  - Manage 100+ legacy VMs per hypervisor type
  - Unified API across all hypervisor types
  - <5 second response time for operations
  - No regression in KVM performance

## Stories

### Story 1: VMware vSphere Driver Implementation
Implement VMware vSphere driver following existing VMDriverFactory pattern, supporting core VM operations (create, start, stop, delete, migrate) via govmomi SDK.

### Story 2: Hyper-V Driver Implementation  
Implement Hyper-V driver with WMI/PowerShell integration, maintaining compatibility with existing event-driven architecture and monitoring systems.

### Story 3: Unified Hypervisor Discovery and Registration
Create hypervisor auto-discovery service that identifies and registers legacy hypervisors, integrating with existing scheduler and resource management.

## Compatibility Requirements

- [x] Existing KVM APIs remain unchanged
- [x] Database schema changes are backward compatible (new hypervisor_type field)
- [x] UI components follow existing VMOperationsDashboard patterns
- [x] Performance impact minimal (<5% overhead)

## Risk Mitigation

- **Primary Risk**: Legacy API instability affecting core operations
- **Mitigation**: Circuit breaker pattern with fallback to read-only mode
- **Rollback Plan**: Feature flag to disable legacy drivers without affecting KVM

## Definition of Done

- [x] All three stories completed with acceptance criteria met
- [x] Existing KVM functionality verified through regression testing
- [x] Integration points working correctly with scheduler
- [x] Documentation updated in architecture docs
- [x] No regression in existing VM operations
- [x] Performance benchmarks show <5% impact

---

## Story Manager Handoff

"Please develop detailed user stories for this brownfield epic. Key considerations:

- This is an enhancement to an existing system running Go 1.23.0 with libvirt/KVM
- Integration points: `VMDriverFactory` in `backend/core/hypervisor/`, event listeners in `backend/core/vm/`
- Existing patterns to follow: Factory pattern for drivers, event-driven updates, Prometheus metrics
- Critical compatibility requirements: Maintain existing KVM API contracts, preserve WebSocket real-time updates
- Each story must include verification that existing KVM functionality remains intact

The epic should maintain system integrity while delivering unified multi-hypervisor management capabilities."

---
*Epic created using BMad Brownfield Create Epic Task*
*Date: 2025-01-30*
*Scope: 3 focused stories for incremental enhancement*