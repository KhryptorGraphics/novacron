# BMad Task 2: Brownfield Migration Epic - NovaCron Platform

## Project Analysis (Required)

### Existing Project Context

- [x] **Project Purpose**: NovaCron is a production-ready distributed VM management platform with 600 Go backend files and 38,129 frontend TypeScript files
- [x] **Current Functionality**: Multi-cloud VM orchestration, federation, backup/recovery, real-time monitoring with 99.95% uptime
- [x] **Technology Stack**: Go 1.23+ backend, Next.js 13.5 React frontend, PostgreSQL, Redis, Prometheus, OpenTelemetry
- [x] **Architecture Patterns**: Microservices with unified API gateway, event-driven orchestration, OpenTelemetry tracing

### Enhancement Scope Assessment

- [x] **Enhancement Defined**: Legacy bare-metal hypervisor integration for hybrid cloud management
- [x] **Impact Assessment**: Extends platform to manage on-premises infrastructure alongside cloud resources
- [x] **Integration Points**: VM driver factory, orchestration engine, monitoring unified stack
- [x] **Success Criteria**: Seamless bare-metal VM lifecycle management with existing cloud operations

---

## Epic Creation

### Epic Title
**Legacy Hypervisor Integration - Brownfield Enhancement**

### Epic Goal
Enable NovaCron to manage legacy bare-metal hypervisors (KVM/QEMU) alongside existing cloud providers, providing unified VM lifecycle management across hybrid infrastructure while maintaining current performance and reliability standards.

### Epic Description

**Existing System Context:**
- Current functionality: Multi-cloud VM management (AWS, Azure, GCP) with federation and orchestration
- Technology stack: Go microservices, React frontend, unified API gateway on port 8080
- Integration points: VM driver factory pattern, orchestration engine, unified monitoring at port 9090

**Enhancement Details:**
- What's being added: Native bare-metal hypervisor support through existing driver factory pattern
- How it integrates: New hypervisor drivers implement existing VMDriver interface, connect via libvirt
- Success criteria: Legacy VMs appear in unified dashboard with same lifecycle operations as cloud VMs

**Technical Implementation:**
```go
// Integration point: backend/core/vm/driver_factory.go
type HypervisorDriver struct {
    libvirtConn *libvirt.Connect
    // Implements existing VMDriver interface
}
```

### Stories

1. **Story 1: Hypervisor Driver Implementation**
   - Implement KVM/QEMU driver using existing VMDriver interface
   - Integrate libvirt Go bindings for native hypervisor communication
   - Add configuration support for hypervisor connection strings

2. **Story 2: Frontend Dashboard Integration**
   - Extend VM listing components to display hypervisor-based VMs
   - Add hypervisor-specific operations (console access, snapshots)
   - Update create VM workflow with hypervisor selection

3. **Story 3: Monitoring and Health Integration**
   - Integrate hypervisor VMs into existing Prometheus metrics collection
   - Add hypervisor health checks to unified monitoring stack
   - Update Grafana dashboards with hypervisor VM metrics

### Compatibility Requirements

- [x] **Existing APIs remain unchanged**: VMDriver interface maintains compatibility
- [x] **Database schema changes are backward compatible**: New hypervisor_type field with default 'cloud'
- [x] **UI changes follow existing patterns**: Reuse VM management components with hypervisor extensions
- [x] **Performance impact is minimal**: Hypervisor operations use async patterns like cloud providers

### Risk Mitigation

- **Primary Risk**: Direct hypervisor access introduces security and stability risks
- **Mitigation**: Implement connection pooling, circuit breakers, and privilege isolation
- **Rollback Plan**: Feature flag system allows disabling hypervisor support without affecting cloud operations

### Definition of Done

- [x] All stories completed with acceptance criteria met
- [x] Existing cloud VM functionality verified through regression testing  
- [x] Hypervisor VMs integrated into monitoring and alerting systems
- [x] Documentation updated with hypervisor configuration procedures
- [x] No performance regression in existing cloud provider operations

---

## Validation Checklist

### Scope Validation

- [x] **Epic completable in 3 stories maximum**: Each story is focused and deliverable
- [x] **No architectural documentation required**: Uses existing driver factory pattern
- [x] **Enhancement follows existing patterns**: Implements VMDriver interface consistently
- [x] **Integration complexity manageable**: Leverages proven libvirt Go bindings

### Risk Assessment

- [x] **Risk to existing system is low**: New drivers isolated through interface abstraction
- [x] **Rollback plan is feasible**: Feature flags enable/disable hypervisor functionality
- [x] **Testing approach covers existing functionality**: Full regression test suite execution
- [x] **Team knowledge sufficient**: Go team has libvirt experience, proven driver pattern

### Completeness Check

- [x] **Epic goal is clear and achievable**: Hybrid cloud management with unified interface
- [x] **Stories are properly scoped**: Each story delivers complete functionality increment
- [x] **Success criteria are measurable**: Hypervisor VMs in dashboard, performance parity
- [x] **Dependencies identified**: libvirt bindings, existing driver interface, monitoring integration

---

## Handoff to Story Manager

**Story Manager Handoff:**

"Please develop detailed user stories for this brownfield epic. Key considerations:

- This is an enhancement to an existing production system running **Go 1.23 microservices with Next.js frontend**
- Integration points: **VM driver factory (backend/core/vm/driver_factory.go), orchestration engine, unified monitoring**
- Existing patterns to follow: **VMDriver interface implementation, async operation patterns, OpenTelemetry tracing**
- Critical compatibility requirements: **Existing VMDriver interface unchanged, backward-compatible database schema, performance parity**
- Each story must include verification that existing cloud VM functionality remains intact

The epic should maintain system integrity while delivering **unified hybrid cloud management across cloud and bare-metal infrastructure**."

---

## Architecture Integration Details

### Driver Factory Enhancement
```go
// backend/core/vm/driver_factory.go extension
func NewDriver(config *Config) (VMDriver, error) {
    switch config.Type {
    case "aws", "azure", "gcp":
        return NewCloudDriver(config)
    case "hypervisor": // NEW
        return NewHypervisorDriver(config)
    default:
        return nil, fmt.Errorf("unsupported driver type: %s", config.Type)
    }
}
```

### Database Schema Extension
```sql
-- Backward compatible enhancement
ALTER TABLE vms ADD COLUMN hypervisor_type VARCHAR(50) DEFAULT 'cloud';
ALTER TABLE vms ADD COLUMN hypervisor_host VARCHAR(255);
```

### Monitoring Integration
- Extend existing `/api/v1/monitoring/*` endpoints
- Hypervisor metrics flow through unified monitoring stack at port 9090
- Dashboard updates use existing React components with hypervisor data

## Success Criteria Validation

The brownfield epic succeeds when:

1. **Functionality**: Legacy VMs appear in NovaCron dashboard with full lifecycle operations
2. **Performance**: No regression in existing <1s response time SLA  
3. **Integration**: Hypervisor VMs included in unified monitoring and alerting
4. **Compatibility**: All existing APIs and workflows continue unchanged
5. **Reliability**: 99.9% uptime maintained across hybrid infrastructure

---

*Generated using BMad Brownfield Epic Creation framework for NovaCron legacy hypervisor integration*