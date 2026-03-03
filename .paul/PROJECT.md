# NovaCron

## What This Is

NovaCron is a distributed L2 hypervisor with P2P nodes + single master architecture. Nodes self-organize by latency to combine processing power and memory across the internet. Features LUN-based storage for efficient cross-nodal VM migration, live replication, and automatic failover ensuring zero data loss. Ubuntu 24.04 Server is provided as the starter OS.

## Core Value

Users run distributed VMs across internet-connected nodes that self-organize by latency, combining compute/memory with LUN-based storage for efficient cross-nodal migration, live replication, and automatic failover — ensuring zero data loss.

## Current State

| Attribute | Value |
|-----------|-------|
| Version | 0.1.0 |
| Status | Development |
| Last Updated | 2026-02-20 |

## Requirements

### Validated (Shipped)

- [x] Distributed consensus (Raft) — backend/core/consensus/
- [x] Live migration orchestrator — backend/core/migration/
- [x] Multi-tier caching (BigCache + Redis) — backend/core/cache/
- [x] ML-based predictive scaling — backend/core/ml/
- [x] KVM/libvirt hypervisor integration — backend/core/hypervisor/
- [x] Multi-cloud federation — backend/core/federation/
- [x] Backup system with CBT — backend/core/backup/
- [x] SDN/Network management — backend/core/network/
- [x] Monitoring system — backend/core/monitoring/
- [x] Auth system (JWT, OAuth2, 2FA, RBAC) — backend/core/auth/

### Active (In Progress)

- [ ] LUN-based storage system for cross-nodal migration
- [ ] Latency-based node clustering
- [ ] Automatic failover with zero data loss

### Planned (Next)

- [ ] Ubuntu 24.04 Server starter OS integration
- [ ] Cross-internet P2P node networking
- [ ] Single master coordination layer

### Out of Scope

- Container-only deployments (KVM is primary)
- On-premise only (designed for cross-internet)

## Target Users

**Primary:** Infrastructure Operators and DevOps Teams
- Managing distributed compute across multiple locations
- Need high availability without manual intervention
- Running VMs that cannot tolerate downtime

**Secondary:** Cloud Service Providers
- Offering distributed VM hosting
- Multi-tenant isolation requirements

## Context

**Business Context:**
Distributed computing platform competing with traditional hypervisors by offering cross-internet clustering, automatic failover, and combined resource pooling.

**Technical Context:**
- Backend: Go 1.24+ with multiple modules
- Frontend: Next.js 13.5
- Database: PostgreSQL
- Cache: Redis + BigCache
- Messaging: NATS
- Hypervisor: KVM/libvirt
- Monitoring: Prometheus, OpenTelemetry

## Constraints

### Technical Constraints
- Must run KVM/libvirt for VM management
- PostgreSQL required for persistence
- NATS required for orchestration events
- Redis required for distributed caching
- Go 1.24+ for backend

### Business Constraints
- Zero data loss guarantee for failover
- Cross-internet latency must be acceptable for node clustering
- Ubuntu 24.04 as primary guest OS

### Compliance Constraints
- Multi-tenant isolation required
- Audit logging for VM operations

## Key Decisions

| Decision | Rationale | Date | Status |
|----------|-----------|------|--------|
| Raft consensus for distributed state | Strong consistency, well-understood | 2025 | Active |
| LUN-based storage for VMs | Efficient cross-nodal migration | 2026 | Active |
| Latency-based node grouping | Optimize for network performance | 2026 | Active |
| Single master P2P architecture | Simplicity + fault tolerance balance | 2026 | Active |

## Success Metrics

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| VM Migration Time | < 5s cross-node | TBD | To Measure |
| Failover Recovery | < 30s automatic | TBD | To Measure |
| Node Cluster Efficiency | > 80% resource utilization | TBD | To Measure |
| Zero Data Loss | 100% consistency | TBD | To Measure |

## Tech Stack

| Layer | Technology | Notes |
|-------|------------|-------|
| Backend | Go 1.24+ | Multiple modules |
| Frontend | Next.js 13.5 | React 18 |
| Database | PostgreSQL | Multi-tenant |
| Cache | Redis + BigCache | L1/L2 tier |
| Messaging | NATS | Orchestration events |
| Hypervisor | KVM/libvirt | Primary driver |
| Monitoring | Prometheus, OpenTelemetry | Full observability |
| Auth | JWT, OAuth2, 2FA, RBAC | Zero-trust |

## Links

| Resource | URL |
|----------|-----|
| Repository | /home/kp/thordrive/novacron |
| API Server | :8090 |
| Frontend | :8092 |
| AI Engine | :8093 |

---
*PROJECT.md — Updated when requirements or context change*
*Last updated: 2026-02-20*