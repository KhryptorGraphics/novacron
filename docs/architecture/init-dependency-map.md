# NovaCron Initialization Dependency Map

**Document Type**: Technical Reference
**Version**: 1.0
**Date**: 2025-11-14

---

## Overview

This document provides a detailed dependency map for all NovaCron components, showing initialization order, dependencies, and parallel execution opportunities.

---

## Dependency Graph Notation

```
Component
├─ Dependency 1 (required)
├─ Dependency 2 (required)
└─ Dependency 3 (optional)

[Component Type]
  Critical: Must succeed for system to function
  Optional: System can function with degraded capability
  Deferred: Can initialize after system ready
```

---

## Complete Dependency Tree

```
Level 0: Bootstrap
┌─────────────────────────────────────────────────────────┐
│ Pre-flight Checks                                        │
│ ├─ Runtime version validation                           │
│ ├─ Directory structure                                   │
│ ├─ Memory availability                                   │
│ └─ Kernel capabilities                                   │
└─────────────────────────────────────────────────────────┘
        │
        ▼
Level 1: Configuration & Logging
┌─────────────────────────────────────────────────────────┐
│ Configuration Loader [Critical]                          │
│ ├─ Load default config                                   │
│ ├─ Load environment config                               │
│ ├─ Apply env var overrides                               │
│ └─ Validate schema                                        │
└─────────────────────────────────────────────────────────┘
        │
        ├─────────────────────────────────────────────────┐
        │                                                  │
        ▼                                                  ▼
┌──────────────────────┐                    ┌──────────────────────┐
│ Logger [Critical]    │                    │ Metrics [Critical]   │
│ ├─ Config            │                    │ ├─ Config            │
│ └─ File system       │                    │ └─ Logger            │
└──────────────────────┘                    └──────────────────────┘
        │                                                  │
        └──────────────────┬───────────────────────────────┘
                          ▼
Level 2: Data Layer
┌─────────────────────────────────────────────────────────┐
│ PostgreSQL Connection [Critical]                         │
│ ├─ Config                                                │
│ ├─ Logger                                                │
│ └─ Network connectivity                                  │
└─────────────────────────────────────────────────────────┘
        │
        ├─────────────────────────────────────────────────┐
        │                                                  │
        ▼                                                  ▼
┌──────────────────────┐                    ┌──────────────────────┐
│ Database Migrations  │                    │ Redis Cache          │
│ [Critical]           │                    │ [Optional]           │
│ ├─ PostgreSQL conn   │                    │ ├─ Config            │
│ └─ Migration files   │                    │ └─ Logger            │
└──────────────────────┘                    └──────────────────────┘
        │                                                  │
        └──────────────────┬───────────────────────────────┘
                          ▼
Level 3: Security Layer
┌─────────────────────────────────────────────────────────┐
│ Security Manager [Critical]                              │
│ ├─ Database connection                                   │
│ ├─ Logger                                                │
│ └─ Config                                                │
│                                                          │
│ Components:                                              │
│ ├─ Encryption Manager                                    │
│ │  ├─ Load/generate master key                          │
│ │  ├─ Initialize AES-256-GCM                            │
│ │  └─ Setup key rotation                                │
│ │                                                        │
│ ├─ Audit Logger                                          │
│ │  ├─ Database connection                               │
│ │  └─ Log signing keys                                  │
│ │                                                        │
│ └─ Zero-Trust Manager [Optional]                         │
│    ├─ Encryption manager                                │
│    ├─ Audit logger                                      │
│    └─ Policy configuration                              │
└─────────────────────────────────────────────────────────┘
        │
        ▼
Level 4: Core Infrastructure (Parallel Initialization)
┌─────────────────────┬─────────────────────┬──────────────────────┐
│                     │                     │                      │
▼                     ▼                     ▼                      │
┌──────────────────┐ ┌──────────────────┐ ┌───────────────────┐  │
│ Network Layer    │ │ VM Manager       │ │ DWCP Manager      │  │
│ [Critical]       │ │ [Critical]       │ │ [Critical]        │  │
│ ├─ Security      │ │ ├─ Security      │ │ ├─ Security       │  │
│ ├─ Config        │ │ ├─ Database      │ │ ├─ Network Layer  │  │
│ └─ Logger        │ │ ├─ Config        │ │ └─ Config         │  │
│                  │ │ └─ Logger        │ │                   │  │
│ Components:      │ │                  │ │ Phase 0:          │  │
│ ├─ OVS Bridge    │ │ Components:      │ │ ├─ Transport     │  │
│ ├─ IPsec         │ │ ├─ KVM Driver    │ │ │  (AMST/TCP)    │  │
│ ├─ Firewall      │ │ ├─ QEMU Driver   │ │ └─ Compression   │  │
│ └─ Load Balancer │ │ └─ Scheduler     │ │    (HDE/Zstd)    │  │
└──────────────────┘ └──────────────────┘ └───────────────────┘  │
        │                     │                     │              │
        └─────────────────────┴─────────────────────┴──────────────┘
                              │
                              ▼
Level 5: Optional Services (Parallel Initialization)
┌─────────────────────┬─────────────────────┬──────────────────────┐
│                     │                     │                      │
▼                     ▼                     ▼                      │
┌──────────────────┐ ┌──────────────────┐ ┌───────────────────┐  │
│ DWCP Phases 1-3  │ │ ML Services      │ │ Agent Spawner     │  │
│ [Optional]       │ │ [Optional]       │ │ [Optional]        │  │
│ ├─ DWCP Phase 0  │ │ ├─ Database      │ │ ├─ API Gateway    │  │
│ └─ Config        │ │ ├─ API Gateway   │ │ └─ MCP Integration│  │
│                  │ │ └─ ML Models     │ │                   │  │
│ Phase 1:         │ │                  │ │ Components:       │  │
│ ├─ Prediction    │ │ Components:      │ │ ├─ Smart Spawner │  │
│ │  (LSTM ML)     │ │ ├─ Task Class.   │ │ └─ Orchestrator  │  │
│                  │ │ ├─ Training Data │ │                   │  │
│ Phase 2:         │ │ └─ Model Serving │ │                   │  │
│ ├─ Sync Layer    │ │                  │ │                   │  │
│ └─ Consensus     │ │                  │ │                   │  │
│    (Raft/Gossip) │ │                  │ │                   │  │
│                  │ │                  │ │                   │  │
│ Phase 3:         │ │                  │ │                   │  │
│ └─ Resilience    │ │                  │ │                   │  │
│    (Circuit Br.) │ │                  │ │                   │  │
└──────────────────┘ └──────────────────┘ └───────────────────┘  │
        │                     │                     │              │
        └─────────────────────┴─────────────────────┴──────────────┘
                              │
                              ▼
Level 6: Application Layer
┌─────────────────────────────────────────────────────────┐
│ API Gateway [Critical]                                   │
│ ├─ All core components                                   │
│ ├─ Security manager                                      │
│ └─ Database                                              │
│                                                          │
│ Routes:                                                  │
│ ├─ VM Management API                                     │
│ ├─ User Authentication                                   │
│ ├─ Admin API                                             │
│ ├─ Monitoring API                                        │
│ └─ WebSocket API                                         │
└─────────────────────────────────────────────────────────┘
        │
        ▼
Level 7: Health & Ready
┌─────────────────────────────────────────────────────────┐
│ Health Check System [Critical]                           │
│ ├─ All components initialized                            │
│ └─ Health endpoints active                               │
│                                                          │
│ Endpoints:                                               │
│ ├─ /health/live (Liveness probe)                        │
│ ├─ /health/ready (Readiness probe)                      │
│ └─ /health/status (Detailed status)                     │
└─────────────────────────────────────────────────────────┘
```

---

## Component Initialization Order

### Sequential Initialization (18-47 seconds total)

| Order | Component | Dependencies | Duration | Critical | Can Parallelize |
|-------|-----------|--------------|----------|----------|-----------------|
| 1 | Pre-flight Checks | None | 0-2s | Yes | No |
| 2 | Configuration Loader | Pre-flight | 1-2s | Yes | No |
| 3 | Logger | Config | 0.5-1s | Yes | No |
| 4 | Metrics | Config, Logger | 0.5-1s | Yes | No |
| 5 | PostgreSQL | Config, Logger | 2-5s | Yes | No |
| 6 | Database Migrations | PostgreSQL | 1-3s | Yes | No |
| 7 | Redis | Config, Logger | 1-2s | No | With PostgreSQL |
| 8 | Security Manager | Database, Logger | 2-4s | Yes | No |
| 9 | Network Layer | Security, Config | 2-3s | Yes | With VM/DWCP |
| 10 | VM Manager | Security, Database | 3-5s | Yes | With Network/DWCP |
| 11 | DWCP Phase 0 | Security, Network | 2-4s | Yes | With Network/VM |
| 12 | DWCP Phases 1-3 | DWCP Phase 0 | 3-8s | No | With ML/Agent |
| 13 | ML Services | Database, API | 2-5s | No | With DWCP/Agent |
| 14 | Agent Spawner | API, MCP | 2-4s | No | With DWCP/ML |
| 15 | API Gateway | All Core | 1-2s | Yes | No |
| 16 | Health Checks | All | 1-2s | Yes | No |

### Parallel Execution Groups

**Group 1: Foundation (Sequential)**
```
1. Pre-flight → Config → Logger → Metrics
Total: ~4-6 seconds
```

**Group 2: Data Layer (Parallel)**
```
PostgreSQL (with migrations) || Redis
Total: ~3-5 seconds (max of both)
```

**Group 3: Security (Sequential)**
```
Security Manager (depends on database)
Total: ~2-4 seconds
```

**Group 4: Core Infrastructure (Parallel)**
```
Network Layer || VM Manager || DWCP Phase 0
Total: ~3-5 seconds (max of three)
```

**Group 5: Optional Services (Parallel)**
```
DWCP Phases 1-3 || ML Services || Agent Spawner
Total: ~3-8 seconds (max of three)
```

**Group 6: Application (Sequential)**
```
API Gateway → Health Checks
Total: ~2-4 seconds
```

**Total with Parallelization: ~18-32 seconds**

---

## Component Dependency Matrix

| Component | Config | Logger | Metrics | DB | Security | Network | VM | DWCP | API |
|-----------|--------|--------|---------|----|----|---------|-------|------|-----|
| Config | - | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Logger | ✓ | - | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Metrics | ✓ | ✓ | - | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Database | ✓ | ✓ | ✗ | - | ✗ | ✗ | ✗ | ✗ | ✗ |
| Security | ✓ | ✓ | ✗ | ✓ | - | ✗ | ✗ | ✗ | ✗ |
| Network | ✓ | ✓ | ✗ | ✗ | ✓ | - | ✗ | ✗ | ✗ |
| VM Manager | ✓ | ✓ | ✗ | ✓ | ✓ | ✗ | - | ✗ | ✗ |
| DWCP | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ | ✗ | - | ✗ |
| ML Services | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ○ |
| Agent Spawner | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ○ |
| API Gateway | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ○ | - |

**Legend**:
- ✓ = Required dependency
- ○ = Optional dependency
- ✗ = No dependency

---

## Dependency Cycle Detection

### Validation Rules

1. **No Circular Dependencies**: DAG (Directed Acyclic Graph) validation
2. **Optional Dependencies**: Can fail without blocking critical path
3. **Critical Path**: Config → Logger → Database → Security → Core → API

### Cycle Detection Algorithm

```go
type DependencyGraph struct {
    nodes map[string]*Node
}

type Node struct {
    name         string
    dependencies []string
    optional     []string
}

func (g *DependencyGraph) DetectCycles() ([][]string, error) {
    visited := make(map[string]bool)
    recStack := make(map[string]bool)
    cycles := make([][]string, 0)

    for name := range g.nodes {
        if !visited[name] {
            if cycle := g.dfs(name, visited, recStack, []string{}); cycle != nil {
                cycles = append(cycles, cycle)
            }
        }
    }

    if len(cycles) > 0 {
        return cycles, fmt.Errorf("circular dependencies detected")
    }

    return nil, nil
}
```

---

## Failure Impact Analysis

### Critical Component Failures

| Component | Impact | Recovery | Degradation |
|-----------|--------|----------|-------------|
| Config | System cannot start | Exit with error | N/A |
| Logger | Degraded logging | Fall back to console | Continue |
| Database | Core functions unavailable | Exit with error | N/A |
| Security | Security breach risk | Exit with error | N/A |
| Network | VM networking unavailable | Exit with error | N/A |
| VM Manager | Cannot manage VMs | Exit with error | N/A |
| DWCP Phase 0 | Standard TCP fallback | Continue degraded | Slower network |

### Optional Component Failures

| Component | Impact | Recovery | Degradation |
|-----------|--------|----------|-------------|
| Redis | No caching | Continue | Slower responses |
| DWCP Phases 1-3 | No ML/consensus | Continue | Standard DWCP |
| ML Services | No task classification | Continue | Manual classification |
| Agent Spawner | No auto-spawning | Continue | Manual spawning |

---

## Initialization Timeline

```
Time (s)  Node.js Frontend          Go Backend              Status
─────────────────────────────────────────────────────────────────────
0         Pre-flight checks    ││   Pre-flight checks      Starting
          ↓                    ││   ↓
2         Config loaded        ││   Config loaded          Config OK
          ↓                    ││   ↓
3         Logger initialized   ││   Zap logger init        Logging OK
          ↓                    ││   ↓
5         PostgreSQL connected ││   PostgreSQL connected   Database OK
          Redis connected      ││
          ↓                    ││   ↓
9         Core services init   ││   Security initialized   Security OK
          ↓                    ││   ↓
          ↓                    ││   Network layer init     Network OK
          ↓                    ││   VM Manager init        VM OK
14        ↓                    ││   DWCP Phase 0 init      DWCP OK
          ↓                    ││   ↓
          Optional services    ││   DWCP Phases 1-3        Optional
          ↓                    ││   ↓
22        Health validated     ││   Health validated       Ready
          ↓                    ││   ↓
          READY ✓              ││   READY ✓                SERVING
─────────────────────────────────────────────────────────────────────
```

---

## Document Control

**Version**: 1.0
**Author**: System Architecture Designer
**Date**: 2025-11-14
