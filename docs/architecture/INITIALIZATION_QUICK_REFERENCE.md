# NovaCron Initialization Architecture - Quick Reference

**Version:** 2.0 | **Date:** 2025-11-10 | **Status:** Ready for Implementation

---

## 📊 At a Glance

| Metric | Value |
|--------|-------|
| **Boot Time Target** | 15-25 seconds |
| **Maximum Boot Time** | 30 seconds |
| **Parallel Speedup** | 2.8-4.4x vs sequential |
| **Initialization Phases** | 4 (Pre-Init, Core, Services, Post-Init) |
| **Component Count** | 12+ core components |
| **DWCP v3 Modules** | 6 (AMST, HDE, PBA, ASS, ACP, ITP) |

---

## 🔄 Initialization Phases

### Phase 1: Pre-Init (2-5s)
- ✅ Detect environment (datacenter/internet/hybrid)
- ✅ Load configuration (YAML/JSON)
- ✅ Initialize logger
- ✅ Validate resources (4 CPU, 8GB RAM, 100GB disk)

### Phase 2: Core Init (5-10s)
**Level 0 (Sequential):**
- SecurityComponent

**Level 1 (Parallel):**
- DatabaseComponent
- CacheComponent
- NetworkComponent

**Level 2 (Sequential):**
- DWCPComponent (AMST, HDE, PBA, ASS, ACP, ITP)

### Phase 3: Services (5-10s)
**All Parallel:**
- OrchestrationComponent
- APIServerComponent
- MonitoringComponent (optional)
- MLEngineComponent (optional)

### Phase 4: Post-Init (2-5s)
- System health check
- Emit startup metrics
- Service discovery registration
- Start background jobs
- Mark system READY

---

## 🧩 Component Dependencies

```
SecurityComponent [no deps]
  ├─→ DatabaseComponent
  ├─→ CacheComponent
  └─→ NetworkComponent
        └─→ DWCPComponent
              ├─→ OrchestrationComponent
              └─→ APIServerComponent
```

---

## ⚠️ Error Handling

| Component | Error Type | Action |
|-----------|------------|--------|
| Security, Database, Network, API | **CRITICAL** | Halt initialization |
| Cache | **DEGRADED** | Retry 3x, continue degraded |
| DWCP | **DEGRADED** | Use v1 fallback |
| Monitoring, ML Engine | **WARNING** | Log warning, continue |

**Retry Policy:** 1s → 2s → 4s (exponential backoff, max 3 attempts)

---

## 🎯 DWCP v3 Components

| Component | Full Name | Purpose |
|-----------|-----------|---------|
| **AMST** | Adaptive Multi-Stream Transport | Multi-stream TCP, RDMA, BBR congestion |
| **HDE** | Hybrid Data Encoding | Compression (zstd/lz4), delta encoding |
| **PBA** | Predictive Bandwidth Allocation | ML-based bandwidth prediction (LSTM) |
| **ASS** | Adaptive State Synchronization | Consensus (Raft/Gossip/PBFT), delta sync |
| **ACP** | Adaptive Congestion Prevention | BBR, CUBIC, adaptive algorithms |
| **ITP** | Intelligent Task Placement | ML-optimized task placement |

---

## 🌐 Network Mode Detection

| Mode | Indicators |
|------|-----------|
| **Datacenter** | Latency <1ms, Bandwidth >10Gbps, RDMA available |
| **Internet** | Latency >50ms, Variable bandwidth, Packet loss |
| **Hybrid** | Mixed conditions, Multi-DC with internet links |

---

## ⚙️ Configuration (novacron.yaml)

### Minimal Configuration
```yaml
system:
  node_id: "novacron-node-1"
  data_dir: "/var/lib/novacron"
  log_level: "info"

dwcp:
  enable_auto_detection: true
  v3_rollout_percentage: 0  # Start with v1

network:
  listen_address: "0.0.0.0"
  bind_port: 9090
  enable_tls: true

storage:
  backend: "sqlite"

security:
  enable_auth: true
  auth_method: "mtls"
```

### DWCP v3 Feature Flags
```yaml
dwcp:
  enable_v3_transport: false      # AMST v3
  enable_v3_compression: false    # HDE v3
  enable_v3_prediction: false     # PBA v3
  enable_v3_state_sync: false     # ASS v3
  enable_v3_consensus: false      # ACP v3
  enable_v3_placement: false      # ITP v3
  v3_rollout_percentage: 0        # 0-100
```

---

## 📈 Performance Targets

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| **CPU** | 4 cores | 8 cores |
| **Memory** | 8 GB | 16 GB |
| **Disk** | 100 GB | 500 GB |
| **Network** | 1 Gbps | 10 Gbps |

---

## 📊 Key Metrics

### Initialization Metrics
```
component_init_duration_seconds{component="security"}
component_init_success_count{component="database"}
phase_duration_seconds{phase="core_init"}
total_boot_time_seconds
parallel_efficiency_ratio
```

### Health Status
```
component_health_status{component="dwcp"} = 1  # healthy
component_health_status{component="cache"} = 0.5  # degraded
```

---

## 🔧 Implementation Files

### Core Framework
- `/backend/core/init/interfaces.go` - Component interfaces
- `/backend/core/init/registry.go` - Component registry
- `/backend/core/initialization/init.go` - Main initializer
- `/backend/core/initialization/orchestrator/orchestrator.go` - Orchestrator
- `/backend/core/initialization/config/loader.go` - Config loader

### DWCP v3 Components (To Implement)
- `/backend/core/network/dwcp/v3/amst.go` - AMST v3
- `/backend/core/network/dwcp/v3/hde.go` - HDE v3
- `/backend/core/network/dwcp/v3/pba.go` - PBA v3
- `/backend/core/network/dwcp/v3/ass.go` - ASS v3
- `/backend/core/network/dwcp/v3/acp.go` - ACP v3
- `/backend/core/network/dwcp/v3/itp.go` - ITP v3

---

## 🚀 Quick Start

### 1. Generate Default Config
```bash
go run main.go init --generate-config novacron.yaml
```

### 2. Start NovaCron
```bash
go run main.go start --config novacron.yaml
```

### 3. Check Health
```bash
curl http://localhost:8080/health
```

### 4. View Metrics
```bash
curl http://localhost:9091/metrics
```

---

## 🧪 Testing

### Unit Tests
```bash
go test ./backend/core/initialization/...
go test ./backend/core/init/...
```

### Integration Tests
```bash
go test -tags=integration ./tests/integration/...
```

### Performance Benchmarks
```bash
go test -bench=. ./tests/performance/...
```

---

## 📚 Documentation

- **Full Architecture:** [INITIALIZATION_ARCHITECTURE_DESIGN_V2.md](./INITIALIZATION_ARCHITECTURE_DESIGN_V2.md)
- **Original Architecture:** [INITIALIZATION_ARCHITECTURE.md](./INITIALIZATION_ARCHITECTURE.md)
- **DWCP v3 Architecture:** [/docs/DWCP_V3_ARCHITECTURE.md](../../DWCP_V3_ARCHITECTURE.md)
- **Dependency Graph:** [diagrams/initialization-dependency-graph.mermaid](./diagrams/initialization-dependency-graph.mermaid)
- **Sequence Diagram:** [diagrams/initialization-sequence.mermaid](./diagrams/initialization-sequence.mermaid)

---

## 🔍 Troubleshooting

### Initialization Fails
1. Check logs: `/var/lib/novacron/novacron.log`
2. Verify config: `novacron validate --config novacron.yaml`
3. Check resources: `df -h && free -h && nproc`

### Slow Boot Time
1. Check metrics: `curl http://localhost:9091/metrics | grep init_duration`
2. Review component logs for bottlenecks
3. Ensure parallel initialization is enabled

### Component Degraded
1. Check health: `curl http://localhost:8080/health/components`
2. Review component-specific logs
3. Verify dependencies are healthy

---

## 📝 ADRs (Architecture Decision Records)

| ADR | Decision | Rationale |
|-----|----------|-----------|
| **001** | Component-based architecture | Modularity, testability, parallel init |
| **002** | Four-phase initialization | Clear separation, easier debugging |
| **003** | Fail-fast for critical | Security and reliability paramount |
| **004** | Graceful degradation for non-critical | Maximize availability |
| **005** | Parallel initialization | 2.8-4.4x faster boot times |

---

## 🎯 Implementation Status

### ✅ Completed
- Core framework (interfaces, registry, orchestrator)
- Configuration loading
- Dependency resolution
- Parallel initialization logic
- Recovery and checkpointing

### ⏳ Pending
- Component implementations (Security, Database, Cache, Network)
- DWCP v3 components (AMST, HDE, PBA, ASS, ACP, ITP)
- Service components (Orchestration, API, Monitoring, ML)
- Environment detector
- Resource validator

---

**Need help?** See full documentation: [INITIALIZATION_ARCHITECTURE_DESIGN_V2.md](./INITIALIZATION_ARCHITECTURE_DESIGN_V2.md)
