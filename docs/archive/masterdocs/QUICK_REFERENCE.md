# DWCP Modules - Quick Reference Guide

## Module Status Summary

| Module | Phase | Status | LOC Needed | Time Est. | Complexity | Test Coverage |
|--------|-------|--------|-----------|-----------|------------|---|
| **Transport (AMST)** | 0-1 | ✅ 95% | 472 | Complete | Low | 85% (8 unit, 2 bench) |
| **Compression (HDE)** | 0-1 | ✅ 95% | 408 | Complete | Low | 90% (9 unit, 2 bench) |
| **Prediction (PBA)** | 2 | ❌ 0% | 2000-3000 | 2-3 weeks | **HIGH** | 0% |
| **Partition (ITP)** | 2 | ❌ 0% | 1500-2000 | 1.5-2 weeks | MEDIUM | 0% |
| **Sync (ASS)** | 3 | ❌ 0% | 2000-2500 | 2-3 weeks | **MED-HIGH** | 0% |
| **Consensus (ACP)** | 3 | ❌ 0% | 1500-2000 | 1.5-2 weeks | MEDIUM | 0% |

**Legend**: ✅ = Implemented | ❌ = Not Implemented

---

## Phase 2 Modules (Currently Needed)

### 1. Prediction Module (PBA)
**File**: `prediction/bandwidth_predictor.go`

**What it does**: ML-based bandwidth prediction using LSTM

**Key Stats**:
- Input: 20 historical/contextual features
- Output: [bandwidth, confidence_lower, confidence_upper]
- Horizon: 5-60 seconds ahead
- Success criterion: ≥ 70% accuracy (30s)

**Missing files**:
```
prediction/
  ├── bandwidth_predictor.go (main interface)
  ├── lstm_model.go (LSTM implementation)
  ├── feature_extractor.go (feature engineering)
  ├── model_training.go (training pipeline)
  ├── bandwidth_predictor_test.go (tests)
  └── trained_model.pb (serialized model)
```

**Blocker**: ML framework selection (TensorFlow/ONNX/Go native)

---

### 2. Partition Module (ITP)
**File**: `partition/task_partitioner.go`

**What it does**: Minimizes cross-WAN traffic via smart task placement

**Key Algorithms**:
- Strongly Connected Components (SCC) - Tarjan/Kosaraju
- Recursive Bisection (RCB) - intra-region load balancing
- CCR/LIF metrics calculation

**Success metrics**:
- Cross-WAN traffic reduction: ≥ 40%
- Load imbalance factor: < 5%
- CCR target: < 0.10

**Missing files**:
```
partition/
  ├── task_partitioner.go (main orchestrator)
  ├── graph_analyzer.go (SCC, dependency analysis)
  ├── load_balancer.go (RCB algorithm)
  ├── partition_metrics.go (CCR/LIF calculation)
  ├── rebalancer.go (dynamic rebalancing)
  └── task_partitioner_test.go (tests)
```

**Blocker**: Task dependency graph interface definition

---

## Phase 3 Modules (After Phase 2)

### 3. Sync Module (ASS)
**File**: `sync/async_state_sync.go`

**What it does**: Distributed state synchronization with eventual consistency

**Key Features**:
- Bounded staleness model (configurable)
- Vector clocks for causality
- Last-Write-Wins conflict resolution
- Periodic reconciliation (5 min)

**Guarantees**:
- Read-Your-Writes: ✅
- Monotonic Reads: ✅
- Eventual Consistency: ✅ (within 2×max_staleness)

**Missing files**:
```
sync/
  ├── async_state_sync.go (core engine)
  ├── vector_clock.go (causality tracking)
  ├── conflict_resolver.go (conflict logic)
  ├── state_store.go (versioned storage)
  ├── reconciliation_engine.go (periodic sync)
  └── async_state_sync_test.go (tests)
```

**Blocker**: State schema definition, consensus integration

---

### 4. Consensus Module (ACP)
**File**: `consensus/adaptive_consensus.go`

**What it does**: Scope-aware consensus selection (Raft/Gossip/CRDT)

**Scope Mapping**:
```
LOCAL    → Raft (strong consistency)
REGIONAL → Multi-Raft (regional quorum)
GLOBAL   → Gossip + CRDT (eventual consistency)
```

**Success criteria**:
- Consensus overhead: < 5% of communication
- Availability during partitions: ✅
- State convergence: within 2×max_staleness

**Missing files**:
```
consensus/
  ├── adaptive_consensus.go (scope selector)
  ├── hybrid_raft_gossip.go (protocol orchestrator)
  ├── crdt_manager.go (Counter, Set, Register)
  ├── quorum_manager.go (region-aware quorum)
  ├── consensus_orchestrator.go (operation router)
  └── adaptive_consensus_test.go (chaos tests)
```

**Blocker**: CRDT library selection, scope definition

---

## Integration Checklist

### Config Changes Needed
- [ ] Add `PartitionConfig` to main `Config` struct
- [ ] Update `DefaultConfig()` with Partition defaults
- [ ] Add Partition validation in `Config.Validate()`

### Manager Changes Needed
- [ ] Uncomment Phase 2 initialization (lines 96-99)
- [ ] Implement `metricsCollectionLoop()` for new modules
- [ ] Add health checks for Prediction/Partition (Phase 2)
- [ ] Add health checks for Sync/Consensus (Phase 3)

### Manager Method TODOs
**File**: `dwcp_manager.go`

```go
// Line 90-94: Phase 0-1 (DONE)
m.transport = transport.New(...)
m.compression = compression.New(...)

// Line 96-99: Phase 2 - Prediction
if m.config.Prediction.Enabled {
    m.prediction = prediction.New(...)
}

// Line 101-104: Phase 3 - Sync
if m.config.Sync.Enabled {
    m.sync = sync.New(...)
}

// Line 107-109: Phase 3 - Consensus
if m.config.Consensus.Enabled {
    m.consensus = consensus.New(...)
}

// Line 233-241: collectMetrics() needs Phase 2-3 updates
// Line 250-254: detectNetworkTier() needs implementation
// Line 257-261: getTransportMode() needs implementation
// Line 277-283: HealthCheck() needs Phase 2-3 checks
```

---

## File Locations Reference

```
/home/kp/novacron/backend/core/network/dwcp/

├── transport/
│   ├── multi_stream_tcp.go ✅ (472 LOC)
│   └── multi_stream_tcp_test.go ✅ (550 LOC)
│
├── compression/
│   ├── delta_encoder.go ✅ (408 LOC)
│   └── delta_encoder_test.go ✅ (540 LOC)
│
├── prediction/ ❌ (EMPTY)
│   ├── bandwidth_predictor.go
│   ├── lstm_model.go
│   ├── feature_extractor.go
│   ├── model_training.go
│   └── bandwidth_predictor_test.go
│
├── partition/ ❌ (EMPTY)
│   ├── task_partitioner.go
│   ├── graph_analyzer.go
│   ├── load_balancer.go
│   ├── partition_metrics.go
│   ├── rebalancer.go
│   └── task_partitioner_test.go
│
├── sync/ ❌ (EMPTY)
│   ├── async_state_sync.go
│   ├── vector_clock.go
│   ├── conflict_resolver.go
│   ├── state_store.go
│   ├── reconciliation_engine.go
│   └── async_state_sync_test.go
│
├── consensus/ ❌ (EMPTY)
│   ├── adaptive_consensus.go
│   ├── hybrid_raft_gossip.go
│   ├── crdt_manager.go
│   ├── quorum_manager.go
│   ├── consensus_orchestrator.go
│   └── adaptive_consensus_test.go
│
├── types.go ✅ (107 LOC)
├── config.go ✅ (198 LOC) [needs PartitionConfig]
├── dwcp_manager.go ✅ (287 LOC) [skeleton with TODOs]
├── integration_test.go ✅ (494 LOC)
└── .analysis/
    ├── MODULES_ANALYSIS.md
    └── QUICK_REFERENCE.md (this file)
```

---

## Performance Targets by Phase

### Phase 1 ✅ ACHIEVED
- AMST throughput: ≥ 50 MB/s over 100ms link
- HDE compression: ≥ 3x ratio
- Test coverage: ≥ 80%

### Phase 2 📋 TODO
- Bandwidth prediction accuracy: ≥ 70% (30s horizon)
- Task partitioning: ≥ 40% cross-WAN traffic reduction
- Load imbalance: < 5%

### Phase 3 📋 TODO
- State convergence: within 2×max_staleness
- System availability: during network partitions
- Consensus overhead: < 5%

### Production 📋 TODO
- WAN efficiency: ≥ 85%
- Communication overhead: ≤ 15%
- Bandwidth utilization: 70-85%
- Load imbalance: < 5%

---

## Estimated Timeline

```
Week 1-2:   Phase 2 foundation (Prediction framework selection, Graph analysis)
Week 3-4:   Phase 2 implementation (Prediction model, Partition algorithms)
Week 5-6:   Phase 2 integration & testing
Week 7-8:   Phase 3 foundation (Vector clocks, Scope detection)
Week 9-10:  Phase 3 implementation (Sync engine, Consensus orchestration)
Week 11-12: Phase 3 integration & chaos testing
Week 13-16: Full integration, performance tuning, documentation
Week 17-20: Security hardening, production validation
```

**Total Estimated Effort**: 20 weeks with 2-3 full-time engineers

---

## Key Dependencies

### For Prediction Module
- ML framework (TensorFlow, ONNX Runtime, or Go native)
- Historical bandwidth data source
- Feature extraction pipeline
- Model training infrastructure

### For Partition Module
- Task dependency graph interface
- Task scheduler integration
- Network topology from DiscoveryEngine
- Performance metrics collection

### For Sync Module
- State schema definition
- Vector clock library or custom implementation
- Versioning strategy
- Reconciliation scheduling

### For Consensus Module
- CRDT library selection (Automerge, YATA, etc.)
- Operation scope classification
- Existing Raft/Gossip integration
- Region identification mechanism

---

## Testing Strategy

### Unit Tests (Per Module)
- Phase 2: 8-10 test functions per module
- Phase 3: 10-12 test functions per module
- Target coverage: ≥ 85%

### Integration Tests
- Phase 2: Multi-region simulation (3+ regions)
- Phase 3: Chaos testing (failures, partitions, latency)

### Performance Tests
- Bandwidth prediction: Accuracy metrics
- Task partitioning: CCR, LIF metrics
- Sync: Convergence time, staleness
- Consensus: Overhead, latency

---

## Next Steps (Immediate)

1. **Code Config Fix** (30 min)
   - Add `PartitionConfig` to main `Config` struct
   - Update validation logic

2. **Select ML Framework** (1-2 hours)
   - Evaluate: TensorFlow Lite, ONNX Runtime, Go-based alternatives
   - Decision: Create issue/PR for selection

3. **Define State Schema** (2-4 hours)
   - Work with core team to define state structure
   - Determine versioning strategy
   - Create interface definitions

4. **Plan Phase 2 Sprint** (4 hours)
   - Assign Prediction implementation
   - Assign Partition implementation
   - Set up testing infrastructure
   - Create task tracking

---

## Key Contacts & Resources

**Specification**: `/home/kp/novacron/docs/architecture/distributed-wan-communication-protocol.md`

**Analysis Report**: `/home/kp/novacron/backend/core/network/dwcp/.analysis/MODULES_ANALYSIS.md`

**Existing Code**:
- Transport: `/home/kp/novacron/backend/core/network/dwcp/transport/`
- Compression: `/home/kp/novacron/backend/core/network/dwcp/compression/`
- Config: `/home/kp/novacron/backend/core/network/dwcp/config.go`
- Manager: `/home/kp/novacron/backend/core/network/dwcp/dwcp_manager.go`

---

**Document Generated**: 2025-11-08  
**Status**: DWCP Phase 0-1 Complete, Phase 2-3 Planning
