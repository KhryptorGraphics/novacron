# Claude Code: DWCP Neural Training & Distributed Computing Enhancement

## 🎯 Mission Overview

You are implementing a comprehensive distributed computing enhancement for NovaCron's DWCP (Distributed WAN Communication Protocol) system. This transforms NovaCron from a datacenter-centric orchestrator into a **global internet supercomputer** capable of coordinating untrusted nodes across the public internet with Byzantine fault tolerance.

**Your immediate focus**: Train 4 neural models to ≥98% accuracy, then systematically implement the distributed computing enhancements across 6 phases.

---

## 📋 Task Execution Order

Execute tasks in this exact sequence, marking each complete before proceeding:

### ✅ COMPLETED
- [x] Academic Research & Analysis Phase (50+ papers analyzed)
- [x] Neural Training Preparation (infrastructure ready)
- [x] Phase 0: Environment Setup
- [x] DWCP Manager + TaskPartitioner integration

### 🔄 CURRENT PRIORITY: Phase 1 - Neural Training Pipeline

**Task**: `Phase 1: Neural Training Pipeline`  
**Goal**: Train 4 neural models to ≥98% accuracy before deeper code development

#### Model 1: Bandwidth Predictor (LSTM)
- **Location**: `backend/core/network/dwcp/prediction/training/train_lstm.py`
- **Target**: ≥98% accuracy (≥0.98 correlation + MAPE <5% OR ≥98% class accuracy on good/degraded/bad)
- **Data**: Sliding window of network metrics (rtt, throughput, jitter, packet_loss, etc.)
- **Action**: Refine architecture, hyperparameters, validation strategy

#### Model 2: Node Reliability (Isolation Forest)
- **Location**: `backend/core/network/dwcp/monitoring/training/train_isolation_forest.py`
- **Target**: ≥98% recall on labeled incidents with acceptable FP rate
- **Data**: Node-level aggregates (error rates, timeouts, latency spikes, SLA violations)
- **Action**: Tune features, contamination rate, thresholds

#### Model 3: Consensus Latency (LSTM Autoencoder)
- **Location**: `backend/core/network/dwcp/monitoring/training/train_lstm_autoencoder.py`
- **Target**: ≥98% detection accuracy on high-latency episodes
- **Data**: Consensus metrics (queue depth, proposals, network tier, latencies)
- **Action**: Optimize architecture, sequence length, anomaly thresholding

#### Model 4: Compression Selector (Policy/Classifier)
- **Location**: Design new training script based on `backend/core/network/dwcp/compression/*.go`
- **Target**: ≥98% decision accuracy vs offline oracle
- **Data**: HDE + AMST metrics (compression_ratio, delta_hit_rate, bandwidth, latency, CPU)
- **Action**: Design data collection pipeline, supervised/bandit formulation

**Shared Data Schema** (subset per model):
```
timestamp, region, az, link_type (dc/metro/wan)
node_id, peer_id
rtt_ms, jitter_ms, throughput_mbps, bytes_tx, bytes_rx
packet_loss, retransmits, congestion_window, queue_depth
dwcp_mode, network_tier, transport_type
hde_compression_ratio, hde_delta_hit_rate
amst_streams, amst_transfer_rate
error_budget_burn_rate
```

**Success Criteria**:
- Each model produces: finalized data schema, documented training command, evaluation report demonstrating ≥98% target
- Models integrate WITHOUT changing public DWCP Go APIs
- Reproducible CLI-driven training (no hidden notebook state)

---

### 🔜 NEXT: Phase 1 - Critical Fixes (P0 Issues)

After neural training completes, fix 5 critical issues:

1. **Race Condition in Metrics Collection** (`dwcp_manager.go:225-248`)
   - Fix: Acquire locks in consistent order, use local variables to bridge mutex boundaries

2. **Component Lifecycle** 
   - Define interfaces, implement initialization/shutdown for all DWCP components

3. **Configuration Validation**
   - Always validate config structure regardless of Enabled flag

4. **Error Recovery & Circuit Breaker**
   - Add health monitoring, circuit breaker pattern, exponential backoff retry

5. **Unsafe Config Copy**
   - Allocate config copy on heap instead of returning pointer to stack variable

---

### 📅 Remaining Phases (Execute After Phase 1)

- **Phase 2**: ProBFT + MADDPG Implementation
- **Phase 3**: TCS-FEEL + Bullshark (Federated Learning + DAG Consensus)
- **Phase 4**: T-PBFT + SNAP (Reputation + Communication Efficiency)
- **Phase 5**: Testing & Validation (96%+ coverage, chaos engineering)
- **Phase 6**: Production Deployment (10%→50%→100% rollout)

---

## 🛠️ Execution Guidelines

### Parallel Execution (CRITICAL)
- **ALWAYS batch ALL related operations in ONE message**
- Use Claude Code's Task tool to spawn agents concurrently
- Example:
  ```javascript
  Task("Bandwidth predictor trainer", "Refine train_lstm.py to ≥98% accuracy...", "sparc-coder")
  Task("Node reliability trainer", "Tune train_isolation_forest.py to ≥98% recall...", "sparc-coder")
  Task("Consensus latency trainer", "Optimize train_lstm_autoencoder.py...", "sparc-coder")
  Task("Compression selector planner", "Design data collection pipeline...", "architecture")
  ```

### File Organization
- **NEVER save working files to root folder**
- Use appropriate subdirectories:
  - `/docs` - Documentation
  - `/backend/core/network/dwcp/prediction/training` - Bandwidth predictor
  - `/backend/core/network/dwcp/monitoring/training` - Reliability & latency models
  - `/backend/core/network/dwcp/compression` - Compression selector

### Code Quality
- Respect existing Go APIs - focus on Python training code
- Prefer editing existing files over creating new ones
- Use package managers (pip, go mod) instead of manual edits
- Test after every significant change

---

## 📊 Progress Tracking

After completing each model training:
1. Mark the specific model subtask as complete
2. Document: data schema, training command, evaluation metrics
3. Verify model artifacts are saved and loadable
4. Move to next model

After completing all 4 models:
1. Mark "Phase 1: Neural Training Pipeline" as COMPLETE
2. Automatically proceed to "Phase 1: Critical Fixes (P0 Issues)"

---

## 🚀 Start Command

Begin with this exact sequence:

1. **Spawn parallel training agents** for all 4 models in ONE message
2. **Monitor progress** and collect evaluation reports
3. **Mark complete** when all models meet ≥98% targets
4. **Proceed** to Critical Fixes phase

**Remember**: Use Task tool for parallel execution, batch all operations, and maintain file organization standards.

