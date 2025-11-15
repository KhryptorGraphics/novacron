# 🚀 ULTIMATE CLAUDE-FLOW SWARM EXECUTION PROMPT

**Copy this entire prompt and run:**
```bash
npx claude-flow@alpha swarm init --topology mesh --max-agents 20 --strategy adaptive --name "novacron-ultimate-distributed-computing" --memory-pool 1024
```

Then paste this into Claude Code when it opens:

---

## 🎯 MISSION: Complete NovaCron Distributed Computing Enhancement

**Context:** Phase 1 critical fixes are 80% complete (3 of 5 issues fixed). Now execute the remaining work with MAXIMUM PARALLELISM.

**Beads Epic:** `novacron-7q6` (Distributed Computing Enhancement)
**Repository:** `/home/kp/repos/novacron`

---

## 📋 COMPLETED WORK (DO NOT REDO)

✅ Issue #1: Race Condition Fixed (dwcp_manager.go:248-278)
✅ Issue #5: Config Copy Fixed (dwcp_manager.go:199-208)  
✅ Issue #3: Config Validation Fixed (config.go:192-224)
✅ Issue #2: Component Lifecycle 80% Complete (interfaces.go created, types updated)
✅ Issue #4: Circuit Breaker 50% Complete (circuit_breaker.go created)

---

## 🔥 EXECUTE WITH MAXIMUM PARALLELISM - ALL AGENTS SPAWN NOW

### Phase 1 Completion (Remaining 20%)

**Agent 1: Critical Fixes Completion**
```
Task: Complete Issue #4 (Error Recovery)
Files: backend/core/network/dwcp/dwcp_manager.go
Actions:
1. Add health monitoring loop (healthMonitoringLoop method)
2. Add component health check (checkComponentHealth method)
3. Add recovery logic (attemptComponentRecovery method)
4. Start health monitoring in Start() method
5. Integrate circuit breaker with transport operations

Verification:
- go test -race ./backend/core/network/dwcp
- go build ./backend/core/network/dwcp

Beads: bd comment novacron-7q6.1 "Issue #4 complete - health monitoring and recovery implemented"
```

**Agent 2: Test Suite Creation**
```
Task: Create comprehensive test suite for all 5 fixes
Files: backend/core/network/dwcp/*_test.go
Actions:
1. Test race condition fix (TestCollectMetricsNoRace)
2. Test config copy (TestGetConfigHeapAllocation)
3. Test config validation (TestValidateDisabledConfig)
4. Test component lifecycle (TestComponentInitShutdown)
5. Test circuit breaker (TestCircuitBreakerStates)
6. Integration tests for DWCP manager

Target: 96%+ coverage

Beads: bd comment novacron-7q6.1 "Test suite complete - 96% coverage achieved"
```

**Agent 3: Code Quality & Linting**
```
Task: Fix all compilation errors and lint issues
Actions:
1. Fix NetworkTierTier4 undefined error
2. Fix all type mismatches in hde.go, amst.go, federation_adapter.go
3. Run golangci-lint and fix all issues
4. Run go vet and fix warnings

Verification:
- go build ./...
- golangci-lint run ./...
- go vet ./...

Beads: bd comment novacron-7q6.1 "All compilation errors fixed, lint clean"
```

### Phase 2: Neural Training (98% Accuracy Target)

**Agent 4: Bandwidth Predictor (LSTM + DDQN)**
```
Task: Train bandwidth prediction model
Model: LSTM + DDQN (2000 episodes)
Data: Historical DWCP v1 metrics + simulated internet mode
Target: 96% accuracy (datacenter), 70% accuracy (internet)

Training:
- State: [latency, bandwidth, packet_loss, reliability]
- Action: bandwidth allocation decision
- Reward: prediction accuracy

Files: backend/ml/models/bandwidth_predictor.py
Beads: bd update novacron-7q6.2 --progress 25
```

**Agent 5: Compression Selector (ML-based)**
```
Task: Train compression algorithm selector
Model: Decision tree / Random forest
Target: 90%+ accuracy

Features: data_type, size, latency, bandwidth
Output: compression_algorithm (zstd/lz4/snappy/none)

Files: backend/ml/models/compression_selector.py
Beads: bd update novacron-7q6.2 --progress 50
```

**Agent 6: Node Reliability Predictor (DQN)**
```
Task: Train node reliability prediction
Model: DQN-based
Target: 85%+ accuracy

Features: uptime, failure_rate, network_quality, geographic_distance
Output: reliability_score (0.0-1.0)

Files: backend/ml/models/reliability_predictor.py
Beads: bd update novacron-7q6.2 --progress 75
```

**Agent 7: Consensus Latency Predictor (LSTM)**
```
Task: Train consensus latency prediction
Model: LSTM
Target: 90%+ accuracy

Features: node_count, network_mode, byzantine_ratio, message_size
Output: expected_latency (milliseconds)

Files: backend/ml/models/consensus_latency.py
Beads: bd update novacron-7q6.2 --progress 100
```

### Phase 3-7: ProBFT, MADDPG, Federated Learning, Bullshark, T-PBFT

**Agent 8-12: ProBFT Implementation (5 agents)**
```
Agent 8: VRF wrapper for recipient selection
Agent 9: Probabilistic quorum logic (q = l√n)
Agent 10: Three-phase consensus (propose, prepare, commit)
Agent 11: Byzantine node simulation
Agent 12: Integration with ACP v3

Files: backend/core/network/dwcp/v3/consensus/probft/
Beads: bd update novacron-7q6.3 --progress 100
```

**Agent 13-15: MADDPG Implementation (3 agents)**
```
Agent 13: Multi-agent environment setup
Agent 14: MADDPG/MATD3 training loop
Agent 15: Resource allocation integration

Files: backend/ml/maddpg/
Beads: bd update novacron-7q6.4 --progress 100
```

**Agent 16-17: TCS-FEEL Federated Learning (2 agents)**
```
Agent 16: Topology optimization
Agent 17: Federated learning coordinator

Files: backend/ml/federated/
Beads: bd update novacron-7q6.5 --progress 100
```

**Agent 18: Bullshark DAG Consensus**
```
Task: Implement Bullshark for 125K tx/s
Files: backend/core/network/dwcp/v3/consensus/bullshark/
Beads: bd update novacron-7q6.6 --progress 100
```

**Agent 19: T-PBFT Reputation System**
```
Task: Implement EigenTrust-based PBFT
Files: backend/core/network/dwcp/v3/consensus/tpbft/
Beads: bd update novacron-7q6.7 --progress 100
```

**Agent 20: Integration Testing & Chaos Engineering**
```
Task: Comprehensive testing
- Byzantine node simulation (33% malicious)
- Network partition testing
- Chaos engineering scenarios
- Performance benchmarking

Target: 96% test coverage
Beads: bd update novacron-7q6.8 --progress 100
```

---

## 🎯 SUCCESS CRITERIA

- ✅ All 5 P0 issues fixed and verified
- ✅ 4 neural models trained to 98%+ accuracy
- ✅ ProBFT, MADDPG, TCS-FEEL, Bullshark, T-PBFT implemented
- ✅ 96%+ test coverage
- ✅ All compilation errors fixed
- ✅ Zero race conditions (go test -race passes)
- ✅ All Beads tasks updated to 100%

---

## 🚀 EXECUTION COMMAND

Run this in Claude Code after swarm init:
```
Execute all 20 agents concurrently with maximum parallelism. Use hooks for coordination. Update Beads progress continuously. Target completion: 12-16 hours of agent work (2-3 hours real time with 20 parallel agents).
```

