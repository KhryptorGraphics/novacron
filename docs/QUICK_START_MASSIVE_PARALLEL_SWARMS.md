# QUICK START: Launch 3 Massive Parallel Swarms (73+ Agents)

## 🚀 One-Command Launch (Copy-Paste Ready)

### Prerequisites
```bash
# Install Claude-Flow
npm install -g claude-flow@alpha

# Verify installation
npx claude-flow@alpha --version
```

---

## 📋 Three Prompts to Launch

### 🧠 SWARM 1: Neural Training (4 agents, 2-4 hours)

**Copy this entire block into Claude Code Terminal 1:**

```text
MISSION: Execute DWCP Neural Training Pipeline

Read execution plan: docs/CLAUDE_CODE_NEURAL_TRAINING_PROMPT.md

CRITICAL RULES:
1. Spawn ALL 4 training agents in ONE message (parallel execution)
2. NEVER save working files to root folder
3. Mark tasks complete as you finish them
4. Batch ALL related operations together

START NOW with Phase 1: Neural Training Pipeline.

Spawn 4 parallel training agents in a SINGLE message:
1. Bandwidth Predictor trainer (train_lstm.py → ≥98% accuracy)
2. Node Reliability trainer (train_isolation_forest.py → ≥98% recall)
3. Consensus Latency trainer (train_lstm_autoencoder.py → ≥98% detection)
4. Compression Selector planner (design new training pipeline)

After neural training completes, automatically proceed to Critical Fixes (P0 Issues), then continue through all remaining phases.

Execute now.
```

---

### 🔧 SWARM 2: Critical Fixes (19 agents, ~60 minutes)

**Copy this entire block into Claude Code Terminal 2:**

```text
MISSION: Execute DWCP Critical Fixes Swarm (Massively Parallel)

Read execution plan: docs/CLAUDE_CODE_CRITICAL_FIXES_SWARM_PROMPT.md

CRITICAL RULES:
1. Spawn ALL 19 agents in ONE message (1 coordinator + 15 fix agents + 3 QA agents)
2. Use Claude-Flow hooks for coordination (pre-task, post-edit, notify, session-restore)
3. NEVER break DWCP sanity checklist (see doc)
4. Run tests after EVERY change: go test -race ./backend/core/network/dwcp/...

SWARM ARCHITECTURE:
- 1 Coordinator (mesh topology, conflict resolution, progress tracking)
- 5 Fix Teams (3 agents each = 15 agents):
  * Team 1: Race Condition Hunters (analyzer, fixer, tester)
  * Team 2: Lifecycle Engineers (architect, implementer, validator)
  * Team 3: Config Validators (analyzer, hardener, tester)
  * Team 4: Resilience Squad (designer, integrator, tester)
  * Team 5: Memory Safety Team (analyzer, fixer, tester)
- 3 QA Agents (integration tester, performance validator, security auditor)

EXECUTION:
1. Initialize Claude-Flow: npx claude-flow@alpha swarm init --topology mesh --max-agents 20
2. Spawn all 19 agents in SINGLE message with Task tool
3. Each team fixes their issue in parallel
4. QA validates after each fix
5. Coordinator ensures no regressions

TARGET: Fix all 5 P0 issues in ~60 minutes with zero regressions.

START NOW. Spawn the swarm.
```

---

### 🌐 SWARM 3: Phases 2-6 MEGA SWARM (54 agents, ~8-11 hours)

**Copy this entire block into Claude Code Terminal 3:**

```text
MISSION: Execute DWCP Phases 2-6 MEGA SWARM (Maximum Parallelism)

Read execution plan: docs/CLAUDE_CODE_PHASES_2_6_MEGA_SWARM_PROMPT.md

CRITICAL RULES:
1. Spawn 54 agents across 5 phases in coordinated waves
2. Use Claude-Flow hierarchical topology for cross-phase coordination
3. Phases 2-4 run in PARALLEL (36 agents simultaneously)
4. Phase 5 starts when Phase 2-4 components ready (8 agents)
5. Phase 6 starts when Phase 5 validates (6 agents)
6. Use hooks for coordination: pre-task, post-edit, notify, session-restore
7. Maintain 96%+ test coverage throughout
8. Verify all research claims (ProBFT O(n√n), MADDPG 20-40%, etc.)

MEGA SWARM ARCHITECTURE (54 agents):
- 4 Command & Control (supreme coordinator, architecture guardian, integration manager, quality czar)
- 12 Phase 2 (6 ProBFT + 6 MADDPG)
- 12 Phase 3 (6 TCS-FEEL + 6 Bullshark)
- 12 Phase 4 (6 T-PBFT + 6 SNAP)
- 8 Phase 5 (comprehensive testing & validation)
- 6 Phase 6 (production deployment)

RESEARCH TARGETS TO VERIFY:
- ProBFT: O(n√n) message complexity, 33% Byzantine fault tolerance
- MADDPG: 20-40% efficiency gains in resource allocation
- TCS-FEEL: 96% accuracy, 99.6% communication reduction
- Bullshark: 6x throughput improvement vs Paxos
- T-PBFT: 26% throughput increase with reputation
- SNAP: 99.6% message reduction in consensus

EXECUTION SEQUENCE:
1. Initialize: npx claude-flow@alpha swarm init --topology hierarchical --max-agents 60
2. Spawn Command & Control (4 agents)
3. Spawn Phase 2-4 teams in PARALLEL (36 agents)
4. Monitor progress, coordinate integration
5. Spawn Phase 5 when components ready (8 agents)
6. Spawn Phase 6 when validation complete (6 agents)
7. Validate all targets met, update Beads epic novacron-7q6

TARGET: Complete distributed computing transformation in ~8-11 hours.

START NOW. Initialize the MEGA SWARM.
```

---

## 📊 Monitoring (Terminal 4)

**Copy this into Terminal 4 for real-time monitoring:**

```bash
# Watch all swarms in real-time
watch -n 30 'clear && echo "=== CLAUDE-FLOW SWARM STATUS ===" && npx claude-flow@alpha swarm status && echo && echo "=== PROCESS STATUS ===" && ps aux | grep -E "claude-code|claude-flow" | grep -v grep | wc -l && echo "Active processes" && echo && echo "=== DISK USAGE ===" && du -sh backend/core/network/dwcp/'
```

---

## ✅ Success Indicators

### Swarm 1 Complete When:
```bash
# Check for trained models
ls -lh backend/core/network/dwcp/prediction/models/*.onnx
ls -lh backend/core/network/dwcp/monitoring/models/*.{pkl,onnx}
ls -lh backend/core/network/dwcp/compression/models/*.onnx

# Verify evaluation reports exist
ls -lh backend/core/network/dwcp/*/training/*_evaluation_report.txt
```

### Swarm 2 Complete When:
```bash
# All tests pass
go test ./backend/core/network/dwcp/...
go test -race ./backend/core/network/dwcp/...
go vet ./backend/core/network/dwcp/...

# Build succeeds
go build ./backend/core/network/dwcp/...
```

### Swarm 3 Complete When:
```bash
# All new components exist
ls backend/core/network/dwcp/consensus/{probft,bullshark,tpbft,snap}.go
ls backend/core/network/dwcp/prediction/maddpg_agent.go
ls backend/core/network/dwcp/federated/tcs_feel.go

# Test coverage ≥96%
go test -cover ./backend/core/network/dwcp/... | grep "coverage:"

# All benchmarks pass
go test -bench=. ./backend/core/network/dwcp/...
```

---

## 🎯 Expected Timeline

```
Hour 0:  Launch all 3 swarms
Hour 1:  Swarm 2 completes (Critical Fixes)
Hour 2:  Swarm 1 completes (Neural Training)
Hour 6:  Swarm 3 Phase 2-4 complete (ProBFT, MADDPG, TCS-FEEL, Bullshark, T-PBFT, SNAP)
Hour 9:  Swarm 3 Phase 5 complete (Testing & Validation)
Hour 11: Swarm 3 Phase 6 complete (Production Deployment)
Hour 12: ALL SWARMS COMPLETE 🎉
```

---

## 🚨 Troubleshooting

### If a swarm gets stuck:
```bash
# Check Claude-Flow status
npx claude-flow@alpha swarm status

# Check for errors
npx claude-flow@alpha hooks session-restore --session-id "novacron-master-swarm"

# Restart specific swarm (example: Swarm 2)
# Just re-paste the prompt into the Claude Code terminal
```

### If tests fail:
```bash
# Run verbose tests to see failures
go test -v ./backend/core/network/dwcp/...

# Check for race conditions
go test -race -v ./backend/core/network/dwcp/...

# Check for vet warnings
go vet ./backend/core/network/dwcp/...
```

---

## 🎉 Completion Checklist

- [ ] Swarm 1: 4 neural models trained to ≥98% accuracy
- [ ] Swarm 2: 5 P0 issues fixed, all tests green
- [ ] Swarm 3 Phase 2: ProBFT + MADDPG implemented and verified
- [ ] Swarm 3 Phase 3: TCS-FEEL + Bullshark implemented and verified
- [ ] Swarm 3 Phase 4: T-PBFT + SNAP implemented and verified
- [ ] Swarm 3 Phase 5: 96%+ test coverage, all benchmarks pass
- [ ] Swarm 3 Phase 6: Production deployment successful (10%→50%→100%)
- [ ] All documentation updated
- [ ] Beads epic `novacron-7q6` marked complete

**When all checked: NovaCron is now a global internet supercomputer! 🚀**

