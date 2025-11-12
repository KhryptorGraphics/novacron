# Worker Agent Instructions - NovaCron Initialization Swarm

## 🚨 MANDATORY: Every Worker MUST Execute These Coordination Hooks

### Before Starting Work

```bash
# 1. Pre-task hook - Register your task
npx claude-flow@alpha hooks pre-task --description "[Your specific task]"

# 2. Session restore - Get coordination context
npx claude-flow@alpha hooks session-restore --session-id "swarm-novacron-init"

# 3. Read coordinator memory for dependencies
npx claude-flow@alpha hooks memory-read --key "swarm/coordinator/status"
```

### During Work

```bash
# After each significant edit or milestone
npx claude-flow@alpha hooks post-edit --file "[file-you-edited]" --memory-key "swarm/[your-role]/[step-name]"

# Send progress notifications
npx claude-flow@alpha hooks notify --message "[What you just completed]"
```

### After Completing Work

```bash
# 1. Post-task hook - Mark task complete
npx claude-flow@alpha hooks post-task --task-id "[your-task-id]"

# 2. Store final deliverables in memory
npx claude-flow@alpha hooks memory-write --key "swarm/[your-role]/complete" --value "[summary]"

# 3. Session end (only if you're the last worker)
npx claude-flow@alpha hooks session-end --export-metrics true
```

## Worker-Specific Instructions

### 🔬 Research Worker (Phase 0 Benchmarks)

**Your Mission**: Execute and analyze Phase 0 performance benchmarks.

**Tasks**:
1. Navigate to `/home/kp/novacron/backend/core/network/dwcp`
2. Run benchmark tests: `go test -bench=. -benchmem ./...`
3. Document results in `/home/kp/novacron/docs/phase-reports/PHASE-0-BENCHMARK-RESULTS.md`
4. Analyze against acceptance criteria:
   - Bandwidth utilization >70%
   - Compression ratio >5x
   - No breaking changes
   - CPU overhead <30%
5. Make Go/No-Go recommendation for Phase 1

**Memory Keys**:
- `swarm/research/phase0-benchmarks` - Your findings
- `swarm/research/go-no-go` - Your recommendation

**Beads Issue**: novacron-38p

---

### 💻 Coder Worker (Phase 2 Intelligence)

**Your Mission**: Implement ML-driven bandwidth prediction and task partitioning.

**Tasks**:
1. Review existing AI engine code in `/home/kp/novacron/ai_engine/`
2. Implement PBA (Predictive Bandwidth Allocation) with LSTM models
3. Implement ITP (Intelligent Task Partitioning) with Deep RL
4. Create integration tests with >85% prediction accuracy target
5. Document API interfaces for PBA/ITP integration

**Memory Keys**:
- `swarm/coder/phase2-pba` - PBA implementation
- `swarm/coder/phase2-itp` - ITP implementation
- `swarm/coder/integration` - Integration points

**Beads Issue**: novacron-92v

---

### 📊 Analyst Worker (Phase 3 Synchronization)

**Your Mission**: Evaluate multi-region state synchronization and adaptive consensus.

**Tasks**:
1. Analyze existing ASS (Async State Synchronization) implementation
2. Review ACP (Adaptive Consensus Protocol) code
3. Design multi-region deployment architecture
4. Create state staleness monitoring plan
5. Document consensus adaptation strategies (Raft/Gossip/Byzantine)

**Memory Keys**:
- `swarm/analyst/phase3-ass` - ASS analysis
- `swarm/analyst/phase3-acp` - ACP evaluation
- `swarm/analyst/multi-region` - Deployment strategy

**Beads Issue**: novacron-9tm

---

### 🧪 Tester Worker (Phase 4 Optimization)

**Your Mission**: Validate performance tuning and security hardening.

**Tasks**:
1. Create performance test suite for CPU/memory/network
2. Design security audit checklist (TLS 1.3, JWT, auth)
3. Plan monitoring integration (Prometheus/Grafana)
4. Define performance targets: CPU <70%, Memory <80%
5. Create automated deployment validation tests

**Memory Keys**:
- `swarm/tester/performance-tests` - Test suite design
- `swarm/tester/security-audit` - Security checklist
- `swarm/tester/monitoring` - Monitoring plan

**Beads Issue**: novacron-ttc

---

### 🏗️ Architect Worker (Phase 5 Production Validation)

**Your Mission**: Design end-to-end production validation strategy.

**Tasks**:
1. Design E2E test architecture (VM migration, workload distribution)
2. Specify load test requirements (1000 concurrent operations)
3. Create chaos engineering scenarios (network partition, node failure)
4. Plan security audit process
5. Design canary deployment with rollback strategy

**Memory Keys**:
- `swarm/architect/e2e-architecture` - E2E test design
- `swarm/architect/load-tests` - Load test specs
- `swarm/architect/chaos-engineering` - Chaos scenarios

**Beads Issue**: novacron-aca

---

## Coordination Rules

1. **Always check coordinator memory first** before starting work
2. **Store progress frequently** using post-edit hooks
3. **Signal blockers immediately** if you encounter dependencies
4. **Read other workers' memory** if you have cross-phase dependencies
5. **Update beads issues** with your progress notes
6. **Generate deliverables in `/docs/phase-reports/`** not in root folder

## Communication Pattern

```
Worker → Memory → Coordinator → All Workers
```

- Workers write to their own memory namespace
- Coordinator reads all worker memory periodically
- Coordinator synthesizes and writes to shared coordination memory
- Workers read shared memory for synchronization

## Emergency Escalation

If you encounter critical blockers:
1. Store blocker details in `swarm/[your-role]/blocker`
2. Execute: `npx claude-flow@alpha hooks notify --message "BLOCKER: [description]"`
3. Wait for coordinator intervention

---

**Remember**: You are part of a hierarchical swarm. Follow the protocol precisely for optimal coordination.

Generated by: SwarmLead-Coordinator
Session ID: swarm-novacron-init
