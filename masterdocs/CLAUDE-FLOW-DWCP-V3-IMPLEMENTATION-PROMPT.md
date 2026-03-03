# 🚀 ULTIMATE BEADS + CLAUDE-FLOW PROMPT: NovaCron DWCP v1 → v3 Upgrade
## Advanced Swarm Orchestration with Neural Training, SPARC TDD, and Beads Project Management

---

## 🎯 MASTER PROMPT FOR CLAUDE-CODE

Copy and paste this ENTIRE prompt to Claude-Code:

```
🚨 CRITICAL MISSION: Upgrade NovaCron DWCP v1.0 → v3.0 with Advanced Orchestration

📋 CONTEXT:
You are upgrading NovaCron's existing DWCP v1.0 to v3.0 using:
- **Beads MCP**: Project management and issue tracking
- **Claude-Flow**: Advanced swarm orchestration, neural training, SPARC TDD
- **Claude Code Task Tool**: Parallel agent execution

NovaCron is 85% complete with production-ready VM management, multi-cloud federation, and existing DWCP v1.0 components.

🔍 EXISTING CODEBASE:
- ✅ DWCP v1.0: `backend/core/network/dwcp/` (AMST, HDE, consensus, sync, partition)
- ✅ Federation: `backend/core/federation/` (cross-cluster, multicloud)
- ✅ Migration: `backend/core/migration/` (live migration, orchestrator)
- ✅ AI/ML: `ai_engine/bandwidth_predictor.py`, `backend/core/ai/`
- ✅ Consensus: `backend/core/consensus/` (Raft, Gossip, Paxos, EPaxos)

📁 ARCHITECTURE DOCUMENTS:
1. Current: `docs/DWCP-NOVACRON-INTEGRATION-ROADMAP.md` (2,462 lines)
2. Target: `docs/research/DWCP-INTERNET-SCALE-DISTRIBUTED-HYPERVISOR.md` (1,064 lines)
3. Analysis: `backend/core/network/dwcp/README_ANALYSIS.md`

🎯 PRIMARY OBJECTIVE:
Upgrade DWCP v1.0 → v3.0 with HYBRID architecture:
- **Datacenter Mode:** 10-100 Gbps, <10ms latency, RDMA, trusted nodes
- **Internet Mode:** 100-900 Mbps, 50-500ms latency, TCP, Byzantine tolerance
- **Hybrid Mode:** Adaptive switching between modes

6 Core Components to Upgrade:
1. AMST v1 → v3: Multi-mode transport (RDMA + TCP)
2. HDE v1 → v3: ML-based compression + CRDT
3. PBA v1 → v3: Enhanced LSTM bandwidth prediction
4. ASS v1 → v3: Hybrid state sync (Raft/EPaxos + Gossip/PBFT)
5. ITP v1 → v3: Enhanced ML-based VM placement
6. ACP v1 → v3: Adaptive consensus protocol

🧠 NEURAL TRAINING TARGET: 98% accuracy on NovaCron patterns
🎯 ORCHESTRATION: Beads MCP + Claude-Flow + Claude Code Task Tool
📊 METHODOLOGY: SPARC TDD (Specification → Pseudocode → Architecture → Refinement → Completion)

⚡ EXECUTION STRATEGY (ALL IN SINGLE MESSAGES):

═══════════════════════════════════════════════════════════════════════════════
STEP 0: INITIALIZE BEADS PROJECT MANAGEMENT (Single Message)
═══════════════════════════════════════════════════════════════════════════════

# Initialize Beads for NovaCron project tracking
mcp__beads__set_context { workspace_root: "/home/kp/novacron" }

# Verify Beads initialization
mcp__beads__where_am_i { workspace_root: "/home/kp/novacron" }

# Create epic for DWCP v1 → v3 upgrade
mcp__beads__create {
  workspace_root: "/home/kp/novacron",
  id: "DWCP-UPGRADE",
  title: "DWCP v1.0 → v3.0 Hybrid Architecture Upgrade",
  description: "Upgrade existing DWCP v1.0 to v3.0 with hybrid datacenter + internet support. Includes AMST, HDE, PBA, ASS, ITP, ACP component upgrades with backward compatibility.",
  issue_type: "epic",
  priority: 1,
  assignee: "claude-code",
  labels: ["dwcp", "upgrade", "v3", "hybrid-architecture"]
}

═══════════════════════════════════════════════════════════════════════════════
STEP 1: INITIALIZE CLAUDE-FLOW SWARM + NEURAL TRAINING (Single Message)
═══════════════════════════════════════════════════════════════════════════════

# Initialize advanced hierarchical swarm with neural training
npx claude-flow@alpha swarm init \
  --topology hierarchical \
  --max-agents 15 \
  --enable-neural true \
  --neural-target-accuracy 0.98 \
  --enable-hooks true \
  --enable-memory true \
  --session-id "novacron-dwcp-v1-to-v3-upgrade" \
  --project-root "/home/kp/novacron"

# Train neural models on EXISTING NovaCron codebase patterns
npx claude-flow@alpha neural train \
  --patterns "dwcp-v1,amst,hde,consensus,federation,migration,multicloud,ml-placement" \
  --training-data "backend/core/network/dwcp/,backend/core/federation/,backend/core/migration/,backend/core/consensus/,ai_engine/" \
  --target-accuracy 0.98 \
  --iterations 1000 \
  --export-model "novacron-dwcp-v1-patterns.json" \
  --enable-pattern-recognition true

# Enable advanced hooks for coordination
npx claude-flow@alpha hooks enable \
  --pre-task true \
  --post-edit true \
  --post-task true \
  --session-restore true \
  --auto-format true \
  --neural-train true

# Initialize SPARC TDD workflow
npx claude-flow@alpha sparc init \
  --project "novacron-dwcp-v3" \
  --methodology "tdd" \
  --enable-pipeline true

═══════════════════════════════════════════════════════════════════════════════
STEP 2: CREATE BEADS TASKS FOR ALL COMPONENTS (Single Message)
═══════════════════════════════════════════════════════════════════════════════

# Create Beads tasks for all 6 DWCP components + integration tasks
mcp__beads__create {
  workspace_root: "/home/kp/novacron",
  id: "DWCP-001",
  title: "Analyze DWCP v1.0 Codebase",
  description: "Analyze existing DWCP v1.0 implementation in backend/core/network/dwcp/. Map all components (AMST, HDE, consensus, sync, partition). Identify integration points with federation and migration. Create upgrade compatibility matrix.",
  issue_type: "task",
  priority: 1,
  assignee: "code-analyzer-agent",
  labels: ["analysis", "dwcp-v1"],
  deps: []
}

mcp__beads__create {
  workspace_root: "/home/kp/novacron",
  id: "DWCP-002",
  title: "Create Migration Strategy (v1 → v3)",
  description: "Create backward-compatible migration strategy. Design dual-mode operation (v1 and v3 simultaneously). Create feature flags for gradual rollout. Design rollback procedures.",
  issue_type: "task",
  priority: 1,
  assignee: "migration-planner-agent",
  labels: ["planning", "migration-strategy"],
  deps: ["DWCP-001"]
}

mcp__beads__create {
  workspace_root: "/home/kp/novacron",
  id: "DWCP-003",
  title: "Upgrade AMST v1 → v3",
  description: "Upgrade Adaptive Multi-Stream Transport with hybrid datacenter + internet support. Add mode detection (datacenter/internet/hybrid). Enhance existing AMST v1 with internet-optimized TCP. Keep RDMA for datacenter mode.",
  issue_type: "task",
  priority: 2,
  assignee: "backend-dev-agent",
  labels: ["amst", "transport", "upgrade"],
  deps: ["DWCP-002"]
}

mcp__beads__create {
  workspace_root: "/home/kp/novacron",
  id: "DWCP-004",
  title: "Upgrade HDE v1 → v3",
  description: "Upgrade Hierarchical Delta Encoding with ML-based compression and CRDT integration. Integrate with ai_engine/ for compression model selection. Add CRDT support for conflict-free state sync.",
  issue_type: "task",
  priority: 2,
  assignee: "backend-dev-agent",
  labels: ["hde", "compression", "ml", "upgrade"],
  deps: ["DWCP-002"]
}

mcp__beads__create {
  workspace_root: "/home/kp/novacron",
  id: "DWCP-005",
  title: "Upgrade PBA v1 → v3",
  description: "Upgrade Predictive Bandwidth Allocation with enhanced LSTM and hybrid mode support. Enhance existing ai_engine/bandwidth_predictor.py. Add multi-mode prediction (datacenter vs internet).",
  issue_type: "task",
  priority: 2,
  assignee: "ml-developer-agent",
  labels: ["pba", "ml", "lstm", "upgrade"],
  deps: ["DWCP-002"]
}

mcp__beads__create {
  workspace_root: "/home/kp/novacron",
  id: "DWCP-006",
  title: "Upgrade ASS v1 → v3 + ACP v1 → v3",
  description: "Upgrade Asynchronous State Synchronization and Adaptive Consensus Protocol. Add mode-aware state sync (strong consistency for datacenter, eventual for internet). Add PBFT for Byzantine tolerance.",
  issue_type: "task",
  priority: 2,
  assignee: "consensus-builder-agent",
  labels: ["ass", "acp", "consensus", "byzantine", "upgrade"],
  deps: ["DWCP-002"]
}

mcp__beads__create {
  workspace_root: "/home/kp/novacron",
  id: "DWCP-007",
  title: "Upgrade ITP v1 → v3",
  description: "Upgrade Intelligent Task Partitioning with enhanced ML-based placement. Enhance existing DQN agent. Add mode-aware placement (datacenter vs internet). Add geographic optimization.",
  issue_type: "task",
  priority: 2,
  assignee: "code-analyzer-agent",
  labels: ["itp", "ml", "placement", "upgrade"],
  deps: ["DWCP-002"]
}

mcp__beads__create {
  workspace_root: "/home/kp/novacron",
  id: "DWCP-008",
  title: "Enhance Migration with DWCP v3",
  description: "Enhance existing migration orchestrator with DWCP v3 support. Add mode detection to migration phases. Integrate AMST v3, HDE v3, PBA v3 with existing migration.",
  issue_type: "task",
  priority: 2,
  assignee: "backend-dev-agent",
  labels: ["migration", "integration"],
  deps: ["DWCP-003", "DWCP-004", "DWCP-005"]
}

mcp__beads__create {
  workspace_root: "/home/kp/novacron",
  id: "DWCP-009",
  title: "Enhance Federation with DWCP v3",
  description: "Enhance existing federation with DWCP v3 support. Integrate v3 components with cross_cluster_components.go. Add internet-scale federation support.",
  issue_type: "task",
  priority: 2,
  assignee: "backend-dev-agent",
  labels: ["federation", "integration"],
  deps: ["DWCP-003", "DWCP-004", "DWCP-005", "DWCP-006"]
}

mcp__beads__create {
  workspace_root: "/home/kp/novacron",
  id: "DWCP-010",
  title: "Add Byzantine Tolerance to Security",
  description: "Enhance existing security with Byzantine tolerance for internet mode. Add node reputation system. Add malicious node detection. Add mode-aware security.",
  issue_type: "task",
  priority: 2,
  assignee: "security-manager-agent",
  labels: ["security", "byzantine"],
  deps: ["DWCP-006"]
}

mcp__beads__create {
  workspace_root: "/home/kp/novacron",
  id: "DWCP-011",
  title: "Add DWCP v3 Metrics to Monitoring",
  description: "Enhance existing monitoring with DWCP v3 metrics. Add mode-specific metrics (datacenter vs internet). Add ML-based anomaly detection for v3 patterns.",
  issue_type: "task",
  priority: 3,
  assignee: "perf-analyzer-agent",
  labels: ["monitoring", "metrics"],
  deps: ["DWCP-003", "DWCP-004", "DWCP-005", "DWCP-006", "DWCP-007"]
}

mcp__beads__create {
  workspace_root: "/home/kp/novacron",
  id: "DWCP-012",
  title: "Create Comprehensive Upgrade Test Suite",
  description: "Create comprehensive test suite for v1 → v3 upgrade. Test backward compatibility (v1 still works). Test dual-mode operation. Test all 6 components. Target: 90%+ coverage.",
  issue_type: "task",
  priority: 3,
  assignee: "tester-agent",
  labels: ["testing", "coverage"],
  deps: ["DWCP-003", "DWCP-004", "DWCP-005", "DWCP-006", "DWCP-007", "DWCP-008", "DWCP-009", "DWCP-010", "DWCP-011"]
}

mcp__beads__create {
  workspace_root: "/home/kp/novacron",
  id: "DWCP-013",
  title: "Create Upgrade Documentation",
  description: "Create comprehensive upgrade documentation. Write UPGRADE_PLAN_V1_TO_V3.md, MIGRATION_STRATEGY_V1_TO_V3.md, UPGRADE_GUIDE_V1_TO_V3.md. Update architecture docs.",
  issue_type: "task",
  priority: 3,
  assignee: "reviewer-agent",
  labels: ["documentation"],
  deps: ["DWCP-012"]
}

# List all ready tasks
mcp__beads__ready {
  workspace_root: "/home/kp/novacron",
  limit: 20
}

═══════════════════════════════════════════════════════════════════════════════
STEP 3: PARALLEL AGENT EXECUTION WITH CLAUDE CODE TASK TOOL (Single Message)
═══════════════════════════════════════════════════════════════════════════════

Execute ALL agents concurrently using Claude Code's Task tool:

🔹 AGENT 1: Code Analyzer (code-analyzer)
Beads Task: DWCP-001
Task: "Analyze existing DWCP v1.0 implementation and create upgrade plan:

**BEFORE Starting:**
```bash
# Mark Beads task as in-progress
mcp__beads__update {
  workspace_root: \"/home/kp/novacron\",
  issue_id: \"DWCP-001\",
  status: \"in_progress\"
}

# Claude-Flow pre-task hook
npx claude-flow@alpha hooks pre-task \
  --description 'Analyze DWCP v1.0 for v3 upgrade' \
  --task-id 'DWCP-001' \
  --session-id 'novacron-dwcp-v1-to-v3-upgrade'

# Restore session context
npx claude-flow@alpha hooks session-restore \
  --session-id 'novacron-dwcp-v1-to-v3-upgrade'
```

**Analysis Tasks:**
1. Map all DWCP v1.0 components:
   - `backend/core/network/dwcp/amst.go` → AMST v1 analysis
   - `backend/core/network/dwcp/hde.go` → HDE v1 analysis
   - `backend/core/network/dwcp/consensus/` → Consensus analysis
   - `backend/core/network/dwcp/sync/` → ASS v1 analysis
   - `backend/core/network/dwcp/partition/` → ITP v1 analysis

2. Identify integration points:
   - `backend/core/federation/cross_cluster_components.go`
   - `backend/core/migration/orchestrator_dwcp.go`
   - `ai_engine/bandwidth_predictor.py`

3. Create upgrade compatibility matrix

**Output:** `backend/core/network/dwcp/UPGRADE_PLAN_V1_TO_V3.md`

**AFTER Completing:**
```bash
# Claude-Flow post-edit hook (auto-format + neural train)
npx claude-flow@alpha hooks post-edit \
  --file 'backend/core/network/dwcp/UPGRADE_PLAN_V1_TO_V3.md' \
  --memory-key 'swarm/analysis/upgrade-plan' \
  --auto-format true \
  --neural-train true

# Claude-Flow post-task hook
npx claude-flow@alpha hooks post-task \
  --task-id 'DWCP-001' \
  --status 'completed'

# Mark Beads task as complete
mcp__beads__close {
  workspace_root: \"/home/kp/novacron\",
  issue_id: \"DWCP-001\",
  reason: \"Analysis complete. Upgrade plan created.\"
}
```"

🔹 AGENT 2: Migration Planner (migration-planner)
Beads Task: DWCP-002
Dependencies: DWCP-001 (must complete first)
Task: "Create backward-compatible migration strategy from v1 to v3:

**BEFORE Starting:**
```bash
# Wait for DWCP-001 to complete
mcp__beads__show { workspace_root: \"/home/kp/novacron\", issue_id: \"DWCP-001\" }

# Mark task as in-progress
mcp__beads__update {
  workspace_root: \"/home/kp/novacron\",
  issue_id: \"DWCP-002\",
  status: \"in_progress\"
}

# Claude-Flow hooks
npx claude-flow@alpha hooks pre-task \
  --description 'Create migration strategy v1 → v3' \
  --task-id 'DWCP-002' \
  --depends-on 'DWCP-001'
```

**Migration Strategy:**
1. **Phase 1: Dual-mode operation** (v1 and v3 simultaneously)
   - Add mode detection: `backend/core/network/dwcp/mode_detector.go`
   - Datacenter mode: Use v1 (RDMA, high-bandwidth)
   - Internet mode: Use v3 (TCP, gigabit)
   - Hybrid mode: Dynamic switching

2. **Phase 2: Component upgrades** (enhance v1 components)
   - AMST v1 → v3: Add internet-optimized transport
   - HDE v1 → v3: Add CRDT integration
   - Consensus v1 → v3: Add PBFT for Byzantine tolerance
   - ASS v1 → v3: Add eventual consistency for internet

3. **Phase 3: Feature flag rollout**
   - Create feature flags: `backend/core/network/dwcp/feature_flags.go`
   - Gradual rollout: 10% → 50% → 100%
   - Rollback capability

**Output:** `backend/core/network/dwcp/MIGRATION_STRATEGY_V1_TO_V3.md`

**AFTER Completing:**
```bash
# Claude-Flow hooks
npx claude-flow@alpha hooks post-edit \
  --file 'backend/core/network/dwcp/MIGRATION_STRATEGY_V1_TO_V3.md' \
  --memory-key 'swarm/planning/migration-strategy'

npx claude-flow@alpha hooks post-task --task-id 'DWCP-002'

# Mark Beads task complete
mcp__beads__close {
  workspace_root: \"/home/kp/novacron\",
  issue_id: \"DWCP-002\",
  reason: \"Migration strategy complete.\"
}
```"

🔹 AGENT 3: AMST Upgrade Engineer (backend-dev)
Beads Task: DWCP-003
Dependencies: DWCP-002
Task: "Upgrade AMST v1 → v3 with hybrid datacenter + internet support using SPARC TDD:

**BEFORE Starting:**
```bash
# Mark task in-progress
mcp__beads__update {
  workspace_root: \"/home/kp/novacron\",
  issue_id: \"DWCP-003\",
  status: \"in_progress\"
}

# Claude-Flow SPARC TDD workflow
npx claude-flow@alpha sparc tdd \"AMST v1 → v3 hybrid transport upgrade\" \
  --component \"amst\" \
  --test-first true \
  --coverage-target 0.90

# Pre-task hook
npx claude-flow@alpha hooks pre-task \
  --description 'Upgrade AMST v1 → v3' \
  --task-id 'DWCP-003'
```

**SPARC Phase 1: Specification**
```bash
npx claude-flow@alpha sparc run spec-pseudocode \"AMST v3 hybrid transport requirements\"
```
- Define mode detection requirements (datacenter/internet/hybrid)
- Define performance targets per mode
- Define backward compatibility requirements

**SPARC Phase 2: Architecture**
```bash
npx claude-flow@alpha sparc run architect \"AMST v3 architecture design\"
```
- Design mode detection system
- Design hybrid transport layer
- Design feature flag system

**SPARC Phase 3: TDD Implementation**

**Existing Code to Enhance:**
- `backend/core/network/dwcp/amst.go` (AMST v1)
- `backend/core/network/dwcp/transport/multi_stream_tcp.go`
- `backend/core/network/dwcp/transport/rdma_transport.go`

**Step 1: Write Tests FIRST**
```bash
# Create test file
cat > backend/core/network/dwcp/transport/amst_v3_test.go
```

**Step 2: Implement Mode Detection**
```go
type AMSTMode int
const (
    ModeDatacenter AMSTMode = iota  // RDMA, 10-100 Gbps, <10ms
    ModeInternet                     // TCP, 100-900 Mbps, 50-500ms
    ModeHybrid                       // Dynamic switching
)
```

**Step 3: Enhance AMST v1**
- Keep RDMA for datacenter mode
- Add internet-optimized TCP
- Add adaptive stream count (4-16 internet, 32-512 datacenter)
- Add congestion control

**Step 4: Integration**
- `backend/core/federation/cross_cluster_components.go` (BandwidthOptimizer)
- `backend/core/migration/orchestrator_dwcp.go` (Migration)

**Files to Create/Modify:**
- `backend/core/network/dwcp/transport/amst_v3.go` (new)
- `backend/core/network/dwcp/transport/amst_v3_test.go` (new)
- `backend/core/network/dwcp/mode_detector.go` (new)
- `backend/core/network/dwcp/amst.go` (modify - add v3 support)

**AFTER Completing:**
```bash
# Run tests
go test -v -race -coverprofile=coverage_amst_v3.out ./backend/core/network/dwcp/transport/

# Claude-Flow hooks (auto-format + neural train)
npx claude-flow@alpha hooks post-edit \
  --file 'backend/core/network/dwcp/transport/amst_v3.go' \
  --memory-key 'swarm/amst/v3-implementation' \
  --auto-format true \
  --neural-train true

# SPARC completion
npx claude-flow@alpha sparc run integration \"AMST v3 integration testing\"

# Mark Beads task complete
mcp__beads__close {
  workspace_root: \"/home/kp/novacron\",
  issue_id: \"DWCP-003\",
  reason: \"AMST v3 upgrade complete with 90%+ test coverage.\"
}
```"

🔹 AGENT 4: HDE Upgrade Engineer (backend-dev)
Task: "Upgrade HDE v1 → v3 with ML-based compression and CRDT integration:

**Existing Code to Enhance:**
- `backend/core/network/dwcp/hde.go` (HDE v1)
- `backend/core/network/dwcp/compression/delta_encoder.go`
- `backend/core/network/dwcp/compression/adaptive_compression.go`
- `backend/core/network/dwcp/sync/crdt/` (CRDT support)

**Upgrade Tasks:**
1. Add ML-based compression selection:
   - Integrate with `ai_engine/` for compression model selection
   - Train model on existing VM memory patterns
   - Adaptive compression level based on network conditions

2. Add CRDT integration for conflict-free state sync:
   - Use existing `backend/core/network/dwcp/sync/crdt/`
   - Integrate with `backend/core/network/dwcp/conflict/` (conflict resolution)
   - Support eventual consistency for internet mode

3. Enhance delta encoding:
   - Keep existing delta encoder
   - Add ML-based delta prediction
   - Optimize for both datacenter and internet modes

**Files to Modify:**
- `backend/core/network/dwcp/hde.go` (add v3 features)
- `backend/core/network/dwcp/compression/hde_v3.go` (new v3 implementation)
- `backend/core/network/dwcp/compression/ml_compression_selector.go` (new ML integration)

**Integration Points:**
- `backend/core/federation/cross_cluster_components.go` (AdaptiveCompressionEngine)
- `ai_engine/` (ML model integration)

Use hooks for coordination."

🔹 AGENT 5: PBA/ML Upgrade Engineer (ml-developer)
Task: "Upgrade PBA v1 → v3 with enhanced LSTM and hybrid mode support:

**Existing Code to Enhance:**
- `backend/core/network/dwcp/prediction/` (PBA v1 with LSTM)
- `backend/core/network/dwcp/prediction/lstm_bandwidth_predictor.go`
- `ai_engine/bandwidth_predictor.py` (Python ML model)
- `backend/core/federation/cross_cluster_components.go` (BandwidthPredictionModel)

**Upgrade Tasks:**
1. Enhance existing LSTM model:
   - Current: Basic bandwidth prediction
   - Upgrade: Multi-mode prediction (datacenter vs internet)
   - Add time-series forecasting for both modes
   - Target: 70%+ accuracy for internet, 85%+ for datacenter

2. Add hybrid mode support:
   - Detect network mode (datacenter/internet/hybrid)
   - Use different prediction models per mode
   - Adaptive model switching

3. Integrate with existing AI engine:
   - Use existing `ai_engine/bandwidth_predictor.py`
   - Add new models: `ai_engine/bandwidth_predictor_v3.py`
   - Expose via existing Go bindings

**Files to Modify:**
- `backend/core/network/dwcp/prediction/lstm_bandwidth_predictor.go` (enhance v1)
- `backend/core/network/dwcp/prediction/pba_v3.go` (new v3 coordinator)
- `ai_engine/bandwidth_predictor_v3.py` (new ML model)

**Integration Points:**
- `backend/core/federation/cross_cluster_components.go` (BandwidthPredictionModel)
- `backend/core/network/dwcp/amst.go` (AMST integration)

Use hooks for ML training and pattern learning."

🔹 AGENT 6: ASS/Consensus Upgrade Engineer (consensus-builder)
Task: "Upgrade ASS v1 → v3 and ACP v1 → v3 with hybrid consensus:

**Existing Code to Enhance:**
- `backend/core/network/dwcp/sync/` (ASS v1 with Gossip)
- `backend/core/network/dwcp/sync/ass_engine.go`
- `backend/core/network/dwcp/sync/crdt/` (CRDT support)
- `backend/core/network/dwcp/consensus/` (Raft, Gossip, Paxos, EPaxos)
- `backend/core/consensus/` (NovaCron consensus layer)

**Upgrade Tasks:**

**ASS v1 → v3:**
1. Add mode-aware state synchronization:
   - Datacenter mode: Strong consistency (Raft/EPaxos)
   - Internet mode: Eventual consistency (Gossip + CRDT)
   - Hybrid mode: Adaptive consistency

2. Enhance existing CRDT integration:
   - Use existing `backend/core/network/dwcp/sync/crdt/`
   - Add conflict resolution for internet mode
   - Integrate with `backend/core/network/dwcp/conflict/`

3. Add Byzantine tolerance for internet mode:
   - Implement PBFT for untrusted nodes
   - Use existing `backend/core/network/dwcp/consensus/` as base
   - Add reputation system for node trust

**ACP v1 → v3:**
1. Enhance existing consensus:
   - Keep Raft for datacenter (already exists)
   - Keep EPaxos for WAN (already exists)
   - Add PBFT for Byzantine tolerance (new)
   - Add Gossip for internet-scale (enhance existing)

2. Add adaptive protocol selection:
   - Detect network mode
   - Select optimal consensus protocol
   - Dynamic switching based on conditions

**Files to Modify:**
- `backend/core/network/dwcp/sync/ass_engine.go` (add v3 features)
- `backend/core/network/dwcp/sync/ass_v3.go` (new v3 implementation)
- `backend/core/network/dwcp/consensus/acp_v3.go` (new adaptive consensus)
- `backend/core/network/dwcp/consensus/pbft.go` (new Byzantine tolerance)

**Integration Points:**
- `backend/core/federation/cross_cluster_components.go` (StateSynchronizationProtocol)
- `backend/core/consensus/` (NovaCron consensus integration)

Use hooks for coordination."

🔹 AGENT 7: ITP Upgrade Engineer (code-analyzer)
Task: "Upgrade ITP v1 → v3 with enhanced ML-based placement:

**Existing Code to Enhance:**
- `backend/core/network/dwcp/partition/` (ITP v1 with DQN agent)
- `backend/core/network/dwcp/partition/dqn_agent.go`
- `backend/core/network/dwcp/partition/rl_environment.go`
- `backend/core/scheduler/` (NovaCron scheduler)
- `backend/core/federation/federation_manager.go` (ScheduleVMCrossCluster)

**Upgrade Tasks:**
1. Enhance existing DQN-based placement:
   - Current: Basic reinforcement learning
   - Upgrade: Multi-mode placement (datacenter vs internet)
   - Add geographic optimization for internet mode
   - Add heterogeneous node support

2. Add mode-aware placement:
   - Datacenter mode: Optimize for performance (low latency, high bandwidth)
   - Internet mode: Optimize for reliability (node uptime, geographic proximity)
   - Hybrid mode: Balance performance and reliability

3. Integrate with existing scheduler:
   - Use existing `backend/core/scheduler/`
   - Enhance `backend/core/federation/federation_manager.go` (ScheduleVMCrossCluster)
   - Add DWCP v3 awareness to placement decisions

**Files to Modify:**
- `backend/core/network/dwcp/partition/dqn_agent.go` (enhance v1)
- `backend/core/network/dwcp/partition/itp_v3.go` (new v3 coordinator)
- `backend/core/network/dwcp/partition/mode_aware_placement.go` (new mode-aware logic)

**Integration Points:**
- `backend/core/scheduler/` (NovaCron scheduler)
- `backend/core/federation/federation_manager.go` (Cross-cluster scheduling)

Use hooks for ML training and pattern learning."

🔹 AGENT 8: Migration Integration Engineer (backend-dev)
Task: "Enhance existing migration with DWCP v3 support:

**Existing Code to Enhance:**
- `backend/core/migration/orchestrator.go` (Live migration orchestrator)
- `backend/core/migration/orchestrator_dwcp.go` (DWCP v1 integration)
- `backend/core/vm/live_migration.go` (VM migration execution)
- `backend/core/vm/vm_migration_execution.go` (Migration phases)

**Upgrade Tasks:**
1. Add DWCP v3 mode support to existing migration:
   - Datacenter mode: Use RDMA, high-bandwidth (existing)
   - Internet mode: Use TCP, compression, deduplication (new)
   - Hybrid mode: Adaptive transport selection (new)

2. Enhance existing migration phases:
   - Pre-copy: Use DWCP v3 AMST for parallel transfer
   - Stop-and-copy: Use DWCP v3 HDE for compression
   - Post-copy: Use DWCP v3 PBA for bandwidth prediction

3. Add predictive prefetching with DWCP v3:
   - Use existing `backend/core/vm/vm_migration_execution.go` (executePredictivePrefetching)
   - Enhance with DWCP v3 bandwidth prediction
   - Optimize for both datacenter and internet modes

**Files to Modify:**
- `backend/core/migration/orchestrator_dwcp.go` (add v3 support)
- `backend/core/migration/orchestrator_dwcp_v3.go` (new v3 integration)
- `backend/core/vm/live_migration.go` (add v3 mode detection)

**Integration Points:**
- `backend/core/network/dwcp/amst.go` (AMST v3)
- `backend/core/network/dwcp/hde.go` (HDE v3)
- `backend/core/network/dwcp/prediction/` (PBA v3)

**Performance Targets:**
- Datacenter mode: <500ms downtime (existing)
- Internet mode: 45-90 seconds for 2GB VM (new)

Use hooks for coordination."

🔹 AGENT 9: Federation Integration Engineer (backend-dev)
Task: "Enhance existing federation with DWCP v3 support:

**Existing Code to Enhance:**
- `backend/core/federation/federation_manager.go` (Federation manager)
- `backend/core/federation/cross_cluster_components.go` (Cross-cluster components)
- `backend/core/network/dwcp/federation_adapter.go` (DWCP v1 federation adapter)
- `backend/core/federation/multicloud/` (Multi-cloud support)

**Upgrade Tasks:**
1. Add DWCP v3 mode to federation:
   - Datacenter federation: Use existing v1 (RDMA, low-latency)
   - Internet federation: Use v3 (TCP, high-latency, Byzantine tolerance)
   - Multi-cloud federation: Use v3 for cross-cloud communication

2. Enhance existing cross-cluster components:
   - BandwidthOptimizer → Integrate AMST v3
   - AdaptiveCompressionEngine → Integrate HDE v3
   - BandwidthPredictionModel → Integrate PBA v3
   - StateSynchronizationProtocol → Integrate ASS v3

3. Add internet-scale discovery:
   - Add DHT for internet-scale node discovery (optional)
   - Add SWIM for membership (optional)
   - Keep existing service mesh for datacenter

**Files to Modify:**
- `backend/core/network/dwcp/federation_adapter.go` (add v3 support)
- `backend/core/federation/cross_cluster_components.go` (integrate v3 components)
- `backend/core/federation/federation_manager.go` (add mode detection)

**Integration Points:**
- `backend/core/network/dwcp/amst.go` (AMST v3)
- `backend/core/network/dwcp/hde.go` (HDE v3)
- `backend/core/network/dwcp/prediction/` (PBA v3)
- `backend/core/network/dwcp/sync/` (ASS v3)

Use hooks for coordination."

🔹 AGENT 10: Security Enhancement Engineer (security-manager)
Task: "Enhance existing security with DWCP v3 Byzantine tolerance:

**Existing Code to Enhance:**
- `backend/core/network/dwcp/security/` (DWCP v1 security)
- `backend/core/network/dwcp/security/tls_manager.go`
- `backend/core/network/dwcp/security/cert_manager.go`
- `backend/core/security/` (NovaCron security layer)

**Upgrade Tasks:**
1. Add Byzantine tolerance for internet mode:
   - Implement node reputation system
   - Add malicious node detection
   - Add proof-of-work for untrusted nodes (optional)

2. Enhance existing TLS/mTLS:
   - Keep existing TLS 1.3 for datacenter
   - Add additional verification for internet mode
   - Add certificate pinning for untrusted networks

3. Add mode-aware security:
   - Datacenter mode: Trust all nodes (existing)
   - Internet mode: Verify all nodes (new)
   - Hybrid mode: Adaptive trust (new)

**Files to Modify:**
- `backend/core/network/dwcp/security/tls_manager.go` (add v3 features)
- `backend/core/network/dwcp/security/byzantine_detector.go` (new Byzantine detection)
- `backend/core/network/dwcp/security/reputation_system.go` (new reputation tracking)

**Integration Points:**
- `backend/core/network/dwcp/consensus/pbft.go` (Byzantine consensus)
- `backend/core/security/` (NovaCron security integration)

Use hooks for security auditing."

🔹 AGENT 11: Monitoring Enhancement Engineer (perf-analyzer)
Task: "Enhance existing monitoring with DWCP v3 metrics:

**Existing Code to Enhance:**
- `backend/core/network/dwcp/monitoring/` (DWCP v1 monitoring)
- `backend/core/network/dwcp/monitoring/metrics_collector.go`
- `backend/core/network/dwcp/monitoring/anomaly_detector.go`
- `backend/core/network/dwcp/metrics/` (Metrics collection)

**Upgrade Tasks:**
1. Add DWCP v3 mode-specific metrics:
   - Datacenter mode: RDMA throughput, latency, packet loss
   - Internet mode: TCP throughput, compression ratio, Byzantine node count
   - Hybrid mode: Mode switching frequency, adaptive performance

2. Enhance existing anomaly detection:
   - Use existing `backend/core/network/dwcp/monitoring/anomaly_detector.go`
   - Add ML-based anomaly detection for v3 patterns
   - Train on existing monitoring data

3. Add neural pattern recognition:
   - Integrate with existing neural training
   - Predict performance bottlenecks
   - Auto-optimize based on patterns

**Files to Modify:**
- `backend/core/network/dwcp/monitoring/metrics_collector.go` (add v3 metrics)
- `backend/core/network/dwcp/monitoring/dwcp_v3_metrics.go` (new v3 metrics)
- `backend/core/network/dwcp/monitoring/neural_optimizer.go` (new neural optimization)

**Integration Points:**
- `backend/core/network/dwcp/metrics/` (Metrics integration)
- Existing Prometheus/Grafana dashboards

Use hooks for performance tracking."

🔹 AGENT 12: Test Engineer (tester)
Task: "Create comprehensive upgrade test suite:

**Test Strategy:**
1. **Backward Compatibility Tests:**
   - Ensure DWCP v1 still works after v3 upgrade
   - Test dual-mode operation (v1 and v3 simultaneously)
   - Test feature flag rollout

2. **Component Upgrade Tests:**
   - Test AMST v1 → v3 upgrade
   - Test HDE v1 → v3 upgrade
   - Test PBA v1 → v3 upgrade
   - Test ASS v1 → v3 upgrade
   - Test ITP v1 → v3 upgrade
   - Test ACP v1 → v3 upgrade

3. **Integration Tests:**
   - Test v3 with existing NovaCron components
   - Test v3 with existing federation
   - Test v3 with existing migration
   - Test v3 with existing multi-cloud

4. **Mode-Specific Tests:**
   - Datacenter mode tests (RDMA, high-bandwidth)
   - Internet mode tests (TCP, gigabit, Byzantine)
   - Hybrid mode tests (adaptive switching)

**Files to Create:**
- `backend/core/network/dwcp/tests/upgrade_test.go` (upgrade tests)
- `backend/core/network/dwcp/tests/backward_compat_test.go` (compatibility tests)
- `backend/core/network/dwcp/tests/mode_switching_test.go` (mode tests)
- `backend/core/network/dwcp/tests/v3_integration_test.go` (integration tests)

**Use Existing Test Infrastructure:**
- `backend/core/network/dwcp/testing/` (Test harness)
- `backend/core/network/dwcp/phase1_*_test.go` (Existing tests)

Target: 90%+ code coverage, all tests passing, zero regressions."

🔹 AGENT 13: Documentation Engineer (reviewer)
Task: "Create comprehensive upgrade documentation:

**Documentation Tasks:**
1. **Upgrade Guide:**
   - Create `backend/core/network/dwcp/UPGRADE_GUIDE_V1_TO_V3.md`
   - Step-by-step upgrade instructions
   - Rollback procedures
   - Troubleshooting guide

2. **API Documentation:**
   - Document new v3 APIs
   - Document backward compatibility
   - Document feature flags
   - Update existing GoDoc comments

3. **Architecture Documentation:**
   - Update `docs/architecture/distributed-wan-communication-protocol.md`
   - Add v3 architecture diagrams
   - Document mode detection logic
   - Document hybrid architecture

4. **Integration Documentation:**
   - Update `docs/DWCP-NOVACRON-INTEGRATION-ROADMAP.md`
   - Document v3 integration points
   - Document migration strategy
   - Document performance targets

**Files to Create/Update:**
- `backend/core/network/dwcp/UPGRADE_GUIDE_V1_TO_V3.md` (new)
- `backend/core/network/dwcp/README.md` (update)
- `docs/architecture/distributed-wan-communication-protocol.md` (update)
- `docs/DWCP-V3-ARCHITECTURE.md` (new)

Use hooks for documentation tracking."

STEP 4: BATCH ALL OPERATIONS (Single Message)
```bash
# Analyze existing DWCP v1.0 structure
cd backend/core/network/dwcp
find . -name "*.go" -type f | wc -l  # Count existing files
du -sh .                              # Check size

# Create v3 upgrade directories (alongside v1, not replacing)
mkdir -p backend/core/network/dwcp/v3/{transport,encoding,prediction,sync,partition,consensus,security,monitoring,tests}
mkdir -p backend/core/network/dwcp/upgrade/{migration,compatibility,feature_flags}

# Backup existing v1 implementation
cp -r backend/core/network/dwcp backend/core/network/dwcp.v1.backup

# Install new dependencies (if needed)
cd backend/core/network/dwcp
go get -u github.com/klauspost/compress/zstd &  # Zstandard (may already exist)
go get -u github.com/hashicorp/raft &           # Raft (may already exist)
go get -u github.com/hashicorp/memberlist &     # Gossip (may already exist)
wait

# Run neural training on EXISTING NovaCron codebase
npx claude-flow@alpha neural train \
  --patterns "dwcp-v1,federation,migration,consensus,multicloud" \
  --training-data "backend/core/network/dwcp/,backend/core/federation/,backend/core/migration/,backend/core/consensus/" \
  --target-accuracy 0.98 \
  --iterations 1000 \
  --export-model "novacron-dwcp-v1-to-v3-patterns.json"

# Run existing DWCP v1 tests (ensure no regressions)
cd backend/core/network/dwcp
go test -v -race -coverprofile=coverage_v1.out ./...

# Run new v3 upgrade tests
go test -v -race -coverprofile=coverage_v3.out ./v3/... ./upgrade/...

# Generate combined coverage report
go tool cover -html=coverage_v3.out -o coverage_v3.html

# Run benchmarks (compare v1 vs v3)
go test -bench=. -benchmem ./... > benchmark_v1_vs_v3.txt

# Verify backward compatibility
go test -v ./upgrade/compatibility/...

# Export session metrics
npx claude-flow@alpha hooks session-end --export-metrics true --session-id "novacron-dwcp-v1-to-v3-upgrade"
```

📊 TODOS (Batch in ONE call):
Create comprehensive task list with 20+ items covering:
1. ✅ Analyze existing DWCP v1.0 codebase
2. ✅ Create upgrade plan (v1 → v3)
3. ✅ Create migration strategy (backward compatibility)
4. ✅ Initialize upgrade swarm with hierarchical topology
5. ✅ Train neural models on existing NovaCron patterns (98% accuracy)
6. ✅ Upgrade AMST v1 → v3 (hybrid datacenter + internet)
7. ✅ Upgrade HDE v1 → v3 (ML-based compression + CRDT)
8. ✅ Upgrade PBA v1 → v3 (enhanced LSTM + hybrid mode)
9. ✅ Upgrade ASS v1 → v3 (mode-aware state sync + Byzantine tolerance)
10. ✅ Upgrade ACP v1 → v3 (adaptive consensus + PBFT)
11. ✅ Upgrade ITP v1 → v3 (enhanced ML placement + mode-aware)
12. ✅ Enhance migration with DWCP v3 support
13. ✅ Enhance federation with DWCP v3 support
14. ✅ Enhance security with Byzantine tolerance
15. ✅ Enhance monitoring with v3 metrics
16. ✅ Create comprehensive upgrade test suite (90%+ coverage)
17. ✅ Test backward compatibility (v1 still works)
18. ✅ Test dual-mode operation (v1 and v3 simultaneously)
19. ✅ Performance benchmarking (v1 vs v3)
20. ✅ Create upgrade documentation
21. ✅ Code review and optimization
22. ✅ Export neural model and session metrics

🎯 SUCCESS CRITERIA:

**1. Backward Compatibility:**
- ✅ DWCP v1.0 still works after upgrade (zero regressions)
- ✅ Dual-mode operation (v1 and v3 run simultaneously)
- ✅ Feature flags for gradual rollout
- ✅ Rollback capability

**2. Code Quality:**
- ✅ All 6 DWCP components upgraded (v1 → v3)
- ✅ 90%+ test coverage (including upgrade tests)
- ✅ All tests passing (v1 tests + v3 tests)
- ✅ Zero critical security vulnerabilities
- ✅ GoDoc comments on all new/modified APIs

**3. Performance (Hybrid Architecture):**
- ✅ **Datacenter Mode:** <500ms migration downtime, 10-100 Gbps, <10ms latency
- ✅ **Internet Mode:** 45-90s migration for 2GB VM, 100-900 Mbps, 70-85% compression
- ✅ **Hybrid Mode:** Adaptive mode switching, dynamic protocol selection
- ✅ Byzantine tolerance: 33% malicious nodes (internet mode)

**4. Neural Training:**
- ✅ 98% accuracy on NovaCron patterns (DWCP v1, federation, migration)
- ✅ Neural model exported as JSON
- ✅ Pattern recognition for optimization

**5. Integration:**
- ✅ DWCP v3 integrated with existing NovaCron (federation, migration, multi-cloud, AI/ML)

**6. Deliverables:**
- ✅ Upgraded DWCP in backend/core/network/dwcp/ (v1 still works)
- ✅ New v3 modules in backend/core/network/dwcp/v3/
- ✅ Upgrade docs: UPGRADE_PLAN, MIGRATION_STRATEGY, UPGRADE_GUIDE
- ✅ Neural model: novacron-dwcp-v1-to-v3-patterns.json
- ✅ Benchmarks: v1 vs v3 performance comparison

🧠 NEURAL TRAINING COMMANDS (Train on EXISTING NovaCron Patterns):
```bash
# Train on existing DWCP v1 patterns
npx claude-flow@alpha neural train \
  --patterns "dwcp-v1,amst,hde,consensus,federation" \
  --training-data "backend/core/network/dwcp/" \
  --target-accuracy 0.98

# Train on NovaCron migration patterns
npx claude-flow@alpha neural train \
  --patterns "live-migration,pre-copy,stop-and-copy,predictive-prefetching" \
  --training-data "backend/core/migration/,backend/core/vm/" \
  --target-accuracy 0.98

# Train on federation and multi-cloud patterns
npx claude-flow@alpha neural train \
  --patterns "federation,cross-cluster,multicloud,bandwidth-optimization" \
  --training-data "backend/core/federation/,backend/core/federation/multicloud/" \
  --target-accuracy 0.98

# Export trained models
npx claude-flow@alpha neural export --model "novacron-dwcp-v1-patterns.json"
npx claude-flow@alpha neural export --model "novacron-migration-patterns.json"
npx claude-flow@alpha neural export --model "novacron-federation-patterns.json"

# Verify accuracy
npx claude-flow@alpha neural status --show-accuracy
```

═══════════════════════════════════════════════════════════════════════════════
STEP 5: FINAL BEADS STATUS CHECK & CLAUDE-FLOW SESSION EXPORT (Single Message)
═══════════════════════════════════════════════════════════════════════════════

# Check all Beads tasks status
mcp__beads__list {
  workspace_root: "/home/kp/novacron",
  limit: 50
}

# Get statistics
mcp__beads__stats {
  workspace_root: "/home/kp/novacron"
}

# Check for any blocked tasks
mcp__beads__blocked {
  workspace_root: "/home/kp/novacron"
}

# Export Claude-Flow session metrics
npx claude-flow@alpha hooks session-end \
  --export-metrics true \
  --session-id "novacron-dwcp-v1-to-v3-upgrade" \
  --generate-summary true

# Export neural models
npx claude-flow@alpha neural export \
  --model "novacron-dwcp-v1-to-v3-patterns.json" \
  --include-metrics true

# Generate performance report
npx claude-flow@alpha benchmark run \
  --component "dwcp-v3" \
  --compare-baseline "dwcp-v1" \
  --export-report "dwcp-v1-vs-v3-benchmark.json"

# Check swarm status
npx claude-flow@alpha swarm status \
  --session-id "novacron-dwcp-v1-to-v3-upgrade" \
  --show-metrics true

🚀 EXECUTION RULES:
1. ⚡ **BEADS MCP**: Use for ALL task management (create, update, close, list, stats)
2. 🔄 **CLAUDE-FLOW HOOKS**: Use BEFORE, DURING, and AFTER each agent task
3. 🧠 **NEURAL TRAINING**: Train to 98% accuracy on NovaCron patterns
4. 📊 **SPARC TDD**: Use for all component upgrades (spec → pseudocode → architect → tdd → integration)
5. 🔧 **PARALLEL EXECUTION**: ALL operations in SINGLE messages
6. ✅ **TEST-FIRST**: Write tests BEFORE implementation
7. 📈 **METRICS**: Export session metrics and neural models at the end

💡 ADVANCED CLAUDE-FLOW FEATURES TO USE:

**1. Swarm Orchestration:**
```bash
npx claude-flow@alpha swarm init --topology hierarchical --max-agents 15
npx claude-flow@alpha swarm scale --agents 20  # Scale up if needed
npx claude-flow@alpha swarm status --show-metrics true
npx claude-flow@alpha swarm monitor --real-time true
```

**2. Neural Training & Pattern Recognition:**
```bash
npx claude-flow@alpha neural train --patterns "dwcp-v1,federation,migration" --target-accuracy 0.98
npx claude-flow@alpha neural status --show-accuracy
npx claude-flow@alpha neural patterns --analyze true
npx claude-flow@alpha neural export --model "novacron-patterns.json"
```

**3. SPARC TDD Workflow:**
```bash
npx claude-flow@alpha sparc tdd "AMST v3 upgrade"
npx claude-flow@alpha sparc run spec-pseudocode "AMST v3 requirements"
npx claude-flow@alpha sparc run architect "AMST v3 design"
npx claude-flow@alpha sparc run integration "AMST v3 integration"
npx claude-flow@alpha sparc pipeline "Complete DWCP v3 upgrade"
```

**4. Advanced Hooks:**
```bash
npx claude-flow@alpha hooks enable --pre-task --post-edit --post-task --auto-format --neural-train
npx claude-flow@alpha hooks pre-task --description "Task description" --task-id "DWCP-001"
npx claude-flow@alpha hooks post-edit --file "file.go" --auto-format true --neural-train true
npx claude-flow@alpha hooks post-task --task-id "DWCP-001" --status "completed"
npx claude-flow@alpha hooks session-restore --session-id "novacron-upgrade"
npx claude-flow@alpha hooks session-end --export-metrics true
```

**5. Performance & Benchmarking:**
```bash
npx claude-flow@alpha benchmark run --component "dwcp-v3"
npx claude-flow@alpha features detect --analyze-bottlenecks true
npx claude-flow@alpha swarm monitor --track-performance true
```

**6. GitHub Integration (Optional):**
```bash
npx claude-flow@alpha github swarm --repo "khryptorgraphics/novacron"
npx claude-flow@alpha repo analyze --path "/home/kp/novacron"
npx claude-flow@alpha pr enhance --branch "dwcp-v3-upgrade"
```

**7. Memory & State Management:**
```bash
npx claude-flow@alpha memory usage --show-stats true
npx claude-flow@alpha hooks session-restore --session-id "novacron-upgrade"
npx claude-flow@alpha hooks notify --message "Component upgrade complete"
```

🎯 FINAL DELIVERABLES:

**1. Code Deliverables:**
- ✅ Upgraded DWCP in `backend/core/network/dwcp/` (v1 still works)
- ✅ New v3 modules in `backend/core/network/dwcp/v3/`
- ✅ Upgrade utilities in `backend/core/network/dwcp/upgrade/`
- ✅ All 6 components upgraded (AMST, HDE, PBA, ASS, ITP, ACP)
- ✅ Integration with existing NovaCron (federation, migration, multi-cloud)

**2. Documentation Deliverables:**
- ✅ `backend/core/network/dwcp/UPGRADE_PLAN_V1_TO_V3.md`
- ✅ `backend/core/network/dwcp/MIGRATION_STRATEGY_V1_TO_V3.md`
- ✅ `backend/core/network/dwcp/UPGRADE_GUIDE_V1_TO_V3.md`
- ✅ Updated architecture docs
- ✅ GoDoc comments on all new/modified APIs

**3. Test Deliverables:**
- ✅ Comprehensive test suite (90%+ coverage)
- ✅ Backward compatibility tests (v1 still works)
- ✅ Dual-mode operation tests
- ✅ Integration tests with existing NovaCron
- ✅ Performance benchmarks (v1 vs v3)

**4. Neural & Metrics Deliverables:**
- ✅ `novacron-dwcp-v1-to-v3-patterns.json` (98% accuracy)
- ✅ `novacron-migration-patterns.json`
- ✅ `novacron-federation-patterns.json`
- ✅ Session metrics export
- ✅ Performance benchmark report

**5. Beads Project Tracking:**
- ✅ All 13 Beads tasks created and tracked
- ✅ Task dependencies managed
- ✅ Progress tracked in real-time
- ✅ Final statistics report

BEGIN IMPLEMENTATION NOW! 🚀
```

---

## 📋 BEADS + CLAUDE-FLOW INTEGRATION SUMMARY

### **Beads MCP Commands Used:**
```bash
# Project Management
mcp__beads__set_context { workspace_root: "/home/kp/novacron" }
mcp__beads__init { workspace_root: "/home/kp/novacron" }
mcp__beads__create { ... }  # Create 13 tasks (DWCP-001 to DWCP-013)
mcp__beads__update { issue_id: "DWCP-XXX", status: "in_progress" }
mcp__beads__close { issue_id: "DWCP-XXX", reason: "Complete" }
mcp__beads__list { workspace_root: "/home/kp/novacron" }
mcp__beads__stats { workspace_root: "/home/kp/novacron" }
mcp__beads__ready { workspace_root: "/home/kp/novacron" }
mcp__beads__blocked { workspace_root: "/home/kp/novacron" }
```

### **Claude-Flow Commands Used:**
```bash
# Swarm Orchestration
npx claude-flow@alpha swarm init --topology hierarchical --max-agents 15
npx claude-flow@alpha swarm status --show-metrics true

# Neural Training (98% accuracy)
npx claude-flow@alpha neural train --patterns "dwcp-v1,federation,migration" --target-accuracy 0.98
npx claude-flow@alpha neural export --model "novacron-patterns.json"

# SPARC TDD Workflow
npx claude-flow@alpha sparc tdd "AMST v3 upgrade"
npx claude-flow@alpha sparc run spec-pseudocode "Requirements"
npx claude-flow@alpha sparc run architect "Design"
npx claude-flow@alpha sparc run integration "Integration"

# Advanced Hooks
npx claude-flow@alpha hooks enable --pre-task --post-edit --post-task --auto-format --neural-train
npx claude-flow@alpha hooks pre-task --description "Task" --task-id "DWCP-XXX"
npx claude-flow@alpha hooks post-edit --file "file.go" --auto-format true --neural-train true
npx claude-flow@alpha hooks post-task --task-id "DWCP-XXX"
npx claude-flow@alpha hooks session-end --export-metrics true

# Performance & Benchmarking
npx claude-flow@alpha benchmark run --component "dwcp-v3"
npx claude-flow@alpha features detect --analyze-bottlenecks true
```

### **Claude Code Task Tool Used:**
- Spawns 13 specialized agents concurrently
- Each agent uses Beads for task tracking
- Each agent uses Claude-Flow hooks for coordination
- Each agent uses SPARC TDD for implementation

---

## 📋 COPY-PASTE CHECKLIST

Before sending to Claude-Code, verify:
- ✅ Beads MCP installed and configured
- ✅ Claude-Flow installed (`npx claude-flow@alpha`)
- ✅ Read architecture docs (DWCP v1 roadmap + v3 target)
- ✅ Understand 6 core components (AMST, HDE, PBA, ASS, ITP, ACP)
- ✅ Understand hybrid architecture (datacenter + internet modes)
- ✅ Neural training target: 98% accuracy on NovaCron patterns
- ✅ All operations in SINGLE messages (parallel execution)
- ✅ Use Beads for task management
- ✅ Use Claude-Flow hooks for coordination
- ✅ Use SPARC TDD for implementation
- ✅ Export metrics and neural models at end

---

## 🎉 SUMMARY

**This prompt uses the MOST ADVANCED orchestration available:**
- ✅ **Beads MCP**: Project management with 13 tracked tasks
- ✅ **Claude-Flow**: Swarm orchestration, neural training (98%), SPARC TDD
- ✅ **Claude Code Task Tool**: Parallel agent execution
- ✅ **Advanced Hooks**: Pre-task, post-edit, post-task, auto-format, neural-train
- ✅ **Hybrid Architecture**: Datacenter + internet modes
- ✅ **Backward Compatible**: DWCP v1 still works
- ✅ **Test-First**: SPARC TDD methodology
- ✅ **98% Neural Accuracy**: Trained on existing NovaCron patterns

**Ready to execute the most advanced DWCP v1 → v3 upgrade!** 🚀✨
