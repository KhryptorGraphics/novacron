# 🚀 PHASE 1: CRITICAL BLOCKERS - Beads + Claude-Flow Prompt
## Production Readiness: 78/100 → 85/100 (Weeks 1-3)

---

## 🎯 MASTER PROMPT FOR CLAUDE-CODE

Copy and paste this ENTIRE prompt to Claude-Code:

```
🚨 CRITICAL MISSION: NovaCron Phase 1 - Critical Blockers Resolution

📋 CONTEXT:
You are fixing CRITICAL production-blocking issues in NovaCron to enable production deployment.
Current Score: 78/100 (CONDITIONAL GO)
Target Score: 85/100 (Phase 1 Complete)

🔍 CURRENT STATE:
- ✅ Frontend: 88/100 - Production ready
- 🔴 Backend: 65/100 - CRITICAL ISSUES
- 🔴 DWCP v3: 0% Go test coverage (BLOCKER)
- 🔴 Security: 5 high-severity vulnerabilities
- 🔴 Compilation: Import cycle errors

📁 PROJECT ROOT: /home/kp/novacron

🎯 PHASE 1 OBJECTIVES:
1. Create DWCP v3 Go test suite (0% → 90%+)
2. Fix all security vulnerabilities (5 → 0)
3. Resolve backend compilation errors
4. Fix import cycles
5. Validate DWCP v3 benchmarks

🧠 ORCHESTRATION: Beads MCP + Claude-Flow + SPARC TDD
📊 METHODOLOGY: Test-First Development
⏱️ TIMELINE: 3 weeks

⚡ EXECUTION STRATEGY (ALL IN SINGLE MESSAGES):

═══════════════════════════════════════════════════════════════════════════════
STEP 0: INITIALIZE BEADS PROJECT MANAGEMENT (Single Message)
═══════════════════════════════════════════════════════════════════════════════

# Initialize Beads for Phase 1 tracking
mcp__beads__set_context { workspace_root: "/home/kp/novacron" }

# Create Phase 1 epic
mcp__beads__create {
  workspace_root: "/home/kp/novacron",
  id: "novacron-READY-P1",
  title: "Phase 1: Critical Blockers Resolution",
  description: "Fix all production-blocking issues: DWCP v3 tests, security vulnerabilities, compilation errors, import cycles. Target: 78/100 → 85/100",
  issue_type: "epic",
  priority: 1,
  assignee: "claude-code",
  labels: ["production-readiness", "phase-1", "critical", "blockers"]
}

═══════════════════════════════════════════════════════════════════════════════
STEP 1: INITIALIZE CLAUDE-FLOW SWARM + NEURAL TRAINING (Single Message)
═══════════════════════════════════════════════════════════════════════════════

# Initialize hierarchical swarm for Phase 1
npx claude-flow@alpha swarm init \
  --topology hierarchical \
  --max-agents 10 \
  --enable-neural true \
  --neural-target-accuracy 0.98 \
  --enable-hooks true \
  --enable-memory true \
  --session-id "novacron-phase1-critical-blockers" \
  --project-root "/home/kp/novacron"

# Train neural models on existing test patterns
npx claude-flow@alpha neural train \
  --patterns "go-testing,security-fixes,import-resolution,dwcp-v3" \
  --training-data "backend/tests/,backend/core/network/dwcp/,backend/core/security/" \
  --target-accuracy 0.98 \
  --iterations 500 \
  --export-model "novacron-phase1-patterns.json"

# Enable advanced hooks
npx claude-flow@alpha hooks enable \
  --pre-task true \
  --post-edit true \
  --post-task true \
  --session-restore true \
  --auto-format true \
  --neural-train true

# Initialize SPARC TDD workflow
npx claude-flow@alpha sparc init \
  --project "novacron-phase1" \
  --methodology "tdd" \
  --enable-pipeline true

═══════════════════════════════════════════════════════════════════════════════
STEP 2: CREATE BEADS TASKS FOR ALL CRITICAL BLOCKERS (Single Message)
═══════════════════════════════════════════════════════════════════════════════

# Task 1: DWCP v3 Go Tests (CRITICAL BLOCKER)
mcp__beads__create {
  workspace_root: "/home/kp/novacron",
  id: "novacron-001",
  title: "Create DWCP v3 Go Test Suite (0% → 90%+)",
  description: "Create comprehensive Go tests for DWCP v3 components: AMST (bandwidth >70%), HDE (compression >5x), PBA (CPU <30%), ASS/ACP (consensus), ITP (placement). Target: 90%+ coverage with benchmarks.",
  issue_type: "task",
  priority: 1,
  assignee: "tester-agent",
  labels: ["dwcp-v3", "testing", "blocker", "critical"],
  deps: []
}

# Task 2: Security Vulnerabilities (CRITICAL BLOCKER)
mcp__beads__create {
  workspace_root: "/home/kp/novacron",
  id: "novacron-002",
  title: "Fix All Security Vulnerabilities (5 → 0)",
  description: "Fix 5 high-severity security vulnerabilities in dependencies. Run security scan, update vulnerable packages, validate fixes. Zero vulnerabilities required for production.",
  issue_type: "task",
  priority: 1,
  assignee: "security-manager-agent",
  labels: ["security", "vulnerabilities", "blocker", "critical"],
  deps: []
}

# Task 3: Backend Compilation Errors (CRITICAL BLOCKER)
mcp__beads__create {
  workspace_root: "/home/kp/novacron",
  id: "novacron-003",
  title: "Resolve Backend Compilation Errors",
  description: "Fix backend compilation errors. Ensure 'go build' succeeds for all cmd/ entry points (api-server, core-server). Validate all imports resolve correctly.",
  issue_type: "task",
  priority: 1,
  assignee: "backend-dev-agent",
  labels: ["backend", "compilation", "blocker", "critical"],
  deps: []
}

# Task 4: Import Cycle Resolution (CRITICAL BLOCKER)
mcp__beads__create {
  workspace_root: "/home/kp/novacron",
  id: "novacron-004",
  title: "Fix Import Cycles (backend/api/federation ↔ backend/core/backup)",
  description: "Resolve import cycle: backend/api/federation → backend/core/federation → backend/core/backup → backend/api/federation. Refactor to eliminate circular dependencies.",
  issue_type: "task",
  priority: 1,
  assignee: "code-analyzer-agent",
  labels: ["import-cycle", "refactoring", "blocker", "critical"],
  deps: ["novacron-003"]
}

# Task 5: DWCP v3 Benchmark Validation (HIGH PRIORITY)
mcp__beads__create {
  workspace_root: "/home/kp/novacron",
  id: "novacron-005",
  title: "Validate DWCP v3 Phase 0 Benchmarks",
  description: "Run and validate DWCP v3 Phase 0 benchmarks: AMST bandwidth >70%, HDE compression >5x, PBA CPU <30%. Document results and compare against targets.",
  issue_type: "task",
  priority: 2,
  assignee: "perf-analyzer-agent",
  labels: ["dwcp-v3", "benchmarks", "validation", "high"],
  deps: ["novacron-001"]
}

# List ready tasks
mcp__beads__ready {
  workspace_root: "/home/kp/novacron",
  limit: 10
}

═══════════════════════════════════════════════════════════════════════════════
STEP 3: PARALLEL AGENT EXECUTION WITH CLAUDE CODE TASK TOOL (Single Message)
═══════════════════════════════════════════════════════════════════════════════

Execute ALL agents concurrently using Claude Code's Task tool:

🔹 AGENT 1: DWCP v3 Test Engineer (tester)
Beads Task: novacron-001
Priority: CRITICAL
Task: "Create comprehensive DWCP v3 Go test suite with 90%+ coverage:

**BEFORE Starting:**
```bash
mcp__beads__update {
  workspace_root: \"/home/kp/novacron\",
  issue_id: \"novacron-001\",
  status: \"in_progress\"
}

npx claude-flow@alpha hooks pre-task \
  --description 'Create DWCP v3 Go tests' \
  --task-id 'novacron-001' \
  --session-id 'novacron-phase1-critical-blockers'

npx claude-flow@alpha sparc tdd \"DWCP v3 Go test suite\"
```

**Test Files to Create** (Test-First Development):

1. **backend/core/network/dwcp/v3/amst_test.go**
   - Test multi-stream TCP (4-16 streams)
   - Test bandwidth optimization (>70% improvement)
   - Test mode detection (datacenter/internet/hybrid)
   - Test congestion control
   - Benchmark: `BenchmarkAMST_Bandwidth`

2. **backend/core/network/dwcp/v3/hde_test.go**
   - Test delta encoding (>5x compression)
   - Test ML-based compression selection
   - Test CRDT integration
   - Test dictionary training
   - Benchmark: `BenchmarkHDE_Compression`

3. **backend/core/network/dwcp/v3/pba_test.go**
   - Test LSTM bandwidth prediction
   - Test CPU overhead (<30%)
   - Test multi-mode prediction
   - Test prediction accuracy (>85%)
   - Benchmark: `BenchmarkPBA_Prediction`

4. **backend/core/network/dwcp/v3/ass_test.go**
   - Test state synchronization
   - Test CRDT conflict resolution
   - Test eventual consistency
   - Test mode-aware sync

5. **backend/core/network/dwcp/v3/acp_test.go**
   - Test adaptive consensus
   - Test PBFT Byzantine tolerance
   - Test Raft/EPaxos switching
   - Test consensus performance

6. **backend/core/network/dwcp/v3/itp_test.go**
   - Test ML-based VM placement
   - Test DQN agent decisions
   - Test geographic optimization
   - Test mode-aware placement

**Test Coverage Requirements**:
- Unit tests: 90%+ coverage
- Integration tests: All components
- Benchmark tests: Performance validation
- Table-driven tests: Edge cases

**AFTER Completing:**
```bash
# Run all tests
go test -v -race -coverprofile=coverage_dwcp_v3.out ./backend/core/network/dwcp/v3/...

# Generate coverage report
go tool cover -html=coverage_dwcp_v3.out -o coverage_dwcp_v3.html

# Run benchmarks
go test -bench=. -benchmem ./backend/core/network/dwcp/v3/... > benchmarks_dwcp_v3.txt

npx claude-flow@alpha hooks post-edit \
  --file 'backend/core/network/dwcp/v3/*_test.go' \
  --auto-format true \
  --neural-train true

mcp__beads__close {
  workspace_root: \"/home/kp/novacron\",
  issue_id: \"novacron-001\",
  reason: \"DWCP v3 tests complete with 90%+ coverage.\"
}
```"

🔹 AGENT 2: Security Engineer (security-manager)
Beads Task: novacron-002
Priority: CRITICAL
Task: "Fix all 5 high-severity security vulnerabilities:

**BEFORE Starting:**
```bash
mcp__beads__update {
  workspace_root: \"/home/kp/novacron\",
  issue_id: \"novacron-002\",
  status: \"in_progress\"
}

npx claude-flow@alpha hooks pre-task \
  --description 'Fix security vulnerabilities' \
  --task-id 'novacron-002'
```

**Security Scan & Fix**:

1. **Run security scan**:
```bash
# Go dependencies
cd backend && go list -json -m all | nancy sleuth

# Frontend dependencies
cd frontend && npm audit --production

# Docker images
trivy image novacron:latest
```

2. **Fix vulnerabilities**:
   - Update vulnerable Go modules
   - Update vulnerable npm packages
   - Patch Docker base images
   - Validate fixes with re-scan

3. **Security hardening**:
   - Replace weak passwords (AUTH_SECRET, GRAFANA_ADMIN_PASSWORD, REDIS_PASSWORD)
   - Enable TLS/SSL in production
   - Configure proper CORS origins
   - Implement rate limiting

**AFTER Completing:**
```bash
# Verify zero vulnerabilities
go list -json -m all | nancy sleuth
npm audit --production

npx claude-flow@alpha hooks post-task --task-id 'novacron-002'

mcp__beads__close {
  workspace_root: \"/home/kp/novacron\",
  issue_id: \"novacron-002\",
  reason: \"All security vulnerabilities fixed. Zero high-severity issues.\"
}
```"

🔹 AGENT 3: Backend Engineer (backend-dev)
Beads Task: novacron-003
Priority: CRITICAL
Task: "Resolve backend compilation errors:

**BEFORE Starting:**
```bash
mcp__beads__update {
  workspace_root: \"/home/kp/novacron\",
  issue_id: \"novacron-003\",
  status: \"in_progress\"
}

npx claude-flow@alpha hooks pre-task \
  --description 'Fix backend compilation' \
  --task-id 'novacron-003'
```

**Compilation Fix Steps**:

1. **Test current compilation**:
```bash
cd backend/cmd/api-server && go build -o /tmp/api-server
cd backend/cmd/core-server && go build -o /tmp/core-server
```

2. **Fix missing imports**:
   - Verify all import paths
   - Add missing dependencies to go.mod
   - Run `go mod tidy`

3. **Fix type errors**:
   - Resolve undefined types
   - Fix interface mismatches
   - Correct function signatures

4. **Validate build**:
```bash
cd backend && go build ./...
```

**AFTER Completing:**
```bash
# Verify all entry points compile
cd backend/cmd/api-server && go build
cd backend/cmd/core-server && go build

npx claude-flow@alpha hooks post-task --task-id 'novacron-003'

mcp__beads__close {
  workspace_root: \"/home/kp/novacron\",
  issue_id: \"novacron-003\",
  reason: \"Backend compiles successfully. All entry points working.\"
}
```"

🔹 AGENT 4: Code Analyzer (code-analyzer)
Beads Task: novacron-004
Dependencies: novacron-003
Priority: CRITICAL
Task: "Fix import cycle: backend/api/federation ↔ backend/core/backup:

**BEFORE Starting:**
```bash
# Wait for novacron-003
mcp__beads__show { workspace_root: \"/home/kp/novacron\", issue_id: \"novacron-003\" }

mcp__beads__update {
  workspace_root: \"/home/kp/novacron\",
  issue_id: \"novacron-004\",
  status: \"in_progress\"
}

npx claude-flow@alpha hooks pre-task \
  --description 'Fix import cycles' \
  --task-id 'novacron-004' \
  --depends-on 'novacron-003'
```

**Import Cycle Analysis**:

1. **Identify cycle**:
```bash
cd backend && go build ./... 2>&1 | grep "import cycle"
```

Current cycle:
```
backend/api/federation → backend/core/federation
backend/core/federation → backend/core/backup
backend/core/backup → backend/api/federation
```

2. **Refactoring strategy**:
   - Extract shared interfaces to `backend/pkg/interfaces/`
   - Move common types to `backend/pkg/types/`
   - Use dependency injection
   - Apply interface segregation

3. **Implementation**:
   - Create `backend/pkg/interfaces/federation.go`
   - Create `backend/pkg/interfaces/backup.go`
   - Refactor imports to use interfaces
   - Remove circular dependencies

**AFTER Completing:**
```bash
# Verify no import cycles
cd backend && go build ./...

npx claude-flow@alpha hooks post-task --task-id 'novacron-004'

mcp__beads__close {
  workspace_root: \"/home/kp/novacron\",
  issue_id: \"novacron-004\",
  reason: \"Import cycles resolved. Backend builds cleanly.\"
}
```"

🔹 AGENT 5: Performance Analyzer (perf-analyzer)
Beads Task: novacron-005
Dependencies: novacron-001
Priority: HIGH
Task: "Validate DWCP v3 Phase 0 benchmarks:

**BEFORE Starting:**
```bash
# Wait for novacron-001
mcp__beads__show { workspace_root: \"/home/kp/novacron\", issue_id: \"novacron-001\" }

mcp__beads__update {
  workspace_root: \"/home/kp/novacron\",
  issue_id: \"novacron-005\",
  status: \"in_progress\"
}

npx claude-flow@alpha hooks pre-task \
  --description 'Validate DWCP v3 benchmarks' \
  --task-id 'novacron-005' \
  --depends-on 'novacron-001'
```

**Benchmark Validation**:

1. **Run DWCP v3 benchmarks**:
```bash
cd backend/core/network/dwcp/v3

# AMST bandwidth benchmark (target: >70% improvement)
go test -bench=BenchmarkAMST_Bandwidth -benchmem -benchtime=10s

# HDE compression benchmark (target: >5x compression)
go test -bench=BenchmarkHDE_Compression -benchmem -benchtime=10s

# PBA CPU overhead benchmark (target: <30% CPU)
go test -bench=BenchmarkPBA_Prediction -benchmem -benchtime=10s
```

2. **Validate targets**:
   - AMST: Bandwidth improvement >70% ✅
   - HDE: Compression ratio >5x ✅
   - PBA: CPU overhead <30% ✅
   - ASS: Sync latency <100ms ✅
   - ACP: Consensus time <500ms ✅
   - ITP: Placement accuracy >90% ✅

3. **Document results**:
   - Create `backend/core/network/dwcp/v3/BENCHMARK_RESULTS.md`
   - Compare against Phase 0 targets
   - Identify optimization opportunities

**AFTER Completing:**
```bash
# Generate benchmark report
go test -bench=. -benchmem ./backend/core/network/dwcp/v3/... > BENCHMARK_RESULTS.txt

npx claude-flow@alpha hooks post-edit \
  --file 'backend/core/network/dwcp/v3/BENCHMARK_RESULTS.md' \
  --neural-train true

mcp__beads__close {
  workspace_root: \"/home/kp/novacron\",
  issue_id: \"novacron-005\",
  reason: \"DWCP v3 benchmarks validated. All targets met.\"
}
```"

═══════════════════════════════════════════════════════════════════════════════
STEP 4: FINAL VALIDATION & METRICS EXPORT (Single Message)
═══════════════════════════════════════════════════════════════════════════════

# Check all Beads tasks status
mcp__beads__list {
  workspace_root: "/home/kp/novacron",
  limit: 20
}

# Get Phase 1 statistics
mcp__beads__stats {
  workspace_root: "/home/kp/novacron"
}

# Export Claude-Flow session metrics
npx claude-flow@alpha hooks session-end \
  --export-metrics true \
  --session-id "novacron-phase1-critical-blockers" \
  --generate-summary true

# Export neural models
npx claude-flow@alpha neural export \
  --model "novacron-phase1-patterns.json" \
  --include-metrics true

# Run final validation
cd backend && go test ./...
cd frontend && npm test

# Generate Phase 1 completion report
cat > docs/PHASE-1-COMPLETION-REPORT.md << 'EOF'
# Phase 1: Critical Blockers - Completion Report

**Status**: ✅ COMPLETE
**Score**: 85/100 (Target: 85/100)
**Duration**: 3 weeks

## Deliverables

✅ DWCP v3 test coverage: 90%+
✅ Security vulnerabilities: 0 (was 5)
✅ Backend compilation: SUCCESS
✅ Import cycles: RESOLVED
✅ Benchmarks: VALIDATED

## Next Steps

Proceed to Phase 2: Quality & Stability
EOF

BEGIN IMPLEMENTATION NOW! 🚀
```

---

## 📋 EXECUTION CHECKLIST

Before running this prompt:
- ✅ Beads MCP installed and configured
- ✅ Claude-Flow installed (`npx claude-flow@alpha`)
- ✅ Review Production Validation Scorecard
- ✅ Understand critical blockers
- ✅ Backend and frontend environments ready

---

## 🎯 SUCCESS CRITERIA

**Phase 1 Complete When**:
- ✅ DWCP v3 test coverage: 90%+
- ✅ Zero high-severity vulnerabilities
- ✅ Backend compiles successfully
- ✅ All import cycles resolved
- ✅ Benchmarks validated
- ✅ Score: 85/100

**Ready to execute Phase 1!** 🚀

