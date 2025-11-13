# 🚀 PHASE 2: QUALITY & STABILITY - Beads + Claude-Flow Prompt
## Production Readiness: 85/100 → 90/100 (Weeks 4-6)

---

## 🎯 MASTER PROMPT FOR CLAUDE-CODE

Copy and paste this ENTIRE prompt to Claude-Code:

```
🚨 PHASE 2 MISSION: NovaCron Quality & Stability Enhancement

📋 CONTEXT:
Phase 1 complete! Now achieving production-grade code quality and comprehensive testing.
Current Score: 85/100 (Phase 1 Complete)
Target Score: 90/100 (Phase 2 Complete)

🔍 CURRENT STATE:
- ✅ DWCP v3 tests: 90%+ coverage
- ✅ Security: 0 vulnerabilities
- ✅ Backend: Compiles successfully
- 🟡 Code quality: 178 TODO/FIXME markers
- 🟡 Test values: 819 hardcoded values
- 🔴 Load testing: Missing

📁 PROJECT ROOT: /home/kp/novacron

🎯 PHASE 2 OBJECTIVES:
1. Remove all TODO/FIXME markers (178 → 0)
2. Replace hardcoded test values (819 → 0)
3. Create comprehensive load testing suite
4. Increase backend test coverage (60% → 80%+)
5. Establish performance baseline

🧠 ORCHESTRATION: Beads MCP + Claude-Flow + SPARC TDD
📊 METHODOLOGY: Code Quality First
⏱️ TIMELINE: 3 weeks

⚡ EXECUTION STRATEGY (ALL IN SINGLE MESSAGES):

═══════════════════════════════════════════════════════════════════════════════
STEP 0: INITIALIZE BEADS PROJECT MANAGEMENT (Single Message)
═══════════════════════════════════════════════════════════════════════════════

# Initialize Beads for Phase 2 tracking
mcp__beads__set_context { workspace_root: "/home/kp/novacron" }

# Create Phase 2 epic
mcp__beads__create {
  workspace_root: "/home/kp/novacron",
  id: "novacron-READY-P2",
  title: "Phase 2: Quality & Stability Enhancement",
  description: "Achieve production-grade code quality: Remove TODO/FIXME, replace hardcoded values, create load tests, increase coverage. Target: 85/100 → 90/100",
  issue_type: "epic",
  priority: 1,
  assignee: "claude-code",
  labels: ["production-readiness", "phase-2", "quality", "stability"]
}

═══════════════════════════════════════════════════════════════════════════════
STEP 1: INITIALIZE CLAUDE-FLOW SWARM + NEURAL TRAINING (Single Message)
═══════════════════════════════════════════════════════════════════════════════

# Initialize hierarchical swarm for Phase 2
npx claude-flow@alpha swarm init \
  --topology hierarchical \
  --max-agents 12 \
  --enable-neural true \
  --neural-target-accuracy 0.98 \
  --enable-hooks true \
  --enable-memory true \
  --session-id "novacron-phase2-quality-stability" \
  --project-root "/home/kp/novacron"

# Train neural models on code quality patterns
npx claude-flow@alpha neural train \
  --patterns "code-cleanup,load-testing,test-coverage,performance-baseline" \
  --training-data "backend/,frontend/,tests/" \
  --target-accuracy 0.98 \
  --iterations 500 \
  --export-model "novacron-phase2-patterns.json"

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
  --project "novacron-phase2" \
  --methodology "tdd" \
  --enable-pipeline true

═══════════════════════════════════════════════════════════════════════════════
STEP 2: CREATE BEADS TASKS FOR QUALITY & STABILITY (Single Message)
═══════════════════════════════════════════════════════════════════════════════

# Task 1: Remove TODO/FIXME Markers
mcp__beads__create {
  workspace_root: "/home/kp/novacron",
  id: "novacron-006",
  title: "Remove All TODO/FIXME Markers (178 → 0)",
  description: "Systematically address all 178 TODO/FIXME markers in production code. Either implement the feature, create a proper task, or remove if obsolete. Zero markers in production code.",
  issue_type: "task",
  priority: 2,
  assignee: "code-quality-agent",
  labels: ["code-quality", "cleanup", "high"],
  deps: []
}

# Task 2: Replace Hardcoded Test Values
mcp__beads__create {
  workspace_root: "/home/kp/novacron",
  id: "novacron-007",
  title: "Replace Hardcoded Test Values (819 → 0)",
  description: "Replace all 819 hardcoded test values with proper configuration, environment variables, or constants. Ensure production code has no test artifacts.",
  issue_type: "task",
  priority: 2,
  assignee: "code-quality-agent",
  labels: ["code-quality", "cleanup", "high"],
  deps: []
}

# Task 3: Load Testing Suite
mcp__beads__create {
  workspace_root: "/home/kp/novacron",
  id: "novacron-008",
  title: "Create Comprehensive Load Testing Suite",
  description: "Create load tests for 1K, 10K, 100K VMs. Test API endpoints, DWCP protocol, database, WebSocket. Use k6 or Locust. Establish performance baselines.",
  issue_type: "task",
  priority: 1,
  assignee: "load-tester-agent",
  labels: ["load-testing", "performance", "critical"],
  deps: []
}

# Task 4: Backend Test Coverage
mcp__beads__create {
  workspace_root: "/home/kp/novacron",
  id: "novacron-009",
  title: "Increase Backend Test Coverage (60% → 80%+)",
  description: "Add unit tests for uncovered backend code. Focus on core services, API handlers, business logic. Target: 80%+ coverage with quality tests.",
  issue_type: "task",
  priority: 2,
  assignee: "backend-tester-agent",
  labels: ["testing", "coverage", "high"],
  deps: []
}

# Task 5: Performance Baseline
mcp__beads__create {
  workspace_root: "/home/kp/novacron",
  id: "novacron-010",
  title: "Establish Performance Baseline",
  description: "Run comprehensive performance benchmarks. Document baseline metrics for API latency, throughput, resource usage. Create performance regression tests.",
  issue_type: "task",
  priority: 2,
  assignee: "perf-optimizer-agent",
  labels: ["performance", "benchmarks", "medium"],
  deps: ["novacron-008"]
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

🔹 AGENT 1: Code Quality Engineer (code-quality)
Beads Tasks: novacron-006, novacron-007
Priority: HIGH
Task: "Clean up production code - remove TODO/FIXME and hardcoded values:

**BEFORE Starting:**
```bash
mcp__beads__update {
  workspace_root: \"/home/kp/novacron\",
  issue_id: \"novacron-006\",
  status: \"in_progress\"
}

npx claude-flow@alpha hooks pre-task \
  --description 'Code quality cleanup' \
  --task-id 'novacron-006,novacron-007'
```

**Code Cleanup Steps**:

1. **Find all TODO/FIXME markers**:
```bash
# Backend
grep -r \"TODO\\|FIXME\" backend/ --include=\"*.go\" > todo_fixme_list.txt

# Frontend
grep -r \"TODO\\|FIXME\" frontend/ --include=\"*.ts\" --include=\"*.tsx\" >> todo_fixme_list.txt
```

2. **Categorize and address**:
   - **Implement**: If quick fix (<30 min), implement now
   - **Create task**: If complex, create Beads task for Phase 3
   - **Remove**: If obsolete or already done, remove marker

3. **Find hardcoded test values**:
```bash
# Common patterns
grep -r \"test@example.com\\|localhost:3000\\|password123\\|admin123\" backend/ frontend/
```

4. **Replace with proper config**:
   - Move to environment variables
   - Use configuration files
   - Create constants file
   - Update tests to use fixtures

**AFTER Completing:**
```bash
# Verify zero markers
grep -r \"TODO\\|FIXME\" backend/ frontend/ --include=\"*.go\" --include=\"*.ts\" --include=\"*.tsx\"

npx claude-flow@alpha hooks post-task --task-id 'novacron-006'

mcp__beads__close {
  workspace_root: \"/home/kp/novacron\",
  issue_id: \"novacron-006\",
  reason: \"All TODO/FIXME markers addressed. Production code clean.\"
}

mcp__beads__close {
  workspace_root: \"/home/kp/novacron\",
  issue_id: \"novacron-007\",
  reason: \"All hardcoded test values replaced with proper configuration.\"
}
```"

🔹 AGENT 2: Load Testing Engineer (load-tester)
Beads Task: novacron-008
Priority: CRITICAL
Task: "Create comprehensive load testing suite:

**BEFORE Starting:**
```bash
mcp__beads__update {
  workspace_root: \"/home/kp/novacron\",
  issue_id: \"novacron-008\",
  status: \"in_progress\"
}

npx claude-flow@alpha hooks pre-task \
  --description 'Create load tests' \
  --task-id 'novacron-008'

npx claude-flow@alpha sparc tdd \"Load testing suite\"
```

**Load Test Implementation**:

1. **Install k6**:
```bash
# Install k6 for load testing
curl https://github.com/grafana/k6/releases/download/v0.48.0/k6-v0.48.0-linux-amd64.tar.gz -L | tar xvz
sudo mv k6-v0.48.0-linux-amd64/k6 /usr/local/bin/
```

2. **Create load test files**:

**tests/load/api_load_test.js** (API endpoints):
```javascript
import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  stages: [
    { duration: '2m', target: 100 },   // Ramp up to 100 users
    { duration: '5m', target: 100 },   // Stay at 100 users
    { duration: '2m', target: 1000 },  // Ramp up to 1K users
    { duration: '5m', target: 1000 },  // Stay at 1K users
    { duration: '2m', target: 0 },     // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p(95)<500'], // 95% requests < 500ms
    http_req_failed: ['rate<0.01'],   // Error rate < 1%
  },
};

export default function () {
  // Test VM list endpoint
  const res = http.get('http://localhost:8080/api/v1/vms');
  check(res, { 'status is 200': (r) => r.status === 200 });
  sleep(1);
}
```

**tests/load/dwcp_load_test.js** (DWCP protocol):
```javascript
// Test DWCP v3 under load
// 10K concurrent VM migrations
```

**tests/load/websocket_load_test.js** (WebSocket):
```javascript
// Test real-time updates
// 100K concurrent connections
```

3. **Run load tests**:
```bash
# 1K VMs
k6 run --vus 1000 --duration 10m tests/load/api_load_test.js

# 10K VMs
k6 run --vus 10000 --duration 10m tests/load/api_load_test.js

# 100K VMs (stress test)
k6 run --vus 100000 --duration 5m tests/load/api_load_test.js
```

**AFTER Completing:**
```bash
# Generate load test report
k6 run --out json=load_test_results.json tests/load/*.js

npx claude-flow@alpha hooks post-edit \
  --file 'tests/load/*.js' \
  --neural-train true

mcp__beads__close {
  workspace_root: \"/home/kp/novacron\",
  issue_id: \"novacron-008\",
  reason: \"Load tests complete. Passing for 1K, 10K, 100K VMs.\"
}
```"

🔹 AGENT 3: Backend Test Engineer (backend-tester)
Beads Task: novacron-009
Priority: HIGH
Task: "Increase backend test coverage to 80%+:

**BEFORE Starting:**
```bash
mcp__beads__update {
  workspace_root: \"/home/kp/novacron\",
  issue_id: \"novacron-009\",
  status: \"in_progress\"
}

npx claude-flow@alpha hooks pre-task \
  --description 'Increase test coverage' \
  --task-id 'novacron-009'
```

**Coverage Improvement Steps**:

1. **Analyze current coverage**:
```bash
cd backend
go test -coverprofile=coverage.out ./...
go tool cover -html=coverage.out -o coverage.html
go tool cover -func=coverage.out | grep total
```

2. **Identify uncovered code**:
```bash
# Find files with <80% coverage
go tool cover -func=coverage.out | awk '$3 < 80.0'
```

3. **Add tests for uncovered areas**:
   - API handlers (backend/api/)
   - Core services (backend/core/)
   - Business logic
   - Error handling paths
   - Edge cases

4. **Focus areas**:
   - `backend/api/vm/` - VM management
   - `backend/core/orchestration/` - Orchestration logic
   - `backend/core/federation/` - Federation
   - `backend/core/backup/` - Backup/restore

**AFTER Completing:**
```bash
# Verify 80%+ coverage
go test -coverprofile=coverage_final.out ./...
go tool cover -func=coverage_final.out | grep total

npx claude-flow@alpha hooks post-task --task-id 'novacron-009'

mcp__beads__close {
  workspace_root: \"/home/kp/novacron\",
  issue_id: \"novacron-009\",
  reason: \"Backend test coverage increased to 80%+.\"
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

# Get Phase 2 statistics
mcp__beads__stats {
  workspace_root: "/home/kp/novacron"
}

# Export Claude-Flow session metrics
npx claude-flow@alpha hooks session-end \
  --export-metrics true \
  --session-id "novacron-phase2-quality-stability" \
  --generate-summary true

# Export neural models
npx claude-flow@alpha neural export \
  --model "novacron-phase2-patterns.json" \
  --include-metrics true

# Run final validation
cd backend && go test -coverprofile=coverage.out ./...
k6 run tests/load/api_load_test.js

# Generate Phase 2 completion report
cat > docs/PHASE-2-COMPLETION-REPORT.md << 'EOF'
# Phase 2: Quality & Stability - Completion Report

**Status**: ✅ COMPLETE
**Score**: 90/100 (Target: 90/100)
**Duration**: 3 weeks

## Deliverables

✅ TODO/FIXME markers: 0 (was 178)
✅ Hardcoded test values: 0 (was 819)
✅ Load tests: PASSING (1K, 10K, 100K VMs)
✅ Backend test coverage: 80%+ (was 60%)
✅ Performance baseline: ESTABLISHED

## Next Steps

Proceed to Phase 3: Production Hardening
EOF

BEGIN IMPLEMENTATION NOW! 🚀
```

---

## 📋 EXECUTION CHECKLIST

Before running this prompt:
- ✅ Phase 1 complete (Score: 85/100)
- ✅ Beads MCP configured
- ✅ Claude-Flow installed
- ✅ Review Phase 1 completion report

---

## 🎯 SUCCESS CRITERIA

**Phase 2 Complete When**:
- ✅ Zero TODO/FIXME in production code
- ✅ Zero hardcoded test values
- ✅ Load tests passing (1K, 10K, 100K VMs)
- ✅ Backend test coverage: 80%+
- ✅ Performance baseline documented
- ✅ Score: 90/100

**Ready to execute Phase 2!** 🚀

