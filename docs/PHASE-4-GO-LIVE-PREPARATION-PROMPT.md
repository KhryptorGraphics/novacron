# 🚀 PHASE 4: GO-LIVE PREPARATION - Beads + Claude-Flow Prompt
## Production Readiness: 93/100 → 95/100 (Weeks 10-12)

---

## 🎯 MASTER PROMPT FOR CLAUDE-CODE

Copy and paste this ENTIRE prompt to Claude-Code:

```
🚨 PHASE 4 MISSION: NovaCron Final Validation & Go-Live Preparation

📋 CONTEXT:
Phase 3 complete! Final validation before production deployment.
Current Score: 93/100 (Phase 3 Complete)
Target Score: 95/100 (PRODUCTION READY)

🔍 CURRENT STATE:
- ✅ Observability: 95% coverage
- ✅ Deployment: Fully automated
- ✅ DR: Tested and documented
- ✅ Performance: Optimized
- ✅ Runbooks: Complete
- 🔴 Production simulation: Not tested
- 🔴 Chaos engineering: Not validated
- 🔴 Security audit: Pending

📁 PROJECT ROOT: /home/kp/novacron

🎯 PHASE 4 OBJECTIVES:
1. Production simulation testing
2. Chaos engineering validation
3. Final security audit
4. Staged rollout plan
5. Go/No-Go decision

🧠 ORCHESTRATION: Beads MCP + Claude-Flow + Production Validation
📊 METHODOLOGY: Comprehensive Validation
⏱️ TIMELINE: 3 weeks

⚡ EXECUTION STRATEGY (ALL IN SINGLE MESSAGES):

═══════════════════════════════════════════════════════════════════════════════
STEP 0: INITIALIZE BEADS PROJECT MANAGEMENT (Single Message)
═══════════════════════════════════════════════════════════════════════════════

# Initialize Beads for Phase 4 tracking
mcp__beads__set_context { workspace_root: "/home/kp/novacron" }

# Create Phase 4 epic
mcp__beads__create {
  workspace_root: "/home/kp/novacron",
  id: "novacron-READY-P4",
  title: "Phase 4: Go-Live Preparation & Final Validation",
  description: "Final validation before production: Simulation testing, chaos engineering, security audit, rollout plan. Target: 93/100 → 95/100 (PRODUCTION READY)",
  issue_type: "epic",
  priority: 1,
  assignee: "claude-code",
  labels: ["production-readiness", "phase-4", "go-live", "validation"]
}

═══════════════════════════════════════════════════════════════════════════════
STEP 1: INITIALIZE CLAUDE-FLOW SWARM + NEURAL TRAINING (Single Message)
═══════════════════════════════════════════════════════════════════════════════

# Initialize hierarchical swarm for Phase 4
npx claude-flow@alpha swarm init \
  --topology hierarchical \
  --max-agents 18 \
  --enable-neural true \
  --neural-target-accuracy 0.98 \
  --enable-hooks true \
  --enable-memory true \
  --session-id "novacron-phase4-go-live" \
  --project-root "/home/kp/novacron"

# Train neural models on validation patterns
npx claude-flow@alpha neural train \
  --patterns "production-simulation,chaos-engineering,security-audit,rollout-planning" \
  --training-data "tests/,backend/,deployment/" \
  --target-accuracy 0.98 \
  --iterations 500 \
  --export-model "novacron-phase4-patterns.json"

# Enable advanced hooks
npx claude-flow@alpha hooks enable \
  --pre-task true \
  --post-edit true \
  --post-task true \
  --session-restore true \
  --auto-format true \
  --neural-train true

# Initialize SPARC workflow
npx claude-flow@alpha sparc init \
  --project "novacron-phase4" \
  --methodology "validation" \
  --enable-pipeline true

═══════════════════════════════════════════════════════════════════════════════
STEP 2: CREATE BEADS TASKS FOR GO-LIVE PREPARATION (Single Message)
═══════════════════════════════════════════════════════════════════════════════

# Task 1: Production Simulation Testing
mcp__beads__create {
  workspace_root: "/home/kp/novacron",
  id: "novacron-016",
  title: "Production Simulation Testing",
  description: "Run full production simulation: 100K VMs, real workloads, 7-day continuous operation. Validate all systems under production conditions.",
  issue_type: "task",
  priority: 1,
  assignee: "qa-engineer",
  labels: ["simulation", "testing", "critical"],
  deps: []
}

# Task 2: Chaos Engineering Validation
mcp__beads__create {
  workspace_root: "/home/kp/novacron",
  id: "novacron-017",
  title: "Chaos Engineering Validation",
  description: "Run chaos tests: Network failures, node crashes, database failures, resource exhaustion. Validate resilience and auto-recovery.",
  issue_type: "task",
  priority: 1,
  assignee: "chaos-engineer",
  labels: ["chaos-engineering", "resilience", "critical"],
  deps: []
}

# Task 3: Final Security Audit
mcp__beads__create {
  workspace_root: "/home/kp/novacron",
  id: "novacron-018",
  title: "Final Security Audit",
  description: "Comprehensive security audit: Penetration testing, vulnerability scan, compliance check, security review. Zero critical/high issues.",
  issue_type: "task",
  priority: 1,
  assignee: "security-auditor",
  labels: ["security", "audit", "critical"],
  deps: []
}

# Task 4: Staged Rollout Plan
mcp__beads__create {
  workspace_root: "/home/kp/novacron",
  id: "novacron-019",
  title: "Create Staged Rollout Plan",
  description: "Design staged rollout: 1% → 10% → 50% → 100%. Define success criteria, rollback triggers, monitoring. Get stakeholder approval.",
  issue_type: "task",
  priority: 1,
  assignee: "release-manager",
  labels: ["rollout", "planning", "critical"],
  deps: ["novacron-016", "novacron-017", "novacron-018"]
}

# Task 5: Go/No-Go Decision
mcp__beads__create {
  workspace_root: "/home/kp/novacron",
  id: "novacron-020",
  title: "Go/No-Go Decision",
  description: "Final go/no-go decision: Review all validation results, stakeholder sign-off, production readiness checklist. Decision: GO or NO-GO.",
  issue_type: "task",
  priority: 1,
  assignee: "stakeholder-coordinator",
  labels: ["go-no-go", "decision", "critical"],
  deps: ["novacron-019"]
}

# List ready tasks
mcp__beads__ready {
  workspace_root: "/home/kp/novacron",
  limit: 10
}

BEGIN IMPLEMENTATION NOW! 🚀
```

---

## 📋 EXECUTION CHECKLIST

Before running this prompt:
- ✅ Phase 3 complete (Score: 93/100)
- ✅ All infrastructure ready
- ✅ Stakeholders identified
- ✅ Production environment prepared

---

## 🎯 SUCCESS CRITERIA

**Phase 4 Complete When**:
- ✅ Production simulation: PASSED (7 days)
- ✅ Chaos tests: PASSED (all scenarios)
- ✅ Security audit: CLEAN (zero critical/high)
- ✅ Rollout plan: APPROVED
- ✅ Go/No-Go: **GO FOR PRODUCTION**
- ✅ Score: 95/100 - **PRODUCTION READY**

**Ready to execute Phase 4!** 🚀

