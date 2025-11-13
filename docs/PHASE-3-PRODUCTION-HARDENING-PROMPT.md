# 🚀 PHASE 3: PRODUCTION HARDENING - Beads + Claude-Flow Prompt
## Production Readiness: 90/100 → 93/100 (Weeks 7-9)

---

## 🎯 MASTER PROMPT FOR CLAUDE-CODE

Copy and paste this ENTIRE prompt to Claude-Code:

```
🚨 PHASE 3 MISSION: NovaCron Production Infrastructure Hardening

📋 CONTEXT:
Phase 2 complete! Now hardening production infrastructure for 99.9% uptime.
Current Score: 90/100 (Phase 2 Complete)
Target Score: 93/100 (Phase 3 Complete)

🔍 CURRENT STATE:
- ✅ Code quality: A grade (zero TODO/FIXME)
- ✅ Load tests: Passing (1K, 10K, 100K VMs)
- ✅ Test coverage: 80%+
- 🟡 Observability: 60% (needs 95%)
- 🔴 Deployment: Manual (needs automation)
- 🔴 DR: Not tested

📁 PROJECT ROOT: /home/kp/novacron

🎯 PHASE 3 OBJECTIVES:
1. Complete observability stack (60% → 95%)
2. Automate deployment pipeline
3. Test and document disaster recovery
4. Optimize performance
5. Create production runbooks

🧠 ORCHESTRATION: Beads MCP + Claude-Flow + SRE Best Practices
📊 METHODOLOGY: Infrastructure as Code
⏱️ TIMELINE: 3 weeks

⚡ EXECUTION STRATEGY (ALL IN SINGLE MESSAGES):

═══════════════════════════════════════════════════════════════════════════════
STEP 0: INITIALIZE BEADS PROJECT MANAGEMENT (Single Message)
═══════════════════════════════════════════════════════════════════════════════

# Initialize Beads for Phase 3 tracking
mcp__beads__set_context { workspace_root: "/home/kp/novacron" }

# Create Phase 3 epic
mcp__beads__create {
  workspace_root: "/home/kp/novacron",
  id: "novacron-READY-P3",
  title: "Phase 3: Production Infrastructure Hardening",
  description: "Harden production infrastructure: Complete observability, automate deployment, test DR, optimize performance, create runbooks. Target: 90/100 → 93/100",
  issue_type: "epic",
  priority: 1,
  assignee: "claude-code",
  labels: ["production-readiness", "phase-3", "infrastructure", "hardening"]
}

═══════════════════════════════════════════════════════════════════════════════
STEP 1: INITIALIZE CLAUDE-FLOW SWARM + NEURAL TRAINING (Single Message)
═══════════════════════════════════════════════════════════════════════════════

# Initialize hierarchical swarm for Phase 3
npx claude-flow@alpha swarm init \
  --topology hierarchical \
  --max-agents 15 \
  --enable-neural true \
  --neural-target-accuracy 0.98 \
  --enable-hooks true \
  --enable-memory true \
  --session-id "novacron-phase3-production-hardening" \
  --project-root "/home/kp/novacron"

# Train neural models on SRE patterns
npx claude-flow@alpha neural train \
  --patterns "observability,deployment-automation,disaster-recovery,performance-optimization" \
  --training-data "backend/monitoring/,deployment/,docs/" \
  --target-accuracy 0.98 \
  --iterations 500 \
  --export-model "novacron-phase3-patterns.json"

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
  --project "novacron-phase3" \
  --methodology "infrastructure" \
  --enable-pipeline true

═══════════════════════════════════════════════════════════════════════════════
STEP 2: CREATE BEADS TASKS FOR PRODUCTION HARDENING (Single Message)
═══════════════════════════════════════════════════════════════════════════════

# Task 1: Complete Observability Stack
mcp__beads__create {
  workspace_root: "/home/kp/novacron",
  id: "novacron-011",
  title: "Complete Observability Stack (60% → 95%)",
  description: "Implement full observability: Prometheus metrics, Grafana dashboards, Loki logs, Jaeger traces, alerting rules. 95% coverage of critical paths.",
  issue_type: "task",
  priority: 1,
  assignee: "observability-engineer",
  labels: ["observability", "monitoring", "critical"],
  deps: []
}

# Task 2: Deployment Automation
mcp__beads__create {
  workspace_root: "/home/kp/novacron",
  id: "novacron-012",
  title: "Automate Deployment Pipeline",
  description: "Create fully automated deployment: CI/CD with GitHub Actions, blue-green deployment, automated rollback, canary releases. Zero-downtime deployments.",
  issue_type: "task",
  priority: 1,
  assignee: "devops-engineer",
  labels: ["deployment", "automation", "critical"],
  deps: []
}

# Task 3: Disaster Recovery
mcp__beads__create {
  workspace_root: "/home/kp/novacron",
  id: "novacron-013",
  title: "Test and Document Disaster Recovery",
  description: "Test DR procedures: Backup/restore, failover, data recovery. Document runbooks. Validate RTO <1hr, RPO <15min.",
  issue_type: "task",
  priority: 2,
  assignee: "sre-engineer",
  labels: ["disaster-recovery", "testing", "high"],
  deps: []
}

# Task 4: Performance Optimization
mcp__beads__create {
  workspace_root: "/home/kp/novacron",
  id: "novacron-014",
  title: "Optimize Performance",
  description: "Optimize critical paths: Database queries, API endpoints, DWCP protocol. Target: p95 latency <100ms, throughput >10K req/s.",
  issue_type: "task",
  priority: 2,
  assignee: "performance-engineer",
  labels: ["performance", "optimization", "medium"],
  deps: []
}

# Task 5: Production Runbooks
mcp__beads__create {
  workspace_root: "/home/kp/novacron",
  id: "novacron-015",
  title: "Create Production Runbooks",
  description: "Document operational procedures: Deployment, rollback, incident response, scaling, troubleshooting. Complete runbook library.",
  issue_type: "task",
  priority: 2,
  assignee: "documentation-engineer",
  labels: ["documentation", "runbooks", "high"],
  deps: ["novacron-011", "novacron-012", "novacron-013"]
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
- ✅ Phase 2 complete (Score: 90/100)
- ✅ Beads MCP configured
- ✅ Claude-Flow installed
- ✅ Infrastructure access ready

---

## 🎯 SUCCESS CRITERIA

**Phase 3 Complete When**:
- ✅ Observability: 95% coverage
- ✅ Deployment: Fully automated
- ✅ DR: Tested (RTO <1hr, RPO <15min)
- ✅ Performance: Optimized (p95 <100ms)
- ✅ Runbooks: Complete
- ✅ Score: 93/100

**Ready to execute Phase 3!** 🚀

