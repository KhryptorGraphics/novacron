# NovaCron Runbook Library - Usage Guide

**Purpose:** Complete guide to using the NovaCron production runbook library
**Audience:** SREs, On-call Engineers, DevOps, Engineering Managers
**Last Updated:** 2025-11-12

---

## Overview

The NovaCron runbook library provides 19 comprehensive operational procedures covering all production scenarios from deployment to disaster recovery. This guide explains how to effectively use the runbook library.

---

## Library Structure

```
docs/runbooks/
├── INDEX.md                          # This index (start here)
├── RUNBOOK-LIBRARY-GUIDE.md          # Usage guide (this document)
│
├── Deployment (4 runbooks)
│   ├── RUNBOOK-DEPLOYMENT.md         # Standard deployment
│   ├── RUNBOOK-ROLLBACK.md           # Emergency rollback
│   ├── RUNBOOK-CANARY.md             # Canary deployment
│   └── RUNBOOK-HOTFIX.md             # Hotfix deployment
│
├── Incident Response (4 runbooks)
│   ├── RUNBOOK-HIGH-ERROR-RATE.md    # Error rate spikes
│   ├── RUNBOOK-HIGH-LATENCY.md       # Latency spikes
│   ├── RUNBOOK-SERVICE-DOWN.md       # Service outages
│   └── RUNBOOK-DATABASE-ISSUES.md    # Database problems
│
├── Disaster Recovery (4 runbooks)
│   ├── DR-DATABASE-FAILURE.md        # Database failure
│   ├── DR-SERVICE-FAILURE.md         # Service failure
│   ├── DR-COMPLETE-RECOVERY.md       # Full system recovery
│   └── DR-DATA-CORRUPTION.md         # Data corruption
│
├── Scaling (3 runbooks)
│   ├── RUNBOOK-SCALE-UP.md           # Increase capacity
│   ├── RUNBOOK-SCALE-DOWN.md         # Decrease capacity
│   └── RUNBOOK-AUTO-SCALING.md       # Auto-scaling config
│
├── Troubleshooting (4 runbooks)
│   ├── RUNBOOK-DEBUG-API.md          # API debugging
│   ├── RUNBOOK-DEBUG-DWCP.md         # DWCP debugging
│   ├── RUNBOOK-DEBUG-DATABASE.md     # Database debugging
│   └── RUNBOOK-DEBUG-PERFORMANCE.md  # Performance debugging
│
└── Maintenance (4 runbooks)
    ├── RUNBOOK-BACKUP.md             # Backup procedures
    ├── RUNBOOK-RESTORE.md            # Restore procedures
    ├── RUNBOOK-UPGRADE.md            # System upgrades
    └── RUNBOOK-SECURITY-PATCH.md     # Security patching
```

---

## Quick Start

### For New Engineers

**Day 1:**
1. Read this guide (15 minutes)
2. Read INDEX.md for overview (10 minutes)
3. Familiarize with runbook locations

**Week 1:**
1. Read all 19 runbooks (3-4 hours total)
2. Set up local environment for testing
3. Shadow on-call engineer

**Week 2:**
1. Practice runbooks in staging
2. Participate in DR drill
3. Ask questions in #sre-team

**Week 3:**
1. Take on-call shift with backup
2. Handle incidents with guidance
3. Start contributing to runbook updates

### For Incident Response

**When alert fires:**
1. **Check PagerDuty** - Alert includes runbook link
2. **Open runbook** - Follow linked runbook
3. **Update Slack** - Post in #incidents channel
4. **Execute steps** - Follow runbook sequentially
5. **Document** - Note deviations and issues
6. **Resolve** - Mark incident resolved
7. **Postmortem** - Schedule within 48 hours

**Example incident flow:**
```
12:00 AM - PagerDuty alert: High API Error Rate
12:01 AM - Open RUNBOOK-HIGH-ERROR-RATE.md
12:02 AM - Post in #incidents: "Responding to high error rate alert"
12:05 AM - Identify cause: Database connection pool exhaustion
12:10 AM - Scale up database connections
12:15 AM - Verify error rate back to normal
12:20 AM - Mark incident resolved
12:25 AM - Document in postmortem template
```

---

## Runbook Standard Format

All runbooks follow this structure:

```markdown
# Runbook Title

**Severity:** Critical/Warning/Info
**RTO/Est. Time:** Target resolution time
**Owner:** Responsible team
**Last Updated:** Date

## Overview
Brief description and when to use

## Detection
How to identify the issue (alerts, symptoms)

## Diagnosis
How to confirm root cause

## Recovery Procedure
Step-by-step resolution steps

## Verification
How to confirm issue is resolved

## Post-Recovery Tasks
Follow-up actions required

## Escalation
When and how to escalate

## Related Runbooks
Links to related procedures

## Testing History
Validation and testing records
```

---

## Best Practices

### During Incidents

**DO:**
✅ Follow runbook steps in order
✅ Document what you're doing in Slack
✅ Take screenshots of errors
✅ Note any deviations from runbook
✅ Ask for help if stuck >15 minutes
✅ Update runbook after incident

**DON'T:**
❌ Skip steps even if you think you know better
❌ Make changes without documenting
❌ Go rogue without telling team
❌ Forget to verify resolution
❌ Skip postmortem documentation

### During Normal Operations

**DO:**
✅ Practice runbooks in staging
✅ Participate in DR drills
✅ Keep runbooks up-to-date
✅ Share knowledge with team
✅ Suggest improvements

**DON'T:**
❌ Assume you remember the steps
❌ Skip testing after changes
❌ Leave outdated information
❌ Work alone on complex issues

---

## Tools and Access

### Required Access

**Production environment:**
- Kubernetes cluster (kubectl)
- AWS Console (read/write)
- PostgreSQL (DBA role)
- Grafana dashboards
- PagerDuty account

**Communication:**
- Slack (#sre-team, #incidents)
- Jira (incident tracking)
- GitHub (runbook updates)

**Verification:**
```bash
# Check your access
kubectl get pods -n novacron
psql -h db.novacron.io -U novacron -d novacron -c "SELECT 1"
aws s3 ls s3://novacron-backups/
```

### Required Tools

**Install these:**
```bash
# kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install kubectl /usr/local/bin/

# psql client
sudo apt-get install postgresql-client

# AWS CLI
pip install awscli

# k6 (load testing)
brew install k6

# jq (JSON processing)
sudo apt-get install jq
```

---

## Training Program

### Onboarding (Weeks 1-4)

**Week 1: Reading**
- Read all 19 runbooks
- Understand system architecture
- Set up local environment

**Week 2: Shadowing**
- Shadow on-call engineer
- Observe incident response
- Ask questions

**Week 3: Practicing**
- Practice runbooks in staging
- Simulate failure scenarios
- Participate in DR drill

**Week 4: On-call**
- Take on-call shift (with backup)
- Handle real incidents
- Get feedback from team

### Ongoing Training

**Monthly:**
- DR drill (1 hour)
- Runbook review (30 minutes)
- Incident postmortems (as needed)

**Quarterly:**
- Chaos engineering exercise (2 hours)
- Full system DR test (4 hours)
- Runbook validation sprint (1 day)

**Annually:**
- Comprehensive system review
- Process improvement workshop
- Advanced training topics

---

## Runbook Maintenance

### When to Update

**Immediate updates required:**
1. Infrastructure changes (new services)
2. Process changes (new tools)
3. After incidents where runbook failed
4. Discovered incorrect steps

**Regular updates:**
1. After each use (validation)
2. Monthly review
3. After major deployments
4. Annual comprehensive review

### Update Process

1. **Create PR:**
   ```bash
   git checkout -b update/runbook-deployment
   # Edit runbook
   git commit -m "Update RUNBOOK-DEPLOYMENT.md: Add step for DNS verification"
   git push origin update/runbook-deployment
   ```

2. **Review:**
   - SRE team review required
   - Test in staging if applicable
   - Get approval from 2+ engineers

3. **Merge and deploy:**
   - Merge to main
   - Update version number
   - Notify team in #sre-updates

4. **Training:**
   - Brief team on changes
   - Add to next drill if significant
   - Update training materials

---

## Metrics and KPIs

### Track These Metrics

**Incident response:**
- Time to acknowledge (target: <5 minutes)
- Time to resolve (varies by runbook)
- Runbook effectiveness (% of successful resolutions)
- Deviations from runbook (lower is better)

**Runbook quality:**
- Accuracy (steps work as documented)
- Completeness (no missing steps)
- Clarity (easy to understand)
- Timeliness (up-to-date with system)

**Team performance:**
- On-call response time
- Incident resolution time
- Postmortem completion rate
- Training completion rate

### Review Cadence

**Weekly:**
- Incident metrics review
- Open issues/feedback

**Monthly:**
- Runbook effectiveness analysis
- Process improvements discussion
- Training needs assessment

**Quarterly:**
- Comprehensive metrics review
- Runbook validation results
- Team performance review

---

## Common Scenarios

### Scenario 1: Database High CPU

**Alert:** Database CPU >90%
**Runbook:** [RUNBOOK-DATABASE-ISSUES.md](runbooks/RUNBOOK-DATABASE-ISSUES.md)

**Quick actions:**
1. Check slow queries: `SELECT * FROM pg_stat_statements ORDER BY total_time DESC LIMIT 10;`
2. Identify long-running queries
3. Kill if necessary: `SELECT pg_terminate_backend(pid);`
4. Scale up if needed
5. Investigate root cause

### Scenario 2: API Latency Spike

**Alert:** API p95 latency >500ms
**Runbook:** [RUNBOOK-HIGH-LATENCY.md](runbooks/RUNBOOK-HIGH-LATENCY.md)

**Quick actions:**
1. Check Grafana dashboard
2. Identify slow endpoints
3. Check database query times
4. Check external service latency
5. Scale up if needed

### Scenario 3: Deployment Rollback

**Trigger:** High error rate after deployment
**Runbook:** [RUNBOOK-ROLLBACK.md](runbooks/RUNBOOK-ROLLBACK.md)

**Quick actions:**
1. Confirm rollback decision
2. Run: `./deployment/rollback/auto-rollback.sh`
3. Wait for rollback (< 5 minutes)
4. Verify error rate drops
5. Investigate deployment issue

---

## FAQ

**Q: What if runbook doesn't work?**
A: Document the issue, escalate to team lead, find alternative solution, update runbook immediately after incident.

**Q: Can I modify runbook during incident?**
A: No, follow existing runbook. Document needed changes and update afterward in postmortem.

**Q: What if I don't have required access?**
A: Escalate immediately to team lead. Don't wait. Critical access should be provisioned within 15 minutes.

**Q: How often should I practice runbooks?**
A: At least monthly in DR drills. New engineers should practice weekly for first month.

**Q: What if runbook is outdated?**
A: Use best judgment, document deviations, update runbook immediately after incident.

---

## Feedback and Improvements

### Providing Feedback

**During incident:**
- Note issues in incident doc
- Discuss in postmortem
- Create Jira ticket with label "runbook"

**During normal operations:**
- Post in #sre-team channel
- Create PR with suggested changes
- Discuss in weekly SRE meeting

### Continuous Improvement

**We track:**
- Runbook usage frequency
- Success rate by runbook
- Time-to-resolution trends
- Feedback and suggestions

**We improve:**
- Monthly runbook updates
- Quarterly comprehensive reviews
- After each major incident
- Based on team feedback

---

## Support and Resources

**Documentation:**
- [INDEX.md](runbooks/INDEX.md) - Runbook index
- [OBSERVABILITY-STACK.md](OBSERVABILITY-STACK.md) - Monitoring guide
- [DR-TEST-RESULTS.md](DR-TEST-RESULTS.md) - DR validation
- [DEPLOYMENT-GUIDE.md](DEPLOYMENT-GUIDE.md) - Deployment guide

**Training:**
- Weekly SRE office hours
- Monthly DR drills
- Quarterly chaos engineering
- Annual training week

**Support:**
- Slack: #sre-team (24/7)
- PagerDuty: On-call engineer
- Email: sre@novacron.io
- Emergency: 1-800-NOVACRON

---

**Guide Version:** 1.0
**Last Updated:** 2025-11-12
**Maintained By:** SRE Team
**Next Review:** 2025-12-12
