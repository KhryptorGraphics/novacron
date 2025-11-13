# NovaCron Production Runbooks - Complete Index

**Last Updated:** 2025-11-12
**Total Runbooks:** 19
**Coverage:** Complete operational procedures for production

---

## Quick Reference

| Category | Count | Critical? |
|----------|-------|-----------|
| Deployment | 4 | Yes |
| Incident Response | 4 | Yes |
| Scaling | 3 | Medium |
| Troubleshooting | 4 | Medium |
| Maintenance | 4 | Low |

---

## Deployment Runbooks (4)

### [RUNBOOK-DEPLOYMENT.md](RUNBOOK-DEPLOYMENT.md)
**Purpose:** Standard production deployment procedure
**When to use:** Normal release deployment
**Est. Time:** 30 minutes
**Risk:** Low (blue-green deployment)

### [RUNBOOK-ROLLBACK.md](RUNBOOK-ROLLBACK.md)
**Purpose:** Emergency rollback procedure
**When to use:** Deployment issues, high error rate, critical bugs
**Est. Time:** <5 minutes
**Risk:** Medium (quick rollback required)

### [RUNBOOK-CANARY.md](RUNBOOK-CANARY.md)
**Purpose:** Canary release deployment
**When to use:** High-risk changes, major version upgrades
**Est. Time:** 2 hours (progressive rollout)
**Risk:** Low (gradual rollout with monitoring)

### [RUNBOOK-HOTFIX.md](RUNBOOK-HOTFIX.md)
**Purpose:** Emergency hotfix deployment
**When to use:** Critical security patches, production-down issues
**Est. Time:** 15 minutes (bypass normal process)
**Risk:** High (reduced testing)

---

## Incident Response Runbooks (4)

### [RUNBOOK-HIGH-ERROR-RATE.md](RUNBOOK-HIGH-ERROR-RATE.md)
**Purpose:** Handle API error rate spikes
**Severity:** Critical
**Trigger:** >1% error rate for 5 minutes
**Response Time:** Immediate

### [RUNBOOK-HIGH-LATENCY.md](RUNBOOK-HIGH-LATENCY.md)
**Purpose:** Handle API latency spikes
**Severity:** Warning
**Trigger:** p95 >500ms for 5 minutes
**Response Time:** <15 minutes

### [RUNBOOK-SERVICE-DOWN.md](RUNBOOK-SERVICE-DOWN.md)
**Purpose:** Complete service outage response
**Severity:** Critical
**Trigger:** Service unavailable for 2 minutes
**Response Time:** Immediate

### [RUNBOOK-DATABASE-ISSUES.md](RUNBOOK-DATABASE-ISSUES.md)
**Purpose:** Database problems (slow queries, connections)
**Severity:** Critical/Warning
**Trigger:** Query p95 >1s or connection pool >90%
**Response Time:** <15 minutes

---

## Disaster Recovery Runbooks (4)

### [DR-DATABASE-FAILURE.md](DR-DATABASE-FAILURE.md)
**Purpose:** Complete database failure recovery
**Severity:** Critical
**RTO:** <1 hour
**RPO:** <15 minutes

### [DR-SERVICE-FAILURE.md](DR-SERVICE-FAILURE.md)
**Purpose:** Application service failure recovery
**Severity:** Critical
**RTO:** <15 minutes
**RPO:** 0 (stateless)

### [DR-COMPLETE-RECOVERY.md](DR-COMPLETE-RECOVERY.md)
**Purpose:** Full system disaster recovery
**Severity:** Critical
**RTO:** <2 hours
**RPO:** <15 minutes

### [DR-DATA-CORRUPTION.md](DR-DATA-CORRUPTION.md)
**Purpose:** Data corruption detection and remediation
**Severity:** Critical
**RTO:** <3 hours
**RPO:** Point-in-time recovery

---

## Scaling Runbooks (3)

### [RUNBOOK-SCALE-UP.md](RUNBOOK-SCALE-UP.md)
**Purpose:** Increase system capacity
**When to use:** High traffic, CPU >70%, capacity planning
**Est. Time:** 15 minutes
**Risk:** Low (zero-downtime scaling)

### [RUNBOOK-SCALE-DOWN.md](RUNBOOK-SCALE-DOWN.md)
**Purpose:** Decrease system capacity
**When to use:** Low traffic periods, cost optimization
**Est. Time:** 20 minutes
**Risk:** Low (gradual scale-down)

### [RUNBOOK-AUTO-SCALING.md](RUNBOOK-AUTO-SCALING.md)
**Purpose:** Configure automatic scaling policies
**When to use:** Initial setup, policy tuning
**Est. Time:** 30 minutes
**Risk:** Low (well-tested policies)

---

## Troubleshooting Runbooks (4)

### [RUNBOOK-DEBUG-API.md](RUNBOOK-DEBUG-API.md)
**Purpose:** Troubleshoot API issues
**Scenarios:** Slow endpoints, errors, timeouts
**Tools:** Logs, traces, metrics, profiling

### [RUNBOOK-DEBUG-DWCP.md](RUNBOOK-DEBUG-DWCP.md)
**Purpose:** Troubleshoot DWCP protocol issues
**Scenarios:** Slow migrations, failures, bandwidth issues
**Tools:** DWCP metrics, packet captures, logs

### [RUNBOOK-DEBUG-DATABASE.md](RUNBOOK-DEBUG-DATABASE.md)
**Purpose:** Troubleshoot database issues
**Scenarios:** Slow queries, connection issues, replication lag
**Tools:** EXPLAIN, pg_stat_statements, logs

### [RUNBOOK-DEBUG-PERFORMANCE.md](RUNBOOK-DEBUG-PERFORMANCE.md)
**Purpose:** General performance troubleshooting
**Scenarios:** High latency, resource exhaustion
**Tools:** Profiling, APM, distributed tracing

---

## Maintenance Runbooks (4)

### [RUNBOOK-BACKUP.md](RUNBOOK-BACKUP.md)
**Purpose:** Backup procedures and validation
**Frequency:** Automated (hourly/daily)
**Manual Trigger:** Pre-major changes
**Validation:** Integrity checks, test restores

### [RUNBOOK-RESTORE.md](RUNBOOK-RESTORE.md)
**Purpose:** Restore from backup
**When to use:** Data loss, corruption, testing
**Est. Time:** 30-60 minutes
**Risk:** High (data loss possible if incorrect)

### [RUNBOOK-UPGRADE.md](RUNBOOK-UPGRADE.md)
**Purpose:** System upgrade procedures
**Scope:** Database, Kubernetes, dependencies
**Est. Time:** 1-3 hours (depends on component)
**Risk:** Medium (tested in staging first)

### [RUNBOOK-SECURITY-PATCH.md](RUNBOOK-SECURITY-PATCH.md)
**Purpose:** Emergency security patching
**When to use:** CVE disclosures, zero-days
**Est. Time:** 30-60 minutes
**Risk:** Low (follow hotfix procedure)

---

## Using This Library

### For On-Call Engineers

**During an incident:**
1. Check PagerDuty alert for runbook link
2. Follow runbook steps sequentially
3. Update incident Slack channel with progress
4. Document deviations in postmortem

**Pro tips:**
- Runbooks are tested procedures - trust them
- Don't skip steps even if you think you know better
- Document everything for postmortem
- Escalate if stuck for >15 minutes

### For New Team Members

**Onboarding:**
1. Read all runbooks in order (2-3 hours)
2. Shadow on-call engineer during incidents
3. Practice runbooks in staging environment
4. Participate in DR drill exercises

**Training exercises:**
- Monthly DR drills
- Quarterly chaos engineering
- Simulated incident response
- Runbook validation sessions

### For Managers

**Metrics to track:**
- Time to resolve by runbook
- Runbook effectiveness (success rate)
- Documentation quality feedback
- Process improvement suggestions

**Review cycle:**
- Monthly: Update runbooks based on incidents
- Quarterly: Full runbook review and validation
- Annually: Major process improvements

---

## Runbook Maintenance

### When to Update

**Immediate update required:**
- Infrastructure changes (new services, architecture)
- Process changes (new tools, workflows)
- After any incident where runbook was insufficient
- When steps are found to be incorrect

**Regular updates:**
- After each use (validate steps still work)
- Monthly review (check for drift)
- After major deployments
- Annually (comprehensive review)

### Update Process

1. **Propose change:**
   - Create PR with runbook updates
   - Explain rationale in PR description
   - Link to related incidents or changes

2. **Review:**
   - SRE team review required
   - Test changes in staging if possible
   - Validate with on-call engineers

3. **Approve and merge:**
   - Update version number
   - Update "Last Updated" date
   - Notify team in #sre-updates

4. **Train:**
   - Brief team on changes
   - Update training materials
   - Add to next drill if significant

---

## Related Documentation

- [OBSERVABILITY-STACK.md](../OBSERVABILITY-STACK.md) - Monitoring and alerting
- [DEPLOYMENT-GUIDE.md](../DEPLOYMENT-GUIDE.md) - Deployment procedures
- [DR-TEST-RESULTS.md](../DR-TEST-RESULTS.md) - DR validation results
- [PERFORMANCE-OPTIMIZATION-REPORT.md](../PERFORMANCE-OPTIMIZATION-REPORT.md) - Performance baselines

---

## Feedback

Runbooks are living documents that improve through use. Please provide feedback:

- **Issues:** Create ticket in Jira with label "runbook"
- **Suggestions:** Post in #sre-team Slack channel
- **Urgent:** DM SRE team lead directly

**Common feedback requests:**
- Steps that didn't work as documented
- Missing information or context
- Unclear instructions
- Better alternatives discovered

---

## Emergency Contacts

**During business hours:**
- SRE Team: #sre-team Slack channel
- Database Team: #dba-team Slack channel
- Security Team: #security-team Slack channel

**After hours:**
- PagerDuty: https://novacron.pagerduty.com
- Emergency Hotline: 1-800-NOVACRON
- On-call rotation: Check PagerDuty schedule

**Escalation:**
1. On-call Engineer (0-15 minutes)
2. Team Lead (15-30 minutes)
3. Engineering Manager (30-60 minutes)
4. VP Engineering (>60 minutes or major incident)
5. CTO (customer-impacting major incident)

---

**Index Version:** 1.0
**Last Updated:** 2025-11-12
**Next Review:** 2025-12-12
**Maintained By:** SRE Team
