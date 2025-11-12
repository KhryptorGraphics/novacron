# DWCP v3 Rollback Procedures

**Document Version:** 1.0
**Date:** 2025-11-10
**Project:** NovaCron DWCP v3 - Phase 6 Production Resilience
**Classification:** Operations Critical

---

## Executive Summary

This document provides comprehensive rollback procedures for DWCP v3 production deployments. It covers automatic rollback triggers, manual rollback processes, emergency procedures, and post-rollback validation.

**Key Features:**
- Automatic rollback based on metrics (error rate, latency, success rate)
- Manual rollback with operator confirmation
- Emergency rollback for critical failures
- <3 minute rollback execution time
- Zero data loss guarantee
- Automated validation

---

## Table of Contents

1. [Rollback Overview](#rollback-overview)
2. [Automatic Rollback](#automatic-rollback)
3. [Manual Rollback](#manual-rollback)
4. [Emergency Rollback](#emergency-rollback)
5. [Rollback Execution Steps](#rollback-execution-steps)
6. [Validation Procedures](#validation-procedures)
7. [Common Issues and Solutions](#common-issues-and-solutions)
8. [Runbooks](#runbooks)

---

## Rollback Overview

### Rollback Modes

| Mode | Trigger | Confirmation | Execution Time | Use Case |
|------|---------|--------------|----------------|----------|
| **Automatic** | Metrics-based | None | 2-3 minutes | Sustained degradation |
| **Manual** | Operator | Required | 2-3 minutes | Planned rollback |
| **Emergency** | Critical failure | None | 1-2 minutes | Total system failure |

### Rollback Triggers

**Automatic Rollback Triggers:**
- Error rate >1% for 2 minutes
- P99 latency >100ms sustained
- Success rate <99% for 2 minutes
- Critical service failures
- Byzantine consensus failures

**Manual Rollback Reasons:**
- Post-deployment issues discovered
- Performance concerns
- Functionality problems
- Business requirements

**Emergency Rollback Triggers:**
- Total service outage
- Data corruption detected
- Security breach
- Catastrophic failures

### Success Criteria

**Rollback is considered successful when:**
- ✅ Previous version running
- ✅ Health score >95/100
- ✅ Error rate <0.1%
- ✅ Latency within baseline
- ✅ All services operational
- ✅ Zero data loss confirmed

---

## Automatic Rollback

### Trigger Conditions

The automatic rollback system monitors these metrics every 30 seconds:

```yaml
thresholds:
  error_rate: 1.0%           # Maximum acceptable error rate
  latency_p99: 100ms         # P99 latency threshold
  success_rate: 99.0%        # Minimum success rate
  degradation_window: 120s   # Sustained degradation time
```

### Decision Engine

The automatic rollback system uses a multi-factor decision engine:

1. **Metric Collection:** Gather current system metrics
2. **Threshold Evaluation:** Check if any thresholds are exceeded
3. **Sustained Check:** Verify degradation is sustained (not transient)
4. **Rollback Decision:** Trigger if 75% of samples show degradation
5. **Execution:** Execute rollback automatically

### Execution Process

```bash
# Automatic rollback is handled by the system
# To monitor automatic rollback:
systemctl status novacron-auto-rollback

# To view rollback logs:
tail -f /var/log/novacron/rollback.log

# To check rollback state:
/home/kp/novacron/scripts/production/auto-rollback.sh status
```

### Post-Rollback Actions

After automatic rollback:
1. Incident created automatically
2. Notifications sent to on-call team
3. Metrics validation performed
4. Root cause analysis initiated
5. Deployment marked as failed

---

## Manual Rollback

### When to Use Manual Rollback

Use manual rollback when:
- Issues discovered post-deployment
- Business decision to rollback
- Gradual performance degradation
- Non-critical functionality issues
- Planned rollback for testing

### Manual Rollback Procedure

#### Step 1: Assess Current State

```bash
# Check current deployment version
curl -s http://localhost:8080/api/v3/version

# Check system health
/home/kp/novacron/scripts/production/health-checker.sh status

# Review recent metrics
curl -s http://localhost:8080/api/v3/metrics/summary | jq .
```

#### Step 2: Prepare for Rollback

```bash
# Verify previous version is available
ls -la /opt/novacron/versions/

# Check rollback system status
/home/kp/novacron/scripts/production/auto-rollback.sh status

# Notify team
curl -X POST http://alerting/notify \
  -d '{"message":"Manual rollback initiated","severity":"high"}'
```

#### Step 3: Execute Manual Rollback

```bash
# Initiate manual rollback with reason
/home/kp/novacron/scripts/production/auto-rollback.sh manual \
  "Performance degradation observed after deployment"

# System will prompt for confirmation:
# Current version: v3.0.0
# Target version: v2.4.0
# Reason: Performance degradation observed after deployment
#
# Confirm rollback? (yes/no): yes
```

#### Step 4: Monitor Rollback Progress

```bash
# Watch rollback logs in real-time
tail -f /var/log/novacron/rollback.log

# Monitor system health during rollback
watch -n 5 '/home/kp/novacron/scripts/production/health-checker.sh status'
```

#### Step 5: Validate Rollback

```bash
# Validate rollback completed successfully
/home/kp/novacron/scripts/production/auto-rollback.sh validate

# Expected output:
# Post-rollback metrics:
#   Error Rate: 0.05%
#   Latency P99: 45ms
#   Success Rate: 99.8%
# Rollback validation: PASSED
```

---

## Emergency Rollback

### When to Use Emergency Rollback

Emergency rollback should be used ONLY for:
- **P0 incidents** - Total service failure
- **Data corruption** - Data integrity compromised
- **Security breaches** - Active security incident
- **Critical failures** - System completely down

### Emergency Rollback Procedure

⚠️ **WARNING:** Emergency rollback skips confirmation steps and executes immediately.

```bash
# Execute emergency rollback
/home/kp/novacron/scripts/production/auto-rollback.sh emergency \
  "Critical: Total service outage"

# No confirmation required - executes immediately
```

### Emergency Rollback Checklist

- [ ] Incident declared (P0)
- [ ] On-call team notified
- [ ] Incident commander assigned
- [ ] Emergency rollback executed
- [ ] Service recovery verified
- [ ] Post-incident review scheduled

---

## Rollback Execution Steps

### Detailed Rollback Process

The rollback system follows these steps:

#### 1. Pause Deployments (5 seconds)

```bash
# API call to pause deployments
curl -X POST http://localhost:8080/api/v3/deployment/pause

# Verification
curl -s http://localhost:8080/api/v3/deployment/status | jq .paused
# Expected: true
```

#### 2. Drain Traffic (30 seconds)

```bash
# Gracefully drain traffic
curl -X POST http://localhost:8080/api/v3/traffic/drain \
  -H "Content-Type: application/json" \
  -d '{"timeout_seconds": 30}'

# Monitor active connections
watch 'curl -s http://localhost:8080/api/v3/metrics/connections'
```

#### 3. Stop Current Version (10 seconds)

```bash
# Stop DWCP v3 services
systemctl stop novacron-dwcp-v3

# Verify stopped
systemctl status novacron-dwcp-v3
# Expected: inactive (dead)
```

#### 4. Restore Previous Version (30 seconds)

```bash
# Switch to previous version
rm -f /opt/novacron/current
ln -s /opt/novacron/versions/v2.4.0 /opt/novacron/current

# Restore configuration
cp /etc/novacron/config.v2.4.0.yaml /etc/novacron/config.yaml

# Restore database schema (if needed)
# psql -U novacron -d novacron < /backup/schema.v2.4.0.sql
```

#### 5. Start Previous Version (10 seconds)

```bash
# Start DWCP v2 services
systemctl start novacron-dwcp-v2

# Verify started
systemctl status novacron-dwcp-v2
# Expected: active (running)
```

#### 6. Health Verification (60 seconds)

```bash
# Wait for services to stabilize
sleep 10

# Check health
for i in {1..12}; do
  if curl -sf http://localhost:8080/api/v3/health > /dev/null; then
    echo "Service healthy"
    break
  fi
  echo "Waiting for health check... ($i/12)"
  sleep 5
done
```

#### 7. Restore Traffic (5 seconds)

```bash
# Resume traffic flow
curl -X POST http://localhost:8080/api/v3/traffic/restore

# Verify traffic flowing
curl -s http://localhost:8080/api/v3/metrics/requests | jq .rate_1m
```

#### 8. Resume Deployments (5 seconds)

```bash
# Re-enable deployment capability
curl -X POST http://localhost:8080/api/v3/deployment/resume
```

### Total Rollback Time

| Phase | Time | Critical |
|-------|------|----------|
| Pause deployments | 5s | No |
| Drain traffic | 30s | Yes |
| Stop services | 10s | No |
| Restore version | 30s | Yes |
| Start services | 10s | Yes |
| Health check | 60s | Yes |
| Restore traffic | 5s | No |
| Resume deployments | 5s | No |
| **Total** | **~2m35s** | - |

---

## Validation Procedures

### Post-Rollback Validation

After rollback completes, validate these areas:

#### 1. Service Health

```bash
# Overall system health
curl -s http://localhost:8080/api/v3/health | jq .

# Expected output:
{
  "status": "healthy",
  "score": 98,
  "components": {
    "consensus": "healthy",
    "network": "healthy",
    "database": "healthy",
    "api": "healthy"
  }
}
```

#### 2. Performance Metrics

```bash
# Check key performance indicators
curl -s http://localhost:8080/api/v3/metrics/latency | jq .

# Expected:
# - P50 latency: <50ms
# - P99 latency: <100ms
# - Error rate: <0.1%
```

#### 3. Data Integrity

```bash
# Run data consistency checks
curl -X POST http://localhost:8080/api/v3/admin/consistency-check

# Check replication status
curl -s http://localhost:8080/api/v3/database/replication | jq .lag_ms
# Expected: <100ms
```

#### 4. Consensus Layer

```bash
# Verify consensus health
curl -s http://localhost:8080/api/v3/consensus/status | jq .

# Expected:
# - Leader elected: true
# - Quorum healthy: true
# - Replication lag: <50ms
# - Byzantine nodes: 0
```

#### 5. Traffic Validation

```bash
# Monitor request success rate
curl -s http://localhost:8080/api/v3/metrics/requests | jq .success_rate_1m
# Expected: >99.5%

# Check for errors
curl -s http://localhost:8080/api/v3/metrics/errors | jq .count_1m
# Expected: <5 per minute
```

### Validation Checklist

After rollback, verify:

- [ ] Previous version running (verify with `/api/v3/version`)
- [ ] Health score >95/100
- [ ] Error rate <0.1%
- [ ] P99 latency <100ms
- [ ] Success rate >99%
- [ ] All services operational
- [ ] Consensus healthy
- [ ] No Byzantine nodes
- [ ] Database replication healthy
- [ ] Traffic flowing normally
- [ ] No active incidents
- [ ] Monitoring functioning
- [ ] Logs being generated
- [ ] Alerts configured

---

## Common Issues and Solutions

### Issue 1: Rollback Fails to Complete

**Symptoms:**
- Rollback script hangs
- Services don't start
- Health checks timeout

**Solution:**
```bash
# 1. Check rollback logs
tail -100 /var/log/novacron/rollback.log

# 2. Verify previous version exists
ls -la /opt/novacron/versions/

# 3. Manually start previous version
systemctl start novacron-dwcp-v2

# 4. Check service logs
journalctl -u novacron-dwcp-v2 -n 100
```

### Issue 2: Health Checks Fail After Rollback

**Symptoms:**
- Services running but health checks fail
- Components marked as unhealthy
- High error rates persist

**Solution:**
```bash
# 1. Check component health individually
curl -s http://localhost:8080/api/v3/health/consensus
curl -s http://localhost:8080/api/v3/health/network
curl -s http://localhost:8080/api/v3/health/database

# 2. Restart specific components
systemctl restart novacron-consensus
systemctl restart novacron-network

# 3. Clear caches
curl -X POST http://localhost:8080/api/v3/cache/clear

# 4. Re-run health checker
/home/kp/novacron/scripts/production/health-checker.sh once
```

### Issue 3: Database Schema Mismatch

**Symptoms:**
- Database errors after rollback
- Schema version mismatch
- Failed database queries

**Solution:**
```bash
# 1. Check current schema version
psql -U novacron -d novacron -c "SELECT version FROM schema_version;"

# 2. Rollback database schema
psql -U novacron -d novacron < /backup/schema.v2.4.0.sql

# 3. Verify schema
psql -U novacron -d novacron -c "SELECT version FROM schema_version;"

# 4. Restart database-dependent services
systemctl restart novacron-api novacron-scheduler
```

### Issue 4: Traffic Not Restoring

**Symptoms:**
- No requests after rollback
- Load balancer not routing traffic
- Zero traffic metrics

**Solution:**
```bash
# 1. Check traffic routing
curl -s http://localhost:8080/api/v3/traffic/status

# 2. Manually restore traffic
curl -X POST http://localhost:8080/api/v3/traffic/restore

# 3. Check load balancer configuration
# (AWS ALB / Nginx / HAProxy specific)

# 4. Verify DNS resolution
dig novacron.example.com

# 5. Test direct connection
curl -v http://localhost:8080/api/v3/health
```

### Issue 5: Consensus Not Forming

**Symptoms:**
- No leader elected
- Quorum lost
- Cluster split

**Solution:**
```bash
# 1. Check cluster status
curl -s http://localhost:8080/api/v3/cluster/status

# 2. Check node connectivity
for i in {0..6}; do
  curl -s http://node-$i:8080/api/v3/health
done

# 3. Restart consensus layer
systemctl restart novacron-consensus

# 4. Force leader election (if safe)
curl -X POST http://localhost:8080/api/v3/consensus/force-election

# 5. Check for network partitions
/home/kp/novacron/scripts/production/health-checker.sh once
```

---

## Runbooks

### Runbook 1: Automatic Rollback Occurred

**Scenario:** Automatic rollback was triggered during deployment.

**Actions:**

1. **Acknowledge Incident**
   ```bash
   # Check rollback reason
   /home/kp/novacron/scripts/production/auto-rollback.sh status
   ```

2. **Validate System Health**
   ```bash
   /home/kp/novacron/scripts/production/health-checker.sh status
   ```

3. **Review Rollback Logs**
   ```bash
   tail -200 /var/log/novacron/rollback.log
   ```

4. **Analyze Root Cause**
   - Review metrics that triggered rollback
   - Check application logs
   - Identify deployment changes

5. **Document Findings**
   - Create incident report
   - Document root cause
   - Identify fix required

6. **Fix and Redeploy**
   - Fix identified issues
   - Re-test in staging
   - Schedule new deployment

### Runbook 2: Manual Rollback Required

**Scenario:** Post-deployment issue requires manual rollback.

**Actions:**

1. **Assess Severity**
   - Is this a P0/P1 incident?
   - Should we use emergency rollback?

2. **Notify Team**
   ```bash
   # Send notification
   curl -X POST http://alerting/notify \
     -d '{"message":"Manual rollback required","severity":"high"}'
   ```

3. **Execute Rollback**
   ```bash
   /home/kp/novacron/scripts/production/auto-rollback.sh manual \
     "Reason for rollback"
   ```

4. **Monitor Progress**
   ```bash
   tail -f /var/log/novacron/rollback.log
   ```

5. **Validate Success**
   ```bash
   /home/kp/novacron/scripts/production/auto-rollback.sh validate
   ```

6. **Post-Rollback Actions**
   - Update incident ticket
   - Schedule retrospective
   - Plan remediation

### Runbook 3: Rollback Failed

**Scenario:** Rollback process failed or validation failed.

**Actions:**

1. **Declare P0 Incident**
   ```bash
   # Escalate immediately
   curl -X POST http://alerting/page -d '{"severity":"P0"}'
   ```

2. **Assess Current State**
   ```bash
   # Check what's running
   systemctl status novacron-*

   # Check versions
   ls -la /opt/novacron/current
   ```

3. **Attempt Recovery**
   ```bash
   # Try emergency rollback
   /home/kp/novacron/scripts/production/auto-rollback.sh emergency \
     "Failed rollback recovery"
   ```

4. **Manual Recovery** (if automation fails)
   - SSH to each node
   - Manually stop v3 services
   - Manually start v2 services
   - Verify health on each node

5. **Service Restoration**
   - Bring up one node at a time
   - Verify cluster forms correctly
   - Gradually restore traffic

6. **Post-Incident**
   - Document what failed
   - Update rollback procedures
   - Fix automation issues
   - Schedule game day to test fix

---

## Appendices

### Appendix A: Rollback Metrics

Track these metrics for rollback performance:

```json
{
  "rollback_duration_seconds": 155,
  "phases": {
    "drain_traffic": 30,
    "stop_services": 10,
    "restore_version": 30,
    "start_services": 10,
    "health_check": 60,
    "restore_traffic": 5
  },
  "validation": {
    "health_score": 98,
    "error_rate": 0.05,
    "latency_p99_ms": 47,
    "success_rate": 99.8
  }
}
```

### Appendix B: Rollback Decision Matrix

| Metric | Threshold | Auto Rollback | Manual Review |
|--------|-----------|---------------|---------------|
| Error Rate | >1% | Yes | N/A |
| Latency P99 | >100ms sustained | Yes | N/A |
| Success Rate | <99% | Yes | N/A |
| CPU | >85% | No | Yes |
| Memory | >90% | No | Yes |
| Disk | >95% | No | Yes |

### Appendix C: Contact Information

**On-Call Rotation:**
- Primary: [On-call engineer]
- Secondary: [Backup engineer]
- Escalation: [Engineering manager]

**Emergency Contacts:**
- Platform Team: platform-oncall@example.com
- Database Team: dba-oncall@example.com
- Security Team: security-oncall@example.com
- Executive: exec-oncall@example.com

---

**Document Control:**
- **Version:** 1.0
- **Status:** Active
- **Classification:** Operations Critical
- **Distribution:** All Engineers, SRE Team, On-Call Team
- **Review Frequency:** Monthly
- **Next Review:** 2025-12-10

---

**END OF ROLLBACK PROCEDURES**
