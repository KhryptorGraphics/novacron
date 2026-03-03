# DWCP v3 Phase 6: Production Alert Playbooks

**Version:** 1.0.0
**Last Updated:** 2025-11-10
**Audience:** On-call Engineers, SRE Team, Incident Response

---

## Table of Contents

1. [Alert Severity Levels](#alert-severity-levels)
2. [Escalation Matrix](#escalation-matrix)
3. [Critical Alert Playbooks](#critical-alert-playbooks)
4. [Warning Alert Playbooks](#warning-alert-playbooks)
5. [Communication Protocols](#communication-protocols)
6. [Rollback Procedures](#rollback-procedures)
7. [Post-Incident Process](#post-incident-process)

---

## Alert Severity Levels

### Severity Classification

| Level | Response Time | Escalation | Wake On-Call | Examples |
|-------|---------------|------------|--------------|----------|
| **CRITICAL** | <5 minutes | Immediate | YES | SLA violations, component failures, NO-GO decision |
| **HIGH** | <15 minutes | After 10 min | YES (if persistent) | High latency, low throughput, memory leaks |
| **WARNING** | <30 minutes | After 30 min | NO | Elevated metrics, anomalies |
| **INFO** | Best effort | None | NO | Informational metrics |

---

## Escalation Matrix

### Escalation Path

```
ALERT FIRES
    ↓
On-Call Primary (Page)
    ↓ (5 minutes, no ACK)
On-Call Secondary (Page)
    ↓ (10 minutes, no resolution)
Team Lead (Call)
    ↓ (15 minutes, critical)
Engineering Manager (Call)
    ↓ (20 minutes, SLA violation)
VP Engineering + CTO (Conference Call)
```

### Team Assignments

| Alert Label | Primary Team | Secondary Team | Subject Matter Expert |
|-------------|--------------|----------------|----------------------|
| `team: oncall` | Platform SRE | DevOps | John Doe |
| `team: performance` | Performance Eng | Platform SRE | Jane Smith |
| `team: network` | Network Eng | Platform SRE | Bob Johnson |
| `team: distributed-systems` | Distributed Systems | Platform SRE | Alice Williams |
| `team: ml` | ML Platform | Data Science | Charlie Brown |

---

## Critical Alert Playbooks

### 🚨 DWCPv3RolloutNoGo

**Severity:** CRITICAL
**Response Time:** IMMEDIATE (<2 minutes)
**Escalation:** Immediate to all teams

#### Symptoms
- GO/NO-GO dashboard shows "NO GO - HALT ROLLOUT"
- Multiple SLA violations (latency, throughput, error rate)
- Overall compliance <50%

#### Immediate Actions (First 5 Minutes)

1. **HALT ROLLOUT IMMEDIATELY**
   ```bash
   # Set feature flag to 0% (complete stop)
   kubectl set env deployment/dwcp-v3 FEATURE_FLAG_ROLLOUT=0

   # Verify rollout stopped
   kubectl rollout status deployment/dwcp-v3
   ```

2. **Gather Initial Context**
   ```bash
   # Check current metrics
   curl http://prometheus:9090/api/v1/query?query=dwcp_v3_sla_compliance

   # Get recent errors
   kubectl logs -l app=dwcp-v3 --tail=100 --timestamps
   ```

3. **Alert Stakeholders**
   ```bash
   # Send emergency notification
   ./scripts/notify-emergency.sh "DWCP v3 ROLLOUT HALTED - NO-GO Decision"
   ```

#### Diagnostic Steps (5-15 Minutes)

4. **Identify Root Cause**
   - Check Grafana Phase 6 dashboard for specific violations
   - Review error logs for patterns:
     ```bash
     kubectl logs -l app=dwcp-v3 --since=10m | grep -i error | head -50
     ```
   - Check component health:
     ```bash
     curl http://dwcp-v3:8080/health
     ```

5. **Assess Impact**
   ```bash
   # Count affected migrations
   kubectl exec -it prometheus-0 -- promtool query instant \
     'sum(rate(dwcp_v3_migration_latency_seconds_count[5m])) * 300'

   # Get error breakdown
   kubectl exec -it prometheus-0 -- promtool query instant \
     'sum by (component,error_type) (increase(dwcp_v3_errors_total[10m]))'
   ```

#### Decision Points (15-30 Minutes)

6. **GO/NO-GO Decision Tree**

   ```
   Is root cause identified?
       ├─ NO → Continue diagnostics, prepare rollback
       └─ YES
           ├─ Can fix in <15 minutes?
           │   ├─ YES → Apply hotfix, monitor closely
           │   └─ NO → ROLLBACK
           └─ Is fix risky?
               ├─ YES → ROLLBACK
               └─ NO → Apply fix, staged rollout
   ```

7. **Execute Decision**

   **Option A: Hotfix**
   ```bash
   # Apply configuration fix
   kubectl apply -f hotfix-config.yaml

   # Verify metrics improve within 5 minutes
   watch -n 5 'curl -s http://prometheus:9090/api/v1/query?query=dwcp_v3_sla_compliance'
   ```

   **Option B: Rollback**
   ```bash
   # Execute rollback (see Rollback Procedures below)
   ./scripts/rollback-dwcp-v3.sh --phase 6 --confirm
   ```

#### Communication Template

```
Subject: [CRITICAL] DWCP v3 Rollout Halted - NO-GO Decision

Status: ROLLOUT HALTED
Time: [TIMESTAMP]
Duration: [DURATION]
Impact: [NUMBER] migrations affected

Root Cause: [BRIEF DESCRIPTION]

Current Actions:
1. [ACTION 1]
2. [ACTION 2]

Next Steps:
- [DECISION: HOTFIX or ROLLBACK]
- ETA: [TIME]

Incident Commander: [NAME]
War Room: [SLACK/ZOOM LINK]
```

---

### 🔴 DWCPv3LatencyP99Critical

**Severity:** CRITICAL
**Response Time:** <5 minutes
**Escalation:** Immediate to performance team

#### Symptoms
- P99 latency >500ms for 2+ minutes
- Migration completion time exceeds SLA
- User-visible delays in VM migrations

#### Immediate Actions

1. **Check Current State**
   ```bash
   # Get current P99 latency
   curl -s "http://prometheus:9090/api/v1/query?query=histogram_quantile(0.99,sum(rate(dwcp_v3_migration_latency_seconds_bucket[5m]))by(le))"

   # Get latency by component
   curl -s "http://prometheus:9090/api/v1/query?query=histogram_quantile(0.99,sum(rate(dwcp_v3_migration_latency_seconds_bucket[5m]))by(le,component))"
   ```

2. **Identify Bottleneck**
   ```bash
   # Check AMST throughput
   curl -s http://localhost:9091/metrics | grep dwcp_v3_amst_throughput

   # Check HDE compression performance
   curl -s http://localhost:9092/metrics | grep dwcp_v3_hde_compression_latency

   # Check network latency
   kubectl exec -it dwcp-v3-pod -- ping -c 5 destination-host
   ```

3. **Quick Mitigations**

   **If AMST bottleneck:**
   ```bash
   # Increase stream count
   kubectl set env deployment/dwcp-v3-amst STREAM_COUNT=16
   ```

   **If HDE bottleneck:**
   ```bash
   # Switch to faster compression
   kubectl set env deployment/dwcp-v3-hde COMPRESSION_ALGO=snappy
   ```

   **If network congestion:**
   ```bash
   # Enable traffic shaping
   kubectl apply -f configs/network-qos-high.yaml
   ```

#### Diagnostic Deep Dive

4. **Performance Profiling**
   ```bash
   # Enable CPU profiling
   curl http://dwcp-v3:6060/debug/pprof/profile?seconds=30 > cpu.prof

   # Analyze with pprof
   go tool pprof -http=:8081 cpu.prof
   ```

5. **Check Resource Saturation**
   ```bash
   # CPU usage
   kubectl top pods -l app=dwcp-v3

   # Memory usage
   kubectl exec -it dwcp-v3-pod -- cat /proc/meminfo

   # Disk I/O
   kubectl exec -it dwcp-v3-pod -- iostat -x 1 5
   ```

#### Resolution

6. **Apply Fix and Verify**
   ```bash
   # After applying fix, monitor for 10 minutes
   watch -n 10 'echo "P99 Latency:" && curl -s "http://prometheus:9090/api/v1/query?query=histogram_quantile(0.99,sum(rate(dwcp_v3_migration_latency_seconds_bucket[5m]))by(le))" | jq -r ".data.result[0].value[1]"'
   ```

7. **Close Alert**
   - Verify P99 latency <500ms for 5 consecutive minutes
   - Document root cause in incident tracker
   - Schedule post-mortem if incident >30 minutes

---

### 🔴 DWCPv3ThroughputCriticalLow

**Severity:** CRITICAL
**Response Time:** <5 minutes
**Escalation:** Immediate to network team

#### Symptoms
- Throughput <2.0 GB/s for 3+ minutes
- Slow migration progress
- AMST streams underutilized

#### Immediate Actions

1. **Check Network Baseline**
   ```bash
   # Test network bandwidth
   kubectl exec -it dwcp-v3-source -- iperf3 -c dwcp-v3-destination -t 10

   # Check RDMA status
   kubectl exec -it dwcp-v3-pod -- ibv_devinfo
   ```

2. **Verify AMST Configuration**
   ```bash
   # Check active streams
   curl -s http://localhost:9091/metrics | grep dwcp_v3_amst_active_streams

   # Check stream utilization
   curl -s http://localhost:9091/metrics | grep dwcp_v3_amst_stream_utilization
   ```

3. **Check for Congestion**
   ```bash
   # Network interface stats
   kubectl exec -it dwcp-v3-pod -- ip -s link

   # Check packet loss
   kubectl exec -it dwcp-v3-pod -- netstat -s | grep -i loss
   ```

#### Quick Fixes

4. **Increase Parallelism**
   ```bash
   # Scale AMST workers
   kubectl scale deployment/dwcp-v3-amst --replicas=8

   # Increase TCP buffer sizes
   kubectl exec -it dwcp-v3-pod -- sysctl -w net.ipv4.tcp_rmem="4096 87380 134217728"
   kubectl exec -it dwcp-v3-pod -- sysctl -w net.ipv4.tcp_wmem="4096 87380 134217728"
   ```

5. **Switch Transport Mode**
   ```bash
   # Force RDMA if available
   kubectl set env deployment/dwcp-v3 TRANSPORT_MODE=rdma_only
   ```

#### Investigation

6. **Network Path Analysis**
   ```bash
   # Traceroute
   kubectl exec -it dwcp-v3-pod -- traceroute -T destination-host

   # MTU discovery
   kubectl exec -it dwcp-v3-pod -- tracepath destination-host
   ```

---

### 🔴 DWCPv3ErrorRateCritical

**Severity:** CRITICAL
**Response Time:** IMMEDIATE (<2 minutes)
**Escalation:** Immediate to on-call

#### Symptoms
- Error rate >10 errors/sec
- Failed migrations
- Component health degradation

#### Immediate Actions

1. **Get Error Breakdown**
   ```bash
   # Error types
   kubectl exec -it prometheus-0 -- promtool query instant \
     'sum by (component,error_type) (rate(dwcp_v3_errors_total[1m]))'

   # Recent error logs
   kubectl logs -l app=dwcp-v3 --since=5m | grep -i error | tail -50
   ```

2. **Identify Failing Component**
   ```bash
   # Component health
   curl http://dwcp-v3:8080/health | jq

   # Check each component
   for component in amst hde pba acp ass itp; do
     echo "=== $component ==="
     curl -s http://localhost:909${i}/health
   done
   ```

3. **Stop Bad Traffic**
   ```bash
   # If specific source causing errors, block it
   kubectl exec -it dwcp-v3-pod -- iptables -A INPUT -s PROBLEM_IP -j DROP
   ```

#### Root Cause Analysis

4. **Check Recent Changes**
   ```bash
   # Recent deployments
   kubectl rollout history deployment/dwcp-v3

   # Recent config changes
   kubectl get configmap dwcp-v3-config -o yaml | tail -20
   ```

5. **Database/State Issues**
   ```bash
   # Check etcd health
   kubectl exec -it etcd-0 -- etcdctl endpoint health

   # Check consensus state
   curl http://localhost:9094/metrics | grep dwcp_v3_acp_consensus
   ```

#### Resolution

6. **Apply Fix**

   **If code bug:**
   ```bash
   # Rollback to previous version
   kubectl rollout undo deployment/dwcp-v3
   ```

   **If configuration issue:**
   ```bash
   # Revert config
   kubectl apply -f configs/dwcp-v3-known-good.yaml
   ```

   **If external dependency:**
   ```bash
   # Add circuit breaker
   kubectl apply -f configs/circuit-breaker-enabled.yaml
   ```

---

## Warning Alert Playbooks

### ⚠️ DWCPv3LatencyP95High

**Severity:** WARNING
**Response Time:** <10 minutes
**Escalation:** After 30 minutes

#### Actions

1. **Monitor Trend**
   ```bash
   # Check if trending toward P99 threshold
   ./scripts/check-latency-trend.sh --metric p95 --duration 30m
   ```

2. **Investigate Proactively**
   - Review recent changes
   - Check for resource saturation
   - Verify no upstream degradation

3. **Prepare Mitigation**
   - Have rollback plan ready
   - Alert team to monitor closely
   - Document observations

---

### ⚠️ DWCPv3ThroughputLow

**Severity:** WARNING
**Response Time:** <10 minutes
**Escalation:** After 30 minutes

#### Actions

1. **Baseline Check**
   ```bash
   # Compare to historical average
   ./scripts/compare-to-baseline.sh --metric throughput --window 7d
   ```

2. **Identify Pattern**
   - Time of day correlation?
   - Specific workload types?
   - Network path changes?

3. **Optimization Opportunities**
   - Adjust AMST stream count
   - Optimize compression settings
   - Review bandwidth allocation

---

## Communication Protocols

### Slack Channels

- **#dwcp-v3-alerts** - All automated alerts
- **#dwcp-v3-incidents** - Active incident coordination
- **#dwcp-v3-oncall** - On-call team communication

### Status Page Updates

Update https://status.internal every 15 minutes during active incidents:

```markdown
[YYYY-MM-DD HH:MM] - INVESTIGATING
We are investigating elevated latency in DWCP v3 migrations.

[YYYY-MM-DD HH:MM] - IDENTIFIED
Root cause identified as network congestion. Applying mitigation.

[YYYY-MM-DD HH:MM] - MONITORING
Mitigation applied. Monitoring for improvement.

[YYYY-MM-DD HH:MM] - RESOLVED
Latency returned to normal. Incident resolved.
```

### Stakeholder Notifications

**Critical Incidents:**
- Initial: Within 5 minutes
- Updates: Every 15 minutes
- Recipients: VP Engineering, Product Lead, CTO

**High Priority:**
- Initial: Within 15 minutes
- Updates: Every 30 minutes
- Recipients: Engineering Manager, Product Lead

---

## Rollback Procedures

### Automated Rollback

```bash
#!/bin/bash
# Execute automatic rollback to DWCP v1

# Stop v3 rollout
kubectl set env deployment/dwcp-v3 FEATURE_FLAG_ROLLOUT=0

# Scale down v3
kubectl scale deployment/dwcp-v3 --replicas=0

# Scale up v1
kubectl scale deployment/dwcp-v1 --replicas=10

# Verify v1 healthy
kubectl wait --for=condition=available --timeout=300s deployment/dwcp-v1

# Update load balancer
kubectl patch service dwcp-service -p '{"spec":{"selector":{"version":"v1"}}}'

echo "Rollback complete. Verify metrics returning to baseline."
```

### Manual Rollback Checklist

- [ ] Get approval from incident commander
- [ ] Announce rollback in #dwcp-v3-incidents
- [ ] Execute rollback script
- [ ] Verify v1 traffic restored (check metrics)
- [ ] Verify no migrations in progress on v3
- [ ] Document rollback reason
- [ ] Schedule post-mortem

---

## Post-Incident Process

### Immediate Actions (Within 1 Hour)

1. **Close Alert**
   - Verify metrics stable
   - Silence alert in AlertManager

2. **Initial Summary**
   - Document timeline
   - Record actions taken
   - Note outstanding questions

### Within 24 Hours

3. **Post-Mortem Document**
   ```markdown
   # DWCP v3 Incident: [TITLE]

   **Date:** YYYY-MM-DD
   **Duration:** X hours Y minutes
   **Severity:** [CRITICAL/HIGH/WARNING]

   ## Impact
   - [Affected users/systems]
   - [Business impact]

   ## Timeline
   - HH:MM - Alert fired
   - HH:MM - Investigation began
   - HH:MM - Root cause identified
   - HH:MM - Mitigation applied
   - HH:MM - Resolved

   ## Root Cause
   [Detailed explanation]

   ## Resolution
   [What fixed it]

   ## Action Items
   - [ ] [Preventive measure 1]
   - [ ] [Monitoring improvement 1]
   - [ ] [Documentation update 1]
   ```

4. **Metrics Review**
   - Extract incident metrics from Prometheus
   - Generate incident report dashboard
   - Share with stakeholders

### Within 1 Week

5. **Post-Mortem Meeting**
   - Blameless review
   - Identify systemic issues
   - Assign action items

6. **Implement Improvements**
   - Update runbooks
   - Improve monitoring
   - Add preventive measures

---

## Quick Reference

### Essential Commands

```bash
# Check overall health
curl http://dwcp-v3:8080/health | jq

# Get current metrics
curl http://dwcp-v3:8080/metrics/summary | jq

# View recent errors
kubectl logs -l app=dwcp-v3 --tail=100 | grep ERROR

# Halt rollout
kubectl set env deployment/dwcp-v3 FEATURE_FLAG_ROLLOUT=0

# Emergency rollback
./scripts/rollback-dwcp-v3.sh --emergency --confirm
```

### Key Metrics Queries

```promql
# P99 Latency
histogram_quantile(0.99, sum(rate(dwcp_v3_migration_latency_seconds_bucket[5m])) by (le))

# Throughput
avg(dwcp_v3_throughput_bytes_per_second)

# Error Rate
sum(rate(dwcp_v3_errors_total[5m]))

# SLA Compliance
avg(dwcp_v3_sla_compliance)
```

### Contact Information

| Role | Name | Phone | Email |
|------|------|-------|-------|
| On-Call Primary | TBD | +1-XXX-XXX-XXXX | oncall@company.com |
| On-Call Secondary | TBD | +1-XXX-XXX-XXXX | oncall-backup@company.com |
| Engineering Manager | TBD | +1-XXX-XXX-XXXX | em@company.com |
| VP Engineering | TBD | +1-XXX-XXX-XXXX | vp-eng@company.com |

---

**Document Version:** 1.0.0
**Last Reviewed:** 2025-11-10
**Next Review:** 2025-12-10
**Owner:** Platform SRE Team
