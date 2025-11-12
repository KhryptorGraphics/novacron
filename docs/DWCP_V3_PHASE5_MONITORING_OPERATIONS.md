# DWCP v3 Phase 5: Production Monitoring Operations Guide

**Version:** 1.0.0
**Date:** 2025-11-10
**Status:** Production Ready

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Dashboard Guide](#dashboard-guide)
4. [Alert Response Procedures](#alert-response-procedures)
5. [Troubleshooting Guide](#troubleshooting-guide)
6. [SLA Tracking and Reporting](#sla-tracking-and-reporting)
7. [Runbooks](#runbooks)
8. [Maintenance Procedures](#maintenance-procedures)

---

## Overview

This guide provides comprehensive operational procedures for the DWCP v3 production monitoring infrastructure deployed in Phase 5. The monitoring stack includes:

- **Prometheus** (HA mode with 3 replicas): Metrics collection and alerting
- **Grafana**: Visualization and dashboards
- **Alertmanager**: Alert routing and notification
- **OpenTelemetry Collector**: Distributed tracing
- **Jaeger/Tempo**: Trace storage and visualization
- **Loki**: Log aggregation and querying
- **Promtail**: Log collection agent
- **Thanos/Cortex**: Long-term metrics storage

### Key Monitoring Objectives

1. **Real-time visibility** into DWCP v3 performance (1-5 second refresh)
2. **SLA compliance tracking**: 99.9% availability, P99 <100ms, error rate <1%
3. **Automatic rollback** on critical failures
4. **Production rollout monitoring** with go/no-go indicators
5. **Byzantine fault detection** and security alerting

---

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DWCP v3 Application                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  AMST    │  │   HDE    │  │   PBA    │  │   ACP    │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
│       │             │              │             │          │
│       └─────────────┴──────────────┴─────────────┘          │
│                         │                                    │
└─────────────────────────┼────────────────────────────────────┘
                          │
                          │ (Metrics, Traces, Logs)
                          │
         ┌────────────────┼────────────────┐
         │                │                │
         ▼                ▼                ▼
┌─────────────┐  ┌──────────────┐  ┌──────────────┐
│ Prometheus  │  │    OTel      │  │   Promtail   │
│  (HA x3)    │  │  Collector   │  │              │
└──────┬──────┘  └──────┬───────┘  └──────┬───────┘
       │                │                  │
       │                ▼                  ▼
       │         ┌──────────────┐  ┌──────────────┐
       │         │    Jaeger    │  │     Loki     │
       │         │    Tempo     │  │              │
       │         └──────────────┘  └──────────────┘
       │
       ▼
┌─────────────┐
│  Thanos/    │  Long-term storage (1 year retention)
│  Cortex     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Grafana    │  Visualization
└─────────────┘
       │
       ▼
┌─────────────┐
│ Alertmanager│  Alert routing & notification
└─────────────┘
```

### Component Responsibilities

| Component | Purpose | Retention | High Availability |
|-----------|---------|-----------|-------------------|
| **Prometheus** | Metrics collection (5s interval) | 30 days local | 3 replicas |
| **Thanos/Cortex** | Long-term metrics storage | 1 year | Yes (distributed) |
| **Grafana** | Dashboard visualization | N/A | 2 replicas |
| **Alertmanager** | Alert routing and notification | N/A | 3 replicas |
| **OTel Collector** | Trace/metric/log collection | N/A | 2 replicas |
| **Jaeger/Tempo** | Trace storage and query | 30 days | Yes |
| **Loki** | Log aggregation | 30 days | 3 replicas |
| **Promtail** | Log collection agent | N/A | DaemonSet |

---

## Dashboard Guide

### 1. Production Rollout Dashboard

**URL:** `https://grafana.novacron.io/d/dwcp-v3-production-rollout`
**Refresh:** 5 seconds
**Purpose:** Monitor feature flag rollout with automatic go/no-go indicators

#### Key Panels

##### GO/NO-GO Status (Top Left)
- **GREEN (GO)**: All SLAs met, safe to continue rollout
- **RED (NO-GO)**: SLA violation detected, automatic rollback triggered

**Query:**
```promql
(sla:availability:5m >= 0.999) and
(sla:latency_p99:5m <= 0.1) and
(sla:error_rate:5m <= 0.01)
```

**Actions on NO-GO:**
1. Automatic rollback triggered within 30 seconds
2. PagerDuty alert sent to on-call engineer
3. Rollout percentage frozen
4. Incident created in monitoring system

##### Rollout Progress
- Shows current percentage of nodes running v3
- Tracks progression: 10% → 50% → 100%
- Color codes:
  - **Blue**: Initial rollout (0-50%)
  - **Yellow**: Partial rollout (50-99%)
  - **Green**: Full deployment (100%)

##### Error Rate Comparison (V3 vs V1)
- Real-time comparison of error rates
- **Red line at 1%**: SLA threshold
- If V3 error rate exceeds V1 by 50% → automatic rollback

**Interpretation:**
- V3 error rate should be ≤ V1 error rate
- Any spike above 1% triggers critical alert
- Sustained error rate >0.5% triggers warning

##### Latency Comparison (V3 vs V1)
- P99 latency comparison
- **Red line at 100ms**: SLA threshold
- If V3 latency >30% slower than V1 → automatic rollback

**Good Performance:**
- V3 P99 latency < 100ms
- V3 latency similar to or better than V1

**Performance Degradation:**
- V3 P99 latency 80-100ms → Warning
- V3 P99 latency >100ms → Critical, rollback

##### Throughput Comparison (V3 vs V1)
- Real-time throughput in MB/s
- **Target**: 2400 MB/s (2.4 GB/s) for datacenter mode
- If V3 throughput <80% of V1 → Warning

##### Node Health Status (Table)
- Shows health of each node
- **GREEN**: Node healthy and responsive
- **RED**: Node down or unreachable

**Columns:**
- **Node**: Node identifier
- **Pod**: Kubernetes pod name
- **Version**: v1 (legacy) or v3 (new)
- **Status**: UP (green) or DOWN (red)

---

### 2. Real-Time Performance Dashboard

**URL:** `https://grafana.novacron.io/d/dwcp-v3-real-time`
**Refresh:** 1 second
**Purpose:** Live monitoring of VM migrations, bandwidth, and consensus

#### Key Panels

##### Active Migrations (Top Left)
- Number of VM migrations currently in progress
- Updates every second
- **Normal**: 1-10 concurrent migrations
- **High load**: >10 migrations

##### Current Mode (Top Center)
- Displays current network mode:
  - **DATACENTER** (green): High-speed RDMA
  - **INTERNET** (blue): TCP-based with compression
  - **HYBRID** (yellow): Mixed mode

##### Real-Time Bandwidth Utilization (Top Right)
- Current bandwidth across all streams
- **Gauge ranges:**
  - 0-1000 MB/s: Red (underutilized)
  - 1000-2000 MB/s: Yellow (moderate)
  - 2000-3000 MB/s: Green (optimal)

##### Byzantine Events (5m)
- Count of Byzantine node detections in last 5 minutes
- **Green (0)**: No security events
- **Yellow (1-4)**: Investigation required
- **Red (≥5)**: Critical security incident

**Response:**
- Any Byzantine event triggers security team alert
- Node is automatically isolated
- Consensus reconfiguration initiated

##### PBFT Consensus P99
- Real-time PBFT consensus latency
- **Targets:**
  - <50ms: Green (excellent)
  - 50-80ms: Yellow (acceptable)
  - >80ms: Red (degraded)

##### Live VM Migrations by Mode
- Stacked area chart showing migration rate
- Broken down by datacenter/internet/hybrid modes
- **Typical rates:**
  - Datacenter: 5-10 migrations/sec
  - Internet: 1-5 migrations/sec

##### Mode Switching Events
- Bar chart showing frequency of mode changes
- **Healthy pattern**: Few switches, stable mode
- **Unhealthy pattern**: Frequent oscillation (>10 switches/min)

**Troubleshooting:**
- Frequent switches indicate network instability
- Check bandwidth prediction accuracy

##### Consensus Protocol Activity
- Shows Raft vs PBFT consensus operations
- **Datacenter mode**: Raft preferred (lower latency)
- **Internet mode**: PBFT required (Byzantine tolerance)

##### Component Latency Heatmap
- Shows latency distribution across all components
- **Color scale:**
  - Blue/Green: <10ms (excellent)
  - Yellow: 10-50ms (good)
  - Orange: 50-100ms (acceptable)
  - Red: >100ms (poor)

**Hotspots:**
- Dark red areas indicate performance bottlenecks
- Investigate component-specific logs

---

### 3. SLA Compliance Dashboard

**URL:** `https://grafana.novacron.io/d/dwcp-v3-sla`
**Refresh:** 30 seconds
**Purpose:** Track compliance with SLA targets and calculate uptime

#### Key Metrics

##### Availability Gauge
- **SLA Target**: 99.9% (3 nines)
- **Calculation**: (Successful migrations / Total migrations) * 100
- **Allowed downtime:**
  - Per day: 1.44 minutes
  - Per week: 10.08 minutes
  - Per month: 43.2 minutes

**Interpretation:**
- **Green (≥99.9%)**: SLA met
- **Yellow (99.5-99.9%)**: At risk
- **Red (<99.5%)**: SLA violation

##### P99 Latency Gauge
- **SLA Target**: <100ms
- **Measurement**: 99th percentile of all operation latencies
- **Warning threshold**: 80ms

**What P99 means:**
- 99% of operations complete faster than this value
- 1% of operations may be slower

##### Error Rate Gauge
- **SLA Target**: <1%
- **Calculation**: (Errors / Total operations) * 100
- **Warning threshold**: 0.5%

**Error types tracked:**
- Network timeouts
- Byzantine failures
- Consensus timeouts
- State sync failures

##### Throughput Gauge
- **SLA Target**: 2.4 GB/s (datacenter mode)
- **Internet mode target**: 100-900 Mbps (adaptive)

##### Availability SLI Tracking (Chart)
- Time-series graph of availability over time
- Red line shows 99.9% threshold
- Dips below line indicate SLA violations

##### Error Budget Remaining
- Calculated from current error rate
- **Formula**: ((1% - current_error_rate) / 1%) * 100
- **Green (>50%)**: Healthy error budget
- **Yellow (25-50%)**: Caution, slow rollout
- **Red (<25%)**: Pause deployments

**Error budget strategy:**
- Use error budget for risky changes
- If budget low, delay non-critical updates
- Reset monthly

##### Uptime by Time Window (Table)
- Shows uptime percentage for:
  - 1 Hour
  - 24 Hours
  - 7 Days
  - 30 Days

**SLA compliance:**
- All windows should show ≥99.9%
- Monthly uptime is primary SLA metric

##### Total Downtime by Window (Table)
- Shows actual downtime in seconds
- **Acceptable monthly downtime**: ≤2592 seconds (43.2 minutes)

##### SLA Compliance Status (Table)
- **Four rows:**
  1. Availability (≥99.9%): ✓ COMPLIANT / ✗ VIOLATION
  2. Latency (≤100ms): ✓ COMPLIANT / ✗ VIOLATION
  3. Error Rate (≤1%): ✓ COMPLIANT / ✗ VIOLATION
  4. Throughput (≥2.4GB/s): ✓ COMPLIANT / ✗ VIOLATION

**All must be COMPLIANT for SLA to be met**

---

## Alert Response Procedures

### Alert Severity Levels

| Severity | Response Time | Notification | Action |
|----------|--------------|--------------|--------|
| **CRITICAL** | <1 minute | PagerDuty page + Slack + Email | Immediate investigation |
| **WARNING** | <15 minutes | Slack + Email | Investigation during business hours |
| **INFO** | <1 hour | Slack | Review and document |

### Critical Alert Runbooks

#### 1. SLAAvailabilityViolation

**Trigger:** Availability <99.9% for 5 minutes

**Automatic Actions:**
1. Auto-rollback triggered to v1
2. PagerDuty page sent
3. Slack message to #dwcp-critical-alerts
4. Email to oncall@novacron.io

**Manual Response:**

```bash
# Step 1: Check rollback status
kubectl get deployment dwcp-v3 -n dwcp-v3-production -o yaml | grep image:

# Step 2: Verify v1 is stable
curl http://dwcp-v1-api:8080/health

# Step 3: Check failed migrations
kubectl logs -n dwcp-v3-production -l app=dwcp-v3 --tail=100 | grep ERROR

# Step 4: Review metrics
# Open: https://grafana.novacron.io/d/dwcp-v3-sla
# Check "Availability SLI Tracking" panel

# Step 5: Investigate root cause
kubectl describe pods -n dwcp-v3-production -l app=dwcp-v3

# Step 6: Update incident ticket
# Document findings in PagerDuty incident
```

**Resolution Criteria:**
- Availability restored to ≥99.9%
- Root cause identified and documented
- Remediation plan created

**Post-Incident:**
1. Conduct post-mortem within 48 hours
2. Update runbook with findings
3. Implement preventive measures

---

#### 2. SLALatencyViolation

**Trigger:** P99 latency >100ms for 3 minutes

**Automatic Actions:**
1. Auto-rollback triggered
2. Critical alert sent

**Manual Response:**

```bash
# Step 1: Check current latency
curl -s http://prometheus:9090/api/v1/query?query=sla:latency_p99:5m

# Step 2: Identify slow component
# Open: https://grafana.novacron.io/d/dwcp-v3-real-time
# Review "Component Latency Heatmap"

# Step 3: Check for resource contention
kubectl top pods -n dwcp-v3-production

# Step 4: Review recent deployments
kubectl rollout history deployment/dwcp-v3 -n dwcp-v3-production

# Step 5: Check network latency
kubectl exec -it dwcp-v3-pod -n dwcp-v3-production -- ping dwcp-v3-peer

# Step 6: Enable debug tracing
kubectl set env deployment/dwcp-v3 TRACE_SAMPLING=100 -n dwcp-v3-production

# Step 7: Analyze traces
# Open: http://jaeger:16686
# Query: service=dwcp-v3 AND duration>100ms
```

**Common Causes:**
1. Network congestion
2. Resource exhaustion (CPU/memory)
3. Consensus delays (too many nodes)
4. Storage I/O bottleneck

**Resolution:**
- Identify bottleneck component
- Scale resources if needed
- Optimize slow code paths

---

#### 3. ByzantineNodeDetected

**Trigger:** Byzantine behavior detected on any node

**Automatic Actions:**
1. Node isolated from cluster
2. Security team paged
3. Consensus reconfiguration initiated

**Manual Response:**

```bash
# Step 1: Identify malicious node
kubectl logs -n dwcp-v3-production -l component=acp | grep "Byzantine detected"

# Step 2: Cordon node (prevent scheduling)
kubectl cordon <node-name>

# Step 3: Drain node (evict pods)
kubectl drain <node-name> --ignore-daemonsets --delete-emptydir-data

# Step 4: Capture forensics
kubectl exec -it dwcp-v3-pod -n dwcp-v3-production -- tar czf /tmp/forensics.tar.gz /var/log

# Step 5: Analyze Byzantine behavior
# Open traces in Jaeger
# Filter: tag:byzantine_detected=true

# Step 6: Remove node from cluster
kubectl delete node <node-name>

# Step 7: Security scan
# Run security scan on node image
# Check for unauthorized access

# Step 8: Update security policies
# Strengthen authentication
# Review access logs
```

**Security Response Checklist:**
- [ ] Node isolated
- [ ] Forensics captured
- [ ] Root cause identified
- [ ] Security team notified
- [ ] Incident report filed
- [ ] Security policies updated

---

#### 4. DWCPv3AllReplicasDown

**Trigger:** All DWCP v3 pods unavailable

**Severity:** EMERGENCY (highest priority)

**Automatic Actions:**
1. Immediate rollback to v1
2. Emergency page to all on-call engineers
3. Phone call to primary on-call

**Manual Response:**

```bash
# Step 1: Confirm all pods down
kubectl get pods -n dwcp-v3-production -l app=dwcp-v3

# Step 2: Check node health
kubectl get nodes

# Step 3: Check cluster events
kubectl get events -n dwcp-v3-production --sort-by='.lastTimestamp'

# Step 4: Force v1 deployment
kubectl scale deployment dwcp-v1 --replicas=10 -n dwcp-v3-production

# Step 5: Check service endpoints
kubectl get endpoints dwcp-v3-api -n dwcp-v3-production

# Step 6: Restart pods (if nodes healthy)
kubectl delete pods -n dwcp-v3-production -l app=dwcp-v3

# Step 7: Check for OOMKilled
kubectl describe pods -n dwcp-v3-production | grep -A 10 "Last State"

# Step 8: Increase resources if needed
kubectl set resources deployment/dwcp-v3 -c=dwcp-v3 --limits=memory=4Gi,cpu=2 -n dwcp-v3-production
```

**Escalation Path:**
1. Primary on-call (0-5 min)
2. Secondary on-call (5-10 min)
3. Engineering lead (10-15 min)
4. VP Engineering (15+ min)

---

### Warning Alert Responses

#### HighLatencyWarning

**Trigger:** P99 latency >80ms for 10 minutes

**Response Window:** 15 minutes

**Investigation Steps:**
1. Review Real-Time Performance dashboard
2. Check component-specific metrics
3. Look for resource constraints
4. Monitor for escalation to critical

**Resolution:** Optimize if latency continues to climb

---

#### ThroughputDegradation

**Trigger:** Throughput <2400 MB/s for 10 minutes

**Possible Causes:**
- Network congestion
- Fewer active migrations
- Resource throttling

**Investigation:**
```bash
# Check network interface stats
kubectl exec -it dwcp-v3-pod -- ifconfig

# Check bandwidth prediction accuracy
curl http://prometheus:9090/api/v1/query?query=dwcp_v3_prediction_accuracy

# Review active stream count
curl http://prometheus:9090/api/v1/query?query=dwcp_v3_active_streams
```

---

## Troubleshooting Guide

### Common Issues

#### 1. Dashboard Not Loading

**Symptoms:**
- Grafana dashboard shows "No data"
- Panels show "N/A"

**Diagnosis:**
```bash
# Check Prometheus status
kubectl get pods -n dwcp-v3-production -l app=prometheus

# Test Prometheus query
curl http://prometheus:9090/api/v1/query?query=up

# Check Grafana logs
kubectl logs -n dwcp-v3-production -l app=grafana --tail=50
```

**Resolution:**
1. Restart Prometheus if not scraping
2. Verify DWCP v3 metrics endpoint is accessible
3. Check Grafana datasource configuration

---

#### 2. Missing Traces

**Symptoms:**
- Jaeger shows no traces
- Trace IDs in logs but not in Jaeger

**Diagnosis:**
```bash
# Check OTel Collector status
kubectl get pods -n dwcp-v3-production -l app=otel-collector

# Check OTel Collector logs
kubectl logs -n dwcp-v3-production -l app=otel-collector | grep ERROR

# Verify Jaeger backend
curl http://jaeger-query:16686/api/services

# Test trace ingestion
curl -X POST http://otel-collector:4318/v1/traces \
  -H "Content-Type: application/json" \
  -d '{"resourceSpans": [{"scopeSpans": [{"spans": [{"name": "test"}]}]}]}'
```

**Common Fixes:**
- Increase sampling rate (if too low)
- Check network connectivity to Jaeger
- Verify trace context propagation

---

#### 3. Log Query Slow

**Symptoms:**
- Loki queries timeout
- Grafana Explore takes >30s

**Diagnosis:**
```bash
# Check Loki ingester status
kubectl get pods -n dwcp-v3-production -l app=loki-ingester

# Check query performance
curl -G -s "http://loki:3100/loki/api/v1/query_range" \
  --data-urlencode 'query={job="dwcp-v3-logs"}' \
  --data-urlencode 'start=2025-11-10T00:00:00Z' \
  --data-urlencode 'end=2025-11-10T23:59:59Z' \
  | jq '.status'

# Check index cache hit rate
curl http://loki:3100/metrics | grep loki_cache_
```

**Optimization:**
1. Add more specific label filters
2. Reduce query time range
3. Use `|= "pattern"` for text search
4. Increase cache size

---

#### 4. Alerts Not Firing

**Symptoms:**
- Expected alert not showing in Alertmanager
- Condition met but no notification

**Diagnosis:**
```bash
# Check Prometheus alert status
curl http://prometheus:9090/api/v1/rules | jq '.data.groups[].rules[] | select(.type=="alerting")'

# Check Alertmanager status
curl http://alertmanager:9093/api/v2/status

# Test alert rule manually
curl -G http://prometheus:9090/api/v1/query \
  --data-urlencode 'query=sla:availability:5m < 0.999'

# Check alert silences
curl http://alertmanager:9093/api/v2/silences
```

**Resolution:**
1. Verify alert rule syntax
2. Check Alertmanager configuration
3. Ensure notification channels configured
4. Remove any active silences

---

## SLA Tracking and Reporting

### Monthly SLA Report Generation

**Automated Report:**
```bash
# Generate monthly availability report
curl -G http://prometheus:9090/api/v1/query \
  --data-urlencode 'query=avg_over_time(sla:availability:5m[30d]) * 100' \
  | jq '.data.result[0].value[1]'

# Calculate total downtime (seconds)
curl -G http://prometheus:9090/api/v1/query \
  --data-urlencode 'query=2592000 * (1 - avg_over_time(sla:availability:5m[30d]))' \
  | jq '.data.result[0].value[1]'

# Get P99 latency average
curl -G http://prometheus:9090/api/v1/query \
  --data-urlencode 'query=avg_over_time(sla:latency_p99:5m[30d]) * 1000' \
  | jq '.data.result[0].value[1]'

# Get average error rate
curl -G http://prometheus:9090/api/v1/query \
  --data-urlencode 'query=avg_over_time(sla:error_rate:5m[30d]) * 100' \
  | jq '.data.result[0].value[1]'
```

**Report Template:**

```markdown
# DWCP v3 Monthly SLA Report - [Month] [Year]

## Executive Summary
- **Availability**: [XX.XX]% (Target: 99.9%)
- **P99 Latency**: [XX]ms (Target: <100ms)
- **Error Rate**: [X.XX]% (Target: <1%)
- **Throughput**: [XXXX] MB/s (Target: 2400 MB/s)

## SLA Compliance
- [✓/✗] Availability: [COMPLIANT/VIOLATION]
- [✓/✗] Latency: [COMPLIANT/VIOLATION]
- [✓/✗] Error Rate: [COMPLIANT/VIOLATION]
- [✓/✗] Throughput: [COMPLIANT/VIOLATION]

## Downtime Analysis
- **Total Downtime**: [XX] minutes ([XX] seconds)
- **Allowed Downtime**: 43.2 minutes
- **Remaining Budget**: [XX] minutes

### Incidents
1. [Date] [Time] - [Duration] - [Description]
2. [Date] [Time] - [Duration] - [Description]

## Performance Highlights
- **Successful Migrations**: [XXXXX]
- **Total Bytes Transferred**: [XX] TB
- **Average Migration Time**: [XX] seconds
- **Byzantine Events Detected**: [X]

## Recommendations
1. [Recommendation 1]
2. [Recommendation 2]
3. [Recommendation 3]
```

---

## Runbooks

### Rollback Procedure

**Manual Rollback (if automatic fails):**

```bash
#!/bin/bash
# Emergency rollback script

echo "=== DWCP v3 EMERGENCY ROLLBACK ==="

# Step 1: Scale down v3
kubectl scale deployment dwcp-v3 --replicas=0 -n dwcp-v3-production
echo "✓ Scaled down v3 to 0 replicas"

# Step 2: Scale up v1
kubectl scale deployment dwcp-v1 --replicas=10 -n dwcp-v3-production
echo "✓ Scaled up v1 to 10 replicas"

# Step 3: Wait for v1 to be ready
kubectl wait --for=condition=available --timeout=300s deployment/dwcp-v1 -n dwcp-v3-production
echo "✓ v1 deployment ready"

# Step 4: Verify health
for i in {1..5}; do
  if curl -f http://dwcp-v1-api:8080/health; then
    echo "✓ v1 health check passed"
    break
  fi
  echo "⚠ Health check failed, retrying..."
  sleep 10
done

# Step 5: Update feature flag
kubectl set env deployment/dwcp-v1 ENABLE_V3=false -n dwcp-v3-production
echo "✓ Feature flag disabled"

# Step 6: Notify team
curl -X POST https://hooks.slack.com/services/YOUR/WEBHOOK/URL \
  -H 'Content-Type: application/json' \
  -d '{"text": "🚨 DWCP v3 has been rolled back to v1. Check monitoring dashboards."}'
echo "✓ Team notified"

echo "=== ROLLBACK COMPLETE ==="
```

---

### Performance Investigation Procedure

```bash
#!/bin/bash
# Performance investigation script

echo "=== DWCP v3 PERFORMANCE INVESTIGATION ==="

# Step 1: Capture current metrics
echo "Capturing metrics snapshot..."
curl -s http://prometheus:9090/api/v1/query?query=sla:availability:5m > /tmp/availability.json
curl -s http://prometheus:9090/api/v1/query?query=sla:latency_p99:5m > /tmp/latency.json
curl -s http://prometheus:9090/api/v1/query?query=sla:error_rate:5m > /tmp/error_rate.json

# Step 2: Check resource utilization
echo "Checking resource utilization..."
kubectl top pods -n dwcp-v3-production -l app=dwcp-v3

# Step 3: Capture recent logs
echo "Capturing recent logs..."
kubectl logs -n dwcp-v3-production -l app=dwcp-v3 --tail=1000 --since=15m > /tmp/dwcp-v3-logs.txt

# Step 4: Check for errors
echo "Analyzing error patterns..."
grep -i error /tmp/dwcp-v3-logs.txt | sort | uniq -c | sort -rn | head -20

# Step 5: Check active migrations
echo "Checking active migrations..."
curl -s http://prometheus:9090/api/v1/query?query=dwcp_v3_active_streams | jq '.data.result[0].value[1]'

# Step 6: Check consensus health
echo "Checking consensus health..."
curl -s http://prometheus:9090/api/v1/query?query=rate(dwcp_v3_component_operations_total{component=\"acp\"}[5m]) | jq '.data.result'

# Step 7: Export traces
echo "Exporting recent traces..."
curl -s "http://jaeger-query:16686/api/traces?service=dwcp-v3&start=$(date -u -d '15 minutes ago' +%s)000000&end=$(date -u +%s)000000&limit=100" > /tmp/traces.json

# Step 8: Generate report
echo "Generating investigation report..."
cat > /tmp/investigation_report.md <<EOF
# Performance Investigation Report
Date: $(date)

## Metrics Snapshot
- Availability: $(jq -r '.data.result[0].value[1]' /tmp/availability.json)
- P99 Latency: $(jq -r '.data.result[0].value[1]' /tmp/latency.json)
- Error Rate: $(jq -r '.data.result[0].value[1]' /tmp/error_rate.json)

## Resource Utilization
$(kubectl top pods -n dwcp-v3-production -l app=dwcp-v3)

## Top Errors
$(grep -i error /tmp/dwcp-v3-logs.txt | sort | uniq -c | sort -rn | head -10)

## Recommendations
[Add manual analysis here]
EOF

echo "✓ Investigation complete. Report saved to /tmp/investigation_report.md"
```

---

## Maintenance Procedures

### Prometheus Maintenance

**Backup Prometheus Data:**
```bash
# Create snapshot
curl -XPOST http://prometheus:9090/api/v1/admin/tsdb/snapshot

# Copy snapshot to backup location
kubectl cp dwcp-v3-production/prometheus-0:/prometheus/snapshots/latest /backup/prometheus-$(date +%Y%m%d).tar.gz
```

**Cleanup Old Data:**
```bash
# Prometheus auto-deletes after retention period (30d)
# To manually trigger cleanup:
curl -XPOST http://prometheus:9090/api/v1/admin/tsdb/clean_tombstones
```

---

### Grafana Dashboard Backup

```bash
# Export all dashboards
for uid in $(curl -s http://admin:password@grafana:3000/api/search | jq -r '.[].uid'); do
  curl -s http://admin:password@grafana:3000/api/dashboards/uid/$uid | jq '.dashboard' > /backup/grafana/dashboard-$uid.json
done

# Import dashboard
curl -X POST http://admin:password@grafana:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @dashboard-backup.json
```

---

### Log Retention Management

**Loki Compaction:**
```bash
# Trigger manual compaction
curl -X POST http://loki:3100/loki/api/v1/compact

# Check compaction status
curl http://loki:3100/loki/api/v1/compact | jq
```

---

## Contact Information

### Escalation Matrix

| Level | Contact | Response Time |
|-------|---------|--------------|
| L1 | On-call Engineer | <5 minutes |
| L2 | Senior SRE | <15 minutes |
| L3 | Engineering Lead | <30 minutes |
| L4 | VP Engineering | <1 hour |

### Alert Channels

- **Critical Alerts**: PagerDuty + Phone
- **Warnings**: Slack #dwcp-warnings
- **Security**: Slack #security-alerts + Email

### Support Contacts

- **SRE Team**: sre-team@novacron.io
- **Security Team**: security-team@novacron.io
- **On-call**: oncall@novacron.io
- **Emergency Hotline**: +1-555-DWCP-911

---

## Appendix

### Useful PromQL Queries

```promql
# Calculate availability over last hour
avg_over_time(sla:availability:5m[1h]) * 100

# Find slowest operations
topk(10, histogram_quantile(0.99, rate(dwcp_v3_component_latency_seconds_bucket[5m])))

# Count errors by component
sum by (component) (rate(dwcp_v3_component_errors_total[5m]))

# Calculate throughput in Gbps
sum(rate(dwcp_v3_bytes_transferred_total[5m])) * 8 / 1000000000

# Find Byzantine events
increase(dwcp_v3_acp_byzantine_detections_total[5m])
```

### Useful LogQL Queries

```logql
# Find all errors in last hour
{job="dwcp-v3-logs"} |= "error" | json

# Migration failures
{job="dwcp-v3-logs"} | json | migration_status="failed"

# Byzantine detection events
{job="dwcp-v3-logs"} | json | byzantine_detected="true"

# Slow operations (>1s)
{job="performance-logs"} | json | duration_ms > 1000
```

---

**Document Version:** 1.0.0
**Last Updated:** 2025-11-10
**Maintained By:** SRE Team
**Review Cycle:** Quarterly
