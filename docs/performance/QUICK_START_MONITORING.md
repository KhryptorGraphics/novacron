# DWCP Performance Monitoring - Quick Start Guide

This guide helps you quickly deploy and use the DWCP performance monitoring and telemetry system.

---

## 🚀 Quick Deploy (5 Minutes)

### 1. Deploy Monitoring Stack

```bash
# Deploy Prometheus
kubectl apply -f deployments/kubernetes/prometheus-config.yml

# Deploy VictoriaMetrics for long-term storage
kubectl apply -f deployments/kubernetes/victoriametrics-cluster.yml

# Deploy Jaeger for distributed tracing
kubectl apply -f deployments/kubernetes/jaeger-deployment.yml

# Deploy Grafana
kubectl apply -f deployments/kubernetes/grafana-deployment.yml
```

### 2. Import Dashboards

```bash
# Import pre-configured dashboards
curl -X POST http://grafana:3000/api/dashboards/import \
  -H "Content-Type: application/json" \
  -d @docs/performance/GRAFANA_DASHBOARDS.json \
  -u admin:admin
```

### 3. Verify Deployment

```bash
# Check all components are running
kubectl get pods -n monitoring

# Expected output:
# prometheus-0                    1/1     Running
# victoria-metrics-vmstorage-0    1/1     Running
# victoria-metrics-vmselect-0     1/1     Running
# victoria-metrics-vminsert-0     1/1     Running
# jaeger-0                        1/1     Running
# grafana-0                       1/1     Running
```

### 4. Access Dashboards

- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Jaeger**: http://localhost:16686

---

## 📊 Key Dashboards

### Executive Overview Dashboard
**URL**: http://grafana:3000/d/dwcp-executive

**Panels**:
- ✅ Consensus Latency SLO (P95 target: 50ms)
- ✅ Migration Success Rate (target: 99%)
- ✅ Error Budget Remaining
- ✅ Total Migrations (24h)
- ✅ Consensus Latency Trend
- ✅ Bandwidth Utilization

### Performance Deep Dive Dashboard
**URL**: http://grafana:3000/d/dwcp-performance

**Panels**:
- 🔥 Consensus Latency Heatmap
- 📡 Network Partition Events
- 💻 Resource Utilization (CPU, Memory, Disk I/O)
- 🛡️ Circuit Breaker States

---

## 🔔 Critical Alerts

### Alert: Consensus Latency SLO Breach

**Trigger**: P95 consensus latency > 50ms for 1 minute

**Runbook**: `/docs/performance/runbooks/consensus-latency-high.md`

**Quick Fix**:
1. Check consensus node health: `kubectl get pods -l component=consensus`
2. Review recent deployments: `kubectl rollout history deployment/dwcp-consensus`
3. Check network latency: `ping <consensus-nodes>`
4. Review metrics: Grafana → Performance Deep Dive → Consensus Latency

### Alert: Circuit Breaker Opened

**Trigger**: Any circuit breaker in "open" state for 30 seconds

**Runbook**: `/docs/performance/runbooks/circuit-breaker-open.md`

**Quick Fix**:
1. Identify affected circuit breaker: `curl http://prometheus:9090/api/v1/query?query=dwcp_circuit_breaker_state==2`
2. Check downstream service health
3. Review error logs: `kubectl logs -l app=dwcp --tail=100 | grep ERROR`
4. Manual override (if safe): `curl -X POST http://dwcp-manager:8080/circuit-breaker/{name}/close`

### Alert: Migration Failure Rate High

**Trigger**: Migration failure rate > 5% for 5 minutes

**Runbook**: `/docs/performance/runbooks/migration-failures.md`

**Quick Fix**:
1. Check VM health: `kubectl get vms -o wide`
2. Review migration logs: `kubectl logs -l component=migration-manager --tail=50`
3. Check network connectivity between nodes
4. Verify storage availability

---

## 📈 Key Metrics to Watch

### Consensus Performance

```promql
# P95 consensus latency (target: < 50ms)
dwcp:consensus_latency:p95

# Consensus timeout rate (target: < 0.5%)
rate(dwcp_consensus_timeouts_total[5m]) / rate(dwcp_consensus_rounds_total[5m])

# Byzantine event rate
rate(dwcp_consensus_byzantine_events_total[5m])
```

### Network Performance

```promql
# Bandwidth utilization
dwcp:bandwidth:total_mbps

# Bandwidth prediction accuracy (target: > 85%)
dwcp:bandwidth:prediction_accuracy_avg

# Partition detection time (target: < 1s P95)
histogram_quantile(0.95, rate(dwcp_partition_detection_time_seconds_bucket[5m]))
```

### Migration Performance

```promql
# Migration success rate (target: > 99%)
sum(rate(dwcp_v3_migration_success_total[5m])) /
sum(rate(dwcp_v3_migration_success_total[5m]) + rate(dwcp_migration_failure_total[5m]))

# Migration duration P95
histogram_quantile(0.95, rate(dwcp_v3_migration_duration_seconds_bucket[5m]))
```

### Resource Utilization

```promql
# CPU utilization by node
dwcp_cpu_usage_percent

# Memory utilization by node
(dwcp_memory_usage_bytes / (dwcp_memory_usage_bytes + dwcp_memory_available_bytes)) * 100

# Disk I/O latency
rate(dwcp_disk_latency_ms_sum[5m]) / rate(dwcp_disk_latency_ms_count[5m])
```

---

## 🔍 Distributed Tracing

### View Traces in Jaeger

1. **Open Jaeger UI**: http://localhost:16686
2. **Select Service**: `dwcp`
3. **Select Operation**:
   - `consensus.round` - Consensus operations
   - `migration` - VM migrations
   - `ml.inference` - ML model predictions
4. **Find Traces**: Click "Find Traces"

### Trace a Specific Migration

```bash
# Get migration trace ID from logs
kubectl logs -l component=migration-manager | grep trace_id

# View trace in Jaeger
curl "http://jaeger:16686/api/traces/{trace_id}"
```

### Analyze Slow Consensus Rounds

1. Filter by latency: `minDuration=50ms`
2. Sort by duration: Descending
3. Click on slowest trace
4. Review span timeline to identify bottleneck

---

## 🤖 Anomaly Detection

### Check Anomaly Detection Status

```bash
# View current anomaly score
curl http://prometheus:9090/api/v1/query?query=dwcp_anomaly_score

# View anomaly alerts
curl http://alertmanager:9093/api/v2/alerts?filter=alertname=PerformanceAnomalyDetected
```

### Investigate Anomaly

1. **Review Alert**: Check alert annotation for anomaly score
2. **Check Correlations**: Review metrics around anomaly timestamp
3. **Examine Traces**: Look for unusual patterns in distributed traces
4. **Check Logs**: Search for errors or warnings
5. **Create Incident**: If impact confirmed, escalate to on-call

---

## 📋 Common Tasks

### Add New Metric

1. **Instrument Code**:
```go
// In your component
import "github.com/khryptorgraphics/novacron/backend/core/network/dwcp/observability"

// Record metric
collector := observability.NewConsensusMetricsCollector(logger)
collector.RecordConsensusRound(latencyMs)
```

2. **Register Collector**:
```go
aggregator.RegisterCollector(collector)
```

3. **Create Dashboard Panel**:
- Add panel to Grafana
- Use PromQL query: `dwcp_your_metric_name`
- Configure thresholds and visualization

### Adjust Alert Threshold

1. **Edit Alert Rule**: Edit `/etc/prometheus/alerts/dwcp_alerts.yml`
2. **Update Threshold**: Modify `expr` value
3. **Apply Changes**: `kubectl apply -f prometheus-alerts.yml`
4. **Verify**: Check Prometheus UI → Alerts

### Export Metrics for Analysis

```bash
# Export 24 hours of consensus latency data
curl -G http://prometheus:9090/api/v1/query_range \
  --data-urlencode 'query=dwcp:consensus_latency:p95' \
  --data-urlencode 'start=2025-11-13T00:00:00Z' \
  --data-urlencode 'end=2025-11-14T00:00:00Z' \
  --data-urlencode 'step=60s' \
  > consensus_latency_24h.json
```

---

## 🛠️ Troubleshooting

### Issue: No Metrics in Grafana

**Check**:
1. Prometheus scraping targets: http://prometheus:9090/targets
2. Verify DWCP pods expose `/metrics` endpoint
3. Check Prometheus logs: `kubectl logs prometheus-0`
4. Verify Grafana data source: Settings → Data Sources → Prometheus

### Issue: High Query Latency

**Fix**:
1. Check recording rules are enabled
2. Increase VictoriaMetrics cache: Edit `storage.cacheSizeIndexDBIndexBlocks`
3. Reduce dashboard auto-refresh interval
4. Optimize PromQL queries (use recording rules)

### Issue: Disk Space Full

**Fix**:
1. Adjust retention: Edit Prometheus `retention` config
2. Enable downsampling for warm tier
3. Move cold data to object storage
4. Clean up old time series: `curl -X POST http://prometheus:9090/api/v1/admin/tsdb/clean_tombstones`

---

## 📞 Support

**Documentation**: `/docs/performance/TELEMETRY_ARCHITECTURE.md`
**Runbooks**: `/docs/performance/runbooks/`
**Issues**: Create ticket in project tracker
**On-Call**: PagerDuty escalation policy

---

## 🎯 Quick Checks

### Daily Health Check (2 minutes)

```bash
# 1. Check all monitoring services are running
kubectl get pods -n monitoring

# 2. Verify Prometheus targets are up
curl -s http://prometheus:9090/api/v1/targets | jq '.data.activeTargets[] | select(.health!="up")'

# 3. Check critical alerts
curl -s http://alertmanager:9093/api/v2/alerts?filter=severity=critical | jq '.'

# 4. View SLO compliance
curl -s http://prometheus:9090/api/v1/query?query=dwcp:consensus_latency:p95 | jq '.data.result[0].value[1]'
```

**Expected Results**:
- All pods: Running
- All targets: Up
- Critical alerts: 0
- Consensus P95: < 0.050 (50ms)

---

**Quick Start Guide Version**: 1.0.0
**Last Updated**: 2025-11-14
