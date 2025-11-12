# DWCP v3 Phase 6: Production Observability Guide

**Version:** 1.0.0
**Date:** 2025-11-10
**Audience:** SRE, DevOps, Platform Engineers
**Status:** Production

---

## Table of Contents

1. [Observability Stack Overview](#observability-stack-overview)
2. [Metrics Collection](#metrics-collection)
3. [Distributed Tracing](#distributed-tracing)
4. [Log Aggregation](#log-aggregation)
5. [Performance Profiling](#performance-profiling)
6. [Dashboard Guide](#dashboard-guide)
7. [Query Cookbook](#query-cookbook)
8. [Troubleshooting Workflows](#troubleshooting-workflows)
9. [Integration Guide](#integration-guide)

---

## Observability Stack Overview

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    DWCP v3 Application                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │   AMST   │  │   HDE    │  │   PBA    │  │   ACP    │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
│       │             │              │             │          │
│       └─────────────┴──────────────┴─────────────┘          │
│                         │                                    │
└─────────────────────────┼────────────────────────────────────┘
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
     ┌─────────────────┐    ┌──────────────────┐
     │   Prometheus    │    │ OpenTelemetry    │
     │   (Metrics)     │    │  Collector       │
     └────────┬────────┘    └────────┬─────────┘
              │                      │
              │         ┌────────────┴────────┐
              │         │                     │
              ▼         ▼                     ▼
     ┌──────────────────────┐      ┌─────────────────┐
     │      Grafana         │      │     Jaeger      │
     │   (Visualization)    │      │    (Traces)     │
     └──────────────────────┘      └─────────────────┘
              │
              ▼
     ┌──────────────────────┐
     │  AlertManager        │
     │  (Alerting)          │
     └──────────────────────┘
              │
              ▼
     ┌──────────────────────┐
     │   Loki / ELK         │
     │   (Logs)             │
     └──────────────────────┘
```

### Component Responsibilities

| Component | Purpose | Port | Access |
|-----------|---------|------|--------|
| **Prometheus** | Metrics storage & querying | 9090 | http://prometheus.internal |
| **Grafana** | Visualization & dashboards | 3000 | http://grafana.internal |
| **AlertManager** | Alert routing & silencing | 9093 | http://alertmanager.internal |
| **OpenTelemetry** | Trace collection | 4317 | grpc://otel.internal:4317 |
| **Jaeger** | Trace visualization | 16686 | http://jaeger.internal |
| **Loki** | Log aggregation | 3100 | http://loki.internal |

---

## Metrics Collection

### 2.1 Instrumentation Points

DWCP v3 exposes metrics at multiple levels:

```
Application Level:
  /metrics              - Main application metrics
  /metrics/summary      - JSON summary endpoint

Component Level:
  /metrics/amst         - AMST-specific metrics
  /metrics/hde          - HDE-specific metrics
  /metrics/pba          - PBA-specific metrics
  /metrics/acp          - ACP-specific metrics
  /metrics/ass          - ASS-specific metrics
  /metrics/itp          - ITP-specific metrics
```

### 2.2 Metric Types

#### Counter Metrics
Monotonically increasing counters:

```promql
dwcp_v3_migrations_total
dwcp_v3_errors_total
dwcp_v3_bytes_transferred_total
```

**Usage:**
```promql
# Migration rate
rate(dwcp_v3_migrations_total[5m])

# Error rate
rate(dwcp_v3_errors_total[1m])
```

#### Gauge Metrics
Point-in-time measurements:

```promql
dwcp_v3_throughput_bytes_per_second
dwcp_v3_component_health
dwcp_v3_active_connections
```

**Usage:**
```promql
# Current throughput
dwcp_v3_throughput_bytes_per_second

# Average over time
avg_over_time(dwcp_v3_throughput_bytes_per_second[10m])
```

#### Histogram Metrics
Distribution of values:

```promql
dwcp_v3_migration_latency_seconds_bucket
dwcp_v3_compression_ratio_bucket
```

**Usage:**
```promql
# P99 latency
histogram_quantile(0.99, sum(rate(dwcp_v3_migration_latency_seconds_bucket[5m])) by (le))

# Average latency
rate(dwcp_v3_migration_latency_seconds_sum[5m]) / rate(dwcp_v3_migration_latency_seconds_count[5m])
```

### 2.3 Custom Metrics Integration

To add custom metrics to your DWCP v3 component:

```go
import (
    "github.com/prometheus/client_golang/prometheus"
    "github.com/prometheus/client_golang/prometheus/promauto"
)

// Define metric
var myMetric = promauto.NewGaugeVec(
    prometheus.GaugeOpts{
        Name: "dwcp_v3_my_custom_metric",
        Help: "Description of my metric",
    },
    []string{"label1", "label2"},
)

// Update metric
myMetric.WithLabelValues("value1", "value2").Set(42.0)
```

### 2.4 Metric Naming Conventions

Follow these naming patterns:

```
dwcp_v3_{component}_{metric}_{unit}

Examples:
dwcp_v3_amst_throughput_bytes_per_second
dwcp_v3_hde_compression_ratio_percent
dwcp_v3_pba_prediction_accuracy_percent
```

**Labels:**
- `component`: amst, hde, pba, acp, ass, itp
- `mode`: datacenter, internet, hybrid
- `transport`: rdma, tcp
- `severity`: info, warning, critical

---

## Distributed Tracing

### 3.1 Trace Instrumentation

DWCP v3 uses OpenTelemetry for distributed tracing:

```go
import (
    "go.opentelemetry.io/otel"
    "go.opentelemetry.io/otel/trace"
)

func MigrateVM(ctx context.Context, vmID string) error {
    // Create span
    ctx, span := otel.Tracer("dwcp-v3").Start(ctx, "migrate_vm",
        trace.WithAttributes(
            attribute.String("vm.id", vmID),
            attribute.String("phase", "6"),
        ),
    )
    defer span.End()

    // Child spans
    ctx, transferSpan := otel.Tracer("dwcp-v3").Start(ctx, "transfer_data")
    // ... transfer logic ...
    transferSpan.End()

    return nil
}
```

### 3.2 Trace Visualization

Access Jaeger UI: http://jaeger.internal:16686

**Key Traces:**
- `migrate_vm` - Complete VM migration
- `amst_transfer` - Data transfer via AMST
- `hde_compress` - Compression operation
- `acp_consensus` - Consensus protocol operation

### 3.3 Trace Sampling

Production sampling configuration:

```yaml
sampling:
  type: probabilistic
  param: 0.1  # 10% of traces

# Critical paths always sampled
  always_sample:
    - "migrate_vm"
    - "acp_consensus"
    - "error_*"
```

### 3.4 Trace Analysis Queries

**Find slow migrations:**
```
service: dwcp-v3
operation: migrate_vm
duration > 500ms
```

**Find failing operations:**
```
service: dwcp-v3
tags: error=true
```

**Trace component interactions:**
```
service: dwcp-v3
tags: component=amst OR component=hde
```

---

## Log Aggregation

### 4.1 Log Structure

DWCP v3 emits structured JSON logs:

```json
{
  "timestamp": "2025-11-10T18:00:00Z",
  "level": "info",
  "component": "amst",
  "message": "Migration completed",
  "vm_id": "vm-12345",
  "latency_ms": 250,
  "throughput_gbps": 2.8,
  "trace_id": "abc123"
}
```

### 4.2 Log Levels

| Level | Usage | Example |
|-------|-------|---------|
| **DEBUG** | Development only | Detailed state information |
| **INFO** | Normal operations | Migration started/completed |
| **WARN** | Degraded performance | High latency detected |
| **ERROR** | Operation failures | Migration failed |
| **FATAL** | System failures | Component crash |

### 4.3 Log Queries (Loki)

**Find errors in last hour:**
```logql
{job="dwcp-v3"} |= "error" | json | level="error"
```

**Trace a specific migration:**
```logql
{job="dwcp-v3"} | json | vm_id="vm-12345"
```

**High latency migrations:**
```logql
{job="dwcp-v3"} | json | latency_ms > 500
```

**Component-specific logs:**
```logql
{job="dwcp-v3"} | json | component="amst"
```

### 4.4 Log Correlation

Link logs to traces using trace_id:

```logql
{job="dwcp-v3"} | json | trace_id="abc123"
```

Then view full trace in Jaeger: http://jaeger.internal/trace/abc123

---

## Performance Profiling

### 5.1 CPU Profiling

Enable CPU profiling endpoint:

```go
import _ "net/http/pprof"

go func() {
    log.Println(http.ListenAndServe("localhost:6060", nil))
}()
```

**Capture profile:**
```bash
# 30 second CPU profile
curl http://dwcp-v3:6060/debug/pprof/profile?seconds=30 > cpu.prof

# Analyze
go tool pprof cpu.prof
(pprof) top10
(pprof) web
```

### 5.2 Memory Profiling

**Capture heap profile:**
```bash
curl http://dwcp-v3:6060/debug/pprof/heap > heap.prof

# Analyze
go tool pprof heap.prof
(pprof) top10
(pprof) list functionName
```

**Live memory stats:**
```bash
curl http://dwcp-v3:6060/debug/pprof/heap?debug=1
```

### 5.3 Goroutine Profiling

**View goroutines:**
```bash
curl http://dwcp-v3:6060/debug/pprof/goroutine?debug=1
```

**Detect goroutine leaks:**
```bash
# Sample 1
curl http://dwcp-v3:6060/debug/pprof/goroutine > g1.prof

# Wait 5 minutes
sleep 300

# Sample 2
curl http://dwcp-v3:6060/debug/pprof/goroutine > g2.prof

# Compare
go tool pprof -base g1.prof g2.prof
```

### 5.4 Continuous Profiling

Production continuous profiling enabled:

```yaml
profiling:
  enabled: true
  interval: 60s
  cpu: true
  heap: true
  goroutine: true
  storage: s3://profiling-bucket/dwcp-v3/
```

Access profiles: http://grafana.internal/d/profiling

---

## Dashboard Guide

### 6.1 Primary Dashboards

#### Phase 6 Live Production Dashboard
**URL:** http://grafana.internal/d/dwcp-v3-phase6-live

**Panels:**
- GO/NO-GO Decision Indicator
- Rollout Progress by Stage
- Migration Latency (P50/P95/P99)
- Throughput Real-Time
- Component Health Matrix
- Error Distribution
- Anomaly Detection Timeline
- v1 vs v3 Comparison

**Refresh:** 5 seconds (live)

#### Real-Time Performance Dashboard
**URL:** http://grafana.internal/d/dwcp-v3-real-time

**Panels:**
- Latency Heatmap
- Throughput by Component
- Error Rate Trends
- Resource Utilization
- Network Flow Diagram

#### SLA Compliance Dashboard
**URL:** http://grafana.internal/d/dwcp-v3-sla

**Panels:**
- Overall SLA Compliance
- SLA Breakdown (Latency/Throughput/Errors)
- Violation Timeline
- Compliance Trends
- Error Budget Burn Rate

### 6.2 Component-Specific Dashboards

#### AMST Dashboard
```
Metrics:
- Throughput by stream count
- RDMA vs TCP performance
- Stream saturation
- Mode switching frequency
```

#### HDE Dashboard
```
Metrics:
- Compression ratio by algorithm
- Deduplication effectiveness
- Compression latency
- Data pattern analysis
```

#### PBA Dashboard
```
Metrics:
- Prediction accuracy
- LSTM model performance
- Bandwidth utilization
- Prediction error distribution
```

#### ACP Dashboard
```
Metrics:
- Consensus latency
- Quorum health
- Byzantine detection
- Leader election frequency
```

### 6.3 Custom Dashboard Creation

Create custom dashboard:

1. **Navigate to Grafana**
2. **Create New Dashboard**
3. **Add Panel**
4. **Select Prometheus Data Source**
5. **Write PromQL Query:**

```promql
# Example: Custom latency view
histogram_quantile(0.99,
  sum(rate(dwcp_v3_migration_latency_seconds_bucket{component=~"$component"}[5m]))
  by (le, component)
)
```

6. **Configure Visualization**
7. **Set Thresholds and Alerts**
8. **Save Dashboard**

### 6.4 Dashboard Variables

Use template variables for filtering:

```
$component:  label_values(dwcp_v3_component_health, component)
$region:     label_values(dwcp_v3_rollout_progress_percent, region)
$interval:   1m,5m,10m,30m,1h
```

---

## Query Cookbook

### 7.1 Performance Queries

**P99 Latency:**
```promql
histogram_quantile(0.99,
  sum(rate(dwcp_v3_migration_latency_seconds_bucket[5m])) by (le)
)
```

**Latency by Component:**
```promql
histogram_quantile(0.99,
  sum(rate(dwcp_v3_migration_latency_seconds_bucket[5m])) by (le, component)
)
```

**Throughput Average:**
```promql
avg(dwcp_v3_throughput_bytes_per_second) / 1e9
```

**Throughput by Transport:**
```promql
avg(dwcp_v3_throughput_bytes_per_second) by (transport) / 1e9
```

### 7.2 Reliability Queries

**Error Rate:**
```promql
sum(rate(dwcp_v3_errors_total[5m]))
```

**Error Rate by Component:**
```promql
sum(rate(dwcp_v3_errors_total[5m])) by (component)
```

**Component Health:**
```promql
dwcp_v3_component_health
```

**Uptime:**
```promql
up{job="dwcp-v3"}
```

### 7.3 Resource Queries

**CPU Usage:**
```promql
avg(dwcp_v3_resource_utilization_percent{resource_type="cpu"})
```

**Memory Usage:**
```promql
avg(dwcp_v3_resource_utilization_percent{resource_type="memory"})
```

**Goroutine Count:**
```promql
dwcp_v3_resource_utilization_percent{resource_type="goroutines"}
```

**Network Utilization:**
```promql
rate(dwcp_v3_network_bytes_total[5m]) / 1e9
```

### 7.4 SLA Queries

**Overall SLA Compliance:**
```promql
avg(dwcp_v3_sla_compliance) * 100
```

**SLA Violations (last 24h):**
```promql
sum(increase(dwcp_v3_sla_violations_total[24h]))
```

**Error Budget Remaining:**
```promql
100 - (
  sum(increase(dwcp_v3_sla_violations_total[30d])) /
  (30 * 24 * 60) * 100
)
```

### 7.5 Anomaly Detection Queries

**Anomaly Count:**
```promql
sum(increase(dwcp_v3_anomaly_detections_total[1h]))
```

**Anomaly by Type:**
```promql
sum(increase(dwcp_v3_anomaly_detections_total[1h])) by (anomaly_type)
```

**Anomaly Rate:**
```promql
rate(dwcp_v3_anomaly_detections_total[5m])
```

---

## Troubleshooting Workflows

### 8.1 High Latency Investigation

```
1. Check P99 latency:
   → histogram_quantile(0.99, ...)

2. Identify component:
   → ... by (component)

3. Check component metrics:
   → Component-specific dashboard

4. View recent logs:
   → {job="dwcp-v3"} | json | latency_ms > 500

5. Analyze traces:
   → Jaeger: duration > 500ms

6. Profile performance:
   → curl /debug/pprof/profile

7. Apply fix and verify
```

### 8.2 Throughput Degradation

```
1. Check current throughput:
   → dwcp_v3_throughput_bytes_per_second

2. Compare to baseline:
   → Historical dashboard

3. Check AMST streams:
   → dwcp_v3_amst_active_streams

4. Verify network:
   → kubectl exec -it pod -- iperf3

5. Check resource saturation:
   → dwcp_v3_resource_utilization_percent

6. Review compression:
   → dwcp_v3_hde_compression_ratio

7. Apply optimization
```

### 8.3 Error Spike Investigation

```
1. Get error rate:
   → sum(rate(dwcp_v3_errors_total[5m]))

2. Break down by type:
   → ... by (error_type)

3. Break down by component:
   → ... by (component)

4. View error logs:
   → {job="dwcp-v3"} | json | level="error"

5. Check component health:
   → dwcp_v3_component_health

6. Trace failing requests:
   → Jaeger: tags: error=true

7. Apply remediation
```

### 8.4 Memory Leak Detection

```
1. Check memory trend:
   → dwcp_v3_resource_utilization_percent{resource_type="memory"}

2. Capture heap profile:
   → curl /debug/pprof/heap

3. Analyze allocations:
   → go tool pprof heap.prof

4. Check goroutine count:
   → curl /debug/pprof/goroutine?debug=1

5. Identify leak source:
   → pprof analysis

6. Apply fix and verify:
   → Monitor memory over time
```

---

## Integration Guide

### 9.1 Prometheus Integration

**Add scrape target:**

```yaml
scrape_configs:
  - job_name: 'my-service'
    scrape_interval: 5s
    static_configs:
      - targets: ['my-service:8080']
```

**Add recording rule:**

```yaml
groups:
  - name: my_rules
    interval: 15s
    rules:
      - record: my_service:latency_p99:5m
        expr: histogram_quantile(0.99, ...)
```

### 9.2 Grafana Integration

**Add data source:**
```
Configuration → Data Sources → Add Prometheus
URL: http://prometheus:9090
```

**Import dashboard:**
```
Dashboard → Import → Upload JSON
File: phase6-production-live.json
```

### 9.3 AlertManager Integration

**Add notification channel:**

```yaml
receivers:
  - name: 'slack'
    slack_configs:
      - api_url: '<slack_webhook_url>'
        channel: '#dwcp-v3-alerts'
```

### 9.4 OpenTelemetry Integration

**Initialize tracer:**

```go
import (
    "go.opentelemetry.io/otel"
    "go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
)

func initTracer() {
    exporter, _ := otlptracegrpc.New(context.Background(),
        otlptracegrpc.WithEndpoint("otel-collector:4317"),
        otlptracegrpc.WithInsecure(),
    )
    // ... configure tracer provider
}
```

---

## Best Practices

### 10.1 Metric Best Practices

- ✅ Use descriptive metric names
- ✅ Include units in metric names
- ✅ Keep cardinality low (<100k)
- ✅ Use consistent label names
- ❌ Don't use high-cardinality labels (user_id, trace_id)
- ❌ Don't expose sensitive data in metrics

### 10.2 Tracing Best Practices

- ✅ Trace critical paths
- ✅ Add context via attributes
- ✅ Sample appropriately (10% production)
- ✅ Always sample errors
- ❌ Don't create too many spans (keep < 50 per trace)
- ❌ Don't include PII in span attributes

### 10.3 Logging Best Practices

- ✅ Use structured logging (JSON)
- ✅ Include trace_id for correlation
- ✅ Log at appropriate levels
- ✅ Include context (vm_id, component)
- ❌ Don't log sensitive data
- ❌ Don't log in hot paths (use metrics/traces)

---

**Document Version:** 1.0.0
**Last Updated:** 2025-11-10
**Maintained By:** Platform SRE Team
**Feedback:** sre-team@company.com
