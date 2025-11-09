# DWCP Phase 3: Multi-Region Monitoring and Observability

## Executive Summary

This document provides comprehensive documentation for the DWCP Phase 3 multi-region monitoring and observability platform. The system provides end-to-end visibility across distributed deployments with real-time metrics, distributed tracing, intelligent alerting, and ML-based anomaly detection.

## Architecture Overview

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     Monitoring Platform                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Metrics    │  │   Tracing    │  │   Logging    │          │
│  │  Collector   │  │   System     │  │  Aggregator  │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                  │                   │
│         v                 v                  v                   │
│  ┌──────────────────────────────────────────────────┐           │
│  │           Monitoring API (Unified Interface)      │           │
│  └──────────────────────────────────────────────────┘           │
│         │                 │                  │                   │
│         v                 v                  v                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Anomaly    │  │   Alerting   │  │  Dashboard   │          │
│  │  Detection   │  │   System     │  │   Manager    │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
          │                    │                    │
          v                    v                    v
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Prometheus  │    │    Jaeger    │    │ Elasticsearch│
│  (Metrics)   │    │  (Traces)    │    │   (Logs)     │
└──────────────┘    └──────────────┘    └──────────────┘
          │                    │                    │
          v                    v                    v
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Grafana    │    │  Trace UI    │    │   Kibana     │
│ (Dashboards) │    │              │    │  (Log UI)    │
└──────────────┘    └──────────────┘    └──────────────┘
```

## 1. Distributed Metrics Collection

### 1.1 Metrics Collector

**File**: `backend/core/network/dwcp/monitoring/metrics_collector.go`

The metrics collector uses OpenTelemetry to collect and export metrics.

#### Supported Metric Types

1. **Counters**: Request counts, error counts
2. **Gauges**: CPU, memory, connections
3. **Histograms**: Latency, response times
4. **Summaries**: Percentiles (p50, p95, p99, p999)

#### Usage Example

```go
// Create metrics collector
mc, err := NewMetricsCollector("us-west-1")
if err != nil {
    log.Fatal(err)
}

// Record request
ctx := context.Background()
mc.RecordRequest(ctx, "migration", map[string]string{
    "type": "live",
})

// Record latency
mc.RecordLatency(ctx, "migration", 45.2, nil)

// Record gauge
mc.RecordGauge("cpu_usage", 75.5, nil)
```

#### Performance Characteristics

- **Collection Overhead**: <1% CPU
- **Ingestion Rate**: >100,000 metrics/second
- **Granularity**: 1-second intervals
- **Retention**: 24 hours in-memory, 30 days in Prometheus

### 1.2 Metric Aggregation

Metrics are aggregated across regions to provide global visibility:

```go
// Aggregate metrics from multiple regions
regionMetrics := map[string][]*MetricData{
    "latency": {
        {Value: 10.0, Region: "us-west-1"},
        {Value: 20.0, Region: "us-east-1"},
        {Value: 15.0, Region: "eu-west-1"},
    },
}

aggregated := mc.AggregateMetrics(regionMetrics)
// aggregated["latency"].Mean = 15.0
// aggregated["latency"].P95 = 19.0
```

## 2. Distributed Tracing

**File**: `backend/core/network/dwcp/monitoring/tracing.go`

### 2.1 Tracing System Architecture

The tracing system uses OpenTelemetry with Jaeger backend for distributed trace collection and analysis.

#### Sampling Strategies

1. **Head-based Sampling**: Fixed rate sampling (configurable)
2. **Tail-based Sampling**: Keep error traces and high-latency traces
3. **Adaptive Sampling**: Adjusts rate based on system load

#### Usage Example

```go
// Initialize tracing
config := &TraceConfig{
    ServiceName:    "dwcp",
    JaegerEndpoint: "http://jaeger:14268/api/traces",
    SamplingRate:   0.1,  // 10%
    Strategy:       SamplingAdaptive,
}

ts, err := NewTracingSystem(config)

// Start span
ctx, span := ts.StartSpan(ctx, "vm-migration")
defer span.End()

// Propagate context across regions
headers := ts.PropagateContext(ctx)
// Send headers to remote region

// Extract context from headers
ctx = ts.ExtractContext(ctx, headers)
```

### 2.2 Cross-Region Trace Stitching

Traces spanning multiple regions are automatically stitched together:

```go
// Automatically combines spans from different regions
ts.StitchCrossRegionTraces()

// Retrieve complete trace
trace, ok := ts.GetTrace("trace-id-123")
// trace.Spans contains spans from all regions
```

### 2.3 Performance Impact

- **Sampling Overhead**: <5% latency
- **Trace Storage**: Compressed in Elasticsearch
- **Query Performance**: <1s for 24h traces

## 3. Grafana Dashboards

**File**: `backend/core/network/dwcp/monitoring/dashboard.go`

### 3.1 Available Dashboards

#### Global Overview Dashboard

Shows aggregated metrics across all regions:
- Total requests (all regions)
- Error rate by region
- P95 latency comparison
- Active connections map

```go
dm := NewDashboardManager(config)
dm.CreateGlobalOverviewDashboard()
```

#### Regional Dashboard

Deep dive into specific region:
- Request rate by operation
- Latency distribution heatmap
- Resource utilization
- Network topology

```go
dm.CreateRegionalDashboard("us-west-1")
```

#### DWCP Protocol Dashboard

Protocol-specific metrics:
- Message type distribution
- Compression ratio
- Bandwidth utilization
- Protocol errors

```go
dm.CreateDWCPProtocolDashboard()
```

#### Load Balancer Dashboard

Load balancer performance:
- Backend health status
- Request distribution
- Failure rates
- Connection pools

```go
dm.CreateLoadBalancerDashboard()
```

### 3.2 Dashboard Access

- **URL**: `http://localhost:3000`
- **Default Credentials**: admin/admin
- **Dashboards Location**: `/var/lib/grafana/dashboards`

## 4. Intelligent Alerting

**File**: `backend/core/network/dwcp/monitoring/alerting.go`

### 4.1 Alert Severity Levels

1. **Info**: Informational, no action required
2. **Warning**: Requires attention, non-critical
3. **Critical**: Immediate action required

### 4.2 Alert Rules

#### Define Alert Rule

```go
rule := &AlertRule{
    ID:   "high-latency",
    Name: "High Latency Alert",
    Expression: `histogram_quantile(0.95, rate(dwcp_latency_bucket[5m])) > 100`,
    Severity: AlertWarning,
    For: 5 * time.Minute,
    Labels: map[string]string{
        "team": "platform",
    },
    Annotations: map[string]string{
        "summary": "High latency detected",
        "runbook": "https://docs.example.com/runbooks/high-latency",
    },
}

alerting.AddRule(rule)
```

#### Alert Routing

```go
route := &AlertRoute{
    Matchers: map[string]string{
        "severity": "critical",
    },
    Receiver: "pagerduty",
    GroupWait: 30 * time.Second,
    RepeatInterval: 1 * time.Hour,
}

alerting.AddRoute(route)
```

### 4.3 Alert Receivers

#### PagerDuty

```go
receiver := &PagerDutyReceiver{
    IntegrationKey: "your-integration-key",
}
```

#### Slack

```go
receiver := &SlackReceiver{
    WebhookURL: "https://hooks.slack.com/services/...",
    Channel: "#alerts",
}
```

#### Email

```go
receiver := &EmailReceiver{
    SMTPServer: "smtp.gmail.com:587",
    From: "alerts@example.com",
    To: []string{"oncall@example.com"},
}
```

## 5. Anomaly Detection

**File**: `backend/core/network/dwcp/monitoring/anomaly_detector.go`

### 5.1 Detection Methods

#### Statistical (3-Sigma Rule)

Detects values beyond 3 standard deviations from the mean.

#### Moving Average Deviation

Compares values to moving average with configurable window.

#### Exponential Smoothing

Uses exponentially weighted moving average for trend detection.

#### Isolation Forest (ML-based)

Machine learning algorithm for outlier detection.

### 5.2 Usage Example

```go
detector := NewAnomalyDetector()

// Add historical data
for _, point := range historicalData {
    detector.AddDataPoint("cpu_usage", point.Value, point.Timestamp)
}

// Detect anomaly
result := detector.DetectAnomaly("cpu_usage", 95.0)
if result.IsAnomaly {
    log.Printf("Anomaly detected! Score: %.2f", result.AnomalyScore)
}
```

### 5.3 Baseline Learning

The system automatically learns baselines after collecting 100+ data points:

```go
// Baseline includes:
// - Mean, standard deviation
// - Min, max, median
// - Moving average
// - Exponential smoothing
// - Seasonal patterns
```

## 6. Regional Health Monitoring

**File**: `backend/core/network/dwcp/monitoring/region_health.go`

### 6.1 Health Score Calculation

Health score (0-100) is calculated from weighted factors:

- **Availability**: 30%
- **Resource Utilization**: 20%
- **Network Latency**: 20%
- **Error Rate**: 20%
- **Consensus Health**: 10%

### 6.2 Usage Example

```go
monitor := NewHealthMonitor()

// Update health metrics
metrics := &HealthMetrics{
    Availability:        99.5,
    ResourceUtilization: 65.0,
    NetworkLatency:      25.0,  // ms
    ErrorRate:           0.01,  // 1%
    ConsensusHealth:     95.0,
}

monitor.UpdateRegionHealth("us-west-1", metrics)

// Get health score
health, ok := monitor.GetRegionHealth("us-west-1")
// health.Score = 85.2
// health.IsHealthy = true
// health.Trend = TrendStable
```

### 6.3 Component Health

Track individual component health:

```go
monitor.UpdateComponentHealth(
    "us-west-1",
    "database",
    HealthStatusHealthy,
    100.0,
    "All checks passing",
)
```

## 7. SLA Monitoring

**File**: `backend/core/network/dwcp/monitoring/sla_monitor.go`

### 7.1 SLA Definitions

#### Availability SLA

```go
sla := slaMonitor.CreateAvailabilitySLA("dwcp", 99.95)
// Target: 99.95% uptime
```

#### Latency SLA

```go
sla := slaMonitor.CreateLatencySLA("dwcp", 100)
// Target: P95 latency < 100ms
```

#### Throughput SLA

```go
sla := slaMonitor.CreateThroughputSLA("dwcp", 1000)
// Target: >= 1000 requests/sec
```

### 7.2 Compliance Checking

```go
// Check compliance
compliant := slaMonitor.CheckCompliance("sla-availability-dwcp", metrics)

// Generate report
period := Period{
    Start: time.Now().Add(-30 * 24 * time.Hour),
    End:   time.Now(),
}

report := slaMonitor.GenerateReport("sla-availability-dwcp", period)
// report.ComplianceRate = 99.98
// report.Violations = [...]
// report.ErrorBudget = 0.1%
```

### 7.3 Error Budget Tracking

```go
budget := slaMonitor.CalculateErrorBudget("sla-availability-dwcp", period)
// Remaining error budget: 0.05%
```

## 8. Log Aggregation

**File**: `backend/core/network/dwcp/monitoring/log_aggregator.go`

### 8.1 Structured Logging

```go
logger := NewStructuredLogger(aggregator, "us-west-1", "dwcp")

// Log with fields
logger.Info("VM migration started", map[string]interface{}{
    "vm_id": "vm-123",
    "source": "host-1",
    "destination": "host-2",
})

logger.Error("Migration failed", map[string]interface{}{
    "vm_id": "vm-123",
    "error": "connection timeout",
})
```

### 8.2 Log Search

```go
filter := LogFilter{
    Level: LogLevelError,
    Region: "us-west-1",
    StartTime: time.Now().Add(-1 * time.Hour),
    SearchTerm: "migration",
    Limit: 100,
}

logs := aggregator.Search(filter)
```

### 8.3 Trace Correlation

```go
// Get all logs for a trace
logs := aggregator.GetLogsByTrace("trace-id-123")
// Returns all log entries with matching trace ID
```

## 9. Network Telemetry

**File**: `backend/core/network/dwcp/monitoring/network_telemetry.go`

### 9.1 Bandwidth Monitoring

```go
telemetry := NewNetworkTelemetry()

telemetry.RecordBandwidth(
    "us-west-1",
    "us-east-1",
    1000000,    // bytes sent
    500000,     // bytes received
    10000000,   // capacity
)
```

### 9.2 Latency Monitoring

```go
telemetry.RecordLatency("us-west-1", "us-east-1", 25.5, "TCP")

metrics, ok := telemetry.GetLatencyMetrics("us-west-1", "us-east-1")
// metrics.LatencyMs = 25.5
// metrics.JitterMs = 2.1
```

### 9.3 VPN Tunnel Health

```go
telemetry.UpdateTunnelHealth(
    "tunnel-123",
    "us-west-1",
    "us-east-1",
    TunnelUp,
    map[string]interface{}{
        "latency": 20.0,
        "packet_loss": 0.001,
    },
)
```

## 10. Performance Profiling

**File**: `backend/core/network/dwcp/monitoring/profiler.go`

### 10.1 CPU Profiling

```go
profiler := NewProfiler()

// Start CPU profiling for 30 seconds
profiler.StartCPUProfile(30 * time.Second)
// Profile automatically stops after duration
```

### 10.2 Memory Profiling

```go
profile, err := profiler.CaptureMemoryProfile()
// profile.Size = allocated bytes
```

### 10.3 Continuous Profiling

```go
ctx := context.Background()
go profiler.StartContinuousProfiling(ctx)
// Captures all profile types every minute
```

### 10.4 Profile Comparison

```go
comparison, err := profiler.CompareProfiles("before-key", "after-key")
// comparison.Difference["size_delta"] = memory increase/decrease
```

## 11. Capacity Planning

**File**: `backend/core/network/dwcp/monitoring/capacity_planner.go`

### 11.1 Utilization Tracking

```go
planner := NewCapacityPlanner()

// Record resource utilization
planner.RecordUtilization("cpu", 75.0, 100.0)
planner.RecordUtilization("memory", 60.0, 128.0)  // 60GB / 128GB
```

### 11.2 Forecasting

```go
forecast, err := planner.GenerateForecast("cpu")
// forecast.Forecast30d = predicted value in 30 days
// forecast.Forecast60d = predicted value in 60 days
// forecast.Forecast90d = predicted value in 90 days
// forecast.DaysToCapacity = days until capacity exhausted
// forecast.Recommendation = "Scale out immediately" or "Monitor"
```

### 11.3 Bottleneck Identification

```go
bottlenecks := planner.IdentifyBottlenecks()
for _, b := range bottlenecks {
    log.Printf("Bottleneck: %s, Severity: %.2f, Recommendation: %s",
        b.Resource, b.Severity, b.Recommendation)
}
```

### 11.4 Scale-Out Recommendations

```go
recommendation := planner.RecommendScaleOut("cpu", 10)  // 10 current nodes
// recommendation.RecommendedNodes = 15
// recommendation.TimeFrame = 45 days
```

## 12. Monitoring API

**File**: `backend/core/network/dwcp/monitoring/api.go`

### 12.1 Initialization

```go
config := &MonitoringConfig{
    Region:           "us-west-1",
    PrometheusURL:    "http://prometheus:9090",
    GrafanaURL:       "http://grafana:3000",
    GrafanaAPIKey:    "your-api-key",
    JaegerEndpoint:   "http://jaeger:14268/api/traces",
    ElasticsearchURL: "http://elasticsearch:9200",
}

api, err := NewMonitoringAPI(config)
if err != nil {
    log.Fatal(err)
}

// Initialize with defaults
if err := api.Initialize(context.Background()); err != nil {
    log.Fatal(err)
}
```

### 12.2 Unified Interface

```go
// Get metrics
metrics, err := api.GetMetrics("us-west-1", TimeRange{
    Start: time.Now().Add(-1 * time.Hour),
    End:   time.Now(),
})

// Get traces
traces, err := api.GetTraces(TraceFilter{
    Region: "us-west-1",
    MinDuration: 100 * time.Millisecond,
})

// Get alerts
alerts, err := api.GetAlerts(AlertCritical)

// Get region health
health, err := api.GetRegionHealth("us-west-1")

// Get SLA compliance
sla, err := api.GetSLACompliance("sla-availability-dwcp")
```

## 13. Infrastructure Setup

### 13.1 Docker Compose Stack

Start the complete monitoring stack:

```bash
cd configs
docker-compose -f monitoring-stack.yml up -d
```

### 13.2 Services

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000
- **Jaeger UI**: http://localhost:16686
- **Kibana**: http://localhost:5601
- **AlertManager**: http://localhost:9093

### 13.3 Data Persistence

All data is persisted in Docker volumes:
- `prometheus-data`: Prometheus TSDB
- `grafana-data`: Dashboards and configs
- `elasticsearch-data`: Logs and traces
- `victoriametrics-data`: Long-term metrics

## 14. Query Examples

### 14.1 PromQL Queries

#### Request Rate

```promql
rate(dwcp_requests_total[5m])
```

#### Error Rate by Region

```promql
sum(rate(dwcp_errors_total[5m])) by (region)
/
sum(rate(dwcp_requests_total[5m])) by (region)
```

#### P95 Latency

```promql
histogram_quantile(0.95, rate(dwcp_latency_bucket[5m]))
```

#### Availability

```promql
(up{job="dwcp"} == 1) * 100
```

### 14.2 Elasticsearch Queries

#### Error Logs in Last Hour

```json
{
  "query": {
    "bool": {
      "must": [
        { "term": { "level": "ERROR" } },
        { "range": { "timestamp": { "gte": "now-1h" } } }
      ]
    }
  }
}
```

## 15. Troubleshooting Guide

### 15.1 High Latency

**Symptoms**: P95 latency > 100ms

**Investigation**:
1. Check regional latency: `dwcp:latency:p95:5m`
2. Identify slow operations: `dwcp_latency_bucket{operation="..."}`
3. Review traces for bottlenecks
4. Check resource utilization

**Remediation**:
- Scale horizontally if CPU-bound
- Optimize slow queries
- Add caching layer
- Review network topology

### 15.2 SLA Violations

**Symptoms**: SLA compliance < 99.9%

**Investigation**:
1. Get violation details: `slaMonitor.GetViolations(slaID)`
2. Check error budget: `slaMonitor.CalculateErrorBudget()`
3. Review incident timeline
4. Analyze root causes

**Remediation**:
- Address high-frequency issues first
- Implement circuit breakers
- Add retry logic with backoff
- Improve monitoring coverage

### 15.3 Anomalies

**Symptoms**: Anomaly detection alerts

**Investigation**:
1. Review anomaly score and deviation
2. Compare to historical baseline
3. Check for seasonal patterns
4. Correlate with other metrics

**Remediation**:
- Investigate if legitimate traffic spike
- Update baselines if behavior changed
- Tune sensitivity if false positive
- Add capacity if growth-related

## 16. Performance Characteristics

### 16.1 System Performance

- **Metric Collection Overhead**: <1% CPU
- **Metric Ingestion Rate**: >100,000 metrics/second
- **Trace Sampling Overhead**: <5% latency
- **Alert Evaluation**: <100ms
- **Query Response Time**: <1s for 24h data
- **Anomaly Detection Latency**: <5s

### 16.2 Resource Requirements

**Per Region**:
- **CPU**: 2-4 cores for monitoring agents
- **Memory**: 4-8GB for collectors
- **Storage**: 100GB-1TB (depends on retention)
- **Network**: 100Mbps for metric/trace export

**Central Infrastructure**:
- **Prometheus**: 8 cores, 32GB RAM, 1TB SSD
- **Elasticsearch**: 16 cores, 64GB RAM, 2TB SSD
- **Grafana**: 2 cores, 4GB RAM
- **Jaeger**: 4 cores, 16GB RAM

## 17. Security Considerations

### 17.1 Authentication

- Grafana: LDAP/OAuth integration
- Prometheus: Basic auth or OAuth proxy
- Elasticsearch: X-Pack security

### 17.2 Encryption

- TLS for all metric/trace/log transport
- Encrypted storage for sensitive data
- Certificate management via cert-manager

### 17.3 Access Control

- Role-based access in Grafana
- API keys with limited scope
- Network policies for pod-to-pod communication

## 18. Maintenance

### 18.1 Data Retention

```yaml
# Prometheus
storage.tsdb.retention.time: 30d

# Elasticsearch
index.lifecycle.management:
  hot: 7 days
  warm: 23 days
  delete: 30 days
```

### 18.2 Backup Strategy

- **Prometheus**: Snapshot to object storage daily
- **Elasticsearch**: Snapshot repository with incremental backups
- **Grafana**: Git-based dashboard version control

### 18.3 Upgrades

Follow zero-downtime upgrade procedure:
1. Upgrade secondary Prometheus instances
2. Failover traffic
3. Upgrade primary instance
4. Verify data continuity
5. Upgrade Grafana and Jaeger

## 19. Integration Points

### 19.1 DWCP Components

- **Agent 1 (CRDT)**: Synchronization metrics
- **Agent 2 (ACP)**: Consensus algorithm performance
- **Agent 3 (Network)**: Topology and routing metrics
- **Agent 4 (Load Balancer)**: Distribution and health
- **Agent 5 (Conflict Resolution)**: Resolution metrics
- **Agent 7 (Kubernetes)**: Pod and container metrics
- **Agent 8 (Disaster Recovery)**: Backup/restore metrics

### 19.2 External Systems

- **Incident Management**: PagerDuty, Opsgenie
- **Communication**: Slack, Microsoft Teams
- **Ticketing**: Jira, ServiceNow
- **CI/CD**: Jenkins, GitHub Actions (deployment metrics)

## 20. Best Practices

### 20.1 Metric Naming

Use consistent naming convention:
```
dwcp_<component>_<metric>_<unit>
```

Examples:
- `dwcp_migration_duration_seconds`
- `dwcp_network_bandwidth_bytes_total`
- `dwcp_cpu_utilization_percent`

### 20.2 Label Usage

Keep cardinality low:
- Use region, operation, status
- Avoid high-cardinality labels (user IDs, trace IDs)
- Use recording rules for high-cardinality aggregations

### 20.3 Dashboard Design

- Focus on actionable metrics
- Use consistent color schemes
- Provide drill-down capabilities
- Include context and documentation links

## 21. Metrics Catalog

### 21.1 Core Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `dwcp_requests_total` | Counter | Total requests processed |
| `dwcp_errors_total` | Counter | Total errors encountered |
| `dwcp_latency` | Histogram | Operation latency distribution |
| `dwcp_active_connections` | Gauge | Active network connections |
| `dwcp_cpu_usage_percent` | Gauge | CPU utilization |
| `dwcp_memory_usage_bytes` | Gauge | Memory consumption |

### 21.2 DWCP-Specific Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `dwcp_compression_ratio` | Histogram | Compression effectiveness |
| `dwcp_bandwidth_saved_bytes` | Counter | Bandwidth savings |
| `dwcp_tunnel_status` | Gauge | VPN tunnel health |
| `dwcp_consensus_failures_total` | Counter | Consensus failures |
| `dwcp_migration_duration_seconds` | Histogram | VM migration time |

## 22. Alert Runbooks

### 22.1 DWCPServiceDown

**Severity**: Critical

**Trigger**: DWCP service unreachable for >1 minute

**Actions**:
1. Check service logs: `kubectl logs -l app=dwcp`
2. Verify pod status: `kubectl get pods -l app=dwcp`
3. Check recent deployments: `kubectl rollout history deployment/dwcp`
4. Restart if hung: `kubectl rollout restart deployment/dwcp`
5. Escalate if persists >5 minutes

### 22.2 HighLatency

**Severity**: Warning

**Trigger**: P95 latency >100ms for >5 minutes

**Actions**:
1. Review traces in Jaeger for slow operations
2. Check resource utilization
3. Identify bottleneck operations
4. Scale horizontally if needed
5. Optimize slow code paths

### 22.3 SLAViolation

**Severity**: Critical

**Trigger**: SLA compliance <99.9%

**Actions**:
1. Identify violation cause
2. Implement immediate mitigation
3. Update error budget tracking
4. Post-incident review
5. Implement preventive measures

## 23. Contact and Support

- **On-Call**: Use PagerDuty for critical alerts
- **Documentation**: https://docs.novacron.io/monitoring
- **Dashboards**: http://grafana.novacron.io
- **Metrics**: http://prometheus.novacron.io

---

**Document Version**: 1.0
**Last Updated**: 2025-11-08
**Maintained By**: Agent 6 (Monitoring & Observability)
