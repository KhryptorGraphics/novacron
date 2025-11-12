# NovaCron Monitoring & Alerting Strategy

## Overview

Comprehensive monitoring and alerting configuration for NovaCron production deployment with DWCP v3 protocol integration.

## Monitoring Stack

### Components

```
┌─────────────────────────────────────────────────────────────────┐
│                     Monitoring Architecture                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Application Layer                                               │
│  ├── NovaCron API (metrics endpoint: :9090/metrics)            │
│  ├── DWCP Protocol (custom metrics)                             │
│  └── Frontend (browser metrics)                                 │
│                          │                                       │
│                          ▼                                       │
│  Metrics Collection                                              │
│  ├── Prometheus (scraping, storage, queries)                    │
│  ├── Node Exporter (system metrics)                             │
│  └── cAdvisor (container metrics)                               │
│                          │                                       │
│                          ▼                                       │
│  Visualization & Alerting                                        │
│  ├── Grafana (dashboards)                                       │
│  ├── Alertmanager (alert routing)                               │
│  └── Slack/PagerDuty (notifications)                            │
│                          │                                       │
│                          ▼                                       │
│  Log Aggregation                                                 │
│  ├── Loki (log storage)                                         │
│  ├── Promtail (log collection)                                  │
│  └── Grafana (log visualization)                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Prometheus Configuration

### prometheus.yml

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'novacron-production'
    environment: 'production'

# Alertmanager configuration
alerting:
  alertmanagers:
  - static_configs:
    - targets:
      - alertmanager:9093

# Load rules
rule_files:
  - '/etc/prometheus/rules/*.yml'

# Scrape configurations
scrape_configs:
  # Prometheus self-monitoring
  - job_name: 'prometheus'
    static_configs:
    - targets: ['localhost:9090']

  # NovaCron API
  - job_name: 'novacron-api'
    kubernetes_sd_configs:
    - role: pod
      namespaces:
        names:
        - novacron
    relabel_configs:
    - source_labels: [__meta_kubernetes_pod_label_app]
      regex: novacron
      action: keep
    - source_labels: [__meta_kubernetes_pod_label_component]
      regex: api
      action: keep
    - source_labels: [__meta_kubernetes_pod_container_port_name]
      regex: metrics
      action: keep
    - source_labels: [__meta_kubernetes_pod_name]
      target_label: pod
    - source_labels: [__meta_kubernetes_namespace]
      target_label: namespace

  # Node Exporter (system metrics)
  - job_name: 'node-exporter'
    kubernetes_sd_configs:
    - role: node
    relabel_configs:
    - source_labels: [__address__]
      regex: '(.*):10250'
      replacement: '${1}:9100'
      target_label: __address__

  # Kubernetes API Server
  - job_name: 'kubernetes-apiservers'
    kubernetes_sd_configs:
    - role: endpoints
    scheme: https
    tls_config:
      ca_file: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
    bearer_token_file: /var/run/secrets/kubernetes.io/serviceaccount/token
    relabel_configs:
    - source_labels: [__meta_kubernetes_namespace, __meta_kubernetes_service_name, __meta_kubernetes_endpoint_port_name]
      action: keep
      regex: default;kubernetes;https

  # cAdvisor (container metrics)
  - job_name: 'kubernetes-cadvisor'
    kubernetes_sd_configs:
    - role: node
    scheme: https
    tls_config:
      ca_file: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
    bearer_token_file: /var/run/secrets/kubernetes.io/serviceaccount/token
    relabel_configs:
    - action: labelmap
      regex: __meta_kubernetes_node_label_(.+)
    - target_label: __address__
      replacement: kubernetes.default.svc:443
    - source_labels: [__meta_kubernetes_node_name]
      regex: (.+)
      target_label: __metrics_path__
      replacement: /api/v1/nodes/${1}/proxy/metrics/cadvisor

  # PostgreSQL Exporter
  - job_name: 'postgres-exporter'
    static_configs:
    - targets: ['postgres-exporter:9187']

  # Redis Exporter
  - job_name: 'redis-exporter'
    static_configs:
    - targets: ['redis-exporter:9121']
```

### Alert Rules

#### application-alerts.yml

```yaml
groups:
- name: application-critical
  interval: 30s
  rules:
  # Service Down
  - alert: ServiceDown
    expr: up{job="novacron-api"} == 0
    for: 1m
    labels:
      severity: critical
      component: api
    annotations:
      summary: "NovaCron API service is down"
      description: "API service {{ $labels.pod }} has been down for more than 1 minute"
      runbook_url: "https://docs.novacron.local/runbooks/service-down"

  # High Error Rate
  - alert: HighErrorRate
    expr: |
      (
        sum(rate(http_requests_total{status=~"5..", job="novacron-api"}[5m]))
        /
        sum(rate(http_requests_total{job="novacron-api"}[5m]))
      ) > 0.01
    for: 2m
    labels:
      severity: critical
      component: api
    annotations:
      summary: "High 5xx error rate detected"
      description: "Error rate is {{ $value | humanizePercentage }} over the last 5 minutes"
      runbook_url: "https://docs.novacron.local/runbooks/high-error-rate"

  # High Response Time
  - alert: HighResponseTime
    expr: |
      histogram_quantile(0.95,
        sum(rate(http_request_duration_seconds_bucket{job="novacron-api"}[5m]))
        by (le)
      ) > 2
    for: 5m
    labels:
      severity: warning
      component: api
    annotations:
      summary: "High API response time"
      description: "95th percentile response time is {{ $value }}s (threshold: 2s)"
      runbook_url: "https://docs.novacron.local/runbooks/high-latency"

  # High Memory Usage
  - alert: HighMemoryUsage
    expr: |
      (
        container_memory_working_set_bytes{pod=~"novacron-api-.*"}
        /
        container_spec_memory_limit_bytes{pod=~"novacron-api-.*"}
      ) > 0.85
    for: 5m
    labels:
      severity: warning
      component: api
    annotations:
      summary: "High memory usage"
      description: "Pod {{ $labels.pod }} memory usage is {{ $value | humanizePercentage }}"
      runbook_url: "https://docs.novacron.local/runbooks/high-memory"

  # High CPU Usage
  - alert: HighCPUUsage
    expr: |
      (
        sum(rate(container_cpu_usage_seconds_total{pod=~"novacron-api-.*"}[5m]))
        by (pod)
        /
        sum(container_spec_cpu_quota{pod=~"novacron-api-.*"})
        by (pod)
        * 100000
      ) > 0.80
    for: 5m
    labels:
      severity: warning
      component: api
    annotations:
      summary: "High CPU usage"
      description: "Pod {{ $labels.pod }} CPU usage is {{ $value | humanizePercentage }}"
      runbook_url: "https://docs.novacron.local/runbooks/high-cpu"

  # Pod Restart
  - alert: PodRestartingFrequently
    expr: rate(kube_pod_container_status_restarts_total{pod=~"novacron-api-.*"}[15m]) > 0.1
    for: 5m
    labels:
      severity: warning
      component: api
    annotations:
      summary: "Pod restarting frequently"
      description: "Pod {{ $labels.pod }} has restarted {{ $value }} times in the last 15 minutes"
      runbook_url: "https://docs.novacron.local/runbooks/pod-crashes"
```

#### dwcp-alerts.yml

```yaml
groups:
- name: dwcp-protocol
  interval: 30s
  rules:
  # High DWCP Latency
  - alert: DWCPHighLatency
    expr: dwcp_latency_milliseconds > 100
    for: 5m
    labels:
      severity: warning
      component: dwcp
    annotations:
      summary: "DWCP protocol experiencing high latency"
      description: "Average DWCP latency is {{ $value }}ms (threshold: 100ms)"
      runbook_url: "https://docs.novacron.local/runbooks/dwcp-latency"

  # Packet Loss
  - alert: DWCPPacketLoss
    expr: dwcp_packet_loss_rate > 0.01
    for: 2m
    labels:
      severity: warning
      component: dwcp
    annotations:
      summary: "DWCP packet loss detected"
      description: "Packet loss rate is {{ $value | humanizePercentage }}"
      runbook_url: "https://docs.novacron.local/runbooks/dwcp-packet-loss"

  # Bandwidth Exhausted
  - alert: DWCPBandwidthExhausted
    expr: |
      (
        dwcp_bandwidth_allocated_bytes
        /
        dwcp_bandwidth_available_bytes
      ) > 0.90
    for: 5m
    labels:
      severity: warning
      component: dwcp
    annotations:
      summary: "DWCP bandwidth nearly exhausted"
      description: "Bandwidth utilization at {{ $value | humanizePercentage }}"
      runbook_url: "https://docs.novacron.local/runbooks/dwcp-bandwidth"

  # Compression Ratio Low
  - alert: DWCPLowCompressionRatio
    expr: dwcp_compression_ratio < 1.5
    for: 10m
    labels:
      severity: info
      component: dwcp
    annotations:
      summary: "DWCP compression ratio is low"
      description: "Compression ratio is {{ $value }} (expected: > 1.5)"
      runbook_url: "https://docs.novacron.local/runbooks/dwcp-compression"

  # Stream Overflow
  - alert: DWCPStreamOverflow
    expr: dwcp_streams_active > dwcp_streams_max * 0.9
    for: 5m
    labels:
      severity: warning
      component: dwcp
    annotations:
      summary: "DWCP stream count approaching limit"
      description: "Active streams: {{ $value }} (max: {{ dwcp_streams_max }})"
      runbook_url: "https://docs.novacron.local/runbooks/dwcp-streams"
```

#### database-alerts.yml

```yaml
groups:
- name: database
  interval: 30s
  rules:
  # Connection Pool Exhausted
  - alert: DatabaseConnectionPoolExhausted
    expr: |
      (
        sum(pg_stat_database_numbackends)
        /
        sum(pg_settings_max_connections)
      ) > 0.90
    for: 2m
    labels:
      severity: critical
      component: database
    annotations:
      summary: "Database connection pool exhausted"
      description: "Connection pool is at {{ $value | humanizePercentage }} capacity"
      runbook_url: "https://docs.novacron.local/runbooks/db-connections"

  # Slow Queries
  - alert: DatabaseSlowQueries
    expr: |
      rate(pg_stat_statements_mean_exec_time_seconds[5m]) > 1
    for: 5m
    labels:
      severity: warning
      component: database
    annotations:
      summary: "Database experiencing slow queries"
      description: "Average query execution time is {{ $value }}s"
      runbook_url: "https://docs.novacron.local/runbooks/db-slow-queries"

  # Replication Lag
  - alert: DatabaseReplicationLag
    expr: pg_replication_lag_seconds > 30
    for: 5m
    labels:
      severity: warning
      component: database
    annotations:
      summary: "Database replication lag detected"
      description: "Replication lag is {{ $value }}s (threshold: 30s)"
      runbook_url: "https://docs.novacron.local/runbooks/db-replication-lag"

  # High Transaction Rate
  - alert: DatabaseHighTransactionRate
    expr: rate(pg_stat_database_xact_commit[5m]) > 10000
    for: 5m
    labels:
      severity: info
      component: database
    annotations:
      summary: "High database transaction rate"
      description: "Transaction rate is {{ $value }} per second"

  # Deadlocks
  - alert: DatabaseDeadlocks
    expr: rate(pg_stat_database_deadlocks[5m]) > 0.1
    for: 5m
    labels:
      severity: warning
      component: database
    annotations:
      summary: "Database deadlocks detected"
      description: "Deadlock rate is {{ $value }} per second"
      runbook_url: "https://docs.novacron.local/runbooks/db-deadlocks"
```

#### cache-alerts.yml

```yaml
groups:
- name: cache
  interval: 30s
  rules:
  # Cache Down
  - alert: CacheDown
    expr: redis_up == 0
    for: 1m
    labels:
      severity: critical
      component: cache
    annotations:
      summary: "Redis cache is down"
      description: "Redis instance is unreachable"
      runbook_url: "https://docs.novacron.local/runbooks/cache-down"

  # High Memory Usage
  - alert: CacheHighMemoryUsage
    expr: |
      (
        redis_memory_used_bytes
        /
        redis_memory_max_bytes
      ) > 0.85
    for: 5m
    labels:
      severity: warning
      component: cache
    annotations:
      summary: "Redis memory usage high"
      description: "Memory usage is {{ $value | humanizePercentage }}"
      runbook_url: "https://docs.novacron.local/runbooks/cache-memory"

  # Low Hit Rate
  - alert: CacheLowHitRate
    expr: |
      (
        rate(redis_keyspace_hits_total[5m])
        /
        (rate(redis_keyspace_hits_total[5m]) + rate(redis_keyspace_misses_total[5m]))
      ) < 0.80
    for: 10m
    labels:
      severity: info
      component: cache
    annotations:
      summary: "Cache hit rate is low"
      description: "Hit rate is {{ $value | humanizePercentage }} (expected: > 80%)"
      runbook_url: "https://docs.novacron.local/runbooks/cache-hit-rate"

  # Evictions
  - alert: CacheHighEvictionRate
    expr: rate(redis_evicted_keys_total[5m]) > 100
    for: 5m
    labels:
      severity: warning
      component: cache
    annotations:
      summary: "High cache eviction rate"
      description: "Eviction rate is {{ $value }} keys per second"
      runbook_url: "https://docs.novacron.local/runbooks/cache-evictions"
```

## Alertmanager Configuration

### alertmanager.yml

```yaml
global:
  resolve_timeout: 5m
  slack_api_url: 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL'

# Alert routing
route:
  receiver: 'default'
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 12h

  routes:
  # Critical alerts -> PagerDuty
  - match:
      severity: critical
    receiver: 'pagerduty-critical'
    continue: true

  # Critical alerts -> Slack #incidents
  - match:
      severity: critical
    receiver: 'slack-incidents'

  # Warning alerts -> Slack #alerts
  - match:
      severity: warning
    receiver: 'slack-alerts'

  # Info alerts -> Slack #monitoring
  - match:
      severity: info
    receiver: 'slack-monitoring'

  # DWCP-specific alerts
  - match:
      component: dwcp
    receiver: 'slack-dwcp'

# Inhibition rules (prevent alert spam)
inhibit_rules:
  # Inhibit warning/info if critical alert is firing
  - source_match:
      severity: 'critical'
    target_match_re:
      severity: 'warning|info'
    equal: ['alertname', 'cluster', 'service']

  # Inhibit individual pod alerts if service is down
  - source_match:
      alertname: 'ServiceDown'
    target_match_re:
      alertname: 'HighMemoryUsage|HighCPUUsage|PodRestartingFrequently'
    equal: ['cluster', 'namespace']

# Receivers
receivers:
- name: 'default'
  slack_configs:
  - channel: '#monitoring'
    title: '{{ .GroupLabels.alertname }}'
    text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'

- name: 'pagerduty-critical'
  pagerduty_configs:
  - service_key: 'YOUR_PAGERDUTY_KEY'
    severity: '{{ .GroupLabels.severity }}'
    description: '{{ .GroupLabels.alertname }}'
    details:
      firing: '{{ .Alerts.Firing | len }}'
      resolved: '{{ .Alerts.Resolved | len }}'

- name: 'slack-incidents'
  slack_configs:
  - channel: '#incidents'
    color: 'danger'
    title: '🚨 CRITICAL: {{ .GroupLabels.alertname }}'
    text: |
      *Severity:* {{ .GroupLabels.severity }}
      *Component:* {{ .GroupLabels.component }}
      {{ range .Alerts }}
      *Alert:* {{ .Annotations.summary }}
      *Description:* {{ .Annotations.description }}
      *Runbook:* {{ .Annotations.runbook_url }}
      {{ end }}

- name: 'slack-alerts'
  slack_configs:
  - channel: '#alerts'
    color: 'warning'
    title: '⚠️ WARNING: {{ .GroupLabels.alertname }}'
    text: |
      *Severity:* {{ .GroupLabels.severity }}
      *Component:* {{ .GroupLabels.component }}
      {{ range .Alerts }}
      *Alert:* {{ .Annotations.summary }}
      *Description:* {{ .Annotations.description }}
      {{ end }}

- name: 'slack-monitoring'
  slack_configs:
  - channel: '#monitoring'
    color: 'good'
    title: 'ℹ️ INFO: {{ .GroupLabels.alertname }}'
    text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'

- name: 'slack-dwcp'
  slack_configs:
  - channel: '#dwcp-protocol'
    title: '📡 DWCP Alert: {{ .GroupLabels.alertname }}'
    text: |
      {{ range .Alerts }}
      *{{ .Annotations.summary }}*
      {{ .Annotations.description }}
      {{ end }}
```

## Grafana Dashboards

### NovaCron Production Dashboard

**Panel Configuration:**

1. **Request Rate**
```promql
sum(rate(http_requests_total{job="novacron-api"}[5m])) by (method, status)
```

2. **Error Rate**
```promql
sum(rate(http_requests_total{job="novacron-api", status=~"5.."}[5m]))
/
sum(rate(http_requests_total{job="novacron-api"}[5m]))
* 100
```

3. **Response Time (Percentiles)**
```promql
histogram_quantile(0.50, sum(rate(http_request_duration_seconds_bucket{job="novacron-api"}[5m])) by (le))
histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{job="novacron-api"}[5m])) by (le))
histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket{job="novacron-api"}[5m])) by (le))
```

4. **Active Connections**
```promql
sum(http_connections_active{job="novacron-api"})
```

5. **DWCP Active Streams**
```promql
dwcp_streams_active
```

6. **DWCP Bandwidth Utilization**
```promql
(dwcp_bandwidth_allocated_bytes / dwcp_bandwidth_available_bytes) * 100
```

7. **DWCP Compression Ratio**
```promql
dwcp_compression_ratio
```

8. **Database Connections**
```promql
sum(pg_stat_database_numbackends)
```

9. **Cache Hit Ratio**
```promql
(
  rate(redis_keyspace_hits_total[5m])
  /
  (rate(redis_keyspace_hits_total[5m]) + rate(redis_keyspace_misses_total[5m]))
) * 100
```

10. **CPU Usage**
```promql
sum(rate(container_cpu_usage_seconds_total{pod=~"novacron-api-.*"}[5m])) by (pod)
```

11. **Memory Usage**
```promql
sum(container_memory_working_set_bytes{pod=~"novacron-api-.*"}) by (pod)
```

12. **Network I/O**
```promql
sum(rate(container_network_receive_bytes_total{pod=~"novacron-api-.*"}[5m])) by (pod)
sum(rate(container_network_transmit_bytes_total{pod=~"novacron-api-.*"}[5m])) by (pod)
```

### DWCP Protocol Dashboard

Dedicated dashboard for DWCP protocol metrics with detailed views of:
- Stream management
- Bandwidth allocation
- Compression efficiency
- Latency distribution
- Packet loss tracking
- Error rates by stream

## Log Aggregation (Loki)

### Promtail Configuration

```yaml
server:
  http_listen_port: 9080
  grpc_listen_port: 0

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
  - job_name: kubernetes-pods
    kubernetes_sd_configs:
    - role: pod
    relabel_configs:
    - source_labels: [__meta_kubernetes_pod_label_app]
      regex: novacron
      action: keep
    - source_labels: [__meta_kubernetes_namespace]
      target_label: namespace
    - source_labels: [__meta_kubernetes_pod_name]
      target_label: pod
    - source_labels: [__meta_kubernetes_container_name]
      target_label: container
    pipeline_stages:
    - json:
        expressions:
          level: level
          timestamp: timestamp
          message: message
    - labels:
        level:
    - timestamp:
        source: timestamp
        format: RFC3339
```

### Useful Log Queries

**Error logs:**
```logql
{app="novacron", namespace="novacron"} |= "error" | json
```

**DWCP protocol errors:**
```logql
{app="novacron", namespace="novacron", component="dwcp"} |= "error" | json
```

**Slow queries:**
```logql
{app="novacron", namespace="novacron"} |= "slow query" | json | duration > 1s
```

**5xx errors:**
```logql
{app="novacron", namespace="novacron"} | json | status >= 500
```

## Health Check Endpoints

### Application Health

**Basic health check:**
```bash
GET /health
Response: {"status": "healthy", "timestamp": "2025-11-10T15:30:00Z"}
```

**Detailed health check:**
```bash
GET /health/detailed
Response:
{
  "status": "healthy",
  "timestamp": "2025-11-10T15:30:00Z",
  "components": {
    "database": {
      "status": "healthy",
      "latency_ms": 5,
      "connections": 15
    },
    "cache": {
      "status": "healthy",
      "latency_ms": 2,
      "hit_rate": 0.85
    },
    "dwcp": {
      "status": "healthy",
      "active_streams": 150,
      "bandwidth_utilization": 0.65
    }
  },
  "version": "v1.6.0",
  "uptime_seconds": 3600
}
```

## Key Metrics Reference

### Application Metrics
- `http_requests_total` - Total HTTP requests
- `http_request_duration_seconds` - Request duration histogram
- `http_connections_active` - Active HTTP connections
- `go_goroutines` - Number of goroutines
- `go_memstats_alloc_bytes` - Allocated memory

### DWCP Protocol Metrics
- `dwcp_streams_active` - Active DWCP streams
- `dwcp_bandwidth_allocated_bytes` - Allocated bandwidth
- `dwcp_bandwidth_available_bytes` - Available bandwidth
- `dwcp_compression_ratio` - Compression ratio
- `dwcp_packet_loss_rate` - Packet loss rate
- `dwcp_latency_milliseconds` - Protocol latency

### Database Metrics
- `pg_stat_database_numbackends` - Active connections
- `pg_stat_database_xact_commit` - Committed transactions
- `pg_stat_statements_mean_exec_time_seconds` - Query execution time
- `pg_replication_lag_seconds` - Replication lag

### Cache Metrics
- `redis_memory_used_bytes` - Memory usage
- `redis_keyspace_hits_total` - Cache hits
- `redis_keyspace_misses_total` - Cache misses
- `redis_evicted_keys_total` - Evicted keys

---

**Document Version:** 1.0.0
**Last Updated:** 2025-11-10
**Owner:** DevOps Team
