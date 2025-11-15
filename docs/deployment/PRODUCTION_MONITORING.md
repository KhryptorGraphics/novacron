# Production Monitoring & Alerting - DWCP Systems

## Overview

This document defines comprehensive monitoring, alerting, and observability strategy for all 6 DWCP production systems. Real-time metrics, automated alerts, and dashboards ensure early issue detection and rapid response.

---

## Monitoring Architecture

### Technology Stack

```yaml
monitoring-stack:
  metrics:
    collection: prometheus
    storage: thanos (long-term)
    query: promql

  visualization:
    dashboards: grafana
    custom-ui: react-dashboard

  alerting:
    manager: alertmanager
    routing: pagerduty
    escalation: opsgenie

  tracing:
    distributed: jaeger
    apm: datadog-apm

  logging:
    collection: fluentd
    storage: elasticsearch
    query: kibana
```

### Data Flow

```
Application Metrics (Prometheus format)
    ↓
Prometheus Scraping (15s interval)
    ↓
Thanos (Long-term Storage)
    ↓
    ├→ Grafana Dashboards (Visualization)
    ├→ AlertManager (Alerting)
    └→ Custom Analysis (ML anomaly detection)
```

---

## Prometheus Metrics Configuration

### Scrape Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'dwcp-production'
    environment: 'prod'

scrape_configs:
  - job_name: 'dwcp-manager'
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        action: keep
        regex: dwcp-manager
      - source_labels: [__meta_kubernetes_pod_name]
        target_label: instance

  - job_name: 'compression-selector'
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        action: keep
        regex: compression-selector

  - job_name: 'probft-consensus'
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        action: keep
        regex: probft-consensus

  - job_name: 'bullshark-consensus'
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        action: keep
        regex: bullshark-consensus

  - job_name: 'tpbft-consensus'
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
        - source_labels: [__meta_kubernetes_pod_label_app]
          action: keep
          regex: tpbft-consensus

  - job_name: 'maddpg-allocator'
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        action: keep
        regex: maddpg-allocator

  - job_name: 'kubernetes-nodes'
    kubernetes_sd_configs:
      - role: node
    scheme: https
    tls_config:
      ca_file: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
```

### Application Instrumentation

**Go Application Example (DWCP Manager):**

```go
package main

import (
    "github.com/prometheus/client_golang/prometheus"
    "github.com/prometheus/client_golang/prometheus/promauto"
    "github.com/prometheus/client_golang/prometheus/promhttp"
)

var (
    // Health check metrics
    healthCheckTotal = promauto.NewCounterVec(
        prometheus.CounterOpts{
            Name: "dwcp_health_check_total",
            Help: "Total number of health checks performed",
        },
        []string{"status", "version"},
    )

    healthCheckDuration = promauto.NewHistogramVec(
        prometheus.HistogramOpts{
            Name: "dwcp_health_check_duration_seconds",
            Help: "Duration of health checks in seconds",
            Buckets: prometheus.DefBuckets,
        },
        []string{"status", "version"},
    )

    // Recovery metrics
    recoveryAttempts = promauto.NewCounterVec(
        prometheus.CounterOpts{
            Name: "dwcp_recovery_attempts_total",
            Help: "Total number of recovery attempts",
        },
        []string{"component", "success", "version"},
    )

    recoveryDuration = promauto.NewHistogramVec(
        prometheus.HistogramOpts{
            Name: "dwcp_recovery_duration_seconds",
            Help: "Duration of recovery operations in seconds",
            Buckets: []float64{0.1, 0.5, 1, 2, 5, 10, 30},
        },
        []string{"component", "version"},
    )

    // Circuit breaker metrics
    circuitBreakerState = promauto.NewGaugeVec(
        prometheus.GaugeOpts{
            Name: "dwcp_circuit_breaker_state",
            Help: "Circuit breaker state (0=closed, 1=open, 2=half-open)",
        },
        []string{"circuit", "version"},
    )

    circuitBreakerTrips = promauto.NewCounterVec(
        prometheus.CounterOpts{
            Name: "dwcp_circuit_breaker_trips_total",
            Help: "Total number of circuit breaker trips",
        },
        []string{"circuit", "version"},
    )

    // Resource metrics
    memoryUsageBytes = promauto.NewGaugeVec(
        prometheus.GaugeOpts{
            Name: "dwcp_memory_usage_bytes",
            Help: "Current memory usage in bytes",
        },
        []string{"type", "version"},
    )

    goroutines = promauto.NewGaugeVec(
        prometheus.GaugeOpts{
            Name: "dwcp_goroutines",
            Help: "Current number of goroutines",
        },
        []string{"version"},
    )
)

func main() {
    // Expose /metrics endpoint
    http.Handle("/metrics", promhttp.Handler())
    http.ListenAndServe(":2112", nil)
}
```

---

## Key Metrics by System

### 1. DWCP Manager Metrics

```yaml
dwcp-manager-metrics:
  health:
    - dwcp_health_check_total{status, version}
    - dwcp_health_check_duration_seconds{status, version}
    - dwcp_component_status{component, status, version}

  recovery:
    - dwcp_recovery_attempts_total{component, success, version}
    - dwcp_recovery_duration_seconds{component, version}
    - dwcp_recovery_success_rate{component, version}

  circuit-breaker:
    - dwcp_circuit_breaker_state{circuit, version}
    - dwcp_circuit_breaker_trips_total{circuit, version}
    - dwcp_circuit_breaker_resets_total{circuit, version}

  performance:
    - dwcp_request_duration_seconds{operation, version}
    - dwcp_requests_total{operation, status, version}
    - dwcp_errors_total{type, version}

  resources:
    - dwcp_memory_usage_bytes{type, version}
    - dwcp_goroutines{version}
    - dwcp_cpu_usage_seconds{version}
```

### 2. Compression Selector Metrics

```yaml
compression-metrics:
  accuracy:
    - compression_algorithm_selected_total{algorithm, version}
    - compression_accuracy{algorithm, version}
    - compression_ratio_actual{algorithm, version}
    - compression_ratio_expected{algorithm, version}

  performance:
    - compression_selection_duration_seconds{version}
    - compression_overhead_percentage{version}
    - compression_throughput_bytes_per_second{version}

  ml-model:
    - model_inference_duration_seconds{model_version}
    - model_predictions_total{prediction, confidence, model_version}
    - model_accuracy_rolling_7d{model_version}
    - model_drift_score{model_version}

  fallback:
    - compression_fallback_total{reason, version}
    - compression_default_algorithm_used{version}
```

### 3. ProBFT Consensus Metrics

```yaml
probft-metrics:
  consensus:
    - probft_consensus_rounds_total{version}
    - probft_consensus_duration_seconds{version}
    - probft_consensus_failures_total{reason, version}

  byzantine:
    - probft_byzantine_events_detected{type, version}
    - probft_byzantine_nodes_total{status, version}
    - probft_signature_verification_failures{version}

  performance:
    - probft_message_count_per_round{message_type, version}
    - probft_network_latency_seconds{peer, version}
    - probft_throughput_transactions_per_second{version}

  security:
    - probft_view_changes_total{reason, version}
    - probft_quorum_availability{version}
    - probft_node_health{node, status, version}
```

### 4. Bullshark Consensus Metrics

```yaml
bullshark-metrics:
  throughput:
    - bullshark_transactions_per_second{version}
    - bullshark_transactions_processed_total{version}
    - bullshark_transaction_confirmation_duration_seconds{version}

  dag:
    - bullshark_dag_depth{version}
    - bullshark_dag_width{version}
    - bullshark_dag_vertices_total{version}
    - bullshark_dag_validation_errors{type, version}

  mempool:
    - bullshark_mempool_size{version}
    - bullshark_mempool_usage_percentage{version}
    - bullshark_transactions_rejected{reason, version}

  performance:
    - bullshark_batch_processing_duration_seconds{version}
    - bullshark_confirmation_latency_seconds{percentile, version}
```

### 5. T-PBFT Consensus Metrics

```yaml
tpbft-metrics:
  trust:
    - tpbft_node_trust_score{node, version}
    - tpbft_trust_score_changes{node, direction, version}
    - tpbft_trust_anomalies_detected{node, version}

  performance:
    - tpbft_message_reduction_percentage{version}
    - tpbft_consensus_rounds_per_second{version}
    - tpbft_performance_improvement{version}

  optimization:
    - tpbft_messages_saved_total{version}
    - tpbft_bandwidth_saved_bytes{version}
```

### 6. MADDPG Resource Allocator Metrics

```yaml
maddpg-metrics:
  allocation:
    - maddpg_resource_allocation{service, resource_type, version}
    - maddpg_allocation_efficiency{version}
    - maddpg_allocation_changes_total{service, version}

  performance:
    - maddpg_reward_score{version}
    - maddpg_allocation_quality{version}
    - maddpg_resource_utilization{service, resource_type, version}

  ml-model:
    - maddpg_model_training_episodes{version}
    - maddpg_model_convergence{version}
    - maddpg_exploration_rate{version}

  safety:
    - maddpg_constraint_violations{type, version}
    - maddpg_resource_starvation_events{service, version}
```

---

## Grafana Dashboards

### Dashboard 1: Canary Health Overview

**Purpose:** Real-time comparison of canary vs baseline health

```yaml
dashboard: canary-health-overview
refresh: 30s
time-range: last-1h

rows:
  - title: "Error Rate Comparison"
    panels:
      - type: graph
        title: "Error Rate (Canary vs Baseline)"
        targets:
          - expr: |
              sum(rate(http_requests_total{status=~"5..",version="canary"}[5m])) /
              sum(rate(http_requests_total{version="canary"}[5m]))
            legend: "Canary Error Rate"

          - expr: |
              sum(rate(http_requests_total{status=~"5..",version="baseline"}[5m])) /
              sum(rate(http_requests_total{version="baseline"}[5m]))
            legend: "Baseline Error Rate"

        alert:
          name: "Canary Error Rate High"
          condition: canary_error_rate > baseline_error_rate * 1.5
          for: 5m
          severity: critical

  - title: "Latency Comparison"
    panels:
      - type: graph
        title: "Latency Percentiles (P50, P95, P99)"
        targets:
          - expr: |
              histogram_quantile(0.50,
                sum(rate(http_request_duration_seconds_bucket{version="canary"}[5m])) by (le)
              )
            legend: "Canary P50"

          - expr: |
              histogram_quantile(0.95,
                sum(rate(http_request_duration_seconds_bucket{version="canary"}[5m])) by (le)
              )
            legend: "Canary P95"

          - expr: |
              histogram_quantile(0.99,
                sum(rate(http_request_duration_seconds_bucket{version="canary"}[5m])) by (le)
              )
            legend: "Canary P99"

          - expr: |
              histogram_quantile(0.99,
                sum(rate(http_request_duration_seconds_bucket{version="baseline"}[5m])) by (le)
              )
            legend: "Baseline P99"

  - title: "Traffic Distribution"
    panels:
      - type: pie-chart
        title: "Traffic Split (Canary vs Baseline)"
        targets:
          - expr: sum(rate(http_requests_total{version="canary"}[5m]))
            legend: "Canary"

          - expr: sum(rate(http_requests_total{version="baseline"}[5m]))
            legend: "Baseline"

  - title: "Success Rate"
    panels:
      - type: gauge
        title: "Canary Success Rate"
        targets:
          - expr: |
              sum(rate(http_requests_total{status!~"5..",version="canary"}[5m])) /
              sum(rate(http_requests_total{version="canary"}[5m]))

        thresholds:
          - value: 0.95
            color: red
          - value: 0.99
            color: yellow
          - value: 0.999
            color: green
```

### Dashboard 2: DWCP Manager Monitoring

```yaml
dashboard: dwcp-manager
refresh: 30s
time-range: last-6h

rows:
  - title: "Health & Availability"
    panels:
      - type: stat
        title: "Health Check Success Rate (24h)"
        targets:
          - expr: |
              sum(rate(dwcp_health_check_total{status="success"}[24h])) /
              sum(rate(dwcp_health_check_total[24h]))
        format: percent
        thresholds:
          - value: 0.999
            color: green

      - type: graph
        title: "Health Check Duration"
        targets:
          - expr: |
              histogram_quantile(0.99,
                sum(rate(dwcp_health_check_duration_seconds_bucket[5m])) by (le)
              )
            legend: "P99 Latency"

  - title: "Recovery Operations"
    panels:
      - type: graph
        title: "Recovery Attempts (Success vs Failure)"
        targets:
          - expr: sum(rate(dwcp_recovery_attempts_total{success="true"}[5m]))
            legend: "Successful Recoveries"

          - expr: sum(rate(dwcp_recovery_attempts_total{success="false"}[5m]))
            legend: "Failed Recoveries"

      - type: heatmap
        title: "Recovery Duration Heatmap"
        targets:
          - expr: sum(rate(dwcp_recovery_duration_seconds_bucket[5m])) by (le)

  - title: "Circuit Breakers"
    panels:
      - type: graph
        title: "Circuit Breaker States"
        targets:
          - expr: dwcp_circuit_breaker_state{circuit="database"}
            legend: "Database Circuit"

          - expr: dwcp_circuit_breaker_state{circuit="api"}
            legend: "API Circuit"

      - type: stat
        title: "Circuit Breaker Trips (1h)"
        targets:
          - expr: sum(increase(dwcp_circuit_breaker_trips_total[1h]))
        thresholds:
          - value: 0
            color: green
          - value: 5
            color: yellow
          - value: 10
            color: red

  - title: "Resource Usage"
    panels:
      - type: graph
        title: "Memory Usage"
        targets:
          - expr: dwcp_memory_usage_bytes{type="heap"} / 1024 / 1024
            legend: "Heap (MB)"

          - expr: dwcp_memory_usage_bytes{type="stack"} / 1024 / 1024
            legend: "Stack (MB)"

      - type: graph
        title: "Goroutines"
        targets:
          - expr: dwcp_goroutines
            legend: "Goroutines"
```

### Dashboard 3: Consensus Performance

```yaml
dashboard: consensus-performance
refresh: 30s
time-range: last-6h

rows:
  - title: "ProBFT Consensus"
    panels:
      - type: graph
        title: "Consensus Round Duration"
        targets:
          - expr: |
              histogram_quantile(0.95,
                sum(rate(probft_consensus_duration_seconds_bucket[5m])) by (le)
              )
            legend: "P95 Consensus Time"

      - type: stat
        title: "Byzantine Events Detected (24h)"
        targets:
          - expr: sum(increase(probft_byzantine_events_detected[24h]))
        thresholds:
          - value: 0
            color: green
          - value: 1
            color: red

  - title: "Bullshark Consensus"
    panels:
      - type: graph
        title: "Transaction Throughput"
        targets:
          - expr: sum(rate(bullshark_transactions_processed_total[1m]))
            legend: "Transactions/sec"

        alert:
          name: "Bullshark Throughput Low"
          condition: throughput < 250000
          for: 5m
          severity: warning

      - type: graph
        title: "DAG Depth & Width"
        targets:
          - expr: bullshark_dag_depth
            legend: "DAG Depth"

          - expr: bullshark_dag_width
            legend: "DAG Width"

  - title: "T-PBFT Trust"
    panels:
      - type: heatmap
        title: "Node Trust Scores"
        targets:
          - expr: tpbft_node_trust_score

      - type: stat
        title: "Message Reduction (%)"
        targets:
          - expr: avg(tpbft_message_reduction_percentage)
        format: percent
```

### Dashboard 4: Resource Allocation (MADDPG)

```yaml
dashboard: resource-allocation
refresh: 30s
time-range: last-6h

rows:
  - title: "Allocation Efficiency"
    panels:
      - type: graph
        title: "Resource Utilization by Service"
        targets:
          - expr: maddpg_resource_utilization{service=~".*",resource_type="cpu"}
            legend: "{{service}} CPU"

          - expr: maddpg_resource_utilization{service=~".*",resource_type="memory"}
            legend: "{{service}} Memory"

      - type: stat
        title: "Overall Allocation Efficiency"
        targets:
          - expr: avg(maddpg_allocation_efficiency)
        format: percent
        thresholds:
          - value: 0.8
            color: red
          - value: 0.9
            color: yellow
          - value: 0.95
            color: green

  - title: "RL Model Performance"
    panels:
      - type: graph
        title: "Reward Score Trend"
        targets:
          - expr: avg(maddpg_reward_score)
            legend: "Average Reward"

      - type: graph
        title: "Allocation Quality vs Baseline"
        targets:
          - expr: maddpg_allocation_quality
            legend: "MADDPG"

          - expr: baseline_allocation_quality
            legend: "Baseline"

  - title: "Safety Metrics"
    panels:
      - type: stat
        title: "Constraint Violations (1h)"
        targets:
          - expr: sum(increase(maddpg_constraint_violations[1h]))
        thresholds:
          - value: 0
            color: green
          - value: 1
            color: red

      - type: stat
        title: "Resource Starvation Events (1h)"
        targets:
          - expr: sum(increase(maddpg_resource_starvation_events[1h]))
        thresholds:
          - value: 0
            color: green
          - value: 1
            color: red
```

---

## AlertManager Configuration

### Alert Rules

```yaml
# prometheus-alerts.yml
groups:
  - name: canary-health
    interval: 30s
    rules:
      - alert: CanaryErrorRateHigh
        expr: |
          sum(rate(http_requests_total{status=~"5..",version="canary"}[5m])) /
          sum(rate(http_requests_total{version="canary"}[5m])) >
          sum(rate(http_requests_total{status=~"5..",version="baseline"}[5m])) /
          sum(rate(http_requests_total{version="baseline"}[5m])) * 1.5
        for: 5m
        labels:
          severity: critical
          component: canary
        annotations:
          summary: "Canary error rate 50% higher than baseline"
          description: "Canary error rate {{ $value }}% is significantly higher than baseline"
          runbook: "https://docs/runbooks/canary-rollback"

      - alert: CanaryLatencyHigh
        expr: |
          histogram_quantile(0.99,
            sum(rate(http_request_duration_seconds_bucket{version="canary"}[5m])) by (le)
          ) >
          histogram_quantile(0.99,
            sum(rate(http_request_duration_seconds_bucket{version="baseline"}[5m])) by (le)
          ) * 2
        for: 5m
        labels:
          severity: critical
          component: canary
        annotations:
          summary: "Canary P99 latency 2x higher than baseline"
          description: "Canary P99 latency {{ $value }}s is 2x baseline"

      - alert: CanarySuccessRateLow
        expr: |
          sum(rate(http_requests_total{status!~"5..",version="canary"}[5m])) /
          sum(rate(http_requests_total{version="canary"}[5m])) < 0.95
        for: 3m
        labels:
          severity: critical
          component: canary
        annotations:
          summary: "Canary success rate below 95%"
          description: "Canary success rate {{ $value }}% is critically low"

  - name: dwcp-manager
    interval: 30s
    rules:
      - alert: DWCPHealthCheckFailure
        expr: sum(rate(dwcp_health_check_total{status="failure"}[5m])) > 0
        for: 2m
        labels:
          severity: warning
          component: dwcp-manager
        annotations:
          summary: "DWCP Manager health check failures detected"

      - alert: DWCPCircuitBreakerStorm
        expr: sum(rate(dwcp_circuit_breaker_trips_total[5m])) > 10
        for: 5m
        labels:
          severity: critical
          component: dwcp-manager
        annotations:
          summary: "Circuit breaker trip storm detected"
          description: "{{ $value }} circuit breaker trips in 5 minutes"

      - alert: DWCPMemoryLeak
        expr: |
          rate(dwcp_memory_usage_bytes{type="heap"}[1h]) > 0 and
          dwcp_memory_usage_bytes{type="heap"} > 8 * 1024 * 1024 * 1024  # 8GB
        for: 4h
        labels:
          severity: warning
          component: dwcp-manager
        annotations:
          summary: "Potential memory leak detected"
          description: "Memory usage increasing steadily: {{ $value }} bytes"

      - alert: DWCPRecoveryLoop
        expr: sum(rate(dwcp_recovery_attempts_total[1m])) > 50
        for: 2m
        labels:
          severity: critical
          component: dwcp-manager
        annotations:
          summary: "Recovery loop detected"
          description: "{{ $value }} recovery attempts per minute"

  - name: compression
    interval: 30s
    rules:
      - alert: CompressionAccuracyLow
        expr: avg(compression_accuracy) < 0.99
        for: 1h
        labels:
          severity: warning
          component: compression
        annotations:
          summary: "Compression accuracy below target"
          description: "Compression accuracy {{ $value }} < 99%"

      - alert: CompressionOverheadHigh
        expr: avg(compression_overhead_percentage) > 0.05
        for: 30m
        labels:
          severity: warning
          component: compression
        annotations:
          summary: "Compression overhead exceeds 5%"
          description: "Overhead {{ $value }}% is too high"

  - name: consensus
    interval: 30s
    rules:
      - alert: ByzantineNodeDetected
        expr: increase(probft_byzantine_events_detected[5m]) > 0
        for: 1m
        labels:
          severity: critical
          component: probft
        annotations:
          summary: "Byzantine node detected!"
          description: "Byzantine event: {{ $labels.type }}"

      - alert: ConsensusFailure
        expr: increase(probft_consensus_failures_total[5m]) > 0
        for: 1m
        labels:
          severity: critical
          component: probft
        annotations:
          summary: "ProBFT consensus failure"
          description: "Consensus failed: {{ $labels.reason }}"

      - alert: BullsharkThroughputLow
        expr: sum(rate(bullshark_transactions_processed_total[1m])) < 250000
        for: 10m
        labels:
          severity: warning
          component: bullshark
        annotations:
          summary: "Bullshark throughput below target"
          description: "Throughput {{ $value }} tx/s < 250K tx/s"

      - alert: BullsharkMempoolFull
        expr: avg(bullshark_mempool_usage_percentage) > 0.95
        for: 5m
        labels:
          severity: critical
          component: bullshark
        annotations:
          summary: "Bullshark mempool critically full"
          description: "Mempool usage {{ $value }}%"

  - name: resource-allocation
    interval: 30s
    rules:
      - alert: ResourceStarvation
        expr: increase(maddpg_resource_starvation_events[5m]) > 0
        for: 1m
        labels:
          severity: critical
          component: maddpg
        annotations:
          summary: "Resource starvation detected"
          description: "Service {{ $labels.service }} starved of resources"

      - alert: AllocationEfficiencyLow
        expr: avg(maddpg_allocation_efficiency) < 0.85
        for: 1h
        labels:
          severity: warning
          component: maddpg
        annotations:
          summary: "Allocation efficiency below baseline"
          description: "Efficiency {{ $value }} < 85%"
```

### AlertManager Routing

```yaml
# alertmanager.yml
global:
  resolve_timeout: 5m
  pagerduty_url: 'https://events.pagerduty.com/v2/enqueue'
  opsgenie_api_url: 'https://api.opsgenie.com/'

route:
  receiver: 'default'
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 12h

  routes:
    # Critical alerts go to PagerDuty immediately
    - match:
        severity: critical
      receiver: pagerduty-critical
      continue: true

    # Warning alerts go to Slack
    - match:
        severity: warning
      receiver: slack-warnings

    # Canary-specific alerts
    - match:
        component: canary
      receiver: canary-team
      group_wait: 10s
      repeat_interval: 5m

receivers:
  - name: 'default'
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/XXX'
        channel: '#production-alerts'
        title: '{{ .CommonAnnotations.summary }}'
        text: '{{ .CommonAnnotations.description }}'

  - name: 'pagerduty-critical'
    pagerduty_configs:
      - service_key: 'PAGERDUTY_SERVICE_KEY'
        description: '{{ .CommonAnnotations.summary }}'

  - name: 'slack-warnings'
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/XXX'
        channel: '#production-warnings'
        title: '⚠️ {{ .CommonAnnotations.summary }}'
        text: '{{ .CommonAnnotations.description }}'

  - name: 'canary-team'
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/XXX'
        channel: '#canary-rollout'
        title: '🚨 CANARY ALERT: {{ .CommonAnnotations.summary }}'
        text: |
          {{ .CommonAnnotations.description }}

          Runbook: {{ .CommonAnnotations.runbook }}
```

---

## Distributed Tracing (Jaeger)

### Configuration

```yaml
# jaeger-config.yml
jaeger:
  agent:
    enabled: true
    port: 6831

  collector:
    enabled: true
    grpc-port: 14250
    http-port: 14268

  query:
    enabled: true
    port: 16686

  storage:
    type: elasticsearch
    elasticsearch:
      server-urls: http://elasticsearch:9200
      index-prefix: jaeger

  sampling:
    strategies:
      - service: dwcp-manager
        type: probabilistic
        param: 0.1  # 10% sampling

      - service: probft-consensus
        type: probabilistic
        param: 1.0  # 100% sampling (critical)
```

### Application Instrumentation

```go
import (
    "github.com/opentracing/opentracing-go"
    "github.com/uber/jaeger-client-go"
)

func initJaeger(serviceName string) io.Closer {
    cfg := &jaeger.Config{
        ServiceName: serviceName,
        Sampler: &jaeger.SamplerConfig{
            Type:  "probabilistic",
            Param: 0.1,
        },
        Reporter: &jaeger.ReporterConfig{
            LogSpans:           true,
            LocalAgentHostPort: "jaeger-agent:6831",
        },
    }

    tracer, closer, err := cfg.NewTracer()
    if err != nil {
        log.Fatal(err)
    }

    opentracing.SetGlobalTracer(tracer)
    return closer
}

// Trace request handling
func handleRequest(w http.ResponseWriter, r *http.Request) {
    span := opentracing.StartSpan("handleRequest")
    defer span.Finish()

    ctx := opentracing.ContextWithSpan(r.Context(), span)

    // Call downstream services with context
    result, err := downstreamService(ctx)
    if err != nil {
        span.SetTag("error", true)
        span.LogKV("error", err.Error())
    }

    w.Write(result)
}
```

---

## Log Aggregation (ELK Stack)

### Fluentd Configuration

```yaml
# fluentd.conf
<source>
  @type tail
  path /var/log/containers/dwcp-*.log
  pos_file /var/log/fluentd-dwcp.pos
  tag kubernetes.dwcp
  format json
  time_key time
  time_format %Y-%m-%dT%H:%M:%S.%NZ
</source>

<filter kubernetes.dwcp>
  @type parser
  key_name log
  <parse>
    @type json
  </parse>
</filter>

<filter kubernetes.**>
  @type kubernetes_metadata
  @id filter_kube_metadata
</filter>

<match kubernetes.dwcp>
  @type elasticsearch
  host elasticsearch
  port 9200
  logstash_format true
  logstash_prefix dwcp
  include_tag_key true
  type_name _doc
  <buffer>
    @type file
    path /var/log/fluentd-buffers/kubernetes.dwcp
    flush_mode interval
    flush_interval 5s
  </buffer>
</match>
```

### Elasticsearch Index Templates

```json
{
  "index_patterns": ["dwcp-*"],
  "settings": {
    "number_of_shards": 3,
    "number_of_replicas": 1,
    "index.lifecycle.name": "dwcp-policy",
    "index.lifecycle.rollover_alias": "dwcp"
  },
  "mappings": {
    "properties": {
      "@timestamp": { "type": "date" },
      "level": { "type": "keyword" },
      "message": { "type": "text" },
      "component": { "type": "keyword" },
      "version": { "type": "keyword" },
      "error": { "type": "text" },
      "trace_id": { "type": "keyword" },
      "span_id": { "type": "keyword" }
    }
  }
}
```

---

## SLO/SLI Definitions

### Service Level Indicators (SLIs)

```yaml
slis:
  availability:
    definition: "Percentage of successful requests"
    measurement: |
      sum(rate(http_requests_total{status!~"5.."}[7d])) /
      sum(rate(http_requests_total[7d]))
    target: 99.9%

  latency:
    definition: "95th percentile request latency"
    measurement: |
      histogram_quantile(0.95,
        sum(rate(http_request_duration_seconds_bucket[7d])) by (le)
      )
    target: < 500ms

  error-rate:
    definition: "Percentage of failed requests"
    measurement: |
      sum(rate(http_requests_total{status=~"5.."}[7d])) /
      sum(rate(http_requests_total[7d]))
    target: < 0.1%

  throughput:
    definition: "Requests processed per second"
    measurement: sum(rate(http_requests_total[7d]))
    target: > 10000 req/s
```

### Service Level Objectives (SLOs)

```yaml
slos:
  availability-slo:
    objective: 99.9%
    time-window: 30d
    error-budget: 0.1% (43.2 minutes/month)

  latency-slo:
    objective: 95% of requests < 500ms
    time-window: 30d

  consensus-slo:
    objective: 99.5% consensus success
    time-window: 30d

  throughput-slo:
    objective: ≥ 300K tx/s (Bullshark)
    time-window: 24h
```

---

## On-Call Runbooks

### Runbook: Canary Rollback

**Scenario:** Canary health metrics degraded

**Detection:** AlertManager fires `CanaryErrorRateHigh` or `CanaryLatencyHigh`

**Steps:**
1. **Verify Alert:**
   ```bash
   # Check Grafana canary dashboard
   # Confirm error rate or latency spike

   # Query Prometheus
   curl -G 'http://prometheus:9090/api/v1/query' \
     --data-urlencode 'query=rate(http_requests_total{status=~"5..",version="canary"}[5m])'
   ```

2. **Execute Rollback:**
   ```bash
   # Automated rollback (preferred)
   ./scripts/canary-rollback.sh --reason "high-error-rate"

   # Manual rollback
   kubectl patch virtualservice dwcp-service \
     -p '{"spec":{"http":[{"route":[{"destination":{"host":"dwcp-baseline"},"weight":100}]}]}}'
   ```

3. **Verify Rollback:**
   ```bash
   # Check traffic routing
   kubectl get virtualservice dwcp-service -o yaml

   # Verify error rate normalized
   # Check Grafana dashboard
   ```

4. **Preserve Evidence:**
   ```bash
   # Collect canary logs
   kubectl logs -l app=dwcp-canary --tail=10000 > canary-rollback-logs-$(date +%Y%m%d-%H%M%S).txt

   # Export metrics
   curl -G 'http://prometheus:9090/api/v1/query_range' \
     --data-urlencode 'query=...' > metrics-export.json
   ```

5. **Post-Incident:**
   - Schedule post-mortem within 24 hours
   - Document root cause
   - Create action items for fixes

---

## Monitoring Cost Optimization

### Metric Retention Policies

```yaml
retention:
  prometheus:
    short-term: 15d
    resolution: 15s

  thanos:
    long-term: 365d
    downsampling:
      - resolution: 5m
        retention: 90d
      - resolution: 1h
        retention: 365d

  logs:
    elasticsearch:
      hot: 7d
      warm: 30d
      cold: 90d
      delete: 180d
```

### Sampling Strategies

```yaml
sampling:
  metrics:
    high-cardinality: 10%  # e.g., per-user metrics
    low-cardinality: 100%  # e.g., system metrics

  traces:
    critical-paths: 100%
    standard-paths: 10%
    background-jobs: 1%

  logs:
    error: 100%
    warning: 100%
    info: 10%
    debug: 0%  # disabled in production
```

---

## Conclusion

This comprehensive monitoring and alerting configuration provides:

1. ✅ **Real-time visibility** into all 6 DWCP systems
2. ✅ **Automated alerting** with PagerDuty/Slack integration
3. ✅ **Canary health tracking** with automated rollback triggers
4. ✅ **Distributed tracing** for end-to-end request tracking
5. ✅ **Log aggregation** for troubleshooting
6. ✅ **SLO/SLI tracking** for reliability goals
7. ✅ **On-call runbooks** for rapid incident response
8. ✅ **Cost optimization** through retention and sampling

**Expected Outcome:** Early detection of issues (<2 minutes), automated rollback (<2 minutes), and comprehensive observability for all production systems.
