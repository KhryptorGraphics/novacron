# DWCP Systems - Monitoring Setup Guide

**Version:** 1.0
**Date:** 2025-11-14
**Stack:** Prometheus + Grafana + Alertmanager

## Prometheus Configuration

### Scrape Configurations

```yaml
# /etc/prometheus/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  # DWCP Manager
  - job_name: 'dwcp-manager'
    static_configs:
      - targets:
        - 'dwcp-manager-01:8080'
        - 'dwcp-manager-02:8080'
        - 'dwcp-manager-03:8080'

  # Compression Selector API
  - job_name: 'compression-api'
    static_configs:
      - targets:
        - 'compression-api-01:5000'
        - 'compression-api-02:5000'

  # ProBFT Consensus
  - job_name: 'probft-consensus'
    static_configs:
      - targets: ['probft-node-01:8080', 'probft-node-02:8080', ...] # All 7 nodes

  # Bullshark Consensus
  - job_name: 'bullshark-consensus'
    static_configs:
      - targets: ['bullshark-node-01:8080', ...] # All 100 nodes

  # T-PBFT Consensus
  - job_name: 'tpbft-consensus'
    static_configs:
      - targets: ['tpbft-node-01:8080', ...] # All 10 nodes

  # MADDPG Allocator
  - job_name: 'maddpg-allocator'
    static_configs:
      - targets:
        - 'maddpg-allocator-01:8080'
        - 'maddpg-allocator-02:8080'
```

## Alert Rules

```yaml
# /etc/prometheus/alert_rules.yml
groups:
  - name: dwcp_critical_alerts
    rules:
      # Service Down Alerts
      - alert: DWCPManagerDown
        expr: up{job="dwcp-manager"} == 0
        for: 30s
        labels:
          severity: critical
        annotations:
          summary: "DWCP Manager down on {{ $labels.instance }}"

      - alert: ConsensusDown
        expr: up{job=~".*-consensus"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Consensus node down: {{ $labels.instance }}"

      # Performance Alerts
      - alert: HighLatency
        expr: histogram_quantile(0.99, dwcp_request_duration_seconds_bucket) > 0.2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High P99 latency: {{ $value }}s"

      # Consensus Alerts
      - alert: QuorumNotAchieved
        expr: probft_quorum_achieved_total == 0
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "ProBFT quorum not achieved for 5 minutes"

      # ML Model Alerts
      - alert: ModelAccuracyDrop
        expr: compression_selector_accuracy < 0.95
        for: 1h
        labels:
          severity: warning
        annotations:
          summary: "Compression selector accuracy dropped to {{ $value }}"
```

## Grafana Dashboards

### Dashboard 1: DWCP Overview
```json
{
  "title": "DWCP Systems Overview",
  "panels": [
    {
      "title": "Service Health",
      "targets": [{"expr": "up{job=~'dwcp.*|.*-consensus|.*-api|.*-allocator'}"}]
    },
    {
      "title": "Request Rate",
      "targets": [{"expr": "rate(dwcp_requests_total[5m])"}]
    },
    {
      "title": "P99 Latency",
      "targets": [{"expr": "histogram_quantile(0.99, dwcp_request_duration_seconds_bucket)"}]
    },
    {
      "title": "Error Rate",
      "targets": [{"expr": "rate(dwcp_errors_total[5m])"}]
    }
  ]
}
```

### Dashboard 2: Consensus Health
```json
{
  "title": "Consensus Health",
  "panels": [
    {
      "title": "Block Finalization Time",
      "targets": [
        {"expr": "probft_finalization_duration_seconds"},
        {"expr": "bullshark_round_duration_seconds"},
        {"expr": "tpbft_consensus_latency_seconds"}
      ]
    },
    {
      "title": "Throughput",
      "targets": [
        {"expr": "rate(probft_blocks_finalized_total[1m])"},
        {"expr": "bullshark_throughput_tx_per_second"},
        {"expr": "rate(tpbft_transactions_committed_total[1m])"}
      ]
    },
    {
      "title": "Quorum Achievement",
      "targets": [{"expr": "probft_quorum_achievement_rate"}]
    }
  ]
}
```

### Dashboard 3: ML Systems
```json
{
  "title": "ML Systems Performance",
  "panels": [
    {
      "title": "Compression Selector Accuracy",
      "targets": [{"expr": "compression_selector_accuracy"}]
    },
    {
      "title": "Prediction Latency",
      "targets": [{"expr": "histogram_quantile(0.99, compression_selector_prediction_duration_bucket)"}]
    },
    {
      "title": "MADDPG Optimization",
      "targets": [{"expr": "maddpg_resource_optimization_percentage"}]
    },
    {
      "title": "Allocation Success Rate",
      "targets": [{"expr": "rate(maddpg_successful_allocations_total[5m]) / rate(maddpg_total_allocations_total[5m])"}]
    }
  ]
}
```

## Alertmanager Configuration

```yaml
# /etc/alertmanager/alertmanager.yml
global:
  resolve_timeout: 5m

route:
  receiver: 'ops-team'
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 12h
  routes:
    - match:
        severity: critical
      receiver: 'pagerduty-critical'
    - match:
        severity: warning
      receiver: 'ops-team'

receivers:
  - name: 'ops-team'
    email_configs:
      - to: 'ops-team@example.com'
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/...'
        channel: '#dwcp-alerts'

  - name: 'pagerduty-critical'
    pagerduty_configs:
      - service_key: '<pagerduty-service-key>'
```

## Performance Baselines

### DWCP Manager
- Health check latency: <5ms
- Circuit breaker closed: 99.9% of time
- CPU usage: 20-40%
- Memory usage: 30-50%

### Compression Selector
- Prediction latency: <10ms (P99)
- Accuracy: >99.5%
- Throughput: >1000 req/s
- CPU usage: 40-60%

### Consensus Systems
- ProBFT finalization: <1s
- Bullshark throughput: >300K tx/s
- T-PBFT latency: <60ms
- Quorum achievement: >99%

### MADDPG Allocator
- Optimization: >25%
- Allocation latency: <50ms
- SLA compliance: >95%
- Model inference: <100ms

---
**Document Version:** 1.0
**Last Updated:** 2025-11-14
**Owner:** SRE Team
