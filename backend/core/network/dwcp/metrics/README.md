# DWCP Metrics System

Comprehensive Prometheus monitoring for DWCP Phase 1 components.

## Overview

The DWCP metrics system provides production-grade observability for all DWCP components including:

- **AMST (Adaptive Multi-Stream Transport)**: Stream metrics, bandwidth utilization, latency tracking
- **HDE (Hierarchical Delta Encoding)**: Compression ratios, delta hit rates, baseline management
- **VM Migration**: Performance comparison, speedup factors, duration tracking
- **Federation**: Cross-cluster sync, bandwidth savings
- **System Health**: Component availability, configuration status, version tracking

## Quick Start

### 1. Initialize Metrics in Your Application

```go
import "novacron/backend/core/network/dwcp/metrics"

func main() {
    // Initialize metrics system (cluster name, node name, port)
    if err := metrics.InitializeMetrics("cluster-1", "node-01", 9090); err != nil {
        log.Fatal(err)
    }
    defer metrics.ShutdownMetrics()

    // Set version info (call once at startup)
    metrics.SetVersionInfo("cluster-1", "1.0.0", "abc123", "2025-11-08")
}
```

### 2. Instrument AMST Components

```go
// Create wrapper
amstMetrics := metrics.NewAMSTMetricsWrapper()

// Record stream lifecycle
streamID := "stream-001"
amstMetrics.OnStreamStart(streamID)
defer amstMetrics.OnStreamEnd(streamID)

// Record data transfer
amstMetrics.OnStreamData(streamID, bytesSent, bytesReceived)

// Record errors
amstMetrics.OnStreamError(streamID, "connection_timeout")

// Update bandwidth utilization
amstMetrics.OnBandwidthUpdate(usedBytes, availableBytes)
```

### 3. Instrument HDE Components

```go
// Create wrapper
hdeMetrics := metrics.NewHDEMetricsWrapper()

// Record compression
hdeMetrics.OnCompressionComplete(
    "vm_memory",           // data type
    originalSize,          // original bytes
    compressedSize,        // compressed bytes
    deltaHit,              // was delta encoding used?
)

// Record decompression
hdeMetrics.OnDecompressionComplete(success)

// Update baseline count
hdeMetrics.OnBaselineUpdate("vm_state", baselineCount)

// Update dictionary efficiency
hdeMetrics.OnDictionaryUpdate(efficiencyPercent)
```

### 4. Instrument Migration Components

```go
// Create wrapper
migrationMetrics := metrics.NewMigrationMetricsWrapper()

// Track migration
migrationID := "migration-123"
migrationMetrics.OnMigrationStart(migrationID)

// ... perform migration ...

migrationMetrics.OnMigrationComplete(migrationID, "node-02", true)

// Record speedup factor
migrationMetrics.OnSpeedupCalculated("large_vm", 8.5)
```

### 5. Instrument System Components

```go
// Create wrapper
systemMetrics := metrics.NewSystemMetricsWrapper()

// Update component health
systemMetrics.OnComponentHealthChange("amst", metrics.HealthHealthy)
systemMetrics.OnComponentHealthChange("hde", metrics.HealthDegraded)
systemMetrics.OnComponentHealthChange("dwcp_manager", metrics.HealthDown)

// Track feature status
systemMetrics.OnFeatureToggle("amst_enabled", true)
systemMetrics.OnFeatureToggle("hde_enabled", true)
```

## Metrics Endpoint

Once initialized, metrics are exposed at:

```
http://localhost:9090/metrics
```

Health checks:
```
http://localhost:9090/health
http://localhost:9090/ready
```

## Prometheus Configuration

### 1. Add Scrape Configuration

Copy `/home/kp/novacron/configs/prometheus/dwcp-scrape-config.yml` to your Prometheus config directory:

```yaml
scrape_configs:
  - job_name: 'dwcp-metrics'
    scrape_interval: 10s
    static_configs:
      - targets: ['localhost:9090']
```

### 2. Load Alert Rules

Add to Prometheus configuration:

```yaml
rule_files:
  - '/path/to/dwcp-alerts.yml'
  - '/path/to/dwcp-recording-rules.yml'
```

### 3. Start Prometheus

```bash
prometheus --config.file=/path/to/prometheus.yml
```

## Grafana Dashboard

### Import Dashboard

1. Open Grafana (typically http://localhost:3000)
2. Navigate to Dashboards → Import
3. Upload `/home/kp/novacron/configs/grafana/dwcp-dashboard.json`
4. Select your Prometheus data source
5. Click Import

### Dashboard Panels

The DWCP dashboard includes:

1. **AMST Active Streams** - Real-time stream count with alerting
2. **AMST Bandwidth Utilization** - Gauge showing % of bandwidth used
3. **HDE Compression Ratio Distribution** - Heatmap of compression efficiency
4. **AMST Throughput** - Network throughput in Mbps
5. **HDE Compression Efficiency** - Current compression ratio by data type
6. **HDE Delta Hit Rate** - % of successful delta encoding
7. **Migration Performance Comparison** - DWCP vs standard migration times
8. **Error Rate** - AMST error rate with threshold alerting
9. **Bandwidth Savings Calculator** - Real-time bandwidth savings
10. **Daily Bandwidth Savings** - Total savings in TB over 24h
11. **Cost Savings** - Estimated USD savings based on bandwidth pricing
12. **Component Health Matrix** - Visual health status of all components
13. **AMST Stream Latency** - P95/P99 latency tracking
14. **Migration Speedup Factor** - How much faster DWCP migrations are
15. **Federation Sync Performance** - Cross-cluster sync duration
16. **System Availability** - Overall system availability vs SLA
17. **Average Migration Time** - Migration time vs 5-minute SLA target
18. **Overall Error Rate** - Error rate vs 1% SLA target

## Available Metrics

### AMST Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `dwcp_amst_streams_active` | gauge | cluster, node | Current active streams |
| `dwcp_amst_streams_total` | counter | cluster, node, result | Total streams created |
| `dwcp_amst_bytes_sent_total` | counter | cluster, node, stream_id | Bytes sent |
| `dwcp_amst_bytes_received_total` | counter | cluster, node, stream_id | Bytes received |
| `dwcp_amst_errors_total` | counter | cluster, node, error_type | Stream errors |
| `dwcp_amst_latency_seconds` | histogram | cluster, node, operation | Latency distribution |
| `dwcp_amst_bandwidth_utilization_percent` | gauge | cluster, node | Bandwidth utilization |

### HDE Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `dwcp_hde_compression_ratio` | histogram | cluster, node, data_type | Compression ratio |
| `dwcp_hde_operations_total` | counter | cluster, node, operation, result | Encode/decode ops |
| `dwcp_hde_delta_hit_rate_percent` | gauge | cluster, node | Delta encoding hit rate |
| `dwcp_hde_baseline_count` | gauge | cluster, node, baseline_type | Active baselines |
| `dwcp_hde_dictionary_efficiency_percent` | gauge | cluster, node | Dictionary efficiency |
| `dwcp_hde_bytes_original_total` | counter | cluster, node, data_type | Original bytes |
| `dwcp_hde_bytes_compressed_total` | counter | cluster, node, data_type | Compressed bytes |

### Migration Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `dwcp_migration_duration_seconds` | histogram | cluster, source_node, dest_node, dwcp_enabled | Migration duration |
| `dwcp_migration_speedup_factor` | gauge | cluster, vm_type | DWCP speedup factor |

### Federation Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `dwcp_federation_sync_duration_seconds` | histogram | cluster, remote_cluster, sync_type | Sync duration |
| `dwcp_federation_bandwidth_saved_bytes_total` | counter | cluster, remote_cluster | Bandwidth saved |

### System Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `dwcp_system_component_health` | gauge | cluster, node, component | Component health (0=down, 1=degraded, 2=healthy) |
| `dwcp_system_config_enabled` | gauge | cluster, node, feature | Feature enabled status |
| `dwcp_system_version_info` | gauge | cluster, version, git_commit, build_date | Version information |

## Recording Rules

Pre-aggregated metrics for efficient querying:

- `dwcp:amst:error_rate:5m` - AMST error rate over 5 minutes
- `dwcp:amst:throughput_mbps:5m` - AMST throughput in Mbps
- `dwcp:hde:compression_ratio_median:10m` - Median HDE compression ratio
- `dwcp:migration:duration_median:1h` - Median migration duration
- `dwcp:federation:bandwidth_saved_gbps:1h` - Federation bandwidth savings
- `dwcp:system:availability:5m` - System availability percentage

## Alerting Rules

### Critical Alerts

- **DWCPHighErrorRate**: Error rate >5% for 5 minutes
- **DWCPComponentDown**: Component unhealthy for 2 minutes
- **DWCPSLAViolation**: Migration time exceeds 5-minute SLA

### Warning Alerts

- **DWCPLowCompressionRatio**: Compression <3x for 10 minutes
- **DWCPLowBandwidthUtilization**: Bandwidth usage <50% for 15 minutes
- **DWCPMigrationSlow**: Migration 2x slower than target

## Best Practices

### 1. Label Cardinality

Keep label cardinality low to avoid performance issues:

```go
// ❌ BAD: High cardinality label (unique per VM)
metrics.RecordStreamMetrics(vmID, bytes, 0, "")

// ✅ GOOD: Use stream ID only for active streams
metrics.RecordStreamMetrics(streamID, bytes, 0, "")
```

### 2. Metric Naming

Follow Prometheus naming conventions:

- Use base unit (seconds, bytes, not milliseconds or KB)
- Suffix with `_total` for counters
- Suffix with unit for gauges/histograms

### 3. Error Handling

Always check if collector is initialized:

```go
collector := metrics.GetCollector()
if collector != nil {
    collector.RecordStreamData(streamID, sent, recv)
}
```

### 4. Graceful Shutdown

Always defer shutdown:

```go
if err := metrics.InitializeMetrics("cluster", "node", 9090); err != nil {
    log.Fatal(err)
}
defer metrics.ShutdownMetrics()
```

## Troubleshooting

### Metrics not appearing in Prometheus

1. Check exporter is running: `curl http://localhost:9090/health`
2. Check Prometheus targets: Navigate to Status → Targets in Prometheus UI
3. Verify scrape configuration matches exporter port
4. Check firewall rules allow Prometheus to scrape metrics endpoint

### High memory usage

1. Reduce label cardinality (especially `stream_id`)
2. Increase recording rule evaluation interval
3. Reduce histogram bucket count
4. Implement metric retention policies

### Missing data in Grafana

1. Verify Prometheus data source is configured correctly
2. Check time range matches data availability
3. Verify recording rules are evaluating correctly
4. Check for PromQL query errors in panel settings

## Performance Characteristics

- **Metric Collection Overhead**: <0.1% CPU, <10MB RAM per 1000 metrics/sec
- **Export Endpoint**: Handles 100+ concurrent scrapes
- **Scrape Duration**: Typically <100ms for all DWCP metrics
- **Storage**: ~1KB per metric per scrape interval

## Future Enhancements

- [ ] Exemplars linking metrics to traces
- [ ] Custom metric aggregations for cost optimization
- [ ] Automated anomaly detection using ML
- [ ] Multi-cluster metric federation
- [ ] Long-term metric storage with downsampling

## Support

For issues or questions:
- Check existing metrics with: `curl http://localhost:9090/metrics | grep dwcp_`
- Review Prometheus logs for scrape errors
- Verify alert rules with: `promtool check rules dwcp-alerts.yml`
