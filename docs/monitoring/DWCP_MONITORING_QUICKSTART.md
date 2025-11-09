# DWCP Monitoring Quick Start Guide

Complete production-grade observability for DWCP Phase 1 in under 10 minutes.

## Prerequisites

- Docker and Docker Compose installed
- Go 1.21+ (for building NovaCron with metrics)
- Prometheus and Grafana (or use provided Docker Compose)

## Quick Setup

### Option 1: Docker Compose (Recommended)

**Start the full monitoring stack:**

```bash
cd /home/kp/novacron/configs
docker-compose -f docker-compose.monitoring.yml up -d
```

This starts:
- **Prometheus** on port 9091 (http://localhost:9091)
- **Grafana** on port 3001 (http://localhost:3001)
- **Alertmanager** on port 9093 (http://localhost:9093)
- **Node Exporter** on port 9100 (system metrics)
- **cAdvisor** on port 8080 (container metrics)

**Access the dashboards:**
- Grafana: http://localhost:3001 (admin / dwcp-admin-2025)
- Prometheus: http://localhost:9091
- Alertmanager: http://localhost:9093

### Option 2: Native Installation

**Install Prometheus:**

```bash
# macOS
brew install prometheus

# Linux (Ubuntu/Debian)
sudo apt-get install prometheus

# Start Prometheus
prometheus --config.file=/home/kp/novacron/configs/prometheus/dwcp-scrape-config.yml
```

**Install Grafana:**

```bash
# macOS
brew install grafana

# Linux (Ubuntu/Debian)
sudo apt-get install grafana

# Start Grafana
sudo systemctl start grafana-server
```

## Integrate DWCP Metrics

### 1. Add Metrics to Your Application

```go
package main

import (
    "context"
    "log"
    "os"
    "os/signal"
    "syscall"

    "novacron/backend/core/network/dwcp/metrics"
)

func main() {
    // Initialize metrics system
    cluster := os.Getenv("CLUSTER_NAME")
    if cluster == "" {
        cluster = "default-cluster"
    }

    node := os.Getenv("NODE_NAME")
    if node == "" {
        node = "node-01"
    }

    if err := metrics.InitializeMetrics(cluster, node, 9090); err != nil {
        log.Fatalf("Failed to initialize metrics: %v", err)
    }
    defer metrics.ShutdownMetrics()

    // Set version info
    metrics.SetVersionInfo(cluster, "1.0.0", "abc123", "2025-11-08")

    // Set initial component health
    systemMetrics := metrics.NewSystemMetricsWrapper()
    systemMetrics.OnComponentHealthChange("amst", metrics.HealthHealthy)
    systemMetrics.OnComponentHealthChange("hde", metrics.HealthHealthy)
    systemMetrics.OnComponentHealthChange("dwcp_manager", metrics.HealthHealthy)

    // Your application logic here
    runApplication()

    // Wait for shutdown signal
    sigChan := make(chan os.Signal, 1)
    signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
    <-sigChan

    log.Println("Shutting down gracefully...")
}

func runApplication() {
    // Your DWCP application code
}
```

### 2. Instrument AMST Streams

```go
func transferData(conn *Connection, data []byte) error {
    // Create metrics wrapper
    amstMetrics := metrics.NewAMSTMetricsWrapper()

    // Start stream
    streamID := generateStreamID()
    amstMetrics.OnStreamStart(streamID)
    defer amstMetrics.OnStreamEnd(streamID)

    // Transfer data
    bytesSent, err := conn.Write(data)
    if err != nil {
        amstMetrics.OnStreamError(streamID, "write_error")
        return err
    }

    // Record metrics
    amstMetrics.OnStreamData(streamID, int64(bytesSent), 0)

    // Update bandwidth utilization
    usedBandwidth := calculateUsedBandwidth()
    availableBandwidth := getAvailableBandwidth()
    amstMetrics.OnBandwidthUpdate(usedBandwidth, availableBandwidth)

    return nil
}
```

### 3. Instrument HDE Compression

```go
func compressData(data []byte) ([]byte, error) {
    hdeMetrics := metrics.NewHDEMetricsWrapper()

    originalSize := int64(len(data))

    // Perform compression
    compressed, deltaHit, err := performCompression(data)
    if err != nil {
        return nil, err
    }

    compressedSize := int64(len(compressed))

    // Record metrics
    hdeMetrics.OnCompressionComplete("vm_memory", originalSize, compressedSize, deltaHit)

    // Update baseline count if needed
    baselineCount := getActiveBaselineCount()
    hdeMetrics.OnBaselineUpdate("vm_state", baselineCount)

    return compressed, nil
}
```

### 4. Instrument VM Migrations

```go
func migrateVM(vm *VM, destNode string, useDWCP bool) error {
    migrationMetrics := metrics.NewMigrationMetricsWrapper()

    migrationID := fmt.Sprintf("migration-%s", vm.ID)
    migrationMetrics.OnMigrationStart(migrationID)

    // Perform migration
    err := performMigration(vm, destNode, useDWCP)

    // Record completion
    migrationMetrics.OnMigrationComplete(migrationID, destNode, useDWCP)

    if useDWCP && err == nil {
        // Calculate and record speedup factor
        speedup := calculateSpeedupFactor(vm)
        migrationMetrics.OnSpeedupCalculated(vm.Type, speedup)
    }

    return err
}
```

## Configure Prometheus

### Update Prometheus to Scrape Your Metrics

Edit `/home/kp/novacron/configs/prometheus/dwcp-scrape-config.yml`:

```yaml
scrape_configs:
  - job_name: 'dwcp-metrics'
    scrape_interval: 10s
    static_configs:
      - targets:
          - 'your-app-host:9090'  # Update with your host
        labels:
          cluster: 'production-cluster'
          service: 'dwcp'
```

### Reload Prometheus Configuration

```bash
# If using Docker
docker-compose -f docker-compose.monitoring.yml restart prometheus

# If using native Prometheus
curl -X POST http://localhost:9091/-/reload
```

## Import Grafana Dashboard

### 1. Access Grafana

Navigate to http://localhost:3001 and login:
- Username: `admin`
- Password: `dwcp-admin-2025`

### 2. Add Prometheus Data Source

1. Click gear icon → "Data Sources"
2. Click "Add data source"
3. Select "Prometheus"
4. Set URL to `http://prometheus:9090` (Docker) or `http://localhost:9091` (native)
5. Click "Save & Test"

### 3. Import DWCP Dashboard

1. Click "+" icon → "Import"
2. Click "Upload JSON file"
3. Select `/home/kp/novacron/configs/grafana/dwcp-dashboard.json`
4. Select Prometheus data source
5. Click "Import"

## Verify Metrics Collection

### Check Metrics Endpoint

```bash
# Verify metrics are being exposed
curl http://localhost:9090/metrics | grep dwcp_

# Should see output like:
# dwcp_amst_streams_active{cluster="cluster-1",node="node-01"} 5
# dwcp_hde_compression_ratio_bucket{cluster="cluster-1",node="node-01",data_type="vm_memory",le="5"} 10
# ...
```

### Check Prometheus Targets

1. Navigate to http://localhost:9091/targets
2. Verify "dwcp-metrics" job is "UP"
3. Check last scrape time and duration

### View Dashboard

1. Navigate to http://localhost:3001
2. Go to Dashboards → DWCP Performance Monitoring
3. You should see real-time metrics:
   - Active AMST streams
   - Compression ratios
   - Migration performance
   - Bandwidth utilization
   - Error rates

## Testing Alerts

### Trigger Test Alert

```bash
# Send test data to trigger high error rate alert
for i in {1..100}; do
  curl -X POST http://localhost:9090/test-error
done

# Wait 5 minutes for alert to fire
```

### Check Alertmanager

Navigate to http://localhost:9093 to see active alerts.

## Production Checklist

- [ ] Metrics exposed at `:9090/metrics`
- [ ] Prometheus scraping successfully (check targets page)
- [ ] Grafana dashboard showing real data
- [ ] Alert rules loaded in Prometheus
- [ ] Alertmanager configured with notification channels
- [ ] Component health metrics reporting correctly
- [ ] Migration metrics showing DWCP vs standard comparison
- [ ] Compression ratio metrics within expected range (>5x)
- [ ] Bandwidth savings being calculated
- [ ] SLA tracking dashboards configured

## Performance Tuning

### Reduce Metric Cardinality

If you see high memory usage in Prometheus:

```go
// Before (high cardinality - unique stream per VM)
amstMetrics.OnStreamData(vmID, bytes, 0, "")

// After (lower cardinality - reuse stream IDs)
streamID := getPooledStreamID()
amstMetrics.OnStreamData(streamID, bytes, 0, "")
```

### Adjust Scrape Intervals

For high-volume environments, increase scrape intervals:

```yaml
scrape_configs:
  - job_name: 'dwcp-metrics'
    scrape_interval: 30s  # Increased from 10s
```

### Enable Recording Rules

Recording rules pre-compute expensive queries:

```yaml
rule_files:
  - '/home/kp/novacron/configs/prometheus/dwcp-recording-rules.yml'
```

## Troubleshooting

### Metrics Not Appearing

**Problem**: `curl http://localhost:9090/metrics` returns 404

**Solution**:
```bash
# Check if exporter is running
curl http://localhost:9090/health

# Check logs for errors
tail -f /var/log/novacron/metrics.log

# Verify initialization
grep "metrics.*initialized" /var/log/novacron/app.log
```

### Prometheus Can't Scrape

**Problem**: Prometheus targets page shows "DOWN"

**Solution**:
```bash
# Check connectivity
curl http://localhost:9090/metrics

# Check firewall
sudo ufw allow 9090/tcp

# Verify scrape config
docker-compose -f docker-compose.monitoring.yml logs prometheus
```

### Grafana Shows No Data

**Problem**: Dashboard panels show "No data"

**Solution**:
1. Verify Prometheus data source is configured
2. Check query syntax in panel settings
3. Ensure time range matches data availability
4. Run query directly in Prometheus to verify data exists

### High Memory Usage

**Problem**: Prometheus using excessive memory

**Solution**:
```bash
# Check metric cardinality
curl http://localhost:9091/api/v1/status/tsdb | jq

# Reduce retention period
prometheus --storage.tsdb.retention.time=7d

# Enable metric relabeling to drop high-cardinality labels
```

## Next Steps

1. **Configure Alerting**: Update `/home/kp/novacron/configs/prometheus/alertmanager.yml` with your email/Slack webhook
2. **Customize Dashboards**: Add panels for your specific use cases
3. **Set Up Long-Term Storage**: Configure remote write to long-term storage (e.g., Thanos, Cortex)
4. **Enable Tracing**: Integrate with Jaeger/Zipkin for distributed tracing
5. **Add Custom Metrics**: Extend the metrics package for application-specific needs

## Reference

- **Metrics Documentation**: `/home/kp/novacron/backend/core/network/dwcp/metrics/README.md`
- **Example Code**: `/home/kp/novacron/backend/core/network/dwcp/metrics/examples_test.go`
- **Prometheus Docs**: https://prometheus.io/docs/
- **Grafana Docs**: https://grafana.com/docs/
- **Go Prometheus Client**: https://github.com/prometheus/client_golang

## Support

For issues or questions:
- Check metrics endpoint: `curl http://localhost:9090/metrics | grep dwcp_`
- Verify component health: `curl http://localhost:9090/metrics | grep component_health`
- Review Prometheus targets: http://localhost:9091/targets
- Check Grafana data source: http://localhost:3001/datasources
