# AMST Quick Start Guide

## TL;DR - Get Started in 5 Minutes

### 1. Basic Usage

```go
package main

import (
    "log"
    "github.com/khryptorgraphics/novacron/backend/core/network/dwcp/transport"
    "go.uber.org/zap"
)

func main() {
    logger, _ := zap.NewProduction()
    
    // Create config
    config := transport.DefaultTransportConfig()
    config.RemoteAddr = "remote-server:5000"
    
    // Create transport (RDMA with TCP fallback)
    t, err := transport.NewRDMATransport(config, logger)
    if err != nil {
        log.Fatal(err)
    }
    defer t.Close()
    
    // Start transport
    if err := t.Start(); err != nil {
        log.Fatal(err)
    }
    
    // Send data
    data := []byte("Hello, AMST!")
    if err := t.Send(data); err != nil {
        log.Fatal(err)
    }
    
    logger.Info("Data sent successfully")
}
```

### 2. Enable BBR Congestion Control

```go
config := transport.DefaultTransportConfig()
config.CongestionAlgorithm = "bbr"  // Requires Linux kernel 4.9+
```

### 3. Configure Stream Count

```go
config := transport.DefaultTransportConfig()
config.MinStreams = 32   // Minimum streams
config.MaxStreams = 512  // Maximum streams
config.AutoTune = true   // Automatically adjust based on network
```

### 4. Monitor Health

```go
// Automatic health monitoring (every 10 seconds)
// Manual health check:
if err := transport.HealthCheck(); err != nil {
    logger.Error("Transport unhealthy", zap.Error(err))
}

// Get stream health details
healthMap := mst.GetStreamHealth()
for id, health := range healthMap {
    logger.Info("Stream health",
        zap.Int("stream_id", id),
        zap.Bool("healthy", health.Healthy),
        zap.Int("reconnects", health.Reconnects))
}
```

### 5. View Metrics

```go
metrics := transport.GetMetrics()
logger.Info("Transport metrics",
    zap.Int32("active_streams", metrics.ActiveStreams),
    zap.Uint64("bytes_sent", metrics.TotalBytesSent),
    zap.Float64("throughput_mbps", metrics.ThroughputMbps),
    zap.String("transport_type", metrics.TransportType))
```

### 6. Production Configuration

```yaml
# /configs/dwcp.yaml
transport:
  amst:
    min_streams: 16
    max_streams: 256
    chunk_size_kb: 256
    auto_tune: true
    pacing_enabled: true
    pacing_rate: 1073741824  # 1 Gbps
    congestion_algorithm: "bbr"
    enable_rdma: false
    health_check_interval: 10s

monitoring:
  prometheus:
    enabled: true
    listen_addr: ":9090"
```

### 7. Prometheus Metrics

Access metrics at `http://localhost:9090/metrics`:

```
novacron_amst_active_streams 32
novacron_amst_throughput_mbps 850.5
novacron_amst_bytes_transferred_total{direction="sent"} 1048576000
novacron_amst_health_status 1
```

## Advanced Usage

### Custom Transport Implementation

```go
type MyTransport struct {
    *transport.RDMATransport
}

func (mt *MyTransport) Send(data []byte) error {
    // Custom preprocessing
    compressed := compress(data)
    
    // Use underlying transport
    return mt.RDMATransport.Send(compressed)
}
```

### Dynamic Stream Adjustment

```go
// Based on network conditions
bandwidth := 1000.0  // 1 Gbps
latency := 50.0      // 50ms

if err := transport.AdjustStreams(bandwidth, latency); err != nil {
    logger.Error("Failed to adjust streams", zap.Error(err))
}
```

### Integration with DWCP Manager

```go
import "github.com/khryptorgraphics/novacron/backend/core/network/dwcp"

manager, err := dwcp.NewManager(config, logger)
if err != nil {
    log.Fatal(err)
}

if err := manager.Start(); err != nil {
    log.Fatal(err)
}

// Get transport for direct use
transport := manager.GetTransport()
```

## Troubleshooting

### BBR Not Working
```bash
# Check if BBR is available
sysctl net.ipv4.tcp_available_congestion_control

# Enable BBR
sudo modprobe tcp_bbr
echo "tcp_bbr" | sudo tee -a /etc/modules-load.d/modules.conf
echo "net.core.default_qdisc=fq" | sudo tee -a /etc/sysctl.conf
echo "net.ipv4.tcp_congestion_control=bbr" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

### Stream Connection Failures
- Check firewall rules
- Verify remote address is reachable
- Check health monitoring logs for reconnection attempts

### Low Throughput
- Increase max_streams
- Enable auto_tune
- Check network latency and bandwidth
- Verify BBR is active

## Performance Tuning

### For Low Latency (<5ms)
```yaml
min_streams: 8
max_streams: 32
chunk_size_kb: 64
pacing_enabled: false
```

### For High Latency (>100ms)
```yaml
min_streams: 64
max_streams: 512
chunk_size_kb: 512
pacing_enabled: true
congestion_algorithm: "bbr"
```

### For High Throughput (10+ Gbps)
```yaml
min_streams: 128
max_streams: 1024
chunk_size_kb: 1024
enable_rdma: true  # If hardware available
```

## Next Steps

1. Read full documentation: `/docs/AMST_PHASE1_IMPLEMENTATION.md`
2. Review configuration: `/configs/dwcp.yaml`
3. Check tests: `/backend/core/network/dwcp/transport/*_test.go`
4. Monitor metrics: `http://localhost:9090/metrics`

---

Questions? Check the implementation doc or review the code in:
- `/backend/core/network/dwcp/transport/multi_stream_tcp.go`
- `/backend/core/network/dwcp/transport/rdma_transport.go`
