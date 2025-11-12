# DWCP v3 Transport Layer

## Overview

AMST (Adaptive Multi-Stream Transport) v3 provides hybrid datacenter + internet transport capabilities with automatic mode detection and switching.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        AMSTv3                                │
│  ┌───────────────────────────────────────────────────────┐  │
│  │           Mode Detector (upgrade package)             │  │
│  │  • Latency measurement (<10ms = DC, >50ms = Internet) │  │
│  │  • Bandwidth estimation (>1Gbps = DC, <1Gbps = Int)   │  │
│  │  • Historical averaging for stable decisions           │  │
│  └───────────────────────────────────────────────────────┘  │
│                             │                                │
│              ┌──────────────┴──────────────┐                │
│              ▼                              ▼                │
│  ┌───────────────────────┐    ┌───────────────────────┐    │
│  │  Datacenter Transport │    │  Internet Transport   │    │
│  │  (v1 RDMA)            │    │  (v3 TCP)             │    │
│  │  • 32-512 streams     │    │  • 4-16 streams       │    │
│  │  • 10-100 Gbps        │    │  • 100-900 Mbps       │    │
│  │  • <10ms latency      │    │  • 50-500ms tolerant  │    │
│  └───────────────────────┘    │  • BBR/CUBIC CC       │    │
│                                │  • Packet pacing      │    │
│                                └───────────────────────┘    │
│                                            │                 │
│                                ┌───────────┴──────────┐     │
│                                ▼                       │     │
│                    ┌──────────────────────┐           │     │
│                    │ Congestion Controller│           │     │
│                    │  • BBR algorithm     │           │     │
│                    │  • CUBIC algorithm   │           │     │
│                    │  • Pacing control    │           │     │
│                    │  • Metrics tracking  │           │     │
│                    └──────────────────────┘           │     │
└────────────────────────────────────────────────────────────┘
```

## Files

### Core Implementation
- **`amst_v3.go`**: Hybrid transport manager with automatic mode switching
- **`tcp_transport_v3.go`**: Internet-optimized TCP transport (v3)
- **`congestion_controller.go`**: BBR and CUBIC congestion control algorithms

### Tests
- **`amst_v3_test.go`**: Comprehensive test suite with benchmarks

## Quick Start

### Basic Usage

```go
package main

import (
    "context"
    "github.com/khryptorgraphics/novacron/backend/core/network/dwcp/v3/transport"
    "github.com/khryptorgraphics/novacron/backend/core/network/dwcp/upgrade"
    "go.uber.org/zap"
)

func main() {
    logger, _ := zap.NewProduction()
    defer logger.Sync()

    // Create mode detector
    detector := upgrade.NewModeDetector()

    // Configure AMST v3
    config := transport.DefaultAMSTv3Config()
    config.EnableDatacenter = true  // Enable RDMA for datacenter
    config.EnableInternet = true    // Enable TCP for internet
    config.AutoMode = true           // Auto-detect mode
    config.CongestionAlgorithm = "bbr"

    // Create transport
    amst, err := transport.NewAMSTv3(config, detector, logger)
    if err != nil {
        logger.Fatal("Failed to create AMST v3", zap.Error(err))
    }

    // Start transport
    ctx := context.Background()
    if err := amst.Start(ctx, "peer-host:9000"); err != nil {
        logger.Fatal("Failed to start AMST v3", zap.Error(err))
    }
    defer amst.Close()

    // Send data (automatically selects optimal transport)
    data := make([]byte, 10*1024*1024) // 10MB
    if err := amst.SendData(ctx, data); err != nil {
        logger.Error("Failed to send data", zap.Error(err))
    }

    // Get metrics
    metrics := amst.GetMetrics()
    logger.Info("Transfer complete",
        zap.String("mode", metrics.Mode),
        zap.Int32("active_streams", metrics.ActiveStreams),
        zap.Uint64("bytes_sent", metrics.TotalBytesSent))
}
```

### Force Specific Mode

```go
// Force datacenter mode (RDMA)
detector := upgrade.NewModeDetector()
detector.ForceMode(upgrade.ModeDatacenter)

// Force internet mode (TCP v3)
detector.ForceMode(upgrade.ModeInternet)

// Use hybrid adaptive mode (default)
detector.ForceMode(upgrade.ModeHybrid)
```

### With Progress Tracking

```go
// Send with progress callback
totalSize := int64(len(data))
err := amst.TransferWithProgress(ctx, data, func(transferred, total int64) {
    progress := float64(transferred) / float64(total) * 100
    logger.Info("Transfer progress",
        zap.Float64("percent", progress),
        zap.Int64("bytes", transferred),
        zap.Int64("total", total))
})
```

## Configuration Options

### AMSTv3Config

```go
type AMSTv3Config struct {
    // Transport selection
    EnableDatacenter bool   // Enable datacenter mode (RDMA)
    EnableInternet   bool   // Enable internet mode (TCP)
    AutoMode         bool   // Auto-detect and switch modes

    // Datacenter settings (v1 compatibility)
    DatacenterStreams int    // 32-512 streams
    RDMADevice        string // RDMA device name
    RDMAPort          int    // RDMA port

    // Internet settings (v3 features)
    InternetStreams     int    // 4-16 streams
    CongestionAlgorithm string // "bbr" or "cubic"
    PacingEnabled       bool   // Enable packet pacing
    PacingRate          int64  // bytes per second

    // Adaptive tuning
    AutoTune            bool          // Enable auto-tuning
    ChunkSizeKB         int           // Chunk size in KB
    ConnectTimeout      time.Duration // Connection timeout
    ModeCheckInterval   time.Duration // Mode check frequency
    ModeSwitchThreshold float64       // Mode switch confidence (0-1)

    // Common settings
    RemoteAddr string // Remote address
    MinStreams int    // Minimum streams
    MaxStreams int    // Maximum streams
}
```

### Defaults

```go
config := DefaultAMSTv3Config()
// Returns:
// - EnableDatacenter: true
// - EnableInternet: true
// - AutoMode: true
// - DatacenterStreams: 64
// - InternetStreams: 8
// - CongestionAlgorithm: "bbr"
// - PacingEnabled: true
// - PacingRate: 1 Gbps
// - AutoTune: true
// - ChunkSizeKB: 256
// - ModeCheckInterval: 5s
// - ModeSwitchThreshold: 0.7
```

## Network Modes

### Datacenter Mode (v1 RDMA)
- **When**: Latency <10ms AND Bandwidth >1 Gbps
- **Transport**: RDMA (v1 compatible)
- **Streams**: 32-512 parallel connections
- **Throughput**: 10-100 Gbps
- **Use Case**: Same-datacenter, high-speed interconnects

### Internet Mode (v3 TCP)
- **When**: Latency >50ms OR Bandwidth <1 Gbps
- **Transport**: TCP v3 with BBR/CUBIC
- **Streams**: 4-16 adaptive streams
- **Throughput**: 100-900 Mbps
- **Use Case**: WAN, cross-datacenter, internet transfers

### Hybrid Mode (Adaptive)
- **When**: Borderline conditions (10-50ms latency)
- **Transport**: Both available, selects based on data size
- **Streams**: Varies by selected transport
- **Throughput**: Optimized for current conditions
- **Use Case**: Dynamic environments, mixed workloads

## Congestion Control

### BBR (Bottleneck Bandwidth and RTT)
- **Best for**: High-latency, high-bandwidth networks
- **Features**:
  - Doesn't reduce cwnd on isolated packet loss
  - Probes for bottleneck bandwidth
  - Maintains minimal RTT
  - Four-phase operation: Startup → Drain → ProbeBW → ProbeRTT
- **Use in**: Internet mode, WAN transfers

### CUBIC
- **Best for**: Datacenter, low-latency networks
- **Features**:
  - Cubic window growth function
  - Fast convergence for fairness
  - TCP-friendly fallback
  - 30% multiplicative decrease on loss
- **Use in**: Datacenter mode, traditional TCP scenarios

## Performance Targets

| Mode | Bandwidth | Latency | Streams | Packet Loss Tolerance |
|------|-----------|---------|---------|----------------------|
| Datacenter | 10-100 Gbps | <10 ms | 32-512 | <0.1% |
| Internet | 100-900 Mbps | 50-500 ms | 4-16 | <5% |
| Hybrid | Adaptive | Variable | 4-512 | <2% |

**Mode Switching**: <2 seconds

## Testing

### Run Unit Tests
```bash
cd v3/transport
go test -v -race .
```

### Run Benchmarks
```bash
go test -bench=. -benchmem
```

### Test Specific Scenarios
```bash
# Datacenter mode
go test -v -run TestAMSTv3_DatacenterMode

# Internet mode
go test -v -run TestAMSTv3_InternetMode

# Hybrid mode
go test -v -run TestAMSTv3_HybridMode

# Mode switching performance
go test -v -run TestAMSTv3_ModeSwitchPerformance

# Backward compatibility
go test -v -run TestAMSTv3_BackwardCompatibility
```

## Metrics

### Available Metrics

```go
type TransportMetrics struct {
    // Stream metrics
    ActiveStreams int32   // Currently active streams
    TotalStreams  int     // Total configured streams

    // Transfer metrics
    TotalBytesSent uint64 // Total bytes sent
    TotalBytesRecv uint64 // Total bytes received

    // Performance metrics
    ThroughputMbps   float64       // Current throughput
    AverageLatencyMs float64       // Average latency
    PacketLossRate   float64       // Packet loss rate (0-1)

    // Mode information
    Mode              string // "datacenter", "internet", "hybrid"
    TransportType     string // "rdma", "tcp-v3", "hybrid"
    CongestionControl string // "bbr", "cubic"

    // Health
    Healthy         bool      // Overall health status
    LastHealthCheck time.Time // Last health check time
}
```

### Accessing Metrics

```go
metrics := amst.GetMetrics()

fmt.Printf("Mode: %s\n", metrics.Mode)
fmt.Printf("Active Streams: %d\n", metrics.ActiveStreams)
fmt.Printf("Throughput: %.2f Mbps\n", metrics.ThroughputMbps)
fmt.Printf("Latency: %.2f ms\n", metrics.AverageLatencyMs)
fmt.Printf("Packet Loss: %.4f%%\n", metrics.PacketLossRate*100)
```

### Prometheus Integration

Metrics are automatically exported to Prometheus:
- `novacron_amst_active_streams`
- `novacron_amst_bytes_transferred_total`
- `novacron_amst_throughput_mbps`
- `novacron_amst_latency_seconds`
- `novacron_amst_packet_loss_rate`
- `novacron_amst_health_status`
- `novacron_amst_congestion_window`

## Troubleshooting

### High Packet Loss
```go
// Reduce chunk size
config.ChunkSizeKB = 64 // Default: 256

// Enable pacing
config.PacingEnabled = true
config.PacingRate = 100 * 1024 * 1024 // 100 Mbps

// Use BBR (more tolerant of loss)
config.CongestionAlgorithm = "bbr"
```

### Low Throughput
```go
// Increase streams
config.InternetStreams = 16 // Max for internet mode

// Increase chunk size (if loss is low)
config.ChunkSizeKB = 512

// Enable auto-tuning
config.AutoTune = true
```

### Mode Not Switching
```go
// Force mode manually
detector.ForceMode(upgrade.ModeInternet)

// Reduce switch threshold
config.ModeSwitchThreshold = 0.5 // Lower = more aggressive

// Check mode detection interval
config.ModeCheckInterval = 2 * time.Second // More frequent checks
```

## Best Practices

1. **Enable Auto-Tuning**: Let AMST v3 optimize parameters automatically
2. **Use BBR for Internet**: Better performance on high-latency networks
3. **Monitor Metrics**: Track throughput and packet loss for tuning
4. **Test Mode Switching**: Verify behavior in your environment
5. **Start with Defaults**: Default config is optimized for most scenarios

## Backward Compatibility

AMST v3 is fully backward compatible with v1:
```go
// v1 API still works
amst := dwcp.NewAMST(v1Config)
amst.Connect(ctx, host, port)
amst.Transfer(ctx, data, progressCallback)

// Transparently uses v3 features when available
```

## Dependencies

- Go 1.21+
- Linux kernel 4.9+ (for BBR support)
- RDMA-capable NICs (optional, for datacenter mode)
- libibverbs (optional, for RDMA)

## See Also

- [AMST V3 Implementation Summary](../../docs/AMST-V3-IMPLEMENTATION-SUMMARY.md)
- [Mode Detector](../upgrade/mode_detector.go)
- [DWCP Specification](../../docs/DWCP-SPECIFICATION.md)
- [Performance Tuning Guide](../../docs/PERFORMANCE-TUNING.md)

## License

Copyright (c) 2025 NovaCron. All rights reserved.
