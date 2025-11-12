# HDE v3 (Hierarchical Delta Encoding v3)

## Overview

HDE v3 is the third generation of the Hierarchical Delta Encoding system for DWCP (Distributed WAN Communication Protocol). It provides highly optimized data compression and synchronization for VM migration across datacenter and internet networks.

## Key Features

### 1. ML-Based Compression Selection
- **Adaptive algorithm selection** based on data characteristics
- **Mode-aware optimization** (datacenter vs internet)
- **Performance learning** from historical compression results
- Automatic data analysis (entropy, compressibility, patterns)

### 2. Enhanced Delta Encoding
- Reuses proven delta encoding from HDE v1
- Block-level delta computation for efficient updates
- Baseline management with automatic refresh
- Deduplication at the block level

### 3. CRDT-Based Conflict-Free Synchronization
- **Conflict-free state replication** using CRDTs (ORMap)
- **Vector clocks** for causal ordering
- **Anti-entropy synchronization** with digest comparison
- Last-Writer-Wins conflict resolution

### 4. Network Mode Optimization

#### Datacenter Mode (v1 compatibility)
- **Prioritizes speed** over compression ratio
- Uses LZ4 (fastest) or Zstd (balanced)
- Optimized for low-latency, high-bandwidth networks
- Target: <10ms latency, >1 Gbps bandwidth

#### Internet Mode (v3 new)
- **Prioritizes compression ratio** over speed
- Uses Zstd Max compression (level 19)
- Optimized for high-latency, bandwidth-constrained networks
- Target: 70-85% bandwidth reduction

#### Hybrid Mode
- Adaptive selection based on network conditions
- ML-based algorithm learning
- Automatic mode switching

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         HDE v3                              │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐  ┌─────────────────┐                 │
│  │ ML Compression   │  │ Delta Encoder   │                 │
│  │ Selector         │  │ (from v1)       │                 │
│  └────────┬─────────┘  └────────┬────────┘                 │
│           │                     │                           │
│           ▼                     ▼                           │
│  ┌──────────────────────────────────────┐                  │
│  │   Compression Engines                │                  │
│  │  - LZ4 (fast)                        │                  │
│  │  - Zstd (balanced)                   │                  │
│  │  - Zstd Max (high compression)       │                  │
│  └──────────────────────────────────────┘                  │
│                                                             │
│  ┌──────────────────────────────────────┐                  │
│  │   CRDT Integration                   │                  │
│  │  - ORMap for metadata                │                  │
│  │  - Vector clocks                     │                  │
│  │  - Conflict resolution               │                  │
│  └──────────────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

## Performance Targets

### Compression Ratios
- **Datacenter Mode**: 20-40% reduction (emphasis on speed)
- **Internet Mode**: 70-85% reduction (emphasis on bandwidth savings)
- **Delta Encoding**: Additional 20-40% on top of compression

### Speed
- **Datacenter Mode**: <10ms compression time for 1MB
- **Internet Mode**: <100ms compression time for 1MB
- **Throughput**: >100 MB/s compression rate

## Usage

### Basic Compression

```go
import "github.com/khryptorgraphics/novacron/backend/core/network/dwcp/v3/encoding"

// Create HDE v3 instance
config := encoding.DefaultHDEv3Config("node-1")
config.NetworkMode = upgrade.ModeInternet
config.EnableMLCompression = true
config.EnableDeltaEncoding = true
config.EnableCRDT = true

hde, err := encoding.NewHDEv3(config)
if err != nil {
    log.Fatal(err)
}
defer hde.Close()

// Compress data
vmData := []byte("VM memory state...")
compressed, err := hde.Compress("vm-123", vmData)
if err != nil {
    log.Fatal(err)
}

log.Printf("Compression: %d -> %d bytes (%.2fx ratio)",
    compressed.OriginalSize,
    compressed.CompressedSize,
    compressed.CompressionRatio())

// Decompress
decompressed, err := hde.Decompress(compressed)
if err != nil {
    log.Fatal(err)
}
```

### Mode Switching

```go
// Start in datacenter mode
hde.UpdateNetworkMode(upgrade.ModeDatacenter)

// Compress with fast algorithm
compressed1, _ := hde.Compress("vm-123", data)
// Uses LZ4 or Zstd (fast)

// Switch to internet mode
hde.UpdateNetworkMode(upgrade.ModeInternet)

// Compress with maximum compression
compressed2, _ := hde.Compress("vm-456", data)
// Uses Zstd Max (high compression)
```

### CRDT Synchronization

```go
// Node 1: Compress and export CRDT state
hde1, _ := encoding.NewHDEv3(config1)
compressed, _ := hde1.Compress("shared-vm", data)
crdtState, _ := hde1.ExportCRDTState()

// Send crdtState to Node 2...

// Node 2: Merge CRDT state
hde2, _ := encoding.NewHDEv3(config2)
err := hde2.MergeRemoteCRDT(crdtState)
if err != nil {
    log.Printf("Conflict detected, resolving...")
}
```

### Metrics Monitoring

```go
// Get comprehensive metrics
metrics := hde.GetMetrics()

fmt.Printf("Compression ratio: %.2fx\n", metrics["compression_ratio"])
fmt.Printf("Bandwidth savings: %.1f%%\n", metrics["compression_percent"])
fmt.Printf("Delta hit rate: %.1f%%\n", metrics["delta_hit_rate"])

// Algorithm usage
algoUsage := metrics["algorithm_usage"].(map[string]int64)
for algo, count := range algoUsage {
    fmt.Printf("  %s: %d uses\n", algo, count)
}

// CRDT stats
fmt.Printf("CRDT merges: %v\n", metrics["crdt_crdt_merges"])
fmt.Printf("Conflict-free: %v\n", metrics["crdt_conflict_free_count"])
```

## Configuration Options

### HDEv3Config

```go
type HDEv3Config struct {
    NodeID string  // Unique node identifier for CRDT

    // Network mode selection
    NetworkMode upgrade.NetworkMode  // ModeDatacenter, ModeInternet, ModeHybrid

    // Feature flags
    EnableMLCompression  bool  // Use ML-based algorithm selection
    EnableDeltaEncoding  bool  // Use delta encoding (v1 compatibility)
    EnableCRDT           bool  // Enable CRDT synchronization

    // Delta encoding settings (from v1)
    DeltaConfig *compression.DeltaEncodingConfig

    // Baseline management
    BaselineRefreshInterval time.Duration  // How often to refresh baselines
    MaxBaselines            int            // Maximum cached baselines

    // Compression selector settings
    SelectorConfig *SelectorConfig

    // Resource limits
    MaxMemoryUsage int64  // Maximum memory for caching
}
```

### SelectorConfig

```go
type SelectorConfig struct {
    // ML learning parameters
    LearningRate     float64  // Learning rate for adaptive selection (0.0-1.0)
    AdaptiveEnabled  bool     // Enable adaptive learning

    // Size thresholds
    SmallDataSize    int  // Below this size: minimal compression
    LargeDataSize    int  // Above this size: aggressive compression

    // Performance targets
    DatacenterTargetSpeed float64  // MB/s target for datacenter mode
    InternetTargetRatio   float64  // Compression ratio target for internet mode
}
```

## Compression Algorithms

### LZ4
- **Speed**: Fastest (>500 MB/s)
- **Ratio**: 2-3x typical
- **Use case**: Datacenter mode, small data
- **Level**: 1

### Zstd (Balanced)
- **Speed**: Fast (>100 MB/s)
- **Ratio**: 3-5x typical
- **Use case**: Hybrid mode, medium data
- **Level**: 3

### Zstd Max
- **Speed**: Moderate (10-50 MB/s)
- **Ratio**: 5-10x typical (up to 85% reduction)
- **Use case**: Internet mode, large data
- **Level**: 19

## Delta Encoding

HDE v3 reuses the proven delta encoding from v1:

1. **Baseline Creation**: First transmission creates a baseline
2. **Delta Computation**: Subsequent transmissions compute deltas
3. **Block-Level Comparison**: Efficient 4KB block comparisons
4. **Automatic Refresh**: Baselines refreshed periodically or after N deltas

### Delta Performance
- **Typical reduction**: 20-40% on top of compression
- **Hit rate**: 60-80% for VM memory (high temporal locality)
- **Overhead**: <1ms for delta computation

## CRDT Synchronization

### Conflict-Free Guarantees
- Uses ORMap (Observed-Remove Map) CRDT
- Vector clocks ensure causal consistency
- Last-Writer-Wins (LWW) conflict resolution
- Anti-entropy synchronization via digests

### Synchronization Protocol
1. **Local Update**: Register baseline with CRDT
2. **Export State**: Serialize CRDT state for transmission
3. **Merge Remote**: Apply remote CRDT state (conflict-free)
4. **Conflict Resolution**: Automatic LWW resolution if needed

## Testing

### Run All Tests
```bash
go test -v ./v3/encoding/
```

### Run Specific Test
```bash
go test -v -run TestHDEv3_PerformanceTargets ./v3/encoding/
```

### Run Benchmarks
```bash
go test -bench=. -benchmem ./v3/encoding/
```

### Race Detection
```bash
go test -race -v ./v3/encoding/
```

## Benchmarks

```
BenchmarkHDEv3_Compress_1KB-96        26200    43517 ns/op    1603 B/op    17 allocs/op
BenchmarkHDEv3_Compress_1MB-96           81 13334725 ns/op    2148 B/op    17 allocs/op
BenchmarkCompressionSelector-96       61956    18510 ns/op       0 B/op     0 allocs/op
```

## Migration from v1

HDE v3 is **backward compatible** with v1:

```go
// v1 code
hde1 := dwcp.NewHDE(v1Config)
compressed := hde1.CompressMemory("vm-123", data, dwcp.CompressionGlobal)

// v3 equivalent
config := encoding.DefaultHDEv3Config("node-1")
config.NetworkMode = upgrade.ModeInternet  // Same as CompressionGlobal
hde3 := encoding.NewHDEv3(config)
compressed := hde3.Compress("vm-123", data)
```

### Key Differences
- v3 uses ML-based algorithm selection (vs fixed Zstd in v1)
- v3 adds CRDT for conflict-free sync
- v3 supports multiple compression algorithms
- v3 has mode-aware optimization

## Troubleshooting

### Low Compression Ratio
- Check data compressibility (entropy)
- Verify network mode matches use case
- Enable ML compression for adaptive selection

### High Latency
- Use datacenter mode for speed priority
- Reduce compression level
- Check baseline refresh interval

### CRDT Conflicts
- Verify vector clocks are updating
- Check node IDs are unique
- Review Last-Writer-Wins semantics

## Future Enhancements (v4)

- Hardware acceleration (Intel QAT, NVIDIA GPUs)
- Brotli compression for static data
- Smart baseline selection based on similarity
- Distributed baseline sharing across cluster
- Predictive compression based on workload patterns

## References

- [DWCP v3 Specification](../../docs/DWCP-V3-QUICK-START.md)
- [Mode Detector](../../upgrade/mode_detector.go)
- [CRDT Interfaces](../../sync/crdt/interfaces.go)
- [Delta Encoder v1](../../compression/delta_encoder.go)

## License

Copyright © 2024 NovaCron Project
