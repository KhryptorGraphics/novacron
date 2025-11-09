# DWCP Quick Start Guide
## Getting Started with Distributed WAN Communication Protocol

**Version:** 1.0  
**Date:** 2025-11-08

---

## Prerequisites

Before implementing DWCP, ensure you have:

1. ✅ Go 1.21+ installed
2. ✅ Access to NovaCron codebase
3. ✅ Multi-region test environment (or network simulation tools)
4. ✅ Understanding of distributed systems concepts
5. ✅ Familiarity with NovaCron architecture

## Phase 1 Implementation (Weeks 1-4)

### Step 1: Create DWCP Directory Structure

```bash
cd backend/core/network
mkdir -p dwcp/{transport,compression,prediction,sync,partition,consensus}
```

### Step 2: Implement Multi-Stream TCP (AMST)

**File**: `backend/core/network/dwcp/transport/multi_stream_tcp.go`

```go
package transport

import (
    "net"
    "sync"
)

// MultiStreamTCP manages multiple parallel TCP connections
type MultiStreamTCP struct {
    streams      []*net.TCPConn
    numStreams   int
    chunkSize    int
    pacingRate   int64
    mu           sync.RWMutex
}

// Config for AMST
type AMSTConfig struct {
    MinStreams   int
    MaxStreams   int
    ChunkSizeKB  int
    AutoTune     bool
    PacingEnabled bool
}

// NewMultiStreamTCP creates a new multi-stream TCP connection
func NewMultiStreamTCP(config AMSTConfig) *MultiStreamTCP {
    return &MultiStreamTCP{
        streams:    make([]*net.TCPConn, 0, config.MaxStreams),
        numStreams: config.MinStreams,
        chunkSize:  config.ChunkSizeKB * 1024,
    }
}

// Send data across all streams in parallel
func (m *MultiStreamTCP) Send(data []byte) error {
    // Implementation: Split data across streams
    // Use goroutines for parallel sending
    // Apply packet pacing if enabled
    return nil
}

// Receive data from all streams
func (m *MultiStreamTCP) Receive() ([]byte, error) {
    // Implementation: Receive from all streams
    // Merge data in correct order
    return nil, nil
}

// AdjustStreams dynamically adjusts the number of streams
func (m *MultiStreamTCP) AdjustStreams(bandwidth, latency float64) {
    // Algorithm from spec:
    // optimal_streams = min(256, max(16, bandwidth_mbps / (latency_ms * 0.1)))
}
```

### Step 3: Implement Delta Encoding (HDE)

**File**: `backend/core/network/dwcp/compression/delta_encoder.go`

```go
package compression

import (
    "crypto/sha256"
    "github.com/klauspost/compress/zstd"
)

type DeltaEncoder struct {
    baseline      map[string][]byte
    compressionLevel int
    encoder       *zstd.Encoder
}

type DeltaState struct {
    BaselineVersion  uint64
    DeltaOperations  []DeltaOp
    CompressionRatio float64
    Checksum         [32]byte
}

type DeltaOp struct {
    OpType    DeltaOpType
    Path      string
    OldValue  []byte
    NewValue  []byte
}

type DeltaOpType int

const (
    DeltaAdd DeltaOpType = iota
    DeltaModify
    DeltaDelete
)

// Encode creates a delta between baseline and current state
func (d *DeltaEncoder) Encode(current map[string][]byte) (*DeltaState, error) {
    // Implementation:
    // 1. Compare current with baseline
    // 2. Generate delta operations
    // 3. Compress delta operations
    // 4. Calculate checksum
    return nil, nil
}

// Decode applies delta to baseline to reconstruct current state
func (d *DeltaEncoder) Decode(delta *DeltaState) (map[string][]byte, error) {
    // Implementation:
    // 1. Decompress delta operations
    // 2. Apply operations to baseline
    // 3. Verify checksum
    return nil, nil
}
```

### Step 4: Integration with Existing Components

**File**: `backend/core/federation/cross_cluster_components.go`

Add DWCP integration:

```go
import (
    "github.com/khryptorgraphics/novacron/backend/core/network/dwcp/transport"
    "github.com/khryptorgraphics/novacron/backend/core/network/dwcp/compression"
)

// Enhance BandwidthOptimizer
type BandwidthOptimizer struct {
    // Existing fields...
    bandwidthMonitor *network.BandwidthMonitor
    compressionEngine *AdaptiveCompressionEngine
    
    // New DWCP fields
    multiStreamTCP *transport.MultiStreamTCP
    deltaEncoder   *compression.DeltaEncoder
}

// Initialize DWCP components
func (b *BandwidthOptimizer) InitDWCP(config DWCPConfig) error {
    b.multiStreamTCP = transport.NewMultiStreamTCP(config.AMST)
    b.deltaEncoder = compression.NewDeltaEncoder(config.HDE)
    return nil
}
```

### Step 5: Configuration

**File**: `backend/core/network/dwcp_config.go`

```go
package network

type DWCPConfig struct {
    Transport    TransportConfig
    Compression  CompressionConfig
    Prediction   PredictionConfig
    Sync         SyncConfig
    Partition    PartitionConfig
    Consensus    ConsensusConfig
}

// Load from YAML
func LoadDWCPConfig(path string) (*DWCPConfig, error) {
    // Implementation
    return nil, nil
}
```

### Step 6: Testing

Create test files:

```bash
# Unit tests
backend/core/network/dwcp/transport/multi_stream_tcp_test.go
backend/core/network/dwcp/compression/delta_encoder_test.go

# Integration tests
backend/core/network/dwcp/integration_test.go
```

Example test:

```go
func TestMultiStreamTCP(t *testing.T) {
    config := AMSTConfig{
        MinStreams: 16,
        MaxStreams: 256,
        ChunkSizeKB: 256,
        AutoTune: true,
    }
    
    mst := NewMultiStreamTCP(config)
    
    // Test data sending
    testData := make([]byte, 1024*1024) // 1MB
    err := mst.Send(testData)
    assert.NoError(t, err)
    
    // Test data receiving
    received, err := mst.Receive()
    assert.NoError(t, err)
    assert.Equal(t, testData, received)
}
```

## Testing Your Implementation

### Local Testing

```bash
# Run unit tests
go test ./backend/core/network/dwcp/...

# Run with coverage
go test -cover ./backend/core/network/dwcp/...

# Run benchmarks
go test -bench=. ./backend/core/network/dwcp/...
```

### Network Simulation

Use `tc` (traffic control) to simulate WAN conditions:

```bash
# Add 100ms latency
sudo tc qdisc add dev eth0 root netem delay 100ms

# Add 1% packet loss
sudo tc qdisc add dev eth0 root netem loss 1%

# Limit bandwidth to 100 Mbps
sudo tc qdisc add dev eth0 root tbf rate 100mbit burst 32kbit latency 400ms
```

## Next Steps

After completing Phase 1:

1. ✅ Verify AMST achieves ≥ 50 MB/s over 100ms link
2. ✅ Verify HDE achieves ≥ 3x compression ratio
3. ✅ Achieve ≥ 80% unit test coverage
4. 📋 Move to Phase 2: Intelligence (Bandwidth Prediction & Task Partitioning)

## Resources

- **Full Specification**: `docs/architecture/distributed-wan-communication-protocol.md`
- **Executive Summary**: `docs/DWCP-EXECUTIVE-SUMMARY.md`
- **Research Papers**: See references in specification
- **NovaCron Architecture**: `docs/architecture/novacron-architecture.md`

## Support

For questions or issues:
1. Review the full specification document
2. Check existing NovaCron components for patterns
3. Consult research papers for algorithmic details
4. Reach out to the development team

---

**Happy Coding!** 🚀


