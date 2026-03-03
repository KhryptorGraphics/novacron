# DWCP v3 API Reference

**Version:** 3.0.0
**Last Updated:** 2025-11-10

## Overview

Complete API reference for DWCP v3 with Go package documentation, configuration options, and usage examples.

---

## Package: `transport/amst_v3`

### AMSTv3

Adaptive Multi-Stream Transport with hybrid datacenter + internet support.

#### Constructor

```go
func NewAMSTv3(
    config *AMSTv3Config,
    detector *upgrade.ModeDetector,
    logger *zap.Logger,
) (*AMSTv3, error)
```

**Example:**
```go
config := transport.DefaultAMSTv3Config()
config.EnableDatacenter = true
config.EnableInternet = true
config.AutoMode = true

detector := upgrade.NewModeDetector()
logger, _ := zap.NewProduction()

amst, err := transport.NewAMSTv3(config, detector, logger)
if err != nil {
    log.Fatal(err)
}
```

#### Methods

##### Start
```go
func (a *AMSTv3) Start(ctx context.Context, remoteAddr string) error
```

Initializes transport and starts mode detection.

**Example:**
```go
err := amst.Start(context.Background(), "10.0.1.50:8080")
if err != nil {
    log.Fatal(err)
}
```

##### SendData
```go
func (a *AMSTv3) SendData(ctx context.Context, data []byte) error
```

Sends data using optimal transport based on current mode.

**Example:**
```go
data := []byte("VM state data")
err := amst.SendData(context.Background(), data)
if err != nil {
    log.Errorf("Send failed: %v", err)
}
```

##### GetCurrentMode
```go
func (a *AMSTv3) GetCurrentMode() upgrade.NetworkMode
```

Returns current network mode.

**Example:**
```go
mode := amst.GetCurrentMode()
fmt.Printf("Current mode: %s\n", mode.String())
// Output: Current mode: datacenter
```

#### Configuration

```go
type AMSTv3Config struct {
    // Transport selection
    EnableDatacenter bool   // Enable datacenter mode (RDMA)
    EnableInternet   bool   // Enable internet mode (TCP)
    AutoMode         bool   // Auto-detect and switch modes
    
    // Datacenter settings
    DatacenterStreams int    // 32-512 streams
    RDMADevice        string // e.g., "mlx5_0"
    RDMAPort          int    // e.g., 1
    
    // Internet settings
    InternetStreams     int    // 4-16 streams
    CongestionAlgorithm string // "bbr" or "cubic"
    PacingEnabled       bool   // Enable packet pacing
    PacingRate          int64  // bytes/second
    
    // Common
    RemoteAddr  string
    MinStreams  int
    MaxStreams  int
}
```

**Default Config:**
```go
func DefaultAMSTv3Config() *AMSTv3Config {
    return &AMSTv3Config{
        EnableDatacenter:    true,
        EnableInternet:      true,
        AutoMode:            true,
        DatacenterStreams:   64,
        InternetStreams:     8,
        CongestionAlgorithm: "bbr",
        PacingEnabled:       true,
        PacingRate:          1000 * 1024 * 1024, // 1 Gbps
        MinStreams:          4,
        MaxStreams:          512,
    }
}
```

---

## Package: `encoding/hde_v3`

### HDEv3

Hierarchical Delta Encoding with ML compression and CRDT sync.

#### Constructor

```go
func NewHDEv3(config *HDEv3Config) (*HDEv3, error)
```

**Example:**
```go
config := encoding.DefaultHDEv3Config("node-1")
config.NetworkMode = upgrade.ModeHybrid
config.EnableMLCompression = true
config.EnableCRDT = true

hde, err := encoding.NewHDEv3(config)
if err != nil {
    log.Fatal(err)
}
```

#### Methods

##### Compress
```go
func (hde *HDEv3) Compress(
    vmID string,
    data []byte,
) (*CompressedDataV3, error)
```

**Example:**
```go
compressed, err := hde.Compress("vm-123", vmState)
if err != nil {
    log.Fatal(err)
}

fmt.Printf("Original: %d bytes\n", compressed.OriginalSize)
fmt.Printf("Compressed: %d bytes\n", compressed.CompressedSize)
fmt.Printf("Ratio: %.2fx\n", compressed.CompressionRatio())
```

##### Decompress
```go
func (hde *HDEv3) Decompress(
    compressed *CompressedDataV3,
) ([]byte, error)
```

**Example:**
```go
original, err := hde.Decompress(compressed)
if err != nil {
    log.Fatal(err)
}
```

#### Configuration

```go
type HDEv3Config struct {
    NodeID string
    
    // Network mode
    NetworkMode upgrade.NetworkMode
    
    // Features
    EnableMLCompression  bool
    EnableDeltaEncoding  bool
    EnableCRDT           bool
    
    // Tuning
    BaselineRefreshInterval time.Duration
    MaxBaselines            int
}
```

---

## Package: `prediction/pba_v3`

### PBAv3

Predictive Bandwidth Allocation with dual LSTM models.

#### Constructor

```go
func NewPBAv3(config *PBAv3Config) (*PBAv3, error)
```

**Example:**
```go
config := prediction.DefaultPBAv3Config()
pba, err := prediction.NewPBAv3(config)
if err != nil {
    log.Fatal(err)
}
```

#### Methods

##### PredictBandwidth
```go
func (p *PBAv3) PredictBandwidth(
    ctx context.Context,
) (*prediction.BandwidthPrediction, error)
```

**Example:**
```go
pred, err := pba.PredictBandwidth(context.Background())
if err != nil {
    log.Fatal(err)
}

fmt.Printf("Predicted bandwidth: %.2f Mbps\n", pred.PredictedBandwidthMbps)
fmt.Printf("Confidence: %.2f%%\n", pred.Confidence*100)
```

##### AddSample
```go
func (p *PBAv3) AddSample(sample prediction.NetworkSample)
```

**Example:**
```go
sample := prediction.NetworkSample{
    Timestamp:    time.Now(),
    BandwidthMbps: 850.5,
    LatencyMs:    12.3,
    PacketLoss:   0.001,
    JitterMs:     2.1,
}
pba.AddSample(sample)
```

---

## Package: `sync/ass_v3`

### ASSv3

Async State Synchronization with Raft/CRDT.

#### Constructor

```go
func NewASSv3(
    nodeID string,
    mode upgrade.NetworkMode,
    logger *zap.Logger,
) (*ASSv3, error)
```

**Example:**
```go
ass, err := sync.NewASSv3("node-1", upgrade.ModeHybrid, logger)
if err != nil {
    log.Fatal(err)
}

err = ass.Start()
```

#### Methods

##### SyncState
```go
func (a *ASSv3) SyncState(
    ctx context.Context,
    state interface{},
) error
```

**Example:**
```go
vmState := &VMState{
    ID:     "vm-123",
    CPU:    4,
    Memory: 8192,
}

err := ass.SyncState(context.Background(), vmState)
if err != nil {
    log.Errorf("Sync failed: %v", err)
}
```

---

## Package: `consensus/acp_v3`

### ACPv3

Adaptive Consensus Protocol with Raft/PBFT.

#### Constructor

```go
func NewACPv3(
    nodeID string,
    mode upgrade.NetworkMode,
    config *ACPConfig,
    logger *zap.Logger,
) (*ACPv3, error)
```

**Example:**
```go
config := &consensus.ACPConfig{
    PBFTConfig: &consensus.PBFTConfig{
        ReplicaCount: 4,
    },
    GossipPeers: []string{"node-2", "node-3"},
}

acp, err := consensus.NewACPv3("node-1", upgrade.ModeHybrid, config, logger)
```

#### Methods

##### Consensus
```go
func (a *ACPv3) Consensus(
    ctx context.Context,
    value interface{},
) error
```

**Example:**
```go
decision := &PlacementDecision{
    VMID:   "vm-123",
    NodeID: "node-5",
}

err := acp.Consensus(context.Background(), decision)
if err != nil {
    log.Errorf("Consensus failed: %v", err)
}
```

---

## Package: `partition/itp_v3`

### ITPv3

Intelligent Task Placement with DQN/Geographic.

#### Constructor

```go
func NewITPv3(mode upgrade.NetworkMode) (*ITPv3, error)
```

**Example:**
```go
itp, err := partition.NewITPv3(upgrade.ModeHybrid)
if err != nil {
    log.Fatal(err)
}
```

#### Methods

##### PlaceVM
```go
func (i *ITPv3) PlaceVM(
    ctx context.Context,
    vm *VM,
    constraints *Constraints,
) (*Node, error)
```

**Example:**
```go
vm := &partition.VM{
    ID:              "vm-123",
    RequestedCPU:    4,
    RequestedMemory: 8 * 1024 * 1024 * 1024, // 8GB
    Priority:        0.9,
}

constraints := &partition.Constraints{
    MaxLatency:   100 * time.Millisecond,
    MinBandwidth: 1000, // Mbps
}

node, err := itp.PlaceVM(context.Background(), vm, constraints)
if err != nil {
    log.Fatal(err)
}

fmt.Printf("VM placed on node: %s\n", node.ID)
```

---

## Package: `upgrade/mode_detector`

### ModeDetector

Automatic network mode detection.

#### Constructor

```go
func NewModeDetector() *ModeDetector
```

**Example:**
```go
detector := upgrade.NewModeDetector()
```

#### Methods

##### DetectMode
```go
func (md *ModeDetector) DetectMode(ctx context.Context) NetworkMode
```

**Example:**
```go
mode := detector.DetectMode(context.Background())
switch mode {
case upgrade.ModeDatacenter:
    fmt.Println("Using datacenter mode (RDMA)")
case upgrade.ModeInternet:
    fmt.Println("Using internet mode (TCP)")
case upgrade.ModeHybrid:
    fmt.Println("Using hybrid mode")
}
```

##### ForceMode
```go
func (md *ModeDetector) ForceMode(mode NetworkMode)
```

**Example:**
```go
// Force datacenter mode for testing
detector.ForceMode(upgrade.ModeDatacenter)
```

---

## Error Codes

| Code | Error | Description |
|------|-------|-------------|
| `ERR_TRANSPORT_UNAVAILABLE` | Transport not available | RDMA or TCP transport not initialized |
| `ERR_MODE_DETECTION_FAILED` | Mode detection failed | Cannot determine network mode |
| `ERR_COMPRESSION_FAILED` | Compression failed | Data compression error |
| `ERR_SYNC_TIMEOUT` | Sync timeout | State synchronization timeout |
| `ERR_CONSENSUS_FAILED` | Consensus failed | Failed to reach consensus |
| `ERR_PLACEMENT_FAILED` | Placement failed | No suitable node found |

**Example Error Handling:**
```go
err := amst.SendData(ctx, data)
if err != nil {
    if errors.Is(err, transport.ErrTransportUnavailable) {
        // Fallback to alternative transport
        log.Warn("Primary transport unavailable, trying fallback")
    } else {
        log.Error("Send failed: %v", err)
    }
}
```

---

## Complete Example: Full v3 System

```go
package main

import (
    "context"
    "log"
    "time"
    
    "github.com/khryptorgraphics/novacron/backend/core/network/dwcp/v3/transport"
    "github.com/khryptorgraphics/novacron/backend/core/network/dwcp/v3/encoding"
    "github.com/khryptorgraphics/novacron/backend/core/network/dwcp/v3/prediction"
    "github.com/khryptorgraphics/novacron/backend/core/network/dwcp/v3/sync"
    "github.com/khryptorgraphics/novacron/backend/core/network/dwcp/v3/consensus"
    "github.com/khryptorgraphics/novacron/backend/core/network/dwcp/v3/partition"
    "github.com/khryptorgraphics/novacron/backend/core/network/dwcp/upgrade"
    "go.uber.org/zap"
)

func main() {
    // Initialize logger
    logger, _ := zap.NewProduction()
    defer logger.Sync()
    
    // 1. Create mode detector
    detector := upgrade.NewModeDetector()
    
    // 2. Initialize AMST v3 (transport)
    amstConfig := transport.DefaultAMSTv3Config()
    amstConfig.AutoMode = true
    amst, err := transport.NewAMSTv3(amstConfig, detector, logger)
    if err != nil {
        log.Fatal(err)
    }
    defer amst.Close()
    
    // 3. Initialize HDE v3 (compression)
    hdeConfig := encoding.DefaultHDEv3Config("node-1")
    hde, err := encoding.NewHDEv3(hdeConfig)
    if err != nil {
        log.Fatal(err)
    }
    defer hde.Close()
    
    // 4. Initialize PBA v3 (prediction)
    pbaConfig := prediction.DefaultPBAv3Config()
    pba, err := prediction.NewPBAv3(pbaConfig)
    if err != nil {
        log.Fatal(err)
    }
    defer pba.Close()
    
    // 5. Initialize ASS v3 (sync)
    ass, err := sync.NewASSv3("node-1", upgrade.ModeHybrid, logger)
    if err != nil {
        log.Fatal(err)
    }
    ass.Start()
    defer ass.Stop()
    
    // 6. Initialize ACP v3 (consensus)
    acpConfig := &consensus.ACPConfig{
        GossipPeers: []string{"node-2", "node-3"},
    }
    acp, err := consensus.NewACPv3("node-1", upgrade.ModeHybrid, acpConfig, logger)
    if err != nil {
        log.Fatal(err)
    }
    acp.Start()
    defer acp.Stop()
    
    // 7. Initialize ITP v3 (placement)
    itp, err := partition.NewITPv3(upgrade.ModeHybrid)
    if err != nil {
        log.Fatal(err)
    }
    
    // 8. Start AMST
    err = amst.Start(context.Background(), "10.0.1.50:8080")
    if err != nil {
        log.Fatal(err)
    }
    
    // 9. Use the system
    vmData := []byte("VM state data")
    
    // Compress
    compressed, err := hde.Compress("vm-123", vmData)
    if err != nil {
        log.Fatal(err)
    }
    
    // Send
    err = amst.SendData(context.Background(), compressed.Marshal())
    if err != nil {
        log.Fatal(err)
    }
    
    // Sync state
    err = ass.SyncState(context.Background(), vmData)
    if err != nil {
        log.Fatal(err)
    }
    
    logger.Info("DWCP v3 system operational",
        zap.String("mode", detector.GetCurrentMode().String()))
}
```

---

## See Also

- Architecture: `/docs/DWCP_V3_ARCHITECTURE.md`
- Operations: `/docs/DWCP_V3_OPERATIONS.md`
- Performance: `/docs/DWCP_V3_PERFORMANCE_TUNING.md`
