# ASS/ACP v1 → v3 Upgrade Guide

**DWCP v3 Consensus & State Synchronization Enhancement**

## Overview

This document describes the upgrade of DWCP's Async State Synchronization (ASS) and Adaptive Consensus Protocol (ACP) from v1 to v3, adding mode-aware operation and Byzantine fault tolerance.

## Key Enhancements

### 1. ASS v1 → v3: Mode-Aware State Synchronization

**v1 Limitations:**
- Single CRDT-based approach
- No adaptation to network conditions
- Same latency expectations for all deployments

**v3 Improvements:**
- **Datacenter Mode**: Raft-based strong consistency (<100ms)
- **Internet Mode**: CRDT-based eventual consistency (5-30 seconds)
- **Hybrid Mode**: Adaptive switching with conflict resolution

### 2. ACP v1 → v3: Byzantine-Tolerant Consensus

**v1 Limitations:**
- Basic consensus without Byzantine tolerance
- No protection against malicious nodes
- Not suitable for untrusted internet environments

**v3 Improvements:**
- **PBFT Integration**: Tolerates up to 33% malicious nodes
- **Adaptive Protocol Selection**: Raft for datacenter, PBFT for internet
- **Automatic Failover**: Falls back to PBFT when Raft fails

## Architecture

```
┌─────────────────────────────────────────────────┐
│           DWCP v3 Consensus Layer               │
├─────────────────────────────────────────────────┤
│                                                 │
│  ┌─────────────┐    ┌──────────────┐           │
│  │   ASS v3    │    │   ACP v3     │           │
│  │  Sync Engine│◄──►│ Consensus    │           │
│  └─────────────┘    └──────────────┘           │
│         │                   │                   │
│         ▼                   ▼                   │
│  ┌─────────────────────────────────┐           │
│  │     Mode Detector                │           │
│  │  (Datacenter/Internet/Hybrid)    │           │
│  └─────────────────────────────────┘           │
│         │                   │                   │
│    ┌────┴────┐         ┌────┴────┐             │
│    ▼         ▼         ▼         ▼             │
│  ┌────┐  ┌──────┐  ┌────┐  ┌──────┐           │
│  │Raft│  │ CRDT │  │Raft│  │ PBFT │           │
│  │Sync│  │ Sync │  │    │  │  f=⌊(n-1)/3⌋    │
│  └────┘  └──────┘  └────┘  └──────┘           │
│   Fast    Eventual  Fast   Byzantine-tolerant  │
│  <100ms   5-30s    <100ms    1-5s              │
└─────────────────────────────────────────────────┘
```

## Implementation Details

### PBFT (Practical Byzantine Fault Tolerance)

**File:** `/backend/core/network/dwcp/v3/consensus/pbft.go`

**Key Features:**
- 3-phase commit protocol (pre-prepare, prepare, commit)
- Byzantine tolerance: f = ⌊(n-1)/3⌋ malicious nodes
- Quorum requirements:
  - Prepare: 2f matching messages
  - Commit: 2f+1 matching messages
- Periodic checkpointing for garbage collection
- View change protocol for liveness

**Example Usage:**
```go
// Create PBFT with 7 replicas (tolerates 2 Byzantine nodes)
pbft, err := NewPBFT(
    "node_0",           // Node ID
    7,                  // Replica count (3f+1 = 7, f=2)
    transport,          // Network transport
    stateMachine,       // State machine
    logger,
)

// Start consensus
err = pbft.Start()

// Execute consensus on a value
ctx := context.Background()
value := map[string]interface{}{"key": "data"}
err = pbft.Consensus(ctx, value)

// Get metrics
metrics := pbft.GetMetrics()
fmt.Printf("Byzantine tolerance: %d nodes\n", metrics.ByzantineTolerance)
```

### ASS v3 State Synchronization

**File:** `/backend/core/network/dwcp/v3/sync/ass_v3.go`

**Key Features:**
- Mode-aware sync strategy selection
- Automatic mode detection and adaptation
- Conflict resolution for hybrid mode
- Performance metrics tracking

**Example Usage:**
```go
// Create ASS v3 in hybrid mode
ass, err := NewASSv3(
    "node_0",
    upgrade.ModeHybrid,
    logger,
)

// Set Raft node for datacenter fallback
ass.SetRaftNode(raftNode)

// Start adaptive mode loop
err = ass.Start()

// Synchronize state (automatically selects strategy)
ctx := context.Background()
state := map[string]interface{}{
    "key1": "value1",
    "key2": 42,
}
err = ass.SyncState(ctx, state)

// Get sync metrics
metrics := ass.GetMetrics()
fmt.Printf("Mode: %s, Avg Latency: %v\n",
    metrics.Mode, metrics.AvgSyncLatency)
```

### ACP v3 Adaptive Consensus

**File:** `/backend/core/network/dwcp/v3/consensus/acp_v3.go`

**Key Features:**
- Automatic protocol selection based on network mode
- Seamless failover from Raft to PBFT
- Performance tracking and metrics
- Health monitoring

**Example Usage:**
```go
// Create ACP v3 configuration
config := &ACPConfig{
    PBFTConfig: &PBFTConfig{
        ReplicaCount: 7,
        Transport:    transport,
        StateMachine: stateMachine,
    },
    GossipPeers: []string{"peer1", "peer2"},
}

// Create ACP v3 in hybrid mode
acp, err := NewACPv3(
    "node_0",
    upgrade.ModeHybrid,
    config,
    logger,
)

// Set Raft node
acp.SetRaftNode(raftNode)

// Start consensus engine
err = acp.Start()

// Execute consensus (automatically adapts)
ctx := context.Background()
value := map[string]interface{}{"operation": "write"}
err = acp.Consensus(ctx, value)

// Check health
healthy := acp.IsHealthy()

// Get expected latency
latency := acp.GetConsensusLatency()
```

## Performance Targets

### Datacenter Mode
- **Consensus Latency**: <100ms (Raft)
- **Sync Latency**: <100ms (Raft)
- **Throughput**: 10,000+ ops/sec
- **Assumptions**: Low latency (<10ms), high bandwidth (>1 Gbps)

### Internet Mode
- **Consensus Latency**: 1-5 seconds (PBFT)
- **Sync Latency**: 5-30 seconds (CRDT)
- **Throughput**: 100-1,000 ops/sec
- **Assumptions**: High latency (50-500ms), lower bandwidth
- **Byzantine Tolerance**: 33% malicious nodes

### Hybrid Mode
- **Consensus Latency**: 100ms-5s (adaptive)
- **Sync Latency**: 100ms-30s (adaptive)
- **Throughput**: 1,000-10,000 ops/sec
- **Behavior**: Tries Raft first, falls back to PBFT

## Migration Guide

### Step 1: Update Imports

```go
// Old v1 imports
import (
    "github.com/khryptorgraphics/novacron/backend/core/network/dwcp/sync"
    "github.com/khryptorgraphics/novacron/backend/core/network/dwcp/consensus"
)

// New v3 imports
import (
    syncv3 "github.com/khryptorgraphics/novacron/backend/core/network/dwcp/v3/sync"
    consensusv3 "github.com/khryptorgraphics/novacron/backend/core/network/dwcp/v3/consensus"
    "github.com/khryptorgraphics/novacron/backend/core/network/dwcp/upgrade"
)
```

### Step 2: Replace ASS Engine

```go
// Old v1 ASS
assEngine := sync.NewASSEngine(nodeID, transport, logger)
assEngine.Start()

// New v3 ASS
mode := upgrade.ModeHybrid // or ModeDatacenter/ModeInternet
assEngine, err := syncv3.NewASSv3(nodeID, mode, logger)
if err != nil {
    log.Fatal(err)
}
assEngine.SetRaftNode(raftNode) // If using Raft
assEngine.Start()
```

### Step 3: Replace Consensus Protocol

```go
// Old v1 consensus
// (basic consensus without Byzantine tolerance)

// New v3 ACP with PBFT
config := &consensusv3.ACPConfig{
    PBFTConfig: &consensusv3.PBFTConfig{
        ReplicaCount: 7,
        Transport:    pbftTransport,
        StateMachine: stateMachine,
    },
    GossipPeers: peers,
}

acp, err := consensusv3.NewACPv3(nodeID, mode, config, logger)
if err != nil {
    log.Fatal(err)
}
acp.SetRaftNode(raftNode)
acp.Start()
```

### Step 4: Update Sync Operations

```go
// Old v1 sync
err := assEngine.Set(key, crdtValue)

// New v3 sync
ctx := context.Background()
state := map[string]interface{}{
    "key": "value",
}
err := assEngine.SyncState(ctx, state)
```

### Step 5: Update Consensus Operations

```go
// Old v1 consensus
// (basic proposal)

// New v3 consensus
ctx := context.Background()
value := map[string]interface{}{
    "operation": "write",
    "data":      data,
}
err := acp.Consensus(ctx, value)
```

## Testing

### Run All Tests

```bash
cd /home/kp/novacron/backend/core/network/dwcp

# Run v3 sync tests
go test -v ./v3/sync/

# Run v3 consensus tests
go test -v ./v3/consensus/

# Run with race detector
go test -v -race ./v3/sync/ ./v3/consensus/

# Run benchmarks
go test -bench=. ./v3/sync/ ./v3/consensus/
```

### Test Results

**Sync Tests:**
```
PASS: TestASSv3_Creation
PASS: TestASSv3_DatacenterSync (10ms latency)
PASS: TestASSv3_InternetSync (CRDT)
PASS: TestASSv3_HybridSync (adaptive)
PASS: TestASSv3_ModeAdaptation
PASS: TestASSv3_ConcurrentSync
```

**Consensus Tests:**
```
PASS: TestPBFT_Creation (4, 7, 10 replicas)
PASS: TestPBFT_ByzantineTolerance (25-30% tolerance)
PASS: TestPBFT_ThreePhaseProtocol
PASS: TestPBFT_Checkpoint
PASS: TestACPv3_HybridConsensusFailover
PASS: TestACPv3_HealthCheck
```

## Monitoring & Metrics

### ASS v3 Metrics

```go
metrics := assEngine.GetMetrics()

// Available metrics:
// - Mode: Current sync mode (datacenter/internet/hybrid)
// - SyncCount: Total sync operations
// - LastSyncTime: Timestamp of last sync
// - AvgSyncLatency: Average sync latency
// - ConflictCount: Number of conflicts resolved
```

### ACP v3 Metrics

```go
metrics := acp.GetMetrics()

// Available metrics:
// - Mode: Current consensus mode
// - ConsensusCount: Total consensus operations
// - LastConsensusTime: Timestamp of last consensus
// - AvgConsensusTime: Average consensus latency
// - FailoverCount: Number of Raft→PBFT failovers
// - PBFTMetrics: Detailed PBFT statistics
```

### PBFT Metrics

```go
pbftMetrics := pbft.GetMetrics()

// Available metrics:
// - View: Current PBFT view number
// - CheckpointSeq: Last checkpoint sequence
// - StableCheckpoint: Stable checkpoint sequence
// - ExecutedRequests: Total executed requests
// - PendingRequests: Requests in queue
// - ByzantineTolerance: f (number of tolerated Byzantine nodes)
// - ReplicaCount: Total replicas
// - IsPrimary: Whether this node is primary
```

## Troubleshooting

### Issue: Tests Failing with "Raft node not initialized"

**Solution:**
```go
// Always set Raft node before using datacenter mode
ass.SetRaftNode(raftNode)
acp.SetRaftNode(raftNode)
```

### Issue: PBFT Consensus Slow

**Possible Causes:**
1. Insufficient replicas (need 3f+1)
2. High network latency
3. Byzantine node attacks

**Solution:**
```go
// Check PBFT metrics
metrics := pbft.GetMetrics()
fmt.Printf("Replicas: %d, f: %d\n",
    metrics.ReplicaCount,
    metrics.ByzantineTolerance)

// Ensure minimum 4 replicas
if metrics.ReplicaCount < 4 {
    // Add more replicas
}
```

### Issue: Frequent Mode Switching in Hybrid Mode

**Solution:**
```go
// Force a specific mode if needed
ass.SetMode(upgrade.ModeDatacenter)
acp.SetMode(upgrade.ModeDatacenter)

// Or adjust mode detector thresholds
detector := upgrade.NewModeDetector()
// Custom configuration
```

## Integration with Existing DWCP

### Backward Compatibility

v3 implementations are in separate packages and can coexist with v1:

```go
// v1 and v3 can run side-by-side during migration
v1Sync := sync.NewASSEngine(nodeID, transport, logger)
v3Sync, _ := syncv3.NewASSv3(nodeID, mode, logger)

// Gradual migration: sync to both
v1Sync.Set(key, value)
v3Sync.SyncState(ctx, state)
```

### Complete Integration Example

```go
package main

import (
    "context"
    "log"

    syncv3 "github.com/khryptorgraphics/novacron/backend/core/network/dwcp/v3/sync"
    consensusv3 "github.com/khryptorgraphics/novacron/backend/core/network/dwcp/v3/consensus"
    "github.com/khryptorgraphics/novacron/backend/core/network/dwcp/upgrade"
    "go.uber.org/zap"
)

func main() {
    logger, _ := zap.NewProduction()
    nodeID := "node_0"
    mode := upgrade.ModeHybrid

    // Create ASS v3
    ass, err := syncv3.NewASSv3(nodeID, mode, logger)
    if err != nil {
        log.Fatal(err)
    }

    // Create ACP v3
    config := &consensusv3.ACPConfig{
        PBFTConfig: &consensusv3.PBFTConfig{
            ReplicaCount: 7,
            Transport:    pbftTransport,
            StateMachine: stateMachine,
        },
        GossipPeers: []string{"peer1", "peer2"},
    }

    acp, err := consensusv3.NewACPv3(nodeID, mode, config, logger)
    if err != nil {
        log.Fatal(err)
    }

    // Set Raft nodes
    ass.SetRaftNode(raftNode)
    acp.SetRaftNode(raftNode)

    // Start engines
    ass.Start()
    acp.Start()

    // Use for state sync and consensus
    ctx := context.Background()

    // Sync state
    state := map[string]interface{}{"key": "value"}
    err = ass.SyncState(ctx, state)

    // Consensus
    value := map[string]interface{}{"op": "write"}
    err = acp.Consensus(ctx, value)

    // Monitor health
    if !acp.IsHealthy() {
        logger.Warn("Consensus unhealthy")
    }

    // Get metrics
    syncMetrics := ass.GetMetrics()
    consensusMetrics := acp.GetMetrics()

    logger.Info("Metrics",
        zap.String("sync_mode", syncMetrics.Mode),
        zap.Duration("avg_sync", syncMetrics.AvgSyncLatency),
        zap.String("consensus_mode", consensusMetrics.Mode),
        zap.Duration("avg_consensus", consensusMetrics.AvgConsensusTime),
    )
}
```

## Next Steps

1. **Deploy in Datacenter Mode** for trusted environments
2. **Deploy in Internet Mode** for untrusted, wide-area deployments
3. **Use Hybrid Mode** for mixed environments
4. **Monitor Metrics** to optimize performance
5. **Tune PBFT Parameters** based on Byzantine threat model

## References

- PBFT Paper: "Practical Byzantine Fault Tolerance" (Castro & Liskov, 1999)
- Raft Paper: "In Search of an Understandable Consensus Algorithm" (Ongaro & Ousterhout, 2014)
- CRDT Survey: "A comprehensive study of CRDTs" (Shapiro et al., 2011)

## Support

For issues or questions:
- File bugs: GitHub Issues
- Documentation: `/docs/DWCP-*.md`
- Tests: `/backend/core/network/dwcp/v3/*/`
