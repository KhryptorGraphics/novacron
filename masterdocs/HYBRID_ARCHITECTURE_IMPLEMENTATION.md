# Hybrid Architecture Implementation

## Overview

The Hybrid Architecture enables automatic switching between datacenter-centric and distributed global internet supercomputer infrastructure modes based on real-time network conditions.

## Architecture Components

### 1. HybridOrchestrator
- **Purpose**: Manages automatic mode detection and switching
- **Features**:
  - Continuous network condition monitoring
  - Automatic mode detection based on latency/bandwidth
  - Cooldown period to prevent flapping
  - Graceful transitions with drain timeout
  - Mode change callbacks for component updates

### 2. ModeAwareAdapter
- **Purpose**: Adapts between v1 and v3 component implementations
- **Features**:
  - Registers both v1 and v3 implementations
  - Switches active components based on mode
  - Validates component implementations
  - Provides component lookup

### 3. HybridManager
- **Purpose**: Coordinates hybrid architecture
- **Features**:
  - Initializes and manages orchestrator and adapter
  - Registers components
  - Tracks metrics and statistics
  - Provides unified interface

## Operation Modes

### Datacenter Mode
- **Target**: Low-latency (<10ms), high-bandwidth (>1 Gbps)
- **Components**: v1 (optimized for performance)
- **Consensus**: Raft
- **Transport**: RDMA
- **Compression**: LZ4 (speed-optimized)

### Internet Mode
- **Target**: High-latency (>50ms), variable bandwidth
- **Components**: v3 (Byzantine-tolerant)
- **Consensus**: PBFT
- **Transport**: TCP with BBR
- **Compression**: zstd-max (size-optimized)

### Hybrid Mode
- **Target**: Borderline conditions
- **Components**: v3 (adaptive)
- **Consensus**: Adaptive (Raft→PBFT fallback)
- **Transport**: Adaptive (RDMA→TCP fallback)

## Configuration

```go
config := &hybrid.HybridConfig{
    Enabled:                      true,
    AutoDetect:                   true,
    DetectionInterval:            10 * time.Second,
    DatacenterLatencyThreshold:   10 * time.Millisecond,
    DatacenterBandwidthThreshold: 1e9, // 1 Gbps
    InternetLatencyThreshold:     50 * time.Millisecond,
    InternetBandwidthThreshold:   1e9, // 1 Gbps
    Hysteresis:                   0.1,
    CooldownPeriod:               30 * time.Second,
    GracefulTransition:           true,
    DrainTimeout:                 10 * time.Second,
}
```

## Usage Example

```go
// Create hybrid manager
hm := hybrid.NewHybridManager(logger, config)

// Initialize
if err := hm.Initialize(ctx); err != nil {
    return err
}

// Register components
hm.RegisterComponent("transport", v1Transport, v3Transport)
hm.RegisterComponent("consensus", v1Consensus, v3Consensus)
hm.RegisterComponent("compression", v1Compression, v3Compression)

// Start automatic mode switching
if err := hm.Start(ctx); err != nil {
    return err
}

// Get active component
transport, err := hm.GetComponent("transport")
if err != nil {
    return err
}

// Use component
v3Transport := transport.(TransportV3)
v3Transport.Send(data)

// Get statistics
stats := hm.GetStats()
fmt.Printf("Current mode: %v\n", stats["current_mode"])
fmt.Printf("Mode changes: %v\n", stats["mode_changes"])
```

## Mode Switching Algorithm

1. **Detection**: Measure latency and bandwidth every 10 seconds
2. **Analysis**: Compare against thresholds with hysteresis
3. **Decision**: Determine optimal mode
4. **Cooldown**: Wait 30 seconds before next switch
5. **Transition**: Gracefully switch components (10s drain timeout)
6. **Callback**: Notify registered callbacks of mode change

## Integration Points

### DWCP Manager
- Register DWCP components with hybrid manager
- Use hybrid manager to get active components
- Listen for mode changes to update configuration

### Feature Flags
- `EnableHybridMode`: Enable/disable hybrid mode
- `EnableModeDetection`: Enable/disable auto-detection
- `V3RolloutPercentage`: Gradual rollout control

### Monitoring
- Track mode changes and reasons
- Monitor successful/failed switches
- Alert on excessive flapping

## Performance Characteristics

| Metric | Datacenter | Internet | Hybrid |
|--------|-----------|----------|--------|
| Latency | <10ms | 50-500ms | Adaptive |
| Bandwidth | >1 Gbps | 100-900 Mbps | Adaptive |
| Consensus | Raft (100ms) | PBFT (5-30s) | Adaptive |
| Throughput | High | Medium | Adaptive |
| Byzantine Tolerance | No | Yes (33%) | Adaptive |

## Future Enhancements

1. **ML-based Mode Selection**: Use neural networks for optimal mode prediction
2. **Per-Connection Mode**: Select mode per connection instead of globally
3. **Gradual Transitions**: Migrate connections gradually during mode switches
4. **Predictive Switching**: Switch before conditions degrade
5. **Multi-Mode Ensemble**: Run multiple modes in parallel for comparison

