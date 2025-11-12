# ITP v3 - Intelligent Task Placement with Mode-Aware Optimization

## Overview

The ITP (Intelligent Task Placement) v3 upgrade enhances the DWCP network's VM placement capabilities with mode-aware placement, geographic optimization, and heterogeneous node support. This implementation provides optimal placement strategies for different network modes: Datacenter, Internet, and Hybrid.

## Key Features

### 1. **Mode-Aware Placement** (`itp_v3.go`)
- **Datacenter Mode**: Optimizes for performance with low latency and high bandwidth
- **Internet Mode**: Optimizes for reliability with geographic distribution
- **Hybrid Mode**: Adaptive switching based on workload characteristics

### 2. **Geographic Optimization** (`geographic_optimizer.go`)
- Cross-region traffic minimization
- Data sovereignty compliance (GDPR, HIPAA)
- Geographic proximity calculations using Haversine formula
- Regulatory zone awareness

### 3. **Heterogeneous Node Support** (`heterogeneous_placement.go`)
- Support for diverse node types:
  - Cloud nodes (elastic, high SLA)
  - Datacenter nodes (high performance, RDMA)
  - Edge nodes (low latency, proximity)
  - Volunteer nodes (cost-effective, variable reliability)
- Capability-based placement (GPU, TPU, SGX, etc.)
- Architecture-specific requirements (x86_64, arm64)

### 4. **DQN-Based Optimization** (`dqn_adapter.go`)
- Standalone DQN-inspired placement algorithm
- Epsilon-greedy exploration strategy
- Q-value approximation for placement decisions
- Bin packing efficiency for datacenter workloads

## Architecture

```
ITPv3
├── Mode Detection (from upgrade package)
├── Placement Engines
│   ├── DQN Placement (Datacenter)
│   ├── Geographic Placement (Internet)
│   └── Hybrid Placement (Adaptive)
├── Optimization Components
│   ├── Geographic Optimizer
│   ├── Heterogeneous Engine
│   └── Resource Allocator
└── Metrics & Monitoring
```

## Usage

### Basic VM Placement

```go
// Create ITPv3 instance with mode
itp, err := NewITPv3(upgrade.ModeInternet)

// Add nodes
node := &Node{
    ID:              "node-1",
    Type:            NodeTypeCloud,
    Region:          "us-west",
    TotalCPU:        16,
    TotalMemory:     32 * 1e9,
    NetworkBandwidth: 10.0,
}
itp.AddNode(node)

// Create VM
vm := &VM{
    ID:              "vm-1",
    RequestedCPU:    4,
    RequestedMemory: 8 * 1e9,
    RequiredRegions: []string{"us-west"},
}

// Place VM
ctx := context.Background()
placedNode, err := itp.PlaceVM(ctx, vm, constraints)
```

### Batch Placement

```go
// Place multiple VMs with global optimization
vms := []*VM{vm1, vm2, vm3}
placements, err := itp.PlaceVMBatch(ctx, vms, constraints)
```

### Geographic Constraints

```go
vm := &VM{
    ID:              "gdpr-vm",
    RequiredRegions: []string{"europe"},
    RequiredLabels: map[string]string{
        "data-sovereignty": "gdpr",
    },
}
```

### Heterogeneous Requirements

```go
vm := &VM{
    ID:           "ml-vm",
    RequestedGPU: 4,
    RequiredLabels: map[string]string{
        "gpu-type":       "nvidia-v100",
        "cpu-arch":       "x86_64",
        "secure-enclave": "true",
    },
}
```

## Performance Characteristics

### Datacenter Mode
- **Placement Latency**: <100ms for 95th percentile
- **Resource Utilization**: 80%+ target
- **Strategy**: Bin packing, performance optimization

### Internet Mode
- **Placement Latency**: <500ms
- **Geographic Distribution**: Cross-region aware
- **Strategy**: Reliability, cost optimization

### Hybrid Mode
- **Adaptive**: Switches based on workload
- **ML Workloads**: Routes to GPU nodes
- **Geographic**: Distributes for fault tolerance

## Metrics

The system tracks:
- Placement success/failure rates
- Resource utilization percentage
- Placement latency (milliseconds)
- Node distribution across regions
- Mode-specific optimizations

## Constraints Support

- **Latency**: Maximum acceptable latency
- **Bandwidth**: Minimum required bandwidth
- **Uptime**: Required availability percentage
- **Cost**: Maximum cost per hour
- **Node Type**: Specific node type requirements
- **Data Locality**: Geographic data sovereignty

## Testing

### Unit Tests
```bash
go test -v ./v3/partition/
```

### Benchmarks
```bash
go test -bench=. -benchmem ./v3/partition/
```

### Test Coverage
- Mode-aware placement scenarios
- Geographic optimization with multiple regions
- Heterogeneous node capabilities
- Resource utilization tracking
- Constraint satisfaction
- Batch placement optimization

## Performance Targets

- **Resource Utilization**: 80%+ across cluster
- **Placement Latency**: <500ms for 99th percentile
- **Geographic Optimization**: Minimize cross-region traffic by 40%
- **Success Rate**: 99.5%+ placement success
- **Scale**: Handle 10,000+ nodes with sub-second planning

## Implementation Notes

1. **Standalone Design**: The v3 implementation is designed to be standalone and doesn't require external ML libraries for the DQN component.

2. **Mode Detection**: Integrates with the DWCP upgrade package's mode detector for automatic mode selection.

3. **Extensibility**: New placement strategies can be added by implementing the `PlacementStrategy` interface.

4. **Thread Safety**: All operations are thread-safe with proper mutex protection.

5. **Error Handling**: Comprehensive error handling with rollback support for batch operations.

## Future Enhancements

- [ ] Machine learning model training for DQN
- [ ] Predictive placement based on historical patterns
- [ ] Energy-aware placement optimization
- [ ] Multi-objective Pareto optimization
- [ ] Real-time migration recommendations
- [ ] Integration with Kubernetes scheduler

## Dependencies

- `github.com/khryptorgraphics/novacron/backend/core/network/dwcp/upgrade` - For network mode detection
- Standard Go libraries only (no external ML dependencies in v3)

## Contributing

When adding new features:
1. Implement mode-specific behavior
2. Add appropriate tests
3. Update benchmarks
4. Document constraints and requirements