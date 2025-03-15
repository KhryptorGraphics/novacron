# VM Telemetry System

The VM Telemetry system is a specialized component of the NovaCron monitoring framework designed to collect detailed metrics from virtual machines. It provides a comprehensive view of VM performance, resource utilization, and health status.

## Architecture

The VM Telemetry system consists of the following components:

1. **VM Telemetry Collector**: Central component that collects metrics from VMs through a VM Manager
2. **VM Manager Interface**: Abstraction that allows integration with different VM providers
3. **Metric Types**: Specialized metric types for VM resources (CPU, memory, disk, network)
4. **Analytics Integration**: Connection to the analytics engine for trend analysis
5. **Alert System**: VM-specific alerting for resource thresholds

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│   VM Manager    │◄────│  VM Telemetry   │────►│  Metrics Store  │
│   (Provider)    │     │   Collector     │     │                 │
│                 │     │                 │     │                 │
└─────────────────┘     └────────┬────────┘     └────────┬────────┘
                                 │                       │
                                 ▼                       ▼
                        ┌─────────────────┐     ┌─────────────────┐
                        │                 │     │                 │
                        │  Alert Manager  │     │ Analytics Engine│
                        │                 │     │                 │
                        └─────────────────┘     └─────────────────┘
```

## Key Features

- **Comprehensive VM Metrics**: Collects CPU, memory, disk, and network metrics
- **Process-Level Metrics**: Visibility into individual processes running in VMs
- **Customizable Detail Level**: Configurable depth of metrics collection
- **Multi-Provider Support**: Abstract VM Manager interface for different hypervisors
- **Low Overhead**: Efficient collection mechanism with batching support
- **Anomaly Detection**: Integration with the analytics engine for pattern detection

## Collected Metrics

### CPU Metrics
- Overall usage percentage
- Per-core usage
- System/user time split
- CPU steal time (hypervisor overhead)
- CPU ready time (wait for physical CPU)
- IO wait time

### Memory Metrics
- Total memory allocation
- Memory usage (percentage and absolute)
- Free memory
- Swap usage
- Page faults
- Memory ballooning metrics

### Disk Metrics
- Disk space usage
- Read/write IOPS
- Read/write throughput
- Read/write latency
- Per-disk metrics for multi-disk VMs

### Network Metrics
- Interface bandwidth usage
- Packet rates
- Error and drop rates
- Per-interface statistics

### Process Metrics
- Process CPU usage
- Process memory usage
- Process I/O operations
- Open file descriptors
- Running time

## Configuration Options

The VM Telemetry Collector can be configured with the following options:

```go
// VMTelemetryCollectorConfig contains configuration for VM telemetry collection
type VMTelemetryCollectorConfig struct {
    // CollectionInterval is how often metrics are collected
    CollectionInterval time.Duration

    // VMManager is used to access VM-specific APIs
    VMManager VMManagerInterface

    // EnabledMetrics configures which metrics to collect
    EnabledMetrics VMMetricTypes

    // Tags are default tags to apply to all metrics
    Tags map[string]string

    // NodeID is the unique identifier for this node
    NodeID string

    // DetailLevel controls the granularity of metrics
    DetailLevel VMMetricDetailLevel
}
```

## Detail Levels

The system supports different levels of metric detail:

- **BasicMetrics**: Only essential health metrics
- **StandardMetrics**: Normal operational metrics (default)
- **DetailedMetrics**: Comprehensive metrics including per-process stats
- **DiagnosticMetrics**: All available metrics for troubleshooting

## Implementing a VM Manager

To integrate with a specific virtualization platform, implement the `VMManagerInterface`:

```go
// VMManagerInterface defines the interface for VM management
type VMManagerInterface interface {
    // GetVMs returns a list of all VM IDs
    GetVMs(ctx context.Context) ([]string, error)

    // GetVMStats retrieves stats for a specific VM
    GetVMStats(ctx context.Context, vmID string, detailLevel VMMetricDetailLevel) (*VMStats, error)
}
```

## Example Usage

```go
// Initialize VM telemetry collector
vmTelemetryConfig := &monitoring.VMTelemetryCollectorConfig{
    CollectionInterval: 30 * time.Second,
    VMManager:          vmManager,
    EnabledMetrics: monitoring.VMMetricTypes{
        CPU:              true,
        Memory:           true,
        Disk:             true,
        Network:          true,
        IOPs:             true,
        ProcessStats:     true,
    },
    NodeID:      "hypervisor-node1",
    DetailLevel: monitoring.StandardMetrics,
}

// Create collector with distributed metric collector
vmTelemetryCollector := monitoring.NewVMTelemetryCollector(vmTelemetryConfig, distributedCollector)

// Start collection
vmTelemetryCollector.Start()
```

## Alerting

Configure VM-specific alerts using standard alert definitions with VM metric names:

```go
// High CPU Alert for VMs
cpuAlert := &monitoring.Alert{
    ID:          "vm-high-cpu-usage",
    Name:        "VM High CPU Usage",
    Description: "VM CPU usage is critically high",
    Severity:    monitoring.AlertSeverityCritical,
    Type:        monitoring.AlertTypeThreshold,
    Condition: monitoring.AlertCondition{
        MetricName: "vm.cpu.usage",
        Operator:   monitoring.AlertConditionOperatorGreaterThan,
        Threshold:  90.0,
        Duration:   1 * time.Minute,
        Tags: map[string]string{
            "component": "vm",
        },
    },
    NotificationChannels: []string{"email", "slack"},
    Enabled:              true,
}
```

## Available VM Metric Names

Here are the metric names you can use for alerting and querying:

### CPU Metrics
- `vm.cpu.usage` - Overall CPU usage percentage
- `vm.cpu.core.usage` - Per-core CPU usage percentage
- `vm.cpu.steal_time` - Hypervisor overhead
- `vm.cpu.ready_time` - CPU scheduling delay
- `vm.cpu.system_time` - CPU time spent in system mode
- `vm.cpu.user_time` - CPU time spent in user mode
- `vm.cpu.iowait_time` - CPU time spent waiting for I/O

### Memory Metrics
- `vm.memory.usage_percent` - Memory usage percentage
- `vm.memory.used` - Used memory in bytes
- `vm.memory.total` - Total memory in bytes
- `vm.memory.free` - Free memory in bytes
- `vm.memory.swap_used` - Swap usage in bytes
- `vm.memory.swap_total` - Total swap in bytes
- `vm.memory.page_faults` - Page faults per second
- `vm.memory.major_page_faults` - Major page faults per second
- `vm.memory.balloon_target` - Ballooning target in bytes
- `vm.memory.balloon_current` - Current balloon size in bytes

### Disk Metrics
- `vm.disk.usage_percent` - Disk usage percentage
- `vm.disk.used` - Used disk space in bytes
- `vm.disk.size` - Total disk size in bytes
- `vm.disk.read_iops` - Disk read operations per second
- `vm.disk.write_iops` - Disk write operations per second
- `vm.disk.read_throughput` - Disk read throughput in bytes per second
- `vm.disk.write_throughput` - Disk write throughput in bytes per second
- `vm.disk.read_latency` - Disk read latency in milliseconds
- `vm.disk.write_latency` - Disk write latency in milliseconds

### Network Metrics
- `vm.network.rx_bytes` - Network receive bandwidth in bytes per second
- `vm.network.tx_bytes` - Network transmit bandwidth in bytes per second
- `vm.network.rx_packets` - Network packets received per second
- `vm.network.tx_packets` - Network packets transmitted per second
- `vm.network.rx_dropped` - Network packets dropped on receive
- `vm.network.tx_dropped` - Network packets dropped on transmit
- `vm.network.rx_errors` - Network receive errors per second
- `vm.network.tx_errors` - Network transmit errors per second

### Process Metrics
- `vm.process.cpu_usage` - Process CPU usage percentage
- `vm.process.memory_usage` - Process memory usage in bytes
- `vm.process.memory_percent` - Process memory usage percentage
- `vm.process.read_iops` - Process read operations per second
- `vm.process.write_iops` - Process write operations per second
- `vm.process.read_throughput` - Process read throughput in bytes per second
- `vm.process.write_throughput` - Process write throughput in bytes per second
- `vm.process.open_files` - Number of open files by the process
- `vm.process.run_time` - Process running time in seconds

## Further Development

Future enhancements for the VM Telemetry system include:

1. **Hypervisor Metrics**: Collecting metrics about the hypervisor itself
2. **Guest Agent Support**: Integration with in-VM agents for more detailed metrics
3. **Cluster-Aware Collection**: Coordinated collection across VM clusters
4. **Resource Correlation**: Correlating VM performance with physical infrastructure
5. **Cross-VM Analysis**: Identifying relationships between VMs
6. **ML-Based Predictions**: Predicting VM resource needs and potential issues
