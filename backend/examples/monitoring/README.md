# NovaCron Monitoring Examples

This directory contains examples demonstrating the NovaCron monitoring system capabilities, from basic metric collection to advanced analytics.

## Structure

- `basic/`: Contains the basic monitoring example with metric collection and alerting
- `enhanced/`: Contains the enhanced monitoring example with distributed metrics, analytics, and anomaly detection
- `vm-telemetry/`: Contains a specialized example for VM telemetry collection with a real-time dashboard

## Running the Examples

Each example has its own dedicated directory with all necessary files to run independently.

### Basic Monitoring Example

```bash
cd basic
go run main.go
```

This example demonstrates:
- Basic metric registry and collection
- Alert definition and triggering
- Console-based notifications
- CPU and memory monitoring with random generation

### Enhanced Monitoring Example

```bash
cd enhanced
go run main.go
```

This example demonstrates the advanced features:
- Distributed metric collection
- In-memory storage with retention policies
- Advanced alerting with thresholds and durations
- Analytics engine with anomaly detection
- VM-specific metric collection
- Enhanced console output

### VM Telemetry Example

```bash
cd vm-telemetry
go run main.go
```

This example demonstrates specialized VM monitoring features:
- Integration with VM management systems
- Detailed VM telemetry collection (CPU, memory, disk, network)
- Process-level metrics inside VMs
- Live VM monitoring dashboard with ANSI terminal graphics
- Mock VM system that simulates realistic workloads
- Alert generation for VM resource thresholds

## Implementation Details

The examples implement the monitoring system as described in [PHASE3_IMPLEMENTATION.md](../../core/monitoring/PHASE3_IMPLEMENTATION.md) with all the core components:

1. **Collection Layer**: Gathers metrics from various sources
2. **Storage Layer**: Stores metrics in a distributed, fault-tolerant manner
3. **Processing Layer**: Analyzes, aggregates, and processes metrics
4. **Alerting Layer**: Generates alerts based on conditions
5. **Notification Layer**: Delivers alerts through multiple channels
6. **Analytics Layer**: Provides insights and predictions

## Additional Notes

These examples are designed to showcase the monitoring capabilities without requiring actual infrastructure. They generate random metrics with occasional spikes to demonstrate alert triggering.

In a real deployment, the collectors would gather metrics from actual system resources, VMs, and services rather than generating random values.
