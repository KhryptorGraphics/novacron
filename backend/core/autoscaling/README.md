# Auto-Scaling System

The Auto-Scaling System provides dynamic resource scaling capabilities based on workload demands, metrics, and predictive patterns. It enables NovaCron to automatically adjust resource capacity in response to changing conditions, optimizing both performance and cost efficiency.

## Architecture

The auto-scaling architecture consists of the following components:

1. **AutoScaling Manager**: Core component that orchestrates scaling operations
2. **Scaling Groups**: Collections of resources that scale together
3. **Scaling Rules**: Definitions for when and how scaling should occur
4. **Metrics Provider**: Source of metrics for scaling decisions
5. **Resource Controller**: Interface to control resources during scaling operations
6. **Predictive Scaling**: Optional component for forecasting future capacity needs

## Scaling Modes

The system supports multiple scaling modes:

1. **Horizontal Scaling (Scale Out/In)**
   - Adds or removes instances to handle workload changes
   - Maintains performance during load increases
   - Reduces costs during periods of low demand
   - Useful for stateless applications or those with shared state stores

2. **Vertical Scaling (Scale Up/Down)**
   - Changes the resources allocated to existing instances
   - Increases CPU, memory, or other resources for existing VMs
   - May require instance restarts depending on hypervisor capabilities
   - Useful for stateful applications or database workloads

3. **Mixed Scaling**
   - Combines horizontal and vertical scaling strategies
   - Optimizes for both resource utilization and cost
   - Uses sophisticated algorithms to determine best scaling approach
   - Provides maximum flexibility for diverse workloads

## Key Features

### Metric-Based Scaling

- Support for diverse metric types:
  - CPU and memory utilization
  - Request counts and queue lengths
  - Response times and latency
  - Custom application metrics
- Multi-dimensional metric evaluation
- Threshold-based scaling triggers
- Smoothing and dampening to prevent oscillation

### Policy-Based Control

- Flexible rule definitions
- Customizable thresholds and scaling increments
- Cooldown periods to prevent scaling thrashing
- Multi-stage evaluation with priority ordering
- Resource boundary enforcement (min/max capacities)

### Predictive Scaling

- Forecast-based capacity planning
- Historical trend analysis
- Time-based scaling schedules
- Machine learning algorithms for prediction
- Confidence-based scaling decisions
- Capacity optimization before demand occurs

### Event Tracking and Analytics

- Detailed event history
- Scaling action auditing
- Performance analytics and metrics
- Cost impact analysis
- Scaling efficiency reporting

## Using the Auto-Scaling System

### Creating a Scaling Group

```go
// Create required components
metricsProvider := metrics.NewPrometheusMetricsProvider("http://prometheus:9090")
resourceController := vm.NewVMResourceController()

// Create auto-scaling manager
scalingManager := autoscaling.NewAutoScalingManager(metricsProvider, resourceController)

// Initialize the manager
ctx := context.Background()
scalingManager.Initialize(ctx)

// Create a scaling group
group := &autoscaling.ScalingGroup{
    ID:             "web-tier",
    Name:           "Web Server Tier",
    ResourceType:   autoscaling.ResourceVM,
    ResourceIDs:    []string{"web-server-1", "web-server-2"},
    ScalingMode:    autoscaling.ScalingModeHorizontal,
    MinCapacity:    2,
    MaxCapacity:    10,
    DesiredCapacity: 2,
    CurrentCapacity: 2,
    Rules: []*autoscaling.ScalingRule{
        {
            ID:                "cpu-utilization-rule",
            Name:              "Scale based on CPU utilization",
            MetricType:        autoscaling.MetricCPUUtilization,
            ScaleOutThreshold: 75.0,  // 75% CPU utilization
            ScaleInThreshold:  25.0,  // 25% CPU utilization
            ScaleOutIncrement: 2,     // Add 2 instances
            ScaleInDecrement:  1,     // Remove 1 instance
            CooldownPeriod:    5 * time.Minute,
            EvaluationPeriods: 3,     // Consider 3 consecutive periods
            Enabled:           true,
        },
    },
    LaunchTemplate: map[string]interface{}{
        "image_id":      "ami-12345678",
        "instance_type": "t3.medium",
        "network_id":    "net-0987654321",
    },
}

// Register the group
scalingManager.RegisterScalingGroup(group)

// Start the scaling loop (evaluate every minute)
scalingManager.StartScalingLoop(1 * time.Minute)
```

### Creating Predictive Scaling

```go
// Enable predictive scaling
scalingManager.EnablePredictiveScaling(autoscaling.PredictiveScalingConfig{
    ForecastHorizon: 24 * time.Hour,    // Look ahead 24 hours
    HistoryWindow:   7 * 24 * time.Hour, // Use 7 days of history
    MinConfidence:   0.8,               // 80% confidence required
    Algorithm:       "exponential_smoothing",
    Schedule: map[string]interface{}{
        "business_hours": map[string]interface{}{
            "days":  []string{"monday", "tuesday", "wednesday", "thursday", "friday"},
            "start": "08:00",
            "end":   "18:00",
            "capacity": 8,
        },
    },
})
```

### Manual Scaling Operations

```go
// Manually set capacity for a group
scalingManager.SetGroupCapacity(ctx, "web-tier", 6)

// Get current group information
group, err := scalingManager.GetScalingGroup("web-tier")

// List all recent scaling events
events := scalingManager.GetScalingEvents(10)
```

## Integration with NovaCron

The Auto-Scaling System integrates with other NovaCron components:

1. **VM Management**: Scale virtual machines and containers
2. **Storage Subsystem**: Scale storage resources based on demand
3. **Monitoring System**: Use monitoring metrics for scaling decisions
4. **Federation**: Scale resources across multiple clusters
5. **Network Overlay**: Configure networking for scaled resources

## Implementation Details

- Thread-safe design with RWMutex protection
- Event-driven architecture
- Background worker for continuous evaluation
- Context-based operations supporting cancellation
- Flexible plugin architecture for metrics and resource controllers
- Comprehensive logging and event tracking
