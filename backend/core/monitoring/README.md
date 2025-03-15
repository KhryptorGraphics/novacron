# Advanced Monitoring and Alerting System

This package provides a comprehensive monitoring and alerting system for NovaCron. It allows for collecting, storing, analyzing, and alerting on metrics from various sources.

## Features

- **Metrics Collection**: Collect metrics from different sources (system, VMs, network, storage)
- **Metric Types**: Support for counters, gauges, histograms, timers, and state metrics
- **Alerting**: Define alert conditions based on metric thresholds and patterns
- **Notifications**: Send alerts through multiple channels (email, webhook, Slack, etc.)
- **Historical Data**: Store and analyze historical metric data
- **Trend Analysis**: Analyze metric trends and predict future values
- **Multi-tenant Support**: Metrics and alerts can be tenant-specific

## Architecture

The monitoring system consists of the following components:

1. **Metric Registry**: Central registry for all metrics
2. **Metric Collectors**: Collect metrics from various sources
3. **Alert Manager**: Evaluates alert conditions and triggers alerts
4. **Notification Manager**: Manages notification channels and templates
5. **History Manager**: Manages historical metric data and cleanup

## Usage Examples

### Basic Metric Collection

```go
// Create metric registry
registry := monitoring.NewMetricRegistry()

// Create a gauge metric
cpuUsage := monitoring.NewGaugeMetric(
    "system.cpu.usage", 
    "CPU Usage", 
    "Percentage of CPU usage",
    "system",
)
cpuUsage.SetUnit("percent")

// Register the metric
registry.RegisterMetric(cpuUsage)

// Record a value
cpuUsage.RecordValue(45.6, nil)

// Get the latest value
value := cpuUsage.GetLastValue()
fmt.Printf("Current CPU usage: %.2f%%\n", value.Value)
```

### Creating and Using a Collector

```go
// Create metric registry
registry := monitoring.NewMetricRegistry()

// Create system collector with 5-second interval
collector := monitoring.NewSystemCollector(registry, 5*time.Second)

// Start the collector
collector.Start()

// Get all metrics provided by the collector
metrics := collector.GetMetrics()
for _, metric := range metrics {
    fmt.Printf("Metric: %s (%s)\n", metric.Name, metric.Description)
}

// Manually trigger a collection (normally automatic)
batches, err := collector.Collect()
if err != nil {
    fmt.Printf("Error collecting metrics: %v\n", err)
}

// Stop collector when done
collector.Stop()
```

### Setting Up Alerting

```go
// Create alert registry
alertRegistry := monitoring.NewAlertRegistry()

// Create a CPU usage alert
cpuAlert := monitoring.NewAlert(
    "cpu-usage-alert",
    "High CPU Usage",
    "Alert when CPU usage exceeds 80%",
    monitoring.AlertSeverityHigh,
    monitoring.AlertCondition{
        Type:     monitoring.ThresholdCondition,
        MetricID: "system.cpu.usage",
        Operator: monitoring.GreaterThanOrEqual,
        Threshold: func() *float64 {
            val := 80.0
            return &val
        }(),
        Period: func() *time.Duration {
            period := 30 * time.Second
            return &period
        }(),
    },
)

// Register the alert
alertRegistry.RegisterAlert(cpuAlert)

// Create alert manager
alertManager := monitoring.NewAlertManager(alertRegistry, registry, 5*time.Second)
alertManager.Start()
```

### Setting Up Notifications

```go
// Create notification manager
notificationManager := monitoring.NewNotificationManager()

// Add default email template
emailTemplate := monitoring.DefaultEmailTemplate()
notificationManager.AddTemplate(emailTemplate)

// Configure email notification
emailConfig := &monitoring.NotificationConfig{
    ID:      "email-config",
    Name:    "Email Notifications",
    Channel: monitoring.EmailChannel,
    Enabled: true,
    Settings: map[string]interface{}{
        "server":      "smtp.example.com",
        "port":        587,
        "username":    "alerts@example.com",
        "password":    "password",
        "fromAddress": "alerts@example.com",
        "toAddresses": []string{"admin@example.com"},
        "useTLS":      true,
    },
}
notificationManager.AddConfig(emailConfig)
notificationManager.CreateEmailNotifier("email-config")

// Send notification
notificationManager.SendNotification(alert, "default-email-template")
```

### Historical Metrics and Analysis

```go
// Create metrics history manager
historyManager := monitoring.NewMetricHistoryManager(
    registry, 
    24*time.Hour, // Data retention
    1*time.Hour,  // Cleanup interval
)
historyManager.Start()

// Get historical values
start := time.Now().Add(-1 * time.Hour)
end := time.Now()
values, err := historyManager.GetHistoricalValues("system.cpu.usage", start, end)
if err != nil {
    fmt.Printf("Error getting historical values: %v\n", err)
}

// Analyze trend
slope, err := historyManager.AnalyzeMetricTrend("system.cpu.usage", 1*time.Hour)
if err != nil {
    fmt.Printf("Error analyzing trend: %v\n", err)
}
fmt.Printf("CPU usage trend: %.2f%%/s\n", slope)

// Predict future value
future := time.Now().Add(30 * time.Minute)
predictedValue, err := historyManager.PredictMetricValue("system.cpu.usage", future)
if err != nil {
    fmt.Printf("Error predicting value: %v\n", err)
}
fmt.Printf("Predicted CPU usage in 30 minutes: %.2f%%\n", predictedValue)
```

## Integration Points

The monitoring system integrates with other NovaCron components:

- **VM Manager**: Collects VM performance metrics
- **Network Manager**: Collects network metrics
- **Storage Manager**: Collects storage metrics
- **Multi-tenant Architecture**: Filters metrics and alerts by tenant
- **Authentication System**: Controls access to metrics and alerts

## Future Enhancements

- **Visual Dashboard**: Provide real-time visualization of metrics and alerts
- **Custom Query Language**: Allow for complex metric queries and aggregations
- **Machine Learning**: Improve anomaly detection and prediction accuracy
- **Event Correlation**: Correlate metrics and events across components
- **Auto-scaling Triggers**: Use metrics to automatically adjust resources
