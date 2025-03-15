package main

import (
	"context"
	"fmt"
	"math/rand"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/monitoring"
	"github.com/khryptorgraphics/novacron/backend/core/storage"
)

func main() {
	fmt.Println("NovaCron Enhanced Monitoring System Demo")
	fmt.Println("========================================")

	// Initialize storage for metrics
	metricStorage := storage.NewInMemoryStorage()

	// Initialize metric collector with storage
	collectorConfig := monitoring.DefaultDistributedMetricCollectorConfig()
	collectorConfig.CollectionInterval = 5 * time.Second
	collectorConfig.NodeID = "example-node-1"
	collectorConfig.ClusterID = "example-cluster"
	collectorConfig.Tags = map[string]string{
		"environment": "development",
		"service":     "monitoring-demo",
	}

	// Create metric collector with storage
	collector := monitoring.NewDistributedMetricCollector(collectorConfig, metricStorage)

	// Set up custom collectors
	systemCollector := NewSystemStatsCollector("system-stats")
	vmCollector := NewVMStatsCollector("vm-stats")
	collector.AddCollector(systemCollector)
	collector.AddCollector(vmCollector)

	// Set up analytics engine
	analyticsConfig := monitoring.DefaultAnalyticsEngineConfig()
	analyticsConfig.ProcessingInterval = 15 * time.Second
	analytics := monitoring.NewAnalyticsEngine(analyticsConfig, collector)

	// Add some custom notification channels
	consoleChannel := monitoring.NewConsoleChannel("console-alerts")
	collector.AlertManager().GetNotificationManager().RegisterChannel(consoleChannel)

	// Configure alerts
	setupAlerts(collector)

	// Start all components
	fmt.Println("Starting metric collection and analytics...")
	collector.Start()
	analytics.Start()

	// Print status
	fmt.Println("\nDemo is running with the following components:")
	fmt.Println("- Distributed Metric Collector")
	fmt.Println("- System Stats Collector")
	fmt.Println("- VM Stats Collector")
	fmt.Println("- Alert Manager with Console Notifications")
	fmt.Println("- Analytics Engine with Anomaly Detection")
	fmt.Println("\nMetrics will be collected every 5 seconds.")
	fmt.Println("Alerts will be triggered when metrics exceed thresholds.")
	fmt.Println("Analytics will process data every 15 seconds.")
	fmt.Println("\nPress Ctrl+C to stop the demo.")

	// Wait for termination signal
	termChan := make(chan os.Signal, 1)
	signal.Notify(termChan, syscall.SIGINT, syscall.SIGTERM)
	<-termChan

	// Graceful shutdown
	fmt.Println("\nShutting down...")
	analytics.Stop()
	collector.Stop()
	fmt.Println("Demo stopped.")
}

func setupAlerts(collector *monitoring.DistributedMetricCollector) {
	// High CPU Alert
	cpuAlert := &monitoring.Alert{
		ID:          "high-cpu-usage",
		Name:        "High CPU Usage",
		Description: "CPU usage is too high",
		Severity:    monitoring.AlertSeverityCritical,
		Type:        monitoring.AlertTypeThreshold,
		Condition: monitoring.AlertCondition{
			MetricName: "system.cpu.usage",
			Operator:   monitoring.AlertConditionOperatorGreaterThan,
			Threshold:  80.0,
			Duration:   1 * time.Minute,
			Tags: map[string]string{
				"component": "system",
			},
		},
		NotificationChannels: []string{"console-alerts"},
		Enabled:              true,
		Status:               monitoring.AlertStatusResolved,
	}
	collector.RegisterAlert(cpuAlert)

	// High Memory Alert
	memoryAlert := &monitoring.Alert{
		ID:          "high-memory-usage",
		Name:        "High Memory Usage",
		Description: "Memory usage is too high",
		Severity:    monitoring.AlertSeverityWarning,
		Type:        monitoring.AlertTypeThreshold,
		Condition: monitoring.AlertCondition{
			MetricName: "system.memory.usage",
			Operator:   monitoring.AlertConditionOperatorGreaterThanOrEqual,
			Threshold:  90.0,
			Duration:   30 * time.Second,
			Tags: map[string]string{
				"component": "system",
			},
		},
		NotificationChannels: []string{"console-alerts"},
		Enabled:              true,
		Status:               monitoring.AlertStatusResolved,
	}
	collector.RegisterAlert(memoryAlert)

	// High Disk Usage Alert
	diskAlert := &monitoring.Alert{
		ID:          "high-disk-usage",
		Name:        "High Disk Usage",
		Description: "Disk usage is too high",
		Severity:    monitoring.AlertSeverityError,
		Type:        monitoring.AlertTypeThreshold,
		Condition: monitoring.AlertCondition{
			MetricName: "system.disk.usage",
			Operator:   monitoring.AlertConditionOperatorGreaterThanOrEqual,
			Threshold:  85.0,
			Duration:   2 * time.Minute,
			Tags: map[string]string{
				"component": "system",
			},
		},
		NotificationChannels: []string{"console-alerts"},
		Enabled:              true,
		Status:               monitoring.AlertStatusResolved,
	}
	collector.RegisterAlert(diskAlert)
}

// SystemStatsCollector collects system statistics
type SystemStatsCollector struct {
	id      string
	enabled bool
	mutex   sync.RWMutex
}

// NewSystemStatsCollector creates a new system stats collector
func NewSystemStatsCollector(id string) *SystemStatsCollector {
	return &SystemStatsCollector{
		id:      id,
		enabled: true,
	}
}

// ID returns the collector ID
func (c *SystemStatsCollector) ID() string {
	return c.id
}

// Enabled returns whether the collector is enabled
func (c *SystemStatsCollector) Enabled() bool {
	c.mutex.RLock()
	defer c.mutex.RUnlock()
	return c.enabled
}

// Collect collects metrics
func (c *SystemStatsCollector) Collect(ctx context.Context) ([]*monitoring.Metric, error) {
	// In a real implementation, these would be actual system metrics
	// For demonstration, we'll generate random values with occasional spikes

	// Generate metrics slice
	metrics := make([]*monitoring.Metric, 0, 3)

	// Generate CPU usage with occasional spikes
	cpuUsage := 50.0 + rand.Float64()*20.0
	if rand.Intn(5) == 0 { // 20% chance of spike
		cpuUsage = 85.0 + rand.Float64()*15.0
	}

	// Generate memory usage with occasional spikes
	memoryUsage := 60.0 + rand.Float64()*15.0
	if rand.Intn(7) == 0 { // ~14% chance of spike
		memoryUsage = 92.0 + rand.Float64()*8.0
	}

	// Generate disk usage (slowly increasing)
	diskUsage := 70.0 + rand.Float64()*10.0

	// Create CPU metric
	cpuMetric := monitoring.NewMetric(
		"system.cpu.usage",
		monitoring.MetricTypeGauge,
		cpuUsage,
		map[string]string{
			"component": "system",
			"unit":      "percent",
		},
	)

	// Create memory metric
	memoryMetric := monitoring.NewMetric(
		"system.memory.usage",
		monitoring.MetricTypeGauge,
		memoryUsage,
		map[string]string{
			"component": "system",
			"unit":      "percent",
		},
	)

	// Create disk metric
	diskMetric := monitoring.NewMetric(
		"system.disk.usage",
		monitoring.MetricTypeGauge,
		diskUsage,
		map[string]string{
			"component": "system",
			"unit":      "percent",
			"device":    "/dev/sda1",
		},
	)

	// Add metrics to the slice
	metrics = append(metrics, cpuMetric, memoryMetric, diskMetric)

	// Print current metrics
	fmt.Printf("System CPU: %.2f%%, Memory: %.2f%%, Disk: %.2f%%\n",
		cpuUsage, memoryUsage, diskUsage)

	return metrics, nil
}

// VMStatsCollector collects VM statistics
type VMStatsCollector struct {
	id      string
	enabled bool
	mutex   sync.RWMutex
}

// NewVMStatsCollector creates a new VM stats collector
func NewVMStatsCollector(id string) *VMStatsCollector {
	return &VMStatsCollector{
		id:      id,
		enabled: true,
	}
}

// ID returns the collector ID
func (c *VMStatsCollector) ID() string {
	return c.id
}

// Enabled returns whether the collector is enabled
func (c *VMStatsCollector) Enabled() bool {
	c.mutex.RLock()
	defer c.mutex.RUnlock()
	return c.enabled
}

// Collect collects metrics
func (c *VMStatsCollector) Collect(ctx context.Context) ([]*monitoring.Metric, error) {
	// In a real implementation, these would be actual VM metrics
	// For demonstration, we'll generate random values

	// Generate metrics slice
	metrics := make([]*monitoring.Metric, 0, 6)

	// Demo VMs
	vms := []string{"vm-1", "vm-2", "vm-3"}

	for _, vm := range vms {
		// Generate CPU usage
		cpuUsage := 30.0 + rand.Float64()*40.0

		// Generate memory usage
		memoryUsage := 40.0 + rand.Float64()*30.0

		// Create CPU metric
		cpuMetric := monitoring.NewMetric(
			"vm.cpu.usage",
			monitoring.MetricTypeGauge,
			cpuUsage,
			map[string]string{
				"component": "vm",
				"unit":      "percent",
				"vm_id":     vm,
			},
		)

		// Create memory metric
		memoryMetric := monitoring.NewMetric(
			"vm.memory.usage",
			monitoring.MetricTypeGauge,
			memoryUsage,
			map[string]string{
				"component": "vm",
				"unit":      "percent",
				"vm_id":     vm,
			},
		)

		// Add metrics to the slice
		metrics = append(metrics, cpuMetric, memoryMetric)
	}

	// Print summary of VM metrics
	fmt.Println("VM metrics collected for 3 VMs")

	return metrics, nil
}
