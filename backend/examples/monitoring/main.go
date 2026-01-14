package main

import (
	"fmt"
	"math/rand"
	"sync"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/monitoring"
)

func main() {
	fmt.Println("NovaCron Monitoring System Demo")
	fmt.Println("===============================")

	// Create metric registry
	registry := monitoring.NewMetricRegistry()

	// Create alert registry
	alertRegistry := monitoring.NewAlertRegistry()

	// Create notification manager
	notificationManager := monitoring.NewNotificationManager()

	// Add default templates
	emailTemplate := monitoring.DefaultEmailTemplate()
	webhookTemplate := monitoring.DefaultWebhookTemplate()
	notificationManager.AddTemplate(emailTemplate)
	notificationManager.AddTemplate(webhookTemplate)

	// Create a console notifier for demonstration
	consoleNotifier := NewConsoleNotifier()
	notificationManager.RegisterNotifier(monitoring.EmailChannel, consoleNotifier)

	// Create alert manager
	alertManager := monitoring.NewAlertManager(alertRegistry, registry, 5*time.Second)
	alertManager.AddNotifier(consoleNotifier)
	alertManager.Start()

	// Create system collector
	systemCollector := monitoring.NewSystemCollector(registry, 2*time.Second)
	systemCollector.Start()

	// Create demo collector for CPU and memory
	demoCollector := NewDemoCollector(registry, 1*time.Second)
	demoCollector.Start()

	// Create collector manager
	collectorManager := monitoring.NewCollectorManager()
	collectorManager.AddCollector(systemCollector)
	collectorManager.AddCollector(demoCollector)

	// Create metrics history manager
	historyManager := monitoring.NewMetricHistoryManager(registry, 24*time.Hour, 1*time.Hour)
	historyManager.Start()

	// Create CPU usage alert
	cpuAlert := monitoring.NewAlert(
		"cpu-usage-alert",
		"High CPU Usage",
		"Alert when CPU usage exceeds 80%",
		monitoring.AlertSeverityHigh,
		monitoring.AlertCondition{
			Type:     monitoring.ThresholdCondition,
			MetricID: "demo.cpu.usage",
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

	// Create memory usage alert
	memAlert := monitoring.NewAlert(
		"memory-usage-alert",
		"High Memory Usage",
		"Alert when memory usage exceeds 90%",
		monitoring.AlertSeverityCritical,
		monitoring.AlertCondition{
			Type:     monitoring.ThresholdCondition,
			MetricID: "demo.memory.usage",
			Operator: monitoring.GreaterThanOrEqual,
			Threshold: func() *float64 {
				val := 90.0
				return &val
			}(),
		},
	)

	// Register the alert
	alertRegistry.RegisterAlert(memAlert)

	fmt.Println("\nDemo is running. Press Ctrl+C to stop.")
	fmt.Println("The system will generate random CPU and memory metrics.")
	fmt.Println("Alerts will be triggered when metrics exceed thresholds.")

	// Keep the program running
	select {}
}

// DemoCollector is a demo collector for CPU and memory metrics
type DemoCollector struct {
	metrics        []*monitoring.Metric
	registry       *monitoring.MetricRegistry
	interval       time.Duration
	stopChan       chan struct{}
	wg             sync.WaitGroup
	lastCollection time.Time
	enabled        bool
	mutex          sync.RWMutex
}

// NewDemoCollector creates a new demo collector
func NewDemoCollector(registry *monitoring.MetricRegistry, interval time.Duration) *DemoCollector {
	return &DemoCollector{
		metrics:  make([]*monitoring.Metric, 0),
		registry: registry,
		interval: interval,
		stopChan: make(chan struct{}),
		enabled:  true,
	}
}

// Start starts the collector
func (c *DemoCollector) Start() error {
	// Register metrics
	if err := c.registerMetrics(); err != nil {
		return err
	}

	c.wg.Add(1)
	go c.run()
	return nil
}

// Stop stops the collector
func (c *DemoCollector) Stop() error {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	if !c.enabled {
		return nil
	}

	close(c.stopChan)
	c.wg.Wait()
	c.enabled = false
	return nil
}

// GetMetrics gets the metrics this collector provides
func (c *DemoCollector) GetMetrics() []*monitoring.Metric {
	c.mutex.RLock()
	defer c.mutex.RUnlock()
	return c.metrics
}

// SetCollectInterval sets the collection interval
func (c *DemoCollector) SetCollectInterval(interval time.Duration) {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	c.interval = interval
}

// Collect collects metrics
func (c *DemoCollector) Collect() ([]monitoring.MetricBatch, error) {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	batches := make([]monitoring.MetricBatch, 0, len(c.metrics))
	c.lastCollection = time.Now()

	// Generate random CPU usage with occasional spikes
	cpuUsage := 40.0 + rand.Float64()*30.0
	if rand.Intn(10) == 0 {
		// 10% chance of a CPU spike
		cpuUsage = 85.0 + rand.Float64()*15.0
	}

	// Generate random memory usage with occasional spikes
	memUsage := 50.0 + rand.Float64()*30.0
	if rand.Intn(15) == 0 {
		// ~7% chance of a memory spike
		memUsage = 92.0 + rand.Float64()*8.0
	}

	// For each metric, collect its value
	for _, metric := range c.metrics {
		batch := monitoring.MetricBatch{
			MetricID:  metric.ID,
			Timestamp: c.lastCollection,
			Values:    make([]monitoring.MetricValue, 0, 1),
		}

		// Collect the appropriate metric value
		switch metric.ID {
		case "demo.cpu.usage":
			metric.RecordValue(cpuUsage, nil)
			batch.Values = append(batch.Values, monitoring.MetricValue{
				Timestamp: c.lastCollection,
				Value:     cpuUsage,
			})
			fmt.Printf("CPU Usage: %.2f%%\n", cpuUsage)
		case "demo.memory.usage":
			metric.RecordValue(memUsage, nil)
			batch.Values = append(batch.Values, monitoring.MetricValue{
				Timestamp: c.lastCollection,
				Value:     memUsage,
			})
			fmt.Printf("Memory Usage: %.2f%%\n", memUsage)
		}

		batches = append(batches, batch)
	}

	return batches, nil
}

// registerMetrics registers the metrics this collector provides
func (c *DemoCollector) registerMetrics() error {
	// Create CPU usage metric
	cpuUsage := monitoring.NewGaugeMetric("demo.cpu.usage", "CPU Usage", "Percentage of CPU usage", "demo")
	cpuUsage.SetUnit("percent")
	c.metrics = append(c.metrics, cpuUsage)
	if err := c.registry.RegisterMetric(cpuUsage); err != nil {
		return err
	}

	// Create memory usage metric
	memUsage := monitoring.NewGaugeMetric("demo.memory.usage", "Memory Usage", "Percentage of memory usage", "demo")
	memUsage.SetUnit("percent")
	c.metrics = append(c.metrics, memUsage)
	if err := c.registry.RegisterMetric(memUsage); err != nil {
		return err
	}

	return nil
}

// run is the main loop of the collector
func (c *DemoCollector) run() {
	defer c.wg.Done()

	ticker := time.NewTicker(c.interval)
	defer ticker.Stop()

	for {
		select {
		case <-c.stopChan:
			return
		case <-ticker.C:
			if _, err := c.Collect(); err != nil {
				fmt.Printf("Error collecting demo metrics: %v\n", err)
			}
		}
	}
}

// ConsoleNotifier is a simple notifier that prints to the console
type ConsoleNotifier struct{}

// NewConsoleNotifier creates a new console notifier
func NewConsoleNotifier() *ConsoleNotifier {
	return &ConsoleNotifier{}
}

// Notify sends a notification
func (n *ConsoleNotifier) Notify(alert *monitoring.Alert) error {
	fmt.Println("\n==== ALERT NOTIFICATION ====")
	fmt.Printf("Name: %s\n", alert.Name)
	fmt.Printf("Severity: %s\n", alert.Severity)
	fmt.Printf("State: %s\n", alert.State)
	fmt.Printf("Description: %s\n", alert.Description)
	if alert.CurrentValue != nil {
		fmt.Printf("Current Value: %.2f\n", *alert.CurrentValue)
	}
	fmt.Println("============================\n")
	return nil
}

// NotifyWithTemplate sends a notification with a template
func (n *ConsoleNotifier) NotifyWithTemplate(alert *monitoring.Alert, renderedBody string, template *monitoring.NotificationTemplate) error {
	fmt.Println("\n==== TEMPLATED ALERT NOTIFICATION ====")
	fmt.Printf("Template: %s\n", template.Name)
	fmt.Printf("Format: %s\n", template.Format)
	fmt.Println("Content:")
	fmt.Println(renderedBody)
	fmt.Println("===============================\n")
	return nil
}
