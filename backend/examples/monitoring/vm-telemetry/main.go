package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/monitoring"
	"github.com/khryptorgraphics/novacron/backend/core/storage"
)

// VM Telemetry Demo
// This example demonstrates the use of the VM Telemetry Collector with a Mock VM Manager.
// It shows how to integrate VM monitoring into the NovaCron system.

func main() {
	fmt.Println("NovaCron VM Telemetry Monitoring Demo")
	fmt.Println("======================================")

	// Initialize storage for metrics
	metricStorage := storage.NewInMemoryStorage()

	// Initialize mock VM manager with predefined VMs
	vmIDs := []string{
		"vm-postgresql-db1",
		"vm-redis-cache1",
		"vm-web-server1",
		"vm-web-server2",
		"vm-batch-processor",
	}
	vmManager := monitoring.NewMockVMManager(vmIDs)

	// Initialize distributed metric collector
	collectorConfig := monitoring.DefaultDistributedMetricCollectorConfig()
	collectorConfig.CollectionInterval = 5 * time.Second
	collectorConfig.NodeID = "hypervisor-node1"
	collectorConfig.ClusterID = "east-datacenter"
	collectorConfig.Tags = map[string]string{
		"environment": "production",
		"service":     "vm-monitoring",
		"datacenter":  "east-us-1",
	}

	// Create metric collector with storage
	collector := monitoring.NewDistributedMetricCollector(collectorConfig, metricStorage)

	// Initialize VM telemetry collector
	vmTelemetryConfig := &monitoring.VMTelemetryCollectorConfig{
		CollectionInterval: 10 * time.Second,
		VMManager:          vmManager,
		EnabledMetrics: monitoring.VMMetricTypes{
			CPU:              true,
			Memory:           true,
			Disk:             true,
			Network:          true,
			IOPs:             true,
			ProcessStats:     true,
			ApplicationStats: false,
			GuestMetrics:     false,
		},
		NodeID:      "hypervisor-node1",
		DetailLevel: monitoring.DetailedMetrics,
		Tags: map[string]string{
			"collector_type": "vm_telemetry",
			"version":        "1.0",
		},
	}

	vmTelemetryCollector := monitoring.NewVMTelemetryCollector(vmTelemetryConfig, collector)

	// Set up alerts
	setupVMAlerts(collector)

	// Set up analytics
	analyticsConfig := monitoring.DefaultAnalyticsEngineConfig()
	analyticsConfig.ProcessingInterval = 15 * time.Second
	analytics := monitoring.NewAnalyticsEngine(analyticsConfig, collector)

	// Console notifier for alerts
	consoleNotifier := &ConsoleNotifier{}
	collector.AlertManager().AddNotifier(consoleNotifier)

	// Start all components
	fmt.Println("Starting VM telemetry collection...")
	collector.Start()
	vmTelemetryCollector.Start()
	analytics.Start()

	// Print demo information
	fmt.Println("\nDemo is running with the following components:")
	fmt.Println("- Distributed Metric Collector")
	fmt.Println("- VM Telemetry Collector")
	fmt.Println("- Analytics Engine")
	fmt.Println("- Mock VM Manager with 5 simulated VMs")
	fmt.Println("- Alert Manager with Console Notifications")

	fmt.Println("\nMonitored VMs:")
	for i, vmID := range vmIDs {
		fmt.Printf("  %d. %s\n", i+1, vmID)
	}

	fmt.Println("\nMetrics will be collected every 10 seconds.")
	fmt.Println("Alerts will be triggered when metrics exceed thresholds.")
	fmt.Println("\nPress Ctrl+C to stop the demo.")

	// Dashboard updater in a separate goroutine
	stopChan := make(chan struct{})
	go updateDashboard(collector, vmManager, stopChan)

	// Wait for termination signal
	termChan := make(chan os.Signal, 1)
	signal.Notify(termChan, syscall.SIGINT, syscall.SIGTERM)
	<-termChan

	// Graceful shutdown
	fmt.Println("\nShutting down...")
	close(stopChan)
	analytics.Stop()
	vmTelemetryCollector.Stop()
	collector.Stop()
	fmt.Println("Demo stopped.")
}

// setupVMAlerts configures alerts for VM metrics
func setupVMAlerts(collector *monitoring.DistributedMetricCollector) {
	// High CPU Alert
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
		NotificationChannels: []string{"console"},
		Enabled:              true,
		Status:               monitoring.AlertStatusResolved,
	}
	collector.RegisterAlert(cpuAlert)

	// High Memory Alert
	memoryAlert := &monitoring.Alert{
		ID:          "vm-high-memory-usage",
		Name:        "VM High Memory Usage",
		Description: "VM memory usage is critically high",
		Severity:    monitoring.AlertSeverityWarning,
		Type:        monitoring.AlertTypeThreshold,
		Condition: monitoring.AlertCondition{
			MetricName: "vm.memory.usage_percent",
			Operator:   monitoring.AlertConditionOperatorGreaterThanOrEqual,
			Threshold:  90.0,
			Duration:   30 * time.Second,
			Tags: map[string]string{
				"component": "vm",
			},
		},
		NotificationChannels: []string{"console"},
		Enabled:              true,
		Status:               monitoring.AlertStatusResolved,
	}
	collector.RegisterAlert(memoryAlert)

	// High Disk Usage Alert
	diskAlert := &monitoring.Alert{
		ID:          "vm-high-disk-usage",
		Name:        "VM High Disk Usage",
		Description: "VM disk usage is critically high",
		Severity:    monitoring.AlertSeverityError,
		Type:        monitoring.AlertTypeThreshold,
		Condition: monitoring.AlertCondition{
			MetricName: "vm.disk.usage_percent",
			Operator:   monitoring.AlertConditionOperatorGreaterThanOrEqual,
			Threshold:  90.0,
			Duration:   2 * time.Minute,
			Tags: map[string]string{
				"component": "vm",
				"disk_type": "system",
			},
		},
		NotificationChannels: []string{"console"},
		Enabled:              true,
		Status:               monitoring.AlertStatusResolved,
	}
	collector.RegisterAlert(diskAlert)

	// Network Throughput Alert
	networkAlert := &monitoring.Alert{
		ID:          "vm-high-network-usage",
		Name:        "VM High Network Usage",
		Description: "VM network usage is unusually high",
		Severity:    monitoring.AlertSeverityWarning,
		Type:        monitoring.AlertTypeThreshold,
		Condition: monitoring.AlertCondition{
			MetricName: "vm.network.rx_bytes",
			Operator:   monitoring.AlertConditionOperatorGreaterThan,
			Threshold:  100 * 1024 * 1024, // 100 MB/s
			Duration:   30 * time.Second,
			Tags: map[string]string{
				"component": "vm",
			},
		},
		NotificationChannels: []string{"console"},
		Enabled:              true,
		Status:               monitoring.AlertStatusResolved,
	}
	collector.RegisterAlert(networkAlert)

	// Disk Latency Alert
	diskLatencyAlert := &monitoring.Alert{
		ID:          "vm-high-disk-latency",
		Name:        "VM High Disk Latency",
		Description: "VM disk latency is critically high",
		Severity:    monitoring.AlertSeverityWarning,
		Type:        monitoring.AlertTypeThreshold,
		Condition: monitoring.AlertCondition{
			MetricName: "vm.disk.write_latency",
			Operator:   monitoring.AlertConditionOperatorGreaterThan,
			Threshold:  50.0, // 50ms
			Duration:   1 * time.Minute,
			Tags: map[string]string{
				"component": "vm",
			},
		},
		NotificationChannels: []string{"console"},
		Enabled:              true,
		Status:               monitoring.AlertStatusResolved,
	}
	collector.RegisterAlert(diskLatencyAlert)
}

// updateDashboard periodically prints a simple dashboard of VM stats
func updateDashboard(collector *monitoring.DistributedMetricCollector, vmManager *monitoring.MockVMManager, stopChan <-chan struct{}) {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			clearScreen()
			printDashboardHeader()

			ctx := context.Background()
			vmIDs, err := vmManager.GetVMs(ctx)
			if err != nil {
				fmt.Println("Error getting VM list:", err)
				continue
			}

			// Get and display stats for each VM
			for _, vmID := range vmIDs {
				stats, err := vmManager.GetVMStats(ctx, vmID, monitoring.StandardMetrics)
				if err != nil {
					fmt.Printf("Error getting stats for VM %s: %v\n", vmID, err)
					continue
				}

				printVMStats(vmID, stats)
			}

			// Sleep briefly to avoid screen flicker
			time.Sleep(100 * time.Millisecond)
		case <-stopChan:
			return
		}
	}
}

// clearScreen clears the terminal screen
func clearScreen() {
	fmt.Print("\033[H\033[2J") // ANSI escape code to clear screen
}

// printDashboardHeader prints the dashboard header
func printDashboardHeader() {
	fmt.Println("╔══════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                       NOVACRON VM MONITORING DASHBOARD                   ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════╝")
	fmt.Println()
}

// printVMStats prints stats for a single VM
func printVMStats(vmID string, stats *monitoring.VMStats) {
	// Print VM header
	fmt.Printf("┌─────────────────────────── VM: %s ───────────────────────────┐\n", vmID)

	// Print CPU stats
	fmt.Printf("│ CPU: %.1f%% (Cores: %d, System: %.1f%%, User: %.1f%%, IOWait: %.1f%%)  │\n",
		stats.CPU.Usage, stats.CPU.NumCPUs, stats.CPU.SystemTime, stats.CPU.UserTime, stats.CPU.IOWaitTime)

	// Print Memory stats
	fmt.Printf("│ Memory: %.1f%% (%.1f GB / %.1f GB) │\n",
		stats.Memory.UsagePercent,
		float64(stats.Memory.Used)/(1024*1024*1024),
		float64(stats.Memory.Total)/(1024*1024*1024))

	// Print Disk stats for each disk
	for _, disk := range stats.Disks {
		fmt.Printf("│ Disk (%s): %.1f%% used, R: %.0f IOPS/%.1f MB/s, W: %.0f IOPS/%.1f MB/s │\n",
			disk.Path,
			disk.UsagePercent,
			disk.ReadIOPS,
			disk.ReadThroughput/(1024*1024),
			disk.WriteIOPS,
			disk.WriteThroughput/(1024*1024))
	}

	// Print Network stats
	for i, net := range stats.Networks {
		if i < 2 { // Only show first 2 interfaces to keep display compact
			fmt.Printf("│ Net (%s): Rx: %.1f MB/s, Tx: %.1f MB/s │\n",
				net.Name,
				net.RxBytes/(1024*1024),
				net.TxBytes/(1024*1024))
		}
	}

	// Print footer
	fmt.Println("└────────────────────────────────────────────────────────────────────────┘")
	fmt.Println()
}

// ConsoleNotifier implements a simple console-based notifier
type ConsoleNotifier struct{}

// Notify sends a notification message
func (n *ConsoleNotifier) Notify(alert *monitoring.Alert) error {
	fmt.Println("\n╔══════════════════════════ ALERT NOTIFICATION ═════════════════════════╗")
	fmt.Printf("║ %s: %s\n", alert.Severity, alert.Name)
	fmt.Printf("║ Description: %s\n", alert.Description)
	fmt.Printf("║ State: %s\n", alert.State)
	if alert.CurrentValue != nil {
		fmt.Printf("║ Current Value: %.2f\n", *alert.CurrentValue)
	}
	fmt.Println("╚═════════════════════════════════════════════════════════════════════════╝")
	return nil
}
