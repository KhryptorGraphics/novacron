package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/monitoring"
	"github.com/khryptorgraphics/novacron/backend/core/storage"
)

// ANSI color constants
const (
	colorReset  = "\033[0m"
	colorRed    = "\033[31m"
	colorGreen  = "\033[32m"
	colorYellow = "\033[33m"
	colorBlue   = "\033[34m"
	colorPurple = "\033[35m"
	colorCyan   = "\033[36m"
	colorWhite  = "\033[37m"
	colorBold   = "\033[1m"
)

func main() {
	fmt.Printf("%s%sNovaCron VM Telemetry Example%s\n\n", colorBold, colorBlue, colorReset)

	// Create context with cancellation
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Set up signal handling to gracefully shut down
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigCh
		fmt.Printf("\n%sReceived shutdown signal. Gracefully shutting down...%s\n", colorYellow, colorReset)
		cancel()
	}()

	// Create a metrics storage system
	// In a real deployment, this would be a distributed storage solution
	metricStorage := storage.NewInMemoryStorage()

	// Create a distributed metric collector
	collectorConfig := monitoring.DefaultDistributedMetricCollectorConfig()
	collectorConfig.NodeID = "local-test-node"
	collector := monitoring.NewDistributedMetricCollector(collectorConfig, metricStorage)

	// Create mock VM manager with test VMs
	mockVMs := []string{
		"vm-01-webserver",
		"vm-02-database",
		"vm-03-cache",
		"vm-04-batch",
		"vm-05-analytics",
	}
	vmManager := monitoring.NewMockVMManager(mockVMs)

	// Configure workload patterns for the mock VMs
	patternConfig := &monitoring.MockVMPatternConfig{
		CPUPatterns: map[string]monitoring.WorkloadPattern{
			"vm-01-webserver": monitoring.NewVariablePattern(50, 30, 0.1),          // Web server: Medium load with spikes
			"vm-02-database":  monitoring.NewStablePattern(65, 10),                 // Database: Stable higher load
			"vm-03-cache":     monitoring.NewSpikyPattern(25, 90, 0.05, 0.01),      // Cache: Low with occasional spikes
			"vm-04-batch":     monitoring.NewCyclicPattern(10, 95, 60*time.Second), // Batch: Cycles between low and high
			"vm-05-analytics": monitoring.NewRandomPattern(40, 30),                 // Analytics: Moderate random load
		},
		MemoryPatterns: map[string]monitoring.WorkloadPattern{
			"vm-01-webserver": monitoring.NewStablePattern(60, 5),                  // Web server: Stable memory
			"vm-02-database":  monitoring.NewStablePattern(80, 3),                  // Database: High stable memory
			"vm-03-cache":     monitoring.NewStablePattern(70, 2),                  // Cache: Fixed memory usage
			"vm-04-batch":     monitoring.NewCyclicPattern(40, 85, 90*time.Second), // Batch: Memory grows and shrinks
			"vm-05-analytics": monitoring.NewVariablePattern(65, 20, 0.08),         // Analytics: Variable memory
		},
		DiskPatterns: map[string]map[string]monitoring.WorkloadPattern{
			"vm-01-webserver": {
				"sda": monitoring.NewStablePattern(20, 3),          // OS disk: Low stable usage
				"sdb": monitoring.NewVariablePattern(40, 15, 0.05), // Data disk: Variable
			},
			"vm-02-database": {
				"sda": monitoring.NewStablePattern(30, 2),                   // OS disk: Low stable usage
				"sdb": monitoring.NewVariablePattern(75, 10, 0.02),          // Data disk: High variable usage
				"sdc": monitoring.NewCyclicPattern(50, 90, 300*time.Second), // Backup disk: Cycles with backups
			},
		},
		NetworkPatterns: map[string]map[string]monitoring.WorkloadPattern{
			"vm-01-webserver": {
				"eth0": monitoring.NewVariablePattern(60, 25, 0.1), // Primary: Variable traffic
			},
			"vm-02-database": {
				"eth0": monitoring.NewStablePattern(40, 15),             // Primary: Consistent traffic
				"eth1": monitoring.NewSpikyPattern(10, 80, 0.01, 0.005), // Backup: Occasional backup traffic
			},
		},
		IOPSPatterns: map[string]map[string]monitoring.WorkloadPattern{
			"vm-02-database": {
				"sdb": monitoring.NewSpikyPattern(100, 5000, 0.1, 0.05), // Database: Spiky IO pattern
			},
			"vm-04-batch": {
				"sda": monitoring.NewCyclicPattern(50, 2000, 120*time.Second), // Batch: Cyclic IO pattern
			},
		},
	}
	vmManager.ConfigureWorkloadPatterns(patternConfig)

	// Create VM telemetry collector configuration
	telemetryConfig := &monitoring.VMTelemetryCollectorConfig{
		CollectionInterval: 2 * time.Second, // Collect every 2 seconds for the demo
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
		Tags: map[string]string{
			"environment": "demo",
			"example":     "vm-telemetry",
		},
		NodeID:      "demo-node",
		DetailLevel: monitoring.StandardMetrics,
	}

	// Create VM telemetry collector
	vmTelemetryCollector := monitoring.NewVMTelemetryCollector(telemetryConfig, collector)

	// Setup real-time dashboard
	dashboard := NewVMTelemetryDashboard(metricStorage, mockVMs)

	// Start collectors
	fmt.Printf("%sStarting VM telemetry collector...%s\n", colorGreen, colorReset)
	err := vmTelemetryCollector.Start()
	if err != nil {
		fmt.Printf("%sError starting VM telemetry collector: %v%s\n", colorRed, err, colorReset)
		return
	}

	// Start dashboard
	fmt.Printf("%sStarting real-time VM telemetry dashboard...%s\n", colorGreen, colorReset)
	dashboard.Start(ctx)

	// Setup alerts
	setupAlerts(ctx, collector)

	// Wait for context cancellation
	<-ctx.Done()

	// Cleanup
	fmt.Printf("%sStopping VM telemetry collector...%s\n", colorYellow, colorReset)
	err = vmTelemetryCollector.Stop()
	if err != nil {
		fmt.Printf("%sError stopping VM telemetry collector: %v%s\n", colorRed, err, colorReset)
	}

	fmt.Printf("%sExample complete!%s\n", colorGreen, colorReset)
}

// VMTelemetryDashboard provides a real-time dashboard for VM telemetry
type VMTelemetryDashboard struct {
	storage     *storage.InMemoryStorage
	vmIDs       []string
	stopCh      chan struct{}
	refreshRate time.Duration
	mutex       sync.Mutex
}

// NewVMTelemetryDashboard creates a new VM telemetry dashboard
func NewVMTelemetryDashboard(metricStorage *storage.InMemoryStorage, vmIDs []string) *VMTelemetryDashboard {
	return &VMTelemetryDashboard{
		storage:     metricStorage,
		vmIDs:       vmIDs,
		stopCh:      make(chan struct{}),
		refreshRate: 3 * time.Second,
	}
}

// Start starts the dashboard refresh loop
func (d *VMTelemetryDashboard) Start(ctx context.Context) {
	go func() {
		ticker := time.NewTicker(d.refreshRate)
		defer ticker.Stop()

		for {
			select {
			case <-ticker.C:
				d.refresh(ctx)
			case <-ctx.Done():
				return
			case <-d.stopCh:
				return
			}
		}
	}()
}

// Stop stops the dashboard
func (d *VMTelemetryDashboard) Stop() {
	close(d.stopCh)
}

// refresh updates and renders the dashboard
func (d *VMTelemetryDashboard) refresh(ctx context.Context) {
	d.mutex.Lock()
	defer d.mutex.Unlock()

	// Clear screen
	fmt.Print("\033[H\033[2J") // ANSI escape codes to clear screen

	// Dashboard header
	fmt.Printf("%s%s[ NovaCron VM Telemetry Dashboard ]%s\n\n", colorBold, colorBlue, colorReset)
	fmt.Printf("Time: %s\n\n", time.Now().Format("2006-01-02 15:04:05"))

	// Loop through VMs and show metrics
	totalCPU := 0.0
	totalMemory := 0.0
	vmCount := 0

	// Create dashboard for each VM
	for _, vmID := range d.vmIDs {
		// CPU Usage
		cpuMetrics, _ := d.getLatestMetrics(ctx, "vm.cpu.usage", vmID)
		var cpuUsage float64
		if len(cpuMetrics) > 0 {
			cpuUsage = cpuMetrics[0].Value.(float64)
			totalCPU += cpuUsage
			vmCount++
		}

		// Memory Usage
		memMetrics, _ := d.getLatestMetrics(ctx, "vm.memory.usage_percent", vmID)
		var memUsage float64
		if len(memMetrics) > 0 {
			memUsage = memMetrics[0].Value.(float64)
			totalMemory += memUsage
		}

		// Disk Usage
		diskMetrics, _ := d.getLatestMetrics(ctx, "vm.disk.usage_percent", vmID)
		var diskUsage string
		if len(diskMetrics) > 0 {
			diskUsage = fmt.Sprintf("%.1f%%", diskMetrics[0].Value.(float64))
		} else {
			diskUsage = "N/A"
		}

		// Network Traffic
		rxMetrics, _ := d.getLatestMetrics(ctx, "vm.network.rx_bytes", vmID)
		txMetrics, _ := d.getLatestMetrics(ctx, "vm.network.tx_bytes", vmID)
		var networkUsage string
		if len(rxMetrics) > 0 && len(txMetrics) > 0 {
			rx := rxMetrics[0].Value.(float64)
			tx := txMetrics[0].Value.(float64)
			networkUsage = fmt.Sprintf("↓%.1f KB/s ↑%.1f KB/s", rx/1024, tx/1024)
		} else {
			networkUsage = "N/A"
		}

		// IOPs
		var iopsUsage string
		readIOPSMetrics, _ := d.getLatestMetrics(ctx, "vm.disk.read_iops", vmID)
		writeIOPSMetrics, _ := d.getLatestMetrics(ctx, "vm.disk.write_iops", vmID)
		if len(readIOPSMetrics) > 0 && len(writeIOPSMetrics) > 0 {
			read := readIOPSMetrics[0].Value.(float64)
			write := writeIOPSMetrics[0].Value.(float64)
			iopsUsage = fmt.Sprintf("R:%.0f W:%.0f", read, write)
		} else {
			iopsUsage = "N/A"
		}

		// VM Status Line
		fmt.Printf("%s%s[%s]%s ", colorBold, colorCyan, vmID, colorReset)

		// Status indicators
		cpuColor := getColorForUsage(cpuUsage)
		memColor := getColorForUsage(memUsage)

		// CPU Bar
		fmt.Printf("CPU: %s%5.1f%% %s%s | ", cpuColor, cpuUsage, colorReset, getBarGraph(cpuUsage, 10))

		// Memory Bar
		fmt.Printf("MEM: %s%5.1f%% %s%s | ", memColor, memUsage, colorReset, getBarGraph(memUsage, 10))

		// Disk Usage
		fmt.Printf("Disk: %5s | ", diskUsage)

		// Network Traffic
		fmt.Printf("Net: %s | ", networkUsage)

		// IOPs
		fmt.Printf("IOPs: %s\n", iopsUsage)
	}

	// Show system summary
	if vmCount > 0 {
		avgCPU := totalCPU / float64(vmCount)
		avgMem := totalMemory / float64(vmCount)

		fmt.Printf("\n%s%sSystem Summary:%s\n", colorBold, colorPurple, colorReset)
		fmt.Printf("Avg CPU: %s%5.1f%% %s%s | ", getColorForUsage(avgCPU), avgCPU, colorReset, getBarGraph(avgCPU, 20))
		fmt.Printf("Avg Memory: %s%5.1f%% %s%s\n", getColorForUsage(avgMem), avgMem, colorReset, getBarGraph(avgMem, 20))
	}

	// Show alerts if any
	fmt.Printf("\n%s%sRecent Alerts:%s\n", colorBold, colorYellow, colorReset)
	alertMetrics, _ := d.getLatestMetrics(ctx, "vm.alert", "")
	if len(alertMetrics) > 0 {
		for i := 0; i < min(5, len(alertMetrics)); i++ {
			metric := alertMetrics[i]
			fmt.Printf("%s[%s] %s: %s%s\n",
				colorRed,
				metric.Timestamp.Format("15:04:05"),
				metric.Tags["vm_id"],
				metric.Tags["message"],
				colorReset)
		}
	} else {
		fmt.Printf("%sNo recent alerts%s\n", colorGreen, colorReset)
	}

	fmt.Printf("\n%sPress Ctrl+C to exit%s\n", colorYellow, colorReset)
}

// getLatestMetrics gets the latest metrics for a given name and VM ID
func (d *VMTelemetryDashboard) getLatestMetrics(ctx context.Context, name string, vmID string) ([]*monitoring.Metric, error) {
	now := time.Now()
	past := now.Add(-10 * time.Minute)

	var tags map[string]string
	if vmID != "" {
		tags = map[string]string{"vm_id": vmID}
	}

	metrics, err := d.storage.GetLatestMetrics(ctx, name, tags, past, now, 1)
	if err != nil {
		return nil, err
	}

	return metrics, nil
}

// getBarGraph returns a bar graph string for the given percentage
func getBarGraph(percentage float64, length int) string {
	// Calculate filled part
	filledLength := int(percentage / 100.0 * float64(length))

	// Create bar
	bar := "["
	for i := 0; i < length; i++ {
		if i < filledLength {
			bar += "█"
		} else {
			bar += " "
		}
	}
	bar += "]"

	return bar
}

// getColorForUsage returns a color based on usage percentage
func getColorForUsage(percentage float64) string {
	if percentage >= 90 {
		return colorRed
	} else if percentage >= 70 {
		return colorYellow
	} else {
		return colorGreen
	}
}

// setupAlerts creates and registers alert definitions
func setupAlerts(ctx context.Context, collector *monitoring.DistributedMetricCollector) {
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
			Duration:   10 * time.Second,
		},
		Actions: []monitoring.AlertAction{
			{
				Type: monitoring.AlertActionConsole,
				Parameters: map[string]string{
					"message": "High CPU Usage Detected",
				},
			},
			{
				Type: monitoring.AlertActionMetric,
				Parameters: map[string]string{
					"name":    "vm.alert",
					"message": "High CPU Usage (>90%)",
				},
			},
		},
		Enabled: true,
	}

	// High Memory Alert
	memoryAlert := &monitoring.Alert{
		ID:          "vm-high-memory-usage",
		Name:        "VM High Memory Usage",
		Description: "VM memory usage is critically high",
		Severity:    monitoring.AlertSeverityCritical,
		Type:        monitoring.AlertTypeThreshold,
		Condition: monitoring.AlertCondition{
			MetricName: "vm.memory.usage_percent",
			Operator:   monitoring.AlertConditionOperatorGreaterThan,
			Threshold:  85.0,
			Duration:   15 * time.Second,
		},
		Actions: []monitoring.AlertAction{
			{
				Type: monitoring.AlertActionConsole,
				Parameters: map[string]string{
					"message": "High Memory Usage Detected",
				},
			},
			{
				Type: monitoring.AlertActionMetric,
				Parameters: map[string]string{
					"name":    "vm.alert",
					"message": "High Memory Usage (>85%)",
				},
			},
		},
		Enabled: true,
	}

	// High Disk Usage Alert
	diskAlert := &monitoring.Alert{
		ID:          "vm-high-disk-usage",
		Name:        "VM High Disk Usage",
		Description: "VM disk usage is critically high",
		Severity:    monitoring.AlertSeverityWarning,
		Type:        monitoring.AlertTypeThreshold,
		Condition: monitoring.AlertCondition{
			MetricName: "vm.disk.usage_percent",
			Operator:   monitoring.AlertConditionOperatorGreaterThan,
			Threshold:  80.0,
			Duration:   30 * time.Second,
		},
		Actions: []monitoring.AlertAction{
			{
				Type: monitoring.AlertActionConsole,
				Parameters: map[string]string{
					"message": "High Disk Usage Detected",
				},
			},
			{
				Type: monitoring.AlertActionMetric,
				Parameters: map[string]string{
					"name":    "vm.alert",
					"message": "High Disk Usage (>80%)",
				},
			},
		},
		Enabled: true,
	}

	// Register alerts
	collector.RegisterAlert(ctx, cpuAlert)
	collector.RegisterAlert(ctx, memoryAlert)
	collector.RegisterAlert(ctx, diskAlert)
}

// min returns the minimum of two integers
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
