package monitoring

import (
	"context"
	"fmt"
	"sync"
	"time"
)

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

// VMMetricTypes represents a set of VM metric collection options
type VMMetricTypes struct {
	CPU              bool
	Memory           bool
	Disk             bool
	Network          bool
	IOPs             bool
	ProcessStats     bool
	ApplicationStats bool
	GuestMetrics     bool
}

// VMMetricDetailLevel represents the level of detail for VM metrics
type VMMetricDetailLevel int

const (
	// BasicMetrics collects only essential metrics
	BasicMetrics VMMetricDetailLevel = iota

	// StandardMetrics collects normal operational metrics
	StandardMetrics

	// DetailedMetrics collects comprehensive metrics including per-process stats
	DetailedMetrics

	// DiagnosticMetrics collects all available metrics for troubleshooting
	DiagnosticMetrics
)

// VMManagerInterface defines the interface for VM management
type VMManagerInterface interface {
	// GetVMs returns a list of all VM IDs
	GetVMs(ctx context.Context) ([]string, error)

	// GetVMStats retrieves stats for a specific VM
	GetVMStats(ctx context.Context, vmID string, detailLevel VMMetricDetailLevel) (*VMStats, error)
}

// DefaultVMTelemetryCollectorConfig returns a default configuration
func DefaultVMTelemetryCollectorConfig() *VMTelemetryCollectorConfig {
	return &VMTelemetryCollectorConfig{
		CollectionInterval: 30 * time.Second,
		EnabledMetrics: VMMetricTypes{
			CPU:              true,
			Memory:           true,
			Disk:             true,
			Network:          true,
			IOPs:             true,
			ProcessStats:     false,
			ApplicationStats: false,
			GuestMetrics:     false,
		},
		Tags:        make(map[string]string),
		DetailLevel: StandardMetrics,
	}
}

// VMTelemetryCollector collects detailed metrics from VMs
type VMTelemetryCollector struct {
	config *VMTelemetryCollectorConfig

	// VM manager for querying VM data
	vmManager VMManagerInterface

	// Collection state
	running      bool
	collectMutex sync.Mutex
	stopChan     chan struct{}
	lastRun      time.Time

	// Metric collector to send metrics to
	metricCollector *DistributedMetricCollector
}

// NewVMTelemetryCollector creates a new VM telemetry collector
func NewVMTelemetryCollector(config *VMTelemetryCollectorConfig, collector *DistributedMetricCollector) *VMTelemetryCollector {
	if config == nil {
		config = DefaultVMTelemetryCollectorConfig()
	}

	return &VMTelemetryCollector{
		config:          config,
		vmManager:       config.VMManager,
		metricCollector: collector,
		stopChan:        make(chan struct{}),
	}
}

// ID returns the collector ID
func (c *VMTelemetryCollector) ID() string {
	return "vm-telemetry-collector"
}

// Enabled returns whether the collector is enabled
func (c *VMTelemetryCollector) Enabled() bool {
	c.collectMutex.Lock()
	defer c.collectMutex.Unlock()
	return c.running
}

// Start begins metric collection
func (c *VMTelemetryCollector) Start() error {
	c.collectMutex.Lock()
	defer c.collectMutex.Unlock()

	if c.running {
		return fmt.Errorf("VM telemetry collector already running")
	}

	c.running = true
	c.stopChan = make(chan struct{})

	// Start the collection goroutine
	go c.collectLoop()

	return nil
}

// Stop halts metric collection
func (c *VMTelemetryCollector) Stop() error {
	c.collectMutex.Lock()
	defer c.collectMutex.Unlock()

	if !c.running {
		return nil
	}

	c.running = false
	close(c.stopChan)

	return nil
}

// collectLoop runs the main collection loop
func (c *VMTelemetryCollector) collectLoop() {
	ticker := time.NewTicker(c.config.CollectionInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			ctx, cancel := context.WithTimeout(context.Background(), c.config.CollectionInterval/2)
			c.Collect(ctx)
			cancel()
		case <-c.stopChan:
			return
		}
	}
}

// Collect collects metrics from all VMs
func (c *VMTelemetryCollector) Collect(ctx context.Context) ([]*Metric, error) {
	if c.vmManager == nil {
		return nil, fmt.Errorf("VM manager not configured")
	}

	// Update last run time
	c.collectMutex.Lock()
	c.lastRun = time.Now()
	c.collectMutex.Unlock()

	// Get list of VMs
	vmIDs, err := c.vmManager.GetVMs(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to get VM list: %w", err)
	}

	// Collect metrics for each VM
	var allMetrics []*Metric
	var collectWG sync.WaitGroup
	metricsChan := make(chan []*Metric)
	errorsChan := make(chan error)

	// Limit concurrent collections
	const maxConcurrent = 10
	semaphore := make(chan struct{}, maxConcurrent)

	// Start collectors for each VM
	for _, vmID := range vmIDs {
		collectWG.Add(1)
		go func(id string) {
			defer collectWG.Done()
			semaphore <- struct{}{}
			defer func() { <-semaphore }()

			metrics, err := c.collectVMMetrics(ctx, id)
			if err != nil {
				errorsChan <- fmt.Errorf("error collecting metrics for VM %s: %w", id, err)
				return
			}
			metricsChan <- metrics
		}(vmID)
	}

	// Close channels when all collectors are done
	go func() {
		collectWG.Wait()
		close(metricsChan)
		close(errorsChan)
	}()

	// Collect errors
	var collectionErrors []error
	go func() {
		for err := range errorsChan {
			collectionErrors = append(collectionErrors, err)
		}
	}()

	// Collect metrics
	for metrics := range metricsChan {
		allMetrics = append(allMetrics, metrics...)
	}

	// Report metrics to the distributed collector
	for _, metric := range allMetrics {
		if err := c.metricCollector.StoreMetric(ctx, metric); err != nil {
			fmt.Printf("Failed to store metric %s: %v\n", metric.Name, err)
		}
	}

	return allMetrics, nil
}

// collectVMMetrics collects metrics for a single VM
func (c *VMTelemetryCollector) collectVMMetrics(ctx context.Context, vmID string) ([]*Metric, error) {
	// Get detailed stats from the VM
	stats, err := c.vmManager.GetVMStats(ctx, vmID, c.config.DetailLevel)
	if err != nil {
		return nil, fmt.Errorf("failed to get VM stats: %w", err)
	}

	// Convert stats to metrics
	var metrics []*Metric
	timestamp := stats.Timestamp

	// Add default tags
	baseTags := make(map[string]string)
	for k, v := range c.config.Tags {
		baseTags[k] = v
	}
	baseTags["vm_id"] = vmID
	baseTags["node_id"] = c.config.NodeID

	// Collect CPU metrics
	if c.config.EnabledMetrics.CPU {
		cpuTags := copyTags(baseTags)
		cpuTags["metric_type"] = "cpu"

		metrics = append(metrics, NewMetric(
			"vm.cpu.usage",
			MetricTypeGauge,
			stats.CPU.Usage,
			cpuTags,
		))

		metrics = append(metrics, NewMetric(
			"vm.cpu.steal_time",
			MetricTypeGauge,
			stats.CPU.StealTime,
			cpuTags,
		))

		metrics = append(metrics, NewMetric(
			"vm.cpu.ready_time",
			MetricTypeGauge,
			stats.CPU.ReadyTime,
			cpuTags,
		))

		metrics = append(metrics, NewMetric(
			"vm.cpu.system_time",
			MetricTypeGauge,
			stats.CPU.SystemTime,
			cpuTags,
		))

		metrics = append(metrics, NewMetric(
			"vm.cpu.user_time",
			MetricTypeGauge,
			stats.CPU.UserTime,
			cpuTags,
		))

		metrics = append(metrics, NewMetric(
			"vm.cpu.iowait_time",
			MetricTypeGauge,
			stats.CPU.IOWaitTime,
			cpuTags,
		))

		// Per-core metrics if available and detail level is high enough
		if len(stats.CPU.CoreUsage) > 0 && c.config.DetailLevel >= StandardMetrics {
			for i, usage := range stats.CPU.CoreUsage {
				coreTags := copyTags(cpuTags)
				coreTags["core"] = fmt.Sprintf("%d", i)
				metrics = append(metrics, NewMetric(
					"vm.cpu.core.usage",
					MetricTypeGauge,
					usage,
					coreTags,
				))
			}
		}
	}

	// Collect memory metrics
	if c.config.EnabledMetrics.Memory {
		memTags := copyTags(baseTags)
		memTags["metric_type"] = "memory"

		metrics = append(metrics, NewMetric(
			"vm.memory.usage_percent",
			MetricTypeGauge,
			stats.Memory.UsagePercent,
			memTags,
		))

		metrics = append(metrics, NewMetric(
			"vm.memory.used",
			MetricTypeGauge,
			float64(stats.Memory.Used),
			memTags,
		))

		metrics = append(metrics, NewMetric(
			"vm.memory.total",
			MetricTypeGauge,
			float64(stats.Memory.Total),
			memTags,
		))

		metrics = append(metrics, NewMetric(
			"vm.memory.free",
			MetricTypeGauge,
			float64(stats.Memory.Free),
			memTags,
		))

		if c.config.DetailLevel >= StandardMetrics {
			metrics = append(metrics, NewMetric(
				"vm.memory.swap_used",
				MetricTypeGauge,
				float64(stats.Memory.SwapUsed),
				memTags,
			))

			metrics = append(metrics, NewMetric(
				"vm.memory.swap_total",
				MetricTypeGauge,
				float64(stats.Memory.SwapTotal),
				memTags,
			))

			metrics = append(metrics, NewMetric(
				"vm.memory.page_faults",
				MetricTypeGauge,
				stats.Memory.PageFaults,
				memTags,
			))

			metrics = append(metrics, NewMetric(
				"vm.memory.major_page_faults",
				MetricTypeGauge,
				stats.Memory.MajorPageFaults,
				memTags,
			))
		}

		// Balloon metrics if available
		if stats.Memory.BalloonTarget > 0 || stats.Memory.BalloonCurrent > 0 {
			metrics = append(metrics, NewMetric(
				"vm.memory.balloon_target",
				MetricTypeGauge,
				float64(stats.Memory.BalloonTarget),
				memTags,
			))

			metrics = append(metrics, NewMetric(
				"vm.memory.balloon_current",
				MetricTypeGauge,
				float64(stats.Memory.BalloonCurrent),
				memTags,
			))
		}
	}

	// Collect disk metrics
	if c.config.EnabledMetrics.Disk {
		for _, disk := range stats.Disks {
			diskTags := copyTags(baseTags)
			diskTags["metric_type"] = "disk"
			diskTags["disk_id"] = disk.DiskID
			diskTags["disk_path"] = disk.Path
			diskTags["disk_type"] = disk.Type

			metrics = append(metrics, NewMetric(
				"vm.disk.usage_percent",
				MetricTypeGauge,
				disk.UsagePercent,
				diskTags,
			))

			metrics = append(metrics, NewMetric(
				"vm.disk.used",
				MetricTypeGauge,
				float64(disk.Used),
				diskTags,
			))

			metrics = append(metrics, NewMetric(
				"vm.disk.size",
				MetricTypeGauge,
				float64(disk.Size),
				diskTags,
			))

			// IO metrics if enabled
			if c.config.EnabledMetrics.IOPs {
				metrics = append(metrics, NewMetric(
					"vm.disk.read_iops",
					MetricTypeGauge,
					disk.ReadIOPS,
					diskTags,
				))

				metrics = append(metrics, NewMetric(
					"vm.disk.write_iops",
					MetricTypeGauge,
					disk.WriteIOPS,
					diskTags,
				))

				metrics = append(metrics, NewMetric(
					"vm.disk.read_throughput",
					MetricTypeGauge,
					disk.ReadThroughput,
					diskTags,
				))

				metrics = append(metrics, NewMetric(
					"vm.disk.write_throughput",
					MetricTypeGauge,
					disk.WriteThroughput,
					diskTags,
				))

				metrics = append(metrics, NewMetric(
					"vm.disk.read_latency",
					MetricTypeGauge,
					disk.ReadLatency,
					diskTags,
				))

				metrics = append(metrics, NewMetric(
					"vm.disk.write_latency",
					MetricTypeGauge,
					disk.WriteLatency,
					diskTags,
				))
			}
		}
	}

	// Collect network metrics
	if c.config.EnabledMetrics.Network {
		for _, network := range stats.Networks {
			netTags := copyTags(baseTags)
			netTags["metric_type"] = "network"
			netTags["interface_id"] = network.InterfaceID
			netTags["interface_name"] = network.Name

			metrics = append(metrics, NewMetric(
				"vm.network.rx_bytes",
				MetricTypeGauge,
				network.RxBytes,
				netTags,
			))

			metrics = append(metrics, NewMetric(
				"vm.network.tx_bytes",
				MetricTypeGauge,
				network.TxBytes,
				netTags,
			))

			metrics = append(metrics, NewMetric(
				"vm.network.rx_packets",
				MetricTypeGauge,
				network.RxPackets,
				netTags,
			))

			metrics = append(metrics, NewMetric(
				"vm.network.tx_packets",
				MetricTypeGauge,
				network.TxPackets,
				netTags,
			))

			if c.config.DetailLevel >= StandardMetrics {
				metrics = append(metrics, NewMetric(
					"vm.network.rx_dropped",
					MetricTypeGauge,
					network.RxDropped,
					netTags,
				))

				metrics = append(metrics, NewMetric(
					"vm.network.tx_dropped",
					MetricTypeGauge,
					network.TxDropped,
					netTags,
				))

				metrics = append(metrics, NewMetric(
					"vm.network.rx_errors",
					MetricTypeGauge,
					network.RxErrors,
					netTags,
				))

				metrics = append(metrics, NewMetric(
					"vm.network.tx_errors",
					MetricTypeGauge,
					network.TxErrors,
					netTags,
				))
			}
		}
	}

	// Collect process metrics if enabled and available
	if c.config.EnabledMetrics.ProcessStats && len(stats.Processes) > 0 &&
		c.config.DetailLevel >= DetailedMetrics {
		for _, process := range stats.Processes {
			procTags := copyTags(baseTags)
			procTags["metric_type"] = "process"
			procTags["process_id"] = fmt.Sprintf("%d", process.PID)
			procTags["process_name"] = process.Name
			procTags["process_command"] = process.Command

			metrics = append(metrics, NewMetric(
				"vm.process.cpu_usage",
				MetricTypeGauge,
				process.CPUUsage,
				procTags,
			))

			metrics = append(metrics, NewMetric(
				"vm.process.memory_usage",
				MetricTypeGauge,
				float64(process.MemoryUsage),
				procTags,
			))

			metrics = append(metrics, NewMetric(
				"vm.process.memory_percent",
				MetricTypeGauge,
				process.MemoryPercent,
				procTags,
			))

			if c.config.EnabledMetrics.IOPs {
				metrics = append(metrics, NewMetric(
					"vm.process.read_iops",
					MetricTypeGauge,
					process.ReadIOPS,
					procTags,
				))

				metrics = append(metrics, NewMetric(
					"vm.process.write_iops",
					MetricTypeGauge,
					process.WriteIOPS,
					procTags,
				))

				metrics = append(metrics, NewMetric(
					"vm.process.read_throughput",
					MetricTypeGauge,
					process.ReadThroughput,
					procTags,
				))

				metrics = append(metrics, NewMetric(
					"vm.process.write_throughput",
					MetricTypeGauge,
					process.WriteThroughput,
					procTags,
				))
			}

			metrics = append(metrics, NewMetric(
				"vm.process.open_files",
				MetricTypeGauge,
				float64(process.OpenFiles),
				procTags,
			))

			metrics = append(metrics, NewMetric(
				"vm.process.run_time",
				MetricTypeGauge,
				process.RunTime,
				procTags,
			))
		}
	}

	// Set timestamp for all metrics
	for _, metric := range metrics {
		metric.Timestamp = timestamp
	}

	return metrics, nil
}

// Helper functions

// copyTags creates a copy of a tag map
func copyTags(tags map[string]string) map[string]string {
	if tags == nil {
		return nil
	}
	result := make(map[string]string, len(tags))
	for k, v := range tags {
		result[k] = v
	}
	return result
}
