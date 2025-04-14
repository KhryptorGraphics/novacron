package vm

import (
	"context"
	"fmt"
	"log"
	"sync"
	"time"
)

// VMMetricsType defines the type of VM metrics
type VMMetricsType string

const (
	// VMMetricsTypeCPU represents CPU metrics
	VMMetricsTypeCPU VMMetricsType = "cpu"
	
	// VMMetricsTypeMemory represents memory metrics
	VMMetricsTypeMemory VMMetricsType = "memory"
	
	// VMMetricsTypeDisk represents disk metrics
	VMMetricsTypeDisk VMMetricsType = "disk"
	
	// VMMetricsTypeNetwork represents network metrics
	VMMetricsTypeNetwork VMMetricsType = "network"
)

// VMMetrics represents metrics for a VM
type VMMetrics struct {
	VMID        string                 `json:"vm_id"`
	NodeID      string                 `json:"node_id"`
	Timestamp   time.Time              `json:"timestamp"`
	CPU         CPUMetrics             `json:"cpu"`
	Memory      MemoryMetrics          `json:"memory"`
	Disk        map[string]DiskMetrics `json:"disk"`
	Network     map[string]NetMetrics  `json:"network"`
	Labels      map[string]string      `json:"labels,omitempty"`
	Annotations map[string]string      `json:"annotations,omitempty"`
}

// CPUMetrics represents CPU metrics
type CPUMetrics struct {
	UsagePercent     float64 `json:"usage_percent"`
	SystemPercent    float64 `json:"system_percent"`
	UserPercent      float64 `json:"user_percent"`
	IOWaitPercent    float64 `json:"iowait_percent"`
	StealPercent     float64 `json:"steal_percent"`
	Cores            int     `json:"cores"`
	ThrottledPeriods int64   `json:"throttled_periods"`
	ThrottledTime    int64   `json:"throttled_time"`
}

// MemoryMetrics represents memory metrics
type MemoryMetrics struct {
	TotalBytes     int64   `json:"total_bytes"`
	UsedBytes      int64   `json:"used_bytes"`
	CacheBytes     int64   `json:"cache_bytes"`
	RSSBytes       int64   `json:"rss_bytes"`
	SwapBytes      int64   `json:"swap_bytes"`
	UsagePercent   float64 `json:"usage_percent"`
	SwapPercent    float64 `json:"swap_percent"`
	MajorPageFaults int64   `json:"major_page_faults"`
	MinorPageFaults int64   `json:"minor_page_faults"`
}

// DiskMetrics represents disk metrics
type DiskMetrics struct {
	Device        string  `json:"device"`
	TotalBytes    int64   `json:"total_bytes"`
	UsedBytes     int64   `json:"used_bytes"`
	UsagePercent  float64 `json:"usage_percent"`
	ReadBytes     int64   `json:"read_bytes"`
	WriteBytes    int64   `json:"write_bytes"`
	ReadOps       int64   `json:"read_ops"`
	WriteOps      int64   `json:"write_ops"`
	ReadLatencyMs float64 `json:"read_latency_ms"`
	WriteLatencyMs float64 `json:"write_latency_ms"`
	IOTimeMs      int64   `json:"io_time_ms"`
}

// NetMetrics represents network metrics
type NetMetrics struct {
	Interface     string `json:"interface"`
	RxBytes       int64  `json:"rx_bytes"`
	TxBytes       int64  `json:"tx_bytes"`
	RxPackets     int64  `json:"rx_packets"`
	TxPackets     int64  `json:"tx_packets"`
	RxErrors      int64  `json:"rx_errors"`
	TxErrors      int64  `json:"tx_errors"`
	RxDropped     int64  `json:"rx_dropped"`
	TxDropped     int64  `json:"tx_dropped"`
	RxBytesPerSec float64 `json:"rx_bytes_per_sec"`
	TxBytesPerSec float64 `json:"tx_bytes_per_sec"`
}

// VMMetricsCollectorConfig holds configuration for the VM metrics collector
type VMMetricsCollectorConfig struct {
	CollectionInterval time.Duration `json:"collection_interval"`
	RetentionPeriod    time.Duration `json:"retention_period"`
	MaxSamplesPerVM    int           `json:"max_samples_per_vm"`
	DetailLevel        string        `json:"detail_level"` // "basic", "standard", "detailed"
}

// DefaultVMMetricsCollectorConfig returns a default configuration
func DefaultVMMetricsCollectorConfig() VMMetricsCollectorConfig {
	return VMMetricsCollectorConfig{
		CollectionInterval: 30 * time.Second,
		RetentionPeriod:    24 * time.Hour,
		MaxSamplesPerVM:    1000,
		DetailLevel:        "standard",
	}
}

// VMMetricsCollector collects metrics from VMs
type VMMetricsCollector struct {
	config         VMMetricsCollectorConfig
	nodeID         string
	metrics        map[string][]VMMetrics // VM ID -> metrics
	metricsMutex   sync.RWMutex
	collectors     map[string]VMMetricsProvider // VM ID -> provider
	collectorsMutex sync.RWMutex
	ctx            context.Context
	cancel         context.CancelFunc
}

// VMMetricsProvider is an interface for collecting VM metrics
type VMMetricsProvider interface {
	// GetMetrics returns metrics for a VM
	GetMetrics(ctx context.Context) (*VMMetrics, error)
	
	// GetVMID returns the VM ID
	GetVMID() string
	
	// Close closes the metrics provider
	Close() error
}

// NewVMMetricsCollector creates a new VM metrics collector
func NewVMMetricsCollector(config VMMetricsCollectorConfig, nodeID string) *VMMetricsCollector {
	ctx, cancel := context.WithCancel(context.Background())
	
	collector := &VMMetricsCollector{
		config:     config,
		nodeID:     nodeID,
		metrics:    make(map[string][]VMMetrics),
		collectors: make(map[string]VMMetricsProvider),
		ctx:        ctx,
		cancel:     cancel,
	}
	
	return collector
}

// Start starts the VM metrics collector
func (c *VMMetricsCollector) Start() error {
	log.Println("Starting VM metrics collector")
	
	// Start the collection loop
	go c.collectMetrics()
	
	// Start the cleanup loop
	go c.cleanupMetrics()
	
	return nil
}

// Stop stops the VM metrics collector
func (c *VMMetricsCollector) Stop() error {
	log.Println("Stopping VM metrics collector")
	c.cancel()
	
	// Close all collectors
	c.collectorsMutex.Lock()
	defer c.collectorsMutex.Unlock()
	
	for _, collector := range c.collectors {
		if err := collector.Close(); err != nil {
			log.Printf("Warning: Failed to close metrics collector for VM %s: %v", collector.GetVMID(), err)
		}
	}
	
	return nil
}

// RegisterVM registers a VM for metrics collection
func (c *VMMetricsCollector) RegisterVM(vmID string, provider VMMetricsProvider) error {
	c.collectorsMutex.Lock()
	defer c.collectorsMutex.Unlock()
	
	// Check if the VM is already registered
	if _, exists := c.collectors[vmID]; exists {
		return fmt.Errorf("VM %s is already registered for metrics collection", vmID)
	}
	
	// Register the VM
	c.collectors[vmID] = provider
	
	// Initialize metrics storage
	c.metricsMutex.Lock()
	c.metrics[vmID] = make([]VMMetrics, 0, c.config.MaxSamplesPerVM)
	c.metricsMutex.Unlock()
	
	log.Printf("Registered VM %s for metrics collection", vmID)
	return nil
}

// UnregisterVM unregisters a VM from metrics collection
func (c *VMMetricsCollector) UnregisterVM(vmID string) error {
	c.collectorsMutex.Lock()
	defer c.collectorsMutex.Unlock()
	
	// Check if the VM is registered
	collector, exists := c.collectors[vmID]
	if !exists {
		return fmt.Errorf("VM %s is not registered for metrics collection", vmID)
	}
	
	// Close the collector
	if err := collector.Close(); err != nil {
		log.Printf("Warning: Failed to close metrics collector for VM %s: %v", vmID, err)
	}
	
	// Unregister the VM
	delete(c.collectors, vmID)
	
	// Keep the metrics for now, they will be cleaned up by the cleanup loop
	
	log.Printf("Unregistered VM %s from metrics collection", vmID)
	return nil
}

// GetMetrics returns metrics for a VM
func (c *VMMetricsCollector) GetMetrics(vmID string, limit int) ([]VMMetrics, error) {
	c.metricsMutex.RLock()
	defer c.metricsMutex.RUnlock()
	
	// Check if the VM has metrics
	metrics, exists := c.metrics[vmID]
	if !exists {
		return nil, fmt.Errorf("no metrics found for VM %s", vmID)
	}
	
	// If limit is 0 or negative, return all metrics
	if limit <= 0 {
		return metrics, nil
	}
	
	// If limit is greater than the number of metrics, return all metrics
	if limit >= len(metrics) {
		return metrics, nil
	}
	
	// Return the most recent metrics
	return metrics[len(metrics)-limit:], nil
}

// GetLatestMetrics returns the latest metrics for a VM
func (c *VMMetricsCollector) GetLatestMetrics(vmID string) (*VMMetrics, error) {
	c.metricsMutex.RLock()
	defer c.metricsMutex.RUnlock()
	
	// Check if the VM has metrics
	metrics, exists := c.metrics[vmID]
	if !exists {
		return nil, fmt.Errorf("no metrics found for VM %s", vmID)
	}
	
	// Check if there are any metrics
	if len(metrics) == 0 {
		return nil, fmt.Errorf("no metrics found for VM %s", vmID)
	}
	
	// Return the latest metrics
	return &metrics[len(metrics)-1], nil
}

// collectMetrics collects metrics from all registered VMs
func (c *VMMetricsCollector) collectMetrics() {
	ticker := time.NewTicker(c.config.CollectionInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-c.ctx.Done():
			return
		case <-ticker.C:
			c.collectAllMetrics()
		}
	}
}

// collectAllMetrics collects metrics from all registered VMs
func (c *VMMetricsCollector) collectAllMetrics() {
	// Get a copy of the collectors map to avoid holding the lock during collection
	c.collectorsMutex.RLock()
	collectors := make(map[string]VMMetricsProvider, len(c.collectors))
	for vmID, collector := range c.collectors {
		collectors[vmID] = collector
	}
	c.collectorsMutex.RUnlock()
	
	// Collect metrics from each VM
	for vmID, collector := range collectors {
		// Create a context with timeout for the collection
		ctx, cancel := context.WithTimeout(c.ctx, c.config.CollectionInterval/2)
		
		// Collect metrics
		metrics, err := collector.GetMetrics(ctx)
		if err != nil {
			log.Printf("Warning: Failed to collect metrics for VM %s: %v", vmID, err)
			cancel()
			continue
		}
		
		// Store the metrics
		c.storeMetrics(vmID, metrics)
		
		cancel()
	}
}

// storeMetrics stores metrics for a VM
func (c *VMMetricsCollector) storeMetrics(vmID string, metrics *VMMetrics) {
	c.metricsMutex.Lock()
	defer c.metricsMutex.Unlock()
	
	// Check if the VM has metrics storage
	if _, exists := c.metrics[vmID]; !exists {
		c.metrics[vmID] = make([]VMMetrics, 0, c.config.MaxSamplesPerVM)
	}
	
	// Add the metrics
	c.metrics[vmID] = append(c.metrics[vmID], *metrics)
	
	// If we have too many metrics, remove the oldest ones
	if len(c.metrics[vmID]) > c.config.MaxSamplesPerVM {
		c.metrics[vmID] = c.metrics[vmID][len(c.metrics[vmID])-c.config.MaxSamplesPerVM:]
	}
}

// cleanupMetrics removes old metrics
func (c *VMMetricsCollector) cleanupMetrics() {
	ticker := time.NewTicker(c.config.RetentionPeriod / 10)
	defer ticker.Stop()
	
	for {
		select {
		case <-c.ctx.Done():
			return
		case <-ticker.C:
			c.removeOldMetrics()
		}
	}
}

// removeOldMetrics removes metrics older than the retention period
func (c *VMMetricsCollector) removeOldMetrics() {
	c.metricsMutex.Lock()
	defer c.metricsMutex.Unlock()
	
	// Get the cutoff time
	cutoff := time.Now().Add(-c.config.RetentionPeriod)
	
	// Check each VM
	for vmID, metrics := range c.metrics {
		// Find the index of the first metric to keep
		keepIndex := 0
		for i, metric := range metrics {
			if metric.Timestamp.After(cutoff) {
				keepIndex = i
				break
			}
		}
		
		// If all metrics are newer than the cutoff, continue
		if keepIndex == 0 {
			continue
		}
		
		// If all metrics are older than the cutoff, remove them all
		if keepIndex == len(metrics) {
			delete(c.metrics, vmID)
			continue
		}
		
		// Remove old metrics
		c.metrics[vmID] = metrics[keepIndex:]
	}
	
	// Check for VMs that are no longer registered
	c.collectorsMutex.RLock()
	defer c.collectorsMutex.RUnlock()
	
	for vmID := range c.metrics {
		if _, exists := c.collectors[vmID]; !exists {
			// VM is no longer registered, check if it has recent metrics
			metrics := c.metrics[vmID]
			if len(metrics) == 0 || metrics[len(metrics)-1].Timestamp.Before(cutoff) {
				// No recent metrics, remove the VM
				delete(c.metrics, vmID)
			}
		}
	}
}
