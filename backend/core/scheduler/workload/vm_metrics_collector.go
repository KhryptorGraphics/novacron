package workload

import (
	"context"
	"fmt"
	"log"
	"math/rand" // For demo purposes only
	"time"
)

// VMMetricsCollector collects resource usage metrics from VMs
type VMMetricsCollector struct {
	// vmManager would normally be a reference to the VM manager component
	// that allows querying VM status. For now, we'll simulate metrics.
	vmManager interface{}

	// supportedMetrics lists the metrics this collector can provide
	supportedMetrics []string

	// refreshInterval is how often the metrics are refreshed
	refreshInterval time.Duration

	// cachedMetrics stores the last collected metrics for each VM
	cachedMetrics map[string]map[string]float64

	// sampleThreshold determines the threshold for detection of variability
	sampleThreshold float64
}

// NewVMMetricsCollector creates a new VM metrics collector
func NewVMMetricsCollector(vmManager interface{}, refreshInterval time.Duration) *VMMetricsCollector {
	return &VMMetricsCollector{
		vmManager: vmManager,
		supportedMetrics: []string{
			"cpu_usage",        // Percentage of CPU usage (0-100)
			"memory_usage",     // Percentage of memory usage (0-100)
			"disk_io",          // Disk I/O operations per second
			"network_io",       // Network I/O bytes per second
			"disk_usage",       // Percentage of disk usage (0-100)
			"page_faults",      // Page faults per second
			"context_switches", // Context switches per second
		},
		refreshInterval: refreshInterval,
		cachedMetrics:   make(map[string]map[string]float64),
		sampleThreshold: 0.1,
	}
}

// GetSupportedMetrics returns the list of supported metrics
func (c *VMMetricsCollector) GetSupportedMetrics() []string {
	return c.supportedMetrics
}

// Start starts the metrics collection
func (c *VMMetricsCollector) Start(ctx context.Context) {
	go c.collectionLoop(ctx)
}

// collectionLoop periodically refreshes metrics
func (c *VMMetricsCollector) collectionLoop(ctx context.Context) {
	ticker := time.NewTicker(c.refreshInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			c.refreshAllVMMetrics()
		}
	}
}

// refreshAllVMMetrics refreshes metrics for all VMs
// In a real implementation, this would query the VM manager
func (c *VMMetricsCollector) refreshAllVMMetrics() {
	// In a real implementation, we would get the list of VMs from the VM manager
	// For now, we'll just maintain metrics for VMs we've seen
	for vmID := range c.cachedMetrics {
		c.refreshVMMetrics(vmID)
	}
}

// refreshVMMetrics refreshes metrics for a specific VM
func (c *VMMetricsCollector) refreshVMMetrics(vmID string) {
	// In a real implementation, we would query the VM manager for metrics
	// For now, we'll simulate metrics

	// Initialize metrics for this VM if needed
	if _, exists := c.cachedMetrics[vmID]; !exists {
		c.cachedMetrics[vmID] = make(map[string]float64)
	}

	// Sample resource usage pattern by VM ID to get consistent behavior
	// This is just for simulation purposes
	vmIDSeed := int64(0)
	for i := 0; i < len(vmID); i++ {
		vmIDSeed += int64(vmID[i])
	}
	rand.Seed(vmIDSeed)

	// Define VM type based on hash of VM ID
	var cpuBaseline, memoryBaseline, diskIOBaseline, networkIOBaseline float64
	var cpuVariation, memoryVariation, diskIOVariation, networkIOVariation float64

	vmType := vmIDSeed % 5
	switch vmType {
	case 0: // CPU intensive
		cpuBaseline = 70
		memoryBaseline = 30
		diskIOBaseline = 20
		networkIOBaseline = 15
		cpuVariation = 20
		memoryVariation = 10
		diskIOVariation = 5
		networkIOVariation = 5
	case 1: // Memory intensive
		cpuBaseline = 30
		memoryBaseline = 75
		diskIOBaseline = 25
		networkIOBaseline = 10
		cpuVariation = 10
		memoryVariation = 15
		diskIOVariation = 10
		networkIOVariation = 5
	case 2: // I/O intensive
		cpuBaseline = 25
		memoryBaseline = 40
		diskIOBaseline = 80
		networkIOBaseline = 30
		cpuVariation = 10
		memoryVariation = 10
		diskIOVariation = 15
		networkIOVariation = 10
	case 3: // Network intensive
		cpuBaseline = 30
		memoryBaseline = 35
		diskIOBaseline = 20
		networkIOBaseline = 75
		cpuVariation = 10
		memoryVariation = 10
		diskIOVariation = 5
		networkIOVariation = 20
	default: // Balanced
		cpuBaseline = 40
		memoryBaseline = 45
		diskIOBaseline = 35
		networkIOBaseline = 40
		cpuVariation = 15
		memoryVariation = 15
		diskIOVariation = 10
		networkIOVariation = 15
	}

	// Update metrics with some randomness to simulate real-world variations
	// Re-seed for true randomness in the fluctuations
	rand.Seed(time.Now().UnixNano())

	c.cachedMetrics[vmID]["cpu_usage"] = getRandomMetric(cpuBaseline, cpuVariation)
	c.cachedMetrics[vmID]["memory_usage"] = getRandomMetric(memoryBaseline, memoryVariation)
	c.cachedMetrics[vmID]["disk_io"] = getRandomMetric(diskIOBaseline, diskIOVariation)
	c.cachedMetrics[vmID]["network_io"] = getRandomMetric(networkIOBaseline, networkIOVariation)
	c.cachedMetrics[vmID]["disk_usage"] = getRandomMetric(50, 10)          // Assume around 50% disk usage
	c.cachedMetrics[vmID]["page_faults"] = getRandomMetric(100, 50)        // Arbitrary units
	c.cachedMetrics[vmID]["context_switches"] = getRandomMetric(1000, 500) // Arbitrary units
}

// GetResourceUsage returns resource usage metrics for a VM
func (c *VMMetricsCollector) GetResourceUsage(ctx context.Context, vmID string) (map[string]float64, error) {
	// Check if metrics exist for this VM
	if _, exists := c.cachedMetrics[vmID]; !exists {
		// First time seeing this VM, initialize and collect metrics
		c.refreshVMMetrics(vmID)
		log.Printf("Initialized metrics collection for VM %s", vmID)
	}

	// Return a copy of the metrics
	metrics := make(map[string]float64)
	for k, v := range c.cachedMetrics[vmID] {
		metrics[k] = v
	}

	return metrics, nil
}

// getRandomMetric returns a random metric value within the given range
func getRandomMetric(baseline, variation float64) float64 {
	min := baseline - variation/2
	if min < 0 {
		min = 0
	}

	max := baseline + variation/2
	if max > 100 && (baseline <= 100) { // Only cap percentage metrics at 100
		max = 100
	}

	return min + rand.Float64()*(max-min)
}

// RegisterVM explicitly registers a VM for metrics collection
func (c *VMMetricsCollector) RegisterVM(vmID string) {
	if _, exists := c.cachedMetrics[vmID]; !exists {
		c.cachedMetrics[vmID] = make(map[string]float64)
		c.refreshVMMetrics(vmID)
		log.Printf("Registered VM %s for metrics collection", vmID)
	}
}

// UnregisterVM stops collecting metrics for a VM
func (c *VMMetricsCollector) UnregisterVM(vmID string) {
	delete(c.cachedMetrics, vmID)
	log.Printf("Unregistered VM %s from metrics collection", vmID)
}

// GetResourceTrend analyzes the trend for a specific resource metric
// This would normally use historical data, but we're simplifying here
func (c *VMMetricsCollector) GetResourceTrend(ctx context.Context, vmID string, metric string, duration time.Duration) (ResourceTrend, error) {
	// Check if the VM exists
	if _, exists := c.cachedMetrics[vmID]; !exists {
		return ResourceTrend{}, fmt.Errorf("no metrics for VM %s", vmID)
	}

	// Check if the metric exists
	if _, exists := c.cachedMetrics[vmID][metric]; !exists {
		return ResourceTrend{}, fmt.Errorf("metric %s not available for VM %s", metric, vmID)
	}

	// In a real implementation, we would analyze historical data
	// For now, we'll return simulated trends
	current := c.cachedMetrics[vmID][metric]

	// Simulate different trends based on the VM ID and metric
	// This is just for demonstration
	rand.Seed(time.Now().UnixNano())
	trend := ResourceTrend{
		MetricName:    metric,
		CurrentValue:  current,
		AverageValue:  current * (0.9 + rand.Float64()*0.2), // Slightly different from current
		MinValue:      current * (0.7 + rand.Float64()*0.2), // Lower than current
		MaxValue:      current * (1.1 + rand.Float64()*0.2), // Higher than current
		TrendSlope:    -0.5 + rand.Float64(),                // Random slope between -0.5 and 0.5
		Variability:   rand.Float64() * 0.5,                 // Random variability between 0 and 0.5
		DataPoints:    10,                                   // Simulate having 10 data points
		TimeRange:     duration,
		PredictedNext: current * (0.95 + rand.Float64()*0.1), // Slight variation for prediction
	}

	return trend, nil
}

// ResourceTrend represents the trend of a resource metric over time
type ResourceTrend struct {
	// MetricName is the name of the metric
	MetricName string

	// CurrentValue is the current value of the metric
	CurrentValue float64

	// AverageValue is the average value over the time period
	AverageValue float64

	// MinValue is the minimum value over the time period
	MinValue float64

	// MaxValue is the maximum value over the time period
	MaxValue float64

	// TrendSlope indicates the direction and strength of the trend
	// Positive values mean increasing usage, negative means decreasing
	TrendSlope float64

	// Variability indicates how much the metric varies (0-1)
	Variability float64

	// DataPoints is the number of data points used for the analysis
	DataPoints int

	// TimeRange is the time period over which the trend was analyzed
	TimeRange time.Duration

	// PredictedNext is the predicted value for the next time period
	PredictedNext float64
}
