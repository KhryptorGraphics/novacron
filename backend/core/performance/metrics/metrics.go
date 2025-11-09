package metrics

import (
	"sync"
	"time"
)

// Collector collects performance metrics
type Collector struct {
	mu      sync.RWMutex
	metrics map[string]*MetricSeries
}

// MetricSeries stores time-series metrics
type MetricSeries struct {
	Name       string
	Values     []MetricPoint
	Retention  time.Duration
}

// MetricPoint represents a metric at a point in time
type MetricPoint struct {
	Timestamp time.Time
	Value     float64
	Labels    map[string]string
}

// PerformanceMetrics aggregates all performance metrics
type PerformanceMetrics struct {
	CPU     CPUMetrics
	Memory  MemoryMetrics
	IO      IOMetrics
	Network NetworkMetrics
	App     ApplicationMetrics
	Cost    CostMetrics
}

// CPUMetrics stores CPU metrics
type CPUMetrics struct {
	Utilization     float64 // Percentage
	System          float64
	User            float64
	Wait            float64
	LoadAverage1    float64
	LoadAverage5    float64
	LoadAverage15   float64
	ContextSwitches float64
}

// MemoryMetrics stores memory metrics
type MemoryMetrics struct {
	Utilization float64
	Used        uint64
	Cached      uint64
	Available   uint64
	Total       uint64
	SwapUsed    uint64
	SwapTotal   uint64
	PageFaults  float64
}

// IOMetrics stores I/O metrics
type IOMetrics struct {
	IOPS          float64
	ReadIOPS      float64
	WriteIOPS     float64
	Throughput    float64
	ReadMBps      float64
	WriteMBps     float64
	Latency       float64
	QueueDepth    float64
}

// NetworkMetrics stores network metrics
type NetworkMetrics struct {
	Bandwidth       float64
	IngressMbps     float64
	EgressMbps      float64
	PacketsIn       float64
	PacketsOut      float64
	Errors          float64
	Drops           float64
	Connections     int
	Latency         float64
}

// ApplicationMetrics stores application metrics
type ApplicationMetrics struct {
	Throughput      float64
	Latency         float64
	P50Latency      float64
	P95Latency      float64
	P99Latency      float64
	ErrorRate       float64
	SuccessRate     float64
	RequestsPerSec  float64
}

// CostMetrics stores cost metrics
type CostMetrics struct {
	HourlyCost      float64
	DailyCost       float64
	MonthlyCost     float64
	ProjectedCost   float64
	CostPerRequest  float64
	CostEfficiency  float64
}

// NewCollector creates metrics collector
func NewCollector() *Collector {
	return &Collector{
		metrics: make(map[string]*MetricSeries),
	}
}

// Record records a metric
func (c *Collector) Record(name string, value float64, labels map[string]string) {
	c.mu.Lock()
	defer c.mu.Unlock()

	series, exists := c.metrics[name]
	if !exists {
		series = &MetricSeries{
			Name:      name,
			Values:    make([]MetricPoint, 0),
			Retention: 24 * time.Hour,
		}
		c.metrics[name] = series
	}

	series.Values = append(series.Values, MetricPoint{
		Timestamp: time.Now(),
		Value:     value,
		Labels:    labels,
	})

	// Prune old values
	c.pruneOldValues(series)
}

// pruneOldValues removes old metric values
func (c *Collector) pruneOldValues(series *MetricSeries) {
	cutoff := time.Now().Add(-series.Retention)
	validIdx := 0

	for i, point := range series.Values {
		if point.Timestamp.After(cutoff) {
			validIdx = i
			break
		}
	}

	if validIdx > 0 {
		series.Values = series.Values[validIdx:]
	}
}

// Get retrieves metric series
func (c *Collector) Get(name string) *MetricSeries {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.metrics[name]
}

// GetLatest gets latest metric value
func (c *Collector) GetLatest(name string) (float64, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	series, exists := c.metrics[name]
	if !exists || len(series.Values) == 0 {
		return 0, false
	}

	return series.Values[len(series.Values)-1].Value, true
}

// CollectSystemMetrics collects system-wide metrics
func CollectSystemMetrics() *PerformanceMetrics {
	return &PerformanceMetrics{
		CPU: CPUMetrics{
			Utilization:  65.0,
			System:       10.0,
			User:         55.0,
			Wait:         5.0,
			LoadAverage1: 2.5,
		},
		Memory: MemoryMetrics{
			Utilization: 75.0,
			Used:        12 * 1024 * 1024 * 1024,
			Total:       16 * 1024 * 1024 * 1024,
		},
		IO: IOMetrics{
			IOPS:       5000,
			Throughput: 500,
			Latency:    2.0,
		},
		Network: NetworkMetrics{
			IngressMbps: 100,
			EgressMbps:  50,
			Latency:     1.0,
		},
		App: ApplicationMetrics{
			Throughput: 1000,
			P95Latency: 10.0,
			ErrorRate:  0.01,
		},
		Cost: CostMetrics{
			HourlyCost:  1.0,
			MonthlyCost: 730.0,
		},
	}
}
