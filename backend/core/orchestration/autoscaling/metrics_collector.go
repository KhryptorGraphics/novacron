package autoscaling

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/sirupsen/logrus"
)

// DefaultMetricsCollector implements the MetricsCollector interface
type DefaultMetricsCollector struct {
	mu              sync.RWMutex
	logger          *logrus.Logger
	metricsStore    map[string][]*MetricsData // targetID -> metrics history
	subscribers     []MetricsHandler
	collectInterval time.Duration
	maxHistorySize  int
	ctx             context.Context
	cancel          context.CancelFunc
	running         bool
}

// NewDefaultMetricsCollector creates a new metrics collector
func NewDefaultMetricsCollector(logger *logrus.Logger) *DefaultMetricsCollector {
	ctx, cancel := context.WithCancel(context.Background())
	
	return &DefaultMetricsCollector{
		logger:          logger,
		metricsStore:    make(map[string][]*MetricsData),
		subscribers:     make([]MetricsHandler, 0),
		collectInterval: 30 * time.Second,
		maxHistorySize:  1440, // 24 hours at 1-minute intervals
		ctx:             ctx,
		cancel:          cancel,
	}
}

// CollectMetrics collects current metrics from the system
func (c *DefaultMetricsCollector) CollectMetrics() (*MetricsData, error) {
	// Simulate metric collection - in real implementation, this would
	// connect to monitoring systems like Prometheus, InfluxDB, etc.
	
	metrics := &MetricsData{
		Timestamp:   time.Now(),
		TargetID:    "cluster-default",
		TargetType:  "cluster",
		CPUUsage:    c.simulateCPUUsage(),
		MemoryUsage: c.simulateMemoryUsage(),
		NetworkIO:   c.simulateNetworkIO(),
		DiskIO:      c.simulateDiskIO(),
		ActiveVMs:   c.simulateActiveVMs(),
		CustomMetrics: map[string]float64{
			"queue_length":   float64(c.simulateQueueLength()),
			"response_time":  c.simulateResponseTime(),
		},
	}

	// Store metrics in history
	c.storeMetrics(metrics)

	// Notify subscribers
	c.notifySubscribers(metrics)

	c.logger.WithFields(logrus.Fields{
		"target_id":    metrics.TargetID,
		"cpu_usage":    metrics.CPUUsage,
		"memory_usage": metrics.MemoryUsage,
		"timestamp":    metrics.Timestamp,
	}).Debug("Metrics collected")

	return metrics, nil
}

// GetHistoricalMetrics gets historical metrics for a time range
func (c *DefaultMetricsCollector) GetHistoricalMetrics(start, end time.Time) ([]*MetricsData, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	var result []*MetricsData
	
	// For simplicity, return all metrics for cluster-default
	// In real implementation, this would filter by target ID and time range
	targetID := "cluster-default"
	
	if metrics, exists := c.metricsStore[targetID]; exists {
		for _, m := range metrics {
			if (m.Timestamp.After(start) || m.Timestamp.Equal(start)) && 
			   (m.Timestamp.Before(end) || m.Timestamp.Equal(end)) {
				result = append(result, m)
			}
		}
	}

	c.logger.WithFields(logrus.Fields{
		"target_id": targetID,
		"start":     start,
		"end":       end,
		"count":     len(result),
	}).Debug("Historical metrics retrieved")

	return result, nil
}

// Subscribe subscribes to real-time metrics updates
func (c *DefaultMetricsCollector) Subscribe(handler MetricsHandler) error {
	if handler == nil {
		return fmt.Errorf("handler cannot be nil")
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	c.subscribers = append(c.subscribers, handler)
	
	c.logger.WithField("subscribers_count", len(c.subscribers)).Info("New subscriber added")
	
	return nil
}

// StartCollection starts the metrics collection process
func (c *DefaultMetricsCollector) StartCollection() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.running {
		return fmt.Errorf("metrics collection already running")
	}

	c.running = true
	go c.collectLoop()

	c.logger.WithField("interval", c.collectInterval).Info("Metrics collection started")
	
	return nil
}

// StopCollection stops the metrics collection process
func (c *DefaultMetricsCollector) StopCollection() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if !c.running {
		return fmt.Errorf("metrics collection not running")
	}

	c.cancel()
	c.running = false

	c.logger.Info("Metrics collection stopped")
	
	return nil
}

// SetCollectionInterval sets the metrics collection interval
func (c *DefaultMetricsCollector) SetCollectionInterval(interval time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.collectInterval = interval
	
	c.logger.WithField("new_interval", interval).Info("Collection interval updated")
}

// GetMetricsHistory returns the current metrics history
func (c *DefaultMetricsCollector) GetMetricsHistory(targetID string) ([]*MetricsData, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if metrics, exists := c.metricsStore[targetID]; exists {
		// Return a copy to avoid race conditions
		result := make([]*MetricsData, len(metrics))
		copy(result, metrics)
		return result, nil
	}

	return nil, fmt.Errorf("no metrics found for target %s", targetID)
}

// Private methods

func (c *DefaultMetricsCollector) collectLoop() {
	ticker := time.NewTicker(c.collectInterval)
	defer ticker.Stop()

	for {
		select {
		case <-c.ctx.Done():
			return
		case <-ticker.C:
			if _, err := c.CollectMetrics(); err != nil {
				c.logger.WithError(err).Error("Failed to collect metrics")
			}
		}
	}
}

func (c *DefaultMetricsCollector) storeMetrics(metrics *MetricsData) {
	c.mu.Lock()
	defer c.mu.Unlock()

	targetID := metrics.TargetID
	
	// Initialize metrics array for target if not exists
	if _, exists := c.metricsStore[targetID]; !exists {
		c.metricsStore[targetID] = make([]*MetricsData, 0, c.maxHistorySize)
	}

	// Add new metrics
	c.metricsStore[targetID] = append(c.metricsStore[targetID], metrics)

	// Trim history if too large
	if len(c.metricsStore[targetID]) > c.maxHistorySize {
		// Remove oldest entries
		excess := len(c.metricsStore[targetID]) - c.maxHistorySize
		c.metricsStore[targetID] = c.metricsStore[targetID][excess:]
	}
}

func (c *DefaultMetricsCollector) notifySubscribers(metrics *MetricsData) {
	c.mu.RLock()
	subscribers := make([]MetricsHandler, len(c.subscribers))
	copy(subscribers, c.subscribers)
	c.mu.RUnlock()

	for _, handler := range subscribers {
		go func(h MetricsHandler) {
			if err := h.HandleMetrics(metrics); err != nil {
				c.logger.WithError(err).Error("Subscriber failed to handle metrics")
			}
		}(handler)
	}
}

// Simulation methods - in real implementation, these would query actual systems

func (c *DefaultMetricsCollector) simulateCPUUsage() float64 {
	// Simulate CPU usage with some patterns
	hour := time.Now().Hour()
	base := 0.3

	// Business hours pattern
	if hour >= 9 && hour <= 17 {
		base = 0.6
	} else if hour >= 18 && hour <= 22 {
		base = 0.4
	}

	// Add some randomness
	variation := (c.randomFloat() - 0.5) * 0.2
	usage := base + variation

	// Ensure within valid range
	if usage < 0 {
		usage = 0
	} else if usage > 1 {
		usage = 1
	}

	return usage
}

func (c *DefaultMetricsCollector) simulateMemoryUsage() float64 {
	// Memory usage typically correlates with CPU but is more stable
	cpuUsage := c.simulateCPUUsage()
	memUsage := cpuUsage * 0.8 + 0.1 // Usually 80% of CPU pattern plus base

	variation := (c.randomFloat() - 0.5) * 0.1
	memUsage += variation

	if memUsage < 0 {
		memUsage = 0
	} else if memUsage > 1 {
		memUsage = 1
	}

	return memUsage
}

func (c *DefaultMetricsCollector) simulateNetworkIO() float64 {
	// Network IO in MB/s
	base := 10.0
	variation := (c.randomFloat() - 0.5) * 5.0
	
	return base + variation
}

func (c *DefaultMetricsCollector) simulateDiskIO() float64 {
	// Disk IO in IOPS
	base := 100.0
	variation := (c.randomFloat() - 0.5) * 50.0
	
	return base + variation
}

func (c *DefaultMetricsCollector) simulateActiveVMs() int {
	base := 10
	variation := int((c.randomFloat() - 0.5) * 4)
	
	result := base + variation
	if result < 1 {
		result = 1
	}
	
	return result
}

func (c *DefaultMetricsCollector) simulateQueueLength() int {
	cpuUsage := c.simulateCPUUsage()
	queueLength := int(cpuUsage * 20) // Higher CPU = longer queue
	
	return queueLength
}

func (c *DefaultMetricsCollector) simulateResponseTime() float64 {
	cpuUsage := c.simulateCPUUsage()
	responseTime := 50.0 + (cpuUsage * 200.0) // Base 50ms + up to 200ms based on load
	
	return responseTime
}

func (c *DefaultMetricsCollector) randomFloat() float64 {
	// Simple pseudo-random number generator
	// In production, use a proper random number generator
	return float64(time.Now().UnixNano()%1000) / 1000.0
}

// MetricsHandlerFunc is a function adapter for MetricsHandler
type MetricsHandlerFunc func(metrics *MetricsData) error

// HandleMetrics implements MetricsHandler interface
func (f MetricsHandlerFunc) HandleMetrics(metrics *MetricsData) error {
	return f(metrics)
}