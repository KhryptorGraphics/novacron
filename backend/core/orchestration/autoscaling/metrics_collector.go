package autoscaling

import (
	"context"
	"fmt"
	"sort"
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
	source          MetricsSource
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

// MetricsSource returns one measured sample. Nil and errors are not stored
// and are not delivered to subscribers.
type MetricsSource func() (*MetricsData, error)

// SetSource installs the only sample the collector will publish.
func (c *DefaultMetricsCollector) SetSource(source MetricsSource) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.source = source
}

// CollectMetrics collects one measured sample and publishes it.
// Without a source it returns an error and does not notify subscribers.
func (c *DefaultMetricsCollector) CollectMetrics() (*MetricsData, error) {
	c.mu.RLock()
	source := c.source
	c.mu.RUnlock()
	if source == nil {
		return nil, fmt.Errorf("metrics source is not configured")
	}

	metrics, err := source()
	if err != nil {
		return nil, err
	}
	if metrics == nil {
		return nil, fmt.Errorf("metrics source returned no sample")
	}
	if metrics.Timestamp.IsZero() {
		metrics.Timestamp = time.Now()
	}

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
	for _, metrics := range c.metricsStore {
		for _, m := range metrics {
			if (m.Timestamp.After(start) || m.Timestamp.Equal(start)) &&
				(m.Timestamp.Before(end) || m.Timestamp.Equal(end)) {
				result = append(result, m)
			}
		}
	}
	sort.Slice(result, func(i, j int) bool {
		return result[i].Timestamp.Before(result[j].Timestamp)
	})

	c.logger.WithFields(logrus.Fields{
		"start": start,
		"end":   end,
		"count": len(result),
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
	if c.source == nil {
		return fmt.Errorf("metrics source is not configured")
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

// MetricsHandlerFunc is a function adapter for MetricsHandler
type MetricsHandlerFunc func(metrics *MetricsData) error

// HandleMetrics implements MetricsHandler interface
func (f MetricsHandlerFunc) HandleMetrics(metrics *MetricsData) error {
	return f(metrics)
}
