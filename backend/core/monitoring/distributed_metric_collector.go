package monitoring

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/storage"
)

const (
	// DefaultCollectionInterval is the default interval for metric collection
	DefaultCollectionInterval = 30 * time.Second

	// DefaultRetentionPeriod is the default period to retain metrics
	DefaultRetentionPeriod = 30 * 24 * time.Hour // 30 days
)

// DistributedMetricCollectorConfig contains configuration for the distributed metric collector
type DistributedMetricCollectorConfig struct {
	// CollectionInterval is how often metrics are collected
	CollectionInterval time.Duration

	// RetentionPeriod is how long metrics are retained
	RetentionPeriod time.Duration

	// StoragePath is where metrics are stored
	StoragePath string

	// Collectors is the list of metric collectors to use
	Collectors []MetricCollector

	// EnableAggregation enables cluster-wide metric aggregation
	EnableAggregation bool

	// AggregationEndpoints is the list of endpoints to send aggregated metrics
	AggregationEndpoints []string

	// NodeID is the unique identifier for this node
	NodeID string

	// ClusterID is the identifier for the cluster
	ClusterID string

	// Tags are additional metadata tags for metrics
	Tags map[string]string
}

// DefaultDistributedMetricCollectorConfig returns the default configuration
func DefaultDistributedMetricCollectorConfig() *DistributedMetricCollectorConfig {
	return &DistributedMetricCollectorConfig{
		CollectionInterval:   DefaultCollectionInterval,
		RetentionPeriod:      DefaultRetentionPeriod,
		StoragePath:          "metrics",
		Collectors:           make([]MetricCollector, 0),
		EnableAggregation:    true,
		AggregationEndpoints: make([]string, 0),
		Tags:                 make(map[string]string),
	}
}

// DistributedMetricCollector collects metrics across a distributed cluster
type DistributedMetricCollector struct {
	config *DistributedMetricCollectorConfig

	// Metric storage
	storage *storage.InMemoryStorage

	// Metric aggregation
	aggregator *MetricAggregator

	// Collection state
	running      bool
	collectMutex sync.Mutex
	stopChan     chan struct{}
	collectors   []MetricCollector

	// Alert manager
	alertManager *AlertManager

	// Cache of recent metrics
	metricCache      map[string]*MetricSeries
	metricCacheMutex sync.RWMutex
}

// NewDistributedMetricCollector creates a new distributed metric collector
func NewDistributedMetricCollector(config *DistributedMetricCollectorConfig, storage *storage.InMemoryStorage) *DistributedMetricCollector {
	if config == nil {
		config = DefaultDistributedMetricCollectorConfig()
	}

	collector := &DistributedMetricCollector{
		config:      config,
		storage:     storage,
		collectors:  config.Collectors,
		stopChan:    make(chan struct{}),
		metricCache: make(map[string]*MetricSeries),
	}

	// Initialize the aggregator if enabled
	if config.EnableAggregation {
		collector.aggregator = NewMetricAggregator(config.NodeID, config.ClusterID, config.AggregationEndpoints)
	}

	// Initialize the alert manager
	collector.alertManager = NewAlertManager(collector)

	return collector
}

// Start begins metric collection
func (d *DistributedMetricCollector) Start() error {
	d.collectMutex.Lock()
	defer d.collectMutex.Unlock()

	if d.running {
		return fmt.Errorf("metric collector already running")
	}

	d.running = true
	d.stopChan = make(chan struct{})

	// Start the aggregator if enabled
	if d.config.EnableAggregation && d.aggregator != nil {
		d.aggregator.Start()
	}

	// Start the alert manager
	d.alertManager.Start()

	// Start the collection goroutine
	go d.collectMetrics()

	return nil
}

// Stop halts metric collection
func (d *DistributedMetricCollector) Stop() error {
	d.collectMutex.Lock()
	defer d.collectMutex.Unlock()

	if !d.running {
		return nil
	}

	d.running = false
	close(d.stopChan)

	// Stop the aggregator if enabled
	if d.config.EnableAggregation && d.aggregator != nil {
		d.aggregator.Stop()
	}

	// Stop the alert manager
	d.alertManager.Stop()

	return nil
}

// AddCollector adds a metric collector
func (d *DistributedMetricCollector) AddCollector(collector MetricCollector) {
	d.collectMutex.Lock()
	defer d.collectMutex.Unlock()

	d.collectors = append(d.collectors, collector)
}

// RemoveCollector removes a metric collector
func (d *DistributedMetricCollector) RemoveCollector(collectorID string) bool {
	d.collectMutex.Lock()
	defer d.collectMutex.Unlock()

	for i, collector := range d.collectors {
		if collector.ID() == collectorID {
			d.collectors = append(d.collectors[:i], d.collectors[i+1:]...)
			return true
		}
	}
	return false
}

// GetMetric retrieves a metric by name and tags
func (d *DistributedMetricCollector) GetMetric(ctx context.Context, name string, tags map[string]string, start, end time.Time) (*MetricSeries, error) {
	// Try cache first for recent metrics
	cacheKey := formatMetricCacheKey(name, tags)

	d.metricCacheMutex.RLock()
	cachedSeries, exists := d.metricCache[cacheKey]
	d.metricCacheMutex.RUnlock()

	// If we have a cached series that covers the requested time range, return it
	if exists && cachedSeries.Covers(start, end) {
		return cachedSeries.Slice(start, end), nil
	}

	// Otherwise fetch from storage
	metricKey := formatMetricStorageKey(name, tags)
	data, err := d.storage.Get(ctx, metricKey)
	if err != nil {
		return nil, fmt.Errorf("failed to get metric from storage: %w", err)
	}

	// Parse the metric data
	series, err := ParseMetricSeries(data)
	if err != nil {
		return nil, fmt.Errorf("failed to parse metric data: %w", err)
	}

	// Cache the series for future use
	d.metricCacheMutex.Lock()
	d.metricCache[cacheKey] = series
	d.metricCacheMutex.Unlock()

	// Return the requested slice
	return series.Slice(start, end), nil
}

// QueryMetrics performs a query across metrics
func (d *DistributedMetricCollector) QueryMetrics(ctx context.Context, query MetricQuery) ([]*MetricSeries, error) {
	// For now, we'll implement a simple query that fetches metrics matching the pattern
	// In a real implementation, this would use a more sophisticated query engine

	// Query the storage for matching metrics
	pattern := query.Pattern
	if pattern == "" {
		pattern = "*"
	}

	keys, err := d.storage.List(ctx, pattern)
	if err != nil {
		return nil, fmt.Errorf("failed to list metrics: %w", err)
	}

	var results []*MetricSeries
	for _, key := range keys {
		// Fetch the metric
		data, err := d.storage.Get(ctx, key)
		if err != nil {
			// Log and continue
			fmt.Printf("failed to get metric %s: %v\n", key, err)
			continue
		}

		// Parse the metric
		series, err := ParseMetricSeries(data)
		if err != nil {
			// Log and continue
			fmt.Printf("failed to parse metric %s: %v\n", key, err)
			continue
		}

		// Apply time range filter
		if !query.Start.IsZero() && !query.End.IsZero() {
			series = series.Slice(query.Start, query.End)
		}

		results = append(results, series)
	}

	return results, nil
}

// StoreMetric stores a metric
func (d *DistributedMetricCollector) StoreMetric(ctx context.Context, metric *Metric) error {
	// Get the series for this metric
	metricKey := formatMetricStorageKey(metric.Name, metric.Tags)

	// Try to get existing series from cache
	cacheKey := formatMetricCacheKey(metric.Name, metric.Tags)

	d.metricCacheMutex.RLock()
	series, exists := d.metricCache[cacheKey]
	d.metricCacheMutex.RUnlock()

	if !exists {
		// Try to load from storage
		data, err := d.storage.Get(ctx, metricKey)
		if err == nil {
			// Parse the existing series
			series, err = ParseMetricSeries(data)
			if err != nil {
				return fmt.Errorf("failed to parse existing metric data: %w", err)
			}
		} else {
			// Create a new series
			series = NewMetricSeries(metric.Name, metric.Tags)
		}

		// Update the cache
		d.metricCacheMutex.Lock()
		d.metricCache[cacheKey] = series
		d.metricCacheMutex.Unlock()
	}

	// Add the metric to the series
	series.AddMetric(metric)

	// Prune old data based on retention period
	retention := d.config.RetentionPeriod
	if retention > 0 {
		cutoff := time.Now().Add(-retention)
		series.PruneOlderThan(cutoff)
	}

	// Serialize the series
	data, err := series.Serialize()
	if err != nil {
		return fmt.Errorf("failed to serialize metric data: %w", err)
	}

	// Store in the distributed storage
	err = d.storage.Put(ctx, metricKey, data)
	if err != nil {
		return fmt.Errorf("failed to store metric data: %w", err)
	}

	// Update the cache
	d.metricCacheMutex.Lock()
	d.metricCache[cacheKey] = series
	d.metricCacheMutex.Unlock()

	// Forward to aggregator if enabled
	if d.config.EnableAggregation && d.aggregator != nil {
		d.aggregator.AddMetric(metric)
	}

	// Check alerts
	d.alertManager.CheckMetric(metric)

	return nil
}

// RegisterAlert registers an alert
func (d *DistributedMetricCollector) RegisterAlert(alert *Alert) error {
	return d.alertManager.RegisterAlert(alert)
}

// DeregisterAlert deregisters an alert
func (d *DistributedMetricCollector) DeregisterAlert(alertID string) bool {
	return d.alertManager.DeregisterAlert(alertID)
}

// collectMetrics is the main collection loop
func (d *DistributedMetricCollector) collectMetrics() {
	ticker := time.NewTicker(d.config.CollectionInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			d.collect()
		case <-d.stopChan:
			return
		}
	}
}

// collect runs a single collection cycle
func (d *DistributedMetricCollector) collect() {
	d.collectMutex.Lock()
	collectors := make([]MetricCollector, len(d.collectors))
	copy(collectors, d.collectors)
	d.collectMutex.Unlock()

	ctx, cancel := context.WithTimeout(context.Background(), d.config.CollectionInterval/2)
	defer cancel()

	wg := sync.WaitGroup{}
	for _, collector := range collectors {
		wg.Add(1)
		go func(c MetricCollector) {
			defer wg.Done()
			metrics, err := c.Collect(ctx)
			if err != nil {
				fmt.Printf("failed to collect metrics from %s: %v\n", c.ID(), err)
				return
			}

			// Add default tags
			for i := range metrics {
				for k, v := range d.config.Tags {
					if _, exists := metrics[i].Tags[k]; !exists {
						if metrics[i].Tags == nil {
							metrics[i].Tags = make(map[string]string)
						}
						metrics[i].Tags[k] = v
					}
				}
			}

			// Store metrics
			for _, metric := range metrics {
				err := d.StoreMetric(ctx, metric)
				if err != nil {
					fmt.Printf("failed to store metric %s: %v\n", metric.Name, err)
				}
			}
		}(collector)
	}

	wg.Wait()
}

// Helper functions

// formatMetricStorageKey formats a storage key for a metric
func formatMetricStorageKey(name string, tags map[string]string) string {
	// Simple implementation for now
	return fmt.Sprintf("metrics:%s:%s", name, formatTags(tags))
}

// formatMetricCacheKey formats a cache key for a metric
func formatMetricCacheKey(name string, tags map[string]string) string {
	return fmt.Sprintf("%s:%s", name, formatTags(tags))
}

// formatTags formats tags into a string
func formatTags(tags map[string]string) string {
	if len(tags) == 0 {
		return ""
	}

	result := ""
	for k, v := range tags {
		if result != "" {
			result += ","
		}
		result += fmt.Sprintf("%s=%s", k, v)
	}
	return result
}
