package monitoring

import (
	"fmt"
	"sync"
	"time"
)

// AggregationMethod defines how metrics are aggregated
type AggregationMethod string

const (
	// AggregationMethodSum sums the metric values
	AggregationMethodSum AggregationMethod = "sum"

	// AggregationMethodAvg averages the metric values
	AggregationMethodAvg AggregationMethod = "avg"

	// AggregationMethodMin takes the minimum value
	AggregationMethodMin AggregationMethod = "min"

	// AggregationMethodMax takes the maximum value
	AggregationMethodMax AggregationMethod = "max"

	// AggregationMethodCount counts occurrences
	AggregationMethodCount AggregationMethod = "count"

	// AggregationMethodP50 returns the 50th percentile (median)
	AggregationMethodP50 AggregationMethod = "p50"

	// AggregationMethodP90 returns the 90th percentile
	AggregationMethodP90 AggregationMethod = "p90"

	// AggregationMethodP95 returns the 95th percentile
	AggregationMethodP95 AggregationMethod = "p95"

	// AggregationMethodP99 returns the 99th percentile
	AggregationMethodP99 AggregationMethod = "p99"
)

// MetricAggregationConfig defines aggregation configuration for a metric
type MetricAggregationConfig struct {
	// MetricName is the name of the metric to aggregate
	MetricName string

	// Method is the aggregation method
	Method AggregationMethod

	// TagsToAggregate are tags to group by for aggregation
	TagsToAggregate []string

	// RemoveTags are tags to remove before forwarding
	RemoveTags []string

	// AddTags are additional tags to add
	AddTags map[string]string

	// Interval is how often to aggregate and forward
	Interval time.Duration
}

// MetricAggregator aggregates metrics from multiple sources
type MetricAggregator struct {
	// Node ID
	nodeID string

	// Cluster ID
	clusterID string

	// Endpoints to forward metrics to
	endpoints []string

	// Buffer for metrics being aggregated
	buffer      map[string][]*Metric
	bufferMutex sync.RWMutex

	// Aggregation configs
	configs      map[string]*MetricAggregationConfig
	configsMutex sync.RWMutex

	// Control flags
	running  bool
	stopChan chan struct{}
	mutex    sync.Mutex
}

// NewMetricAggregator creates a new metric aggregator
func NewMetricAggregator(nodeID, clusterID string, endpoints []string) *MetricAggregator {
	return &MetricAggregator{
		nodeID:    nodeID,
		clusterID: clusterID,
		endpoints: endpoints,
		buffer:    make(map[string][]*Metric),
		configs:   make(map[string]*MetricAggregationConfig),
		stopChan:  make(chan struct{}),
	}
}

// Start starts the aggregator
func (m *MetricAggregator) Start() error {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	if m.running {
		return fmt.Errorf("aggregator already running")
	}

	m.running = true
	m.stopChan = make(chan struct{})

	// Start the aggregation goroutine
	go m.startAggregationLoop()

	return nil
}

// Stop stops the aggregator
func (m *MetricAggregator) Stop() error {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	if !m.running {
		return nil
	}

	m.running = false
	close(m.stopChan)

	return nil
}

// AddAggregationConfig adds an aggregation configuration
func (m *MetricAggregator) AddAggregationConfig(config *MetricAggregationConfig) {
	m.configsMutex.Lock()
	defer m.configsMutex.Unlock()

	m.configs[config.MetricName] = config
}

// RemoveAggregationConfig removes an aggregation configuration
func (m *MetricAggregator) RemoveAggregationConfig(metricName string) bool {
	m.configsMutex.Lock()
	defer m.configsMutex.Unlock()

	if _, exists := m.configs[metricName]; !exists {
		return false
	}

	delete(m.configs, metricName)
	return true
}

// AddMetric adds a metric to the aggregation buffer
func (m *MetricAggregator) AddMetric(metric *Metric) {
	// Check if we care about this metric
	m.configsMutex.RLock()
	_, exists := m.configs[metric.Name]
	m.configsMutex.RUnlock()

	if !exists {
		// No aggregation config for this metric, no need to buffer it
		return
	}

	// Add to buffer
	m.bufferMutex.Lock()
	defer m.bufferMutex.Unlock()

	// Create a deep copy to avoid any shared state
	metricCopy := *metric
	metricCopy.Tags = make(map[string]string, len(metric.Tags))
	for k, v := range metric.Tags {
		metricCopy.Tags[k] = v
	}

	m.buffer[metric.Name] = append(m.buffer[metric.Name], &metricCopy)

	// Maybe trigger aggregation if buffer is large
	if len(m.buffer[metric.Name]) > 1000 {
		go m.aggregateMetric(metric.Name)
	}
}

// startAggregationLoop runs the aggregation loop
func (m *MetricAggregator) startAggregationLoop() {
	// Create a ticker for each aggregation interval
	intervalTickers := make(map[time.Duration]*time.Ticker)
	lastTrigger := make(map[string]time.Time)

	// Initial metrics scan to set up tickers
	m.configsMutex.RLock()
	for metricName, config := range m.configs {
		// Only create a ticker if one doesn't exist for this interval
		if _, exists := intervalTickers[config.Interval]; !exists {
			intervalTickers[config.Interval] = time.NewTicker(config.Interval)
		}
		lastTrigger[metricName] = time.Now()
	}
	m.configsMutex.RUnlock()

	// Cleanup mechanism
	cleanup := func() {
		for _, ticker := range intervalTickers {
			ticker.Stop()
		}
	}
	defer cleanup()

	for {
		select {
		case <-m.stopChan:
			return
		default:
			// Check if any intervals have elapsed
			for interval, ticker := range intervalTickers {
				select {
				case <-ticker.C:
					// Trigger aggregation for all metrics with this interval
					m.configsMutex.RLock()
					for metricName, config := range m.configs {
						if config.Interval == interval {
							// Check if it's time to aggregate this metric
							if time.Since(lastTrigger[metricName]) >= config.Interval {
								// Aggregate and forward this metric
								go m.aggregateMetric(metricName)
								lastTrigger[metricName] = time.Now()
							}
						}
					}
					m.configsMutex.RUnlock()
				default:
					// No tick yet, continue
				}
			}

			// Don't hammer the CPU
			time.Sleep(100 * time.Millisecond)
		}
	}
}

// aggregateMetric aggregates a specific metric
func (m *MetricAggregator) aggregateMetric(metricName string) {
	// Get the aggregation config
	m.configsMutex.RLock()
	config, exists := m.configs[metricName]
	m.configsMutex.RUnlock()

	if !exists {
		// No config for this metric
		return
	}

	// Extract metrics from buffer
	m.bufferMutex.Lock()
	metrics := m.buffer[metricName]
	delete(m.buffer, metricName) // Clear buffer
	m.bufferMutex.Unlock()

	if len(metrics) == 0 {
		// No metrics to aggregate
		return
	}

	// Group metrics by aggregation tags
	groups := make(map[string][]*Metric)
	for _, metric := range metrics {
		// Create a key based on the tags to aggregate by
		key := ""
		for _, tag := range config.TagsToAggregate {
			value, exists := metric.Tags[tag]
			if exists {
				if key != "" {
					key += ":"
				}
				key += tag + "=" + value
			}
		}

		// Add to the appropriate group
		groups[key] = append(groups[key], metric)
	}

	// Aggregate each group
	for _, groupMetrics := range groups {
		// Create aggregated metric
		aggregated := &Metric{
			Name:      metricName,
			Type:      groupMetrics[0].Type, // Preserve type
			Timestamp: time.Now(),           // Use current time
			Tags:      make(map[string]string),
		}

		// Apply tag operations
		// 1. Copy all tags from original metrics
		for _, tag := range config.TagsToAggregate {
			for _, metric := range groupMetrics {
				if value, exists := metric.Tags[tag]; exists {
					aggregated.Tags[tag] = value
					break // Use the first one we find
				}
			}
		}

		// 2. Remove specified tags
		for _, tag := range config.RemoveTags {
			delete(aggregated.Tags, tag)
		}

		// 3. Add new tags
		for k, v := range config.AddTags {
			aggregated.Tags[k] = v
		}

		// Add standard metadata
		aggregated.Tags["node_id"] = m.nodeID
		aggregated.Tags["cluster_id"] = m.clusterID
		aggregated.Tags["aggregation_method"] = string(config.Method)

		// Apply aggregation function
		switch config.Method {
		case AggregationMethodSum:
			sum := 0.0
			for _, metric := range groupMetrics {
				sum += metric.Value
			}
			aggregated.Value = sum

		case AggregationMethodAvg:
			sum := 0.0
			for _, metric := range groupMetrics {
				sum += metric.Value
			}
			if len(groupMetrics) > 0 {
				aggregated.Value = sum / float64(len(groupMetrics))
			}

		case AggregationMethodMin:
			if len(groupMetrics) > 0 {
				min := groupMetrics[0].Value
				for _, metric := range groupMetrics[1:] {
					if metric.Value < min {
						min = metric.Value
					}
				}
				aggregated.Value = min
			}

		case AggregationMethodMax:
			if len(groupMetrics) > 0 {
				max := groupMetrics[0].Value
				for _, metric := range groupMetrics[1:] {
					if metric.Value > max {
						max = metric.Value
					}
				}
				aggregated.Value = max
			}

		case AggregationMethodCount:
			aggregated.Value = float64(len(groupMetrics))

		case AggregationMethodP50, AggregationMethodP90, AggregationMethodP95, AggregationMethodP99:
			percentile := 50.0
			switch config.Method {
			case AggregationMethodP90:
				percentile = 90.0
			case AggregationMethodP95:
				percentile = 95.0
			case AggregationMethodP99:
				percentile = 99.0
			}

			// Extract values
			values := make([]float64, len(groupMetrics))
			for i, metric := range groupMetrics {
				values[i] = metric.Value
			}

			// Sort values
			for i := 0; i < len(values); i++ {
				for j := i + 1; j < len(values); j++ {
					if values[j] < values[i] {
						values[i], values[j] = values[j], values[i]
					}
				}
			}

			// Calculate percentile
			index := (percentile / 100.0) * float64(len(values)-1)
			if index == float64(int(index)) {
				// Exact index
				aggregated.Value = values[int(index)]
			} else {
				// Interpolate
				lower := int(index)
				upper := lower + 1
				weight := index - float64(lower)
				aggregated.Value = (1-weight)*values[lower] + weight*values[upper]
			}
		}

		// Forward the aggregated metric
		m.forwardMetric(aggregated)
	}
}

// forwardMetric forwards a metric to all configured endpoints
func (m *MetricAggregator) forwardMetric(metric *Metric) {
	// In a real implementation, this would send metrics to the configured endpoints
	// For now, we'll just log them
	fmt.Printf("Forwarding aggregated metric to %v: %s (value = %.2f, tags = %v)\n",
		m.endpoints, metric.Name, metric.Value, metric.Tags)

	// TODO: Implement real forwarding to endpoints
	// For each endpoint:
	for _, endpoint := range m.endpoints {
		// In a real implementation, this would use HTTP or other transport
		// to send the metric to remote collectors
		_ = endpoint // Just to avoid unused variable warning
	}
}

// GetMetricBuffer gets current buffer size for a metric
func (m *MetricAggregator) GetMetricBuffer(metricName string) int {
	m.bufferMutex.RLock()
	defer m.bufferMutex.RUnlock()

	return len(m.buffer[metricName])
}
