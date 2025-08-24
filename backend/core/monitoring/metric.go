package monitoring

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"
)

// MetricType represents the type of metric
type MetricType string

const (
	// MetricTypeGauge represents a gauge metric (a value that can go up and down)
	MetricTypeGauge MetricType = "gauge"

	// MetricTypeCounter represents a counter metric (a value that only increases)
	MetricTypeCounter MetricType = "counter"

	// MetricTypeHistogram represents a histogram metric (distribution of values)
	MetricTypeHistogram MetricType = "histogram"
)

// Metric represents a single metric data point
type Metric struct {
	// Name of the metric
	Name string `json:"name"`

	// Type of metric
	Type MetricType `json:"type"`

	// Value of the metric
	Value float64 `json:"value"`

	// Timestamp of the metric
	Timestamp time.Time `json:"timestamp"`

	// Tags associated with the metric
	Tags map[string]string `json:"tags"`

	// Unit of the metric (e.g., bytes, seconds, count)
	Unit string `json:"unit,omitempty"`

	// Source of the metric (e.g., node ID, component name)
	Source string `json:"source,omitempty"`
}

// NewMetric creates a new metric
func NewMetric(name string, metricType MetricType, value float64, tags map[string]string) *Metric {
	return &Metric{
		Name:      name,
		Type:      metricType,
		Value:     value,
		Timestamp: time.Now(),
		Tags:      tags,
	}
}

// WithUnit sets the unit of the metric
func (m *Metric) WithUnit(unit string) *Metric {
	m.Unit = unit
	return m
}

// WithSource sets the source of the metric
func (m *Metric) WithSource(source string) *Metric {
	m.Source = source
	return m
}

// WithTimestamp sets the timestamp of the metric
func (m *Metric) WithTimestamp(timestamp time.Time) *Metric {
	m.Timestamp = timestamp
	return m
}

// MetricSeries represents a series of metrics with the same name and tags
type MetricSeries struct {
	// Name of the metric series
	Name string `json:"name"`

	// Tags associated with the metric series
	Tags map[string]string `json:"tags"`

	// Metrics in the series
	Metrics []*Metric `json:"metrics"`
}

// NewMetricSeries creates a new metric series
func NewMetricSeries(name string, tags map[string]string) *MetricSeries {
	return &MetricSeries{
		Name:    name,
		Tags:    tags,
		Metrics: make([]*Metric, 0),
	}
}

// AddMetric adds a metric to the series
func (s *MetricSeries) AddMetric(metric *Metric) {
	s.Metrics = append(s.Metrics, metric)
}

// Covers checks if the series covers the given time range
func (s *MetricSeries) Covers(start, end time.Time) bool {
	if len(s.Metrics) == 0 {
		return false
	}

	// Find earliest and latest timestamps
	earliest := s.Metrics[0].Timestamp
	latest := s.Metrics[0].Timestamp

	for _, m := range s.Metrics {
		if m.Timestamp.Before(earliest) {
			earliest = m.Timestamp
		}
		if m.Timestamp.After(latest) {
			latest = m.Timestamp
		}
	}

	// Check if the series covers the time range
	return !earliest.After(start) && !latest.Before(end)
}

// Slice returns a subset of the series within the given time range
func (s *MetricSeries) Slice(start, end time.Time) *MetricSeries {
	result := NewMetricSeries(s.Name, s.Tags)

	for _, m := range s.Metrics {
		if !m.Timestamp.Before(start) && !m.Timestamp.After(end) {
			result.AddMetric(m)
		}
	}

	return result
}

// PruneOlderThan removes metrics older than the given time
func (s *MetricSeries) PruneOlderThan(cutoff time.Time) {
	if len(s.Metrics) == 0 {
		return
	}

	var newMetrics []*Metric
	for _, m := range s.Metrics {
		if !m.Timestamp.Before(cutoff) {
			newMetrics = append(newMetrics, m)
		}
	}

	s.Metrics = newMetrics
}

// Serialize serializes the metric series to JSON
func (s *MetricSeries) Serialize() ([]byte, error) {
	return json.Marshal(s)
}

// ParseMetricSeries parses a metric series from JSON
func ParseMetricSeries(data []byte) (*MetricSeries, error) {
	var series MetricSeries
	err := json.Unmarshal(data, &series)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal metric series: %w", err)
	}
	return &series, nil
}

// MetricQuery defines parameters for querying metrics
type MetricQuery struct {
	// Pattern is the name pattern to match (supports wildcards)
	Pattern string `json:"pattern"`

	// Tags to filter by
	Tags map[string]string `json:"tags"`

	// Start time of the query range
	Start time.Time `json:"start"`

	// End time of the query range
	End time.Time `json:"end"`

	// Aggregation function to apply
	Aggregation string `json:"aggregation,omitempty"`

	// GroupBy defines how to group metrics
	GroupBy []string `json:"group_by,omitempty"`
}

// MetricCollector is an interface for collecting metrics
type MetricCollector interface {
	// ID returns the ID of the collector
	ID() string

	// Collect collects metrics
	Collect(ctx context.Context) ([]*Metric, error)

	// Enabled returns whether the collector is enabled
	Enabled() bool
}

// MetricRegistry manages metrics in memory
type MetricRegistry struct {
	metrics map[string]*MetricSeries
	mutex   sync.RWMutex
}

// NewMetricRegistry creates a new metric registry
func NewMetricRegistry() *MetricRegistry {
	return &MetricRegistry{
		metrics: make(map[string]*MetricSeries),
	}
}

// Register adds a metric to the registry
func (r *MetricRegistry) Register(metric *Metric) {
	r.mutex.Lock()
	defer r.mutex.Unlock()
	
	// Create a key from metric name and tags
	key := r.createKey(metric.Name, metric.Tags)
	
	// Get or create series
	series, exists := r.metrics[key]
	if !exists {
		series = NewMetricSeries(metric.Name, metric.Tags)
		r.metrics[key] = series
	}
	
	// Add metric to series
	series.AddMetric(metric)
}

// Query queries metrics from the registry
func (r *MetricRegistry) Query(query MetricQuery) ([]*MetricSeries, error) {
	r.mutex.RLock()
	defer r.mutex.RUnlock()
	
	var results []*MetricSeries
	
	for _, series := range r.metrics {
		// Check if series matches query
		if r.matches(series, query) {
			// Return time-sliced series if applicable
			sliced := series.Slice(query.Start, query.End)
			if len(sliced.Metrics) > 0 {
				results = append(results, sliced)
			}
		}
	}
	
	return results, nil
}

// GetMetrics returns all metrics for a given name
func (r *MetricRegistry) GetMetrics(name string) ([]*MetricSeries, error) {
	r.mutex.RLock()
	defer r.mutex.RUnlock()
	
	var results []*MetricSeries
	for _, series := range r.metrics {
		if series.Name == name {
			results = append(results, series)
		}
	}
	
	return results, nil
}

// Cleanup removes metrics older than the specified duration
func (r *MetricRegistry) Cleanup(maxAge time.Duration) {
	r.mutex.Lock()
	defer r.mutex.Unlock()
	
	cutoff := time.Now().Add(-maxAge)
	
	// Prune old metrics from each series
	for key, series := range r.metrics {
		series.PruneOlderThan(cutoff)
		
		// Remove empty series
		if len(series.Metrics) == 0 {
			delete(r.metrics, key)
		}
	}
}

// createKey creates a unique key for a metric series
func (r *MetricRegistry) createKey(name string, tags map[string]string) string {
	key := name
	
	// Sort tags to ensure consistent keys
	var tagKeys []string
	for k := range tags {
		tagKeys = append(tagKeys, k)
	}
	
	// Simple deterministic key generation
	for _, k := range tagKeys {
		key += fmt.Sprintf(":%s=%s", k, tags[k])
	}
	
	return key
}

// matches checks if a metric series matches a query
func (r *MetricRegistry) matches(series *MetricSeries, query MetricQuery) bool {
	// Check name pattern (simple exact match for now)
	if query.Pattern != "" && series.Name != query.Pattern {
		return false
	}
	
	// Check tags
	for k, v := range query.Tags {
		if seriesValue, exists := series.Tags[k]; !exists || seriesValue != v {
			return false
		}
	}
	
	return true
}

// MetricBatch represents a batch of metrics to be processed together
type MetricBatch struct {
	Metrics   []*Metric `json:"metrics"`
	Timestamp time.Time `json:"timestamp"`
	Source    string    `json:"source"`
}

// NewMetricBatch creates a new metric batch
func NewMetricBatch(source string) *MetricBatch {
	return &MetricBatch{
		Metrics:   make([]*Metric, 0),
		Timestamp: time.Now(),
		Source:    source,
	}
}

// AddMetric adds a metric to the batch
func (b *MetricBatch) AddMetric(metric *Metric) {
	b.Metrics = append(b.Metrics, metric)
}

// Size returns the number of metrics in the batch
func (b *MetricBatch) Size() int {
	return len(b.Metrics)
}

// IsEmpty returns true if the batch is empty
func (b *MetricBatch) IsEmpty() bool {
	return len(b.Metrics) == 0
}
