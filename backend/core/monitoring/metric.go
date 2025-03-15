package monitoring

import (
	"context"
	"encoding/json"
	"fmt"
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
