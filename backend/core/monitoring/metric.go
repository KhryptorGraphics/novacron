package monitoring

import (
	"fmt"
	"sync"
	"time"
)

// MetricType represents the type of a metric
type MetricType string

const (
	// CounterMetric is a metric that accumulates values
	CounterMetric MetricType = "counter"

	// GaugeMetric is a metric that can go up and down
	GaugeMetric MetricType = "gauge"

	// HistogramMetric is a metric that tracks value distribution
	HistogramMetric MetricType = "histogram"

	// TimerMetric is a specialized metric for timing operations
	TimerMetric MetricType = "timer"

	// StateMetric represents a metric with discrete states
	StateMetric MetricType = "state"
)

// MetricValue represents the value of a metric
type MetricValue struct {
	// Timestamp is when the value was recorded
	Timestamp time.Time `json:"timestamp"`

	// Value is the metric value
	Value float64 `json:"value"`

	// Labels are additional labels for this specific value
	Labels map[string]string `json:"labels,omitempty"`
}

// Metric represents a monitored metric
type Metric struct {
	// ID is the unique identifier for the metric
	ID string `json:"id"`

	// Name is the human-readable name of the metric
	Name string `json:"name"`

	// Description is a description of the metric
	Description string `json:"description"`

	// Type is the type of the metric
	Type MetricType `json:"type"`

	// Unit is the unit of the metric (e.g., "seconds", "bytes")
	Unit string `json:"unit,omitempty"`

	// Labels are the labels associated with the metric
	Labels map[string]string `json:"labels"`

	// Source is the source of the metric
	Source string `json:"source"`

	// TenantID is the ID of the tenant this metric belongs to
	TenantID string `json:"tenantId,omitempty"`

	// ResourceID is the ID of the resource this metric is related to
	ResourceID string `json:"resourceId,omitempty"`

	// ResourceType is the type of the resource this metric is related to
	ResourceType string `json:"resourceType,omitempty"`

	// Tags are additional tags for the metric
	Tags []string `json:"tags,omitempty"`

	// Min is the minimum expected value (for anomaly detection)
	Min *float64 `json:"min,omitempty"`

	// Max is the maximum expected value (for anomaly detection)
	Max *float64 `json:"max,omitempty"`

	// WarningThreshold is the threshold for warning alerts
	WarningThreshold *float64 `json:"warningThreshold,omitempty"`

	// CriticalThreshold is the threshold for critical alerts
	CriticalThreshold *float64 `json:"criticalThreshold,omitempty"`

	// Values are the values of the metric
	Values []MetricValue `json:"-"`

	// isActive indicates if this metric is actively collected
	isActive bool

	// collectInterval is the interval at which this metric is collected
	collectInterval time.Duration

	// LastValue is the most recent value of the metric
	LastValue *MetricValue `json:"lastValue,omitempty"`

	// mutex protects the metric
	mutex sync.RWMutex
}

// MetricRegistry keeps track of all metrics
type MetricRegistry struct {
	metrics     map[string]*Metric
	metricMutex sync.RWMutex
}

// NewMetricRegistry creates a new metric registry
func NewMetricRegistry() *MetricRegistry {
	return &MetricRegistry{
		metrics: make(map[string]*Metric),
	}
}

// RegisterMetric registers a metric with the registry
func (r *MetricRegistry) RegisterMetric(metric *Metric) error {
	r.metricMutex.Lock()
	defer r.metricMutex.Unlock()

	if _, exists := r.metrics[metric.ID]; exists {
		return fmt.Errorf("metric already exists: %s", metric.ID)
	}

	r.metrics[metric.ID] = metric
	return nil
}

// GetMetric gets a metric by ID
func (r *MetricRegistry) GetMetric(id string) (*Metric, error) {
	r.metricMutex.RLock()
	defer r.metricMutex.RUnlock()

	metric, exists := r.metrics[id]
	if !exists {
		return nil, fmt.Errorf("metric not found: %s", id)
	}

	return metric, nil
}

// ListMetrics lists all metrics
func (r *MetricRegistry) ListMetrics() []*Metric {
	r.metricMutex.RLock()
	defer r.metricMutex.RUnlock()

	metrics := make([]*Metric, 0, len(r.metrics))
	for _, metric := range r.metrics {
		metrics = append(metrics, metric)
	}

	return metrics
}

// RemoveMetric removes a metric
func (r *MetricRegistry) RemoveMetric(id string) error {
	r.metricMutex.Lock()
	defer r.metricMutex.Unlock()

	if _, exists := r.metrics[id]; !exists {
		return fmt.Errorf("metric not found: %s", id)
	}

	delete(r.metrics, id)
	return nil
}

// FilterMetrics filters metrics by criteria
func (r *MetricRegistry) FilterMetrics(filter func(*Metric) bool) []*Metric {
	r.metricMutex.RLock()
	defer r.metricMutex.RUnlock()

	filtered := make([]*Metric, 0)
	for _, metric := range r.metrics {
		if filter(metric) {
			filtered = append(filtered, metric)
		}
	}

	return filtered
}

// RecordValue records a value for a metric
func (m *Metric) RecordValue(value float64, labels map[string]string) {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	now := time.Now()
	mv := MetricValue{
		Timestamp: now,
		Value:     value,
		Labels:    labels,
	}

	m.Values = append(m.Values, mv)
	m.LastValue = &mv

	// Limit the number of stored values (circular buffer)
	if len(m.Values) > 1000 {
		m.Values = m.Values[1:]
	}
}

// GetValues gets the values for a metric within a time range
func (m *Metric) GetValues(start, end time.Time) []MetricValue {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	if end.IsZero() {
		end = time.Now()
	}

	values := make([]MetricValue, 0)
	for _, v := range m.Values {
		if (start.IsZero() || !v.Timestamp.Before(start)) && !v.Timestamp.After(end) {
			values = append(values, v)
		}
	}

	return values
}

// GetLastValue gets the last value of a metric
func (m *Metric) GetLastValue() *MetricValue {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	if len(m.Values) == 0 {
		return nil
	}

	return &m.Values[len(m.Values)-1]
}

// NewMetric creates a new metric
func NewMetric(id, name, description string, metricType MetricType, source string) *Metric {
	return &Metric{
		ID:          id,
		Name:        name,
		Description: description,
		Type:        metricType,
		Source:      source,
		Labels:      make(map[string]string),
		Values:      make([]MetricValue, 0),
		isActive:    true,
	}
}

// NewCounterMetric creates a new counter metric
func NewCounterMetric(id, name, description, source string) *Metric {
	return NewMetric(id, name, description, CounterMetric, source)
}

// NewGaugeMetric creates a new gauge metric
func NewGaugeMetric(id, name, description, source string) *Metric {
	return NewMetric(id, name, description, GaugeMetric, source)
}

// NewHistogramMetric creates a new histogram metric
func NewHistogramMetric(id, name, description, source string) *Metric {
	return NewMetric(id, name, description, HistogramMetric, source)
}

// NewTimerMetric creates a new timer metric
func NewTimerMetric(id, name, description, source string) *Metric {
	metric := NewMetric(id, name, description, TimerMetric, source)
	metric.Unit = "seconds"
	return metric
}

// NewStateMetric creates a new state metric
func NewStateMetric(id, name, description, source string) *Metric {
	return NewMetric(id, name, description, StateMetric, source)
}

// SetWarningThreshold sets the warning threshold for a metric
func (m *Metric) SetWarningThreshold(threshold float64) {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	m.WarningThreshold = &threshold
}

// SetCriticalThreshold sets the critical threshold for a metric
func (m *Metric) SetCriticalThreshold(threshold float64) {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	m.CriticalThreshold = &threshold
}

// SetMinValue sets the minimum expected value for a metric
func (m *Metric) SetMinValue(min float64) {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	m.Min = &min
}

// SetMaxValue sets the maximum expected value for a metric
func (m *Metric) SetMaxValue(max float64) {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	m.Max = &max
}

// SetUnit sets the unit for a metric
func (m *Metric) SetUnit(unit string) {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	m.Unit = unit
}

// AddTag adds a tag to a metric
func (m *Metric) AddTag(tag string) {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	for _, t := range m.Tags {
		if t == tag {
			return
		}
	}

	m.Tags = append(m.Tags, tag)
}

// RemoveTag removes a tag from a metric
func (m *Metric) RemoveTag(tag string) {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	for i, t := range m.Tags {
		if t == tag {
			m.Tags = append(m.Tags[:i], m.Tags[i+1:]...)
			return
		}
	}
}

// AddLabel adds a label to a metric
func (m *Metric) AddLabel(key, value string) {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	m.Labels[key] = value
}

// RemoveLabel removes a label from a metric
func (m *Metric) RemoveLabel(key string) {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	delete(m.Labels, key)
}

// SetActive sets whether a metric is active
func (m *Metric) SetActive(active bool) {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	m.isActive = active
}

// IsActive checks if a metric is active
func (m *Metric) IsActive() bool {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	return m.isActive
}

// SetCollectInterval sets the collection interval for a metric
func (m *Metric) SetCollectInterval(interval time.Duration) {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	m.collectInterval = interval
}

// GetCollectInterval gets the collection interval for a metric
func (m *Metric) GetCollectInterval() time.Duration {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	return m.collectInterval
}

// MetricBatch represents a batch of metric values
type MetricBatch struct {
	// MetricID is the ID of the metric
	MetricID string `json:"metricId"`

	// Values are the values of the metric
	Values []MetricValue `json:"values"`

	// Timestamp is when the batch was created
	Timestamp time.Time `json:"timestamp"`
}

// MetricCollector collects metrics from a source
type MetricCollector interface {
	// Collect collects metrics
	Collect() ([]MetricBatch, error)

	// GetMetrics gets the metrics this collector provides
	GetMetrics() []*Metric

	// Start starts the collector
	Start() error

	// Stop stops the collector
	Stop() error

	// SetCollectInterval sets the collection interval
	SetCollectInterval(interval time.Duration)
}
