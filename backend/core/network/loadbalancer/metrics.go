package loadbalancer

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
)

// MetricsCollector provides comprehensive metrics collection and monitoring
type MetricsCollector struct {
	// Configuration
	config           MetricsConfig
	
	// Metric stores
	counters         map[string]*Counter
	gauges           map[string]*Gauge
	histograms       map[string]*Histogram
	timers           map[string]*Timer
	
	// Synchronization
	countersMutex    sync.RWMutex
	gaugesMutex      sync.RWMutex
	histogramsMutex  sync.RWMutex
	timersMutex      sync.RWMutex
	
	// Time series storage
	timeSeries       *TimeSeriesStorage
	
	// Aggregation and alerts
	aggregator       *MetricsAggregator
	alertManager     *AlertManager
	
	// Exporters
	exporters        []MetricsExporter
	exportersMutex   sync.RWMutex
	
	// Runtime state
	ctx              context.Context
	cancel           context.CancelFunc
	initialized      bool
	startTime        time.Time
}

// MetricsConfig holds metrics collection configuration
type MetricsConfig struct {
	// Collection settings
	EnableCollection     bool              `json:"enable_collection"`
	CollectionInterval   time.Duration     `json:"collection_interval"`
	RetentionPeriod      time.Duration     `json:"retention_period"`
	MaxMetricsPerType    int               `json:"max_metrics_per_type"`
	
	// Time series settings
	TimeSeriesBuckets    int               `json:"time_series_buckets"`
	BucketDuration       time.Duration     `json:"bucket_duration"`
	CompactionInterval   time.Duration     `json:"compaction_interval"`
	
	// Histogram settings
	DefaultHistogramBuckets []float64      `json:"default_histogram_buckets"`
	MaxHistogramBuckets  int               `json:"max_histogram_buckets"`
	
	// Aggregation settings
	EnableAggregation    bool              `json:"enable_aggregation"`
	AggregationWindow    time.Duration     `json:"aggregation_window"`
	AggregationFunctions []string          `json:"aggregation_functions"`
	
	// Alert settings
	EnableAlerts         bool              `json:"enable_alerts"`
	AlertEvaluationInterval time.Duration  `json:"alert_evaluation_interval"`
	AlertRetentionPeriod time.Duration     `json:"alert_retention_period"`
	
	// Export settings
	EnableExport         bool              `json:"enable_export"`
	ExportInterval       time.Duration     `json:"export_interval"`
	ExportFormat         string            `json:"export_format"`
	ExportEndpoints      []ExportEndpoint  `json:"export_endpoints"`
	
	// Performance settings
	MetricBufferSize     int               `json:"metric_buffer_size"`
	MaxConcurrentExports int               `json:"max_concurrent_exports"`
	
	// HTTP server settings
	EnableHTTPServer     bool              `json:"enable_http_server"`
	HTTPServerPort       int               `json:"http_server_port"`
	HTTPServerPath       string            `json:"http_server_path"`
}

// ExportEndpoint defines a metrics export endpoint
type ExportEndpoint struct {
	Name         string            `json:"name"`
	Type         string            `json:"type"`
	URL          string            `json:"url"`
	Headers      map[string]string `json:"headers"`
	Timeout      time.Duration     `json:"timeout"`
	RetryCount   int               `json:"retry_count"`
	RetryDelay   time.Duration     `json:"retry_delay"`
	Enabled      bool              `json:"enabled"`
}

// Counter represents a monotonically increasing metric
type Counter struct {
	Name        string            `json:"name"`
	Help        string            `json:"help"`
	Labels      map[string]string `json:"labels"`
	Value       int64             `json:"value"`
	CreatedAt   time.Time         `json:"created_at"`
	UpdatedAt   time.Time         `json:"updated_at"`
	mutex       sync.RWMutex
}

// Gauge represents a metric that can go up and down
type Gauge struct {
	Name        string            `json:"name"`
	Help        string            `json:"help"`
	Labels      map[string]string `json:"labels"`
	Value       float64           `json:"value"`
	CreatedAt   time.Time         `json:"created_at"`
	UpdatedAt   time.Time         `json:"updated_at"`
	mutex       sync.RWMutex
}

// Histogram represents a distribution of values
type Histogram struct {
	Name        string            `json:"name"`
	Help        string            `json:"help"`
	Labels      map[string]string `json:"labels"`
	Buckets     []float64         `json:"buckets"`
	Counts      []int64           `json:"counts"`
	Sum         float64           `json:"sum"`
	Count       int64             `json:"count"`
	CreatedAt   time.Time         `json:"created_at"`
	UpdatedAt   time.Time         `json:"updated_at"`
	mutex       sync.RWMutex
}

// Timer represents timing measurements
type Timer struct {
	Name        string            `json:"name"`
	Help        string            `json:"help"`
	Labels      map[string]string `json:"labels"`
	Count       int64             `json:"count"`
	Sum         time.Duration     `json:"sum"`
	Min         time.Duration     `json:"min"`
	Max         time.Duration     `json:"max"`
	Mean        time.Duration     `json:"mean"`
	P50         time.Duration     `json:"p50"`
	P95         time.Duration     `json:"p95"`
	P99         time.Duration     `json:"p99"`
	Values      []time.Duration   `json:"-"` // Keep last N values for percentile calculation
	CreatedAt   time.Time         `json:"created_at"`
	UpdatedAt   time.Time         `json:"updated_at"`
	mutex       sync.RWMutex
}

// TimeSeriesStorage stores time series data
type TimeSeriesStorage struct {
	series      map[string]*TimeSeries
	mutex       sync.RWMutex
	buckets     int
	bucketDuration time.Duration
}

// TimeSeries represents a time series of values
type TimeSeries struct {
	Name        string                 `json:"name"`
	Labels      map[string]string      `json:"labels"`
	Type        string                 `json:"type"`
	DataPoints  []DataPoint            `json:"data_points"`
	StartTime   time.Time              `json:"start_time"`
	EndTime     time.Time              `json:"end_time"`
	mutex       sync.RWMutex
}

// DataPoint represents a single data point in a time series
type DataPoint struct {
	Timestamp   time.Time   `json:"timestamp"`
	Value       float64     `json:"value"`
	Labels      map[string]string `json:"labels,omitempty"`
}

// MetricsAggregator aggregates metrics over time windows
type MetricsAggregator struct {
	aggregations map[string]*Aggregation
	mutex        sync.RWMutex
	windowSize   time.Duration
}

// Aggregation represents an aggregated metric
type Aggregation struct {
	Name        string              `json:"name"`
	Type        string              `json:"type"`
	Function    AggregationFunction `json:"function"`
	WindowSize  time.Duration       `json:"window_size"`
	Value       float64             `json:"value"`
	Count       int64               `json:"count"`
	LastUpdate  time.Time           `json:"last_update"`
}

// AggregationFunction defines aggregation functions
type AggregationFunction string

const (
	AggFuncSum     AggregationFunction = "sum"
	AggFuncAvg     AggregationFunction = "avg"
	AggFuncMin     AggregationFunction = "min"
	AggFuncMax     AggregationFunction = "max"
	AggFuncCount   AggregationFunction = "count"
	AggFuncRate    AggregationFunction = "rate"
	AggFuncP50     AggregationFunction = "p50"
	AggFuncP95     AggregationFunction = "p95"
	AggFuncP99     AggregationFunction = "p99"
)

// AlertManager manages metric alerts
type AlertManager struct {
	rules       map[string]*AlertRule
	alerts      map[string]*Alert
	mutex       sync.RWMutex
	config      *MetricsConfig
}

// AlertRule is now defined in types.go

// Alert represents a triggered alert
type Alert struct {
	ID          string                 `json:"id"`
	RuleID      string                 `json:"rule_id"`
	RuleName    string                 `json:"rule_name"`
	Status      AlertStatus            `json:"status"`
	Severity    AlertSeverity          `json:"severity"`
	Message     string                 `json:"message"`
	Value       float64                `json:"value"`
	Threshold   float64                `json:"threshold"`
	Labels      map[string]string      `json:"labels"`
	Annotations map[string]string      `json:"annotations"`
	StartsAt    time.Time              `json:"starts_at"`
	EndsAt      time.Time              `json:"ends_at,omitempty"`
	UpdatedAt   time.Time              `json:"updated_at"`
	GeneratorURL string                `json:"generator_url,omitempty"`
}

// Types and enums
type AlertCondition string
type AlertSeverity string
type AlertStatus string

const (
	AlertConditionGreater     AlertCondition = "greater"
	AlertConditionLess        AlertCondition = "less"
	AlertConditionEqual       AlertCondition = "equal"
	AlertConditionNotEqual    AlertCondition = "not_equal"
	
	AlertSeverityCritical     AlertSeverity = "critical"
	AlertSeverityWarning      AlertSeverity = "warning"
	AlertSeverityInfo         AlertSeverity = "info"
	
	AlertStatusFiring         AlertStatus = "firing"
	AlertStatusResolved       AlertStatus = "resolved"
	AlertStatusPending        AlertStatus = "pending"
)

// MetricsExporter defines interface for metrics exporters
type MetricsExporter interface {
	Export(ctx context.Context, metrics *MetricsSnapshot) error
	Name() string
	Configure(config map[string]interface{}) error
}

// MetricsSnapshot represents a point-in-time snapshot of all metrics
type MetricsSnapshot struct {
	Timestamp   time.Time              `json:"timestamp"`
	Counters    map[string]*Counter    `json:"counters"`
	Gauges      map[string]*Gauge      `json:"gauges"`
	Histograms  map[string]*Histogram  `json:"histograms"`
	Timers      map[string]*Timer      `json:"timers"`
	TimeSeries  map[string]*TimeSeries `json:"time_series"`
	Metadata    map[string]interface{} `json:"metadata"`
}

// NewMetricsCollector creates a new metrics collector
func NewMetricsCollector(config MetricsConfig) *MetricsCollector {
	ctx, cancel := context.WithCancel(context.Background())
	
	return &MetricsCollector{
		config:     config,
		counters:   make(map[string]*Counter),
		gauges:     make(map[string]*Gauge),
		histograms: make(map[string]*Histogram),
		timers:     make(map[string]*Timer),
		exporters:  make([]MetricsExporter, 0),
		ctx:        ctx,
		cancel:     cancel,
		startTime:  time.Now(),
	}
}

// Start initializes and starts the metrics collector
func (mc *MetricsCollector) Start() error {
	if mc.initialized {
		return fmt.Errorf("metrics collector already started")
	}
	
	// Initialize time series storage
	mc.timeSeries = &TimeSeriesStorage{
		series:         make(map[string]*TimeSeries),
		buckets:        mc.config.TimeSeriesBuckets,
		bucketDuration: mc.config.BucketDuration,
	}
	
	// Initialize aggregator
	if mc.config.EnableAggregation {
		mc.aggregator = &MetricsAggregator{
			aggregations: make(map[string]*Aggregation),
			windowSize:   mc.config.AggregationWindow,
		}
	}
	
	// Initialize alert manager
	if mc.config.EnableAlerts {
		mc.alertManager = &AlertManager{
			rules:  make(map[string]*AlertRule),
			alerts: make(map[string]*Alert),
			config: &mc.config,
		}
	}
	
	// Register built-in exporters
	mc.registerBuiltinExporters()
	
	// Start background processes
	if mc.config.EnableCollection {
		go mc.collectionLoop()
	}
	
	if mc.config.EnableAggregation {
		go mc.aggregationLoop()
	}
	
	if mc.config.EnableAlerts {
		go mc.alertEvaluationLoop()
	}
	
	if mc.config.EnableExport {
		go mc.exportLoop()
	}
	
	go mc.compactionLoop()
	go mc.cleanupLoop()
	
	// Start HTTP server if enabled
	if mc.config.EnableHTTPServer {
		go mc.startHTTPServer()
	}
	
	mc.initialized = true
	return nil
}

// Stop stops the metrics collector
func (mc *MetricsCollector) Stop() error {
	mc.cancel()
	
	// Final export
	if mc.config.EnableExport {
		mc.exportMetrics()
	}
	
	mc.initialized = false
	return nil
}

// Counter operations

// GetOrCreateCounter gets or creates a counter
func (mc *MetricsCollector) GetOrCreateCounter(name, help string, labels map[string]string) *Counter {
	key := mc.buildMetricKey(name, labels)
	
	mc.countersMutex.RLock()
	counter, exists := mc.counters[key]
	mc.countersMutex.RUnlock()
	
	if exists {
		return counter
	}
	
	mc.countersMutex.Lock()
	defer mc.countersMutex.Unlock()
	
	// Double-check after acquiring write lock
	if counter, exists := mc.counters[key]; exists {
		return counter
	}
	
	counter = &Counter{
		Name:      name,
		Help:      help,
		Labels:    labels,
		Value:     0,
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}
	
	mc.counters[key] = counter
	return counter
}

// IncrementCounter increments a counter by 1
func (mc *MetricsCollector) IncrementCounter(name string, labels map[string]string) {
	counter := mc.GetOrCreateCounter(name, "", labels)
	counter.Inc()
}

// AddToCounter adds a value to a counter
func (mc *MetricsCollector) AddToCounter(name string, value int64, labels map[string]string) {
	counter := mc.GetOrCreateCounter(name, "", labels)
	counter.Add(value)
}

// Inc increments the counter by 1
func (c *Counter) Inc() {
	c.Add(1)
}

// Add adds the given value to the counter
func (c *Counter) Add(value int64) {
	atomic.AddInt64(&c.Value, value)
	c.mutex.Lock()
	c.UpdatedAt = time.Now()
	c.mutex.Unlock()
}

// Get returns the current counter value
func (c *Counter) Get() int64 {
	return atomic.LoadInt64(&c.Value)
}

// Gauge operations

// GetOrCreateGauge gets or creates a gauge
func (mc *MetricsCollector) GetOrCreateGauge(name, help string, labels map[string]string) *Gauge {
	key := mc.buildMetricKey(name, labels)
	
	mc.gaugesMutex.RLock()
	gauge, exists := mc.gauges[key]
	mc.gaugesMutex.RUnlock()
	
	if exists {
		return gauge
	}
	
	mc.gaugesMutex.Lock()
	defer mc.gaugesMutex.Unlock()
	
	// Double-check after acquiring write lock
	if gauge, exists := mc.gauges[key]; exists {
		return gauge
	}
	
	gauge = &Gauge{
		Name:      name,
		Help:      help,
		Labels:    labels,
		Value:     0,
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}
	
	mc.gauges[key] = gauge
	return gauge
}

// SetGauge sets a gauge value
func (mc *MetricsCollector) SetGauge(name string, value float64, labels map[string]string) {
	gauge := mc.GetOrCreateGauge(name, "", labels)
	gauge.Set(value)
}

// Set sets the gauge to the given value
func (g *Gauge) Set(value float64) {
	g.mutex.Lock()
	defer g.mutex.Unlock()
	
	g.Value = value
	g.UpdatedAt = time.Now()
}

// Inc increments the gauge by 1
func (g *Gauge) Inc() {
	g.Add(1)
}

// Dec decrements the gauge by 1
func (g *Gauge) Dec() {
	g.Add(-1)
}

// Add adds the given value to the gauge
func (g *Gauge) Add(value float64) {
	g.mutex.Lock()
	defer g.mutex.Unlock()
	
	g.Value += value
	g.UpdatedAt = time.Now()
}

// Get returns the current gauge value
func (g *Gauge) Get() float64 {
	g.mutex.RLock()
	defer g.mutex.RUnlock()
	return g.Value
}

// Histogram operations

// GetOrCreateHistogram gets or creates a histogram
func (mc *MetricsCollector) GetOrCreateHistogram(name, help string, buckets []float64, labels map[string]string) *Histogram {
	key := mc.buildMetricKey(name, labels)
	
	mc.histogramsMutex.RLock()
	histogram, exists := mc.histograms[key]
	mc.histogramsMutex.RUnlock()
	
	if exists {
		return histogram
	}
	
	mc.histogramsMutex.Lock()
	defer mc.histogramsMutex.Unlock()
	
	// Double-check after acquiring write lock
	if histogram, exists := mc.histograms[key]; exists {
		return histogram
	}
	
	if buckets == nil {
		buckets = mc.config.DefaultHistogramBuckets
	}
	
	histogram = &Histogram{
		Name:      name,
		Help:      help,
		Labels:    labels,
		Buckets:   buckets,
		Counts:    make([]int64, len(buckets)),
		Sum:       0,
		Count:     0,
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}
	
	mc.histograms[key] = histogram
	return histogram
}

// ObserveHistogram adds an observation to a histogram
func (mc *MetricsCollector) ObserveHistogram(name string, value float64, labels map[string]string) {
	histogram := mc.GetOrCreateHistogram(name, "", nil, labels)
	histogram.Observe(value)
}

// Observe adds a single observation to the histogram
func (h *Histogram) Observe(value float64) {
	h.mutex.Lock()
	defer h.mutex.Unlock()
	
	// Find appropriate bucket
	for i, bucket := range h.Buckets {
		if value <= bucket {
			h.Counts[i]++
			break
		}
	}
	
	h.Sum += value
	h.Count++
	h.UpdatedAt = time.Now()
}

// Timer operations

// GetOrCreateTimer gets or creates a timer
func (mc *MetricsCollector) GetOrCreateTimer(name, help string, labels map[string]string) *Timer {
	key := mc.buildMetricKey(name, labels)
	
	mc.timersMutex.RLock()
	timer, exists := mc.timers[key]
	mc.timersMutex.RUnlock()
	
	if exists {
		return timer
	}
	
	mc.timersMutex.Lock()
	defer mc.timersMutex.Unlock()
	
	// Double-check after acquiring write lock
	if timer, exists := mc.timers[key]; exists {
		return timer
	}
	
	timer = &Timer{
		Name:      name,
		Help:      help,
		Labels:    labels,
		Values:    make([]time.Duration, 0, 1000), // Keep last 1000 values
		Min:       time.Duration(^uint64(0) >> 1),  // Max int64
		Max:       0,
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}
	
	mc.timers[key] = timer
	return timer
}

// RecordDuration records a duration measurement
func (mc *MetricsCollector) RecordDuration(name string, duration time.Duration, labels map[string]string) {
	timer := mc.GetOrCreateTimer(name, "", labels)
	timer.Record(duration)
}

// TimeFunction times a function execution
func (mc *MetricsCollector) TimeFunction(name string, labels map[string]string, fn func()) {
	start := time.Now()
	fn()
	duration := time.Since(start)
	mc.RecordDuration(name, duration, labels)
}

// Record records a duration measurement
func (t *Timer) Record(duration time.Duration) {
	t.mutex.Lock()
	defer t.mutex.Unlock()
	
	t.Count++
	t.Sum += duration
	
	if duration < t.Min {
		t.Min = duration
	}
	if duration > t.Max {
		t.Max = duration
	}
	
	// Add to values for percentile calculation
	t.Values = append(t.Values, duration)
	
	// Keep only last 1000 values
	if len(t.Values) > 1000 {
		t.Values = t.Values[len(t.Values)-1000:]
	}
	
	// Update mean
	t.Mean = t.Sum / time.Duration(t.Count)
	
	// Calculate percentiles
	t.calculatePercentiles()
	
	t.UpdatedAt = time.Now()
}

// calculatePercentiles calculates timer percentiles
func (t *Timer) calculatePercentiles() {
	if len(t.Values) == 0 {
		return
	}
	
	// Simple percentile calculation (would use more efficient algorithm in production)
	sorted := make([]time.Duration, len(t.Values))
	copy(sorted, t.Values)
	
	// Simple bubble sort for demonstration
	for i := 0; i < len(sorted)-1; i++ {
		for j := i + 1; j < len(sorted); j++ {
			if sorted[i] > sorted[j] {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}
	
	p50Index := int(float64(len(sorted)) * 0.5)
	p95Index := int(float64(len(sorted)) * 0.95)
	p99Index := int(float64(len(sorted)) * 0.99)
	
	if p50Index < len(sorted) {
		t.P50 = sorted[p50Index]
	}
	if p95Index < len(sorted) {
		t.P95 = sorted[p95Index]
	}
	if p99Index < len(sorted) {
		t.P99 = sorted[p99Index]
	}
}

// Time series operations

// RecordTimeSeries records a time series data point
func (mc *MetricsCollector) RecordTimeSeries(name string, value float64, labels map[string]string) {
	mc.timeSeries.mutex.Lock()
	defer mc.timeSeries.mutex.Unlock()
	
	key := mc.buildMetricKey(name, labels)
	series, exists := mc.timeSeries.series[key]
	
	if !exists {
		series = &TimeSeries{
			Name:       name,
			Labels:     labels,
			Type:       "gauge", // Default type
			DataPoints: make([]DataPoint, 0),
			StartTime:  time.Now(),
		}
		mc.timeSeries.series[key] = series
	}
	
	series.mutex.Lock()
	defer series.mutex.Unlock()
	
	dataPoint := DataPoint{
		Timestamp: time.Now(),
		Value:     value,
		Labels:    labels,
	}
	
	series.DataPoints = append(series.DataPoints, dataPoint)
	series.EndTime = dataPoint.Timestamp
	
	// Limit number of data points
	maxPoints := mc.config.TimeSeriesBuckets
	if len(series.DataPoints) > maxPoints {
		series.DataPoints = series.DataPoints[len(series.DataPoints)-maxPoints:]
		series.StartTime = series.DataPoints[0].Timestamp
	}
}

// GetTimeSeries retrieves time series data
func (mc *MetricsCollector) GetTimeSeries(name string, labels map[string]string, start, end time.Time) (*TimeSeries, error) {
	key := mc.buildMetricKey(name, labels)
	
	mc.timeSeries.mutex.RLock()
	series, exists := mc.timeSeries.series[key]
	mc.timeSeries.mutex.RUnlock()
	
	if !exists {
		return nil, fmt.Errorf("time series %s not found", key)
	}
	
	series.mutex.RLock()
	defer series.mutex.RUnlock()
	
	// Filter data points by time range
	var filteredPoints []DataPoint
	for _, point := range series.DataPoints {
		if (start.IsZero() || point.Timestamp.After(start) || point.Timestamp.Equal(start)) &&
		   (end.IsZero() || point.Timestamp.Before(end) || point.Timestamp.Equal(end)) {
			filteredPoints = append(filteredPoints, point)
		}
	}
	
	return &TimeSeries{
		Name:       series.Name,
		Labels:     series.Labels,
		Type:       series.Type,
		DataPoints: filteredPoints,
		StartTime:  start,
		EndTime:    end,
	}, nil
}

// Alert management

// AddAlertRule adds an alert rule
func (mc *MetricsCollector) AddAlertRule(rule *AlertRule) error {
	if mc.alertManager == nil {
		return fmt.Errorf("alert manager not initialized")
	}
	
	if rule.ID == "" {
		rule.ID = uuid.New().String()
	}
	
	rule.CreatedAt = time.Now()
	rule.UpdatedAt = time.Now()
	
	mc.alertManager.mutex.Lock()
	mc.alertManager.rules[rule.ID] = rule
	mc.alertManager.mutex.Unlock()
	
	return nil
}

// GetActiveAlerts returns currently active alerts
func (mc *MetricsCollector) GetActiveAlerts() []*Alert {
	if mc.alertManager == nil {
		return nil
	}
	
	mc.alertManager.mutex.RLock()
	defer mc.alertManager.mutex.RUnlock()
	
	var activeAlerts []*Alert
	for _, alert := range mc.alertManager.alerts {
		if alert.Status == AlertStatusFiring {
			alertCopy := *alert
			activeAlerts = append(activeAlerts, &alertCopy)
		}
	}
	
	return activeAlerts
}

// Background processes

// collectionLoop periodically collects metrics
func (mc *MetricsCollector) collectionLoop() {
	ticker := time.NewTicker(mc.config.CollectionInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-mc.ctx.Done():
			return
		case <-ticker.C:
			mc.collectSystemMetrics()
		}
	}
}

// aggregationLoop performs metric aggregations
func (mc *MetricsCollector) aggregationLoop() {
	ticker := time.NewTicker(mc.config.AggregationWindow)
	defer ticker.Stop()
	
	for {
		select {
		case <-mc.ctx.Done():
			return
		case <-ticker.C:
			mc.performAggregations()
		}
	}
}

// alertEvaluationLoop evaluates alert rules
func (mc *MetricsCollector) alertEvaluationLoop() {
	ticker := time.NewTicker(mc.config.AlertEvaluationInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-mc.ctx.Done():
			return
		case <-ticker.C:
			mc.evaluateAlerts()
		}
	}
}

// exportLoop periodically exports metrics
func (mc *MetricsCollector) exportLoop() {
	ticker := time.NewTicker(mc.config.ExportInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-mc.ctx.Done():
			return
		case <-ticker.C:
			mc.exportMetrics()
		}
	}
}

// compactionLoop compacts time series data
func (mc *MetricsCollector) compactionLoop() {
	ticker := time.NewTicker(mc.config.CompactionInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-mc.ctx.Done():
			return
		case <-ticker.C:
			mc.compactTimeSeries()
		}
	}
}

// cleanupLoop cleans up old data
func (mc *MetricsCollector) cleanupLoop() {
	ticker := time.NewTicker(time.Hour) // Cleanup every hour
	defer ticker.Stop()
	
	for {
		select {
		case <-mc.ctx.Done():
			return
		case <-ticker.C:
			mc.cleanupOldData()
		}
	}
}

// collectSystemMetrics collects system-level metrics
func (mc *MetricsCollector) collectSystemMetrics() {
	now := time.Now()
	uptime := now.Sub(mc.startTime).Seconds()
	
	// Record uptime
	mc.SetGauge("system_uptime_seconds", uptime, nil)
	
	// Record timestamp
	mc.SetGauge("system_timestamp", float64(now.Unix()), nil)
	
	// Record metric counts
	mc.countersMutex.RLock()
	mc.SetGauge("metrics_counters_total", float64(len(mc.counters)), nil)
	mc.countersMutex.RUnlock()
	
	mc.gaugesMutex.RLock()
	mc.SetGauge("metrics_gauges_total", float64(len(mc.gauges)), nil)
	mc.gaugesMutex.RUnlock()
	
	mc.histogramsMutex.RLock()
	mc.SetGauge("metrics_histograms_total", float64(len(mc.histograms)), nil)
	mc.histogramsMutex.RUnlock()
	
	mc.timersMutex.RLock()
	mc.SetGauge("metrics_timers_total", float64(len(mc.timers)), nil)
	mc.timersMutex.RUnlock()
}

// performAggregations performs metric aggregations
func (mc *MetricsCollector) performAggregations() {
	if mc.aggregator == nil {
		return
	}
	
	// Simplified aggregation - in practice this would be more sophisticated
	mc.aggregator.mutex.Lock()
	defer mc.aggregator.mutex.Unlock()
	
	// Example: aggregate request rate
	if counter := mc.counters["http_requests_total"]; counter != nil {
		aggregation := &Aggregation{
			Name:       "http_requests_rate",
			Type:       "counter",
			Function:   AggFuncRate,
			WindowSize: mc.config.AggregationWindow,
			Value:      float64(counter.Get()),
			Count:      1,
			LastUpdate: time.Now(),
		}
		mc.aggregator.aggregations["http_requests_rate"] = aggregation
	}
}

// evaluateAlerts evaluates alert rules
func (mc *MetricsCollector) evaluateAlerts() {
	if mc.alertManager == nil {
		return
	}
	
	mc.alertManager.mutex.Lock()
	defer mc.alertManager.mutex.Unlock()
	
	for _, rule := range mc.alertManager.rules {
		if !rule.Enabled {
			continue
		}
		
		// Simplified alert evaluation
		triggered := mc.evaluateAlertRule(rule)
		
		alertID := fmt.Sprintf("%s_%s", rule.ID, "instance")
		existingAlert, exists := mc.alertManager.alerts[alertID]
		
		if triggered {
			if !exists || existingAlert.Status != AlertStatusFiring {
				// Create new alert
				alert := &Alert{
					ID:           alertID,
					RuleID:       rule.ID,
					RuleName:     rule.Name,
					Status:       AlertStatusFiring,
					Severity:     rule.Severity,
					Message:      fmt.Sprintf("Alert %s is firing", rule.Name),
					Threshold:    rule.Threshold,
					Labels:       rule.Labels,
					Annotations:  rule.Annotations,
					StartsAt:     time.Now(),
					UpdatedAt:    time.Now(),
				}
				mc.alertManager.alerts[alertID] = alert
			}
		} else {
			if exists && existingAlert.Status == AlertStatusFiring {
				// Resolve alert
				existingAlert.Status = AlertStatusResolved
				existingAlert.EndsAt = time.Now()
				existingAlert.UpdatedAt = time.Now()
			}
		}
	}
}

// evaluateAlertRule evaluates a single alert rule
func (mc *MetricsCollector) evaluateAlertRule(rule *AlertRule) bool {
	// Simplified rule evaluation - in practice this would parse the query
	// and evaluate it against the metrics
	
	// Example: evaluate CPU usage > 80%
	if rule.Query == "cpu_usage" {
		if gauge := mc.gauges["cpu_usage_percent"]; gauge != nil {
			value := gauge.Get()
			switch rule.Condition {
			case AlertConditionGreater:
				return value > rule.Threshold
			case AlertConditionLess:
				return value < rule.Threshold
			}
		}
	}
	
	return false
}

// exportMetrics exports metrics to configured endpoints
func (mc *MetricsCollector) exportMetrics() {
	snapshot := mc.CreateSnapshot()
	
	mc.exportersMutex.RLock()
	exporters := make([]MetricsExporter, len(mc.exporters))
	copy(exporters, mc.exporters)
	mc.exportersMutex.RUnlock()
	
	for _, exporter := range exporters {
		go func(exp MetricsExporter) {
			ctx, cancel := context.WithTimeout(mc.ctx, 30*time.Second)
			defer cancel()
			
			if err := exp.Export(ctx, snapshot); err != nil {
				// Log error (simplified)
				fmt.Printf("Failed to export metrics via %s: %v\n", exp.Name(), err)
			}
		}(exporter)
	}
}

// compactTimeSeries compacts time series data
func (mc *MetricsCollector) compactTimeSeries() {
	mc.timeSeries.mutex.Lock()
	defer mc.timeSeries.mutex.Unlock()
	
	for _, series := range mc.timeSeries.series {
		series.mutex.Lock()
		
		// Remove old data points
		cutoff := time.Now().Add(-mc.config.RetentionPeriod)
		var compactedPoints []DataPoint
		
		for _, point := range series.DataPoints {
			if point.Timestamp.After(cutoff) {
				compactedPoints = append(compactedPoints, point)
			}
		}
		
		series.DataPoints = compactedPoints
		if len(compactedPoints) > 0 {
			series.StartTime = compactedPoints[0].Timestamp
			series.EndTime = compactedPoints[len(compactedPoints)-1].Timestamp
		}
		
		series.mutex.Unlock()
	}
}

// cleanupOldData removes old metrics and data
func (mc *MetricsCollector) cleanupOldData() {
	cutoff := time.Now().Add(-mc.config.RetentionPeriod)
	
	// Cleanup alerts
	if mc.alertManager != nil {
		mc.alertManager.mutex.Lock()
		for id, alert := range mc.alertManager.alerts {
			if alert.Status == AlertStatusResolved && alert.EndsAt.Before(cutoff) {
				delete(mc.alertManager.alerts, id)
			}
		}
		mc.alertManager.mutex.Unlock()
	}
}

// HTTP server for metrics endpoint

// startHTTPServer starts the HTTP server for metrics endpoint
func (mc *MetricsCollector) startHTTPServer() {
	port := mc.config.HTTPServerPort
	if port == 0 {
		port = 9090 // Default Prometheus port
	}
	
	path := mc.config.HTTPServerPath
	if path == "" {
		path = "/metrics"
	}
	
	http.HandleFunc(path, mc.handleMetricsRequest)
	http.HandleFunc("/health", mc.handleHealthRequest)
	http.HandleFunc("/alerts", mc.handleAlertsRequest)
	
	addr := fmt.Sprintf(":%d", port)
	fmt.Printf("Starting metrics HTTP server on %s\n", addr)
	
	if err := http.ListenAndServe(addr, nil); err != nil {
		fmt.Printf("Metrics HTTP server error: %v\n", err)
	}
}

// handleMetricsRequest handles HTTP requests for metrics
func (mc *MetricsCollector) handleMetricsRequest(w http.ResponseWriter, r *http.Request) {
	snapshot := mc.CreateSnapshot()
	
	format := r.URL.Query().Get("format")
	if format == "" {
		format = mc.config.ExportFormat
	}
	
	switch format {
	case "prometheus":
		mc.writePrometheusFormat(w, snapshot)
	default:
		mc.writeJSONFormat(w, snapshot)
	}
}

// handleHealthRequest handles health check requests
func (mc *MetricsCollector) handleHealthRequest(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
	w.Write([]byte("OK"))
}

// handleAlertsRequest handles alert requests
func (mc *MetricsCollector) handleAlertsRequest(w http.ResponseWriter, r *http.Request) {
	alerts := mc.GetActiveAlerts()
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(alerts)
}

// writeJSONFormat writes metrics in JSON format
func (mc *MetricsCollector) writeJSONFormat(w http.ResponseWriter, snapshot *MetricsSnapshot) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(snapshot)
}

// writePrometheusFormat writes metrics in Prometheus format
func (mc *MetricsCollector) writePrometheusFormat(w http.ResponseWriter, snapshot *MetricsSnapshot) {
	w.Header().Set("Content-Type", "text/plain")
	
	// Write counters
	for _, counter := range snapshot.Counters {
		fmt.Fprintf(w, "# HELP %s %s\n", counter.Name, counter.Help)
		fmt.Fprintf(w, "# TYPE %s counter\n", counter.Name)
		fmt.Fprintf(w, "%s %d\n", mc.formatLabels(counter.Name, counter.Labels), counter.Value)
	}
	
	// Write gauges
	for _, gauge := range snapshot.Gauges {
		fmt.Fprintf(w, "# HELP %s %s\n", gauge.Name, gauge.Help)
		fmt.Fprintf(w, "# TYPE %s gauge\n", gauge.Name)
		fmt.Fprintf(w, "%s %f\n", mc.formatLabels(gauge.Name, gauge.Labels), gauge.Value)
	}
	
	// Write histograms
	for _, histogram := range snapshot.Histograms {
		fmt.Fprintf(w, "# HELP %s %s\n", histogram.Name, histogram.Help)
		fmt.Fprintf(w, "# TYPE %s histogram\n", histogram.Name)
		
		for i, bucket := range histogram.Buckets {
			labels := make(map[string]string)
			for k, v := range histogram.Labels {
				labels[k] = v
			}
			labels["le"] = fmt.Sprintf("%f", bucket)
			fmt.Fprintf(w, "%s_bucket %d\n", mc.formatLabels(histogram.Name, labels), histogram.Counts[i])
		}
		
		fmt.Fprintf(w, "%s_count %d\n", mc.formatLabels(histogram.Name, histogram.Labels), histogram.Count)
		fmt.Fprintf(w, "%s_sum %f\n", mc.formatLabels(histogram.Name, histogram.Labels), histogram.Sum)
	}
}

// formatLabels formats labels for Prometheus output
func (mc *MetricsCollector) formatLabels(name string, labels map[string]string) string {
	if len(labels) == 0 {
		return name
	}
	
	var labelPairs []string
	for k, v := range labels {
		labelPairs = append(labelPairs, fmt.Sprintf(`%s="%s"`, k, v))
	}
	
	return fmt.Sprintf("%s{%s}", name, strings.Join(labelPairs, ","))
}

// Utility methods

// CreateSnapshot creates a point-in-time snapshot of all metrics
func (mc *MetricsCollector) CreateSnapshot() *MetricsSnapshot {
	snapshot := &MetricsSnapshot{
		Timestamp:  time.Now(),
		Counters:   make(map[string]*Counter),
		Gauges:     make(map[string]*Gauge),
		Histograms: make(map[string]*Histogram),
		Timers:     make(map[string]*Timer),
		TimeSeries: make(map[string]*TimeSeries),
		Metadata:   make(map[string]interface{}),
	}
	
	// Copy counters
	mc.countersMutex.RLock()
	for k, v := range mc.counters {
		counterCopy := *v
		snapshot.Counters[k] = &counterCopy
	}
	mc.countersMutex.RUnlock()
	
	// Copy gauges
	mc.gaugesMutex.RLock()
	for k, v := range mc.gauges {
		gaugeCopy := *v
		snapshot.Gauges[k] = &gaugeCopy
	}
	mc.gaugesMutex.RUnlock()
	
	// Copy histograms
	mc.histogramsMutex.RLock()
	for k, v := range mc.histograms {
		histogramCopy := *v
		snapshot.Histograms[k] = &histogramCopy
	}
	mc.histogramsMutex.RUnlock()
	
	// Copy timers
	mc.timersMutex.RLock()
	for k, v := range mc.timers {
		timerCopy := *v
		snapshot.Timers[k] = &timerCopy
	}
	mc.timersMutex.RUnlock()
	
	// Add metadata
	snapshot.Metadata["collector_version"] = "1.0.0"
	snapshot.Metadata["start_time"] = mc.startTime
	snapshot.Metadata["uptime_seconds"] = time.Since(mc.startTime).Seconds()
	
	return snapshot
}

// buildMetricKey builds a unique key for a metric
func (mc *MetricsCollector) buildMetricKey(name string, labels map[string]string) string {
	if len(labels) == 0 {
		return name
	}
	
	key := name
	for k, v := range labels {
		key += fmt.Sprintf("_%s_%s", k, v)
	}
	
	return key
}

// registerBuiltinExporters registers built-in metrics exporters
func (mc *MetricsCollector) registerBuiltinExporters() {
	// Register console exporter
	consoleExporter := &ConsoleExporter{}
	mc.RegisterExporter(consoleExporter)
	
	// Register HTTP exporters based on configuration
	for _, endpoint := range mc.config.ExportEndpoints {
		if !endpoint.Enabled {
			continue
		}
		
		switch endpoint.Type {
		case "prometheus":
			httpExporter := &HTTPExporter{endpoint: endpoint}
			mc.RegisterExporter(httpExporter)
		}
	}
}

// RegisterExporter registers a metrics exporter
func (mc *MetricsCollector) RegisterExporter(exporter MetricsExporter) {
	mc.exportersMutex.Lock()
	mc.exporters = append(mc.exporters, exporter)
	mc.exportersMutex.Unlock()
}

// Built-in exporters

// ConsoleExporter exports metrics to console
type ConsoleExporter struct{}

func (ce *ConsoleExporter) Export(ctx context.Context, metrics *MetricsSnapshot) error {
	fmt.Printf("=== Metrics Snapshot at %s ===\n", metrics.Timestamp.Format(time.RFC3339))
	fmt.Printf("Counters: %d, Gauges: %d, Histograms: %d, Timers: %d\n",
		len(metrics.Counters), len(metrics.Gauges), len(metrics.Histograms), len(metrics.Timers))
	return nil
}

func (ce *ConsoleExporter) Name() string {
	return "console"
}

func (ce *ConsoleExporter) Configure(config map[string]interface{}) error {
	return nil
}

// HTTPExporter exports metrics via HTTP
type HTTPExporter struct {
	endpoint ExportEndpoint
}

func (he *HTTPExporter) Export(ctx context.Context, metrics *MetricsSnapshot) error {
	// Implementation would send metrics to HTTP endpoint
	return nil
}

func (he *HTTPExporter) Name() string {
	return fmt.Sprintf("http_%s", he.endpoint.Name)
}

func (he *HTTPExporter) Configure(config map[string]interface{}) error {
	return nil
}

// DefaultMetricsConfig returns default metrics configuration
func DefaultMetricsConfig() MetricsConfig {
	return MetricsConfig{
		EnableCollection:         true,
		CollectionInterval:       30 * time.Second,
		RetentionPeriod:          24 * time.Hour,
		MaxMetricsPerType:        10000,
		TimeSeriesBuckets:        1440, // 24 hours of minutes
		BucketDuration:           time.Minute,
		CompactionInterval:       time.Hour,
		DefaultHistogramBuckets:  []float64{0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100},
		MaxHistogramBuckets:      100,
		EnableAggregation:        true,
		AggregationWindow:        5 * time.Minute,
		AggregationFunctions:     []string{"sum", "avg", "min", "max", "count", "rate"},
		EnableAlerts:             true,
		AlertEvaluationInterval:  30 * time.Second,
		AlertRetentionPeriod:     7 * 24 * time.Hour,
		EnableExport:             true,
		ExportInterval:           60 * time.Second,
		ExportFormat:             "json",
		ExportEndpoints:          []ExportEndpoint{},
		MetricBufferSize:         1000,
		MaxConcurrentExports:     5,
		EnableHTTPServer:         true,
		HTTPServerPort:           9090,
		HTTPServerPath:           "/metrics",
	}
}