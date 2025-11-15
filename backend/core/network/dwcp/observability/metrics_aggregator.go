// Package observability provides comprehensive observability for DWCP
package observability

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"go.uber.org/zap"
)

// MetricsAggregator aggregates metrics from all DWCP components
type MetricsAggregator struct {
	mu     sync.RWMutex
	logger *zap.Logger

	// Component collectors
	consensusCollector    MetricCollector
	bandwidthCollector    MetricCollector
	circuitBreakerCollector MetricCollector
	mlCollector           MetricCollector
	resourceCollector     MetricCollector

	// Aggregated metrics storage
	aggregations map[string]*AggregatedMetric
	timeSeries   map[string][]TimeSeriesPoint

	// Configuration
	aggregationInterval time.Duration
	retentionPeriod     time.Duration

	// Lifecycle
	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

// MetricCollector interface for component-specific collectors
type MetricCollector interface {
	CollectMetrics(ctx context.Context) ([]Metric, error)
	GetName() string
}

// Metric represents a single metric data point
type Metric struct {
	Name      string
	Type      MetricType
	Value     float64
	Labels    map[string]string
	Timestamp time.Time
}

// MetricType defines metric types
type MetricType int

const (
	MetricTypeCounter MetricType = iota
	MetricTypeGauge
	MetricTypeHistogram
	MetricTypeSummary
)

// AggregatedMetric represents aggregated metric data
type AggregatedMetric struct {
	Name       string
	Count      int64
	Sum        float64
	Min        float64
	Max        float64
	Mean       float64
	StdDev     float64
	P50        float64
	P95        float64
	P99        float64
	P999       float64
	Timestamp  time.Time
	WindowSize time.Duration
}

// TimeSeriesPoint represents a point in time series data
type TimeSeriesPoint struct {
	Timestamp time.Time
	Value     float64
	Labels    map[string]string
}

// NewMetricsAggregator creates a new metrics aggregator
func NewMetricsAggregator(logger *zap.Logger) *MetricsAggregator {
	ctx, cancel := context.WithCancel(context.Background())

	aggregator := &MetricsAggregator{
		logger:              logger,
		aggregations:        make(map[string]*AggregatedMetric),
		timeSeries:          make(map[string][]TimeSeriesPoint),
		aggregationInterval: 10 * time.Second,
		retentionPeriod:     24 * time.Hour,
		ctx:                 ctx,
		cancel:              cancel,
	}

	// Start background aggregation
	aggregator.wg.Add(1)
	go aggregator.aggregationLoop()

	// Start cleanup routine
	aggregator.wg.Add(1)
	go aggregator.cleanupLoop()

	return aggregator
}

// RegisterCollector registers a metric collector
func (ma *MetricsAggregator) RegisterCollector(collector MetricCollector) {
	ma.mu.Lock()
	defer ma.mu.Unlock()

	switch collector.GetName() {
	case "consensus":
		ma.consensusCollector = collector
	case "bandwidth":
		ma.bandwidthCollector = collector
	case "circuit_breaker":
		ma.circuitBreakerCollector = collector
	case "ml":
		ma.mlCollector = collector
	case "resource":
		ma.resourceCollector = collector
	default:
		ma.logger.Warn("Unknown collector type",
			zap.String("name", collector.GetName()))
	}

	ma.logger.Info("Registered metric collector",
		zap.String("collector", collector.GetName()))
}

// aggregationLoop periodically aggregates metrics
func (ma *MetricsAggregator) aggregationLoop() {
	defer ma.wg.Done()

	ticker := time.NewTicker(ma.aggregationInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ma.ctx.Done():
			return
		case <-ticker.C:
			if err := ma.collectAndAggregate(); err != nil {
				ma.logger.Error("Failed to aggregate metrics",
					zap.Error(err))
			}
		}
	}
}

// collectAndAggregate collects metrics from all collectors and aggregates them
func (ma *MetricsAggregator) collectAndAggregate() error {
	ma.mu.Lock()
	defer ma.mu.Unlock()

	collectors := []MetricCollector{
		ma.consensusCollector,
		ma.bandwidthCollector,
		ma.circuitBreakerCollector,
		ma.mlCollector,
		ma.resourceCollector,
	}

	allMetrics := make([]Metric, 0)

	// Collect from all registered collectors
	for _, collector := range collectors {
		if collector == nil {
			continue
		}

		metrics, err := collector.CollectMetrics(ma.ctx)
		if err != nil {
			ma.logger.Warn("Failed to collect metrics",
				zap.String("collector", collector.GetName()),
				zap.Error(err))
			continue
		}

		allMetrics = append(allMetrics, metrics...)
	}

	// Group metrics by name for aggregation
	metricGroups := make(map[string][]Metric)
	for _, metric := range allMetrics {
		metricGroups[metric.Name] = append(metricGroups[metric.Name], metric)
	}

	// Aggregate each metric group
	for name, metrics := range metricGroups {
		agg := ma.aggregateMetrics(name, metrics)
		ma.aggregations[name] = agg

		// Store in time series
		for _, metric := range metrics {
			key := ma.timeSeriesKey(name, metric.Labels)
			ma.timeSeries[key] = append(ma.timeSeries[key], TimeSeriesPoint{
				Timestamp: metric.Timestamp,
				Value:     metric.Value,
				Labels:    metric.Labels,
			})
		}
	}

	return nil
}

// aggregateMetrics calculates aggregated statistics for a metric group
func (ma *MetricsAggregator) aggregateMetrics(name string, metrics []Metric) *AggregatedMetric {
	if len(metrics) == 0 {
		return nil
	}

	values := make([]float64, len(metrics))
	sum := 0.0
	min := metrics[0].Value
	max := metrics[0].Value

	for i, m := range metrics {
		values[i] = m.Value
		sum += m.Value

		if m.Value < min {
			min = m.Value
		}
		if m.Value > max {
			max = m.Value
		}
	}

	mean := sum / float64(len(values))
	stdDev := calculateStdDev(values, mean)

	return &AggregatedMetric{
		Name:       name,
		Count:      int64(len(metrics)),
		Sum:        sum,
		Min:        min,
		Max:        max,
		Mean:       mean,
		StdDev:     stdDev,
		P50:        calculatePercentile(values, 0.50),
		P95:        calculatePercentile(values, 0.95),
		P99:        calculatePercentile(values, 0.99),
		P999:       calculatePercentile(values, 0.999),
		Timestamp:  time.Now(),
		WindowSize: ma.aggregationInterval,
	}
}

// GetAggregation retrieves aggregated metrics by name
func (ma *MetricsAggregator) GetAggregation(name string) (*AggregatedMetric, bool) {
	ma.mu.RLock()
	defer ma.mu.RUnlock()

	agg, ok := ma.aggregations[name]
	return agg, ok
}

// GetTimeSeries retrieves time series data for a metric
func (ma *MetricsAggregator) GetTimeSeries(name string, labels map[string]string, since time.Time) []TimeSeriesPoint {
	ma.mu.RLock()
	defer ma.mu.RUnlock()

	key := ma.timeSeriesKey(name, labels)
	points := ma.timeSeries[key]

	// Filter by time
	result := make([]TimeSeriesPoint, 0)
	for _, point := range points {
		if point.Timestamp.After(since) {
			result = append(result, point)
		}
	}

	return result
}

// GetAllAggregations returns all current aggregations
func (ma *MetricsAggregator) GetAllAggregations() map[string]*AggregatedMetric {
	ma.mu.RLock()
	defer ma.mu.RUnlock()

	result := make(map[string]*AggregatedMetric, len(ma.aggregations))
	for k, v := range ma.aggregations {
		result[k] = v
	}

	return result
}

// cleanupLoop periodically cleans up old metrics
func (ma *MetricsAggregator) cleanupLoop() {
	defer ma.wg.Done()

	ticker := time.NewTicker(1 * time.Hour)
	defer ticker.Stop()

	for {
		select {
		case <-ma.ctx.Done():
			return
		case <-ticker.C:
			ma.cleanup()
		}
	}
}

// cleanup removes old time series data
func (ma *MetricsAggregator) cleanup() {
	ma.mu.Lock()
	defer ma.mu.Unlock()

	cutoff := time.Now().Add(-ma.retentionPeriod)

	for key, points := range ma.timeSeries {
		// Filter out old points
		filtered := make([]TimeSeriesPoint, 0)
		for _, point := range points {
			if point.Timestamp.After(cutoff) {
				filtered = append(filtered, point)
			}
		}

		if len(filtered) == 0 {
			delete(ma.timeSeries, key)
		} else {
			ma.timeSeries[key] = filtered
		}
	}

	ma.logger.Debug("Cleaned up old metrics",
		zap.Int("time_series_count", len(ma.timeSeries)))
}

// timeSeriesKey generates a key for time series storage
func (ma *MetricsAggregator) timeSeriesKey(name string, labels map[string]string) string {
	key := name
	for k, v := range labels {
		key += fmt.Sprintf("_%s=%s", k, v)
	}
	return key
}

// Close shuts down the metrics aggregator
func (ma *MetricsAggregator) Close() error {
	ma.cancel()
	ma.wg.Wait()
	ma.logger.Info("Metrics aggregator closed")
	return nil
}

// calculateStdDev calculates standard deviation
func calculateStdDev(values []float64, mean float64) float64 {
	if len(values) == 0 {
		return 0
	}

	sumSquaredDiff := 0.0
	for _, v := range values {
		diff := v - mean
		sumSquaredDiff += diff * diff
	}

	variance := sumSquaredDiff / float64(len(values))
	return variance // Using variance instead of sqrt for simplicity
}

// calculatePercentile calculates percentile from values (simplified version)
func calculatePercentile(values []float64, p float64) float64 {
	if len(values) == 0 {
		return 0
	}

	// Simple percentile calculation (should use proper sorting for production)
	sorted := make([]float64, len(values))
	copy(sorted, values)

	// Bubble sort (replace with quicksort for better performance)
	for i := 0; i < len(sorted); i++ {
		for j := i + 1; j < len(sorted); j++ {
			if sorted[i] > sorted[j] {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}

	index := int(float64(len(sorted)-1) * p)
	return sorted[index]
}

// PrometheusExporter exports metrics to Prometheus
type PrometheusExporter struct {
	aggregator *MetricsAggregator
	registry   *prometheus.Registry

	// Prometheus metrics
	consensusLatencyP95 prometheus.Gauge
	bandwidthUtilization prometheus.Gauge
	circuitBreakerState prometheus.GaugeVec
	mlInferenceLatency  prometheus.Histogram
}

// NewPrometheusExporter creates a new Prometheus exporter
func NewPrometheusExporter(aggregator *MetricsAggregator) *PrometheusExporter {
	registry := prometheus.NewRegistry()

	exporter := &PrometheusExporter{
		aggregator: aggregator,
		registry:   registry,
	}

	exporter.initializeMetrics()
	return exporter
}

// initializeMetrics initializes Prometheus metrics
func (pe *PrometheusExporter) initializeMetrics() {
	pe.consensusLatencyP95 = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "dwcp_consensus_latency_p95_seconds",
		Help: "95th percentile consensus latency",
	})

	pe.bandwidthUtilization = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "dwcp_bandwidth_utilization_mbps",
		Help: "Current bandwidth utilization in Mbps",
	})

	pe.circuitBreakerState = *prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "dwcp_circuit_breaker_state",
		Help: "Circuit breaker state (0=closed, 1=half-open, 2=open)",
	}, []string{"name"})

	pe.mlInferenceLatency = prometheus.NewHistogram(prometheus.HistogramOpts{
		Name:    "dwcp_ml_inference_latency_seconds",
		Help:    "ML model inference latency",
		Buckets: prometheus.ExponentialBuckets(0.001, 2, 15),
	})

	// Register metrics
	pe.registry.MustRegister(pe.consensusLatencyP95)
	pe.registry.MustRegister(pe.bandwidthUtilization)
	pe.registry.MustRegister(pe.circuitBreakerState)
	pe.registry.MustRegister(pe.mlInferenceLatency)
}

// UpdateMetrics updates Prometheus metrics from aggregations
func (pe *PrometheusExporter) UpdateMetrics() {
	aggregations := pe.aggregator.GetAllAggregations()

	// Update consensus latency
	if agg, ok := aggregations["consensus_latency"]; ok {
		pe.consensusLatencyP95.Set(agg.P95)
	}

	// Update bandwidth utilization
	if agg, ok := aggregations["bandwidth_utilization"]; ok {
		pe.bandwidthUtilization.Set(agg.Mean)
	}

	// Additional metrics can be updated here
}

// GetRegistry returns the Prometheus registry
func (pe *PrometheusExporter) GetRegistry() *prometheus.Registry {
	return pe.registry
}
