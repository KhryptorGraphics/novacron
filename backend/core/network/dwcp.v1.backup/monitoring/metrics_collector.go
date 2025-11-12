// Package monitoring provides comprehensive multi-region monitoring and observability
package monitoring

import (
	"context"
	"fmt"
	"sync"
	"time"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/exporters/prometheus"
	"go.opentelemetry.io/otel/metric"
	sdkmetric "go.opentelemetry.io/otel/sdk/metric"
)

// MetricType defines the type of metric
type MetricType int

const (
	MetricTypeCounter MetricType = iota
	MetricTypeGauge
	MetricTypeHistogram
	MetricTypeSummary
)

// MetricData represents collected metric data
type MetricData struct {
	Name      string
	Type      MetricType
	Value     float64
	Labels    map[string]string
	Timestamp time.Time
	Region    string
}

// MetricsCollector handles distributed metrics collection
type MetricsCollector struct {
	mu               sync.RWMutex
	region           string
	meterProvider    *sdkmetric.MeterProvider
	meter            metric.Meter

	// Counters
	requestCounter   metric.Int64Counter
	errorCounter     metric.Int64Counter

	// Gauges
	cpuGauge         metric.Float64ObservableGauge
	memoryGauge      metric.Float64ObservableGauge
	connectionGauge  metric.Int64ObservableGauge

	// Histograms
	latencyHistogram metric.Float64Histogram
	responseHistogram metric.Float64Histogram

	// Storage
	metrics          map[string]*MetricData
	aggregations     map[string]*AggregatedMetric

	// Configuration
	granularity      time.Duration
	retentionPeriod  time.Duration
}

// AggregatedMetric represents aggregated metric data
type AggregatedMetric struct {
	Name       string
	Count      int64
	Sum        float64
	Min        float64
	Max        float64
	Mean       float64
	P50        float64
	P95        float64
	P99        float64
	P999       float64
	Timestamp  time.Time
}

// NewMetricsCollector creates a new metrics collector
func NewMetricsCollector(region string) (*MetricsCollector, error) {
	// Create Prometheus exporter
	exporter, err := prometheus.New()
	if err != nil {
		return nil, fmt.Errorf("failed to create prometheus exporter: %w", err)
	}

	// Create meter provider
	provider := sdkmetric.NewMeterProvider(
		sdkmetric.WithReader(exporter),
	)

	// Set global meter provider
	otel.SetMeterProvider(provider)

	// Create meter
	meter := provider.Meter("dwcp-monitoring")

	mc := &MetricsCollector{
		region:          region,
		meterProvider:   provider,
		meter:           meter,
		metrics:         make(map[string]*MetricData),
		aggregations:    make(map[string]*AggregatedMetric),
		granularity:     time.Second,
		retentionPeriod: 24 * time.Hour,
	}

	// Initialize instruments
	if err := mc.initializeInstruments(); err != nil {
		return nil, fmt.Errorf("failed to initialize instruments: %w", err)
	}

	return mc, nil
}

// initializeInstruments initializes OpenTelemetry instruments
func (mc *MetricsCollector) initializeInstruments() error {
	var err error

	// Counters
	mc.requestCounter, err = mc.meter.Int64Counter(
		"dwcp.requests.total",
		metric.WithDescription("Total number of DWCP requests"),
		metric.WithUnit("{requests}"),
	)
	if err != nil {
		return fmt.Errorf("failed to create request counter: %w", err)
	}

	mc.errorCounter, err = mc.meter.Int64Counter(
		"dwcp.errors.total",
		metric.WithDescription("Total number of DWCP errors"),
		metric.WithUnit("{errors}"),
	)
	if err != nil {
		return fmt.Errorf("failed to create error counter: %w", err)
	}

	// Histograms
	mc.latencyHistogram, err = mc.meter.Float64Histogram(
		"dwcp.latency",
		metric.WithDescription("DWCP operation latency"),
		metric.WithUnit("ms"),
	)
	if err != nil {
		return fmt.Errorf("failed to create latency histogram: %w", err)
	}

	mc.responseHistogram, err = mc.meter.Float64Histogram(
		"dwcp.response_time",
		metric.WithDescription("DWCP response time"),
		metric.WithUnit("ms"),
	)
	if err != nil {
		return fmt.Errorf("failed to create response histogram: %w", err)
	}

	return nil
}

// RecordRequest records a request counter
func (mc *MetricsCollector) RecordRequest(ctx context.Context, operation string, labels map[string]string) {
	attrs := mc.buildAttributes(operation, labels)
	mc.requestCounter.Add(ctx, 1, metric.WithAttributes(attrs...))
}

// RecordError records an error counter
func (mc *MetricsCollector) RecordError(ctx context.Context, operation string, errorType string, labels map[string]string) {
	attrs := mc.buildAttributes(operation, labels)
	attrs = append(attrs, attribute.String("error_type", errorType))
	mc.errorCounter.Add(ctx, 1, metric.WithAttributes(attrs...))
}

// RecordLatency records latency histogram
func (mc *MetricsCollector) RecordLatency(ctx context.Context, operation string, latencyMs float64, labels map[string]string) {
	attrs := mc.buildAttributes(operation, labels)
	mc.latencyHistogram.Record(ctx, latencyMs, metric.WithAttributes(attrs...))

	// Store for aggregation
	mc.mu.Lock()
	defer mc.mu.Unlock()

	key := fmt.Sprintf("%s_%s_latency", mc.region, operation)
	mc.metrics[key] = &MetricData{
		Name:      "latency",
		Type:      MetricTypeHistogram,
		Value:     latencyMs,
		Labels:    labels,
		Timestamp: time.Now(),
		Region:    mc.region,
	}
}

// RecordResponseTime records response time histogram
func (mc *MetricsCollector) RecordResponseTime(ctx context.Context, operation string, responseTimeMs float64, labels map[string]string) {
	attrs := mc.buildAttributes(operation, labels)
	mc.responseHistogram.Record(ctx, responseTimeMs, metric.WithAttributes(attrs...))
}

// RecordGauge records a gauge metric
func (mc *MetricsCollector) RecordGauge(name string, value float64, labels map[string]string) {
	mc.mu.Lock()
	defer mc.mu.Unlock()

	key := fmt.Sprintf("%s_%s", mc.region, name)
	mc.metrics[key] = &MetricData{
		Name:      name,
		Type:      MetricTypeGauge,
		Value:     value,
		Labels:    labels,
		Timestamp: time.Now(),
		Region:    mc.region,
	}
}

// AggregateMetrics aggregates metrics across regions
func (mc *MetricsCollector) AggregateMetrics(regionMetrics map[string][]*MetricData) map[string]*AggregatedMetric {
	mc.mu.Lock()
	defer mc.mu.Unlock()

	aggregated := make(map[string]*AggregatedMetric)

	for metricName, metrics := range regionMetrics {
		if len(metrics) == 0 {
			continue
		}

		agg := &AggregatedMetric{
			Name:      metricName,
			Count:     int64(len(metrics)),
			Min:       metrics[0].Value,
			Max:       metrics[0].Value,
			Timestamp: time.Now(),
		}

		var sum float64
		values := make([]float64, len(metrics))

		for i, m := range metrics {
			sum += m.Value
			values[i] = m.Value

			if m.Value < agg.Min {
				agg.Min = m.Value
			}
			if m.Value > agg.Max {
				agg.Max = m.Value
			}
		}

		agg.Sum = sum
		agg.Mean = sum / float64(len(metrics))

		// Calculate percentiles
		agg.P50 = calculatePercentile(values, 0.50)
		agg.P95 = calculatePercentile(values, 0.95)
		agg.P99 = calculatePercentile(values, 0.99)
		agg.P999 = calculatePercentile(values, 0.999)

		aggregated[metricName] = agg
		mc.aggregations[metricName] = agg
	}

	return aggregated
}

// GetMetrics retrieves metrics for a time range
func (mc *MetricsCollector) GetMetrics(since time.Time) []*MetricData {
	mc.mu.RLock()
	defer mc.mu.RUnlock()

	var result []*MetricData
	for _, metric := range mc.metrics {
		if metric.Timestamp.After(since) {
			result = append(result, metric)
		}
	}

	return result
}

// GetAggregation retrieves aggregated metrics
func (mc *MetricsCollector) GetAggregation(metricName string) (*AggregatedMetric, bool) {
	mc.mu.RLock()
	defer mc.mu.RUnlock()

	agg, ok := mc.aggregations[metricName]
	return agg, ok
}

// Cleanup removes old metrics
func (mc *MetricsCollector) Cleanup() {
	mc.mu.Lock()
	defer mc.mu.Unlock()

	cutoff := time.Now().Add(-mc.retentionPeriod)

	for key, metric := range mc.metrics {
		if metric.Timestamp.Before(cutoff) {
			delete(mc.metrics, key)
		}
	}
}

// Shutdown gracefully shuts down the metrics collector
func (mc *MetricsCollector) Shutdown(ctx context.Context) error {
	return mc.meterProvider.Shutdown(ctx)
}

// buildAttributes builds OpenTelemetry attributes
func (mc *MetricsCollector) buildAttributes(operation string, labels map[string]string) []attribute.KeyValue {
	attrs := []attribute.KeyValue{
		attribute.String("region", mc.region),
		attribute.String("operation", operation),
	}

	for k, v := range labels {
		attrs = append(attrs, attribute.String(k, v))
	}

	return attrs
}

// calculatePercentile calculates percentile from values
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

// StartAutoCleanup starts automatic cleanup of old metrics
func (mc *MetricsCollector) StartAutoCleanup(ctx context.Context, interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			mc.Cleanup()
		}
	}
}
