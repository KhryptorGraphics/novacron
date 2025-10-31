package observability

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// MetricsCollector collects and exports metrics
type MetricsCollector struct {
	mu         sync.RWMutex
	metrics    map[string]*Metric
	exporters  []MetricsExporter
	config     *MetricsConfig
}

// Metric represents a metric
type Metric struct {
	Name      string
	Type      MetricType
	Value     float64
	Labels    map[string]string
	Timestamp time.Time
}

// MetricType represents type of metric
type MetricType string

const (
	MetricTypeCounter   MetricType = "counter"
	MetricTypeGauge     MetricType = "gauge"
	MetricTypeHistogram MetricType = "histogram"
	MetricTypeSummary   MetricType = "summary"
)

// MetricsConfig configuration for metrics
type MetricsConfig struct {
	EnablePrometheus bool
	EnableGraphite   bool
	ExportInterval   time.Duration
	RetentionPeriod  time.Duration
}

// MetricsExporter exports metrics to external systems
type MetricsExporter interface {
	Export(ctx context.Context, metrics []*Metric) error
	Name() string
}

// NewMetricsCollector creates a new metrics collector
func NewMetricsCollector(config *MetricsConfig) *MetricsCollector {
	mc := &MetricsCollector{
		metrics:   make(map[string]*Metric),
		exporters: make([]MetricsExporter, 0),
		config:    config,
	}
	
	// Initialize exporters
	if config.EnablePrometheus {
		mc.exporters = append(mc.exporters, NewPrometheusExporter())
	}
	
	if config.EnableGraphite {
		mc.exporters = append(mc.exporters, NewGraphiteExporter())
	}
	
	return mc
}

// RecordMetric records a metric
func (mc *MetricsCollector) RecordMetric(name string, metricType MetricType, value float64, labels map[string]string) {
	mc.mu.Lock()
	defer mc.mu.Unlock()
	
	metric := &Metric{
		Name:      name,
		Type:      metricType,
		Value:     value,
		Labels:    labels,
		Timestamp: time.Now(),
	}
	
	mc.metrics[name] = metric
}

// GetMetric returns a metric by name
func (mc *MetricsCollector) GetMetric(name string) (*Metric, error) {
	mc.mu.RLock()
	defer mc.mu.RUnlock()
	
	metric, exists := mc.metrics[name]
	if !exists {
		return nil, fmt.Errorf("metric %s not found", name)
	}
	
	return metric, nil
}

// StartExporting starts exporting metrics
func (mc *MetricsCollector) StartExporting(ctx context.Context) {
	ticker := time.NewTicker(mc.config.ExportInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			mc.exportMetrics(ctx)
		}
	}
}

// exportMetrics exports metrics to all exporters
func (mc *MetricsCollector) exportMetrics(ctx context.Context) {
	mc.mu.RLock()
	metrics := make([]*Metric, 0, len(mc.metrics))
	for _, metric := range mc.metrics {
		metrics = append(metrics, metric)
	}
	mc.mu.RUnlock()
	
	for _, exporter := range mc.exporters {
		if err := exporter.Export(ctx, metrics); err != nil {
			fmt.Printf("Failed to export to %s: %v\n", exporter.Name(), err)
		}
	}
}

// PrometheusExporter exports metrics to Prometheus
type PrometheusExporter struct{}

func NewPrometheusExporter() *PrometheusExporter {
	return &PrometheusExporter{}
}

func (pe *PrometheusExporter) Export(ctx context.Context, metrics []*Metric) error {
	// Implement Prometheus export
	return nil
}

func (pe *PrometheusExporter) Name() string {
	return "prometheus"
}

// GraphiteExporter exports metrics to Graphite
type GraphiteExporter struct{}

func NewGraphiteExporter() *GraphiteExporter {
	return &GraphiteExporter{}
}

func (ge *GraphiteExporter) Export(ctx context.Context, metrics []*Metric) error {
	// Implement Graphite export
	return nil
}

func (ge *GraphiteExporter) Name() string {
	return "graphite"
}

// TracingManager manages distributed tracing
type TracingManager struct {
	mu     sync.RWMutex
	spans  map[string]*Span
	config *TracingConfig
}

// Span represents a trace span
type Span struct {
	TraceID    string
	SpanID     string
	ParentID   string
	Operation  string
	StartTime  time.Time
	EndTime    time.Time
	Duration   time.Duration
	Tags       map[string]string
	Logs       []SpanLog
	Status     SpanStatus
}

// SpanLog represents a log entry in a span
type SpanLog struct {
	Timestamp time.Time
	Message   string
	Fields    map[string]interface{}
}

// SpanStatus represents span status
type SpanStatus string

const (
	SpanStatusOK    SpanStatus = "ok"
	SpanStatusError SpanStatus = "error"
)

// TracingConfig configuration for tracing
type TracingConfig struct {
	EnableJaeger     bool
	EnableZipkin     bool
	SamplingRate     float64
	MaxSpansPerTrace int
}

// NewTracingManager creates a new tracing manager
func NewTracingManager(config *TracingConfig) *TracingManager {
	return &TracingManager{
		spans:  make(map[string]*Span),
		config: config,
	}
}

// StartSpan starts a new span
func (tm *TracingManager) StartSpan(ctx context.Context, operation string, parentID string) *Span {
	span := &Span{
		TraceID:   fmt.Sprintf("trace-%d", time.Now().UnixNano()),
		SpanID:    fmt.Sprintf("span-%d", time.Now().UnixNano()),
		ParentID:  parentID,
		Operation: operation,
		StartTime: time.Now(),
		Tags:      make(map[string]string),
		Logs:      make([]SpanLog, 0),
		Status:    SpanStatusOK,
	}
	
	tm.mu.Lock()
	tm.spans[span.SpanID] = span
	tm.mu.Unlock()
	
	return span
}

// FinishSpan finishes a span
func (tm *TracingManager) FinishSpan(spanID string) {
	tm.mu.Lock()
	defer tm.mu.Unlock()
	
	if span, exists := tm.spans[spanID]; exists {
		span.EndTime = time.Now()
		span.Duration = span.EndTime.Sub(span.StartTime)
	}
}

// AddSpanLog adds a log to a span
func (tm *TracingManager) AddSpanLog(spanID, message string, fields map[string]interface{}) {
	tm.mu.Lock()
	defer tm.mu.Unlock()
	
	if span, exists := tm.spans[spanID]; exists {
		span.Logs = append(span.Logs, SpanLog{
			Timestamp: time.Now(),
			Message:   message,
			Fields:    fields,
		})
	}
}

