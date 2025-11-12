package monitoring

import (
	"context"
	"fmt"
	"sync"
	"time"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/exporters/jaeger"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.4.0"
	"go.opentelemetry.io/otel/trace"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
)

// ObservabilityIntegration provides OpenTelemetry tracing and structured logging
type ObservabilityIntegration struct {
	mu sync.RWMutex

	// Configuration
	serviceName string
	nodeID      string
	clusterID   string

	// OpenTelemetry
	tracerProvider *sdktrace.TracerProvider
	tracer         trace.Tracer

	// Structured logging
	logger        *zap.Logger
	logCollector  *LogCollector

	// Span tracking
	activeSpans   map[string]trace.Span
	spanHistory   []*SpanSummary
	maxSpanHistory int

	// Performance profiling
	profiler *PerformanceProfiler

	ctx    context.Context
	cancel context.CancelFunc
}

// LogCollector collects and aggregates structured logs
type LogCollector struct {
	mu           sync.RWMutex
	logs         []*StructuredLog
	maxLogs      int
	errorCount   int64
	warningCount int64
	infoCount    int64
}

// StructuredLog represents a structured log entry
type StructuredLog struct {
	Timestamp time.Time              `json:"timestamp"`
	Level     string                 `json:"level"`
	Component string                 `json:"component"`
	Message   string                 `json:"message"`
	Fields    map[string]interface{} `json:"fields"`
	TraceID   string                 `json:"trace_id,omitempty"`
	SpanID    string                 `json:"span_id,omitempty"`
}

// SpanSummary summarizes a completed span
type SpanSummary struct {
	TraceID      string
	SpanID       string
	Name         string
	StartTime    time.Time
	EndTime      time.Time
	Duration     time.Duration
	Status       string
	Attributes   map[string]interface{}
	Events       []SpanEvent
}

// SpanEvent represents an event within a span
type SpanEvent struct {
	Timestamp  time.Time
	Name       string
	Attributes map[string]interface{}
}

// PerformanceProfiler profiles CPU and memory usage
type PerformanceProfiler struct {
	mu sync.RWMutex

	// CPU profiling
	cpuSamples    []float64
	avgCPU        float64

	// Memory profiling
	memorySamples []uint64
	avgMemory     uint64

	// I/O profiling
	ioReadBytes   uint64
	ioWriteBytes  uint64
}

// NewObservabilityIntegration creates a new observability integration
func NewObservabilityIntegration(
	serviceName, nodeID, clusterID string,
	jaegerEndpoint string,
	logger *zap.Logger,
) (*ObservabilityIntegration, error) {
	ctx, cancel := context.WithCancel(context.Background())

	oi := &ObservabilityIntegration{
		serviceName:    serviceName,
		nodeID:         nodeID,
		clusterID:      clusterID,
		logger:         logger,
		logCollector:   newLogCollector(10000),
		activeSpans:    make(map[string]trace.Span),
		spanHistory:    make([]*SpanSummary, 0),
		maxSpanHistory: 1000,
		profiler:       &PerformanceProfiler{},
		ctx:            ctx,
		cancel:         cancel,
	}

	// Initialize OpenTelemetry
	if err := oi.initOpenTelemetry(jaegerEndpoint); err != nil {
		return nil, fmt.Errorf("failed to initialize OpenTelemetry: %w", err)
	}

	// Start background profiling
	go oi.profileLoop()

	return oi, nil
}

// initOpenTelemetry initializes OpenTelemetry with Jaeger exporter
func (oi *ObservabilityIntegration) initOpenTelemetry(endpoint string) error {
	// Create Jaeger exporter
	exp, err := jaeger.New(jaeger.WithCollectorEndpoint(jaeger.WithEndpoint(endpoint)))
	if err != nil {
		return fmt.Errorf("failed to create Jaeger exporter: %w", err)
	}

	// Create resource with service information
	res, err := resource.New(oi.ctx,
		resource.WithAttributes(
			semconv.ServiceNameKey.String(oi.serviceName),
			attribute.String("node.id", oi.nodeID),
			attribute.String("cluster.id", oi.clusterID),
			attribute.String("version", "v3"),
		),
	)
	if err != nil {
		return fmt.Errorf("failed to create resource: %w", err)
	}

	// Create tracer provider
	oi.tracerProvider = sdktrace.NewTracerProvider(
		sdktrace.WithBatcher(exp),
		sdktrace.WithResource(res),
		sdktrace.WithSampler(sdktrace.AlwaysSample()),
	)

	otel.SetTracerProvider(oi.tracerProvider)

	// Get tracer
	oi.tracer = oi.tracerProvider.Tracer(oi.serviceName)

	oi.logger.Info("OpenTelemetry initialized",
		zap.String("endpoint", endpoint),
		zap.String("service", oi.serviceName))

	return nil
}

// StartSpan starts a new trace span
func (oi *ObservabilityIntegration) StartSpan(
	ctx context.Context,
	name string,
	attrs ...attribute.KeyValue,
) (context.Context, trace.Span) {
	ctx, span := oi.tracer.Start(ctx, name, trace.WithAttributes(attrs...))

	// Track active span
	spanID := span.SpanContext().SpanID().String()
	oi.mu.Lock()
	oi.activeSpans[spanID] = span
	oi.mu.Unlock()

	return ctx, span
}

// EndSpan ends a trace span and records its summary
func (oi *ObservabilityIntegration) EndSpan(span trace.Span, err error) {
	if err != nil {
		span.RecordError(err)
		span.SetStatus(codes.Error, err.Error())
	} else {
		span.SetStatus(codes.Ok, "")
	}

	// Record span summary
	spanCtx := span.SpanContext()
	spanID := spanCtx.SpanID().String()

	oi.mu.Lock()
	delete(oi.activeSpans, spanID)
	oi.mu.Unlock()

	span.End()
}

// TraceMigration traces a complete VM migration
func (oi *ObservabilityIntegration) TraceMigration(
	ctx context.Context,
	vmID string,
	sourceNode string,
	destNode string,
	mode string,
) (context.Context, trace.Span) {
	return oi.StartSpan(ctx, "vm.migration",
		attribute.String("vm.id", vmID),
		attribute.String("source.node", sourceNode),
		attribute.String("dest.node", destNode),
		attribute.String("network.mode", mode),
	)
}

// TraceComponent traces a component operation
func (oi *ObservabilityIntegration) TraceComponent(
	ctx context.Context,
	component string,
	operation string,
) (context.Context, trace.Span) {
	return oi.StartSpan(ctx, fmt.Sprintf("%s.%s", component, operation),
		attribute.String("component", component),
		attribute.String("operation", operation),
	)
}

// RecordEvent records an event within the current span
func (oi *ObservabilityIntegration) RecordEvent(
	ctx context.Context,
	name string,
	attrs ...attribute.KeyValue,
) {
	span := trace.SpanFromContext(ctx)
	if span.IsRecording() {
		span.AddEvent(name, trace.WithAttributes(attrs...))
	}
}

// Log creates a structured log entry with trace correlation
func (oi *ObservabilityIntegration) Log(
	ctx context.Context,
	level zapcore.Level,
	component string,
	message string,
	fields ...zap.Field,
) {
	// Extract trace information from context
	span := trace.SpanFromContext(ctx)
	spanCtx := span.SpanContext()

	structuredFields := make(map[string]interface{})
	for _, field := range fields {
		structuredFields[field.Key] = field.Interface
	}

	logEntry := &StructuredLog{
		Timestamp: time.Now(),
		Level:     level.String(),
		Component: component,
		Message:   message,
		Fields:    structuredFields,
		TraceID:   spanCtx.TraceID().String(),
		SpanID:    spanCtx.SpanID().String(),
	}

	// Add to log collector
	oi.logCollector.Add(logEntry)

	// Log using zap
	zapFields := append(fields,
		zap.String("trace_id", logEntry.TraceID),
		zap.String("span_id", logEntry.SpanID),
		zap.String("component", component),
	)

	switch level {
	case zapcore.ErrorLevel:
		oi.logger.Error(message, zapFields...)
	case zapcore.WarnLevel:
		oi.logger.Warn(message, zapFields...)
	case zapcore.InfoLevel:
		oi.logger.Info(message, zapFields...)
	default:
		oi.logger.Debug(message, zapFields...)
	}
}

// SearchLogs searches logs with filters
func (oi *ObservabilityIntegration) SearchLogs(
	query string,
	level string,
	component string,
	since time.Time,
	limit int,
) []*StructuredLog {
	return oi.logCollector.Search(query, level, component, since, limit)
}

// GetRecentLogs returns recent logs
func (oi *ObservabilityIntegration) GetRecentLogs(limit int) []*StructuredLog {
	return oi.logCollector.GetRecent(limit)
}

// GetLogStats returns log statistics
func (oi *ObservabilityIntegration) GetLogStats() map[string]interface{} {
	oi.logCollector.mu.RLock()
	defer oi.logCollector.mu.RUnlock()

	return map[string]interface{}{
		"total_logs":   len(oi.logCollector.logs),
		"error_count":  oi.logCollector.errorCount,
		"warning_count": oi.logCollector.warningCount,
		"info_count":   oi.logCollector.infoCount,
	}
}

// GetActiveSpans returns currently active spans
func (oi *ObservabilityIntegration) GetActiveSpans() []map[string]interface{} {
	oi.mu.RLock()
	defer oi.mu.RUnlock()

	spans := make([]map[string]interface{}, 0, len(oi.activeSpans))
	for spanID, span := range oi.activeSpans {
		spanCtx := span.SpanContext()
		spans = append(spans, map[string]interface{}{
			"span_id":  spanID,
			"trace_id": spanCtx.TraceID().String(),
		})
	}

	return spans
}

// GetSpanHistory returns historical span summaries
func (oi *ObservabilityIntegration) GetSpanHistory(limit int) []*SpanSummary {
	oi.mu.RLock()
	defer oi.mu.RUnlock()

	if limit > len(oi.spanHistory) {
		limit = len(oi.spanHistory)
	}

	history := make([]*SpanSummary, limit)
	copy(history, oi.spanHistory[len(oi.spanHistory)-limit:])

	return history
}

// Profile methods

func (oi *ObservabilityIntegration) profileLoop() {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-oi.ctx.Done():
			return
		case <-ticker.C:
			oi.collectProfileData()
		}
	}
}

func (oi *ObservabilityIntegration) collectProfileData() {
	// TODO: Implement actual profiling using pprof or similar
	// This is a placeholder for demonstration

	oi.profiler.mu.Lock()
	defer oi.profiler.mu.Unlock()

	// Collect CPU sample (placeholder)
	cpuSample := 45.0 // Would get from runtime
	oi.profiler.cpuSamples = append(oi.profiler.cpuSamples, cpuSample)
	if len(oi.profiler.cpuSamples) > 100 {
		oi.profiler.cpuSamples = oi.profiler.cpuSamples[1:]
	}

	// Calculate average
	sum := 0.0
	for _, s := range oi.profiler.cpuSamples {
		sum += s
	}
	oi.profiler.avgCPU = sum / float64(len(oi.profiler.cpuSamples))
}

// GetProfilingData returns performance profiling data
func (oi *ObservabilityIntegration) GetProfilingData() map[string]interface{} {
	oi.profiler.mu.RLock()
	defer oi.profiler.mu.RUnlock()

	return map[string]interface{}{
		"avg_cpu_percent": oi.profiler.avgCPU,
		"avg_memory_mb":   oi.profiler.avgMemory,
		"io_read_bytes":   oi.profiler.ioReadBytes,
		"io_write_bytes":  oi.profiler.ioWriteBytes,
	}
}

// Close shuts down observability integration
func (oi *ObservabilityIntegration) Close() error {
	oi.cancel()

	if oi.tracerProvider != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := oi.tracerProvider.Shutdown(ctx); err != nil {
			return fmt.Errorf("failed to shutdown tracer provider: %w", err)
		}
	}

	oi.logger.Info("Observability integration closed")
	return nil
}

// LogCollector methods

func newLogCollector(maxLogs int) *LogCollector {
	return &LogCollector{
		logs:    make([]*StructuredLog, 0, maxLogs),
		maxLogs: maxLogs,
	}
}

func (lc *LogCollector) Add(log *StructuredLog) {
	lc.mu.Lock()
	defer lc.mu.Unlock()

	lc.logs = append(lc.logs, log)
	if len(lc.logs) > lc.maxLogs {
		lc.logs = lc.logs[1:]
	}

	// Update counters
	switch log.Level {
	case "error":
		lc.errorCount++
	case "warn":
		lc.warningCount++
	case "info":
		lc.infoCount++
	}
}

func (lc *LogCollector) Search(
	query string,
	level string,
	component string,
	since time.Time,
	limit int,
) []*StructuredLog {
	lc.mu.RLock()
	defer lc.mu.RUnlock()

	results := make([]*StructuredLog, 0)

	for _, log := range lc.logs {
		// Filter by time
		if log.Timestamp.Before(since) {
			continue
		}

		// Filter by level
		if level != "" && log.Level != level {
			continue
		}

		// Filter by component
		if component != "" && log.Component != component {
			continue
		}

		// Simple text search in message
		if query != "" {
			// Would use more sophisticated search in production
			// For now, just check if query is in message
			// (placeholder implementation)
		}

		results = append(results, log)

		if len(results) >= limit {
			break
		}
	}

	return results
}

func (lc *LogCollector) GetRecent(limit int) []*StructuredLog {
	lc.mu.RLock()
	defer lc.mu.RUnlock()

	if limit > len(lc.logs) {
		limit = len(lc.logs)
	}

	recent := make([]*StructuredLog, limit)
	copy(recent, lc.logs[len(lc.logs)-limit:])

	return recent
}
