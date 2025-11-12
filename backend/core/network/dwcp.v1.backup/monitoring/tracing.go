package monitoring

import (
	"context"
	"fmt"
	"sync"
	"time"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/exporters/jaeger"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.4.0"
	"go.opentelemetry.io/otel/trace"
)

// SamplingStrategy defines trace sampling strategy
type SamplingStrategy int

const (
	SamplingHead SamplingStrategy = iota
	SamplingTail
	SamplingAdaptive
)

// TraceConfig configures distributed tracing
type TraceConfig struct {
	ServiceName      string
	JaegerEndpoint   string
	SamplingRate     float64
	Strategy         SamplingStrategy
	MaxSpansPerTrace int
}

// TracingSystem manages distributed tracing
type TracingSystem struct {
	mu             sync.RWMutex
	config         *TraceConfig
	tracerProvider *sdktrace.TracerProvider
	tracer         trace.Tracer

	// Trace storage
	traces         map[string]*StoredTrace
	spans          map[string]*StoredSpan

	// Sampling
	sampler        sdktrace.Sampler

	// Statistics
	totalTraces    int64
	sampledTraces  int64
	droppedTraces  int64
}

// StoredTrace represents a stored trace
type StoredTrace struct {
	TraceID    string
	SpanCount  int
	StartTime  time.Time
	EndTime    time.Time
	Duration   time.Duration
	RootSpan   *StoredSpan
	Spans      []*StoredSpan
	Attributes map[string]interface{}
	Region     string
}

// StoredSpan represents a stored span
type StoredSpan struct {
	SpanID     string
	TraceID    string
	ParentID   string
	Name       string
	StartTime  time.Time
	EndTime    time.Time
	Duration   time.Duration
	Attributes map[string]interface{}
	Events     []SpanEvent
	Status     string
	Region     string
}

// SpanEvent represents a span event
type SpanEvent struct {
	Name       string
	Timestamp  time.Time
	Attributes map[string]interface{}
}

// TraceFilter filters traces for querying
type TraceFilter struct {
	TraceID    string
	ServiceName string
	Operation  string
	MinDuration time.Duration
	MaxDuration time.Duration
	StartTime  time.Time
	EndTime    time.Time
	HasError   bool
	Region     string
}

// NewTracingSystem creates a new distributed tracing system
func NewTracingSystem(config *TraceConfig) (*TracingSystem, error) {
	// Create Jaeger exporter
	exporter, err := jaeger.New(
		jaeger.WithCollectorEndpoint(
			jaeger.WithEndpoint(config.JaegerEndpoint),
		),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create jaeger exporter: %w", err)
	}

	// Create resource
	res, err := resource.New(
		context.Background(),
		resource.WithAttributes(
			semconv.ServiceNameKey.String(config.ServiceName),
		),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create resource: %w", err)
	}

	// Create sampler based on strategy
	var sampler sdktrace.Sampler
	switch config.Strategy {
	case SamplingHead:
		sampler = sdktrace.TraceIDRatioBased(config.SamplingRate)
	case SamplingTail:
		// Tail-based sampling (always record, decide later)
		sampler = sdktrace.AlwaysSample()
	case SamplingAdaptive:
		// Adaptive sampling (starts with head-based)
		sampler = sdktrace.ParentBased(sdktrace.TraceIDRatioBased(config.SamplingRate))
	default:
		sampler = sdktrace.TraceIDRatioBased(config.SamplingRate)
	}

	// Create tracer provider
	provider := sdktrace.NewTracerProvider(
		sdktrace.WithBatcher(exporter),
		sdktrace.WithResource(res),
		sdktrace.WithSampler(sampler),
	)

	// Set global tracer provider
	otel.SetTracerProvider(provider)

	ts := &TracingSystem{
		config:         config,
		tracerProvider: provider,
		tracer:         provider.Tracer(config.ServiceName),
		traces:         make(map[string]*StoredTrace),
		spans:          make(map[string]*StoredSpan),
		sampler:        sampler,
	}

	return ts, nil
}

// StartSpan starts a new span
func (ts *TracingSystem) StartSpan(ctx context.Context, name string, attrs ...attribute.KeyValue) (context.Context, trace.Span) {
	return ts.tracer.Start(ctx, name, trace.WithAttributes(attrs...))
}

// PropagateContext propagates trace context across regions
func (ts *TracingSystem) PropagateContext(ctx context.Context) map[string]string {
	spanCtx := trace.SpanContextFromContext(ctx)
	if !spanCtx.IsValid() {
		return nil
	}

	return map[string]string{
		"traceparent": fmt.Sprintf("00-%s-%s-%02x",
			spanCtx.TraceID().String(),
			spanCtx.SpanID().String(),
			spanCtx.TraceFlags()),
	}
}

// ExtractContext extracts trace context from propagation headers
func (ts *TracingSystem) ExtractContext(ctx context.Context, headers map[string]string) context.Context {
	// Parse traceparent header
	traceparent, ok := headers["traceparent"]
	if !ok {
		return ctx
	}

	// Simple parsing (production should use proper W3C trace context parser)
	var traceID trace.TraceID
	var spanID trace.SpanID
	var flags byte

	fmt.Sscanf(traceparent, "00-%s-%s-%02x", &traceID, &spanID, &flags)

	spanCtx := trace.NewSpanContext(trace.SpanContextConfig{
		TraceID:    traceID,
		SpanID:     spanID,
		TraceFlags: trace.TraceFlags(flags),
	})

	return trace.ContextWithSpanContext(ctx, spanCtx)
}

// RecordSpan stores a span for querying
func (ts *TracingSystem) RecordSpan(span *StoredSpan) {
	ts.mu.Lock()
	defer ts.mu.Unlock()

	ts.spans[span.SpanID] = span

	// Update or create trace
	trace, ok := ts.traces[span.TraceID]
	if !ok {
		trace = &StoredTrace{
			TraceID:    span.TraceID,
			StartTime:  span.StartTime,
			Attributes: make(map[string]interface{}),
			Region:     span.Region,
		}
		ts.traces[span.TraceID] = trace
	}

	trace.Spans = append(trace.Spans, span)
	trace.SpanCount++

	// Update trace timing
	if span.StartTime.Before(trace.StartTime) {
		trace.StartTime = span.StartTime
	}
	if span.EndTime.After(trace.EndTime) {
		trace.EndTime = span.EndTime
		trace.Duration = trace.EndTime.Sub(trace.StartTime)
	}

	// Set root span
	if span.ParentID == "" {
		trace.RootSpan = span
	}
}

// GetTraces retrieves traces matching filter
func (ts *TracingSystem) GetTraces(filter TraceFilter) []*StoredTrace {
	ts.mu.RLock()
	defer ts.mu.RUnlock()

	var result []*StoredTrace

	for _, trace := range ts.traces {
		if ts.matchesFilter(trace, filter) {
			result = append(result, trace)
		}
	}

	return result
}

// GetTrace retrieves a specific trace
func (ts *TracingSystem) GetTrace(traceID string) (*StoredTrace, bool) {
	ts.mu.RLock()
	defer ts.mu.RUnlock()

	trace, ok := ts.traces[traceID]
	return trace, ok
}

// StitchCrossRegionTraces combines spans from different regions into unified traces
func (ts *TracingSystem) StitchCrossRegionTraces() {
	ts.mu.Lock()
	defer ts.mu.Unlock()

	// Group spans by trace ID
	traceSpans := make(map[string][]*StoredSpan)
	for _, span := range ts.spans {
		traceSpans[span.TraceID] = append(traceSpans[span.TraceID], span)
	}

	// Reconstruct complete traces
	for traceID, spans := range traceSpans {
		trace := ts.traces[traceID]
		if trace == nil {
			continue
		}

		// Sort spans by start time
		// Build parent-child relationships
		trace.Spans = spans
		trace.SpanCount = len(spans)

		// Find earliest and latest times
		for _, span := range spans {
			if span.StartTime.Before(trace.StartTime) {
				trace.StartTime = span.StartTime
			}
			if span.EndTime.After(trace.EndTime) {
				trace.EndTime = span.EndTime
			}
		}

		trace.Duration = trace.EndTime.Sub(trace.StartTime)
	}
}

// ApplyTailBasedSampling applies tail-based sampling decisions
func (ts *TracingSystem) ApplyTailBasedSampling() {
	ts.mu.Lock()
	defer ts.mu.Unlock()

	for traceID, trace := range ts.traces {
		// Keep traces with errors
		hasError := false
		for _, span := range trace.Spans {
			if span.Status == "error" {
				hasError = true
				break
			}
		}

		// Keep high-latency traces
		isHighLatency := trace.Duration > 100*time.Millisecond

		// Drop trace if not interesting
		if !hasError && !isHighLatency {
			delete(ts.traces, traceID)
			ts.droppedTraces++
		} else {
			ts.sampledTraces++
		}
	}
}

// AdaptSamplingRate adjusts sampling rate based on load
func (ts *TracingSystem) AdaptSamplingRate(currentLoad float64) {
	// Simple adaptive algorithm
	targetSampleRate := 1.0 - (currentLoad * 0.9) // Reduce sampling at high load

	if targetSampleRate < 0.01 {
		targetSampleRate = 0.01 // Minimum 1%
	}

	ts.config.SamplingRate = targetSampleRate

	// Update sampler
	ts.sampler = sdktrace.TraceIDRatioBased(targetSampleRate)
}

// GetStatistics returns tracing statistics
func (ts *TracingSystem) GetStatistics() map[string]interface{} {
	ts.mu.RLock()
	defer ts.mu.RUnlock()

	return map[string]interface{}{
		"total_traces":   ts.totalTraces,
		"sampled_traces": ts.sampledTraces,
		"dropped_traces": ts.droppedTraces,
		"sampling_rate":  ts.config.SamplingRate,
		"active_traces":  len(ts.traces),
		"active_spans":   len(ts.spans),
	}
}

// Shutdown gracefully shuts down the tracing system
func (ts *TracingSystem) Shutdown(ctx context.Context) error {
	return ts.tracerProvider.Shutdown(ctx)
}

// matchesFilter checks if trace matches filter criteria
func (ts *TracingSystem) matchesFilter(trace *StoredTrace, filter TraceFilter) bool {
	if filter.TraceID != "" && trace.TraceID != filter.TraceID {
		return false
	}

	if filter.MinDuration > 0 && trace.Duration < filter.MinDuration {
		return false
	}

	if filter.MaxDuration > 0 && trace.Duration > filter.MaxDuration {
		return false
	}

	if !filter.StartTime.IsZero() && trace.StartTime.Before(filter.StartTime) {
		return false
	}

	if !filter.EndTime.IsZero() && trace.EndTime.After(filter.EndTime) {
		return false
	}

	if filter.Region != "" && trace.Region != filter.Region {
		return false
	}

	if filter.HasError {
		hasError := false
		for _, span := range trace.Spans {
			if span.Status == "error" {
				hasError = true
				break
			}
		}
		if !hasError {
			return false
		}
	}

	return true
}
