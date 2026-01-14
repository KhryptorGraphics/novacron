package tracing

import (
	"context"
	"fmt"
	"log"
	"sync"
	"time"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/exporters/jaeger"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp"
	"go.opentelemetry.io/otel/exporters/stdout/stdouttrace"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.17.0"
	"go.opentelemetry.io/otel/trace"
)

// TracingIntegration manages OpenTelemetry distributed tracing
type TracingIntegration struct {
	// OpenTelemetry components
	tracer   trace.Tracer
	provider *sdktrace.TracerProvider
	
	// Configuration
	config *TracingConfig
	
	// Active spans tracking
	activeSpans map[string]trace.Span
	
	// Metrics integration
	spanMetrics *SpanMetrics
	
	// Sampling strategies
	sampler sdktrace.Sampler
	
	// Concurrency control
	mutex sync.RWMutex
}

// TracingConfig represents the configuration for tracing
type TracingConfig struct {
	// Service identification
	ServiceName    string `json:"service_name"`
	ServiceVersion string `json:"service_version"`
	Environment    string `json:"environment"`
	
	// Exporter configuration
	ExporterType   string `json:"exporter_type"` // "jaeger", "otlp", "stdout", "multi"
	JaegerEndpoint string `json:"jaeger_endpoint"`
	OTLPEndpoint   string `json:"otlp_endpoint"`
	
	// Sampling configuration
	SamplingRate     float64 `json:"sampling_rate"`
	SamplingStrategy string  `json:"sampling_strategy"` // "always", "never", "ratio", "adaptive"
	
	// Resource attributes
	ResourceAttributes map[string]string `json:"resource_attributes"`
	
	// Batch processing
	BatchTimeout    time.Duration `json:"batch_timeout"`
	BatchSize       int           `json:"batch_size"`
	ExportTimeout   time.Duration `json:"export_timeout"`
	
	// Advanced features
	EnableMetrics        bool `json:"enable_metrics"`
	EnableLogCorrelation bool `json:"enable_log_correlation"`
	MaxSpansPerTrace     int  `json:"max_spans_per_trace"`
	MaxAttributeLength   int  `json:"max_attribute_length"`
	
	// Custom instrumentation
	AutoInstrument bool     `json:"auto_instrument"`
	InstrumentedOperations []string `json:"instrumented_operations"`
}

// DefaultTracingConfig returns a default tracing configuration
func DefaultTracingConfig() *TracingConfig {
	return &TracingConfig{
		ServiceName:            "novacron",
		ServiceVersion:         "1.0.0",
		Environment:            "production",
		ExporterType:           "jaeger",
		JaegerEndpoint:         "http://localhost:14268/api/traces",
		OTLPEndpoint:           "http://localhost:4318/v1/traces",
		SamplingRate:           0.1,
		SamplingStrategy:       "ratio",
		BatchTimeout:           5 * time.Second,
		BatchSize:              512,
		ExportTimeout:          30 * time.Second,
		EnableMetrics:          true,
		EnableLogCorrelation:   true,
		MaxSpansPerTrace:       1000,
		MaxAttributeLength:     1000,
		AutoInstrument:         true,
		InstrumentedOperations: []string{
			"vm.create", "vm.migrate", "vm.delete",
			"storage.read", "storage.write", "storage.tier",
			"network.route", "network.balance",
			"auth.login", "auth.authorize",
			"backup.create", "backup.restore",
		},
		ResourceAttributes: map[string]string{
			"service.instance.id": "novacron-1",
			"deployment.environment": "production",
		},
	}
}

// SpanMetrics tracks metrics for spans
type SpanMetrics struct {
	SpanCount       int64         `json:"span_count"`
	ErrorCount      int64         `json:"error_count"`
	AverageDuration time.Duration `json:"average_duration"`
	LastUpdated     time.Time     `json:"last_updated"`
}

// SpanContext represents additional context for spans
type SpanContext struct {
	TraceID    string            `json:"trace_id"`
	SpanID     string            `json:"span_id"`
	Operation  string            `json:"operation"`
	Component  string            `json:"component"`
	Tags       map[string]string `json:"tags"`
	StartTime  time.Time         `json:"start_time"`
	Duration   time.Duration     `json:"duration"`
	Error      error             `json:"error,omitempty"`
}

// NovaCronSpan represents a custom span with additional NovaCron-specific data
type NovaCronSpan struct {
	trace.Span
	Context   *SpanContext
	Component string
	Operation string
	Metadata  map[string]interface{}
}

// NewTracingIntegration creates a new tracing integration
func NewTracingIntegration(config *TracingConfig) (*TracingIntegration, error) {
	if config == nil {
		config = DefaultTracingConfig()
	}

	integration := &TracingIntegration{
		config:      config,
		activeSpans: make(map[string]trace.Span),
		spanMetrics: &SpanMetrics{
			LastUpdated: time.Now(),
		},
	}

	// Initialize OpenTelemetry
	if err := integration.initializeOTel(); err != nil {
		return nil, fmt.Errorf("failed to initialize OpenTelemetry: %w", err)
	}

	return integration, nil
}

// Start starts the tracing integration
func (t *TracingIntegration) Start() error {
	log.Printf("Starting OpenTelemetry tracing integration...")
	log.Printf("Service: %s, Version: %s, Environment: %s", 
		t.config.ServiceName, t.config.ServiceVersion, t.config.Environment)
	log.Printf("Exporter: %s, Sampling Rate: %.2f", 
		t.config.ExporterType, t.config.SamplingRate)
	
	return nil
}

// Stop stops the tracing integration
func (t *TracingIntegration) Stop() error {
	log.Println("Stopping OpenTelemetry tracing integration...")
	
	if t.provider != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		
		if err := t.provider.Shutdown(ctx); err != nil {
			return fmt.Errorf("failed to shutdown tracer provider: %w", err)
		}
	}
	
	log.Println("OpenTelemetry tracing integration stopped")
	return nil
}

// StartSpan starts a new span for a NovaCron operation
func (t *TracingIntegration) StartSpan(ctx context.Context, component, operation string, opts ...trace.SpanStartOption) (context.Context, *NovaCronSpan) {
	spanName := fmt.Sprintf("%s.%s", component, operation)
	
	// Add default attributes
	opts = append(opts, trace.WithAttributes(
		semconv.ServiceNameKey.String(t.config.ServiceName),
		semconv.ServiceVersionKey.String(t.config.ServiceVersion),
		attribute.String("component", component),
		attribute.String("operation", operation),
		attribute.String("environment", t.config.Environment),
	))
	
	ctx, span := t.tracer.Start(ctx, spanName, opts...)
	
	spanCtx := &SpanContext{
		TraceID:   span.SpanContext().TraceID().String(),
		SpanID:    span.SpanContext().SpanID().String(),
		Operation: operation,
		Component: component,
		Tags:      make(map[string]string),
		StartTime: time.Now(),
	}
	
	novaCronSpan := &NovaCronSpan{
		Span:      span,
		Context:   spanCtx,
		Component: component,
		Operation: operation,
		Metadata:  make(map[string]interface{}),
	}
	
	// Track active spans
	t.mutex.Lock()
	t.activeSpans[spanCtx.SpanID] = span
	t.mutex.Unlock()
	
	return ctx, novaCronSpan
}

// FinishSpan finishes a span and updates metrics
func (t *TracingIntegration) FinishSpan(span *NovaCronSpan, err error) {
	if span == nil || span.Span == nil {
		return
	}
	
	// Set error status if there was an error
	if err != nil {
		span.Span.SetStatus(codes.Error, err.Error())
		span.Span.RecordError(err)
		span.Context.Error = err
	} else {
		span.Span.SetStatus(codes.Ok, "")
	}
	
	// Calculate duration
	span.Context.Duration = time.Since(span.Context.StartTime)
	
	// Update metrics
	t.updateSpanMetrics(span, err != nil)
	
	// Remove from active spans
	t.mutex.Lock()
	delete(t.activeSpans, span.Context.SpanID)
	t.mutex.Unlock()
	
	// Finish the span
	span.Span.End()
}

// AddSpanAttribute adds an attribute to a span
func (t *TracingIntegration) AddSpanAttribute(span *NovaCronSpan, key string, value interface{}) {
	if span == nil || span.Span == nil {
		return
	}
	
	// Convert value to appropriate attribute type
	var attr attribute.KeyValue
	switch v := value.(type) {
	case string:
		if len(v) > t.config.MaxAttributeLength {
			v = v[:t.config.MaxAttributeLength]
		}
		attr = attribute.String(key, v)
	case int:
		attr = attribute.Int(key, v)
	case int64:
		attr = attribute.Int64(key, v)
	case float64:
		attr = attribute.Float64(key, v)
	case bool:
		attr = attribute.Bool(key, v)
	default:
		attr = attribute.String(key, fmt.Sprintf("%v", v))
	}
	
	span.Span.SetAttributes(attr)
	span.Context.Tags[key] = fmt.Sprintf("%v", value)
}

// AddSpanEvent adds an event to a span
func (t *TracingIntegration) AddSpanEvent(span *NovaCronSpan, name string, attributes ...attribute.KeyValue) {
	if span == nil || span.Span == nil {
		return
	}
	
	span.Span.AddEvent(name, trace.WithAttributes(attributes...))
}

// TraceVMMigration creates a trace for VM migration
func (t *TracingIntegration) TraceVMMigration(ctx context.Context, vmID, sourceNode, targetNode string) (context.Context, *NovaCronSpan) {
	ctx, span := t.StartSpan(ctx, "vm", "migrate")
	
	t.AddSpanAttribute(span, "vm.id", vmID)
	t.AddSpanAttribute(span, "vm.source_node", sourceNode)
	t.AddSpanAttribute(span, "vm.target_node", targetNode)
	t.AddSpanAttribute(span, "migration.type", "live")
	
	return ctx, span
}

// TraceStorageOperation creates a trace for storage operations
func (t *TracingIntegration) TraceStorageOperation(ctx context.Context, operation, tier string, size int64) (context.Context, *NovaCronSpan) {
	ctx, span := t.StartSpan(ctx, "storage", operation)
	
	t.AddSpanAttribute(span, "storage.tier", tier)
	t.AddSpanAttribute(span, "storage.size", size)
	t.AddSpanAttribute(span, "storage.operation", operation)
	
	return ctx, span
}

// TraceNetworkRequest creates a trace for network requests
func (t *TracingIntegration) TraceNetworkRequest(ctx context.Context, method, url string) (context.Context, *NovaCronSpan) {
	ctx, span := t.StartSpan(ctx, "network", "request")
	
	t.AddSpanAttribute(span, "http.method", method)
	t.AddSpanAttribute(span, "http.url", url)
	
	return ctx, span
}

// TraceAuthentication creates a trace for authentication operations
func (t *TracingIntegration) TraceAuthentication(ctx context.Context, userID, method string) (context.Context, *NovaCronSpan) {
	ctx, span := t.StartSpan(ctx, "auth", "authenticate")
	
	t.AddSpanAttribute(span, "user.id", userID)
	t.AddSpanAttribute(span, "auth.method", method)
	
	return ctx, span
}

// TraceBackupOperation creates a trace for backup operations
func (t *TracingIntegration) TraceBackupOperation(ctx context.Context, operation, vmID string, size int64) (context.Context, *NovaCronSpan) {
	ctx, span := t.StartSpan(ctx, "backup", operation)
	
	t.AddSpanAttribute(span, "backup.vm_id", vmID)
	t.AddSpanAttribute(span, "backup.size", size)
	t.AddSpanAttribute(span, "backup.operation", operation)
	
	return ctx, span
}

// GetActiveSpans returns the currently active spans
func (t *TracingIntegration) GetActiveSpans() map[string]trace.Span {
	t.mutex.RLock()
	defer t.mutex.RUnlock()
	
	result := make(map[string]trace.Span)
	for id, span := range t.activeSpans {
		result[id] = span
	}
	return result
}

// GetSpanMetrics returns span metrics
func (t *TracingIntegration) GetSpanMetrics() *SpanMetrics {
	t.mutex.RLock()
	defer t.mutex.RUnlock()
	
	return &SpanMetrics{
		SpanCount:       t.spanMetrics.SpanCount,
		ErrorCount:      t.spanMetrics.ErrorCount,
		AverageDuration: t.spanMetrics.AverageDuration,
		LastUpdated:     t.spanMetrics.LastUpdated,
	}
}

// InjectTraceContext injects trace context into headers
func (t *TracingIntegration) InjectTraceContext(ctx context.Context, headers map[string]string) {
	propagator := otel.GetTextMapPropagator()
	propagator.Inject(ctx, &mapCarrier{data: headers})
}

// ExtractTraceContext extracts trace context from headers
func (t *TracingIntegration) ExtractTraceContext(ctx context.Context, headers map[string]string) context.Context {
	propagator := otel.GetTextMapPropagator()
	return propagator.Extract(ctx, &mapCarrier{data: headers})
}

// Helper methods

func (t *TracingIntegration) initializeOTel() error {
	// Create resource
	res, err := t.createResource()
	if err != nil {
		return fmt.Errorf("failed to create resource: %w", err)
	}
	
	// Create exporter
	exporter, err := t.createExporter()
	if err != nil {
		return fmt.Errorf("failed to create exporter: %w", err)
	}
	
	// Create sampler
	sampler := t.createSampler()
	
	// Create tracer provider
	t.provider = sdktrace.NewTracerProvider(
		sdktrace.WithBatcher(exporter,
			sdktrace.WithBatchTimeout(t.config.BatchTimeout),
			sdktrace.WithMaxExportBatchSize(t.config.BatchSize),
			sdktrace.WithExportTimeout(t.config.ExportTimeout),
		),
		sdktrace.WithResource(res),
		sdktrace.WithSampler(sampler),
	)
	
	// Set global tracer provider
	otel.SetTracerProvider(t.provider)
	
	// Set global propagator
	otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator(
		propagation.TraceContext{},
		propagation.Baggage{},
	))
	
	// Create tracer
	t.tracer = t.provider.Tracer(t.config.ServiceName)
	
	return nil
}

func (t *TracingIntegration) createResource() (*resource.Resource, error) {
	attributes := []attribute.KeyValue{
		semconv.ServiceNameKey.String(t.config.ServiceName),
		semconv.ServiceVersionKey.String(t.config.ServiceVersion),
		semconv.DeploymentEnvironmentKey.String(t.config.Environment),
	}
	
	// Add custom resource attributes
	for key, value := range t.config.ResourceAttributes {
		attributes = append(attributes, attribute.String(key, value))
	}
	
	return resource.Merge(
		resource.Default(),
		resource.NewWithAttributes(
			semconv.SchemaURL,
			attributes...,
		),
	)
}

func (t *TracingIntegration) createExporter() (sdktrace.SpanExporter, error) {
	switch t.config.ExporterType {
	case "jaeger":
		return jaeger.New(jaeger.WithCollectorEndpoint(jaeger.WithEndpoint(t.config.JaegerEndpoint)))
	case "otlp":
		return otlptracehttp.New(context.Background(),
			otlptracehttp.WithEndpoint(t.config.OTLPEndpoint),
			otlptracehttp.WithInsecure(),
		)
	case "stdout":
		return stdouttrace.New(stdouttrace.WithPrettyPrint())
	default:
		return stdouttrace.New(stdouttrace.WithPrettyPrint())
	}
}

func (t *TracingIntegration) createSampler() sdktrace.Sampler {
	switch t.config.SamplingStrategy {
	case "always":
		return sdktrace.AlwaysSample()
	case "never":
		return sdktrace.NeverSample()
	case "ratio":
		return sdktrace.TraceIDRatioBased(t.config.SamplingRate)
	case "adaptive":
		// Implement adaptive sampling logic here
		// For now, use ratio-based sampling
		return sdktrace.TraceIDRatioBased(t.config.SamplingRate)
	default:
		return sdktrace.TraceIDRatioBased(t.config.SamplingRate)
	}
}

func (t *TracingIntegration) updateSpanMetrics(span *NovaCronSpan, hasError bool) {
	t.mutex.Lock()
	defer t.mutex.Unlock()
	
	t.spanMetrics.SpanCount++
	if hasError {
		t.spanMetrics.ErrorCount++
	}
	
	// Update average duration
	if t.spanMetrics.SpanCount == 1 {
		t.spanMetrics.AverageDuration = span.Context.Duration
	} else {
		// Calculate running average
		totalDuration := t.spanMetrics.AverageDuration * time.Duration(t.spanMetrics.SpanCount-1)
		totalDuration += span.Context.Duration
		t.spanMetrics.AverageDuration = totalDuration / time.Duration(t.spanMetrics.SpanCount)
	}
	
	t.spanMetrics.LastUpdated = time.Now()
}

// mapCarrier implements the TextMapCarrier interface for propagation
type mapCarrier struct {
	data map[string]string
}

func (c *mapCarrier) Get(key string) string {
	return c.data[key]
}

func (c *mapCarrier) Set(key string, value string) {
	c.data[key] = value
}

func (c *mapCarrier) Keys() []string {
	keys := make([]string, 0, len(c.data))
	for k := range c.data {
		keys = append(keys, k)
	}
	return keys
}

// CorrelatedLogger provides logging with trace correlation
type CorrelatedLogger struct {
	tracer trace.Tracer
}

// NewCorrelatedLogger creates a new correlated logger
func NewCorrelatedLogger(tracer trace.Tracer) *CorrelatedLogger {
	return &CorrelatedLogger{tracer: tracer}
}

// LogWithTrace logs a message with trace context
func (l *CorrelatedLogger) LogWithTrace(ctx context.Context, level, message string, fields ...attribute.KeyValue) {
	span := trace.SpanFromContext(ctx)
	if span.SpanContext().IsValid() {
		// Add trace information to log fields
		traceFields := []attribute.KeyValue{
			attribute.String("trace_id", span.SpanContext().TraceID().String()),
			attribute.String("span_id", span.SpanContext().SpanID().String()),
		}
		fields = append(traceFields, fields...)
	}
	
	// Log the message (this would integrate with your actual logging framework)
	log.Printf("[%s] %s - %v", level, message, fields)
}

// Middleware for HTTP request tracing
type TracingMiddleware struct {
	integration *TracingIntegration
}

// NewTracingMiddleware creates a new tracing middleware
func NewTracingMiddleware(integration *TracingIntegration) *TracingMiddleware {
	return &TracingMiddleware{integration: integration}
}

// WrapHandler wraps an HTTP handler with tracing
func (m *TracingMiddleware) WrapHandler(operationName string, handler func(ctx context.Context) error) func(ctx context.Context) error {
	return func(ctx context.Context) error {
		ctx, span := m.integration.StartSpan(ctx, "http", operationName)
		defer func() {
			m.integration.FinishSpan(span, nil)
		}()
		
		return handler(ctx)
	}
}

// Custom span types for different NovaCron operations
type VMOperationSpan struct {
	*NovaCronSpan
	VMID       string
	NodeID     string
	Operation  string
}

type StorageOperationSpan struct {
	*NovaCronSpan
	Tier      string
	Size      int64
	Operation string
}

type NetworkOperationSpan struct {
	*NovaCronSpan
	Protocol  string
	Bandwidth int64
	Latency   time.Duration
}

// Factory methods for specialized spans
func (t *TracingIntegration) NewVMOperationSpan(ctx context.Context, vmID, nodeID, operation string) (context.Context, *VMOperationSpan) {
	ctx, baseSpan := t.StartSpan(ctx, "vm", operation)
	
	t.AddSpanAttribute(baseSpan, "vm.id", vmID)
	t.AddSpanAttribute(baseSpan, "vm.node_id", nodeID)
	
	return ctx, &VMOperationSpan{
		NovaCronSpan: baseSpan,
		VMID:         vmID,
		NodeID:       nodeID,
		Operation:    operation,
	}
}

func (t *TracingIntegration) NewStorageOperationSpan(ctx context.Context, tier string, size int64, operation string) (context.Context, *StorageOperationSpan) {
	ctx, baseSpan := t.StartSpan(ctx, "storage", operation)
	
	t.AddSpanAttribute(baseSpan, "storage.tier", tier)
	t.AddSpanAttribute(baseSpan, "storage.size", size)
	
	return ctx, &StorageOperationSpan{
		NovaCronSpan: baseSpan,
		Tier:         tier,
		Size:         size,
		Operation:    operation,
	}
}

func (t *TracingIntegration) NewNetworkOperationSpan(ctx context.Context, protocol string, bandwidth int64, operation string) (context.Context, *NetworkOperationSpan) {
	ctx, baseSpan := t.StartSpan(ctx, "network", operation)
	
	t.AddSpanAttribute(baseSpan, "network.protocol", protocol)
	t.AddSpanAttribute(baseSpan, "network.bandwidth", bandwidth)
	
	return ctx, &NetworkOperationSpan{
		NovaCronSpan: baseSpan,
		Protocol:     protocol,
		Bandwidth:    bandwidth,
	}
}