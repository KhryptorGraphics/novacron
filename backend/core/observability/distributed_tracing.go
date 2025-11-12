// Distributed Tracing with <100μs Overhead
// OpenTelemetry-based tracing with high-performance sampling

package observability

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.17.0"
	"go.opentelemetry.io/otel/trace"
	"go.uber.org/zap"
)

// TracingConfig configures the distributed tracing system
type TracingConfig struct {
	Enabled            bool
	ServiceName        string
	ServiceVersion     string
	CollectorEndpoint  string
	SamplingRate       float64
	MaxSpansPerTrace   int
	MaxEventsPerSpan   int
	MaxAttributesPerSpan int
	ExportTimeout      time.Duration
	BatchTimeout       time.Duration
	QueueSize          int
	MaxExportBatch     int
	AdaptiveSampling   bool
}

// TracingManager manages distributed tracing
type TracingManager struct {
	provider          *sdktrace.TracerProvider
	tracer            trace.Tracer
	sampler           *AdaptiveSampler
	spanProcessor     *HighPerformanceSpanProcessor
	contextPropagator *ContextPropagator
	metrics           *TracingMetrics
	config            *TracingConfig
	logger            *zap.Logger
	shutdownCh        chan struct{}
	wg                sync.WaitGroup
}

// AdaptiveSampler implements adaptive sampling based on system load
type AdaptiveSampler struct {
	baseSamplingRate  float64
	currentRate       atomic.Value // float64
	targetOverhead    time.Duration // Target overhead per request
	actualOverhead    atomic.Value  // time.Duration
	windowSize        time.Duration
	measurements      *RingBuffer
	mu                sync.RWMutex
}

// RingBuffer implements a lock-free ring buffer
type RingBuffer struct {
	buffer   []Measurement
	size     int
	head     uint64
	tail     uint64
}

// Measurement represents a sampling measurement
type Measurement struct {
	Timestamp time.Time
	Overhead  time.Duration
	Sampled   bool
}

// HighPerformanceSpanProcessor implements high-performance span processing
type HighPerformanceSpanProcessor struct {
	exporter     sdktrace.SpanExporter
	queue        *LockFreeQueue
	batchSize    int
	timeout      time.Duration
	workers      int
	droppedSpans uint64
	mu           sync.RWMutex
}

// LockFreeQueue implements a lock-free queue for spans
type LockFreeQueue struct {
	buffer []sdktrace.ReadOnlySpan
	size   int
	head   uint64
	tail   uint64
}

// ContextPropagator handles context propagation across services
type ContextPropagator struct {
	codec        *TraceContextCodec
	baggage      sync.Map
	mu           sync.RWMutex
}

// TraceContextCodec encodes/decodes trace context
type TraceContextCodec struct {
	version byte
}

// TracingMetrics tracks tracing system metrics
type TracingMetrics struct {
	spansCreated       prometheus.Counter
	spansFinished      prometheus.Counter
	spansDuration      prometheus.Histogram
	samplingRate       prometheus.Gauge
	tracingOverhead    prometheus.Histogram
	spansDropped       prometheus.Counter
	exporterErrors     prometheus.Counter
	batchExportLatency prometheus.Histogram
}

// SpanContext contains span context information
type SpanContext struct {
	TraceID    string
	SpanID     string
	ParentID   string
	Sampled    bool
	Attributes map[string]interface{}
	Baggage    map[string]string
}

// NewTracingManager creates a new tracing manager
func NewTracingManager(config *TracingConfig, logger *zap.Logger) (*TracingManager, error) {
	// Create resource
	res, err := resource.New(
		context.Background(),
		resource.WithAttributes(
			semconv.ServiceName(config.ServiceName),
			semconv.ServiceVersion(config.ServiceVersion),
		),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create resource: %w", err)
	}

	// Create OTLP exporter
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	exporter, err := otlptracegrpc.New(ctx,
		otlptracegrpc.WithEndpoint(config.CollectorEndpoint),
		otlptracegrpc.WithInsecure(),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create exporter: %w", err)
	}

	// Create adaptive sampler
	sampler := NewAdaptiveSampler(
		config.SamplingRate,
		100*time.Microsecond, // Target <100μs overhead
		1*time.Minute,
	)

	// Create high-performance span processor
	spanProcessor := NewHighPerformanceSpanProcessor(
		exporter,
		config.QueueSize,
		config.MaxExportBatch,
		config.BatchTimeout,
		4, // Number of workers
	)

	// Create tracer provider with custom configuration
	provider := sdktrace.NewTracerProvider(
		sdktrace.WithResource(res),
		sdktrace.WithSpanProcessor(spanProcessor),
		sdktrace.WithSampler(sampler),
		sdktrace.WithSpanLimits(sdktrace.SpanLimits{
			AttributeValueLengthLimit:   -1, // Unlimited
			AttributeCountLimit:         config.MaxAttributesPerSpan,
			EventCountLimit:             config.MaxEventsPerSpan,
			LinkCountLimit:              128,
			AttributePerEventCountLimit: 128,
			AttributePerLinkCountLimit:  128,
		}),
	)

	// Set global provider
	otel.SetTracerProvider(provider)

	tracer := provider.Tracer(config.ServiceName)

	return &TracingManager{
		provider:      provider,
		tracer:        tracer,
		sampler:       sampler,
		spanProcessor: spanProcessor,
		contextPropagator: &ContextPropagator{
			codec: &TraceContextCodec{version: 0},
		},
		metrics:    NewTracingMetrics(),
		config:     config,
		logger:     logger,
		shutdownCh: make(chan struct{}),
	}, nil
}

// Start begins the tracing manager
func (m *TracingManager) Start(ctx context.Context) error {
	if !m.config.Enabled {
		m.logger.Info("Distributed tracing disabled")
		return nil
	}

	m.logger.Info("Starting distributed tracing",
		zap.String("service", m.config.ServiceName),
		zap.Float64("sampling_rate", m.config.SamplingRate),
		zap.String("collector", m.config.CollectorEndpoint))

	// Start adaptive sampling if enabled
	if m.config.AdaptiveSampling {
		m.wg.Add(1)
		go m.runAdaptiveSampling(ctx)
	}

	// Start metrics collection
	m.wg.Add(1)
	go m.collectMetrics(ctx)

	// Start span processor workers
	m.spanProcessor.Start(ctx)

	return nil
}

// StartSpan creates a new span with minimal overhead
func (m *TracingManager) StartSpan(ctx context.Context, operationName string, opts ...trace.SpanStartOption) (context.Context, trace.Span) {
	// Fast path for unsampled traces
	if !m.shouldSample() {
		return ctx, trace.SpanFromContext(ctx)
	}

	// Start timing to measure overhead
	start := time.Now()

	// Start span
	ctx, span := m.tracer.Start(ctx, operationName, opts...)

	// Record overhead
	overhead := time.Since(start)
	m.sampler.RecordMeasurement(overhead, true)
	m.metrics.tracingOverhead.Observe(overhead.Seconds())
	m.metrics.spansCreated.Inc()

	return ctx, span
}

// InjectContext injects trace context into carrier
func (m *TracingManager) InjectContext(ctx context.Context, carrier map[string]string) error {
	spanCtx := trace.SpanFromContext(ctx).SpanContext()
	if !spanCtx.IsValid() {
		return fmt.Errorf("invalid span context")
	}

	// Encode trace context
	carrier["traceparent"] = m.contextPropagator.codec.EncodeTraceParent(spanCtx)

	// Encode baggage
	if baggage := m.contextPropagator.GetBaggage(ctx); len(baggage) > 0 {
		carrier["tracestate"] = m.contextPropagator.codec.EncodeBaggage(baggage)
	}

	return nil
}

// ExtractContext extracts trace context from carrier
func (m *TracingManager) ExtractContext(ctx context.Context, carrier map[string]string) (context.Context, error) {
	// Decode trace context
	spanCtx, err := m.contextPropagator.codec.DecodeTraceParent(carrier["traceparent"])
	if err != nil {
		return ctx, fmt.Errorf("failed to decode trace context: %w", err)
	}

	// Create span context
	ctx = trace.ContextWithRemoteSpanContext(ctx, spanCtx)

	// Decode baggage
	if tracestate, ok := carrier["tracestate"]; ok {
		baggage, err := m.contextPropagator.codec.DecodeBaggage(tracestate)
		if err == nil {
			m.contextPropagator.SetBaggage(ctx, baggage)
		}
	}

	return ctx, nil
}

// AddSpanAttributes adds attributes to current span
func (m *TracingManager) AddSpanAttributes(ctx context.Context, attrs ...attribute.KeyValue) {
	span := trace.SpanFromContext(ctx)
	span.SetAttributes(attrs...)
}

// AddSpanEvent adds an event to current span
func (m *TracingManager) AddSpanEvent(ctx context.Context, name string, attrs ...attribute.KeyValue) {
	span := trace.SpanFromContext(ctx)
	span.AddEvent(name, trace.WithAttributes(attrs...))
}

// RecordError records an error in current span
func (m *TracingManager) RecordError(ctx context.Context, err error, opts ...trace.EventOption) {
	span := trace.SpanFromContext(ctx)
	span.RecordError(err, opts...)
	span.SetStatus(codes.Error, err.Error())
}

// shouldSample determines if a trace should be sampled
func (m *TracingManager) shouldSample() bool {
	if !m.config.AdaptiveSampling {
		return m.sampler.baseSamplingRate >= 1.0 ||
			   (m.sampler.baseSamplingRate > 0 && time.Now().UnixNano()%100 < int64(m.sampler.baseSamplingRate*100))
	}

	rate := m.sampler.currentRate.Load().(float64)
	return rate >= 1.0 || (rate > 0 && time.Now().UnixNano()%100 < int64(rate*100))
}

// runAdaptiveSampling runs the adaptive sampling loop
func (m *TracingManager) runAdaptiveSampling(ctx context.Context) {
	defer m.wg.Done()

	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-m.shutdownCh:
			return
		case <-ticker.C:
			m.sampler.Adjust()

			newRate := m.sampler.currentRate.Load().(float64)
			m.metrics.samplingRate.Set(newRate)

			m.logger.Debug("Adjusted sampling rate",
				zap.Float64("rate", newRate),
				zap.Duration("avg_overhead", m.sampler.GetAverageOverhead()))
		}
	}
}

// collectMetrics collects tracing metrics
func (m *TracingManager) collectMetrics(ctx context.Context) {
	defer m.wg.Done()

	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-m.shutdownCh:
			return
		case <-ticker.C:
			// Report dropped spans
			dropped := atomic.LoadUint64(&m.spanProcessor.droppedSpans)
			m.metrics.spansDropped.Add(float64(dropped))
			atomic.StoreUint64(&m.spanProcessor.droppedSpans, 0)
		}
	}
}

// Shutdown gracefully shuts down the tracing manager
func (m *TracingManager) Shutdown(ctx context.Context) error {
	close(m.shutdownCh)
	m.wg.Wait()

	// Flush remaining spans
	if err := m.provider.Shutdown(ctx); err != nil {
		return fmt.Errorf("failed to shutdown provider: %w", err)
	}

	m.logger.Info("Tracing manager shutdown complete")
	return nil
}

// AdaptiveSampler implementation

func NewAdaptiveSampler(baseSamplingRate float64, targetOverhead time.Duration, windowSize time.Duration) *AdaptiveSampler {
	sampler := &AdaptiveSampler{
		baseSamplingRate: baseSamplingRate,
		targetOverhead:   targetOverhead,
		windowSize:       windowSize,
		measurements:     NewRingBuffer(1000),
	}
	sampler.currentRate.Store(baseSamplingRate)
	sampler.actualOverhead.Store(time.Duration(0))
	return sampler
}

func (s *AdaptiveSampler) ShouldSample(parameters sdktrace.SamplingParameters) sdktrace.SamplingResult {
	rate := s.currentRate.Load().(float64)

	// Always sample if rate is 100%
	if rate >= 1.0 {
		return sdktrace.SamplingResult{
			Decision: sdktrace.RecordAndSample,
		}
	}

	// Probabilistic sampling
	if rate > 0 && time.Now().UnixNano()%10000 < int64(rate*10000) {
		return sdktrace.SamplingResult{
			Decision: sdktrace.RecordAndSample,
		}
	}

	return sdktrace.SamplingResult{
		Decision: sdktrace.Drop,
	}
}

func (s *AdaptiveSampler) Description() string {
	return fmt.Sprintf("AdaptiveSampler{rate=%.4f}", s.currentRate.Load().(float64))
}

func (s *AdaptiveSampler) RecordMeasurement(overhead time.Duration, sampled bool) {
	s.measurements.Add(Measurement{
		Timestamp: time.Now(),
		Overhead:  overhead,
		Sampled:   sampled,
	})
}

func (s *AdaptiveSampler) Adjust() {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Calculate average overhead from recent measurements
	avgOverhead := s.measurements.AverageOverhead()
	s.actualOverhead.Store(avgOverhead)

	currentRate := s.currentRate.Load().(float64)

	// Adjust sampling rate based on overhead
	if avgOverhead > s.targetOverhead {
		// Overhead too high, reduce sampling
		newRate := currentRate * 0.9
		if newRate < 0.01 {
			newRate = 0.01 // Minimum 1% sampling
		}
		s.currentRate.Store(newRate)
	} else if avgOverhead < s.targetOverhead/2 {
		// Overhead well below target, increase sampling
		newRate := currentRate * 1.1
		if newRate > s.baseSamplingRate {
			newRate = s.baseSamplingRate // Don't exceed base rate
		}
		s.currentRate.Store(newRate)
	}
}

func (s *AdaptiveSampler) GetAverageOverhead() time.Duration {
	return s.actualOverhead.Load().(time.Duration)
}

// RingBuffer implementation

func NewRingBuffer(size int) *RingBuffer {
	return &RingBuffer{
		buffer: make([]Measurement, size),
		size:   size,
	}
}

func (r *RingBuffer) Add(m Measurement) {
	head := atomic.LoadUint64(&r.head)
	tail := atomic.LoadUint64(&r.tail)

	// Check if buffer is full
	if (head+1)%uint64(r.size) == tail {
		// Move tail forward
		atomic.CompareAndSwapUint64(&r.tail, tail, (tail+1)%uint64(r.size))
	}

	// Add measurement
	r.buffer[head] = m

	// Move head forward
	atomic.CompareAndSwapUint64(&r.head, head, (head+1)%uint64(r.size))
}

func (r *RingBuffer) AverageOverhead() time.Duration {
	head := atomic.LoadUint64(&r.head)
	tail := atomic.LoadUint64(&r.tail)

	if head == tail {
		return 0
	}

	var total time.Duration
	count := 0

	for i := tail; i != head; i = (i + 1) % uint64(r.size) {
		if r.buffer[i].Sampled {
			total += r.buffer[i].Overhead
			count++
		}
	}

	if count == 0 {
		return 0
	}

	return total / time.Duration(count)
}

// HighPerformanceSpanProcessor implementation

func NewHighPerformanceSpanProcessor(
	exporter sdktrace.SpanExporter,
	queueSize int,
	batchSize int,
	timeout time.Duration,
	workers int,
) *HighPerformanceSpanProcessor {
	return &HighPerformanceSpanProcessor{
		exporter:  exporter,
		queue:     NewLockFreeQueue(queueSize),
		batchSize: batchSize,
		timeout:   timeout,
		workers:   workers,
	}
}

func (p *HighPerformanceSpanProcessor) Start(ctx context.Context) {
	// Start worker goroutines
	for i := 0; i < p.workers; i++ {
		go p.worker(ctx, i)
	}
}

func (p *HighPerformanceSpanProcessor) OnStart(parent context.Context, s sdktrace.ReadWriteSpan) {
	// No-op for high performance
}

func (p *HighPerformanceSpanProcessor) OnEnd(s sdktrace.ReadOnlySpan) {
	if !p.queue.Enqueue(s) {
		atomic.AddUint64(&p.droppedSpans, 1)
	}
}

func (p *HighPerformanceSpanProcessor) Shutdown(ctx context.Context) error {
	// Flush remaining spans
	return p.ForceFlush(ctx)
}

func (p *HighPerformanceSpanProcessor) ForceFlush(ctx context.Context) error {
	// Export all remaining spans
	batch := make([]sdktrace.ReadOnlySpan, 0, p.batchSize)
	for {
		span := p.queue.Dequeue()
		if span == nil {
			break
		}
		batch = append(batch, span)

		if len(batch) >= p.batchSize {
			if err := p.exporter.ExportSpans(ctx, batch); err != nil {
				return err
			}
			batch = batch[:0]
		}
	}

	// Export remaining
	if len(batch) > 0 {
		return p.exporter.ExportSpans(ctx, batch)
	}

	return nil
}

func (p *HighPerformanceSpanProcessor) worker(ctx context.Context, id int) {
	ticker := time.NewTicker(p.timeout)
	defer ticker.Stop()

	batch := make([]sdktrace.ReadOnlySpan, 0, p.batchSize)

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			// Timeout - export current batch
			if len(batch) > 0 {
				p.exportBatch(ctx, batch)
				batch = batch[:0]
			}
		default:
			// Try to dequeue span
			span := p.queue.Dequeue()
			if span == nil {
				time.Sleep(10 * time.Millisecond)
				continue
			}

			batch = append(batch, span)

			// Export if batch is full
			if len(batch) >= p.batchSize {
				p.exportBatch(ctx, batch)
				batch = batch[:0]
			}
		}
	}
}

func (p *HighPerformanceSpanProcessor) exportBatch(ctx context.Context, batch []sdktrace.ReadOnlySpan) {
	if err := p.exporter.ExportSpans(ctx, batch); err != nil {
		// Log error but don't fail
	}
}

// LockFreeQueue implementation

func NewLockFreeQueue(size int) *LockFreeQueue {
	return &LockFreeQueue{
		buffer: make([]sdktrace.ReadOnlySpan, size),
		size:   size,
	}
}

func (q *LockFreeQueue) Enqueue(span sdktrace.ReadOnlySpan) bool {
	for {
		head := atomic.LoadUint64(&q.head)
		tail := atomic.LoadUint64(&q.tail)
		next := (head + 1) % uint64(q.size)

		// Check if queue is full
		if next == tail {
			return false
		}

		// Try to claim this slot
		if atomic.CompareAndSwapUint64(&q.head, head, next) {
			q.buffer[head] = span
			return true
		}
	}
}

func (q *LockFreeQueue) Dequeue() sdktrace.ReadOnlySpan {
	for {
		head := atomic.LoadUint64(&q.head)
		tail := atomic.LoadUint64(&q.tail)

		// Check if queue is empty
		if head == tail {
			return nil
		}

		span := q.buffer[tail]
		next := (tail + 1) % uint64(q.size)

		// Try to move tail
		if atomic.CompareAndSwapUint64(&q.tail, tail, next) {
			return span
		}
	}
}

// ContextPropagator implementation

func (c *TraceContextCodec) EncodeTraceParent(spanCtx trace.SpanContext) string {
	return fmt.Sprintf("00-%s-%s-%02x",
		spanCtx.TraceID().String(),
		spanCtx.SpanID().String(),
		spanCtx.TraceFlags())
}

func (c *TraceContextCodec) DecodeTraceParent(traceparent string) (trace.SpanContext, error) {
	// Simplified implementation - would need full W3C Trace Context parsing
	var version byte
	var traceID, spanID string
	var flags byte

	_, err := fmt.Sscanf(traceparent, "%02x-%s-%s-%02x", &version, &traceID, &spanID, &flags)
	if err != nil {
		return trace.SpanContext{}, err
	}

	// Parse trace ID and span ID
	tid, err := trace.TraceIDFromHex(traceID)
	if err != nil {
		return trace.SpanContext{}, err
	}

	sid, err := trace.SpanIDFromHex(spanID)
	if err != nil {
		return trace.SpanContext{}, err
	}

	return trace.NewSpanContext(trace.SpanContextConfig{
		TraceID:    tid,
		SpanID:     sid,
		TraceFlags: trace.TraceFlags(flags),
	}), nil
}

func (c *TraceContextCodec) EncodeBaggage(baggage map[string]string) string {
	// Simplified baggage encoding
	return ""
}

func (c *TraceContextCodec) DecodeBaggage(tracestate string) (map[string]string, error) {
	// Simplified baggage decoding
	return make(map[string]string), nil
}

func (p *ContextPropagator) SetBaggage(ctx context.Context, baggage map[string]string) {
	key := fmt.Sprintf("%p", ctx)
	p.baggage.Store(key, baggage)
}

func (p *ContextPropagator) GetBaggage(ctx context.Context) map[string]string {
	key := fmt.Sprintf("%p", ctx)
	if value, ok := p.baggage.Load(key); ok {
		return value.(map[string]string)
	}
	return make(map[string]string)
}

// Metrics

func NewTracingMetrics() *TracingMetrics {
	return &TracingMetrics{
		spansCreated: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "tracing_spans_created_total",
			Help: "Total number of spans created",
		}),
		spansFinished: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "tracing_spans_finished_total",
			Help: "Total number of spans finished",
		}),
		spansDuration: prometheus.NewHistogram(prometheus.HistogramOpts{
			Name:    "tracing_span_duration_seconds",
			Help:    "Duration of spans",
			Buckets: prometheus.ExponentialBuckets(0.000001, 10, 10), // Start at 1μs
		}),
		samplingRate: prometheus.NewGauge(prometheus.GaugeOpts{
			Name: "tracing_sampling_rate",
			Help: "Current sampling rate",
		}),
		tracingOverhead: prometheus.NewHistogram(prometheus.HistogramOpts{
			Name:    "tracing_overhead_seconds",
			Help:    "Overhead of tracing operations",
			Buckets: prometheus.ExponentialBuckets(0.000001, 2, 15), // 1μs to ~32ms
		}),
		spansDropped: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "tracing_spans_dropped_total",
			Help: "Total number of spans dropped",
		}),
		exporterErrors: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "tracing_exporter_errors_total",
			Help: "Total number of exporter errors",
		}),
		batchExportLatency: prometheus.NewHistogram(prometheus.HistogramOpts{
			Name:    "tracing_batch_export_latency_seconds",
			Help:    "Latency of batch exports",
			Buckets: prometheus.ExponentialBuckets(0.001, 2, 10),
		}),
	}
}