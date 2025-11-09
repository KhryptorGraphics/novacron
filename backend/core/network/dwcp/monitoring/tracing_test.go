package monitoring

import (
	"context"
	"testing"
	"time"
)

func TestTracingSystem(t *testing.T) {
	t.Run("NewTracingSystem", func(t *testing.T) {
		config := &TraceConfig{
			ServiceName:    "test-service",
			JaegerEndpoint: "http://localhost:14268/api/traces",
			SamplingRate:   0.1,
			Strategy:       SamplingHead,
		}

		ts, err := NewTracingSystem(config)
		if err != nil {
			t.Fatalf("Failed to create tracing system: %v", err)
		}
		if ts == nil {
			t.Fatal("Expected tracing system, got nil")
		}
	})

	t.Run("StartSpan", func(t *testing.T) {
		config := &TraceConfig{
			ServiceName:    "test-service",
			JaegerEndpoint: "http://localhost:14268/api/traces",
			SamplingRate:   1.0,
			Strategy:       SamplingHead,
		}

		ts, _ := NewTracingSystem(config)
		ctx := context.Background()

		ctx, span := ts.StartSpan(ctx, "test-span")
		if span == nil {
			t.Fatal("Expected span, got nil")
		}

		span.End()
	})

	t.Run("PropagateContext", func(t *testing.T) {
		config := &TraceConfig{
			ServiceName:    "test-service",
			JaegerEndpoint: "http://localhost:14268/api/traces",
			SamplingRate:   1.0,
			Strategy:       SamplingHead,
		}

		ts, _ := NewTracingSystem(config)
		ctx := context.Background()

		ctx, span := ts.StartSpan(ctx, "test-span")
		defer span.End()

		headers := ts.PropagateContext(ctx)
		if headers == nil {
			t.Error("Expected trace context headers")
		}
	})

	t.Run("RecordSpan", func(t *testing.T) {
		config := &TraceConfig{
			ServiceName:    "test-service",
			JaegerEndpoint: "http://localhost:14268/api/traces",
			SamplingRate:   1.0,
			Strategy:       SamplingHead,
		}

		ts, _ := NewTracingSystem(config)

		span := &StoredSpan{
			SpanID:    "span-123",
			TraceID:   "trace-456",
			Name:      "test-span",
			StartTime: time.Now(),
			EndTime:   time.Now().Add(100 * time.Millisecond),
			Region:    "us-west-1",
		}

		ts.RecordSpan(span)

		trace, ok := ts.GetTrace("trace-456")
		if !ok {
			t.Error("Expected trace to be stored")
		}

		if trace.SpanCount != 1 {
			t.Errorf("Expected 1 span, got %d", trace.SpanCount)
		}
	})

	t.Run("StitchCrossRegionTraces", func(t *testing.T) {
		config := &TraceConfig{
			ServiceName:    "test-service",
			JaegerEndpoint: "http://localhost:14268/api/traces",
			SamplingRate:   1.0,
			Strategy:       SamplingHead,
		}

		ts, _ := NewTracingSystem(config)

		// Record spans from different regions
		span1 := &StoredSpan{
			SpanID:    "span-1",
			TraceID:   "trace-1",
			Name:      "span-us-west",
			StartTime: time.Now(),
			EndTime:   time.Now().Add(50 * time.Millisecond),
			Region:    "us-west-1",
		}

		span2 := &StoredSpan{
			SpanID:    "span-2",
			TraceID:   "trace-1",
			Name:      "span-us-east",
			StartTime: time.Now().Add(60 * time.Millisecond),
			EndTime:   time.Now().Add(120 * time.Millisecond),
			Region:    "us-east-1",
		}

		ts.RecordSpan(span1)
		ts.RecordSpan(span2)

		ts.StitchCrossRegionTraces()

		trace, ok := ts.GetTrace("trace-1")
		if !ok {
			t.Error("Expected stitched trace")
		}

		if trace.SpanCount != 2 {
			t.Errorf("Expected 2 spans, got %d", trace.SpanCount)
		}
	})

	t.Run("AdaptSamplingRate", func(t *testing.T) {
		config := &TraceConfig{
			ServiceName:    "test-service",
			JaegerEndpoint: "http://localhost:14268/api/traces",
			SamplingRate:   0.5,
			Strategy:       SamplingAdaptive,
		}

		ts, _ := NewTracingSystem(config)

		// High load should reduce sampling
		ts.AdaptSamplingRate(0.9)

		if ts.config.SamplingRate >= 0.5 {
			t.Error("Expected sampling rate to decrease under high load")
		}

		// Low load should increase sampling
		ts.AdaptSamplingRate(0.1)

		if ts.config.SamplingRate <= 0.1 {
			t.Error("Expected sampling rate to increase under low load")
		}
	})
}
