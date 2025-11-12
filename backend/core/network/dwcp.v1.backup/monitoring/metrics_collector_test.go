package monitoring

import (
	"context"
	"testing"
	"time"
)

func TestMetricsCollector(t *testing.T) {
	t.Run("NewMetricsCollector", func(t *testing.T) {
		mc, err := NewMetricsCollector("us-west-1")
		if err != nil {
			t.Fatalf("Failed to create metrics collector: %v", err)
		}
		if mc == nil {
			t.Fatal("Expected metrics collector, got nil")
		}
	})

	t.Run("RecordRequest", func(t *testing.T) {
		mc, _ := NewMetricsCollector("us-west-1")
		ctx := context.Background()

		labels := map[string]string{
			"operation": "test",
		}

		mc.RecordRequest(ctx, "test-op", labels)
		// Test passes if no panic
	})

	t.Run("RecordError", func(t *testing.T) {
		mc, _ := NewMetricsCollector("us-west-1")
		ctx := context.Background()

		labels := map[string]string{
			"operation": "test",
		}

		mc.RecordError(ctx, "test-op", "test-error", labels)
		// Test passes if no panic
	})

	t.Run("RecordLatency", func(t *testing.T) {
		mc, _ := NewMetricsCollector("us-west-1")
		ctx := context.Background()

		mc.RecordLatency(ctx, "test-op", 50.0, nil)

		metrics := mc.GetMetrics(time.Now().Add(-1 * time.Minute))
		if len(metrics) == 0 {
			t.Error("Expected metrics to be recorded")
		}
	})

	t.Run("AggregateMetrics", func(t *testing.T) {
		mc, _ := NewMetricsCollector("us-west-1")

		regionMetrics := map[string][]*MetricData{
			"latency": {
				{Value: 10.0},
				{Value: 20.0},
				{Value: 30.0},
			},
		}

		agg := mc.AggregateMetrics(regionMetrics)

		if agg["latency"] == nil {
			t.Error("Expected aggregated latency metric")
		}

		if agg["latency"].Mean != 20.0 {
			t.Errorf("Expected mean 20.0, got %f", agg["latency"].Mean)
		}
	})

	t.Run("Cleanup", func(t *testing.T) {
		mc, _ := NewMetricsCollector("us-west-1")
		mc.retentionPeriod = 1 * time.Millisecond

		mc.RecordGauge("test", 100.0, nil)
		time.Sleep(10 * time.Millisecond)

		mc.Cleanup()

		metrics := mc.GetMetrics(time.Time{})
		if len(metrics) > 0 {
			t.Error("Expected old metrics to be cleaned up")
		}
	})
}

func TestAggregatedMetric(t *testing.T) {
	t.Run("CalculatePercentiles", func(t *testing.T) {
		values := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

		p50 := calculatePercentile(values, 0.50)
		p95 := calculatePercentile(values, 0.95)
		p99 := calculatePercentile(values, 0.99)

		if p50 < 1 || p50 > 10 {
			t.Errorf("P50 out of range: %f", p50)
		}

		if p95 < p50 {
			t.Error("P95 should be >= P50")
		}

		if p99 < p95 {
			t.Error("P99 should be >= P95")
		}
	})
}
