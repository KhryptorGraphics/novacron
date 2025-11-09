package inference

import (
	"context"
	"testing"
	"time"
)

func TestInferenceEngine(t *testing.T) {
	config := &InferenceConfig{
		MaxBatchSize:  32,
		CacheEnabled:  true,
		CacheTTL:      5 * time.Minute,
		NumWorkers:    2,
		LatencyTarget: 10 * time.Millisecond,
	}

	engine := NewInferenceEngine(config)

	// Load model
	weights := [][]float64{
		{0.5, 0.3, 0.2},
		{0.1, 0.4, 0.5},
	}

	err := engine.LoadModel("test_model", "v1", weights, "custom")
	if err != nil {
		t.Fatalf("failed to load model: %v", err)
	}

	// Single prediction
	ctx := context.Background()
	input := []float64{0.1, 0.2, 0.3}

	response, err := engine.Predict(ctx, "test_model", "v1", input)
	if err != nil {
		t.Fatalf("prediction failed: %v", err)
	}

	if response.Latency > config.LatencyTarget {
		t.Errorf("latency %v exceeds target %v", response.Latency, config.LatencyTarget)
	}

	t.Logf("Prediction: %v, Latency: %v", response.Prediction, response.Latency)

	// Test caching
	response2, _ := engine.Predict(ctx, "test_model", "v1", input)
	if !response2.FromCache {
		t.Error("expected cached response")
	}

	// Check metrics
	metrics := engine.GetMetrics()
	if metrics.TotalRequests < 2 {
		t.Error("metrics not tracking requests")
	}

	if metrics.CacheHits < 1 {
		t.Error("cache hits not recorded")
	}

	t.Logf("Metrics: Requests=%d, CacheHits=%d, AvgLatency=%.2fms",
		metrics.TotalRequests, metrics.CacheHits, metrics.AvgLatency)
}

func BenchmarkInference(b *testing.B) {
	engine := NewInferenceEngine(nil)
	weights := [][]float64{{0.5, 0.3}, {0.1, 0.4}}
	engine.LoadModel("bench", "v1", weights, "custom")

	input := []float64{0.1, 0.2}
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		engine.Predict(ctx, "bench", "v1", input)
	}
}
