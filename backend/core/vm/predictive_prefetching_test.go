package vm

import (
	"context"
	"testing"
	"time"

	"github.com/sirupsen/logrus"
)

func TestPredictivePrefetchingEngine_Creation(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.ErrorLevel) // Reduce noise in tests

	engine, err := NewPredictivePrefetchingEngine(logger)
	if err != nil {
		t.Fatalf("Failed to create predictive prefetching engine: %v", err)
	}

	if engine == nil {
		t.Fatal("Expected non-nil engine")
	}

	if engine.logger == nil {
		t.Fatal("Expected logger to be set")
	}

	if engine.aiModel == nil {
		t.Fatal("Expected AI model to be initialized")
	}

	if engine.cacheManager == nil {
		t.Fatal("Expected cache manager to be initialized")
	}
}

func TestPredictivePrefetchingEngine_PredictMigrationAccess(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.ErrorLevel)

	engine, err := NewPredictivePrefetchingEngine(logger)
	if err != nil {
		t.Fatalf("Failed to create predictive prefetching engine: %v", err)
	}

	ctx := context.Background()
	vmID := "test-vm-123"
	
	migrationSpec := &MigrationSpec{
		Type:               MigrationTypeLive,
		SourceNode:         "node-1",
		DestinationNode:    "node-2",
		NetworkBandwidth:   1024 * 1024 * 100, // 100 MB/s
		EstimatedDuration:  30 * time.Minute,
		CompressionEnabled: true,
		EncryptionEnabled:  false,
	}

	predictionResult, err := engine.PredictMigrationAccess(ctx, vmID, migrationSpec)
	if err != nil {
		t.Fatalf("Failed to predict migration access: %v", err)
	}

	if predictionResult == nil {
		t.Fatal("Expected non-nil prediction result")
	}

	if predictionResult.VMData == nil {
		t.Fatal("Expected VM data to be set")
	}

	if predictionResult.VMData.VMID != vmID {
		t.Errorf("Expected VM ID %s, got %s", vmID, predictionResult.VMData.VMID)
	}

	if predictionResult.Confidence < 0 || predictionResult.Confidence > 1 {
		t.Errorf("Expected confidence between 0 and 1, got %f", predictionResult.Confidence)
	}

	if len(predictionResult.PredictedAccess) == 0 {
		t.Error("Expected at least some access predictions")
	}
}

func TestPredictivePrefetchingEngine_ExecutePredictivePrefetching(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.ErrorLevel)

	engine, err := NewPredictivePrefetchingEngine(logger)
	if err != nil {
		t.Fatalf("Failed to create predictive prefetching engine: %v", err)
	}

	ctx := context.Background()

	// Create mock prediction result
	predictionResult := &PredictionResult{
		PredictionID: "test-prediction-123",
		VMData: &VMDataFeatures{
			VMID: "test-vm-123",
		},
		PredictedAccess: []*AccessPrediction{
			{
				PageID:        "page_test_vm_1",
				Probability:   0.9,
				AccessTime:    time.Now().Add(5 * time.Minute),
				AccessPattern: "sequential",
				Priority:      9,
				Size:          4096,
			},
			{
				PageID:        "page_test_vm_2",
				Probability:   0.8,
				AccessTime:    time.Now().Add(10 * time.Minute),
				AccessPattern: "random",
				Priority:      8,
				Size:          4096,
			},
		},
		Confidence:     0.85,
		ModelUsed:      "v2.1.0",
		PredictionTime: time.Now(),
		ExpiresAt:      time.Now().Add(10 * time.Minute),
	}

	prefetchPolicy := &PrefetchPolicy{
		MinConfidenceThreshold: 0.7,
		MaxPrefetchItems:       100,
		MaxPrefetchSize:        10 * 1024 * 1024, // 10 MB
		PrefetchAheadTime:      5 * time.Minute,
		EvictionPolicy:         EvictionPolicyAIPriority,
	}

	prefetchResult, err := engine.ExecutePredictivePrefetching(ctx, predictionResult, prefetchPolicy)
	if err != nil {
		t.Fatalf("Failed to execute predictive prefetching: %v", err)
	}

	if prefetchResult == nil {
		t.Fatal("Expected non-nil prefetch result")
	}

	if prefetchResult.PredictionID != predictionResult.PredictionID {
		t.Errorf("Expected prediction ID %s, got %s", predictionResult.PredictionID, prefetchResult.PredictionID)
	}

	if len(prefetchResult.PrefetchedItems) == 0 {
		t.Error("Expected at least some prefetched items")
	}

	if prefetchResult.TotalBytesPreloaded <= 0 {
		t.Error("Expected positive bytes preloaded")
	}
}

func TestPredictivePrefetchingEngine_ValidateTargets(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.ErrorLevel)

	engine, err := NewPredictivePrefetchingEngine(logger)
	if err != nil {
		t.Fatalf("Failed to create predictive prefetching engine: %v", err)
	}

	// Set good metrics to pass validation
	engine.prefetchMetrics.TotalPredictions = 100
	engine.prefetchMetrics.SuccessfulPredictions = 85
	engine.prefetchMetrics.CacheHitRatioImprovement = 0.35
	engine.prefetchMetrics.AveragePredictionTime = 5 * time.Millisecond

	err = engine.ValidatePrefetchingTargets()
	if err != nil {
		t.Errorf("Expected validation to pass, got error: %v", err)
	}

	// Test with poor metrics
	engine.prefetchMetrics.SuccessfulPredictions = 50 // Below 85% accuracy
	err = engine.ValidatePrefetchingTargets()
	if err == nil {
		t.Error("Expected validation to fail with poor accuracy")
	}
}

func TestMigrationExecutorImpl_PredictivePrefetching_Integration(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.ErrorLevel)

	executor, err := NewVMMigrationExecutor(logger, "/tmp/test-migrations")
	if err != nil {
		t.Fatalf("Failed to create migration executor: %v", err)
	}

	// Test that predictive prefetching is enabled by default
	if !executor.prefetchingEnabled {
		t.Error("Expected predictive prefetching to be enabled by default")
	}

	// Test disabling predictive prefetching
	executor.EnablePredictivePrefetching(false)
	if executor.prefetchingEnabled {
		t.Error("Expected predictive prefetching to be disabled")
	}

	// Test re-enabling predictive prefetching
	executor.EnablePredictivePrefetching(true)
	if !executor.prefetchingEnabled {
		t.Error("Expected predictive prefetching to be enabled")
	}
}

func TestDefaultPredictivePrefetchingConfig(t *testing.T) {
	config := DefaultPredictivePrefetchingConfig()

	if config == nil {
		t.Fatal("Expected non-nil config")
	}

	if !config.Enabled {
		t.Error("Expected predictive prefetching to be enabled by default")
	}

	if config.PredictionAccuracy != TARGET_PREDICTION_ACCURACY {
		t.Errorf("Expected prediction accuracy %f, got %f", TARGET_PREDICTION_ACCURACY, config.PredictionAccuracy)
	}

	if config.PredictionLatencyMs != TARGET_PREDICTION_LATENCY_MS {
		t.Errorf("Expected prediction latency %d ms, got %d ms", TARGET_PREDICTION_LATENCY_MS, config.PredictionLatencyMs)
	}

	if config.ModelType != "neural_network" {
		t.Errorf("Expected model type 'neural_network', got '%s'", config.ModelType)
	}

	if !config.ContinuousLearning {
		t.Error("Expected continuous learning to be enabled by default")
	}

	if len(config.AIModelConfig) == 0 {
		t.Error("Expected AI model config to have default parameters")
	}
}

func BenchmarkPredictivePrefetching_PredictAccess(b *testing.B) {
	logger := logrus.New()
	logger.SetLevel(logrus.ErrorLevel)

	engine, err := NewPredictivePrefetchingEngine(logger)
	if err != nil {
		b.Fatalf("Failed to create predictive prefetching engine: %v", err)
	}

	ctx := context.Background()
	vmID := "benchmark-vm"
	
	migrationSpec := &MigrationSpec{
		Type:               MigrationTypeLive,
		SourceNode:         "node-1",
		DestinationNode:    "node-2",
		NetworkBandwidth:   1024 * 1024 * 100,
		EstimatedDuration:  30 * time.Minute,
		CompressionEnabled: true,
		EncryptionEnabled:  false,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := engine.PredictMigrationAccess(ctx, vmID, migrationSpec)
		if err != nil {
			b.Fatalf("Prediction failed: %v", err)
		}
	}
}