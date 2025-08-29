package vm

import (
	"context"
	"fmt"
	"time"

	"github.com/sirupsen/logrus"
)

// DemonstratePredictivePrefetching shows how predictive prefetching works
func DemonstratePredictivePrefetching() error {
	fmt.Println("=== NovaCron Predictive Prefetching Demo ===\n")

	// Create logger
	logger := logrus.New()
	logger.SetLevel(logrus.InfoLevel)
	logger.SetFormatter(&logrus.TextFormatter{
		FullTimestamp: true,
		ForceColors:   true,
	})

	// Create predictive prefetching engine
	fmt.Println("1. Initializing AI-Driven Predictive Prefetching Engine...")
	engine, err := NewPredictivePrefetchingEngine(logger)
	if err != nil {
		return fmt.Errorf("failed to create predictive prefetching engine: %w", err)
	}

	// Display engine configuration
	fmt.Printf("✓ Engine created successfully\n")
	fmt.Printf("  - AI Model: %s (v%s)\n", engine.aiModel.ModelType, engine.aiModel.ModelVersion)
	fmt.Printf("  - Target Accuracy: %.1f%%\n", TARGET_PREDICTION_ACCURACY*100)
	fmt.Printf("  - Target Latency: ≤%dms\n", TARGET_PREDICTION_LATENCY_MS)
	fmt.Printf("  - Cache Size: %d MB\n", engine.cacheManager.CacheSize/(1024*1024))
	fmt.Println()

	// Create sample VM and migration scenario
	fmt.Println("2. Setting up Migration Scenario...")
	vmID := "demo-web-server-001"
	migrationSpec := &MigrationSpec{
		Type:               MigrationTypeLive,
		SourceNode:         "datacenter-east-node-01",
		DestinationNode:    "datacenter-west-node-03",
		NetworkBandwidth:   1024 * 1024 * 100, // 100 MB/s
		EstimatedDuration:  30 * time.Minute,
		CompressionEnabled: true,
		EncryptionEnabled:  false,
	}

	fmt.Printf("✓ Migration configured\n")
	fmt.Printf("  - VM ID: %s\n", vmID)
	fmt.Printf("  - Type: %s migration\n", migrationSpec.Type)
	fmt.Printf("  - Route: %s → %s\n", migrationSpec.SourceNode, migrationSpec.DestinationNode)
	fmt.Printf("  - Bandwidth: %d MB/s\n", migrationSpec.NetworkBandwidth/(1024*1024))
	fmt.Println()

	// Execute AI prediction
	fmt.Println("3. Generating AI-Driven Access Predictions...")
	ctx := context.Background()
	
	start := time.Now()
	predictionResult, err := engine.PredictMigrationAccess(ctx, vmID, migrationSpec)
	if err != nil {
		return fmt.Errorf("AI prediction failed: %w", err)
	}
	predictionLatency := time.Since(start)

	fmt.Printf("✓ AI predictions generated in %v\n", predictionLatency)
	fmt.Printf("  - Predictions: %d access patterns\n", len(predictionResult.PredictedAccess))
	fmt.Printf("  - Confidence: %.2f%% (target: %.1f%%)\n", predictionResult.Confidence*100, TARGET_PREDICTION_ACCURACY*100)
	fmt.Printf("  - Model: %s\n", predictionResult.ModelUsed)

	// Show top predictions
	if len(predictionResult.PredictedAccess) > 0 {
		fmt.Printf("  - Top predictions:\n")
		for i, pred := range predictionResult.PredictedAccess {
			if i >= 3 { // Show only top 3
				break
			}
			fmt.Printf("    • %s (%.1f%% probability, %s pattern)\n", 
				pred.PageID, pred.Probability*100, pred.AccessPattern)
		}
	}
	fmt.Println()

	// Execute predictive prefetching
	fmt.Println("4. Executing Intelligent Prefetching...")
	prefetchPolicy := &PrefetchPolicy{
		MinConfidenceThreshold: 0.8, // 80% confidence threshold
		MaxPrefetchItems:       100,
		MaxPrefetchSize:        50 * 1024 * 1024, // 50 MB
		PrefetchAheadTime:      10 * time.Minute,
		EvictionPolicy:         EvictionPolicyAIPriority,
	}

	prefetchResult, err := engine.ExecutePredictivePrefetching(ctx, predictionResult, prefetchPolicy)
	if err != nil {
		return fmt.Errorf("predictive prefetching failed: %w", err)
	}

	fmt.Printf("✓ Intelligent prefetching completed in %v\n", prefetchResult.Duration)
	fmt.Printf("  - Items prefetched: %d\n", len(prefetchResult.PrefetchedItems))
	fmt.Printf("  - Data preloaded: %.2f MB\n", float64(prefetchResult.TotalBytesPreloaded)/(1024*1024))
	fmt.Printf("  - Cache hit improvement: %.1f%%\n", prefetchResult.CacheHitImprovement*100)
	fmt.Println()

	// Validate performance targets
	fmt.Println("5. Validating Performance Targets...")
	err = engine.ValidatePrefetchingTargets()
	if err != nil {
		fmt.Printf("⚠ Performance targets not fully met: %v\n", err)
	} else {
		fmt.Printf("✓ All performance targets achieved!\n")
	}

	// Get comprehensive metrics
	metrics := engine.GetPrefetchingMetrics()
	fmt.Printf("  - Prediction accuracy: %.2f%% (target: %.1f%%)\n", 
		metrics.PredictionAccuracy*100, TARGET_PREDICTION_ACCURACY*100)
	fmt.Printf("  - Cache improvement: %.1f%% (target: %.1f%%)\n", 
		metrics.CacheHitRatioImprovement*100, TARGET_CACHE_HIT_IMPROVEMENT*100)
	fmt.Printf("  - Prediction latency: %v (target: ≤%dms)\n", 
		metrics.AveragePredictionTime, TARGET_PREDICTION_LATENCY_MS)
	fmt.Println()

	// Show migration performance benefits
	fmt.Println("6. Expected Migration Performance Benefits:")
	fmt.Printf("  • Faster data transfer due to intelligent prefetching\n")
	fmt.Printf("  • Reduced migration downtime\n")
	fmt.Printf("  • Optimized network bandwidth usage\n")
	fmt.Printf("  • AI-driven continuous improvement\n")
	fmt.Println()

	fmt.Println("=== Demo Complete ===")
	fmt.Println("Predictive prefetching is now active and ready to accelerate VM migrations!")

	return nil
}

// RunPredictivePrefetchingDemo is a helper function to run the demo
func RunPredictivePrefetchingDemo() {
	if err := DemonstratePredictivePrefetching(); err != nil {
		fmt.Printf("Demo failed: %v\n", err)
	}
}