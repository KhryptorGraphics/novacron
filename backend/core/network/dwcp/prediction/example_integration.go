package prediction

import (
	"context"
	"fmt"
	"time"

	"go.uber.org/zap"
)

// Example demonstrates full PBA system integration with DWCP AMST

func ExamplePBAIntegration() {
	// Initialize logger
	logger, _ := zap.NewProduction()
	defer logger.Sync()

	// Step 1: Create and start data collector
	fmt.Println("Step 1: Initializing data collector...")
	collector := NewDataCollector(1*time.Minute, 10000)
	collector.Start()
	defer collector.Stop()

	// Wait for initial data collection
	fmt.Println("Collecting initial network samples...")
	time.Sleep(2 * time.Minute)

	// Export initial training data
	err := collector.ExportForTraining("/tmp/training_data.csv")
	if err != nil {
		logger.Error("Failed to export training data", zap.Error(err))
		return
	}
	fmt.Println("Training data exported to /tmp/training_data.csv")

	// Step 2: Train model (in production, this would be done offline)
	fmt.Println("\nStep 2: Training LSTM model...")
	fmt.Println("Run: python3 training/train_lstm.py --data /tmp/training_data.csv --output ./models")

	// Step 3: Initialize prediction service with trained model
	fmt.Println("\nStep 3: Loading LSTM predictor...")
	predictionService, err := NewPredictionService(
		"./models/bandwidth_lstm_v1.onnx",
		1*time.Minute,
	)
	if err != nil {
		logger.Error("Failed to create prediction service", zap.Error(err))
		return
	}
	defer predictionService.Stop()

	err = predictionService.Start()
	if err != nil {
		logger.Error("Failed to start prediction service", zap.Error(err))
		return
	}

	// Step 4: Create AMST optimizer
	fmt.Println("\nStep 4: Initializing AMST optimizer...")
	optimizer := NewAMSTOptimizer(predictionService, logger)
	optimizer.Start()
	defer optimizer.Stop()

	// Step 5: Main prediction loop
	fmt.Println("\nStep 5: Running prediction loop...\n")

	ticker := time.NewTicker(1 * time.Minute)
	defer ticker.Stop()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	for {
		select {
		case <-ctx.Done():
			fmt.Println("\nExample completed")
			return

		case <-ticker.C:
			// Get current prediction
			prediction := predictionService.GetPrediction()
			if prediction == nil {
				fmt.Println("Waiting for prediction...")
				continue
			}

			// Display prediction
			fmt.Printf("\n[%s] Prediction (confidence: %.2f)\n",
				time.Now().Format("15:04:05"),
				prediction.Confidence)
			fmt.Printf("  Bandwidth: %.1f Mbps\n", prediction.PredictedBandwidthMbps)
			fmt.Printf("  Latency:   %.1f ms\n", prediction.PredictedLatencyMs)
			fmt.Printf("  Loss:      %.2f%%\n", prediction.PredictedPacketLoss*100)
			fmt.Printf("  Jitter:    %.1f ms\n", prediction.PredictedJitterMs)

			// Get optimized parameters
			params := optimizer.GetCurrentParameters()
			fmt.Printf("\nOptimized AMST Parameters:\n")
			fmt.Printf("  Streams:     %d\n", params.NumStreams)
			fmt.Printf("  Buffer Size: %d bytes (%.1f KB)\n",
				params.BufferSize, float64(params.BufferSize)/1024)
			fmt.Printf("  Chunk Size:  %d bytes (%.1f KB)\n",
				params.ChunkSize, float64(params.ChunkSize)/1024)
			fmt.Printf("  Pacing Rate: %.1f Mbps\n",
				float64(params.PacingRate)*8/1000000)
			fmt.Printf("  Reason: %s\n", params.Reason)

			// Check if stream adjustment needed
			if shouldAdjust, newStreams, reason := optimizer.ShouldAdjustStreams(); shouldAdjust {
				fmt.Printf("\n⚠️  Stream adjustment recommended:\n")
				fmt.Printf("  Current: %d streams\n", params.NumStreams)
				fmt.Printf("  Recommended: %d streams\n", newStreams)
				fmt.Printf("  Reason: %s\n", reason)
			}

			// Check for preemptive optimization
			if preemptive := optimizer.PreemptiveOptimization(); preemptive != nil {
				fmt.Printf("\n🔮 Preemptive optimization available:\n")
				fmt.Printf("  Streams: %d → %d\n", params.NumStreams, preemptive.NumStreams)
				fmt.Printf("  Confidence: %.2f\n", preemptive.Confidence)
			}

			// Get predictor metrics
			metrics := predictionService.predictor.GetMetrics()
			fmt.Printf("\nPredictor Performance:\n")
			fmt.Printf("  Model Version: %s\n", metrics.ModelVersion)
			fmt.Printf("  Inferences: %d\n", metrics.InferenceCount)
			fmt.Printf("  Avg Latency: %.2f ms\n", metrics.AvgInferenceMs)
			fmt.Printf("  Accuracy: %.1f%%\n", metrics.Accuracy*100)

			// Get data collector statistics
			stats := collector.GetStatistics()
			fmt.Printf("\nData Collection:\n")
			fmt.Printf("  Samples: %d\n", stats.SampleCount)
			fmt.Printf("  Avg Bandwidth: %.1f Mbps\n", stats.AvgBandwidth)
			fmt.Printf("  Avg Latency: %.1f ms\n", stats.AvgLatency)

			fmt.Println("\n" + "─"*60)
		}
	}
}

// ExampleQuickStart demonstrates minimal PBA setup
func ExampleQuickStart() {
	logger, _ := zap.NewProduction()

	// Initialize with defaults
	service, err := NewPredictionService("./models/bandwidth_lstm_v1.onnx", 1*time.Minute)
	if err != nil {
		logger.Fatal("Failed to create service", zap.Error(err))
	}

	service.Start()
	defer service.Stop()

	// Get prediction
	time.Sleep(2 * time.Minute) // Wait for samples

	if pred := service.GetPrediction(); pred != nil {
		fmt.Printf("Predicted bandwidth: %.1f Mbps\n", pred.PredictedBandwidthMbps)
		fmt.Printf("Optimal streams: %d\n", service.GetOptimalStreamCount())
		fmt.Printf("Optimal buffer: %d bytes\n", service.GetOptimalBufferSize())
	}
}

// ExampleABTesting demonstrates A/B testing between models
func ExampleABTesting() error {
	logger, _ := zap.NewProduction()

	// Create primary service
	service, err := NewPredictionService("./models/bandwidth_lstm_v1.onnx", 1*time.Minute)
	if err != nil {
		return err
	}
	defer service.Stop()

	// Enable A/B testing with alternate model
	err = service.EnableABTesting("./models/bandwidth_lstm_v2.onnx")
	if err != nil {
		return err
	}

	service.Start()

	// Run for test duration
	time.Sleep(1 * time.Hour)

	// Get results
	results := service.GetABTestResults()
	if results != nil {
		fmt.Printf("A/B Test Results:\n")
		fmt.Printf("  Primary Accuracy: %.2f%%\n", results.PrimaryAccuracy*100)
		fmt.Printf("  Alternate Accuracy: %.2f%%\n", results.AlternateAccuracy*100)
		fmt.Printf("  Primary Latency: %v\n", results.PrimaryLatency)
		fmt.Printf("  Alternate Latency: %v\n", results.AlternateLatency)
		fmt.Printf("  Predictions: %d\n", results.PredictionCount)
		fmt.Printf("  Winner: %s\n", results.WinningModel)
	}

	return nil
}

// ExampleMetricsExport demonstrates exporting metrics for analysis
func ExampleMetricsExport() error {
	service, err := NewPredictionService("./models/bandwidth_lstm_v1.onnx", 1*time.Minute)
	if err != nil {
		return err
	}
	defer service.Stop()

	service.Start()

	// Run for some time
	time.Sleep(30 * time.Minute)

	// Export metrics
	err = service.ExportMetrics("/tmp/pba_metrics.json")
	if err != nil {
		return err
	}

	fmt.Println("Metrics exported to /tmp/pba_metrics.json")
	return nil
}

// ExamplePrometheusIntegration demonstrates Prometheus metrics
func ExamplePrometheusIntegration() {
	// Metrics are automatically registered with Prometheus
	// Available metrics:
	// - dwcp_pba_current_bandwidth_mbps
	// - dwcp_pba_current_latency_ms
	// - dwcp_pba_current_packet_loss_ratio
	// - dwcp_pba_current_jitter_ms
	// - dwcp_pba_sample_count
	// - dwcp_pba_prediction_accuracy
	// - dwcp_pba_prediction_latency_ms
	// - dwcp_pba_model_version
	// - dwcp_pba_confidence
	// - dwcp_pba_retrain_total
	// - dwcp_pba_predictions_total

	fmt.Println("Prometheus metrics available at :9090/metrics")
	fmt.Println("\nExample queries:")
	fmt.Println("  rate(dwcp_pba_predictions_total[5m])")
	fmt.Println("  dwcp_pba_prediction_accuracy")
	fmt.Println("  histogram_quantile(0.95, dwcp_pba_prediction_latency_ms)")
}

// ExampleCustomOptimizer demonstrates custom optimization logic
func ExampleCustomOptimizer() {
	logger, _ := zap.NewProduction()

	service, _ := NewPredictionService("./models/bandwidth_lstm_v1.onnx", 1*time.Minute)
	service.Start()
	defer service.Stop()

	optimizer := NewAMSTOptimizer(service, logger)

	// Customize bounds
	optimizer.minStreams = 4
	optimizer.maxStreams = 128
	optimizer.minBuffer = 32768
	optimizer.maxBuffer = 524288

	optimizer.Start()
	defer optimizer.Stop()

	// Get optimization history
	time.Sleep(5 * time.Minute)

	history := optimizer.GetOptimizationHistory()
	fmt.Printf("Optimization history: %d entries\n", len(history))

	for _, record := range history {
		fmt.Printf("[%s] Streams: %d, Confidence: %.2f, Applied: %v\n",
			record.Timestamp.Format("15:04:05"),
			record.Streams,
			record.Confidence,
			record.Applied)
	}

	// Check effectiveness
	effectiveness := optimizer.GetOptimizationEffectiveness()
	fmt.Printf("\nOptimization effectiveness: %.2fx\n", effectiveness)
}
