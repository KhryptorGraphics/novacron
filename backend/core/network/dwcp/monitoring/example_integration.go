package monitoring

import (
	"context"
	"fmt"
	"time"

	"go.uber.org/zap"
)

// ExampleIntegration demonstrates how to integrate the anomaly detection system
func ExampleIntegration() {
	// Initialize logger
	logger, err := zap.NewProduction()
	if err != nil {
		panic(err)
	}
	defer logger.Sync()

	// Load configuration from file
	config, err := LoadConfigFromFile("configs/monitoring/anomaly-detection.yaml")
	if err != nil {
		logger.Fatal("Failed to load config", zap.Error(err))
	}

	// Validate configuration
	if err := config.Validate(); err != nil {
		logger.Fatal("Invalid configuration", zap.Error(err))
	}

	// Create anomaly detector
	detector, err := NewAnomalyDetector(config.Detector, logger)
	if err != nil {
		logger.Fatal("Failed to create detector", zap.Error(err))
	}

	// Create alert manager
	alertManager := NewAlertManager(config.Alert, logger)

	// Create monitoring pipeline
	pipeline := NewMonitoringPipeline(
		detector,
		alertManager,
		config.CheckInterval,
		logger,
	)

	// Train detector with historical normal data
	normalData := loadHistoricalData(logger)
	if len(normalData) > 0 {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
		defer cancel()

		if err := pipeline.TrainDetector(ctx, normalData); err != nil {
			logger.Error("Failed to train detector", zap.Error(err))
		} else {
			logger.Info("Detector training completed successfully")
		}
	}

	// Start the monitoring pipeline
	if err := pipeline.Start(); err != nil {
		logger.Fatal("Failed to start pipeline", zap.Error(err))
	}
	defer pipeline.Stop()

	logger.Info("Anomaly detection system started",
		zap.Duration("check_interval", config.CheckInterval),
		zap.Int("buffer_size", config.BufferSize))

	// Start feeding metrics
	go feedMetrics(pipeline, logger)

	// Monitor statistics
	go monitorStatistics(pipeline, logger)

	// Keep running
	select {}
}

// loadHistoricalData loads historical normal operating data for training
func loadHistoricalData(logger *zap.Logger) []*MetricVector {
	// In production, this would load from:
	// - Prometheus query (last 7 days of normal data)
	// - Database export
	// - CSV file
	// - Time-series database

	logger.Info("Loading historical data for training...")

	// For this example, generate synthetic normal data
	data := make([]*MetricVector, 10000)
	baseTime := time.Now().Add(-7 * 24 * time.Hour)

	for i := 0; i < len(data); i++ {
		// Simulate hourly data with daily seasonality
		hour := i % 24
		seasonal := float64(hour) / 24.0 * 10.0 // Simple seasonal pattern

		data[i] = &MetricVector{
			Timestamp:   baseTime.Add(time.Duration(i) * time.Hour),
			Bandwidth:   100.0 + seasonal,
			Latency:     10.0 - seasonal*0.2,
			PacketLoss:  0.01,
			Jitter:      1.0 + seasonal*0.05,
			CPUUsage:    50.0 + seasonal,
			MemoryUsage: 60.0,
			ErrorRate:   0.001,
		}
	}

	logger.Info("Historical data loaded",
		zap.Int("samples", len(data)),
		zap.Time("start", data[0].Timestamp),
		zap.Time("end", data[len(data)-1].Timestamp))

	return data
}

// feedMetrics continuously feeds metrics to the pipeline
func feedMetrics(pipeline *MonitoringPipeline, logger *zap.Logger) {
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	logger.Info("Starting metrics collection")

	for range ticker.C {
		// In production, these would be real metric values from:
		// - Prometheus queries
		// - DWCP network measurements
		// - System monitoring agents
		// - Application metrics

		metrics := &MetricVector{
			Timestamp:   time.Now(),
			Bandwidth:   getCurrentBandwidth(),
			Latency:     getCurrentLatency(),
			PacketLoss:  getCurrentPacketLoss(),
			Jitter:      getCurrentJitter(),
			CPUUsage:    getCurrentCPUUsage(),
			MemoryUsage: getCurrentMemoryUsage(),
			ErrorRate:   getCurrentErrorRate(),
		}

		if err := pipeline.ProcessMetrics(metrics); err != nil {
			logger.Error("Failed to process metrics", zap.Error(err))
		}
	}
}

// monitorStatistics periodically logs pipeline statistics
func monitorStatistics(pipeline *MonitoringPipeline, logger *zap.Logger) {
	ticker := time.NewTicker(1 * time.Minute)
	defer ticker.Stop()

	for range ticker.C {
		stats := pipeline.GetStats()

		logger.Info("Pipeline statistics",
			zap.Int64("metrics_processed", stats.MetricsProcessed),
			zap.Int64("anomalies_detected", stats.AnomaliesDetected),
			zap.Int64("alerts_sent", stats.AlertsSent),
			zap.Duration("avg_latency", stats.AverageLatency),
			zap.Int64("errors", stats.ErrorCount),
			zap.Time("last_check", stats.LastCheckTime))
	}
}

// Mock metric collection functions
// In production, these would query real systems

func getCurrentBandwidth() float64 {
	// Query current network bandwidth utilization
	// Example: Prometheus query, SNMP, NetFlow
	return 105.0 // Mbps
}

func getCurrentLatency() float64 {
	// Measure current network latency
	// Example: Ping, HTTP probe, TCP handshake timing
	return 11.0 // ms
}

func getCurrentPacketLoss() float64 {
	// Calculate packet loss rate
	// Example: Network interface counters, probe results
	return 0.01 // %
}

func getCurrentJitter() float64 {
	// Measure network jitter
	// Example: Standard deviation of latency samples
	return 1.1 // ms
}

func getCurrentCPUUsage() float64 {
	// Get CPU utilization
	// Example: /proc/stat, container metrics, cloud provider API
	return 51.0 // %
}

func getCurrentMemoryUsage() float64 {
	// Get memory utilization
	// Example: /proc/meminfo, container metrics
	return 61.0 // %
}

func getCurrentErrorRate() float64 {
	// Calculate error rate
	// Example: HTTP 5xx / total requests, failed transfers
	return 0.001 // %
}

// ExampleManualDetection shows how to manually trigger detection
func ExampleManualDetection() {
	logger, _ := zap.NewProduction()
	defer logger.Sync()

	// Create detector
	config := DefaultDetectorConfig()
	detector, _ := NewAnomalyDetector(config, logger)

	// Create a metric vector
	metrics := &MetricVector{
		Timestamp:   time.Now(),
		Bandwidth:   50.0,  // Anomalously low
		Latency:     100.0, // Anomalously high
		PacketLoss:  5.0,   // Anomalously high
		Jitter:      20.0,  // Anomalously high
		CPUUsage:    95.0,  // Anomalously high
		MemoryUsage: 95.0,  // Anomalously high
		ErrorRate:   0.1,   // Anomalously high
	}

	// Detect anomalies
	ctx := context.Background()
	anomalies, err := detector.Detect(ctx, metrics)
	if err != nil {
		logger.Error("Detection failed", zap.Error(err))
		return
	}

	// Process results
	for _, anomaly := range anomalies {
		logger.Warn("Anomaly detected",
			zap.String("metric", anomaly.MetricName),
			zap.Float64("value", anomaly.Value),
			zap.Float64("expected", anomaly.Expected),
			zap.Float64("deviation", anomaly.Deviation),
			zap.String("severity", anomaly.Severity.String()),
			zap.Float64("confidence", anomaly.Confidence),
			zap.String("model", anomaly.ModelType),
			zap.String("description", anomaly.Description))
	}
}

// ExampleCustomAlertHandler shows how to implement custom alert handling
func ExampleCustomAlertHandler() {
	logger, _ := zap.NewProduction()
	defer logger.Sync()

	// Create alert manager with custom webhook
	config := &AlertConfig{
		WebhookEnabled: true,
		WebhookURL:     "https://your-webhook.com/dwcp-alerts",
		ThrottleDuration: 5 * time.Minute,
	}

	alertManager := NewAlertManager(config, logger)

	// Send a test alert
	anomaly := &Anomaly{
		Timestamp:   time.Now(),
		MetricName:  "bandwidth",
		Value:       50.0,
		Expected:    100.0,
		Deviation:   50.0,
		Severity:    SeverityCritical,
		Confidence:  0.95,
		ModelType:   "ensemble",
		Description: "Bandwidth dropped by 50%",
	}

	if err := alertManager.SendCriticalAlert(anomaly); err != nil {
		logger.Error("Failed to send alert", zap.Error(err))
	}
}

// ExampleTrainingWorkflow shows the complete training workflow
func ExampleTrainingWorkflow() {
	logger, _ := zap.NewProduction()
	defer logger.Sync()

	// Step 1: Collect normal operating data
	logger.Info("Step 1: Collecting normal operating data...")
	normalData := collectNormalData(logger, 7*24*time.Hour) // 7 days

	// Step 2: Create detector
	logger.Info("Step 2: Creating detector...")
	config := DefaultDetectorConfig()
	detector, _ := NewAnomalyDetector(config, logger)

	// Step 3: Train detector
	logger.Info("Step 3: Training detector...")
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	if err := detector.Train(ctx, normalData); err != nil {
		logger.Fatal("Training failed", zap.Error(err))
	}

	logger.Info("Training completed successfully")

	// Step 4: Validate on test data
	logger.Info("Step 4: Validating on test data...")
	testData := loadTestData(logger)

	truePositives := 0
	falsePositives := 0

	for _, testCase := range testData {
		anomalies, _ := detector.Detect(ctx, testCase.Metrics)

		if testCase.IsAnomaly {
			if len(anomalies) > 0 {
				truePositives++
			}
		} else {
			if len(anomalies) > 0 {
				falsePositives++
			}
		}
	}

	accuracy := float64(truePositives) / float64(len(testData)) * 100
	falsePositiveRate := float64(falsePositives) / float64(len(testData)) * 100

	logger.Info("Validation results",
		zap.Float64("accuracy", accuracy),
		zap.Float64("false_positive_rate", falsePositiveRate))
}

// TestCase represents a labeled test case
type TestCase struct {
	Metrics   *MetricVector
	IsAnomaly bool
	Label     string
}

func collectNormalData(logger *zap.Logger, duration time.Duration) []*MetricVector {
	// In production: Query Prometheus for normal periods
	// Filter out known incidents
	// Ensure data quality
	return loadHistoricalData(logger)
}

func loadTestData(logger *zap.Logger) []*TestCase {
	// Load labeled test data
	// Include both normal and anomalous cases
	return []*TestCase{
		{
			Metrics:   &MetricVector{Bandwidth: 105.0, Latency: 11.0},
			IsAnomaly: false,
			Label:     "normal",
		},
		{
			Metrics:   &MetricVector{Bandwidth: 50.0, Latency: 100.0},
			IsAnomaly: true,
			Label:     "bandwidth_drop",
		},
	}
}

// init function for example - not used in production
func init() {
	fmt.Println("DWCP Anomaly Detection - Example Integration")
	fmt.Println("This file demonstrates how to integrate the anomaly detection system")
	fmt.Println("See the main() function in your application for actual usage")
}
