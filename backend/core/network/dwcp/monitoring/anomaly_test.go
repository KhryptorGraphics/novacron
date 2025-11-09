package monitoring

import (
	"context"
	"math"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
)

// TestMetricVector_ToSlice tests metric vector conversion
func TestMetricVector_ToSlice(t *testing.T) {
	mv := &MetricVector{
		Timestamp:   time.Now(),
		Bandwidth:   100.0,
		Latency:     10.0,
		PacketLoss:  0.01,
		Jitter:      1.0,
		CPUUsage:    50.0,
		MemoryUsage: 60.0,
		ErrorRate:   0.001,
	}

	slice := mv.ToSlice()

	assert.Equal(t, 7, len(slice))
	assert.Equal(t, 100.0, slice[0])
	assert.Equal(t, 10.0, slice[1])
	assert.Equal(t, 0.01, slice[2])
}

// TestIsolationForest_Detection tests Isolation Forest anomaly detection
func TestIsolationForest_Detection(t *testing.T) {
	logger := zap.NewNop()
	detector, err := NewIsolationForestModel("models/isolation_forest.onnx", logger)
	require.NoError(t, err)

	// Generate normal training data
	normalData := generateNormalData(1000)

	// Train the detector
	ctx := context.Background()
	err = detector.Train(ctx, normalData)
	require.NoError(t, err)

	// Test with normal data
	normalMetric := &MetricVector{
		Timestamp:   time.Now(),
		Bandwidth:   105.0,
		Latency:     11.0,
		PacketLoss:  0.01,
		Jitter:      1.1,
		CPUUsage:    51.0,
		MemoryUsage: 61.0,
		ErrorRate:   0.001,
	}

	anomaly, err := detector.Detect(ctx, normalMetric)
	require.NoError(t, err)
	assert.Nil(t, anomaly, "Normal data should not be detected as anomaly")

	// Test with anomalous data
	anomalousMetric := &MetricVector{
		Timestamp:   time.Now(),
		Bandwidth:   50.0,  // Very low
		Latency:     100.0, // Very high
		PacketLoss:  5.0,   // Very high
		Jitter:      20.0,  // Very high
		CPUUsage:    95.0,
		MemoryUsage: 95.0,
		ErrorRate:   0.1, // Very high
	}

	anomaly, err = detector.Detect(ctx, anomalousMetric)
	require.NoError(t, err)
	assert.NotNil(t, anomaly, "Anomalous data should be detected")

	if anomaly != nil {
		assert.Greater(t, anomaly.Confidence, 0.5)
		assert.NotEmpty(t, anomaly.MetricName)
		assert.Equal(t, "isolation_forest", anomaly.ModelType)
	}
}

// TestLSTMAutoencoder_Detection tests LSTM autoencoder anomaly detection
func TestLSTMAutoencoder_Detection(t *testing.T) {
	logger := zap.NewNop()
	detector, err := NewLSTMAutoencoderModel("models/lstm_autoencoder.onnx", logger)
	require.NoError(t, err)

	// Generate time series data
	timeSeriesData := generateTimeSeriesData(100)

	// Train the detector
	ctx := context.Background()
	err = detector.Train(ctx, timeSeriesData)
	require.NoError(t, err)

	// Test with normal pattern
	for i := 0; i < 10; i++ {
		normalMetric := &MetricVector{
			Timestamp:   time.Now(),
			Bandwidth:   100.0 + float64(i),
			Latency:     10.0 + float64(i)*0.1,
			PacketLoss:  0.01,
			Jitter:      1.0,
			CPUUsage:    50.0,
			MemoryUsage: 60.0,
			ErrorRate:   0.001,
		}

		anomaly, err := detector.Detect(ctx, normalMetric)
		require.NoError(t, err)

		if i < detector.windowSize {
			assert.Nil(t, anomaly, "Should not detect until window is full")
		}
	}

	// Test with sudden spike (anomaly)
	spikeMetric := &MetricVector{
		Timestamp:   time.Now(),
		Bandwidth:   200.0, // Sudden spike
		Latency:     50.0,  // Sudden spike
		PacketLoss:  0.01,
		Jitter:      1.0,
		CPUUsage:    50.0,
		MemoryUsage: 60.0,
		ErrorRate:   0.001,
	}

	anomaly, err := detector.Detect(ctx, spikeMetric)
	require.NoError(t, err)

	if anomaly != nil {
		assert.Equal(t, "lstm_autoencoder", anomaly.ModelType)
		assert.Greater(t, anomaly.Confidence, 0.0)
	}
}

// TestZScoreDetector_Detection tests Z-score statistical anomaly detection
func TestZScoreDetector_Detection(t *testing.T) {
	logger := zap.NewNop()
	detector := NewZScoreDetector(100, 3.0, logger)

	// Generate training data
	normalData := generateNormalData(100)

	// Train the detector
	ctx := context.Background()
	err := detector.Train(ctx, normalData)
	require.NoError(t, err)

	// Test with normal data
	normalMetric := &MetricVector{
		Timestamp:   time.Now(),
		Bandwidth:   105.0,
		Latency:     11.0,
		PacketLoss:  0.01,
		Jitter:      1.1,
		CPUUsage:    51.0,
		MemoryUsage: 61.0,
		ErrorRate:   0.001,
	}

	anomaly, err := detector.Detect(ctx, normalMetric)
	require.NoError(t, err)
	assert.Nil(t, anomaly, "Normal data should not trigger Z-score alert")

	// Test with outlier
	outlierMetric := &MetricVector{
		Timestamp:   time.Now(),
		Bandwidth:   200.0, // 4+ sigma from mean
		Latency:     10.0,
		PacketLoss:  0.01,
		Jitter:      1.0,
		CPUUsage:    50.0,
		MemoryUsage: 60.0,
		ErrorRate:   0.001,
	}

	anomaly, err = detector.Detect(ctx, outlierMetric)
	require.NoError(t, err)

	if anomaly != nil {
		assert.Equal(t, "zscore", anomaly.ModelType)
		assert.Greater(t, anomaly.Confidence, 0.5)
		assert.Equal(t, "bandwidth", anomaly.MetricName)
	}
}

// TestSeasonalESD_Detection tests Seasonal ESD anomaly detection
func TestSeasonalESD_Detection(t *testing.T) {
	logger := zap.NewNop()
	detector := NewSeasonalESDDetector(24, 10, logger)

	// Generate seasonal data
	seasonalData := generateSeasonalData(100, 24)

	// Train the detector
	ctx := context.Background()
	err := detector.Train(ctx, seasonalData)
	require.NoError(t, err)

	// Test with data following seasonal pattern
	normalMetric := &MetricVector{
		Timestamp:   time.Now(),
		Bandwidth:   100.0,
		Latency:     10.0,
		PacketLoss:  0.01,
		Jitter:      1.0,
		CPUUsage:    50.0,
		MemoryUsage: 60.0,
		ErrorRate:   0.001,
	}

	anomaly, err := detector.Detect(ctx, normalMetric)
	require.NoError(t, err)

	// Add more data points to build up seasonal pattern
	for i := 0; i < 50; i++ {
		metric := seasonalData[i]
		_, _ = detector.Detect(ctx, metric)
	}

	// Test with anomaly breaking seasonal pattern
	anomalousMetric := &MetricVector{
		Timestamp:   time.Now(),
		Bandwidth:   200.0, // Breaking expected seasonal pattern
		Latency:     10.0,
		PacketLoss:  0.01,
		Jitter:      1.0,
		CPUUsage:    50.0,
		MemoryUsage: 60.0,
		ErrorRate:   0.001,
	}

	anomaly, err = detector.Detect(ctx, anomalousMetric)
	require.NoError(t, err)

	if anomaly != nil {
		assert.Equal(t, "seasonal_esd", anomaly.ModelType)
	}
}

// TestEnsembleDetector_Aggregation tests ensemble detector aggregation
func TestEnsembleDetector_Aggregation(t *testing.T) {
	logger := zap.NewNop()
	ensemble := NewEnsembleDetector(0.6, logger)

	metrics := &MetricVector{
		Timestamp:   time.Now(),
		Bandwidth:   150.0,
		Latency:     20.0,
		PacketLoss:  0.05,
		Jitter:      2.0,
		CPUUsage:    70.0,
		MemoryUsage: 80.0,
		ErrorRate:   0.01,
	}

	// Test with no detections
	results := []DetectorResult{}
	anomaly := ensemble.Aggregate(results, metrics)
	assert.Nil(t, anomaly)

	// Test with single detection (below threshold)
	results = []DetectorResult{
		{
			Detector: "zscore",
			Anomaly: &Anomaly{
				MetricName: "bandwidth",
				Value:      150.0,
				Expected:   100.0,
				Confidence: 0.5,
			},
			Weight: 0.3,
		},
	}
	anomaly = ensemble.Aggregate(results, metrics)
	assert.Nil(t, anomaly, "Single low-confidence detection should not trigger")

	// Test with multiple detections (above threshold)
	results = []DetectorResult{
		{
			Detector: "isolation_forest",
			Anomaly: &Anomaly{
				MetricName: "bandwidth",
				Value:      150.0,
				Expected:   100.0,
				Confidence: 0.8,
			},
			Weight: 0.3,
		},
		{
			Detector: "zscore",
			Anomaly: &Anomaly{
				MetricName: "bandwidth",
				Value:      150.0,
				Expected:   100.0,
				Confidence: 0.9,
			},
			Weight: 0.3,
		},
		{
			Detector: "lstm_autoencoder",
			Anomaly: &Anomaly{
				MetricName: "bandwidth",
				Value:      150.0,
				Expected:   100.0,
				Confidence: 0.7,
			},
			Weight: 0.3,
		},
	}

	anomaly = ensemble.Aggregate(results, metrics)
	assert.NotNil(t, anomaly, "Multiple high-confidence detections should trigger")

	if anomaly != nil {
		assert.Equal(t, "ensemble", anomaly.ModelType)
		assert.Greater(t, anomaly.Confidence, 0.6)
		assert.Equal(t, "bandwidth", anomaly.MetricName)
	}
}

// TestMonitoringPipeline_ProcessMetrics tests the monitoring pipeline
func TestMonitoringPipeline_ProcessMetrics(t *testing.T) {
	logger := zap.NewNop()

	config := DefaultDetectorConfig()
	detector, err := NewAnomalyDetector(config, logger)
	require.NoError(t, err)

	alertConfig := DefaultAlertConfig()
	alertManager := NewAlertManager(alertConfig, logger)

	pipeline := NewMonitoringPipeline(
		detector,
		alertManager,
		1*time.Second,
		logger,
	)

	// Start pipeline
	err = pipeline.Start()
	require.NoError(t, err)
	defer pipeline.Stop()

	// Process some metrics
	for i := 0; i < 10; i++ {
		metric := &MetricVector{
			Timestamp:   time.Now(),
			Bandwidth:   100.0 + float64(i),
			Latency:     10.0,
			PacketLoss:  0.01,
			Jitter:      1.0,
			CPUUsage:    50.0,
			MemoryUsage: 60.0,
			ErrorRate:   0.001,
		}

		err := pipeline.ProcessMetrics(metric)
		require.NoError(t, err)
	}

	// Check stats
	stats := pipeline.GetStats()
	assert.Equal(t, int64(10), stats.MetricsProcessed)
}

// Helper functions

func generateNormalData(n int) []*MetricVector {
	data := make([]*MetricVector, n)
	for i := 0; i < n; i++ {
		data[i] = &MetricVector{
			Timestamp:   time.Now().Add(time.Duration(i) * time.Minute),
			Bandwidth:   100.0 + float64(i%10),
			Latency:     10.0 + float64(i%5)*0.5,
			PacketLoss:  0.01 + float64(i%3)*0.001,
			Jitter:      1.0 + float64(i%4)*0.1,
			CPUUsage:    50.0 + float64(i%8)*2,
			MemoryUsage: 60.0 + float64(i%6)*1,
			ErrorRate:   0.001 + float64(i%5)*0.0001,
		}
	}
	return data
}

func generateTimeSeriesData(n int) []*MetricVector {
	data := make([]*MetricVector, n)
	for i := 0; i < n; i++ {
		// Add gradual trend
		trend := float64(i) * 0.1
		data[i] = &MetricVector{
			Timestamp:   time.Now().Add(time.Duration(i) * time.Minute),
			Bandwidth:   100.0 + trend,
			Latency:     10.0 + trend*0.05,
			PacketLoss:  0.01,
			Jitter:      1.0,
			CPUUsage:    50.0 + trend*0.2,
			MemoryUsage: 60.0 + trend*0.1,
			ErrorRate:   0.001,
		}
	}
	return data
}

func generateSeasonalData(n, period int) []*MetricVector {
	data := make([]*MetricVector, n)
	for i := 0; i < n; i++ {
		// Add seasonal pattern
		seasonal := 10.0 * math.Sin(2*math.Pi*float64(i)/float64(period))
		data[i] = &MetricVector{
			Timestamp:   time.Now().Add(time.Duration(i) * time.Hour),
			Bandwidth:   100.0 + seasonal,
			Latency:     10.0 - seasonal*0.2,
			PacketLoss:  0.01,
			Jitter:      1.0,
			CPUUsage:    50.0 + seasonal*0.5,
			MemoryUsage: 60.0,
			ErrorRate:   0.001,
		}
	}
	return data
}

// Benchmarks

func BenchmarkIsolationForest_Detect(b *testing.B) {
	logger := zap.NewNop()
	detector, _ := NewIsolationForestModel("models/isolation_forest.onnx", logger)

	normalData := generateNormalData(1000)
	ctx := context.Background()
	detector.Train(ctx, normalData)

	metric := &MetricVector{
		Bandwidth:   105.0,
		Latency:     11.0,
		PacketLoss:  0.01,
		Jitter:      1.1,
		CPUUsage:    51.0,
		MemoryUsage: 61.0,
		ErrorRate:   0.001,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		detector.Detect(ctx, metric)
	}
}

func BenchmarkZScoreDetector_Detect(b *testing.B) {
	logger := zap.NewNop()
	detector := NewZScoreDetector(100, 3.0, logger)

	normalData := generateNormalData(100)
	ctx := context.Background()
	detector.Train(ctx, normalData)

	metric := &MetricVector{
		Bandwidth:   105.0,
		Latency:     11.0,
		PacketLoss:  0.01,
		Jitter:      1.1,
		CPUUsage:    51.0,
		MemoryUsage: 61.0,
		ErrorRate:   0.001,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		detector.Detect(ctx, metric)
	}
}
