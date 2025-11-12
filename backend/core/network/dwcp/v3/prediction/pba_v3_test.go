package prediction

import (
	"context"
	"testing"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/prediction"
	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/upgrade"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestPBAv3Initialization tests PBA v3 initialization
func TestPBAv3Initialization(t *testing.T) {
	// Skip if ONNX models not available
	t.Skip("Skipping test that requires ONNX models")

	config := DefaultPBAv3Config()
	pba, err := NewPBAv3(config)
	require.NoError(t, err)
	defer pba.Close()

	assert.NotNil(t, pba.datacenterPredictor)
	assert.NotNil(t, pba.internetPredictor)
	assert.NotNil(t, pba.modeDetector)
	assert.NotNil(t, pba.metrics)
}

// TestDefaultConfig tests default configuration
func TestDefaultConfig(t *testing.T) {
	config := DefaultPBAv3Config()

	assert.True(t, config.EnableHybridMode)
	assert.True(t, config.EnableModeDetection)
	assert.Equal(t, upgrade.ModeHybrid, config.DefaultMode)
	assert.Equal(t, 10, config.DatacenterSequenceLength)
	assert.Equal(t, 60, config.InternetSequenceLength)
	assert.Equal(t, 0.85, config.DatacenterAccuracyTarget)
	assert.Equal(t, 0.70, config.InternetAccuracyTarget)
	assert.Equal(t, 100*time.Millisecond, config.PredictionLatencyTarget)
}

// TestBandwidthHistory tests bandwidth history management
func TestBandwidthHistory(t *testing.T) {
	history := NewBandwidthHistory(10)

	// Test empty history
	assert.Equal(t, 0, history.Len())
	recent := history.GetRecent(5)
	assert.Nil(t, recent)

	// Add samples
	for i := 0; i < 15; i++ {
		sample := prediction.NetworkSample{
			Timestamp:     time.Now(),
			BandwidthMbps: float64(1000 + i*100),
			LatencyMs:     float64(5 + i),
			PacketLoss:    0.001,
			JitterMs:      1.0,
			TimeOfDay:     float64(i % 24),
		}
		history.Add(sample)
	}

	// Should only keep last 10
	assert.Equal(t, 10, history.Len())

	// Get recent samples
	recent = history.GetRecent(5)
	assert.Equal(t, 5, len(recent))

	// Get more than available
	recent = history.GetRecent(20)
	assert.Equal(t, 10, len(recent))
}

// TestModeSelection tests mode selection logic
func TestModeSelection(t *testing.T) {
	t.Skip("Skipping test that requires ONNX models")

	config := DefaultPBAv3Config()
	pba, err := NewPBAv3(config)
	require.NoError(t, err)
	defer pba.Close()

	// Default should be hybrid
	mode := pba.selectMode()
	assert.Equal(t, upgrade.ModeHybrid, mode)

	// Force datacenter mode
	pba.mu.Lock()
	pba.currentMode = upgrade.ModeDatacenter
	pba.mu.Unlock()

	mode = pba.selectMode()
	assert.Equal(t, upgrade.ModeDatacenter, mode)

	// Force internet mode
	pba.mu.Lock()
	pba.currentMode = upgrade.ModeInternet
	pba.mu.Unlock()

	mode = pba.selectMode()
	assert.Equal(t, upgrade.ModeInternet, mode)
}

// TestConfidenceAdjustment tests confidence adjustment for different modes
func TestConfidenceAdjustment(t *testing.T) {
	t.Skip("Skipping test that requires ONNX models")

	config := DefaultPBAv3Config()
	pba, err := NewPBAv3(config)
	require.NoError(t, err)
	defer pba.Close()

	tests := []struct {
		name            string
		mode            upgrade.NetworkMode
		inputConfidence float64
		minExpected     float64
		maxExpected     float64
	}{
		{
			name:            "datacenter high confidence",
			mode:            upgrade.ModeDatacenter,
			inputConfidence: 0.9,
			minExpected:     0.8,
			maxExpected:     1.0,
		},
		{
			name:            "datacenter low confidence",
			mode:            upgrade.ModeDatacenter,
			inputConfidence: 0.5,
			minExpected:     0.2,
			maxExpected:     0.6,
		},
		{
			name:            "internet high confidence",
			mode:            upgrade.ModeInternet,
			inputConfidence: 0.8,
			minExpected:     0.6,
			maxExpected:     1.0,
		},
		{
			name:            "internet low confidence",
			mode:            upgrade.ModeInternet,
			inputConfidence: 0.4,
			minExpected:     0.2,
			maxExpected:     0.5,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			adjusted := pba.adjustConfidenceForMode(tt.inputConfidence, tt.mode)
			assert.GreaterOrEqual(t, adjusted, tt.minExpected)
			assert.LessOrEqual(t, adjusted, tt.maxExpected)
		})
	}
}

// TestMetricsTracking tests metrics tracking
func TestMetricsTracking(t *testing.T) {
	t.Skip("Skipping test that requires ONNX models")

	config := DefaultPBAv3Config()
	pba, err := NewPBAv3(config)
	require.NoError(t, err)
	defer pba.Close()

	// Initial metrics
	metrics := pba.GetMetrics()
	assert.Equal(t, uint64(0), metrics.TotalPredictions)
	assert.Equal(t, uint64(0), metrics.DatacenterPredictions)
	assert.Equal(t, uint64(0), metrics.InternetPredictions)

	// Simulate predictions
	pred := &prediction.BandwidthPrediction{
		PredictedBandwidthMbps: 5000.0,
		Confidence:             0.9,
	}

	// Datacenter prediction
	pba.updateMetrics(upgrade.ModeDatacenter, 50*time.Millisecond, pred)
	metrics = pba.GetMetrics()
	assert.Equal(t, uint64(1), metrics.TotalPredictions)
	assert.Equal(t, uint64(1), metrics.DatacenterPredictions)
	assert.Equal(t, uint64(0), metrics.InternetPredictions)

	// Internet prediction
	pba.updateMetrics(upgrade.ModeInternet, 80*time.Millisecond, pred)
	metrics = pba.GetMetrics()
	assert.Equal(t, uint64(2), metrics.TotalPredictions)
	assert.Equal(t, uint64(1), metrics.DatacenterPredictions)
	assert.Equal(t, uint64(1), metrics.InternetPredictions)
}

// TestAddSample tests adding network samples
func TestAddSample(t *testing.T) {
	t.Skip("Skipping test that requires ONNX models")

	config := DefaultPBAv3Config()
	pba, err := NewPBAv3(config)
	require.NoError(t, err)
	defer pba.Close()

	// Add samples
	for i := 0; i < 100; i++ {
		sample := prediction.NetworkSample{
			Timestamp:     time.Now(),
			BandwidthMbps: float64(5000 + i*10),
			LatencyMs:     float64(5 + i%10),
			PacketLoss:    0.001,
			JitterMs:      1.0,
			TimeOfDay:     float64(i % 24),
			DayOfWeek:     float64(i % 7),
		}
		pba.AddSample(sample)
	}

	// Check history
	assert.Greater(t, pba.datacenterHistory.Len(), 0)
	assert.Greater(t, pba.internetHistory.Len(), 0)
}

// TestEnsemblePredictions tests ensemble prediction logic
func TestEnsemblePredictions(t *testing.T) {
	t.Skip("Skipping test that requires ONNX models")

	config := DefaultPBAv3Config()
	pba, err := NewPBAv3(config)
	require.NoError(t, err)
	defer pba.Close()

	dcPred := &prediction.BandwidthPrediction{
		PredictedBandwidthMbps: 8000.0,
		PredictedLatencyMs:     5.0,
		PredictedPacketLoss:    0.0001,
		PredictedJitterMs:      1.0,
		Confidence:             0.9,
	}

	inetPred := &prediction.BandwidthPrediction{
		PredictedBandwidthMbps: 400.0,
		PredictedLatencyMs:     100.0,
		PredictedPacketLoss:    0.01,
		PredictedJitterMs:      10.0,
		Confidence:             0.7,
	}

	ensemble := pba.ensemblePredictions(dcPred, inetPred)

	// Ensemble should be weighted average
	assert.Greater(t, ensemble.PredictedBandwidthMbps, inetPred.PredictedBandwidthMbps)
	assert.Less(t, ensemble.PredictedBandwidthMbps, dcPred.PredictedBandwidthMbps)
	assert.Greater(t, ensemble.PredictedLatencyMs, dcPred.PredictedLatencyMs)
	assert.Less(t, ensemble.PredictedLatencyMs, inetPred.PredictedLatencyMs)

	// Should use higher confidence
	assert.Equal(t, dcPred.Confidence, ensemble.Confidence)
}

// BenchmarkPrediction benchmarks prediction performance
func BenchmarkPrediction(b *testing.B) {
	b.Skip("Skipping benchmark that requires ONNX models")

	config := DefaultPBAv3Config()
	pba, err := NewPBAv3(config)
	if err != nil {
		b.Fatal(err)
	}
	defer pba.Close()

	// Generate sample data
	for i := 0; i < 100; i++ {
		sample := prediction.NetworkSample{
			Timestamp:     time.Now(),
			BandwidthMbps: 5000.0,
			LatencyMs:     5.0,
			PacketLoss:    0.001,
			JitterMs:      1.0,
			TimeOfDay:     float64(12),
			DayOfWeek:     float64(3),
		}
		pba.AddSample(sample)
	}

	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := pba.PredictBandwidth(ctx)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// TestPredictionLatency tests that predictions meet latency targets
func TestPredictionLatency(t *testing.T) {
	t.Skip("Skipping test that requires ONNX models")

	config := DefaultPBAv3Config()
	pba, err := NewPBAv3(config)
	require.NoError(t, err)
	defer pba.Close()

	// Add sufficient history
	for i := 0; i < 100; i++ {
		sample := prediction.NetworkSample{
			Timestamp:     time.Now(),
			BandwidthMbps: 5000.0,
			LatencyMs:     5.0,
			PacketLoss:    0.001,
			JitterMs:      1.0,
			TimeOfDay:     float64(12),
			DayOfWeek:     float64(3),
		}
		pba.AddSample(sample)
	}

	ctx := context.Background()

	// Test prediction latency
	iterations := 10
	var totalLatency time.Duration

	for i := 0; i < iterations; i++ {
		start := time.Now()
		_, err := pba.PredictBandwidth(ctx)
		latency := time.Since(start)
		totalLatency += latency

		if err != nil {
			t.Logf("Prediction %d failed: %v", i, err)
		}
	}

	avgLatency := totalLatency / time.Duration(iterations)
	t.Logf("Average prediction latency: %v", avgLatency)

	// Target: <100ms for datacenter, <150ms overall
	assert.Less(t, avgLatency, 150*time.Millisecond,
		"Average latency should be <150ms")
}
