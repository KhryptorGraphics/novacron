package prediction

import (
	"context"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDataCollector(t *testing.T) {
	t.Run("CreateCollector", func(t *testing.T) {
		collector := NewDataCollector(1*time.Second, 1000)
		require.NotNil(t, collector)
		assert.Equal(t, 1000, collector.maxSamples)
	})

	t.Run("CreateCollectorNormalizesNonPositiveMaxSamples", func(t *testing.T) {
		require.NotPanics(t, func() {
			collector := NewDataCollector(1*time.Second, -1)
			require.NotNil(t, collector)
			assert.Equal(t, 1, collector.maxSamples)
			collector.addSample(NetworkSample{Timestamp: time.Now(), BandwidthMbps: 100, LatencyMs: 20})
			assert.Len(t, collector.GetRecentSamples(10), 1)
		})
	})

	t.Run("CreateCollectorNormalizesNonPositiveInterval", func(t *testing.T) {
		collector := NewDataCollector(0, 100)
		require.NotNil(t, collector)
		assert.Equal(t, time.Second, collector.collectInterval)
		require.NotPanics(t, func() {
			collector.Start()
			collector.Stop()
		})
	})

	t.Run("CollectSamples", func(t *testing.T) {
		collector := NewDataCollector(100*time.Millisecond, 100)
		collector.Start()
		defer collector.Stop()

		// Wait for some samples
		time.Sleep(500 * time.Millisecond)

		samples := collector.GetRecentSamples(10)
		assert.NotEmpty(t, samples)
		assert.LessOrEqual(t, len(samples), 10)
	})

	t.Run("GetStatistics", func(t *testing.T) {
		collector := NewDataCollector(100*time.Millisecond, 100)

		// Manually add some samples
		for i := 0; i < 10; i++ {
			sample := NetworkSample{
				Timestamp:     time.Now(),
				BandwidthMbps: 100.0 + float64(i),
				LatencyMs:     20.0 + float64(i),
				PacketLoss:    0.01,
				JitterMs:      2.0,
				TimeOfDay:     12,
				DayOfWeek:     1,
			}
			collector.addSample(sample)
		}

		stats := collector.GetStatistics()
		assert.Equal(t, 10, stats.SampleCount)
		assert.Greater(t, stats.AvgBandwidth, 0.0)
		assert.Greater(t, stats.MaxBandwidth, stats.MinBandwidth)
	})

	t.Run("TimeRangeQuery", func(t *testing.T) {
		collector := NewDataCollector(1*time.Second, 100)

		start := time.Now()
		for i := 0; i < 5; i++ {
			sample := NetworkSample{
				Timestamp:     start.Add(time.Duration(i) * time.Minute),
				BandwidthMbps: 100.0,
				LatencyMs:     20.0,
			}
			collector.addSample(sample)
		}

		// Query middle range
		rangeStart := start.Add(1 * time.Minute)
		rangeEnd := start.Add(4 * time.Minute)
		samples := collector.GetSamplesByTimeRange(rangeStart, rangeEnd)

		assert.Equal(t, 2, len(samples)) // Should get samples at 2 and 3 minutes
	})

	t.Run("RestartAfterStopCollectsSamples", func(t *testing.T) {
		collector := NewDataCollector(10*time.Millisecond, 100)
		collector.dataPath = filepath.Join(t.TempDir(), "samples.jsonl")

		collector.Start()
		time.Sleep(35 * time.Millisecond)
		collector.Stop()

		collector.mu.Lock()
		collector.samples = nil
		collector.mu.Unlock()
		require.NoError(t, os.Remove(collector.dataPath))

		collector.Start()
		time.Sleep(35 * time.Millisecond)
		defer collector.Stop()

		assert.NotEmpty(t, collector.GetRecentSamples(10))
	})

	t.Run("SaveDataCreatesConfiguredDirectory", func(t *testing.T) {
		collector := NewDataCollector(1*time.Second, 100)
		collector.dataPath = filepath.Join(t.TempDir(), "nested", "samples.jsonl")
		collector.addSample(NetworkSample{
			Timestamp:     time.Now(),
			BandwidthMbps: 100,
			LatencyMs:     20,
			PacketLoss:    0.01,
			JitterMs:      2,
		})

		require.NoError(t, collector.saveData())
		require.FileExists(t, collector.dataPath)
	})

	t.Run("GetRecentSamplesRejectsNegativeCount", func(t *testing.T) {
		collector := NewDataCollector(1*time.Second, 100)
		collector.addSample(NetworkSample{
			Timestamp:     time.Now(),
			BandwidthMbps: 100,
			LatencyMs:     20,
		})

		require.NotPanics(t, func() {
			assert.Empty(t, collector.GetRecentSamples(-1))
		})
	})

	t.Run("ExportForTrainingCreatesConfiguredDirectory", func(t *testing.T) {
		collector := NewDataCollector(1*time.Second, 100)
		outputPath := filepath.Join(t.TempDir(), "nested", "training.csv")
		collector.addSample(NetworkSample{
			Timestamp:     time.Now(),
			BandwidthMbps: 100,
			LatencyMs:     20,
			PacketLoss:    0.01,
			JitterMs:      2,
			TimeOfDay:     12,
			DayOfWeek:     1,
		})

		require.NoError(t, collector.ExportForTraining(outputPath))
		require.FileExists(t, outputPath)
	})

	t.Run("LoadHistoricalDataSkipsMalformedJSONLLine", func(t *testing.T) {
		collector := NewDataCollector(1*time.Second, 100)
		collector.dataPath = filepath.Join(t.TempDir(), "samples.jsonl")

		first := NetworkSample{Timestamp: time.Now(), BandwidthMbps: 100, LatencyMs: 20}
		second := NetworkSample{Timestamp: time.Now(), BandwidthMbps: 200, LatencyMs: 30}
		firstJSON, err := json.Marshal(first)
		require.NoError(t, err)
		secondJSON, err := json.Marshal(second)
		require.NoError(t, err)
		require.NoError(t, os.WriteFile(
			collector.dataPath,
			[]byte(string(firstJSON)+"\nnot-json\n"+string(secondJSON)+"\n"),
			0644,
		))

		done := make(chan error, 1)
		go func() {
			done <- collector.loadHistoricalData()
		}()

		select {
		case err := <-done:
			require.NoError(t, err)
		case <-time.After(time.Second):
			t.Fatal("loadHistoricalData did not return after malformed line")
		}

		samples := collector.GetRecentSamples(10)
		require.Len(t, samples, 2)
		assert.Equal(t, 100.0, samples[0].BandwidthMbps)
		assert.Equal(t, 200.0, samples[1].BandwidthMbps)
	})
}

func TestPredictionService(t *testing.T) {
	t.Run("MultipleServicesReuseMetrics", func(t *testing.T) {
		var first *PredictionService
		var second *PredictionService

		require.NotPanics(t, func() {
			var err error
			first, err = NewPredictionService("", 10*time.Millisecond)
			require.NoError(t, err)
			second, err = NewPredictionService("", 10*time.Millisecond)
			require.NoError(t, err)
		})

		require.NotNil(t, first)
		require.NotNil(t, second)
		require.NoError(t, first.Stop())
		require.NoError(t, second.Stop())
	})

	t.Run("StartStopLifecycle", func(t *testing.T) {
		service, err := NewPredictionService("", 10*time.Millisecond)
		require.NoError(t, err)
		require.NotNil(t, service)

		assert.False(t, service.IsRunning())
		require.NoError(t, service.Start(context.Background()))
		assert.True(t, service.IsRunning())
		require.NoError(t, service.Stop())
		assert.False(t, service.IsRunning())
	})

	t.Run("CreateServiceNormalizesNonPositiveUpdateInterval", func(t *testing.T) {
		service, err := NewPredictionService("", 0)
		require.NoError(t, err)
		require.NotNil(t, service)
		defer service.Stop()

		assert.Equal(t, time.Second, service.updateInterval)
	})

	t.Run("UpdateAccuracyIgnoresInvalidActualTelemetry", func(t *testing.T) {
		service, err := NewPredictionService("", 10*time.Millisecond)
		require.NoError(t, err)
		defer service.Stop()

		predictedAt := time.Now().Add(-30 * time.Second)
		service.predictionHistory = []PredictionHistoryEntry{
			{
				Prediction: BandwidthPrediction{
					PredictedBandwidthMbps: 100,
					PredictedLatencyMs:     20,
				},
				Timestamp: predictedAt,
			},
		}
		service.collector.addSample(NetworkSample{
			Timestamp:     predictedAt.Add(time.Second),
			BandwidthMbps: 0,
			LatencyMs:     0,
		})

		service.updateAccuracy()

		outputPath := filepath.Join(t.TempDir(), "nested", "metrics.json")
		require.NoError(t, service.ExportMetrics(outputPath))
		require.FileExists(t, outputPath)
		raw, err := os.ReadFile(outputPath)
		require.NoError(t, err)

		var metrics map[string]interface{}
		require.NoError(t, json.Unmarshal(raw, &metrics))
		if averageError, ok := metrics["average_error"].(float64); ok {
			assert.False(t, math.IsNaN(averageError))
			assert.False(t, math.IsInf(averageError, 0))
		}
		if accuracy, ok := metrics["accuracy"].(float64); ok {
			assert.False(t, math.IsNaN(accuracy))
			assert.False(t, math.IsInf(accuracy, 0))
		}
	})

	t.Run("OptimalStreamCountKeepsMinimumAfterPacketLossAdjustment", func(t *testing.T) {
		service, err := NewPredictionService("", 10*time.Millisecond)
		require.NoError(t, err)
		defer service.Stop()

		service.mu.Lock()
		service.currentPrediction = &BandwidthPrediction{
			PredictedBandwidthMbps: 20.0,
			PredictedLatencyMs:     0.0,
			PredictedPacketLoss:    0.05,
			Confidence:             0.85,
			ValidUntil:             time.Now().Add(15 * time.Minute),
		}
		service.mu.Unlock()

		assert.Equal(t, 2, service.GetOptimalStreamCount())
	})

	t.Run("OptimalStreamCountFallsBackForInvalidPrediction", func(t *testing.T) {
		service, err := NewPredictionService("", 10*time.Millisecond)
		require.NoError(t, err)
		defer service.Stop()

		service.mu.Lock()
		service.currentPrediction = &BandwidthPrediction{
			PredictedBandwidthMbps: math.NaN(),
			PredictedLatencyMs:     math.Inf(1),
			PredictedPacketLoss:    -0.1,
			Confidence:             0.85,
			ValidUntil:             time.Now().Add(15 * time.Minute),
		}
		service.mu.Unlock()

		assert.Equal(t, 4, service.GetOptimalStreamCount())
	})

	t.Run("OptimalBufferSizeFallsBackForInvalidPrediction", func(t *testing.T) {
		service, err := NewPredictionService("", 10*time.Millisecond)
		require.NoError(t, err)
		defer service.Stop()

		service.mu.Lock()
		service.currentPrediction = &BandwidthPrediction{
			PredictedBandwidthMbps: math.NaN(),
			PredictedLatencyMs:     math.Inf(1),
			PredictedJitterMs:      -5,
			Confidence:             0.85,
		}
		service.mu.Unlock()

		assert.Equal(t, 65536, service.GetOptimalBufferSize())
	})

	t.Run("OptimalBufferSizeClampsHugeFinitePrediction", func(t *testing.T) {
		service, err := NewPredictionService("", 10*time.Millisecond)
		require.NoError(t, err)
		defer service.Stop()

		service.mu.Lock()
		service.currentPrediction = &BandwidthPrediction{
			PredictedBandwidthMbps: math.MaxFloat64,
			PredictedLatencyMs:     1000,
			PredictedJitterMs:      1000,
			Confidence:             0.85,
		}
		service.mu.Unlock()

		assert.Equal(t, 1048576, service.GetOptimalBufferSize())
	})

	t.Run("CreateService", func(t *testing.T) {
		service, err := NewPredictionService("", 1*time.Minute)
		require.NoError(t, err)
		require.NotNil(t, service)
		defer service.Stop()
	})

	t.Run("OptimalStreamCount", func(t *testing.T) {
		service, err := NewPredictionService("", 1*time.Minute)
		require.NoError(t, err)
		defer service.Stop()

		// Manually set a prediction for testing
		service.mu.Lock()
		service.currentPrediction = &BandwidthPrediction{
			PredictedBandwidthMbps: 200.0,
			PredictedLatencyMs:     30.0,
			PredictedPacketLoss:    0.01,
			Confidence:             0.85,
			ValidUntil:             time.Now().Add(15 * time.Minute),
		}
		service.mu.Unlock()

		streamCount := service.GetOptimalStreamCount()
		assert.Greater(t, streamCount, 0)
		assert.LessOrEqual(t, streamCount, 16)
	})

	t.Run("OptimalBufferSize", func(t *testing.T) {
		service, err := NewPredictionService("", 1*time.Minute)
		require.NoError(t, err)
		defer service.Stop()

		service.mu.Lock()
		service.currentPrediction = &BandwidthPrediction{
			PredictedBandwidthMbps: 100.0,
			PredictedLatencyMs:     50.0,
			PredictedJitterMs:      5.0,
			Confidence:             0.8,
		}
		service.mu.Unlock()

		bufferSize := service.GetOptimalBufferSize()
		assert.GreaterOrEqual(t, bufferSize, 16384) // Min 16KB
		assert.LessOrEqual(t, bufferSize, 1048576)  // Max 1MB
	})
}

func TestAMSTOptimizer(t *testing.T) {
	t.Run("CreateOptimizer", func(t *testing.T) {
		t.Skip("Requires prediction service")

		// Would need a mock prediction service
		// optimizer := NewAMSTOptimizer(mockService, logger)
		// assert.NotNil(t, optimizer)
	})

	t.Run("CalculateOptimalParameters", func(t *testing.T) {
		t.Skip("Requires prediction service")

		// Test parameter calculation with various predictions
		// predictions := []BandwidthPrediction{
		//     {PredictedBandwidthMbps: 100, PredictedLatencyMs: 20},
		//     {PredictedBandwidthMbps: 500, PredictedLatencyMs: 100},
		//     {PredictedBandwidthMbps: 50, PredictedLatencyMs: 10, PredictedPacketLoss: 0.05},
		// }
	})

	t.Run("PreemptiveOptimization", func(t *testing.T) {
		t.Skip("Requires prediction service")

		// Test that preemptive optimization only triggers with high confidence
		// and significant parameter changes
	})
}

func TestNetworkSample(t *testing.T) {
	t.Run("CreateSample", func(t *testing.T) {
		sample := NetworkSample{
			Timestamp:     time.Now(),
			BandwidthMbps: 100.0,
			LatencyMs:     20.0,
			PacketLoss:    0.01,
			JitterMs:      2.0,
			TimeOfDay:     14,
			DayOfWeek:     3,
			NodeID:        "node1",
			PeerID:        "peer1",
		}

		assert.Greater(t, sample.BandwidthMbps, 0.0)
		assert.Greater(t, sample.LatencyMs, 0.0)
		assert.GreaterOrEqual(t, sample.PacketLoss, 0.0)
		assert.LessOrEqual(t, sample.PacketLoss, 1.0)
		assert.GreaterOrEqual(t, sample.TimeOfDay, 0)
		assert.LessOrEqual(t, sample.TimeOfDay, 23)
		assert.GreaterOrEqual(t, sample.DayOfWeek, 0)
		assert.LessOrEqual(t, sample.DayOfWeek, 6)
	})
}

func TestBandwidthPrediction(t *testing.T) {
	t.Run("CreatePrediction", func(t *testing.T) {
		prediction := BandwidthPrediction{
			PredictedBandwidthMbps: 150.0,
			PredictedLatencyMs:     25.0,
			PredictedPacketLoss:    0.005,
			PredictedJitterMs:      3.0,
			Confidence:             0.85,
			ValidUntil:             time.Now().Add(15 * time.Minute),
			ModelVersion:           "v1",
			PredictionTime:         time.Now(),
		}

		assert.Greater(t, prediction.PredictedBandwidthMbps, 0.0)
		assert.Greater(t, prediction.Confidence, 0.0)
		assert.LessOrEqual(t, prediction.Confidence, 1.0)
		assert.True(t, prediction.ValidUntil.After(time.Now()))
	})

	t.Run("PredictionExpiry", func(t *testing.T) {
		prediction := BandwidthPrediction{
			ValidUntil: time.Now().Add(-1 * time.Minute), // Expired
		}

		assert.True(t, time.Now().After(prediction.ValidUntil))
	})
}

// Benchmarks

func BenchmarkDataCollection(b *testing.B) {
	collector := NewDataCollector(1*time.Second, 10000)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sample := collector.collectSample()
		collector.addSample(sample)
	}
}

func BenchmarkGetRecentSamples(b *testing.B) {
	collector := NewDataCollector(1*time.Second, 10000)

	// Add 1000 samples
	for i := 0; i < 1000; i++ {
		sample := NetworkSample{
			Timestamp:     time.Now(),
			BandwidthMbps: 100.0,
			LatencyMs:     20.0,
		}
		collector.addSample(sample)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = collector.GetRecentSamples(10)
	}
}

func BenchmarkCalculateStatistics(b *testing.B) {
	collector := NewDataCollector(1*time.Second, 10000)

	// Add 1000 samples
	for i := 0; i < 1000; i++ {
		sample := NetworkSample{
			Timestamp:     time.Now(),
			BandwidthMbps: 100.0 + float64(i),
			LatencyMs:     20.0 + float64(i%100),
		}
		collector.addSample(sample)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = collector.GetStatistics()
	}
}

// Integration tests

func TestIntegrationPredictionPipeline(t *testing.T) {
	t.Run("EndToEndPrediction", func(t *testing.T) {
		t.Skip("Requires full setup with ONNX model")

		// This would test the full pipeline:
		// 1. Data collector gathers samples
		// 2. Prediction service uses LSTM to predict
		// 3. AMST optimizer calculates parameters
		// 4. Parameters are applied to transport
	})
}

// Mock helpers for testing

type MockPredictor struct {
	predictions []BandwidthPrediction
	callCount   int
}

func (m *MockPredictor) Predict(history []NetworkSample) (*BandwidthPrediction, error) {
	if m.callCount >= len(m.predictions) {
		m.callCount = 0
	}
	pred := m.predictions[m.callCount]
	m.callCount++
	return &pred, nil
}

func TestMockPredictor(t *testing.T) {
	t.Run("MockPredictions", func(t *testing.T) {
		mock := &MockPredictor{
			predictions: []BandwidthPrediction{
				{PredictedBandwidthMbps: 100, Confidence: 0.9},
				{PredictedBandwidthMbps: 200, Confidence: 0.85},
			},
		}

		history := make([]NetworkSample, 10)
		pred1, err := mock.Predict(history)
		require.NoError(t, err)
		assert.Equal(t, 100.0, pred1.PredictedBandwidthMbps)

		pred2, err := mock.Predict(history)
		require.NoError(t, err)
		assert.Equal(t, 200.0, pred2.PredictedBandwidthMbps)

		// Should cycle back
		pred3, err := mock.Predict(history)
		require.NoError(t, err)
		assert.Equal(t, 100.0, pred3.PredictedBandwidthMbps)
	})
}
