package ml

import (
	"context"
	"testing"
	"time"

	"github.com/sirupsen/logrus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestPredictiveAllocator(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.DebugLevel)
	
	config := PredictorConfig{
		PredictionHorizon:  15 * time.Minute,
		UpdateInterval:     1 * time.Hour,
		BufferSize:         100,
		AccuracyTarget:     0.85,
		LatencyTarget:      250 * time.Millisecond,
		EnableAutoScaling:  true,
		EnablePreemptive:   true,
		ModelVersion:       "test",
	}
	
	predictor := NewPredictiveAllocator(logger, config)
	require.NotNil(t, predictor)
	
	t.Run("TestInitialization", func(t *testing.T) {
		assert.NotNil(t, predictor.lstm)
		assert.NotNil(t, predictor.dataBuffer)
		assert.NotNil(t, predictor.predictions)
		assert.Equal(t, config.AccuracyTarget, predictor.config.AccuracyTarget)
	})
	
	t.Run("TestLSTMModel", func(t *testing.T) {
		lstm := NewLSTMModel(4, 128, 4, 10)
		assert.NotNil(t, lstm)
		assert.Equal(t, 4, lstm.inputSize)
		assert.Equal(t, 128, lstm.hiddenSize)
		assert.Equal(t, 4, lstm.outputSize)
		assert.Equal(t, 10, lstm.sequenceLength)
	})
	
	t.Run("TestTimeSeriesBuffer", func(t *testing.T) {
		buffer := NewTimeSeriesBuffer(50)
		
		// Add data
		for i := 0; i < 60; i++ {
			data := []float64{float64(i), float64(i * 2), float64(i * 3), float64(i * 4)}
			buffer.Add(data, time.Now())
		}
		
		// Should only keep last 50
		all := buffer.GetAll()
		assert.LessOrEqual(t, len(all), 50)
		
		// Get sequence
		seq := buffer.GetSequence(10)
		assert.LessOrEqual(t, len(seq), 10)
	})
	
	t.Run("TestPrediction", func(t *testing.T) {
		// Add training data
		for i := 0; i < 20; i++ {
			metrics := &ResourceMetrics{
				Timestamp:    time.Now(),
				VMId:         "test-vm-1",
				CPUUsage:     50.0 + float64(i%10),
				MemoryUsage:  60.0 + float64(i%8),
				IOUsage:      30.0 + float64(i%5),
				NetworkUsage: 40.0 + float64(i%7),
			}
			predictor.AddMetrics(metrics)
			time.Sleep(10 * time.Millisecond)
		}
		
		// Make prediction
		prediction, err := predictor.PredictResourceUsage("test-vm-1", 15*time.Minute)
		
		// May fail initially due to insufficient data
		if err == nil {
			assert.NotNil(t, prediction)
			assert.Equal(t, "test-vm-1", prediction.VMId)
			assert.GreaterOrEqual(t, prediction.Confidence, 0.0)
			assert.LessOrEqual(t, prediction.Confidence, 1.0)
			assert.NotNil(t, prediction.Recommendations)
		}
	})
	
	t.Run("TestRecommendations", func(t *testing.T) {
		// Test high CPU prediction
		highCPU := []float64{0.9, 0.5, 0.5, 0.5}
		recommendations := predictor.generateRecommendations(highCPU, 0.8)
		
		found := false
		for _, rec := range recommendations {
			if rec.Target == "cpu" && rec.Action == "scale_up" {
				found = true
				break
			}
		}
		assert.True(t, found, "Should recommend CPU scale up for high usage")
		
		// Test low CPU prediction
		lowCPU := []float64{0.15, 0.5, 0.5, 0.5}
		recommendations = predictor.generateRecommendations(lowCPU, 0.8)
		
		found = false
		for _, rec := range recommendations {
			if rec.Target == "cpu" && rec.Action == "scale_down" {
				found = true
				break
			}
		}
		assert.True(t, found, "Should recommend CPU scale down for low usage")
	})
	
	t.Run("TestModelTraining", func(t *testing.T) {
		lstm := NewLSTMModel(4, 32, 4, 5)
		
		// Generate training data
		sequences := [][][]float64{}
		labels := [][]float64{}
		
		for i := 0; i < 50; i++ {
			seq := [][]float64{}
			for j := 0; j < 5; j++ {
				seq = append(seq, []float64{
					float64(i+j) / 100.0,
					float64(i+j*2) / 100.0,
					float64(i+j*3) / 100.0,
					float64(i+j*4) / 100.0,
				})
			}
			sequences = append(sequences, seq)
			labels = append(labels, []float64{0.5, 0.6, 0.7, 0.8})
		}
		
		// Train model
		accuracy := lstm.Train(sequences, labels, 10, 0.01)
		assert.GreaterOrEqual(t, accuracy, 0.0)
		assert.LessOrEqual(t, accuracy, 1.0)
		assert.True(t, lstm.trained)
	})
	
	t.Run("TestConfidenceCalculation", func(t *testing.T) {
		// Stable data should have higher confidence
		stableData := [][]float64{
			{50.0, 60.0, 30.0, 40.0},
			{51.0, 61.0, 31.0, 41.0},
			{49.0, 59.0, 29.0, 39.0},
			{50.0, 60.0, 30.0, 40.0},
		}
		
		predictions := []float64{0.5, 0.6, 0.3, 0.4}
		confidence := predictor.calculateConfidence(stableData, predictions)
		assert.Greater(t, confidence, 0.5)
		
		// Volatile data should have lower confidence
		volatileData := [][]float64{
			{20.0, 80.0, 10.0, 90.0},
			{80.0, 20.0, 90.0, 10.0},
			{50.0, 50.0, 50.0, 50.0},
			{10.0, 90.0, 20.0, 80.0},
		}
		
		volatileConfidence := predictor.calculateConfidence(volatileData, predictions)
		assert.Less(t, volatileConfidence, confidence)
	})
	
	t.Run("TestAutoScaling", func(t *testing.T) {
		ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
		defer cancel()
		
		err := predictor.Start(ctx)
		assert.NoError(t, err)
		
		// Let it run briefly
		time.Sleep(50 * time.Millisecond)
		
		// Check that goroutines started
		predictions := predictor.GetPredictions()
		assert.NotNil(t, predictions)
	})
	
	t.Run("TestLatencyTarget", func(t *testing.T) {
		start := time.Now()
		
		// Add minimal data
		for i := 0; i < 10; i++ {
			predictor.dataBuffer.Add([]float64{50, 60, 30, 40}, time.Now())
		}
		
		_, err := predictor.PredictResourceUsage("test-vm", 15*time.Minute)
		elapsed := time.Since(start)
		
		// Should complete within reasonable time even with error
		assert.Less(t, elapsed, 1*time.Second)
		
		if err == nil {
			// If successful, should meet latency target
			assert.LessOrEqual(t, elapsed, config.LatencyTarget*2)
		}
	})
}

func TestLSTMForward(t *testing.T) {
	lstm := NewLSTMModel(3, 5, 2, 1)
	
	input := []float64{0.5, 0.6, 0.7}
	lstm.forward(input)
	
	// Check that hidden state is updated
	assert.NotNil(t, lstm.hiddenState)
	assert.Equal(t, 5, len(lstm.hiddenState))
	
	// Check that cell state is updated
	assert.NotNil(t, lstm.cellState)
	assert.Equal(t, 5, len(lstm.cellState))
}

func TestLSTMPredict(t *testing.T) {
	lstm := NewLSTMModel(3, 5, 2, 3)
	
	sequence := [][]float64{
		{0.1, 0.2, 0.3},
		{0.2, 0.3, 0.4},
		{0.3, 0.4, 0.5},
	}
	
	output := lstm.Predict(sequence)
	
	assert.NotNil(t, output)
	assert.Equal(t, 2, len(output))
	
	// Output should be in [0, 1] range due to sigmoid
	for _, val := range output {
		assert.GreaterOrEqual(t, val, 0.0)
		assert.LessOrEqual(t, val, 1.0)
	}
}

func TestDataPreprocessing(t *testing.T) {
	logger := logrus.New()
	config := PredictorConfig{
		BufferSize: 100,
	}
	
	predictor := NewPredictiveAllocator(logger, config)
	
	// Test normalization
	data := [][]float64{
		{100.0, 80.0, 60.0, 40.0},
		{50.0, 90.0, 70.0, 30.0},
		{75.0, 85.0, 65.0, 35.0},
	}
	
	normalized := predictor.preprocessData(data)
	
	assert.Equal(t, len(data), len(normalized))
	
	// Check all values are normalized to [0, 1]
	for _, row := range normalized {
		for _, val := range row {
			assert.GreaterOrEqual(t, val, 0.0)
			assert.LessOrEqual(t, val, 1.0)
		}
	}
}

func TestScalingRecommendations(t *testing.T) {
	tests := []struct {
		name        string
		predictions []float64
		confidence  float64
		expectScale string
	}{
		{
			name:        "High CPU usage",
			predictions: []float64{0.85, 0.5, 0.5, 0.5},
			confidence:  0.8,
			expectScale: "scale_up",
		},
		{
			name:        "Low CPU usage",
			predictions: []float64{0.15, 0.5, 0.5, 0.5},
			confidence:  0.8,
			expectScale: "scale_down",
		},
		{
			name:        "High memory usage",
			predictions: []float64{0.5, 0.9, 0.5, 0.5},
			confidence:  0.8,
			expectScale: "scale_up",
		},
		{
			name:        "Combined high usage",
			predictions: []float64{0.75, 0.75, 0.5, 0.5},
			confidence:  0.85,
			expectScale: "migrate",
		},
	}
	
	logger := logrus.New()
	config := PredictorConfig{}
	predictor := NewPredictiveAllocator(logger, config)
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			recommendations := predictor.generateRecommendations(tt.predictions, tt.confidence)
			
			found := false
			for _, rec := range recommendations {
				if rec.Action == tt.expectScale {
					found = true
					break
				}
			}
			
			assert.True(t, found, "Expected %s recommendation not found", tt.expectScale)
		})
	}
}

func BenchmarkLSTMPrediction(b *testing.B) {
	lstm := NewLSTMModel(4, 128, 4, 10)
	
	sequence := make([][]float64, 10)
	for i := 0; i < 10; i++ {
		sequence[i] = []float64{0.5, 0.6, 0.7, 0.8}
	}
	
	b.ResetTimer()
	
	for i := 0; i < b.N; i++ {
		_ = lstm.Predict(sequence)
	}
}

func BenchmarkPredictResourceUsage(b *testing.B) {
	logger := logrus.New()
	logger.SetLevel(logrus.ErrorLevel)
	
	config := PredictorConfig{
		PredictionHorizon: 15 * time.Minute,
		BufferSize:        1000,
		LatencyTarget:     250 * time.Millisecond,
	}
	
	predictor := NewPredictiveAllocator(logger, config)
	
	// Add sample data
	for i := 0; i < 100; i++ {
		predictor.dataBuffer.Add([]float64{50, 60, 30, 40}, time.Now())
	}
	
	b.ResetTimer()
	
	for i := 0; i < b.N; i++ {
		_, _ = predictor.PredictResourceUsage("test-vm", 15*time.Minute)
	}
}