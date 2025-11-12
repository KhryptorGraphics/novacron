package benchmarks

import (
	"fmt"
	"math/rand"
	"sync/atomic"
	"testing"
	"time"
)

// BenchmarkPBALSTMPredictionLatency tests LSTM inference latency
func BenchmarkPBALSTMPredictionLatency(b *testing.B) {
	scenarios := []struct {
		name         string
		sequenceLen  int
		hiddenSize   int
		batchSize    int
	}{
		{"Small_Seq10_Hidden32_Batch1", 10, 32, 1},
		{"Medium_Seq50_Hidden64_Batch1", 50, 64, 1},
		{"Large_Seq100_Hidden128_Batch1", 100, 128, 1},
		{"Small_Seq10_Hidden32_Batch8", 10, 32, 8},
		{"Medium_Seq50_Hidden64_Batch8", 50, 64, 8},
		{"Large_Seq100_Hidden128_Batch8", 100, 128, 8},
	}

	for _, sc := range scenarios {
		b.Run(sc.name, func(b *testing.B) {
			benchmarkLSTMLatency(b, sc.sequenceLen, sc.hiddenSize, sc.batchSize)
		})
	}
}

func benchmarkLSTMLatency(b *testing.B, sequenceLen, hiddenSize, batchSize int) {
	b.ReportAllocs()

	// Initialize mock LSTM model
	model := newMockLSTM(sequenceLen, hiddenSize)

	// Generate input sequences
	inputs := make([][][]float32, batchSize)
	for i := 0; i < batchSize; i++ {
		inputs[i] = generateSequence(sequenceLen, 10) // 10 features
	}

	var totalLatency time.Duration

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		start := time.Now()

		// Run inference
		_ = model.predict(inputs)

		totalLatency += time.Since(start)
	}

	b.StopTimer()

	avgLatencyMs := float64(totalLatency.Microseconds()) / float64(b.N) / 1000.0
	predictionsPerSecond := float64(b.N*batchSize) / b.Elapsed().Seconds()

	b.ReportMetric(avgLatencyMs, "ms/prediction")
	b.ReportMetric(predictionsPerSecond, "predictions/sec")
}

// BenchmarkPBAPredictionAccuracy tests prediction accuracy over time
func BenchmarkPBAPredictionAccuracy(b *testing.B) {
	scenarios := []struct {
		name     string
		pattern  string
		horizon  int
	}{
		{"Stable_1min", "stable", 60},
		{"Linear_1min", "linear", 60},
		{"Periodic_1min", "periodic", 60},
		{"Volatile_1min", "volatile", 60},
		{"Stable_5min", "stable", 300},
		{"Linear_5min", "linear", 300},
	}

	for _, sc := range scenarios {
		b.Run(sc.name, func(b *testing.B) {
			benchmarkPredictionAccuracy(b, sc.pattern, sc.horizon)
		})
	}
}

func benchmarkPredictionAccuracy(b *testing.B, pattern string, horizon int) {
	b.ReportAllocs()

	model := newMockLSTM(100, 64)

	var totalError float64
	var predictions int64

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		// Generate historical data
		history := generateBandwidthPattern(pattern, 100)

		// Generate actual future values
		actual := generateBandwidthPattern(pattern, horizon)

		// Make predictions
		predicted := model.predictBandwidth(history, horizon)

		// Calculate error
		for j := 0; j < len(predicted) && j < len(actual); j++ {
			error := abs(predicted[j] - actual[j]) / actual[j]
			totalError += error
			atomic.AddInt64(&predictions, 1)
		}
	}

	b.StopTimer()

	avgError := totalError / float64(predictions) * 100
	accuracy := 100 - avgError

	b.ReportMetric(accuracy, "accuracy_%")
	b.ReportMetric(avgError, "error_%")
}

// BenchmarkPBAModelInferenceThroughput tests throughput of model inference
func BenchmarkPBAModelInferenceThroughput(b *testing.B) {
	batchSizes := []int{1, 4, 8, 16, 32, 64}

	for _, batchSize := range batchSizes {
		b.Run(fmt.Sprintf("Batch%d", batchSize), func(b *testing.B) {
			benchmarkInferenceThroughput(b, batchSize)
		})
	}
}

func benchmarkInferenceThroughput(b *testing.B, batchSize int) {
	b.ReportAllocs()

	model := newMockLSTM(50, 64)
	inputs := make([][][]float32, batchSize)
	for i := 0; i < batchSize; i++ {
		inputs[i] = generateSequence(50, 10)
	}

	var totalPredictions int64
	startTime := time.Now()

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = model.predict(inputs)
		atomic.AddInt64(&totalPredictions, int64(batchSize))
	}

	b.StopTimer()

	duration := time.Since(startTime)
	predictionsPerSecond := float64(totalPredictions) / duration.Seconds()

	b.ReportMetric(predictionsPerSecond, "predictions/sec")
	b.ReportMetric(float64(batchSize), "batch_size")
}

// BenchmarkPBABandwidthForecast tests bandwidth forecasting performance
func BenchmarkPBABandwidthForecast(b *testing.B) {
	scenarios := []struct {
		name          string
		historyLength int
		forecastSteps int
	}{
		{"Short_History100_Forecast10", 100, 10},
		{"Short_History100_Forecast60", 100, 60},
		{"Medium_History500_Forecast10", 500, 10},
		{"Medium_History500_Forecast60", 500, 60},
		{"Long_History1000_Forecast10", 1000, 10},
		{"Long_History1000_Forecast60", 1000, 60},
	}

	for _, sc := range scenarios {
		b.Run(sc.name, func(b *testing.B) {
			benchmarkBandwidthForecast(b, sc.historyLength, sc.forecastSteps)
		})
	}
}

func benchmarkBandwidthForecast(b *testing.B, historyLength, forecastSteps int) {
	b.ReportAllocs()

	model := newMockLSTM(historyLength, 64)
	history := generateBandwidthPattern("periodic", historyLength)

	var totalLatency time.Duration

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		start := time.Now()

		_ = model.predictBandwidth(history, forecastSteps)

		totalLatency += time.Since(start)
	}

	b.StopTimer()

	avgLatencyMs := float64(totalLatency.Microseconds()) / float64(b.N) / 1000.0
	forecastsPerSecond := float64(b.N) / b.Elapsed().Seconds()

	b.ReportMetric(avgLatencyMs, "ms/forecast")
	b.ReportMetric(forecastsPerSecond, "forecasts/sec")
}

// BenchmarkPBAAdaptiveWindowSizing tests adaptive window size adjustment
func BenchmarkPBAAdaptiveWindowSizing(b *testing.B) {
	scenarios := []struct {
		name      string
		volatility float64
	}{
		{"Low_Volatility", 0.1},
		{"Medium_Volatility", 0.3},
		{"High_Volatility", 0.6},
		{"Extreme_Volatility", 0.9},
	}

	for _, sc := range scenarios {
		b.Run(sc.name, func(b *testing.B) {
			benchmarkAdaptiveWindowSizing(b, sc.volatility)
		})
	}
}

func benchmarkAdaptiveWindowSizing(b *testing.B, volatility float64) {
	b.ReportAllocs()

	var totalAdjustments int64
	var totalLatency time.Duration

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		start := time.Now()

		// Generate bandwidth measurements with volatility
		measurements := make([]float64, 100)
		for j := range measurements {
			measurements[j] = 1000.0 + rand.Float64()*volatility*1000.0
		}

		// Calculate variance
		variance := calculateVariance(measurements)

		// Adjust window size based on variance
		windowSize := adjustWindowSize(variance)
		_ = windowSize

		atomic.AddInt64(&totalAdjustments, 1)
		totalLatency += time.Since(start)
	}

	b.StopTimer()

	avgLatencyUs := float64(totalLatency.Microseconds()) / float64(b.N)
	adjustmentsPerSecond := float64(totalAdjustments) / b.Elapsed().Seconds()

	b.ReportMetric(avgLatencyUs, "us/adjustment")
	b.ReportMetric(adjustmentsPerSecond, "adjustments/sec")
}

// BenchmarkPBAFeatureExtraction tests feature extraction from metrics
func BenchmarkPBAFeatureExtraction(b *testing.B) {
	metricCounts := []int{10, 50, 100, 500, 1000}

	for _, count := range metricCounts {
		b.Run(fmt.Sprintf("%dMetrics", count), func(b *testing.B) {
			benchmarkFeatureExtraction(b, count)
		})
	}
}

func benchmarkFeatureExtraction(b *testing.B, metricCount int) {
	b.ReportAllocs()

	// Generate sample metrics
	metrics := make([][]float64, metricCount)
	for i := range metrics {
		metrics[i] = generateBandwidthPattern("periodic", 100)
	}

	var totalFeatures int64

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		features := extractFeatures(metrics)
		atomic.AddInt64(&totalFeatures, int64(len(features)))
	}

	b.StopTimer()

	featuresPerSecond := float64(totalFeatures) / b.Elapsed().Seconds()

	b.ReportMetric(featuresPerSecond, "features/sec")
	b.ReportMetric(float64(metricCount), "metrics")
}

// BenchmarkPBAModelUpdate tests online model update performance
func BenchmarkPBAModelUpdate(b *testing.B) {
	updateSizes := []int{1, 10, 100, 1000}

	for _, size := range updateSizes {
		b.Run(fmt.Sprintf("%dSamples", size), func(b *testing.B) {
			benchmarkModelUpdate(b, size)
		})
	}
}

func benchmarkModelUpdate(b *testing.B, updateSize int) {
	b.ReportAllocs()

	model := newMockLSTM(50, 64)

	// Generate update samples
	samples := make([][][]float32, updateSize)
	for i := range samples {
		samples[i] = generateSequence(50, 10)
	}

	var totalUpdates int64

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		model.update(samples)
		atomic.AddInt64(&totalUpdates, 1)
	}

	b.StopTimer()

	updatesPerSecond := float64(totalUpdates) / b.Elapsed().Seconds()
	samplesPerSecond := float64(totalUpdates*int64(updateSize)) / b.Elapsed().Seconds()

	b.ReportMetric(updatesPerSecond, "updates/sec")
	b.ReportMetric(samplesPerSecond, "samples/sec")
}

// Helper types and functions

type mockLSTM struct {
	sequenceLen int
	hiddenSize  int
	weights     [][]float32
}

func newMockLSTM(sequenceLen, hiddenSize int) *mockLSTM {
	weights := make([][]float32, hiddenSize)
	for i := range weights {
		weights[i] = make([]float32, sequenceLen)
		for j := range weights[i] {
			weights[i][j] = rand.Float32()
		}
	}

	return &mockLSTM{
		sequenceLen: sequenceLen,
		hiddenSize:  hiddenSize,
		weights:     weights,
	}
}

func (m *mockLSTM) predict(inputs [][][]float32) [][]float32 {
	results := make([][]float32, len(inputs))
	for i := range inputs {
		results[i] = make([]float32, m.hiddenSize)
		for j := range results[i] {
			results[i][j] = rand.Float32()
		}
	}
	return results
}

func (m *mockLSTM) predictBandwidth(history []float64, steps int) []float64 {
	predictions := make([]float64, steps)
	for i := range predictions {
		predictions[i] = history[len(history)-1] + rand.Float64()*100 - 50
	}
	return predictions
}

func (m *mockLSTM) update(samples [][][]float32) {
	// Simulate weight update
	for i := range m.weights {
		for j := range m.weights[i] {
			m.weights[i][j] += rand.Float32() * 0.01
		}
	}
}

func generateSequence(length, features int) [][]float32 {
	seq := make([][]float32, length)
	for i := range seq {
		seq[i] = make([]float32, features)
		for j := range seq[i] {
			seq[i][j] = rand.Float32()
		}
	}
	return seq
}

func generateBandwidthPattern(pattern string, length int) []float64 {
	values := make([]float64, length)

	switch pattern {
	case "stable":
		for i := range values {
			values[i] = 1000.0 + rand.Float64()*10
		}
	case "linear":
		for i := range values {
			values[i] = 1000.0 + float64(i)*2
		}
	case "periodic":
		for i := range values {
			values[i] = 1000.0 + 200*sinApprox(float64(i)/10)
		}
	case "volatile":
		for i := range values {
			values[i] = 1000.0 + rand.Float64()*500
		}
	}

	return values
}

func calculateVariance(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}

	mean := 0.0
	for _, v := range values {
		mean += v
	}
	mean /= float64(len(values))

	variance := 0.0
	for _, v := range values {
		diff := v - mean
		variance += diff * diff
	}
	variance /= float64(len(values))

	return variance
}

func adjustWindowSize(variance float64) int {
	// Larger window for stable bandwidth, smaller for volatile
	if variance < 100 {
		return 100
	} else if variance < 1000 {
		return 50
	}
	return 20
}

func extractFeatures(metrics [][]float64) []float64 {
	features := make([]float64, len(metrics)*5)

	for i, metric := range metrics {
		if len(metric) == 0 {
			continue
		}

		// Extract basic statistical features
		mean, min, max := 0.0, metric[0], metric[0]
		for _, v := range metric {
			mean += v
			if v < min {
				min = v
			}
			if v > max {
				max = v
			}
		}
		mean /= float64(len(metric))

		variance := calculateVariance(metric)

		features[i*5] = mean
		features[i*5+1] = min
		features[i*5+2] = max
		features[i*5+3] = variance
		features[i*5+4] = max - min // range
	}

	return features
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

func sinApprox(x float64) float64 {
	// Fast sine approximation
	const pi = 3.14159265359
	x = x - float64(int(x/(2*pi)))*(2*pi)

	if x < 0 {
		x = pi*2 + x
	}

	if x > pi {
		return -(x - pi)
	}
	return x
}
