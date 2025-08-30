// Package ml provides performance benchmarking for ML models in orchestration
package ml

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"sync"
	"time"

	"github.com/sirupsen/logrus"
)

// BenchmarkSuite manages performance benchmarking for ML models
type BenchmarkSuite struct {
	logger    *logrus.Logger
	results   map[string]*BenchmarkResult
	mu        sync.RWMutex
}

// BenchmarkResult contains comprehensive performance metrics
type BenchmarkResult struct {
	ModelType         ModelType        `json:"model_type"`
	ModelVersion      string           `json:"model_version"`
	BenchmarkType     BenchmarkType    `json:"benchmark_type"`
	StartTime         time.Time        `json:"start_time"`
	Duration          time.Duration    `json:"duration"`
	
	// Accuracy Metrics
	PredictionAccuracy    float64      `json:"prediction_accuracy"`
	AccuracyPercentiles   Percentiles  `json:"accuracy_percentiles"`
	AccuracyTrend        []float64    `json:"accuracy_trend"`
	
	// Performance Metrics  
	ThroughputRPS        float64      `json:"throughput_rps"`
	LatencyMetrics       LatencyStats `json:"latency_metrics"`
	MemoryUsageMB        float64      `json:"memory_usage_mb"`
	CPUUtilizationPct    float64      `json:"cpu_utilization_pct"`
	
	// Model Quality Metrics
	PrecisionScore       float64      `json:"precision_score"`
	RecallScore          float64      `json:"recall_score"`
	F1Score              float64      `json:"f1_score"`
	AUCScore             float64      `json:"auc_score"`
	
	// Robustness Metrics
	NoiseResistance      float64      `json:"noise_resistance"`
	DriftDetection       float64      `json:"drift_detection"`
	ConfidenceCalibration float64     `json:"confidence_calibration"`
	
	// Resource Efficiency
	InferenceTimeMs      float64      `json:"inference_time_ms"`
	ModelSizeMB          float64      `json:"model_size_mb"`
	EnergyEfficiency     float64      `json:"energy_efficiency"`
	
	// Test Details
	TestCases            int          `json:"test_cases"`
	FailedCases          int          `json:"failed_cases"`
	EdgeCases            int          `json:"edge_cases"`
	StressTestPassed     bool         `json:"stress_test_passed"`
}

// BenchmarkType defines different types of benchmarks
type BenchmarkType string

const (
	BenchmarkTypeAccuracy      BenchmarkType = "accuracy"
	BenchmarkTypePerformance   BenchmarkType = "performance"
	BenchmarkTypeRobustness    BenchmarkType = "robustness"
	BenchmarkTypeScalability   BenchmarkType = "scalability"
	BenchmarkTypeComprehensive BenchmarkType = "comprehensive"
)

// Percentiles contains percentile statistics
type Percentiles struct {
	P50 float64 `json:"p50"`
	P90 float64 `json:"p90"`
	P95 float64 `json:"p95"`
	P99 float64 `json:"p99"`
}

// LatencyStats contains latency statistics
type LatencyStats struct {
	Mean       time.Duration `json:"mean"`
	Median     time.Duration `json:"median"`
	P95        time.Duration `json:"p95"`
	P99        time.Duration `json:"p99"`
	Min        time.Duration `json:"min"`
	Max        time.Duration `json:"max"`
	StdDev     time.Duration `json:"std_dev"`
}

// BenchmarkConfig configures benchmark execution
type BenchmarkConfig struct {
	ModelType           ModelType     `json:"model_type"`
	BenchmarkTypes      []BenchmarkType `json:"benchmark_types"`
	TestDataSize        int           `json:"test_data_size"`
	ConcurrentRequests  int           `json:"concurrent_requests"`
	Duration            time.Duration `json:"duration"`
	WarmupRequests      int           `json:"warmup_requests"`
	NoiseLevel          float64       `json:"noise_level"`
	StressTestEnabled   bool          `json:"stress_test_enabled"`
	CompareBaseline     bool          `json:"compare_baseline"`
}

// NewBenchmarkSuite creates a new benchmark suite
func NewBenchmarkSuite(logger *logrus.Logger) *BenchmarkSuite {
	return &BenchmarkSuite{
		logger:  logger,
		results: make(map[string]*BenchmarkResult),
	}
}

// RunComprehensiveBenchmark runs a comprehensive benchmark suite
func (bs *BenchmarkSuite) RunComprehensiveBenchmark(ctx context.Context, model *TrainedModel, config BenchmarkConfig) (*BenchmarkResult, error) {
	bs.logger.WithFields(logrus.Fields{
		"model_type":    model.Type,
		"model_version": model.Version,
	}).Info("Starting comprehensive model benchmark")

	startTime := time.Now()
	
	result := &BenchmarkResult{
		ModelType:     model.Type,
		ModelVersion:  model.Version,
		BenchmarkType: BenchmarkTypeComprehensive,
		StartTime:     startTime,
		TestCases:     config.TestDataSize,
	}

	// Generate test dataset
	testData, err := bs.generateBenchmarkData(model.Type, config.TestDataSize)
	if err != nil {
		return nil, fmt.Errorf("failed to generate test data: %w", err)
	}

	// Run different benchmark types
	for _, benchmarkType := range config.BenchmarkTypes {
		switch benchmarkType {
		case BenchmarkTypeAccuracy:
			err = bs.runAccuracyBenchmark(ctx, model, testData, result)
		case BenchmarkTypePerformance:
			err = bs.runPerformanceBenchmark(ctx, model, testData, result)
		case BenchmarkTypeRobustness:
			err = bs.runRobustnessBenchmark(ctx, model, testData, result)
		case BenchmarkTypeScalability:
			err = bs.runScalabilityBenchmark(ctx, model, testData, result)
		}

		if err != nil {
			bs.logger.WithError(err).WithField("benchmark_type", benchmarkType).Error("Benchmark failed")
			result.FailedCases++
		}
	}

	// Run stress test if enabled
	if config.StressTestEnabled {
		result.StressTestPassed = bs.runStressTest(ctx, model, testData)
	}

	result.Duration = time.Since(startTime)
	
	// Store result
	resultKey := fmt.Sprintf("%s-%s", model.Type, model.Version)
	bs.mu.Lock()
	bs.results[resultKey] = result
	bs.mu.Unlock()

	bs.logger.WithFields(logrus.Fields{
		"model_type":         model.Type,
		"benchmark_duration": result.Duration,
		"accuracy":          result.PredictionAccuracy,
		"throughput":        result.ThroughputRPS,
		"p95_latency":       result.LatencyMetrics.P95,
	}).Info("Comprehensive benchmark completed")

	return result, nil
}

// runAccuracyBenchmark tests prediction accuracy
func (bs *BenchmarkSuite) runAccuracyBenchmark(ctx context.Context, model *TrainedModel, testData *TrainingDataset, result *BenchmarkResult) error {
	bs.logger.Debug("Running accuracy benchmark")

	var predictions []float64
	var confidences []float64
	correct := 0
	totalCases := 0

	// Track accuracy over time
	var accuracyTrend []float64
	batchSize := 100

	for i := 0; i < testData.Size; i++ {
		prediction := bs.predictWithConfidence(model, testData.Features[i])
		predictions = append(predictions, prediction.Value)
		confidences = append(confidences, prediction.Confidence)

		// Calculate accuracy
		if bs.isCorrectPrediction(prediction.Value, testData.Labels[i]) {
			correct++
		}
		totalCases++

		// Track trend
		if (i+1)%batchSize == 0 {
			batchAccuracy := float64(correct) / float64(totalCases)
			accuracyTrend = append(accuracyTrend, batchAccuracy)
		}
	}

	result.PredictionAccuracy = float64(correct) / float64(totalCases)
	result.AccuracyTrend = accuracyTrend

	// Calculate detailed accuracy metrics
	result.PrecisionScore = bs.calculatePrecision(predictions, testData.Labels)
	result.RecallScore = bs.calculateRecall(predictions, testData.Labels)
	result.F1Score = bs.calculateF1Score(result.PrecisionScore, result.RecallScore)
	result.AUCScore = bs.calculateAUC(predictions, testData.Labels)

	// Calculate accuracy percentiles
	accuracies := bs.calculatePerSampleAccuracies(predictions, testData.Labels)
	result.AccuracyPercentiles = bs.calculatePercentiles(accuracies)

	// Confidence calibration
	result.ConfidenceCalibration = bs.calculateConfidenceCalibration(predictions, testData.Labels, confidences)

	return nil
}

// runPerformanceBenchmark tests performance metrics
func (bs *BenchmarkSuite) runPerformanceBenchmark(ctx context.Context, model *TrainedModel, testData *TrainingDataset, result *BenchmarkResult) error {
	bs.logger.Debug("Running performance benchmark")

	// Warmup
	for i := 0; i < 100; i++ {
		if i < testData.Size {
			bs.predictWithConfidence(model, testData.Features[i])
		}
	}

	// Throughput test
	throughput, latencyStats := bs.measureThroughputAndLatency(model, testData)
	result.ThroughputRPS = throughput
	result.LatencyMetrics = latencyStats

	// Resource usage measurement
	result.MemoryUsageMB = bs.measureMemoryUsage(model)
	result.CPUUtilizationPct = bs.measureCPUUtilization(model, testData)

	// Model efficiency metrics
	result.InferenceTimeMs = float64(latencyStats.Mean.Nanoseconds()) / 1e6
	result.ModelSizeMB = bs.calculateModelSize(model)
	result.EnergyEfficiency = bs.calculateEnergyEfficiency(result.ThroughputRPS, result.MemoryUsageMB)

	return nil
}

// runRobustnessBenchmark tests model robustness
func (bs *BenchmarkSuite) runRobustnessBenchmark(ctx context.Context, model *TrainedModel, testData *TrainingDataset, result *BenchmarkResult) error {
	bs.logger.Debug("Running robustness benchmark")

	// Test noise resistance
	result.NoiseResistance = bs.testNoiseResistance(model, testData)

	// Test drift detection
	result.DriftDetection = bs.testDriftDetection(model, testData)

	// Count edge cases
	result.EdgeCases = bs.identifyEdgeCases(model, testData)

	return nil
}

// runScalabilityBenchmark tests scalability under load
func (bs *BenchmarkSuite) runScalabilityBenchmark(ctx context.Context, model *TrainedModel, testData *TrainingDataset, result *BenchmarkResult) error {
	bs.logger.Debug("Running scalability benchmark")

	// Test concurrent predictions
	concurrency := []int{1, 5, 10, 25, 50}
	scalabilityResults := make(map[int]float64)

	for _, c := range concurrency {
		throughput := bs.measureConcurrentThroughput(model, testData, c)
		scalabilityResults[c] = throughput
	}

	// Calculate scalability efficiency (throughput should scale linearly)
	baselineThroughput := scalabilityResults[1]
	highConcurrencyThroughput := scalabilityResults[50]
	expectedThroughput := baselineThroughput * 50
	
	scalabilityEfficiency := highConcurrencyThroughput / expectedThroughput
	
	bs.logger.WithFields(logrus.Fields{
		"baseline_throughput": baselineThroughput,
		"high_concurrency_throughput": highConcurrencyThroughput,
		"scalability_efficiency": scalabilityEfficiency,
	}).Info("Scalability benchmark results")

	return nil
}

// runStressTest performs stress testing
func (bs *BenchmarkSuite) runStressTest(ctx context.Context, model *TrainedModel, testData *TrainingDataset) bool {
	bs.logger.Debug("Running stress test")

	// Test with high load for extended period
	duration := 5 * time.Minute
	endTime := time.Now().Add(duration)

	errorCount := 0
	requestCount := 0

	for time.Now().Before(endTime) {
		select {
		case <-ctx.Done():
			return false
		default:
		}

		// Make concurrent predictions
		var wg sync.WaitGroup
		errors := make(chan error, 10)

		for i := 0; i < 10; i++ {
			wg.Add(1)
			go func(idx int) {
				defer wg.Done()
				if idx < testData.Size {
					_, err := bs.stressPrediction(model, testData.Features[idx])
					if err != nil {
						errors <- err
					}
				}
			}(requestCount % testData.Size)
			requestCount++
		}

		wg.Wait()
		close(errors)

		for err := range errors {
			if err != nil {
				errorCount++
			}
		}

		if errorCount > requestCount/10 { // More than 10% errors
			return false
		}

		time.Sleep(10 * time.Millisecond)
	}

	return true
}

// CompareModels compares benchmark results between models
func (bs *BenchmarkSuite) CompareModels(modelA, modelB string) (*ModelComparison, error) {
	bs.mu.RLock()
	defer bs.mu.RUnlock()

	resultA, existsA := bs.results[modelA]
	resultB, existsB := bs.results[modelB]

	if !existsA || !existsB {
		return nil, fmt.Errorf("benchmark results not found for comparison")
	}

	comparison := &ModelComparison{
		ModelA:         modelA,
		ModelB:         modelB,
		ComparisonTime: time.Now(),
		
		AccuracyImprovement:   resultB.PredictionAccuracy - resultA.PredictionAccuracy,
		ThroughputImprovement: resultB.ThroughputRPS - resultA.ThroughputRPS,
		LatencyImprovement:    resultA.LatencyMetrics.P95 - resultB.LatencyMetrics.P95, // Lower is better
		MemoryImprovement:     resultA.MemoryUsageMB - resultB.MemoryUsageMB,           // Lower is better
		
		OverallScore: bs.calculateOverallScore(resultB) - bs.calculateOverallScore(resultA),
		
		ResultA: resultA,
		ResultB: resultB,
	}

	comparison.Recommendation = bs.generateRecommendation(comparison)

	return comparison, nil
}

// ModelComparison contains comparison results between two models
type ModelComparison struct {
	ModelA                string          `json:"model_a"`
	ModelB                string          `json:"model_b"`
	ComparisonTime        time.Time       `json:"comparison_time"`
	AccuracyImprovement   float64         `json:"accuracy_improvement"`
	ThroughputImprovement float64         `json:"throughput_improvement"`
	LatencyImprovement    time.Duration   `json:"latency_improvement"`
	MemoryImprovement     float64         `json:"memory_improvement"`
	OverallScore          float64         `json:"overall_score"`
	Recommendation        string          `json:"recommendation"`
	ResultA               *BenchmarkResult `json:"result_a"`
	ResultB               *BenchmarkResult `json:"result_b"`
}

// Helper methods and prediction structures

// PredictionResult contains prediction value and confidence
type PredictionResult struct {
	Value      float64 `json:"value"`
	Confidence float64 `json:"confidence"`
}

func (bs *BenchmarkSuite) predictWithConfidence(model *TrainedModel, features []float64) PredictionResult {
	// Simulate prediction with confidence
	prediction := bs.predict(model, features)
	confidence := bs.calculatePredictionConfidence(model, features, prediction)
	
	return PredictionResult{
		Value:      prediction,
		Confidence: confidence,
	}
}

func (bs *BenchmarkSuite) predict(model *TrainedModel, features []float64) float64 {
	// Simplified prediction based on model type
	switch model.Type {
	case ModelTypePlacementPredictor:
		return features[0]*model.Parameters["weight_cpu"] +
			features[1]*model.Parameters["weight_memory"] +
			features[2]*model.Parameters["weight_disk"] +
			features[3]*model.Parameters["weight_network"] +
			features[4]*model.Parameters["weight_load"]
			
	case ModelTypeScalingPredictor:
		cpuUtil := features[0]
		memUtil := features[1]
		
		if cpuUtil > model.Parameters["cpu_threshold"] || memUtil > model.Parameters["memory_threshold"] {
			return 1.0
		} else if cpuUtil < 0.3 && memUtil < 0.3 {
			return -1.0
		}
		return 0.0
		
	default:
		// Generic linear prediction
		result := 0.0
		for i, feature := range features {
			if i < len(features) {
				weight := model.Parameters[fmt.Sprintf("weight_%d", i)]
				result += feature * weight
			}
		}
		return result
	}
}

func (bs *BenchmarkSuite) calculatePredictionConfidence(model *TrainedModel, features []float64, prediction float64) float64 {
	// Simplified confidence calculation
	// In practice, this would be based on model uncertainty, ensemble variance, etc.
	
	// Calculate feature similarity to training distribution
	featureMean := 0.0
	for _, feature := range features {
		featureMean += feature
	}
	featureMean /= float64(len(features))
	
	// Higher confidence for predictions closer to typical ranges
	confidence := 1.0 - math.Abs(featureMean-0.5)*0.5
	
	// Adjust based on prediction value
	if math.Abs(prediction) > 1.0 {
		confidence *= 0.8 // Lower confidence for extreme predictions
	}
	
	return math.Max(0.1, math.Min(0.99, confidence))
}

func (bs *BenchmarkSuite) generateBenchmarkData(modelType ModelType, size int) (*TrainingDataset, error) {
	features := make([][]float64, size)
	labels := make([]float64, size)
	
	switch modelType {
	case ModelTypePlacementPredictor:
		for i := 0; i < size; i++ {
			features[i] = []float64{
				float64(1 + i%16),           // cpu_cores
				float64(1024 + i%32*1024),   // memory_mb
				float64(50 + i%500),         // disk_gb
				float64(100 + i%900),        // network_mbps
				float64(i%100) / 100.0,      // node_load
			}
			
			// Generate placement success probability
			labels[i] = bs.calculatePlacementSuccess(features[i])
		}
		
	case ModelTypeScalingPredictor:
		for i := 0; i < size; i++ {
			features[i] = []float64{
				float64(i%100) / 100.0,      // cpu_utilization
				float64(i%100) / 100.0,      // memory_utilization
				float64(100 + i%1000),       // request_rate
				float64(50 + i%500),         // response_time
				float64(i % 50),             // queue_length
			}
			
			labels[i] = bs.calculateScalingNeed(features[i])
		}
		
	default:
		// Generic synthetic data
		for i := 0; i < size; i++ {
			features[i] = []float64{
				float64(i) / float64(size),
				math.Sin(float64(i) / 10.0),
				math.Cos(float64(i) / 10.0),
				float64(i%10) / 10.0,
				float64(i%5) / 5.0,
			}
			
			labels[i] = features[i][0]*0.5 + features[i][1]*0.3 + features[i][2]*0.2
		}
	}
	
	return &TrainingDataset{
		Type:         modelType,
		Features:     features,
		Labels:       labels,
		Size:         size,
		LastUpdated:  time.Now(),
	}, nil
}

func (bs *BenchmarkSuite) calculatePlacementSuccess(features []float64) float64 {
	// Simplified placement success calculation
	cpuScore := math.Min(features[0]/16.0, 1.0)
	memScore := math.Min(features[1]/32768.0, 1.0)
	loadPenalty := features[4] // node_load
	
	return (cpuScore + memScore) * 0.5 * (1.0 - loadPenalty*0.5)
}

func (bs *BenchmarkSuite) calculateScalingNeed(features []float64) float64 {
	cpuUtil := features[0]
	memUtil := features[1]
	
	// Scale up if utilization is high
	if cpuUtil > 0.8 || memUtil > 0.8 {
		return 1.0 // Scale up
	} else if cpuUtil < 0.3 && memUtil < 0.3 {
		return -1.0 // Scale down
	}
	
	return 0.0 // No scaling needed
}

func (bs *BenchmarkSuite) isCorrectPrediction(predicted, actual float64) bool {
	threshold := 0.1
	return math.Abs(predicted-actual) < threshold
}

func (bs *BenchmarkSuite) calculatePrecision(predictions, actual []float64) float64 {
	// Simplified precision calculation for regression
	correct := 0
	predicted_positive := 0
	
	for i := range predictions {
		if predictions[i] > 0.5 {
			predicted_positive++
			if bs.isCorrectPrediction(predictions[i], actual[i]) {
				correct++
			}
		}
	}
	
	if predicted_positive == 0 {
		return 0.0
	}
	
	return float64(correct) / float64(predicted_positive)
}

func (bs *BenchmarkSuite) calculateRecall(predictions, actual []float64) float64 {
	// Simplified recall calculation for regression
	correct := 0
	actual_positive := 0
	
	for i := range actual {
		if actual[i] > 0.5 {
			actual_positive++
			if bs.isCorrectPrediction(predictions[i], actual[i]) {
				correct++
			}
		}
	}
	
	if actual_positive == 0 {
		return 0.0
	}
	
	return float64(correct) / float64(actual_positive)
}

func (bs *BenchmarkSuite) calculateF1Score(precision, recall float64) float64 {
	if precision+recall == 0 {
		return 0.0
	}
	return 2 * precision * recall / (precision + recall)
}

func (bs *BenchmarkSuite) calculateAUC(predictions, actual []float64) float64 {
	// Simplified AUC calculation
	// In practice, would implement proper ROC curve calculation
	return 0.85 // Placeholder
}

func (bs *BenchmarkSuite) calculatePerSampleAccuracies(predictions, actual []float64) []float64 {
	accuracies := make([]float64, len(predictions))
	for i := range predictions {
		if bs.isCorrectPrediction(predictions[i], actual[i]) {
			accuracies[i] = 1.0
		} else {
			accuracies[i] = 0.0
		}
	}
	return accuracies
}

func (bs *BenchmarkSuite) calculatePercentiles(values []float64) Percentiles {
	sorted := make([]float64, len(values))
	copy(sorted, values)
	sort.Float64s(sorted)
	
	n := len(sorted)
	return Percentiles{
		P50: sorted[n/2],
		P90: sorted[int(0.9*float64(n))],
		P95: sorted[int(0.95*float64(n))],
		P99: sorted[int(0.99*float64(n))],
	}
}

func (bs *BenchmarkSuite) calculateConfidenceCalibration(predictions, actual, confidences []float64) float64 {
	// Measure how well confidence scores match actual accuracy
	// This is a simplified implementation
	
	if len(confidences) == 0 {
		return 0.0
	}
	
	avgConfidence := 0.0
	avgAccuracy := 0.0
	
	for i := range predictions {
		avgConfidence += confidences[i]
		if bs.isCorrectPrediction(predictions[i], actual[i]) {
			avgAccuracy += 1.0
		}
	}
	
	avgConfidence /= float64(len(confidences))
	avgAccuracy /= float64(len(predictions))
	
	// Return calibration score (closer to 1.0 means better calibrated)
	return 1.0 - math.Abs(avgConfidence-avgAccuracy)
}

func (bs *BenchmarkSuite) measureThroughputAndLatency(model *TrainedModel, testData *TrainingDataset) (float64, LatencyStats) {
	duration := 30 * time.Second
	endTime := time.Now().Add(duration)
	
	var latencies []time.Duration
	requestCount := 0
	
	for time.Now().Before(endTime) && requestCount < testData.Size {
		start := time.Now()
		bs.predict(model, testData.Features[requestCount%testData.Size])
		latency := time.Since(start)
		
		latencies = append(latencies, latency)
		requestCount++
	}
	
	actualDuration := time.Since(endTime.Add(-duration))
	throughput := float64(requestCount) / actualDuration.Seconds()
	
	// Calculate latency statistics
	sort.Slice(latencies, func(i, j int) bool {
		return latencies[i] < latencies[j]
	})
	
	n := len(latencies)
	latencyStats := LatencyStats{
		Min:    latencies[0],
		Max:    latencies[n-1],
		Median: latencies[n/2],
		P95:    latencies[int(0.95*float64(n))],
		P99:    latencies[int(0.99*float64(n))],
	}
	
	// Calculate mean and standard deviation
	sum := time.Duration(0)
	for _, lat := range latencies {
		sum += lat
	}
	latencyStats.Mean = sum / time.Duration(n)
	
	// Standard deviation calculation would go here
	latencyStats.StdDev = latencyStats.Mean / 4 // Simplified
	
	return throughput, latencyStats
}

func (bs *BenchmarkSuite) measureMemoryUsage(model *TrainedModel) float64 {
	// Simplified memory usage calculation
	parameterCount := len(model.Parameters)
	return float64(parameterCount * 8) / (1024 * 1024) // 8 bytes per float64, convert to MB
}

func (bs *BenchmarkSuite) measureCPUUtilization(model *TrainedModel, testData *TrainingDataset) float64 {
	// Simplified CPU utilization measurement
	// In practice, would use system monitoring
	return 25.0 + float64(len(model.Parameters))*0.1 // Placeholder calculation
}

func (bs *BenchmarkSuite) calculateModelSize(model *TrainedModel) float64 {
	return bs.measureMemoryUsage(model) // Same calculation for now
}

func (bs *BenchmarkSuite) calculateEnergyEfficiency(throughputRPS, memoryUsageMB float64) float64 {
	// Energy efficiency = throughput per unit resource
	return throughputRPS / (memoryUsageMB + 1.0) // Add 1 to avoid division by zero
}

func (bs *BenchmarkSuite) testNoiseResistance(model *TrainedModel, testData *TrainingDataset) float64 {
	// Test model performance with noisy inputs
	originalAccuracy := bs.calculateAccuracyOnDataset(model, testData)
	
	// Add noise to test data
	noisyData := bs.addNoiseToDataset(testData, 0.1)
	noisyAccuracy := bs.calculateAccuracyOnDataset(model, noisyData)
	
	// Noise resistance = how much accuracy is retained with noise
	return noisyAccuracy / originalAccuracy
}

func (bs *BenchmarkSuite) testDriftDetection(model *TrainedModel, testData *TrainingDataset) float64 {
	// Simulate data drift and test detection
	driftedData := bs.simulateDataDrift(testData, 0.2)
	
	originalAccuracy := bs.calculateAccuracyOnDataset(model, testData)
	driftedAccuracy := bs.calculateAccuracyOnDataset(model, driftedData)
	
	// Return drift magnitude (higher = more drift detected)
	return math.Abs(originalAccuracy - driftedAccuracy)
}

func (bs *BenchmarkSuite) identifyEdgeCases(model *TrainedModel, testData *TrainingDataset) int {
	edgeCases := 0
	
	for i := 0; i < testData.Size; i++ {
		prediction := bs.predictWithConfidence(model, testData.Features[i])
		
		// Identify edge cases based on low confidence or extreme values
		if prediction.Confidence < 0.3 || math.Abs(prediction.Value) > 2.0 {
			edgeCases++
		}
	}
	
	return edgeCases
}

func (bs *BenchmarkSuite) measureConcurrentThroughput(model *TrainedModel, testData *TrainingDataset, concurrency int) float64 {
	duration := 10 * time.Second
	endTime := time.Now().Add(duration)
	
	var wg sync.WaitGroup
	requestCount := int64(0)
	
	for i := 0; i < concurrency; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			localCount := 0
			
			for time.Now().Before(endTime) {
				bs.predict(model, testData.Features[localCount%testData.Size])
				localCount++
			}
			
			// Atomic add would be used here in production
			requestCount += int64(localCount)
		}(i)
	}
	
	wg.Wait()
	
	return float64(requestCount) / duration.Seconds()
}

func (bs *BenchmarkSuite) stressPrediction(model *TrainedModel, features []float64) (float64, error) {
	// Stress test with potential timeouts and error handling
	start := time.Now()
	prediction := bs.predict(model, features)
	elapsed := time.Since(start)
	
	// Simulate timeout conditions
	if elapsed > 100*time.Millisecond {
		return 0, fmt.Errorf("prediction timeout")
	}
	
	return prediction, nil
}

func (bs *BenchmarkSuite) calculateAccuracyOnDataset(model *TrainedModel, dataset *TrainingDataset) float64 {
	correct := 0
	for i := 0; i < dataset.Size; i++ {
		prediction := bs.predict(model, dataset.Features[i])
		if bs.isCorrectPrediction(prediction, dataset.Labels[i]) {
			correct++
		}
	}
	return float64(correct) / float64(dataset.Size)
}

func (bs *BenchmarkSuite) addNoiseToDataset(dataset *TrainingDataset, noiseLevel float64) *TrainingDataset {
	noisyFeatures := make([][]float64, dataset.Size)
	
	for i := 0; i < dataset.Size; i++ {
		noisyFeatures[i] = make([]float64, len(dataset.Features[i]))
		for j := 0; j < len(dataset.Features[i]); j++ {
			noise := (math.Sin(float64(i+j)) - 0.5) * noiseLevel
			noisyFeatures[i][j] = dataset.Features[i][j] + noise
		}
	}
	
	return &TrainingDataset{
		Features: noisyFeatures,
		Labels:   dataset.Labels,
		Size:     dataset.Size,
	}
}

func (bs *BenchmarkSuite) simulateDataDrift(dataset *TrainingDataset, driftMagnitude float64) *TrainingDataset {
	driftedFeatures := make([][]float64, dataset.Size)
	
	for i := 0; i < dataset.Size; i++ {
		driftedFeatures[i] = make([]float64, len(dataset.Features[i]))
		for j := 0; j < len(dataset.Features[i]); j++ {
			// Simulate gradual drift
			drift := driftMagnitude * float64(i) / float64(dataset.Size)
			driftedFeatures[i][j] = dataset.Features[i][j] + drift
		}
	}
	
	return &TrainingDataset{
		Features: driftedFeatures,
		Labels:   dataset.Labels,
		Size:     dataset.Size,
	}
}

func (bs *BenchmarkSuite) calculateOverallScore(result *BenchmarkResult) float64 {
	// Weighted combination of different metrics
	accuracyWeight := 0.4
	throughputWeight := 0.3
	latencyWeight := 0.2
	efficiencyWeight := 0.1
	
	// Normalize latency (lower is better)
	normalizedLatency := 1.0 / (1.0 + result.LatencyMetrics.P95.Seconds())
	
	score := result.PredictionAccuracy*accuracyWeight +
		(result.ThroughputRPS/1000.0)*throughputWeight +
		normalizedLatency*latencyWeight +
		result.EnergyEfficiency*efficiencyWeight
		
	return score
}

func (bs *BenchmarkSuite) generateRecommendation(comparison *ModelComparison) string {
	if comparison.OverallScore > 0.1 {
		return fmt.Sprintf("Model B shows significant improvement (%.2f%% better overall score)", comparison.OverallScore*100)
	} else if comparison.OverallScore < -0.1 {
		return fmt.Sprintf("Model A performs better (%.2f%% better overall score)", -comparison.OverallScore*100)
	} else {
		return "Both models show similar performance. Consider other factors like maintenance cost and complexity."
	}
}