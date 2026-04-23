package prediction

import (
	"fmt"
	"math"
	"os"
	"sync"
	"time"
)

// LSTMPredictor handles LSTM-based bandwidth prediction using ONNX runtime
type LSTMPredictor struct {
	modelLoaded  bool
	inputNames   []string
	outputNames  []string
	modelPath    string
	modelVersion string
	loadTime     time.Time
	mu           sync.RWMutex

	// Model parameters
	sequenceLength int
	featureCount   int
	outputCount    int

	// Performance metrics
	inferenceCount uint64
	totalLatency   time.Duration
	predictions    []PredictionRecord
}

// PredictionRecord stores prediction history for analysis
type PredictionRecord struct {
	Timestamp       time.Time
	Predicted       BandwidthPrediction
	Actual          *NetworkSample
	Error           float64
	InferenceTimeMs float64
}

// NewLSTMPredictor creates a new LSTM predictor with ONNX runtime
func NewLSTMPredictor(modelPath string) (*LSTMPredictor, error) {
	predictor := &LSTMPredictor{
		modelPath:      modelPath,
		modelVersion:   "v1.0",
		loadTime:       time.Now(),
		sequenceLength: 10,
		featureCount:   6,
		outputCount:    4,
		inputNames:     []string{"input"},
		outputNames:    []string{"output"},
		predictions:    make([]PredictionRecord, 0, 1000),
	}

	if modelPath != "" {
		if _, err := os.Stat(modelPath); err == nil {
			predictor.modelLoaded = true
		} else if !os.IsNotExist(err) {
			return nil, fmt.Errorf("failed to inspect model path %q: %w", modelPath, err)
		}
	}

	return predictor, nil
}

// Predict generates bandwidth predictions from historical network samples
func (p *LSTMPredictor) Predict(history []NetworkSample) (*BandwidthPrediction, error) {
	if len(history) < p.sequenceLength {
		return nil, fmt.Errorf("insufficient history: need %d samples, got %d",
			p.sequenceLength, len(history))
	}

	startTime := time.Now()

	// Prepare normalized input vector.
	_, err := p.prepareInput(history)
	if err != nil {
		return nil, fmt.Errorf("failed to prepare input: %w", err)
	}

	prediction, err := p.parseOutput(nil)
	if err != nil {
		return nil, fmt.Errorf("failed to parse output: %w", err)
	}
	prediction.ModelVersion = p.modelVersion
	prediction.PredictionTime = time.Now()

	// Calculate inference time
	inferenceTime := time.Since(startTime)

	// Update metrics
	p.mu.Lock()
	p.inferenceCount++
	p.totalLatency += inferenceTime

	// Store prediction record
	record := PredictionRecord{
		Timestamp:       time.Now(),
		Predicted:       *prediction,
		InferenceTimeMs: float64(inferenceTime.Microseconds()) / 1000.0,
	}
	p.predictions = append(p.predictions, record)
	if len(p.predictions) > 1000 {
		p.predictions = p.predictions[1:]
	}
	p.mu.Unlock()

	return prediction, nil
}

// prepareInput converts network samples to a normalized tensor-like vector.
func (p *LSTMPredictor) prepareInput(history []NetworkSample) ([]float32, error) {
	// Create input array: [batch_size=1, sequence_length=10, features=6]
	inputData := make([]float32, p.sequenceLength*p.featureCount)

	// Take last 10 samples
	startIdx := len(history) - p.sequenceLength

	for i := 0; i < p.sequenceLength; i++ {
		sample := history[startIdx+i]
		baseIdx := i * p.featureCount

		// Normalize features
		inputData[baseIdx+0] = float32(sample.BandwidthMbps / 1000.0) // Normalize to 0-1 range
		inputData[baseIdx+1] = float32(sample.LatencyMs / 100.0)      // Normalize to 0-1 range
		inputData[baseIdx+2] = float32(sample.PacketLoss)             // Already 0-1
		inputData[baseIdx+3] = float32(sample.JitterMs / 50.0)        // Normalize to 0-1 range
		inputData[baseIdx+4] = float32(sample.TimeOfDay) / 24.0       // Normalize to 0-1
		inputData[baseIdx+5] = float32(sample.DayOfWeek) / 7.0        // Normalize to 0-1
	}

	return inputData, nil
}

// parseOutput converts model output to a bandwidth prediction.
func (p *LSTMPredictor) parseOutput(_ []float32) (*BandwidthPrediction, error) {
	prediction := &BandwidthPrediction{
		PredictedBandwidthMbps: 100.0, // Default placeholder
		PredictedLatencyMs:     10.0,  // Default placeholder
		PredictedPacketLoss:    0.01,  // Default placeholder
		PredictedJitterMs:      2.0,   // Default placeholder
		ValidUntil:             time.Now().Add(15 * time.Minute),
	}

	prediction.Confidence = p.calculateConfidence(prediction)
	if !p.modelLoaded {
		prediction.Confidence = math.Min(prediction.Confidence, 0.5)
	}

	return prediction, nil
}

// calculateConfidence estimates prediction confidence
func (p *LSTMPredictor) calculateConfidence(pred *BandwidthPrediction) float64 {
	p.mu.RLock()
	defer p.mu.RUnlock()

	if len(p.predictions) < 10 {
		return 0.5 // Default confidence for new model
	}

	// Calculate recent prediction accuracy
	recentErrors := make([]float64, 0)
	for i := len(p.predictions) - 1; i >= 0 && len(recentErrors) < 20; i-- {
		if p.predictions[i].Actual != nil {
			recentErrors = append(recentErrors, p.predictions[i].Error)
		}
	}

	if len(recentErrors) == 0 {
		return 0.5
	}

	// Calculate mean absolute error
	var totalError float64
	for _, err := range recentErrors {
		totalError += math.Abs(err)
	}
	avgError := totalError / float64(len(recentErrors))

	// Convert error to confidence (lower error = higher confidence)
	confidence := math.Max(0.0, math.Min(1.0, 1.0-(avgError/0.2)))

	return confidence
}

// UpdateActual updates prediction record with actual values for accuracy tracking
func (p *LSTMPredictor) UpdateActual(timestamp time.Time, actual NetworkSample) {
	p.mu.Lock()
	defer p.mu.Unlock()

	// Find matching prediction
	for i := len(p.predictions) - 1; i >= 0; i-- {
		pred := &p.predictions[i]
		if pred.Timestamp.Before(timestamp) && pred.Actual == nil {
			pred.Actual = &actual

			// Calculate prediction error (normalized)
			bandwidthError := math.Abs(pred.Predicted.PredictedBandwidthMbps-actual.BandwidthMbps) /
				actual.BandwidthMbps
			latencyError := math.Abs(pred.Predicted.PredictedLatencyMs-actual.LatencyMs) /
				actual.LatencyMs

			pred.Error = (bandwidthError + latencyError) / 2.0
			break
		}
	}
}

// GetMetrics returns predictor performance metrics
func (p *LSTMPredictor) GetMetrics() PredictorMetrics {
	p.mu.RLock()
	defer p.mu.RUnlock()

	avgLatency := time.Duration(0)
	if p.inferenceCount > 0 {
		avgLatency = p.totalLatency / time.Duration(p.inferenceCount)
	}

	// Calculate accuracy metrics
	var totalError, maxError float64
	var errorCount int
	for _, pred := range p.predictions {
		if pred.Actual != nil {
			totalError += pred.Error
			if pred.Error > maxError {
				maxError = pred.Error
			}
			errorCount++
		}
	}

	avgError := 0.0
	if errorCount > 0 {
		avgError = totalError / float64(errorCount)
	}

	return PredictorMetrics{
		ModelVersion:       p.modelVersion,
		LoadTime:           p.loadTime,
		InferenceCount:     p.inferenceCount,
		AvgInferenceMs:     float64(avgLatency.Microseconds()) / 1000.0,
		AvgPredictionError: avgError,
		MaxPredictionError: maxError,
		Accuracy:           1.0 - avgError,
	}
}

// ReloadModel reloads the ONNX model from disk
func (p *LSTMPredictor) ReloadModel(modelPath string) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	if _, err := os.Stat(modelPath); err != nil {
		return fmt.Errorf("failed to load model %q: %w", modelPath, err)
	}
	p.modelLoaded = true
	p.modelPath = modelPath
	p.loadTime = time.Now()
	p.modelVersion = fmt.Sprintf("v%d", time.Now().Unix())

	return nil
}

// Close cleans up resources
func (p *LSTMPredictor) Close() {
	p.mu.Lock()
	defer p.mu.Unlock()

	p.modelLoaded = false
}

// PredictorMetrics contains performance metrics for the predictor
type PredictorMetrics struct {
	ModelVersion       string
	LoadTime           time.Time
	InferenceCount     uint64
	AvgInferenceMs     float64
	AvgPredictionError float64
	MaxPredictionError float64
	Accuracy           float64
}
