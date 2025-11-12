package prediction

import (
	"fmt"
	"math"
	"sync"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/prediction"
	ort "github.com/yalue/onnxruntime_go"
)

// LSTMPredictorV3 is the v3 LSTM predictor optimized for internet scenarios
// Key differences from v1:
// - Longer sequence length (60 vs 10 timesteps)
// - More complex model (256/128 vs 128/64 LSTM units)
// - Higher dropout for regularization (0.3 vs 0.2)
// - Lower accuracy target (70% vs 85%)
type LSTMPredictorV3 struct {
	session       *ort.DynamicAdvancedSession
	inputNames    []string
	outputNames   []string
	modelPath     string
	modelVersion  string
	loadTime      time.Time
	mu            sync.RWMutex

	// Model parameters (v3 specific)
	sequenceLength int // 60 timesteps for internet
	featureCount   int // 5 features
	outputCount    int // 4 outputs

	// Performance metrics
	inferenceCount uint64
	totalLatency   time.Duration
	predictions    []prediction.PredictionRecord

	// v3 specific: adaptive confidence adjustment
	recentErrors      []float64
	confidenceDecay   float64
	minConfidence     float64
	targetAccuracy    float64
}

// NewLSTMPredictorV3 creates a new v3 LSTM predictor for internet scenarios
func NewLSTMPredictorV3(modelPath string) (*LSTMPredictorV3, error) {
	// Initialize ONNX runtime
	ort.SetSharedLibraryPath("/usr/local/lib/libonnxruntime.so")
	err := ort.InitializeEnvironment()
	if err != nil {
		return nil, fmt.Errorf("failed to initialize ONNX runtime: %w", err)
	}

	// Create session options
	options, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("failed to create session options: %w", err)
	}
	defer options.Destroy()

	// Set optimization level
	err = options.SetGraphOptimizationLevel(ort.GraphOptimizationLevelEnableAll)
	if err != nil {
		return nil, fmt.Errorf("failed to set optimization level: %w", err)
	}

	// Create inference session
	session, err := ort.NewDynamicAdvancedSession(modelPath,
		[]string{"input"}, []string{"output"}, options)
	if err != nil {
		return nil, fmt.Errorf("failed to create ONNX session: %w", err)
	}

	predictor := &LSTMPredictorV3{
		session:        session,
		modelPath:      modelPath,
		modelVersion:   "v3.0-internet",
		loadTime:       time.Now(),
		sequenceLength: 60, // Longer for internet variability
		featureCount:   5,
		outputCount:    4,
		inputNames:     []string{"input"},
		outputNames:    []string{"output"},
		predictions:    make([]prediction.PredictionRecord, 0, 1000),
		recentErrors:   make([]float64, 0, 50),
		confidenceDecay: 0.95,
		minConfidence:  0.40, // Lower minimum for internet
		targetAccuracy: 0.70, // 70% target for internet
	}

	return predictor, nil
}

// Predict generates bandwidth predictions from historical network samples
// Requires 60 timesteps for internet mode (vs 10 for datacenter)
func (p *LSTMPredictorV3) Predict(history []prediction.NetworkSample) (*prediction.BandwidthPrediction, error) {
	if len(history) < p.sequenceLength {
		return nil, fmt.Errorf("insufficient history: need %d samples, got %d",
			p.sequenceLength, len(history))
	}

	startTime := time.Now()

	// Prepare input tensor
	inputTensor, err := p.prepareInput(history)
	if err != nil {
		return nil, fmt.Errorf("failed to prepare input: %w", err)
	}
	defer inputTensor.Destroy()

	// Run inference
	outputs, err := p.session.Run([]ort.Value{inputTensor})
	if err != nil {
		return nil, fmt.Errorf("inference failed: %w", err)
	}
	defer func() {
		for _, output := range outputs {
			output.Destroy()
		}
	}()

	// Parse output
	pred, err := p.parseOutput(outputs[0])
	if err != nil {
		return nil, fmt.Errorf("failed to parse output: %w", err)
	}

	// Calculate inference time
	inferenceTime := time.Since(startTime)

	// Update metrics
	p.mu.Lock()
	p.inferenceCount++
	p.totalLatency += inferenceTime

	// Store prediction record
	record := prediction.PredictionRecord{
		Timestamp:       time.Now(),
		Predicted:       *pred,
		InferenceTimeMs: float64(inferenceTime.Microseconds()) / 1000.0,
	}
	p.predictions = append(p.predictions, record)
	if len(p.predictions) > 1000 {
		p.predictions = p.predictions[1:]
	}
	p.mu.Unlock()

	return pred, nil
}

// prepareInput converts network samples to ONNX tensor format
// v3: Uses 60 timesteps with 5 features each
func (p *LSTMPredictorV3) prepareInput(history []prediction.NetworkSample) (ort.Value, error) {
	// Create input array: [batch_size=1, sequence_length=60, features=5]
	inputData := make([]float32, p.sequenceLength*p.featureCount)

	// Take last 60 samples
	startIdx := len(history) - p.sequenceLength

	for i := 0; i < p.sequenceLength; i++ {
		sample := history[startIdx+i]
		baseIdx := i * p.featureCount

		// Normalize features for internet conditions
		// Different normalization ranges than datacenter
		inputData[baseIdx+0] = float32(sample.BandwidthMbps / 1000.0)  // 0-1 range (max 1 Gbps)
		inputData[baseIdx+1] = float32(sample.LatencyMs / 500.0)       // 0-1 range (max 500ms)
		inputData[baseIdx+2] = float32(sample.PacketLoss / 0.05)       // 0-1 range (max 5%)
		inputData[baseIdx+3] = float32(sample.JitterMs / 100.0)        // 0-1 range (max 100ms)
		inputData[baseIdx+4] = float32(sample.TimeOfDay) / 24.0        // 0-1 range
	}

	// Create tensor with shape [1, 60, 5]
	shape := []int64{1, int64(p.sequenceLength), int64(p.featureCount)}
	tensor, err := ort.NewTensor(shape, inputData)
	if err != nil {
		return nil, fmt.Errorf("failed to create input tensor: %w", err)
	}

	return tensor, nil
}

// parseOutput converts ONNX output to bandwidth prediction
func (p *LSTMPredictorV3) parseOutput(output ort.Value) (*prediction.BandwidthPrediction, error) {
	// Get output tensor
	outputData, err := output.GetFloatData()
	if err != nil {
		return nil, fmt.Errorf("failed to get output data: %w", err)
	}

	if len(outputData) < p.outputCount {
		return nil, fmt.Errorf("unexpected output size: %d", len(outputData))
	}

	// Denormalize predictions (internet ranges)
	pred := &prediction.BandwidthPrediction{
		PredictedBandwidthMbps: float64(outputData[0]) * 1000.0, // Max 1 Gbps
		PredictedLatencyMs:     float64(outputData[1]) * 500.0,  // Max 500ms
		PredictedPacketLoss:    float64(outputData[2]) * 0.05,   // Max 5%
		PredictedJitterMs:      float64(outputData[3]) * 100.0,  // Max 100ms
		ValidUntil:             time.Now().Add(15 * time.Minute),
	}

	// Calculate confidence based on prediction variance and recent accuracy
	pred.Confidence = p.calculateConfidenceV3(pred)

	return pred, nil
}

// calculateConfidenceV3 estimates prediction confidence for internet scenarios
// Uses adaptive confidence based on recent prediction errors
func (p *LSTMPredictorV3) calculateConfidenceV3(pred *prediction.BandwidthPrediction) float64 {
	p.mu.RLock()
	defer p.mu.RUnlock()

	// Start with base confidence for internet (70%)
	baseConfidence := p.targetAccuracy

	if len(p.predictions) < 5 {
		// Not enough history, use base confidence
		return baseConfidence
	}

	// Calculate recent prediction accuracy
	recentPredictions := p.predictions
	if len(recentPredictions) > 20 {
		recentPredictions = recentPredictions[len(recentPredictions)-20:]
	}

	var totalError float64
	var errorCount int

	for _, rec := range recentPredictions {
		if rec.Actual != nil {
			totalError += rec.Error
			errorCount++
		}
	}

	if errorCount == 0 {
		return baseConfidence
	}

	avgError := totalError / float64(errorCount)

	// Adjust confidence based on recent accuracy
	// Lower error = higher confidence (up to target)
	// Higher error = lower confidence (down to minimum)
	confidenceAdjustment := 1.0 - (avgError * 2.0) // avgError is 0-0.5 typically

	adjustedConfidence := baseConfidence * confidenceAdjustment

	// Clamp to reasonable range for internet
	adjustedConfidence = math.Max(p.minConfidence, math.Min(p.targetAccuracy, adjustedConfidence))

	return adjustedConfidence
}

// UpdateActual updates prediction record with actual values for accuracy tracking
func (p *LSTMPredictorV3) UpdateActual(timestamp time.Time, actual prediction.NetworkSample) {
	p.mu.Lock()
	defer p.mu.Unlock()

	// Find matching prediction within time window
	for i := len(p.predictions) - 1; i >= 0; i-- {
		pred := &p.predictions[i]
		timeDiff := timestamp.Sub(pred.Timestamp).Abs()

		// Match predictions within 1 minute window
		if timeDiff < time.Minute && pred.Actual == nil {
			pred.Actual = &actual

			// Calculate prediction error (normalized)
			bandwidthError := math.Abs(pred.Predicted.PredictedBandwidthMbps-actual.BandwidthMbps) /
				math.Max(actual.BandwidthMbps, 1.0)
			latencyError := math.Abs(pred.Predicted.PredictedLatencyMs-actual.LatencyMs) /
				math.Max(actual.LatencyMs, 1.0)

			pred.Error = (bandwidthError + latencyError) / 2.0

			// Update recent errors for adaptive confidence
			p.recentErrors = append(p.recentErrors, pred.Error)
			if len(p.recentErrors) > 50 {
				p.recentErrors = p.recentErrors[1:]
			}

			break
		}
	}
}

// GetMetrics returns predictor performance metrics
func (p *LSTMPredictorV3) GetMetrics() prediction.PredictorMetrics {
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

	accuracy := 1.0 - avgError
	// Ensure accuracy meets internet target (70%)
	if accuracy < p.targetAccuracy {
		// Log degraded performance
		// TODO: Add structured logging
	}

	return prediction.PredictorMetrics{
		ModelVersion:       p.modelVersion,
		LoadTime:           p.loadTime,
		InferenceCount:     p.inferenceCount,
		AvgInferenceMs:     float64(avgLatency.Microseconds()) / 1000.0,
		AvgPredictionError: avgError,
		MaxPredictionError: maxError,
		Accuracy:           accuracy,
	}
}

// ReloadModel reloads the ONNX model from disk
func (p *LSTMPredictorV3) ReloadModel(modelPath string) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	// Destroy old session
	if p.session != nil {
		p.session.Destroy()
	}

	// Create new session
	options, err := ort.NewSessionOptions()
	if err != nil {
		return fmt.Errorf("failed to create session options: %w", err)
	}
	defer options.Destroy()

	session, err := ort.NewDynamicAdvancedSession(modelPath,
		p.inputNames, p.outputNames, options)
	if err != nil {
		return fmt.Errorf("failed to create new session: %w", err)
	}

	p.session = session
	p.modelPath = modelPath
	p.loadTime = time.Now()
	p.modelVersion = fmt.Sprintf("v3.0-internet-%d", time.Now().Unix())

	return nil
}

// Close cleans up resources
func (p *LSTMPredictorV3) Close() {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.session != nil {
		p.session.Destroy()
	}
}

// GetSequenceLength returns the required sequence length
func (p *LSTMPredictorV3) GetSequenceLength() int {
	return p.sequenceLength
}

// GetTargetAccuracy returns the target accuracy for internet mode
func (p *LSTMPredictorV3) GetTargetAccuracy() float64 {
	return p.targetAccuracy
}

// GetRecentAccuracy returns the recent prediction accuracy
func (p *LSTMPredictorV3) GetRecentAccuracy() float64 {
	metrics := p.GetMetrics()
	return metrics.Accuracy
}
