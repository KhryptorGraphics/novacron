package prediction

import (
	"fmt"
	"log"
	"math"
	"os"
	"sync"
	"time"

	ort "github.com/yalue/onnxruntime_go"
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

	inference bandwidthInferenceSession
}

type bandwidthInferenceSession interface {
	Run(input []float32) ([]float32, error)
	Destroy() error
}

type onnxBandwidthSession struct {
	session      *ort.AdvancedSession
	inputTensor  *ort.Tensor[float32]
	outputTensor *ort.Tensor[float32]
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
			session, err := newONNXBandwidthSession(modelPath, predictor.sequenceLength, predictor.featureCount, predictor.outputCount)
			if err != nil {
				log.Printf("Warning: Could not load LSTM bandwidth model from %s, using history forecast fallback: %v", modelPath, err)
			} else {
				predictor.inference = session
				predictor.modelLoaded = true
			}
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
	inputData, err := p.prepareInput(history)
	if err != nil {
		return nil, fmt.Errorf("failed to prepare input: %w", err)
	}

	outputData, modelUsed, err := p.runInference(inputData)
	if err != nil {
		return nil, fmt.Errorf("failed to run inference: %w", err)
	}

	prediction, err := p.parseOutput(outputData, history, modelUsed)
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

func (p *LSTMPredictor) runInference(input []float32) ([]float32, bool, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if !p.modelLoaded || p.inference == nil {
		return nil, false, nil
	}
	if len(input) != p.sequenceLength*p.featureCount {
		return nil, false, fmt.Errorf("invalid LSTM input vector length %d", len(input))
	}

	output, err := p.inference.Run(input)
	if err != nil {
		return nil, false, err
	}

	return output, true, nil
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

// parseOutput converts normalized model output to a bandwidth prediction.
func (p *LSTMPredictor) parseOutput(output []float32, history []NetworkSample, modelUsed bool) (*BandwidthPrediction, error) {
	var prediction *BandwidthPrediction
	if modelUsed {
		if len(output) != p.outputCount {
			return nil, fmt.Errorf("invalid LSTM output length %d", len(output))
		}
		for i, value := range output {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return nil, fmt.Errorf("invalid LSTM output at %d: %f", i, value)
			}
		}

		prediction = &BandwidthPrediction{
			PredictedBandwidthMbps: math.Max(0, float64(output[0])*1000.0),
			PredictedLatencyMs:     math.Max(0, float64(output[1])*100.0),
			PredictedPacketLoss:    math.Max(0, math.Min(1, float64(output[2]))),
			PredictedJitterMs:      math.Max(0, float64(output[3])*50.0),
			ValidUntil:             time.Now().Add(15 * time.Minute),
		}
	} else {
		prediction = p.forecastFromHistory(history)
	}

	prediction.Confidence = p.calculateConfidence(prediction)
	if !modelUsed {
		prediction.Confidence = math.Min(prediction.Confidence, 0.5)
	}

	return prediction, nil
}

func newONNXBandwidthSession(modelPath string, sequenceLength, featureCount, outputCount int) (*onnxBandwidthSession, error) {
	if err := ensurePBAONNXEnvironment(); err != nil {
		return nil, err
	}

	inputs, outputs, err := ort.GetInputOutputInfo(modelPath)
	if err != nil {
		return nil, fmt.Errorf("inspect ONNX model: %w", err)
	}
	if len(inputs) != 1 {
		return nil, fmt.Errorf("expected one ONNX input, got %d", len(inputs))
	}
	if len(outputs) != 1 {
		return nil, fmt.Errorf("expected one ONNX output, got %d", len(outputs))
	}
	if inputs[0].OrtValueType != ort.ONNXTypeTensor || inputs[0].DataType != ort.TensorElementDataTypeFloat {
		return nil, fmt.Errorf("expected float tensor input, got %s", inputs[0].String())
	}
	if outputs[0].OrtValueType != ort.ONNXTypeTensor || outputs[0].DataType != ort.TensorElementDataTypeFloat {
		return nil, fmt.Errorf("expected float tensor output, got %s", outputs[0].String())
	}
	if !lstmInputShapeAllows(inputs[0].Dimensions, sequenceLength, featureCount) {
		return nil, fmt.Errorf("expected LSTM input shape compatible with [%d,%d], got %s", sequenceLength, featureCount, inputs[0].Dimensions.String())
	}
	if !lstmOutputShapeAllows(outputs[0].Dimensions, outputCount) {
		return nil, fmt.Errorf("expected LSTM output shape compatible with %d targets, got %s", outputCount, outputs[0].Dimensions.String())
	}

	inputTensor, err := ort.NewTensor[float32](ort.Shape{1, int64(sequenceLength), int64(featureCount)}, make([]float32, sequenceLength*featureCount))
	if err != nil {
		return nil, fmt.Errorf("create LSTM input tensor: %w", err)
	}
	outputTensor, err := ort.NewTensor[float32](ort.Shape{1, int64(outputCount)}, make([]float32, outputCount))
	if err != nil {
		_ = inputTensor.Destroy()
		return nil, fmt.Errorf("create LSTM output tensor: %w", err)
	}

	session, err := ort.NewAdvancedSession(
		modelPath,
		[]string{inputs[0].Name},
		[]string{outputs[0].Name},
		[]ort.Value{inputTensor},
		[]ort.Value{outputTensor},
		nil,
	)
	if err != nil {
		_ = inputTensor.Destroy()
		_ = outputTensor.Destroy()
		return nil, fmt.Errorf("create LSTM ONNX session: %w", err)
	}

	return &onnxBandwidthSession{
		session:      session,
		inputTensor:  inputTensor,
		outputTensor: outputTensor,
	}, nil
}

var pbaONNXEnvMu sync.Mutex

func ensurePBAONNXEnvironment() error {
	pbaONNXEnvMu.Lock()
	defer pbaONNXEnvMu.Unlock()

	if ort.IsInitialized() {
		return nil
	}
	if libraryPath := os.Getenv("ONNXRUNTIME_SHARED_LIBRARY_PATH"); libraryPath != "" {
		ort.SetSharedLibraryPath(libraryPath)
	}
	return ort.InitializeEnvironment()
}

func lstmInputShapeAllows(shape ort.Shape, sequenceLength, featureCount int) bool {
	if len(shape) < 2 {
		return false
	}
	return shape[len(shape)-2] == int64(sequenceLength) && shape[len(shape)-1] == int64(featureCount)
}

func lstmOutputShapeAllows(shape ort.Shape, outputCount int) bool {
	if len(shape) == 0 {
		return false
	}
	return shape[len(shape)-1] == int64(outputCount)
}

func (session *onnxBandwidthSession) Run(input []float32) ([]float32, error) {
	copy(session.inputTensor.GetData(), input)
	session.outputTensor.ZeroContents()

	if err := session.session.Run(); err != nil {
		return nil, fmt.Errorf("run LSTM ONNX session: %w", err)
	}

	output := make([]float32, len(session.outputTensor.GetData()))
	copy(output, session.outputTensor.GetData())
	return output, nil
}

func (session *onnxBandwidthSession) Destroy() error {
	var err error
	if session.session != nil {
		if e := session.session.Destroy(); e != nil {
			err = e
		}
		session.session = nil
	}
	if session.inputTensor != nil {
		if e := session.inputTensor.Destroy(); err == nil && e != nil {
			err = e
		}
		session.inputTensor = nil
	}
	if session.outputTensor != nil {
		if e := session.outputTensor.Destroy(); err == nil && e != nil {
			err = e
		}
		session.outputTensor = nil
	}
	return err
}

func (p *LSTMPredictor) forecastFromHistory(history []NetworkSample) *BandwidthPrediction {
	startIdx := len(history) - p.sequenceLength
	recent := history[startIdx:]
	first := recent[0]
	last := recent[len(recent)-1]

	var bandwidth, latency, packetLoss, jitter float64
	for _, sample := range recent {
		bandwidth += sample.BandwidthMbps
		latency += sample.LatencyMs
		packetLoss += sample.PacketLoss
		jitter += sample.JitterMs
	}

	count := float64(len(recent))
	trendScale := 1.0 / count
	bandwidthTrend := (last.BandwidthMbps - first.BandwidthMbps) * trendScale
	latencyTrend := (last.LatencyMs - first.LatencyMs) * trendScale
	packetLossTrend := (last.PacketLoss - first.PacketLoss) * trendScale
	jitterTrend := (last.JitterMs - first.JitterMs) * trendScale

	return &BandwidthPrediction{
		PredictedBandwidthMbps: math.Max(0, bandwidth/count+bandwidthTrend),
		PredictedLatencyMs:     math.Max(0, latency/count+latencyTrend),
		PredictedPacketLoss:    math.Max(0, packetLoss/count+packetLossTrend),
		PredictedJitterMs:      math.Max(0, jitter/count+jitterTrend),
		ValidUntil:             time.Now().Add(15 * time.Minute),
	}
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
	var finiteErrors int
	for _, err := range recentErrors {
		if math.IsNaN(err) || math.IsInf(err, 0) {
			continue
		}
		totalError += math.Abs(err)
		finiteErrors++
	}
	if finiteErrors == 0 {
		return 0.5
	}
	avgError := totalError / float64(finiteErrors)

	// Convert error to confidence (lower error = higher confidence)
	confidence := math.Max(0.0, math.Min(1.0, 1.0-(avgError/0.2)))

	return confidence
}

// UpdateActual updates prediction record with actual values for accuracy tracking
func (p *LSTMPredictor) UpdateActual(timestamp time.Time, actual NetworkSample) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if actual.BandwidthMbps <= 0 || actual.LatencyMs <= 0 ||
		math.IsNaN(actual.BandwidthMbps) || math.IsNaN(actual.LatencyMs) ||
		math.IsInf(actual.BandwidthMbps, 0) || math.IsInf(actual.LatencyMs, 0) {
		return
	}

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
			if math.IsNaN(pred.Error) || math.IsInf(pred.Error, 0) {
				continue
			}
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
	if _, err := os.Stat(modelPath); err != nil {
		return fmt.Errorf("failed to load model %q: %w", modelPath, err)
	}
	session, err := newONNXBandwidthSession(modelPath, p.sequenceLength, p.featureCount, p.outputCount)
	if err != nil {
		return fmt.Errorf("failed to load model %q: %w", modelPath, err)
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	if p.inference != nil {
		_ = p.inference.Destroy()
	}
	p.inference = session
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

	if p.inference != nil {
		_ = p.inference.Destroy()
		p.inference = nil
	}
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
