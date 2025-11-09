package monitoring

import (
	"context"
	"fmt"
	"math"
	"time"

	"go.uber.org/zap"
)

// LSTMAutoencoderModel implements time-series anomaly detection using LSTM Autoencoder
type LSTMAutoencoderModel struct {
	modelPath      string
	threshold      float64
	logger         *zap.Logger

	// Time series buffer
	timeSeriesBuffer []*MetricVector
	windowSize       int

	// Simplified implementation without TensorFlow/ONNX
	// In production, use ONNX Runtime or TensorFlow
	weights        map[string][]float64
}

// NewLSTMAutoencoderModel creates a new LSTM Autoencoder model
func NewLSTMAutoencoderModel(modelPath string, logger *zap.Logger) (*LSTMAutoencoderModel, error) {
	if logger == nil {
		logger = zap.NewNop()
	}

	return &LSTMAutoencoderModel{
		modelPath:        modelPath,
		threshold:        0.05, // MSE threshold
		logger:           logger,
		windowSize:       10,    // 10 time steps
		timeSeriesBuffer: make([]*MetricVector, 0, 10),
		weights:          make(map[string][]float64),
	}, nil
}

// Detect detects anomalies in time series using LSTM Autoencoder
func (lam *LSTMAutoencoderModel) Detect(ctx context.Context, metrics *MetricVector) (*Anomaly, error) {
	// Add to buffer
	lam.timeSeriesBuffer = append(lam.timeSeriesBuffer, metrics)

	// Keep only last windowSize entries
	if len(lam.timeSeriesBuffer) > lam.windowSize {
		lam.timeSeriesBuffer = lam.timeSeriesBuffer[1:]
	}

	// Need full window to detect
	if len(lam.timeSeriesBuffer) < lam.windowSize {
		return nil, nil
	}

	// Calculate reconstruction error
	mse := lam.calculateReconstructionError()

	// Check if anomaly
	isAnomaly := mse > lam.threshold

	if !isAnomaly {
		return nil, nil
	}

	// Calculate confidence
	confidence := math.Min(mse/lam.threshold, 1.0)

	// Find which metric contributes most to error
	metricName, actualValue, expectedValue := lam.findMostAnomalousMetric()
	deviation := math.Abs(actualValue - expectedValue)

	severity := calculateSeverity(confidence, deviation/expectedValue*100)

	return &Anomaly{
		Timestamp:   time.Now(),
		MetricName:  metricName,
		Value:       actualValue,
		Expected:    expectedValue,
		Deviation:   deviation,
		Severity:    severity,
		Confidence:  confidence,
		ModelType:   "lstm_autoencoder",
		Description: fmt.Sprintf("LSTM Autoencoder detected time-series anomaly in %s (MSE: %.4f)", metricName, mse),
		Context: map[string]interface{}{
			"mse":         mse,
			"window_size": lam.windowSize,
		},
	}, nil
}

// Train trains the LSTM Autoencoder
func (lam *LSTMAutoencoderModel) Train(ctx context.Context, normalData []*MetricVector) error {
	if len(normalData) < lam.windowSize {
		return fmt.Errorf("insufficient training data: need at least %d samples", lam.windowSize)
	}

	lam.logger.Info("Training LSTM Autoencoder",
		zap.Int("samples", len(normalData)),
		zap.Int("window_size", lam.windowSize))

	// In a real implementation, this would train the LSTM model
	// For now, we'll calculate baseline statistics

	// Calculate mean and stddev for each metric
	for i := 0; i < 7; i++ {
		values := make([]float64, len(normalData))
		for j, mv := range normalData {
			values[j] = mv.ToSlice()[i]
		}

		m := mean(values)
		s := stddev(values)

		lam.weights[fmt.Sprintf("mean_%d", i)] = []float64{m}
		lam.weights[fmt.Sprintf("stddev_%d", i)] = []float64{s}
	}

	// Calculate reconstruction errors on training data to set threshold
	var errors []float64
	for i := lam.windowSize; i < len(normalData); i++ {
		lam.timeSeriesBuffer = normalData[i-lam.windowSize : i]
		err := lam.calculateReconstructionError()
		errors = append(errors, err)
	}

	// Set threshold at 95th percentile
	lam.threshold = percentile(errors, 0.95)

	lam.logger.Info("LSTM Autoencoder training completed",
		zap.Float64("threshold", lam.threshold))

	return nil
}

// Name returns the detector name
func (lam *LSTMAutoencoderModel) Name() string {
	return "lstm_autoencoder"
}

// calculateReconstructionError calculates the MSE reconstruction error
func (lam *LSTMAutoencoderModel) calculateReconstructionError() float64 {
	if len(lam.timeSeriesBuffer) < lam.windowSize {
		return 0
	}

	// Simplified reconstruction error calculation
	// In production, this would use the actual LSTM autoencoder

	totalError := 0.0
	count := 0

	// For each metric
	for i := 0; i < 7; i++ {
		// Extract time series for this metric
		series := make([]float64, lam.windowSize)
		for j, mv := range lam.timeSeriesBuffer {
			series[j] = mv.ToSlice()[i]
		}

		// Simple prediction: exponential moving average
		alpha := 0.3
		predicted := series[0]

		for j := 1; j < len(series); j++ {
			predicted = alpha*series[j] + (1-alpha)*predicted
			error := math.Pow(series[j]-predicted, 2)
			totalError += error
			count++
		}
	}

	return totalError / float64(count)
}

// findMostAnomalousMetric identifies which metric has highest reconstruction error
func (lam *LSTMAutoencoderModel) findMostAnomalousMetric() (string, float64, float64) {
	metricNames := []string{
		"bandwidth", "latency", "packet_loss", "jitter",
		"cpu_usage", "memory_usage", "error_rate",
	}

	maxError := 0.0
	maxMetric := metricNames[0]
	actualValue := 0.0
	expectedValue := 0.0

	// For each metric
	for i := 0; i < 7; i++ {
		// Extract time series
		series := make([]float64, lam.windowSize)
		for j, mv := range lam.timeSeriesBuffer {
			series[j] = mv.ToSlice()[i]
		}

		// Calculate prediction error for latest value
		alpha := 0.3
		predicted := series[0]

		for j := 1; j < len(series)-1; j++ {
			predicted = alpha*series[j] + (1-alpha)*predicted
		}

		// Final prediction
		finalPrediction := alpha*series[len(series)-2] + (1-alpha)*predicted
		error := math.Abs(series[len(series)-1] - finalPrediction)

		if error > maxError {
			maxError = error
			maxMetric = metricNames[i]
			actualValue = series[len(series)-1]
			expectedValue = finalPrediction
		}
	}

	return maxMetric, actualValue, expectedValue
}

// percentile calculates the nth percentile
func percentile(values []float64, p float64) float64 {
	if len(values) == 0 {
		return 0
	}

	sorted := make([]float64, len(values))
	copy(sorted, values)

	// Simple sort
	for i := 0; i < len(sorted); i++ {
		for j := i + 1; j < len(sorted); j++ {
			if sorted[i] > sorted[j] {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}

	idx := int(float64(len(sorted)) * p)
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}

	return sorted[idx]
}
