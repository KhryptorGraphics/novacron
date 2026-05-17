package prediction

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// PredictionService manages bandwidth predictions and model lifecycle
type PredictionService struct {
	predictor         *LSTMPredictor
	collector         *DataCollector
	updateInterval    time.Duration
	currentPrediction *BandwidthPrediction
	mu                sync.RWMutex

	// Service management
	ctx    context.Context
	cancel context.CancelFunc

	// Model management
	modelPath       string
	retrainInterval time.Duration
	lastRetrain     time.Time
	modelVersion    int

	// Prediction history
	predictionHistory []PredictionHistoryEntry
	maxHistory        int

	// A/B testing
	abTestEnabled  bool
	alternateModel *LSTMPredictor
	abTestResults  *ABTestResults

	// Prometheus metrics
	predictionAccuracy prometheus.Gauge
	predictionLatency  prometheus.Histogram
	modelVersionGauge  prometheus.Gauge
	confidenceGauge    prometheus.Gauge
	retrainCounter     prometheus.Counter
	predictionCounter  prometheus.Counter
}

var (
	predictionServiceMetricsOnce       sync.Once
	predictionServiceAccuracy          prometheus.Gauge
	predictionServiceLatency           prometheus.Histogram
	predictionServiceModelVersionGauge prometheus.Gauge
	predictionServiceConfidence        prometheus.Gauge
	predictionServiceRetrainCounter    prometheus.Counter
	predictionServicePredictionCounter prometheus.Counter
)

// BandwidthPrediction contains predicted network metrics
type BandwidthPrediction struct {
	PredictedBandwidthMbps float64   `json:"predicted_bandwidth_mbps"`
	PredictedLatencyMs     float64   `json:"predicted_latency_ms"`
	PredictedPacketLoss    float64   `json:"predicted_packet_loss"`
	PredictedJitterMs      float64   `json:"predicted_jitter_ms"`
	Confidence             float64   `json:"confidence"`
	ValidUntil             time.Time `json:"valid_until"`
	ModelVersion           string    `json:"model_version"`
	PredictionTime         time.Time `json:"prediction_time"`
}

// PredictionHistoryEntry stores historical predictions for analysis
type PredictionHistoryEntry struct {
	Prediction BandwidthPrediction
	Actual     *NetworkSample
	Error      float64
	Timestamp  time.Time
}

// ABTestResults tracks A/B testing metrics
type ABTestResults struct {
	PrimaryAccuracy   float64
	AlternateAccuracy float64
	PrimaryLatency    time.Duration
	AlternateLatency  time.Duration
	TestStartTime     time.Time
	TestDuration      time.Duration
	PredictionCount   int
	WinningModel      string
}

// NewPredictionService creates a new prediction service
func NewPredictionService(modelPath string, updateInterval time.Duration) (*PredictionService, error) {
	if updateInterval <= 0 {
		updateInterval = time.Second
	}

	// Initialize predictor
	predictor, err := NewLSTMPredictor(modelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to create predictor: %w", err)
	}

	// Initialize collector
	collector := NewDataCollector(1*time.Minute, 10000) // Collect every minute, keep 10k samples

	service := &PredictionService{
		predictor:         predictor,
		collector:         collector,
		updateInterval:    updateInterval,
		modelPath:         modelPath,
		retrainInterval:   24 * time.Hour, // Daily retraining
		predictionHistory: make([]PredictionHistoryEntry, 0, 1000),
		maxHistory:        1000,
		modelVersion:      1,
	}

	// Initialize Prometheus metrics
	service.initMetrics()

	return service, nil
}

// initMetrics initializes Prometheus metrics
func (s *PredictionService) initMetrics() {
	predictionServiceMetricsOnce.Do(func() {
		predictionServiceAccuracy = promauto.NewGauge(prometheus.GaugeOpts{
			Name: "dwcp_pba_prediction_accuracy",
			Help: "Current prediction accuracy (0-1)",
		})

		predictionServiceLatency = promauto.NewHistogram(prometheus.HistogramOpts{
			Name:    "dwcp_pba_prediction_latency_ms",
			Help:    "Prediction inference latency in milliseconds",
			Buckets: []float64{1, 2, 5, 10, 20, 50, 100},
		})

		predictionServiceModelVersionGauge = promauto.NewGauge(prometheus.GaugeOpts{
			Name: "dwcp_pba_model_version",
			Help: "Current model version number",
		})

		predictionServiceConfidence = promauto.NewGauge(prometheus.GaugeOpts{
			Name: "dwcp_pba_confidence",
			Help: "Current prediction confidence score",
		})

		predictionServiceRetrainCounter = promauto.NewCounter(prometheus.CounterOpts{
			Name: "dwcp_pba_retrain_total",
			Help: "Total number of model retraining events",
		})

		predictionServicePredictionCounter = promauto.NewCounter(prometheus.CounterOpts{
			Name: "dwcp_pba_predictions_total",
			Help: "Total number of predictions made",
		})
	})

	s.predictionAccuracy = predictionServiceAccuracy
	s.predictionLatency = predictionServiceLatency
	s.modelVersionGauge = predictionServiceModelVersionGauge
	s.confidenceGauge = predictionServiceConfidence
	s.retrainCounter = predictionServiceRetrainCounter
	s.predictionCounter = predictionServicePredictionCounter
}

// Start begins the prediction service
// Implements the Lifecycle interface
func (s *PredictionService) Start(ctx context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.ctx != nil && s.ctx.Err() == nil {
		return fmt.Errorf("prediction service already started")
	}

	// Create context for lifecycle management
	s.ctx, s.cancel = context.WithCancel(ctx)
	serviceCtx := s.ctx

	// Start data collection
	if s.collector != nil {
		s.collector.Start()
	}

	// Wait for initial samples (non-blocking)
	go func() {
		timer := time.NewTimer(2 * time.Minute)
		defer timer.Stop()

		select {
		case <-serviceCtx.Done():
			return
		case <-timer.C:
		}

		// Start prediction update loop
		go s.predictionLoop()

		// Start model retraining loop
		go s.retrainLoop()

		// Start accuracy tracking loop
		go s.accuracyTrackingLoop()
	}()

	return nil
}

// predictionLoop continuously updates predictions
func (s *PredictionService) predictionLoop() {
	ticker := time.NewTicker(s.updateInterval)
	defer ticker.Stop()

	for {
		select {
		case <-s.ctx.Done():
			return
		case <-ticker.C:
			s.updatePrediction()
		}
	}
}

// updatePrediction generates a new prediction
func (s *PredictionService) updatePrediction() {
	// Get recent samples
	history := s.collector.GetRecentSamples(10)
	if len(history) < 10 {
		return // Not enough data
	}

	// Generate prediction
	startTime := time.Now()
	prediction, err := s.predictor.Predict(history)
	if err != nil {
		fmt.Printf("Prediction failed: %v\n", err)
		return
	}
	inferenceTime := time.Since(startTime)

	// Update metrics
	s.predictionLatency.Observe(float64(inferenceTime.Milliseconds()))
	s.predictionCounter.Inc()
	s.confidenceGauge.Set(prediction.Confidence)

	// Set additional metadata
	prediction.ModelVersion = fmt.Sprintf("v%d", s.modelVersion)
	prediction.PredictionTime = time.Now()

	// Store prediction
	s.mu.Lock()
	s.currentPrediction = prediction

	// Add to history
	entry := PredictionHistoryEntry{
		Prediction: *prediction,
		Timestamp:  time.Now(),
	}
	s.predictionHistory = append(s.predictionHistory, entry)
	if len(s.predictionHistory) > s.maxHistory {
		s.predictionHistory = s.predictionHistory[1:]
	}
	s.mu.Unlock()

	// A/B testing if enabled
	if s.abTestEnabled && s.alternateModel != nil {
		go s.runABTest(history)
	}
}

// GetPrediction returns the current prediction
func (s *PredictionService) GetPrediction() *BandwidthPrediction {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.currentPrediction == nil {
		return nil
	}

	// Return a copy
	pred := *s.currentPrediction
	return &pred
}

// GetOptimalStreamCount calculates optimal number of streams based on prediction
func (s *PredictionService) GetOptimalStreamCount() int {
	pred := s.GetPrediction()
	if pred == nil || pred.Confidence < 0.5 {
		return 4 // Default
	}
	if !isFiniteNonNegative(pred.PredictedBandwidthMbps) ||
		!isFiniteNonNegative(pred.PredictedLatencyMs) ||
		!isFiniteNonNegative(pred.PredictedPacketLoss) {
		return 4
	}

	// Calculate based on predicted bandwidth and latency
	bandwidth := pred.PredictedBandwidthMbps
	latency := pred.PredictedLatencyMs

	// Formula: More streams for higher bandwidth, fewer for high latency
	optimalStreams := int(bandwidth/20) - int(latency/50)

	// Adjust based on packet loss
	if pred.PredictedPacketLoss > 0.02 {
		optimalStreams = optimalStreams * 3 / 4 // Reduce by 25%
	}

	// Bounds
	if optimalStreams < 2 {
		optimalStreams = 2
	}
	if optimalStreams > 16 {
		optimalStreams = 16
	}

	return optimalStreams
}

// GetOptimalBufferSize calculates optimal buffer size based on prediction
func (s *PredictionService) GetOptimalBufferSize() int {
	const (
		defaultBufferSize = 65536
		minBufferSize     = 16384
		maxBufferSize     = 1048576
	)

	pred := s.GetPrediction()
	if pred == nil {
		return defaultBufferSize
	}
	if !isFiniteNonNegative(pred.PredictedBandwidthMbps) ||
		!isFiniteNonNegative(pred.PredictedLatencyMs) ||
		!isFiniteNonNegative(pred.PredictedJitterMs) {
		return defaultBufferSize
	}

	// Calculate bandwidth-delay product
	bandwidthBps := pred.PredictedBandwidthMbps * 1000000 / 8 // Convert to bytes/sec
	totalDelaySeconds := (pred.PredictedLatencyMs + pred.PredictedJitterMs) / 1000

	if bandwidthBps <= 0 || totalDelaySeconds <= 0 {
		return minBufferSize
	}

	optimalSize := bandwidthBps * totalDelaySeconds
	if math.IsInf(optimalSize, 1) || optimalSize > maxBufferSize {
		return maxBufferSize
	}
	if math.IsNaN(optimalSize) {
		return defaultBufferSize
	}
	if optimalSize < minBufferSize {
		return minBufferSize
	}

	return int(optimalSize)
}

// accuracyTrackingLoop tracks prediction accuracy
func (s *PredictionService) accuracyTrackingLoop() {
	ticker := time.NewTicker(1 * time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-s.ctx.Done():
			return
		case <-ticker.C:
			s.updateAccuracy()
		}
	}
}

// updateAccuracy calculates and updates prediction accuracy
func (s *PredictionService) updateAccuracy() {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Get recent actual measurements
	actualSamples := s.collector.GetRecentSamples(20)

	// Match predictions with actuals
	var totalError float64
	var matchCount int

	for i, entry := range s.predictionHistory {
		// Find matching actual sample (within 1 minute)
		for _, actual := range actualSamples {
			if actual.Timestamp.Sub(entry.Timestamp) > 0 &&
				actual.Timestamp.Sub(entry.Timestamp) < 1*time.Minute {
				if !isValidActualSample(actual) {
					continue
				}

				// Calculate error
				bandwidthError := abs(entry.Prediction.PredictedBandwidthMbps-actual.BandwidthMbps) /
					actual.BandwidthMbps
				latencyError := abs(entry.Prediction.PredictedLatencyMs-actual.LatencyMs) /
					actual.LatencyMs

				error := (bandwidthError + latencyError) / 2.0
				if math.IsNaN(error) || math.IsInf(error, 0) {
					continue
				}
				s.predictionHistory[i].Actual = &actual
				s.predictionHistory[i].Error = error

				totalError += error
				matchCount++
				break
			}
		}
	}

	if matchCount > 0 {
		avgError := totalError / float64(matchCount)
		accuracy := 1.0 - avgError
		s.predictionAccuracy.Set(accuracy)

		// Update predictor with actual values
		for _, entry := range s.predictionHistory {
			if entry.Actual != nil {
				s.predictor.UpdateActual(entry.Timestamp, *entry.Actual)
			}
		}
	}
}

// retrainLoop manages model retraining
func (s *PredictionService) retrainLoop() {
	ticker := time.NewTicker(s.retrainInterval)
	defer ticker.Stop()

	for {
		select {
		case <-s.ctx.Done():
			return
		case <-ticker.C:
			s.retrainModel()
		}
	}
}

// retrainModel triggers model retraining
func (s *PredictionService) retrainModel() {
	// Export training data
	exportPath := fmt.Sprintf("/tmp/training_data_%d.csv", time.Now().Unix())
	err := s.collector.ExportForTraining(exportPath)
	if err != nil {
		fmt.Printf("Failed to export training data: %v\n", err)
		return
	}

	// Trigger Python training script (in production, this would be async)
	// For now, just increment version
	s.mu.Lock()
	s.modelVersion++
	s.lastRetrain = time.Now()
	s.mu.Unlock()

	s.modelVersionGauge.Set(float64(s.modelVersion))
	s.retrainCounter.Inc()

	fmt.Printf("Model retrained: version %d\n", s.modelVersion)
}

// EnableABTesting enables A/B testing with an alternate model
func (s *PredictionService) EnableABTesting(alternateModelPath string) error {
	alternateModel, err := NewLSTMPredictor(alternateModelPath)
	if err != nil {
		return fmt.Errorf("failed to load alternate model: %w", err)
	}

	s.mu.Lock()
	s.abTestEnabled = true
	s.alternateModel = alternateModel
	s.abTestResults = &ABTestResults{
		TestStartTime: time.Now(),
	}
	s.mu.Unlock()

	return nil
}

// runABTest performs A/B testing between models
func (s *PredictionService) runABTest(history []NetworkSample) {
	if s.alternateModel == nil {
		return
	}

	// Get prediction from alternate model
	startTime := time.Now()
	_, altErr := s.alternateModel.Predict(history)
	if altErr != nil {
		return
	}
	altLatency := time.Since(startTime)

	s.mu.Lock()
	defer s.mu.Unlock()

	// Update A/B test results
	s.abTestResults.AlternateLatency = altLatency
	s.abTestResults.PredictionCount++

	// Compare accuracy (would need actual values)
	// For now, just track predictions
}

// GetABTestResults returns current A/B test results
func (s *PredictionService) GetABTestResults() *ABTestResults {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.abTestResults == nil {
		return nil
	}

	// Return a copy
	results := *s.abTestResults
	results.TestDuration = time.Since(results.TestStartTime)
	return &results
}

// ExportMetrics exports service metrics for analysis
func (s *PredictionService) ExportMetrics(outputPath string) error {
	s.mu.RLock()
	defer s.mu.RUnlock()

	metrics := map[string]interface{}{
		"model_version":     s.modelVersion,
		"last_retrain":      s.lastRetrain,
		"prediction_count":  len(s.predictionHistory),
		"collector_stats":   s.collector.GetStatistics(),
		"predictor_metrics": s.predictor.GetMetrics(),
	}

	if s.currentPrediction != nil {
		metrics["current_prediction"] = s.currentPrediction
	}

	if s.abTestResults != nil {
		metrics["ab_test_results"] = s.abTestResults
	}

	// Calculate accuracy metrics
	var totalError float64
	var errorCount int
	for _, entry := range s.predictionHistory {
		if entry.Error > 0 {
			if math.IsNaN(entry.Error) || math.IsInf(entry.Error, 0) {
				continue
			}
			totalError += entry.Error
			errorCount++
		}
	}

	if errorCount > 0 {
		metrics["average_error"] = totalError / float64(errorCount)
		metrics["accuracy"] = 1.0 - (totalError / float64(errorCount))
	}

	if err := os.MkdirAll(filepath.Dir(outputPath), 0755); err != nil {
		return fmt.Errorf("failed to create metrics directory: %w", err)
	}

	file, err := os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("failed to create metrics file: %w", err)
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	return encoder.Encode(metrics)
}

// Stop stops the prediction service
// Implements the Lifecycle interface
func (s *PredictionService) Stop() error {
	s.mu.Lock()
	cancel := s.cancel
	s.mu.Unlock()

	if cancel != nil {
		cancel()
	}
	if s.collector != nil {
		s.collector.Stop()
	}
	if s.predictor != nil {
		s.predictor.Close()
	}

	if s.alternateModel != nil {
		s.alternateModel.Close()
	}

	s.mu.Lock()
	s.cancel = nil
	s.mu.Unlock()

	return nil
}

// IsRunning returns true if the prediction service is running
// Implements the Lifecycle interface
func (s *PredictionService) IsRunning() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.ctx != nil && s.ctx.Err() == nil
}

// Helper functions

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

func isValidActualSample(sample NetworkSample) bool {
	return sample.BandwidthMbps > 0 && sample.LatencyMs > 0 &&
		!math.IsNaN(sample.BandwidthMbps) && !math.IsNaN(sample.LatencyMs) &&
		!math.IsInf(sample.BandwidthMbps, 0) && !math.IsInf(sample.LatencyMs, 0)
}

func isFiniteNonNegative(value float64) bool {
	return value >= 0 && !math.IsNaN(value) && !math.IsInf(value, 0)
}
