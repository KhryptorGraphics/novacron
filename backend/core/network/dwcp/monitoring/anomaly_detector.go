package monitoring

import (
	"context"
	"fmt"
	"math"
	"sync"
	"time"

	"go.uber.org/zap"
)

// SeverityLevel represents the severity of an anomaly
type SeverityLevel int

const (
	SeverityInfo SeverityLevel = iota
	SeverityWarning
	SeverityCritical
)

func (s SeverityLevel) String() string {
	switch s {
	case SeverityInfo:
		return "info"
	case SeverityWarning:
		return "warning"
	case SeverityCritical:
		return "critical"
	default:
		return "unknown"
	}
}

// Anomaly represents a detected anomaly
type Anomaly struct {
	Timestamp   time.Time
	MetricName  string
	Value       float64
	Expected    float64
	Deviation   float64
	Severity    SeverityLevel
	Confidence  float64
	ModelType   string
	Description string
	Context     map[string]interface{}
}

// AnomalyResult represents the result of anomaly detection for a single metric
type AnomalyResult struct {
	MetricName  string
	IsAnomaly   bool
	Severity    SeverityLevel
	Confidence  float64
	Value       float64
	Expected    float64
	Deviation   float64
	Timestamp   time.Time
	Description string
}

// MetricVector represents a point in time with multiple metrics
type MetricVector struct {
	Timestamp   time.Time
	Bandwidth   float64 // Mbps
	Latency     float64 // ms
	PacketLoss  float64 // percentage
	Jitter      float64 // ms
	CPUUsage    float64 // percentage
	MemoryUsage float64 // percentage
	ErrorRate   float64 // percentage
}

func (mv *MetricVector) ToSlice() []float64 {
	return []float64{
		mv.Bandwidth,
		mv.Latency,
		mv.PacketLoss,
		mv.Jitter,
		mv.CPUUsage,
		mv.MemoryUsage,
		mv.ErrorRate,
	}
}

// Detector interface for all anomaly detection models
type Detector interface {
	Detect(ctx context.Context, metrics *MetricVector) (*Anomaly, error)
	Train(ctx context.Context, normalData []*MetricVector) error
	Name() string
}

// AnomalyDetector coordinates multiple detection models
type AnomalyDetector struct {
	isolationForest *IsolationForestModel
	lstmAutoencoder *LSTMAutoencoderModel
	zscoreDetector  *ZScoreDetector
	seasonalESD     *SeasonalESDDetector
	ensemble        *EnsembleDetector

	logger  *zap.Logger
	mu      sync.RWMutex
	enabled bool
	config  *DetectorConfig
}

// DetectorConfig holds configuration for the anomaly detector
type DetectorConfig struct {
	EnableIsolationForest bool
	EnableLSTM            bool
	EnableZScore          bool
	EnableSeasonalESD     bool

	EnsembleThreshold float64

	IsolationForestPath string
	LSTMModelPath       string

	ZScoreWindow    int
	ZScoreThreshold float64

	SeasonalPeriod       int
	SeasonalMaxAnomalies int

	AlertOnWarning  bool
	AlertOnCritical bool
}

// DefaultDetectorConfig returns default configuration
func DefaultDetectorConfig() *DetectorConfig {
	return &DetectorConfig{
		EnableIsolationForest: true,
		EnableLSTM:            true,
		EnableZScore:          true,
		EnableSeasonalESD:     true,

		EnsembleThreshold: 0.6,

		IsolationForestPath: "backend/core/network/dwcp/monitoring/models/isolation_forest.onnx",
		LSTMModelPath:       "backend/core/network/dwcp/monitoring/models/lstm_autoencoder.onnx",

		ZScoreWindow:    100,
		ZScoreThreshold: 3.0,

		SeasonalPeriod:       24, // 24 hours
		SeasonalMaxAnomalies: 10,

		AlertOnWarning:  true,
		AlertOnCritical: true,
	}
}

// NewAnomalyDetector creates a new anomaly detector
func NewAnomalyDetector(config *DetectorConfig, logger *zap.Logger) (*AnomalyDetector, error) {
	if config == nil {
		config = DefaultDetectorConfig()
	}

	if logger == nil {
		logger = zap.NewNop()
	}

	ad := &AnomalyDetector{
		logger:  logger,
		enabled: true,
		config:  config,
	}

	// Initialize detectors
	var err error

	if config.EnableIsolationForest {
		ad.isolationForest, err = NewIsolationForestModel(config.IsolationForestPath, logger)
		if err != nil {
			logger.Warn("Failed to initialize Isolation Forest", zap.Error(err))
		}
	}

	if config.EnableLSTM {
		ad.lstmAutoencoder, err = NewLSTMAutoencoderModel(config.LSTMModelPath, logger)
		if err != nil {
			logger.Warn("Failed to initialize LSTM Autoencoder", zap.Error(err))
		}
	}

	if config.EnableZScore {
		ad.zscoreDetector = NewZScoreDetector(config.ZScoreWindow, config.ZScoreThreshold, logger)
	}

	if config.EnableSeasonalESD {
		ad.seasonalESD = NewSeasonalESDDetector(config.SeasonalPeriod, config.SeasonalMaxAnomalies, logger)
	}

	// Initialize ensemble detector
	ad.ensemble = NewEnsembleDetector(config.EnsembleThreshold, logger)

	return ad, nil
}

// DetectAnomaly detects anomaly in a single metric value
func (ad *AnomalyDetector) DetectAnomaly(metricName string, value float64) *AnomalyResult {
	ad.mu.RLock()
	if !ad.enabled {
		ad.mu.RUnlock()
		return &AnomalyResult{
			MetricName: metricName,
			IsAnomaly:  false,
			Confidence: 0.0,
			Value:      value,
			Timestamp:  time.Now(),
		}
	}
	ad.mu.RUnlock()

	// Simple heuristic-based anomaly detection for single metrics
	// TODO: Integrate with full anomaly detection pipeline
	isAnomaly := false
	severity := SeverityInfo
	confidence := 0.0

	// Basic threshold-based detection
	// Typical ranges: bandwidth 0-10000 Mbps, latency 0-500ms, packet loss 0-100%
	switch metricName {
	case "bandwidth":
		if value > 10000 || value < 0 {
			isAnomaly = true
			severity = SeverityCritical
			confidence = 0.9
		}
	case "latency":
		if value > 500 || value < 0 {
			isAnomaly = true
			severity = SeverityWarning
			confidence = 0.8
		}
	case "packet_loss":
		if value > 100 || value < 0 {
			isAnomaly = true
			severity = SeverityCritical
			confidence = 0.95
		}
	}

	return &AnomalyResult{
		MetricName:  metricName,
		IsAnomaly:   isAnomaly,
		Severity:    severity,
		Confidence:  confidence,
		Value:       value,
		Expected:    0.0, // TODO: Calculate expected value
		Deviation:   0.0, // TODO: Calculate deviation
		Timestamp:   time.Now(),
		Description: fmt.Sprintf("Metric: %s, Value: %.2f", metricName, value),
	}
}

// Detect runs anomaly detection using all enabled detectors
func (ad *AnomalyDetector) Detect(ctx context.Context, metrics *MetricVector) ([]*Anomaly, error) {
	ad.mu.RLock()
	if !ad.enabled {
		ad.mu.RUnlock()
		return nil, nil
	}
	ad.mu.RUnlock()

	var anomalies []*Anomaly
	var detectorResults []DetectorResult

	// Run all detectors in parallel
	resultChan := make(chan DetectorResult, 4)
	errChan := make(chan error, 4)

	detectorCount := 0

	// Isolation Forest
	if ad.isolationForest != nil {
		detectorCount++
		go func() {
			anomaly, err := ad.isolationForest.Detect(ctx, metrics)
			if err != nil {
				errChan <- err
				return
			}
			resultChan <- DetectorResult{
				Detector: "isolation_forest",
				Anomaly:  anomaly,
				Weight:   0.3,
			}
		}()
	}

	// LSTM Autoencoder
	if ad.lstmAutoencoder != nil {
		detectorCount++
		go func() {
			anomaly, err := ad.lstmAutoencoder.Detect(ctx, metrics)
			if err != nil {
				errChan <- err
				return
			}
			resultChan <- DetectorResult{
				Detector: "lstm_autoencoder",
				Anomaly:  anomaly,
				Weight:   0.3,
			}
		}()
	}

	// Z-Score
	if ad.zscoreDetector != nil {
		detectorCount++
		go func() {
			anomaly, err := ad.zscoreDetector.Detect(ctx, metrics)
			if err != nil {
				errChan <- err
				return
			}
			resultChan <- DetectorResult{
				Detector: "zscore",
				Anomaly:  anomaly,
				Weight:   0.2,
			}
		}()
	}

	// Seasonal ESD
	if ad.seasonalESD != nil {
		detectorCount++
		go func() {
			anomaly, err := ad.seasonalESD.Detect(ctx, metrics)
			if err != nil {
				errChan <- err
				return
			}
			resultChan <- DetectorResult{
				Detector: "seasonal_esd",
				Anomaly:  anomaly,
				Weight:   0.2,
			}
		}()
	}

	// Collect results
	for i := 0; i < detectorCount; i++ {
		select {
		case result := <-resultChan:
			detectorResults = append(detectorResults, result)
			if result.Anomaly != nil {
				anomalies = append(anomalies, result.Anomaly)
			}
		case err := <-errChan:
			ad.logger.Error("Detector error", zap.Error(err))
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}

	// Run ensemble detection
	ensembleAnomaly := ad.ensemble.Aggregate(detectorResults, metrics)
	if ensembleAnomaly != nil {
		anomalies = append(anomalies, ensembleAnomaly)
	}

	return anomalies, nil
}

// Train trains all trainable detectors
func (ad *AnomalyDetector) Train(ctx context.Context, normalData []*MetricVector) error {
	ad.mu.Lock()
	defer ad.mu.Unlock()

	var errors []error

	// Train Z-Score detector
	if ad.zscoreDetector != nil {
		if err := ad.zscoreDetector.Train(ctx, normalData); err != nil {
			errors = append(errors, fmt.Errorf("zscore training failed: %w", err))
		}
	}

	// Train Seasonal ESD
	if ad.seasonalESD != nil {
		if err := ad.seasonalESD.Train(ctx, normalData); err != nil {
			errors = append(errors, fmt.Errorf("seasonal esd training failed: %w", err))
		}
	}

	if len(errors) > 0 {
		return fmt.Errorf("training errors: %v", errors)
	}

	ad.logger.Info("Anomaly detector training completed",
		zap.Int("samples", len(normalData)))

	return nil
}

// Enable enables anomaly detection
func (ad *AnomalyDetector) Enable() {
	ad.mu.Lock()
	defer ad.mu.Unlock()
	ad.enabled = true
	ad.logger.Info("Anomaly detection enabled")
}

// Disable disables anomaly detection
func (ad *AnomalyDetector) Disable() {
	ad.mu.Lock()
	defer ad.mu.Unlock()
	ad.enabled = false
	ad.logger.Info("Anomaly detection disabled")
}

// calculateSeverity determines severity based on confidence and deviation
func calculateSeverity(confidence, deviation float64) SeverityLevel {
	if confidence > 0.9 && deviation > 5.0 {
		return SeverityCritical
	}
	if confidence > 0.7 && deviation > 3.0 {
		return SeverityWarning
	}
	return SeverityInfo
}

// mean calculates the mean of a slice
func mean(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}

// stddev calculates the standard deviation
func stddev(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	m := mean(values)
	variance := 0.0
	for _, v := range values {
		variance += math.Pow(v-m, 2)
	}
	return math.Sqrt(variance / float64(len(values)))
}

// median calculates the median value
func median(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}

	sorted := make([]float64, len(values))
	copy(sorted, values)

	// Simple bubble sort for small arrays
	for i := 0; i < len(sorted); i++ {
		for j := i + 1; j < len(sorted); j++ {
			if sorted[i] > sorted[j] {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}

	mid := len(sorted) / 2
	if len(sorted)%2 == 0 {
		return (sorted[mid-1] + sorted[mid]) / 2
	}
	return sorted[mid]
}
