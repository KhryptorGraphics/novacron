package monitoring

import (
	"context"
	"fmt"
	"math"
	"sync"
	"time"

	"go.uber.org/zap"
)

// AnomalyDetector implements ML-based anomaly detection for DWCP v3 metrics
type AnomalyDetector struct {
	mu sync.RWMutex

	// Configuration
	config *AnomalyDetectorConfig

	// Statistical models
	bandwidthModel    *StatisticalModel
	latencyModel      *StatisticalModel
	compressionModel  *StatisticalModel
	consensusModel    *StatisticalModel

	// Anomaly tracking
	anomalies         []*Anomaly
	maxAnomalyHistory int

	// Alert callbacks
	alertCallbacks []AlertCallback

	logger *zap.Logger
	ctx    context.Context
	cancel context.CancelFunc
}

// AnomalyDetectorConfig configuration for anomaly detection
type AnomalyDetectorConfig struct {
	// Detection thresholds (in standard deviations)
	BandwidthThreshold    float64
	LatencyThreshold      float64
	CompressionThreshold  float64
	ConsensusThreshold    float64

	// Window sizes for moving averages
	ShortWindowSize  int // Fast detection (e.g., 10 samples)
	LongWindowSize   int // Baseline (e.g., 100 samples)

	// Alert settings
	EnableAlerts          bool
	AlertCooldownPeriod   time.Duration
	MinAnomalyConfidence  float64 // 0-1
}

// StatisticalModel tracks statistical properties for anomaly detection
type StatisticalModel struct {
	mu sync.RWMutex

	Name string

	// Historical data
	samples      []float64
	maxSamples   int

	// Statistical properties
	Mean         float64
	StdDev       float64
	Min          float64
	Max          float64

	// Moving averages
	ShortMA      float64 // Fast moving average
	LongMA       float64 // Slow moving average

	// Anomaly detection
	LastAnomaly  time.Time
	AnomalyCount int64
}

// Anomaly represents a detected anomaly
type Anomaly struct {
	Timestamp   time.Time
	Component   string
	Metric      string
	Value       float64
	Expected    float64
	Deviation   float64 // In standard deviations
	Confidence  float64 // 0-1
	Severity    string  // "low", "medium", "high", "critical"
	Description string
	Resolved    bool
}

// AlertCallback is called when an anomaly is detected
type AlertCallback func(anomaly *Anomaly)

// NewAnomalyDetector creates a new anomaly detector
func NewAnomalyDetector(config *AnomalyDetectorConfig, logger *zap.Logger) *AnomalyDetector {
	if config == nil {
		config = DefaultAnomalyDetectorConfig()
	}

	ctx, cancel := context.WithCancel(context.Background())

	detector := &AnomalyDetector{
		config:            config,
		bandwidthModel:    newStatisticalModel("bandwidth", config.LongWindowSize),
		latencyModel:      newStatisticalModel("latency", config.LongWindowSize),
		compressionModel:  newStatisticalModel("compression", config.LongWindowSize),
		consensusModel:    newStatisticalModel("consensus", config.LongWindowSize),
		anomalies:         make([]*Anomaly, 0),
		maxAnomalyHistory: 1000,
		alertCallbacks:    make([]AlertCallback, 0),
		logger:            logger,
		ctx:               ctx,
		cancel:            cancel,
	}

	// Start background anomaly cleanup
	go detector.cleanupLoop()

	return detector
}

// DefaultAnomalyDetectorConfig returns default configuration
func DefaultAnomalyDetectorConfig() *AnomalyDetectorConfig {
	return &AnomalyDetectorConfig{
		BandwidthThreshold:    3.0, // 3 sigma
		LatencyThreshold:      3.0, // 3 sigma
		CompressionThreshold:  2.5, // 2.5 sigma
		ConsensusThreshold:    2.5, // 2.5 sigma
		ShortWindowSize:       10,
		LongWindowSize:        100,
		EnableAlerts:          true,
		AlertCooldownPeriod:   5 * time.Minute,
		MinAnomalyConfidence:  0.80, // 80%
	}
}

// CheckBandwidth checks for bandwidth anomalies
func (ad *AnomalyDetector) CheckBandwidth(throughputMbps float64) *Anomaly {
	ad.bandwidthModel.addSample(throughputMbps)

	if !ad.bandwidthModel.hasEnoughSamples() {
		return nil
	}

	deviation := ad.bandwidthModel.calculateDeviation(throughputMbps)

	if math.Abs(deviation) > ad.config.BandwidthThreshold {
		confidence := ad.calculateConfidence(deviation, ad.config.BandwidthThreshold)
		if confidence >= ad.config.MinAnomalyConfidence {
			anomaly := &Anomaly{
				Timestamp:   time.Now(),
				Component:   "amst",
				Metric:      "bandwidth",
				Value:       throughputMbps,
				Expected:    ad.bandwidthModel.Mean,
				Deviation:   deviation,
				Confidence:  confidence,
				Severity:    ad.calculateSeverity(deviation),
				Description: fmt.Sprintf("Bandwidth anomaly: %.2f Mbps (expected %.2f ± %.2f)",
					throughputMbps, ad.bandwidthModel.Mean, ad.bandwidthModel.StdDev),
			}

			ad.recordAnomaly(anomaly)
			return anomaly
		}
	}

	return nil
}

// CheckLatency checks for latency anomalies
func (ad *AnomalyDetector) CheckLatency(latencyMs float64) *Anomaly {
	ad.latencyModel.addSample(latencyMs)

	if !ad.latencyModel.hasEnoughSamples() {
		return nil
	}

	deviation := ad.latencyModel.calculateDeviation(latencyMs)

	// Only alert on high latency (positive deviation)
	if deviation > ad.config.LatencyThreshold {
		confidence := ad.calculateConfidence(deviation, ad.config.LatencyThreshold)
		if confidence >= ad.config.MinAnomalyConfidence {
			anomaly := &Anomaly{
				Timestamp:   time.Now(),
				Component:   "amst",
				Metric:      "latency",
				Value:       latencyMs,
				Expected:    ad.latencyModel.Mean,
				Deviation:   deviation,
				Confidence:  confidence,
				Severity:    ad.calculateSeverity(deviation),
				Description: fmt.Sprintf("Latency spike: %.2f ms (expected %.2f ± %.2f)",
					latencyMs, ad.latencyModel.Mean, ad.latencyModel.StdDev),
			}

			ad.recordAnomaly(anomaly)
			return anomaly
		}
	}

	return nil
}

// CheckCompressionRatio checks for compression ratio anomalies
func (ad *AnomalyDetector) CheckCompressionRatio(ratio float64) *Anomaly {
	ad.compressionModel.addSample(ratio)

	if !ad.compressionModel.hasEnoughSamples() {
		return nil
	}

	deviation := ad.compressionModel.calculateDeviation(ratio)

	// Alert on unusually low compression (positive deviation from norm)
	if deviation > ad.config.CompressionThreshold {
		confidence := ad.calculateConfidence(deviation, ad.config.CompressionThreshold)
		if confidence >= ad.config.MinAnomalyConfidence {
			anomaly := &Anomaly{
				Timestamp:   time.Now(),
				Component:   "hde",
				Metric:      "compression_ratio",
				Value:       ratio,
				Expected:    ad.compressionModel.Mean,
				Deviation:   deviation,
				Confidence:  confidence,
				Severity:    ad.calculateSeverity(deviation),
				Description: fmt.Sprintf("Compression ratio anomaly: %.2fx (expected %.2f ± %.2f)",
					ratio, ad.compressionModel.Mean, ad.compressionModel.StdDev),
			}

			ad.recordAnomaly(anomaly)
			return anomaly
		}
	}

	return nil
}

// CheckConsensusLatency checks for consensus timeout anomalies
func (ad *AnomalyDetector) CheckConsensusLatency(latencyMs float64) *Anomaly {
	ad.consensusModel.addSample(latencyMs)

	if !ad.consensusModel.hasEnoughSamples() {
		return nil
	}

	deviation := ad.consensusModel.calculateDeviation(latencyMs)

	if deviation > ad.config.ConsensusThreshold {
		confidence := ad.calculateConfidence(deviation, ad.config.ConsensusThreshold)
		if confidence >= ad.config.MinAnomalyConfidence {
			anomaly := &Anomaly{
				Timestamp:   time.Now(),
				Component:   "acp",
				Metric:      "consensus_latency",
				Value:       latencyMs,
				Expected:    ad.consensusModel.Mean,
				Deviation:   deviation,
				Confidence:  confidence,
				Severity:    ad.calculateSeverity(deviation),
				Description: fmt.Sprintf("Consensus timeout anomaly: %.2f ms (expected %.2f ± %.2f)",
					latencyMs, ad.consensusModel.Mean, ad.consensusModel.StdDev),
			}

			ad.recordAnomaly(anomaly)
			return anomaly
		}
	}

	return nil
}

// RegisterAlertCallback registers a callback for anomaly alerts
func (ad *AnomalyDetector) RegisterAlertCallback(callback AlertCallback) {
	ad.mu.Lock()
	defer ad.mu.Unlock()
	ad.alertCallbacks = append(ad.alertCallbacks, callback)
}

// recordAnomaly records an anomaly and triggers alerts
func (ad *AnomalyDetector) recordAnomaly(anomaly *Anomaly) {
	ad.mu.Lock()
	defer ad.mu.Unlock()

	// Check cooldown period
	model := ad.getModelForAnomaly(anomaly)
	if model != nil && time.Since(model.LastAnomaly) < ad.config.AlertCooldownPeriod {
		ad.logger.Debug("Anomaly suppressed due to cooldown",
			zap.String("component", anomaly.Component),
			zap.String("metric", anomaly.Metric))
		return
	}

	// Add to history
	ad.anomalies = append(ad.anomalies, anomaly)
	if len(ad.anomalies) > ad.maxAnomalyHistory {
		ad.anomalies = ad.anomalies[1:]
	}

	// Update model
	if model != nil {
		model.LastAnomaly = time.Now()
		model.AnomalyCount++
	}

	// Log anomaly
	ad.logger.Warn("Anomaly detected",
		zap.String("component", anomaly.Component),
		zap.String("metric", anomaly.Metric),
		zap.Float64("value", anomaly.Value),
		zap.Float64("expected", anomaly.Expected),
		zap.Float64("deviation_sigma", anomaly.Deviation),
		zap.Float64("confidence", anomaly.Confidence),
		zap.String("severity", anomaly.Severity))

	// Trigger alerts
	if ad.config.EnableAlerts {
		for _, callback := range ad.alertCallbacks {
			go callback(anomaly)
		}
	}
}

// getModelForAnomaly returns the model associated with an anomaly
func (ad *AnomalyDetector) getModelForAnomaly(anomaly *Anomaly) *StatisticalModel {
	switch anomaly.Metric {
	case "bandwidth":
		return ad.bandwidthModel
	case "latency":
		return ad.latencyModel
	case "compression_ratio":
		return ad.compressionModel
	case "consensus_latency":
		return ad.consensusModel
	default:
		return nil
	}
}

// calculateConfidence calculates anomaly confidence based on deviation
func (ad *AnomalyDetector) calculateConfidence(deviation, threshold float64) float64 {
	// Confidence increases with deviation beyond threshold
	// 0.8 at threshold, approaches 1.0 as deviation increases
	if deviation < threshold {
		return 0.0
	}

	excess := math.Abs(deviation) - threshold
	confidence := 0.8 + (0.2 * (excess / threshold))
	if confidence > 1.0 {
		confidence = 1.0
	}

	return confidence
}

// calculateSeverity determines anomaly severity based on deviation
func (ad *AnomalyDetector) calculateSeverity(deviation float64) string {
	absDeviation := math.Abs(deviation)

	if absDeviation > 5.0 {
		return "critical"
	} else if absDeviation > 4.0 {
		return "high"
	} else if absDeviation > 3.0 {
		return "medium"
	}
	return "low"
}

// GetAnomalies returns recent anomalies
func (ad *AnomalyDetector) GetAnomalies(since time.Time) []*Anomaly {
	ad.mu.RLock()
	defer ad.mu.RUnlock()

	anomalies := make([]*Anomaly, 0)
	for _, anomaly := range ad.anomalies {
		if anomaly.Timestamp.After(since) {
			anomalies = append(anomalies, anomaly)
		}
	}

	return anomalies
}

// GetAnomalyStats returns anomaly statistics
func (ad *AnomalyDetector) GetAnomalyStats() map[string]interface{} {
	ad.mu.RLock()
	defer ad.mu.RUnlock()

	// Count by severity
	severityCounts := map[string]int{
		"low":      0,
		"medium":   0,
		"high":     0,
		"critical": 0,
	}

	// Count by component
	componentCounts := make(map[string]int)

	for _, anomaly := range ad.anomalies {
		severityCounts[anomaly.Severity]++
		componentCounts[anomaly.Component]++
	}

	return map[string]interface{}{
		"total_anomalies":      len(ad.anomalies),
		"severity_distribution": severityCounts,
		"component_distribution": componentCounts,
		"bandwidth_anomalies":  ad.bandwidthModel.AnomalyCount,
		"latency_anomalies":    ad.latencyModel.AnomalyCount,
		"compression_anomalies": ad.compressionModel.AnomalyCount,
		"consensus_anomalies":  ad.consensusModel.AnomalyCount,
	}
}

// ResolveAnomaly marks an anomaly as resolved
func (ad *AnomalyDetector) ResolveAnomaly(index int) error {
	ad.mu.Lock()
	defer ad.mu.Unlock()

	if index < 0 || index >= len(ad.anomalies) {
		return fmt.Errorf("invalid anomaly index: %d", index)
	}

	ad.anomalies[index].Resolved = true
	ad.logger.Info("Anomaly resolved",
		zap.Int("index", index),
		zap.String("component", ad.anomalies[index].Component),
		zap.String("metric", ad.anomalies[index].Metric))

	return nil
}

// cleanupLoop periodically cleans up old anomalies
func (ad *AnomalyDetector) cleanupLoop() {
	ticker := time.NewTicker(1 * time.Hour)
	defer ticker.Stop()

	for {
		select {
		case <-ad.ctx.Done():
			return
		case <-ticker.C:
			ad.cleanupOldAnomalies()
		}
	}
}

// cleanupOldAnomalies removes old resolved anomalies
func (ad *AnomalyDetector) cleanupOldAnomalies() {
	ad.mu.Lock()
	defer ad.mu.Unlock()

	cutoff := time.Now().Add(-24 * time.Hour)
	newAnomalies := make([]*Anomaly, 0)

	for _, anomaly := range ad.anomalies {
		// Keep if recent or unresolved
		if anomaly.Timestamp.After(cutoff) || !anomaly.Resolved {
			newAnomalies = append(newAnomalies, anomaly)
		}
	}

	ad.anomalies = newAnomalies
	ad.logger.Debug("Cleaned up old anomalies",
		zap.Int("remaining", len(ad.anomalies)))
}

// Close stops the anomaly detector
func (ad *AnomalyDetector) Close() error {
	ad.cancel()
	ad.logger.Info("Anomaly detector closed")
	return nil
}

// StatisticalModel methods

func newStatisticalModel(name string, maxSamples int) *StatisticalModel {
	return &StatisticalModel{
		Name:       name,
		samples:    make([]float64, 0, maxSamples),
		maxSamples: maxSamples,
	}
}

func (sm *StatisticalModel) addSample(value float64) {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	sm.samples = append(sm.samples, value)
	if len(sm.samples) > sm.maxSamples {
		sm.samples = sm.samples[1:]
	}

	// Update statistics
	sm.calculateStatistics()
}

func (sm *StatisticalModel) hasEnoughSamples() bool {
	sm.mu.RLock()
	defer sm.mu.RUnlock()
	return len(sm.samples) >= 10 // Minimum samples for reliable detection
}

func (sm *StatisticalModel) calculateStatistics() {
	if len(sm.samples) == 0 {
		return
	}

	// Calculate mean
	sum := 0.0
	sm.Min = sm.samples[0]
	sm.Max = sm.samples[0]

	for _, v := range sm.samples {
		sum += v
		if v < sm.Min {
			sm.Min = v
		}
		if v > sm.Max {
			sm.Max = v
		}
	}

	sm.Mean = sum / float64(len(sm.samples))

	// Calculate standard deviation
	variance := 0.0
	for _, v := range sm.samples {
		diff := v - sm.Mean
		variance += diff * diff
	}
	variance /= float64(len(sm.samples))
	sm.StdDev = math.Sqrt(variance)

	// Calculate moving averages
	shortWindowStart := len(sm.samples) - 10
	if shortWindowStart < 0 {
		shortWindowStart = 0
	}

	shortSum := 0.0
	for i := shortWindowStart; i < len(sm.samples); i++ {
		shortSum += sm.samples[i]
	}
	sm.ShortMA = shortSum / float64(len(sm.samples)-shortWindowStart)

	longSum := 0.0
	longWindowStart := len(sm.samples) - 100
	if longWindowStart < 0 {
		longWindowStart = 0
	}
	for i := longWindowStart; i < len(sm.samples); i++ {
		longSum += sm.samples[i]
	}
	sm.LongMA = longSum / float64(len(sm.samples)-longWindowStart)
}

func (sm *StatisticalModel) calculateDeviation(value float64) float64 {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	if sm.StdDev == 0 {
		return 0
	}

	return (value - sm.Mean) / sm.StdDev
}
