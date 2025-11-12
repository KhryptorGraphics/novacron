package prediction

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/prediction"
	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/upgrade"
)

// PBAv3 is the v3 Predictive Bandwidth Allocation engine
// Supports both datacenter and internet modes with hybrid switching
type PBAv3 struct {
	mu sync.RWMutex

	// Mode detection
	modeDetector *upgrade.ModeDetector
	currentMode  upgrade.NetworkMode

	// Predictors for each mode
	datacenterPredictor *prediction.LSTMPredictor // v1 predictor (optimized for datacenter)
	internetPredictor   *LSTMPredictorV3           // v3 predictor (optimized for internet)

	// Configuration
	config *PBAv3Config

	// Metrics
	metrics *PBAv3Metrics

	// Historical data management
	datacenterHistory *BandwidthHistory
	internetHistory   *BandwidthHistory
}

// PBAv3Config configuration for PBA v3
type PBAv3Config struct {
	// Mode settings
	EnableHybridMode    bool
	EnableModeDetection bool
	DefaultMode         upgrade.NetworkMode

	// Model paths
	DatacenterModelPath string
	InternetModelPath   string

	// Prediction parameters
	DatacenterSequenceLength int
	InternetSequenceLength   int
	PredictionHorizon        time.Duration

	// Performance targets
	DatacenterAccuracyTarget float64 // 0.85 (85%)
	InternetAccuracyTarget   float64 // 0.70 (70%)
	PredictionLatencyTarget  time.Duration
}

// PBAv3Metrics tracks performance metrics
type PBAv3Metrics struct {
	mu sync.RWMutex

	// Prediction counts
	TotalPredictions        uint64
	DatacenterPredictions   uint64
	InternetPredictions     uint64
	HybridModeSwitches      uint64

	// Accuracy tracking
	DatacenterAccuracy      float64
	InternetAccuracy        float64
	OverallAccuracy         float64

	// Latency tracking
	AvgPredictionLatency    time.Duration
	MaxPredictionLatency    time.Duration
	P95PredictionLatency    time.Duration

	// Mode-specific latency
	DatacenterAvgLatency    time.Duration
	InternetAvgLatency      time.Duration
}

// BandwidthHistory maintains historical bandwidth data for predictions
type BandwidthHistory struct {
	mu      sync.RWMutex
	samples []prediction.NetworkSample
	maxSize int
}

// DefaultPBAv3Config returns default configuration
func DefaultPBAv3Config() *PBAv3Config {
	return &PBAv3Config{
		EnableHybridMode:         true,
		EnableModeDetection:      true,
		DefaultMode:              upgrade.ModeHybrid,
		DatacenterModelPath:      "/var/lib/dwcp/models/datacenter_bandwidth_predictor.onnx",
		InternetModelPath:        "/var/lib/dwcp/models/internet_bandwidth_predictor.onnx",
		DatacenterSequenceLength: 10,  // Shorter for stable networks
		InternetSequenceLength:   60,  // Longer for variable networks
		PredictionHorizon:        15 * time.Minute,
		DatacenterAccuracyTarget: 0.85, // 85%
		InternetAccuracyTarget:   0.70, // 70%
		PredictionLatencyTarget:  100 * time.Millisecond,
	}
}

// NewPBAv3 creates a new PBA v3 instance
func NewPBAv3(config *PBAv3Config) (*PBAv3, error) {
	if config == nil {
		config = DefaultPBAv3Config()
	}

	pba := &PBAv3{
		config:            config,
		metrics:           &PBAv3Metrics{},
		modeDetector:      upgrade.NewModeDetector(),
		currentMode:       config.DefaultMode,
		datacenterHistory: NewBandwidthHistory(config.DatacenterSequenceLength),
		internetHistory:   NewBandwidthHistory(config.InternetSequenceLength),
	}

	// Initialize datacenter predictor (v1)
	var err error
	pba.datacenterPredictor, err = prediction.NewLSTMPredictor(config.DatacenterModelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize datacenter predictor: %w", err)
	}

	// Initialize internet predictor (v3)
	pba.internetPredictor, err = NewLSTMPredictorV3(config.InternetModelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize internet predictor: %w", err)
	}

	// Start mode detection if enabled
	if config.EnableModeDetection {
		go pba.autoModeDetection(context.Background())
	}

	return pba, nil
}

// PredictBandwidth predicts bandwidth using the appropriate mode predictor
func (p *PBAv3) PredictBandwidth(ctx context.Context) (*prediction.BandwidthPrediction, error) {
	startTime := time.Now()

	// Determine which mode to use
	mode := p.selectMode()

	var pred *prediction.BandwidthPrediction
	var err error

	switch mode {
	case upgrade.ModeDatacenter:
		pred, err = p.predictDatacenter(ctx)
	case upgrade.ModeInternet:
		pred, err = p.predictInternet(ctx)
	case upgrade.ModeHybrid:
		pred, err = p.predictHybrid(ctx)
	default:
		return nil, fmt.Errorf("unknown network mode: %v", mode)
	}

	if err != nil {
		return nil, err
	}

	// Update metrics
	latency := time.Since(startTime)
	p.updateMetrics(mode, latency, pred)

	return pred, nil
}

// predictDatacenter uses v1 predictor optimized for datacenter
func (p *PBAv3) predictDatacenter(ctx context.Context) (*prediction.BandwidthPrediction, error) {
	p.mu.RLock()
	history := p.datacenterHistory.GetRecent(p.config.DatacenterSequenceLength)
	p.mu.RUnlock()

	if len(history) < p.config.DatacenterSequenceLength {
		return nil, fmt.Errorf("insufficient datacenter history: need %d, have %d",
			p.config.DatacenterSequenceLength, len(history))
	}

	pred, err := p.datacenterPredictor.Predict(history)
	if err != nil {
		return nil, fmt.Errorf("datacenter prediction failed: %w", err)
	}

	// Adjust confidence based on mode
	pred.Confidence = p.adjustConfidenceForMode(pred.Confidence, upgrade.ModeDatacenter)

	return pred, nil
}

// predictInternet uses v3 predictor optimized for internet
func (p *PBAv3) predictInternet(ctx context.Context) (*prediction.BandwidthPrediction, error) {
	p.mu.RLock()
	history := p.internetHistory.GetRecent(p.config.InternetSequenceLength)
	p.mu.RUnlock()

	if len(history) < p.config.InternetSequenceLength {
		return nil, fmt.Errorf("insufficient internet history: need %d, have %d",
			p.config.InternetSequenceLength, len(history))
	}

	pred, err := p.internetPredictor.Predict(history)
	if err != nil {
		return nil, fmt.Errorf("internet prediction failed: %w", err)
	}

	// Adjust confidence based on mode
	pred.Confidence = p.adjustConfidenceForMode(pred.Confidence, upgrade.ModeInternet)

	return pred, nil
}

// predictHybrid intelligently selects between datacenter and internet predictors
func (p *PBAv3) predictHybrid(ctx context.Context) (*prediction.BandwidthPrediction, error) {
	// Try both predictors and use confidence-weighted ensemble
	dcPred, dcErr := p.predictDatacenter(ctx)
	inetPred, inetErr := p.predictInternet(ctx)

	// If only one succeeds, use it
	if dcErr != nil && inetErr == nil {
		return inetPred, nil
	}
	if inetErr != nil && dcErr == nil {
		return dcPred, nil
	}
	if dcErr != nil && inetErr != nil {
		return nil, fmt.Errorf("both predictors failed: dc=%v, inet=%v", dcErr, inetErr)
	}

	// Both succeeded - use confidence-weighted ensemble
	return p.ensemblePredictions(dcPred, inetPred), nil
}

// ensemblePredictions combines predictions from multiple models
func (p *PBAv3) ensemblePredictions(
	dcPred, inetPred *prediction.BandwidthPrediction,
) *prediction.BandwidthPrediction {
	// Weight by confidence
	totalConfidence := dcPred.Confidence + inetPred.Confidence
	dcWeight := dcPred.Confidence / totalConfidence
	inetWeight := inetPred.Confidence / totalConfidence

	// Weighted average of predictions
	ensembleBandwidth := dcPred.PredictedBandwidthMbps*dcWeight +
		inetPred.PredictedBandwidthMbps*inetWeight

	ensembleLatency := dcPred.PredictedLatencyMs*dcWeight +
		inetPred.PredictedLatencyMs*inetWeight

	ensemblePacketLoss := dcPred.PredictedPacketLoss*dcWeight +
		inetPred.PredictedPacketLoss*inetWeight

	ensembleJitter := dcPred.PredictedJitterMs*dcWeight +
		inetPred.PredictedJitterMs*inetWeight

	// Take max confidence
	ensembleConfidence := dcPred.Confidence
	if inetPred.Confidence > dcPred.Confidence {
		ensembleConfidence = inetPred.Confidence
	}

	return &prediction.BandwidthPrediction{
		PredictedBandwidthMbps: ensembleBandwidth,
		PredictedLatencyMs:     ensembleLatency,
		PredictedPacketLoss:    ensemblePacketLoss,
		PredictedJitterMs:      ensembleJitter,
		Confidence:             ensembleConfidence,
		ValidUntil:             time.Now().Add(p.config.PredictionHorizon),
	}
}

// selectMode determines which predictor to use
func (p *PBAv3) selectMode() upgrade.NetworkMode {
	if !p.config.EnableHybridMode {
		return p.config.DefaultMode
	}

	p.mu.RLock()
	mode := p.currentMode
	p.mu.RUnlock()

	return mode
}

// adjustConfidenceForMode adjusts confidence based on mode-specific targets
func (p *PBAv3) adjustConfidenceForMode(
	confidence float64,
	mode upgrade.NetworkMode,
) float64 {
	switch mode {
	case upgrade.ModeDatacenter:
		// Datacenter target: 85% accuracy
		// Scale confidence relative to target
		return confidence * (confidence / p.config.DatacenterAccuracyTarget)
	case upgrade.ModeInternet:
		// Internet target: 70% accuracy
		return confidence * (confidence / p.config.InternetAccuracyTarget)
	default:
		return confidence
	}
}

// AddSample adds a network sample to history for both modes
func (p *PBAv3) AddSample(sample prediction.NetworkSample) {
	p.mu.Lock()
	defer p.mu.Unlock()

	// Add to both histories
	p.datacenterHistory.Add(sample)
	p.internetHistory.Add(sample)

	// Update predictors with actual values
	p.datacenterPredictor.UpdateActual(sample.Timestamp, sample)
	p.internetPredictor.UpdateActual(sample.Timestamp, sample)
}

// autoModeDetection runs mode detection loop
func (p *PBAv3) autoModeDetection(ctx context.Context) {
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			newMode := p.modeDetector.DetectMode(ctx)
			p.mu.Lock()
			if newMode != p.currentMode {
				p.currentMode = newMode
				p.metrics.mu.Lock()
				p.metrics.HybridModeSwitches++
				p.metrics.mu.Unlock()
			}
			p.mu.Unlock()
		}
	}
}

// updateMetrics updates performance metrics
func (p *PBAv3) updateMetrics(
	mode upgrade.NetworkMode,
	latency time.Duration,
	pred *prediction.BandwidthPrediction,
) {
	p.metrics.mu.Lock()
	defer p.metrics.mu.Unlock()

	p.metrics.TotalPredictions++

	switch mode {
	case upgrade.ModeDatacenter:
		p.metrics.DatacenterPredictions++
		p.metrics.DatacenterAvgLatency = p.updateAvgDuration(
			p.metrics.DatacenterAvgLatency,
			latency,
			p.metrics.DatacenterPredictions,
		)
	case upgrade.ModeInternet:
		p.metrics.InternetPredictions++
		p.metrics.InternetAvgLatency = p.updateAvgDuration(
			p.metrics.InternetAvgLatency,
			latency,
			p.metrics.InternetPredictions,
		)
	}

	// Update overall latency stats
	p.metrics.AvgPredictionLatency = p.updateAvgDuration(
		p.metrics.AvgPredictionLatency,
		latency,
		p.metrics.TotalPredictions,
	)

	if latency > p.metrics.MaxPredictionLatency {
		p.metrics.MaxPredictionLatency = latency
	}
}

// updateAvgDuration updates running average duration
func (p *PBAv3) updateAvgDuration(
	currentAvg time.Duration,
	newValue time.Duration,
	count uint64,
) time.Duration {
	if count == 0 {
		return newValue
	}
	total := currentAvg.Nanoseconds()*int64(count-1) + newValue.Nanoseconds()
	return time.Duration(total / int64(count))
}

// GetMetrics returns current metrics
func (p *PBAv3) GetMetrics() *PBAv3Metrics {
	p.metrics.mu.RLock()
	defer p.metrics.mu.RUnlock()

	// Return copy
	return &PBAv3Metrics{
		TotalPredictions:      p.metrics.TotalPredictions,
		DatacenterPredictions: p.metrics.DatacenterPredictions,
		InternetPredictions:   p.metrics.InternetPredictions,
		HybridModeSwitches:    p.metrics.HybridModeSwitches,
		DatacenterAccuracy:    p.metrics.DatacenterAccuracy,
		InternetAccuracy:      p.metrics.InternetAccuracy,
		OverallAccuracy:       p.metrics.OverallAccuracy,
		AvgPredictionLatency:  p.metrics.AvgPredictionLatency,
		MaxPredictionLatency:  p.metrics.MaxPredictionLatency,
		P95PredictionLatency:  p.metrics.P95PredictionLatency,
		DatacenterAvgLatency:  p.metrics.DatacenterAvgLatency,
		InternetAvgLatency:    p.metrics.InternetAvgLatency,
	}
}

// Close cleans up resources
func (p *PBAv3) Close() {
	if p.datacenterPredictor != nil {
		p.datacenterPredictor.Close()
	}
	if p.internetPredictor != nil {
		p.internetPredictor.Close()
	}
}

// NewBandwidthHistory creates a new bandwidth history buffer
func NewBandwidthHistory(maxSize int) *BandwidthHistory {
	return &BandwidthHistory{
		samples: make([]prediction.NetworkSample, 0, maxSize),
		maxSize: maxSize,
	}
}

// Add adds a sample to history
func (h *BandwidthHistory) Add(sample prediction.NetworkSample) {
	h.mu.Lock()
	defer h.mu.Unlock()

	h.samples = append(h.samples, sample)
	if len(h.samples) > h.maxSize {
		h.samples = h.samples[1:]
	}
}

// GetRecent returns the most recent n samples
func (h *BandwidthHistory) GetRecent(n int) []prediction.NetworkSample {
	h.mu.RLock()
	defer h.mu.RUnlock()

	if len(h.samples) == 0 {
		return nil
	}

	if n >= len(h.samples) {
		// Return copy of all samples
		result := make([]prediction.NetworkSample, len(h.samples))
		copy(result, h.samples)
		return result
	}

	// Return copy of last n samples
	start := len(h.samples) - n
	result := make([]prediction.NetworkSample, n)
	copy(result, h.samples[start:])
	return result
}

// Len returns the number of samples
func (h *BandwidthHistory) Len() int {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return len(h.samples)
}
