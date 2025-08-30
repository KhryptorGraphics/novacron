package healing

import (
	"fmt"
	"math"
	"sync"
	"time"

	"github.com/sirupsen/logrus"
)

// PhiAccrualFailureDetector implements the Phi Accrual failure detection algorithm
type PhiAccrualFailureDetector struct {
	mu              sync.RWMutex
	logger          *logrus.Logger
	config          *FailureDetectorConfig
	samples         map[string]*SampleWindow
	intervalHistory map[string]*IntervalHistory
}

// SampleWindow holds recent samples for a target
type SampleWindow struct {
	Samples    []*HealthSample
	MaxSize    int
	CurrentPos int
	Full       bool
}

// IntervalHistory tracks heartbeat intervals for Phi Accrual
type IntervalHistory struct {
	Intervals    []float64
	Mean         float64
	Variance     float64
	LastArrival  time.Time
	SampleCount  int
}

// SimpleThresholdDetector implements a simple threshold-based failure detector
type SimpleThresholdDetector struct {
	mu      sync.RWMutex
	logger  *logrus.Logger
	config  *FailureDetectorConfig
	samples map[string]*SampleWindow
}

// NewPhiAccrualFailureDetector creates a new Phi Accrual failure detector
func NewPhiAccrualFailureDetector(logger *logrus.Logger) *PhiAccrualFailureDetector {
	return &PhiAccrualFailureDetector{
		logger:          logger,
		samples:         make(map[string]*SampleWindow),
		intervalHistory: make(map[string]*IntervalHistory),
		config: &FailureDetectorConfig{
			Algorithm:             AlgorithmPhiAccrual,
			SampleWindowSize:      100,
			MinSamplesForDecision: 5,
			PhiThreshold:          8.0,
			AcceptableHeartbeat:   10 * time.Second,
		},
	}
}

// NewSimpleThresholdDetector creates a new simple threshold detector
func NewSimpleThresholdDetector(logger *logrus.Logger) *SimpleThresholdDetector {
	return &SimpleThresholdDetector{
		logger:  logger,
		samples: make(map[string]*SampleWindow),
		config: &FailureDetectorConfig{
			Algorithm:             AlgorithmSimpleThreshold,
			SampleWindowSize:      20,
			MinSamplesForDecision: 3,
			HealthyThreshold:      0.8,
			UnhealthyThreshold:    0.3,
		},
	}
}

// AddSample adds a new health sample (Phi Accrual)
func (pd *PhiAccrualFailureDetector) AddSample(targetID string, sample *HealthSample) error {
	pd.mu.Lock()
	defer pd.mu.Unlock()

	// Initialize sample window if not exists
	if _, exists := pd.samples[targetID]; !exists {
		pd.samples[targetID] = &SampleWindow{
			Samples: make([]*HealthSample, pd.config.SampleWindowSize),
			MaxSize: pd.config.SampleWindowSize,
		}
		pd.intervalHistory[targetID] = &IntervalHistory{
			Intervals: make([]float64, 0),
		}
	}

	window := pd.samples[targetID]
	history := pd.intervalHistory[targetID]

	// Add sample to window
	window.Samples[window.CurrentPos] = sample
	window.CurrentPos = (window.CurrentPos + 1) % window.MaxSize
	if window.CurrentPos == 0 {
		window.Full = true
	}

	// Update interval history for Phi calculation
	if !history.LastArrival.IsZero() {
		interval := sample.Timestamp.Sub(history.LastArrival).Seconds()
		if interval > 0 {
			history.Intervals = append(history.Intervals, interval)
			
			// Keep only recent intervals
			if len(history.Intervals) > 1000 {
				history.Intervals = history.Intervals[500:]
			}
			
			// Update statistics
			pd.updateStatistics(history)
		}
	}
	
	history.LastArrival = sample.Timestamp
	history.SampleCount++

	pd.logger.WithFields(logrus.Fields{
		"target_id": targetID,
		"healthy":   sample.Healthy,
		"timestamp": sample.Timestamp,
		"phi":       pd.calculatePhi(targetID, time.Now()),
	}).Debug("Health sample added to Phi Accrual detector")

	return nil
}

// IsHealthy determines if a target is healthy using Phi Accrual (Phi Accrual)
func (pd *PhiAccrualFailureDetector) IsHealthy(targetID string) (*HealthAssessment, error) {
	pd.mu.RLock()
	defer pd.mu.RUnlock()

	history, exists := pd.intervalHistory[targetID]
	if !exists {
		return &HealthAssessment{
			TargetID:   targetID,
			Healthy:    true,
			HealthScore: 1.0,
			Confidence: 0.0,
			Reasons:    []string{"No data available"},
			Timestamp:  time.Now(),
		}, nil
	}

	if history.SampleCount < pd.config.MinSamplesForDecision {
		return &HealthAssessment{
			TargetID:   targetID,
			Healthy:    true,
			HealthScore: 1.0,
			Confidence: 0.5,
			Reasons:    []string{"Insufficient samples for decision"},
			Timestamp:  time.Now(),
		}, nil
	}

	// Calculate Phi value
	phi := pd.calculatePhi(targetID, time.Now())
	
	// Determine health based on Phi threshold
	healthy := phi < pd.config.PhiThreshold
	confidence := pd.calculateConfidence(phi)
	healthScore := pd.calculateHealthScore(phi)
	
	reasons := []string{}
	if !healthy {
		reasons = append(reasons, fmt.Sprintf("Phi value %.2f exceeds threshold %.2f", phi, pd.config.PhiThreshold))
	}
	
	if time.Since(history.LastArrival) > pd.config.AcceptableHeartbeat*2 {
		reasons = append(reasons, "No recent heartbeat received")
	}

	return &HealthAssessment{
		TargetID:    targetID,
		Healthy:     healthy,
		HealthScore: healthScore,
		Confidence:  confidence,
		Reasons:     reasons,
		Timestamp:   time.Now(),
		Metadata: map[string]interface{}{
			"phi_value":      phi,
			"phi_threshold":  pd.config.PhiThreshold,
			"last_heartbeat": history.LastArrival,
			"sample_count":   history.SampleCount,
		},
	}, nil
}

// GetHealthScore gets the current health score for a target (Phi Accrual)
func (pd *PhiAccrualFailureDetector) GetHealthScore(targetID string) (float64, error) {
	assessment, err := pd.IsHealthy(targetID)
	if err != nil {
		return 0.0, err
	}
	return assessment.HealthScore, nil
}

// Configure configures the failure detector parameters (Phi Accrual)
func (pd *PhiAccrualFailureDetector) Configure(config *FailureDetectorConfig) error {
	pd.mu.Lock()
	defer pd.mu.Unlock()

	if config == nil {
		return fmt.Errorf("config cannot be nil")
	}

	pd.config = config

	pd.logger.WithFields(logrus.Fields{
		"phi_threshold":        config.PhiThreshold,
		"sample_window_size":   config.SampleWindowSize,
		"acceptable_heartbeat": config.AcceptableHeartbeat,
	}).Info("Phi Accrual detector configured")

	return nil
}

// AddSample adds a new health sample (Simple Threshold)
func (st *SimpleThresholdDetector) AddSample(targetID string, sample *HealthSample) error {
	st.mu.Lock()
	defer st.mu.Unlock()

	// Initialize sample window if not exists
	if _, exists := st.samples[targetID]; !exists {
		st.samples[targetID] = &SampleWindow{
			Samples: make([]*HealthSample, st.config.SampleWindowSize),
			MaxSize: st.config.SampleWindowSize,
		}
	}

	window := st.samples[targetID]

	// Add sample to window
	window.Samples[window.CurrentPos] = sample
	window.CurrentPos = (window.CurrentPos + 1) % window.MaxSize
	if window.CurrentPos == 0 {
		window.Full = true
	}

	st.logger.WithFields(logrus.Fields{
		"target_id": targetID,
		"healthy":   sample.Healthy,
		"timestamp": sample.Timestamp,
	}).Debug("Health sample added to simple threshold detector")

	return nil
}

// IsHealthy determines if a target is healthy using simple thresholds (Simple Threshold)
func (st *SimpleThresholdDetector) IsHealthy(targetID string) (*HealthAssessment, error) {
	st.mu.RLock()
	defer st.mu.RUnlock()

	window, exists := st.samples[targetID]
	if !exists {
		return &HealthAssessment{
			TargetID:   targetID,
			Healthy:    true,
			HealthScore: 1.0,
			Confidence: 0.0,
			Reasons:    []string{"No data available"},
			Timestamp:  time.Now(),
		}, nil
	}

	// Count recent samples
	sampleCount := window.CurrentPos
	if window.Full {
		sampleCount = window.MaxSize
	}

	if sampleCount < st.config.MinSamplesForDecision {
		return &HealthAssessment{
			TargetID:   targetID,
			Healthy:    true,
			HealthScore: 1.0,
			Confidence: 0.5,
			Reasons:    []string{"Insufficient samples for decision"},
			Timestamp:  time.Now(),
		}, nil
	}

	// Calculate health ratio from recent samples
	healthySamples := 0
	totalSamples := sampleCount
	var lastSample *HealthSample

	for i := 0; i < sampleCount; i++ {
		sample := window.Samples[i]
		if sample != nil {
			if sample.Healthy {
				healthySamples++
			}
			if lastSample == nil || sample.Timestamp.After(lastSample.Timestamp) {
				lastSample = sample
			}
		}
	}

	healthRatio := float64(healthySamples) / float64(totalSamples)
	
	// Determine health based on thresholds
	var healthy bool
	var reasons []string

	if healthRatio >= st.config.HealthyThreshold {
		healthy = true
	} else if healthRatio <= st.config.UnhealthyThreshold {
		healthy = false
		reasons = append(reasons, fmt.Sprintf("Health ratio %.2f%% below threshold %.2f%%", 
			healthRatio*100, st.config.UnhealthyThreshold*100))
	} else {
		// In between thresholds - consider suspicious
		healthy = false
		reasons = append(reasons, fmt.Sprintf("Health ratio %.2f%% in suspicious range", healthRatio*100))
	}

	// Check for stale data
	if lastSample != nil && time.Since(lastSample.Timestamp) > 5*time.Minute {
		reasons = append(reasons, "Stale health data")
	}

	confidence := st.calculateSimpleConfidence(healthRatio, sampleCount)
	healthScore := healthRatio

	return &HealthAssessment{
		TargetID:    targetID,
		Healthy:     healthy,
		HealthScore: healthScore,
		Confidence:  confidence,
		Reasons:     reasons,
		Timestamp:   time.Now(),
		Metadata: map[string]interface{}{
			"health_ratio":       healthRatio,
			"healthy_samples":    healthySamples,
			"total_samples":      totalSamples,
			"healthy_threshold":  st.config.HealthyThreshold,
			"unhealthy_threshold": st.config.UnhealthyThreshold,
		},
	}, nil
}

// GetHealthScore gets the current health score for a target (Simple Threshold)
func (st *SimpleThresholdDetector) GetHealthScore(targetID string) (float64, error) {
	assessment, err := st.IsHealthy(targetID)
	if err != nil {
		return 0.0, err
	}
	return assessment.HealthScore, nil
}

// Configure configures the failure detector parameters (Simple Threshold)
func (st *SimpleThresholdDetector) Configure(config *FailureDetectorConfig) error {
	st.mu.Lock()
	defer st.mu.Unlock()

	if config == nil {
		return fmt.Errorf("config cannot be nil")
	}

	st.config = config

	st.logger.WithFields(logrus.Fields{
		"healthy_threshold":   config.HealthyThreshold,
		"unhealthy_threshold": config.UnhealthyThreshold,
		"sample_window_size":  config.SampleWindowSize,
	}).Info("Simple threshold detector configured")

	return nil
}

// Private methods for Phi Accrual detector

func (pd *PhiAccrualFailureDetector) calculatePhi(targetID string, now time.Time) float64 {
	history := pd.intervalHistory[targetID]
	
	if len(history.Intervals) < 2 || history.Variance <= 0 {
		return 0.0 // Not enough data or no variance
	}

	timeSinceLastHeartbeat := now.Sub(history.LastArrival).Seconds()
	
	// Calculate probability that the node is down
	// Using cumulative normal distribution function
	standardDeviation := math.Sqrt(history.Variance)
	
	if standardDeviation <= 0 {
		return 0.0
	}

	// Z-score
	z := (timeSinceLastHeartbeat - history.Mean) / standardDeviation
	
	// Convert to Phi (higher phi = more likely to be failed)
	// Phi = -log10(1 - CDF(z))
	cdf := pd.normalCDF(z)
	
	if cdf >= 0.9999 {
		return 10.0 // Cap at reasonable value
	}
	
	phi := -math.Log10(1.0 - cdf)
	
	if math.IsInf(phi, 0) || math.IsNaN(phi) || phi < 0 {
		return 0.0
	}
	
	return phi
}

func (pd *PhiAccrualFailureDetector) normalCDF(z float64) float64 {
	// Approximation of cumulative standard normal distribution
	// Using Hart's approximation
	if z < -6.0 {
		return 0.0
	}
	if z > 6.0 {
		return 1.0
	}
	
	return 0.5 * (1.0 + math.Erf(z/math.Sqrt(2.0)))
}

func (pd *PhiAccrualFailureDetector) updateStatistics(history *IntervalHistory) {
	if len(history.Intervals) == 0 {
		return
	}

	// Calculate mean
	sum := 0.0
	for _, interval := range history.Intervals {
		sum += interval
	}
	history.Mean = sum / float64(len(history.Intervals))

	// Calculate variance
	if len(history.Intervals) > 1 {
		varianceSum := 0.0
		for _, interval := range history.Intervals {
			diff := interval - history.Mean
			varianceSum += diff * diff
		}
		history.Variance = varianceSum / float64(len(history.Intervals)-1)
	}
}

func (pd *PhiAccrualFailureDetector) calculateConfidence(phi float64) float64 {
	// Confidence decreases as phi increases
	// Use sigmoid function to map phi to confidence [0,1]
	confidence := 1.0 / (1.0 + phi/pd.config.PhiThreshold)
	return math.Max(0.1, confidence)
}

func (pd *PhiAccrualFailureDetector) calculateHealthScore(phi float64) float64 {
	// Health score decreases as phi increases
	if phi >= pd.config.PhiThreshold {
		return 0.0
	}
	
	healthScore := 1.0 - (phi / pd.config.PhiThreshold)
	return math.Max(0.0, math.Min(1.0, healthScore))
}

// Private methods for Simple Threshold detector

func (st *SimpleThresholdDetector) calculateSimpleConfidence(healthRatio float64, sampleCount int) float64 {
	// Base confidence on sample count and distance from thresholds
	sampleConfidence := math.Min(1.0, float64(sampleCount)/20.0) // Full confidence with 20+ samples
	
	// Threshold confidence - higher when clearly above/below thresholds
	var thresholdConfidence float64
	if healthRatio >= st.config.HealthyThreshold {
		thresholdConfidence = math.Min(1.0, (healthRatio-st.config.HealthyThreshold)/(1.0-st.config.HealthyThreshold))
	} else if healthRatio <= st.config.UnhealthyThreshold {
		thresholdConfidence = math.Min(1.0, (st.config.UnhealthyThreshold-healthRatio)/st.config.UnhealthyThreshold)
	} else {
		// In suspicious range - low confidence
		thresholdConfidence = 0.3
	}
	
	return (sampleConfidence + thresholdConfidence) / 2.0
}