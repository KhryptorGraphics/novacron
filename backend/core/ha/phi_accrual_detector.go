package ha

import (
	"math"
	"sync"
	"time"

	"go.uber.org/zap"
)

// PhiAccrualDetector implements the Phi Accrual failure detection algorithm
// It provides adaptive failure detection with configurable thresholds
type PhiAccrualDetector struct {
	name      string
	threshold float64 // Phi threshold for marking as failed

	// Heartbeat tracking
	intervals     []time.Duration // Sliding window of heartbeat intervals
	windowSize    int
	lastHeartbeat time.Time
	firstHeartbeat time.Time

	// Statistics
	mean              float64
	variance          float64
	standardDeviation float64

	// State
	suspected bool
	failed    bool
	phi       float64

	mu     sync.RWMutex
	logger *zap.Logger
}

// NewPhiAccrualDetector creates a new Phi Accrual failure detector
func NewPhiAccrualDetector(name string, threshold float64, windowSize int, logger *zap.Logger) *PhiAccrualDetector {
	if logger == nil {
		logger = zap.NewNop()
	}

	if windowSize <= 0 {
		windowSize = 200 // Default window size
	}

	if threshold <= 0 {
		threshold = 8.0 // Default threshold (P_later ~ 0.0003)
	}

	return &PhiAccrualDetector{
		name:       name,
		threshold:  threshold,
		windowSize: windowSize,
		intervals:  make([]time.Duration, 0, windowSize),
		logger:     logger,
	}
}

// Heartbeat records a heartbeat from the monitored node
func (d *PhiAccrualDetector) Heartbeat() {
	d.mu.Lock()
	defer d.mu.Unlock()

	now := time.Now()

	// First heartbeat
	if d.firstHeartbeat.IsZero() {
		d.firstHeartbeat = now
		d.lastHeartbeat = now
		d.suspected = false
		d.failed = false
		d.phi = 0
		return
	}

	// Calculate interval
	interval := now.Sub(d.lastHeartbeat)
	d.lastHeartbeat = now

	// Add to sliding window
	d.addInterval(interval)

	// Update statistics
	d.updateStatistics()

	// Reset suspicion
	d.suspected = false
	d.failed = false
	d.phi = 0

	d.logger.Debug("Heartbeat received",
		zap.String("detector", d.name),
		zap.Duration("interval", interval),
		zap.Float64("mean", d.mean),
		zap.Float64("stdDev", d.standardDeviation))
}

// GetPhi calculates and returns the current phi value
func (d *PhiAccrualDetector) GetPhi() float64 {
	d.mu.RLock()
	defer d.mu.RUnlock()

	if d.firstHeartbeat.IsZero() || len(d.intervals) < 2 {
		return 0
	}

	timeSinceLastHeartbeat := time.Since(d.lastHeartbeat)
	phi := d.calculatePhi(timeSinceLastHeartbeat)

	return phi
}

// IsSuspected returns true if the node is suspected to have failed
func (d *PhiAccrualDetector) IsSuspected() bool {
	phi := d.GetPhi()

	d.mu.Lock()
	defer d.mu.Unlock()

	d.phi = phi
	d.suspected = phi >= d.threshold

	if d.suspected && !d.failed {
		d.logger.Warn("Node suspected",
			zap.String("detector", d.name),
			zap.Float64("phi", phi),
			zap.Float64("threshold", d.threshold))
	}

	return d.suspected
}

// IsFailed returns true if the node is confirmed failed
func (d *PhiAccrualDetector) IsFailed() bool {
	phi := d.GetPhi()

	d.mu.Lock()
	defer d.mu.Unlock()

	d.phi = phi
	// Use higher threshold for confirmed failure
	d.failed = phi >= (d.threshold * 2)

	if d.failed {
		d.logger.Error("Node failed",
			zap.String("detector", d.name),
			zap.Float64("phi", phi),
			zap.Float64("threshold", d.threshold*2))
	}

	return d.failed
}

// SetThreshold updates the phi threshold
func (d *PhiAccrualDetector) SetThreshold(threshold float64) {
	d.mu.Lock()
	defer d.mu.Unlock()

	d.threshold = threshold
	d.logger.Info("Threshold updated",
		zap.String("detector", d.name),
		zap.Float64("threshold", threshold))
}

// GetStatistics returns current statistics
func (d *PhiAccrualDetector) GetStatistics() (mean, variance, stdDev float64, sampleSize int) {
	d.mu.RLock()
	defer d.mu.RUnlock()

	return d.mean, d.variance, d.standardDeviation, len(d.intervals)
}

// Reset clears all state
func (d *PhiAccrualDetector) Reset() {
	d.mu.Lock()
	defer d.mu.Unlock()

	d.intervals = make([]time.Duration, 0, d.windowSize)
	d.firstHeartbeat = time.Time{}
	d.lastHeartbeat = time.Time{}
	d.mean = 0
	d.variance = 0
	d.standardDeviation = 0
	d.suspected = false
	d.failed = false
	d.phi = 0

	d.logger.Info("Detector reset", zap.String("detector", d.name))
}

// Private methods

// addInterval adds an interval to the sliding window
func (d *PhiAccrualDetector) addInterval(interval time.Duration) {
	if len(d.intervals) >= d.windowSize {
		// Remove oldest interval
		d.intervals = d.intervals[1:]
	}
	d.intervals = append(d.intervals, interval)
}

// updateStatistics recalculates mean, variance, and standard deviation
func (d *PhiAccrualDetector) updateStatistics() {
	if len(d.intervals) == 0 {
		return
	}

	// Calculate mean
	var sum time.Duration
	for _, interval := range d.intervals {
		sum += interval
	}
	d.mean = float64(sum) / float64(len(d.intervals))

	// Calculate variance
	var squaredDiffSum float64
	for _, interval := range d.intervals {
		diff := float64(interval) - d.mean
		squaredDiffSum += diff * diff
	}
	d.variance = squaredDiffSum / float64(len(d.intervals))

	// Calculate standard deviation
	d.standardDeviation = math.Sqrt(d.variance)
}

// calculatePhi calculates the phi value for a given time since last heartbeat
func (d *PhiAccrualDetector) calculatePhi(timeSinceLastHeartbeat time.Duration) float64 {
	if d.mean == 0 || d.standardDeviation == 0 {
		return 0
	}

	// Convert to same unit (nanoseconds as float64)
	t := float64(timeSinceLastHeartbeat)

	// Calculate probability later (P_later)
	// Using cumulative distribution function of exponential distribution
	pLater := math.Exp(-t / d.mean)

	// Handle edge cases
	if pLater <= 0 {
		return math.MaxFloat64 // Maximum phi
	}
	if pLater >= 1 {
		return 0 // Minimum phi
	}

	// Calculate phi = -log10(P_later)
	phi := -math.Log10(pLater)

	// Apply Gaussian distribution adjustment for more accuracy
	z := (t - d.mean) / d.standardDeviation
	gaussianAdjustment := 1 - cdf(z)

	// Combine exponential and gaussian for hybrid approach
	hybridPhi := phi * (1 + gaussianAdjustment)

	return hybridPhi
}

// cdf calculates the cumulative distribution function of standard normal distribution
func cdf(x float64) float64 {
	// Using approximation of error function
	return 0.5 * (1 + erf(x/math.Sqrt2))
}

// erf approximates the error function
func erf(x float64) float64 {
	// Abramowitz and Stegun approximation
	a1 := 0.254829592
	a2 := -0.284496736
	a3 := 1.421413741
	a4 := -1.453152027
	a5 := 1.061405429
	p := 0.3275911

	sign := 1.0
	if x < 0 {
		sign = -1.0
	}
	x = math.Abs(x)

	t := 1.0 / (1.0 + p*x)
	y := 1.0 - (((((a5*t+a4)*t+a3)*t+a2)*t+a1)*t*math.Exp(-x*x))

	return sign * y
}

// PhiAccrualDetectorMetrics contains detector metrics
type PhiAccrualDetectorMetrics struct {
	Name              string
	CurrentPhi        float64
	Threshold         float64
	IsSuspected       bool
	IsFailed          bool
	LastHeartbeat     time.Time
	HeartbeatCount    int64
	MeanInterval      float64
	StdDeviation      float64
	WindowSize        int
	CurrentWindowSize int
}

// GetMetrics returns current metrics
func (d *PhiAccrualDetector) GetMetrics() PhiAccrualDetectorMetrics {
	d.mu.RLock()
	defer d.mu.RUnlock()

	return PhiAccrualDetectorMetrics{
		Name:              d.name,
		CurrentPhi:        d.phi,
		Threshold:         d.threshold,
		IsSuspected:       d.suspected,
		IsFailed:          d.failed,
		LastHeartbeat:     d.lastHeartbeat,
		MeanInterval:      d.mean,
		StdDeviation:      d.standardDeviation,
		WindowSize:        d.windowSize,
		CurrentWindowSize: len(d.intervals),
	}
}

// AdaptiveThreshold adjusts threshold based on network conditions
func (d *PhiAccrualDetector) AdaptiveThreshold(networkJitter float64) {
	d.mu.Lock()
	defer d.mu.Unlock()

	// Adjust threshold based on network jitter
	// Higher jitter = higher threshold to reduce false positives
	baseThreshold := 8.0
	jitterMultiplier := 1.0 + (networkJitter / 100.0) // networkJitter as percentage

	newThreshold := baseThreshold * jitterMultiplier

	// Cap threshold between 5 and 20
	if newThreshold < 5.0 {
		newThreshold = 5.0
	} else if newThreshold > 20.0 {
		newThreshold = 20.0
	}

	d.threshold = newThreshold

	d.logger.Debug("Adaptive threshold adjusted",
		zap.String("detector", d.name),
		zap.Float64("newThreshold", newThreshold),
		zap.Float64("networkJitter", networkJitter))
}