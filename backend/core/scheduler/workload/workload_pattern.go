package workload

import (
	"math"
	"time"
)

// WorkloadPattern represents a recognized pattern in resource usage
type WorkloadPattern struct {
	// PatternType identifies the kind of pattern (diurnal, weekly, etc.)
	PatternType string

	// Type is an alias for PatternType for compatibility
	Type string

	// CycleDuration is the duration of one pattern cycle
	CycleDuration time.Duration

	// Duration is an alias for CycleDuration for compatibility
	Duration time.Duration

	// ConfidenceScore indicates how confident we are in this pattern (0-1)
	ConfidenceScore float64

	// Confidence is an alias for ConfidenceScore for compatibility
	Confidence float64

	// PeakTimestamps contains timestamps when peaks are expected
	PeakTimestamps []time.Time

	// TroughTimestamps contains timestamps when troughs are expected
	TroughTimestamps []time.Time

	// ResourceType indicates which resource this pattern applies to (CPU, memory, etc.)
	ResourceType string

	// PeakValues contains the expected peak values
	PeakValues []float64

	// TroughValues contains the expected trough values
	TroughValues []float64
}

// Pattern types
const (
	// PatternTypeDaily represents a daily pattern (24-hour cycle)
	PatternTypeDaily = "daily"

	// PatternTypeWeekly represents a weekly pattern (7-day cycle)
	PatternTypeWeekly = "weekly"

	// PatternTypeMonthly represents a monthly pattern
	PatternTypeMonthly = "monthly"

	// PatternTypeCustom represents a custom periodic pattern
	PatternTypeCustom = "custom"

	// PatternTypeDiurnal represents a day/night pattern
	PatternTypeDiurnal = "diurnal"

	// PatternTypeBursty represents a pattern with sudden bursts
	PatternTypeBursty = "bursty"

	// Additional pattern types for compatibility
	SteadyPattern   = "steady"
	BurstPattern    = "burst"
	PeriodicPattern = "periodic"
	GrowthPattern   = "growth"
	DeclinePattern  = "decline"
)


// PatternBasedPredictor uses detected patterns to predict future resource usage
type PatternBasedPredictor struct {
	// Patterns contains the detected workload patterns
	Patterns []WorkloadPattern

	// BaselineUsage is the average resource usage
	BaselineUsage float64

	// StdDeviation is the standard deviation of resource usage
	StdDeviation float64
}

// PredictUsage predicts resource usage at a future time
func (p *PatternBasedPredictor) PredictUsage(resourceType string, timestamp time.Time) (float64, float64, error) {
	// Start with baseline
	prediction := p.BaselineUsage
	confidence := 0.5 // Default confidence

	// Apply pattern adjustments
	for _, pattern := range p.Patterns {
		if pattern.ResourceType != resourceType {
			continue
		}

		// Calculate position in the cycle
		cyclePosition := getCyclePosition(timestamp, pattern)

		// Find closest peak or trough
		adjustment, patternConfidence := calculatePatternAdjustment(cyclePosition, pattern)

		// Apply the adjustment weighted by pattern confidence
		prediction += adjustment * pattern.ConfidenceScore

		// Update overall confidence
		confidence = (confidence + patternConfidence*pattern.ConfidenceScore) / 2
	}

	return prediction, confidence, nil
}

// getCyclePosition calculates the position within a pattern cycle
func getCyclePosition(timestamp time.Time, pattern WorkloadPattern) float64 {
	switch pattern.PatternType {
	case PatternTypeDaily:
		// Position in day (0.0 to 1.0)
		midnight := time.Date(timestamp.Year(), timestamp.Month(), timestamp.Day(), 0, 0, 0, 0, timestamp.Location())
		secondsInDay := 24 * 60 * 60
		secondsSinceMidnight := timestamp.Sub(midnight).Seconds()
		return secondsSinceMidnight / float64(secondsInDay)

	case PatternTypeWeekly:
		// Position in week (0.0 to 1.0)
		dayOfWeek := float64(timestamp.Weekday())
		hourOfDay := float64(timestamp.Hour())
		return (dayOfWeek*24 + hourOfDay) / (7 * 24)

	case PatternTypeMonthly:
		// Position in month (0.0 to 1.0)
		dayOfMonth := float64(timestamp.Day() - 1) // 0-based
		daysInMonth := float64(daysInMonth(timestamp.Year(), timestamp.Month()))
		return dayOfMonth / daysInMonth

	default:
		// For custom patterns, calculate based on duration
		// Find a reference timestamp
		refTime := pattern.PeakTimestamps[0]
		if len(pattern.PeakTimestamps) == 0 && len(pattern.TroughTimestamps) > 0 {
			refTime = pattern.TroughTimestamps[0]
		}

		// Calculate seconds since reference time
		secondsSinceRef := timestamp.Sub(refTime).Seconds()

		// Normalize to cycle duration and take modulo
		cycleDurationSeconds := pattern.CycleDuration.Seconds()
		return math.Mod(secondsSinceRef, cycleDurationSeconds) / cycleDurationSeconds
	}
}

// calculatePatternAdjustment determines the adjustment based on cycle position
func calculatePatternAdjustment(cyclePosition float64, pattern WorkloadPattern) (float64, float64) {
	// Default values
	adjustment := 0.0
	confidence := 0.5

	// Convert peak and trough timestamps to cycle positions
	peakPositions := make([]float64, len(pattern.PeakTimestamps))
	for i, peakTime := range pattern.PeakTimestamps {
		peakPositions[i] = getCyclePosition(peakTime, pattern)
	}

	troughPositions := make([]float64, len(pattern.TroughTimestamps))
	for i, troughTime := range pattern.TroughTimestamps {
		troughPositions[i] = getCyclePosition(troughTime, pattern)
	}

	// Find closest peak and trough
	closestPeakDist := 1.0
	closestPeakIdx := -1
	for i, pos := range peakPositions {
		dist := cyclicDistance(cyclePosition, pos)
		if dist < closestPeakDist {
			closestPeakDist = dist
			closestPeakIdx = i
		}
	}

	closestTroughDist := 1.0
	closestTroughIdx := -1
	for i, pos := range troughPositions {
		dist := cyclicDistance(cyclePosition, pos)
		if dist < closestTroughDist {
			closestTroughDist = dist
			closestTroughIdx = i
		}
	}

	// Determine adjustment based on proximity to peaks and troughs
	if closestPeakIdx >= 0 && closestTroughIdx >= 0 {
		// We have both peaks and troughs to consider
		if closestPeakDist < closestTroughDist {
			// Closer to a peak
			peakValue := 0.0
			if closestPeakIdx < len(pattern.PeakValues) {
				peakValue = pattern.PeakValues[closestPeakIdx]
			}

			// Adjustment scaled by distance (closer = stronger effect)
			adjustment = peakValue * (1.0 - closestPeakDist)
			confidence = 1.0 - closestPeakDist
		} else {
			// Closer to a trough
			troughValue := 0.0
			if closestTroughIdx < len(pattern.TroughValues) {
				troughValue = pattern.TroughValues[closestTroughIdx]
			}

			// Adjustment scaled by distance (closer = stronger effect)
			adjustment = troughValue * (1.0 - closestTroughDist)
			confidence = 1.0 - closestTroughDist
		}
	} else if closestPeakIdx >= 0 {
		// Only peaks available
		peakValue := 0.0
		if closestPeakIdx < len(pattern.PeakValues) {
			peakValue = pattern.PeakValues[closestPeakIdx]
		}

		adjustment = peakValue * (1.0 - closestPeakDist)
		confidence = 1.0 - closestPeakDist
	} else if closestTroughIdx >= 0 {
		// Only troughs available
		troughValue := 0.0
		if closestTroughIdx < len(pattern.TroughValues) {
			troughValue = pattern.TroughValues[closestTroughIdx]
		}

		adjustment = troughValue * (1.0 - closestTroughDist)
		confidence = 1.0 - closestTroughDist
	}

	return adjustment, confidence
}

// cyclicDistance calculates distance between two positions in a cycle (0.0 to 1.0)
func cyclicDistance(a, b float64) float64 {
	// Ensure values are in range [0, 1]
	a = a - float64(int(a))
	if a < 0 {
		a += 1.0
	}

	b = b - float64(int(b))
	if b < 0 {
		b += 1.0
	}

	// Direct distance
	directDist := abs(a - b)

	// Cyclic distance (might be shorter to go the other way around the cycle)
	return min(directDist, 1.0-directDist)
}

// abs returns the absolute value of x
func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

// min returns the smaller of x or y
func min(x, y float64) float64 {
	if x < y {
		return x
	}
	return y
}

// daysInMonth returns the number of days in the specified month
func daysInMonth(year int, month time.Month) int {
	// Jump to the first day of the next month, then back up one day
	nextMonth := month + 1
	nextYear := year
	if nextMonth > 12 {
		nextMonth = 1
		nextYear++
	}

	firstOfNextMonth := time.Date(nextYear, nextMonth, 1, 0, 0, 0, 0, time.UTC)
	lastDay := firstOfNextMonth.Add(-time.Hour * 24)
	return lastDay.Day()
}

// WorkloadPatternAnalyzer analyzes VM resource usage to detect patterns
type WorkloadPatternAnalyzer struct {
	// TimeSeriesData for each resource type
	data map[string]TimeSeriesData

	// Detected patterns for each resource type
	patterns map[string][]WorkloadPattern

	// Predictors for each resource type
	predictors map[string]*PatternBasedPredictor

	// Detectors contains the pattern detection algorithms
	detectors []PatternDetector

	// MinimumDataPoints is the minimum number of data points needed for analysis
	MinimumDataPoints int
}

// NewWorkloadPatternAnalyzer creates a new workload pattern analyzer
func NewWorkloadPatternAnalyzer() *WorkloadPatternAnalyzer {
	return &WorkloadPatternAnalyzer{
		data:              make(map[string]TimeSeriesData),
		patterns:          make(map[string][]WorkloadPattern),
		predictors:        make(map[string]*PatternBasedPredictor),
		MinimumDataPoints: 24, // At least 24 data points by default
	}
}

// AddDetector adds a pattern detection algorithm
func (w *WorkloadPatternAnalyzer) AddDetector(detector PatternDetector) {
	w.detectors = append(w.detectors, detector)
}

// AddDataPoint adds a single data point for a resource
func (w *WorkloadPatternAnalyzer) AddDataPoint(resourceType string, timestamp time.Time, value float64) {
	// Ensure the resource type exists in data map
	if _, exists := w.data[resourceType]; !exists {
		w.data[resourceType] = TimeSeriesData{
			ResourceType: resourceType,
			Timestamps:   make([]time.Time, 0),
			Values:       make([]float64, 0),
		}
	}

	// Add the data point
	d := w.data[resourceType]
	d.Timestamps = append(d.Timestamps, timestamp)
	d.Values = append(d.Values, value)
	w.data[resourceType] = d
}

// AnalyzePatterns analyzes the collected data to detect patterns
func (w *WorkloadPatternAnalyzer) AnalyzePatterns() error {
	// Process each resource type
	for resourceType, timeSeries := range w.data {
		// Skip if not enough data points
		if len(timeSeries.Timestamps) < w.MinimumDataPoints {
			continue
		}

		// Calculate baseline and standard deviation
		baseline, stdDev := calculateBaseline(timeSeries.Values)

		// Create predictor
		predictor := &PatternBasedPredictor{
			Patterns:      make([]WorkloadPattern, 0),
			BaselineUsage: baseline,
			StdDeviation:  stdDev,
		}

		// Run each detector
		var detectedPatterns []WorkloadPattern
		for _, detector := range w.detectors {
			patterns, err := detector.DetectPatterns(timeSeries.Timestamps, timeSeries.Values)
			if err != nil {
				return err
			}

			// Set the resource type on each pattern
			for i := range patterns {
				patterns[i].ResourceType = resourceType
			}

			detectedPatterns = append(detectedPatterns, patterns...)
		}

		// Store detected patterns and predictor
		w.patterns[resourceType] = detectedPatterns
		predictor.Patterns = detectedPatterns
		w.predictors[resourceType] = predictor
	}

	return nil
}

// GetPatterns returns detected patterns for a resource type
func (w *WorkloadPatternAnalyzer) GetPatterns(resourceType string) []WorkloadPattern {
	return w.patterns[resourceType]
}

// PredictUsage predicts future resource usage
func (w *WorkloadPatternAnalyzer) PredictUsage(resourceType string, timestamp time.Time) (float64, float64, error) {
	// Get predictor for this resource
	predictor, exists := w.predictors[resourceType]
	if !exists {
		// Return baseline if no predictor exists
		baseline := 0.0
		if ts, exists := w.data[resourceType]; exists && len(ts.Values) > 0 {
			baseline = average(ts.Values)
		}
		return baseline, 0.1, nil // Low confidence
	}

	// Use predictor to get prediction and confidence
	return predictor.PredictUsage(resourceType, timestamp)
}

// calculateBaseline calculates the baseline (average) and standard deviation
func calculateBaseline(values []float64) (float64, float64) {
	if len(values) == 0 {
		return 0, 0
	}

	// Calculate average
	avg := average(values)

	// Calculate standard deviation
	sumSquaredDiff := 0.0
	for _, v := range values {
		diff := v - avg
		sumSquaredDiff += diff * diff
	}

	stdDev := 0.0
	if len(values) > 1 {
		stdDev = sqrt(sumSquaredDiff / float64(len(values)-1))
	}

	return avg, stdDev
}

// average calculates the average of a slice of values
func average(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}

	sum := 0.0
	for _, v := range values {
		sum += v
	}

	return sum / float64(len(values))
}

// sqrt calculates the square root (simplified implementation)
func sqrt(x float64) float64 {
	// For simplicity, using a simple Newton's method
	if x <= 0 {
		return 0
	}

	z := x / 2.0
	for i := 0; i < 10; i++ { // 10 iterations should be sufficient for convergence
		z = z - (z*z-x)/(2*z)
	}

	return z
}

// MigrationWindow represents an optimal time window for migration
type MigrationWindow struct {
	// StartTime is when the window starts
	StartTime time.Time

	// EndTime is when the window ends
	EndTime time.Time

	// Quality indicates the suitability of this window (0-1)
	Quality float64

	// Reason explains why this window was selected
	Reason string
}

// EnhancedWorkloadProfile represents a VM workload profile with advanced analysis
type EnhancedWorkloadProfile struct {
	// VMID is the VM identifier
	VMID string

	// ResourceProfiles contains resource usage profiles by resource type
	ResourceProfiles map[string]*ResourceProfile

	// RecognizedPatterns are patterns detected in the workload
	RecognizedPatterns map[string][]WorkloadPattern

	// WorkloadStability indicates how stable this workload is (0-1)
	WorkloadStability float64

	// LastUpdated is when this profile was last updated
	LastUpdated time.Time

	// SampleCount is the number of samples used to build this profile
	SampleCount int
}

// ResourceProfile represents resource usage for a specific resource type
type ResourceProfile struct {
	// ResourceType is the type of resource (CPU, memory, etc.)
	ResourceType string

	// AverageUsage is the average resource usage
	AverageUsage float64

	// PeakUsage is the peak resource usage
	PeakUsage float64

	// MinimumUsage is the minimum resource usage
	MinimumUsage float64

	// StandardDeviation is the standard deviation of resource usage
	StandardDeviation float64

	// PredictionModel contains parameters for usage prediction
	PredictionModel map[string]interface{}
}

