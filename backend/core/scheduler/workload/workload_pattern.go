package workload

import (
	"fmt"
	"math"
	"sort"
	"time"
)

// PatternType represents the type of resource usage pattern
type PatternType string

// Pattern types
const (
	PatternTypeDiurnal    PatternType = "diurnal"     // Daily pattern
	PatternTypeWeekly     PatternType = "weekly"      // Weekly pattern
	PatternTypeMonthly    PatternType = "monthly"     // Monthly pattern
	PatternTypeSporadic   PatternType = "sporadic"    // Irregular spikes
	PatternTypeConstant   PatternType = "constant"    // Steady usage
	PatternTypeGrowing    PatternType = "growing"     // Steadily increasing
	PatternTypeDeclining  PatternType = "declining"   // Steadily decreasing
	PatternTypeUnknown    PatternType = "unknown"     // Unknown pattern
	PatternTypeBursty     PatternType = "bursty"      // Short periods of intensive activity
	PatternTypeCyclical   PatternType = "cyclical"    // General repeating pattern
	PatternTypeOnOff      PatternType = "on-off"      // Alternating between high and low usage
	PatternTypeMultiModal PatternType = "multi-modal" // Multiple distinct usage levels
)

// WorkloadPattern represents a recognized pattern in resource usage
type WorkloadPattern struct {
	// PatternType identifies the kind of pattern
	PatternType PatternType

	// ResourceType is the type of resource this pattern applies to
	ResourceType string

	// CycleDuration is the duration of one pattern cycle
	CycleDuration time.Duration

	// ConfidenceScore indicates how confident we are in this pattern (0-1)
	ConfidenceScore float64

	// PeakTimestamps contains timestamps when peaks are expected
	PeakTimestamps []time.Time

	// TroughTimestamps contains timestamps when troughs are expected
	TroughTimestamps []time.Time

	// Description is a human-readable description of the pattern
	Description string

	// FirstDetected is when this pattern was first detected
	FirstDetected time.Time

	// LastConfirmed is when this pattern was last confirmed
	LastConfirmed time.Time

	// Parameters contains pattern-specific parameters
	Parameters map[string]float64
}

// String returns a string representation of the pattern
func (p *WorkloadPattern) String() string {
	return fmt.Sprintf("%s pattern on %s (confidence: %.2f, cycle: %v)",
		p.PatternType, p.ResourceType, p.ConfidenceScore, p.CycleDuration)
}

// NextPeak predicts the next peak time after the given time
func (p *WorkloadPattern) NextPeak(after time.Time) time.Time {
	if len(p.PeakTimestamps) == 0 {
		return time.Time{} // Zero time if no peaks
	}

	// Sort peaks if needed
	if !sort.SliceIsSorted(p.PeakTimestamps, func(i, j int) bool {
		return p.PeakTimestamps[i].Before(p.PeakTimestamps[j])
	}) {
		sort.Slice(p.PeakTimestamps, func(i, j int) bool {
			return p.PeakTimestamps[i].Before(p.PeakTimestamps[j])
		})
	}

	// Find the next peak after the given time
	for _, peak := range p.PeakTimestamps {
		if peak.After(after) {
			return peak
		}
	}

	// If no future peak in the list, project based on cycle duration
	if p.CycleDuration > 0 {
		lastPeak := p.PeakTimestamps[len(p.PeakTimestamps)-1]
		for {
			lastPeak = lastPeak.Add(p.CycleDuration)
			if lastPeak.After(after) {
				return lastPeak
			}
		}
	}

	return time.Time{} // Zero time if can't predict
}

// NextTrough predicts the next trough time after the given time
func (p *WorkloadPattern) NextTrough(after time.Time) time.Time {
	if len(p.TroughTimestamps) == 0 {
		return time.Time{} // Zero time if no troughs
	}

	// Sort troughs if needed
	if !sort.SliceIsSorted(p.TroughTimestamps, func(i, j int) bool {
		return p.TroughTimestamps[i].Before(p.TroughTimestamps[j])
	}) {
		sort.Slice(p.TroughTimestamps, func(i, j int) bool {
			return p.TroughTimestamps[i].Before(p.TroughTimestamps[j])
		})
	}

	// Find the next trough after the given time
	for _, trough := range p.TroughTimestamps {
		if trough.After(after) {
			return trough
		}
	}

	// If no future trough in the list, project based on cycle duration
	if p.CycleDuration > 0 {
		lastTrough := p.TroughTimestamps[len(p.TroughTimestamps)-1]
		for {
			lastTrough = lastTrough.Add(p.CycleDuration)
			if lastTrough.After(after) {
				return lastTrough
			}
		}
	}

	return time.Time{} // Zero time if can't predict
}

// IsActivePattern checks if the pattern is still relevant
func (p *WorkloadPattern) IsActivePattern(now time.Time) bool {
	// If the pattern was confirmed recently, it's still active
	return now.Sub(p.LastConfirmed) < p.CycleDuration*3
}

// PredictUsage predicts resource usage at a specific time
func (p *WorkloadPattern) PredictUsage(at time.Time) float64 {
	if p.Parameters == nil || len(p.Parameters) == 0 {
		return -1 // Can't predict without parameters
	}

	switch p.PatternType {
	case PatternTypeDiurnal, PatternTypeWeekly, PatternTypeMonthly, PatternTypeCyclical:
		return p.predictCyclicalUsage(at)
	case PatternTypeConstant:
		return p.Parameters["baseline"]
	case PatternTypeGrowing:
		return p.predictTrendUsage(at, true)
	case PatternTypeDeclining:
		return p.predictTrendUsage(at, false)
	default:
		return -1 // Can't predict for other pattern types
	}
}

// predictCyclicalUsage predicts usage for cyclical patterns
func (p *WorkloadPattern) predictCyclicalUsage(at time.Time) float64 {
	baseline := p.Parameters["baseline"]
	amplitude := p.Parameters["amplitude"]
	periodSeconds := p.CycleDuration.Seconds()

	if periodSeconds <= 0 {
		return baseline
	}

	// Find reference time (first peak, if available)
	referenceTime := time.Time{}
	if len(p.PeakTimestamps) > 0 {
		referenceTime = p.PeakTimestamps[0]
	} else if p.FirstDetected.Unix() > 0 {
		referenceTime = p.FirstDetected
	} else {
		return baseline // Can't predict without reference
	}

	// Calculate phase within the cycle
	secondsSinceReference := at.Sub(referenceTime).Seconds()
	phase := math.Mod(secondsSinceReference, periodSeconds) / periodSeconds

	// Generate sinusoidal pattern (simplified model)
	return baseline + amplitude*math.Sin(2*math.Pi*phase)
}

// predictTrendUsage predicts usage for trending patterns
func (p *WorkloadPattern) predictTrendUsage(at time.Time, isGrowing bool) float64 {
	baseline := p.Parameters["baseline"]
	slope := p.Parameters["slope"]
	if !isGrowing {
		slope = -slope // Negative slope for declining
	}

	// Find reference time
	referenceTime := p.FirstDetected
	if referenceTime.Unix() <= 0 {
		return baseline
	}

	// Calculate time since reference in hours
	hoursSinceReference := at.Sub(referenceTime).Hours()

	// Linear trend model
	return baseline + slope*hoursSinceReference
}

// EnhancedWorkloadProfile extends WorkloadProfile with advanced pattern recognition
type EnhancedWorkloadProfile struct {
	// Embed the basic WorkloadProfile
	*WorkloadProfile

	// RecognizedPatterns are patterns detected in the workload
	RecognizedPatterns map[string]*WorkloadPattern

	// PredictionModel contains parameters for usage prediction
	PredictionModel map[string]interface{}

	// WorkloadStability indicates how stable this workload is (0-1)
	WorkloadStability float64

	// ResourceCorrelations maps pairs of resources to their correlation coefficient
	ResourceCorrelations map[string]float64

	// OptimalMigrationWindows suggests good times for migration
	OptimalMigrationWindows []MigrationWindow
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

// NewEnhancedProfile creates an enhanced profile from a basic profile
func NewEnhancedProfile(baseProfile *WorkloadProfile) *EnhancedWorkloadProfile {
	return &EnhancedWorkloadProfile{
		WorkloadProfile:         baseProfile,
		RecognizedPatterns:      make(map[string]*WorkloadPattern),
		PredictionModel:         make(map[string]interface{}),
		WorkloadStability:       0.5, // Default value
		ResourceCorrelations:    make(map[string]float64),
		OptimalMigrationWindows: []MigrationWindow{},
	}
}

// PredictResourceUsage predicts resource usage at a future time
func (p *EnhancedWorkloadProfile) PredictResourceUsage(resourceType string, at time.Time) float64 {
	// First check if we have a pattern for this resource
	pattern, exists := p.RecognizedPatterns[resourceType]
	if exists && pattern.ConfidenceScore > 0.5 {
		prediction := pattern.PredictUsage(at)
		if prediction >= 0 {
			return prediction
		}
	}

	// Fall back to simple prediction from the base profile
	if usage, exists := p.ResourceUsagePatterns[resourceType]; exists {
		return usage.PredictedUsage
	}

	return -1 // Can't predict
}

// GetOptimalMigrationWindows gets optimal migration windows in a time range
func (p *EnhancedWorkloadProfile) GetOptimalMigrationWindows(start, end time.Time) []MigrationWindow {
	windows := make([]MigrationWindow, 0)

	// Start with any pre-computed windows
	for _, window := range p.OptimalMigrationWindows {
		if window.StartTime.After(start) && window.EndTime.Before(end) {
			windows = append(windows, window)
		}
	}

	// If we don't have any pre-computed windows in the range, find some
	if len(windows) == 0 {
		windows = p.findMigrationWindows(start, end)
	}

	// Sort windows by quality (best first)
	sort.Slice(windows, func(i, j int) bool {
		return windows[i].Quality > windows[j].Quality
	})

	return windows
}

// findMigrationWindows identifies good migration windows
func (p *EnhancedWorkloadProfile) findMigrationWindows(start, end time.Time) []MigrationWindow {
	windows := make([]MigrationWindow, 0)

	// Look for periods of low resource usage
	for resourceType, pattern := range p.RecognizedPatterns {
		if pattern.ConfidenceScore < 0.6 {
			continue // Skip low-confidence patterns
		}

		// For each trough, create a window
		current := start
		for current.Before(end) {
			trough := pattern.NextTrough(current)
			if trough.IsZero() || trough.After(end) {
				break
			}

			// Create a window around the trough
			windowDuration := pattern.CycleDuration / 4
			if windowDuration > 2*time.Hour {
				windowDuration = 2 * time.Hour
			}

			window := MigrationWindow{
				StartTime: trough.Add(-windowDuration / 2),
				EndTime:   trough.Add(windowDuration / 2),
				Quality:   pattern.ConfidenceScore * 0.8, // Scale by confidence
				Reason:    fmt.Sprintf("Low %s usage expected", resourceType),
			}

			// Ensure window is within the requested range
			if window.StartTime.Before(start) {
				window.StartTime = start
			}
			if window.EndTime.After(end) {
				window.EndTime = end
			}

			windows = append(windows, window)
			current = trough.Add(pattern.CycleDuration / 2)
		}
	}

	return windows
}

// CalculateWorkloadStability calculates overall workload stability
func (p *EnhancedWorkloadProfile) CalculateWorkloadStability() {
	if len(p.RecognizedPatterns) == 0 {
		p.WorkloadStability = 0.3 // Default value for unknown stability
		return
	}

	// Calculate weighted average of pattern confidence scores
	totalWeight := 0.0
	weightedSum := 0.0

	for _, pattern := range p.RecognizedPatterns {
		// Skip certain pattern types that don't indicate stability
		if pattern.PatternType == PatternTypeSporadic || pattern.PatternType == PatternTypeUnknown {
			continue
		}

		weight := 1.0
		switch pattern.ResourceType {
		case "cpu_usage":
			weight = 1.5 // CPU is more important
		case "memory_usage":
			weight = 1.2 // Memory is important too
		}

		weightedSum += pattern.ConfidenceScore * weight
		totalWeight += weight
	}

	if totalWeight > 0 {
		p.WorkloadStability = weightedSum / totalWeight
	} else {
		p.WorkloadStability = 0.3
	}
}

// IsStableWorkload determines if the workload is stable enough for prediction
func (p *EnhancedWorkloadProfile) IsStableWorkload() bool {
	return p.WorkloadStability >= 0.6
}

// DetectPatterns analyzes historical data to detect patterns
// This would normally be a sophisticated algorithm, simplified here
func (p *EnhancedWorkloadProfile) DetectPatterns() {
	now := time.Now()

	// Analyze each resource type
	for resourceName, resourcePattern := range p.ResourceUsagePatterns {
		// Skip if we don't have enough data points
		if len(resourcePattern.UsagePattern) < 10 {
			continue
		}

		// Convert map to slices for time series analysis
		timestamps := make([]time.Time, 0, len(resourcePattern.UsagePattern))
		values := make([]float64, 0, len(resourcePattern.UsagePattern))

		for ts, val := range resourcePattern.UsagePattern {
			timestamps = append(timestamps, ts)
			values = append(values, val)
		}

		// Sort by timestamp
		sort.Slice(timestamps, func(i, j int) bool {
			return timestamps[i].Before(timestamps[j])
		})
		sort.Slice(values, func(i, j int) bool {
			return timestamps[i].Before(timestamps[j])
		})

		// Calculate basic statistics
		var sum, sumSquares float64
		for _, val := range values {
			sum += val
			sumSquares += val * val
		}

		mean := sum / float64(len(values))
		variance := (sumSquares / float64(len(values))) - (mean * mean)
		stdDev := math.Sqrt(variance)

		// Simple pattern detection - check variability
		if stdDev < 5.0 {
			// Low variability - likely constant workload
			p.RecognizedPatterns[resourceName] = &WorkloadPattern{
				PatternType:     PatternTypeConstant,
				ResourceType:    resourceName,
				CycleDuration:   24 * time.Hour, // Nominal value
				ConfidenceScore: 0.9,
				Description:     "Constant workload with low variability",
				FirstDetected:   now,
				LastConfirmed:   now,
				Parameters: map[string]float64{
					"baseline": mean,
					"stddev":   stdDev,
				},
			}
		} else {
			// Check for basic daily pattern (simplified)
			// In a real implementation, this would use proper time series analysis
			isDailyPattern := false
			if len(timestamps) >= 24 {
				// Simple heuristic - check if there's a pattern in hourly averages
				hourlyAverages := make([]float64, 24)
				hourCounts := make([]int, 24)

				for i, ts := range timestamps {
					hour := ts.Hour()
					hourlyAverages[hour] += values[i]
					hourCounts[hour]++
				}

				// Normalize
				hourlyVariation := 0.0
				for i := 0; i < 24; i++ {
					if hourCounts[i] > 0 {
						hourlyAverages[i] /= float64(hourCounts[i])
						hourlyVariation += math.Abs(hourlyAverages[i] - mean)
					}
				}

				// If hourly variation is significant, it might be a daily pattern
				if hourlyVariation > stdDev*3 {
					isDailyPattern = true

					// Find peak and trough hours
					peakHour := 0
					troughHour := 0
					peakValue := hourlyAverages[0]
					troughValue := hourlyAverages[0]

					for i := 1; i < 24; i++ {
						if hourlyAverages[i] > peakValue {
							peakValue = hourlyAverages[i]
							peakHour = i
						}
						if hourlyAverages[i] < troughValue {
							troughValue = hourlyAverages[i]
							troughHour = i
						}
					}

					// Create peak and trough timestamps
					peakTimestamps := make([]time.Time, 0)
					troughTimestamps := make([]time.Time, 0)

					// Use yesterday as reference
					yesterday := now.AddDate(0, 0, -1)
					peakTime := time.Date(yesterday.Year(), yesterday.Month(), yesterday.Day(), peakHour, 0, 0, 0, yesterday.Location())
					troughTime := time.Date(yesterday.Year(), yesterday.Month(), yesterday.Day(), troughHour, 0, 0, 0, yesterday.Location())

					peakTimestamps = append(peakTimestamps, peakTime)
					troughTimestamps = append(troughTimestamps, troughTime)

					p.RecognizedPatterns[resourceName] = &WorkloadPattern{
						PatternType:      PatternTypeDiurnal,
						ResourceType:     resourceName,
						CycleDuration:    24 * time.Hour,
						ConfidenceScore:  0.7,
						PeakTimestamps:   peakTimestamps,
						TroughTimestamps: troughTimestamps,
						Description:      fmt.Sprintf("Daily pattern with peak at %d:00 and trough at %d:00", peakHour, troughHour),
						FirstDetected:    now,
						LastConfirmed:    now,
						Parameters: map[string]float64{
							"baseline":  mean,
							"amplitude": (peakValue - troughValue) / 2,
						},
					}
				}
			}

			// If not a daily pattern, check for other patterns
			if !isDailyPattern {
				// Check for growth or decline
				if len(timestamps) >= 2 {
					firstVal := values[0]
					lastVal := values[len(values)-1]
					duration := timestamps[len(timestamps)-1].Sub(timestamps[0])

					// Calculate slope (change per hour)
					slopePerHour := (lastVal - firstVal) / duration.Hours()

					// If significant change over time
					if math.Abs(slopePerHour) > 0.1*mean {
						patternType := PatternTypeGrowing
						if slopePerHour < 0 {
							patternType = PatternTypeDeclining
						}

						p.RecognizedPatterns[resourceName] = &WorkloadPattern{
							PatternType:     patternType,
							ResourceType:    resourceName,
							CycleDuration:   0, // Not applicable
							ConfidenceScore: 0.6,
							Description:     fmt.Sprintf("%s workload with rate %.2f per hour", patternType, math.Abs(slopePerHour)),
							FirstDetected:   timestamps[0],
							LastConfirmed:   now,
							Parameters: map[string]float64{
								"baseline": firstVal,
								"slope":    slopePerHour,
							},
						}
					} else {
						// Default to sporadic if no other pattern detected
						p.RecognizedPatterns[resourceName] = &WorkloadPattern{
							PatternType:     PatternTypeSporadic,
							ResourceType:    resourceName,
							CycleDuration:   0, // Not applicable
							ConfidenceScore: 0.5,
							Description:     "Variable workload with no clear pattern",
							FirstDetected:   now,
							LastConfirmed:   now,
							Parameters: map[string]float64{
								"baseline": mean,
								"stddev":   stdDev,
							},
						}
					}
				}
			}
		}
	}

	// After detecting individual patterns, calculate overall stability
	p.CalculateWorkloadStability()
}
