package workload

import (
	"math"
	"sort"
	"time"
)

// WorkloadProfileAdapter adapts to existing WorkloadProfile interfaces
type WorkloadProfileAdapter struct {
	// VMID is the VM identifier
	VMID string

	// HistoryDuration is how long of history we have
	HistoryDuration time.Duration

	// LastUpdated is when this profile was last updated
	LastUpdated time.Time

	// ResourceUsage maps resource types to usage statistics
	ResourceUsage map[string]ResourceUsageStats
}

// ResourceUsageStats contains statistics about resource usage
type ResourceUsageStats struct {
	// AverageUsage is the average resource usage
	AverageUsage float64

	// PeakUsage is the peak resource usage
	PeakUsage float64

	// MinimumUsage is the minimum resource usage
	MinimumUsage float64

	// StandardDeviation is the standard deviation of resource usage
	StandardDeviation float64

	// Samples contains the historical samples
	Samples []ResourceSample
}

// ResourceSample represents a single resource usage sample
type ResourceSample struct {
	// Timestamp is when the sample was taken
	Timestamp time.Time

	// Value is the resource usage value
	Value float64
}

// EnhancedWorkloadProfile with additional methods to support the scheduler
func (p *EnhancedWorkloadProfile) DetectPatterns() {
	// Detect patterns across each resource type in the profile
	for resourceType, resourceProfile := range p.ResourceProfiles {
		// Create pattern detector based on resource type
		var detector PatternDetector

		// Use different detectors based on resource type
		switch resourceType {
		case "cpu":
			detector = &simplePatternDetector{
				patternTypes:  []string{PatternTypeDaily, PatternTypeDiurnal, PatternTypeBursty},
				minConfidence: 0.5,
			}
		case "memory":
			detector = &simplePatternDetector{
				patternTypes:  []string{PatternTypeDaily, PatternTypeWeekly},
				minConfidence: 0.6,
			}
		default:
			detector = &simplePatternDetector{
				patternTypes:  []string{PatternTypeDaily},
				minConfidence: 0.5,
			}
		}

		// Create time series from samples if available
		if data, ok := p.resourceData[resourceType]; ok && len(data.Timestamps) > 0 {
			patterns, _ := detector.DetectPatterns(data.Timestamps, data.Values)

			// Set the resource type on each pattern
			for i := range patterns {
				patterns[i].ResourceType = resourceType
			}

			// Store patterns
			p.RecognizedPatterns[resourceType] = patterns

			// Update stability based on pattern confidence
			totalConfidence := 0.0
			for _, pattern := range patterns {
				totalConfidence += pattern.ConfidenceScore
			}

			// Average confidence over all patterns
			if len(patterns) > 0 {
				confidence := totalConfidence / float64(len(patterns))
				// Update the overall workload stability using a weighted average
				p.WorkloadStability = (p.WorkloadStability*0.7 + confidence*0.3)
			}
		}

		// Ensure ResourceProfile has a PredictionModel
		if resourceProfile.PredictionModel == nil {
			resourceProfile.PredictionModel = make(map[string]interface{})
		}

		// Update PredictionModel with pattern information
		if patterns, ok := p.RecognizedPatterns[resourceType]; ok && len(patterns) > 0 {
			resourceProfile.PredictionModel["patterns"] = patterns
		}
	}
}

// GetOptimalMigrationWindows finds optimal time windows for migration
func (p *EnhancedWorkloadProfile) GetOptimalMigrationWindows(start, end time.Time) []MigrationWindow {
	if p.WorkloadStability < 0.4 {
		// For unstable workloads, just return a default window
		return []MigrationWindow{
			{
				StartTime: start.Add(1 * time.Hour),
				EndTime:   start.Add(3 * time.Hour),
				Quality:   0.5,
				Reason:    "Default window for unstable workload",
			},
		}
	}

	// For stable workloads, find optimal windows based on resource usage patterns
	var windows []MigrationWindow

	// Check every hour in the range
	for t := start; t.Before(end); t = t.Add(1 * time.Hour) {
		// Predict resource usage for this time
		cpuUsage := p.PredictResourceUsage("cpu", t)
		memUsage := p.PredictResourceUsage("memory", t)

		// If both CPU and memory usage are below thresholds, this is a good window
		if cpuUsage < 50 && memUsage < 60 {
			// Find the end of this low-usage window
			windowEnd := t
			for check := t.Add(30 * time.Minute); check.Before(end); check = check.Add(30 * time.Minute) {
				nextCpuUsage := p.PredictResourceUsage("cpu", check)
				nextMemUsage := p.PredictResourceUsage("memory", check)

				if nextCpuUsage < 50 && nextMemUsage < 60 {
					windowEnd = check
				} else {
					break
				}
			}

			// If the window is at least 1 hour long, add it
			if windowEnd.Sub(t) >= 1*time.Hour {
				quality := 1.0 - ((cpuUsage/100 + memUsage/100) / 2)
				windows = append(windows, MigrationWindow{
					StartTime: t,
					EndTime:   windowEnd,
					Quality:   quality,
					Reason:    "Low resource usage window",
				})

				// Skip to the end of this window
				t = windowEnd
			}
		}
	}

	// If no windows found, return a default window during typical off-hours
	if len(windows) == 0 {
		// Try to find a night-time window (2am-4am local time)
		midnight := time.Date(start.Year(), start.Month(), start.Day(), 0, 0, 0, 0, start.Location())
		nightStart := midnight.Add(2 * time.Hour)
		nightEnd := midnight.Add(4 * time.Hour)

		// If it's already past tonight's window, use tomorrow's
		if start.After(nightEnd) {
			nightStart = nightStart.Add(24 * time.Hour)
			nightEnd = nightEnd.Add(24 * time.Hour)
		}

		return []MigrationWindow{
			{
				StartTime: nightStart,
				EndTime:   nightEnd,
				Quality:   0.7,
				Reason:    "Off-hours window",
			},
		}
	}

	// Sort windows by quality (best first)
	sort.Slice(windows, func(i, j int) bool {
		return windows[i].Quality > windows[j].Quality
	})

	return windows
}

// PredictResourceUsage predicts resource usage at a future time
func (p *EnhancedWorkloadProfile) PredictResourceUsage(resourceType string, timestamp time.Time) float64 {
	// Get patterns for this resource
	patterns, ok := p.RecognizedPatterns[resourceType]
	if !ok || len(patterns) == 0 {
		// No patterns, return average usage
		if profile, ok := p.ResourceProfiles[resourceType]; ok {
			return profile.AverageUsage
		}
		return -1 // Unknown
	}

	// Start with the average usage
	prediction := p.ResourceProfiles[resourceType].AverageUsage

	// Apply pattern adjustments
	for _, pattern := range patterns {
		// Skip patterns with low confidence
		if pattern.ConfidenceScore < 0.3 {
			continue
		}

		// Calculate position in the cycle
		cyclePosition := getCyclePosition(timestamp, pattern)

		// Find closest peak or trough
		adjustment, _ := calculatePatternAdjustment(cyclePosition, pattern)

		// Apply the adjustment weighted by pattern confidence
		prediction += adjustment * pattern.ConfidenceScore
	}

	// Ensure prediction is in a reasonable range
	prediction = math.Max(0, prediction)
	prediction = math.Min(100, prediction)

	return prediction
}

// IsStableWorkload checks if the workload has a stable pattern
func (p *EnhancedWorkloadProfile) IsStableWorkload() bool {
	return p.WorkloadStability >= 0.6
}

// NewEnhancedProfile creates a new enhanced workload profile from a WorkloadProfileAdapter
func NewEnhancedProfile(vmID string) *EnhancedWorkloadProfile {
	return &EnhancedWorkloadProfile{
		VMID:               vmID,
		ResourceProfiles:   make(map[string]*ResourceProfile),
		RecognizedPatterns: make(map[string][]WorkloadPattern),
		WorkloadStability:  0.5, // Default medium stability
		LastUpdated:        time.Now(),
		SampleCount:        0,
		resourceData:       make(map[string]TimeSeriesData),
	}
}

// SetWorkloadProfile updates the enhanced profile from a WorkloadProfileAdapter
func (p *EnhancedWorkloadProfile) SetWorkloadProfile(profile *WorkloadProfileAdapter) {
	p.LastUpdated = profile.LastUpdated

	// Convert resource usage stats to ResourceProfiles
	for resourceType, stats := range profile.ResourceUsage {
		resourceProfile, exists := p.ResourceProfiles[resourceType]
		if !exists {
			resourceProfile = &ResourceProfile{
				ResourceType:      resourceType,
				AverageUsage:      stats.AverageUsage,
				PeakUsage:         stats.PeakUsage,
				MinimumUsage:      stats.MinimumUsage,
				StandardDeviation: stats.StandardDeviation,
				PredictionModel:   make(map[string]interface{}),
			}
			p.ResourceProfiles[resourceType] = resourceProfile
		} else {
			resourceProfile.AverageUsage = stats.AverageUsage
			resourceProfile.PeakUsage = stats.PeakUsage
			resourceProfile.MinimumUsage = stats.MinimumUsage
			resourceProfile.StandardDeviation = stats.StandardDeviation
		}

		// Convert samples to TimeSeriesData for pattern detection
		timestamps := make([]time.Time, 0, len(stats.Samples))
		values := make([]float64, 0, len(stats.Samples))

		for _, sample := range stats.Samples {
			timestamps = append(timestamps, sample.Timestamp)
			values = append(values, sample.Value)
		}

		p.resourceData[resourceType] = TimeSeriesData{
			ResourceType: resourceType,
			Timestamps:   timestamps,
			Values:       values,
		}
	}

	p.SampleCount = 0
	for _, data := range p.resourceData {
		p.SampleCount += len(data.Values)
	}
}

// simple pattern detector for the implementation
type simplePatternDetector struct {
	patternTypes  []string
	minConfidence float64
}

func (d *simplePatternDetector) Name() string {
	return "SimplePatternDetector"
}

func (d *simplePatternDetector) Configure(options map[string]interface{}) error {
	return nil
}

func (d *simplePatternDetector) DetectPatterns(timePoints []time.Time, values []float64) ([]WorkloadPattern, error) {
	if len(timePoints) < 24 || len(values) < 24 {
		return nil, nil // Not enough data
	}

	var patterns []WorkloadPattern

	// Check for daily patterns
	if contains(d.patternTypes, PatternTypeDaily) {
		dailyPattern := detectDailyPattern(timePoints, values)
		if dailyPattern.ConfidenceScore >= d.minConfidence {
			patterns = append(patterns, dailyPattern)
		}
	}

	// Check for diurnal patterns (day/night)
	if contains(d.patternTypes, PatternTypeDiurnal) {
		diurnalPattern := detectDiurnalPattern(timePoints, values)
		if diurnalPattern.ConfidenceScore >= d.minConfidence {
			patterns = append(patterns, diurnalPattern)
		}
	}

	// Check for bursty patterns
	if contains(d.patternTypes, PatternTypeBursty) {
		burstyPattern := detectBurstyPattern(timePoints, values)
		if burstyPattern.ConfidenceScore >= d.minConfidence {
			patterns = append(patterns, burstyPattern)
		}
	}

	// Check for weekly patterns
	if contains(d.patternTypes, PatternTypeWeekly) && len(timePoints) >= 7*24 {
		weeklyPattern := detectWeeklyPattern(timePoints, values)
		if weeklyPattern.ConfidenceScore >= d.minConfidence {
			patterns = append(patterns, weeklyPattern)
		}
	}

	return patterns, nil
}

// Helper function to check if a slice contains a string
func contains(slice []string, str string) bool {
	for _, item := range slice {
		if item == str {
			return true
		}
	}
	return false
}

// Detect daily pattern (basic implementation)
func detectDailyPattern(timestamps []time.Time, values []float64) WorkloadPattern {
	// Group by hour of day
	hourlyAvg := make([]float64, 24)
	hourlyCount := make([]int, 24)

	for i, ts := range timestamps {
		hour := ts.Hour()
		hourlyAvg[hour] += values[i]
		hourlyCount[hour]++
	}

	// Calculate averages
	for i := 0; i < 24; i++ {
		if hourlyCount[i] > 0 {
			hourlyAvg[i] /= float64(hourlyCount[i])
		}
	}

	// Find peaks and troughs
	var peakHours []int
	var troughHours []int

	// Simple peak/trough detection
	for i := 0; i < 24; i++ {
		prev := (i + 23) % 24
		next := (i + 1) % 24

		// Peak if higher than both neighbors
		if hourlyAvg[i] > hourlyAvg[prev] && hourlyAvg[i] > hourlyAvg[next] {
			peakHours = append(peakHours, i)
		}

		// Trough if lower than both neighbors
		if hourlyAvg[i] < hourlyAvg[prev] && hourlyAvg[i] < hourlyAvg[next] {
			troughHours = append(troughHours, i)
		}
	}

	// Convert hours to timestamps (use today)
	today := time.Now()
	startOfDay := time.Date(today.Year(), today.Month(), today.Day(), 0, 0, 0, 0, today.Location())

	peakTimestamps := make([]time.Time, len(peakHours))
	peakValues := make([]float64, len(peakHours))
	for i, hour := range peakHours {
		peakTimestamps[i] = startOfDay.Add(time.Duration(hour) * time.Hour)
		peakValues[i] = hourlyAvg[hour]
	}

	troughTimestamps := make([]time.Time, len(troughHours))
	troughValues := make([]float64, len(troughHours))
	for i, hour := range troughHours {
		troughTimestamps[i] = startOfDay.Add(time.Duration(hour) * time.Hour)
		troughValues[i] = hourlyAvg[hour]
	}

	// Calculate confidence based on stability of pattern
	// More peaks/troughs and larger differences indicate stronger pattern
	confidence := 0.0
	if len(peakHours) > 0 && len(troughHours) > 0 {
		// Calculate average peak and trough values
		avgPeak := 0.0
		for _, v := range peakValues {
			avgPeak += v
		}
		avgPeak /= float64(len(peakValues))

		avgTrough := 0.0
		for _, v := range troughValues {
			avgTrough += v
		}
		avgTrough /= float64(len(troughValues))

		// Larger difference between peaks and troughs = stronger pattern
		diff := avgPeak - avgTrough
		maxDiff := 100.0 // Assuming values are in percentage
		confidence = math.Min(0.9, diff/maxDiff*0.9)
	}

	return WorkloadPattern{
		PatternType:      PatternTypeDaily,
		CycleDuration:    24 * time.Hour,
		ConfidenceScore:  confidence,
		PeakTimestamps:   peakTimestamps,
		TroughTimestamps: troughTimestamps,
		PeakValues:       peakValues,
		TroughValues:     troughValues,
	}
}

// Detect diurnal pattern (day/night)
func detectDiurnalPattern(timestamps []time.Time, values []float64) WorkloadPattern {
	// Group into day (7am-7pm) and night (7pm-7am)
	var dayValues []float64
	var nightValues []float64

	for i, ts := range timestamps {
		hour := ts.Hour()
		if hour >= 7 && hour < 19 {
			dayValues = append(dayValues, values[i])
		} else {
			nightValues = append(nightValues, values[i])
		}
	}

	// Calculate averages
	dayAvg := average(dayValues)
	nightAvg := average(nightValues)

	// Create pattern with day peak and night trough
	today := time.Now()
	startOfDay := time.Date(today.Year(), today.Month(), today.Day(), 0, 0, 0, 0, today.Location())

	// Peak at noon, trough at midnight
	peakTimestamps := []time.Time{startOfDay.Add(12 * time.Hour)}
	troughTimestamps := []time.Time{startOfDay, startOfDay.Add(24 * time.Hour)}

	peakValues := []float64{dayAvg}
	troughValues := []float64{nightAvg, nightAvg}

	// Confidence based on day/night difference
	diff := math.Abs(dayAvg - nightAvg)
	confidence := math.Min(0.9, diff/100.0*0.8)

	return WorkloadPattern{
		PatternType:      PatternTypeDiurnal,
		CycleDuration:    24 * time.Hour,
		ConfidenceScore:  confidence,
		PeakTimestamps:   peakTimestamps,
		TroughTimestamps: troughTimestamps,
		PeakValues:       peakValues,
		TroughValues:     troughValues,
	}
}

// Detect weekly pattern
func detectWeeklyPattern(timestamps []time.Time, values []float64) WorkloadPattern {
	// Group by day of week
	dailyAvg := make([]float64, 7)
	dailyCount := make([]int, 7)

	for i, ts := range timestamps {
		day := int(ts.Weekday())
		dailyAvg[day] += values[i]
		dailyCount[day]++
	}

	// Calculate averages
	for i := 0; i < 7; i++ {
		if dailyCount[i] > 0 {
			dailyAvg[i] /= float64(dailyCount[i])
		}
	}

	// Find peaks and troughs
	var peakDays []int
	var troughDays []int

	// Simple peak/trough detection
	for i := 0; i < 7; i++ {
		prev := (i + 6) % 7
		next := (i + 1) % 7

		// Peak if higher than both neighbors
		if dailyAvg[i] > dailyAvg[prev] && dailyAvg[i] > dailyAvg[next] {
			peakDays = append(peakDays, i)
		}

		// Trough if lower than both neighbors
		if dailyAvg[i] < dailyAvg[prev] && dailyAvg[i] < dailyAvg[next] {
			troughDays = append(troughDays, i)
		}
	}

	// If no clear peaks/troughs, look for weekday/weekend pattern
	if len(peakDays) == 0 || len(troughDays) == 0 {
		// Calculate weekday and weekend averages
		weekdayValues := make([]float64, 0)
		weekendValues := make([]float64, 0)

		for i := 0; i < 5; i++ {
			if dailyCount[i] > 0 {
				weekdayValues = append(weekdayValues, dailyAvg[i])
			}
		}

		for i := 5; i < 7; i++ {
			if dailyCount[i] > 0 {
				weekendValues = append(weekendValues, dailyAvg[i])
			}
		}

		weekdayAvg := average(weekdayValues)
		weekendAvg := average(weekendValues)

		// If weekday > weekend, weekdays are peaks
		if weekdayAvg > weekendAvg {
			peakDays = []int{1, 2, 3, 4, 5} // Mon-Fri
			troughDays = []int{0, 6}        // Sun, Sat
		} else {
			peakDays = []int{0, 6}            // Sun, Sat
			troughDays = []int{1, 2, 3, 4, 5} // Mon-Fri
		}
	}

	// Convert days to timestamps (use this week)
	today := time.Now()
	startOfWeek := today.AddDate(0, 0, -int(today.Weekday()))
	startOfWeek = time.Date(startOfWeek.Year(), startOfWeek.Month(), startOfWeek.Day(), 0, 0, 0, 0, startOfWeek.Location())

	peakTimestamps := make([]time.Time, len(peakDays))
	peakValues := make([]float64, len(peakDays))
	for i, day := range peakDays {
		peakTimestamps[i] = startOfWeek.AddDate(0, 0, day)
		peakValues[i] = dailyAvg[day]
	}

	troughTimestamps := make([]time.Time, len(troughDays))
	troughValues := make([]float64, len(troughDays))
	for i, day := range troughDays {
		troughTimestamps[i] = startOfWeek.AddDate(0, 0, day)
		troughValues[i] = dailyAvg[day]
	}

	// Calculate confidence
	confidence := 0.0
	if len(peakDays) > 0 && len(troughDays) > 0 {
		// Calculate average peak and trough values
		avgPeak := average(peakValues)
		avgTrough := average(troughValues)

		// Larger difference between peaks and troughs = stronger pattern
		diff := avgPeak - avgTrough
		maxDiff := 100.0 // Assuming values are in percentage
		confidence = math.Min(0.9, diff/maxDiff*0.8)
	}

	return WorkloadPattern{
		PatternType:      PatternTypeWeekly,
		CycleDuration:    7 * 24 * time.Hour,
		ConfidenceScore:  confidence,
		PeakTimestamps:   peakTimestamps,
		TroughTimestamps: troughTimestamps,
		PeakValues:       peakValues,
		TroughValues:     troughValues,
	}
}

// Detect bursty pattern
func detectBurstyPattern(timestamps []time.Time, values []float64) WorkloadPattern {
	if len(values) < 10 {
		return WorkloadPattern{
			PatternType:     PatternTypeBursty,
			ConfidenceScore: 0,
		}
	}

	// Calculate baseline and standard deviation
	baseline, stdDev := calculateBaseline(values)

	// Count the number of spikes (values > baseline + 2*stdDev)
	spikeCount := 0
	var spikeValues []float64
	var spikeTimestamps []time.Time

	for i, v := range values {
		if v > baseline+2*stdDev {
			spikeCount++
			spikeValues = append(spikeValues, v)
			spikeTimestamps = append(spikeTimestamps, timestamps[i])
		}
	}

	// Calculate spike frequency (spikes per hour)
	duration := timestamps[len(timestamps)-1].Sub(timestamps[0])
	hoursDuration := duration.Hours()
	if hoursDuration < 1 {
		hoursDuration = 1
	}

	spikeFrequency := float64(spikeCount) / hoursDuration

	// Calculate confidence based on spike frequency and magnitude
	frequencyConfidence := math.Min(0.9, spikeFrequency*0.5)

	magnitudeConfidence := 0.0
	if len(spikeValues) > 0 {
		// Average spike magnitude relative to baseline
		avgSpikeMagnitude := 0.0
		for _, v := range spikeValues {
			avgSpikeMagnitude += (v - baseline)
		}
		avgSpikeMagnitude /= float64(len(spikeValues))

		magnitudeConfidence = math.Min(0.9, avgSpikeMagnitude/100.0*0.8)
	}

	// Combined confidence
	confidence := (frequencyConfidence + magnitudeConfidence) / 2

	burstPattern := WorkloadPattern{
		PatternType:     PatternTypeBursty,
		ConfidenceScore: confidence,
		PeakTimestamps:  spikeTimestamps,
		PeakValues:      spikeValues,
		Parameters:      make(map[string]float64),
	}

	// Add parameters for bursty workload
	burstPattern.Parameters["baseline"] = baseline
	burstPattern.Parameters["stddev"] = stdDev
	burstPattern.Parameters["spike_frequency"] = spikeFrequency
	burstPattern.Parameters["amplitude"] = stdDev * 2

	return burstPattern
}
