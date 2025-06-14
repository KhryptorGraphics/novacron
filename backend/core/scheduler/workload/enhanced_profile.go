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

// GetAverageUsage returns the average usage for a resource type
func (p *WorkloadProfileAdapter) GetAverageUsage(resourceType string) float64 {
	if stats, ok := p.ResourceUsage[resourceType]; ok {
		return stats.AverageUsage
	}
	return 0
}

// GetPeakUsage returns the peak usage for a resource type
func (p *WorkloadProfileAdapter) GetPeakUsage(resourceType string) float64 {
	if stats, ok := p.ResourceUsage[resourceType]; ok {
		return stats.PeakUsage
	}
	return 0
}

// GetMinimumUsage returns the minimum usage for a resource type
func (p *WorkloadProfileAdapter) GetMinimumUsage(resourceType string) float64 {
	if stats, ok := p.ResourceUsage[resourceType]; ok {
		return stats.MinimumUsage
	}
	return 0
}

// GetStandardDeviation returns the standard deviation for a resource type
func (p *WorkloadProfileAdapter) GetStandardDeviation(resourceType string) float64 {
	if stats, ok := p.ResourceUsage[resourceType]; ok {
		return stats.StandardDeviation
	}
	return 0
}

// IsStableWorkload returns true if the workload is stable
func (p *WorkloadProfileAdapter) IsStableWorkload() bool {
	// Consider a workload stable if CPU standard deviation is < 20% of average
	cpuStats, ok := p.ResourceUsage["cpu"]
	if !ok || cpuStats.AverageUsage == 0 {
		return false
	}
	
	return cpuStats.StandardDeviation/cpuStats.AverageUsage < 0.2
}

// GetPatterns analyzes the workload and returns detected patterns
func (p *WorkloadProfileAdapter) GetPatterns() []WorkloadPattern {
	var patterns []WorkloadPattern
	
	// Analyze each resource type
	for resourceType, stats := range p.ResourceUsage {
		if len(stats.Samples) < 10 {
			continue // Not enough data
		}
		
		// Simple pattern detection based on variance
		if stats.StandardDeviation/stats.AverageUsage < 0.1 {
			// Steady pattern
			patterns = append(patterns, WorkloadPattern{
				Type:         SteadyPattern,
				ResourceType: resourceType,
				Confidence:   0.8,
				Duration:     p.HistoryDuration,
			})
		} else if stats.PeakUsage > stats.AverageUsage*2 {
			// Burst pattern
			patterns = append(patterns, WorkloadPattern{
				Type:         BurstPattern,
				ResourceType: resourceType,
				Confidence:   0.7,
				Duration:     p.HistoryDuration,
			})
		}
	}
	
	return patterns
}

// UpdateStats updates the resource usage statistics
func (p *WorkloadProfileAdapter) UpdateStats(resourceType string, samples []ResourceSample) {
	if len(samples) == 0 {
		return
	}
	
	// Calculate statistics
	var sum, min, max float64
	min = math.MaxFloat64
	
	for _, sample := range samples {
		sum += sample.Value
		if sample.Value < min {
			min = sample.Value
		}
		if sample.Value > max {
			max = sample.Value
		}
	}
	
	avg := sum / float64(len(samples))
	
	// Calculate standard deviation
	var varianceSum float64
	for _, sample := range samples {
		diff := sample.Value - avg
		varianceSum += diff * diff
	}
	stdDev := math.Sqrt(varianceSum / float64(len(samples)))
	
	// Update stats
	if p.ResourceUsage == nil {
		p.ResourceUsage = make(map[string]ResourceUsageStats)
	}
	
	p.ResourceUsage[resourceType] = ResourceUsageStats{
		AverageUsage:      avg,
		PeakUsage:         max,
		MinimumUsage:      min,
		StandardDeviation: stdDev,
		Samples:           samples,
	}
	
	p.LastUpdated = time.Now()
}

// GetRecommendedResources returns recommended resource allocations
func (p *WorkloadProfileAdapter) GetRecommendedResources() map[string]float64 {
	recommendations := make(map[string]float64)
	
	for resourceType, stats := range p.ResourceUsage {
		// Recommend peak usage + 20% buffer
		recommendations[resourceType] = stats.PeakUsage * 1.2
	}
	
	return recommendations
}

// GetPredictedUsage predicts future usage based on historical patterns
func (p *WorkloadProfileAdapter) GetPredictedUsage(resourceType string, futureTime time.Duration) float64 {
	stats, ok := p.ResourceUsage[resourceType]
	if !ok || len(stats.Samples) < 2 {
		return 0
	}
	
	// Simple linear prediction based on recent trend
	n := len(stats.Samples)
	if n < 2 {
		return stats.AverageUsage
	}
	
	// Get the last few samples
	recentSamples := stats.Samples
	if n > 10 {
		recentSamples = stats.Samples[n-10:]
	}
	
	// Calculate trend
	firstValue := recentSamples[0].Value
	lastValue := recentSamples[len(recentSamples)-1].Value
	timeDiff := recentSamples[len(recentSamples)-1].Timestamp.Sub(recentSamples[0].Timestamp).Seconds()
	
	if timeDiff == 0 {
		return stats.AverageUsage
	}
	
	trend := (lastValue - firstValue) / timeDiff
	
	// Project into future
	predicted := lastValue + trend*futureTime.Seconds()
	
	// Bound the prediction
	if predicted < 0 {
		predicted = 0
	}
	if predicted > 100 {
		predicted = 100
	}
	
	return predicted
}

// GetHistoricalSamples returns historical samples for a resource type
func (p *WorkloadProfileAdapter) GetHistoricalSamples(resourceType string) []ResourceSample {
	if stats, ok := p.ResourceUsage[resourceType]; ok {
		return stats.Samples
	}
	return nil
}

// Clone creates a copy of the workload profile
func (p *WorkloadProfileAdapter) Clone() *WorkloadProfileAdapter {
	clone := &WorkloadProfileAdapter{
		VMID:            p.VMID,
		HistoryDuration: p.HistoryDuration,
		LastUpdated:     p.LastUpdated,
		ResourceUsage:   make(map[string]ResourceUsageStats),
	}
	
	for k, v := range p.ResourceUsage {
		samples := make([]ResourceSample, len(v.Samples))
		copy(samples, v.Samples)
		
		clone.ResourceUsage[k] = ResourceUsageStats{
			AverageUsage:      v.AverageUsage,
			PeakUsage:         v.PeakUsage,
			MinimumUsage:      v.MinimumUsage,
			StandardDeviation: v.StandardDeviation,
			Samples:           samples,
		}
	}
	
	return clone
}

// MergeWith merges another profile into this one
func (p *WorkloadProfileAdapter) MergeWith(other *WorkloadProfileAdapter) {
	for resourceType, otherStats := range other.ResourceUsage {
		if existingStats, ok := p.ResourceUsage[resourceType]; ok {
			// Merge samples
			allSamples := append(existingStats.Samples, otherStats.Samples...)
			
			// Sort by timestamp
			sort.Slice(allSamples, func(i, j int) bool {
				return allSamples[i].Timestamp.Before(allSamples[j].Timestamp)
			})
			
			// Update stats with merged samples
			p.UpdateStats(resourceType, allSamples)
		} else {
			// Add new resource type
			p.ResourceUsage[resourceType] = otherStats
		}
	}
	
	p.LastUpdated = time.Now()
}

// NewEnhancedProfile creates a new enhanced workload profile
func NewEnhancedProfile(vmID string) *EnhancedWorkloadProfile {
	return &EnhancedWorkloadProfile{
		VMID:               vmID,
		ResourceProfiles:   make(map[string]*ResourceProfile),
		RecognizedPatterns: make(map[string][]WorkloadPattern),
		WorkloadStability:  0.5, // Default medium stability
		LastUpdated:        time.Now(),
		SampleCount:        0,
	}
}

// IsStableWorkload returns true if the workload is stable
func (p *EnhancedWorkloadProfile) IsStableWorkload() bool {
	return p.WorkloadStability >= 0.6
}

// SetWorkloadProfile updates the enhanced profile from a WorkloadProfileAdapter
func (p *EnhancedWorkloadProfile) SetWorkloadProfile(profile *WorkloadProfileAdapter) {
	p.LastUpdated = profile.LastUpdated
	
	// Convert resource usage to resource profiles
	for resourceType, stats := range profile.ResourceUsage {
		p.ResourceProfiles[resourceType] = &ResourceProfile{
			ResourceType:      resourceType,
			AverageUsage:      stats.AverageUsage,
			PeakUsage:         stats.PeakUsage,
			MinimumUsage:      stats.MinimumUsage,
			StandardDeviation: stats.StandardDeviation,
			PredictionModel:   make(map[string]interface{}),
		}
		
		p.SampleCount += len(stats.Samples)
	}
	
	// Detect patterns
	patterns := profile.GetPatterns()
	for _, pattern := range patterns {
		if p.RecognizedPatterns[pattern.ResourceType] == nil {
			p.RecognizedPatterns[pattern.ResourceType] = []WorkloadPattern{}
		}
		p.RecognizedPatterns[pattern.ResourceType] = append(
			p.RecognizedPatterns[pattern.ResourceType], 
			pattern,
		)
	}
	
	// Calculate workload stability
	p.calculateStability()
}

// calculateStability calculates the overall workload stability
func (p *EnhancedWorkloadProfile) calculateStability() {
	if len(p.ResourceProfiles) == 0 {
		p.WorkloadStability = 0.5
		return
	}
	
	var totalStability float64
	for _, profile := range p.ResourceProfiles {
		if profile.AverageUsage > 0 {
			// Lower coefficient of variation means higher stability
			cv := profile.StandardDeviation / profile.AverageUsage
			stability := 1.0 - math.Min(cv, 1.0)
			totalStability += stability
		}
	}
	
	p.WorkloadStability = totalStability / float64(len(p.ResourceProfiles))
}

// GetResourceProfile returns the resource profile for a specific resource type
func (p *EnhancedWorkloadProfile) GetResourceProfile(resourceType string) *ResourceProfile {
	return p.ResourceProfiles[resourceType]
}

// GetPatterns returns all recognized patterns for a resource type
func (p *EnhancedWorkloadProfile) GetPatterns(resourceType string) []WorkloadPattern {
	return p.RecognizedPatterns[resourceType]
}

// GetAllPatterns returns all recognized patterns
func (p *EnhancedWorkloadProfile) GetAllPatterns() []WorkloadPattern {
	var allPatterns []WorkloadPattern
	for _, patterns := range p.RecognizedPatterns {
		allPatterns = append(allPatterns, patterns...)
	}
	return allPatterns
}

// GetRecommendedResources returns recommended resource allocations
func (p *EnhancedWorkloadProfile) GetRecommendedResources() map[string]float64 {
	recommendations := make(map[string]float64)
	
	for resourceType, profile := range p.ResourceProfiles {
		// Base recommendation on peak usage with stability-based buffer
		buffer := 1.2 // 20% buffer
		if p.WorkloadStability < 0.5 {
			buffer = 1.5 // 50% buffer for unstable workloads
		}
		
		recommendations[resourceType] = profile.PeakUsage * buffer
	}
	
	return recommendations
}

// TimeSeriesData represents time series data for pattern detection
type TimeSeriesData struct {
	ResourceType string
	Timestamps   []time.Time
	Values       []float64
}

// PatternDetector analyzes time series data to detect workload patterns
type PatternDetector struct {
	windowSize      int
	minConfidence   float64
	seasonalPeriods []time.Duration
}

// NewPatternDetector creates a new pattern detector
func NewPatternDetector() *PatternDetector {
	return &PatternDetector{
		windowSize:    20,
		minConfidence: 0.6,
		seasonalPeriods: []time.Duration{
			24 * time.Hour,       // Daily
			7 * 24 * time.Hour,   // Weekly
			30 * 24 * time.Hour,  // Monthly
		},
	}
}

// DetectPatterns analyzes time series data and returns detected patterns
func (d *PatternDetector) DetectPatterns(timestamps []time.Time, values []float64) ([]WorkloadPattern, error) {
	if len(timestamps) != len(values) || len(values) < d.windowSize {
		return nil, nil // Not enough data
	}
	
	var patterns []WorkloadPattern
	
	// Detect steady pattern
	if pattern := d.detectSteadyPattern(values); pattern != nil {
		patterns = append(patterns, *pattern)
	}
	
	// Detect burst pattern
	if pattern := d.detectBurstPattern(timestamps, values); pattern != nil {
		patterns = append(patterns, *pattern)
	}
	
	// Detect periodic pattern
	if pattern := d.detectPeriodicPattern(timestamps, values); pattern != nil {
		patterns = append(patterns, *pattern)
	}
	
	// Detect growth pattern
	if pattern := d.detectGrowthPattern(timestamps, values); pattern != nil {
		patterns = append(patterns, *pattern)
	}
	
	return patterns, nil
}

// detectSteadyPattern detects if the workload is steady
func (d *PatternDetector) detectSteadyPattern(values []float64) *WorkloadPattern {
	if len(values) == 0 {
		return nil
	}
	
	// Calculate mean and standard deviation
	var sum float64
	for _, v := range values {
		sum += v
	}
	mean := sum / float64(len(values))
	
	var varianceSum float64
	for _, v := range values {
		diff := v - mean
		varianceSum += diff * diff
	}
	stdDev := math.Sqrt(varianceSum / float64(len(values)))
	
	// Check if coefficient of variation is low
	if mean > 0 && stdDev/mean < 0.1 {
		return &WorkloadPattern{
			Type:       SteadyPattern,
			Confidence: math.Max(0.9-stdDev/mean, d.minConfidence),
		}
	}
	
	return nil
}

// detectBurstPattern detects burst patterns in the workload
func (d *PatternDetector) detectBurstPattern(timestamps []time.Time, values []float64) *WorkloadPattern {
	if len(values) < 10 {
		return nil
	}
	
	// Calculate baseline (median of lower 75% of values)
	sorted := make([]float64, len(values))
	copy(sorted, values)
	sort.Float64s(sorted)
	baseline := sorted[len(sorted)*3/4]
	
	// Count bursts (values > 2x baseline)
	burstCount := 0
	for _, v := range values {
		if v > baseline*2 {
			burstCount++
		}
	}
	
	burstRatio := float64(burstCount) / float64(len(values))
	if burstRatio > 0.05 && burstRatio < 0.3 {
		// Calculate average burst intensity
		var burstSum float64
		var burstValues int
		for _, v := range values {
			if v > baseline*2 {
				burstSum += v / baseline
				burstValues++
			}
		}
		
		avgBurstIntensity := 1.0
		if burstValues > 0 {
			avgBurstIntensity = burstSum / float64(burstValues)
		}
		
		pattern := &WorkloadPattern{
			Type:       BurstPattern,
			Confidence: math.Min(0.8, d.minConfidence+burstRatio),
		}
		
		// Store burst characteristics in a simple way
		pattern.Duration = time.Duration(avgBurstIntensity * float64(time.Hour))
		
		return pattern
	}
	
	return nil
}

// detectPeriodicPattern detects periodic patterns
func (d *PatternDetector) detectPeriodicPattern(timestamps []time.Time, values []float64) *WorkloadPattern {
	// Simple periodicity detection - check for daily patterns
	if len(values) < 48 { // Need at least 2 days of hourly data
		return nil
	}
	
	// Group values by hour of day
	hourlyAvg := make(map[int][]float64)
	for i, ts := range timestamps {
		hour := ts.Hour()
		hourlyAvg[hour] = append(hourlyAvg[hour], values[i])
	}
	
	// Calculate variance across hours
	var hourlyMeans []float64
	for hour := 0; hour < 24; hour++ {
		if vals, ok := hourlyAvg[hour]; ok && len(vals) > 0 {
			sum := 0.0
			for _, v := range vals {
				sum += v
			}
			hourlyMeans = append(hourlyMeans, sum/float64(len(vals)))
		}
	}
	
	if len(hourlyMeans) < 12 {
		return nil
	}
	
	// Check if there's significant variation by hour
	var min, max float64 = math.MaxFloat64, 0
	for _, mean := range hourlyMeans {
		if mean < min {
			min = mean
		}
		if mean > max {
			max = mean
		}
	}
	
	if max > min*1.5 {
		return &WorkloadPattern{
			Type:       PeriodicPattern,
			Confidence: math.Min(0.7, (max-min)/max),
			Duration:   24 * time.Hour,
		}
	}
	
	return nil
}

// detectGrowthPattern detects growth or decline patterns
func (d *PatternDetector) detectGrowthPattern(timestamps []time.Time, values []float64) *WorkloadPattern {
	if len(values) < d.windowSize {
		return nil
	}
	
	// Simple linear regression to detect trend
	n := float64(len(values))
	var sumX, sumY, sumXY, sumX2 float64
	
	for i, v := range values {
		x := float64(i)
		sumX += x
		sumY += v
		sumXY += x * v
		sumX2 += x * x
	}
	
	// Calculate slope
	slope := (n*sumXY - sumX*sumY) / (n*sumX2 - sumX*sumX)
	
	// Calculate R-squared for confidence
	meanY := sumY / n
	var ssTotal, ssResidual float64
	
	for i, v := range values {
		predicted := slope*float64(i) + (sumY-slope*sumX)/n
		ssTotal += (v - meanY) * (v - meanY)
		ssResidual += (v - predicted) * (v - predicted)
	}
	
	rSquared := 1 - ssResidual/ssTotal
	
	// Significant positive or negative trend
	if math.Abs(slope) > 0.1 && rSquared > 0.5 {
		patternType := GrowthPattern
		if slope < 0 {
			patternType = DeclinePattern
		}
		
		return &WorkloadPattern{
			Type:       patternType,
			Confidence: math.Min(rSquared, 0.9),
		}
	}
	
	return nil
}