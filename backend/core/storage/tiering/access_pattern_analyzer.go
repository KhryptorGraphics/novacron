package tiering

import (
	"context"
	"fmt"
	"math"
	"sync"
	"time"
)

// AccessPatternAnalyzer uses machine learning patterns to detect hot/cold data
type AccessPatternAnalyzer struct {
	// Historical access data
	accessHistory map[string]*AccessHistory
	// Prediction models
	models map[string]PredictionModel
	// Configuration
	config AnalyzerConfig
	// Mutex for thread safety
	mu sync.RWMutex
	// Metrics
	metrics *AnalyzerMetrics
}

// AccessHistory tracks access patterns for a volume
type AccessHistory struct {
	VolumeID         string
	AccessTimestamps []time.Time
	AccessSizes      []int64
	AccessTypes      []AccessType
	// Time series data for ML
	HourlyAccess    [24]int     // Access count per hour of day
	DailyAccess     [7]int      // Access count per day of week
	WeeklyPattern   [4]float64  // Weekly access pattern (4 weeks)
	// Statistical features
	Mean            float64
	StdDev          float64
	Variance        float64
	AccessBurstiness float64
	// Prediction data
	PredictedTemp   Temperature
	ConfidenceScore float64
	LastUpdated     time.Time
}

// AccessType represents the type of access
type AccessType int

const (
	AccessTypeRead AccessType = iota
	AccessTypeWrite
	AccessTypeSequential
	AccessTypeRandom
)

// Temperature represents data temperature
type Temperature int

const (
	TemperatureHot Temperature = iota
	TemperatureWarm
	TemperatureCold
	TemperatureFrozen
)

// AnalyzerConfig defines configuration for the access pattern analyzer
type AnalyzerConfig struct {
	// History retention period
	HistoryRetentionDays int
	// Minimum access count for analysis
	MinAccessCount int
	// Hot data threshold (accesses per day)
	HotThreshold float64
	// Warm data threshold (accesses per day)
	WarmThreshold float64
	// Cold data threshold (accesses per day)
	ColdThreshold float64
	// Enable ML predictions
	EnableMLPredictions bool
	// Prediction window (hours)
	PredictionWindow int
	// Burst detection sensitivity (0-1)
	BurstSensitivity float64
}

// PredictionModel interface for different ML models
type PredictionModel interface {
	Predict(history *AccessHistory) (Temperature, float64)
	Train(histories []*AccessHistory)
	GetName() string
}

// AnalyzerMetrics tracks analyzer performance
type AnalyzerMetrics struct {
	TotalAnalyses       int64
	CorrectPredictions  int64
	IncorrectPredictions int64
	PredictionAccuracy  float64
	AnalysisLatency     time.Duration
	mu                  sync.RWMutex
}

// NewAccessPatternAnalyzer creates a new access pattern analyzer
func NewAccessPatternAnalyzer() *AccessPatternAnalyzer {
	config := AnalyzerConfig{
		HistoryRetentionDays: 30,
		MinAccessCount:      10,
		HotThreshold:        100.0,  // 100+ accesses/day = hot
		WarmThreshold:       10.0,   // 10-100 accesses/day = warm
		ColdThreshold:       1.0,    // 1-10 accesses/day = cold
		EnableMLPredictions: true,
		PredictionWindow:    24,     // 24 hours ahead
		BurstSensitivity:   0.7,
	}

	analyzer := &AccessPatternAnalyzer{
		accessHistory: make(map[string]*AccessHistory),
		models:       make(map[string]PredictionModel),
		config:       config,
		metrics:      &AnalyzerMetrics{},
	}

	// Initialize ML models
	analyzer.models["exponential_smoothing"] = NewExponentialSmoothingModel()
	analyzer.models["markov_chain"] = NewMarkovChainModel()
	analyzer.models["neural_network"] = NewSimpleNeuralNetworkModel()

	return analyzer
}

// RecordAccess records an access event for analysis
func (apa *AccessPatternAnalyzer) RecordAccess(volumeID string, accessType AccessType, size int64) {
	apa.mu.Lock()
	defer apa.mu.Unlock()

	history, exists := apa.accessHistory[volumeID]
	if !exists {
		history = &AccessHistory{
			VolumeID:         volumeID,
			AccessTimestamps: make([]time.Time, 0),
			AccessSizes:      make([]int64, 0),
			AccessTypes:      make([]AccessType, 0),
			LastUpdated:      time.Now(),
		}
		apa.accessHistory[volumeID] = history
	}

	now := time.Now()
	history.AccessTimestamps = append(history.AccessTimestamps, now)
	history.AccessSizes = append(history.AccessSizes, size)
	history.AccessTypes = append(history.AccessTypes, accessType)

	// Update hourly and daily patterns
	hour := now.Hour()
	dayOfWeek := int(now.Weekday())
	history.HourlyAccess[hour]++
	history.DailyAccess[dayOfWeek]++

	// Update weekly pattern (rolling 4-week window)
	weekIndex := (now.Unix() / 604800) % 4 // Week number mod 4
	history.WeeklyPattern[weekIndex]++

	// Recalculate statistics
	apa.updateStatistics(history)
}

// updateStatistics recalculates statistical features for a volume
func (apa *AccessPatternAnalyzer) updateStatistics(history *AccessHistory) {
	if len(history.AccessTimestamps) < 2 {
		return
	}

	// Calculate inter-arrival times
	interArrivals := make([]float64, 0)
	for i := 1; i < len(history.AccessTimestamps); i++ {
		interval := history.AccessTimestamps[i].Sub(history.AccessTimestamps[i-1]).Seconds()
		interArrivals = append(interArrivals, interval)
	}

	// Calculate mean
	sum := 0.0
	for _, interval := range interArrivals {
		sum += interval
	}
	history.Mean = sum / float64(len(interArrivals))

	// Calculate variance and standard deviation
	sumSquaredDiff := 0.0
	for _, interval := range interArrivals {
		diff := interval - history.Mean
		sumSquaredDiff += diff * diff
	}
	history.Variance = sumSquaredDiff / float64(len(interArrivals))
	history.StdDev = math.Sqrt(history.Variance)

	// Calculate burstiness (coefficient of variation squared)
	if history.Mean > 0 {
		cv := history.StdDev / history.Mean
		history.AccessBurstiness = cv * cv
	}
}

// AnalyzeVolume performs comprehensive analysis on a volume's access patterns
func (apa *AccessPatternAnalyzer) AnalyzeVolume(ctx context.Context, volumeID string) (*AccessAnalysis, error) {
	apa.mu.RLock()
	history, exists := apa.accessHistory[volumeID]
	apa.mu.RUnlock()

	if !exists || len(history.AccessTimestamps) < apa.config.MinAccessCount {
		return &AccessAnalysis{
			VolumeID:    volumeID,
			Temperature: TemperatureCold,
			Confidence:  0.5,
			Recommendation: TierCold,
		}, nil
	}

	// Calculate access frequency
	accessFrequency := apa.calculateAccessFrequency(history)

	// Determine temperature based on thresholds
	temperature := apa.classifyTemperature(accessFrequency)

	// Apply ML predictions if enabled
	var mlTemperature Temperature
	var confidence float64
	if apa.config.EnableMLPredictions {
		mlTemperature, confidence = apa.applyMLModels(history)
		
		// Combine rule-based and ML predictions
		if confidence > 0.7 {
			temperature = mlTemperature
		}
	}

	// Detect access patterns
	patterns := apa.detectPatterns(history)

	// Generate tier recommendation
	recommendation := apa.recommendTier(temperature, patterns, history)

	// Update metrics
	apa.metrics.mu.Lock()
	apa.metrics.TotalAnalyses++
	apa.metrics.mu.Unlock()

	return &AccessAnalysis{
		VolumeID:       volumeID,
		Temperature:    temperature,
		Confidence:     confidence,
		AccessFrequency: accessFrequency,
		Patterns:       patterns,
		Recommendation: recommendation,
		Burstiness:    history.AccessBurstiness,
		LastAnalyzed:  time.Now(),
	}, nil
}

// AccessAnalysis represents the result of analyzing access patterns
type AccessAnalysis struct {
	VolumeID        string
	Temperature     Temperature
	Confidence      float64
	AccessFrequency float64
	Patterns        []Pattern
	Recommendation  TierLevel
	Burstiness     float64
	LastAnalyzed   time.Time
}

// Pattern represents a detected access pattern
type Pattern struct {
	Type        PatternType
	Strength    float64 // 0-1 indicating pattern strength
	Description string
}

// PatternType represents types of access patterns
type PatternType int

const (
	PatternPeriodic PatternType = iota
	PatternBursty
	PatternSequential
	PatternRandom
	PatternTimeOfDay
	PatternDayOfWeek
)

// calculateAccessFrequency calculates access frequency in accesses per day
func (apa *AccessPatternAnalyzer) calculateAccessFrequency(history *AccessHistory) float64 {
	if len(history.AccessTimestamps) == 0 {
		return 0
	}

	// Calculate time span
	firstAccess := history.AccessTimestamps[0]
	lastAccess := history.AccessTimestamps[len(history.AccessTimestamps)-1]
	daySpan := lastAccess.Sub(firstAccess).Hours() / 24.0

	if daySpan < 1 {
		daySpan = 1
	}

	return float64(len(history.AccessTimestamps)) / daySpan
}

// classifyTemperature classifies data temperature based on access frequency
func (apa *AccessPatternAnalyzer) classifyTemperature(frequency float64) Temperature {
	if frequency >= apa.config.HotThreshold {
		return TemperatureHot
	} else if frequency >= apa.config.WarmThreshold {
		return TemperatureWarm
	} else if frequency >= apa.config.ColdThreshold {
		return TemperatureCold
	}
	return TemperatureFrozen
}

// applyMLModels applies machine learning models for prediction
func (apa *AccessPatternAnalyzer) applyMLModels(history *AccessHistory) (Temperature, float64) {
	predictions := make(map[Temperature]float64)

	// Get predictions from all models
	for _, model := range apa.models {
		temp, conf := model.Predict(history)
		predictions[temp] += conf
	}

	// Find the temperature with highest combined confidence
	var bestTemp Temperature
	var bestConf float64
	for temp, conf := range predictions {
		avgConf := conf / float64(len(apa.models))
		if avgConf > bestConf {
			bestTemp = temp
			bestConf = avgConf
		}
	}

	return bestTemp, bestConf
}

// detectPatterns detects various access patterns in the data
func (apa *AccessPatternAnalyzer) detectPatterns(history *AccessHistory) []Pattern {
	patterns := make([]Pattern, 0)

	// Detect time-of-day pattern
	if todPattern := apa.detectTimeOfDayPattern(history); todPattern.Strength > 0.5 {
		patterns = append(patterns, todPattern)
	}

	// Detect day-of-week pattern
	if dowPattern := apa.detectDayOfWeekPattern(history); dowPattern.Strength > 0.5 {
		patterns = append(patterns, dowPattern)
	}

	// Detect burstiness
	if history.AccessBurstiness > apa.config.BurstSensitivity {
		patterns = append(patterns, Pattern{
			Type:        PatternBursty,
			Strength:    math.Min(history.AccessBurstiness, 1.0),
			Description: "Bursty access pattern detected",
		})
	}

	// Detect sequential vs random
	if seqPattern := apa.detectSequentialPattern(history); seqPattern.Strength > 0.5 {
		patterns = append(patterns, seqPattern)
	}

	return patterns
}

// detectTimeOfDayPattern detects if access follows time-of-day patterns
func (apa *AccessPatternAnalyzer) detectTimeOfDayPattern(history *AccessHistory) Pattern {
	// Calculate variance in hourly access
	var mean float64
	for _, count := range history.HourlyAccess {
		mean += float64(count)
	}
	mean /= 24

	var variance float64
	for _, count := range history.HourlyAccess {
		diff := float64(count) - mean
		variance += diff * diff
	}
	variance /= 24

	// High variance indicates strong time-of-day pattern
	strength := math.Min(variance/(mean*mean+1), 1.0)

	return Pattern{
		Type:        PatternTimeOfDay,
		Strength:    strength,
		Description: "Access concentrated at specific hours",
	}
}

// detectDayOfWeekPattern detects if access follows day-of-week patterns
func (apa *AccessPatternAnalyzer) detectDayOfWeekPattern(history *AccessHistory) Pattern {
	// Calculate variance in daily access
	var mean float64
	for _, count := range history.DailyAccess {
		mean += float64(count)
	}
	mean /= 7

	var variance float64
	for _, count := range history.DailyAccess {
		diff := float64(count) - mean
		variance += diff * diff
	}
	variance /= 7

	strength := math.Min(variance/(mean*mean+1), 1.0)

	return Pattern{
		Type:        PatternDayOfWeek,
		Strength:    strength,
		Description: "Access concentrated on specific days",
	}
}

// detectSequentialPattern detects sequential vs random access
func (apa *AccessPatternAnalyzer) detectSequentialPattern(history *AccessHistory) Pattern {
	sequential := 0
	random := 0

	for _, accessType := range history.AccessTypes {
		if accessType == AccessTypeSequential {
			sequential++
		} else if accessType == AccessTypeRandom {
			random++
		}
	}

	total := sequential + random
	if total == 0 {
		return Pattern{Type: PatternRandom, Strength: 0}
	}

	seqRatio := float64(sequential) / float64(total)
	if seqRatio > 0.7 {
		return Pattern{
			Type:        PatternSequential,
			Strength:    seqRatio,
			Description: "Predominantly sequential access",
		}
	}

	return Pattern{
		Type:        PatternRandom,
		Strength:    1 - seqRatio,
		Description: "Predominantly random access",
	}
}

// recommendTier recommends the appropriate storage tier
func (apa *AccessPatternAnalyzer) recommendTier(temp Temperature, patterns []Pattern, history *AccessHistory) TierLevel {
	// Base recommendation on temperature
	var baseTier TierLevel
	switch temp {
	case TemperatureHot:
		baseTier = TierHot
	case TemperatureWarm:
		baseTier = TierWarm
	case TemperatureCold:
		baseTier = TierCold
	case TemperatureFrozen:
		baseTier = TierCold
	}

	// Adjust based on patterns
	for _, pattern := range patterns {
		if pattern.Type == PatternBursty && pattern.Strength > 0.8 {
			// Bursty patterns might benefit from hot tier
			if baseTier > TierHot {
				baseTier = TierHot
			}
		} else if pattern.Type == PatternTimeOfDay && pattern.Strength > 0.7 {
			// Predictable patterns might work well in warm tier
			if baseTier == TierCold {
				baseTier = TierWarm
			}
		}
	}

	return baseTier
}

// PredictFutureAccess predicts future access patterns
func (apa *AccessPatternAnalyzer) PredictFutureAccess(volumeID string, hours int) (*AccessPrediction, error) {
	apa.mu.RLock()
	history, exists := apa.accessHistory[volumeID]
	apa.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("no history for volume %s", volumeID)
	}

	// Use ensemble of models for prediction
	predictions := make([]float64, hours)
	
	// Simple moving average as baseline
	recentAccesses := apa.getRecentAccessCount(history, 24)
	baselineRate := float64(recentAccesses) / 24.0

	for i := 0; i < hours; i++ {
		predictions[i] = baselineRate
		
		// Adjust for time-of-day pattern
		futureHour := (time.Now().Hour() + i) % 24
		hourlyAvg := float64(history.HourlyAccess[futureHour]) / float64(math.Max(1, float64(len(history.AccessTimestamps))/24))
		predictions[i] *= (1 + hourlyAvg)
		
		// Adjust for day-of-week pattern
		futureDayOfWeek := int((time.Now().AddDate(0, 0, i/24)).Weekday())
		dailyAvg := float64(history.DailyAccess[futureDayOfWeek]) / float64(math.Max(1, float64(len(history.AccessTimestamps))/7))
		predictions[i] *= (1 + dailyAvg*0.5)
	}

	// Calculate confidence based on data quality
	confidence := math.Min(float64(len(history.AccessTimestamps))/100.0, 1.0)

	return &AccessPrediction{
		VolumeID:          volumeID,
		PredictedAccesses: predictions,
		Confidence:        confidence,
		TimeHorizon:       hours,
	}, nil
}

// AccessPrediction represents predicted future access
type AccessPrediction struct {
	VolumeID          string
	PredictedAccesses []float64
	Confidence        float64
	TimeHorizon       int
}

// getRecentAccessCount gets access count in recent hours
func (apa *AccessPatternAnalyzer) getRecentAccessCount(history *AccessHistory, hours int) int {
	cutoff := time.Now().Add(-time.Duration(hours) * time.Hour)
	count := 0
	
	for _, timestamp := range history.AccessTimestamps {
		if timestamp.After(cutoff) {
			count++
		}
	}
	
	return count
}

// CleanupOldData removes old access history data
func (apa *AccessPatternAnalyzer) CleanupOldData() {
	apa.mu.Lock()
	defer apa.mu.Unlock()

	cutoff := time.Now().AddDate(0, 0, -apa.config.HistoryRetentionDays)

	for volumeID, history := range apa.accessHistory {
		// Filter out old timestamps
		newTimestamps := make([]time.Time, 0)
		newSizes := make([]int64, 0)
		newTypes := make([]AccessType, 0)

		for i, timestamp := range history.AccessTimestamps {
			if timestamp.After(cutoff) {
				newTimestamps = append(newTimestamps, timestamp)
				newSizes = append(newSizes, history.AccessSizes[i])
				newTypes = append(newTypes, history.AccessTypes[i])
			}
		}

		if len(newTimestamps) == 0 {
			delete(apa.accessHistory, volumeID)
		} else {
			history.AccessTimestamps = newTimestamps
			history.AccessSizes = newSizes
			history.AccessTypes = newTypes
			apa.updateStatistics(history)
		}
	}
}

// GetMetrics returns analyzer metrics
func (apa *AccessPatternAnalyzer) GetMetrics() map[string]interface{} {
	apa.metrics.mu.RLock()
	defer apa.metrics.mu.RUnlock()

	accuracy := float64(0)
	if apa.metrics.TotalAnalyses > 0 {
		accuracy = float64(apa.metrics.CorrectPredictions) / float64(apa.metrics.CorrectPredictions+apa.metrics.IncorrectPredictions)
	}

	return map[string]interface{}{
		"total_analyses":        apa.metrics.TotalAnalyses,
		"correct_predictions":   apa.metrics.CorrectPredictions,
		"incorrect_predictions": apa.metrics.IncorrectPredictions,
		"prediction_accuracy":   accuracy,
		"analysis_latency_ms":   apa.metrics.AnalysisLatency.Milliseconds(),
		"volumes_tracked":       len(apa.accessHistory),
	}
}