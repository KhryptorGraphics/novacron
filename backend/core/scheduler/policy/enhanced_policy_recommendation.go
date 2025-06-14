package policy

import (
	"context"
	"fmt"
	"log"
	"math"
	"sort"
	"time"
)

// EnhancedPolicyRecommendationEngine extends the basic PolicyRecommendationEngine
// with advanced features such as ML-based recommendations, historical analysis,
// and recommendation quality tracking
type EnhancedPolicyRecommendationEngine struct {
	// Base recommendation engine
	*PolicyRecommendationEngine
	
	// HistoricalAnalyzer analyzes historical data
	HistoricalAnalyzer *PolicyHistoricalAnalyzer
	
	// MLRecommender provides ML-based recommendations
	MLRecommender *PolicyMLRecommender
	
	// QualityTracker tracks recommendation quality
	QualityTracker *RecommendationQualityTracker
}

// NewEnhancedPolicyRecommendationEngine creates a new enhanced policy recommendation engine
func NewEnhancedPolicyRecommendationEngine(engine *PolicyEngine) *EnhancedPolicyRecommendationEngine {
	baseEngine := NewPolicyRecommendationEngine(engine)
	
	return &EnhancedPolicyRecommendationEngine{
		PolicyRecommendationEngine: baseEngine,
		HistoricalAnalyzer:        NewPolicyHistoricalAnalyzer(),
		MLRecommender:             NewPolicyMLRecommender(),
		QualityTracker:            NewRecommendationQualityTracker(),
	}
}

// PolicyHistoricalAnalyzer analyzes historical policy performance data
type PolicyHistoricalAnalyzer struct {
	// PerformanceData stores historical performance data
	PerformanceData []*PolicyPerformanceData
	
	// Patterns stores identified patterns
	Patterns []*PolicyPattern
}

// PolicyPerformanceData represents historical policy performance data
type PolicyPerformanceData struct {
	// Timestamp is when the data was collected
	Timestamp time.Time
	
	// PolicyID is the ID of the policy
	PolicyID string
	
	// Configuration is the policy configuration
	Configuration *PolicyConfiguration
	
	// Metrics are performance metrics
	Metrics map[string]float64
}

// PolicyPattern represents a pattern in policy performance
type PolicyPattern struct {
	// ID is a unique identifier for this pattern
	ID string
	
	// Description describes the pattern
	Description string
	
	// PolicyID is the ID of the policy
	PolicyID string
	
	// Conditions are the conditions under which this pattern occurs
	Conditions map[string]interface{}
	
	// Metrics are performance metrics
	Metrics map[string]float64
	
	// Confidence is the confidence in this pattern (0.0 to 1.0)
	Confidence float64
}

// NewPolicyHistoricalAnalyzer creates a new policy historical analyzer
func NewPolicyHistoricalAnalyzer() *PolicyHistoricalAnalyzer {
	return &PolicyHistoricalAnalyzer{
		PerformanceData: make([]*PolicyPerformanceData, 0),
		Patterns:        make([]*PolicyPattern, 0),
	}
}

// AddPerformanceData adds performance data
func (a *PolicyHistoricalAnalyzer) AddPerformanceData(data *PolicyPerformanceData) {
	a.PerformanceData = append(a.PerformanceData, data)
}

// AnalyzePatterns analyzes patterns in performance data
func (a *PolicyHistoricalAnalyzer) AnalyzePatterns() []*PolicyPattern {
	// In a real implementation, this would analyze the performance data
	// to identify patterns. For now, we'll just return a placeholder.
	
	if len(a.Patterns) == 0 && len(a.PerformanceData) > 0 {
		// Create a sample pattern
		pattern := &PolicyPattern{
			ID:          "pattern-001",
			Description: "High CPU utilization pattern",
			PolicyID:    a.PerformanceData[0].PolicyID,
			Conditions: map[string]interface{}{
				"time_of_day": "evening",
				"day_of_week": "weekday",
			},
			Metrics: map[string]float64{
				"cpu_utilization":    0.85,
				"memory_utilization": 0.65,
			},
			Confidence: 0.75,
		}
		
		a.Patterns = append(a.Patterns, pattern)
	}
	
	return a.Patterns
}

// GetPolicyPerformanceHistory gets performance history for a policy
func (a *PolicyHistoricalAnalyzer) GetPolicyPerformanceHistory(policyID string) []*PolicyPerformanceData {
	history := make([]*PolicyPerformanceData, 0)
	
	for _, data := range a.PerformanceData {
		if data.PolicyID == policyID {
			history = append(history, data)
		}
	}
	
	return history
}

// PolicyMLRecommender provides ML-based policy recommendations
type PolicyMLRecommender struct {
	// TrainingData is the data used to train the ML model
	TrainingData []*PolicyTrainingData
	
	// ModelVersion is the version of the ML model
	ModelVersion string
	
	// LastTrainingTime is when the model was last trained
	LastTrainingTime time.Time
}

// PolicyTrainingData represents training data for the ML model
type PolicyTrainingData struct {
	// Features are input features
	Features map[string]float64
	
	// Labels are output labels
	Labels map[string]float64
}

// NewPolicyMLRecommender creates a new policy ML recommender
func NewPolicyMLRecommender() *PolicyMLRecommender {
	return &PolicyMLRecommender{
		TrainingData:    make([]*PolicyTrainingData, 0),
		ModelVersion:    "0.1",
		LastTrainingTime: time.Now(),
	}
}

// AddTrainingData adds training data
func (r *PolicyMLRecommender) AddTrainingData(data *PolicyTrainingData) {
	r.TrainingData = append(r.TrainingData, data)
}

// TrainModel trains the ML model
func (r *PolicyMLRecommender) TrainModel() error {
	// In a real implementation, this would train an ML model
	// For now, we'll just update the model version and training time
	
	r.ModelVersion = fmt.Sprintf("0.%d", len(r.TrainingData) % 100 + 1)
	r.LastTrainingTime = time.Now()
	
	return nil
}

// GetRecommendation gets an ML-based recommendation
func (r *PolicyMLRecommender) GetRecommendation(features map[string]float64) (map[string]float64, float64) {
	// In a real implementation, this would use the ML model to generate a recommendation
	// For now, we'll just return a placeholder
	
	recommendation := make(map[string]float64)
	confidence := 0.0
	
	if len(r.TrainingData) > 0 {
		// Use a simple heuristic based on the training data
		for _, data := range r.TrainingData {
			similarity := calculateFeatureSimilarity(features, data.Features)
			if similarity > confidence {
				confidence = similarity
				recommendation = data.Labels
			}
		}
	}
	
	return recommendation, confidence
}

// calculateFeatureSimilarity calculates the similarity between feature sets
func calculateFeatureSimilarity(features1, features2 map[string]float64) float64 {
	// In a real implementation, this would use a proper similarity metric
	// For now, we'll use a simple Euclidean distance-based similarity
	
	squaredSum := 0.0
	count := 0
	
	for key, value1 := range features1 {
		if value2, exists := features2[key]; exists {
			squaredSum += math.Pow(value1 - value2, 2)
			count++
		}
	}
	
	if count == 0 {
		return 0.0
	}
	
	distance := math.Sqrt(squaredSum / float64(count))
	similarity := 1.0 / (1.0 + distance)
	
	return similarity
}

// RecommendationQualityTracker tracks the quality of recommendations
type RecommendationQualityTracker struct {
	// QualityData stores quality data for recommendations
	QualityData map[string]*RecommendationQualityData
}

// RecommendationQualityData represents quality data for a recommendation
type RecommendationQualityData struct {
	// RecommendationID is the ID of the recommendation
	RecommendationID string
	
	// AppliedAt is when the recommendation was applied
	AppliedAt time.Time
	
	// BaselineMetrics are metrics before applying the recommendation
	BaselineMetrics map[string]float64
	
	// ResultMetrics are metrics after applying the recommendation
	ResultMetrics map[string]float64
	
	// ImprovementPercentage is the percentage improvement
	ImprovementPercentage float64
	
	// UserRating is the user's rating of the recommendation (0.0 to 5.0)
	UserRating float64
}

// NewRecommendationQualityTracker creates a new recommendation quality tracker
func NewRecommendationQualityTracker() *RecommendationQualityTracker {
	return &RecommendationQualityTracker{
		QualityData: make(map[string]*RecommendationQualityData),
	}
}

// TrackRecommendationQuality tracks the quality of a recommendation
func (t *RecommendationQualityTracker) TrackRecommendationQuality(
	recommendationID string, baselineMetrics, resultMetrics map[string]float64) *RecommendationQualityData {
	
	// Calculate improvement percentage
	improvementPercentage := 0.0
	if len(baselineMetrics) > 0 && len(resultMetrics) > 0 {
		totalImprovement := 0.0
		metricCount := 0
		
		for key, baselineValue := range baselineMetrics {
			if resultValue, exists := resultMetrics[key]; exists {
				improvement := (resultValue - baselineValue) / baselineValue * 100.0
				totalImprovement += improvement
				metricCount++
			}
		}
		
		if metricCount > 0 {
			improvementPercentage = totalImprovement / float64(metricCount)
		}
	}
	
	// Create quality data
	qualityData := &RecommendationQualityData{
		RecommendationID:      recommendationID,
		AppliedAt:             time.Now(),
		BaselineMetrics:       baselineMetrics,
		ResultMetrics:         resultMetrics,
		ImprovementPercentage: improvementPercentage,
	}
	
	t.QualityData[recommendationID] = qualityData
	
	return qualityData
}

// SetUserRating sets the user's rating for a recommendation
func (t *RecommendationQualityTracker) SetUserRating(recommendationID string, rating float64) bool {
	qualityData, exists := t.QualityData[recommendationID]
	if !exists {
		return false
	}
	
	qualityData.UserRating = rating
	return true
}

// GetRecommendationQuality gets quality data for a recommendation
func (t *RecommendationQualityTracker) GetRecommendationQuality(recommendationID string) *RecommendationQualityData {
	return t.QualityData[recommendationID]
}

// GetTopRecommendations gets the top recommendations by improvement percentage
func (t *RecommendationQualityTracker) GetTopRecommendations(limit int) []*RecommendationQualityData {
	// Convert map to slice
	qualityDataSlice := make([]*RecommendationQualityData, 0, len(t.QualityData))
	for _, data := range t.QualityData {
		qualityDataSlice = append(qualityDataSlice, data)
	}
	
	// Sort by improvement percentage
	sort.Slice(qualityDataSlice, func(i, j int) bool {
		return qualityDataSlice[i].ImprovementPercentage > qualityDataSlice[j].ImprovementPercentage
	})
	
	// Limit results
	if limit > 0 && limit < len(qualityDataSlice) {
		qualityDataSlice = qualityDataSlice[:limit]
	}
	
	return qualityDataSlice
}

// GenerateEnhancedRecommendations generates enhanced policy recommendations
func (e *EnhancedPolicyRecommendationEngine) GenerateEnhancedRecommendations(ctx context.Context) ([]*PolicyRecommendation, error) {
	// Get basic recommendations
	basicRecommendations, err := e.PolicyRecommendationEngine.GenerateRecommendations(ctx)
	if err != nil {
		return nil, err
	}
	
	// Analyze historical patterns
	patterns := e.HistoricalAnalyzer.AnalyzePatterns()
	
	// Generate ML-based recommendations
	mlRecommendations := e.generateMLRecommendations()
	
	// Combine recommendations
	combinedRecommendations := e.combineRecommendations(basicRecommendations, patterns, mlRecommendations)
	
	// Track recommendations
	for _, rec := range combinedRecommendations {
		e.QualityTracker.TrackRecommendationQuality(
			rec.ID,
			map[string]float64{"baseline_score": 50.0},
			map[string]float64{"expected_score": rec.ExpectedImprovementScore},
		)
	}
	
	return combinedRecommendations, nil
}

// generateMLRecommendations generates ML-based recommendations
func (e *EnhancedPolicyRecommendationEngine) generateMLRecommendations() []*PolicyRecommendation {
	// In a real implementation, this would use the ML model to generate recommendations
	// For now, we'll just return a placeholder
	
	recommendations := make([]*PolicyRecommendation, 0)
	
	// Create a sample ML-based recommendation
	recommendation := &PolicyRecommendation{
		ID:                     fmt.Sprintf("ml-rec-%d", time.Now().UnixNano()),
		Name:                   "ML-Based Resource Optimization",
		Description:            "Recommendation based on machine learning analysis of historical performance",
		RecommendationType:     "ml_optimization",
		ExpectedImprovementScore: 85.0,
		Confidence:             0.82,
		RecommendedPolicies:    make(map[string]*PolicyConfiguration),
		CreatedAt:              time.Now(),
	}
	
	// Add a sample policy configuration
	recommendation.RecommendedPolicies["resource-optimization"] = &PolicyConfiguration{
		PolicyID: "resource-optimization",
		Priority: 80,
		Enabled:  true,
		ParameterValues: map[string]interface{}{
			"cpu_weight":    2.5,
			"memory_weight": 1.8,
			"io_weight":     1.2,
		},
	}
	
	recommendations = append(recommendations, recommendation)
	
	return recommendations
}

// combineRecommendations combines recommendations from different sources
func (e *EnhancedPolicyRecommendationEngine) combineRecommendations(
	basicRecs []*PolicyRecommendation, patterns []*PolicyPattern, mlRecs []*PolicyRecommendation) []*PolicyRecommendation {
	
	// Start with basic recommendations
	combinedRecs := make([]*PolicyRecommendation, len(basicRecs))
	copy(combinedRecs, basicRecs)
	
	// Add ML-based recommendations
	combinedRecs = append(combinedRecs, mlRecs...)
	
	// Enhance recommendations with pattern insights
	for _, pattern := range patterns {
		// Find related recommendations
		for _, rec := range combinedRecs {
			for policyID := range rec.RecommendedPolicies {
				if policyID == pattern.PolicyID {
					// Enhance the recommendation with pattern insights
					rec.Description += fmt.Sprintf(" (Pattern: %s, Confidence: %.2f)", 
						pattern.Description, pattern.Confidence)
					
					// Adjust confidence based on pattern confidence
					rec.Confidence = (rec.Confidence + pattern.Confidence) / 2.0
					
					// Adjust expected improvement score
					if score, exists := pattern.Metrics["improvement_score"]; exists {
						rec.ExpectedImprovementScore = (rec.ExpectedImprovementScore + score) / 2.0
					}
				}
			}
		}
	}
	
	// Sort by confidence and expected improvement
	sort.Slice(combinedRecs, func(i, j int) bool {
		// Primary sort by confidence
		if combinedRecs[i].Confidence != combinedRecs[j].Confidence {
			return combinedRecs[i].Confidence > combinedRecs[j].Confidence
		}
		
		// Secondary sort by expected improvement
		return combinedRecs[i].ExpectedImprovementScore > combinedRecs[j].ExpectedImprovementScore
	})
	
	return combinedRecs
}

// TrackRecommendationApplication tracks the application of a recommendation
func (e *EnhancedPolicyRecommendationEngine) TrackRecommendationApplication(
	ctx context.Context, recommendationID string, baselineMetrics map[string]float64) error {
	
	// Find the recommendation
	var recommendation *PolicyRecommendation
	for _, rec := range e.RecommendationHistory {
		if rec.ID == recommendationID {
			recommendation = rec
			break
		}
	}
	
	if recommendation == nil {
		return fmt.Errorf("recommendation with ID %s not found", recommendationID)
	}
	
	// Mark as applied
	recommendation.Applied = true
	recommendation.AppliedAt = time.Now()
	
	// Store baseline metrics
	log.Printf("Tracking application of recommendation %s with baseline metrics: %v", 
		recommendationID, baselineMetrics)
	
	return nil
}

// EvaluateRecommendationImpact evaluates the impact of an applied recommendation
func (e *EnhancedPolicyRecommendationEngine) EvaluateRecommendationImpact(
	ctx context.Context, recommendationID string, resultMetrics map[string]float64) (*RecommendationImpact, error) {
	
	// Find the recommendation
	var recommendation *PolicyRecommendation
	for _, rec := range e.RecommendationHistory {
		if rec.ID == recommendationID {
			recommendation = rec
			break
		}
	}
	
	if recommendation == nil {
		return nil, fmt.Errorf("recommendation with ID %s not found", recommendationID)
	}
	
	if !recommendation.Applied {
		return nil, fmt.Errorf("recommendation with ID %s has not been applied", recommendationID)
	}
	
	// Get quality data
	qualityData := e.QualityTracker.GetRecommendationQuality(recommendationID)
	if qualityData == nil {
		return nil, fmt.Errorf("quality data for recommendation with ID %s not found", recommendationID)
	}
	
	// Update result metrics
	qualityData.ResultMetrics = resultMetrics
	
	// Calculate impact
	impact := &RecommendationImpact{
		RecommendationID:      recommendationID,
		ExpectedImprovement:   recommendation.ExpectedImprovementScore,
		ActualImprovement:     qualityData.ImprovementPercentage,
		Metrics:               resultMetrics,
		EvaluationTime:        time.Now(),
		DurationSinceApplied:  time.Since(recommendation.AppliedAt),
	}
	
	return impact, nil
}

// EnhancedRecommendationImpact represents the impact of an applied recommendation
type EnhancedRecommendationImpact struct {
	// RecommendationID is the ID of the recommendation
	RecommendationID string
	
	// ExpectedImprovement is the expected improvement
	ExpectedImprovement float64
	
	// ActualImprovement is the actual improvement
	ActualImprovement float64
	
	// Metrics are the result metrics
	Metrics map[string]float64
	
	// EvaluationTime is when the impact was evaluated
	EvaluationTime time.Time
	
	// DurationSinceApplied is the duration since the recommendation was applied
	DurationSinceApplied time.Duration
}
