// Package advisor provides proactive recommendations
package advisor

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/yourusername/novacron/backend/core/cognitive"
)

// ProactiveAdvisor generates proactive recommendations
type ProactiveAdvisor struct {
	config          *cognitive.CognitiveConfig
	analyzers       map[string]Analyzer
	recommendations chan *cognitive.Recommendation
	lock            sync.RWMutex
	metrics         AdvisorMetrics
}

// Analyzer interface for different types of analysis
type Analyzer interface {
	Analyze(ctx context.Context, systemState interface{}) ([]*cognitive.Recommendation, error)
	GetType() string
}

// AdvisorMetrics tracks advisor performance
type AdvisorMetrics struct {
	TotalRecommendations int64
	AcceptedRecommendations int64
	RejectedRecommendations int64
	AvgConfidence        float64
}

// NewProactiveAdvisor creates a new proactive advisor
func NewProactiveAdvisor(config *cognitive.CognitiveConfig) *ProactiveAdvisor {
	pa := &ProactiveAdvisor{
		config:          config,
		analyzers:       make(map[string]Analyzer),
		recommendations: make(chan *cognitive.Recommendation, 100),
	}

	// Register default analyzers
	pa.RegisterAnalyzer(NewCostAnalyzer())
	pa.RegisterAnalyzer(NewSecurityAnalyzer())
	pa.RegisterAnalyzer(NewPerformanceAnalyzer())
	pa.RegisterAnalyzer(NewCapacityAnalyzer())

	return pa
}

// RegisterAnalyzer registers a new analyzer
func (pa *ProactiveAdvisor) RegisterAnalyzer(analyzer Analyzer) {
	pa.lock.Lock()
	defer pa.lock.Unlock()
	pa.analyzers[analyzer.GetType()] = analyzer
}

// AnalyzeAndRecommend performs analysis and generates recommendations
func (pa *ProactiveAdvisor) AnalyzeAndRecommend(ctx context.Context, systemState interface{}) ([]*cognitive.Recommendation, error) {
	var allRecommendations []*cognitive.Recommendation

	// Run all analyzers
	for _, analyzer := range pa.analyzers {
		recs, err := analyzer.Analyze(ctx, systemState)
		if err != nil {
			continue // Log error but continue with other analyzers
		}

		// Filter by confidence threshold
		for _, rec := range recs {
			if rec.Confidence >= pa.config.MinConfidenceScore {
				allRecommendations = append(allRecommendations, rec)
			}
		}
	}

	// Sort by impact and confidence
	allRecommendations = pa.prioritizeRecommendations(allRecommendations)

	// Update metrics
	pa.lock.Lock()
	pa.metrics.TotalRecommendations += int64(len(allRecommendations))
	pa.lock.Unlock()

	return allRecommendations, nil
}

// prioritizeRecommendations sorts recommendations by priority
func (pa *ProactiveAdvisor) prioritizeRecommendations(recs []*cognitive.Recommendation) []*cognitive.Recommendation {
	// Sort by: High impact + High confidence first
	// Simplified - in production use more sophisticated scoring
	return recs
}

// GetRecommendations returns pending recommendations
func (pa *ProactiveAdvisor) GetRecommendations() []*cognitive.Recommendation {
	var recs []*cognitive.Recommendation
	for {
		select {
		case rec := <-pa.recommendations:
			recs = append(recs, rec)
		default:
			return recs
		}
	}
}

// RecordAcceptance records recommendation acceptance
func (pa *ProactiveAdvisor) RecordAcceptance(recID string, accepted bool) {
	pa.lock.Lock()
	defer pa.lock.Unlock()

	if accepted {
		pa.metrics.AcceptedRecommendations++
	} else {
		pa.metrics.RejectedRecommendations++
	}
}

// GetMetrics returns advisor metrics
func (pa *ProactiveAdvisor) GetMetrics() AdvisorMetrics {
	pa.lock.RLock()
	defer pa.lock.RUnlock()
	return pa.metrics
}

// CostAnalyzer analyzes cost optimization opportunities
type CostAnalyzer struct{}

// NewCostAnalyzer creates a cost analyzer
func NewCostAnalyzer() *CostAnalyzer {
	return &CostAnalyzer{}
}

// Analyze implements Analyzer interface
func (ca *CostAnalyzer) Analyze(ctx context.Context, systemState interface{}) ([]*cognitive.Recommendation, error) {
	var recommendations []*cognitive.Recommendation

	// Example: Recommend spot instances
	rec := &cognitive.Recommendation{
		Type:        "cost",
		Title:       "Migrate to Spot Instances",
		Description: "40% of your workload is suitable for spot instances, which could reduce costs by 26%",
		Impact:      "High",
		Effort:      "Medium",
		Savings:     3200.00,
		Confidence:  0.88,
		Actions:     []string{"Identify stateless workloads", "Configure spot instance policies", "Enable automatic failover"},
		Metadata:    map[string]interface{}{"potential_savings": "$3,200/month"},
		CreatedAt:   time.Now(),
	}
	recommendations = append(recommendations, rec)

	// Example: Recommend reserved instances
	rec2 := &cognitive.Recommendation{
		Type:        "cost",
		Title:       "Purchase Reserved Instances",
		Description: "Your stable workloads would save 35% with 1-year reserved instances",
		Impact:      "High",
		Effort:      "Low",
		Savings:     4500.00,
		Confidence:  0.92,
		Actions:     []string{"Analyze workload stability", "Purchase reserved instances", "Monitor utilization"},
		Metadata:    map[string]interface{}{"commitment_period": "1 year"},
		CreatedAt:   time.Now(),
	}
	recommendations = append(recommendations, rec2)

	return recommendations, nil
}

// GetType returns analyzer type
func (ca *CostAnalyzer) GetType() string {
	return "cost"
}

// SecurityAnalyzer analyzes security improvements
type SecurityAnalyzer struct{}

// NewSecurityAnalyzer creates a security analyzer
func NewSecurityAnalyzer() *SecurityAnalyzer {
	return &SecurityAnalyzer{}
}

// Analyze implements Analyzer interface
func (sa *SecurityAnalyzer) Analyze(ctx context.Context, systemState interface{}) ([]*cognitive.Recommendation, error) {
	var recommendations []*cognitive.Recommendation

	// Example: Encryption recommendation
	rec := &cognitive.Recommendation{
		Type:        "security",
		Title:       "Enable Encryption at Rest",
		Description: "3 storage volumes are not encrypted. Enable encryption to meet compliance requirements",
		Impact:      "High",
		Effort:      "Low",
		Savings:     0,
		Confidence:  0.98,
		Actions:     []string{"Enable volume encryption", "Rotate encryption keys", "Update backup policies"},
		Metadata:    map[string]interface{}{"affected_volumes": 3, "compliance": "GDPR"},
		CreatedAt:   time.Now(),
	}
	recommendations = append(recommendations, rec)

	return recommendations, nil
}

// GetType returns analyzer type
func (sa *SecurityAnalyzer) GetType() string {
	return "security"
}

// PerformanceAnalyzer analyzes performance optimization
type PerformanceAnalyzer struct{}

// NewPerformanceAnalyzer creates a performance analyzer
func NewPerformanceAnalyzer() *PerformanceAnalyzer {
	return &PerformanceAnalyzer{}
}

// Analyze implements Analyzer interface
func (pa *PerformanceAnalyzer) Analyze(ctx context.Context, systemState interface{}) ([]*cognitive.Recommendation, error) {
	var recommendations []*cognitive.Recommendation

	// Example: Database optimization
	rec := &cognitive.Recommendation{
		Type:        "performance",
		Title:       "Optimize Database Connection Pool",
		Description: "Database connection pool is frequently exhausted, causing latency spikes",
		Impact:      "High",
		Effort:      "Low",
		Savings:     0,
		Confidence:  0.92,
		Actions:     []string{"Increase pool size from 10 to 50", "Enable connection recycling", "Monitor pool metrics"},
		Metadata:    map[string]interface{}{"current_pool_size": 10, "recommended_pool_size": 50},
		CreatedAt:   time.Now(),
	}
	recommendations = append(recommendations, rec)

	return recommendations, nil
}

// GetType returns analyzer type
func (pa *PerformanceAnalyzer) GetType() string {
	return "performance"
}

// CapacityAnalyzer analyzes capacity planning
type CapacityAnalyzer struct{}

// NewCapacityAnalyzer creates a capacity analyzer
func NewCapacityAnalyzer() *CapacityAnalyzer {
	return &CapacityAnalyzer{}
}

// Analyze implements Analyzer interface
func (ca *CapacityAnalyzer) Analyze(ctx context.Context, systemState interface{}) ([]*cognitive.Recommendation, error) {
	var recommendations []*cognitive.Recommendation

	// Example: Capacity expansion
	rec := &cognitive.Recommendation{
		Type:        "capacity",
		Title:       "Plan Capacity Expansion",
		Description: "Current growth trends suggest you'll need 30% more capacity in 3 months",
		Impact:      "Medium",
		Effort:      "Medium",
		Savings:     0,
		Confidence:  0.85,
		Actions:     []string{"Review growth projections", "Plan infrastructure expansion", "Budget for additional resources"},
		Metadata:    map[string]interface{}{"timeframe": "3 months", "growth_rate": "30%"},
		CreatedAt:   time.Now(),
	}
	recommendations = append(recommendations, rec)

	return recommendations, nil
}

// GetType returns analyzer type
func (ca *CapacityAnalyzer) GetType() string {
	return "capacity"
}
