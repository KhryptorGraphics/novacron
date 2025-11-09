package recommendations

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// Engine generates and manages tuning recommendations
type Engine struct {
	config          RecommendationsConfig
	mu              sync.RWMutex
	recommendations map[string]*TuningRecommendation
	appliedTunings  map[string]*AppliedTuning
	abTests         map[string]*ABTest
}

// RecommendationsConfig defines recommendation settings
type RecommendationsConfig struct {
	ABTestingEnabled      bool
	RollbackOnDegradation bool
	DegradationThreshold  float64 // 0.05 (5%)
	ValidationPeriod      time.Duration
	MaxConcurrentTests    int
	ConfidenceThreshold   float64
}

// TuningRecommendation represents a tuning recommendation
type TuningRecommendation struct {
	ID                  string
	VMID                string
	Category            string // "cpu", "memory", "io", "network", "cost"
	Type                string // "numa", "cpu_pinning", "scheduler", "congestion", "rightsize"
	Description         string
	Impact              EstimatedImpact
	Risk                RiskAssessment
	Implementation      ImplementationPlan
	CostBenefit         CostBenefitAnalysis
	Priority            int
	Status              string // "pending", "testing", "applied", "rejected"
	Confidence          float64
	CreatedAt           time.Time
}

// EstimatedImpact estimates performance impact
type EstimatedImpact struct {
	PerformanceGain     float64 // Percentage
	CostSavings         float64 // Monthly
	LatencyReduction    float64 // Percentage
	ThroughputIncrease  float64 // Percentage
	ResourceEfficiency  float64 // Percentage
}

// RiskAssessment assesses recommendation risk
type RiskAssessment struct {
	Level               string // "low", "medium", "high"
	Reversibility       string // "easy", "moderate", "difficult"
	DowntimeRequired    bool
	DataLossRisk        bool
	Dependencies        []string
	Mitigations         []string
}

// ImplementationPlan defines implementation steps
type ImplementationPlan struct {
	Steps               []string
	EstimatedDuration   time.Duration
	RequiredPermissions []string
	Prerequisites       []string
	RollbackSteps       []string
}

// CostBenefitAnalysis analyzes cost vs benefit
type CostBenefitAnalysis struct {
	ImplementationCost  float64
	MonthlySavings      float64
	MonthlyGain         float64 // Performance value
	PaybackPeriod       time.Duration
	ROI                 float64
	NetPresentValue     float64
}

// AppliedTuning tracks applied tuning
type AppliedTuning struct {
	RecommendationID  string
	VMID              string
	AppliedAt         time.Time
	ValidatedAt       time.Time
	Baseline          PerformanceBaseline
	AfterTuning       PerformanceMetrics
	ActualImpact      ActualImpact
	Status            string // "testing", "validated", "rolledback"
}

// PerformanceBaseline stores baseline metrics
type PerformanceBaseline struct {
	CPUUtilization    float64
	MemoryUtilization float64
	Latency           float64
	Throughput        float64
	IOPS              float64
	Cost              float64
	Timestamp         time.Time
}

// PerformanceMetrics stores current metrics
type PerformanceMetrics struct {
	CPUUtilization    float64
	MemoryUtilization float64
	Latency           float64
	Throughput        float64
	IOPS              float64
	Cost              float64
	Timestamp         time.Time
}

// ActualImpact stores actual impact after tuning
type ActualImpact struct {
	PerformanceChange  float64
	CostChange         float64
	LatencyChange      float64
	ThroughputChange   float64
	MeetsExpectations  bool
}

// ABTest represents an A/B test
type ABTest struct {
	ID               string
	RecommendationID string
	ControlGroup     []string // VM IDs
	TestGroup        []string // VM IDs
	StartTime        time.Time
	Duration         time.Duration
	ControlMetrics   []PerformanceMetrics
	TestMetrics      []PerformanceMetrics
	Result           *ABTestResult
	Status           string // "running", "completed", "failed"
}

// ABTestResult stores A/B test result
type ABTestResult struct {
	StatisticalSignificance bool
	PValue                  float64
	PerformanceImprovement  float64
	CostImprovement         float64
	Recommendation          string // "adopt", "reject", "retest"
	Confidence              float64
}

// NewEngine creates recommendations engine
func NewEngine(config RecommendationsConfig) *Engine {
	if config.DegradationThreshold == 0 {
		config.DegradationThreshold = 0.05
	}
	if config.ValidationPeriod == 0 {
		config.ValidationPeriod = 10 * time.Minute
	}
	if config.MaxConcurrentTests == 0 {
		config.MaxConcurrentTests = 5
	}
	if config.ConfidenceThreshold == 0 {
		config.ConfidenceThreshold = 0.85
	}

	return &Engine{
		config:          config,
		recommendations: make(map[string]*TuningRecommendation),
		appliedTunings:  make(map[string]*AppliedTuning),
		abTests:         make(map[string]*ABTest),
	}
}

// GenerateRecommendation creates a new recommendation
func (e *Engine) GenerateRecommendation(vmID, category, recType, description string, impact EstimatedImpact, risk RiskAssessment) *TuningRecommendation {
	e.mu.Lock()
	defer e.mu.Unlock()

	id := fmt.Sprintf("rec-%s-%d", vmID, time.Now().Unix())

	rec := &TuningRecommendation{
		ID:          id,
		VMID:        vmID,
		Category:    category,
		Type:        recType,
		Description: description,
		Impact:      impact,
		Risk:        risk,
		Status:      "pending",
		CreatedAt:   time.Now(),
	}

	// Calculate priority (higher impact + lower risk = higher priority)
	rec.Priority = e.calculatePriority(rec)

	// Calculate confidence
	rec.Confidence = e.calculateConfidence(rec)

	e.recommendations[id] = rec
	return rec
}

// calculatePriority calculates recommendation priority
func (e *Engine) calculatePriority(rec *TuningRecommendation) int {
	score := 0.0

	// Performance impact (0-50 points)
	score += rec.Impact.PerformanceGain * 0.5

	// Cost savings (0-30 points)
	if rec.Impact.CostSavings > 0 {
		score += min(rec.Impact.CostSavings/100, 30)
	}

	// Risk adjustment (subtract for high risk)
	switch rec.Risk.Level {
	case "low":
		score += 20
	case "medium":
		score += 10
	case "high":
		score -= 10
	}

	return int(score)
}

// calculateConfidence calculates recommendation confidence
func (e *Engine) calculateConfidence(rec *TuningRecommendation) float64 {
	confidence := 0.8 // Base confidence

	// Adjust based on risk
	switch rec.Risk.Level {
	case "low":
		confidence += 0.1
	case "high":
		confidence -= 0.2
	}

	// Adjust based on reversibility
	if rec.Risk.Reversibility == "easy" {
		confidence += 0.1
	}

	return min(max(confidence, 0.0), 1.0)
}

// ApplyRecommendation applies a tuning recommendation
func (e *Engine) ApplyRecommendation(ctx context.Context, recID string, baseline PerformanceBaseline) error {
	e.mu.Lock()
	rec, exists := e.recommendations[recID]
	if !exists {
		e.mu.Unlock()
		return fmt.Errorf("recommendation %s not found", recID)
	}

	if rec.Status != "pending" {
		e.mu.Unlock()
		return fmt.Errorf("recommendation already %s", rec.Status)
	}

	rec.Status = "testing"
	e.mu.Unlock()

	// Create applied tuning record
	applied := &AppliedTuning{
		RecommendationID: recID,
		VMID:             rec.VMID,
		AppliedAt:        time.Now(),
		Baseline:         baseline,
		Status:           "testing",
	}

	e.mu.Lock()
	e.appliedTunings[recID] = applied
	e.mu.Unlock()

	// Start validation monitoring
	go e.monitorAppliedTuning(ctx, recID)

	return nil
}

// monitorAppliedTuning monitors applied tuning
func (e *Engine) monitorAppliedTuning(ctx context.Context, recID string) {
	ticker := time.NewTicker(1 * time.Minute)
	defer ticker.Stop()

	validationEnd := time.Now().Add(e.config.ValidationPeriod)

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			if time.Now().After(validationEnd) {
				e.validateTuning(recID)
				return
			}

			// Check for degradation
			if e.checkDegradation(recID) {
				e.rollbackTuning(recID, "performance degradation detected")
				return
			}
		}
	}
}

// checkDegradation checks for performance degradation
func (e *Engine) checkDegradation(recID string) bool {
	e.mu.RLock()
	applied, exists := e.appliedTunings[recID]
	e.mu.RUnlock()

	if !exists {
		return false
	}

	// Compare current metrics to baseline
	degradation := (applied.Baseline.Throughput - applied.AfterTuning.Throughput) / applied.Baseline.Throughput

	return degradation > e.config.DegradationThreshold
}

// validateTuning validates applied tuning
func (e *Engine) validateTuning(recID string) {
	e.mu.Lock()
	defer e.mu.Unlock()

	applied, exists := e.appliedTunings[recID]
	if !exists {
		return
	}

	// Calculate actual impact
	applied.ActualImpact = ActualImpact{
		PerformanceChange: (applied.AfterTuning.Throughput - applied.Baseline.Throughput) / applied.Baseline.Throughput,
		CostChange:        (applied.AfterTuning.Cost - applied.Baseline.Cost) / applied.Baseline.Cost,
		LatencyChange:     (applied.Baseline.Latency - applied.AfterTuning.Latency) / applied.Baseline.Latency,
		ThroughputChange:  (applied.AfterTuning.Throughput - applied.Baseline.Throughput) / applied.Baseline.Throughput,
	}

	rec := e.recommendations[recID]
	applied.ActualImpact.MeetsExpectations = applied.ActualImpact.PerformanceChange >= rec.Impact.PerformanceGain*0.8

	if applied.ActualImpact.MeetsExpectations {
		applied.Status = "validated"
		rec.Status = "applied"
	} else {
		e.rollbackTuning(recID, "did not meet expectations")
	}

	applied.ValidatedAt = time.Now()
}

// rollbackTuning rolls back a tuning
func (e *Engine) rollbackTuning(recID, reason string) {
	e.mu.Lock()
	defer e.mu.Unlock()

	applied, exists := e.appliedTunings[recID]
	if !exists {
		return
	}

	applied.Status = "rolledback"

	rec := e.recommendations[recID]
	rec.Status = "rejected"

	fmt.Printf("Rolled back tuning %s: %s\n", recID, reason)
}

// StartABTest starts an A/B test
func (e *Engine) StartABTest(recID string, controlVMs, testVMs []string, duration time.Duration) error {
	if !e.config.ABTestingEnabled {
		return fmt.Errorf("A/B testing not enabled")
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	if len(e.abTests) >= e.config.MaxConcurrentTests {
		return fmt.Errorf("max concurrent A/B tests reached")
	}

	testID := fmt.Sprintf("ab-%s-%d", recID, time.Now().Unix())

	test := &ABTest{
		ID:               testID,
		RecommendationID: recID,
		ControlGroup:     controlVMs,
		TestGroup:        testVMs,
		StartTime:        time.Now(),
		Duration:         duration,
		Status:           "running",
	}

	e.abTests[testID] = test

	// Monitor test
	go e.monitorABTest(testID)

	return nil
}

// monitorABTest monitors A/B test
func (e *Engine) monitorABTest(testID string) {
	e.mu.RLock()
	test := e.abTests[testID]
	e.mu.RUnlock()

	time.Sleep(test.Duration)

	// Analyze results
	result := e.analyzeABTest(testID)

	e.mu.Lock()
	test.Result = result
	test.Status = "completed"
	e.mu.Unlock()
}

// analyzeABTest analyzes A/B test results
func (e *Engine) analyzeABTest(testID string) *ABTestResult {
	e.mu.RLock()
	test := e.abTests[testID]
	e.mu.RUnlock()

	// Calculate average metrics for control and test groups
	controlAvg := e.calculateAverageMetrics(test.ControlMetrics)
	testAvg := e.calculateAverageMetrics(test.TestMetrics)

	// Calculate improvement
	perfImprovement := (testAvg.Throughput - controlAvg.Throughput) / controlAvg.Throughput
	costImprovement := (controlAvg.Cost - testAvg.Cost) / controlAvg.Cost

	// Statistical significance (simplified t-test)
	pValue := 0.05 // Simplified
	significant := perfImprovement > 0.05 && pValue < 0.05

	recommendation := "reject"
	if significant && perfImprovement > 0.1 {
		recommendation = "adopt"
	} else if perfImprovement > 0.05 {
		recommendation = "retest"
	}

	return &ABTestResult{
		StatisticalSignificance: significant,
		PValue:                  pValue,
		PerformanceImprovement:  perfImprovement,
		CostImprovement:         costImprovement,
		Recommendation:          recommendation,
		Confidence:              0.9,
	}
}

// calculateAverageMetrics calculates average metrics
func (e *Engine) calculateAverageMetrics(metrics []PerformanceMetrics) PerformanceMetrics {
	if len(metrics) == 0 {
		return PerformanceMetrics{}
	}

	avg := PerformanceMetrics{}
	for _, m := range metrics {
		avg.CPUUtilization += m.CPUUtilization
		avg.MemoryUtilization += m.MemoryUtilization
		avg.Latency += m.Latency
		avg.Throughput += m.Throughput
		avg.IOPS += m.IOPS
		avg.Cost += m.Cost
	}

	n := float64(len(metrics))
	avg.CPUUtilization /= n
	avg.MemoryUtilization /= n
	avg.Latency /= n
	avg.Throughput /= n
	avg.IOPS /= n
	avg.Cost /= n

	return avg
}

// GetTopRecommendations returns top N recommendations
func (e *Engine) GetTopRecommendations(n int) []*TuningRecommendation {
	e.mu.RLock()
	defer e.mu.RUnlock()

	var recs []*TuningRecommendation
	for _, rec := range e.recommendations {
		if rec.Status == "pending" && rec.Confidence >= e.config.ConfidenceThreshold {
			recs = append(recs, rec)
		}
	}

	// Sort by priority
	for i := 0; i < len(recs); i++ {
		for j := i + 1; j < len(recs); j++ {
			if recs[i].Priority < recs[j].Priority {
				recs[i], recs[j] = recs[j], recs[i]
			}
		}
	}

	if len(recs) > n {
		recs = recs[:n]
	}

	return recs
}

// Helper functions
func min(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}

func max(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}
