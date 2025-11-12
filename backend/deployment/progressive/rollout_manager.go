// Package progressive provides multi-region progressive rollout automation
// with geographic strategy, customer segment targeting, risk scoring, and
// automated decision making based on real-time metrics.
package progressive

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sync"
	"time"
)

// RolloutManager orchestrates progressive multi-region deployments
type RolloutManager struct {
	mu                 sync.RWMutex
	rollouts           map[string]*ProgressiveRollout
	regionManager      RegionManager
	segmentTargeting   SegmentTargeting
	riskAssessor       RiskAssessor
	decisionEngine     DecisionEngine
	metricsAggregator  MetricsAggregator
	notificationService NotificationService
	ctx                context.Context
	cancel             context.CancelFunc
	wg                 sync.WaitGroup
}

// ProgressiveRollout represents a progressive multi-region rollout
type ProgressiveRollout struct {
	ID                 string                `json:"id"`
	Name               string                `json:"name"`
	Version            string                `json:"version"`
	Strategy           *RolloutStrategy      `json:"strategy"`
	Status             RolloutStatus         `json:"status"`
	CurrentPhase       int                   `json:"current_phase"`
	Phases             []RolloutPhase        `json:"phases"`
	Regions            []RegionDeployment    `json:"regions"`
	Segments           []SegmentDeployment   `json:"segments"`
	RiskScore          float64               `json:"risk_score"`
	RiskLevel          RiskLevel             `json:"risk_level"`
	Metrics            *AggregatedMetrics    `json:"metrics"`
	Decisions          []AutomatedDecision   `json:"decisions"`
	StartTime          time.Time             `json:"start_time"`
	EndTime            *time.Time            `json:"end_time,omitempty"`
	PauseReason        string                `json:"pause_reason,omitempty"`
	AutomatedActions   bool                  `json:"automated_actions"`
}

// RolloutStrategy defines the rollout strategy
type RolloutStrategy struct {
	Type                StrategyType      `json:"type"`
	GeographicOrder     []string          `json:"geographic_order"`
	SegmentPriority     []string          `json:"segment_priority"`
	PhaseDuration       time.Duration     `json:"phase_duration"`
	ValidationThreshold *ValidationThreshold `json:"validation_threshold"`
	RollbackPolicy      *RollbackPolicy   `json:"rollback_policy"`
	AutomatedDecisions  bool              `json:"automated_decisions"`
}

// StrategyType defines rollout strategy types
type StrategyType string

const (
	StrategyGeographic      StrategyType = "geographic"
	StrategySegmented       StrategyType = "segmented"
	StrategyRiskBased       StrategyType = "risk_based"
	StrategyAdaptive        StrategyType = "adaptive"
)

// ValidationThreshold defines validation thresholds
type ValidationThreshold struct {
	MinSuccessRate     float64       `json:"min_success_rate"`
	MaxErrorRate       float64       `json:"max_error_rate"`
	MaxLatencyP99      time.Duration `json:"max_latency_p99"`
	MinHealthScore     float64       `json:"min_health_score"`
	SampleSize         int           `json:"sample_size"`
}

// RollbackPolicy defines when and how to rollback
type RollbackPolicy struct {
	AutoRollback          bool          `json:"auto_rollback"`
	RollbackOnError       bool          `json:"rollback_on_error"`
	ErrorThreshold        float64       `json:"error_threshold"`
	LatencyThreshold      time.Duration `json:"latency_threshold"`
	HealthScoreThreshold  float64       `json:"health_score_threshold"`
	MaxFailedPhases       int           `json:"max_failed_phases"`
}

// RolloutStatus represents rollout status
type RolloutStatus string

const (
	RolloutStatusInitializing RolloutStatus = "initializing"
	RolloutStatusProgressing  RolloutStatus = "progressing"
	RolloutStatusValidating   RolloutStatus = "validating"
	RolloutStatusPaused       RolloutStatus = "paused"
	RolloutStatusRollingBack  RolloutStatus = "rolling_back"
	RolloutStatusCompleted    RolloutStatus = "completed"
	RolloutStatusFailed       RolloutStatus = "failed"
)

// RolloutPhase represents a phase in the rollout
type RolloutPhase struct {
	PhaseNumber     int               `json:"phase_number"`
	Name            string            `json:"name"`
	Regions         []string          `json:"regions"`
	Segments        []string          `json:"segments"`
	TrafficPercent  float64           `json:"traffic_percent"`
	Status          PhaseStatus       `json:"status"`
	StartTime       *time.Time        `json:"start_time,omitempty"`
	EndTime         *time.Time        `json:"end_time,omitempty"`
	Duration        time.Duration     `json:"duration"`
	Metrics         *PhaseMetrics     `json:"metrics,omitempty"`
	Passed          bool              `json:"passed"`
	FailureReason   string            `json:"failure_reason,omitempty"`
}

// PhaseStatus represents phase status
type PhaseStatus string

const (
	PhaseStatusPending    PhaseStatus = "pending"
	PhaseStatusActive     PhaseStatus = "active"
	PhaseStatusValidating PhaseStatus = "validating"
	PhaseStatusCompleted  PhaseStatus = "completed"
	PhaseStatusFailed     PhaseStatus = "failed"
)

// RegionDeployment represents deployment in a region
type RegionDeployment struct {
	Region          string            `json:"region"`
	Status          DeploymentStatus  `json:"status"`
	TrafficPercent  float64           `json:"traffic_percent"`
	StartTime       time.Time         `json:"start_time"`
	EndTime         *time.Time        `json:"end_time,omitempty"`
	Metrics         *RegionMetrics    `json:"metrics"`
	HealthScore     float64           `json:"health_score"`
	Population      int64             `json:"population"`
	ActiveUsers     int64             `json:"active_users"`
}

// SegmentDeployment represents deployment to a customer segment
type SegmentDeployment struct {
	Segment         string            `json:"segment"`
	Description     string            `json:"description"`
	Status          DeploymentStatus  `json:"status"`
	TrafficPercent  float64           `json:"traffic_percent"`
	StartTime       time.Time         `json:"start_time"`
	EndTime         *time.Time        `json:"end_time,omitempty"`
	Metrics         *SegmentMetrics   `json:"metrics"`
	UserCount       int64             `json:"user_count"`
	FeedbackScore   float64           `json:"feedback_score"`
}

// DeploymentStatus represents deployment status
type DeploymentStatus string

const (
	DeploymentStatusPending   DeploymentStatus = "pending"
	DeploymentStatusDeploying DeploymentStatus = "deploying"
	DeploymentStatusActive    DeploymentStatus = "active"
	DeploymentStatusValidating DeploymentStatus = "validating"
	DeploymentStatusCompleted DeploymentStatus = "completed"
	DeploymentStatusFailed    DeploymentStatus = "failed"
)

// PhaseMetrics contains metrics for a rollout phase
type PhaseMetrics struct {
	TotalRequests     int64         `json:"total_requests"`
	SuccessfulRequests int64        `json:"successful_requests"`
	FailedRequests    int64         `json:"failed_requests"`
	SuccessRate       float64       `json:"success_rate"`
	ErrorRate         float64       `json:"error_rate"`
	LatencyP50        time.Duration `json:"latency_p50"`
	LatencyP95        time.Duration `json:"latency_p95"`
	LatencyP99        time.Duration `json:"latency_p99"`
	HealthScore       float64       `json:"health_score"`
	UserSatisfaction  float64       `json:"user_satisfaction"`
	Timestamp         time.Time     `json:"timestamp"`
}

// RegionMetrics contains region-specific metrics
type RegionMetrics struct {
	Region            string        `json:"region"`
	RequestsPerSecond float64       `json:"requests_per_second"`
	ErrorRate         float64       `json:"error_rate"`
	LatencyP99        time.Duration `json:"latency_p99"`
	Availability      float64       `json:"availability"`
	UserCount         int64         `json:"user_count"`
	Timestamp         time.Time     `json:"timestamp"`
}

// SegmentMetrics contains segment-specific metrics
type SegmentMetrics struct {
	Segment           string        `json:"segment"`
	ActiveUsers       int64         `json:"active_users"`
	ErrorRate         float64       `json:"error_rate"`
	LatencyP99        time.Duration `json:"latency_p99"`
	FeatureAdoption   float64       `json:"feature_adoption"`
	Churn             float64       `json:"churn"`
	Timestamp         time.Time     `json:"timestamp"`
}

// AggregatedMetrics contains aggregated metrics across all regions/segments
type AggregatedMetrics struct {
	TotalRequests      int64            `json:"total_requests"`
	GlobalSuccessRate  float64          `json:"global_success_rate"`
	GlobalErrorRate    float64          `json:"global_error_rate"`
	GlobalLatencyP99   time.Duration    `json:"global_latency_p99"`
	RegionMetrics      []RegionMetrics  `json:"region_metrics"`
	SegmentMetrics     []SegmentMetrics `json:"segment_metrics"`
	OverallHealthScore float64          `json:"overall_health_score"`
	Timestamp          time.Time        `json:"timestamp"`
}

// AutomatedDecision represents an automated decision
type AutomatedDecision struct {
	DecisionID      string          `json:"decision_id"`
	Type            DecisionType    `json:"type"`
	Action          DecisionAction  `json:"action"`
	Reason          string          `json:"reason"`
	Confidence      float64         `json:"confidence"`
	RiskScore       float64         `json:"risk_score"`
	BasedOnMetrics  interface{}     `json:"based_on_metrics"`
	Timestamp       time.Time       `json:"timestamp"`
	Executed        bool            `json:"executed"`
	Result          string          `json:"result,omitempty"`
}

// DecisionType represents types of automated decisions
type DecisionType string

const (
	DecisionTypeProceed   DecisionType = "proceed"
	DecisionTypePause     DecisionType = "pause"
	DecisionTypeRollback  DecisionType = "rollback"
	DecisionTypeAccelerate DecisionType = "accelerate"
	DecisionTypeDecelerate DecisionType = "decelerate"
)

// DecisionAction represents the action to take
type DecisionAction string

const (
	ActionProceedNextPhase    DecisionAction = "proceed_next_phase"
	ActionPauseRollout        DecisionAction = "pause_rollout"
	ActionInitiateRollback    DecisionAction = "initiate_rollback"
	ActionIncreaseTraffic     DecisionAction = "increase_traffic"
	ActionDecreaseTraffic     DecisionAction = "decrease_traffic"
	ActionExtendValidation    DecisionAction = "extend_validation"
	ActionSkipValidation      DecisionAction = "skip_validation"
)

// RiskLevel represents risk assessment levels
type RiskLevel string

const (
	RiskLevelLow      RiskLevel = "low"
	RiskLevelMedium   RiskLevel = "medium"
	RiskLevelHigh     RiskLevel = "high"
	RiskLevelCritical RiskLevel = "critical"
)

// RegionManager manages region deployments
type RegionManager interface {
	DeployToRegion(ctx context.Context, region string, version string, trafficPercent float64) error
	GetRegionStatus(region string) (*RegionDeployment, error)
	ListRegions() ([]string, error)
	GetRegionPriority(strategy StrategyType) ([]string, error)
}

// SegmentTargeting manages customer segment targeting
type SegmentTargeting interface {
	DeployToSegment(ctx context.Context, segment string, version string, trafficPercent float64) error
	GetSegmentStatus(segment string) (*SegmentDeployment, error)
	ListSegments() ([]string, error)
	GetSegmentPriority() ([]string, error)
}

// RiskAssessor assesses deployment risk
type RiskAssessor interface {
	AssessRisk(ctx context.Context, rolloutID string) (*RiskAssessment, error)
	CalculateRiskScore(metrics *AggregatedMetrics) float64
	GetRiskLevel(score float64) RiskLevel
}

// RiskAssessment contains risk assessment results
type RiskAssessment struct {
	RolloutID       string            `json:"rollout_id"`
	RiskScore       float64           `json:"risk_score"`
	RiskLevel       RiskLevel         `json:"risk_level"`
	Factors         []RiskFactor      `json:"factors"`
	Recommendation  string            `json:"recommendation"`
	Confidence      float64           `json:"confidence"`
	Timestamp       time.Time         `json:"timestamp"`
}

// RiskFactor represents a factor contributing to risk
type RiskFactor struct {
	Name        string  `json:"name"`
	Weight      float64 `json:"weight"`
	Score       float64 `json:"score"`
	Impact      string  `json:"impact"`
	Description string  `json:"description"`
}

// DecisionEngine makes automated decisions
type DecisionEngine interface {
	MakeDecision(ctx context.Context, rollout *ProgressiveRollout) (*AutomatedDecision, error)
	EvaluatePhase(ctx context.Context, phase *RolloutPhase, metrics *PhaseMetrics) (bool, string, error)
	ShouldRollback(ctx context.Context, rollout *ProgressiveRollout) (bool, string)
}

// MetricsAggregator aggregates metrics from multiple sources
type MetricsAggregator interface {
	AggregateMetrics(ctx context.Context, rolloutID string) (*AggregatedMetrics, error)
	GetPhaseMetrics(ctx context.Context, rolloutID string, phaseNumber int) (*PhaseMetrics, error)
	GetRegionMetrics(ctx context.Context, region string) (*RegionMetrics, error)
	GetSegmentMetrics(ctx context.Context, segment string) (*SegmentMetrics, error)
}

// NotificationService sends notifications
type NotificationService interface {
	NotifyPhaseComplete(rolloutID string, phase *RolloutPhase) error
	NotifyDecision(rolloutID string, decision *AutomatedDecision) error
	NotifyRollback(rolloutID string, reason string) error
	NotifyCompletion(rolloutID string, success bool) error
}

// NewRolloutManager creates a new rollout manager
func NewRolloutManager(
	regionManager RegionManager,
	segmentTargeting SegmentTargeting,
	riskAssessor RiskAssessor,
	decisionEngine DecisionEngine,
	metricsAggregator MetricsAggregator,
	notificationService NotificationService,
) *RolloutManager {
	ctx, cancel := context.WithCancel(context.Background())

	return &RolloutManager{
		rollouts:            make(map[string]*ProgressiveRollout),
		regionManager:       regionManager,
		segmentTargeting:    segmentTargeting,
		riskAssessor:        riskAssessor,
		decisionEngine:      decisionEngine,
		metricsAggregator:   metricsAggregator,
		notificationService: notificationService,
		ctx:                 ctx,
		cancel:              cancel,
	}
}

// CreateRollout creates a new progressive rollout
func (rm *RolloutManager) CreateRollout(
	name string,
	version string,
	strategy *RolloutStrategy,
) (*ProgressiveRollout, error) {
	rm.mu.Lock()
	defer rm.mu.Unlock()

	rolloutID := fmt.Sprintf("%s-%d", name, time.Now().Unix())

	rollout := &ProgressiveRollout{
		ID:               rolloutID,
		Name:             name,
		Version:          version,
		Strategy:         strategy,
		Status:           RolloutStatusInitializing,
		CurrentPhase:     0,
		AutomatedActions: strategy.AutomatedDecisions,
		StartTime:        time.Now(),
	}

	// Generate rollout phases based on strategy
	phases, err := rm.generatePhases(strategy)
	if err != nil {
		return nil, fmt.Errorf("failed to generate phases: %w", err)
	}
	rollout.Phases = phases

	rm.rollouts[rolloutID] = rollout

	// Start monitoring
	rm.wg.Add(1)
	go rm.monitorRollout(rolloutID)

	return rollout, nil
}

// generatePhases generates rollout phases based on strategy
func (rm *RolloutManager) generatePhases(strategy *RolloutStrategy) ([]RolloutPhase, error) {
	phases := []RolloutPhase{}

	switch strategy.Type {
	case StrategyGeographic:
		return rm.generateGeographicPhases(strategy)
	case StrategySegmented:
		return rm.generateSegmentedPhases(strategy)
	case StrategyRiskBased:
		return rm.generateRiskBasedPhases(strategy)
	case StrategyAdaptive:
		return rm.generateAdaptivePhases(strategy)
	default:
		return phases, fmt.Errorf("unknown strategy type: %s", strategy.Type)
	}
}

// generateGeographicPhases generates phases based on geographic order
func (rm *RolloutManager) generateGeographicPhases(strategy *RolloutStrategy) ([]RolloutPhase, error) {
	phases := []RolloutPhase{}

	if len(strategy.GeographicOrder) == 0 {
		return nil, fmt.Errorf("geographic order not specified")
	}

	// Phase 1: Early adopter regions (first 20%)
	earlyCount := int(math.Max(1, float64(len(strategy.GeographicOrder))*0.2))
	phases = append(phases, RolloutPhase{
		PhaseNumber:    1,
		Name:           "Early Adopters",
		Regions:        strategy.GeographicOrder[:earlyCount],
		TrafficPercent: 100.0,
		Status:         PhaseStatusPending,
		Duration:       strategy.PhaseDuration,
	})

	// Phase 2: Next 30%
	nextCount := int(math.Max(1, float64(len(strategy.GeographicOrder))*0.3))
	if earlyCount+nextCount <= len(strategy.GeographicOrder) {
		phases = append(phases, RolloutPhase{
			PhaseNumber:    2,
			Name:           "Expansion",
			Regions:        strategy.GeographicOrder[earlyCount : earlyCount+nextCount],
			TrafficPercent: 100.0,
			Status:         PhaseStatusPending,
			Duration:       strategy.PhaseDuration,
		})
	}

	// Phase 3: Remaining 50%
	if earlyCount+nextCount < len(strategy.GeographicOrder) {
		phases = append(phases, RolloutPhase{
			PhaseNumber:    3,
			Name:           "Full Rollout",
			Regions:        strategy.GeographicOrder[earlyCount+nextCount:],
			TrafficPercent: 100.0,
			Status:         PhaseStatusPending,
			Duration:       strategy.PhaseDuration,
		})
	}

	return phases, nil
}

// generateSegmentedPhases generates phases based on customer segments
func (rm *RolloutManager) generateSegmentedPhases(strategy *RolloutStrategy) ([]RolloutPhase, error) {
	phases := []RolloutPhase{}

	if len(strategy.SegmentPriority) == 0 {
		return nil, fmt.Errorf("segment priority not specified")
	}

	for i, segment := range strategy.SegmentPriority {
		phases = append(phases, RolloutPhase{
			PhaseNumber:    i + 1,
			Name:           fmt.Sprintf("Segment: %s", segment),
			Segments:       []string{segment},
			TrafficPercent: 100.0,
			Status:         PhaseStatusPending,
			Duration:       strategy.PhaseDuration,
		})
	}

	return phases, nil
}

// generateRiskBasedPhases generates phases based on risk assessment
func (rm *RolloutManager) generateRiskBasedPhases(strategy *RolloutStrategy) ([]RolloutPhase, error) {
	// Start with low-risk regions/segments, gradually expand
	phases := []RolloutPhase{
		{
			PhaseNumber:    1,
			Name:           "Low Risk",
			TrafficPercent: 10.0,
			Status:         PhaseStatusPending,
			Duration:       strategy.PhaseDuration,
		},
		{
			PhaseNumber:    2,
			Name:           "Medium Risk",
			TrafficPercent: 30.0,
			Status:         PhaseStatusPending,
			Duration:       strategy.PhaseDuration,
		},
		{
			PhaseNumber:    3,
			Name:           "High Traffic",
			TrafficPercent: 100.0,
			Status:         PhaseStatusPending,
			Duration:       strategy.PhaseDuration,
		},
	}

	return phases, nil
}

// generateAdaptivePhases generates adaptive phases
func (rm *RolloutManager) generateAdaptivePhases(strategy *RolloutStrategy) ([]RolloutPhase, error) {
	// Adaptive strategy adjusts phases based on real-time metrics
	phases := []RolloutPhase{
		{
			PhaseNumber:    1,
			Name:           "Initial",
			TrafficPercent: 5.0,
			Status:         PhaseStatusPending,
			Duration:       strategy.PhaseDuration / 2,
		},
		{
			PhaseNumber:    2,
			Name:           "Adaptive Expansion",
			TrafficPercent: 25.0,
			Status:         PhaseStatusPending,
			Duration:       strategy.PhaseDuration,
		},
		{
			PhaseNumber:    3,
			Name:           "Adaptive Full",
			TrafficPercent: 100.0,
			Status:         PhaseStatusPending,
			Duration:       strategy.PhaseDuration,
		},
	}

	return phases, nil
}

// StartRollout starts the progressive rollout
func (rm *RolloutManager) StartRollout(rolloutID string) error {
	rm.mu.Lock()
	rollout, exists := rm.rollouts[rolloutID]
	if !exists {
		rm.mu.Unlock()
		return fmt.Errorf("rollout %s not found", rolloutID)
	}

	rollout.Status = RolloutStatusProgressing
	rm.mu.Unlock()

	return rm.executeNextPhase(rolloutID)
}

// executeNextPhase executes the next phase in the rollout
func (rm *RolloutManager) executeNextPhase(rolloutID string) error {
	rm.mu.RLock()
	rollout, exists := rm.rollouts[rolloutID]
	if !exists {
		rm.mu.RUnlock()
		return fmt.Errorf("rollout %s not found", rolloutID)
	}

	if rollout.CurrentPhase >= len(rollout.Phases) {
		rm.mu.RUnlock()
		return rm.completeRollout(rolloutID)
	}

	phase := &rollout.Phases[rollout.CurrentPhase]
	rm.mu.RUnlock()

	// Start phase
	now := time.Now()
	phase.StartTime = &now
	phase.Status = PhaseStatusActive

	// Deploy to regions
	for _, region := range phase.Regions {
		if err := rm.regionManager.DeployToRegion(rm.ctx, region, rollout.Version, phase.TrafficPercent); err != nil {
			return fmt.Errorf("failed to deploy to region %s: %w", region, err)
		}
	}

	// Deploy to segments
	for _, segment := range phase.Segments {
		if err := rm.segmentTargeting.DeployToSegment(rm.ctx, segment, rollout.Version, phase.TrafficPercent); err != nil {
			return fmt.Errorf("failed to deploy to segment %s: %w", segment, err)
		}
	}

	// Wait for phase duration
	timer := time.NewTimer(phase.Duration)
	ticker := time.NewTicker(30 * time.Second)
	defer timer.Stop()
	defer ticker.Stop()

	for {
		select {
		case <-timer.C:
			// Phase duration complete, validate
			return rm.validateAndProceed(rolloutID)

		case <-ticker.C:
			// Continuous monitoring and automated decisions
			if rollout.AutomatedActions {
				if err := rm.makeAutomatedDecision(rolloutID); err != nil {
					return err
				}
			}

		case <-rm.ctx.Done():
			return fmt.Errorf("rollout cancelled")
		}
	}
}

// validateAndProceed validates current phase and proceeds to next
func (rm *RolloutManager) validateAndProceed(rolloutID string) error {
	rm.mu.RLock()
	rollout, exists := rm.rollouts[rolloutID]
	if !exists {
		rm.mu.RUnlock()
		return fmt.Errorf("rollout %s not found", rolloutID)
	}

	phase := &rollout.Phases[rollout.CurrentPhase]
	rm.mu.RUnlock()

	// Collect phase metrics
	metrics, err := rm.metricsAggregator.GetPhaseMetrics(rm.ctx, rolloutID, rollout.CurrentPhase)
	if err != nil {
		return fmt.Errorf("failed to collect phase metrics: %w", err)
	}

	phase.Metrics = metrics

	// Evaluate phase success
	passed, reason, err := rm.decisionEngine.EvaluatePhase(rm.ctx, phase, metrics)
	if err != nil {
		return fmt.Errorf("failed to evaluate phase: %w", err)
	}

	phase.Passed = passed
	if !passed {
		phase.Status = PhaseStatusFailed
		phase.FailureReason = reason
		return rm.initiateRollback(rolloutID, reason)
	}

	// Phase passed
	now := time.Now()
	phase.EndTime = &now
	phase.Status = PhaseStatusCompleted

	// Notify phase completion
	_ = rm.notificationService.NotifyPhaseComplete(rolloutID, phase)

	// Progress to next phase
	rm.mu.Lock()
	rollout.CurrentPhase++
	rm.mu.Unlock()

	return rm.executeNextPhase(rolloutID)
}

// makeAutomatedDecision makes automated decision based on current state
func (rm *RolloutManager) makeAutomatedDecision(rolloutID string) error {
	rm.mu.RLock()
	rollout, exists := rm.rollouts[rolloutID]
	if !exists {
		rm.mu.RUnlock()
		return fmt.Errorf("rollout %s not found", rolloutID)
	}
	rm.mu.RUnlock()

	// Make decision
	decision, err := rm.decisionEngine.MakeDecision(rm.ctx, rollout)
	if err != nil {
		return fmt.Errorf("failed to make decision: %w", err)
	}

	// Record decision
	rm.mu.Lock()
	rollout.Decisions = append(rollout.Decisions, *decision)
	rm.mu.Unlock()

	// Notify decision
	_ = rm.notificationService.NotifyDecision(rolloutID, decision)

	// Execute decision if confidence is high
	if decision.Confidence >= 0.8 {
		return rm.executeDecision(rolloutID, decision)
	}

	return nil
}

// executeDecision executes an automated decision
func (rm *RolloutManager) executeDecision(rolloutID string, decision *AutomatedDecision) error {
	switch decision.Action {
	case ActionProceedNextPhase:
		return rm.validateAndProceed(rolloutID)
	case ActionPauseRollout:
		return rm.PauseRollout(rolloutID, decision.Reason)
	case ActionInitiateRollback:
		return rm.initiateRollback(rolloutID, decision.Reason)
	case ActionIncreaseTraffic:
		// Increase traffic percentage
		return nil
	case ActionDecreaseTraffic:
		// Decrease traffic percentage
		return nil
	default:
		return fmt.Errorf("unknown decision action: %s", decision.Action)
	}
}

// initiateRollback initiates a rollback
func (rm *RolloutManager) initiateRollback(rolloutID string, reason string) error {
	rm.mu.Lock()
	rollout, exists := rm.rollouts[rolloutID]
	if !exists {
		rm.mu.Unlock()
		return fmt.Errorf("rollout %s not found", rolloutID)
	}

	rollout.Status = RolloutStatusRollingBack
	rm.mu.Unlock()

	// Notify rollback
	_ = rm.notificationService.NotifyRollback(rolloutID, reason)

	// Rollback all deployed regions and segments
	// Implementation would revert traffic routing

	rm.mu.Lock()
	rollout.Status = RolloutStatusFailed
	now := time.Now()
	rollout.EndTime = &now
	rm.mu.Unlock()

	return nil
}

// completeRollout completes the rollout
func (rm *RolloutManager) completeRollout(rolloutID string) error {
	rm.mu.Lock()
	rollout, exists := rm.rollouts[rolloutID]
	if !exists {
		rm.mu.Unlock()
		return fmt.Errorf("rollout %s not found", rolloutID)
	}

	rollout.Status = RolloutStatusCompleted
	now := time.Now()
	rollout.EndTime = &now
	rm.mu.Unlock()

	// Notify completion
	_ = rm.notificationService.NotifyCompletion(rolloutID, true)

	return nil
}

// PauseRollout pauses a rollout
func (rm *RolloutManager) PauseRollout(rolloutID string, reason string) error {
	rm.mu.Lock()
	defer rm.mu.Unlock()

	rollout, exists := rm.rollouts[rolloutID]
	if !exists {
		return fmt.Errorf("rollout %s not found", rolloutID)
	}

	rollout.Status = RolloutStatusPaused
	rollout.PauseReason = reason

	return nil
}

// ResumeRollout resumes a paused rollout
func (rm *RolloutManager) ResumeRollout(rolloutID string) error {
	rm.mu.Lock()
	rollout, exists := rm.rollouts[rolloutID]
	if !exists {
		rm.mu.Unlock()
		return fmt.Errorf("rollout %s not found", rolloutID)
	}

	if rollout.Status != RolloutStatusPaused {
		rm.mu.Unlock()
		return fmt.Errorf("rollout is not paused")
	}

	rollout.Status = RolloutStatusProgressing
	rollout.PauseReason = ""
	rm.mu.Unlock()

	return rm.executeNextPhase(rolloutID)
}

// monitorRollout continuously monitors a rollout
func (rm *RolloutManager) monitorRollout(rolloutID string) {
	defer rm.wg.Done()

	ticker := time.NewTicker(1 * time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			rm.mu.RLock()
			rollout, exists := rm.rollouts[rolloutID]
			if !exists {
				rm.mu.RUnlock()
				return
			}

			// Stop monitoring if rollout is complete or failed
			if rollout.Status == RolloutStatusCompleted || rollout.Status == RolloutStatusFailed {
				rm.mu.RUnlock()
				return
			}
			rm.mu.RUnlock()

			// Collect metrics
			metrics, err := rm.metricsAggregator.AggregateMetrics(rm.ctx, rolloutID)
			if err == nil {
				rm.mu.Lock()
				rollout.Metrics = metrics
				rm.mu.Unlock()
			}

			// Assess risk
			assessment, err := rm.riskAssessor.AssessRisk(rm.ctx, rolloutID)
			if err == nil {
				rm.mu.Lock()
				rollout.RiskScore = assessment.RiskScore
				rollout.RiskLevel = assessment.RiskLevel
				rm.mu.Unlock()
			}

		case <-rm.ctx.Done():
			return
		}
	}
}

// GetRollout retrieves a rollout
func (rm *RolloutManager) GetRollout(rolloutID string) (*ProgressiveRollout, error) {
	rm.mu.RLock()
	defer rm.mu.RUnlock()

	rollout, exists := rm.rollouts[rolloutID]
	if !exists {
		return nil, fmt.Errorf("rollout %s not found", rolloutID)
	}

	return rollout, nil
}

// ListRollouts lists all rollouts
func (rm *RolloutManager) ListRollouts() []*ProgressiveRollout {
	rm.mu.RLock()
	defer rm.mu.RUnlock()

	rollouts := make([]*ProgressiveRollout, 0, len(rm.rollouts))
	for _, rollout := range rm.rollouts {
		rollouts = append(rollouts, rollout)
	}

	return rollouts
}

// Shutdown gracefully shuts down the manager
func (rm *RolloutManager) Shutdown(ctx context.Context) error {
	rm.cancel()

	done := make(chan struct{})
	go func() {
		rm.wg.Wait()
		close(done)
	}()

	select {
	case <-done:
		return nil
	case <-ctx.Done():
		return fmt.Errorf("shutdown timeout exceeded")
	}
}

// MarshalJSON implements custom JSON marshaling
func (pr *ProgressiveRollout) MarshalJSON() ([]byte, error) {
	type Alias ProgressiveRollout

	return json.Marshal(&struct {
		*Alias
		Progress float64 `json:"progress"`
		Duration string  `json:"duration"`
	}{
		Alias: (*Alias)(pr),
		Progress: func() float64 {
			if len(pr.Phases) == 0 {
				return 0.0
			}
			return float64(pr.CurrentPhase) / float64(len(pr.Phases))
		}(),
		Duration: func() string {
			if pr.EndTime != nil {
				return pr.EndTime.Sub(pr.StartTime).String()
			}
			return time.Since(pr.StartTime).String()
		}(),
	})
}
