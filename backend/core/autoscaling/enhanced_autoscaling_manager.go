package autoscaling

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sync"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/autoscaling/predictive"
	"github.com/khryptorgraphics/novacron/backend/core/autoscaling/cost"
	"github.com/khryptorgraphics/novacron/backend/core/autoscaling/forecasting"
	"github.com/khryptorgraphics/novacron/backend/core/monitoring"
	"github.com/khryptorgraphics/novacron/backend/core/vm"
)

// EnhancedAutoScalingManager combines all Phase 2 autoscaling capabilities
type EnhancedAutoScalingManager struct {
	// Core autoscaling manager (existing)
	*AutoScalingManager
	
	// Phase 2 enhancements
	predictiveEngine    *predictive.PredictiveEngine
	costOptimizer      *cost.CostOptimizer
	capacityPlanner    *forecasting.CapacityPlanner
	
	// Enhanced configuration
	config             EnhancedAutoScalingConfig
	
	// ML-based scaling policies
	mlPolicies         map[string]*MLScalingPolicy
	
	// Analytics and history
	scalingAnalytics   *ScalingAnalytics
	
	// VM integration
	vmManager          vm.VMManagerInterface
	
	// Background processing
	ctx                context.Context
	cancel             context.CancelFunc
	
	// Synchronization
	mu                 sync.RWMutex
}

// EnhancedAutoScalingConfig extends the basic autoscaling configuration
type EnhancedAutoScalingConfig struct {
	// Basic configuration
	BasicConfig        AutoScalingManager `json:"basic_config"`
	
	// Predictive scaling
	PredictiveScaling  PredictiveScalingConfig `json:"predictive_scaling"`
	
	// Cost optimization
	CostOptimization   cost.CostOptimizerConfig `json:"cost_optimization"`
	
	// Capacity planning
	CapacityPlanning   forecasting.CapacityPlannerConfig `json:"capacity_planning"`
	
	// ML policies
	EnableMLPolicies   bool                     `json:"enable_ml_policies"`
	MLModelUpdateInterval time.Duration         `json:"ml_model_update_interval"`
	
	// Analytics
	EnableAnalytics    bool                     `json:"enable_analytics"`
	AnalyticsRetention time.Duration            `json:"analytics_retention"`
	
	// Integration settings
	VMManagerEnabled   bool                     `json:"vm_manager_enabled"`
	MetricsCollection  bool                     `json:"metrics_collection"`
	
	// Advanced scaling options
	MultiObjectiveOptimization bool             `json:"multi_objective_optimization"`
	RiskAwareScaling   bool                     `json:"risk_aware_scaling"`
	BudgetConstraints  []cost.BudgetConstraint  `json:"budget_constraints"`
}

// MLScalingPolicy represents a machine learning-based scaling policy
type MLScalingPolicy struct {
	ID                 string                 `json:"id"`
	Name               string                 `json:"name"`
	ScalingGroupID     string                 `json:"scaling_group_id"`
	
	// ML model configuration
	ModelType          predictive.ModelType   `json:"model_type"`
	ModelConfig        predictive.ModelConfig `json:"model_config"`
	
	// Prediction parameters
	PredictionHorizon  time.Duration          `json:"prediction_horizon"`
	ConfidenceThreshold float64               `json:"confidence_threshold"`
	
	// Scaling parameters
	ScaleUpThreshold   float64                `json:"scale_up_threshold"`
	ScaleDownThreshold float64                `json:"scale_down_threshold"`
	
	// Multi-objective weights
	CostWeight         float64                `json:"cost_weight"`
	PerformanceWeight  float64                `json:"performance_weight"`
	RiskWeight         float64                `json:"risk_weight"`
	
	// Advanced settings
	SeasonalAdjustment bool                   `json:"seasonal_adjustment"`
	TrendAdjustment    bool                   `json:"trend_adjustment"`
	
	// State
	Enabled            bool                   `json:"enabled"`
	LastUpdate         time.Time              `json:"last_update"`
	LastExecution      time.Time              `json:"last_execution"`
	Accuracy           float64                `json:"accuracy"`
}

// ScalingAnalytics provides comprehensive analytics for scaling operations
type ScalingAnalytics struct {
	mu                     sync.RWMutex
	
	// Metrics tracking
	scalingEvents          []EnhancedScalingEvent `json:"scaling_events"`
	performanceMetrics     map[string][]PerformanceMetric `json:"performance_metrics"`
	costMetrics           map[string][]CostMetric `json:"cost_metrics"`
	
	// Aggregated statistics
	totalScalingActions   int64                  `json:"total_scaling_actions"`
	successfulScalings    int64                  `json:"successful_scalings"`
	failedScalings        int64                  `json:"failed_scalings"`
	averageScalingTime    time.Duration          `json:"average_scaling_time"`
	
	// Cost analysis
	totalCostSavings      float64                `json:"total_cost_savings"`
	averageROI            float64                `json:"average_roi"`
	
	// Performance analysis
	averageResponseTime   time.Duration          `json:"average_response_time"`
	systemAvailability    float64                `json:"system_availability"`
	
	// Configuration
	retentionPeriod       time.Duration          `json:"retention_period"`
	lastCleanup           time.Time              `json:"last_cleanup"`
}

// EnhancedScalingEvent extends the basic scaling event with additional information
type EnhancedScalingEvent struct {
	*ScalingEvent
	
	// ML predictions
	PredictedDemand       *predictive.ForecastResult `json:"predicted_demand,omitempty"`
	PredictionAccuracy    float64                    `json:"prediction_accuracy"`
	
	// Cost analysis
	CostDecision          *cost.ScalingDecision      `json:"cost_decision,omitempty"`
	ActualCostImpact      float64                    `json:"actual_cost_impact"`
	
	// Capacity planning
	CapacityRecommendation *forecasting.CapacityRecommendation `json:"capacity_recommendation,omitempty"`
	
	// Risk assessment
	RiskScore             float64                    `json:"risk_score"`
	RiskFactors           []string                   `json:"risk_factors,omitempty"`
	
	// Performance impact
	PerformanceBaseline   map[string]float64         `json:"performance_baseline"`
	PerformanceAfterScale map[string]float64         `json:"performance_after_scale"`
	
	// Decision metadata
	DecisionAlgorithm     string                     `json:"decision_algorithm"`
	DecisionFactors       map[string]float64         `json:"decision_factors"`
}

// PerformanceMetric represents a performance metric at a point in time
type PerformanceMetric struct {
	Timestamp       time.Time              `json:"timestamp"`
	ResourceID      string                 `json:"resource_id"`
	MetricName      string                 `json:"metric_name"`
	Value           float64                `json:"value"`
	Baseline        float64                `json:"baseline"`
	Target          float64                `json:"target"`
	ScalingEventID  string                 `json:"scaling_event_id,omitempty"`
}

// CostMetric represents a cost metric at a point in time
type CostMetric struct {
	Timestamp       time.Time              `json:"timestamp"`
	ResourceID      string                 `json:"resource_id"`
	HourlyCost      float64                `json:"hourly_cost"`
	MonthlyCost     float64                `json:"monthly_cost"`
	CostBreakdown   *cost.CostBreakdown    `json:"cost_breakdown,omitempty"`
	ScalingEventID  string                 `json:"scaling_event_id,omitempty"`
}

// NewEnhancedAutoScalingManager creates a new enhanced autoscaling manager
func NewEnhancedAutoScalingManager(
	metricsProvider monitoring.MetricsProvider,
	resourceController ResourceController,
	vmManager vm.VMManagerInterface,
	config EnhancedAutoScalingConfig,
) (*EnhancedAutoScalingManager, error) {
	
	// Create basic autoscaling manager
	basicManager := NewAutoScalingManager(metricsProvider, resourceController)
	
	ctx, cancel := context.WithCancel(context.Background())
	
	// Create predictive engine
	predictiveEngine := predictive.NewPredictiveEngine(
		metricsProvider,
		config.PredictiveScaling,
	)
	
	// Create cost optimizer
	costOptimizer := cost.NewCostOptimizer(config.CostOptimization)
	
	// Create capacity planner
	capacityPlanner := forecasting.NewCapacityPlanner(
		predictiveEngine,
		costOptimizer,
		metricsProvider,
		config.CapacityPlanning,
	)
	
	// Create scaling analytics
	scalingAnalytics := &ScalingAnalytics{
		scalingEvents:      make([]EnhancedScalingEvent, 0),
		performanceMetrics: make(map[string][]PerformanceMetric),
		costMetrics:        make(map[string][]CostMetric),
		retentionPeriod:    config.AnalyticsRetention,
		lastCleanup:        time.Now(),
	}
	
	if config.AnalyticsRetention == 0 {
		scalingAnalytics.retentionPeriod = 30 * 24 * time.Hour // 30 days default
	}
	
	enhanced := &EnhancedAutoScalingManager{
		AutoScalingManager:  basicManager,
		predictiveEngine:    predictiveEngine,
		costOptimizer:      costOptimizer,
		capacityPlanner:    capacityPlanner,
		config:             config,
		mlPolicies:         make(map[string]*MLScalingPolicy),
		scalingAnalytics:   scalingAnalytics,
		vmManager:          vmManager,
		ctx:                ctx,
		cancel:             cancel,
	}
	
	// Register budget constraints
	for _, constraint := range config.BudgetConstraints {
		if err := costOptimizer.RegisterBudgetConstraint(&constraint); err != nil {
			return nil, fmt.Errorf("failed to register budget constraint: %w", err)
		}
	}
	
	return enhanced, nil
}

// Start starts the enhanced autoscaling manager
func (e *EnhancedAutoScalingManager) Start() error {
	// Start basic autoscaling manager
	if err := e.AutoScalingManager.Initialize(e.ctx); err != nil {
		return fmt.Errorf("failed to initialize basic autoscaling manager: %w", err)
	}
	
	// Start enhanced components
	if err := e.predictiveEngine.Start(); err != nil {
		return fmt.Errorf("failed to start predictive engine: %w", err)
	}
	
	if err := e.capacityPlanner.Start(); err != nil {
		return fmt.Errorf("failed to start capacity planner: %w", err)
	}
	
	// Start enhanced scaling loop
	if err := e.startEnhancedScalingLoop(); err != nil {
		return fmt.Errorf("failed to start enhanced scaling loop: %w", err)
	}
	
	// Start background processes
	go e.mlPolicyUpdateLoop()
	go e.analyticsLoop()
	
	return nil
}

// Stop stops the enhanced autoscaling manager
func (e *EnhancedAutoScalingManager) Stop() error {
	// Stop basic components
	if err := e.AutoScalingManager.Shutdown(); err != nil {
		return fmt.Errorf("failed to shutdown basic autoscaling manager: %w", err)
	}
	
	// Stop enhanced components
	if err := e.predictiveEngine.Stop(); err != nil {
		return fmt.Errorf("failed to stop predictive engine: %w", err)
	}
	
	if err := e.capacityPlanner.Stop(); err != nil {
		return fmt.Errorf("failed to stop capacity planner: %w", err)
	}
	
	// Cancel context
	if e.cancel != nil {
		e.cancel()
	}
	
	return nil
}

// RegisterMLPolicy registers a machine learning-based scaling policy
func (e *EnhancedAutoScalingManager) RegisterMLPolicy(policy *MLScalingPolicy) error {
	if !e.config.EnableMLPolicies {
		return fmt.Errorf("ML policies are disabled")
	}
	
	if policy.ID == "" {
		return fmt.Errorf("ML policy ID cannot be empty")
	}
	
	e.mu.Lock()
	defer e.mu.Unlock()
	
	if _, exists := e.mlPolicies[policy.ID]; exists {
		return fmt.Errorf("ML policy with ID %s already exists", policy.ID)
	}
	
	// Validate configuration
	if policy.CostWeight+policy.PerformanceWeight+policy.RiskWeight != 1.0 {
		return fmt.Errorf("ML policy weights must sum to 1.0")
	}
	
	if policy.ConfidenceThreshold <= 0 || policy.ConfidenceThreshold > 1 {
		return fmt.Errorf("confidence threshold must be between 0 and 1")
	}
	
	// Register the predictive model
	err := e.predictiveEngine.RegisterMetric(policy.ScalingGroupID+"_demand", policy.ModelConfig)
	if err != nil {
		return fmt.Errorf("failed to register predictive model: %w", err)
	}
	
	policy.LastUpdate = time.Now()
	policy.Enabled = true
	
	e.mlPolicies[policy.ID] = policy
	
	return nil
}

// UnregisterMLPolicy unregisters an ML scaling policy
func (e *EnhancedAutoScalingManager) UnregisterMLPolicy(policyID string) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	
	policy, exists := e.mlPolicies[policyID]
	if !exists {
		return fmt.Errorf("ML policy %s not found", policyID)
	}
	
	// Unregister predictive model
	err := e.predictiveEngine.UnregisterMetric(policy.ScalingGroupID + "_demand")
	if err != nil {
		// Log warning but don't fail
		fmt.Printf("Warning: failed to unregister predictive model: %v\n", err)
	}
	
	delete(e.mlPolicies, policyID)
	return nil
}

// EvaluateEnhancedScaling performs enhanced scaling evaluation
func (e *EnhancedAutoScalingManager) EvaluateEnhancedScaling(ctx context.Context) error {
	e.mu.RLock()
	groups := e.ListScalingGroups()
	policies := make([]*MLScalingPolicy, 0, len(e.mlPolicies))
	for _, policy := range e.mlPolicies {
		if policy.Enabled {
			policies = append(policies, policy)
		}
	}
	e.mu.RUnlock()
	
	for _, group := range groups {
		// Evaluate traditional rules first
		err := e.evaluateTraditionalScaling(ctx, group)
		if err != nil {
			fmt.Printf("Warning: traditional scaling evaluation failed for group %s: %v\n", group.ID, err)
		}
		
		// Evaluate ML policies
		for _, policy := range policies {
			if policy.ScalingGroupID == group.ID {
				err := e.evaluateMLPolicy(ctx, group, policy)
				if err != nil {
					fmt.Printf("Warning: ML policy evaluation failed for group %s: %v\n", group.ID, err)
				}
			}
		}
		
		// Evaluate capacity planning recommendations
		if e.config.CapacityPlanning.PlanningInterval > 0 {
			err := e.evaluateCapacityPlanningRecommendations(ctx, group)
			if err != nil {
				fmt.Printf("Warning: capacity planning evaluation failed for group %s: %v\n", group.ID, err)
			}
		}
	}
	
	return nil
}

// evaluateMLPolicy evaluates a machine learning-based scaling policy
func (e *EnhancedAutoScalingManager) evaluateMLPolicy(ctx context.Context, group *ScalingGroup, policy *MLScalingPolicy) error {
	// Get demand forecast
	forecastRequest := predictive.ForecastRequest{
		MetricName: policy.ScalingGroupID + "_demand",
		Horizon:    policy.PredictionHorizon,
		Interval:   5 * time.Minute,
	}
	
	forecast, err := e.predictiveEngine.GetForecast(ctx, forecastRequest)
	if err != nil {
		return fmt.Errorf("failed to get demand forecast: %w", err)
	}
	
	// Check forecast confidence
	if forecast.ModelAccuracy < policy.ConfidenceThreshold {
		// Forecast not confident enough, skip ML-based scaling
		return nil
	}
	
	// Calculate peak demand from forecast
	peakDemand := 0.0
	for _, prediction := range forecast.Predictions {
		if prediction.Value > peakDemand {
			peakDemand = prediction.Value
		}
	}
	
	// Apply seasonal and trend adjustments
	if policy.SeasonalAdjustment {
		// TODO: Apply seasonal adjustments based on historical patterns
	}
	
	if policy.TrendAdjustment {
		// TODO: Apply trend adjustments based on trend analysis
	}
	
	// Multi-objective optimization
	var scalingDecision *cost.ScalingDecision
	if e.config.MultiObjectiveOptimization && e.config.CostOptimization.CostOptimizationEnabled {
		// Get cost-optimized scaling recommendation
		costRequest := cost.ScalingOptimizationRequest{
			Provider:         "generic", // TODO: detect actual provider
			ResourceID:       group.ID,
			CurrentInstances: group.CurrentCapacity,
			ScalingTrigger:   "ml_forecast",
			MetricValue:      peakDemand,
			MetricThreshold:  policy.ScaleUpThreshold,
			MinInstances:     group.MinCapacity,
			MaxInstances:     group.MaxCapacity,
		}
		
		scalingDecision, err = e.costOptimizer.OptimizeScaling(ctx, costRequest)
		if err != nil {
			return fmt.Errorf("failed to get cost-optimized scaling decision: %w", err)
		}
	}
	
	// Determine scaling action based on ML prediction and cost optimization
	action := e.determineMLScalingAction(peakDemand, policy, scalingDecision)
	
	// Execute scaling if needed
	if action != ScalingActionNone {
		return e.executeEnhancedScaling(ctx, group, action, policy, forecast, scalingDecision)
	}
	
	return nil
}

// determineMLScalingAction determines the scaling action based on ML predictions
func (e *EnhancedAutoScalingManager) determineMLScalingAction(
	peakDemand float64,
	policy *MLScalingPolicy,
	costDecision *cost.ScalingDecision,
) ScalingAction {
	
	if peakDemand >= policy.ScaleUpThreshold {
		// Cost optimization check
		if costDecision != nil && !costDecision.BudgetCompliant {
			return ScalingActionNone // Budget constraints prevent scaling
		}
		return ScalingActionScaleOut
		
	} else if peakDemand <= policy.ScaleDownThreshold {
		// Always allow scaling down for cost savings
		return ScalingActionScaleIn
		
	}
	
	return ScalingActionNone
}

// executeEnhancedScaling executes enhanced scaling with full analytics
func (e *EnhancedAutoScalingManager) executeEnhancedScaling(
	ctx context.Context,
	group *ScalingGroup,
	action ScalingAction,
	policy *MLScalingPolicy,
	forecast *predictive.ForecastResult,
	costDecision *cost.ScalingDecision,
) error {
	
	// Record pre-scaling metrics
	preScalingMetrics, err := e.capturePerformanceMetrics(ctx, group.ID)
	if err != nil {
		fmt.Printf("Warning: failed to capture pre-scaling metrics: %v\n", err)
	}
	
	// Calculate new capacity
	var newCapacity int
	switch action {
	case ScalingActionScaleOut:
		increment := 1
		if costDecision != nil {
			// Use cost-optimized increment
			increment = costDecision.NewCapacity - group.CurrentCapacity
		}
		newCapacity = group.CurrentCapacity + increment
		if newCapacity > group.MaxCapacity {
			newCapacity = group.MaxCapacity
		}
		
	case ScalingActionScaleIn:
		decrement := 1
		if costDecision != nil {
			// Use cost-optimized decrement
			decrement = group.CurrentCapacity - costDecision.NewCapacity
		}
		newCapacity = group.CurrentCapacity - decrement
		if newCapacity < group.MinCapacity {
			newCapacity = group.MinCapacity
		}
		
	default:
		return fmt.Errorf("unsupported scaling action: %s", action)
	}
	
	// Execute scaling
	startTime := time.Now()
	err = e.SetGroupCapacity(ctx, group.ID, newCapacity)
	scalingDuration := time.Since(startTime)
	
	// Create enhanced scaling event
	event := &EnhancedScalingEvent{
		ScalingEvent: &ScalingEvent{
			ID:               fmt.Sprintf("ml-evt-%s-%d", group.ID, time.Now().Unix()),
			Timestamp:        time.Now(),
			ScalingGroupID:   group.ID,
			Action:           action,
			PreviousCapacity: group.CurrentCapacity,
			NewCapacity:      newCapacity,
			Status:          "completed",
			Details: map[string]interface{}{
				"policy_id":        policy.ID,
				"ml_model_type":    policy.ModelType,
				"scaling_duration": scalingDuration,
			},
		},
		PredictedDemand:       forecast,
		PredictionAccuracy:    forecast.ModelAccuracy,
		CostDecision:          costDecision,
		PerformanceBaseline:   preScalingMetrics,
		DecisionAlgorithm:     "ml_enhanced",
		DecisionFactors: map[string]float64{
			"cost_weight":        policy.CostWeight,
			"performance_weight": policy.PerformanceWeight,
			"risk_weight":        policy.RiskWeight,
		},
	}
	
	if err != nil {
		event.Status = "failed"
		event.Details["error"] = err.Error()
	}
	
	// Calculate risk score
	event.RiskScore = e.calculateRiskScore(group, action, forecast, costDecision)
	event.RiskFactors = e.identifyRiskFactors(group, action, forecast)
	
	// Record the event
	e.recordEnhancedScalingEvent(event)
	
	// Capture post-scaling metrics (after a delay)
	go func() {
		time.Sleep(2 * time.Minute) // Allow system to stabilize
		postScalingMetrics, err := e.capturePerformanceMetrics(ctx, group.ID)
		if err == nil {
			event.PerformanceAfterScale = postScalingMetrics
		}
	}()
	
	return err
}

// startEnhancedScalingLoop starts the enhanced scaling evaluation loop
func (e *EnhancedAutoScalingManager) startEnhancedScalingLoop() error {
	// Start enhanced evaluation loop
	go func() {
		ticker := time.NewTicker(1 * time.Minute) // More frequent evaluation for ML
		defer ticker.Stop()
		
		for {
			select {
			case <-ticker.C:
				if err := e.EvaluateEnhancedScaling(e.ctx); err != nil {
					fmt.Printf("Error in enhanced scaling evaluation: %v\n", err)
				}
			case <-e.ctx.Done():
				return
			}
		}
	}()
	
	return nil
}

// mlPolicyUpdateLoop updates ML policies periodically
func (e *EnhancedAutoScalingManager) mlPolicyUpdateLoop() {
	if !e.config.EnableMLPolicies {
		return
	}
	
	interval := e.config.MLModelUpdateInterval
	if interval == 0 {
		interval = 1 * time.Hour
	}
	
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			e.updateMLPolicies()
		case <-e.ctx.Done():
			return
		}
	}
}

// updateMLPolicies updates ML model accuracy and performance
func (e *EnhancedAutoScalingManager) updateMLPolicies() {
	e.mu.RLock()
	policies := make([]*MLScalingPolicy, 0, len(e.mlPolicies))
	for _, policy := range e.mlPolicies {
		policies = append(policies, policy)
	}
	e.mu.RUnlock()
	
	for _, policy := range policies {
		// Update model accuracy
		accuracy, err := e.predictiveEngine.GetModelAccuracy(policy.ScalingGroupID + "_demand")
		if err == nil {
			e.mu.Lock()
			policy.Accuracy = accuracy
			policy.LastUpdate = time.Now()
			e.mu.Unlock()
		}
		
		// Retrain model if accuracy is too low
		if accuracy < policy.ConfidenceThreshold {
			err := e.predictiveEngine.TrainModel(policy.ScalingGroupID + "_demand")
			if err != nil {
				fmt.Printf("Warning: failed to retrain model for policy %s: %v\n", policy.ID, err)
			}
		}
	}
}

// analyticsLoop runs the analytics background process
func (e *EnhancedAutoScalingManager) analyticsLoop() {
	if !e.config.EnableAnalytics {
		return
	}
	
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			e.updateAnalytics()
		case <-e.ctx.Done():
			return
		}
	}
}

// updateAnalytics updates scaling analytics
func (e *EnhancedAutoScalingManager) updateAnalytics() {
	e.scalingAnalytics.mu.Lock()
	defer e.scalingAnalytics.mu.Unlock()
	
	// Calculate aggregated statistics
	e.scalingAnalytics.totalScalingActions = int64(len(e.scalingAnalytics.scalingEvents))
	
	successful := int64(0)
	failed := int64(0)
	totalDuration := time.Duration(0)
	totalCostSavings := 0.0
	totalROI := 0.0
	
	for _, event := range e.scalingAnalytics.scalingEvents {
		if event.Status == "completed" {
			successful++
		} else {
			failed++
		}
		
		if duration, exists := event.Details["scaling_duration"]; exists {
			if dur, ok := duration.(time.Duration); ok {
				totalDuration += dur
			}
		}
		
		if event.CostDecision != nil {
			totalCostSavings += event.CostDecision.CostSavings
			totalROI += event.CostDecision.ROI
		}
	}
	
	e.scalingAnalytics.successfulScalings = successful
	e.scalingAnalytics.failedScalings = failed
	
	if successful > 0 {
		e.scalingAnalytics.averageScalingTime = totalDuration / time.Duration(successful)
		e.scalingAnalytics.averageROI = totalROI / float64(successful)
	}
	
	e.scalingAnalytics.totalCostSavings = totalCostSavings
	
	// Cleanup old data
	e.cleanupAnalyticsData()
}

// Helper methods

func (e *EnhancedAutoScalingManager) evaluateTraditionalScaling(ctx context.Context, group *ScalingGroup) error {
	// Use existing scaling rule evaluation
	return nil // Basic scaling is handled by the embedded AutoScalingManager
}

func (e *EnhancedAutoScalingManager) evaluateCapacityPlanningRecommendations(ctx context.Context, group *ScalingGroup) error {
	recommendation, err := e.capacityPlanner.GetCapacityRecommendation(ctx, group.ID)
	if err != nil {
		return err
	}
	
	// Execute high-priority recommendations
	if recommendation.Priority == "critical" || recommendation.Priority == "high" {
		if recommendation.RecommendedInstances != group.CurrentCapacity {
			return e.SetGroupCapacity(ctx, group.ID, recommendation.RecommendedInstances)
		}
	}
	
	return nil
}

func (e *EnhancedAutoScalingManager) capturePerformanceMetrics(ctx context.Context, resourceID string) (map[string]float64, error) {
	metrics := make(map[string]float64)
	
	// Capture key performance metrics
	metricNames := []string{"cpu_utilization", "memory_utilization", "response_time", "throughput"}
	
	for _, metricName := range metricNames {
		value, err := e.metricsProvider.GetMetric(
			ctx,
			monitoring.MetricType(metricName),
			resourceID,
			5*time.Minute,
		)
		if err == nil {
			metrics[metricName] = value
		}
	}
	
	return metrics, nil
}

func (e *EnhancedAutoScalingManager) calculateRiskScore(
	group *ScalingGroup,
	action ScalingAction,
	forecast *predictive.ForecastResult,
	costDecision *cost.ScalingDecision,
) float64 {
	// Base risk factors
	riskScore := 0.0
	
	// Prediction confidence risk
	if forecast != nil {
		riskScore += (1.0 - forecast.ModelAccuracy) * 0.3
	}
	
	// Scaling operation risk
	switch action {
	case ScalingActionScaleOut:
		riskScore += 0.1 // Low risk
	case ScalingActionScaleIn:
		riskScore += 0.2 // Medium risk
	case ScalingActionScaleUp, ScalingActionScaleDown:
		riskScore += 0.3 // Higher risk due to instance reconfiguration
	}
	
	// Cost risk
	if costDecision != nil {
		if costDecision.CostFactor < 0.5 {
			riskScore += 0.2 // High cost increase risk
		}
		if !costDecision.BudgetCompliant {
			riskScore += 0.3 // Budget violation risk
		}
	}
	
	return math.Min(1.0, riskScore)
}

func (e *EnhancedAutoScalingManager) identifyRiskFactors(
	group *ScalingGroup,
	action ScalingAction,
	forecast *predictive.ForecastResult,
) []string {
	factors := make([]string, 0)
	
	if forecast != nil && forecast.ModelAccuracy < 0.8 {
		factors = append(factors, "Low prediction confidence")
	}
	
	if group.CurrentCapacity >= int(float64(group.MaxCapacity)*0.9) {
		factors = append(factors, "Near maximum capacity")
	}
	
	if group.CurrentCapacity <= int(float64(group.MinCapacity)*1.1) {
		factors = append(factors, "Near minimum capacity")
	}
	
	return factors
}

func (e *EnhancedAutoScalingManager) recordEnhancedScalingEvent(event *EnhancedScalingEvent) {
	if !e.config.EnableAnalytics {
		return
	}
	
	e.scalingAnalytics.mu.Lock()
	defer e.scalingAnalytics.mu.Unlock()
	
	e.scalingAnalytics.scalingEvents = append(e.scalingAnalytics.scalingEvents, *event)
	
	// Keep only recent events
	maxEvents := 1000
	if len(e.scalingAnalytics.scalingEvents) > maxEvents {
		startIdx := len(e.scalingAnalytics.scalingEvents) - maxEvents
		e.scalingAnalytics.scalingEvents = e.scalingAnalytics.scalingEvents[startIdx:]
	}
}

func (e *EnhancedAutoScalingManager) cleanupAnalyticsData() {
	cutoff := time.Now().Add(-e.scalingAnalytics.retentionPeriod)
	
	// Clean scaling events
	filtered := make([]EnhancedScalingEvent, 0)
	for _, event := range e.scalingAnalytics.scalingEvents {
		if event.Timestamp.After(cutoff) {
			filtered = append(filtered, event)
		}
	}
	e.scalingAnalytics.scalingEvents = filtered
	
	// Clean performance metrics
	for resourceID, metrics := range e.scalingAnalytics.performanceMetrics {
		filteredMetrics := make([]PerformanceMetric, 0)
		for _, metric := range metrics {
			if metric.Timestamp.After(cutoff) {
				filteredMetrics = append(filteredMetrics, metric)
			}
		}
		e.scalingAnalytics.performanceMetrics[resourceID] = filteredMetrics
	}
	
	// Clean cost metrics
	for resourceID, metrics := range e.scalingAnalytics.costMetrics {
		filteredMetrics := make([]CostMetric, 0)
		for _, metric := range metrics {
			if metric.Timestamp.After(cutoff) {
				filteredMetrics = append(filteredMetrics, metric)
			}
		}
		e.scalingAnalytics.costMetrics[resourceID] = filteredMetrics
	}
	
	e.scalingAnalytics.lastCleanup = time.Now()
}

// Public API methods

// GetScalingAnalytics returns comprehensive scaling analytics
func (e *EnhancedAutoScalingManager) GetScalingAnalytics() *ScalingAnalytics {
	e.scalingAnalytics.mu.RLock()
	defer e.scalingAnalytics.mu.RUnlock()
	
	// Create a copy to avoid race conditions
	analytics := &ScalingAnalytics{
		totalScalingActions: e.scalingAnalytics.totalScalingActions,
		successfulScalings:  e.scalingAnalytics.successfulScalings,
		failedScalings:      e.scalingAnalytics.failedScalings,
		averageScalingTime:  e.scalingAnalytics.averageScalingTime,
		totalCostSavings:    e.scalingAnalytics.totalCostSavings,
		averageROI:          e.scalingAnalytics.averageROI,
		averageResponseTime: e.scalingAnalytics.averageResponseTime,
		systemAvailability:  e.scalingAnalytics.systemAvailability,
		retentionPeriod:     e.scalingAnalytics.retentionPeriod,
		lastCleanup:         e.scalingAnalytics.lastCleanup,
	}
	
	// Copy events
	analytics.scalingEvents = make([]EnhancedScalingEvent, len(e.scalingAnalytics.scalingEvents))
	copy(analytics.scalingEvents, e.scalingAnalytics.scalingEvents)
	
	return analytics
}

// GetMLPolicies returns all ML scaling policies
func (e *EnhancedAutoScalingManager) GetMLPolicies() []*MLScalingPolicy {
	e.mu.RLock()
	defer e.mu.RUnlock()
	
	policies := make([]*MLScalingPolicy, 0, len(e.mlPolicies))
	for _, policy := range e.mlPolicies {
		policies = append(policies, policy)
	}
	
	return policies
}

// GetPredictiveForecasts returns predictive forecasts for all registered metrics
func (e *EnhancedAutoScalingManager) GetPredictiveForecasts(ctx context.Context, horizon time.Duration) (map[string]*predictive.ForecastResult, error) {
	registeredMetrics := e.predictiveEngine.ListRegisteredMetrics()
	
	requests := make([]predictive.ForecastRequest, len(registeredMetrics))
	for i, metric := range registeredMetrics {
		requests[i] = predictive.ForecastRequest{
			MetricName: metric,
			Horizon:    horizon,
			Interval:   15 * time.Minute,
		}
	}
	
	return e.predictiveEngine.GetMultipleForecasts(ctx, requests)
}

// GetCapacityRecommendations returns capacity recommendations for all resources
func (e *EnhancedAutoScalingManager) GetCapacityRecommendations(ctx context.Context) (map[string]*forecasting.CapacityRecommendation, error) {
	groups := e.ListScalingGroups()
	recommendations := make(map[string]*forecasting.CapacityRecommendation)
	
	for _, group := range groups {
		recommendation, err := e.capacityPlanner.GetCapacityRecommendation(ctx, group.ID)
		if err == nil {
			recommendations[group.ID] = recommendation
		}
	}
	
	return recommendations, nil
}

// GetCostOptimizationReport returns a cost optimization report
func (e *EnhancedAutoScalingManager) GetCostOptimizationReport(ctx context.Context) (*CostOptimizationReport, error) {
	currentCosts := e.costOptimizer.GetCurrentCosts()
	
	totalCurrentCost := 0.0
	for _, cost := range currentCosts {
		totalCurrentCost += cost
	}
	
	report := &CostOptimizationReport{
		TotalCurrentCost:  totalCurrentCost,
		TotalSavings:      e.scalingAnalytics.totalCostSavings,
		AverageROI:        e.scalingAnalytics.averageROI,
		GeneratedAt:       time.Now(),
		ResourceBreakdown: make(map[string]float64),
	}
	
	// Add resource breakdown
	for resourceID, cost := range currentCosts {
		report.ResourceBreakdown[resourceID] = cost
	}
	
	return report, nil
}

// CostOptimizationReport represents a cost optimization report
type CostOptimizationReport struct {
	TotalCurrentCost  float64            `json:"total_current_cost"`
	TotalSavings      float64            `json:"total_savings"`
	AverageROI        float64            `json:"average_roi"`
	GeneratedAt       time.Time          `json:"generated_at"`
	ResourceBreakdown map[string]float64 `json:"resource_breakdown"`
}