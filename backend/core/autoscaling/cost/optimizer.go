package cost

import (
	"context"
	"fmt"
	"math"
	"sync"
	"time"
)

// CostModel represents a cost model for different resource types and cloud providers
type CostModel interface {
	// GetHourlyCost returns the hourly cost for a specific resource configuration
	GetHourlyCost(resourceType string, config ResourceConfig) (float64, error)
	
	// GetPredictedCost returns the predicted cost over a time period
	GetPredictedCost(resourceType string, config ResourceConfig, duration time.Duration) (float64, error)
	
	// GetScalingCost returns the cost of a scaling operation (startup, shutdown, migration costs)
	GetScalingCost(operation ScalingOperation) (float64, error)
	
	// GetCostBreakdown returns a detailed cost breakdown
	GetCostBreakdown(resourceType string, config ResourceConfig) (*CostBreakdown, error)
	
	// SupportsResourceType checks if the cost model supports a resource type
	SupportsResourceType(resourceType string) bool
	
	// GetProviderName returns the name of the cloud provider
	GetProviderName() string
}

// ResourceConfig represents configuration for a resource
type ResourceConfig struct {
	ResourceType   string                 `json:"resource_type"` // vm, container, storage, etc.
	InstanceType   string                 `json:"instance_type"` // t3.medium, n1-standard-1, etc.
	Region         string                 `json:"region"`
	Zone           string                 `json:"zone,omitempty"`
	
	// Resource specifications
	CPUCores       int                    `json:"cpu_cores"`
	MemoryGB       float64                `json:"memory_gb"`
	StorageGB      int                    `json:"storage_gb"`
	NetworkMbps    int                    `json:"network_mbps,omitempty"`
	
	// Pricing options
	PricingModel   string                 `json:"pricing_model"` // on-demand, spot, reserved
	ReservedTerm   int                    `json:"reserved_term,omitempty"` // months for reserved instances
	
	// Additional attributes
	Attributes     map[string]interface{} `json:"attributes,omitempty"`
}

// ScalingOperation represents a scaling operation with associated costs
type ScalingOperation struct {
	Type           string                 `json:"type"` // scale_out, scale_in, scale_up, scale_down
	FromConfig     ResourceConfig         `json:"from_config"`
	ToConfig       ResourceConfig         `json:"to_config"`
	InstanceCount  int                    `json:"instance_count"`
	EstimatedTime  time.Duration          `json:"estimated_time"`
	
	// Migration-related costs
	DataTransferGB float64                `json:"data_transfer_gb,omitempty"`
	MigrationCost  float64                `json:"migration_cost,omitempty"`
}

// CostBreakdown provides detailed cost information
type CostBreakdown struct {
	TotalHourlyCost   float64            `json:"total_hourly_cost"`
	ComputeCost       float64            `json:"compute_cost"`
	StorageCost       float64            `json:"storage_cost"`
	NetworkCost       float64            `json:"network_cost"`
	OperatingCost     float64            `json:"operating_cost"`
	
	// Discounts and savings
	Discounts         []CostDiscount     `json:"discounts,omitempty"`
	Savings           float64            `json:"savings"`
	
	// Additional cost factors
	AdditionalCosts   map[string]float64 `json:"additional_costs,omitempty"`
}

// CostDiscount represents a pricing discount
type CostDiscount struct {
	Name        string  `json:"name"`
	Type        string  `json:"type"` // percentage, fixed
	Value       float64 `json:"value"`
	Description string  `json:"description"`
}

// BudgetConstraint represents budget constraints for scaling decisions
type BudgetConstraint struct {
	Name           string        `json:"name"`
	MaxHourlyCost  float64       `json:"max_hourly_cost"`
	MaxDailyCost   float64       `json:"max_daily_cost"`
	MaxMonthlyCost float64       `json:"max_monthly_cost"`
	
	// Time-based constraints
	TimeWindow     string        `json:"time_window"` // daily, weekly, monthly
	ResetTime      time.Time     `json:"reset_time"`
	
	// Resource-specific constraints
	ResourceLimits map[string]BudgetResourceLimit `json:"resource_limits,omitempty"`
	
	// Alert thresholds
	WarningThreshold float64     `json:"warning_threshold"` // percentage of budget
	AlertThreshold   float64     `json:"alert_threshold"`
}

// BudgetResourceLimit represents budget limits for specific resource types
type BudgetResourceLimit struct {
	ResourceType   string  `json:"resource_type"`
	MaxHourlyCost  float64 `json:"max_hourly_cost"`
	MaxInstances   int     `json:"max_instances"`
	Priority       int     `json:"priority"` // 1 = highest priority
}

// ScalingDecision represents a cost-optimized scaling decision
type ScalingDecision struct {
	RecommendedAction string             `json:"recommended_action"` // scale_out, scale_in, no_action
	Confidence        float64            `json:"confidence"`
	
	// Cost analysis
	CurrentCost       float64            `json:"current_cost"`
	ProjectedCost     float64            `json:"projected_cost"`
	CostSavings       float64            `json:"cost_savings"`
	ROI               float64            `json:"roi"` // Return on Investment
	
	// Resource optimization
	OptimalConfig     ResourceConfig     `json:"optimal_config"`
	AlternativeConfigs []ResourceConfig   `json:"alternative_configs,omitempty"`
	
	// Decision factors
	CostFactor        float64            `json:"cost_factor"`        // 0-1
	PerformanceFactor float64            `json:"performance_factor"` // 0-1
	RiskFactor        float64            `json:"risk_factor"`        // 0-1
	
	// Constraints
	BudgetCompliant   bool               `json:"budget_compliant"`
	ConstraintViolations []string        `json:"constraint_violations,omitempty"`
	
	// Metadata
	DecisionReason    string             `json:"decision_reason"`
	GeneratedAt       time.Time          `json:"generated_at"`
	ValidUntil        time.Time          `json:"valid_until"`
}

// CostOptimizer provides cost-aware scaling decisions
type CostOptimizer struct {
	mu               sync.RWMutex
	costModels       map[string]CostModel    // provider -> cost model
	budgetConstraints map[string]*BudgetConstraint // constraint name -> constraint
	
	// Optimization settings
	config           CostOptimizerConfig
	
	// Cost tracking
	currentCosts     map[string]float64     // resource ID -> current hourly cost
	costHistory      map[string][]CostPoint // resource ID -> cost history
	budgetUsage      map[string]*BudgetUsage // constraint name -> usage
}

// CostOptimizerConfig configures the cost optimizer
type CostOptimizerConfig struct {
	// Optimization weights (must sum to 1.0)
	CostWeight        float64 `json:"cost_weight"`        // 0.4 = 40% weight on cost
	PerformanceWeight float64 `json:"performance_weight"` // 0.4 = 40% weight on performance  
	RiskWeight        float64 `json:"risk_weight"`        // 0.2 = 20% weight on risk
	
	// Decision thresholds
	MinCostSavings    float64 `json:"min_cost_savings"`    // minimum $ savings to trigger scaling
	MinROIThreshold   float64 `json:"min_roi_threshold"`   // minimum ROI % to justify scaling
	MaxRiskScore      float64 `json:"max_risk_score"`      // maximum acceptable risk score (0-1)
	
	// Time horizons for cost analysis
	ShortTermHorizon  time.Duration `json:"short_term_horizon"`  // 1 hour
	MediumTermHorizon time.Duration `json:"medium_term_horizon"` // 1 day
	LongTermHorizon   time.Duration `json:"long_term_horizon"`   // 1 week
	
	// Spot instance settings
	UseSpotInstances    bool    `json:"use_spot_instances"`
	MaxSpotRisk         float64 `json:"max_spot_risk"`         // maximum acceptable spot termination risk
	SpotSavingsThreshold float64 `json:"spot_savings_threshold"` // minimum savings % to use spot
	
	// Reserved instance settings  
	ConsiderReservedInstances bool `json:"consider_reserved_instances"`
	ReservedInstanceLookback  time.Duration `json:"reserved_instance_lookback"` // historical usage period to analyze
	
	// Update intervals
	CostUpdateInterval    time.Duration `json:"cost_update_interval"`
	BudgetCheckInterval   time.Duration `json:"budget_check_interval"`
}

// CostPoint represents cost at a specific point in time
type CostPoint struct {
	Timestamp time.Time `json:"timestamp"`
	Cost      float64   `json:"cost"`
	ResourceID string   `json:"resource_id"`
}

// BudgetUsage tracks budget usage over time
type BudgetUsage struct {
	ConstraintName string      `json:"constraint_name"`
	CurrentUsage   float64     `json:"current_usage"`
	PercentUsed    float64     `json:"percent_used"`
	ResetTime      time.Time   `json:"reset_time"`
	
	// Usage tracking
	HourlyUsage    []float64   `json:"hourly_usage"`
	DailyUsage     []float64   `json:"daily_usage"`
	MonthlyUsage   []float64   `json:"monthly_usage"`
	
	// Alerts
	LastWarning    time.Time   `json:"last_warning"`
	LastAlert      time.Time   `json:"last_alert"`
}

// NewCostOptimizer creates a new cost optimizer
func NewCostOptimizer(config CostOptimizerConfig) *CostOptimizer {
	// Set default weights if not provided
	if config.CostWeight+config.PerformanceWeight+config.RiskWeight == 0 {
		config.CostWeight = 0.5
		config.PerformanceWeight = 0.3  
		config.RiskWeight = 0.2
	}
	
	// Set default thresholds
	if config.MinCostSavings == 0 {
		config.MinCostSavings = 0.10 // $0.10/hour minimum savings
	}
	if config.MinROIThreshold == 0 {
		config.MinROIThreshold = 0.05 // 5% minimum ROI
	}
	if config.MaxRiskScore == 0 {
		config.MaxRiskScore = 0.3 // 30% maximum risk
	}
	
	// Set default time horizons
	if config.ShortTermHorizon == 0 {
		config.ShortTermHorizon = 1 * time.Hour
	}
	if config.MediumTermHorizon == 0 {
		config.MediumTermHorizon = 24 * time.Hour
	}
	if config.LongTermHorizon == 0 {
		config.LongTermHorizon = 7 * 24 * time.Hour
	}
	
	// Set default update intervals
	if config.CostUpdateInterval == 0 {
		config.CostUpdateInterval = 5 * time.Minute
	}
	if config.BudgetCheckInterval == 0 {
		config.BudgetCheckInterval = 1 * time.Hour
	}
	
	return &CostOptimizer{
		costModels:       make(map[string]CostModel),
		budgetConstraints: make(map[string]*BudgetConstraint),
		config:           config,
		currentCosts:     make(map[string]float64),
		costHistory:      make(map[string][]CostPoint),
		budgetUsage:      make(map[string]*BudgetUsage),
	}
}

// RegisterCostModel registers a cost model for a provider
func (o *CostOptimizer) RegisterCostModel(provider string, model CostModel) error {
	o.mu.Lock()
	defer o.mu.Unlock()
	
	if provider == "" {
		return fmt.Errorf("provider name cannot be empty")
	}
	
	if model == nil {
		return fmt.Errorf("cost model cannot be nil")
	}
	
	o.costModels[provider] = model
	return nil
}

// RegisterBudgetConstraint registers a budget constraint
func (o *CostOptimizer) RegisterBudgetConstraint(constraint *BudgetConstraint) error {
	if constraint == nil {
		return fmt.Errorf("constraint cannot be nil")
	}
	
	if constraint.Name == "" {
		return fmt.Errorf("constraint name cannot be empty")
	}
	
	o.mu.Lock()
	defer o.mu.Unlock()
	
	o.budgetConstraints[constraint.Name] = constraint
	
	// Initialize budget usage tracking
	o.budgetUsage[constraint.Name] = &BudgetUsage{
		ConstraintName: constraint.Name,
		ResetTime:      constraint.ResetTime,
		HourlyUsage:    make([]float64, 0),
		DailyUsage:     make([]float64, 0),
		MonthlyUsage:   make([]float64, 0),
	}
	
	return nil
}

// OptimizeScaling provides cost-optimized scaling recommendations
func (o *CostOptimizer) OptimizeScaling(ctx context.Context, request ScalingOptimizationRequest) (*ScalingDecision, error) {
	o.mu.RLock()
	defer o.mu.RUnlock()
	
	// Get cost model for the provider
	costModel, exists := o.costModels[request.Provider]
	if !exists {
		return nil, fmt.Errorf("no cost model registered for provider: %s", request.Provider)
	}
	
	// Calculate current cost
	currentCost, err := o.calculateCurrentCost(costModel, request.CurrentConfig, request.CurrentInstances)
	if err != nil {
		return nil, fmt.Errorf("failed to calculate current cost: %w", err)
	}
	
	// Generate scaling options
	scalingOptions, err := o.generateScalingOptions(ctx, costModel, request)
	if err != nil {
		return nil, fmt.Errorf("failed to generate scaling options: %w", err)
	}
	
	// Evaluate each option
	bestDecision := &ScalingDecision{
		RecommendedAction: "no_action",
		CurrentCost:       currentCost,
		ProjectedCost:     currentCost,
		Confidence:        0.0,
		GeneratedAt:       time.Now(),
		ValidUntil:        time.Now().Add(15 * time.Minute),
	}
	
	for _, option := range scalingOptions {
		decision, err := o.evaluateScalingOption(ctx, costModel, request, option)
		if err != nil {
			continue // Skip invalid options
		}
		
		// Calculate combined score
		score := o.calculateCombinedScore(decision)
		if score > bestDecision.Confidence {
			bestDecision = decision
			bestDecision.Confidence = score
		}
	}
	
	// Check budget constraints
	o.checkBudgetCompliance(bestDecision)
	
	return bestDecision, nil
}

// ScalingOptimizationRequest represents a request for scaling optimization
type ScalingOptimizationRequest struct {
	Provider         string          `json:"provider"`
	ResourceID       string          `json:"resource_id"`
	CurrentConfig    ResourceConfig  `json:"current_config"`
	CurrentInstances int             `json:"current_instances"`
	
	// Scaling context
	ScalingTrigger   string          `json:"scaling_trigger"` // cpu_high, memory_high, custom
	MetricValue      float64         `json:"metric_value"`
	MetricThreshold  float64         `json:"metric_threshold"`
	
	// Demand forecasting
	PredictedDemand  []DemandPoint   `json:"predicted_demand,omitempty"`
	DemandConfidence float64         `json:"demand_confidence,omitempty"`
	
	// Constraints
	MinInstances     int             `json:"min_instances"`
	MaxInstances     int             `json:"max_instances"`
	ConstraintNames  []string        `json:"constraint_names,omitempty"`
}

// DemandPoint represents predicted demand at a point in time
type DemandPoint struct {
	Timestamp time.Time `json:"timestamp"`
	Demand    float64   `json:"demand"`    // normalized demand (0-1)
	Confidence float64  `json:"confidence"`
}

// ScalingOption represents a potential scaling configuration
type ScalingOption struct {
	Action         string         `json:"action"`
	TargetConfig   ResourceConfig `json:"target_config"`
	TargetInstances int           `json:"target_instances"`
	ScalingTime    time.Duration  `json:"scaling_time"`
	
	// Cost estimates
	HourlyCost     float64        `json:"hourly_cost"`
	ScalingCost    float64        `json:"scaling_cost"`
	
	// Performance estimates
	PerformanceGain float64       `json:"performance_gain"` // expected performance improvement
	PerformanceRisk float64       `json:"performance_risk"` // risk of performance degradation
	
	// Risk factors
	SpotTerminationRisk float64   `json:"spot_termination_risk,omitempty"`
	AvailabilityRisk    float64   `json:"availability_risk,omitempty"`
}

// calculateCurrentCost calculates the current hourly cost
func (o *CostOptimizer) calculateCurrentCost(model CostModel, config ResourceConfig, instances int) (float64, error) {
	hourlyCost, err := model.GetHourlyCost("vm", config)
	if err != nil {
		return 0, fmt.Errorf("failed to get hourly cost: %w", err)
	}
	
	return hourlyCost * float64(instances), nil
}

// generateScalingOptions generates potential scaling configurations
func (o *CostOptimizer) generateScalingOptions(ctx context.Context, model CostModel, request ScalingOptimizationRequest) ([]*ScalingOption, error) {
	options := make([]*ScalingOption, 0)
	
	// Generate scale out options
	for instances := request.CurrentInstances + 1; instances <= request.MaxInstances; instances++ {
		option, err := o.createScalingOption("scale_out", model, request, instances, request.CurrentConfig)
		if err == nil {
			options = append(options, option)
		}
	}
	
	// Generate scale in options  
	for instances := request.CurrentInstances - 1; instances >= request.MinInstances; instances-- {
		option, err := o.createScalingOption("scale_in", model, request, instances, request.CurrentConfig)
		if err == nil {
			options = append(options, option)
		}
	}
	
	// Generate vertical scaling options (different instance types)
	verticalOptions := o.generateVerticalScalingOptions(model, request)
	options = append(options, verticalOptions...)
	
	// Generate spot instance options if enabled
	if o.config.UseSpotInstances {
		spotOptions := o.generateSpotInstanceOptions(model, request)
		options = append(options, spotOptions...)
	}
	
	return options, nil
}

// createScalingOption creates a scaling option for horizontal scaling
func (o *CostOptimizer) createScalingOption(action string, model CostModel, request ScalingOptimizationRequest, 
	targetInstances int, config ResourceConfig) (*ScalingOption, error) {
	
	// Calculate costs
	hourlyCost, err := model.GetHourlyCost("vm", config)
	if err != nil {
		return nil, err
	}
	
	totalHourlyCost := hourlyCost * float64(targetInstances)
	
	// Estimate scaling cost
	scalingOp := ScalingOperation{
		Type:          action,
		FromConfig:    request.CurrentConfig,
		ToConfig:      config,
		InstanceCount: abs(targetInstances - request.CurrentInstances),
		EstimatedTime: time.Duration(abs(targetInstances-request.CurrentInstances)) * 2 * time.Minute, // 2 min per instance
	}
	
	scalingCost, err := model.GetScalingCost(scalingOp)
	if err != nil {
		scalingCost = 0 // Assume no scaling cost if not available
	}
	
	// Estimate performance impact
	performanceGain := o.estimatePerformanceGain(action, request.CurrentInstances, targetInstances)
	performanceRisk := o.estimatePerformanceRisk(action, config)
	
	return &ScalingOption{
		Action:          action,
		TargetConfig:    config,
		TargetInstances: targetInstances,
		ScalingTime:     scalingOp.EstimatedTime,
		HourlyCost:      totalHourlyCost,
		ScalingCost:     scalingCost,
		PerformanceGain: performanceGain,
		PerformanceRisk: performanceRisk,
	}, nil
}

// generateVerticalScalingOptions generates options for vertical scaling
func (o *CostOptimizer) generateVerticalScalingOptions(model CostModel, request ScalingOptimizationRequest) []*ScalingOption {
	options := make([]*ScalingOption, 0)
	
	// Generate larger instance types (scale up)
	largerConfigs := o.generateLargerConfigs(request.CurrentConfig)
	for _, config := range largerConfigs {
		if option, err := o.createScalingOption("scale_up", model, request, request.CurrentInstances, config); err == nil {
			options = append(options, option)
		}
	}
	
	// Generate smaller instance types (scale down)
	smallerConfigs := o.generateSmallerConfigs(request.CurrentConfig)
	for _, config := range smallerConfigs {
		if option, err := o.createScalingOption("scale_down", model, request, request.CurrentInstances, config); err == nil {
			options = append(options, option)
		}
	}
	
	return options
}

// generateSpotInstanceOptions generates spot instance options
func (o *CostOptimizer) generateSpotInstanceOptions(model CostModel, request ScalingOptimizationRequest) []*ScalingOption {
	options := make([]*ScalingOption, 0)
	
	// Create spot instance version of current config
	spotConfig := request.CurrentConfig
	spotConfig.PricingModel = "spot"
	
	// Generate spot scaling options
	for instances := request.MinInstances; instances <= request.MaxInstances; instances++ {
		if instances == request.CurrentInstances {
			continue // Skip current configuration
		}
		
		action := "scale_out"
		if instances < request.CurrentInstances {
			action = "scale_in"
		}
		
		option, err := o.createScalingOption(action+"_spot", model, request, instances, spotConfig)
		if err == nil {
			// Add spot-specific risk
			option.SpotTerminationRisk = o.estimateSpotTerminationRisk(spotConfig)
			if option.SpotTerminationRisk <= o.config.MaxSpotRisk {
				options = append(options, option)
			}
		}
	}
	
	return options
}

// evaluateScalingOption evaluates a scaling option and creates a decision
func (o *CostOptimizer) evaluateScalingOption(ctx context.Context, model CostModel, request ScalingOptimizationRequest, 
	option *ScalingOption) (*ScalingDecision, error) {
	
	// Calculate cost metrics
	currentCost, _ := o.calculateCurrentCost(model, request.CurrentConfig, request.CurrentInstances)
	projectedCost := option.HourlyCost
	costSavings := currentCost - projectedCost
	
	// Calculate ROI (return on investment)
	roi := 0.0
	if option.ScalingCost > 0 {
		roi = (costSavings * o.config.MediumTermHorizon.Hours()) / option.ScalingCost
	}
	
	// Calculate decision factors (0-1 scale)
	costFactor := o.calculateCostFactor(costSavings, currentCost)
	performanceFactor := option.PerformanceGain - option.PerformanceRisk
	riskFactor := 1.0 - (option.PerformanceRisk + option.SpotTerminationRisk + option.AvailabilityRisk)
	
	// Ensure factors are in valid range
	performanceFactor = math.Max(0, math.Min(1, performanceFactor))
	riskFactor = math.Max(0, math.Min(1, riskFactor))
	
	decision := &ScalingDecision{
		RecommendedAction: option.Action,
		CurrentCost:       currentCost,
		ProjectedCost:     projectedCost,
		CostSavings:       costSavings,
		ROI:               roi,
		OptimalConfig:     option.TargetConfig,
		CostFactor:        costFactor,
		PerformanceFactor: performanceFactor,
		RiskFactor:        riskFactor,
		BudgetCompliant:   true, // Will be checked later
		DecisionReason:    fmt.Sprintf("Cost savings: $%.2f/hour, ROI: %.1f%%", costSavings, roi*100),
		GeneratedAt:       time.Now(),
		ValidUntil:        time.Now().Add(15 * time.Minute),
	}
	
	return decision, nil
}

// calculateCombinedScore calculates a weighted score for a scaling decision
func (o *CostOptimizer) calculateCombinedScore(decision *ScalingDecision) float64 {
	// Weighted combination of factors
	score := (o.config.CostWeight * decision.CostFactor) +
		(o.config.PerformanceWeight * decision.PerformanceFactor) +
		(o.config.RiskWeight * decision.RiskFactor)
	
	// Apply thresholds
	if decision.CostSavings < o.config.MinCostSavings {
		score *= 0.5 // Penalize if savings too small
	}
	
	if decision.ROI < o.config.MinROIThreshold {
		score *= 0.5 // Penalize if ROI too low
	}
	
	if decision.RiskFactor < (1.0 - o.config.MaxRiskScore) {
		score *= 0.3 // Heavily penalize if risk too high
	}
	
	return math.Max(0, math.Min(1, score))
}

// calculateCostFactor calculates the cost factor (0-1) based on cost savings
func (o *CostOptimizer) calculateCostFactor(savings, currentCost float64) float64 {
	if currentCost == 0 {
		return 0
	}
	
	savingsPercent := savings / currentCost
	
	// Convert savings percentage to 0-1 scale
	// 50% savings = 1.0, 0% savings = 0.5, -50% (increased cost) = 0.0
	factor := 0.5 + (savingsPercent / 1.0) // Normalize around 50% savings
	return math.Max(0, math.Min(1, factor))
}

// checkBudgetCompliance checks if a decision violates budget constraints
func (o *CostOptimizer) checkBudgetCompliance(decision *ScalingDecision) {
	decision.BudgetCompliant = true
	decision.ConstraintViolations = make([]string, 0)
	
	for name, constraint := range o.budgetConstraints {
		usage, exists := o.budgetUsage[name]
		if !exists {
			continue
		}
		
		// Check hourly budget
		if constraint.MaxHourlyCost > 0 && decision.ProjectedCost > constraint.MaxHourlyCost {
			decision.BudgetCompliant = false
			decision.ConstraintViolations = append(decision.ConstraintViolations,
				fmt.Sprintf("Exceeds hourly budget %s: $%.2f > $%.2f", name, decision.ProjectedCost, constraint.MaxHourlyCost))
		}
		
		// Check daily budget
		if constraint.MaxDailyCost > 0 {
			projectedDailyCost := decision.ProjectedCost * 24
			currentDailyUsage := usage.CurrentUsage * 24
			if currentDailyUsage+projectedDailyCost > constraint.MaxDailyCost {
				decision.BudgetCompliant = false
				decision.ConstraintViolations = append(decision.ConstraintViolations,
					fmt.Sprintf("Exceeds daily budget %s: $%.2f > $%.2f", name, currentDailyUsage+projectedDailyCost, constraint.MaxDailyCost))
			}
		}
	}
}

// Helper functions

// generateLargerConfigs generates configurations with more resources
func (o *CostOptimizer) generateLargerConfigs(current ResourceConfig) []ResourceConfig {
	configs := make([]ResourceConfig, 0)
	
	// Double CPU and memory
	config1 := current
	config1.CPUCores = current.CPUCores * 2
	config1.MemoryGB = current.MemoryGB * 2
	configs = append(configs, config1)
	
	// 1.5x CPU and memory  
	config2 := current
	config2.CPUCores = int(float64(current.CPUCores) * 1.5)
	config2.MemoryGB = current.MemoryGB * 1.5
	configs = append(configs, config2)
	
	return configs
}

// generateSmallerConfigs generates configurations with fewer resources
func (o *CostOptimizer) generateSmallerConfigs(current ResourceConfig) []ResourceConfig {
	configs := make([]ResourceConfig, 0)
	
	// Half CPU and memory
	if current.CPUCores >= 2 {
		config1 := current
		config1.CPUCores = current.CPUCores / 2
		config1.MemoryGB = current.MemoryGB / 2
		configs = append(configs, config1)
	}
	
	// 0.75x CPU and memory
	config2 := current
	config2.CPUCores = int(float64(current.CPUCores) * 0.75)
	config2.MemoryGB = current.MemoryGB * 0.75
	if config2.CPUCores >= 1 && config2.MemoryGB >= 1 {
		configs = append(configs, config2)
	}
	
	return configs
}

// estimatePerformanceGain estimates performance improvement from scaling
func (o *CostOptimizer) estimatePerformanceGain(action string, currentInstances, targetInstances int) float64 {
	switch action {
	case "scale_out":
		// Linear performance gain with diminishing returns
		ratio := float64(targetInstances) / float64(currentInstances)
		return math.Min(0.8, math.Log(ratio)*0.5) // Cap at 80% gain
		
	case "scale_in":
		// Performance loss with scaling in
		ratio := float64(currentInstances) / float64(targetInstances) 
		return -math.Min(0.6, math.Log(ratio)*0.3) // Cap at 60% loss
		
	case "scale_up":
		// Vertical scaling gain (generally good for single-threaded workloads)
		return 0.4 // Assume 40% gain
		
	case "scale_down":
		// Vertical scaling loss
		return -0.3 // Assume 30% loss
		
	default:
		return 0.0
	}
}

// estimatePerformanceRisk estimates the risk of performance issues
func (o *CostOptimizer) estimatePerformanceRisk(action string, config ResourceConfig) float64 {
	baseRisk := 0.05 // 5% base risk
	
	// Higher risk for spot instances
	if config.PricingModel == "spot" {
		baseRisk += 0.15
	}
	
	// Higher risk for scaling operations
	switch action {
	case "scale_out", "scale_up":
		baseRisk += 0.05 // 5% additional risk
	case "scale_in", "scale_down":
		baseRisk += 0.10 // 10% additional risk
	}
	
	return math.Min(1.0, baseRisk)
}

// estimateSpotTerminationRisk estimates the risk of spot instance termination
func (o *CostOptimizer) estimateSpotTerminationRisk(config ResourceConfig) float64 {
	if config.PricingModel != "spot" {
		return 0.0
	}
	
	// Base spot termination risk varies by instance type and region
	baseRisk := 0.20 // 20% base risk
	
	// Popular instance types have higher termination risk
	if config.InstanceType == "m5.large" || config.InstanceType == "t3.medium" {
		baseRisk += 0.10
	}
	
	// Some regions have higher spot termination rates
	if config.Region == "us-west-1" || config.Region == "eu-west-1" {
		baseRisk += 0.05
	}
	
	return math.Min(1.0, baseRisk)
}

// abs returns the absolute value of an integer
func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// GetCurrentCosts returns current costs for all tracked resources
func (o *CostOptimizer) GetCurrentCosts() map[string]float64 {
	o.mu.RLock()
	defer o.mu.RUnlock()
	
	costs := make(map[string]float64)
	for resourceID, cost := range o.currentCosts {
		costs[resourceID] = cost
	}
	
	return costs
}

// UpdateResourceCost updates the current cost for a resource
func (o *CostOptimizer) UpdateResourceCost(resourceID string, cost float64) {
	o.mu.Lock()
	defer o.mu.Unlock()
	
	o.currentCosts[resourceID] = cost
	
	// Add to cost history
	point := CostPoint{
		Timestamp:  time.Now(),
		Cost:       cost,
		ResourceID: resourceID,
	}
	
	if _, exists := o.costHistory[resourceID]; !exists {
		o.costHistory[resourceID] = make([]CostPoint, 0)
	}
	
	o.costHistory[resourceID] = append(o.costHistory[resourceID], point)
	
	// Keep only recent history (last 24 hours)
	cutoff := time.Now().Add(-24 * time.Hour)
	filtered := make([]CostPoint, 0)
	for _, p := range o.costHistory[resourceID] {
		if p.Timestamp.After(cutoff) {
			filtered = append(filtered, p)
		}
	}
	o.costHistory[resourceID] = filtered
}

// GetCostHistory returns cost history for a resource
func (o *CostOptimizer) GetCostHistory(resourceID string) []CostPoint {
	o.mu.RLock()
	defer o.mu.RUnlock()
	
	if history, exists := o.costHistory[resourceID]; exists {
		result := make([]CostPoint, len(history))
		copy(result, history)
		return result
	}
	
	return []CostPoint{}
}

// GetBudgetUsage returns budget usage for a constraint
func (o *CostOptimizer) GetBudgetUsage(constraintName string) (*BudgetUsage, error) {
	o.mu.RLock()
	defer o.mu.RUnlock()
	
	if usage, exists := o.budgetUsage[constraintName]; exists {
		return usage, nil
	}
	
	return nil, fmt.Errorf("budget usage not found for constraint: %s", constraintName)
}

// UpdateBudgetUsage updates budget usage for a constraint
func (o *CostOptimizer) UpdateBudgetUsage(constraintName string, cost float64) error {
	o.mu.Lock()
	defer o.mu.Unlock()
	
	usage, exists := o.budgetUsage[constraintName]
	if !exists {
		return fmt.Errorf("budget constraint not found: %s", constraintName)
	}
	
	constraint := o.budgetConstraints[constraintName]
	
	// Update current usage
	usage.CurrentUsage = cost
	
	// Calculate percentage used
	if constraint.MaxHourlyCost > 0 {
		usage.PercentUsed = (cost / constraint.MaxHourlyCost) * 100
	}
	
	// Check for budget alerts
	if usage.PercentUsed >= constraint.AlertThreshold {
		usage.LastAlert = time.Now()
	} else if usage.PercentUsed >= constraint.WarningThreshold {
		usage.LastWarning = time.Now()
	}
	
	return nil
}