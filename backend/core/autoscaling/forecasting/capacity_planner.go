package forecasting

import (
	"context"
	"fmt"
	"math"
	"sort"
	"sync"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/autoscaling/predictive"
	"github.com/khryptorgraphics/novacron/backend/core/autoscaling/cost"
	"github.com/khryptorgraphics/novacron/backend/core/monitoring"
)

// CapacityPlanner provides resource forecasting and capacity planning
type CapacityPlanner struct {
	mu                  sync.RWMutex
	predictiveEngine    *predictive.PredictiveEngine
	costOptimizer       *cost.CostOptimizer
	metricsProvider     monitoring.MetricsProvider
	config              CapacityPlannerConfig
	
	// Resource tracking
	resources           map[string]*ResourceCapacity
	demandForecasts     map[string]*DemandForecast
	capacityRecommendations map[string]*CapacityRecommendation
	
	// Background processing
	ctx                 context.Context
	cancel              context.CancelFunc
	updateTicker        *time.Ticker
	
	// Historical data
	demandHistory       map[string][]DemandPoint
	capacityHistory     map[string][]CapacityPoint
	
	// Bottleneck detection
	bottleneckDetector  *BottleneckDetector
}

// CapacityPlannerConfig configures the capacity planner
type CapacityPlannerConfig struct {
	// Forecasting horizons
	ShortTermHorizon    time.Duration `json:"short_term_horizon"`    // 4 hours
	MediumTermHorizon   time.Duration `json:"medium_term_horizon"`   // 24 hours
	LongTermHorizon     time.Duration `json:"long_term_horizon"`     // 7 days
	
	// Capacity management
	TargetUtilization   float64       `json:"target_utilization"`    // 0.70 = 70%
	MaxUtilization      float64       `json:"max_utilization"`       // 0.85 = 85%
	BufferPercent       float64       `json:"buffer_percent"`        // 0.20 = 20% buffer
	
	// Planning intervals
	PlanningInterval    time.Duration `json:"planning_interval"`     // how often to generate plans
	ForecastInterval    time.Duration `json:"forecast_interval"`     // how often to update forecasts
	
	// Bottleneck detection
	EnableBottleneckDetection bool    `json:"enable_bottleneck_detection"`
	BottleneckThreshold       float64 `json:"bottleneck_threshold"`  // 0.90 = 90%
	
	// Cost optimization
	CostOptimizationEnabled   bool    `json:"cost_optimization_enabled"`
	MaxCostIncrease           float64 `json:"max_cost_increase"`     // 0.15 = 15% max cost increase
	
	// Data retention
	DemandHistoryRetention    time.Duration `json:"demand_history_retention"`
	CapacityHistoryRetention  time.Duration `json:"capacity_history_retention"`
}

// ResourceCapacity represents current resource capacity information
type ResourceCapacity struct {
	ResourceID       string                 `json:"resource_id"`
	ResourceType     string                 `json:"resource_type"`
	TotalCapacity    ResourceMetrics        `json:"total_capacity"`
	UsedCapacity     ResourceMetrics        `json:"used_capacity"`
	AvailableCapacity ResourceMetrics       `json:"available_capacity"`
	UtilizationPercent float64              `json:"utilization_percent"`
	
	// Scaling information
	MinInstances     int                    `json:"min_instances"`
	MaxInstances     int                    `json:"max_instances"`
	CurrentInstances int                    `json:"current_instances"`
	InstanceConfig   cost.ResourceConfig    `json:"instance_config"`
	
	// Time tracking
	LastUpdated      time.Time              `json:"last_updated"`
	
	// Metadata
	Tags             map[string]string      `json:"tags"`
	Attributes       map[string]interface{} `json:"attributes"`
}

// ResourceMetrics represents resource metrics
type ResourceMetrics struct {
	CPU         float64 `json:"cpu"`          // CPU cores
	Memory      float64 `json:"memory"`       // Memory in GB
	Storage     float64 `json:"storage"`      // Storage in GB
	Network     float64 `json:"network"`      // Network bandwidth in Mbps
	IOPS        float64 `json:"iops"`         // Storage IOPS
	CustomMetrics map[string]float64 `json:"custom_metrics,omitempty"`
}

// DemandForecast represents forecasted resource demand
type DemandForecast struct {
	ResourceID      string              `json:"resource_id"`
	MetricName      string              `json:"metric_name"`
	ForecastHorizon time.Duration       `json:"forecast_horizon"`
	GeneratedAt     time.Time           `json:"generated_at"`
	Confidence      float64             `json:"confidence"`
	
	// Forecast data points
	Predictions     []DemandPrediction  `json:"predictions"`
	
	// Seasonality information
	SeasonalPattern *SeasonalPattern    `json:"seasonal_pattern,omitempty"`
	
	// Trend information
	TrendDirection  string              `json:"trend_direction"` // increasing, decreasing, stable
	TrendStrength   float64             `json:"trend_strength"`  // 0-1
}

// DemandPrediction represents a single demand prediction point
type DemandPrediction struct {
	Timestamp       time.Time `json:"timestamp"`
	PredictedDemand float64   `json:"predicted_demand"`
	Confidence      float64   `json:"confidence"`
	Lower           float64   `json:"lower_bound"`
	Upper           float64   `json:"upper_bound"`
	
	// Contributing factors
	BaselineDemand  float64   `json:"baseline_demand"`
	SeasonalFactor  float64   `json:"seasonal_factor"`
	TrendFactor     float64   `json:"trend_factor"`
}

// SeasonalPattern represents detected seasonal patterns
type SeasonalPattern struct {
	PeriodType      string             `json:"period_type"`      // hourly, daily, weekly
	PeriodLength    int                `json:"period_length"`    // hours in the period
	Peaks           []SeasonalPeak     `json:"peaks"`
	Valleys         []SeasonalValley   `json:"valleys"`
	Strength        float64            `json:"strength"`         // 0-1
}

// SeasonalPeak represents a peak in seasonal pattern
type SeasonalPeak struct {
	TimeOffset      int     `json:"time_offset"`      // offset from period start (hours)
	Magnitude       float64 `json:"magnitude"`        // multiplier over baseline
	Duration        int     `json:"duration"`         // duration in hours
}

// SeasonalValley represents a valley in seasonal pattern
type SeasonalValley struct {
	TimeOffset      int     `json:"time_offset"`      // offset from period start (hours)
	Magnitude       float64 `json:"magnitude"`        // multiplier under baseline
	Duration        int     `json:"duration"`         // duration in hours
}

// CapacityRecommendation represents a capacity planning recommendation
type CapacityRecommendation struct {
	ResourceID           string                 `json:"resource_id"`
	RecommendationType   string                 `json:"recommendation_type"` // scale_up, scale_down, no_change
	Priority             string                 `json:"priority"`            // critical, high, medium, low
	Confidence           float64                `json:"confidence"`
	
	// Current state
	CurrentCapacity      ResourceMetrics        `json:"current_capacity"`
	CurrentUtilization   float64                `json:"current_utilization"`
	
	// Recommended state
	RecommendedCapacity  ResourceMetrics        `json:"recommended_capacity"`
	RecommendedInstances int                    `json:"recommended_instances"`
	RecommendedConfig    cost.ResourceConfig    `json:"recommended_config"`
	
	// Justification
	Reason               string                 `json:"reason"`
	RiskAssessment       *RiskAssessment        `json:"risk_assessment"`
	
	// Timeline
	ImplementBy          time.Time              `json:"implement_by"`
	EstimatedDuration    time.Duration          `json:"estimated_duration"`
	
	// Cost analysis
	CostImpact           *CostImpactAnalysis    `json:"cost_impact"`
	
	// Alternative options
	Alternatives         []AlternativeOption    `json:"alternatives,omitempty"`
}

// RiskAssessment represents risk analysis for a recommendation
type RiskAssessment struct {
	OverallRisk         string    `json:"overall_risk"`     // low, medium, high, critical
	RiskScore           float64   `json:"risk_score"`       // 0-1
	RiskFactors         []string  `json:"risk_factors"`
	MitigationStrategies []string `json:"mitigation_strategies"`
	
	// Specific risks
	PerformanceRisk     float64   `json:"performance_risk"`
	CostRisk            float64   `json:"cost_risk"`
	AvailabilityRisk    float64   `json:"availability_risk"`
	SecurityRisk        float64   `json:"security_risk"`
}

// CostImpactAnalysis represents cost impact analysis
type CostImpactAnalysis struct {
	CurrentMonthlyCost    float64   `json:"current_monthly_cost"`
	ProjectedMonthlyCost  float64   `json:"projected_monthly_cost"`
	CostDifference        float64   `json:"cost_difference"`
	CostChangePercent     float64   `json:"cost_change_percent"`
	
	// Breakdown
	ComputeCostChange     float64   `json:"compute_cost_change"`
	StorageCostChange     float64   `json:"storage_cost_change"`
	NetworkCostChange     float64   `json:"network_cost_change"`
	
	// ROI analysis
	ExpectedSavings       float64   `json:"expected_savings"`
	PaybackPeriod         time.Duration `json:"payback_period"`
	ROI                   float64   `json:"roi"`
}

// AlternativeOption represents an alternative capacity option
type AlternativeOption struct {
	Description         string              `json:"description"`
	RecommendedConfig   cost.ResourceConfig `json:"recommended_config"`
	EstimatedCost       float64             `json:"estimated_cost"`
	ExpectedPerformance float64             `json:"expected_performance"`
	RiskLevel           string              `json:"risk_level"`
	Pros                []string            `json:"pros"`
	Cons                []string            `json:"cons"`
}

// DemandPoint represents demand at a specific point in time
type DemandPoint struct {
	Timestamp time.Time `json:"timestamp"`
	Value     float64   `json:"value"`
	MetricName string   `json:"metric_name"`
}

// CapacityPoint represents capacity at a specific point in time
type CapacityPoint struct {
	Timestamp   time.Time       `json:"timestamp"`
	Capacity    ResourceMetrics `json:"capacity"`
	Utilization float64         `json:"utilization"`
	Instances   int             `json:"instances"`
}

// BottleneckDetector detects performance bottlenecks
type BottleneckDetector struct {
	thresholds      map[string]float64    // metric -> threshold
	detectionWindow time.Duration
	minSamples      int
}

// NewCapacityPlanner creates a new capacity planner
func NewCapacityPlanner(
	predictiveEngine *predictive.PredictiveEngine,
	costOptimizer *cost.CostOptimizer,
	metricsProvider monitoring.MetricsProvider,
	config CapacityPlannerConfig,
) *CapacityPlanner {
	
	// Set defaults
	if config.ShortTermHorizon == 0 {
		config.ShortTermHorizon = 4 * time.Hour
	}
	if config.MediumTermHorizon == 0 {
		config.MediumTermHorizon = 24 * time.Hour
	}
	if config.LongTermHorizon == 0 {
		config.LongTermHorizon = 7 * 24 * time.Hour
	}
	if config.TargetUtilization == 0 {
		config.TargetUtilization = 0.70
	}
	if config.MaxUtilization == 0 {
		config.MaxUtilization = 0.85
	}
	if config.BufferPercent == 0 {
		config.BufferPercent = 0.20
	}
	if config.PlanningInterval == 0 {
		config.PlanningInterval = 30 * time.Minute
	}
	if config.ForecastInterval == 0 {
		config.ForecastInterval = 15 * time.Minute
	}
	if config.BottleneckThreshold == 0 {
		config.BottleneckThreshold = 0.90
	}
	if config.MaxCostIncrease == 0 {
		config.MaxCostIncrease = 0.15
	}
	if config.DemandHistoryRetention == 0 {
		config.DemandHistoryRetention = 30 * 24 * time.Hour
	}
	if config.CapacityHistoryRetention == 0 {
		config.CapacityHistoryRetention = 30 * 24 * time.Hour
	}
	
	ctx, cancel := context.WithCancel(context.Background())
	
	// Initialize bottleneck detector
	bottleneckDetector := &BottleneckDetector{
		thresholds: map[string]float64{
			"cpu_utilization":    config.BottleneckThreshold,
			"memory_utilization": config.BottleneckThreshold,
			"disk_io_wait":       0.30, // 30% I/O wait time
			"network_utilization": 0.80, // 80% network utilization
		},
		detectionWindow: 10 * time.Minute,
		minSamples:      5,
	}
	
	return &CapacityPlanner{
		predictiveEngine:        predictiveEngine,
		costOptimizer:          costOptimizer,
		metricsProvider:        metricsProvider,
		config:                 config,
		resources:              make(map[string]*ResourceCapacity),
		demandForecasts:        make(map[string]*DemandForecast),
		capacityRecommendations: make(map[string]*CapacityRecommendation),
		ctx:                    ctx,
		cancel:                 cancel,
		demandHistory:          make(map[string][]DemandPoint),
		capacityHistory:        make(map[string][]CapacityPoint),
		bottleneckDetector:     bottleneckDetector,
	}
}

// Start starts the capacity planner background processes
func (cp *CapacityPlanner) Start() error {
	cp.mu.Lock()
	defer cp.mu.Unlock()
	
	// Start background processes
	go cp.forecastingLoop()
	go cp.capacityPlanningLoop()
	
	if cp.config.EnableBottleneckDetection {
		go cp.bottleneckDetectionLoop()
	}
	
	go cp.dataCleanupLoop()
	
	return nil
}

// Stop stops the capacity planner
func (cp *CapacityPlanner) Stop() error {
	if cp.cancel != nil {
		cp.cancel()
	}
	
	if cp.updateTicker != nil {
		cp.updateTicker.Stop()
	}
	
	return nil
}

// RegisterResource registers a resource for capacity planning
func (cp *CapacityPlanner) RegisterResource(resource *ResourceCapacity) error {
	if resource.ResourceID == "" {
		return fmt.Errorf("resource ID cannot be empty")
	}
	
	cp.mu.Lock()
	defer cp.mu.Unlock()
	
	resource.LastUpdated = time.Now()
	cp.resources[resource.ResourceID] = resource
	
	// Initialize history
	cp.demandHistory[resource.ResourceID] = make([]DemandPoint, 0)
	cp.capacityHistory[resource.ResourceID] = make([]CapacityPoint, 0)
	
	return nil
}

// UnregisterResource unregisters a resource
func (cp *CapacityPlanner) UnregisterResource(resourceID string) error {
	cp.mu.Lock()
	defer cp.mu.Unlock()
	
	delete(cp.resources, resourceID)
	delete(cp.demandForecasts, resourceID)
	delete(cp.capacityRecommendations, resourceID)
	delete(cp.demandHistory, resourceID)
	delete(cp.capacityHistory, resourceID)
	
	return nil
}

// GetCapacityForecast generates a capacity forecast for a resource
func (cp *CapacityPlanner) GetCapacityForecast(ctx context.Context, resourceID string, horizon time.Duration) (*DemandForecast, error) {
	cp.mu.RLock()
	resource, exists := cp.resources[resourceID]
	cp.mu.RUnlock()
	
	if !exists {
		return nil, fmt.Errorf("resource not found: %s", resourceID)
	}
	
	// Get demand forecast from predictive engine
	request := predictive.ForecastRequest{
		MetricName: "cpu_utilization", // primary metric for capacity planning
		Horizon:    horizon,
		Interval:   15 * time.Minute,
	}
	
	forecastResult, err := cp.predictiveEngine.GetForecast(ctx, request)
	if err != nil {
		return nil, fmt.Errorf("failed to get forecast: %w", err)
	}
	
	// Convert to demand forecast
	demandForecast := &DemandForecast{
		ResourceID:      resourceID,
		MetricName:      "cpu_utilization",
		ForecastHorizon: horizon,
		GeneratedAt:     time.Now(),
		Confidence:      forecastResult.ModelAccuracy,
		Predictions:     make([]DemandPrediction, len(forecastResult.Predictions)),
	}
	
	// Detect seasonal patterns
	demandForecast.SeasonalPattern = cp.detectSeasonalPatterns(resourceID)
	
	// Calculate trend
	demandForecast.TrendDirection, demandForecast.TrendStrength = cp.calculateTrend(resourceID)
	
	// Convert predictions
	for i, prediction := range forecastResult.Predictions {
		demandForecast.Predictions[i] = DemandPrediction{
			Timestamp:       prediction.Timestamp,
			PredictedDemand: prediction.Value,
			Confidence:      prediction.Confidence,
			Lower:           prediction.Lower,
			Upper:           prediction.Upper,
			BaselineDemand:  prediction.Value,
			SeasonalFactor:  1.0, // Will be calculated if seasonal pattern exists
			TrendFactor:     1.0, // Will be calculated based on trend
		}
	}
	
	// Apply seasonal adjustments if pattern detected
	if demandForecast.SeasonalPattern != nil {
		cp.applySeasonalAdjustments(demandForecast)
	}
	
	// Cache the forecast
	cp.mu.Lock()
	cp.demandForecasts[resourceID] = demandForecast
	cp.mu.Unlock()
	
	return demandForecast, nil
}

// GetCapacityRecommendation generates capacity recommendations for a resource
func (cp *CapacityPlanner) GetCapacityRecommendation(ctx context.Context, resourceID string) (*CapacityRecommendation, error) {
	cp.mu.RLock()
	resource, exists := cp.resources[resourceID]
	cp.mu.RUnlock()
	
	if !exists {
		return nil, fmt.Errorf("resource not found: %s", resourceID)
	}
	
	// Get demand forecast
	forecast, err := cp.GetCapacityForecast(ctx, resourceID, cp.config.MediumTermHorizon)
	if err != nil {
		return nil, fmt.Errorf("failed to get capacity forecast: %w", err)
	}
	
	// Calculate peak demand
	peakDemand := cp.calculatePeakDemand(forecast)
	
	// Generate recommendation
	recommendation := &CapacityRecommendation{
		ResourceID:         resourceID,
		CurrentCapacity:    resource.TotalCapacity,
		CurrentUtilization: resource.UtilizationPercent,
	}
	
	// Determine recommendation type
	if peakDemand > cp.config.MaxUtilization {
		recommendation.RecommendationType = "scale_up"
		recommendation.Priority = cp.calculatePriority(peakDemand, cp.config.MaxUtilization)
	} else if peakDemand < cp.config.TargetUtilization-cp.config.BufferPercent {
		recommendation.RecommendationType = "scale_down"
		recommendation.Priority = "low"
	} else {
		recommendation.RecommendationType = "no_change"
		recommendation.Priority = "low"
	}
	
	// Calculate recommended capacity
	err = cp.calculateRecommendedCapacity(recommendation, resource, peakDemand)
	if err != nil {
		return nil, fmt.Errorf("failed to calculate recommended capacity: %w", err)
	}
	
	// Perform risk assessment
	recommendation.RiskAssessment = cp.assessRisk(resource, recommendation, forecast)
	
	// Calculate cost impact if cost optimization is enabled
	if cp.config.CostOptimizationEnabled {
		recommendation.CostImpact, err = cp.calculateCostImpact(resource, recommendation)
		if err != nil {
			// Log error but don't fail the recommendation
			fmt.Printf("Warning: failed to calculate cost impact: %v\n", err)
		}
	}
	
	// Generate alternatives
	recommendation.Alternatives = cp.generateAlternatives(resource, recommendation)
	
	// Set timeline
	recommendation.ImplementBy = time.Now().Add(cp.getImplementationTimeline(recommendation.Priority))
	recommendation.EstimatedDuration = cp.getEstimatedDuration(recommendation.RecommendationType)
	
	// Cache the recommendation
	cp.mu.Lock()
	cp.capacityRecommendations[resourceID] = recommendation
	cp.mu.Unlock()
	
	return recommendation, nil
}

// GetBottlenecks detects current bottlenecks in resources
func (cp *CapacityPlanner) GetBottlenecks(ctx context.Context) (map[string]*BottleneckAnalysis, error) {
	if !cp.config.EnableBottleneckDetection {
		return make(map[string]*BottleneckAnalysis), nil
	}
	
	bottlenecks := make(map[string]*BottleneckAnalysis)
	
	cp.mu.RLock()
	resources := make([]*ResourceCapacity, 0, len(cp.resources))
	for _, resource := range cp.resources {
		resources = append(resources, resource)
	}
	cp.mu.RUnlock()
	
	for _, resource := range resources {
		bottleneck := cp.bottleneckDetector.DetectBottleneck(ctx, resource, cp.metricsProvider)
		if bottleneck != nil {
			bottlenecks[resource.ResourceID] = bottleneck
		}
	}
	
	return bottlenecks, nil
}

// BottleneckAnalysis represents bottleneck analysis results
type BottleneckAnalysis struct {
	ResourceID        string             `json:"resource_id"`
	BottleneckType    string             `json:"bottleneck_type"`    // cpu, memory, storage, network
	Severity          string             `json:"severity"`           // low, medium, high, critical
	CurrentUtilization float64           `json:"current_utilization"`
	Threshold         float64            `json:"threshold"`
	Duration          time.Duration      `json:"duration"`           // how long bottleneck has persisted
	
	// Impact analysis
	PerformanceImpact string             `json:"performance_impact"`
	AffectedMetrics   []string           `json:"affected_metrics"`
	
	// Recommendations
	ImmediateActions  []string           `json:"immediate_actions"`
	LongTermSolutions []string           `json:"long_term_solutions"`
}

// DetectBottleneck detects bottlenecks for a resource
func (bd *BottleneckDetector) DetectBottleneck(ctx context.Context, resource *ResourceCapacity, 
	metricsProvider monitoring.MetricsProvider) *BottleneckAnalysis {
	
	// Check each metric for bottlenecks
	for metricName, threshold := range bd.thresholds {
		// Get recent metric values
		endTime := time.Now()
		startTime := endTime.Add(-bd.detectionWindow)
		
		metricHistory, err := metricsProvider.GetMetricHistory(
			ctx,
			monitoring.MetricType(metricName),
			resource.ResourceID,
			startTime,
			endTime,
		)
		if err != nil {
			continue
		}
		
		if len(metricHistory) < bd.minSamples {
			continue
		}
		
		// Calculate average utilization over the window
		totalUtilization := 0.0
		count := 0
		for _, value := range metricHistory {
			totalUtilization += value
			count++
		}
		
		if count == 0 {
			continue
		}
		
		avgUtilization := totalUtilization / float64(count)
		
		// Check if this is a bottleneck
		if avgUtilization >= threshold {
			severity := bd.calculateSeverity(avgUtilization, threshold)
			
			analysis := &BottleneckAnalysis{
				ResourceID:         resource.ResourceID,
				BottleneckType:     metricName,
				Severity:           severity,
				CurrentUtilization: avgUtilization,
				Threshold:          threshold,
				Duration:           bd.detectionWindow,
				PerformanceImpact:  bd.getPerformanceImpact(metricName, avgUtilization),
				AffectedMetrics:    bd.getAffectedMetrics(metricName),
				ImmediateActions:   bd.getImmediateActions(metricName, severity),
				LongTermSolutions:  bd.getLongTermSolutions(metricName, resource),
			}
			
			return analysis
		}
	}
	
	return nil
}

// Helper methods for capacity planner

// detectSeasonalPatterns detects seasonal patterns in demand
func (cp *CapacityPlanner) detectSeasonalPatterns(resourceID string) *SeasonalPattern {
	cp.mu.RLock()
	history, exists := cp.demandHistory[resourceID]
	cp.mu.RUnlock()
	
	if !exists || len(history) < 168 { // Need at least 1 week of data
		return nil
	}
	
	// Simple seasonal pattern detection (would use more sophisticated algorithms in production)
	hourlyAverages := make(map[int]float64) // hour of day -> average demand
	hourlyCounts := make(map[int]int)
	
	for _, point := range history {
		hour := point.Timestamp.Hour()
		hourlyAverages[hour] += point.Value
		hourlyCounts[hour]++
	}
	
	// Calculate averages
	for hour := 0; hour < 24; hour++ {
		if count := hourlyCounts[hour]; count > 0 {
			hourlyAverages[hour] /= float64(count)
		}
	}
	
	// Find peaks and valleys
	peaks := make([]SeasonalPeak, 0)
	valleys := make([]SeasonalValley, 0)
	
	// Calculate baseline (daily average)
	baseline := 0.0
	for _, avg := range hourlyAverages {
		baseline += avg
	}
	baseline /= 24.0
	
	// Detect peaks and valleys
	for hour := 0; hour < 24; hour++ {
		avg := hourlyAverages[hour]
		ratio := avg / baseline
		
		if ratio > 1.2 { // 20% above baseline
			peak := SeasonalPeak{
				TimeOffset: hour,
				Magnitude:  ratio,
				Duration:   1, // simplified to 1 hour
			}
			peaks = append(peaks, peak)
		} else if ratio < 0.8 { // 20% below baseline
			valley := SeasonalValley{
				TimeOffset: hour,
				Magnitude:  ratio,
				Duration:   1, // simplified to 1 hour
			}
			valleys = append(valleys, valley)
		}
	}
	
	if len(peaks) > 0 || len(valleys) > 0 {
		return &SeasonalPattern{
			PeriodType:   "daily",
			PeriodLength: 24,
			Peaks:        peaks,
			Valleys:      valleys,
			Strength:     0.5, // simplified strength calculation
		}
	}
	
	return nil
}

// calculateTrend calculates trend direction and strength
func (cp *CapacityPlanner) calculateTrend(resourceID string) (string, float64) {
	cp.mu.RLock()
	history, exists := cp.demandHistory[resourceID]
	cp.mu.RUnlock()
	
	if !exists || len(history) < 10 {
		return "stable", 0.0
	}
	
	// Simple linear trend calculation
	n := len(history)
	if n < 2 {
		return "stable", 0.0
	}
	
	// Use last 10 points for trend
	startIdx := n - 10
	if startIdx < 0 {
		startIdx = 0
	}
	
	recentHistory := history[startIdx:]
	if len(recentHistory) < 2 {
		return "stable", 0.0
	}
	
	// Calculate slope
	firstValue := recentHistory[0].Value
	lastValue := recentHistory[len(recentHistory)-1].Value
	
	if math.Abs(lastValue-firstValue) < 0.05*firstValue { // Less than 5% change
		return "stable", 0.1
	}
	
	if lastValue > firstValue {
		strength := math.Min(1.0, (lastValue-firstValue)/firstValue)
		return "increasing", strength
	} else {
		strength := math.Min(1.0, (firstValue-lastValue)/firstValue)
		return "decreasing", strength
	}
}

// applySeasonalAdjustments applies seasonal pattern adjustments to forecast
func (cp *CapacityPlanner) applySeasonalAdjustments(forecast *DemandForecast) {
	if forecast.SeasonalPattern == nil {
		return
	}
	
	pattern := forecast.SeasonalPattern
	
	for i := range forecast.Predictions {
		prediction := &forecast.Predictions[i]
		hour := prediction.Timestamp.Hour()
		
		// Find applicable peak or valley
		seasonalFactor := 1.0
		
		for _, peak := range pattern.Peaks {
			if hour >= peak.TimeOffset && hour < peak.TimeOffset+peak.Duration {
				seasonalFactor = peak.Magnitude
				break
			}
		}
		
		if seasonalFactor == 1.0 { // No peak found, check valleys
			for _, valley := range pattern.Valleys {
				if hour >= valley.TimeOffset && hour < valley.TimeOffset+valley.Duration {
					seasonalFactor = valley.Magnitude
					break
				}
			}
		}
		
		// Apply seasonal adjustment
		prediction.SeasonalFactor = seasonalFactor
		prediction.PredictedDemand = prediction.BaselineDemand * seasonalFactor
		prediction.Upper = prediction.Upper * seasonalFactor
		prediction.Lower = prediction.Lower * seasonalFactor
	}
}

// calculatePeakDemand calculates peak demand from forecast
func (cp *CapacityPlanner) calculatePeakDemand(forecast *DemandForecast) float64 {
	if len(forecast.Predictions) == 0 {
		return 0.0
	}
	
	maxDemand := 0.0
	for _, prediction := range forecast.Predictions {
		if prediction.PredictedDemand > maxDemand {
			maxDemand = prediction.PredictedDemand
		}
	}
	
	// Add buffer based on confidence
	buffer := (1.0 - forecast.Confidence) * cp.config.BufferPercent
	return maxDemand * (1.0 + buffer)
}

// calculatePriority calculates priority level based on demand and thresholds
func (cp *CapacityPlanner) calculatePriority(demand, threshold float64) string {
	if demand > threshold*1.2 {
		return "critical"
	} else if demand > threshold*1.1 {
		return "high"
	} else if demand > threshold {
		return "medium"
	} else {
		return "low"
	}
}

// calculateRecommendedCapacity calculates the recommended capacity
func (cp *CapacityPlanner) calculateRecommendedCapacity(
	recommendation *CapacityRecommendation,
	resource *ResourceCapacity,
	peakDemand float64,
) error {
	
	currentCapacity := resource.TotalCapacity
	
	switch recommendation.RecommendationType {
	case "scale_up":
		// Calculate required capacity to handle peak demand with target utilization
		requiredCapacity := peakDemand / cp.config.TargetUtilization
		
		// Calculate new instance count
		currentPerInstanceCapacity := currentCapacity.CPU / float64(resource.CurrentInstances)
		requiredInstances := int(math.Ceil(requiredCapacity / currentPerInstanceCapacity))
		
		// Ensure within limits
		if requiredInstances > resource.MaxInstances {
			requiredInstances = resource.MaxInstances
		}
		
		recommendation.RecommendedInstances = requiredInstances
		recommendation.RecommendedCapacity = ResourceMetrics{
			CPU:     currentPerInstanceCapacity * float64(requiredInstances),
			Memory:  (currentCapacity.Memory / float64(resource.CurrentInstances)) * float64(requiredInstances),
			Storage: (currentCapacity.Storage / float64(resource.CurrentInstances)) * float64(requiredInstances),
			Network: (currentCapacity.Network / float64(resource.CurrentInstances)) * float64(requiredInstances),
		}
		
		recommendation.Reason = fmt.Sprintf("Peak demand %.1f%% exceeds threshold %.1f%%, scaling from %d to %d instances", 
			peakDemand*100, cp.config.MaxUtilization*100, resource.CurrentInstances, requiredInstances)
		
	case "scale_down":
		// Calculate optimal capacity for current demand
		optimalCapacity := peakDemand / cp.config.TargetUtilization
		
		currentPerInstanceCapacity := currentCapacity.CPU / float64(resource.CurrentInstances)
		optimalInstances := int(math.Ceil(optimalCapacity / currentPerInstanceCapacity))
		
		// Ensure within limits
		if optimalInstances < resource.MinInstances {
			optimalInstances = resource.MinInstances
		}
		
		recommendation.RecommendedInstances = optimalInstances
		recommendation.RecommendedCapacity = ResourceMetrics{
			CPU:     currentPerInstanceCapacity * float64(optimalInstances),
			Memory:  (currentCapacity.Memory / float64(resource.CurrentInstances)) * float64(optimalInstances),
			Storage: (currentCapacity.Storage / float64(resource.CurrentInstances)) * float64(optimalInstances),
			Network: (currentCapacity.Network / float64(resource.CurrentInstances)) * float64(optimalInstances),
		}
		
		recommendation.Reason = fmt.Sprintf("Peak demand %.1f%% allows scaling down from %d to %d instances while maintaining target utilization", 
			peakDemand*100, resource.CurrentInstances, optimalInstances)
		
	case "no_change":
		recommendation.RecommendedInstances = resource.CurrentInstances
		recommendation.RecommendedCapacity = currentCapacity
		recommendation.Reason = fmt.Sprintf("Current capacity is optimal for projected demand %.1f%%", peakDemand*100)
	}
	
	recommendation.RecommendedConfig = resource.InstanceConfig
	recommendation.Confidence = 0.8 // Base confidence
	
	return nil
}

// Additional helper methods would continue here...
// (assessRisk, calculateCostImpact, generateAlternatives, etc.)

// forecastingLoop runs the background forecasting process
func (cp *CapacityPlanner) forecastingLoop() {
	ticker := time.NewTicker(cp.config.ForecastInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			cp.updateForecasts()
		case <-cp.ctx.Done():
			return
		}
	}
}

// capacityPlanningLoop runs the background capacity planning process
func (cp *CapacityPlanner) capacityPlanningLoop() {
	ticker := time.NewTicker(cp.config.PlanningInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			cp.generateCapacityRecommendations()
		case <-cp.ctx.Done():
			return
		}
	}
}

// bottleneckDetectionLoop runs the background bottleneck detection process
func (cp *CapacityPlanner) bottleneckDetectionLoop() {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			cp.detectBottlenecks()
		case <-cp.ctx.Done():
			return
		}
	}
}

// dataCleanupLoop runs the background data cleanup process
func (cp *CapacityPlanner) dataCleanupLoop() {
	ticker := time.NewTicker(1 * time.Hour)
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			cp.cleanupHistoricalData()
		case <-cp.ctx.Done():
			return
		}
	}
}

// updateForecasts updates demand forecasts for all resources
func (cp *CapacityPlanner) updateForecasts() {
	cp.mu.RLock()
	resourceIDs := make([]string, 0, len(cp.resources))
	for resourceID := range cp.resources {
		resourceIDs = append(resourceIDs, resourceID)
	}
	cp.mu.RUnlock()
	
	for _, resourceID := range resourceIDs {
		_, err := cp.GetCapacityForecast(context.Background(), resourceID, cp.config.MediumTermHorizon)
		if err != nil {
			fmt.Printf("Warning: failed to update forecast for resource %s: %v\n", resourceID, err)
		}
	}
}

// generateCapacityRecommendations generates capacity recommendations for all resources
func (cp *CapacityPlanner) generateCapacityRecommendations() {
	cp.mu.RLock()
	resourceIDs := make([]string, 0, len(cp.resources))
	for resourceID := range cp.resources {
		resourceIDs = append(resourceIDs, resourceID)
	}
	cp.mu.RUnlock()
	
	for _, resourceID := range resourceIDs {
		_, err := cp.GetCapacityRecommendation(context.Background(), resourceID)
		if err != nil {
			fmt.Printf("Warning: failed to generate capacity recommendation for resource %s: %v\n", resourceID, err)
		}
	}
}

// detectBottlenecks detects bottlenecks for all resources
func (cp *CapacityPlanner) detectBottlenecks() {
	_, err := cp.GetBottlenecks(context.Background())
	if err != nil {
		fmt.Printf("Warning: failed to detect bottlenecks: %v\n", err)
	}
}

// cleanupHistoricalData cleans up old historical data
func (cp *CapacityPlanner) cleanupHistoricalData() {
	cp.mu.Lock()
	defer cp.mu.Unlock()
	
	demandCutoff := time.Now().Add(-cp.config.DemandHistoryRetention)
	capacityCutoff := time.Now().Add(-cp.config.CapacityHistoryRetention)
	
	// Clean demand history
	for resourceID, history := range cp.demandHistory {
		filtered := make([]DemandPoint, 0)
		for _, point := range history {
			if point.Timestamp.After(demandCutoff) {
				filtered = append(filtered, point)
			}
		}
		cp.demandHistory[resourceID] = filtered
	}
	
	// Clean capacity history
	for resourceID, history := range cp.capacityHistory {
		filtered := make([]CapacityPoint, 0)
		for _, point := range history {
			if point.Timestamp.After(capacityCutoff) {
				filtered = append(filtered, point)
			}
		}
		cp.capacityHistory[resourceID] = filtered
	}
}

// Placeholder implementations for additional helper methods

func (cp *CapacityPlanner) assessRisk(resource *ResourceCapacity, recommendation *CapacityRecommendation, forecast *DemandForecast) *RiskAssessment {
	// Simplified risk assessment
	return &RiskAssessment{
		OverallRisk:         "medium",
		RiskScore:          0.4,
		RiskFactors:        []string{"Demand uncertainty", "Scaling complexity"},
		MitigationStrategies: []string{"Gradual scaling", "Monitor closely"},
		PerformanceRisk:    0.3,
		CostRisk:          0.2,
		AvailabilityRisk:  0.1,
		SecurityRisk:      0.1,
	}
}

func (cp *CapacityPlanner) calculateCostImpact(resource *ResourceCapacity, recommendation *CapacityRecommendation) (*CostImpactAnalysis, error) {
	// Simplified cost impact calculation
	return &CostImpactAnalysis{
		CurrentMonthlyCost:   1000.0,
		ProjectedMonthlyCost: 1200.0,
		CostDifference:       200.0,
		CostChangePercent:    20.0,
	}, nil
}

func (cp *CapacityPlanner) generateAlternatives(resource *ResourceCapacity, recommendation *CapacityRecommendation) []AlternativeOption {
	// Simplified alternatives generation
	return []AlternativeOption{
		{
			Description:         "Use larger instance types instead of more instances",
			EstimatedCost:       1100.0,
			ExpectedPerformance: 0.9,
			RiskLevel:          "medium",
			Pros:               []string{"Better performance", "Simpler management"},
			Cons:               []string{"Higher cost", "Less redundancy"},
		},
	}
}

func (cp *CapacityPlanner) getImplementationTimeline(priority string) time.Duration {
	switch priority {
	case "critical":
		return 30 * time.Minute
	case "high":
		return 2 * time.Hour
	case "medium":
		return 8 * time.Hour
	default:
		return 24 * time.Hour
	}
}

func (cp *CapacityPlanner) getEstimatedDuration(recommendationType string) time.Duration {
	switch recommendationType {
	case "scale_up", "scale_out":
		return 10 * time.Minute
	case "scale_down", "scale_in":
		return 5 * time.Minute
	default:
		return 0
	}
}

// Bottleneck detector helper methods

func (bd *BottleneckDetector) calculateSeverity(utilization, threshold float64) string {
	ratio := utilization / threshold
	if ratio > 1.5 {
		return "critical"
	} else if ratio > 1.3 {
		return "high"
	} else if ratio > 1.1 {
		return "medium"
	} else {
		return "low"
	}
}

func (bd *BottleneckDetector) getPerformanceImpact(metricName string, utilization float64) string {
	switch metricName {
	case "cpu_utilization":
		if utilization > 0.95 {
			return "Severe performance degradation, response times significantly increased"
		} else if utilization > 0.90 {
			return "Moderate performance impact, some requests may be delayed"
		} else {
			return "Minor performance impact"
		}
	case "memory_utilization":
		if utilization > 0.95 {
			return "Memory exhaustion risk, potential system instability"
		} else if utilization > 0.90 {
			return "High memory pressure, increased garbage collection"
		} else {
			return "Elevated memory usage"
		}
	default:
		return "Performance impact detected"
	}
}

func (bd *BottleneckDetector) getAffectedMetrics(metricName string) []string {
	switch metricName {
	case "cpu_utilization":
		return []string{"response_time", "throughput", "queue_length"}
	case "memory_utilization":
		return []string{"gc_time", "allocation_rate", "response_time"}
	case "disk_io_wait":
		return []string{"disk_latency", "throughput", "response_time"}
	case "network_utilization":
		return []string{"network_latency", "packet_loss", "throughput"}
	default:
		return []string{"response_time", "throughput"}
	}
}

func (bd *BottleneckDetector) getImmediateActions(metricName, severity string) []string {
	actions := []string{}
	
	switch metricName {
	case "cpu_utilization":
		actions = append(actions, "Scale out instances", "Optimize CPU-intensive operations")
		if severity == "critical" {
			actions = append(actions, "Enable request throttling", "Reduce worker threads")
		}
	case "memory_utilization":
		actions = append(actions, "Scale up instance memory", "Clear unnecessary caches")
		if severity == "critical" {
			actions = append(actions, "Force garbage collection", "Restart instances")
		}
	case "disk_io_wait":
		actions = append(actions, "Optimize disk I/O operations", "Use faster storage")
	case "network_utilization":
		actions = append(actions, "Optimize network usage", "Enable compression")
	}
	
	return actions
}

func (bd *BottleneckDetector) getLongTermSolutions(metricName string, resource *ResourceCapacity) []string {
	solutions := []string{}
	
	switch metricName {
	case "cpu_utilization":
		solutions = append(solutions, 
			"Implement horizontal auto-scaling",
			"Optimize application algorithms",
			"Consider using faster CPU instance types")
	case "memory_utilization":
		solutions = append(solutions,
			"Implement memory-based auto-scaling", 
			"Optimize memory usage in application",
			"Consider memory-optimized instance types")
	case "disk_io_wait":
		solutions = append(solutions,
			"Migrate to SSD storage",
			"Implement database query optimization",
			"Consider storage-optimized instances")
	case "network_utilization":
		solutions = append(solutions,
			"Implement content delivery network",
			"Optimize data transfer protocols",
			"Consider network-optimized instances")
	}
	
	return solutions
}