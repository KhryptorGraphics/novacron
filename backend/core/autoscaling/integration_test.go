package autoscaling

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/autoscaling/predictive"
	"github.com/khryptorgraphics/novacron/backend/core/autoscaling/cost"
	"github.com/khryptorgraphics/novacron/backend/core/autoscaling/forecasting"
	"github.com/khryptorgraphics/novacron/backend/core/monitoring"
	"github.com/khryptorgraphics/novacron/backend/core/vm"
)

// TestEnhancedAutoScalingIntegration tests the complete enhanced autoscaling system
func TestEnhancedAutoScalingIntegration(t *testing.T) {
	// Create mock components
	metricsProvider := &MockMetricsProvider{}
	resourceController := &MockResourceController{}
	vmManager := &MockVMManager{}
	
	// Configure enhanced autoscaling
	config := EnhancedAutoScalingConfig{
		PredictiveScaling: predictive.PredictiveEngineConfig{
			DefaultModels: map[string]predictive.ModelConfig{
				"cpu_utilization": {
					Type:              predictive.ModelTypeARIMA,
					Parameters:        map[string]interface{}{"p": 1, "d": 1, "q": 1},
					TrainingWindow:    24 * time.Hour,
					PredictionHorizon: 4 * time.Hour,
					MinDataPoints:     10,
					MaxDataPoints:     1000,
					UpdateFrequency:   15 * time.Minute,
				},
			},
			ModelUpdateInterval:    1 * time.Hour,
			ForecastUpdateInterval: 15 * time.Minute,
			ForecastCacheTTL:       30 * time.Minute,
			AutoModelSelection:     true,
			MinAccuracyThreshold:   0.6,
		},
		CostOptimization: cost.CostOptimizerConfig{
			CostWeight:        0.4,
			PerformanceWeight: 0.4,
			RiskWeight:        0.2,
			MinCostSavings:    0.10,
			MinROIThreshold:   0.05,
			MaxRiskScore:      0.3,
			UseSpotInstances:  true,
			MaxSpotRisk:       0.2,
		},
		CapacityPlanning: forecasting.CapacityPlannerConfig{
			ShortTermHorizon:     4 * time.Hour,
			MediumTermHorizon:    24 * time.Hour,
			LongTermHorizon:      7 * 24 * time.Hour,
			TargetUtilization:    0.70,
			MaxUtilization:       0.85,
			BufferPercent:        0.20,
			EnableBottleneckDetection: true,
			BottleneckThreshold:  0.90,
		},
		EnableMLPolicies:   true,
		MLModelUpdateInterval: 1 * time.Hour,
		EnableAnalytics:    true,
		AnalyticsRetention: 7 * 24 * time.Hour,
		VMManagerEnabled:   true,
		MetricsCollection:  true,
		MultiObjectiveOptimization: true,
		RiskAwareScaling:   true,
		BudgetConstraints: []cost.BudgetConstraint{
			{
				Name:           "dev-budget",
				MaxHourlyCost:  100.0,
				MaxDailyCost:   2000.0,
				MaxMonthlyCost: 50000.0,
				TimeWindow:     "monthly",
				WarningThreshold: 0.8,
				AlertThreshold:   0.95,
			},
		},
	}
	
	// Create enhanced autoscaling manager
	enhanced, err := NewEnhancedAutoScalingManager(metricsProvider, resourceController, vmManager, config)
	if err != nil {
		t.Fatalf("Failed to create enhanced autoscaling manager: %v", err)
	}
	
	// Start the manager
	err = enhanced.Start()
	if err != nil {
		t.Fatalf("Failed to start enhanced autoscaling manager: %v", err)
	}
	defer enhanced.Stop()
	
	// Test scenario: Register a scaling group
	group := &ScalingGroup{
		ID:              "test-web-tier",
		Name:            "Test Web Tier",
		ResourceType:    ResourceVM,
		ResourceIDs:     []string{"web-1", "web-2"},
		ScalingMode:     ScalingModeHorizontal,
		MinCapacity:     2,
		MaxCapacity:     10,
		DesiredCapacity: 2,
		CurrentCapacity: 2,
		Rules: []*ScalingRule{
			{
				ID:                "cpu-rule",
				Name:              "CPU Utilization Rule",
				MetricType:        MetricCPUUtilization,
				ScaleOutThreshold: 75.0,
				ScaleInThreshold:  25.0,
				ScaleOutIncrement: 1,
				ScaleInDecrement:  1,
				CooldownPeriod:    5 * time.Minute,
				EvaluationPeriods: 2,
				Enabled:           true,
			},
		},
		LaunchTemplate: map[string]interface{}{
			"instance_type": "t3.medium",
			"image_id":      "ami-12345",
		},
	}
	
	err = enhanced.RegisterScalingGroup(group)
	if err != nil {
		t.Fatalf("Failed to register scaling group: %v", err)
	}
	
	// Test scenario: Register an ML policy
	mlPolicy := &MLScalingPolicy{
		ID:                  "ml-policy-1",
		Name:                "ML-based CPU Scaling",
		ScalingGroupID:      "test-web-tier",
		ModelType:           predictive.ModelTypeARIMA,
		ModelConfig: predictive.ModelConfig{
			Type:              predictive.ModelTypeARIMA,
			Parameters:        map[string]interface{}{"p": 1, "d": 1, "q": 1},
			TrainingWindow:    24 * time.Hour,
			PredictionHorizon: 2 * time.Hour,
			MinDataPoints:     20,
			UpdateFrequency:   30 * time.Minute,
		},
		PredictionHorizon:   2 * time.Hour,
		ConfidenceThreshold: 0.7,
		ScaleUpThreshold:    80.0,
		ScaleDownThreshold:  30.0,
		CostWeight:          0.4,
		PerformanceWeight:   0.4,
		RiskWeight:          0.2,
		SeasonalAdjustment:  true,
		TrendAdjustment:     true,
		Enabled:             true,
	}
	
	err = enhanced.RegisterMLPolicy(mlPolicy)
	if err != nil {
		t.Fatalf("Failed to register ML policy: %v", err)
	}
	
	// Test scenario: Simulate high CPU usage
	metricsProvider.SetMetricValue("cpu_utilization", "test-web-tier", 85.0)
	
	// Wait for scaling evaluation
	time.Sleep(2 * time.Second)
	
	// Test scenario: Evaluate enhanced scaling
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	
	err = enhanced.EvaluateEnhancedScaling(ctx)
	if err != nil {
		t.Fatalf("Enhanced scaling evaluation failed: %v", err)
	}
	
	// Verify scaling occurred
	updatedGroup, err := enhanced.GetScalingGroup("test-web-tier")
	if err != nil {
		t.Fatalf("Failed to get updated scaling group: %v", err)
	}
	
	if updatedGroup.CurrentCapacity <= 2 {
		t.Errorf("Expected scaling to occur, but capacity remained at %d", updatedGroup.CurrentCapacity)
	}
	
	// Test scenario: Get analytics
	analytics := enhanced.GetScalingAnalytics()
	if analytics.totalScalingActions == 0 {
		t.Error("Expected scaling actions to be recorded in analytics")
	}
	
	// Test scenario: Get ML policies
	policies := enhanced.GetMLPolicies()
	if len(policies) != 1 {
		t.Errorf("Expected 1 ML policy, got %d", len(policies))
	}
	
	// Test scenario: Get predictive forecasts
	forecasts, err := enhanced.GetPredictiveForecasts(ctx, 4*time.Hour)
	if err != nil {
		t.Fatalf("Failed to get predictive forecasts: %v", err)
	}
	
	if len(forecasts) == 0 {
		t.Error("Expected predictive forecasts to be available")
	}
	
	// Test scenario: Get capacity recommendations
	recommendations, err := enhanced.GetCapacityRecommendations(ctx)
	if err != nil {
		t.Fatalf("Failed to get capacity recommendations: %v", err)
	}
	
	if _, exists := recommendations["test-web-tier"]; !exists {
		t.Error("Expected capacity recommendation for test-web-tier")
	}
	
	// Test scenario: Get cost optimization report
	costReport, err := enhanced.GetCostOptimizationReport(ctx)
	if err != nil {
		t.Fatalf("Failed to get cost optimization report: %v", err)
	}
	
	if costReport.TotalCurrentCost <= 0 {
		t.Error("Expected positive total current cost in report")
	}
	
	t.Logf("Integration test completed successfully")
	t.Logf("Scaling actions: %d", analytics.totalScalingActions)
	t.Logf("Current capacity: %d", updatedGroup.CurrentCapacity)
	t.Logf("Cost report: $%.2f current cost", costReport.TotalCurrentCost)
}

// TestMLPolicyEvaluation tests ML policy evaluation specifically
func TestMLPolicyEvaluation(t *testing.T) {
	metricsProvider := &MockMetricsProvider{}
	resourceController := &MockResourceController{}
	vmManager := &MockVMManager{}
	
	config := EnhancedAutoScalingConfig{
		EnableMLPolicies: true,
		PredictiveScaling: predictive.PredictiveEngineConfig{
			MinAccuracyThreshold: 0.6,
		},
		CostOptimization: cost.CostOptimizerConfig{
			CostOptimizationEnabled: true,
		},
		MultiObjectiveOptimization: true,
	}
	
	enhanced, err := NewEnhancedAutoScalingManager(metricsProvider, resourceController, vmManager, config)
	if err != nil {
		t.Fatalf("Failed to create enhanced autoscaling manager: %v", err)
	}
	
	err = enhanced.Start()
	if err != nil {
		t.Fatalf("Failed to start enhanced autoscaling manager: %v", err)
	}
	defer enhanced.Stop()
	
	// Create a test group
	group := &ScalingGroup{
		ID:              "ml-test-group",
		MinCapacity:     1,
		MaxCapacity:     5,
		CurrentCapacity: 2,
	}
	
	err = enhanced.RegisterScalingGroup(group)
	if err != nil {
		t.Fatalf("Failed to register scaling group: %v", err)
	}
	
	// Create ML policy with different weight configurations
	testCases := []struct {
		name           string
		costWeight     float64
		perfWeight     float64
		riskWeight     float64
		expectedAction ScalingAction
	}{
		{"Cost-focused", 0.7, 0.2, 0.1, ScalingActionScaleOut},
		{"Performance-focused", 0.2, 0.7, 0.1, ScalingActionScaleOut},
		{"Risk-averse", 0.2, 0.2, 0.6, ScalingActionNone},
		{"Balanced", 0.33, 0.33, 0.34, ScalingActionScaleOut},
	}
	
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			policy := &MLScalingPolicy{
				ID:                  "test-policy-" + tc.name,
				ScalingGroupID:      "ml-test-group",
				ModelType:           predictive.ModelTypeLinear,
				PredictionHorizon:   1 * time.Hour,
				ConfidenceThreshold: 0.8,
				ScaleUpThreshold:    70.0,
				ScaleDownThreshold:  30.0,
				CostWeight:          tc.costWeight,
				PerformanceWeight:   tc.perfWeight,
				RiskWeight:          tc.riskWeight,
				Enabled:             true,
			}
			
			err := enhanced.RegisterMLPolicy(policy)
			if err != nil {
				t.Fatalf("Failed to register ML policy: %v", err)
			}
			
			// Simulate high demand scenario
			metricsProvider.SetMetricValue("cpu_utilization", "ml-test-group", 80.0)
			
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			
			err = enhanced.EvaluateEnhancedScaling(ctx)
			if err != nil {
				t.Errorf("ML policy evaluation failed: %v", err)
			}
			
			// Clean up
			enhanced.UnregisterMLPolicy(policy.ID)
		})
	}
}

// TestCostOptimizationIntegration tests cost-aware scaling decisions
func TestCostOptimizationIntegration(t *testing.T) {
	metricsProvider := &MockMetricsProvider{}
	resourceController := &MockResourceController{}
	vmManager := &MockVMManager{}
	
	config := EnhancedAutoScalingConfig{
		CostOptimization: cost.CostOptimizerConfig{
			CostOptimizationEnabled: true,
			CostWeight:              0.6,
			PerformanceWeight:       0.3,
			RiskWeight:              0.1,
			MinCostSavings:          0.05,
			UseSpotInstances:        true,
			MaxSpotRisk:             0.2,
		},
		BudgetConstraints: []cost.BudgetConstraint{
			{
				Name:             "test-budget",
				MaxHourlyCost:    50.0,
				WarningThreshold: 0.8,
				AlertThreshold:   0.9,
			},
		},
	}
	
	enhanced, err := NewEnhancedAutoScalingManager(metricsProvider, resourceController, vmManager, config)
	if err != nil {
		t.Fatalf("Failed to create enhanced autoscaling manager: %v", err)
	}
	
	err = enhanced.Start()
	if err != nil {
		t.Fatalf("Failed to start enhanced autoscaling manager: %v", err)
	}
	defer enhanced.Stop()
	
	// Register AWS cost model
	awsModel := cost.NewAWSCostModel("us-east-1")
	err = enhanced.costOptimizer.RegisterCostModel("aws", awsModel)
	if err != nil {
		t.Fatalf("Failed to register AWS cost model: %v", err)
	}
	
	// Test cost-optimized scaling
	group := &ScalingGroup{
		ID:              "cost-test-group",
		MinCapacity:     1,
		MaxCapacity:     10,
		CurrentCapacity: 3,
	}
	
	err = enhanced.RegisterScalingGroup(group)
	if err != nil {
		t.Fatalf("Failed to register scaling group: %v", err)
	}
	
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	
	// Test cost optimization request
	costRequest := cost.ScalingOptimizationRequest{
		Provider:         "aws",
		ResourceID:       "cost-test-group",
		CurrentInstances: 3,
		ScalingTrigger:   "high_cpu",
		MetricValue:      85.0,
		MetricThreshold:  80.0,
		MinInstances:     1,
		MaxInstances:     10,
	}
	
	decision, err := enhanced.costOptimizer.OptimizeScaling(ctx, costRequest)
	if err != nil {
		t.Fatalf("Cost optimization failed: %v", err)
	}
	
	if decision.RecommendedAction == "no_action" {
		t.Error("Expected scaling recommendation from cost optimizer")
	}
	
	// Verify budget compliance
	if !decision.BudgetCompliant {
		t.Error("Expected budget compliance for reasonable scaling")
	}
	
	t.Logf("Cost optimization decision: %s", decision.RecommendedAction)
	t.Logf("Projected cost: $%.2f", decision.ProjectedCost)
	t.Logf("Cost savings: $%.2f", decision.CostSavings)
	t.Logf("ROI: %.2f%%", decision.ROI*100)
}

// TestCapacityPlanningIntegration tests capacity planning functionality
func TestCapacityPlanningIntegration(t *testing.T) {
	metricsProvider := &MockMetricsProvider{}
	resourceController := &MockResourceController{}
	vmManager := &MockVMManager{}
	
	config := EnhancedAutoScalingConfig{
		CapacityPlanning: forecasting.CapacityPlannerConfig{
			TargetUtilization:         0.70,
			MaxUtilization:           0.85,
			BufferPercent:            0.15,
			EnableBottleneckDetection: true,
			BottleneckThreshold:      0.90,
		},
	}
	
	enhanced, err := NewEnhancedAutoScalingManager(metricsProvider, resourceController, vmManager, config)
	if err != nil {
		t.Fatalf("Failed to create enhanced autoscaling manager: %v", err)
	}
	
	err = enhanced.Start()
	if err != nil {
		t.Fatalf("Failed to start enhanced autoscaling manager: %v", err)
	}
	defer enhanced.Stop()
	
	// Register resource for capacity planning
	resourceCapacity := &forecasting.ResourceCapacity{
		ResourceID:      "capacity-test-resource",
		ResourceType:    "vm",
		TotalCapacity: forecasting.ResourceMetrics{
			CPU:    8.0,
			Memory: 16.0,
			Storage: 100.0,
		},
		UsedCapacity: forecasting.ResourceMetrics{
			CPU:    6.0,
			Memory: 12.0,
			Storage: 75.0,
		},
		UtilizationPercent: 0.75,
		MinInstances:       1,
		MaxInstances:       10,
		CurrentInstances:   4,
	}
	
	err = enhanced.capacityPlanner.RegisterResource(resourceCapacity)
	if err != nil {
		t.Fatalf("Failed to register resource for capacity planning: %v", err)
	}
	
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	
	// Get capacity forecast
	forecast, err := enhanced.capacityPlanner.GetCapacityForecast(ctx, "capacity-test-resource", 4*time.Hour)
	if err != nil {
		t.Fatalf("Failed to get capacity forecast: %v", err)
	}
	
	if len(forecast.Predictions) == 0 {
		t.Error("Expected capacity forecast predictions")
	}
	
	// Get capacity recommendation
	recommendation, err := enhanced.capacityPlanner.GetCapacityRecommendation(ctx, "capacity-test-resource")
	if err != nil {
		t.Fatalf("Failed to get capacity recommendation: %v", err)
	}
	
	if recommendation.RecommendationType == "" {
		t.Error("Expected capacity recommendation type")
	}
	
	// Test bottleneck detection
	bottlenecks, err := enhanced.capacityPlanner.GetBottlenecks(ctx)
	if err != nil {
		t.Fatalf("Failed to get bottlenecks: %v", err)
	}
	
	t.Logf("Capacity forecast confidence: %.2f", forecast.Confidence)
	t.Logf("Recommendation: %s (priority: %s)", recommendation.RecommendationType, recommendation.Priority)
	t.Logf("Bottlenecks detected: %d", len(bottlenecks))
}

// Mock implementations for testing

type MockMetricsProvider struct {
	metrics map[string]float64
	mu      sync.RWMutex
}

func (m *MockMetricsProvider) GetMetric(ctx context.Context, metricType monitoring.MetricType, resourceID string, timeRange time.Duration) (float64, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	
	key := string(metricType) + ":" + resourceID
	if value, exists := m.metrics[key]; exists {
		return value, nil
	}
	
	return 50.0, nil // Default value
}

func (m *MockMetricsProvider) GetMetricHistory(ctx context.Context, metricType monitoring.MetricType, resourceID string, start, end time.Time) (map[time.Time]float64, error) {
	history := make(map[time.Time]float64)
	
	// Generate mock historical data
	current := start
	for current.Before(end) {
		history[current] = 50.0 + float64(current.Hour())*2.0 // Simulate daily pattern
		current = current.Add(1 * time.Hour)
	}
	
	return history, nil
}

func (m *MockMetricsProvider) RegisterCallback(ctx context.Context, metricType monitoring.MetricType, resourceID string, threshold float64, callback func(float64)) {
	// Mock implementation
}

func (m *MockMetricsProvider) SetMetricValue(metricType, resourceID string, value float64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	if m.metrics == nil {
		m.metrics = make(map[string]float64)
	}
	
	key := metricType + ":" + resourceID
	m.metrics[key] = value
}

type MockResourceController struct{}

func (m *MockResourceController) ScaleResourceGroup(ctx context.Context, groupID string, desiredCapacity int) error {
	return nil
}

func (m *MockResourceController) GetResourceGroupCapacity(ctx context.Context, groupID string) (int, error) {
	return 2, nil
}

func (m *MockResourceController) GetResourceUtilization(ctx context.Context, resourceID string, metricType MetricType) (float64, error) {
	return 50.0, nil
}

func (m *MockResourceController) CreateResource(ctx context.Context, template map[string]interface{}) (string, error) {
	return "mock-resource-id", nil
}

func (m *MockResourceController) DeleteResource(ctx context.Context, resourceID string) error {
	return nil
}

func (m *MockResourceController) ResizeResource(ctx context.Context, resourceID string, newSize map[string]interface{}) error {
	return nil
}

type MockVMManager struct{}

func (m *MockVMManager) GetVM(vmID string) (*vm.VM, error) {
	return &vm.VM{
		ID:     vmID,
		Name:   "mock-vm",
		Status: vm.StateRunning,
	}, nil
}