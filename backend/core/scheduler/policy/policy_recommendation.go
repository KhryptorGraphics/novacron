package policy

import (
	"context"
	"fmt"
	"sort"
	"time"
)

// PolicyRecommendationEngine generates recommendations for policy configurations
// based on workload patterns and historical performance data
type PolicyRecommendationEngine struct {
	// Engine is the underlying policy engine
	Engine *PolicyEngine

	// Simulator is used to simulate different policy configurations
	Simulator *PolicySimulator

	// HistoricalData stores past performance data for analysis
	HistoricalData []*PlacementPerformanceData

	// RecommendationHistory stores past recommendations
	RecommendationHistory []*PolicyRecommendation
}

// PlacementPerformanceData captures data about placement performance for analysis
type PlacementPerformanceData struct {
	// Timestamp records when this data was collected
	Timestamp time.Time

	// ActivePolicies contains the active policy configurations at the time
	ActivePolicies map[string]*PolicyConfiguration

	// VMTypeDistribution records the distribution of VM types
	VMTypeDistribution map[string]int

	// ResourceUsage records overall resource utilization
	ResourceUsage ResourceUtilizationData

	// SuccessRate records the percentage of successful placements
	SuccessRate float64

	// AvgPlacementLatency records the average time taken for placement decisions
	AvgPlacementLatency time.Duration

	// FailureReasons maps reasons for placement failures to counts
	FailureReasons map[string]int
}

// ResourceUtilizationData tracks resource utilization metrics
type ResourceUtilizationData struct {
	// CPUUtilization as a percentage across the cluster
	CPUUtilization float64

	// MemoryUtilization as a percentage across the cluster
	MemoryUtilization float64

	// StorageUtilization as a percentage across the cluster
	StorageUtilization float64

	// NetworkUtilization as a percentage of available bandwidth
	NetworkUtilization float64

	// ResourceSkew measures imbalance in resource utilization across nodes
	ResourceSkew float64
}

// PolicyRecommendation represents a recommended policy configuration
type PolicyRecommendation struct {
	// ID uniquely identifies this recommendation
	ID string

	// Timestamp when this recommendation was generated
	Timestamp time.Time

	// Reason explains why this recommendation was made
	Reason string

	// Category indicates the type of recommendation
	Category RecommendationCategory

	// CurrentPolicies contains the current policy configurations
	CurrentPolicies map[string]*PolicyConfiguration

	// RecommendedPolicies contains the recommended policy configurations
	RecommendedPolicies map[string]*PolicyConfiguration

	// ExpectedImpact describes the expected impact of adopting this recommendation
	ExpectedImpact RecommendationImpact

	// SimulationResults contains comparison results if simulations were run
	SimulationResults *SimulationComparison

	// Applied indicates whether this recommendation was applied
	Applied bool

	// AppliedAt records when this recommendation was applied
	AppliedAt time.Time
}

// RecommendationCategory indicates the type of policy recommendation
type RecommendationCategory string

const (
	// ResourceUtilizationOptimization aims to improve resource utilization
	ResourceUtilizationOptimization RecommendationCategory = "resource_utilization"

	// PlacementSuccessRateImprovement aims to increase successful placements
	PlacementSuccessRateImprovement RecommendationCategory = "placement_success_rate"

	// PerformanceImprovement aims to improve overall system performance
	PerformanceImprovement RecommendationCategory = "performance"

	// ResourceBalancing aims to balance resources across nodes
	ResourceBalancing RecommendationCategory = "resource_balancing"

	// ConflictResolution resolves conflicts between policies
	ConflictResolution RecommendationCategory = "conflict_resolution"

	// SecurityEnhancement improves security through policy adjustments
	SecurityEnhancement RecommendationCategory = "security_enhancement"

	// FailureReduction aims to reduce placement failures
	FailureReduction RecommendationCategory = "failure_reduction"
)

// RecommendationImpact describes the expected impact of a recommendation
type RecommendationImpact struct {
	// SuccessRateChange is the expected change in placement success rate
	SuccessRateChange float64

	// ResourceUtilizationChange is the expected change in resource utilization
	ResourceUtilizationChange float64

	// PerformanceChange is the expected change in system performance
	PerformanceChange float64

	// DescriptiveImpact provides a textual description of the impact
	DescriptiveImpact string
}

// NewPolicyRecommendationEngine creates a new policy recommendation engine
func NewPolicyRecommendationEngine(engine *PolicyEngine) *PolicyRecommendationEngine {
	return &PolicyRecommendationEngine{
		Engine:                engine,
		Simulator:             NewPolicySimulator(engine),
		HistoricalData:        make([]*PlacementPerformanceData, 0),
		RecommendationHistory: make([]*PolicyRecommendation, 0),
	}
}

// RecordPerformanceData records performance data for future analysis
func (e *PolicyRecommendationEngine) RecordPerformanceData(data *PlacementPerformanceData) {
	e.HistoricalData = append(e.HistoricalData, data)

	// Trim history if it gets too large (keep last 100 entries)
	if len(e.HistoricalData) > 100 {
		e.HistoricalData = e.HistoricalData[len(e.HistoricalData)-100:]
	}
}

// GenerateRecommendations analyzes historical data and generates policy recommendations
func (e *PolicyRecommendationEngine) GenerateRecommendations(ctx context.Context) ([]*PolicyRecommendation, error) {
	recommendations := make([]*PolicyRecommendation, 0)

	// Ensure we have enough historical data
	if len(e.HistoricalData) < 5 {
		return recommendations, fmt.Errorf("insufficient historical data for meaningful recommendations")
	}

	// Get current active policies
	currentPolicies := make(map[string]*PolicyConfiguration)
	for id, conf := range e.Engine.ActiveConfigurations {
		currentPolicies[id] = conf
	}

	// Generate different types of recommendations
	utilRecs, err := e.generateUtilizationRecommendations(ctx, currentPolicies)
	if err == nil {
		recommendations = append(recommendations, utilRecs...)
	}

	balanceRecs, err := e.generateBalancingRecommendations(ctx, currentPolicies)
	if err == nil {
		recommendations = append(recommendations, balanceRecs...)
	}

	failureRecs, err := e.generateFailureReductionRecommendations(ctx, currentPolicies)
	if err == nil {
		recommendations = append(recommendations, failureRecs...)
	}

	// Add to recommendation history
	for _, rec := range recommendations {
		e.RecommendationHistory = append(e.RecommendationHistory, rec)
	}

	// Sort recommendations by expected impact
	sort.Slice(recommendations, func(i, j int) bool {
		// Higher success rate change is better
		if recommendations[i].ExpectedImpact.SuccessRateChange != recommendations[j].ExpectedImpact.SuccessRateChange {
			return recommendations[i].ExpectedImpact.SuccessRateChange > recommendations[j].ExpectedImpact.SuccessRateChange
		}

		// Higher resource utilization is better
		if recommendations[i].ExpectedImpact.ResourceUtilizationChange != recommendations[j].ExpectedImpact.ResourceUtilizationChange {
			return recommendations[i].ExpectedImpact.ResourceUtilizationChange > recommendations[j].ExpectedImpact.ResourceUtilizationChange
		}

		// Higher performance change is better
		return recommendations[i].ExpectedImpact.PerformanceChange > recommendations[j].ExpectedImpact.PerformanceChange
	})

	return recommendations, nil
}

// generateUtilizationRecommendations generates recommendations for improving resource utilization
func (e *PolicyRecommendationEngine) generateUtilizationRecommendations(ctx context.Context, currentPolicies map[string]*PolicyConfiguration) ([]*PolicyRecommendation, error) {
	recommendations := make([]*PolicyRecommendation, 0)

	// Analyze resource utilization trends
	avgCPUUtil := 0.0
	avgMemUtil := 0.0

	for _, data := range e.HistoricalData {
		avgCPUUtil += data.ResourceUsage.CPUUtilization
		avgMemUtil += data.ResourceUsage.MemoryUtilization
	}

	if len(e.HistoricalData) > 0 {
		avgCPUUtil /= float64(len(e.HistoricalData))
		avgMemUtil /= float64(len(e.HistoricalData))
	}

	// If average utilization is low, recommend consolidation policies
	if avgCPUUtil < 40.0 && avgMemUtil < 40.0 {
		recommendedPolicies := make(map[string]*PolicyConfiguration)

		// Copy current policies
		for id, conf := range currentPolicies {
			copied := *conf
			recommendedPolicies[id] = &copied
		}

		// Find or create consolidation policy
		consolidationPolicyID := "resource-consolidation"
		var consolidationPolicy *PolicyConfiguration

		if conf, exists := recommendedPolicies[consolidationPolicyID]; exists {
			consolidationPolicy = conf

			// Increase consolidation policy priority
			if consolidationPolicy.Priority < 80 {
				consolidationPolicy.Priority = 80
			}

			// Adjust parameter values for more aggressive consolidation
			params := consolidationPolicy.ParameterValues
			if threshold, exists := params["consolidation_threshold"]; exists {
				if thresholdVal, ok := threshold.(float64); ok && thresholdVal < 75.0 {
					params["consolidation_threshold"] = 75.0
				}
			}
		} else {
			// Create new consolidation policy configuration
			consolidationPolicy = &PolicyConfiguration{
				PolicyID: consolidationPolicyID,
				Priority: 80,
				ParameterValues: map[string]interface{}{
					"consolidation_threshold": 75.0,
					"min_resources_available": 20.0,
				},
				Enabled: true,
			}
			recommendedPolicies[consolidationPolicyID] = consolidationPolicy
		}

		// Create recommendation
		recommendation := &PolicyRecommendation{
			ID:                  fmt.Sprintf("rec-%d", time.Now().UnixNano()),
			Timestamp:           time.Now(),
			Reason:              fmt.Sprintf("Low resource utilization detected (CPU: %.1f%%, Memory: %.1f%%)", avgCPUUtil, avgMemUtil),
			Category:            ResourceUtilizationOptimization,
			CurrentPolicies:     currentPolicies,
			RecommendedPolicies: recommendedPolicies,
			ExpectedImpact: RecommendationImpact{
				SuccessRateChange:         0.0,  // Doesn't directly impact success rate
				ResourceUtilizationChange: 25.0, // Expectation of 25% improvement in utilization
				PerformanceChange:         -5.0, // Slight negative impact on performance due to consolidation
				DescriptiveImpact:         "Expected to improve resource utilization through workload consolidation, but may slightly impact performance due to higher resource contention",
			},
			Applied: false,
		}

		recommendations = append(recommendations, recommendation)
	}

	// If utilization is very high, recommend expansion policies
	if avgCPUUtil > 85.0 || avgMemUtil > 85.0 {
		recommendedPolicies := make(map[string]*PolicyConfiguration)

		// Copy current policies
		for id, conf := range currentPolicies {
			copied := *conf
			recommendedPolicies[id] = &copied
		}

		// Find resource distribution policy or create one
		distributionPolicyID := "resource-distribution"
		var distributionPolicy *PolicyConfiguration

		if conf, exists := recommendedPolicies[distributionPolicyID]; exists {
			distributionPolicy = conf

			// Increase distribution policy priority
			if distributionPolicy.Priority < 85 {
				distributionPolicy.Priority = 85
			}

			// Adjust parameter values for more aggressive distribution
			params := distributionPolicy.ParameterValues
			if threshold, exists := params["distribution_threshold"]; exists {
				if thresholdVal, ok := threshold.(float64); ok && thresholdVal > 70.0 {
					params["distribution_threshold"] = 70.0
				}
			}
		} else {
			// Create new distribution policy configuration
			distributionPolicy = &PolicyConfiguration{
				PolicyID: distributionPolicyID,
				Priority: 85,
				ParameterValues: map[string]interface{}{
					"distribution_threshold": 70.0,
					"spread_factor":          1.5,
				},
				Enabled: true,
			}
			recommendedPolicies[distributionPolicyID] = distributionPolicy
		}

		// Create recommendation
		recommendation := &PolicyRecommendation{
			ID:                  fmt.Sprintf("rec-%d", time.Now().UnixNano()),
			Timestamp:           time.Now(),
			Reason:              fmt.Sprintf("High resource utilization detected (CPU: %.1f%%, Memory: %.1f%%)", avgCPUUtil, avgMemUtil),
			Category:            ResourceBalancing,
			CurrentPolicies:     currentPolicies,
			RecommendedPolicies: recommendedPolicies,
			ExpectedImpact: RecommendationImpact{
				SuccessRateChange:         5.0,  // Small improvement in success rate
				ResourceUtilizationChange: -5.0, // Slight decrease in utilization due to spreading
				PerformanceChange:         10.0, // Improvement in performance due to less contention
				DescriptiveImpact:         "Expected to reduce resource contention by distributing workloads more evenly, improving performance and success rate but slightly reducing overall utilization",
			},
			Applied: false,
		}

		recommendations = append(recommendations, recommendation)
	}

	return recommendations, nil
}

// generateBalancingRecommendations generates recommendations for improving resource balancing
func (e *PolicyRecommendationEngine) generateBalancingRecommendations(ctx context.Context, currentPolicies map[string]*PolicyConfiguration) ([]*PolicyRecommendation, error) {
	recommendations := make([]*PolicyRecommendation, 0)

	// Analyze resource skew trends
	avgResourceSkew := 0.0
	for _, data := range e.HistoricalData {
		avgResourceSkew += data.ResourceUsage.ResourceSkew
	}

	if len(e.HistoricalData) > 0 {
		avgResourceSkew /= float64(len(e.HistoricalData))
	}

	// If resource skew is high, recommend balancing policies
	if avgResourceSkew > 0.3 { // 30% skew is high
		recommendedPolicies := make(map[string]*PolicyConfiguration)

		// Copy current policies
		for id, conf := range currentPolicies {
			copied := *conf
			recommendedPolicies[id] = &copied
		}

		// Find or create balancing policy
		balancingPolicyID := "workload-balancing"
		var balancingPolicy *PolicyConfiguration

		if conf, exists := recommendedPolicies[balancingPolicyID]; exists {
			balancingPolicy = conf

			// Increase balancing policy priority
			if balancingPolicy.Priority < 75 {
				balancingPolicy.Priority = 75
			}

			// Enable it if not enabled
			balancingPolicy.Enabled = true

			// Adjust parameter values for better balancing
			params := balancingPolicy.ParameterValues
			params["cpu_weight"] = 1.5
			params["memory_weight"] = 1.5
			params["network_weight"] = 1.0
		} else {
			// Create new balancing policy configuration
			balancingPolicy = &PolicyConfiguration{
				PolicyID: balancingPolicyID,
				Priority: 75,
				ParameterValues: map[string]interface{}{
					"cpu_weight":      1.5,
					"memory_weight":   1.5,
					"network_weight":  1.0,
					"balance_trigger": 0.2,
				},
				Enabled: true,
			}
			recommendedPolicies[balancingPolicyID] = balancingPolicy
		}

		// Create recommendation
		recommendation := &PolicyRecommendation{
			ID:                  fmt.Sprintf("rec-%d", time.Now().UnixNano()),
			Timestamp:           time.Now(),
			Reason:              fmt.Sprintf("High resource imbalance detected (Skew: %.1f%%)", avgResourceSkew*100),
			Category:            ResourceBalancing,
			CurrentPolicies:     currentPolicies,
			RecommendedPolicies: recommendedPolicies,
			ExpectedImpact: RecommendationImpact{
				SuccessRateChange:         3.0, // Small improvement in success rate
				ResourceUtilizationChange: 5.0, // Small improvement in utilization
				PerformanceChange:         8.0, // Moderate improvement in performance
				DescriptiveImpact:         "Expected to improve resource balance across nodes, reducing hotspots and improving overall system stability and performance",
			},
			Applied: false,
		}

		recommendations = append(recommendations, recommendation)
	}

	return recommendations, nil
}

// generateFailureReductionRecommendations generates recommendations for reducing placement failures
func (e *PolicyRecommendationEngine) generateFailureReductionRecommendations(ctx context.Context, currentPolicies map[string]*PolicyConfiguration) ([]*PolicyRecommendation, error) {
	recommendations := make([]*PolicyRecommendation, 0)

	// Analyze failure reasons
	failureReasons := make(map[string]int)

	for _, data := range e.HistoricalData {
		for reason, count := range data.FailureReasons {
			failureReasons[reason] += count
		}
	}

	// Find top failure reason
	var topReason string
	var topCount int

	for reason, count := range failureReasons {
		if count > topCount {
			topReason = reason
			topCount = count
		}
	}

	// If we have significant failures, recommend policy changes
	if topCount > 10 {
		recommendedPolicies := make(map[string]*PolicyConfiguration)

		// Copy current policies
		for id, conf := range currentPolicies {
			copied := *conf
			recommendedPolicies[id] = &copied
		}

		// Customize recommendation based on failure reason
		recommendationText := ""
		var failurePolicy *PolicyConfiguration

		switch {
		case topReason == "Insufficient memory":
			// Memory constraints need adjustment
			memoryPolicyID := "memory-resource-allocation"
			if conf, exists := recommendedPolicies[memoryPolicyID]; exists {
				failurePolicy = conf

				// Adjust memory headroom
				params := failurePolicy.ParameterValues
				if headroom, exists := params["memory_headroom_percent"]; exists {
					if headroomVal, ok := headroom.(float64); ok && headroomVal < 15.0 {
						params["memory_headroom_percent"] = headroomVal + 5.0
					}
				} else {
					params["memory_headroom_percent"] = 15.0
				}
			} else {
				// Create new memory policy
				failurePolicy = &PolicyConfiguration{
					PolicyID: memoryPolicyID,
					Priority: 90, // High priority for resource constraints
					ParameterValues: map[string]interface{}{
						"memory_headroom_percent": 15.0,
					},
					Enabled: true,
				}
				recommendedPolicies[memoryPolicyID] = failurePolicy
			}
			recommendationText = "Increase memory allocation headroom to reduce placement failures due to insufficient memory"

		case topReason == "Insufficient CPU":
			// CPU constraints need adjustment
			cpuPolicyID := "cpu-resource-allocation"
			if conf, exists := recommendedPolicies[cpuPolicyID]; exists {
				failurePolicy = conf

				// Adjust CPU headroom
				params := failurePolicy.ParameterValues
				if headroom, exists := params["cpu_headroom_percent"]; exists {
					if headroomVal, ok := headroom.(float64); ok && headroomVal < 15.0 {
						params["cpu_headroom_percent"] = headroomVal + 5.0
					}
				} else {
					params["cpu_headroom_percent"] = 15.0
				}
			} else {
				// Create new CPU policy
				failurePolicy = &PolicyConfiguration{
					PolicyID: cpuPolicyID,
					Priority: 90, // High priority for resource constraints
					ParameterValues: map[string]interface{}{
						"cpu_headroom_percent": 15.0,
					},
					Enabled: true,
				}
				recommendedPolicies[cpuPolicyID] = failurePolicy
			}
			recommendationText = "Increase CPU allocation headroom to reduce placement failures due to insufficient CPU"

		case topReason == "Affinity rule violation":
			// Affinity rules need adjustment
			affinityPolicyID := "affinity-rules"
			if conf, exists := recommendedPolicies[affinityPolicyID]; exists {
				failurePolicy = conf

				// Lower the priority slightly to allow more flexibility
				if failurePolicy.Priority > 60 {
					failurePolicy.Priority -= 10
				}

				// Make some rules soft constraints instead of hard
				params := failurePolicy.ParameterValues
				params["enforce_as_soft_constraint"] = true
			} else {
				// Create new affinity policy with soft constraints
				failurePolicy = &PolicyConfiguration{
					PolicyID: affinityPolicyID,
					Priority: 70,
					ParameterValues: map[string]interface{}{
						"enforce_as_soft_constraint": true,
						"soft_constraint_weight":     0.7,
					},
					Enabled: true,
				}
				recommendedPolicies[affinityPolicyID] = failurePolicy
			}
			recommendationText = "Adjust affinity rules to use soft constraints where possible to reduce placement failures"

		default:
			// Generic failure handling
			flexibilityPolicyID := "placement-flexibility"
			if conf, exists := recommendedPolicies[flexibilityPolicyID]; exists {
				failurePolicy = conf
				failurePolicy.Priority += 5
				failurePolicy.Enabled = true
			} else {
				// Create new flexibility policy
				failurePolicy = &PolicyConfiguration{
					PolicyID: flexibilityPolicyID,
					Priority: 65,
					ParameterValues: map[string]interface{}{
						"constraint_relaxation": 0.2, // 20% relaxation
					},
					Enabled: true,
				}
				recommendedPolicies[flexibilityPolicyID] = failurePolicy
			}
			recommendationText = fmt.Sprintf("Add placement flexibility policy to reduce failures due to '%s'", topReason)
		}

		// Create recommendation
		recommendation := &PolicyRecommendation{
			ID:                  fmt.Sprintf("rec-%d", time.Now().UnixNano()),
			Timestamp:           time.Now(),
			Reason:              fmt.Sprintf("High placement failure rate detected (Top reason: %s, Count: %d)", topReason, topCount),
			Category:            FailureReduction,
			CurrentPolicies:     currentPolicies,
			RecommendedPolicies: recommendedPolicies,
			ExpectedImpact: RecommendationImpact{
				SuccessRateChange:         15.0, // Significant improvement in success rate
				ResourceUtilizationChange: -2.0, // Slight decrease in utilization due to more headroom
				PerformanceChange:         5.0,  // Moderate improvement in performance
				DescriptiveImpact:         recommendationText,
			},
			Applied: false,
		}

		recommendations = append(recommendations, recommendation)
	}

	return recommendations, nil
}

// ApplyRecommendation applies a policy recommendation to the engine
func (e *PolicyRecommendationEngine) ApplyRecommendation(ctx context.Context, recommendationID string) error {
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

	// Apply each recommended policy configuration
	for policyID, config := range recommendation.RecommendedPolicies {
		// Check if policy exists
		_, err := e.Engine.GetPolicy(policyID)
		if err != nil {
			// Policy doesn't exist, might need to be created
			// In a real implementation, this would create the policy
			return fmt.Errorf("policy %s not found", policyID)
		}

		// Activate with recommended configuration
		err = e.Engine.ActivatePolicy(policyID, config)
		if err != nil {
			return fmt.Errorf("failed to activate policy %s: %v", policyID, err)
		}
	}

	// Mark recommendation as applied
	recommendation.Applied = true
	recommendation.AppliedAt = time.Now()

	return nil
}

// SimulateRecommendation simulates the impact of a policy recommendation
func (e *PolicyRecommendationEngine) SimulateRecommendation(ctx context.Context, recommendationID string, vms []map[string]interface{}, nodes []map[string]interface{}) (*SimulationComparison, error) {
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

	// Store original configurations
	originalConfigs := make(map[string]*PolicyConfiguration)
	for id, conf := range e.Engine.ActiveConfigurations {
		originalConfigs[id] = conf
	}

	// Run simulation with current configurations
	currentSimName := fmt.Sprintf("current-config-%s", recommendationID)
	currentSimResult, err := e.Simulator.RunSimulation(ctx, currentSimName, "Current configuration", vms, nodes)
	if err != nil {
		return nil, fmt.Errorf("failed to simulate current configuration: %v", err)
	}

	// Apply recommended configurations temporarily
	for policyID, config := range recommendation.RecommendedPolicies {
		// Skip policies that don't exist
		_, err := e.Engine.GetPolicy(policyID)
		if err != nil {
			continue
		}

		// Activate with recommended configuration
		err = e.Engine.ActivatePolicy(policyID, config)
		if err != nil {
			// Restore original configurations
			for id, conf := range originalConfigs {
				e.Engine.ActivatePolicy(id, conf)
			}
			return nil, fmt.Errorf("failed to activate policy %s for simulation: %v", policyID, err)
		}
	}

	// Run simulation with recommended configurations
	recommendedSimName := fmt.Sprintf("recommended-config-%s", recommendationID)
	recommendedSimResult, err := e.Simulator.RunSimulation(ctx, recommendedSimName, "Recommended configuration", vms, nodes)
	if err != nil {
		// Restore original configurations
		for id, conf := range originalConfigs {
			e.Engine.ActivatePolicy(id, conf)
		}
		return nil, fmt.Errorf("failed to simulate recommended configuration: %v", err)
	}

	// Restore original configurations
	for id, conf := range originalConfigs {
		e.Engine.ActivatePolicy(id, conf)
	}

	// Compare simulation results
	comparison, err := e.Simulator.CompareSimulations(currentSimResult.ID, recommendedSimResult.ID)
	if err != nil {
		return nil, fmt.Errorf("failed to compare simulations: %v", err)
	}

	// Save comparison to recommendation
	recommendation.SimulationResults = comparison

	return comparison, nil
}

// GetRecommendationHistory returns the history of policy recommendations
func (e *PolicyRecommendationEngine) GetRecommendationHistory() []*PolicyRecommendation {
	return e.RecommendationHistory
}

// CollectCurrentPerformanceData collects current performance data
func (e *PolicyRecommendationEngine) CollectCurrentPerformanceData(ctx context.Context) (*PlacementPerformanceData, error) {
	// In a real implementation, this would collect actual performance metrics
	// from the system. For simplicity, we'll create sample data

	// Get current active policies
	activePolicies := make(map[string]*PolicyConfiguration)
	for id, conf := range e.Engine.ActiveConfigurations {
		activePolicies[id] = conf
	}

	// Create sample performance data
	data := &PlacementPerformanceData{
		Timestamp:      time.Now(),
		ActivePolicies: activePolicies,
		VMTypeDistribution: map[string]int{
			"web":      10,
			"database": 5,
			"compute":  8,
			"storage":  3,
		},
		ResourceUsage: ResourceUtilizationData{
			CPUUtilization:     65.0,
			MemoryUtilization:  72.0,
			StorageUtilization: 45.0,
			NetworkUtilization: 30.0,
			ResourceSkew:       0.22, // 22% skew
		},
		SuccessRate:         90.0, // 90% success rate
		AvgPlacementLatency: 20 * time.Millisecond,
		FailureReasons: map[string]int{
			"Insufficient memory":     5,
			"Affinity rule violation": 3,
			"Insufficient CPU":        2,
			"Network constraints":     1,
		},
	}

	// Record the data
	e.RecordPerformanceData(data)

	return data, nil
}
