	package policy

import (
	"context"
	"encoding/json"
	"fmt"
	"time"
)

// PolicyMemoryIntegration provides integration between the policy framework
// and the code memory system for enhanced policy management
type PolicyMemoryIntegration struct {
	// Engine is the underlying policy engine
	Engine *PolicyEngine

	// Simulator for policy evaluation
	Simulator *PolicySimulator

	// RecommendationEngine for policy recommendations
	RecommendationEngine *PolicyRecommendationEngine

	// CodeMemoryClient for interacting with the code memory system
	CodeMemoryClient CodeMemoryClientInterface
}

// CodeMemoryClientInterface defines the interface for code memory interactions
type CodeMemoryClientInterface interface {
	// StorePolicy stores a policy with metadata and description in code memory
	StorePolicy(policyID string, policyDef string, metadata map[string]string) (string, error)

	// GetPolicy retrieves a policy by ID
	GetPolicy(policyID string) (PolicyMemoryEntry, error)

	// SearchPolicies searches for policies matching a description
	SearchPolicies(description string, limit int) ([]PolicyMemoryEntry, error)

	// StoreSimulationResult stores a simulation result in code memory
	StoreSimulationResult(simID string, result *SimulationResult) (string, error)

	// GetSimulationResult retrieves a simulation result by ID
	GetSimulationResult(simID string) (*SimulationResult, error)

	// SearchSimulationResults searches for similar simulation results
	SearchSimulationResults(description string, limit int) ([]SimulationResultEntry, error)
}

// PolicyMemoryEntry represents a policy entry retrieved from code memory
type PolicyMemoryEntry struct {
	// ID of the policy in code memory
	ID string

	// PolicyID is the unique identifier of the policy in the system
	PolicyID string

	// Definition is the full policy definition in DSL format
	Definition string

	// Type indicates the policy type (placement, migration, etc.)
	Type string

	// Description provides details about this policy
	Description string

	// Author is the creator of this policy
	Author string

	// CreatedAt is when this policy was created
	CreatedAt time.Time

	// Tags for categorizing and searching policies
	Tags []string

	// Score indicates the relevance to a search query (0.0-1.0)
	Score float64
}

// SimulationResultEntry represents a simulation result entry from code memory
type SimulationResultEntry struct {
	// ID of the simulation result in code memory
	ID string

	// SimulationID is the unique identifier of the simulation in the system
	SimulationID string

	// Name is a human-readable name for this simulation
	Name string

	// Description provides details about this simulation
	Description string

	// Timestamp when the simulation was run
	Timestamp time.Time

	// ScenarioType describes the type of scenario (e.g., "high-load", "maintenance")
	ScenarioType string

	// PolicyIDs used in this simulation
	PolicyIDs []string

	// Score indicates the relevance to a search query (0.0-1.0)
	Score float64
}

// NewPolicyMemoryIntegration creates a new integration between policy and code memory
func NewPolicyMemoryIntegration(engine *PolicyEngine, memoryClient CodeMemoryClientInterface) *PolicyMemoryIntegration {
	simulator := NewPolicySimulator(engine)
	recommender := NewPolicyRecommendationEngine(engine)

	return &PolicyMemoryIntegration{
		Engine:               engine,
		Simulator:            simulator,
		RecommendationEngine: recommender,
		CodeMemoryClient:     memoryClient,
	}
}

// StorePolicyInMemory stores a policy in the code memory system
func (i *PolicyMemoryIntegration) StorePolicyInMemory(ctx context.Context, policy *SchedulingPolicy) (string, error) {
	// Serialize the policy to its string representation
	policyJSON, err := json.MarshalIndent(policy, "", "  ")
	if err != nil {
		return "", fmt.Errorf("error serializing policy: %w", err)
	}

	// Create metadata for search
	metadata := map[string]string{
		"policy_id":   policy.ID,
		"type":        string(policy.Type),
		"description": policy.Description,
		"author":      policy.CreatedBy, // Use CreatedBy instead of Author
		"created_at":  policy.CreatedAt.Format(time.RFC3339),
		"version":     policy.Version,
	}

	// Add tags based on rules
	var tags []string
	for _, rule := range policy.Rules {
		tags = append(tags, fmt.Sprintf("rule:%s", rule.ID))

		// Add tags based on rule properties
		if rule.IsHardConstraint {
			tags = append(tags, "hard-constraint")
		} else {
			tags = append(tags, "soft-constraint")
		}
	}

	// Add tags to metadata
	for i, tag := range tags {
		metadata[fmt.Sprintf("tag_%d", i)] = tag
	}

	// Store in code memory
	return i.CodeMemoryClient.StorePolicy(policy.ID, string(policyJSON), metadata)
}

// FindSimilarPolicies finds similar policies based on description
func (i *PolicyMemoryIntegration) FindSimilarPolicies(ctx context.Context, description string, limit int) ([]PolicyMemoryEntry, error) {
	return i.CodeMemoryClient.SearchPolicies(description, limit)
}

// StoreSimulationInMemory stores a simulation result in the code memory system
func (i *PolicyMemoryIntegration) StoreSimulationInMemory(ctx context.Context, result *SimulationResult) (string, error) {
	return i.CodeMemoryClient.StoreSimulationResult(result.ID, result)
}

// FindSimilarSimulations finds similar simulation results based on description
func (i *PolicyMemoryIntegration) FindSimilarSimulations(ctx context.Context, description string, limit int) ([]SimulationResultEntry, error) {
	return i.CodeMemoryClient.SearchSimulationResults(description, limit)
}

// GenerateSimulationScenariosFromHistory generates simulation scenarios from historical data
func (i *PolicyMemoryIntegration) GenerateSimulationScenariosFromHistory(ctx context.Context, count int) ([]SimulationScenarioData, error) {
	// Find diverse historical simulation results
	entries, err := i.CodeMemoryClient.SearchSimulationResults("diverse workload scenarios", count*2)
	if err != nil {
		return nil, fmt.Errorf("error searching for historical simulations: %w", err)
	}

	// Create result array
	scenarios := make([]SimulationScenarioData, 0, count)

	// Set to track scenario types for diversity
	seenTypes := make(map[string]bool)

	// Process simulation entries
	for _, entry := range entries {
		if len(scenarios) >= count {
			break
		}

		// Skip if we've already seen this scenario type
		if seenTypes[entry.ScenarioType] {
			continue
		}

		// Get full simulation data
		simResult, err := i.CodeMemoryClient.GetSimulationResult(entry.SimulationID)
		if err != nil {
			continue
		}

		// Add to scenarios
		scenarios = append(scenarios, simResult.ScenarioData)
		seenTypes[entry.ScenarioType] = true
	}

	return scenarios, nil
}

// FindHistoricalPolicyPerformance finds historical performance data for a policy
func (i *PolicyMemoryIntegration) FindHistoricalPolicyPerformance(ctx context.Context, policyID string) (map[string]interface{}, error) {
	// Find simulation results that used this policy
	entries, err := i.CodeMemoryClient.SearchSimulationResults(fmt.Sprintf("policy:%s", policyID), 50)
	if err != nil {
		return nil, fmt.Errorf("error searching for policy simulations: %w", err)
	}

	// Track statistics
	stats := map[string]interface{}{
		"simulation_count":       len(entries),
		"avg_success_rate":       0.0,
		"avg_filtered_nodes":     0.0,
		"common_filter_reasons":  make(map[string]int),
		"scenario_types":         make(map[string]int),
		"historical_performance": make([]map[string]interface{}, 0),
	}

	// Process each simulation result
	totalSuccessRate := 0.0
	totalFilteredNodes := 0

	for _, entry := range entries {
		// Get full simulation data
		simResult, err := i.CodeMemoryClient.GetSimulationResult(entry.SimulationID)
		if err != nil {
			continue
		}

		// Track scenario type
		if scenarioType, ok := stats["scenario_types"].(map[string]int); ok {
			scenarioType[entry.ScenarioType]++
		}

		// Add to total success rate
		totalSuccessRate += simResult.Summary.PlacementPercentage

		// Process filter reasons
		for _, evalResult := range simResult.Results {
			if evalResult.Filtered {
				totalFilteredNodes++

				// Track filter reason
				if reasons, ok := stats["common_filter_reasons"].(map[string]int); ok {
					reasons[evalResult.FilterReason]++
				}
			}
		}

		// Add to historical performance
		perfEntry := map[string]interface{}{
			"simulation_id":     simResult.ID,
			"name":              simResult.Name,
			"timestamp":         simResult.Timestamp,
			"success_rate":      simResult.Summary.PlacementPercentage,
			"total_placements":  simResult.Summary.TotalPlacements,
			"failed_placements": simResult.Summary.FailedPlacements,
		}

		if historical, ok := stats["historical_performance"].([]map[string]interface{}); ok {
			stats["historical_performance"] = append(historical, perfEntry)
		}
	}

	// Calculate averages
	simCount := float64(len(entries))
	if simCount > 0 {
		stats["avg_success_rate"] = totalSuccessRate / simCount
	}

	if totalFilteredNodes > 0 {
		stats["avg_filtered_nodes"] = float64(totalFilteredNodes) / simCount
	}

	return stats, nil
}

// RecommendPolicyParameters recommends parameter values for a policy based on historical performance
func (i *PolicyMemoryIntegration) RecommendPolicyParameters(ctx context.Context, policyID string) (map[string]interface{}, error) {
	// Find historical performance data
	performanceData, err := i.FindHistoricalPolicyPerformance(ctx, policyID)
	if err != nil {
		return nil, err
	}

	// Get the policy
	policy, err := i.Engine.GetPolicy(policyID)
	if err != nil {
		return nil, fmt.Errorf("error getting policy %s: %w", policyID, err)
	}

	// Get similar policies
	similarPolicies, err := i.FindSimilarPolicies(ctx, policy.Description, 10)
	if err != nil {
		return nil, fmt.Errorf("error finding similar policies: %w", err)
	}

	// Get current parameters
	currentParams := make(map[string]interface{})
	if config, exists := i.Engine.ActiveConfigurations[policyID]; exists {
		currentParams = config.ParameterValues
	}

	// Analyze parameter patterns from similar successful policies
	recommendedParams := make(map[string]interface{})

	// Start with current parameters
	for k, v := range currentParams {
		recommendedParams[k] = v
	}

	// Look at similar policies with better performance
	avgSuccessRate, _ := performanceData["avg_success_rate"].(float64)

	// Find successful parameter patterns
	parameterCounts := make(map[string]map[interface{}]int)
	parameterValues := make(map[string][]interface{})

	for _, p := range similarPolicies {
		// Get simulation results for this policy
		simStats, err := i.FindHistoricalPolicyPerformance(ctx, p.PolicyID)
		if err != nil {
			continue
		}

		// Check if this policy performs better
		simAvgSuccess, _ := simStats["avg_success_rate"].(float64)
		if simAvgSuccess <= avgSuccessRate {
			continue
		}

		// Get the policy configuration
		polEntry, err := i.CodeMemoryClient.GetPolicy(p.PolicyID)
		if err != nil {
			continue
		}

		var pol SchedulingPolicy
		if err := json.Unmarshal([]byte(polEntry.Definition), &pol); err != nil {
			continue
		}

		// Get parameters from this policy
		config, exists := i.Engine.ActiveConfigurations[pol.ID]
		if !exists {
			continue
		}

		// Analyze parameter values
		for k, v := range config.ParameterValues {
			// Initialize maps if needed
			if _, exists := parameterCounts[k]; !exists {
				parameterCounts[k] = make(map[interface{}]int)
			}
			if _, exists := parameterValues[k]; !exists {
				parameterValues[k] = make([]interface{}, 0)
			}

			// Count this value
			parameterCounts[k][v]++
			parameterValues[k] = append(parameterValues[k], v)
		}
	}

	// Find most common value for each parameter
	for param, counts := range parameterCounts {
		var bestValue interface{}
		var bestCount int

		for value, count := range counts {
			if count > bestCount {
				bestValue = value
				bestCount = count
			}
		}

		// Only recommend if we have a clear winner
		if bestCount > 1 {
			recommendedParams[param] = bestValue
		}
	}

	return recommendedParams, nil
}

// SearchMemoryForFilterReason searches code memory for solutions to a specific filter reason
func (i *PolicyMemoryIntegration) SearchMemoryForFilterReason(ctx context.Context, filterReason string) ([]map[string]interface{}, error) {
	// Search for policies that might address this filter reason
	similarPolicies, err := i.CodeMemoryClient.SearchPolicies(filterReason, 5)
	if err != nil {
		return nil, fmt.Errorf("error finding policies for filter reason: %w", err)
	}

	// Search for simulations with this filter reason
	simEntries, err := i.CodeMemoryClient.SearchSimulationResults(filterReason, 10)
	if err != nil {
		return nil, fmt.Errorf("error finding simulations for filter reason: %w", err)
	}

	// Prepare solutions
	solutions := make([]map[string]interface{}, 0)

	// Add policy-based solutions
	for _, policy := range similarPolicies {
		solution := map[string]interface{}{
			"solution_type": "policy",
			"policy_id":     policy.PolicyID,
			"name":          policy.PolicyID,
			"description":   policy.Description,
			"relevance":     policy.Score,
			"suggestion":    fmt.Sprintf("Consider applying policy '%s' which is designed to address similar constraints", policy.PolicyID),
		}
		solutions = append(solutions, solution)
	}

	// Add simulation-based solutions
	for _, sim := range simEntries {
		// Get the full simulation data
		simResult, err := i.CodeMemoryClient.GetSimulationResult(sim.SimulationID)
		if err != nil {
			continue
		}

		// Find successful placements after filtering for this reason
		successfulPlacements := 0
		for _, evalResult := range simResult.Results {
			if evalResult.FilterReason == filterReason && !evalResult.Filtered {
				successfulPlacements++
			}
		}

		if successfulPlacements > 0 {
			policyIDs := simResult.PolicyIDs

			solution := map[string]interface{}{
				"solution_type": "simulation",
				"simulation_id": simResult.ID,
				"name":          simResult.Name,
				"policies_used": policyIDs,
				"success_count": successfulPlacements,
				"relevance":     sim.Score,
				"suggestion":    fmt.Sprintf("In a similar scenario, %d placements succeeded with policies: %v", successfulPlacements, policyIDs),
			}
			solutions = append(solutions, solution)
		}
	}

	return solutions, nil
}

// CreateCodeExample generates a code example for using a policy
func (i *PolicyMemoryIntegration) CreateCodeExample(ctx context.Context, policyID string) (string, error) {
	policy, err := i.Engine.GetPolicy(policyID)
	if err != nil {
		return "", fmt.Errorf("error getting policy %s: %w", policyID, err)
	}

	// Create a sample Go code using this policy
	codeExample := fmt.Sprintf(`// Example usage of the "%s" policy
package main

import (
	"context"
	"log"
	
	"github.com/khryptorgraphics/novacron/backend/core/scheduler/policy"
)

func main() {
	// Create a new policy engine
	engine := policy.NewPolicyEngine()
	
	// Get the %s policy - this assumes it's already registered
	%s, err := engine.GetPolicy("%s")
	if err != nil {
		log.Fatalf("Error getting policy: %%v", err)
	}
	
	// Create a configuration for the policy
	config := &policy.PolicyConfiguration{
		PolicyID: "%s",
		Priority: 75,
		ParameterValues: map[string]interface{}{
`, policy.ID, policy.ID, safeVariableName(policy.ID), policy.ID, policy.ID)

	// Add some default parameter values
	if config, exists := i.Engine.ActiveConfigurations[policy.ID]; exists && config.ParameterValues != nil {
		for k, v := range config.ParameterValues {
			switch value := v.(type) {
			case string:
				codeExample += fmt.Sprintf(`			"%s": "%s",
`, k, value)
			case float64:
				codeExample += fmt.Sprintf(`			"%s": %.2f,
`, k, value)
			case bool:
				codeExample += fmt.Sprintf(`			"%s": %t,
`, k, value)
			default:
				codeExample += fmt.Sprintf(`			"%s": %v,
`, k, v)
			}
		}
	}

	codeExample += `		},
		Enabled: true,
	}
	
	// Activate the policy with the configuration
	if err := engine.ActivatePolicy("%s", config); err != nil {
		log.Fatalf("Error activating policy: %v", err)
	}
	
	// Now the policy is active and will be used for evaluations
	log.Printf("Successfully activated policy: %s")
	
	// Create sample VM and nodes for evaluation
	vm := map[string]interface{}{
		"id":   "vm-123",
		"name": "sample-vm",
`

	// Add some common VM properties for demonstration
	codeExample += fmt.Sprintf(`		"cpu_cores": 2,
		"memory_gb": 4.0,
		"labels": map[string]string{
			"workload-type": "app",
			"environment": "staging",
		},
`)

	// Try to infer some VM properties from policy rules
	for _, rule := range policy.Rules {
		// We won't try to parse the Expression directly as it's complex
		// Instead, just include the rule IDs as comments to show policy coverage
		codeExample += fmt.Sprintf(`		// Property inferred from rule: %s
`, rule.ID)
	}

	codeExample += `	}
	
	nodes := []map[string]interface{}{
		{
			"id":   "node-1",
			"name": "node-west-1a",
`

	// Add some common node properties for demonstration
	codeExample += fmt.Sprintf(`			"cpu_cores": 16,
			"memory_gb": 64.0,
			"gpu_available": true,
			"datacenter": "west-1a",
			"labels": map[string]string{
				"node-type": "compute",
				"environment": "production",
			},
`)

	// Try to infer some node properties from policy rules
	for _, rule := range policy.Rules {
		// Just include the rule names as comments to show policy coverage
		codeExample += fmt.Sprintf(`			// Property inferred from rule: %s
`, rule.ID)
	}

	codeExample += `		},
		// Additional nodes would be defined here
	}
	
	// Evaluate the policies for this VM against all nodes
	filteredNodes, err := engine.EvaluatePlacementPolicies(context.Background(), vm, nodes)
	if err != nil {
		log.Fatalf("Error evaluating policies: %v", err)
	}
	
	// Process the results
	log.Printf("Filtered %d nodes down to %d candidates", len(nodes), len(filteredNodes))
	
	for _, node := range filteredNodes {
		nodeID, _ := node["id"].(string)
		log.Printf("Node %s is a suitable placement target", nodeID)
	}
}
`

	return codeExample, nil
}

// Helper function to create a safe variable name
func safeVariableName(name string) string {
	// Replace dashes with underscores
	result := ""
	for _, r := range name {
		if r == '-' {
			result += "_"
		} else {
			result += string(r)
		}
	}
	return result
}
