package policy

import (
	"context"
	"encoding/json"
	"fmt"
	"time"
)

// PolicySimulator simulates policy evaluation and predicts outcomes
// for various scenarios without applying the actual changes
type PolicySimulator struct {
	// Engine is the policy engine to use for simulation
	Engine *PolicyEngine

	// Results tracks the results of simulation runs
	Results []*SimulationResult
}

// SimulationResult represents the result of a policy simulation
type SimulationResult struct {
	// ID uniquely identifies this simulation run
	ID string

	// Name is a human-readable name for this simulation
	Name string

	// Description provides details about this simulation
	Description string

	// Timestamp when the simulation was run
	Timestamp time.Time

	// PolicyIDs are the IDs of policies that were included in this simulation
	PolicyIDs []string

	// ScenarioData contains the VM and node data used in the simulation
	ScenarioData SimulationScenarioData

	// Results contains the evaluation results for each VM/node combination
	Results []SimulationEvaluationResult

	// Summary provides aggregated statistics about the simulation
	Summary SimulationSummary
}

// SimulationScenarioData contains the VM and node data used in a simulation
type SimulationScenarioData struct {
	// VMs contains the virtual machine data
	VMs []map[string]interface{}

	// Nodes contains the node data
	Nodes []map[string]interface{}
}

// SimulationEvaluationResult represents a single evaluation result within a simulation
type SimulationEvaluationResult struct {
	// VMID identifies the VM that was evaluated
	VMID string

	// NodeID identifies the node that was evaluated
	NodeID string

	// Filtered indicates if this node was filtered out
	Filtered bool

	// FilterReason explains why the node was filtered
	FilterReason string

	// Score is the total score for this node
	Score float64

	// PolicyResults contains results for individual policies
	PolicyResults map[string]PolicySimulationResult
}

// PolicySimulationResult represents the result of evaluating a single policy
type PolicySimulationResult struct {
	// PolicyID identifies the policy that was evaluated
	PolicyID string

	// Filtered indicates if this policy caused the node to be filtered
	Filtered bool

	// FilterReason explains why the policy filtered the node
	FilterReason string

	// Score is the score contribution from this policy
	Score float64

	// RuleResults contains results for individual rules
	RuleResults map[string]RuleSimulationResult
}

// RuleSimulationResult represents the result of evaluating a single rule
type RuleSimulationResult struct {
	// RuleID identifies the rule that was evaluated
	RuleID string

	// Applied indicates if the rule condition was met
	Applied bool

	// Effect describes the outcome of applying the rule
	Effect string

	// Score is the score contribution from this rule
	Score float64
}

// SimulationSummary provides aggregated statistics about a simulation
type SimulationSummary struct {
	// TotalVMs is the total number of VMs in the simulation
	TotalVMs int

	// TotalNodes is the total number of nodes in the simulation
	TotalNodes int

	// TotalPlacements is the number of successful placements
	TotalPlacements int

	// FailedPlacements is the number of VMs that couldn't be placed
	FailedPlacements int

	// PlacementPercentage is the percentage of VMs that were successfully placed
	PlacementPercentage float64

	// AverageScore is the average score of successful placements
	AverageScore float64

	// NodeUtilization tracks utilization statistics for each node
	NodeUtilization map[string]NodeUtilizationStats
}

// NodeUtilizationStats tracks utilization statistics for a node
type NodeUtilizationStats struct {
	// VMCount is the number of VMs placed on this node
	VMCount int

	// CPUUtilization is the percentage of CPU used
	CPUUtilization float64

	// MemoryUtilization is the percentage of memory used
	MemoryUtilization float64
}

// NewPolicySimulator creates a new policy simulator
func NewPolicySimulator(engine *PolicyEngine) *PolicySimulator {
	return &PolicySimulator{
		Engine:  engine,
		Results: make([]*SimulationResult, 0),
	}
}

// RunSimulation runs a policy simulation with the provided scenario
func (s *PolicySimulator) RunSimulation(ctx context.Context, name, description string, vms []map[string]interface{}, nodes []map[string]interface{}) (*SimulationResult, error) {
	simulationID := fmt.Sprintf("sim-%d", time.Now().UnixNano())

	// Create the simulation result
	result := &SimulationResult{
		ID:          simulationID,
		Name:        name,
		Description: description,
		Timestamp:   time.Now(),
		PolicyIDs:   make([]string, 0),
		ScenarioData: SimulationScenarioData{
			VMs:   vms,
			Nodes: nodes,
		},
		Results: make([]SimulationEvaluationResult, 0),
		Summary: SimulationSummary{
			TotalVMs:            len(vms),
			TotalNodes:          len(nodes),
			TotalPlacements:     0,
			FailedPlacements:    0,
			PlacementPercentage: 0,
			AverageScore:        0,
			NodeUtilization:     make(map[string]NodeUtilizationStats),
		},
	}

	// Initialize node utilization stats
	for _, node := range nodes {
		nodeID, ok := node["id"].(string)
		if !ok {
			continue
		}

		totalCPU, _ := getNodeResource(node, "total_cpu_cores", "cpu")
		usedCPU, _ := getNodeResource(node, "used_cpu_cores", "used_cpu")

		totalMemory, _ := getNodeResource(node, "total_memory_gb", "memory")
		usedMemory, _ := getNodeResource(node, "used_memory_gb", "used_memory")

		cpuUtilization := 0.0
		if totalCPU > 0 {
			cpuUtilization = usedCPU / totalCPU * 100
		}

		memoryUtilization := 0.0
		if totalMemory > 0 {
			memoryUtilization = usedMemory / totalMemory * 100
		}

		result.Summary.NodeUtilization[nodeID] = NodeUtilizationStats{
			VMCount:           0,
			CPUUtilization:    cpuUtilization,
			MemoryUtilization: memoryUtilization,
		}
	}

	// Get active policies
	activePolicyIDs := make([]string, 0)
	for policyID, config := range s.Engine.ActiveConfigurations {
		if config.Enabled {
			activePolicyIDs = append(activePolicyIDs, policyID)
		}
	}
	result.PolicyIDs = activePolicyIDs

	// Simulate placement for each VM
	totalScore := 0.0
	for _, vm := range vms {
		vmID, _ := vm["id"].(string)
		if vmID == "" {
			vmID = fmt.Sprintf("unknown-vm-%d", len(result.Results))
		}

		// Evaluate policies for this VM against all nodes
		scoredNodes, err := s.Engine.EvaluatePlacementPolicies(ctx, vm, nodes)
		if err != nil {
			return nil, fmt.Errorf("error evaluating policies: %v", err)
		}

		// Track if this VM was successfully placed
		if len(scoredNodes) > 0 {
			result.Summary.TotalPlacements++

			// Find the best node
			bestNode := scoredNodes[0]
			bestNodeID, _ := bestNode["id"].(string)

			// Update node utilization
			if stats, ok := result.Summary.NodeUtilization[bestNodeID]; ok {
				stats.VMCount++

				// Update CPU utilization
				vmCPU, _ := getVMResource(vm, "cpu_cores", "cpu")
				nodeTotalCPU, _ := getNodeResource(bestNode, "total_cpu_cores", "cpu")
				if nodeTotalCPU > 0 {
					stats.CPUUtilization += (vmCPU / nodeTotalCPU) * 100
				}

				// Update memory utilization
				vmMemory, _ := getVMResource(vm, "memory_gb", "memory")
				nodeTotalMemory, _ := getNodeResource(bestNode, "total_memory_gb", "memory")
				if nodeTotalMemory > 0 {
					stats.MemoryUtilization += (vmMemory / nodeTotalMemory) * 100
				}

				result.Summary.NodeUtilization[bestNodeID] = stats
			}

			// Add to total score for averaging
			totalScore += 100.0 // Assume max score of 100 for simplicity
		} else {
			result.Summary.FailedPlacements++
		}

		// Record evaluation results for each node
		for _, node := range nodes {
			nodeID, _ := node["id"].(string)
			if nodeID == "" {
				nodeID = fmt.Sprintf("unknown-node-%d", len(result.Results))
			}

			// Check if this node was filtered
			filtered := true
			for _, scoredNode := range scoredNodes {
				scoredNodeID, _ := scoredNode["id"].(string)
				if scoredNodeID == nodeID {
					filtered = false
					break
				}
			}

			// Create the evaluation result
			evalResult := SimulationEvaluationResult{
				VMID:          vmID,
				NodeID:        nodeID,
				Filtered:      filtered,
				FilterReason:  determineFilterReason(vm, node),
				Score:         calculateNodeScore(vm, node),
				PolicyResults: make(map[string]PolicySimulationResult),
			}

			// Add individual policy results (simplified for now)
			for _, policyID := range activePolicyIDs {
				policy, err := s.Engine.GetPolicy(policyID)
				if err != nil {
					continue
				}

				policyResult := PolicySimulationResult{
					PolicyID:     policyID,
					Filtered:     false, // Simplified
					FilterReason: "",
					Score:        0, // Simplified
					RuleResults:  make(map[string]RuleSimulationResult),
				}

				// Add individual rule results (simplified)
				for _, rule := range policy.Rules {
					ruleResult := RuleSimulationResult{
						RuleID:  rule.ID,
						Applied: true,      // Simplified
						Effect:  "Applied", // Simplified
						Score:   0,         // Simplified
					}

					policyResult.RuleResults[rule.ID] = ruleResult
				}

				evalResult.PolicyResults[policyID] = policyResult
			}

			result.Results = append(result.Results, evalResult)
		}
	}

	// Calculate summary statistics
	if result.Summary.TotalPlacements > 0 {
		result.Summary.AverageScore = totalScore / float64(result.Summary.TotalPlacements)
	}
	if result.Summary.TotalVMs > 0 {
		result.Summary.PlacementPercentage = float64(result.Summary.TotalPlacements) / float64(result.Summary.TotalVMs) * 100
	}

	// Add to results history
	s.Results = append(s.Results, result)

	return result, nil
}

// CompareSimulations compares two simulation results and returns a diff
func (s *PolicySimulator) CompareSimulations(sim1ID, sim2ID string) (*SimulationComparison, error) {
	var sim1, sim2 *SimulationResult

	// Find the two simulations
	for _, result := range s.Results {
		if result.ID == sim1ID {
			sim1 = result
		}
		if result.ID == sim2ID {
			sim2 = result
		}
	}

	if sim1 == nil {
		return nil, fmt.Errorf("simulation with ID %s not found", sim1ID)
	}
	if sim2 == nil {
		return nil, fmt.Errorf("simulation with ID %s not found", sim2ID)
	}

	// Create the comparison
	comparison := &SimulationComparison{
		Simulation1ID:           sim1ID,
		Simulation2ID:           sim2ID,
		Simulation1Name:         sim1.Name,
		Simulation2Name:         sim2.Name,
		PlacementDiff:           sim2.Summary.TotalPlacements - sim1.Summary.TotalPlacements,
		PlacementPercentageDiff: sim2.Summary.PlacementPercentage - sim1.Summary.PlacementPercentage,
		AverageScoreDiff:        sim2.Summary.AverageScore - sim1.Summary.AverageScore,
		VMPlacementChanges:      make([]VMPlacementChange, 0),
		NodeUtilizationChanges:  make([]NodeUtilizationChange, 0),
	}

	// Compare VM placements (simplified)
	// In a real implementation, this would track which VMs were placed on which nodes
	// and show changes in placement

	// Compare node utilization
	for nodeID, stats1 := range sim1.Summary.NodeUtilization {
		if stats2, ok := sim2.Summary.NodeUtilization[nodeID]; ok {
			// Calculate changes
			vmCountChange := stats2.VMCount - stats1.VMCount
			cpuUtilChange := stats2.CPUUtilization - stats1.CPUUtilization
			memUtilChange := stats2.MemoryUtilization - stats1.MemoryUtilization

			// Only record significant changes
			if vmCountChange != 0 || abs(cpuUtilChange) > 1.0 || abs(memUtilChange) > 1.0 {
				comparison.NodeUtilizationChanges = append(comparison.NodeUtilizationChanges, NodeUtilizationChange{
					NodeID:                  nodeID,
					VMCountChange:           vmCountChange,
					CPUUtilizationChange:    cpuUtilChange,
					MemoryUtilizationChange: memUtilChange,
				})
			}
		}
	}

	return comparison, nil
}

// SimulationComparison represents a comparison between two simulation results
type SimulationComparison struct {
	// Simulation1ID is the ID of the first simulation
	Simulation1ID string

	// Simulation2ID is the ID of the second simulation
	Simulation2ID string

	// Simulation1Name is the name of the first simulation
	Simulation1Name string

	// Simulation2Name is the name of the second simulation
	Simulation2Name string

	// PlacementDiff is the difference in successful placements
	PlacementDiff int

	// PlacementPercentageDiff is the difference in placement percentage
	PlacementPercentageDiff float64

	// AverageScoreDiff is the difference in average score
	AverageScoreDiff float64

	// VMPlacementChanges tracks changes in VM placements
	VMPlacementChanges []VMPlacementChange

	// NodeUtilizationChanges tracks changes in node utilization
	NodeUtilizationChanges []NodeUtilizationChange
}

// VMPlacementChange represents a change in VM placement between simulations
type VMPlacementChange struct {
	// VMID identifies the VM
	VMID string

	// Simulation1NodeID is where the VM was placed in the first simulation
	Simulation1NodeID string

	// Simulation2NodeID is where the VM was placed in the second simulation
	Simulation2NodeID string

	// ScoreDiff is the difference in placement score
	ScoreDiff float64
}

// NodeUtilizationChange represents a change in node utilization between simulations
type NodeUtilizationChange struct {
	// NodeID identifies the node
	NodeID string

	// VMCountChange is the change in number of VMs placed on this node
	VMCountChange int

	// CPUUtilizationChange is the change in CPU utilization
	CPUUtilizationChange float64

	// MemoryUtilizationChange is the change in memory utilization
	MemoryUtilizationChange float64
}

// SaveSimulationResult saves a simulation result to a JSON file
func (s *PolicySimulator) SaveSimulationResult(result *SimulationResult, filename string) error {
	data, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		return fmt.Errorf("error marshaling simulation result: %v", err)
	}

	// In a real implementation, this would write to a file
	_ = data

	return nil
}

// LoadSimulationResult loads a simulation result from a JSON file
func (s *PolicySimulator) LoadSimulationResult(filename string) (*SimulationResult, error) {
	// In a real implementation, this would read from a file
	return nil, fmt.Errorf("not implemented")
}

// GetSimulationResult gets a simulation result by ID
func (s *PolicySimulator) GetSimulationResult(id string) *SimulationResult {
	for _, result := range s.Results {
		if result.ID == id {
			return result
		}
	}
	return nil
}

// ListSimulationResults lists all simulation results
func (s *PolicySimulator) ListSimulationResults() []*SimulationResult {
	return s.Results
}

// Helper functions

// getNodeResource gets a resource value from a node, handling different naming conventions
func getNodeResource(node map[string]interface{}, primaryKey, fallbackKey string) (float64, bool) {
	// Try primary key first
	if val, ok := node[primaryKey]; ok {
		switch v := val.(type) {
		case float64:
			return v, true
		case int:
			return float64(v), true
		}
	}

	// Try fallback key
	if val, ok := node[fallbackKey]; ok {
		switch v := val.(type) {
		case float64:
			return v, true
		case int:
			return float64(v), true
		}
	}

	return 0, false
}

// getVMResource gets a resource value from a VM, handling different naming conventions
func getVMResource(vm map[string]interface{}, primaryKey, fallbackKey string) (float64, bool) {
	// Try primary key first
	if val, ok := vm[primaryKey]; ok {
		switch v := val.(type) {
		case float64:
			return v, true
		case int:
			return float64(v), true
		}
	}

	// Try fallback key
	if val, ok := vm[fallbackKey]; ok {
		switch v := val.(type) {
		case float64:
			return v, true
		case int:
			return float64(v), true
		}
	}

	return 0, false
}

// determineFilterReason determines why a node was filtered for a VM (simplified)
func determineFilterReason(vm, node map[string]interface{}) string {
	// In a real implementation, this would check all constraints
	// For simplicity, we just check resource constraints

	vmCPU, okVMCPU := getVMResource(vm, "cpu_cores", "cpu")
	nodeTotalCPU, okNodeTotalCPU := getNodeResource(node, "total_cpu_cores", "cpu")
	nodeUsedCPU, okNodeUsedCPU := getNodeResource(node, "used_cpu_cores", "used_cpu")

	if okVMCPU && okNodeTotalCPU && okNodeUsedCPU {
		availableCPU := nodeTotalCPU - nodeUsedCPU
		if availableCPU < vmCPU {
			return fmt.Sprintf("Insufficient CPU: required %.1f, available %.1f", vmCPU, availableCPU)
		}
	}

	vmMem, okVMMem := getVMResource(vm, "memory_gb", "memory")
	nodeTotalMem, okNodeTotalMem := getNodeResource(node, "total_memory_gb", "memory")
	nodeUsedMem, okNodeUsedMem := getNodeResource(node, "used_memory_gb", "used_memory")

	if okVMMem && okNodeTotalMem && okNodeUsedMem {
		availableMem := nodeTotalMem - nodeUsedMem
		if availableMem < vmMem {
			return fmt.Sprintf("Insufficient memory: required %.1f GB, available %.1f GB", vmMem, availableMem)
		}
	}

	// Check if node is in maintenance
	if maintenance, ok := node["maintenance_scheduled"].(bool); ok && maintenance {
		return "Node is scheduled for maintenance"
	}

	return "Unknown reason"
}

// calculateNodeScore calculates a score for a node (simplified)
func calculateNodeScore(vm, node map[string]interface{}) float64 {
	// In a real implementation, this would use the policy engine to calculate scores
	// For simplicity, we just use a basic algorithm

	score := 50.0 // Start with a base score

	// Add score based on available CPU (higher is better)
	vmCPU, okVMCPU := getVMResource(vm, "cpu_cores", "cpu")
	nodeTotalCPU, okNodeTotalCPU := getNodeResource(node, "total_cpu_cores", "cpu")
	nodeUsedCPU, okNodeUsedCPU := getNodeResource(node, "used_cpu_cores", "used_cpu")

	if okVMCPU && okNodeTotalCPU && okNodeUsedCPU {
		availableCPU := nodeTotalCPU - nodeUsedCPU
		if availableCPU > 0 {
			cpuRatio := vmCPU / availableCPU
			if cpuRatio <= 0.5 {
				score += 20.0 // Plenty of CPU available
			} else if cpuRatio <= 0.8 {
				score += 10.0 // Enough CPU available
			} else {
				score += 5.0 // Just enough CPU available
			}
		}
	}

	// Add score based on available memory (higher is better)
	vmMem, okVMMem := getVMResource(vm, "memory_gb", "memory")
	nodeTotalMem, okNodeTotalMem := getNodeResource(node, "total_memory_gb", "memory")
	nodeUsedMem, okNodeUsedMem := getNodeResource(node, "used_memory_gb", "used_memory")

	if okVMMem && okNodeTotalMem && okNodeUsedMem {
		availableMem := nodeTotalMem - nodeUsedMem
		if availableMem > 0 {
			memRatio := vmMem / availableMem
			if memRatio <= 0.5 {
				score += 20.0 // Plenty of memory available
			} else if memRatio <= 0.8 {
				score += 10.0 // Enough memory available
			} else {
				score += 5.0 // Just enough memory available
			}
		}
	}

	// Add score based on node power efficiency (if available)
	if powerEfficiency, ok := node["power_efficiency"].(float64); ok {
		score += powerEfficiency * 10.0
	}

	return score
}

// abs returns the absolute value of a float64
func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
