package complex_policies

import (
	"fmt"
	"log"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/scheduler/policy"
)

// AdvancedPlacementDemo demonstrates a more complex policy scenario
// involving multiple placement policies with different priorities
func AdvancedPlacementDemo() {
	fmt.Println("Running Advanced Placement Policy Demo...")

	// Create a policy engine
	engine := policy.NewPolicyEngine()

	// Register several complex policies
	registerComplexPolicies(engine)

	// Create simulation data
	vms := createSampleVMs()
	nodes := createSampleNodes()

	fmt.Printf("Evaluating policies for %d VMs across %d candidate nodes\n", len(vms), len(nodes))

	// Evaluate placement for each VM
	for i, vm := range vms {
		fmt.Printf("\n[VM %d: %s]\n", i+1, vm["name"])
		fmt.Printf("  Type: %s, Memory: %.1f GB, CPU: %d cores\n",
			vm["type"], vm["memory_gb"], vm["cpu_cores"])

		// Get all scores for this VM
		startTime := time.Now()
		results, err := evaluateAllNodes(engine, vm, nodes)
		if err != nil {
			log.Fatalf("Error evaluating policies: %v", err)
		}

		duration := time.Since(startTime)
		fmt.Printf("  Evaluation completed in %.2f ms\n", float64(duration.Microseconds())/1000.0)

		// Print results
		fmt.Println("  Results (filtered/scored nodes):")
		for _, result := range results {
			if result.Filtered {
				fmt.Printf("  - [FILTERED] Node %s: %s\n", result.Node["name"], result.FilterReason)
			} else {
				fmt.Printf("  - [SCORE %.2f] Node %s\n", result.Score, result.Node["name"])
			}
		}

		// Find the best node (highest score among non-filtered)
		bestNode, bestScore := findBestNode(results)
		if bestNode != nil {
			fmt.Printf("  Best placement: Node %s (score: %.2f)\n", bestNode["name"], bestScore)
		} else {
			fmt.Println("  No suitable nodes found for this VM")
		}
	}

	fmt.Println("\nAdvanced Placement Policy Demo completed")
}

// NodeEvaluationResult stores the evaluation result for a single node
type NodeEvaluationResult struct {
	Node         map[string]interface{}
	Score        float64
	Filtered     bool
	FilterReason string
}

// registerComplexPolicies registers multiple policies with the engine
func registerComplexPolicies(engine *policy.PolicyEngine) {
	// 1. Resource availability policy (hard constraint)
	resourcePolicy := createResourcePolicy()
	if err := engine.RegisterPolicy(resourcePolicy); err != nil {
		log.Fatalf("Error registering resource policy: %v", err)
	}

	// 2. Workload balancing policy (soft constraint with medium priority)
	balancingPolicy := createBalancingPolicy()
	if err := engine.RegisterPolicy(balancingPolicy); err != nil {
		log.Fatalf("Error registering balancing policy: %v", err)
	}

	// 3. Energy efficiency policy (soft constraint with lower priority)
	energyPolicy := createEnergyPolicy()
	if err := engine.RegisterPolicy(energyPolicy); err != nil {
		log.Fatalf("Error registering energy policy: %v", err)
	}

	// 4. Affinity/anti-affinity policy (mix of hard and soft constraints)
	affinityPolicy := createAffinityPolicy()
	if err := engine.RegisterPolicy(affinityPolicy); err != nil {
		log.Fatalf("Error registering affinity policy: %v", err)
	}

	// Activate all policies with different priorities and configurations
	activatePolicies(engine)

	fmt.Println("Registered and activated 4 complex policies")
}

// createResourcePolicy creates a policy that enforces resource constraints
func createResourcePolicy() *policy.Policy {
	// This would normally use the policy parser with DSL,
	// but we're creating it programmatically for this example
	return &policy.Policy{
		ID:          "resource-constraints",
		Name:        "Resource Constraints Policy",
		Type:        "placement",
		Description: "Enforces resource availability constraints",
		Rules: []*policy.Rule{
			{
				ID:             "memory-availability",
				Description:    "Ensure node has enough available memory",
				HardConstraint: true,
				// The actual conditions would be parsed from expressions
				// but we're simplifying by using mock expressions in this example
			},
			{
				ID:             "cpu-availability",
				Description:    "Ensure node has enough available CPU cores",
				HardConstraint: true,
			},
		},
	}
}

// createBalancingPolicy creates a load balancing policy
func createBalancingPolicy() *policy.Policy {
	return &policy.Policy{
		ID:          "workload-balancing",
		Name:        "Workload Balancing Policy",
		Type:        "placement",
		Description: "Distributes VMs evenly across nodes",
		Rules: []*policy.Rule{
			{
				ID:             "balance-cpu-usage",
				Description:    "Prefer nodes with lower CPU usage",
				HardConstraint: false,
			},
			{
				ID:             "balance-memory-usage",
				Description:    "Prefer nodes with lower memory usage",
				HardConstraint: false,
			},
			{
				ID:             "balance-vm-count",
				Description:    "Prefer nodes with fewer VMs",
				HardConstraint: false,
			},
		},
	}
}

// createEnergyPolicy creates an energy efficiency policy
func createEnergyPolicy() *policy.Policy {
	return &policy.Policy{
		ID:          "energy-efficiency",
		Name:        "Energy Efficiency Policy",
		Type:        "placement",
		Description: "Optimizes for energy efficiency",
		Rules: []*policy.Rule{
			{
				ID:             "power-usage",
				Description:    "Prefer nodes with better power efficiency",
				HardConstraint: false,
			},
			{
				ID:             "consolidation",
				Description:    "Consolidate VMs on fewer nodes when possible",
				HardConstraint: false,
			},
		},
	}
}

// createAffinityPolicy creates VM affinity/anti-affinity policies
func createAffinityPolicy() *policy.Policy {
	return &policy.Policy{
		ID:          "affinity-rules",
		Name:        "VM Affinity Rules",
		Type:        "placement",
		Description: "Enforces VM placement affinity and anti-affinity",
		Rules: []*policy.Rule{
			{
				ID:             "vm-anti-affinity",
				Description:    "Ensure VMs of the same group are on different nodes",
				HardConstraint: true,
			},
			{
				ID:             "service-affinity",
				Description:    "Try to place related services on the same node",
				HardConstraint: false,
			},
		},
	}
}

// activatePolicies activates all registered policies with configurations
func activatePolicies(engine *policy.PolicyEngine) {
	// Resource policy (highest priority - hard constraints)
	resourceConfig := &policy.PolicyConfiguration{
		PolicyID: "resource-constraints",
		Priority: 100,
		ParameterValues: map[string]interface{}{
			"memory_headroom_percent": 10.0,
			"cpu_headroom_percent":    15.0,
		},
	}
	if err := engine.ActivatePolicy("resource-constraints", resourceConfig); err != nil {
		log.Fatalf("Error activating resource policy: %v", err)
	}

	// Balancing policy (medium priority)
	balancingConfig := &policy.PolicyConfiguration{
		PolicyID: "workload-balancing",
		Priority: 70,
		ParameterValues: map[string]interface{}{
			"cpu_weight":    1.0,
			"memory_weight": 1.0,
			"count_weight":  1.5,
		},
	}
	if err := engine.ActivatePolicy("workload-balancing", balancingConfig); err != nil {
		log.Fatalf("Error activating balancing policy: %v", err)
	}

	// Energy policy (lower priority)
	energyConfig := &policy.PolicyConfiguration{
		PolicyID: "energy-efficiency",
		Priority: 50,
		ParameterValues: map[string]interface{}{
			"power_weight":        2.0,
			"consolidation_limit": 85.0,
		},
	}
	if err := engine.ActivatePolicy("energy-efficiency", energyConfig); err != nil {
		log.Fatalf("Error activating energy policy: %v", err)
	}

	// Affinity policy (mixed priority - contains both hard and soft constraints)
	affinityConfig := &policy.PolicyConfiguration{
		PolicyID: "affinity-rules",
		Priority: 80,
		ParameterValues: map[string]interface{}{
			"affinity_score_impact": 25.0,
		},
	}
	if err := engine.ActivatePolicy("affinity-rules", affinityConfig); err != nil {
		log.Fatalf("Error activating affinity policy: %v", err)
	}
}

// createSampleVMs creates a set of sample VMs for testing
func createSampleVMs() []map[string]interface{} {
	return []map[string]interface{}{
		{
			"id":         "vm-001",
			"name":       "web-server-1",
			"type":       "web",
			"memory_gb":  4.0,
			"cpu_cores":  2,
			"group":      "web-tier",
			"priority":   "high",
			"deployment": "frontend",
		},
		{
			"id":         "vm-002",
			"name":       "web-server-2",
			"type":       "web",
			"memory_gb":  4.0,
			"cpu_cores":  2,
			"group":      "web-tier",
			"priority":   "high",
			"deployment": "frontend",
		},
		{
			"id":         "vm-003",
			"name":       "db-primary",
			"type":       "database",
			"memory_gb":  16.0,
			"cpu_cores":  4,
			"group":      "db-tier",
			"priority":   "critical",
			"deployment": "backend",
		},
		{
			"id":         "vm-004",
			"name":       "analytics-worker",
			"type":       "worker",
			"memory_gb":  8.0,
			"cpu_cores":  8,
			"group":      "analytics",
			"priority":   "low",
			"deployment": "backend",
		},
		{
			"id":         "vm-005",
			"name":       "cache-server",
			"type":       "cache",
			"memory_gb":  32.0,
			"cpu_cores":  2,
			"group":      "cache-tier",
			"priority":   "medium",
			"deployment": "middleware",
		},
	}
}

// createSampleNodes creates a set of sample nodes for testing
func createSampleNodes() []map[string]interface{} {
	return []map[string]interface{}{
		{
			"id":                    "node-001",
			"name":                  "compute-01",
			"total_memory_gb":       128.0,
			"used_memory_gb":        48.0,
			"total_cpu_cores":       32,
			"used_cpu_cores":        12,
			"vm_count":              5,
			"power_efficiency":      0.85,
			"datacenter":            "dc-east",
			"rack":                  "rack-a1",
			"maintenance_scheduled": false,
			"hosted_vms":            []string{"vm-010", "vm-011", "vm-012", "vm-013", "vm-014"},
		},
		{
			"id":                    "node-002",
			"name":                  "compute-02",
			"total_memory_gb":       128.0,
			"used_memory_gb":        96.0,
			"total_cpu_cores":       32,
			"used_cpu_cores":        24,
			"vm_count":              10,
			"power_efficiency":      0.9,
			"datacenter":            "dc-east",
			"rack":                  "rack-a2",
			"maintenance_scheduled": false,
			"hosted_vms": []string{"vm-020", "vm-021", "vm-022", "vm-023", "vm-024",
				"vm-025", "vm-026", "vm-027", "vm-028", "vm-029"},
		},
		{
			"id":                    "node-003",
			"name":                  "compute-03",
			"total_memory_gb":       256.0,
			"used_memory_gb":        64.0,
			"total_cpu_cores":       64,
			"used_cpu_cores":        16,
			"vm_count":              3,
			"power_efficiency":      0.75,
			"datacenter":            "dc-west",
			"rack":                  "rack-b1",
			"maintenance_scheduled": true,
			"hosted_vms":            []string{"vm-030", "vm-031", "vm-032"},
		},
		{
			"id":                    "node-004",
			"name":                  "compute-04",
			"total_memory_gb":       256.0,
			"used_memory_gb":        128.0,
			"total_cpu_cores":       64,
			"used_cpu_cores":        32,
			"vm_count":              8,
			"power_efficiency":      0.8,
			"datacenter":            "dc-west",
			"rack":                  "rack-b2",
			"maintenance_scheduled": false,
			"hosted_vms": []string{"vm-040", "vm-041", "vm-042", "vm-043",
				"vm-044", "vm-045", "vm-046", "vm-047"},
		},
		{
			"id":                    "node-005",
			"name":                  "compute-05",
			"total_memory_gb":       512.0,
			"used_memory_gb":        128.0,
			"total_cpu_cores":       128,
			"used_cpu_cores":        32,
			"vm_count":              6,
			"power_efficiency":      0.95,
			"datacenter":            "dc-central",
			"rack":                  "rack-c1",
			"maintenance_scheduled": false,
			"hosted_vms":            []string{"vm-050", "vm-051", "vm-052", "vm-053", "vm-054", "vm-055"},
		},
	}
}

// evaluateAllNodes runs policy evaluation for a VM against all candidate nodes
func evaluateAllNodes(engine *policy.PolicyEngine, vm map[string]interface{},
	candidateNodes []map[string]interface{}) ([]NodeEvaluationResult, error) {
	results := make([]NodeEvaluationResult, 0, len(candidateNodes))

	for _, node := range candidateNodes {
		// This would normally use the policy evaluation context for rule evaluation
		// but we're just simulating the policy engine behavior here

		// For each rule in each active policy, evaluate it
		// This is a simplified version of what the policy engine would do internally
		isFiltered := false
		filterReason := ""
		totalScore := 0.0

		// Check if node has enough memory (hard constraint example)
		requiredMemory := vm["memory_gb"].(float64)
		availableMemory := node["total_memory_gb"].(float64) - node["used_memory_gb"].(float64)
		if availableMemory < requiredMemory {
			isFiltered = true
			filterReason = fmt.Sprintf("Insufficient memory: required %.1f GB, available %.1f GB",
				requiredMemory, availableMemory)
		}

		// Check if node has enough CPU (hard constraint example)
		requiredCPU := vm["cpu_cores"].(int)
		availableCPU := node["total_cpu_cores"].(int) - node["used_cpu_cores"].(int)
		if availableCPU < requiredCPU {
			isFiltered = true
			filterReason = fmt.Sprintf("Insufficient CPU: required %d cores, available %d cores",
				requiredCPU, availableCPU)
		}

		// Check maintenance status (hard constraint example)
		if node["maintenance_scheduled"].(bool) {
			isFiltered = true
			filterReason = "Node is scheduled for maintenance"
		}

		// If not filtered, calculate scores (soft constraints)
		if !isFiltered {
			// Score based on available memory (higher is better)
			memoryScore := (availableMemory / node["total_memory_gb"].(float64)) * 25.0

			// Score based on available CPU (higher is better)
			cpuScore := float64(availableCPU) / float64(node["total_cpu_cores"].(int)) * 25.0

			// Score based on VM count (lower count is better)
			vmCountScore := (1.0 - (float64(node["vm_count"].(int)) / 15.0)) * 20.0
			if vmCountScore < 0 {
				vmCountScore = 0
			}

			// Score based on power efficiency (higher is better)
			powerScore := node["power_efficiency"].(float64) * 30.0

			// Combined score
			totalScore = memoryScore + cpuScore + vmCountScore + powerScore
		}

		// Store result
		results = append(results, NodeEvaluationResult{
			Node:         node,
			Score:        totalScore,
			Filtered:     isFiltered,
			FilterReason: filterReason,
		})
	}

	return results, nil
}

// findBestNode finds the highest scoring node from evaluation results
func findBestNode(results []NodeEvaluationResult) (map[string]interface{}, float64) {
	var bestNode map[string]interface{}
	var bestScore float64 = -1.0

	for _, result := range results {
		if !result.Filtered && result.Score > bestScore {
			bestScore = result.Score
			bestNode = result.Node
		}
	}

	return bestNode, bestScore
}
