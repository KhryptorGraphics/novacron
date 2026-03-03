package advanced

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/scheduler/policy"
)

// RunAdvancedPolicyExample demonstrates the advanced policy engine capabilities
func RunAdvancedPolicyExample() {
	fmt.Println("Running Advanced Policy Engine Example...")

	// Create an advanced policy engine
	engine := policy.NewAdvancedPolicyEngine()

	// Create a custom policy language parser
	parser := policy.NewCustomPolicyLanguageParser()

	// Parse a policy definition
	policyDef := `policy "high-performance-workload" {
		id = "high-performance"
		type = "placement"
		description = "Optimizes placement for high-performance workloads"

		metadata {
			created_by = "admin"
			created_at = "2025-03-15T10:00:00Z"
			category = "performance"
		}

		parameters {
			parameter "cpu_weight" {
				type = "float"
				description = "Weight for CPU scoring"
				default = 2.0
				min = 0.0
				max = 10.0
			}
			parameter "memory_weight" {
				type = "float"
				description = "Weight for memory scoring"
				default = 1.5
				min = 0.0
				max = 10.0
			}
			parameter "network_weight" {
				type = "float"
				description = "Weight for network scoring"
				default = 1.0
				min = 0.0
				max = 10.0
			}
		}

		rules {
			rule "Require High-Performance Node" {
				id = "require-high-perf"
				description = "Requires a node with high-performance capability"
				hard_constraint = true
				when {
					vm.labels["workload-type"] == "high-performance"
				}
				then {
					filter "Node does not have high-performance capability"
				}
			}

			rule "Prefer CPU Availability" {
				id = "prefer-cpu"
				description = "Prefers nodes with more available CPU"
				hard_constraint = false
				when {
					true
				}
				then {
					score param.cpu_weight * (node.total_cpu_cores - node.used_cpu_cores) / node.total_cpu_cores * 100 "CPU availability score"
				}
			}

			rule "Prefer Memory Availability" {
				id = "prefer-memory"
				description = "Prefers nodes with more available memory"
				hard_constraint = false
				when {
					true
				}
				then {
					score param.memory_weight * (node.total_memory_gb - node.used_memory_gb) / node.total_memory_gb * 100 "Memory availability score"
				}
			}

			rule "Prefer Network Performance" {
				id = "prefer-network"
				description = "Prefers nodes with better network performance"
				hard_constraint = false
				when {
					vm.labels["network-intensive"] == "true"
				}
				then {
					score param.network_weight * node.network_performance_score "Network performance score"
					log "Applied network performance scoring" level="debug"
				}
			}
		}
	}`

	// Parse the policy
	highPerfPolicy, err := parser.ParseCustomPolicy(policyDef, "admin")
	if err != nil {
		log.Fatalf("Error parsing policy: %v", err)
	}

	// Create policy configuration
	highPerfConfig := &policy.PolicyConfiguration{
		PolicyID: highPerfPolicy.ID,
		Priority: 100,
		Enabled:  true,
		ParameterValues: map[string]interface{}{
			"cpu_weight":     3.0,
			"memory_weight":  2.0,
			"network_weight": 1.5,
		},
	}

	// Apply the policy
	ctx := context.Background()
	err = engine.ApplyPolicy(ctx, highPerfPolicy, highPerfConfig, "admin", "Initial policy creation")
	if err != nil {
		log.Fatalf("Error applying policy: %v", err)
	}

	fmt.Println("Successfully applied high-performance policy")

	// Create sample VMs and nodes for simulation
	vms, nodes := createSampleWorkloads()

	// Create a simulation scenario
	simulator := policy.NewEnhancedPolicySimulator(engine.PolicyEngine)
	scenarioID := "scenario-001"
	scenario := simulator.ScenarioManager.CreateScenario(
		scenarioID,
		"High-Performance Workload Scenario",
		"A scenario with high-performance workloads",
		vms,
		nodes,
		[]string{"high-performance", "production"},
	)

	fmt.Printf("Created simulation scenario with %d VMs and %d nodes\n", len(scenario.VMs), len(scenario.Nodes))

	// Run a simulation
	simulationResult, err := simulator.RunEnhancedSimulation(ctx,
		scenarioID,
		"High-Performance Policy Simulation",
		"Simulating the high-performance policy",
	)
	if err != nil {
		log.Fatalf("Error running simulation: %v", err)
	}

	fmt.Printf("Simulation results: %d/%d VMs successfully placed (%.1f%%)\n",
		simulationResult.Summary.TotalPlacements,
		simulationResult.Summary.TotalVMs,
		simulationResult.Summary.PlacementPercentage,
	)

	// Visualize the simulation result
	resultText, err := simulator.VisualizeSimulationResult(simulationResult.ID, "text")
	if err != nil {
		log.Fatalf("Error visualizing simulation result: %v", err)
	}

	fmt.Println("\nSimulation Result:")
	fmt.Println(string(resultText))

	// Create a modified policy configuration
	modifiedConfig := &policy.PolicyConfiguration{
		PolicyID: highPerfPolicy.ID,
		Priority: 100,
		Enabled:  true,
		ParameterValues: map[string]interface{}{
			"cpu_weight":     5.0, // Increased from 3.0
			"memory_weight":  1.0, // Decreased from 2.0
			"network_weight": 2.5, // Increased from 1.5
		},
	}

	// Analyze the impact of the modified configuration
	impactAnalyzer := policy.NewPolicyImpactAnalyzer(engine.PolicyEngine)
	impact, err := impactAnalyzer.AnalyzePolicyImpact(ctx, highPerfPolicy.ID, modifiedConfig, vms, nodes)
	if err != nil {
		log.Fatalf("Error analyzing policy impact: %v", err)
	}

	fmt.Println("\nPolicy Impact Analysis:")
	fmt.Printf("Placement difference: %+d\n", impact.SimulationComparison.PlacementDiff)
	fmt.Printf("Average score difference: %+.2f\n", impact.SimulationComparison.AverageScoreDiff)

	// Generate policy recommendations
	recommendationEngine := policy.NewEnhancedPolicyRecommendationEngine(engine.PolicyEngine)
	recommendations, err := recommendationEngine.GenerateEnhancedRecommendations(ctx)
	if err != nil {
		log.Fatalf("Error generating recommendations: %v", err)
	}

	fmt.Printf("\nGenerated %d policy recommendations\n", len(recommendations))
	for i, rec := range recommendations {
		fmt.Printf("%d. %s (Confidence: %.2f, Expected Improvement: %.1f)\n",
			i+1, rec.Name, rec.Confidence, rec.ExpectedImprovementScore)
		fmt.Printf("   Description: %s\n", rec.Description)
	}

	// Apply a recommendation
	if len(recommendations) > 0 {
		rec := recommendations[0]
		fmt.Printf("\nApplying recommendation: %s\n", rec.Name)

		// Track baseline metrics
		baselineMetrics := map[string]float64{
			"placement_success_rate": simulationResult.Summary.PlacementPercentage,
			"average_score":          simulationResult.Summary.AverageScore,
		}

		err = recommendationEngine.TrackRecommendationApplication(ctx, rec.ID, baselineMetrics)
		if err != nil {
			log.Fatalf("Error tracking recommendation application: %v", err)
		}

		// Apply the recommendation
		err = engine.RecommendationEngine.ApplyRecommendation(ctx, rec.ID)
		if err != nil {
			log.Fatalf("Error applying recommendation: %v", err)
		}

		// Run another simulation to see the impact
		newSimulationResult, err := simulator.RunEnhancedSimulation(ctx,
			scenarioID,
			"Post-Recommendation Simulation",
			"Simulating after applying the recommendation",
		)
		if err != nil {
			log.Fatalf("Error running simulation: %v", err)
		}

		// Evaluate the impact
		resultMetrics := map[string]float64{
			"placement_success_rate": newSimulationResult.Summary.PlacementPercentage,
			"average_score":          newSimulationResult.Summary.AverageScore,
		}

		impact, err := recommendationEngine.EvaluateRecommendationImpact(ctx, rec.ID, resultMetrics)
		if err != nil {
			log.Fatalf("Error evaluating recommendation impact: %v", err)
		}

		fmt.Println("\nRecommendation Impact:")
		fmt.Printf("Expected improvement: %.1f%%\n", impact.ExpectedImprovement)
		fmt.Printf("Actual improvement: %.1f%%\n", impact.ActualImprovement)
		fmt.Printf("New placement success rate: %.1f%%\n", resultMetrics["placement_success_rate"])
		fmt.Printf("New average score: %.2f\n", resultMetrics["average_score"])
	}

	fmt.Println("\nAdvanced Policy Engine Example completed successfully")
}

// createSampleWorkloads creates sample VMs and nodes for simulation
func createSampleWorkloads() ([]map[string]interface{}, []map[string]interface{}) {
	// Create sample VMs
	vms := []map[string]interface{}{
		{
			"id":        "vm-001",
			"name":      "high-perf-1",
			"cpu_cores": 16.0,
			"memory_gb": 64.0,
			"labels": map[string]string{
				"workload-type":     "high-performance",
				"network-intensive": "true",
			},
		},
		{
			"id":        "vm-002",
			"name":      "high-perf-2",
			"cpu_cores": 32.0,
			"memory_gb": 128.0,
			"labels": map[string]string{
				"workload-type":     "high-performance",
				"network-intensive": "true",
			},
		},
		{
			"id":        "vm-003",
			"name":      "web-server-1",
			"cpu_cores": 4.0,
			"memory_gb": 16.0,
			"labels": map[string]string{
				"workload-type":     "web",
				"network-intensive": "true",
			},
		},
		{
			"id":        "vm-004",
			"name":      "db-server-1",
			"cpu_cores": 8.0,
			"memory_gb": 32.0,
			"labels": map[string]string{
				"workload-type":     "database",
				"network-intensive": "false",
			},
		},
		{
			"id":        "vm-005",
			"name":      "high-perf-3",
			"cpu_cores": 24.0,
			"memory_gb": 96.0,
			"labels": map[string]string{
				"workload-type":     "high-performance",
				"network-intensive": "true",
			},
		},
	}

	// Create sample nodes
	nodes := []map[string]interface{}{
		{
			"id":                        "node-001",
			"name":                      "high-perf-node-1",
			"total_cpu_cores":           128.0,
			"used_cpu_cores":            32.0,
			"total_memory_gb":           512.0,
			"used_memory_gb":            128.0,
			"vm_count":                  4,
			"datacenter":                "dc-east",
			"rack":                      "rack-a1",
			"maintenance_scheduled":     false,
			"network_performance_score": 90.0,
			"labels": map[string]string{
				"capability": "high-performance",
			},
		},
		{
			"id":                        "node-002",
			"name":                      "high-perf-node-2",
			"total_cpu_cores":           128.0,
			"used_cpu_cores":            64.0,
			"total_memory_gb":           512.0,
			"used_memory_gb":            256.0,
			"vm_count":                  8,
			"datacenter":                "dc-east",
			"rack":                      "rack-a2",
			"maintenance_scheduled":     false,
			"network_performance_score": 85.0,
			"labels": map[string]string{
				"capability": "high-performance",
			},
		},
		{
			"id":                        "node-003",
			"name":                      "standard-node-1",
			"total_cpu_cores":           64.0,
			"used_cpu_cores":            32.0,
			"total_memory_gb":           256.0,
			"used_memory_gb":            128.0,
			"vm_count":                  8,
			"datacenter":                "dc-east",
			"rack":                      "rack-b1",
			"maintenance_scheduled":     false,
			"network_performance_score": 70.0,
			"labels": map[string]string{
				"capability": "standard",
			},
		},
		{
			"id":                        "node-004",
			"name":                      "standard-node-2",
			"total_cpu_cores":           64.0,
			"used_cpu_cores":            16.0,
			"total_memory_gb":           256.0,
			"used_memory_gb":            64.0,
			"vm_count":                  4,
			"datacenter":                "dc-east",
			"rack":                      "rack-b2",
			"maintenance_scheduled":     false,
			"network_performance_score": 75.0,
			"labels": map[string]string{
				"capability": "standard",
			},
		},
		{
			"id":                        "node-005",
			"name":                      "high-perf-node-3",
			"total_cpu_cores":           128.0,
			"used_cpu_cores":            96.0,
			"total_memory_gb":           512.0,
			"used_memory_gb":            384.0,
			"vm_count":                  12,
			"datacenter":                "dc-west",
			"rack":                      "rack-c1",
			"maintenance_scheduled":     false,
			"network_performance_score": 95.0,
			"labels": map[string]string{
				"capability": "high-performance",
			},
		},
	}

	return vms, nodes
}
