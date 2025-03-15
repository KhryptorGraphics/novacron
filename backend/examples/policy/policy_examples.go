package main

import (
	"context"
	"fmt"
	"log"

	"github.com/khryptorgraphics/novacron/backend/core/scheduler/policy"
)

// Using a comment to force a rebuild with the replace directive in go.mod
// This should use the local version of the policy package

// Example policy definitions in the policy DSL format
const highPerformanceGpuPolicy = `policy "high-performance-gpu" {
    id = "gpu-policy-001"
    type = "placement"
    description = "Prioritize GPU-intensive workloads on high-performance nodes"
    
    metadata {
        version = "1.0"
        author = "System Admin"
        category = "performance"
    }

    rules {
        rule "require-gpu-capability" {
            id = "rule-gpu-req"
            description = "Ensure node has GPU capability"
            priority = 100
            weight = 10
            hard_constraint = true
            
            when {
                vm.labels["workload-type"] == "gpu" || vm.resource_requests["gpu"] > 0
            }
            
            then {
                filter "Node does not have GPU capability"
                log "Checking GPU capability requirement" level="debug"
            }
        }
        
        rule "prefer-high-gpu-capacity" {
            id = "rule-gpu-pref"
            description = "Prefer nodes with highest GPU capacity"
            priority = 90
            weight = 8
            hard_constraint = false
            
            when {
                candidateNode.capabilities["gpu"] == true
            }
            
            then {
                score candidateNode.resources["gpu.capacity"] * 10 "GPU capacity score"
                log "Scoring node based on GPU capacity" level="debug"
            }
        }
        
        rule "avoid-high-cpu-util" {
            id = "rule-cpu-util"
            description = "Avoid nodes with high CPU utilization"
            priority = 80
            weight = 5
            hard_constraint = false
            
            when {
                true
            }
            
            then {
                score -candidateNode.metrics["cpu.utilization"] * 5 "CPU utilization penalty"
            }
        }
    }
    
    parameters {
        param "gpu_score_weight" {
            type = "float"
            default = 10.0
            description = "Weight for GPU capacity in scoring"
            min = 1.0
            max = 50.0
        }
        
        param "cpu_penalty_weight" {
            type = "float"
            default = 5.0
            description = "Weight for CPU utilization penalty"
            min = 0.0
            max = 20.0
        }
        
        param "min_gpu_memory" {
            type = "int"
            default = 8
            description = "Minimum GPU memory in GB"
            min = 4
            max = 32
        }
    }
}
`

const memoryAwarePolicy = `policy "memory-aware-placement" {
    type = "placement"
    description = "Place memory-intensive workloads efficiently"
    
    rules {
        rule "check-memory-requirements" {
            priority = 100
            hard_constraint = true
            
            when {
                vm.resource_requests["memory"] > candidateNode.resources["memory.available"]
            }
            
            then {
                filter "Insufficient memory available"
                log "Node has insufficient memory" level="info"
            }
        }
        
        rule "prefer-memory-balanced" {
            when {
                vm.resource_requests["memory"] > 0
            }
            
            then {
                score (candidateNode.resources["memory.available"] / vm.resource_requests["memory"]) * 5 "Memory balance score"
            }
        }
    }
}
`

const wanMigrationPolicy = `policy "wan-migration-optimizer" {
    type = "migration"
    description = "Optimize WAN migrations to reduce data transfer and downtime"
    
    rules {
        rule "bandwidth-threshold" {
            hard_constraint = true
            
            when {
                sourceNode.metrics["network.bandwidth.to." + candidateNode.id] < $param.min_bandwidth
            }
            
            then {
                filter "Insufficient bandwidth for WAN migration"
            }
        }
        
        rule "latency-score" {
            when {
                true
            }
            
            then {
                score -sourceNode.metrics["network.latency.to." + candidateNode.id] * 10 "Latency penalty"
            }
        }
        
        rule "data-transfer-cost" {
            when {
                sourceNode.datacenter != candidateNode.datacenter
            }
            
            then {
                score -vm.metrics["disk.used"] * sourceNode.metrics["network.cost.to." + candidateNode.id] "Data transfer cost"
            }
        }
    }
    
    parameters {
        param "min_bandwidth" {
            type = "float"
            default = 100.0
            description = "Minimum bandwidth in Mbps for WAN migration"
            min = 50.0
            max = 1000.0
        }
    }
}
`

func main() {
	parser := policy.NewPolicyParser()

	// Parse the high-performance GPU policy
	gpuPolicy, err := parser.ParsePolicy(highPerformanceGpuPolicy, "admin")
	if err != nil {
		log.Fatalf("Error parsing GPU policy: %v", err)
	}

	fmt.Println("Successfully parsed GPU policy:", gpuPolicy.Name)
	fmt.Println("Rules:", len(gpuPolicy.Rules))
	fmt.Println("Parameters:", len(gpuPolicy.Parameters))

	// Parse the memory-aware policy
	memoryPolicy, err := parser.ParsePolicy(memoryAwarePolicy, "admin")
	if err != nil {
		log.Fatalf("Error parsing memory policy: %v", err)
	}

	fmt.Println("\nSuccessfully parsed memory policy:", memoryPolicy.Name)
	fmt.Println("Rules:", len(memoryPolicy.Rules))

	// Parse the WAN migration policy
	migrationPolicy, err := parser.ParsePolicy(wanMigrationPolicy, "admin")
	if err != nil {
		log.Fatalf("Error parsing migration policy: %v", err)
	}

	fmt.Println("\nSuccessfully parsed migration policy:", migrationPolicy.Name)
	fmt.Println("Rules:", len(migrationPolicy.Rules))

	// Create a policy engine
	engine := policy.NewPolicyEngine()

	// Register policies
	if err := engine.RegisterPolicy(gpuPolicy); err != nil {
		log.Fatalf("Error registering GPU policy: %v", err)
	}

	if err := engine.RegisterPolicy(memoryPolicy); err != nil {
		log.Fatalf("Error registering memory policy: %v", err)
	}

	if err := engine.RegisterPolicy(migrationPolicy); err != nil {
		log.Fatalf("Error registering migration policy: %v", err)
	}

	// Activate the GPU policy
	gpuConfig := &policy.PolicyConfiguration{
		PolicyID: gpuPolicy.ID,
		Priority: 100,
		ParameterValues: map[string]interface{}{
			"gpu_score_weight":   15.0,
			"cpu_penalty_weight": 8.0,
			"min_gpu_memory":     16,
		},
	}

	if err := engine.ActivatePolicy(gpuPolicy.ID, gpuConfig); err != nil {
		log.Fatalf("Error activating GPU policy: %v", err)
	}

	// Example of a VM with GPU workload
	gpuVM := map[string]interface{}{
		"id": "vm-001",
		"labels": map[string]interface{}{
			"workload-type": "gpu",
		},
		"resource_requests": map[string]interface{}{
			"gpu":    1.0,
			"cpu":    8.0,
			"memory": 32.0,
		},
	}

	// Example of candidate nodes
	candidateNodes := []map[string]interface{}{
		{
			"id": "node-001",
			"capabilities": map[string]interface{}{
				"gpu": true,
			},
			"resources": map[string]interface{}{
				"gpu.capacity":     4.0,
				"memory.available": 64.0,
			},
			"metrics": map[string]interface{}{
				"cpu.utilization": 0.3,
			},
		},
		{
			"id": "node-002",
			"capabilities": map[string]interface{}{
				"gpu": true,
			},
			"resources": map[string]interface{}{
				"gpu.capacity":     8.0,
				"memory.available": 128.0,
			},
			"metrics": map[string]interface{}{
				"cpu.utilization": 0.5,
			},
		},
		{
			"id": "node-003",
			"capabilities": map[string]interface{}{
				"gpu": false,
			},
			"resources": map[string]interface{}{
				"gpu.capacity":     0.0,
				"memory.available": 256.0,
			},
			"metrics": map[string]interface{}{
				"cpu.utilization": 0.1,
			},
		},
	}

	// Evaluate placement policies for the VM
	filteredNodes, err := engine.EvaluatePlacementPolicies(context.Background(), gpuVM, candidateNodes)
	if err != nil {
		log.Fatalf("Error evaluating placement policies: %v", err)
	}

	fmt.Println("\nPlacement results:")
	fmt.Printf("Original candidate count: %d\n", len(candidateNodes))
	fmt.Printf("Filtered candidate count: %d\n", len(filteredNodes))

	for i, node := range filteredNodes {
		fmt.Printf("Rank %d: Node %s (GPU Capacity: %.1f)\n",
			i+1, node["id"], node["resources"].(map[string]interface{})["gpu.capacity"])
	}

	// Example of policy formatting
	fmt.Println("\nFormatted policy:")
	formattedPolicy := policy.FormatPolicy(gpuPolicy)
	fmt.Println(formattedPolicy)
}
