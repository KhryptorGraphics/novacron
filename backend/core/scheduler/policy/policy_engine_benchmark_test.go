package policy

import (
	"context"
	"fmt"
	"math/rand"
	"testing"
	"time"
)

// setupBenchmarkEngine creates a policy engine with sample policies for benchmarking
func setupBenchmarkEngine(b *testing.B) *PolicyEngine {
	engine := NewPolicyEngine()

	// Register sample policies for benchmarking
	policies := createBenchmarkPolicies(5) // 5 policies with different rules

	for _, p := range policies {
		err := engine.RegisterPolicy(p)
		if err != nil {
			b.Fatalf("Failed to register policy: %v", err)
		}

		// Activate the policy with sample config
		config := &PolicyConfiguration{
			PolicyID: p.ID,
			Priority: int(rand.Float64() * 100), // Use int for Priority
			ParameterValues: map[string]interface{}{
				"weight": rand.Float64() * 10,
			},
			Enabled: true,
		}

		err = engine.ActivatePolicy(p.ID, config)
		if err != nil {
			b.Fatalf("Failed to activate policy: %v", err)
		}
	}

	return engine
}

// createBenchmarkPolicies creates n sample policies for benchmarking
func createBenchmarkPolicies(n int) []*SchedulingPolicy {
	policies := make([]*SchedulingPolicy, n)

	for i := 0; i < n; i++ {
		policyID := fmt.Sprintf("benchmark-policy-%d", i)

		rules := make([]*PolicyRule, i+1) // Each policy has i+1 rules
		for j := 0; j < i+1; j++ {
			// Create a mock condition expression
			mockCondition := &MockExpression{Value: true}

			rules[j] = &PolicyRule{
				ID:               fmt.Sprintf("rule-%d", j),
				Description:      fmt.Sprintf("Benchmark rule %d for policy %d", j, i),
				IsHardConstraint: j%2 == 0, // Alternate between hard and soft constraints
				Condition:        mockCondition,
				Actions:          []PolicyAction{&LogAction{Message: "Benchmark action", Level: "INFO"}},
			}
		}

		policies[i] = &SchedulingPolicy{
			ID:          policyID,
			Name:        fmt.Sprintf("Benchmark Policy %d", i),
			Type:        PlacementPolicy,
			Description: fmt.Sprintf("Policy for benchmarking with %d rules", i+1),
			Rules:       rules,
			Status:      PolicyStatusActive,
			CreatedAt:   time.Now(),
			UpdatedAt:   time.Now(),
		}
	}

	return policies
}

// createBenchmarkVMContext creates a context with random VM and node data
func createBenchmarkVMContext() *PolicyEvaluationContext {
	vm := map[string]interface{}{
		"id":         fmt.Sprintf("vm-%d", rand.Intn(1000)),
		"memory_gb":  float64(1 + rand.Intn(64)),
		"cpu_cores":  1 + rand.Intn(16),
		"priority":   rand.Intn(5),
		"type":       []string{"web", "db", "cache", "compute", "storage"}[rand.Intn(5)],
		"created_at": time.Now().Add(-time.Duration(rand.Intn(30)) * 24 * time.Hour),
	}

	sourceNode := map[string]interface{}{
		"id":                 fmt.Sprintf("source-node-%d", rand.Intn(100)),
		"available_memory":   float64(4 + rand.Intn(252)),
		"available_cpu":      4 + rand.Intn(60),
		"datacenter":         []string{"east", "west", "central"}[rand.Intn(3)],
		"rack":               fmt.Sprintf("rack-%d", rand.Intn(10)),
		"power_utilization":  0.5 + rand.Float64()*0.5,
		"network_bandwidth":  1 + rand.Intn(40),
		"maintenance_window": time.Now().Add(time.Duration(rand.Intn(30)) * 24 * time.Hour),
	}

	candidateNode := map[string]interface{}{
		"id":                 fmt.Sprintf("candidate-node-%d", rand.Intn(100)),
		"available_memory":   float64(4 + rand.Intn(252)),
		"available_cpu":      4 + rand.Intn(60),
		"datacenter":         []string{"east", "west", "central"}[rand.Intn(3)],
		"rack":               fmt.Sprintf("rack-%d", rand.Intn(10)),
		"power_utilization":  0.5 + rand.Float64()*0.5,
		"network_bandwidth":  1 + rand.Intn(40),
		"maintenance_window": time.Now().Add(time.Duration(rand.Intn(30)) * 24 * time.Hour),
	}

	return NewPolicyEvaluationContext(vm, sourceNode, candidateNode)
}

// BenchmarkPolicyEngineSimple benchmarks a simple policy evaluation
func BenchmarkPolicyEngineSimple(b *testing.B) {
	ctx := context.Background()
	engine := setupBenchmarkEngine(b)
	vm := map[string]interface{}{"id": "test-vm", "memory": 8, "cpu": 4}
	nodes := []map[string]interface{}{
		{"id": "node-1", "memory": 32, "cpu": 16},
		{"id": "node-2", "memory": 64, "cpu": 32},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Evaluate policies using the actual method from PolicyEngine
		_, _ = engine.EvaluatePlacementPolicies(ctx, vm, nodes)
	}
}

// BenchmarkPolicyEngineComplex benchmarks a more complex policy evaluation with many contexts
func BenchmarkPolicyEngineComplex(b *testing.B) {
	ctx := context.Background()
	rand.Seed(time.Now().UnixNano())
	engine := setupBenchmarkEngine(b)

	// Pre-create VMs and nodes to avoid allocation during benchmark
	vms := make([]map[string]interface{}, 20)
	for i := 0; i < 20; i++ {
		vms[i] = map[string]interface{}{
			"id":        fmt.Sprintf("vm-%d", i),
			"memory_gb": float64(1 + rand.Intn(64)),
			"cpu_cores": 1 + rand.Intn(16),
			"priority":  rand.Intn(5),
		}
	}

	nodes := make([]map[string]interface{}, 10)
	for i := 0; i < 10; i++ {
		nodes[i] = map[string]interface{}{
			"id":               fmt.Sprintf("node-%d", i),
			"available_memory": float64(64 + rand.Intn(192)),
			"available_cpu":    16 + rand.Intn(48),
			"datacenter":       []string{"east", "west", "central"}[rand.Intn(3)],
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Use a different VM each time, cycling through the pre-created ones
		vm := vms[i%20]
		_, _ = engine.EvaluatePlacementPolicies(ctx, vm, nodes)
	}
}

// BenchmarkPolicyFuzzing performs fuzzy testing by randomly creating
// different policy configurations and testing them
func BenchmarkPolicyFuzzing(b *testing.B) {
	rand.Seed(time.Now().UnixNano())

	// Create a policy engine for each benchmark iteration
	// with random policies and configurations
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Create a new engine for each iteration to test different configurations
		engine := NewPolicyEngine()

		// Random number of policies between 1 and 10
		numPolicies := 1 + rand.Intn(10)
		policies := createBenchmarkPolicies(numPolicies)

		// Register and activate random policies
		for _, p := range policies {
			if err := engine.RegisterPolicy(p); err != nil {
				b.Fatalf("Failed to register policy: %v", err)
			}

			// Random configuration
			config := &PolicyConfiguration{
				PolicyID: p.ID,
				Priority: int(rand.Float64() * 100), // Cast to int
				ParameterValues: map[string]interface{}{
					"weight":     rand.Float64() * 10,
					"threshold":  rand.Float64() * 5,
					"min_value":  rand.Float64() * 2,
					"max_value":  rand.Float64()*5 + 5,
					"multiplier": rand.Float64() * 3,
				},
				Enabled: true, // Make sure it's enabled
			}

			if err := engine.ActivatePolicy(p.ID, config); err != nil {
				b.Fatalf("Failed to activate policy: %v", err)
			}
		}

		// Create sample VM and nodes for evaluation
		ctx := context.Background()
		vm := map[string]interface{}{"id": fmt.Sprintf("vm-test-%d", i), "memory_gb": 8, "cpu_cores": 4}
		nodes := []map[string]interface{}{
			{"id": fmt.Sprintf("node-test-%d", i), "available_memory": 64, "available_cpu": 16},
		}

		// Use the proper evaluation method
		_, _ = engine.EvaluatePlacementPolicies(ctx, vm, nodes)
	}
}

// BenchmarkPolicyEngineConcurrent benchmarks policy evaluation with concurrent access
func BenchmarkPolicyEngineConcurrent(b *testing.B) {
	engine := setupBenchmarkEngine(b)

	// Create shared test data
	testCtx := context.Background()
	testNodes := []map[string]interface{}{
		{"id": "concurrent-node-1", "available_memory": 128, "available_cpu": 32},
		{"id": "concurrent-node-2", "available_memory": 256, "available_cpu": 64},
	}

	// Pre-create VMs to avoid allocation during benchmark
	testVMs := make([]map[string]interface{}, 100)
	for i := 0; i < 100; i++ {
		testVMs[i] = map[string]interface{}{
			"id":        fmt.Sprintf("concurrent-vm-%d", i),
			"memory_gb": float64(4 + rand.Intn(28)),
			"cpu_cores": 2 + rand.Intn(14),
		}
	}

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		// Each goroutine gets a different index counter
		localIndex := 0
		for pb.Next() {
			// Use a different VM each time, cycling through the pre-created ones
			vm := testVMs[localIndex%100]
			localIndex++
			// Use the actual method from the engine
			_, _ = engine.EvaluatePlacementPolicies(testCtx, vm, testNodes)
		}
	})
}

// Test helper function to ensure benchmark setup is valid
func TestBenchmarkSetup(t *testing.T) {
	// Create a sample engine to verify setup works
	engine := setupBenchmarkEngine(&testing.B{})
	if engine == nil {
		t.Fatalf("Failed to set up benchmark engine")
	}

	// Verify policies are registered
	if len(engine.Policies) == 0 {
		t.Fatalf("No policies registered in benchmark engine")
	}

	// Verify contexts can be created
	ctx := createBenchmarkVMContext()
	if ctx == nil {
		t.Fatalf("Failed to create benchmark VM context")
	}

	// Verify policy evaluation works
	testCtx := context.Background()
	vm := map[string]interface{}{"id": "test-vm", "memory_gb": 8, "cpu_cores": 4}
	nodes := []map[string]interface{}{{"id": "test-node", "available_memory": 64, "available_cpu": 16}}

	result, err := engine.EvaluatePlacementPolicies(testCtx, vm, nodes)
	if err != nil {
		t.Fatalf("Policy evaluation failed: %v", err)
	}

	t.Logf("Policy evaluation result: %d filtered nodes", len(result))
}
