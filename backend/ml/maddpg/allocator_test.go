package maddpg

import (
	"os"
	"path/filepath"
	"testing"
)

func TestNodeGetObservation(t *testing.T) {
	node := &Node{
		ID:                0,
		CPUCapacity:       100.0,
		MemoryCapacity:    64.0,
		BandwidthCapacity: 1000.0,
		StorageCapacity:   500.0,
		CPUUsage:          50.0,
		MemoryUsage:       32.0,
		BandwidthUsage:    500.0,
		StorageUsage:      250.0,
	}

	obs := node.GetObservation()

	if len(obs) != 8 {
		t.Errorf("Expected observation length 8, got %d", len(obs))
	}

	// Check CPU utilization
	expectedCPUUtil := 0.5
	if !floatEquals(obs[0], expectedCPUUtil, 1e-6) {
		t.Errorf("Expected CPU utilization %.2f, got %.2f", expectedCPUUtil, obs[0])
	}

	// Check memory utilization
	expectedMemUtil := 0.5
	if !floatEquals(obs[1], expectedMemUtil, 1e-6) {
		t.Errorf("Expected memory utilization %.2f, got %.2f", expectedMemUtil, obs[1])
	}

	// Check CPU available
	expectedCPUAvail := 0.5
	if !floatEquals(obs[4], expectedCPUAvail, 1e-6) {
		t.Errorf("Expected CPU available %.2f, got %.2f", expectedCPUAvail, obs[4])
	}
}

func TestMADDPGModelInitialization(t *testing.T) {
	// Skip if model doesn't exist
	modelPath := "./models/maddpg/test"
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skip("Test model not found, skipping")
	}

	model, err := NewMADDPGModel(modelPath, 5)
	if err != nil {
		t.Fatalf("Failed to create MADDPG model: %v", err)
	}

	if model.NumAgents != 5 {
		t.Errorf("Expected 5 agents, got %d", model.NumAgents)
	}

	if model.StateDim != 8 {
		t.Errorf("Expected state dim 8, got %d", model.StateDim)
	}

	if model.ActionDim != 4 {
		t.Errorf("Expected action dim 4, got %d", model.ActionDim)
	}
}

func TestMADDPGModelPredict(t *testing.T) {
	// Skip if model doesn't exist
	modelPath := "./models/maddpg/test"
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skip("Test model not found, skipping")
	}

	model, err := NewMADDPGModel(modelPath, 3)
	if err != nil {
		t.Fatalf("Failed to create MADDPG model: %v", err)
	}

	// Create test states
	states := [][]float64{
		{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5},
		{0.3, 0.3, 0.3, 0.3, 0.7, 0.7, 0.7, 0.7},
		{0.8, 0.8, 0.8, 0.8, 0.2, 0.2, 0.2, 0.2},
	}

	actions, err := model.Predict(states)
	if err != nil {
		t.Fatalf("Prediction failed: %v", err)
	}

	if len(actions) != 3 {
		t.Errorf("Expected 3 actions, got %d", len(actions))
	}

	for i, action := range actions {
		if len(action) != 4 {
			t.Errorf("Agent %d: expected action dim 4, got %d", i, len(action))
		}

		// Check actions are in [0, 1]
		for j, val := range action {
			if val < 0.0 || val > 1.0 {
				t.Errorf("Agent %d action %d out of bounds: %.2f", i, j, val)
			}
		}
	}
}

func TestResourceAllocatorCreation(t *testing.T) {
	// Create test nodes
	nodes := []*Node{
		{ID: 0, CPUCapacity: 100, MemoryCapacity: 64, BandwidthCapacity: 1000, StorageCapacity: 500},
		{ID: 1, CPUCapacity: 150, MemoryCapacity: 96, BandwidthCapacity: 1500, StorageCapacity: 750},
		{ID: 2, CPUCapacity: 120, MemoryCapacity: 80, BandwidthCapacity: 1200, StorageCapacity: 600},
	}

	// Skip if model doesn't exist
	modelPath := "./models/maddpg/test"
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skip("Test model not found, skipping")
	}

	allocator, err := NewResourceAllocator(modelPath, nodes)
	if err != nil {
		t.Fatalf("Failed to create allocator: %v", err)
	}

	if len(allocator.nodes) != 3 {
		t.Errorf("Expected 3 nodes, got %d", len(allocator.nodes))
	}

	metrics := allocator.GetMetrics()
	if metrics.TotalAllocations != 0 {
		t.Errorf("Expected 0 initial allocations, got %d", metrics.TotalAllocations)
	}
}

func TestResourceAllocation(t *testing.T) {
	// Create test nodes
	nodes := []*Node{
		{ID: 0, CPUCapacity: 100, MemoryCapacity: 64, BandwidthCapacity: 1000, StorageCapacity: 500},
		{ID: 1, CPUCapacity: 150, MemoryCapacity: 96, BandwidthCapacity: 1500, StorageCapacity: 750},
		{ID: 2, CPUCapacity: 120, MemoryCapacity: 80, BandwidthCapacity: 1200, StorageCapacity: 600},
	}

	// Skip if model doesn't exist
	modelPath := "./models/maddpg/test"
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skip("Test model not found, skipping")
	}

	allocator, err := NewResourceAllocator(modelPath, nodes)
	if err != nil {
		t.Fatalf("Failed to create allocator: %v", err)
	}

	// Create test workloads
	workloads := []Workload{
		{ID: 1, CPURequirement: 20, MemoryRequirement: 8, BandwidthRequirement: 100, StorageRequirement: 50, Priority: 1.0},
		{ID: 2, CPURequirement: 30, MemoryRequirement: 16, BandwidthRequirement: 200, StorageRequirement: 100, Priority: 2.0},
		{ID: 3, CPURequirement: 25, MemoryRequirement: 12, BandwidthRequirement: 150, StorageRequirement: 75, Priority: 1.5},
	}

	allocations, err := allocator.AllocateResources(workloads)
	if err != nil {
		t.Fatalf("Allocation failed: %v", err)
	}

	// Should allocate all workloads
	if len(allocations) != 3 {
		t.Errorf("Expected 3 allocations, got %d", len(allocations))
	}

	// Check allocation validity
	for _, alloc := range allocations {
		if alloc.NodeID < 0 || alloc.NodeID >= len(nodes) {
			t.Errorf("Invalid node ID: %d", alloc.NodeID)
		}

		if alloc.CPUAlloc <= 0 {
			t.Errorf("Invalid CPU allocation: %.2f", alloc.CPUAlloc)
		}
	}

	// Check metrics
	metrics := allocator.GetMetrics()
	if metrics.TotalAllocations != 3 {
		t.Errorf("Expected 3 total allocations, got %d", metrics.TotalAllocations)
	}

	if metrics.SuccessfulAllocs < 1 {
		t.Errorf("Expected at least 1 successful allocation, got %d", metrics.SuccessfulAllocs)
	}
}

func TestAllocationMetrics(t *testing.T) {
	nodes := []*Node{
		{ID: 0, CPUCapacity: 100, MemoryCapacity: 64, BandwidthCapacity: 1000, StorageCapacity: 500},
	}

	modelPath := "./models/maddpg/test"
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skip("Test model not found, skipping")
	}

	allocator, err := NewResourceAllocator(modelPath, nodes)
	if err != nil {
		t.Fatalf("Failed to create allocator: %v", err)
	}

	// Allocate some workloads
	workloads := []Workload{
		{ID: 1, CPURequirement: 20, MemoryRequirement: 8, BandwidthRequirement: 100, StorageRequirement: 50},
		{ID: 2, CPURequirement: 200, MemoryRequirement: 160, BandwidthRequirement: 2000, StorageRequirement: 1000}, // Too big
	}

	_, err = allocator.AllocateResources(workloads)
	if err != nil {
		t.Fatalf("Allocation failed: %v", err)
	}

	metrics := allocator.GetMetrics()

	// Should have 1 success and 1 failure
	expectedTotal := 2
	if metrics.TotalAllocations != expectedTotal {
		t.Errorf("Expected %d total allocations, got %d", expectedTotal, metrics.TotalAllocations)
	}

	if metrics.SuccessfulAllocs+metrics.FailedAllocs != metrics.TotalAllocations {
		t.Errorf("Success + failed should equal total")
	}
}

func TestPerformanceReport(t *testing.T) {
	nodes := []*Node{
		{ID: 0, CPUCapacity: 100, MemoryCapacity: 64, BandwidthCapacity: 1000, StorageCapacity: 500},
	}

	modelPath := "./models/maddpg/test"
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skip("Test model not found, skipping")
	}

	allocator, err := NewResourceAllocator(modelPath, nodes)
	if err != nil {
		t.Fatalf("Failed to create allocator: %v", err)
	}

	report := allocator.PerformanceReport()

	// Check report has required fields
	requiredFields := []string{
		"total_allocations",
		"successful_allocs",
		"failed_allocs",
		"success_rate",
		"sla_violations",
		"sla_violation_rate",
		"avg_utilization",
		"num_nodes",
		"model_path",
	}

	for _, field := range requiredFields {
		if _, ok := report[field]; !ok {
			t.Errorf("Report missing required field: %s", field)
		}
	}
}

func TestAllocationHistory(t *testing.T) {
	nodes := []*Node{
		{ID: 0, CPUCapacity: 100, MemoryCapacity: 64, BandwidthCapacity: 1000, StorageCapacity: 500},
	}

	modelPath := "./models/maddpg/test"
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skip("Test model not found, skipping")
	}

	allocator, err := NewResourceAllocator(modelPath, nodes)
	if err != nil {
		t.Fatalf("Failed to create allocator: %v", err)
	}

	// Allocate some workloads
	workloads := []Workload{
		{ID: 1, CPURequirement: 20, MemoryRequirement: 8, BandwidthRequirement: 100, StorageRequirement: 50},
		{ID: 2, CPURequirement: 15, MemoryRequirement: 6, BandwidthRequirement: 80, StorageRequirement: 40},
	}

	_, err = allocator.AllocateResources(workloads)
	if err != nil {
		t.Fatalf("Allocation failed: %v", err)
	}

	history := allocator.GetAllocationHistory(10)

	if len(history) == 0 {
		t.Error("Expected non-empty allocation history")
	}

	// Check history entries are valid
	for _, alloc := range history {
		if alloc.WorkloadID == 0 {
			t.Error("Invalid workload ID in history")
		}
		if alloc.NodeID < 0 {
			t.Error("Invalid node ID in history")
		}
	}
}

// Helper function to compare floats
func floatEquals(a, b, epsilon float64) bool {
	diff := a - b
	if diff < 0 {
		diff = -diff
	}
	return diff < epsilon
}

// Benchmark tests
func BenchmarkResourceAllocation(b *testing.B) {
	nodes := []*Node{
		{ID: 0, CPUCapacity: 100, MemoryCapacity: 64, BandwidthCapacity: 1000, StorageCapacity: 500},
		{ID: 1, CPUCapacity: 150, MemoryCapacity: 96, BandwidthCapacity: 1500, StorageCapacity: 750},
		{ID: 2, CPUCapacity: 120, MemoryCapacity: 80, BandwidthCapacity: 1200, StorageCapacity: 600},
	}

	modelPath := "./models/maddpg/test"
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		b.Skip("Test model not found, skipping")
	}

	allocator, err := NewResourceAllocator(modelPath, nodes)
	if err != nil {
		b.Fatalf("Failed to create allocator: %v", err)
	}

	workloads := []Workload{
		{ID: 1, CPURequirement: 20, MemoryRequirement: 8, BandwidthRequirement: 100, StorageRequirement: 50},
		{ID: 2, CPURequirement: 30, MemoryRequirement: 16, BandwidthRequirement: 200, StorageRequirement: 100},
		{ID: 3, CPURequirement: 25, MemoryRequirement: 12, BandwidthRequirement: 150, StorageRequirement: 75},
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, err := allocator.AllocateResources(workloads)
		if err != nil {
			b.Fatalf("Allocation failed: %v", err)
		}
	}
}
