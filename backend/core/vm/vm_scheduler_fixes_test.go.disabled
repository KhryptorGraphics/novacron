package vm

import (
	"context"
	"testing"
)

// TestVMSchedulerResourceAllocation tests the ResourceAllocation type
func TestVMSchedulerResourceAllocation(t *testing.T) {
	// Test ResourceAllocation struct creation
	allocation := ResourceAllocation{
		VMID:     "test-vm-001",
		NodeID:   "node-001",
		CPUCores: 2,
		MemoryMB: 1024,
		DiskGB:   50,
	}

	// Verify all fields are accessible
	if allocation.VMID != "test-vm-001" {
		t.Errorf("VMID should be 'test-vm-001', got '%s'", allocation.VMID)
	}

	if allocation.NodeID != "node-001" {
		t.Errorf("NodeID should be 'node-001', got '%s'", allocation.NodeID)
	}

	if allocation.CPUCores != 2 {
		t.Errorf("CPUCores should be 2, got %d", allocation.CPUCores)
	}

	if allocation.MemoryMB != 1024 {
		t.Errorf("MemoryMB should be 1024, got %d", allocation.MemoryMB)
	}

	if allocation.DiskGB != 50 {
		t.Errorf("DiskGB should be 50, got %d", allocation.DiskGB)
	}

	t.Log("ResourceAllocation struct works correctly")
}

// TestVMSchedulerGetActiveAllocationsIntegration tests GetActiveAllocations in context
func TestVMSchedulerGetActiveAllocationsIntegration(t *testing.T) {
	config := SchedulerConfig{
		Algorithm: "round-robin",
		Weights: map[string]float64{
			"cpu":    1.0,
			"memory": 1.0,
		},
	}

	scheduler := NewVMScheduler(config)

	// Register a test node
	nodeInfo := &NodeResourceInfo{
		NodeID:      "test-node-001",
		CPUCores:    8,
		MemoryMB:    8192,
		DiskGB:      500,
		Status:      "available",
		CPUUsage:    0.1,
		MemoryUsage: 0.2,
		DiskUsage:   0.3,
	}

	err := scheduler.RegisterNode(nodeInfo)
	if err != nil {
		t.Fatalf("Failed to register node: %v", err)
	}

	// Test GetActiveAllocations
	allocations := scheduler.GetActiveAllocations()
	if allocations == nil {
		t.Fatal("GetActiveAllocations should return a non-nil map")
	}

	// Should be empty initially
	if len(allocations) != 0 {
		t.Errorf("Expected 0 allocations initially, got %d", len(allocations))
	}

	// Test that we can add to the allocations map (simulating real usage)
	testAllocation := ResourceAllocation{
		VMID:     "test-vm-001",
		NodeID:   "test-node-001",
		CPUCores: 2,
		MemoryMB: 1024,
		DiskGB:   50,
	}

	allocations["test-vm-001"] = testAllocation

	// Verify the allocation was added
	if len(allocations) != 1 {
		t.Errorf("Expected 1 allocation after adding, got %d", len(allocations))
	}

	retrievedAllocation, exists := allocations["test-vm-001"]
	if !exists {
		t.Error("Allocation should exist after adding")
	}

	if retrievedAllocation.VMID != testAllocation.VMID {
		t.Errorf("Retrieved allocation VMID should be '%s', got '%s'",
			testAllocation.VMID, retrievedAllocation.VMID)
	}

	t.Log("VMScheduler GetActiveAllocations integration works correctly")
}

// TestVMSchedulerWithVMManager tests scheduler integration with VM manager
func TestVMSchedulerWithVMManager(t *testing.T) {
	config := VMManagerConfig{
		DefaultDriver: VMTypeKVM,
		Drivers: map[VMType]VMDriverConfig{
			VMTypeKVM: {
				Enabled: true,
				Config:  map[string]interface{}{},
			},
		},
		Scheduler: VMSchedulerConfig{
			Type:   "default",
			Config: map[string]interface{}{},
		},
	}

	manager, err := NewVMManager(config)
	if err != nil {
		t.Fatalf("Failed to create VM manager: %v", err)
	}
	defer manager.Shutdown()

	// Test that scheduler is accessible and has GetActiveAllocations method
	if manager.scheduler == nil {
		t.Fatal("VM manager should have a scheduler")
	}

	// Test GetActiveAllocations method exists and works
	allocations := manager.scheduler.GetActiveAllocations()
	if allocations == nil {
		t.Error("Scheduler GetActiveAllocations should return non-nil map")
	}

	t.Log("VMScheduler integration with VMManager works correctly")
}

// TestSchedulerNodeRegistration tests node registration functionality
func TestSchedulerNodeRegistration(t *testing.T) {
	config := SchedulerConfig{
		Algorithm: "round-robin",
		Weights: map[string]float64{
			"cpu":    1.0,
			"memory": 1.0,
		},
	}

	scheduler := NewVMScheduler(config)

	// Test registering multiple nodes
	nodes := []*NodeResourceInfo{
		{
			NodeID:      "node-001",
			CPUCores:    4,
			MemoryMB:    4096,
			DiskGB:      250,
			Status:      "available",
			CPUUsage:    0.1,
			MemoryUsage: 0.2,
			DiskUsage:   0.1,
		},
		{
			NodeID:      "node-002",
			CPUCores:    8,
			MemoryMB:    8192,
			DiskGB:      500,
			Status:      "available",
			CPUUsage:    0.2,
			MemoryUsage: 0.3,
			DiskUsage:   0.2,
		},
	}

	for _, node := range nodes {
		err := scheduler.RegisterNode(node)
		if err != nil {
			t.Errorf("Failed to register node %s: %v", node.NodeID, err)
		}
	}

	// Test that nodes are registered
	for _, node := range nodes {
		retrievedNode, err := scheduler.GetNodeResourceInfo(node.NodeID)
		if err != nil {
			t.Errorf("Failed to get node %s: %v", node.NodeID, err)
			continue
		}

		if retrievedNode.NodeID != node.NodeID {
			t.Errorf("Retrieved node ID should be %s, got %s",
				node.NodeID, retrievedNode.NodeID)
		}
	}

	t.Log("Scheduler node registration works correctly")
}

// TestSchedulerVMScheduling tests VM scheduling functionality
func TestSchedulerVMScheduling(t *testing.T) {
	config := SchedulerConfig{
		Algorithm: "round-robin",
		Weights: map[string]float64{
			"cpu":    1.0,
			"memory": 1.0,
		},
	}

	scheduler := NewVMScheduler(config)

	// Register a node
	nodeInfo := &NodeResourceInfo{
		NodeID:      "scheduler-test-node",
		CPUCores:    8,
		MemoryMB:    8192,
		DiskGB:      500,
		Status:      "available",
		CPUUsage:    0.1,
		MemoryUsage: 0.2,
		DiskUsage:   0.1,
	}

	err := scheduler.RegisterNode(nodeInfo)
	if err != nil {
		t.Fatalf("Failed to register node: %v", err)
	}

	// Create a test VM
	vmConfig := VMConfig{
		ID:        "scheduler-test-vm",
		Name:      "scheduler-test-vm",
		Command:   "/bin/sleep",
		Args:      []string{"30"},
		CPUShares: 1024,
		MemoryMB:  512,
		RootFS:    "/tmp",
	}

	vm, err := NewVM(vmConfig)
	if err != nil {
		t.Fatalf("Failed to create VM: %v", err)
	}

	// Test VM scheduling
	ctx := context.Background()
	selectedNodeID, err := scheduler.ScheduleVM(ctx, vm)
	if err != nil {
		t.Fatalf("Failed to schedule VM: %v", err)
	}

	if selectedNodeID != nodeInfo.NodeID {
		t.Errorf("Selected node should be %s, got %s", nodeInfo.NodeID, selectedNodeID)
	}

	t.Log("VM scheduling works correctly")
}
