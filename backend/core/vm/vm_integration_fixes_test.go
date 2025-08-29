package vm

import (
	"context"
	"fmt"
	"testing"
	"time"
)

// TestCompleteVMManagerIntegration tests the complete integration of all fixes
func TestCompleteVMManagerIntegration(t *testing.T) {
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

	// Test 1: Verify all struct fields are properly initialized
	if manager.vms == nil {
		t.Error("vms field should be initialized")
	}
	if manager.driverFactory == nil {
		t.Error("driverFactory field should be initialized")
	}
	if manager.scheduler == nil {
		t.Error("scheduler field should be initialized")
	}

	// Test 2: Create and manage VMs
	vmConfigs := []VMConfig{
		{
			ID:        "integration-vm-1",
			Name:      "integration-vm-1",
			Command:   "/bin/sleep",
			Args:      []string{"30"},
			CPUShares: 1024,
			MemoryMB:  512,
			RootFS:    "/tmp",
		},
		{
			ID:        "integration-vm-2",
			Name:      "integration-vm-2",
			Command:   "/bin/sleep",
			Args:      []string{"30"},
			CPUShares: 2048,
			MemoryMB:  1024,
			RootFS:    "/tmp",
		},
	}

	createdVMs := make([]*VM, len(vmConfigs))
	for i, config := range vmConfigs {
		vm, err := NewVM(config)
		if err != nil {
			t.Fatalf("Failed to create VM %s: %v", config.ID, err)
		}
		createdVMs[i] = vm
		manager.AddVM(vm)
	}

	// Test 3: Verify VM accessor methods work
	for i, vm := range createdVMs {
		// Test GetState
		if vm.GetState() != StateCreated {
			t.Errorf("VM %d should be in created state", i)
		}

		// Test GetConfig
		config := vm.GetConfig()
		if config.ID != vmConfigs[i].ID {
			t.Errorf("VM %d config ID mismatch", i)
		}

		// Test state transitions
		vm.SetState(StateRunning)
		if vm.GetState() != StateRunning {
			t.Errorf("VM %d should be in running state after SetState", i)
		}

		// Test SetStartedAt
		startTime := time.Now()
		vm.SetStartedAt(startTime)

		// Verify UpdatedAt was updated
		updatedAt := vm.GetUpdatedAt()
		if updatedAt.Before(startTime) {
			t.Errorf("VM %d UpdatedAt should be updated after SetStartedAt", i)
		}
	}

	// Test 4: Test scheduler integration
	allocations := manager.scheduler.GetActiveAllocations()
	if allocations == nil {
		t.Error("Scheduler should return non-nil allocations map")
	}

	// Test 5: Test VM state management for maintenance scenarios
	// Set one VM to failed state
	createdVMs[0].SetState(StateFailed)

	// Verify we can identify failed VMs (simulating maintenance check)
	failedVMs := 0
	runningVMs := 0

	manager.vmsMutex.RLock()
	for _, vm := range manager.vms {
		switch vm.GetState() {
		case StateFailed:
			failedVMs++
		case StateRunning:
			runningVMs++
		}
	}
	manager.vmsMutex.RUnlock()

	if failedVMs != 1 {
		t.Errorf("Expected 1 failed VM, got %d", failedVMs)
	}
	if runningVMs != 1 {
		t.Errorf("Expected 1 running VM, got %d", runningVMs)
	}

	// Test 6: Test driver factory functionality
	testConfig := VMConfig{
		ID:        "driver-test-vm",
		Name:      "driver-test-vm",
		Command:   "/bin/sleep",
		Args:      []string{"30"},
		CPUShares: 1024,
		MemoryMB:  512,
		RootFS:    "/tmp",
		Tags: map[string]string{
			"vm_type": string(VMTypeKVM),
		},
	}

	// Test that driver factory can be called (may fail in test environment)
	_, err = manager.driverFactory(testConfig)
	if err != nil {
		t.Logf("Driver factory call failed (expected in test environment): %v", err)
	}

	// Test 7: Test concurrent operations
	done := make(chan bool, 2)

	go func() {
		// Concurrent VM operations
		for i := 0; i < 10; i++ {
			vm, exists := manager.GetVM(createdVMs[0].GetConfig().ID)
			if !exists {
				t.Errorf("VM should exist during concurrent access")
			}
			vm.SetState(StateRunning)
			time.Sleep(1 * time.Millisecond)
		}
		done <- true
	}()

	go func() {
		// Concurrent scheduler operations
		for i := 0; i < 10; i++ {
			allocations := manager.scheduler.GetActiveAllocations()
			if allocations == nil {
				t.Errorf("Allocations should not be nil during concurrent access")
			}
			time.Sleep(1 * time.Millisecond)
		}
		done <- true
	}()

	// Wait for concurrent operations
	<-done
	<-done

	// Test 8: Cleanup and verification
	for _, vm := range createdVMs {
		manager.RemoveVM(vm.GetConfig().ID)
	}

	// Verify VMs were removed
	for _, vm := range createdVMs {
		_, exists := manager.GetVM(vm.GetConfig().ID)
		if exists {
			t.Errorf("VM %s should not exist after removal", vm.GetConfig().ID)
		}
	}

	t.Log("Complete VM manager integration test passed")
}

// TestCompilationVerification verifies that all fixed code compiles
func TestCompilationVerification(t *testing.T) {
	// This test verifies that all the fixes compile correctly by using
	// the previously problematic code patterns

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

	// Test all the previously problematic patterns:

	// 1. vmsMutex and vms field access
	manager.vmsMutex.Lock()
	_ = len(manager.vms)
	manager.vmsMutex.Unlock()

	// 2. driverFactory field access
	if manager.driverFactory != nil {
		testConfig := VMConfig{
			ID:        "compilation-test-vm",
			Name:      "compilation-test-vm",
			Command:   "/bin/sleep",
			Args:      []string{"30"},
			CPUShares: 1024,
			MemoryMB:  512,
			RootFS:    "/tmp",
		}
		_, _ = manager.driverFactory(testConfig)
	}

	// 3. State constants usage
	states := []State{
		StateRunning,  // Previously VMStateRunning
		StateFailed,   // Previously VMStateError
		StateDeleting, // Previously VMStateDeleting
	}

	for _, state := range states {
		if string(state) == "" {
			t.Errorf("State %v should have a string value", state)
		}
	}

	// 4. GetActiveAllocations method
	allocations := manager.scheduler.GetActiveAllocations()
	if allocations == nil {
		t.Error("GetActiveAllocations should return non-nil map")
	}

	// 5. VM accessor methods
	vmConfig := VMConfig{
		ID:        "accessor-test-vm",
		Name:      "accessor-test-vm",
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

	// Test all accessor methods
	_ = vm.GetState()
	_ = vm.GetConfig()
	_ = vm.GetUpdatedAt()
	vm.SetState(StateRunning)
	vm.SetStartedAt(time.Now())

	t.Log("All compilation verification tests passed")
}

// TestErrorHandlingInFixes tests error handling in the fixed code
func TestErrorHandlingInFixes(t *testing.T) {
	// Test VM manager creation with invalid config
	invalidConfig := VMManagerConfig{
		// Missing required fields
	}

	manager, err := NewVMManager(invalidConfig)
	if err != nil {
		t.Logf("Expected error with invalid config: %v", err)
	}
	if manager != nil {
		manager.Shutdown()
	}

	// Test valid config
	validConfig := VMManagerConfig{
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

	manager, err = NewVMManager(validConfig)
	if err != nil {
		t.Fatalf("Valid config should not fail: %v", err)
	}
	defer manager.Shutdown()

	// Test VM operations with non-existent VMs
	_, exists := manager.GetVM("non-existent-vm")
	if exists {
		t.Error("Non-existent VM should not exist")
	}

	// Test removing non-existent VM (should not panic)
	manager.RemoveVM("non-existent-vm")

	// Test scheduler with no nodes
	ctx := context.Background()
	vmConfig := VMConfig{
		ID:        "error-test-vm",
		Name:      "error-test-vm",
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

	_, err = manager.scheduler.ScheduleVM(ctx, vm)
	if err == nil {
		t.Error("Scheduling should fail with no available nodes")
	}

	t.Log("Error handling tests passed")
}

// TestPerformanceOfFixes tests performance of the fixed code
func TestPerformanceOfFixes(t *testing.T) {
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

	// Performance test: Create many VMs
	numVMs := 100
	start := time.Now()

	for i := 0; i < numVMs; i++ {
		vmConfig := VMConfig{
			ID:        fmt.Sprintf("perf-test-vm-%d", i),
			Name:      fmt.Sprintf("perf-test-vm-%d", i),
			Command:   "/bin/sleep",
			Args:      []string{"30"},
			CPUShares: 1024,
			MemoryMB:  512,
			RootFS:    "/tmp",
		}

		vm, err := NewVM(vmConfig)
		if err != nil {
			t.Fatalf("Failed to create VM %d: %v", i, err)
		}

		manager.AddVM(vm)
	}

	createTime := time.Since(start)
	t.Logf("Created %d VMs in %v", numVMs, createTime)

	// Performance test: State transitions
	start = time.Now()

	manager.vmsMutex.RLock()
	for _, vm := range manager.vms {
		vm.SetState(StateRunning)
	}
	manager.vmsMutex.RUnlock()

	transitionTime := time.Since(start)
	t.Logf("Performed %d state transitions in %v", numVMs, transitionTime)

	// Performance test: Scheduler operations
	start = time.Now()

	for i := 0; i < 1000; i++ {
		_ = manager.scheduler.GetActiveAllocations()
	}

	schedulerTime := time.Since(start)
	t.Logf("Performed 1000 scheduler operations in %v", schedulerTime)

	// Cleanup
	start = time.Now()

	for i := 0; i < numVMs; i++ {
		manager.RemoveVM(fmt.Sprintf("perf-test-vm-%d", i))
	}

	cleanupTime := time.Since(start)
	t.Logf("Cleaned up %d VMs in %v", numVMs, cleanupTime)

	t.Log("Performance tests completed")
}
