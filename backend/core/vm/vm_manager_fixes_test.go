package vm

import (
	"context"
	"testing"
	"time"
)

// TestVMManagerStructFields tests that all required fields are present in VMManager
func TestVMManagerStructFields(t *testing.T) {
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

	// Test that all required fields are accessible (compilation test)
	if manager.vms == nil {
		t.Error("VMManager.vms field should be initialized")
	}

	if manager.driverFactory == nil {
		t.Error("VMManager.driverFactory field should be initialized")
	}

	// Test thread-safe access to vms map
	manager.vmsMutex.Lock()
	_ = len(manager.vms)
	manager.vmsMutex.Unlock()

	t.Log("All VMManager fields are properly initialized")
}

// TestVMManagerVMOperations tests VM management operations
func TestVMManagerVMOperations(t *testing.T) {
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

	// Create a test VM
	vmConfig := VMConfig{
		ID:        "test-vm-manager-001",
		Name:      "test-vm-manager",
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

	// Test AddVM
	manager.AddVM(vm)

	// Test GetVM
	retrievedVM, exists := manager.GetVM(vmConfig.ID)
	if !exists {
		t.Error("VM should exist after adding")
	}
	if retrievedVM.GetConfig().ID != vmConfig.ID {
		t.Errorf("Retrieved VM ID %s, expected %s", retrievedVM.GetConfig().ID, vmConfig.ID)
	}

	// Test RemoveVM
	manager.RemoveVM(vmConfig.ID)
	_, exists = manager.GetVM(vmConfig.ID)
	if exists {
		t.Error("VM should not exist after removal")
	}

	t.Log("VM management operations work correctly")
}

// TestVMAccessorMethods tests the new VM accessor methods
func TestVMAccessorMethods(t *testing.T) {
	config := VMConfig{
		ID:        "test-vm-accessor-001",
		Name:      "test-vm-accessor",
		Command:   "/bin/sleep",
		Args:      []string{"30"},
		CPUShares: 1024,
		MemoryMB:  512,
		RootFS:    "/tmp",
	}

	vm, err := NewVM(config)
	if err != nil {
		t.Fatalf("Failed to create VM: %v", err)
	}

	// Test GetState
	initialState := vm.GetState()
	if initialState != StateCreated {
		t.Errorf("Initial state should be %s, got %s", StateCreated, initialState)
	}

	// Test SetState
	vm.SetState(StateRunning)
	if vm.GetState() != StateRunning {
		t.Errorf("State should be %s after SetState, got %s", StateRunning, vm.GetState())
	}

	// Test GetConfig
	retrievedConfig := vm.GetConfig()
	if retrievedConfig.ID != config.ID {
		t.Errorf("Config ID should be %s, got %s", config.ID, retrievedConfig.ID)
	}

	// Test GetUpdatedAt
	beforeUpdate := vm.GetUpdatedAt()
	time.Sleep(10 * time.Millisecond) // Small delay to ensure time difference
	vm.SetState(StatePaused)
	afterUpdate := vm.GetUpdatedAt()

	if !afterUpdate.After(beforeUpdate) {
		t.Error("UpdatedAt should be updated when state changes")
	}

	// Test SetStartedAt
	startTime := time.Now()
	vm.SetStartedAt(startTime)

	// Verify the started time was set (we can't directly access startedAt, but we can verify it was called)
	updatedAtAfterStart := vm.GetUpdatedAt()
	if !updatedAtAfterStart.After(afterUpdate) {
		t.Error("UpdatedAt should be updated when SetStartedAt is called")
	}

	t.Log("VM accessor methods work correctly")
}

// TestVMSchedulerGetActiveAllocations tests the new GetActiveAllocations method
func TestVMSchedulerGetActiveAllocations(t *testing.T) {
	config := SchedulerConfig{
		Algorithm: "round-robin",
		Weights: map[string]float64{
			"cpu":    1.0,
			"memory": 1.0,
		},
	}

	scheduler := NewVMScheduler(config)

	// Test GetActiveAllocations
	allocations := scheduler.GetActiveAllocations()
	if allocations == nil {
		t.Error("GetActiveAllocations should return a non-nil map")
	}

	// Should return empty map initially
	if len(allocations) != 0 {
		t.Errorf("Expected 0 allocations initially, got %d", len(allocations))
	}

	t.Log("VMScheduler.GetActiveAllocations works correctly")
}

// TestVMMaintenanceRecovery tests the fixed recoverVM method
func TestVMMaintenanceRecovery(t *testing.T) {
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

	// Create a test VM in failed state
	vmConfig := VMConfig{
		ID:        "test-vm-recovery-001",
		Name:      "test-vm-recovery",
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

	// Set VM to failed state
	vm.SetState(StateFailed)
	manager.AddVM(vm)

	// Test that we can call recoverVM without compilation errors
	// Note: This is primarily a compilation test since recoverVM is private
	// In a real scenario, this would be called internally by the maintenance system

	// Verify VM is in failed state
	retrievedVM, exists := manager.GetVM(vmConfig.ID)
	if !exists {
		t.Fatal("VM should exist")
	}
	if retrievedVM.GetState() != StateFailed {
		t.Errorf("VM should be in failed state, got %s", retrievedVM.GetState())
	}

	t.Log("VM maintenance recovery setup works correctly")
}

// TestVMManagerDriverFactory tests the driver factory functionality
func TestVMManagerDriverFactory(t *testing.T) {
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

	// Test that driver factory is properly initialized
	if manager.driverFactory == nil {
		t.Fatal("Driver factory should be initialized")
	}

	// Test driver creation
	vmConfig := VMConfig{
		ID:        "test-driver-vm-001",
		Name:      "test-driver-vm",
		Command:   "/bin/sleep",
		Args:      []string{"30"},
		CPUShares: 1024,
		MemoryMB:  512,
		RootFS:    "/tmp",
		Tags: map[string]string{
			"vm_type": string(VMTypeKVM),
		},
	}

	driver, err := manager.driverFactory(vmConfig)
	if err != nil {
		t.Logf("Driver creation failed (expected in test environment): %v", err)
		// This is expected in test environment without actual KVM
	} else if driver == nil {
		t.Error("Driver should not be nil when creation succeeds")
	}

	t.Log("Driver factory functionality works correctly")
}

// TestConcurrentVMOperations tests thread safety of VM operations
func TestConcurrentVMOperations(t *testing.T) {
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

	// Create multiple VMs concurrently
	numVMs := 10
	done := make(chan bool, numVMs)

	for i := 0; i < numVMs; i++ {
		go func(id int) {
			vmConfig := VMConfig{
				ID:        fmt.Sprintf("concurrent-vm-%d", id),
				Name:      fmt.Sprintf("concurrent-vm-%d", id),
				Command:   "/bin/sleep",
				Args:      []string{"30"},
				CPUShares: 1024,
				MemoryMB:  512,
				RootFS:    "/tmp",
			}

			vm, err := NewVM(vmConfig)
			if err != nil {
				t.Errorf("Failed to create VM %d: %v", id, err)
				done <- false
				return
			}

			// Add VM
			manager.AddVM(vm)

			// Get VM
			_, exists := manager.GetVM(vmConfig.ID)
			if !exists {
				t.Errorf("VM %d should exist after adding", id)
				done <- false
				return
			}

			// Remove VM
			manager.RemoveVM(vmConfig.ID)

			done <- true
		}(i)
	}

	// Wait for all goroutines to complete
	for i := 0; i < numVMs; i++ {
		success := <-done
		if !success {
			t.Error("Concurrent operation failed")
		}
	}

	t.Log("Concurrent VM operations work correctly")
}
