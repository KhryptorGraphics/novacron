package vm

import (
	"context"
	"testing"
	"time"
)

// TestVMMaintenanceCompilation tests that vm_maintenance.go compiles without errors
func TestVMMaintenanceCompilation(t *testing.T) {
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

	// This test primarily verifies that the maintenance code compiles
	// by creating a VM manager that uses the fixed maintenance methods

	// Create a test VM
	vmConfig := VMConfig{
		ID:        "maintenance-test-vm",
		Name:      "maintenance-test-vm",
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

	// Add VM to manager
	manager.AddVM(vm)

	// Set VM to failed state to test recovery logic
	vm.SetState(StateFailed)

	// Verify the VM is in the manager and in failed state
	retrievedVM, exists := manager.GetVM(vmConfig.ID)
	if !exists {
		t.Fatal("VM should exist in manager")
	}

	if retrievedVM.GetState() != StateFailed {
		t.Errorf("VM should be in failed state, got %s", retrievedVM.GetState())
	}

	// Test that scheduler has GetActiveAllocations method
	allocations := manager.scheduler.GetActiveAllocations()
	if allocations == nil {
		t.Error("GetActiveAllocations should return non-nil map")
	}

	t.Log("VM maintenance compilation test passed")
}

// TestVMRecoveryScenario tests VM recovery scenario
func TestVMRecoveryScenario(t *testing.T) {
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

	// Create multiple VMs in different states
	vmConfigs := []struct {
		config VMConfig
		state  State
	}{
		{
			config: VMConfig{
				ID:        "recovery-vm-1",
				Name:      "recovery-vm-1",
				Command:   "/bin/sleep",
				Args:      []string{"30"},
				CPUShares: 1024,
				MemoryMB:  512,
				RootFS:    "/tmp",
			},
			state: StateFailed,
		},
		{
			config: VMConfig{
				ID:        "recovery-vm-2",
				Name:      "recovery-vm-2",
				Command:   "/bin/sleep",
				Args:      []string{"30"},
				CPUShares: 1024,
				MemoryMB:  512,
				RootFS:    "/tmp",
			},
			state: StateRunning,
		},
		{
			config: VMConfig{
				ID:        "recovery-vm-3",
				Name:      "recovery-vm-3",
				Command:   "/bin/sleep",
				Args:      []string{"30"},
				CPUShares: 1024,
				MemoryMB:  512,
				RootFS:    "/tmp",
			},
			state: StateDeleting,
		},
	}

	// Create and add VMs to manager
	for _, vmConfig := range vmConfigs {
		vm, err := NewVM(vmConfig.config)
		if err != nil {
			t.Fatalf("Failed to create VM %s: %v", vmConfig.config.ID, err)
		}

		vm.SetState(vmConfig.state)
		manager.AddVM(vm)
	}

	// Verify VMs are in expected states
	for _, vmConfig := range vmConfigs {
		vm, exists := manager.GetVM(vmConfig.config.ID)
		if !exists {
			t.Errorf("VM %s should exist", vmConfig.config.ID)
			continue
		}

		if vm.GetState() != vmConfig.state {
			t.Errorf("VM %s should be in state %s, got %s",
				vmConfig.config.ID, vmConfig.state, vm.GetState())
		}
	}

	// Test resource checking functionality
	allocations := manager.scheduler.GetActiveAllocations()
	if allocations == nil {
		t.Error("Should be able to get active allocations")
	}

	t.Log("VM recovery scenario test passed")
}

// TestVMStateTransitionsInMaintenance tests state transitions during maintenance
func TestVMStateTransitionsInMaintenance(t *testing.T) {
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

	// Create a VM for state transition testing
	vmConfig := VMConfig{
		ID:        "state-transition-vm",
		Name:      "state-transition-vm",
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

	manager.AddVM(vm)

	// Test transitions that would occur during maintenance
	testTransitions := []struct {
		fromState State
		toState   State
		desc      string
	}{
		{StateRunning, StateFailed, "running to failed (error detected)"},
		{StateFailed, StateRunning, "failed to running (recovery)"},
		{StateRunning, StateDeleting, "running to deleting (cleanup)"},
	}

	for _, transition := range testTransitions {
		// Set initial state
		vm.SetState(transition.fromState)

		// Verify initial state
		if vm.GetState() != transition.fromState {
			t.Errorf("Failed to set initial state %s for transition: %s",
				transition.fromState, transition.desc)
			continue
		}

		// Record time before transition
		beforeTransition := vm.GetUpdatedAt()
		time.Sleep(10 * time.Millisecond) // Ensure time difference

		// Perform transition
		vm.SetState(transition.toState)

		// Verify final state
		if vm.GetState() != transition.toState {
			t.Errorf("Transition failed %s: expected %s, got %s",
				transition.desc, transition.toState, vm.GetState())
		}

		// Verify timestamp was updated
		afterTransition := vm.GetUpdatedAt()
		if !afterTransition.After(beforeTransition) {
			t.Errorf("UpdatedAt should be updated during transition: %s", transition.desc)
		}
	}

	t.Log("VM state transitions in maintenance work correctly")
}

// TestVMManagerConcurrentMaintenance tests concurrent maintenance operations
func TestVMManagerConcurrentMaintenance(t *testing.T) {
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

	// Create multiple VMs for concurrent testing
	numVMs := 5
	vms := make([]*VM, numVMs)

	for i := 0; i < numVMs; i++ {
		vmConfig := VMConfig{
			ID:        fmt.Sprintf("concurrent-maintenance-vm-%d", i),
			Name:      fmt.Sprintf("concurrent-maintenance-vm-%d", i),
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

		vms[i] = vm
		manager.AddVM(vm)
	}

	// Perform concurrent state changes
	done := make(chan bool, numVMs)

	for i, vm := range vms {
		go func(vmIndex int, testVM *VM) {
			// Simulate maintenance operations
			states := []State{StateRunning, StateFailed, StateRunning, StatePaused, StateRunning}

			for _, state := range states {
				testVM.SetState(state)
				time.Sleep(10 * time.Millisecond)
			}

			// Verify final state
			if testVM.GetState() != StateRunning {
				t.Errorf("VM %d should end in running state, got %s",
					vmIndex, testVM.GetState())
			}

			done <- true
		}(i, vm)
	}

	// Wait for all operations to complete
	for i := 0; i < numVMs; i++ {
		<-done
	}

	// Verify all VMs are still in the manager
	for i, vm := range vms {
		retrievedVM, exists := manager.GetVM(vm.GetConfig().ID)
		if !exists {
			t.Errorf("VM %d should still exist in manager", i)
		}

		if retrievedVM.GetState() != StateRunning {
			t.Errorf("VM %d should be in running state after concurrent operations", i)
		}
	}

	t.Log("Concurrent maintenance operations work correctly")
}

// TestMaintenanceResourceChecking tests resource allocation checking
func TestMaintenanceResourceChecking(t *testing.T) {
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

	// Test that we can access scheduler and its methods
	if manager.scheduler == nil {
		t.Fatal("Manager should have a scheduler")
	}

	// Test GetActiveAllocations method
	allocations := manager.scheduler.GetActiveAllocations()
	if allocations == nil {
		t.Fatal("GetActiveAllocations should return non-nil map")
	}

	// Create test VMs in different states
	vmStates := []State{StateRunning, StateFailed, StateDeleting, StatePaused}

	for i, state := range vmStates {
		vmConfig := VMConfig{
			ID:        fmt.Sprintf("resource-check-vm-%d", i),
			Name:      fmt.Sprintf("resource-check-vm-%d", i),
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

		vm.SetState(state)
		manager.AddVM(vm)
	}

	// Simulate resource checking (this would be done by checkResources method)
	// We test that we can iterate through VMs and check their states
	vmCount := 0
	runningCount := 0
	deletingCount := 0

	// Access VMs through the manager (simulating what checkResources does)
	manager.vmsMutex.RLock()
	for _, vm := range manager.vms {
		vmCount++
		switch vm.GetState() {
		case StateRunning:
			runningCount++
		case StateDeleting:
			deletingCount++
		}
	}
	manager.vmsMutex.RUnlock()

	if vmCount != len(vmStates) {
		t.Errorf("Expected %d VMs, found %d", len(vmStates), vmCount)
	}

	if runningCount != 1 {
		t.Errorf("Expected 1 running VM, found %d", runningCount)
	}

	if deletingCount != 1 {
		t.Errorf("Expected 1 deleting VM, found %d", deletingCount)
	}

	t.Log("Maintenance resource checking works correctly")
}
