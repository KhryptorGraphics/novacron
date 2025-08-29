package vm

import (
	"context"
	"testing"
	"time"
)

// Simple compilation test without external dependencies
func TestVMCompilation(t *testing.T) {
	// Test VM creation
	config := VMConfig{
		ID:        "compile-test-vm",
		Name:      "compile-test",
		Command:   "/bin/echo",
		Args:      []string{"hello"},
		CPUShares: 1024,
		MemoryMB:  512,
		RootFS:    "/tmp",
	}

	vm, err := NewVM(config)
	if err != nil {
		t.Fatalf("Failed to create VM: %v", err)
	}

	// Test basic VM methods
	if vm.ID() != config.ID {
		t.Errorf("Expected VM ID %s, got %s", config.ID, vm.ID())
	}

	if vm.Name() != config.Name {
		t.Errorf("Expected VM name %s, got %s", config.Name, vm.Name())
	}

	if vm.State() != StateCreated {
		t.Errorf("Expected initial state %s, got %s", StateCreated, vm.State())
	}

	// Test state changes
	vm.SetState(StateRunning)
	if vm.State() != StateRunning {
		t.Errorf("Expected state %s after SetState, got %s", StateRunning, vm.State())
	}

	// Test resource usage
	usage := vm.GetResourceUsage()
	if usage.CPUPercent < 0 {
		t.Error("CPU usage should not be negative")
	}

	// Test resource limits update
	err = vm.UpdateResourceLimits(2048, 1024)
	if err != nil {
		t.Fatalf("Failed to update resource limits: %v", err)
	}

	updatedConfig := vm.GetConfig()
	if updatedConfig.CPUShares != 2048 {
		t.Errorf("Expected updated CPU shares 2048, got %d", updatedConfig.CPUShares)
	}

	if updatedConfig.MemoryMB != 1024 {
		t.Errorf("Expected updated memory 1024MB, got %d", updatedConfig.MemoryMB)
	}

	// Test VM cloning
	clonedVM, err := vm.Clone("cloned-vm")
	if err != nil {
		t.Fatalf("Failed to clone VM: %v", err)
	}

	if clonedVM.Name() != "cloned-vm" {
		t.Errorf("Expected cloned VM name 'cloned-vm', got %s", clonedVM.Name())
	}

	if clonedVM.ID() == vm.ID() {
		t.Error("Cloned VM should have different ID from original")
	}

	// Test VM info
	info := vm.GetInfo()
	if info.ID != vm.ID() {
		t.Errorf("Expected info ID %s, got %s", vm.ID(), info.ID)
	}

	if info.Name != vm.Name() {
		t.Errorf("Expected info name %s, got %s", vm.Name(), info.Name)
	}
}

func TestVMManagerFixedCompilation(t *testing.T) {
	// Test VM manager creation
	config := DefaultVMManagerConfig()
	manager, err := NewVMManagerFixed(config, "test-node")
	if err != nil {
		t.Fatalf("Failed to create VM manager: %v", err)
	}

	// Test starting the manager
	err = manager.Start()
	if err != nil {
		t.Fatalf("Failed to start VM manager: %v", err)
	}
	defer manager.Stop()

	// Test VM creation through manager
	vmConfig := VMConfig{
		ID:        "manager-test-vm",
		Name:      "manager-test",
		Command:   "/bin/echo",
		Args:      []string{"test"},
		CPUShares: 1024,
		MemoryMB:  512,
		RootFS:    "/tmp",
	}

	ctx := context.Background()
	vm, err := manager.CreateVM(ctx, vmConfig)
	if err != nil {
		t.Fatalf("Failed to create VM through manager: %v", err)
	}

	// Test VM retrieval
	retrievedVM, err := manager.GetVM(vm.ID())
	if err != nil {
		t.Fatalf("Failed to retrieve VM: %v", err)
	}

	if retrievedVM.ID() != vm.ID() {
		t.Errorf("Expected retrieved VM ID %s, got %s", vm.ID(), retrievedVM.ID())
	}

	// Test VM listing
	allVMs := manager.ListVMs()
	if len(allVMs) != 1 {
		t.Errorf("Expected 1 VM in list, got %d", len(allVMs))
	}

	// Test VM info through manager
	info, err := manager.GetVMInfo(vm.ID())
	if err != nil {
		t.Fatalf("Failed to get VM info: %v", err)
	}

	if info.ID != vm.ID() {
		t.Errorf("Expected info ID %s, got %s", vm.ID(), info.ID)
	}

	// Test VM deletion
	err = manager.DeleteVM(ctx, vm.ID())
	if err != nil {
		t.Fatalf("Failed to delete VM: %v", err)
	}

	// Verify VM was deleted
	_, err = manager.GetVM(vm.ID())
	if err == nil {
		t.Error("Expected error when getting deleted VM")
	}
}

func TestVMEventSystem(t *testing.T) {
	config := DefaultVMManagerConfig()
	manager, err := NewVMManagerFixed(config, "event-test-node")
	if err != nil {
		t.Fatalf("Failed to create VM manager: %v", err)
	}

	err = manager.Start()
	if err != nil {
		t.Fatalf("Failed to start VM manager: %v", err)
	}
	defer manager.Stop()

	// Track events
	receivedEvents := make([]VMEvent, 0)
	eventListener := func(event VMEvent) {
		receivedEvents = append(receivedEvents, event)
	}

	manager.AddEventListener(eventListener)

	// Create VM to generate events
	vmConfig := VMConfig{
		ID:        "event-vm",
		Name:      "event-test",
		Command:   "/bin/echo",
		Args:      []string{"event"},
		CPUShares: 1024,
		MemoryMB:  512,
		RootFS:    "/tmp",
	}

	ctx := context.Background()
	vm, err := manager.CreateVM(ctx, vmConfig)
	if err != nil {
		t.Fatalf("Failed to create VM: %v", err)
	}

	// Wait for events
	time.Sleep(100 * time.Millisecond)

	// Verify creation event
	if len(receivedEvents) == 0 {
		t.Error("Expected to receive VM creation event")
	} else {
		event := receivedEvents[0]
		if event.Type != VMEventCreated {
			t.Errorf("Expected event type %s, got %s", VMEventCreated, event.Type)
		}
		if event.VMID != vm.ID() {
			t.Errorf("Expected event VM ID %s, got %s", vm.ID(), event.VMID)
		}
	}

	// Test removing event listener
	manager.RemoveEventListener(eventListener)

	// Clean up
	manager.DeleteVM(ctx, vm.ID())
}

func TestDefaultConfigurations(t *testing.T) {
	// Test default VM manager config
	config := DefaultVMManagerConfig()

	if config.UpdateInterval <= 0 {
		t.Error("Default update interval should be positive")
	}

	if config.CleanupInterval <= 0 {
		t.Error("Default cleanup interval should be positive")
	}

	if config.RetentionPeriod <= 0 {
		t.Error("Default retention period should be positive")
	}

	if config.DefaultVMType == "" {
		t.Error("Default VM type should not be empty")
	}
}

func TestVMLifecycleStates(t *testing.T) {
	config := VMConfig{
		ID:        "state-test-vm",
		Name:      "state-test",
		Command:   "/bin/echo",
		Args:      []string{"state"},
		CPUShares: 1024,
		MemoryMB:  512,
		RootFS:    "/tmp",
	}

	vm, err := NewVM(config)
	if err != nil {
		t.Fatalf("Failed to create VM: %v", err)
	}

	// Test state transitions
	states := []State{
		StateCreated,
		StateRunning,
		StatePaused,
		StateRunning,
		StateStopped,
		StateFailed,
	}

	for _, state := range states {
		vm.SetState(state)
		if vm.State() != state {
			t.Errorf("Expected state %s, got %s", state, vm.State())
		}
	}

	// Test IsRunning method
	vm.SetState(StateRunning)
	if !vm.IsRunning() {
		t.Error("Expected VM to be running")
	}

	vm.SetState(StateStopped)
	if vm.IsRunning() {
		t.Error("Expected VM to not be running")
	}
}
