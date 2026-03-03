package vm

import (
	"context"
	"testing"
	"time"
)

func TestVMLifecycle(t *testing.T) {
	// Create a test VM configuration
	config := VMConfig{
		ID:        "test-vm-001",
		Name:      "test-vm",
		Command:   "/bin/sleep",
		Args:      []string{"30"},
		CPUShares: 1024,
		MemoryMB:  512,
		RootFS:    "/tmp",
		Tags: map[string]string{
			"test": "true",
		},
	}

	// Create VM
	vm, err := NewVM(config)
	if err != nil {
		t.Fatalf("Failed to create VM: %v", err)
	}

	// Verify initial state
	if vm.State() != StateCreated {
		t.Errorf("Expected initial state to be %s, got %s", StateCreated, vm.State())
	}

	if vm.ID() != config.ID {
		t.Errorf("Expected VM ID %s, got %s", config.ID, vm.ID())
	}

	if vm.Name() != config.Name {
		t.Errorf("Expected VM name %s, got %s", config.Name, vm.Name())
	}

	// Test VM start
	err = vm.Start()
	if err != nil {
		t.Fatalf("Failed to start VM: %v", err)
	}

	// Verify running state
	if vm.State() != StateRunning {
		t.Errorf("Expected state to be %s after start, got %s", StateRunning, vm.State())
	}

	if !vm.IsRunning() {
		t.Error("Expected VM to be running")
	}

	// Wait a moment for the process to be stable
	time.Sleep(2 * time.Second)

	// Test VM pause
	err = vm.Pause()
	if err != nil {
		t.Fatalf("Failed to pause VM: %v", err)
	}

	if vm.State() != StatePaused {
		t.Errorf("Expected state to be %s after pause, got %s", StatePaused, vm.State())
	}

	// Test VM resume
	err = vm.Resume()
	if err != nil {
		t.Fatalf("Failed to resume VM: %v", err)
	}

	if vm.State() != StateRunning {
		t.Errorf("Expected state to be %s after resume, got %s", StateRunning, vm.State())
	}

	// Test VM stop
	err = vm.Stop()
	if err != nil {
		t.Fatalf("Failed to stop VM: %v", err)
	}

	if vm.State() != StateStopped {
		t.Errorf("Expected state to be %s after stop, got %s", StateStopped, vm.State())
	}

	// Test cleanup
	err = vm.Cleanup()
	if err != nil {
		t.Fatalf("Failed to cleanup VM: %v", err)
	}
}

func TestVMReboot(t *testing.T) {
	config := VMConfig{
		ID:        "test-vm-reboot",
		Name:      "test-vm-reboot",
		Command:   "/bin/sleep",
		Args:      []string{"60"},
		CPUShares: 1024,
		MemoryMB:  512,
		RootFS:    "/tmp",
	}

	vm, err := NewVM(config)
	if err != nil {
		t.Fatalf("Failed to create VM: %v", err)
	}

	// Start the VM
	err = vm.Start()
	if err != nil {
		t.Fatalf("Failed to start VM: %v", err)
	}

	// Wait for stable state
	time.Sleep(2 * time.Second)

	// Test reboot
	err = vm.Reboot()
	if err != nil {
		t.Fatalf("Failed to reboot VM: %v", err)
	}

	// Verify VM is running after reboot
	if vm.State() != StateRunning {
		t.Errorf("Expected state to be %s after reboot, got %s", StateRunning, vm.State())
	}

	// Clean up
	vm.Stop()
	vm.Cleanup()
}

func TestVMClone(t *testing.T) {
	config := VMConfig{
		ID:        "test-vm-original",
		Name:      "original-vm",
		Command:   "/bin/sleep",
		Args:      []string{"30"},
		CPUShares: 1024,
		MemoryMB:  512,
		RootFS:    "/tmp",
		Tags: map[string]string{
			"environment": "test",
		},
	}

	originalVM, err := NewVM(config)
	if err != nil {
		t.Fatalf("Failed to create original VM: %v", err)
	}

	// Clone the VM
	clonedVM, err := originalVM.Clone("cloned-vm")
	if err != nil {
		t.Fatalf("Failed to clone VM: %v", err)
	}

	// Verify clone properties
	if clonedVM.Name() != "cloned-vm" {
		t.Errorf("Expected cloned VM name to be 'cloned-vm', got %s", clonedVM.Name())
	}

	if clonedVM.ID() == originalVM.ID() {
		t.Error("Cloned VM should have different ID from original")
	}

	// Verify cloned config
	clonedConfig := clonedVM.GetConfig()
	if clonedConfig.CPUShares != config.CPUShares {
		t.Errorf("Expected cloned VM CPU shares %d, got %d", config.CPUShares, clonedConfig.CPUShares)
	}

	if clonedConfig.MemoryMB != config.MemoryMB {
		t.Errorf("Expected cloned VM memory %d, got %d", config.MemoryMB, clonedConfig.MemoryMB)
	}

	// Verify clone tags include original tags plus clone metadata
	if clonedConfig.Tags["environment"] != "test" {
		t.Error("Cloned VM should inherit original tags")
	}

	if clonedConfig.Tags["cloned_from"] != originalVM.ID() {
		t.Error("Cloned VM should have clone metadata")
	}
}

func TestVMResourceManagement(t *testing.T) {
	config := VMConfig{
		ID:        "test-vm-resources",
		Name:      "resource-test-vm",
		Command:   "/bin/sleep",
		Args:      []string{"30"},
		CPUShares: 512,
		MemoryMB:  256,
		RootFS:    "/tmp",
	}

	vm, err := NewVM(config)
	if err != nil {
		t.Fatalf("Failed to create VM: %v", err)
	}

	// Test resource limit updates
	err = vm.UpdateResourceLimits(1024, 512)
	if err != nil {
		t.Fatalf("Failed to update resource limits: %v", err)
	}

	// Verify updated limits
	updatedConfig := vm.GetConfig()
	if updatedConfig.CPUShares != 1024 {
		t.Errorf("Expected CPU shares to be 1024, got %d", updatedConfig.CPUShares)
	}

	if updatedConfig.MemoryMB != 512 {
		t.Errorf("Expected memory to be 512MB, got %d", updatedConfig.MemoryMB)
	}

	// Test getting resource usage
	usage := vm.GetResourceUsage()
	if usage.CPUPercent < 0 {
		t.Error("CPU usage should not be negative")
	}
}

func TestVMManagerFixed(t *testing.T) {
	// Create VM manager
	config := DefaultVMManagerConfig()
	manager, err := NewVMManagerFixed(config, "test-node-1")
	if err != nil {
		t.Fatalf("Failed to create VM manager: %v", err)
	}

	// Start the manager
	err = manager.Start()
	if err != nil {
		t.Fatalf("Failed to start VM manager: %v", err)
	}
	defer manager.Stop()

	// Create a VM through the manager
	vmConfig := VMConfig{
		ID:        "manager-test-vm",
		Name:      "manager-test",
		Command:   "/bin/sleep",
		Args:      []string{"10"},
		CPUShares: 1024,
		MemoryMB:  512,
		RootFS:    "/tmp",
	}

	ctx := context.Background()
	vm, err := manager.CreateVM(ctx, vmConfig)
	if err != nil {
		t.Fatalf("Failed to create VM through manager: %v", err)
	}

	// Verify VM was created
	retrievedVM, err := manager.GetVM(vm.ID())
	if err != nil {
		t.Fatalf("Failed to retrieve VM from manager: %v", err)
	}

	if retrievedVM.ID() != vm.ID() {
		t.Errorf("Retrieved VM ID %s doesn't match created VM ID %s", retrievedVM.ID(), vm.ID())
	}

	// Test VM operations through manager
	err = manager.StartVM(ctx, vm.ID())
	if err != nil {
		t.Fatalf("Failed to start VM through manager: %v", err)
	}

	// Wait for stable state
	time.Sleep(2 * time.Second)

	// Verify VM is running
	info, err := manager.GetVMInfo(vm.ID())
	if err != nil {
		t.Fatalf("Failed to get VM info: %v", err)
	}

	if info.State != StateRunning {
		t.Errorf("Expected VM state to be %s, got %s", StateRunning, info.State)
	}

	// Test pause/resume through manager
	err = manager.PauseVM(ctx, vm.ID())
	if err != nil {
		t.Fatalf("Failed to pause VM through manager: %v", err)
	}

	err = manager.ResumeVM(ctx, vm.ID())
	if err != nil {
		t.Fatalf("Failed to resume VM through manager: %v", err)
	}

	// Test reboot through manager
	err = manager.RebootVM(ctx, vm.ID())
	if err != nil {
		t.Fatalf("Failed to reboot VM through manager: %v", err)
	}

	// Wait for reboot to complete
	time.Sleep(3 * time.Second)

	// Test stop through manager
	err = manager.StopVM(ctx, vm.ID())
	if err != nil {
		t.Fatalf("Failed to stop VM through manager: %v", err)
	}

	// Test delete through manager
	err = manager.DeleteVM(ctx, vm.ID())
	if err != nil {
		t.Fatalf("Failed to delete VM through manager: %v", err)
	}

	// Verify VM was deleted
	_, err = manager.GetVM(vm.ID())
	if err == nil {
		t.Error("Expected error when getting deleted VM")
	}
}

func TestVMEvents(t *testing.T) {
	config := DefaultVMManagerConfig()
	manager, err := NewVMManagerFixed(config, "test-node-events")
	if err != nil {
		t.Fatalf("Failed to create VM manager: %v", err)
	}

	err = manager.Start()
	if err != nil {
		t.Fatalf("Failed to start VM manager: %v", err)
	}
	defer manager.Stop()

	// Event tracking
	receivedEvents := make([]VMEvent, 0)
	eventListener := func(event VMEvent) {
		receivedEvents = append(receivedEvents, event)
	}

	manager.AddEventListener(eventListener)

	// Create VM to generate events
	vmConfig := VMConfig{
		ID:        "event-test-vm",
		Name:      "event-test",
		Command:   "/bin/sleep",
		Args:      []string{"5"},
		CPUShares: 1024,
		MemoryMB:  512,
		RootFS:    "/tmp",
	}

	ctx := context.Background()
	vm, err := manager.CreateVM(ctx, vmConfig)
	if err != nil {
		t.Fatalf("Failed to create VM: %v", err)
	}

	// Wait for events to be processed
	time.Sleep(100 * time.Millisecond)

	// Verify creation event was received
	if len(receivedEvents) == 0 {
		t.Error("Expected to receive VM creation event")
	} else {
		createEvent := receivedEvents[0]
		if createEvent.Type != VMEventCreated {
			t.Errorf("Expected first event to be %s, got %s", VMEventCreated, createEvent.Type)
		}
		if createEvent.VMID != vm.ID() {
			t.Errorf("Expected event VM ID %s, got %s", vm.ID(), createEvent.VMID)
		}
	}

	// Clean up
	manager.DeleteVM(ctx, vm.ID())
}

func TestVMListOperations(t *testing.T) {
	config := DefaultVMManagerConfig()
	manager, err := NewVMManagerFixed(config, "test-node-list")
	if err != nil {
		t.Fatalf("Failed to create VM manager: %v", err)
	}

	err = manager.Start()
	if err != nil {
		t.Fatalf("Failed to start VM manager: %v", err)
	}
	defer manager.Stop()

	ctx := context.Background()

	// Create multiple VMs
	vmConfigs := []VMConfig{
		{ID: "list-test-vm-1", Name: "vm1", Command: "/bin/sleep", Args: []string{"30"}, CPUShares: 1024, MemoryMB: 512, RootFS: "/tmp"},
		{ID: "list-test-vm-2", Name: "vm2", Command: "/bin/sleep", Args: []string{"30"}, CPUShares: 1024, MemoryMB: 512, RootFS: "/tmp"},
		{ID: "list-test-vm-3", Name: "vm3", Command: "/bin/sleep", Args: []string{"30"}, CPUShares: 1024, MemoryMB: 512, RootFS: "/tmp"},
	}

	createdVMs := make([]*VM, 0)
	for _, config := range vmConfigs {
		vm, err := manager.CreateVM(ctx, config)
		if err != nil {
			t.Fatalf("Failed to create VM %s: %v", config.Name, err)
		}
		createdVMs = append(createdVMs, vm)
	}

	// Test listing all VMs
	allVMs := manager.ListVMs()
	if len(allVMs) != 3 {
		t.Errorf("Expected 3 VMs, got %d", len(allVMs))
	}

	// Start one VM
	err = manager.StartVM(ctx, createdVMs[0].ID())
	if err != nil {
		t.Fatalf("Failed to start VM: %v", err)
	}

	time.Sleep(2 * time.Second)

	// Test listing VMs by state
	runningVMs := manager.ListVMsByState(StateRunning)
	if len(runningVMs) != 1 {
		t.Errorf("Expected 1 running VM, got %d", len(runningVMs))
	}

	createdVMs_state := manager.ListVMsByState(StateCreated)
	if len(createdVMs_state) != 2 {
		t.Errorf("Expected 2 created VMs, got %d", len(createdVMs_state))
	}

	// Clean up
	for _, vm := range createdVMs {
		manager.DeleteVM(ctx, vm.ID())
	}
}
