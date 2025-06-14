package vm

import (
	"context"
	"os"
	"testing"
	"time"
)

// TestVMLifecycleMinimal tests VM lifecycle operations without external dependencies
func TestVMLifecycleMinimal(t *testing.T) {
	// Create a test VM configuration
	config := VMConfig{
		ID:        "test-vm-minimal",
		Name:      "test-vm-minimal",
		Command:   "/bin/sleep",
		Args:      []string{"10"},
		CPUShares: 1024,
		MemoryMB:  512,
		RootFS:    "/tmp",
		Tags: map[string]string{
			"test": "minimal",
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

	// Test basic properties
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

	// Wait for process to stabilize
	time.Sleep(1 * time.Second)

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

// TestVMCloneMinimal tests VM cloning without external dependencies
func TestVMCloneMinimal(t *testing.T) {
	config := VMConfig{
		ID:        "test-vm-clone-original",
		Name:      "original-vm",
		Command:   "/bin/echo",
		Args:      []string{"hello"},
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

// TestVMResourceManagementMinimal tests resource management without external dependencies
func TestVMResourceManagementMinimal(t *testing.T) {
	config := VMConfig{
		ID:        "test-vm-resources-minimal",
		Name:      "resource-test-vm",
		Command:   "/bin/echo",
		Args:      []string{"test"},
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

// TestVMRebootMinimal tests VM reboot without external dependencies
func TestVMRebootMinimal(t *testing.T) {
	config := VMConfig{
		ID:        "test-vm-reboot-minimal",
		Name:      "test-vm-reboot",
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

	// Start the VM
	err = vm.Start()
	if err != nil {
		t.Fatalf("Failed to start VM: %v", err)
	}

	// Wait for stable state
	time.Sleep(1 * time.Second)

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

// TestKVMDriverEnhancedMinimal tests KVM driver without external dependencies
func TestKVMDriverEnhancedMinimal(t *testing.T) {
	// Skip if qemu-system-x86_64 is not available
	if _, err := os.Stat("/usr/bin/qemu-system-x86_64"); os.IsNotExist(err) {
		t.Skip("QEMU not available, skipping KVM driver test")
	}

	// Create KVM driver
	driver, err := NewKVMDriverEnhanced("")
	if err != nil {
		t.Fatalf("Failed to create KVM driver: %v", err)
	}

	config := VMConfig{
		ID:        "kvm-test-vm",
		Name:      "kvm-test",
		Command:   "/bin/echo",
		Args:      []string{"test"},
		CPUShares: 1024,
		MemoryMB:  512,
		RootFS:    "/tmp",
	}

	ctx := context.Background()

	// Test VM creation
	vmID, err := driver.Create(ctx, config)
	if err != nil {
		t.Fatalf("Failed to create VM through KVM driver: %v", err)
	}

	if vmID != config.ID {
		t.Errorf("Expected VM ID %s, got %s", config.ID, vmID)
	}

	// Test getting VM status
	status, err := driver.GetStatus(ctx, vmID)
	if err != nil {
		t.Fatalf("Failed to get VM status: %v", err)
	}

	if status != VMState(StateCreated) {
		t.Errorf("Expected VM status %s, got %s", StateCreated, status)
	}

	// Test getting VM info
	info, err := driver.GetInfo(ctx, vmID)
	if err != nil {
		t.Fatalf("Failed to get VM info: %v", err)
	}

	if info.ID != vmID {
		t.Errorf("Expected VM info ID %s, got %s", vmID, info.ID)
	}

	// Test listing VMs
	vms, err := driver.ListVMs(ctx)
	if err != nil {
		t.Fatalf("Failed to list VMs: %v", err)
	}

	if len(vms) != 1 {
		t.Errorf("Expected 1 VM in list, got %d", len(vms))
	}

	// Test creating snapshot
	snapshotID, err := driver.Snapshot(ctx, vmID, "test-snapshot", nil)
	if err != nil {
		t.Fatalf("Failed to create snapshot: %v", err)
	}

	if snapshotID == "" {
		t.Error("Expected non-empty snapshot ID")
	}

	// Test VM deletion
	err = driver.Delete(ctx, vmID)
	if err != nil {
		t.Fatalf("Failed to delete VM: %v", err)
	}

	// Verify VM was deleted
	_, err = driver.GetStatus(ctx, vmID)
	if err == nil {
		t.Error("Expected error when getting status of deleted VM")
	}
}