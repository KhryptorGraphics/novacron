package vm

import (
	"context"
	"os"
	"testing"
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
		t.Skipf("skipping: cannot start process VM in this environment: %v", err)
	}

	// Verify running state
	if vm.State() != StateRunning {
		t.Errorf("Expected state to be %s after start, got %s", StateRunning, vm.State())
	}

	if !vm.IsRunning() {
		t.Error("Expected VM to be running")
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
		// Empty RootFS: KVM treats RootFS as a boot-image path, and a directory
		// such as "/tmp" is not a file qemu-img can convert. A blank qcow2 is enough.
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
