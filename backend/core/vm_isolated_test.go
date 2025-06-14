package core

import (
	"context"
	"testing"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/storage"
	"github.com/khryptorgraphics/novacron/backend/core/vm"
)

// TestVMLifecycleIsolated tests VM lifecycle without any external dependencies
func TestVMLifecycleIsolated(t *testing.T) {
	// Create a test VM configuration
	config := vm.VMConfig{
		ID:        "test-vm-isolated",
		Name:      "test-vm",
		Command:   "/bin/sleep",
		Args:      []string{"5"},
		CPUShares: 1024,
		MemoryMB:  512,
		RootFS:    "/tmp",
		Tags: map[string]string{
			"test": "isolated",
		},
	}

	// Create VM
	testVM, err := vm.NewVM(config)
	if err != nil {
		t.Fatalf("Failed to create VM: %v", err)
	}

	// Verify initial state
	if testVM.State() != vm.StateCreated {
		t.Errorf("Expected initial state to be %s, got %s", vm.StateCreated, testVM.State())
	}

	// Test VM start
	err = testVM.Start()
	if err != nil {
		t.Fatalf("Failed to start VM: %v", err)
	}

	// Verify running state
	if testVM.State() != vm.StateRunning {
		t.Errorf("Expected state to be %s after start, got %s", vm.StateRunning, testVM.State())
	}

	// Wait for process to stabilize
	time.Sleep(1 * time.Second)

	// Test VM pause
	err = testVM.Pause()
	if err != nil {
		t.Fatalf("Failed to pause VM: %v", err)
	}

	if testVM.State() != vm.StatePaused {
		t.Errorf("Expected state to be %s after pause, got %s", vm.StatePaused, testVM.State())
	}

	// Test VM resume
	err = testVM.Resume()
	if err != nil {
		t.Fatalf("Failed to resume VM: %v", err)
	}

	if testVM.State() != vm.StateRunning {
		t.Errorf("Expected state to be %s after resume, got %s", vm.StateRunning, testVM.State())
	}

	// Test VM stop
	err = testVM.Stop()
	if err != nil {
		t.Fatalf("Failed to stop VM: %v", err)
	}

	if testVM.State() != vm.StateStopped {
		t.Errorf("Expected state to be %s after stop, got %s", vm.StateStopped, testVM.State())
	}

	// Test cleanup
	err = testVM.Cleanup()
	if err != nil {
		t.Fatalf("Failed to cleanup VM: %v", err)
	}

	t.Logf("VM lifecycle test completed successfully")
}

// TestVMManagerIsolated tests VM manager functionality
func TestVMManagerIsolated(t *testing.T) {
	// Create VM manager
	config := vm.DefaultVMManagerConfig()
	manager, err := vm.NewVMManagerFixed(config, "test-node-isolated")
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
	vmConfig := vm.VMConfig{
		ID:        "manager-test-vm-isolated",
		Name:      "manager-test",
		Command:   "/bin/sleep",
		Args:      []string{"3"},
		CPUShares: 1024,
		MemoryMB:  512,
		RootFS:    "/tmp",
	}

	ctx := context.Background()
	testVM, err := manager.CreateVM(ctx, vmConfig)
	if err != nil {
		t.Fatalf("Failed to create VM through manager: %v", err)
	}

	// Test VM operations through manager
	err = manager.StartVM(ctx, testVM.ID())
	if err != nil {
		t.Fatalf("Failed to start VM through manager: %v", err)
	}

	// Wait for stable state
	time.Sleep(1 * time.Second)

	// Verify VM is running
	info, err := manager.GetVMInfo(testVM.ID())
	if err != nil {
		t.Fatalf("Failed to get VM info: %v", err)
	}

	if info.State != vm.StateRunning {
		t.Errorf("Expected VM state to be %s, got %s", vm.StateRunning, info.State)
	}

	// Test stop and delete
	err = manager.StopVM(ctx, testVM.ID())
	if err != nil {
		t.Fatalf("Failed to stop VM through manager: %v", err)
	}

	err = manager.DeleteVM(ctx, testVM.ID())
	if err != nil {
		t.Fatalf("Failed to delete VM through manager: %v", err)
	}

	t.Logf("VM manager test completed successfully")
}

// TestVMStorageIntegrationIsolated tests VM with storage integration
func TestVMStorageIntegrationIsolated(t *testing.T) {
	// Create storage service
	storageConfig := storage.DefaultStorageConfig()
	storageService := storage.NewBaseStorageService(storageConfig)
	
	err := storageService.Start()
	if err != nil {
		t.Fatalf("Failed to start storage service: %v", err)
	}
	defer storageService.Stop()

	// Create VM storage manager
	storageManager := vm.NewVMStorageManager(storageService)

	// Create VM manager
	vmConfig := vm.DefaultVMManagerConfig()
	vmManager, err := vm.NewVMManagerFixed(vmConfig, "test-node-storage-isolated")
	if err != nil {
		t.Fatalf("Failed to create VM manager: %v", err)
	}

	err = vmManager.Start()
	if err != nil {
		t.Fatalf("Failed to start VM manager: %v", err)
	}
	defer vmManager.Stop()

	ctx := context.Background()

	// Test creating VM with storage
	vmCfg := vm.VMConfig{
		ID:        "storage-test-vm-isolated",
		Name:      "storage-test",
		Command:   "/bin/sleep",
		Args:      []string{"2"},
		CPUShares: 1024,
		MemoryMB:  512,
		RootFS:    "/tmp",
	}

	testVM, err := vmManager.CreateVMWithStorage(ctx, vmCfg, storageManager, 5, 2) // 5GB boot, 2GB data
	if err != nil {
		t.Fatalf("Failed to create VM with storage: %v", err)
	}

	// Verify VM was created
	if testVM.ID() != vmCfg.ID {
		t.Errorf("Expected VM ID %s, got %s", vmCfg.ID, testVM.ID())
	}

	// Test getting VM volumes
	volumes, err := storageManager.GetVMVolumes(ctx, testVM.ID())
	if err != nil {
		t.Fatalf("Failed to get VM volumes: %v", err)
	}

	if len(volumes) < 1 {
		t.Error("Expected at least 1 volume (boot volume)")
	}

	// Test getting storage stats
	stats, err := storageManager.GetVMStorageStats(ctx, testVM.ID())
	if err != nil {
		t.Fatalf("Failed to get VM storage stats: %v", err)
	}

	if stats.VMID != testVM.ID() {
		t.Errorf("Expected stats VM ID %s, got %s", testVM.ID(), stats.VMID)
	}

	if stats.VolumeCount == 0 {
		t.Error("Expected at least 1 volume in stats")
	}

	// Clean up
	err = vmManager.DeleteVMWithStorage(ctx, testVM.ID(), storageManager)
	if err != nil {
		t.Fatalf("Failed to delete VM with storage: %v", err)
	}

	t.Logf("VM storage integration test completed successfully")
}

// TestVMCloneIsolated tests VM cloning functionality
func TestVMCloneIsolated(t *testing.T) {
	config := vm.VMConfig{
		ID:        "test-vm-clone-original-isolated",
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

	originalVM, err := vm.NewVM(config)
	if err != nil {
		t.Fatalf("Failed to create original VM: %v", err)
	}

	// Clone the VM
	clonedVM, err := originalVM.Clone("cloned-vm-isolated")
	if err != nil {
		t.Fatalf("Failed to clone VM: %v", err)
	}

	// Verify clone properties
	if clonedVM.Name() != "cloned-vm-isolated" {
		t.Errorf("Expected cloned VM name to be 'cloned-vm-isolated', got %s", clonedVM.Name())
	}

	if clonedVM.ID() == originalVM.ID() {
		t.Error("Cloned VM should have different ID from original")
	}

	// Verify cloned config
	clonedConfig := clonedVM.GetConfig()
	if clonedConfig.CPUShares != config.CPUShares {
		t.Errorf("Expected cloned VM CPU shares %d, got %d", config.CPUShares, clonedConfig.CPUShares)
	}

	// Verify clone tags include original tags plus clone metadata
	if clonedConfig.Tags["environment"] != "test" {
		t.Error("Cloned VM should inherit original tags")
	}

	if clonedConfig.Tags["cloned_from"] != originalVM.ID() {
		t.Error("Cloned VM should have clone metadata")
	}

	t.Logf("VM clone test completed successfully")
}