package verification

import (
	"context"
	"testing"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/vm"
)

func TestVMUpdateEncapsulation(t *testing.T) {
	// Test that VM.ApplyUpdateSpec properly encapsulates config updates
	config := vm.VMConfig{
		ID:        "test-vm",
		Name:      "test",
		Command:   "sleep",
		Args:      []string{"3600"},
		CPUShares: 1024,
		MemoryMB:  512,
		Type:      vm.VMTypeProcess,
	}

	testVM, err := vm.NewVM(config)
	if err != nil {
		t.Fatalf("Failed to create VM: %v", err)
	}

	// Apply update spec
	newName := "updated-test"
	newCPU := 2048
	newMemory := int64(1024)
	updateSpec := vm.VMUpdateSpec{
		Name:   &newName,
		CPU:    &newCPU,
		Memory: &newMemory,
		Tags:   map[string]string{"test": "value"},
	}

	err = testVM.ApplyUpdateSpec(updateSpec)
	if err != nil {
		t.Fatalf("Failed to apply update spec: %v", err)
	}

	// Verify updates were applied
	updatedConfig := testVM.Config()
	if updatedConfig.Name != newName {
		t.Errorf("Expected name %s, got %s", newName, updatedConfig.Name)
	}
	if updatedConfig.CPUShares != newCPU {
		t.Errorf("Expected CPU %d, got %d", newCPU, updatedConfig.CPUShares)
	}
	if updatedConfig.MemoryMB != int(newMemory) {
		t.Errorf("Expected memory %d, got %d", int(newMemory), updatedConfig.MemoryMB)
	}
	if updatedConfig.Tags["test"] != "value" {
		t.Errorf("Expected tag test=value, got %v", updatedConfig.Tags)
	}

	t.Log("VM update encapsulation test passed")
}

func TestVMManagerUpdateWithResourceClamping(t *testing.T) {
	// Test that VMManager.UpdateVM uses encapsulation and clamps negative resources
	managerConfig := vm.DefaultVMManagerConfig()
	manager, err := vm.NewVMManager(managerConfig)
	if err != nil {
		t.Fatalf("Failed to create VM manager: %v", err)
	}

	// Create a VM
	config := vm.VMConfig{
		ID:        "test-vm-2",
		Name:      "test-vm-2",
		Command:   "sleep",
		Args:      []string{"3600"},
		CPUShares: 2048,
		MemoryMB:  1024,
		Type:      vm.VMTypeProcess,
	}

	testVM, err := vm.NewVM(config)
	if err != nil {
		t.Fatalf("Failed to create VM: %v", err)
	}

	// Set VM to stopped state for update
	testVM.SetState(vm.StateStopped)
	
	// Add VM to manager
	manager.AddVM(testVM)

	// Test update with smaller resources (should work and clamp negative accounting)
	ctx := context.Background()
	newCPU := 512      // Reduce from 2048
	newMemory := int64(256) // Reduce from 1024
	updateSpec := vm.VMUpdateSpec{
		CPU:    &newCPU,
		Memory: &newMemory,
	}

	err = manager.UpdateVM(ctx, "test-vm-2", updateSpec)
	if err != nil {
		t.Fatalf("UpdateVM failed: %v", err)
	}

	// Verify config was updated through encapsulation
	updatedVM, err := manager.GetVM("test-vm-2")
	if err != nil {
		t.Fatalf("Failed to get updated VM: %v", err)
	}

	updatedConfig := updatedVM.Config()
	if updatedConfig.CPUShares != newCPU {
		t.Errorf("Expected CPU %d, got %d", newCPU, updatedConfig.CPUShares)
	}
	if updatedConfig.MemoryMB != int(newMemory) {
		t.Errorf("Expected memory %d, got %d", int(newMemory), updatedConfig.MemoryMB)
	}

	// Verify resource accounting doesn't go negative (clamping)
	currentCPU, currentMem := manager.GetCurrentAllocations()
	if currentCPU < 0 {
		t.Errorf("CPU allocation went negative: %d", currentCPU)
	}
	if currentMem < 0 {
		t.Errorf("Memory allocation went negative: %d", currentMem)
	}

	t.Log("VM manager update with resource clamping test passed")
}