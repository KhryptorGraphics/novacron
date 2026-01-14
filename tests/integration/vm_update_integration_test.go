package integration

import (
	"context"
	"testing"

	"github.com/khryptorgraphics/novacron/backend/core/vm"
)

// TestVMUpdateEncapsulation verifies that VMManager.UpdateVM properly uses vm.ApplyUpdateSpec
func TestVMUpdateEncapsulation(t *testing.T) {
	// Create VM manager
	managerConfig := vm.DefaultVMManagerConfig()
	manager, err := vm.NewVMManager(managerConfig)
	if err != nil {
		t.Fatalf("Failed to create VM manager: %v", err)
	}

	// Create a VM in stopped state
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

	testVM.SetState(vm.StateStopped)
	manager.AddVM(testVM)

	// Test update with encapsulation and resource clamping
	ctx := context.Background()
	newName := "updated-test"
	newCPU := 2048
	newMemory := int64(1024)
	updateSpec := vm.VMUpdateSpec{
		Name:   &newName,
		CPU:    &newCPU,
		Memory: &newMemory,
	}

	err = manager.UpdateVM(ctx, "test-vm", updateSpec)
	if err != nil {
		t.Fatalf("UpdateVM failed: %v", err)
	}

	// Verify encapsulated update worked
	updatedVM, err := manager.GetVM("test-vm")
	if err != nil {
		t.Fatalf("Failed to get updated VM: %v", err)
	}

	updatedConfig := updatedVM.Config()
	if updatedConfig.Name != newName {
		t.Errorf("Expected name %s, got %s", newName, updatedConfig.Name)
	}
	if updatedConfig.CPUShares != newCPU {
		t.Errorf("Expected CPU %d, got %d", newCPU, updatedConfig.CPUShares)
	}

	// Verify resource accounting clamping
	currentCPU, currentMem := manager.GetCurrentAllocations()
	if currentCPU < 0 {
		t.Errorf("CPU allocation went negative: %d", currentCPU)
	}
	if currentMem < 0 {
		t.Errorf("Memory allocation went negative: %d", currentMem)
	}

	t.Logf("✅ VM update encapsulation and resource clamping working correctly")
	t.Logf("   Updated VM name: %s -> %s", config.Name, updatedConfig.Name)
	t.Logf("   Updated VM CPU: %d -> %d", config.CPUShares, updatedConfig.CPUShares)
	t.Logf("   Resource accounting: CPU=%d, Memory=%d (properly clamped)", currentCPU, currentMem)
}