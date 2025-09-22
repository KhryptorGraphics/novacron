package integration

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/vm"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestVMOperations tests the newly implemented VM operations
func TestVMOperations(t *testing.T) {
	// Setup VM manager with test configuration
	config := vm.DefaultVMManagerConfig()
	config.Drivers[vm.VMTypeContainerd] = vm.VMDriverConfigManager{
		Enabled: true,
		Config: map[string]interface{}{
			"address":   "/tmp/test-containerd.sock",
			"namespace": "novacron-test",
		},
	}
	
	manager, err := vm.NewVMManager(config)
	require.NoError(t, err)
	
	ctx := context.Background()
	
	t.Run("UpdateVM", func(t *testing.T) {
		// Create a test VM
		createReq := vm.CreateVMRequest{
			Name: "test-vm-update",
			Spec: vm.VMConfig{
				Name:       "test-vm-update",
				CPUShares:  2,
				MemoryMB:   1024,
				DiskSizeGB: 10,
				Command:    "/bin/sh",
			},
		}
		
		testVM, err := manager.CreateVM(ctx, createReq)
		require.NoError(t, err)
		require.NotNil(t, testVM)
		
		// Test updating VM configuration
		updateSpec := vm.VMUpdateSpec{
			CPU:    &[]int{4}[0],
			Memory: &[]int64{2048}[0],
			Tags: map[string]string{
				"environment": "test",
				"purpose":     "integration-test",
			},
		}
		
		err = manager.UpdateVM(ctx, testVM.ID(), updateSpec)
		assert.NoError(t, err)
		
		// Verify updates were applied
		updatedVM, err := manager.GetVM(testVM.ID())
		require.NoError(t, err)
		config := updatedVM.Config()
		assert.Equal(t, 4, config.CPUShares)
		assert.Equal(t, 2048, config.MemoryMB)
		assert.Equal(t, "test", config.Tags["environment"])
		assert.Equal(t, "integration-test", config.Tags["purpose"])
		
		// Test validation errors
		t.Run("InvalidState", func(t *testing.T) {
			// Start VM to put it in running state
			err = manager.StartVM(ctx, testVM.ID())
			require.NoError(t, err)
			
			// Try to update running VM (should fail)
			err = manager.UpdateVM(ctx, testVM.ID(), updateSpec)
			assert.Error(t, err)
			assert.Contains(t, err.Error(), "INVALID_STATE")
			
			// Stop VM for cleanup
			err = manager.StopVM(ctx, testVM.ID())
			require.NoError(t, err)
		})
		
		t.Run("VMNotFound", func(t *testing.T) {
			err = manager.UpdateVM(ctx, "nonexistent-vm", updateSpec)
			assert.Error(t, err)
			assert.Contains(t, err.Error(), "VM_NOT_FOUND")
		})
		
		// Cleanup
		err = manager.DeleteVM(ctx, testVM.ID())
		assert.NoError(t, err)
	})
	
	t.Run("MigrateVM", func(t *testing.T) {
		// Create a test VM
		createReq := vm.CreateVMRequest{
			Name: "test-vm-migrate",
			Spec: vm.VMConfig{
				Name:       "test-vm-migrate",
				CPUShares:  2,
				MemoryMB:   1024,
				DiskSizeGB: 10,
				Command:    "/bin/sh",
			},
		}
		
		testVM, err := manager.CreateVM(ctx, createReq)
		require.NoError(t, err)
		require.NotNil(t, testVM)
		
		t.Run("UnsupportedDriver", func(t *testing.T) {
			// Test migration with containerd (which doesn't support migration)
			options := map[string]string{
				"migration_type": "live",
			}
			
			err = manager.MigrateVM(ctx, testVM.ID(), "target-node", options)
			assert.Error(t, err)
			assert.Contains(t, err.Error(), "OPERATION_NOT_SUPPORTED")
		})
		
		t.Run("VMNotFound", func(t *testing.T) {
			options := map[string]string{}
			err = manager.MigrateVM(ctx, "nonexistent-vm", "target-node", options)
			assert.Error(t, err)
			assert.Contains(t, err.Error(), "VM_NOT_FOUND")
		})
		
		// Cleanup
		err = manager.DeleteVM(ctx, testVM.ID())
		assert.NoError(t, err)
	})
	
	t.Run("CreateSnapshot", func(t *testing.T) {
		// Create a test VM
		createReq := vm.CreateVMRequest{
			Name: "test-vm-snapshot",
			Spec: vm.VMConfig{
				Name:       "test-vm-snapshot",
				CPUShares:  2,
				MemoryMB:   1024,
				DiskSizeGB: 10,
				Command:    "/bin/sh",
			},
		}
		
		testVM, err := manager.CreateVM(ctx, createReq)
		require.NoError(t, err)
		require.NotNil(t, testVM)
		
		t.Run("UnsupportedDriver", func(t *testing.T) {
			// Test snapshot with containerd (which doesn't support snapshots)
			options := map[string]string{
				"description": "test snapshot",
			}
			
			_, err = manager.CreateSnapshot(ctx, testVM.ID(), "test-snapshot", options)
			assert.Error(t, err)
			assert.Contains(t, err.Error(), "OPERATION_NOT_SUPPORTED")
		})
		
		t.Run("VMNotFound", func(t *testing.T) {
			options := map[string]string{}
			_, err = manager.CreateSnapshot(ctx, "nonexistent-vm", "snapshot", options)
			assert.Error(t, err)
			assert.Contains(t, err.Error(), "VM_NOT_FOUND")
		})
		
		// Cleanup
		err = manager.DeleteVM(ctx, testVM.ID())
		assert.NoError(t, err)
	})
}

// TestVMLifecycle tests basic VM lifecycle operations
func TestVMLifecycle(t *testing.T) {
	config := vm.DefaultVMManagerConfig()
	config.Drivers[vm.VMTypeContainerd] = vm.VMDriverConfigManager{
		Enabled: true,
		Config: map[string]interface{}{
			"address":   "/tmp/test-containerd.sock",
			"namespace": "novacron-test",
		},
	}
	
	manager, err := vm.NewVMManager(config)
	require.NoError(t, err)
	
	ctx := context.Background()
	
	// Create VM
	createReq := vm.CreateVMRequest{
		Name: "test-vm-lifecycle",
		Spec: vm.VMConfig{
			Name:       "test-vm-lifecycle",
			CPUShares:  2,
			MemoryMB:   1024,
			DiskSizeGB: 10,
			Command:    "/bin/sh",
		},
	}
	
	testVM, err := manager.CreateVM(ctx, createReq)
	require.NoError(t, err)
	assert.NotNil(t, testVM)
	assert.Equal(t, vm.StateStopped, testVM.State())
	
	// Start VM
	err = manager.StartVM(ctx, testVM.ID())
	assert.NoError(t, err)
	
	updatedVM, err := manager.GetVM(testVM.ID())
	require.NoError(t, err)
	assert.Equal(t, vm.StateRunning, updatedVM.State())
	
	// Pause VM
	err = manager.PauseVM(ctx, testVM.ID())
	assert.NoError(t, err)
	
	updatedVM, err = manager.GetVM(testVM.ID())
	require.NoError(t, err)
	assert.Equal(t, vm.StatePaused, updatedVM.State())
	
	// Resume VM
	err = manager.ResumeVM(ctx, testVM.ID())
	assert.NoError(t, err)
	
	updatedVM, err = manager.GetVM(testVM.ID())
	require.NoError(t, err)
	assert.Equal(t, vm.StateRunning, updatedVM.State())
	
	// Stop VM
	err = manager.StopVM(ctx, testVM.ID())
	assert.NoError(t, err)
	
	updatedVM, err = manager.GetVM(testVM.ID())
	require.NoError(t, err)
	assert.Equal(t, vm.StateStopped, updatedVM.State())
	
	// Restart VM
	err = manager.RestartVM(ctx, testVM.ID())
	assert.NoError(t, err)
	
	updatedVM, err = manager.GetVM(testVM.ID())
	require.NoError(t, err)
	assert.Equal(t, vm.StateRunning, updatedVM.State())
	
	// Stop and Delete VM
	err = manager.StopVM(ctx, testVM.ID())
	assert.NoError(t, err)
	
	err = manager.DeleteVM(ctx, testVM.ID())
	assert.NoError(t, err)
	
	// Verify VM is deleted
	_, err = manager.GetVM(testVM.ID())
	assert.Error(t, err)
}

// TestVMErrorHandling tests error scenarios
func TestVMErrorHandling(t *testing.T) {
	config := vm.DefaultVMManagerConfig()
	manager, err := vm.NewVMManager(config)
	require.NoError(t, err)
	
	ctx := context.Background()
	
	t.Run("OperationsOnNonExistentVM", func(t *testing.T) {
		vmID := "nonexistent-vm"
		
		err = manager.StartVM(ctx, vmID)
		assert.Error(t, err)
		
		err = manager.StopVM(ctx, vmID)
		assert.Error(t, err)
		
		err = manager.RestartVM(ctx, vmID)
		assert.Error(t, err)
		
		err = manager.PauseVM(ctx, vmID)
		assert.Error(t, err)
		
		err = manager.ResumeVM(ctx, vmID)
		assert.Error(t, err)
		
		err = manager.DeleteVM(ctx, vmID)
		assert.Error(t, err)
	})
	
	t.Run("InvalidStateTransitions", func(t *testing.T) {
		// Create and start a VM
		createReq := vm.CreateVMRequest{
			Name: "test-vm-states",
			Spec: vm.VMConfig{
				Name:       "test-vm-states",
				CPUShares:  1,
				MemoryMB:   512,
				Command:    "/bin/sh",
			},
		}
		
		testVM, err := manager.CreateVM(ctx, createReq)
		require.NoError(t, err)
		
		// Try to pause a stopped VM (should fail)
		err = manager.PauseVM(ctx, testVM.ID())
		assert.Error(t, err)
		
		// Try to resume a stopped VM (should fail)
		err = manager.ResumeVM(ctx, testVM.ID())
		assert.Error(t, err)
		
		// Start VM
		err = manager.StartVM(ctx, testVM.ID())
		assert.NoError(t, err)
		
		// Try to start an already running VM (should succeed - idempotent)
		err = manager.StartVM(ctx, testVM.ID())
		assert.NoError(t, err)
		
		// Cleanup
		err = manager.DeleteVM(ctx, testVM.ID())
		assert.NoError(t, err)
	})
}

// TestConcurrentVMOperations tests concurrent VM operations
func TestConcurrentVMOperations(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping concurrent operations test in short mode")
	}
	
	config := vm.DefaultVMManagerConfig()
	config.Drivers[vm.VMTypeContainerd] = vm.VMDriverConfigManager{
		Enabled: true,
		Config: map[string]interface{}{
			"address":   "/tmp/test-containerd.sock",
			"namespace": "novacron-test",
		},
	}
	
	manager, err := vm.NewVMManager(config)
	require.NoError(t, err)
	
	ctx := context.Background()
	numVMs := 5
	
	// Create multiple VMs concurrently
	createChan := make(chan error, numVMs)
	vmIDs := make([]string, numVMs)
	
	for i := 0; i < numVMs; i++ {
		go func(idx int) {
			createReq := vm.CreateVMRequest{
				Name: fmt.Sprintf("test-vm-concurrent-%d", idx),
				Spec: vm.VMConfig{
					Name:       fmt.Sprintf("test-vm-concurrent-%d", idx),
					CPUShares:  1,
					MemoryMB:   512,
					Command:    "/bin/sh",
				},
			}
			
			testVM, err := manager.CreateVM(ctx, createReq)
			if err != nil {
				createChan <- err
				return
			}
			
			vmIDs[idx] = testVM.ID()
			createChan <- nil
		}(i)
	}
	
	// Wait for all creations to complete
	for i := 0; i < numVMs; i++ {
		err := <-createChan
		assert.NoError(t, err)
	}
	
	// Start all VMs concurrently
	startChan := make(chan error, numVMs)
	for i := 0; i < numVMs; i++ {
		go func(vmID string) {
			startChan <- manager.StartVM(ctx, vmID)
		}(vmIDs[i])
	}
	
	// Wait for all starts to complete
	for i := 0; i < numVMs; i++ {
		err := <-startChan
		assert.NoError(t, err)
	}
	
	// Verify all VMs are running
	for i := 0; i < numVMs; i++ {
		vm, err := manager.GetVM(vmIDs[i])
		require.NoError(t, err)
		assert.Equal(t, vm.StateRunning, vm.State())
	}
	
	// Clean up all VMs concurrently
	deleteChan := make(chan error, numVMs)
	for i := 0; i < numVMs; i++ {
		go func(vmID string) {
			deleteChan <- manager.DeleteVM(ctx, vmID)
		}(vmIDs[i])
	}
	
	// Wait for all deletions to complete
	for i := 0; i < numVMs; i++ {
		err := <-deleteChan
		assert.NoError(t, err)
	}
}

// Helper function for concurrent operations
func runConcurrentOperation(t *testing.T, manager *vm.VMManager, vmID string, operation func() error) {
	done := make(chan error, 1)
	timeout := time.After(5 * time.Second)
	
	go func() {
		done <- operation()
	}()
	
	select {
	case err := <-done:
		if err != nil {
			t.Errorf("Operation failed: %v", err)
		}
	case <-timeout:
		t.Error("Operation timed out")
	}
}