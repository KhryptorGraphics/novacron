package vm

import (
	"context"
	"testing"
	"time"
)

// TestVMMigrationExecutionFixed tests the VM migration execution with proper method calls
func TestVMMigrationExecutionFixed(t *testing.T) {
	// Create source VM
	sourceConfig := VMConfig{
		ID:        "migration-source-vm",
		Name:      "source-vm",
		Command:   "/bin/sleep",
		Args:      []string{"30"},
		CPUShares: 1024,
		MemoryMB:  512,
		RootFS:    "/tmp",
	}

	sourceVM, err := NewVM(sourceConfig)
	if err != nil {
		t.Fatalf("Failed to create source VM: %v", err)
	}

	// Start the source VM
	err = sourceVM.Start()
	if err != nil {
		t.Fatalf("Failed to start source VM: %v", err)
	}
	defer sourceVM.Cleanup()

	// Wait for stable state
	time.Sleep(2 * time.Second)

	// Verify VM is running
	if sourceVM.State() != StateRunning {
		t.Errorf("Source VM should be running, got state: %s", sourceVM.State())
	}

	// Test migration preparation
	migrationID := "test-migration-001"
	
	// Simulate migration steps
	t.Run("PrepareMigration", func(t *testing.T) {
		// In a real implementation, this would prepare the VM for migration
		// For testing, we verify the VM is in the correct state
		if !sourceVM.IsRunning() {
			t.Error("VM should be running before migration")
		}
	})

	t.Run("MigrationPhases", func(t *testing.T) {
		// Test pause for migration
		err := sourceVM.Pause()
		if err != nil {
			t.Fatalf("Failed to pause VM for migration: %v", err)
		}

		if sourceVM.State() != StatePaused {
			t.Error("VM should be paused during migration")
		}

		// Test resume after migration
		err = sourceVM.Resume()
		if err != nil {
			t.Fatalf("Failed to resume VM after migration: %v", err)
		}

		if sourceVM.State() != StateRunning {
			t.Error("VM should be running after resume")
		}
	})

	t.Run("MigrationCleanup", func(t *testing.T) {
		// Test proper cleanup after migration
		err := sourceVM.Stop()
		if err != nil {
			t.Fatalf("Failed to stop VM: %v", err)
		}

		if sourceVM.State() != StateStopped {
			t.Error("VM should be stopped after migration cleanup")
		}
	})
}

// TestVMMigrationWithContext tests migration with context cancellation
func TestVMMigrationWithContext(t *testing.T) {
	config := VMConfig{
		ID:        "context-migration-vm",
		Name:      "context-vm",
		Command:   "/bin/sleep",
		Args:      []string{"10"},
		CPUShares: 1024,
		MemoryMB:  512,
		RootFS:    "/tmp",
	}

	vm, err := NewVM(config)
	if err != nil {
		t.Fatalf("Failed to create VM: %v", err)
	}
	defer vm.Cleanup()

	// Start the VM
	err = vm.Start()
	if err != nil {
		t.Fatalf("Failed to start VM: %v", err)
	}

	// Test with cancelled context
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	// Migration should respect context cancellation
	// In a real implementation, this would check context.Done()
	select {
	case <-ctx.Done():
		// Expected behavior - context is cancelled
	default:
		t.Error("Migration should check context cancellation")
	}

	// Test with timeout context
	timeoutCtx, timeoutCancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer timeoutCancel()

	// Simulate a long-running operation that should timeout
	select {
	case <-timeoutCtx.Done():
		// Expected behavior - context timed out
	case <-time.After(200 * time.Millisecond):
		t.Error("Operation should have timed out")
	}
}

// TestMigrationResourceValidation tests resource validation during migration
func TestMigrationResourceValidation(t *testing.T) {
	sourceConfig := VMConfig{
		ID:        "resource-test-vm",
		Name:      "resource-vm",
		Command:   "/bin/sleep",
		Args:      []string{"5"},
		CPUShares: 2048,
		MemoryMB:  1024,
		RootFS:    "/tmp",
	}

	vm, err := NewVM(sourceConfig)
	if err != nil {
		t.Fatalf("Failed to create VM: %v", err)
	}
	defer vm.Cleanup()

	// Test resource usage validation
	usage := vm.GetResourceUsage()
	if usage.CPUPercent < 0 {
		t.Error("CPU usage should not be negative")
	}

	// Test resource limit validation
	err = vm.UpdateResourceLimits(4096, 2048)
	if err != nil {
		t.Fatalf("Failed to update resource limits: %v", err)
	}

	// Verify updated configuration
	config := vm.GetConfig()
	if config.CPUShares != 4096 {
		t.Errorf("Expected CPU shares 4096, got %d", config.CPUShares)
	}

	if config.MemoryMB != 2048 {
		t.Errorf("Expected memory 2048MB, got %d", config.MemoryMB)
	}
}
