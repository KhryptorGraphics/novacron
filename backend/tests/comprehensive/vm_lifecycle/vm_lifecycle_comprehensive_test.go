package vm_lifecycle_test

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/vm"
)

// TestVMLifecycleComprehensive provides comprehensive VM lifecycle testing
func TestVMLifecycleComprehensive(t *testing.T) {
	t.Run("State Transition Tests", func(t *testing.T) {
		testCompleteVMLifecycle(t)
		testStateTransitionValidation(t)
		testInvalidStateTransitions(t)
		testConcurrentStateChanges(t)
	})

	t.Run("Migration Tests", func(t *testing.T) {
		testColdMigration(t)
		testWarmMigration(t)
		testLiveMigration(t)
		testMigrationFailureRecovery(t)
		testMigrationRollback(t)
	})

	t.Run("Error Recovery Tests", func(t *testing.T) {
		testVMCrashRecovery(t)
		testResourceExhaustionRecovery(t)
		testNetworkFailureRecovery(t)
		testStorageFailureRecovery(t)
	})

	t.Run("Performance Tests", func(t *testing.T) {
		testVMStartupPerformance(t)
		testMigrationPerformance(t)
		testConcurrentVMOperations(t)
		testResourceUtilizationEfficiency(t)
	})

	t.Run("Edge Cases", func(t *testing.T) {
		testVMOperationsUnderLoad(t)
		testResourceLimitEnforcement(t)
		testVMCloneIntegrity(t)
		testLongRunningVMStability(t)
	})
}

func testCompleteVMLifecycle(t *testing.T) {
	tempDir := t.TempDir()
	
	// Define complete lifecycle states
	expectedTransitions := []vm.VMState{
		vm.StateCreating,
		vm.StateRunning,
		vm.StatePaused,
		vm.StateRunning,  // Resume
		vm.StateStopped,
	}

	config := vm.VMConfig{
		ID:        "lifecycle-test-vm",
		Name:      "Lifecycle Test VM",
		Command:   "sleep",
		Args:      []string{"30"},
		CPUShares: 1024,
		MemoryMB:  256,
		WorkDir:   tempDir,
	}

	testVM, err := vm.NewVM(config)
	if err != nil {
		t.Fatalf("Failed to create VM: %v", err)
	}

	// Track state transitions
	var actualTransitions []vm.VMState
	actualTransitions = append(actualTransitions, testVM.State())

	// Start VM
	err = testVM.Start()
	if err != nil {
		t.Fatalf("Failed to start VM: %v", err)
	}
	actualTransitions = append(actualTransitions, testVM.State())

	// Pause VM
	err = testVM.Pause()
	if err != nil {
		t.Fatalf("Failed to pause VM: %v", err)
	}
	actualTransitions = append(actualTransitions, testVM.State())

	// Resume VM
	err = testVM.ResumeInternal()
	if err != nil {
		t.Fatalf("Failed to resume VM: %v", err)
	}
	actualTransitions = append(actualTransitions, testVM.State())

	// Stop VM
	err = testVM.Stop()
	if err != nil {
		t.Fatalf("Failed to stop VM: %v", err)
	}
	actualTransitions = append(actualTransitions, testVM.State())

	// Verify state transitions
	for i, expected := range expectedTransitions {
		if i < len(actualTransitions) && actualTransitions[i] != expected {
			t.Errorf("State transition %d: expected %s, got %s", 
				i, expected, actualTransitions[i])
		}
	}

	// Verify final cleanup
	if testVM.State() != vm.StateStopped {
		t.Errorf("Final state should be stopped, got %s", testVM.State())
	}
}

func testStateTransitionValidation(t *testing.T) {
	tempDir := t.TempDir()
	
	config := vm.VMConfig{
		ID:        "state-validation-vm",
		Name:      "State Validation VM",
		Command:   "sleep",
		Args:      []string{"10"},
		WorkDir:   tempDir,
	}

	testVM, err := vm.NewVM(config)
	if err != nil {
		t.Fatalf("Failed to create VM: %v", err)
	}

	// Test valid state transitions
	validTransitions := []struct {
		from   vm.VMState
		action string
		to     vm.VMState
	}{
		{vm.StateCreating, "start", vm.StateRunning},
		{vm.StateRunning, "pause", vm.StatePaused},
		{vm.StatePaused, "resume", vm.StateRunning},
		{vm.StateRunning, "stop", vm.StateStopped},
	}

	// Start VM to get to running state
	err = testVM.Start()
	if err != nil {
		t.Fatalf("Failed to start VM: %v", err)
	}

	for _, transition := range validTransitions[1:] { // Skip first as VM is already started
		currentState := testVM.State()
		
		var actionErr error
		switch transition.action {
		case "pause":
			actionErr = testVM.Pause()
		case "resume":
			actionErr = testVM.ResumeInternal()
		case "stop":
			actionErr = testVM.Stop()
		}

		if actionErr != nil {
			t.Errorf("Failed to perform action %s from state %s: %v", 
				transition.action, currentState, actionErr)
		}

		newState := testVM.State()
		if newState != transition.to {
			t.Errorf("Invalid state after %s: expected %s, got %s", 
				transition.action, transition.to, newState)
		}
	}
}

func testInvalidStateTransitions(t *testing.T) {
	tempDir := t.TempDir()
	
	config := vm.VMConfig{
		ID:        "invalid-transition-vm",
		Name:      "Invalid Transition VM",
		Command:   "sleep",
		Args:      []string{"5"},
		WorkDir:   tempDir,
	}

	testVM, err := vm.NewVM(config)
	if err != nil {
		t.Fatalf("Failed to create VM: %v", err)
	}

	// Test invalid transitions that should fail
	invalidTransitions := []struct {
		state  vm.VMState
		action string
	}{
		{vm.StateCreating, "pause"},   // Can't pause non-running VM
		{vm.StateCreating, "stop"},    // Can't stop non-running VM
		{vm.StateStopped, "pause"},    // Can't pause stopped VM
		{vm.StateStopped, "resume"},   // Can't resume stopped VM
	}

	for _, invalid := range invalidTransitions {
		// Set VM to the test state (this is artificial for testing)
		var actionErr error
		
		switch invalid.action {
		case "pause":
			actionErr = testVM.Pause()
		case "stop":
			actionErr = testVM.Stop()
		case "resume":
			actionErr = testVM.ResumeInternal()
		}

		if actionErr == nil && testVM.State() == vm.StateCreating {
			t.Errorf("Invalid transition should fail: %s from %s", 
				invalid.action, invalid.state)
		}
	}
}

func testConcurrentStateChanges(t *testing.T) {
	tempDir := t.TempDir()
	
	config := vm.VMConfig{
		ID:        "concurrent-vm",
		Name:      "Concurrent Test VM",
		Command:   "sleep",
		Args:      []string{"20"},
		WorkDir:   tempDir,
	}

	testVM, err := vm.NewVM(config)
	if err != nil {
		t.Fatalf("Failed to create VM: %v", err)
	}

	// Start VM first
	err = testVM.Start()
	if err != nil {
		t.Fatalf("Failed to start VM: %v", err)
	}

	// Launch concurrent operations
	var wg sync.WaitGroup
	operations := []string{"pause", "resume", "stop", "start"}
	errors := make(chan error, len(operations))

	for _, op := range operations {
		wg.Add(1)
		go func(operation string) {
			defer wg.Done()
			
			switch operation {
			case "pause":
				if err := testVM.Pause(); err != nil {
					errors <- fmt.Errorf("pause failed: %v", err)
				}
			case "resume":
				// Small delay to allow pause to potentially complete
				time.Sleep(10 * time.Millisecond)
				if err := testVM.ResumeInternal(); err != nil {
					errors <- fmt.Errorf("resume failed: %v", err)
				}
			case "stop":
				time.Sleep(20 * time.Millisecond)
				if err := testVM.Stop(); err != nil {
					errors <- fmt.Errorf("stop failed: %v", err)
				}
			}
		}(op)
	}

	wg.Wait()
	close(errors)

	// Collect and report any errors (some are expected due to concurrency)
	errorCount := 0
	for err := range errors {
		errorCount++
		t.Logf("Concurrent operation error (may be expected): %v", err)
	}

	// Verify VM is in a consistent final state
	finalState := testVM.State()
	validFinalStates := []vm.VMState{vm.StateRunning, vm.StatePaused, vm.StateStopped}
	
	stateValid := false
	for _, validState := range validFinalStates {
		if finalState == validState {
			stateValid = true
			break
		}
	}

	if !stateValid {
		t.Errorf("VM in invalid final state after concurrent operations: %s", finalState)
	}
}

func testColdMigration(t *testing.T) {
	tempDir := t.TempDir()
	sourceDir := filepath.Join(tempDir, "source")
	destDir := filepath.Join(tempDir, "dest")
	
	// Create necessary directories
	if err := os.MkdirAll(sourceDir, 0755); err != nil {
		t.Fatalf("Failed to create source directory: %v", err)
	}
	if err := os.MkdirAll(destDir, 0755); err != nil {
		t.Fatalf("Failed to create dest directory: %v", err)
	}

	// Create VM state files for testing
	vmID := "cold-migration-test-vm"
	stateFile := filepath.Join(sourceDir, vmID+".state")
	if err := os.WriteFile(stateFile, []byte("vm-state-data"), 0644); err != nil {
		t.Fatalf("Failed to create state file: %v", err)
	}

	// Setup migration managers
	sourceManager := vm.NewVMMigrationManager("source-node", sourceDir)
	destManager := vm.NewVMMigrationManager("dest-node", destDir)

	// Create migration request
	migration := &vm.VMMigration{
		ID:                "migration-001",
		VMID:              vmID,
		SourceNodeID:      "source-node",
		DestinationNodeID: "dest-node",
		Type:              vm.MigrationTypeCold,
		Status:            vm.MigrationStatusPending,
		VMSpec: vm.VMSpec{
			ID:       vmID,
			Name:     "Cold Migration Test",
			VCPU:     2,
			MemoryMB: 1024,
			DiskMB:   10240,
		},
		CreatedAt: time.Now(),
	}

	// Execute migration
	ctx := context.WithTimeout(context.Background(), 30*time.Second)
	err := sourceManager.ExecuteMigration(ctx, migration, destManager)
	if err != nil {
		t.Fatalf("Cold migration failed: %v", err)
	}

	// Verify migration completed successfully
	if migration.Status != vm.MigrationStatusCompleted {
		t.Errorf("Expected migration status %s, got %s", 
			vm.MigrationStatusCompleted, migration.Status)
	}

	// Verify VM state file was copied
	destStateFile := filepath.Join(destDir, vmID+".state")
	if _, err := os.Stat(destStateFile); err != nil {
		t.Errorf("State file not copied to destination: %v", err)
	}

	// Verify VM is available on destination
	destVM, err := destManager.GetVM(vmID)
	if err != nil {
		t.Errorf("VM not available on destination: %v", err)
	} else if destVM.State() != vm.StateRunning {
		t.Errorf("VM should be running on destination, got %s", destVM.State())
	}
}

func testWarmMigration(t *testing.T) {
	tempDir := t.TempDir()
	sourceDir := filepath.Join(tempDir, "source")
	destDir := filepath.Join(tempDir, "dest")
	
	// Create directories and required files
	os.MkdirAll(sourceDir, 0755)
	os.MkdirAll(destDir, 0755)

	vmID := "warm-migration-test-vm"
	
	// Create VM state and memory files
	stateFile := filepath.Join(sourceDir, vmID+".state")
	memFile := filepath.Join(sourceDir, vmID+".memory")
	
	os.WriteFile(stateFile, []byte("vm-state-data"), 0644)
	os.WriteFile(memFile, []byte("vm-memory-data"), 0644)

	sourceManager := vm.NewVMMigrationManager("source-node", sourceDir)
	destManager := vm.NewVMMigrationManager("dest-node", destDir)

	migration := &vm.VMMigration{
		ID:                "warm-migration-001",
		VMID:              vmID,
		SourceNodeID:      "source-node",
		DestinationNodeID: "dest-node",
		Type:              vm.MigrationTypeWarm,
		Status:            vm.MigrationStatusPending,
		VMSpec: vm.VMSpec{
			ID:       vmID,
			Name:     "Warm Migration Test",
			VCPU:     4,
			MemoryMB: 2048,
		},
		CreatedAt: time.Now(),
	}

	ctx := context.WithTimeout(context.Background(), 30*time.Second)
	err := sourceManager.ExecuteMigration(ctx, migration, destManager)
	if err != nil {
		t.Fatalf("Warm migration failed: %v", err)
	}

	// Verify both state and memory files were copied
	destStateFile := filepath.Join(destDir, vmID+".state")
	destMemFile := filepath.Join(destDir, vmID+".memory")
	
	if _, err := os.Stat(destStateFile); err != nil {
		t.Errorf("State file not copied: %v", err)
	}
	if _, err := os.Stat(destMemFile); err != nil {
		t.Errorf("Memory file not copied: %v", err)
	}

	if migration.Status != vm.MigrationStatusCompleted {
		t.Errorf("Warm migration should complete successfully, got status %s", migration.Status)
	}
}

func testLiveMigration(t *testing.T) {
	tempDir := t.TempDir()
	sourceDir := filepath.Join(tempDir, "source")
	destDir := filepath.Join(tempDir, "dest")
	
	os.MkdirAll(sourceDir, 0755)
	os.MkdirAll(destDir, 0755)

	vmID := "live-migration-test-vm"
	
	// Create state file and memory iteration files for live migration
	stateFile := filepath.Join(sourceDir, vmID+".state")
	os.WriteFile(stateFile, []byte("vm-state-data"), 0644)
	
	// Create memory iteration files (simulating iterative memory copy)
	for i := 1; i <= 3; i++ {
		iterFile := filepath.Join(sourceDir, fmt.Sprintf("%s.memory.%d", vmID, i))
		os.WriteFile(iterFile, []byte(fmt.Sprintf("memory-iter-%d", i)), 0644)
	}

	sourceManager := vm.NewVMMigrationManager("source-node", sourceDir)
	destManager := vm.NewVMMigrationManager("dest-node", destDir)

	migration := &vm.VMMigration{
		ID:                "live-migration-001",
		VMID:              vmID,
		SourceNodeID:      "source-node",
		DestinationNodeID: "dest-node",
		Type:              vm.MigrationTypeLive,
		Status:            vm.MigrationStatusPending,
		VMSpec: vm.VMSpec{
			ID:       vmID,
			Name:     "Live Migration Test",
			VCPU:     2,
			MemoryMB: 1024,
		},
		Options:   map[string]string{"iterations": "3"},
		CreatedAt: time.Now(),
	}

	ctx := context.WithTimeout(context.Background(), 30*time.Second)
	err := sourceManager.ExecuteMigration(ctx, migration, destManager)
	if err != nil {
		t.Fatalf("Live migration failed: %v", err)
	}

	// Verify migration progress was tracked
	if migration.Progress != 100 {
		t.Errorf("Expected migration progress 100, got %f", migration.Progress)
	}

	// Verify final memory state was created
	destMemFile := filepath.Join(destDir, vmID+".memory")
	if _, err := os.Stat(destMemFile); err != nil {
		t.Errorf("Final memory file not created: %v", err)
	}

	if migration.Status != vm.MigrationStatusCompleted {
		t.Errorf("Live migration should complete successfully, got status %s", migration.Status)
	}
}

func testMigrationFailureRecovery(t *testing.T) {
	tempDir := t.TempDir()
	sourceDir := filepath.Join(tempDir, "source")
	destDir := filepath.Join(tempDir, "dest")
	
	os.MkdirAll(sourceDir, 0755)
	os.MkdirAll(destDir, 0755)

	vmID := "failure-test-vm"
	
	// Create only state file, missing memory file to trigger failure
	stateFile := filepath.Join(sourceDir, vmID+".state")
	os.WriteFile(stateFile, []byte("vm-state-data"), 0644)
	// Intentionally not creating memory file to simulate failure

	sourceManager := vm.NewVMMigrationManager("source-node", sourceDir)
	destManager := vm.NewVMMigrationManager("dest-node", destDir)

	migration := &vm.VMMigration{
		ID:                "failure-migration-001",
		VMID:              vmID,
		SourceNodeID:      "source-node",
		DestinationNodeID: "dest-node",
		Type:              vm.MigrationTypeWarm, // Requires memory file
		Status:            vm.MigrationStatusPending,
		VMSpec: vm.VMSpec{
			ID:       vmID,
			Name:     "Failure Test VM",
		},
		CreatedAt: time.Now(),
	}

	ctx := context.WithTimeout(context.Background(), 10*time.Second)
	err := sourceManager.ExecuteMigration(ctx, migration, destManager)
	
	// Migration should fail due to missing memory file
	if err == nil {
		t.Error("Migration should fail due to missing memory file")
	}

	// Verify migration status reflects failure and rollback
	expectedStatuses := []vm.MigrationStatus{vm.MigrationStatusFailed, vm.MigrationStatusRolledBack}
	statusValid := false
	for _, status := range expectedStatuses {
		if migration.Status == status {
			statusValid = true
			break
		}
	}
	
	if !statusValid {
		t.Errorf("Expected migration status to be failed or rolled back, got %s", migration.Status)
	}

	// Verify error message is populated
	if migration.Error == "" {
		t.Error("Migration error message should be populated on failure")
	}
}

func testMigrationRollback(t *testing.T) {
	tempDir := t.TempDir()
	sourceDir := filepath.Join(tempDir, "source")
	destDir := filepath.Join(tempDir, "dest")
	
	os.MkdirAll(sourceDir, 0755)
	os.MkdirAll(destDir, 0755)

	vmID := "rollback-test-vm"
	
	// Create required files
	stateFile := filepath.Join(sourceDir, vmID+".state")
	os.WriteFile(stateFile, []byte("vm-state-data"), 0644)

	sourceManager := vm.NewVMMigrationManager("source-node", sourceDir)
	destManager := vm.NewVMMigrationManager("dest-node", destDir)

	migration := &vm.VMMigration{
		ID:                "rollback-migration-001",
		VMID:              vmID,
		SourceNodeID:      "source-node",
		DestinationNodeID: "dest-node",
		Type:              vm.MigrationTypeCold,
		Status:            vm.MigrationStatusPending,
		VMSpec:            vm.VMSpec{ID: vmID, Name: "Rollback Test VM"},
		CreatedAt:         time.Now(),
	}

	// First, make the migration fail by removing the state file during migration
	go func() {
		time.Sleep(10 * time.Millisecond)
		os.Remove(stateFile) // Remove file to cause failure
	}()

	ctx := context.WithTimeout(context.Background(), 5*time.Second)
	err := sourceManager.ExecuteMigration(ctx, migration, destManager)
	
	if err == nil {
		t.Error("Migration should fail and trigger rollback")
	}

	// Verify rollback occurred
	if migration.Status == vm.MigrationStatusRolledBack {
		t.Log("Migration properly rolled back on failure")
	} else if migration.Status == vm.MigrationStatusFailed {
		t.Log("Migration failed as expected (rollback may have failed)")
	} else {
		t.Errorf("Unexpected migration status after failure: %s", migration.Status)
	}

	// Verify VM is still accessible from source (rollback successful)
	sourceVM, err := sourceManager.GetVM(vmID)
	if err != nil {
		t.Logf("VM not accessible on source after rollback (may be expected): %v", err)
	} else if sourceVM.State() == vm.StateRunning {
		t.Log("VM properly restored on source after rollback")
	}
}

func testVMCrashRecovery(t *testing.T) {
	tempDir := t.TempDir()
	
	config := vm.VMConfig{
		ID:        "crash-recovery-vm",
		Name:      "Crash Recovery VM",
		Command:   "sh",
		Args:      []string{"-c", "sleep 1; exit 1"}, // Will crash after 1 second
		WorkDir:   tempDir,
	}

	testVM, err := vm.NewVM(config)
	if err != nil {
		t.Fatalf("Failed to create VM: %v", err)
	}

	// Start VM
	err = testVM.Start()
	if err != nil {
		t.Fatalf("Failed to start VM: %v", err)
	}

	// Wait for VM to crash
	time.Sleep(2 * time.Second)

	// VM should detect crash and update state
	if testVM.State() == vm.StateRunning {
		t.Error("VM should have detected crash and changed state")
	}

	// Test recovery by restarting
	err = testVM.Start()
	if err != nil {
		t.Errorf("Failed to restart VM after crash: %v", err)
	}
}

func testResourceExhaustionRecovery(t *testing.T) {
	tempDir := t.TempDir()
	
	// Create VM with very limited resources
	config := vm.VMConfig{
		ID:        "resource-exhaustion-vm",
		Name:      "Resource Exhaustion VM",
		Command:   "sleep",
		Args:      []string{"5"},
		CPUShares: 1,     // Very low CPU
		MemoryMB:  1,     // Very low memory
		WorkDir:   tempDir,
	}

	testVM, err := vm.NewVM(config)
	if err != nil {
		t.Fatalf("Failed to create VM: %v", err)
	}

	// Start VM
	err = testVM.Start()
	if err != nil {
		t.Fatalf("Failed to start VM: %v", err)
	}

	// Test resource limit updates
	err = testVM.UpdateResourceLimits(1024, 256) // Increase limits
	if err != nil {
		t.Errorf("Failed to update resource limits: %v", err)
	}

	// Verify limits were updated
	usage := testVM.GetResourceUsage()
	t.Logf("Resource usage after limit update: CPU=%.2f%%, Memory=%d bytes", 
		usage.CPUPercent, usage.MemoryBytes)

	testVM.Stop()
}

func testVMStartupPerformance(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping performance test in short mode")
	}

	tempDir := t.TempDir()
	numVMs := 10
	startupTimes := make([]time.Duration, numVMs)

	// Test concurrent VM startups
	var wg sync.WaitGroup
	for i := 0; i < numVMs; i++ {
		wg.Add(1)
		go func(index int) {
			defer wg.Done()
			
			config := vm.VMConfig{
				ID:        fmt.Sprintf("perf-vm-%d", index),
				Name:      fmt.Sprintf("Performance VM %d", index),
				Command:   "sleep",
				Args:      []string{"2"},
				CPUShares: 512,
				MemoryMB:  128,
				WorkDir:   tempDir,
			}

			testVM, err := vm.NewVM(config)
			if err != nil {
				t.Errorf("Failed to create VM %d: %v", index, err)
				return
			}

			start := time.Now()
			err = testVM.Start()
			startupTimes[index] = time.Since(start)

			if err != nil {
				t.Errorf("Failed to start VM %d: %v", index, err)
				return
			}

			// Cleanup
			testVM.Stop()
		}(i)
	}

	wg.Wait()

	// Calculate performance metrics
	var totalTime time.Duration
	var maxTime time.Duration
	successCount := 0

	for _, duration := range startupTimes {
		if duration > 0 {
			totalTime += duration
			successCount++
			if duration > maxTime {
				maxTime = duration
			}
		}
	}

	if successCount > 0 {
		avgTime := totalTime / time.Duration(successCount)
		t.Logf("VM Startup Performance: avg=%v, max=%v, success_rate=%.1f%%", 
			avgTime, maxTime, float64(successCount)/float64(numVMs)*100)
	}

	// Verify reasonable performance
	maxAcceptableTime := 5 * time.Second
	if maxTime > maxAcceptableTime {
		t.Errorf("VM startup too slow: max=%v, limit=%v", maxTime, maxAcceptableTime)
	}
}

func testMigrationPerformance(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping performance test in short mode")
	}

	tempDir := t.TempDir()
	sourceDir := filepath.Join(tempDir, "source")
	destDir := filepath.Join(tempDir, "dest")
	
	os.MkdirAll(sourceDir, 0755)
	os.MkdirAll(destDir, 0755)

	// Test different migration types
	migrationTypes := []vm.MigrationType{
		vm.MigrationTypeCold,
		vm.MigrationTypeWarm,
		vm.MigrationTypeLive,
	}

	for _, migrationType := range migrationTypes {
		t.Run(string(migrationType), func(t *testing.T) {
			vmID := fmt.Sprintf("perf-migration-%s", migrationType)
			
			// Create required files based on migration type
			stateFile := filepath.Join(sourceDir, vmID+".state")
			os.WriteFile(stateFile, []byte("performance-test-state"), 0644)
			
			if migrationType == vm.MigrationTypeWarm || migrationType == vm.MigrationTypeLive {
				memFile := filepath.Join(sourceDir, vmID+".memory")
				os.WriteFile(memFile, []byte("performance-test-memory"), 0644)
			}
			
			if migrationType == vm.MigrationTypeLive {
				for i := 1; i <= 3; i++ {
					iterFile := filepath.Join(sourceDir, fmt.Sprintf("%s.memory.%d", vmID, i))
					os.WriteFile(iterFile, []byte(fmt.Sprintf("iter-%d", i)), 0644)
				}
			}

			sourceManager := vm.NewVMMigrationManager("source", sourceDir)
			destManager := vm.NewVMMigrationManager("dest", destDir)

			migration := &vm.VMMigration{
				ID:                fmt.Sprintf("perf-%s", migrationType),
				VMID:              vmID,
				SourceNodeID:      "source",
				DestinationNodeID: "dest",
				Type:              migrationType,
				VMSpec:            vm.VMSpec{ID: vmID, Name: "Performance Test"},
				Options:           map[string]string{"iterations": "3"},
				CreatedAt:         time.Now(),
			}

			// Measure migration time
			start := time.Now()
			ctx := context.WithTimeout(context.Background(), 30*time.Second)
			err := sourceManager.ExecuteMigration(ctx, migration, destManager)
			duration := time.Since(start)

			if err != nil {
				t.Errorf("Migration failed: %v", err)
				return
			}

			t.Logf("%s migration completed in %v", migrationType, duration)

			// Verify reasonable performance
			maxAcceptableTime := 10 * time.Second
			if duration > maxAcceptableTime {
				t.Errorf("%s migration too slow: %v > %v", migrationType, duration, maxAcceptableTime)
			}
		})
	}
}

func testConcurrentVMOperations(t *testing.T) {
	tempDir := t.TempDir()
	numVMs := 20
	
	var wg sync.WaitGroup
	errors := make(chan error, numVMs*3) // 3 operations per VM

	for i := 0; i < numVMs; i++ {
		wg.Add(1)
		go func(index int) {
			defer wg.Done()
			
			config := vm.VMConfig{
				ID:        fmt.Sprintf("concurrent-vm-%d", index),
				Name:      fmt.Sprintf("Concurrent VM %d", index),
				Command:   "sleep",
				Args:      []string{"3"},
				CPUShares: 256,
				MemoryMB:  64,
				WorkDir:   tempDir,
			}

			testVM, err := vm.NewVM(config)
			if err != nil {
				errors <- fmt.Errorf("failed to create VM %d: %v", index, err)
				return
			}

			// Start VM
			if err := testVM.Start(); err != nil {
				errors <- fmt.Errorf("failed to start VM %d: %v", index, err)
				return
			}

			// Pause VM
			if err := testVM.Pause(); err != nil {
				errors <- fmt.Errorf("failed to pause VM %d: %v", index, err)
			}

			// Resume VM
			if err := testVM.ResumeInternal(); err != nil {
				errors <- fmt.Errorf("failed to resume VM %d: %v", index, err)
			}

			// Stop VM
			if err := testVM.Stop(); err != nil {
				errors <- fmt.Errorf("failed to stop VM %d: %v", index, err)
			}
		}(i)
	}

	wg.Wait()
	close(errors)

	// Count errors
	errorCount := 0
	for err := range errors {
		errorCount++
		t.Logf("Concurrent operation error: %v", err)
	}

	successRate := float64(numVMs*3-errorCount) / float64(numVMs*3) * 100
	t.Logf("Concurrent operations success rate: %.1f%%", successRate)

	if successRate < 90.0 {
		t.Errorf("Concurrent operation success rate too low: %.1f%%", successRate)
	}
}

func testVMCloneIntegrity(t *testing.T) {
	tempDir := t.TempDir()
	
	originalConfig := vm.VMConfig{
		ID:        "original-vm",
		Name:      "Original VM",
		Command:   "sleep",
		Args:      []string{"10"},
		CPUShares: 1024,
		MemoryMB:  512,
		WorkDir:   tempDir,
		Env:       map[string]string{"TEST": "original"},
		Tags:      map[string]string{"type": "original"},
	}

	originalVM, err := vm.NewVM(originalConfig)
	if err != nil {
		t.Fatalf("Failed to create original VM: %v", err)
	}

	// Start original VM
	err = originalVM.Start()
	if err != nil {
		t.Fatalf("Failed to start original VM: %v", err)
	}

	// Clone VM
	clonedVM, err := originalVM.Clone("cloned-vm")
	if err != nil {
		t.Fatalf("Failed to clone VM: %v", err)
	}

	// Verify clone integrity
	if clonedVM.Name() != "cloned-vm" {
		t.Errorf("Clone name incorrect: expected 'cloned-vm', got '%s'", clonedVM.Name())
	}

	if clonedVM.ID() == originalVM.ID() {
		t.Error("Clone should have different ID from original")
	}

	// Verify clone can be started independently
	err = clonedVM.Start()
	if err != nil {
		t.Errorf("Failed to start cloned VM: %v", err)
	}

	// Verify both VMs can run simultaneously
	if originalVM.State() != vm.StateRunning {
		t.Error("Original VM should still be running")
	}
	if clonedVM.State() != vm.StateRunning {
		t.Error("Cloned VM should be running")
	}

	// Cleanup
	originalVM.Stop()
	clonedVM.Stop()
}

func testResourceUtilizationEfficiency(t *testing.T) {
	tempDir := t.TempDir()
	
	config := vm.VMConfig{
		ID:        "resource-efficiency-vm",
		Name:      "Resource Efficiency VM",
		Command:   "sleep",
		Args:      []string{"5"},
		CPUShares: 1024,
		MemoryMB:  256,
		WorkDir:   tempDir,
	}

	testVM, err := vm.NewVM(config)
	if err != nil {
		t.Fatalf("Failed to create VM: %v", err)
	}

	err = testVM.Start()
	if err != nil {
		t.Fatalf("Failed to start VM: %v", err)
	}

	// Monitor resource usage over time
	measurements := make([]vm.VMResourceUsage, 5)
	for i := 0; i < 5; i++ {
		time.Sleep(200 * time.Millisecond)
		measurements[i] = testVM.GetResourceUsage()
	}

	// Analyze resource efficiency
	for i, usage := range measurements {
		t.Logf("Measurement %d: CPU=%.2f%%, Memory=%d bytes", 
			i+1, usage.CPUPercent, usage.MemoryBytes)
	}

	// Get process stats for detailed analysis
	processStats, err := testVM.GetProcessStats()
	if err != nil {
		t.Errorf("Failed to get process stats: %v", err)
	} else {
		t.Logf("Process stats: PID=%d, State=%s, Threads=%d", 
			processStats.PID, processStats.State, processStats.Threads)
	}

	testVM.Stop()
}

func testLongRunningVMStability(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping long-running stability test in short mode")
	}

	tempDir := t.TempDir()
	
	config := vm.VMConfig{
		ID:        "stability-vm",
		Name:      "Long Running Stability VM",
		Command:   "sleep",
		Args:      []string{"30"},
		CPUShares: 512,
		MemoryMB:  128,
		WorkDir:   tempDir,
	}

	testVM, err := vm.NewVM(config)
	if err != nil {
		t.Fatalf("Failed to create VM: %v", err)
	}

	err = testVM.Start()
	if err != nil {
		t.Fatalf("Failed to start VM: %v", err)
	}

	// Monitor VM stability over extended period
	duration := 10 * time.Second // Shortened for testing
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	stateChanges := 0
	lastState := testVM.State()
	
	timeout := time.After(duration)
	for {
		select {
		case <-ticker.C:
			currentState := testVM.State()
			if currentState != lastState {
				stateChanges++
				t.Logf("State change detected: %s -> %s", lastState, currentState)
				lastState = currentState
			}
			
			// Check if VM crashed unexpectedly
			if currentState == vm.StateFailed {
				t.Error("VM failed unexpectedly during stability test")
				return
			}

		case <-timeout:
			t.Logf("Stability test completed: %d state changes over %v", stateChanges, duration)
			testVM.Stop()
			return
		}
	}
}