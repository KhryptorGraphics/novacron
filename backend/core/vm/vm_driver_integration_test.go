package vm

import (
	"context"
	"testing"
	"time"
)

// TestVMDriverIntegration tests integration between all VM drivers
func TestVMDriverIntegration(t *testing.T) {
	// Initialize driver manager
	config := DefaultVMDriverConfig("integration-test-node")
	manager := NewVMDriverManager(config)
	defer manager.Close()

	// Test VM configurations for each driver type
	testVMs := []struct {
		name          string
		vmType        VMType
		config        VMConfig
		expectSuccess bool
	}{
		{
			name:   "Docker Container",
			vmType: VMTypeContainer,
			config: VMConfig{
				ID:        "test-docker-integration",
				Name:      "test-docker",
				CPUShares: 512,
				MemoryMB:  256,
				RootFS:    "alpine:latest",
				Command:   "/bin/sh",
				Args:      []string{"-c", "sleep 30"},
				Tags:      map[string]string{"vm_type": "container"},
			},
			expectSuccess: false, // Likely fails without Docker daemon
		},
		{
			name:   "Containerd Container",
			vmType: VMTypeContainerd,
			config: VMConfig{
				ID:        "test-containerd-integration",
				Name:      "test-containerd",
				CPUShares: 512,
				MemoryMB:  256,
				RootFS:    "alpine:latest",
				Command:   "/bin/sh",
				Args:      []string{"-c", "sleep 30"},
				Tags:      map[string]string{"vm_type": "containerd"},
			},
			expectSuccess: false, // Likely fails without containerd daemon
		},
		{
			name:   "KVM Virtual Machine",
			vmType: VMTypeKVM,
			config: VMConfig{
				ID:        "test-kvm-integration",
				Name:      "test-kvm",
				CPUShares: 1024,
				MemoryMB:  512,
				RootFS:    "/tmp/test-disk.img",
				Tags:      map[string]string{"vm_type": "kvm"},
			},
			expectSuccess: false, // Likely fails without QEMU/KVM
		},
		{
			name:   "Kata Containers",
			vmType: VMTypeKataContainers,
			config: VMConfig{
				ID:        "test-kata-integration",
				Name:      "test-kata",
				CPUShares: 1024,
				MemoryMB:  512,
				RootFS:    "alpine:latest",
				Command:   "/bin/sh",
				Args:      []string{"-c", "sleep 30"},
				Tags:      map[string]string{"vm_type": "kata-containers"},
			},
			expectSuccess: false, // Likely fails without Kata runtime
		},
	}

	// Test driver creation and basic operations
	for _, testVM := range testVMs {
		t.Run(testVM.name, func(t *testing.T) {
			// Get driver for VM type
			driver, err := manager.GetDriver(testVM.config)
			if err != nil {
				if testVM.expectSuccess {
					t.Errorf("Expected to get driver for %s, but got error: %v", testVM.name, err)
				} else {
					t.Logf("Failed to get driver for %s (expected in test environment): %v", testVM.name, err)
				}
				return
			}

			if driver == nil {
				t.Fatalf("Driver should not be nil for %s", testVM.name)
			}

			// Test driver capabilities
			t.Logf("%s driver capabilities:", testVM.name)
			t.Logf("  - Pause support: %t", driver.SupportsPause())
			t.Logf("  - Resume support: %t", driver.SupportsResume())
			t.Logf("  - Snapshot support: %t", driver.SupportsSnapshot())
			t.Logf("  - Migration support: %t", driver.SupportsMigrate())

			// Test VM lifecycle operations
			testVMLifecycle(t, driver, testVM.config, testVM.expectSuccess)
		})
	}
}

// testVMLifecycle tests the complete VM lifecycle for a driver
func testVMLifecycle(t *testing.T, driver VMDriver, config VMConfig, expectSuccess bool) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	// Test Create operation
	vmID, err := driver.Create(ctx, config)
	if err != nil {
		if expectSuccess {
			t.Errorf("Create should succeed, got error: %v", err)
		} else {
			t.Logf("Create failed as expected: %v", err)
		}
		return
	}

	if vmID == "" {
		t.Error("VM ID should not be empty")
		return
	}

	t.Logf("Successfully created VM with ID: %s", vmID)

	// Clean up VM at the end
	defer func() {
		if err := driver.Delete(ctx, vmID); err != nil {
			t.Logf("Warning: Failed to delete VM %s: %v", vmID, err)
		}
	}()

	// Test GetStatus operation
	status, err := driver.GetStatus(ctx, vmID)
	if err != nil {
		t.Errorf("GetStatus should succeed, got error: %v", err)
		return
	}

	expectedStatus := []State{StateCreated, StateStopped}
	statusValid := false
	for _, expected := range expectedStatus {
		if status == expected {
			statusValid = true
			break
		}
	}

	if !statusValid {
		t.Logf("VM status is %s (expected one of %v)", status, expectedStatus)
	}

	// Test GetInfo operation
	info, err := driver.GetInfo(ctx, vmID)
	if err != nil {
		t.Errorf("GetInfo should succeed, got error: %v", err)
		return
	}

	if info == nil {
		t.Error("VM info should not be nil")
		return
	}

	if info.ID != vmID {
		t.Errorf("VM info ID should be %s, got %s", vmID, info.ID)
	}

	// Test Start operation (if VM supports it)
	if status == StateCreated || status == StateStopped {
		err := driver.Start(ctx, vmID)
		if err != nil {
			t.Logf("Start operation failed (may be expected): %v", err)
		} else {
			t.Logf("Successfully started VM %s", vmID)

			// Allow some time for VM to start
			time.Sleep(1 * time.Second)

			// Test GetStatus after start
			newStatus, err := driver.GetStatus(ctx, vmID)
			if err != nil {
				t.Logf("GetStatus after start failed: %v", err)
			} else {
				t.Logf("VM status after start: %s", newStatus)
			}

			// Test Pause/Resume if supported
			if driver.SupportsPause() && driver.SupportsResume() {
				testPauseResume(t, driver, vmID)
			}

			// Test Stop operation
			err = driver.Stop(ctx, vmID)
			if err != nil {
				t.Logf("Stop operation failed: %v", err)
			} else {
				t.Logf("Successfully stopped VM %s", vmID)
			}
		}
	}

	// Test ListVMs operation
	vms, err := driver.ListVMs(ctx)
	if err != nil {
		t.Errorf("ListVMs should succeed, got error: %v", err)
	} else {
		found := false
		for _, vm := range vms {
			if vm.ID == vmID {
				found = true
				break
			}
		}
		if !found {
			t.Logf("VM %s not found in ListVMs result (may be expected)", vmID)
		} else {
			t.Logf("VM %s found in ListVMs result", vmID)
		}
	}

	// Test Snapshot if supported
	if driver.SupportsSnapshot() {
		testSnapshot(t, driver, vmID)
	}

	t.Logf("Completed lifecycle test for VM %s", vmID)
}

// testPauseResume tests pause and resume operations
func testPauseResume(t *testing.T, driver VMDriver, vmID string) {
	ctx := context.Background()

	// Test Pause
	err := driver.Pause(ctx, vmID)
	if err != nil {
		t.Logf("Pause operation failed: %v", err)
		return
	}

	t.Logf("Successfully paused VM %s", vmID)

	// Check status after pause
	status, err := driver.GetStatus(ctx, vmID)
	if err != nil {
		t.Logf("GetStatus after pause failed: %v", err)
	} else if status == StatePaused {
		t.Logf("VM correctly shows paused status")
	} else {
		t.Logf("VM status after pause: %s (expected: %s)", status, StatePaused)
	}

	// Test Resume
	err = driver.Resume(ctx, vmID)
	if err != nil {
		t.Logf("Resume operation failed: %v", err)
		return
	}

	t.Logf("Successfully resumed VM %s", vmID)

	// Check status after resume
	status, err = driver.GetStatus(ctx, vmID)
	if err != nil {
		t.Logf("GetStatus after resume failed: %v", err)
	} else if status == StateRunning {
		t.Logf("VM correctly shows running status after resume")
	} else {
		t.Logf("VM status after resume: %s", status)
	}
}

// testSnapshot tests snapshot operations
func testSnapshot(t *testing.T, driver VMDriver, vmID string) {
	ctx := context.Background()

	snapshotName := "test-snapshot"
	params := map[string]string{
		"description": "Integration test snapshot",
	}

	snapshotID, err := driver.Snapshot(ctx, vmID, snapshotName, params)
	if err != nil {
		t.Logf("Snapshot operation failed: %v", err)
		return
	}

	if snapshotID == "" {
		t.Error("Snapshot ID should not be empty")
		return
	}

	t.Logf("Successfully created snapshot %s for VM %s", snapshotID, vmID)
}

// TestDriverCompatibility tests compatibility between different drivers
func TestDriverCompatibility(t *testing.T) {
	config := DefaultVMDriverConfig("compatibility-test-node")
	manager := NewVMDriverManager(config)
	defer manager.Close()

	// Test that all supported driver types can be instantiated
	supportedTypes := manager.ListSupportedTypes()

	for _, vmType := range supportedTypes {
		t.Run(string(vmType), func(t *testing.T) {
			vmConfig := VMConfig{
				ID:   "test-" + string(vmType),
				Name: "test-vm",
				Tags: map[string]string{"vm_type": string(vmType)},
			}

			driver, err := manager.GetDriver(vmConfig)
			if err != nil {
				// Some drivers may fail without proper runtime environment
				if vmType == VMTypeProcess {
					// Process driver is not yet implemented
					expectedErr := "process driver not yet implemented"
					if err.Error() != expectedErr {
						t.Errorf("Expected process driver error message '%s', got '%s'", expectedErr, err.Error())
					}
				} else {
					t.Logf("Driver %s failed to initialize (expected in test environment): %v", vmType, err)
				}
				return
			}

			if driver == nil {
				t.Errorf("Driver %s should not be nil when initialization succeeds", vmType)
				return
			}

			// Test that driver implements all required methods
			testDriverInterface(t, driver, vmType)
		})
	}
}

// testDriverInterface tests that a driver properly implements the VMDriver interface
func testDriverInterface(t *testing.T, driver VMDriver, vmType VMType) {
	// Test capability methods (should not panic)
	pauseSupport := driver.SupportsPause()
	resumeSupport := driver.SupportsResume()
	snapshotSupport := driver.SupportsSnapshot()
	migrateSupport := driver.SupportsMigrate()

	t.Logf("Driver %s capabilities: Pause=%t, Resume=%t, Snapshot=%t, Migrate=%t",
		vmType, pauseSupport, resumeSupport, snapshotSupport, migrateSupport)

	// Verify logical consistency
	if resumeSupport && !pauseSupport {
		t.Errorf("Driver %s supports resume but not pause (illogical)", vmType)
	}

	// Test that methods exist and can be called (even if they fail)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// These operations should fail gracefully, not panic
	testConfig := VMConfig{
		ID:   "interface-test",
		Name: "interface-test",
	}

	// Test all interface methods exist
	_, _ = driver.Create(ctx, testConfig)
	_ = driver.Start(ctx, "test-id")
	_ = driver.Stop(ctx, "test-id")
	_ = driver.Delete(ctx, "test-id")
	_, _ = driver.GetStatus(ctx, "test-id")
	_, _ = driver.GetInfo(ctx, "test-id")
	_, _ = driver.GetMetrics(ctx, "test-id")
	_, _ = driver.ListVMs(ctx)

	// Test optional operations
	_ = driver.Pause(ctx, "test-id")
	_ = driver.Resume(ctx, "test-id")
	_, _ = driver.Snapshot(ctx, "test-id", "test-snapshot", nil)
	_ = driver.Migrate(ctx, "test-id", "target-node", nil)

	t.Logf("Driver %s interface test completed successfully", vmType)
}

// BenchmarkDriverFactory benchmarks driver factory performance
func BenchmarkDriverFactory(b *testing.B) {
	config := DefaultVMDriverConfig("benchmark-node")
	factory := NewVMDriverFactory(config)

	vmConfig := VMConfig{
		ID:   "benchmark-vm",
		Name: "benchmark",
		Tags: map[string]string{"vm_type": "container"},
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, _ = factory(vmConfig) // Ignore errors for benchmarking
	}
}

// BenchmarkDriverManager benchmarks driver manager performance
func BenchmarkDriverManager(b *testing.B) {
	config := DefaultVMDriverConfig("benchmark-node")
	manager := NewVMDriverManager(config)
	defer manager.Close()

	vmConfig := VMConfig{
		ID:   "benchmark-vm",
		Name: "benchmark",
		Tags: map[string]string{"vm_type": "container"},
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, _ = manager.GetDriver(vmConfig) // Ignore errors for benchmarking
	}
}
