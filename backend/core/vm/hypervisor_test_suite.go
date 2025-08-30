package vm

import (
	"context"
	"fmt"
	"testing"
	"time"
)

// HypervisorTestSuite provides comprehensive testing infrastructure for all hypervisor drivers
type HypervisorTestSuite struct {
	driver       VMDriver
	driverType   VMType
	testConfig   VMConfig
	cleanup      []func() error
	t            *testing.T
	capabilities DriverCapabilities
}

// DriverCapabilities tracks what a driver supports for capability-based testing
type DriverCapabilities struct {
	SupportsCreate    bool
	SupportsStart     bool
	SupportsStop      bool
	SupportsDelete    bool
	SupportsStatus    bool
	SupportsInfo      bool
	SupportsMetrics   bool
	SupportsListVMs   bool
	SupportsPause     bool
	SupportsResume    bool
	SupportsSnapshot  bool
	SupportsMigrate   bool
}

// NewHypervisorTestSuite creates a new test suite for a hypervisor driver
func NewHypervisorTestSuite(t *testing.T, driver VMDriver, driverType VMType) *HypervisorTestSuite {
	suite := &HypervisorTestSuite{
		driver:     driver,
		driverType: driverType,
		t:          t,
		cleanup:    make([]func() error, 0),
	}

	// Detect driver capabilities
	suite.capabilities = suite.detectCapabilities()

	// Generate test VM config based on driver type
	suite.testConfig = suite.generateTestConfig()

	return suite
}

// AddCleanup adds a cleanup function to be executed after tests
func (s *HypervisorTestSuite) AddCleanup(fn func() error) {
	s.cleanup = append(s.cleanup, fn)
}

// Cleanup performs all registered cleanup operations
func (s *HypervisorTestSuite) Cleanup() {
	for _, cleanupFn := range s.cleanup {
		if err := cleanupFn(); err != nil {
			s.t.Logf("Cleanup error: %v", err)
		}
	}
	s.cleanup = s.cleanup[:0]
}

// detectCapabilities discovers what operations the driver supports
func (s *HypervisorTestSuite) detectCapabilities() DriverCapabilities {
	caps := DriverCapabilities{
		// Basic operations - assumed all drivers support these
		SupportsCreate:  true,
		SupportsStart:   true,
		SupportsStop:    true,
		SupportsDelete:  true,
		SupportsStatus:  true,
		SupportsInfo:    true,
		SupportsMetrics: true,
		SupportsListVMs: true,
	}

	// Check optional operations
	caps.SupportsPause = s.driver.SupportsPause()
	caps.SupportsResume = s.driver.SupportsResume()
	caps.SupportsSnapshot = s.driver.SupportsSnapshot()
	caps.SupportsMigrate = s.driver.SupportsMigrate()

	return caps
}

// generateTestConfig creates appropriate test configuration for the driver type
func (s *HypervisorTestSuite) generateTestConfig() VMConfig {
	baseConfig := VMConfig{
		ID:        fmt.Sprintf("test-vm-%d", time.Now().UnixNano()),
		Name:      fmt.Sprintf("test-vm-%s", s.driverType),
		CPUShares: 2,
		MemoryMB:  512,
		Env:       make(map[string]string),
		Tags:      map[string]string{"test": "true", "driver": string(s.driverType)},
	}

	switch s.driverType {
	case VMTypeKVM:
		baseConfig.DiskSizeGB = 8
		baseConfig.RootFS = "/var/lib/novacron/test-image.qcow2"
	case VMTypeContainer:
		baseConfig.RootFS = "alpine:latest"
		baseConfig.Command = "sleep"
		baseConfig.Args = []string{"3600"}
	case VMTypeContainerd:
		baseConfig.RootFS = "alpine:latest"
		baseConfig.Command = "sleep"
		baseConfig.Args = []string{"3600"}
	case VMTypeKataContainers:
		baseConfig.RootFS = "alpine:latest"
		baseConfig.Command = "sleep"
		baseConfig.Args = []string{"3600"}
	default:
		// Use container defaults for unknown types
		baseConfig.RootFS = "alpine:latest"
		baseConfig.Command = "sleep"
		baseConfig.Args = []string{"60"}
	}

	return baseConfig
}

// TestBasicLifecycle tests create, start, stop, delete operations
func (s *HypervisorTestSuite) TestBasicLifecycle(ctx context.Context) error {
	s.t.Run("BasicLifecycle", func(t *testing.T) {
		vmID, err := s.driver.Create(ctx, s.testConfig)
		if err != nil {
			t.Fatalf("Failed to create VM: %v", err)
		}
		s.AddCleanup(func() error {
			return s.driver.Delete(ctx, vmID)
		})

		// Test VM status after creation
		status, err := s.driver.GetStatus(ctx, vmID)
		if err != nil {
			t.Errorf("Failed to get status after create: %v", err)
		} else if status != StateCreated && status != StateStopped {
			t.Errorf("Expected VM to be created or stopped, got: %v", status)
		}

		// Start the VM
		if err := s.driver.Start(ctx, vmID); err != nil {
			t.Errorf("Failed to start VM: %v", err)
		}

		// Wait for VM to be running
		if err := s.waitForState(ctx, vmID, StateRunning, 30*time.Second); err != nil {
			t.Errorf("VM failed to reach running state: %v", err)
		}

		// Stop the VM
		if err := s.driver.Stop(ctx, vmID); err != nil {
			t.Errorf("Failed to stop VM: %v", err)
		}

		// Wait for VM to be stopped
		if err := s.waitForState(ctx, vmID, StateStopped, 30*time.Second); err != nil {
			t.Errorf("VM failed to reach stopped state: %v", err)
		}

		// Delete the VM
		if err := s.driver.Delete(ctx, vmID); err != nil {
			t.Errorf("Failed to delete VM: %v", err)
		}
	})

	return nil
}

// TestPauseResume tests pause and resume operations if supported
func (s *HypervisorTestSuite) TestPauseResume(ctx context.Context) error {
	if !s.capabilities.SupportsPause || !s.capabilities.SupportsResume {
		s.t.Skip("Driver doesn't support pause/resume")
		return nil
	}

	s.t.Run("PauseResume", func(t *testing.T) {
		vmID, err := s.driver.Create(ctx, s.testConfig)
		if err != nil {
			t.Fatalf("Failed to create VM: %v", err)
		}
		s.AddCleanup(func() error {
			return s.driver.Delete(ctx, vmID)
		})

		// Start VM
		if err := s.driver.Start(ctx, vmID); err != nil {
			t.Fatalf("Failed to start VM: %v", err)
		}

		// Wait for running state
		if err := s.waitForState(ctx, vmID, StateRunning, 30*time.Second); err != nil {
			t.Fatalf("VM failed to reach running state: %v", err)
		}

		// Pause VM
		if err := s.driver.Pause(ctx, vmID); err != nil {
			t.Errorf("Failed to pause VM: %v", err)
		}

		// Check paused state
		if err := s.waitForState(ctx, vmID, StatePaused, 10*time.Second); err != nil {
			t.Errorf("VM failed to reach paused state: %v", err)
		}

		// Resume VM
		if err := s.driver.Resume(ctx, vmID); err != nil {
			t.Errorf("Failed to resume VM: %v", err)
		}

		// Check running state
		if err := s.waitForState(ctx, vmID, StateRunning, 10*time.Second); err != nil {
			t.Errorf("VM failed to return to running state: %v", err)
		}
	})

	return nil
}

// TestSnapshot tests snapshot creation if supported
func (s *HypervisorTestSuite) TestSnapshot(ctx context.Context) error {
	if !s.capabilities.SupportsSnapshot {
		s.t.Skip("Driver doesn't support snapshots")
		return nil
	}

	s.t.Run("Snapshot", func(t *testing.T) {
		vmID, err := s.driver.Create(ctx, s.testConfig)
		if err != nil {
			t.Fatalf("Failed to create VM: %v", err)
		}
		s.AddCleanup(func() error {
			return s.driver.Delete(ctx, vmID)
		})

		// Start VM
		if err := s.driver.Start(ctx, vmID); err != nil {
			t.Fatalf("Failed to start VM: %v", err)
		}

		// Wait for running state
		if err := s.waitForState(ctx, vmID, StateRunning, 30*time.Second); err != nil {
			t.Fatalf("VM failed to reach running state: %v", err)
		}

		// Create snapshot
		snapshotID, err := s.driver.Snapshot(ctx, vmID, "test-snapshot", nil)
		if err != nil {
			t.Errorf("Failed to create snapshot: %v", err)
		} else if snapshotID == "" {
			t.Errorf("Snapshot ID should not be empty")
		}
	})

	return nil
}

// TestGetInfo tests VM information retrieval
func (s *HypervisorTestSuite) TestGetInfo(ctx context.Context) error {
	s.t.Run("GetInfo", func(t *testing.T) {
		vmID, err := s.driver.Create(ctx, s.testConfig)
		if err != nil {
			t.Fatalf("Failed to create VM: %v", err)
		}
		s.AddCleanup(func() error {
			return s.driver.Delete(ctx, vmID)
		})

		// Get VM info
		info, err := s.driver.GetInfo(ctx, vmID)
		if err != nil {
			t.Errorf("Failed to get VM info: %v", err)
		} else if info == nil {
			t.Errorf("VM info should not be nil")
		} else {
			if info.ID != vmID {
				t.Errorf("Expected VM ID %s, got %s", vmID, info.ID)
			}
			if info.Name == "" {
				t.Errorf("VM name should not be empty")
			}
		}
	})

	return nil
}

// TestGetMetrics tests VM metrics retrieval
func (s *HypervisorTestSuite) TestGetMetrics(ctx context.Context) error {
	s.t.Run("GetMetrics", func(t *testing.T) {
		vmID, err := s.driver.Create(ctx, s.testConfig)
		if err != nil {
			t.Fatalf("Failed to create VM: %v", err)
		}
		s.AddCleanup(func() error {
			return s.driver.Delete(ctx, vmID)
		})

		// Get VM metrics
		metrics, err := s.driver.GetMetrics(ctx, vmID)
		if err != nil {
			t.Errorf("Failed to get VM metrics: %v", err)
		} else if metrics == nil {
			t.Errorf("VM metrics should not be nil")
		}
	})

	return nil
}

// TestListVMs tests VM listing functionality
func (s *HypervisorTestSuite) TestListVMs(ctx context.Context) error {
	s.t.Run("ListVMs", func(t *testing.T) {
		// Get initial VM count
		initialVMs, err := s.driver.ListVMs(ctx)
		if err != nil {
			t.Fatalf("Failed to list VMs initially: %v", err)
		}
		initialCount := len(initialVMs)

		// Create test VM
		vmID, err := s.driver.Create(ctx, s.testConfig)
		if err != nil {
			t.Fatalf("Failed to create VM: %v", err)
		}
		s.AddCleanup(func() error {
			return s.driver.Delete(ctx, vmID)
		})

		// List VMs again
		vms, err := s.driver.ListVMs(ctx)
		if err != nil {
			t.Errorf("Failed to list VMs: %v", err)
		} else if len(vms) != initialCount+1 {
			t.Errorf("Expected %d VMs, got %d", initialCount+1, len(vms))
		} else {
			// Check if our VM is in the list
			found := false
			for _, vm := range vms {
				if vm.ID == vmID {
					found = true
					break
				}
			}
			if !found {
				t.Errorf("Created VM %s not found in VM list", vmID)
			}
		}
	})

	return nil
}

// TestErrorHandling tests various error conditions
func (s *HypervisorTestSuite) TestErrorHandling(ctx context.Context) error {
	s.t.Run("ErrorHandling", func(t *testing.T) {
		// Test operations on non-existent VM
		nonExistentID := "non-existent-vm-id"

		if _, err := s.driver.GetStatus(ctx, nonExistentID); err == nil {
			t.Errorf("Expected error when getting status of non-existent VM")
		}

		if _, err := s.driver.GetInfo(ctx, nonExistentID); err == nil {
			t.Errorf("Expected error when getting info of non-existent VM")
		}

		if err := s.driver.Start(ctx, nonExistentID); err == nil {
			t.Errorf("Expected error when starting non-existent VM")
		}

		if err := s.driver.Stop(ctx, nonExistentID); err == nil {
			t.Errorf("Expected error when stopping non-existent VM")
		}

		if err := s.driver.Delete(ctx, nonExistentID); err == nil {
			t.Errorf("Expected error when deleting non-existent VM")
		}

		// Test invalid configuration
		invalidConfig := VMConfig{
			// Missing required fields
			Name: "",
			ID:   "",
		}

		if _, err := s.driver.Create(ctx, invalidConfig); err == nil {
			t.Errorf("Expected error when creating VM with invalid config")
		}
	})

	return nil
}

// TestConcurrentOperations tests concurrent access to the driver
func (s *HypervisorTestSuite) TestConcurrentOperations(ctx context.Context) error {
	s.t.Run("ConcurrentOperations", func(t *testing.T) {
		const numVMs = 5
		vmIDs := make([]string, numVMs)
		errors := make(chan error, numVMs)

		// Create VMs concurrently
		for i := 0; i < numVMs; i++ {
			go func(index int) {
				config := s.testConfig
				config.ID = fmt.Sprintf("concurrent-test-vm-%d-%d", index, time.Now().UnixNano())
				config.Name = fmt.Sprintf("concurrent-test-%d", index)

				vmID, err := s.driver.Create(ctx, config)
				if err != nil {
					errors <- fmt.Errorf("failed to create VM %d: %w", index, err)
					return
				}

				vmIDs[index] = vmID
				errors <- nil
			}(i)
		}

		// Collect results
		for i := 0; i < numVMs; i++ {
			if err := <-errors; err != nil {
				t.Errorf("Concurrent VM creation error: %v", err)
			}
		}

		// Cleanup
		for _, vmID := range vmIDs {
			if vmID != "" {
				s.AddCleanup(func() error {
					return s.driver.Delete(ctx, vmID)
				})
			}
		}
	})

	return nil
}

// RunAllTests executes all test cases in the suite
func (s *HypervisorTestSuite) RunAllTests(ctx context.Context) error {
	defer s.Cleanup()

	tests := []func(context.Context) error{
		s.TestBasicLifecycle,
		s.TestPauseResume,
		s.TestSnapshot,
		s.TestGetInfo,
		s.TestGetMetrics,
		s.TestListVMs,
		s.TestErrorHandling,
		s.TestConcurrentOperations,
	}

	for _, test := range tests {
		if err := test(ctx); err != nil {
			return err
		}
	}

	return nil
}

// waitForState waits for a VM to reach a specific state within the timeout
func (s *HypervisorTestSuite) waitForState(ctx context.Context, vmID string, expectedState State, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		status, err := s.driver.GetStatus(ctx, vmID)
		if err != nil {
			return fmt.Errorf("failed to get VM status: %w", err)
		}

		if status == expectedState {
			return nil
		}

		// Check for failed state
		if status == StateFailed {
			return fmt.Errorf("VM entered failed state while waiting for %s", expectedState)
		}

		time.Sleep(1 * time.Second)
	}

	return fmt.Errorf("timeout waiting for VM to reach state %s", expectedState)
}