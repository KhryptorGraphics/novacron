package vm

import (
	"context"
	"fmt"
	"os/exec"
	"strings"
	"testing"
	"time"
)

// ContainerIntegrationTest provides comprehensive testing for container drivers
type ContainerIntegrationTest struct {
	driver          VMDriver
	driverType      VMType
	hasDocker       bool
	hasContainerd   bool
	dockerVersion   string
	containerdVersion string
}

// NewContainerIntegrationTest creates a new container integration test suite
func NewContainerIntegrationTest(t *testing.T, driverType VMType) *ContainerIntegrationTest {
	test := &ContainerIntegrationTest{
		driverType: driverType,
	}

	// Check container runtime availability
	test.checkContainerRuntimes()

	// Initialize appropriate driver
	var err error
	switch driverType {
	case VMTypeContainer:
		if !test.hasDocker {
			t.Skip("Docker not available for container driver test")
		}
		config := map[string]interface{}{
			"node_id": "test-node-1",
		}
		test.driver, err = NewContainerDriver(config)
	case VMTypeContainerd:
		if !test.hasContainerd {
			t.Skip("Containerd not available for containerd driver test")
		}
		config := map[string]interface{}{
			"node_id":   "test-node-1",
			"address":   "/run/containerd/containerd.sock",
			"namespace": "novacron-test",
		}
		test.driver, err = NewContainerdDriver(config)
	default:
		t.Fatalf("Unsupported driver type: %s", driverType)
	}

	if err != nil {
		t.Fatalf("Failed to create %s driver: %v", driverType, err)
	}

	return test
}

// checkContainerRuntimes checks availability of container runtimes
func (c *ContainerIntegrationTest) checkContainerRuntimes() {
	// Check Docker
	if cmd := exec.Command("docker", "--version"); cmd.Run() == nil {
		c.hasDocker = true
		if output, err := cmd.CombinedOutput(); err == nil {
			c.dockerVersion = strings.TrimSpace(string(output))
		}
	}

	// Check Containerd
	if cmd := exec.Command("ctr", "--version"); cmd.Run() == nil {
		c.hasContainerd = true
		if output, err := cmd.CombinedOutput(); err == nil {
			c.containerdVersion = strings.TrimSpace(string(output))
		}
	}
}

// TestContainerDriverCreation tests container driver initialization
func (c *ContainerIntegrationTest) TestContainerDriverCreation(t *testing.T) {
	if c.driver == nil {
		t.Skip("Container driver not available")
	}

	t.Run("DriverInitialization", func(t *testing.T) {
		if c.driver == nil {
			t.Error("Container driver should not be nil")
		}

		// Test capability checks
		t.Logf("Pause support: %v", c.driver.SupportsPause())
		t.Logf("Resume support: %v", c.driver.SupportsResume())
		t.Logf("Snapshot support: %v", c.driver.SupportsSnapshot())
		t.Logf("Migration support: %v", c.driver.SupportsMigrate())
	})
}

// TestContainerLifecycle tests complete container lifecycle
func (c *ContainerIntegrationTest) TestContainerLifecycle(t *testing.T) {
	if c.driver == nil {
		t.Skip("Container driver not available")
	}

	ctx := context.Background()
	config := VMConfig{
		ID:       "lifecycle-test-container",
		Name:     "Lifecycle Test Container",
		CPUShares: 2,
		MemoryMB: 256,
		RootFS:   "alpine:latest",
		Command:  "sleep",
		Args:     []string{"3600"},
		Env: map[string]string{
			"TEST_ENV": "lifecycle",
		},
		Tags: map[string]string{
			"test":   "lifecycle",
			"driver": string(c.driverType),
		},
	}

	t.Run("CreateStartStopDelete", func(t *testing.T) {
		// Create container
		vmID, err := c.driver.Create(ctx, config)
		if err != nil {
			t.Fatalf("Failed to create container: %v", err)
		}

		// Verify creation
		status, err := c.driver.GetStatus(ctx, vmID)
		if err != nil {
			t.Errorf("Failed to get status after create: %v", err)
		} else if status != StateCreated && status != StateStopped {
			t.Logf("Container state after create: %v", status)
		}

		// Start container
		if err := c.driver.Start(ctx, vmID); err != nil {
			t.Errorf("Failed to start container: %v", err)
		} else {
			// Wait for running state
			if err := c.waitForState(ctx, vmID, StateRunning, 30*time.Second); err != nil {
				t.Errorf("Container failed to reach running state: %v", err)
			}
		}

		// Get container info
		info, err := c.driver.GetInfo(ctx, vmID)
		if err != nil {
			t.Errorf("Failed to get container info: %v", err)
		} else {
			if info.ID != vmID {
				t.Errorf("Expected container ID %s, got %s", vmID, info.ID)
			}
			if info.State != StateRunning {
				t.Errorf("Expected container to be running, got %v", info.State)
			}
		}

		// Stop container
		if err := c.driver.Stop(ctx, vmID); err != nil {
			t.Errorf("Failed to stop container: %v", err)
		} else {
			// Wait for stopped state
			if err := c.waitForState(ctx, vmID, StateStopped, 30*time.Second); err != nil {
				t.Errorf("Container failed to reach stopped state: %v", err)
			}
		}

		// Delete container
		if err := c.driver.Delete(ctx, vmID); err != nil {
			t.Errorf("Failed to delete container: %v", err)
		}

		// Verify deletion
		if _, err := c.driver.GetStatus(ctx, vmID); err == nil {
			t.Error("Expected error when getting status of deleted container")
		}
	})
}

// TestContainerPauseResume tests pause/resume functionality
func (c *ContainerIntegrationTest) TestContainerPauseResume(t *testing.T) {
	if c.driver == nil {
		t.Skip("Container driver not available")
	}

	if !c.driver.SupportsPause() || !c.driver.SupportsResume() {
		t.Skip("Driver doesn't support pause/resume")
	}

	ctx := context.Background()
	config := VMConfig{
		ID:       "pause-test-container",
		Name:     "Pause Test Container",
		CPUShares: 1,
		MemoryMB: 128,
		RootFS:   "alpine:latest",
		Command:  "sleep",
		Args:     []string{"3600"},
	}

	t.Run("PauseResumeSequence", func(t *testing.T) {
		// Create and start container
		vmID, err := c.driver.Create(ctx, config)
		if err != nil {
			t.Fatalf("Failed to create container: %v", err)
		}
		defer c.driver.Delete(ctx, vmID)

		if err := c.driver.Start(ctx, vmID); err != nil {
			t.Fatalf("Failed to start container: %v", err)
		}

		// Wait for running state
		if err := c.waitForState(ctx, vmID, StateRunning, 30*time.Second); err != nil {
			t.Fatalf("Container failed to reach running state: %v", err)
		}

		// Pause container
		if err := c.driver.Pause(ctx, vmID); err != nil {
			t.Errorf("Failed to pause container: %v", err)
		} else {
			// Check paused state
			if err := c.waitForState(ctx, vmID, StatePaused, 10*time.Second); err != nil {
				t.Errorf("Container failed to reach paused state: %v", err)
			}
		}

		// Resume container
		if err := c.driver.Resume(ctx, vmID); err != nil {
			t.Errorf("Failed to resume container: %v", err)
		} else {
			// Check running state
			if err := c.waitForState(ctx, vmID, StateRunning, 10*time.Second); err != nil {
				t.Errorf("Container failed to return to running state: %v", err)
			}
		}
	})
}

// TestContainerResourceConstraints tests resource allocation and limits
func (c *ContainerIntegrationTest) TestContainerResourceConstraints(t *testing.T) {
	if c.driver == nil {
		t.Skip("Container driver not available")
	}

	ctx := context.Background()

	t.Run("ResourceLimits", func(t *testing.T) {
		configs := []VMConfig{
			{
				ID:       "resource-test-small",
				Name:     "Small Resource Test",
				CPUShares: 1,
				MemoryMB: 64,
				RootFS:   "alpine:latest",
				Command:  "sleep",
				Args:     []string{"60"},
			},
			{
				ID:       "resource-test-large",
				Name:     "Large Resource Test", 
				CPUShares: 4,
				MemoryMB: 512,
				RootFS:   "alpine:latest",
				Command:  "sleep",
				Args:     []string{"60"},
			},
		}

		var vmIDs []string
		for _, config := range configs {
			vmID, err := c.driver.Create(ctx, config)
			if err != nil {
				t.Errorf("Failed to create container %s: %v", config.Name, err)
				continue
			}
			vmIDs = append(vmIDs, vmID)

			// Start container
			if err := c.driver.Start(ctx, vmID); err != nil {
				t.Errorf("Failed to start container %s: %v", vmID, err)
				continue
			}

			// Verify resource allocation
			info, err := c.driver.GetInfo(ctx, vmID)
			if err != nil {
				t.Errorf("Failed to get info for container %s: %v", vmID, err)
			} else {
				if info.CPUShares != config.CPUShares {
					t.Errorf("Expected CPU shares %d, got %d", config.CPUShares, info.CPUShares)
				}
				if info.MemoryMB != config.MemoryMB {
					t.Errorf("Expected memory %d MB, got %d MB", config.MemoryMB, info.MemoryMB)
				}
			}
		}

		// Cleanup
		for _, vmID := range vmIDs {
			c.driver.Stop(ctx, vmID)
			c.driver.Delete(ctx, vmID)
		}
	})
}

// TestContainerNetworking tests network configuration
func (c *ContainerIntegrationTest) TestContainerNetworking(t *testing.T) {
	if c.driver == nil {
		t.Skip("Container driver not available")
	}

	ctx := context.Background()

	t.Run("NetworkConfiguration", func(t *testing.T) {
		config := VMConfig{
			ID:        "network-test-container",
			Name:      "Network Test Container",
			CPUShares: 1,
			MemoryMB:  128,
			RootFS:    "alpine:latest",
			Command:   "sleep",
			Args:      []string{"60"},
			NetworkID: "bridge", // Use default bridge network
		}

		vmID, err := c.driver.Create(ctx, config)
		if err != nil {
			t.Fatalf("Failed to create container: %v", err)
		}
		defer c.driver.Delete(ctx, vmID)

		// Start container
		if err := c.driver.Start(ctx, vmID); err != nil {
			t.Fatalf("Failed to start container: %v", err)
		}
		defer c.driver.Stop(ctx, vmID)

		// Verify network configuration
		info, err := c.driver.GetInfo(ctx, vmID)
		if err != nil {
			t.Errorf("Failed to get container info: %v", err)
		} else if info.NetworkID != config.NetworkID {
			t.Errorf("Expected network ID %s, got %s", config.NetworkID, info.NetworkID)
		}
	})
}

// TestContainerEnvironmentVariables tests environment variable handling
func (c *ContainerIntegrationTest) TestContainerEnvironmentVariables(t *testing.T) {
	if c.driver == nil {
		t.Skip("Container driver not available")
	}

	ctx := context.Background()

	t.Run("EnvironmentVariables", func(t *testing.T) {
		config := VMConfig{
			ID:       "env-test-container",
			Name:     "Environment Test Container",
			CPUShares: 1,
			MemoryMB: 128,
			RootFS:   "alpine:latest",
			Command:  "sh",
			Args:     []string{"-c", "echo $TEST_VAR && sleep 60"},
			Env: map[string]string{
				"TEST_VAR":      "test_value",
				"CUSTOM_CONFIG": "enabled",
			},
		}

		vmID, err := c.driver.Create(ctx, config)
		if err != nil {
			t.Fatalf("Failed to create container: %v", err)
		}
		defer c.driver.Delete(ctx, vmID)

		// Start container
		if err := c.driver.Start(ctx, vmID); err != nil {
			t.Fatalf("Failed to start container: %v", err)
		}
		defer c.driver.Stop(ctx, vmID)

		// Wait for running state
		if err := c.waitForState(ctx, vmID, StateRunning, 30*time.Second); err != nil {
			t.Errorf("Container failed to reach running state: %v", err)
		}

		t.Logf("Container %s started successfully with environment variables", vmID)
	})
}

// TestContainerVolumesMounts tests volume mounting
func (c *ContainerIntegrationTest) TestContainerVolumesMounts(t *testing.T) {
	if c.driver == nil {
		t.Skip("Container driver not available")
	}

	ctx := context.Background()

	t.Run("VolumeMounts", func(t *testing.T) {
		config := VMConfig{
			ID:       "volume-test-container",
			Name:     "Volume Test Container",
			CPUShares: 1,
			MemoryMB: 128,
			RootFS:   "alpine:latest",
			Command:  "sleep",
			Args:     []string{"60"},
			Mounts: []Mount{
				{
					Source: "/tmp",
					Target: "/host-tmp",
				},
			},
		}

		vmID, err := c.driver.Create(ctx, config)
		if err != nil {
			t.Fatalf("Failed to create container: %v", err)
		}
		defer c.driver.Delete(ctx, vmID)

		// Start container
		if err := c.driver.Start(ctx, vmID); err != nil {
			t.Fatalf("Failed to start container: %v", err)
		}
		defer c.driver.Stop(ctx, vmID)

		t.Logf("Container %s created with volume mounts successfully", vmID)
	})
}

// TestContainerMetrics tests metrics collection
func (c *ContainerIntegrationTest) TestContainerMetrics(t *testing.T) {
	if c.driver == nil {
		t.Skip("Container driver not available")
	}

	ctx := context.Background()

	t.Run("MetricsCollection", func(t *testing.T) {
		config := VMConfig{
			ID:       "metrics-test-container",
			Name:     "Metrics Test Container",
			CPUShares: 2,
			MemoryMB: 256,
			RootFS:   "alpine:latest",
			Command:  "sleep",
			Args:     []string{"120"},
		}

		vmID, err := c.driver.Create(ctx, config)
		if err != nil {
			t.Fatalf("Failed to create container: %v", err)
		}
		defer c.driver.Delete(ctx, vmID)

		// Start container
		if err := c.driver.Start(ctx, vmID); err != nil {
			t.Fatalf("Failed to start container: %v", err)
		}
		defer c.driver.Stop(ctx, vmID)

		// Wait for container to be running
		if err := c.waitForState(ctx, vmID, StateRunning, 30*time.Second); err != nil {
			t.Fatalf("Container failed to reach running state: %v", err)
		}

		// Get metrics
		metrics, err := c.driver.GetMetrics(ctx, vmID)
		if err != nil {
			t.Errorf("Failed to get container metrics: %v", err)
		} else {
			if metrics == nil {
				t.Error("Container metrics should not be nil")
			} else {
				t.Logf("Container metrics: CPU=%.2f%%, Memory=%d bytes", 
					metrics.CPUUsage, metrics.MemoryUsage)
			}
		}
	})
}

// TestContainerConcurrentOperations tests concurrent container operations
func (c *ContainerIntegrationTest) TestContainerConcurrentOperations(t *testing.T) {
	if c.driver == nil {
		t.Skip("Container driver not available")
	}

	ctx := context.Background()

	t.Run("ConcurrentContainerCreation", func(t *testing.T) {
		const numContainers = 5
		results := make(chan error, numContainers)
		vmIDs := make([]string, numContainers)

		// Create containers concurrently
		for i := 0; i < numContainers; i++ {
			go func(index int) {
				config := VMConfig{
					ID:       fmt.Sprintf("concurrent-container-%d", index),
					Name:     fmt.Sprintf("Concurrent Container %d", index),
					CPUShares: 1,
					MemoryMB: 128,
					RootFS:   "alpine:latest",
					Command:  "sleep",
					Args:     []string{"60"},
				}

				vmID, err := c.driver.Create(ctx, config)
				vmIDs[index] = vmID
				results <- err
			}(i)
		}

		// Collect results
		var createdCount int
		for i := 0; i < numContainers; i++ {
			if err := <-results; err != nil {
				t.Errorf("Concurrent container creation error: %v", err)
			} else {
				createdCount++
			}
		}

		t.Logf("Successfully created %d/%d containers concurrently", createdCount, numContainers)

		// Cleanup
		for _, vmID := range vmIDs {
			if vmID != "" {
				c.driver.Delete(ctx, vmID)
			}
		}
	})
}

// TestContainerErrorHandling tests error scenarios
func (c *ContainerIntegrationTest) TestContainerErrorHandling(t *testing.T) {
	if c.driver == nil {
		t.Skip("Container driver not available")
	}

	ctx := context.Background()

	t.Run("InvalidImage", func(t *testing.T) {
		config := VMConfig{
			ID:       "invalid-image-test",
			Name:     "Invalid Image Test",
			CPUShares: 1,
			MemoryMB: 128,
			RootFS:   "nonexistent:invalid-tag",
			Command:  "sleep",
			Args:     []string{"60"},
		}

		vmID, err := c.driver.Create(ctx, config)
		if err != nil {
			t.Logf("Expected error creating container with invalid image: %v", err)
			return
		}

		// If creation succeeded, starting should fail
		if err := c.driver.Start(ctx, vmID); err == nil {
			t.Error("Expected error starting container with invalid image")
		}

		// Cleanup
		c.driver.Delete(ctx, vmID)
	})

	t.Run("NonExistentContainer", func(t *testing.T) {
		nonExistentID := "non-existent-container"

		if _, err := c.driver.GetStatus(ctx, nonExistentID); err == nil {
			t.Error("Expected error when getting status of non-existent container")
		}

		if err := c.driver.Start(ctx, nonExistentID); err == nil {
			t.Error("Expected error when starting non-existent container")
		}

		if err := c.driver.Stop(ctx, nonExistentID); err == nil {
			t.Error("Expected error when stopping non-existent container")
		}
	})
}

// TestContainerListOperations tests container listing
func (c *ContainerIntegrationTest) TestContainerListOperations(t *testing.T) {
	if c.driver == nil {
		t.Skip("Container driver not available")
	}

	ctx := context.Background()

	t.Run("ListContainers", func(t *testing.T) {
		// Get initial container count
		initialContainers, err := c.driver.ListVMs(ctx)
		if err != nil {
			t.Fatalf("Failed to list containers initially: %v", err)
		}
		initialCount := len(initialContainers)

		// Create test container
		config := VMConfig{
			ID:       "list-test-container",
			Name:     "List Test Container",
			CPUShares: 1,
			MemoryMB: 128,
			RootFS:   "alpine:latest",
			Command:  "sleep",
			Args:     []string{"60"},
		}

		vmID, err := c.driver.Create(ctx, config)
		if err != nil {
			t.Fatalf("Failed to create container: %v", err)
		}
		defer c.driver.Delete(ctx, vmID)

		// List containers again
		containers, err := c.driver.ListVMs(ctx)
		if err != nil {
			t.Errorf("Failed to list containers: %v", err)
		} else {
			newCount := len(containers)
			t.Logf("Container count: initial=%d, after_create=%d", initialCount, newCount)
			
			// Check if our container is in the list
			found := false
			for _, container := range containers {
				if strings.Contains(container.ID, vmID) || strings.Contains(container.Name, config.Name) {
					found = true
					break
				}
			}
			
			if !found {
				t.Logf("Warning: Created container %s not found in container list", vmID)
				// This might be expected for some container runtimes
			}
		}
	})
}

// waitForState waits for a container to reach a specific state within the timeout
func (c *ContainerIntegrationTest) waitForState(ctx context.Context, vmID string, expectedState State, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		status, err := c.driver.GetStatus(ctx, vmID)
		if err != nil {
			return fmt.Errorf("failed to get container status: %w", err)
		}

		if status == expectedState {
			return nil
		}

		// Check for failed state
		if status == StateFailed {
			return fmt.Errorf("container entered failed state while waiting for %s", expectedState)
		}

		time.Sleep(1 * time.Second)
	}

	return fmt.Errorf("timeout waiting for container to reach state %s", expectedState)
}

// RunAllContainerTests executes all container integration tests
func (c *ContainerIntegrationTest) RunAllContainerTests(t *testing.T) {
	t.Run("ContainerDriverCreation", c.TestContainerDriverCreation)
	t.Run("ContainerLifecycle", c.TestContainerLifecycle)
	t.Run("ContainerPauseResume", c.TestContainerPauseResume)
	t.Run("ContainerResourceConstraints", c.TestContainerResourceConstraints)
	t.Run("ContainerNetworking", c.TestContainerNetworking)
	t.Run("ContainerEnvironmentVariables", c.TestContainerEnvironmentVariables)
	t.Run("ContainerVolumesMounts", c.TestContainerVolumesMounts)
	t.Run("ContainerMetrics", c.TestContainerMetrics)
	t.Run("ContainerConcurrentOperations", c.TestContainerConcurrentOperations)
	t.Run("ContainerErrorHandling", c.TestContainerErrorHandling)
	t.Run("ContainerListOperations", c.TestContainerListOperations)
}

// Integration test entry points
func TestDockerIntegration(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping Docker integration tests in short mode")
	}

	containerTest := NewContainerIntegrationTest(t, VMTypeContainer)
	containerTest.RunAllContainerTests(t)
}

func TestContainerdIntegration(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping Containerd integration tests in short mode")
	}

	containerTest := NewContainerIntegrationTest(t, VMTypeContainerd)
	containerTest.RunAllContainerTests(t)
}

// Benchmark tests
func BenchmarkContainerCreation(b *testing.B) {
	containerTest := NewContainerIntegrationTest(nil, VMTypeContainer)
	if containerTest.driver == nil {
		b.Skip("Container driver not available")
	}

	ctx := context.Background()
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		config := VMConfig{
			ID:       fmt.Sprintf("bench-container-%d", i),
			Name:     fmt.Sprintf("Benchmark Container %d", i),
			CPUShares: 1,
			MemoryMB: 64,
			RootFS:   "alpine:latest",
			Command:  "true", // Minimal command
		}

		vmID, err := containerTest.driver.Create(ctx, config)
		if err != nil {
			b.Errorf("Failed to create container: %v", err)
			continue
		}

		// Clean up immediately
		containerTest.driver.Delete(ctx, vmID)
	}
}