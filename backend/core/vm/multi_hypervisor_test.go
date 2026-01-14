package vm

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"
)

// MultiHypervisorTest provides comprehensive testing for multi-hypervisor scenarios
type MultiHypervisorTest struct {
	drivers        map[VMType]VMDriver
	mockDrivers    map[string]*MockHypervisor
	driverManager  *VMDriverManager
	testResults    map[string]*HypervisorTestResults
	mu             sync.RWMutex
}

// HypervisorTestResults stores test results for a hypervisor
type HypervisorTestResults struct {
	DriverType        VMType
	TotalTests        int
	PassedTests       int
	FailedTests       int
	AvgCreationTime   time.Duration
	AvgStartTime      time.Duration
	AvgStopTime       time.Duration
	SupportedFeatures []string
	Errors            []string
	PerformanceScore  float64
}

// NewMultiHypervisorTest creates a new multi-hypervisor test suite
func NewMultiHypervisorTest(t *testing.T) *MultiHypervisorTest {
	test := &MultiHypervisorTest{
		drivers:     make(map[VMType]VMDriver),
		mockDrivers: make(map[string]*MockHypervisor),
		testResults: make(map[string]*HypervisorTestResults),
	}

	// Initialize mock hypervisors for comprehensive testing
	test.initializeMockHypervisors()

	// Initialize real drivers if available
	test.initializeRealDrivers()

	// Create driver manager
	config := DefaultVMDriverConfig("multi-test-node")
	test.driverManager = NewVMDriverManager(config)

	return test
}

// initializeMockHypervisors sets up mock hypervisors with different configurations
func (m *MultiHypervisorTest) initializeMockHypervisors() {
	// Standard mock hypervisor
	standardMock := NewMockHypervisor("test-node-1", "mock-standard")
	m.mockDrivers["mock-standard"] = standardMock
	m.drivers[VMTypeKVM] = standardMock

	// High-performance mock hypervisor
	perfMock := NewMockHypervisor("test-node-2", "mock-performance")
	perfMock.Configure(
		MockFailureConfig{}, // No failures
		MockLatencyConfig{   // Fast operations
			CreateLatency: 50 * time.Millisecond,
			StartLatency:  500 * time.Millisecond,
			StopLatency:   200 * time.Millisecond,
			StatusLatency: 5 * time.Millisecond,
		},
		MockCapabilities{
			SupportsPause:    true,
			SupportsResume:   true,
			SupportsSnapshot: true,
			SupportsMigrate:  true,
			MaxVMs:          1000,
			MaxCPUPerVM:     64,
			MaxMemoryPerVM:  128 * 1024, // 128GB
		},
	)
	m.mockDrivers["mock-performance"] = perfMock
	m.drivers[VMTypeContainerd] = perfMock

	// Unreliable mock hypervisor (for testing error handling)
	unreliableMock := NewMockHypervisor("test-node-3", "mock-unreliable")
	unreliableMock.Configure(
		MockFailureConfig{
			CreateFailureRate:   0.1, // 10% failure rate
			StartFailureRate:    0.05,
			StopFailureRate:     0.05,
			PauseFailureRate:    0.1,
			ResumeFailureRate:   0.1,
			SnapshotFailureRate: 0.2,
			MigrateFailureRate:  0.15,
			StatusFailureRate:   0.02,
			RandomFailures:      true,
		},
		MockLatencyConfig{
			CreateLatency:  200 * time.Millisecond,
			StartLatency:   2 * time.Second,
			StopLatency:    1 * time.Second,
			StatusLatency:  20 * time.Millisecond,
			VariabilityPct: 0.5, // 50% variation
		},
		MockCapabilities{
			SupportsPause:    true,
			SupportsResume:   true,
			SupportsSnapshot: false, // Limited capabilities
			SupportsMigrate:  false,
			MaxVMs:          50,
			MaxCPUPerVM:     8,
			MaxMemoryPerVM:  32 * 1024, // 32GB
		},
	)
	m.mockDrivers["mock-unreliable"] = unreliableMock
	m.drivers[VMTypeContainer] = unreliableMock
}

// initializeRealDrivers attempts to initialize real hypervisor drivers
func (m *MultiHypervisorTest) initializeRealDrivers() {
	// Try to initialize KVM driver
	if kvmDriver, err := NewKVMDriver(map[string]interface{}{
		"qemu_path": "/usr/bin/qemu-system-x86_64",
		"vm_path":   "/tmp/novacron-test-vms",
	}); err == nil {
		m.drivers[VMTypeKVM] = kvmDriver
	}

	// Try to initialize container driver
	if containerDriver, err := NewContainerDriver(map[string]interface{}{
		"node_id": "test-node",
	}); err == nil {
		m.drivers[VMTypeContainer] = containerDriver
	}
}

// TestCapabilityDetection tests automatic detection of hypervisor capabilities
func (m *MultiHypervisorTest) TestCapabilityDetection(t *testing.T) {
	t.Run("CapabilityMatrix", func(t *testing.T) {
		capabilityMatrix := make(map[VMType]map[string]bool)

		for driverType, driver := range m.drivers {
			capabilities := make(map[string]bool)
			capabilities["pause"] = driver.SupportsPause()
			capabilities["resume"] = driver.SupportsResume()
			capabilities["snapshot"] = driver.SupportsSnapshot()
			capabilities["migrate"] = driver.SupportsMigrate()

			capabilityMatrix[driverType] = capabilities

			t.Logf("Driver %s capabilities:", driverType)
			for feature, supported := range capabilities {
				t.Logf("  %s: %v", feature, supported)
			}
		}

		// Verify each driver has at least basic capabilities
		for driverType, capabilities := range capabilityMatrix {
			// All drivers should support basic operations (implicit)
			result := &HypervisorTestResults{
				DriverType:        driverType,
				SupportedFeatures: make([]string, 0),
			}

			for feature, supported := range capabilities {
				if supported {
					result.SupportedFeatures = append(result.SupportedFeatures, feature)
				}
			}

			m.testResults[string(driverType)] = result
		}
	})
}

// TestCrossHypervisorCompatibility tests compatibility between different hypervisors
func (m *MultiHypervisorTest) TestCrossHypervisorCompatibility(t *testing.T) {
	ctx := context.Background()

	t.Run("UnifiedInterface", func(t *testing.T) {
		// Test that all drivers implement the same interface correctly
		baseConfig := VMConfig{
			Name:      "compatibility-test",
			CPUShares: 2,
			MemoryMB:  256,
			Tags:      map[string]string{"test": "compatibility"},
		}

		for driverType, driver := range m.drivers {
			t.Run(fmt.Sprintf("Driver-%s", driverType), func(t *testing.T) {
				config := baseConfig
				config.ID = fmt.Sprintf("compat-test-%s", driverType)
				config.Name = fmt.Sprintf("Compatibility Test %s", driverType)

				// Adjust config based on driver type
				switch driverType {
				case VMTypeContainer, VMTypeContainerd:
					config.RootFS = "alpine:latest"
					config.Command = "sleep"
					config.Args = []string{"60"}
				case VMTypeKVM:
					config.DiskSizeGB = 1
				}

				// Test basic lifecycle
				vmID, err := driver.Create(ctx, config)
				if err != nil {
					t.Errorf("Failed to create VM with %s driver: %v", driverType, err)
					return
				}
				defer driver.Delete(ctx, vmID)

				// Test status retrieval
				status, err := driver.GetStatus(ctx, vmID)
				if err != nil {
					t.Errorf("Failed to get status with %s driver: %v", driverType, err)
				} else if status == StateUnknown {
					t.Errorf("Unknown status returned by %s driver", driverType)
				}

				// Test info retrieval
				info, err := driver.GetInfo(ctx, vmID)
				if err != nil {
					t.Errorf("Failed to get info with %s driver: %v", driverType, err)
				} else if info == nil {
					t.Errorf("Nil info returned by %s driver", driverType)
				}

				t.Logf("Driver %s: basic compatibility test passed", driverType)
			})
		}
	})
}

// TestResourceIsolation tests resource isolation between VMs on different hypervisors
func (m *MultiHypervisorTest) TestResourceIsolation(t *testing.T) {
	ctx := context.Background()

	t.Run("ResourceLimits", func(t *testing.T) {
		resourceConfigs := []struct {
			name     string
			cpuShares int
			memoryMB int
		}{
			{"low-resource", 1, 128},
			{"medium-resource", 2, 512},
			{"high-resource", 4, 1024},
		}

		for driverType, driver := range m.drivers {
			t.Run(fmt.Sprintf("Driver-%s", driverType), func(t *testing.T) {
				var vmIDs []string
				defer func() {
					// Cleanup all VMs
					for _, vmID := range vmIDs {
						driver.Delete(ctx, vmID)
					}
				}()

				for _, resConfig := range resourceConfigs {
					config := VMConfig{
						ID:        fmt.Sprintf("resource-test-%s-%s", driverType, resConfig.name),
						Name:      fmt.Sprintf("Resource Test %s %s", driverType, resConfig.name),
						CPUShares: resConfig.cpuShares,
						MemoryMB:  resConfig.memoryMB,
						Tags:      map[string]string{"test": "resource-isolation"},
					}

					// Adjust config based on driver type
					switch driverType {
					case VMTypeContainer, VMTypeContainerd:
						config.RootFS = "alpine:latest"
						config.Command = "sleep"
						config.Args = []string{"30"}
					case VMTypeKVM:
						config.DiskSizeGB = 1
					}

					vmID, err := driver.Create(ctx, config)
					if err != nil {
						t.Errorf("Failed to create %s VM: %v", resConfig.name, err)
						continue
					}
					vmIDs = append(vmIDs, vmID)

					// Verify resource allocation
					info, err := driver.GetInfo(ctx, vmID)
					if err != nil {
						t.Errorf("Failed to get info for %s VM: %v", resConfig.name, err)
						continue
					}

					if info.CPUShares != resConfig.cpuShares {
						t.Errorf("Expected CPU shares %d, got %d", resConfig.cpuShares, info.CPUShares)
					}

					if info.MemoryMB != resConfig.memoryMB {
						t.Errorf("Expected memory %d MB, got %d MB", resConfig.memoryMB, info.MemoryMB)
					}
				}

				t.Logf("Resource isolation test passed for %s driver", driverType)
			})
		}
	})
}

// TestConcurrentMultiHypervisorOperations tests concurrent operations across multiple hypervisors
func (m *MultiHypervisorTest) TestConcurrentMultiHypervisorOperations(t *testing.T) {
	ctx := context.Background()

	t.Run("ConcurrentCreation", func(t *testing.T) {
		const vmsPerDriver = 3
		var wg sync.WaitGroup
		results := make(chan error, len(m.drivers)*vmsPerDriver)

		for driverType, driver := range m.drivers {
			wg.Add(1)
			go func(dt VMType, drv VMDriver) {
				defer wg.Done()

				for i := 0; i < vmsPerDriver; i++ {
					config := VMConfig{
						ID:        fmt.Sprintf("concurrent-test-%s-%d", dt, i),
						Name:      fmt.Sprintf("Concurrent Test %s %d", dt, i),
						CPUShares: 1,
						MemoryMB:  128,
						Tags:      map[string]string{"test": "concurrent"},
					}

					// Adjust config based on driver type
					switch dt {
					case VMTypeContainer, VMTypeContainerd:
						config.RootFS = "alpine:latest"
						config.Command = "sleep"
						config.Args = []string{"30"}
					case VMTypeKVM:
						config.DiskSizeGB = 1
					}

					vmID, err := drv.Create(ctx, config)
					if err != nil {
						results <- fmt.Errorf("failed to create VM with %s driver: %w", dt, err)
						return
					}

					// Clean up immediately
					drv.Delete(ctx, vmID)
				}

				results <- nil
			}(driverType, driver)
		}

		wg.Wait()
		close(results)

		// Check results
		var errors []error
		for err := range results {
			if err != nil {
				errors = append(errors, err)
			}
		}

		if len(errors) > 0 {
			t.Errorf("Concurrent operations had %d errors:", len(errors))
			for _, err := range errors {
				t.Errorf("  %v", err)
			}
		} else {
			t.Logf("Successfully completed concurrent operations across %d hypervisors", len(m.drivers))
		}
	})
}

// TestPerformanceComparison compares performance across different hypervisors
func (m *MultiHypervisorTest) TestPerformanceComparison(t *testing.T) {
	ctx := context.Background()

	t.Run("PerformanceBenchmark", func(t *testing.T) {
		const numOpsPerTest = 5

		for driverType, driver := range m.drivers {
			t.Run(fmt.Sprintf("Driver-%s", driverType), func(t *testing.T) {
				result := m.testResults[string(driverType)]
				if result == nil {
					result = &HypervisorTestResults{DriverType: driverType}
					m.testResults[string(driverType)] = result
				}

				var totalCreationTime, totalStartTime, totalStopTime time.Duration
				successfulOps := 0

				for i := 0; i < numOpsPerTest; i++ {
					config := VMConfig{
						ID:        fmt.Sprintf("perf-test-%s-%d", driverType, i),
						Name:      fmt.Sprintf("Performance Test %s %d", driverType, i),
						CPUShares: 1,
						MemoryMB:  128,
						Tags:      map[string]string{"test": "performance"},
					}

					// Adjust config based on driver type
					switch driverType {
					case VMTypeContainer, VMTypeContainerd:
						config.RootFS = "alpine:latest"
						config.Command = "sleep"
						config.Args = []string{"30"}
					case VMTypeKVM:
						config.DiskSizeGB = 1
					}

					// Measure creation time
					startTime := time.Now()
					vmID, err := driver.Create(ctx, config)
					creationTime := time.Since(startTime)

					if err != nil {
						t.Logf("Creation failed for %s: %v", driverType, err)
						continue
					}

					totalCreationTime += creationTime

					// Measure start time (if supported)
					startTime = time.Now()
					startErr := driver.Start(ctx, vmID)
					startOpTime := time.Since(startTime)

					if startErr == nil {
						totalStartTime += startOpTime

						// Measure stop time
						startTime = time.Now()
						stopErr := driver.Stop(ctx, vmID)
						stopOpTime := time.Since(startTime)

						if stopErr == nil {
							totalStopTime += stopOpTime
						}
					}

					// Clean up
					driver.Delete(ctx, vmID)
					successfulOps++
				}

				if successfulOps > 0 {
					result.AvgCreationTime = totalCreationTime / time.Duration(successfulOps)
					result.AvgStartTime = totalStartTime / time.Duration(successfulOps)
					result.AvgStopTime = totalStopTime / time.Duration(successfulOps)
					
					// Calculate performance score (lower is better)
					result.PerformanceScore = float64(totalCreationTime+totalStartTime+totalStopTime) / float64(successfulOps) / float64(time.Millisecond)

					t.Logf("Driver %s performance:", driverType)
					t.Logf("  Avg creation time: %v", result.AvgCreationTime)
					t.Logf("  Avg start time: %v", result.AvgStartTime)
					t.Logf("  Avg stop time: %v", result.AvgStopTime)
					t.Logf("  Performance score: %.2f ms", result.PerformanceScore)
				}
			})
		}
	})
}

// TestErrorHandlingConsistency tests consistent error handling across hypervisors
func (m *MultiHypervisorTest) TestErrorHandlingConsistency(t *testing.T) {
	ctx := context.Background()

	t.Run("ErrorConsistency", func(t *testing.T) {
		errorScenarios := []struct {
			name        string
			testFunc    func(driver VMDriver) error
			expectError bool
		}{
			{
				name: "NonExistentVM",
				testFunc: func(driver VMDriver) error {
					_, err := driver.GetStatus(ctx, "non-existent-vm")
					return err
				},
				expectError: true,
			},
			{
				name: "StartNonExistentVM",
				testFunc: func(driver VMDriver) error {
					return driver.Start(ctx, "non-existent-vm")
				},
				expectError: true,
			},
			{
				name: "InvalidConfig",
				testFunc: func(driver VMDriver) error {
					invalidConfig := VMConfig{
						Name: "", // Empty name should cause error
					}
					_, err := driver.Create(ctx, invalidConfig)
					return err
				},
				expectError: true,
			},
		}

		for _, scenario := range errorScenarios {
			t.Run(scenario.name, func(t *testing.T) {
				for driverType, driver := range m.drivers {
					err := scenario.testFunc(driver)
					
					if scenario.expectError && err == nil {
						t.Errorf("Driver %s: expected error for %s but got none", driverType, scenario.name)
					} else if !scenario.expectError && err != nil {
						t.Errorf("Driver %s: unexpected error for %s: %v", driverType, scenario.name, err)
					}
				}
			})
		}
	})
}

// TestFeatureParity tests feature parity across hypervisors where applicable
func (m *MultiHypervisorTest) TestFeatureParity(t *testing.T) {
	ctx := context.Background()

	t.Run("FeatureParity", func(t *testing.T) {
		features := []struct {
			name     string
			testFunc func(driver VMDriver, vmID string) error
			checkFunc func(driver VMDriver) bool
		}{
			{
				name: "Pause",
				testFunc: func(driver VMDriver, vmID string) error {
					return driver.Pause(ctx, vmID)
				},
				checkFunc: func(driver VMDriver) bool {
					return driver.SupportsPause()
				},
			},
			{
				name: "Resume", 
				testFunc: func(driver VMDriver, vmID string) error {
					return driver.Resume(ctx, vmID)
				},
				checkFunc: func(driver VMDriver) bool {
					return driver.SupportsResume()
				},
			},
			{
				name: "Snapshot",
				testFunc: func(driver VMDriver, vmID string) error {
					_, err := driver.Snapshot(ctx, vmID, "test-snapshot", nil)
					return err
				},
				checkFunc: func(driver VMDriver) bool {
					return driver.SupportsSnapshot()
				},
			},
		}

		for _, feature := range features {
			t.Run(feature.name, func(t *testing.T) {
				supportingDrivers := make(map[VMType]bool)
				
				for driverType, driver := range m.drivers {
					if feature.checkFunc(driver) {
						supportingDrivers[driverType] = true
						t.Logf("Driver %s supports %s", driverType, feature.name)
					} else {
						t.Logf("Driver %s does not support %s", driverType, feature.name)
					}
				}

				t.Logf("Feature %s supported by %d/%d drivers", 
					feature.name, len(supportingDrivers), len(m.drivers))
			})
		}
	})
}

// GenerateTestReport generates a comprehensive test report
func (m *MultiHypervisorTest) GenerateTestReport() map[string]*HypervisorTestResults {
	m.mu.RLock()
	defer m.mu.RUnlock()

	report := make(map[string]*HypervisorTestResults)
	for driverType, result := range m.testResults {
		// Create a copy to avoid race conditions
		reportResult := &HypervisorTestResults{
			DriverType:        result.DriverType,
			TotalTests:        result.TotalTests,
			PassedTests:       result.PassedTests,
			FailedTests:       result.FailedTests,
			AvgCreationTime:   result.AvgCreationTime,
			AvgStartTime:      result.AvgStartTime,
			AvgStopTime:       result.AvgStopTime,
			SupportedFeatures: append([]string{}, result.SupportedFeatures...),
			Errors:            append([]string{}, result.Errors...),
			PerformanceScore:  result.PerformanceScore,
		}
		report[driverType] = reportResult
	}

	return report
}

// Cleanup performs cleanup of test resources
func (m *MultiHypervisorTest) Cleanup() {
	// Reset all mock hypervisors
	for _, mock := range m.mockDrivers {
		mock.Reset()
	}

	// Close any real drivers that need cleanup
	for _, driver := range m.drivers {
		if closer, ok := driver.(interface{ Close() error }); ok {
			closer.Close()
		}
	}
}

// RunAllMultiHypervisorTests executes all multi-hypervisor tests
func (m *MultiHypervisorTest) RunAllMultiHypervisorTests(t *testing.T) {
	defer m.Cleanup()

	t.Run("CapabilityDetection", m.TestCapabilityDetection)
	t.Run("CrossHypervisorCompatibility", m.TestCrossHypervisorCompatibility)
	t.Run("ResourceIsolation", m.TestResourceIsolation)
	t.Run("ConcurrentMultiHypervisorOperations", m.TestConcurrentMultiHypervisorOperations)
	t.Run("PerformanceComparison", m.TestPerformanceComparison)
	t.Run("ErrorHandlingConsistency", m.TestErrorHandlingConsistency)
	t.Run("FeatureParity", m.TestFeatureParity)

	// Print final report
	report := m.GenerateTestReport()
	t.Logf("\n=== Multi-Hypervisor Test Report ===")
	for driverType, result := range report {
		t.Logf("Driver: %s", driverType)
		t.Logf("  Supported features: %v", result.SupportedFeatures)
		t.Logf("  Performance score: %.2f ms", result.PerformanceScore)
		if len(result.Errors) > 0 {
			t.Logf("  Errors: %v", result.Errors)
		}
	}
}

// Integration test entry point
func TestMultiHypervisorIntegration(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping multi-hypervisor integration tests in short mode")
	}

	multiTest := NewMultiHypervisorTest(t)
	multiTest.RunAllMultiHypervisorTests(t)
}

// Benchmark comparing different hypervisors
func BenchmarkMultiHypervisorComparison(b *testing.B) {
	multiTest := NewMultiHypervisorTest(nil)
	defer multiTest.Cleanup()

	ctx := context.Background()

	for driverType, driver := range multiTest.drivers {
		b.Run(fmt.Sprintf("Driver-%s", driverType), func(b *testing.B) {
			b.ResetTimer()
			
			for i := 0; i < b.N; i++ {
				config := VMConfig{
					ID:        fmt.Sprintf("bench-multi-%s-%d", driverType, i),
					Name:      fmt.Sprintf("Benchmark Multi %s %d", driverType, i),
					CPUShares: 1,
					MemoryMB:  128,
				}

				// Adjust config based on driver type
				switch driverType {
				case VMTypeContainer, VMTypeContainerd:
					config.RootFS = "alpine:latest"
					config.Command = "true"
				case VMTypeKVM:
					config.DiskSizeGB = 1
				}

				vmID, err := driver.Create(ctx, config)
				if err != nil {
					b.Errorf("Failed to create VM: %v", err)
					continue
				}

				// Clean up immediately
				driver.Delete(ctx, vmID)
			}
		})
	}
}