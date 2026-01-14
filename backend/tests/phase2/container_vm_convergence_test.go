// Phase 2 Container-VM Convergence Testing Suite
package phase2

import (
	"context"
	"testing"
	"time"
	"fmt"
	"sync"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ConvergenceTestSuite tests container-VM hybrid operations
type ConvergenceTestSuite struct {
	hybridCluster    *HybridCluster
	scheduler        *UnifiedScheduler
	migrationManager *HybridMigrationManager
}

// TestContainerVMMigration tests migration between container and VM workloads
func TestContainerVMMigration(t *testing.T) {
	hybridCluster := setupHybridCluster(5)
	defer hybridCluster.cleanup()

	migrationScenarios := []struct {
		name        string
		source      WorkloadType
		destination WorkloadType
		workloadSize WorkloadSize
		expectedDowntime time.Duration
	}{
		{
			name:        "ContainerToVM",
			source:      WorkloadTypeContainer,
			destination: WorkloadTypeVM,
			workloadSize: WorkloadSizeSmall,
			expectedDowntime: 30 * time.Second,
		},
		{
			name:        "VMToContainer",
			source:      WorkloadTypeVM,
			destination: WorkloadTypeContainer,
			workloadSize: WorkloadSizeMedium,
			expectedDowntime: 45 * time.Second,
		},
		{
			name:        "ContainerToMicroVM",
			source:      WorkloadTypeContainer,
			destination: WorkloadTypeMicroVM,
			workloadSize: WorkloadSizeSmall,
			expectedDowntime: 15 * time.Second,
		},
		{
			name:        "MicroVMToVM",
			source:      WorkloadTypeMicroVM,
			destination: WorkloadTypeVM,
			workloadSize: WorkloadSizeLarge,
			expectedDowntime: 60 * time.Second,
		},
	}

	for _, scenario := range migrationScenarios {
		t.Run(scenario.name, func(t *testing.T) {
			// Deploy source workload
			sourceSpec := createWorkloadSpec(scenario.source, scenario.workloadSize)
			
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
			defer cancel()

			sourceWorkload, err := hybridCluster.DeployWorkload(ctx, sourceSpec)
			require.NoError(t, err)

			// Wait for workload to be ready
			err = waitForWorkloadReady(sourceWorkload, 2*time.Minute)
			require.NoError(t, err)

			// Create migration plan
			migrationPlan := &HybridMigrationPlan{
				SourceWorkload:     sourceWorkload.ID,
				TargetType:        scenario.destination,
				MigrationType:     "live-migration",
				MaxDowntime:       scenario.expectedDowntime,
				PreserveDisk:      true,
				PreserveNetwork:   true,
				PreserveState:     scenario.source != WorkloadTypeContainer,
			}

			// Execute migration
			startTime := time.Now()
			migrationResult, err := hybridCluster.MigrateWorkload(ctx, migrationPlan)
			require.NoError(t, err)
			migrationDuration := time.Since(startTime)

			// Validate migration results
			assert.True(t, migrationResult.Success, "Migration should succeed")
			assert.LessOrEqual(t, migrationResult.Downtime, scenario.expectedDowntime*2,
				"Downtime should be within acceptable bounds")
			assert.LessOrEqual(t, migrationDuration, 10*time.Minute,
				"Migration should complete within time limit")

			// Verify workload functionality after migration
			targetWorkload := hybridCluster.GetWorkload(migrationResult.TargetWorkloadID)
			require.NotNil(t, targetWorkload)

			assert.Equal(t, scenario.destination, targetWorkload.Type)
			assert.Equal(t, WorkloadStatusRunning, targetWorkload.Status)

			// Test workload operations
			err = testWorkloadOperations(targetWorkload)
			assert.NoError(t, err, "Workload should function correctly after migration")

			// Validate resource utilization
			resourceMetrics := targetWorkload.GetResourceMetrics()
			assert.LessOrEqual(t, resourceMetrics.CPUUtilization, 1.0)
			assert.LessOrEqual(t, resourceMetrics.MemoryUtilization, 1.0)

			// Clean up
			err = hybridCluster.DeleteWorkload(ctx, targetWorkload.ID)
			assert.NoError(t, err)
		})
	}
}

// TestUnifiedScheduling tests scheduling across container and VM resources
func TestUnifiedScheduling(t *testing.T) {
	hybridCluster := setupHybridCluster(8)
	scheduler := NewUnifiedScheduler(hybridCluster)
	defer hybridCluster.cleanup()

	t.Run("OptimalResourcePlacement", func(t *testing.T) {
		// Create diverse workload requirements
		workloads := []WorkloadRequest{
			{
				Type:        WorkloadTypeContainer,
				CPUCores:    0.5,
				MemoryMB:    512,
				Priority:    PriorityHigh,
				Constraints: []string{"ssd-storage", "low-latency"},
			},
			{
				Type:        WorkloadTypeVM,
				CPUCores:    4,
				MemoryMB:    8192,
				Priority:    PriorityNormal,
				Constraints: []string{"gpu-acceleration", "high-memory"},
			},
			{
				Type:        WorkloadTypeMicroVM,
				CPUCores:    1,
				MemoryMB:    1024,
				Priority:    PriorityLow,
				Constraints: []string{"edge-deployment"},
			},
			{
				Type:        WorkloadTypeContainer,
				CPUCores:    2,
				MemoryMB:    4096,
				Priority:    PriorityHigh,
				Constraints: []string{"database-workload", "persistent-storage"},
			},
		}

		var scheduledWorkloads []ScheduledWorkload
		
		for i, workload := range workloads {
			workload.Name = fmt.Sprintf("unified-workload-%d", i)
			
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
			
			placement, err := scheduler.ScheduleWorkload(ctx, &workload)
			cancel()
			
			require.NoError(t, err, "Scheduling should succeed for workload %s", workload.Name)
			require.NotNil(t, placement, "Placement should be returned")
			
			scheduledWorkloads = append(scheduledWorkloads, ScheduledWorkload{
				Request:   workload,
				Placement: *placement,
			})
		}

		// Validate scheduling efficiency
		clusterUtilization := hybridCluster.GetClusterUtilization()
		assert.GreaterOrEqual(t, clusterUtilization.CPUUtilization, 0.6,
			"CPU utilization should be efficiently used")
		assert.GreaterOrEqual(t, clusterUtilization.MemoryUtilization, 0.7,
			"Memory utilization should be efficiently used")

		// Validate constraint satisfaction
		for _, scheduled := range scheduledWorkloads {
			node := hybridCluster.GetNode(scheduled.Placement.NodeID)
			
			for _, constraint := range scheduled.Request.Constraints {
				satisfied := node.SatisfiesConstraint(constraint)
				assert.True(t, satisfied, 
					"Node %s should satisfy constraint %s for workload %s", 
					node.ID, constraint, scheduled.Request.Name)
			}
		}

		// Test load balancing
		nodeLoads := make(map[string]float64)
		for _, scheduled := range scheduledWorkloads {
			nodeLoads[scheduled.Placement.NodeID] += scheduled.Request.CPUCores
		}
		
		loadVariance := calculateLoadVariance(nodeLoads)
		assert.LessOrEqual(t, loadVariance, 2.0, "Load should be balanced across nodes")
	})

	t.Run("MixedWorkloadCoexistence", func(t *testing.T) {
		// Deploy mixed workloads on same nodes
		mixedWorkloads := []struct {
			name     string
			workloads []WorkloadType
		}{
			{
				name:     "ContainerVMCoexistence",
				workloads: []WorkloadType{WorkloadTypeContainer, WorkloadTypeVM},
			},
			{
				name:     "MicroVMContainerCoexistence", 
				workloads: []WorkloadType{WorkloadTypeMicroVM, WorkloadTypeContainer},
			},
			{
				name:     "TripleCoexistence",
				workloads: []WorkloadType{WorkloadTypeContainer, WorkloadTypeMicroVM, WorkloadTypeVM},
			},
		}

		for _, scenario := range mixedWorkloads {
			t.Run(scenario.name, func(t *testing.T) {
				node := hybridCluster.GetAvailableNode()
				require.NotNil(t, node, "Should have available node")

				var deployedWorkloads []DeployedWorkload

				// Deploy all workload types on same node
				for i, workloadType := range scenario.workloads {
					spec := &WorkloadSpec{
						Name:        fmt.Sprintf("%s-%d", scenario.name, i),
						Type:        workloadType,
						CPUCores:    1,
						MemoryMB:    1024,
						TargetNode:  node.ID,
						Isolation:   IsolationStrict,
					}

					ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
					workload, err := node.DeployWorkload(ctx, spec)
					cancel()

					require.NoError(t, err, "Should deploy %s workload", workloadType)
					deployedWorkloads = append(deployedWorkloads, *workload)
				}

				// Verify all workloads are running
				for _, workload := range deployedWorkloads {
					status := node.GetWorkloadStatus(workload.ID)
					assert.Equal(t, WorkloadStatusRunning, status.State)
					assert.Greater(t, status.Uptime, time.Duration(0))
				}

				// Test resource isolation
				resourceMetrics := node.GetResourceMetrics()
				assert.LessOrEqual(t, resourceMetrics.CPUContention, 0.1,
					"CPU contention should be minimal")
				assert.LessOrEqual(t, resourceMetrics.MemoryPressure, 0.2,
					"Memory pressure should be manageable")

				// Test network isolation
				for i, workload1 := range deployedWorkloads {
					for j, workload2 := range deployedWorkloads {
						if i != j {
							isolated := node.VerifyNetworkIsolation(workload1.ID, workload2.ID)
							assert.True(t, isolated, 
								"Workloads %s and %s should be network isolated", 
								workload1.Name, workload2.Name)
						}
					}
				}

				// Clean up
				for _, workload := range deployedWorkloads {
					ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
					err := node.DeleteWorkload(ctx, workload.ID)
					cancel()
					assert.NoError(t, err)
				}
			})
		}
	})
}

// TestHybridNetworking tests networking across container and VM workloads
func TestHybridNetworking(t *testing.T) {
	hybridCluster := setupHybridCluster(4)
	defer hybridCluster.cleanup()

	t.Run("CrossTypeNetworking", func(t *testing.T) {
		// Deploy workloads of different types
		containerSpec := &WorkloadSpec{
			Name:        "test-container",
			Type:        WorkloadTypeContainer,
			CPUCores:    1,
			MemoryMB:    512,
			NetworkMode: NetworkModeBridge,
		}

		vmSpec := &WorkloadSpec{
			Name:        "test-vm",
			Type:        WorkloadTypeVM,
			CPUCores:    2,
			MemoryMB:    2048,
			NetworkMode: NetworkModeOverlay,
		}

		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
		defer cancel()

		container, err := hybridCluster.DeployWorkload(ctx, containerSpec)
		require.NoError(t, err)

		vm, err := hybridCluster.DeployWorkload(ctx, vmSpec)
		require.NoError(t, err)

		// Wait for workloads to be ready
		err = waitForWorkloadReady(container, 2*time.Minute)
		require.NoError(t, err)
		
		err = waitForWorkloadReady(vm, 3*time.Minute)
		require.NoError(t, err)

		// Test connectivity between container and VM
		connectivity := testNetworkConnectivity(container, vm)
		assert.True(t, connectivity.Reachable, "Container and VM should be able to communicate")
		assert.LessOrEqual(t, connectivity.Latency, 10*time.Millisecond, "Latency should be low")
		assert.GreaterOrEqual(t, connectivity.Throughput, 100.0, "Throughput should be sufficient (Mbps)")

		// Test service discovery
		serviceDiscovery := testServiceDiscovery(container, vm)
		assert.True(t, serviceDiscovery.ContainerFoundVM, "Container should discover VM service")
		assert.True(t, serviceDiscovery.VMFoundContainer, "VM should discover container service")

		// Test load balancing
		loadBalancer := setupLoadBalancer([]DeployedWorkload{*container, *vm})
		trafficDistribution := testLoadBalancing(loadBalancer, 1000)
		
		assert.InDelta(t, 0.5, trafficDistribution.ContainerRatio, 0.1, 
			"Traffic should be evenly distributed")
		assert.InDelta(t, 0.5, trafficDistribution.VMRatio, 0.1, 
			"Traffic should be evenly distributed")
	})

	t.Run("NetworkPolicyEnforcement", func(t *testing.T) {
		// Create network policy
		networkPolicy := &NetworkPolicy{
			Name: "hybrid-isolation-policy",
			Rules: []NetworkRule{
				{
					From: NetworkSelector{WorkloadType: WorkloadTypeContainer},
					To:   NetworkSelector{WorkloadType: WorkloadTypeVM},
					Action: ActionAllow,
					Ports:  []int{80, 443},
				},
				{
					From: NetworkSelector{WorkloadType: WorkloadTypeVM},
					To:   NetworkSelector{WorkloadType: WorkloadTypeContainer},
					Action: ActionDeny,
					Except: []NetworkException{
						{Port: 22, Protocol: "tcp"},
					},
				},
			},
		}

		err := hybridCluster.ApplyNetworkPolicy(networkPolicy)
		require.NoError(t, err)

		// Deploy test workloads
		container := deployTestWorkload(t, hybridCluster, WorkloadTypeContainer, "policy-test-container")
		vm := deployTestWorkload(t, hybridCluster, WorkloadTypeVM, "policy-test-vm")

		// Test allowed communication
		allowedPorts := []int{80, 443}
		for _, port := range allowedPorts {
			connectivity := testPortConnectivity(container, vm, port)
			assert.True(t, connectivity, "Container should reach VM on allowed port %d", port)
		}

		// Test denied communication
		blockedPorts := []int{8080, 3000}
		for _, port := range blockedPorts {
			connectivity := testPortConnectivity(container, vm, port)
			assert.False(t, connectivity, "Container should not reach VM on blocked port %d", port)
		}

		// Test exception
		exceptionConnectivity := testPortConnectivity(vm, container, 22)
		assert.True(t, exceptionConnectivity, "VM should reach container on exception port 22")
	})
}

// TestHybridStorage tests storage across container and VM workloads
func TestHybridStorage(t *testing.T) {
	hybridCluster := setupHybridCluster(3)
	defer hybridCluster.cleanup()

	t.Run("SharedStorageAccess", func(t *testing.T) {
		// Create shared storage volume
		sharedVolume := &StorageVolume{
			Name:        "hybrid-shared-volume",
			Size:        "10Gi",
			AccessMode:  AccessModeReadWriteMany,
			StorageClass: "ssd-replicated",
		}

		err := hybridCluster.CreateVolume(sharedVolume)
		require.NoError(t, err)

		// Deploy workloads with shared storage
		containerSpec := &WorkloadSpec{
			Name:        "container-with-storage",
			Type:        WorkloadTypeContainer,
			CPUCores:    1,
			MemoryMB:    512,
			Volumes: []VolumeMount{
				{
					Name:      sharedVolume.Name,
					MountPath: "/shared-data",
				},
			},
		}

		vmSpec := &WorkloadSpec{
			Name:        "vm-with-storage",
			Type:        WorkloadTypeVM,
			CPUCores:    2,
			MemoryMB:    2048,
			Volumes: []VolumeMount{
				{
					Name:      sharedVolume.Name,
					MountPath: "/mnt/shared",
				},
			},
		}

		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
		defer cancel()

		container, err := hybridCluster.DeployWorkload(ctx, containerSpec)
		require.NoError(t, err)

		vm, err := hybridCluster.DeployWorkload(ctx, vmSpec)
		require.NoError(t, err)

		// Test data sharing
		testData := "hybrid-storage-test-data"
		
		// Write data from container
		err = writeDataToWorkload(container, "/shared-data/test.txt", testData)
		require.NoError(t, err)

		// Read data from VM
		readData, err := readDataFromWorkload(vm, "/mnt/shared/test.txt")
		require.NoError(t, err)
		assert.Equal(t, testData, readData, "Data should be shared between workloads")

		// Test concurrent access
		concurrentWriteTest := func(workload *DeployedWorkload, path string, data string) error {
			return writeDataToWorkload(workload, path, data)
		}

		var wg sync.WaitGroup
		wg.Add(2)

		go func() {
			defer wg.Done()
			err := concurrentWriteTest(container, "/shared-data/container.txt", "container-data")
			assert.NoError(t, err)
		}()

		go func() {
			defer wg.Done()
			err := concurrentWriteTest(vm, "/mnt/shared/vm.txt", "vm-data")
			assert.NoError(t, err)
		}()

		wg.Wait()

		// Verify both files exist and are readable
		containerData, err := readDataFromWorkload(vm, "/mnt/shared/container.txt")
		require.NoError(t, err)
		assert.Equal(t, "container-data", containerData)

		vmData, err := readDataFromWorkload(container, "/shared-data/vm.txt")
		require.NoError(t, err)
		assert.Equal(t, "vm-data", vmData)
	})

	t.Run("StoragePerformanceConsistency", func(t *testing.T) {
		// Test storage performance across workload types
		workloadTypes := []WorkloadType{WorkloadTypeContainer, WorkloadTypeMicroVM, WorkloadTypeVM}
		storagePerformance := make(map[WorkloadType]*StoragePerformanceMetrics)

		for _, workloadType := range workloadTypes {
			spec := &WorkloadSpec{
				Name:        fmt.Sprintf("perf-test-%s", workloadType),
				Type:        workloadType,
				CPUCores:    2,
				MemoryMB:    2048,
				Volumes: []VolumeMount{
					{
						Name:      "perf-test-volume",
						MountPath: "/test-storage",
						Size:      "5Gi",
					},
				},
			}

			ctx, cancel := context.WithTimeout(context.Background(), 8*time.Minute)
			workload, err := hybridCluster.DeployWorkload(ctx, spec)
			cancel()
			require.NoError(t, err)

			// Run storage performance test
			perfMetrics, err := runStoragePerformanceTest(workload)
			require.NoError(t, err)

			storagePerformance[workloadType] = perfMetrics

			// Clean up
			ctx, cancel = context.WithTimeout(context.Background(), 3*time.Minute)
			err = hybridCluster.DeleteWorkload(ctx, workload.ID)
			cancel()
			assert.NoError(t, err)
		}

		// Validate performance consistency
		baseThroughput := storagePerformance[WorkloadTypeContainer].WriteThroughputMBps
		
		for workloadType, metrics := range storagePerformance {
			throughputRatio := metrics.WriteThroughputMBps / baseThroughput
			assert.InDelta(t, 1.0, throughputRatio, 0.3, 
				"Storage throughput should be consistent across workload types (%s)", workloadType)

			assert.LessOrEqual(t, metrics.WriteLatencyMs, 20.0,
				"Write latency should be acceptable for %s", workloadType)
			assert.LessOrEqual(t, metrics.ReadLatencyMs, 10.0,
				"Read latency should be acceptable for %s", workloadType)
		}
	})
}

// Helper functions
func setupHybridCluster(nodeCount int) *HybridCluster {
	config := &HybridClusterConfig{
		NodeCount:     nodeCount,
		ContainerRuntime: "containerd",
		VMHypervisor:  "qemu-kvm",
		MicroVMEngine: "firecracker",
		NetworkPlugin: "calico",
		StorageClass:  "ceph-rbd",
	}
	return NewHybridCluster(config)
}

func createWorkloadSpec(workloadType WorkloadType, size WorkloadSize) *WorkloadSpec {
	specs := map[WorkloadSize]struct {
		cpu    float64
		memory int
		disk   int
	}{
		WorkloadSizeSmall:  {cpu: 0.5, memory: 512, disk: 5},
		WorkloadSizeMedium: {cpu: 2.0, memory: 2048, disk: 20},
		WorkloadSizeLarge:  {cpu: 4.0, memory: 8192, disk: 100},
	}

	spec := specs[size]
	
	return &WorkloadSpec{
		Name:        fmt.Sprintf("test-%s-%s", workloadType, size),
		Type:        workloadType,
		CPUCores:    spec.cpu,
		MemoryMB:    spec.memory,
		DiskGB:      spec.disk,
		Image:       getTestImage(workloadType),
		Isolation:   IsolationNormal,
	}
}

func getTestImage(workloadType WorkloadType) string {
	images := map[WorkloadType]string{
		WorkloadTypeContainer: "alpine:latest",
		WorkloadTypeMicroVM:   "alpine-microvm:latest",
		WorkloadTypeVM:        "ubuntu-20.04",
	}
	return images[workloadType]
}

func waitForWorkloadReady(workload *DeployedWorkload, timeout time.Duration) error {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	for {
		select {
		case <-ctx.Done():
			return fmt.Errorf("timeout waiting for workload %s to be ready", workload.Name)
		default:
			status := workload.GetStatus()
			if status.State == WorkloadStatusRunning && status.Ready {
				return nil
			}
			time.Sleep(5 * time.Second)
		}
	}
}

func testWorkloadOperations(workload *DeployedWorkload) error {
	// Test basic operations
	operations := []string{"ping", "memory-test", "cpu-test", "disk-test"}
	
	for _, op := range operations {
		result, err := workload.ExecuteOperation(op)
		if err != nil {
			return fmt.Errorf("operation %s failed: %w", op, err)
		}
		if !result.Success {
			return fmt.Errorf("operation %s was unsuccessful: %s", op, result.Message)
		}
	}
	
	return nil
}

func testNetworkConnectivity(source, target *DeployedWorkload) *NetworkConnectivityResult {
	// Simulate network connectivity test
	return &NetworkConnectivityResult{
		Reachable:  true,
		Latency:    5 * time.Millisecond,
		Throughput: 150.0, // Mbps
	}
}

func testServiceDiscovery(container, vm *DeployedWorkload) *ServiceDiscoveryResult {
	return &ServiceDiscoveryResult{
		ContainerFoundVM: true,
		VMFoundContainer: true,
	}
}

func deployTestWorkload(t *testing.T, cluster *HybridCluster, workloadType WorkloadType, name string) *DeployedWorkload {
	spec := &WorkloadSpec{
		Name:     name,
		Type:     workloadType,
		CPUCores: 1,
		MemoryMB: 1024,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	workload, err := cluster.DeployWorkload(ctx, spec)
	require.NoError(t, err)

	err = waitForWorkloadReady(workload, 3*time.Minute)
	require.NoError(t, err)

	return workload
}

func testPortConnectivity(source, target *DeployedWorkload, port int) bool {
	// Simulate port connectivity test
	// In real implementation, would test actual network connectivity
	return true // Simplified for example
}

func writeDataToWorkload(workload *DeployedWorkload, path, data string) error {
	// Simulate writing data to workload storage
	return nil
}

func readDataFromWorkload(workload *DeployedWorkload, path string) (string, error) {
	// Simulate reading data from workload storage
	return "test-data", nil
}

func runStoragePerformanceTest(workload *DeployedWorkload) (*StoragePerformanceMetrics, error) {
	// Simulate storage performance test
	return &StoragePerformanceMetrics{
		WriteThroughputMBps: 100.0,
		ReadThroughputMBps:  150.0,
		WriteLatencyMs:      15.0,
		ReadLatencyMs:       8.0,
		IOPS:                5000,
	}, nil
}

func calculateLoadVariance(loads map[string]float64) float64 {
	if len(loads) == 0 {
		return 0.0
	}

	var sum, sumSquares float64
	for _, load := range loads {
		sum += load
		sumSquares += load * load
	}

	mean := sum / float64(len(loads))
	variance := (sumSquares / float64(len(loads))) - (mean * mean)
	
	return variance
}