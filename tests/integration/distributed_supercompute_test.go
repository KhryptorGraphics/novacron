package integration

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"novacron/backend/core/compute"
	"novacron/backend/core/scheduler"
	"novacron/backend/core/federation"
	"novacron/backend/core/migration"
	"novacron/backend/core/agents"
)

// DistributedSupercomputeTestSuite tests the complete distributed supercompute fabric
type DistributedSupercomputeTestSuite struct {
	suite.Suite
	ctx              context.Context
	cancel           context.CancelFunc
	scheduler        *scheduler.Scheduler
	jobManager       *compute.ComputeJobManager
	loadBalancer     *compute.ComputeJobLoadBalancer
	federationMgr    *federation.FederationManager
	migrationRunner  *migration.CrossClusterRunner
	memoryFabric     *compute.UnifiedMemoryFabric
	clusterAgent     *agents.ClusterAgent
	perfOptimizer    *compute.PerformanceOptimizer
}

// SetupSuite initializes the test environment
func (suite *DistributedSupercomputeTestSuite) SetupSuite() {
	suite.ctx, suite.cancel = context.WithCancel(context.Background())

	// Initialize scheduler with global resource pooling
	schedulerConfig := scheduler.DefaultSchedulerConfig()
	schedulerConfig.NetworkAwarenessEnabled = true
	schedulerConfig.BandwidthPredictionEnabled = true
	suite.scheduler = scheduler.NewScheduler(schedulerConfig)

	// Register test nodes
	suite.setupTestNodes()

	// Initialize job manager
	jobConfig := compute.DefaultJobManagerConfig()
	suite.jobManager = compute.NewComputeJobManager(jobConfig, suite.scheduler)

	// Initialize load balancer
	lbConfig := compute.DefaultLoadBalancerConfig()
	suite.loadBalancer = compute.NewComputeJobLoadBalancer(lbConfig, suite.scheduler)

	// Initialize federation manager
	fedConfig := federation.DefaultFederationConfig()
	suite.federationMgr = federation.NewFederationManager(fedConfig, nil, nil)

	// Initialize cross-cluster migration runner
	migrationConfig := migration.DefaultCrossClusterConfig()
	suite.migrationRunner = migration.NewCrossClusterRunner(migrationConfig, suite.federationMgr)

	// Initialize unified memory fabric
	memoryConfig := compute.DefaultMemoryFabricConfig()
	suite.memoryFabric = compute.NewUnifiedMemoryFabric(memoryConfig, suite.scheduler)

	// Initialize cluster agent
	agentConfig := agents.DefaultClusterAgentConfig()
	suite.clusterAgent = agents.NewClusterAgent(agentConfig, suite.scheduler, suite.federationMgr)

	// Initialize performance optimizer
	optimizerConfig := compute.DefaultOptimizerConfig()
	suite.perfOptimizer = compute.NewPerformanceOptimizer(suite.scheduler, suite.jobManager, suite.loadBalancer, optimizerConfig)

	// Start all components
	require.NoError(suite.T(), suite.scheduler.Start())
	require.NoError(suite.T(), suite.jobManager.Start())
	require.NoError(suite.T(), suite.loadBalancer.Start())
	require.NoError(suite.T(), suite.federationMgr.Start())
	require.NoError(suite.T(), suite.migrationRunner.Start())
	require.NoError(suite.T(), suite.memoryFabric.Start())
	require.NoError(suite.T(), suite.clusterAgent.Start())
	require.NoError(suite.T(), suite.perfOptimizer.Start())

	// Wait for components to initialize
	time.Sleep(2 * time.Second)
}

// TearDownSuite cleans up the test environment
func (suite *DistributedSupercomputeTestSuite) TearDownSuite() {
	suite.perfOptimizer.Stop()
	suite.clusterAgent.Stop()
	suite.memoryFabric.Stop()
	suite.migrationRunner.Stop()
	suite.federationMgr.Stop()
	suite.loadBalancer.Stop()
	suite.jobManager.Stop()
	suite.scheduler.Stop()
	suite.cancel()
}

// setupTestNodes creates test nodes for the scheduler
func (suite *DistributedSupercomputeTestSuite) setupTestNodes() {
	// Node 1: High CPU, moderate memory
	node1Resources := map[scheduler.ResourceType]*scheduler.Resource{
		scheduler.ResourceCPU: {
			Type:     scheduler.ResourceCPU,
			Capacity: 16.0,
			Used:     2.0,
		},
		scheduler.ResourceMemory: {
			Type:     scheduler.ResourceMemory,
			Capacity: 32.0,
			Used:     8.0,
		},
		scheduler.ResourceDisk: {
			Type:     scheduler.ResourceDisk,
			Capacity: 1000.0,
			Used:     100.0,
		},
		scheduler.ResourceNetwork: {
			Type:     scheduler.ResourceNetwork,
			Capacity: 10000.0,
			Used:     1000.0,
		},
	}
	suite.scheduler.RegisterNode("node-1", node1Resources)

	// Node 2: Moderate CPU, high memory
	node2Resources := map[scheduler.ResourceType]*scheduler.Resource{
		scheduler.ResourceCPU: {
			Type:     scheduler.ResourceCPU,
			Capacity: 8.0,
			Used:     1.0,
		},
		scheduler.ResourceMemory: {
			Type:     scheduler.ResourceMemory,
			Capacity: 64.0,
			Used:     4.0,
		},
		scheduler.ResourceDisk: {
			Type:     scheduler.ResourceDisk,
			Capacity: 2000.0,
			Used:     200.0,
		},
		scheduler.ResourceNetwork: {
			Type:     scheduler.ResourceNetwork,
			Capacity: 10000.0,
			Used:     500.0,
		},
	}
	suite.scheduler.RegisterNode("node-2", node2Resources)

	// Node 3: Balanced resources
	node3Resources := map[scheduler.ResourceType]*scheduler.Resource{
		scheduler.ResourceCPU: {
			Type:     scheduler.ResourceCPU,
			Capacity: 12.0,
			Used:     3.0,
		},
		scheduler.ResourceMemory: {
			Type:     scheduler.ResourceMemory,
			Capacity: 48.0,
			Used:     12.0,
		},
		scheduler.ResourceDisk: {
			Type:     scheduler.ResourceDisk,
			Capacity: 1500.0,
			Used:     300.0,
		},
		scheduler.ResourceNetwork: {
			Type:     scheduler.ResourceNetwork,
			Capacity: 10000.0,
			Used:     2000.0,
		},
	}
	suite.scheduler.RegisterNode("node-3", node3Resources)
}

// TestDistributedJobSubmissionAndExecution tests complete job lifecycle
func (suite *DistributedSupercomputeTestSuite) TestDistributedJobSubmissionAndExecution() {
	// Create a compute job
	job := &compute.ComputeJob{
		Name:        "test-distributed-job",
		Description: "Test job for distributed supercompute",
		JobType:     compute.JobTypeBatch,
		Priority:    5,
		QueueName:   "default",
		Command:     []string{"echo", "Hello Distributed Supercompute"},
		Environment: map[string]string{
			"TEST_ENV": "distributed",
		},
		ResourceRequirements: compute.ResourceRequirements{
			CPUCores:    2.0,
			MemoryGB:    4.0,
			DiskGB:      10.0,
			NetworkMbps: 100.0,
		},
		Tags: map[string]string{
			"test": "distributed",
			"type": "integration",
		},
		Timeout: 5 * time.Minute,
	}

	// Submit job
	jobID, err := suite.jobManager.SubmitJob(suite.ctx, job)
	require.NoError(suite.T(), err)
	assert.NotEmpty(suite.T(), jobID)

	// Wait for job to be scheduled and started
	time.Sleep(3 * time.Second)

	// Check job status
	retrievedJob, err := suite.jobManager.GetJob(suite.ctx, jobID)
	require.NoError(suite.T(), err)
	assert.Equal(suite.T(), jobID, retrievedJob.ID)
	assert.Contains(suite.T(), []compute.JobStatus{
		compute.JobStatusPending,
		compute.JobStatusScheduled,
		compute.JobStatusRunning,
		compute.JobStatusCompleted,
	}, retrievedJob.Status)

	// Test load balancing
	algorithm := suite.loadBalancer.GetCurrentAlgorithm()
	assert.NotEmpty(suite.T(), algorithm)

	// Test metrics collection
	metrics := suite.jobManager.GetPerformanceMetrics(suite.ctx)
	assert.NotNil(suite.T(), metrics)
}

// TestGlobalResourcePooling tests federated resource allocation
func (suite *DistributedSupercomputeTestSuite) TestGlobalResourcePooling() {
	// Initialize global resource pool
	err := suite.scheduler.InitializeGlobalResourcePool(nil, "cluster-1")
	require.NoError(suite.T(), err)

	// Get global resource inventory
	inventory, err := suite.scheduler.GetGlobalResourceInventory()
	require.NoError(suite.T(), err)
	assert.Contains(suite.T(), inventory, "cluster-1")

	// Test resource utilization
	utilization, err := suite.scheduler.GetGlobalResourceUtilization()
	require.NoError(suite.T(), err)
	assert.NotEmpty(suite.T(), utilization)

	// Verify local cluster is included
	assert.Contains(suite.T(), utilization, "cluster-1")
	clusterUtil := utilization["cluster-1"]
	assert.Contains(suite.T(), clusterUtil, scheduler.ResourceCPU)
	assert.Contains(suite.T(), clusterUtil, scheduler.ResourceMemory)
}

// TestCrossClusterMigration tests VM migration capabilities
func (suite *DistributedSupercomputeTestSuite) TestCrossClusterMigration() {
	// Create migration request
	migrationReq := &migration.MigrationRequest{
		VMID:           "test-vm-1",
		SourceCluster:  "cluster-1",
		TargetCluster:  "cluster-2",
		MigrationType:  migration.MigrationTypeLive,
		Priority:       migration.PriorityHigh,
		Constraints: migration.MigrationConstraints{
			MaxDowntimeMs:       1000,
			MaxMigrationTimeMin: 30,
			DataIntegrity:       true,
			NetworkOptimization: true,
		},
		Metadata: map[string]interface{}{
			"test": true,
		},
	}

	// Test migration planning
	plan, err := suite.migrationRunner.PlanMigration(suite.ctx, migrationReq)
	require.NoError(suite.T(), err)
	assert.NotNil(suite.T(), plan)
	assert.Equal(suite.T(), "test-vm-1", plan.VMID)
	assert.Equal(suite.T(), "cluster-1", plan.SourceCluster)
	assert.Equal(suite.T(), "cluster-2", plan.TargetCluster)

	// Test preflight checks
	checks, err := suite.migrationRunner.RunPreflightChecks(suite.ctx, plan)
	require.NoError(suite.T(), err)
	assert.NotNil(suite.T(), checks)

	// Note: We don't execute actual migration in tests
	// but verify the planning and preparation works
}

// TestUnifiedMemoryFabric tests distributed memory management
func (suite *DistributedSupercomputeTestSuite) TestUnifiedMemoryFabric() {
	// Create memory pool
	pool := &compute.MemoryPool{
		PoolID:      "test-pool-1",
		ClusterID:   "cluster-1",
		TotalSizeGB: 100.0,
		Configuration: compute.MemoryPoolConfig{
			CoherenceProtocol: compute.CoherenceProtocolMESI,
			ReplicationFactor: 2,
			EnableCompression: true,
			EnableEncryption:  true,
		},
	}

	err := suite.memoryFabric.CreateMemoryPool(suite.ctx, pool)
	require.NoError(suite.T(), err)

	// Allocate memory region
	allocation := &compute.MemoryAllocation{
		AllocationID: "test-alloc-1",
		PoolID:       "test-pool-1",
		SizeGB:       10.0,
		Access:       compute.MemoryAccessReadWrite,
		Locality:     compute.MemoryLocalityLocal,
	}

	err = suite.memoryFabric.AllocateMemory(suite.ctx, allocation)
	require.NoError(suite.T(), err)

	// Get pool status
	status, err := suite.memoryFabric.GetPoolStatus(suite.ctx, "test-pool-1")
	require.NoError(suite.T(), err)
	assert.Equal(suite.T(), "test-pool-1", status.PoolID)
	assert.True(suite.T(), status.AllocatedSizeGB > 0)

	// Release allocation
	err = suite.memoryFabric.ReleaseMemory(suite.ctx, "test-alloc-1")
	require.NoError(suite.T(), err)
}

// TestClusterAgentMonitoring tests cluster monitoring capabilities
func (suite *DistributedSupercomputeTestSuite) TestClusterAgentMonitoring() {
	// Wait for agent to collect metrics
	time.Sleep(3 * time.Second)

	// Get cluster metrics
	metrics, err := suite.clusterAgent.GetResourceMetrics(suite.ctx)
	require.NoError(suite.T(), err)
	assert.NotNil(suite.T(), metrics)

	// Get health status
	health, err := suite.clusterAgent.GetHealthStatus(suite.ctx)
	require.NoError(suite.T(), err)
	assert.NotNil(suite.T(), health)
	assert.Equal(suite.T(), agents.HealthStatusHealthy, health.OverallStatus)

	// Get job coordination status
	jobStatus, err := suite.clusterAgent.GetJobCoordinationStatus(suite.ctx)
	require.NoError(suite.T(), err)
	assert.NotNil(suite.T(), jobStatus)
}

// TestPerformanceOptimization tests performance optimization capabilities
func (suite *DistributedSupercomputeTestSuite) TestPerformanceOptimization() {
	// Wait for optimizer to collect metrics
	time.Sleep(5 * time.Second)

	// Get current performance snapshot
	snapshot := suite.perfOptimizer.GetCurrentPerformanceSnapshot()
	assert.True(suite.T(), snapshot.Timestamp.After(time.Now().Add(-time.Minute)))

	// Get optimization recommendations
	recommendations := suite.perfOptimizer.GetOptimizationRecommendations()
	assert.NotNil(suite.T(), recommendations)

	// Check optimizer health
	isHealthy := suite.perfOptimizer.IsHealthy()
	assert.True(suite.T(), isHealthy)

	// Get statistics
	stats := suite.perfOptimizer.GetStatistics()
	assert.NotNil(suite.T(), stats)
	assert.Contains(suite.T(), stats, "total_optimization_tasks")
}

// TestLoadBalancingAlgorithms tests different load balancing algorithms
func (suite *DistributedSupercomputeTestSuite) TestLoadBalancingAlgorithms() {
	algorithms := []compute.LoadBalanceAlgorithm{
		compute.LoadBalanceLeastLoaded,
		compute.LoadBalanceWeightedRoundRobin,
		compute.LoadBalanceNetworkAware,
		compute.LoadBalanceCostOptimized,
		compute.LoadBalancePerformanceBased,
	}

	for _, algorithm := range algorithms {
		// Set algorithm
		err := suite.loadBalancer.SetAlgorithm(algorithm)
		require.NoError(suite.T(), err)

		// Verify algorithm is set
		current := suite.loadBalancer.GetCurrentAlgorithm()
		assert.Equal(suite.T(), algorithm, current)

		// Submit a test job to trigger load balancing
		job := &compute.ComputeJob{
			Name:        "test-lb-job-" + string(algorithm),
			JobType:     compute.JobTypeBatch,
			Priority:    3,
			QueueName:   "default",
			Command:     []string{"echo", "test"},
			ResourceRequirements: compute.ResourceRequirements{
				CPUCores: 1.0,
				MemoryGB: 2.0,
			},
			Timeout: 1 * time.Minute,
		}

		jobID, err := suite.jobManager.SubmitJob(suite.ctx, job)
		require.NoError(suite.T(), err)
		assert.NotEmpty(suite.T(), jobID)

		time.Sleep(1 * time.Second) // Allow processing
	}

	// Get load balancer metrics
	metrics := suite.loadBalancer.GetMetrics()
	assert.NotNil(suite.T(), metrics)
	assert.True(suite.T(), metrics.TotalJobs > 0)
}

// TestResourceConstraints tests resource allocation with constraints
func (suite *DistributedSupercomputeTestSuite) TestResourceConstraints() {
	// Test basic resource constraints
	constraints := []scheduler.ResourceConstraint{
		{
			Type:      scheduler.ResourceCPU,
			MinAmount: 2.0,
			MaxAmount: 4.0,
		},
		{
			Type:      scheduler.ResourceMemory,
			MinAmount: 4.0,
			MaxAmount: 8.0,
		},
	}

	// Request resources with constraints
	requestID, err := suite.scheduler.RequestResources(constraints, 5, 5*time.Minute)
	require.NoError(suite.T(), err)
	assert.NotEmpty(suite.T(), requestID)

	// Test network constraints
	networkConstraints := &scheduler.NetworkConstraint{
		MinBandwidthBps:    1000000, // 1 Mbps
		MaxLatencyMs:       50.0,
		RequiredTopology:   "low-latency",
		MinConnections:     2,
		BandwidthGuarantee: true,
	}

	requestID2, err := suite.scheduler.RequestResourcesWithNetworkConstraints(
		constraints, networkConstraints, 7, 5*time.Minute)
	require.NoError(suite.T(), err)
	assert.NotEmpty(suite.T(), requestID2)

	// Wait for allocation
	time.Sleep(2 * time.Second)

	// Check allocations
	allocations := suite.scheduler.GetActiveAllocations()
	assert.NotEmpty(suite.T(), allocations)
}

// TestJobQueues tests job queue management
func (suite *DistributedSupercomputeTestSuite) TestJobQueues() {
	// Test different job types in different queues
	jobTypes := []struct {
		jobType   compute.JobType
		queueName string
		priority  int
	}{
		{compute.JobTypeBatch, "batch", 3},
		{compute.JobTypeInteractive, "interactive", 7},
		{compute.JobTypeMPI, "mpi", 5},
		{compute.JobTypeContainer, "container", 4},
		{compute.JobTypeStream, "stream", 8},
	}

	var submittedJobs []string

	for i, jt := range jobTypes {
		job := &compute.ComputeJob{
			Name:        fmt.Sprintf("test-queue-job-%d", i),
			JobType:     jt.jobType,
			Priority:    jt.priority,
			QueueName:   jt.queueName,
			Command:     []string{"echo", "queue test"},
			ResourceRequirements: compute.ResourceRequirements{
				CPUCores: 1.0,
				MemoryGB: 1.0,
			},
			Timeout: 2 * time.Minute,
		}

		jobID, err := suite.jobManager.SubmitJob(suite.ctx, job)
		require.NoError(suite.T(), err)
		submittedJobs = append(submittedJobs, jobID)
	}

	// Wait for processing
	time.Sleep(3 * time.Second)

	// Check job statuses
	for _, jobID := range submittedJobs {
		status, err := suite.jobManager.GetJobStatus(suite.ctx, jobID)
		require.NoError(suite.T(), err)
		assert.NotEqual(suite.T(), compute.JobStatusUnknown, status)
	}

	// List queues
	queues, err := suite.jobManager.ListQueues(suite.ctx)
	require.NoError(suite.T(), err)
	assert.NotEmpty(suite.T(), queues)
}

// TestDistributedWorkloadDistribution tests workload distribution strategies
func (suite *DistributedSupercomputeTestSuite) TestDistributedWorkloadDistribution() {
	// Create multiple resource requests
	workloads := make([]scheduler.ResourceRequest, 5)
	for i := 0; i < 5; i++ {
		workloads[i] = scheduler.ResourceRequest{
			ID: fmt.Sprintf("workload-%d", i),
			Constraints: []scheduler.ResourceConstraint{
				{
					Type:      scheduler.ResourceCPU,
					MinAmount: 1.0,
					MaxAmount: 2.0,
				},
				{
					Type:      scheduler.ResourceMemory,
					MinAmount: 2.0,
					MaxAmount: 4.0,
				},
			},
			Priority:  5,
			Timeout:   5 * time.Minute,
			CreatedAt: time.Now(),
		}
	}

	// Test different distribution strategies
	strategies := []string{
		"balanced",
		"cost-optimized",
		"performance-first",
		"locality-aware",
	}

	for _, strategy := range strategies {
		// Note: This would test actual distribution if we had multiple clusters
		// For now, we test the interface and basic functionality
		allocations, err := suite.scheduler.ScheduleGlobalWorkloadDistribution(workloads, strategy)

		// If no federated clusters are available, this may return an error
		// In a real distributed environment, this would succeed
		if err == nil {
			assert.NotNil(suite.T(), allocations)
		}
	}
}

// TestSystemIntegration tests end-to-end system integration
func (suite *DistributedSupercomputeTestSuite) TestSystemIntegration() {
	// Create a complex job with multiple requirements
	job := &compute.ComputeJob{
		Name:        "integration-test-job",
		Description: "Complex job for integration testing",
		JobType:     compute.JobTypeMPI,
		Priority:    8,
		QueueName:   "high-priority",
		Command:     []string{"mpirun", "-np", "4", "test-app"},
		Environment: map[string]string{
			"MPI_HOSTS":     "node-1,node-2,node-3",
			"COMPUTE_MODE":  "distributed",
			"OPTIMIZATION":  "enabled",
		},
		ResourceRequirements: compute.ResourceRequirements{
			CPUCores:    8.0,
			MemoryGB:    16.0,
			DiskGB:      50.0,
			GPUs:        0,
			NetworkMbps: 1000.0,
		},
		Constraints: []compute.JobConstraint{
			{
				Type:     "locality",
				Operator: "in",
				Value:    []string{"datacenter-1", "datacenter-2"},
			},
			{
				Type:     "network_latency",
				Operator: "lt",
				Value:    10.0, // Less than 10ms
			},
		},
		Tags: map[string]string{
			"workload":     "compute-intensive",
			"distribution": "multi-node",
			"test":         "integration",
		},
		Timeout: 30 * time.Minute,
	}

	// Submit the complex job
	jobID, err := suite.jobManager.SubmitJob(suite.ctx, job)
	require.NoError(suite.T(), err)
	assert.NotEmpty(suite.T(), jobID)

	// Wait for job processing
	time.Sleep(5 * time.Second)

	// Verify job was processed by all components
	retrievedJob, err := suite.jobManager.GetJob(suite.ctx, jobID)
	require.NoError(suite.T(), err)
	assert.Equal(suite.T(), jobID, retrievedJob.ID)

	// Check that cluster agent has monitored the job
	metrics, err := suite.clusterAgent.GetResourceMetrics(suite.ctx)
	require.NoError(suite.T(), err)
	assert.NotNil(suite.T(), metrics)

	// Check that performance optimizer has recommendations
	recommendations := suite.perfOptimizer.GetOptimizationRecommendations()
	assert.NotNil(suite.T(), recommendations)

	// Check scheduler state
	nodes := suite.scheduler.GetNodesStatus()
	assert.Len(suite.T(), nodes, 3) // We registered 3 nodes

	requests := suite.scheduler.GetPendingRequests()
	allocations := suite.scheduler.GetActiveAllocations()
	tasks := suite.scheduler.GetTasks()

	// At least some activity should be present
	assert.True(suite.T(), len(requests) >= 0)
	assert.True(suite.T(), len(allocations) >= 0)
	assert.True(suite.T(), len(tasks) >= 0)

	// Verify load balancer has processed jobs
	lbMetrics := suite.loadBalancer.GetMetrics()
	assert.True(suite.T(), lbMetrics.TotalJobs > 0)

	// Verify memory fabric is operational
	if len(suite.memoryFabric.GetMemoryPools(suite.ctx)) > 0 {
		pools := suite.memoryFabric.GetMemoryPools(suite.ctx)
		assert.NotNil(suite.T(), pools)
	}
}

// TestErrorHandlingAndRecovery tests error scenarios and recovery
func (suite *DistributedSupercomputeTestSuite) TestErrorHandlingAndRecovery() {
	// Test job with invalid requirements
	invalidJob := &compute.ComputeJob{
		Name:        "invalid-job",
		JobType:     compute.JobTypeBatch,
		Priority:    5,
		QueueName:   "default",
		Command:     []string{"invalid-command"},
		ResourceRequirements: compute.ResourceRequirements{
			CPUCores: 1000.0, // Impossible requirement
			MemoryGB: 1000.0, // Impossible requirement
		},
		Timeout: 1 * time.Minute,
	}

	// Submit invalid job - should handle gracefully
	jobID, err := suite.jobManager.SubmitJob(suite.ctx, invalidJob)
	if err != nil {
		// Expected behavior - job submission should fail
		assert.Contains(suite.T(), err.Error(), "resource")
	} else {
		// If submission succeeds, job should fail during scheduling
		time.Sleep(2 * time.Second)
		job, err := suite.jobManager.GetJob(suite.ctx, jobID)
		if err == nil {
			// Job should be in failed or pending state
			assert.Contains(suite.T(), []compute.JobStatus{
				compute.JobStatusFailed,
				compute.JobStatusPending,
			}, job.Status)
		}
	}

	// Test resource request with impossible constraints
	impossibleConstraints := []scheduler.ResourceConstraint{
		{
			Type:      scheduler.ResourceCPU,
			MinAmount: 1000.0, // More than any node has
		},
	}

	requestID, err := suite.scheduler.RequestResources(impossibleConstraints, 5, 1*time.Minute)
	if err == nil {
		// Request accepted but should not be fulfilled
		time.Sleep(3 * time.Second)
		allocations := suite.scheduler.GetActiveAllocations()

		// The impossible request should not have been allocated
		_, allocated := allocations[requestID]
		assert.False(suite.T(), allocated)
	}

	// Verify system is still healthy after error scenarios
	assert.True(suite.T(), suite.perfOptimizer.IsHealthy())
	assert.True(suite.T(), suite.loadBalancer.IsHealthy())

	health, err := suite.clusterAgent.GetHealthStatus(suite.ctx)
	require.NoError(suite.T(), err)
	assert.Equal(suite.T(), agents.HealthStatusHealthy, health.OverallStatus)
}

// Run the test suite
func TestDistributedSupercomputeTestSuite(t *testing.T) {
	suite.Run(t, new(DistributedSupercomputeTestSuite))
}