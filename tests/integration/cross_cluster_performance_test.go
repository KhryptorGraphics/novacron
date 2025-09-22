package integration

import (
	"context"
	"fmt"
	"math"
	"sync"
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

// CrossClusterPerformanceTestSuite tests performance of distributed supercompute across clusters
type CrossClusterPerformanceTestSuite struct {
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
	benchmarkResults map[string]*BenchmarkResult
	resultsMutex     sync.RWMutex
}

// BenchmarkResult stores the results of a performance benchmark
type BenchmarkResult struct {
	TestName            string                 `json:"test_name"`
	StartTime           time.Time              `json:"start_time"`
	Duration            time.Duration          `json:"duration"`
	Operations          int                    `json:"operations"`
	OperationsPerSecond float64                `json:"operations_per_second"`
	AverageLatencyMs    float64                `json:"average_latency_ms"`
	P95LatencyMs        float64                `json:"p95_latency_ms"`
	P99LatencyMs        float64                `json:"p99_latency_ms"`
	ThroughputMbps      float64                `json:"throughput_mbps"`
	CPUUtilization      float64                `json:"cpu_utilization"`
	MemoryUtilization   float64                `json:"memory_utilization"`
	NetworkUtilization  float64                `json:"network_utilization"`
	ErrorCount          int                    `json:"error_count"`
	ErrorRate           float64                `json:"error_rate"`
	Metadata            map[string]interface{} `json:"metadata"`
}

// BenchmarkMetrics tracks metrics during benchmark execution
type BenchmarkMetrics struct {
	latencies           []time.Duration
	throughputSamples   []float64
	cpuSamples         []float64
	memorySamples      []float64
	networkSamples     []float64
	errors             int
	mutex              sync.RWMutex
}

// SetupSuite initializes the performance test environment
func (suite *CrossClusterPerformanceTestSuite) SetupSuite() {
	suite.ctx, suite.cancel = context.WithCancel(context.Background())
	suite.benchmarkResults = make(map[string]*BenchmarkResult)

	// Initialize components with performance-optimized configurations
	suite.setupPerformanceOptimizedEnvironment()

	// Start all components
	require.NoError(suite.T(), suite.scheduler.Start())
	require.NoError(suite.T(), suite.jobManager.Start())
	require.NoError(suite.T(), suite.loadBalancer.Start())
	require.NoError(suite.T(), suite.federationMgr.Start())
	require.NoError(suite.T(), suite.migrationRunner.Start())
	require.NoError(suite.T(), suite.memoryFabric.Start())
	require.NoError(suite.T(), suite.clusterAgent.Start())
	require.NoError(suite.T(), suite.perfOptimizer.Start())

	// Wait for components to fully initialize
	time.Sleep(5 * time.Second)
}

// TearDownSuite cleans up the performance test environment
func (suite *CrossClusterPerformanceTestSuite) TearDownSuite() {
	// Print benchmark results summary
	suite.printBenchmarkSummary()

	// Stop all components
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

// setupPerformanceOptimizedEnvironment creates a high-performance test environment
func (suite *CrossClusterPerformanceTestSuite) setupPerformanceOptimizedEnvironment() {
	// High-performance scheduler configuration
	schedulerConfig := scheduler.DefaultSchedulerConfig()
	schedulerConfig.AllocationInterval = 100 * time.Millisecond // Fast allocation
	schedulerConfig.NetworkAwarenessEnabled = true
	schedulerConfig.BandwidthPredictionEnabled = true
	schedulerConfig.MaxNetworkUtilization = 95.0 // Allow high utilization
	suite.scheduler = scheduler.NewScheduler(schedulerConfig)

	// Create multiple high-capacity nodes
	suite.setupHighCapacityNodes()

	// High-throughput job manager
	jobConfig := compute.DefaultJobManagerConfig()
	jobConfig.MaxConcurrentJobs = 1000
	jobConfig.JobProcessingInterval = 50 * time.Millisecond
	jobConfig.EnableBatchProcessing = true
	suite.jobManager = compute.NewComputeJobManager(jobConfig, suite.scheduler)

	// Performance-optimized load balancer
	lbConfig := compute.DefaultLoadBalancerConfig()
	lbConfig.Algorithm = compute.LoadBalanceMLPredictive // Use ML-based balancing
	lbConfig.EnableHealthChecks = true
	lbConfig.HealthCheckInterval = 1 * time.Second
	suite.loadBalancer = compute.NewComputeJobLoadBalancer(lbConfig, suite.scheduler)

	// High-performance federation manager
	fedConfig := federation.DefaultFederationConfig()
	fedConfig.HeartbeatInterval = 1 * time.Second
	fedConfig.EnableCompression = true
	fedConfig.MaxConcurrentOperations = 500
	suite.federationMgr = federation.NewFederationManager(fedConfig, nil, nil)

	// Optimized migration runner
	migrationConfig := migration.DefaultCrossClusterConfig()
	migrationConfig.EnableWANOptimization = true
	migrationConfig.MaxConcurrentMigrations = 10
	migrationConfig.BandwidthLimitMbps = 10000 // 10 Gbps
	suite.migrationRunner = migration.NewCrossClusterRunner(migrationConfig, suite.federationMgr)

	// High-performance memory fabric
	memoryConfig := compute.DefaultMemoryFabricConfig()
	memoryConfig.CoherenceProtocol = compute.CoherenceProtocolMOESI // Most advanced protocol
	memoryConfig.EnableCompression = true
	memoryConfig.EnablePrefetching = true
	memoryConfig.MaxBandwidthGbps = 100 // 100 Gbps memory bandwidth
	suite.memoryFabric = compute.NewUnifiedMemoryFabric(memoryConfig, suite.scheduler)

	// High-frequency cluster agent
	agentConfig := agents.DefaultClusterAgentConfig()
	agentConfig.MetricsCollectionInterval = 100 * time.Millisecond
	agentConfig.HealthCheckInterval = 1 * time.Second
	agentConfig.EnableAdvancedMetrics = true
	suite.clusterAgent = agents.NewClusterAgent(agentConfig, suite.scheduler, suite.federationMgr)

	// Real-time performance optimizer
	optimizerConfig := compute.DefaultOptimizerConfig()
	optimizerConfig.OptimizationInterval = 30 * time.Second
	optimizerConfig.MetricsCollectionInterval = 1 * time.Second
	optimizerConfig.EnableAIOptimization = true
	optimizerConfig.EnableAutoScaling = true
	optimizerConfig.EnablePredictiveOptimization = true
	suite.perfOptimizer = compute.NewPerformanceOptimizer(suite.scheduler, suite.jobManager, suite.loadBalancer, optimizerConfig)
}

// setupHighCapacityNodes creates high-performance test nodes
func (suite *CrossClusterPerformanceTestSuite) setupHighCapacityNodes() {
	// Create 10 high-capacity nodes for performance testing
	for i := 0; i < 10; i++ {
		nodeID := fmt.Sprintf("perf-node-%d", i+1)
		resources := map[scheduler.ResourceType]*scheduler.Resource{
			scheduler.ResourceCPU: {
				Type:     scheduler.ResourceCPU,
				Capacity: 64.0, // 64 cores
				Used:     float64(i) * 2.0, // Varying initial load
			},
			scheduler.ResourceMemory: {
				Type:     scheduler.ResourceMemory,
				Capacity: 256.0, // 256 GB
				Used:     float64(i) * 8.0, // Varying initial load
			},
			scheduler.ResourceDisk: {
				Type:     scheduler.ResourceDisk,
				Capacity: 10000.0, // 10 TB
				Used:     float64(i) * 100.0,
			},
			scheduler.ResourceNetwork: {
				Type:     scheduler.ResourceNetwork,
				Capacity: 100000.0, // 100 Gbps
				Used:     float64(i) * 1000.0,
			},
		}
		suite.scheduler.RegisterNode(nodeID, resources)
	}
}

// BenchmarkJobSubmissionThroughput tests job submission performance
func (suite *CrossClusterPerformanceTestSuite) TestBenchmarkJobSubmissionThroughput() {
	testName := "job_submission_throughput"
	metrics := &BenchmarkMetrics{
		latencies:         make([]time.Duration, 0),
		throughputSamples: make([]float64, 0),
		cpuSamples:       make([]float64, 0),
		memorySamples:    make([]float64, 0),
		networkSamples:   make([]float64, 0),
	}

	startTime := time.Now()
	numJobs := 1000
	concurrency := 50

	// Create job templates
	jobTemplate := &compute.ComputeJob{
		Name:        "benchmark-job",
		Description: "Performance benchmark job",
		JobType:     compute.JobTypeBatch,
		Priority:    5,
		QueueName:   "benchmark",
		Command:     []string{"echo", "benchmark"},
		Environment: map[string]string{
			"BENCHMARK": "true",
		},
		ResourceRequirements: compute.ResourceRequirements{
			CPUCores:    2.0,
			MemoryGB:    4.0,
			DiskGB:      10.0,
			NetworkMbps: 100.0,
		},
		Tags: map[string]string{
			"benchmark": "throughput",
		},
		Timeout: 5 * time.Minute,
	}

	// Channel for job submission
	jobChan := make(chan int, numJobs)
	for i := 0; i < numJobs; i++ {
		jobChan <- i
	}
	close(jobChan)

	// Worker goroutines for concurrent job submission
	var wg sync.WaitGroup
	wg.Add(concurrency)

	for w := 0; w < concurrency; w++ {
		go func(workerID int) {
			defer wg.Done()

			for jobIndex := range jobChan {
				jobStartTime := time.Now()

				// Create job with unique name
				job := *jobTemplate
				job.Name = fmt.Sprintf("benchmark-job-%d", jobIndex)

				jobID, err := suite.jobManager.SubmitJob(suite.ctx, &job)
				jobLatency := time.Since(jobStartTime)

				metrics.mutex.Lock()
				if err != nil {
					metrics.errors++
				} else {
					metrics.latencies = append(metrics.latencies, jobLatency)
				}
				metrics.mutex.Unlock()

				// Verify job was submitted
				assert.NotEmpty(suite.T(), jobID)
			}
		}(w)
	}

	// Monitor system metrics during benchmark
	go suite.monitorSystemMetrics(metrics, testName)

	// Wait for all jobs to be submitted
	wg.Wait()
	duration := time.Since(startTime)

	// Wait a bit more for job processing
	time.Sleep(5 * time.Second)

	// Calculate results
	result := suite.calculateBenchmarkResult(testName, startTime, duration, numJobs, metrics)
	suite.storeBenchmarkResult(testName, result)

	// Assertions for performance benchmarks
	assert.Greater(suite.T(), result.OperationsPerSecond, 100.0, "Should handle at least 100 job submissions per second")
	assert.Less(suite.T(), result.AverageLatencyMs, 100.0, "Average job submission latency should be under 100ms")
	assert.Less(suite.T(), result.ErrorRate, 0.01, "Error rate should be less than 1%")

	suite.T().Logf("Job Submission Benchmark Results:")
	suite.T().Logf("  Operations/sec: %.2f", result.OperationsPerSecond)
	suite.T().Logf("  Average Latency: %.2f ms", result.AverageLatencyMs)
	suite.T().Logf("  P95 Latency: %.2f ms", result.P95LatencyMs)
	suite.T().Logf("  Error Rate: %.2f%%", result.ErrorRate*100)
}

// BenchmarkLoadBalancingPerformance tests load balancer performance with different algorithms
func (suite *CrossClusterPerformanceTestSuite) TestBenchmarkLoadBalancingPerformance() {
	algorithms := []compute.LoadBalanceAlgorithm{
		compute.LoadBalanceLeastLoaded,
		compute.LoadBalanceWeightedRoundRobin,
		compute.LoadBalanceNetworkAware,
		compute.LoadBalanceCostOptimized,
		compute.LoadBalancePerformanceBased,
		compute.LoadBalanceMLPredictive,
	}

	for _, algorithm := range algorithms {
		testName := fmt.Sprintf("load_balancing_%s", algorithm)
		suite.benchmarkLoadBalancingAlgorithm(algorithm, testName)
	}

	// Verify that all algorithms performed reasonably
	suite.resultsMutex.RLock()
	for _, result := range suite.benchmarkResults {
		if result.TestName != "job_submission_throughput" {
			assert.Greater(suite.T(), result.OperationsPerSecond, 50.0,
				fmt.Sprintf("Algorithm %s should handle at least 50 ops/sec", result.TestName))
		}
	}
	suite.resultsMutex.RUnlock()
}

// benchmarkLoadBalancingAlgorithm benchmarks a specific load balancing algorithm
func (suite *CrossClusterPerformanceTestSuite) benchmarkLoadBalancingAlgorithm(algorithm compute.LoadBalanceAlgorithm, testName string) {
	metrics := &BenchmarkMetrics{
		latencies:         make([]time.Duration, 0),
		throughputSamples: make([]float64, 0),
		cpuSamples:       make([]float64, 0),
		memorySamples:    make([]float64, 0),
		networkSamples:   make([]float64, 0),
	}

	// Set the algorithm
	err := suite.loadBalancer.SetAlgorithm(algorithm)
	require.NoError(suite.T(), err)

	startTime := time.Now()
	numJobs := 500
	concurrency := 25

	// Job template for load balancing test
	jobTemplate := &compute.ComputeJob{
		Name:        "lb-benchmark-job",
		Description: fmt.Sprintf("Load balancing benchmark for %s", algorithm),
		JobType:     compute.JobTypeBatch,
		Priority:    5,
		QueueName:   "loadbalance",
		Command:     []string{"echo", "load balancing test"},
		ResourceRequirements: compute.ResourceRequirements{
			CPUCores:    1.0,
			MemoryGB:    2.0,
			DiskGB:      5.0,
			NetworkMbps: 50.0,
		},
		Tags: map[string]string{
			"benchmark": "load_balancing",
			"algorithm": string(algorithm),
		},
		Timeout: 3 * time.Minute,
	}

	// Submit jobs concurrently
	var wg sync.WaitGroup
	jobChan := make(chan int, numJobs)
	for i := 0; i < numJobs; i++ {
		jobChan <- i
	}
	close(jobChan)

	wg.Add(concurrency)
	for w := 0; w < concurrency; w++ {
		go func() {
			defer wg.Done()
			for jobIndex := range jobChan {
				jobStartTime := time.Now()

				job := *jobTemplate
				job.Name = fmt.Sprintf("lb-benchmark-job-%s-%d", algorithm, jobIndex)

				jobID, err := suite.jobManager.SubmitJob(suite.ctx, &job)
				jobLatency := time.Since(jobStartTime)

				metrics.mutex.Lock()
				if err != nil {
					metrics.errors++
				} else {
					metrics.latencies = append(metrics.latencies, jobLatency)
				}
				metrics.mutex.Unlock()

				assert.NotEmpty(suite.T(), jobID)
			}
		}()
	}

	// Monitor metrics
	go suite.monitorSystemMetrics(metrics, testName)

	wg.Wait()
	duration := time.Since(startTime)

	// Wait for job processing
	time.Sleep(3 * time.Second)

	// Calculate and store results
	result := suite.calculateBenchmarkResult(testName, startTime, duration, numJobs, metrics)
	suite.storeBenchmarkResult(testName, result)

	suite.T().Logf("Load Balancing Benchmark (%s):", algorithm)
	suite.T().Logf("  Operations/sec: %.2f", result.OperationsPerSecond)
	suite.T().Logf("  Average Latency: %.2f ms", result.AverageLatencyMs)
	suite.T().Logf("  CPU Utilization: %.2f%%", result.CPUUtilization)
}

// BenchmarkResourceAllocationPerformance tests resource allocation performance
func (suite *CrossClusterPerformanceTestSuite) TestBenchmarkResourceAllocationPerformance() {
	testName := "resource_allocation_performance"
	metrics := &BenchmarkMetrics{
		latencies:         make([]time.Duration, 0),
		throughputSamples: make([]float64, 0),
		cpuSamples:       make([]float64, 0),
		memorySamples:    make([]float64, 0),
		networkSamples:   make([]float64, 0),
	}

	startTime := time.Now()
	numRequests := 1000
	concurrency := 40

	// Create resource request template
	requestTemplate := []scheduler.ResourceConstraint{
		{
			Type:      scheduler.ResourceCPU,
			MinAmount: 1.0,
			MaxAmount: 4.0,
		},
		{
			Type:      scheduler.ResourceMemory,
			MinAmount: 2.0,
			MaxAmount: 8.0,
		},
	}

	// Submit resource requests concurrently
	var wg sync.WaitGroup
	requestChan := make(chan int, numRequests)
	for i := 0; i < numRequests; i++ {
		requestChan <- i
	}
	close(requestChan)

	wg.Add(concurrency)
	for w := 0; w < concurrency; w++ {
		go func() {
			defer wg.Done()
			for range requestChan {
				reqStartTime := time.Now()

				requestID, err := suite.scheduler.RequestResources(
					requestTemplate, 5, 5*time.Minute)
				reqLatency := time.Since(reqStartTime)

				metrics.mutex.Lock()
				if err != nil {
					metrics.errors++
				} else {
					metrics.latencies = append(metrics.latencies, reqLatency)
				}
				metrics.mutex.Unlock()

				assert.NotEmpty(suite.T(), requestID)
			}
		}()
	}

	// Monitor metrics
	go suite.monitorSystemMetrics(metrics, testName)

	wg.Wait()
	duration := time.Since(startTime)

	// Wait for allocation processing
	time.Sleep(5 * time.Second)

	// Calculate results
	result := suite.calculateBenchmarkResult(testName, startTime, duration, numRequests, metrics)
	suite.storeBenchmarkResult(testName, result)

	// Performance assertions
	assert.Greater(suite.T(), result.OperationsPerSecond, 80.0, "Should handle at least 80 resource requests per second")
	assert.Less(suite.T(), result.AverageLatencyMs, 200.0, "Average allocation latency should be under 200ms")

	suite.T().Logf("Resource Allocation Benchmark Results:")
	suite.T().Logf("  Operations/sec: %.2f", result.OperationsPerSecond)
	suite.T().Logf("  Average Latency: %.2f ms", result.AverageLatencyMs)
	suite.T().Logf("  P95 Latency: %.2f ms", result.P95LatencyMs)
}

// BenchmarkMemoryFabricPerformance tests unified memory fabric performance
func (suite *CrossClusterPerformanceTestSuite) TestBenchmarkMemoryFabricPerformance() {
	testName := "memory_fabric_performance"
	metrics := &BenchmarkMetrics{
		latencies:         make([]time.Duration, 0),
		throughputSamples: make([]float64, 0),
		cpuSamples:       make([]float64, 0),
		memorySamples:    make([]float64, 0),
		networkSamples:   make([]float64, 0),
	}

	// Create a large memory pool for testing
	pool := &compute.MemoryPool{
		PoolID:      "benchmark-pool",
		ClusterID:   "cluster-1",
		TotalSizeGB: 1000.0, // 1 TB pool
		Configuration: compute.MemoryPoolConfig{
			CoherenceProtocol: compute.CoherenceProtocolMOESI,
			ReplicationFactor: 2,
			EnableCompression: true,
			EnableEncryption:  false, // Disable for performance
		},
	}

	err := suite.memoryFabric.CreateMemoryPool(suite.ctx, pool)
	require.NoError(suite.T(), err)

	startTime := time.Now()
	numAllocations := 500
	concurrency := 20

	// Concurrent memory allocations
	var wg sync.WaitGroup
	allocChan := make(chan int, numAllocations)
	for i := 0; i < numAllocations; i++ {
		allocChan <- i
	}
	close(allocChan)

	wg.Add(concurrency)
	for w := 0; w < concurrency; w++ {
		go func(workerID int) {
			defer wg.Done()
			for allocIndex := range allocChan {
				allocStartTime := time.Now()

				allocation := &compute.MemoryAllocation{
					AllocationID: fmt.Sprintf("benchmark-alloc-%d-%d", workerID, allocIndex),
					PoolID:       "benchmark-pool",
					SizeGB:       1.0 + float64(allocIndex%10), // Varying sizes
					Access:       compute.MemoryAccessReadWrite,
					Locality:     compute.MemoryLocalityLocal,
				}

				err := suite.memoryFabric.AllocateMemory(suite.ctx, allocation)
				allocLatency := time.Since(allocStartTime)

				metrics.mutex.Lock()
				if err != nil {
					metrics.errors++
				} else {
					metrics.latencies = append(metrics.latencies, allocLatency)
				}
				metrics.mutex.Unlock()
			}
		}(w)
	}

	// Monitor metrics
	go suite.monitorSystemMetrics(metrics, testName)

	wg.Wait()
	duration := time.Since(startTime)

	// Calculate results
	result := suite.calculateBenchmarkResult(testName, startTime, duration, numAllocations, metrics)
	suite.storeBenchmarkResult(testName, result)

	// Performance assertions
	assert.Greater(suite.T(), result.OperationsPerSecond, 40.0, "Should handle at least 40 memory allocations per second")
	assert.Less(suite.T(), result.AverageLatencyMs, 500.0, "Average allocation latency should be under 500ms")

	suite.T().Logf("Memory Fabric Benchmark Results:")
	suite.T().Logf("  Operations/sec: %.2f", result.OperationsPerSecond)
	suite.T().Logf("  Average Latency: %.2f ms", result.AverageLatencyMs)
	suite.T().Logf("  Memory Utilization: %.2f%%", result.MemoryUtilization)

	// Clean up
	suite.memoryFabric.DestroyMemoryPool(suite.ctx, "benchmark-pool")
}

// BenchmarkPerformanceOptimizer tests performance optimizer efficiency
func (suite *CrossClusterPerformanceTestSuite) TestBenchmarkPerformanceOptimizer() {
	testName := "performance_optimizer"

	// Wait for optimizer to collect baseline metrics
	time.Sleep(10 * time.Second)

	startTime := time.Now()

	// Trigger various optimization scenarios
	optimizationScenarios := []struct {
		name        string
		jobLoad     int
		resourceLoad float64
	}{
		{"low_load", 50, 0.3},
		{"medium_load", 200, 0.6},
		{"high_load", 500, 0.9},
	}

	for _, scenario := range optimizationScenarios {
		suite.T().Logf("Running optimization scenario: %s", scenario.name)

		// Submit jobs to create the scenario
		for i := 0; i < scenario.jobLoad; i++ {
			job := &compute.ComputeJob{
				Name:     fmt.Sprintf("opt-test-%s-%d", scenario.name, i),
				JobType:  compute.JobTypeBatch,
				Priority: 3,
				QueueName: "optimization-test",
				Command:   []string{"echo", "optimization test"},
				ResourceRequirements: compute.ResourceRequirements{
					CPUCores: 1.0 * scenario.resourceLoad,
					MemoryGB: 2.0 * scenario.resourceLoad,
				},
				Timeout: 2 * time.Minute,
			}

			_, err := suite.jobManager.SubmitJob(suite.ctx, job)
			require.NoError(suite.T(), err)
		}

		// Wait for optimization to kick in
		time.Sleep(45 * time.Second)

		// Get optimization recommendations
		recommendations := suite.perfOptimizer.GetOptimizationRecommendations()
		assert.NotNil(suite.T(), recommendations)

		// Get performance snapshot
		snapshot := suite.perfOptimizer.GetCurrentPerformanceSnapshot()
		assert.True(suite.T(), snapshot.Timestamp.After(time.Now().Add(-time.Minute)))

		suite.T().Logf("Scenario %s - CPU Util: %.2f%%, Memory Util: %.2f%%, Active Workloads: %d",
			scenario.name, snapshot.GlobalCPUUtilization, snapshot.GlobalMemoryUtilization, snapshot.ActiveWorkloads)
	}

	duration := time.Since(startTime)

	// Get optimizer statistics
	stats := suite.perfOptimizer.GetStatistics()
	require.NotNil(suite.T(), stats)

	// Create benchmark result
	result := &BenchmarkResult{
		TestName:            testName,
		StartTime:           startTime,
		Duration:            duration,
		Operations:          len(optimizationScenarios),
		OperationsPerSecond: float64(len(optimizationScenarios)) / duration.Seconds(),
		Metadata: map[string]interface{}{
			"optimizer_stats": stats,
			"scenarios_tested": len(optimizationScenarios),
		},
	}

	suite.storeBenchmarkResult(testName, result)

	// Verify optimizer is healthy and active
	assert.True(suite.T(), suite.perfOptimizer.IsHealthy())

	suite.T().Logf("Performance Optimizer Benchmark Results:")
	suite.T().Logf("  Scenarios tested: %d", len(optimizationScenarios))
	suite.T().Logf("  Total optimization tasks: %v", stats["total_optimization_tasks"])
	suite.T().Logf("  Running tasks: %v", stats["running_tasks"])
	suite.T().Logf("  Completed tasks: %v", stats["completed_tasks"])
}

// BenchmarkSystemScalability tests system scalability under increasing load
func (suite *CrossClusterPerformanceTestSuite) TestBenchmarkSystemScalability() {
	testName := "system_scalability"

	loadLevels := []int{100, 250, 500, 750, 1000}
	scalabilityResults := make(map[int]*BenchmarkResult)

	for _, loadLevel := range loadLevels {
		suite.T().Logf("Testing scalability at load level: %d jobs", loadLevel)

		metrics := &BenchmarkMetrics{
			latencies:         make([]time.Duration, 0),
			throughputSamples: make([]float64, 0),
			cpuSamples:       make([]float64, 0),
			memorySamples:    make([]float64, 0),
			networkSamples:   make([]float64, 0),
		}

		startTime := time.Now()

		// Submit jobs at this load level
		concurrency := min(loadLevel/10, 50) // Adaptive concurrency
		jobChan := make(chan int, loadLevel)
		for i := 0; i < loadLevel; i++ {
			jobChan <- i
		}
		close(jobChan)

		var wg sync.WaitGroup
		wg.Add(concurrency)

		for w := 0; w < concurrency; w++ {
			go func(workerID int) {
				defer wg.Done()
				for jobIndex := range jobChan {
					jobStartTime := time.Now()

					job := &compute.ComputeJob{
						Name:     fmt.Sprintf("scale-test-%d-%d-%d", loadLevel, workerID, jobIndex),
						JobType:  compute.JobTypeBatch,
						Priority: 5,
						QueueName: "scalability",
						Command:   []string{"echo", "scalability test"},
						ResourceRequirements: compute.ResourceRequirements{
							CPUCores: 1.0,
							MemoryGB: 2.0,
						},
						Timeout: 5 * time.Minute,
					}

					jobID, err := suite.jobManager.SubmitJob(suite.ctx, &job)
					jobLatency := time.Since(jobStartTime)

					metrics.mutex.Lock()
					if err != nil {
						metrics.errors++
					} else {
						metrics.latencies = append(metrics.latencies, jobLatency)
					}
					metrics.mutex.Unlock()

					assert.NotEmpty(suite.T(), jobID)
				}
			}(w)
		}

		// Monitor system metrics
		go suite.monitorSystemMetrics(metrics, fmt.Sprintf("%s_%d", testName, loadLevel))

		wg.Wait()
		duration := time.Since(startTime)

		// Wait for system to process
		time.Sleep(time.Duration(loadLevel/100) * time.Second)

		// Calculate results for this load level
		result := suite.calculateBenchmarkResult(fmt.Sprintf("%s_%d", testName, loadLevel), startTime, duration, loadLevel, metrics)
		scalabilityResults[loadLevel] = result

		suite.T().Logf("Load Level %d - Ops/sec: %.2f, Avg Latency: %.2f ms, Error Rate: %.2f%%",
			loadLevel, result.OperationsPerSecond, result.AverageLatencyMs, result.ErrorRate*100)

		// Brief pause between load levels
		time.Sleep(5 * time.Second)
	}

	// Analyze scalability characteristics
	suite.analyzeScalabilityResults(scalabilityResults)

	// Store overall scalability result
	overallResult := &BenchmarkResult{
		TestName:  testName,
		StartTime: time.Now(),
		Duration:  0, // Will be calculated from individual results
		Operations: func() int {
			total := 0
			for load := range scalabilityResults {
				total += load
			}
			return total
		}(),
		Metadata: map[string]interface{}{
			"load_levels":         loadLevels,
			"scalability_results": scalabilityResults,
		},
	}

	suite.storeBenchmarkResult(testName, overallResult)
}

// analyzeScalabilityResults analyzes scalability characteristics
func (suite *CrossClusterPerformanceTestSuite) analyzeScalabilityResults(results map[int]*BenchmarkResult) {
	suite.T().Log("Scalability Analysis:")

	// Check if throughput scales linearly
	previousOPS := 0.0
	for _, load := range []int{100, 250, 500, 750, 1000} {
		if result, exists := results[load]; exists {
			if previousOPS > 0 {
				scalingFactor := result.OperationsPerSecond / previousOPS
				suite.T().Logf("  Load %d: %.2f ops/sec (scaling factor: %.2fx)", load, result.OperationsPerSecond, scalingFactor)

				// Ideal scaling would be proportional to load increase
				// In practice, we expect some degradation
				assert.Greater(suite.T(), scalingFactor, 0.8, "Scaling factor should be at least 0.8x")
			} else {
				suite.T().Logf("  Load %d: %.2f ops/sec (baseline)", load, result.OperationsPerSecond)
			}
			previousOPS = result.OperationsPerSecond

			// Check that error rate doesn't increase dramatically
			assert.Less(suite.T(), result.ErrorRate, 0.05, fmt.Sprintf("Error rate should remain under 5%% at load %d", load))

			// Check that latency doesn't increase too much
			assert.Less(suite.T(), result.AverageLatencyMs, 2000.0, fmt.Sprintf("Average latency should remain under 2s at load %d", load))
		}
	}
}

// monitorSystemMetrics monitors system metrics during benchmark execution
func (suite *CrossClusterPerformanceTestSuite) monitorSystemMetrics(metrics *BenchmarkMetrics, testName string) {
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	// Monitor for up to 2 minutes
	timeout := time.After(2 * time.Minute)

	for {
		select {
		case <-timeout:
			return
		case <-ticker.C:
			// Get performance snapshot
			snapshot := suite.perfOptimizer.GetCurrentPerformanceSnapshot()

			metrics.mutex.Lock()
			metrics.cpuSamples = append(metrics.cpuSamples, snapshot.GlobalCPUUtilization)
			metrics.memorySamples = append(metrics.memorySamples, snapshot.GlobalMemoryUtilization)
			metrics.networkSamples = append(metrics.networkSamples, snapshot.GlobalNetworkUtilization)
			metrics.throughputSamples = append(metrics.throughputSamples, snapshot.TotalThroughput)
			metrics.mutex.Unlock()
		}
	}
}

// calculateBenchmarkResult calculates benchmark results from collected metrics
func (suite *CrossClusterPerformanceTestSuite) calculateBenchmarkResult(testName string, startTime time.Time, duration time.Duration, operations int, metrics *BenchmarkMetrics) *BenchmarkResult {
	metrics.mutex.RLock()
	defer metrics.mutex.RUnlock()

	result := &BenchmarkResult{
		TestName:        testName,
		StartTime:       startTime,
		Duration:        duration,
		Operations:      operations,
		ErrorCount:      metrics.errors,
		Metadata:        make(map[string]interface{}),
	}

	if duration.Seconds() > 0 {
		result.OperationsPerSecond = float64(operations) / duration.Seconds()
	}

	if operations > 0 {
		result.ErrorRate = float64(metrics.errors) / float64(operations)
	}

	// Calculate latency statistics
	if len(metrics.latencies) > 0 {
		// Sort latencies for percentile calculation
		sortedLatencies := make([]time.Duration, len(metrics.latencies))
		copy(sortedLatencies, metrics.latencies)
		sort.Slice(sortedLatencies, func(i, j int) bool {
			return sortedLatencies[i] < sortedLatencies[j]
		})

		// Average latency
		var totalLatency time.Duration
		for _, latency := range metrics.latencies {
			totalLatency += latency
		}
		result.AverageLatencyMs = float64(totalLatency.Nanoseconds()) / float64(len(metrics.latencies)) / 1e6

		// Percentiles
		p95Index := int(float64(len(sortedLatencies)) * 0.95)
		p99Index := int(float64(len(sortedLatencies)) * 0.99)

		if p95Index < len(sortedLatencies) {
			result.P95LatencyMs = float64(sortedLatencies[p95Index].Nanoseconds()) / 1e6
		}
		if p99Index < len(sortedLatencies) {
			result.P99LatencyMs = float64(sortedLatencies[p99Index].Nanoseconds()) / 1e6
		}
	}

	// Calculate resource utilization averages
	if len(metrics.cpuSamples) > 0 {
		var totalCPU float64
		for _, cpu := range metrics.cpuSamples {
			totalCPU += cpu
		}
		result.CPUUtilization = totalCPU / float64(len(metrics.cpuSamples))
	}

	if len(metrics.memorySamples) > 0 {
		var totalMemory float64
		for _, memory := range metrics.memorySamples {
			totalMemory += memory
		}
		result.MemoryUtilization = totalMemory / float64(len(metrics.memorySamples))
	}

	if len(metrics.networkSamples) > 0 {
		var totalNetwork float64
		for _, network := range metrics.networkSamples {
			totalNetwork += network
		}
		result.NetworkUtilization = totalNetwork / float64(len(metrics.networkSamples))
	}

	if len(metrics.throughputSamples) > 0 {
		var totalThroughput float64
		for _, throughput := range metrics.throughputSamples {
			totalThroughput += throughput
		}
		result.ThroughputMbps = totalThroughput / float64(len(metrics.throughputSamples))
	}

	return result
}

// storeBenchmarkResult stores benchmark results
func (suite *CrossClusterPerformanceTestSuite) storeBenchmarkResult(testName string, result *BenchmarkResult) {
	suite.resultsMutex.Lock()
	defer suite.resultsMutex.Unlock()
	suite.benchmarkResults[testName] = result
}

// printBenchmarkSummary prints a summary of all benchmark results
func (suite *CrossClusterPerformanceTestSuite) printBenchmarkSummary() {
	suite.T().Log("\n" + "="*80)
	suite.T().Log("DISTRIBUTED SUPERCOMPUTE PERFORMANCE BENCHMARK SUMMARY")
	suite.T().Log("="*80)

	suite.resultsMutex.RLock()
	defer suite.resultsMutex.RUnlock()

	for testName, result := range suite.benchmarkResults {
		suite.T().Logf("\nTest: %s", testName)
		suite.T().Logf("  Duration: %v", result.Duration)
		suite.T().Logf("  Operations: %d", result.Operations)
		suite.T().Logf("  Operations/sec: %.2f", result.OperationsPerSecond)
		suite.T().Logf("  Average Latency: %.2f ms", result.AverageLatencyMs)
		suite.T().Logf("  P95 Latency: %.2f ms", result.P95LatencyMs)
		suite.T().Logf("  P99 Latency: %.2f ms", result.P99LatencyMs)
		suite.T().Logf("  CPU Utilization: %.2f%%", result.CPUUtilization)
		suite.T().Logf("  Memory Utilization: %.2f%%", result.MemoryUtilization)
		suite.T().Logf("  Network Utilization: %.2f%%", result.NetworkUtilization)
		suite.T().Logf("  Throughput: %.2f Mbps", result.ThroughputMbps)
		suite.T().Logf("  Error Count: %d", result.ErrorCount)
		suite.T().Logf("  Error Rate: %.4f%%", result.ErrorRate*100)
	}

	suite.T().Log("\n" + "="*80)
}

// min returns the minimum of two integers
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// Run the performance test suite
func TestCrossClusterPerformanceTestSuite(t *testing.T) {
	suite.Run(t, new(CrossClusterPerformanceTestSuite))
}