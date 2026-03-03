package e2e

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"math/rand"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
)

// E2ETestConfig holds configuration for E2E tests
type E2ETestConfig struct {
	APIURL          string
	MonitoringURL   string
	PrometheusURL   string
	APIKey          string
	InsecureSkipTLS bool
	TLSConfig       *tls.Config
}

// APIClient provides utilities for testing HTTP APIs
type APIClient struct {
	BaseURL    string
	APIKey     string
	HTTPClient *http.Client
	AuthToken  string
}

// MonitoringClient provides utilities for testing monitoring endpoints
type MonitoringClient struct {
	BaseURL    string
	HTTPClient *http.Client
}

// SupercomputeScenariosSuite provides comprehensive end-to-end tests for supercompute scenarios
type SupercomputeScenariosSuite struct {
	suite.Suite
	ctx              context.Context
	cancel           context.CancelFunc
	config           *E2ETestConfig
	apiClient        *APIClient
	monitoringClient *MonitoringClient
	clusters         map[string]*ClusterInfo
	workloadManager  *WorkloadManager
	metricsCollector *MetricsCollector
}

// getE2ETestConfig reads configuration from environment variables
func getE2ETestConfig() *E2ETestConfig {
	apiURL := os.Getenv("E2E_API_URL")
	if apiURL == "" {
		apiURL = "http://localhost:8080"
	}

	monitoringURL := os.Getenv("E2E_MONITORING_URL")
	if monitoringURL == "" {
		monitoringURL = "http://localhost:9090"
	}

	prometheusURL := os.Getenv("E2E_PROMETHEUS_URL")
	if prometheusURL == "" {
		prometheusURL = "http://prometheus:9090"
	}

	apiKey := os.Getenv("E2E_API_KEY")
	insecureSkipTLS := os.Getenv("E2E_INSECURE_SKIP_TLS") == "true"

	var tlsConfig *tls.Config
	if strings.HasPrefix(apiURL, "https://") || strings.HasPrefix(monitoringURL, "https://") || strings.HasPrefix(prometheusURL, "https://") {
		tlsConfig = &tls.Config{
			InsecureSkipVerify: insecureSkipTLS,
		}
	}

	return &E2ETestConfig{
		APIURL:          apiURL,
		MonitoringURL:   monitoringURL,
		PrometheusURL:   prometheusURL,
		APIKey:          apiKey,
		InsecureSkipTLS: insecureSkipTLS,
		TLSConfig:       tlsConfig,
	}
}

// NewAPIClient creates a new API client with TLS support
func NewAPIClient(baseURL, apiKey string, tlsConfig *tls.Config) *APIClient {
	transport := &http.Transport{}
	if tlsConfig != nil {
		transport.TLSClientConfig = tlsConfig
	}

	return &APIClient{
		BaseURL: strings.TrimSuffix(baseURL, "/"),
		APIKey:  apiKey,
		HTTPClient: &http.Client{
			Timeout:   30 * time.Second,
			Transport: transport,
		},
	}
}

// NewMonitoringClient creates a new monitoring client with TLS support
func NewMonitoringClient(baseURL string, tlsConfig *tls.Config) *MonitoringClient {
	transport := &http.Transport{}
	if tlsConfig != nil {
		transport.TLSClientConfig = tlsConfig
	}

	return &MonitoringClient{
		BaseURL: strings.TrimSuffix(baseURL, "/"),
		HTTPClient: &http.Client{
			Timeout:   30 * time.Second,
			Transport: transport,
		},
	}
}

// SetupSuite initializes the test environment
func (s *SupercomputeScenariosSuite) SetupSuite() {
	s.ctx, s.cancel = context.WithTimeout(context.Background(), 30*time.Minute)

	// Load configuration from environment
	s.config = getE2ETestConfig()

	// Initialize clients with configurable URLs and TLS support
	s.apiClient = NewAPIClient(s.config.APIURL, s.config.APIKey, s.config.TLSConfig)
	s.monitoringClient = NewMonitoringClient(s.config.MonitoringURL, s.config.TLSConfig)
	s.workloadManager = NewWorkloadManager(s.apiClient)
	s.metricsCollector = NewMetricsCollector(s.monitoringClient)

	// Setup multi-cluster environment
	s.setupClusters()

	// Verify environment readiness
	s.verifyEnvironment()
}

// TearDownSuite cleans up resources
func (s *SupercomputeScenariosSuite) TearDownSuite() {
	// Clean up all workloads
	s.workloadManager.CleanupAll()

	// Collect final metrics
	s.metricsCollector.GenerateReport()

	s.cancel()
}

// TestDistributedScientificComputing tests complete scientific computing workflow
func (s *SupercomputeScenariosSuite) TestDistributedScientificComputing() {
	s.T().Run("MPI_MultiNode_Job", func(t *testing.T) {
		// Submit MPI job across multiple nodes
		job := &ScientificJob{
			Name:         "mpi-simulation",
			Type:         "MPI",
			Nodes:        16,
			CoresPerNode: 32,
			Memory:       "256GB",
			Application:  "weather-prediction",
			InputData: DataSet{
				Size:     "1TB",
				Location: "s3://scientific-data/weather",
			},
		}

		// Submit job
		jobID, err := s.workloadManager.SubmitScientificJob(job)
		require.NoError(t, err)

		// Monitor job execution
		s.monitorJobExecution(t, jobID, 10*time.Minute)

		// Verify results
		results, err := s.workloadManager.GetJobResults(jobID)
		require.NoError(t, err)
		assert.Equal(t, "completed", results.Status)
		assert.NotEmpty(t, results.OutputLocation)

		// Validate performance metrics
		metrics := s.metricsCollector.GetJobMetrics(jobID)
		assert.Greater(t, metrics.TFLOPS, 100.0)
		assert.Less(t, metrics.NetworkLatency, 10*time.Millisecond)
	})

	s.T().Run("Distributed_Data_Processing", func(t *testing.T) {
		// Setup distributed data processing pipeline
		pipeline := &DataPipeline{
			Name: "genomics-analysis",
			Stages: []PipelineStage{
				{Name: "data-ingestion", Parallelism: 10},
				{Name: "preprocessing", Parallelism: 20},
				{Name: "analysis", Parallelism: 50},
				{Name: "aggregation", Parallelism: 5},
			},
			InputData: DataSet{
				Size:     "5TB",
				Format:   "FASTQ",
				Location: "s3://genomics/raw-data",
			},
		}

		// Execute pipeline
		pipelineID, err := s.workloadManager.ExecuteDataPipeline(pipeline)
		require.NoError(t, err)

		// Monitor pipeline progress
		s.monitorPipelineExecution(t, pipelineID, 20*time.Minute)

		// Verify data integrity
		s.verifyDataIntegrity(t, pipelineID)

		// Check resource utilization
		utilization := s.metricsCollector.GetResourceUtilization(pipelineID)
		assert.Greater(t, utilization.CPUEfficiency, 0.8)
		assert.Greater(t, utilization.MemoryEfficiency, 0.7)
	})

	s.T().Run("Cross_Cluster_Resource_Allocation", func(t *testing.T) {
		// Test resource allocation across multiple clusters
		allocation := &ResourceAllocation{
			JobID: "scientific-workload-001",
			Requirements: ResourceRequirements{
				CPUs:    512,
				Memory:  "4TB",
				GPUs:    8,
				Storage: "10TB",
			},
			Constraints: []Constraint{
				{Type: "affinity", Value: "gpu-cluster"},
				{Type: "spread", Value: "availability-zones"},
			},
		}

		// Request allocation
		allocationID, err := s.workloadManager.AllocateResources(allocation)
		require.NoError(t, err)

		// Verify allocation across clusters
		details, err := s.workloadManager.GetAllocationDetails(allocationID)
		require.NoError(t, err)
		assert.GreaterOrEqual(t, len(details.Clusters), 2)
		assert.Equal(t, allocation.Requirements.CPUs, details.TotalCPUs)

		// Test resource scaling
		s.testResourceScaling(t, allocationID)
	})
}

// TestMachineLearningTrainingPipeline tests distributed ML training
func (s *SupercomputeScenariosSuite) TestMachineLearningTrainingPipeline() {
	s.T().Run("Distributed_Model_Training", func(t *testing.T) {
		// Setup distributed training job
		trainingJob := &MLTrainingJob{
			Name:      "llm-training",
			Framework: "PyTorch",
			Model: ModelConfig{
				Type:       "Transformer",
				Parameters: 175e9, // 175B parameters
			},
			Data: DataConfig{
				Dataset:  "common-crawl",
				Size:     "100TB",
				Sharding: "automatic",
			},
			Resources: TrainingResources{
				GPUs:             64,
				GPUType:          "A100",
				Workers:          8,
				ParameterServers: 4,
			},
			Strategy: "data-parallel",
		}

		// Submit training job
		jobID, err := s.workloadManager.SubmitMLTraining(trainingJob)
		require.NoError(t, err)

		// Monitor training progress
		s.monitorTrainingProgress(t, jobID)

		// Verify model checkpoints
		checkpoints, err := s.workloadManager.GetCheckpoints(jobID)
		require.NoError(t, err)
		assert.GreaterOrEqual(t, len(checkpoints), 5)

		// Validate training metrics
		metrics := s.metricsCollector.GetTrainingMetrics(jobID)
		assert.Less(t, metrics.Loss, 2.0)
		assert.Greater(t, metrics.Throughput, 1000.0) // samples/sec
	})

	s.T().Run("Model_Synchronization", func(t *testing.T) {
		// Test model parameter synchronization across workers
		syncTest := &SynchronizationTest{
			Workers:    16,
			ModelSize:  "10GB",
			SyncMethod: "all-reduce",
		}

		// Start synchronization test
		testID, err := s.workloadManager.TestSynchronization(syncTest)
		require.NoError(t, err)

		// Measure synchronization performance
		syncMetrics := s.metricsCollector.GetSyncMetrics(testID)
		assert.Less(t, syncMetrics.AvgSyncTime, 100*time.Millisecond)
		assert.Greater(t, syncMetrics.Bandwidth, 10.0) // GB/s

		// Verify consistency
		s.verifyModelConsistency(t, testID)
	})

	s.T().Run("Gradient_Aggregation", func(t *testing.T) {
		// Test gradient aggregation strategies
		strategies := []string{"synchronous", "asynchronous", "federated"}

		for _, strategy := range strategies {
			t.Run(strategy, func(t *testing.T) {
				config := &GradientConfig{
					Strategy:     strategy,
					Workers:      32,
					BatchSize:    512,
					LearningRate: 0.001,
				}

				// Execute gradient aggregation test
				testID, err := s.workloadManager.TestGradientAggregation(config)
				require.NoError(t, err)

				// Verify convergence
				convergence := s.metricsCollector.GetConvergenceMetrics(testID)
				assert.True(t, convergence.Converged)
				assert.Less(t, convergence.Epochs, 100)
			})
		}
	})
}

// TestHighPerformanceComputing tests HPC workloads
func (s *SupercomputeScenariosSuite) TestHighPerformanceComputing() {
	s.T().Run("Parallel_Computing_Jobs", func(t *testing.T) {
		// Submit parallel computing job
		job := &HPCJob{
			Name:        "cfd-simulation",
			Type:        "OpenMP",
			Parallelism: 128,
			Memory:      "512GB",
			Compute: ComputeConfig{
				FloatingPointOps: "double",
				Vectorization:    "AVX512",
			},
		}

		jobID, err := s.workloadManager.SubmitHPCJob(job)
		require.NoError(t, err)

		// Monitor execution
		s.monitorHPCExecution(t, jobID)

		// Verify performance
		perf := s.metricsCollector.GetHPCPerformance(jobID)
		assert.Greater(t, perf.GFLOPS, 1000.0)
		assert.Greater(t, perf.ParallelEfficiency, 0.85)
	})

	s.T().Run("GPU_Accelerated_Workloads", func(t *testing.T) {
		// Test GPU-accelerated computation
		workload := &GPUWorkload{
			Name:      "molecular-dynamics",
			GPUs:      8,
			GPUType:   "V100",
			Framework: "CUDA",
			Kernel: KernelConfig{
				BlockSize: 256,
				GridSize:  1024,
				SharedMem: "48KB",
			},
		}

		workloadID, err := s.workloadManager.SubmitGPUWorkload(workload)
		require.NoError(t, err)

		// Monitor GPU utilization
		gpuMetrics := s.monitorGPUUtilization(t, workloadID)
		assert.Greater(t, gpuMetrics.Utilization, 0.9)
		assert.Greater(t, gpuMetrics.MemoryBandwidth, 500.0) // GB/s
	})

	s.T().Run("Memory_Intensive_Applications", func(t *testing.T) {
		// Test memory-intensive workloads
		app := &MemoryIntensiveApp{
			Name:          "graph-analytics",
			MemorySize:    "2TB",
			AccessPattern: "random",
			DataStructure: "sparse-matrix",
		}

		appID, err := s.workloadManager.DeployMemoryApp(app)
		require.NoError(t, err)

		// Monitor memory performance
		memMetrics := s.metricsCollector.GetMemoryMetrics(appID)
		assert.Less(t, memMetrics.PageFaults, 100)
		assert.Greater(t, memMetrics.Bandwidth, 100.0) // GB/s
	})
}

// TestRealtimeDistributedProcessing tests real-time processing scenarios
func (s *SupercomputeScenariosSuite) TestRealtimeDistributedProcessing() {
	s.T().Run("Streaming_Data_Processing", func(t *testing.T) {
		// Setup streaming pipeline
		stream := &StreamingPipeline{
			Name:       "iot-analytics",
			InputRate:  1000000, // events/sec
			WindowSize: 10 * time.Second,
			Processing: []StreamProcessor{
				{Name: "filter", Parallelism: 10},
				{Name: "aggregate", Parallelism: 5},
				{Name: "ml-inference", Parallelism: 20},
			},
		}

		streamID, err := s.workloadManager.DeployStreamingPipeline(stream)
		require.NoError(t, err)

		// Send test events
		s.sendStreamingEvents(t, streamID, 60*time.Second)

		// Verify latency
		latencyMetrics := s.metricsCollector.GetStreamingLatency(streamID)
		assert.Less(t, latencyMetrics.P99, 100*time.Millisecond)
		assert.Less(t, latencyMetrics.P95, 50*time.Millisecond)
	})

	s.T().Run("Event_Driven_Computing", func(t *testing.T) {
		// Deploy event-driven application
		app := &EventDrivenApp{
			Name: "financial-trading",
			EventSources: []EventSource{
				{Type: "market-data", Rate: 50000},
				{Type: "news-feed", Rate: 1000},
			},
			Processors: []EventProcessor{
				{Name: "risk-calculator", Instances: 10},
				{Name: "order-executor", Instances: 5},
			},
			LatencyRequirement: 10 * time.Millisecond,
		}

		appID, err := s.workloadManager.DeployEventDrivenApp(app)
		require.NoError(t, err)

		// Generate events and measure response
		s.generateEvents(t, appID, 5*time.Minute)

		// Verify SLA compliance
		slaMetrics := s.metricsCollector.GetSLAMetrics(appID)
		assert.Greater(t, slaMetrics.Compliance, 0.999)
	})
}

// TestMultiTenantSupercompute tests multi-tenant scenarios
func (s *SupercomputeScenariosSuite) TestMultiTenantSupercompute() {
	s.T().Run("Resource_Isolation", func(t *testing.T) {
		// Create multiple tenants
		tenants := s.createTenants(t, 5)

		// Deploy workloads for each tenant
		var workloads []string
		for _, tenant := range tenants {
			workload := &TenantWorkload{
				TenantID: tenant.ID,
				Resources: ResourceQuota{
					CPUs:   100,
					Memory: "500GB",
					GPUs:   2,
				},
			}

			wID, err := s.workloadManager.DeployTenantWorkload(workload)
			require.NoError(t, err)
			workloads = append(workloads, wID)
		}

		// Verify isolation
		for i, wID := range workloads {
			isolation := s.verifyResourceIsolation(t, wID, tenants[i].ID)
			assert.True(t, isolation.CPUIsolated)
			assert.True(t, isolation.MemoryIsolated)
			assert.True(t, isolation.NetworkIsolated)
		}
	})

	s.T().Run("Performance_Guarantees", func(t *testing.T) {
		// Setup QoS for different tenant tiers
		tiers := []TenantTier{
			{Name: "platinum", GuaranteedPerformance: 1.0},
			{Name: "gold", GuaranteedPerformance: 0.8},
			{Name: "silver", GuaranteedPerformance: 0.6},
		}

		for _, tier := range tiers {
			t.Run(tier.Name, func(t *testing.T) {
				// Deploy workload with QoS
				workload := &QoSWorkload{
					Tier: tier.Name,
					SLA: SLAConfig{
						MinThroughput: 1000 * tier.GuaranteedPerformance,
						MaxLatency:    time.Duration(100/tier.GuaranteedPerformance) * time.Millisecond,
					},
				}

				wID, err := s.workloadManager.DeployQoSWorkload(workload)
				require.NoError(t, err)

				// Run under load
				s.generateLoad(t, wID, 10*time.Minute)

				// Verify SLA
				metrics := s.metricsCollector.GetQoSMetrics(wID)
				assert.GreaterOrEqual(t, metrics.Throughput, workload.SLA.MinThroughput)
				assert.LessOrEqual(t, metrics.Latency, workload.SLA.MaxLatency)
			})
		}
	})
}

// TestDisasterRecoveryAndFailover tests DR scenarios
func (s *SupercomputeScenariosSuite) TestDisasterRecoveryAndFailover() {
	s.T().Run("Cluster_Failure_Recovery", func(t *testing.T) {
		// Deploy critical workload
		workload := s.deployCriticalWorkload(t)

		// Simulate cluster failure
		failedCluster := s.clusters["us-east-1"]
		err := s.simulateClusterFailure(failedCluster)
		require.NoError(t, err)

		// Verify automatic failover
		time.Sleep(30 * time.Second)

		status, err := s.workloadManager.GetWorkloadStatus(workload.ID)
		require.NoError(t, err)
		assert.Equal(t, "running", status.State)
		assert.NotEqual(t, failedCluster.ID, status.ClusterID)

		// Verify no data loss
		s.verifyDataIntegrity(t, workload.ID)

		// Restore cluster
		err = s.restoreCluster(failedCluster)
		require.NoError(t, err)
	})

	s.T().Run("Data_Recovery", func(t *testing.T) {
		// Create test data
		dataID := s.createTestData(t, "10GB")

		// Simulate data corruption
		err := s.simulateDataCorruption(dataID)
		require.NoError(t, err)

		// Trigger recovery
		recoveryID, err := s.workloadManager.RecoverData(dataID)
		require.NoError(t, err)

		// Monitor recovery
		s.monitorRecovery(t, recoveryID)

		// Verify data integrity
		integrity := s.verifyDataIntegrity(t, dataID)
		assert.True(t, integrity)
	})

	s.T().Run("Service_Continuity", func(t *testing.T) {
		// Deploy service with HA
		service := &HAService{
			Name:     "critical-api",
			Replicas: 5,
			Zones:    []string{"us-east-1a", "us-east-1b", "us-west-2a"},
		}

		serviceID, err := s.workloadManager.DeployHAService(service)
		require.NoError(t, err)

		// Simulate rolling failures
		s.simulateRollingFailures(t, serviceID)

		// Verify service availability
		availability := s.metricsCollector.GetAvailability(serviceID)
		assert.Greater(t, availability, 0.9999) // 4 nines
	})
}

// TestAutoScalingAndOptimization tests automatic scaling and optimization
func (s *SupercomputeScenariosSuite) TestAutoScalingAndOptimization() {
	s.T().Run("Predictive_Scaling", func(t *testing.T) {
		// Deploy workload with predictive scaling
		workload := &AutoScaleWorkload{
			Name: "web-service",
			ScalingPolicy: ScalingPolicy{
				Type:        "predictive",
				MinReplicas: 2,
				MaxReplicas: 100,
				Metrics: []ScalingMetric{
					{Type: "cpu", Target: 70},
					{Type: "memory", Target: 80},
					{Type: "latency", Target: 100},
				},
			},
		}

		workloadID, err := s.workloadManager.DeployAutoScaleWorkload(workload)
		require.NoError(t, err)

		// Generate variable load
		s.generateVariableLoad(t, workloadID, 30*time.Minute)

		// Verify scaling behavior
		scalingMetrics := s.metricsCollector.GetScalingMetrics(workloadID)
		assert.Less(t, scalingMetrics.UnderProvisionedTime, 1*time.Minute)
		assert.Less(t, scalingMetrics.OverProvisionedTime, 5*time.Minute)
	})

	s.T().Run("Resource_Optimization", func(t *testing.T) {
		// Deploy workload for optimization
		workload := s.deployOptimizableWorkload(t)

		// Enable optimization
		err := s.workloadManager.EnableOptimization(workload.ID, OptimizationConfig{
			Goals: []OptimizationGoal{
				{Type: "cost", Weight: 0.3},
				{Type: "performance", Weight: 0.5},
				{Type: "availability", Weight: 0.2},
			},
		})
		require.NoError(t, err)

		// Run for optimization period
		time.Sleep(15 * time.Minute)

		// Verify optimization results
		results := s.metricsCollector.GetOptimizationResults(workload.ID)
		assert.Greater(t, results.CostReduction, 0.2)
		assert.Greater(t, results.PerformanceGain, 0.15)
	})

	s.T().Run("Cost_Optimization", func(t *testing.T) {
		// Test cost-aware scheduling
		workload := &CostOptimizedWorkload{
			Name: "batch-processing",
			Constraints: CostConstraints{
				MaxHourlyCost:  100.0,
				PreferSpot:     true,
				RegionFlexible: true,
			},
		}

		workloadID, err := s.workloadManager.DeployCostOptimized(workload)
		require.NoError(t, err)

		// Monitor cost
		costMetrics := s.metricsCollector.GetCostMetrics(workloadID)
		assert.Less(t, costMetrics.HourlyCost, workload.Constraints.MaxHourlyCost)
		assert.Greater(t, costMetrics.SpotUsage, 0.7)
	})
}

// TestExternalSystemIntegration tests integration with external systems
func (s *SupercomputeScenariosSuite) TestExternalSystemIntegration() {
	s.T().Run("Storage_System_Integration", func(t *testing.T) {
		// Test S3 integration
		s3Config := &StorageConfig{
			Type:   "S3",
			Bucket: "novacron-test",
			Region: "us-east-1",
		}

		storageID, err := s.workloadManager.ConfigureStorage(s3Config)
		require.NoError(t, err)

		// Test data operations
		testData := generateTestData(10 * 1024 * 1024) // 10MB

		// Upload
		uploadID, err := s.workloadManager.UploadData(storageID, "test-file", testData)
		require.NoError(t, err)

		// Download
		downloadedData, err := s.workloadManager.DownloadData(storageID, "test-file")
		require.NoError(t, err)
		assert.Equal(t, testData, downloadedData)
	})

	s.T().Run("Monitoring_Integration", func(t *testing.T) {
		// Test Prometheus integration
		promConfig := &MonitoringConfig{
			Type:     "Prometheus",
			Endpoint: s.config.PrometheusURL,
			Interval: 30 * time.Second,
		}

		monitorID, err := s.workloadManager.ConfigureMonitoring(promConfig)
		require.NoError(t, err)

		// Verify metrics collection
		time.Sleep(1 * time.Minute)

		metrics, err := s.monitoringClient.QueryMetrics("novacron_*")
		require.NoError(t, err)
		assert.NotEmpty(t, metrics)
	})

	s.T().Run("Job_Scheduler_Integration", func(t *testing.T) {
		// Test Slurm integration
		slurmConfig := &SchedulerConfig{
			Type:     "Slurm",
			Endpoint: "slurm-controller:6817",
		}

		schedulerID, err := s.workloadManager.ConfigureScheduler(slurmConfig)
		require.NoError(t, err)

		// Submit job via scheduler
		job := &ScheduledJob{
			Name:      "scheduled-job",
			Command:   "/usr/bin/simulation",
			Resources: "nodes=4:ppn=16",
		}

		jobID, err := s.workloadManager.SubmitScheduledJob(schedulerID, job)
		require.NoError(t, err)

		// Verify job execution
		status, err := s.workloadManager.GetScheduledJobStatus(jobID)
		require.NoError(t, err)
		assert.Contains(t, []string{"pending", "running", "completed"}, status)
	})
}

// Helper functions

func (s *SupercomputeScenariosSuite) setupClusters() {
	s.clusters = map[string]*ClusterInfo{
		"us-east-1": {
			ID:       "cluster-use1",
			Region:   "us-east-1",
			Capacity: 1000,
		},
		"us-west-2": {
			ID:       "cluster-usw2",
			Region:   "us-west-2",
			Capacity: 800,
		},
		"eu-west-1": {
			ID:       "cluster-euw1",
			Region:   "eu-west-1",
			Capacity: 600,
		},
	}
}

func (s *SupercomputeScenariosSuite) verifyEnvironment() {
	// Verify API connectivity
	health, err := s.apiClient.GetHealth()
	s.Require().NoError(err)
	s.Require().Equal("healthy", health.Status)

	// Verify cluster availability
	for _, cluster := range s.clusters {
		status, err := s.apiClient.GetClusterStatus(cluster.ID)
		s.Require().NoError(err)
		s.Require().Equal("ready", status)
	}
}

func (s *SupercomputeScenariosSuite) monitorJobExecution(t *testing.T, jobID string, timeout time.Duration) {
	ctx, cancel := context.WithTimeout(s.ctx, timeout)
	defer cancel()

	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			t.Fatal("Job execution timeout")
		case <-ticker.C:
			status, err := s.workloadManager.GetJobStatus(jobID)
			require.NoError(t, err)

			if status.State == "completed" {
				return
			}

			if status.State == "failed" {
				t.Fatalf("Job failed: %s", status.Error)
			}
		}
	}
}

func (s *SupercomputeScenariosSuite) monitorPipelineExecution(t *testing.T, pipelineID string, timeout time.Duration) {
	ctx, cancel := context.WithTimeout(s.ctx, timeout)
	defer cancel()

	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			t.Fatal("Pipeline execution timeout")
		case <-ticker.C:
			progress, err := s.workloadManager.GetPipelineProgress(pipelineID)
			require.NoError(t, err)

			t.Logf("Pipeline progress: %d%%", progress.Percentage)

			if progress.Percentage == 100 {
				return
			}
		}
	}
}

func (s *SupercomputeScenariosSuite) verifyDataIntegrity(t *testing.T, id string) bool {
	checksum, err := s.workloadManager.CalculateChecksum(id)
	require.NoError(t, err)

	expected, err := s.workloadManager.GetExpectedChecksum(id)
	require.NoError(t, err)

	return checksum == expected
}

func (s *SupercomputeScenariosSuite) testResourceScaling(t *testing.T, allocationID string) {
	// Scale up
	err := s.workloadManager.ScaleResources(allocationID, 2.0)
	require.NoError(t, err)

	time.Sleep(30 * time.Second)

	// Verify scaling
	details, err := s.workloadManager.GetAllocationDetails(allocationID)
	require.NoError(t, err)
	assert.Equal(t, 1024, details.TotalCPUs)

	// Scale down
	err = s.workloadManager.ScaleResources(allocationID, 0.5)
	require.NoError(t, err)
}

// TestSuite execution
func TestSupercomputeScenarios(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping supercompute scenarios in short mode")
	}

	suite.Run(t, new(SupercomputeScenariosSuite))
}
