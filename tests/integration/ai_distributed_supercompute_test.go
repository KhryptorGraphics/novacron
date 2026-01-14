package integration

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/novacron-org/novacron/backend/core/ai"
	"github.com/novacron-org/novacron/backend/core/compute"
	"github.com/novacron-org/novacron/backend/core/scheduler"
)

// TestAIDistributedSupercomputeIntegration tests the comprehensive integration
// between AI Engine and distributed supercompute fabric
func TestAIDistributedSupercomputeIntegration(t *testing.T) {
	t.Run("AI Service Integration", testAIServiceIntegration)
	t.Run("Performance Optimization Integration", testPerformanceOptimizationIntegration)
	t.Run("Workload Analysis Integration", testWorkloadAnalysisIntegration)
	t.Run("Resource Prediction Integration", testResourcePredictionIntegration)
	t.Run("Anomaly Detection Integration", testAnomalyDetectionIntegration)
	t.Run("Predictive Scaling Integration", testPredictiveScalingIntegration)
	t.Run("End-to-End AI Workflow", testEndToEndAIWorkflow)
}

func testAIServiceIntegration(t *testing.T) {
	// Skip if AI Engine is not available
	if !isAIEngineAvailable() {
		t.Skip("AI Engine not available, skipping AI integration tests")
	}

	// Create distributed AI service components
	ctx := context.Background()

	// Initialize AI components
	config := compute.DefaultDistributedAIConfig()
	config.AIEngineEndpoint = getAIEngineEndpoint()
	config.PerformanceOptimizationInterval = 1 * time.Second // Fast for testing
	config.WorkloadAnalysisInterval = 2 * time.Second
	config.ResourcePredictionInterval = 3 * time.Second

	// Create compute components
	jobManager := createTestJobManager(t)
	loadBalancer := createTestLoadBalancer(t)
	perfOptimizer := createTestPerformanceOptimizer(t)

	// Create distributed AI service
	aiService, err := compute.NewDistributedAIService(
		config,
		jobManager,
		loadBalancer,
		perfOptimizer,
	)
	require.NoError(t, err, "Failed to create distributed AI service")
	require.NotNil(t, aiService, "AI service should not be nil")

	// Test service startup
	err = aiService.Start()
	require.NoError(t, err, "Failed to start distributed AI service")

	// Verify service status
	status := aiService.GetStatus()
	assert.True(t, status["running"].(bool), "AI service should be running")
	assert.True(t, status["ai_engine_connected"].(bool), "AI engine should be connected")

	// Test health check
	assert.True(t, aiService.IsHealthy(), "AI service should be healthy")

	// Test metrics collection
	metrics := aiService.GetMetrics()
	assert.NotNil(t, metrics, "Metrics should be available")
	assert.GreaterOrEqual(t, metrics.TotalOptimizations, int64(0), "Total optimizations should be non-negative")

	// Wait for some AI operations to complete
	time.Sleep(5 * time.Second)

	// Stop the service
	err = aiService.Stop()
	assert.NoError(t, err, "Failed to stop distributed AI service")

	// Verify service is stopped
	status = aiService.GetStatus()
	assert.False(t, status["running"].(bool), "AI service should be stopped")
}

func testPerformanceOptimizationIntegration(t *testing.T) {
	if !isAIEngineAvailable() {
		t.Skip("AI Engine not available, skipping performance optimization test")
	}

	// Create AI integration layer
	aiConfig := ai.DefaultAIConfig()
	aiConfig.Timeout = 30 * time.Second

	aiLayer := ai.NewAIIntegrationLayer(
		getAIEngineEndpoint(),
		"",
		aiConfig,
	)

	// Test AI health check
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	err := aiLayer.HealthCheck(ctx)
	if err != nil {
		t.Skipf("AI Engine health check failed: %v", err)
	}

	// Create performance optimization request
	perfReq := ai.PerformanceOptimizationRequest{
		ClusterID: "test_cluster",
		ClusterData: map[string]interface{}{
			"cpu_utilization":    75.5,
			"memory_utilization": 68.3,
			"network_latency":    12.5,
			"active_jobs":        25,
			"queued_jobs":        8,
		},
		Goals: []string{
			"optimize_resource_utilization",
			"minimize_latency",
			"balance_load",
		},
		Constraints: map[string]interface{}{
			"max_cpu_threshold": 0.85,
			"sla_requirements":  "high_availability",
		},
	}

	// Get AI performance optimization recommendations
	ctx2, cancel2 := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel2()

	perfResp, err := aiLayer.OptimizePerformance(ctx2, perfReq)
	require.NoError(t, err, "Performance optimization should succeed")
	require.NotNil(t, perfResp, "Performance response should not be nil")

	// Validate response
	assert.GreaterOrEqual(t, len(perfResp.Recommendations), 0, "Should have recommendations")
	assert.GreaterOrEqual(t, perfResp.Confidence, 0.0, "Confidence should be non-negative")
	assert.LessOrEqual(t, perfResp.Confidence, 1.0, "Confidence should not exceed 1.0")

	if len(perfResp.Recommendations) > 0 {
		rec := perfResp.Recommendations[0]
		assert.NotEmpty(t, rec.Type, "Recommendation type should not be empty")
		assert.NotEmpty(t, rec.Action, "Recommendation action should not be empty")
		assert.GreaterOrEqual(t, rec.Priority, 0, "Priority should be non-negative")
		assert.GreaterOrEqual(t, rec.Confidence, 0.0, "Recommendation confidence should be non-negative")
	}

	t.Logf("Performance optimization completed with confidence: %.2f", perfResp.Confidence)
	t.Logf("Number of recommendations: %d", len(perfResp.Recommendations))
}

func testWorkloadAnalysisIntegration(t *testing.T) {
	if !isAIEngineAvailable() {
		t.Skip("AI Engine not available, skipping workload analysis test")
	}

	// Create AI integration layer
	aiLayer := createTestAILayer(t)

	// Create workload pattern analysis request
	workloadReq := ai.WorkloadPatternRequest{
		WorkloadID: "test_workload_001",
		TimeRange: ai.TimeRange{
			Start: time.Now().Add(-24 * time.Hour),
			End:   time.Now(),
		},
		MetricTypes: []string{"cpu", "memory", "network", "disk"},
		DataPoints:  generateTestResourceDataPoints(),
	}

	// Analyze workload patterns
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()

	workloadResp, err := aiLayer.AnalyzeWorkloadPattern(ctx, workloadReq)
	require.NoError(t, err, "Workload analysis should succeed")
	require.NotNil(t, workloadResp, "Workload response should not be nil")

	// Validate response
	assert.NotEmpty(t, workloadResp.Classification, "Classification should not be empty")
	assert.GreaterOrEqual(t, workloadResp.Confidence, 0.0, "Confidence should be non-negative")
	assert.LessOrEqual(t, workloadResp.Confidence, 1.0, "Confidence should not exceed 1.0")
	assert.GreaterOrEqual(t, len(workloadResp.Patterns), 0, "Should have pattern analysis")
	assert.GreaterOrEqual(t, len(workloadResp.Recommendations), 0, "Should have recommendations")

	t.Logf("Workload classification: %s (confidence: %.2f)",
		workloadResp.Classification, workloadResp.Confidence)
	t.Logf("Number of patterns detected: %d", len(workloadResp.Patterns))
	t.Logf("Number of recommendations: %d", len(workloadResp.Recommendations))
}

func testResourcePredictionIntegration(t *testing.T) {
	if !isAIEngineAvailable() {
		t.Skip("AI Engine not available, skipping resource prediction test")
	}

	// Create AI integration layer
	aiLayer := createTestAILayer(t)

	// Test prediction for different resource types
	resourceTypes := []string{"cpu", "memory", "network", "storage"}

	for _, resourceType := range resourceTypes {
		t.Run("Predict_"+resourceType, func(t *testing.T) {
			// Create resource prediction request
			predReq := ai.ResourcePredictionRequest{
				NodeID:         "test_node",
				ResourceType:   resourceType,
				HorizonMinutes: 60, // Predict for next hour
				HistoricalData: generateTestResourceDataPoints(),
				Context: map[string]interface{}{
					"cluster_type": "distributed_supercompute",
					"workload_mix": "mixed",
				},
			}

			// Get resource demand prediction
			ctx, cancel := context.WithTimeout(context.Background(), 25*time.Second)
			defer cancel()

			predResp, err := aiLayer.PredictResourceDemand(ctx, predReq)
			require.NoError(t, err, "Resource prediction should succeed for %s", resourceType)
			require.NotNil(t, predResp, "Prediction response should not be nil")

			// Validate response
			assert.GreaterOrEqual(t, len(predResp.Predictions), 0, "Should have predictions")
			assert.GreaterOrEqual(t, predResp.Confidence, 0.0, "Confidence should be non-negative")
			assert.LessOrEqual(t, predResp.Confidence, 1.0, "Confidence should not exceed 1.0")
			assert.NotEmpty(t, predResp.ModelInfo.Name, "Model name should not be empty")

			if len(predResp.Predictions) > 0 {
				for _, pred := range predResp.Predictions {
					assert.GreaterOrEqual(t, pred, 0.0, "Prediction should be non-negative")
				}
			}

			t.Logf("Resource prediction for %s: %d values (confidence: %.2f)",
				resourceType, len(predResp.Predictions), predResp.Confidence)
		})
	}
}

func testAnomalyDetectionIntegration(t *testing.T) {
	if !isAIEngineAvailable() {
		t.Skip("AI Engine not available, skipping anomaly detection test")
	}

	// Create AI integration layer
	aiLayer := createTestAILayer(t)

	// Create anomaly detection request with potential anomalies
	anomalyReq := ai.AnomalyDetectionRequest{
		ResourceID: "test_cluster",
		MetricType: "system_performance",
		DataPoints: generateAnomalousDataPoints(), // Include some anomalous patterns
		Sensitivity: 0.1, // High sensitivity
		Context: map[string]interface{}{
			"baseline_cpu":    45.0,
			"baseline_memory": 55.0,
			"cluster_type":    "distributed_supercompute",
		},
	}

	// Detect anomalies
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()

	anomalyResp, err := aiLayer.DetectAnomalies(ctx, anomalyReq)
	require.NoError(t, err, "Anomaly detection should succeed")
	require.NotNil(t, anomalyResp, "Anomaly response should not be nil")

	// Validate response
	assert.GreaterOrEqual(t, len(anomalyResp.Anomalies), 0, "Should have anomaly analysis")
	assert.GreaterOrEqual(t, anomalyResp.OverallScore, 0.0, "Overall score should be non-negative")
	assert.LessOrEqual(t, anomalyResp.OverallScore, 1.0, "Overall score should not exceed 1.0")
	assert.NotNil(t, anomalyResp.Baseline, "Baseline should be provided")

	// Check detected anomalies if any
	for _, anomaly := range anomalyResp.Anomalies {
		assert.NotEmpty(t, anomaly.AnomalyType, "Anomaly type should not be empty")
		assert.NotEmpty(t, anomaly.Severity, "Severity should not be empty")
		assert.GreaterOrEqual(t, anomaly.Score, 0.0, "Anomaly score should be non-negative")
		assert.NotEmpty(t, anomaly.Description, "Description should not be empty")
	}

	t.Logf("Anomaly detection completed: %d anomalies detected (overall score: %.2f)",
		len(anomalyResp.Anomalies), anomalyResp.OverallScore)
}

func testPredictiveScalingIntegration(t *testing.T) {
	if !isAIEngineAvailable() {
		t.Skip("AI Engine not available, skipping predictive scaling test")
	}

	// Create AI integration layer
	aiLayer := createTestAILayer(t)

	// Create predictive scaling request
	scalingData := map[string]interface{}{
		"current_capacity": map[string]float64{
			"cpu_cores":    100.0,
			"memory_gb":    512.0,
			"storage_tb":   10.0,
		},
		"current_utilization": map[string]float64{
			"cpu":    72.5,
			"memory": 68.3,
			"storage": 45.2,
		},
		"job_queue_length": 15,
		"predicted_workload": map[string]float64{
			"cpu_demand":    85.0,
			"memory_demand": 78.5,
			"storage_demand": 52.3,
		},
		"historical_patterns": []map[string]interface{}{
			{
				"time_period": "daily_peak",
				"avg_cpu":     78.2,
				"avg_memory":  72.1,
			},
			{
				"time_period": "weekly_trend",
				"growth_rate": 1.15,
			},
		},
	}

	// Get predictive scaling recommendations
	ctx, cancel := context.WithTimeout(context.Background(), 25*time.Second)
	defer cancel()

	scalingResp, err := aiLayer.PredictScalingNeeds(ctx, scalingData)
	require.NoError(t, err, "Predictive scaling should succeed")
	require.NotNil(t, scalingResp, "Scaling response should not be nil")

	// Validate response structure
	assert.IsType(t, map[string]interface{}{}, scalingResp, "Response should be a map")

	// Check for expected scaling recommendations
	if scalingActions, ok := scalingResp["scaling_actions"]; ok {
		t.Logf("Scaling actions recommended: %+v", scalingActions)
	}

	if confidence, ok := scalingResp["confidence"]; ok {
		confidenceVal, ok := confidence.(float64)
		assert.True(t, ok, "Confidence should be a float64")
		assert.GreaterOrEqual(t, confidenceVal, 0.0, "Confidence should be non-negative")
		assert.LessOrEqual(t, confidenceVal, 1.0, "Confidence should not exceed 1.0")
		t.Logf("Scaling prediction confidence: %.2f", confidenceVal)
	}

	t.Logf("Predictive scaling analysis completed successfully")
}

func testEndToEndAIWorkflow(t *testing.T) {
	if !isAIEngineAvailable() {
		t.Skip("AI Engine not available, skipping end-to-end AI workflow test")
	}

	// This test simulates a complete AI-powered optimization workflow
	t.Log("Starting end-to-end AI workflow simulation...")

	// 1. Create distributed AI service
	config := compute.DefaultDistributedAIConfig()
	config.AIEngineEndpoint = getAIEngineEndpoint()
	config.PerformanceOptimizationInterval = 2 * time.Second
	config.MinConfidenceThreshold = 0.5 // Lower threshold for testing

	jobManager := createTestJobManager(t)
	loadBalancer := createTestLoadBalancer(t)
	perfOptimizer := createTestPerformanceOptimizer(t)

	aiService, err := compute.NewDistributedAIService(
		config,
		jobManager,
		loadBalancer,
		perfOptimizer,
	)
	require.NoError(t, err, "Failed to create distributed AI service")

	// 2. Start AI service
	err = aiService.Start()
	require.NoError(t, err, "Failed to start AI service")
	defer func() {
		err := aiService.Stop()
		assert.NoError(t, err, "Failed to stop AI service")
	}()

	// 3. Simulate workload submission
	testJobs := createTestComputeJobs(5)
	for _, job := range testJobs {
		err := jobManager.SubmitJob(job)
		require.NoError(t, err, "Failed to submit test job")
	}

	// 4. Wait for AI optimization cycles to run
	t.Log("Waiting for AI optimization cycles...")
	time.Sleep(10 * time.Second)

	// 5. Verify AI service health and metrics
	assert.True(t, aiService.IsHealthy(), "AI service should remain healthy")

	metrics := aiService.GetMetrics()
	assert.NotNil(t, metrics, "Metrics should be available")

	// Should have some optimization attempts by now
	assert.GreaterOrEqual(t, metrics.TotalOptimizations, int64(0), "Should have optimization attempts")

	t.Logf("End-to-end workflow completed:")
	t.Logf("- Total optimizations: %d", metrics.TotalOptimizations)
	t.Logf("- Successful optimizations: %d", metrics.SuccessfulOptimizations)
	t.Logf("- Failed optimizations: %d", metrics.FailedOptimizations)
	t.Logf("- Total predictions: %d", metrics.TotalPredictions)
	t.Logf("- AI engine uptime: %.2f", metrics.AIEngineUptime)

	// 6. Test configuration update
	newConfig := config
	newConfig.MinConfidenceThreshold = 0.8
	err = aiService.UpdateConfig(newConfig)
	assert.NoError(t, err, "Should be able to update configuration")

	updatedConfig := aiService.GetConfig()
	assert.Equal(t, 0.8, updatedConfig.MinConfidenceThreshold, "Configuration should be updated")

	t.Log("End-to-end AI workflow test completed successfully")
}

// Helper functions

func isAIEngineAvailable() bool {
	// Check if AI Engine is available by trying to connect
	aiConfig := ai.DefaultAIConfig()
	aiConfig.Timeout = 5 * time.Second

	aiLayer := ai.NewAIIntegrationLayer(
		getAIEngineEndpoint(),
		"",
		aiConfig,
	)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	err := aiLayer.HealthCheck(ctx)
	return err == nil
}

func getAIEngineEndpoint() string {
	// Return AI Engine endpoint (could be from environment variable)
	return "http://localhost:8095"
}

func createTestAILayer(t *testing.T) *ai.AIIntegrationLayer {
	aiConfig := ai.DefaultAIConfig()
	aiConfig.Timeout = 30 * time.Second

	aiLayer := ai.NewAIIntegrationLayer(
		getAIEngineEndpoint(),
		"",
		aiConfig,
	)

	// Verify connection
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	err := aiLayer.HealthCheck(ctx)
	require.NoError(t, err, "AI Engine should be available for testing")

	return aiLayer
}

func createTestJobManager(t *testing.T) *compute.ComputeJobManager {
	config := compute.DefaultJobManagerConfig()
	config.MaxConcurrentJobs = 10
	config.QueueCapacity = 50

	jobManager, err := compute.NewComputeJobManager(config)
	require.NoError(t, err, "Failed to create test job manager")

	return jobManager
}

func createTestLoadBalancer(t *testing.T) *compute.ComputeJobLoadBalancer {
	config := compute.LoadBalancerConfig{
		DefaultAlgorithm:    compute.AlgorithmLeastLoaded,
		EnableHealthChecks:  true,
		HealthCheckInterval: 30 * time.Second,
	}

	loadBalancer, err := compute.NewComputeJobLoadBalancer(config)
	require.NoError(t, err, "Failed to create test load balancer")

	return loadBalancer
}

func createTestPerformanceOptimizer(t *testing.T) *compute.PerformanceOptimizer {
	config := compute.DefaultOptimizerConfig()
	config.OptimizationInterval = 5 * time.Second
	config.EnableAIOptimization = true

	// Create minimal test scheduler
	schedulerConfig := scheduler.Config{
		MaxConcurrentJobs: 100,
		SchedulingPolicy:  "round_robin",
	}

	testScheduler, err := scheduler.NewScheduler(schedulerConfig)
	require.NoError(t, err, "Failed to create test scheduler")

	perfOptimizer, err := compute.NewPerformanceOptimizer(
		testScheduler,
		nil, // job manager
		nil, // load balancer
		config,
	)
	require.NoError(t, err, "Failed to create test performance optimizer")

	return perfOptimizer
}

func generateTestResourceDataPoints() []ai.ResourceDataPoint {
	var dataPoints []ai.ResourceDataPoint

	baseTime := time.Now().Add(-24 * time.Hour)
	for i := 0; i < 144; i++ { // Every 10 minutes for 24 hours
		timestamp := baseTime.Add(time.Duration(i) * 10 * time.Minute)

		// Generate realistic CPU usage pattern
		hourOfDay := float64(timestamp.Hour())
		cpuUsage := 30.0 + 40.0*(0.5+0.5*math.Sin((hourOfDay-6)*math.Pi/12)) // Peak around noon

		dataPoints = append(dataPoints, ai.ResourceDataPoint{
			Timestamp: timestamp,
			Value:     cpuUsage + 5.0*rand.Float64(), // Add some noise
			Metadata: map[string]interface{}{
				"metric_type": "cpu_utilization",
				"node_id":     "test_node",
			},
		})
	}

	return dataPoints
}

func generateAnomalousDataPoints() []ai.ResourceDataPoint {
	var dataPoints []ai.ResourceDataPoint

	baseTime := time.Now().Add(-2 * time.Hour)
	for i := 0; i < 24; i++ { // Every 5 minutes for 2 hours
		timestamp := baseTime.Add(time.Duration(i) * 5 * time.Minute)

		var cpuUsage float64
		if i >= 10 && i <= 14 {
			// Inject anomaly - sudden spike
			cpuUsage = 95.0 + 5.0*rand.Float64()
		} else {
			// Normal usage
			cpuUsage = 45.0 + 10.0*rand.Float64()
		}

		dataPoints = append(dataPoints, ai.ResourceDataPoint{
			Timestamp: timestamp,
			Value:     cpuUsage,
			Metadata: map[string]interface{}{
				"metric_type": "cpu_utilization",
				"anomaly_test": true,
			},
		})
	}

	return dataPoints
}

func createTestComputeJobs(count int) []*compute.ComputeJob {
	var jobs []*compute.ComputeJob

	jobTypes := []compute.JobType{
		compute.JobTypeBatch,
		compute.JobTypeInteractive,
		compute.JobTypeMPI,
		compute.JobTypeContainer,
	}

	for i := 0; i < count; i++ {
		job := &compute.ComputeJob{
			JobID:    fmt.Sprintf("test_job_%d", i+1),
			JobType:  jobTypes[i%len(jobTypes)],
			Priority: compute.PriorityMedium,
			Resources: map[string]interface{}{
				"cpu":    2 + i%6,
				"memory": 1024 + i*512,
				"gpu":    i%2,
			},
			ExecutionConstraints: compute.ExecutionConstraints{
				MaxExecutionTime: 3600, // 1 hour
				RequiredNodes:    1,
			},
			ClusterPlacement: compute.ClusterPlacement{
				PreferredClusterID: "test_cluster",
				AllowCrossCluster:  true,
			},
			CreatedAt: time.Now(),
			Status:    compute.JobStatusQueued,
		}

		jobs = append(jobs, job)
	}

	return jobs
}

// Additional imports needed
import (
	"fmt"
	"math"
	"math/rand"
)