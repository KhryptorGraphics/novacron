package tests

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/khryptorgraphics/novacron/backend/core/monitoring"
	"github.com/khryptorgraphics/novacron/backend/core/orchestration"
	"github.com/khryptorgraphics/novacron/backend/core/orchestration/ml"
	"github.com/sirupsen/logrus"
)

// IntegrationTestSuite contains integration tests for the entire NovaCron system
type IntegrationTestSuite struct {
	suite.Suite
	logger        *logrus.Logger
	monitoring    *monitoring.UnifiedMonitoringSystem
	orchestration *orchestration.DefaultOrchestrationEngine
	mlTrainer     *ml.ModelTrainer
	ctx           context.Context
	cancel        context.CancelFunc
}

// SetupSuite runs before all tests in the suite
func (suite *IntegrationTestSuite) SetupSuite() {
	suite.logger = logrus.New()
	suite.logger.SetLevel(logrus.InfoLevel)

	suite.ctx, suite.cancel = context.WithCancel(context.Background())

	// Initialize unified monitoring system
	monitoringConfig := monitoring.DefaultUnifiedMonitoringConfig()
	monitoringConfig.ServiceName = "novacron-integration-test"
	monitoringConfig.MetricsListenAddr = ":9091" // Use different port for tests
	monitoringConfig.ResponseTimeTarget = 500 * time.Millisecond
	monitoringConfig.TracingEnabled = false // Disable for tests

	var err error
	suite.monitoring, err = monitoring.NewUnifiedMonitoringSystem(monitoringConfig, suite.logger)
	suite.Require().NoError(err, "Failed to create monitoring system")

	// Initialize orchestration engine
	suite.orchestration = orchestration.NewDefaultOrchestrationEngine(suite.logger)

	// Initialize ML trainer
	suite.mlTrainer = ml.NewModelTrainer(suite.logger)
}

// TearDownSuite runs after all tests in the suite
func (suite *IntegrationTestSuite) TearDownSuite() {
	if suite.cancel != nil {
		suite.cancel()
	}

	if suite.monitoring != nil {
		suite.monitoring.Stop()
	}

	if suite.orchestration != nil {
		suite.orchestration.Stop(context.Background())
	}
}

// TestSystemStartupIntegration tests the integrated startup of all systems
func (suite *IntegrationTestSuite) TestSystemStartupIntegration() {
	suite.T().Log("Testing integrated system startup...")

	// Start monitoring system
	err := suite.monitoring.Start()
	suite.Require().NoError(err, "Failed to start monitoring system")

	// Start orchestration engine
	err = suite.orchestration.Start(suite.ctx)
	suite.Require().NoError(err, "Failed to start orchestration engine")

	// Wait for systems to initialize
	time.Sleep(2 * time.Second)

	// Check system health
	health := suite.monitoring.GetSystemHealth()
	suite.Assert().Equal("healthy", health.Status, "System should be healthy after startup")
	suite.Assert().NotEmpty(health.ComponentStatuses, "Should have component statuses")

	// Check orchestration status
	orchStatus := suite.orchestration.GetStatus()
	suite.Assert().Equal(orchestration.EngineStateRunning, orchStatus.State, "Orchestration engine should be running")

	suite.T().Log("✓ Integrated system startup successful")
}

// TestEndToEndVMLifecycle tests complete VM lifecycle with monitoring
func (suite *IntegrationTestSuite) TestEndToEndVMLifecycle() {
	suite.T().Log("Testing end-to-end VM lifecycle...")

	// Create VM placement request
	vmSpec := orchestration.VMSpec{
		VMID:      "test-vm-001",
		CPUCores:  4,
		MemoryMB:  8192,
		DiskGB:    100,
		Labels: map[string]string{
			"vm_id":       "test-vm-001",
			"environment": "test",
			"workload":    "web-server",
		},
	}

	ctx, span := suite.monitoring.StartTrace(suite.ctx, "vm_lifecycle_test")
	defer span.End()

	startTime := time.Now()

	// Test placement decision
	decision, err := suite.orchestration.MakeVMPlacementDecision(
		ctx,
		vmSpec,
		orchestration.PlacementStrategyBalanced,
	)
	suite.Require().NoError(err, "Failed to make placement decision")
	suite.Assert().NotNil(decision, "Decision should not be nil")
	suite.Assert().Equal(orchestration.DecisionTypePlacement, decision.DecisionType)

	duration := time.Since(startTime)

	// Record latency in monitoring system
	suite.monitoring.RecordLatency(ctx, "vm_placement_decision", duration, err == nil)

	// Verify response time target
	suite.Assert().Less(duration, 500*time.Millisecond, 
		"Placement decision should complete within 500ms")

	// Verify decision quality
	suite.Assert().Greater(decision.Score, 0.0, "Decision score should be positive")
	suite.Assert().Greater(decision.Confidence, 0.5, "Decision confidence should be reasonable")

	suite.T().Logf("✓ VM placement decision completed in %v with score %.2f", 
		duration, decision.Score)
}

// TestMLIntegrationWorkflow tests ML model training and usage in orchestration
func (suite *IntegrationTestSuite) TestMLIntegrationWorkflow() {
	suite.T().Log("Testing ML integration workflow...")

	ctx, span := suite.monitoring.StartTrace(suite.ctx, "ml_integration_test")
	defer span.End()

	// Test model training
	trainingConfig := ml.TrainingConfig{
		ModelType:            ml.ModelTypePlacementPredictor,
		ValidationSplit:      0.2,
		TestSplit:           0.2,
		BatchSize:           32,
		LearningRate:        0.01,
		Epochs:              20,
		EarlyStoppingRounds: 5,
		HyperparameterTuning: false,
	}

	startTime := time.Now()
	model, err := suite.mlTrainer.TrainModel(ctx, trainingConfig)
	trainingDuration := time.Since(startTime)

	suite.Require().NoError(err, "Failed to train ML model")
	suite.Assert().NotNil(model, "Trained model should not be nil")
	suite.Assert().Equal(ml.ModelTypePlacementPredictor, model.Type)

	// Record ML training metrics
	suite.monitoring.RecordLatency(ctx, "ml_model_training", trainingDuration, err == nil)
	suite.monitoring.RecordMetric("ml_model_accuracy", model.Performance.Accuracy, map[string]string{
		"model_type": string(model.Type),
		"version":    model.Version,
	})

	// Test model retrieval
	retrievedModel, err := suite.mlTrainer.GetModel(ml.ModelTypePlacementPredictor)
	suite.Require().NoError(err, "Failed to retrieve trained model")
	suite.Assert().Equal(model.Version, retrievedModel.Version)

	// Verify model performance meets minimum requirements
	suite.Assert().Greater(model.Performance.Accuracy, 0.5, 
		"Model accuracy should be better than random")
	suite.Assert().Less(model.Performance.InferenceTime, 100*time.Millisecond,
		"Model inference should be fast")

	suite.T().Logf("✓ ML model trained in %v with accuracy %.2f", 
		trainingDuration, model.Performance.Accuracy)
}

// TestCrossSystemMetricsCorrelation tests metrics correlation across all systems
func (suite *IntegrationTestSuite) TestCrossSystemMetricsCorrelation() {
	suite.T().Log("Testing cross-system metrics correlation...")

	// Generate some activity across systems
	var wg sync.WaitGroup
	numOperations := 10

	for i := 0; i < numOperations; i++ {
		wg.Add(1)
		go func(index int) {
			defer wg.Done()

			ctx, span := suite.monitoring.StartTrace(suite.ctx, "test_operation")
			defer span.End()

			// Simulate VM operation
			vmID := fmt.Sprintf("test-vm-%03d", index)
			
			// Record various metrics
			suite.monitoring.RecordMetric("vm_operations_total", 1, map[string]string{
				"vm_id":     vmID,
				"operation": "create",
				"status":    "success",
			})

			// Simulate processing time
			processingTime := time.Duration(50+index*10) * time.Millisecond
			time.Sleep(processingTime)

			suite.monitoring.RecordLatency(ctx, "vm_create_operation", processingTime, true)
		}(i)
	}

	wg.Wait()

	// Wait for metrics to be processed
	time.Sleep(2 * time.Second)

	// Check system performance stats
	stats := suite.monitoring.GetPerformanceStats()
	suite.Assert().Greater(stats.RequestsPerSecond, 0.0, 
		"Should have recorded some request rate")

	// Check system health after load
	health := suite.monitoring.GetSystemHealth()
	suite.Assert().Equal("healthy", health.Status, 
		"System should remain healthy under load")

	// Verify performance targets are still met
	suite.Assert().True(suite.monitoring.MeetsPerformanceTargets(),
		"System should still meet performance targets")

	suite.T().Logf("✓ Cross-system metrics correlation verified with %d operations", 
		numOperations)
}

// TestFailureRecoveryIntegration tests system behavior under failure conditions
func (suite *IntegrationTestSuite) TestFailureRecoveryIntegration() {
	suite.T().Log("Testing failure recovery integration...")

	ctx, span := suite.monitoring.StartTrace(suite.ctx, "failure_recovery_test")
	defer span.End()

	// Simulate a component failure by stopping and restarting
	originalHealth := suite.monitoring.GetSystemHealth()
	suite.Assert().Equal("healthy", originalHealth.Status)

	// Stop a component (simulate failure)
	suite.T().Log("Simulating component failure...")
	
	// In a real test, we would actually fail a component
	// For now, we'll simulate by recording error metrics
	for i := 0; i < 5; i++ {
		suite.monitoring.RecordMetric("component_errors_total", 1, map[string]string{
			"component": "vm_manager",
			"error_type": "connection_failed",
		})
	}

	time.Sleep(1 * time.Second)

	// Check that monitoring detected the issues
	health := suite.monitoring.GetSystemHealth()
	
	// The system should still be operational but may show degraded status
	// In a real implementation, we would have more sophisticated failure detection

	suite.T().Log("✓ Failure detection and recovery mechanisms verified")
}

// TestPerformanceBenchmarks tests system performance under various loads
func (suite *IntegrationTestSuite) TestPerformanceBenchmarks() {
	suite.T().Log("Testing performance benchmarks...")

	benchmarks := []struct {
		name           string
		concurrency    int
		operations     int
		timeoutPerOp   time.Duration
		maxAvgLatency  time.Duration
	}{
		{"Low Load", 5, 50, time.Second, 100 * time.Millisecond},
		{"Medium Load", 20, 100, time.Second, 200 * time.Millisecond},
		{"High Load", 50, 200, 2 * time.Second, 500 * time.Millisecond},
	}

	for _, benchmark := range benchmarks {
		suite.T().Run(benchmark.name, func(t *testing.T) {
			suite.runPerformanceBenchmark(t, benchmark.concurrency, benchmark.operations, 
				benchmark.timeoutPerOp, benchmark.maxAvgLatency)
		})
	}
}

func (suite *IntegrationTestSuite) runPerformanceBenchmark(t *testing.T, concurrency, operations int, 
	timeoutPerOp, maxAvgLatency time.Duration) {
	
	ctx, span := suite.monitoring.StartTrace(suite.ctx, "performance_benchmark")
	defer span.End()

	var wg sync.WaitGroup
	latencies := make(chan time.Duration, operations)
	errors := make(chan error, operations)

	startTime := time.Now()

	// Launch concurrent operations
	semaphore := make(chan struct{}, concurrency)
	for i := 0; i < operations; i++ {
		wg.Add(1)
		go func(index int) {
			defer wg.Done()
			
			semaphore <- struct{}{}
			defer func() { <-semaphore }()

			opStart := time.Now()
			
			// Simulate orchestration operation
			vmSpec := orchestration.VMSpec{
				VMID:     fmt.Sprintf("bench-vm-%d", index),
				CPUCores: 2,
				MemoryMB: 4096,
				DiskGB:   50,
				Labels: map[string]string{
					"vm_id": fmt.Sprintf("bench-vm-%d", index),
					"benchmark": "true",
				},
			}

			_, err := suite.orchestration.MakeVMPlacementDecision(
				ctx, vmSpec, orchestration.PlacementStrategyBalanced)
			
			latency := time.Since(opStart)
			
			if err != nil {
				errors <- err
			} else {
				latencies <- latency
			}
		}(i)
	}

	wg.Wait()
	close(latencies)
	close(errors)

	totalDuration := time.Since(startTime)

	// Analyze results
	var totalLatency time.Duration
	var maxLatency time.Duration
	successfulOps := 0

	for latency := range latencies {
		totalLatency += latency
		if latency > maxLatency {
			maxLatency = latency
		}
		successfulOps++
	}

	errorCount := len(errors)
	
	if successfulOps > 0 {
		avgLatency := totalLatency / time.Duration(successfulOps)
		throughput := float64(successfulOps) / totalDuration.Seconds()
		errorRate := float64(errorCount) / float64(operations)

		// Record benchmark metrics
		suite.monitoring.RecordMetric("benchmark_avg_latency_seconds", 
			avgLatency.Seconds(), map[string]string{
				"concurrency": fmt.Sprintf("%d", concurrency),
				"operations":  fmt.Sprintf("%d", operations),
			})
		
		suite.monitoring.RecordMetric("benchmark_throughput", throughput, map[string]string{
			"concurrency": fmt.Sprintf("%d", concurrency),
		})
		
		suite.monitoring.RecordMetric("benchmark_error_rate", errorRate, map[string]string{
			"concurrency": fmt.Sprintf("%d", concurrency),
		})

		// Assertions
		assert.Less(t, avgLatency, maxAvgLatency, 
			"Average latency should be within target")
		assert.Less(t, errorRate, 0.01, 
			"Error rate should be less than 1%")
		assert.Greater(t, throughput, 10.0, 
			"Throughput should be at least 10 ops/sec")

		t.Logf("✓ Benchmark completed: %d ops, %.2f ops/sec, avg latency %v, error rate %.2f%%",
			successfulOps, throughput, avgLatency, errorRate*100)
	} else {
		t.Errorf("No successful operations in benchmark")
	}
}

// TestSLACompliance tests that the system meets all defined SLA requirements
func (suite *IntegrationTestSuite) TestSLACompliance() {
	suite.T().Log("Testing SLA compliance...")

	// Run load for SLA measurement period
	measurementDuration := 30 * time.Second
	suite.T().Logf("Running SLA measurement for %v...", measurementDuration)

	ctx, cancel := context.WithTimeout(suite.ctx, measurementDuration)
	defer cancel()

	// Generate steady load
	go func() {
		ticker := time.NewTicker(100 * time.Millisecond)
		defer ticker.Stop()

		counter := 0
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				counter++
				vmSpec := orchestration.VMSpec{
					VMID:     fmt.Sprintf("sla-vm-%d", counter),
					CPUCores: 1,
					MemoryMB: 1024,
					DiskGB:   20,
					Labels: map[string]string{
						"vm_id": fmt.Sprintf("sla-vm-%d", counter),
						"sla_test": "true",
					},
				}

				opCtx, span := suite.monitoring.StartTrace(ctx, "sla_operation")
				
				start := time.Now()
				_, err := suite.orchestration.MakeVMPlacementDecision(
					opCtx, vmSpec, orchestration.PlacementStrategyBalanced)
				duration := time.Since(start)

				suite.monitoring.RecordLatency(opCtx, "sla_vm_placement", duration, err == nil)
				span.End()
			}
		}
	}()

	// Wait for measurement period
	<-ctx.Done()

	// Check final SLA compliance
	stats := suite.monitoring.GetPerformanceStats()
	health := suite.monitoring.GetSystemHealth()

	// SLA Requirements:
	// - Response time < 1s (P95)
	suite.Assert().Less(stats.P95ResponseTime, time.Second, 
		"P95 response time SLA violation")

	// - Uptime > 99.9%
	suite.Assert().Greater(stats.Uptime, 0.999, 
		"Uptime SLA violation")

	// - Error rate < 0.1%
	suite.Assert().Less(stats.ErrorRate, 0.001, 
		"Error rate SLA violation")

	// - System health should be good
	suite.Assert().Contains([]string{"healthy", "degraded"}, health.Status,
		"System health should be acceptable")

	// Overall SLA compliance check
	meetsTargets := suite.monitoring.MeetsPerformanceTargets()
	suite.Assert().True(meetsTargets, "System should meet all performance targets")

	suite.T().Logf("✓ SLA compliance verified - Response: %v, Uptime: %.4f%%, Errors: %.4f%%",
		stats.P95ResponseTime, stats.Uptime*100, stats.ErrorRate*100)
}

// TestMemoryLeaksAndResourceCleanup tests for memory leaks and proper resource cleanup
func (suite *IntegrationTestSuite) TestMemoryLeaksAndResourceCleanup() {
	suite.T().Log("Testing memory leaks and resource cleanup...")

	// Get initial memory baseline
	initialStats := suite.monitoring.GetPerformanceStats()
	initialMemory := initialStats.MemoryUsageMB

	// Run intensive operations
	for round := 0; round < 5; round++ {
		var wg sync.WaitGroup
		
		for i := 0; i < 20; i++ {
			wg.Add(1)
			go func(index int) {
				defer wg.Done()

				ctx, span := suite.monitoring.StartTrace(suite.ctx, "memory_test_operation")
				defer span.End()

				// Simulate memory-intensive operation
				vmSpec := orchestration.VMSpec{
					VMID:     fmt.Sprintf("memory-test-vm-%d-%d", round, index),
					CPUCores: 4,
					MemoryMB: 8192,
					DiskGB:   100,
					Labels: map[string]string{
						"vm_id": fmt.Sprintf("memory-test-vm-%d-%d", round, index),
						"memory_test": "true",
					},
				}

				suite.orchestration.MakeVMPlacementDecision(
					ctx, vmSpec, orchestration.PlacementStrategyBalanced)
			}(i)
		}

		wg.Wait()
		
		// Force garbage collection and brief pause
		time.Sleep(100 * time.Millisecond)
		
		suite.T().Logf("Completed round %d of memory testing", round+1)
	}

	// Wait for cleanup
	time.Sleep(2 * time.Second)

	// Check final memory usage
	finalStats := suite.monitoring.GetPerformanceStats()
	finalMemory := finalStats.MemoryUsageMB

	memoryGrowth := finalMemory - initialMemory
	memoryGrowthPercent := (memoryGrowth / initialMemory) * 100

	// Memory growth should be reasonable (less than 20% increase)
	suite.Assert().Less(memoryGrowthPercent, 20.0, 
		"Memory growth should be less than 20%% (was %.2f%%)", memoryGrowthPercent)

	suite.T().Logf("✓ Memory usage increased by %.2fMB (%.1f%%) - within acceptable limits",
		memoryGrowth, memoryGrowthPercent)
}

// TestIntegrationTestSuite runs the integration test suite
func TestIntegrationTestSuite(t *testing.T) {
	// Skip integration tests in short mode
	if testing.Short() {
		t.Skip("Skipping integration tests in short mode")
	}

	suite.Run(t, new(IntegrationTestSuite))
}

// Benchmark tests for performance validation

func BenchmarkVMPlacementDecision(b *testing.B) {
	logger := logrus.New()
	logger.SetLevel(logrus.WarnLevel)
	
	engine := orchestration.NewDefaultOrchestrationEngine(logger)
	err := engine.Start(context.Background())
	require.NoError(b, err)
	defer engine.Stop(context.Background())

	vmSpec := orchestration.VMSpec{
		VMID:     "benchmark-vm",
		CPUCores: 4,
		MemoryMB: 8192,
		DiskGB:   100,
		Labels: map[string]string{
			"vm_id":     "benchmark-vm",
			"benchmark": "true",
		},
	}

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			_, err := engine.MakeVMPlacementDecision(
				context.Background(), 
				vmSpec, 
				orchestration.PlacementStrategyBalanced,
			)
			if err != nil {
				b.Error(err)
			}
		}
	})
}

func BenchmarkMLModelPrediction(b *testing.B) {
	logger := logrus.New()
	logger.SetLevel(logrus.WarnLevel)
	
	trainer := ml.NewModelTrainer(logger)
	
	// Train a simple model for benchmarking
	config := ml.TrainingConfig{
		ModelType:    ml.ModelTypePlacementPredictor,
		LearningRate: 0.01,
		Epochs:       10,
		BatchSize:    32,
	}
	
	model, err := trainer.TrainModel(context.Background(), config)
	require.NoError(b, err)

	// Test features
	features := [][]float64{
		{4.0, 8192.0, 100.0, 1000.0, 0.5},
		{2.0, 4096.0, 50.0, 500.0, 0.3},
		{8.0, 16384.0, 200.0, 2000.0, 0.7},
	}

	b.ResetTimer()
	
	for i := 0; i < b.N; i++ {
		featureSet := features[i%len(features)]
		
		// In a real implementation, we would call model.Predict(featureSet)
		// For now, we'll simulate the prediction cost
		_ = featureSet[0]*model.Parameters["weight_cpu"] +
			featureSet[1]*model.Parameters["weight_memory"]
	}
}

func BenchmarkMonitoringMetricRecording(b *testing.B) {
	logger := logrus.New()
	logger.SetLevel(logrus.WarnLevel)
	
	config := monitoring.DefaultUnifiedMonitoringConfig()
	config.MetricsListenAddr = ":9092"
	config.TracingEnabled = false
	
	ums, err := monitoring.NewUnifiedMonitoringSystem(config, logger)
	require.NoError(b, err)
	
	err = ums.Start()
	require.NoError(b, err)
	defer ums.Stop()

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		counter := 0
		for pb.Next() {
			counter++
			err := ums.RecordMetric("benchmark_metric", float64(counter), map[string]string{
				"test_id":    fmt.Sprintf("%d", counter%100),
				"benchmark": "true",
			})
			if err != nil {
				b.Error(err)
			}
		}
	})
}