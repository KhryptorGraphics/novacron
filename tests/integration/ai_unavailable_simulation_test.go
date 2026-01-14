package integration

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/migration"
	"github.com/khryptorgraphics/novacron/backend/core/scheduler"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestRealWorldAIUnavailableScenarios tests realistic scenarios where AI services fail
func TestRealWorldAIUnavailableScenarios(t *testing.T) {
	t.Run("AIServiceDownScenario", func(t *testing.T) {
		// Simulate AI service that's completely down (connection refused)
		config := scheduler.DefaultSchedulerConfig()
		aiConfig := scheduler.DefaultAISchedulerConfig()
		aiConfig.Enabled = true
		aiConfig.AIEngineURL = "http://localhost:9999" // Non-existent service
		aiConfig.RequestTimeout = 2 * time.Second
		aiConfig.MaxRetries = 1

		sched := scheduler.NewSchedulerWithAI(config, aiConfig)
		err := sched.Start()
		require.NoError(t, err)
		defer sched.Stop()

		// Register nodes
		resources := map[scheduler.ResourceType]*scheduler.Resource{
			scheduler.ResourceCPU: {
				Type:     scheduler.ResourceCPU,
				Capacity: 100,
				Used:     30,
			},
		}
		err = sched.RegisterNode("node1", resources)
		require.NoError(t, err)

		// Try to get AI recommendations (should fallback)
		constraints := []scheduler.ResourceConstraint{
			{Type: scheduler.ResourceCPU, MinAmount: 10},
		}
		recommendations, err := sched.GetAISchedulingRecommendations(constraints)

		// Should not fail, should fallback gracefully
		assert.NoError(t, err)
		assert.NotNil(t, recommendations)

		// Verify AI metrics show failures
		aiMetrics := sched.GetAIMetrics()
		if aiProvider, ok := sched.GetAIProvider().(*scheduler.SafeAIProvider); ok {
			fallbackMetrics := aiProvider.GetMetrics()
			assert.Greater(t, fallbackMetrics["fallback_calls"].(int64), int64(0))
			assert.Greater(t, fallbackMetrics["ai_failures"].(int64), int64(0))
		}
	})

	t.Run("AIServiceSlowResponseScenario", func(t *testing.T) {
		// Create a mock AI service that responds very slowly
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			time.Sleep(10 * time.Second) // Simulate very slow response
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(`{"prediction": [0.5], "confidence": 0.8}`))
		}))
		defer server.Close()

		config := scheduler.DefaultSchedulerConfig()
		aiConfig := scheduler.DefaultAISchedulerConfig()
		aiConfig.Enabled = true
		aiConfig.AIEngineURL = server.URL
		aiConfig.RequestTimeout = 3 * time.Second // Short timeout
		aiConfig.MaxRetries = 1

		sched := scheduler.NewSchedulerWithAI(config, aiConfig)
		err := sched.Start()
		require.NoError(t, err)
		defer sched.Stop()

		// Register nodes
		resources := map[scheduler.ResourceType]*scheduler.Resource{
			scheduler.ResourceCPU: {Type: scheduler.ResourceCPU, Capacity: 100, Used: 20},
		}
		err = sched.RegisterNode("node1", resources)
		require.NoError(t, err)

		start := time.Now()
		constraints := []scheduler.ResourceConstraint{
			{Type: scheduler.ResourceCPU, MinAmount: 10},
		}
		recommendations, err := sched.GetAISchedulingRecommendations(constraints)
		elapsed := time.Since(start)

		// Should fallback quickly due to timeout
		assert.NoError(t, err)
		assert.Less(t, elapsed, 6*time.Second) // Should not wait full 10 seconds
		assert.NotNil(t, recommendations)
	})

	t.Run("AIServiceIntermittentFailuresScenario", func(t *testing.T) {
		// Create a mock AI service that fails intermittently
		callCount := 0
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			callCount++
			if callCount%2 == 1 {
				// Odd calls fail
				w.WriteHeader(http.StatusInternalServerError)
				return
			}
			// Even calls succeed
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(`{"prediction": [0.7], "confidence": 0.9}`))
		}))
		defer server.Close()

		config := scheduler.DefaultSchedulerConfig()
		aiConfig := scheduler.DefaultAISchedulerConfig()
		aiConfig.Enabled = true
		aiConfig.AIEngineURL = server.URL
		aiConfig.RequestTimeout = 5 * time.Second
		aiConfig.MaxRetries = 1

		sched := scheduler.NewSchedulerWithAI(config, aiConfig)

		// Make multiple requests - should handle intermittent failures gracefully
		for i := 0; i < 5; i++ {
			if safeProvider, ok := sched.GetAIProvider().(*scheduler.SafeAIProvider); ok {
				predictions, confidence, err := safeProvider.PredictResourceDemand(
					"node1", scheduler.ResourceCPU, 60)

				// Should never fail due to fallback
				assert.NoError(t, err)
				assert.NotEmpty(t, predictions)
				assert.Greater(t, confidence, 0.0)
			}
		}

		// Verify some AI calls succeeded and some failed
		if safeProvider, ok := sched.GetAIProvider().(*scheduler.SafeAIProvider); ok {
			metrics := safeProvider.GetMetrics()
			assert.Greater(t, metrics["total_calls"].(int64), int64(0))
			// Should have some AI failures due to intermittent issues
			assert.Greater(t, metrics["ai_failures"].(int64), int64(0))
		}
	})
}

// TestMigrationWithAIFailures tests migration scenarios with AI service failures
func TestMigrationWithAIFailures(t *testing.T) {
	t.Run("MigrationAIServiceUnavailable", func(t *testing.T) {
		config := migration.MigrationConfig{
			MaxDowntime:        30 * time.Second,
			TargetTransferRate: 100 * 1024 * 1024,
			MemoryIterations:   3,
		}

		aiConfig := &migration.AIConfig{
			Enabled:  true,
			Endpoint: "http://localhost:9998", // Non-existent service
			Timeout:  2 * time.Second,
		}

		// Create orchestrator with failing AI service
		orchestrator, err := migration.NewLiveMigrationOrchestratorWithAI(config, aiConfig)
		require.NoError(t, err)
		defer orchestrator.Close()

		// Migration should still work with fallback
		ctx := context.Background()
		migrationID, err := orchestrator.MigrateVM(ctx, "vm1", "node1", "node2",
			migration.MigrationOptions{Priority: 5})
		assert.NoError(t, err)
		assert.NotEmpty(t, migrationID)

		// Check that AI metrics show fallback usage
		aiMetrics := orchestrator.GetAIMetrics()
		if aiMetrics["ai_enabled"].(bool) {
			// If we can access the safe provider, check its metrics
			// In a real implementation, we'd expose this through the orchestrator
		}
	})

	t.Run("MigrationWithPartialAIFailure", func(t *testing.T) {
		// Create AI service that fails only certain endpoints
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			switch r.URL.Path {
			case "/predict/migration_time":
				// This endpoint works
				w.WriteHeader(http.StatusOK)
				w.Write([]byte(`{"duration_seconds": 300, "confidence": 0.8}`))
			case "/optimize/migration_strategy":
				// This endpoint fails
				w.WriteHeader(http.StatusInternalServerError)
			case "/analyze/anomalies":
				// This endpoint is slow
				time.Sleep(6 * time.Second)
				w.WriteHeader(http.StatusOK)
				w.Write([]byte(`{"anomalies": []}`))
			default:
				w.WriteHeader(http.StatusNotFound)
			}
		}))
		defer server.Close()

		config := migration.MigrationConfig{
			MaxDowntime:        30 * time.Second,
			TargetTransferRate: 100 * 1024 * 1024,
		}

		aiConfig := &migration.AIConfig{
			Enabled:  true,
			Endpoint: server.URL,
			Timeout:  3 * time.Second,
		}

		orchestrator, err := migration.NewLiveMigrationOrchestratorWithAI(config, aiConfig)
		require.NoError(t, err)
		defer orchestrator.Close()

		// Test mixed AI success/failure scenarios
		ctx := context.Background()
		migrationID, err := orchestrator.MigrateVM(ctx, "vm1", "node1", "node2",
			migration.MigrationOptions{Priority: 3})
		assert.NoError(t, err)
		assert.NotEmpty(t, migrationID)
	})
}

// TestSystemStabilityUnderAIFailure verifies system remains stable when AI fails
func TestSystemStabilityUnderAIFailure(t *testing.T) {
	t.Run("HighLoadWithAIFailures", func(t *testing.T) {
		// Create failing AI service
		config := scheduler.DefaultSchedulerConfig()
		aiConfig := scheduler.DefaultAISchedulerConfig()
		aiConfig.Enabled = true
		aiConfig.AIEngineURL = "http://invalid-ai-service:9999"
		aiConfig.RequestTimeout = 1 * time.Second
		aiConfig.MaxRetries = 1

		sched := scheduler.NewSchedulerWithAI(config, aiConfig)
		err := sched.Start()
		require.NoError(t, err)
		defer sched.Stop()

		// Register multiple nodes
		for i := 1; i <= 5; i++ {
			resources := map[scheduler.ResourceType]*scheduler.Resource{
				scheduler.ResourceCPU: {
					Type:     scheduler.ResourceCPU,
					Capacity: 100,
					Used:     float64(i * 5),
				},
				scheduler.ResourceMemory: {
					Type:     scheduler.ResourceMemory,
					Capacity: 16384,
					Used:     float64(i * 1024),
				},
			}
			err = sched.RegisterNode(fmt.Sprintf("node%d", i), resources)
			require.NoError(t, err)
		}

		// Generate high load of requests
		var requestIDs []string
		for i := 0; i < 20; i++ {
			constraints := []scheduler.ResourceConstraint{
				{Type: scheduler.ResourceCPU, MinAmount: 5, MaxAmount: 15},
				{Type: scheduler.ResourceMemory, MinAmount: 512, MaxAmount: 2048},
			}
			requestID, err := sched.RequestResources(constraints, i%5+1, 30*time.Minute)
			assert.NoError(t, err)
			requestIDs = append(requestIDs, requestID)
		}

		// Wait for processing
		time.Sleep(8 * time.Second)

		// System should remain stable despite AI failures
		nodeStatus := sched.GetNodesStatus()
		assert.Equal(t, 5, len(nodeStatus))

		allocations := sched.GetActiveAllocations()
		assert.Greater(t, len(allocations), 0)

		pendingRequests := sched.GetPendingRequests()
		// Some requests should be processed even without AI
		assert.Less(t, len(pendingRequests), 20)
	})

	t.Run("GracefulDegradationMetrics", func(t *testing.T) {
		config := scheduler.DefaultSchedulerConfig()

		// Create scheduler with no AI initially
		sched1 := scheduler.NewScheduler(config)
		err := sched1.Start()
		require.NoError(t, err)
		defer sched1.Stop()

		// Create scheduler with failing AI
		aiConfig := scheduler.DefaultAISchedulerConfig()
		aiConfig.Enabled = true
		aiConfig.AIEngineURL = "http://nonexistent:9999"
		aiConfig.RequestTimeout = 1 * time.Second
		sched2 := scheduler.NewSchedulerWithAI(config, aiConfig)
		err = sched2.Start()
		require.NoError(t, err)
		defer sched2.Stop()

		// Both schedulers should function similarly
		resources := map[scheduler.ResourceType]*scheduler.Resource{
			scheduler.ResourceCPU: {Type: scheduler.ResourceCPU, Capacity: 100, Used: 20},
		}

		err = sched1.RegisterNode("node1", resources)
		require.NoError(t, err)
		err = sched2.RegisterNode("node1", resources)
		require.NoError(t, err)

		constraints := []scheduler.ResourceConstraint{
			{Type: scheduler.ResourceCPU, MinAmount: 10},
		}

		req1, err := sched1.RequestResources(constraints, 1, time.Hour)
		assert.NoError(t, err)
		req2, err := sched2.RequestResources(constraints, 1, time.Hour)
		assert.NoError(t, err)

		// Both should succeed
		assert.NotEmpty(t, req1)
		assert.NotEmpty(t, req2)

		// Wait for allocations
		time.Sleep(6 * time.Second)

		// Check both have similar behavior
		alloc1 := sched1.GetActiveAllocations()
		alloc2 := sched2.GetActiveAllocations()

		// Both should have allocations
		assert.Greater(t, len(alloc1), 0)
		assert.Greater(t, len(alloc2), 0)
	})
}

// Helper function to verify AI provider fallback behavior
func (m *MockFailingAIProvider) GetAIProvider() scheduler.AIProvider {
	return m
}

// TestFallbackStrategyComprehensive tests the fallback strategies in detail
func TestFallbackStrategyComprehensive(t *testing.T) {
	t.Run("SchedulerFallbackStrategies", func(t *testing.T) {
		config := scheduler.DefaultSchedulerConfig()
		fallback := scheduler.NewFallbackSchedulingStrategy(config)

		// Test resource prediction patterns
		predictions, confidence := fallback.PredictResourceDemand("node1", scheduler.ResourceCPU, 60)
		assert.NotEmpty(t, predictions)
		assert.Equal(t, 0.6, confidence)
		assert.Equal(t, 12, len(predictions)) // 60 minutes / 5-minute intervals

		// Test different times of day
		// Note: This would need time mocking in a real test

		// Test performance optimization
		clusterData := map[string]interface{}{
			"cpu_usage":   0.85,
			"node_count":  4,
		}
		recommendations := fallback.OptimizePerformance(clusterData)
		assert.Contains(t, recommendations, "scale_up")
		assert.Equal(t, true, recommendations["scale_up"])

		// Test anomaly detection with various metrics
		testCases := []struct {
			name     string
			metrics  map[string]float64
			expected bool
		}{
			{
				name:     "Normal metrics",
				metrics:  map[string]float64{"cpu_usage": 0.7, "memory_usage": 0.6},
				expected: false,
			},
			{
				name:     "High CPU usage",
				metrics:  map[string]float64{"cpu_usage": 0.95, "memory_usage": 0.6},
				expected: true,
			},
			{
				name:     "High error rate",
				metrics:  map[string]float64{"cpu_usage": 0.5, "error_rate": 0.1},
				expected: true,
			},
			{
				name:     "Rapid CPU change",
				metrics:  map[string]float64{"cpu_usage": 0.8, "prev_cpu_usage": 0.3},
				expected: true,
			},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				isAnomaly, score, recommendations := fallback.DetectAnomalies(tc.metrics)
				assert.Equal(t, tc.expected, isAnomaly)
				if isAnomaly {
					assert.Greater(t, score, 0.0)
					assert.NotEmpty(t, recommendations)
				}
			})
		}
	})

	t.Run("MigrationFallbackStrategies", func(t *testing.T) {
		config := migration.MigrationConfig{
			MaxDowntime:        30 * time.Second,
			TargetTransferRate: 100 * 1024 * 1024,
		}
		fallback := migration.NewFallbackMigrationStrategy(config)

		// Test migration time prediction for different VM sizes
		testCases := []struct {
			vmSize   string
			expected time.Duration
		}{
			{"small", 2 * time.Minute},
			{"medium", 5 * time.Minute},
			{"large", 10 * time.Minute},
			{"xlarge", 20 * time.Minute},
		}

		for _, tc := range testCases {
			t.Run(fmt.Sprintf("VM size %s", tc.vmSize), func(t *testing.T) {
				duration, confidence := fallback.PredictMigrationTime("node1", "node2", tc.vmSize)
				expectedWithOverhead := time.Duration(float64(tc.expected) * 1.3) // Network overhead
				assert.Equal(t, expectedWithOverhead, duration)
				assert.Equal(t, 0.6, confidence)
			})
		}

		// Test bandwidth prediction
		bandwidth := fallback.PredictBandwidthRequirements("large", "congested")
		assert.Greater(t, bandwidth, int64(0))

		// Test strategy optimization for different workloads
		workloadTests := []struct {
			workload string
			expected migration.MigrationType
		}{
			{"database", migration.MigrationTypePostCopy},
			{"web", migration.MigrationTypePreCopy},
			{"batch", migration.MigrationTypeCold},
		}

		for _, test := range workloadTests {
			vmData := map[string]interface{}{"workload_type": test.workload}
			networkData := map[string]interface{}{"bandwidth": int64(1024 * 1024 * 1024)}
			strategy := fallback.OptimizeMigrationStrategy(vmData, networkData)
			assert.Equal(t, test.expected, strategy.Type)
		}
	})
}

// BenchmarkFallbackPerformance benchmarks fallback performance vs AI calls
func BenchmarkFallbackPerformance(t *testing.B) {
	config := scheduler.DefaultSchedulerConfig()
	fallback := scheduler.NewFallbackSchedulingStrategy(config)

	t.Run("ResourcePrediction", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			fallback.PredictResourceDemand("node1", scheduler.ResourceCPU, 60)
		}
	})

	t.Run("AnomalyDetection", func(b *testing.B) {
		metrics := map[string]float64{
			"cpu_usage":    0.85,
			"memory_usage": 0.75,
			"disk_usage":   0.60,
		}
		for i := 0; i < b.N; i++ {
			fallback.DetectAnomalies(metrics)
		}
	})
}