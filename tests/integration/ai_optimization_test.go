package integration_test

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/khryptorgraphics/novacron/backend/core/ai"
	"github.com/khryptorgraphics/novacron/backend/core/migration"
	"github.com/khryptorgraphics/novacron/backend/core/scheduler"
	"github.com/khryptorgraphics/novacron/backend/core/vm"
)

// TestAISchedulerIntegration tests AI integration with the scheduler
func TestAISchedulerIntegration(t *testing.T) {
	// Create mock AI server
	aiServer := createMockAIServer(t)
	defer aiServer.Close()

	// Create AI integration layer
	aiConfig := ai.DefaultAIConfig()
	aiConfig.Timeout = 5 * time.Second
	aiLayer := ai.NewAIIntegrationLayer(aiServer.URL, "test-key", aiConfig)

	// Create scheduler with AI integration
	config := scheduler.SchedulerConfig{
		MaxCPUOvercommit:    2.0,
		MaxMemoryOvercommit: 1.5,
		SchedulingInterval:  10 * time.Second,
		MetricsRetention:    24 * time.Hour,
	}

	aiSchedulerConfig := scheduler.AIConfig{
		Enabled:                     true,
		Endpoint:                    aiServer.URL,
		Timeout:                     30 * time.Second,
		ConfidenceThreshold:         0.8,
		RetryAttempts:               3,
		EnableOptimization:          true,
		EnableAnomalyDetection:      true,
		EnablePredictiveAdjustments: true,
	}

	sched, err := scheduler.NewSchedulerWithAI(config, &aiSchedulerConfig)
	require.NoError(t, err)
	defer sched.Close()

	// Test AI-powered resource prediction
	t.Run("ResourcePrediction", func(t *testing.T) {
		ctx := context.Background()

		req := ai.ResourcePredictionRequest{
			NodeID:        "node-1",
			ResourceType:  "cpu",
			HorizonMinutes: 60,
			HistoricalData: generateMockResourceData(),
		}

		resp, err := aiLayer.PredictResourceDemand(ctx, req)
		require.NoError(t, err)
		assert.NotNil(t, resp)
		assert.Greater(t, resp.Confidence, 0.5)
		assert.NotEmpty(t, resp.Predictions)
	})

	// Test AI-powered performance optimization
	t.Run("PerformanceOptimization", func(t *testing.T) {
		ctx := context.Background()

		req := ai.PerformanceOptimizationRequest{
			ClusterID: "cluster-1",
			ClusterData: map[string]interface{}{
				"nodes":     3,
				"cpu_usage": 75.5,
				"memory_usage": 60.2,
				"network_load": 45.8,
			},
			Goals: []string{"minimize_latency", "maximize_throughput"},
		}

		resp, err := aiLayer.OptimizePerformance(ctx, req)
		require.NoError(t, err)
		assert.NotNil(t, resp)
		assert.NotEmpty(t, resp.Recommendations)
		assert.Greater(t, resp.Confidence, 0.6)
	})

	// Test anomaly detection
	t.Run("AnomalyDetection", func(t *testing.T) {
		ctx := context.Background()

		req := ai.AnomalyDetectionRequest{
			ResourceID:  "vm-123",
			MetricType:  "cpu",
			DataPoints:  generateAnomalousData(),
			Sensitivity: 0.8,
		}

		resp, err := aiLayer.DetectAnomalies(ctx, req)
		require.NoError(t, err)
		assert.NotNil(t, resp)
		// Should detect anomalies in our generated data
		assert.NotEmpty(t, resp.Anomalies)
	})

	// Test scheduler AI integration
	t.Run("SchedulerAIIntegration", func(t *testing.T) {
		// Create test VMs
		testVMs := []*vm.VM{
			{ID: "vm-1", CPUCores: 2, MemoryMB: 4096, DiskGB: 50},
			{ID: "vm-2", CPUCores: 4, MemoryMB: 8192, DiskGB: 100},
			{ID: "vm-3", CPUCores: 1, MemoryMB: 2048, DiskGB: 25},
		}

		// Create test nodes
		testNodes := []*scheduler.Node{
			{ID: "node-1", CPUCores: 8, MemoryMB: 16384, DiskGB: 500, Status: scheduler.NodeStatusReady},
			{ID: "node-2", CPUCores: 16, MemoryMB: 32768, DiskGB: 1000, Status: scheduler.NodeStatusReady},
		}

		// Add nodes to scheduler
		for _, node := range testNodes {
			err := sched.AddNode(node)
			require.NoError(t, err)
		}

		// Schedule VMs with AI assistance
		for _, testVM := range testVMs {
			result, err := sched.ScheduleVM(context.Background(), testVM)
			require.NoError(t, err)
			assert.NotEmpty(t, result.NodeID)
			assert.Greater(t, result.Confidence, 0.0)

			// Verify AI recommendation was considered
			assert.NotNil(t, result.AIRecommendation)
		}

		// Test AI metrics collection
		metrics := sched.GetAIMetrics()
		assert.True(t, metrics["ai_enabled"].(bool))
		assert.GreaterOrEqual(t, metrics["optimization_success"].(int64), int64(0))
	})
}

// TestAIMigrationIntegration tests AI integration with migration orchestrator
func TestAIMigrationIntegration(t *testing.T) {
	// Create mock AI server
	aiServer := createMockAIServer(t)
	defer aiServer.Close()

	// Create migration orchestrator with AI integration
	migrationConfig := migration.MigrationConfig{
		MaxDowntime:             30 * time.Second,
		TargetTransferRate:      20 * 1024 * 1024 * 1024, // 20 GB/min
		SuccessRateTarget:       0.999,
		EnableCompression:       true,
		CompressionType:         migration.CompressionTypeLZ4,
		CompressionLevel:        6,
		EnableEncryption:        true,
		EnableDeltaSync:         true,
		BandwidthLimit:          1024 * 1024 * 1024, // 1 GB/s
		AdaptiveBandwidth:       true,
		QoSPriority:             migration.QoSPriorityCritical,
		MemoryIterations:        5,
		DirtyPageThreshold:      1000,
		ConvergenceTimeout:      5 * time.Minute,
		EnableCheckpointing:     true,
		CheckpointInterval:      1 * time.Minute,
		RetryAttempts:           3,
		RetryDelay:              5 * time.Second,
		MaxCPUUsage:             0.8,
		MaxMemoryUsage:          8 * 1024 * 1024 * 1024, // 8 GB
		MaxConcurrentMigrations: 3,
	}

	aiConfig := &migration.AIConfig{
		Enabled:                     true,
		Endpoint:                    aiServer.URL,
		Timeout:                     30 * time.Second,
		ConfidenceThreshold:         0.8,
		RetryAttempts:               3,
		EnableOptimization:          true,
		EnableAnomalyDetection:      true,
		EnablePredictiveAdjustments: true,
	}

	orchestrator, err := migration.NewLiveMigrationOrchestratorWithAI(migrationConfig, aiConfig)
	require.NoError(t, err)
	defer orchestrator.Close()

	// Test AI-optimized migration
	t.Run("AIOptimizedMigration", func(t *testing.T) {
		ctx := context.Background()

		options := migration.MigrationOptions{
			Priority: 5,
			Force:    false,
		}

		migrationID, err := orchestrator.MigrateVM(ctx, "vm-test", "node-1", "node-2", options)
		require.NoError(t, err)
		assert.NotEmpty(t, migrationID)

		// Wait for migration to start processing
		time.Sleep(2 * time.Second)

		// Check migration status
		status, err := orchestrator.GetMigrationStatus(migrationID)
		require.NoError(t, err)
		assert.NotNil(t, status)

		// Verify AI metrics are being collected
		aiMetrics := orchestrator.GetAIMetrics()
		assert.True(t, aiMetrics["ai_enabled"].(bool))
	})

	// Test AI anomaly detection during migration
	t.Run("MigrationAnomalyDetection", func(t *testing.T) {
		// This would be tested with a longer-running integration test
		// that monitors for anomalies during actual migration
		metrics := orchestrator.GetAIMetrics()
		assert.NotNil(t, metrics)
		assert.Contains(t, metrics, "anomalies_detected")
	})
}

// TestAIIntegrationLayerResilience tests resilience features of AI integration
func TestAIIntegrationLayerResilience(t *testing.T) {
	t.Run("CircuitBreaker", func(t *testing.T) {
		// Create AI server that fails
		failingServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusInternalServerError)
		}))
		defer failingServer.Close()

		aiConfig := ai.DefaultAIConfig()
		aiConfig.CircuitBreakerThreshold = 3
		aiConfig.CircuitBreakerTimeout = 1 * time.Second
		aiConfig.Retries = 1

		aiLayer := ai.NewAIIntegrationLayer(failingServer.URL, "test-key", aiConfig)

		ctx := context.Background()
		req := ai.ResourcePredictionRequest{
			NodeID:        "node-1",
			ResourceType:  "cpu",
			HorizonMinutes: 60,
		}

		// Make requests until circuit breaker opens
		for i := 0; i < 5; i++ {
			_, err := aiLayer.PredictResourceDemand(ctx, req)
			assert.Error(t, err)
		}

		// Verify circuit breaker is open
		metrics := aiLayer.GetMetrics()
		assert.Greater(t, metrics["circuit_breaker_trips"].(int64), int64(0))
		assert.Equal(t, "open", metrics["circuit_breaker_state"].(string))
	})

	t.Run("Caching", func(t *testing.T) {
		callCount := 0
		cachingServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			callCount++
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			fmt.Fprintf(w, `{
				"id": "test",
				"success": true,
				"data": {
					"predictions": [0.5, 0.6, 0.7],
					"confidence": 0.85,
					"model_info": {
						"name": "test_model",
						"version": "1.0",
						"accuracy": 0.9
					}
				}
			}`)
		}))
		defer cachingServer.Close()

		aiConfig := ai.DefaultAIConfig()
		aiConfig.CacheTTL = 1 * time.Minute

		aiLayer := ai.NewAIIntegrationLayer(cachingServer.URL, "test-key", aiConfig)

		ctx := context.Background()
		req := ai.ResourcePredictionRequest{
			NodeID:        "node-1",
			ResourceType:  "cpu",
			HorizonMinutes: 60,
		}

		// First request - should hit the server
		_, err := aiLayer.PredictResourceDemand(ctx, req)
		require.NoError(t, err)
		assert.Equal(t, 1, callCount)

		// Second request - should hit cache
		_, err = aiLayer.PredictResourceDemand(ctx, req)
		require.NoError(t, err)
		assert.Equal(t, 1, callCount) // Same count, cache hit

		// Verify cache metrics
		metrics := aiLayer.GetMetrics()
		assert.Greater(t, metrics["cache_hits"].(int64), int64(0))
	})

	t.Run("RateLimiting", func(t *testing.T) {
		slowServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			time.Sleep(100 * time.Millisecond) // Simulate slow response
			w.WriteHeader(http.StatusOK)
			fmt.Fprintf(w, `{"id": "test", "success": true, "data": {}}`)
		}))
		defer slowServer.Close()

		aiConfig := ai.DefaultAIConfig()
		aiConfig.MaxConnections = 2 // Low limit for testing
		aiConfig.Timeout = 200 * time.Millisecond

		aiLayer := ai.NewAIIntegrationLayer(slowServer.URL, "test-key", aiConfig)

		ctx := context.Background()
		req := ai.ResourcePredictionRequest{
			NodeID:        "node-1",
			ResourceType:  "cpu",
			HorizonMinutes: 60,
		}

		// Start multiple concurrent requests
		errChan := make(chan error, 5)
		for i := 0; i < 5; i++ {
			go func() {
				_, err := aiLayer.PredictResourceDemand(ctx, req)
				errChan <- err
			}()
		}

		// Collect results
		errors := 0
		for i := 0; i < 5; i++ {
			err := <-errChan
			if err != nil {
				errors++
			}
		}

		// Should have some rate limiting errors
		assert.Greater(t, errors, 0)
	})
}

// TestAIWorkloadPatternAnalysis tests workload pattern analysis
func TestAIWorkloadPatternAnalysis(t *testing.T) {
	aiServer := createMockAIServer(t)
	defer aiServer.Close()

	aiConfig := ai.DefaultAIConfig()
	aiLayer := ai.NewAIIntegrationLayer(aiServer.URL, "test-key", aiConfig)

	t.Run("WorkloadPatternAnalysis", func(t *testing.T) {
		ctx := context.Background()

		req := ai.WorkloadPatternRequest{
			WorkloadID: "workload-1",
			TimeRange: ai.TimeRange{
				Start: time.Now().Add(-24 * time.Hour),
				End:   time.Now(),
			},
			MetricTypes: []string{"cpu", "memory", "network"},
			DataPoints:  generateWorkloadData(),
		}

		resp, err := aiLayer.AnalyzeWorkloadPattern(ctx, req)
		require.NoError(t, err)
		assert.NotNil(t, resp)
		assert.NotEmpty(t, resp.Patterns)
		assert.NotEmpty(t, resp.Classification)
		assert.Greater(t, resp.Confidence, 0.5)
	})
}

// Helper functions for creating mock data and servers

func createMockAIServer(t *testing.T) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")

		// Simple mock responses based on service type
		var response string
		switch r.URL.Path {
		case "/api/v1/process":
			response = `{
				"id": "test-id",
				"success": true,
				"data": {
					"predictions": [0.5, 0.6, 0.7, 0.65, 0.8],
					"confidence": 0.85,
					"model_info": {
						"name": "test_model",
						"version": "1.0",
						"training_data": "synthetic",
						"accuracy": 0.92,
						"last_trained": "2024-01-01T00:00:00Z"
					},
					"recommendations": [
						{
							"type": "scaling",
							"target": "cpu",
							"action": "scale_up",
							"parameters": {"factor": 1.5},
							"priority": 1,
							"impact": "high",
							"confidence": 0.9
						}
					],
					"expected_gains": {
						"throughput": 0.25,
						"latency": -0.15
					},
					"risk_assessment": {
						"overall_risk": 0.2,
						"risk_factors": ["resource_contention"],
						"mitigations": ["gradual_scaling"]
					},
					"anomalies": [
						{
							"timestamp": "2024-01-01T12:00:00Z",
							"anomaly_type": "spike",
							"severity": "warning",
							"score": 0.8,
							"description": "CPU usage spike detected",
							"affected_metrics": ["cpu"],
							"recommendations": ["investigate_workload"]
						}
					],
					"overall_score": 0.3,
					"baseline": {
						"cpu": 0.4,
						"memory": 0.6
					},
					"patterns": [
						{
							"type": "periodic",
							"start_time": "2024-01-01T09:00:00Z",
							"end_time": "2024-01-01T17:00:00Z",
							"intensity": 0.8,
							"frequency": "daily",
							"confidence": 0.9,
							"description": "Daily business hours pattern"
						}
					],
					"classification": "cpu_intensive",
					"seasonality": {
						"has_seasonality": true,
						"period": "86400s",
						"strength": 0.7,
						"components": ["daily"],
						"peak_times": ["2024-01-01T14:00:00Z"],
						"low_times": ["2024-01-01T02:00:00Z"]
					}
				},
				"confidence": 0.85,
				"process_time": 0.123,
				"model_version": "1.0"
			}`
		default:
			response = `{"id": "test", "success": true, "data": {}}`
		}

		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, response)
	}))
}

func generateMockResourceData() []ai.ResourceDataPoint {
	var data []ai.ResourceDataPoint
	now := time.Now()

	for i := 0; i < 100; i++ {
		data = append(data, ai.ResourceDataPoint{
			Timestamp: now.Add(time.Duration(-i) * time.Minute),
			Value:     0.3 + 0.4*float64(i%24)/24.0, // Simulate daily pattern
		})
	}

	return data
}

func generateAnomalousData() []ai.ResourceDataPoint {
	var data []ai.ResourceDataPoint
	now := time.Now()

	for i := 0; i < 50; i++ {
		value := 0.5 // Normal value

		// Add anomalies
		if i == 25 {
			value = 2.0 // Spike anomaly
		}

		data = append(data, ai.ResourceDataPoint{
			Timestamp: now.Add(time.Duration(-i) * time.Minute),
			Value:     value,
		})
	}

	return data
}

func generateWorkloadData() []ai.ResourceDataPoint {
	var data []ai.ResourceDataPoint
	now := time.Now()

	for i := 0; i < 144; i++ { // 24 hours of 10-minute intervals
		// Simulate business hours pattern
		hour := i / 6
		var value float64

		if hour >= 9 && hour <= 17 {
			value = 0.7 + 0.2*float64(i%6)/6.0 // High during business hours
		} else {
			value = 0.2 + 0.1*float64(i%6)/6.0 // Low during off hours
		}

		data = append(data, ai.ResourceDataPoint{
			Timestamp: now.Add(time.Duration(-i*10) * time.Minute),
			Value:     value,
		})
	}

	return data
}

// TestAIHealthCheck tests AI service health checking
func TestAIHealthCheck(t *testing.T) {
	t.Run("HealthyService", func(t *testing.T) {
		healthyServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusOK)
			fmt.Fprint(w, `{"id": "health", "success": true, "data": {"status": "healthy"}}`)
		}))
		defer healthyServer.Close()

		aiLayer := ai.NewAIIntegrationLayer(healthyServer.URL, "test-key", ai.DefaultAIConfig())

		err := aiLayer.HealthCheck(context.Background())
		assert.NoError(t, err)
	})

	t.Run("UnhealthyService", func(t *testing.T) {
		unhealthyServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusServiceUnavailable)
		}))
		defer unhealthyServer.Close()

		aiLayer := ai.NewAIIntegrationLayer(unhealthyServer.URL, "test-key", ai.DefaultAIConfig())

		err := aiLayer.HealthCheck(context.Background())
		assert.Error(t, err)
	})
}

// TestAIModelManagement tests AI model management functionality
func TestAIModelManagement(t *testing.T) {
	aiServer := createMockAIServer(t)
	defer aiServer.Close()

	aiLayer := ai.NewAIIntegrationLayer(aiServer.URL, "test-key", ai.DefaultAIConfig())

	t.Run("GetModelInfo", func(t *testing.T) {
		ctx := context.Background()

		modelInfo, err := aiLayer.GetModelInfo(ctx, "resource_prediction")
		require.NoError(t, err)
		assert.NotNil(t, modelInfo)
		assert.NotEmpty(t, modelInfo.Name)
		assert.NotEmpty(t, modelInfo.Version)
		assert.Greater(t, modelInfo.Accuracy, 0.0)
	})

	t.Run("TrainModel", func(t *testing.T) {
		ctx := context.Background()

		trainingData := generateMockResourceData()
		err := aiLayer.TrainModel(ctx, "resource_prediction", trainingData)
		assert.NoError(t, err)
	})
}

// TestAIPredictiveScaling tests predictive scaling functionality
func TestAIPredictiveScaling(t *testing.T) {
	aiServer := createMockAIServer(t)
	defer aiServer.Close()

	aiLayer := ai.NewAIIntegrationLayer(aiServer.URL, "test-key", ai.DefaultAIConfig())

	t.Run("PredictScalingNeeds", func(t *testing.T) {
		ctx := context.Background()

		clusterData := map[string]interface{}{
			"cluster_id": "cluster-1",
			"nodes": []map[string]interface{}{
				{
					"id": "node-1",
					"cpu_usage": 0.8,
					"memory_usage": 0.6,
					"disk_usage": 0.4,
				},
				{
					"id": "node-2",
					"cpu_usage": 0.9,
					"memory_usage": 0.7,
					"disk_usage": 0.5,
				},
			},
			"workload_trends": map[string]interface{}{
				"cpu": []float64{0.6, 0.7, 0.8, 0.85, 0.9},
				"memory": []float64{0.5, 0.55, 0.6, 0.65, 0.7},
			},
		}

		resp, err := aiLayer.PredictScalingNeeds(ctx, clusterData)
		require.NoError(t, err)
		assert.NotNil(t, resp)

		// Should contain scaling recommendations
		assert.Contains(t, resp, "recommendations")
	})
}

// TestAINegativePathsAndSchemaMismatches tests error handling for malformed data and schema mismatches
func TestAINegativePathsAndSchemaMismatches(t *testing.T) {
	t.Run("SchemaMismatchTests", func(t *testing.T) {
		t.Run("MissingRequiredFields", func(t *testing.T) {
			// Create AI server that returns response with missing required fields
			malformedServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				// Missing "predictions" and "confidence" fields
				fmt.Fprint(w, `{
					"id": "test-id",
					"success": true,
					"data": {
						"model_info": {
							"name": "test_model",
							"version": "1.0"
						}
					}
				}`)
			}))
			defer malformedServer.Close()

			aiConfig := ai.DefaultAIConfig()
			aiConfig.CircuitBreakerThreshold = 2
			aiLayer := ai.NewAIIntegrationLayer(malformedServer.URL, "test-key", aiConfig)

			ctx := context.Background()
			req := ai.ResourcePredictionRequest{
				NodeID:        "node-1",
				ResourceType:  "cpu",
				HorizonMinutes: 60,
				HistoricalData: generateMockResourceData(),
			}

			_, err := aiLayer.PredictResourceDemand(ctx, req)
			assert.Error(t, err)
			assert.Contains(t, err.Error(), "failed to parse prediction response")

			// Verify circuit breaker increments failure count
			metrics := aiLayer.GetMetrics()
			assert.Greater(t, metrics["failed_requests"].(int64), int64(0))
		})

		t.Run("WrongDataTypes", func(t *testing.T) {
			// Create AI server that returns response with wrong data types
			wrongTypesServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				// Wrong data types: predictions should be []float64, confidence should be float64
				fmt.Fprint(w, `{
					"id": "test-id",
					"success": true,
					"data": {
						"predictions": "invalid_string_instead_of_array",
						"confidence": "invalid_string_instead_of_float",
						"model_info": {
							"name": "test_model",
							"version": "1.0",
							"accuracy": "should_be_float"
						}
					}
				}`)
			}))
			defer wrongTypesServer.Close()

			aiConfig := ai.DefaultAIConfig()
			aiConfig.CircuitBreakerThreshold = 2
			aiLayer := ai.NewAIIntegrationLayer(wrongTypesServer.URL, "test-key", aiConfig)

			ctx := context.Background()
			req := ai.ResourcePredictionRequest{
				NodeID:        "node-1",
				ResourceType:  "cpu",
				HorizonMinutes: 60,
				HistoricalData: generateMockResourceData(),
			}

			_, err := aiLayer.PredictResourceDemand(ctx, req)
			assert.Error(t, err)
			assert.Contains(t, err.Error(), "failed to parse prediction response")

			// Verify circuit breaker failure tracking
			metrics := aiLayer.GetMetrics()
			assert.Greater(t, metrics["failed_requests"].(int64), int64(0))
		})

		t.Run("InvalidJSONResponse", func(t *testing.T) {
			// Create AI server that returns invalid JSON
			invalidJSONServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				// Invalid JSON - missing closing brace and quotes
				fmt.Fprint(w, `{
					"id": test-id,
					"success": true,
					"data": {
						"predictions": [0.5, 0.6, 0.7
					}
				`)
			}))
			defer invalidJSONServer.Close()

			aiConfig := ai.DefaultAIConfig()
			aiConfig.CircuitBreakerThreshold = 2
			aiLayer := ai.NewAIIntegrationLayer(invalidJSONServer.URL, "test-key", aiConfig)

			ctx := context.Background()
			req := ai.ResourcePredictionRequest{
				NodeID:        "node-1",
				ResourceType:  "cpu",
				HorizonMinutes: 60,
				HistoricalData: generateMockResourceData(),
			}

			_, err := aiLayer.PredictResourceDemand(ctx, req)
			assert.Error(t, err)
			assert.Contains(t, err.Error(), "failed to parse AI response")

			// Verify circuit breaker tracks failures
			metrics := aiLayer.GetMetrics()
			assert.Greater(t, metrics["failed_requests"].(int64), int64(0))
		})

		t.Run("EmptyResponse", func(t *testing.T) {
			// Create AI server that returns empty response
			emptyServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				// Empty response body
				fmt.Fprint(w, "")
			}))
			defer emptyServer.Close()

			aiConfig := ai.DefaultAIConfig()
			aiConfig.CircuitBreakerThreshold = 2
			aiLayer := ai.NewAIIntegrationLayer(emptyServer.URL, "test-key", aiConfig)

			ctx := context.Background()
			req := ai.ResourcePredictionRequest{
				NodeID:        "node-1",
				ResourceType:  "cpu",
				HorizonMinutes: 60,
				HistoricalData: generateMockResourceData(),
			}

			_, err := aiLayer.PredictResourceDemand(ctx, req)
			assert.Error(t, err)
			assert.Contains(t, err.Error(), "failed to parse AI response")

			// Verify metrics tracking
			metrics := aiLayer.GetMetrics()
			assert.Greater(t, metrics["failed_requests"].(int64), int64(0))
		})

		t.Run("TimeoutScenario", func(t *testing.T) {
			// Create AI server that never responds (simulates timeout)
			timeoutServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				// Sleep longer than the client timeout
				time.Sleep(2 * time.Second)
				w.WriteHeader(http.StatusOK)
				fmt.Fprint(w, `{"id": "test", "success": true, "data": {}}`)
			}))
			defer timeoutServer.Close()

			aiConfig := ai.DefaultAIConfig()
			aiConfig.Timeout = 500 * time.Millisecond // Short timeout for test
			aiConfig.CircuitBreakerThreshold = 2
			aiConfig.Retries = 1 // Reduce retries for faster test
			aiLayer := ai.NewAIIntegrationLayer(timeoutServer.URL, "test-key", aiConfig)

			ctx := context.Background()
			req := ai.ResourcePredictionRequest{
				NodeID:        "node-1",
				ResourceType:  "cpu",
				HorizonMinutes: 60,
				HistoricalData: generateMockResourceData(),
			}

			_, err := aiLayer.PredictResourceDemand(ctx, req)
			assert.Error(t, err)
			assert.Contains(t, err.Error(), "HTTP request failed")

			// Verify timeout leads to circuit breaker failure
			metrics := aiLayer.GetMetrics()
			assert.Greater(t, metrics["failed_requests"].(int64), int64(0))
		})

		t.Run("AIProcessingFailure", func(t *testing.T) {
			// Create AI server that returns success=false
			failureServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				fmt.Fprint(w, `{
					"id": "test-id",
					"success": false,
					"error": "Model training failed: insufficient data quality",
					"confidence": 0.0
				}`)
			}))
			defer failureServer.Close()

			aiConfig := ai.DefaultAIConfig()
			aiConfig.CircuitBreakerThreshold = 2
			aiLayer := ai.NewAIIntegrationLayer(failureServer.URL, "test-key", aiConfig)

			ctx := context.Background()
			req := ai.ResourcePredictionRequest{
				NodeID:        "node-1",
				ResourceType:  "cpu",
				HorizonMinutes: 60,
				HistoricalData: generateMockResourceData(),
			}

			_, err := aiLayer.PredictResourceDemand(ctx, req)
			assert.Error(t, err)
			assert.Contains(t, err.Error(), "AI processing failed")
			assert.Contains(t, err.Error(), "Model training failed: insufficient data quality")

			// Verify AI processing failure increments circuit breaker
			metrics := aiLayer.GetMetrics()
			assert.Greater(t, metrics["failed_requests"].(int64), int64(0))
		})

		t.Run("HTTPErrorStatusCodes", func(t *testing.T) {
			testCases := []struct {
				name       string
				statusCode int
				body       string
			}{
				{"BadRequest", http.StatusBadRequest, "Invalid request parameters"},
				{"Unauthorized", http.StatusUnauthorized, "Authentication failed"},
				{"InternalServerError", http.StatusInternalServerError, "Internal server error occurred"},
				{"ServiceUnavailable", http.StatusServiceUnavailable, "Service temporarily unavailable"},
			}

			for _, tc := range testCases {
				t.Run(tc.name, func(t *testing.T) {
					errorServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
						w.WriteHeader(tc.statusCode)
						fmt.Fprint(w, tc.body)
					}))
					defer errorServer.Close()

					aiConfig := ai.DefaultAIConfig()
					aiConfig.CircuitBreakerThreshold = 2
					aiConfig.Retries = 1 // Reduce for faster test
					aiLayer := ai.NewAIIntegrationLayer(errorServer.URL, "test-key", aiConfig)

					ctx := context.Background()
					req := ai.ResourcePredictionRequest{
						NodeID:        "node-1",
						ResourceType:  "cpu",
						HorizonMinutes: 60,
						HistoricalData: generateMockResourceData(),
					}

					_, err := aiLayer.PredictResourceDemand(ctx, req)
					assert.Error(t, err)
					assert.Contains(t, err.Error(), fmt.Sprintf("AI service returned status %d", tc.statusCode))

					// Verify error status codes trigger circuit breaker
					metrics := aiLayer.GetMetrics()
					assert.Greater(t, metrics["failed_requests"].(int64), int64(0))
				})
			}
		})

		t.Run("CircuitBreakerBehavior", func(t *testing.T) {
			// Create failing server to trigger circuit breaker
			failingServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusInternalServerError)
				fmt.Fprint(w, "Simulated server failure")
			}))
			defer failingServer.Close()

			aiConfig := ai.DefaultAIConfig()
			aiConfig.CircuitBreakerThreshold = 3
			aiConfig.Retries = 1 // Reduce retries for faster test
			aiLayer := ai.NewAIIntegrationLayer(failingServer.URL, "test-key", aiConfig)

			ctx := context.Background()
			req := ai.ResourcePredictionRequest{
				NodeID:        "node-1",
				ResourceType:  "cpu",
				HorizonMinutes: 60,
				HistoricalData: generateMockResourceData(),
			}

			// Make requests to trigger circuit breaker
			for i := 0; i < 5; i++ {
				_, err := aiLayer.PredictResourceDemand(ctx, req)
				assert.Error(t, err)

				// After threshold, should get circuit breaker error
				if i >= 3 {
					assert.Contains(t, err.Error(), "circuit breaker is open")
				}
			}

			// Verify circuit breaker state and metrics
			metrics := aiLayer.GetMetrics()
			assert.Greater(t, metrics["circuit_breaker_trips"].(int64), int64(0))
			assert.Equal(t, "open", metrics["circuit_breaker_state"].(string))
			assert.Greater(t, metrics["failed_requests"].(int64), int64(0))
			assert.Equal(t, int64(0), metrics["successful_requests"].(int64))
		})

		t.Run("FallbackPathsForDifferentServices", func(t *testing.T) {
			// Test fallback behavior for different AI service types
			malformedServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				// Return malformed response for all services
				fmt.Fprint(w, `{
					"id": "test-id",
					"success": true,
					"data": {
						"incomplete": "data"
					}
				}`)
			}))
			defer malformedServer.Close()

			aiLayer := ai.NewAIIntegrationLayer(malformedServer.URL, "test-key", ai.DefaultAIConfig())
			ctx := context.Background()

			// Test ResourcePrediction fallback
			predReq := ai.ResourcePredictionRequest{
				NodeID:        "node-1",
				ResourceType:  "cpu",
				HorizonMinutes: 60,
				HistoricalData: generateMockResourceData(),
			}
			_, err := aiLayer.PredictResourceDemand(ctx, predReq)
			assert.Error(t, err)
			assert.Contains(t, err.Error(), "failed to parse prediction response")

			// Test PerformanceOptimization fallback
			optReq := ai.PerformanceOptimizationRequest{
				ClusterID: "cluster-1",
				ClusterData: map[string]interface{}{
					"nodes": 3,
					"cpu_usage": 75.5,
				},
				Goals: []string{"minimize_latency"},
			}
			_, err = aiLayer.OptimizePerformance(ctx, optReq)
			assert.Error(t, err)
			assert.Contains(t, err.Error(), "failed to parse optimization response")

			// Test AnomalyDetection fallback
			anomReq := ai.AnomalyDetectionRequest{
				ResourceID:  "vm-123",
				MetricType:  "cpu",
				DataPoints:  generateAnomalousData(),
				Sensitivity: 0.8,
			}
			_, err = aiLayer.DetectAnomalies(ctx, anomReq)
			assert.Error(t, err)
			assert.Contains(t, err.Error(), "failed to parse anomaly response")

			// Test WorkloadPattern fallback
			workloadReq := ai.WorkloadPatternRequest{
				WorkloadID: "workload-1",
				TimeRange: ai.TimeRange{
					Start: time.Now().Add(-24 * time.Hour),
					End:   time.Now(),
				},
				MetricTypes: []string{"cpu", "memory"},
				DataPoints:  generateWorkloadData(),
			}
			_, err = aiLayer.AnalyzeWorkloadPattern(ctx, workloadReq)
			assert.Error(t, err)
			assert.Contains(t, err.Error(), "failed to parse pattern response")

			// Verify all services properly handle schema mismatches
			metrics := aiLayer.GetMetrics()
			assert.Greater(t, metrics["failed_requests"].(int64), int64(3)) // At least 4 failed requests
		})

		t.Run("ContextCancellation", func(t *testing.T) {
			// Create slow server to test context cancellation
			slowServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				time.Sleep(2 * time.Second)
				w.WriteHeader(http.StatusOK)
				fmt.Fprint(w, `{"id": "test", "success": true, "data": {}}`)
			}))
			defer slowServer.Close()

			aiLayer := ai.NewAIIntegrationLayer(slowServer.URL, "test-key", ai.DefaultAIConfig())

			// Create context that cancels quickly
			ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
			defer cancel()

			req := ai.ResourcePredictionRequest{
				NodeID:        "node-1",
				ResourceType:  "cpu",
				HorizonMinutes: 60,
				HistoricalData: generateMockResourceData(),
			}

			_, err := aiLayer.PredictResourceDemand(ctx, req)
			assert.Error(t, err)
			assert.True(t,
				err == context.DeadlineExceeded ||
				err == context.Canceled ||
				err.Error() == "context deadline exceeded",
				"Expected context cancellation error, got: %v", err)
		})

		t.Run("MalformedNestedStructures", func(t *testing.T) {
			// Test deeply nested malformed structures
			nestedMalformedServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				// Malformed nested structures - wrong types in model_info, recommendations, etc.
				fmt.Fprint(w, `{
					"id": "test-id",
					"success": true,
					"data": {
						"predictions": [0.5, 0.6, 0.7],
						"confidence": 0.85,
						"model_info": {
							"name": ["should_be_string"],
							"version": 123,
							"accuracy": "not_a_number",
							"last_trained": "invalid_date_format"
						},
						"recommendations": [
							{
								"type": 123,
								"target": ["should_be_string"],
								"action": null,
								"parameters": "should_be_object",
								"priority": "should_be_number",
								"confidence": []
							}
						],
						"risk_assessment": {
							"overall_risk": "should_be_number",
							"risk_factors": "should_be_array",
							"mitigations": 123
						}
					}
				}`)
			}))
			defer nestedMalformedServer.Close()

			aiLayer := ai.NewAIIntegrationLayer(nestedMalformedServer.URL, "test-key", ai.DefaultAIConfig())
			ctx := context.Background()

			// Test with performance optimization (has complex nested structures)
			req := ai.PerformanceOptimizationRequest{
				ClusterID: "cluster-1",
				ClusterData: map[string]interface{}{
					"nodes": 3,
					"cpu_usage": 75.5,
				},
				Goals: []string{"minimize_latency"},
			}

			_, err := aiLayer.OptimizePerformance(ctx, req)
			assert.Error(t, err)
			assert.Contains(t, err.Error(), "failed to parse optimization response")

			// Verify malformed nested structures are handled properly
			metrics := aiLayer.GetMetrics()
			assert.Greater(t, metrics["failed_requests"].(int64), int64(0))
		})
	})
}