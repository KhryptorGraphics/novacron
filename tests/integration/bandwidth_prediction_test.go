package integration

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/network"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestBandwidthPredictionIntegration(t *testing.T) {
	ctx := context.Background()
	logger := log.Default()

	t.Run("AI Performance Predictor Client", func(t *testing.T) {
		// Create mock prediction server
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			switch r.URL.Path {
			case "/predict":
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				fmt.Fprintf(w, `{
					"success": true,
					"prediction": {
						"predicted_bandwidth": 150.5,
						"confidence_interval": [120.0, 180.0],
						"prediction_confidence": 0.85,
						"optimal_time_window": null,
						"alternative_routes": [],
						"congestion_forecast": {"hour_1": 0.3, "hour_2": 0.5},
						"recommendation": "GOOD_CONDITIONS: Proceed with current routing plan"
					}
				}`)
			case "/metrics":
				w.WriteHeader(http.StatusOK)
			case "/workload":
				w.WriteHeader(http.StatusOK)
			case "/performance":
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				fmt.Fprintf(w, `{
					"models": {
						"random_forest": {
							"mae": 12.5,
							"mse": 250.0,
							"r2_score": 0.85,
							"training_samples": 10000
						}
					}
				}`)
			case "/health":
				w.WriteHeader(http.StatusOK)
			default:
				w.WriteHeader(http.StatusNotFound)
			}
		}))
		defer server.Close()

		// Create AI predictor client
		predictor := network.NewAIPerformancePredictor(server.URL, logger)

		// Test health check
		healthy := predictor.IsHealthy(ctx)
		assert.True(t, healthy, "Predictor should be healthy")

		// Test bandwidth prediction
		request := network.PredictionRequest{
			SourceNode: "node-1",
			TargetNode: "node-2",
			WorkloadChars: network.WorkloadCharacteristics{
				VMID:                "vm-test",
				WorkloadType:        "interactive",
				CPUCores:            4,
				MemoryGB:            8.0,
				StorageGB:           100.0,
				NetworkIntensive:    true,
				ExpectedConnections: 10,
				DataTransferPattern: "burst",
				PeakHours:           []int{9, 10, 16, 17},
				HistoricalBandwidth: 100.0,
			},
			TimeHorizonHours:   24,
			ConfidenceLevel:    0.95,
			IncludeUncertainty: true,
		}

		prediction, err := predictor.PredictBandwidth(ctx, request)
		require.NoError(t, err, "Prediction should succeed")
		assert.NotNil(t, prediction)
		assert.Equal(t, 150.5, prediction.PredictedBandwidth)
		assert.Equal(t, 0.85, prediction.PredictionConfidence)
		assert.Equal(t, "GOOD_CONDITIONS: Proceed with current routing plan", prediction.Recommendation)
		assert.Len(t, prediction.ConfidenceInterval, 2)
		assert.Equal(t, 120.0, prediction.ConfidenceInterval[0])
		assert.Equal(t, 180.0, prediction.ConfidenceInterval[1])

		// Test caching - second request should be from cache
		prediction2, err := predictor.PredictBandwidth(ctx, request)
		require.NoError(t, err)
		assert.Equal(t, prediction.PredictedBandwidth, prediction2.PredictedBandwidth)

		// Test storing network metrics
		metrics := network.NetworkMetrics{
			Timestamp:         time.Now(),
			SourceNode:        "node-1",
			TargetNode:        "node-2",
			BandwidthMbps:     150.0,
			LatencyMs:         10.5,
			PacketLoss:        0.001,
			JitterMs:          2.0,
			ThroughputMbps:    145.0,
			ConnectionQuality: 0.95,
			RouteHops:         3,
			CongestionLevel:   0.3,
		}

		err = predictor.StoreNetworkMetrics(ctx, metrics)
		assert.NoError(t, err, "Should store network metrics")

		// Test storing workload characteristics
		workload := network.WorkloadCharacteristics{
			VMID:                "vm-test-2",
			WorkloadType:        "batch",
			CPUCores:            8,
			MemoryGB:            16.0,
			StorageGB:           200.0,
			NetworkIntensive:    false,
			ExpectedConnections: 5,
			DataTransferPattern: "steady",
			PeakHours:           []int{1, 2, 3},
			HistoricalBandwidth: 50.0,
		}

		err = predictor.StoreWorkloadCharacteristics(ctx, workload)
		assert.NoError(t, err, "Should store workload characteristics")

		// Test getting model performance
		performance, err := predictor.GetModelPerformance(ctx)
		require.NoError(t, err)
		assert.NotNil(t, performance)
		assert.Contains(t, performance.Models, "random_forest")
		assert.Equal(t, 12.5, performance.Models["random_forest"].MAE)
		assert.Equal(t, 0.85, performance.Models["random_forest"].R2Score)
	})

	t.Run("Heuristic Performance Predictor", func(t *testing.T) {
		// Create heuristic predictor
		predictor := network.NewHeuristicPerformancePredictor(logger)

		// Test health check (always healthy)
		healthy := predictor.IsHealthy(ctx)
		assert.True(t, healthy)

		// Test bandwidth prediction
		request := network.PredictionRequest{
			SourceNode: "node-1",
			TargetNode: "node-2",
			WorkloadChars: network.WorkloadCharacteristics{
				VMID:                "vm-heuristic",
				WorkloadType:        "streaming",
				CPUCores:            2,
				MemoryGB:            4.0,
				StorageGB:           50.0,
				NetworkIntensive:    true,
				ExpectedConnections: 20,
				DataTransferPattern: "steady",
				PeakHours:           []int{8, 9, 10, 16, 17, 18},
				HistoricalBandwidth: 80.0,
			},
			TimeHorizonHours:   12,
			ConfidenceLevel:    0.90,
			IncludeUncertainty: false,
		}

		prediction, err := predictor.PredictBandwidth(ctx, request)
		require.NoError(t, err)
		assert.NotNil(t, prediction)
		assert.Greater(t, prediction.PredictedBandwidth, 0.0)
		assert.Equal(t, 0.7, prediction.PredictionConfidence) // Fixed confidence for heuristic
		assert.NotEmpty(t, prediction.Recommendation)
		assert.NotEmpty(t, prediction.CongestionForecast)

		// Check congestion forecast
		for key, congestion := range prediction.CongestionForecast {
			assert.GreaterOrEqual(t, congestion, 0.0)
			assert.LessOrEqual(t, congestion, 1.0)
			t.Logf("Congestion %s: %.2f", key, congestion)
		}

		// Test that store methods don't error (they're no-ops)
		err = predictor.StoreNetworkMetrics(ctx, network.NetworkMetrics{})
		assert.NoError(t, err)

		err = predictor.StoreWorkloadCharacteristics(ctx, network.WorkloadCharacteristics{})
		assert.NoError(t, err)

		// Test model performance (returns fixed values)
		performance, err := predictor.GetModelPerformance(ctx)
		require.NoError(t, err)
		assert.Contains(t, performance.Models, "heuristic")
		assert.Equal(t, 15.0, performance.Models["heuristic"].MAE)
		assert.Equal(t, 0.6, performance.Models["heuristic"].R2Score)
	})

	t.Run("Predictor Factory", func(t *testing.T) {
		factory := network.NewPredictorFactory(logger)

		// Test creating AI predictor
		aiConfig := map[string]interface{}{
			"base_url": "http://localhost:8080",
		}
		aiPredictor, err := factory.CreatePredictor("ai", aiConfig)
		require.NoError(t, err)
		assert.NotNil(t, aiPredictor)

		// Test creating heuristic predictor
		heuristicPredictor, err := factory.CreatePredictor("heuristic", nil)
		require.NoError(t, err)
		assert.NotNil(t, heuristicPredictor)

		// Test invalid predictor type
		_, err = factory.CreatePredictor("invalid", nil)
		assert.Error(t, err)

		// Test AI predictor without base_url
		_, err = factory.CreatePredictor("ai", map[string]interface{}{})
		assert.Error(t, err)
	})

	t.Run("Performance Predictor Manager with Fallback", func(t *testing.T) {
		// Create mock failing primary server
		failingServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/health" {
				w.WriteHeader(http.StatusServiceUnavailable)
			} else {
				w.WriteHeader(http.StatusInternalServerError)
			}
		}))
		defer failingServer.Close()

		// Create predictors
		primary := network.NewAIPerformancePredictor(failingServer.URL, logger)
		fallback := network.NewHeuristicPerformancePredictor(logger)

		// Create manager
		manager := network.NewPerformancePredictorManager(primary, fallback, logger)

		// Test health check (should use fallback since primary is unhealthy)
		healthy := manager.IsHealthy(ctx)
		assert.True(t, healthy, "Manager should be healthy via fallback")

		// Test prediction (should use fallback)
		request := network.PredictionRequest{
			SourceNode: "node-1",
			TargetNode: "node-2",
			WorkloadChars: network.WorkloadCharacteristics{
				VMID:                "vm-manager",
				WorkloadType:        "compute",
				CPUCores:            16,
				MemoryGB:            32.0,
				StorageGB:           500.0,
				NetworkIntensive:    false,
				ExpectedConnections: 2,
				DataTransferPattern: "burst",
				PeakHours:           []int{},
				HistoricalBandwidth: 25.0,
			},
			TimeHorizonHours:   6,
			ConfidenceLevel:    0.80,
			IncludeUncertainty: true,
		}

		prediction, err := manager.PredictBandwidth(ctx, request)
		require.NoError(t, err, "Should fall back to heuristic predictor")
		assert.NotNil(t, prediction)
		assert.Greater(t, prediction.PredictedBandwidth, 0.0)
		assert.Equal(t, 0.7, prediction.PredictionConfidence) // Heuristic confidence
	})

	t.Run("Prediction Request Validation", func(t *testing.T) {
		predictor := network.NewHeuristicPerformancePredictor(logger)

		// Test with extreme values
		request := network.PredictionRequest{
			SourceNode: "node-1",
			TargetNode: "node-2",
			WorkloadChars: network.WorkloadCharacteristics{
				VMID:                "vm-extreme",
				WorkloadType:        "interactive",
				CPUCores:            1000, // Extreme value
				MemoryGB:            10000.0,
				StorageGB:           100000.0,
				NetworkIntensive:    true,
				ExpectedConnections: 10000,
				DataTransferPattern: "burst",
				PeakHours:           []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
				HistoricalBandwidth: 10000.0,
			},
			TimeHorizonHours:   168, // 1 week
			ConfidenceLevel:    0.99,
			IncludeUncertainty: true,
		}

		prediction, err := predictor.PredictBandwidth(ctx, request)
		require.NoError(t, err, "Should handle extreme values gracefully")
		assert.NotNil(t, prediction)
		assert.Greater(t, prediction.PredictedBandwidth, 0.0)
	})
}