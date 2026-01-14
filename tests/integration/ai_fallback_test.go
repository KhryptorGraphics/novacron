package integration

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/migration"
	"github.com/khryptorgraphics/novacron/backend/core/scheduler"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
)

// MockFailingAIProvider simulates an AI provider that always fails
type MockFailingAIProvider struct {
	mock.Mock
	failureDelay time.Duration
}

func NewMockFailingAIProvider(delay time.Duration) *MockFailingAIProvider {
	return &MockFailingAIProvider{failureDelay: delay}
}

func (m *MockFailingAIProvider) PredictResourceDemand(nodeID string, resourceType scheduler.ResourceType, horizonMinutes int) ([]float64, float64, error) {
	if m.failureDelay > 0 {
		time.Sleep(m.failureDelay)
	}
	return nil, 0, errors.New("AI service unavailable")
}

func (m *MockFailingAIProvider) OptimizePerformance(clusterData map[string]interface{}, goals []string) (map[string]interface{}, error) {
	if m.failureDelay > 0 {
		time.Sleep(m.failureDelay)
	}
	return nil, errors.New("AI service unavailable")
}

func (m *MockFailingAIProvider) AnalyzeWorkload(vmID string, workloadData []map[string]interface{}) (map[string]interface{}, error) {
	return nil, errors.New("AI service unavailable")
}

func (m *MockFailingAIProvider) DetectAnomalies(metrics map[string]float64) (bool, float64, []string, error) {
	return false, 0, nil, errors.New("AI service unavailable")
}

func (m *MockFailingAIProvider) GetScalingRecommendations(vmID string, currentResources map[string]float64, historicalData []map[string]interface{}) ([]map[string]interface{}, error) {
	return nil, errors.New("AI service unavailable")
}

func (m *MockFailingAIProvider) OptimizeMigration(vmID string, sourceHost string, targetHosts []string, vmMetrics map[string]float64) (map[string]interface{}, error) {
	return nil, errors.New("AI service unavailable")
}

func (m *MockFailingAIProvider) OptimizeBandwidth(networkID string, trafficData []map[string]interface{}, qosRequirements map[string]float64) (map[string]interface{}, error) {
	return nil, errors.New("AI service unavailable")
}

// TestSchedulerAIFallback tests scheduler behavior when AI is unavailable
func TestSchedulerAIFallback(t *testing.T) {
	t.Run("FallbackOnAIFailure", func(t *testing.T) {
		config := scheduler.DefaultSchedulerConfig()
		failingProvider := NewMockFailingAIProvider(0)
		safeProvider := scheduler.NewSafeAIProvider(failingProvider, config)

		// Test resource demand prediction fallback
		predictions, confidence, err := safeProvider.PredictResourceDemand("node1", scheduler.ResourceCPU, 60)
		assert.NoError(t, err)
		assert.NotNil(t, predictions)
		assert.Greater(t, len(predictions), 0)
		assert.Equal(t, 0.6, confidence) // Fallback confidence

		// Test performance optimization fallback
		clusterData := map[string]interface{}{
			"cpu_usage":  0.85,
			"node_count": 5,
		}
		recommendations, err := safeProvider.OptimizePerformance(clusterData, []string{"balance"})
		assert.NoError(t, err)
		assert.NotNil(t, recommendations)
		assert.Contains(t, recommendations, "scale_up")

		// Test anomaly detection fallback
		metrics := map[string]float64{
			"cpu_usage":    0.95,
			"memory_usage": 0.88,
			"error_rate":   0.08,
		}
		isAnomaly, score, recs, err := safeProvider.DetectAnomalies(metrics)
		assert.NoError(t, err)
		assert.True(t, isAnomaly)
		assert.Greater(t, score, 0.0)
		assert.NotEmpty(t, recs)

		// Check metrics
		fallbackMetrics := safeProvider.GetMetrics()
		assert.Equal(t, int64(3), fallbackMetrics["total_calls"])
		assert.Equal(t, int64(3), fallbackMetrics["fallback_calls"])
		assert.Equal(t, int64(3), fallbackMetrics["ai_failures"])
	})

	t.Run("FallbackOnAITimeout", func(t *testing.T) {
		config := scheduler.DefaultSchedulerConfig()
		slowProvider := NewMockFailingAIProvider(10 * time.Second) // Slow response
		safeProvider := scheduler.NewSafeAIProvider(slowProvider, config)

		start := time.Now()
		predictions, confidence, err := safeProvider.PredictResourceDemand("node1", scheduler.ResourceMemory, 30)
		elapsed := time.Since(start)

		assert.NoError(t, err)
		assert.NotNil(t, predictions)
		assert.Less(t, elapsed, 6*time.Second) // Should timeout quickly
		assert.Equal(t, 0.6, confidence)        // Fallback confidence
	})

	t.Run("SchedulerWithoutAI", func(t *testing.T) {
		// Test scheduler can work completely without AI
		config := scheduler.DefaultSchedulerConfig()
		config.NetworkAwarenessEnabled = true

		sched := scheduler.NewScheduler(config)
		assert.NotNil(t, sched)

		// Start scheduler
		err := sched.Start()
		assert.NoError(t, err)

		// Register nodes
		resources := map[scheduler.ResourceType]*scheduler.Resource{
			scheduler.ResourceCPU: {
				Type:     scheduler.ResourceCPU,
				Capacity: 100,
				Used:     20,
			},
			scheduler.ResourceMemory: {
				Type:     scheduler.ResourceMemory,
				Capacity: 16384,
				Used:     4096,
			},
		}
		err = sched.RegisterNode("node1", resources)
		assert.NoError(t, err)

		// Request resources (should work without AI)
		constraints := []scheduler.ResourceConstraint{
			{
				Type:      scheduler.ResourceCPU,
				MinAmount: 10,
				MaxAmount: 50,
			},
		}
		requestID, err := sched.RequestResources(constraints, 5, 1*time.Hour)
		assert.NoError(t, err)
		assert.NotEmpty(t, requestID)

		// Clean up
		err = sched.Stop()
		assert.NoError(t, err)
	})
}

// TestMigrationAIFallback tests migration behavior when AI is unavailable
func TestMigrationAIFallback(t *testing.T) {
	t.Run("MigrationWithoutAI", func(t *testing.T) {
		config := migration.MigrationConfig{
			MaxDowntime:        30 * time.Second,
			TargetTransferRate: 100 * 1024 * 1024, // 100 MB/s
			EnableCompression:  true,
			CompressionLevel:   5,
			MemoryIterations:   3,
		}

		// Create orchestrator without AI
		orchestrator, err := migration.NewLiveMigrationOrchestrator(config)
		assert.NoError(t, err)
		assert.NotNil(t, orchestrator)

		// Migration should still be possible
		ctx := context.Background()
		migrationID, err := orchestrator.MigrateVM(ctx, "vm1", "node1", "node2", migration.MigrationOptions{
			Priority: 5,
		})
		assert.NoError(t, err)
		assert.NotEmpty(t, migrationID)

		// Check metrics (AI should be disabled)
		aiMetrics := orchestrator.GetAIMetrics()
		assert.False(t, aiMetrics["ai_enabled"].(bool))

		// Clean up
		err = orchestrator.Close()
		assert.NoError(t, err)
	})

	t.Run("MigrationFallbackStrategy", func(t *testing.T) {
		config := migration.MigrationConfig{
			MaxDowntime:        30 * time.Second,
			TargetTransferRate: 100 * 1024 * 1024,
		}

		fallback := migration.NewFallbackMigrationStrategy(config)

		// Test migration time prediction
		duration, confidence := fallback.PredictMigrationTime("node1", "node2", "large")
		assert.Greater(t, duration, time.Duration(0))
		assert.Equal(t, 0.6, confidence)

		// Test bandwidth requirements
		bandwidth := fallback.PredictBandwidthRequirements("medium", "normal")
		assert.Greater(t, bandwidth, int64(0))

		// Test migration strategy optimization
		vmData := map[string]interface{}{
			"memory_size":   "large",
			"workload_type": "database",
		}
		networkData := map[string]interface{}{
			"bandwidth": int64(500 * 1024 * 1024), // 500 MB/s
		}
		strategy := fallback.OptimizeMigrationStrategy(vmData, networkData)
		assert.Equal(t, migration.MigrationTypePostCopy, strategy.Type) // Database prefers post-copy
		assert.Equal(t, 0.5, strategy.Confidence)

		// Test anomaly detection
		metrics := map[string]interface{}{
			"transfer_rate": int64(10 * 1024 * 1024), // 10 MB/s (slow)
			"dirty_pages":   int64(10000),
			"iterations":    int32(4),
		}
		anomalies := fallback.DetectAnomalies(metrics)
		assert.NotEmpty(t, anomalies)
		assert.Equal(t, "warning", anomalies[0].Severity)

		// Test dynamic adjustments
		adjustments := fallback.RecommendDynamicAdjustments("migration1", metrics)
		assert.NotEmpty(t, adjustments)
		// Should recommend bandwidth increase due to slow transfer
		found := false
		for _, adj := range adjustments {
			if adj.Parameter == "bandwidth_limit" {
				found = true
				break
			}
		}
		assert.True(t, found)
	})

	t.Run("SafeMigrationProvider", func(t *testing.T) {
		config := migration.MigrationConfig{
			MaxDowntime:        30 * time.Second,
			TargetTransferRate: 100 * 1024 * 1024,
		}

		// Create failing AI provider
		failingProvider := &MockFailingMigrationAIProvider{}
		safeProvider := migration.NewSafeMigrationAIProvider(failingProvider, config)

		// Test all methods fallback correctly
		duration, confidence, err := safeProvider.PredictMigrationTime("node1", "node2", "medium")
		assert.NoError(t, err)
		assert.Greater(t, duration, time.Duration(0))
		assert.Equal(t, 0.6, confidence)

		bandwidth, err := safeProvider.PredictBandwidthRequirements("large", "congested")
		assert.NoError(t, err)
		assert.Greater(t, bandwidth, int64(0))

		path, err := safeProvider.PredictOptimalPath("node1", "node3", map[string]interface{}{})
		assert.NoError(t, err)
		assert.NotEmpty(t, path)

		// Check fallback metrics
		metrics := safeProvider.GetMetrics()
		assert.Greater(t, metrics["fallback_calls"].(int64), int64(0))
		assert.Equal(t, float64(1), metrics["fallback_rate"].(float64))
	})
}

// MockFailingMigrationAIProvider simulates a failing migration AI provider
type MockFailingMigrationAIProvider struct{}

func (m *MockFailingMigrationAIProvider) PredictMigrationTime(sourceNode, destNode, vmSize string) (time.Duration, float64, error) {
	return 0, 0, errors.New("AI service unavailable")
}

func (m *MockFailingMigrationAIProvider) PredictBandwidthRequirements(vmSize, networkConditions string) (int64, error) {
	return 0, errors.New("AI service unavailable")
}

func (m *MockFailingMigrationAIProvider) PredictOptimalPath(sourceNode, destNode string, networkTopology map[string]interface{}) ([]string, error) {
	return nil, errors.New("AI service unavailable")
}

func (m *MockFailingMigrationAIProvider) OptimizeMigrationStrategy(vmData, networkData map[string]interface{}) (migration.MigrationStrategy, error) {
	return migration.MigrationStrategy{}, errors.New("AI service unavailable")
}

func (m *MockFailingMigrationAIProvider) OptimizeCompressionSettings(dataProfile map[string]interface{}) (migration.CompressionConfig, error) {
	return migration.CompressionConfig{}, errors.New("AI service unavailable")
}

func (m *MockFailingMigrationAIProvider) OptimizeMemoryIterations(vmMemoryPattern map[string]interface{}) (int, error) {
	return 0, errors.New("AI service unavailable")
}

func (m *MockFailingMigrationAIProvider) AnalyzeNetworkConditions(nodeID string) (migration.NetworkConditions, error) {
	return migration.NetworkConditions{}, errors.New("AI service unavailable")
}

func (m *MockFailingMigrationAIProvider) DetectAnomalies(migrationMetrics map[string]interface{}) ([]migration.AnomalyAlert, error) {
	return nil, errors.New("AI service unavailable")
}

func (m *MockFailingMigrationAIProvider) RecommendDynamicAdjustments(migrationID string, currentMetrics map[string]interface{}) ([]migration.AdjustmentRecommendation, error) {
	return nil, errors.New("AI service unavailable")
}

func (m *MockFailingMigrationAIProvider) AnalyzeMigrationPatterns(historicalData []migration.MigrationRecord) ([]migration.PatternInsight, error) {
	return nil, errors.New("AI service unavailable")
}

func (m *MockFailingMigrationAIProvider) PredictFailureRisk(migrationParams map[string]interface{}) (float64, error) {
	return 0, errors.New("AI service unavailable")
}

// TestE2EWithoutAI tests end-to-end functionality without AI
func TestE2EWithoutAI(t *testing.T) {
	t.Run("CompleteWorkflowWithoutAI", func(t *testing.T) {
		// Initialize scheduler without AI
		schedConfig := scheduler.DefaultSchedulerConfig()
		sched := scheduler.NewScheduler(schedConfig)
		err := sched.Start()
		assert.NoError(t, err)
		defer sched.Stop()

		// Register multiple nodes
		for i := 1; i <= 3; i++ {
			resources := map[scheduler.ResourceType]*scheduler.Resource{
				scheduler.ResourceCPU: {
					Type:     scheduler.ResourceCPU,
					Capacity: 100,
					Used:     float64(i * 10),
				},
				scheduler.ResourceMemory: {
					Type:     scheduler.ResourceMemory,
					Capacity: 32768,
					Used:     float64(i * 2048),
				},
			}
			err = sched.RegisterNode(fmt.Sprintf("node%d", i), resources)
			assert.NoError(t, err)
		}

		// Request resources multiple times
		for i := 0; i < 5; i++ {
			constraints := []scheduler.ResourceConstraint{
				{
					Type:      scheduler.ResourceCPU,
					MinAmount: 5,
					MaxAmount: 20,
				},
				{
					Type:      scheduler.ResourceMemory,
					MinAmount: 1024,
					MaxAmount: 4096,
				},
			}
			requestID, err := sched.RequestResources(constraints, i+1, 30*time.Minute)
			assert.NoError(t, err)
			assert.NotEmpty(t, requestID)
		}

		// Wait for allocations
		time.Sleep(6 * time.Second)

		// Check allocations were made
		allocations := sched.GetActiveAllocations()
		assert.Greater(t, len(allocations), 0)

		// Initialize migration orchestrator without AI
		migConfig := migration.MigrationConfig{
			MaxDowntime:          30 * time.Second,
			TargetTransferRate:   100 * 1024 * 1024,
			EnableCompression:    true,
			EnableCheckpointing:  true,
			MaxConcurrentMigrations: 2,
		}
		migOrchestrator, err := migration.NewLiveMigrationOrchestrator(migConfig)
		assert.NoError(t, err)
		defer migOrchestrator.Close()

		// Start migrations
		ctx := context.Background()
		var migrationIDs []string
		for i := 0; i < 3; i++ {
			migID, err := migOrchestrator.MigrateVM(ctx,
				fmt.Sprintf("vm%d", i),
				fmt.Sprintf("node%d", (i%3)+1),
				fmt.Sprintf("node%d", ((i+1)%3)+1),
				migration.MigrationOptions{Priority: i + 1})
			assert.NoError(t, err)
			migrationIDs = append(migrationIDs, migID)
		}

		// Check migration statuses
		for _, migID := range migrationIDs {
			status, err := migOrchestrator.GetMigrationStatus(migID)
			assert.NoError(t, err)
			assert.NotNil(t, status)
		}

		// Verify system still functions
		metrics := migOrchestrator.GetMetrics()
		assert.NotNil(t, metrics)
		assert.GreaterOrEqual(t, metrics["total_migrations"].(int64), int64(3))
	})
}