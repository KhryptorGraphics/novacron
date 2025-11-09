package healing

import (
	"context"
	"testing"
	"time"

	"github.com/novacron/backend/core/autonomous"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
)

// TestHealingEngine tests the autonomous healing engine
func TestHealingEngine(t *testing.T) {
	logger := zap.NewNop()
	config := &autonomous.AutonomousConfig{
		EnableHealing:      true,
		HealingInterval:    100 * time.Millisecond,
		SubSecondDetection: true,
		MaxHealingAttempts: 3,
		HealingTimeout:     30 * time.Second,
	}

	engine := NewHealingEngine(config, logger)
	assert.NotNil(t, engine)

	t.Run("SubSecondFaultDetection", func(t *testing.T) {
		ctx := context.Background()
		start := time.Now()

		// Register mock health checker
		mockChecker := &MockHealthChecker{
			component: "test-service",
			healthy:   false,
		}
		engine.faultDetector.RegisterHealthChecker("test-service", mockChecker)

		// Detect faults
		faults := engine.faultDetector.Detect(ctx)

		// Verify sub-second detection
		duration := time.Since(start)
		assert.Less(t, duration, 1*time.Second, "Fault detection should be sub-second")
		assert.NotEmpty(t, faults, "Should detect unhealthy service")
	})

	t.Run("AutomaticHealing", func(t *testing.T) {
		ctx := context.Background()

		// Create a fault
		fault := &Fault{
			ID:          "test-fault-1",
			Type:        "service_failure",
			Component:   "api-server",
			Description: "Service not responding",
			Severity:    0.8,
			Timestamp:   time.Now(),
		}

		// Handle fault
		engine.handleFault(ctx, fault)

		// Check healing history
		history := engine.GetHealingHistory()
		assert.NotEmpty(t, history, "Should have healing events in history")

		// Verify healing success
		if len(history) > 0 {
			lastEvent := history[len(history)-1]
			assert.Equal(t, HealingSuccess, lastEvent.Status, "Healing should succeed")
		}
	})

	t.Run("HealingSuccessRate", func(t *testing.T) {
		// Simulate multiple healing events
		for i := 0; i < 10; i++ {
			event := &HealingEvent{
				ID:        fmt.Sprintf("heal-%d", i),
				Success:   i < 9, // 90% success rate
				Timestamp: time.Now(),
				Duration:  time.Duration(i) * time.Second,
			}
			engine.healingHistory = append(engine.healingHistory, event)
		}

		successRate := engine.GetSuccessRate()
		assert.Equal(t, 0.9, successRate, "Success rate should be 90%")
	})

	t.Run("PredictiveMaintenance", func(t *testing.T) {
		ctx := context.Background()

		// Get predictions
		predictions := engine.predictiveEngine.Predict(ctx)
		assert.NotEmpty(t, predictions, "Should generate predictions")

		// Verify prediction quality
		for _, pred := range predictions {
			assert.Greater(t, pred.Probability, 0.5, "Should only return high-probability predictions")
			assert.Less(t, pred.TimeUntil, 72*time.Hour, "Predictions should be within 72h horizon")
		}
	})
}

// TestFaultDetector tests the fault detection system
func TestFaultDetector(t *testing.T) {
	logger := zap.NewNop()
	detector := NewFaultDetector(logger)

	t.Run("AnomalyDetection", func(t *testing.T) {
		// Train anomaly detector with normal values
		for i := 0; i < 100; i++ {
			detector.anomalyDetector.IsAnomaly("cpu_usage", 0.5+float64(i%10)*0.01)
		}

		// Test anomaly detection
		isAnomaly := detector.anomalyDetector.IsAnomaly("cpu_usage", 0.95)
		assert.True(t, isAnomaly, "Should detect high CPU usage as anomaly")

		isNormal := detector.anomalyDetector.IsAnomaly("cpu_usage", 0.52)
		assert.False(t, isNormal, "Should not flag normal values as anomaly")
	})

	t.Run("CascadingFailureDetection", func(t *testing.T) {
		// Simulate multiple component failures
		for i := 0; i < 5; i++ {
			fault := &Fault{
				Component: fmt.Sprintf("service-%d", i),
				Type:      "failure",
			}
			detector.alertManager.ProcessFault(fault)
		}

		hasCascading := detector.alertManager.HasCascadingFailures()
		assert.True(t, hasCascading, "Should detect cascading failures")
	})

	t.Run("ThunderingHerdDetection", func(t *testing.T) {
		// Simulate many alerts in short time
		for i := 0; i < 25; i++ {
			fault := &Fault{
				Component: "api",
				Type:      "overload",
			}
			detector.alertManager.ProcessFault(fault)
		}

		hasThundering := detector.alertManager.HasThunderingHerd()
		assert.True(t, hasThundering, "Should detect thundering herd")
	})
}

// TestRootCauseAnalyzer tests root cause analysis
func TestRootCauseAnalyzer(t *testing.T) {
	logger := zap.NewNop()
	analyzer := NewRootCauseAnalyzer(logger)

	t.Run("AnalyzeRootCause", func(t *testing.T) {
		fault := &Fault{
			Type:      "service_failure",
			Component: "database",
			Metrics: map[string]float64{
				"cpu_usage":    0.95,
				"memory_usage": 0.89,
				"connections":  1000,
			},
		}

		ctx := context.Background()
		rootCause := analyzer.Analyze(ctx, fault)

		assert.NotNil(t, rootCause)
		assert.NotEmpty(t, rootCause.Cause)
		assert.Greater(t, rootCause.Confidence, 0.5)
		assert.NotEmpty(t, rootCause.Evidence)
	})
}

// TestRemediator tests remediation actions
func TestRemediator(t *testing.T) {
	logger := zap.NewNop()
	remediator := NewRemediator(logger)

	t.Run("ExecuteRemediation", func(t *testing.T) {
		ctx := context.Background()
		rootCause := &RootCause{
			Component: "vm-1",
			Cause:     "resource_exhaustion",
			Impact:    ImpactHigh,
		}

		err := remediator.Execute(ctx, autonomous.ScaleOut, rootCause)
		assert.NoError(t, err, "Remediation should execute successfully")
	})

	t.Run("RollbackSupport", func(t *testing.T) {
		ctx := context.Background()

		// Execute action
		remediator.Execute(ctx, autonomous.ConfigRollback, &RootCause{})

		// Attempt rollback
		err := remediator.Rollback(ctx, autonomous.ConfigRollback)
		assert.NoError(t, err, "Should support rollback")
	})
}

// Benchmark tests
func BenchmarkFaultDetection(b *testing.B) {
	logger := zap.NewNop()
	detector := NewFaultDetector(logger)
	ctx := context.Background()

	// Register multiple health checkers
	for i := 0; i < 10; i++ {
		checker := &MockHealthChecker{
			component: fmt.Sprintf("service-%d", i),
			healthy:   true,
		}
		detector.RegisterHealthChecker(checker.component, checker)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		detector.Detect(ctx)
	}
}

func BenchmarkAnomalyDetection(b *testing.B) {
	logger := zap.NewNop()
	detector := NewAnomalyDetector(logger)

	// Pre-train with normal data
	for i := 0; i < 100; i++ {
		detector.IsAnomaly("metric", 0.5)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		detector.IsAnomaly("metric", 0.5+float64(i%100)*0.01)
	}
}

// Mock implementations for testing

type MockHealthChecker struct {
	component string
	healthy   bool
}

func (m *MockHealthChecker) Check(ctx context.Context) (*HealthStatus, error) {
	return &HealthStatus{
		Healthy:   m.healthy,
		Component: m.component,
		Message:   "Mock health check",
		Metrics: map[string]float64{
			"cpu_usage":    0.5,
			"memory_usage": 0.6,
		},
	}, nil
}

func (m *MockHealthChecker) GetComponent() string {
	return m.component
}

// Integration test for end-to-end healing
func TestEndToEndHealing(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test")
	}

	logger := zap.NewNop()
	config := autonomous.DefaultConfig()
	engine := NewHealingEngine(config, logger)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Start healing engine
	err := engine.Start(ctx)
	require.NoError(t, err)

	// Simulate service failure
	mockChecker := &MockHealthChecker{
		component: "critical-service",
		healthy:   false,
	}
	engine.faultDetector.RegisterHealthChecker("critical-service", mockChecker)

	// Wait for healing
	time.Sleep(2 * time.Second)

	// Verify healing occurred
	history := engine.GetHealingHistory()
	assert.NotEmpty(t, history, "Should have attempted healing")

	// Check success rate
	successRate := engine.GetSuccessRate()
	assert.Greater(t, successRate, 0.5, "Success rate should be reasonable")
}

// Test fault injection for chaos engineering
func TestFaultInjection(t *testing.T) {
	logger := zap.NewNop()
	config := autonomous.DefaultConfig()
	engine := NewHealingEngine(config, logger)

	scenarios := []struct {
		name      string
		faultType string
		severity  float64
		expected  autonomous.HealingAction
	}{
		{
			name:      "VMFailure",
			faultType: "vm_failure",
			severity:  0.9,
			expected:  autonomous.VMMigration,
		},
		{
			name:      "NetworkPartition",
			faultType: "network_partition",
			severity:  0.8,
			expected:  autonomous.NetworkReroute,
		},
		{
			name:      "ResourceExhaustion",
			faultType: "resource_exhaustion",
			severity:  0.7,
			expected:  autonomous.ScaleOut,
		},
		{
			name:      "PerformanceDegradation",
			faultType: "performance_degradation",
			severity:  0.5,
			expected:  autonomous.PerformanceTune,
		},
	}

	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			rootCause := &RootCause{
				Cause:  scenario.faultType,
				Impact: ImpactLevel(scenario.severity * 5),
			}

			action := engine.selectHealingAction(rootCause)
			assert.Equal(t, scenario.expected, action,
				"Should select appropriate healing action for %s", scenario.faultType)
		})
	}
}

// Missing imports that need to be added to the file
import (
	"context"
	"fmt"
)