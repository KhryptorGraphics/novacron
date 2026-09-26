package healing

import (
	"testing"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/orchestration/events"
	"github.com/sirupsen/logrus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestHealthCheckLoopSkipsNilSource(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.ErrorLevel)
	eventBus := events.NewNATSEventBus(logger)
	hc := NewDefaultHealingController(logger, eventBus)

	// Register a target but DO NOT set a health source
	target := &HealingTarget{
		ID:      "test-target",
		Enabled: true,
		HealthCheckConfig: &HealthCheckConfig{
			Interval:  1 * time.Second,
			Timeout:   5 * time.Second,
			CheckType: HealthCheckTypeMetrics,
		},
	}
	require.NoError(t, hc.RegisterTarget(target))

	// Start monitoring - should not panic even with nil source
	require.NoError(t, hc.StartMonitoring())
	defer hc.StopMonitoring()

	// Wait for one healthCheckLoop tick
	time.Sleep(50 * time.Millisecond)

	// Should not panic, should just skip
	assert.NoError(t, hc.StopMonitoring())
}

func TestStrategyRecoverReturnsError(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.ErrorLevel)

	strategies := map[string]RecoveryStrategy{
		"restart":  NewRestartRecoveryStrategy(logger),
		"migrate":  NewMigrateRecoveryStrategy(logger),
		"scale":    NewScaleRecoveryStrategy(logger),
		"failover": NewFailoverRecoveryStrategy(logger),
	}

	for name, strategy := range strategies {
		t.Run(name, func(t *testing.T) {
			result, err := strategy.Recover(&FailureInfo{TargetID: "test"}, &HealingTarget{ID: "test"})
			require.Error(t, err, "strategy %s should return error", name)
			assert.False(t, result.Success, "strategy %s should not report success", name)
		})
	}
}

func TestConsiderHealingRespectsMaxAttempts(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.ErrorLevel)
	eventBus := events.NewNATSEventBus(logger)
	hc := NewDefaultHealingController(logger, eventBus)

	target := &HealingTarget{
		ID:      "test-target",
		Enabled: true,
		HealthCheckConfig: &HealthCheckConfig{
			Interval:         1 * time.Second,
			Timeout:          5 * time.Second,
			CheckType:        HealthCheckTypeMetrics,
			FailureThreshold: 1,
		},
		RecoveryConfig: &RecoveryConfig{
			EnableAutoRecovery:  true,
			MaxRecoveryAttempts: 2,
		},
	}
	require.NoError(t, hc.RegisterTarget(target))

	// Directly test considerHealing's max-attempts gate (same package, so accessible)
	status := &HealthStatus{
		TargetID:            target.ID,
		Healthy:             false,
		ConsecutiveFailures: 1,
		RecoveryStatus: &RecoveryStatus{
			Attempts: 2,
		},
	}
	assessment := &HealthAssessment{
		TargetID: target.ID,
		Healthy:  false,
	}

	err := hc.considerHealing(target, status, assessment)
	require.NoError(t, err)

	// Attempts should remain 2 (not incremented) because max is 2
	assert.Equal(t, 2, status.RecoveryStatus.Attempts)
}
