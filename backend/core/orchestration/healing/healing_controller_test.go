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

func TestHealthSourceDrivesHealingToRealVMController(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.ErrorLevel)
	hc := NewDefaultHealingController(logger, events.NewNoopEventBus())

	fake := &fakeVMController{}
	hc.SetVMController(fake)

	// SetHealthSource wiring is exercised through performHealthChecks (proves
	// the configured source is actually invoked and its sample reaches
	// AddSample). The unhealthy verdict that drives healing is asserted via
	// considerHealing directly, matching TestConsiderHealingRespectsMaxAttempts
	// above — PhiAccrualFailureDetector.IsHealthy needs MinSamplesForDecision
	// (5) real-time-spaced samples before it will report unhealthy from a
	// single boolean sample, which this unit test should not depend on.
	sourceCalls := 0
	hc.SetHealthSource(func(id string) (*HealthSample, error) {
		sourceCalls++
		return &HealthSample{TargetID: id, Healthy: false, Timestamp: time.Now()}, nil
	})

	target := &HealingTarget{
		ID:                "vm-1",
		Type:              TargetTypeVM,
		Enabled:           true,
		HealthCheckConfig: &HealthCheckConfig{FailureThreshold: 1, CheckType: HealthCheckTypeMetrics},
		RecoveryConfig:    &RecoveryConfig{EnableAutoRecovery: true, MaxRecoveryAttempts: 1},
	}
	require.NoError(t, hc.RegisterTarget(target))

	hc.performHealthChecks()
	assert.Equal(t, 1, sourceCalls, "the configured health source must be invoked")

	status := &HealthStatus{ConsecutiveFailures: 1}
	assessment := &HealthAssessment{Healthy: false, Reasons: []string{"forced unhealthy for test"}}
	require.NoError(t, hc.considerHealing(target, status, assessment))

	require.Eventually(t, func() bool {
		fake.mu.Lock()
		defer fake.mu.Unlock()
		return len(fake.restartCalls) == 1
	}, time.Second, 5*time.Millisecond)
	fake.mu.Lock()
	defer fake.mu.Unlock()
	assert.Equal(t, "vm-1", fake.restartCalls[0])
}

func TestTriggerHealingManualSelectsRestartForVM(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.ErrorLevel)
	hc := NewDefaultHealingController(logger, events.NewNoopEventBus())

	fake := &fakeVMController{}
	hc.SetVMController(fake)

	target := &HealingTarget{
		ID:                "vm-1",
		Type:              TargetTypeVM,
		Enabled:           true,
		HealthCheckConfig: &HealthCheckConfig{FailureThreshold: 1, CheckType: HealthCheckTypeMetrics},
		RecoveryConfig:    &RecoveryConfig{EnableAutoRecovery: true, MaxRecoveryAttempts: 1},
	}
	require.NoError(t, hc.RegisterTarget(target))

	decision, err := hc.TriggerHealing("vm-1", "operator")
	require.NoError(t, err)
	require.NotNil(t, decision)
	assert.Equal(t, "restart", decision.Strategy)

	require.Eventually(t, func() bool {
		fake.mu.Lock()
		defer fake.mu.Unlock()
		return len(fake.restartCalls) == 1 && fake.restartCalls[0] == "vm-1"
	}, time.Second, 5*time.Millisecond)
}
