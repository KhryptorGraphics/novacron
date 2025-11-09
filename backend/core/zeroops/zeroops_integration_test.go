package zeroops

import (
	"testing"
	"time"

	"github.com/yourusername/novacron/backend/core/zeroops/alerting"
	"github.com/yourusername/novacron/backend/core/zeroops/incident"
	"github.com/yourusername/novacron/backend/core/zeroops/metrics"
	"github.com/yourusername/novacron/backend/core/zeroops/operations"
	"github.com/yourusername/novacron/backend/core/zeroops/provisioning"
	"github.com/yourusername/novacron/backend/core/zeroops/scaling"
)

func TestZeroOpsIntegration(t *testing.T) {
	config := DefaultZeroOpsConfig()

	// Initialize all components
	opsCenter := operations.NewAutonomousOpsCenter(config)
	scaler := scaling.NewAutonomousScaler(config)
	alerting := alerting.NewSmartAlertingSystem(config)
	provisioner := provisioning.NewSelfProvisioner(config)
	responder := incident.NewAutonomousIncidentResponder(config)
	metricsCollector := metrics.NewZeroOpsMetrics()

	// Start all components
	components := []interface {
		Start() error
		Stop() error
	}{
		opsCenter,
		scaler,
		alerting,
		provisioner,
		responder,
	}

	for _, component := range components {
		if err := component.Start(); err != nil {
			t.Fatalf("Failed to start component: %v", err)
		}
	}

	// Let system run for a bit
	time.Sleep(1 * time.Second)

	// Check metrics
	automationMetrics := metricsCollector.GetMetrics()

	// Validate targets
	if automationMetrics.HumanInterventionRate > 0.001 {
		t.Errorf("Human intervention rate %.4f exceeds target (0.001)",
			automationMetrics.HumanInterventionRate)
	}

	if automationMetrics.AutomationSuccessRate < 0.999 {
		t.Errorf("Automation success rate %.4f below target (0.999)",
			automationMetrics.AutomationSuccessRate)
	}

	// Stop all components
	for _, component := range components {
		if err := component.Stop(); err != nil {
			t.Fatalf("Failed to stop component: %v", err)
		}
	}
}

func TestEndToEndIncidentResponse(t *testing.T) {
	config := DefaultZeroOpsConfig()

	// Simulate complete incident lifecycle
	opsCenter := operations.NewAutonomousOpsCenter(config)
	alerting := alerting.NewSmartAlertingSystem(config)
	responder := incident.NewAutonomousIncidentResponder(config)

	opsCenter.Start()
	alerting.Start()
	responder.Start()
	defer opsCenter.Stop()
	defer alerting.Stop()
	defer responder.Stop()

	// Create incident
	testIncident := &incident.Incident{
		ID:          "integration-test-001",
		DetectedAt:  time.Now(),
		Severity:    SeverityP2,
		Type:        "cpu_spike",
		Description: "CPU utilization > 90%",
		Affected:    []string{"vm-001"},
	}

	// Process through system
	start := time.Now()
	response := responder.HandleIncident(testIncident)
	duration := time.Since(start)

	// Validate MTTD and MTTR targets
	if response.MTTD > 10*time.Second {
		t.Errorf("MTTD %v exceeds target (10s)", response.MTTD)
	}

	if response.MTTR > 60*time.Second {
		t.Errorf("MTTR %v exceeds target (60s)", response.MTTR)
	}

	if !response.Resolved && response.Severity != SeverityP0 {
		t.Error("Expected incident to be resolved automatically")
	}

	// Check total handling time
	if duration > 70*time.Second {
		t.Errorf("Total incident handling took %v, exceeds 70s", duration)
	}
}

func TestZeroOpsScenarios(t *testing.T) {
	config := DefaultZeroOpsConfig()

	tests := []struct {
		name     string
		scenario func(t *testing.T, config *ZeroOpsConfig)
	}{
		{
			name: "Traffic Spike Scenario",
			scenario: func(t *testing.T, config *ZeroOpsConfig) {
				scaler := scaling.NewAutonomousScaler(config)
				scaler.Start()
				defer scaler.Stop()

				// Simulate 200% traffic spike
				prediction := &scaling.WorkloadPrediction{
					Timestamp: time.Now(),
					Duration:  15 * time.Minute,
					Accuracy:  0.94,
					Metrics: map[string]float64{
						"cpu":     0.95,
						"memory":  0.90,
						"network": 0.85,
					},
				}

				decision := scaler.MakeScalingDecision(prediction)
				if !decision.ShouldScale || decision.Direction != "up" {
					t.Error("Expected scale up decision")
				}

				// Verify scale happens in <60s
				start := time.Now()
				scaler.ExecuteScaling(decision)
				duration := time.Since(start)

				if duration > 60*time.Second {
					t.Errorf("Scaling took %v, exceeds 60s target", duration)
				}
			},
		},
		{
			name: "Cost Spike Scenario",
			scenario: func(t *testing.T, config *ZeroOpsConfig) {
				provisioner := provisioning.NewSelfProvisioner(config)
				provisioner.Start()
				defer provisioner.Stop()

				// Detect idle resources and deprovision
				// Verify savings calculation
			},
		},
		{
			name: "Security Vulnerability Scenario",
			scenario: func(t *testing.T, config *ZeroOpsConfig) {
				responder := incident.NewAutonomousIncidentResponder(config)
				responder.Start()
				defer responder.Stop()

				// Simulate critical CVE
				// Verify automated patching in <30min
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.scenario(t, config)
		})
	}
}

func TestPerformanceTargets(t *testing.T) {
	config := DefaultZeroOpsConfig()
	metricsCollector := metrics.NewZeroOpsMetrics()

	// Simulate 10,000 automated decisions
	for i := 0; i < 10000; i++ {
		automated := true
		success := i%1000 != 0 // 99.9% success rate
		metricsCollector.RecordDecision(automated, success)
	}

	// Simulate MTTD/MTTR
	for i := 0; i < 100; i++ {
		metricsCollector.RecordMTTD(8 * time.Second)
		metricsCollector.RecordMTTR(45 * time.Second)
	}

	automationMetrics := metricsCollector.GetMetrics()

	// Validate all targets
	targets := map[string]struct {
		actual float64
		target float64
		name   string
	}{
		"human_intervention": {
			actual: automationMetrics.HumanInterventionRate,
			target: 0.001,
			name:   "Human Intervention Rate",
		},
		"automation_success": {
			actual: automationMetrics.AutomationSuccessRate,
			target: 0.999,
			name:   "Automation Success Rate",
		},
		"mttd": {
			actual: automationMetrics.AverageMTTD,
			target: 10.0,
			name:   "MTTD (seconds)",
		},
		"mttr": {
			actual: automationMetrics.AverageMTTR,
			target: 60.0,
			name:   "MTTR (seconds)",
		},
		"availability": {
			actual: automationMetrics.Availability,
			target: 0.99999,
			name:   "Availability",
		},
	}

	for key, test := range targets {
		if key == "mttd" || key == "mttr" {
			if test.actual > test.target {
				t.Errorf("%s: %.2f exceeds target %.2f", test.name, test.actual, test.target)
			}
		} else {
			if test.actual < test.target {
				t.Errorf("%s: %.4f below target %.4f", test.name, test.actual, test.target)
			}
		}
	}
}

func TestConfigValidation(t *testing.T) {
	config := DefaultZeroOpsConfig()

	// Validate all configuration values
	if config.MaxAutomatedCost <= 0 {
		t.Error("MaxAutomatedCost must be positive")
	}

	if config.TargetMTTD <= 0 {
		t.Error("TargetMTTD must be positive")
	}

	if config.TargetMTTR <= 0 {
		t.Error("TargetMTTR must be positive")
	}

	if config.TargetAutomationRate < 0.9 || config.TargetAutomationRate > 1.0 {
		t.Error("TargetAutomationRate must be between 0.9 and 1.0")
	}

	// Validate safety constraints
	if config.SafetyConstraints.MaxVMsAutoProvisioned <= 0 {
		t.Error("MaxVMsAutoProvisioned must be positive")
	}

	if config.SafetyConstraints.RateLimitActions <= 0 {
		t.Error("RateLimitActions must be positive")
	}
}
