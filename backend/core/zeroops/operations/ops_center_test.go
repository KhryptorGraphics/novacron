package operations

import (
	"testing"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/zeroops"
)

func TestAutonomousOpsCenter(t *testing.T) {
	config := zeroops.DefaultZeroOpsConfig()
	opsCenter := NewAutonomousOpsCenter(config)

	// Test start/stop
	err := opsCenter.Start()
	if err != nil {
		t.Fatalf("Failed to start ops center: %v", err)
	}

	time.Sleep(100 * time.Millisecond)

	err = opsCenter.Stop()
	if err != nil {
		t.Fatalf("Failed to stop ops center: %v", err)
	}
}

func TestIncidentHandling(t *testing.T) {
	config := zeroops.DefaultZeroOpsConfig()
	opsCenter := NewAutonomousOpsCenter(config)

	incident := &Incident{
		ID:          "test-001",
		DetectedAt:  time.Now(),
		Severity:    zeroops.SeverityP2,
		Type:        "high_cpu",
		Description: "CPU utilization > 90%",
		Affected:    []string{"vm-001", "vm-002"},
		Metrics:     map[string]float64{"cpu": 0.95},
	}

	opsCenter.handleIncident(incident)

	metrics := opsCenter.GetMetrics()
	if metrics.TotalDecisions == 0 {
		t.Error("Expected decisions to be recorded")
	}
}

func TestDecisionEngine(t *testing.T) {
	config := zeroops.DefaultZeroOpsConfig()
	engine := NewDecisionEngine(config)

	incident := &Incident{
		ID:          "test-002",
		DetectedAt:  time.Now(),
		Severity:    zeroops.SeverityP3,
		Type:        "memory_leak",
		Description: "Memory usage increasing",
	}

	decision := engine.DecideAction(incident)

	if decision == nil {
		t.Fatal("Expected decision, got nil")
	}

	if decision.Confidence < 0 || decision.Confidence > 1 {
		t.Errorf("Invalid confidence: %f", decision.Confidence)
	}
}

func TestMTTDTarget(t *testing.T) {
	config := zeroops.DefaultZeroOpsConfig()
	opsCenter := NewAutonomousOpsCenter(config)

	start := time.Now()
	incident := &Incident{
		ID:         "test-003",
		DetectedAt: start,
		Severity:   zeroops.SeverityP1,
	}

	opsCenter.handleIncident(incident)

	metrics := opsCenter.GetMetrics()
	if metrics.AverageMTTD > 10.0 {
		t.Errorf("MTTD %.2fs exceeds target (10s)", metrics.AverageMTTD)
	}
}

func TestMTTRTarget(t *testing.T) {
	config := zeroops.DefaultZeroOpsConfig()
	opsCenter := NewAutonomousOpsCenter(config)

	incident := &Incident{
		ID:         "test-004",
		DetectedAt: time.Now().Add(-5 * time.Second),
		Severity:   zeroops.SeverityP1,
		Type:       "service_down",
	}

	start := time.Now()
	opsCenter.handleIncident(incident)
	duration := time.Since(start)

	if duration > 60*time.Second {
		t.Errorf("MTTR %v exceeds target (60s)", duration)
	}
}

func TestHumanInterventionRate(t *testing.T) {
	config := zeroops.DefaultZeroOpsConfig()
	config.HumanApproval = false // Zero human intervention
	opsCenter := NewAutonomousOpsCenter(config)

	// Simulate 1000 incidents
	for i := 0; i < 1000; i++ {
		incident := &Incident{
			ID:         string(rune(i)),
			DetectedAt: time.Now(),
			Severity:   zeroops.SeverityP3,
			Type:       "minor_issue",
		}
		opsCenter.handleIncident(incident)
	}

	metrics := opsCenter.GetMetrics()
	if metrics.HumanInterventionRate > 0.001 {
		t.Errorf("Human intervention rate %.4f exceeds target (0.001)", metrics.HumanInterventionRate)
	}
}

func TestAutomationSuccessRate(t *testing.T) {
	config := zeroops.DefaultZeroOpsConfig()
	opsCenter := NewAutonomousOpsCenter(config)

	opsCenter.metrics.RecordMTTR(30 * time.Second)
	opsCenter.metrics.totalDecisions = 1000
	opsCenter.metrics.humanInterventions = 1

	metrics := opsCenter.GetMetrics()
	if metrics.AutomationSuccessRate < 0.999 {
		t.Errorf("Automation success rate %.4f below target (0.999)", metrics.AutomationSuccessRate)
	}
}
