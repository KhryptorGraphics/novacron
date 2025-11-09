package compliance

import (
	"context"
	"testing"
	"time"
)

func TestComplianceFramework(t *testing.T) {
	ctx := context.Background()

	t.Run("Initialize Framework", func(t *testing.T) {
		standards := []string{"soc2", "iso27001", "hipaa"}
		cf := NewComplianceFramework(standards, 24*time.Hour)

		if len(cf.controls) == 0 {
			t.Error("Expected controls to be initialized")
		}

		if len(cf.enabledStandards) != 3 {
			t.Errorf("Expected 3 standards, got %d", len(cf.enabledStandards))
		}
	})

	t.Run("Assess Control", func(t *testing.T) {
		cf := NewComplianceFramework([]string{"soc2"}, 24*time.Hour)

		// Get first control
		var controlID string
		for id := range cf.controls {
			controlID = id
			break
		}

		control, err := cf.AssessControl(ctx, controlID)
		if err != nil {
			t.Fatalf("Failed to assess control: %v", err)
		}

		if control.Status == "" {
			t.Error("Expected control status to be set")
		}

		if control.LastAssessed.IsZero() {
			t.Error("Expected LastAssessed to be set")
		}
	})

	t.Run("Generate Compliance Report", func(t *testing.T) {
		cf := NewComplianceFramework([]string{"soc2"}, 24*time.Hour)

		report, err := cf.GenerateComplianceReport(ctx, SOC2Type2)
		if err != nil {
			t.Fatalf("Failed to generate report: %v", err)
		}

		if report.Standard != SOC2Type2 {
			t.Errorf("Expected standard %s, got %s", SOC2Type2, report.Standard)
		}

		if report.TotalControls == 0 {
			t.Error("Expected controls to be assessed")
		}

		if report.OverallScore == 0 {
			t.Error("Expected compliance score to be calculated")
		}
	})

	t.Run("Continuous Monitoring", func(t *testing.T) {
		cf := NewComplianceFramework([]string{"soc2"}, 24*time.Hour)

		err := cf.StartContinuousMonitoring(ctx)
		if err != nil {
			t.Fatalf("Failed to start monitoring: %v", err)
		}

		// Wait briefly
		time.Sleep(100 * time.Millisecond)

		cf.StopContinuousMonitoring()
	})

	t.Run("Evidence Collection", func(t *testing.T) {
		ec := newEvidenceCollector()

		err := ec.CollectEvidence("CC6.1", "log", "Access log reviewed", "/var/log/access.log")
		if err != nil {
			t.Fatalf("Failed to collect evidence: %v", err)
		}

		if len(ec.evidenceStore["CC6.1"]) != 1 {
			t.Error("Expected evidence to be stored")
		}
	})

	t.Run("Compliance Metrics", func(t *testing.T) {
		cf := NewComplianceFramework([]string{"soc2"}, 24*time.Hour)

		// Perform some assessments
		for id := range cf.controls {
			cf.AssessControl(ctx, id)
			break
		}

		metrics := cf.GetMetrics()

		if metrics.ControlAssessmentCount == 0 {
			t.Error("Expected assessment count to be tracked")
		}
	})

	t.Run("Multiple Standards", func(t *testing.T) {
		standards := []string{"soc2", "iso27001", "hipaa", "pci-dss"}
		cf := NewComplianceFramework(standards, 24*time.Hour)

		// Should have controls from all standards
		hasSOC2 := false
		hasISO := false
		hasHIPAA := false
		hasPCI := false

		for _, control := range cf.controls {
			switch control.Standard {
			case SOC2Type2:
				hasSOC2 = true
			case ISO27001:
				hasISO = true
			case HIPAA:
				hasHIPAA = true
			case PCIDSS:
				hasPCI = true
			}
		}

		if !hasSOC2 || !hasISO || !hasHIPAA || !hasPCI {
			t.Error("Expected controls from all standards")
		}
	})

	t.Run("Compliance Target", func(t *testing.T) {
		cf := NewComplianceFramework([]string{"soc2"}, 24*time.Hour)
		cf.complianceTarget = 0.95

		report, _ := cf.GenerateComplianceReport(ctx, SOC2Type2)

		status := cf.getComplianceStatus(report.OverallScore)
		if status == "" {
			t.Error("Expected compliance status to be calculated")
		}
	})
}

func TestComplianceAutomation(t *testing.T) {
	ctx := context.Background()

	t.Run("Automated Assessment", func(t *testing.T) {
		cf := NewComplianceFramework([]string{"soc2"}, 24*time.Hour)

		control := &ComplianceControl{
			ID:       "TEST-1",
			Standard: SOC2Type2,
			Category: "encryption",
		}

		status, evidence := cf.performAutomatedAssessment(ctx, control)

		if status == "" {
			t.Error("Expected status to be returned")
		}

		if len(evidence) == 0 {
			t.Error("Expected evidence to be collected")
		}
	})

	t.Run("Auto Remediation", func(t *testing.T) {
		cf := NewComplianceFramework([]string{"soc2"}, 24*time.Hour)
		cf.autoRemediation = true

		re := cf.remediationEngine
		re.mu.Lock()
		re.remediationRules["TEST-1"] = RemediationRule{
			ControlID:   "TEST-1",
			Condition:   "status == non-compliant",
			Actions:     []string{"enable-encryption", "enable-audit-logging"},
			AutoExecute: true,
		}
		re.mu.Unlock()

		control := &ComplianceControl{
			ID:     "TEST-1",
			Status: ControlNonCompliant,
		}

		err := re.Remediate(control)
		if err != nil {
			t.Fatalf("Failed to remediate: %v", err)
		}
	})
}

func BenchmarkComplianceAssessment(b *testing.B) {
	ctx := context.Background()
	cf := NewComplianceFramework([]string{"soc2"}, 24*time.Hour)

	var controlID string
	for id := range cf.controls {
		controlID = id
		break
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cf.AssessControl(ctx, controlID)
	}
}
