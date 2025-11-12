// Package compliance_test provides comprehensive compliance testing
package compliance_test

import (
	"context"
	"testing"
	"time"

	"novacron/backend/core/compliance"
	"novacron/backend/core/compliance/audit"
	"novacron/backend/core/compliance/frameworks"
	"novacron/backend/core/compliance/policy"
	"novacron/backend/core/compliance/posture"
	"novacron/backend/core/governance"
)

// TestSOC2Compliance tests SOC2 Type II compliance
func TestSOC2Compliance(t *testing.T) {
	engine := frameworks.NewSOC2Engine()

	t.Run("Initial Assessment", func(t *testing.T) {
		ctx := context.Background()
		report, err := engine.AssessCompliance(ctx)
		if err != nil {
			t.Fatalf("Assessment failed: %v", err)
		}

		// Verify score is within acceptable range
		if report.Score < 90.0 {
			t.Errorf("SOC2 score too low: %.1f%% (expected >= 90%%)", report.Score)
		}

		// Verify all controls are checked
		if len(report.Controls) < 15 {
			t.Errorf("Not all controls checked: %d (expected 15)", len(report.Controls))
		}

		// Verify automated controls
		automatedCount := 0
		controls := engine.GetControls()
		for _, control := range controls {
			if control.Automated {
				automatedCount++
			}
		}

		if automatedCount < 13 {
			t.Errorf("Not enough automated controls: %d (expected >= 13)", automatedCount)
		}

		t.Logf("SOC2 Assessment: Score=%.1f%%, Controls=%d, Automated=%d",
			report.Score, len(report.Controls), automatedCount)
	})

	t.Run("Control Validation", func(t *testing.T) {
		controls := engine.GetControls()

		// Verify critical controls
		criticalControls := []string{
			"SOC2-CC6.1", // Logical Access
			"SOC2-CC6.8", // Encryption
			"SOC2-CC7.2", // Monitoring
		}

		for _, controlID := range criticalControls {
			control, err := engine.GetControl(controlID)
			if err != nil {
				t.Errorf("Critical control missing: %s", controlID)
				continue
			}

			if control.Status == compliance.StatusNonCompliant {
				t.Errorf("Critical control non-compliant: %s", controlID)
			}
		}
	})

	t.Run("Evidence Collection", func(t *testing.T) {
		// Verify evidence is collected for all controls
		controls := engine.GetControls()
		for _, control := range controls {
			if control.Automated && len(control.Evidence) == 0 {
				t.Errorf("No evidence for automated control: %s", control.ID)
			}
		}
	})
}

// TestGDPRCompliance tests GDPR compliance
func TestGDPRCompliance(t *testing.T) {
	engine := frameworks.NewGDPREngine()

	t.Run("Privacy Rights Implementation", func(t *testing.T) {
		ctx := context.Background()

		// Test Right to Access (Art. 15)
		accessReq := &compliance.DataPrivacyRequest{
			Type:        compliance.RequestAccessData,
			SubjectID:   "user-123",
			RequestedBy: "user@example.com",
		}

		err := engine.ProcessPrivacyRequest(ctx, accessReq)
		if err != nil {
			t.Errorf("Failed to process access request: %v", err)
		}

		// Test Right to Erasure (Art. 17)
		erasureReq := &compliance.DataPrivacyRequest{
			Type:        compliance.RequestErasure,
			SubjectID:   "user-456",
			RequestedBy: "user@example.com",
		}

		err = engine.ProcessPrivacyRequest(ctx, erasureReq)
		if err != nil {
			t.Errorf("Failed to process erasure request: %v", err)
		}

		// Verify requests completed
		requests, err := engine.GetPrivacyRequests("completed")
		if err != nil {
			t.Errorf("Failed to get requests: %v", err)
		}

		if len(requests) < 2 {
			t.Errorf("Not all requests completed: %d", len(requests))
		}
	})

	t.Run("Consent Management", func(t *testing.T) {
		ctx := context.Background()

		// Record consent
		consent := &compliance.ConsentRecord{
			SubjectID:      "user-789",
			Purpose:        "marketing",
			ConsentGiven:   true,
			LegalBasis:     "Art. 6(1)(a)",
			DataCategories: []string{"email", "name"},
		}

		err := engine.RecordConsent(ctx, consent)
		if err != nil {
			t.Errorf("Failed to record consent: %v", err)
		}

		// Validate consent
		valid, err := engine.ValidateConsent("user-789", "marketing")
		if err != nil {
			t.Errorf("Failed to validate consent: %v", err)
		}

		if !valid {
			t.Error("Consent should be valid")
		}
	})

	t.Run("Compliance Assessment", func(t *testing.T) {
		ctx := context.Background()
		report, err := engine.AssessCompliance(ctx)
		if err != nil {
			t.Fatalf("GDPR assessment failed: %v", err)
		}

		if report.Score < 90.0 {
			t.Errorf("GDPR score too low: %.1f%%", report.Score)
		}

		t.Logf("GDPR Assessment: Score=%.1f%%, Status=%s",
			report.Score, report.Status)
	})
}

// TestHIPAACompliance tests HIPAA compliance
func TestHIPAACompliance(t *testing.T) {
	engine := frameworks.NewHIPAAEngine()

	t.Run("PHI Access Logging", func(t *testing.T) {
		ctx := context.Background()

		// Log PHI access
		log := &compliance.PHIAccessLog{
			UserID:     "doctor-123",
			UserRole:   "physician",
			PatientID:  "patient-456",
			Action:     "view_record",
			Resource:   "medical_record",
			Purpose:    "treatment",
			Authorized: true,
			IPAddress:  "192.168.1.100",
		}

		err := engine.LogPHIAccess(ctx, log)
		if err != nil {
			t.Errorf("Failed to log PHI access: %v", err)
		}

		// Retrieve logs
		logs, err := engine.GetPHIAccessLogs("patient-456", time.Now().Add(-1*time.Hour))
		if err != nil {
			t.Errorf("Failed to get PHI logs: %v", err)
		}

		if len(logs) == 0 {
			t.Error("PHI access log not found")
		}
	})

	t.Run("BAA Management", func(t *testing.T) {
		ctx := context.Background()

		// Register BAA
		baa := &compliance.BAA{
			EntityName:       "Cloud Provider Inc",
			EntityType:       "business_associate",
			SignedDate:       time.Now().AddDate(0, -6, 0),
			ExpiryDate:       time.Now().AddDate(1, 6, 0),
			ContactName:      "John Doe",
			ContactEmail:     "john@cloudprovider.com",
			Services:         []string{"hosting", "backup"},
			PHICategories:    []string{"medical_records", "billing"},
			SecurityControls: []string{"encryption", "access_control"},
		}

		err := engine.RegisterBAA(ctx, baa)
		if err != nil {
			t.Errorf("Failed to register BAA: %v", err)
		}

		// Check BAA compliance
		valid, err := engine.CheckBAACompliance("Cloud Provider Inc")
		if err != nil {
			t.Errorf("BAA check failed: %v", err)
		}

		if !valid {
			t.Error("BAA should be valid")
		}
	})

	t.Run("Compliance Assessment", func(t *testing.T) {
		ctx := context.Background()
		report, err := engine.AssessCompliance(ctx)
		if err != nil {
			t.Fatalf("HIPAA assessment failed: %v", err)
		}

		if report.Score < 85.0 {
			t.Errorf("HIPAA score too low: %.1f%%", report.Score)
		}

		t.Logf("HIPAA Assessment: Score=%.1f%%, Status=%s",
			report.Score, report.Status)
	})
}

// TestPolicyEngine tests policy-as-code enforcement
func TestPolicyEngine(t *testing.T) {
	engine := policy.NewEngine()

	t.Run("Policy Evaluation", func(t *testing.T) {
		ctx := context.Background()

		// Test MFA enforcement policy
		decision, err := engine.Evaluate(ctx, &compliance.PolicyRequest{
			Principal: "user@company.com",
			Action:    "vm:access",
			Resource:  "vm:production/vm-123",
			Context: map[string]interface{}{
				"mfa_verified": false,
				"environment":  "production",
			},
		})

		if err != nil {
			t.Errorf("Policy evaluation failed: %v", err)
		}

		// Should be denied due to missing MFA
		if decision.Allowed {
			t.Error("Expected access denial for missing MFA")
		}

		// Test with MFA
		decision, err = engine.Evaluate(ctx, &compliance.PolicyRequest{
			Principal: "user@company.com",
			Action:    "vm:access",
			Resource:  "vm:production/vm-123",
			Context: map[string]interface{}{
				"mfa_verified": true,
				"environment":  "production",
			},
		})

		if err != nil {
			t.Errorf("Policy evaluation failed: %v", err)
		}

		if !decision.Allowed {
			t.Errorf("Expected access allowed with MFA: %v", decision.Reasons)
		}
	})

	t.Run("Policy Testing", func(t *testing.T) {
		policies := engine.ListPolicies()

		if len(policies) < 10 {
			t.Errorf("Expected at least 10 default policies, got %d", len(policies))
		}

		// Test each policy has required fields
		for _, policy := range policies {
			if policy.Name == "" {
				t.Error("Policy missing name")
			}
			if len(policy.Rules) == 0 {
				t.Errorf("Policy %s has no rules", policy.Name)
			}
		}
	})

	t.Run("Batch Evaluation", func(t *testing.T) {
		ctx := context.Background()

		requests := []*compliance.PolicyRequest{
			{
				Principal: "user1@company.com",
				Action:    "vm:delete",
				Resource:  "vm:dev/vm-1",
				Context:   map[string]interface{}{"mfa_verified": true},
			},
			{
				Principal: "user2@company.com",
				Action:    "vm:delete",
				Resource:  "vm:prod/vm-2",
				Context:   map[string]interface{}{"mfa_verified": false},
			},
		}

		decisions, err := engine.EvaluateBatch(ctx, requests)
		if err != nil {
			t.Errorf("Batch evaluation failed: %v", err)
		}

		if len(decisions) != len(requests) {
			t.Errorf("Expected %d decisions, got %d", len(requests), len(decisions))
		}
	})
}

// TestGovernanceEngine tests governance automation
func TestGovernanceEngine(t *testing.T) {
	engine := governance.NewEngine()

	t.Run("Resource Tagging", func(t *testing.T) {
		ctx := context.Background()

		// Valid tags
		err := engine.ValidateTags(ctx, "vm-123", "vm", map[string]string{
			"owner":       "engineering",
			"project":     "web-app",
			"environment": "production",
			"cost_center": "CC-1234",
		})

		if err != nil {
			t.Errorf("Valid tags rejected: %v", err)
		}

		// Missing required tags
		err = engine.ValidateTags(ctx, "vm-456", "vm", map[string]string{
			"owner": "engineering",
		})

		if err == nil {
			t.Error("Missing tags should be rejected")
		}
	})

	t.Run("Budget Management", func(t *testing.T) {
		ctx := context.Background()

		// Set budget
		err := engine.SetBudget(ctx, "engineering", 50000.00, compliance.Period{
			Start: time.Now(),
			End:   time.Now().AddDate(0, 1, 0),
		})

		if err != nil {
			t.Errorf("Failed to set budget: %v", err)
		}

		// Get budget alerts
		alerts, err := engine.GetBudgetAlerts(ctx)
		if err != nil {
			t.Errorf("Failed to get budget alerts: %v", err)
		}

		t.Logf("Budget alerts: %d", len(alerts))
	})

	t.Run("Access Reviews", func(t *testing.T) {
		ctx := context.Background()

		// Schedule access review
		err := engine.ScheduleAccessReview(ctx, "production", 90*24*time.Hour)
		if err != nil {
			t.Errorf("Failed to schedule review: %v", err)
		}

		// Get pending reviews
		reviews, err := engine.GetPendingAccessReviews(ctx)
		if err != nil {
			t.Errorf("Failed to get reviews: %v", err)
		}

		if len(reviews) == 0 {
			t.Error("Access review not scheduled")
		}
	})

	t.Run("Auto Remediation", func(t *testing.T) {
		ctx := context.Background()

		// Test auto-remediation
		err := engine.AutoRemediateViolations(ctx, compliance.FrameworkSOC2)
		if err != nil {
			t.Errorf("Auto-remediation failed: %v", err)
		}

		// Check remediation status
		statuses, err := engine.GetRemediationStatus(ctx)
		if err != nil {
			t.Errorf("Failed to get remediation status: %v", err)
		}

		if len(statuses) == 0 {
			t.Error("No remediation records found")
		}
	})
}

// TestAuditBlockchain tests tamper-proof audit logging
func TestAuditBlockchain(t *testing.T) {
	log := audit.NewBlockchainAuditLog()

	t.Run("Event Logging", func(t *testing.T) {
		// Log events
		for i := 0; i < 150; i++ {
			event := &compliance.AuditEvent{
				EventType: "test_event",
				Actor: compliance.Actor{
					ID:   "user-123",
					Type: "user",
					Name: "Test User",
				},
				Action: "test_action",
				Resource: compliance.Resource{
					ID:   "resource-456",
					Type: "test_resource",
				},
				Result:   "success",
				Severity: "info",
			}

			err := log.LogEvent(event)
			if err != nil {
				t.Errorf("Failed to log event: %v", err)
			}
		}

		stats := log.GetChainStats()
		t.Logf("Chain stats: %v", stats)

		// Should have created at least one block (100 events)
		if stats["total_blocks"].(int) < 2 {
			t.Error("Not enough blocks created")
		}
	})

	t.Run("Integrity Verification", func(t *testing.T) {
		valid, errors := log.VerifyIntegrity()
		if !valid {
			t.Errorf("Chain integrity violated: %v", errors)
		}

		t.Log("Chain integrity verified")
	})

	t.Run("Event Query", func(t *testing.T) {
		filter := audit.EventFilter{
			EventType: "test_event",
			Action:    "test_action",
		}

		events, err := log.QueryEvents(filter)
		if err != nil {
			t.Errorf("Query failed: %v", err)
		}

		if len(events) == 0 {
			t.Error("No events found")
		}

		t.Logf("Found %d events", len(events))
	})

	t.Run("Latest Events", func(t *testing.T) {
		events, err := log.GetLatestEvents(10)
		if err != nil {
			t.Errorf("Failed to get latest events: %v", err)
		}

		if len(events) > 10 {
			t.Error("Returned too many events")
		}
	})
}

// TestSecurityPosture tests security posture management
func TestSecurityPosture(t *testing.T) {
	engine := posture.NewEngine()

	t.Run("Posture Assessment", func(t *testing.T) {
		ctx := context.Background()

		currentPosture, err := engine.AssessPosture(ctx)
		if err != nil {
			t.Fatalf("Posture assessment failed: %v", err)
		}

		// Verify score
		if currentPosture.OverallScore < 70.0 {
			t.Errorf("Security score too low: %.1f", currentPosture.OverallScore)
		}

		// Verify risk level is set
		if currentPosture.RiskLevel == "" {
			t.Error("Risk level not set")
		}

		// Verify compliance scores
		if len(currentPosture.Compliance) == 0 {
			t.Error("No compliance scores")
		}

		t.Logf("Security Posture: Score=%.1f, Risk=%s, Vulns=%d",
			currentPosture.OverallScore,
			currentPosture.RiskLevel,
			currentPosture.Vulnerabilities.Total)
	})

	t.Run("Vulnerability Scanning", func(t *testing.T) {
		ctx := context.Background()

		// Run assessment to trigger scans
		_, err := engine.AssessPosture(ctx)
		if err != nil {
			t.Errorf("Scan failed: %v", err)
		}

		// Get critical vulnerabilities
		criticalVulns, err := engine.GetVulnerabilities("critical")
		if err != nil {
			t.Errorf("Failed to get vulnerabilities: %v", err)
		}

		t.Logf("Found %d critical vulnerabilities", len(criticalVulns))

		// Verify vulnerability structure
		for _, vuln := range criticalVulns {
			if vuln.ID == "" || vuln.Title == "" {
				t.Error("Vulnerability missing required fields")
			}
		}
	})

	t.Run("Recommendations", func(t *testing.T) {
		ctx := context.Background()

		posture, err := engine.AssessPosture(ctx)
		if err != nil {
			t.Errorf("Assessment failed: %v", err)
		}

		if len(posture.Recommendations) == 0 {
			t.Log("No recommendations (system is in good shape)")
		} else {
			t.Logf("Generated %d recommendations", len(posture.Recommendations))

			for _, rec := range posture.Recommendations {
				t.Logf("[%s] %s", rec.Priority, rec.Title)
			}
		}
	})
}

// TestIntegration tests end-to-end compliance workflow
func TestIntegration(t *testing.T) {
	t.Run("Complete Compliance Check", func(t *testing.T) {
		ctx := context.Background()

		// 1. Assess all frameworks
		soc2 := frameworks.NewSOC2Engine()
		gdpr := frameworks.NewGDPREngine()
		hipaa := frameworks.NewHIPAAEngine()

		soc2Report, _ := soc2.AssessCompliance(ctx)
		gdprReport, _ := gdpr.AssessCompliance(ctx)
		hipaaReport, _ := hipaa.AssessCompliance(ctx)

		// 2. Check security posture
		postureEngine := posture.NewEngine()
		securityPosture, _ := postureEngine.AssessPosture(ctx)

		// 3. Generate summary
		t.Log("=== COMPLIANCE SUMMARY ===")
		t.Logf("SOC2: %.1f%% (%s)", soc2Report.Score, soc2Report.Status)
		t.Logf("GDPR: %.1f%% (%s)", gdprReport.Score, gdprReport.Status)
		t.Logf("HIPAA: %.1f%% (%s)", hipaaReport.Score, hipaaReport.Status)
		t.Logf("Security: %.1f (%s)", securityPosture.OverallScore, securityPosture.RiskLevel)

		// Verify all frameworks meet minimum standards
		if soc2Report.Score < 90.0 || gdprReport.Score < 90.0 || hipaaReport.Score < 85.0 {
			t.Error("One or more frameworks below minimum compliance threshold")
		}

		if securityPosture.OverallScore < 70.0 {
			t.Error("Security posture below acceptable threshold")
		}
	})
}

// Benchmark tests

func BenchmarkPolicyEvaluation(b *testing.B) {
	engine := policy.NewEngine()
	ctx := context.Background()

	request := &compliance.PolicyRequest{
		Principal: "user@company.com",
		Action:    "vm:access",
		Resource:  "vm:production/vm-123",
		Context: map[string]interface{}{
			"mfa_verified": true,
			"environment":  "production",
		},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = engine.Evaluate(ctx, request)
	}
}

func BenchmarkAuditLogging(b *testing.B) {
	log := audit.NewBlockchainAuditLog()

	event := &compliance.AuditEvent{
		EventType: "benchmark_event",
		Actor: compliance.Actor{
			ID:   "user-123",
			Type: "user",
		},
		Action: "test_action",
		Resource: compliance.Resource{
			ID:   "resource-456",
			Type: "test_resource",
		},
		Result: "success",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = log.LogEvent(event)
	}
}
