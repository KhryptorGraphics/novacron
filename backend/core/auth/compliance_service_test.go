package auth

import (
	"testing"
)

// TestComplianceChecks_RealSignalsReflectState is the regression test for
// novacron-yvo. Every checkXxx in compliance_service.go used to
// unconditionally return a fixed value (40 always `true`, 1 always
// `false`) regardless of real system state. Each subtest here wires (or
// unwires) the same real config/service a production caller would toggle
// and asserts the corresponding check's result actually changes.
func TestComplianceChecks_RealSignalsReflectState(t *testing.T) {
	t.Run("password policy follows PasswordSecurityService config", func(t *testing.T) {
		svc := NewComplianceService(NewInMemoryAuditService(), NewEncryptionService(DefaultEncryptionConfig()))

		if got := svc.checkPasswordPolicy(); got.Status != NotTested {
			t.Fatalf("expected NotTested with no password service wired, got %s (%s)", got.Status, got.Reason)
		}

		weak := NewPasswordSecurityService(PasswordSecurityConfig{MinLength: 4, HashAlgorithm: "argon2"})
		svc.SetPasswordService(weak)
		if got := svc.checkPasswordPolicy(); got.Status != NonCompliant {
			t.Fatalf("expected NonCompliant with a weak password policy, got %s (%s)", got.Status, got.Reason)
		}

		strong := NewPasswordSecurityService(DefaultPasswordSecurityConfig())
		svc.SetPasswordService(strong)
		if got := svc.checkPasswordPolicy(); got.Status != Compliant {
			t.Fatalf("expected Compliant with a strong password policy, got %s (%s)", got.Status, got.Reason)
		}
	})

	t.Run("default-password guard follows ForbidCommonPasswords flag", func(t *testing.T) {
		svc := NewComplianceService(NewInMemoryAuditService(), NewEncryptionService(DefaultEncryptionConfig()))

		cfg := DefaultPasswordSecurityConfig()
		cfg.ForbidCommonPasswords = false
		svc.SetPasswordService(NewPasswordSecurityService(cfg))
		if got := svc.checkDefaultPasswords(); got.Status != NonCompliant {
			t.Fatalf("expected NonCompliant with ForbidCommonPasswords=false, got %s (%s)", got.Status, got.Reason)
		}

		cfg.ForbidCommonPasswords = true
		svc.SetPasswordService(NewPasswordSecurityService(cfg))
		if got := svc.checkDefaultPasswords(); got.Status != Compliant {
			t.Fatalf("expected Compliant with ForbidCommonPasswords=true, got %s (%s)", got.Status, got.Reason)
		}
	})

	t.Run("MFA check follows 2FA service enrollment", func(t *testing.T) {
		svc := NewComplianceService(NewInMemoryAuditService(), NewEncryptionService(DefaultEncryptionConfig()))

		if got := svc.checkMFAEnabled(); got.Status != NotTested {
			t.Fatalf("expected NotTested with no 2FA service wired, got %s (%s)", got.Status, got.Reason)
		}

		tfs := NewTwoFactorService("test", []byte("0123456789abcdef0123456789abcdef"))
		svc.SetTwoFactorService(tfs)
		if got := svc.checkMFAEnabled(); got.Status != NotTested {
			t.Fatalf("expected NotTested with a 2FA service wired but no enrolled users, got %s (%s)", got.Status, got.Reason)
		}

		// Same package: manipulate the service's own state directly rather
		// than driving the full TOTP setup/verify flow, since what's under
		// test is checkMFAEnabled's use of GetStats(), not TOTP itself.
		tfs.userSecrets["user-1"] = &UserTwoFactor{UserID: "user-1", Enabled: false}
		if got := svc.checkMFAEnabled(); got.Status != NonCompliant {
			t.Fatalf("expected NonCompliant with 0 of 1 enrolled users having MFA enabled, got %s (%s)", got.Status, got.Reason)
		}

		tfs.userSecrets["user-1"].Enabled = true
		if got := svc.checkMFAEnabled(); got.Status != Compliant {
			t.Fatalf("expected Compliant with 1 of 1 enrolled users having MFA enabled, got %s (%s)", got.Status, got.Reason)
		}
	})

	t.Run("encryption-at-rest, stored-data, and integrity checks follow active key state", func(t *testing.T) {
		enc := NewEncryptionService(DefaultEncryptionConfig())
		svc := NewComplianceService(NewInMemoryAuditService(), enc)

		if got := svc.checkEncryptionAtRest(); got.Status != NonCompliant {
			t.Fatalf("expected NonCompliant with no active keys, got %s (%s)", got.Status, got.Reason)
		}
		if got := svc.checkDataIntegrity(); got.Status != NonCompliant {
			t.Fatalf("expected checkDataIntegrity to follow the encryption-at-rest signal, got %s (%s)", got.Status, got.Reason)
		}

		if _, err := enc.GenerateKey(""); err != nil {
			t.Fatalf("GenerateKey failed: %v", err)
		}

		if got := svc.checkEncryptionAtRest(); got.Status != Compliant {
			t.Fatalf("expected Compliant after generating a key, got %s (%s)", got.Status, got.Reason)
		}
		if got := svc.checkStoredDataEncryption(); got.Status != Compliant {
			t.Fatalf("expected checkStoredDataEncryption to follow the encryption-at-rest signal, got %s (%s)", got.Status, got.Reason)
		}
		if got := svc.checkDataIntegrity(); got.Status != Compliant {
			t.Fatalf("expected checkDataIntegrity to follow the encryption-at-rest signal, got %s (%s)", got.Status, got.Reason)
		}
		if got := svc.checkKeyManagement(); got.Status != Compliant {
			t.Fatalf("expected Compliant key management with an active key and a configured rotation interval, got %s (%s)", got.Status, got.Reason)
		}
	})

	t.Run("strong cryptography follows the configured default algorithm", func(t *testing.T) {
		strongSvc := NewComplianceService(NewInMemoryAuditService(), NewEncryptionService(DefaultEncryptionConfig()))
		if got := strongSvc.checkStrongCryptography(); got.Status != Compliant {
			t.Fatalf("expected Compliant with the AES-256-GCM default, got %s (%s)", got.Status, got.Reason)
		}

		weakSvc := NewComplianceService(NewInMemoryAuditService(), NewEncryptionService(EncryptionConfig{DefaultAlgorithm: "DES"}))
		if got := weakSvc.checkStrongCryptography(); got.Status != NonCompliant {
			t.Fatalf("expected NonCompliant with a weak configured algorithm, got %s (%s)", got.Status, got.Reason)
		}
	})

	t.Run("audit-backed checks follow whether an audit service is wired", func(t *testing.T) {
		withAudit := NewComplianceService(NewInMemoryAuditService(), NewEncryptionService(DefaultEncryptionConfig()))
		if got := withAudit.checkLoggingEnabled(); got.Status != Compliant {
			t.Fatalf("expected Compliant with an audit service wired, got %s (%s)", got.Status, got.Reason)
		}
		if got := withAudit.checkAuditControls(); got.Status != Compliant {
			t.Fatalf("expected checkAuditControls to follow the same audit signal, got %s (%s)", got.Status, got.Reason)
		}

		withoutAudit := NewComplianceService(nil, NewEncryptionService(DefaultEncryptionConfig()))
		if got := withoutAudit.checkLoggingEnabled(); got.Status != NonCompliant {
			t.Fatalf("expected NonCompliant with no audit service configured, got %s (%s)", got.Status, got.Reason)
		}
	})

	t.Run("RBAC checks follow real role/permission definitions", func(t *testing.T) {
		svc := NewComplianceService(NewInMemoryAuditService(), NewEncryptionService(DefaultEncryptionConfig()))

		if got := svc.checkAccessControls(); got.Status != NotTested {
			t.Fatalf("expected NotTested with no role service wired, got %s (%s)", got.Status, got.Reason)
		}

		// Bypass NewRoleMemoryStore's system-role seeding so the "zero
		// roles defined" state is actually reachable for this test.
		roles := &RoleMemoryStore{roles: make(map[string]*Role)}
		svc.SetRoleService(roles)
		if got := svc.checkAccessControls(); got.Status != NonCompliant {
			t.Fatalf("expected NonCompliant with zero roles defined, got %s (%s)", got.Status, got.Reason)
		}

		if err := roles.Create(&Role{ID: "viewer", Name: "Viewer"}); err != nil {
			t.Fatalf("Create role failed: %v", err)
		}
		if got := svc.checkAccessControls(); got.Status != NonCompliant {
			t.Fatalf("expected NonCompliant with a role that grants no permissions, got %s (%s)", got.Status, got.Reason)
		}

		if err := roles.AddPermission("viewer", Permission{Resource: "vm", Action: "read", Effect: "allow"}); err != nil {
			t.Fatalf("AddPermission failed: %v", err)
		}
		if got := svc.checkAccessControls(); got.Status != Compliant {
			t.Fatalf("expected Compliant once a role grants an explicit permission, got %s (%s)", got.Status, got.Reason)
		}
		// checkAccessManagement and checkTechnicalAccessControl test the
		// same real RBAC signal from HIPAA's framework angle.
		if got := svc.checkAccessManagement(); got.Status != Compliant {
			t.Fatalf("expected checkAccessManagement to track the same RBAC signal, got %s (%s)", got.Status, got.Reason)
		}
		if got := svc.checkTechnicalAccessControl(); got.Status != Compliant {
			t.Fatalf("expected checkTechnicalAccessControl to track the same RBAC signal, got %s (%s)", got.Status, got.Reason)
		}
	})

	t.Run("network checks follow zero-trust policy and microsegment state", func(t *testing.T) {
		svc := NewComplianceService(NewInMemoryAuditService(), NewEncryptionService(DefaultEncryptionConfig()))

		if got := svc.checkFirewallConfiguration(); got.Status != NotTested {
			t.Fatalf("expected NotTested with no zero-trust service wired, got %s (%s)", got.Status, got.Reason)
		}
		if got := svc.checkTLSEnabled(); got.Status != NotTested {
			t.Fatalf("expected NotTested with no zero-trust service wired, got %s (%s)", got.Status, got.Reason)
		}

		zt := NewZeroTrustNetworkService(NewInMemoryAuditService(), NewEncryptionService(DefaultEncryptionConfig()))
		svc.SetZeroTrustNetworkService(zt)

		if got := svc.checkFirewallConfiguration(); got.Status != NonCompliant {
			t.Fatalf("expected NonCompliant with no policies configured, got %s (%s)", got.Status, got.Reason)
		}
		if got := svc.checkNetworkSegmentation(); got.Status != NonCompliant {
			t.Fatalf("expected NonCompliant with no microsegments defined, got %s (%s)", got.Status, got.Reason)
		}

		denyAll := &NetworkPolicy{
			ID:          "deny-all",
			Name:        "Deny All",
			Enabled:     true,
			Action:      NetworkPolicyDeny,
			Source:      NetworkPolicySelector{Any: true},
			Destination: NetworkPolicySelector{Any: true},
		}
		if err := zt.CreatePolicy(denyAll); err != nil {
			t.Fatalf("CreatePolicy(deny-all) failed: %v", err)
		}
		if got := svc.checkFirewallConfiguration(); got.Status != Compliant {
			t.Fatalf("expected Compliant after adding a default-deny policy, got %s (%s)", got.Status, got.Reason)
		}
		if got := svc.checkTLSEnabled(); got.Status != NonCompliant {
			t.Fatalf("expected NonCompliant TLS check before any mTLS policy exists, got %s (%s)", got.Status, got.Reason)
		}

		requireMTLS := &NetworkPolicy{
			ID:          "require-mtls",
			Name:        "Require mTLS",
			Enabled:     true,
			Action:      NetworkPolicyRequireMTLS,
			Source:      NetworkPolicySelector{Any: true},
			Destination: NetworkPolicySelector{Any: true},
		}
		if err := zt.CreatePolicy(requireMTLS); err != nil {
			t.Fatalf("CreatePolicy(require-mtls) failed: %v", err)
		}
		if got := svc.checkTLSEnabled(); got.Status != Compliant {
			t.Fatalf("expected Compliant after adding an mTLS policy, got %s (%s)", got.Status, got.Reason)
		}
		// checkEncryptionInTransit and checkTransmissionSecurity /
		// checkTransmissionEncryption test the same real mTLS-policy
		// signal from other frameworks' angles.
		if got := svc.checkEncryptionInTransit(); got.Status != Compliant {
			t.Fatalf("expected checkEncryptionInTransit to track the same mTLS signal, got %s (%s)", got.Status, got.Reason)
		}

		segment := &Microsegment{ID: "seg-1", Name: "seg-1", TenantID: "tenant-1", Isolation: IsolationStrict}
		if err := zt.CreateMicrosegment(segment); err != nil {
			t.Fatalf("CreateMicrosegment failed: %v", err)
		}
		if got := svc.checkNetworkSegmentation(); got.Status != Compliant {
			t.Fatalf("expected Compliant after defining a microsegment, got %s (%s)", got.Status, got.Reason)
		}
	})
}

// TestComplianceChecks_NeverFabricateAValue is the direct regression test
// for novacron-yvo's core complaint: on a ComplianceService with none of
// the optional real-signal sources wired (the situation every checkXxx used
// to see), no check may unconditionally report Compliant. The only checks
// allowed to be Compliant here are the ones with a real signal that is
// always available from the two required constructor dependencies (a
// non-nil audit service, and an encryption service whose configured
// default algorithm is genuinely strong).
func TestComplianceChecks_NeverFabricateAValue(t *testing.T) {
	svc := NewComplianceService(NewInMemoryAuditService(), NewEncryptionService(DefaultEncryptionConfig()))

	alwaysCompliant := map[string]bool{
		"checkLoggingEnabled":     true,
		"checkAuditControls":      true,
		"checkStrongCryptography": true,
	}

	checks := map[string]func() CheckResult{
		"checkPolicyExists":           func() CheckResult { return svc.checkPolicyExists("x") },
		"checkTrainingRecords":        func() CheckResult { return svc.checkTrainingRecords("x") },
		"checkPolicyCommunication":    func() CheckResult { return svc.checkPolicyCommunication("x") },
		"checkMFAEnabled":             svc.checkMFAEnabled,
		"checkPasswordPolicy":         svc.checkPasswordPolicy,
		"checkAccessReviews":          svc.checkAccessReviews,
		"checkTLSEnabled":             svc.checkTLSEnabled,
		"checkEncryptionInTransit":    svc.checkEncryptionInTransit,
		"checkLoggingEnabled":         svc.checkLoggingEnabled,
		"checkMonitoring":             svc.checkMonitoring,
		"checkAlertingEnabled":        svc.checkAlertingEnabled,
		"checkPrivacyByDesign":        svc.checkPrivacyByDesign,
		"checkDataMinimization":       svc.checkDataMinimization,
		"checkEncryptionAtRest":       svc.checkEncryptionAtRest,
		"checkAccessControls":         svc.checkAccessControls,
		"checkBackupSecurity":         svc.checkBackupSecurity,
		"checkBreachProcedures":       svc.checkBreachProcedures,
		"checkNotificationTiming":     svc.checkNotificationTiming,
		"checkDPIAProcess":            svc.checkDPIAProcess,
		"checkRiskAssessment":         svc.checkRiskAssessment,
		"checkSecurityOfficer":        svc.checkSecurityOfficer,
		"checkWorkforceTraining":      svc.checkWorkforceTraining,
		"checkAccessManagement":       svc.checkAccessManagement,
		"checkFacilityAccess":         svc.checkFacilityAccess,
		"checkWorkstationSecurity":    svc.checkWorkstationSecurity,
		"checkDeviceControls":         svc.checkDeviceControls,
		"checkTechnicalAccessControl": svc.checkTechnicalAccessControl,
		"checkAuditControls":          svc.checkAuditControls,
		"checkDataIntegrity":          svc.checkDataIntegrity,
		"checkTransmissionSecurity":   svc.checkTransmissionSecurity,
		"checkBusinessAssociates":     svc.checkBusinessAssociates,
		"checkContractualSafeguards":  svc.checkContractualSafeguards,
		"checkFirewallConfiguration":  svc.checkFirewallConfiguration,
		"checkNetworkSegmentation":    svc.checkNetworkSegmentation,
		"checkDefaultPasswords":       svc.checkDefaultPasswords,
		"checkSecurityParameters":     svc.checkSecurityParameters,
		"checkStoredDataEncryption":   svc.checkStoredDataEncryption,
		"checkDataRetention":          svc.checkDataRetention,
		"checkKeyManagement":          svc.checkKeyManagement,
		"checkTransmissionEncryption": svc.checkTransmissionEncryption,
		"checkStrongCryptography":     svc.checkStrongCryptography,
	}

	const wantChecks = 41
	if len(checks) != wantChecks {
		t.Fatalf("expected exactly %d compliance checks (one per original hardcoded stub), got %d — update this test if a check was added or removed", wantChecks, len(checks))
	}

	for name, check := range checks {
		got := check()
		if got.Reason == "" {
			t.Errorf("%s: CheckResult has no Reason", name)
		}
		if alwaysCompliant[name] {
			if got.Status != Compliant {
				t.Errorf("%s: expected Compliant (has a real always-on signal), got %s (%s)", name, got.Status, got.Reason)
			}
			continue
		}
		if got.Status == Compliant {
			t.Errorf("%s: returned Compliant with no real signal wired — this is the fabricated-success bug novacron-yvo fixed", name)
		}
	}
}
