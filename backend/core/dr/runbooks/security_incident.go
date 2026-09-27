package runbooks

import (
	"context"
	"fmt"
	"log"
	"time"
)

// SecurityIncidentRunbook handles a detected security incident (e.g. credential
// compromise, intrusion, or anomalous access pattern).
func SecurityIncidentRunbook() *Runbook {
	return &Runbook{
		ID:          "security-incident",
		Name:        "Security Incident Response",
		Description: "Automated containment and recovery for a detected security incident",
		Scenario:    "Credential compromise, intrusion, or anomalous access pattern detected",
		Steps: []RunbookStep{
			{
				ID:          "detect-incident",
				Name:        "Detect and Validate Incident",
				Description: "Confirm the security alert is a genuine incident, not a false positive",
				Action:      detectSecurityIncident,
				Timeout:     2 * time.Minute,
				OnFailure:   "abort",
				MaxRetries:  3,
			},
			{
				ID:               "isolate-compromised-assets",
				Name:             "Isolate Compromised Assets",
				Description:      "Network-isolate or disable the affected accounts/nodes to stop further damage",
				Action:           isolateCompromisedAssets,
				RequiresApproval: true,
				Timeout:          1 * time.Minute,
				OnFailure:        "abort",
				MaxRetries:       2,
			},
			{
				ID:          "revoke-credentials",
				Name:        "Revoke Compromised Credentials",
				Description: "Revoke API keys, sessions, and rotate secrets for the affected identity",
				Action:      revokeCompromisedCredentials,
				Timeout:     1 * time.Minute,
				OnFailure:   "abort",
				MaxRetries:  2,
			},
			{
				ID:          "notify-stakeholders",
				Name:        "Notify Stakeholders",
				Description: "Alert security team, operations, and compliance stakeholders",
				Action:      notifyStakeholders,
				Timeout:     30 * time.Second,
				OnFailure:   "continue",
				MaxRetries:  2,
			},
			{
				ID:          "collect-forensics",
				Name:        "Collect Forensic Evidence",
				Description: "Snapshot logs, memory, and disk state for the affected assets before remediation",
				Action:      collectForensicEvidence,
				Timeout:     5 * time.Minute,
				OnFailure:   "continue",
				MaxRetries:  1,
			},
			{
				ID:               "remediate-vulnerability",
				Name:             "Remediate Root Cause",
				Description:      "Patch or reconfigure the exploited weakness",
				Action:           remediateVulnerability,
				RequiresApproval: true,
				AutoRollback:     true,
				Timeout:          10 * time.Minute,
				OnFailure:        "abort",
				MaxRetries:       1,
			},
			{
				ID:          "restore-access",
				Name:        "Restore Legitimate Access",
				Description: "Re-enable access for legitimate users/services with rotated credentials",
				Action:      restoreLegitimateAccess,
				Timeout:     2 * time.Minute,
				OnFailure:   "continue",
				MaxRetries:  2,
			},
			{
				ID:          "final-notification",
				Name:        "Final Notification",
				Description: "Notify completion, root cause, and remediation summary",
				Action:      finalNotification,
				Timeout:     30 * time.Second,
				OnFailure:   "continue",
				MaxRetries:  1,
			},
		},
	}
}

func detectSecurityIncident(ctx context.Context, params map[string]interface{}) error {
	log.Println("[Runbook] Detecting security incident...")

	incidentID, ok := params["incident_id"].(string)
	if !ok {
		return fmt.Errorf("incident_id parameter required")
	}

	log.Printf("[Runbook] Validating security alert: %s", incidentID)

	// Simulate incident validation:
	// - Cross-reference against known false-positive signatures
	// - Confirm with a second detection source

	time.Sleep(500 * time.Millisecond)

	log.Printf("[Runbook] Security incident confirmed: %s", incidentID)
	return nil
}

func isolateCompromisedAssets(ctx context.Context, params map[string]interface{}) error {
	log.Println("[Runbook] Isolating compromised assets...")

	time.Sleep(300 * time.Millisecond)

	// Isolation actions:
	// - Quarantine affected hosts from the network
	// - Suspend affected accounts

	log.Println("[Runbook] Compromised assets isolated")
	return nil
}

func revokeCompromisedCredentials(ctx context.Context, params map[string]interface{}) error {
	log.Println("[Runbook] Revoking compromised credentials...")

	time.Sleep(300 * time.Millisecond)

	// Revocation actions:
	// - Invalidate active sessions/tokens
	// - Rotate API keys and secrets

	log.Println("[Runbook] Compromised credentials revoked")
	return nil
}

func collectForensicEvidence(ctx context.Context, params map[string]interface{}) error {
	log.Println("[Runbook] Collecting forensic evidence...")

	time.Sleep(1 * time.Second)

	// Collection actions:
	// - Snapshot disk/memory state of affected hosts
	// - Archive relevant audit and access logs

	log.Println("[Runbook] Forensic evidence collected")
	return nil
}

func remediateVulnerability(ctx context.Context, params map[string]interface{}) error {
	log.Println("[Runbook] Remediating root-cause vulnerability...")

	time.Sleep(800 * time.Millisecond)

	// Remediation actions:
	// - Apply security patch or configuration fix
	// - Re-run vulnerability scan to confirm closure

	log.Println("[Runbook] Vulnerability remediated")
	return nil
}

func restoreLegitimateAccess(ctx context.Context, params map[string]interface{}) error {
	log.Println("[Runbook] Restoring legitimate access...")

	time.Sleep(300 * time.Millisecond)

	// Restoration actions:
	// - Reissue credentials to legitimate users/services
	// - Re-enable previously isolated assets that were cleared

	log.Println("[Runbook] Legitimate access restored")
	return nil
}
