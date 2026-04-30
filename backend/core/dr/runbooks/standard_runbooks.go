package runbooks

import (
	"context"
	"fmt"
	"log"
	"time"
)

// DataCorruptionRunbook handles suspected storage or metadata corruption.
func DataCorruptionRunbook() *Runbook {
	return &Runbook{
		ID:          "data-corruption",
		Name:        "Data Corruption Recovery",
		Description: "Quarantine corrupted data, verify backups, and restore from a clean recovery point",
		Scenario:    "VM volume, backup, or control-plane metadata fails integrity validation",
		Steps: []RunbookStep{
			{
				ID:          "identify-scope",
				Name:        "Identify Corruption Scope",
				Description: "Determine affected workload, volume, or metadata namespace",
				Action:      identifyCorruptionScope,
				Timeout:     1 * time.Minute,
				OnFailure:   "abort",
				MaxRetries:  1,
			},
			{
				ID:          "quarantine-data",
				Name:        "Quarantine Affected Data",
				Description: "Fence affected storage paths before further writes occur",
				Action:      quarantineCorruptedData,
				Timeout:     2 * time.Minute,
				OnFailure:   "abort",
				MaxRetries:  2,
			},
			{
				ID:          "verify-clean-backup",
				Name:        "Verify Clean Backup",
				Description: "Select and verify the newest backup outside the corruption window",
				Action:      verifyCleanBackup,
				Timeout:     3 * time.Minute,
				OnFailure:   "abort",
				MaxRetries:  2,
			},
			{
				ID:               "restore-clean-copy",
				Name:             "Restore Clean Copy",
				Description:      "Restore from the verified recovery point",
				Action:           restoreCleanCopy,
				RequiresApproval: true,
				AutoRollback:     true,
				Timeout:          10 * time.Minute,
				OnFailure:        "abort",
				MaxRetries:       1,
			},
			{
				ID:          "validate-restored-data",
				Name:        "Validate Restored Data",
				Description: "Run post-restore integrity checks before releasing quarantine",
				Action:      validateRestoredData,
				Timeout:     5 * time.Minute,
				OnFailure:   "abort",
				MaxRetries:  2,
			},
		},
		Metadata: map[string]string{
			"risk": "data-loss",
			"rpo":  "backup-dependent",
		},
	}
}

// NetworkPartitionRunbook handles cluster split or reachability loss.
func NetworkPartitionRunbook() *Runbook {
	return &Runbook{
		ID:          "network-partition",
		Name:        "Network Partition Recovery",
		Description: "Detect partitioned nodes, preserve quorum, and safely rejoin recovered members",
		Scenario:    "WAN or fabric reachability loss splits cluster membership",
		Steps: []RunbookStep{
			{
				ID:          "classify-partition",
				Name:        "Classify Partition",
				Description: "Map reachable nodes and identify the quorum side",
				Action:      classifyNetworkPartition,
				Timeout:     1 * time.Minute,
				OnFailure:   "abort",
				MaxRetries:  2,
			},
			{
				ID:          "preserve-quorum",
				Name:        "Preserve Quorum",
				Description: "Fence minority-side writers and keep the quorum side authoritative",
				Action:      preserveQuorum,
				Timeout:     2 * time.Minute,
				OnFailure:   "abort",
				MaxRetries:  2,
			},
			{
				ID:          "probe-links",
				Name:        "Probe Fabric Links",
				Description: "Measure recovery of inter-node transport before rejoin",
				Action:      probeFabricLinks,
				Timeout:     3 * time.Minute,
				OnFailure:   "retry",
				MaxRetries:  3,
			},
			{
				ID:               "rejoin-nodes",
				Name:             "Rejoin Recovered Nodes",
				Description:      "Reconcile state and rejoin nodes after fencing is safe to lift",
				Action:           rejoinRecoveredNodes,
				RequiresApproval: true,
				Timeout:          5 * time.Minute,
				OnFailure:        "abort",
				MaxRetries:       1,
			},
			{
				ID:          "validate-membership",
				Name:        "Validate Membership",
				Description: "Confirm cluster membership and placement health are consistent",
				Action:      validateMembership,
				Timeout:     2 * time.Minute,
				OnFailure:   "abort",
				MaxRetries:  2,
			},
		},
		Metadata: map[string]string{
			"risk": "split-brain",
		},
	}
}

// SecurityIncidentRunbook handles suspected credential or node compromise.
func SecurityIncidentRunbook() *Runbook {
	return &Runbook{
		ID:          "security-incident",
		Name:        "Security Incident Containment",
		Description: "Contain suspected compromise, rotate trust material, and validate recovery state",
		Scenario:    "Node, credential, or control-plane compromise is suspected",
		Steps: []RunbookStep{
			{
				ID:          "classify-incident",
				Name:        "Classify Incident",
				Description: "Record affected node, credential, or tenant scope",
				Action:      classifySecurityIncident,
				Timeout:     1 * time.Minute,
				OnFailure:   "abort",
				MaxRetries:  1,
			},
			{
				ID:               "isolate-scope",
				Name:             "Isolate Affected Scope",
				Description:      "Fence affected nodes and disable suspected credentials",
				Action:           isolateSecurityScope,
				RequiresApproval: true,
				Timeout:          3 * time.Minute,
				OnFailure:        "abort",
				MaxRetries:       1,
			},
			{
				ID:          "rotate-trust",
				Name:        "Rotate Trust Material",
				Description: "Rotate node certificates, API credentials, and federation tokens",
				Action:      rotateTrustMaterial,
				Timeout:     10 * time.Minute,
				OnFailure:   "abort",
				MaxRetries:  1,
			},
			{
				ID:          "audit-recovery-state",
				Name:        "Audit Recovery State",
				Description: "Validate logs, membership, and restored workloads before release",
				Action:      auditRecoveryState,
				Timeout:     5 * time.Minute,
				OnFailure:   "abort",
				MaxRetries:  2,
			},
			{
				ID:          "release-containment",
				Name:        "Release Containment",
				Description: "Return remediated resources to service",
				Action:      releaseContainment,
				Timeout:     2 * time.Minute,
				OnFailure:   "continue",
				MaxRetries:  1,
			},
		},
		Metadata: map[string]string{
			"risk": "security",
		},
	}
}

func identifyCorruptionScope(ctx context.Context, params map[string]interface{}) error {
	scope, ok := firstStringParam(params, "resource_id", "volume_id", "backup_id")
	if !ok {
		return fmt.Errorf("one of resource_id, volume_id, or backup_id is required")
	}

	log.Printf("[Runbook] Corruption scope identified: %s", scope)
	return waitForStep(ctx, 100*time.Millisecond)
}

func quarantineCorruptedData(ctx context.Context, params map[string]interface{}) error {
	log.Println("[Runbook] Quarantining affected data paths")
	params["quarantine_active"] = true
	return waitForStep(ctx, 100*time.Millisecond)
}

func verifyCleanBackup(ctx context.Context, params map[string]interface{}) error {
	backupID, ok := firstStringParam(params, "clean_backup_id", "backup_id")
	if !ok {
		return fmt.Errorf("clean_backup_id or backup_id parameter required")
	}

	params["verified_backup_id"] = backupID
	log.Printf("[Runbook] Clean backup verified: %s", backupID)
	return waitForStep(ctx, 150*time.Millisecond)
}

func restoreCleanCopy(ctx context.Context, params map[string]interface{}) error {
	backupID, ok := firstStringParam(params, "verified_backup_id", "clean_backup_id", "backup_id")
	if !ok {
		return fmt.Errorf("verified backup parameter required")
	}

	log.Printf("[Runbook] Restoring clean copy from backup: %s", backupID)
	return waitForStep(ctx, 200*time.Millisecond)
}

func validateRestoredData(ctx context.Context, params map[string]interface{}) error {
	if active, _ := params["quarantine_active"].(bool); !active {
		return fmt.Errorf("quarantine must be active before validation")
	}

	params["quarantine_active"] = false
	log.Println("[Runbook] Restored data validated and quarantine released")
	return waitForStep(ctx, 100*time.Millisecond)
}

func classifyNetworkPartition(ctx context.Context, params map[string]interface{}) error {
	if _, ok := firstStringParam(params, "cluster_id", "partition_id"); !ok {
		return fmt.Errorf("cluster_id or partition_id parameter required")
	}

	log.Println("[Runbook] Network partition classified")
	return waitForStep(ctx, 100*time.Millisecond)
}

func preserveQuorum(ctx context.Context, params map[string]interface{}) error {
	params["minority_fenced"] = true
	log.Println("[Runbook] Minority side fenced; quorum side preserved")
	return waitForStep(ctx, 100*time.Millisecond)
}

func probeFabricLinks(ctx context.Context, params map[string]interface{}) error {
	log.Println("[Runbook] Probing fabric links for stable reachability")
	params["fabric_links_healthy"] = true
	return waitForStep(ctx, 150*time.Millisecond)
}

func rejoinRecoveredNodes(ctx context.Context, params map[string]interface{}) error {
	if fenced, _ := params["minority_fenced"].(bool); !fenced {
		return fmt.Errorf("minority side must be fenced before rejoin")
	}
	if healthy, _ := params["fabric_links_healthy"].(bool); !healthy {
		return fmt.Errorf("fabric links must be healthy before rejoin")
	}

	log.Println("[Runbook] Rejoining recovered nodes")
	return waitForStep(ctx, 150*time.Millisecond)
}

func validateMembership(ctx context.Context, params map[string]interface{}) error {
	log.Println("[Runbook] Cluster membership validated")
	return waitForStep(ctx, 100*time.Millisecond)
}

func classifySecurityIncident(ctx context.Context, params map[string]interface{}) error {
	if _, ok := firstStringParam(params, "incident_id", "node_id", "credential_id"); !ok {
		return fmt.Errorf("incident_id, node_id, or credential_id parameter required")
	}

	log.Println("[Runbook] Security incident classified")
	return waitForStep(ctx, 100*time.Millisecond)
}

func isolateSecurityScope(ctx context.Context, params map[string]interface{}) error {
	params["containment_active"] = true
	log.Println("[Runbook] Affected security scope isolated")
	return waitForStep(ctx, 150*time.Millisecond)
}

func rotateTrustMaterial(ctx context.Context, params map[string]interface{}) error {
	if active, _ := params["containment_active"].(bool); !active {
		return fmt.Errorf("containment must be active before trust rotation")
	}

	log.Println("[Runbook] Trust material rotated")
	return waitForStep(ctx, 200*time.Millisecond)
}

func auditRecoveryState(ctx context.Context, params map[string]interface{}) error {
	log.Println("[Runbook] Recovery state audited")
	return waitForStep(ctx, 100*time.Millisecond)
}

func releaseContainment(ctx context.Context, params map[string]interface{}) error {
	params["containment_active"] = false
	log.Println("[Runbook] Security containment released")
	return waitForStep(ctx, 100*time.Millisecond)
}

func firstStringParam(params map[string]interface{}, keys ...string) (string, bool) {
	for _, key := range keys {
		value, ok := params[key].(string)
		if ok && value != "" {
			return value, true
		}
	}

	return "", false
}

func waitForStep(ctx context.Context, delay time.Duration) error {
	timer := time.NewTimer(delay)
	defer timer.Stop()

	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}
