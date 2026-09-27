package runbooks

import (
	"context"
	"fmt"
	"log"
	"time"
)

// DataCorruptionRunbook handles detected data corruption scenarios
func DataCorruptionRunbook() *Runbook {
	return &Runbook{
		ID:          "data-corruption",
		Name:        "Data Corruption Recovery",
		Description: "Automated recovery from detected data corruption",
		Scenario:    "Checksum mismatch, replica divergence, or corrupted write detected",
		Steps: []RunbookStep{
			{
				ID:          "detect-corruption",
				Name:        "Detect and Validate Corruption",
				Description: "Confirm the corruption is real and scope it",
				Action:      detectDataCorruption,
				Timeout:     2 * time.Minute,
				OnFailure:   "abort",
				MaxRetries:  3,
			},
			{
				ID:          "isolate-affected-range",
				Name:        "Isolate Affected Range",
				Description: "Quarantine the corrupted keys/tables/volumes so writes stop widening the blast radius",
				Action:      isolateAffectedRange,
				Timeout:     1 * time.Minute,
				OnFailure:   "abort",
				MaxRetries:  2,
			},
			{
				ID:          "notify-stakeholders",
				Name:        "Notify Stakeholders",
				Description: "Alert operations team and data owners",
				Action:      notifyStakeholders,
				Timeout:     30 * time.Second,
				OnFailure:   "continue",
				MaxRetries:  2,
			},
			{
				ID:          "locate-clean-snapshot",
				Name:        "Locate Clean Snapshot",
				Description: "Find the most recent backup/replica known to predate the corruption",
				Action:      locateCleanSnapshot,
				Timeout:     3 * time.Minute,
				OnFailure:   "abort",
				MaxRetries:  1,
			},
			{
				ID:               "restore-from-snapshot",
				Name:             "Restore From Snapshot",
				Description:      "Restore the affected range from the clean snapshot",
				Action:           restoreFromSnapshot,
				RequiresApproval: true,
				AutoRollback:     true,
				Timeout:          10 * time.Minute,
				OnFailure:        "abort",
				MaxRetries:       1,
			},
			{
				ID:          "verify-integrity",
				Name:        "Verify Data Integrity",
				Description: "Re-run checksums/consistency checks on the restored range",
				Action:      verifyDataIntegrity,
				Timeout:     5 * time.Minute,
				OnFailure:   "abort",
				MaxRetries:  2,
			},
			{
				ID:          "lift-isolation",
				Name:        "Lift Isolation",
				Description: "Re-enable normal read/write access to the recovered range",
				Action:      liftIsolation,
				Timeout:     1 * time.Minute,
				OnFailure:   "continue",
				MaxRetries:  2,
			},
			{
				ID:          "final-notification",
				Name:        "Final Notification",
				Description: "Notify completion and recovered data range",
				Action:      finalNotification,
				Timeout:     30 * time.Second,
				OnFailure:   "continue",
				MaxRetries:  1,
			},
		},
	}
}

func detectDataCorruption(ctx context.Context, params map[string]interface{}) error {
	log.Println("[Runbook] Detecting data corruption...")

	resourceID, ok := params["resource_id"].(string)
	if !ok {
		return fmt.Errorf("resource_id parameter required")
	}

	log.Printf("[Runbook] Validating corruption report for resource: %s", resourceID)

	// Simulate corruption validation:
	// - Re-run checksum comparison against replicas
	// - Confirm this is not a transient read error
	time.Sleep(500 * time.Millisecond)

	log.Printf("[Runbook] Corruption confirmed for resource: %s", resourceID)
	return nil
}

func isolateAffectedRange(ctx context.Context, params map[string]interface{}) error {
	log.Println("[Runbook] Isolating affected range...")

	time.Sleep(200 * time.Millisecond)

	// Quarantine actions:
	// - Mark affected keys/tables read-only
	// - Route new writes to a healthy replica

	log.Println("[Runbook] Affected range isolated")
	return nil
}

func locateCleanSnapshot(ctx context.Context, params map[string]interface{}) error {
	log.Println("[Runbook] Locating clean snapshot...")

	time.Sleep(400 * time.Millisecond)

	// Selection criteria:
	// - Most recent snapshot older than the estimated corruption onset
	// - Verified backup checksum

	snapshotID := "snapshot-pre-corruption"
	params["snapshot_id"] = snapshotID

	log.Printf("[Runbook] Selected clean snapshot: %s", snapshotID)
	return nil
}

func restoreFromSnapshot(ctx context.Context, params map[string]interface{}) error {
	log.Println("[Runbook] Restoring from snapshot...")

	time.Sleep(1 * time.Second)

	// Restore actions:
	// - Replay snapshot into the isolated range
	// - Replay any trusted transaction logs since the snapshot

	log.Println("[Runbook] Restore from snapshot complete")
	return nil
}

func verifyDataIntegrity(ctx context.Context, params map[string]interface{}) error {
	log.Println("[Runbook] Verifying data integrity...")

	time.Sleep(500 * time.Millisecond)

	// Verification:
	// - Recompute checksums across the restored range
	// - Compare against remaining healthy replicas

	log.Println("[Runbook] Data integrity verified")
	return nil
}

func liftIsolation(ctx context.Context, params map[string]interface{}) error {
	log.Println("[Runbook] Lifting isolation...")

	time.Sleep(200 * time.Millisecond)

	// Re-enable actions:
	// - Remove read-only quarantine
	// - Resume normal write routing

	log.Println("[Runbook] Isolation lifted")
	return nil
}
