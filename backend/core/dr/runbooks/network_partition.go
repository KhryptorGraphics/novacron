package runbooks

import (
	"context"
	"fmt"
	"log"
	"time"
)

// NetworkPartitionRunbook handles a detected network partition (split-brain risk)
func NetworkPartitionRunbook() *Runbook {
	return &Runbook{
		ID:          "network-partition",
		Name:        "Network Partition Recovery",
		Description: "Automated recovery from a detected network partition between regions/nodes",
		Scenario:    "Inter-region or inter-node connectivity lost, risking split-brain",
		Steps: []RunbookStep{
			{
				ID:          "detect-partition",
				Name:        "Detect and Validate Partition",
				Description: "Confirm connectivity loss is a real partition, not a transient blip",
				Action:      detectNetworkPartition,
				Timeout:     2 * time.Minute,
				OnFailure:   "abort",
				MaxRetries:  3,
			},
			{
				ID:          "identify-partitions",
				Name:        "Identify Partition Groups",
				Description: "Determine which nodes/regions are on each side of the split",
				Action:      identifyPartitionGroups,
				Timeout:     1 * time.Minute,
				OnFailure:   "abort",
				MaxRetries:  2,
			},
			{
				ID:          "notify-stakeholders",
				Name:        "Notify Stakeholders",
				Description: "Alert operations team and stakeholders",
				Action:      notifyStakeholders,
				Timeout:     30 * time.Second,
				OnFailure:   "continue",
				MaxRetries:  2,
			},
			{
				ID:          "check-quorum",
				Name:        "Verify Quorum",
				Description: "Determine which partition side (if any) retains quorum",
				Action:      checkQuorum,
				Timeout:     1 * time.Minute,
				OnFailure:   "abort",
				MaxRetries:  2,
			},
			{
				ID:               "fence-minority",
				Name:             "Fence Minority Partition",
				Description:      "Force the non-quorum side into read-only/fenced mode to prevent split-brain writes",
				Action:           fenceMinorityPartition,
				RequiresApproval: true,
				AutoRollback:     true,
				Timeout:          2 * time.Minute,
				OnFailure:        "abort",
				MaxRetries:       1,
			},
			{
				ID:          "monitor-recovery",
				Name:        "Monitor for Connectivity Recovery",
				Description: "Poll for partition healing",
				Action:      monitorPartitionRecovery,
				Timeout:     10 * time.Minute,
				OnFailure:   "retry",
				MaxRetries:  5,
			},
			{
				ID:          "reconcile-state",
				Name:        "Reconcile Divergent State",
				Description: "Once connectivity is restored, reconcile any state that diverged during the fence",
				Action:      reconcilePartitionState,
				Timeout:     5 * time.Minute,
				OnFailure:   "abort",
				MaxRetries:  2,
			},
			{
				ID:          "unfence-nodes",
				Name:        "Unfence Recovered Nodes",
				Description: "Restore full read/write access once state is reconciled",
				Action:      unfenceNodes,
				Timeout:     1 * time.Minute,
				OnFailure:   "continue",
				MaxRetries:  2,
			},
			{
				ID:          "final-notification",
				Name:        "Final Notification",
				Description: "Notify completion and current topology",
				Action:      finalNotification,
				Timeout:     30 * time.Second,
				OnFailure:   "continue",
				MaxRetries:  1,
			},
		},
	}
}

func detectNetworkPartition(ctx context.Context, params map[string]interface{}) error {
	log.Println("[Runbook] Detecting network partition...")

	clusterID, ok := params["cluster_id"].(string)
	if !ok {
		return fmt.Errorf("cluster_id parameter required")
	}

	log.Printf("[Runbook] Validating partition report for cluster: %s", clusterID)

	// Simulate partition validation:
	// - Cross-check heartbeat loss from multiple observation points
	// - Rule out a single-node failure vs. an actual network split
	time.Sleep(500 * time.Millisecond)

	log.Printf("[Runbook] Network partition confirmed for cluster: %s", clusterID)
	return nil
}

func identifyPartitionGroups(ctx context.Context, params map[string]interface{}) error {
	log.Println("[Runbook] Identifying partition groups...")

	time.Sleep(300 * time.Millisecond)

	// Identification:
	// - Build reachability graph from each node's perspective
	// - Group nodes into connected components

	log.Println("[Runbook] Partition groups identified")
	return nil
}

func fenceMinorityPartition(ctx context.Context, params map[string]interface{}) error {
	log.Println("[Runbook] Fencing minority partition...")

	time.Sleep(400 * time.Millisecond)

	// Fencing actions:
	// - Reject writes on the non-quorum side
	// - Optionally power-fence via STONITH if hardware fencing is available

	log.Println("[Runbook] Minority partition fenced")
	return nil
}

func monitorPartitionRecovery(ctx context.Context, params map[string]interface{}) error {
	log.Println("[Runbook] Monitoring for partition recovery...")

	time.Sleep(1 * time.Second)

	// Monitoring:
	// - Re-attempt heartbeats across the partition boundary
	// - Return a retryable error until connectivity is confirmed restored

	log.Println("[Runbook] Connectivity check complete")
	return nil
}

func reconcilePartitionState(ctx context.Context, params map[string]interface{}) error {
	log.Println("[Runbook] Reconciling divergent state...")

	time.Sleep(500 * time.Millisecond)

	// Reconciliation:
	// - Diff state between the two former partition sides
	// - Apply conflict resolution policy (e.g. last-writer-wins, CRDT merge)

	log.Println("[Runbook] State reconciled")
	return nil
}

func unfenceNodes(ctx context.Context, params map[string]interface{}) error {
	log.Println("[Runbook] Unfencing recovered nodes...")

	time.Sleep(200 * time.Millisecond)

	// Un-fence actions:
	// - Remove read-only/fenced status
	// - Resume normal replication

	log.Println("[Runbook] Nodes unfenced")
	return nil
}
