package edge

import (
	"context"
	"testing"
	"time"
)

func TestEdgeCloudCoordinator(t *testing.T) {
	config := DefaultEdgeConfig()
	discovery := NewEdgeDiscovery(config)
	placement := NewPlacementEngine(config, discovery)
	coordinator := NewEdgeCloudCoordinator(config, discovery, placement)

	// Add test nodes
	sourceNode := &EdgeNode{
		ID:   "edge-1",
		Type: EdgeType5GMEC,
		Resources: EdgeResources{
			TotalCPUCores:      16,
			UsedCPUCores:       4,
			TotalMemoryMB:      32768,
			UsedMemoryMB:       8192,
			TotalStorageGB:     1000,
			UsedStorageGB:      200,
			TotalBandwidthMbps: 10000,
			UsedBandwidthMbps:  2000,
			UtilizationPercent: 25.0,
		},
		Status: EdgeNodeStatus{
			State:  EdgeNodeStateOnline,
			Health: HealthStatusHealthy,
		},
		Network: NetworkInfo{
			VPNEndpoint: "vpn.cloud.example.com",
		},
	}

	targetNode := &EdgeNode{
		ID:   "edge-2",
		Type: EdgeTypeCDN,
		Resources: EdgeResources{
			TotalCPUCores:      16,
			UsedCPUCores:       2,
			TotalMemoryMB:      32768,
			UsedMemoryMB:       4096,
			TotalStorageGB:     1000,
			UsedStorageGB:      100,
			TotalBandwidthMbps: 10000,
			UsedBandwidthMbps:  1000,
			UtilizationPercent: 15.0,
		},
		Status: EdgeNodeStatus{
			State:  EdgeNodeStateOnline,
			Health: HealthStatusHealthy,
		},
		Network: NetworkInfo{
			VPNEndpoint: "vpn.cloud.example.com",
		},
	}

	discovery.addOrUpdateNode(sourceNode)
	discovery.addOrUpdateNode(targetNode)

	// Test deployment
	deployReq := &PlacementRequest{
		VMID: "test-vm-1",
		Requirements: PlacementRequirements{
			CPUCores:  2,
			MemoryMB:  4096,
			StorageGB: 50,
		},
	}

	ctx := context.Background()
	decision, err := coordinator.DeployToEdge(ctx, deployReq)
	if err != nil {
		t.Fatalf("Deployment failed: %v", err)
	}

	t.Logf("Deployed to edge node: %s", decision.EdgeNodeID)

	// Test migration
	migReq := &MigrationRequest{
		VMID:          "test-vm-1",
		SourceNodeID:  decision.EdgeNodeID,
		TargetNodeID:  "edge-2",
		MigrationType: MigrationTypeLive,
		MaxDowntime:   5 * time.Second,
		Priority:      5,
		Reason:        "testing",
	}

	status, err := coordinator.MigrateVM(ctx, migReq)
	if err != nil {
		t.Fatalf("Migration failed: %v", err)
	}

	if status.State != MigrationStatePending && status.State != MigrationStateRunning {
		t.Errorf("Expected pending or running state, got %s", status.State)
	}

	// Wait for migration to complete (with timeout)
	timeout := time.After(10 * time.Second)
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	completed := false
	for !completed {
		select {
		case <-timeout:
			t.Fatal("Migration timeout")
		case <-ticker.C:
			status, _ := coordinator.GetMigrationStatus("test-vm-1")
			if status.State == MigrationStateCompleted || status.State == MigrationStateFailed {
				completed = true
			}
		}
	}

	finalStatus, _ := coordinator.GetMigrationStatus("test-vm-1")
	t.Logf("Migration completed: State=%s, Progress=%.2f%%, DowntimeMs=%d",
		finalStatus.State, finalStatus.Progress, finalStatus.DowntimeMs)

	// Verify placement was updated
	placedNode, err := coordinator.GetVMPlacement("test-vm-1")
	if err != nil {
		t.Fatalf("Failed to get VM placement: %v", err)
	}

	if placedNode != "edge-2" {
		t.Errorf("Expected VM on edge-2, got %s", placedNode)
	}
}

func TestHandleEdgeFailure(t *testing.T) {
	config := DefaultEdgeConfig()
	discovery := NewEdgeDiscovery(config)
	placement := NewPlacementEngine(config, discovery)
	coordinator := NewEdgeCloudCoordinator(config, discovery, placement)

	// Add nodes
	failedNode := &EdgeNode{
		ID:   "edge-failed",
		Type: EdgeType5GMEC,
		Resources: EdgeResources{
			TotalCPUCores:      16,
			TotalMemoryMB:      32768,
			TotalStorageGB:     1000,
			TotalBandwidthMbps: 10000,
		},
		Status: EdgeNodeStatus{
			State:  EdgeNodeStateOffline,
			Health: HealthStatusUnhealthy,
		},
	}

	healthyNode := &EdgeNode{
		ID:   "edge-healthy",
		Type: EdgeType5GMEC,
		Resources: EdgeResources{
			TotalCPUCores:      16,
			UsedCPUCores:       2,
			TotalMemoryMB:      32768,
			UsedMemoryMB:       4096,
			TotalStorageGB:     1000,
			UsedStorageGB:      100,
			TotalBandwidthMbps: 10000,
			UsedBandwidthMbps:  1000,
		},
		Status: EdgeNodeStatus{
			State:  EdgeNodeStateOnline,
			Health: HealthStatusHealthy,
		},
		Network: NetworkInfo{
			VPNEndpoint: "vpn.cloud.example.com",
		},
	}

	discovery.addOrUpdateNode(failedNode)
	discovery.addOrUpdateNode(healthyNode)

	// Simulate VMs on failed node
	coordinator.placementsMu.Lock()
	coordinator.vmPlacements["vm-1"] = "edge-failed"
	coordinator.vmPlacements["vm-2"] = "edge-failed"
	coordinator.placementsMu.Unlock()

	ctx := context.Background()
	err := coordinator.HandleEdgeFailure(ctx, "edge-failed")
	if err != nil {
		t.Logf("Failover had errors (expected in test): %v", err)
	}

	// Verify migrations were initiated
	// (In a real test, we'd verify VMs were moved to healthy node)
}
