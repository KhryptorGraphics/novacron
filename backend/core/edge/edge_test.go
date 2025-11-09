package edge

import (
	"context"
	"testing"
	"time"
)

func TestEdgeComputing(t *testing.T) {
	config := DefaultEdgeConfig()
	config.DiscoveryInterval = 5 * time.Second

	ec, err := NewEdgeComputing(config)
	if err != nil {
		t.Fatalf("Failed to create edge computing: %v", err)
	}

	// Test Start
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	err = ec.Start(ctx)
	if err != nil {
		t.Fatalf("Failed to start edge computing: %v", err)
	}

	// Add test edge node
	testNode := &EdgeNode{
		ID:   "integration-test-node",
		Name: "Integration Test Node",
		Type: EdgeType5GMEC,
		Location: GeoLocation{
			Latitude:  37.7749,
			Longitude: -122.4194,
			Country:   "US",
			Region:    "us-west-1",
			City:      "San Francisco",
		},
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
			State:     EdgeNodeStateOnline,
			Health:    HealthStatusHealthy,
			ActiveVMs: 5,
		},
		Latency: LatencyMetrics{
			RTTAvg: 15 * time.Millisecond,
		},
		Cost: CostMetrics{
			CostPerHour: 0.5,
			Currency:    "USD",
		},
		Capabilities: EdgeCapability{
			Type:            EdgeType5GMEC,
			SupportsGPU:     false,
			SupportsARM64:   true,
			SupportsX86:     true,
			MaxVMs:          20,
			HasLocalStorage: true,
			NetworkSlicing:  true,
			UltraLowLatency: true,
		},
	}

	ec.Discovery.addOrUpdateNode(testNode)

	// Test DeployVM
	deployReq := &ProvisionRequest{
		VMID:  "integration-test-vm",
		Name:  "Integration Test VM",
		Image: "alpine:latest",
		Requirements: PlacementRequirements{
			CPUCores:     2,
			MemoryMB:     4096,
			StorageGB:    50,
			BandwidthMbps: 100,
		},
		Constraints: PlacementConstraints{
			MaxLatency: 100 * time.Millisecond,
		},
		UserLocation: &GeoLocation{
			Latitude:  37.7749,
			Longitude: -122.4194,
		},
		Metadata: map[string]string{
			"environment": "test",
		},
	}

	vm, err := ec.DeployVM(ctx, deployReq)
	if err != nil {
		t.Fatalf("Failed to deploy VM: %v", err)
	}

	if vm.VMID != "integration-test-vm" {
		t.Errorf("Expected VM ID integration-test-vm, got %s", vm.VMID)
	}

	if vm.State != VMStateRunning {
		t.Errorf("Expected VM state running, got %s", vm.State)
	}

	t.Logf("VM deployed: ID=%s, Node=%s, ProvisionTime=%v",
		vm.VMID, vm.EdgeNodeID, vm.ProvisionTime)

	// Test GetStatus
	status, err := ec.GetStatus(ctx)
	if err != nil {
		t.Fatalf("Failed to get status: %v", err)
	}

	if status.Dashboard.TotalEdgeNodes == 0 {
		t.Error("Expected non-zero edge nodes")
	}

	t.Logf("Edge Status: Nodes=%d, VMs=%d, CPUUtil=%.2f%%",
		status.Dashboard.TotalEdgeNodes,
		status.Dashboard.TotalVMs,
		status.Dashboard.CPUUtilizationPct)

	// Test Stop
	err = ec.Stop()
	if err != nil {
		t.Fatalf("Failed to stop edge computing: %v", err)
	}
}

func TestConfigValidation(t *testing.T) {
	tests := []struct {
		name      string
		config    *EdgeConfig
		expectErr bool
	}{
		{
			name:      "valid config",
			config:    DefaultEdgeConfig(),
			expectErr: false,
		},
		{
			name: "invalid discovery interval",
			config: &EdgeConfig{
				DiscoveryInterval: 0,
				PlacementWeights: PlacementWeights{
					Latency:   0.5,
					Resources: 0.3,
					Cost:      0.2,
				},
				MaxEdgeLatency:   100 * time.Millisecond,
				MigrationTimeout: 5 * time.Second,
				MinEdgeResources: ResourceReq{
					MinCPUCores: 1,
				},
			},
			expectErr: true,
		},
		{
			name: "invalid placement weights",
			config: &EdgeConfig{
				DiscoveryInterval: 30 * time.Second,
				PlacementWeights: PlacementWeights{
					Latency:   0.5,
					Resources: 0.3,
					Cost:      0.1, // Sum = 0.9, should be 1.0
				},
				MaxEdgeLatency:   100 * time.Millisecond,
				MigrationTimeout: 5 * time.Second,
				MinEdgeResources: ResourceReq{
					MinCPUCores: 1,
				},
			},
			expectErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.config.Validate()
			if tt.expectErr && err == nil {
				t.Error("Expected error but got none")
			}
			if !tt.expectErr && err != nil {
				t.Errorf("Unexpected error: %v", err)
			}
		})
	}
}

func TestPerformanceTargets(t *testing.T) {
	config := DefaultEdgeConfig()
	ec, err := NewEdgeComputing(config)
	if err != nil {
		t.Fatalf("Failed to create edge computing: %v", err)
	}

	// Add test node
	testNode := &EdgeNode{
		ID:   "perf-test-node",
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
			UtilizationPercent: 20.0,
		},
		Status: EdgeNodeStatus{
			State:  EdgeNodeStateOnline,
			Health: HealthStatusHealthy,
		},
		Latency: LatencyMetrics{
			RTTAvg: 10 * time.Millisecond,
		},
	}

	ec.Discovery.addOrUpdateNode(testNode)

	// Test placement decision time (<100ms target)
	req := &PlacementRequest{
		VMID: "perf-test-vm",
		Requirements: PlacementRequirements{
			CPUCores:  2,
			MemoryMB:  4096,
			StorageGB: 50,
		},
	}

	ctx := context.Background()
	start := time.Now()
	decision, err := ec.Placement.PlaceVM(ctx, req)
	placementTime := time.Since(start)

	if err != nil {
		t.Fatalf("Placement failed: %v", err)
	}

	// Target: <100ms, ideal: <50ms
	if placementTime > 100*time.Millisecond {
		t.Errorf("Placement time %v exceeds 100ms target", placementTime)
	} else if placementTime > 50*time.Millisecond {
		t.Logf("Warning: Placement time %v exceeds 50ms ideal", placementTime)
	} else {
		t.Logf("✓ Placement time %v meets performance target", placementTime)
	}

	// Test provision time (<30s target)
	start = time.Now()
	_, err = ec.VMLifecycle.ProvisionVM(ctx, &ProvisionRequest{
		VMID:  "perf-provision-vm",
		Name:  "Performance Test VM",
		Image: "alpine:latest",
		Requirements: PlacementRequirements{
			CPUCores:  2,
			MemoryMB:  4096,
			StorageGB: 50,
		},
	})
	provisionTime := time.Since(start)

	if err != nil {
		t.Fatalf("Provision failed: %v", err)
	}

	// Target: <30s
	if provisionTime > 30*time.Second {
		t.Errorf("Provision time %v exceeds 30s target", provisionTime)
	} else {
		t.Logf("✓ Provision time %v meets performance target", provisionTime)
	}

	t.Logf("\nPerformance Summary:")
	t.Logf("  Placement Decision: %v (target: <100ms, ideal: <50ms)", placementTime)
	t.Logf("  VM Provisioning:    %v (target: <30s)", provisionTime)
	t.Logf("  Edge Latency:       %v (target: <100ms)", decision.EstimatedLatency)
}
