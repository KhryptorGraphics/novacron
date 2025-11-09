package edge

import (
	"context"
	"testing"
	"time"
)

func TestPlacementEngine(t *testing.T) {
	config := DefaultEdgeConfig()
	discovery := NewEdgeDiscovery(config)
	placement := NewPlacementEngine(config, discovery)

	// Add test edge nodes
	nodes := []*EdgeNode{
		{
			ID:   "edge-us-west-1",
			Type: EdgeType5GMEC,
			Location: GeoLocation{
				Latitude:  37.7749,
				Longitude: -122.4194,
				Country:   "US",
				Region:    "us-west-1",
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
				State:  EdgeNodeStateOnline,
				Health: HealthStatusHealthy,
			},
			Latency: LatencyMetrics{
				RTTAvg: 10 * time.Millisecond,
			},
			Cost: CostMetrics{
				CostPerHour: 0.5,
			},
		},
		{
			ID:   "edge-us-east-1",
			Type: EdgeTypeCDN,
			Location: GeoLocation{
				Latitude:  40.7128,
				Longitude: -74.0060,
				Country:   "US",
				Region:    "us-east-1",
			},
			Resources: EdgeResources{
				TotalCPUCores:      8,
				UsedCPUCores:       6,
				TotalMemoryMB:      16384,
				UsedMemoryMB:       12288,
				TotalStorageGB:     500,
				UsedStorageGB:      300,
				TotalBandwidthMbps: 5000,
				UsedBandwidthMbps:  3000,
				UtilizationPercent: 75.0,
			},
			Status: EdgeNodeStatus{
				State:  EdgeNodeStateOnline,
				Health: HealthStatusHealthy,
			},
			Latency: LatencyMetrics{
				RTTAvg: 50 * time.Millisecond,
			},
			Cost: CostMetrics{
				CostPerHour: 0.3,
			},
		},
	}

	for _, node := range nodes {
		discovery.addOrUpdateNode(node)
	}

	// Test placement request
	req := &PlacementRequest{
		VMID: "test-vm-1",
		UserLocation: &GeoLocation{
			Latitude:  37.7749,
			Longitude: -122.4194,
		},
		Requirements: PlacementRequirements{
			CPUCores:  2,
			MemoryMB:  4096,
			StorageGB: 50,
		},
		Constraints: PlacementConstraints{
			MaxLatency: 100 * time.Millisecond,
		},
	}

	ctx := context.Background()
	decision, err := placement.PlaceVM(ctx, req)
	if err != nil {
		t.Fatalf("Placement failed: %v", err)
	}

	if decision.EdgeNodeID == "" {
		t.Error("Expected non-empty edge node ID")
	}

	if decision.Score <= 0 {
		t.Error("Expected positive placement score")
	}

	t.Logf("Placement decision: Node=%s, Score=%.2f, Latency=%v, Reason=%s",
		decision.EdgeNodeID, decision.Score, decision.EstimatedLatency, decision.Reason)

	// Verify decision time meets target
	if decision.DecisionTime > config.TargetPlacementTime {
		t.Logf("Warning: Placement decision time %v exceeds target %v",
			decision.DecisionTime, config.TargetPlacementTime)
	}
}

func TestPlacementConstraints(t *testing.T) {
	config := DefaultEdgeConfig()
	discovery := NewEdgeDiscovery(config)
	placement := NewPlacementEngine(config, discovery)

	// Add EU node
	euNode := &EdgeNode{
		ID:   "edge-eu-1",
		Type: EdgeType5GMEC,
		Location: GeoLocation{
			Latitude:  48.8566,
			Longitude: 2.3522,
			Country:   "FR",
			Region:    "eu-west-1",
		},
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
			RTTAvg: 15 * time.Millisecond,
		},
		Capabilities: EdgeCapability{
			SupportsX86: true,
		},
	}

	discovery.addOrUpdateNode(euNode)

	// Test data residency constraint
	req := &PlacementRequest{
		VMID: "test-vm-gdpr",
		Requirements: PlacementRequirements{
			CPUCores:  2,
			MemoryMB:  4096,
			StorageGB: 50,
		},
		Constraints: PlacementConstraints{
			DataResidency: "EU",
		},
	}

	ctx := context.Background()
	decision, err := placement.PlaceVM(ctx, req)
	if err != nil {
		t.Fatalf("GDPR placement failed: %v", err)
	}

	if decision.EdgeNodeID != "edge-eu-1" {
		t.Errorf("Expected EU node for GDPR compliance, got %s", decision.EdgeNodeID)
	}
}

func TestCalculateDistance(t *testing.T) {
	config := DefaultEdgeConfig()
	discovery := NewEdgeDiscovery(config)
	placement := NewPlacementEngine(config, discovery)

	// San Francisco to New York
	sf := &GeoLocation{Latitude: 37.7749, Longitude: -122.4194}
	ny := &GeoLocation{Latitude: 40.7128, Longitude: -74.0060}

	distance := placement.calculateDistance(sf, ny)

	// Should be approximately 4135 km
	if distance < 4000 || distance > 4300 {
		t.Errorf("Expected distance ~4135 km, got %.2f km", distance)
	}
}

func TestRecommendPlacement(t *testing.T) {
	config := DefaultEdgeConfig()
	discovery := NewEdgeDiscovery(config)
	placement := NewPlacementEngine(config, discovery)

	// Add multiple nodes
	for i := 1; i <= 5; i++ {
		node := &EdgeNode{
			ID:   fmt.Sprintf("edge-%d", i),
			Type: EdgeType5GMEC,
			Resources: EdgeResources{
				TotalCPUCores:      16,
				UsedCPUCores:       i * 2,
				TotalMemoryMB:      32768,
				UsedMemoryMB:       int64(i * 4096),
				TotalStorageGB:     1000,
				UsedStorageGB:      int64(i * 100),
				TotalBandwidthMbps: 10000,
				UsedBandwidthMbps:  i * 1000,
				UtilizationPercent: float64(i * 20),
			},
			Status: EdgeNodeStatus{
				State:  EdgeNodeStateOnline,
				Health: HealthStatusHealthy,
			},
			Latency: LatencyMetrics{
				RTTAvg: time.Duration(i*10) * time.Millisecond,
			},
			Cost: CostMetrics{
				CostPerHour: float64(i) * 0.1,
			},
		}
		discovery.addOrUpdateNode(node)
	}

	req := &PlacementRequest{
		VMID: "test-vm-1",
		Requirements: PlacementRequirements{
			CPUCores:  2,
			MemoryMB:  4096,
			StorageGB: 50,
		},
	}

	ctx := context.Background()
	recommendations, err := placement.RecommendPlacement(ctx, req, 3)
	if err != nil {
		t.Fatalf("Failed to get recommendations: %v", err)
	}

	if len(recommendations) != 3 {
		t.Errorf("Expected 3 recommendations, got %d", len(recommendations))
	}

	// Verify recommendations are sorted by score
	for i := 1; i < len(recommendations); i++ {
		if recommendations[i].Score > recommendations[i-1].Score {
			t.Error("Recommendations not sorted by score")
		}
	}
}
