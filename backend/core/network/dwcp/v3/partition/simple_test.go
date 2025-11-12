package partition

import (
	"context"
	"testing"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/upgrade"
)

// TestSimpleITPv3 tests basic ITPv3 functionality without external dependencies
func TestSimpleITPv3(t *testing.T) {
	// Create ITPv3 without DQN dependency
	itp := &ITPv3{
		mode:                upgrade.ModeInternet,
		internetPlacer:     NewGeographicPlacer(),
		nodes:              make(map[string]*Node),
		regions:            make(map[string]*Region),
	}

	// Add test node
	node := &Node{
		ID:              "test-node",
		Name:            "test",
		Type:            NodeTypeCloud,
		Region:          "us-west",
		TotalCPU:        16,
		TotalMemory:     32 * 1e9,
		TotalDisk:       1000 * 1e9,
		AvailableCPU:    16,
		AvailableMemory: 32 * 1e9,
		AvailableDisk:   1000 * 1e9,
		NetworkBandwidth: 10.0,
		CostPerHour:     0.5,
		Uptime:          time.Hour * 24 * 30,
		Labels:          make(map[string]string),
	}
	itp.AddNode(node)

	// Create VM
	vm := &VM{
		ID:              "test-vm",
		Name:            "test-vm",
		RequestedCPU:    4,
		RequestedMemory: 8 * 1e9,
		RequestedDisk:   100 * 1e9,
		Priority:        0.8,
	}

	// Place VM
	ctx := context.Background()
	placedNode, err := itp.PlaceVM(ctx, vm, nil)
	if err != nil {
		t.Fatalf("Failed to place VM: %v", err)
	}

	if placedNode == nil {
		t.Fatal("Placement returned nil node")
	}

	if placedNode.ID != "test-node" {
		t.Errorf("Expected node ID 'test-node', got '%s'", placedNode.ID)
	}

	// Check resource allocation
	if node.AvailableCPU != 12 {
		t.Errorf("Expected 12 available CPUs, got %d", node.AvailableCPU)
	}

	if node.AvailableMemory != 24*1e9 {
		t.Errorf("Expected 24GB available memory, got %d", node.AvailableMemory)
	}

	// Check metrics
	metrics := itp.GetMetrics()
	if metrics["placement_success"] != int64(1) {
		t.Errorf("Expected 1 successful placement, got %v", metrics["placement_success"])
	}

	t.Logf("Successfully placed VM on node %s", placedNode.Name)
	t.Logf("Resource utilization: %.2f%%", metrics["resource_utilization"].(float64)*100)
}

// TestGeographicOptimizer tests the geographic optimizer
func TestGeographicOptimizer(t *testing.T) {
	optimizer := NewGeographicOptimizer()

	// Add regions
	regions := []*Region{
		{
			ID:        "us-west",
			Name:      "US West",
			Latitude:  37.7749,
			Longitude: -122.4194,
		},
		{
			ID:              "europe",
			Name:            "Europe",
			Latitude:        50.1109,
			Longitude:       8.6821,
			DataSovereignty: true,
			ComplianceZone:  "gdpr",
		},
	}

	for _, region := range regions {
		optimizer.AddRegion(region)
	}

	// Create nodes in different regions
	nodes := []*Node{
		{
			ID:              "us-node",
			Region:          "us-west",
			Type:            NodeTypeCloud,
			TotalCPU:        16,
			AvailableCPU:    16,
			TotalMemory:     32 * 1e9,
			AvailableMemory: 32 * 1e9,
			CostPerHour:     1.0,
			Uptime:          time.Hour * 24 * 100,
		},
		{
			ID:              "eu-node",
			Region:          "europe",
			Type:            NodeTypeCloud,
			TotalCPU:        16,
			AvailableCPU:    16,
			TotalMemory:     32 * 1e9,
			AvailableMemory: 32 * 1e9,
			CostPerHour:     1.2,
			Uptime:          time.Hour * 24 * 150,
			Labels: map[string]string{
				"compliance": "gdpr",
			},
		},
	}

	// Test VM with GDPR requirement
	vm := &VM{
		ID:              "gdpr-vm",
		RequestedCPU:    4,
		RequestedMemory: 8 * 1e9,
		RequiredLabels: map[string]string{
			"data-sovereignty": "gdpr",
		},
	}

	// Find optimal placement
	optimal, err := optimizer.OptimalPlacement(vm, nil, nodes)
	if err != nil {
		t.Fatalf("Failed to find optimal placement: %v", err)
	}

	// Should prefer EU node for GDPR compliance
	if optimal.ID != "eu-node" {
		t.Errorf("Expected EU node for GDPR compliance, got %s", optimal.ID)
	}

	t.Logf("Successfully placed GDPR-compliant VM on %s", optimal.ID)
}

// TestHeterogeneousPlacement tests heterogeneous node placement
func TestHeterogeneousPlacement(t *testing.T) {
	engine := NewHeterogeneousPlacementEngine()

	// Register node capabilities
	capabilities := []*NodeCapabilities{
		{
			NodeID:          "gpu-node",
			NodeType:        NodeTypeDatacenter,
			CPUArchitecture: "x86_64",
			GPUTypes:        []string{"nvidia-v100", "nvidia-a100"},
			HasNVME:         true,
			HasRDMA:         true,
			MaxBandwidth:    100.0,
			MinLatency:      1 * time.Millisecond,
			SLAGuarantee:    99.99,
		},
		{
			NodeID:          "edge-node",
			NodeType:        NodeTypeEdge,
			CPUArchitecture: "arm64",
			MaxBandwidth:    1.0,
			MinLatency:      20 * time.Millisecond,
			SLAGuarantee:    99.0,
		},
	}

	for _, cap := range capabilities {
		engine.RegisterNodeCapabilities(cap.NodeID, cap)
	}

	// Create nodes
	nodes := []*Node{
		{
			ID:              "gpu-node",
			Type:            NodeTypeDatacenter,
			TotalCPU:        32,
			AvailableCPU:    32,
			TotalMemory:     128 * 1e9,
			AvailableMemory: 128 * 1e9,
			TotalGPU:        8,
			AvailableGPU:    8,
			NetworkBandwidth: 100.0,
		},
		{
			ID:              "edge-node",
			Type:            NodeTypeEdge,
			TotalCPU:        8,
			AvailableCPU:    8,
			TotalMemory:     16 * 1e9,
			AvailableMemory: 16 * 1e9,
			NetworkBandwidth: 1.0,
		},
	}

	// Test VM requiring GPU
	vm := &VM{
		ID:              "ml-vm",
		RequestedCPU:    8,
		RequestedMemory: 32 * 1e9,
		RequestedGPU:    2,
		RequiredLabels: map[string]string{
			"gpu-type": "nvidia-v100",
		},
	}

	// Place VM
	ctx := context.Background()
	node, err := engine.PlaceVM(ctx, vm, nodes)
	if err != nil {
		t.Fatalf("Failed to place VM: %v", err)
	}

	// Should select GPU node
	if node.ID != "gpu-node" {
		t.Errorf("Expected GPU node for ML workload, got %s", node.ID)
	}

	t.Logf("Successfully placed ML VM on %s with GPU support", node.ID)
}

// TestResourceUtilizationCalculation tests resource utilization metrics
func TestResourceUtilizationCalculation(t *testing.T) {
	itp := &ITPv3{
		mode:           upgrade.ModeDatacenter,
		nodes:          make(map[string]*Node),
		regions:        make(map[string]*Region),
	}

	// Add nodes with different utilization levels
	nodes := []*Node{
		{
			ID:              "node-1",
			TotalCPU:        16,
			AvailableCPU:    8,  // 50% used
			TotalMemory:     32 * 1e9,
			AvailableMemory: 16 * 1e9, // 50% used
		},
		{
			ID:              "node-2",
			TotalCPU:        32,
			AvailableCPU:    8,  // 75% used
			TotalMemory:     64 * 1e9,
			AvailableMemory: 32 * 1e9, // 50% used
		},
	}

	for _, node := range nodes {
		itp.AddNode(node)
	}

	// Update utilization
	itp.updateResourceUtilization()

	// Calculate expected utilization
	// CPU: (8/16 + 24/32) / 2 = (0.5 + 0.75) / 2 = 0.625
	// Memory: (16/32 + 32/64) / 2 = (0.5 + 0.5) / 2 = 0.5
	// Average: (0.625 + 0.5) / 2 = 0.5625

	expectedUtil := 0.5625
	actualUtil := itp.resourceUtilization

	if actualUtil < expectedUtil-0.01 || actualUtil > expectedUtil+0.01 {
		t.Errorf("Expected utilization %.4f, got %.4f", expectedUtil, actualUtil)
	}

	t.Logf("Resource utilization: %.2f%%", actualUtil*100)
}

// BenchmarkSimplePlacement benchmarks basic placement performance
func BenchmarkSimplePlacement(b *testing.B) {
	itp := &ITPv3{
		mode:           upgrade.ModeInternet,
		internetPlacer: NewGeographicPlacer(),
		nodes:          make(map[string]*Node),
		regions:        make(map[string]*Region),
	}

	// Add 100 nodes
	for i := 0; i < 100; i++ {
		node := &Node{
			ID:              string(rune('a' + (i % 26))),
			Name:            string(rune('a' + (i % 26))),
			Type:            NodeType(i % 4),
			Region:          []string{"us-west", "us-east", "europe", "asia"}[i%4],
			TotalCPU:        16,
			TotalMemory:     32 * 1e9,
			AvailableCPU:    16,
			AvailableMemory: 32 * 1e9,
			NetworkBandwidth: 10.0,
			CostPerHour:     0.5,
			Uptime:          time.Hour * 24 * 30,
			Labels:          make(map[string]string),
		}
		itp.AddNode(node)
	}

	ctx := context.Background()
	vm := &VM{
		ID:              "bench-vm",
		RequestedCPU:    4,
		RequestedMemory: 8 * 1e9,
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		node, err := itp.PlaceVM(ctx, vm, nil)
		if err == nil && node != nil {
			// Clean up for next iteration
			itp.deallocateResources(node, vm)
		}
	}

	metrics := itp.GetMetrics()
	b.ReportMetric(float64(metrics["placement_latency"].(int64)), "ms/op")
}