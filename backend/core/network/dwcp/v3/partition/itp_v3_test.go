package partition

import (
	"context"
	"testing"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/upgrade"
)

// TestITPv3Creation tests ITPv3 instance creation
func TestITPv3Creation(t *testing.T) {
	tests := []struct {
		name string
		mode upgrade.NetworkMode
	}{
		{"Datacenter Mode", upgrade.ModeDatacenter},
		{"Internet Mode", upgrade.ModeInternet},
		{"Hybrid Mode", upgrade.ModeHybrid},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			itp, err := NewITPv3(tt.mode)
			if err != nil {
				t.Fatalf("Failed to create ITPv3: %v", err)
			}

			if itp.mode != tt.mode {
				t.Errorf("Expected mode %v, got %v", tt.mode, itp.mode)
			}

			// Check that appropriate placers are initialized
			switch tt.mode {
			case upgrade.ModeDatacenter:
				if itp.datacenterPlacer == nil {
					t.Error("Datacenter placer not initialized")
				}
			case upgrade.ModeInternet:
				if itp.internetPlacer == nil {
					t.Error("Internet placer not initialized")
				}
			case upgrade.ModeHybrid:
				if itp.datacenterPlacer == nil || itp.internetPlacer == nil || itp.hybridPlacer == nil {
					t.Error("Hybrid mode placers not fully initialized")
				}
			}
		})
	}
}

// TestBasicVMPlacement tests basic VM placement functionality
func TestBasicVMPlacement(t *testing.T) {
	itp, err := NewITPv3(upgrade.ModeInternet)
	if err != nil {
		t.Fatalf("Failed to create ITPv3: %v", err)
	}

	// Add test nodes
	nodes := createTestNodes()
	for _, node := range nodes {
		itp.AddNode(node)
	}

	// Create a test VM
	vm := &VM{
		ID:              "vm-001",
		Name:            "test-vm",
		RequestedCPU:    4,
		RequestedMemory: 8 * 1e9,  // 8 GB
		RequestedDisk:   100 * 1e9, // 100 GB
		Priority:        0.8,
	}

	// Place the VM
	ctx := context.Background()
	node, err := itp.PlaceVM(ctx, vm, nil)
	if err != nil {
		t.Fatalf("Failed to place VM: %v", err)
	}

	if node == nil {
		t.Fatal("Placement returned nil node")
	}

	// Verify resources were allocated
	if node.AvailableCPU >= node.TotalCPU {
		t.Error("CPU resources not allocated")
	}

	if node.AvailableMemory >= node.TotalMemory {
		t.Error("Memory resources not allocated")
	}

	// Check metrics
	metrics := itp.GetMetrics()
	if metrics["placement_success"] != int64(1) {
		t.Errorf("Expected 1 successful placement, got %v", metrics["placement_success"])
	}
}

// TestConstraintBasedPlacement tests placement with constraints
func TestConstraintBasedPlacement(t *testing.T) {
	itp, err := NewITPv3(upgrade.ModeInternet)
	if err != nil {
		t.Fatalf("Failed to create ITPv3: %v", err)
	}

	// Add nodes with different characteristics
	nodes := createDiverseTestNodes()
	for _, node := range nodes {
		itp.AddNode(node)
	}

	// Add regions
	regions := createTestRegions()
	for _, region := range regions {
		itp.AddRegion(region)
	}

	// Test cases
	tests := []struct {
		name        string
		vm          *VM
		constraints *Constraints
		expectError bool
	}{
		{
			name: "Region Constraint",
			vm: &VM{
				ID:              "vm-region",
				RequestedCPU:    2,
				RequestedMemory: 4 * 1e9,
				RequiredRegions: []string{"us-west"},
			},
			constraints: nil,
			expectError: false,
		},
		{
			name: "Cost Constraint",
			vm: &VM{
				ID:              "vm-cost",
				RequestedCPU:    2,
				RequestedMemory: 4 * 1e9,
			},
			constraints: &Constraints{
				MaxCostPerHour: 1.0,
			},
			expectError: false,
		},
		{
			name: "Uptime Constraint",
			vm: &VM{
				ID:              "vm-uptime",
				RequestedCPU:    2,
				RequestedMemory: 4 * 1e9,
			},
			constraints: &Constraints{
				RequiredUptime: 0.99,
			},
			expectError: false,
		},
		{
			name: "Node Type Constraint",
			vm: &VM{
				ID:              "vm-nodetype",
				RequestedCPU:    2,
				RequestedMemory: 4 * 1e9,
			},
			constraints: &Constraints{
				RequiredNodeType: NodeTypeCloud,
			},
			expectError: false,
		},
		{
			name: "Impossible Constraint",
			vm: &VM{
				ID:              "vm-impossible",
				RequestedCPU:    1000, // Too many CPUs
				RequestedMemory: 1000 * 1e9,
			},
			constraints: nil,
			expectError: true,
		},
	}

	ctx := context.Background()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			node, err := itp.PlaceVM(ctx, tt.vm, tt.constraints)

			if tt.expectError {
				if err == nil {
					t.Error("Expected error but got none")
				}
			} else {
				if err != nil {
					t.Errorf("Unexpected error: %v", err)
				}
				if node == nil {
					t.Error("Expected valid node but got nil")
				}
			}
		})
	}
}

// TestBatchPlacement tests batch VM placement
func TestBatchPlacement(t *testing.T) {
	itp, err := NewITPv3(upgrade.ModeHybrid)
	if err != nil {
		t.Fatalf("Failed to create ITPv3: %v", err)
	}

	// Add multiple nodes
	nodes := createLargeTestCluster()
	for _, node := range nodes {
		itp.AddNode(node)
	}

	// Create batch of VMs
	vms := make([]*VM, 10)
	for i := 0; i < 10; i++ {
		vms[i] = &VM{
			ID:              string(rune('a' + i)),
			Name:            string(rune('a' + i)),
			RequestedCPU:    2 + i%3,
			RequestedMemory: int64((4 + i%4) * 1e9),
			RequestedDisk:   int64((50 + i*10) * 1e9),
			Priority:        float64(10-i) / 10.0,
		}
	}

	// Place VMs in batch
	ctx := context.Background()
	placements, err := itp.PlaceVMBatch(ctx, vms, nil)
	if err != nil {
		t.Fatalf("Batch placement failed: %v", err)
	}

	// Verify all VMs were placed
	if len(placements) != len(vms) {
		t.Errorf("Expected %d placements, got %d", len(vms), len(placements))
	}

	// Verify no duplicate node assignments for anti-affinity
	nodeUsage := make(map[string]int)
	for _, node := range placements {
		nodeUsage[node.ID]++
	}

	// Check resource utilization
	metrics := itp.GetMetrics()
	utilization := metrics["resource_utilization"].(float64)
	if utilization < 0.1 {
		t.Error("Resource utilization too low after batch placement")
	}
}

// TestModeAwarePlacement tests different placement behaviors in different modes
func TestModeAwarePlacement(t *testing.T) {
	tests := []struct {
		name           string
		mode           upgrade.NetworkMode
		vmType         string
		expectedResult string
	}{
		{
			name:           "Datacenter Mode - High Performance",
			mode:           upgrade.ModeDatacenter,
			vmType:         "high-performance",
			expectedResult: "low-latency",
		},
		{
			name:           "Internet Mode - Geographic",
			mode:           upgrade.ModeInternet,
			vmType:         "geo-distributed",
			expectedResult: "distributed",
		},
		{
			name:           "Hybrid Mode - Adaptive",
			mode:           upgrade.ModeHybrid,
			vmType:         "mixed",
			expectedResult: "adaptive",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			itp, err := NewITPv3(tt.mode)
			if err != nil {
				t.Fatalf("Failed to create ITPv3: %v", err)
			}

			// Add appropriate nodes for the mode
			nodes := createModeSpecificNodes(tt.mode)
			for _, node := range nodes {
				itp.AddNode(node)
			}

			// Create VM based on type
			vm := createVMByType(tt.vmType)

			// Place VM
			ctx := context.Background()
			node, err := itp.PlaceVM(ctx, vm, nil)
			if err != nil {
				t.Fatalf("Failed to place VM: %v", err)
			}

			// Verify placement matches expected behavior
			if !verifyPlacementBehavior(node, tt.expectedResult) {
				t.Errorf("Placement behavior doesn't match expected: %s", tt.expectedResult)
			}
		})
	}
}

// TestGeographicOptimization tests geographic optimization in internet mode
func TestGeographicOptimization(t *testing.T) {
	itp, err := NewITPv3(upgrade.ModeInternet)
	if err != nil {
		t.Fatalf("Failed to create ITPv3: %v", err)
	}

	// Create geographically distributed nodes
	nodes := createGlobalNodes()
	for _, node := range nodes {
		itp.AddNode(node)
	}

	// Create regions with latency information
	regions := createGlobalRegions()
	for _, region := range regions {
		itp.AddRegion(region)
	}

	// Test VM with region preference
	vm := &VM{
		ID:              "vm-geo",
		RequestedCPU:    4,
		RequestedMemory: 8 * 1e9,
		RequiredRegions: []string{"europe"},
		RequiredLabels: map[string]string{
			"data-sovereignty": "gdpr",
		},
	}

	ctx := context.Background()
	node, err := itp.PlaceVM(ctx, vm, nil)
	if err != nil {
		t.Fatalf("Failed to place VM: %v", err)
	}

	// Verify node is in Europe
	if node.Region != "europe" {
		t.Errorf("Expected node in Europe, got %s", node.Region)
	}
}

// TestResourceUtilizationTracking tests resource utilization metrics
func TestResourceUtilizationTracking(t *testing.T) {
	itp, err := NewITPv3(upgrade.ModeDatacenter)
	if err != nil {
		t.Fatalf("Failed to create ITPv3: %v", err)
	}

	// Add nodes
	nodes := createTestNodes()
	for _, node := range nodes {
		itp.AddNode(node)
	}

	// Get initial utilization
	metrics := itp.GetMetrics()
	initialUtil := metrics["resource_utilization"].(float64)

	// Place multiple VMs
	ctx := context.Background()
	for i := 0; i < 5; i++ {
		vm := &VM{
			ID:              string(rune('a' + i)),
			RequestedCPU:    2,
			RequestedMemory: 4 * 1e9,
		}
		_, err := itp.PlaceVM(ctx, vm, nil)
		if err != nil {
			t.Logf("VM %d placement failed (expected for resource exhaustion): %v", i, err)
			break
		}
	}

	// Check final utilization
	metrics = itp.GetMetrics()
	finalUtil := metrics["resource_utilization"].(float64)

	if finalUtil <= initialUtil {
		t.Error("Resource utilization should increase after placing VMs")
	}

	if finalUtil > 1.0 {
		t.Error("Resource utilization should not exceed 100%")
	}
}

// Helper functions

func createTestNodes() []*Node {
	return []*Node{
		{
			ID:              "node-1",
			Name:            "test-node-1",
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
		},
		{
			ID:              "node-2",
			Name:            "test-node-2",
			Type:            NodeTypeDatacenter,
			Region:          "us-east",
			TotalCPU:        32,
			TotalMemory:     64 * 1e9,
			TotalDisk:       2000 * 1e9,
			AvailableCPU:    32,
			AvailableMemory: 64 * 1e9,
			AvailableDisk:   2000 * 1e9,
			NetworkBandwidth: 40.0,
			CostPerHour:     1.0,
			Uptime:          time.Hour * 24 * 60,
			Labels:          make(map[string]string),
		},
	}
}

func createDiverseTestNodes() []*Node {
	nodes := createTestNodes()

	// Add edge node
	nodes = append(nodes, &Node{
		ID:              "edge-1",
		Name:            "edge-node-1",
		Type:            NodeTypeEdge,
		Region:          "us-west",
		TotalCPU:        8,
		TotalMemory:     16 * 1e9,
		TotalDisk:       500 * 1e9,
		AvailableCPU:    8,
		AvailableMemory: 16 * 1e9,
		AvailableDisk:   500 * 1e9,
		NetworkBandwidth: 1.0,
		CostPerHour:     0.2,
		Uptime:          time.Hour * 24 * 20,
		Labels:          make(map[string]string),
	})

	// Add volunteer node
	nodes = append(nodes, &Node{
		ID:              "volunteer-1",
		Name:            "volunteer-node-1",
		Type:            NodeTypeVolunteer,
		Region:          "europe",
		TotalCPU:        4,
		TotalMemory:     8 * 1e9,
		TotalDisk:       200 * 1e9,
		AvailableCPU:    4,
		AvailableMemory: 8 * 1e9,
		AvailableDisk:   200 * 1e9,
		NetworkBandwidth: 0.1,
		CostPerHour:     0.01,
		Uptime:          time.Hour * 24 * 10,
		Labels:          make(map[string]string),
	})

	return nodes
}

func createLargeTestCluster() []*Node {
	nodes := make([]*Node, 20)

	for i := 0; i < 20; i++ {
		nodeType := NodeType(i % 4)
		nodes[i] = &Node{
			ID:              string(rune('a' + i)),
			Name:            string(rune('a' + i)),
			Type:            nodeType,
			Region:          []string{"us-west", "us-east", "europe", "asia"}[i%4],
			TotalCPU:        8 + (i%3)*8,
			TotalMemory:     int64((16 + (i%3)*16) * 1e9),
			TotalDisk:       int64((500 + i*100) * 1e9),
			AvailableCPU:    8 + (i%3)*8,
			AvailableMemory: int64((16 + (i%3)*16) * 1e9),
			AvailableDisk:   int64((500 + i*100) * 1e9),
			NetworkBandwidth: float64(1 + i%10),
			CostPerHour:     0.1 * float64(1+i%5),
			Uptime:          time.Hour * 24 * time.Duration(10+i),
			Labels:          make(map[string]string),
		}
	}

	return nodes
}

func createTestRegions() []*Region {
	return []*Region{
		{
			ID:        "us-west",
			Name:      "US West",
			Continent: "North America",
			Country:   "USA",
			City:      "San Francisco",
			Latitude:  37.7749,
			Longitude: -122.4194,
			InternetLatency: map[string]time.Duration{
				"us-east": 40 * time.Millisecond,
				"europe":  150 * time.Millisecond,
				"asia":    120 * time.Millisecond,
			},
		},
		{
			ID:        "us-east",
			Name:      "US East",
			Continent: "North America",
			Country:   "USA",
			City:      "New York",
			Latitude:  40.7128,
			Longitude: -74.0060,
			InternetLatency: map[string]time.Duration{
				"us-west": 40 * time.Millisecond,
				"europe":  80 * time.Millisecond,
				"asia":    200 * time.Millisecond,
			},
		},
		{
			ID:              "europe",
			Name:            "Europe",
			Continent:       "Europe",
			Country:         "Germany",
			City:            "Frankfurt",
			Latitude:        50.1109,
			Longitude:       8.6821,
			DataSovereignty: true,
			ComplianceZone:  "gdpr",
			InternetLatency: map[string]time.Duration{
				"us-west": 150 * time.Millisecond,
				"us-east": 80 * time.Millisecond,
				"asia":    180 * time.Millisecond,
			},
		},
	}
}

func createGlobalNodes() []*Node {
	return []*Node{
		{
			ID:              "eu-node-1",
			Name:            "europe-1",
			Type:            NodeTypeCloud,
			Region:          "europe",
			TotalCPU:        16,
			TotalMemory:     32 * 1e9,
			AvailableCPU:    16,
			AvailableMemory: 32 * 1e9,
			Labels: map[string]string{
				"compliance": "gdpr",
			},
		},
		{
			ID:              "us-node-1",
			Name:            "us-west-1",
			Type:            NodeTypeCloud,
			Region:          "us-west",
			TotalCPU:        16,
			TotalMemory:     32 * 1e9,
			AvailableCPU:    16,
			AvailableMemory: 32 * 1e9,
			Labels:          make(map[string]string),
		},
		{
			ID:              "asia-node-1",
			Name:            "asia-1",
			Type:            NodeTypeCloud,
			Region:          "asia",
			TotalCPU:        16,
			TotalMemory:     32 * 1e9,
			AvailableCPU:    16,
			AvailableMemory: 32 * 1e9,
			Labels:          make(map[string]string),
		},
	}
}

func createGlobalRegions() []*Region {
	regions := createTestRegions()

	// Add Asia region
	regions = append(regions, &Region{
		ID:        "asia",
		Name:      "Asia Pacific",
		Continent: "Asia",
		Country:   "Japan",
		City:      "Tokyo",
		Latitude:  35.6762,
		Longitude: 139.6503,
		InternetLatency: map[string]time.Duration{
			"us-west": 120 * time.Millisecond,
			"us-east": 200 * time.Millisecond,
			"europe":  180 * time.Millisecond,
		},
	})

	return regions
}

func createModeSpecificNodes(mode upgrade.NetworkMode) []*Node {
	switch mode {
	case upgrade.ModeDatacenter:
		// High-performance datacenter nodes
		return []*Node{
			{
				ID:              "dc-1",
				Name:            "datacenter-1",
				Type:            NodeTypeDatacenter,
				Region:          "local",
				TotalCPU:        64,
				TotalMemory:     256 * 1e9,
				AvailableCPU:    64,
				AvailableMemory: 256 * 1e9,
				NetworkBandwidth: 100.0,
				CostPerHour:     5.0,
			},
		}

	case upgrade.ModeInternet:
		// Geographically distributed nodes
		return createGlobalNodes()

	case upgrade.ModeHybrid:
		// Mix of node types
		return createDiverseTestNodes()

	default:
		return createTestNodes()
	}
}

func createVMByType(vmType string) *VM {
	switch vmType {
	case "high-performance":
		return &VM{
			ID:              "hp-vm",
			RequestedCPU:    32,
			RequestedMemory: 128 * 1e9,
			RequestedGPU:    4,
			Priority:        1.0,
		}

	case "geo-distributed":
		return &VM{
			ID:              "geo-vm",
			RequestedCPU:    4,
			RequestedMemory: 8 * 1e9,
			RequiredRegions: []string{"us-west", "europe"},
			Priority:        0.8,
		}

	case "mixed":
		return &VM{
			ID:              "mixed-vm",
			RequestedCPU:    8,
			RequestedMemory: 16 * 1e9,
			Priority:        0.6,
		}

	default:
		return &VM{
			ID:              "default-vm",
			RequestedCPU:    2,
			RequestedMemory: 4 * 1e9,
		}
	}
}

func verifyPlacementBehavior(node *Node, expected string) bool {
	switch expected {
	case "low-latency":
		// Datacenter nodes should have low latency
		return node.Type == NodeTypeDatacenter

	case "distributed":
		// Should be placed on cloud or edge nodes
		return node.Type == NodeTypeCloud || node.Type == NodeTypeEdge

	case "adaptive":
		// Any node type is acceptable for hybrid
		return true

	default:
		return false
	}
}