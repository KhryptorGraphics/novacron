package scheduler

import (
	"testing"

	"github.com/khryptorgraphics/novacron/backend/core/scheduler/network"
	"github.com/khryptorgraphics/novacron/backend/core/scheduler/workload"
)

func TestNetworkAwareScheduler_Creation(t *testing.T) {
	// Create dependencies
	baseScheduler := NewScheduler(DefaultSchedulerConfig())
	workloadAnalyzer := workload.NewWorkloadAnalyzer(workload.DefaultWorkloadAnalyzerConfig())
	networkTopology := network.NewNetworkTopology()

	// Create network-aware scheduler
	config := DefaultNetworkAwareSchedulerConfig()
	scheduler := NewNetworkAwareScheduler(
		config,
		baseScheduler,
		workloadAnalyzer,
		networkTopology,
	)

	if scheduler == nil {
		t.Fatalf("Failed to create NetworkAwareScheduler")
	}

	if scheduler.ResourceAwareScheduler == nil {
		t.Fatalf("ResourceAwareScheduler base not set")
	}

	if scheduler.networkTopology == nil {
		t.Fatalf("NetworkTopology not set")
	}
}

func TestNetworkAwareScheduler_VMPairKey(t *testing.T) {
	// Test that VM pair keys are consistent regardless of order
	key1 := getVMPairKey("vm-1", "vm-2")
	key2 := getVMPairKey("vm-2", "vm-1")

	if key1 != key2 {
		t.Errorf("VM pair key generation is not consistent: %s != %s", key1, key2)
	}
}

func TestNetworkAwareScheduler_TrackVMCommunication(t *testing.T) {
	// Create scheduler
	scheduler := createTestNetworkAwareScheduler()

	// Test tracking VM communication
	scheduler.TrackVMCommunication("vm-1", "vm-2", 100.0, 500.0)

	// Get the pair key
	key := getVMPairKey("vm-1", "vm-2")

	// Check if communication was recorded
	scheduler.vmCommunicationMutex.RLock()
	comm, exists := scheduler.vmCommunications[key]
	scheduler.vmCommunicationMutex.RUnlock()

	if !exists {
		t.Fatalf("Communication not recorded")
	}

	if comm.SourceVMID != "vm-1" || comm.DestinationVMID != "vm-2" {
		t.Errorf("Incorrect VM IDs recorded")
	}

	if comm.Bandwidth != 100.0 || comm.PacketRate != 500.0 {
		t.Errorf("Incorrect metrics recorded")
	}

	// Test updating existing communication
	scheduler.TrackVMCommunication("vm-1", "vm-2", 200.0, 1000.0)

	scheduler.vmCommunicationMutex.RLock()
	comm, exists = scheduler.vmCommunications[key]
	scheduler.vmCommunicationMutex.RUnlock()

	if !exists {
		t.Fatalf("Communication not recorded after update")
	}

	if comm.Bandwidth != 200.0 || comm.PacketRate != 1000.0 {
		t.Errorf("Metrics not updated correctly")
	}
}

func TestNetworkAwareScheduler_VMLocation(t *testing.T) {
	// Create scheduler
	scheduler := createTestNetworkAwareScheduler()

	// Test updating VM location
	scheduler.UpdateVMLocation("vm-1", "node-1")

	// Check if location was recorded
	scheduler.vmLocationCacheMutex.RLock()
	nodeID, exists := scheduler.vmLocationCache["vm-1"]
	scheduler.vmLocationCacheMutex.RUnlock()

	if !exists {
		t.Fatalf("VM location not recorded")
	}

	if nodeID != "node-1" {
		t.Errorf("Incorrect node ID recorded")
	}
}

func TestNetworkAwareScheduler_LatencyRequirement(t *testing.T) {
	// Create scheduler
	scheduler := createTestNetworkAwareScheduler()

	// Test setting latency requirement
	scheduler.SetVMLatencyRequirement("vm-1", "vm-2", 5.0)

	// Get the pair key
	key := getVMPairKey("vm-1", "vm-2")

	// Check if requirement was recorded
	scheduler.vmCommunicationMutex.RLock()
	comm, exists := scheduler.vmCommunications[key]
	scheduler.vmCommunicationMutex.RUnlock()

	if !exists {
		t.Fatalf("Latency requirement not recorded")
	}

	if comm.Latency != 5.0 {
		t.Errorf("Incorrect latency requirement recorded")
	}
}

func TestNetworkAwareScheduler_VMAffinityGroup(t *testing.T) {
	// Create scheduler
	scheduler := createTestNetworkAwareScheduler()

	// Test creating affinity group
	err := scheduler.CreateVMAffinityGroup("test-group", []string{"vm-1", "vm-2", "vm-3"})
	if err != nil {
		t.Fatalf("Failed to create affinity group: %v", err)
	}

	// Check if group was created
	scheduler.vmAffinityGroupMutex.RLock()
	vmIDs, exists := scheduler.vmAffinityGroups["test-group"]
	scheduler.vmAffinityGroupMutex.RUnlock()

	if !exists {
		t.Fatalf("Affinity group not created")
	}

	if len(vmIDs) != 3 {
		t.Errorf("Incorrect number of VMs in affinity group")
	}

	// Test creating duplicate group (should error)
	err = scheduler.CreateVMAffinityGroup("test-group", []string{"vm-4", "vm-5"})
	if err == nil {
		t.Errorf("Creating duplicate affinity group should fail")
	}
}

func TestNetworkAwareScheduler_ScoreNetworkTopology(t *testing.T) {
	// Create scheduler with topology
	scheduler := createTestNetworkAwareSchedulerWithTopology()

	// Add VMs with communication patterns
	scheduler.UpdateVMLocation("vm-1", "node-1")
	scheduler.UpdateVMLocation("vm-2", "node-2")
	scheduler.TrackVMCommunication("vm-1", "vm-2", 100.0, 500.0)

	// Score placement for a new VM that communicates with vm-1
	scheduler.TrackVMCommunication("vm-3", "vm-1", 200.0, 1000.0)

	// Get scores for different nodes
	score1 := scheduler.scoreNetworkTopology("vm-3", "node-1")
	score2 := scheduler.scoreNetworkTopology("vm-3", "node-2")
	score3 := scheduler.scoreNetworkTopology("vm-3", "node-3")

	// In this simplified test, placing vm-3 on node-1 should be better
	// because it communicates with vm-1 which is on node-1
	if score1 <= score2 || score1 <= score3 {
		t.Errorf("Network topology scoring incorrect: node-1 (%.2f) should score higher than node-2 (%.2f) and node-3 (%.2f)",
			score1, score2, score3)
	}
}

// Helper functions for tests

func createTestNetworkAwareScheduler() *NetworkAwareScheduler {
	baseScheduler := NewScheduler(DefaultSchedulerConfig())
	workloadAnalyzer := workload.NewWorkloadAnalyzer(workload.DefaultWorkloadAnalyzerConfig())
	networkTopology := network.NewNetworkTopology()

	config := DefaultNetworkAwareSchedulerConfig()
	return NewNetworkAwareScheduler(
		config,
		baseScheduler,
		workloadAnalyzer,
		networkTopology,
	)
}

func createTestNetworkAwareSchedulerWithTopology() *NetworkAwareScheduler {
	baseScheduler := NewScheduler(DefaultSchedulerConfig())
	workloadAnalyzer := workload.NewWorkloadAnalyzer(workload.DefaultWorkloadAnalyzerConfig())
	networkTopology := network.NewNetworkTopology()

	// Create a simple topology
	dc1 := network.NetworkLocation{
		Datacenter: "dc1",
		Zone:       "zone1",
		Rack:       "rack1",
	}

	// Add nodes
	for i := 1; i <= 3; i++ {
		nodeID := "node-" + string(rune('0'+i))
		networkTopology.AddNode(&network.NetworkNode{
			ID:       nodeID,
			Type:     "hypervisor",
			Location: dc1,
		})
	}

	// Add links
	networkTopology.AddLink(&network.NetworkLink{
		SourceID:      "node-1",
		DestinationID: "node-2",
		Bandwidth:     10000,
		Latency:       1.0,
		Type:          network.LinkTypeSameDatacenter,
	})
	networkTopology.AddLink(&network.NetworkLink{
		SourceID:      "node-2",
		DestinationID: "node-1",
		Bandwidth:     10000,
		Latency:       1.0,
		Type:          network.LinkTypeSameDatacenter,
	})
	networkTopology.AddLink(&network.NetworkLink{
		SourceID:      "node-2",
		DestinationID: "node-3",
		Bandwidth:     10000,
		Latency:       1.0,
		Type:          network.LinkTypeSameDatacenter,
	})
	networkTopology.AddLink(&network.NetworkLink{
		SourceID:      "node-3",
		DestinationID: "node-2",
		Bandwidth:     10000,
		Latency:       1.0,
		Type:          network.LinkTypeSameDatacenter,
	})
	networkTopology.AddLink(&network.NetworkLink{
		SourceID:      "node-1",
		DestinationID: "node-3",
		Bandwidth:     5000,
		Latency:       2.0,
		Type:          network.LinkTypeSameDatacenter,
	})
	networkTopology.AddLink(&network.NetworkLink{
		SourceID:      "node-3",
		DestinationID: "node-1",
		Bandwidth:     5000,
		Latency:       2.0,
		Type:          network.LinkTypeSameDatacenter,
	})

	config := DefaultNetworkAwareSchedulerConfig()
	scheduler := NewNetworkAwareScheduler(
		config,
		baseScheduler,
		workloadAnalyzer,
		networkTopology,
	)

	return scheduler
}
