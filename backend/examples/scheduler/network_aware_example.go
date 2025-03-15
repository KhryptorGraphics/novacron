package multi_tenant

import (
	"fmt"
	"log"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/scheduler"
	"github.com/khryptorgraphics/novacron/backend/core/scheduler/network"
)

func RunNetworkAwareExample() {
	// Create scheduler factory with network awareness
	factoryConfig := scheduler.DefaultSchedulerFactoryConfig()
	factoryConfig.SchedulerType = scheduler.SchedulerTypeNetworkAware
	factory := scheduler.NewSchedulerFactory(factoryConfig)

	// Create the scheduler
	schedulerInstance, err := factory.CreateScheduler()
	if err != nil {
		log.Fatalf("Failed to create scheduler: %v", err)
	}

	// Cast to network-aware scheduler to access specific methods
	networkScheduler, ok := schedulerInstance.(*scheduler.NetworkAwareScheduler)
	if !ok {
		log.Fatalf("Failed to cast to NetworkAwareScheduler")
	}

	// Start the scheduler
	if err := networkScheduler.Start(); err != nil {
		log.Fatalf("Failed to start scheduler: %v", err)
	}
	defer networkScheduler.Stop()

	// Get network topology for configuration
	networkTopology := factory.GetNetworkTopology()

	// Define network nodes with locations
	setupNetworkTopology(networkTopology)

	// Register nodes with the scheduler
	setupNodes(networkScheduler)

	// Track VM locations and communications
	setupVMs(networkScheduler)

	// Request VM placement with network awareness
	requestID, err := networkScheduler.RequestPlacement(
		"vm-001",
		scheduler.PolicyNetworkAware,
		[]scheduler.PlacementConstraint{},
		map[string]float64{
			"cpu":     2.0,
			"memory":  4096.0,
			"disk":    10240.0,
			"network": 1000.0,
		},
		1, // Priority
	)

	if err != nil {
		log.Fatalf("Failed to request placement: %v", err)
	}

	// Wait for placement result
	time.Sleep(1 * time.Second)
	result, err := networkScheduler.GetPlacementResult(requestID)
	if err != nil {
		log.Fatalf("Failed to get placement result: %v", err)
	}

	// Display placement result
	fmt.Printf("VM placed on node: %s\n", result.SelectedNode)
	fmt.Printf("Placement score: %.2f\n", result.Score)
	fmt.Printf("Reasoning: %s\n", result.Reasoning)
	fmt.Printf("Alternative nodes: %v\n", result.AlternativeNodes)

	// Request placement for a communicating VM
	requestID2, err := networkScheduler.RequestPlacement(
		"vm-002",
		scheduler.PolicyNetworkAware,
		[]scheduler.PlacementConstraint{},
		map[string]float64{
			"cpu":     2.0,
			"memory":  4096.0,
			"disk":    10240.0,
			"network": 1000.0,
		},
		1, // Priority
	)

	if err != nil {
		log.Fatalf("Failed to request placement: %v", err)
	}

	// Wait for placement result
	time.Sleep(1 * time.Second)
	result2, err := networkScheduler.GetPlacementResult(requestID2)
	if err != nil {
		log.Fatalf("Failed to get placement result: %v", err)
	}

	// Display placement result
	fmt.Printf("\nCommunicating VM placed on node: %s\n", result2.SelectedNode)
	fmt.Printf("Placement score: %.2f\n", result2.Score)
	fmt.Printf("Reasoning: %s\n", result2.Reasoning)

	// Show migration planning example
	fmt.Println("\n=== Migration Planning Example ===")
	exampleMigrationPlanning(networkScheduler, networkTopology)
}

// setupNetworkTopology defines nodes and links in the network topology
func setupNetworkTopology(topology *network.NetworkTopology) {
	// Define datacenter locations
	dc1 := network.NetworkLocation{
		Datacenter: "dc-east",
		Zone:       "zone-1",
		Rack:       "rack-1",
	}

	dc2 := network.NetworkLocation{
		Datacenter: "dc-west",
		Zone:       "zone-1",
		Rack:       "rack-1",
	}

	// Add nodes to topology
	nodes := []struct {
		id       string
		nodeType string
		location network.NetworkLocation
	}{
		{"node-1", "hypervisor", dc1},
		{"node-2", "hypervisor", dc1},
		{"node-3", "hypervisor", dc1},
		{"node-4", "hypervisor", dc2},
		{"node-5", "hypervisor", dc2},
	}

	for _, n := range nodes {
		topology.AddNode(&network.NetworkNode{
			ID:       n.id,
			Type:     n.nodeType,
			Location: n.location,
			Attributes: map[string]interface{}{
				"cores":   24,
				"memory":  128,
				"storage": 2048,
			},
		})
	}

	// Add network links
	links := []struct {
		source    string
		dest      string
		bandwidth float64
		latency   float64
		linkType  network.LinkType
	}{
		{"node-1", "node-2", 10000, 0.5, network.LinkTypeSameDatacenter}, // 10 Gbps, 0.5ms
		{"node-1", "node-3", 10000, 0.5, network.LinkTypeSameDatacenter}, // 10 Gbps, 0.5ms
		{"node-2", "node-3", 10000, 0.5, network.LinkTypeSameDatacenter}, // 10 Gbps, 0.5ms
		{"node-4", "node-5", 10000, 0.5, network.LinkTypeSameDatacenter}, // 10 Gbps, 0.5ms
		{"node-1", "node-4", 1000, 50, network.LinkTypeInterDatacenter},  // 1 Gbps, 50ms
		{"node-2", "node-4", 1000, 50, network.LinkTypeInterDatacenter},  // 1 Gbps, 50ms
		{"node-3", "node-5", 1000, 50, network.LinkTypeInterDatacenter},  // 1 Gbps, 50ms
	}

	for _, l := range links {
		topology.AddLink(&network.NetworkLink{
			SourceID:      l.source,
			DestinationID: l.dest,
			Type:          l.linkType,
			Bandwidth:     l.bandwidth,
			Latency:       l.latency,
			Loss:          0.0,
			Jitter:        0.0,
			Cost:          0.0, // Will be calculated by topology
			Utilization:   0.2, // 20% baseline utilization
		})

		// Add reverse link (for bidirectional)
		topology.AddLink(&network.NetworkLink{
			SourceID:      l.dest,
			DestinationID: l.source,
			Type:          l.linkType,
			Bandwidth:     l.bandwidth,
			Latency:       l.latency,
			Loss:          0.0,
			Jitter:        0.0,
			Cost:          0.0, // Will be calculated by topology
			Utilization:   0.2, // 20% baseline utilization
		})
	}

	log.Printf("Network topology created with %d nodes and %d links",
		len(nodes), len(links)*2)
}

// setupNodes registers nodes with the scheduler
func setupNodes(s *scheduler.NetworkAwareScheduler) {
	nodes := []struct {
		id        string
		resources map[scheduler.ResourceType]*scheduler.Resource
	}{
		{
			"node-1",
			map[scheduler.ResourceType]*scheduler.Resource{
				scheduler.ResourceCPU:     {Capacity: 48.0, Used: 10.0},
				scheduler.ResourceMemory:  {Capacity: 128 * 1024, Used: 32 * 1024},
				scheduler.ResourceDisk:    {Capacity: 2048 * 1024, Used: 512 * 1024},
				scheduler.ResourceNetwork: {Capacity: 10000, Used: 2000},
			},
		},
		{
			"node-2",
			map[scheduler.ResourceType]*scheduler.Resource{
				scheduler.ResourceCPU:     {Capacity: 48.0, Used: 24.0},
				scheduler.ResourceMemory:  {Capacity: 128 * 1024, Used: 64 * 1024},
				scheduler.ResourceDisk:    {Capacity: 2048 * 1024, Used: 1024 * 1024},
				scheduler.ResourceNetwork: {Capacity: 10000, Used: 5000},
			},
		},
		{
			"node-3",
			map[scheduler.ResourceType]*scheduler.Resource{
				scheduler.ResourceCPU:     {Capacity: 48.0, Used: 12.0},
				scheduler.ResourceMemory:  {Capacity: 128 * 1024, Used: 40 * 1024},
				scheduler.ResourceDisk:    {Capacity: 2048 * 1024, Used: 768 * 1024},
				scheduler.ResourceNetwork: {Capacity: 10000, Used: 3000},
			},
		},
		{
			"node-4",
			map[scheduler.ResourceType]*scheduler.Resource{
				scheduler.ResourceCPU:     {Capacity: 64.0, Used: 16.0},
				scheduler.ResourceMemory:  {Capacity: 256 * 1024, Used: 64 * 1024},
				scheduler.ResourceDisk:    {Capacity: 4096 * 1024, Used: 1024 * 1024},
				scheduler.ResourceNetwork: {Capacity: 10000, Used: 2500},
			},
		},
		{
			"node-5",
			map[scheduler.ResourceType]*scheduler.Resource{
				scheduler.ResourceCPU:     {Capacity: 64.0, Used: 32.0},
				scheduler.ResourceMemory:  {Capacity: 256 * 1024, Used: 128 * 1024},
				scheduler.ResourceDisk:    {Capacity: 4096 * 1024, Used: 2048 * 1024},
				scheduler.ResourceNetwork: {Capacity: 10000, Used: 5000},
			},
		},
	}

	for _, node := range nodes {
		if err := s.UpdateNodeResources(node.id, node.resources); err != nil {
			log.Fatalf("Failed to update node resources for %s: %v", node.id, err)
		}
	}

	log.Printf("Registered %d nodes with the scheduler", len(nodes))
}

// setupVMs sets up VMs with locations and communication patterns
func setupVMs(s *scheduler.NetworkAwareScheduler) {
	// Register existing VM locations
	vms := map[string]string{
		"vm-001": "node-1",
		"vm-003": "node-2",
		"vm-004": "node-3",
		"vm-005": "node-4",
		"vm-006": "node-5",
	}

	for vmID, nodeID := range vms {
		s.UpdateVMLocation(vmID, nodeID)
	}

	// Record communication patterns
	communications := []struct {
		sourceVM   string
		destVM     string
		bandwidth  float64 // In Mbps
		packetRate float64 // Packets per second
	}{
		{"vm-001", "vm-003", 200, 1000},
		{"vm-001", "vm-002", 500, 2000}, // vm-002 not placed yet
		{"vm-003", "vm-004", 100, 500},
		{"vm-004", "vm-005", 50, 200},
		{"vm-005", "vm-006", 300, 1500},
	}

	for _, comm := range communications {
		s.TrackVMCommunication(comm.sourceVM, comm.destVM, comm.bandwidth, comm.packetRate)
	}

	// Set latency requirements
	latencyRequirements := []struct {
		sourceVM string
		destVM   string
		latency  float64 // In milliseconds
	}{
		{"vm-001", "vm-002", 5}, // vm-002 must be placed with low latency to vm-001
		{"vm-003", "vm-004", 10},
	}

	for _, req := range latencyRequirements {
		s.SetVMLatencyRequirement(req.sourceVM, req.destVM, req.latency)
	}

	// Create affinity groups
	s.CreateVMAffinityGroup("db-cluster", []string{"vm-001", "vm-002", "vm-003"})
	s.CreateVMAffinityGroup("web-tier", []string{"vm-004", "vm-005", "vm-006"})

	log.Printf("Set up %d VMs, %d communication patterns, %d latency requirements, and 2 affinity groups",
		len(vms), len(communications), len(latencyRequirements))
}

// exampleMigrationPlanning demonstrates migration planning
func exampleMigrationPlanning(s *scheduler.NetworkAwareScheduler, topology *network.NetworkTopology) {
	// Let's say we want to migrate vm-004 to another node
	vmID := "vm-004"
	currentNodeID := "node-3"

	// Find all available nodes except the current one
	allNodes := topology.GetAllNodes()
	candidateNodes := make([]string, 0)
	for _, node := range allNodes {
		if node.ID != currentNodeID {
			candidateNodes = append(candidateNodes, node.ID)
		}
	}

	fmt.Printf("Planning migration for VM %s from node %s\n", vmID, currentNodeID)
	fmt.Printf("Candidate destination nodes: %v\n", candidateNodes)

	// Request placements for each candidate node
	results := make([]*scheduler.PlacementResult, 0)

	for _, nodeID := range candidateNodes {
		// This is a simplification - in real code you'd use the migration cost estimator directly
		requestID, err := s.RequestPlacement(
			vmID,
			scheduler.PolicyNetworkAware,
			[]scheduler.PlacementConstraint{},
			map[string]float64{
				"cpu":     2.0,
				"memory":  4096.0,
				"disk":    10240.0,
				"network": 1000.0,
			},
			1, // Priority
		)

		if err != nil {
			log.Printf("Failed to request placement to %s: %v", nodeID, err)
			continue
		}

		// Wait for the result
		time.Sleep(100 * time.Millisecond)
		result, err := s.GetPlacementResult(requestID)
		if err != nil {
			log.Printf("Failed to get placement result for %s: %v", nodeID, err)
			continue
		}

		results = append(results, result)
	}

	// Sort by score (highest first)
	for i := 0; i < len(results); i++ {
		for j := i + 1; j < len(results); j++ {
			if results[j].Score > results[i].Score {
				results[i], results[j] = results[j], results[i]
			}
		}
	}

	// Display the top 3 options
	fmt.Println("\nTop migration options:")
	count := min(3, len(results))
	for i := 0; i < count; i++ {
		fmt.Printf("%d. Node %s - Score: %.2f - %s\n",
			i+1, results[i].SelectedNode, results[i].Score, results[i].Reasoning)
	}

	// The system would typically execute the migration for the top choice
	if len(results) > 0 {
		bestNode := results[0].SelectedNode
		fmt.Printf("\nRecommended migration action: Move VM %s from %s to %s\n",
			vmID, currentNodeID, bestNode)
	} else {
		fmt.Println("\nNo viable migration targets found")
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
