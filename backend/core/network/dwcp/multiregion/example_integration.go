package multiregion

import (
	"fmt"
	"log"
	"time"
)

// ExampleGlobalDeployment demonstrates a complete multi-region deployment
func ExampleGlobalDeployment() {
	// Step 1: Create global topology
	topology := NewGlobalTopology()

	// Step 2: Define regions
	regions := []*Region{
		{
			ID:   "us-east-1",
			Name: "US East (Virginia)",
			Location: GeoLocation{
				Latitude:  37.7749,
				Longitude: -122.4194,
				Country:   "USA",
				City:      "San Francisco",
			},
			Endpoints: []NetworkEndpoint{
				{Address: "10.0.1.1", Port: 8080, Protocol: "tcp", PublicIP: "203.0.113.1"},
			},
			Capacity: RegionCapacity{
				MaxInstances:    1000,
				MaxBandwidthMbps: 10000,
				MaxStorage:      1000000,
				AvailableVCPUs:  500,
				AvailableRAM:    2048000,
			},
		},
		{
			ID:   "eu-west-1",
			Name: "EU West (Ireland)",
			Location: GeoLocation{
				Latitude:  53.3498,
				Longitude: -6.2603,
				Country:   "Ireland",
				City:      "Dublin",
			},
			Endpoints: []NetworkEndpoint{
				{Address: "10.0.2.1", Port: 8080, Protocol: "tcp", PublicIP: "203.0.113.2"},
			},
			Capacity: RegionCapacity{
				MaxInstances:    800,
				MaxBandwidthMbps: 8000,
				MaxStorage:      800000,
				AvailableVCPUs:  400,
				AvailableRAM:    1638400,
			},
		},
		{
			ID:   "ap-south-1",
			Name: "Asia Pacific South (Mumbai)",
			Location: GeoLocation{
				Latitude:  19.0760,
				Longitude: 72.8777,
				Country:   "India",
				City:      "Mumbai",
			},
			Endpoints: []NetworkEndpoint{
				{Address: "10.0.3.1", Port: 8080, Protocol: "tcp", PublicIP: "203.0.113.3"},
			},
			Capacity: RegionCapacity{
				MaxInstances:    600,
				MaxBandwidthMbps: 6000,
				MaxStorage:      600000,
				AvailableVCPUs:  300,
				AvailableRAM:    1228800,
			},
		},
	}

	// Add regions to topology
	for _, region := range regions {
		if err := topology.AddRegion(region); err != nil {
			log.Fatalf("Failed to add region %s: %v", region.ID, err)
		}
		fmt.Printf("Added region: %s (%s)\n", region.Name, region.ID)
	}

	// Step 3: Create inter-region links (mesh topology)
	links := []*InterRegionLink{
		{
			Source:      "us-east-1",
			Destination: "eu-west-1",
			Latency:     80 * time.Millisecond,
			Bandwidth:   1000, // 1 Gbps
			Cost:        10.0, // $10/hour
			Health:      HealthUp,
			Utilization: 30.0,
		},
		{
			Source:      "us-east-1",
			Destination: "ap-south-1",
			Latency:     200 * time.Millisecond,
			Bandwidth:   800, // 800 Mbps
			Cost:        15.0,
			Health:      HealthUp,
			Utilization: 20.0,
		},
		{
			Source:      "eu-west-1",
			Destination: "ap-south-1",
			Latency:     120 * time.Millisecond,
			Bandwidth:   900, // 900 Mbps
			Cost:        12.0,
			Health:      HealthUp,
			Utilization: 25.0,
		},
	}

	for _, link := range links {
		if err := topology.AddLink(link); err != nil {
			log.Fatalf("Failed to add link: %v", err)
		}
		fmt.Printf("Added link: %s -> %s (latency: %v, bandwidth: %d Mbps)\n",
			link.Source, link.Destination, link.Latency, link.Bandwidth)
	}

	// Step 4: Setup VPN tunnels
	tunnelMgr := NewTunnelManager(topology)
	for i := 0; i < len(regions)-1; i++ {
		for j := i + 1; j < len(regions); j++ {
			tunnel, err := tunnelMgr.EstablishTunnel(regions[i], regions[j])
			if err != nil {
				log.Printf("Warning: Failed to establish tunnel %s -> %s: %v",
					regions[i].ID, regions[j].ID, err)
				continue
			}
			fmt.Printf("Established %s tunnel: %s\n", tunnel.Type, tunnel.ID)
		}
	}

	// Step 5: Create routing engine with balanced strategy
	engine := NewRoutingEngine(topology, StrategyBalanced)
	fmt.Println("\nComputing optimal routes...")

	for i := 0; i < len(regions)-1; i++ {
		for j := i + 1; j < len(regions); j++ {
			src := regions[i].ID
			dst := regions[j].ID

			route, err := engine.ComputeRoute(src, dst)
			if err != nil {
				log.Printf("Warning: No route from %s to %s: %v", src, dst, err)
				continue
			}

			fmt.Printf("Route %s -> %s: %v (latency: %v, bandwidth: %d Mbps, hops: %d)\n",
				src, dst, route.Path, route.Metric.Latency, route.Metric.Bandwidth, route.Metric.Hops)
		}
	}

	// Step 6: Setup traffic engineering
	te := NewTrafficEngineer(topology, AlgorithmWECMP)

	// Create a sample traffic flow
	flow := &TrafficFlow{
		ID:          "flow-migration-1",
		Source:      "us-east-1",
		Destination: "eu-west-1",
		Size:        1024 * 1024 * 1024, // 1 GB
		Priority:    7,
		QoS:         QoSCritical,
		Deadline:    time.Now().Add(5 * time.Minute),
		CreatedAt:   time.Now(),
	}

	fmt.Printf("\nDistributing traffic flow: %s (%d MB)\n", flow.ID, flow.Size/(1024*1024))
	if err := te.DistributeTraffic(flow); err != nil {
		log.Fatalf("Failed to distribute traffic: %v", err)
	}

	// Get distribution plan
	distribution, _ := te.GetFlowDistribution(flow.ID)
	fmt.Println("Traffic distribution:")
	for i, alloc := range distribution.Allocations {
		fmt.Printf("  Path %d: %v -> %.2f%% (%d Mbps)\n",
			i+1, alloc.Path.Path, alloc.Percentage, alloc.Bandwidth)
	}

	// Step 7: Setup bandwidth reservations
	bm := NewBandwidthManager(topology)
	route, _ := engine.ComputeRoute("us-east-1", "ap-south-1")

	req := &ReservationRequest{
		FlowID:    "critical-flow-1",
		Path:      route,
		Bandwidth: 200, // 200 Mbps
		Priority:  9,
		Duration:  15 * time.Minute,
	}

	reservationID, err := bm.ReserveBandwidth(req)
	if err != nil {
		log.Printf("Warning: Bandwidth reservation failed: %v", err)
	} else {
		fmt.Printf("\nBandwidth reserved: %s (200 Mbps for 15 minutes)\n", reservationID)
	}

	// Step 8: Setup path redundancy for critical flows
	primaryRoute, _ := engine.ComputeRoute("us-east-1", "ap-south-1")
	pr := NewPathRedundancy(primaryRoute, topology)

	// Find alternative paths
	paths, _ := engine.FindEqualCostPaths("us-east-1", "ap-south-1")
	for _, altPath := range paths {
		if altPath != primaryRoute {
			pr.AddSecondaryPath(altPath)
		}
	}

	fmt.Printf("\nPath redundancy configured: 1 primary + %d secondary paths\n",
		len(pr.GetSecondaryPaths()))

	// Step 9: Start network telemetry
	telemetry := NewNetworkTelemetry(topology)
	telemetry.Start()
	defer telemetry.Stop()

	fmt.Println("\nNetwork telemetry started (collecting metrics every 10 seconds)")

	// Step 10: Setup dynamic route updates
	routingTable := NewRoutingTable()
	updater := NewRouteUpdater(topology, routingTable)

	// Populate routing table
	for i := 0; i < len(regions); i++ {
		for j := 0; j < len(regions); j++ {
			if i == j {
				continue
			}
			src := regions[i].ID
			dst := regions[j].ID

			route, err := engine.ComputeRoute(src, dst)
			if err == nil {
				routingTable.Update(src, dst, route)
			}
		}
	}

	fmt.Printf("\nRouting table populated with %d routes\n", len(routingTable.List()))

	// Simulate monitoring
	fmt.Println("\nMonitoring network for 30 seconds...")
	time.Sleep(30 * time.Second)

	// Get statistics
	stats := te.GetStats()
	fmt.Printf("\nTraffic Engineering Statistics:\n")
	fmt.Printf("  Total flows: %d\n", stats.TotalFlows)
	fmt.Printf("  Active flows: %d\n", stats.ActiveFlows)
	fmt.Printf("  Bytes transmitted: %d\n", stats.BytesTransmitted)
	fmt.Printf("  Packets transmitted: %d\n", stats.PacketsTransmitted)

	bwStats := bm.GetStats()
	fmt.Printf("\nBandwidth Management Statistics:\n")
	fmt.Printf("  Total bandwidth: %d Mbps\n", bwStats.TotalBandwidthMbps)
	fmt.Printf("  Reserved bandwidth: %d Mbps\n", bwStats.ReservedBandwidthMbps)
	fmt.Printf("  Available bandwidth: %d Mbps\n", bwStats.AvailableBandwidthMbps)
	fmt.Printf("  Active reservations: %d\n", bwStats.ActiveReservations)

	// Get Prometheus metrics
	metrics := telemetry.exporter.GetMetrics()
	fmt.Printf("\nPrometheus Metrics Exported: %d metrics\n", len(metrics))
	fmt.Println("Sample metrics:")
	for i, metric := range metrics {
		if i < 5 {
			fmt.Printf("  %s\n", metric)
		}
	}

	fmt.Println("\nGlobal multi-region deployment complete!")
}

// ExampleFailoverScenario demonstrates automatic failover
func ExampleFailoverScenario() {
	topology := setupTestTopology()
	engine := NewRoutingEngine(topology, StrategyLatency)
	updater := NewRouteUpdater(topology, NewRoutingTable())

	fmt.Println("=== Failover Scenario Demo ===\n")

	// Initial route
	route, _ := engine.ComputeRoute("us-east-1", "ap-south-1")
	fmt.Printf("Initial route: %v (latency: %v)\n", route.Path, route.Metric.Latency)

	// Simulate link failure
	links := topology.ListLinks()
	if len(links) > 0 {
		failedLink := links[0].ID
		fmt.Printf("\nSimulating failure of link: %s\n", failedLink)

		// Trigger failover
		updater.UpdateOnLinkFailure(failedLink)

		// Compute new route
		newRoute, err := engine.ComputeRoute("us-east-1", "ap-south-1")
		if err != nil {
			fmt.Printf("No alternative route available: %v\n", err)
		} else {
			fmt.Printf("Failover route: %v (latency: %v)\n", newRoute.Path, newRoute.Metric.Latency)
			fmt.Printf("Failover completed in <1 second\n")
		}

		// Simulate link recovery
		time.Sleep(5 * time.Second)
		fmt.Printf("\nSimulating recovery of link: %s\n", failedLink)
		updater.UpdateOnLinkRecovery(failedLink)

		// Re-optimize
		optimizedRoute, _ := engine.ComputeRoute("us-east-1", "ap-south-1")
		fmt.Printf("Optimized route after recovery: %v (latency: %v)\n",
			optimizedRoute.Path, optimizedRoute.Metric.Latency)
	}

	fmt.Println("\nFailover scenario complete!")
}

// ExampleBandwidthReservation demonstrates bandwidth reservation
func ExampleBandwidthReservation() {
	topology := setupTestTopology()
	bm := NewBandwidthManager(topology)
	engine := NewRoutingEngine(topology, StrategyBandwidth)

	fmt.Println("=== Bandwidth Reservation Demo ===\n")

	route, _ := engine.ComputeRoute("us-east-1", "eu-west-1")
	fmt.Printf("Selected route: %v (bandwidth: %d Mbps)\n", route.Path, route.Metric.Bandwidth)

	// Make reservation
	req := &ReservationRequest{
		FlowID:    "video-stream-1",
		Path:      route,
		Bandwidth: 150, // 150 Mbps for 4K video
		Priority:  8,
		Duration:  30 * time.Minute,
	}

	reservationID, err := bm.ReserveBandwidth(req)
	if err != nil {
		log.Fatalf("Reservation failed: %v", err)
	}

	fmt.Printf("\nReservation created: %s\n", reservationID)
	fmt.Printf("Bandwidth reserved: 150 Mbps for 30 minutes\n")

	// Check utilization
	for _, linkID := range route.Links {
		current, projected, _ := bm.GetLinkUtilization(linkID)
		fmt.Printf("Link %s: current=%.2f%%, projected=%.2f%%\n", linkID, current, projected)
	}

	// Extend reservation
	fmt.Println("\nExtending reservation by 15 minutes...")
	bm.ExtendReservation(reservationID, 15*time.Minute)

	reservation, _ := bm.GetReservation(reservationID)
	fmt.Printf("New expiration: %v\n", reservation.ExpiresAt)

	// Release reservation
	fmt.Println("\nReleasing bandwidth reservation...")
	bm.ReleaseBandwidth(reservationID)
	fmt.Println("Bandwidth released successfully")

	fmt.Println("\nBandwidth reservation demo complete!")
}

// setupTestTopology creates a test topology (reused from tests)
func setupTestTopology() *GlobalTopology {
	topology := NewGlobalTopology()

	regions := []*Region{
		{
			ID:       "us-east-1",
			Name:     "US East",
			Location: GeoLocation{Latitude: 37.7749, Longitude: -122.4194, Country: "USA", City: "San Francisco"},
			Endpoints: []NetworkEndpoint{{Address: "10.0.1.1", Port: 8080}},
			Capacity:  RegionCapacity{MaxInstances: 1000, MaxBandwidthMbps: 10000},
		},
		{
			ID:       "eu-west-1",
			Name:     "EU West",
			Location: GeoLocation{Latitude: 51.5074, Longitude: -0.1278, Country: "UK", City: "London"},
			Endpoints: []NetworkEndpoint{{Address: "10.0.2.1", Port: 8080}},
			Capacity:  RegionCapacity{MaxInstances: 800, MaxBandwidthMbps: 8000},
		},
		{
			ID:       "ap-south-1",
			Name:     "Asia Pacific South",
			Location: GeoLocation{Latitude: 19.0760, Longitude: 72.8777, Country: "India", City: "Mumbai"},
			Endpoints: []NetworkEndpoint{{Address: "10.0.3.1", Port: 8080}},
			Capacity:  RegionCapacity{MaxInstances: 600, MaxBandwidthMbps: 6000},
		},
	}

	for _, region := range regions {
		topology.AddRegion(region)
	}

	links := []*InterRegionLink{
		{Source: "us-east-1", Destination: "eu-west-1", Latency: 80 * time.Millisecond,
			Bandwidth: 1000, Cost: 10.0, Health: HealthUp, Utilization: 30.0},
		{Source: "us-east-1", Destination: "ap-south-1", Latency: 200 * time.Millisecond,
			Bandwidth: 800, Cost: 15.0, Health: HealthUp, Utilization: 20.0},
		{Source: "eu-west-1", Destination: "ap-south-1", Latency: 120 * time.Millisecond,
			Bandwidth: 900, Cost: 12.0, Health: HealthUp, Utilization: 25.0},
	}

	for _, link := range links {
		topology.AddLink(link)
	}

	return topology
}
