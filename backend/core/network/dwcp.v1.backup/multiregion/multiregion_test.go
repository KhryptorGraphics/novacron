package multiregion

import (
	"testing"
	"time"
)

// TestGlobalTopology tests global topology management
func TestGlobalTopology(t *testing.T) {
	topology := NewGlobalTopology()

	// Test adding regions
	usEast := &Region{
		ID:   "us-east-1",
		Name: "US East",
		Location: GeoLocation{
			Latitude:  37.7749,
			Longitude: -122.4194,
			Country:   "USA",
			City:      "San Francisco",
		},
		Endpoints: []NetworkEndpoint{
			{Address: "10.0.1.1", Port: 8080, Protocol: "tcp"},
		},
		Capacity: RegionCapacity{
			MaxInstances:    1000,
			MaxBandwidthMbps: 10000,
			MaxStorage:      1000000,
		},
	}

	euWest := &Region{
		ID:   "eu-west-1",
		Name: "EU West",
		Location: GeoLocation{
			Latitude:  51.5074,
			Longitude: -0.1278,
			Country:   "UK",
			City:      "London",
		},
		Endpoints: []NetworkEndpoint{
			{Address: "10.0.2.1", Port: 8080, Protocol: "tcp"},
		},
		Capacity: RegionCapacity{
			MaxInstances:    800,
			MaxBandwidthMbps: 8000,
			MaxStorage:      800000,
		},
	}

	if err := topology.AddRegion(usEast); err != nil {
		t.Fatalf("Failed to add US East region: %v", err)
	}

	if err := topology.AddRegion(euWest); err != nil {
		t.Fatalf("Failed to add EU West region: %v", err)
	}

	// Test duplicate region
	if err := topology.AddRegion(usEast); err == nil {
		t.Error("Expected error when adding duplicate region")
	}

	// Test adding link
	link := &InterRegionLink{
		Source:      "us-east-1",
		Destination: "eu-west-1",
		Latency:     80 * time.Millisecond,
		Bandwidth:   1000, // 1 Gbps
		Cost:        10.0,
		Health:      HealthUp,
	}

	if err := topology.AddLink(link); err != nil {
		t.Fatalf("Failed to add link: %v", err)
	}

	// Test retrieving regions
	regions := topology.ListRegions()
	if len(regions) != 2 {
		t.Errorf("Expected 2 regions, got %d", len(regions))
	}

	// Test retrieving links
	links := topology.ListLinks()
	if len(links) != 1 {
		t.Errorf("Expected 1 link, got %d", len(links))
	}

	// Test link health update
	linkID := links[0].ID
	if err := topology.UpdateLinkHealth(linkID, HealthDegraded); err != nil {
		t.Fatalf("Failed to update link health: %v", err)
	}

	updatedLink, _ := topology.GetLink(linkID)
	if updatedLink.Health != HealthDegraded {
		t.Error("Link health not updated")
	}
}

// TestRoutingEngine tests routing algorithms
func TestRoutingEngine(t *testing.T) {
	topology := setupTestTopology()
	engine := NewRoutingEngine(topology, StrategyLatency)

	// Test Dijkstra (latency optimization)
	route, err := engine.ComputeRoute("us-east-1", "ap-south-1")
	if err != nil {
		t.Fatalf("Failed to compute route: %v", err)
	}

	if route == nil {
		t.Fatal("Route is nil")
	}

	if route.Destination != "ap-south-1" {
		t.Errorf("Expected destination ap-south-1, got %s", route.Destination)
	}

	t.Logf("Route: %v (latency: %v, hops: %d)", route.Path, route.Metric.Latency, route.Metric.Hops)

	// Test widest path (bandwidth optimization)
	engineBW := NewRoutingEngine(topology, StrategyBandwidth)
	routeBW, err := engineBW.ComputeRoute("us-east-1", "ap-south-1")
	if err != nil {
		t.Fatalf("Failed to compute bandwidth-optimized route: %v", err)
	}

	t.Logf("Bandwidth route: %v (bandwidth: %d Mbps)", routeBW.Path, routeBW.Metric.Bandwidth)

	// Test equal-cost paths for ECMP
	paths, err := engine.FindEqualCostPaths("us-east-1", "eu-west-1")
	if err != nil {
		t.Fatalf("Failed to find equal-cost paths: %v", err)
	}

	t.Logf("Found %d equal-cost paths", len(paths))
}

// TestTunnelManager tests VPN tunnel management
func TestTunnelManager(t *testing.T) {
	topology := setupTestTopology()
	tunnelMgr := NewTunnelManager(topology)

	usEast, _ := topology.GetRegion("us-east-1")
	euWest, _ := topology.GetRegion("eu-west-1")

	// Establish tunnel
	tunnel, err := tunnelMgr.EstablishTunnel(usEast, euWest)
	if err != nil {
		t.Fatalf("Failed to establish tunnel: %v", err)
	}

	if tunnel.Status != TunnelStatusUp {
		t.Errorf("Expected tunnel status UP, got %s", tunnel.Status)
	}

	if tunnel.Type != TunnelWireGuard {
		t.Errorf("Expected WireGuard tunnel, got %s", tunnel.Type)
	}

	// Check encryption config
	if tunnel.Encryption.Algorithm != "ChaCha20-Poly1305" {
		t.Errorf("Expected ChaCha20-Poly1305, got %s", tunnel.Encryption.Algorithm)
	}

	// List tunnels
	tunnels := tunnelMgr.ListTunnels()
	if len(tunnels) != 1 {
		t.Errorf("Expected 1 tunnel, got %d", len(tunnels))
	}

	// Teardown tunnel
	if err := tunnelMgr.TeardownTunnel(tunnel.ID); err != nil {
		t.Fatalf("Failed to teardown tunnel: %v", err)
	}

	tunnels = tunnelMgr.ListTunnels()
	if len(tunnels) != 0 {
		t.Errorf("Expected 0 tunnels after teardown, got %d", len(tunnels))
	}
}

// TestTrafficEngineer tests traffic distribution
func TestTrafficEngineer(t *testing.T) {
	topology := setupTestTopology()
	te := NewTrafficEngineer(topology, AlgorithmWECMP)

	flow := &TrafficFlow{
		ID:          "flow-1",
		Source:      "us-east-1",
		Destination: "eu-west-1",
		Size:        1024 * 1024 * 100, // 100 MB
		Priority:    5,
		QoS:         QoSInteractive,
		CreatedAt:   time.Now(),
	}

	// Distribute traffic
	if err := te.DistributeTraffic(flow); err != nil {
		t.Fatalf("Failed to distribute traffic: %v", err)
	}

	// Get flow distribution
	distribution, err := te.GetFlowDistribution(flow.ID)
	if err != nil {
		t.Fatalf("Failed to get flow distribution: %v", err)
	}

	t.Logf("Flow distributed across %d paths", len(distribution.Allocations))
	for i, alloc := range distribution.Allocations {
		t.Logf("  Path %d: %.2f%% (%d Mbps)", i+1, alloc.Percentage, alloc.Bandwidth)
	}

	// Get stats
	stats := te.GetStats()
	if stats.TotalFlows != 1 {
		t.Errorf("Expected 1 total flow, got %d", stats.TotalFlows)
	}

	if stats.ActiveFlows != 1 {
		t.Errorf("Expected 1 active flow, got %d", stats.ActiveFlows)
	}

	// Complete flow
	if err := te.CompleteFlow(flow.ID); err != nil {
		t.Fatalf("Failed to complete flow: %v", err)
	}

	stats = te.GetStats()
	if stats.ActiveFlows != 0 {
		t.Errorf("Expected 0 active flows after completion, got %d", stats.ActiveFlows)
	}
}

// TestPathRedundancy tests failover mechanisms
func TestPathRedundancy(t *testing.T) {
	topology := setupTestTopology()
	engine := NewRoutingEngine(topology, StrategyLatency)

	// Get primary path
	primary, err := engine.ComputeRoute("us-east-1", "ap-south-1")
	if err != nil {
		t.Fatalf("Failed to compute primary route: %v", err)
	}

	pr := NewPathRedundancy(primary, topology)

	// Add secondary paths
	// Simulate by marking primary link as down temporarily
	if len(primary.Links) > 0 {
		firstLink, _ := topology.GetLink(primary.Links[0])
		originalHealth := firstLink.Health
		firstLink.Health = HealthDown

		secondary, err := engine.ComputeRoute("us-east-1", "ap-south-1")
		if err == nil {
			pr.AddSecondaryPath(secondary)
		}

		firstLink.Health = originalHealth
	}

	// Test send with failover
	testData := []byte("test data")
	if err := pr.SendWithFailover(testData); err != nil {
		t.Fatalf("Send with failover failed: %v", err)
	}

	// Test path monitoring
	health := pr.monitor.GetPathHealth(primary)
	t.Logf("Primary path health: healthy=%v, latency=%v", health.IsHealthy, health.Latency)
}

// TestBandwidthManager tests bandwidth reservation
func TestBandwidthManager(t *testing.T) {
	topology := setupTestTopology()
	bm := NewBandwidthManager(topology)
	engine := NewRoutingEngine(topology, StrategyBandwidth)

	route, err := engine.ComputeRoute("us-east-1", "eu-west-1")
	if err != nil {
		t.Fatalf("Failed to compute route: %v", err)
	}

	// Make reservation request
	req := &ReservationRequest{
		FlowID:    "flow-1",
		Path:      route,
		Bandwidth: 100, // 100 Mbps
		Priority:  5,
		Duration:  10 * time.Minute,
	}

	reservationID, err := bm.ReserveBandwidth(req)
	if err != nil {
		t.Fatalf("Failed to reserve bandwidth: %v", err)
	}

	t.Logf("Created reservation: %s", reservationID)

	// Check reservation
	reservation, err := bm.GetReservation(reservationID)
	if err != nil {
		t.Fatalf("Failed to get reservation: %v", err)
	}

	if reservation.Bandwidth != 100 {
		t.Errorf("Expected 100 Mbps reserved, got %d", reservation.Bandwidth)
	}

	// Get stats
	stats := bm.GetStats()
	t.Logf("Bandwidth stats: total=%d, reserved=%d, available=%d",
		stats.TotalBandwidthMbps, stats.ReservedBandwidthMbps, stats.AvailableBandwidthMbps)

	if stats.ReservedBandwidthMbps < 100 {
		t.Errorf("Expected at least 100 Mbps reserved, got %d", stats.ReservedBandwidthMbps)
	}

	// Test extending reservation
	if err := bm.ExtendReservation(reservationID, 5*time.Minute); err != nil {
		t.Fatalf("Failed to extend reservation: %v", err)
	}

	// Release bandwidth
	if err := bm.ReleaseBandwidth(reservationID); err != nil {
		t.Fatalf("Failed to release bandwidth: %v", err)
	}

	// Verify release
	_, err = bm.GetReservation(reservationID)
	if err != ErrReservationNotFound {
		t.Error("Expected reservation not found after release")
	}
}

// TestNetworkTelemetry tests metrics collection
func TestNetworkTelemetry(t *testing.T) {
	topology := setupTestTopology()
	telemetry := NewNetworkTelemetry(topology)

	// Collect metrics manually (normally runs in background)
	telemetry.collectLinkMetrics()
	telemetry.collectRegionMetrics()

	// Check collected metrics
	metrics := telemetry.collector.GetAllMetrics()
	if len(metrics) == 0 {
		t.Error("Expected metrics to be collected")
	}

	t.Logf("Collected %d metrics", len(metrics))

	// Check Prometheus export
	promMetrics := telemetry.exporter.GetMetrics()
	if len(promMetrics) == 0 {
		t.Error("Expected Prometheus metrics")
	}

	t.Logf("Exported %d Prometheus metrics", len(promMetrics))
	for i, metric := range promMetrics {
		if i < 5 { // Log first 5 metrics
			t.Logf("  %s", metric)
		}
	}
}

// TestRouteUpdater tests dynamic route updates
func TestRouteUpdater(t *testing.T) {
	topology := setupTestTopology()
	routingTable := NewRoutingTable()
	engine := NewRoutingEngine(topology, StrategyLatency)

	// Populate initial routes
	regions := topology.ListRegions()
	for i := 0; i < len(regions)-1; i++ {
		for j := i + 1; j < len(regions); j++ {
			src := regions[i].ID
			dst := regions[j].ID

			route, err := engine.ComputeRoute(src, dst)
			if err == nil {
				routingTable.Update(src, dst, route)
			}
		}
	}

	updater := NewRouteUpdater(topology, routingTable)

	// Subscribe to updates (use non-logging subscriber to avoid race)
	subscriber := &testSubscriber{t: t, silent: true}
	updater.Subscribe(subscriber)

	initialRoutes := len(routingTable.List())
	t.Logf("Initial routing table has %d routes", initialRoutes)

	// Simulate link failure
	links := topology.ListLinks()
	if len(links) > 0 {
		failedLink := links[0].ID
		t.Logf("Simulating failure of link: %s", failedLink)

		if err := updater.UpdateOnLinkFailure(failedLink); err != nil {
			t.Fatalf("Failed to update on link failure: %v", err)
		}

		// Verify link is marked as down
		link, _ := topology.GetLink(failedLink)
		if link.Health != HealthDown {
			t.Error("Link should be marked as down")
		}
	}

	// Test link recovery
	if len(links) > 0 {
		recoveredLink := links[0].ID
		t.Logf("Simulating recovery of link: %s", recoveredLink)

		if err := updater.UpdateOnLinkRecovery(recoveredLink); err != nil {
			t.Fatalf("Failed to update on link recovery: %v", err)
		}

		link, _ := topology.GetLink(recoveredLink)
		if link.Health != HealthUp {
			t.Error("Link should be marked as up")
		}
	}
}

// testSubscriber implements RouteUpdateSubscriber for testing
type testSubscriber struct {
	t      *testing.T
	silent bool
}

func (ts *testSubscriber) OnRouteUpdate(update *RouteUpdate) {
	if !ts.silent {
		ts.t.Logf("Route update: type=%s, destination=%s, reason=%s",
			update.Type, update.Destination, update.Reason)
	}
}

// TestPerformanceMetrics tests routing performance
func TestPerformanceMetrics(t *testing.T) {
	topology := setupTestTopology()
	engine := NewRoutingEngine(topology, StrategyLatency)

	// Measure route computation time
	start := time.Now()
	iterations := 100

	for i := 0; i < iterations; i++ {
		_, err := engine.ComputeRoute("us-east-1", "ap-south-1")
		if err != nil {
			t.Fatalf("Route computation failed: %v", err)
		}
	}

	elapsed := time.Since(start)
	avgTime := elapsed / time.Duration(iterations)

	t.Logf("Average route computation time: %v", avgTime)

	// Verify performance requirement (< 10ms)
	if avgTime > 10*time.Millisecond {
		t.Errorf("Route computation too slow: %v (expected < 10ms)", avgTime)
	}
}

// setupTestTopology creates a test topology with multiple regions
func setupTestTopology() *GlobalTopology {
	topology := NewGlobalTopology()

	// Create regions
	regions := []*Region{
		{
			ID:   "us-east-1",
			Name: "US East",
			Location: GeoLocation{
				Latitude: 37.7749, Longitude: -122.4194,
				Country: "USA", City: "San Francisco",
			},
			Endpoints: []NetworkEndpoint{{Address: "10.0.1.1", Port: 8080}},
			Capacity:  RegionCapacity{MaxInstances: 1000, MaxBandwidthMbps: 10000},
		},
		{
			ID:   "eu-west-1",
			Name: "EU West",
			Location: GeoLocation{
				Latitude: 51.5074, Longitude: -0.1278,
				Country: "UK", City: "London",
			},
			Endpoints: []NetworkEndpoint{{Address: "10.0.2.1", Port: 8080}},
			Capacity:  RegionCapacity{MaxInstances: 800, MaxBandwidthMbps: 8000},
		},
		{
			ID:   "ap-south-1",
			Name: "Asia Pacific South",
			Location: GeoLocation{
				Latitude: 19.0760, Longitude: 72.8777,
				Country: "India", City: "Mumbai",
			},
			Endpoints: []NetworkEndpoint{{Address: "10.0.3.1", Port: 8080}},
			Capacity:  RegionCapacity{MaxInstances: 600, MaxBandwidthMbps: 6000},
		},
	}

	for _, region := range regions {
		topology.AddRegion(region)
	}

	// Create links (mesh topology)
	links := []*InterRegionLink{
		{
			Source: "us-east-1", Destination: "eu-west-1",
			Latency: 80 * time.Millisecond, Bandwidth: 1000,
			Cost: 10.0, Health: HealthUp, Utilization: 30.0,
		},
		{
			Source: "us-east-1", Destination: "ap-south-1",
			Latency: 200 * time.Millisecond, Bandwidth: 800,
			Cost: 15.0, Health: HealthUp, Utilization: 20.0,
		},
		{
			Source: "eu-west-1", Destination: "ap-south-1",
			Latency: 120 * time.Millisecond, Bandwidth: 900,
			Cost: 12.0, Health: HealthUp, Utilization: 25.0,
		},
	}

	for _, link := range links {
		topology.AddLink(link)
	}

	return topology
}

// Benchmark tests
func BenchmarkRouteComputation(b *testing.B) {
	topology := setupTestTopology()
	engine := NewRoutingEngine(topology, StrategyLatency)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		engine.ComputeRoute("us-east-1", "ap-south-1")
	}
}

func BenchmarkBandwidthReservation(b *testing.B) {
	topology := setupTestTopology()
	bm := NewBandwidthManager(topology)
	engine := NewRoutingEngine(topology, StrategyBandwidth)

	route, _ := engine.ComputeRoute("us-east-1", "eu-west-1")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		req := &ReservationRequest{
			FlowID:    "flow-bench",
			Path:      route,
			Bandwidth: 50,
			Priority:  5,
			Duration:  1 * time.Minute,
		}
		reservationID, _ := bm.ReserveBandwidth(req)
		bm.ReleaseBandwidth(reservationID)
	}
}
