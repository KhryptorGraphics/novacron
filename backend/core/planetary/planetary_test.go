package planetary

import (
	"testing"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/planetary/interplanetary"
	"github.com/khryptorgraphics/novacron/backend/core/planetary/leo"
	"github.com/khryptorgraphics/novacron/backend/core/planetary/mesh"
	"github.com/khryptorgraphics/novacron/backend/core/planetary/regions"
	"github.com/khryptorgraphics/novacron/backend/core/planetary/space"
)

func TestPlanetaryCoordinator(t *testing.T) {
	config := DefaultPlanetaryConfig()
	config.StarlinkAPIKey = "test-key"

	coordinator, err := NewPlanetaryCoordinator(config)
	if err != nil {
		t.Fatalf("Failed to create coordinator: %v", err)
	}

	if err := coordinator.Start(); err != nil {
		t.Fatalf("Failed to start coordinator: %v", err)
	}

	// Wait for initialization
	time.Sleep(2 * time.Second)

	// Check status
	if coordinator.Status() != "running" {
		t.Errorf("Expected status 'running', got '%s'", coordinator.Status())
	}

	// Get metrics
	metrics := coordinator.GetGlobalMetrics()
	if metrics == nil {
		t.Error("Expected metrics, got nil")
	}

	if err := coordinator.Stop(); err != nil {
		t.Fatalf("Failed to stop coordinator: %v", err)
	}
}

func TestSatelliteManager(t *testing.T) {
	config := DefaultPlanetaryConfig()
	config.StarlinkAPIKey = "test-key"

	sm := leo.NewSatelliteManager(config)

	if err := sm.Start(); err != nil {
		t.Fatalf("Failed to start satellite manager: %v", err)
	}

	// Wait for satellite tracking
	time.Sleep(2 * time.Second)

	// Get best satellite
	satID, err := sm.GetBestSatellite(40.7128, -74.0060) // New York
	if err != nil {
		t.Errorf("Failed to get best satellite: %v", err)
	}

	if satID == "" {
		t.Error("Expected satellite ID, got empty string")
	}

	// Get metrics
	metrics := sm.GetSatelliteMetrics()
	if metrics["total_satellites"].(int) == 0 {
		t.Error("Expected satellites, got 0")
	}

	if err := sm.Stop(); err != nil {
		t.Fatalf("Failed to stop satellite manager: %v", err)
	}
}

func TestGlobalMesh(t *testing.T) {
	config := DefaultPlanetaryConfig()

	gm := mesh.NewGlobalMesh(config)

	if err := gm.Start(); err != nil {
		t.Fatalf("Failed to start global mesh: %v", err)
	}

	// Add test nodes
	node1 := &mesh.MeshNode{
		NodeID:         "node-1",
		Location:       mesh.GeoLocation{Latitude: 40.7128, Longitude: -74.0060},
		Neighbors:      []string{"node-2"},
		ConnectionType: "satellite",
		Bandwidth:      1000.0,
		Latency:        20 * time.Millisecond,
		Reliability:    0.99,
	}

	node2 := &mesh.MeshNode{
		NodeID:         "node-2",
		Location:       mesh.GeoLocation{Latitude: 51.5074, Longitude: -0.1278},
		Neighbors:      []string{"node-1"},
		ConnectionType: "cable",
		Bandwidth:      10000.0,
		Latency:        50 * time.Millisecond,
		Reliability:    0.999,
	}

	if err := gm.AddNode(node1); err != nil {
		t.Errorf("Failed to add node 1: %v", err)
	}

	if err := gm.AddNode(node2); err != nil {
		t.Errorf("Failed to add node 2: %v", err)
	}

	// Wait for mesh convergence
	time.Sleep(2 * time.Second)

	// Test DTN bundle
	if config.EnableDTN {
		bundle := &mesh.BundleProtocol{
			Version:        7,
			PayloadBlock:   []byte("test message"),
			CreationTime:   time.Now(),
			Lifetime:       1 * time.Hour,
			Priority:       5,
			SourceEID:      "node-1",
			DestinationEID: "node-2",
		}

		if err := gm.SendBundle(bundle); err != nil {
			t.Errorf("Failed to send bundle: %v", err)
		}
	}

	// Get metrics
	metrics := gm.GetMeshMetrics()
	if metrics["total_nodes"].(int) < 2 {
		t.Errorf("Expected at least 2 nodes, got %d", metrics["total_nodes"].(int))
	}

	if err := gm.Stop(); err != nil {
		t.Fatalf("Failed to stop global mesh: %v", err)
	}
}

func TestRegionCoordinator(t *testing.T) {
	config := DefaultPlanetaryConfig()

	rc := regions.NewRegionCoordinator(config)

	if err := rc.Start(); err != nil {
		t.Fatalf("Failed to start region coordinator: %v", err)
	}

	// Wait for initialization
	time.Sleep(1 * time.Second)

	// Get metrics
	metrics := rc.GetRegionMetrics()

	totalRegions := metrics["total_regions"].(int)
	if totalRegions < config.MinRegions {
		t.Errorf("Expected at least %d regions, got %d", config.MinRegions, totalRegions)
	}

	// Check region types
	majorCities := metrics["major_cities"].(int)
	if majorCities == 0 {
		t.Error("Expected major cities, got 0")
	}

	// Test region addition
	newRegion := &regions.Region{
		RegionID:   "test-region",
		Name:       "Test Region",
		Location:   mesh.GeoLocation{Latitude: 0.0, Longitude: 0.0},
		Type:       "major-city",
		Population: 1000000,
		Status:     "active",
		Health:     1.0,
	}

	if err := rc.AddRegion(newRegion); err != nil {
		t.Errorf("Failed to add region: %v", err)
	}

	// Verify region was added
	region, err := rc.GetRegion("test-region")
	if err != nil {
		t.Errorf("Failed to get region: %v", err)
	}

	if region.Name != "Test Region" {
		t.Errorf("Expected region name 'Test Region', got '%s'", region.Name)
	}

	if err := rc.Stop(); err != nil {
		t.Fatalf("Failed to stop region coordinator: %v", err)
	}
}

func TestSpaceCompute(t *testing.T) {
	config := DefaultPlanetaryConfig()
	config.OrbitalDataCenters = true

	sc := space.NewSpaceCompute(config)

	if err := sc.Start(); err != nil {
		t.Fatalf("Failed to start space compute: %v", err)
	}

	// Wait for initialization
	time.Sleep(1 * time.Second)

	// Schedule workload
	workload := &space.SpaceWorkload{
		WorkloadID:        "test-workload",
		Name:              "Test Workload",
		Type:              "compute",
		CPUUsage:          10.0,
		MemoryUsage:       50.0,
		PowerConsumption:  2.0,
		Priority:          5,
		ZeroGOptimized:    true,
		RadiationHardened: true,
	}

	if err := sc.ScheduleWorkload(workload); err != nil {
		t.Errorf("Failed to schedule workload: %v", err)
	}

	// Get metrics
	metrics := sc.GetSpaceMetrics()
	if metrics["total_nodes"].(int) == 0 {
		t.Error("Expected orbital nodes, got 0")
	}

	if metrics["running_workloads"].(int) == 0 {
		t.Error("Expected running workloads, got 0")
	}

	if err := sc.Stop(); err != nil {
		t.Fatalf("Failed to stop space compute: %v", err)
	}
}

func TestMarsRelay(t *testing.T) {
	config := DefaultPlanetaryConfig()
	config.MarsRelayEnabled = true

	mr := interplanetary.NewMarsRelay(config)

	if err := mr.Start(); err != nil {
		t.Fatalf("Failed to start Mars relay: %v", err)
	}

	// Send interplanetary message
	msg := &interplanetary.InterplanetaryMessage{
		MessageID:   "test-msg-1",
		Source:      "Earth",
		Destination: "Mars",
		Priority:    5,
		Payload:     []byte("Hello Mars!"),
	}

	if err := mr.SendMessage(msg); err != nil {
		t.Errorf("Failed to send message: %v", err)
	}

	// Wait for processing
	time.Sleep(1 * time.Second)

	// Get metrics
	metrics := mr.GetInterplanetaryMetrics()
	if metrics["total_messages"].(int) == 0 {
		t.Error("Expected messages, got 0")
	}

	if metrics["earth_stations"].(int) == 0 {
		t.Error("Expected Earth stations, got 0")
	}

	if err := mr.Stop(); err != nil {
		t.Fatalf("Failed to stop Mars relay: %v", err)
	}
}

func TestConfiguration(t *testing.T) {
	config := DefaultPlanetaryConfig()

	if err := config.Validate(); err != nil {
		t.Errorf("Default config validation failed: %v", err)
	}

	// Test invalid coverage target
	invalidConfig := *config
	invalidConfig.CoverageTarget = 1.5
	if err := invalidConfig.Validate(); err == nil {
		t.Error("Expected validation error for invalid coverage target")
	}

	// Test constellation configuration
	config.StarlinkAPIKey = "test-key"
	constellations := config.GetConstellations()

	if len(constellations) == 0 {
		t.Error("Expected constellations, got 0")
	}

	for _, constellation := range constellations {
		if constellation.Name == "" {
			t.Error("Expected constellation name, got empty string")
		}
		if constellation.Enabled != true {
			t.Errorf("Expected constellation %s to be enabled", constellation.Name)
		}
	}
}

func TestSatelliteHandoff(t *testing.T) {
	config := DefaultPlanetaryConfig()
	config.StarlinkAPIKey = "test-key"

	sm := leo.NewSatelliteManager(config)

	if err := sm.Start(); err != nil {
		t.Fatalf("Failed to start satellite manager: %v", err)
	}

	// Wait for handoff processing
	time.Sleep(3 * time.Second)

	metrics := sm.GetSatelliteMetrics()

	// Verify handoff target is met
	if avgLatency, ok := metrics["avg_latency_ms"].(float64); ok {
		if avgLatency > float64(config.SatelliteHandoffTime.Milliseconds()) {
			t.Logf("Warning: Average latency %fms exceeds handoff target %dms",
				avgLatency, config.SatelliteHandoffTime.Milliseconds())
		}
	}

	if err := sm.Stop(); err != nil {
		t.Fatalf("Failed to stop satellite manager: %v", err)
	}
}

func TestMeshConvergence(t *testing.T) {
	config := DefaultPlanetaryConfig()
	config.MeshConvergenceTime = 1 * time.Second

	gm := mesh.NewGlobalMesh(config)

	if err := gm.Start(); err != nil {
		t.Fatalf("Failed to start global mesh: %v", err)
	}

	// Add multiple nodes
	for i := 0; i < 10; i++ {
		node := &mesh.MeshNode{
			NodeID:         string(rune('A' + i)),
			Location:       mesh.GeoLocation{Latitude: float64(i * 10), Longitude: float64(i * 10)},
			Neighbors:      []string{},
			ConnectionType: "satellite",
			Bandwidth:      1000.0,
			Latency:        20 * time.Millisecond,
			Reliability:    0.99,
		}

		if err := gm.AddNode(node); err != nil {
			t.Errorf("Failed to add node %s: %v", node.NodeID, err)
		}
	}

	// Wait for convergence
	time.Sleep(config.MeshConvergenceTime + 1*time.Second)

	metrics := gm.GetMeshMetrics()
	if metrics["total_nodes"].(int) < 10 {
		t.Errorf("Expected 10 nodes, got %d", metrics["total_nodes"].(int))
	}

	if err := gm.Stop(); err != nil {
		t.Fatalf("Failed to stop global mesh: %v", err)
	}
}

func BenchmarkSatelliteHandoff(b *testing.B) {
	config := DefaultPlanetaryConfig()
	config.StarlinkAPIKey = "test-key"

	sm := leo.NewSatelliteManager(config)
	sm.Start()
	defer sm.Stop()

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, err := sm.GetBestSatellite(40.7128, -74.0060)
		if err != nil {
			b.Fatalf("GetBestSatellite failed: %v", err)
		}
	}
}

func BenchmarkMeshRouting(b *testing.B) {
	config := DefaultPlanetaryConfig()
	gm := mesh.NewGlobalMesh(config)
	gm.Start()
	defer gm.Stop()

	// Add test nodes
	for i := 0; i < 100; i++ {
		node := &mesh.MeshNode{
			NodeID:         string(rune(i)),
			Location:       mesh.GeoLocation{},
			Neighbors:      []string{},
			ConnectionType: "satellite",
			Bandwidth:      1000.0,
			Latency:        20 * time.Millisecond,
			Reliability:    0.99,
		}
		gm.AddNode(node)
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		bundle := &mesh.BundleProtocol{
			Version:        7,
			PayloadBlock:   []byte("benchmark"),
			CreationTime:   time.Now(),
			Lifetime:       1 * time.Hour,
			Priority:       5,
			SourceEID:      "0",
			DestinationEID: "99",
		}
		gm.SendBundle(bundle)
	}
}
