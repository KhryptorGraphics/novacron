package edge

import (
	"context"
	"testing"
	"time"
)

func TestEdgeDiscovery(t *testing.T) {
	config := DefaultEdgeConfig()
	config.DiscoveryInterval = 1 * time.Second
	discovery := NewEdgeDiscovery(config)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Test discovery start
	err := discovery.Start(ctx)
	if err != nil {
		t.Fatalf("Failed to start discovery: %v", err)
	}

	// Add a test node
	testNode := &EdgeNode{
		ID:   "test-node-1",
		Name: "Test Edge Node",
		Type: EdgeType5GMEC,
		Location: GeoLocation{
			Latitude:  37.7749,
			Longitude: -122.4194,
			Country:   "US",
			Region:    "us-west-1",
			City:      "San Francisco",
		},
		Resources: EdgeResources{
			TotalCPUCores:  16,
			UsedCPUCores:   4,
			TotalMemoryMB:  32768,
			UsedMemoryMB:   8192,
			TotalStorageGB: 1000,
			UsedStorageGB:  200,
			TotalBandwidthMbps: 10000,
			UsedBandwidthMbps:  2000,
		},
		Status: EdgeNodeStatus{
			State:  EdgeNodeStateOnline,
			Health: HealthStatusHealthy,
			ActiveVMs: 5,
		},
	}

	discovery.addOrUpdateNode(testNode)

	// Test GetNode
	node, err := discovery.GetNode("test-node-1")
	if err != nil {
		t.Fatalf("Failed to get node: %v", err)
	}
	if node.ID != "test-node-1" {
		t.Errorf("Expected node ID test-node-1, got %s", node.ID)
	}

	// Test GetAllNodes
	nodes := discovery.GetAllNodes()
	if len(nodes) != 1 {
		t.Errorf("Expected 1 node, got %d", len(nodes))
	}

	// Test GetHealthyNodes
	healthyNodes := discovery.GetHealthyNodes()
	if len(healthyNodes) != 1 {
		t.Errorf("Expected 1 healthy node, got %d", len(healthyNodes))
	}

	// Test AssessCapabilities
	err = discovery.AssessCapabilities(testNode)
	if err != nil {
		t.Errorf("Failed to assess capabilities: %v", err)
	}

	// Test Stop
	err = discovery.Stop()
	if err != nil {
		t.Fatalf("Failed to stop discovery: %v", err)
	}
}

func TestMeasureLatency(t *testing.T) {
	config := DefaultEdgeConfig()
	discovery := NewEdgeDiscovery(config)

	testNode := &EdgeNode{
		ID: "test-node-1",
		Network: NetworkInfo{
			PublicIP: "8.8.8.8", // Google DNS for testing
		},
	}

	ctx := context.Background()
	metrics, err := discovery.MeasureLatency(ctx, testNode)

	// This may fail in restricted networks, so we accept both success and failure
	if err == nil {
		if metrics.RTTAvg == 0 {
			t.Error("Expected non-zero RTT average")
		}
		t.Logf("Latency metrics: RTT Avg=%v, Min=%v, Max=%v",
			metrics.RTTAvg, metrics.RTTMin, metrics.RTTMax)
	}
}

func TestGetNodesByType(t *testing.T) {
	config := DefaultEdgeConfig()
	discovery := NewEdgeDiscovery(config)

	// Add nodes of different types
	mecNode := &EdgeNode{
		ID:   "mec-1",
		Type: EdgeType5GMEC,
		Status: EdgeNodeStatus{
			State:  EdgeNodeStateOnline,
			Health: HealthStatusHealthy,
		},
	}

	cdnNode := &EdgeNode{
		ID:   "cdn-1",
		Type: EdgeTypeCDN,
		Status: EdgeNodeStatus{
			State:  EdgeNodeStateOnline,
			Health: HealthStatusHealthy,
		},
	}

	discovery.addOrUpdateNode(mecNode)
	discovery.addOrUpdateNode(cdnNode)

	// Test filtering by type
	mecNodes := discovery.GetNodesByType(EdgeType5GMEC)
	if len(mecNodes) != 1 {
		t.Errorf("Expected 1 MEC node, got %d", len(mecNodes))
	}

	cdnNodes := discovery.GetNodesByType(EdgeTypeCDN)
	if len(cdnNodes) != 1 {
		t.Errorf("Expected 1 CDN node, got %d", len(cdnNodes))
	}
}
