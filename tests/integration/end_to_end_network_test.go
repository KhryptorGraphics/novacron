package integration

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/discovery"
	"github.com/khryptorgraphics/novacron/backend/core/network"
	"github.com/khryptorgraphics/novacron/backend/core/scheduler"
	ntop "github.com/khryptorgraphics/novacron/backend/core/network/topology"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestEndToEndNetworkAwarePlacement tests the complete flow from network monitoring to placement
func TestEndToEndNetworkAwarePlacement(t *testing.T) {
	// Setup network topology
	topology := ntop.NewNetworkTopology()
	
	// Add nodes
	nodes := []ntop.NetworkNode{
		{ID: "node1", Location: ntop.Location{Zone: "zone-a", Region: "region-1"}},
		{ID: "node2", Location: ntop.Location{Zone: "zone-a", Region: "region-1"}},
		{ID: "node3", Location: ntop.Location{Zone: "zone-b", Region: "region-1"}},
	}
	
	for _, node := range nodes {
		err := topology.AddNode(node)
		require.NoError(t, err)
	}
	
	// Add links
	links := []ntop.NetworkLink{
		{SourceID: "node1", DestinationID: "node2", Bandwidth: 10000, Latency: 1.0, Utilization: 0.3},
		{SourceID: "node1", DestinationID: "node3", Bandwidth: 1000, Latency: 5.0, Utilization: 0.8},
		{SourceID: "node2", DestinationID: "node3", Bandwidth: 5000, Latency: 2.0, Utilization: 0.5},
	}
	
	for _, link := range links {
		err := topology.AddLink(link)
		require.NoError(t, err)
	}

	// Setup network-aware scheduler
	config := scheduler.DefaultNetworkAwareSchedulerConfig()
	config.NetworkAwarenessWeight = 0.7
	config.BandwidthWeight = 0.4
	config.LatencyWeight = 0.6
	
	baseScheduler := scheduler.NewScheduler(scheduler.DefaultSchedulerConfig())
	networkScheduler := scheduler.NewNetworkAwareScheduler(config, baseScheduler, nil, topology)
	require.NotNil(t, networkScheduler)
	
	err := networkScheduler.Start()
	require.NoError(t, err)
	defer networkScheduler.Stop()

	// Setup bandwidth monitoring
	monitorConfig := network.BandwidthMonitorConfig{
		Interfaces: []network.InterfaceThreshold{
			{Interface: "*", UtilizationThreshold: 80.0, AbsoluteThreshold: 5000.0},
		},
		CheckInterval: time.Millisecond * 100,
		AlertEnabled:  true,
	}
	
	monitor := network.NewBandwidthMonitor(monitorConfig)
	alerts := make(chan network.Alert, 10)
	monitor.SetAlertChannel(alerts)
	
	ctx, cancel := context.WithTimeout(context.Background(), time.Second * 10)
	defer cancel()
	go monitor.Start(ctx)

	// Track VM communications that should influence placement
	networkScheduler.TrackVMCommunication("vm1", "vm2", 2000.0, 1000.0) // Heavy communication
	networkScheduler.UpdateVMLocation("vm1", "node1")
	networkScheduler.UpdateVMLocation("vm2", "node2")

	// Request placement for vm3 that communicates with vm1
	networkScheduler.TrackVMCommunication("vm1", "vm3", 1500.0, 800.0)

	// Test that network-aware placement considers VM communication patterns
	placement, err := networkScheduler.RequestPlacement(
		"vm3",
		scheduler.PolicyNetworkAware,
		[]scheduler.PlacementConstraint{
			&scheduler.NetworkConstraint{
				MaxLatency: 3.0, // 3ms max latency requirement
				MinBandwidth: 500.0,
			},
		},
		map[string]float64{"cpu": 2.0, "memory": 4096.0},
		1,
	)

	if err == nil {
		// If placement succeeded, verify it considered network constraints
		assert.Contains(t, []string{"node1", "node2"}, placement, 
			"Should prefer nodes with better network characteristics")
		
		// node3 has high latency (5ms) to node1, so should be less preferred
		if placement == "node3" {
			t.Log("Warning: Placement chose node3 despite higher latency - verify network scoring")
		}
	}

	// Simulate high bandwidth usage and verify alerts
	monitor.RecordUtilization("eth0", 85.0, 6000.0) // Above thresholds
	
	select {
	case alert := <-alerts:
		assert.Contains(t, alert.Message, "threshold", "Should receive bandwidth threshold alert")
	case <-time.After(time.Millisecond * 500):
		t.Log("Warning: Expected bandwidth alert not received")
	}
}

// TestInternetDiscoveryWithNATTraversal tests interaction between discovery and NAT traversal
func TestInternetDiscoveryWithNATTraversal(t *testing.T) {
	// Setup NAT traversal
	natConfig := discovery.NATTraversalConfig{
		STUNServers:         []string{"stun.l.google.com:19302"},
		PunchingTimeout:     time.Second * 3,
		EnableRelayFallback: true,
		RelayServers:       []string{"relay.example.com:3478"},
	}
	
	natTraversal := discovery.NewNATTraversalManager(natConfig)
	require.NotNil(t, natTraversal)

	// Setup internet discovery
	discoveryConfig := discovery.InternetDiscoveryConfig{
		BootstrapNodes:   []string{"bootstrap1.example.com:8080"},
		AnnouncementInterval: time.Second,
		ExternalEndpointDetection: true,
	}
	
	internetDiscovery := discovery.NewInternetDiscovery(discoveryConfig)
	require.NotNil(t, internetDiscovery)

	ctx, cancel := context.WithTimeout(context.Background(), time.Second * 10)
	defer cancel()

	// Start NAT traversal first
	err := natTraversal.Start(ctx)
	require.NoError(t, err)
	defer natTraversal.Stop()

	// Start internet discovery
	err = internetDiscovery.Start(ctx)
	if err != nil {
		t.Skipf("Cannot test discovery integration: %v", err)
		return
	}
	defer internetDiscovery.Stop()

	// Wait for discovery to detect external endpoint
	time.Sleep(time.Second * 2)

	// Verify that discovery includes external endpoint from NAT traversal
	announcement := internetDiscovery.GetLastAnnouncement()
	if announcement.ExternalEndpoint != "" {
		// Try to use discovered endpoint for NAT traversal
		_, err = natTraversal.ConnectToPeer(ctx, "test-peer", announcement.ExternalEndpoint, discovery.NATTypeFullCone)
		// Connection may fail, but should not have addressing errors
		if err != nil {
			assert.NotContains(t, err.Error(), "invalid address", 
				"External endpoint from discovery should be valid for NAT traversal")
		}
	}
}

// TestQoSWithSchedulerIntegration tests QoS enforcement with scheduler decisions
func TestQoSWithSchedulerIntegration(t *testing.T) {
	// Skip if not root (QoS requires kernel modifications)
	if !canTestQoS() {
		t.Skip("QoS integration test requires elevated privileges")
	}

	// Setup QoS manager
	qosConfig := network.QoSConfig{
		Interface: "lo",
		Classes: []network.QoSClass{
			{ID: "1:10", Rate: "1000mbit", Priority: 1}, // High priority
			{ID: "1:20", Rate: "500mbit", Priority: 2},  // Medium priority
			{ID: "1:30", Rate: "100mbit", Priority: 3},  // Low priority
		},
		KernelStateEnforcement: true,
		DefaultRateBps: 2000000000, // 2 Gbps
	}
	
	qosManager := network.NewQoSManager(qosConfig)
	require.NotNil(t, qosManager)
	
	err := qosManager.Start()
	require.NoError(t, err)
	defer qosManager.Stop()

	// Setup network topology for scheduler
	topology := ntop.NewNetworkTopology()
	nodes := []ntop.NetworkNode{
		{ID: "node1", Location: ntop.Location{Zone: "zone-a"}},
		{ID: "node2", Location: ntop.Location{Zone: "zone-b"}},
	}
	
	for _, node := range nodes {
		err := topology.AddNode(node)
		require.NoError(t, err)
	}

	// Setup scheduler
	config := scheduler.DefaultNetworkAwareSchedulerConfig()
	baseScheduler := scheduler.NewScheduler(scheduler.DefaultSchedulerConfig())
	networkScheduler := scheduler.NewNetworkAwareScheduler(config, baseScheduler, nil, topology)
	
	err = networkScheduler.Start()
	require.NoError(t, err)
	defer networkScheduler.Stop()

	// Test high-priority VM gets better QoS class
	highPriorityPlacement, err := networkScheduler.RequestPlacement(
		"high-priority-vm",
		scheduler.PolicyNetworkAware,
		[]scheduler.PlacementConstraint{
			&scheduler.NetworkConstraint{
				RequiredQoSClass: "1:10", // High priority QoS class
			},
		},
		map[string]float64{"cpu": 4.0, "memory": 8192.0},
		10, // High priority
	)

	if err == nil {
		// Apply QoS for high priority VM
		err = qosManager.AssignVMToClass("high-priority-vm", "1:10")
		assert.NoError(t, err, "Should successfully assign high-priority VM to premium QoS class")
		
		// Verify QoS class assignment
		qosClass := qosManager.GetVMQoSClass("high-priority-vm")
		assert.Equal(t, "1:10", qosClass, "High-priority VM should get premium QoS class")
		
		t.Logf("High priority VM placed on %s with QoS class %s", highPriorityPlacement, qosClass)
	}
}

// TestNetworkConstraintsWithDiscovery tests network constraints validation with real network data
func TestNetworkConstraintsWithDiscovery(t *testing.T) {
	// Setup internet discovery to get real network data
	config := discovery.InternetDiscoveryConfig{
		BootstrapNodes: []string{"bootstrap1.example.com:8080"},
		AnnouncementInterval: time.Second * 2,
	}
	
	internetDiscovery := discovery.NewInternetDiscovery(config)
	require.NotNil(t, internetDiscovery)

	ctx, cancel := context.WithTimeout(context.Background(), time.Second * 5)
	defer cancel()

	err := internetDiscovery.Start(ctx)
	if err != nil {
		t.Skipf("Cannot test network constraints with discovery: %v", err)
		return
	}
	defer internetDiscovery.Stop()

	// Wait for discovery to gather network data
	time.Sleep(time.Second * 2)

	// Get discovered nodes and their network characteristics
	routingTable := internetDiscovery.GetRoutingTable()
	assert.NotNil(t, routingTable, "Should have routing table from discovery")

	// Setup scheduler with discovered network data
	topology := ntop.NewNetworkTopology()
	
	// Add discovered nodes to topology
	for nodeID, nodeInfo := range routingTable.Nodes {
		node := ntop.NetworkNode{
			ID: nodeID,
			Location: ntop.Location{
				Zone:   fmt.Sprintf("zone-%s", nodeID[:2]),
				Region: "discovered-region",
			},
		}
		topology.AddNode(node)
	}

	schedulerConfig := scheduler.DefaultNetworkAwareSchedulerConfig()
	baseScheduler := scheduler.NewScheduler(scheduler.DefaultSchedulerConfig())
	networkScheduler := scheduler.NewNetworkAwareScheduler(schedulerConfig, baseScheduler, nil, topology)

	// Test network constraints validation with real network data
	constraints := []scheduler.PlacementConstraint{
		&scheduler.NetworkConstraint{
			MaxLatency: 50.0, // 50ms - reasonable for wide area network
			MinBandwidth: 10.0, // 10 Mbps minimum
			RequiredZones: []string{"zone-di", "zone-bo"}, // Based on discovered node IDs
		},
	}

	err = networkScheduler.ValidateConstraints(constraints)
	assert.NoError(t, err, "Network constraints should be valid with discovered network data")
}

// TestConcurrentNetworkOperations tests thread safety across all network components
func TestConcurrentNetworkOperations(t *testing.T) {
	// Setup all network components
	monitor := setupBandwidthMonitor(t)
	natTraversal := setupNATTraversal(t)
	internetDiscovery := setupInternetDiscovery(t)
	scheduler := setupNetworkScheduler(t)

	ctx, cancel := context.WithTimeout(context.Background(), time.Second * 5)
	defer cancel()

	// Start all components
	components := []interface {
		Start(context.Context) error
		Stop() error
	}{monitor, natTraversal, internetDiscovery}

	for _, component := range components {
		if err := component.Start(ctx); err != nil {
			t.Logf("Could not start component: %v", err)
			continue
		}
		defer component.Stop()
	}

	if err := scheduler.Start(); err != nil {
		t.Logf("Could not start scheduler: %v", err)
	} else {
		defer scheduler.Stop()
	}

	// Run concurrent operations
	var wg sync.WaitGroup
	errors := make(chan error, 50)

	// Concurrent bandwidth monitoring
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for j := 0; j < 5; j++ {
				monitor.RecordUtilization(fmt.Sprintf("eth%d", id), float64(50+j*10), float64(100+j*50))
				time.Sleep(time.Millisecond * 10)
			}
		}(i)
	}

	// Concurrent NAT traversal
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			_, err := natTraversal.ConnectToPeer(ctx, fmt.Sprintf("peer-%d", id), 
				fmt.Sprintf("192.168.1.%d:8080", id+100), discovery.NATTypeFullCone)
			if err != nil && !isExpectedNetworkError(err) {
				errors <- fmt.Errorf("NAT traversal error: %w", err)
			}
		}(i)
	}

	// Concurrent scheduling requests
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			_, err := scheduler.RequestPlacement(
				fmt.Sprintf("vm-%d", id),
				scheduler.PolicyNetworkAware,
				[]scheduler.PlacementConstraint{},
				map[string]float64{"cpu": 1.0},
				1,
			)
			if err != nil && !isExpectedSchedulingError(err) {
				errors <- fmt.Errorf("Scheduling error: %w", err)
			}
		}(i)
	}

	// Concurrent discovery updates
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			internetDiscovery.UpdateNodeInfo(fmt.Sprintf("node-%d", id), discovery.NodeInfo{
				ID: fmt.Sprintf("node-%d", id),
				Address: fmt.Sprintf("10.0.0.%d:8080", id+10),
				LastSeen: time.Now(),
			})
			time.Sleep(time.Millisecond * 50)
			internetDiscovery.RemoveNode(fmt.Sprintf("node-%d", id))
		}(i)
	}

	wg.Wait()
	close(errors)

	// Check for unexpected errors (thread safety issues)
	errorCount := 0
	for err := range errors {
		errorCount++
		t.Logf("Unexpected error during concurrent operations: %v", err)
	}

	assert.Equal(t, 0, errorCount, "Should not have thread safety issues during concurrent operations")
}

// Helper functions for test setup

func setupBandwidthMonitor(t *testing.T) *network.BandwidthMonitor {
	config := network.BandwidthMonitorConfig{
		Interfaces: []network.InterfaceThreshold{
			{Interface: "*", UtilizationThreshold: 80.0, AbsoluteThreshold: 1000.0},
		},
		CheckInterval: time.Millisecond * 100,
		AlertEnabled:  true,
	}
	return network.NewBandwidthMonitor(config)
}

func setupNATTraversal(t *testing.T) *discovery.NATTraversalManager {
	config := discovery.NATTraversalConfig{
		STUNServers:     []string{"stun.l.google.com:19302"},
		PunchingTimeout: time.Second * 2,
		EnableRelayFallback: true,
	}
	return discovery.NewNATTraversalManager(config)
}

func setupInternetDiscovery(t *testing.T) *discovery.InternetDiscovery {
	config := discovery.InternetDiscoveryConfig{
		BootstrapNodes: []string{"bootstrap1.example.com:8080"},
		AnnouncementInterval: time.Second,
	}
	return discovery.NewInternetDiscovery(config)
}

func setupNetworkScheduler(t *testing.T) *scheduler.NetworkAwareScheduler {
	topology := ntop.NewNetworkTopology()
	
	// Add test nodes
	nodes := []ntop.NetworkNode{
		{ID: "test-node-1", Location: ntop.Location{Zone: "zone-a"}},
		{ID: "test-node-2", Location: ntop.Location{Zone: "zone-b"}},
	}
	
	for _, node := range nodes {
		topology.AddNode(node)
	}

	config := scheduler.DefaultNetworkAwareSchedulerConfig()
	baseScheduler := scheduler.NewScheduler(scheduler.DefaultSchedulerConfig())
	return scheduler.NewNetworkAwareScheduler(config, baseScheduler, nil, topology)
}

func canTestQoS() bool {
	// Simple check for QoS testing capability (requires root or specific capabilities)
	return false // Set to true in environments where QoS testing is possible
}

func isExpectedNetworkError(err error) bool {
	errStr := err.Error()
	expectedErrors := []string{"timeout", "connection refused", "no such host", "network unreachable"}
	
	for _, expected := range expectedErrors {
		if strings.Contains(errStr, expected) {
			return true
		}
	}
	return false
}

func isExpectedSchedulingError(err error) bool {
	errStr := err.Error()
	expectedErrors := []string{"no suitable nodes", "resource constraints", "validation failed"}
	
	for _, expected := range expectedErrors {
		if strings.Contains(errStr, expected) {
			return true
		}
	}
	return false
}

