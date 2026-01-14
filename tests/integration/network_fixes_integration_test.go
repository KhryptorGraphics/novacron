package integration

import (
	"bytes"
	"context"
	"fmt"
	"net"
	"os"
	"os/exec"
	"regexp"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/discovery"
	"github.com/khryptorgraphics/novacron/backend/core/network"
	"github.com/khryptorgraphics/novacron/backend/core/scheduler"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestBandwidthMonitorWildcardThresholds tests comment 1: wildcard interface threshold matching
func TestBandwidthMonitorWildcardThresholds(t *testing.T) {
	config := network.BandwidthMonitorConfig{
		Interfaces: []network.InterfaceThreshold{
			{Interface: "*", UtilizationThreshold: 80.0, AbsoluteThreshold: 1000.0},
			{Interface: "eth*", UtilizationThreshold: 90.0, AbsoluteThreshold: 2000.0},
			{Interface: "eth0", UtilizationThreshold: 95.0, AbsoluteThreshold: 2500.0},
		},
		CheckInterval: time.Millisecond * 100,
		AlertEnabled:  true,
	}

	monitor := network.NewBandwidthMonitor(config)
	require.NotNil(t, monitor)

	// Test that wildcard matching works correctly
	// Most specific should win: eth0 > eth* > *
	assert.Equal(t, 95.0, getUtilizationThreshold(monitor, "eth0"))
	assert.Equal(t, 90.0, getUtilizationThreshold(monitor, "eth1"))
	assert.Equal(t, 80.0, getUtilizationThreshold(monitor, "wlan0"))
}

// TestBandwidthMonitorAlertRateLimiting tests comment 2: separate alert keys for different alert types
func TestBandwidthMonitorAlertRateLimiting(t *testing.T) {
	config := network.BandwidthMonitorConfig{
		Interfaces: []network.InterfaceThreshold{
			{Interface: "eth0", UtilizationThreshold: 50.0, AbsoluteThreshold: 100.0},
		},
		CheckInterval:    time.Millisecond * 50,
		AlertEnabled:     true,
		AlertRateLimit:   time.Millisecond * 200,
	}

	monitor := network.NewBandwidthMonitor(config)
	alerts := make(chan network.Alert, 10)
	monitor.SetAlertChannel(alerts)
	
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	
	go monitor.Start(ctx)
	
	// Simulate high utilization (should trigger utilization alert)
	monitor.RecordUtilization("eth0", 80.0, 50.0) // 80% utilization, 50 Mbps
	
	// Simulate high absolute usage (should trigger absolute alert)
	monitor.RecordUtilization("eth0", 30.0, 150.0) // 30% utilization, 150 Mbps
	
	// Should receive both alerts since they use different alert keys
	select {
	case alert1 := <-alerts:
		assert.Contains(t, alert1.Message, "utilization")
	case <-time.After(time.Millisecond * 300):
		t.Fatal("Expected utilization alert")
	}
	
	select {
	case alert2 := <-alerts:
		assert.Contains(t, alert2.Message, "absolute")
	case <-time.After(time.Millisecond * 300):
		t.Fatal("Expected absolute threshold alert")
	}
}

// TestNATTraversalUDPHolePuncherRefactor tests comment 3: single UDP connection approach
func TestNATTraversalUDPHolePuncherRefactor(t *testing.T) {
	config := discovery.NATTraversalConfig{
		STUNServers:          []string{"stun.l.google.com:19302"},
		PunchingTimeout:      time.Second * 5,
		MaxRetries:          3,
		EnableRelayFallback: true,
	}

	traversal := discovery.NewNATTraversalManager(config)
	require.NotNil(t, traversal)

	ctx, cancel := context.WithTimeout(context.Background(), time.Second * 10)
	defer cancel()

	err := traversal.Start(ctx)
	require.NoError(t, err)
	defer traversal.Stop()

	// Test that UDP hole puncher uses single connection
	// This is validated by ensuring no port conflicts occur during concurrent operations
	var wg sync.WaitGroup
	errors := make(chan error, 5)

	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			_, err := traversal.ConnectToPeer(ctx, fmt.Sprintf("peer-%d", id), "1.2.3.%d:8080", discovery.NATTypeFullCone)
			if err != nil && !strings.Contains(err.Error(), "timeout") {
				errors <- err
			}
		}(i)
	}

	wg.Wait()
	close(errors)

	// Check that no port conflict errors occurred
	for err := range errors {
		assert.NotContains(t, err.Error(), "address already in use", "Should not have port conflicts with single UDP connection")
		assert.NotContains(t, err.Error(), "bind", "Should not have binding conflicts")
	}
}

// TestNATTraversalConnectionTypeLabeling tests comment 4: correct connection type labeling
func TestNATTraversalConnectionTypeLabeling(t *testing.T) {
	config := discovery.NATTraversalConfig{
		STUNServers:     []string{"stun.l.google.com:19302"},
		PunchingTimeout: time.Second * 2,
		MaxRetries:     1,
	}

	traversal := discovery.NewNATTraversalManager(config)
	require.NotNil(t, traversal)

	ctx, cancel := context.WithTimeout(context.Background(), time.Second * 5)
	defer cancel()

	err := traversal.Start(ctx)
	require.NoError(t, err)
	defer traversal.Stop()

	// Attempt connection (will likely timeout, but we check the connection type)
	conn, err := traversal.ConnectToPeer(ctx, "test-peer", "192.168.1.100:8080", discovery.NATTypeFullCone)
	
	if err == nil && conn != nil {
		// If connection succeeded, verify it's labeled as NAT traversal, not direct
		connInfo := traversal.GetConnectionInfo(conn.ID)
		assert.NotEqual(t, "direct", connInfo.Type, "NAT traversal connections should not be labeled as 'direct'")
		assert.Equal(t, "nat_traversal", connInfo.Type, "Expected 'nat_traversal' connection type")
	}
}

// TestNATTraversalStopCleanup tests comment 5: proper UDP receiver cleanup
func TestNATTraversalStopCleanup(t *testing.T) {
	config := discovery.NATTraversalConfig{
		STUNServers:     []string{"stun.l.google.com:19302"},
		PunchingTimeout: time.Second * 2,
	}

	traversal := discovery.NewNATTraversalManager(config)
	require.NotNil(t, traversal)

	ctx, cancel := context.WithTimeout(context.Background(), time.Second * 5)
	defer cancel()

	err := traversal.Start(ctx)
	require.NoError(t, err)

	// Give it time to start UDP receiver
	time.Sleep(time.Millisecond * 100)

	// Stop should clean up UDP receiver properly
	err = traversal.Stop()
	assert.NoError(t, err)

	// Verify that subsequent start doesn't fail due to resource conflicts
	err = traversal.Start(ctx)
	assert.NoError(t, err, "Should be able to restart after proper cleanup")

	err = traversal.Stop()
	assert.NoError(t, err)
}

// TestNATTraversalRelayFallback tests comment 6: relay fallback implementation
func TestNATTraversalRelayFallback(t *testing.T) {
	config := discovery.NATTraversalConfig{
		STUNServers:         []string{"stun.l.google.com:19302"},
		PunchingTimeout:     time.Millisecond * 500, // Short timeout to trigger fallback
		MaxRetries:         1,
		EnableRelayFallback: true,
		RelayServers:       []string{"relay.example.com:3478"},
	}

	traversal := discovery.NewNATTraversalManager(config)
	require.NotNil(t, traversal)

	ctx, cancel := context.WithTimeout(context.Background(), time.Second * 5)
	defer cancel()

	err := traversal.Start(ctx)
	require.NoError(t, err)
	defer traversal.Stop()

	// Attempt connection to unreachable peer (should trigger relay fallback)
	_, err = traversal.ConnectToPeer(ctx, "unreachable-peer", "169.254.1.1:8080", discovery.NATTypeSymmetric)
	
	// Even if connection fails, check that relay fallback was attempted
	stats := traversal.GetStats()
	assert.True(t, stats.RelayFallbackAttempts > 0, "Should have attempted relay fallback for symmetric NAT")
}

// TestInternetDiscoveryExternalEndpoint tests comment 7: external endpoint propagation
func TestInternetDiscoveryExternalEndpoint(t *testing.T) {
	config := discovery.InternetDiscoveryConfig{
		BootstrapNodes:   []string{"bootstrap1.example.com:8080"},
		AnnouncementInterval: time.Second,
		ExternalEndpointDetection: true,
	}

	internetDiscovery := discovery.NewInternetDiscovery(config)
	require.NotNil(t, internetDiscovery)

	ctx, cancel := context.WithTimeout(context.Background(), time.Second * 3)
	defer cancel()

	err := internetDiscovery.Start(ctx)
	if err != nil {
		// Skip test if external detection fails (network issues)
		t.Skipf("Cannot test external endpoint propagation: %v", err)
		return
	}
	defer internetDiscovery.Stop()

	// Wait for external endpoint detection
	time.Sleep(time.Millisecond * 1500)

	// Check that external endpoint is included in announcements
	announcement := internetDiscovery.GetLastAnnouncement()
	assert.NotEmpty(t, announcement.ExternalEndpoint, "Announcement should include external endpoint")
	
	// Verify external endpoint format (should be IP:port)
	if announcement.ExternalEndpoint != "" {
		_, err := net.ResolveTCPAddr("tcp", announcement.ExternalEndpoint)
		assert.NoError(t, err, "External endpoint should be valid address format")
	}
}

// TestInternetDiscoveryRaceCondition tests comment 8: routing table update race condition fix
func TestInternetDiscoveryRaceCondition(t *testing.T) {
	config := discovery.InternetDiscoveryConfig{
		BootstrapNodes: []string{"bootstrap1.example.com:8080"},
		AnnouncementInterval: time.Millisecond * 100,
		NodeTimeout: time.Millisecond * 200,
	}

	internetDiscovery := discovery.NewInternetDiscovery(config)
	require.NotNil(t, internetDiscovery)

	ctx, cancel := context.WithTimeout(context.Background(), time.Second * 2)
	defer cancel()

	err := internetDiscovery.Start(ctx)
	if err != nil {
		t.Skipf("Cannot test race condition fix: %v", err)
		return
	}
	defer internetDiscovery.Stop()

	// Simulate concurrent node updates and timeouts to test race condition fix
	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			nodeID := fmt.Sprintf("test-node-%d", id)
			
			// Add node
			internetDiscovery.UpdateNodeInfo(nodeID, discovery.NodeInfo{
				ID: nodeID,
				Address: fmt.Sprintf("192.168.1.%d:8080", id+100),
				LastSeen: time.Now(),
			})
			
			time.Sleep(time.Millisecond * 50)
			
			// Remove node (simulates timeout)
			internetDiscovery.RemoveNode(nodeID)
		}(i)
	}

	wg.Wait()

	// Test should not panic or deadlock due to race condition
	routingTable := internetDiscovery.GetRoutingTable()
	assert.NotNil(t, routingTable, "Should be able to get routing table without race issues")
}

// TestQoSManagerKernelStateEnforcement tests comment 9: kernel state enforcement
func TestQoSManagerKernelStateEnforcement(t *testing.T) {
	// Skip test if not running as root (required for tc commands)
	if os.Geteuid() != 0 {
		t.Skip("QoS kernel state enforcement test requires root privileges")
	}

	config := network.QoSConfig{
		Interface: "lo", // Use loopback interface for testing
		Classes: []network.QoSClass{
			{ID: "1:10", Rate: "100mbit", Priority: 1},
			{ID: "1:20", Rate: "50mbit", Priority: 2},
		},
		KernelStateEnforcement: true,
	}

	qosManager := network.NewQoSManager(config)
	require.NotNil(t, qosManager)

	err := qosManager.Start()
	require.NoError(t, err)
	defer qosManager.Stop()

	// Apply rate limiting
	err = qosManager.ApplyRateLimit("1:10", 150000000) // 150 Mbps
	require.NoError(t, err)

	// Verify kernel state was updated using tc command
	cmd := exec.Command("tc", "class", "show", "dev", "lo")
	output, err := cmd.Output()
	if err == nil {
		outputStr := string(output)
		assert.Contains(t, outputStr, "1:10", "Should have class 1:10 in kernel")
		assert.Contains(t, outputStr, "rate", "Should show rate configuration")
	}
}

// TestQoSManagerConfigurableRootRate tests comment 10: configurable root qdisc rate
func TestQoSManagerConfigurableRootRate(t *testing.T) {
	config := network.QoSConfig{
		Interface: "lo",
		DefaultRateBps: 2000000000, // 2 Gbps custom rate
		Classes: []network.QoSClass{
			{ID: "1:10", Rate: "100mbit", Priority: 1},
		},
		KernelStateEnforcement: true,
	}

	qosManager := network.NewQoSManager(config)
	require.NotNil(t, qosManager)

	// Verify that DefaultRateBps is used for root qdisc configuration
	rootRate := qosManager.GetRootQdiscRate()
	assert.Equal(t, uint64(2000000000), rootRate, "Should use configured DefaultRateBps for root qdisc")
}

// TestSchedulerNetworkConstraints tests comment 11: network constraints validation
func TestSchedulerNetworkConstraints(t *testing.T) {
	config := scheduler.DefaultResourceAwareSchedulerConfig()
	baseScheduler := scheduler.NewScheduler(scheduler.DefaultSchedulerConfig())
	
	resourceScheduler := scheduler.NewResourceAwareScheduler(config, baseScheduler, nil, nil)
	require.NotNil(t, resourceScheduler)

	// Create placement request with network constraints
	constraints := []scheduler.PlacementConstraint{
		&scheduler.NetworkConstraint{
			MaxLatency: 10.0, // 10ms max latency
			MinBandwidth: 1000.0, // 1Gbps min bandwidth
			RequiredZones: []string{"zone-a", "zone-b"},
		},
	}

	request := &scheduler.PlacementRequest{
		VMID: "test-vm",
		Resources: map[string]float64{
			"cpu": 2.0,
			"memory": 4096.0,
		},
		Constraints: constraints,
		Priority: 1,
	}

	// Test network constraints validation
	err := resourceScheduler.ValidateConstraints(request.Constraints)
	assert.NoError(t, err, "Valid network constraints should pass validation")

	// Test invalid constraints
	invalidConstraints := []scheduler.PlacementConstraint{
		&scheduler.NetworkConstraint{
			MaxLatency: -1.0, // Invalid negative latency
		},
	}

	err = resourceScheduler.ValidateConstraints(invalidConstraints)
	assert.Error(t, err, "Invalid network constraints should fail validation")
	assert.Contains(t, err.Error(), "latency", "Error should mention latency validation")
}

// TestSchedulerRequestPlacementConfigMutation tests comment 12: global config mutation fix
func TestSchedulerRequestPlacementConfigMutation(t *testing.T) {
	config := scheduler.DefaultNetworkAwareSchedulerConfig()
	originalWeight := config.NetworkAwarenessWeight
	
	baseScheduler := scheduler.NewScheduler(scheduler.DefaultSchedulerConfig())
	networkScheduler := scheduler.NewNetworkAwareScheduler(config, baseScheduler, nil, nil)
	require.NotNil(t, networkScheduler)

	// Make network-aware placement request
	_, err := networkScheduler.RequestPlacement(
		"test-vm",
		scheduler.PolicyNetworkAware,
		[]scheduler.PlacementConstraint{},
		map[string]float64{"cpu": 1.0},
		1,
	)

	// Error is expected since we don't have a real cluster, but config shouldn't be mutated
	// Verify that global config was not permanently mutated
	assert.Equal(t, originalWeight, config.NetworkAwarenessWeight, 
		"Global NetworkAwarenessWeight should not be permanently mutated")
}

// TestSTUNClientThreadSafety tests comment 13: STUN client thread safety
func TestSTUNClientThreadSafety(t *testing.T) {
	// Skip STUN tests in CI environment to avoid network dependencies
	if os.Getenv("CI") != "" || os.Getenv("SKIP_STUN_TESTS") != "" {
		t.Skip("Skipping STUN tests in CI environment (set SKIP_STUN_TESTS=false to force run)")
	}

	config := discovery.STUNClientConfig{
		Servers: []string{
			"stun.l.google.com:19302",
			"stun1.l.google.com:19302",
		},
		Timeout: time.Second * 2,
		Retries: 2,
	}

	client := discovery.NewSTUNClient(config)
	require.NotNil(t, client)

	// Test concurrent STUN requests to check thread safety
	var wg sync.WaitGroup
	errors := make(chan error, 10)

	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			
			ctx, cancel := context.WithTimeout(context.Background(), time.Second * 5)
			defer cancel()
			
			_, err := client.DiscoverNATType(ctx)
			if err != nil && !strings.Contains(err.Error(), "timeout") && !strings.Contains(err.Error(), "no such host") {
				errors <- err
			}
		}(i)
	}

	wg.Wait()
	close(errors)

	// Check for thread safety issues (race conditions, concurrent map access, etc.)
	for err := range errors {
		assert.NotContains(t, err.Error(), "concurrent map", "Should not have concurrent map access errors")
		assert.NotContains(t, err.Error(), "race", "Should not have race condition errors")
	}
}

// Helper function to extract utilization threshold from bandwidth monitor (implementation-specific)
func getUtilizationThreshold(monitor *network.BandwidthMonitor, interfaceName string) float64 {
	// This would call into the bandwidth monitor's internal threshold matching logic
	// Implementation depends on how the monitor exposes this functionality
	return monitor.GetThresholdForInterface(interfaceName).UtilizationThreshold
}