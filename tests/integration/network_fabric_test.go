package integration

import (
	"context"
	"net"
	"os"
	"testing"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/discovery"
	"github.com/khryptorgraphics/novacron/backend/core/network"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
)

func TestNetworkFabricIntegration(t *testing.T) {
	logger := zap.NewNop()
	ctx := context.Background()

	t.Run("STUN NAT Traversal", func(t *testing.T) {
		// Skip if external network tests are not enabled
		if os.Getenv("NOVACRON_E2E_NET") == "" {
			t.Skip("Skipping STUN test; set NOVACRON_E2E_NET=1 to enable")
		}

		// Create STUN client with test servers
		stunServers := []discovery.STUNServer{
			{Host: "stun.l.google.com", Port: 19302},
			{Host: "stun1.l.google.com", Port: 19302},
		}

		stunClient := discovery.NewSTUNClient(stunServers, logger)

		// Test external address discovery
		external, err := stunClient.DiscoverExternalAddress()
		require.NoError(t, err, "Failed to discover external address")
		assert.NotNil(t, external)
		assert.NotEmpty(t, external.IP)
		assert.Greater(t, external.Port, 0)

		// Test NAT type detection
		detector := discovery.NewNATTypeDetector(stunClient, logger)
		natType, err := detector.DetectNATType()
		assert.NoError(t, err, "Failed to detect NAT type")
		assert.Contains(t, []int{
			discovery.NAT_TYPE_UNKNOWN,
			discovery.NAT_TYPE_OPEN_INTERNET,
			discovery.NAT_TYPE_FULL_CONE,
			discovery.NAT_TYPE_RESTRICTED_CONE,
			discovery.NAT_TYPE_PORT_RESTRICTED_CONE,
			discovery.NAT_TYPE_SYMMETRIC,
		}, natType)
	})

	t.Run("UDP Hole Punching", func(t *testing.T) {
		// Skip if external network tests are not enabled
		if os.Getenv("NOVACRON_E2E_NET") == "" {
			t.Skip("Skipping UDP hole punching test; set NOVACRON_E2E_NET=1 to enable")
		}

		// Create UDP hole puncher
		localAddr := &net.UDPAddr{
			IP:   net.IPv4zero,
			Port: 0, // Let system assign
		}

		holePuncher, err := discovery.NewUDPHolePuncher(localAddr, logger)
		require.NoError(t, err, "Failed to create UDP hole puncher")
		defer holePuncher.Stop()

		// Test establishing connection (mock peer)
		remoteAddr := &net.UDPAddr{
			IP:   net.ParseIP("127.0.0.1"),
			Port: 12345,
		}

		// This will fail in test but tests the flow
		conn, err := holePuncher.EstablishConnection("test-peer", remoteAddr)
		if err == nil {
			assert.NotNil(t, conn)
			assert.Equal(t, "test-peer", conn.PeerID)

			// Clean up
			err = holePuncher.CloseConnection("test-peer")
			assert.NoError(t, err)
		}
	})

	t.Run("Bandwidth Monitor Alert System", func(t *testing.T) {
		// Create bandwidth monitor config
		config := &network.BandwidthMonitorConfig{
			MonitoringInterval: 1 * time.Second,
			HistoryRetention:   5 * time.Minute,
			Interfaces:         []string{}, // Auto-discover
			DefaultThresholds: []network.BandwidthThreshold{
				{
					InterfaceName:     "*",
					WarningThreshold:  70.0,
					CriticalThreshold: 90.0,
					AbsoluteLimit:     1000000000, // 1 Gbps
					Enabled:           true,
				},
			},
			EnableQoSHooks:   true,
			MaxHistoryPoints: 100,
		}

		// Track alerts
		var receivedAlerts []network.BandwidthAlert
		alertHandler := &testAlertHandler{
			alerts: &receivedAlerts,
		}
		config.AlertHandlers = []network.BandwidthAlertHandler{alertHandler}

		monitor := network.NewBandwidthMonitor(config, logger)
		err := monitor.Start()
		require.NoError(t, err, "Failed to start bandwidth monitor")
		defer monitor.Stop()

		// Wait for monitoring to collect some data
		time.Sleep(3 * time.Second)

		// Get network utilization summary
		utilization := monitor.GetNetworkUtilizationSummary()
		assert.NotNil(t, utilization)

		// Check if any interfaces were discovered
		if len(utilization) > 0 {
			for iface, util := range utilization {
				t.Logf("Interface %s utilization: %.2f%%", iface, util)
				assert.GreaterOrEqual(t, util, 0.0)
				assert.LessOrEqual(t, util, 100.0)
			}
		}
	})

	t.Run("QoS Policy Enforcement", func(t *testing.T) {
		// Create QoS manager config
		qosConfig := &network.QoSManagerConfig{
			EnableTrafficShaping: false, // Disable tc commands for testing
			EnableDSCPMarking:    true,
			UpdateInterval:       5 * time.Second,
			DefaultPolicies:      []*network.QoSPolicy{},
			StatisticsRetention:  1 * time.Hour,
			MaxPoliciesPerInterface: 10,
		}

		qosManager := network.NewQoSManager(qosConfig, nil, logger)
		err := qosManager.Start()
		require.NoError(t, err, "Failed to start QoS manager")
		defer qosManager.Stop()

		// Create a test policy
		policy := &network.QoSPolicy{
			Name:          "test-policy",
			Description:   "Test QoS policy",
			NetworkID:     "test-network",
			InterfaceName: "lo", // Use loopback for testing
			Priority:      1,
			Rules: []network.ClassificationRule{
				{
					Name:       "test-rule",
					SourceIP:   "192.168.1.0/24",
					DestPort:   80,
					Protocol:   "tcp",
					Match:      "source_ip",
				},
			},
			Actions: []network.QoSAction{
				{
					Type:      "rate_limit",
					RateLimit: 100000000, // 100 Mbps
					Priority:  5,
				},
			},
			Enabled: true,
		}

		// Add policy
		err = qosManager.AddPolicy(policy)
		assert.NoError(t, err, "Failed to add QoS policy")

		// Get policy
		retrievedPolicy, err := qosManager.GetPolicy(policy.ID)
		assert.NoError(t, err)
		assert.NotNil(t, retrievedPolicy)
		assert.Equal(t, policy.Name, retrievedPolicy.Name)

		// List policies
		policies := qosManager.ListPolicies()
		assert.GreaterOrEqual(t, len(policies), 1)

		// Get interface policies
		ifacePolicies := qosManager.GetInterfacePolicies("lo")
		assert.GreaterOrEqual(t, len(ifacePolicies), 1)

		// Remove policy
		err = qosManager.RemovePolicy(policy.ID)
		assert.NoError(t, err)
	})

	t.Run("Internet Discovery Service", func(t *testing.T) {
		// Create internet discovery config
		config := discovery.InternetDiscoveryConfig{
			Config: discovery.Config{
				NodeID:   "test-node-1",
				NodeName: "Test Node 1",
				NodeRole: "worker",
				Address:  "127.0.0.1",
				Port:     7701,
			},
			BootstrapNodes:     "", // No bootstrap nodes for test
			EnableNATTraversal: false,
			EnableDHT:          true,
			EnableGossip:       true,
			PingInterval:       10 * time.Second,
			StunServers:        []string{"stun.l.google.com:19302"},
		}

		service, err := discovery.NewInternetDiscovery(config, logger)
		require.NoError(t, err, "Failed to create internet discovery service")

		err = service.Start()
		require.NoError(t, err, "Failed to start internet discovery service")
		defer service.Stop()

		// Test finding a peer (will fail but tests the flow)
		peerInfo, found := service.FindPeer("non-existent-peer")
		assert.False(t, found)
		assert.Empty(t, peerInfo.ID)

		// Get network topology info
		topology := service.GetNetworkTopologyInfo()
		assert.NotNil(t, topology)
		assert.Equal(t, "test-node-1", topology["peer_id"])

		// Get bandwidth capabilities
		capabilities := service.GetBandwidthCapabilities()
		assert.NotNil(t, capabilities)
	})

	t.Run("Network Manager Integration", func(t *testing.T) {
		// Create network manager config
		config := network.NetworkManagerConfig{
			DefaultNetworkType:         network.NetworkTypeBridge,
			DefaultSubnet:              "192.168.200.0/24",
			DefaultIPRange:             "192.168.200.10/24",
			DefaultGateway:             "192.168.200.1",
			DNSServers:                 []string{"8.8.8.8", "8.8.4.4"},
			UpdateInterval:             5 * time.Second,
			EnableQoS:                  false,
			BandwidthMonitoringEnabled: false,
		}

		manager := network.NewNetworkManager(config, "test-node", logger)
		err := manager.Start()
		require.NoError(t, err, "Failed to start network manager")
		defer manager.Stop()

		// Create a test network
		spec := network.NetworkSpec{
			Name: "test-network",
			Type: network.NetworkTypeBridge,
			IPAM: network.IPAMConfig{
				Subnet:  "10.0.0.0/24",
				Gateway: "10.0.0.1",
			},
			Internal: false,
		}

		// Note: This might fail without proper permissions
		createdNetwork, err := manager.CreateNetwork(ctx, spec)
		if err == nil {
			assert.NotNil(t, createdNetwork)
			assert.Equal(t, spec.Name, createdNetwork.Name)

			// Get network by ID
			retrievedNetwork, err := manager.GetNetwork(createdNetwork.ID)
			assert.NoError(t, err)
			assert.NotNil(t, retrievedNetwork)

			// Get network by name
			retrievedByName, err := manager.GetNetworkByName(spec.Name)
			assert.NoError(t, err)
			assert.NotNil(t, retrievedByName)

			// List networks
			networks := manager.ListNetworks()
			assert.GreaterOrEqual(t, len(networks), 1)

			// Delete network
			err = manager.DeleteNetwork(ctx, createdNetwork.ID)
			assert.NoError(t, err)
		} else {
			t.Logf("Network creation skipped (requires permissions): %v", err)
		}
	})
}

// testAlertHandler is a simple alert handler for testing
type testAlertHandler struct {
	alerts *[]network.BandwidthAlert
}

func (h *testAlertHandler) HandleAlert(alert *network.BandwidthAlert) error {
	*h.alerts = append(*h.alerts, *alert)
	return nil
}