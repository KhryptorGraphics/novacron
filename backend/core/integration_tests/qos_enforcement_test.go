package integration_tests

import (
	"fmt"
	"testing"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/network"
	"go.uber.org/zap"
)

func TestQoSPolicyEnforcement(t *testing.T) {
	logger := zap.NewNop()
	
	// Create bandwidth monitor
	bmConfig := &network.BandwidthMonitorConfig{
		MonitoringInterval: 1 * time.Second,
		HistoryRetention:   10 * time.Minute,
		Interfaces:         []string{"eth0"},
		EnableQoSHooks:     true,
	}
	bandwidthMonitor := network.NewBandwidthMonitor(bmConfig, logger)
	
	// Create QoS manager
	qosConfig := &network.QoSManagerConfig{
		EnableTrafficShaping: true,
		EnableDSCPMarking:    true,
		UpdateInterval:       5 * time.Second,
		MaxPoliciesPerInterface: 10,
	}
	qosManager := network.NewQoSManager(qosConfig, bandwidthMonitor, logger)
	
	// Start services
	err := bandwidthMonitor.Start()
	if err != nil {
		t.Fatalf("Failed to start bandwidth monitor: %v", err)
	}
	defer bandwidthMonitor.Stop()
	
	err = qosManager.Start()
	if err != nil {
		t.Fatalf("Failed to start QoS manager: %v", err)
	}
	defer qosManager.Stop()
	
	// Test policy creation and enforcement
	t.Run("RateLimitPolicy", func(t *testing.T) {
		policy := &network.QoSPolicy{
			Name:          "test-rate-limit",
			Description:   "Test rate limiting policy",
			InterfaceName: "eth0",
			Priority:      1,
			Rules: []network.ClassificationRule{
				{
					Name:     "ssh-traffic",
					DestPort: 22,
					Protocol: "tcp",
				},
			},
			Actions: []network.QoSAction{
				{
					Type:       "rate_limit",
					RateLimit:  100000000, // 100 Mbps
					BurstLimit: 10000000,  // 10 MB burst
					Priority:   5,
				},
			},
			Enabled: true,
		}
		
		err := qosManager.AddPolicy(policy)
		if err != nil {
			// This might fail if tc commands aren't available
			t.Logf("Warning: Failed to add policy (may require root): %v", err)
			return
		}
		
		// Verify policy was added
		retrieved, err := qosManager.GetPolicy(policy.ID)
		if err != nil {
			t.Errorf("Failed to retrieve policy: %v", err)
		}
		
		if retrieved.Name != policy.Name {
			t.Errorf("Policy name mismatch: expected %s, got %s", 
				policy.Name, retrieved.Name)
		}
		
		// Check that tc class was created
		if len(qosManager.AppliedClasses) == 0 {
			t.Error("No traffic classes were applied")
		}
		
		// Test policy removal
		err = qosManager.RemovePolicy(policy.ID)
		if err != nil {
			t.Errorf("Failed to remove policy: %v", err)
		}
		
		_, err = qosManager.GetPolicy(policy.ID)
		if err == nil {
			t.Error("Policy still exists after removal")
		}
	})

	t.Run("DSCPMarkingPolicy", func(t *testing.T) {
		policy := &network.QoSPolicy{
			Name:          "test-dscp",
			Description:   "Test DSCP marking policy",
			InterfaceName: "eth0",
			Priority:      2,
			Rules: []network.ClassificationRule{
				{
					Name:     "voip-traffic",
					DestPort: 5060,
					Protocol: "udp",
				},
			},
			Actions: []network.QoSAction{
				{
					Type:     "dscp_mark",
					DSCPMark: 46, // EF (Expedited Forwarding)
					Priority: 1,
				},
			},
			Enabled: true,
		}
		
		err := qosManager.AddPolicy(policy)
		if err != nil {
			t.Logf("Warning: Failed to add DSCP policy: %v", err)
			return
		}
		
		// Verify DSCP marking would be applied
		policies := qosManager.GetInterfacePolicies("eth0")
		found := false
		for _, p := range policies {
			if p.Name == "test-dscp" {
				found = true
				for _, action := range p.Actions {
					if action.Type == "dscp_mark" && action.DSCPMark == 46 {
						t.Log("DSCP marking policy configured correctly")
					}
				}
			}
		}
		
		if !found {
			t.Error("DSCP policy not found in interface policies")
		}
	})

	t.Run("BandwidthAlertResponse", func(t *testing.T) {
		// Create a policy with rate limiting
		policy := &network.QoSPolicy{
			Name:          "test-congestion",
			Description:   "Test congestion response",
			InterfaceName: "eth0",
			NetworkID:     "test-network",
			Priority:      3,
			Actions: []network.QoSAction{
				{
					Type:       "rate_limit",
					RateLimit:  200000000, // 200 Mbps
					BurstLimit: 20000000,  // 20 MB
				},
			},
			Enabled: true,
		}
		
		err := qosManager.AddPolicy(policy)
		if err != nil {
			t.Logf("Warning: Failed to add congestion policy: %v", err)
			return
		}
		
		// Simulate high bandwidth utilization alert
		qosManager.HandleBandwidthAlert("eth0", 85.0)
		
		// Check that rate limit was reduced
		updatedPolicy, err := qosManager.GetPolicy(policy.ID)
		if err != nil {
			t.Errorf("Failed to get updated policy: %v", err)
			return
		}
		
		for _, action := range updatedPolicy.Actions {
			if action.Type == "rate_limit" {
				expectedRate := uint64(200000000 * 0.8) // Should be reduced by 20%
				if action.RateLimit != expectedRate {
					t.Errorf("Rate limit not adjusted correctly: expected %d, got %d",
						expectedRate, action.RateLimit)
				} else {
					t.Log("Rate limit correctly reduced in response to congestion")
				}
			}
		}
	})

	t.Run("PolicyStatistics", func(t *testing.T) {
		// Wait for statistics update
		time.Sleep(6 * time.Second)
		
		policies := qosManager.ListPolicies()
		for _, policy := range policies {
			if policy.Statistics.LastUpdated.IsZero() {
				t.Errorf("Policy %s has no statistics update", policy.Name)
			}
			t.Logf("Policy %s stats: Throughput=%d bps, Utilization=%.2f%%",
				policy.Name, policy.Statistics.ThroughputBps, 
				policy.Statistics.UtilizationPercent)
		}
	})
}

func TestQoSReconciliation(t *testing.T) {
	logger := zap.NewNop()

	qosConfig := &network.QoSManagerConfig{
		EnableTrafficShaping: true,
		UpdateInterval:       1 * time.Second,
	}
	qosManager := network.NewQoSManager(qosConfig, nil, logger)

	err := qosManager.Start()
	if err != nil {
		t.Fatalf("Failed to start QoS manager: %v", err)
	}
	defer qosManager.Stop()

	// Add a policy
	policy := &network.QoSPolicy{
		Name:          "test-reconcile",
		InterfaceName: "eth0",
		Actions: []network.QoSAction{
			{
				Type:      "rate_limit",
				RateLimit: 50000000, // 50 Mbps
			},
		},
		Enabled: true,
	}

	err = qosManager.AddPolicy(policy)
	if err != nil {
		t.Logf("Warning: Failed to add policy for reconciliation test: %v", err)
		return
	}

	// Simulate applied classes being cleared (as if tc was reset)
	qosManager.AppliedClasses = make(map[string]string)

	// Trigger reconciliation
	qosManager.ReconcileState()

	// Check that policy was re-applied
	if len(qosManager.AppliedClasses) == 0 {
		t.Error("Policy was not re-applied during reconciliation")
	} else {
		t.Log("Policy successfully re-applied during reconciliation")
	}
}

func TestNetworkQoSStatus(t *testing.T) {
	logger := zap.NewNop()

	qosManager := network.NewQoSManager(&network.QoSManagerConfig{}, nil, logger)

	// Add multiple policies for a network
	networkID := "test-network-123"
	for i := 0; i < 3; i++ {
		policy := &network.QoSPolicy{
			Name:      fmt.Sprintf("policy-%d", i),
			NetworkID: networkID,
			Enabled:   i%2 == 0, // Alternate enabled/disabled
		}
		qosManager.AddPolicy(policy)
	}

	// Get network status
	status := qosManager.GetNetworkQoSStatus(networkID)

	totalPolicies, ok := status["total_policies"].(int)
	if !ok || totalPolicies != 3 {
		t.Errorf("Expected 3 total policies, got %v", status["total_policies"])
	}

	activePolicies, ok := status["active_policies"].(int)
	if !ok || activePolicies != 2 {
		t.Errorf("Expected 2 active policies, got %v", status["active_policies"])
	}

	t.Logf("Network QoS status: total=%d, active=%d", 
		totalPolicies, activePolicies)
}