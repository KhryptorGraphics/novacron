package consensus

import (
	"context"
	"testing"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/upgrade"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
)

// MockRaftNode implements RaftNode for testing
type MockRaftNode struct {
	proposeDelay time.Duration
	isLeader     bool
}

func (mrn *MockRaftNode) Propose(ctx context.Context, data []byte) error {
	if mrn.proposeDelay > 0 {
		select {
		case <-time.After(mrn.proposeDelay):
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	return nil
}

func (mrn *MockRaftNode) ReadIndex(ctx context.Context) (uint64, error) {
	return 0, nil
}

func (mrn *MockRaftNode) IsLeader() bool {
	return mrn.isLeader
}

func TestACPv3_Creation(t *testing.T) {
	logger := zap.NewNop()

	tests := []struct {
		name   string
		mode   upgrade.NetworkMode
		config *ACPConfig
	}{
		{
			name: "Datacenter mode",
			mode: upgrade.ModeDatacenter,
			config: &ACPConfig{
				GossipPeers: []string{"peer1", "peer2"},
			},
		},
		{
			name: "Internet mode with PBFT",
			mode: upgrade.ModeInternet,
			config: &ACPConfig{
				PBFTConfig: &PBFTConfig{
					ReplicaCount: 4,
					Transport:    NewMockTransport(),
					StateMachine: NewMockStateMachine(),
				},
				GossipPeers: []string{"peer1", "peer2"},
			},
		},
		{
			name: "Hybrid mode",
			mode: upgrade.ModeHybrid,
			config: &ACPConfig{
				PBFTConfig: &PBFTConfig{
					ReplicaCount: 7,
					Transport:    NewMockTransport(),
					StateMachine: NewMockStateMachine(),
				},
				GossipPeers: []string{"peer1", "peer2", "peer3"},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			acp, err := NewACPv3("node_0", tt.mode, tt.config, logger)
			require.NoError(t, err)
			assert.NotNil(t, acp)
			assert.Equal(t, tt.mode, acp.mode)
			assert.NotNil(t, acp.raft)
			assert.NotNil(t, acp.gossip)
		})
	}
}

func TestACPv3_DatacenterConsensus(t *testing.T) {
	logger := zap.NewNop()

	config := &ACPConfig{
		GossipPeers: []string{"peer1"},
	}

	acp, err := NewACPv3("node_0", upgrade.ModeDatacenter, config, logger)
	require.NoError(t, err)

	// Set mock Raft node with fast response
	mockRaft := &MockRaftNode{
		proposeDelay: 10 * time.Millisecond,
		isLeader:     true,
	}
	acp.SetRaftNode(mockRaft)

	ctx := context.Background()
	value := map[string]interface{}{"key": "test", "value": "data"}

	start := time.Now()
	err = acp.Consensus(ctx, value)
	latency := time.Since(start)

	assert.NoError(t, err)
	// Should complete in <100ms for datacenter mode
	assert.Less(t, latency, 100*time.Millisecond)
}

func TestACPv3_InternetConsensus(t *testing.T) {
	logger := zap.NewNop()

	config := &ACPConfig{
		PBFTConfig: &PBFTConfig{
			ReplicaCount: 4,
			Transport:    NewMockTransport(),
			StateMachine: NewMockStateMachine(),
		},
		GossipPeers: []string{"peer1"},
	}

	acp, err := NewACPv3("node_0", upgrade.ModeInternet, config, logger)
	require.NoError(t, err)

	err = acp.Start()
	require.NoError(t, err)
	defer acp.Stop()

	ctx := context.Background()
	value := map[string]interface{}{"key": "test", "value": "data"}

	err = acp.Consensus(ctx, value)
	assert.NoError(t, err)
}

func TestACPv3_HybridConsensusFailover(t *testing.T) {
	logger := zap.NewNop()

	config := &ACPConfig{
		PBFTConfig: &PBFTConfig{
			ReplicaCount: 4,
			Transport:    NewMockTransport(),
			StateMachine: NewMockStateMachine(),
		},
		GossipPeers: []string{"peer1"},
	}

	acp, err := NewACPv3("node_0", upgrade.ModeHybrid, config, logger)
	require.NoError(t, err)

	// Set Raft node that times out (simulating network issues)
	slowRaft := &MockRaftNode{
		proposeDelay: 200 * time.Millisecond, // Exceeds 100ms datacenter target
		isLeader:     true,
	}
	acp.SetRaftNode(slowRaft)

	err = acp.Start()
	require.NoError(t, err)
	defer acp.Stop()

	ctx := context.Background()
	value := map[string]interface{}{"key": "test", "value": "data"}

	// Should failover to PBFT when Raft times out
	err = acp.Consensus(ctx, value)
	assert.NoError(t, err)

	// Verify failover occurred
	metrics := acp.GetMetrics()
	assert.Greater(t, metrics.FailoverCount, int64(0))
}

func TestACPv3_ModeAdaptation(t *testing.T) {
	logger := zap.NewNop()

	config := &ACPConfig{
		PBFTConfig: &PBFTConfig{
			ReplicaCount: 4,
			Transport:    NewMockTransport(),
			StateMachine: NewMockStateMachine(),
		},
		GossipPeers: []string{"peer1"},
	}

	acp, err := NewACPv3("node_0", upgrade.ModeHybrid, config, logger)
	require.NoError(t, err)

	// Initially hybrid mode
	assert.Equal(t, upgrade.ModeHybrid, acp.GetMode())

	// Switch to datacenter mode
	acp.SetMode(upgrade.ModeDatacenter)
	assert.Equal(t, upgrade.ModeDatacenter, acp.GetMode())

	// Switch to internet mode
	acp.SetMode(upgrade.ModeInternet)
	assert.Equal(t, upgrade.ModeInternet, acp.GetMode())
}

func TestACPv3_ConsensusLatency(t *testing.T) {
	logger := zap.NewNop()

	tests := []struct {
		name            string
		mode            upgrade.NetworkMode
		expectedLatency time.Duration
		tolerance       time.Duration
	}{
		{
			name:            "Datacenter mode",
			mode:            upgrade.ModeDatacenter,
			expectedLatency: 100 * time.Millisecond,
			tolerance:       50 * time.Millisecond,
		},
		{
			name:            "Internet mode",
			mode:            upgrade.ModeInternet,
			expectedLatency: 2 * time.Second,
			tolerance:       1 * time.Second,
		},
		{
			name:            "Hybrid mode",
			mode:            upgrade.ModeHybrid,
			expectedLatency: 500 * time.Millisecond,
			tolerance:       250 * time.Millisecond,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := &ACPConfig{
				PBFTConfig: &PBFTConfig{
					ReplicaCount: 4,
					Transport:    NewMockTransport(),
					StateMachine: NewMockStateMachine(),
				},
				GossipPeers: []string{"peer1"},
			}

			acp, err := NewACPv3("node_0", tt.mode, config, logger)
			require.NoError(t, err)

			latency := acp.GetConsensusLatency()

			// Verify latency is within expected range
			assert.GreaterOrEqual(t, latency, tt.expectedLatency-tt.tolerance)
			assert.LessOrEqual(t, latency, tt.expectedLatency+tt.tolerance)
		})
	}
}

func TestACPv3_HealthCheck(t *testing.T) {
	logger := zap.NewNop()

	config := &ACPConfig{
		PBFTConfig: &PBFTConfig{
			ReplicaCount: 4,
			Transport:    NewMockTransport(),
			StateMachine: NewMockStateMachine(),
		},
		GossipPeers: []string{"peer1"},
	}

	tests := []struct {
		name           string
		mode           upgrade.NetworkMode
		setupRaft      bool
		raftIsLeader   bool
		expectedHealth bool
	}{
		{
			name:           "Datacenter healthy (Raft leader)",
			mode:           upgrade.ModeDatacenter,
			setupRaft:      true,
			raftIsLeader:   true,
			expectedHealth: true,
		},
		{
			name:           "Datacenter unhealthy (not leader)",
			mode:           upgrade.ModeDatacenter,
			setupRaft:      true,
			raftIsLeader:   false,
			expectedHealth: false,
		},
		{
			name:           "Internet healthy (PBFT available)",
			mode:           upgrade.ModeInternet,
			setupRaft:      false,
			raftIsLeader:   false,
			expectedHealth: true,
		},
		{
			name:           "Hybrid healthy (PBFT available)",
			mode:           upgrade.ModeHybrid,
			setupRaft:      false,
			raftIsLeader:   false,
			expectedHealth: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			acp, err := NewACPv3("node_0", tt.mode, config, logger)
			require.NoError(t, err)

			if tt.setupRaft {
				mockRaft := &MockRaftNode{
					isLeader: tt.raftIsLeader,
				}
				acp.SetRaftNode(mockRaft)
			}

			healthy := acp.IsHealthy()
			assert.Equal(t, tt.expectedHealth, healthy)
		})
	}
}

func TestACPv3_Metrics(t *testing.T) {
	logger := zap.NewNop()

	config := &ACPConfig{
		PBFTConfig: &PBFTConfig{
			ReplicaCount: 4,
			Transport:    NewMockTransport(),
			StateMachine: NewMockStateMachine(),
		},
		GossipPeers: []string{"peer1", "peer2"},
	}

	acp, err := NewACPv3("node_0", upgrade.ModeInternet, config, logger)
	require.NoError(t, err)

	// Perform some consensus operations
	ctx := context.Background()
	for i := 0; i < 5; i++ {
		value := map[string]interface{}{"key": "test", "value": i}
		_ = acp.Consensus(ctx, value)
	}

	metrics := acp.GetMetrics()

	assert.NotNil(t, metrics)
	assert.Equal(t, upgrade.ModeInternet.String(), metrics.Mode)
	assert.GreaterOrEqual(t, metrics.ConsensusCount, int64(5))
	assert.NotNil(t, metrics.PBFTMetrics)
}

func BenchmarkACPv3_DatacenterConsensus(b *testing.B) {
	logger := zap.NewNop()

	config := &ACPConfig{
		GossipPeers: []string{"peer1"},
	}

	acp, err := NewACPv3("node_0", upgrade.ModeDatacenter, config, logger)
	require.NoError(b, err)

	mockRaft := &MockRaftNode{
		proposeDelay: 5 * time.Millisecond,
		isLeader:     true,
	}
	acp.SetRaftNode(mockRaft)

	ctx := context.Background()
	value := map[string]interface{}{"key": "bench", "value": "data"}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = acp.Consensus(ctx, value)
	}
}

func BenchmarkACPv3_InternetConsensus(b *testing.B) {
	logger := zap.NewNop()

	config := &ACPConfig{
		PBFTConfig: &PBFTConfig{
			ReplicaCount: 4,
			Transport:    NewMockTransport(),
			StateMachine: NewMockStateMachine(),
		},
		GossipPeers: []string{"peer1"},
	}

	acp, err := NewACPv3("node_0", upgrade.ModeInternet, config, logger)
	require.NoError(b, err)

	_ = acp.Start()
	defer acp.Stop()

	ctx := context.Background()
	value := map[string]interface{}{"key": "bench", "value": "data"}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = acp.Consensus(ctx, value)
	}
}
