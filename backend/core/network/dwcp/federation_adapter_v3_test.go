package dwcp

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/upgrade"
	"github.com/khryptorgraphics/novacron/backend/core/shared"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap/zaptest"
)

func TestEncodeConsensusLogReplication(t *testing.T) {
	logs := []shared.ConsensusLog{
		{
			Term:  7,
			Index: 42,
			Type:  shared.LogTypeCommand,
			Data:  []byte(`{"op":"migrate","vm":"vm-123"}`),
		},
		{
			Term:  7,
			Index: 43,
			Type:  shared.LogTypeConfig,
			Data:  []byte("config-change"),
		},
	}

	data, err := encodeConsensusLogReplication(logs)
	require.NoError(t, err)
	require.NotEmpty(t, data)

	var payload consensusLogReplicationPayload
	require.NoError(t, json.Unmarshal(data, &payload))
	assert.Equal(t, "consensus_logs", payload.Type)
	assert.Equal(t, consensusLogReplicationPayloadVersion, payload.Version)
	require.Len(t, payload.Logs, len(logs))
	assert.Equal(t, logs, payload.Logs)
}

func TestFederationAdapterV3ReplicateLogsRoutesPayload(t *testing.T) {
	adapter, err := NewFederationAdapterV3(zaptest.NewLogger(t), &FederationAdapterConfig{
		NodeID:             "node-1",
		DefaultMode:        upgrade.ModeHybrid,
		EnableAdaptiveMode: false,
	})
	require.NoError(t, err)
	defer adapter.Close()

	require.NoError(t, adapter.RegisterCluster(&ClusterConnectionV3{
		ClusterID: "cluster-1",
		Region:    "region-1",
		Connected: true,
		Healthy:   true,
	}))

	logs := []shared.ConsensusLog{
		{
			Term:  1,
			Index: 1,
			Type:  shared.LogTypeNoOp,
			Data:  []byte("heartbeat"),
		},
	}

	err = adapter.ReplicateLogs(context.Background(), logs, []string{"cluster-1"})
	require.NoError(t, err)
	assert.Equal(t, uint64(1), adapter.GetMetrics().TotalRoutes)
	assert.Equal(t, uint64(1), adapter.GetMetrics().HybridRoutes)
}
