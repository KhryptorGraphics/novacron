package sync

import (
	"context"
	"testing"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/upgrade"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
)

// MockRaftNode for testing Raft integration
type MockRaftNode struct {
	proposeDelay time.Duration
	isLeader     bool
	proposals    [][]byte
}

func (mrn *MockRaftNode) Propose(ctx context.Context, data []byte) error {
	mrn.proposals = append(mrn.proposals, data)
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

func TestASSv3_Creation(t *testing.T) {
	logger := zap.NewNop()

	tests := []struct {
		name   string
		nodeID string
		mode   upgrade.NetworkMode
	}{
		{
			name:   "Datacenter mode",
			nodeID: "node_0",
			mode:   upgrade.ModeDatacenter,
		},
		{
			name:   "Internet mode",
			nodeID: "node_1",
			mode:   upgrade.ModeInternet,
		},
		{
			name:   "Hybrid mode",
			nodeID: "node_2",
			mode:   upgrade.ModeHybrid,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ass, err := NewASSv3(tt.nodeID, tt.mode, logger)
			require.NoError(t, err)
			assert.NotNil(t, ass)
			assert.Equal(t, tt.nodeID, ass.nodeID)
			assert.Equal(t, tt.mode, ass.mode)
			assert.NotNil(t, ass.raftSync)
			assert.NotNil(t, ass.crdtSync)
			assert.NotNil(t, ass.conflictResolver)
		})
	}
}

func TestASSv3_DatacenterSync(t *testing.T) {
	logger := zap.NewNop()

	ass, err := NewASSv3("node_0", upgrade.ModeDatacenter, logger)
	require.NoError(t, err)

	// Setup mock Raft node
	mockRaft := &MockRaftNode{
		proposeDelay: 10 * time.Millisecond,
		isLeader:     true,
		proposals:    make([][]byte, 0),
	}
	ass.SetRaftNode(mockRaft)

	ctx := context.Background()
	state := map[string]interface{}{
		"key1": "value1",
		"key2": "value2",
	}

	start := time.Now()
	err = ass.SyncState(ctx, state)
	latency := time.Since(start)

	assert.NoError(t, err)
	// Should complete in <100ms for datacenter mode
	assert.Less(t, latency, 100*time.Millisecond)
	assert.Equal(t, 1, len(mockRaft.proposals))
}

func TestASSv3_InternetSync(t *testing.T) {
	logger := zap.NewNop()

	ass, err := NewASSv3("node_0", upgrade.ModeInternet, logger)
	require.NoError(t, err)

	ctx := context.Background()
	state := map[string]interface{}{
		"key1": "value1",
		"key2": 42,
	}

	err = ass.SyncState(ctx, state)
	assert.NoError(t, err)

	// Verify CRDT was updated
	metrics := ass.GetMetrics()
	assert.Equal(t, int64(1), metrics.SyncCount)
}

func TestASSv3_HybridSync(t *testing.T) {
	logger := zap.NewNop()

	ass, err := NewASSv3("node_0", upgrade.ModeHybrid, logger)
	require.NoError(t, err)

	// Setup slow Raft to trigger fallback
	slowRaft := &MockRaftNode{
		proposeDelay: 200 * time.Millisecond, // Exceeds datacenter target
		isLeader:     true,
		proposals:    make([][]byte, 0),
	}
	ass.SetRaftNode(slowRaft)

	ctx := context.Background()
	state := map[string]interface{}{
		"key": "value",
	}

	err = ass.SyncState(ctx, state)
	assert.NoError(t, err)

	// Should have fallen back to CRDT
	metrics := ass.GetMetrics()
	assert.Greater(t, metrics.ConflictCount, 0)
}

func TestASSv3_ModeAdaptation(t *testing.T) {
	logger := zap.NewNop()

	ass, err := NewASSv3("node_0", upgrade.ModeHybrid, logger)
	require.NoError(t, err)

	err = ass.Start()
	require.NoError(t, err)
	defer ass.Stop()

	// Initially hybrid
	assert.Equal(t, upgrade.ModeHybrid, ass.GetMode())

	// Switch to datacenter
	ass.SetMode(upgrade.ModeDatacenter)
	assert.Equal(t, upgrade.ModeDatacenter, ass.GetMode())

	// Switch to internet
	ass.SetMode(upgrade.ModeInternet)
	assert.Equal(t, upgrade.ModeInternet, ass.GetMode())
}

func TestASSv3_ConcurrentSync(t *testing.T) {
	logger := zap.NewNop()

	ass, err := NewASSv3("node_0", upgrade.ModeInternet, logger)
	require.NoError(t, err)

	ctx := context.Background()
	concurrency := 10

	errChan := make(chan error, concurrency)

	for i := 0; i < concurrency; i++ {
		go func(id int) {
			state := map[string]interface{}{
				"key": id,
			}
			errChan <- ass.SyncState(ctx, state)
		}(i)
	}

	// Collect errors
	for i := 0; i < concurrency; i++ {
		err := <-errChan
		assert.NoError(t, err)
	}

	metrics := ass.GetMetrics()
	assert.Equal(t, int64(concurrency), metrics.SyncCount)
}

func TestASSv3_Metrics(t *testing.T) {
	logger := zap.NewNop()

	ass, err := NewASSv3("node_0", upgrade.ModeDatacenter, logger)
	require.NoError(t, err)

	mockRaft := &MockRaftNode{
		proposeDelay: 5 * time.Millisecond,
		isLeader:     true,
	}
	ass.SetRaftNode(mockRaft)

	ctx := context.Background()

	// Perform multiple syncs
	for i := 0; i < 5; i++ {
		state := map[string]interface{}{"iteration": i}
		_ = ass.SyncState(ctx, state)
	}

	metrics := ass.GetMetrics()

	assert.NotNil(t, metrics)
	assert.Equal(t, upgrade.ModeDatacenter.String(), metrics.Mode)
	assert.Equal(t, int64(5), metrics.SyncCount)
	assert.Greater(t, metrics.AvgSyncLatency, time.Duration(0))
}

func TestASSv3_SyncLatencyTracking(t *testing.T) {
	logger := zap.NewNop()

	ass, err := NewASSv3("node_0", upgrade.ModeInternet, logger)
	require.NoError(t, err)

	ctx := context.Background()

	// Perform several syncs
	for i := 0; i < 10; i++ {
		state := map[string]interface{}{"key": i}
		_ = ass.SyncState(ctx, state)
		time.Sleep(10 * time.Millisecond)
	}

	metrics := ass.GetMetrics()

	// Verify latency tracking
	assert.Greater(t, metrics.AvgSyncLatency, time.Duration(0))
	assert.NotEqual(t, metrics.LastSyncTime, time.Time{})
}

func TestRaftStateSync_WithoutRaftNode(t *testing.T) {
	logger := zap.NewNop()

	ass, err := NewASSv3("node_0", upgrade.ModeDatacenter, logger)
	require.NoError(t, err)

	ctx := context.Background()
	state := map[string]interface{}{"key": "value"}

	// Should fail without Raft node
	err = ass.SyncState(ctx, state)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "Raft node not initialized")
}

func TestCRDTStateSync_VectorClock(t *testing.T) {
	logger := zap.NewNop()

	ass, err := NewASSv3("node_0", upgrade.ModeInternet, logger)
	require.NoError(t, err)

	ctx := context.Background()

	// Perform multiple syncs
	for i := 0; i < 5; i++ {
		state := map[string]interface{}{"iteration": i}
		err := ass.SyncState(ctx, state)
		assert.NoError(t, err)
	}

	// Verify vector clock incremented
	ass.crdtSync.mu.RLock()
	clockValue := ass.crdtSync.vectorClock[ass.nodeID]
	ass.crdtSync.mu.RUnlock()

	assert.Equal(t, uint64(5), clockValue)
}

func TestConflictResolver_Recording(t *testing.T) {
	logger := zap.NewNop()
	resolver := NewConflictResolver(ResolveLastWriteWins, logger)

	// Record some conflicts
	for i := 0; i < 5; i++ {
		event := ConflictEvent{
			Timestamp: time.Now(),
			Key:       "test_key",
			Strategy:  ResolveLastWriteWins,
			Resolved:  true,
			Latency:   time.Millisecond * time.Duration(i),
		}
		resolver.RecordConflict(event)
	}

	assert.Equal(t, 5, len(resolver.history))
}

func TestConflictResolver_HistoryLimit(t *testing.T) {
	logger := zap.NewNop()
	resolver := NewConflictResolver(ResolveLastWriteWins, logger)

	// Record more than limit
	for i := 0; i < 1500; i++ {
		event := ConflictEvent{
			Timestamp: time.Now(),
			Key:       "test_key",
			Strategy:  ResolveLastWriteWins,
			Resolved:  true,
			Latency:   time.Millisecond,
		}
		resolver.RecordConflict(event)
	}

	// Should be limited to 1000
	assert.Equal(t, 1000, len(resolver.history))
}

func BenchmarkASSv3_DatacenterSync(b *testing.B) {
	logger := zap.NewNop()

	ass, err := NewASSv3("node_0", upgrade.ModeDatacenter, logger)
	require.NoError(b, err)

	mockRaft := &MockRaftNode{
		proposeDelay: 5 * time.Millisecond,
		isLeader:     true,
	}
	ass.SetRaftNode(mockRaft)

	ctx := context.Background()
	state := map[string]interface{}{"key": "value"}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = ass.SyncState(ctx, state)
	}
}

func BenchmarkASSv3_InternetSync(b *testing.B) {
	logger := zap.NewNop()

	ass, err := NewASSv3("node_0", upgrade.ModeInternet, logger)
	require.NoError(b, err)

	ctx := context.Background()
	state := map[string]interface{}{"key": "value"}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = ass.SyncState(ctx, state)
	}
}

func BenchmarkASSv3_HybridSync(b *testing.B) {
	logger := zap.NewNop()

	ass, err := NewASSv3("node_0", upgrade.ModeHybrid, logger)
	require.NoError(b, err)

	mockRaft := &MockRaftNode{
		proposeDelay: 10 * time.Millisecond,
		isLeader:     true,
	}
	ass.SetRaftNode(mockRaft)

	ctx := context.Background()
	state := map[string]interface{}{"key": "value"}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = ass.SyncState(ctx, state)
	}
}
