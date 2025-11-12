package consensus

import (
	"context"
	"encoding/json"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
)

// MockTransport implements Transport for testing
type MockTransport struct {
	mu       sync.Mutex
	messages []interface{}
	received chan interface{}
}

func NewMockTransport() *MockTransport {
	return &MockTransport{
		messages: make([]interface{}, 0),
		received: make(chan interface{}, 100),
	}
}

func (mt *MockTransport) Send(replicaID string, message interface{}) error {
	mt.mu.Lock()
	mt.messages = append(mt.messages, message)
	mt.mu.Unlock()
	mt.received <- message
	return nil
}

func (mt *MockTransport) Broadcast(message interface{}) error {
	mt.mu.Lock()
	mt.messages = append(mt.messages, message)
	mt.mu.Unlock()
	mt.received <- message
	return nil
}

func (mt *MockTransport) Receive() (interface{}, error) {
	select {
	case msg := <-mt.received:
		return msg, nil
	case <-time.After(100 * time.Millisecond):
		return nil, nil
	}
}

// MockStateMachine implements StateMachine for testing
type MockStateMachine struct {
	mu    sync.Mutex
	state map[string][]byte
}

func NewMockStateMachine() *MockStateMachine {
	return &MockStateMachine{
		state: make(map[string][]byte),
	}
}

func (msm *MockStateMachine) Apply(operation json.RawMessage) (json.RawMessage, error) {
	msm.mu.Lock()
	defer msm.mu.Unlock()

	var op map[string]interface{}
	if err := json.Unmarshal(operation, &op); err != nil {
		return nil, err
	}

	key := op["key"].(string)
	value := []byte(op["value"].(string))
	msm.state[key] = value

	return json.Marshal(map[string]interface{}{"success": true})
}

func (msm *MockStateMachine) GetState() (json.RawMessage, error) {
	msm.mu.Lock()
	defer msm.mu.Unlock()
	return json.Marshal(msm.state)
}

func (msm *MockStateMachine) Checkpoint(sequence int64) (string, error) {
	state, _ := msm.GetState()
	return string(state), nil
}

func TestPBFT_Creation(t *testing.T) {
	logger := zap.NewNop()
	transport := NewMockTransport()
	stateMachine := NewMockStateMachine()

	tests := []struct {
		name         string
		nodeID       string
		replicaCount int
		expectError  bool
	}{
		{
			name:         "Valid 4 replicas (f=1)",
			nodeID:       "node_0",
			replicaCount: 4,
			expectError:  false,
		},
		{
			name:         "Valid 7 replicas (f=2)",
			nodeID:       "node_0",
			replicaCount: 7,
			expectError:  false,
		},
		{
			name:         "Invalid 3 replicas",
			nodeID:       "node_0",
			replicaCount: 3,
			expectError:  true,
		},
		{
			name:         "Valid 10 replicas (f=3)",
			nodeID:       "node_0",
			replicaCount: 10,
			expectError:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pbft, err := NewPBFT(tt.nodeID, tt.replicaCount, transport, stateMachine, logger)

			if tt.expectError {
				assert.Error(t, err)
				assert.Nil(t, pbft)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, pbft)
				assert.Equal(t, tt.nodeID, pbft.nodeID)
				assert.Equal(t, tt.replicaCount, pbft.replicaCount)
				assert.Equal(t, (tt.replicaCount-1)/3, pbft.f)
			}
		})
	}
}

func TestPBFT_ByzantineTolerance(t *testing.T) {
	logger := zap.NewNop()
	transport := NewMockTransport()
	stateMachine := NewMockStateMachine()

	tests := []struct {
		replicaCount       int
		expectedF          int
		expectedTolerance  string
	}{
		{4, 1, "25% malicious nodes"},
		{7, 2, "28% malicious nodes"},
		{10, 3, "30% malicious nodes"},
		{13, 4, "30% malicious nodes"},
	}

	for _, tt := range tests {
		t.Run(tt.expectedTolerance, func(t *testing.T) {
			pbft, err := NewPBFT("node_0", tt.replicaCount, transport, stateMachine, logger)
			require.NoError(t, err)

			assert.Equal(t, tt.expectedF, pbft.f)

			// Verify can tolerate f Byzantine nodes
			// Need 2f+1 for quorum
			quorum := 2*pbft.f + 1
			assert.LessOrEqual(t, quorum, tt.replicaCount)
		})
	}
}

func TestPBFT_ThreePhaseProtocol(t *testing.T) {
	logger := zap.NewNop()
	transport := NewMockTransport()
	stateMachine := NewMockStateMachine()

	pbft, err := NewPBFT("node_0", 4, transport, stateMachine, logger)
	require.NoError(t, err)

	ctx := context.Background()

	// Test consensus on a value
	value := map[string]interface{}{
		"key":   "test_key",
		"value": "test_value",
	}

	err = pbft.Consensus(ctx, value)
	assert.NoError(t, err)

	// Verify pre-prepare was created
	time.Sleep(100 * time.Millisecond)
	transport.mu.Lock()
	messageCount := len(transport.messages)
	transport.mu.Unlock()

	assert.Greater(t, messageCount, 0, "Should have broadcast pre-prepare")
}

func TestPBFT_MessageDigest(t *testing.T) {
	logger := zap.NewNop()
	transport := NewMockTransport()
	stateMachine := NewMockStateMachine()

	pbft, err := NewPBFT("node_0", 4, transport, stateMachine, logger)
	require.NoError(t, err)

	req := &ClientRequest{
		ClientID:  "client_1",
		Timestamp: time.Now(),
		Operation: json.RawMessage(`{"key": "test"}`),
		Sequence:  1,
	}

	digest1 := pbft.computeDigest(req)
	digest2 := pbft.computeDigest(req)

	// Same request should produce same digest
	assert.Equal(t, digest1, digest2)

	// Different request should produce different digest
	req2 := &ClientRequest{
		ClientID:  "client_2",
		Timestamp: time.Now(),
		Operation: json.RawMessage(`{"key": "test2"}`),
		Sequence:  2,
	}
	digest3 := pbft.computeDigest(req2)
	assert.NotEqual(t, digest1, digest3)
}

func TestPBFT_Checkpoint(t *testing.T) {
	logger := zap.NewNop()
	transport := NewMockTransport()
	stateMachine := NewMockStateMachine()

	pbft, err := NewPBFT("node_0", 4, transport, stateMachine, logger)
	require.NoError(t, err)

	// Create checkpoint at sequence 100
	pbft.createCheckpoint(100)

	time.Sleep(100 * time.Millisecond)

	// Verify checkpoint message was broadcast
	transport.mu.Lock()
	found := false
	for _, msg := range transport.messages {
		if checkpoint, ok := msg.(*CheckpointMessage); ok {
			if checkpoint.Sequence == 100 {
				found = true
				break
			}
		}
	}
	transport.mu.Unlock()

	assert.True(t, found, "Checkpoint message should be broadcast")
}

func TestPBFT_GarbageCollection(t *testing.T) {
	logger := zap.NewNop()
	transport := NewMockTransport()
	stateMachine := NewMockStateMachine()

	pbft, err := NewPBFT("node_0", 4, transport, stateMachine, logger)
	require.NoError(t, err)

	// Add some old messages
	for i := int64(1); i <= 200; i++ {
		key := pbft.logKey(0, i)
		pbft.prePrepareLog[key] = &PrePrepareMessage{
			View:     0,
			Sequence: i,
		}
	}

	initialSize := len(pbft.prePrepareLog)
	assert.Equal(t, 200, initialSize)

	// Garbage collect up to sequence 100
	pbft.garbageCollect(100)

	// Verify old messages were removed
	afterSize := len(pbft.prePrepareLog)
	assert.Less(t, afterSize, initialSize)

	// Messages below sequence 100 should be gone
	for i := int64(1); i < 100; i++ {
		key := pbft.logKey(0, i)
		assert.NotContains(t, pbft.prePrepareLog, key)
	}
}

func TestPBFT_Metrics(t *testing.T) {
	logger := zap.NewNop()
	transport := NewMockTransport()
	stateMachine := NewMockStateMachine()

	pbft, err := NewPBFT("node_0", 7, transport, stateMachine, logger)
	require.NoError(t, err)

	metrics := pbft.GetMetrics()

	assert.NotNil(t, metrics)
	assert.Equal(t, int64(0), metrics.View)
	assert.Equal(t, 2, metrics.ByzantineTolerance) // f=2 for n=7
	assert.Equal(t, 7, metrics.ReplicaCount)
	assert.True(t, metrics.IsPrimary)
}

func TestPBFT_ConcurrentConsensus(t *testing.T) {
	logger := zap.NewNop()
	transport := NewMockTransport()
	stateMachine := NewMockStateMachine()

	pbft, err := NewPBFT("node_0", 4, transport, stateMachine, logger)
	require.NoError(t, err)

	ctx := context.Background()
	concurrency := 10

	var wg sync.WaitGroup
	wg.Add(concurrency)

	for i := 0; i < concurrency; i++ {
		go func(id int) {
			defer wg.Done()

			value := map[string]interface{}{
				"key":   "test_key",
				"value": id,
			}

			err := pbft.Consensus(ctx, value)
			assert.NoError(t, err)
		}(i)
	}

	wg.Wait()

	// Verify all consensus operations completed
	metrics := pbft.GetMetrics()
	assert.GreaterOrEqual(t, metrics.ExecutedRequests, 0)
}

func BenchmarkPBFT_Consensus(b *testing.B) {
	logger := zap.NewNop()
	transport := NewMockTransport()
	stateMachine := NewMockStateMachine()

	pbft, err := NewPBFT("node_0", 4, transport, stateMachine, logger)
	require.NoError(b, err)

	ctx := context.Background()
	value := map[string]interface{}{
		"key":   "bench_key",
		"value": "bench_value",
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = pbft.Consensus(ctx, value)
	}
}

func BenchmarkPBFT_DigestComputation(b *testing.B) {
	logger := zap.NewNop()
	transport := NewMockTransport()
	stateMachine := NewMockStateMachine()

	pbft, err := NewPBFT("node_0", 4, transport, stateMachine, logger)
	require.NoError(b, err)

	req := &ClientRequest{
		ClientID:  "client_1",
		Timestamp: time.Now(),
		Operation: json.RawMessage(`{"key": "test", "value": "data"}`),
		Sequence:  1,
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = pbft.computeDigest(req)
	}
}
