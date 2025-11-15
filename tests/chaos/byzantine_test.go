package chaos_test

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/v3"
)

// TestByzantineNodeBehavior simulates 33% Byzantine nodes and verifies consensus
func TestByzantineNodeBehavior(t *testing.T) {
	totalNodes := 30
	byzantineNodes := 10 // 33%
	honestNodes := totalNodes - byzantineNodes

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Initialize network
	nodes := make([]*dwcp.Node, totalNodes)
	for i := 0; i < totalNodes; i++ {
		nodes[i] = dwcp.NewNode(&dwcp.Config{
			NodeID:           i,
			TotalNodes:       totalNodes,
			ByzantineTolerance: byzantineNodes,
			Protocol:         "ProBFT",
		})
	}

	// Mark Byzantine nodes
	byzantineSet := make(map[int]bool)
	for i := 0; i < byzantineNodes; i++ {
		byzantineSet[i] = true
		nodes[i].MarkAsByzantine()
	}

	// Start all nodes
	var wg sync.WaitGroup
	for _, node := range nodes {
		wg.Add(1)
		go func(n *dwcp.Node) {
			defer wg.Done()
			n.Start(ctx)
		}(node)
	}

	// Byzantine attack scenarios
	t.Run("RandomMessageDrop", func(t *testing.T) {
		proposal := &dwcp.Block{
			Height:    1,
			Timestamp: time.Now().Unix(),
			Data:      []byte("test-transaction"),
		}

		// Byzantine nodes drop random messages
		for nodeID := range byzantineSet {
			go func(id int) {
				nodes[id].SetMessageDropRate(0.5) // Drop 50% of messages
			}(nodeID)
		}

		// Submit proposal
		consensus, err := nodes[honestNodes].ProposeBlock(ctx, proposal)
		require.NoError(t, err)
		assert.True(t, consensus, "Consensus should be achieved despite message drops")
	})

	t.Run("MessageDelayAttack", func(t *testing.T) {
		proposal := &dwcp.Block{
			Height:    2,
			Timestamp: time.Now().Unix(),
			Data:      []byte("delayed-transaction"),
		}

		// Byzantine nodes delay messages
		for nodeID := range byzantineSet {
			go func(id int) {
				nodes[id].SetMessageDelay(2 * time.Second)
			}(nodeID)
		}

		start := time.Now()
		consensus, err := nodes[honestNodes].ProposeBlock(ctx, proposal)
		duration := time.Since(start)

		require.NoError(t, err)
		assert.True(t, consensus)
		assert.Less(t, duration, 5*time.Second, "Consensus should complete within timeout")
	})

	t.Run("Equivocation", func(t *testing.T) {
		// Byzantine nodes send conflicting votes
		proposal1 := &dwcp.Block{Height: 3, Data: []byte("block-A")}
		proposal2 := &dwcp.Block{Height: 3, Data: []byte("block-B")}

		// Byzantine nodes equivocate
		for nodeID := range byzantineSet {
			go func(id int) {
				nodes[id].Vote(ctx, proposal1)
				nodes[id].Vote(ctx, proposal2) // Double vote
			}(nodeID)
		}

		// Honest nodes vote on proposal1
		consensus, err := nodes[honestNodes].ProposeBlock(ctx, proposal1)
		require.NoError(t, err)
		assert.True(t, consensus, "Honest majority should achieve consensus")

		// Verify equivocation is detected
		detected := nodes[honestNodes].DetectEquivocation()
		assert.GreaterOrEqual(t, len(detected), byzantineNodes, "All Byzantine nodes should be detected")
	})

	t.Run("FakeBlockProposal", func(t *testing.T) {
		// Byzantine node proposes invalid block
		fakeBlock := &dwcp.Block{
			Height:    100, // Wrong height
			Timestamp: time.Now().Add(24 * time.Hour).Unix(), // Future timestamp
			Data:      []byte("invalid-data"),
		}

		consensus, err := nodes[0].ProposeBlock(ctx, fakeBlock)
		assert.Error(t, err, "Invalid block should be rejected")
		assert.False(t, consensus)
	})

	t.Run("CoordinatedByzantineAttack", func(t *testing.T) {
		// All Byzantine nodes coordinate to propose same invalid block
		coordBlock := &dwcp.Block{
			Height:    4,
			Timestamp: time.Now().Unix(),
			Data:      []byte("coordinated-attack"),
		}

		// Byzantine nodes vote for coordBlock
		var votesWg sync.WaitGroup
		for nodeID := range byzantineSet {
			votesWg.Add(1)
			go func(id int) {
				defer votesWg.Done()
				nodes[id].Vote(ctx, coordBlock)
			}(nodeID)
		}

		// Honest nodes vote for valid block
		validBlock := &dwcp.Block{
			Height:    4,
			Timestamp: time.Now().Unix(),
			Data:      []byte("valid-transaction"),
		}

		consensus, err := nodes[honestNodes].ProposeBlock(ctx, validBlock)
		require.NoError(t, err)
		assert.True(t, consensus, "Honest majority (70%) should win")
	})

	// Verify final state consistency
	t.Run("StateConsistency", func(t *testing.T) {
		time.Sleep(1 * time.Second) // Allow propagation

		// All honest nodes should have same state
		referenceState := nodes[honestNodes].GetState()
		for i := byzantineNodes; i < totalNodes; i++ {
			state := nodes[i].GetState()
			assert.Equal(t, referenceState.Height, state.Height, "Node %d state mismatch", i)
			assert.Equal(t, referenceState.Hash, state.Hash, "Node %d hash mismatch", i)
		}
	})

	cancel()
	wg.Wait()
}

// TestByzantineRecovery tests network recovery after Byzantine nodes are removed
func TestByzantineRecovery(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()

	nodes := setupNetwork(t, 10, 3) // 10 nodes, 3 Byzantine

	// Phase 1: Network with Byzantine nodes
	proposal1 := &dwcp.Block{Height: 1, Data: []byte("pre-recovery")}
	consensus1, err := nodes[7].ProposeBlock(ctx, proposal1)
	require.NoError(t, err)
	assert.True(t, consensus1)

	// Phase 2: Remove Byzantine nodes
	for i := 0; i < 3; i++ {
		nodes[i].Stop()
	}

	// Phase 3: Network continues with remaining honest nodes
	proposal2 := &dwcp.Block{Height: 2, Data: []byte("post-recovery")}
	consensus2, err := nodes[7].ProposeBlock(ctx, proposal2)
	require.NoError(t, err)
	assert.True(t, consensus2, "Network should recover after Byzantine removal")
}

// setupNetwork creates test network
func setupNetwork(t *testing.T, total, byzantine int) []*dwcp.Node {
	nodes := make([]*dwcp.Node, total)
	for i := 0; i < total; i++ {
		nodes[i] = dwcp.NewNode(&dwcp.Config{
			NodeID:           i,
			TotalNodes:       total,
			ByzantineTolerance: byzantine,
		})
		if i < byzantine {
			nodes[i].MarkAsByzantine()
		}
	}
	return nodes
}
