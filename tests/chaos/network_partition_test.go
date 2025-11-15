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

// TestNetworkPartition simulates network splits and recovery
func TestNetworkPartition(t *testing.T) {
	totalNodes := 20
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	nodes := make([]*dwcp.Node, totalNodes)
	for i := 0; i < totalNodes; i++ {
		nodes[i] = dwcp.NewNode(&dwcp.Config{
			NodeID:     i,
			TotalNodes: totalNodes,
			Protocol:   "ProBFT",
		})
	}

	// Start all nodes
	var startWg sync.WaitGroup
	for _, node := range nodes {
		startWg.Add(1)
		go func(n *dwcp.Node) {
			defer startWg.Done()
			n.Start(ctx)
		}(node)
	}
	startWg.Wait()
	time.Sleep(500 * time.Millisecond) // Allow network formation

	t.Run("50-50Split", func(t *testing.T) {
		// Create network partition: 50% vs 50%
		partition1 := nodes[0:10]  // Group A
		partition2 := nodes[10:20] // Group B

		// Isolate groups
		isolatePartitions(partition1, partition2)

		// Both partitions attempt consensus
		proposal := &dwcp.Block{Height: 1, Data: []byte("partition-test")}

		// Group A attempts consensus
		consensus1, err1 := partition1[0].ProposeBlock(ctx, proposal)

		// Group B attempts consensus
		consensus2, err2 := partition2[0].ProposeBlock(ctx, proposal)

		// With 50-50 split, neither should achieve consensus (need 2/3 majority)
		assert.False(t, consensus1, "Partition 1 should not achieve consensus")
		assert.False(t, consensus2, "Partition 2 should not achieve consensus")
		assert.Error(t, err1, "Should timeout without quorum")
		assert.Error(t, err2, "Should timeout without quorum")

		// Verify both partitions halted
		assert.True(t, partition1[0].IsHalted(), "Partition 1 should halt without quorum")
		assert.True(t, partition2[0].IsHalted(), "Partition 2 should halt without quorum")

		// Rejoin network
		rejoinPartitions(partition1, partition2)
		time.Sleep(1 * time.Second)

		// After rejoin, consensus should be achievable
		proposal2 := &dwcp.Block{Height: 2, Data: []byte("post-rejoin")}
		consensus3, err3 := nodes[0].ProposeBlock(ctx, proposal2)
		require.NoError(t, err3)
		assert.True(t, consensus3, "Full network should achieve consensus")
	})

	t.Run("70-30Split", func(t *testing.T) {
		// Majority partition (70%) vs minority (30%)
		majority := nodes[0:14]  // 70%
		minority := nodes[14:20] // 30%

		isolatePartitions(majority, minority)

		proposal := &dwcp.Block{Height: 3, Data: []byte("70-30-split")}

		// Majority should achieve consensus
		consensusMajority, errMajority := majority[0].ProposeBlock(ctx, proposal)
		require.NoError(t, errMajority)
		assert.True(t, consensusMajority, "Majority partition should achieve consensus")

		// Minority should not
		consensusMinority, errMinority := minority[0].ProposeBlock(ctx, proposal)
		assert.False(t, consensusMinority)
		assert.Error(t, errMinority)

		// Rejoin and verify state reconciliation
		rejoinPartitions(majority, minority)
		time.Sleep(2 * time.Second) // Allow state sync

		// Minority should sync to majority state
		majorityState := majority[0].GetState()
		minorityState := minority[0].GetState()
		assert.Equal(t, majorityState.Height, minorityState.Height, "States should reconcile")
		assert.Equal(t, majorityState.Hash, minorityState.Hash, "Hashes should match")
	})

	t.Run("FlappingPartition", func(t *testing.T) {
		// Simulate network instability with repeated partitions
		group1 := nodes[0:10]
		group2 := nodes[10:20]

		for i := 0; i < 5; i++ {
			// Partition
			isolatePartitions(group1, group2)
			time.Sleep(200 * time.Millisecond)

			// Rejoin
			rejoinPartitions(group1, group2)
			time.Sleep(200 * time.Millisecond)
		}

		// After flapping, network should stabilize
		proposal := &dwcp.Block{Height: 4, Data: []byte("post-flapping")}
		consensus, err := nodes[0].ProposeBlock(ctx, proposal)
		require.NoError(t, err)
		assert.True(t, consensus, "Network should stabilize after flapping")
	})

	t.Run("TriplePartition", func(t *testing.T) {
		// Split into 3 groups
		group1 := nodes[0:7]   // 35%
		group2 := nodes[7:14]  // 35%
		group3 := nodes[14:20] // 30%

		isolatePartitions(group1, group2)
		isolatePartitions(group1, group3)
		isolatePartitions(group2, group3)

		proposal := &dwcp.Block{Height: 5, Data: []byte("triple-partition")}

		// No partition should achieve consensus (all < 67%)
		c1, _ := group1[0].ProposeBlock(ctx, proposal)
		c2, _ := group2[0].ProposeBlock(ctx, proposal)
		c3, _ := group3[0].ProposeBlock(ctx, proposal)

		assert.False(t, c1, "Group 1 should not achieve consensus")
		assert.False(t, c2, "Group 2 should not achieve consensus")
		assert.False(t, c3, "Group 3 should not achieve consensus")

		// Merge all groups
		rejoinPartitions(group1, group2)
		rejoinPartitions(group1, group3)
		time.Sleep(1 * time.Second)

		// Full network should achieve consensus
		proposal2 := &dwcp.Block{Height: 6, Data: []byte("post-merge")}
		consensus, err := nodes[0].ProposeBlock(ctx, proposal2)
		require.NoError(t, err)
		assert.True(t, consensus)
	})

	cancel()
	for _, node := range nodes {
		node.Stop()
	}
}

// TestPartitionWithByzantineNodes tests partitions combined with Byzantine behavior
func TestPartitionWithByzantineNodes(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	totalNodes := 15
	byzantineNodes := 3

	nodes := make([]*dwcp.Node, totalNodes)
	for i := 0; i < totalNodes; i++ {
		nodes[i] = dwcp.NewNode(&dwcp.Config{
			NodeID:           i,
			TotalNodes:       totalNodes,
			ByzantineTolerance: byzantineNodes,
		})
		if i < byzantineNodes {
			nodes[i].MarkAsByzantine()
		}
	}

	// Start nodes
	for _, node := range nodes {
		go node.Start(ctx)
	}
	time.Sleep(500 * time.Millisecond)

	// Create partition with Byzantine nodes in one group
	byzantineGroup := nodes[0:8]  // Contains all 3 Byzantine nodes
	honestGroup := nodes[8:15]    // All honest

	isolatePartitions(byzantineGroup, honestGroup)

	proposal := &dwcp.Block{Height: 1, Data: []byte("byzantine-partition")}

	// Byzantine group should not achieve consensus (3/8 < 2/3)
	c1, _ := byzantineGroup[4].ProposeBlock(ctx, proposal) // Use honest node in Byzantine group
	assert.False(t, c1, "Byzantine-heavy partition should not achieve consensus")

	// Honest group should not achieve consensus (7/15 < 2/3 of total)
	c2, _ := honestGroup[0].ProposeBlock(ctx, proposal)
	assert.False(t, c2, "Minority honest partition should not achieve consensus")

	// Rejoin
	rejoinPartitions(byzantineGroup, honestGroup)
	time.Sleep(1 * time.Second)

	// Full network with honest majority should achieve consensus
	proposal2 := &dwcp.Block{Height: 2, Data: []byte("post-rejoin")}
	consensus, err := honestGroup[0].ProposeBlock(ctx, proposal2)
	require.NoError(t, err)
	assert.True(t, consensus, "Honest majority should achieve consensus")

	cancel()
}

// Helper functions

func isolatePartitions(group1, group2 []*dwcp.Node) {
	// Simulate network partition by blocking communication
	for _, n1 := range group1 {
		for _, n2 := range group2 {
			n1.BlockPeer(n2.ID())
			n2.BlockPeer(n1.ID())
		}
	}
}

func rejoinPartitions(group1, group2 []*dwcp.Node) {
	// Remove network partition
	for _, n1 := range group1 {
		for _, n2 := range group2 {
			n1.UnblockPeer(n2.ID())
			n2.UnblockPeer(n1.ID())
		}
	}
}
