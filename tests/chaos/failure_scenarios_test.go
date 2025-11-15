package chaos_test

import (
	"context"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/v3"
)

// TestNodeCrashScenarios simulates various node crash patterns
func TestNodeCrashScenarios(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	t.Run("Random10PercentCrash", func(t *testing.T) {
		nodeCount := 30
		crashCount := 3 // 10%

		nodes := setupNetwork(t, nodeCount, 0)
		startNetwork(nodes, ctx)

		// Submit transaction
		proposal := &dwcp.Block{Height: 1, Data: []byte("crash-test")}
		consensus1, err := nodes[crashCount].ProposeBlock(ctx, proposal)
		require.NoError(t, err)
		assert.True(t, consensus1)

		// Crash random nodes
		for i := 0; i < crashCount; i++ {
			nodes[i].Crash() // Immediate stop without cleanup
		}

		// Network should continue
		time.Sleep(1 * time.Second)
		proposal2 := &dwcp.Block{Height: 2, Data: []byte("post-crash")}
		consensus2, err := nodes[crashCount].ProposeBlock(ctx, proposal2)
		require.NoError(t, err)
		assert.True(t, consensus2, "Network should survive 10% crash")
	})

	t.Run("LeaderCrashDuringConsensus", func(t *testing.T) {
		nodes := setupNetwork(t, 20, 0)
		startNetwork(nodes, ctx)

		proposal := &dwcp.Block{Height: 1, Data: []byte("leader-crash")}

		// Start consensus
		consensusChan := make(chan bool)
		go func() {
			consensus, _ := nodes[0].ProposeBlock(ctx, proposal)
			consensusChan <- consensus
		}()

		// Crash leader mid-consensus
		time.Sleep(100 * time.Millisecond)
		nodes[0].Crash()

		// New leader should be elected
		select {
		case consensus := <-consensusChan:
			assert.False(t, consensus, "Original consensus should fail")
		case <-time.After(5 * time.Second):
			t.Fatal("Consensus timeout")
		}

		// Re-propose with new leader
		time.Sleep(1 * time.Second)
		consensus2, err := nodes[1].ProposeBlock(ctx, proposal)
		require.NoError(t, err)
		assert.True(t, consensus2, "New leader should achieve consensus")
	})

	t.Run("CascadingFailure", func(t *testing.T) {
		nodes := setupNetwork(t, 30, 0)
		startNetwork(nodes, ctx)

		// Initial proposal
		proposal := &dwcp.Block{Height: 1, Data: []byte("cascade-test")}
		consensus1, err := nodes[10].ProposeBlock(ctx, proposal)
		require.NoError(t, err)
		assert.True(t, consensus1)

		// Cascading crashes (every 500ms)
		for i := 0; i < 5; i++ {
			nodes[i].Crash()
			time.Sleep(500 * time.Millisecond)
		}

		// Network should still function
		proposal2 := &dwcp.Block{Height: 2, Data: []byte("post-cascade")}
		consensus2, err := nodes[10].ProposeBlock(ctx, proposal2)
		require.NoError(t, err)
		assert.True(t, consensus2, "Network should survive cascading failure")
	})
}

// TestMemoryExhaustion simulates memory pressure scenarios
func TestMemoryExhaustion(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping memory exhaustion test in short mode")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	nodes := setupNetwork(t, 10, 0)
	startNetwork(nodes, ctx)

	// Monitor initial memory
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	initialAlloc := m.Alloc

	// Create memory pressure with large blocks
	largeData := make([]byte, 10*1024*1024) // 10MB
	for i := 0; i < 100; i++ {
		block := &dwcp.Block{
			Height: int64(i),
			Data:   largeData,
		}
		nodes[0].ProposeBlock(ctx, block)
	}

	// Check memory growth
	runtime.ReadMemStats(&m)
	memoryGrowth := m.Alloc - initialAlloc

	// Memory should not grow excessively (< 500MB)
	assert.Less(t, memoryGrowth, uint64(500*1024*1024),
		"Memory growth should be bounded")

	// Verify garbage collection works
	runtime.GC()
	runtime.ReadMemStats(&m)
	afterGC := m.Alloc
	assert.Less(t, afterGC, m.Alloc, "GC should reclaim memory")
}

// TestDiskFullScenario simulates disk space exhaustion
func TestDiskFullScenario(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()

	nodes := setupNetwork(t, 5, 0)
	startNetwork(nodes, ctx)

	// Simulate disk full
	nodes[0].SimulateDiskFull()

	// Node should handle gracefully
	proposal := &dwcp.Block{Height: 1, Data: []byte("disk-full-test")}
	_, err := nodes[0].ProposeBlock(ctx, proposal)
	assert.Error(t, err, "Should fail gracefully on disk full")

	// Other nodes should continue
	consensus, err := nodes[1].ProposeBlock(ctx, proposal)
	require.NoError(t, err)
	assert.True(t, consensus, "Other nodes should continue")
}

// TestCPUSaturation tests behavior under CPU pressure
func TestCPUSaturation(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	nodes := setupNetwork(t, 10, 0)
	startNetwork(nodes, ctx)

	// Create CPU pressure with crypto operations
	var wg sync.WaitGroup
	for i := 0; i < runtime.NumCPU()*2; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 1000; j++ {
				nodes[0].ComputeHash([]byte("cpu-stress"))
			}
		}()
	}

	// Submit transaction under load
	proposal := &dwcp.Block{Height: 1, Data: []byte("cpu-test")}
	start := time.Now()
	consensus, err := nodes[0].ProposeBlock(ctx, proposal)
	latency := time.Since(start)

	require.NoError(t, err)
	assert.True(t, consensus)

	// Latency should be acceptable even under load
	assert.Less(t, latency, 10*time.Second, "Should complete despite CPU pressure")

	wg.Wait()
}

// TestNetworkCongestion simulates network saturation
func TestNetworkCongestion(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 40*time.Second)
	defer cancel()

	nodes := setupNetwork(t, 20, 0)
	startNetwork(nodes, ctx)

	// Saturate network with traffic
	var trafficWg sync.WaitGroup
	for i := 0; i < 10; i++ {
		trafficWg.Add(1)
		go func(nodeID int) {
			defer trafficWg.Done()
			for j := 0; j < 100; j++ {
				spam := &dwcp.Block{
					Height: int64(j),
					Data:   make([]byte, 1024*1024), // 1MB
				}
				nodes[nodeID].ProposeBlock(ctx, spam)
			}
		}(i)
	}

	// Important transaction during congestion
	time.Sleep(2 * time.Second)
	importantTx := &dwcp.Block{
		Height:   1000,
		Data:     []byte("important"),
		Priority: dwcp.HighPriority,
	}

	start := time.Now()
	consensus, err := nodes[15].ProposeBlock(ctx, importantTx)
	latency := time.Since(start)

	require.NoError(t, err)
	assert.True(t, consensus)

	// High priority should complete reasonably fast
	assert.Less(t, latency, 15*time.Second, "Priority tx should cut through congestion")

	trafficWg.Wait()
}

// TestClockSkew simulates time synchronization issues
func TestClockSkew(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()

	nodes := setupNetwork(t, 10, 0)
	startNetwork(nodes, ctx)

	// Introduce clock skew
	nodes[0].SetClockSkew(5 * time.Minute)  // 5 min ahead
	nodes[1].SetClockSkew(-3 * time.Minute) // 3 min behind

	// Timestamp validation should reject out-of-sync proposals
	futureBlock := &dwcp.Block{
		Height:    1,
		Timestamp: time.Now().Add(10 * time.Minute).Unix(),
		Data:      []byte("future-block"),
	}

	consensus, err := nodes[0].ProposeBlock(ctx, futureBlock)
	assert.False(t, consensus, "Future blocks should be rejected")
	assert.Error(t, err)

	// Normal proposal should still work
	normalBlock := &dwcp.Block{
		Height:    1,
		Timestamp: time.Now().Unix(),
		Data:      []byte("normal-block"),
	}

	consensus2, err := nodes[5].ProposeBlock(ctx, normalBlock)
	require.NoError(t, err)
	assert.True(t, consensus2)
}

// TestSplitBrain simulates complete network partition and rejoin
func TestSplitBrain(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 40*time.Second)
	defer cancel()

	nodes := setupNetwork(t, 20, 0)
	startNetwork(nodes, ctx)

	// Initial consensus
	block1 := &dwcp.Block{Height: 1, Data: []byte("pre-split")}
	consensus1, _ := nodes[0].ProposeBlock(ctx, block1)
	assert.True(t, consensus1)

	// Complete network split
	group1 := nodes[0:12]  // 60%
	group2 := nodes[12:20] // 40%
	isolatePartitions(group1, group2)

	// Both groups propose different blocks
	block2a := &dwcp.Block{Height: 2, Data: []byte("group-A")}
	block2b := &dwcp.Block{Height: 2, Data: []byte("group-B")}

	c1, _ := group1[0].ProposeBlock(ctx, block2a)
	c2, _ := group2[0].ProposeBlock(ctx, block2b)

	assert.True(t, c1, "Majority group should achieve consensus")
	assert.False(t, c2, "Minority group should not")

	// Rejoin network
	rejoinPartitions(group1, group2)
	time.Sleep(3 * time.Second)

	// Verify minority adopts majority state
	state1 := group1[0].GetState()
	state2 := group2[0].GetState()

	assert.Equal(t, state1.Height, state2.Height, "Heights should reconcile")
	assert.Equal(t, state1.Hash, state2.Hash, "States should converge")
	assert.Contains(t, state2.Blocks, block2a, "Minority should adopt majority block")
}

// Helper functions

func startNetwork(nodes []*dwcp.Node, ctx context.Context) {
	for _, node := range nodes {
		go node.Start(ctx)
	}
	time.Sleep(1 * time.Second) // Network formation
}
