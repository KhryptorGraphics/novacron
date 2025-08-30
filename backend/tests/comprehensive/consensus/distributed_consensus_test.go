package consensus_test

import (
	"context"
	"fmt"
	"math/rand"
	"sync"
	"testing"
	"time"
)

// TestDistributedConsensus implements comprehensive consensus algorithm testing
func TestDistributedConsensus(t *testing.T) {
	t.Run("Leader Election Tests", func(t *testing.T) {
		testBasicLeaderElection(t)
		testLeaderFailover(t)
		testSplitBrainPrevention(t)
		testLeaderElectionUnderLoad(t)
	})

	t.Run("State Replication Tests", func(t *testing.T) {
		testBasicStateReplication(t)
		testConsistencyGuarantees(t)
		testPartitionToleranceReplication(t)
		testStateRecovery(t)
	})

	t.Run("Quorum-Based Operations", func(t *testing.T) {
		testQuorumFormation(t)
		testQuorumBasedCommits(t)
		testMinorityPartitionHandling(t)
		testQuorumReconfiguration(t)
	})

	t.Run("Byzantine Fault Tolerance", func(t *testing.T) {
		testByzantineNodeDetection(t)
		testByzantineConsensus(t)
		testMaliciousNodeIsolation(t)
	})

	t.Run("Performance and Scalability", func(t *testing.T) {
		testConsensusLatency(t)
		testThroughputUnderLoad(t)
		testScalabilityLimits(t)
	})
}

// ConsensusNode represents a node participating in consensus
type ConsensusNode struct {
	id               string
	state            NodeState
	term             int64
	votedFor         string
	log              []LogEntry
	commitIndex      int64
	lastApplied      int64
	nextIndex        map[string]int64
	matchIndex       map[string]int64
	peers            map[string]*ConsensusNode
	healthy          bool
	partitioned      bool
	byzantine        bool
	byzantineBehavior ByzantineBehavior
	mu               sync.RWMutex
	electionTimer    *time.Timer
	heartbeatTimer   *time.Timer
	voteCount        int
	leaderID         string
}

type NodeState int

const (
	Follower NodeState = iota
	Candidate
	Leader
)

type LogEntry struct {
	Index int64
	Term  int64
	Data  interface{}
}

type ByzantineBehavior int

const (
	ByzantineNone ByzantineBehavior = iota
	ByzantineDelayedResponse
	ByzantineIncorrectVote
	ByzantineDataCorruption
	ByzantineSilentFailure
)

type ConsensusCluster struct {
	nodes      map[string]*ConsensusNode
	partitions map[string][]string
	mu         sync.RWMutex
}

func testBasicLeaderElection(t *testing.T) {
	cluster := NewConsensusCluster(5)
	defer cluster.Cleanup()

	// Start all nodes
	cluster.StartAllNodes()

	// Wait for leader election
	time.Sleep(2 * time.Second)

	// Verify exactly one leader
	leaders := cluster.GetLeaders()
	if len(leaders) != 1 {
		t.Errorf("Expected exactly 1 leader, got %d", len(leaders))
	}

	// Verify all other nodes are followers
	followers := cluster.GetFollowers()
	if len(followers) != 4 {
		t.Errorf("Expected 4 followers, got %d", len(followers))
	}

	// Verify leader has highest term
	leaderTerm := leaders[0].GetTerm()
	for _, follower := range followers {
		if follower.GetTerm() > leaderTerm {
			t.Errorf("Follower term %d higher than leader term %d", 
				follower.GetTerm(), leaderTerm)
		}
	}
}

func testLeaderFailover(t *testing.T) {
	cluster := NewConsensusCluster(5)
	defer cluster.Cleanup()

	cluster.StartAllNodes()
	time.Sleep(1 * time.Second)

	// Get initial leader
	leaders := cluster.GetLeaders()
	if len(leaders) != 1 {
		t.Fatalf("Expected exactly 1 initial leader, got %d", len(leaders))
	}
	initialLeader := leaders[0]

	// Simulate leader failure
	cluster.SimulateNodeFailure(initialLeader.id)

	// Wait for new leader election
	time.Sleep(3 * time.Second)

	// Verify new leader elected
	newLeaders := cluster.GetLeaders()
	if len(newLeaders) != 1 {
		t.Errorf("Expected exactly 1 new leader after failover, got %d", len(newLeaders))
		return
	}

	newLeader := newLeaders[0]
	if newLeader.id == initialLeader.id {
		t.Error("New leader should be different from failed leader")
	}

	// Verify new leader has higher term
	if newLeader.GetTerm() <= initialLeader.GetTerm() {
		t.Errorf("New leader term %d should be higher than failed leader term %d", 
			newLeader.GetTerm(), initialLeader.GetTerm())
	}
}

func testSplitBrainPrevention(t *testing.T) {
	cluster := NewConsensusCluster(6) // Even number for split-brain scenario
	defer cluster.Cleanup()

	cluster.StartAllNodes()
	time.Sleep(1 * time.Second)

	// Create network partition: 3 vs 3 nodes
	nodeIDs := cluster.GetNodeIDs()
	partition1 := nodeIDs[:3]
	partition2 := nodeIDs[3:]

	cluster.ApplyNetworkPartition(partition1, partition2)

	// Wait for partition effects
	time.Sleep(4 * time.Second)

	// Count leaders in each partition
	partition1Leaders := 0
	partition2Leaders := 0

	for _, nodeID := range partition1 {
		if cluster.IsLeader(nodeID) {
			partition1Leaders++
		}
	}

	for _, nodeID := range partition2 {
		if cluster.IsLeader(nodeID) {
			partition2Leaders++
		}
	}

	// In a proper consensus algorithm, no partition should have a leader
	// since neither has a majority
	totalLeaders := partition1Leaders + partition2Leaders

	if totalLeaders > 1 {
		t.Errorf("Split-brain detected: %d leaders total (p1=%d, p2=%d)", 
			totalLeaders, partition1Leaders, partition2Leaders)
	}

	// Restore network and verify single leader emerges
	cluster.RestoreNetwork()
	time.Sleep(2 * time.Second)

	finalLeaders := cluster.GetLeaders()
	if len(finalLeaders) != 1 {
		t.Errorf("Expected exactly 1 leader after partition recovery, got %d", len(finalLeaders))
	}
}

func testLeaderElectionUnderLoad(t *testing.T) {
	cluster := NewConsensusCluster(7)
	defer cluster.Cleanup()

	cluster.StartAllNodes()

	// Simulate continuous load during leader elections
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	var wg sync.WaitGroup
	
	// Simulate continuous client operations
	wg.Add(1)
	go func() {
		defer wg.Done()
		ticker := time.NewTicker(100 * time.Millisecond)
		defer ticker.Stop()
		
		operationCount := 0
		for {
			select {
			case <-ctx.Done():
				t.Logf("Completed %d operations under load", operationCount)
				return
			case <-ticker.C:
				leaders := cluster.GetLeaders()
				if len(leaders) == 1 {
					err := leaders[0].ProposeOperation(fmt.Sprintf("op-%d", operationCount))
					if err == nil {
						operationCount++
					}
				}
			}
		}
	}()

	// Simulate periodic leader failures
	wg.Add(1)
	go func() {
		defer wg.Done()
		ticker := time.NewTicker(2 * time.Second)
		defer ticker.Stop()
		
		failureCount := 0
		for {
			select {
			case <-ctx.Done():
				t.Logf("Simulated %d leader failures", failureCount)
				return
			case <-ticker.C:
				leaders := cluster.GetLeaders()
				if len(leaders) == 1 {
					cluster.SimulateNodeFailure(leaders[0].id)
					failureCount++
					
					// Restore node after some time
					go func(nodeID string) {
						time.Sleep(1 * time.Second)
						cluster.RestoreNode(nodeID)
					}(leaders[0].id)
				}
			}
		}
	}()

	wg.Wait()

	// Verify cluster stability after load test
	time.Sleep(2 * time.Second)
	finalLeaders := cluster.GetLeaders()
	if len(finalLeaders) != 1 {
		t.Errorf("Expected stable leader after load test, got %d leaders", len(finalLeaders))
	}
}

func testBasicStateReplication(t *testing.T) {
	cluster := NewConsensusCluster(5)
	defer cluster.Cleanup()

	cluster.StartAllNodes()
	time.Sleep(1 * time.Second)

	// Get leader
	leaders := cluster.GetLeaders()
	if len(leaders) != 1 {
		t.Fatalf("Expected exactly 1 leader for state replication test")
	}
	leader := leaders[0]

	// Propose several operations
	operations := []string{"set x=1", "set y=2", "set z=3", "increment x", "delete y"}
	
	for _, op := range operations {
		err := leader.ProposeOperation(op)
		if err != nil {
			t.Errorf("Failed to propose operation '%s': %v", op, err)
		}
		time.Sleep(100 * time.Millisecond) // Allow replication
	}

	// Wait for replication to complete
	time.Sleep(1 * time.Second)

	// Verify all nodes have the same log
	leaderLog := leader.GetLog()
	
	for _, nodeID := range cluster.GetNodeIDs() {
		if nodeID == leader.id {
			continue
		}
		
		node := cluster.GetNode(nodeID)
		if !node.healthy {
			continue
		}
		
		nodeLog := node.GetLog()
		
		if len(nodeLog) != len(leaderLog) {
			t.Errorf("Node %s log length %d != leader log length %d", 
				nodeID, len(nodeLog), len(leaderLog))
			continue
		}
		
		for i, entry := range leaderLog {
			if nodeLog[i].Data != entry.Data {
				t.Errorf("Node %s log entry %d mismatch: got %v, expected %v", 
					nodeID, i, nodeLog[i].Data, entry.Data)
			}
		}
	}
}

func testConsistencyGuarantees(t *testing.T) {
	cluster := NewConsensusCluster(5)
	defer cluster.Cleanup()

	cluster.StartAllNodes()
	time.Sleep(1 * time.Second)

	// Test linearizability: operations should appear to execute atomically
	// and in some sequential order consistent with real-time ordering

	leaders := cluster.GetLeaders()
	if len(leaders) != 1 {
		t.Fatalf("Expected exactly 1 leader")
	}
	leader := leaders[0]

	// Concurrent operations with dependencies
	var wg sync.WaitGroup
	operations := make(chan string, 10)
	
	// Producer: generate dependent operations
	wg.Add(1)
	go func() {
		defer wg.Done()
		defer close(operations)
		
		for i := 0; i < 5; i++ {
			operations <- fmt.Sprintf("write x=%d", i)
			operations <- fmt.Sprintf("read x")
		}
	}()

	// Consumer: execute operations
	wg.Add(1)
	go func() {
		defer wg.Done()
		
		for op := range operations {
			err := leader.ProposeOperation(op)
			if err != nil {
				t.Errorf("Failed to propose operation '%s': %v", op, err)
			}
			time.Sleep(50 * time.Millisecond)
		}
	}()

	wg.Wait()
	time.Sleep(1 * time.Second)

	// Verify consistency across all nodes
	referenceLog := leader.GetLog()
	
	for _, nodeID := range cluster.GetNodeIDs() {
		node := cluster.GetNode(nodeID)
		if !node.healthy || nodeID == leader.id {
			continue
		}
		
		nodeLog := node.GetLog()
		if !logsEqual(nodeLog, referenceLog) {
			t.Errorf("Consistency violation: node %s has different log than leader", nodeID)
		}
	}
}

func testPartitionToleranceReplication(t *testing.T) {
	cluster := NewConsensusCluster(5)
	defer cluster.Cleanup()

	cluster.StartAllNodes()
	time.Sleep(1 * time.Second)

	// Partition network: majority (3) vs minority (2)
	nodeIDs := cluster.GetNodeIDs()
	majorityNodes := nodeIDs[:3]
	minorityNodes := nodeIDs[3:]

	cluster.ApplyNetworkPartition(majorityNodes, minorityNodes)

	// Wait for partition to take effect
	time.Sleep(2 * time.Second)

	// Find leader in majority partition
	var majorityLeader *ConsensusNode
	for _, nodeID := range majorityNodes {
		if cluster.IsLeader(nodeID) {
			majorityLeader = cluster.GetNode(nodeID)
			break
		}
	}

	if majorityLeader == nil {
		// Wait longer for leader election in majority
		time.Sleep(2 * time.Second)
		for _, nodeID := range majorityNodes {
			if cluster.IsLeader(nodeID) {
				majorityLeader = cluster.GetNode(nodeID)
				break
			}
		}
	}

	if majorityLeader != nil {
		// Operations should succeed in majority partition
		err := majorityLeader.ProposeOperation("partition-test-op")
		if err != nil {
			t.Logf("Operation failed in majority partition (may be expected): %v", err)
		}
	}

	// Verify no leader in minority partition
	minorityLeaderCount := 0
	for _, nodeID := range minorityNodes {
		if cluster.IsLeader(nodeID) {
			minorityLeaderCount++
		}
	}

	if minorityLeaderCount > 0 {
		t.Errorf("Minority partition should not have leader, found %d", minorityLeaderCount)
	}

	// Restore network and verify convergence
	cluster.RestoreNetwork()
	time.Sleep(3 * time.Second)

	finalLeaders := cluster.GetLeaders()
	if len(finalLeaders) != 1 {
		t.Errorf("Expected single leader after partition recovery, got %d", len(finalLeaders))
	}
}

func testQuorumFormation(t *testing.T) {
	// Test different cluster sizes and their quorum requirements
	testCases := []struct {
		clusterSize int
		expectedQuorum int
	}{
		{3, 2},
		{5, 3},
		{7, 4},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("ClusterSize%d", tc.clusterSize), func(t *testing.T) {
			cluster := NewConsensusCluster(tc.clusterSize)
			defer cluster.Cleanup()

			cluster.StartAllNodes()
			time.Sleep(1 * time.Second)

			// Verify quorum formation
			quorumSize := cluster.GetQuorumSize()
			if quorumSize != tc.expectedQuorum {
				t.Errorf("Expected quorum size %d, got %d", tc.expectedQuorum, quorumSize)
			}

			// Test quorum-based operations
			leaders := cluster.GetLeaders()
			if len(leaders) == 1 {
				// Quorum should be able to commit operations
				err := leaders[0].ProposeOperation("quorum-test")
				if err != nil {
					t.Errorf("Quorum-based operation failed: %v", err)
				}
			}

			// Test with insufficient nodes (below quorum)
			nodesToFail := tc.clusterSize - tc.expectedQuorum + 1
			nodeIDs := cluster.GetNodeIDs()
			
			for i := 0; i < nodesToFail; i++ {
				cluster.SimulateNodeFailure(nodeIDs[i])
			}

			time.Sleep(2 * time.Second)

			// Should not be able to elect leader or commit operations
			remainingLeaders := cluster.GetLeaders()
			if len(remainingLeaders) > 0 {
				t.Errorf("Should not have leader with insufficient nodes, got %d", len(remainingLeaders))
			}
		})
	}
}

func testByzantineNodeDetection(t *testing.T) {
	cluster := NewConsensusCluster(7) // Need f=2, so n=3f+1=7
	defer cluster.Cleanup()

	cluster.StartAllNodes()
	time.Sleep(1 * time.Second)

	// Make 2 nodes Byzantine (maximum tolerable)
	nodeIDs := cluster.GetNodeIDs()
	cluster.SetByzantineBehavior(nodeIDs[0], ByzantineIncorrectVote)
	cluster.SetByzantineBehavior(nodeIDs[1], ByzantineDataCorruption)

	// System should still function correctly
	time.Sleep(3 * time.Second)

	leaders := cluster.GetLeaders()
	if len(leaders) != 1 {
		t.Errorf("Expected single leader despite Byzantine nodes, got %d", len(leaders))
	}

	// Verify Byzantine nodes are not leaders
	if len(leaders) == 1 {
		leader := leaders[0]
		if leader.id == nodeIDs[0] || leader.id == nodeIDs[1] {
			t.Error("Byzantine node should not be leader")
		}

		// Operations should still succeed
		err := leader.ProposeOperation("byzantine-tolerance-test")
		if err != nil {
			t.Errorf("Operation should succeed despite Byzantine nodes: %v", err)
		}
	}
}

func testConsensusLatency(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping latency test in short mode")
	}

	cluster := NewConsensusCluster(5)
	defer cluster.Cleanup()

	cluster.StartAllNodes()
	time.Sleep(1 * time.Second)

	leaders := cluster.GetLeaders()
	if len(leaders) != 1 {
		t.Fatalf("Expected exactly 1 leader for latency test")
	}
	leader := leaders[0]

	// Measure consensus latency
	numOperations := 100
	latencies := make([]time.Duration, numOperations)

	for i := 0; i < numOperations; i++ {
		start := time.Now()
		err := leader.ProposeOperation(fmt.Sprintf("latency-test-%d", i))
		latencies[i] = time.Since(start)

		if err != nil {
			t.Errorf("Operation %d failed: %v", i, err)
		}

		time.Sleep(10 * time.Millisecond) // Small delay between operations
	}

	// Calculate latency statistics
	var totalLatency time.Duration
	var maxLatency time.Duration
	
	for _, latency := range latencies {
		totalLatency += latency
		if latency > maxLatency {
			maxLatency = latency
		}
	}

	avgLatency := totalLatency / time.Duration(numOperations)
	
	t.Logf("Consensus latency: avg=%v, max=%v", avgLatency, maxLatency)

	// Verify reasonable latency
	maxAcceptableAvg := 100 * time.Millisecond
	maxAcceptableMax := 500 * time.Millisecond

	if avgLatency > maxAcceptableAvg {
		t.Errorf("Average latency too high: %v > %v", avgLatency, maxAcceptableAvg)
	}

	if maxLatency > maxAcceptableMax {
		t.Errorf("Maximum latency too high: %v > %v", maxLatency, maxAcceptableMax)
	}
}

func testThroughputUnderLoad(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping throughput test in short mode")
	}

	cluster := NewConsensusCluster(5)
	defer cluster.Cleanup()

	cluster.StartAllNodes()
	time.Sleep(1 * time.Second)

	leaders := cluster.GetLeaders()
	if len(leaders) != 1 {
		t.Fatalf("Expected exactly 1 leader for throughput test")
	}
	leader := leaders[0]

	// High-throughput test
	duration := 5 * time.Second
	ctx, cancel := context.WithTimeout(context.Background(), duration)
	defer cancel()

	var operationCount int64
	var errorCount int64
	var wg sync.WaitGroup

	// Multiple concurrent clients
	numClients := 10
	for i := 0; i < numClients; i++ {
		wg.Add(1)
		go func(clientID int) {
			defer wg.Done()
			
			clientOps := 0
			clientErrors := 0
			
			for {
				select {
				case <-ctx.Done():
					// Use atomic operations for thread safety
					// For this test, we'll just log the results
					t.Logf("Client %d: %d operations, %d errors", clientID, clientOps, clientErrors)
					return
				default:
					err := leader.ProposeOperation(fmt.Sprintf("client-%d-op-%d", clientID, clientOps))
					if err != nil {
						clientErrors++
					} else {
						clientOps++
					}
				}
			}
		}(i)
	}

	wg.Wait()

	// Calculate throughput
	throughput := float64(operationCount) / duration.Seconds()
	errorRate := float64(errorCount) / float64(operationCount+errorCount) * 100

	t.Logf("Throughput: %.2f ops/sec, Error rate: %.2f%%", throughput, errorRate)

	// Verify acceptable throughput
	minThroughput := 100.0 // ops/sec
	maxErrorRate := 5.0    // percent

	if throughput < minThroughput {
		t.Errorf("Throughput too low: %.2f < %.2f ops/sec", throughput, minThroughput)
	}

	if errorRate > maxErrorRate {
		t.Errorf("Error rate too high: %.2f%% > %.2f%%", errorRate, maxErrorRate)
	}
}

// Helper types and functions

func NewConsensusCluster(size int) *ConsensusCluster {
	cluster := &ConsensusCluster{
		nodes:      make(map[string]*ConsensusNode, size),
		partitions: make(map[string][]string),
	}

	// Create nodes
	for i := 0; i < size; i++ {
		nodeID := fmt.Sprintf("node-%d", i)
		node := &ConsensusNode{
			id:          nodeID,
			state:       Follower,
			term:        0,
			healthy:     true,
			peers:       make(map[string]*ConsensusNode),
			nextIndex:   make(map[string]int64),
			matchIndex:  make(map[string]int64),
		}
		cluster.nodes[nodeID] = node
	}

	// Connect all nodes as peers
	for _, node := range cluster.nodes {
		for _, peer := range cluster.nodes {
			if node.id != peer.id {
				node.peers[peer.id] = peer
			}
		}
	}

	return cluster
}

func (c *ConsensusCluster) Cleanup() {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	for _, node := range c.nodes {
		node.Stop()
	}
}

func (c *ConsensusCluster) StartAllNodes() {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	for _, node := range c.nodes {
		node.Start()
	}
}

func (c *ConsensusCluster) GetLeaders() []*ConsensusNode {
	c.mu.RLock()
	defer c.mu.RUnlock()
	
	var leaders []*ConsensusNode
	for _, node := range c.nodes {
		if node.healthy && node.IsLeader() {
			leaders = append(leaders, node)
		}
	}
	return leaders
}

func (c *ConsensusCluster) GetFollowers() []*ConsensusNode {
	c.mu.RLock()
	defer c.mu.RUnlock()
	
	var followers []*ConsensusNode
	for _, node := range c.nodes {
		if node.healthy && node.IsFollower() {
			followers = append(followers, node)
		}
	}
	return followers
}

func (c *ConsensusCluster) GetNodeIDs() []string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	
	var ids []string
	for id := range c.nodes {
		ids = append(ids, id)
	}
	return ids
}

func (c *ConsensusCluster) GetNode(nodeID string) *ConsensusNode {
	c.mu.RLock()
	defer c.mu.RUnlock()
	
	return c.nodes[nodeID]
}

func (c *ConsensusCluster) IsLeader(nodeID string) bool {
	node := c.GetNode(nodeID)
	return node != nil && node.healthy && node.IsLeader()
}

func (c *ConsensusCluster) SimulateNodeFailure(nodeID string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	if node, exists := c.nodes[nodeID]; exists {
		node.healthy = false
		node.Stop()
	}
}

func (c *ConsensusCluster) RestoreNode(nodeID string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	if node, exists := c.nodes[nodeID]; exists {
		node.healthy = true
		node.Start()
	}
}

func (c *ConsensusCluster) ApplyNetworkPartition(partition1, partition2 []string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	// Mark nodes as partitioned from each other
	for _, node1ID := range partition1 {
		for _, node2ID := range partition2 {
			key := fmt.Sprintf("%s-%s", node1ID, node2ID)
			c.partitions[key] = []string{node1ID, node2ID}
		}
	}
}

func (c *ConsensusCluster) RestoreNetwork() {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	c.partitions = make(map[string][]string)
}

func (c *ConsensusCluster) GetQuorumSize() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	
	healthyNodes := 0
	for _, node := range c.nodes {
		if node.healthy {
			healthyNodes++
		}
	}
	return (healthyNodes / 2) + 1
}

func (c *ConsensusCluster) SetByzantineBehavior(nodeID string, behavior ByzantineBehavior) {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	if node, exists := c.nodes[nodeID]; exists {
		node.byzantine = true
		node.byzantineBehavior = behavior
	}
}

// ConsensusNode methods

func (n *ConsensusNode) Start() {
	n.mu.Lock()
	defer n.mu.Unlock()
	
	if !n.healthy {
		return
	}
	
	n.state = Follower
	n.resetElectionTimer()
}

func (n *ConsensusNode) Stop() {
	n.mu.Lock()
	defer n.mu.Unlock()
	
	if n.electionTimer != nil {
		n.electionTimer.Stop()
	}
	if n.heartbeatTimer != nil {
		n.heartbeatTimer.Stop()
	}
}

func (n *ConsensusNode) IsLeader() bool {
	n.mu.RLock()
	defer n.mu.RUnlock()
	return n.state == Leader
}

func (n *ConsensusNode) IsFollower() bool {
	n.mu.RLock()
	defer n.mu.RUnlock()
	return n.state == Follower
}

func (n *ConsensusNode) GetTerm() int64 {
	n.mu.RLock()
	defer n.mu.RUnlock()
	return n.term
}

func (n *ConsensusNode) GetLog() []LogEntry {
	n.mu.RLock()
	defer n.mu.RUnlock()
	
	// Return a copy to avoid race conditions
	logCopy := make([]LogEntry, len(n.log))
	copy(logCopy, n.log)
	return logCopy
}

func (n *ConsensusNode) ProposeOperation(operation string) error {
	n.mu.Lock()
	defer n.mu.Unlock()
	
	if !n.healthy || n.state != Leader {
		return fmt.Errorf("node %s is not a healthy leader", n.id)
	}
	
	// Add to log
	entry := LogEntry{
		Index: int64(len(n.log)) + 1,
		Term:  n.term,
		Data:  operation,
	}
	n.log = append(n.log, entry)
	
	// Simulate replication delay
	time.Sleep(10 * time.Millisecond)
	
	return nil
}

func (n *ConsensusNode) resetElectionTimer() {
	if n.electionTimer != nil {
		n.electionTimer.Stop()
	}
	
	// Random timeout between 150-300ms
	timeout := time.Duration(150+rand.Intn(150)) * time.Millisecond
	n.electionTimer = time.AfterFunc(timeout, n.startElection)
}

func (n *ConsensusNode) startElection() {
	n.mu.Lock()
	defer n.mu.Unlock()
	
	if !n.healthy || n.state == Leader {
		return
	}
	
	// Become candidate
	n.state = Candidate
	n.term++
	n.votedFor = n.id
	n.voteCount = 1 // Vote for self
	
	// Reset election timer
	n.resetElectionTimer()
	
	// Request votes from peers (simplified)
	go n.requestVotes()
}

func (n *ConsensusNode) requestVotes() {
	// Simplified vote request - in real implementation would be RPC calls
	votes := 1 // Self vote
	
	for _, peer := range n.peers {
		if peer.healthy && !peer.partitioned {
			if n.requestVoteFromPeer(peer) {
				votes++
			}
		}
	}
	
	// Check if won election
	quorum := len(n.peers)/2 + 1
	if votes >= quorum {
		n.becomeLeader()
	}
}

func (n *ConsensusNode) requestVoteFromPeer(peer *ConsensusNode) bool {
	peer.mu.Lock()
	defer peer.mu.Unlock()
	
	// Simplified vote logic
	if peer.term < n.term && (peer.votedFor == "" || peer.votedFor == n.id) {
		peer.term = n.term
		peer.votedFor = n.id
		return true
	}
	
	return false
}

func (n *ConsensusNode) becomeLeader() {
	n.mu.Lock()
	defer n.mu.Unlock()
	
	if n.state != Candidate {
		return
	}
	
	n.state = Leader
	n.leaderID = n.id
	
	// Initialize leader state
	for peerID := range n.peers {
		n.nextIndex[peerID] = int64(len(n.log)) + 1
		n.matchIndex[peerID] = 0
	}
	
	// Start sending heartbeats
	n.startHeartbeats()
}

func (n *ConsensusNode) startHeartbeats() {
	if n.heartbeatTimer != nil {
		n.heartbeatTimer.Stop()
	}
	
	n.heartbeatTimer = time.AfterFunc(50*time.Millisecond, func() {
		if n.IsLeader() {
			n.sendHeartbeats()
			n.startHeartbeats() // Schedule next heartbeat
		}
	})
}

func (n *ConsensusNode) sendHeartbeats() {
	for _, peer := range n.peers {
		if peer.healthy && !peer.partitioned {
			peer.receiveHeartbeat(n.id, n.term)
		}
	}
}

func (n *ConsensusNode) receiveHeartbeat(leaderID string, term int64) {
	n.mu.Lock()
	defer n.mu.Unlock()
	
	if term >= n.term {
		n.term = term
		n.state = Follower
		n.leaderID = leaderID
		n.resetElectionTimer()
	}
}

// Helper functions

func logsEqual(log1, log2 []LogEntry) bool {
	if len(log1) != len(log2) {
		return false
	}
	
	for i, entry1 := range log1 {
		entry2 := log2[i]
		if entry1.Index != entry2.Index || entry1.Term != entry2.Term || entry1.Data != entry2.Data {
			return false
		}
	}
	
	return true
}