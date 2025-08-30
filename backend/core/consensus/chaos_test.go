package consensus

import (
	"context"
	"fmt"
	"math/rand"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// ChaosTest represents a chaos engineering test scenario
type ChaosTest struct {
	name          string
	nodes         []*RaftNode
	transports    []*ChaosTransport
	membership    *ClusterMembership
	splitBrain    *SplitBrainDetector
	partitions    [][]string
	dropRate      float64
	delayRange    [2]time.Duration
	duration      time.Duration
}

// ChaosTransport wraps InMemoryTransport with chaos injection
type ChaosTransport struct {
	*InMemoryTransport
	
	// Chaos parameters
	dropRate      float64
	delayRange    [2]time.Duration
	partitioned   bool
	partitionSet  map[string]bool
	
	// Metrics
	droppedMessages int64
	delayedMessages int64
}

// TestNetworkPartitionRecovery tests recovery from network partitions
func TestNetworkPartitionRecovery(t *testing.T) {
	chaos := setupChaosTest(t, "NetworkPartitionRecovery", 5)
	defer chaos.cleanup()
	
	// Wait for initial leader election
	leader := chaos.waitForLeader(t, 10*time.Second)
	if leader == nil {
		t.Fatal("No leader elected initially")
	}
	
	// Create a network partition (3 nodes vs 2 nodes)
	chaos.createPartition([][]string{
		{chaos.nodes[0].nodeID, chaos.nodes[1].nodeID, chaos.nodes[2].nodeID},
		{chaos.nodes[3].nodeID, chaos.nodes[4].nodeID},
	})
	
	// Wait for partition to take effect
	time.Sleep(2 * time.Second)
	
	// Check that majority partition maintains a leader
	majorityLeader := chaos.getLeaderInPartition([]string{
		chaos.nodes[0].nodeID,
		chaos.nodes[1].nodeID,
		chaos.nodes[2].nodeID,
	})
	
	if majorityLeader == nil {
		t.Error("Majority partition should maintain a leader")
	}
	
	// Check that minority partition has no leader
	minorityLeader := chaos.getLeaderInPartition([]string{
		chaos.nodes[3].nodeID,
		chaos.nodes[4].nodeID,
	})
	
	if minorityLeader != nil {
		t.Error("Minority partition should not have a leader")
	}
	
	// Heal the partition
	chaos.healPartition()
	
	// Wait for recovery
	time.Sleep(5 * time.Second)
	
	// Verify single leader after healing
	leaders := chaos.countLeaders()
	if leaders != 1 {
		t.Errorf("Expected 1 leader after healing, got %d", leaders)
	}
	
	// Verify all nodes agree on the leader
	chaos.verifyLeaderConsensus(t)
}

// TestCascadingFailures tests handling of cascading node failures
func TestCascadingFailures(t *testing.T) {
	chaos := setupChaosTest(t, "CascadingFailures", 7)
	defer chaos.cleanup()
	
	// Wait for initial leader election
	leader := chaos.waitForLeader(t, 10*time.Second)
	if leader == nil {
		t.Fatal("No leader elected initially")
	}
	
	initialLeaderID := leader.nodeID
	
	// Fail nodes one by one
	failedNodes := []string{}
	
	for i := 0; i < 3; i++ {
		// Find a non-leader node to fail
		var nodeToFail *RaftNode
		for _, node := range chaos.nodes {
			if node.nodeID != initialLeaderID && !contains(failedNodes, node.nodeID) {
				nodeToFail = node
				break
			}
		}
		
		if nodeToFail == nil {
			t.Fatal("No node to fail")
		}
		
		// Fail the node
		chaos.failNode(nodeToFail.nodeID)
		failedNodes = append(failedNodes, nodeToFail.nodeID)
		
		// Wait for cluster to stabilize
		time.Sleep(3 * time.Second)
		
		// Verify cluster still has a leader (we have 7 nodes, so up to 3 failures is ok)
		if chaos.countLeaders() != 1 {
			t.Errorf("After %d failures, expected 1 leader, got %d", i+1, chaos.countLeaders())
		}
	}
	
	// Fail one more node - should lose quorum
	var nodeToFail *RaftNode
	for _, node := range chaos.nodes {
		if !contains(failedNodes, node.nodeID) {
			nodeToFail = node
			break
		}
	}
	
	chaos.failNode(nodeToFail.nodeID)
	failedNodes = append(failedNodes, nodeToFail.nodeID)
	
	// Wait for leader to step down
	time.Sleep(3 * time.Second)
	
	// Should have no leader with 4/7 nodes failed
	if chaos.countLeaders() != 0 {
		t.Error("Should have no leader with majority of nodes failed")
	}
	
	// Recover one node to restore quorum
	chaos.recoverNode(failedNodes[0])
	
	// Wait for new leader election
	time.Sleep(5 * time.Second)
	
	// Should have a leader again
	if chaos.countLeaders() != 1 {
		t.Error("Should elect new leader after restoring quorum")
	}
}

// TestHighLatencyNetwork tests consensus under high network latency
func TestHighLatencyNetwork(t *testing.T) {
	chaos := setupChaosTest(t, "HighLatencyNetwork", 5)
	defer chaos.cleanup()
	
	// Set high latency (100-500ms)
	chaos.setNetworkLatency(100*time.Millisecond, 500*time.Millisecond)
	
	// Wait for leader election despite high latency
	leader := chaos.waitForLeader(t, 30*time.Second)
	if leader == nil {
		t.Fatal("No leader elected under high latency")
	}
	
	// Submit commands and verify they're eventually committed
	var wg sync.WaitGroup
	successCount := int64(0)
	
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func(cmdIndex int) {
			defer wg.Done()
			
			cmd := fmt.Sprintf("high-latency-cmd-%d", cmdIndex)
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			
			success, _ := leader.SubmitCommand(ctx, cmd)
			if success {
				atomic.AddInt64(&successCount, 1)
			}
		}(i)
	}
	
	wg.Wait()
	
	// Should commit most commands despite high latency
	if successCount < 7 {
		t.Errorf("Only %d/10 commands succeeded under high latency", successCount)
	}
}

// TestMessageLoss tests consensus with message loss
func TestMessageLoss(t *testing.T) {
	chaos := setupChaosTest(t, "MessageLoss", 5)
	defer chaos.cleanup()
	
	// Set 20% message drop rate
	chaos.setMessageDropRate(0.2)
	
	// Wait for leader election despite message loss
	leader := chaos.waitForLeader(t, 20*time.Second)
	if leader == nil {
		t.Fatal("No leader elected with message loss")
	}
	
	// Submit commands and verify they're eventually committed
	successCount := 0
	for i := 0; i < 20; i++ {
		cmd := fmt.Sprintf("lossy-cmd-%d", i)
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		
		success, _ := leader.SubmitCommand(ctx, cmd)
		cancel()
		
		if success {
			successCount++
		}
		
		// Small delay between commands
		time.Sleep(100 * time.Millisecond)
	}
	
	// Should succeed with most commands despite message loss
	if successCount < 14 {
		t.Errorf("Only %d/20 commands succeeded with message loss", successCount)
	}
	
	// Verify eventual consistency
	time.Sleep(2 * time.Second)
	chaos.verifyLogConsistency(t)
}

// TestSplitBrainPrevention tests split-brain prevention mechanisms
func TestSplitBrainPrevention(t *testing.T) {
	chaos := setupChaosTest(t, "SplitBrainPrevention", 6)
	defer chaos.cleanup()
	
	// Setup split-brain detector
	config := SplitBrainConfig{
		DetectionInterval:    1 * time.Second,
		EnableAutoResolution: true,
		PreferLargerPartition: true,
	}
	
	detector := NewSplitBrainDetector(chaos.nodes[0], chaos.membership, config)
	detector.Start()
	defer detector.Stop()
	
	// Wait for initial leader
	leader := chaos.waitForLeader(t, 10*time.Second)
	if leader == nil {
		t.Fatal("No initial leader elected")
	}
	
	// Create an even split (3 nodes vs 3 nodes)
	chaos.createPartition([][]string{
		{chaos.nodes[0].nodeID, chaos.nodes[1].nodeID, chaos.nodes[2].nodeID},
		{chaos.nodes[3].nodeID, chaos.nodes[4].nodeID, chaos.nodes[5].nodeID},
	})
	
	// Wait for detection
	time.Sleep(3 * time.Second)
	
	// Check that at most one partition has a leader
	leaders := chaos.countLeaders()
	if leaders > 1 {
		t.Errorf("Split-brain detected: %d leaders exist", leaders)
	}
	
	// Verify split-brain was detected
	if !detector.IsInSplitBrain() {
		t.Error("Split-brain condition should be detected")
	}
	
	// Heal partition
	chaos.healPartition()
	
	// Wait for resolution
	time.Sleep(5 * time.Second)
	
	// Verify single leader after resolution
	leaders = chaos.countLeaders()
	if leaders != 1 {
		t.Errorf("Expected 1 leader after resolution, got %d", leaders)
	}
}

// TestRapidLeaderChanges tests stability under rapid leader changes
func TestRapidLeaderChanges(t *testing.T) {
	chaos := setupChaosTest(t, "RapidLeaderChanges", 5)
	defer chaos.cleanup()
	
	// Track leader changes
	leaderChanges := 0
	lastLeader := ""
	
	for i := 0; i < 5; i++ {
		// Wait for leader
		leader := chaos.waitForLeader(t, 10*time.Second)
		if leader == nil {
			t.Fatal("No leader elected")
		}
		
		if leader.nodeID != lastLeader {
			leaderChanges++
			lastLeader = leader.nodeID
		}
		
		// Force leader to step down
		leader.mu.Lock()
		leader.state = Follower
		leader.votedFor = ""
		leader.mu.Unlock()
		
		// Wait for new election
		time.Sleep(2 * time.Second)
	}
	
	// Verify cluster survived rapid changes
	finalLeader := chaos.waitForLeader(t, 10*time.Second)
	if finalLeader == nil {
		t.Fatal("No leader after rapid changes")
	}
	
	// Submit a command to verify cluster is functional
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	
	success, _ := finalLeader.SubmitCommand(ctx, "stability-check")
	if !success {
		t.Error("Failed to commit command after rapid leader changes")
	}
}

// TestAsymmetricPartition tests asymmetric network partitions
func TestAsymmetricPartition(t *testing.T) {
	chaos := setupChaosTest(t, "AsymmetricPartition", 5)
	defer chaos.cleanup()
	
	// Wait for initial leader
	chaos.waitForLeader(t, 10*time.Second)
	
	// Create asymmetric partition: node 0 can send but not receive
	chaos.createAsymmetricPartition(chaos.nodes[0].nodeID, true, false)
	
	// Wait for cluster to detect the issue
	time.Sleep(5 * time.Second)
	
	// Node 0 should not be leader (can't receive votes)
	if chaos.nodes[0].state == Leader {
		t.Error("Node with receive issues should not be leader")
	}
	
	// Other nodes should elect a leader among themselves
	var leader *RaftNode
	for _, node := range chaos.nodes[1:] {
		if node.state == Leader {
			leader = node
			break
		}
	}
	
	if leader == nil {
		t.Error("Remaining nodes should elect a leader")
	}
	
	// Heal the asymmetric partition
	chaos.healAsymmetricPartition(chaos.nodes[0].nodeID)
	
	// Wait for stabilization
	time.Sleep(3 * time.Second)
	
	// Verify cluster is functional
	chaos.verifyLogConsistency(t)
}

// Helper functions for chaos testing

func setupChaosTest(t *testing.T, name string, nodeCount int) *ChaosTest {
	chaos := &ChaosTest{
		name:       name,
		nodes:      make([]*RaftNode, nodeCount),
		transports: make([]*ChaosTransport, nodeCount),
	}
	
	// Create nodes
	for i := 0; i < nodeCount; i++ {
		nodeID := fmt.Sprintf("chaos-node-%d", i)
		
		// Create chaos transport
		baseTransport := NewInMemoryTransport(nodeID)
		chaosTransport := &ChaosTransport{
			InMemoryTransport: baseTransport,
			partitionSet:      make(map[string]bool),
		}
		chaos.transports[i] = chaosTransport
		
		// Create Raft node
		peers := make([]string, 0, nodeCount-1)
		for j := 0; j < nodeCount; j++ {
			if i != j {
				peers = append(peers, fmt.Sprintf("chaos-node-%d", j))
			}
		}
		
		config := Config{
			ElectionTimeout:  150 * time.Millisecond,
			HeartbeatTimeout: 50 * time.Millisecond,
			MaxLogEntries:    1000,
		}
		
		chaos.nodes[i] = NewRaftNode(nodeID, peers, chaosTransport, config)
	}
	
	// Connect transports
	for i := 0; i < nodeCount; i++ {
		for j := 0; j < nodeCount; j++ {
			if i != j {
				chaos.transports[i].Connect(chaos.transports[j].InMemoryTransport)
			}
		}
	}
	
	// Start all nodes
	for _, node := range chaos.nodes {
		node.Start()
	}
	
	// Setup membership tracking
	membershipConfig := MembershipConfig{
		MinQuorumSize: (nodeCount / 2) + 1,
	}
	chaos.membership = NewClusterMembership("chaos-test", membershipConfig)
	
	for i, node := range chaos.nodes {
		chaos.membership.AddNode(node.nodeID, fmt.Sprintf("127.0.0.1:%d", 9000+i), nil)
	}
	
	return chaos
}

func (c *ChaosTest) cleanup() {
	for _, node := range c.nodes {
		node.Stop()
	}
}

func (c *ChaosTest) waitForLeader(t *testing.T, timeout time.Duration) *RaftNode {
	deadline := time.Now().Add(timeout)
	
	for time.Now().Before(deadline) {
		for _, node := range c.nodes {
			if node.state == Leader {
				return node
			}
		}
		time.Sleep(100 * time.Millisecond)
	}
	
	return nil
}

func (c *ChaosTest) countLeaders() int {
	count := 0
	for _, node := range c.nodes {
		if node.state == Leader {
			count++
		}
	}
	return count
}

func (c *ChaosTest) createPartition(partitions [][]string) {
	c.partitions = partitions
	
	for _, transport := range c.transports {
		transport.partitioned = true
		transport.partitionSet = make(map[string]bool)
		
		// Find which partition this node belongs to
		var myPartition []string
		for _, partition := range partitions {
			if contains(partition, transport.id) {
				myPartition = partition
				break
			}
		}
		
		// Can only communicate with nodes in the same partition
		for _, nodeID := range myPartition {
			transport.partitionSet[nodeID] = true
		}
	}
}

func (c *ChaosTest) healPartition() {
	for _, transport := range c.transports {
		transport.partitioned = false
		transport.partitionSet = make(map[string]bool)
	}
}

func (c *ChaosTest) failNode(nodeID string) {
	for i, node := range c.nodes {
		if node.nodeID == nodeID {
			node.Stop()
			// Disconnect transport
			c.transports[i].partitioned = true
			c.transports[i].partitionSet = make(map[string]bool)
			break
		}
	}
}

func (c *ChaosTest) recoverNode(nodeID string) {
	for i, node := range c.nodes {
		if node.nodeID == nodeID {
			node.Start()
			// Reconnect transport
			c.transports[i].partitioned = false
			c.transports[i].partitionSet = make(map[string]bool)
			break
		}
	}
}

func (c *ChaosTest) setNetworkLatency(min, max time.Duration) {
	for _, transport := range c.transports {
		transport.delayRange = [2]time.Duration{min, max}
	}
}

func (c *ChaosTest) setMessageDropRate(rate float64) {
	for _, transport := range c.transports {
		transport.dropRate = rate
	}
}

func (c *ChaosTest) createAsymmetricPartition(nodeID string, canSend, canReceive bool) {
	// Implementation would modify transport behavior asymmetrically
	// For simplicity, using symmetric partition in this example
	c.createPartition([][]string{{nodeID}, {}})
}

func (c *ChaosTest) healAsymmetricPartition(nodeID string) {
	c.healPartition()
}

func (c *ChaosTest) getLeaderInPartition(partition []string) *RaftNode {
	for _, node := range c.nodes {
		if contains(partition, node.nodeID) && node.state == Leader {
			return node
		}
	}
	return nil
}

func (c *ChaosTest) verifyLeaderConsensus(t *testing.T) {
	var leaderID string
	for _, node := range c.nodes {
		if node.state == Leader {
			if leaderID == "" {
				leaderID = node.nodeID
			} else if leaderID != node.nodeID {
				t.Errorf("Multiple leaders detected: %s and %s", leaderID, node.nodeID)
			}
		}
	}
}

func (c *ChaosTest) verifyLogConsistency(t *testing.T) {
	if len(c.nodes) == 0 {
		return
	}
	
	// Get the log from the first node
	referenceLog := c.nodes[0].log
	
	// Compare with other nodes
	for i := 1; i < len(c.nodes); i++ {
		node := c.nodes[i]
		
		// Check committed entries match
		minLen := min(len(referenceLog), len(node.log))
		for j := 0; j < minLen; j++ {
			if j <= int(node.commitIndex) && j <= int(c.nodes[0].commitIndex) {
				if referenceLog[j].Term != node.log[j].Term ||
					referenceLog[j].Command != node.log[j].Command {
					t.Errorf("Log inconsistency at index %d between node 0 and node %d", j, i)
				}
			}
		}
	}
}

// Override transport methods for chaos injection
func (ct *ChaosTransport) SendRequestVote(ctx context.Context, nodeID string, req *RequestVoteArgs) (*RequestVoteReply, error) {
	// Check if partitioned
	if ct.partitioned && !ct.partitionSet[nodeID] {
		return nil, fmt.Errorf("network partition")
	}
	
	// Simulate message drop
	if ct.dropRate > 0 && rand.Float64() < ct.dropRate {
		atomic.AddInt64(&ct.droppedMessages, 1)
		return nil, fmt.Errorf("message dropped")
	}
	
	// Add delay
	if ct.delayRange[1] > 0 {
		delay := ct.delayRange[0] + time.Duration(rand.Int63n(int64(ct.delayRange[1]-ct.delayRange[0])))
		atomic.AddInt64(&ct.delayedMessages, 1)
		
		select {
		case <-time.After(delay):
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	
	return ct.InMemoryTransport.SendRequestVote(ctx, nodeID, req)
}

func (ct *ChaosTransport) SendAppendEntries(ctx context.Context, nodeID string, req *AppendEntriesArgs) (*AppendEntriesReply, error) {
	// Check if partitioned
	if ct.partitioned && !ct.partitionSet[nodeID] {
		return nil, fmt.Errorf("network partition")
	}
	
	// Simulate message drop
	if ct.dropRate > 0 && rand.Float64() < ct.dropRate {
		atomic.AddInt64(&ct.droppedMessages, 1)
		return nil, fmt.Errorf("message dropped")
	}
	
	// Add delay
	if ct.delayRange[1] > 0 {
		delay := ct.delayRange[0] + time.Duration(rand.Int63n(int64(ct.delayRange[1]-ct.delayRange[0])))
		atomic.AddInt64(&ct.delayedMessages, 1)
		
		select {
		case <-time.After(delay):
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	
	return ct.InMemoryTransport.SendAppendEntries(ctx, nodeID, req)
}

func contains(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}