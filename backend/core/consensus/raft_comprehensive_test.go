package consensus

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"
)

// TestSingleNodeElection tests that a single node becomes leader
func TestSingleNodeElection(t *testing.T) {
	transport := NewInMemoryTransport("node1")
	node := NewRaftNode("node1", []string{"node1"}, transport)
	transport.SetRaftNode(node)
	
	node.Start()
	defer node.Stop()
	
	// Wait for election
	time.Sleep(500 * time.Millisecond)
	
	// Check if node became leader
	term, isLeader := node.GetState()
	if !isLeader {
		t.Fatal("Single node should become leader")
	}
	
	if term < 1 {
		t.Fatalf("Expected term >= 1, got %d", term)
	}
	
	if node.GetLeader() != "node1" {
		t.Fatalf("Expected leader to be node1, got %s", node.GetLeader())
	}
}

// TestThreeNodeElection tests leader election with three nodes
func TestThreeNodeElection(t *testing.T) {
	nodes, transports := createCluster(3)
	defer stopCluster(nodes)
	
	// Start all nodes
	for _, node := range nodes {
		node.Start()
	}
	
	// Wait for election
	time.Sleep(1 * time.Second)
	
	// Check that exactly one leader was elected
	leaders := 0
	var leaderID string
	var leaderTerm int64
	
	for id, node := range nodes {
		term, isLeader := node.GetState()
		if isLeader {
			leaders++
			leaderID = id
			leaderTerm = term
		}
	}
	
	if leaders != 1 {
		t.Fatalf("Expected exactly 1 leader, got %d", leaders)
	}
	
	// Check all nodes agree on the leader
	for _, node := range nodes {
		if node.GetLeader() != leaderID {
			t.Fatalf("Node disagrees on leader: expected %s, got %s", 
				leaderID, node.GetLeader())
		}
	}
	
	t.Logf("Leader elected: %s at term %d", leaderID, leaderTerm)
}

// TestLogReplication tests that log entries are replicated across the cluster
func TestLogReplication(t *testing.T) {
	nodes, _ := createCluster(3)
	defer stopCluster(nodes)
	
	// Start all nodes
	for _, node := range nodes {
		node.Start()
	}
	
	// Wait for leader election
	time.Sleep(1 * time.Second)
	
	// Find the leader
	var leader *RaftNode
	for _, node := range nodes {
		if node.IsLeader() {
			leader = node
			break
		}
	}
	
	if leader == nil {
		t.Fatal("No leader found")
	}
	
	// Submit commands to the leader
	commands := []string{"cmd1", "cmd2", "cmd3"}
	indices := make([]int64, 0)
	
	for _, cmd := range commands {
		index, _, ok := leader.Submit(cmd)
		if !ok {
			t.Fatalf("Failed to submit command %s", cmd)
		}
		indices = append(indices, index)
		t.Logf("Submitted command %s at index %d", cmd, index)
	}
	
	// Wait for replication
	time.Sleep(500 * time.Millisecond)
	
	// Check that all nodes have the same log
	for id, node := range nodes {
		node.mu.RLock()
		logLen := len(node.log)
		commitIndex := node.commitIndex
		node.mu.RUnlock()
		
		if logLen < len(commands) {
			t.Errorf("Node %s has incomplete log: %d entries, expected >= %d", 
				id, logLen, len(commands))
		}
		
		t.Logf("Node %s: log length=%d, commit index=%d", id, logLen, commitIndex)
	}
	
	// Verify commands are applied
	appliedCommands := make(map[string][]interface{})
	for id, node := range nodes {
		applyCh := node.GetApplyChan()
		
		// Collect applied commands
		timeout := time.After(1 * time.Second)
		for i := 0; i < len(commands); i++ {
			select {
			case msg := <-applyCh:
				if msg.CommandValid {
					appliedCommands[id] = append(appliedCommands[id], msg.Command)
				}
			case <-timeout:
				t.Logf("Node %s: timeout waiting for command %d", id, i+1)
				break
			}
		}
	}
	
	// Verify all nodes applied the same commands
	for id, cmds := range appliedCommands {
		if len(cmds) != len(commands) {
			t.Errorf("Node %s applied %d commands, expected %d", 
				id, len(cmds), len(commands))
		}
		
		for i, cmd := range cmds {
			if cmd != commands[i] {
				t.Errorf("Node %s: command mismatch at index %d: got %v, expected %s",
					id, i, cmd, commands[i])
			}
		}
	}
}

// TestLeaderFailover tests that a new leader is elected when the current leader fails
func TestLeaderFailover(t *testing.T) {
	nodes, _ := createCluster(5)
	
	// Start all nodes
	for _, node := range nodes {
		node.Start()
	}
	
	// Wait for leader election
	time.Sleep(1 * time.Second)
	
	// Find and stop the leader
	var oldLeaderID string
	var oldLeaderTerm int64
	for id, node := range nodes {
		term, isLeader := node.GetState()
		if isLeader {
			oldLeaderID = id
			oldLeaderTerm = term
			t.Logf("Stopping leader %s at term %d", id, term)
			node.Stop()
			delete(nodes, id) // Remove from active nodes
			break
		}
	}
	
	if oldLeaderID == "" {
		t.Fatal("No initial leader found")
	}
	
	// Wait for new leader election
	time.Sleep(2 * time.Second)
	
	// Check that a new leader was elected
	var newLeaderID string
	var newLeaderTerm int64
	leaders := 0
	
	for id, node := range nodes {
		term, isLeader := node.GetState()
		if isLeader {
			leaders++
			newLeaderID = id
			newLeaderTerm = term
		}
	}
	
	if leaders != 1 {
		t.Fatalf("Expected exactly 1 new leader, got %d", leaders)
	}
	
	if newLeaderID == oldLeaderID {
		t.Fatal("New leader should be different from old leader")
	}
	
	if newLeaderTerm <= oldLeaderTerm {
		t.Fatalf("New term %d should be greater than old term %d", 
			newLeaderTerm, oldLeaderTerm)
	}
	
	t.Logf("New leader elected: %s at term %d", newLeaderID, newLeaderTerm)
	
	// Stop remaining nodes
	for _, node := range nodes {
		node.Stop()
	}
}

// TestNetworkPartition tests behavior during network partition
func TestNetworkPartition(t *testing.T) {
	// Create a 5-node cluster
	nodes, transports := createCluster(5)
	defer stopCluster(nodes)
	
	// Start all nodes
	for _, node := range nodes {
		node.Start()
	}
	
	// Wait for leader election
	time.Sleep(1 * time.Second)
	
	// Find the leader
	var leaderID string
	for id, node := range nodes {
		if node.IsLeader() {
			leaderID = id
			break
		}
	}
	
	if leaderID == "" {
		t.Fatal("No leader found")
	}
	
	t.Logf("Initial leader: %s", leaderID)
	
	// Create network partition: leader + 1 node vs 3 nodes
	partition1 := []string{leaderID}
	partition2 := []string{}
	
	// Find one more node for partition1 and rest for partition2
	for id := range nodes {
		if id != leaderID {
			if len(partition1) < 2 {
				partition1 = append(partition1, id)
			} else {
				partition2 = append(partition2, id)
			}
		}
	}
	
	t.Logf("Partition 1 (minority): %v", partition1)
	t.Logf("Partition 2 (majority): %v", partition2)
	
	// Simulate network partition
	for _, id1 := range partition1 {
		for _, id2 := range partition2 {
			// Disconnect nodes in different partitions
			disconnectNodes(transports[id1], transports[id2])
		}
	}
	
	// Wait for new election in majority partition
	time.Sleep(2 * time.Second)
	
	// Check that minority partition has no leader
	minorityLeaders := 0
	for _, id := range partition1 {
		if nodes[id].IsLeader() {
			minorityLeaders++
		}
	}
	
	if minorityLeaders > 0 {
		t.Errorf("Minority partition should have no leader, but has %d", minorityLeaders)
	}
	
	// Check that majority partition elected a new leader
	majorityLeaders := 0
	var newLeaderID string
	for _, id := range partition2 {
		if nodes[id].IsLeader() {
			majorityLeaders++
			newLeaderID = id
		}
	}
	
	if majorityLeaders != 1 {
		t.Fatalf("Majority partition should have exactly 1 leader, got %d", majorityLeaders)
	}
	
	t.Logf("New leader in majority partition: %s", newLeaderID)
	
	// Verify old leader stepped down
	if nodes[leaderID].IsLeader() {
		t.Error("Old leader in minority partition should have stepped down")
	}
}

// TestConcurrentSubmissions tests concurrent command submissions
func TestConcurrentSubmissions(t *testing.T) {
	nodes, _ := createCluster(3)
	defer stopCluster(nodes)
	
	// Start all nodes
	for _, node := range nodes {
		node.Start()
	}
	
	// Wait for leader election
	time.Sleep(1 * time.Second)
	
	// Find the leader
	var leader *RaftNode
	for _, node := range nodes {
		if node.IsLeader() {
			leader = node
			break
		}
	}
	
	if leader == nil {
		t.Fatal("No leader found")
	}
	
	// Submit commands concurrently
	numGoroutines := 10
	commandsPerGoroutine := 10
	var wg sync.WaitGroup
	successCount := make([]int, numGoroutines)
	
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			
			for j := 0; j < commandsPerGoroutine; j++ {
				cmd := fmt.Sprintf("goroutine-%d-cmd-%d", id, j)
				_, _, ok := leader.Submit(cmd)
				if ok {
					successCount[id]++
				}
				time.Sleep(10 * time.Millisecond)
			}
		}(i)
	}
	
	wg.Wait()
	
	// Calculate total successful submissions
	totalSuccess := 0
	for _, count := range successCount {
		totalSuccess += count
	}
	
	expectedTotal := numGoroutines * commandsPerGoroutine
	if totalSuccess < expectedTotal*8/10 { // Allow 20% failure rate
		t.Errorf("Too many failed submissions: %d/%d succeeded", 
			totalSuccess, expectedTotal)
	}
	
	t.Logf("Concurrent submissions: %d/%d succeeded", totalSuccess, expectedTotal)
	
	// Wait for replication
	time.Sleep(1 * time.Second)
	
	// Verify logs are consistent across nodes
	logLengths := make(map[string]int)
	for id, node := range nodes {
		node.mu.RLock()
		logLengths[id] = len(node.log)
		node.mu.RUnlock()
	}
	
	// All nodes should have similar log lengths (allowing some variance)
	minLen := totalSuccess
	maxLen := 0
	for id, length := range logLengths {
		t.Logf("Node %s log length: %d", id, length)
		if length < minLen {
			minLen = length
		}
		if length > maxLen {
			maxLen = length
		}
	}
	
	if maxLen-minLen > 10 {
		t.Errorf("Log lengths vary too much: min=%d, max=%d", minLen, maxLen)
	}
}

// TestSnapshotInstallation tests snapshot installation
func TestSnapshotInstallation(t *testing.T) {
	transport := NewInMemoryTransport("node1")
	node := NewRaftNode("node1", []string{"node1"}, transport)
	transport.SetRaftNode(node)
	
	// Manually set up some log entries
	node.log = []LogEntry{
		{Term: 1, Index: 1, Command: "cmd1"},
		{Term: 1, Index: 2, Command: "cmd2"},
		{Term: 2, Index: 3, Command: "cmd3"},
		{Term: 2, Index: 4, Command: "cmd4"},
	}
	node.commitIndex = 2
	node.lastApplied = 2
	
	// Create snapshot args
	snapshotArgs := &InstallSnapshotArgs{
		Term:              3,
		LeaderID:          "leader",
		LastIncludedIndex: 3,
		LastIncludedTerm:  2,
		Data:              []byte("snapshot-data"),
		Done:              true,
	}
	
	// Handle snapshot installation
	reply := node.HandleInstallSnapshot(snapshotArgs)
	
	// Verify reply
	if reply.Term != 3 {
		t.Errorf("Expected reply term 3, got %d", reply.Term)
	}
	
	// Verify log was truncated correctly
	if len(node.log) != 1 {
		t.Errorf("Expected 1 log entry after snapshot, got %d", len(node.log))
	}
	
	if node.log[0].Index != 4 {
		t.Errorf("Expected remaining log entry index 4, got %d", node.log[0].Index)
	}
	
	// Verify indices were updated
	if node.commitIndex != 3 {
		t.Errorf("Expected commit index 3, got %d", node.commitIndex)
	}
	
	if node.lastApplied != 3 {
		t.Errorf("Expected last applied 3, got %d", node.lastApplied)
	}
}

// Helper functions

func createCluster(size int) (map[string]*RaftNode, map[string]*InMemoryTransport) {
	nodes := make(map[string]*RaftNode)
	transports := make(map[string]*InMemoryTransport)
	peers := make([]string, size)
	
	// Create node IDs
	for i := 0; i < size; i++ {
		peers[i] = fmt.Sprintf("node%d", i+1)
	}
	
	// Create transports
	for _, id := range peers {
		transports[id] = NewInMemoryTransport(id)
	}
	
	// Connect all transports
	for _, t1 := range transports {
		for _, t2 := range transports {
			if t1 != t2 {
				t1.Connect(t2)
			}
		}
	}
	
	// Create nodes
	for _, id := range peers {
		node := NewRaftNode(id, peers, transports[id])
		transports[id].SetRaftNode(node)
		nodes[id] = node
	}
	
	return nodes, transports
}

func stopCluster(nodes map[string]*RaftNode) {
	for _, node := range nodes {
		node.Stop()
	}
}

func disconnectNodes(t1, t2 *InMemoryTransport) {
	t1.mu.Lock()
	delete(t1.nodes, t2.id)
	t1.mu.Unlock()
	
	t2.mu.Lock()
	delete(t2.nodes, t1.id)
	t2.mu.Unlock()
}

// Benchmark tests

func BenchmarkSingleNodeSubmission(b *testing.B) {
	transport := NewInMemoryTransport("node1")
	node := NewRaftNode("node1", []string{"node1"}, transport)
	transport.SetRaftNode(node)
	
	node.Start()
	defer node.Stop()
	
	// Wait for leader election
	time.Sleep(500 * time.Millisecond)
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		node.Submit(fmt.Sprintf("cmd-%d", i))
	}
}

func BenchmarkThreeNodeSubmission(b *testing.B) {
	nodes, _ := createCluster(3)
	defer stopCluster(nodes)
	
	// Start all nodes
	for _, node := range nodes {
		node.Start()
	}
	
	// Wait for leader election
	time.Sleep(1 * time.Second)
	
	// Find the leader
	var leader *RaftNode
	for _, node := range nodes {
		if node.IsLeader() {
			leader = node
			break
		}
	}
	
	if leader == nil {
		b.Fatal("No leader found")
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		leader.Submit(fmt.Sprintf("cmd-%d", i))
	}
}

func BenchmarkConcurrentSubmission(b *testing.B) {
	nodes, _ := createCluster(3)
	defer stopCluster(nodes)
	
	// Start all nodes
	for _, node := range nodes {
		node.Start()
	}
	
	// Wait for leader election
	time.Sleep(1 * time.Second)
	
	// Find the leader
	var leader *RaftNode
	for _, node := range nodes {
		if node.IsLeader() {
			leader = node
			break
		}
	}
	
	if leader == nil {
		b.Fatal("No leader found")
	}
	
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			leader.Submit(fmt.Sprintf("cmd-%d", i))
			i++
		}
	})
}