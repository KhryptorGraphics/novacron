package consensus

import (
	"testing"
	"time"
)

func TestRaftNode_Creation(t *testing.T) {
	peers := []string{"node1", "node2", "node3"}
	transport := NewInMemoryTransport("node1")
	
	node := NewRaftNode("node1", peers, transport)
	
	if node.id != "node1" {
		t.Errorf("Expected node ID to be 'node1', got %s", node.id)
	}
	
	if node.state != Follower {
		t.Errorf("Expected initial state to be Follower, got %s", node.state)
	}
	
	if node.currentTerm != 0 {
		t.Errorf("Expected initial term to be 0, got %d", node.currentTerm)
	}
	
	if len(node.peers) != 3 {
		t.Errorf("Expected 3 peers, got %d", len(node.peers))
	}
}

func TestRaftNode_SingleNodeElection(t *testing.T) {
	// Create a single-node cluster
	peers := []string{"node1"}
	transport := NewInMemoryTransport("node1")
	
	node := NewRaftNode("node1", peers, transport)
	transport.SetRaftNode(node)
	
	node.Start()
	defer node.Stop()
	
	// Wait for election timeout and leader election
	time.Sleep(350 * time.Millisecond)
	
	if !node.IsLeader() {
		t.Error("Single node should become leader")
	}
	
	term, isLeader := node.GetState()
	if !isLeader {
		t.Error("GetState should return isLeader=true")
	}
	if term <= 0 {
		t.Errorf("Expected positive term, got %d", term)
	}
}

func TestRaftNode_ThreeNodeElection(t *testing.T) {
	// Create a 3-node cluster
	nodes := make([]*RaftNode, 3)
	transports := make([]*InMemoryTransport, 3)
	
	peers := []string{"node1", "node2", "node3"}
	
	// Create nodes and transports
	for i := 0; i < 3; i++ {
		nodeID := peers[i]
		transports[i] = NewInMemoryTransport(nodeID)
		nodes[i] = NewRaftNode(nodeID, peers, transports[i])
		transports[i].SetRaftNode(nodes[i])
	}
	
	// Connect transports
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			if i != j {
				transports[i].Connect(transports[j])
			}
		}
	}
	
	// Start all nodes
	for i := 0; i < 3; i++ {
		nodes[i].Start()
	}
	defer func() {
		for i := 0; i < 3; i++ {
			nodes[i].Stop()
		}
	}()
	
	// Wait for leader election
	time.Sleep(500 * time.Millisecond)
	
	// Count leaders
	leaders := 0
	var leaderNode *RaftNode
	for i := 0; i < 3; i++ {
		if nodes[i].IsLeader() {
			leaders++
			leaderNode = nodes[i]
		}
	}
	
	if leaders != 1 {
		t.Errorf("Expected exactly 1 leader, got %d", leaders)
	}
	
	if leaderNode == nil {
		t.Fatal("No leader found")
	}
	
	// Check that all nodes agree on the leader
	leaderID := leaderNode.id
	for i := 0; i < 3; i++ {
		if nodes[i].GetLeader() != leaderID {
			t.Errorf("Node %s thinks leader is %s, but actual leader is %s",
				nodes[i].id, nodes[i].GetLeader(), leaderID)
		}
	}
}

func TestRaftNode_LogReplication(t *testing.T) {
	// Create a 3-node cluster
	nodes := make([]*RaftNode, 3)
	transports := make([]*InMemoryTransport, 3)
	
	peers := []string{"node1", "node2", "node3"}
	
	// Create nodes and transports
	for i := 0; i < 3; i++ {
		nodeID := peers[i]
		transports[i] = NewInMemoryTransport(nodeID)
		nodes[i] = NewRaftNode(nodeID, peers, transports[i])
		transports[i].SetRaftNode(nodes[i])
	}
	
	// Connect transports
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			if i != j {
				transports[i].Connect(transports[j])
			}
		}
	}
	
	// Start all nodes
	for i := 0; i < 3; i++ {
		nodes[i].Start()
	}
	defer func() {
		for i := 0; i < 3; i++ {
			nodes[i].Stop()
		}
	}()
	
	// Wait for leader election
	time.Sleep(500 * time.Millisecond)
	
	// Find the leader
	var leader *RaftNode
	for i := 0; i < 3; i++ {
		if nodes[i].IsLeader() {
			leader = nodes[i]
			break
		}
	}
	
	if leader == nil {
		t.Fatal("No leader found")
	}
	
	// Submit commands
	commands := []string{"command1", "command2", "command3"}
	for _, cmd := range commands {
		index, term, ok := leader.Submit(cmd)
		if !ok {
			t.Errorf("Failed to submit command: %s", cmd)
		}
		if index <= 0 {
			t.Errorf("Expected positive index for command %s, got %d", cmd, index)
		}
		if term <= 0 {
			t.Errorf("Expected positive term for command %s, got %d", cmd, term)
		}
	}
	
	// Wait for replication
	time.Sleep(200 * time.Millisecond)
	
	// Check that all nodes have the same log length
	expectedLogLength := len(commands)
	for i := 0; i < 3; i++ {
		nodes[i].mu.RLock()
		logLength := len(nodes[i].log)
		nodes[i].mu.RUnlock()
		
		if logLength != expectedLogLength {
			t.Errorf("Node %s has log length %d, expected %d",
				nodes[i].id, logLength, expectedLogLength)
		}
	}
}

func TestRaftNode_LeaderFailover(t *testing.T) {
	// Create a 3-node cluster
	nodes := make([]*RaftNode, 3)
	transports := make([]*InMemoryTransport, 3)
	
	peers := []string{"node1", "node2", "node3"}
	
	// Create nodes and transports
	for i := 0; i < 3; i++ {
		nodeID := peers[i]
		transports[i] = NewInMemoryTransport(nodeID)
		nodes[i] = NewRaftNode(nodeID, peers, transports[i])
		transports[i].SetRaftNode(nodes[i])
	}
	
	// Connect transports
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			if i != j {
				transports[i].Connect(transports[j])
			}
		}
	}
	
	// Start all nodes
	for i := 0; i < 3; i++ {
		nodes[i].Start()
	}
	defer func() {
		for i := 0; i < 3; i++ {
			nodes[i].Stop()
		}
	}()
	
	// Wait for initial leader election
	time.Sleep(500 * time.Millisecond)
	
	// Find the initial leader
	var initialLeader *RaftNode
	initialLeaderIndex := -1
	for i := 0; i < 3; i++ {
		if nodes[i].IsLeader() {
			initialLeader = nodes[i]
			initialLeaderIndex = i
			break
		}
	}
	
	if initialLeader == nil {
		t.Fatal("No initial leader found")
	}
	
	// Submit a command to establish log entry
	_, _, ok := initialLeader.Submit("test_command")
	if !ok {
		t.Error("Failed to submit command to initial leader")
	}
	
	// Wait for replication
	time.Sleep(100 * time.Millisecond)
	
	// Stop the leader
	initialLeader.Stop()
	
	// Wait for new leader election (increased timeout for failover)
	time.Sleep(1500 * time.Millisecond)
	
	// Check that a new leader was elected
	newLeaderCount := 0
	var newLeader *RaftNode
	for i := 0; i < 3; i++ {
		if i != initialLeaderIndex && nodes[i].IsLeader() {
			newLeaderCount++
			newLeader = nodes[i]
		}
	}
	
	if newLeaderCount != 1 {
		t.Errorf("Expected exactly 1 new leader after failover, got %d", newLeaderCount)
	}
	
	if newLeader == nil {
		t.Error("No new leader elected after failover")
		return
	}
	
	// Verify the new leader can accept commands
	_, _, ok = newLeader.Submit("post_failover_command")
	if !ok {
		t.Error("New leader cannot accept commands")
	}
}

func TestRaftNode_RequestVote(t *testing.T) {
	peers := []string{"node1", "node2"}
	transport := NewInMemoryTransport("node1")
	
	node := NewRaftNode("node1", peers, transport)
	
	// Test voting for a candidate with up-to-date log
	args := &RequestVoteArgs{
		Term:         1,
		CandidateID:  "node2",
		LastLogIndex: 0,
		LastLogTerm:  0,
	}
	
	reply := node.HandleRequestVote(args)
	
	if !reply.VoteGranted {
		t.Error("Should grant vote to candidate with up-to-date log")
	}
	
	if reply.Term != 1 {
		t.Errorf("Expected term 1 in reply, got %d", reply.Term)
	}
	
	// Test not voting for same term again
	args2 := &RequestVoteArgs{
		Term:         1,
		CandidateID:  "node3",
		LastLogIndex: 0,
		LastLogTerm:  0,
	}
	
	reply2 := node.HandleRequestVote(args2)
	
	if reply2.VoteGranted {
		t.Error("Should not grant vote to different candidate in same term")
	}
}

func TestRaftNode_AppendEntries(t *testing.T) {
	peers := []string{"node1", "node2"}
	transport := NewInMemoryTransport("node1")
	
	node := NewRaftNode("node1", peers, transport)
	
	// Test heartbeat (empty entries)
	args := &AppendEntriesArgs{
		Term:         1,
		LeaderID:     "node2",
		PrevLogIndex: 0,
		PrevLogTerm:  0,
		Entries:      []LogEntry{},
		LeaderCommit: 0,
	}
	
	reply := node.HandleAppendEntries(args)
	
	if !reply.Success {
		t.Error("Heartbeat should succeed")
	}
	
	if reply.Term != 1 {
		t.Errorf("Expected term 1 in reply, got %d", reply.Term)
	}
	
	if node.GetLeader() != "node2" {
		t.Errorf("Expected leader to be node2, got %s", node.GetLeader())
	}
	
	// Test appending entries
	entries := []LogEntry{
		{Term: 1, Index: 1, Command: "test_command_1"},
		{Term: 1, Index: 2, Command: "test_command_2"},
	}
	
	args2 := &AppendEntriesArgs{
		Term:         1,
		LeaderID:     "node2",
		PrevLogIndex: 0,
		PrevLogTerm:  0,
		Entries:      entries,
		LeaderCommit: 2,
	}
	
	reply2 := node.HandleAppendEntries(args2)
	
	if !reply2.Success {
		t.Error("Append entries should succeed")
	}
	
	// Check that entries were added
	node.mu.RLock()
	logLength := len(node.log)
	commitIndex := node.commitIndex
	node.mu.RUnlock()
	
	if logLength != 2 {
		t.Errorf("Expected log length 2, got %d", logLength)
	}
	
	if commitIndex != 2 {
		t.Errorf("Expected commit index 2, got %d", commitIndex)
	}
}

func TestRaftNode_Statistics(t *testing.T) {
	peers := []string{"node1"}
	transport := NewInMemoryTransport("node1")
	
	node := NewRaftNode("node1", peers, transport)
	transport.SetRaftNode(node)
	
	node.Start()
	defer node.Stop()
	
	// Wait for election
	time.Sleep(350 * time.Millisecond)
	
	stats := node.GetStats()
	
	if stats.ElectionsWon != 1 {
		t.Errorf("Expected 1 election won, got %d", stats.ElectionsWon)
	}
	
	if stats.TermsLeader != 1 {
		t.Errorf("Expected 1 term as leader, got %d", stats.TermsLeader)
	}
	
	if stats.LastLeaderElection.IsZero() {
		t.Error("Last leader election time should be set")
	}
}

func TestRaftNode_ApplyChannel(t *testing.T) {
	peers := []string{"node1"}
	transport := NewInMemoryTransport("node1")
	
	node := NewRaftNode("node1", peers, transport)
	transport.SetRaftNode(node)
	
	node.Start()
	defer node.Stop()
	
	// Wait for election
	time.Sleep(350 * time.Millisecond)
	
	// Submit commands
	commands := []string{"cmd1", "cmd2", "cmd3"}
	for _, cmd := range commands {
		node.Submit(cmd)
	}
	
	// Read applied messages
	applyCh := node.GetApplyChan()
	appliedCommands := make([]string, 0)
	
	timeout := time.NewTimer(500 * time.Millisecond)
	defer timeout.Stop()
	
	for len(appliedCommands) < len(commands) {
		select {
		case msg := <-applyCh:
			if msg.CommandValid {
				appliedCommands = append(appliedCommands, msg.Command.(string))
			}
		case <-timeout.C:
			t.Fatal("Timeout waiting for commands to be applied")
		}
	}
	
	if len(appliedCommands) != len(commands) {
		t.Errorf("Expected %d applied commands, got %d", len(commands), len(appliedCommands))
	}
	
	// Check that commands are in order
	for i, expectedCmd := range commands {
		if appliedCommands[i] != expectedCmd {
			t.Errorf("Expected command %s at index %d, got %s", expectedCmd, i, appliedCommands[i])
		}
	}
}

func TestRaftNode_NetworkPartition(t *testing.T) {
	// Create a 5-node cluster to test network partitions
	nodes := make([]*RaftNode, 5)
	transports := make([]*InMemoryTransport, 5)
	
	peers := []string{"node1", "node2", "node3", "node4", "node5"}
	
	// Create nodes and transports
	for i := 0; i < 5; i++ {
		nodeID := peers[i]
		transports[i] = NewInMemoryTransport(nodeID)
		nodes[i] = NewRaftNode(nodeID, peers, transports[i])
		transports[i].SetRaftNode(nodes[i])
	}
	
	// Connect transports (full mesh)
	for i := 0; i < 5; i++ {
		for j := 0; j < 5; j++ {
			if i != j {
				transports[i].Connect(transports[j])
			}
		}
	}
	
	// Start all nodes
	for i := 0; i < 5; i++ {
		nodes[i].Start()
	}
	defer func() {
		for i := 0; i < 5; i++ {
			nodes[i].Stop()
		}
	}()
	
	// Wait for initial leader election
	time.Sleep(600 * time.Millisecond)
	
	// Find initial leader
	var initialLeader *RaftNode
	for i := 0; i < 5; i++ {
		if nodes[i].IsLeader() {
			initialLeader = nodes[i]
			break
		}
	}
	
	if initialLeader == nil {
		t.Fatal("No initial leader found")
	}
	
	// Submit initial command
	_, _, ok := initialLeader.Submit("before_partition")
	if !ok {
		t.Error("Failed to submit command before partition")
	}
	
	time.Sleep(100 * time.Millisecond)
	
	// Create network partition: isolate nodes 0 and 1 from nodes 2, 3, 4
	// Clear connections for nodes 0 and 1 to nodes 2, 3, 4
	for i := 0; i < 2; i++ {
		transports[i].mu.Lock()
		newNodes := make(map[string]*InMemoryTransport)
		for j := 0; j < 2; j++ {
			if i != j {
				newNodes[peers[j]] = transports[j]
			}
		}
		transports[i].nodes = newNodes
		transports[i].mu.Unlock()
	}
	
	// Clear connections for nodes 2, 3, 4 to nodes 0, 1
	for i := 2; i < 5; i++ {
		transports[i].mu.Lock()
		newNodes := make(map[string]*InMemoryTransport)
		for j := 2; j < 5; j++ {
			if i != j {
				newNodes[peers[j]] = transports[j]
			}
		}
		transports[i].nodes = newNodes
		transports[i].mu.Unlock()
	}
	
	// Wait for new election in majority partition
	time.Sleep(1000 * time.Millisecond)
	
	// Check that majority partition (nodes 2, 3, 4) has a leader
	majorityLeaderCount := 0
	var majorityLeader *RaftNode
	for i := 2; i < 5; i++ {
		if nodes[i].IsLeader() {
			majorityLeaderCount++
			majorityLeader = nodes[i]
		}
	}
	
	if majorityLeaderCount != 1 {
		t.Errorf("Expected 1 leader in majority partition, got %d", majorityLeaderCount)
	}
	
	// Check that minority partition (nodes 0, 1) has no leader
	minorityLeaderCount := 0
	for i := 0; i < 2; i++ {
		if nodes[i].IsLeader() {
			minorityLeaderCount++
		}
	}
	
	if minorityLeaderCount != 0 {
		t.Errorf("Expected 0 leaders in minority partition, got %d", minorityLeaderCount)
	}
	
	// Submit command to majority leader
	if majorityLeader != nil {
		_, _, ok := majorityLeader.Submit("during_partition")
		if !ok {
			t.Error("Majority leader should be able to accept commands")
		}
	}
}