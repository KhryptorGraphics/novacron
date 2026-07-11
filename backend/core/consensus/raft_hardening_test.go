package consensus

import (
	"testing"
	"time"
)

// waitForLeader blocks until n reports itself as leader, or fails the test
// after a generous timeout. Used by tests below that need a real election to
// complete before exercising persistence/detection/shutdown behavior.
func waitForLeader(t *testing.T, n *RaftNode) {
	t.Helper()
	deadline := time.After(3 * time.Second)
	for {
		if n.IsLeader() {
			return
		}
		select {
		case <-deadline:
			t.Fatalf("node %s never became leader", n.nodeID)
		case <-time.After(10 * time.Millisecond):
		}
	}
}

// TestRaftNode_PersistenceSurvivesRestart_NoDoubleVote proves currentTerm,
// votedFor, and the log are reloaded from durable storage on restart, and
// that a restarted node cannot grant a second vote in a term it already
// voted in. Without loading persisted state (raft.go previously kept
// currentTerm/votedFor/log in memory only), a fresh node would start at
// term 0 / votedFor "" and this test's final assertion -- the vote-granted
// check -- would fail (a double-vote would be granted).
func TestRaftNode_PersistenceSurvivesRestart_NoDoubleVote(t *testing.T) {
	dir := t.TempDir()

	storage1, err := NewFileRaftStorage(dir)
	if err != nil {
		t.Fatalf("failed to create file storage: %v", err)
	}

	peers := []string{"node1"}
	node1 := NewRaftNodeWithStorage("node1", peers, NewInMemoryTransport("node1"), storage1)
	node1.Start()

	waitForLeader(t, node1)

	if _, _, ok := node1.Submit("committed-command"); !ok {
		t.Fatal("expected leader to accept Submit")
	}

	// Let the (synchronous, in-lock) persistence writes from the election
	// and the Submit land before we read state back out.
	time.Sleep(50 * time.Millisecond)

	node1.mu.RLock()
	termBeforeRestart := node1.currentTerm
	votedForBeforeRestart := node1.votedFor
	logLenBeforeRestart := len(node1.log)
	node1.mu.RUnlock()

	if termBeforeRestart == 0 {
		t.Fatal("expected currentTerm > 0 after a completed election")
	}
	if votedForBeforeRestart != "node1" {
		t.Fatalf("expected node1 to have voted for itself, got %q", votedForBeforeRestart)
	}
	if logLenBeforeRestart != 1 {
		t.Fatalf("expected 1 committed log entry before restart, got %d", logLenBeforeRestart)
	}

	// Simulate a process restart: Stop() closes storage1's file handles,
	// then a fresh FileRaftStorage + RaftNode are created over the SAME
	// on-disk directory, exactly as a real process restart would do.
	node1.Stop()

	storage2, err := NewFileRaftStorage(dir)
	if err != nil {
		t.Fatalf("failed to reopen file storage after restart: %v", err)
	}
	defer storage2.Close()

	restarted := NewRaftNodeWithStorage("node1", peers, NewInMemoryTransport("node1"), storage2)

	restarted.mu.RLock()
	term := restarted.currentTerm
	votedFor := restarted.votedFor
	logLen := len(restarted.log)
	restarted.mu.RUnlock()

	if term != termBeforeRestart {
		t.Errorf("currentTerm did not survive restart: got %d, want %d", term, termBeforeRestart)
	}
	if votedFor != votedForBeforeRestart {
		t.Errorf("votedFor did not survive restart: got %q, want %q", votedFor, votedForBeforeRestart)
	}
	if logLen != logLenBeforeRestart {
		t.Fatalf("log did not survive restart: got %d entries, want %d", logLen, logLenBeforeRestart)
	}
	if restarted.log[0].Command != "committed-command" {
		t.Errorf("restored log entry command = %v, want %q", restarted.log[0].Command, "committed-command")
	}

	// The safety property under test: a restarted node must remember it
	// already voted this term and refuse a second candidate's request for
	// the same term (double-vote). If persisted state were not reloaded,
	// votedFor would read back as "" here and this vote would be wrongly
	// granted.
	reply := restarted.HandleRequestVote(&RequestVoteArgs{
		Term:         term,
		CandidateID:  "some-other-candidate",
		LastLogIndex: int64(logLen),
		LastLogTerm:  term,
	})
	if reply.VoteGranted {
		t.Fatal("restarted node granted a second vote in a term it already voted in (double-vote)")
	}
}

// TestSplitBrainDetector_DetectsMultipleRealLeaders proves checkMultipleLeaders
// observes a genuine multi-leader partition through isNodeLeader's remote
// QueryLeaderState RPC. nodeB is a fully independent, real RaftNode reachable
// only over an InMemoryTransport -- not a stub -- so this exercises the real
// transport round trip. With the old hardcoded-false stub for remote nodes,
// isNodeLeader(nodeB) always returns false regardless of nodeB's actual
// state, so leadersFound would be 1 (nodeA only) instead of 2.
func TestSplitBrainDetector_DetectsMultipleRealLeaders(t *testing.T) {
	transportA := NewInMemoryTransport("nodeA")
	transportB := NewInMemoryTransport("nodeB")
	transportA.Connect(transportB)

	nodeA := NewRaftNode("nodeA", []string{"nodeA"}, transportA)
	nodeB := NewRaftNode("nodeB", []string{"nodeB"}, transportB)
	transportA.SetRaftNode(nodeA)
	transportB.SetRaftNode(nodeB)

	nodeA.Start()
	defer nodeA.Stop()
	nodeB.Start()
	defer nodeB.Stop()

	// Each is a single-node cluster, so each elects itself leader
	// independently -- exactly what a real network partition produces.
	waitForLeader(t, nodeA)
	waitForLeader(t, nodeB)

	membership := NewClusterMembership("nodeA", MembershipConfig{})
	sbd := NewSplitBrainDetector(nodeA, membership, SplitBrainConfig{})

	// Feed checkMultipleLeaders two partitions directly (bypassing the
	// heartbeat-heuristic identifyPartitions, which is unrelated to the bug
	// under test) so this exercises exactly the isNodeLeader code path the
	// fix touches: one local check (nodeA) and one remote RPC (nodeB).
	partitions := []*Partition{
		{ID: "p-nodeA", Nodes: []string{"nodeA"}},
		{ID: "p-nodeB", Nodes: []string{"nodeB"}},
	}

	leadersFound := sbd.checkMultipleLeaders(partitions)
	if leadersFound != 2 {
		t.Fatalf("expected split-brain detector to observe 2 real leaders across partitions, got %d (multipleLeaders=%v)",
			leadersFound, sbd.multipleLeaders)
	}
}

// TestLockManager_ProcessCommandsExitsOnStop proves RaftNode.Stop() causes
// LockManager.processCommands to exit promptly instead of leaking forever.
// processCommands used to be `for msg := range applyCh`, and Stop() never
// closes applyCh (its sole writer, applyLoop, could still be mid-send), so
// the consumer would block on that channel read forever after Stop().
func TestLockManager_ProcessCommandsExitsOnStop(t *testing.T) {
	raft := NewRaftNode("node1", []string{"node1"}, NewInMemoryTransport("node1"))
	raft.Start()

	waitForLeader(t, raft)

	lm := NewLockManager(raft, "node1")

	raft.Stop()

	select {
	case <-lm.Done():
		// processCommands exited cleanly; goroutine is gone.
	case <-time.After(2 * time.Second):
		t.Fatal("LockManager.processCommands did not exit after RaftNode.Stop(): goroutine leaked")
	}
}
