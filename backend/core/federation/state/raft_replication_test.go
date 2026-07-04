package state

import (
	"context"
	"testing"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/consensus"
)

// TestRaftReplication_WriteLeaderReadFollower proves REAL cross-instance state
// replication: two independent GeoDistributedState instances (distinct stores),
// each backed by its own consensus.RaftNode, joined into ONE Raft group over real
// localhost HTTP sockets. A write on the leader is observed on the follower ONLY
// after the Raft entry commits and the follower applies it -- not via any side
// channel (the follower rejects direct writes; applyReplicated is its sole writer).
//
// In a 2-node group majority=2, so a write cannot commit until it is replicated to
// the follower; a committed read therefore proves replication.
func TestRaftReplication_WriteLeaderReadFollower(t *testing.T) {
	addrs := map[string]string{
		"A": "127.0.0.1:17061",
		"B": "127.0.0.1:17062",
	}
	peers := []string{"A", "B"}

	// Build both transports + nodes and wire them BEFORE starting, so early
	// RequestVote/AppendEntries RPCs find a live server + a set raft node.
	tA := consensus.NewHTTPTransport(addrs, addrs["A"])
	tB := consensus.NewHTTPTransport(addrs, addrs["B"])
	nodeA := consensus.NewRaftNode("A", peers, tA)
	nodeB := consensus.NewRaftNode("B", peers, tB)
	tA.SetRaftNode(nodeA)
	tB.SetRaftNode(nodeB)
	if err := tA.Start(); err != nil {
		t.Fatalf("transport A start: %v", err)
	}
	if err := tB.Start(); err != nil {
		t.Fatalf("transport B start: %v", err)
	}
	defer tA.Stop()
	defer tB.Stop()
	nodeA.Start()
	nodeB.Start()
	defer nodeA.Stop()
	defer nodeB.Stop()

	newCfg := func(region string) *StateConfig {
		return &StateConfig{
			LocalRegion:       region,
			Regions:           []string{"A", "B"},
			SyncInterval:      time.Second,
			ConsistencyLevel:  ConsistencyEventual,
			ReplicationFactor: 2,
			UseVectorClock:    true,
		}
	}
	gdsA, err := NewGeoDistributedState(newCfg("A"))
	if err != nil {
		t.Fatal(err)
	}
	gdsB, err := NewGeoDistributedState(newCfg("B"))
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	gdsA.Start(ctx)
	gdsB.Start(ctx)
	defer gdsA.Stop()
	defer gdsB.Stop()

	gdsA.AttachReplicator(nodeA)
	gdsB.AttachReplicator(nodeB)

	// Wait for a leader (2-node split votes may take a few rounds).
	deadline := time.Now().Add(6 * time.Second)
	for time.Now().Before(deadline) && !nodeA.IsLeader() && !nodeB.IsLeader() {
		time.Sleep(50 * time.Millisecond)
	}

	var leaderGDS, followerGDS *GeoDistributedState
	var followerNode *consensus.RaftNode
	var followerRepl *RaftReplicator
	var leaderID, followerID string
	switch {
	case nodeA.IsLeader():
		leaderGDS, followerGDS, followerNode, followerRepl = gdsA, gdsB, nodeB, gdsB.replicator
		leaderID, followerID = "A", "B"
	case nodeB.IsLeader():
		leaderGDS, followerGDS, followerNode, followerRepl = gdsB, gdsA, nodeA, gdsA.replicator
		leaderID, followerID = "B", "A"
	default:
		t.Fatal("no leader elected within 6s")
	}
	t.Logf("Raft group formed: leader=%s follower=%s", leaderID, followerID)

	const key = "svc/endpoint"
	const want = "https://leader.region.local:9443"

	// Baseline: follower must NOT have the key, and has applied no command for it.
	if _, err := followerGDS.Get(ctx, key, ConsistencyEventual); err == nil {
		t.Fatalf("follower %s already had key %q before any write", followerID, key)
	}
	beforeApplied := followerRepl.AppliedCount()
	beforeCommitted := followerNode.GetStats().LogEntriesCommitted

	// A direct write on the follower must be rejected (proves no side channel).
	if err := followerGDS.Put(ctx, key, "should-be-rejected", 0); err == nil {
		t.Fatalf("follower %s accepted a direct write; commit-path invariant broken", followerID)
	}

	// WRITE on the leader instance.
	if err := leaderGDS.Put(ctx, key, want, 0); err != nil {
		t.Fatalf("leader %s Put: %v", leaderID, err)
	}

	// READ back on the FOLLOWER instance, polling until the Raft commit applies.
	var got interface{}
	readDeadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(readDeadline) {
		if e, err := followerGDS.Get(ctx, key, ConsistencyEventual); err == nil {
			got = e.Value
			break
		}
		time.Sleep(50 * time.Millisecond)
	}
	if got == nil {
		t.Fatalf("value never replicated to follower %s within 5s", followerID)
	}
	if got != want {
		t.Fatalf("follower %s value = %v, want %v", followerID, got, want)
	}

	afterApplied := followerRepl.AppliedCount()
	afterCommitted := followerNode.GetStats().LogEntriesCommitted
	if afterApplied <= beforeApplied {
		t.Fatalf("follower %s applied no committed Raft entry (before=%d after=%d)",
			followerID, beforeApplied, afterApplied)
	}
	if afterCommitted <= beforeCommitted {
		t.Fatalf("follower %s Raft commit index did not advance (before=%d after=%d)",
			followerID, beforeCommitted, afterCommitted)
	}

	t.Logf("REPLICATED via Raft commit: wrote %s=%q on leader %s; read %s=%v on follower %s "+
		"(follower applied %d->%d entries, LogEntriesCommitted %d->%d)",
		key, want, leaderID, key, got, followerID,
		beforeApplied, afterApplied, beforeCommitted, afterCommitted)
}
