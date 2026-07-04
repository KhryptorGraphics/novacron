// Real cross-instance state replication built on backend/core/consensus/raft.go.
//
// SCOPE (honest): this wires the local store to ONE Raft group. A write on the
// leader is appended to the Raft log, committed by a quorum, and applied to every
// member's store via applyLoop. That is real, verifiable cross-instance replication
// with read-your-writes on the leader and asynchronous (commit-gated) visibility on
// followers. It is NOT multi-group, partition-tolerant geo-distribution despite the
// package name geo_distributed_state — that remains deferred.
package state

import (
	"encoding/json"
	"fmt"
	"sync/atomic"

	"github.com/khryptorgraphics/novacron/backend/core/consensus"
)

// stateCommand is the unit replicated through the Raft log. Every committed
// command is applied to every member's local store by applyLoop.
type stateCommand struct {
	Entry *StateEntry `json:"entry"`
}

// RaftReplicator makes the GeoDistributedState store the replicated state machine
// of a single consensus.RaftNode group.
type RaftReplicator struct {
	node    *consensus.RaftNode
	apply   func(*StateEntry)
	applied int64 // committed entries applied locally (atomic; B-side commit proof)
}

// NewRaftReplicator starts applying committed Raft entries into the store via apply.
func NewRaftReplicator(node *consensus.RaftNode, apply func(*StateEntry)) *RaftReplicator {
	r := &RaftReplicator{node: node, apply: apply}
	go r.applyLoop()
	return r
}

// applyLoop applies every committed Raft entry to the local store. This is the
// ONLY path that writes replicated entries into a follower's store, which is what
// makes "the follower holds the value => it was committed through Raft" sound.
// ponytail: ranges a channel raft.go never closes -> one goroutine leaks at
// teardown; harmless for an exiting process. Add a done-chan if reused long-lived.
func (r *RaftReplicator) applyLoop() {
	for msg := range r.node.GetApplyChan() {
		if !msg.CommandValid {
			continue
		}
		cmd, err := decodeCommand(msg.Command)
		if err != nil || cmd.Entry == nil {
			continue
		}
		r.apply(cmd.Entry)
		atomic.AddInt64(&r.applied, 1)
	}
}

// Replicate appends an entry to the Raft log. Only the leader may append; on a
// follower it returns an error naming the current leader so callers route writes there.
func (r *RaftReplicator) Replicate(entry *StateEntry) error {
	if !r.node.IsLeader() {
		return fmt.Errorf("not raft leader (leader=%q)", r.node.GetLeader())
	}
	if _, _, ok := r.node.Submit(stateCommand{Entry: entry}); !ok {
		return fmt.Errorf("raft rejected submit (no longer leader)")
	}
	return nil
}

func (r *RaftReplicator) IsLeader() bool      { return r.node.IsLeader() }
func (r *RaftReplicator) Leader() string      { return r.node.GetLeader() }
func (r *RaftReplicator) AppliedCount() int64 { return atomic.LoadInt64(&r.applied) }

// decodeCommand normalizes the two shapes a committed command arrives in: the
// leader keeps the original Go value; a follower receives it JSON-decoded into a
// generic map. Marshal-then-unmarshal yields a *stateCommand for both.
func decodeCommand(cmd interface{}) (*stateCommand, error) {
	b, err := json.Marshal(cmd)
	if err != nil {
		return nil, err
	}
	var sc stateCommand
	if err := json.Unmarshal(b, &sc); err != nil {
		return nil, err
	}
	return &sc, nil
}
