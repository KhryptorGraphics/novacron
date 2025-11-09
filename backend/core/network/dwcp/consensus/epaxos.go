package consensus

import (
	"fmt"
	"sync"
)

// EPaxosConsensus implements the EPaxos (Egalitarian Paxos) consensus algorithm
type EPaxosConsensus struct {
	mu sync.RWMutex

	nodeID       string
	replicaID    int32
	stateMachine StateMachine

	// EPaxos state
	instances map[InstanceID]*Instance
	executed  map[InstanceID]bool
	deps      *DependencyGraph

	// Configuration
	peers      []string
	fastQuorum int
	slowQuorum int
}

// Instance represents an EPaxos instance
type Instance struct {
	Command      Command
	Ballot       Ballot
	Status       InstanceStatus
	Dependencies []InstanceID
	Seq          uint64
	Committed    bool
}

// DependencyGraph tracks dependencies between instances
type DependencyGraph struct {
	mu   sync.RWMutex
	deps map[InstanceID][]InstanceID
}

// PreAcceptRequest is an EPaxos PreAccept RPC request
type PreAcceptRequest struct {
	InstanceID   InstanceID
	Command      Command
	Ballot       Ballot
	Dependencies []InstanceID
	Seq          uint64
}

// PreAcceptReply is an EPaxos PreAccept RPC response
type PreAcceptReply struct {
	OK           bool
	Ballot       Ballot
	Dependencies []InstanceID
	Seq          uint64
	FastPath     bool
}

// AcceptRequest is an EPaxos Accept RPC request
type AcceptRequest struct {
	InstanceID   InstanceID
	Ballot       Ballot
	Dependencies []InstanceID
	Seq          uint64
}

// AcceptReply is an EPaxos Accept RPC response
type AcceptReply struct {
	OK     bool
	Ballot Ballot
}

// CommitRequest is an EPaxos Commit RPC request
type CommitRequest struct {
	InstanceID   InstanceID
	Dependencies []InstanceID
	Seq          uint64
}

// NewEPaxosConsensus creates a new EPaxos consensus instance
func NewEPaxosConsensus(nodeID string, sm StateMachine) *EPaxosConsensus {
	replicaID := hashNodeID(nodeID)
	numReplicas := 5 // Default, should be configurable

	return &EPaxosConsensus{
		nodeID:       nodeID,
		replicaID:    replicaID,
		stateMachine: sm,
		instances:    make(map[InstanceID]*Instance),
		executed:     make(map[InstanceID]bool),
		deps:         NewDependencyGraph(),
		fastQuorum:   (numReplicas+1)/2 + (numReplicas+1)/4, // F + ⌊F/2⌋ + 1
		slowQuorum:   numReplicas/2 + 1,                     // F + 1
	}
}

// NewDependencyGraph creates a new dependency graph
func NewDependencyGraph() *DependencyGraph {
	return &DependencyGraph{
		deps: make(map[InstanceID][]InstanceID),
	}
}

// Propose proposes a new command using EPaxos
func (ep *EPaxosConsensus) Propose(key string, value []byte) error {
	cmd := Command{
		Type:      "write",
		Key:       key,
		Value:     value,
		Timestamp: NewTimestamp(),
	}

	return ep.PreAccept(cmd)
}

// PreAccept initiates the PreAccept phase
func (ep *EPaxosConsensus) PreAccept(cmd Command) error {
	ep.mu.Lock()
	instID := ep.getNextInstanceID()
	ballot := ep.makeBallot()

	inst := &Instance{
		Command: cmd,
		Ballot:  ballot,
		Status:  StatusPreAccepted,
	}

	// Compute dependencies
	inst.Dependencies = ep.computeDependencies(cmd)
	inst.Seq = ep.computeSeq(inst.Dependencies)

	ep.instances[instID] = inst
	ep.mu.Unlock()

	// Send PreAccept to fast quorum
	replies, err := ep.sendPreAccept(instID, inst)
	if err != nil {
		return err
	}

	// Check if can commit on fast path
	if ep.canCommitFast(replies, inst) {
		inst.Status = StatusCommitted
		inst.Committed = true
		ep.commit(instID, inst.Dependencies, inst.Seq)
		return nil
	}

	// Slow path: run Paxos accept phase
	return ep.runAcceptPhase(instID, inst)
}

// computeDependencies computes dependencies for a command
func (ep *EPaxosConsensus) computeDependencies(cmd Command) []InstanceID {
	deps := make([]InstanceID, 0)

	ep.mu.RLock()
	defer ep.mu.RUnlock()

	// Find all instances that conflict with this command
	for id, inst := range ep.instances {
		if inst.Status >= StatusPreAccepted && ep.conflicts(cmd, inst.Command) {
			deps = append(deps, id)
		}
	}

	return deps
}

// conflicts checks if two commands conflict
func (ep *EPaxosConsensus) conflicts(cmd1, cmd2 Command) bool {
	// Commands conflict if they access the same key
	return cmd1.Key == cmd2.Key
}

// computeSeq computes the sequence number based on dependencies
func (ep *EPaxosConsensus) computeSeq(deps []InstanceID) uint64 {
	var maxSeq uint64 = 0

	ep.mu.RLock()
	defer ep.mu.RUnlock()

	for _, depID := range deps {
		if inst, ok := ep.instances[depID]; ok {
			if inst.Seq > maxSeq {
				maxSeq = inst.Seq
			}
		}
	}

	return maxSeq + 1
}

// sendPreAccept sends PreAccept requests to replicas
func (ep *EPaxosConsensus) sendPreAccept(instID InstanceID, inst *Instance) ([]*PreAcceptReply, error) {
	replies := make([]*PreAcceptReply, 0)

	// Send to self
	reply := ep.handlePreAccept(&PreAcceptRequest{
		InstanceID:   instID,
		Command:      inst.Command,
		Ballot:       inst.Ballot,
		Dependencies: inst.Dependencies,
		Seq:          inst.Seq,
	})
	replies = append(replies, reply)

	// Send to peers (simulated)
	// In real implementation, would send RPCs to peers

	if len(replies) < ep.fastQuorum {
		return nil, fmt.Errorf("failed to get fast quorum")
	}

	return replies, nil
}

// handlePreAccept handles a PreAccept request
func (ep *EPaxosConsensus) handlePreAccept(req *PreAcceptRequest) *PreAcceptReply {
	ep.mu.Lock()
	defer ep.mu.Unlock()

	// Store instance
	if _, exists := ep.instances[req.InstanceID]; !exists {
		ep.instances[req.InstanceID] = &Instance{
			Command:      req.Command,
			Ballot:       req.Ballot,
			Status:       StatusPreAccepted,
			Dependencies: req.Dependencies,
			Seq:          req.Seq,
		}
	}

	// Compute local dependencies
	localDeps := ep.computeDependencies(req.Command)
	localSeq := ep.computeSeq(localDeps)

	return &PreAcceptReply{
		OK:           true,
		Ballot:       req.Ballot,
		Dependencies: localDeps,
		Seq:          localSeq,
		FastPath:     true,
	}
}

// canCommitFast checks if instance can commit on fast path
func (ep *EPaxosConsensus) canCommitFast(replies []*PreAcceptReply, inst *Instance) bool {
	if len(replies) < ep.fastQuorum {
		return false
	}

	// Check if all replies agree on dependencies and seq
	for _, reply := range replies {
		if !reply.OK || !reply.FastPath {
			return false
		}

		if !ep.dependenciesEqual(reply.Dependencies, inst.Dependencies) {
			return false
		}

		if reply.Seq != inst.Seq {
			return false
		}
	}

	return true
}

// dependenciesEqual checks if two dependency lists are equal
func (ep *EPaxosConsensus) dependenciesEqual(deps1, deps2 []InstanceID) bool {
	if len(deps1) != len(deps2) {
		return false
	}

	depMap := make(map[InstanceID]bool)
	for _, d := range deps1 {
		depMap[d] = true
	}

	for _, d := range deps2 {
		if !depMap[d] {
			return false
		}
	}

	return true
}

// runAcceptPhase runs the Paxos accept phase (slow path)
func (ep *EPaxosConsensus) runAcceptPhase(instID InstanceID, inst *Instance) error {
	// Send Accept to slow quorum
	acceptReplies := make([]*AcceptReply, 0)

	// Send to self
	reply := ep.handleAccept(&AcceptRequest{
		InstanceID:   instID,
		Ballot:       inst.Ballot,
		Dependencies: inst.Dependencies,
		Seq:          inst.Seq,
	})
	acceptReplies = append(acceptReplies, reply)

	// Send to peers (simulated)
	// In real implementation, would send RPCs to peers

	if len(acceptReplies) < ep.slowQuorum {
		return fmt.Errorf("failed to get slow quorum")
	}

	// Commit
	inst.Status = StatusCommitted
	inst.Committed = true
	ep.commit(instID, inst.Dependencies, inst.Seq)

	return nil
}

// handleAccept handles an Accept request
func (ep *EPaxosConsensus) handleAccept(req *AcceptRequest) *AcceptReply {
	ep.mu.Lock()
	defer ep.mu.Unlock()

	inst, exists := ep.instances[req.InstanceID]
	if !exists || req.Ballot.GreaterThan(inst.Ballot) {
		if !exists {
			inst = &Instance{}
			ep.instances[req.InstanceID] = inst
		}

		inst.Ballot = req.Ballot
		inst.Dependencies = req.Dependencies
		inst.Seq = req.Seq
		inst.Status = StatusAccepted
	}

	return &AcceptReply{
		OK:     true,
		Ballot: inst.Ballot,
	}
}

// commit commits an instance
func (ep *EPaxosConsensus) commit(instID InstanceID, deps []InstanceID, seq uint64) {
	ep.mu.Lock()
	defer ep.mu.Unlock()

	inst := ep.instances[instID]
	inst.Dependencies = deps
	inst.Seq = seq
	inst.Status = StatusCommitted

	// Try to execute
	ep.tryExecute(instID)
}

// tryExecute attempts to execute committed instances
func (ep *EPaxosConsensus) tryExecute(instID InstanceID) {
	if ep.executed[instID] {
		return
	}

	inst := ep.instances[instID]
	if inst.Status != StatusCommitted {
		return
	}

	// Execute dependencies first
	for _, depID := range inst.Dependencies {
		ep.tryExecute(depID)
	}

	// Execute this instance
	ep.stateMachine.Apply(inst.Command)
	ep.executed[instID] = true
	inst.Status = StatusExecuted
}

// LoadSnapshot loads a snapshot into EPaxos
func (ep *EPaxosConsensus) LoadSnapshot(snapshot *Snapshot) error {
	ep.mu.Lock()
	defer ep.mu.Unlock()

	return ep.stateMachine.Restore(snapshot)
}

// getNextInstanceID generates the next instance ID
func (ep *EPaxosConsensus) getNextInstanceID() InstanceID {
	// Find max sequence for this replica
	var maxSeq uint64 = 0
	for id := range ep.instances {
		if id.ReplicaID == ep.replicaID && id.Sequence > maxSeq {
			maxSeq = id.Sequence
		}
	}

	return InstanceID{
		ReplicaID: ep.replicaID,
		Sequence:  maxSeq + 1,
	}
}

// makeBallot creates a new ballot
func (ep *EPaxosConsensus) makeBallot() Ballot {
	ts := NewTimestamp()
	return Ballot{
		Number:    uint64(ts.Wall / 1000000), // Convert to milliseconds
		ReplicaID: ep.replicaID,
	}
}
