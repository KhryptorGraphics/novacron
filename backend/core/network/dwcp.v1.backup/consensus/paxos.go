package consensus

import (
	"fmt"
	"sync"
)

// PaxosConsensus implements the Paxos consensus algorithm
type PaxosConsensus struct {
	mu sync.RWMutex

	nodeID       string
	stateMachine StateMachine

	// Paxos state
	instances map[uint64]*PaxosInstance
	minSlot   uint64
	maxSlot   uint64

	// Proposer state
	ballot Ballot

	// Acceptor state
	promisedBallot map[uint64]Ballot
	acceptedBallot map[uint64]Ballot
	acceptedValue  map[uint64][]byte

	// Learner state
	learnedValues map[uint64][]byte

	peers []string
}

// PaxosInstance represents a single Paxos instance
type PaxosInstance struct {
	Slot     uint64
	Ballot   Ballot
	Value    []byte
	Status   InstanceStatus
	Promised map[string]bool
	Accepted map[string]bool
}

// PrepareRequest is a Paxos Prepare RPC request
type PrepareRequest struct {
	Slot   uint64
	Ballot Ballot
}

// PrepareResponse is a Paxos Prepare RPC response
type PrepareResponse struct {
	Promised       bool
	AcceptedBallot Ballot
	AcceptedValue  []byte
}

// PaxosAcceptRequest is a Paxos Accept RPC request
type PaxosAcceptRequest struct {
	Slot   uint64
	Ballot Ballot
	Value  []byte
}

// PaxosAcceptResponse is a Paxos Accept RPC response
type PaxosAcceptResponse struct {
	Accepted bool
	Ballot   Ballot
}

// NewPaxosConsensus creates a new Paxos consensus instance
func NewPaxosConsensus(nodeID string, sm StateMachine) *PaxosConsensus {
	return &PaxosConsensus{
		nodeID:         nodeID,
		stateMachine:   sm,
		instances:      make(map[uint64]*PaxosInstance),
		promisedBallot: make(map[uint64]Ballot),
		acceptedBallot: make(map[uint64]Ballot),
		acceptedValue:  make(map[uint64][]byte),
		learnedValues:  make(map[uint64][]byte),
		ballot: Ballot{
			Number:    0,
			ReplicaID: hashNodeID(nodeID),
		},
	}
}

// Propose proposes a new value using Paxos
func (p *PaxosConsensus) Propose(key string, value []byte) error {
	p.mu.Lock()
	slot := p.maxSlot + 1
	p.maxSlot = slot

	// Increment ballot number
	p.ballot.Number++
	ballot := p.ballot
	p.mu.Unlock()

	// Phase 1: Prepare
	prepareResp, err := p.sendPrepare(slot, ballot)
	if err != nil {
		return fmt.Errorf("prepare phase failed: %w", err)
	}

	// Use highest accepted value if any
	proposedValue := value
	highestBallot := Ballot{Number: 0, ReplicaID: 0}
	for _, resp := range prepareResp {
		if resp.Promised && resp.AcceptedBallot.GreaterThan(highestBallot) {
			highestBallot = resp.AcceptedBallot
			proposedValue = resp.AcceptedValue
		}
	}

	// Phase 2: Accept
	if err := p.sendAccept(slot, ballot, proposedValue); err != nil {
		return fmt.Errorf("accept phase failed: %w", err)
	}

	return nil
}

// sendPrepare sends Prepare requests to all acceptors
func (p *PaxosConsensus) sendPrepare(slot uint64, ballot Ballot) ([]*PrepareResponse, error) {
	responses := make([]*PrepareResponse, 0)

	// Send to self
	resp := p.handlePrepare(&PrepareRequest{
		Slot:   slot,
		Ballot: ballot,
	})
	responses = append(responses, resp)

	// Send to peers (simulated)
	// In real implementation, would send RPCs to peers

	// Check for quorum
	promiseCount := 0
	for _, resp := range responses {
		if resp.Promised {
			promiseCount++
		}
	}

	quorum := len(p.peers)/2 + 1
	if promiseCount < quorum {
		return nil, fmt.Errorf("failed to get quorum of promises")
	}

	return responses, nil
}

// sendAccept sends Accept requests to all acceptors
func (p *PaxosConsensus) sendAccept(slot uint64, ballot Ballot, value []byte) error {
	acceptCount := 0

	// Send to self
	resp := p.handleAccept(&PaxosAcceptRequest{
		Slot:   slot,
		Ballot: ballot,
		Value:  value,
	})

	if resp.Accepted {
		acceptCount++
	}

	// Send to peers (simulated)
	// In real implementation, would send RPCs to peers

	// Check for quorum
	quorum := len(p.peers)/2 + 1
	if acceptCount < quorum {
		return fmt.Errorf("failed to get quorum of accepts")
	}

	// Learn the value
	p.learn(slot, value)

	return nil
}

// handlePrepare handles a Prepare request (acceptor role)
func (p *PaxosConsensus) handlePrepare(req *PrepareRequest) *PrepareResponse {
	p.mu.Lock()
	defer p.mu.Unlock()

	promised := false
	var acceptedBallot Ballot
	var acceptedValue []byte

	// Check if we can promise
	promisedBallot, exists := p.promisedBallot[req.Slot]
	if !exists || req.Ballot.GreaterThan(promisedBallot) {
		p.promisedBallot[req.Slot] = req.Ballot
		promised = true

		// Return any previously accepted value
		if ballot, ok := p.acceptedBallot[req.Slot]; ok {
			acceptedBallot = ballot
			acceptedValue = p.acceptedValue[req.Slot]
		}
	}

	return &PrepareResponse{
		Promised:       promised,
		AcceptedBallot: acceptedBallot,
		AcceptedValue:  acceptedValue,
	}
}

// handleAccept handles an Accept request (acceptor role)
func (p *PaxosConsensus) handleAccept(req *PaxosAcceptRequest) *PaxosAcceptResponse {
	p.mu.Lock()
	defer p.mu.Unlock()

	accepted := false

	// Check if ballot is at least as high as promised
	promisedBallot, exists := p.promisedBallot[req.Slot]
	if !exists || !promisedBallot.GreaterThan(req.Ballot) {
		p.acceptedBallot[req.Slot] = req.Ballot
		p.acceptedValue[req.Slot] = req.Value
		accepted = true
	}

	return &PaxosAcceptResponse{
		Accepted: accepted,
		Ballot:   req.Ballot,
	}
}

// learn records a learned value (learner role)
func (p *PaxosConsensus) learn(slot uint64, value []byte) {
	p.mu.Lock()
	defer p.mu.Unlock()

	p.learnedValues[slot] = value

	// Apply to state machine in order
	for p.minSlot <= p.maxSlot {
		if val, ok := p.learnedValues[p.minSlot]; ok {
			cmd := Command{
				Type:  "write",
				Value: val,
			}
			p.stateMachine.Apply(cmd)
			p.minSlot++
		} else {
			break
		}
	}
}

// LoadSnapshot loads a snapshot into Paxos
func (p *PaxosConsensus) LoadSnapshot(snapshot *Snapshot) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	if err := p.stateMachine.Restore(snapshot); err != nil {
		return err
	}

	p.minSlot = snapshot.Index
	p.maxSlot = snapshot.Index

	return nil
}

// hashNodeID converts a node ID to a replica ID
func hashNodeID(nodeID string) int32 {
	var hash int32
	for _, c := range nodeID {
		hash = hash*31 + int32(c)
	}
	return hash
}
