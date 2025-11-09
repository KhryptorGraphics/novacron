package consensus

import (
	"fmt"
	"math/rand"
	"sync"
	"time"
)

// RaftState represents the state of a Raft node
type RaftState int

const (
	StateFollower RaftState = iota
	StateCandidate
	StateLeader
)

func (s RaftState) String() string {
	switch s {
	case StateFollower:
		return "Follower"
	case StateCandidate:
		return "Candidate"
	case StateLeader:
		return "Leader"
	default:
		return "Unknown"
	}
}

// LogEntry represents a Raft log entry
type LogEntry struct {
	Term    uint64
	Index   uint64
	Command Command
}

// RaftConsensus implements the Raft consensus algorithm
type RaftConsensus struct {
	mu sync.RWMutex

	// Persistent state
	nodeID      string
	currentTerm uint64
	votedFor    string
	log         []LogEntry

	// Volatile state
	state       RaftState
	commitIndex uint64
	lastApplied uint64

	// Leader state
	nextIndex  map[string]uint64
	matchIndex map[string]uint64

	// Configuration
	peers        []string
	stateMachine StateMachine

	// Timers
	electionTimeout  time.Duration
	heartbeatTimeout time.Duration
	lastHeartbeat    time.Time

	// Channels
	proposeCh chan Proposal
	applyCh   chan LogEntry
	stopCh    chan struct{}
}

// VoteRequest is a Raft RequestVote RPC request
type VoteRequest struct {
	Term         uint64
	CandidateID  string
	LastLogIndex uint64
	LastLogTerm  uint64
}

// VoteResponse is a Raft RequestVote RPC response
type VoteResponse struct {
	Term        uint64
	VoteGranted bool
}

// AppendEntriesRequest is a Raft AppendEntries RPC request
type AppendEntriesRequest struct {
	Term         uint64
	LeaderID     string
	PrevLogIndex uint64
	PrevLogTerm  uint64
	Entries      []LogEntry
	LeaderCommit uint64
}

// AppendEntriesResponse is a Raft AppendEntries RPC response
type AppendEntriesResponse struct {
	Term    uint64
	Success bool
}

// NewRaftConsensus creates a new Raft consensus instance
func NewRaftConsensus(nodeID string, sm StateMachine) *RaftConsensus {
	return &RaftConsensus{
		nodeID:           nodeID,
		state:            StateFollower,
		log:              make([]LogEntry, 0),
		nextIndex:        make(map[string]uint64),
		matchIndex:       make(map[string]uint64),
		stateMachine:     sm,
		electionTimeout:  time.Duration(150+rand.Intn(150)) * time.Millisecond,
		heartbeatTimeout: 50 * time.Millisecond,
		proposeCh:        make(chan Proposal, 100),
		applyCh:          make(chan LogEntry, 100),
		stopCh:           make(chan struct{}),
	}
}

// Propose proposes a new value to the Raft cluster
func (r *RaftConsensus) Propose(key string, value []byte) error {
	r.mu.RLock()
	if r.state != StateLeader {
		r.mu.RUnlock()
		return fmt.Errorf("not leader")
	}
	r.mu.RUnlock()

	proposal := Proposal{
		Key:       key,
		Value:     value,
		Timestamp: time.Now(),
	}

	select {
	case r.proposeCh <- proposal:
		return nil
	case <-time.After(5 * time.Second):
		return fmt.Errorf("proposal timeout")
	}
}

// RequestVote handles RequestVote RPC
func (r *RaftConsensus) RequestVote(req *VoteRequest) *VoteResponse {
	r.mu.Lock()
	defer r.mu.Unlock()

	// Update term if necessary
	if req.Term > r.currentTerm {
		r.currentTerm = req.Term
		r.state = StateFollower
		r.votedFor = ""
	}

	granted := false

	// Grant vote if:
	// 1. Candidate's term is at least as current
	// 2. Haven't voted or already voted for this candidate
	// 3. Candidate's log is at least as up-to-date
	if req.Term >= r.currentTerm &&
		(r.votedFor == "" || r.votedFor == req.CandidateID) &&
		r.isLogUpToDate(req.LastLogIndex, req.LastLogTerm) {
		r.votedFor = req.CandidateID
		granted = true
		r.lastHeartbeat = time.Now()
	}

	return &VoteResponse{
		Term:        r.currentTerm,
		VoteGranted: granted,
	}
}

// AppendEntries handles AppendEntries RPC
func (r *RaftConsensus) AppendEntries(req *AppendEntriesRequest) *AppendEntriesResponse {
	r.mu.Lock()
	defer r.mu.Unlock()

	// Update term if necessary
	if req.Term > r.currentTerm {
		r.currentTerm = req.Term
		r.state = StateFollower
		r.votedFor = ""
	}

	// Reject if term is old
	if req.Term < r.currentTerm {
		return &AppendEntriesResponse{
			Term:    r.currentTerm,
			Success: false,
		}
	}

	// Reset election timeout
	r.lastHeartbeat = time.Now()

	// Check log consistency
	if req.PrevLogIndex > 0 {
		if req.PrevLogIndex > uint64(len(r.log)) {
			return &AppendEntriesResponse{
				Term:    r.currentTerm,
				Success: false,
			}
		}

		prevEntry := r.log[req.PrevLogIndex-1]
		if prevEntry.Term != req.PrevLogTerm {
			// Truncate conflicting entries
			r.log = r.log[:req.PrevLogIndex-1]
			return &AppendEntriesResponse{
				Term:    r.currentTerm,
				Success: false,
			}
		}
	}

	// Append new entries
	for i, entry := range req.Entries {
		index := req.PrevLogIndex + uint64(i) + 1
		if index <= uint64(len(r.log)) {
			// Overwrite conflicting entry
			r.log[index-1] = entry
		} else {
			// Append new entry
			r.log = append(r.log, entry)
		}
	}

	// Update commit index
	if req.LeaderCommit > r.commitIndex {
		r.commitIndex = minUint64(req.LeaderCommit, uint64(len(r.log)))
		r.applyCommitted()
	}

	return &AppendEntriesResponse{
		Term:    r.currentTerm,
		Success: true,
	}
}

// isLogUpToDate checks if candidate's log is at least as up-to-date
func (r *RaftConsensus) isLogUpToDate(lastLogIndex, lastLogTerm uint64) bool {
	if len(r.log) == 0 {
		return true
	}

	lastEntry := r.log[len(r.log)-1]
	if lastLogTerm != lastEntry.Term {
		return lastLogTerm >= lastEntry.Term
	}

	return lastLogIndex >= lastEntry.Index
}

// getLastLogIndex returns the index of the last log entry
func (r *RaftConsensus) getLastLogIndex() uint64 {
	if len(r.log) == 0 {
		return 0
	}
	return r.log[len(r.log)-1].Index
}

// getLastLogTerm returns the term of the last log entry
func (r *RaftConsensus) getLastLogTerm() uint64 {
	if len(r.log) == 0 {
		return 0
	}
	return r.log[len(r.log)-1].Term
}

// applyCommitted applies committed but not yet applied log entries
func (r *RaftConsensus) applyCommitted() {
	for r.lastApplied < r.commitIndex {
		r.lastApplied++
		if r.lastApplied <= uint64(len(r.log)) {
			entry := r.log[r.lastApplied-1]
			select {
			case r.applyCh <- entry:
			default:
				// Channel full, will retry
			}
		}
	}
}

// LoadSnapshot loads a snapshot into Raft
func (r *RaftConsensus) LoadSnapshot(snapshot *Snapshot) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if err := r.stateMachine.Restore(snapshot); err != nil {
		return err
	}

	r.lastApplied = snapshot.Index
	r.commitIndex = snapshot.Index
	r.currentTerm = snapshot.Term

	// Truncate log up to snapshot
	if snapshot.Index > 0 && snapshot.Index <= uint64(len(r.log)) {
		r.log = r.log[snapshot.Index:]
	}

	return nil
}

// becomeLeader transitions to leader state
func (r *RaftConsensus) becomeLeader() {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.state = StateLeader

	// Initialize leader state
	lastLogIndex := r.getLastLogIndex()
	for _, peer := range r.peers {
		r.nextIndex[peer] = lastLogIndex + 1
		r.matchIndex[peer] = 0
	}
}

// becomeFollower transitions to follower state
func (r *RaftConsensus) becomeFollower(term uint64) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.state = StateFollower
	r.currentTerm = term
	r.votedFor = ""
}

// becomeCandidate transitions to candidate state
func (r *RaftConsensus) becomeCandidate() {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.state = StateCandidate
	r.currentTerm++
	r.votedFor = r.nodeID
}

func minUint64(a, b uint64) uint64 {
	if a < b {
		return a
	}
	return b
}
