package federation

import (
	"context"
	"encoding/json"
	"fmt"
	"math/rand"
	"sync"
	"sync/atomic"
	"time"
)

// RaftConsensus implements the ConsensusManager interface using Raft protocol
type RaftConsensus struct {
	config       *FederationConfig
	nodeID       string
	currentTerm  uint64
	votedFor     string
	log          []LogEntry
	commitIndex  uint64
	lastApplied  uint64
	state        ConsensusRole
	stateMu      sync.RWMutex
	
	// Leader state
	nextIndex    map[string]uint64
	matchIndex   map[string]uint64
	
	// Candidate state
	votesReceived map[string]bool
	
	// Cluster membership
	peers        map[string]string // nodeID -> address
	peersMu      sync.RWMutex
	
	// Channels
	appendEntriesCh chan *RaftMessage
	voteRequestCh   chan *RaftMessage
	heartbeatCh     chan struct{}
	stopCh          chan struct{}
	
	// Timers
	electionTimer  *time.Timer
	heartbeatTimer *time.Timer
	
	// State machine
	stateMachine map[string][]byte
	smMu         sync.RWMutex
	
	logger       Logger
	isRunning    atomic.Bool
}

// NewRaftConsensus creates a new Raft consensus manager
func NewRaftConsensus(config *FederationConfig, logger Logger) (*RaftConsensus, error) {
	r := &RaftConsensus{
		config:          config,
		nodeID:          config.NodeID,
		currentTerm:     0,
		votedFor:        "",
		log:             make([]LogEntry, 0),
		commitIndex:     0,
		lastApplied:     0,
		state:           RoleFollower,
		nextIndex:       make(map[string]uint64),
		matchIndex:      make(map[string]uint64),
		votesReceived:   make(map[string]bool),
		peers:           make(map[string]string),
		appendEntriesCh: make(chan *RaftMessage, 100),
		voteRequestCh:   make(chan *RaftMessage, 100),
		heartbeatCh:     make(chan struct{}, 1),
		stopCh:          make(chan struct{}),
		stateMachine:    make(map[string][]byte),
		logger:          logger,
	}
	
	// Initialize with a noop entry at index 0
	r.log = append(r.log, LogEntry{
		Index:     0,
		Term:      0,
		Type:      EntryNoop,
		Timestamp: time.Now(),
	})
	
	return r, nil
}

// Start starts the Raft consensus protocol
func (r *RaftConsensus) Start(ctx context.Context) error {
	if r.isRunning.Load() {
		return fmt.Errorf("consensus already running")
	}
	
	r.logger.Info("Starting Raft consensus", "node_id", r.nodeID)
	
	// Start as follower
	r.becomeFollower(0)
	
	// Start main loop
	go r.run(ctx)
	
	r.isRunning.Store(true)
	
	return nil
}

// Stop stops the Raft consensus protocol
func (r *RaftConsensus) Stop(ctx context.Context) error {
	if !r.isRunning.Load() {
		return fmt.Errorf("consensus not running")
	}
	
	r.logger.Info("Stopping Raft consensus")
	
	close(r.stopCh)
	
	// Stop timers
	if r.electionTimer != nil {
		r.electionTimer.Stop()
	}
	if r.heartbeatTimer != nil {
		r.heartbeatTimer.Stop()
	}
	
	r.isRunning.Store(false)
	
	return nil
}

// ProposeValue proposes a value to be committed to the log
func (r *RaftConsensus) ProposeValue(ctx context.Context, key string, value []byte) error {
	r.stateMu.RLock()
	isLeader := r.state == RoleLeader
	r.stateMu.RUnlock()
	
	if !isLeader {
		return fmt.Errorf("not the leader")
	}
	
	// Create log entry
	entry := LogEntry{
		Index:     uint64(len(r.log)),
		Term:      r.currentTerm,
		Type:      EntryCommand,
		Data:      value,
		Timestamp: time.Now(),
	}
	
	// Append to local log
	r.log = append(r.log, entry)
	
	// Replicate to followers
	r.replicateEntries()
	
	return nil
}

// GetValue retrieves a value from the state machine
func (r *RaftConsensus) GetValue(ctx context.Context, key string) ([]byte, error) {
	r.smMu.RLock()
	defer r.smMu.RUnlock()
	
	value, exists := r.stateMachine[key]
	if !exists {
		return nil, fmt.Errorf("key not found: %s", key)
	}
	
	return value, nil
}

// GetLeader returns the current leader's node ID
func (r *RaftConsensus) GetLeader() (string, error) {
	r.stateMu.RLock()
	defer r.stateMu.RUnlock()
	
	if r.state == RoleLeader {
		return r.nodeID, nil
	}
	
	// In a real implementation, followers would track the leader
	// For now, return empty if not leader
	return "", fmt.Errorf("leader unknown")
}

// IsLeader checks if this node is the leader
func (r *RaftConsensus) IsLeader() bool {
	r.stateMu.RLock()
	defer r.stateMu.RUnlock()
	
	return r.state == RoleLeader
}

// AddNode adds a new node to the cluster
func (r *RaftConsensus) AddNode(ctx context.Context, nodeID string, address string) error {
	r.peersMu.Lock()
	defer r.peersMu.Unlock()
	
	r.peers[nodeID] = address
	
	// If leader, initialize indices for new node
	r.stateMu.RLock()
	isLeader := r.state == RoleLeader
	r.stateMu.RUnlock()
	
	if isLeader {
		r.nextIndex[nodeID] = uint64(len(r.log))
		r.matchIndex[nodeID] = 0
	}
	
	r.logger.Info("Added node to cluster", "node_id", nodeID, "address", address)
	
	return nil
}

// RemoveNode removes a node from the cluster
func (r *RaftConsensus) RemoveNode(ctx context.Context, nodeID string) error {
	r.peersMu.Lock()
	defer r.peersMu.Unlock()
	
	delete(r.peers, nodeID)
	delete(r.nextIndex, nodeID)
	delete(r.matchIndex, nodeID)
	
	r.logger.Info("Removed node from cluster", "node_id", nodeID)
	
	return nil
}

// Main Raft loop
func (r *RaftConsensus) run(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case <-r.stopCh:
			return
			
		case msg := <-r.appendEntriesCh:
			r.handleAppendEntries(msg)
			
		case msg := <-r.voteRequestCh:
			r.handleVoteRequest(msg)
			
		case <-r.electionTimer.C:
			r.stateMu.RLock()
			state := r.state
			r.stateMu.RUnlock()
			
			if state != RoleLeader {
				r.startElection()
			}
			
		case <-r.heartbeatTimer.C:
			r.stateMu.RLock()
			isLeader := r.state == RoleLeader
			r.stateMu.RUnlock()
			
			if isLeader {
				r.sendHeartbeats()
				r.resetHeartbeatTimer()
			}
		}
	}
}

// State transitions

func (r *RaftConsensus) becomeFollower(term uint64) {
	r.stateMu.Lock()
	defer r.stateMu.Unlock()
	
	r.logger.Info("Becoming follower", "term", term)
	
	r.state = RoleFollower
	r.currentTerm = term
	r.votedFor = ""
	
	r.resetElectionTimer()
}

func (r *RaftConsensus) becomeCandidate() {
	r.stateMu.Lock()
	defer r.stateMu.Unlock()
	
	r.logger.Info("Becoming candidate", "term", r.currentTerm+1)
	
	r.state = RoleCandidate
	r.currentTerm++
	r.votedFor = r.nodeID
	r.votesReceived = map[string]bool{
		r.nodeID: true, // Vote for self
	}
	
	r.resetElectionTimer()
}

func (r *RaftConsensus) becomeLeader() {
	r.stateMu.Lock()
	defer r.stateMu.Unlock()
	
	r.logger.Info("Becoming leader", "term", r.currentTerm)
	
	r.state = RoleLeader
	
	// Initialize leader state
	r.peersMu.RLock()
	for peerID := range r.peers {
		r.nextIndex[peerID] = uint64(len(r.log))
		r.matchIndex[peerID] = 0
	}
	r.peersMu.RUnlock()
	
	// Send initial heartbeat
	go r.sendHeartbeats()
	r.resetHeartbeatTimer()
}

// Election logic

func (r *RaftConsensus) startElection() {
	r.becomeCandidate()
	
	r.logger.Info("Starting election", "term", r.currentTerm)
	
	// Request votes from all peers
	r.peersMu.RLock()
	peers := make(map[string]string)
	for k, v := range r.peers {
		peers[k] = v
	}
	r.peersMu.RUnlock()
	
	lastLogIndex := uint64(len(r.log) - 1)
	lastLogTerm := uint64(0)
	if lastLogIndex > 0 {
		lastLogTerm = r.log[lastLogIndex].Term
	}
	
	voteRequest := &RaftMessage{
		Term:         r.currentTerm,
		Type:         MessageVoteRequest,
		From:         r.nodeID,
		PrevLogIndex: lastLogIndex,
		PrevLogTerm:  lastLogTerm,
	}
	
	// Send vote requests in parallel
	var wg sync.WaitGroup
	for peerID := range peers {
		wg.Add(1)
		go func(id string) {
			defer wg.Done()
			r.sendVoteRequest(id, voteRequest)
		}(peerID)
	}
	
	// Wait for votes with timeout
	go func() {
		time.Sleep(r.config.ElectionTimeout / 2)
		r.checkElectionResult()
	}()
}

func (r *RaftConsensus) handleVoteRequest(msg *RaftMessage) {
	r.stateMu.Lock()
	defer r.stateMu.Unlock()
	
	response := &RaftMessage{
		Term:        r.currentTerm,
		Type:        MessageVoteResponse,
		From:        r.nodeID,
		To:          msg.From,
		VoteGranted: false,
	}
	
	// Check term
	if msg.Term < r.currentTerm {
		// Reply with current term
		r.sendVoteResponse(msg.From, response)
		return
	}
	
	// Update term if newer
	if msg.Term > r.currentTerm {
		r.currentTerm = msg.Term
		r.votedFor = ""
		r.state = RoleFollower
	}
	
	// Grant vote if haven't voted or voted for candidate
	if r.votedFor == "" || r.votedFor == msg.From {
		// Check log consistency
		lastLogIndex := uint64(len(r.log) - 1)
		lastLogTerm := uint64(0)
		if lastLogIndex > 0 {
			lastLogTerm = r.log[lastLogIndex].Term
		}
		
		if msg.PrevLogTerm > lastLogTerm || 
		   (msg.PrevLogTerm == lastLogTerm && msg.PrevLogIndex >= lastLogIndex) {
			r.votedFor = msg.From
			response.VoteGranted = true
			response.Term = r.currentTerm
			r.resetElectionTimer()
		}
	}
	
	r.sendVoteResponse(msg.From, response)
}

func (r *RaftConsensus) handleVoteResponse(msg *RaftMessage) {
	r.stateMu.Lock()
	defer r.stateMu.Unlock()
	
	// Ignore if not candidate
	if r.state != RoleCandidate {
		return
	}
	
	// Check term
	if msg.Term > r.currentTerm {
		r.currentTerm = msg.Term
		r.state = RoleFollower
		r.votedFor = ""
		return
	}
	
	// Record vote
	if msg.VoteGranted {
		r.votesReceived[msg.From] = true
	}
	
	r.checkElectionResult()
}

func (r *RaftConsensus) checkElectionResult() {
	r.stateMu.RLock()
	votesNeeded := (len(r.peers) + 1) / 2 + 1
	votesReceived := len(r.votesReceived)
	r.stateMu.RUnlock()
	
	if votesReceived >= votesNeeded {
		r.becomeLeader()
	}
}

// Log replication

func (r *RaftConsensus) sendHeartbeats() {
	r.peersMu.RLock()
	peers := make(map[string]string)
	for k, v := range r.peers {
		peers[k] = v
	}
	r.peersMu.RUnlock()
	
	for peerID := range peers {
		go r.sendAppendEntries(peerID, true)
	}
}

func (r *RaftConsensus) replicateEntries() {
	r.peersMu.RLock()
	peers := make(map[string]string)
	for k, v := range r.peers {
		peers[k] = v
	}
	r.peersMu.RUnlock()
	
	for peerID := range peers {
		go r.sendAppendEntries(peerID, false)
	}
}

func (r *RaftConsensus) sendAppendEntries(peerID string, isHeartbeat bool) {
	r.stateMu.RLock()
	
	nextIdx := r.nextIndex[peerID]
	prevLogIndex := nextIdx - 1
	prevLogTerm := uint64(0)
	
	if prevLogIndex > 0 && prevLogIndex < uint64(len(r.log)) {
		prevLogTerm = r.log[prevLogIndex].Term
	}
	
	var entries []LogEntry
	if !isHeartbeat && nextIdx < uint64(len(r.log)) {
		entries = r.log[nextIdx:]
	}
	
	msg := &RaftMessage{
		Term:         r.currentTerm,
		Type:         MessageAppendEntries,
		From:         r.nodeID,
		To:           peerID,
		PrevLogIndex: prevLogIndex,
		PrevLogTerm:  prevLogTerm,
		Entries:      entries,
		CommitIndex:  r.commitIndex,
	}
	
	r.stateMu.RUnlock()
	
	// Send message (in real implementation, would use network transport)
	// For now, simulate with channel
	select {
	case r.appendEntriesCh <- msg:
	case <-time.After(100 * time.Millisecond):
		r.logger.Warn("Failed to send append entries", "peer", peerID)
	}
}

func (r *RaftConsensus) handleAppendEntries(msg *RaftMessage) {
	r.stateMu.Lock()
	defer r.stateMu.Unlock()
	
	response := &RaftMessage{
		Term:    r.currentTerm,
		Type:    MessageAppendResponse,
		From:    r.nodeID,
		To:      msg.From,
		Success: false,
	}
	
	// Check term
	if msg.Term < r.currentTerm {
		r.sendAppendResponse(msg.From, response)
		return
	}
	
	// Update term and become follower if needed
	if msg.Term > r.currentTerm || r.state == RoleCandidate {
		r.currentTerm = msg.Term
		r.votedFor = ""
		r.state = RoleFollower
	}
	
	r.resetElectionTimer()
	
	// Check log consistency
	if msg.PrevLogIndex > 0 {
		if msg.PrevLogIndex >= uint64(len(r.log)) || 
		   r.log[msg.PrevLogIndex].Term != msg.PrevLogTerm {
			r.sendAppendResponse(msg.From, response)
			return
		}
	}
	
	// Append entries
	if len(msg.Entries) > 0 {
		// Remove conflicting entries
		for i, entry := range msg.Entries {
			idx := msg.PrevLogIndex + uint64(i) + 1
			if idx < uint64(len(r.log)) {
				if r.log[idx].Term != entry.Term {
					r.log = r.log[:idx]
				}
			}
			
			if idx >= uint64(len(r.log)) {
				r.log = append(r.log, entry)
			}
		}
	}
	
	// Update commit index
	if msg.CommitIndex > r.commitIndex {
		r.commitIndex = min(msg.CommitIndex, uint64(len(r.log)-1))
		r.applyCommittedEntries()
	}
	
	response.Success = true
	r.sendAppendResponse(msg.From, response)
}

func (r *RaftConsensus) handleAppendResponse(msg *RaftMessage) {
	r.stateMu.Lock()
	defer r.stateMu.Unlock()
	
	// Ignore if not leader
	if r.state != RoleLeader {
		return
	}
	
	// Check term
	if msg.Term > r.currentTerm {
		r.currentTerm = msg.Term
		r.state = RoleFollower
		r.votedFor = ""
		return
	}
	
	if msg.Success {
		// Update indices
		r.nextIndex[msg.From] = uint64(len(r.log))
		r.matchIndex[msg.From] = uint64(len(r.log) - 1)
		
		// Check if we can commit
		r.updateCommitIndex()
	} else {
		// Decrement nextIndex and retry
		if r.nextIndex[msg.From] > 1 {
			r.nextIndex[msg.From]--
		}
		go r.sendAppendEntries(msg.From, false)
	}
}

func (r *RaftConsensus) updateCommitIndex() {
	// Find median of match indices
	matches := make([]uint64, 0, len(r.peers)+1)
	matches = append(matches, uint64(len(r.log)-1)) // Self
	
	for _, idx := range r.matchIndex {
		matches = append(matches, idx)
	}
	
	// Sort and find median
	for i := 0; i < len(matches); i++ {
		for j := i + 1; j < len(matches); j++ {
			if matches[i] > matches[j] {
				matches[i], matches[j] = matches[j], matches[i]
			}
		}
	}
	
	median := matches[len(matches)/2]
	
	// Commit if majority replicated and from current term
	if median > r.commitIndex && r.log[median].Term == r.currentTerm {
		r.commitIndex = median
		r.applyCommittedEntries()
	}
}

func (r *RaftConsensus) applyCommittedEntries() {
	for r.lastApplied < r.commitIndex {
		r.lastApplied++
		entry := r.log[r.lastApplied]
		
		if entry.Type == EntryCommand {
			// Apply to state machine
			var cmd map[string]interface{}
			if err := json.Unmarshal(entry.Data, &cmd); err == nil {
				if key, ok := cmd["key"].(string); ok {
					if value, ok := cmd["value"].([]byte); ok {
						r.smMu.Lock()
						r.stateMachine[key] = value
						r.smMu.Unlock()
					}
				}
			}
		}
	}
}

// Network simulation helpers (would be replaced with real network transport)

func (r *RaftConsensus) sendVoteRequest(peerID string, msg *RaftMessage) {
	// Simulate network send
	time.Sleep(time.Duration(rand.Intn(10)) * time.Millisecond)
	
	// In real implementation, would send over network
	// For now, just log
	r.logger.Debug("Sending vote request", "to", peerID, "term", msg.Term)
}

func (r *RaftConsensus) sendVoteResponse(peerID string, msg *RaftMessage) {
	// Simulate network send
	time.Sleep(time.Duration(rand.Intn(10)) * time.Millisecond)
	
	r.logger.Debug("Sending vote response", "to", peerID, "granted", msg.VoteGranted)
	
	// If this is a response to ourselves, handle it
	if peerID == r.nodeID {
		r.handleVoteResponse(msg)
	}
}

func (r *RaftConsensus) sendAppendResponse(peerID string, msg *RaftMessage) {
	// Simulate network send
	time.Sleep(time.Duration(rand.Intn(10)) * time.Millisecond)
	
	r.logger.Debug("Sending append response", "to", peerID, "success", msg.Success)
	
	// If this is a response to ourselves, handle it
	if peerID == r.nodeID {
		r.handleAppendResponse(msg)
	}
}

// Timer management

func (r *RaftConsensus) resetElectionTimer() {
	timeout := r.config.ElectionTimeout + 
		time.Duration(rand.Intn(int(r.config.ElectionTimeout)))
	
	if r.electionTimer == nil {
		r.electionTimer = time.NewTimer(timeout)
	} else {
		r.electionTimer.Stop()
		r.electionTimer.Reset(timeout)
	}
}

func (r *RaftConsensus) resetHeartbeatTimer() {
	if r.heartbeatTimer == nil {
		r.heartbeatTimer = time.NewTimer(r.config.HeartbeatInterval)
	} else {
		r.heartbeatTimer.Stop()
		r.heartbeatTimer.Reset(r.config.HeartbeatInterval)
	}
}

// Helper functions

func min(a, b uint64) uint64 {
	if a < b {
		return a
	}
	return b
}