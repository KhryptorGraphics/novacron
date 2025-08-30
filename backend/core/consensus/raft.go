package consensus

import (
	"context"
	"crypto/rand"
	"log"
	"math/big"
	"sync"
	"time"
)

// NodeState represents the state of a Raft node
type NodeState int

const (
	Follower NodeState = iota
	Candidate
	Leader
)

func (s NodeState) String() string {
	switch s {
	case Follower:
		return "Follower"
	case Candidate:
		return "Candidate" 
	case Leader:
		return "Leader"
	default:
		return "Unknown"
	}
}

// RaftNode represents a single node in the Raft cluster
type RaftNode struct {
	// Persistent state on all servers
	currentTerm int64
	votedFor    string
	log         []LogEntry
	
	// Volatile state on all servers
	commitIndex int64
	lastApplied int64
	
	// Volatile state on leaders (reinitialized after election)
	nextIndex  map[string]int64
	matchIndex map[string]int64
	
	// Node identification
	id       string
	peers    []string
	state    NodeState
	leaderID string
	
	// Timing and election
	electionTimeout  time.Duration
	heartbeatTimeout time.Duration
	lastHeartbeat    time.Time
	electionTimer    *time.Timer
	
	// Communication
	transport Transport
	
	// Synchronization
	mu sync.RWMutex
	
	// Channels for internal coordination
	applyCh  chan ApplyMsg
	ctx      context.Context
	cancel   context.CancelFunc
	
	// Statistics
	stats NodeStats
}

// LogEntry represents an entry in the distributed log
type LogEntry struct {
	Term    int64       `json:"term"`
	Index   int64       `json:"index"`
	Command interface{} `json:"command"`
	Data    []byte      `json:"data"`
}

// ApplyMsg is sent to the application when a log entry is committed
type ApplyMsg struct {
	CommandValid bool
	Command      interface{}
	CommandIndex int64
	
	// For snapshots
	SnapshotValid bool
	Snapshot      []byte
	SnapshotTerm  int64
	SnapshotIndex int64
}

// NodeStats tracks node performance metrics
type NodeStats struct {
	TermsLeader           int64     `json:"terms_leader"`
	ElectionsWon          int64     `json:"elections_won"`
	ElectionsLost         int64     `json:"elections_lost"`
	HeartbeatsSent        int64     `json:"heartbeats_sent"`
	HeartbeatsReceived    int64     `json:"heartbeats_received"`
	LogEntriesCommitted   int64     `json:"log_entries_committed"`
	LastLeaderElection    time.Time `json:"last_leader_election"`
	TotalDowntime         time.Duration `json:"total_downtime"`
	mu                    sync.RWMutex
}

// Transport interface for network communication
type Transport interface {
	SendRequestVote(ctx context.Context, nodeID string, req *RequestVoteArgs) (*RequestVoteReply, error)
	SendAppendEntries(ctx context.Context, nodeID string, req *AppendEntriesArgs) (*AppendEntriesReply, error)
	SendSnapshot(ctx context.Context, nodeID string, req *InstallSnapshotArgs) (*InstallSnapshotReply, error)
}

// RPC message types
type RequestVoteArgs struct {
	Term         int64  `json:"term"`
	CandidateID  string `json:"candidate_id"`
	LastLogIndex int64  `json:"last_log_index"`
	LastLogTerm  int64  `json:"last_log_term"`
}

type RequestVoteReply struct {
	Term        int64 `json:"term"`
	VoteGranted bool  `json:"vote_granted"`
}

type AppendEntriesArgs struct {
	Term         int64      `json:"term"`
	LeaderID     string     `json:"leader_id"`
	PrevLogIndex int64      `json:"prev_log_index"`
	PrevLogTerm  int64      `json:"prev_log_term"`
	Entries      []LogEntry `json:"entries"`
	LeaderCommit int64      `json:"leader_commit"`
}

type AppendEntriesReply struct {
	Term    int64 `json:"term"`
	Success bool  `json:"success"`
	
	// Fast backup optimization
	ConflictTerm  int64 `json:"conflict_term"`
	ConflictIndex int64 `json:"conflict_index"`
}

type InstallSnapshotArgs struct {
	Term              int64  `json:"term"`
	LeaderID          string `json:"leader_id"`
	LastIncludedIndex int64  `json:"last_included_index"`
	LastIncludedTerm  int64  `json:"last_included_term"`
	Data              []byte `json:"data"`
	Done              bool   `json:"done"`
}

type InstallSnapshotReply struct {
	Term int64 `json:"term"`
}

// NewRaftNode creates a new Raft node
func NewRaftNode(id string, peers []string, transport Transport) *RaftNode {
	ctx, cancel := context.WithCancel(context.Background())
	
	node := &RaftNode{
		id:               id,
		peers:            peers,
		state:            Follower,
		currentTerm:      0,
		votedFor:         "",
		log:              make([]LogEntry, 0),
		commitIndex:      0,
		lastApplied:      0,
		nextIndex:        make(map[string]int64),
		matchIndex:       make(map[string]int64),
		electionTimeout:  randomElectionTimeout(),
		heartbeatTimeout: 50 * time.Millisecond,
		transport:        transport,
		applyCh:          make(chan ApplyMsg, 100),
		ctx:              ctx,
		cancel:           cancel,
	}
	
	// Initialize peer indices
	for _, peer := range peers {
		if peer != id {
			node.nextIndex[peer] = 1
			node.matchIndex[peer] = 0
		}
	}
	
	return node
}

// Start starts the Raft node
func (rn *RaftNode) Start() {
	log.Printf("Starting Raft node %s with peers %v", rn.id, rn.peers)
	
	// Start the main loop
	go rn.run()
	
	// Start apply goroutine
	go rn.applyLoop()
}

// Stop stops the Raft node
func (rn *RaftNode) Stop() {
	log.Printf("Stopping Raft node %s", rn.id)
	rn.cancel()
}

// Submit submits a command to the cluster
func (rn *RaftNode) Submit(command interface{}) (int64, int64, bool) {
	rn.mu.Lock()
	defer rn.mu.Unlock()
	
	if rn.state != Leader {
		return 0, 0, false
	}
	
	// Create new log entry
	entry := LogEntry{
		Term:    rn.currentTerm,
		Index:   int64(len(rn.log)) + 1,
		Command: command,
	}
	
	// Add to log
	rn.log = append(rn.log, entry)
	
	log.Printf("Node %s: Added command to log at index %d, term %d", 
		rn.id, entry.Index, entry.Term)
	
	// For single-node clusters, commit immediately
	if len(rn.peers) == 1 {
		rn.commitIndex = entry.Index
		log.Printf("Node %s: Updated commit index to %d (single-node)", rn.id, rn.commitIndex)
	} else {
		// Start replication for multi-node clusters
		go rn.replicateToAll()
	}
	
	return entry.Index, entry.Term, true
}

// GetApplyChan returns the channel for applied commands
func (rn *RaftNode) GetApplyChan() <-chan ApplyMsg {
	return rn.applyCh
}

// IsLeader returns whether this node is the leader
func (rn *RaftNode) IsLeader() bool {
	rn.mu.RLock()
	defer rn.mu.RUnlock()
	return rn.state == Leader
}

// GetState returns the current state and term
func (rn *RaftNode) GetState() (int64, bool) {
	rn.mu.RLock()
	defer rn.mu.RUnlock()
	return rn.currentTerm, rn.state == Leader
}

// GetLeader returns the current leader ID
func (rn *RaftNode) GetLeader() string {
	rn.mu.RLock()
	defer rn.mu.RUnlock()
	return rn.leaderID
}

// GetStats returns node statistics
func (rn *RaftNode) GetStats() NodeStats {
	rn.stats.mu.RLock()
	defer rn.stats.mu.RUnlock()
	return rn.stats
}

// Main run loop
func (rn *RaftNode) run() {
	rn.resetElectionTimer()
	
	for {
		select {
		case <-rn.ctx.Done():
			return
		case <-rn.electionTimer.C:
			rn.mu.Lock()
			if rn.state != Leader {
				log.Printf("Node %s: Election timeout, starting election", rn.id)
				rn.startElection()
			}
			rn.mu.Unlock()
		}
	}
}

// Apply loop processes committed entries
func (rn *RaftNode) applyLoop() {
	for {
		select {
		case <-rn.ctx.Done():
			return
		default:
			rn.mu.Lock()
			
			if rn.commitIndex > rn.lastApplied {
				rn.lastApplied++
				entry := rn.log[rn.lastApplied-1]
				
				msg := ApplyMsg{
					CommandValid: true,
					Command:      entry.Command,
					CommandIndex: entry.Index,
				}
				
				rn.mu.Unlock()
				
				select {
				case rn.applyCh <- msg:
					rn.stats.mu.Lock()
					rn.stats.LogEntriesCommitted++
					rn.stats.mu.Unlock()
				case <-rn.ctx.Done():
					return
				}
			} else {
				rn.mu.Unlock()
				time.Sleep(10 * time.Millisecond)
			}
		}
	}
}

// Start election
func (rn *RaftNode) startElection() {
	rn.state = Candidate
	rn.currentTerm++
	rn.votedFor = rn.id
	rn.resetElectionTimer()
	
	log.Printf("Node %s: Starting election for term %d", rn.id, rn.currentTerm)
	
	// Vote for self
	votes := 1
	votesNeeded := len(rn.peers)/2 + 1
	
	
	// Check if we already have enough votes (single node case)
	if votes >= votesNeeded {
		rn.becomeLeader()
		return
	}
	
	lastLogIndex := int64(len(rn.log))
	lastLogTerm := int64(0)
	if lastLogIndex > 0 {
		lastLogTerm = rn.log[lastLogIndex-1].Term
	}
	
	// Request votes from all peers
	for _, peer := range rn.peers {
		if peer == rn.id {
			continue
		}
		
		go func(peerID string) {
			req := &RequestVoteArgs{
				Term:         rn.currentTerm,
				CandidateID:  rn.id,
				LastLogIndex: lastLogIndex,
				LastLogTerm:  lastLogTerm,
			}
			
			ctx, cancel := context.WithTimeout(rn.ctx, 100*time.Millisecond)
			defer cancel()
			
			reply, err := rn.transport.SendRequestVote(ctx, peerID, req)
			if err != nil {
				log.Printf("Node %s: Failed to request vote from %s: %v", rn.id, peerID, err)
				return
			}
			
			rn.mu.Lock()
			defer rn.mu.Unlock()
			
			// Check if we're still a candidate and in the same term
			if rn.state != Candidate || rn.currentTerm != req.Term {
				return
			}
			
			// Update term if newer
			if reply.Term > rn.currentTerm {
				rn.currentTerm = reply.Term
				rn.votedFor = ""
				rn.state = Follower
				rn.resetElectionTimer()
				return
			}
			
			// Count vote
			if reply.VoteGranted {
				votes++
				log.Printf("Node %s: Received vote from %s (%d/%d)", 
					rn.id, peerID, votes, votesNeeded)
				
				// Check if we won
				if votes >= votesNeeded {
					rn.becomeLeader()
				}
			}
		}(peer)
	}
}

// Become leader
func (rn *RaftNode) becomeLeader() {
	if rn.state != Candidate {
		return
	}
	
	log.Printf("Node %s: Became leader for term %d", rn.id, rn.currentTerm)
	
	rn.state = Leader
	rn.leaderID = rn.id
	
	// Initialize leader state
	lastLogIndex := int64(len(rn.log))
	for _, peer := range rn.peers {
		if peer != rn.id {
			rn.nextIndex[peer] = lastLogIndex + 1
			rn.matchIndex[peer] = 0
		}
	}
	
	// Update stats
	rn.stats.mu.Lock()
	rn.stats.ElectionsWon++
	rn.stats.TermsLeader++
	rn.stats.LastLeaderElection = time.Now()
	rn.stats.mu.Unlock()
	
	// Send initial heartbeats
	go rn.sendHeartbeats()
}

// Send heartbeats to all followers
func (rn *RaftNode) sendHeartbeats() {
	ticker := time.NewTicker(rn.heartbeatTimeout)
	defer ticker.Stop()
	
	for {
		select {
		case <-rn.ctx.Done():
			return
		case <-ticker.C:
			rn.mu.RLock()
			if rn.state != Leader {
				rn.mu.RUnlock()
				return
			}
			rn.mu.RUnlock()
			
			rn.replicateToAll()
		}
	}
}

// Replicate to all followers
func (rn *RaftNode) replicateToAll() {
	rn.mu.RLock()
	if rn.state != Leader {
		rn.mu.RUnlock()
		return
	}
	
	for _, peer := range rn.peers {
		if peer != rn.id {
			go rn.replicateToPeer(peer)
		}
	}
	rn.mu.RUnlock()
}

// Replicate to a specific peer
func (rn *RaftNode) replicateToPeer(peerID string) {
	rn.mu.Lock()
	if rn.state != Leader {
		rn.mu.Unlock()
		return
	}
	
	nextIndex := rn.nextIndex[peerID]
	prevLogIndex := nextIndex - 1
	prevLogTerm := int64(0)
	
	if prevLogIndex > 0 && prevLogIndex <= int64(len(rn.log)) {
		prevLogTerm = rn.log[prevLogIndex-1].Term
	}
	
	// Prepare entries to send
	var entries []LogEntry
	if nextIndex <= int64(len(rn.log)) {
		entries = rn.log[nextIndex-1:]
	}
	
	req := &AppendEntriesArgs{
		Term:         rn.currentTerm,
		LeaderID:     rn.id,
		PrevLogIndex: prevLogIndex,
		PrevLogTerm:  prevLogTerm,
		Entries:      entries,
		LeaderCommit: rn.commitIndex,
	}
	
	term := rn.currentTerm
	rn.mu.Unlock()
	
	ctx, cancel := context.WithTimeout(rn.ctx, 100*time.Millisecond)
	defer cancel()
	
	reply, err := rn.transport.SendAppendEntries(ctx, peerID, req)
	if err != nil {
		log.Printf("Node %s: Failed to send append entries to %s: %v", rn.id, peerID, err)
		return
	}
	
	rn.mu.Lock()
	defer rn.mu.Unlock()
	
	// Check if we're still leader and in the same term
	if rn.state != Leader || rn.currentTerm != term {
		return
	}
	
	// Update term if newer
	if reply.Term > rn.currentTerm {
		rn.currentTerm = reply.Term
		rn.votedFor = ""
		rn.state = Follower
		rn.leaderID = ""
		rn.resetElectionTimer()
		return
	}
	
	if reply.Success {
		// Update indices
		if len(entries) > 0 {
			rn.nextIndex[peerID] = entries[len(entries)-1].Index + 1
			rn.matchIndex[peerID] = entries[len(entries)-1].Index
		}
		
		// Update commit index
		rn.updateCommitIndex()
		
		rn.stats.mu.Lock()
		rn.stats.HeartbeatsSent++
		rn.stats.mu.Unlock()
	} else {
		// Backup next index
		rn.nextIndex[peerID] = max(1, rn.nextIndex[peerID]-1)
		log.Printf("Node %s: Append entries failed for %s, backing up to %d", 
			rn.id, peerID, rn.nextIndex[peerID])
	}
}

// Update commit index based on match indices
func (rn *RaftNode) updateCommitIndex() {
	if rn.state != Leader {
		return
	}
	
	// For single-node clusters, commit all entries immediately
	if len(rn.peers) == 1 {
		if int64(len(rn.log)) > rn.commitIndex {
			rn.commitIndex = int64(len(rn.log))
			log.Printf("Node %s: Updated commit index to %d (single-node)", rn.id, rn.commitIndex)
		}
		return
	}
	
	// Find the highest index that is replicated on a majority
	for n := int64(len(rn.log)); n > rn.commitIndex; n-- {
		count := 1 // Count self
		
		for _, peer := range rn.peers {
			if peer != rn.id && rn.matchIndex[peer] >= n {
				count++
			}
		}
		
		// Check if majority and from current term
		if count > len(rn.peers)/2 && rn.log[n-1].Term == rn.currentTerm {
			rn.commitIndex = n
			log.Printf("Node %s: Updated commit index to %d", rn.id, rn.commitIndex)
			break
		}
	}
}

// Reset election timer with random timeout
func (rn *RaftNode) resetElectionTimer() {
	if rn.electionTimer != nil {
		rn.electionTimer.Stop()
	}
	rn.electionTimer = time.NewTimer(randomElectionTimeout())
}

// Generate random election timeout
func randomElectionTimeout() time.Duration {
	// Random timeout between 150-300ms
	min := 150
	max := 300
	n, _ := rand.Int(rand.Reader, big.NewInt(int64(max-min)))
	return time.Duration(min+int(n.Int64())) * time.Millisecond
}

// Helper function
func max(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}

// RPC handlers

// HandleRequestVote handles RequestVote RPC
func (rn *RaftNode) HandleRequestVote(args *RequestVoteArgs) *RequestVoteReply {
	rn.mu.Lock()
	defer rn.mu.Unlock()
	
	reply := &RequestVoteReply{
		Term:        rn.currentTerm,
		VoteGranted: false,
	}
	
	// Reply false if term < currentTerm
	if args.Term < rn.currentTerm {
		return reply
	}
	
	// If RPC request or response contains term T > currentTerm:
	// set currentTerm = T, convert to follower
	if args.Term > rn.currentTerm {
		rn.currentTerm = args.Term
		rn.votedFor = ""
		rn.state = Follower
		rn.leaderID = ""
	}
	
	// Update term in reply
	reply.Term = rn.currentTerm
	
	lastLogIndex := int64(len(rn.log))
	lastLogTerm := int64(0)
	if lastLogIndex > 0 {
		lastLogTerm = rn.log[lastLogIndex-1].Term
	}
	
	// Grant vote if:
	// - Haven't voted for anyone else in this term
	// - Candidate's log is at least as up-to-date as receiver's log
	if (rn.votedFor == "" || rn.votedFor == args.CandidateID) &&
		(args.LastLogTerm > lastLogTerm || 
		 (args.LastLogTerm == lastLogTerm && args.LastLogIndex >= lastLogIndex)) {
		
		rn.votedFor = args.CandidateID
		reply.VoteGranted = true
		rn.resetElectionTimer()
		
		log.Printf("Node %s: Granted vote to %s for term %d", 
			rn.id, args.CandidateID, args.Term)
	}
	
	return reply
}

// HandleAppendEntries handles AppendEntries RPC
func (rn *RaftNode) HandleAppendEntries(args *AppendEntriesArgs) *AppendEntriesReply {
	rn.mu.Lock()
	defer rn.mu.Unlock()
	
	reply := &AppendEntriesReply{
		Term:    rn.currentTerm,
		Success: false,
	}
	
	// Reply false if term < currentTerm
	if args.Term < rn.currentTerm {
		return reply
	}
	
	// Convert to follower if newer term
	if args.Term > rn.currentTerm {
		rn.currentTerm = args.Term
		rn.votedFor = ""
		rn.state = Follower
	}
	
	// Update leader and reset election timer
	rn.leaderID = args.LeaderID
	rn.state = Follower
	rn.resetElectionTimer()
	
	rn.stats.mu.Lock()
	rn.stats.HeartbeatsReceived++
	rn.stats.mu.Unlock()
	
	reply.Term = rn.currentTerm
	
	// Reply false if log doesn't contain an entry at prevLogIndex
	// whose term matches prevLogTerm
	if args.PrevLogIndex > 0 {
		if args.PrevLogIndex > int64(len(rn.log)) {
			reply.ConflictIndex = int64(len(rn.log)) + 1
			return reply
		}
		
		if rn.log[args.PrevLogIndex-1].Term != args.PrevLogTerm {
			reply.ConflictTerm = rn.log[args.PrevLogIndex-1].Term
			// Find first index with conflicting term
			for i := args.PrevLogIndex - 1; i >= 1; i-- {
				if rn.log[i-1].Term != reply.ConflictTerm {
					reply.ConflictIndex = i + 1
					break
				}
			}
			if reply.ConflictIndex == 0 {
				reply.ConflictIndex = 1
			}
			return reply
		}
	}
	
	// If an existing entry conflicts with a new one (same index but different terms),
	// delete the existing entry and all that follow it
	for i, entry := range args.Entries {
		index := args.PrevLogIndex + int64(i) + 1
		if index <= int64(len(rn.log)) {
			if rn.log[index-1].Term != entry.Term {
				// Remove conflicting entries
				rn.log = rn.log[:index-1]
				break
			}
		}
	}
	
	// Append any new entries not already in the log
	for i, entry := range args.Entries {
		index := args.PrevLogIndex + int64(i) + 1
		if index > int64(len(rn.log)) {
			rn.log = append(rn.log, entry)
		}
	}
	
	// If leaderCommit > commitIndex, set commitIndex = min(leaderCommit, index of last new entry)
	if args.LeaderCommit > rn.commitIndex {
		lastNewIndex := args.PrevLogIndex + int64(len(args.Entries))
		rn.commitIndex = min(args.LeaderCommit, lastNewIndex)
	}
	
	reply.Success = true
	return reply
}

// Helper function
func min(a, b int64) int64 {
	if a < b {
		return a
	}
	return b
}