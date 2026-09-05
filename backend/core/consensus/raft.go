package consensus

import (
	"context"
	"crypto/rand"
	"log"
	"math/big"
	"sync"
	"time"
)

// RaftNodeState represents the state of a Raft node
type RaftNodeState int

const (
	Follower RaftNodeState = iota
	Candidate
	Leader
)

func (s RaftNodeState) String() string {
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
	// Leader check-quorum state. lastAck[peer] records when a peer last
	// responded to this node's AppendEntries (any reply proves reachability);
	// leadershipStart anchors the grace period before quorum enforcement
	// begins. Both are guarded by mu.
	lastAck         map[string]time.Time
	leadershipStart time.Time

	// Node identification
	nodeID   string
	peers    []string
	state    RaftNodeState
	leaderID string

	// Timing and election. The election deadline is tracked as
	// lastHeartbeat + electionTimeout and evaluated by run()'s poll loop
	// (see resetElectionTimer); there is deliberately no *time.Timer here,
	// because a timer object swapped out by a concurrent reset would strand
	// run()'s select on a stopped channel and no election would ever fire.
	electionTimeout  time.Duration
	heartbeatTimeout time.Duration
	lastHeartbeat    time.Time

	// lastLeaderContact is the pre-vote liveness clock: the last time this
	// node processed a message from a sitting leader (AppendEntries /
	// InstallSnapshot with term >= currentTerm). It is deliberately separate
	// from lastHeartbeat, which resetElectionTimer() also bumps on the node's
	// OWN campaign activity (startElection/startPreVote) and on granted votes
	// -- using it for the pre-vote freshness check would make a campaigning
	// node look leader-fed and refuse peers' probes, deadlocking re-election
	// after leader death. Guarded by mu. Zero value => probe-granting (a
	// freshly started node has no leader to protect).
	lastLeaderContact time.Time

	// preVoteInFlight is true while a pre-vote campaign is being fanned out.
	// Guarded by mu. Cleared on every term adoption and state transition.
	preVoteInFlight bool

	// Communication
	transport Transport
	storage   RaftStorage

	// Synchronization
	mu sync.RWMutex

	// Channels for internal coordination. applyCh is the default committed-entry
	// stream returned by GetApplyChan(); subscribers are additional independent
	// streams handed out by Subscribe(). applyLoop fans every committed entry
	// out to applyCh and to each subscriber, so multiple state machines attached
	// to one node (e.g. several LockManagers) each receive the full stream
	// instead of competing for messages on a single channel.
	applyCh     chan ApplyMsg
	subscribers []chan ApplyMsg
	subMu       sync.Mutex
	ctx         context.Context
	cancel      context.CancelFunc

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
	TermsLeader         int64         `json:"terms_leader"`
	ElectionsWon        int64         `json:"elections_won"`
	ElectionsLost       int64         `json:"elections_lost"`
	HeartbeatsSent      int64         `json:"heartbeats_sent"`
	HeartbeatsReceived  int64         `json:"heartbeats_received"`
	LogEntriesCommitted int64         `json:"log_entries_committed"`
	LastLeaderElection  time.Time     `json:"last_leader_election"`
	TotalDowntime       time.Duration `json:"total_downtime"`
	mu                  sync.RWMutex
}

// Transport interface for network communication
type Transport interface {
	SendRequestVote(ctx context.Context, nodeID string, req *RequestVoteArgs) (*RequestVoteReply, error)
	SendAppendEntries(ctx context.Context, nodeID string, req *AppendEntriesArgs) (*AppendEntriesReply, error)
	SendSnapshot(ctx context.Context, nodeID string, req *InstallSnapshotArgs) (*InstallSnapshotReply, error)
	// QueryLeaderState asks a peer node whether it currently believes itself
	// to be the cluster leader. Used by split-brain detection to observe
	// real remote leader state instead of assuming remote nodes are never
	// leaders.
	QueryLeaderState(ctx context.Context, nodeID string) (*LeaderStateReply, error)
}

// LeaderStateArgs requests a node's current leader-state snapshot.
type LeaderStateArgs struct {
	RequesterID string `json:"requester_id"`
}

// LeaderStateReply reports whether a node currently believes it is the
// cluster leader, along with its term and its view of the current leader.
type LeaderStateReply struct {
	NodeID   string `json:"node_id"`
	Term     int64  `json:"term"`
	IsLeader bool   `json:"is_leader"`
	LeaderID string `json:"leader_id"`
}

// RPC message types
type RequestVoteArgs struct {
	Term         int64  `json:"term"`
	CandidateID  string `json:"candidate_id"`
	LastLogIndex int64  `json:"last_log_index"`
	LastLogTerm  int64  `json:"last_log_term"`
	// PreVote marks this RequestVote as a pre-vote probe: a pre-vote is
	// answered from a read-only snapshot of the receiver's state and NEVER
	// mutates currentTerm, votedFor, state, leaderID, or the election timer,
	// so a partitioned candidate cannot inflate its term through rejected
	// probes and the healed majority's leader keeps its term. (Beads
	// novacron-fpg, novacron-5ng.)
	PreVote bool `json:"pre_vote"`
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

// NewRaftNode creates a new Raft node with non-durable (in-memory) storage.
// Existing callers keep working unmodified; use NewRaftNodeWithStorage for
// a node whose currentTerm/votedFor/log survive a process restart.
func NewRaftNode(id string, peers []string, transport Transport) *RaftNode {
	return NewRaftNodeWithStorage(id, peers, transport, NewInMemoryRaftStorage())
}

// NewRaftNodeWithStorage creates a new Raft node backed by the given
// durable storage. On creation it reloads any previously persisted
// currentTerm, votedFor, and log, so a restart can never forget an
// already-cast vote (double-vote) or lose committed entries. A nil storage
// falls back to a fresh, non-durable InMemoryRaftStorage.
func NewRaftNodeWithStorage(id string, peers []string, transport Transport, storage RaftStorage) *RaftNode {
	if storage == nil {
		storage = NewInMemoryRaftStorage()
	}

	ctx, cancel := context.WithCancel(context.Background())

	currentTerm, votedFor, persistedLog, err := storage.Load()
	if err != nil {
		log.Printf("Node %s: failed to load persisted raft state, starting fresh: %v", id, err)
		currentTerm, votedFor, persistedLog = 0, "", nil
	}
	if persistedLog == nil {
		persistedLog = make([]LogEntry, 0)
	}

	node := &RaftNode{
		nodeID:           id,
		peers:            peers,
		state:            Follower,
		currentTerm:      currentTerm,
		votedFor:         votedFor,
		log:              persistedLog,
		commitIndex:      0,
		lastApplied:      0,
		nextIndex:        make(map[string]int64),
		matchIndex:       make(map[string]int64),
		electionTimeout:  randomElectionTimeout(),
		heartbeatTimeout: 50 * time.Millisecond,
		transport:        transport,
		storage:          storage,
		applyCh:          make(chan ApplyMsg, 100),
		lastAck:          make(map[string]time.Time),
		leadershipStart:  time.Time{},
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
	log.Printf("Starting Raft node %s with peers %v", rn.nodeID, rn.peers)

	// Start the main loop
	go rn.run()

	// Start apply goroutine
	go rn.applyLoop()
}

// Stop stops the Raft node. It cancels the node's context; Done() exposes
// that cancellation so dependent consumers (e.g. LockManager.processCommands)
// can select on it and exit cleanly. applyCh itself is intentionally never
// closed here: applyLoop is the sole writer and closing a channel a
// still-running writer might send on would risk a send-on-closed-channel
// panic, so shutdown is coordinated via context cancellation instead.
func (rn *RaftNode) Stop() {
	log.Printf("Stopping Raft node %s", rn.nodeID)
	rn.cancel()
	if rn.storage != nil {
		if err := rn.storage.Close(); err != nil {
			log.Printf("Node %s: failed to close raft storage: %v", rn.nodeID, err)
		}
	}
}

// Done returns a channel that is closed once the node has been stopped.
// Goroutines that consume node output (e.g. GetApplyChan()) but must not
// block forever after Stop() should select on this alongside their channel
// read.
func (rn *RaftNode) Done() <-chan struct{} {
	return rn.ctx.Done()
}

// persistState durably saves currentTerm and votedFor. Callers MUST hold
// rn.mu. Best-effort: a failure is logged rather than propagated, matching
// this package's existing non-fatal handling of transport/IO errors.
func (rn *RaftNode) persistState() {
	if rn.storage == nil {
		return
	}
	if err := rn.storage.SaveState(rn.currentTerm, rn.votedFor); err != nil {
		log.Printf("Node %s: failed to persist raft term/vote: %v", rn.nodeID, err)
	}
}

// persistAppend durably appends newly-added log entries. Callers MUST hold
// rn.mu.
func (rn *RaftNode) persistAppend(entries []LogEntry) {
	if rn.storage == nil || len(entries) == 0 {
		return
	}
	if err := rn.storage.AppendLog(entries); err != nil {
		log.Printf("Node %s: failed to persist raft log entries: %v", rn.nodeID, err)
	}
}

// persistTruncate discards persisted log entries at and after fromIndex.
// Callers MUST hold rn.mu.
func (rn *RaftNode) persistTruncate(fromIndex int64) {
	if rn.storage == nil {
		return
	}
	if err := rn.storage.TruncateLog(fromIndex); err != nil {
		log.Printf("Node %s: failed to truncate persisted raft log: %v", rn.nodeID, err)
	}
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
	rn.persistAppend([]LogEntry{entry})

	log.Printf("Node %s: Added command to log at index %d, term %d",
		rn.nodeID, entry.Index, entry.Term)

	// For single-node clusters, commit immediately
	if len(rn.peers) == 1 {
		rn.commitIndex = entry.Index
		log.Printf("Node %s: Updated commit index to %d (single-node)", rn.nodeID, rn.commitIndex)
	} else {
		// Start replication for multi-node clusters
		go rn.replicateToAll()
	}

	return entry.Index, entry.Term, true
}

// GetApplyChan returns the default channel for applied commands. It is a
// single shared stream: only one consumer should range over it. Callers that
// need an independent, complete copy of the committed stream (so several
// consumers can each build identical state) must use Subscribe instead.
func (rn *RaftNode) GetApplyChan() <-chan ApplyMsg {
	return rn.applyCh
}

// Subscribe registers an additional consumer and returns a channel that will
// receive every entry committed after this call. Unlike GetApplyChan, each
// Subscribe caller gets its own channel and its own full copy of the stream,
// so multiple state machines (e.g. several LockManagers) attached to one node
// build identical applied state instead of racing for individual messages.
// Subscribe before submitting the commands you need to observe: entries
// committed prior to the call are not replayed onto the new channel.
func (rn *RaftNode) Subscribe() <-chan ApplyMsg {
	ch := make(chan ApplyMsg, cap(rn.applyCh))
	rn.subMu.Lock()
	rn.subscribers = append(rn.subscribers, ch)
	rn.subMu.Unlock()
	return ch
}

// deliver fans a committed ApplyMsg out to the default apply channel and to
// every channel registered via Subscribe. It returns false if the node was
// stopped mid-send so applyLoop can exit. Sends are blocking (bounded by each
// channel's buffer) to preserve at-most-once, in-order delivery; a consumer
// that never drains its channel can stall delivery, so every subscriber is
// expected to read continuously until the node stops.
func (rn *RaftNode) deliver(msg ApplyMsg) bool {
	select {
	case rn.applyCh <- msg:
	case <-rn.ctx.Done():
		return false
	}

	rn.subMu.Lock()
	subs := make([]chan ApplyMsg, len(rn.subscribers))
	copy(subs, rn.subscribers)
	rn.subMu.Unlock()

	for _, ch := range subs {
		select {
		case ch <- msg:
		case <-rn.ctx.Done():
			return false
		}
	}
	return true
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

// GetStats returns a snapshot of node statistics.
func (rn *RaftNode) GetStats() NodeStats {
	rn.stats.mu.RLock()
	defer rn.stats.mu.RUnlock()
	// Field-wise snapshot: NodeStats embeds a sync.RWMutex, so returning the
	// struct by value copies the lock (go vet "return copies lock value").
	// Copy only the data fields; the zero-value mutex of the snapshot is
	// private to the copy.
	return NodeStats{
		TermsLeader:         rn.stats.TermsLeader,
		ElectionsWon:        rn.stats.ElectionsWon,
		ElectionsLost:       rn.stats.ElectionsLost,
		HeartbeatsSent:      rn.stats.HeartbeatsSent,
		HeartbeatsReceived:  rn.stats.HeartbeatsReceived,
		LogEntriesCommitted: rn.stats.LogEntriesCommitted,
		LastLeaderElection:  rn.stats.LastLeaderElection,
		TotalDowntime:       rn.stats.TotalDowntime,
	}
}

// Main run loop
func (rn *RaftNode) run() {
	rn.mu.Lock()
	rn.resetElectionTimer()
	rn.mu.Unlock()

	// Poll the election deadline on a short tick. Comparing elapsed time
	// against lastHeartbeat (instead of selecting on a swappable *time.Timer.C)
	// is robust to resets issued from other goroutines -- an incoming heartbeat
	// or granted vote just moves lastHeartbeat forward, and the next tick sees
	// it. The old timer-channel approach could park this select forever on a
	// stopped timer after such a reset, so a follower whose leader died never
	// started a new election.
	ticker := time.NewTicker(10 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-rn.ctx.Done():
			return
		case <-ticker.C:
			rn.mu.Lock()
			if rn.state != Leader {
				// Follower/candidate election-timeout path
				if time.Since(rn.lastHeartbeat) >= rn.electionTimeout {
					log.Printf("Node %s: Election timeout, starting pre-vote", rn.nodeID)
					rn.startPreVote()
				}
			} else if rn.leadershipStart.Add(rn.electionTimeout).Before(time.Now()) {
				// Leader check-quorum (Raft thesis §9.6, etcd "check-quorum"):
				// after a full election-timeout grace period in office, if a
				// majority of peers has not acked within one election timeout,
				// this leadership is no longer backed by quorum. Step down
				// rather than serve a minority as leader -- the stale-claim
				// that survives a partition is the split-brain window this
				// node can otherwise never observe across the partition.
				acked := 1 // self
				for _, at := range rn.lastAck {
					if time.Since(at) < rn.electionTimeout {
						acked++
					}
				}
				if acked <= len(rn.peers)/2 {
					log.Printf("Node %s: check-quorum failed (%d/%d acks within %v), stepping down from term %d",
						rn.nodeID, acked, len(rn.peers), rn.electionTimeout, rn.currentTerm)
					rn.stepDownToFollower()
				}
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

				if !rn.deliver(msg) {
					return
				}

				rn.stats.mu.Lock()
				rn.stats.LogEntriesCommitted++
				rn.stats.mu.Unlock()
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
	rn.votedFor = rn.nodeID
	rn.persistState()
	rn.resetElectionTimer()

	log.Printf("Node %s: Starting election for term %d", rn.nodeID, rn.currentTerm)

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
		if peer == rn.nodeID {
			continue
		}

		go func(peerID string) {
			req := &RequestVoteArgs{
				Term:         rn.currentTerm,
				CandidateID:  rn.nodeID,
				LastLogIndex: lastLogIndex,
				LastLogTerm:  lastLogTerm,
			}

			ctx, cancel := context.WithTimeout(rn.ctx, 100*time.Millisecond)
			defer cancel()

			reply, err := rn.transport.SendRequestVote(ctx, peerID, req)
			if err != nil {
				log.Printf("Node %s: Failed to request vote from %s: %v", rn.nodeID, peerID, err)
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
				rn.preVoteInFlight = false
				rn.persistState()
				rn.resetElectionTimer()
				return
			}

			// Count vote
			if reply.VoteGranted {
				votes++
				log.Printf("Node %s: Received vote from %s (%d/%d)",
					rn.nodeID, peerID, votes, votesNeeded)

				// Check if we won
				if votes >= votesNeeded {
					rn.becomeLeader()
				}
			}
		}(peer)
	}
}

// startPreVote runs the pre-vote phase of an election (beads novacron-fpg,
// novacron-5ng). Callers MUST hold rn.mu (run()'s tick path holds it; the
// pre-vote reply path re-enters under lock).
//
// Instead of blindly incrementing the term, the node first probes the
// cluster with PreVote RequestVote RPCs (Term = currentTerm + 1, answered
// read-only by receivers). Only when a majority grants the probes does the
// node proceed to a real startElection() -- at which point its term
// increment can actually win. A minority partition can therefore never
// inflate its term past a majority the majority itself would grant: each
// probe is refused by any receiver that still hears its leader, so the
// healed cluster keeps the majority leader's term. If the probe round is
// denied or times out, the flag clears and the next election timeout
// retries the probe; the term is untouched either way.
func (rn *RaftNode) startPreVote() {
	if rn.preVoteInFlight {
		return
	}
	rn.preVoteInFlight = true
	rn.resetElectionTimer()

	log.Printf("Node %s: Starting pre-vote for prospective term %d", rn.nodeID, rn.currentTerm+1)

	proposedTerm := rn.currentTerm + 1
	lastLogIndex := int64(len(rn.log))
	lastLogTerm := int64(0)
	if lastLogIndex > 0 {
		lastLogTerm = rn.log[lastLogIndex-1].Term
	}

	needed := len(rn.peers)/2 + 1
	// Single node (or no quorum possible with no peers beyond self): go
	// straight to the real election -- the term increment is safe because
	// there is nobody to disrupt.
	if needed <= 1 {
		rn.preVoteInFlight = false
		rn.startElection()
		return
	}

	granted := 1 // self
	responded := 0
	peerCount := len(rn.peers) - 1

	for _, peer := range rn.peers {
		if peer == rn.nodeID {
			continue
		}

		go func(peerID string) {
			req := &RequestVoteArgs{
				Term:         proposedTerm,
				CandidateID:  rn.nodeID,
				LastLogIndex: lastLogIndex,
				LastLogTerm:  lastLogTerm,
				PreVote:      true,
			}

			ctx, cancel := context.WithTimeout(rn.ctx, 100*time.Millisecond)
			defer cancel()

			reply, err := rn.transport.SendRequestVote(ctx, peerID, req)

			rn.mu.Lock()
			defer rn.mu.Unlock()

			// The campaign is over if it was superseded (leader won
			// elsewhere / a real election started / term adopted).
			if !rn.preVoteInFlight || rn.state == Leader {
				return
			}

			if err == nil && reply.Term > rn.currentTerm {
				// A peer is in a term we have not seen: adopt it. This is
				// the only pre-vote outcome that moves our term, and it is
				// driven by REAL information (the peer's current term), not
				// by our own speculative increment.
				rn.currentTerm = reply.Term
				rn.votedFor = ""
				rn.state = Follower
				rn.preVoteInFlight = false
				rn.persistState()
				rn.resetElectionTimer()
				return
			}

			if err == nil && reply.VoteGranted {
				granted++
			}

			responded++
			switch {
			case granted >= needed:
				// Majority of the cluster would grant a real vote in
				// proposedTerm: run the real election. granted/responded
				// are dead after this, and preVoteInFlight=false makes any
				// straggler reply a no-op.
				rn.preVoteInFlight = false
				rn.startElection()
			case responded == peerCount:
				// All peers answered but quorum denied: back off until the
				// next election timeout.
				rn.preVoteInFlight = false
				rn.resetElectionTimer()
			}
		}(peer)
	}
}

// Become leader
func (rn *RaftNode) becomeLeader() {
	if rn.state != Candidate {
		return
	}

	log.Printf("Node %s: Became leader for term %d", rn.nodeID, rn.currentTerm)

	rn.state = Leader
	rn.leaderID = rn.nodeID
	// Defensive: a leadership can only begin via startElection (the pre-vote
	// campaign already cleared its flag), but keep the invariant explicit --
	// no pre-vote may be in flight while leading.
	rn.preVoteInFlight = false
	// begins runs from this instant.
	rn.leadershipStart = time.Now()
	// Fresh quorum-tracking state for the new leadership term.
	for peer := range rn.lastAck {
		rn.lastAck[peer] = time.Time{}
	}

	// Initialize leader state
	lastLogIndex := int64(len(rn.log))
	for _, peer := range rn.peers {
		if peer != rn.nodeID {
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

// stepDownToFollower demotes the node to follower in its current term. Callers
// MUST hold rn.mu. currentTerm and votedFor are deliberately untouched: a
// same-term step-down must not clear votedFor (that would allow double-voting
// within the term), and the term is unchanged by construction. The election
// timer IS reset so the demoted node can campaign again. Any in-flight
// pre-vote campaign is abandoned with the leadership.
func (rn *RaftNode) stepDownToFollower() {
	rn.state = Follower
	rn.leaderID = ""
	rn.preVoteInFlight = false
	rn.resetElectionTimer()
}
func (rn *RaftNode) replicateToAll() {
	rn.mu.RLock()
	if rn.state != Leader {
		rn.mu.RUnlock()
		return
	}

	for _, peer := range rn.peers {
		if peer != rn.nodeID {
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
		LeaderID:     rn.nodeID,
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
		log.Printf("Node %s: Failed to send append entries to %s: %v", rn.nodeID, peerID, err)
		return
	}

	rn.mu.Lock()
	defer rn.mu.Unlock()

	// Check if we're still leader and in the same term
	if rn.state != Leader || rn.currentTerm != term {
		return
	}

	// A reply proves the peer is reachable and still in this term: record
	// the ack regardless of Success (a log-mismatch reply still proves
	// liveness), before the state checks below.
	rn.lastAck[peerID] = time.Now()

	// Update term if newer
	if reply.Term > rn.currentTerm {
		rn.currentTerm = reply.Term
		rn.votedFor = ""
		rn.state = Follower
		rn.leaderID = ""
		rn.preVoteInFlight = false
		rn.persistState()
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
			rn.nodeID, peerID, rn.nextIndex[peerID])
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
			log.Printf("Node %s: Updated commit index to %d (single-node)", rn.nodeID, rn.commitIndex)
		}
		return
	}

	// Find the highest index that is replicated on a majority
	for n := int64(len(rn.log)); n > rn.commitIndex; n-- {
		count := 1 // Count self

		for _, peer := range rn.peers {
			if peer != rn.nodeID && rn.matchIndex[peer] >= n {
				count++
			}
		}

		// Check if majority and from current term
		if count > len(rn.peers)/2 && rn.log[n-1].Term == rn.currentTerm {
			rn.commitIndex = n
			log.Printf("Node %s: Updated commit index to %d", rn.nodeID, rn.commitIndex)
			break
		}
	}
}

// resetElectionTimer records a fresh election deadline: the current time plus a
// new random timeout. run()'s poll loop starts an election once
// time.Since(lastHeartbeat) >= electionTimeout. Callers MUST hold rn.mu.
// (Replaces an earlier *time.Timer implementation whose timer object was
// swapped out from under run()'s select, which stranded the loop after a
// concurrent reset and prevented failover elections.)
func (rn *RaftNode) resetElectionTimer() {
	rn.lastHeartbeat = time.Now()
	rn.electionTimeout = randomElectionTimeout()
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

	// Reply false if term < currentTerm (stale candidate). Computed BEFORE
	// the pre-vote branch: a pre-vote is answered from a read-only snapshot
	// and must not advance the receiver's term.
	if args.Term < rn.currentTerm {
		return reply
	}

	// Compute the receiver's log recency exactly as the real-vote grant
	// check below does; the pre-vote branch needs it too.
	lastLogIndex := int64(len(rn.log))
	lastLogTerm := int64(0)
	if lastLogIndex > 0 {
		lastLogTerm = rn.log[lastLogIndex-1].Term
	}

	// Pre-vote probe: answered read-only. The grant mirrors the real-vote
	// log-recency rule but adds two liveness conditions -- the receiver must
	// not currently be leading, and it must not currently believe it has a
	// living leader. Leader liveness is measured on lastLeaderContact, the
	// clock bumped ONLY by leader-originated messages (see the field comment):
	// lastHeartbeat is also bumped by the node's own campaign activity and
	// would make a campaigning node look leader-fed. A pre-vote NEVER mutates
	// currentTerm, votedFor, state, leaderID, or the election timer: a node
	// still hearing its leader refuses, so a partitioned candidate cannot
	// inflate terms and disrupt the majority on heal (beads novacron-fpg,
	// novacron-5ng).
	if args.PreVote {
		grantsLog := args.LastLogTerm > lastLogTerm ||
			(args.LastLogTerm == lastLogTerm && args.LastLogIndex >= lastLogIndex)
		// Zero lastLeaderContact means "no leader contact ever observed":
		// a fresh node must grant probes so a cluster can elect its first
		// leader (and re-elect after this node restarts).
		noLeaderContact := rn.lastLeaderContact.IsZero() ||
			time.Since(rn.lastLeaderContact) >= rn.electionTimeout
		reply.VoteGranted = args.Term >= rn.currentTerm &&
			rn.state != Leader &&
			noLeaderContact &&
			grantsLog
		return reply
	}
	// If RPC request or response contains term T > currentTerm:
	// set currentTerm = T, convert to follower
	if args.Term > rn.currentTerm {
		rn.currentTerm = args.Term
		rn.votedFor = ""
		rn.state = Follower
		rn.leaderID = ""
		rn.preVoteInFlight = false
		rn.persistState()
	}

	// Update term in reply
	reply.Term = rn.currentTerm

	// Grant vote if:
	// - Haven't voted for anyone else in this term
	// - Candidate's log is at least as up-to-date as receiver's log
	if (rn.votedFor == "" || rn.votedFor == args.CandidateID) &&
		(args.LastLogTerm > lastLogTerm ||
			(args.LastLogTerm == lastLogTerm && args.LastLogIndex >= lastLogIndex)) {

		rn.votedFor = args.CandidateID
		rn.persistState()
		reply.VoteGranted = true
		rn.resetElectionTimer()

		log.Printf("Node %s: Granted vote to %s for term %d",
			rn.nodeID, args.CandidateID, args.Term)
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
		rn.preVoteInFlight = false
		rn.persistState()
	}

	// Update leader and reset election timer. This is genuine leader
	// contact: it feeds both the election timer and the pre-vote liveness
	// clock lastLeaderContact, which the pre-vote receiver branch reads.
	rn.leaderID = args.LeaderID
	rn.state = Follower
	rn.lastLeaderContact = time.Now()
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
				rn.persistTruncate(index)
				break
			}
		}
	}

	// Append any new entries not already in the log
	var appended []LogEntry
	for i, entry := range args.Entries {
		index := args.PrevLogIndex + int64(i) + 1
		if index > int64(len(rn.log)) {
			rn.log = append(rn.log, entry)
			appended = append(appended, entry)
		}
	}
	rn.persistAppend(appended)

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

// HandleQueryLeaderState reports this node's current leader-state snapshot.
// Remote split-brain detection uses this to observe whether a peer actually
// believes itself to be the leader, instead of assuming it never is.
func (rn *RaftNode) HandleQueryLeaderState() *LeaderStateReply {
	rn.mu.RLock()
	defer rn.mu.RUnlock()
	return &LeaderStateReply{
		NodeID:   rn.nodeID,
		Term:     rn.currentTerm,
		IsLeader: rn.state == Leader,
		LeaderID: rn.leaderID,
	}
}
