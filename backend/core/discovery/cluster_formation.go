package discovery

import (
	"context"
	"crypto/rand"
	"encoding/binary"
	"fmt"
	"log"
	"sort"
	"sync"
	"time"

	"go.uber.org/zap"
)

// ClusterState represents the state of a cluster
type ClusterState string

const (
	// ClusterStateInitializing indicates the cluster is initializing
	ClusterStateInitializing ClusterState = "initializing"

	// ClusterStateForming indicates the cluster is in the process of forming
	ClusterStateForming ClusterState = "forming"

	// ClusterStateJoining indicates the node is joining an existing cluster
	ClusterStateJoining ClusterState = "joining"

	// ClusterStateLeaving indicates the node is leaving the cluster
	ClusterStateLeaving ClusterState = "leaving"

	// ClusterStateFormed indicates the cluster is fully formed
	ClusterStateFormed ClusterState = "formed"

	// ClusterStateDegraded indicates the cluster is formed but in a degraded state
	ClusterStateDegraded ClusterState = "degraded"

	// ClusterStateSplit indicates the cluster has split into multiple clusters
	ClusterStateSplit ClusterState = "split"
)

// LeaderState represents the state of a node in leader election
type LeaderState string

const (
	// LeaderStateFollower indicates the node is a follower
	LeaderStateFollower LeaderState = "follower"

	// LeaderStateCandidate indicates the node is a candidate for leadership
	LeaderStateCandidate LeaderState = "candidate"

	// LeaderStateLeader indicates the node is the leader
	LeaderStateLeader LeaderState = "leader"

	// LeaderStateObserver indicates the node is an observer (non-voting)
	LeaderStateObserver LeaderState = "observer"
)

// ClusterFormationConfig contains configuration for cluster formation
type ClusterFormationConfig struct {
	// Configuration for the underlying discovery service
	DiscoveryConfig InternetDiscoveryConfig

	// Minimum number of manager nodes required for a healthy cluster
	MinManagerNodes int

	// Maximum number of manager nodes for optimal performance
	MaxManagerNodes int

	// Whether this node can be elected as a leader
	CanBeLeader bool

	// Leader election timeout range (min, max) in milliseconds
	ElectionTimeoutRange [2]int

	// Heartbeat interval in milliseconds
	HeartbeatIntervalMs int

	// Cluster name
	ClusterName string

	// Cluster secret for authentication
	ClusterSecret string

	// Join token for joining the cluster (if any)
	JoinToken string

	// Maximum time to wait for cluster formation
	FormationTimeout time.Duration
}

// DefaultClusterFormationConfig returns the default configuration for cluster formation
func DefaultClusterFormationConfig() ClusterFormationConfig {
	return ClusterFormationConfig{
		DiscoveryConfig:      DefaultInternetDiscoveryConfig(),
		MinManagerNodes:      1,
		MaxManagerNodes:      7,
		CanBeLeader:          true,
		ElectionTimeoutRange: [2]int{150, 300},
		HeartbeatIntervalMs:  50,
		ClusterName:          "novacron-cluster",
		ClusterSecret:        "",
		JoinToken:            "",
		FormationTimeout:     2 * time.Minute,
	}
}

// ClusterMemberStatus represents the status of a cluster member
type ClusterMemberStatus struct {
	// Node information
	NodeInfo

	// Leader state
	LeaderState LeaderState

	// Whether this node is active in the cluster
	Active bool

	// Current term in the leader election
	CurrentTerm uint64

	// ID of the leader from this node's perspective
	LeaderID string

	// Last time a heartbeat was received from the leader
	LastHeartbeat time.Time

	// Connectivity quality (0-100)
	ConnectivityQuality int

	// Whether this node has voted in the current term
	VotedInCurrentTerm bool

	// ID of the node this node voted for in the current term
	VotedFor string
}

// ClusterStatus represents the status of a cluster
type ClusterStatus struct {
	// Current state of the cluster
	State ClusterState

	// Cluster ID
	ID string

	// Cluster name
	Name string

	// Leader ID (if any)
	LeaderID string

	// Current term in the leader election
	CurrentTerm uint64

	// IDs of manager nodes in the cluster
	ManagerNodeIDs []string

	// IDs of worker nodes in the cluster
	WorkerNodeIDs []string

	// Total number of nodes in the cluster
	TotalNodes int

	// Number of healthy nodes in the cluster
	HealthyNodes int

	// When the cluster was formed
	FormedAt time.Time

	// Current members of the cluster
	Members map[string]ClusterMemberStatus
}

// ClusterEventType represents the type of cluster event
type ClusterEventType string

const (
	// ClusterEventStateChanged indicates the cluster state has changed
	ClusterEventStateChanged ClusterEventType = "state_changed"

	// ClusterEventLeaderChanged indicates the cluster leader has changed
	ClusterEventLeaderChanged ClusterEventType = "leader_changed"

	// ClusterEventMemberJoined indicates a new member has joined the cluster
	ClusterEventMemberJoined ClusterEventType = "member_joined"

	// ClusterEventMemberLeft indicates a member has left the cluster
	ClusterEventMemberLeft ClusterEventType = "member_left"

	// ClusterEventMemberFailed indicates a member has failed
	ClusterEventMemberFailed ClusterEventType = "member_failed"

	// ClusterEventConfigChanged indicates the cluster configuration has changed
	ClusterEventConfigChanged ClusterEventType = "config_changed"

	// ClusterEventElectionStarted indicates a leader election has started
	ClusterEventElectionStarted ClusterEventType = "election_started"
)

// ClusterEvent represents an event in the cluster
type ClusterEvent struct {
	// Type of event
	Type ClusterEventType

	// ID of the node that triggered the event
	NodeID string

	// Additional data for the event
	Data interface{}

	// Time of the event
	Timestamp time.Time
}

// ClusterEventListener is a function that handles cluster events
type ClusterEventListener func(event ClusterEvent)

// HeartbeatMessage is sent from the leader to followers
type HeartbeatMessage struct {
	// Current term
	Term uint64

	// Leader ID
	LeaderID string

	// Current cluster configuration
	ClusterStatus ClusterStatus

	// Timestamp of the message
	Timestamp time.Time
}

// VoteRequest is sent by candidates to request votes
type VoteRequest struct {
	// Candidate's term
	Term uint64

	// Candidate's ID
	CandidateID string

	// Candidate's last log index
	LastLogIndex uint64

	// Candidate's last log term
	LastLogTerm uint64
}

// VoteResponse is sent in response to a vote request
type VoteResponse struct {
	// Current term, for candidate to update itself
	Term uint64

	// Whether the vote was granted
	VoteGranted bool

	// ID of the node responding
	NodeID string
}

// AppendEntriesRequest is sent by the leader to replicate log entries
type AppendEntriesRequest struct {
	// Leader's term
	Term uint64

	// Leader's ID
	LeaderID string

	// Index of log entry immediately preceding new ones
	PrevLogIndex uint64

	// Term of prevLogIndex entry
	PrevLogTerm uint64

	// Log entries to store (empty for heartbeat)
	Entries []LogEntry

	// Leader's commit index
	LeaderCommit uint64
}

// AppendEntriesResponse is sent in response to append entries
type AppendEntriesResponse struct {
	// Current term, for leader to update itself
	Term uint64

	// Success indicator
	Success bool

	// ID of the node responding
	NodeID string

	// Next index to try for this follower
	NextIndex uint64
}

// LogEntry represents an entry in the Raft log
type LogEntry struct {
	// Term when entry was received by leader
	Term uint64

	// Index of the log entry
	Index uint64

	// Command to apply to the state machine
	Command []byte

	// Type of command
	CommandType string
}

// ClusterFormation manages the formation and operation of a cluster
type ClusterFormation struct {
	// Configuration
	config ClusterFormationConfig

	// Current status of the cluster
	status ClusterStatus

	// Status of this node
	selfStatus ClusterMemberStatus

	// Underlying discovery service
	discovery *InternetDiscoveryService

	// Event listeners
	listeners []ClusterEventListener

	// Raft log entries
	log []LogEntry

	// Last applied log index
	lastApplied uint64

	// Commit index
	commitIndex uint64

	// Next index for each node
	nextIndex map[string]uint64

	// Match index for each node
	matchIndex map[string]uint64

	// Election timer
	electionTimer *time.Timer

	// Heartbeat timer
	heartbeatTimer *time.Timer

	// Random number generator for election timeouts
	randGenerator *randSource

	// Leader election vote count
	voteCount int

	// Mutexes
	statusMutex    sync.RWMutex
	listenersMutex sync.RWMutex
	logMutex       sync.RWMutex
	electionMutex  sync.RWMutex
	heartbeatMutex sync.RWMutex

	// Context for shutdown
	ctx       context.Context
	cancel    context.CancelFunc
	isRunning bool
	runMutex  sync.RWMutex
}

// randSource provides a safe source of random numbers
type randSource struct {
	mu  sync.Mutex
	src []byte
}

// Int returns a random integer in the range [0,n)
func (r *randSource) Int(n int) int {
	r.mu.Lock()
	defer r.mu.Unlock()

	if len(r.src) < 8 {
		r.src = make([]byte, 8)
		rand.Read(r.src)
	}

	num := int(binary.BigEndian.Uint64(r.src) % uint64(n))

	// Refill random source
	rand.Read(r.src)

	return num
}

// NewClusterFormation creates a new cluster formation manager
func NewClusterFormation(config ClusterFormationConfig) (*ClusterFormation, error) {
	// Create discovery service
	logger := zap.NewNop() // TODO: Get logger from config
	discovery, err := NewInternetDiscovery(config.DiscoveryConfig, logger)
	if err != nil {
		return nil, fmt.Errorf("failed to create discovery service: %w", err)
	}

	ctx, cancel := context.WithCancel(context.Background())

	// Initialize cluster status
	status := ClusterStatus{
		State:        ClusterStateInitializing,
		ID:           generateClusterID(config.ClusterName),
		Name:         config.ClusterName,
		CurrentTerm:  0,
		TotalNodes:   0,
		HealthyNodes: 0,
		Members:      make(map[string]ClusterMemberStatus),
	}

	// Initialize self status
	selfStatus := ClusterMemberStatus{
		NodeInfo: NodeInfo{
			ID:        config.DiscoveryConfig.NodeID,
			Name:      config.DiscoveryConfig.NodeName,
			Role:      config.DiscoveryConfig.NodeRole,
			Address:   config.DiscoveryConfig.Address,
			Port:      config.DiscoveryConfig.Port,
			Available: true,
		},
		LeaderState:         LeaderStateFollower,
		Active:              true,
		CurrentTerm:         0,
		LeaderID:            "",
		LastHeartbeat:       time.Now(),
		ConnectivityQuality: 100,
		VotedInCurrentTerm:  false,
		VotedFor:            "",
	}

	// Create random source
	randSrc := &randSource{
		src: make([]byte, 8),
	}
	rand.Read(randSrc.src)

	cf := &ClusterFormation{
		config:        config,
		status:        status,
		selfStatus:    selfStatus,
		discovery:     discovery,
		listeners:     make([]ClusterEventListener, 0),
		log:           make([]LogEntry, 0),
		lastApplied:   0,
		commitIndex:   0,
		nextIndex:     make(map[string]uint64),
		matchIndex:    make(map[string]uint64),
		randGenerator: randSrc,
		ctx:           ctx,
		cancel:        cancel,
		isRunning:     false,
	}

	// Setup discovery service listener
	discovery.AddListener(cf.handleNodeEvent)

	return cf, nil
}

// Start starts the cluster formation process
func (cf *ClusterFormation) Start() error {
	cf.runMutex.Lock()
	defer cf.runMutex.Unlock()

	if cf.isRunning {
		return fmt.Errorf("cluster formation already running")
	}

	// Start discovery service
	if err := cf.discovery.Start(); err != nil {
		return fmt.Errorf("failed to start discovery service: %w", err)
	}

	// Initialize election timer
	cf.resetElectionTimer()

	// Start cluster formation
	cf.setClusterState(ClusterStateForming)

	// Start cluster formation process in a goroutine
	go cf.formCluster()

	cf.isRunning = true
	return nil
}

// Stop stops the cluster formation process
func (cf *ClusterFormation) Stop() error {
	cf.runMutex.Lock()
	defer cf.runMutex.Unlock()

	if !cf.isRunning {
		return nil
	}

	// Cancel context to stop all goroutines
	cf.cancel()

	// Stop timers
	if cf.electionTimer != nil {
		cf.electionTimer.Stop()
	}
	if cf.heartbeatTimer != nil {
		cf.heartbeatTimer.Stop()
	}

	// Stop discovery service
	if err := cf.discovery.Stop(); err != nil {
		return fmt.Errorf("failed to stop discovery service: %w", err)
	}

	cf.isRunning = false
	return nil
}

// AddListener adds a listener for cluster events
func (cf *ClusterFormation) AddListener(listener ClusterEventListener) {
	cf.listenersMutex.Lock()
	defer cf.listenersMutex.Unlock()

	cf.listeners = append(cf.listeners, listener)
}

// RemoveListener removes a listener for cluster events
func (cf *ClusterFormation) RemoveListener(listener ClusterEventListener) {
	cf.listenersMutex.Lock()
	defer cf.listenersMutex.Unlock()

	for i, l := range cf.listeners {
		if fmt.Sprintf("%p", l) == fmt.Sprintf("%p", listener) {
			cf.listeners = append(cf.listeners[:i], cf.listeners[i+1:]...)
			break
		}
	}
}

// GetStatus returns the current status of the cluster
func (cf *ClusterFormation) GetStatus() ClusterStatus {
	cf.statusMutex.RLock()
	defer cf.statusMutex.RUnlock()

	// Make a copy of the status to avoid race conditions
	status := cf.status
	status.Members = make(map[string]ClusterMemberStatus, len(cf.status.Members))
	for id, member := range cf.status.Members {
		status.Members[id] = member
	}

	return status
}

// GetSelfStatus returns the status of this node
func (cf *ClusterFormation) GetSelfStatus() ClusterMemberStatus {
	cf.statusMutex.RLock()
	defer cf.statusMutex.RUnlock()

	return cf.selfStatus
}

// IsLeader returns whether this node is the leader
func (cf *ClusterFormation) IsLeader() bool {
	cf.statusMutex.RLock()
	defer cf.statusMutex.RUnlock()

	return cf.selfStatus.LeaderState == LeaderStateLeader
}

// GetLeaderID returns the ID of the current leader
func (cf *ClusterFormation) GetLeaderID() string {
	cf.statusMutex.RLock()
	defer cf.statusMutex.RUnlock()

	return cf.status.LeaderID
}

// handleNodeEvent handles events from the discovery service
func (cf *ClusterFormation) handleNodeEvent(eventType EventType, nodeInfo NodeInfo) {
	switch eventType {
	case EventNodeJoined:
		cf.handleNodeJoined(nodeInfo)
	case EventNodeLeft:
		cf.handleNodeLeft(nodeInfo)
	case EventNodeUpdated:
		cf.handleNodeUpdated(nodeInfo)
	}
}

// handleNodeJoined handles a node joining event
func (cf *ClusterFormation) handleNodeJoined(nodeInfo NodeInfo) {
	cf.statusMutex.Lock()
	defer cf.statusMutex.Unlock()

	// Add node to members
	cf.status.Members[nodeInfo.ID] = ClusterMemberStatus{
		NodeInfo:            nodeInfo,
		LeaderState:         LeaderStateFollower,
		Active:              true,
		CurrentTerm:         cf.status.CurrentTerm,
		LeaderID:            cf.status.LeaderID,
		LastHeartbeat:       time.Now(),
		ConnectivityQuality: 100,
		VotedInCurrentTerm:  false,
		VotedFor:            "",
	}

	// Update node counts
	cf.updateNodeCounts()

	// Notify listeners
	cf.notifyListeners(ClusterEventMemberJoined, nodeInfo.ID, nodeInfo)

	// If we're the leader, send heartbeat to the new node
	if cf.selfStatus.LeaderState == LeaderStateLeader {
		go cf.sendHeartbeat(nodeInfo.ID)
	}
}

// handleNodeLeft handles a node leaving event
func (cf *ClusterFormation) handleNodeLeft(nodeInfo NodeInfo) {
	cf.statusMutex.Lock()
	defer cf.statusMutex.Unlock()

	// Remove node from members
	delete(cf.status.Members, nodeInfo.ID)

	// Update node counts
	cf.updateNodeCounts()

	// Notify listeners
	cf.notifyListeners(ClusterEventMemberLeft, nodeInfo.ID, nodeInfo)

	// If the leader left, start an election
	if nodeInfo.ID == cf.status.LeaderID {
		cf.status.LeaderID = ""
		cf.notifyListeners(ClusterEventLeaderChanged, "", nil)

		if cf.config.CanBeLeader && nodeInfo.Role == RoleManager {
			go cf.startElection()
		}
	}
}

// handleNodeUpdated handles a node update event
func (cf *ClusterFormation) handleNodeUpdated(nodeInfo NodeInfo) {
	cf.statusMutex.Lock()
	defer cf.statusMutex.Unlock()

	// Update node in members if it exists
	if member, exists := cf.status.Members[nodeInfo.ID]; exists {
		member.NodeInfo = nodeInfo
		cf.status.Members[nodeInfo.ID] = member

		// Update node counts
		cf.updateNodeCounts()
	}
}

// formCluster attempts to form or join a cluster
func (cf *ClusterFormation) formCluster() {
	// Context with timeout for cluster formation
	ctx, cancel := context.WithTimeout(cf.ctx, cf.config.FormationTimeout)
	defer cancel()

	log.Printf("Starting cluster formation process for cluster '%s'", cf.config.ClusterName)

	// Wait for discovery service to find some nodes
	formationStart := time.Now()
	nodeDiscoveryTimeout := 30 * time.Second
	nodeDiscoveryDeadline := formationStart.Add(nodeDiscoveryTimeout)

	for time.Now().Before(nodeDiscoveryDeadline) {
		// Check if context is done
		select {
		case <-ctx.Done():
			cf.setClusterState(ClusterStateDegraded)
			log.Printf("Cluster formation timed out")
			return
		default:
			// Continue
		}

		// Get current nodes from discovery
		nodes := cf.discovery.GetNodes()
		if len(nodes) > 1 { // 1+ other nodes besides ourselves
			break
		}

		// Wait a bit before checking again
		time.Sleep(1 * time.Second)
	}

	// Determine whether to create a new cluster or join an existing one
	existingLeader := cf.findExistingLeader()
	if existingLeader == "" {
		// No existing leader found, try to become the leader
		if cf.config.CanBeLeader {
			log.Printf("No existing leader found, attempting to form a new cluster")
			cf.becomeLeader()
		} else {
			// Can't be leader, wait for a leader to appear
			log.Printf("No existing leader found, waiting for a leader")
			cf.waitForLeader(ctx)
		}
	} else {
		// Found an existing leader, join their cluster
		log.Printf("Found existing leader %s, joining their cluster", existingLeader)
		cf.joinExistingCluster(existingLeader)
	}
}

// findExistingLeader attempts to find an existing leader in the discovered nodes
func (cf *ClusterFormation) findExistingLeader() string {
	// This is a simplified implementation
	// In a real system, this would query all discovered nodes to find the leader

	// For now, just return empty string to indicate no leader found
	return ""
}

// becomeLeader attempts to become the leader of a new cluster
func (cf *ClusterFormation) becomeLeader() {
	cf.statusMutex.Lock()
	defer cf.statusMutex.Unlock()

	// Become leader
	cf.selfStatus.LeaderState = LeaderStateLeader
	cf.selfStatus.CurrentTerm = 1
	cf.status.CurrentTerm = 1
	cf.status.LeaderID = cf.config.DiscoveryConfig.NodeID
	cf.status.FormedAt = time.Now()
	cf.setClusterState(ClusterStateFormed)

	// Stop election timer
	if cf.electionTimer != nil {
		cf.electionTimer.Stop()
	}

	// Start heartbeat timer
	cf.startHeartbeatTimer()

	// Notify listeners
	cf.notifyListeners(ClusterEventLeaderChanged, cf.config.DiscoveryConfig.NodeID, nil)

	// Send initial heartbeat to all nodes
	go cf.sendHeartbeatToAll()
}

// joinExistingCluster attempts to join an existing cluster
func (cf *ClusterFormation) joinExistingCluster(leaderID string) {
	cf.statusMutex.Lock()

	// Update state
	cf.setClusterState(ClusterStateJoining)
	cf.selfStatus.LeaderID = leaderID

	cf.statusMutex.Unlock()

	// Check if leader exists in discovered nodes
	_, exists := cf.discovery.GetNodeByID(leaderID)
	if !exists {
		log.Printf("Leader %s not found in discovered nodes", leaderID)
		return
	}

	// TODO: In a real implementation, this would send a join request to the leader

	// For now, just simulate joining by updating our state
	cf.statusMutex.Lock()
	defer cf.statusMutex.Unlock()

	cf.status.LeaderID = leaderID
	cf.status.FormedAt = time.Now()
	cf.setClusterState(ClusterStateFormed)
}

// waitForLeader waits for a leader to appear
func (cf *ClusterFormation) waitForLeader(ctx context.Context) {
	// Wait for a leader to appear or timeout
	for {
		select {
		case <-ctx.Done():
			cf.setClusterState(ClusterStateDegraded)
			log.Printf("Timed out waiting for leader")
			return
		default:
			// Check if leader has been found
			cf.statusMutex.RLock()
			leaderID := cf.status.LeaderID
			cf.statusMutex.RUnlock()

			if leaderID != "" {
				// Leader found
				return
			}

			// Wait a bit before checking again
			time.Sleep(1 * time.Second)
		}
	}
}

// startElection starts a leader election
func (cf *ClusterFormation) startElection() {
	cf.electionMutex.Lock()
	defer cf.electionMutex.Unlock()

	// Increment current term and become candidate
	cf.statusMutex.Lock()
	cf.status.CurrentTerm++
	cf.selfStatus.CurrentTerm = cf.status.CurrentTerm
	cf.selfStatus.LeaderState = LeaderStateCandidate
	cf.selfStatus.VotedInCurrentTerm = true
	cf.selfStatus.VotedFor = cf.config.DiscoveryConfig.NodeID
	cf.voteCount = 1 // Vote for self

	term := cf.status.CurrentTerm
	cf.statusMutex.Unlock()

	// Notify listeners that an election has started
	cf.notifyListeners(ClusterEventElectionStarted, cf.config.DiscoveryConfig.NodeID, term)

	// Reset election timer
	cf.resetElectionTimer()

	// Request votes from all other nodes
	cf.requestVotesFromAll()

	// Check if we have enough votes to become leader
	cf.checkVotes()
}

// requestVotesFromAll requests votes from all other nodes
func (cf *ClusterFormation) requestVotesFromAll() {
	cf.statusMutex.RLock()

	// Prepare vote request
	request := VoteRequest{
		Term:         cf.status.CurrentTerm,
		CandidateID:  cf.config.DiscoveryConfig.NodeID,
		LastLogIndex: 0,
		LastLogTerm:  0,
	}

	// Get list of nodes to request votes from
	var nodes []NodeInfo
	for id, member := range cf.status.Members {
		if id != cf.config.DiscoveryConfig.NodeID && member.Role == RoleManager {
			nodes = append(nodes, member.NodeInfo)
		}
	}

	cf.statusMutex.RUnlock()

	// Send vote requests to all nodes
	for _, node := range nodes {
		go cf.sendVoteRequest(node.ID, request)
	}
}

// sendVoteRequest sends a vote request to a node
func (cf *ClusterFormation) sendVoteRequest(nodeID string, request VoteRequest) {
	// TODO: In a real implementation, this would send an RPC to the node

	// For simulation, assume the vote is granted
	response := VoteResponse{
		Term:        request.Term,
		VoteGranted: true,
		NodeID:      nodeID,
	}

	// Process the response
	cf.handleVoteResponse(response)
}

// handleVoteResponse handles a vote response
func (cf *ClusterFormation) handleVoteResponse(response VoteResponse) {
	cf.electionMutex.Lock()
	defer cf.electionMutex.Unlock()

	cf.statusMutex.RLock()
	currentTerm := cf.status.CurrentTerm
	leaderState := cf.selfStatus.LeaderState
	cf.statusMutex.RUnlock()

	// Ignore if we're no longer a candidate or term doesn't match
	if leaderState != LeaderStateCandidate || response.Term != currentTerm {
		return
	}

	// If vote granted, increment vote count
	if response.VoteGranted {
		cf.voteCount++
		cf.checkVotes()
	} else if response.Term > currentTerm {
		// If response contains higher term, revert to follower
		cf.becomeFollower(response.Term)
	}
}

// checkVotes checks if we have enough votes to become leader
func (cf *ClusterFormation) checkVotes() {
	cf.statusMutex.RLock()

	// Check if we're still a candidate
	if cf.selfStatus.LeaderState != LeaderStateCandidate {
		cf.statusMutex.RUnlock()
		return
	}

	// Count total manager nodes
	managerCount := 0
	for _, member := range cf.status.Members {
		if member.Role == RoleManager {
			managerCount++
		}
	}

	cf.statusMutex.RUnlock()

	// Need majority of manager nodes to win
	neededVotes := (managerCount / 2) + 1

	if cf.voteCount >= neededVotes {
		// Won the election
		cf.becomeLeader()
	}
}

// becomeFollower transitions to follower state
func (cf *ClusterFormation) becomeFollower(term uint64) {
	cf.statusMutex.Lock()
	defer cf.statusMutex.Unlock()

	// Update state
	oldLeaderState := cf.selfStatus.LeaderState
	cf.selfStatus.LeaderState = LeaderStateFollower
	cf.selfStatus.CurrentTerm = term
	cf.status.CurrentTerm = term
	cf.selfStatus.VotedInCurrentTerm = false
	cf.selfStatus.VotedFor = ""

	// Stop heartbeat timer if we were the leader
	if oldLeaderState == LeaderStateLeader && cf.heartbeatTimer != nil {
		cf.heartbeatTimer.Stop()
	}

	// Reset election timer
	cf.resetElectionTimer()
}

// startHeartbeatTimer starts the heartbeat timer
func (cf *ClusterFormation) startHeartbeatTimer() {
	cf.heartbeatMutex.Lock()
	defer cf.heartbeatMutex.Unlock()

	// Stop existing timer if any
	if cf.heartbeatTimer != nil {
		cf.heartbeatTimer.Stop()
	}

	// Create new timer
	cf.heartbeatTimer = time.AfterFunc(
		time.Duration(cf.config.HeartbeatIntervalMs)*time.Millisecond,
		func() {
			cf.sendHeartbeatToAll()
			cf.startHeartbeatTimer() // Restart timer
		},
	)
}

// sendHeartbeatToAll sends heartbeats to all other nodes
func (cf *ClusterFormation) sendHeartbeatToAll() {
	cf.statusMutex.RLock()

	// Check if we're still the leader
	if cf.selfStatus.LeaderState != LeaderStateLeader {
		cf.statusMutex.RUnlock()
		return
	}

	// Get list of nodes to send heartbeats to
	var nodes []string
	for id := range cf.status.Members {
		if id != cf.config.DiscoveryConfig.NodeID {
			nodes = append(nodes, id)
		}
	}

	cf.statusMutex.RUnlock()

	// Send heartbeats to all nodes
	for _, nodeID := range nodes {
		go cf.sendHeartbeat(nodeID)
	}
}

// sendHeartbeat sends a heartbeat to a specific node
func (cf *ClusterFormation) sendHeartbeat(nodeID string) {
	cf.statusMutex.RLock()

	// Get current term and other values needed for the heartbeat
	term := cf.status.CurrentTerm
	leaderID := cf.config.DiscoveryConfig.NodeID

	cf.statusMutex.RUnlock()

	// TODO: In a real implementation, this would construct and send
	// a HeartbeatMessage via RPC to the node
	log.Printf("Sending heartbeat to node %s (term: %d, leader: %s)",
		nodeID, term, leaderID)
}

// resetElectionTimer resets the election timeout timer
func (cf *ClusterFormation) resetElectionTimer() {
	cf.electionMutex.Lock()
	defer cf.electionMutex.Unlock()

	// Stop existing timer if any
	if cf.electionTimer != nil {
		cf.electionTimer.Stop()
	}

	// Generate random timeout within the configured range
	min := cf.config.ElectionTimeoutRange[0]
	max := cf.config.ElectionTimeoutRange[1]
	timeout := min + cf.randGenerator.Int(max-min+1)

	// Create new timer
	cf.electionTimer = time.AfterFunc(
		time.Duration(timeout)*time.Millisecond,
		func() {
			// Start election if we're already a follower or candidate
			cf.statusMutex.RLock()
			state := cf.selfStatus.LeaderState
			canBeLeader := cf.config.CanBeLeader
			cf.statusMutex.RUnlock()

			if (state == LeaderStateFollower || state == LeaderStateCandidate) && canBeLeader {
				cf.startElection()
			}
		},
	)
}

// setClusterState sets the state of the cluster
func (cf *ClusterFormation) setClusterState(state ClusterState) {
	// Skip if state hasn't changed
	if cf.status.State == state {
		return
	}

	oldState := cf.status.State
	cf.status.State = state

	// Notify listeners of state change
	cf.notifyListeners(ClusterEventStateChanged, cf.config.DiscoveryConfig.NodeID, state)

	log.Printf("Cluster state changed from %s to %s", oldState, state)
}

// updateNodeCounts updates the node counts in the cluster status
func (cf *ClusterFormation) updateNodeCounts() {
	// Reset node lists
	cf.status.ManagerNodeIDs = make([]string, 0)
	cf.status.WorkerNodeIDs = make([]string, 0)

	// Count nodes by role
	cf.status.TotalNodes = len(cf.status.Members)
	cf.status.HealthyNodes = 0

	for id, member := range cf.status.Members {
		if member.Available {
			cf.status.HealthyNodes++
		}

		if member.Role == RoleManager {
			cf.status.ManagerNodeIDs = append(cf.status.ManagerNodeIDs, id)
		} else {
			cf.status.WorkerNodeIDs = append(cf.status.WorkerNodeIDs, id)
		}
	}

	// Sort IDs for consistent order
	sort.Strings(cf.status.ManagerNodeIDs)
	sort.Strings(cf.status.WorkerNodeIDs)
}

// notifyListeners notifies all listeners of an event
func (cf *ClusterFormation) notifyListeners(eventType ClusterEventType, nodeID string, data interface{}) {
	cf.listenersMutex.RLock()
	listeners := make([]ClusterEventListener, len(cf.listeners))
	copy(listeners, cf.listeners)
	cf.listenersMutex.RUnlock()

	event := ClusterEvent{
		Type:      eventType,
		NodeID:    nodeID,
		Data:      data,
		Timestamp: time.Now(),
	}

	for _, listener := range listeners {
		go listener(event)
	}
}

// generateClusterID generates a unique ID for the cluster
func generateClusterID(clusterName string) string {
	// In a real implementation, this would be a more complex algorithm
	// For simplicity, just hash the cluster name with a timestamp
	now := time.Now().UnixNano()
	return fmt.Sprintf("%s-%d", clusterName, now)
}
