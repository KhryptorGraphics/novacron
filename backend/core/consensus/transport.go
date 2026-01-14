package consensus

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"
)

// HTTPTransport implements Transport interface using HTTP
type HTTPTransport struct {
	nodeAddresses map[string]string
	client        *http.Client
	server        *http.Server
	raftNode      *RaftNode
	mu            sync.RWMutex
}

// NewHTTPTransport creates a new HTTP-based transport
func NewHTTPTransport(nodeAddresses map[string]string, bindAddr string) *HTTPTransport {
	transport := &HTTPTransport{
		nodeAddresses: nodeAddresses,
		client: &http.Client{
			Timeout: 200 * time.Millisecond,
		},
	}
	
	// Create HTTP server
	mux := http.NewServeMux()
	mux.HandleFunc("/raft/request_vote", transport.handleRequestVote)
	mux.HandleFunc("/raft/append_entries", transport.handleAppendEntries)
	mux.HandleFunc("/raft/install_snapshot", transport.handleInstallSnapshot)
	
	transport.server = &http.Server{
		Addr:    bindAddr,
		Handler: mux,
	}
	
	return transport
}

// SetRaftNode sets the Raft node for this transport
func (t *HTTPTransport) SetRaftNode(node *RaftNode) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.raftNode = node
}

// Start starts the HTTP server
func (t *HTTPTransport) Start() error {
	go func() {
		if err := t.server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			panic(fmt.Sprintf("HTTP transport server failed: %v", err))
		}
	}()
	return nil
}

// Stop stops the HTTP server
func (t *HTTPTransport) Stop() error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	return t.server.Shutdown(ctx)
}

// SendRequestVote sends a RequestVote RPC to a peer
func (t *HTTPTransport) SendRequestVote(ctx context.Context, nodeID string, req *RequestVoteArgs) (*RequestVoteReply, error) {
	t.mu.RLock()
	addr, exists := t.nodeAddresses[nodeID]
	t.mu.RUnlock()
	
	if !exists {
		return nil, fmt.Errorf("unknown node: %s", nodeID)
	}
	
	url := fmt.Sprintf("http://%s/raft/request_vote", addr)
	reply := &RequestVoteReply{}
	err := t.sendRPC(ctx, url, req, reply)
	return reply, err
}

// SendAppendEntries sends an AppendEntries RPC to a peer
func (t *HTTPTransport) SendAppendEntries(ctx context.Context, nodeID string, req *AppendEntriesArgs) (*AppendEntriesReply, error) {
	t.mu.RLock()
	addr, exists := t.nodeAddresses[nodeID]
	t.mu.RUnlock()
	
	if !exists {
		return nil, fmt.Errorf("unknown node: %s", nodeID)
	}
	
	url := fmt.Sprintf("http://%s/raft/append_entries", addr)
	reply := &AppendEntriesReply{}
	err := t.sendRPC(ctx, url, req, reply)
	return reply, err
}

// SendSnapshot sends an InstallSnapshot RPC to a peer
func (t *HTTPTransport) SendSnapshot(ctx context.Context, nodeID string, req *InstallSnapshotArgs) (*InstallSnapshotReply, error) {
	t.mu.RLock()
	addr, exists := t.nodeAddresses[nodeID]
	t.mu.RUnlock()
	
	if !exists {
		return nil, fmt.Errorf("unknown node: %s", nodeID)
	}
	
	url := fmt.Sprintf("http://%s/raft/install_snapshot", addr)
	reply := &InstallSnapshotReply{}
	err := t.sendRPC(ctx, url, req, reply)
	return reply, err
}

// Generic RPC sender
func (t *HTTPTransport) sendRPC(ctx context.Context, url string, request interface{}, response interface{}) error {
	// Marshal request
	body, err := json.Marshal(request)
	if err != nil {
		return fmt.Errorf("failed to marshal request: %v", err)
	}
	
	// Create HTTP request
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(body))
	if err != nil {
		return fmt.Errorf("failed to create request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	
	// Send request
	resp, err := t.client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to send request: %v", err)
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("request failed with status: %d", resp.StatusCode)
	}
	
	// Read response
	responseBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("failed to read response: %v", err)
	}
	
	// Unmarshal response
	if err := json.Unmarshal(responseBody, response); err != nil {
		return fmt.Errorf("failed to unmarshal response: %v", err)
	}
	
	return nil
}

// HTTP handlers

func (t *HTTPTransport) handleRequestVote(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	
	// Read request body
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "Failed to read body", http.StatusBadRequest)
		return
	}
	
	// Parse request
	var req RequestVoteArgs
	if err := json.Unmarshal(body, &req); err != nil {
		http.Error(w, "Failed to parse request", http.StatusBadRequest)
		return
	}
	
	// Process request
	t.mu.RLock()
	raftNode := t.raftNode
	t.mu.RUnlock()
	
	if raftNode == nil {
		http.Error(w, "Raft node not initialized", http.StatusServiceUnavailable)
		return
	}
	
	reply := raftNode.HandleRequestVote(&req)
	
	// Send response
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(reply); err != nil {
		http.Error(w, "Failed to encode response", http.StatusInternalServerError)
	}
}

func (t *HTTPTransport) handleAppendEntries(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	
	// Read request body
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "Failed to read body", http.StatusBadRequest)
		return
	}
	
	// Parse request
	var req AppendEntriesArgs
	if err := json.Unmarshal(body, &req); err != nil {
		http.Error(w, "Failed to parse request", http.StatusBadRequest)
		return
	}
	
	// Process request
	t.mu.RLock()
	raftNode := t.raftNode
	t.mu.RUnlock()
	
	if raftNode == nil {
		http.Error(w, "Raft node not initialized", http.StatusServiceUnavailable)
		return
	}
	
	reply := raftNode.HandleAppendEntries(&req)
	
	// Send response
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(reply); err != nil {
		http.Error(w, "Failed to encode response", http.StatusInternalServerError)
	}
}

func (t *HTTPTransport) handleInstallSnapshot(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	
	// Read request body
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "Failed to read body", http.StatusBadRequest)
		return
	}
	
	// Parse request
	var req InstallSnapshotArgs
	if err := json.Unmarshal(body, &req); err != nil {
		http.Error(w, "Failed to parse request", http.StatusBadRequest)
		return
	}
	
	// Process request
	t.mu.RLock()
	raftNode := t.raftNode
	t.mu.RUnlock()
	
	if raftNode == nil {
		http.Error(w, "Raft node not initialized", http.StatusServiceUnavailable)
		return
	}
	
	reply := raftNode.HandleInstallSnapshot(&req)
	
	// Send response
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(reply); err != nil {
		http.Error(w, "Failed to encode response", http.StatusInternalServerError)
	}
}

// HandleInstallSnapshot handles InstallSnapshot RPC
func (rn *RaftNode) HandleInstallSnapshot(args *InstallSnapshotArgs) *InstallSnapshotReply {
	rn.mu.Lock()
	defer rn.mu.Unlock()
	
	reply := &InstallSnapshotReply{
		Term: rn.currentTerm,
	}
	
	// Reply immediately if term < currentTerm
	if args.Term < rn.currentTerm {
		return reply
	}
	
	// Convert to follower if newer term
	if args.Term > rn.currentTerm {
		rn.currentTerm = args.Term
		rn.votedFor = ""
		rn.state = Follower
		rn.leaderID = args.LeaderID
	}
	
	rn.resetElectionTimer()
	reply.Term = rn.currentTerm
	
	// Save snapshot data (in a real implementation, this would be more sophisticated)
	// For now, we'll just truncate the log and update indices
	
	// If existing log entry has same index and term as snapshot's last included entry,
	// retain log entries following it
	if args.LastIncludedIndex <= int64(len(rn.log)) {
		if rn.log[args.LastIncludedIndex-1].Term == args.LastIncludedTerm {
			// Keep entries after the snapshot
			rn.log = rn.log[args.LastIncludedIndex:]
			// Adjust indices in the remaining log entries
			for i := range rn.log {
				rn.log[i].Index = int64(i) + args.LastIncludedIndex + 1
			}
		} else {
			// Discard entire log
			rn.log = make([]LogEntry, 0)
		}
	} else {
		// Discard entire log
		rn.log = make([]LogEntry, 0)
	}
	
	// Update state
	rn.commitIndex = args.LastIncludedIndex
	rn.lastApplied = args.LastIncludedIndex
	
	return reply
}

// InMemoryTransport implements Transport for testing
type InMemoryTransport struct {
	nodes map[string]*InMemoryTransport
	id    string
	raftNode *RaftNode
	mu    sync.RWMutex
	
	// Simulate network conditions
	dropRate   float64 // Probability of dropping messages
	delayRange [2]time.Duration // Min and max delay
}

// NewInMemoryTransport creates a new in-memory transport for testing
func NewInMemoryTransport(id string) *InMemoryTransport {
	return &InMemoryTransport{
		nodes: make(map[string]*InMemoryTransport),
		id:    id,
	}
}

// Connect connects this transport to another
func (t *InMemoryTransport) Connect(other *InMemoryTransport) {
	t.mu.Lock()
	defer t.mu.Unlock()
	
	t.nodes[other.id] = other
	
	other.mu.Lock()
	defer other.mu.Unlock()
	other.nodes[t.id] = t
}

// SetRaftNode sets the Raft node
func (t *InMemoryTransport) SetRaftNode(node *RaftNode) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.raftNode = node
}

// SendRequestVote sends a RequestVote RPC
func (t *InMemoryTransport) SendRequestVote(ctx context.Context, nodeID string, req *RequestVoteArgs) (*RequestVoteReply, error) {
	t.mu.RLock()
	target, exists := t.nodes[nodeID]
	t.mu.RUnlock()
	
	if !exists {
		return nil, fmt.Errorf("unknown node: %s", nodeID)
	}
	
	// Simulate network delay
	if t.delayRange[1] > 0 {
		delay := t.delayRange[0] + time.Duration(float64(t.delayRange[1]-t.delayRange[0])*0.5)
		select {
		case <-time.After(delay):
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	
	target.mu.RLock()
	raftNode := target.raftNode
	target.mu.RUnlock()
	
	if raftNode == nil {
		return nil, fmt.Errorf("target node not initialized")
	}
	
	return raftNode.HandleRequestVote(req), nil
}

// SendAppendEntries sends an AppendEntries RPC
func (t *InMemoryTransport) SendAppendEntries(ctx context.Context, nodeID string, req *AppendEntriesArgs) (*AppendEntriesReply, error) {
	t.mu.RLock()
	target, exists := t.nodes[nodeID]
	t.mu.RUnlock()
	
	if !exists {
		return nil, fmt.Errorf("unknown node: %s", nodeID)
	}
	
	// Simulate network delay
	if t.delayRange[1] > 0 {
		delay := t.delayRange[0] + time.Duration(float64(t.delayRange[1]-t.delayRange[0])*0.5)
		select {
		case <-time.After(delay):
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	
	target.mu.RLock()
	raftNode := target.raftNode
	target.mu.RUnlock()
	
	if raftNode == nil {
		return nil, fmt.Errorf("target node not initialized")
	}
	
	return raftNode.HandleAppendEntries(req), nil
}

// SendSnapshot sends an InstallSnapshot RPC
func (t *InMemoryTransport) SendSnapshot(ctx context.Context, nodeID string, req *InstallSnapshotArgs) (*InstallSnapshotReply, error) {
	t.mu.RLock()
	target, exists := t.nodes[nodeID]
	t.mu.RUnlock()
	
	if !exists {
		return nil, fmt.Errorf("unknown node: %s", nodeID)
	}
	
	// Simulate network delay
	if t.delayRange[1] > 0 {
		delay := t.delayRange[0] + time.Duration(float64(t.delayRange[1]-t.delayRange[0])*0.5)
		select {
		case <-time.After(delay):
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	
	target.mu.RLock()
	raftNode := target.raftNode
	target.mu.RUnlock()
	
	if raftNode == nil {
		return nil, fmt.Errorf("target node not initialized")
	}
	
	return raftNode.HandleInstallSnapshot(req), nil
}

// SetNetworkConditions sets network simulation parameters
func (t *InMemoryTransport) SetNetworkConditions(dropRate float64, minDelay, maxDelay time.Duration) {
	t.mu.Lock()
	defer t.mu.Unlock()
	
	t.dropRate = dropRate
	t.delayRange[0] = minDelay
	t.delayRange[1] = maxDelay
}