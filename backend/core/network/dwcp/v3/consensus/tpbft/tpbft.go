// Package tpbft implements Trust-based PBFT consensus
package tpbft

import (
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
	"time"
)

// Message types for T-PBFT protocol
const (
	PrePrepare = "PRE_PREPARE"
	Prepare    = "PREPARE"
	Commit     = "COMMIT"
	ViewChange = "VIEW_CHANGE"
	NewView    = "NEW_VIEW"
)

// Request represents a client request to be processed
type Request struct {
	ID        string    `json:"id"`
	Timestamp time.Time `json:"timestamp"`
	Data      []byte    `json:"data"`
	ClientID  string    `json:"client_id"`
}

// Message represents a PBFT protocol message
type Message struct {
	Type      string    `json:"type"`
	View      int       `json:"view"`
	Sequence  int       `json:"sequence"`
	NodeID    string    `json:"node_id"`
	Digest    string    `json:"digest"`
	Request   *Request  `json:"request,omitempty"`
	Timestamp time.Time `json:"timestamp"`
	Signature []byte    `json:"signature,omitempty"`
}

// TPBFT implements Trust-based PBFT consensus with EigenTrust reputation
type TPBFT struct {
	mu           sync.RWMutex
	nodeID       string
	view         int
	sequence     int
	trustMgr     *EigenTrust
	committee    []string          // Trusted committee members
	committeeSize int              // Target committee size
	messages     map[string][]*Message // Message log: digest -> messages
	prepared     map[string]bool   // Prepared certificates
	committed    map[string]bool   // Committed certificates
	executed     map[string]bool   // Executed requests
	results      map[string][]byte // Execution results

	// Performance metrics
	consensusLatency time.Duration
	throughput       float64
	totalRequests    int

	// Thresholds (for f Byzantine nodes, need 2f+1 messages)
	f               int // Max Byzantine nodes
	prepareThreshold int // 2f
	commitThreshold  int // 2f+1
}

// NewTPBFT creates a new T-PBFT instance
func NewTPBFT(nodeID string, trustMgr *EigenTrust) *TPBFT {
	return &TPBFT{
		nodeID:       nodeID,
		view:         0,
		sequence:     0,
		trustMgr:     trustMgr,
		committeeSize: 10, // Default committee size
		messages:     make(map[string][]*Message),
		prepared:     make(map[string]bool),
		committed:    make(map[string]bool),
		executed:     make(map[string]bool),
		results:      make(map[string][]byte),
		f:            3, // Allow up to 3 Byzantine nodes
	}
}

// SelectCommittee selects committee members based on trust scores
// Higher trust = higher probability of selection (26% throughput increase over random)
func (t *TPBFT) SelectCommittee() []string {
	t.mu.Lock()
	defer t.mu.Unlock()

	// Get top N most trusted nodes
	topNodes := t.trustMgr.GetTopNodes(t.committeeSize)

	// Update Byzantine tolerance based on committee size
	t.committee = topNodes
	t.f = (len(topNodes) - 1) / 3
	t.prepareThreshold = 2 * t.f
	t.commitThreshold = 2*t.f + 1

	return topNodes
}

// Consensus executes the T-PBFT consensus protocol for a request
func (t *TPBFT) Consensus(request Request) error {
	startTime := time.Now()

	// 1. Calculate request digest
	digest, err := t.calculateDigest(request)
	if err != nil {
		return fmt.Errorf("failed to calculate digest: %w", err)
	}

	// 2. Pre-prepare phase (leader only)
	if t.isLeader() {
		if err := t.prePrepare(request, digest); err != nil {
			return fmt.Errorf("pre-prepare failed: %w", err)
		}
	}

	// 3. Prepare phase (all replicas)
	if err := t.prepare(digest); err != nil {
		return fmt.Errorf("prepare failed: %w", err)
	}

	// 4. Commit phase (all replicas)
	if err := t.commit(digest); err != nil {
		return fmt.Errorf("commit failed: %w", err)
	}

	// 5. Execute request
	if err := t.execute(digest, request); err != nil {
		return fmt.Errorf("execute failed: %w", err)
	}

	// Update metrics
	t.consensusLatency = time.Since(startTime)
	t.totalRequests++
	t.throughput = float64(t.totalRequests) / time.Since(startTime).Seconds()

	return nil
}

// prePrepare initiates the consensus protocol (leader only)
func (t *TPBFT) prePrepare(request Request, digest string) error {
	t.mu.Lock()
	t.sequence++
	msg := &Message{
		Type:      PrePrepare,
		View:      t.view,
		Sequence:  t.sequence,
		NodeID:    t.nodeID,
		Digest:    digest,
		Request:   &request,
		Timestamp: time.Now(),
	}
	t.mu.Unlock()

	// Broadcast to committee
	return t.broadcast(msg)
}

// prepare validates pre-prepare and enters prepare phase
func (t *TPBFT) prepare(digest string) error {
	// Verify pre-prepare from trusted leader
	if !t.verifyPrePrepare(digest) {
		return errors.New("invalid pre-prepare message")
	}

	// Send prepare message
	msg := &Message{
		Type:      Prepare,
		View:      t.view,
		Sequence:  t.sequence,
		NodeID:    t.nodeID,
		Digest:    digest,
		Timestamp: time.Now(),
	}

	if err := t.broadcast(msg); err != nil {
		return err
	}

	// Wait for 2f prepare messages from trusted nodes
	return t.waitForPrepared(digest)
}

// commit enters the commit phase after prepare certificate
func (t *TPBFT) commit(digest string) error {
	// Verify prepare certificate
	if !t.isPrepared(digest) {
		return errors.New("prepare certificate not achieved")
	}

	// Send commit message
	msg := &Message{
		Type:      Commit,
		View:      t.view,
		Sequence:  t.sequence,
		NodeID:    t.nodeID,
		Digest:    digest,
		Timestamp: time.Now(),
	}

	if err := t.broadcast(msg); err != nil {
		return err
	}

	// Wait for 2f+1 commit messages from trusted nodes
	return t.waitForCommitted(digest)
}

// execute executes the request after commit certificate
func (t *TPBFT) execute(digest string, request Request) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	// Check if already executed
	if t.executed[digest] {
		return nil
	}

	// Verify commit certificate
	if !t.committed[digest] {
		return errors.New("commit certificate not achieved")
	}

	// Execute request (placeholder - actual execution depends on application)
	result := t.executeRequest(request)
	t.results[digest] = result
	t.executed[digest] = true

	// Record successful interaction with all committee members
	for _, nodeID := range t.committee {
		if nodeID != t.nodeID {
			t.trustMgr.RecordSuccessfulInteraction(t.nodeID, nodeID)
		}
	}

	return nil
}

// executeRequest processes the actual request (application-specific)
func (t *TPBFT) executeRequest(request Request) []byte {
	// Placeholder - implement actual state machine execution
	return []byte(fmt.Sprintf("executed: %s", request.ID))
}

// verifyPrePrepare verifies the pre-prepare message from leader
func (t *TPBFT) verifyPrePrepare(digest string) bool {
	t.mu.RLock()
	defer t.mu.RUnlock()

	messages := t.messages[digest]
	for _, msg := range messages {
		if msg.Type == PrePrepare {
			// Verify leader is trusted
			leaderTrust := t.trustMgr.GetTrustScore(msg.NodeID)
			return leaderTrust > 0.5 // Require high trust for leader
		}
	}
	return false
}

// waitForPrepared waits for prepare certificate (2f messages)
func (t *TPBFT) waitForPrepared(digest string) error {
	timeout := time.After(5 * time.Second)
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-timeout:
			return errors.New("prepare timeout")
		case <-ticker.C:
			if t.checkPrepared(digest) {
				return nil
			}
		}
	}
}

// checkPrepared checks if prepare certificate is achieved
func (t *TPBFT) checkPrepared(digest string) bool {
	t.mu.Lock()
	defer t.mu.Unlock()

	messages := t.messages[digest]
	prepareCount := 0
	trustedCount := 0

	for _, msg := range messages {
		if msg.Type == Prepare {
			prepareCount++
			// Weight by trust score (trust-based voting)
			trust := t.trustMgr.GetTrustScore(msg.NodeID)
			if trust > 0.6 {
				trustedCount++
			}
		}
	}

	// Need 2f trusted prepare messages
	if trustedCount >= t.prepareThreshold {
		t.prepared[digest] = true
		return true
	}
	return false
}

// waitForCommitted waits for commit certificate (2f+1 messages)
func (t *TPBFT) waitForCommitted(digest string) error {
	timeout := time.After(5 * time.Second)
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-timeout:
			return errors.New("commit timeout")
		case <-ticker.C:
			if t.checkCommitted(digest) {
				return nil
			}
		}
	}
}

// checkCommitted checks if commit certificate is achieved
func (t *TPBFT) checkCommitted(digest string) bool {
	t.mu.Lock()
	defer t.mu.Unlock()

	messages := t.messages[digest]
	commitCount := 0
	trustedCount := 0

	for _, msg := range messages {
		if msg.Type == Commit {
			commitCount++
			// Weight by trust score
			trust := t.trustMgr.GetTrustScore(msg.NodeID)
			if trust > 0.6 {
				trustedCount++
			}
		}
	}

	// Need 2f+1 trusted commit messages
	if trustedCount >= t.commitThreshold {
		t.committed[digest] = true
		return true
	}
	return false
}

// isPrepared checks if request has prepare certificate
func (t *TPBFT) isPrepared(digest string) bool {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.prepared[digest]
}

// isLeader checks if this node is the current leader
func (t *TPBFT) isLeader() bool {
	// Simple leader selection: primary = view mod committee_size
	if len(t.committee) == 0 {
		return false
	}
	primaryIndex := t.view % len(t.committee)
	return t.committee[primaryIndex] == t.nodeID
}

// broadcast sends a message to all committee members
func (t *TPBFT) broadcast(msg *Message) error {
	// Add to local message log
	t.mu.Lock()
	t.messages[msg.Digest] = append(t.messages[msg.Digest], msg)
	t.mu.Unlock()

	// Placeholder - implement actual network broadcast
	return nil
}

// HandleMessage processes incoming PBFT messages
func (t *TPBFT) HandleMessage(msg *Message) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	// Add to message log
	t.messages[msg.Digest] = append(t.messages[msg.Digest], msg)

	// Update trust based on message validity
	if t.validateMessage(msg) {
		t.trustMgr.RecordSuccessfulInteraction(t.nodeID, msg.NodeID)
	} else {
		t.trustMgr.RecordFailedInteraction(t.nodeID, msg.NodeID)
	}

	return nil
}

// validateMessage performs basic message validation
func (t *TPBFT) validateMessage(msg *Message) bool {
	// Check view number
	if msg.View != t.view {
		return false
	}

	// Check sender is in committee
	for _, node := range t.committee {
		if node == msg.NodeID {
			return true
		}
	}
	return false
}

// calculateDigest computes SHA-256 digest of request
func (t *TPBFT) calculateDigest(request Request) (string, error) {
	data, err := json.Marshal(request)
	if err != nil {
		return "", err
	}
	hash := sha256.Sum256(data)
	return fmt.Sprintf("%x", hash), nil
}

// GetMetrics returns performance metrics
func (t *TPBFT) GetMetrics() map[string]interface{} {
	t.mu.RLock()
	defer t.mu.RUnlock()

	return map[string]interface{}{
		"view":              t.view,
		"sequence":          t.sequence,
		"committee_size":    len(t.committee),
		"f":                 t.f,
		"total_requests":    t.totalRequests,
		"throughput":        t.throughput,
		"consensus_latency": t.consensusLatency.Milliseconds(),
		"prepared_count":    len(t.prepared),
		"committed_count":   len(t.committed),
		"executed_count":    len(t.executed),
	}
}
