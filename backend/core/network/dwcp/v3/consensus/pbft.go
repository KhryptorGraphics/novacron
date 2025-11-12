package consensus

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"go.uber.org/zap"
)

// PBFT implements Practical Byzantine Fault Tolerance consensus
// Tolerates up to f = (n-1)/3 Byzantine (malicious) nodes
// Optimal for internet-scale deployments with untrusted nodes
type PBFT struct {
	mu sync.RWMutex

	nodeID       string
	replicaCount int
	f            int // Byzantine tolerance: f = (n-1)/3

	// View management
	view         int64
	primaryID    string
	isPrimary    bool
	viewChangeID int64

	// Message logs
	prePrepareLog map[string]*PrePrepareMessage
	prepareLog    map[string]map[string]*PrepareMessage
	commitLog     map[string]map[string]*CommitMessage

	// Request tracking
	requestQueue  []*ClientRequest
	executedReqs  map[string]bool
	checkpointSeq int64

	// Checkpoint state
	checkpoints      map[int64]map[string]*CheckpointMessage
	stableCheckpoint int64

	// Network
	peers     map[string]*Replica
	transport Transport

	// State machine
	stateMachine StateMachine

	logger *zap.Logger
	ctx    context.Context
	cancel context.CancelFunc
}

// PrePrepareMessage is sent by primary to initiate consensus
type PrePrepareMessage struct {
	View      int64          `json:"view"`
	Sequence  int64          `json:"sequence"`
	Digest    string         `json:"digest"`
	Request   *ClientRequest `json:"request"`
	Timestamp time.Time      `json:"timestamp"`
}

// PrepareMessage is sent by replicas after receiving pre-prepare
type PrepareMessage struct {
	View      int64     `json:"view"`
	Sequence  int64     `json:"sequence"`
	Digest    string    `json:"digest"`
	ReplicaID string    `json:"replica_id"`
	Timestamp time.Time `json:"timestamp"`
}

// CommitMessage is sent after receiving 2f prepare messages
type CommitMessage struct {
	View      int64     `json:"view"`
	Sequence  int64     `json:"sequence"`
	Digest    string    `json:"digest"`
	ReplicaID string    `json:"replica_id"`
	Timestamp time.Time `json:"timestamp"`
}

// ClientRequest represents a client operation request
type ClientRequest struct {
	ClientID  string          `json:"client_id"`
	Timestamp time.Time       `json:"timestamp"`
	Operation json.RawMessage `json:"operation"`
	Sequence  int64           `json:"sequence"`
}

// CheckpointMessage is sent periodically for garbage collection
type CheckpointMessage struct {
	Sequence  int64     `json:"sequence"`
	Digest    string    `json:"digest"`
	ReplicaID string    `json:"replica_id"`
	Timestamp time.Time `json:"timestamp"`
}

// ViewChangeMessage is sent to trigger view change
type ViewChangeMessage struct {
	NewView   int64                        `json:"new_view"`
	ReplicaID string                       `json:"replica_id"`
	Prepared  map[int64]*PreparedCertificate `json:"prepared"`
	Timestamp time.Time                    `json:"timestamp"`
}

// PreparedCertificate proves a request was prepared
type PreparedCertificate struct {
	PrePrepare *PrePrepareMessage          `json:"pre_prepare"`
	Prepares   map[string]*PrepareMessage  `json:"prepares"`
}

// Replica represents a PBFT replica node
type Replica struct {
	ID       string `json:"id"`
	Endpoint string `json:"endpoint"`
	PublicKey []byte `json:"public_key,omitempty"`
}

// Transport interface for network communication
type Transport interface {
	Send(replicaID string, message interface{}) error
	Broadcast(message interface{}) error
	Receive() (interface{}, error)
}

// StateMachine interface for state updates
type StateMachine interface {
	Apply(operation json.RawMessage) (json.RawMessage, error)
	GetState() (json.RawMessage, error)
	Checkpoint(sequence int64) (string, error)
}

// NewPBFT creates a new PBFT consensus instance
func NewPBFT(nodeID string, replicaCount int, transport Transport, stateMachine StateMachine, logger *zap.Logger) (*PBFT, error) {
	// Validate replica count (must be 3f+1 for f Byzantine tolerance)
	if replicaCount < 4 {
		return nil, fmt.Errorf("minimum 4 replicas required for Byzantine tolerance")
	}

	f := (replicaCount - 1) / 3
	if replicaCount != 3*f+1 {
		logger.Warn("Replica count not optimal",
			zap.Int("count", replicaCount),
			zap.Int("optimal", 3*f+1),
			zap.Int("f", f))
	}

	ctx, cancel := context.WithCancel(context.Background())

	pbft := &PBFT{
		nodeID:        nodeID,
		replicaCount:  replicaCount,
		f:             f,
		view:          0,
		viewChangeID:  0,
		prePrepareLog: make(map[string]*PrePrepareMessage),
		prepareLog:    make(map[string]map[string]*PrepareMessage),
		commitLog:     make(map[string]map[string]*CommitMessage),
		requestQueue:  make([]*ClientRequest, 0),
		executedReqs:  make(map[string]bool),
		checkpoints:   make(map[int64]map[string]*CheckpointMessage),
		checkpointSeq: 0,
		stableCheckpoint: 0,
		peers:         make(map[string]*Replica),
		transport:     transport,
		stateMachine:  stateMachine,
		logger:        logger,
		ctx:           ctx,
		cancel:        cancel,
	}

	// Determine if this node is the primary
	// Simple strategy: primary is first node in sorted replica list
	pbft.updatePrimary()

	return pbft, nil
}

// Start begins PBFT consensus processing
func (p *PBFT) Start() error {
	p.logger.Info("Starting PBFT consensus",
		zap.String("node_id", p.nodeID),
		zap.Int("replicas", p.replicaCount),
		zap.Int("byzantine_tolerance", p.f),
		zap.Bool("is_primary", p.isPrimary))

	go p.processMessages()
	go p.checkpointLoop()

	return nil
}

// Stop halts PBFT consensus
func (p *PBFT) Stop() error {
	p.logger.Info("Stopping PBFT consensus", zap.String("node_id", p.nodeID))
	p.cancel()
	return nil
}

// Consensus executes PBFT 3-phase consensus on a proposal
func (p *PBFT) Consensus(ctx context.Context, value interface{}) error {
	// Convert value to operation
	operation, err := json.Marshal(value)
	if err != nil {
		return fmt.Errorf("failed to marshal operation: %w", err)
	}

	request := &ClientRequest{
		ClientID:  p.nodeID,
		Timestamp: time.Now(),
		Operation: operation,
		Sequence:  p.getNextSequence(),
	}

	// If we're the primary, initiate pre-prepare
	if p.isPrimary {
		return p.handleClientRequest(request)
	}

	// Otherwise, forward to primary
	return p.forwardToPrimary(request)
}

// handleClientRequest processes a client request (primary only)
func (p *PBFT) handleClientRequest(req *ClientRequest) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	// Check if already executed
	reqID := p.requestID(req)
	if p.executedReqs[reqID] {
		p.logger.Debug("Request already executed", zap.String("request_id", reqID))
		return nil
	}

	// Create pre-prepare message
	digest := p.computeDigest(req)
	sequence := req.Sequence

	prePrepare := &PrePrepareMessage{
		View:      p.view,
		Sequence:  sequence,
		Digest:    digest,
		Request:   req,
		Timestamp: time.Now(),
	}

	// Store in log
	key := p.logKey(p.view, sequence)
	p.prePrepareLog[key] = prePrepare

	// Broadcast pre-prepare to all replicas
	if err := p.transport.Broadcast(prePrepare); err != nil {
		return fmt.Errorf("failed to broadcast pre-prepare: %w", err)
	}

	p.logger.Info("Sent pre-prepare",
		zap.Int64("view", p.view),
		zap.Int64("sequence", sequence),
		zap.String("digest", digest))

	return nil
}

// handlePrePrepare processes pre-prepare message (backup replicas)
func (p *PBFT) handlePrePrepare(msg *PrePrepareMessage) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	// Validate view
	if msg.View != p.view {
		p.logger.Warn("Pre-prepare view mismatch",
			zap.Int64("expected", p.view),
			zap.Int64("received", msg.View))
		return fmt.Errorf("view mismatch")
	}

	// Verify digest
	if p.computeDigest(msg.Request) != msg.Digest {
		return fmt.Errorf("digest mismatch")
	}

	// Store pre-prepare
	key := p.logKey(msg.View, msg.Sequence)
	p.prePrepareLog[key] = msg

	// Send prepare message
	prepare := &PrepareMessage{
		View:      msg.View,
		Sequence:  msg.Sequence,
		Digest:    msg.Digest,
		ReplicaID: p.nodeID,
		Timestamp: time.Now(),
	}

	if err := p.transport.Broadcast(prepare); err != nil {
		return fmt.Errorf("failed to broadcast prepare: %w", err)
	}

	p.logger.Info("Sent prepare",
		zap.Int64("view", msg.View),
		zap.Int64("sequence", msg.Sequence))

	return nil
}

// handlePrepare processes prepare message
func (p *PBFT) handlePrepare(msg *PrepareMessage) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	key := p.logKey(msg.View, msg.Sequence)

	// Initialize prepare log for this request if needed
	if p.prepareLog[key] == nil {
		p.prepareLog[key] = make(map[string]*PrepareMessage)
	}

	// Store prepare message
	p.prepareLog[key][msg.ReplicaID] = msg

	// Check if we have 2f prepares (quorum)
	if len(p.prepareLog[key]) >= 2*p.f {
		// Get pre-prepare for this request
		_, exists := p.prePrepareLog[key]
		if !exists {
			return fmt.Errorf("pre-prepare not found for sequence %d", msg.Sequence)
		}

		// Send commit message
		commit := &CommitMessage{
			View:      msg.View,
			Sequence:  msg.Sequence,
			Digest:    msg.Digest,
			ReplicaID: p.nodeID,
			Timestamp: time.Now(),
		}

		if err := p.transport.Broadcast(commit); err != nil {
			return fmt.Errorf("failed to broadcast commit: %w", err)
		}

		p.logger.Info("Sent commit (prepared)",
			zap.Int64("view", msg.View),
			zap.Int64("sequence", msg.Sequence),
			zap.Int("prepares", len(p.prepareLog[key])))

		// Store our own commit
		return p.handleCommit(commit)
	}

	return nil
}

// handleCommit processes commit message
func (p *PBFT) handleCommit(msg *CommitMessage) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	key := p.logKey(msg.View, msg.Sequence)

	// Initialize commit log for this request if needed
	if p.commitLog[key] == nil {
		p.commitLog[key] = make(map[string]*CommitMessage)
	}

	// Store commit message
	p.commitLog[key][msg.ReplicaID] = msg

	// Check if we have 2f+1 commits (quorum for execution)
	if len(p.commitLog[key]) >= 2*p.f+1 {
		// Get pre-prepare for execution
		prePrepare, exists := p.prePrepareLog[key]
		if !exists {
			return fmt.Errorf("pre-prepare not found for sequence %d", msg.Sequence)
		}

		// Execute request
		reqID := p.requestID(prePrepare.Request)
		if !p.executedReqs[reqID] {
			result, err := p.stateMachine.Apply(prePrepare.Request.Operation)
			if err != nil {
				p.logger.Error("Failed to apply operation",
					zap.Error(err),
					zap.Int64("sequence", msg.Sequence))
				return err
			}

			p.executedReqs[reqID] = true

			var resultBytes []byte
			if result != nil {
				resultBytes = result
			}

			p.logger.Info("Executed request (committed)",
				zap.Int64("view", msg.View),
				zap.Int64("sequence", msg.Sequence),
				zap.Int("commits", len(p.commitLog[key])),
				zap.ByteString("result", resultBytes))

			// Trigger checkpoint if needed
			if msg.Sequence%100 == 0 {
				go p.createCheckpoint(msg.Sequence)
			}
		}
	}

	return nil
}

// createCheckpoint creates a state checkpoint for garbage collection
func (p *PBFT) createCheckpoint(sequence int64) {
	digest, err := p.stateMachine.Checkpoint(sequence)
	if err != nil {
		p.logger.Error("Failed to create checkpoint", zap.Error(err))
		return
	}

	checkpoint := &CheckpointMessage{
		Sequence:  sequence,
		Digest:    digest,
		ReplicaID: p.nodeID,
		Timestamp: time.Now(),
	}

	if err := p.transport.Broadcast(checkpoint); err != nil {
		p.logger.Error("Failed to broadcast checkpoint", zap.Error(err))
		return
	}

	p.logger.Info("Created checkpoint",
		zap.Int64("sequence", sequence),
		zap.String("digest", digest))
}

// handleCheckpoint processes checkpoint message
func (p *PBFT) handleCheckpoint(msg *CheckpointMessage) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.checkpoints[msg.Sequence] == nil {
		p.checkpoints[msg.Sequence] = make(map[string]*CheckpointMessage)
	}

	p.checkpoints[msg.Sequence][msg.ReplicaID] = msg

	// If we have 2f+1 matching checkpoints, it becomes stable
	if len(p.checkpoints[msg.Sequence]) >= 2*p.f+1 {
		p.stableCheckpoint = msg.Sequence
		p.garbageCollect(msg.Sequence)

		p.logger.Info("Stable checkpoint reached",
			zap.Int64("sequence", msg.Sequence))
	}

	return nil
}

// garbageCollect removes old messages below stable checkpoint
func (p *PBFT) garbageCollect(sequence int64) {
	// Remove old pre-prepares
	for key := range p.prePrepareLog {
		if p.sequenceFromKey(key) < sequence {
			delete(p.prePrepareLog, key)
		}
	}

	// Remove old prepares
	for key := range p.prepareLog {
		if p.sequenceFromKey(key) < sequence {
			delete(p.prepareLog, key)
		}
	}

	// Remove old commits
	for key := range p.commitLog {
		if p.sequenceFromKey(key) < sequence {
			delete(p.commitLog, key)
		}
	}

	// Remove old checkpoints
	for seq := range p.checkpoints {
		if seq < sequence-100 { // Keep some history
			delete(p.checkpoints, seq)
		}
	}
}

// processMessages is the main message processing loop
func (p *PBFT) processMessages() {
	for {
		select {
		case <-p.ctx.Done():
			return
		default:
			msg, err := p.transport.Receive()
			if err != nil {
				continue
			}

			switch m := msg.(type) {
			case *PrePrepareMessage:
				_ = p.handlePrePrepare(m)
			case *PrepareMessage:
				_ = p.handlePrepare(m)
			case *CommitMessage:
				_ = p.handleCommit(m)
			case *CheckpointMessage:
				_ = p.handleCheckpoint(m)
			case *ViewChangeMessage:
				_ = p.handleViewChange(m)
			}
		}
	}
}

// checkpointLoop periodically creates checkpoints
func (p *PBFT) checkpointLoop() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-p.ctx.Done():
			return
		case <-ticker.C:
			p.mu.RLock()
			seq := p.checkpointSeq
			p.mu.RUnlock()

			if seq > 0 && seq%100 == 0 {
				go p.createCheckpoint(seq)
			}
		}
	}
}

// handleViewChange processes view change message
func (p *PBFT) handleViewChange(msg *ViewChangeMessage) error {
	// TODO: Implement view change protocol
	// This is critical for liveness when primary fails
	p.logger.Info("View change requested",
		zap.Int64("new_view", msg.NewView),
		zap.String("from", msg.ReplicaID))
	return nil
}

// Helper functions

func (p *PBFT) updatePrimary() {
	// Primary is determined by: view mod replicaCount
	// For now, use simple strategy: node_0 is primary
	p.isPrimary = p.nodeID == "node_0" // TODO: Better primary selection
	if p.isPrimary {
		p.primaryID = p.nodeID
	}
}

func (p *PBFT) forwardToPrimary(req *ClientRequest) error {
	return p.transport.Send(p.primaryID, req)
}

func (p *PBFT) getNextSequence() int64 {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.checkpointSeq++
	return p.checkpointSeq
}

func (p *PBFT) computeDigest(req *ClientRequest) string {
	data, _ := json.Marshal(req)
	hash := sha256.Sum256(data)
	return hex.EncodeToString(hash[:])
}

func (p *PBFT) requestID(req *ClientRequest) string {
	return fmt.Sprintf("%s:%d", req.ClientID, req.Sequence)
}

func (p *PBFT) logKey(view, sequence int64) string {
	return fmt.Sprintf("%d:%d", view, sequence)
}

func (p *PBFT) sequenceFromKey(key string) int64 {
	var view, seq int64
	fmt.Sscanf(key, "%d:%d", &view, &seq)
	return seq
}

// GetMetrics returns PBFT performance metrics
func (p *PBFT) GetMetrics() *PBFTMetrics {
	p.mu.RLock()
	defer p.mu.RUnlock()

	return &PBFTMetrics{
		View:              p.view,
		CheckpointSeq:     p.checkpointSeq,
		StableCheckpoint:  p.stableCheckpoint,
		ExecutedRequests:  len(p.executedReqs),
		PendingRequests:   len(p.requestQueue),
		PrePrepareLog:     len(p.prePrepareLog),
		PrepareLog:        len(p.prepareLog),
		CommitLog:         len(p.commitLog),
		ByzantineTolerance: p.f,
		ReplicaCount:      p.replicaCount,
		IsPrimary:         p.isPrimary,
	}
}

// PBFTMetrics contains PBFT performance statistics
type PBFTMetrics struct {
	View               int64 `json:"view"`
	CheckpointSeq      int64 `json:"checkpoint_seq"`
	StableCheckpoint   int64 `json:"stable_checkpoint"`
	ExecutedRequests   int   `json:"executed_requests"`
	PendingRequests    int   `json:"pending_requests"`
	PrePrepareLog      int   `json:"pre_prepare_log_size"`
	PrepareLog         int   `json:"prepare_log_size"`
	CommitLog          int   `json:"commit_log_size"`
	ByzantineTolerance int   `json:"byzantine_tolerance"`
	ReplicaCount       int   `json:"replica_count"`
	IsPrimary          bool  `json:"is_primary"`
}
