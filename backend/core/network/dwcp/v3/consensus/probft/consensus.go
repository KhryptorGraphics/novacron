// Package probft implements Probabilistic Byzantine Fault Tolerance consensus
package probft

import (
	"context"
	"crypto/ed25519"
	"errors"
	"fmt"
	"sync"
	"time"
)

// Phase represents the consensus phase
type Phase int

const (
	PhaseIdle Phase = iota
	PhasePrePrepare
	PhasePrepare
	PhaseCommit
	PhaseFinalized
)

// Block represents a consensus block
type Block struct {
	Height    uint64
	Hash      []byte
	Data      []byte
	Timestamp time.Time
	Proposer  string
	VRFProof  *VRFProof
}

// Node represents a consensus participant
type Node struct {
	ID        string
	PublicKey ed25519.PublicKey
	Address   string
	IsActive  bool
}

// Message represents a consensus message
type Message struct {
	Type      MessageType
	Phase     Phase
	Block     *Block
	NodeID    string
	Signature []byte
	Timestamp time.Time
}

// MessageType defines consensus message types
type MessageType int

const (
	MessagePrePrepare MessageType = iota
	MessagePrepare
	MessageCommit
	MessageViewChange
)

// ConsensusState tracks the current consensus state
type ConsensusState struct {
	Phase           Phase
	Height          uint64
	View            uint64
	ProposedBlock   *Block
	PrepareVotes    map[string]*Message
	CommitVotes     map[string]*Message
	QuorumSize      int
	StartTime       time.Time
	mu              sync.RWMutex
}

// ProBFT implements the Probabilistic BFT consensus engine
type ProBFT struct {
	nodeID     string
	nodes      map[string]*Node
	vrf        *VRF
	config     QuorumConfig
	state      *ConsensusState

	// Channels
	messageChan chan *Message
	blockChan   chan *Block
	errorChan   chan error

	// Callbacks
	onBlockFinalized func(*Block) error

	// Context and cancellation
	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup

	mu sync.RWMutex
}

// NewProBFT creates a new ProBFT consensus engine
func NewProBFT(nodeID string, vrf *VRF, config QuorumConfig) (*ProBFT, error) {
	if err := ValidateQuorumConfig(config); err != nil {
		return nil, fmt.Errorf("invalid quorum config: %w", err)
	}

	ctx, cancel := context.WithCancel(context.Background())

	state := &ConsensusState{
		Phase:        PhaseIdle,
		Height:       0,
		View:         0,
		PrepareVotes: make(map[string]*Message),
		CommitVotes:  make(map[string]*Message),
		QuorumSize:   CalculateQuorum(config.TotalNodes),
	}

	return &ProBFT{
		nodeID:      nodeID,
		nodes:       make(map[string]*Node),
		vrf:         vrf,
		config:      config,
		state:       state,
		messageChan: make(chan *Message, 1000),
		blockChan:   make(chan *Block, 100),
		errorChan:   make(chan error, 100),
		ctx:         ctx,
		cancel:      cancel,
	}, nil
}

// Start starts the consensus engine
func (p *ProBFT) Start() error {
	p.wg.Add(1)
	go p.consensusLoop()
	return nil
}

// Stop stops the consensus engine
func (p *ProBFT) Stop() error {
	p.cancel()
	p.wg.Wait()
	close(p.messageChan)
	close(p.blockChan)
	close(p.errorChan)
	return nil
}

// AddNode adds a node to the consensus network
func (p *ProBFT) AddNode(node *Node) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	if node == nil {
		return errors.New("node cannot be nil")
	}

	p.nodes[node.ID] = node

	// Recalculate quorum
	p.updateQuorum()

	return nil
}

// RemoveNode removes a node from the consensus network
func (p *ProBFT) RemoveNode(nodeID string) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	delete(p.nodes, nodeID)
	p.updateQuorum()

	return nil
}

// updateQuorum recalculates quorum size based on active nodes
func (p *ProBFT) updateQuorum() {
	activeCount := 0
	for _, node := range p.nodes {
		if node.IsActive {
			activeCount++
		}
	}

	p.state.mu.Lock()
	p.state.QuorumSize = CalculateQuorum(activeCount)
	p.state.mu.Unlock()
}

// ProposeBlock proposes a new block for consensus
// Phase 1: Pre-prepare with VRF leader election
func (p *ProBFT) ProposeBlock(block *Block) error {
	p.state.mu.Lock()
	defer p.state.mu.Unlock()

	if p.state.Phase != PhaseIdle {
		return fmt.Errorf("cannot propose block in phase %d", p.state.Phase)
	}

	// Generate VRF proof for leader election
	input := []byte(fmt.Sprintf("%d:%d", p.state.Height, p.state.View))
	proof, err := p.vrf.Prove(input)
	if err != nil {
		return fmt.Errorf("failed to generate VRF proof: %w", err)
	}

	block.VRFProof = proof
	block.Proposer = p.nodeID
	block.Height = p.state.Height
	block.Timestamp = time.Now()

	// Verify we are the legitimate leader
	activeNodes := p.getActiveNodes()
	leaderIndex := SelectLeader(proof.Output, len(activeNodes))
	if activeNodes[leaderIndex].ID != p.nodeID {
		return errors.New("not the designated leader for this round")
	}

	// Transition to pre-prepare phase
	p.state.Phase = PhasePrePrepare
	p.state.ProposedBlock = block
	p.state.StartTime = time.Now()

	// Broadcast pre-prepare message
	msg := &Message{
		Type:      MessagePrePrepare,
		Phase:     PhasePrePrepare,
		Block:     block,
		NodeID:    p.nodeID,
		Timestamp: time.Now(),
	}

	go p.broadcastMessage(msg)

	return nil
}

// HandleMessage processes incoming consensus messages
func (p *ProBFT) HandleMessage(msg *Message) error {
	if msg == nil {
		return errors.New("message cannot be nil")
	}

	select {
	case p.messageChan <- msg:
		return nil
	case <-p.ctx.Done():
		return errors.New("consensus engine stopped")
	}
}

// consensusLoop is the main consensus processing loop
func (p *ProBFT) consensusLoop() {
	defer p.wg.Done()

	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-p.ctx.Done():
			return

		case msg := <-p.messageChan:
			if err := p.processMessage(msg); err != nil {
				p.errorChan <- fmt.Errorf("error processing message: %w", err)
			}

		case <-ticker.C:
			p.checkTimeout()
		}
	}
}

// processMessage processes a consensus message based on current phase
func (p *ProBFT) processMessage(msg *Message) error {
	p.state.mu.Lock()
	defer p.state.mu.Unlock()

	switch msg.Type {
	case MessagePrePrepare:
		return p.handlePrePrepare(msg)
	case MessagePrepare:
		return p.handlePrepare(msg)
	case MessageCommit:
		return p.handleCommit(msg)
	case MessageViewChange:
		return p.handleViewChange(msg)
	default:
		return fmt.Errorf("unknown message type: %d", msg.Type)
	}
}

// handlePrePrepare handles pre-prepare phase messages
func (p *ProBFT) handlePrePrepare(msg *Message) error {
	if p.state.Phase != PhaseIdle && p.state.Phase != PhasePrePrepare {
		return nil // Ignore if not in correct phase
	}

	// Verify VRF proof
	activeNodes := p.getActiveNodes()
	input := []byte(fmt.Sprintf("%d:%d", msg.Block.Height, p.state.View))

	leaderIndex := SelectLeader(msg.Block.VRFProof.Output, len(activeNodes))
	expectedLeader := activeNodes[leaderIndex]

	if expectedLeader.ID != msg.NodeID {
		return errors.New("invalid leader for this round")
	}

	if !VerifyVRF(expectedLeader.PublicKey, input, msg.Block.VRFProof) {
		return errors.New("invalid VRF proof")
	}

	// Accept block and move to prepare phase
	p.state.ProposedBlock = msg.Block
	p.state.Phase = PhasePrepare
	p.state.PrepareVotes = make(map[string]*Message)

	// Send prepare message
	prepareMsg := &Message{
		Type:      MessagePrepare,
		Phase:     PhasePrepare,
		Block:     msg.Block,
		NodeID:    p.nodeID,
		Timestamp: time.Now(),
	}

	go p.broadcastMessage(prepareMsg)

	return nil
}

// handlePrepare handles prepare phase messages (Phase 2)
func (p *ProBFT) handlePrepare(msg *Message) error {
	if p.state.Phase != PhasePrepare {
		return nil
	}

	// Store prepare vote
	p.state.PrepareVotes[msg.NodeID] = msg

	// Check if we have probabilistic quorum
	if len(p.state.PrepareVotes) >= p.state.QuorumSize {
		// Move to commit phase
		p.state.Phase = PhaseCommit
		p.state.CommitVotes = make(map[string]*Message)

		// Send commit message
		commitMsg := &Message{
			Type:      MessageCommit,
			Phase:     PhaseCommit,
			Block:     p.state.ProposedBlock,
			NodeID:    p.nodeID,
			Timestamp: time.Now(),
		}

		go p.broadcastMessage(commitMsg)
	}

	return nil
}

// handleCommit handles commit phase messages (Phase 3)
func (p *ProBFT) handleCommit(msg *Message) error {
	if p.state.Phase != PhaseCommit {
		return nil
	}

	// Store commit vote
	p.state.CommitVotes[msg.NodeID] = msg

	// Check if we have probabilistic quorum for finalization
	if len(p.state.CommitVotes) >= p.state.QuorumSize {
		// Finalize block
		p.state.Phase = PhaseFinalized

		if p.onBlockFinalized != nil {
			go func() {
				if err := p.onBlockFinalized(p.state.ProposedBlock); err != nil {
					p.errorChan <- fmt.Errorf("block finalization callback error: %w", err)
				}
			}()
		}

		// Send finalized block
		select {
		case p.blockChan <- p.state.ProposedBlock:
		case <-p.ctx.Done():
			return errors.New("consensus stopped")
		}

		// Reset for next round
		p.state.Height++
		p.state.Phase = PhaseIdle
		p.state.ProposedBlock = nil
		p.state.PrepareVotes = make(map[string]*Message)
		p.state.CommitVotes = make(map[string]*Message)
	}

	return nil
}

// handleViewChange handles view change for leader rotation
func (p *ProBFT) handleViewChange(msg *Message) error {
	p.state.View++
	p.state.Phase = PhaseIdle
	p.state.PrepareVotes = make(map[string]*Message)
	p.state.CommitVotes = make(map[string]*Message)
	return nil
}

// checkTimeout checks for consensus timeout and triggers view change
func (p *ProBFT) checkTimeout() {
	p.state.mu.RLock()
	phase := p.state.Phase
	startTime := p.state.StartTime
	p.state.mu.RUnlock()

	if phase != PhaseIdle && time.Since(startTime) > 30*time.Second {
		// Timeout - trigger view change
		viewChangeMsg := &Message{
			Type:      MessageViewChange,
			NodeID:    p.nodeID,
			Timestamp: time.Now(),
		}
		_ = p.HandleMessage(viewChangeMsg)
	}
}

// broadcastMessage broadcasts a message to all nodes
func (p *ProBFT) broadcastMessage(msg *Message) {
	// In a real implementation, this would send to network
	// For now, we'll just handle it locally for testing
	_ = p.HandleMessage(msg)
}

// getActiveNodes returns list of active nodes
func (p *ProBFT) getActiveNodes() []*Node {
	nodes := make([]*Node, 0, len(p.nodes))
	for _, node := range p.nodes {
		if node.IsActive {
			nodes = append(nodes, node)
		}
	}
	return nodes
}

// GetFinalizedBlocks returns channel for finalized blocks
func (p *ProBFT) GetFinalizedBlocks() <-chan *Block {
	return p.blockChan
}

// GetErrors returns channel for errors
func (p *ProBFT) GetErrors() <-chan error {
	return p.errorChan
}

// SetBlockFinalizedCallback sets callback for block finalization
func (p *ProBFT) SetBlockFinalizedCallback(callback func(*Block) error) {
	p.onBlockFinalized = callback
}

// GetState returns current consensus state (for monitoring)
func (p *ProBFT) GetState() ConsensusState {
	p.state.mu.RLock()
	defer p.state.mu.RUnlock()

	// Return copy to prevent external modification
	return ConsensusState{
		Phase:         p.state.Phase,
		Height:        p.state.Height,
		View:          p.state.View,
		ProposedBlock: p.state.ProposedBlock,
		QuorumSize:    p.state.QuorumSize,
		StartTime:     p.state.StartTime,
	}
}
