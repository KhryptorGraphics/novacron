package consensus

import (
	"fmt"
	"sync"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/v3/consensus/probft"
)

// ProBFTConsensus wraps the v3 ProBFT implementation to conform to the StateMachine interface
type ProBFTConsensus struct {
	mu           sync.RWMutex
	nodeID       string
	engine       *probft.ProBFT
	vrf          *probft.VRF
	blockChan    <-chan *probft.Block
	errorChan    <-chan error
	running      bool
	stateMachine StateMachine
}

// NewProBFTConsensus creates a new ProBFT consensus wrapper
func NewProBFTConsensus(nodeID string, sm StateMachine) (*ProBFTConsensus, error) {
	vrf, err := probft.NewVRF()
	if err != nil {
		return nil, fmt.Errorf("failed to create VRF: %w", err)
	}
	config := probft.QuorumConfig{
		TotalNodes:      4, // Minimum for Byzantine tolerance (3f+1 where f=1)
		ByzantineNodes:  1,
		SecurityParam:   0.99,
		ConfidenceLevel: 0.95,
	}

	engine, err := probft.NewProBFT(nodeID, vrf, config)
	if err != nil {
		return nil, fmt.Errorf("failed to create ProBFT: %w", err)
	}

	p := &ProBFTConsensus{
		nodeID:       nodeID,
		engine:       engine,
		vrf:          vrf,
		blockChan:    engine.GetFinalizedBlocks(),
		errorChan:    engine.GetErrors(),
		stateMachine: sm,
	}

	// Set callback for block finalization
	engine.SetBlockFinalizedCallback(p.onBlockFinalized)

	return p, nil
}

// Start starts the ProBFT consensus engine
func (p *ProBFTConsensus) Start() error {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.running {
		return fmt.Errorf("ProBFT already running")
	}

	if err := p.engine.Start(); err != nil {
		return err
	}

	p.running = true

	// Start block processing goroutine
	go p.processBlocks()
	go p.processErrors()

	return nil
}

// Stop stops the ProBFT consensus engine
func (p *ProBFTConsensus) Stop() error {
	p.mu.Lock()
	defer p.mu.Unlock()

	if !p.running {
		return nil
	}

	if err := p.engine.Stop(); err != nil {
		return err
	}

	p.running = false
	return nil
}

// Propose proposes a value to the ProBFT consensus
func (p *ProBFTConsensus) Propose(key string, value []byte) error {
	p.mu.RLock()
	defer p.mu.RUnlock()

	if !p.running {
		return fmt.Errorf("ProBFT not running")
	}

	block := &probft.Block{
		Data:      value,
		Proposer:  p.nodeID,
		Timestamp: time.Now(),
	}

	return p.engine.ProposeBlock(block)
}

// AddNode adds a node to the ProBFT network
// AddNode adds a node to the ProBFT network
func (p *ProBFTConsensus) AddNode(nodeID string, publicKey []byte) error {
	p.mu.RLock()
	defer p.mu.RUnlock()

	node := &probft.Node{
		ID:        nodeID,
		PublicKey: publicKey,
		IsActive:  true,
	}

	return p.engine.AddNode(node)
}

// onBlockFinalized is called when a block is finalized
func (p *ProBFTConsensus) onBlockFinalized(block *probft.Block) error {
	// Apply the finalized block to the state machine
	cmd := Command{
		Type:  "set",
		Key:   string(block.Data),
		Value: block.Data,
	}
	_, err := p.stateMachine.Apply(cmd)
	return err
}

// processBlocks processes finalized blocks
func (p *ProBFTConsensus) processBlocks() {
	for block := range p.blockChan {
		// Block is already applied via callback, but we can log/monitor here
		_ = block
	}
}

// processErrors processes errors from the ProBFT engine
func (p *ProBFTConsensus) processErrors() {
	for err := range p.errorChan {
		// Log errors
		fmt.Printf("ProBFT error: %v\n", err)
	}
}

// LoadSnapshot loads a snapshot into ProBFT
func (p *ProBFTConsensus) LoadSnapshot(snapshot *Snapshot) error {
	// ProBFT doesn't use traditional snapshots, but we can use this for state transfer
	// For now, return nil (no-op)
	return nil
}

// IsRunning returns whether ProBFT is running
func (p *ProBFTConsensus) IsRunning() bool {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.running
}

// GetState returns current ProBFT state
func (p *ProBFTConsensus) GetState() *probft.ConsensusState {
	state := p.engine.GetState()
	return &state
}