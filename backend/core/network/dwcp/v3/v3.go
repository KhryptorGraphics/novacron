// Package v3 provides the v3 implementation of DWCP (Distributed Weighted Consensus Protocol)
// This package serves as a namespace for v3 consensus protocols including ProBFT, Bullshark, and T-PBFT
package v3

import (
	"context"
	"time"
)

// Node represents a node in the DWCP v3 network
type Node struct {
	ID         int
	Config     *Config
	ctx        context.Context
	cancel     context.CancelFunc
	consensusEngine interface{}
}

// Config holds the configuration for a DWCP v3 node
type Config struct {
	NodeID           int
	TotalNodes       int
	ByzantineNodes   int
	NetworkLatency   time.Duration
	ConsensusTimeout time.Duration
	MaxBlockSize     int
}

// NewNode creates a new DWCP v3 node
func NewNode(config *Config) *Node {
	ctx, cancel := context.WithCancel(context.Background())
	return &Node{
		ID:     config.NodeID,
		Config: config,
		ctx:    ctx,
		cancel: cancel,
	}
}

// Start starts the node
func (n *Node) Start() error {
	// Stub implementation for testing
	return nil
}

// Stop stops the node gracefully
func (n *Node) Stop() error {
	if n.cancel != nil {
		n.cancel()
	}
	return nil
}

// ProposeBlock proposes a new block (stub for chaos testing)
func (n *Node) ProposeBlock(data []byte) error {
	return nil
}

// GetConsensusState returns the current consensus state (stub)
func (n *Node) GetConsensusState() string {
	return "active"
}

// IsByzantine returns whether this node is Byzantine (for testing)
func (n *Node) IsByzantine() bool {
	return false
}

// SetByzantine sets Byzantine behavior (for chaos testing)
func (n *Node) SetByzantine(byzantine bool) {
	// Stub for testing
}
