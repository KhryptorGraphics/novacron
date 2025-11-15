// Package probft provides integration with ACP v3 consensus
package probft

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"
)

// ACPIntegration provides integration between ProBFT and ACP v3
type ACPIntegration struct {
	probft    *ProBFT
	acpConfig ACPConfig

	// State management
	isRunning bool
	mu        sync.RWMutex

	// Metrics
	metrics *ConsensusMetrics
}

// ACPConfig holds ACP v3 configuration
type ACPConfig struct {
	NetworkID       string
	ChainID         uint64
	MinValidators   int
	MaxValidators   int
	BlockTime       time.Duration
	EnableProBFT    bool
	ByzantineTolerance float64 // Target Byzantine tolerance (e.g., 0.33 for 33%)
}

// ConsensusMetrics tracks consensus performance
type ConsensusMetrics struct {
	BlocksFinalized     uint64
	AverageBlockTime    time.Duration
	VRFComputations     uint64
	QuorumReached       uint64
	ViewChanges         uint64
	ByzantineDetected   uint64
	LastFinalizedHeight uint64
	LastFinalizedTime   time.Time

	mu sync.RWMutex
}

// NewACPIntegration creates a new ACP integration instance
func NewACPIntegration(probft *ProBFT, config ACPConfig) (*ACPIntegration, error) {
	if probft == nil {
		return nil, errors.New("probft instance cannot be nil")
	}

	if config.MinValidators < 4 {
		return nil, errors.New("minimum validators must be at least 4 for Byzantine tolerance")
	}

	return &ACPIntegration{
		probft:    probft,
		acpConfig: config,
		metrics:   &ConsensusMetrics{},
	}, nil
}

// Start starts the integrated consensus system
func (a *ACPIntegration) Start(ctx context.Context) error {
	a.mu.Lock()
	if a.isRunning {
		a.mu.Unlock()
		return errors.New("consensus already running")
	}
	a.isRunning = true
	a.mu.Unlock()

	// Start ProBFT engine
	if err := a.probft.Start(); err != nil {
		return fmt.Errorf("failed to start ProBFT: %w", err)
	}

	// Start monitoring goroutine
	go a.monitorConsensus(ctx)

	return nil
}

// Stop stops the integrated consensus system
func (a *ACPIntegration) Stop() error {
	a.mu.Lock()
	if !a.isRunning {
		a.mu.Unlock()
		return errors.New("consensus not running")
	}
	a.isRunning = false
	a.mu.Unlock()

	return a.probft.Stop()
}

// monitorConsensus monitors consensus progress and metrics
func (a *ACPIntegration) monitorConsensus(ctx context.Context) {
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return

		case block := <-a.probft.GetFinalizedBlocks():
			a.handleFinalizedBlock(block)

		case err := <-a.probft.GetErrors():
			a.handleConsensusError(err)

		case <-ticker.C:
			a.updateMetrics()
		}
	}
}

// handleFinalizedBlock processes a finalized block
func (a *ACPIntegration) handleFinalizedBlock(block *Block) {
	if block == nil {
		return
	}

	a.metrics.mu.Lock()
	defer a.metrics.mu.Unlock()

	a.metrics.BlocksFinalized++
	a.metrics.LastFinalizedHeight = block.Height
	now := time.Now()

	if !a.metrics.LastFinalizedTime.IsZero() {
		blockTime := now.Sub(a.metrics.LastFinalizedTime)
		// Update average with exponential moving average
		if a.metrics.AverageBlockTime == 0 {
			a.metrics.AverageBlockTime = blockTime
		} else {
			a.metrics.AverageBlockTime = (a.metrics.AverageBlockTime*9 + blockTime) / 10
		}
	}

	a.metrics.LastFinalizedTime = now

	// Track VRF computations
	if block.VRFProof != nil {
		a.metrics.VRFComputations++
	}
}

// handleConsensusError handles consensus errors
func (a *ACPIntegration) handleConsensusError(err error) {
	// Log error and update metrics
	// In production, this would integrate with logging system
	fmt.Printf("Consensus error: %v\n", err)
}

// updateMetrics updates consensus metrics
func (a *ACPIntegration) updateMetrics() {
	state := a.probft.GetState()

	a.metrics.mu.Lock()
	defer a.metrics.mu.Unlock()

	// Track quorum reaches
	prepareCount := len(state.PrepareVotes)
	commitCount := len(state.CommitVotes)

	if prepareCount >= state.QuorumSize || commitCount >= state.QuorumSize {
		a.metrics.QuorumReached++
	}
}

// GetMetrics returns current consensus metrics
func (a *ACPIntegration) GetMetrics() ConsensusMetrics {
	a.metrics.mu.RLock()
	defer a.metrics.mu.RUnlock()

	// Return copy
	return ConsensusMetrics{
		BlocksFinalized:     a.metrics.BlocksFinalized,
		AverageBlockTime:    a.metrics.AverageBlockTime,
		VRFComputations:     a.metrics.VRFComputations,
		QuorumReached:       a.metrics.QuorumReached,
		ViewChanges:         a.metrics.ViewChanges,
		ByzantineDetected:   a.metrics.ByzantineDetected,
		LastFinalizedHeight: a.metrics.LastFinalizedHeight,
		LastFinalizedTime:   a.metrics.LastFinalizedTime,
	}
}

// ValidateACPCompatibility validates ProBFT configuration against ACP requirements
func (a *ACPIntegration) ValidateACPCompatibility() error {
	config := a.probft.config

	// Validate Byzantine tolerance
	maxByzantine := CalculateMaxByzantineNodes(config.TotalNodes)
	actualTolerance := float64(maxByzantine) / float64(config.TotalNodes)

	if actualTolerance < a.acpConfig.ByzantineTolerance {
		return fmt.Errorf("Byzantine tolerance %.2f%% below required %.2f%%",
			actualTolerance*100, a.acpConfig.ByzantineTolerance*100)
	}

	// Validate validator count
	activeNodes := 0
	for _, node := range a.probft.nodes {
		if node.IsActive {
			activeNodes++
		}
	}

	if activeNodes < a.acpConfig.MinValidators {
		return fmt.Errorf("active validators %d below minimum %d",
			activeNodes, a.acpConfig.MinValidators)
	}

	if activeNodes > a.acpConfig.MaxValidators {
		return fmt.Errorf("active validators %d exceeds maximum %d",
			activeNodes, a.acpConfig.MaxValidators)
	}

	return nil
}

// AdaptToNetworkConditions dynamically adjusts consensus parameters
func (a *ACPIntegration) AdaptToNetworkConditions(condition NetworkCondition) error {
	config := a.probft.config

	// Calculate adaptive quorum
	result, err := CalculateAdaptiveQuorum(config, condition)
	if err != nil {
		return fmt.Errorf("failed to calculate adaptive quorum: %w", err)
	}

	// Update quorum size
	a.probft.state.mu.Lock()
	a.probft.state.QuorumSize = result.QuorumSize
	a.probft.state.mu.Unlock()

	return nil
}

// GetConsensusStatus returns current consensus status
func (a *ACPIntegration) GetConsensusStatus() map[string]interface{} {
	a.mu.RLock()
	isRunning := a.isRunning
	a.mu.RUnlock()

	state := a.probft.GetState()
	metrics := a.GetMetrics()

	return map[string]interface{}{
		"running":             isRunning,
		"phase":               state.Phase,
		"height":              state.Height,
		"view":                state.View,
		"quorum_size":         state.QuorumSize,
		"blocks_finalized":    metrics.BlocksFinalized,
		"average_block_time":  metrics.AverageBlockTime.Seconds(),
		"vrf_computations":    metrics.VRFComputations,
		"quorum_reached":      metrics.QuorumReached,
		"byzantine_detected":  metrics.ByzantineDetected,
		"last_finalized":      metrics.LastFinalizedHeight,
	}
}

// ExportProBFTConfig exports ProBFT configuration for ACP
func (a *ACPIntegration) ExportProBFTConfig() map[string]interface{} {
	config := a.probft.config
	maxByzantine := CalculateMaxByzantineNodes(config.TotalNodes)
	tolerance := CalculateByzantineTolerance(config.TotalNodes, maxByzantine)

	return map[string]interface{}{
		"total_nodes":         config.TotalNodes,
		"max_byzantine_nodes": maxByzantine,
		"byzantine_tolerance": fmt.Sprintf("%.1f%%", tolerance),
		"quorum_size":         CalculateQuorum(config.TotalNodes),
		"security_param":      config.SecurityParam,
		"confidence_level":    config.ConfidenceLevel,
		"acp_compatible":      a.ValidateACPCompatibility() == nil,
	}
}

// SimulateByzantineScenario simulates Byzantine attack scenario for testing
func (a *ACPIntegration) SimulateByzantineScenario(byzantineRatio float64) error {
	config := a.probft.config
	byzantineCount := int(float64(config.TotalNodes) * byzantineRatio)

	if byzantineCount > CalculateMaxByzantineNodes(config.TotalNodes) {
		return fmt.Errorf("Byzantine ratio %.2f%% exceeds maximum tolerance",
			byzantineRatio*100)
	}

	// Update metrics to reflect Byzantine detection
	a.metrics.mu.Lock()
	a.metrics.ByzantineDetected += uint64(byzantineCount)
	a.metrics.mu.Unlock()

	// Test if consensus can continue
	condition := NetworkCondition{
		ActiveNodes:    config.TotalNodes - byzantineCount,
		ByzantineRatio: byzantineRatio,
		FailureRate:    byzantineRatio,
	}

	return a.AdaptToNetworkConditions(condition)
}
