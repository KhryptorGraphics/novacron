package consensus

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/upgrade"
	"go.uber.org/zap"
)

// ACPv3 implements Adaptive Consensus Protocol v3
// Automatically selects optimal consensus algorithm based on network mode:
// - Datacenter: Raft (fast, trusted, <100ms)
// - Internet: PBFT (Byzantine-tolerant, 1-5 seconds)
// - Hybrid: Adaptive switching with fallback
type ACPv3 struct {
	mu sync.RWMutex

	nodeID string
	mode   upgrade.NetworkMode

	// Consensus implementations
	raft   *RaftConsensus
	pbft   *PBFT
	gossip *GossipConsensus

	// Mode detector
	modeDetector *upgrade.ModeDetector

	// Metrics
	consensusCount    int64
	lastConsensusTime time.Time
	avgConsensusTime  time.Duration
	failoverCount     int64

	logger *zap.Logger
	ctx    context.Context
	cancel context.CancelFunc
}

// RaftConsensus implements Raft consensus for datacenter mode
type RaftConsensus struct {
	mu       sync.RWMutex
	nodeID   string
	raftNode RaftNode
	logger   *zap.Logger
}

// GossipConsensus implements gossip-based eventual consistency
type GossipConsensus struct {
	mu       sync.RWMutex
	nodeID   string
	peers    []string
	fanout   int
	interval time.Duration
	logger   *zap.Logger
}

// RaftNode interface for Raft integration
type RaftNode interface {
	Propose(ctx context.Context, data []byte) error
	ReadIndex(ctx context.Context) (uint64, error)
	IsLeader() bool
}

// NewACPv3 creates a new Adaptive Consensus Protocol v3
func NewACPv3(nodeID string, mode upgrade.NetworkMode, config *ACPConfig, logger *zap.Logger) (*ACPv3, error) {
	ctx, cancel := context.WithCancel(context.Background())

	acp := &ACPv3{
		nodeID:       nodeID,
		mode:         mode,
		modeDetector: upgrade.NewModeDetector(),
		logger:       logger,
		ctx:          ctx,
		cancel:       cancel,
	}

	// Initialize Raft for datacenter mode
	acp.raft = &RaftConsensus{
		nodeID: nodeID,
		logger: logger,
	}

	// Initialize PBFT for internet mode
	if config.PBFTConfig != nil {
		pbft, err := NewPBFT(
			nodeID,
			config.PBFTConfig.ReplicaCount,
			config.PBFTConfig.Transport,
			config.PBFTConfig.StateMachine,
			logger,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to initialize PBFT: %w", err)
		}
		acp.pbft = pbft
	}

	// Initialize Gossip for discovery and fallback
	acp.gossip = &GossipConsensus{
		nodeID:   nodeID,
		peers:    config.GossipPeers,
		fanout:   3,
		interval: 5 * time.Second,
		logger:   logger,
	}

	return acp, nil
}

// ACPConfig contains configuration for ACP v3
type ACPConfig struct {
	PBFTConfig  *PBFTConfig
	GossipPeers []string
}

// PBFTConfig contains PBFT-specific configuration
type PBFTConfig struct {
	ReplicaCount int
	Transport    Transport
	StateMachine StateMachine
}

// Start begins ACP v3 operation
func (a *ACPv3) Start() error {
	a.logger.Info("Starting ACP v3",
		zap.String("node_id", a.nodeID),
		zap.String("mode", a.mode.String()))

	// Start mode-specific consensus engines
	switch a.mode {
	case upgrade.ModeDatacenter:
		// Raft is typically started externally
		a.logger.Info("Datacenter mode: Using Raft consensus")

	case upgrade.ModeInternet:
		if a.pbft != nil {
			if err := a.pbft.Start(); err != nil {
				return fmt.Errorf("failed to start PBFT: %w", err)
			}
			a.logger.Info("Internet mode: Using PBFT consensus",
				zap.Int("byzantine_tolerance", a.pbft.f))
		}

	case upgrade.ModeHybrid:
		// Start both for adaptive switching
		if a.pbft != nil {
			if err := a.pbft.Start(); err != nil {
				return fmt.Errorf("failed to start PBFT: %w", err)
			}
		}
		go a.adaptiveModeLoop()
		a.logger.Info("Hybrid mode: Adaptive consensus enabled")
	}

	return nil
}

// Stop halts ACP v3 operation
func (a *ACPv3) Stop() error {
	a.logger.Info("Stopping ACP v3", zap.String("node_id", a.nodeID))

	a.cancel()

	if a.pbft != nil {
		if err := a.pbft.Stop(); err != nil {
			a.logger.Error("Failed to stop PBFT", zap.Error(err))
		}
	}

	return nil
}

// Consensus executes consensus on a value using mode-appropriate algorithm
func (a *ACPv3) Consensus(ctx context.Context, value interface{}) error {
	startTime := time.Now()
	defer func() {
		a.mu.Lock()
		a.consensusCount++
		a.lastConsensusTime = time.Now()
		duration := time.Since(startTime)
		// Update average (exponential moving average)
		if a.avgConsensusTime == 0 {
			a.avgConsensusTime = duration
		} else {
			a.avgConsensusTime = (a.avgConsensusTime*9 + duration) / 10
		}
		a.mu.Unlock()
	}()

	a.mu.RLock()
	mode := a.mode
	a.mu.RUnlock()

	switch mode {
	case upgrade.ModeDatacenter:
		// Use Raft for fast, trusted consensus (<100ms)
		return a.datacenterConsensus(ctx, value)

	case upgrade.ModeInternet:
		// Use PBFT for Byzantine-tolerant consensus (1-5 seconds)
		return a.internetConsensus(ctx, value)

	case upgrade.ModeHybrid:
		// Adaptive: Try Raft first, fallback to PBFT
		return a.hybridConsensus(ctx, value)

	default:
		return fmt.Errorf("unknown network mode: %v", mode)
	}
}

// datacenterConsensus uses Raft for strong consistency
func (a *ACPv3) datacenterConsensus(ctx context.Context, value interface{}) error {
	if a.raft.raftNode == nil {
		return fmt.Errorf("Raft node not initialized")
	}

	// Serialize value
	data, err := serializeValue(value)
	if err != nil {
		return fmt.Errorf("failed to serialize value: %w", err)
	}

	// Create timeout context for datacenter (<100ms target)
	timeoutCtx, cancel := context.WithTimeout(ctx, 100*time.Millisecond)
	defer cancel()

	// Propose to Raft
	if err := a.raft.raftNode.Propose(timeoutCtx, data); err != nil {
		return fmt.Errorf("Raft consensus failed: %w", err)
	}

	a.logger.Debug("Datacenter consensus completed (Raft)")

	return nil
}

// internetConsensus uses PBFT for Byzantine tolerance
func (a *ACPv3) internetConsensus(ctx context.Context, value interface{}) error {
	if a.pbft == nil {
		return fmt.Errorf("PBFT not initialized")
	}

	// Create timeout context for internet (1-5 seconds acceptable)
	timeoutCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	// Execute PBFT consensus
	if err := a.pbft.Consensus(timeoutCtx, value); err != nil {
		return fmt.Errorf("PBFT consensus failed: %w", err)
	}

	a.logger.Debug("Internet consensus completed (PBFT)")

	return nil
}

// hybridConsensus adaptively chooses best consensus algorithm
func (a *ACPv3) hybridConsensus(ctx context.Context, value interface{}) error {
	// Detect current network conditions
	detectedMode := a.modeDetector.DetectMode(ctx)

	a.logger.Debug("Hybrid consensus mode detected",
		zap.String("mode", detectedMode.String()))

	// Try Raft first if conditions are good
	if detectedMode == upgrade.ModeDatacenter && a.raft.raftNode != nil {
		timeoutCtx, cancel := context.WithTimeout(ctx, 100*time.Millisecond)
		defer cancel()

		if err := a.datacenterConsensus(timeoutCtx, value); err == nil {
			return nil
		} else {
			// Raft failed, record failover
			a.mu.Lock()
			a.failoverCount++
			a.mu.Unlock()

			a.logger.Warn("Raft consensus failed, failing over to PBFT", zap.Error(err))
		}
	}

	// Fallback to PBFT for reliability
	if a.pbft != nil {
		return a.internetConsensus(ctx, value)
	}

	return fmt.Errorf("no consensus mechanism available")
}

// adaptiveModeLoop continuously adapts to network conditions
func (a *ACPv3) adaptiveModeLoop() {
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-a.ctx.Done():
			return
		case <-ticker.C:
			// Detect optimal mode
			newMode := a.modeDetector.DetectMode(a.ctx)

			a.mu.Lock()
			if newMode != a.mode {
				a.logger.Info("Consensus mode changed",
					zap.String("old_mode", a.mode.String()),
					zap.String("new_mode", newMode.String()))
				a.mode = newMode
			}
			a.mu.Unlock()
		}
	}
}

// SetRaftNode sets the Raft node for datacenter consensus
func (a *ACPv3) SetRaftNode(node RaftNode) {
	a.raft.mu.Lock()
	defer a.raft.mu.Unlock()
	a.raft.raftNode = node
}

// SetMode manually sets the consensus mode
func (a *ACPv3) SetMode(mode upgrade.NetworkMode) {
	a.mu.Lock()
	defer a.mu.Unlock()

	a.logger.Info("Consensus mode changed",
		zap.String("old_mode", a.mode.String()),
		zap.String("new_mode", mode.String()))

	a.mode = mode
}

// GetMode returns the current consensus mode
func (a *ACPv3) GetMode() upgrade.NetworkMode {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.mode
}

// GetMetrics returns ACP v3 performance metrics
func (a *ACPv3) GetMetrics() *ACPv3Metrics {
	a.mu.RLock()
	defer a.mu.RUnlock()

	metrics := &ACPv3Metrics{
		Mode:              a.mode.String(),
		ConsensusCount:    a.consensusCount,
		LastConsensusTime: a.lastConsensusTime,
		AvgConsensusTime:  a.avgConsensusTime,
		FailoverCount:     a.failoverCount,
	}

	// Add PBFT metrics if available
	if a.pbft != nil {
		metrics.PBFTMetrics = a.pbft.GetMetrics()
	}

	return metrics
}

// ACPv3Metrics contains performance statistics
type ACPv3Metrics struct {
	Mode              string        `json:"mode"`
	ConsensusCount    int64         `json:"consensus_count"`
	LastConsensusTime time.Time     `json:"last_consensus_time"`
	AvgConsensusTime  time.Duration `json:"avg_consensus_time"`
	FailoverCount     int64         `json:"failover_count"`
	PBFTMetrics       *PBFTMetrics  `json:"pbft_metrics,omitempty"`
}

// IsHealthy checks if ACP v3 is operating normally
func (a *ACPv3) IsHealthy() bool {
	a.mu.RLock()
	mode := a.mode
	a.mu.RUnlock()

	switch mode {
	case upgrade.ModeDatacenter:
		return a.raft.raftNode != nil && a.raft.raftNode.IsLeader()
	case upgrade.ModeInternet:
		return a.pbft != nil
	case upgrade.ModeHybrid:
		return a.raft.raftNode != nil || a.pbft != nil
	default:
		return false
	}
}

// GetConsensusLatency returns expected consensus latency for current mode
func (a *ACPv3) GetConsensusLatency() time.Duration {
	a.mu.RLock()
	mode := a.mode
	a.mu.RUnlock()

	switch mode {
	case upgrade.ModeDatacenter:
		return 100 * time.Millisecond // Raft target
	case upgrade.ModeInternet:
		return 2 * time.Second // PBFT typical
	case upgrade.ModeHybrid:
		return 500 * time.Millisecond // Average
	default:
		return 1 * time.Second
	}
}

// Helper functions

func serializeValue(value interface{}) ([]byte, error) {
	// TODO: Implement proper serialization
	return []byte(fmt.Sprintf("%v", value)), nil
}
