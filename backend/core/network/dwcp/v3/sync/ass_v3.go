package sync

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/sync/crdt"
	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/upgrade"
	"go.uber.org/zap"
)

// ASSv3 implements mode-aware Async State Synchronization
// Adapts synchronization strategy based on network conditions:
// - Datacenter mode: Raft for strong consistency (<100ms)
// - Internet mode: CRDT for eventual consistency (5-30 seconds)
// - Hybrid mode: Adaptive switching with conflict resolution
type ASSv3 struct {
	mu sync.RWMutex

	nodeID string
	mode   upgrade.NetworkMode

	// Mode-specific synchronizers
	raftSync *RaftStateSync
	crdtSync *CRDTStateSync

	// Conflict resolution for hybrid mode
	conflictResolver *ConflictResolver

	// Mode detector for automatic switching
	modeDetector *upgrade.ModeDetector

	// Metrics
	syncCount      int64
	lastSyncTime   time.Time
	avgSyncLatency time.Duration

	logger *zap.Logger
	ctx    context.Context
	cancel context.CancelFunc
}

// RaftStateSync implements strong consistency synchronization for datacenter mode
type RaftStateSync struct {
	mu       sync.RWMutex
	nodeID   string
	raftNode RaftNode
	logger   *zap.Logger
}

// CRDTStateSync implements eventual consistency synchronization for internet mode
type CRDTStateSync struct {
	mu         sync.RWMutex
	nodeID     string
	crdtStore  map[string]crdt.CvRDT
	vectorClock crdt.VectorClock
	logger     *zap.Logger
}

// ConflictResolver handles conflicts in hybrid mode
type ConflictResolver struct {
	mu       sync.RWMutex
	strategy ResolveStrategy
	history  []ConflictEvent
	logger   *zap.Logger
}

// RaftNode interface for Raft consensus integration
type RaftNode interface {
	Propose(ctx context.Context, data []byte) error
	ReadIndex(ctx context.Context) (uint64, error)
	IsLeader() bool
}

// ResolveStrategy defines conflict resolution strategy
type ResolveStrategy int

const (
	ResolveLastWriteWins ResolveStrategy = iota
	ResolveMerge
	ResolveCustom
)

// ConflictEvent records a conflict resolution event
type ConflictEvent struct {
	Timestamp time.Time
	Key       string
	Strategy  ResolveStrategy
	Resolved  bool
	Latency   time.Duration
}

// NewASSv3 creates a new ASS v3 engine
func NewASSv3(nodeID string, mode upgrade.NetworkMode, logger *zap.Logger) (*ASSv3, error) {
	ctx, cancel := context.WithCancel(context.Background())

	ass := &ASSv3{
		nodeID:           nodeID,
		mode:             mode,
		conflictResolver: NewConflictResolver(ResolveLastWriteWins, logger),
		modeDetector:     upgrade.NewModeDetector(),
		syncCount:        0,
		lastSyncTime:     time.Now(),
		logger:           logger,
		ctx:              ctx,
		cancel:           cancel,
	}

	// Initialize mode-specific synchronizers
	ass.raftSync = &RaftStateSync{
		nodeID: nodeID,
		logger: logger,
	}

	ass.crdtSync = &CRDTStateSync{
		nodeID:      nodeID,
		crdtStore:   make(map[string]crdt.CvRDT),
		vectorClock: make(crdt.VectorClock),
		logger:      logger,
	}

	return ass, nil
}

// Start begins ASS v3 operation
func (a *ASSv3) Start() error {
	a.logger.Info("Starting ASS v3",
		zap.String("node_id", a.nodeID),
		zap.String("mode", a.mode.String()))

	// Start mode detection loop for hybrid mode
	if a.mode == upgrade.ModeHybrid {
		go a.adaptiveModeLoop()
	}

	return nil
}

// Stop halts ASS v3 operation
func (a *ASSv3) Stop() error {
	a.logger.Info("Stopping ASS v3", zap.String("node_id", a.nodeID))
	a.cancel()
	return nil
}

// SyncState synchronizes state using mode-appropriate strategy
func (a *ASSv3) SyncState(ctx context.Context, state interface{}) error {
	startTime := time.Now()
	defer func() {
		a.mu.Lock()
		a.syncCount++
		a.lastSyncTime = time.Now()
		latency := time.Since(startTime)
		// Update average latency (exponential moving average)
		if a.avgSyncLatency == 0 {
			a.avgSyncLatency = latency
		} else {
			a.avgSyncLatency = (a.avgSyncLatency*9 + latency) / 10
		}
		a.mu.Unlock()
	}()

	a.mu.RLock()
	mode := a.mode
	a.mu.RUnlock()

	switch mode {
	case upgrade.ModeDatacenter:
		// Use Raft for strong consistency (<100ms target)
		return a.raftSync.Sync(ctx, state)

	case upgrade.ModeInternet:
		// Use CRDT for eventual consistency (5-30 seconds acceptable)
		return a.crdtSync.Sync(ctx, state)

	case upgrade.ModeHybrid:
		// Adaptive: Try Raft first, fallback to CRDT
		return a.hybridSync(ctx, state)

	default:
		return fmt.Errorf("unknown network mode: %v", mode)
	}
}

// Sync implements strong consistency synchronization via Raft
func (rs *RaftStateSync) Sync(ctx context.Context, state interface{}) error {
	if rs.raftNode == nil {
		return fmt.Errorf("Raft node not initialized")
	}

	// Serialize state
	data, err := serializeState(state)
	if err != nil {
		return fmt.Errorf("failed to serialize state: %w", err)
	}

	// Propose to Raft cluster
	if err := rs.raftNode.Propose(ctx, data); err != nil {
		return fmt.Errorf("Raft proposal failed: %w", err)
	}

	rs.logger.Debug("Raft state synchronized",
		zap.String("node_id", rs.nodeID),
		zap.Int("bytes", len(data)))

	return nil
}

// Sync implements eventual consistency synchronization via CRDT
func (cs *CRDTStateSync) Sync(ctx context.Context, state interface{}) error {
	cs.mu.Lock()
	defer cs.mu.Unlock()

	// Convert state to CRDT
	key := "state"

	// Increment vector clock
	cs.vectorClock[cs.nodeID]++

	// Create or update CRDT
	crdtValue, err := stateToCRDT(cs.nodeID, state)
	if err != nil {
		return fmt.Errorf("failed to convert to CRDT: %w", err)
	}

	// Merge with existing CRDT if present
	if existing, exists := cs.crdtStore[key]; exists {
		if err := existing.Merge(crdtValue); err != nil {
			return fmt.Errorf("CRDT merge failed: %w", err)
		}
		cs.crdtStore[key] = existing
	} else {
		cs.crdtStore[key] = crdtValue
	}

	cs.logger.Debug("CRDT state synchronized",
		zap.String("node_id", cs.nodeID),
		zap.String("vector_clock", fmt.Sprintf("%v", cs.vectorClock)))

	return nil
}

// hybridSync uses adaptive strategy for hybrid mode
func (a *ASSv3) hybridSync(ctx context.Context, state interface{}) error {
	startTime := time.Now()

	// Detect current network conditions
	detectedMode := a.modeDetector.DetectMode(ctx)

	a.logger.Debug("Hybrid sync mode detected",
		zap.String("mode", detectedMode.String()))

	// Create context with timeout based on mode
	var syncCtx context.Context
	var cancel context.CancelFunc

	switch detectedMode {
	case upgrade.ModeDatacenter:
		// Short timeout for datacenter
		syncCtx, cancel = context.WithTimeout(ctx, 100*time.Millisecond)
	case upgrade.ModeInternet:
		// Longer timeout for internet
		syncCtx, cancel = context.WithTimeout(ctx, 5*time.Second)
	default:
		syncCtx, cancel = context.WithTimeout(ctx, 1*time.Second)
	}
	defer cancel()

	// Try Raft first if conditions look good
	if detectedMode == upgrade.ModeDatacenter {
		if err := a.raftSync.Sync(syncCtx, state); err == nil {
			return nil
		} else {
			a.logger.Warn("Raft sync failed, falling back to CRDT", zap.Error(err))
		}
	}

	// Fallback to CRDT
	if err := a.crdtSync.Sync(syncCtx, state); err != nil {
		return fmt.Errorf("both Raft and CRDT sync failed: %w", err)
	}

	// Record conflict for learning
	a.conflictResolver.RecordConflict(ConflictEvent{
		Timestamp: time.Now(),
		Key:       "state",
		Strategy:  a.conflictResolver.strategy,
		Resolved:  true,
		Latency:   time.Since(startTime),
	})

	return nil
}

// adaptiveModeLoop continuously adapts to network conditions
func (a *ASSv3) adaptiveModeLoop() {
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
				a.logger.Info("Network mode changed",
					zap.String("old_mode", a.mode.String()),
					zap.String("new_mode", newMode.String()))
				a.mode = newMode
			}
			a.mu.Unlock()
		}
	}
}

// SetRaftNode sets the Raft node for strong consistency mode
func (a *ASSv3) SetRaftNode(node RaftNode) {
	a.raftSync.mu.Lock()
	defer a.raftSync.mu.Unlock()
	a.raftSync.raftNode = node
}

// SetMode manually sets the synchronization mode
func (a *ASSv3) SetMode(mode upgrade.NetworkMode) {
	a.mu.Lock()
	defer a.mu.Unlock()

	a.logger.Info("Mode changed",
		zap.String("old_mode", a.mode.String()),
		zap.String("new_mode", mode.String()))

	a.mode = mode
}

// GetMode returns the current synchronization mode
func (a *ASSv3) GetMode() upgrade.NetworkMode {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.mode
}

// GetMetrics returns ASS v3 performance metrics
func (a *ASSv3) GetMetrics() *ASSv3Metrics {
	a.mu.RLock()
	defer a.mu.RUnlock()

	return &ASSv3Metrics{
		Mode:           a.mode.String(),
		SyncCount:      a.syncCount,
		LastSyncTime:   a.lastSyncTime,
		AvgSyncLatency: a.avgSyncLatency,
		ConflictCount:  len(a.conflictResolver.history),
	}
}

// ASSv3Metrics contains performance statistics
type ASSv3Metrics struct {
	Mode           string        `json:"mode"`
	SyncCount      int64         `json:"sync_count"`
	LastSyncTime   time.Time     `json:"last_sync_time"`
	AvgSyncLatency time.Duration `json:"avg_sync_latency"`
	ConflictCount  int           `json:"conflict_count"`
}

// NewConflictResolver creates a new conflict resolver
func NewConflictResolver(strategy ResolveStrategy, logger *zap.Logger) *ConflictResolver {
	return &ConflictResolver{
		strategy: strategy,
		history:  make([]ConflictEvent, 0),
		logger:   logger,
	}
}

// RecordConflict records a conflict event
func (cr *ConflictResolver) RecordConflict(event ConflictEvent) {
	cr.mu.Lock()
	defer cr.mu.Unlock()

	cr.history = append(cr.history, event)

	// Limit history size
	if len(cr.history) > 1000 {
		cr.history = cr.history[1:]
	}
}

// Helper functions

func serializeState(state interface{}) ([]byte, error) {
	// TODO: Implement proper serialization
	// For now, use simple byte conversion
	return []byte(fmt.Sprintf("%v", state)), nil
}

func stateToCRDT(nodeID string, state interface{}) (crdt.CvRDT, error) {
	// TODO: Implement proper state to CRDT conversion
	// For now, use LWW register
	register := crdt.NewLWWRegister(nodeID)
	data, err := serializeState(state)
	if err != nil {
		return nil, err
	}
	register.Set(data)
	return register, nil
}
