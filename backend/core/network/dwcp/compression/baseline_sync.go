package compression

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"go.uber.org/zap"
)

// BaselineSynchronizer manages baseline state synchronization across cluster nodes
type BaselineSynchronizer struct {
	// Local baselines
	localBaselines  map[string]*BaselineState
	baselineMutex   sync.RWMutex

	// Remote node tracking
	remoteNodes     map[string]*RemoteNode
	nodesMutex      sync.RWMutex

	// Configuration
	config          *BaselineSyncConfig
	logger          *zap.Logger

	// Synchronization
	syncTicker      *time.Ticker
	stopChan        chan struct{}
}

// RemoteNode represents a remote cluster node
type RemoteNode struct {
	NodeID        string
	Address       string
	LastSync      time.Time
	BaselineCount int
	Status        NodeStatus
}

// NodeStatus represents the status of a remote node
type NodeStatus string

const (
	NodeStatusOnline  NodeStatus = "online"
	NodeStatusOffline NodeStatus = "offline"
	NodeStatusSyncing NodeStatus = "syncing"
)

// BaselineSyncConfig configuration for baseline synchronization
type BaselineSyncConfig struct {
	Enabled            bool          `json:"enabled" yaml:"enabled"`
	SyncInterval       time.Duration `json:"sync_interval" yaml:"sync_interval"`
	MaxStaleness       time.Duration `json:"max_staleness" yaml:"max_staleness"`
	EnableVersioning   bool          `json:"enable_versioning" yaml:"enable_versioning"`
	ConflictResolution string        `json:"conflict_resolution" yaml:"conflict_resolution"` // "lww", "newest", "manual"
}

// BaselineVersion represents a versioned baseline for conflict resolution
type BaselineVersion struct {
	Version   int       `json:"version"`
	Data      []byte    `json:"data"`
	Timestamp time.Time `json:"timestamp"`
	NodeID    string    `json:"node_id"`
}

// DefaultBaselineSyncConfig returns sensible defaults
func DefaultBaselineSyncConfig() *BaselineSyncConfig {
	return &BaselineSyncConfig{
		Enabled:            false, // Disabled by default until cluster is configured
		SyncInterval:       5 * time.Second,
		MaxStaleness:       30 * time.Second,
		EnableVersioning:   true,
		ConflictResolution: "lww", // Last-Write-Wins
	}
}

// NewBaselineSynchronizer creates a new baseline synchronizer
func NewBaselineSynchronizer(config *BaselineSyncConfig, logger *zap.Logger) *BaselineSynchronizer {
	if config == nil {
		config = DefaultBaselineSyncConfig()
	}

	if logger == nil {
		logger, _ = zap.NewProduction()
	}

	bs := &BaselineSynchronizer{
		localBaselines: make(map[string]*BaselineState),
		remoteNodes:    make(map[string]*RemoteNode),
		config:         config,
		logger:         logger,
		stopChan:       make(chan struct{}),
	}

	// Start sync scheduler if enabled
	if config.Enabled && config.SyncInterval > 0 {
		bs.startSyncScheduler()
	}

	return bs
}

// RegisterNode registers a remote node for baseline synchronization
func (bs *BaselineSynchronizer) RegisterNode(nodeID, address string) error {
	bs.nodesMutex.Lock()
	defer bs.nodesMutex.Unlock()

	bs.remoteNodes[nodeID] = &RemoteNode{
		NodeID:        nodeID,
		Address:       address,
		LastSync:      time.Time{},
		BaselineCount: 0,
		Status:        NodeStatusOffline,
	}

	bs.logger.Info("Registered remote node",
		zap.String("node_id", nodeID),
		zap.String("address", address))

	return nil
}

// UnregisterNode removes a remote node
func (bs *BaselineSynchronizer) UnregisterNode(nodeID string) {
	bs.nodesMutex.Lock()
	defer bs.nodesMutex.Unlock()

	delete(bs.remoteNodes, nodeID)

	bs.logger.Info("Unregistered remote node", zap.String("node_id", nodeID))
}

// SetBaseline stores a baseline locally
func (bs *BaselineSynchronizer) SetBaseline(key string, baseline *BaselineState) {
	bs.baselineMutex.Lock()
	defer bs.baselineMutex.Unlock()

	bs.localBaselines[key] = baseline

	// Trigger async sync if enabled
	if bs.config.Enabled {
		go bs.syncBaseline(key, baseline)
	}
}

// GetBaseline retrieves a baseline locally
func (bs *BaselineSynchronizer) GetBaseline(key string) (*BaselineState, bool) {
	bs.baselineMutex.RLock()
	defer bs.baselineMutex.RUnlock()

	baseline, exists := bs.localBaselines[key]
	return baseline, exists
}

// DeleteBaseline removes a baseline
func (bs *BaselineSynchronizer) DeleteBaseline(key string) {
	bs.baselineMutex.Lock()
	defer bs.baselineMutex.Unlock()

	delete(bs.localBaselines, key)

	bs.logger.Debug("Deleted baseline", zap.String("key", key))
}

// SyncWithCluster synchronizes all baselines with cluster nodes
func (bs *BaselineSynchronizer) SyncWithCluster(ctx context.Context) error {
	if !bs.config.Enabled {
		return nil
	}

	bs.nodesMutex.RLock()
	nodes := make([]*RemoteNode, 0, len(bs.remoteNodes))
	for _, node := range bs.remoteNodes {
		nodes = append(nodes, node)
	}
	bs.nodesMutex.RUnlock()

	// Sync with each node
	for _, node := range nodes {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
			if err := bs.syncWithNode(ctx, node); err != nil {
				bs.logger.Warn("Failed to sync with node",
					zap.String("node_id", node.NodeID),
					zap.Error(err))
			}
		}
	}

	return nil
}

// syncWithNode synchronizes baselines with a specific node
func (bs *BaselineSynchronizer) syncWithNode(ctx context.Context, node *RemoteNode) error {
	// Update node status
	bs.nodesMutex.Lock()
	node.Status = NodeStatusSyncing
	bs.nodesMutex.Unlock()

	// TODO: Implement actual network sync using DWCP transport
	// For now, this is a placeholder for the sync protocol

	// Simulate sync delay
	select {
	case <-time.After(10 * time.Millisecond):
	case <-ctx.Done():
		return ctx.Err()
	}

	// Update node status
	bs.nodesMutex.Lock()
	node.Status = NodeStatusOnline
	node.LastSync = time.Now()
	bs.nodesMutex.Unlock()

	return nil
}

// syncBaseline synchronizes a single baseline with cluster
func (bs *BaselineSynchronizer) syncBaseline(key string, baseline *BaselineState) {
	// TODO: Implement async baseline push to cluster nodes
	// This would use DWCP transport to send baseline updates

	bs.logger.Debug("Syncing baseline",
		zap.String("key", key),
		zap.Int("size", len(baseline.Data)))
}

// ResolveConflict resolves baseline conflicts using configured strategy
func (bs *BaselineSynchronizer) ResolveConflict(local, remote *BaselineState) *BaselineState {
	switch bs.config.ConflictResolution {
	case "lww": // Last-Write-Wins
		if remote.Timestamp.After(local.Timestamp) {
			return remote
		}
		return local

	case "newest":
		if remote.Timestamp.After(local.Timestamp) {
			return remote
		}
		return local

	case "manual":
		// In manual mode, keep local and log conflict
		bs.logger.Warn("Baseline conflict detected - keeping local",
			zap.Time("local_ts", local.Timestamp),
			zap.Time("remote_ts", remote.Timestamp))
		return local

	default:
		return local
	}
}

// MigrateBaseline migrates a baseline to a new version format
func (bs *BaselineSynchronizer) MigrateBaseline(key string, oldVersion, newVersion int) error {
	bs.baselineMutex.Lock()
	defer bs.baselineMutex.Unlock()

	baseline, exists := bs.localBaselines[key]
	if !exists {
		return fmt.Errorf("baseline not found: %s", key)
	}

	// TODO: Implement actual version migration logic
	// For now, just update timestamp
	baseline.Timestamp = time.Now()

	bs.logger.Info("Migrated baseline",
		zap.String("key", key),
		zap.Int("old_version", oldVersion),
		zap.Int("new_version", newVersion))

	return nil
}

// ExportBaselines exports all baselines for backup
func (bs *BaselineSynchronizer) ExportBaselines() ([]byte, error) {
	bs.baselineMutex.RLock()
	defer bs.baselineMutex.RUnlock()

	export := make(map[string]*BaselineState)
	for key, baseline := range bs.localBaselines {
		export[key] = baseline
	}

	data, err := json.Marshal(export)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal baselines: %w", err)
	}

	return data, nil
}

// ImportBaselines imports baselines from backup
func (bs *BaselineSynchronizer) ImportBaselines(data []byte) error {
	var imported map[string]*BaselineState
	if err := json.Unmarshal(data, &imported); err != nil {
		return fmt.Errorf("failed to unmarshal baselines: %w", err)
	}

	bs.baselineMutex.Lock()
	defer bs.baselineMutex.Unlock()

	for key, baseline := range imported {
		bs.localBaselines[key] = baseline
	}

	bs.logger.Info("Imported baselines", zap.Int("count", len(imported)))

	return nil
}

// CleanupDeletedVMs removes baselines for VMs that no longer exist
func (bs *BaselineSynchronizer) CleanupDeletedVMs(activeVMIDs []string) int {
	activeSet := make(map[string]bool)
	for _, vmID := range activeVMIDs {
		activeSet[vmID] = true
	}

	bs.baselineMutex.Lock()
	defer bs.baselineMutex.Unlock()

	deleted := 0
	for key := range bs.localBaselines {
		// Extract VM ID from key (format: "vm-{id}-{resource}")
		// This is a simple check - production would parse properly
		isActive := false
		for vmID := range activeSet {
			if len(key) > len(vmID) && key[:len(vmID)] == vmID {
				isActive = true
				break
			}
		}

		if !isActive {
			delete(bs.localBaselines, key)
			deleted++
		}
	}

	if deleted > 0 {
		bs.logger.Info("Cleaned up baselines for deleted VMs", zap.Int("count", deleted))
	}

	return deleted
}

// startSyncScheduler starts the periodic sync scheduler
func (bs *BaselineSynchronizer) startSyncScheduler() {
	bs.syncTicker = time.NewTicker(bs.config.SyncInterval)

	go func() {
		for {
			select {
			case <-bs.syncTicker.C:
				ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
				if err := bs.SyncWithCluster(ctx); err != nil {
					bs.logger.Error("Cluster sync failed", zap.Error(err))
				}
				cancel()

			case <-bs.stopChan:
				bs.syncTicker.Stop()
				return
			}
		}
	}()

	bs.logger.Info("Baseline sync scheduler started",
		zap.Duration("interval", bs.config.SyncInterval))
}

// GetStats returns synchronization statistics
func (bs *BaselineSynchronizer) GetStats() map[string]interface{} {
	bs.baselineMutex.RLock()
	localCount := len(bs.localBaselines)
	bs.baselineMutex.RUnlock()

	bs.nodesMutex.RLock()
	nodeCount := len(bs.remoteNodes)
	onlineNodes := 0
	for _, node := range bs.remoteNodes {
		if node.Status == NodeStatusOnline {
			onlineNodes++
		}
	}
	bs.nodesMutex.RUnlock()

	return map[string]interface{}{
		"enabled":         bs.config.Enabled,
		"local_baselines": localCount,
		"remote_nodes":    nodeCount,
		"online_nodes":    onlineNodes,
		"sync_interval":   bs.config.SyncInterval,
	}
}

// Close stops the sync scheduler and releases resources
func (bs *BaselineSynchronizer) Close() error {
	if bs.syncTicker != nil {
		close(bs.stopChan)
	}
	return nil
}
