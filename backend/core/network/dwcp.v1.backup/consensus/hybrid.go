package consensus

import (
	"fmt"
	"sync"
)

// HybridConsensus implements a hybrid consensus approach that uses
// Raft for intra-region and eventual consistency for cross-region
type HybridConsensus struct {
	mu sync.RWMutex

	nodeID      string
	localRegion string

	// Intra-region: Strong consistency with Raft
	intraRegion *RaftConsensus

	// Cross-region: Eventual consistency
	crossRegion *EventualConsistency

	// Region management
	regions    map[string]*RegionInfo
	keyRegions map[string]string // key -> region mapping

	stateMachine StateMachine
}

// RegionInfo contains information about a region
type RegionInfo struct {
	RegionID  string
	Nodes     []string
	Leader    string
	Healthy   bool
	Latency   int64 // milliseconds
}

// NewHybridConsensus creates a new hybrid consensus instance
func NewHybridConsensus(nodeID, localRegion string, raft *RaftConsensus, eventual *EventualConsistency) *HybridConsensus {
	return &HybridConsensus{
		nodeID:       nodeID,
		localRegion:  localRegion,
		intraRegion:  raft,
		crossRegion:  eventual,
		regions:      make(map[string]*RegionInfo),
		keyRegions:   make(map[string]string),
		stateMachine: raft.stateMachine,
	}
}

// Propose proposes a value using the appropriate consensus mechanism
func (hc *HybridConsensus) Propose(key string, value []byte) error {
	region := hc.getKeyRegion(key)

	hc.mu.RLock()
	isLocal := region == hc.localRegion
	hc.mu.RUnlock()

	if isLocal {
		// Local region: use Raft for strong consistency
		return hc.intraRegion.Propose(key, value)
	}

	// Cross-region: use eventual consistency
	return hc.crossRegion.Update(key, value)
}

// Get retrieves a value
func (hc *HybridConsensus) Get(key string) ([]byte, error) {
	region := hc.getKeyRegion(key)

	hc.mu.RLock()
	isLocal := region == hc.localRegion
	hc.mu.RUnlock()

	if isLocal {
		// Read from local Raft state machine
		snapshot, err := hc.stateMachine.Snapshot()
		if err != nil {
			return nil, err
		}

		if val, ok := snapshot.Data[key]; ok {
			return val, nil
		}
		return nil, fmt.Errorf("key not found")
	}

	// Read from eventual consistency store
	val, ok := hc.crossRegion.Get(key)
	if !ok {
		return nil, fmt.Errorf("key not found")
	}
	return val, nil
}

// getKeyRegion determines which region owns a key
func (hc *HybridConsensus) getKeyRegion(key string) string {
	hc.mu.RLock()
	defer hc.mu.RUnlock()

	// Check explicit mapping
	if region, ok := hc.keyRegions[key]; ok {
		return region
	}

	// Default to local region
	return hc.localRegion
}

// SetKeyRegion explicitly sets the region for a key
func (hc *HybridConsensus) SetKeyRegion(key, region string) {
	hc.mu.Lock()
	defer hc.mu.Unlock()

	hc.keyRegions[key] = region
}

// AddRegion registers a new region
func (hc *HybridConsensus) AddRegion(regionID string, nodes []string) {
	hc.mu.Lock()
	defer hc.mu.Unlock()

	hc.regions[regionID] = &RegionInfo{
		RegionID: regionID,
		Nodes:    nodes,
		Healthy:  true,
	}
}

// UpdateRegionHealth updates the health status of a region
func (hc *HybridConsensus) UpdateRegionHealth(regionID string, healthy bool, latency int64) {
	hc.mu.Lock()
	defer hc.mu.Unlock()

	if region, ok := hc.regions[regionID]; ok {
		region.Healthy = healthy
		region.Latency = latency
	}
}

// GetRegionInfo returns information about a region
func (hc *HybridConsensus) GetRegionInfo(regionID string) (*RegionInfo, error) {
	hc.mu.RLock()
	defer hc.mu.RUnlock()

	region, ok := hc.regions[regionID]
	if !ok {
		return nil, fmt.Errorf("region not found: %s", regionID)
	}

	return region, nil
}

// SyncCrossRegion synchronizes state with other regions
func (hc *HybridConsensus) SyncCrossRegion() error {
	hc.mu.RLock()
	regions := make([]*RegionInfo, 0, len(hc.regions))
	for _, region := range hc.regions {
		if region.RegionID != hc.localRegion && region.Healthy {
			regions = append(regions, region)
		}
	}
	hc.mu.RUnlock()

	// Perform gossip with each remote region
	for _, region := range regions {
		if err := hc.syncWithRegion(region); err != nil {
			// Log error but continue with other regions
			fmt.Printf("Failed to sync with region %s: %v\n", region.RegionID, err)
		}
	}

	return nil
}

// syncWithRegion synchronizes with a specific region
func (hc *HybridConsensus) syncWithRegion(region *RegionInfo) error {
	// Get local state snapshot
	snapshot, err := hc.stateMachine.Snapshot()
	if err != nil {
		return err
	}

	// In real implementation, would:
	// 1. Send snapshot to region leader
	// 2. Receive region's snapshot
	// 3. Merge using CRDT semantics

	// For now, merge into eventual consistency store
	for key, value := range snapshot.Data {
		if hc.getKeyRegion(key) != hc.localRegion {
			hc.crossRegion.Update(key, value)
		}
	}

	return nil
}

// LoadSnapshot loads a snapshot into hybrid consensus
func (hc *HybridConsensus) LoadSnapshot(snapshot *Snapshot) error {
	// Load into both Raft and eventual consistency
	if err := hc.intraRegion.LoadSnapshot(snapshot); err != nil {
		return err
	}

	return hc.crossRegion.LoadSnapshot(snapshot)
}

// MigrateKey migrates a key from one region to another
func (hc *HybridConsensus) MigrateKey(key, targetRegion string) error {
	hc.mu.Lock()
	currentRegion := hc.getKeyRegion(key)
	hc.mu.Unlock()

	if currentRegion == targetRegion {
		return nil // Already in target region
	}

	// Get current value
	value, err := hc.Get(key)
	if err != nil {
		return fmt.Errorf("failed to get key for migration: %w", err)
	}

	// Update region mapping
	hc.SetKeyRegion(key, targetRegion)

	// Write to new region
	if targetRegion == hc.localRegion {
		return hc.intraRegion.Propose(key, value)
	}

	return hc.crossRegion.Update(key, value)
}

// GetRegionLoad returns the current load on each region
func (hc *HybridConsensus) GetRegionLoad() map[string]int {
	hc.mu.RLock()
	defer hc.mu.RUnlock()

	load := make(map[string]int)
	for key, region := range hc.keyRegions {
		load[region]++
		_ = key // Use key to avoid unused warning
	}

	return load
}

// RebalanceKeys rebalances keys across regions based on load
func (hc *HybridConsensus) RebalanceKeys(targetRegion string, count int) error {
	// Find keys to migrate
	keysToMigrate := make([]string, 0, count)

	hc.mu.RLock()
	for key, region := range hc.keyRegions {
		if region != targetRegion && len(keysToMigrate) < count {
			keysToMigrate = append(keysToMigrate, key)
		}
	}
	hc.mu.RUnlock()

	// Migrate keys
	for _, key := range keysToMigrate {
		if err := hc.MigrateKey(key, targetRegion); err != nil {
			return fmt.Errorf("failed to migrate key %s: %w", key, err)
		}
	}

	return nil
}
