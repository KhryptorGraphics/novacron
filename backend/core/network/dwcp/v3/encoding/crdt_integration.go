package encoding

import (
	"bytes"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/sync/crdt"
)

// CRDTIntegration provides conflict-free state synchronization for HDE
type CRDTIntegration struct {
	mu sync.RWMutex

	// CRDT state for baseline metadata
	baselineMetadata *crdt.ORMap // Observed-Remove Map for baseline info

	// Vector clock for causal ordering
	vectorClock crdt.VectorClock

	// Node identifier
	nodeID string

	// Synchronization state
	syncStates map[string]*SyncState
}

// SyncState represents synchronization state for a baseline
type SyncState struct {
	BaselineID   string
	Version      uint64
	VectorClock  crdt.VectorClock
	LastSync     time.Time
	ConflictFree bool
}

// BaselineMetadata represents metadata stored in CRDT
type BaselineMetadata struct {
	ID        string           `json:"id"`
	NodeID    string           `json:"node_id"`
	Version   uint64           `json:"version"`
	Hash      []byte           `json:"hash"`
	Timestamp time.Time        `json:"timestamp"`
	Size      int64            `json:"size"`
	Clock     crdt.VectorClock `json:"clock"`
}

// NewCRDTIntegration creates a new CRDT integration instance
func NewCRDTIntegration(nodeID string) *CRDTIntegration {
	ci := &CRDTIntegration{
		nodeID:           nodeID,
		baselineMetadata: crdt.NewORMap(nodeID),
		vectorClock:      make(crdt.VectorClock),
		syncStates:       make(map[string]*SyncState),
	}

	// Initialize vector clock for this node
	ci.vectorClock[nodeID] = 0

	return ci
}

// RegisterBaseline registers a baseline with CRDT for conflict-free sync
func (ci *CRDTIntegration) RegisterBaseline(id string, hash []byte, size int64) error {
	ci.mu.Lock()
	defer ci.mu.Unlock()

	// Increment vector clock
	ci.vectorClock.Increment(ci.nodeID)

	// Create metadata
	metadata := &BaselineMetadata{
		ID:        id,
		NodeID:    ci.nodeID,
		Version:   ci.vectorClock[ci.nodeID],
		Hash:      hash,
		Timestamp: time.Now(),
		Size:      size,
		Clock:     ci.vectorClock.Clone(),
	}

	// Store in CRDT map
	metadataBytes, err := json.Marshal(metadata)
	if err != nil {
		return fmt.Errorf("failed to marshal metadata: %w", err)
	}

	ci.baselineMetadata.SetLWW(id, string(metadataBytes))

	// Update sync state
	ci.syncStates[id] = &SyncState{
		BaselineID:   id,
		Version:      metadata.Version,
		VectorClock:  ci.vectorClock.Clone(),
		LastSync:     time.Now(),
		ConflictFree: true,
	}

	return nil
}

// GetBaseline retrieves baseline metadata from CRDT
func (ci *CRDTIntegration) GetBaseline(id string) (*BaselineMetadata, error) {
	ci.mu.RLock()
	defer ci.mu.RUnlock()

	value, exists := ci.baselineMetadata.GetLWW(id)
	if !exists {
		return nil, fmt.Errorf("baseline not found: %s", id)
	}

	var metadata BaselineMetadata
	if err := json.Unmarshal([]byte(value.(string)), &metadata); err != nil {
		return nil, fmt.Errorf("failed to unmarshal metadata: %w", err)
	}

	return &metadata, nil
}

// MergeRemoteState merges remote CRDT state for conflict resolution
func (ci *CRDTIntegration) MergeRemoteState(remoteData []byte) error {
	ci.mu.Lock()
	defer ci.mu.Unlock()

	// Deserialize remote CRDT state
	remoteCRDT := crdt.NewORMap(ci.nodeID)
	if err := remoteCRDT.Unmarshal(remoteData); err != nil {
		return fmt.Errorf("failed to unmarshal remote CRDT: %w", err)
	}

	// Merge with local CRDT
	if err := ci.baselineMetadata.Merge(remoteCRDT); err != nil {
		return fmt.Errorf("failed to merge CRDT: %w", err)
	}

	// Update vector clock
	remoteMetadata := ci.extractVectorClockFromRemote(remoteCRDT)
	ci.vectorClock.Update(remoteMetadata)

	return nil
}

// DetectConflict checks if there's a conflict for a baseline
func (ci *CRDTIntegration) DetectConflict(id string, remoteMetadata *BaselineMetadata) (bool, error) {
	ci.mu.RLock()
	defer ci.mu.RUnlock()

	localMetadata, err := ci.GetBaseline(id)
	if err != nil {
		// No local baseline, no conflict
		return false, nil
	}

	// Compare vector clocks for causality
	ordering := localMetadata.Clock.Compare(remoteMetadata.Clock)

	switch ordering {
	case crdt.OrderingBefore:
		// Local is older, no conflict
		return false, nil
	case crdt.OrderingAfter:
		// Local is newer, no conflict
		return false, nil
	case crdt.OrderingEqual:
		// Same version, check hash
		if !bytes.Equal(localMetadata.Hash, remoteMetadata.Hash) {
			// Same version but different content - conflict!
			return true, nil
		}
		return false, nil
	case crdt.OrderingConcurrent:
		// Concurrent updates - conflict!
		return true, nil
	default:
		return false, fmt.Errorf("unknown ordering: %v", ordering)
	}
}

// ResolveConflict resolves conflicts using CRDT semantics (LWW)
func (ci *CRDTIntegration) ResolveConflict(
	id string,
	localMetadata *BaselineMetadata,
	remoteMetadata *BaselineMetadata,
) (*BaselineMetadata, error) {
	// Last-Writer-Wins strategy
	// Choose the baseline with higher timestamp
	if remoteMetadata.Timestamp.After(localMetadata.Timestamp) {
		return remoteMetadata, nil
	}

	// If timestamps equal, use nodeID as tiebreaker (deterministic)
	if remoteMetadata.Timestamp.Equal(localMetadata.Timestamp) {
		if remoteMetadata.NodeID > localMetadata.NodeID {
			return remoteMetadata, nil
		}
	}

	return localMetadata, nil
}

// ExportState exports CRDT state for synchronization
func (ci *CRDTIntegration) ExportState() ([]byte, error) {
	ci.mu.RLock()
	defer ci.mu.RUnlock()

	return ci.baselineMetadata.Marshal()
}

// GetDigest returns a compact digest for anti-entropy sync
func (ci *CRDTIntegration) GetDigest() (*crdt.Digest, error) {
	ci.mu.RLock()
	defer ci.mu.RUnlock()

	digest := &crdt.Digest{
		NodeID:      ci.nodeID,
		VectorClock: ci.vectorClock.Clone(),
		Checksums:   make(map[string]string),
		Timestamp:   time.Now(),
	}

	// Create checksums for all baselines
	keys := ci.baselineMetadata.Keys()
	for _, key := range keys {
		value, exists := ci.baselineMetadata.GetLWW(key)
		if exists {
			// Simple hash of metadata
			hash := fmt.Sprintf("%x", hashString(value.(string)))
			digest.Checksums[key] = hash
		}
	}

	return digest, nil
}

// CompareDig compares digests to find differences
func (ci *CRDTIntegration) CompareDigest(remoteDigest *crdt.Digest) (*crdt.Delta, error) {
	ci.mu.RLock()
	defer ci.mu.RUnlock()

	delta := &crdt.Delta{
		Missing: make([]string, 0),
		Theirs:  make([]string, 0),
		Stale:   make([]string, 0),
	}

	localDigest, err := ci.GetDigest()
	if err != nil {
		return nil, err
	}

	// Find keys we're missing
	for key, remoteHash := range remoteDigest.Checksums {
		localHash, exists := localDigest.Checksums[key]
		if !exists {
			delta.Missing = append(delta.Missing, key)
		} else if localHash != remoteHash {
			// Different versions - check vector clock
			ordering := localDigest.VectorClock.Compare(remoteDigest.VectorClock)
			if ordering == crdt.OrderingBefore {
				delta.Stale = append(delta.Stale, key)
			}
		}
	}

	// Find keys they're missing
	for key := range localDigest.Checksums {
		if _, exists := remoteDigest.Checksums[key]; !exists {
			delta.Theirs = append(delta.Theirs, key)
		}
	}

	return delta, nil
}

// GetSyncState returns synchronization state for a baseline
func (ci *CRDTIntegration) GetSyncState(id string) (*SyncState, bool) {
	ci.mu.RLock()
	defer ci.mu.RUnlock()

	state, exists := ci.syncStates[id]
	return state, exists
}

// UpdateSyncState updates synchronization state
func (ci *CRDTIntegration) UpdateSyncState(id string, conflictFree bool) {
	ci.mu.Lock()
	defer ci.mu.Unlock()

	if state, exists := ci.syncStates[id]; exists {
		state.LastSync = time.Now()
		state.ConflictFree = conflictFree
		state.VectorClock = ci.vectorClock.Clone()
	}
}

// GetVectorClock returns current vector clock
func (ci *CRDTIntegration) GetVectorClock() crdt.VectorClock {
	ci.mu.RLock()
	defer ci.mu.RUnlock()
	return ci.vectorClock.Clone()
}

// Helper functions

func (ci *CRDTIntegration) extractVectorClockFromRemote(remoteCRDT crdt.CvRDT) crdt.VectorClock {
	// Extract vector clock from remote CRDT metadata
	// This is a simplified implementation
	clock := make(crdt.VectorClock)

	// In a full implementation, we'd extract clocks from all metadata entries
	// For now, return empty clock (will be updated by merge)
	return clock
}

func hashString(s string) []byte {
	// Simple hash function for checksums
	h := uint64(5381)
	for _, c := range s {
		h = ((h << 5) + h) + uint64(c)
	}

	result := make([]byte, 8)
	for i := 0; i < 8; i++ {
		result[i] = byte(h >> (i * 8))
	}
	return result
}

// GetStats returns CRDT integration statistics
func (ci *CRDTIntegration) GetStats() map[string]interface{} {
	ci.mu.RLock()
	defer ci.mu.RUnlock()

	conflictFreeCount := 0
	for _, state := range ci.syncStates {
		if state.ConflictFree {
			conflictFreeCount++
		}
	}

	return map[string]interface{}{
		"node_id":             ci.nodeID,
		"baseline_count":      len(ci.syncStates),
		"conflict_free_count": conflictFreeCount,
		"vector_clock_size":   len(ci.vectorClock),
		"crdt_keys":           len(ci.baselineMetadata.Keys()),
	}
}
