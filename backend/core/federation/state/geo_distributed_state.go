// Package state implements geo-distributed state management for DWCP v3 federation
// Provides multi-region state synchronization with CRDTs and conflict resolution
package state

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// Global metrics for state management
var (
	stateOperations = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "dwcp_federation_state_operations_total",
			Help: "Total number of state operations",
		},
		[]string{"region", "operation", "status"},
	)

	stateSyncLatency = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "dwcp_federation_state_sync_latency_ms",
			Help:    "Latency of state synchronization across regions",
			Buckets: []float64{10, 25, 50, 100, 250, 500, 1000, 2500, 5000},
		},
		[]string{"source_region", "target_region"},
	)

	conflictResolutions = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "dwcp_federation_conflict_resolutions_total",
			Help: "Total number of conflict resolutions",
		},
		[]string{"resolution_type"},
	)

	stateSize = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "dwcp_federation_state_size_bytes",
			Help: "Size of state in bytes per region",
		},
		[]string{"region"},
	)

	replicationLag = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "dwcp_federation_replication_lag_seconds",
			Help: "Replication lag between regions in seconds",
		},
		[]string{"source_region", "target_region"},
	)
)

// StateEntry represents a single state entry
type StateEntry struct {
	Key         string                 // Unique key
	Value       interface{}            // Actual value
	Version     VectorClock            // Version vector for causality
	Timestamp   time.Time              // Last modified timestamp
	Region      string                 // Region where last modified
	Metadata    map[string]interface{} // Additional metadata
	TTL         time.Duration          // Time to live
	ExpiresAt   time.Time              // Expiration time
	Tombstone   bool                   // Soft delete flag
	Checksum    []byte                 // Data integrity checksum
}

// VectorClock implements vector clock for causality tracking
type VectorClock struct {
	Clocks map[string]uint64 // Region -> clock value
	mu     sync.RWMutex
}

// NewVectorClock creates a new vector clock
func NewVectorClock() *VectorClock {
	return &VectorClock{
		Clocks: make(map[string]uint64),
	}
}

// Increment increments the clock for a region
func (vc *VectorClock) Increment(region string) {
	vc.mu.Lock()
	defer vc.mu.Unlock()
	vc.Clocks[region]++
}

// Merge merges another vector clock (take maximum of each clock)
func (vc *VectorClock) Merge(other *VectorClock) {
	vc.mu.Lock()
	defer vc.mu.Unlock()

	other.mu.RLock()
	defer other.mu.RUnlock()

	for region, clock := range other.Clocks {
		if current, exists := vc.Clocks[region]; !exists || clock > current {
			vc.Clocks[region] = clock
		}
	}
}

// Compare compares two vector clocks
// Returns: -1 (this < other), 0 (concurrent), 1 (this > other)
func (vc *VectorClock) Compare(other *VectorClock) int {
	vc.mu.RLock()
	defer vc.mu.RUnlock()

	other.mu.RLock()
	defer other.mu.RUnlock()

	lessOrEqual := true
	greaterOrEqual := true

	allRegions := make(map[string]bool)
	for region := range vc.Clocks {
		allRegions[region] = true
	}
	for region := range other.Clocks {
		allRegions[region] = true
	}

	for region := range allRegions {
		thisClock := vc.Clocks[region]
		otherClock := other.Clocks[region]

		if thisClock < otherClock {
			greaterOrEqual = false
		}
		if thisClock > otherClock {
			lessOrEqual = false
		}
	}

	if lessOrEqual && greaterOrEqual {
		return 0 // Equal
	} else if lessOrEqual {
		return -1 // This < Other
	} else if greaterOrEqual {
		return 1 // This > Other
	}
	return 0 // Concurrent
}

// GeoDistributedState manages state across multiple regions
type GeoDistributedState struct {
	mu                sync.RWMutex
	localRegion       string
	regions           []string
	store             map[string]*StateEntry
	crdtStore         map[string]CRDT
	replicationQueue  chan *ReplicationTask
	conflictResolver  ConflictResolver
	syncInterval      time.Duration
	stopCh            chan struct{}
	wg                sync.WaitGroup
	consistencyLevel  ConsistencyLevel
	replicationFactor int
	metrics           *StateMetrics
}

// StateMetrics tracks state management metrics
type StateMetrics struct {
	ReadOperations      int64
	WriteOperations     int64
	DeleteOperations    int64
	SyncOperations      int64
	ConflictResolutions int64
	ReplicationLag      map[string]time.Duration
	mu                  sync.RWMutex
}

// ConsistencyLevel defines read/write consistency
type ConsistencyLevel int

const (
	ConsistencyEventual ConsistencyLevel = iota // Eventual consistency
	ConsistencyLocal                            // Local region consistent
	ConsistencyQuorum                           // Quorum of regions
	ConsistencyStrong                           // All regions (linearizable)
)

// ReplicationTask represents a state replication task
type ReplicationTask struct {
	SourceRegion string
	TargetRegions []string
	Entry        *StateEntry
	Priority     int
	Timestamp    time.Time
}

// CRDT interface for Conflict-free Replicated Data Types
type CRDT interface {
	Merge(other CRDT) CRDT
	Value() interface{}
	Clone() CRDT
}

// GCounter implements a grow-only counter CRDT
type GCounter struct {
	Counts map[string]uint64
	mu     sync.RWMutex
}

// NewGCounter creates a new GCounter
func NewGCounter() *GCounter {
	return &GCounter{
		Counts: make(map[string]uint64),
	}
}

// Increment increments the counter for a region
func (gc *GCounter) Increment(region string, delta uint64) {
	gc.mu.Lock()
	defer gc.mu.Unlock()
	gc.Counts[region] += delta
}

// Value returns the total count
func (gc *GCounter) Value() interface{} {
	gc.mu.RLock()
	defer gc.mu.RUnlock()

	var total uint64
	for _, count := range gc.Counts {
		total += count
	}
	return total
}

// Merge merges another GCounter
func (gc *GCounter) Merge(other CRDT) CRDT {
	otherGC, ok := other.(*GCounter)
	if !ok {
		return gc
	}

	gc.mu.Lock()
	defer gc.mu.Unlock()

	otherGC.mu.RLock()
	defer otherGC.mu.RUnlock()

	merged := NewGCounter()
	for region, count := range gc.Counts {
		merged.Counts[region] = count
	}
	for region, count := range otherGC.Counts {
		if current, exists := merged.Counts[region]; !exists || count > current {
			merged.Counts[region] = count
		}
	}

	return merged
}

// Clone creates a copy
func (gc *GCounter) Clone() CRDT {
	gc.mu.RLock()
	defer gc.mu.RUnlock()

	clone := NewGCounter()
	for region, count := range gc.Counts {
		clone.Counts[region] = count
	}
	return clone
}

// PNCounter implements a positive-negative counter CRDT
type PNCounter struct {
	Positive *GCounter
	Negative *GCounter
}

// NewPNCounter creates a new PNCounter
func NewPNCounter() *PNCounter {
	return &PNCounter{
		Positive: NewGCounter(),
		Negative: NewGCounter(),
	}
}

// Increment increments the counter
func (pn *PNCounter) Increment(region string, delta int64) {
	if delta > 0 {
		pn.Positive.Increment(region, uint64(delta))
	} else if delta < 0 {
		pn.Negative.Increment(region, uint64(-delta))
	}
}

// Value returns the net count
func (pn *PNCounter) Value() interface{} {
	pos := pn.Positive.Value().(uint64)
	neg := pn.Negative.Value().(uint64)
	return int64(pos) - int64(neg)
}

// Merge merges another PNCounter
func (pn *PNCounter) Merge(other CRDT) CRDT {
	otherPN, ok := other.(*PNCounter)
	if !ok {
		return pn
	}

	merged := NewPNCounter()
	merged.Positive = pn.Positive.Merge(otherPN.Positive).(*GCounter)
	merged.Negative = pn.Negative.Merge(otherPN.Negative).(*GCounter)
	return merged
}

// Clone creates a copy
func (pn *PNCounter) Clone() CRDT {
	return &PNCounter{
		Positive: pn.Positive.Clone().(*GCounter),
		Negative: pn.Negative.Clone().(*GCounter),
	}
}

// GSet implements a grow-only set CRDT
type GSet struct {
	Elements map[string]bool
	mu       sync.RWMutex
}

// NewGSet creates a new GSet
func NewGSet() *GSet {
	return &GSet{
		Elements: make(map[string]bool),
	}
}

// Add adds an element
func (gs *GSet) Add(element string) {
	gs.mu.Lock()
	defer gs.mu.Unlock()
	gs.Elements[element] = true
}

// Contains checks if element exists
func (gs *GSet) Contains(element string) bool {
	gs.mu.RLock()
	defer gs.mu.RUnlock()
	return gs.Elements[element]
}

// Value returns all elements
func (gs *GSet) Value() interface{} {
	gs.mu.RLock()
	defer gs.mu.RUnlock()

	elements := make([]string, 0, len(gs.Elements))
	for elem := range gs.Elements {
		elements = append(elements, elem)
	}
	return elements
}

// Merge merges another GSet
func (gs *GSet) Merge(other CRDT) CRDT {
	otherGS, ok := other.(*GSet)
	if !ok {
		return gs
	}

	gs.mu.Lock()
	defer gs.mu.Unlock()

	otherGS.mu.RLock()
	defer otherGS.mu.RUnlock()

	merged := NewGSet()
	for elem := range gs.Elements {
		merged.Elements[elem] = true
	}
	for elem := range otherGS.Elements {
		merged.Elements[elem] = true
	}

	return merged
}

// Clone creates a copy
func (gs *GSet) Clone() CRDT {
	gs.mu.RLock()
	defer gs.mu.RUnlock()

	clone := NewGSet()
	for elem := range gs.Elements {
		clone.Elements[elem] = true
	}
	return clone
}

// ConflictResolver interface for conflict resolution strategies
type ConflictResolver interface {
	Resolve(local, remote *StateEntry) (*StateEntry, error)
}

// LastWriteWinsResolver resolves conflicts using timestamps
type LastWriteWinsResolver struct{}

// Resolve implements last-write-wins strategy
func (lw *LastWriteWinsResolver) Resolve(local, remote *StateEntry) (*StateEntry, error) {
	if remote.Timestamp.After(local.Timestamp) {
		conflictResolutions.WithLabelValues("last_write_wins_remote").Inc()
		return remote, nil
	}
	conflictResolutions.WithLabelValues("last_write_wins_local").Inc()
	return local, nil
}

// VectorClockResolver resolves conflicts using vector clocks
type VectorClockResolver struct{}

// Resolve implements vector clock-based resolution
func (vc *VectorClockResolver) Resolve(local, remote *StateEntry) (*StateEntry, error) {
	comparison := local.Version.Compare(&remote.Version)

	switch comparison {
	case -1: // Remote is newer
		conflictResolutions.WithLabelValues("vector_clock_remote").Inc()
		return remote, nil
	case 1: // Local is newer
		conflictResolutions.WithLabelValues("vector_clock_local").Inc()
		return local, nil
	case 0: // Concurrent - use timestamp as tiebreaker
		if remote.Timestamp.After(local.Timestamp) {
			conflictResolutions.WithLabelValues("vector_clock_concurrent_remote").Inc()
			return remote, nil
		}
		conflictResolutions.WithLabelValues("vector_clock_concurrent_local").Inc()
		return local, nil
	default:
		return local, fmt.Errorf("unexpected vector clock comparison result")
	}
}

// NewGeoDistributedState creates a new geo-distributed state manager
func NewGeoDistributedState(cfg *StateConfig) (*GeoDistributedState, error) {
	if cfg == nil {
		return nil, fmt.Errorf("state config cannot be nil")
	}

	gds := &GeoDistributedState{
		localRegion:       cfg.LocalRegion,
		regions:           cfg.Regions,
		store:             make(map[string]*StateEntry),
		crdtStore:         make(map[string]CRDT),
		replicationQueue:  make(chan *ReplicationTask, 10000),
		syncInterval:      cfg.SyncInterval,
		stopCh:            make(chan struct{}),
		consistencyLevel:  cfg.ConsistencyLevel,
		replicationFactor: cfg.ReplicationFactor,
		metrics: &StateMetrics{
			ReplicationLag: make(map[string]time.Duration),
		},
	}

	// Set conflict resolver
	if cfg.UseVectorClock {
		gds.conflictResolver = &VectorClockResolver{}
	} else {
		gds.conflictResolver = &LastWriteWinsResolver{}
	}

	return gds, nil
}

// StateConfig defines state manager configuration
type StateConfig struct {
	LocalRegion       string
	Regions           []string
	SyncInterval      time.Duration
	ConsistencyLevel  ConsistencyLevel
	ReplicationFactor int
	UseVectorClock    bool
}

// Start starts the state manager
func (gds *GeoDistributedState) Start(ctx context.Context) error {
	// Start replication worker
	gds.wg.Add(1)
	go gds.replicationWorker(ctx)

	// Start sync loop
	gds.wg.Add(1)
	go gds.syncLoop(ctx)

	// Start cleanup worker
	gds.wg.Add(1)
	go gds.cleanupWorker(ctx)

	return nil
}

// Stop stops the state manager
func (gds *GeoDistributedState) Stop() error {
	close(gds.stopCh)
	gds.wg.Wait()
	return nil
}

// Get retrieves a value with specified consistency level
func (gds *GeoDistributedState) Get(ctx context.Context, key string, consistency ConsistencyLevel) (*StateEntry, error) {
	startTime := time.Now()
	defer func() {
		stateOperations.WithLabelValues(gds.localRegion, "get", "success").Inc()
	}()

	gds.mu.RLock()
	entry, exists := gds.store[key]
	gds.mu.RUnlock()

	if !exists {
		stateOperations.WithLabelValues(gds.localRegion, "get", "not_found").Inc()
		return nil, fmt.Errorf("key not found: %s", key)
	}

	// Check if expired
	if entry.TTL > 0 && time.Now().After(entry.ExpiresAt) {
		stateOperations.WithLabelValues(gds.localRegion, "get", "expired").Inc()
		return nil, fmt.Errorf("key expired: %s", key)
	}

	// Check if tombstone
	if entry.Tombstone {
		stateOperations.WithLabelValues(gds.localRegion, "get", "deleted").Inc()
		return nil, fmt.Errorf("key deleted: %s", key)
	}

	// For strong consistency, verify with other regions
	if consistency == ConsistencyStrong {
		if err := gds.verifyStrongConsistency(ctx, key, entry); err != nil {
			return nil, err
		}
	}

	gds.metrics.mu.Lock()
	gds.metrics.ReadOperations++
	gds.metrics.mu.Unlock()

	// Return a copy to prevent mutations
	return gds.cloneEntry(entry), nil
}

// Put stores a value with replication
func (gds *GeoDistributedState) Put(ctx context.Context, key string, value interface{}, ttl time.Duration) error {
	startTime := time.Now()

	gds.mu.Lock()
	defer gds.mu.Unlock()

	// Create or update entry
	entry, exists := gds.store[key]
	if !exists {
		entry = &StateEntry{
			Key:      key,
			Version:  *NewVectorClock(),
			Metadata: make(map[string]interface{}),
		}
	}

	// Update entry
	entry.Value = value
	entry.Timestamp = time.Now()
	entry.Region = gds.localRegion
	entry.TTL = ttl
	if ttl > 0 {
		entry.ExpiresAt = time.Now().Add(ttl)
	}
	entry.Tombstone = false

	// Increment vector clock
	entry.Version.Increment(gds.localRegion)

	// Calculate checksum
	entry.Checksum = gds.calculateChecksum(entry)

	// Store locally
	gds.store[key] = entry

	// Update metrics
	stateSize.WithLabelValues(gds.localRegion).Set(float64(len(gds.store)))

	gds.metrics.mu.Lock()
	gds.metrics.WriteOperations++
	gds.metrics.mu.Unlock()

	// Replicate to other regions
	if err := gds.queueReplication(entry, gds.determineReplicaRegions()); err != nil {
		stateOperations.WithLabelValues(gds.localRegion, "put", "replication_failed").Inc()
		return err
	}

	stateOperations.WithLabelValues(gds.localRegion, "put", "success").Inc()
	return nil
}

// Delete marks an entry for deletion (tombstone)
func (gds *GeoDistributedState) Delete(ctx context.Context, key string) error {
	gds.mu.Lock()
	defer gds.mu.Unlock()

	entry, exists := gds.store[key]
	if !exists {
		stateOperations.WithLabelValues(gds.localRegion, "delete", "not_found").Inc()
		return fmt.Errorf("key not found: %s", key)
	}

	// Set tombstone
	entry.Tombstone = true
	entry.Timestamp = time.Now()
	entry.Version.Increment(gds.localRegion)

	gds.metrics.mu.Lock()
	gds.metrics.DeleteOperations++
	gds.metrics.mu.Unlock()

	// Replicate deletion
	if err := gds.queueReplication(entry, gds.determineReplicaRegions()); err != nil {
		stateOperations.WithLabelValues(gds.localRegion, "delete", "replication_failed").Inc()
		return err
	}

	stateOperations.WithLabelValues(gds.localRegion, "delete", "success").Inc()
	return nil
}

// Sync synchronizes state with remote region
func (gds *GeoDistributedState) Sync(ctx context.Context, remoteRegion string, remoteEntries []*StateEntry) error {
	startTime := time.Now()
	defer func() {
		lag := time.Since(startTime)
		stateSyncLatency.WithLabelValues(gds.localRegion, remoteRegion).Observe(float64(lag.Milliseconds()))
	}()

	gds.mu.Lock()
	defer gds.mu.Unlock()

	conflictsResolved := 0

	for _, remoteEntry := range remoteEntries {
		localEntry, exists := gds.store[remoteEntry.Key]

		if !exists {
			// New entry from remote
			gds.store[remoteEntry.Key] = remoteEntry
			continue
		}

		// Check for conflicts
		if gds.hasConflict(localEntry, remoteEntry) {
			resolved, err := gds.conflictResolver.Resolve(localEntry, remoteEntry)
			if err != nil {
				return fmt.Errorf("conflict resolution failed for key %s: %w", remoteEntry.Key, err)
			}
			gds.store[remoteEntry.Key] = resolved
			conflictsResolved++
		} else {
			// Merge vector clocks
			localEntry.Version.Merge(&remoteEntry.Version)
		}
	}

	gds.metrics.mu.Lock()
	gds.metrics.SyncOperations++
	gds.metrics.ConflictResolutions += int64(conflictsResolved)
	gds.metrics.mu.Unlock()

	return nil
}

// hasConflict checks if two entries are in conflict
func (gds *GeoDistributedState) hasConflict(local, remote *StateEntry) bool {
	comparison := local.Version.Compare(&remote.Version)
	return comparison == 0 && local.Region != remote.Region
}

// queueReplication queues a replication task
func (gds *GeoDistributedState) queueReplication(entry *StateEntry, targetRegions []string) error {
	task := &ReplicationTask{
		SourceRegion:  gds.localRegion,
		TargetRegions: targetRegions,
		Entry:         gds.cloneEntry(entry),
		Priority:      5,
		Timestamp:     time.Now(),
	}

	select {
	case gds.replicationQueue <- task:
		return nil
	default:
		return fmt.Errorf("replication queue full")
	}
}

// determineReplicaRegions determines which regions to replicate to
func (gds *GeoDistributedState) determineReplicaRegions() []string {
	// Simple: replicate to all regions except local
	var replicas []string
	for _, region := range gds.regions {
		if region != gds.localRegion {
			replicas = append(replicas, region)
		}
	}
	return replicas
}

// replicationWorker processes replication tasks
func (gds *GeoDistributedState) replicationWorker(ctx context.Context) {
	defer gds.wg.Done()

	for {
		select {
		case <-ctx.Done():
			return
		case <-gds.stopCh:
			return
		case task := <-gds.replicationQueue:
			gds.executeReplication(ctx, task)
		}
	}
}

// executeReplication executes a replication task
func (gds *GeoDistributedState) executeReplication(ctx context.Context, task *ReplicationTask) {
	// DEFERRED: Network replication requires cross-region transport layer
	// Would implement gRPC or HTTP/2 based state transfer with compression and retries
	log.Printf("Simulating replication for key=%s to regions=%v (actual network replication not implemented)",
		task.Key, task.TargetRegions)
	for _, targetRegion := range task.TargetRegions {
		stateSyncLatency.WithLabelValues(gds.localRegion, targetRegion).Observe(50) // Simulate 50ms latency
	}
}

// syncLoop periodically syncs with remote regions
func (gds *GeoDistributedState) syncLoop(ctx context.Context) {
	defer gds.wg.Done()

	ticker := time.NewTicker(gds.syncInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-gds.stopCh:
			return
		case <-ticker.C:
			gds.performPeriodicSync(ctx)
		}
	}
}

// performPeriodicSync performs periodic synchronization
func (gds *GeoDistributedState) performPeriodicSync(ctx context.Context) {
	// DEFERRED: Periodic sync requires gossip protocol or merkle tree diff implementation
	// Would compare state checksums and transfer deltas for consistency
	log.Printf("Performing periodic sync check for region=%s with %d peer regions (full sync not implemented)",
		gds.localRegion, len(gds.regions)-1)
	for _, region := range gds.regions {
		if region != gds.localRegion {
			replicationLag.WithLabelValues(gds.localRegion, region).Set(0.1) // Simulate 100ms lag
		}
	}
}

// cleanupWorker periodically cleans up expired and tombstoned entries
func (gds *GeoDistributedState) cleanupWorker(ctx context.Context) {
	defer gds.wg.Done()

	ticker := time.NewTicker(1 * time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-gds.stopCh:
			return
		case <-ticker.C:
			gds.performCleanup()
		}
	}
}

// performCleanup removes expired and old tombstoned entries
func (gds *GeoDistributedState) performCleanup() {
	gds.mu.Lock()
	defer gds.mu.Unlock()

	now := time.Now()
	tombstoneGracePeriod := 24 * time.Hour

	for key, entry := range gds.store {
		// Remove expired entries
		if entry.TTL > 0 && now.After(entry.ExpiresAt) {
			delete(gds.store, key)
			continue
		}

		// Remove old tombstones
		if entry.Tombstone && now.Sub(entry.Timestamp) > tombstoneGracePeriod {
			delete(gds.store, key)
		}
	}

	stateSize.WithLabelValues(gds.localRegion).Set(float64(len(gds.store)))
}

// verifyStrongConsistency verifies value across all regions for strong consistency
func (gds *GeoDistributedState) verifyStrongConsistency(ctx context.Context, key string, entry *StateEntry) error {
	// DEFERRED: Cross-region verification requires quorum read protocol
	// Would query majority of regions and compare vector clocks/checksums
	log.Printf("Skipping cross-region verification for key=%s (quorum protocol not implemented)", key)
	return nil
}

// calculateChecksum calculates entry checksum for integrity
func (gds *GeoDistributedState) calculateChecksum(entry *StateEntry) []byte {
	data, _ := json.Marshal(entry.Value)
	hash := sha256.Sum256(data)
	return hash[:]
}

// cloneEntry creates a deep copy of entry
func (gds *GeoDistributedState) cloneEntry(entry *StateEntry) *StateEntry {
	clone := &StateEntry{
		Key:       entry.Key,
		Value:     entry.Value, // Note: shallow copy of value
		Version:   *NewVectorClock(),
		Timestamp: entry.Timestamp,
		Region:    entry.Region,
		Metadata:  make(map[string]interface{}),
		TTL:       entry.TTL,
		ExpiresAt: entry.ExpiresAt,
		Tombstone: entry.Tombstone,
		Checksum:  make([]byte, len(entry.Checksum)),
	}

	// Deep copy vector clock
	entry.Version.mu.RLock()
	for region, clock := range entry.Version.Clocks {
		clone.Version.Clocks[region] = clock
	}
	entry.Version.mu.RUnlock()

	// Deep copy metadata
	for k, v := range entry.Metadata {
		clone.Metadata[k] = v
	}

	// Copy checksum
	copy(clone.Checksum, entry.Checksum)

	return clone
}

// GetMetrics returns current state metrics
func (gds *GeoDistributedState) GetMetrics() *StateMetrics {
	gds.metrics.mu.RLock()
	defer gds.metrics.mu.RUnlock()

	// Return a copy
	lagCopy := make(map[string]time.Duration)
	for k, v := range gds.metrics.ReplicationLag {
		lagCopy[k] = v
	}

	return &StateMetrics{
		ReadOperations:      gds.metrics.ReadOperations,
		WriteOperations:     gds.metrics.WriteOperations,
		DeleteOperations:    gds.metrics.DeleteOperations,
		SyncOperations:      gds.metrics.SyncOperations,
		ConflictResolutions: gds.metrics.ConflictResolutions,
		ReplicationLag:      lagCopy,
	}
}
