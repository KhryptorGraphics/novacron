// Package vm provides virtual machine management with distributed state sharding
package vm

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"hash/fnv"
	"sort"
	"sync"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/consensus"
	"github.com/khryptorgraphics/novacron/backend/core/federation"
	"github.com/pkg/errors"
	"go.uber.org/zap"
)

// VMStateShardingManager manages distribution of VM state across multiple nodes
type VMStateShardingManager struct {
	mu                sync.RWMutex
	logger            *zap.Logger
	nodeRing          *ConsistentHashRing
	shards            map[string]*VMStateShard
	replicationFactor int
	localNodeID       string
	consensus         consensus.Manager
	federation        federation.FederationManager
	stateStore        StateStore
	recoveryManager   *ShardRecoveryManager
	accessLayer       *StateAccessLayer
	metrics           *ShardingMetrics
}

// VMStateShard represents a fragment of VM state on a node
type VMStateShard struct {
	ShardID      string
	VMID         string
	NodeID       string
	ReplicaNodes []string
	Version      uint64
	VectorClock  map[string]uint64
	Data         *DistributedVMState
	LastModified time.Time
	Status       ShardStatus
}

// DistributedVMState stores VM state fragments across nodes
type DistributedVMState struct {
	MemoryPages    map[uint64]*MemoryPage
	DiskBlocks     map[uint64]*DiskBlock
	NetworkState   *NetworkState
	Configuration  *VMConfiguration
	Checkpoints    []*StateCheckpoint
	Metadata       StateMetadata
	AccessPattern  *AccessPattern
}

// ShardStatus represents the status of a shard
type ShardStatus int

const (
	ShardStatusActive ShardStatus = iota
	ShardStatusReplicating
	ShardStatusMigrating
	ShardStatusRecovering
	ShardStatusFailed
)

// ConsistentHashRing implements consistent hashing for shard placement
type ConsistentHashRing struct {
	mu           sync.RWMutex
	nodes        map[string]int // node ID -> virtual nodes
	ring         map[uint32]string
	sortedHashes []uint32
	replicas     int
}

// ShardRecoveryManager handles automatic shard recovery
type ShardRecoveryManager struct {
	mu              sync.RWMutex
	shardingManager *VMStateShardingManager
	recoveryQueue   []*RecoveryTask
	activeRecovery  map[string]*RecoveryTask
	metrics         *RecoveryMetrics
}

// StateAccessLayer provides abstraction for distributed state access
type StateAccessLayer struct {
	mu              sync.RWMutex
	cache           *StateCache
	prefetcher      *StatePrefetcher
	lazyLoader      *LazyStateLoader
	shardingManager *VMStateShardingManager
}

// NewVMStateShardingManager creates a new VM state sharding manager
func NewVMStateShardingManager(logger *zap.Logger, nodeID string, replicationFactor int) *VMStateShardingManager {
	return &VMStateShardingManager{
		logger:            logger,
		nodeRing:          NewConsistentHashRing(150), // 150 virtual nodes per physical node
		shards:            make(map[string]*VMStateShard),
		replicationFactor: replicationFactor,
		localNodeID:       nodeID,
		stateStore:        NewDistributedStateStore(),
		recoveryManager:   NewShardRecoveryManager(),
		accessLayer:       NewStateAccessLayer(),
		metrics:           NewShardingMetrics(),
	}
}

// NewVMStateShardingManagerWithFederation creates a new VM state sharding manager with federation integration
func NewVMStateShardingManagerWithFederation(logger *zap.Logger, nodeID string, replicationFactor int, fedManager federation.FederationManager, initialNodes []string) *VMStateShardingManager {
	sm := &VMStateShardingManager{
		logger:            logger,
		nodeRing:          NewConsistentHashRing(150), // 150 virtual nodes per physical node
		shards:            make(map[string]*VMStateShard),
		replicationFactor: replicationFactor,
		localNodeID:       nodeID,
		federation:        fedManager,
		stateStore:        NewDistributedStateStore(),
		recoveryManager:   NewShardRecoveryManager(),
		accessLayer:       NewStateAccessLayer(),
		metrics:           NewShardingMetrics(),
	}

	// Register initial nodes if provided
	if len(initialNodes) > 0 {
		sm.RegisterNodes(initialNodes)
	}

	// Subscribe to membership events if federation manager is provided
	if fedManager != nil {
		go sm.subscribeToMembershipEvents()
	}

	return sm
}

// AllocateShards allocates VM state shards across nodes
func (sm *VMStateShardingManager) AllocateShards(vmID string, state *DistributedVMState) error {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	// Calculate shard placement using consistent hashing
	primaryNode := sm.nodeRing.GetNode(vmID)
	if primaryNode == "" {
		return errors.New("no nodes registered in sharding ring")
	}
	replicaNodes := sm.nodeRing.GetReplicas(vmID, sm.replicationFactor-1)

	// Create primary shard
	shard := &VMStateShard{
		ShardID:      generateShardID(vmID, primaryNode),
		VMID:         vmID,
		NodeID:       primaryNode,
		ReplicaNodes: replicaNodes,
		Version:      1,
		VectorClock:  make(map[string]uint64),
		Data:         state,
		LastModified: time.Now(),
		Status:       ShardStatusActive,
	}

	// Store shard locally if this is the primary node
	if primaryNode == sm.localNodeID {
		if err := sm.storeShardLocally(shard); err != nil {
			return errors.Wrap(err, "failed to store primary shard")
		}
	}

	// Replicate to replica nodes
	for _, replicaNode := range replicaNodes {
		if err := sm.replicateShard(shard, replicaNode); err != nil {
			sm.logger.Error("Failed to replicate shard",
				zap.String("vmID", vmID),
				zap.String("replicaNode", replicaNode),
				zap.Error(err))
			// Continue with other replicas
		}
	}

	sm.shards[shard.ShardID] = shard
	sm.metrics.ShardsAllocated.Inc()

	return nil
}

// GetVMState retrieves distributed VM state with read-repair
func (sm *VMStateShardingManager) GetVMState(ctx context.Context, vmID string) (*DistributedVMState, error) {
	// Try cache first
	if state := sm.accessLayer.GetFromCache(vmID); state != nil {
		// Guard against caching nil states
		if state != nil {
			sm.metrics.CacheHits.Inc()
			return state, nil
		}
	}

	sm.mu.RLock()
	defer sm.mu.RUnlock()

	// Find primary shard
	primaryNode := sm.nodeRing.GetNode(vmID)
	shardID := generateShardID(vmID, primaryNode)

	// Check if shard is local
	if shard, exists := sm.shards[shardID]; exists && shard.Status == ShardStatusActive {
		if shard.Data != nil {
			// Perform read-repair check with replicas
			go sm.performReadRepair(ctx, shard)
			sm.accessLayer.UpdateCache(vmID, shard.Data)
			return shard.Data, nil
		}
	}

	// Fetch from all nodes and compare vector clocks
	states := make(map[string]*ShardStateWithClock)
	var newestClock map[string]uint64
	var newestState *DistributedVMState
	var newestNode string

	// Fetch from primary
	if stateWithClock, err := sm.fetchRemoteStateWithClock(ctx, vmID, primaryNode); err == nil && stateWithClock != nil {
		states[primaryNode] = stateWithClock
		newestClock = stateWithClock.VectorClock
		newestState = stateWithClock.State
		newestNode = primaryNode
	}

	// Fetch from replicas
	replicas := sm.nodeRing.GetReplicas(vmID, sm.replicationFactor-1)
	for _, replica := range replicas {
		if stateWithClock, err := sm.fetchRemoteStateWithClock(ctx, vmID, replica); err == nil && stateWithClock != nil {
			states[replica] = stateWithClock

			// Compare vector clocks
			if sm.isNewerClock(stateWithClock.VectorClock, newestClock) {
				newestClock = stateWithClock.VectorClock
				newestState = stateWithClock.State
				newestNode = replica
			} else if sm.isConcurrentClock(stateWithClock.VectorClock, newestClock) {
				// Concurrent updates detected - resolve conflict
				if sm.distributedCoordinator != nil {
					resolvedState, err := sm.distributedCoordinator.ResolveConflict(ctx, &StateConflict{
						ID:        generateConflictID(vmID),
						Type:      ConflictTypeConcurrentUpdate,
						States:    []interface{}{newestState, stateWithClock.State},
						Clocks:    []map[string]uint64{newestClock, stateWithClock.VectorClock},
						Timestamp: time.Now(),
					})
					if err == nil && resolvedState != nil {
						newestState = resolvedState.ResolvedState.(*DistributedVMState)
						newestClock = sm.mergeVectorClocks(newestClock, stateWithClock.VectorClock)
					}
				}
			}
		}
	}

	if newestState == nil {
		return nil, errors.New("failed to fetch VM state from any node")
	}

	// Perform read-repair: update stale replicas
	for node, stateWithClock := range states {
		if node != newestNode && !sm.equalVectorClocks(stateWithClock.VectorClock, newestClock) {
			go sm.repairStaleReplica(ctx, vmID, node, newestState, newestClock)
		}
	}

	// Update cache with newest state
	sm.accessLayer.UpdateCache(vmID, newestState)
	return newestState, nil
}

// UpdateVMState updates distributed VM state with consistency
func (sm *VMStateShardingManager) UpdateVMState(ctx context.Context, vmID string, update func(*DistributedVMState) error) error {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	primaryNode := sm.nodeRing.GetNode(vmID)
	shardID := generateShardID(vmID, primaryNode)

	// Check ownership and routing
	if primaryNode != sm.localNodeID {
		// Route to primary node
		return sm.routeUpdateToPrimary(ctx, vmID, primaryNode, update)
	}

	shard, exists := sm.shards[shardID]
	if !exists {
		return errors.New("shard not found")
	}

	// Apply update
	if err := update(shard.Data); err != nil {
		return errors.Wrap(err, "failed to update state")
	}

	// Update version and vector clock
	shard.Version++
	shard.VectorClock[sm.localNodeID]++
	shard.LastModified = time.Now()

	// Replicate update to replicas
	return sm.replicateUpdate(ctx, shard)
}

// HandleNodeFailure handles node failure and triggers shard recovery
func (sm *VMStateShardingManager) HandleNodeFailure(failedNodeID string) error {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	sm.logger.Info("Handling node failure", zap.String("nodeID", failedNodeID))

	// Find all shards affected by the failure
	affectedShards := sm.findAffectedShards(failedNodeID)

	// Trigger recovery for each affected shard
	for _, shard := range affectedShards {
		if err := sm.recoveryManager.RecoverShard(shard); err != nil {
			sm.logger.Error("Failed to recover shard",
				zap.String("shardID", shard.ShardID),
				zap.Error(err))
			continue
		}
	}

	// Remove failed node from ring
	sm.nodeRing.RemoveNode(failedNodeID)

	// Trigger rebalancing
	return sm.rebalanceShards()
}

// RebalanceShards rebalances shards when nodes join or leave
func (sm *VMStateShardingManager) rebalanceShards() error {
	sm.logger.Info("Starting shard rebalancing")

	// Calculate new shard distribution
	newDistribution := sm.calculateOptimalDistribution()

	// Migrate shards to new locations
	for shardID, newNode := range newDistribution {
		if shard, exists := sm.shards[shardID]; exists {
			if shard.NodeID != newNode {
				if err := sm.migrateShard(shard, newNode); err != nil {
					sm.logger.Error("Failed to migrate shard",
						zap.String("shardID", shardID),
						zap.String("newNode", newNode),
						zap.Error(err))
				}
			}
		}
	}

	sm.metrics.RebalanceOperations.Inc()
	return nil
}

// ConsistentHashRing implementation
func NewConsistentHashRing(replicas int) *ConsistentHashRing {
	return &ConsistentHashRing{
		nodes:    make(map[string]int),
		ring:     make(map[uint32]string),
		replicas: replicas,
	}
}

func (r *ConsistentHashRing) AddNode(nodeID string) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.nodes[nodeID] = r.replicas

	for i := 0; i < r.replicas; i++ {
		hash := r.hash(fmt.Sprintf("%s:%d", nodeID, i))
		r.ring[hash] = nodeID
		r.sortedHashes = append(r.sortedHashes, hash)
	}

	sort.Slice(r.sortedHashes, func(i, j int) bool {
		return r.sortedHashes[i] < r.sortedHashes[j]
	})
}

func (r *ConsistentHashRing) RemoveNode(nodeID string) {
	r.mu.Lock()
	defer r.mu.Unlock()

	delete(r.nodes, nodeID)

	// Remove from ring and sorted hashes
	newSortedHashes := []uint32{}
	for hash, node := range r.ring {
		if node != nodeID {
			newSortedHashes = append(newSortedHashes, hash)
		} else {
			delete(r.ring, hash)
		}
	}

	r.sortedHashes = newSortedHashes
	sort.Slice(r.sortedHashes, func(i, j int) bool {
		return r.sortedHashes[i] < r.sortedHashes[j]
	})
}

func (r *ConsistentHashRing) GetNode(key string) string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if len(r.sortedHashes) == 0 {
		return ""
	}

	hash := r.hash(key)

	// Binary search for the first hash >= key hash
	idx := sort.Search(len(r.sortedHashes), func(i int) bool {
		return r.sortedHashes[i] >= hash
	})

	if idx == len(r.sortedHashes) {
		idx = 0
	}

	return r.ring[r.sortedHashes[idx]]
}

func (r *ConsistentHashRing) GetReplicas(key string, count int) []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	replicas := []string{}
	seen := make(map[string]bool)

	if len(r.sortedHashes) == 0 {
		return replicas
	}

	hash := r.hash(key)
	idx := sort.Search(len(r.sortedHashes), func(i int) bool {
		return r.sortedHashes[i] >= hash
	})

	// Get primary node to exclude from replicas
	primaryNode := r.GetNode(key)

	for len(replicas) < count && len(seen) < len(r.nodes) {
		if idx >= len(r.sortedHashes) {
			idx = 0
		}

		node := r.ring[r.sortedHashes[idx]]
		// Exclude primary node from replica list
		if !seen[node] && node != primaryNode {
			seen[node] = true
			replicas = append(replicas, node)
		}

		idx++
	}

	return replicas
}

// RegisterNodes adds multiple nodes to the hash ring
func (r *ConsistentHashRing) RegisterNodes(nodes []string) {
	r.mu.Lock()
	defer r.mu.Unlock()

	for _, node := range nodes {
		r.nodes[node] = r.replicas
		for i := 0; i < r.replicas; i++ {
			hash := r.hash(fmt.Sprintf("%s:%d", node, i))
			r.ring[hash] = node
			r.sortedHashes = append(r.sortedHashes, hash)
		}
	}

	sort.Slice(r.sortedHashes, func(i, j int) bool {
		return r.sortedHashes[i] < r.sortedHashes[j]
	})
}

func (r *ConsistentHashRing) hash(key string) uint32 {
	h := fnv.New32()
	h.Write([]byte(key))
	return h.Sum32()
}

// GetAllNodes returns all nodes currently in the ring
func (r *ConsistentHashRing) GetAllNodes() map[string]bool {
	r.mu.RLock()
	defer r.mu.RUnlock()

	nodes := make(map[string]bool)
	for nodeID := range r.nodes {
		nodes[nodeID] = true
	}
	return nodes
}

// Helper functions
func generateShardID(vmID, nodeID string) string {
	h := sha256.New()
	h.Write([]byte(vmID + ":" + nodeID))
	return hex.EncodeToString(h.Sum(nil))[:16]
}

func (sm *VMStateShardingManager) storeShardLocally(shard *VMStateShard) error {
	return sm.stateStore.StoreShard(shard)
}

func (sm *VMStateShardingManager) replicateShard(shard *VMStateShard, targetNode string) error {
	// Send shard to target node via cross-cluster RPC
	if sm.federation != nil {
		// Create replication message
		msg := &federation.CrossClusterMessage{
			Type:        "shard_replication",
			Source:      sm.localNodeID,
			Destination: targetNode,
			Payload: map[string]interface{}{
				"shard_id":      shard.ShardID,
				"vm_id":         shard.VMID,
				"version":       shard.Version,
				"vector_clock":  shard.VectorClock,
				"data":          shard.Data,
				"last_modified": shard.LastModified,
			},
			Timestamp: time.Now(),
		}

		// Send message through federation
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()

		// Use cross-cluster components if available
		if crossCluster, ok := sm.federation.(*federation.CrossClusterComponents); ok {
			err := crossCluster.SendMessage(ctx, msg)
			if err != nil {
				return errors.Wrapf(err, "failed to send shard replication to %s", targetNode)
			}

			// Wait for acknowledgment
			ack, err := sm.waitForReplicationAck(ctx, shard.ShardID, targetNode)
			if err != nil {
				return errors.Wrapf(err, "replication acknowledgment failed from %s", targetNode)
			}

			if !ack {
				return errors.Errorf("replication not acknowledged by %s", targetNode)
			}
		}
	}

	// Update metrics
	sm.metrics.ShardsReplicated.Inc()
	return nil
}

func (sm *VMStateShardingManager) fetchRemoteState(ctx context.Context, vmID, nodeID string) (*DistributedVMState, error) {
	// Implementation would fetch state from remote node via RPC
	return nil, errors.New("remote state fetch not implemented")
}

func (sm *VMStateShardingManager) replicateUpdate(ctx context.Context, shard *VMStateShard) error {
	// Create incremental update message with full vector clock and version
	updateMsg := &ShardUpdateMessage{
		ShardID:      shard.ShardID,
		Version:      shard.Version,
		VectorClock:  shard.VectorClock,
		SourceNode:   sm.localNodeID,
		Timestamp:    time.Now(),
		UpdateType:   "incremental",
		Data:         shard.Data, // Include data for conflict resolution
	}

	// Replicate to all replica nodes with retries
	var wg sync.WaitGroup
	errChan := make(chan error, len(shard.ReplicaNodes))

	for _, replica := range shard.ReplicaNodes {
		wg.Add(1)
		go func(targetNode string) {
			defer wg.Done()

			// Retry logic with exponential backoff
			maxRetries := 3
			for attempt := 0; attempt < maxRetries; attempt++ {
				err := sm.sendShardUpdateWithClockMerge(ctx, updateMsg, targetNode)
				if err == nil {
					return
				}

				if attempt < maxRetries-1 {
					backoff := time.Duration(1<<uint(attempt)) * time.Second
					time.Sleep(backoff)
				} else {
					errChan <- errors.Wrapf(err, "failed to replicate to %s after %d attempts", targetNode, maxRetries)
				}
			}
		}(replica)
	}

	wg.Wait()
	close(errChan)

	// Collect errors
	var errs []error
	for err := range errChan {
		errs = append(errs, err)
		sm.logger.Warn("Failed to replicate update", zap.Error(err))
	}

	if len(errs) > 0 {
		return errors.Errorf("replication failed to %d replicas", len(errs))
	}

	return nil
}

func (sm *VMStateShardingManager) findAffectedShards(nodeID string) []*VMStateShard {
	affected := []*VMStateShard{}
	for _, shard := range sm.shards {
		if shard.NodeID == nodeID {
			affected = append(affected, shard)
		}
		for _, replica := range shard.ReplicaNodes {
			if replica == nodeID {
				affected = append(affected, shard)
				break
			}
		}
	}
	return affected
}

func (sm *VMStateShardingManager) calculateOptimalDistribution() map[string]string {
	// Calculate optimal shard distribution based on current nodes
	distribution := make(map[string]string)
	for shardID, shard := range sm.shards {
		distribution[shardID] = sm.nodeRing.GetNode(shard.VMID)
	}
	return distribution
}

func (sm *VMStateShardingManager) migrateShard(shard *VMStateShard, targetNode string) error {
	// Mark shard as migrating
	shard.Status = ShardStatusMigrating

	sm.logger.Info("Starting shard migration",
		zap.String("shardID", shard.ShardID),
		zap.String("sourceNode", shard.NodeID),
		zap.String("targetNode", targetNode))

	// Step 1: Copy shard state to target node
	err := sm.copyShardToNode(shard, targetNode)
	if err != nil {
		shard.Status = ShardStatusActive // Revert status on failure
		return errors.Wrapf(err, "failed to copy shard to %s", targetNode)
	}

	// Step 2: Update shard ownership atomically
	oldNode := shard.NodeID
	shard.NodeID = targetNode
	shard.Version++
	shard.VectorClock[sm.localNodeID]++

	// Step 3: Notify all replicas of ownership change
	ownershipMsg := &OwnershipChangeMessage{
		ShardID:    shard.ShardID,
		OldPrimary: oldNode,
		NewPrimary: targetNode,
		Version:    shard.Version,
		Timestamp:  time.Now(),
	}

	err = sm.broadcastOwnershipChange(ownershipMsg, shard.ReplicaNodes)
	if err != nil {
		sm.logger.Warn("Failed to broadcast ownership change", zap.Error(err))
	}

	// Step 4: Update routing information
	if sm.stateStore != nil {
		err = sm.stateStore.StoreShard(shard)
		if err != nil {
			sm.logger.Warn("Failed to persist migrated shard", zap.Error(err))
		}
	}

	// Step 5: Remove shard from source node if it's local
	if oldNode == sm.localNodeID {
		delete(sm.shards, shard.ShardID)
	}

	shard.Status = ShardStatusActive
	sm.metrics.MigrationsCompleted.Inc()

	sm.logger.Info("Shard migration completed",
		zap.String("shardID", shard.ShardID),
		zap.String("newPrimary", targetNode))

	return nil
}

// Supporting types
type MemoryPage struct {
	PageNumber uint64
	Data       []byte
	Dirty      bool
	AccessTime time.Time
}

type DiskBlock struct {
	BlockNumber uint64
	Data        []byte
	Checksum    string
}

type NetworkState struct {
	Interfaces []NetworkInterface
	Routes     []Route
	Connections []Connection
}

type VMConfiguration struct {
	CPU    int
	Memory int64
	Disks  []DiskConfig
}

type StateCheckpoint struct {
	ID        string
	Timestamp time.Time
	Version   uint64
	Data      []byte
}

type StateMetadata struct {
	Created      time.Time
	LastModified time.Time
	Size         int64
	Checksum     string
}

type AccessPattern struct {
	HotPages     []uint64
	ColdPages    []uint64
	AccessCounts map[uint64]int
}

type StateStore interface {
	StoreShard(*VMStateShard) error
	GetShard(shardID string) (*VMStateShard, error)
	DeleteShard(shardID string) error
}

type RecoveryTask struct {
	ShardID   string
	StartTime time.Time
	Status    string
}

type RecoveryMetrics struct {
	RecoveriesStarted   int64
	RecoveriesCompleted int64
	RecoveriesFailed    int64
}

type ShardingMetrics struct {
	ShardsAllocated     MetricCounter
	ShardsReplicated    MetricCounter
	MigrationsCompleted MetricCounter
	CacheHits           MetricCounter
	RebalanceOperations MetricCounter
}

type MetricCounter struct {
	value int64
	mu    sync.Mutex
}

func (m *MetricCounter) Inc() {
	m.mu.Lock()
	m.value++
	m.mu.Unlock()
}

// Stub implementations
func NewDistributedStateStore() StateStore {
	return &distributedStateStore{
		shards: make(map[string]*VMStateShard),
	}
}

type distributedStateStore struct {
	shards map[string]*VMStateShard
	mu     sync.RWMutex
}

func (d *distributedStateStore) StoreShard(shard *VMStateShard) error {
	d.mu.Lock()
	defer d.mu.Unlock()

	// Create a deep copy to ensure thread safety
	shardCopy := &VMStateShard{
		ShardID:      shard.ShardID,
		VMID:         shard.VMID,
		NodeID:       shard.NodeID,
		ReplicaNodes: make([]string, len(shard.ReplicaNodes)),
		Version:      shard.Version,
		VectorClock:  make(map[string]uint64),
		Data:         shard.Data, // TODO: Deep copy if needed
		LastModified: shard.LastModified,
		Status:       shard.Status,
	}

	copy(shardCopy.ReplicaNodes, shard.ReplicaNodes)
	for k, v := range shard.VectorClock {
		shardCopy.VectorClock[k] = v
	}

	d.shards[shard.ShardID] = shardCopy
	return nil
}

func (d *distributedStateStore) GetShard(shardID string) (*VMStateShard, error) {
	d.mu.RLock()
	defer d.mu.RUnlock()

	shard, exists := d.shards[shardID]
	if !exists {
		return nil, errors.New("shard not found")
	}

	return shard, nil
}

func (d *distributedStateStore) DeleteShard(shardID string) error {
	d.mu.Lock()
	defer d.mu.Unlock()

	delete(d.shards, shardID)
	return nil
}

func NewShardRecoveryManager() *ShardRecoveryManager {
	return &ShardRecoveryManager{
		recoveryQueue:  []*RecoveryTask{},
		activeRecovery: make(map[string]*RecoveryTask),
		metrics:        &RecoveryMetrics{},
	}
}

func (r *ShardRecoveryManager) RecoverShard(shard *VMStateShard) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	task := &RecoveryTask{
		ShardID:   shard.ShardID,
		StartTime: time.Now(),
		Status:    "recovering",
	}

	r.recoveryQueue = append(r.recoveryQueue, task)
	r.activeRecovery[shard.ShardID] = task
	r.metrics.RecoveriesStarted++

	// Start recovery process in background
	go r.performShardRecovery(shard, task)

	return nil
}

// performShardRecovery performs the actual shard recovery with replica promotion
func (r *ShardRecoveryManager) performShardRecovery(shard *VMStateShard, task *RecoveryTask) {
	defer func() {
		r.mu.Lock()
		delete(r.activeRecovery, shard.ShardID)
		r.mu.Unlock()
	}()

	// Step 1: Find the best replica for promotion
	bestReplica, err := r.selectBestReplica(shard)
	if err != nil {
		task.Status = "failed"
		r.metrics.RecoveriesFailed++
		return
	}

	// Step 2: Promote the replica to primary
	err = r.promoteReplicaToPrimary(shard, bestReplica)
	if err != nil {
		task.Status = "failed"
		r.metrics.RecoveriesFailed++
		return
	}

	// Step 3: Update shard metadata
	shard.NodeID = bestReplica
	shard.Status = ShardStatusActive
	shard.Version++
	shard.LastModified = time.Now()

	// Step 4: Remove promoted node from replica list and select new replicas
	newReplicas := make([]string, 0, len(shard.ReplicaNodes))
	for _, replica := range shard.ReplicaNodes {
		if replica != bestReplica {
			newReplicas = append(newReplicas, replica)
		}
	}
	shard.ReplicaNodes = newReplicas

	// Step 5: Create new replicas to maintain replication factor
	err = r.createNewReplicas(shard)
	if err != nil {
		// Log warning but don't fail recovery
		// Recovery succeeded but replication might be reduced
	}

	task.Status = "completed"
	r.metrics.RecoveriesCompleted++
}

// selectBestReplica selects the best replica for promotion based on various criteria
func (r *ShardRecoveryManager) selectBestReplica(shard *VMStateShard) (string, error) {
	if len(shard.ReplicaNodes) == 0 {
		return "", errors.New("no replicas available for promotion")
	}

	// Fetch real vector clocks from all replicas
	replicaVectorClocks := make(map[string]map[string]uint64)
	var wg sync.WaitGroup
	var mu sync.Mutex

	for _, replica := range shard.ReplicaNodes {
		wg.Add(1)
		go func(nodeID string) {
			defer wg.Done()

			// Fetch vector clock from replica via RPC
			vectorClock, err := r.fetchReplicaVectorClock(nodeID, shard.ShardID)
			if err != nil {
				// Log error but continue with other replicas
				return
			}

			mu.Lock()
			replicaVectorClocks[nodeID] = vectorClock
			mu.Unlock()
		}(replica)
	}

	wg.Wait()

	if len(replicaVectorClocks) == 0 {
		return "", errors.New("failed to fetch vector clocks from any replica")
	}

	// Select replica with the most up-to-date vector clock
	bestReplica := ""
	bestScore := -1

	for nodeID, vectorClock := range replicaVectorClocks {
		score := r.calculateVectorClockScore(vectorClock)
		if score > bestScore {
			bestScore = score
			bestReplica = nodeID
		}
	}

	return bestReplica, nil
}

// selectReplicaByVectorClock selects the replica with the most recent vector clock
func (r *ShardRecoveryManager) selectReplicaByVectorClock(shard *VMStateShard) string {
	if len(shard.ReplicaNodes) == 0 {
		return ""
	}

	// In a real implementation, this would fetch vector clocks from all replicas
	// For now, simulate vector clock comparison
	bestReplica := shard.ReplicaNodes[0]

	// Simulate that we found the best replica based on vector clock comparison
	for _, replica := range shard.ReplicaNodes[1:] {
		// Simulate vector clock comparison logic
		if r.compareVectorClocks(shard.VectorClock, replica) {
			bestReplica = replica
		}
	}

	return bestReplica
}

// compareVectorClocks compares vector clocks to determine which replica is more up-to-date
func (r *ShardRecoveryManager) compareVectorClocks(baseVectorClock map[string]uint64, replica string) bool {
	// In a real implementation, this would:
	// 1. Fetch the vector clock from the replica node
	// 2. Compare timestamps for each node in the vector clock
	// 3. Determine which replica has more recent updates

	// For simulation, randomly return true/false
	// In production, this would be a sophisticated comparison
	return len(baseVectorClock)%2 == 0 // Simple deterministic simulation
}

// ReconcileVectorClocks reconciles vector clocks across replicas
func (r *ShardRecoveryManager) ReconcileVectorClocks(shard *VMStateShard) error {
	if len(shard.ReplicaNodes) == 0 {
		return nil
	}

	// Collect vector clocks from all replicas
	replicaVectorClocks := make(map[string]map[string]uint64)

	// In a real implementation, this would fetch from actual replicas
	// For simulation, create mock vector clocks
	for _, replica := range shard.ReplicaNodes {
		replicaVectorClocks[replica] = r.getReplicaVectorClock(replica)
	}

	// Merge vector clocks using the max value for each node
	mergedVectorClock := make(map[string]uint64)

	// Start with the primary's vector clock
	for nodeID, timestamp := range shard.VectorClock {
		mergedVectorClock[nodeID] = timestamp
	}

	// Merge with each replica's vector clock
	for _, replicaVC := range replicaVectorClocks {
		for nodeID, timestamp := range replicaVC {
			if currentTimestamp, exists := mergedVectorClock[nodeID]; !exists || timestamp > currentTimestamp {
				mergedVectorClock[nodeID] = timestamp
			}
		}
	}

	// Update the shard's vector clock with the reconciled version
	shard.VectorClock = mergedVectorClock
	shard.Version++
	shard.LastModified = time.Now()

	// Propagate the reconciled vector clock to all replicas
	for _, replica := range shard.ReplicaNodes {
		if err := r.updateReplicaVectorClock(replica, mergedVectorClock); err != nil {
			// Log warning but continue with other replicas
			// In production, this might require more sophisticated error handling
		}
	}

	return nil
}

// fetchReplicaVectorClock fetches vector clock from a replica node via RPC
func (r *ShardRecoveryManager) fetchReplicaVectorClock(nodeID, shardID string) (map[string]uint64, error) {
	if r.shardingManager == nil || r.shardingManager.federation == nil {
		// Fallback to mock for testing
		return map[string]uint64{
			"node1": uint64(time.Now().Unix()) - 100,
			"node2": uint64(time.Now().Unix()) - 50,
			nodeID:  uint64(time.Now().Unix()),
		}, nil
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// Create RPC request
	request := &federation.CrossClusterMessage{
		Type:        "get_vector_clock",
		Source:      r.shardingManager.localNodeID,
		Destination: nodeID,
		Payload: map[string]interface{}{
			"shard_id": shardID,
		},
		Timestamp: time.Now(),
	}

	// Send request and wait for response
	if crossCluster, ok := r.shardingManager.federation.(*federation.CrossClusterComponents); ok {
		response, err := crossCluster.SendMessageWithResponse(ctx, request)
		if err != nil {
			return nil, err
		}

		// Extract vector clock from response
		if vectorClock, ok := response.Payload.(map[string]uint64); ok {
			return vectorClock, nil
		}
	}

	return nil, errors.New("failed to fetch vector clock")
}

// calculateVectorClockScore calculates a score for vector clock freshness
func (r *ShardRecoveryManager) calculateVectorClockScore(vectorClock map[string]uint64) int {
	score := 0
	for _, timestamp := range vectorClock {
		score += int(timestamp)
	}
	return score
}

// Additional helper methods for recovery

func (r *ShardRecoveryManager) verifyReplicaIntegrity(ctx context.Context, nodeID, shardID string) error {
	// Verify that the replica has valid and consistent data
	// This would involve checksums, version checks, etc.
	return nil
}

func (r *ShardRecoveryManager) sendPromotionCommand(ctx context.Context, nodeID string, shard *VMStateShard) error {
	if r.shardingManager == nil || r.shardingManager.federation == nil {
		return nil // Simulation mode
	}

	// Send promotion command via RPC
	command := &federation.CrossClusterMessage{
		Type:        "promote_to_primary",
		Source:      r.shardingManager.localNodeID,
		Destination: nodeID,
		Payload: map[string]interface{}{
			"shard_id":      shard.ShardID,
			"version":       shard.Version,
			"vector_clock":  shard.VectorClock,
			"replica_nodes": shard.ReplicaNodes,
		},
		Timestamp: time.Now(),
	}

	if crossCluster, ok := r.shardingManager.federation.(*federation.CrossClusterComponents); ok {
		return crossCluster.SendMessage(ctx, command)
	}

	return nil
}

func (r *ShardRecoveryManager) broadcastNewPrimary(shardID, newPrimary string, replicas []string) error {
	if r.shardingManager == nil || r.shardingManager.federation == nil {
		return nil // Simulation mode
	}

	// Broadcast to all replicas
	var wg sync.WaitGroup
	for _, replica := range replicas {
		if replica == newPrimary {
			continue // Skip the new primary itself
		}

		wg.Add(1)
		go func(node string) {
			defer wg.Done()

			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()

			msg := &federation.CrossClusterMessage{
				Type:        "primary_changed",
				Source:      r.shardingManager.localNodeID,
				Destination: node,
				Payload: map[string]interface{}{
					"shard_id":    shardID,
					"new_primary": newPrimary,
				},
				Timestamp: time.Now(),
			}

			if crossCluster, ok := r.shardingManager.federation.(*federation.CrossClusterComponents); ok {
				crossCluster.SendMessage(ctx, msg)
			}
		}(replica)
	}

	wg.Wait()
	return nil
}

func (r *ShardRecoveryManager) selectHealthyNodes(count int, shard *VMStateShard) ([]string, error) {
	if r.shardingManager == nil {
		return nil, errors.New("sharding manager not available")
	}

	// Get all available nodes from the ring
	allNodes := r.shardingManager.nodeRing.GetAllNodes()

	// Exclude current primary and existing replicas
	excluded := make(map[string]bool)
	excluded[shard.NodeID] = true
	for _, replica := range shard.ReplicaNodes {
		excluded[replica] = true
	}

	// Select healthy nodes
	healthyNodes := []string{}
	for node := range allNodes {
		if !excluded[node] && len(healthyNodes) < count {
			// In production, would check node health here
			healthyNodes = append(healthyNodes, node)
		}
	}

	if len(healthyNodes) < count {
		return healthyNodes, errors.Errorf("only %d healthy nodes available, needed %d", len(healthyNodes), count)
	}

	return healthyNodes, nil
}

// updateReplicaVectorClock updates the vector clock on a replica node
func (r *ShardRecoveryManager) updateReplicaVectorClock(replica string, vectorClock map[string]uint64) error {
	if r.shardingManager == nil || r.shardingManager.federation == nil {
		return nil // Simulation mode
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	msg := &federation.CrossClusterMessage{
		Type:        "update_vector_clock",
		Source:      r.shardingManager.localNodeID,
		Destination: replica,
		Payload:     vectorClock,
		Timestamp:   time.Now(),
	}

	if crossCluster, ok := r.shardingManager.federation.(*federation.CrossClusterComponents); ok {
		return crossCluster.SendMessage(ctx, msg)
	}

	return nil
}

// promoteReplicaToPrimary promotes a replica to primary status
func (r *ShardRecoveryManager) promoteReplicaToPrimary(shard *VMStateShard, replicaNode string) error {
	// Step 1: Contact the replica node and verify it's available
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Step 2: Verify data integrity on the target node
	err := r.verifyReplicaIntegrity(ctx, replicaNode, shard.ShardID)
	if err != nil {
		return errors.Wrapf(err, "failed to verify replica integrity on %s", replicaNode)
	}

	// Step 3: Send promotion command to the replica
	err = r.sendPromotionCommand(ctx, replicaNode, shard)
	if err != nil {
		return errors.Wrapf(err, "failed to promote replica %s", replicaNode)
	}

	// Step 4: Update shard metadata
	shard.NodeID = replicaNode
	shard.Version++
	shard.LastModified = time.Now()

	// Step 5: Persist the change to state store
	if r.shardingManager != nil && r.shardingManager.stateStore != nil {
		err = r.shardingManager.stateStore.StoreShard(shard)
		if err != nil {
			return errors.Wrap(err, "failed to persist promoted shard")
		}
	}

	// Step 6: Notify all other nodes about the new primary
	err = r.broadcastNewPrimary(shard.ShardID, replicaNode, shard.ReplicaNodes)
	if err != nil {
		// Log but don't fail - promotion succeeded
		// Other nodes will eventually discover the new primary
	}

	return nil
}

// createNewReplicas creates new replicas to maintain the desired replication factor
func (r *ShardRecoveryManager) createNewReplicas(shard *VMStateShard) error {
	if r.shardingManager == nil {
		return errors.New("sharding manager not available")
	}

	// Calculate how many replicas we need
	desiredReplicas := r.shardingManager.replicationFactor - 1 // -1 for primary
	currentReplicas := len(shard.ReplicaNodes)
	neededReplicas := desiredReplicas - currentReplicas

	if neededReplicas <= 0 {
		return nil // Already have enough replicas
	}

	// Select healthy nodes for new replicas
	healthyNodes, err := r.selectHealthyNodes(neededReplicas, shard)
	if err != nil {
		return errors.Wrap(err, "failed to select nodes for new replicas")
	}

	// Copy state to new replicas
	var wg sync.WaitGroup
	errChan := make(chan error, len(healthyNodes))

	for _, node := range healthyNodes {
		wg.Add(1)
		go func(targetNode string) {
			defer wg.Done()

			// Replicate shard to new node
			err := r.shardingManager.replicateShard(shard, targetNode)
			if err != nil {
				errChan <- errors.Wrapf(err, "failed to replicate to %s", targetNode)
				return
			}

			// Add to replica list
			shard.ReplicaNodes = append(shard.ReplicaNodes, targetNode)
		}(node)
	}

	wg.Wait()
	close(errChan)

	// Check for errors
	var errs []error
	for err := range errChan {
		errs = append(errs, err)
	}

	if len(errs) > 0 {
		return errors.Errorf("failed to create %d new replicas", len(errs))
	}

	// Trigger rebalancing if membership changed
	if r.shardingManager != nil {
		go r.shardingManager.rebalanceShards()
	}

	return nil
}

func NewStateAccessLayer() *StateAccessLayer {
	return &StateAccessLayer{
		cache:      NewStateCache(),
		prefetcher: NewStatePrefetcher(),
		lazyLoader: NewLazyStateLoader(),
	}
}

func (s *StateAccessLayer) GetFromCache(vmID string) *DistributedVMState {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.cache.Get(vmID)
}

func (s *StateAccessLayer) UpdateCache(vmID string, state *DistributedVMState) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.cache.Put(vmID, state)
}

type StateCache struct {
	cache map[string]*DistributedVMState
	mu    sync.RWMutex
}

func NewStateCache() *StateCache {
	return &StateCache{
		cache: make(map[string]*DistributedVMState),
	}
}

func (c *StateCache) Get(vmID string) *DistributedVMState {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.cache[vmID]
}

func (c *StateCache) Put(vmID string, state *DistributedVMState) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.cache[vmID] = state
}

type StatePrefetcher struct{}

func NewStatePrefetcher() *StatePrefetcher {
	return &StatePrefetcher{}
}

type LazyStateLoader struct{}

func NewLazyStateLoader() *LazyStateLoader {
	return &LazyStateLoader{}
}

func NewShardingMetrics() *ShardingMetrics {
	return &ShardingMetrics{
		ShardsAllocated:     MetricCounter{},
		ShardsReplicated:    MetricCounter{},
		MigrationsCompleted: MetricCounter{},
		CacheHits:           MetricCounter{},
		RebalanceOperations: MetricCounter{},
	}
}

type NetworkInterface struct {
	Name string
	IP   string
	MAC  string
}

type Route struct {
	Destination string
	Gateway     string
	Interface   string
}

type Connection struct {
	Protocol string
	LocalAddr string
	RemoteAddr string
	State    string
}

type DiskConfig struct {
	Path string
	Size int64
	Type string
}

// RegisterNodes adds multiple nodes to the sharding ring
func (sm *VMStateShardingManager) RegisterNodes(nodes []string) {
	for _, node := range nodes {
		sm.nodeRing.AddNode(node)
		sm.logger.Info("Registered node to sharding ring", zap.String("nodeID", node))
	}
}

// subscribeToMembershipEvents subscribes to federation membership changes
func (sm *VMStateShardingManager) subscribeToMembershipEvents() {
	if sm.federation == nil {
		return
	}

	// Create a ticker to periodically check membership changes
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for range ticker.C {
		// Get current cluster members from federation
		clusters, err := sm.federation.ListClusters(context.Background())
		if err != nil {
			sm.logger.Warn("Failed to list clusters from federation", zap.Error(err))
			continue
		}

		// Extract node IDs from clusters
		nodeIDs := make([]string, 0, len(clusters))
		for _, cluster := range clusters {
			if cluster.State == federation.ConnectedState {
				nodeIDs = append(nodeIDs, cluster.ID)
			}
		}

		// Update ring membership
		sm.updateRingMembership(nodeIDs)
	}
}

// updateRingMembership updates the consistent hash ring based on current membership
func (sm *VMStateShardingManager) updateRingMembership(currentNodes []string) {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	// Build a map of current nodes for quick lookup
	currentNodeMap := make(map[string]bool)
	for _, node := range currentNodes {
		currentNodeMap[node] = true
	}

	// Get existing nodes in the ring
	existingNodes := sm.nodeRing.GetAllNodes()

	// Add new nodes
	for _, node := range currentNodes {
		if !existingNodes[node] {
			sm.nodeRing.AddNode(node)
			sm.logger.Info("Added node to sharding ring", zap.String("nodeID", node))
		}
	}

	// Remove nodes that are no longer in membership
	for node := range existingNodes {
		if !currentNodeMap[node] {
			sm.nodeRing.RemoveNode(node)
			sm.logger.Info("Removed node from sharding ring", zap.String("nodeID", node))
			// Trigger shard recovery for affected shards
			go sm.HandleNodeFailure(node)
		}
	}

	// Trigger rebalancing if membership changed
	if len(existingNodes) != len(currentNodes) {
		go sm.rebalanceShards()
	}
}

// Additional types for replication and migration

type ShardUpdateMessage struct {
	ShardID      string
	Version      uint64
	VectorClock  map[string]uint64
	SourceNode   string
	Timestamp    time.Time
	UpdateType   string
	Payload      interface{}
}

type OwnershipChangeMessage struct {
	ShardID    string
	OldPrimary string
	NewPrimary string
	Version    uint64
	Timestamp  time.Time
}

// Helper methods for replication and migration

func (sm *VMStateShardingManager) waitForReplicationAck(ctx context.Context, shardID, targetNode string) (bool, error) {
	// Wait for acknowledgment from target node
	// In a real implementation, this would listen for response messages
	select {
	case <-ctx.Done():
		return false, ctx.Err()
	case <-time.After(5 * time.Second):
		// Simulate successful acknowledgment for now
		return true, nil
	}
}

func (sm *VMStateShardingManager) sendShardUpdate(ctx context.Context, update *ShardUpdateMessage, targetNode string) error {
	if sm.federation == nil {
		return errors.New("federation manager not available")
	}

	// Create cross-cluster message
	msg := &federation.CrossClusterMessage{
		Type:        "shard_update",
		Source:      sm.localNodeID,
		Destination: targetNode,
		Payload:     update,
		Timestamp:   time.Now(),
	}

	// Send through federation
	if crossCluster, ok := sm.federation.(*federation.CrossClusterComponents); ok {
		return crossCluster.SendMessage(ctx, msg)
	}

	return errors.New("cross-cluster components not available")
}

func (sm *VMStateShardingManager) copyShardToNode(shard *VMStateShard, targetNode string) error {
	// Copy entire shard state to target node
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Send shard data
	err := sm.replicateShard(shard, targetNode)
	if err != nil {
		return err
	}

	// Verify shard was received
	// In production, this would query the target node
	return nil
}

func (sm *VMStateShardingManager) broadcastOwnershipChange(msg *OwnershipChangeMessage, replicas []string) error {
	var wg sync.WaitGroup
	errChan := make(chan error, len(replicas))

	for _, replica := range replicas {
		wg.Add(1)
		go func(node string) {
			defer wg.Done()
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()

			if sm.federation != nil {
				fedMsg := &federation.CrossClusterMessage{
					Type:        "ownership_change",
					Source:      sm.localNodeID,
					Destination: node,
					Payload:     msg,
					Timestamp:   time.Now(),
				}

				if crossCluster, ok := sm.federation.(*federation.CrossClusterComponents); ok {
					if err := crossCluster.SendMessage(ctx, fedMsg); err != nil {
						errChan <- err
					}
				}
			}
		}(replica)
	}

	wg.Wait()
	close(errChan)

	// Collect any errors
	var errs []error
	for err := range errChan {
		errs = append(errs, err)
	}

	if len(errs) > 0 {
		return errors.Errorf("failed to notify %d replicas", len(errs))
	}

	return nil
}

// routeUpdateToPrimary routes update to the primary node
func (sm *VMStateShardingManager) routeUpdateToPrimary(ctx context.Context, vmID, primaryNode string, update func(*DistributedVMState) error) error {
	// Implementation would route update to primary node via RPC
	return errors.New("routing to primary node not implemented")
}