package storage

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"
)

// NodeInfo represents information about a storage node
// This is a copy of discovery.NodeInfo to avoid import issues
type NodeInfo struct {
	// Unique identifier for the node
	ID string `json:"id"`

	// Human-readable name for the node
	Name string `json:"name"`

	// Role of the node (manager, worker)
	Role string `json:"role"`

	// Network address of the node
	Address string `json:"address"`

	// Port the node is listening on
	Port int `json:"port"`

	// Additional tags/metadata for the node
	Tags map[string]string `json:"tags,omitempty"`

	// When the node joined the cluster
	JoinedAt time.Time `json:"joined_at"`

	// When the node was last seen
	LastSeen time.Time `json:"last_seen"`

	// Whether the node is available
	Available bool `json:"available"`
}

// DistributedStorageConfig contains configuration for distributed storage
type DistributedStorageConfig struct {
	// Root directory for storage
	RootDir string `json:"root_dir"`

	// Maximum storage capacity in bytes (0 = unlimited)
	MaxCapacity int64 `json:"max_capacity"`

	// Default replication factor
	DefaultReplicationFactor int `json:"default_replication_factor"`

	// Whether to enable encryption by default
	DefaultEncryption bool `json:"default_encryption"`

	// Whether to enable data sharding
	EnableSharding bool `json:"enable_sharding"`

	// Size of shards in bytes
	ShardSize int64 `json:"shard_size"`

	// Consistency protocol to use (eventual, strong, causal)
	ConsistencyProtocol string `json:"consistency_protocol"`

	// Number of storage nodes to query for healing operations
	HealingQueryCount int `json:"healing_query_count"`

	// Interval for health checks
	HealthCheckInterval time.Duration `json:"health_check_interval"`

	// Interval for data healing
	HealingInterval time.Duration `json:"healing_interval"`

	// Whether to enable synchronous replication
	SynchronousReplication bool `json:"synchronous_replication"`
}

// DefaultDistributedStorageConfig returns a default configuration
func DefaultDistributedStorageConfig() DistributedStorageConfig {
	return DistributedStorageConfig{
		RootDir:                  "/var/lib/novacron/distributed",
		MaxCapacity:              0, // Unlimited
		DefaultReplicationFactor: 3,
		DefaultEncryption:        false,
		EnableSharding:           true,
		ShardSize:                64 * 1024 * 1024, // 64 MB
		ConsistencyProtocol:      "eventual",
		HealingQueryCount:        5,
		HealthCheckInterval:      1 * time.Minute,
		HealingInterval:          1 * time.Hour,
		SynchronousReplication:   false,
	}
}

// VolumeShardInfo contains information about a shard of a distributed volume
type VolumeShardInfo struct {
	// Unique identifier for the shard
	ID string `json:"id"`

	// Index of the shard within the volume
	Index int `json:"index"`

	// Size of the shard in bytes
	Size int64 `json:"size"`

	// Checksum of the shard data
	Checksum string `json:"checksum"`

	// Nodes where this shard is stored
	NodeIDs []string `json:"node_ids"`

	// Whether this shard needs healing
	NeedsHealing bool `json:"needs_healing"`

	// When the shard was last verified
	LastVerified time.Time `json:"last_verified"`
}

// DistributedVolumeInfo extends VolumeInfo with distributed-specific data
type DistributedVolumeInfo struct {
	// Base volume information
	VolumeInfo

	// Information about shards
	Shards []VolumeShardInfo `json:"shards"`

	// Total number of shards
	ShardCount int `json:"shard_count"`

	// Number of nodes storing this volume
	NodeCount int `json:"node_count"`

	// Consistency level for the volume
	ConsistencyLevel string `json:"consistency_level"`

	// Replication policy (sync or async)
	ReplicationPolicy string `json:"replication_policy"`

	// Name of the placement group
	PlacementGroup string `json:"placement_group"`

	// Health status as a percentage
	HealthPercentage float64 `json:"health_percentage"`

	// Last time the volume was healed
	LastHealed time.Time `json:"last_healed"`
}

// DistributedVolume represents a distributed storage volume
type DistributedVolume struct {
	// Base volume info from the existing system
	baseVolume *Volume

	// Extended information for distributed volumes
	DistInfo DistributedVolumeInfo `json:"dist_info"`

	// Mutex for synchronizing access to the volume
	mu sync.RWMutex
}

// ShardPlacementStrategy determines how shards are placed across nodes
type ShardPlacementStrategy string

const (
	// ShardPlacementRandom places shards randomly across nodes
	ShardPlacementRandom ShardPlacementStrategy = "random"

	// ShardPlacementBalanced tries to balance shards evenly across nodes
	ShardPlacementBalanced ShardPlacementStrategy = "balanced"

	// ShardPlacementLocalityAware places related shards on the same node
	ShardPlacementLocalityAware ShardPlacementStrategy = "locality_aware"

	// ShardPlacementZoneAware places shards across availability zones
	ShardPlacementZoneAware ShardPlacementStrategy = "zone_aware"
)

// ConsistencyLevel defines the consistency guarantees for distributed volumes
type ConsistencyLevel string

const (
	// ConsistencyEventual provides eventual consistency
	ConsistencyEventual ConsistencyLevel = "eventual"

	// ConsistencyStrong provides strong consistency
	ConsistencyStrong ConsistencyLevel = "strong"

	// ConsistencyCausal provides causal consistency
	ConsistencyCausal ConsistencyLevel = "causal"
)

// ErrShardNotFound indicates a shard was not found
var ErrShardNotFound = errors.New("shard not found")

// ErrNotEnoughReplicas indicates there are not enough replicas
var ErrNotEnoughReplicas = errors.New("not enough replicas")

// ErrInconsistentState indicates the distributed state is inconsistent
var ErrInconsistentState = errors.New("inconsistent distributed state")

// DistributedStorageService extends the StorageManager with distributed capabilities
type DistributedStorageService struct {
	// Base storage manager
	baseManager *StorageManager

	// Configuration for distributed storage
	config DistributedStorageConfig

	// Distributed volumes (volumeID -> DistributedVolume)
	distVolumes map[string]*DistributedVolume

	// Mutex for synchronizing access to distVolumes
	volMutex sync.RWMutex

	// Known storage nodes (nodeID -> NodeInfo)
	nodes map[string]NodeInfo

	// Mutex for synchronizing access to nodes
	nodeMutex sync.RWMutex

	// Shard placement strategy
	placementStrategy ShardPlacementStrategy

	// Context for cancellation
	ctx    context.Context
	cancel context.CancelFunc
}

// NewDistributedStorageService creates a new distributed storage service
func NewDistributedStorageService(
	baseManager *StorageManager,
	config DistributedStorageConfig,
) (*DistributedStorageService, error) {
	ctx, cancel := context.WithCancel(context.Background())

	service := &DistributedStorageService{
		baseManager:       baseManager,
		config:            config,
		distVolumes:       make(map[string]*DistributedVolume),
		nodes:             make(map[string]NodeInfo),
		placementStrategy: ShardPlacementBalanced,
		ctx:               ctx,
		cancel:            cancel,
	}

	return service, nil
}

// Start starts the distributed storage service
func (s *DistributedStorageService) Start() error {
	log.Println("Starting distributed storage service")

	// Create the base directory for distributed storage
	if err := os.MkdirAll(s.config.RootDir, 0755); err != nil {
		return fmt.Errorf("failed to create distributed storage directory: %w", err)
	}

	// Start health checking
	go s.runHealthChecks()

	// Start data healing
	go s.runDataHealing()

	return nil
}

// Stop stops the distributed storage service
func (s *DistributedStorageService) Stop() error {
	log.Println("Stopping distributed storage service")
	s.cancel()
	return nil
}

// AddNode adds or updates a storage node
func (s *DistributedStorageService) AddNode(node NodeInfo) {
	s.nodeMutex.Lock()
	defer s.nodeMutex.Unlock()

	s.nodes[node.ID] = node
	log.Printf("Added/updated storage node: %s (%s)", node.Name, node.ID)
}

// RemoveNode removes a storage node
func (s *DistributedStorageService) RemoveNode(nodeID string) {
	s.nodeMutex.Lock()
	defer s.nodeMutex.Unlock()

	if _, exists := s.nodes[nodeID]; exists {
		delete(s.nodes, nodeID)
		log.Printf("Removed storage node: %s", nodeID)

		// Trigger rebalancing for volumes that had data on this node
		go s.rebalanceForNode(nodeID)
	}
}

// GetAvailableNodes returns a list of available storage nodes
func (s *DistributedStorageService) GetAvailableNodes() []NodeInfo {
	s.nodeMutex.RLock()
	defer s.nodeMutex.RUnlock()

	nodes := make([]NodeInfo, 0, len(s.nodes))
	for _, node := range s.nodes {
		if node.Available {
			nodes = append(nodes, node)
		}
	}

	return nodes
}

// CreateDistributedVolume creates a new distributed volume
func (s *DistributedStorageService) CreateDistributedVolume(
	ctx context.Context,
	spec VolumeSpec,
	replicationFactor int,
) (*Volume, error) {
	// Modify the spec for distributed storage
	if spec.Type == "" {
		spec.Type = VolumeTypeCeph // Default to Ceph for distributed volumes
	}

	// Set appropriate options for the underlying storage
	if spec.Options == nil {
		spec.Options = make(map[string]string)
	}
	spec.Options["distributed"] = "true"
	spec.Options["replication_factor"] = fmt.Sprintf("%d", replicationFactor)

	// Create the base volume
	volume, err := s.baseManager.CreateVolume(ctx, spec)
	if err != nil {
		return nil, fmt.Errorf("failed to create base volume: %w", err)
	}

	// Add distributed information
	distVolume := &DistributedVolume{
		baseVolume: volume,
		DistInfo: DistributedVolumeInfo{
			ShardCount:        calculateShardCount(int64(volume.SizeMB)*1024*1024, s.config.ShardSize),
			NodeCount:         0,
			ConsistencyLevel:  s.config.ConsistencyProtocol,
			ReplicationPolicy: s.getReplicationPolicy(),
			PlacementGroup:    "", // Will be assigned later
			HealthPercentage:  100.0,
			LastHealed:        time.Now(),
		},
	}

	// Initialize shards
	shards := make([]VolumeShardInfo, distVolume.DistInfo.ShardCount)
	for i := 0; i < distVolume.DistInfo.ShardCount; i++ {
		shards[i] = VolumeShardInfo{
			ID:           generateShardID(volume.ID, i),
			Index:        i,
			Size:         s.config.ShardSize,
			Checksum:     "",
			NodeIDs:      []string{},
			NeedsHealing: false,
			LastVerified: time.Now(),
		}
	}
	distVolume.DistInfo.Shards = shards

	// Store the distributed volume
	s.volMutex.Lock()
	s.distVolumes[volume.ID] = distVolume
	s.volMutex.Unlock()

	// Place the shards on nodes
	if err := s.placeShards(ctx, distVolume, replicationFactor); err != nil {
		// If placement fails, delete the volume and return error
		s.baseManager.DeleteVolume(ctx, volume.ID)
		s.volMutex.Lock()
		delete(s.distVolumes, volume.ID)
		s.volMutex.Unlock()
		return nil, fmt.Errorf("failed to place shards: %w", err)
	}

	log.Printf("Created distributed volume: %s (%s) with %d shards and replication factor %d",
		volume.Name, volume.ID, distVolume.DistInfo.ShardCount, replicationFactor)

	return volume, nil
}

// GetDistributedVolume returns a distributed volume by ID
func (s *DistributedStorageService) GetDistributedVolume(
	ctx context.Context,
	volumeID string,
) (*DistributedVolume, error) {
	s.volMutex.RLock()
	defer s.volMutex.RUnlock()

	distVolume, exists := s.distVolumes[volumeID]
	if !exists {
		// Check if it's a base volume that we don't know about yet
		volume, err := s.baseManager.GetVolume(volumeID)
		if err != nil {
			return nil, err
		}

		// Check if it's marked as distributed
		if volume.Options == nil || volume.Options["distributed"] != "true" {
			return nil, fmt.Errorf("volume %s is not a distributed volume", volumeID)
		}

		// It's a distributed volume we haven't loaded yet
		// In a real implementation, this would load distributed metadata
		return nil, fmt.Errorf("distributed metadata for volume %s not loaded", volumeID)
	}

	return distVolume, nil
}

// ReadShard reads data from a specific shard
func (s *DistributedStorageService) ReadShard(
	ctx context.Context,
	volumeID string,
	shardIndex int,
) ([]byte, error) {
	// Get the distributed volume
	distVolume, err := s.GetDistributedVolume(ctx, volumeID)
	if err != nil {
		return nil, err
	}

	distVolume.mu.RLock()
	defer distVolume.mu.RUnlock()

	// Check shard index
	if shardIndex < 0 || shardIndex >= len(distVolume.DistInfo.Shards) {
		return nil, fmt.Errorf("invalid shard index: %d", shardIndex)
	}

	shard := distVolume.DistInfo.Shards[shardIndex]

	// Check if the shard has any replicas
	if len(shard.NodeIDs) == 0 {
		return nil, ErrShardNotFound
	}

	// Try to read from each node until successful
	var lastErr error
	for _, nodeID := range shard.NodeIDs {
		// In a real implementation, this would connect to the node and read the shard
		// For now, simulate reading from the local filesystem
		shardPath := filepath.Join(s.config.RootDir, volumeID, fmt.Sprintf("shard_%d", shardIndex))
		data, err := os.ReadFile(shardPath)
		if err == nil {
			log.Printf("Read shard %d from node %s", shardIndex, nodeID)
			return data, nil
		}
		lastErr = err
	}

	return nil, fmt.Errorf("failed to read shard: %w", lastErr)
}

// WriteShard writes data to a specific shard
func (s *DistributedStorageService) WriteShard(
	ctx context.Context,
	volumeID string,
	shardIndex int,
	data []byte,
) error {
	// Get the distributed volume
	distVolume, err := s.GetDistributedVolume(ctx, volumeID)
	if err != nil {
		return err
	}

	distVolume.mu.Lock()
	defer distVolume.mu.Unlock()

	// Check shard index
	if shardIndex < 0 || shardIndex >= len(distVolume.DistInfo.Shards) {
		return fmt.Errorf("invalid shard index: %d", shardIndex)
	}

	shard := &distVolume.DistInfo.Shards[shardIndex]

	// Check if the shard has any replicas
	if len(shard.NodeIDs) == 0 {
		return ErrShardNotFound
	}

	// Calculate checksum
	checksum := calculateChecksum(data)
	shard.Checksum = checksum

	// Write to all replicas
	var successCount int
	var lastErr error
	for _, nodeID := range shard.NodeIDs {
		// In a real implementation, this would connect to the node and write the shard
		// For now, simulate writing to the local filesystem
		shardPath := filepath.Join(s.config.RootDir, volumeID, fmt.Sprintf("shard_%d", shardIndex))
		if err := os.MkdirAll(filepath.Dir(shardPath), 0755); err != nil {
			lastErr = err
			continue
		}
		if err := os.WriteFile(shardPath, data, 0644); err != nil {
			lastErr = err
			continue
		}
		log.Printf("Wrote shard %d to node %s", shardIndex, nodeID)
		successCount++

		// For synchronous replication, break after first success
		if !s.config.SynchronousReplication {
			break
		}
	}

	// Check if we wrote to enough replicas
	requiredSuccess := 1
	if s.config.SynchronousReplication {
		// For synchronous replication, we need to write to all replicas
		requiredSuccess = len(shard.NodeIDs)
	}

	if successCount < requiredSuccess {
		return fmt.Errorf("failed to write to enough replicas: %w", lastErr)
	}

	return nil
}

// RepairVolume attempts to repair a distributed volume
func (s *DistributedStorageService) RepairVolume(
	ctx context.Context,
	volumeID string,
) error {
	// Get the distributed volume
	distVolume, err := s.GetDistributedVolume(ctx, volumeID)
	if err != nil {
		return err
	}

	distVolume.mu.Lock()
	defer distVolume.mu.Unlock()

	// Iterate through all shards
	for i := range distVolume.DistInfo.Shards {
		shard := &distVolume.DistInfo.Shards[i]
		if shard.NeedsHealing {
			if err := s.repairShard(ctx, distVolume, i); err != nil {
				return fmt.Errorf("failed to repair shard %d: %w", i, err)
			}
		}
	}

	// Update the health percentage
	totalShards := len(distVolume.DistInfo.Shards)
	healthyShards := 0
	for _, shard := range distVolume.DistInfo.Shards {
		if !shard.NeedsHealing {
			healthyShards++
		}
	}
	distVolume.DistInfo.HealthPercentage = float64(healthyShards) * 100.0 / float64(totalShards)
	distVolume.DistInfo.LastHealed = time.Now()

	log.Printf("Repaired volume %s, health now at %.2f%%", volumeID, distVolume.DistInfo.HealthPercentage)

	return nil
}

// RebalanceVolume redistributes shards across available nodes
func (s *DistributedStorageService) RebalanceVolume(
	ctx context.Context,
	volumeID string,
) error {
	// Get the distributed volume
	distVolume, err := s.GetDistributedVolume(ctx, volumeID)
	if err != nil {
		return err
	}

	// Get available nodes
	availableNodes := s.GetAvailableNodes()
	if len(availableNodes) < s.config.DefaultReplicationFactor {
		return fmt.Errorf("not enough available nodes for replication factor %d", s.config.DefaultReplicationFactor)
	}

	distVolume.mu.Lock()
	defer distVolume.mu.Unlock()

	// Calculate target distribution
	// In a real implementation, this would be more sophisticated
	targetNodesPerShard := min(len(availableNodes), s.config.DefaultReplicationFactor)

	// Rebalance each shard
	for i := range distVolume.DistInfo.Shards {
		shard := &distVolume.DistInfo.Shards[i]

		// Calculate how many nodes we need to add or remove
		currentNodes := len(shard.NodeIDs)
		if currentNodes < targetNodesPerShard {
			// Need to add nodes
			nodesToAdd := targetNodesPerShard - currentNodes
			candidateNodes := availableNodes

			// Filter out nodes that already have this shard
			var newCandidates []NodeInfo
			for _, node := range candidateNodes {
				if !contains(shard.NodeIDs, node.ID) {
					newCandidates = append(newCandidates, node)
				}
			}

			// Add up to nodesToAdd new nodes
			for i := 0; i < nodesToAdd && i < len(newCandidates); i++ {
				// In a real implementation, we would copy the shard to this node
				shard.NodeIDs = append(shard.NodeIDs, newCandidates[i].ID)
			}
		} else if currentNodes > targetNodesPerShard {
			// Need to remove nodes
			nodesToRemove := currentNodes - targetNodesPerShard

			// Sort nodes by some criteria (e.g., health, load)
			// For now, just remove the last ones
			shard.NodeIDs = shard.NodeIDs[:currentNodes-nodesToRemove]
		}
	}

	log.Printf("Rebalanced volume %s across %d nodes", volumeID, len(availableNodes))

	return nil
}

// repairShard attempts to repair a single shard
func (s *DistributedStorageService) repairShard(
	ctx context.Context,
	volume *DistributedVolume,
	shardIndex int,
) error {
	shard := &volume.DistInfo.Shards[shardIndex]

	// Get available nodes
	availableNodes := s.GetAvailableNodes()
	if len(availableNodes) == 0 {
		return fmt.Errorf("no available nodes to repair shard")
	}

	// Find the best copy of the shard
	_, err := s.findBestShardCopy(ctx, volume.baseVolume.ID, shardIndex)
	if err != nil {
		return fmt.Errorf("no valid copy of shard found: %w", err)
	}

	// Get nodes that don't have this shard
	var candidateNodes []NodeInfo
	for _, node := range availableNodes {
		if !contains(shard.NodeIDs, node.ID) {
			candidateNodes = append(candidateNodes, node)
		}
	}

	// Calculate how many new copies we need
	targetReplicas := s.config.DefaultReplicationFactor
	currentReplicas := len(shard.NodeIDs)
	neededReplicas := targetReplicas - currentReplicas

	// Copy to new nodes
	for i := 0; i < neededReplicas && i < len(candidateNodes); i++ {
		// In a real implementation, this would copy the shard to the node
		nodeID := candidateNodes[i].ID
		log.Printf("Copying shard %d of volume %s to node %s",
			shardIndex, volume.baseVolume.ID, nodeID)

		// Add node to shard's node list
		shard.NodeIDs = append(shard.NodeIDs, nodeID)
	}

	// Update shard info
	shard.NeedsHealing = false
	shard.LastVerified = time.Now()

	return nil
}

// findBestShardCopy finds the best copy of a shard across all replicas
func (s *DistributedStorageService) findBestShardCopy(
	ctx context.Context,
	volumeID string,
	shardIndex int,
) ([]byte, error) {
	// Get the distributed volume
	distVolume, err := s.GetDistributedVolume(ctx, volumeID)
	if err != nil {
		return nil, err
	}

	distVolume.mu.RLock()
	defer distVolume.mu.RUnlock()

	// Check shard index
	if shardIndex < 0 || shardIndex >= len(distVolume.DistInfo.Shards) {
		return nil, fmt.Errorf("invalid shard index: %d", shardIndex)
	}

	shard := distVolume.DistInfo.Shards[shardIndex]

	// Check if the shard has any replicas
	if len(shard.NodeIDs) == 0 {
		return nil, ErrShardNotFound
	}

	// In a real implementation, this would retrieve copies from multiple nodes
	// and perform validation to find the best one

	// For now, just return the first copy we find
	for _, nodeID := range shard.NodeIDs {
		// Simulate reading from the node
		shardPath := filepath.Join(s.config.RootDir, volumeID, fmt.Sprintf("shard_%d", shardIndex))
		data, err := os.ReadFile(shardPath)
		if err == nil {
			log.Printf("Found valid copy of shard %d on node %s", shardIndex, nodeID)
			return data, nil
		}
	}

	return nil, ErrShardNotFound
}

// rebalanceForNode handles rebalancing when a node is removed
func (s *DistributedStorageService) rebalanceForNode(nodeID string) {
	// Get all distributed volumes
	s.volMutex.RLock()
	volumeIDs := make([]string, 0, len(s.distVolumes))
	for id := range s.distVolumes {
		volumeIDs = append(volumeIDs, id)
	}
	s.volMutex.RUnlock()

	// For each volume, check if it has data on the removed node
	for _, volumeID := range volumeIDs {
		// We use a separate context for each volume to avoid cancellation affecting others
		ctx := context.TODO()

		// Get the volume
		distVolume, err := s.GetDistributedVolume(ctx, volumeID)
		if err != nil {
			log.Printf("Error getting volume %s for rebalancing: %v", volumeID, err)
			continue
		}

		// Check if any shards were on this node
		needsRebalance := false
		distVolume.mu.RLock()
		for _, shard := range distVolume.DistInfo.Shards {
			if contains(shard.NodeIDs, nodeID) {
				needsRebalance = true
				break
			}
		}
		distVolume.mu.RUnlock()

		if needsRebalance {
			log.Printf("Rebalancing volume %s due to node %s removal", volumeID, nodeID)
			if err := s.RebalanceVolume(ctx, volumeID); err != nil {
				log.Printf("Error rebalancing volume %s: %v", volumeID, err)
			}
		}
	}
}

// runHealthChecks periodically checks the health of all shards
func (s *DistributedStorageService) runHealthChecks() {
	ticker := time.NewTicker(s.config.HealthCheckInterval)
	defer ticker.Stop()

	for {
		select {
		case <-s.ctx.Done():
			return
		case <-ticker.C:
			s.checkAllVolumesHealth()
		}
	}
}

// checkAllVolumesHealth checks the health of all shards in all volumes
func (s *DistributedStorageService) checkAllVolumesHealth() {
	// Get all distributed volumes
	s.volMutex.RLock()
	volumeIDs := make([]string, 0, len(s.distVolumes))
	for id := range s.distVolumes {
		volumeIDs = append(volumeIDs, id)
	}
	s.volMutex.RUnlock()

	// Check each volume
	for _, volumeID := range volumeIDs {
		ctx := context.TODO()
		if err := s.checkVolumeHealth(ctx, volumeID); err != nil {
			log.Printf("Error checking health of volume %s: %v", volumeID, err)
		}
	}
}

// checkVolumeHealth checks the health of all shards in a volume
func (s *DistributedStorageService) checkVolumeHealth(ctx context.Context, volumeID string) error {
	// Get the distributed volume
	distVolume, err := s.GetDistributedVolume(ctx, volumeID)
	if err != nil {
		return err
	}

	distVolume.mu.Lock()
	defer distVolume.mu.Unlock()

	// Check each shard
	for i := range distVolume.DistInfo.Shards {
		shard := &distVolume.DistInfo.Shards[i]

		// Check if the shard has enough replicas
		if len(shard.NodeIDs) < s.config.DefaultReplicationFactor {
			shard.NeedsHealing = true
			continue
		}

		// Check if the shard is accessible on each node
		for _, nodeID := range shard.NodeIDs {
			// In a real implementation, this would check if the node is healthy
			// and if the shard is accessible
			// For now, just use the node availability from our node list
			s.nodeMutex.RLock()
			node, exists := s.nodes[nodeID]
			s.nodeMutex.RUnlock()

			if !exists || !node.Available {
				shard.NeedsHealing = true
				break
			}
		}
	}

	// Update the health percentage
	totalShards := len(distVolume.DistInfo.Shards)
	healthyShards := 0
	for _, shard := range distVolume.DistInfo.Shards {
		if !shard.NeedsHealing {
			healthyShards++
		}
	}
	distVolume.DistInfo.HealthPercentage = float64(healthyShards) * 100.0 / float64(totalShards)

	return nil
}

// runDataHealing periodically heals unhealthy volumes
func (s *DistributedStorageService) runDataHealing() {
	ticker := time.NewTicker(s.config.HealingInterval)
	defer ticker.Stop()

	for {
		select {
		case <-s.ctx.Done():
			return
		case <-ticker.C:
			s.healAllVolumes()
		}
	}
}

// healAllVolumes heals all volumes that need it
func (s *DistributedStorageService) healAllVolumes() {
	// Get all distributed volumes
	s.volMutex.RLock()
	volumeIDs := make([]string, 0, len(s.distVolumes))
	for id := range s.distVolumes {
		volumeIDs = append(volumeIDs, id)
	}
	s.volMutex.RUnlock()

	// Check each volume
	for _, volumeID := range volumeIDs {
		ctx := context.TODO()

		// Get the volume
		distVolume, err := s.GetDistributedVolume(ctx, volumeID)
		if err != nil {
			log.Printf("Error getting volume %s for healing: %v", volumeID, err)
			continue
		}

		// Check if the volume needs healing
		var needsHealing bool
		distVolume.mu.RLock()
		for _, shard := range distVolume.DistInfo.Shards {
			if shard.NeedsHealing {
				needsHealing = true
				break
			}
		}
		healthPct := distVolume.DistInfo.HealthPercentage
		distVolume.mu.RUnlock()

		if needsHealing || healthPct < 100.0 {
			log.Printf("Healing volume %s (health: %.2f%%)", volumeID, healthPct)
			if err := s.RepairVolume(ctx, volumeID); err != nil {
				log.Printf("Error healing volume %s: %v", volumeID, err)
			}
		}
	}
}

// placeShards places shards of a volume on nodes
func (s *DistributedStorageService) placeShards(
	ctx context.Context,
	volume *DistributedVolume,
	replicationFactor int,
) error {
	// Get available nodes
	availableNodes := s.GetAvailableNodes()
	if len(availableNodes) < replicationFactor {
		return fmt.Errorf("not enough available nodes (%d) for replication factor %d",
			len(availableNodes), replicationFactor)
	}

	// Update node count
	volume.DistInfo.NodeCount = len(availableNodes)

	// Place each shard according to the placement strategy
	switch s.placementStrategy {
	case ShardPlacementRandom:
		return s.placeShardsByRandom(volume, availableNodes, replicationFactor)
	case ShardPlacementBalanced:
		return s.placeShardsByBalanced(volume, availableNodes, replicationFactor)
	case ShardPlacementLocalityAware:
		return s.placeShardsByLocality(volume, availableNodes, replicationFactor)
	case ShardPlacementZoneAware:
		return s.placeShardsByZone(volume, availableNodes, replicationFactor)
	default:
		return s.placeShardsByBalanced(volume, availableNodes, replicationFactor)
	}
}

// placeShardsByRandom places shards randomly across nodes
func (s *DistributedStorageService) placeShardsByRandom(
	volume *DistributedVolume,
	nodes []NodeInfo,
	replicationFactor int,
) error {
	volume.mu.Lock()
	defer volume.mu.Unlock()

	for i := range volume.DistInfo.Shards {
		shard := &volume.DistInfo.Shards[i]

		// Shuffle nodes for random placement
		shuffledNodes := make([]NodeInfo, len(nodes))
		copy(shuffledNodes, nodes)
		shuffleNodes(shuffledNodes)

		// Place shard on first replicationFactor nodes
		for j := 0; j < replicationFactor && j < len(shuffledNodes); j++ {
			shard.NodeIDs = append(shard.NodeIDs, shuffledNodes[j].ID)
		}
	}

	return nil
}

// placeShardsByBalanced places shards evenly across nodes
func (s *DistributedStorageService) placeShardsByBalanced(
	volume *DistributedVolume,
	nodes []NodeInfo,
	replicationFactor int,
) error {
	volume.mu.Lock()
	defer volume.mu.Unlock()

	// Track how many shards each node has
	nodeShardCount := make(map[string]int)
	for _, node := range nodes {
		nodeShardCount[node.ID] = 0
	}

	// Assign shards to nodes with the fewest shards
	for i := range volume.DistInfo.Shards {
		shard := &volume.DistInfo.Shards[i]

		// Sort nodes by shard count
		type nodeCount struct {
			ID    string
			Count int
		}
		sortedNodes := make([]nodeCount, 0, len(nodes))
		for _, node := range nodes {
			sortedNodes = append(sortedNodes, nodeCount{
				ID:    node.ID,
				Count: nodeShardCount[node.ID],
			})
		}
		sort.Slice(sortedNodes, func(i, j int) bool {
			return sortedNodes[i].Count < sortedNodes[j].Count
		})

		// Assign to nodes with fewest shards
		for j := 0; j < replicationFactor && j < len(sortedNodes); j++ {
			nodeID := sortedNodes[j].ID
			shard.NodeIDs = append(shard.NodeIDs, nodeID)
			nodeShardCount[nodeID]++
		}
	}

	return nil
}

// placeShardsByLocality places related shards on the same node
func (s *DistributedStorageService) placeShardsByLocality(
	volume *DistributedVolume,
	nodes []NodeInfo,
	replicationFactor int,
) error {
	// This is a simplified implementation
	// In a real system, this would use locality information to group related shards

	volume.mu.Lock()
	defer volume.mu.Unlock()

	// Calculate how many shards per node
	totalShards := len(volume.DistInfo.Shards)
	shardsPerNode := totalShards / len(nodes)
	if shardsPerNode < 1 {
		shardsPerNode = 1
	}

	// Assign shards to nodes
	nodeIndex := 0
	for i := range volume.DistInfo.Shards {
		shard := &volume.DistInfo.Shards[i]

		// Place primary copy
		primaryNode := nodeIndex % len(nodes)
		shard.NodeIDs = append(shard.NodeIDs, nodes[primaryNode].ID)

		// Place replicas on other nodes
		for j := 1; j < replicationFactor && j < len(nodes); j++ {
			replicaNode := (primaryNode + j) % len(nodes)
			shard.NodeIDs = append(shard.NodeIDs, nodes[replicaNode].ID)
		}

		// Move to next node after shardsPerNode shards
		if (i+1)%shardsPerNode == 0 {
			nodeIndex++
		}
	}

	return nil
}

// placeShardsByZone places shards across availability zones
func (s *DistributedStorageService) placeShardsByZone(
	volume *DistributedVolume,
	nodes []NodeInfo,
	replicationFactor int,
) error {
	// This is a simplified implementation
	// In a real system, this would use zone information from node metadata

	// Simulate zones by dividing nodes into groups
	zones := make(map[string][]NodeInfo)
	for i, node := range nodes {
		zoneID := fmt.Sprintf("zone-%d", i%3) // Simulate 3 zones
		zones[zoneID] = append(zones[zoneID], node)
	}

	volume.mu.Lock()
	defer volume.mu.Unlock()

	// Place each shard across different zones
	zoneIDs := make([]string, 0, len(zones))
	for zoneID := range zones {
		zoneIDs = append(zoneIDs, zoneID)
	}

	for i := range volume.DistInfo.Shards {
		shard := &volume.DistInfo.Shards[i]

		// Place replicas across different zones
		for j := 0; j < replicationFactor && j < len(zoneIDs); j++ {
			zoneID := zoneIDs[j]
			zoneNodes := zones[zoneID]

			if len(zoneNodes) > 0 {
				// Pick a node from this zone (round-robin)
				nodeIndex := i % len(zoneNodes)
				shard.NodeIDs = append(shard.NodeIDs, zoneNodes[nodeIndex].ID)
			}
		}
	}

	return nil
}

// Helper functions

// getReplicationPolicy returns the replication policy based on configuration
func (s *DistributedStorageService) getReplicationPolicy() string {
	if s.config.SynchronousReplication {
		return "sync"
	}
	return "async"
}

// calculateShardCount calculates the number of shards for a volume
func calculateShardCount(sizeBytes, shardSize int64) int {
	count := sizeBytes / shardSize
	if sizeBytes%shardSize != 0 {
		count++
	}
	return int(count)
}

// generateShardID generates a unique ID for a shard
func generateShardID(volumeID string, index int) string {
	return fmt.Sprintf("%s-shard-%d", volumeID, index)
}

// calculateChecksum calculates a checksum for shard data
func calculateChecksum(data []byte) string {
	// This is a simplified implementation
	// In a real system, this would use a proper hash function
	return base64.StdEncoding.EncodeToString(data[:min(len(data), 100)])
}

// contains checks if a string is in a slice
func contains(slice []string, s string) bool {
	for _, item := range slice {
		if item == s {
			return true
		}
	}
	return false
}

// shuffleNodes randomly shuffles a slice of nodes
func shuffleNodes(nodes []NodeInfo) {
	rand.Shuffle(len(nodes), func(i, j int) {
		nodes[i], nodes[j] = nodes[j], nodes[i]
	})
}

// min returns the minimum of two integers
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
