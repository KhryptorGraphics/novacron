package sync

import (
	"context"
	"fmt"
	"time"

	"go.uber.org/zap"
)

// NovaCronIntegration integrates ASS with NovaCron federation
type NovaCronIntegration struct {
	engine          *ASSEngine
	clusterMetadata *ClusterMetadata
	logger          *zap.Logger
	ctx             context.Context
	cancel          context.CancelFunc
}

// IntegrationConfig holds configuration for NovaCron integration
type IntegrationConfig struct {
	NodeID              string
	Region              string
	AntiEntropyInterval time.Duration
	GossipFanout        int
	GossipInterval      time.Duration
	MaxGossipHops       int
	Transport           Transport
	Logger              *zap.Logger
}

// NewNovaCronIntegration creates a new NovaCron integration
func NewNovaCronIntegration(config IntegrationConfig) *NovaCronIntegration {
	ctx, cancel := context.WithCancel(context.Background())

	// Create ASS engine
	engine := NewASSEngine(config.NodeID, config.Transport, config.Logger)

	// Configure gossip protocol
	engine.gossip = NewGossipProtocol(
		engine,
		config.GossipFanout,
		config.GossipInterval,
		config.MaxGossipHops,
		config.Logger,
	)

	// Configure anti-entropy service
	engine.antiEntropy = NewAntiEntropyService(
		engine,
		config.AntiEntropyInterval,
		config.Logger,
	)

	// Create cluster metadata
	clusterMetadata := NewClusterMetadata(config.NodeID)

	integration := &NovaCronIntegration{
		engine:          engine,
		clusterMetadata: clusterMetadata,
		logger:          config.Logger,
		ctx:             ctx,
		cancel:          cancel,
	}

	return integration
}

// Start starts the NovaCron integration
func (ni *NovaCronIntegration) Start() error {
	ni.logger.Info("Starting NovaCron ASS integration")

	// Start ASS engine
	if err := ni.engine.Start(); err != nil {
		return fmt.Errorf("failed to start ASS engine: %w", err)
	}

	// Store cluster metadata in ASS
	if err := ni.storeClusterMetadata(); err != nil {
		return fmt.Errorf("failed to store cluster metadata: %w", err)
	}

	// Start periodic metadata sync
	go ni.periodicMetadataSync()

	ni.logger.Info("NovaCron ASS integration started successfully")
	return nil
}

// Stop stops the NovaCron integration
func (ni *NovaCronIntegration) Stop() error {
	ni.logger.Info("Stopping NovaCron ASS integration")

	ni.cancel()

	// Stop ASS engine
	if err := ni.engine.Stop(); err != nil {
		return fmt.Errorf("failed to stop ASS engine: %w", err)
	}

	ni.logger.Info("NovaCron ASS integration stopped")
	return nil
}

// RegisterRegion registers a peer region for synchronization
func (ni *NovaCronIntegration) RegisterRegion(id, region, endpoint string) {
	peer := &RegionPeer{
		ID:          id,
		Region:      region,
		Endpoint:    endpoint,
		LastSync:    time.Time{},
		VectorClock: make(VectorClock),
	}

	ni.engine.RegisterPeer(peer)
	ni.logger.Info("Registered peer region",
		zap.String("peer_id", id),
		zap.String("region", region))
}

// UpdateVMState updates VM state with eventual consistency
func (ni *NovaCronIntegration) UpdateVMState(vmID string, state VMState) error {
	// Update cluster metadata
	if err := ni.clusterMetadata.UpdateVMState(vmID, state); err != nil {
		return err
	}

	// Store in ASS engine for synchronization
	return ni.storeClusterMetadata()
}

// GetVMState retrieves VM state
func (ni *NovaCronIntegration) GetVMState(vmID string) (*VMState, error) {
	// Sync cluster metadata first
	if err := ni.loadClusterMetadata(); err != nil {
		ni.logger.Warn("Failed to load cluster metadata", zap.Error(err))
	}

	return ni.clusterMetadata.GetVMState(vmID)
}

// ListVMs returns all VMs across all regions
func (ni *NovaCronIntegration) ListVMs() ([]VMState, error) {
	// Sync cluster metadata first
	if err := ni.loadClusterMetadata(); err != nil {
		ni.logger.Warn("Failed to load cluster metadata", zap.Error(err))
	}

	return ni.clusterMetadata.ListVMs()
}

// UpdateNodeStatus updates node status with eventual consistency
func (ni *NovaCronIntegration) UpdateNodeStatus(nodeID string, status NodeStatus) error {
	if err := ni.clusterMetadata.UpdateNodeStatus(nodeID, status); err != nil {
		return err
	}

	return ni.storeClusterMetadata()
}

// GetNodeStatus retrieves node status
func (ni *NovaCronIntegration) GetNodeStatus(nodeID string) (*NodeStatus, error) {
	if err := ni.loadClusterMetadata(); err != nil {
		ni.logger.Warn("Failed to load cluster metadata", zap.Error(err))
	}

	return ni.clusterMetadata.GetNodeStatus(nodeID)
}

// ListNodes returns all nodes across all regions
func (ni *NovaCronIntegration) ListNodes() ([]NodeStatus, error) {
	if err := ni.loadClusterMetadata(); err != nil {
		ni.logger.Warn("Failed to load cluster metadata", zap.Error(err))
	}

	return ni.clusterMetadata.ListNodes()
}

// AssignVM assigns a VM to a node with eventual consistency
func (ni *NovaCronIntegration) AssignVM(vmID, nodeID string) error {
	if err := ni.clusterMetadata.AssignVM(vmID, nodeID); err != nil {
		return err
	}

	return ni.storeClusterMetadata()
}

// GetVMAssignment retrieves VM assignment
func (ni *NovaCronIntegration) GetVMAssignment(vmID string) (string, error) {
	if err := ni.loadClusterMetadata(); err != nil {
		ni.logger.Warn("Failed to load cluster metadata", zap.Error(err))
	}

	return ni.clusterMetadata.GetVMAssignment(vmID)
}

// SyncWithRegion manually triggers synchronization with a specific region
func (ni *NovaCronIntegration) SyncWithRegion(regionID string) error {
	return ni.engine.SyncWithRegion(regionID)
}

// GetStats returns integration statistics
func (ni *NovaCronIntegration) GetStats() IntegrationStats {
	clusterStats := ni.clusterMetadata.GetStats()
	gossipStats := ni.engine.gossip.GetStats()
	antiEntropyStats := ni.engine.antiEntropy.GetStats()

	return IntegrationStats{
		ClusterStats:     clusterStats,
		GossipStats:      gossipStats,
		AntiEntropyStats: antiEntropyStats,
		PeerCount:        len(ni.engine.regions),
		CRDTCount:        len(ni.engine.crdtStore.data),
	}
}

// IntegrationStats represents integration statistics
type IntegrationStats struct {
	ClusterStats     ClusterStats      `json:"cluster_stats"`
	GossipStats      GossipStats       `json:"gossip_stats"`
	AntiEntropyStats AntiEntropyStats  `json:"anti_entropy_stats"`
	PeerCount        int               `json:"peer_count"`
	CRDTCount        int               `json:"crdt_count"`
}

func (ni *NovaCronIntegration) storeClusterMetadata() error {
	// Marshal cluster metadata
	data, err := ni.clusterMetadata.Marshal()
	if err != nil {
		return err
	}

	// Create OR-Map to store the metadata
	metadataMap := NewORMap(ni.engine.nodeID)
	metadataMap.SetLWW("cluster_metadata", string(data))

	// Store in ASS engine
	return ni.engine.Set("cluster_metadata", metadataMap)
}

func (ni *NovaCronIntegration) loadClusterMetadata() error {
	// Get cluster metadata from ASS engine
	value, exists := ni.engine.Get("cluster_metadata")
	if !exists {
		return nil // No metadata yet
	}

	metadataMap, ok := value.(*ORMap)
	if !ok {
		return fmt.Errorf("invalid cluster metadata type")
	}

	data, exists := metadataMap.GetLWW("cluster_metadata")
	if !exists {
		return nil
	}

	dataStr, ok := data.(string)
	if !ok {
		return fmt.Errorf("invalid cluster metadata format")
	}

	// Unmarshal into cluster metadata
	return ni.clusterMetadata.Unmarshal([]byte(dataStr))
}

func (ni *NovaCronIntegration) periodicMetadataSync() {
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ni.ctx.Done():
			return
		case <-ticker.C:
			if err := ni.loadClusterMetadata(); err != nil {
				ni.logger.Error("Failed to sync cluster metadata", zap.Error(err))
			}
		}
	}
}

// MigrateExistingState migrates existing cluster state to CRDT-based ASS
func MigrateExistingState(integration *NovaCronIntegration, vms []VMState, nodes []NodeStatus) error {
	integration.logger.Info("Migrating existing cluster state to ASS",
		zap.Int("vms", len(vms)),
		zap.Int("nodes", len(nodes)))

	// Migrate VM states
	for _, vm := range vms {
		if err := integration.clusterMetadata.UpdateVMState(vm.ID, vm); err != nil {
			return fmt.Errorf("failed to migrate VM %s: %w", vm.ID, err)
		}
	}

	// Migrate node statuses
	for _, node := range nodes {
		if err := integration.clusterMetadata.UpdateNodeStatus(node.ID, node); err != nil {
			return fmt.Errorf("failed to migrate node %s: %w", node.ID, err)
		}
	}

	// Store migrated data
	if err := integration.storeClusterMetadata(); err != nil {
		return fmt.Errorf("failed to store migrated metadata: %w", err)
	}

	integration.logger.Info("Successfully migrated cluster state to ASS")
	return nil
}

// Import ORMap type
import (
	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/sync/crdt"
)

type ORMap = crdt.ORMap
type VectorClock = crdt.VectorClock

func NewORMap(nodeID string) *ORMap {
	return crdt.NewORMap(nodeID)
}
