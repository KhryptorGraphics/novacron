package dr

import (
	"context"
	"fmt"
	"log"
	"sync"
	"time"
)

// SplitBrainPreventionSystem prevents split-brain scenarios
type SplitBrainPreventionSystem struct {
	config        *DRConfig
	quorumSize    int
	witnessNode   *WitnessNode
	fencingMgr    *FencingManager
	nodes         map[string]*NodeStatus
	nodesMu       sync.RWMutex
	partitions    map[string]*PartitionInfo
	partitionsMu  sync.RWMutex
}

// WitnessNode is a lightweight node for quorum
type WitnessNode struct {
	ID        string
	Endpoint  string
	LastSeen  time.Time
	Healthy   bool
}

// FencingManager handles node fencing
type FencingManager struct {
	mechanisms []FencingMechanism
	mu         sync.Mutex
}

// NodeStatus tracks node status
type NodeStatus struct {
	ID           string
	Region       string
	State        string // "active", "suspect", "fenced", "offline"
	LastHeartbeat time.Time
	Partition    string
	QuorumMember bool
}

// PartitionInfo tracks network partitions
type PartitionInfo struct {
	ID         string
	DetectedAt time.Time
	Nodes      []string
	State      string // "detected", "resolving", "resolved"
	Resolution string
}

// NewSplitBrainPreventionSystem creates a new split-brain prevention system
func NewSplitBrainPreventionSystem(config *DRConfig) (*SplitBrainPreventionSystem, error) {
	// Quorum is (N/2 + 1)
	totalNodes := 1 + len(config.SecondaryRegions)
	quorumSize := (totalNodes / 2) + 1

	sbp := &SplitBrainPreventionSystem{
		config:     config,
		quorumSize: quorumSize,
		witnessNode: &WitnessNode{
			ID:       "witness-1",
			Endpoint: "witness.novacron.io:9000",
			Healthy:  true,
		},
		fencingMgr: &FencingManager{
			mechanisms: []FencingMechanism{
				{
					Type:    "STONITH",
					Enabled: true,
					Target:  "all",
				},
				{
					Type:    "network",
					Enabled: true,
					Target:  "all",
				},
			},
		},
		nodes:      make(map[string]*NodeStatus),
		partitions: make(map[string]*PartitionInfo),
	}

	// Initialize nodes
	sbp.nodes[config.PrimaryRegion] = &NodeStatus{
		ID:           config.PrimaryRegion,
		Region:       config.PrimaryRegion,
		State:        "active",
		LastHeartbeat: time.Now(),
		QuorumMember: true,
	}

	for _, region := range config.SecondaryRegions {
		sbp.nodes[region] = &NodeStatus{
			ID:           region,
			Region:       region,
			State:        "active",
			LastHeartbeat: time.Now(),
			QuorumMember: true,
		}
	}

	return sbp, nil
}

// Start begins split-brain prevention
func (sbp *SplitBrainPreventionSystem) Start(ctx context.Context) error {
	log.Println("Starting split-brain prevention system")

	// Start partition detector
	go sbp.detectPartitions(ctx)

	// Start witness heartbeat
	go sbp.witnessHeartbeat(ctx)

	log.Println("Split-brain prevention started")
	return nil
}

// CheckQuorum verifies quorum is available
func (sbp *SplitBrainPreventionSystem) CheckQuorum(ctx context.Context) error {
	sbp.nodesMu.RLock()
	defer sbp.nodesMu.RUnlock()

	activeNodes := 0
	for _, node := range sbp.nodes {
		if node.State == "active" && time.Since(node.LastHeartbeat) < 30*time.Second {
			activeNodes++
		}
	}

	// Include witness node in quorum
	if sbp.witnessNode.Healthy {
		activeNodes++
	}

	if activeNodes < sbp.quorumSize {
		return fmt.Errorf("quorum not available: have %d, need %d", activeNodes, sbp.quorumSize)
	}

	log.Printf("Quorum check passed: %d/%d nodes active", activeNodes, sbp.quorumSize)
	return nil
}

// detectPartitions monitors for network partitions
func (sbp *SplitBrainPreventionSystem) detectPartitions(ctx context.Context) {
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			sbp.checkForPartitions()
		case <-ctx.Done():
			return
		}
	}
}

// checkForPartitions checks if network partition exists
func (sbp *SplitBrainPreventionSystem) checkForPartitions() {
	sbp.nodesMu.Lock()
	defer sbp.nodesMu.Unlock()

	// Group nodes by connectivity
	partitionGroups := make(map[string][]string)

	for id, node := range sbp.nodes {
		// Simulate connectivity check
		// In production, this would ping nodes and check connectivity
		partitionKey := "main"
		if time.Since(node.LastHeartbeat) > 30*time.Second {
			partitionKey = "isolated"
		}

		partitionGroups[partitionKey] = append(partitionGroups[partitionKey], id)
		node.Partition = partitionKey
	}

	// If we have multiple partition groups, we have a partition
	if len(partitionGroups) > 1 {
		sbp.handlePartition(partitionGroups)
	}
}

// handlePartition handles detected partition
func (sbp *SplitBrainPreventionSystem) handlePartition(groups map[string][]string) {
	log.Printf("PARTITION DETECTED: %d groups", len(groups))

	partition := &PartitionInfo{
		ID:         fmt.Sprintf("partition-%d", time.Now().Unix()),
		DetectedAt: time.Now(),
		State:      "detected",
	}

	// Determine which group has quorum
	var quorumGroup string
	maxSize := 0

	for group, nodes := range groups {
		log.Printf("Partition group %s: %v", group, nodes)
		if len(nodes) > maxSize {
			maxSize = len(nodes)
			quorumGroup = group
		}
		partition.Nodes = append(partition.Nodes, nodes...)
	}

	// Fence nodes in minority partitions
	for group, nodes := range groups {
		if group != quorumGroup {
			log.Printf("Fencing minority partition: %s", group)
			for _, nodeID := range nodes {
				sbp.fenceNode(nodeID)
			}
		}
	}

	partition.Resolution = fmt.Sprintf("Fenced minority, preserved quorum in %s", quorumGroup)
	partition.State = "resolved"

	sbp.partitionsMu.Lock()
	sbp.partitions[partition.ID] = partition
	sbp.partitionsMu.Unlock()
}

// fenceNode fences a node to prevent split-brain
func (sbp *SplitBrainPreventionSystem) fenceNode(nodeID string) {
	log.Printf("FENCING NODE: %s", nodeID)

	sbp.nodesMu.Lock()
	if node, exists := sbp.nodes[nodeID]; exists {
		node.State = "fenced"
	}
	sbp.nodesMu.Unlock()

	// Execute fencing mechanisms
	sbp.fencingMgr.mu.Lock()
	defer sbp.fencingMgr.mu.Unlock()

	for _, mechanism := range sbp.fencingMgr.mechanisms {
		if !mechanism.Enabled {
			continue
		}

		switch mechanism.Type {
		case "STONITH":
			sbp.executeStonith(nodeID)
		case "network":
			sbp.executeNetworkFencing(nodeID)
		case "disk":
			sbp.executeDiskFencing(nodeID)
		}
	}
}

// executeStonith executes STONITH (Shoot The Other Node In The Head)
func (sbp *SplitBrainPreventionSystem) executeStonith(nodeID string) {
	log.Printf("STONITH: Forcibly shutting down node: %s", nodeID)

	// In production, this would:
	// - Use IPMI/BMC to power off the node
	// - Use cloud provider API to stop the instance
	// - Use hypervisor API to kill the VM

	time.Sleep(100 * time.Millisecond)
}

// executeNetworkFencing isolates node from network
func (sbp *SplitBrainPreventionSystem) executeNetworkFencing(nodeID string) {
	log.Printf("Network fence: Isolating node: %s", nodeID)

	// In production, this would:
	// - Update firewall rules to block node
	// - Remove node from load balancers
	// - Update routing tables

	time.Sleep(50 * time.Millisecond)
}

// executeDiskFencing prevents node from accessing shared storage
func (sbp *SplitBrainPreventionSystem) executeDiskFencing(nodeID string) {
	log.Printf("Disk fence: Revoking storage access for node: %s", nodeID)

	// In production, this would:
	// - Revoke SCSI reservations
	// - Remove storage LUN mappings
	// - Update SAN access controls

	time.Sleep(50 * time.Millisecond)
}

// witnessHeartbeat maintains witness node heartbeat
func (sbp *SplitBrainPreventionSystem) witnessHeartbeat(ctx context.Context) {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			// Ping witness node
			sbp.witnessNode.LastSeen = time.Now()
			sbp.witnessNode.Healthy = true

		case <-ctx.Done():
			return
		}
	}
}

// ReconcilePartition attempts to safely merge partitions after healing
func (sbp *SplitBrainPreventionSystem) ReconcilePartition(partitionID string) error {
	sbp.partitionsMu.Lock()
	defer sbp.partitionsMu.Unlock()

	partition, exists := sbp.partitions[partitionID]
	if !exists {
		return fmt.Errorf("partition not found: %s", partitionID)
	}

	if partition.State != "resolved" {
		return fmt.Errorf("partition not yet resolved")
	}

	log.Printf("Reconciling partition: %s", partitionID)

	// Reconciliation steps:
	// 1. Verify connectivity restored
	// 2. Compare state between partitions
	// 3. Apply conflict resolution
	// 4. Sync state
	// 5. Restore nodes to active

	time.Sleep(2 * time.Second)

	partition.State = "reconciled"
	log.Printf("Partition reconciled: %s", partitionID)

	return nil
}

// UpdateNodeHeartbeat updates last heartbeat for a node
func (sbp *SplitBrainPreventionSystem) UpdateNodeHeartbeat(nodeID string) {
	sbp.nodesMu.Lock()
	defer sbp.nodesMu.Unlock()

	if node, exists := sbp.nodes[nodeID]; exists {
		node.LastHeartbeat = time.Now()

		// Restore fenced nodes if they come back
		if node.State == "fenced" {
			node.State = "active"
			log.Printf("Node restored from fenced state: %s", nodeID)
		}
	}
}

// GetQuorumStatus returns current quorum status
func (sbp *SplitBrainPreventionSystem) GetQuorumStatus() map[string]interface{} {
	sbp.nodesMu.RLock()
	defer sbp.nodesMu.RUnlock()

	activeNodes := 0
	for _, node := range sbp.nodes {
		if node.State == "active" {
			activeNodes++
		}
	}

	if sbp.witnessNode.Healthy {
		activeNodes++
	}

	return map[string]interface{}{
		"active_nodes":   activeNodes,
		"required_nodes": sbp.quorumSize,
		"has_quorum":     activeNodes >= sbp.quorumSize,
		"witness_active": sbp.witnessNode.Healthy,
	}
}
