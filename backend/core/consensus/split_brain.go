package consensus

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// SplitBrainDetector detects and resolves split-brain scenarios
type SplitBrainDetector struct {
	mu sync.RWMutex
	
	// Core components
	raftNode      *RaftNode
	membership    *ClusterMembership
	
	// Detection state
	partitionMap  map[string]*Partition
	lastLeaderSeen time.Time
	multipleLeaders []string
	
	// Configuration
	config        SplitBrainConfig
	
	// Resolution strategies
	resolver      ResolutionStrategy
	
	// Monitoring
	stopChan      chan struct{}
	metrics       SplitBrainMetrics
	
	// Witness nodes for tie-breaking
	witnesses     []string
}

// Partition represents a network partition
type Partition struct {
	ID            string
	Nodes         []string
	Leader        string
	Size          int
	LastContact   time.Time
	HasQuorum     bool
	DataVersion   int64
}

// SplitBrainConfig contains configuration for split-brain detection
type SplitBrainConfig struct {
	DetectionInterval   time.Duration
	LeaderTimeout       time.Duration
	PartitionThreshold  time.Duration
	EnableAutoResolution bool
	PreferLargerPartition bool
	UseWitnessNodes     bool
	MaxLeaderElections  int
}

// SplitBrainMetrics tracks split-brain related metrics
type SplitBrainMetrics struct {
	DetectionsCount     int64
	ResolutionsCount    int64
	LastDetection       time.Time
	LastResolution      time.Time
	FalsePositives      int64
	ResolutionFailures  int64
	AverageResolutionTime time.Duration
}

// ResolutionStrategy defines how to resolve split-brain scenarios
type ResolutionStrategy interface {
	Resolve(partitions []*Partition) (*Partition, error)
}

// QuorumBasedResolver resolves split-brain by selecting the partition with quorum
type QuorumBasedResolver struct {
	minQuorumSize int
}

// MajorityBasedResolver selects the partition with the most nodes
type MajorityBasedResolver struct{}

// WitnessBasedResolver uses witness nodes to break ties
type WitnessBasedResolver struct {
	witnesses []string
}

// DataVersionResolver selects the partition with the most recent data
type DataVersionResolver struct{}

// NewSplitBrainDetector creates a new split-brain detector
func NewSplitBrainDetector(raftNode *RaftNode, membership *ClusterMembership, config SplitBrainConfig) *SplitBrainDetector {
	if config.DetectionInterval == 0 {
		config.DetectionInterval = 5 * time.Second
	}
	if config.LeaderTimeout == 0 {
		config.LeaderTimeout = 10 * time.Second
	}
	if config.PartitionThreshold == 0 {
		config.PartitionThreshold = 30 * time.Second
	}
	
	sbd := &SplitBrainDetector{
		raftNode:     raftNode,
		membership:   membership,
		partitionMap: make(map[string]*Partition),
		config:       config,
		stopChan:     make(chan struct{}),
		witnesses:    make([]string, 0),
	}
	
	// Select resolution strategy based on configuration
	if config.UseWitnessNodes {
		sbd.resolver = &WitnessBasedResolver{witnesses: sbd.witnesses}
	} else if config.PreferLargerPartition {
		sbd.resolver = &MajorityBasedResolver{}
	} else {
		sbd.resolver = &QuorumBasedResolver{minQuorumSize: membership.quorumNodes}
	}
	
	return sbd
}

// Start begins split-brain detection
func (sbd *SplitBrainDetector) Start() {
	go sbd.detectionLoop()
}

// Stop stops split-brain detection
func (sbd *SplitBrainDetector) Stop() {
	close(sbd.stopChan)
}

// AddWitnessNode adds a witness node for tie-breaking
func (sbd *SplitBrainDetector) AddWitnessNode(nodeAddress string) {
	sbd.mu.Lock()
	defer sbd.mu.Unlock()
	
	sbd.witnesses = append(sbd.witnesses, nodeAddress)
}

// detectionLoop continuously monitors for split-brain scenarios
func (sbd *SplitBrainDetector) detectionLoop() {
	ticker := time.NewTicker(sbd.config.DetectionInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			sbd.detectSplitBrain()
		case <-sbd.stopChan:
			return
		}
	}
}

// detectSplitBrain checks for split-brain conditions
func (sbd *SplitBrainDetector) detectSplitBrain() {
	sbd.mu.Lock()
	defer sbd.mu.Unlock()
	
	// Get current cluster state
	nodes := sbd.membership.GetAllNodes()
	
	// Identify partitions
	partitions := sbd.identifyPartitions(nodes)
	
	// Check for multiple leaders
	leadersFound := sbd.checkMultipleLeaders(partitions)
	
	if leadersFound > 1 {
		sbd.metrics.DetectionsCount++
		sbd.metrics.LastDetection = time.Now()
		
		if sbd.config.EnableAutoResolution {
			sbd.resolveSplitBrain(partitions)
		} else {
			// Log the detection for manual intervention
			fmt.Printf("Split-brain detected: %d leaders found in %d partitions\n", leadersFound, len(partitions))
		}
	}
}

// identifyPartitions identifies network partitions in the cluster
func (sbd *SplitBrainDetector) identifyPartitions(nodes map[string]*NodeInfo) []*Partition {
	partitions := make([]*Partition, 0)
	visited := make(map[string]bool)
	
	for nodeID, node := range nodes {
		if visited[nodeID] {
			continue
		}
		
		// Create a new partition starting from this node
		partition := &Partition{
			ID:          fmt.Sprintf("partition-%d", len(partitions)),
			Nodes:       []string{nodeID},
			LastContact: node.LastHeartbeat,
		}
		
		visited[nodeID] = true
		
		// Find all nodes reachable from this node
		sbd.expandPartition(partition, nodes, visited)
		
		// Determine if this partition has quorum
		partition.HasQuorum = len(partition.Nodes) > len(nodes)/2
		partition.Size = len(partition.Nodes)
		
		partitions = append(partitions, partition)
	}
	
	return partitions
}

// expandPartition expands a partition to include all reachable nodes
func (sbd *SplitBrainDetector) expandPartition(partition *Partition, nodes map[string]*NodeInfo, visited map[string]bool) {
	// In a real implementation, this would check actual network connectivity
	// For now, we'll use heartbeat timing to determine reachability
	
	threshold := time.Now().Add(-sbd.config.PartitionThreshold)
	
	for nodeID, node := range nodes {
		if visited[nodeID] {
			continue
		}
		
		// Check if this node is reachable (recent heartbeat)
		if node.LastHeartbeat.After(threshold) {
			partition.Nodes = append(partition.Nodes, nodeID)
			visited[nodeID] = true
			
			// Update partition's last contact time
			if node.LastHeartbeat.After(partition.LastContact) {
				partition.LastContact = node.LastHeartbeat
			}
		}
	}
}

// checkMultipleLeaders checks if multiple partitions have leaders
func (sbd *SplitBrainDetector) checkMultipleLeaders(partitions []*Partition) int {
	leadersFound := 0
	sbd.multipleLeaders = []string{}
	
	for _, partition := range partitions {
		// Check if any node in this partition thinks it's a leader
		for _, nodeID := range partition.Nodes {
			if sbd.isNodeLeader(nodeID) {
				partition.Leader = nodeID
				leadersFound++
				sbd.multipleLeaders = append(sbd.multipleLeaders, nodeID)
				break
			}
		}
	}
	
	return leadersFound
}

// isNodeLeader checks if a node believes it's the leader
func (sbd *SplitBrainDetector) isNodeLeader(nodeID string) bool {
	// Check with the Raft node
	if sbd.raftNode.nodeID == nodeID {
		return sbd.raftNode.state == Leader
	}
	
	// For remote nodes, we'd need to query them
	// For now, return false for remote nodes
	return false
}

// resolveSplitBrain attempts to resolve a split-brain scenario
func (sbd *SplitBrainDetector) resolveSplitBrain(partitions []*Partition) {
	startTime := time.Now()
	
	// Use the configured resolution strategy
	winningPartition, err := sbd.resolver.Resolve(partitions)
	if err != nil {
		sbd.metrics.ResolutionFailures++
		fmt.Printf("Failed to resolve split-brain: %v\n", err)
		return
	}
	
	// Force non-winning partitions to step down
	for _, partition := range partitions {
		if partition.ID != winningPartition.ID {
			sbd.forcePartitionStepDown(partition)
		}
	}
	
	sbd.metrics.ResolutionsCount++
	sbd.metrics.LastResolution = time.Now()
	sbd.metrics.AverageResolutionTime = time.Since(startTime)
	
	fmt.Printf("Split-brain resolved: partition %s selected as authoritative\n", winningPartition.ID)
}

// forcePartitionStepDown forces all leaders in a partition to step down
func (sbd *SplitBrainDetector) forcePartitionStepDown(partition *Partition) {
	for _, nodeID := range partition.Nodes {
		if sbd.isNodeLeader(nodeID) {
			if nodeID == sbd.raftNode.nodeID {
				// Force local node to become follower
				sbd.raftNode.mu.Lock()
				sbd.raftNode.state = Follower
				sbd.raftNode.votedFor = ""
				sbd.raftNode.mu.Unlock()
			} else {
				// Send step-down command to remote node
				// In a real implementation, this would send an RPC
				fmt.Printf("Forcing node %s to step down\n", nodeID)
			}
		}
	}
}

// GetMetrics returns current split-brain metrics
func (sbd *SplitBrainDetector) GetMetrics() SplitBrainMetrics {
	sbd.mu.RLock()
	defer sbd.mu.RUnlock()
	
	return sbd.metrics
}

// IsInSplitBrain checks if the cluster is currently in a split-brain state
func (sbd *SplitBrainDetector) IsInSplitBrain() bool {
	sbd.mu.RLock()
	defer sbd.mu.RUnlock()
	
	return len(sbd.multipleLeaders) > 1
}

// GetPartitions returns the current partition map
func (sbd *SplitBrainDetector) GetPartitions() []*Partition {
	sbd.mu.RLock()
	defer sbd.mu.RUnlock()
	
	partitions := make([]*Partition, 0, len(sbd.partitionMap))
	for _, partition := range sbd.partitionMap {
		partitions = append(partitions, partition)
	}
	
	return partitions
}

// Resolution Strategy Implementations

// Resolve selects the partition with quorum
func (qbr *QuorumBasedResolver) Resolve(partitions []*Partition) (*Partition, error) {
	for _, partition := range partitions {
		if partition.HasQuorum {
			return partition, nil
		}
	}
	
	// If no partition has quorum, select the largest
	var largest *Partition
	maxSize := 0
	
	for _, partition := range partitions {
		if partition.Size > maxSize {
			largest = partition
			maxSize = partition.Size
		}
	}
	
	if largest != nil {
		return largest, nil
	}
	
	return nil, fmt.Errorf("no valid partition found")
}

// Resolve selects the partition with the most nodes
func (mbr *MajorityBasedResolver) Resolve(partitions []*Partition) (*Partition, error) {
	var largest *Partition
	maxSize := 0
	
	for _, partition := range partitions {
		if partition.Size > maxSize {
			largest = partition
			maxSize = partition.Size
		}
	}
	
	if largest != nil {
		return largest, nil
	}
	
	return nil, fmt.Errorf("no valid partition found")
}

// Resolve uses witness nodes to select a partition
func (wbr *WitnessBasedResolver) Resolve(partitions []*Partition) (*Partition, error) {
	// Query witness nodes to determine which partition they can reach
	// For now, return the first partition with a leader
	for _, partition := range partitions {
		if partition.Leader != "" {
			return partition, nil
		}
	}
	
	// Fallback to largest partition
	return (&MajorityBasedResolver{}).Resolve(partitions)
}

// Resolve selects the partition with the most recent data
func (dvr *DataVersionResolver) Resolve(partitions []*Partition) (*Partition, error) {
	var newest *Partition
	maxVersion := int64(0)
	
	for _, partition := range partitions {
		if partition.DataVersion > maxVersion {
			newest = partition
			maxVersion = partition.DataVersion
		}
	}
	
	if newest != nil {
		return newest, nil
	}
	
	return nil, fmt.Errorf("no valid partition found")
}

// FenceNode isolates a node to prevent it from causing split-brain
func (sbd *SplitBrainDetector) FenceNode(nodeID string) error {
	sbd.mu.Lock()
	defer sbd.mu.Unlock()
	
	// Remove the node from membership
	err := sbd.membership.RemoveNode(nodeID, "Fenced due to split-brain prevention")
	if err != nil {
		return fmt.Errorf("failed to fence node %s: %v", nodeID, err)
	}
	
	// If it's the local node, step down
	if nodeID == sbd.raftNode.nodeID {
		sbd.raftNode.mu.Lock()
		sbd.raftNode.state = Follower
		sbd.raftNode.votedFor = ""
		sbd.raftNode.mu.Unlock()
	}
	
	return nil
}

// PreventSplitBrain implements proactive split-brain prevention
func (sbd *SplitBrainDetector) PreventSplitBrain(ctx context.Context) error {
	// Check if we're about to lose quorum
	healthyNodes := sbd.membership.GetHealthyNodes()
	totalNodes := len(sbd.membership.GetAllNodes())
	
	if len(healthyNodes) <= totalNodes/2 {
		// About to lose quorum - take preventive action
		if sbd.raftNode.state == Leader {
			// Step down as leader to prevent split-brain
			sbd.raftNode.mu.Lock()
			sbd.raftNode.state = Follower
			sbd.raftNode.votedFor = ""
			sbd.raftNode.mu.Unlock()
			
			return fmt.Errorf("stepped down to prevent split-brain: only %d/%d nodes healthy", len(healthyNodes), totalNodes)
		}
	}
	
	return nil
}