package consensus

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// NodeState represents the current state of a cluster node
type NodeState int

const (
	NodeStateUnknown NodeState = iota
	NodeStateHealthy
	NodeStateDegraded
	NodeStateUnhealthy
	NodeStateOffline
)

// MembershipChange represents a change to cluster membership
type MembershipChange struct {
	Type      MembershipChangeType
	NodeID    string
	Timestamp time.Time
	Reason    string
}

// MembershipChangeType defines types of membership changes
type MembershipChangeType int

const (
	MembershipAdd MembershipChangeType = iota
	MembershipRemove
	MembershipUpdate
)

// NodeInfo contains information about a cluster node
type NodeInfo struct {
	ID             string
	Address        string
	State          NodeState
	LastHeartbeat  time.Time
	JoinedAt       time.Time
	Metadata       map[string]string
	HealthMetrics  HealthMetrics
	VoteWeight     int // For weighted voting in heterogeneous clusters
	Zone           string // For zone-aware placement
	Rack           string // For rack-aware placement
}

// HealthMetrics contains health monitoring data for a node
type HealthMetrics struct {
	CPUUsage           float64
	MemoryUsage        float64
	DiskUsage          float64
	NetworkLatency     time.Duration
	RequestRate        float64
	ErrorRate          float64
	LastCheckTime      time.Time
	ConsecutiveFailures int
}

// ClusterMembership manages cluster membership and health monitoring
type ClusterMembership struct {
	mu sync.RWMutex
	
	// Node information
	nodes          map[string]*NodeInfo
	localNodeID    string
	
	// Configuration
	config         MembershipConfig
	
	// Health monitoring
	healthChecker  HealthChecker
	stopChan       chan struct{}
	
	// Change notifications
	changeListeners []func(MembershipChange)
	
	// Quorum tracking
	minNodes       int
	quorumNodes    int
	
	// Metrics
	metrics        MembershipMetrics
}

// MembershipConfig contains configuration for cluster membership
type MembershipConfig struct {
	HeartbeatInterval    time.Duration
	HealthCheckInterval  time.Duration
	FailureThreshold     int
	RecoveryThreshold    int
	MaxNodeFailures      int
	EnableAutoRecovery   bool
	EnableZoneAwareness  bool
	MinQuorumSize        int
}

// MembershipMetrics tracks membership-related metrics
type MembershipMetrics struct {
	TotalNodes          int
	HealthyNodes        int
	UnhealthyNodes      int
	MembershipChanges   int64
	LastQuorumLoss      time.Time
	QuorumLossCount     int64
}

// HealthChecker defines the interface for health checking
type HealthChecker interface {
	CheckHealth(ctx context.Context, nodeID string, address string) (*HealthMetrics, error)
}

// DefaultHealthChecker implements basic health checking
type DefaultHealthChecker struct {
	timeout time.Duration
}

// NewDefaultHealthChecker creates a new default health checker
func NewDefaultHealthChecker(timeout time.Duration) *DefaultHealthChecker {
	return &DefaultHealthChecker{
		timeout: timeout,
	}
}

// CheckHealth performs a health check on a node
func (dhc *DefaultHealthChecker) CheckHealth(ctx context.Context, nodeID string, address string) (*HealthMetrics, error) {
	// In a real implementation, this would make actual health check calls
	// For now, return simulated metrics
	return &HealthMetrics{
		CPUUsage:        0.5,
		MemoryUsage:     0.6,
		DiskUsage:       0.4,
		NetworkLatency:  10 * time.Millisecond,
		RequestRate:     100,
		ErrorRate:       0.01,
		LastCheckTime:   time.Now(),
	}, nil
}

// NewClusterMembership creates a new cluster membership manager
func NewClusterMembership(localNodeID string, config MembershipConfig) *ClusterMembership {
	if config.HeartbeatInterval == 0 {
		config.HeartbeatInterval = 5 * time.Second
	}
	if config.HealthCheckInterval == 0 {
		config.HealthCheckInterval = 10 * time.Second
	}
	if config.FailureThreshold == 0 {
		config.FailureThreshold = 3
	}
	if config.RecoveryThreshold == 0 {
		config.RecoveryThreshold = 2
	}
	if config.MinQuorumSize == 0 {
		config.MinQuorumSize = 2
	}
	
	return &ClusterMembership{
		nodes:           make(map[string]*NodeInfo),
		localNodeID:     localNodeID,
		config:          config,
		healthChecker:   NewDefaultHealthChecker(5 * time.Second),
		stopChan:        make(chan struct{}),
		changeListeners: make([]func(MembershipChange), 0),
		minNodes:        config.MinQuorumSize,
		quorumNodes:     config.MinQuorumSize,
	}
}

// Start begins health monitoring
func (cm *ClusterMembership) Start() {
	go cm.healthMonitorLoop()
}

// Stop stops health monitoring
func (cm *ClusterMembership) Stop() {
	close(cm.stopChan)
}

// AddNode adds a new node to the cluster
func (cm *ClusterMembership) AddNode(nodeID string, address string, metadata map[string]string) error {
	cm.mu.Lock()
	defer cm.mu.Unlock()
	
	if _, exists := cm.nodes[nodeID]; exists {
		return fmt.Errorf("node %s already exists", nodeID)
	}
	
	node := &NodeInfo{
		ID:            nodeID,
		Address:       address,
		State:         NodeStateUnknown,
		LastHeartbeat: time.Now(),
		JoinedAt:      time.Now(),
		Metadata:      metadata,
		VoteWeight:    1,
	}
	
	// Extract zone and rack information if available
	if zone, ok := metadata["zone"]; ok {
		node.Zone = zone
	}
	if rack, ok := metadata["rack"]; ok {
		node.Rack = rack
	}
	
	cm.nodes[nodeID] = node
	cm.updateMetrics()
	
	// Notify listeners
	cm.notifyChange(MembershipChange{
		Type:      MembershipAdd,
		NodeID:    nodeID,
		Timestamp: time.Now(),
		Reason:    "Node joined cluster",
	})
	
	return nil
}

// RemoveNode removes a node from the cluster
func (cm *ClusterMembership) RemoveNode(nodeID string, reason string) error {
	cm.mu.Lock()
	defer cm.mu.Unlock()
	
	if _, exists := cm.nodes[nodeID]; !exists {
		return fmt.Errorf("node %s does not exist", nodeID)
	}
	
	delete(cm.nodes, nodeID)
	cm.updateMetrics()
	
	// Notify listeners
	cm.notifyChange(MembershipChange{
		Type:      MembershipRemove,
		NodeID:    nodeID,
		Timestamp: time.Now(),
		Reason:    reason,
	})
	
	return nil
}

// UpdateNodeHealth updates the health status of a node
func (cm *ClusterMembership) UpdateNodeHealth(nodeID string, metrics HealthMetrics) error {
	cm.mu.Lock()
	defer cm.mu.Unlock()
	
	node, exists := cm.nodes[nodeID]
	if !exists {
		return fmt.Errorf("node %s does not exist", nodeID)
	}
	
	node.HealthMetrics = metrics
	node.LastHeartbeat = time.Now()
	
	// Update node state based on health metrics
	oldState := node.State
	node.State = cm.calculateNodeState(metrics)
	
	if oldState != node.State {
		cm.notifyChange(MembershipChange{
			Type:      MembershipUpdate,
			NodeID:    nodeID,
			Timestamp: time.Now(),
			Reason:    fmt.Sprintf("State changed from %v to %v", oldState, node.State),
		})
	}
	
	cm.updateMetrics()
	return nil
}

// GetNode returns information about a specific node
func (cm *ClusterMembership) GetNode(nodeID string) (*NodeInfo, bool) {
	cm.mu.RLock()
	defer cm.mu.RUnlock()
	
	node, exists := cm.nodes[nodeID]
	if !exists {
		return nil, false
	}
	
	// Return a copy to prevent external modification
	nodeCopy := *node
	return &nodeCopy, true
}

// GetAllNodes returns information about all nodes
func (cm *ClusterMembership) GetAllNodes() map[string]*NodeInfo {
	cm.mu.RLock()
	defer cm.mu.RUnlock()
	
	nodesCopy := make(map[string]*NodeInfo)
	for id, node := range cm.nodes {
		nodeCopy := *node
		nodesCopy[id] = &nodeCopy
	}
	
	return nodesCopy
}

// GetHealthyNodes returns all healthy nodes
func (cm *ClusterMembership) GetHealthyNodes() []string {
	cm.mu.RLock()
	defer cm.mu.RUnlock()
	
	var healthyNodes []string
	for id, node := range cm.nodes {
		if node.State == NodeStateHealthy {
			healthyNodes = append(healthyNodes, id)
		}
	}
	
	return healthyNodes
}

// HasQuorum checks if the cluster has quorum
func (cm *ClusterMembership) HasQuorum() bool {
	cm.mu.RLock()
	defer cm.mu.RUnlock()
	
	healthyCount := 0
	for _, node := range cm.nodes {
		if node.State == NodeStateHealthy || node.State == NodeStateDegraded {
			healthyCount++
		}
	}
	
	return healthyCount >= cm.quorumNodes
}

// RegisterChangeListener registers a listener for membership changes
func (cm *ClusterMembership) RegisterChangeListener(listener func(MembershipChange)) {
	cm.mu.Lock()
	defer cm.mu.Unlock()
	
	cm.changeListeners = append(cm.changeListeners, listener)
}

// GetMetrics returns current membership metrics
func (cm *ClusterMembership) GetMetrics() MembershipMetrics {
	cm.mu.RLock()
	defer cm.mu.RUnlock()
	
	return cm.metrics
}

// healthMonitorLoop continuously monitors node health
func (cm *ClusterMembership) healthMonitorLoop() {
	ticker := time.NewTicker(cm.config.HealthCheckInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			cm.performHealthChecks()
		case <-cm.stopChan:
			return
		}
	}
}

// performHealthChecks checks the health of all nodes
func (cm *ClusterMembership) performHealthChecks() {
	nodes := cm.GetAllNodes()
	
	for nodeID, node := range nodes {
		if nodeID == cm.localNodeID {
			// Skip checking ourselves
			continue
		}
		
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		metrics, err := cm.healthChecker.CheckHealth(ctx, nodeID, node.Address)
		cancel()
		
		if err != nil {
			cm.handleHealthCheckFailure(nodeID)
		} else {
			cm.UpdateNodeHealth(nodeID, *metrics)
		}
	}
}

// handleHealthCheckFailure handles a failed health check
func (cm *ClusterMembership) handleHealthCheckFailure(nodeID string) {
	cm.mu.Lock()
	defer cm.mu.Unlock()
	
	node, exists := cm.nodes[nodeID]
	if !exists {
		return
	}
	
	node.HealthMetrics.ConsecutiveFailures++
	
	if node.HealthMetrics.ConsecutiveFailures >= cm.config.FailureThreshold {
		node.State = NodeStateUnhealthy
		
		if node.HealthMetrics.ConsecutiveFailures >= cm.config.FailureThreshold*2 {
			node.State = NodeStateOffline
		}
	}
	
	cm.updateMetrics()
}

// calculateNodeState determines node state based on health metrics
func (cm *ClusterMembership) calculateNodeState(metrics HealthMetrics) NodeState {
	if metrics.ConsecutiveFailures > 0 {
		if metrics.ConsecutiveFailures >= cm.config.FailureThreshold {
			return NodeStateUnhealthy
		}
		return NodeStateDegraded
	}
	
	// Check resource usage
	if metrics.CPUUsage > 0.9 || metrics.MemoryUsage > 0.9 || metrics.DiskUsage > 0.9 {
		return NodeStateDegraded
	}
	
	// Check error rate
	if metrics.ErrorRate > 0.05 {
		return NodeStateDegraded
	}
	
	// Check network latency
	if metrics.NetworkLatency > 100*time.Millisecond {
		return NodeStateDegraded
	}
	
	return NodeStateHealthy
}

// updateMetrics updates membership metrics
func (cm *ClusterMembership) updateMetrics() {
	cm.metrics.TotalNodes = len(cm.nodes)
	cm.metrics.HealthyNodes = 0
	cm.metrics.UnhealthyNodes = 0
	
	for _, node := range cm.nodes {
		switch node.State {
		case NodeStateHealthy:
			cm.metrics.HealthyNodes++
		case NodeStateUnhealthy, NodeStateOffline:
			cm.metrics.UnhealthyNodes++
		}
	}
	
	// Check for quorum loss
	if !cm.HasQuorum() && cm.metrics.LastQuorumLoss.IsZero() {
		cm.metrics.LastQuorumLoss = time.Now()
		cm.metrics.QuorumLossCount++
	}
}

// notifyChange notifies all listeners of a membership change
func (cm *ClusterMembership) notifyChange(change MembershipChange) {
	cm.metrics.MembershipChanges++
	
	for _, listener := range cm.changeListeners {
		go listener(change)
	}
}

// GetZoneDistribution returns the distribution of nodes across zones
func (cm *ClusterMembership) GetZoneDistribution() map[string][]string {
	cm.mu.RLock()
	defer cm.mu.RUnlock()
	
	distribution := make(map[string][]string)
	for id, node := range cm.nodes {
		if node.Zone != "" {
			distribution[node.Zone] = append(distribution[node.Zone], id)
		}
	}
	
	return distribution
}

// IsNodeHealthy checks if a specific node is healthy
func (cm *ClusterMembership) IsNodeHealthy(nodeID string) bool {
	cm.mu.RLock()
	defer cm.mu.RUnlock()
	
	node, exists := cm.nodes[nodeID]
	if !exists {
		return false
	}
	
	return node.State == NodeStateHealthy || node.State == NodeStateDegraded
}

// GetLeaderCandidates returns nodes eligible to be leaders
func (cm *ClusterMembership) GetLeaderCandidates() []string {
	cm.mu.RLock()
	defer cm.mu.RUnlock()
	
	var candidates []string
	for id, node := range cm.nodes {
		if node.State == NodeStateHealthy {
			candidates = append(candidates, id)
		}
	}
	
	return candidates
}