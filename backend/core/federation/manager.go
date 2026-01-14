package federation

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	// Metrics
	federationNodes = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "novacron_federation_nodes_total",
		Help: "Total number of nodes in the federation",
	}, []string{"state"})

	federationRequests = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "novacron_federation_requests_total",
		Help: "Total number of resource requests",
	}, []string{"status"})

	federationLatency = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "novacron_federation_operation_duration_seconds",
		Help:    "Duration of federation operations",
		Buckets: prometheus.DefBuckets,
	}, []string{"operation"})
)

// Manager implements the FederationManager interface
type Manager struct {
	config           *FederationConfig
	consensus        ConsensusManager
	discovery        DiscoveryManager
	resourcePool     ResourceManager
	healthChecker    HealthChecker
	nodes            map[string]*Node
	nodesMu          sync.RWMutex
	localNode        *Node
	isRunning        atomic.Bool
	stopCh           chan struct{}
	eventCh          chan interface{}
	metrics          *FederationMetrics
	logger           Logger
}

// FederationMetrics tracks federation performance metrics
type FederationMetrics struct {
	NodesTotal         int64
	NodesHealthy       int64
	ResourcesAllocated int64
	RequestsTotal      int64
	RequestsSuccessful int64
	RequestsFailed     int64
	NetworkLatency     time.Duration
	ConsensusLatency   time.Duration
}

// Logger interface for federation logging
type Logger interface {
	Debug(msg string, args ...interface{})
	Info(msg string, args ...interface{})
	Warn(msg string, args ...interface{})
	Error(msg string, args ...interface{})
}

// NewManager creates a new federation manager
func NewManager(config *FederationConfig, logger Logger) (*Manager, error) {
	if config == nil {
		return nil, fmt.Errorf("federation config is required")
	}

	// Generate node ID if not provided
	if config.NodeID == "" {
		config.NodeID = generateNodeID()
	}

	// Create local node
	localNode := &Node{
		ID:        config.NodeID,
		ClusterID: config.ClusterID,
		Address:   config.AdvertiseAddress,
		State:     NodeStateDiscovering,
		Role:      RoleFollower,
		JoinedAt:  time.Now(),
		Version:   "1.0.0",
		Metadata:  make(map[string]string),
	}

	manager := &Manager{
		config:    config,
		nodes:     make(map[string]*Node),
		localNode: localNode,
		stopCh:    make(chan struct{}),
		eventCh:   make(chan interface{}, 1000),
		metrics:   &FederationMetrics{},
		logger:    logger,
	}

	// Initialize components
	if err := manager.initializeComponents(); err != nil {
		return nil, fmt.Errorf("failed to initialize components: %w", err)
	}

	return manager, nil
}

// initializeComponents initializes all federation components
func (m *Manager) initializeComponents() error {
	// Initialize consensus manager (Raft)
	consensus, err := NewRaftConsensus(m.config, m.logger)
	if err != nil {
		return fmt.Errorf("failed to create consensus manager: %w", err)
	}
	m.consensus = consensus

	// Initialize discovery manager
	discovery, err := NewDiscoveryManager(m.config, m.logger)
	if err != nil {
		return fmt.Errorf("failed to create discovery manager: %w", err)
	}
	m.discovery = discovery

	// Initialize resource pool
	resourcePool, err := NewResourcePool(m.config, m.logger)
	if err != nil {
		return fmt.Errorf("failed to create resource pool: %w", err)
	}
	m.resourcePool = resourcePool

	// Initialize health checker
	healthChecker, err := NewHealthChecker(m.config, m.logger)
	if err != nil {
		return fmt.Errorf("failed to create health checker: %w", err)
	}
	m.healthChecker = healthChecker

	// Register health handler
	m.healthChecker.RegisterHealthHandler(m)

	return nil
}

// Start starts the federation manager
func (m *Manager) Start(ctx context.Context) error {
	if m.isRunning.Load() {
		return fmt.Errorf("federation manager already running")
	}

	m.logger.Info("Starting federation manager", "node_id", m.localNode.ID, "cluster_id", m.config.ClusterID)

	// Start consensus manager
	if err := m.consensus.Start(ctx); err != nil {
		return fmt.Errorf("failed to start consensus: %w", err)
	}

	// Start discovery manager
	if err := m.discovery.Start(ctx); err != nil {
		return fmt.Errorf("failed to start discovery: %w", err)
	}

	// Start health monitoring
	if err := m.healthChecker.StartMonitoring(ctx, m.config.HeartbeatInterval); err != nil {
		return fmt.Errorf("failed to start health monitoring: %w", err)
	}

	// Update local node state
	m.localNode.State = NodeStateJoining
	m.addNode(m.localNode)

	// Start background workers
	go m.eventProcessor(ctx)
	go m.nodeMaintenanceLoop(ctx)
	go m.metricsCollector(ctx)

	m.isRunning.Store(true)

	// Attempt to join existing federation if addresses provided
	if len(m.config.JoinAddresses) > 0 {
		go func() {
			time.Sleep(2 * time.Second) // Allow services to initialize
			if err := m.JoinFederation(ctx, m.config.JoinAddresses); err != nil {
				m.logger.Error("Failed to join federation", "error", err)
			}
		}()
	} else {
		// Bootstrap as single-node federation
		m.localNode.State = NodeStateActive
		m.localNode.Role = RoleLeader
	}

	m.logger.Info("Federation manager started successfully")
	return nil
}

// Stop stops the federation manager
func (m *Manager) Stop(ctx context.Context) error {
	if !m.isRunning.Load() {
		return fmt.Errorf("federation manager not running")
	}

	m.logger.Info("Stopping federation manager")

	// Leave federation gracefully
	if err := m.LeaveFederation(ctx); err != nil {
		m.logger.Error("Error leaving federation", "error", err)
	}

	// Stop components
	if err := m.healthChecker.StopMonitoring(ctx); err != nil {
		m.logger.Error("Error stopping health checker", "error", err)
	}

	if err := m.discovery.Stop(ctx); err != nil {
		m.logger.Error("Error stopping discovery", "error", err)
	}

	if err := m.consensus.Stop(ctx); err != nil {
		m.logger.Error("Error stopping consensus", "error", err)
	}

	close(m.stopCh)
	m.isRunning.Store(false)

	m.logger.Info("Federation manager stopped")
	return nil
}

// JoinFederation joins an existing federation
func (m *Manager) JoinFederation(ctx context.Context, joinAddresses []string) error {
	timer := prometheus.NewTimer(federationLatency.WithLabelValues("join"))
	defer timer.ObserveDuration()

	m.logger.Info("Joining federation", "addresses", joinAddresses)

	// Discover nodes from join addresses
	for _, addr := range joinAddresses {
		if err := m.discovery.RegisterNode(ctx, &Node{
			Address: addr,
			State:   NodeStateUnknown,
		}); err != nil {
			m.logger.Warn("Failed to register bootstrap node", "address", addr, "error", err)
		}
	}

	// Perform discovery
	discoveredNodes, err := m.discovery.DiscoverNodes(ctx)
	if err != nil {
		return fmt.Errorf("failed to discover nodes: %w", err)
	}

	// Add discovered nodes
	for _, node := range discoveredNodes {
		m.addNode(node)
		if err := m.consensus.AddNode(ctx, node.ID, node.Address); err != nil {
			m.logger.Warn("Failed to add node to consensus", "node_id", node.ID, "error", err)
		}
	}

	// Update local node state
	m.localNode.State = NodeStateActive

	m.logger.Info("Successfully joined federation", "nodes", len(discoveredNodes))
	federationRequests.WithLabelValues("join_success").Inc()

	return nil
}

// LeaveFederation leaves the federation
func (m *Manager) LeaveFederation(ctx context.Context) error {
	timer := prometheus.NewTimer(federationLatency.WithLabelValues("leave"))
	defer timer.ObserveDuration()

	m.logger.Info("Leaving federation")

	// Update local node state
	m.localNode.State = NodeStateLeaving

	// Notify other nodes
	m.broadcastNodeLeave()

	// Release all allocated resources
	allocations, err := m.resourcePool.GetAllocations(ctx)
	if err == nil {
		for _, allocation := range allocations {
			if allocation.NodeID == m.localNode.ID {
				_ = m.resourcePool.ReleaseResources(ctx, allocation.ID)
			}
		}
	}

	// Remove from consensus
	if err := m.consensus.RemoveNode(ctx, m.localNode.ID); err != nil {
		m.logger.Warn("Failed to remove node from consensus", "error", err)
	}

	// Unregister from discovery
	if err := m.discovery.UnregisterNode(ctx, m.localNode.ID); err != nil {
		m.logger.Warn("Failed to unregister from discovery", "error", err)
	}

	m.localNode.State = NodeStateOffline
	federationRequests.WithLabelValues("leave_success").Inc()

	return nil
}

// GetNodes returns all nodes in the federation
func (m *Manager) GetNodes(ctx context.Context) ([]*Node, error) {
	m.nodesMu.RLock()
	defer m.nodesMu.RUnlock()

	nodes := make([]*Node, 0, len(m.nodes))
	for _, node := range m.nodes {
		nodes = append(nodes, node)
	}

	return nodes, nil
}

// GetNode returns a specific node
func (m *Manager) GetNode(ctx context.Context, nodeID string) (*Node, error) {
	m.nodesMu.RLock()
	defer m.nodesMu.RUnlock()

	node, exists := m.nodes[nodeID]
	if !exists {
		return nil, fmt.Errorf("node not found: %s", nodeID)
	}

	return node, nil
}

// GetLeader returns the current leader node
func (m *Manager) GetLeader(ctx context.Context) (*Node, error) {
	leaderID, err := m.consensus.GetLeader()
	if err != nil {
		return nil, fmt.Errorf("failed to get leader: %w", err)
	}

	return m.GetNode(ctx, leaderID)
}

// IsLeader checks if the local node is the leader
func (m *Manager) IsLeader() bool {
	return m.consensus.IsLeader()
}

// RequestResources requests resources from the federation
func (m *Manager) RequestResources(ctx context.Context, request *ResourceRequest) (*ResourceAllocation, error) {
	timer := prometheus.NewTimer(federationLatency.WithLabelValues("resource_request"))
	defer timer.ObserveDuration()

	if request == nil {
		return nil, fmt.Errorf("resource request is nil")
	}

	// Generate request ID if not provided
	if request.ID == "" {
		request.ID = generateRequestID()
	}

	request.RequesterID = m.localNode.ID
	request.CreatedAt = time.Now()

	m.logger.Info("Requesting resources", "request_id", request.ID, "cpu", request.CPUCores, "memory", request.MemoryGB)

	// Attempt allocation
	allocation, err := m.resourcePool.AllocateResources(ctx, request)
	if err != nil {
		federationRequests.WithLabelValues("resource_failed").Inc()
		return nil, fmt.Errorf("failed to allocate resources: %w", err)
	}

	federationRequests.WithLabelValues("resource_success").Inc()
	m.metrics.ResourcesAllocated++

	m.logger.Info("Resources allocated", "allocation_id", allocation.ID, "node_id", allocation.NodeID)

	return allocation, nil
}

// ReleaseResources releases allocated resources
func (m *Manager) ReleaseResources(ctx context.Context, allocationID string) error {
	timer := prometheus.NewTimer(federationLatency.WithLabelValues("resource_release"))
	defer timer.ObserveDuration()

	m.logger.Info("Releasing resources", "allocation_id", allocationID)

	if err := m.resourcePool.ReleaseResources(ctx, allocationID); err != nil {
		return fmt.Errorf("failed to release resources: %w", err)
	}

	m.metrics.ResourcesAllocated--
	federationRequests.WithLabelValues("release_success").Inc()

	return nil
}

// GetHealth returns the health status of the local node
func (m *Manager) GetHealth(ctx context.Context) (*HealthCheck, error) {
	return m.healthChecker.CheckHealth(ctx, m.localNode)
}

// Health handler implementations

// OnNodeHealthy handles healthy node events
func (m *Manager) OnNodeHealthy(node *Node) {
	m.logger.Debug("Node healthy", "node_id", node.ID)
	node.State = NodeStateActive
	federationNodes.WithLabelValues("healthy").Inc()
}

// OnNodeUnhealthy handles unhealthy node events
func (m *Manager) OnNodeUnhealthy(node *Node, issues []string) {
	m.logger.Warn("Node unhealthy", "node_id", node.ID, "issues", issues)
	node.State = NodeStateUnhealthy
	federationNodes.WithLabelValues("unhealthy").Inc()
	federationNodes.WithLabelValues("healthy").Dec()
}

// OnNodeOffline handles offline node events
func (m *Manager) OnNodeOffline(node *Node) {
	m.logger.Warn("Node offline", "node_id", node.ID)
	node.State = NodeStateOffline
	federationNodes.WithLabelValues("offline").Inc()
	federationNodes.WithLabelValues("healthy").Dec()

	// Remove from consensus if offline for too long
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_ = m.consensus.RemoveNode(ctx, node.ID)
}

// Internal methods

func (m *Manager) addNode(node *Node) {
	m.nodesMu.Lock()
	defer m.nodesMu.Unlock()

	m.nodes[node.ID] = node
	federationNodes.WithLabelValues(string(node.State)).Inc()
}

func (m *Manager) removeNode(nodeID string) {
	m.nodesMu.Lock()
	defer m.nodesMu.Unlock()

	if node, exists := m.nodes[nodeID]; exists {
		federationNodes.WithLabelValues(string(node.State)).Dec()
		delete(m.nodes, nodeID)
	}
}

func (m *Manager) broadcastNodeLeave() {
	msg := &GossipMessage{
		ID:        generateMessageID(),
		Type:      GossipNodeLeave,
		Source:    m.localNode.ID,
		Data:      m.localNode,
		TTL:       3,
		Timestamp: time.Now(),
	}

	m.eventCh <- msg
}

func (m *Manager) eventProcessor(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case <-m.stopCh:
			return
		case event := <-m.eventCh:
			m.handleEvent(event)
		}
	}
}

func (m *Manager) handleEvent(event interface{}) {
	switch e := event.(type) {
	case *GossipMessage:
		m.handleGossipMessage(e)
	case *RaftMessage:
		m.handleRaftMessage(e)
	default:
		m.logger.Warn("Unknown event type", "type", fmt.Sprintf("%T", event))
	}
}

func (m *Manager) handleGossipMessage(msg *GossipMessage) {
	switch msg.Type {
	case GossipNodeJoin:
		if node, ok := msg.Data.(*Node); ok {
			m.addNode(node)
		}
	case GossipNodeLeave:
		if node, ok := msg.Data.(*Node); ok {
			m.removeNode(node.ID)
		}
	case GossipNodeUpdate:
		if node, ok := msg.Data.(*Node); ok {
			m.nodesMu.Lock()
			m.nodes[node.ID] = node
			m.nodesMu.Unlock()
		}
	}
}

func (m *Manager) handleRaftMessage(msg *RaftMessage) {
	// Forward to consensus manager
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	data, _ := json.Marshal(msg)
	_ = m.consensus.ProposeValue(ctx, fmt.Sprintf("raft_%s", msg.Type), data)
}

func (m *Manager) nodeMaintenanceLoop(ctx context.Context) {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-m.stopCh:
			return
		case <-ticker.C:
			m.performNodeMaintenance()
		}
	}
}

func (m *Manager) performNodeMaintenance() {
	m.nodesMu.RLock()
	nodes := make([]*Node, 0, len(m.nodes))
	for _, node := range m.nodes {
		nodes = append(nodes, node)
	}
	m.nodesMu.RUnlock()

	now := time.Now()
	for _, node := range nodes {
		// Check for stale nodes
		if now.Sub(node.LastSeen) > 5*m.config.HeartbeatInterval {
			m.logger.Warn("Node appears stale", "node_id", node.ID, "last_seen", node.LastSeen)
			node.State = NodeStateUnhealthy
		}

		// Clean up offline nodes
		if node.State == NodeStateOffline && now.Sub(node.LastSeen) > 10*m.config.HeartbeatInterval {
			m.removeNode(node.ID)
		}
	}
}

func (m *Manager) metricsCollector(ctx context.Context) {
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-m.stopCh:
			return
		case <-ticker.C:
			m.collectMetrics()
		}
	}
}

func (m *Manager) collectMetrics() {
	m.nodesMu.RLock()
	m.metrics.NodesTotal = int64(len(m.nodes))
	
	healthy := int64(0)
	for _, node := range m.nodes {
		if node.State == NodeStateActive {
			healthy++
		}
	}
	m.metrics.NodesHealthy = healthy
	m.nodesMu.RUnlock()

	// Update Prometheus metrics
	federationNodes.WithLabelValues("total").Set(float64(m.metrics.NodesTotal))
}

// Helper functions

func generateNodeID() string {
	b := make([]byte, 16)
	rand.Read(b)
	return hex.EncodeToString(b)
}

func generateRequestID() string {
	b := make([]byte, 8)
	rand.Read(b)
	return "req_" + hex.EncodeToString(b)
}

func generateMessageID() string {
	b := make([]byte, 8)
	rand.Read(b)
	return "msg_" + hex.EncodeToString(b)
}