// Package edge provides edge computing capabilities for NovaCron
package edge

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gorilla/websocket"
	"github.com/prometheus/client_golang/prometheus"
)

// NodeType represents the type of edge node
type NodeType string

const (
	NodeTypeCompute  NodeType = "compute"
	NodeTypeStorage  NodeType = "storage"
	NodeTypeGateway  NodeType = "gateway"
	NodeTypeHybrid   NodeType = "hybrid"
	NodeTypeInference NodeType = "inference"
)

// NodeState represents the state of an edge node
type NodeState string

const (
	NodeStateDiscovering NodeState = "discovering"
	NodeStateRegistering NodeState = "registering"
	NodeStateActive      NodeState = "active"
	NodeStateIdle        NodeState = "idle"
	NodeStateBusy        NodeState = "busy"
	NodeStateDraining    NodeState = "draining"
	NodeStateMaintenance NodeState = "maintenance"
	NodeStateOffline     NodeState = "offline"
	NodeStateFailed      NodeState = "failed"
)

// NodeCapability represents a specific capability of an edge node
type NodeCapability struct {
	Type        string                 `json:"type"`
	Version     string                 `json:"version"`
	Capacity    float64                `json:"capacity"`
	Available   float64                `json:"available"`
	Properties  map[string]interface{} `json:"properties"`
	LastUpdated time.Time              `json:"last_updated"`
}

// EdgeNode represents an edge computing node
type EdgeNode struct {
	ID           string                    `json:"id"`
	Name         string                    `json:"name"`
	Type         NodeType                  `json:"type"`
	State        NodeState                 `json:"state"`
	Location     GeographicLocation        `json:"location"`
	Capabilities map[string]NodeCapability `json:"capabilities"`
	Resources    EdgeResources             `json:"resources"`
	Network      NetworkInfo               `json:"network"`
	Health       HealthStatus              `json:"health"`
	Metadata     map[string]string         `json:"metadata"`
	RegisteredAt time.Time                 `json:"registered_at"`
	LastSeen     time.Time                 `json:"last_seen"`
	LastHeartbeat time.Time                `json:"last_heartbeat"`
	mu           sync.RWMutex
}

// GeographicLocation represents the geographic location of an edge node
type GeographicLocation struct {
	Latitude     float64 `json:"latitude"`
	Longitude    float64 `json:"longitude"`
	City         string  `json:"city"`
	Country      string  `json:"country"`
	Region       string  `json:"region"`
	DataCenter   string  `json:"datacenter"`
	Zone         string  `json:"zone"`
	Provider     string  `json:"provider"`
}

// EdgeResources represents the resources available on an edge node
type EdgeResources struct {
	CPU         CPUResources         `json:"cpu"`
	Memory      MemoryResources      `json:"memory"`
	Storage     StorageResources     `json:"storage"`
	Network     NetworkResources     `json:"network"`
	GPU         []GPUResource        `json:"gpu,omitempty"`
	Accelerators []AcceleratorResource `json:"accelerators,omitempty"`
}

// CPUResources represents CPU resources
type CPUResources struct {
	Cores      int     `json:"cores"`
	Threads    int     `json:"threads"`
	Frequency  float64 `json:"frequency_ghz"`
	Usage      float64 `json:"usage_percent"`
	Available  float64 `json:"available_percent"`
	Architecture string `json:"architecture"`
}

// MemoryResources represents memory resources
type MemoryResources struct {
	Total     uint64  `json:"total_bytes"`
	Used      uint64  `json:"used_bytes"`
	Available uint64  `json:"available_bytes"`
	Usage     float64 `json:"usage_percent"`
	SwapTotal uint64  `json:"swap_total_bytes"`
	SwapUsed  uint64  `json:"swap_used_bytes"`
}

// StorageResources represents storage resources
type StorageResources struct {
	Total     uint64  `json:"total_bytes"`
	Used      uint64  `json:"used_bytes"`
	Available uint64  `json:"available_bytes"`
	Usage     float64 `json:"usage_percent"`
	Type      string  `json:"type"` // SSD, HDD, NVMe
	IOPS      int64   `json:"iops"`
}

// NetworkResources represents network resources
type NetworkResources struct {
	BandwidthMbps   float64 `json:"bandwidth_mbps"`
	LatencyMs       float64 `json:"latency_ms"`
	PacketLoss      float64 `json:"packet_loss_percent"`
	ConnectionType  string  `json:"connection_type"`
	PublicIP        string  `json:"public_ip"`
	PrivateIP       string  `json:"private_ip"`
}

// GPUResource represents GPU resources
type GPUResource struct {
	ID          string  `json:"id"`
	Model       string  `json:"model"`
	MemoryBytes uint64  `json:"memory_bytes"`
	Usage       float64 `json:"usage_percent"`
	Temperature float64 `json:"temperature_celsius"`
}

// AcceleratorResource represents specialized accelerator resources
type AcceleratorResource struct {
	Type        string  `json:"type"` // TPU, FPGA, etc.
	Model       string  `json:"model"`
	Capacity    float64 `json:"capacity"`
	Usage       float64 `json:"usage_percent"`
}

// NetworkInfo contains network information for an edge node
type NetworkInfo struct {
	PublicIP       string            `json:"public_ip"`
	PrivateIP      string            `json:"private_ip"`
	Hostname       string            `json:"hostname"`
	Port           int               `json:"port"`
	Protocol       string            `json:"protocol"`
	NAT            bool              `json:"nat"`
	Firewall       bool              `json:"firewall"`
	ConnectedPeers []string          `json:"connected_peers"`
	Routes         []NetworkRoute    `json:"routes"`
	Latencies      map[string]float64 `json:"latencies"` // To other nodes
}

// NetworkRoute represents a network route
type NetworkRoute struct {
	Destination string  `json:"destination"`
	Gateway     string  `json:"gateway"`
	Metric      int     `json:"metric"`
	Latency     float64 `json:"latency_ms"`
}

// HealthStatus represents the health status of an edge node
type HealthStatus struct {
	Status       string             `json:"status"`
	Score        float64            `json:"score"` // 0-100
	Issues       []HealthIssue      `json:"issues"`
	Metrics      map[string]float64 `json:"metrics"`
	LastCheck    time.Time          `json:"last_check"`
}

// HealthIssue represents a health issue
type HealthIssue struct {
	Type        string    `json:"type"`
	Severity    string    `json:"severity"`
	Message     string    `json:"message"`
	DetectedAt  time.Time `json:"detected_at"`
}

// NodeManager manages edge nodes
type NodeManager struct {
	nodes           sync.Map // map[string]*EdgeNode
	nodesByLocation sync.Map // map[string][]*EdgeNode
	nodesByType     sync.Map // map[NodeType][]*EdgeNode
	discovery       *DiscoveryService
	provisioner     *NodeProvisioner
	healthChecker   *HealthChecker
	metrics         *NodeMetrics
	config          *NodeManagerConfig
	ctx             context.Context
	cancel          context.CancelFunc
	wg              sync.WaitGroup
}

// NodeManagerConfig contains configuration for the node manager
type NodeManagerConfig struct {
	DiscoveryInterval    time.Duration
	HealthCheckInterval  time.Duration
	HeartbeatTimeout     time.Duration
	MaxNodesPerLocation  int
	AutoProvision        bool
	ProvisionThreshold   float64
	EnableGeoRouting     bool
	EnableLoadBalancing  bool
	MetricsEnabled       bool
}

// DiscoveryService handles node discovery
type DiscoveryService struct {
	protocols     []DiscoveryProtocol
	discovered    sync.Map
	mu            sync.RWMutex
	broadcastAddr string
	multicastAddr string
}

// DiscoveryProtocol represents a node discovery protocol
type DiscoveryProtocol interface {
	Discover(ctx context.Context) ([]*EdgeNode, error)
	Announce(node *EdgeNode) error
}

// NodeProvisioner handles automatic node provisioning
type NodeProvisioner struct {
	templates     map[NodeType]*NodeTemplate
	provisioning  sync.Map
	maxConcurrent int
	mu            sync.RWMutex
}

// NodeTemplate represents a template for provisioning nodes
type NodeTemplate struct {
	Type         NodeType
	MinResources EdgeResources
	InitScript   string
	Config       map[string]interface{}
}

// HealthChecker monitors node health
type HealthChecker struct {
	checks       []HealthCheck
	mu           sync.RWMutex
	checkTimeout time.Duration
}

// HealthCheck represents a health check
type HealthCheck interface {
	Check(node *EdgeNode) (*HealthStatus, error)
}

// NodeMetrics tracks node metrics
type NodeMetrics struct {
	nodeCount        *prometheus.GaugeVec
	nodeHealth       *prometheus.GaugeVec
	nodeUtilization  *prometheus.GaugeVec
	discoveryLatency prometheus.Histogram
	provisioningTime prometheus.Histogram
}

// NewNodeManager creates a new node manager
func NewNodeManager(config *NodeManagerConfig) *NodeManager {
	ctx, cancel := context.WithCancel(context.Background())

	nm := &NodeManager{
		config:        config,
		ctx:           ctx,
		cancel:        cancel,
		discovery:     NewDiscoveryService(),
		provisioner:   NewNodeProvisioner(),
		healthChecker: NewHealthChecker(),
		metrics:       NewNodeMetrics(),
	}

	// Start background services
	nm.wg.Add(3)
	go nm.discoveryLoop()
	go nm.healthCheckLoop()
	go nm.provisioningLoop()

	return nm
}

// NewDiscoveryService creates a new discovery service
func NewDiscoveryService() *DiscoveryService {
	return &DiscoveryService{
		protocols:     []DiscoveryProtocol{},
		broadcastAddr: "239.255.255.250:1900",
		multicastAddr: "ff02::1",
	}
}

// NewNodeProvisioner creates a new node provisioner
func NewNodeProvisioner() *NodeProvisioner {
	return &NodeProvisioner{
		templates:     make(map[NodeType]*NodeTemplate),
		maxConcurrent: 10,
	}
}

// NewHealthChecker creates a new health checker
func NewHealthChecker() *HealthChecker {
	return &HealthChecker{
		checks:       []HealthCheck{},
		checkTimeout: 30 * time.Second,
	}
}

// NewNodeMetrics creates new node metrics
func NewNodeMetrics() *NodeMetrics {
	return &NodeMetrics{
		nodeCount: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "edge_nodes_total",
				Help: "Total number of edge nodes",
			},
			[]string{"type", "state", "location"},
		),
		nodeHealth: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "edge_node_health_score",
				Help: "Health score of edge nodes",
			},
			[]string{"node_id", "type"},
		),
		nodeUtilization: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "edge_node_utilization",
				Help: "Resource utilization of edge nodes",
			},
			[]string{"node_id", "resource"},
		),
		discoveryLatency: prometheus.NewHistogram(
			prometheus.HistogramOpts{
				Name:    "edge_discovery_duration_seconds",
				Help:    "Time taken to discover nodes",
				Buckets: prometheus.DefBuckets,
			},
		),
		provisioningTime: prometheus.NewHistogram(
			prometheus.HistogramOpts{
				Name:    "edge_provisioning_duration_seconds",
				Help:    "Time taken to provision nodes",
				Buckets: prometheus.DefBuckets,
			},
		),
	}
}

// RegisterNode registers a new edge node
func (nm *NodeManager) RegisterNode(node *EdgeNode) error {
	// Validate node
	if err := nm.validateNode(node); err != nil {
		return fmt.Errorf("node validation failed: %w", err)
	}

	// Detect capabilities
	nm.detectCapabilities(node)

	// Set initial state
	node.State = NodeStateRegistering
	node.RegisteredAt = time.Now()
	node.LastSeen = time.Now()
	node.LastHeartbeat = time.Now()

	// Store node
	nm.nodes.Store(node.ID, node)

	// Index by location
	nm.indexByLocation(node)

	// Index by type
	nm.indexByType(node)

	// Update metrics
	nm.updateNodeMetrics(node)

	// Trigger provisioning if needed
	if nm.config.AutoProvision {
		nm.provisioner.QueueForProvisioning(node)
	}

	node.State = NodeStateActive

	return nil
}

// UnregisterNode unregisters an edge node
func (nm *NodeManager) UnregisterNode(nodeID string) error {
	nodeInterface, exists := nm.nodes.Load(nodeID)
	if !exists {
		return fmt.Errorf("node %s not found", nodeID)
	}

	node := nodeInterface.(*EdgeNode)

	// Set state to offline
	node.State = NodeStateOffline

	// Remove from indexes
	nm.removeFromIndexes(node)

	// Delete node
	nm.nodes.Delete(nodeID)

	// Update metrics
	nm.updateNodeMetrics(node)

	return nil
}

// GetNode retrieves a node by ID
func (nm *NodeManager) GetNode(nodeID string) (*EdgeNode, error) {
	nodeInterface, exists := nm.nodes.Load(nodeID)
	if !exists {
		return nil, fmt.Errorf("node %s not found", nodeID)
	}

	return nodeInterface.(*EdgeNode), nil
}

// GetNodesByLocation retrieves nodes by location
func (nm *NodeManager) GetNodesByLocation(location string) ([]*EdgeNode, error) {
	nodesInterface, exists := nm.nodesByLocation.Load(location)
	if !exists {
		return []*EdgeNode{}, nil
	}

	return nodesInterface.([]*EdgeNode), nil
}

// GetNodesByType retrieves nodes by type
func (nm *NodeManager) GetNodesByType(nodeType NodeType) ([]*EdgeNode, error) {
	nodesInterface, exists := nm.nodesByType.Load(nodeType)
	if !exists {
		return []*EdgeNode{}, nil
	}

	return nodesInterface.([]*EdgeNode), nil
}

// GetActiveNodes retrieves all active nodes
func (nm *NodeManager) GetActiveNodes() []*EdgeNode {
	var activeNodes []*EdgeNode

	nm.nodes.Range(func(key, value interface{}) bool {
		node := value.(*EdgeNode)
		if node.State == NodeStateActive || node.State == NodeStateIdle || node.State == NodeStateBusy {
			activeNodes = append(activeNodes, node)
		}
		return true
	})

	return activeNodes
}

// UpdateNodeHealth updates the health status of a node
func (nm *NodeManager) UpdateNodeHealth(nodeID string, health *HealthStatus) error {
	node, err := nm.GetNode(nodeID)
	if err != nil {
		return err
	}

	node.mu.Lock()
	defer node.mu.Unlock()

	node.Health = *health
	node.LastSeen = time.Now()

	// Update state based on health
	if health.Score < 30 {
		node.State = NodeStateFailed
	} else if health.Score < 60 {
		node.State = NodeStateMaintenance
	} else if node.State == NodeStateFailed || node.State == NodeStateMaintenance {
		node.State = NodeStateActive
	}

	// Update metrics
	nm.metrics.nodeHealth.WithLabelValues(node.ID, string(node.Type)).Set(health.Score)

	return nil
}

// Heartbeat processes a heartbeat from a node
func (nm *NodeManager) Heartbeat(nodeID string, resources *EdgeResources) error {
	node, err := nm.GetNode(nodeID)
	if err != nil {
		return err
	}

	node.mu.Lock()
	defer node.mu.Unlock()

	node.LastHeartbeat = time.Now()
	node.LastSeen = time.Now()

	if resources != nil {
		node.Resources = *resources
		nm.updateResourceMetrics(node)
	}

	// Update state if offline
	if node.State == NodeStateOffline {
		node.State = NodeStateActive
	}

	return nil
}

// discoveryLoop runs the discovery service
func (nm *NodeManager) discoveryLoop() {
	defer nm.wg.Done()

	ticker := time.NewTicker(nm.config.DiscoveryInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			nm.discoverNodes()
		case <-nm.ctx.Done():
			return
		}
	}
}

// healthCheckLoop runs health checks
func (nm *NodeManager) healthCheckLoop() {
	defer nm.wg.Done()

	ticker := time.NewTicker(nm.config.HealthCheckInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			nm.checkNodeHealth()
		case <-nm.ctx.Done():
			return
		}
	}
}

// provisioningLoop handles automatic provisioning
func (nm *NodeManager) provisioningLoop() {
	defer nm.wg.Done()

	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			nm.checkProvisioningNeeds()
		case <-nm.ctx.Done():
			return
		}
	}
}

// discoverNodes discovers new edge nodes
func (nm *NodeManager) discoverNodes() {
	start := time.Now()
	defer func() {
		nm.metrics.discoveryLatency.Observe(time.Since(start).Seconds())
	}()

	for _, protocol := range nm.discovery.protocols {
		nodes, err := protocol.Discover(nm.ctx)
		if err != nil {
			continue
		}

		for _, node := range nodes {
			// Check if already registered
			if _, exists := nm.nodes.Load(node.ID); !exists {
				nm.RegisterNode(node)
			}
		}
	}
}

// checkNodeHealth checks the health of all nodes
func (nm *NodeManager) checkNodeHealth() {
	nm.nodes.Range(func(key, value interface{}) bool {
		node := value.(*EdgeNode)

		// Check heartbeat timeout
		if time.Since(node.LastHeartbeat) > nm.config.HeartbeatTimeout {
			node.State = NodeStateOffline
			return true
		}

		// Run health checks
		for _, check := range nm.healthChecker.checks {
			health, err := check.Check(node)
			if err == nil {
				nm.UpdateNodeHealth(node.ID, health)
			}
		}

		return true
	})
}

// checkProvisioningNeeds checks if new nodes need to be provisioned
func (nm *NodeManager) checkProvisioningNeeds() {
	if !nm.config.AutoProvision {
		return
	}

	// Calculate current utilization
	activeNodes := nm.GetActiveNodes()
	totalCapacity := nm.calculateTotalCapacity(activeNodes)
	currentUtilization := nm.calculateUtilization(activeNodes)

	// Check if provisioning is needed
	if currentUtilization > nm.config.ProvisionThreshold {
		// Determine node type to provision
		nodeType := nm.determineNodeTypeToProvision(totalCapacity)

		// Provision new node
		nm.provisionNode(nodeType)
	}
}

// Helper methods

func (nm *NodeManager) validateNode(node *EdgeNode) error {
	if node.ID == "" {
		return fmt.Errorf("node ID is required")
	}
	if node.Type == "" {
		return fmt.Errorf("node type is required")
	}
	return nil
}

func (nm *NodeManager) detectCapabilities(node *EdgeNode) {
	if node.Capabilities == nil {
		node.Capabilities = make(map[string]NodeCapability)
	}

	// Detect compute capability
	if node.Resources.CPU.Cores > 0 {
		node.Capabilities["compute"] = NodeCapability{
			Type:     "compute",
			Version:  "1.0",
			Capacity: float64(node.Resources.CPU.Cores),
			Available: float64(node.Resources.CPU.Cores) * (100 - node.Resources.CPU.Usage) / 100,
			Properties: map[string]interface{}{
				"architecture": node.Resources.CPU.Architecture,
				"frequency":    node.Resources.CPU.Frequency,
			},
			LastUpdated: time.Now(),
		}
	}

	// Detect GPU capability
	if len(node.Resources.GPU) > 0 {
		node.Capabilities["gpu"] = NodeCapability{
			Type:     "gpu",
			Version:  "1.0",
			Capacity: float64(len(node.Resources.GPU)),
			Available: float64(len(node.Resources.GPU)),
			Properties: map[string]interface{}{
				"models": node.Resources.GPU,
			},
			LastUpdated: time.Now(),
		}
	}

	// Detect storage capability
	if node.Resources.Storage.Total > 0 {
		node.Capabilities["storage"] = NodeCapability{
			Type:     "storage",
			Version:  "1.0",
			Capacity: float64(node.Resources.Storage.Total),
			Available: float64(node.Resources.Storage.Available),
			Properties: map[string]interface{}{
				"type": node.Resources.Storage.Type,
				"iops": node.Resources.Storage.IOPS,
			},
			LastUpdated: time.Now(),
		}
	}
}

func (nm *NodeManager) indexByLocation(node *EdgeNode) {
	location := fmt.Sprintf("%s:%s", node.Location.Country, node.Location.City)

	nodesInterface, _ := nm.nodesByLocation.LoadOrStore(location, []*EdgeNode{})
	nodes := nodesInterface.([]*EdgeNode)
	nodes = append(nodes, node)
	nm.nodesByLocation.Store(location, nodes)
}

func (nm *NodeManager) indexByType(node *EdgeNode) {
	nodesInterface, _ := nm.nodesByType.LoadOrStore(node.Type, []*EdgeNode{})
	nodes := nodesInterface.([]*EdgeNode)
	nodes = append(nodes, node)
	nm.nodesByType.Store(node.Type, nodes)
}

func (nm *NodeManager) removeFromIndexes(node *EdgeNode) {
	// Remove from location index
	location := fmt.Sprintf("%s:%s", node.Location.Country, node.Location.City)
	if nodesInterface, exists := nm.nodesByLocation.Load(location); exists {
		nodes := nodesInterface.([]*EdgeNode)
		filtered := []*EdgeNode{}
		for _, n := range nodes {
			if n.ID != node.ID {
				filtered = append(filtered, n)
			}
		}
		nm.nodesByLocation.Store(location, filtered)
	}

	// Remove from type index
	if nodesInterface, exists := nm.nodesByType.Load(node.Type); exists {
		nodes := nodesInterface.([]*EdgeNode)
		filtered := []*EdgeNode{}
		for _, n := range nodes {
			if n.ID != node.ID {
				filtered = append(filtered, n)
			}
		}
		nm.nodesByType.Store(node.Type, filtered)
	}
}

func (nm *NodeManager) updateNodeMetrics(node *EdgeNode) {
	location := fmt.Sprintf("%s:%s", node.Location.Country, node.Location.City)
	nm.metrics.nodeCount.WithLabelValues(
		string(node.Type),
		string(node.State),
		location,
	).Set(1)
}

func (nm *NodeManager) updateResourceMetrics(node *EdgeNode) {
	nm.metrics.nodeUtilization.WithLabelValues(node.ID, "cpu").Set(node.Resources.CPU.Usage)
	nm.metrics.nodeUtilization.WithLabelValues(node.ID, "memory").Set(node.Resources.Memory.Usage)
	nm.metrics.nodeUtilization.WithLabelValues(node.ID, "storage").Set(node.Resources.Storage.Usage)
}

func (nm *NodeManager) calculateTotalCapacity(nodes []*EdgeNode) float64 {
	var totalCapacity float64
	for _, node := range nodes {
		totalCapacity += float64(node.Resources.CPU.Cores)
		totalCapacity += float64(node.Resources.Memory.Total) / (1024 * 1024 * 1024) // GB
		totalCapacity += float64(node.Resources.Storage.Total) / (1024 * 1024 * 1024) // GB
	}
	return totalCapacity
}

func (nm *NodeManager) calculateUtilization(nodes []*EdgeNode) float64 {
	if len(nodes) == 0 {
		return 0
	}

	var totalUtilization float64
	for _, node := range nodes {
		totalUtilization += node.Resources.CPU.Usage
		totalUtilization += node.Resources.Memory.Usage
		totalUtilization += node.Resources.Storage.Usage
	}

	return totalUtilization / float64(len(nodes)*3) // Average across CPU, Memory, Storage
}

func (nm *NodeManager) determineNodeTypeToProvision(capacity float64) NodeType {
	// Logic to determine which type of node to provision
	// based on current capacity and demand
	if capacity < 100 {
		return NodeTypeCompute
	} else if capacity < 500 {
		return NodeTypeHybrid
	}
	return NodeTypeStorage
}

func (nm *NodeManager) provisionNode(nodeType NodeType) {
	start := time.Now()
	defer func() {
		nm.metrics.provisioningTime.Observe(time.Since(start).Seconds())
	}()

	template, exists := nm.provisioner.templates[nodeType]
	if !exists {
		return
	}

	// Create new node from template
	node := &EdgeNode{
		ID:           fmt.Sprintf("edge-%d", time.Now().UnixNano()),
		Type:         nodeType,
		State:        NodeStateDiscovering,
		Resources:    template.MinResources,
		RegisteredAt: time.Now(),
	}

	// Register the provisioned node
	nm.RegisterNode(node)
}

// QueueForProvisioning queues a node for provisioning
func (p *NodeProvisioner) QueueForProvisioning(node *EdgeNode) {
	p.provisioning.Store(node.ID, node)
}

// Stop stops the node manager
func (nm *NodeManager) Stop() {
	nm.cancel()
	nm.wg.Wait()
}

// GetDistance calculates the distance between two geographic points
func GetDistance(lat1, lon1, lat2, lon2 float64) float64 {
	const R = 6371 // Earth radius in kilometers

	dLat := (lat2 - lat1) * math.Pi / 180
	dLon := (lon2 - lon1) * math.Pi / 180

	a := math.Sin(dLat/2)*math.Sin(dLat/2) +
		math.Cos(lat1*math.Pi/180)*math.Cos(lat2*math.Pi/180)*
		math.Sin(dLon/2)*math.Sin(dLon/2)

	c := 2 * math.Atan2(math.Sqrt(a), math.Sqrt(1-a))

	return R * c
}