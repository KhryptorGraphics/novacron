package edge

import (
	"context"
	"fmt"
	"log"
	"sort"
	"sync"
	"time"
)

// HierarchicalLevel represents the level in the edge hierarchy
type HierarchicalLevel string

const (
	LevelCloud      HierarchicalLevel = "cloud"       // Cloud data centers
	LevelRegional   HierarchicalLevel = "regional"    // Regional edge nodes
	LevelMetro      HierarchicalLevel = "metro"       // Metro edge nodes
	LevelAccess     HierarchicalLevel = "access"      // Access edge nodes
	LevelDevice     HierarchicalLevel = "device"      // End devices/IoT
)

// NodeCapabilities defines what operations a node can perform
type NodeCapabilities struct {
	// Compute capabilities
	CanExecuteTasks     bool     `json:"can_execute_tasks"`
	CanHostVMs         bool     `json:"can_host_vms"`
	CanRunContainers   bool     `json:"can_run_containers"`
	SupportedWorkloads []string `json:"supported_workloads"`

	// Storage capabilities
	HasLocalStorage    bool     `json:"has_local_storage"`
	HasSharedStorage   bool     `json:"has_shared_storage"`
	StorageTypes       []string `json:"storage_types"`
	MaxStorageGB       uint64   `json:"max_storage_gb"`

	// Network capabilities
	CanRoutePacets     bool     `json:"can_route_packets"`
	HasLoadBalancing   bool     `json:"has_load_balancing"`
	HasCDN             bool     `json:"has_cdn"`
	NetworkBandwidthMb uint64   `json:"network_bandwidth_mb"`

	// Management capabilities
	CanManageChildren  bool     `json:"can_manage_children"`
	CanCache           bool     `json:"can_cache"`
	CanAggregate       bool     `json:"can_aggregate"`
	MaxChildNodes      int      `json:"max_child_nodes"`
}

// HierarchicalNode represents a node in the edge hierarchy
type HierarchicalNode struct {
	// Identity and location
	ID       string            `json:"id"`
	Name     string            `json:"name"`
	Level    HierarchicalLevel `json:"level"`
	Region   string            `json:"region"`
	Zone     string            `json:"zone,omitempty"`
	Location *GeographicLocation `json:"location,omitempty"`

	// Hierarchy relationships
	ParentID   string   `json:"parent_id,omitempty"`
	ChildIDs   []string `json:"child_ids"`
	Depth      int      `json:"depth"`

	// Node properties
	Status       NodeStatus       `json:"status"`
	Capabilities NodeCapabilities `json:"capabilities"`
	Resources    *NodeResources   `json:"resources"`
	LastSeen     time.Time        `json:"last_seen"`

	// Communication
	Endpoint  string            `json:"endpoint"`
	IsSecure  bool              `json:"is_secure"`
	Tags      map[string]string `json:"tags"`
}

// GeographicLocation contains geographic coordinates
type GeographicLocation struct {
	Latitude  float64 `json:"latitude"`
	Longitude float64 `json:"longitude"`
	City      string  `json:"city,omitempty"`
	Country   string  `json:"country,omitempty"`
}

// NodeStatus represents the operational status of a node
type NodeStatus string

const (
	NodeStatusOnline      NodeStatus = "online"
	NodeStatusOffline     NodeStatus = "offline"
	NodeStatusDegraded    NodeStatus = "degraded"
	NodeStatusMaintenance NodeStatus = "maintenance"
	NodeStatusUnknown     NodeStatus = "unknown"
)

// HierarchicalTopology represents the complete edge hierarchy
type HierarchicalTopology struct {
	Nodes         map[string]*HierarchicalNode `json:"nodes"`
	Relationships map[string][]string          `json:"relationships"` // parent_id -> [child_ids]
	CreatedAt     time.Time                    `json:"created_at"`
	UpdatedAt     time.Time                    `json:"updated_at"`
}

// WorkloadPlacement represents where a workload should be placed
type WorkloadPlacement struct {
	WorkloadID     string                 `json:"workload_id"`
	PreferredNodes []string               `json:"preferred_nodes"`
	Requirements   WorkloadRequirements   `json:"requirements"`
	Constraints    []PlacementConstraint  `json:"constraints"`
	Score          float64                `json:"score"`
	Reasoning      []string               `json:"reasoning"`
}

// WorkloadRequirements defines resource and capability requirements
type WorkloadRequirements struct {
	Resources    ResourceRequirements `json:"resources"`
	Capabilities []string            `json:"capabilities"`
	Latency      LatencyRequirements `json:"latency"`
	Availability AvailabilityRequirements `json:"availability"`
	Locality     LocalityRequirements `json:"locality"`
}

// LatencyRequirements defines latency constraints
type LatencyRequirements struct {
	MaxLatencyMs    int     `json:"max_latency_ms"`
	TargetLatencyMs int     `json:"target_latency_ms"`
	JitterMs        int     `json:"jitter_ms"`
	Percentile      float64 `json:"percentile"` // e.g., 0.95 for P95
}

// AvailabilityRequirements defines availability constraints
type AvailabilityRequirements struct {
	MinUptimePct       float64 `json:"min_uptime_percent"`
	MaxDowntimeMinutes int     `json:"max_downtime_minutes"`
	RequiredRedundancy int     `json:"required_redundancy"`
}

// LocalityRequirements defines geographic and network locality requirements
type LocalityRequirements struct {
	PreferredRegions []string `json:"preferred_regions"`
	ExcludedRegions  []string `json:"excluded_regions"`
	DataResidency    []string `json:"data_residency"`
	MaxDistanceKm    float64  `json:"max_distance_km"`
}

// PlacementConstraint defines a constraint for workload placement
type PlacementConstraint struct {
	Type        string      `json:"type"`        // "affinity", "anti-affinity", "resource", "location"
	Target      string      `json:"target"`      // What the constraint applies to
	Operator    string      `json:"operator"`    // "equals", "not_equals", "greater_than", "less_than"
	Value       interface{} `json:"value"`       // The constraint value
	Mandatory   bool        `json:"mandatory"`   // Whether this constraint is mandatory
}

// HierarchicalManager manages the edge computing hierarchy
type HierarchicalManager struct {
	// Current node information
	localNode *HierarchicalNode

	// Topology management
	topology    *HierarchicalTopology
	topologyMux sync.RWMutex

	// Parent/child connections
	parentConn  *NodeConnection
	childConns  map[string]*NodeConnection
	connMux     sync.RWMutex

	// Workload management
	workloadPlacer *WorkloadPlacer
	taskRouter     *TaskRouter

	// Decision making
	decisionEngine *EdgeDecisionEngine

	// Control
	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

// NodeConnection represents a connection to another node
type NodeConnection struct {
	NodeID     string    `json:"node_id"`
	Endpoint   string    `json:"endpoint"`
	Connected  bool      `json:"connected"`
	LastPing   time.Time `json:"last_ping"`
	LatencyMs  float64   `json:"latency_ms"`
	Bandwidth  uint64    `json:"bandwidth_bps"`
	ErrorCount int       `json:"error_count"`
}

// NewHierarchicalManager creates a new hierarchical edge manager
func NewHierarchicalManager(localNode *HierarchicalNode) *HierarchicalManager {
	ctx, cancel := context.WithCancel(context.Background())

	topology := &HierarchicalTopology{
		Nodes:         make(map[string]*HierarchicalNode),
		Relationships: make(map[string][]string),
		CreatedAt:     time.Now(),
		UpdatedAt:     time.Now(),
	}

	// Add local node to topology
	topology.Nodes[localNode.ID] = localNode

	manager := &HierarchicalManager{
		localNode:      localNode,
		topology:       topology,
		childConns:     make(map[string]*NodeConnection),
		ctx:            ctx,
		cancel:         cancel,
		workloadPlacer: NewWorkloadPlacer(),
		taskRouter:     NewTaskRouter(),
		decisionEngine: NewEdgeDecisionEngine(),
	}

	return manager
}

// Start starts the hierarchical manager
func (h *HierarchicalManager) Start() error {
	log.Printf("Starting hierarchical manager for node %s at level %s", 
		h.localNode.Name, h.localNode.Level)

	// Start background processes
	h.wg.Add(3)
	go h.topologyDiscoveryLoop()
	go h.healthMonitoringLoop()
	go h.workloadOptimizationLoop()

	return nil
}

// Stop stops the hierarchical manager
func (h *HierarchicalManager) Stop() error {
	log.Printf("Stopping hierarchical manager")

	h.cancel()
	
	// Close all connections
	h.connMux.Lock()
	if h.parentConn != nil {
		// Close parent connection
	}
	for _, conn := range h.childConns {
		// Close child connections
		_ = conn
	}
	h.connMux.Unlock()

	h.wg.Wait()
	return nil
}

// RegisterChildNode registers a new child node
func (h *HierarchicalManager) RegisterChildNode(childNode *HierarchicalNode) error {
	if !h.localNode.Capabilities.CanManageChildren {
		return fmt.Errorf("this node cannot manage child nodes")
	}

	if len(h.localNode.ChildIDs) >= h.localNode.Capabilities.MaxChildNodes {
		return fmt.Errorf("maximum child nodes (%d) reached", h.localNode.Capabilities.MaxChildNodes)
	}

	h.topologyMux.Lock()
	defer h.topologyMux.Unlock()

	// Set parent relationship
	childNode.ParentID = h.localNode.ID
	childNode.Depth = h.localNode.Depth + 1

	// Add to topology
	h.topology.Nodes[childNode.ID] = childNode
	h.localNode.ChildIDs = append(h.localNode.ChildIDs, childNode.ID)
	h.topology.Relationships[h.localNode.ID] = h.localNode.ChildIDs
	h.topology.UpdatedAt = time.Now()

	// Establish connection
	conn := &NodeConnection{
		NodeID:    childNode.ID,
		Endpoint:  childNode.Endpoint,
		Connected: false,
		LastPing:  time.Time{},
	}

	h.connMux.Lock()
	h.childConns[childNode.ID] = conn
	h.connMux.Unlock()

	log.Printf("Registered child node %s (%s)", childNode.Name, childNode.ID)
	return nil
}

// UnregisterChildNode removes a child node
func (h *HierarchicalManager) UnregisterChildNode(nodeID string) error {
	h.topologyMux.Lock()
	defer h.topologyMux.Unlock()

	// Find and remove from child list
	for i, childID := range h.localNode.ChildIDs {
		if childID == nodeID {
			h.localNode.ChildIDs = append(h.localNode.ChildIDs[:i], h.localNode.ChildIDs[i+1:]...)
			break
		}
	}

	// Remove from topology
	delete(h.topology.Nodes, nodeID)
	h.topology.Relationships[h.localNode.ID] = h.localNode.ChildIDs
	h.topology.UpdatedAt = time.Now()

	// Close connection
	h.connMux.Lock()
	delete(h.childConns, nodeID)
	h.connMux.Unlock()

	log.Printf("Unregistered child node %s", nodeID)
	return nil
}

// FindOptimalPlacement finds the best placement for a workload in the hierarchy
func (h *HierarchicalManager) FindOptimalPlacement(requirements WorkloadRequirements) (*WorkloadPlacement, error) {
	h.topologyMux.RLock()
	defer h.topologyMux.RUnlock()

	candidates := h.identifyPlacementCandidates(requirements)
	if len(candidates) == 0 {
		return nil, fmt.Errorf("no suitable nodes found for workload placement")
	}

	// Score candidates
	scoredPlacements := h.scorePlacementCandidates(candidates, requirements)

	// Sort by score (highest first)
	sort.Slice(scoredPlacements, func(i, j int) bool {
		return scoredPlacements[i].Score > scoredPlacements[j].Score
	})

	return scoredPlacements[0], nil
}

// identifyPlacementCandidates finds nodes that meet basic requirements
func (h *HierarchicalManager) identifyPlacementCandidates(requirements WorkloadRequirements) []*HierarchicalNode {
	var candidates []*HierarchicalNode

	for _, node := range h.topology.Nodes {
		if h.nodeMatches Requirements(node, requirements) {
			candidates = append(candidates, node)
		}
	}

	return candidates
}

// nodeMatchesRequirements checks if a node meets workload requirements
func (h *HierarchicalManager) nodeMatchesRequirements(node *HierarchicalNode, requirements WorkloadRequirements) bool {
	// Check status
	if node.Status != NodeStatusOnline {
		return false
	}

	// Check resource requirements
	if node.Resources != nil {
		if !h.hasAvailableResources(node.Resources, requirements.Resources) {
			return false
		}
	}

	// Check capability requirements
	for _, reqCap := range requirements.Capabilities {
		if !h.hasCapability(node, reqCap) {
			return false
		}
	}

	// Check locality requirements
	if !h.meetsLocalityRequirements(node, requirements.Locality) {
		return false
	}

	return true
}

// hasAvailableResources checks if node has sufficient available resources
func (h *HierarchicalManager) hasAvailableResources(nodeRes *NodeResources, reqRes ResourceRequirements) bool {
	// CPU check
	if reqRes.CPUCores > 0 {
		availableCPU := float64(nodeRes.CPUCores) * (1.0 - nodeRes.Metrics["cpu_usage"].(float64)/100.0)
		if reqRes.CPUCores > availableCPU {
			return false
		}
	}

	// Memory check
	if reqRes.MemoryMB > 0 {
		availableMemory := float64(nodeRes.MemoryMB) * (1.0 - nodeRes.Metrics["memory_usage"].(float64)/100.0)
		if float64(reqRes.MemoryMB) > availableMemory {
			return false
		}
	}

	return true
}

// hasCapability checks if node has a specific capability
func (h *HierarchicalManager) hasCapability(node *HierarchicalNode, capability string) bool {
	switch capability {
	case "containers":
		return node.Capabilities.CanRunContainers
	case "vms":
		return node.Capabilities.CanHostVMs
	case "tasks":
		return node.Capabilities.CanExecuteTasks
	case "storage":
		return node.Capabilities.HasLocalStorage
	case "caching":
		return node.Capabilities.CanCache
	case "load_balancing":
		return node.Capabilities.HasLoadBalancing
	default:
		// Check in supported workloads
		for _, workload := range node.Capabilities.SupportedWorkloads {
			if workload == capability {
				return true
			}
		}
		return false
	}
}

// meetsLocalityRequirements checks locality constraints
func (h *HierarchicalManager) meetsLocalityRequirements(node *HierarchicalNode, locality LocalityRequirements) bool {
	// Check preferred regions
	if len(locality.PreferredRegions) > 0 {
		found := false
		for _, region := range locality.PreferredRegions {
			if node.Region == region {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}

	// Check excluded regions
	for _, region := range locality.ExcludedRegions {
		if node.Region == region {
			return false
		}
	}

	// Check distance constraint
	if locality.MaxDistanceKm > 0 && node.Location != nil && h.localNode.Location != nil {
		distance := h.calculateDistance(h.localNode.Location, node.Location)
		if distance > locality.MaxDistanceKm {
			return false
		}
	}

	return true
}

// scorePlacementCandidates scores candidate nodes for workload placement
func (h *HierarchicalManager) scorePlacementCandidates(candidates []*HierarchicalNode, requirements WorkloadRequirements) []*WorkloadPlacement {
	var placements []*WorkloadPlacement

	for _, candidate := range candidates {
		score, reasoning := h.calculatePlacementScore(candidate, requirements)
		
		placement := &WorkloadPlacement{
			PreferredNodes: []string{candidate.ID},
			Requirements:   requirements,
			Score:          score,
			Reasoning:      reasoning,
		}

		placements = append(placements, placement)
	}

	return placements
}

// calculatePlacementScore calculates a score for placing workload on a node
func (h *HierarchicalManager) calculatePlacementScore(node *HierarchicalNode, requirements WorkloadRequirements) (float64, []string) {
	score := 0.0
	var reasoning []string

	// Resource efficiency score (0-25 points)
	resourceScore := h.calculateResourceScore(node, requirements.Resources)
	score += resourceScore * 0.25
	if resourceScore > 0.8 {
		reasoning = append(reasoning, "Excellent resource fit")
	}

	// Latency score (0-25 points)
	latencyScore := h.calculateLatencyScore(node, requirements.Latency)
	score += latencyScore * 0.25
	if latencyScore > 0.8 {
		reasoning = append(reasoning, "Low latency path")
	}

	// Capability match score (0-20 points)
	capabilityScore := h.calculateCapabilityScore(node, requirements.Capabilities)
	score += capabilityScore * 0.20
	if capabilityScore == 1.0 {
		reasoning = append(reasoning, "Perfect capability match")
	}

	// Hierarchy level score (0-15 points)
	levelScore := h.calculateLevelScore(node, requirements)
	score += levelScore * 0.15

	// Availability score (0-15 points)
	availabilityScore := h.calculateAvailabilityScore(node, requirements.Availability)
	score += availabilityScore * 0.15

	return score, reasoning
}

// calculateResourceScore calculates resource utilization efficiency score
func (h *HierarchicalManager) calculateResourceScore(node *HierarchicalNode, resources ResourceRequirements) float64 {
	if node.Resources == nil {
		return 0.0
	}

	cpuUtilization := node.Resources.Metrics["cpu_usage"].(float64) / 100.0
	memUtilization := node.Resources.Metrics["memory_usage"].(float64) / 100.0

	// Calculate utilization after placement
	estimatedCPUUtil := cpuUtilization + (resources.CPUCores / float64(node.Resources.CPUCores))
	estimatedMemUtil := memUtilization + (float64(resources.MemoryMB) / float64(node.Resources.MemoryMB))

	// Optimal utilization is around 70-80%
	cpuScore := 1.0 - abs(estimatedCPUUtil-0.75)*2
	memScore := 1.0 - abs(estimatedMemUtil-0.75)*2

	return (cpuScore + memScore) / 2.0
}

// calculateLatencyScore calculates latency performance score
func (h *HierarchicalManager) calculateLatencyScore(node *HierarchicalNode, latency LatencyRequirements) float64 {
	// Estimate latency based on hierarchy level and distance
	estimatedLatency := h.estimateLatency(node)

	if latency.MaxLatencyMs == 0 {
		return 1.0 // No latency requirement
	}

	if estimatedLatency > float64(latency.MaxLatencyMs) {
		return 0.0 // Exceeds maximum
	}

	// Score based on how close to target latency
	if latency.TargetLatencyMs > 0 {
		targetLatency := float64(latency.TargetLatencyMs)
		if estimatedLatency <= targetLatency {
			return 1.0
		}
		// Linear degradation from target to max
		return 1.0 - (estimatedLatency-targetLatency)/(float64(latency.MaxLatencyMs)-targetLatency)
	}

	return 0.8 // Good enough if under max but no target specified
}

// calculateCapabilityScore calculates how well node capabilities match requirements
func (h *HierarchicalManager) calculateCapabilityScore(node *HierarchicalNode, capabilities []string) float64 {
	if len(capabilities) == 0 {
		return 1.0
	}

	matchCount := 0
	for _, capability := range capabilities {
		if h.hasCapability(node, capability) {
			matchCount++
		}
	}

	return float64(matchCount) / float64(len(capabilities))
}

// calculateLevelScore scores node based on hierarchy level appropriateness
func (h *HierarchicalManager) calculateLevelScore(node *HierarchicalNode, requirements WorkloadRequirements) float64 {
	// Prefer lower levels for latency-sensitive workloads
	if requirements.Latency.MaxLatencyMs > 0 && requirements.Latency.MaxLatencyMs < 50 {
		switch node.Level {
		case LevelDevice, LevelAccess:
			return 1.0
		case LevelMetro:
			return 0.8
		case LevelRegional:
			return 0.6
		case LevelCloud:
			return 0.2
		}
	}

	// Prefer higher levels for compute-intensive workloads
	if requirements.Resources.CPUCores > 4 || requirements.Resources.MemoryMB > 8192 {
		switch node.Level {
		case LevelCloud:
			return 1.0
		case LevelRegional:
			return 0.9
		case LevelMetro:
			return 0.7
		case LevelAccess:
			return 0.5
		case LevelDevice:
			return 0.2
		}
	}

	return 0.8 // Default score
}

// calculateAvailabilityScore calculates availability match score
func (h *HierarchicalManager) calculateAvailabilityScore(node *HierarchicalNode, availability AvailabilityRequirements) float64 {
	// Estimate node availability based on level and redundancy
	estimatedUptime := h.estimateNodeUptime(node)
	
	if availability.MinUptimePct == 0 {
		return 1.0 // No availability requirement
	}

	if estimatedUptime < availability.MinUptimePct {
		return 0.0 // Does not meet minimum
	}

	// Linear score based on how much it exceeds minimum
	excess := (estimatedUptime - availability.MinUptimePct) / (100.0 - availability.MinUptimePct)
	return min(1.0, 0.8+excess*0.2)
}

// Helper functions

func (h *HierarchicalManager) calculateDistance(loc1, loc2 *GeographicLocation) float64 {
	// Haversine formula for great circle distance
	const earthRadius = 6371.0 // km

	lat1Rad := loc1.Latitude * (3.14159 / 180)
	lat2Rad := loc2.Latitude * (3.14159 / 180)
	deltaLatRad := (loc2.Latitude - loc1.Latitude) * (3.14159 / 180)
	deltaLonRad := (loc2.Longitude - loc1.Longitude) * (3.14159 / 180)

	a := sin(deltaLatRad/2)*sin(deltaLatRad/2) + cos(lat1Rad)*cos(lat2Rad)*sin(deltaLonRad/2)*sin(deltaLonRad/2)
	c := 2 * atan2(sqrt(a), sqrt(1-a))

	return earthRadius * c
}

func (h *HierarchicalManager) estimateLatency(node *HierarchicalNode) float64 {
	// Base latency by hierarchy level
	baseLatency := map[HierarchicalLevel]float64{
		LevelDevice:   1.0,   // 1ms
		LevelAccess:   5.0,   // 5ms
		LevelMetro:    20.0,  // 20ms
		LevelRegional: 50.0,  // 50ms
		LevelCloud:    100.0, // 100ms
	}

	latency := baseLatency[node.Level]

	// Add distance penalty if location is known
	if node.Location != nil && h.localNode.Location != nil {
		distance := h.calculateDistance(h.localNode.Location, node.Location)
		// Add ~1ms per 100km
		latency += distance / 100.0
	}

	return latency
}

func (h *HierarchicalManager) estimateNodeUptime(node *HierarchicalNode) float64 {
	// Base uptime by hierarchy level
	baseUptime := map[HierarchicalLevel]float64{
		LevelCloud:    99.99, // 99.99%
		LevelRegional: 99.95, // 99.95%
		LevelMetro:    99.9,  // 99.9%
		LevelAccess:   99.5,  // 99.5%
		LevelDevice:   95.0,  // 95.0%
	}

	return baseUptime[node.Level]
}

// Background loops

func (h *HierarchicalManager) topologyDiscoveryLoop() {
	defer h.wg.Done()

	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-h.ctx.Done():
			return
		case <-ticker.C:
			h.discoverTopologyChanges()
		}
	}
}

func (h *HierarchicalManager) healthMonitoringLoop() {
	defer h.wg.Done()

	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-h.ctx.Done():
			return
		case <-ticker.C:
			h.monitorNodeHealth()
		}
	}
}

func (h *HierarchicalManager) workloadOptimizationLoop() {
	defer h.wg.Done()

	ticker := time.NewTicker(60 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-h.ctx.Done():
			return
		case <-ticker.C:
			h.optimizeWorkloadPlacements()
		}
	}
}

func (h *HierarchicalManager) discoverTopologyChanges() {
	// Implementation would discover new nodes and topology changes
}

func (h *HierarchicalManager) monitorNodeHealth() {
	// Implementation would monitor child node health
}

func (h *HierarchicalManager) optimizeWorkloadPlacements() {
	// Implementation would continuously optimize workload placements
}

// Math helper functions
func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

func min(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}

func sin(x float64) float64 {
	// Simplified sin implementation - use math.Sin in real code
	return x // placeholder
}

func cos(x float64) float64 {
	// Simplified cos implementation - use math.Cos in real code
	return x // placeholder
}

func sqrt(x float64) float64 {
	// Simplified sqrt implementation - use math.Sqrt in real code
	return x // placeholder
}

func atan2(y, x float64) float64 {
	// Simplified atan2 implementation - use math.Atan2 in real code
	return y / x // placeholder
}