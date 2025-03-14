package network

import (
	"fmt"
	"math"
	"sort"
	"sync"
)

// LinkType represents the type of network link
type LinkType string

const (
	// LinkTypeSameMachine represents links within the same physical machine
	LinkTypeSameMachine LinkType = "same_machine"

	// LinkTypeSameRack represents links within the same rack
	LinkTypeSameRack LinkType = "same_rack"

	// LinkTypeSameDatacenter represents links within the same datacenter
	LinkTypeSameDatacenter LinkType = "same_datacenter"

	// LinkTypeInterDatacenter represents links between datacenters
	LinkTypeInterDatacenter LinkType = "inter_datacenter"

	// LinkTypeWAN represents links over wide area networks
	LinkTypeWAN LinkType = "wan"
)

// NetworkLink represents a network connection between two nodes
type NetworkLink struct {
	// SourceID is the ID of the source node
	SourceID string

	// DestinationID is the ID of the destination node
	DestinationID string

	// Type is the type of link
	Type LinkType

	// Bandwidth is the available bandwidth in Mbps
	Bandwidth float64

	// Latency is the average latency in milliseconds
	Latency float64

	// Loss is the packet loss percentage (0.0-1.0)
	Loss float64

	// Jitter is the latency variation in milliseconds
	Jitter float64

	// Cost is a relative cost value for using this link
	Cost float64

	// Utilization is the current utilization (0.0-1.0)
	Utilization float64

	// Distance is the physical distance between nodes in meters
	// 0 means unknown
	Distance float64

	// Attributes contains additional link attributes
	Attributes map[string]interface{}
}

// NetworkNode represents a node in the network topology
type NetworkNode struct {
	// ID is the unique identifier for the node
	ID string

	// Type is the type of node (e.g., "hypervisor", "switch", "router")
	Type string

	// Location describes the physical location hierarchy
	Location NetworkLocation

	// Attributes contains additional node attributes
	Attributes map[string]interface{}
}

// NetworkLocation describes the physical location of a network node
type NetworkLocation struct {
	// Datacenter is the identifier of the datacenter
	Datacenter string

	// Room is the identifier of the room within the datacenter
	Room string

	// Row is the row identifier
	Row string

	// Rack is the rack identifier
	Rack string

	// Position is the position within the rack
	Position int

	// Coordinates contains geographic coordinates if available
	Coordinates *GeoCoordinates

	// Region is the geographic region
	Region string

	// Zone is the availability zone
	Zone string
}

// GeoCoordinates represents geographic coordinates
type GeoCoordinates struct {
	// Latitude in decimal degrees
	Latitude float64

	// Longitude in decimal degrees
	Longitude float64

	// Altitude in meters above sea level (optional)
	Altitude float64
}

// LinkCostEstimator calculates costs for network transfers
type LinkCostEstimator struct {
	// LinkCostByType maps link types to their default costs
	LinkCostByType map[LinkType]float64

	// BandwidthFactor is the impact of bandwidth on cost (0.0-1.0)
	BandwidthFactor float64

	// LatencyFactor is the impact of latency on cost (0.0-1.0)
	LatencyFactor float64

	// LossFactor is the impact of packet loss on cost (0.0-1.0)
	LossFactor float64

	// UtilizationFactor is the impact of current utilization on cost (0.0-1.0)
	UtilizationFactor float64
}

// NewDefaultLinkCostEstimator creates a default link cost estimator
func NewDefaultLinkCostEstimator() *LinkCostEstimator {
	return &LinkCostEstimator{
		LinkCostByType: map[LinkType]float64{
			LinkTypeSameMachine:     0.1,
			LinkTypeSameRack:        0.3,
			LinkTypeSameDatacenter:  0.5,
			LinkTypeInterDatacenter: 0.8,
			LinkTypeWAN:             1.0,
		},
		BandwidthFactor:   0.3,
		LatencyFactor:     0.3,
		LossFactor:        0.2,
		UtilizationFactor: 0.2,
	}
}

// CalculateLinkCost calculates the cost of using a network link
func (e *LinkCostEstimator) CalculateLinkCost(link *NetworkLink) float64 {
	baseCost, exists := e.LinkCostByType[link.Type]
	if !exists {
		// Default to highest cost if type is unknown
		baseCost = e.LinkCostByType[LinkTypeWAN]
	}

	// Calculate bandwidth factor (higher bandwidth = lower cost)
	bandwidthCost := 0.0
	if link.Bandwidth > 0 {
		// Normalize bandwidth: 1 Gbps is optimal, higher is better
		normBandwidth := math.Min(link.Bandwidth/1000.0, 1.0)
		bandwidthCost = (1.0 - normBandwidth) * e.BandwidthFactor
	}

	// Calculate latency factor (higher latency = higher cost)
	latencyCost := 0.0
	if link.Latency > 0 {
		// Normalize latency: 0ms is optimal, 100ms is bad
		normLatency := math.Min(link.Latency/100.0, 1.0)
		latencyCost = normLatency * e.LatencyFactor
	}

	// Calculate loss factor (higher loss = higher cost)
	lossCost := link.Loss * e.LossFactor

	// Calculate utilization factor (higher utilization = higher cost)
	utilizationCost := link.Utilization * e.UtilizationFactor

	// Calculate total cost
	totalCost := baseCost + bandwidthCost + latencyCost + lossCost + utilizationCost

	// Ensure cost is in range [0.0, 1.0]
	return math.Min(math.Max(totalCost, 0.0), 1.0)
}

// NetworkTopology represents the network topology of a cluster
type NetworkTopology struct {
	// nodes maps node IDs to network nodes
	nodes      map[string]*NetworkNode
	nodesMutex sync.RWMutex

	// links maps source ID + destination ID to network links
	links      map[string]*NetworkLink
	linksMutex sync.RWMutex

	// linkCostEstimator calculates costs for network links
	linkCostEstimator *LinkCostEstimator
}

// NewNetworkTopology creates a new network topology
func NewNetworkTopology() *NetworkTopology {
	return &NetworkTopology{
		nodes:             make(map[string]*NetworkNode),
		links:             make(map[string]*NetworkLink),
		linkCostEstimator: NewDefaultLinkCostEstimator(),
	}
}

// AddNode adds a node to the topology
func (t *NetworkTopology) AddNode(node *NetworkNode) error {
	if node == nil || node.ID == "" {
		return fmt.Errorf("node cannot be nil and must have an ID")
	}

	t.nodesMutex.Lock()
	defer t.nodesMutex.Unlock()

	t.nodes[node.ID] = node
	return nil
}

// GetNode gets a node from the topology
func (t *NetworkTopology) GetNode(nodeID string) (*NetworkNode, error) {
	t.nodesMutex.RLock()
	defer t.nodesMutex.RUnlock()

	node, exists := t.nodes[nodeID]
	if !exists {
		return nil, fmt.Errorf("node not found: %s", nodeID)
	}

	return node, nil
}

// AddLink adds a link to the topology
func (t *NetworkTopology) AddLink(link *NetworkLink) error {
	if link == nil || link.SourceID == "" || link.DestinationID == "" {
		return fmt.Errorf("link cannot be nil and must have source and destination IDs")
	}

	t.linksMutex.Lock()
	defer t.linksMutex.Unlock()

	// Create a unique key for the link
	key := fmt.Sprintf("%s:%s", link.SourceID, link.DestinationID)
	t.links[key] = link

	// Calculate and update the link cost if not set
	if link.Cost == 0 {
		link.Cost = t.linkCostEstimator.CalculateLinkCost(link)
	}

	return nil
}

// GetLink gets a link between two nodes
func (t *NetworkTopology) GetLink(sourceID, destID string) (*NetworkLink, error) {
	t.linksMutex.RLock()
	defer t.linksMutex.RUnlock()

	key := fmt.Sprintf("%s:%s", sourceID, destID)
	link, exists := t.links[key]
	if !exists {
		return nil, fmt.Errorf("link not found: %s -> %s", sourceID, destID)
	}

	return link, nil
}

// GetNetworkCost calculates the network cost between two nodes
func (t *NetworkTopology) GetNetworkCost(sourceID, destID string) (float64, error) {
	// If same node, cost is minimal
	if sourceID == destID {
		return 0.0, nil
	}

	// Try to get a direct link
	t.linksMutex.RLock()
	key := fmt.Sprintf("%s:%s", sourceID, destID)
	link, exists := t.links[key]
	t.linksMutex.RUnlock()

	if exists {
		return link.Cost, nil
	}

	// If no direct link, determine link type based on location hierarchy
	t.nodesMutex.RLock()
	sourceNode, sourceExists := t.nodes[sourceID]
	destNode, destExists := t.nodes[destID]
	t.nodesMutex.RUnlock()

	if !sourceExists || !destExists {
		return 0.0, fmt.Errorf("source or destination node not found")
	}

	// Determine link type based on location
	linkType := determineLinkType(sourceNode.Location, destNode.Location)

	// Create a synthetic link for cost calculation
	syntheticLink := &NetworkLink{
		SourceID:      sourceID,
		DestinationID: destID,
		Type:          linkType,
		// Use default values for other properties
	}

	// Calculate the cost
	return t.linkCostEstimator.CalculateLinkCost(syntheticLink), nil
}

// determineLinkType determines the type of link between two locations
func determineLinkType(loc1, loc2 NetworkLocation) LinkType {
	// Different regions mean WAN
	if loc1.Region != loc2.Region {
		return LinkTypeWAN
	}

	// Different datacenters mean inter-datacenter
	if loc1.Datacenter != loc2.Datacenter {
		return LinkTypeInterDatacenter
	}

	// Same datacenter but different racks
	if loc1.Rack != loc2.Rack {
		return LinkTypeSameDatacenter
	}

	// Same rack
	return LinkTypeSameRack
}

// GetAllNodes gets all nodes in the topology
func (t *NetworkTopology) GetAllNodes() []*NetworkNode {
	t.nodesMutex.RLock()
	defer t.nodesMutex.RUnlock()

	nodes := make([]*NetworkNode, 0, len(t.nodes))
	for _, node := range t.nodes {
		nodes = append(nodes, node)
	}

	return nodes
}

// GetAllLinks gets all links in the topology
func (t *NetworkTopology) GetAllLinks() []*NetworkLink {
	t.linksMutex.RLock()
	defer t.linksMutex.RUnlock()

	links := make([]*NetworkLink, 0, len(t.links))
	for _, link := range t.links {
		links = append(links, link)
	}

	return links
}

// UpdateLinkUtilization updates the utilization of a link
func (t *NetworkTopology) UpdateLinkUtilization(sourceID, destID string, utilization float64) error {
	t.linksMutex.Lock()
	defer t.linksMutex.Unlock()

	key := fmt.Sprintf("%s:%s", sourceID, destID)
	link, exists := t.links[key]
	if !exists {
		return fmt.Errorf("link not found: %s -> %s", sourceID, destID)
	}

	link.Utilization = math.Max(0.0, math.Min(1.0, utilization))

	// Recalculate the link cost
	link.Cost = t.linkCostEstimator.CalculateLinkCost(link)

	return nil
}

// GetNodesInSameZone gets all nodes in the same zone as the given node
func (t *NetworkTopology) GetNodesInSameZone(nodeID string) ([]*NetworkNode, error) {
	t.nodesMutex.RLock()
	node, exists := t.nodes[nodeID]
	if !exists {
		t.nodesMutex.RUnlock()
		return nil, fmt.Errorf("node not found: %s", nodeID)
	}

	zone := node.Location.Zone
	datacenter := node.Location.Datacenter

	// Find all nodes in the same zone
	nodes := make([]*NetworkNode, 0)
	for _, otherNode := range t.nodes {
		if otherNode.Location.Zone == zone && otherNode.Location.Datacenter == datacenter {
			nodes = append(nodes, otherNode)
		}
	}
	t.nodesMutex.RUnlock()

	return nodes, nil
}

// GetNodesInSameRack gets all nodes in the same rack as the given node
func (t *NetworkTopology) GetNodesInSameRack(nodeID string) ([]*NetworkNode, error) {
	t.nodesMutex.RLock()
	node, exists := t.nodes[nodeID]
	if !exists {
		t.nodesMutex.RUnlock()
		return nil, fmt.Errorf("node not found: %s", nodeID)
	}

	rack := node.Location.Rack
	datacenter := node.Location.Datacenter

	// Find all nodes in the same rack
	nodes := make([]*NetworkNode, 0)
	for _, otherNode := range t.nodes {
		if otherNode.Location.Rack == rack && otherNode.Location.Datacenter == datacenter {
			nodes = append(nodes, otherNode)
		}
	}
	t.nodesMutex.RUnlock()

	return nodes, nil
}

// CalculateNetworkDistance calculates the network distance between two nodes
// Returns a normalized value between 0.0 and 1.0
func (t *NetworkTopology) CalculateNetworkDistance(sourceID, destID string) (float64, error) {
	// Same node has zero distance
	if sourceID == destID {
		return 0.0, nil
	}

	// Get cost which serves as our distance metric
	cost, err := t.GetNetworkCost(sourceID, destID)
	if err != nil {
		return 1.0, err
	}

	return cost, nil
}

// FindClosestNodes finds the n closest nodes to the given node
func (t *NetworkTopology) FindClosestNodes(nodeID string, n int) ([]*NetworkNode, error) {
	t.nodesMutex.RLock()
	if _, exists := t.nodes[nodeID]; !exists {
		t.nodesMutex.RUnlock()
		return nil, fmt.Errorf("node not found: %s", nodeID)
	}

	// Get all node IDs except the given node
	nodeIDs := make([]string, 0, len(t.nodes)-1)
	for id := range t.nodes {
		if id != nodeID {
			nodeIDs = append(nodeIDs, id)
		}
	}
	t.nodesMutex.RUnlock()

	// Calculate distance to each node
	type nodeDistance struct {
		nodeID   string
		distance float64
	}

	distances := make([]nodeDistance, 0, len(nodeIDs))
	for _, id := range nodeIDs {
		dist, err := t.CalculateNetworkDistance(nodeID, id)
		if err != nil {
			continue
		}
		distances = append(distances, nodeDistance{nodeID: id, distance: dist})
	}

	// Sort by distance (ascending)
	sort.Slice(distances, func(i, j int) bool {
		return distances[i].distance < distances[j].distance
	})

	// Take top n
	count := n
	if count > len(distances) {
		count = len(distances)
	}

	// Get the node objects
	result := make([]*NetworkNode, 0, count)
	t.nodesMutex.RLock()
	defer t.nodesMutex.RUnlock()

	for i := 0; i < count; i++ {
		if node, exists := t.nodes[distances[i].nodeID]; exists {
			result = append(result, node)
		}
	}

	return result, nil
}
