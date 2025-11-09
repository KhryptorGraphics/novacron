package mesh

import (
	"context"
	"fmt"
	"sync"
	"time"

)

// BundleProtocol implements RFC 5050 Bundle Protocol for Delay-Tolerant Networking
type BundleProtocol struct {
	Version         int       `json:"version"`
	PrimaryBlock    []byte    `json:"primary_block"`
	PayloadBlock    []byte    `json:"payload_block"`
	CreationTime    time.Time `json:"creation_time"`
	Lifetime        time.Duration `json:"lifetime"`
	Priority        int       `json:"priority"`
	SourceEID       string    `json:"source_eid"`     // Endpoint Identifier
	DestinationEID  string    `json:"destination_eid"`
	CustodyTransfer bool      `json:"custody_transfer"`
	FragmentOffset  uint64    `json:"fragment_offset"`
	TotalADULength  uint64    `json:"total_adu_length"`
}

// MeshNode represents a node in the global mesh network
type MeshNode struct {
	NodeID           string                 `json:"node_id"`
	Location         GeoLocation            `json:"location"`
	Neighbors        []string               `json:"neighbors"`
	ConnectionType   string                 `json:"connection_type"` // satellite, cable, terrestrial
	Bandwidth        float64                `json:"bandwidth"`       // Mbps
	Latency          time.Duration          `json:"latency"`
	Reliability      float64                `json:"reliability"`     // 0-1
	LastSeen         time.Time              `json:"last_seen"`
	Status           string                 `json:"status"`          // active, degraded, offline
	Bundles          map[string]*BundleProtocol `json:"bundles"`
	RoutingTable     map[string][]string    `json:"routing_table"`
}

// GeoLocation represents a geographic location
type GeoLocation struct {
	Latitude   float64 `json:"latitude"`
	Longitude  float64 `json:"longitude"`
	Altitude   float64 `json:"altitude"` // meters
	Region     string  `json:"region"`
}

// GlobalMesh manages the global mesh network
type GlobalMesh struct {
	config          *planetary.PlanetaryConfig
	nodes           map[string]*MeshNode
	routes          map[string][]string  // destination -> path
	bundles         map[string]*BundleProtocol
	dtnEnabled      bool
	storeForward    map[string][]*BundleProtocol // node -> pending bundles
	opportunistic   bool
	mu              sync.RWMutex
	ctx             context.Context
	cancel          context.CancelFunc
	convergenceTime time.Duration
}

// NewGlobalMesh creates a new global mesh network
func NewGlobalMesh(config *planetary.PlanetaryConfig) *GlobalMesh {
	ctx, cancel := context.WithCancel(context.Background())

	return &GlobalMesh{
		config:          config,
		nodes:           make(map[string]*MeshNode),
		routes:          make(map[string][]string),
		bundles:         make(map[string]*BundleProtocol),
		dtnEnabled:      config.EnableDTN,
		storeForward:    make(map[string][]*BundleProtocol),
		opportunistic:   config.OpportunisticRouting,
		ctx:             ctx,
		cancel:          cancel,
		convergenceTime: config.MeshConvergenceTime,
	}
}

// Start starts the global mesh network
func (gm *GlobalMesh) Start() error {
	// Start mesh convergence
	go gm.convergeMesh()

	// Start DTN bundle processing
	if gm.dtnEnabled {
		go gm.processBundles()
	}

	// Start opportunistic routing
	if gm.opportunistic {
		go gm.opportunisticRouting()
	}

	// Start store-and-forward
	if gm.config.StoreAndForward {
		go gm.storeAndForwardProcessing()
	}

	// Start mesh health monitoring
	go gm.monitorMeshHealth()

	return nil
}

// Stop stops the global mesh network
func (gm *GlobalMesh) Stop() error {
	gm.cancel()
	return nil
}

// AddNode adds a node to the mesh network
func (gm *GlobalMesh) AddNode(node *MeshNode) error {
	gm.mu.Lock()
	defer gm.mu.Unlock()

	if node.NodeID == "" {
		return fmt.Errorf("node ID cannot be empty")
	}

	node.LastSeen = time.Now()
	node.Status = "active"
	node.Bundles = make(map[string]*BundleProtocol)
	node.RoutingTable = make(map[string][]string)

	gm.nodes[node.NodeID] = node

	// Trigger mesh reconvergence
	go gm.reconvergeMesh()

	return nil
}

// RemoveNode removes a node from the mesh network
func (gm *GlobalMesh) RemoveNode(nodeID string) error {
	gm.mu.Lock()
	defer gm.mu.Unlock()

	delete(gm.nodes, nodeID)

	// Trigger mesh reconvergence
	go gm.reconvergeMesh()

	return nil
}

// SendBundle sends a bundle through the DTN
func (gm *GlobalMesh) SendBundle(bundle *BundleProtocol) error {
	if !gm.dtnEnabled {
		return planetary.ErrDTNNotEnabled
	}

	gm.mu.Lock()
	bundleID := fmt.Sprintf("%s-%d", bundle.SourceEID, time.Now().UnixNano())
	gm.bundles[bundleID] = bundle
	gm.mu.Unlock()

	// Find route to destination
	path, err := gm.findRoute(bundle.DestinationEID)
	if err != nil {
		// Store for later delivery
		if gm.config.StoreAndForward {
			gm.storeBundle(bundle)
			return nil
		}
		return err
	}

	// Forward bundle along path
	return gm.forwardBundle(bundle, path)
}

// findRoute finds a route to a destination using mesh routing
func (gm *GlobalMesh) findRoute(destination string) ([]string, error) {
	gm.mu.RLock()
	defer gm.mu.RUnlock()

	if path, exists := gm.routes[destination]; exists {
		return path, nil
	}

	// Use Dijkstra's algorithm to find shortest path
	path := gm.dijkstraPath(destination)
	if len(path) == 0 {
		return nil, planetary.ErrNoRouteToDest
	}

	return path, nil
}

// dijkstraPath implements Dijkstra's shortest path algorithm
func (gm *GlobalMesh) dijkstraPath(destination string) []string {
	// Simplified implementation
	// In production, use a proper graph algorithm library

	visited := make(map[string]bool)
	distances := make(map[string]float64)
	previous := make(map[string]string)

	// Initialize distances
	for nodeID := range gm.nodes {
		distances[nodeID] = float64(1<<63 - 1) // Max float64
	}

	// Find a source node (any active node)
	var source string
	for nodeID, node := range gm.nodes {
		if node.Status == "active" {
			source = nodeID
			break
		}
	}

	if source == "" {
		return nil
	}

	distances[source] = 0

	for len(visited) < len(gm.nodes) {
		// Find minimum distance unvisited node
		var current string
		minDist := float64(1<<63 - 1)

		for nodeID := range gm.nodes {
			if !visited[nodeID] && distances[nodeID] < minDist {
				current = nodeID
				minDist = distances[nodeID]
			}
		}

		if current == "" {
			break
		}

		visited[current] = true

		if current == destination {
			break
		}

		// Update distances to neighbors
		node := gm.nodes[current]
		for _, neighbor := range node.Neighbors {
			if visited[neighbor] {
				continue
			}

			// Calculate edge weight (latency + 1/bandwidth)
			neighborNode := gm.nodes[neighbor]
			weight := float64(neighborNode.Latency.Milliseconds()) + (1000.0 / neighborNode.Bandwidth)

			alt := distances[current] + weight
			if alt < distances[neighbor] {
				distances[neighbor] = alt
				previous[neighbor] = current
			}
		}
	}

	// Reconstruct path
	path := []string{}
	current := destination

	for current != source {
		path = append([]string{current}, path...)
		prev, exists := previous[current]
		if !exists {
			return nil
		}
		current = prev
	}

	path = append([]string{source}, path...)

	return path
}

// forwardBundle forwards a bundle along a path
func (gm *GlobalMesh) forwardBundle(bundle *BundleProtocol, path []string) error {
	// Simulate bundle forwarding
	for i := 0; i < len(path)-1; i++ {
		currentNode := path[i]
		nextNode := path[i+1]

		node := gm.nodes[currentNode]
		if node == nil {
			return fmt.Errorf("node %s not found", currentNode)
		}

		// Simulate transmission delay
		time.Sleep(node.Latency)

		// Check if next hop is available
		nextNodeObj := gm.nodes[nextNode]
		if nextNodeObj == nil || nextNodeObj.Status != "active" {
			// Store and forward
			if gm.config.StoreAndForward {
				gm.storeBundle(bundle)
				return nil
			}
			return fmt.Errorf("next hop %s unavailable", nextNode)
		}
	}

	return nil
}

// storeBundle stores a bundle for later delivery
func (gm *GlobalMesh) storeBundle(bundle *BundleProtocol) {
	gm.mu.Lock()
	defer gm.mu.Unlock()

	// Find best node to store bundle
	var bestNode string
	var bestReliability float64 = 0

	for nodeID, node := range gm.nodes {
		if node.Status == "active" && node.Reliability > bestReliability {
			bestNode = nodeID
			bestReliability = node.Reliability
		}
	}

	if bestNode != "" {
		gm.storeForward[bestNode] = append(gm.storeForward[bestNode], bundle)
	}
}

// convergeMesh performs mesh convergence
func (gm *GlobalMesh) convergeMesh() {
	ticker := time.NewTicker(gm.convergenceTime)
	defer ticker.Stop()

	for {
		select {
		case <-gm.ctx.Done():
			return
		case <-ticker.C:
			gm.reconvergeMesh()
		}
	}
}

// reconvergeMesh reconverges the mesh network
func (gm *GlobalMesh) reconvergeMesh() {
	gm.mu.Lock()
	defer gm.mu.Unlock()

	// Rebuild routing tables for all nodes
	for _, node := range gm.nodes {
		node.RoutingTable = make(map[string][]string)

		for destID := range gm.nodes {
			if destID == node.NodeID {
				continue
			}

			// Calculate path to destination
			path := gm.dijkstraPath(destID)
			if len(path) > 0 {
				node.RoutingTable[destID] = path
			}
		}
	}

	// Update global routes
	gm.routes = make(map[string][]string)
	for destID := range gm.nodes {
		path := gm.dijkstraPath(destID)
		if len(path) > 0 {
			gm.routes[destID] = path
		}
	}
}

// processBundles processes DTN bundles
func (gm *GlobalMesh) processBundles() {
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-gm.ctx.Done():
			return
		case <-ticker.C:
			gm.processStoredBundles()
		}
	}
}

// processStoredBundles processes stored bundles
func (gm *GlobalMesh) processStoredBundles() {
	gm.mu.Lock()
	defer gm.mu.Unlock()

	for nodeID, bundles := range gm.storeForward {
		if len(bundles) == 0 {
			continue
		}

		// Try to deliver stored bundles
		delivered := make([]int, 0)

		for i, bundle := range bundles {
			// Check if bundle expired
			if time.Since(bundle.CreationTime) > bundle.Lifetime {
				delivered = append(delivered, i)
				continue
			}

			// Try to find route
			path, err := gm.findRoute(bundle.DestinationEID)
			if err == nil {
				// Forward bundle
				go gm.forwardBundle(bundle, path)
				delivered = append(delivered, i)
			}
		}

		// Remove delivered bundles
		newBundles := make([]*BundleProtocol, 0)
		for i, bundle := range bundles {
			isDelivered := false
			for _, idx := range delivered {
				if i == idx {
					isDelivered = true
					break
				}
			}
			if !isDelivered {
				newBundles = append(newBundles, bundle)
			}
		}

		gm.storeForward[nodeID] = newBundles
	}
}

// opportunisticRouting performs opportunistic routing
func (gm *GlobalMesh) opportunisticRouting() {
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-gm.ctx.Done():
			return
		case <-ticker.C:
			gm.findOpportunisticRoutes()
		}
	}
}

// findOpportunisticRoutes finds opportunistic routing paths
func (gm *GlobalMesh) findOpportunisticRoutes() {
	gm.mu.RLock()
	defer gm.mu.RUnlock()

	// Look for alternative paths with better metrics
	for destID := range gm.routes {
		currentPath := gm.routes[destID]
		if len(currentPath) == 0 {
			continue
		}

		// Calculate current path metric
		currentMetric := gm.calculatePathMetric(currentPath)

		// Try to find better path
		betterPath := gm.findBetterPath(destID, currentMetric)
		if len(betterPath) > 0 {
			gm.routes[destID] = betterPath
		}
	}
}

// calculatePathMetric calculates the metric for a path
func (gm *GlobalMesh) calculatePathMetric(path []string) float64 {
	metric := 0.0

	for i := 0; i < len(path)-1; i++ {
		node := gm.nodes[path[i+1]]
		if node == nil {
			return float64(1<<63 - 1)
		}

		metric += float64(node.Latency.Milliseconds()) + (1000.0 / node.Bandwidth)
	}

	return metric
}

// findBetterPath finds a better path to destination
func (gm *GlobalMesh) findBetterPath(destination string, currentMetric float64) []string {
	// Use alternative routing algorithm (e.g., A*)
	// For simplicity, using Dijkstra again
	path := gm.dijkstraPath(destination)

	if len(path) > 0 {
		newMetric := gm.calculatePathMetric(path)
		if newMetric < currentMetric {
			return path
		}
	}

	return nil
}

// storeAndForwardProcessing processes store-and-forward operations
func (gm *GlobalMesh) storeAndForwardProcessing() {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-gm.ctx.Done():
			return
		case <-ticker.C:
			gm.cleanupExpiredBundles()
		}
	}
}

// cleanupExpiredBundles cleans up expired bundles
func (gm *GlobalMesh) cleanupExpiredBundles() {
	gm.mu.Lock()
	defer gm.mu.Unlock()

	for nodeID, bundles := range gm.storeForward {
		activeBundles := make([]*BundleProtocol, 0)

		for _, bundle := range bundles {
			if time.Since(bundle.CreationTime) <= bundle.Lifetime {
				activeBundles = append(activeBundles, bundle)
			}
		}

		gm.storeForward[nodeID] = activeBundles
	}
}

// monitorMeshHealth monitors mesh network health
func (gm *GlobalMesh) monitorMeshHealth() {
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-gm.ctx.Done():
			return
		case <-ticker.C:
			gm.checkNodeHealth()
		}
	}
}

// checkNodeHealth checks health of all nodes
func (gm *GlobalMesh) checkNodeHealth() {
	gm.mu.Lock()
	defer gm.mu.Unlock()

	now := time.Now()

	for nodeID, node := range gm.nodes {
		// Mark nodes as offline if not seen in 30 seconds
		if now.Sub(node.LastSeen) > 30*time.Second {
			node.Status = "offline"
		} else if now.Sub(node.LastSeen) > 10*time.Second {
			node.Status = "degraded"
		} else {
			node.Status = "active"
		}

		gm.nodes[nodeID] = node
	}
}

// GetMeshMetrics returns mesh network metrics
func (gm *GlobalMesh) GetMeshMetrics() map[string]interface{} {
	gm.mu.RLock()
	defer gm.mu.RUnlock()

	totalNodes := len(gm.nodes)
	activeNodes := 0
	totalBundles := len(gm.bundles)
	storedBundles := 0
	avgLatency := 0.0
	avgBandwidth := 0.0

	for _, node := range gm.nodes {
		if node.Status == "active" {
			activeNodes++
		}
		avgLatency += float64(node.Latency.Milliseconds())
		avgBandwidth += node.Bandwidth
	}

	for _, bundles := range gm.storeForward {
		storedBundles += len(bundles)
	}

	if totalNodes > 0 {
		avgLatency /= float64(totalNodes)
		avgBandwidth /= float64(totalNodes)
	}

	return map[string]interface{}{
		"total_nodes":     totalNodes,
		"active_nodes":    activeNodes,
		"total_bundles":   totalBundles,
		"stored_bundles":  storedBundles,
		"avg_latency_ms":  avgLatency,
		"avg_bandwidth":   avgBandwidth,
		"total_routes":    len(gm.routes),
		"dtn_enabled":     gm.dtnEnabled,
		"opportunistic":   gm.opportunistic,
	}
}
