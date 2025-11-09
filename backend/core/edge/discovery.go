package edge

import (
	"context"
	"fmt"
	"net"
	"sync"
	"time"
)

// EdgeDiscovery handles discovery of edge nodes
type EdgeDiscovery struct {
	config     *EdgeConfig
	nodes      map[string]*EdgeNode
	mu         sync.RWMutex
	stopCh     chan struct{}
	wg         sync.WaitGroup

	// Discovery callbacks
	onNodeDiscovered func(*EdgeNode)
	onNodeLost       func(string)

	// Discovery plugins
	mecDiscovery     MECDiscovery
	cdnDiscovery     CDNDiscovery
	iotDiscovery     IoTDiscovery
	telcoDiscovery   TelcoDiscovery
}

// MECDiscovery interface for 5G MEC discovery
type MECDiscovery interface {
	DiscoverMECNodes(ctx context.Context) ([]*EdgeNode, error)
}

// CDNDiscovery interface for CDN edge discovery
type CDNDiscovery interface {
	DiscoverCDNNodes(ctx context.Context) ([]*EdgeNode, error)
}

// IoTDiscovery interface for IoT gateway discovery
type IoTDiscovery interface {
	DiscoverIoTGateways(ctx context.Context) ([]*EdgeNode, error)
}

// TelcoDiscovery interface for telco edge discovery
type TelcoDiscovery interface {
	DiscoverTelcoNodes(ctx context.Context) ([]*EdgeNode, error)
}

// NewEdgeDiscovery creates a new edge discovery instance
func NewEdgeDiscovery(config *EdgeConfig) *EdgeDiscovery {
	return &EdgeDiscovery{
		config: config,
		nodes:  make(map[string]*EdgeNode),
		stopCh: make(chan struct{}),
	}
}

// Start starts the edge discovery service
func (ed *EdgeDiscovery) Start(ctx context.Context) error {
	ed.wg.Add(1)
	go ed.discoveryLoop(ctx)

	return nil
}

// Stop stops the edge discovery service
func (ed *EdgeDiscovery) Stop() error {
	close(ed.stopCh)
	ed.wg.Wait()
	return nil
}

// discoveryLoop runs periodic discovery
func (ed *EdgeDiscovery) discoveryLoop(ctx context.Context) {
	defer ed.wg.Done()

	ticker := time.NewTicker(ed.config.DiscoveryInterval)
	defer ticker.Stop()

	// Initial discovery
	ed.runDiscovery(ctx)

	for {
		select {
		case <-ctx.Done():
			return
		case <-ed.stopCh:
			return
		case <-ticker.C:
			ed.runDiscovery(ctx)
		}
	}
}

// runDiscovery runs discovery across all enabled edge types
func (ed *EdgeDiscovery) runDiscovery(ctx context.Context) {
	discoveryCtx, cancel := context.WithTimeout(ctx, ed.config.DiscoveryTimeout)
	defer cancel()

	var wg sync.WaitGroup
	nodesChan := make(chan *EdgeNode, 100)

	// Discover 5G MEC nodes
	if ed.config.EnableMEC && ed.mecDiscovery != nil {
		wg.Add(1)
		go func() {
			defer wg.Done()
			nodes, err := ed.mecDiscovery.DiscoverMECNodes(discoveryCtx)
			if err == nil {
				for _, node := range nodes {
					nodesChan <- node
				}
			}
		}()
	}

	// Discover CDN edge nodes
	if ed.config.EnableCDNEdge && ed.cdnDiscovery != nil {
		wg.Add(1)
		go func() {
			defer wg.Done()
			nodes, err := ed.cdnDiscovery.DiscoverCDNNodes(discoveryCtx)
			if err == nil {
				for _, node := range nodes {
					nodesChan <- node
				}
			}
		}()
	}

	// Discover IoT gateways
	if ed.config.EnableIoTGateway && ed.iotDiscovery != nil {
		wg.Add(1)
		go func() {
			defer wg.Done()
			nodes, err := ed.iotDiscovery.DiscoverIoTGateways(discoveryCtx)
			if err == nil {
				for _, node := range nodes {
					nodesChan <- node
				}
			}
		}()
	}

	// Discover telco edge nodes
	if ed.config.EnableTelcoEdge && ed.telcoDiscovery != nil {
		wg.Add(1)
		go func() {
			defer wg.Done()
			nodes, err := ed.telcoDiscovery.DiscoverTelcoNodes(discoveryCtx)
			if err == nil {
				for _, node := range nodes {
					nodesChan <- node
				}
			}
		}()
	}

	// Wait for all discoveries to complete
	go func() {
		wg.Wait()
		close(nodesChan)
	}()

	// Process discovered nodes
	discoveredIDs := make(map[string]bool)
	for node := range nodesChan {
		ed.addOrUpdateNode(node)
		discoveredIDs[node.ID] = true
	}

	// Mark nodes as lost if not discovered
	ed.markLostNodes(discoveredIDs)
}

// addOrUpdateNode adds or updates an edge node
func (ed *EdgeDiscovery) addOrUpdateNode(node *EdgeNode) {
	ed.mu.Lock()
	defer ed.mu.Unlock()

	now := time.Now()
	node.UpdatedAt = now
	node.LastSeenAt = now

	existing, exists := ed.nodes[node.ID]
	if !exists {
		node.CreatedAt = now
		ed.nodes[node.ID] = node

		if ed.onNodeDiscovered != nil {
			ed.onNodeDiscovered(node)
		}
	} else {
		// Update existing node
		existing.Location = node.Location
		existing.Resources = node.Resources
		existing.Status = node.Status
		existing.Latency = node.Latency
		existing.Cost = node.Cost
		existing.Network = node.Network
		existing.UpdatedAt = now
		existing.LastSeenAt = now
	}
}

// markLostNodes marks nodes as lost if not discovered
func (ed *EdgeDiscovery) markLostNodes(discoveredIDs map[string]bool) {
	ed.mu.Lock()
	defer ed.mu.Unlock()

	timeout := ed.config.DiscoveryInterval * 3
	cutoff := time.Now().Add(-timeout)

	for id, node := range ed.nodes {
		if !discoveredIDs[id] && node.LastSeenAt.Before(cutoff) {
			node.Status.State = EdgeNodeStateOffline
			node.Status.Health = HealthStatusUnhealthy

			if ed.onNodeLost != nil {
				ed.onNodeLost(id)
			}
		}
	}
}

// GetNode retrieves an edge node by ID
func (ed *EdgeDiscovery) GetNode(id string) (*EdgeNode, error) {
	ed.mu.RLock()
	defer ed.mu.RUnlock()

	node, exists := ed.nodes[id]
	if !exists {
		return nil, ErrEdgeNodeNotFound
	}

	return node, nil
}

// GetAllNodes returns all discovered edge nodes
func (ed *EdgeDiscovery) GetAllNodes() []*EdgeNode {
	ed.mu.RLock()
	defer ed.mu.RUnlock()

	nodes := make([]*EdgeNode, 0, len(ed.nodes))
	for _, node := range ed.nodes {
		nodes = append(nodes, node)
	}

	return nodes
}

// GetHealthyNodes returns all healthy edge nodes
func (ed *EdgeDiscovery) GetHealthyNodes() []*EdgeNode {
	ed.mu.RLock()
	defer ed.mu.RUnlock()

	nodes := make([]*EdgeNode, 0)
	for _, node := range ed.nodes {
		if node.Status.State == EdgeNodeStateOnline &&
		   node.Status.Health == HealthStatusHealthy {
			nodes = append(nodes, node)
		}
	}

	return nodes
}

// GetNodesByType returns nodes of a specific type
func (ed *EdgeDiscovery) GetNodesByType(edgeType EdgeType) []*EdgeNode {
	ed.mu.RLock()
	defer ed.mu.RUnlock()

	nodes := make([]*EdgeNode, 0)
	for _, node := range ed.nodes {
		if node.Type == edgeType {
			nodes = append(nodes, node)
		}
	}

	return nodes
}

// MeasureLatency measures latency to an edge node
func (ed *EdgeDiscovery) MeasureLatency(ctx context.Context, node *EdgeNode) (*LatencyMetrics, error) {
	if node.Network.PublicIP == "" {
		return nil, fmt.Errorf("node has no public IP")
	}

	var metrics LatencyMetrics
	samples := 5
	rtts := make([]time.Duration, 0, samples)

	// Measure RTT multiple times
	for i := 0; i < samples; i++ {
		start := time.Now()

		conn, err := net.DialTimeout("tcp", fmt.Sprintf("%s:443", node.Network.PublicIP), 2*time.Second)
		if err != nil {
			continue
		}
		conn.Close()

		rtt := time.Since(start)
		rtts = append(rtts, rtt)
	}

	if len(rtts) == 0 {
		return nil, fmt.Errorf("failed to measure latency")
	}

	// Calculate statistics
	var sum, min, max time.Duration
	min = rtts[0]
	max = rtts[0]

	for _, rtt := range rtts {
		sum += rtt
		if rtt < min {
			min = rtt
		}
		if rtt > max {
			max = rtt
		}
	}

	metrics.RTTAvg = sum / time.Duration(len(rtts))
	metrics.RTTMin = min
	metrics.RTTMax = max
	metrics.MeasuredAt = time.Now()

	// Calculate jitter
	if len(rtts) > 1 {
		var jitterSum float64
		for i := 1; i < len(rtts); i++ {
			diff := float64(rtts[i]-rtts[i-1]) / float64(time.Millisecond)
			if diff < 0 {
				diff = -diff
			}
			jitterSum += diff
		}
		metrics.JitterMs = jitterSum / float64(len(rtts)-1)
	}

	// Packet loss (simplified)
	metrics.PacketLoss = float64(samples-len(rtts)) / float64(samples) * 100.0

	return &metrics, nil
}

// AssessCapabilities assesses the capabilities of an edge node
func (ed *EdgeDiscovery) AssessCapabilities(node *EdgeNode) error {
	// Check minimum resource requirements
	if node.Resources.TotalCPUCores < ed.config.MinEdgeResources.MinCPUCores {
		return ErrInsufficientEdgeResources
	}

	if node.Resources.TotalMemoryMB < ed.config.MinEdgeResources.MinMemoryMB {
		return ErrInsufficientEdgeResources
	}

	if node.Resources.TotalStorageGB < ed.config.MinEdgeResources.MinStorageGB {
		return ErrInsufficientEdgeResources
	}

	if node.Resources.TotalBandwidthMbps < ed.config.MinEdgeResources.MinBandwidthMbps {
		return ErrInsufficientEdgeResources
	}

	// Update utilization percentage
	availableCPU := node.Resources.TotalCPUCores - node.Resources.UsedCPUCores
	availableMemory := node.Resources.TotalMemoryMB - node.Resources.UsedMemoryMB

	cpuUtil := float64(node.Resources.UsedCPUCores) / float64(node.Resources.TotalCPUCores)
	memUtil := float64(node.Resources.UsedMemoryMB) / float64(node.Resources.TotalMemoryMB)

	node.Resources.UtilizationPercent = (cpuUtil + memUtil) / 2.0 * 100.0

	// Check if node can accept more VMs
	if node.Status.ActiveVMs >= ed.config.MaxEdgeVMsPerNode {
		return fmt.Errorf("max VMs reached on node")
	}

	// Check resource buffer
	if availableCPU < int(float64(node.Resources.TotalCPUCores)*ed.config.EdgeResourceBuffer) {
		return ErrInsufficientEdgeResources
	}

	if availableMemory < int64(float64(node.Resources.TotalMemoryMB)*ed.config.EdgeResourceBuffer) {
		return ErrInsufficientEdgeResources
	}

	return nil
}

// SetMECDiscovery sets the MEC discovery plugin
func (ed *EdgeDiscovery) SetMECDiscovery(plugin MECDiscovery) {
	ed.mecDiscovery = plugin
}

// SetCDNDiscovery sets the CDN discovery plugin
func (ed *EdgeDiscovery) SetCDNDiscovery(plugin CDNDiscovery) {
	ed.cdnDiscovery = plugin
}

// SetIoTDiscovery sets the IoT discovery plugin
func (ed *EdgeDiscovery) SetIoTDiscovery(plugin IoTDiscovery) {
	ed.iotDiscovery = plugin
}

// SetTelcoDiscovery sets the telco discovery plugin
func (ed *EdgeDiscovery) SetTelcoDiscovery(plugin TelcoDiscovery) {
	ed.telcoDiscovery = plugin
}

// OnNodeDiscovered sets the callback for when a node is discovered
func (ed *EdgeDiscovery) OnNodeDiscovered(callback func(*EdgeNode)) {
	ed.onNodeDiscovered = callback
}

// OnNodeLost sets the callback for when a node is lost
func (ed *EdgeDiscovery) OnNodeLost(callback func(string)) {
	ed.onNodeLost = callback
}
