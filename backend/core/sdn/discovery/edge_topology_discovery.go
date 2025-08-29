package discovery

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"net"
	"sync"
	"time"

	network "github.com/khryptorgraphics/novacron/backend/core/scheduler/network"
)

// EdgeNode represents an edge computing node in the network
type EdgeNode struct {
	ID                string                 `json:"id"`
	Name              string                 `json:"name"`
	Type              EdgeNodeType           `json:"type"`
	Location          EdgeLocation           `json:"location"`
	Capabilities      EdgeCapabilities       `json:"capabilities"`
	NetworkInterfaces []NetworkInterface     `json:"network_interfaces"`
	Status            EdgeNodeStatus         `json:"status"`
	Metrics           EdgeMetrics            `json:"metrics"`
	LastSeen          time.Time              `json:"last_seen"`
	Metadata          map[string]interface{} `json:"metadata,omitempty"`
}

type EdgeNodeType string
type EdgeNodeStatus string

const (
	EdgeNodeTypeFull        EdgeNodeType = "full"        // Full edge node with compute, storage, network
	EdgeNodeTypeCompute     EdgeNodeType = "compute"     // Compute-focused edge node
	EdgeNodeTypeStorage     EdgeNodeType = "storage"     // Storage-focused edge node
	EdgeNodeTypeNetwork     EdgeNodeType = "network"     // Network-focused edge node
	EdgeNodeTypeGateway     EdgeNodeType = "gateway"     // Edge gateway node
	EdgeNodeTypeAccelerator EdgeNodeType = "accelerator" // AI/ML accelerator node

	EdgeNodeStatusOnline     EdgeNodeStatus = "online"
	EdgeNodeStatusOffline    EdgeNodeStatus = "offline"
	EdgeNodeStatusDegraded   EdgeNodeStatus = "degraded"
	EdgeNodeStatusMaintenance EdgeNodeStatus = "maintenance"
)

// EdgeLocation represents the physical/logical location of an edge node
type EdgeLocation struct {
	Region          string         `json:"region"`
	Zone            string         `json:"zone"`
	Site            string         `json:"site"`
	Coordinates     *GeoCoordinates `json:"coordinates,omitempty"`
	NetworkTier     NetworkTier    `json:"network_tier"`
	ConnectivityType ConnectivityType `json:"connectivity_type"`
	ISP             string         `json:"isp,omitempty"`
	ASN             int            `json:"asn,omitempty"`
}

type NetworkTier int
type ConnectivityType string

const (
	NetworkTierCore NetworkTier = 1 // Core network
	NetworkTierEdge NetworkTier = 2 // Edge network
	NetworkTierFar  NetworkTier = 3 // Far edge network

	ConnectivityTypeFiber     ConnectivityType = "fiber"
	ConnectivityType5G        ConnectivityType = "5g"
	ConnectivityType4G        ConnectivityType = "4g"
	ConnectivityTypeWiFi      ConnectivityType = "wifi"
	ConnectivityTypeSatellite ConnectivityType = "satellite"
)

// GeoCoordinates represents geographic coordinates
type GeoCoordinates struct {
	Latitude  float64 `json:"latitude"`
	Longitude float64 `json:"longitude"`
	Altitude  float64 `json:"altitude,omitempty"`
}

// EdgeCapabilities describes what an edge node can do
type EdgeCapabilities struct {
	CPU           CPUCapabilities       `json:"cpu"`
	Memory        MemoryCapabilities    `json:"memory"`
	Storage       StorageCapabilities   `json:"storage"`
	Network       NetworkCapabilities   `json:"network"`
	GPU           GPUCapabilities       `json:"gpu,omitempty"`
	Accelerators  []AcceleratorInfo     `json:"accelerators,omitempty"`
	Features      []string              `json:"features,omitempty"`
}

// CPUCapabilities describes CPU capabilities
type CPUCapabilities struct {
	Architecture string  `json:"architecture"`
	Cores        int     `json:"cores"`
	Threads      int     `json:"threads"`
	BaseFreqMHz  int     `json:"base_freq_mhz"`
	MaxFreqMHz   int     `json:"max_freq_mhz"`
	CacheL3MB    int     `json:"cache_l3_mb"`
	PowerWatts   int     `json:"power_watts"`
	Features     []string `json:"features,omitempty"`
}

// MemoryCapabilities describes memory capabilities
type MemoryCapabilities struct {
	TotalGB       int    `json:"total_gb"`
	AvailableGB   int    `json:"available_gb"`
	Type          string `json:"type"` // DDR4, DDR5, HBM, etc.
	SpeedMhz      int    `json:"speed_mhz"`
	BandwidthGBps int    `json:"bandwidth_gbps"`
}

// StorageCapabilities describes storage capabilities
type StorageCapabilities struct {
	Types     []StorageType `json:"types"`
	TotalTB   int           `json:"total_tb"`
	AvailTB   int           `json:"available_tb"`
	IOPSMax   int           `json:"iops_max"`
	BandwidthMBps int       `json:"bandwidth_mbps"`
}

type StorageType struct {
	Type        string `json:"type"` // SSD, NVMe, HDD, etc.
	SizeGB      int    `json:"size_gb"`
	Interface   string `json:"interface"` // SATA, NVMe, etc.
	Performance string `json:"performance"` // High, Medium, Low
}

// NetworkCapabilities describes network capabilities
type NetworkCapabilities struct {
	Interfaces      []NetworkInterface `json:"interfaces"`
	MaxBandwidthMbps int               `json:"max_bandwidth_mbps"`
	Protocols       []string          `json:"protocols"`
	Features        []string          `json:"features"` // SR-IOV, DPDK, etc.
}

// NetworkInterface represents a network interface
type NetworkInterface struct {
	Name          string   `json:"name"`
	Type          string   `json:"type"` // ethernet, wifi, cellular, etc.
	MACAddress    string   `json:"mac_address"`
	IPAddresses   []string `json:"ip_addresses"`
	SpeedMbps     int      `json:"speed_mbps"`
	MTU           int      `json:"mtu"`
	Status        string   `json:"status"` // up, down, unknown
	CarrierStatus string   `json:"carrier_status"` // connected, disconnected
}

// GPUCapabilities describes GPU capabilities
type GPUCapabilities struct {
	Model         string `json:"model"`
	Memory        int    `json:"memory_gb"`
	ComputeUnits  int    `json:"compute_units"`
	PowerWatts    int    `json:"power_watts"`
	Architecture  string `json:"architecture"`
	Features      []string `json:"features,omitempty"`
}

// AcceleratorInfo describes specialized accelerators
type AcceleratorInfo struct {
	Type         string `json:"type"` // TPU, VPU, FPGA, etc.
	Model        string `json:"model"`
	Performance  string `json:"performance"` // TOPS, FLOPS, etc.
	PowerWatts   int    `json:"power_watts"`
	Features     []string `json:"features,omitempty"`
}

// EdgeMetrics contains runtime metrics for an edge node
type EdgeMetrics struct {
	CPU        CPUMetrics        `json:"cpu"`
	Memory     MemoryMetrics     `json:"memory"`
	Storage    StorageMetrics    `json:"storage"`
	Network    NetworkMetrics    `json:"network"`
	Power      PowerMetrics      `json:"power"`
	Thermal    ThermalMetrics    `json:"thermal"`
	Timestamp  time.Time         `json:"timestamp"`
}

// CPUMetrics contains CPU utilization metrics
type CPUMetrics struct {
	UtilizationPercent float64 `json:"utilization_percent"`
	LoadAvg1m          float64 `json:"load_avg_1m"`
	LoadAvg5m          float64 `json:"load_avg_5m"`
	LoadAvg15m         float64 `json:"load_avg_15m"`
	FreqMHz            int     `json:"freq_mhz"`
	Temperature        float64 `json:"temperature_celsius"`
}

// MemoryMetrics contains memory utilization metrics
type MemoryMetrics struct {
	UsedPercent     float64 `json:"used_percent"`
	AvailableGB     int     `json:"available_gb"`
	BufferedGB      int     `json:"buffered_gb"`
	CachedGB        int     `json:"cached_gb"`
	SwapUsedPercent float64 `json:"swap_used_percent"`
}

// StorageMetrics contains storage utilization metrics
type StorageMetrics struct {
	UsedPercent   float64 `json:"used_percent"`
	AvailableTB   float64 `json:"available_tb"`
	ReadIOPS      int     `json:"read_iops"`
	WriteIOPS     int     `json:"write_iops"`
	ReadMBps      int     `json:"read_mbps"`
	WriteMBps     int     `json:"write_mbps"`
	QueueDepth    int     `json:"queue_depth"`
}

// NetworkMetrics contains network utilization metrics
type NetworkMetrics struct {
	RxMbps          float64            `json:"rx_mbps"`
	TxMbps          float64            `json:"tx_mbps"`
	RxPacketsPerSec int                `json:"rx_packets_per_sec"`
	TxPacketsPerSec int                `json:"tx_packets_per_sec"`
	ErrorsPerSec    int                `json:"errors_per_sec"`
	DropsPerSec     int                `json:"drops_per_sec"`
	Interfaces      map[string]InterfaceMetrics `json:"interfaces"`
}

// InterfaceMetrics contains per-interface metrics
type InterfaceMetrics struct {
	RxMbps       float64 `json:"rx_mbps"`
	TxMbps       float64 `json:"tx_mbps"`
	RxPackets    int64   `json:"rx_packets"`
	TxPackets    int64   `json:"tx_packets"`
	RxErrors     int64   `json:"rx_errors"`
	TxErrors     int64   `json:"tx_errors"`
	RxDrops      int64   `json:"rx_drops"`
	TxDrops      int64   `json:"tx_drops"`
	Utilization  float64 `json:"utilization_percent"`
}

// PowerMetrics contains power consumption metrics
type PowerMetrics struct {
	ConsumptionWatts float64 `json:"consumption_watts"`
	VoltageV         float64 `json:"voltage_v"`
	CurrentA         float64 `json:"current_a"`
	BatteryPercent   float64 `json:"battery_percent,omitempty"`
	UPSStatus        string  `json:"ups_status,omitempty"`
}

// ThermalMetrics contains thermal metrics
type ThermalMetrics struct {
	CPUTempC     float64 `json:"cpu_temp_c"`
	GPUTempC     float64 `json:"gpu_temp_c,omitempty"`
	AmbientTempC float64 `json:"ambient_temp_c"`
	FanSpeedRPM  int     `json:"fan_speed_rpm"`
	ThermalState string  `json:"thermal_state"` // normal, warning, critical
}

// EdgeTopologyDiscovery manages discovery of edge network topology
type EdgeTopologyDiscovery struct {
	// Discovery state
	nodes      map[string]*EdgeNode
	nodesMutex sync.RWMutex
	
	links      map[string]*network.NetworkLink
	linksMutex sync.RWMutex
	
	// Configuration
	config DiscoveryConfig
	
	// Context for cancellation
	ctx    context.Context
	cancel context.CancelFunc
	
	// Event handlers
	eventHandlers []EdgeEventHandler
	eventMutex    sync.RWMutex
}

// DiscoveryConfig contains configuration for edge discovery
type DiscoveryConfig struct {
	DiscoveryInterval      time.Duration `json:"discovery_interval"`
	MetricsInterval        time.Duration `json:"metrics_interval"`
	HealthCheckInterval    time.Duration `json:"health_check_interval"`
	NodeTimeout            time.Duration `json:"node_timeout"`
	EnableAutoDiscovery    bool          `json:"enable_auto_discovery"`
	EnableLatencyMeasure   bool          `json:"enable_latency_measure"`
	EnableBandwidthMeasure bool          `json:"enable_bandwidth_measure"`
	MaxConcurrentProbes    int           `json:"max_concurrent_probes"`
}

// DefaultDiscoveryConfig returns default discovery configuration
func DefaultDiscoveryConfig() DiscoveryConfig {
	return DiscoveryConfig{
		DiscoveryInterval:      30 * time.Second,
		MetricsInterval:        10 * time.Second,
		HealthCheckInterval:    5 * time.Second,
		NodeTimeout:            60 * time.Second,
		EnableAutoDiscovery:    true,
		EnableLatencyMeasure:   true,
		EnableBandwidthMeasure: false, // Can be expensive
		MaxConcurrentProbes:    50,
	}
}

// EdgeEventHandler handles edge node events
type EdgeEventHandler interface {
	OnNodeDiscovered(node *EdgeNode)
	OnNodeUpdated(node *EdgeNode)
	OnNodeLost(nodeID string)
	OnLinkDiscovered(link *network.NetworkLink)
	OnLinkUpdated(link *network.NetworkLink)
	OnLinkLost(linkID string)
}

// NewEdgeTopologyDiscovery creates a new edge topology discovery service
func NewEdgeTopologyDiscovery(config DiscoveryConfig) *EdgeTopologyDiscovery {
	ctx, cancel := context.WithCancel(context.Background())
	
	return &EdgeTopologyDiscovery{
		nodes:  make(map[string]*EdgeNode),
		links:  make(map[string]*network.NetworkLink),
		config: config,
		ctx:    ctx,
		cancel: cancel,
	}
}

// Start starts the edge topology discovery service
func (d *EdgeTopologyDiscovery) Start() error {
	log.Println("Starting Edge Topology Discovery")
	
	// Start discovery loops
	go d.discoveryLoop()
	go d.metricsCollectionLoop()
	go d.healthCheckLoop()
	
	if d.config.EnableAutoDiscovery {
		go d.autoDiscoveryLoop()
	}
	
	log.Println("Edge Topology Discovery started")
	return nil
}

// Stop stops the edge topology discovery service
func (d *EdgeTopologyDiscovery) Stop() error {
	log.Println("Stopping Edge Topology Discovery")
	d.cancel()
	return nil
}

// RegisterEventHandler registers an event handler
func (d *EdgeTopologyDiscovery) RegisterEventHandler(handler EdgeEventHandler) {
	d.eventMutex.Lock()
	defer d.eventMutex.Unlock()
	
	d.eventHandlers = append(d.eventHandlers, handler)
}

// discoveryLoop runs the main discovery loop
func (d *EdgeTopologyDiscovery) discoveryLoop() {
	ticker := time.NewTicker(d.config.DiscoveryInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-d.ctx.Done():
			return
		case <-ticker.C:
			d.runDiscovery()
		}
	}
}

// runDiscovery performs edge node and link discovery
func (d *EdgeTopologyDiscovery) runDiscovery() {
	log.Println("Running edge topology discovery")
	
	// Discover nodes
	go d.discoverNodes()
	
	// Discover links between nodes
	go d.discoverLinks()
}

// discoverNodes discovers edge nodes in the network
func (d *EdgeTopologyDiscovery) discoverNodes() {
	// Multiple discovery methods can be used:
	// 1. mDNS/Bonjour discovery
	// 2. DHCP lease scanning
	// 3. Network scanning
	// 4. BGP route advertisements
	// 5. Manual registration
	
	// Example: Network scanning (simplified)
	networks := []string{
		"192.168.1.0/24",
		"10.0.0.0/24",
		"172.16.0.0/24",
	}
	
	for _, network := range networks {
		go d.scanNetwork(network)
	}
}

// scanNetwork scans a network subnet for edge nodes
func (d *EdgeTopologyDiscovery) scanNetwork(subnet string) {
	_, ipNet, err := net.ParseCIDR(subnet)
	if err != nil {
		log.Printf("Invalid subnet: %s", subnet)
		return
	}
	
	// Generate IP addresses in the subnet
	ips := d.generateIPsInSubnet(ipNet)
	
	// Limit concurrent scans
	semaphore := make(chan struct{}, d.config.MaxConcurrentProbes)
	var wg sync.WaitGroup
	
	for _, ip := range ips {
		wg.Add(1)
		go func(ipAddr string) {
			defer wg.Done()
			semaphore <- struct{}{} // Acquire
			defer func() { <-semaphore }() // Release
			
			d.probeHost(ipAddr)
		}(ip)
	}
	
	wg.Wait()
}

// generateIPsInSubnet generates IP addresses within a subnet
func (d *EdgeTopologyDiscovery) generateIPsInSubnet(ipNet *net.IPNet) []string {
	var ips []string
	
	// Convert subnet mask to prefix length
	ones, bits := ipNet.Mask.Size()
	if bits != 32 || ones > 30 { // Avoid scanning very large subnets
		return ips
	}
	
	// Generate IPs (simplified for /24 networks)
	baseIP := ipNet.IP.To4()
	if baseIP == nil {
		return ips
	}
	
	// For /24 network, scan .1 to .254
	for i := 1; i < 255; i++ {
		ip := fmt.Sprintf("%d.%d.%d.%d", baseIP[0], baseIP[1], baseIP[2], i)
		ips = append(ips, ip)
	}
	
	return ips
}

// probeHost probes a host to determine if it's an edge node
func (d *EdgeTopologyDiscovery) probeHost(ip string) {
	// Try different methods to identify if this is an edge node
	
	// 1. Try SSH (port 22)
	if d.isPortOpen(ip, 22) {
		if node := d.identifyEdgeNodeSSH(ip); node != nil {
			d.addOrUpdateNode(node)
			return
		}
	}
	
	// 2. Try HTTP/HTTPS (ports 80, 443)
	if d.isPortOpen(ip, 80) || d.isPortOpen(ip, 443) {
		if node := d.identifyEdgeNodeHTTP(ip); node != nil {
			d.addOrUpdateNode(node)
			return
		}
	}
	
	// 3. Try SNMP (port 161)
	if d.isPortOpen(ip, 161) {
		if node := d.identifyEdgeNodeSNMP(ip); node != nil {
			d.addOrUpdateNode(node)
			return
		}
	}
	
	// 4. Try custom edge node API (configurable port)
	edgePorts := []int{9000, 9001, 8080, 8081}
	for _, port := range edgePorts {
		if d.isPortOpen(ip, port) {
			if node := d.identifyEdgeNodeAPI(ip, port); node != nil {
				d.addOrUpdateNode(node)
				return
			}
		}
	}
}

// isPortOpen checks if a TCP port is open on a host
func (d *EdgeTopologyDiscovery) isPortOpen(host string, port int) bool {
	conn, err := net.DialTimeout("tcp", fmt.Sprintf("%s:%d", host, port), 2*time.Second)
	if err != nil {
		return false
	}
	conn.Close()
	return true
}

// identifyEdgeNodeSSH identifies edge node via SSH
func (d *EdgeTopologyDiscovery) identifyEdgeNodeSSH(ip string) *EdgeNode {
	// In a real implementation, this would:
	// 1. SSH to the host
	// 2. Run commands to gather system info
	// 3. Determine if it's an edge computing node
	// 4. Return node information
	
	// Mock implementation
	node := &EdgeNode{
		ID:   fmt.Sprintf("edge-%s", ip),
		Name: fmt.Sprintf("edge-node-%s", ip),
		Type: EdgeNodeTypeFull,
		Location: EdgeLocation{
			Region: "unknown",
			Zone:   "unknown",
			Site:   ip,
		},
		Status:   EdgeNodeStatusOnline,
		LastSeen: time.Now(),
	}
	
	return node
}

// identifyEdgeNodeHTTP identifies edge node via HTTP
func (d *EdgeTopologyDiscovery) identifyEdgeNodeHTTP(ip string) *EdgeNode {
	// Mock implementation
	return nil
}

// identifyEdgeNodeSNMP identifies edge node via SNMP
func (d *EdgeTopologyDiscovery) identifyEdgeNodeSNMP(ip string) *EdgeNode {
	// Mock implementation  
	return nil
}

// identifyEdgeNodeAPI identifies edge node via custom API
func (d *EdgeTopologyDiscovery) identifyEdgeNodeAPI(ip string, port int) *EdgeNode {
	// Mock implementation
	return nil
}

// addOrUpdateNode adds or updates an edge node
func (d *EdgeTopologyDiscovery) addOrUpdateNode(node *EdgeNode) {
	d.nodesMutex.Lock()
	defer d.nodesMutex.Unlock()
	
	existing, exists := d.nodes[node.ID]
	if exists {
		// Update existing node
		existing.Status = node.Status
		existing.Metrics = node.Metrics
		existing.LastSeen = time.Now()
		
		// Notify handlers
		d.notifyNodeUpdated(existing)
	} else {
		// Add new node
		d.nodes[node.ID] = node
		
		// Notify handlers
		d.notifyNodeDiscovered(node)
	}
	
	log.Printf("Added/updated edge node: %s (%s)", node.Name, node.ID)
}

// discoverLinks discovers network links between edge nodes
func (d *EdgeTopologyDiscovery) discoverLinks() {
	d.nodesMutex.RLock()
	nodeList := make([]*EdgeNode, 0, len(d.nodes))
	for _, node := range d.nodes {
		nodeList = append(nodeList, node)
	}
	d.nodesMutex.RUnlock()
	
	// Test connectivity between all pairs of nodes
	for i, node1 := range nodeList {
		for j, node2 := range nodeList[i+1:] {
			go d.measureLink(node1, node2)
			
			// Add rate limiting to avoid overwhelming the network
			if (i+j)%10 == 0 {
				time.Sleep(100 * time.Millisecond)
			}
		}
	}
}

// measureLink measures network characteristics between two nodes
func (d *EdgeTopologyDiscovery) measureLink(node1, node2 *EdgeNode) {
	// Extract IP addresses from nodes
	ip1 := d.extractPrimaryIP(node1)
	ip2 := d.extractPrimaryIP(node2)
	
	if ip1 == "" || ip2 == "" {
		return
	}
	
	// Measure latency
	var latency time.Duration
	if d.config.EnableLatencyMeasure {
		latency = d.measureLatency(ip1, ip2)
	}
	
	// Estimate bandwidth (optional, can be expensive)
	var bandwidth float64
	if d.config.EnableBandwidthMeasure {
		bandwidth = d.measureBandwidth(ip1, ip2)
	} else {
		bandwidth = d.estimateBandwidth(node1, node2) // Use heuristics
	}
	
	// Determine link type
	linkType := d.determineLinkType(node1, node2)
	
	// Create network link
	link := &network.NetworkLink{
		SourceID:      node1.ID,
		DestinationID: node2.ID,
		Type:          linkType,
		Bandwidth:     bandwidth,
		Latency:       float64(latency.Nanoseconds()) / 1e6, // Convert to milliseconds
		Loss:          0.0, // Would need actual measurement
		Jitter:        0.0, // Would need actual measurement
		Cost:          0.0, // Will be calculated by cost estimator
		Utilization:   0.0, // Initial value
		Distance:      d.calculateDistance(node1, node2),
	}
	
	d.addOrUpdateLink(link)
}

// extractPrimaryIP extracts the primary IP address from an edge node
func (d *EdgeTopologyDiscovery) extractPrimaryIP(node *EdgeNode) string {
	for _, iface := range node.NetworkInterfaces {
		if len(iface.IPAddresses) > 0 && iface.Status == "up" {
			return iface.IPAddresses[0]
		}
	}
	
	// Fallback: extract from node ID if it contains IP
	if net.ParseIP(node.Location.Site) != nil {
		return node.Location.Site
	}
	
	return ""
}

// measureLatency measures network latency between two IPs
func (d *EdgeTopologyDiscovery) measureLatency(ip1, ip2 string) time.Duration {
	// Simple TCP connect time measurement
	start := time.Now()
	conn, err := net.DialTimeout("tcp", fmt.Sprintf("%s:22", ip2), 2*time.Second)
	if err != nil {
		return 100 * time.Millisecond // Default high latency
	}
	latency := time.Since(start)
	conn.Close()
	
	return latency
}

// measureBandwidth measures network bandwidth between two IPs
func (d *EdgeTopologyDiscovery) measureBandwidth(ip1, ip2 string) float64 {
	// This would implement actual bandwidth measurement
	// For now, return a default value
	return 1000.0 // 1 Gbps default
}

// estimateBandwidth estimates bandwidth based on node capabilities
func (d *EdgeTopologyDiscovery) estimateBandwidth(node1, node2 *EdgeNode) float64 {
	// Use the minimum of both nodes' maximum interface speeds
	max1 := d.getMaxInterfaceSpeed(node1)
	max2 := d.getMaxInterfaceSpeed(node2)
	
	if max1 < max2 {
		return float64(max1)
	}
	return float64(max2)
}

// getMaxInterfaceSpeed gets the maximum interface speed of a node
func (d *EdgeTopologyDiscovery) getMaxInterfaceSpeed(node *EdgeNode) int {
	maxSpeed := 100 // Default 100 Mbps
	
	for _, iface := range node.NetworkInterfaces {
		if iface.SpeedMbps > maxSpeed {
			maxSpeed = iface.SpeedMbps
		}
	}
	
	return maxSpeed
}

// determineLinkType determines the type of network link between nodes
func (d *EdgeTopologyDiscovery) determineLinkType(node1, node2 *EdgeNode) network.LinkType {
	// Same site
	if node1.Location.Site == node2.Location.Site {
		return network.LinkTypeSameMachine
	}
	
	// Same zone
	if node1.Location.Zone == node2.Location.Zone {
		return network.LinkTypeSameDatacenter
	}
	
	// Same region
	if node1.Location.Region == node2.Location.Region {
		return network.LinkTypeInterDatacenter
	}
	
	// Different regions
	return network.LinkTypeWAN
}

// calculateDistance calculates the distance between two nodes
func (d *EdgeTopologyDiscovery) calculateDistance(node1, node2 *EdgeNode) float64 {
	if node1.Location.Coordinates == nil || node2.Location.Coordinates == nil {
		return 0.0 // Unknown distance
	}
	
	// Haversine formula for calculating distance between two points on Earth
	return d.haversineDistance(
		node1.Location.Coordinates.Latitude,
		node1.Location.Coordinates.Longitude,
		node2.Location.Coordinates.Latitude,
		node2.Location.Coordinates.Longitude,
	)
}

// haversineDistance calculates the haversine distance between two geographic points
func (d *EdgeTopologyDiscovery) haversineDistance(lat1, lon1, lat2, lon2 float64) float64 {
	const earthRadiusKm = 6371.0
	
	// Convert degrees to radians
	lat1Rad := lat1 * math.Pi / 180
	lon1Rad := lon1 * math.Pi / 180
	lat2Rad := lat2 * math.Pi / 180
	lon2Rad := lon2 * math.Pi / 180
	
	deltaLat := lat2Rad - lat1Rad
	deltaLon := lon2Rad - lon1Rad
	
	a := math.Sin(deltaLat/2)*math.Sin(deltaLat/2) +
		math.Cos(lat1Rad)*math.Cos(lat2Rad)*
			math.Sin(deltaLon/2)*math.Sin(deltaLon/2)
	c := 2 * math.Atan2(math.Sqrt(a), math.Sqrt(1-a))
	
	return earthRadiusKm * c * 1000 // Convert to meters
}

// addOrUpdateLink adds or updates a network link
func (d *EdgeTopologyDiscovery) addOrUpdateLink(link *network.NetworkLink) {
	d.linksMutex.Lock()
	defer d.linksMutex.Unlock()
	
	linkID := fmt.Sprintf("%s-%s", link.SourceID, link.DestinationID)
	
	existing, exists := d.links[linkID]
	if exists {
		// Update existing link
		existing.Bandwidth = link.Bandwidth
		existing.Latency = link.Latency
		existing.Distance = link.Distance
		
		// Notify handlers
		d.notifyLinkUpdated(existing)
	} else {
		// Add new link
		d.links[linkID] = link
		
		// Notify handlers
		d.notifyLinkDiscovered(link)
	}
	
	log.Printf("Added/updated network link: %s -> %s (%.1f ms, %.0f Mbps)",
		link.SourceID, link.DestinationID, link.Latency, link.Bandwidth)
}

// metricsCollectionLoop collects metrics from edge nodes
func (d *EdgeTopologyDiscovery) metricsCollectionLoop() {
	ticker := time.NewTicker(d.config.MetricsInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-d.ctx.Done():
			return
		case <-ticker.C:
			d.collectMetrics()
		}
	}
}

// collectMetrics collects metrics from all discovered edge nodes
func (d *EdgeTopologyDiscovery) collectMetrics() {
	d.nodesMutex.RLock()
	nodes := make([]*EdgeNode, 0, len(d.nodes))
	for _, node := range d.nodes {
		nodes = append(nodes, node)
	}
	d.nodesMutex.RUnlock()
	
	// Collect metrics from all nodes concurrently
	var wg sync.WaitGroup
	semaphore := make(chan struct{}, d.config.MaxConcurrentProbes)
	
	for _, node := range nodes {
		wg.Add(1)
		go func(n *EdgeNode) {
			defer wg.Done()
			semaphore <- struct{}{}
			defer func() { <-semaphore }()
			
			d.collectNodeMetrics(n)
		}(node)
	}
	
	wg.Wait()
}

// collectNodeMetrics collects metrics from a single edge node
func (d *EdgeTopologyDiscovery) collectNodeMetrics(node *EdgeNode) {
	// In a real implementation, this would collect actual metrics
	// via SNMP, SSH, HTTP API, or agent protocols
	
	// Mock metrics for demonstration
	metrics := EdgeMetrics{
		CPU: CPUMetrics{
			UtilizationPercent: 25.0 + float64(time.Now().Second()%50), // Simulated variance
			LoadAvg1m:          1.2,
			LoadAvg5m:          1.0,
			LoadAvg15m:         0.8,
			FreqMHz:            2400,
			Temperature:        65.0,
		},
		Memory: MemoryMetrics{
			UsedPercent:     60.0,
			AvailableGB:     16,
			BufferedGB:      2,
			CachedGB:        8,
			SwapUsedPercent: 10.0,
		},
		Network: NetworkMetrics{
			RxMbps:          100.0,
			TxMbps:          80.0,
			RxPacketsPerSec: 1000,
			TxPacketsPerSec: 800,
			ErrorsPerSec:    0,
			DropsPerSec:     0,
		},
		Power: PowerMetrics{
			ConsumptionWatts: 150.0,
			VoltageV:         12.0,
			CurrentA:         12.5,
		},
		Thermal: ThermalMetrics{
			CPUTempC:     65.0,
			AmbientTempC: 25.0,
			FanSpeedRPM:  2000,
			ThermalState: "normal",
		},
		Timestamp: time.Now(),
	}
	
	// Update node metrics
	d.nodesMutex.Lock()
	if existingNode, exists := d.nodes[node.ID]; exists {
		existingNode.Metrics = metrics
		existingNode.LastSeen = time.Now()
	}
	d.nodesMutex.Unlock()
}

// healthCheckLoop performs health checks on edge nodes
func (d *EdgeTopologyDiscovery) healthCheckLoop() {
	ticker := time.NewTicker(d.config.HealthCheckInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-d.ctx.Done():
			return
		case <-ticker.C:
			d.performHealthChecks()
		}
	}
}

// performHealthChecks performs health checks on all nodes
func (d *EdgeTopologyDiscovery) performHealthChecks() {
	d.nodesMutex.Lock()
	defer d.nodesMutex.Unlock()
	
	now := time.Now()
	for nodeID, node := range d.nodes {
		// Check if node has timed out
		if now.Sub(node.LastSeen) > d.config.NodeTimeout {
			if node.Status != EdgeNodeStatusOffline {
				node.Status = EdgeNodeStatusOffline
				log.Printf("Node %s marked as offline", nodeID)
				
				// Notify handlers
				go d.notifyNodeLost(nodeID)
			}
		}
	}
}

// autoDiscoveryLoop performs automatic discovery of new edge nodes
func (d *EdgeTopologyDiscovery) autoDiscoveryLoop() {
	// This could implement various auto-discovery mechanisms:
	// 1. Listening for mDNS announcements
	// 2. Monitoring DHCP logs
	// 3. BGP route advertisements
	// 4. Cloud provider APIs
	// 5. Container orchestrator APIs
	
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()
	
	for {
		select {
		case <-d.ctx.Done():
			return
		case <-ticker.C:
			log.Println("Running auto-discovery")
			// Implementation would go here
		}
	}
}

// Event notification methods
func (d *EdgeTopologyDiscovery) notifyNodeDiscovered(node *EdgeNode) {
	d.eventMutex.RLock()
	defer d.eventMutex.RUnlock()
	
	for _, handler := range d.eventHandlers {
		go handler.OnNodeDiscovered(node)
	}
}

func (d *EdgeTopologyDiscovery) notifyNodeUpdated(node *EdgeNode) {
	d.eventMutex.RLock()
	defer d.eventMutex.RUnlock()
	
	for _, handler := range d.eventHandlers {
		go handler.OnNodeUpdated(node)
	}
}

func (d *EdgeTopologyDiscovery) notifyNodeLost(nodeID string) {
	d.eventMutex.RLock()
	defer d.eventMutex.RUnlock()
	
	for _, handler := range d.eventHandlers {
		go handler.OnNodeLost(nodeID)
	}
}

func (d *EdgeTopologyDiscovery) notifyLinkDiscovered(link *network.NetworkLink) {
	d.eventMutex.RLock()
	defer d.eventMutex.RUnlock()
	
	for _, handler := range d.eventHandlers {
		go handler.OnLinkDiscovered(link)
	}
}

func (d *EdgeTopologyDiscovery) notifyLinkUpdated(link *network.NetworkLink) {
	d.eventMutex.RLock()
	defer d.eventMutex.RUnlock()
	
	for _, handler := range d.eventHandlers {
		go handler.OnLinkUpdated(link)
	}
}

func (d *EdgeTopologyDiscovery) notifyLinkLost(linkID string) {
	d.eventMutex.RLock()
	defer d.eventMutex.RUnlock()
	
	for _, handler := range d.eventHandlers {
		go handler.OnLinkLost(linkID)
	}
}

// Public API methods

// GetNodes returns all discovered edge nodes
func (d *EdgeTopologyDiscovery) GetNodes() []*EdgeNode {
	d.nodesMutex.RLock()
	defer d.nodesMutex.RUnlock()
	
	nodes := make([]*EdgeNode, 0, len(d.nodes))
	for _, node := range d.nodes {
		nodes = append(nodes, node)
	}
	
	return nodes
}

// GetNode returns a specific edge node
func (d *EdgeTopologyDiscovery) GetNode(nodeID string) (*EdgeNode, error) {
	d.nodesMutex.RLock()
	defer d.nodesMutex.RUnlock()
	
	node, exists := d.nodes[nodeID]
	if !exists {
		return nil, fmt.Errorf("node not found: %s", nodeID)
	}
	
	return node, nil
}

// GetLinks returns all discovered network links
func (d *EdgeTopologyDiscovery) GetLinks() []*network.NetworkLink {
	d.linksMutex.RLock()
	defer d.linksMutex.RUnlock()
	
	links := make([]*network.NetworkLink, 0, len(d.links))
	for _, link := range d.links {
		links = append(links, link)
	}
	
	return links
}