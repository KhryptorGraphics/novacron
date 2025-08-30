package topology

import (
	"context"
	"fmt"
	"log"
	"net"
	"os/exec"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/khryptorgraphics/novacron/backend/core/network/ovs"
)

// TopologyDiscoveryEngine discovers and maintains network topology
type TopologyDiscoveryEngine struct {
	// Configuration
	config DiscoveryConfig
	
	// State management
	topology     *NetworkTopology
	nodes        map[string]*NetworkNode
	links        map[string]*NetworkLink
	subnets      map[string]*NetworkSubnet
	topologyMutex sync.RWMutex
	nodesMutex   sync.RWMutex
	linksMutex   sync.RWMutex
	subnetsMutex sync.RWMutex
	
	// Dependencies
	bridgeManager *ovs.BridgeManager
	
	// Discovery state
	discoveryTasks map[string]*DiscoveryTask
	tasksMutex     sync.RWMutex
	
	// Runtime
	ctx         context.Context
	cancel      context.CancelFunc
	initialized bool
}

// DiscoveryConfig holds configuration for topology discovery
type DiscoveryConfig struct {
	// Discovery intervals
	FullDiscoveryInterval    time.Duration `json:"full_discovery_interval"`
	IncrementalInterval     time.Duration `json:"incremental_interval"`
	LinkStateCheckInterval  time.Duration `json:"link_state_check_interval"`
	HostDiscoveryInterval   time.Duration `json:"host_discovery_interval"`
	
	// Discovery methods
	EnableLLDPDiscovery     bool `json:"enable_lldp_discovery"`
	EnableARPDiscovery      bool `json:"enable_arp_discovery"`
	EnableBGPDiscovery      bool `json:"enable_bgp_discovery"`
	EnableOSPFDiscovery     bool `json:"enable_ospf_discovery"`
	EnableSNMPDiscovery     bool `json:"enable_snmp_discovery"`
	EnableNetConfDiscovery  bool `json:"enable_netconf_discovery"`
	
	// Discovery scope
	MaxHopCount             int      `json:"max_hop_count"`
	DiscoveryNetworks       []string `json:"discovery_networks"`
	ExcludeNetworks         []string `json:"exclude_networks"`
	
	// Performance
	MaxConcurrentDiscovery  int           `json:"max_concurrent_discovery"`
	DiscoveryTimeout        time.Duration `json:"discovery_timeout"`
	RetryAttempts          int           `json:"retry_attempts"`
	RetryDelay             time.Duration `json:"retry_delay"`
	
	// SNMP configuration
	SNMPCommunity          string `json:"snmp_community"`
	SNMPVersion           string `json:"snmp_version"`
	SNMPPort              int    `json:"snmp_port"`
	
	// Credentials for device access
	SSHUsername           string            `json:"ssh_username"`
	SSHKeyPath           string            `json:"ssh_key_path"`
	DeviceCredentials    map[string]string `json:"device_credentials"`
}

// DefaultDiscoveryConfig returns default discovery configuration
func DefaultDiscoveryConfig() DiscoveryConfig {
	return DiscoveryConfig{
		FullDiscoveryInterval:   300 * time.Second,
		IncrementalInterval:     60 * time.Second,
		LinkStateCheckInterval:  30 * time.Second,
		HostDiscoveryInterval:   120 * time.Second,
		EnableLLDPDiscovery:     true,
		EnableARPDiscovery:      true,
		EnableBGPDiscovery:      false,
		EnableOSPFDiscovery:     false,
		EnableSNMPDiscovery:     false,
		EnableNetConfDiscovery:  false,
		MaxHopCount:            10,
		DiscoveryNetworks:      []string{"0.0.0.0/0"},
		ExcludeNetworks:        []string{"127.0.0.0/8", "169.254.0.0/16"},
		MaxConcurrentDiscovery: 50,
		DiscoveryTimeout:       30 * time.Second,
		RetryAttempts:         3,
		RetryDelay:            5 * time.Second,
		SNMPCommunity:         "public",
		SNMPVersion:           "2c",
		SNMPPort:              161,
		SSHUsername:           "admin",
		DeviceCredentials:     make(map[string]string),
	}
}

// NetworkTopology represents the complete network topology
type NetworkTopology struct {
	ID              string                 `json:"id"`
	Name            string                 `json:"name"`
	Nodes           map[string]*NetworkNode `json:"nodes"`
	Links           map[string]*NetworkLink `json:"links"`
	Subnets         map[string]*NetworkSubnet `json:"subnets"`
	Metadata        TopologyMetadata       `json:"metadata"`
	Version         int                    `json:"version"`
	CreatedAt       time.Time              `json:"created_at"`
	LastUpdated     time.Time              `json:"last_updated"`
	LastFullScan    time.Time              `json:"last_full_scan"`
}

// NetworkNode represents a node in the network topology
type NetworkNode struct {
	ID                string                 `json:"id"`
	Name              string                 `json:"name"`
	Type              NodeType               `json:"type"`
	IPAddresses       []string               `json:"ip_addresses"`
	MACAddresses      []string               `json:"mac_addresses"`
	Interfaces        map[string]*NetworkInterface `json:"interfaces"`
	Properties        NodeProperties         `json:"properties"`
	Location          NodeLocation           `json:"location"`
	Status            NodeStatus             `json:"status"`
	Capabilities      []string               `json:"capabilities"`
	Protocols         []string               `json:"protocols"`
	RoutingTable      []RouteEntry           `json:"routing_table,omitempty"`
	ARPTable          []ARPEntry             `json:"arp_table,omitempty"`
	ConnectedLinks    []string               `json:"connected_links"`
	Metadata          map[string]interface{} `json:"metadata,omitempty"`
	DiscoveredAt      time.Time              `json:"discovered_at"`
	LastSeen          time.Time              `json:"last_seen"`
	LastUpdated       time.Time              `json:"last_updated"`
}

// NetworkInterface represents a network interface on a node
type NetworkInterface struct {
	ID              string                 `json:"id"`
	Name            string                 `json:"name"`
	Description     string                 `json:"description,omitempty"`
	Type            InterfaceType          `json:"type"`
	MACAddress      string                 `json:"mac_address"`
	IPAddresses     []string               `json:"ip_addresses"`
	MTU             int                    `json:"mtu"`
	Speed           int64                  `json:"speed_bps"`
	Duplex          string                 `json:"duplex"`
	Status          InterfaceStatus        `json:"status"`
	VLANs           []VLANInfo             `json:"vlans,omitempty"`
	Statistics      InterfaceStatistics    `json:"statistics"`
	Properties      map[string]interface{} `json:"properties,omitempty"`
	LastUpdated     time.Time              `json:"last_updated"`
}

// NetworkLink represents a link between two network nodes
type NetworkLink struct {
	ID              string                 `json:"id"`
	Name            string                 `json:"name"`
	Type            LinkType               `json:"type"`
	SourceNodeID    string                 `json:"source_node_id"`
	TargetNodeID    string                 `json:"target_node_id"`
	SourceInterface string                 `json:"source_interface"`
	TargetInterface string                 `json:"target_interface"`
	Bandwidth       int64                  `json:"bandwidth_bps"`
	Latency         time.Duration          `json:"latency"`
	PacketLoss      float64                `json:"packet_loss_percent"`
	Status          LinkStatus             `json:"status"`
	Utilization     LinkUtilization        `json:"utilization"`
	Properties      map[string]interface{} `json:"properties,omitempty"`
	DiscoveredAt    time.Time              `json:"discovered_at"`
	LastChecked     time.Time              `json:"last_checked"`
	LastUpdated     time.Time              `json:"last_updated"`
}

// NetworkSubnet represents a network subnet
type NetworkSubnet struct {
	ID              string                 `json:"id"`
	Name            string                 `json:"name"`
	CIDR            string                 `json:"cidr"`
	Gateway         string                 `json:"gateway,omitempty"`
	DNSServers      []string               `json:"dns_servers,omitempty"`
	VLANTag         int                    `json:"vlan_tag,omitempty"`
	ConnectedNodes  []string               `json:"connected_nodes"`
	HostCount       int                    `json:"host_count"`
	ActiveHosts     int                    `json:"active_hosts"`
	Properties      map[string]interface{} `json:"properties,omitempty"`
	DiscoveredAt    time.Time              `json:"discovered_at"`
	LastScanned     time.Time              `json:"last_scanned"`
	LastUpdated     time.Time              `json:"last_updated"`
}

// TopologyMetadata contains metadata about the topology
type TopologyMetadata struct {
	NodeCount        int                    `json:"node_count"`
	LinkCount        int                    `json:"link_count"`
	SubnetCount      int                    `json:"subnet_count"`
	MaxHopCount      int                    `json:"max_hop_count"`
	DiscoveryMethods []string               `json:"discovery_methods"`
	Coverage         float64                `json:"coverage_percent"`
	Health           TopologyHealth         `json:"health"`
	Statistics       TopologyStatistics     `json:"statistics"`
}

// Enums and supporting types
type NodeType string
type InterfaceType string
type LinkType string
type NodeStatus string
type InterfaceStatus string
type LinkStatus string

const (
	NodeTypeHost       NodeType = "host"
	NodeTypeSwitch     NodeType = "switch"
	NodeTypeRouter     NodeType = "router"
	NodeTypeFirewall   NodeType = "firewall"
	NodeTypeLoadBalancer NodeType = "load_balancer"
	NodeTypeVirtual    NodeType = "virtual"
	NodeTypeContainer  NodeType = "container"
	NodeTypeUnknown    NodeType = "unknown"
	
	InterfaceTypeEthernet   InterfaceType = "ethernet"
	InterfaceTypeWireless   InterfaceType = "wireless"
	InterfaceTypeLoopback   InterfaceType = "loopback"
	InterfaceTypeVirtual    InterfaceType = "virtual"
	InterfaceTypeTunnel     InterfaceType = "tunnel"
	
	LinkTypeEthernet     LinkType = "ethernet"
	LinkTypeWireless     LinkType = "wireless"
	LinkTypeVirtual      LinkType = "virtual"
	LinkTypeTunnel       LinkType = "tunnel"
	LinkTypeLogical      LinkType = "logical"
	
	NodeStatusUp         NodeStatus = "up"
	NodeStatusDown       NodeStatus = "down"
	NodeStatusUnreachable NodeStatus = "unreachable"
	NodeStatusUnknown    NodeStatus = "unknown"
	
	InterfaceStatusUp    InterfaceStatus = "up"
	InterfaceStatusDown  InterfaceStatus = "down"
	InterfaceStatusError InterfaceStatus = "error"
	
	LinkStatusUp         LinkStatus = "up"
	LinkStatusDown       LinkStatus = "down"
	LinkStatusDegraded   LinkStatus = "degraded"
	LinkStatusUnknown    LinkStatus = "unknown"
)

// Supporting data structures
type NodeProperties struct {
	Vendor          string                 `json:"vendor,omitempty"`
	Model          string                 `json:"model,omitempty"`
	OSName         string                 `json:"os_name,omitempty"`
	OSVersion      string                 `json:"os_version,omitempty"`
	SerialNumber   string                 `json:"serial_number,omitempty"`
	Description    string                 `json:"description,omitempty"`
	Contact        string                 `json:"contact,omitempty"`
	Location       string                 `json:"location,omitempty"`
	Uptime         time.Duration          `json:"uptime,omitempty"`
	CPUInfo        map[string]interface{} `json:"cpu_info,omitempty"`
	MemoryInfo     map[string]interface{} `json:"memory_info,omitempty"`
	StorageInfo    map[string]interface{} `json:"storage_info,omitempty"`
}

type NodeLocation struct {
	Datacenter     string  `json:"datacenter,omitempty"`
	Rack          string  `json:"rack,omitempty"`
	Position      string  `json:"position,omitempty"`
	Building      string  `json:"building,omitempty"`
	Floor         string  `json:"floor,omitempty"`
	Room          string  `json:"room,omitempty"`
	Coordinates   LatLng  `json:"coordinates,omitempty"`
}

type LatLng struct {
	Latitude  float64 `json:"latitude"`
	Longitude float64 `json:"longitude"`
}

type VLANInfo struct {
	ID          int    `json:"id"`
	Name        string `json:"name,omitempty"`
	Description string `json:"description,omitempty"`
	Type        string `json:"type,omitempty"`
}

type InterfaceStatistics struct {
	BytesIn         int64     `json:"bytes_in"`
	BytesOut        int64     `json:"bytes_out"`
	PacketsIn       int64     `json:"packets_in"`
	PacketsOut      int64     `json:"packets_out"`
	ErrorsIn        int64     `json:"errors_in"`
	ErrorsOut       int64     `json:"errors_out"`
	DroppedIn       int64     `json:"dropped_in"`
	DroppedOut      int64     `json:"dropped_out"`
	Utilization     float64   `json:"utilization_percent"`
	LastUpdated     time.Time `json:"last_updated"`
}

type LinkUtilization struct {
	InboundPercent  float64   `json:"inbound_percent"`
	OutboundPercent float64   `json:"outbound_percent"`
	AveragePercent  float64   `json:"average_percent"`
	PeakPercent     float64   `json:"peak_percent"`
	LastUpdated     time.Time `json:"last_updated"`
}

type RouteEntry struct {
	Destination string `json:"destination"`
	Gateway     string `json:"gateway"`
	Interface   string `json:"interface"`
	Metric      int    `json:"metric"`
	Protocol    string `json:"protocol,omitempty"`
}

type ARPEntry struct {
	IPAddress   string `json:"ip_address"`
	MACAddress  string `json:"mac_address"`
	Interface   string `json:"interface"`
	Type        string `json:"type"`
}

type TopologyHealth struct {
	OverallScore    float64              `json:"overall_score"`
	NodeHealth      float64              `json:"node_health"`
	LinkHealth      float64              `json:"link_health"`
	Issues          []HealthIssue        `json:"issues"`
	LastAssessed    time.Time            `json:"last_assessed"`
}

type HealthIssue struct {
	ID          string                 `json:"id"`
	Severity    string                 `json:"severity"`
	Type        string                 `json:"type"`
	Description string                 `json:"description"`
	Node        string                 `json:"node,omitempty"`
	Link        string                 `json:"link,omitempty"`
	FirstSeen   time.Time              `json:"first_seen"`
	LastSeen    time.Time              `json:"last_seen"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

type TopologyStatistics struct {
	DiscoveryRuns       int       `json:"discovery_runs"`
	LastRunDuration     time.Duration `json:"last_run_duration"`
	AverageRunDuration  time.Duration `json:"average_run_duration"`
	NodesDiscovered     int       `json:"nodes_discovered"`
	LinksDiscovered     int       `json:"links_discovered"`
	ErrorsEncountered   int       `json:"errors_encountered"`
	LastSuccessfulRun   time.Time `json:"last_successful_run"`
	LastFailedRun       time.Time `json:"last_failed_run,omitempty"`
}

// DiscoveryTask represents a discovery task
type DiscoveryTask struct {
	ID            string                 `json:"id"`
	Type          DiscoveryTaskType      `json:"type"`
	Target        string                 `json:"target"`
	Status        DiscoveryTaskStatus    `json:"status"`
	Progress      float64                `json:"progress"`
	Results       interface{}            `json:"results,omitempty"`
	Error         string                 `json:"error,omitempty"`
	StartedAt     time.Time              `json:"started_at"`
	CompletedAt   time.Time              `json:"completed_at,omitempty"`
	Metadata      map[string]interface{} `json:"metadata,omitempty"`
}

type DiscoveryTaskType string
type DiscoveryTaskStatus string

const (
	TaskTypeFullDiscovery        DiscoveryTaskType = "full_discovery"
	TaskTypeIncrementalDiscovery DiscoveryTaskType = "incremental_discovery"
	TaskTypeHostDiscovery        DiscoveryTaskType = "host_discovery"
	TaskTypeLinkDiscovery        DiscoveryTaskType = "link_discovery"
	TaskTypeSubnetDiscovery      DiscoveryTaskType = "subnet_discovery"
	
	TaskStatusPending    DiscoveryTaskStatus = "pending"
	TaskStatusRunning    DiscoveryTaskStatus = "running"
	TaskStatusCompleted  DiscoveryTaskStatus = "completed"
	TaskStatusFailed     DiscoveryTaskStatus = "failed"
	TaskStatusCancelled  DiscoveryTaskStatus = "cancelled"
)

// NewTopologyDiscoveryEngine creates a new topology discovery engine
func NewTopologyDiscoveryEngine(config DiscoveryConfig, bridgeManager *ovs.BridgeManager) *TopologyDiscoveryEngine {
	ctx, cancel := context.WithCancel(context.Background())
	
	return &TopologyDiscoveryEngine{
		config:         config,
		topology:       &NetworkTopology{
			ID:      uuid.New().String(),
			Name:    "Network Topology",
			Nodes:   make(map[string]*NetworkNode),
			Links:   make(map[string]*NetworkLink),
			Subnets: make(map[string]*NetworkSubnet),
			Version: 1,
			CreatedAt: time.Now(),
		},
		nodes:           make(map[string]*NetworkNode),
		links:           make(map[string]*NetworkLink),
		subnets:         make(map[string]*NetworkSubnet),
		bridgeManager:   bridgeManager,
		discoveryTasks:  make(map[string]*DiscoveryTask),
		ctx:             ctx,
		cancel:          cancel,
		initialized:     false,
	}
}

// Start starts the topology discovery engine
func (tde *TopologyDiscoveryEngine) Start() error {
	tde.topologyMutex.Lock()
	defer tde.topologyMutex.Unlock()
	
	if tde.initialized {
		return fmt.Errorf("topology discovery engine already started")
	}
	
	log.Println("Starting Network Topology Discovery Engine")
	
	// Initial topology discovery
	go tde.performInitialDiscovery()
	
	// Start periodic discovery loops
	go tde.fullDiscoveryLoop()
	go tde.incrementalDiscoveryLoop()
	go tde.linkStateCheckLoop()
	go tde.hostDiscoveryLoop()
	
	// Start task management
	go tde.taskManagementLoop()
	
	tde.initialized = true
	log.Println("Network Topology Discovery Engine started successfully")
	return nil
}

// Stop stops the topology discovery engine
func (tde *TopologyDiscoveryEngine) Stop() error {
	log.Println("Stopping Network Topology Discovery Engine")
	tde.cancel()
	tde.initialized = false
	return nil
}

// performInitialDiscovery performs initial network discovery
func (tde *TopologyDiscoveryEngine) performInitialDiscovery() {
	log.Println("Performing initial network topology discovery")
	
	task := &DiscoveryTask{
		ID:        uuid.New().String(),
		Type:      TaskTypeFullDiscovery,
		Target:    "initial",
		Status:    TaskStatusRunning,
		StartedAt: time.Now(),
	}
	
	tde.tasksMutex.Lock()
	tde.discoveryTasks[task.ID] = task
	tde.tasksMutex.Unlock()
	
	if err := tde.performFullDiscovery(context.Background()); err != nil {
		log.Printf("Initial discovery failed: %v", err)
		task.Status = TaskStatusFailed
		task.Error = err.Error()
	} else {
		task.Status = TaskStatusCompleted
		task.Progress = 100.0
	}
	
	task.CompletedAt = time.Now()
}

// fullDiscoveryLoop runs full topology discovery periodically
func (tde *TopologyDiscoveryEngine) fullDiscoveryLoop() {
	ticker := time.NewTicker(tde.config.FullDiscoveryInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-tde.ctx.Done():
			return
		case <-ticker.C:
			log.Println("Running scheduled full topology discovery")
			if err := tde.performFullDiscovery(tde.ctx); err != nil {
				log.Printf("Scheduled full discovery failed: %v", err)
			}
		}
	}
}

// incrementalDiscoveryLoop runs incremental discovery periodically
func (tde *TopologyDiscoveryEngine) incrementalDiscoveryLoop() {
	ticker := time.NewTicker(tde.config.IncrementalInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-tde.ctx.Done():
			return
		case <-ticker.C:
			if err := tde.performIncrementalDiscovery(tde.ctx); err != nil {
				log.Printf("Incremental discovery failed: %v", err)
			}
		}
	}
}

// linkStateCheckLoop periodically checks link states
func (tde *TopologyDiscoveryEngine) linkStateCheckLoop() {
	ticker := time.NewTicker(tde.config.LinkStateCheckInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-tde.ctx.Done():
			return
		case <-ticker.C:
			tde.checkAllLinkStates()
		}
	}
}

// hostDiscoveryLoop runs host discovery periodically
func (tde *TopologyDiscoveryEngine) hostDiscoveryLoop() {
	ticker := time.NewTicker(tde.config.HostDiscoveryInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-tde.ctx.Done():
			return
		case <-ticker.C:
			if err := tde.performHostDiscovery(tde.ctx); err != nil {
				log.Printf("Host discovery failed: %v", err)
			}
		}
	}
}

// performFullDiscovery performs a complete network topology discovery
func (tde *TopologyDiscoveryEngine) performFullDiscovery(ctx context.Context) error {
	startTime := time.Now()
	
	// Discover local system first
	if err := tde.discoverLocalSystem(); err != nil {
		log.Printf("Warning: Failed to discover local system: %v", err)
	}
	
	// Discover OVS bridges and network infrastructure
	if err := tde.discoverOVSInfrastructure(); err != nil {
		log.Printf("Warning: Failed to discover OVS infrastructure: %v", err)
	}
	
	// Discover network interfaces
	if err := tde.discoverNetworkInterfaces(); err != nil {
		log.Printf("Warning: Failed to discover network interfaces: %v", err)
	}
	
	// Discover subnets
	if err := tde.discoverSubnets(); err != nil {
		log.Printf("Warning: Failed to discover subnets: %v", err)
	}
	
	// LLDP discovery if enabled
	if tde.config.EnableLLDPDiscovery {
		if err := tde.performLLDPDiscovery(); err != nil {
			log.Printf("Warning: LLDP discovery failed: %v", err)
		}
	}
	
	// ARP-based discovery if enabled
	if tde.config.EnableARPDiscovery {
		if err := tde.performARPDiscovery(); err != nil {
			log.Printf("Warning: ARP discovery failed: %v", err)
		}
	}
	
	// Update topology metadata
	tde.updateTopologyMetadata()
	
	// Update timestamps
	tde.topologyMutex.Lock()
	tde.topology.LastFullScan = time.Now()
	tde.topology.LastUpdated = time.Now()
	tde.topology.Version++
	tde.topologyMutex.Unlock()
	
	duration := time.Since(startTime)
	log.Printf("Full topology discovery completed in %v", duration)
	
	return nil
}

// discoverLocalSystem discovers the local system and its properties
func (tde *TopologyDiscoveryEngine) discoverLocalSystem() error {
	// Get hostname
	hostname, err := exec.Command("hostname").Output()
	if err != nil {
		return fmt.Errorf("failed to get hostname: %w", err)
	}
	
	hostName := strings.TrimSpace(string(hostname))
	nodeID := fmt.Sprintf("host-%s", hostName)
	
	// Get system information
	properties := NodeProperties{}
	
	// OS information
	if osInfo, err := exec.Command("uname", "-a").Output(); err == nil {
		properties.OSName = strings.TrimSpace(string(osInfo))
	}
	
	// Create local node
	node := &NetworkNode{
		ID:           nodeID,
		Name:         hostName,
		Type:         NodeTypeHost,
		IPAddresses:  []string{},
		MACAddresses: []string{},
		Interfaces:   make(map[string]*NetworkInterface),
		Properties:   properties,
		Status:       NodeStatusUp,
		Capabilities: []string{"host"},
		Protocols:    []string{},
		ConnectedLinks: []string{},
		DiscoveredAt: time.Now(),
		LastSeen:     time.Now(),
		LastUpdated:  time.Now(),
	}
	
	// Store the local node
	tde.nodesMutex.Lock()
	tde.nodes[nodeID] = node
	tde.nodesMutex.Unlock()
	
	tde.topologyMutex.Lock()
	tde.topology.Nodes[nodeID] = node
	tde.topologyMutex.Unlock()
	
	return nil
}

// discoverOVSInfrastructure discovers OVS bridges and virtual infrastructure
func (tde *TopologyDiscoveryEngine) discoverOVSInfrastructure() error {
	if tde.bridgeManager == nil {
		return fmt.Errorf("bridge manager not available")
	}
	
	bridges := tde.bridgeManager.ListBridges()
	for _, bridge := range bridges {
		nodeID := fmt.Sprintf("ovs-bridge-%s", bridge.Name)
		
		node := &NetworkNode{
			ID:          nodeID,
			Name:        bridge.Name,
			Type:        NodeTypeSwitch,
			IPAddresses: []string{},
			MACAddresses: []string{},
			Interfaces:  make(map[string]*NetworkInterface),
			Properties: NodeProperties{
				Vendor:      "Open vSwitch",
				Description: fmt.Sprintf("OVS Bridge %s", bridge.Name),
			},
			Status:      NodeStatusUp,
			Capabilities: []string{"switching", "openflow"},
			Protocols:   []string{"openflow", "stp", "rstp"},
			ConnectedLinks: []string{},
			DiscoveredAt: time.Now(),
			LastSeen:    time.Now(),
			LastUpdated: time.Now(),
		}
		
		// Add bridge ports as interfaces
		for _, port := range bridge.Ports {
			interfaceID := fmt.Sprintf("%s-%s", nodeID, port.Name)
			
			iface := &NetworkInterface{
				ID:          interfaceID,
				Name:        port.Name,
				Type:        tde.convertPortTypeToInterfaceType(port.Type),
				Status:      tde.convertPortStatusToInterfaceStatus(port.Status),
				MTU:         1500, // Default MTU
				Statistics:  InterfaceStatistics{LastUpdated: time.Now()},
				LastUpdated: time.Now(),
			}
			
			node.Interfaces[port.Name] = iface
		}
		
		// Store the bridge node
		tde.nodesMutex.Lock()
		tde.nodes[nodeID] = node
		tde.nodesMutex.Unlock()
		
		tde.topologyMutex.Lock()
		tde.topology.Nodes[nodeID] = node
		tde.topologyMutex.Unlock()
	}
	
	return nil
}

// discoverNetworkInterfaces discovers network interfaces on the local system
func (tde *TopologyDiscoveryEngine) discoverNetworkInterfaces() error {
	// Get interface list
	cmd := exec.Command("ip", "link", "show")
	output, err := cmd.Output()
	if err != nil {
		return fmt.Errorf("failed to get interface list: %w", err)
	}
	
	lines := strings.Split(string(output), "\n")
	for _, line := range lines {
		if strings.Contains(line, ": ") && !strings.HasPrefix(line, " ") {
			if err := tde.parseInterfaceLine(line); err != nil {
				log.Printf("Warning: Failed to parse interface line '%s': %v", line, err)
			}
		}
	}
	
	return nil
}

// parseInterfaceLine parses a single interface line from ip link show
func (tde *TopologyDiscoveryEngine) parseInterfaceLine(line string) error {
	// Parse interface name and properties
	// Example: "2: eth0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc pfifo_fast state UP mode DEFAULT group default qlen 1000"
	
	parts := strings.Split(line, ":")
	if len(parts) < 3 {
		return fmt.Errorf("invalid interface line format")
	}
	
	interfaceName := strings.TrimSpace(parts[1])
	if interfaceName == "lo" || strings.HasPrefix(interfaceName, "veth") {
		return nil // Skip loopback and veth interfaces
	}
	
	// Create interface entry
	interfaceID := fmt.Sprintf("local-interface-%s", interfaceName)
	
	iface := &NetworkInterface{
		ID:          interfaceID,
		Name:        interfaceName,
		Type:        InterfaceTypeEthernet,
		Status:      tde.parseInterfaceStatus(line),
		MTU:         tde.parseInterfaceMTU(line),
		Statistics:  InterfaceStatistics{LastUpdated: time.Now()},
		LastUpdated: time.Now(),
	}
	
	// Get interface details
	if err := tde.getInterfaceDetails(iface); err != nil {
		log.Printf("Warning: Failed to get details for interface %s: %v", interfaceName, err)
	}
	
	// Find or create local host node
	localNodeID := "host-local"
	tde.nodesMutex.Lock()
	if node, exists := tde.nodes[localNodeID]; exists {
		node.Interfaces[interfaceName] = iface
	}
	tde.nodesMutex.Unlock()
	
	return nil
}

// getInterfaceDetails gets detailed information about an interface
func (tde *TopologyDiscoveryEngine) getInterfaceDetails(iface *NetworkInterface) error {
	// Get MAC address
	cmd := exec.Command("ip", "link", "show", iface.Name)
	output, err := cmd.Output()
	if err == nil {
		if mac := tde.extractMACAddress(string(output)); mac != "" {
			iface.MACAddress = mac
		}
	}
	
	// Get IP addresses
	cmd = exec.Command("ip", "addr", "show", iface.Name)
	output, err = cmd.Output()
	if err == nil {
		iface.IPAddresses = tde.extractIPAddresses(string(output))
	}
	
	return nil
}

// discoverSubnets discovers network subnets
func (tde *TopologyDiscoveryEngine) discoverSubnets() error {
	// Get routing table to discover connected subnets
	cmd := exec.Command("ip", "route", "show")
	output, err := cmd.Output()
	if err != nil {
		return fmt.Errorf("failed to get routing table: %w", err)
	}
	
	lines := strings.Split(string(output), "\n")
	for _, line := range lines {
		if subnet := tde.parseSubnetFromRoute(line); subnet != nil {
			subnetID := fmt.Sprintf("subnet-%s", strings.Replace(subnet.CIDR, "/", "-", -1))
			
			tde.subnetsMutex.Lock()
			tde.subnets[subnetID] = subnet
			tde.subnetsMutex.Unlock()
			
			tde.topologyMutex.Lock()
			tde.topology.Subnets[subnetID] = subnet
			tde.topologyMutex.Unlock()
		}
	}
	
	return nil
}

// parseSubnetFromRoute parses subnet information from a route line
func (tde *TopologyDiscoveryEngine) parseSubnetFromRoute(line string) *NetworkSubnet {
	// Example: "192.168.1.0/24 dev eth0 proto kernel scope link src 192.168.1.10"
	
	parts := strings.Fields(line)
	if len(parts) < 3 {
		return nil
	}
	
	cidr := parts[0]
	if !strings.Contains(cidr, "/") {
		return nil
	}
	
	// Validate CIDR
	if _, _, err := net.ParseCIDR(cidr); err != nil {
		return nil
	}
	
	subnet := &NetworkSubnet{
		ID:           fmt.Sprintf("subnet-%s", strings.Replace(cidr, "/", "-", -1)),
		Name:         fmt.Sprintf("Subnet %s", cidr),
		CIDR:         cidr,
		ConnectedNodes: []string{},
		DiscoveredAt: time.Now(),
		LastScanned:  time.Now(),
		LastUpdated:  time.Now(),
	}
	
	// Extract gateway if present
	for i, part := range parts {
		if part == "src" && i+1 < len(parts) {
			subnet.Gateway = parts[i+1]
			break
		}
	}
	
	return subnet
}

// performLLDPDiscovery performs LLDP-based network discovery
func (tde *TopologyDiscoveryEngine) performLLDPDiscovery() error {
	log.Println("Performing LLDP discovery")
	
	// Check if lldpctl is available
	if _, err := exec.LookPath("lldpctl"); err != nil {
		return fmt.Errorf("lldpctl not available: %w", err)
	}
	
	// Get LLDP neighbors
	cmd := exec.Command("lldpctl", "-f", "xml")
	output, err := cmd.Output()
	if err != nil {
		return fmt.Errorf("failed to run lldpctl: %w", err)
	}
	
	// Parse LLDP output (simplified - would use proper XML parsing)
	log.Printf("LLDP output received: %d bytes", len(output))
	
	return nil
}

// performARPDiscovery performs ARP-based network discovery
func (tde *TopologyDiscoveryEngine) performARPDiscovery() error {
	log.Println("Performing ARP-based discovery")
	
	// Get ARP table
	cmd := exec.Command("ip", "neigh", "show")
	output, err := cmd.Output()
	if err != nil {
		return fmt.Errorf("failed to get ARP table: %w", err)
	}
	
	lines := strings.Split(string(output), "\n")
	for _, line := range lines {
		if entry := tde.parseARPEntry(line); entry != nil {
			tde.processARPEntry(entry)
		}
	}
	
	return nil
}

// parseARPEntry parses an ARP entry from ip neigh output
func (tde *TopologyDiscoveryEngine) parseARPEntry(line string) *ARPEntry {
	// Example: "192.168.1.1 dev eth0 lladdr 00:11:22:33:44:55 REACHABLE"
	
	parts := strings.Fields(line)
	if len(parts) < 6 {
		return nil
	}
	
	ipAddress := parts[0]
	interface_ := ""
	macAddress := ""
	
	for i, part := range parts {
		if part == "dev" && i+1 < len(parts) {
			interface_ = parts[i+1]
		}
		if part == "lladdr" && i+1 < len(parts) {
			macAddress = parts[i+1]
		}
	}
	
	if ipAddress == "" || macAddress == "" {
		return nil
	}
	
	return &ARPEntry{
		IPAddress:  ipAddress,
		MACAddress: macAddress,
		Interface:  interface_,
		Type:       "dynamic",
	}
}

// processARPEntry processes an ARP entry and creates/updates nodes
func (tde *TopologyDiscoveryEngine) processARPEntry(entry *ARPEntry) {
	nodeID := fmt.Sprintf("host-%s", entry.IPAddress)
	
	tde.nodesMutex.Lock()
	defer tde.nodesMutex.Unlock()
	
	node, exists := tde.nodes[nodeID]
	if !exists {
		node = &NetworkNode{
			ID:           nodeID,
			Name:         entry.IPAddress,
			Type:         NodeTypeHost,
			IPAddresses:  []string{entry.IPAddress},
			MACAddresses: []string{entry.MACAddress},
			Interfaces:   make(map[string]*NetworkInterface),
			Properties:   NodeProperties{},
			Status:       NodeStatusUp,
			Capabilities: []string{"host"},
			ConnectedLinks: []string{},
			DiscoveredAt: time.Now(),
			LastSeen:     time.Now(),
			LastUpdated:  time.Now(),
		}
		
		tde.nodes[nodeID] = node
		
		tde.topologyMutex.Lock()
		tde.topology.Nodes[nodeID] = node
		tde.topologyMutex.Unlock()
	} else {
		node.LastSeen = time.Now()
		node.LastUpdated = time.Now()
		
		// Add IP/MAC if not already present
		if !tde.containsString(node.IPAddresses, entry.IPAddress) {
			node.IPAddresses = append(node.IPAddresses, entry.IPAddress)
		}
		if !tde.containsString(node.MACAddresses, entry.MACAddress) {
			node.MACAddresses = append(node.MACAddresses, entry.MACAddress)
		}
	}
}

// Helper functions

func (tde *TopologyDiscoveryEngine) convertPortTypeToInterfaceType(portType ovs.PortType) InterfaceType {
	switch portType {
	case ovs.PortTypeVXLAN, ovs.PortTypeGRE, ovs.PortTypeGeneve:
		return InterfaceTypeTunnel
	case ovs.PortTypeInternal:
		return InterfaceTypeVirtual
	case ovs.PortTypeVETH:
		return InterfaceTypeVirtual
	default:
		return InterfaceTypeEthernet
	}
}

func (tde *TopologyDiscoveryEngine) convertPortStatusToInterfaceStatus(status ovs.PortStatus) InterfaceStatus {
	if status.Link {
		return InterfaceStatusUp
	}
	return InterfaceStatusDown
}

func (tde *TopologyDiscoveryEngine) parseInterfaceStatus(line string) InterfaceStatus {
	if strings.Contains(strings.ToUpper(line), "UP") {
		return InterfaceStatusUp
	}
	return InterfaceStatusDown
}

func (tde *TopologyDiscoveryEngine) parseInterfaceMTU(line string) int {
	parts := strings.Fields(line)
	for i, part := range parts {
		if part == "mtu" && i+1 < len(parts) {
			if mtu, err := strconv.Atoi(parts[i+1]); err == nil {
				return mtu
			}
		}
	}
	return 1500 // Default MTU
}

func (tde *TopologyDiscoveryEngine) extractMACAddress(output string) string {
	lines := strings.Split(output, "\n")
	for _, line := range lines {
		if strings.Contains(line, "link/ether") {
			parts := strings.Fields(line)
			for i, part := range parts {
				if part == "link/ether" && i+1 < len(parts) {
					return parts[i+1]
				}
			}
		}
	}
	return ""
}

func (tde *TopologyDiscoveryEngine) extractIPAddresses(output string) []string {
	var addresses []string
	lines := strings.Split(output, "\n")
	
	for _, line := range lines {
		if strings.Contains(line, "inet ") {
			parts := strings.Fields(line)
			for i, part := range parts {
				if part == "inet" && i+1 < len(parts) {
					if cidr := parts[i+1]; strings.Contains(cidr, "/") {
						if ip, _, err := net.ParseCIDR(cidr); err == nil {
							addresses = append(addresses, ip.String())
						}
					}
				}
			}
		}
	}
	
	return addresses
}

func (tde *TopologyDiscoveryEngine) containsString(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}

// Additional methods for incremental discovery, link state checking, etc.

func (tde *TopologyDiscoveryEngine) performIncrementalDiscovery(ctx context.Context) error {
	// Implementation for incremental discovery
	log.Println("Performing incremental topology discovery")
	return nil
}

func (tde *TopologyDiscoveryEngine) performHostDiscovery(ctx context.Context) error {
	// Implementation for host discovery
	log.Println("Performing host discovery")
	return nil
}

func (tde *TopologyDiscoveryEngine) checkAllLinkStates() {
	// Implementation for checking all link states
	log.Println("Checking all link states")
}

func (tde *TopologyDiscoveryEngine) updateTopologyMetadata() {
	tde.topologyMutex.Lock()
	defer tde.topologyMutex.Unlock()
	
	tde.topology.Metadata = TopologyMetadata{
		NodeCount:        len(tde.topology.Nodes),
		LinkCount:        len(tde.topology.Links),
		SubnetCount:      len(tde.topology.Subnets),
		DiscoveryMethods: tde.getEnabledDiscoveryMethods(),
		Health: TopologyHealth{
			OverallScore: 100.0,
			NodeHealth:   100.0,
			LinkHealth:   100.0,
			Issues:       []HealthIssue{},
			LastAssessed: time.Now(),
		},
		Statistics: TopologyStatistics{
			LastSuccessfulRun: time.Now(),
		},
	}
}

func (tde *TopologyDiscoveryEngine) getEnabledDiscoveryMethods() []string {
	var methods []string
	
	if tde.config.EnableLLDPDiscovery {
		methods = append(methods, "LLDP")
	}
	if tde.config.EnableARPDiscovery {
		methods = append(methods, "ARP")
	}
	if tde.config.EnableBGPDiscovery {
		methods = append(methods, "BGP")
	}
	if tde.config.EnableOSPFDiscovery {
		methods = append(methods, "OSPF")
	}
	if tde.config.EnableSNMPDiscovery {
		methods = append(methods, "SNMP")
	}
	
	return methods
}

func (tde *TopologyDiscoveryEngine) taskManagementLoop() {
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()
	
	for {
		select {
		case <-tde.ctx.Done():
			return
		case <-ticker.C:
			tde.cleanupCompletedTasks()
		}
	}
}

func (tde *TopologyDiscoveryEngine) cleanupCompletedTasks() {
	tde.tasksMutex.Lock()
	defer tde.tasksMutex.Unlock()
	
	cutoff := time.Now().Add(-1 * time.Hour) // Keep tasks for 1 hour
	for taskID, task := range tde.discoveryTasks {
		if task.Status == TaskStatusCompleted || task.Status == TaskStatusFailed {
			if task.CompletedAt.Before(cutoff) {
				delete(tde.discoveryTasks, taskID)
			}
		}
	}
}

// Public API methods

// GetTopology returns the current network topology
func (tde *TopologyDiscoveryEngine) GetTopology() *NetworkTopology {
	tde.topologyMutex.RLock()
	defer tde.topologyMutex.RUnlock()
	
	// Return a copy to avoid race conditions
	topology := *tde.topology
	topology.Nodes = make(map[string]*NetworkNode)
	topology.Links = make(map[string]*NetworkLink)
	topology.Subnets = make(map[string]*NetworkSubnet)
	
	for id, node := range tde.topology.Nodes {
		nodeCopy := *node
		topology.Nodes[id] = &nodeCopy
	}
	
	for id, link := range tde.topology.Links {
		linkCopy := *link
		topology.Links[id] = &linkCopy
	}
	
	for id, subnet := range tde.topology.Subnets {
		subnetCopy := *subnet
		topology.Subnets[id] = &subnetCopy
	}
	
	return &topology
}

// GetNode returns a specific network node
func (tde *TopologyDiscoveryEngine) GetNode(nodeID string) (*NetworkNode, error) {
	tde.nodesMutex.RLock()
	defer tde.nodesMutex.RUnlock()
	
	node, exists := tde.nodes[nodeID]
	if !exists {
		return nil, fmt.Errorf("node %s not found", nodeID)
	}
	
	// Return a copy
	nodeCopy := *node
	return &nodeCopy, nil
}

// GetNodes returns all network nodes
func (tde *TopologyDiscoveryEngine) GetNodes() []*NetworkNode {
	tde.nodesMutex.RLock()
	defer tde.nodesMutex.RUnlock()
	
	nodes := make([]*NetworkNode, 0, len(tde.nodes))
	for _, node := range tde.nodes {
		nodeCopy := *node
		nodes = append(nodes, &nodeCopy)
	}
	
	// Sort by name for consistent ordering
	sort.Slice(nodes, func(i, j int) bool {
		return nodes[i].Name < nodes[j].Name
	})
	
	return nodes
}

// GetLinks returns all network links
func (tde *TopologyDiscoveryEngine) GetLinks() []*NetworkLink {
	tde.linksMutex.RLock()
	defer tde.linksMutex.RUnlock()
	
	links := make([]*NetworkLink, 0, len(tde.links))
	for _, link := range tde.links {
		linkCopy := *link
		links = append(links, &linkCopy)
	}
	
	return links
}

// GetSubnets returns all discovered subnets
func (tde *TopologyDiscoveryEngine) GetSubnets() []*NetworkSubnet {
	tde.subnetsMutex.RLock()
	defer tde.subnetsMutex.RUnlock()
	
	subnets := make([]*NetworkSubnet, 0, len(tde.subnets))
	for _, subnet := range tde.subnets {
		subnetCopy := *subnet
		subnets = append(subnets, &subnetCopy)
	}
	
	return subnets
}

// TriggerDiscovery triggers an immediate discovery of the specified type
func (tde *TopologyDiscoveryEngine) TriggerDiscovery(discoveryType DiscoveryTaskType) (*DiscoveryTask, error) {
	task := &DiscoveryTask{
		ID:        uuid.New().String(),
		Type:      discoveryType,
		Target:    "manual",
		Status:    TaskStatusPending,
		StartedAt: time.Now(),
	}
	
	tde.tasksMutex.Lock()
	tde.discoveryTasks[task.ID] = task
	tde.tasksMutex.Unlock()
	
	// Execute discovery based on type
	go func() {
		task.Status = TaskStatusRunning
		
		var err error
		switch discoveryType {
		case TaskTypeFullDiscovery:
			err = tde.performFullDiscovery(context.Background())
		case TaskTypeIncrementalDiscovery:
			err = tde.performIncrementalDiscovery(context.Background())
		case TaskTypeHostDiscovery:
			err = tde.performHostDiscovery(context.Background())
		default:
			err = fmt.Errorf("unsupported discovery type: %s", discoveryType)
		}
		
		if err != nil {
			task.Status = TaskStatusFailed
			task.Error = err.Error()
		} else {
			task.Status = TaskStatusCompleted
			task.Progress = 100.0
		}
		
		task.CompletedAt = time.Now()
	}()
	
	return task, nil
}

// GetDiscoveryTasks returns all discovery tasks
func (tde *TopologyDiscoveryEngine) GetDiscoveryTasks() []*DiscoveryTask {
	tde.tasksMutex.RLock()
	defer tde.tasksMutex.RUnlock()
	
	tasks := make([]*DiscoveryTask, 0, len(tde.discoveryTasks))
	for _, task := range tde.discoveryTasks {
		taskCopy := *task
		tasks = append(tasks, &taskCopy)
	}
	
	// Sort by start time (newest first)
	sort.Slice(tasks, func(i, j int) bool {
		return tasks[i].StartedAt.After(tasks[j].StartedAt)
	})
	
	return tasks
}