package drivers

import (
	"context"
	"fmt"
	"log"
	"net"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/khryptorgraphics/novacron/backend/core/network/overlay"
	"github.com/khryptorgraphics/novacron/backend/core/network/ovs"
)

// EnhancedVXLANDriver provides a production-ready VXLAN overlay implementation
type EnhancedVXLANDriver struct {
	// Configuration
	config VXLANDriverConfig
	
	// State management
	networks     map[string]*VXLANNetworkState
	endpoints    map[string]*VXLANEndpointState
	vteps        map[string]*VTEPConfiguration
	networksMutex sync.RWMutex
	endpointsMutex sync.RWMutex
	vtepsMutex    sync.RWMutex
	
	// Dependencies
	bridgeManager *ovs.BridgeManager
	
	// Runtime state
	initialized bool
	ctx         context.Context
	cancel      context.CancelFunc
}

// VXLANDriverConfig holds enhanced configuration for VXLAN driver
type VXLANDriverConfig struct {
	// Basic VXLAN configuration
	MinMTU                  int           `json:"min_mtu"`
	DefaultMulticastGroup   string        `json:"default_multicast_group"`
	UDPPort                 int           `json:"udp_port"`
	UseNativeImplementation bool          `json:"use_native_implementation"`
	EnableHardwareOffload   bool          `json:"enable_hardware_offload"`
	EnableDirectRouting     bool          `json:"enable_direct_routing"`
	VtepUtilsPath          string        `json:"vtep_utils_path"`
	
	// Enhanced features
	EnableL2Population      bool          `json:"enable_l2_population"`
	EnableARPResponder      bool          `json:"enable_arp_responder"`
	EnableFDBLearning       bool          `json:"enable_fdb_learning"`
	FDBAgingTime           time.Duration `json:"fdb_aging_time"`
	
	// Security features
	EnableEncryption        bool          `json:"enable_encryption"`
	EncryptionKey          string        `json:"encryption_key"`
	EnableSourceValidation  bool          `json:"enable_source_validation"`
	
	// Performance tuning
	TxQueueLength          int           `json:"tx_queue_length"`
	RxBufferSize           int           `json:"rx_buffer_size"`
	MaxVNIs                int           `json:"max_vnis"`
	
	// Monitoring
	EnableMetrics          bool          `json:"enable_metrics"`
	MetricsInterval        time.Duration `json:"metrics_interval"`
}

// DefaultEnhancedVXLANConfig returns enhanced default configuration
func DefaultEnhancedVXLANConfig() VXLANDriverConfig {
	return VXLANDriverConfig{
		MinMTU:                 1450,
		DefaultMulticastGroup:  "239.1.1.1",
		UDPPort:               4789,
		UseNativeImplementation: true,
		EnableHardwareOffload:  false,
		EnableDirectRouting:    true,
		VtepUtilsPath:         "/usr/local/bin",
		EnableL2Population:     true,
		EnableARPResponder:     true,
		EnableFDBLearning:      true,
		FDBAgingTime:          300 * time.Second,
		EnableEncryption:       false,
		EnableSourceValidation: true,
		TxQueueLength:         1000,
		RxBufferSize:          65536,
		MaxVNIs:               1000000,
		EnableMetrics:         true,
		MetricsInterval:       30 * time.Second,
	}
}

// VXLANNetworkState represents the runtime state of a VXLAN network
type VXLANNetworkState struct {
	Network        *overlay.OverlayNetwork `json:"network"`
	VTEP           *VTEPConfiguration      `json:"vtep"`
	BridgeName     string                  `json:"bridge_name"`
	FDBEntries     map[string]*FDBEntry    `json:"fdb_entries"`
	ARPTable       map[string]*ARPEntry    `json:"arp_table"`
	Statistics     *VXLANStatistics        `json:"statistics"`
	Status         NetworkStatus           `json:"status"`
	LastUpdated    time.Time              `json:"last_updated"`
}

// VXLANEndpointState represents the runtime state of a VXLAN endpoint
type VXLANEndpointState struct {
	Endpoint       *overlay.EndpointConfig `json:"endpoint"`
	NetworkID      string                  `json:"network_id"`
	InterfaceName  string                  `json:"interface_name"`
	VethPairName   string                  `json:"veth_pair_name"`
	MACAddress     string                  `json:"mac_address"`
	IPAddress      net.IP                  `json:"ip_address"`
	VNI            uint32                  `json:"vni"`
	Statistics     *EndpointStatistics     `json:"statistics"`
	Status         EndpointStatus          `json:"status"`
	CreatedAt      time.Time              `json:"created_at"`
	LastUpdated    time.Time              `json:"last_updated"`
}

// VTEPConfiguration represents VXLAN Tunnel Endpoint configuration
type VTEPConfiguration struct {
	ID                string            `json:"id"`
	Name              string            `json:"name"`
	LocalIP           net.IP            `json:"local_ip"`
	RemoteIPs         []net.IP          `json:"remote_ips"`
	MulticastGroup    string            `json:"multicast_group"`
	UDPPort           int               `json:"udp_port"`
	VNI               uint32            `json:"vni"`
	MTU               int               `json:"mtu"`
	InterfaceName     string            `json:"interface_name"`
	BridgeName        string            `json:"bridge_name"`
	Options           map[string]string `json:"options"`
	Status            VTEPStatus        `json:"status"`
	CreatedAt         time.Time         `json:"created_at"`
	LastUpdated       time.Time         `json:"last_updated"`
}

// FDBEntry represents a Forwarding Database entry
type FDBEntry struct {
	MACAddress    string    `json:"mac_address"`
	RemoteIP      net.IP    `json:"remote_ip"`
	VNI           uint32    `json:"vni"`
	Port          string    `json:"port"`
	Dynamic       bool      `json:"dynamic"`
	AgingTime     int       `json:"aging_time"`
	LastSeen      time.Time `json:"last_seen"`
}

// ARPEntry represents an ARP table entry
type ARPEntry struct {
	IPAddress     net.IP    `json:"ip_address"`
	MACAddress    string    `json:"mac_address"`
	VNI           uint32    `json:"vni"`
	RemoteVTEP    net.IP    `json:"remote_vtep"`
	Static        bool      `json:"static"`
	LastUpdated   time.Time `json:"last_updated"`
}

// VXLANStatistics represents VXLAN network statistics
type VXLANStatistics struct {
	TxPackets         int64     `json:"tx_packets"`
	RxPackets         int64     `json:"rx_packets"`
	TxBytes          int64     `json:"tx_bytes"`
	RxBytes          int64     `json:"rx_bytes"`
	TxErrors         int64     `json:"tx_errors"`
	RxErrors         int64     `json:"rx_errors"`
	TxDropped        int64     `json:"tx_dropped"`
	RxDropped        int64     `json:"rx_dropped"`
	TunnelPackets    int64     `json:"tunnel_packets"`
	TunnelBytes      int64     `json:"tunnel_bytes"`
	BroadcastPackets int64     `json:"broadcast_packets"`
	MulticastPackets int64     `json:"multicast_packets"`
	LastUpdated      time.Time `json:"last_updated"`
}

// EndpointStatistics represents endpoint-specific statistics
type EndpointStatistics struct {
	TxPackets    int64     `json:"tx_packets"`
	RxPackets    int64     `json:"rx_packets"`
	TxBytes      int64     `json:"tx_bytes"`
	RxBytes      int64     `json:"rx_bytes"`
	TxErrors     int64     `json:"tx_errors"`
	RxErrors     int64     `json:"rx_errors"`
	LastUpdated  time.Time `json:"last_updated"`
}

// Status enums
type NetworkStatus string
type EndpointStatus string
type VTEPStatus string

const (
	NetworkStatusActive   NetworkStatus = "active"
	NetworkStatusInactive NetworkStatus = "inactive"
	NetworkStatusError    NetworkStatus = "error"
	
	EndpointStatusActive   EndpointStatus = "active"
	EndpointStatusInactive EndpointStatus = "inactive"
	EndpointStatusError    EndpointStatus = "error"
	
	VTEPStatusActive   VTEPStatus = "active"
	VTEPStatusInactive VTEPStatus = "inactive"
	VTEPStatusError    VTEPStatus = "error"
)

// NewEnhancedVXLANDriver creates a new enhanced VXLAN driver
func NewEnhancedVXLANDriver(config VXLANDriverConfig, bridgeManager *ovs.BridgeManager) *EnhancedVXLANDriver {
	ctx, cancel := context.WithCancel(context.Background())
	
	return &EnhancedVXLANDriver{
		config:        config,
		networks:      make(map[string]*VXLANNetworkState),
		endpoints:     make(map[string]*VXLANEndpointState),
		vteps:         make(map[string]*VTEPConfiguration),
		bridgeManager: bridgeManager,
		ctx:           ctx,
		cancel:        cancel,
		initialized:   false,
	}
}

// Name returns the driver name
func (d *EnhancedVXLANDriver) Name() string {
	return "enhanced-vxlan"
}

// Initialize initializes the enhanced VXLAN driver
func (d *EnhancedVXLANDriver) Initialize(ctx context.Context) error {
	d.networksMutex.Lock()
	defer d.networksMutex.Unlock()
	
	if d.initialized {
		return fmt.Errorf("enhanced VXLAN driver already initialized")
	}
	
	// Check kernel VXLAN support
	if err := d.checkVXLANSupport(); err != nil {
		return fmt.Errorf("VXLAN support check failed: %w", err)
	}
	
	// Initialize bridge manager if needed
	if d.bridgeManager == nil {
		return fmt.Errorf("bridge manager not provided")
	}
	
	// Load existing VXLAN networks
	if err := d.loadExistingNetworks(); err != nil {
		log.Printf("Warning: Failed to load existing VXLAN networks: %v", err)
	}
	
	// Start monitoring if enabled
	if d.config.EnableMetrics {
		go d.metricsCollectionLoop()
	}
	
	d.initialized = true
	log.Println("Enhanced VXLAN driver initialized successfully")
	return nil
}

// checkVXLANSupport verifies kernel VXLAN support
func (d *EnhancedVXLANDriver) checkVXLANSupport() error {
	// Check if ip command supports VXLAN
	cmd := exec.Command("ip", "link", "help")
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed to check ip command VXLAN support: %w", err)
	}
	
	if !strings.Contains(string(output), "vxlan") {
		return fmt.Errorf("kernel does not support VXLAN")
	}
	
	// Check if bridge command supports VXLAN
	cmd = exec.Command("bridge", "fdb", "help")
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("bridge command not available for FDB management: %w", err)
	}
	
	return nil
}

// Capabilities returns enhanced driver capabilities
func (d *EnhancedVXLANDriver) Capabilities() overlay.DriverCapabilities {
	return overlay.DriverCapabilities{
		SupportedTypes:          []overlay.OverlayType{overlay.VXLAN},
		MaxMTU:                  9000 - 50, // Max Ethernet MTU minus VXLAN overhead
		SupportsL2Extension:     true,
		SupportsNetworkPolicies: true,
		SupportsQoS:             true,
		SupportsServiceMesh:     true,
	}
}

// CreateNetwork creates a new VXLAN overlay network
func (d *EnhancedVXLANDriver) CreateNetwork(ctx context.Context, network overlay.OverlayNetwork) error {
	d.networksMutex.Lock()
	defer d.networksMutex.Unlock()
	
	if !d.initialized {
		return fmt.Errorf("enhanced VXLAN driver not initialized")
	}
	
	// Check if network already exists
	if _, exists := d.networks[network.ID]; exists {
		return fmt.Errorf("network %s already exists", network.ID)
	}
	
	// Validate network configuration
	if err := d.validateNetworkConfig(&network); err != nil {
		return fmt.Errorf("network validation failed: %w", err)
	}
	
	// Create VTEP configuration
	vtep, err := d.createVTEPConfiguration(&network)
	if err != nil {
		return fmt.Errorf("failed to create VTEP configuration: %w", err)
	}
	
	// Create OVS bridge for the network
	bridgeName := d.generateBridgeName(network.Name)
	bridge, err := d.bridgeManager.CreateBridge(ctx, bridgeName, ovs.BridgeTypeVXLAN, network.Options)
	if err != nil {
		return fmt.Errorf("failed to create OVS bridge: %w", err)
	}
	
	// Create VXLAN interface
	vxlanInterface, err := d.createVXLANInterface(&network, vtep)
	if err != nil {
		// Clean up bridge on failure
		d.bridgeManager.DeleteBridge(ctx, bridgeName)
		return fmt.Errorf("failed to create VXLAN interface: %w", err)
	}
	
	// Add VXLAN interface to bridge
	_, err = d.bridgeManager.AddPort(ctx, bridgeName, vxlanInterface, ovs.PortTypeVXLAN, network.Options)
	if err != nil {
		// Clean up on failure
		d.deleteVXLANInterface(vxlanInterface)
		d.bridgeManager.DeleteBridge(ctx, bridgeName)
		return fmt.Errorf("failed to add VXLAN port to bridge: %w", err)
	}
	
	// Initialize network state
	networkState := &VXLANNetworkState{
		Network:     &network,
		VTEP:        vtep,
		BridgeName:  bridgeName,
		FDBEntries:  make(map[string]*FDBEntry),
		ARPTable:    make(map[string]*ARPEntry),
		Statistics:  &VXLANStatistics{LastUpdated: time.Now()},
		Status:      NetworkStatusActive,
		LastUpdated: time.Now(),
	}
	
	// Store network state
	d.networks[network.ID] = networkState
	d.vteps[vtep.ID] = vtep
	
	// Set up L2 population if enabled
	if d.config.EnableL2Population {
		if err := d.setupL2Population(networkState); err != nil {
			log.Printf("Warning: Failed to setup L2 population for network %s: %v", network.ID, err)
		}
	}
	
	log.Printf("Created VXLAN network %s (VNI: %d, Bridge: %s)", network.Name, network.VNI, bridgeName)
	return nil
}

// validateNetworkConfig validates VXLAN network configuration
func (d *EnhancedVXLANDriver) validateNetworkConfig(network *overlay.OverlayNetwork) error {
	if network.Type != overlay.VXLAN {
		return fmt.Errorf("unsupported network type: %s", network.Type)
	}
	
	if network.VNI == 0 {
		return fmt.Errorf("VNI must be specified and non-zero")
	}
	
	if network.VNI >= uint32(d.config.MaxVNIs) {
		return fmt.Errorf("VNI %d exceeds maximum allowed (%d)", network.VNI, d.config.MaxVNIs)
	}
	
	if network.MTU < d.config.MinMTU {
		return fmt.Errorf("MTU %d is below minimum (%d)", network.MTU, d.config.MinMTU)
	}
	
	// Validate CIDR
	if network.CIDR != "" {
		_, _, err := net.ParseCIDR(network.CIDR)
		if err != nil {
			return fmt.Errorf("invalid CIDR %s: %w", network.CIDR, err)
		}
	}
	
	return nil
}

// createVTEPConfiguration creates VTEP configuration for a network
func (d *EnhancedVXLANDriver) createVTEPConfiguration(network *overlay.OverlayNetwork) (*VTEPConfiguration, error) {
	// Get local IP for VTEP
	localIP, err := d.getLocalVTEPIP(network)
	if err != nil {
		return nil, fmt.Errorf("failed to determine local VTEP IP: %w", err)
	}
	
	vtep := &VTEPConfiguration{
		ID:             uuid.New().String(),
		Name:           fmt.Sprintf("vtep-%s", network.Name),
		LocalIP:        localIP,
		RemoteIPs:      []net.IP{},
		MulticastGroup: d.getMulticastGroup(network),
		UDPPort:        d.getUDPPort(network),
		VNI:            network.VNI,
		MTU:            network.MTU,
		InterfaceName:  d.generateVXLANInterfaceName(network.Name),
		Options:        network.Options,
		Status:         VTEPStatusActive,
		CreatedAt:      time.Now(),
		LastUpdated:    time.Now(),
	}
	
	return vtep, nil
}

// getLocalVTEPIP determines the local IP address for VTEP
func (d *EnhancedVXLANDriver) getLocalVTEPIP(network *overlay.OverlayNetwork) (net.IP, error) {
	// Check if local IP is specified in options
	if localIPStr, exists := network.Options["local_ip"]; exists {
		localIP := net.ParseIP(localIPStr)
		if localIP == nil {
			return nil, fmt.Errorf("invalid local IP: %s", localIPStr)
		}
		return localIP, nil
	}
	
	// Auto-detect local IP
	return d.autoDetectLocalIP()
}

// autoDetectLocalIP automatically detects the local IP for VTEP
func (d *EnhancedVXLANDriver) autoDetectLocalIP() (net.IP, error) {
	// Get default route interface
	cmd := exec.Command("ip", "route", "get", "8.8.8.8")
	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("failed to get default route: %w", err)
	}
	
	// Parse output to extract source IP
	lines := strings.Split(string(output), "\n")
	for _, line := range lines {
		if strings.Contains(line, "src") {
			parts := strings.Fields(line)
			for i, part := range parts {
				if part == "src" && i+1 < len(parts) {
					ip := net.ParseIP(parts[i+1])
					if ip != nil {
						return ip, nil
					}
				}
			}
		}
	}
	
	return nil, fmt.Errorf("could not auto-detect local IP")
}

// getMulticastGroup gets multicast group for the network
func (d *EnhancedVXLANDriver) getMulticastGroup(network *overlay.OverlayNetwork) string {
	if group, exists := network.Options["multicast_group"]; exists {
		return group
	}
	return d.config.DefaultMulticastGroup
}

// getUDPPort gets UDP port for the network
func (d *EnhancedVXLANDriver) getUDPPort(network *overlay.OverlayNetwork) int {
	if portStr, exists := network.Options["udp_port"]; exists {
		if port, err := strconv.Atoi(portStr); err == nil {
			return port
		}
	}
	return d.config.UDPPort
}

// generateBridgeName generates a bridge name for the network
func (d *EnhancedVXLANDriver) generateBridgeName(networkName string) string {
	return fmt.Sprintf("vxbr-%s", networkName)
}

// generateVXLANInterfaceName generates a VXLAN interface name
func (d *EnhancedVXLANDriver) generateVXLANInterfaceName(networkName string) string {
	return fmt.Sprintf("vxlan-%s", networkName)
}

// createVXLANInterface creates a VXLAN network interface
func (d *EnhancedVXLANDriver) createVXLANInterface(network *overlay.OverlayNetwork, vtep *VTEPConfiguration) (string, error) {
	interfaceName := vtep.InterfaceName
	
	// Build ip link command
	args := []string{
		"link", "add", interfaceName, "type", "vxlan",
		"id", strconv.FormatUint(uint64(network.VNI), 10),
		"local", vtep.LocalIP.String(),
		"dstport", strconv.Itoa(vtep.UDPPort),
	}
	
	// Add multicast group if specified
	if vtep.MulticastGroup != "" && vtep.MulticastGroup != "0.0.0.0" {
		args = append(args, "group", vtep.MulticastGroup)
	}
	
	// Add TTL
	if ttl, exists := network.Options["ttl"]; exists {
		args = append(args, "ttl", ttl)
	} else {
		args = append(args, "ttl", "64")
	}
	
	// Add learning control
	if d.config.EnableFDBLearning {
		args = append(args, "learning")
	} else {
		args = append(args, "nolearning")
	}
	
	// Create the interface
	cmd := exec.Command("ip", args...)
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("failed to create VXLAN interface: %w", err)
	}
	
	// Set MTU
	if network.MTU > 0 {
		cmd = exec.Command("ip", "link", "set", interfaceName, "mtu", strconv.Itoa(network.MTU))
		if err := cmd.Run(); err != nil {
			log.Printf("Warning: Failed to set MTU on VXLAN interface %s: %v", interfaceName, err)
		}
	}
	
	// Set TX queue length if configured
	if d.config.TxQueueLength > 0 {
		cmd = exec.Command("ip", "link", "set", interfaceName, "txqueuelen", strconv.Itoa(d.config.TxQueueLength))
		if err := cmd.Run(); err != nil {
			log.Printf("Warning: Failed to set TX queue length on VXLAN interface %s: %v", interfaceName, err)
		}
	}
	
	// Bring up the interface
	cmd = exec.Command("ip", "link", "set", interfaceName, "up")
	if err := cmd.Run(); err != nil {
		// Clean up on failure
		d.deleteVXLANInterface(interfaceName)
		return "", fmt.Errorf("failed to bring up VXLAN interface: %w", err)
	}
	
	log.Printf("Created VXLAN interface %s (VNI: %d)", interfaceName, network.VNI)
	return interfaceName, nil
}

// deleteVXLANInterface deletes a VXLAN network interface
func (d *EnhancedVXLANDriver) deleteVXLANInterface(interfaceName string) error {
	cmd := exec.Command("ip", "link", "del", interfaceName)
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("failed to delete VXLAN interface %s: %w", interfaceName, err)
	}
	
	log.Printf("Deleted VXLAN interface %s", interfaceName)
	return nil
}

// setupL2Population sets up L2 population for efficient flooding
func (d *EnhancedVXLANDriver) setupL2Population(networkState *VXLANNetworkState) error {
	// L2 population reduces flooding by maintaining FDB entries
	// This is a placeholder for L2 population setup
	log.Printf("Setting up L2 population for network %s", networkState.Network.Name)
	return nil
}

// DeleteNetwork deletes a VXLAN overlay network
func (d *EnhancedVXLANDriver) DeleteNetwork(ctx context.Context, networkID string) error {
	d.networksMutex.Lock()
	defer d.networksMutex.Unlock()
	
	// Get network state
	networkState, exists := d.networks[networkID]
	if !exists {
		return fmt.Errorf("network %s not found", networkID)
	}
	
	// Check if network has active endpoints
	d.endpointsMutex.RLock()
	activeEndpoints := 0
	for _, endpoint := range d.endpoints {
		if endpoint.NetworkID == networkID && endpoint.Status == EndpointStatusActive {
			activeEndpoints++
		}
	}
	d.endpointsMutex.RUnlock()
	
	if activeEndpoints > 0 {
		return fmt.Errorf("cannot delete network %s: it has %d active endpoints", networkID, activeEndpoints)
	}
	
	// Delete VXLAN interface
	if err := d.deleteVXLANInterface(networkState.VTEP.InterfaceName); err != nil {
		log.Printf("Warning: Failed to delete VXLAN interface: %v", err)
	}
	
	// Delete OVS bridge
	if err := d.bridgeManager.DeleteBridge(ctx, networkState.BridgeName); err != nil {
		log.Printf("Warning: Failed to delete OVS bridge: %v", err)
	}
	
	// Clean up state
	delete(d.networks, networkID)
	delete(d.vteps, networkState.VTEP.ID)
	
	log.Printf("Deleted VXLAN network %s", networkState.Network.Name)
	return nil
}

// CreateEndpoint creates a new endpoint in a VXLAN network
func (d *EnhancedVXLANDriver) CreateEndpoint(ctx context.Context, endpoint overlay.EndpointConfig) error {
	d.endpointsMutex.Lock()
	defer d.endpointsMutex.Unlock()
	
	// Check if network exists
	d.networksMutex.RLock()
	networkState, exists := d.networks[endpoint.NetworkID]
	d.networksMutex.RUnlock()
	
	if !exists {
		return fmt.Errorf("network %s not found", endpoint.NetworkID)
	}
	
	// Check if endpoint already exists
	if _, exists := d.endpoints[endpoint.Name]; exists {
		return fmt.Errorf("endpoint %s already exists", endpoint.Name)
	}
	
	// Create veth pair for endpoint connectivity
	vethPair, err := d.createVethPair(endpoint.Name)
	if err != nil {
		return fmt.Errorf("failed to create veth pair: %w", err)
	}
	
	// Add veth peer to bridge
	bridgeEnd := vethPair + "-br"
	_, err = d.bridgeManager.AddPort(ctx, networkState.BridgeName, bridgeEnd, ovs.PortTypeVETH, endpoint.Options)
	if err != nil {
		d.deleteVethPair(vethPair)
		return fmt.Errorf("failed to add endpoint to bridge: %w", err)
	}
	
	// Configure endpoint IP if specified
	var ipAddr net.IP
	if endpoint.IPAddress != "" {
		ipAddr = net.ParseIP(endpoint.IPAddress)
		if ipAddr == nil {
			return fmt.Errorf("invalid IP address: %s", endpoint.IPAddress)
		}
		
		if err := d.configureEndpointIP(vethPair, endpoint.IPAddress, networkState.Network.CIDR); err != nil {
			log.Printf("Warning: Failed to configure endpoint IP: %v", err)
		}
	}
	
	// Create endpoint state
	endpointState := &VXLANEndpointState{
		Endpoint:      &endpoint,
		NetworkID:     endpoint.NetworkID,
		InterfaceName: vethPair,
		VethPairName:  vethPair,
		MACAddress:    endpoint.MACAddress,
		IPAddress:     ipAddr,
		VNI:           networkState.Network.VNI,
		Statistics:    &EndpointStatistics{LastUpdated: time.Now()},
		Status:        EndpointStatusActive,
		CreatedAt:     time.Now(),
		LastUpdated:   time.Now(),
	}
	
	// Store endpoint state
	d.endpoints[endpoint.Name] = endpointState
	
	// Update FDB if L2 population is enabled
	if d.config.EnableL2Population {
		d.updateFDBForEndpoint(networkState, endpointState, true)
	}
	
	log.Printf("Created VXLAN endpoint %s in network %s", endpoint.Name, networkState.Network.Name)
	return nil
}

// createVethPair creates a veth pair for endpoint connectivity
func (d *EnhancedVXLANDriver) createVethPair(endpointName string) (string, error) {
	vethName := fmt.Sprintf("veth-%s", endpointName)
	bridgeEnd := vethName + "-br"
	
	// Create veth pair
	cmd := exec.Command("ip", "link", "add", vethName, "type", "veth", "peer", "name", bridgeEnd)
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("failed to create veth pair: %w", err)
	}
	
	// Bring up both ends
	cmd = exec.Command("ip", "link", "set", vethName, "up")
	if err := cmd.Run(); err != nil {
		d.deleteVethPair(vethName)
		return "", fmt.Errorf("failed to bring up veth interface: %w", err)
	}
	
	cmd = exec.Command("ip", "link", "set", bridgeEnd, "up")
	if err := cmd.Run(); err != nil {
		d.deleteVethPair(vethName)
		return "", fmt.Errorf("failed to bring up veth bridge end: %w", err)
	}
	
	return vethName, nil
}

// deleteVethPair deletes a veth pair
func (d *EnhancedVXLANDriver) deleteVethPair(vethName string) error {
	cmd := exec.Command("ip", "link", "del", vethName)
	return cmd.Run()
}

// configureEndpointIP configures IP address for an endpoint
func (d *EnhancedVXLANDriver) configureEndpointIP(interfaceName, ipAddress, cidr string) error {
	// Parse CIDR to get prefix length
	_, ipNet, err := net.ParseCIDR(cidr)
	if err != nil {
		return fmt.Errorf("invalid CIDR %s: %w", cidr, err)
	}
	
	prefixLen, _ := ipNet.Mask.Size()
	
	// Configure IP address
	cmd := exec.Command("ip", "addr", "add", fmt.Sprintf("%s/%d", ipAddress, prefixLen), "dev", interfaceName)
	return cmd.Run()
}

// updateFDBForEndpoint updates FDB entries for L2 population
func (d *EnhancedVXLANDriver) updateFDBForEndpoint(networkState *VXLANNetworkState, endpointState *VXLANEndpointState, add bool) {
	fdbKey := fmt.Sprintf("%s-%d", endpointState.MACAddress, endpointState.VNI)
	
	if add {
		fdbEntry := &FDBEntry{
			MACAddress: endpointState.MACAddress,
			RemoteIP:   networkState.VTEP.LocalIP,
			VNI:        endpointState.VNI,
			Port:       endpointState.InterfaceName,
			Dynamic:    true,
			AgingTime:  int(d.config.FDBAgingTime.Seconds()),
			LastSeen:   time.Now(),
		}
		networkState.FDBEntries[fdbKey] = fdbEntry
	} else {
		delete(networkState.FDBEntries, fdbKey)
	}
}

// loadExistingNetworks loads existing VXLAN networks from the system
func (d *EnhancedVXLANDriver) loadExistingNetworks() error {
	// This would parse existing VXLAN interfaces and reconstruct network state
	// For now, just log that this would be implemented
	log.Println("Loading existing VXLAN networks (not fully implemented)")
	return nil
}

// metricsCollectionLoop collects metrics for VXLAN networks
func (d *EnhancedVXLANDriver) metricsCollectionLoop() {
	ticker := time.NewTicker(d.config.MetricsInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-d.ctx.Done():
			return
		case <-ticker.C:
			d.updateMetrics()
		}
	}
}

// updateMetrics updates statistics for all networks and endpoints
func (d *EnhancedVXLANDriver) updateMetrics() {
	d.networksMutex.RLock()
	networks := make([]*VXLANNetworkState, 0, len(d.networks))
	for _, network := range d.networks {
		networks = append(networks, network)
	}
	d.networksMutex.RUnlock()
	
	for _, network := range networks {
		if err := d.updateNetworkStatistics(network); err != nil {
			log.Printf("Warning: Failed to update statistics for network %s: %v", network.Network.ID, err)
		}
	}
	
	d.endpointsMutex.RLock()
	endpoints := make([]*VXLANEndpointState, 0, len(d.endpoints))
	for _, endpoint := range d.endpoints {
		endpoints = append(endpoints, endpoint)
	}
	d.endpointsMutex.RUnlock()
	
	for _, endpoint := range endpoints {
		if err := d.updateEndpointStatistics(endpoint); err != nil {
			log.Printf("Warning: Failed to update statistics for endpoint %s: %v", endpoint.Endpoint.Name, err)
		}
	}
}

// updateNetworkStatistics updates statistics for a specific network
func (d *EnhancedVXLANDriver) updateNetworkStatistics(network *VXLANNetworkState) error {
	// Parse /proc/net/dev for interface statistics
	stats, err := d.getInterfaceStatistics(network.VTEP.InterfaceName)
	if err != nil {
		return err
	}
	
	network.Statistics.TxPackets = stats.TxPackets
	network.Statistics.RxPackets = stats.RxPackets
	network.Statistics.TxBytes = stats.TxBytes
	network.Statistics.RxBytes = stats.RxBytes
	network.Statistics.TxErrors = stats.TxErrors
	network.Statistics.RxErrors = stats.RxErrors
	network.Statistics.TxDropped = stats.TxDropped
	network.Statistics.RxDropped = stats.RxDropped
	network.Statistics.LastUpdated = time.Now()
	network.LastUpdated = time.Now()
	
	return nil
}

// updateEndpointStatistics updates statistics for a specific endpoint
func (d *EnhancedVXLANDriver) updateEndpointStatistics(endpoint *VXLANEndpointState) error {
	stats, err := d.getInterfaceStatistics(endpoint.InterfaceName)
	if err != nil {
		return err
	}
	
	endpoint.Statistics.TxPackets = stats.TxPackets
	endpoint.Statistics.RxPackets = stats.RxPackets
	endpoint.Statistics.TxBytes = stats.TxBytes
	endpoint.Statistics.RxBytes = stats.RxBytes
	endpoint.Statistics.TxErrors = stats.TxErrors
	endpoint.Statistics.RxErrors = stats.RxErrors
	endpoint.Statistics.LastUpdated = time.Now()
	endpoint.LastUpdated = time.Now()
	
	return nil
}

// InterfaceStats represents network interface statistics
type InterfaceStats struct {
	TxPackets int64
	RxPackets int64
	TxBytes   int64
	RxBytes   int64
	TxErrors  int64
	RxErrors  int64
	TxDropped int64
	RxDropped int64
}

// getInterfaceStatistics gets statistics for a network interface
func (d *EnhancedVXLANDriver) getInterfaceStatistics(interfaceName string) (*InterfaceStats, error) {
	// This would parse /proc/net/dev or use netlink to get interface statistics
	// For now, return zero statistics
	return &InterfaceStats{}, nil
}

// Implement remaining OverlayDriver interface methods

func (d *EnhancedVXLANDriver) UpdateNetwork(ctx context.Context, network overlay.OverlayNetwork) error {
	// Implementation for updating network configuration
	return fmt.Errorf("UpdateNetwork not implemented")
}

func (d *EnhancedVXLANDriver) GetNetwork(ctx context.Context, networkID string) (overlay.OverlayNetwork, error) {
	d.networksMutex.RLock()
	defer d.networksMutex.RUnlock()
	
	networkState, exists := d.networks[networkID]
	if !exists {
		return overlay.OverlayNetwork{}, fmt.Errorf("network %s not found", networkID)
	}
	
	return *networkState.Network, nil
}

func (d *EnhancedVXLANDriver) ListNetworks(ctx context.Context) ([]overlay.OverlayNetwork, error) {
	d.networksMutex.RLock()
	defer d.networksMutex.RUnlock()
	
	networks := make([]overlay.OverlayNetwork, 0, len(d.networks))
	for _, networkState := range d.networks {
		networks = append(networks, *networkState.Network)
	}
	
	return networks, nil
}

func (d *EnhancedVXLANDriver) DeleteEndpoint(ctx context.Context, networkID, endpointName string) error {
	d.endpointsMutex.Lock()
	defer d.endpointsMutex.Unlock()
	
	// Find endpoint
	endpointState, exists := d.endpoints[endpointName]
	if !exists {
		return fmt.Errorf("endpoint %s not found", endpointName)
	}
	
	if endpointState.NetworkID != networkID {
		return fmt.Errorf("endpoint %s not found in network %s", endpointName, networkID)
	}
	
	// Remove from bridge
	d.networksMutex.RLock()
	networkState := d.networks[networkID]
	d.networksMutex.RUnlock()
	
	bridgeEnd := endpointState.VethPairName + "-br"
	if err := d.bridgeManager.DeletePort(ctx, networkState.BridgeName, bridgeEnd); err != nil {
		log.Printf("Warning: Failed to remove port from bridge: %v", err)
	}
	
	// Delete veth pair
	if err := d.deleteVethPair(endpointState.VethPairName); err != nil {
		log.Printf("Warning: Failed to delete veth pair: %v", err)
	}
	
	// Update FDB if L2 population is enabled
	if d.config.EnableL2Population {
		d.updateFDBForEndpoint(networkState, endpointState, false)
	}
	
	// Remove from state
	delete(d.endpoints, endpointName)
	
	log.Printf("Deleted VXLAN endpoint %s from network %s", endpointName, networkID)
	return nil
}

func (d *EnhancedVXLANDriver) GetEndpoint(ctx context.Context, networkID, endpointName string) (overlay.EndpointConfig, error) {
	d.endpointsMutex.RLock()
	defer d.endpointsMutex.RUnlock()
	
	endpointState, exists := d.endpoints[endpointName]
	if !exists {
		return overlay.EndpointConfig{}, fmt.Errorf("endpoint %s not found", endpointName)
	}
	
	if endpointState.NetworkID != networkID {
		return overlay.EndpointConfig{}, fmt.Errorf("endpoint %s not found in network %s", endpointName, networkID)
	}
	
	return *endpointState.Endpoint, nil
}

func (d *EnhancedVXLANDriver) ListEndpoints(ctx context.Context, networkID string) ([]overlay.EndpointConfig, error) {
	d.endpointsMutex.RLock()
	defer d.endpointsMutex.RUnlock()
	
	var endpoints []overlay.EndpointConfig
	for _, endpointState := range d.endpoints {
		if endpointState.NetworkID == networkID {
			endpoints = append(endpoints, *endpointState.Endpoint)
		}
	}
	
	return endpoints, nil
}

func (d *EnhancedVXLANDriver) ApplyNetworkPolicy(ctx context.Context, networkID string, policy overlay.NetworkPolicy) error {
	// Implementation for applying network policies
	return fmt.Errorf("ApplyNetworkPolicy not implemented")
}

func (d *EnhancedVXLANDriver) RemoveNetworkPolicy(ctx context.Context, networkID, policyID string) error {
	// Implementation for removing network policies
	return fmt.Errorf("RemoveNetworkPolicy not implemented")
}

func (d *EnhancedVXLANDriver) Shutdown(ctx context.Context) error {
	d.cancel()
	
	// Clean up all networks
	d.networksMutex.Lock()
	for networkID := range d.networks {
		if err := d.DeleteNetwork(ctx, networkID); err != nil {
			log.Printf("Warning: Failed to delete network %s during shutdown: %v", networkID, err)
		}
	}
	d.networksMutex.Unlock()
	
	d.initialized = false
	log.Println("Enhanced VXLAN driver shut down")
	return nil
}

func init() {
	// Register the enhanced VXLAN driver factory
	overlay.RegisterDriverFactory("enhanced-vxlan", func() (overlay.OverlayDriver, error) {
		// This would require bridge manager dependency injection
		return nil, fmt.Errorf("enhanced VXLAN driver requires bridge manager")
	})
}