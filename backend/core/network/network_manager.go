package network

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"os/exec"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
)

// NetworkType defines the type of network
type NetworkType string

const (
	// NetworkTypeBridge is a bridge network (local connectivity)
	NetworkTypeBridge NetworkType = "bridge"
	
	// NetworkTypeOverlay is an overlay network (across nodes)
	NetworkTypeOverlay NetworkType = "overlay"
	
	// NetworkTypeMacvlan is a macvlan network (direct access to physical network)
	NetworkTypeMacvlan NetworkType = "macvlan"
)

// IPAMConfig represents IP Address Management configuration
type IPAMConfig struct {
	Subnet     string   `json:"subnet"`
	Gateway    string   `json:"gateway,omitempty"`
	IPRange    string   `json:"ip_range,omitempty"`
	AuxAddresses map[string]string `json:"aux_addresses,omitempty"`
	DNSServers []string `json:"dns_servers,omitempty"`
}

// NetworkSpec defines a network configuration
type NetworkSpec struct {
	Name        string      `json:"name"`
	Type        NetworkType `json:"type"`
	IPAM        IPAMConfig  `json:"ipam"`
	Internal    bool        `json:"internal"`
	EnableIPv6  bool        `json:"enable_ipv6"`
	Labels      map[string]string `json:"labels,omitempty"`
	Options     map[string]string `json:"options,omitempty"`
}

// Network represents a network in the system
type Network struct {
	ID          string      `json:"id"`
	Name        string      `json:"name"`
	Type        NetworkType `json:"type"`
	IPAM        IPAMConfig  `json:"ipam"`
	Internal    bool        `json:"internal"`
	EnableIPv6  bool        `json:"enable_ipv6"`
	Labels      map[string]string `json:"labels,omitempty"`
	Options     map[string]string `json:"options,omitempty"`
	CreatedAt   time.Time   `json:"created_at"`
	NodeID      string      `json:"node_id"`
	NetworkInfo NetworkInfo `json:"network_info"`
}

// NetworkInfo contains runtime information about a network
type NetworkInfo struct {
	Active      bool      `json:"active"`
	ConnectedVMs []string `json:"connected_vms"`
	LastUpdated time.Time `json:"last_updated"`
	// Additional metrics could be added here
}

// NetworkEventType represents network event types
type NetworkEventType string

const (
	// NetworkEventCreated is emitted when a network is created
	NetworkEventCreated NetworkEventType = "created"
	
	// NetworkEventDeleted is emitted when a network is deleted
	NetworkEventDeleted NetworkEventType = "deleted"
	
	// NetworkEventUpdated is emitted when a network is updated
	NetworkEventUpdated NetworkEventType = "updated"
	
	// NetworkEventError is emitted on network errors
	NetworkEventError NetworkEventType = "error"
)

// NetworkEvent represents an event related to networks
type NetworkEvent struct {
	Type      NetworkEventType `json:"type"`
	Network   Network         `json:"network"`
	Timestamp time.Time       `json:"timestamp"`
	NodeID    string          `json:"node_id"`
	Message   string          `json:"message,omitempty"`
}

// NetworkEventListener is a callback for network events
type NetworkEventListener func(event NetworkEvent)

// NetworkManagerConfig holds configuration for the network manager
type NetworkManagerConfig struct {
	DefaultNetworkType NetworkType `json:"default_network_type"`
	DefaultSubnet      string      `json:"default_subnet"`
	DefaultIPRange     string      `json:"default_ip_range"`
	DefaultGateway     string      `json:"default_gateway"`
	DNSServers         []string    `json:"dns_servers"`
	UpdateInterval     time.Duration `json:"update_interval"`
}

// DefaultNetworkManagerConfig returns a default configuration
func DefaultNetworkManagerConfig() NetworkManagerConfig {
	return NetworkManagerConfig{
		DefaultNetworkType: NetworkTypeBridge,
		DefaultSubnet:      "192.168.100.0/24",
		DefaultIPRange:     "192.168.100.10/24",
		DefaultGateway:     "192.168.100.1",
		DNSServers:         []string{"8.8.8.8", "8.8.4.4"},
		UpdateInterval:     30 * time.Second,
	}
}

// NetworkManager manages virtual networks
type NetworkManager struct {
	networks       map[string]*Network
	networksByName map[string]string // name -> id
	networksMutex  sync.RWMutex
	eventListeners []NetworkEventListener
	eventMutex     sync.RWMutex
	config         NetworkManagerConfig
	nodeID         string
	ctx            context.Context
	cancel         context.CancelFunc
}

// NewNetworkManager creates a new network manager
func NewNetworkManager(config NetworkManagerConfig, nodeID string) *NetworkManager {
	ctx, cancel := context.WithCancel(context.Background())
	
	manager := &NetworkManager{
		networks:       make(map[string]*Network),
		networksByName: make(map[string]string),
		config:         config,
		nodeID:         nodeID,
		ctx:            ctx,
		cancel:         cancel,
	}
	
	return manager
}

// Start starts the network manager
func (m *NetworkManager) Start() error {
	log.Println("Starting network manager")
	
	// Load existing networks
	if err := m.loadNetworks(); err != nil {
		log.Printf("Warning: Failed to load existing networks: %v", err)
	}
	
	// Start the update loop
	go m.updateNetworks()
	
	return nil
}

// Stop stops the network manager
func (m *NetworkManager) Stop() error {
	log.Println("Stopping network manager")
	m.cancel()
	return nil
}

// AddEventListener adds a listener for network events
func (m *NetworkManager) AddEventListener(listener NetworkEventListener) {
	m.eventMutex.Lock()
	defer m.eventMutex.Unlock()
	
	m.eventListeners = append(m.eventListeners, listener)
}

// RemoveEventListener removes an event listener
func (m *NetworkManager) RemoveEventListener(listener NetworkEventListener) {
	m.eventMutex.Lock()
	defer m.eventMutex.Unlock()
	
	for i, l := range m.eventListeners {
		if fmt.Sprintf("%p", l) == fmt.Sprintf("%p", listener) {
			m.eventListeners = append(m.eventListeners[:i], m.eventListeners[i+1:]...)
			break
		}
	}
}

// CreateNetwork creates a new network
func (m *NetworkManager) CreateNetwork(ctx context.Context, spec NetworkSpec) (*Network, error) {
	// Validate the spec
	if spec.Name == "" {
		return nil, fmt.Errorf("network name cannot be empty")
	}
	
	m.networksMutex.Lock()
	// Check if a network with this name already exists
	if id, exists := m.networksByName[spec.Name]; exists {
		m.networksMutex.Unlock()
		return nil, fmt.Errorf("network with name %s already exists (ID: %s)", spec.Name, id)
	}
	m.networksMutex.Unlock()
	
	// If network type is not specified, use default
	if spec.Type == "" {
		spec.Type = m.config.DefaultNetworkType
	}
	
	// If subnet is not specified, use default
	if spec.IPAM.Subnet == "" {
		spec.IPAM.Subnet = m.config.DefaultSubnet
	}
	
	// If gateway is not specified, use default
	if spec.IPAM.Gateway == "" {
		spec.IPAM.Gateway = m.config.DefaultGateway
	}
	
	// If IP range is not specified, use default
	if spec.IPAM.IPRange == "" {
		spec.IPAM.IPRange = m.config.DefaultIPRange
	}
	
	// If DNS servers are not specified, use default
	if len(spec.IPAM.DNSServers) == 0 {
		spec.IPAM.DNSServers = m.config.DNSServers
	}
	
	// Generate a unique ID for the network
	networkID := uuid.New().String()
	
	// Create the network
	network := &Network{
		ID:          networkID,
		Name:        spec.Name,
		Type:        spec.Type,
		IPAM:        spec.IPAM,
		Internal:    spec.Internal,
		EnableIPv6:  spec.EnableIPv6,
		Labels:      spec.Labels,
		Options:     spec.Options,
		CreatedAt:   time.Now(),
		NodeID:      m.nodeID,
		NetworkInfo: NetworkInfo{
			Active:      false,
			ConnectedVMs: []string{},
			LastUpdated: time.Now(),
		},
	}
	
	// Configure the network based on its type
	var err error
	switch spec.Type {
	case NetworkTypeBridge:
		err = m.createBridgeNetwork(network)
	case NetworkTypeOverlay:
		err = m.createOverlayNetwork(network)
	case NetworkTypeMacvlan:
		err = m.createMacvlanNetwork(network)
	default:
		err = fmt.Errorf("unsupported network type: %s", spec.Type)
	}
	
	if err != nil {
		return nil, fmt.Errorf("failed to create network: %w", err)
	}
	
	// Update the network status
	network.NetworkInfo.Active = true
	
	// Store the network
	m.networksMutex.Lock()
	m.networks[networkID] = network
	m.networksByName[network.Name] = networkID
	m.networksMutex.Unlock()
	
	// Emit network created event
	m.emitEvent(NetworkEvent{
		Type:      NetworkEventCreated,
		Network:   *network,
		Timestamp: time.Now(),
		NodeID:    m.nodeID,
		Message:   fmt.Sprintf("Network %s created", network.Name),
	})
	
	log.Printf("Created network %s (ID: %s) of type %s", network.Name, networkID, network.Type)
	
	return network, nil
}

// GetNetwork returns a network by ID
func (m *NetworkManager) GetNetwork(networkID string) (*Network, error) {
	m.networksMutex.RLock()
	defer m.networksMutex.RUnlock()
	
	network, exists := m.networks[networkID]
	if !exists {
		return nil, fmt.Errorf("network %s not found", networkID)
	}
	
	return network, nil
}

// GetNetworkByName returns a network by name
func (m *NetworkManager) GetNetworkByName(name string) (*Network, error) {
	m.networksMutex.RLock()
	networkID, exists := m.networksByName[name]
	if !exists {
		m.networksMutex.RUnlock()
		return nil, fmt.Errorf("network with name %s not found", name)
	}
	
	network, exists := m.networks[networkID]
	m.networksMutex.RUnlock()
	
	if !exists {
		return nil, fmt.Errorf("network with ID %s not found (inconsistent state)", networkID)
	}
	
	return network, nil
}

// ListNetworks returns all networks
func (m *NetworkManager) ListNetworks() []*Network {
	m.networksMutex.RLock()
	defer m.networksMutex.RUnlock()
	
	networks := make([]*Network, 0, len(m.networks))
	for _, network := range m.networks {
		networks = append(networks, network)
	}
	
	return networks
}

// DeleteNetwork deletes a network
func (m *NetworkManager) DeleteNetwork(ctx context.Context, networkID string) error {
	// Get the network
	m.networksMutex.Lock()
	network, exists := m.networks[networkID]
	if !exists {
		m.networksMutex.Unlock()
		return fmt.Errorf("network %s not found", networkID)
	}
	
	// Check if the network has connected VMs
	if len(network.NetworkInfo.ConnectedVMs) > 0 {
		m.networksMutex.Unlock()
		return fmt.Errorf("cannot delete network %s: it has %d connected VMs", 
			networkID, len(network.NetworkInfo.ConnectedVMs))
	}
	
	// Delete the network
	var err error
	switch network.Type {
	case NetworkTypeBridge:
		err = m.deleteBridgeNetwork(network)
	case NetworkTypeOverlay:
		err = m.deleteOverlayNetwork(network)
	case NetworkTypeMacvlan:
		err = m.deleteMacvlanNetwork(network)
	default:
		err = fmt.Errorf("unsupported network type: %s", network.Type)
	}
	
	if err != nil {
		m.networksMutex.Unlock()
		return fmt.Errorf("failed to delete network: %w", err)
	}
	
	// Remove the network from our maps
	delete(m.networks, networkID)
	delete(m.networksByName, network.Name)
	m.networksMutex.Unlock()
	
	// Emit network deleted event
	m.emitEvent(NetworkEvent{
		Type:      NetworkEventDeleted,
		Network:   *network,
		Timestamp: time.Now(),
		NodeID:    m.nodeID,
		Message:   fmt.Sprintf("Network %s deleted", network.Name),
	})
	
	log.Printf("Deleted network %s (ID: %s)", network.Name, networkID)
	
	return nil
}

// ConnectVM connects a VM to a network
func (m *NetworkManager) ConnectVM(ctx context.Context, networkID, vmID string) error {
	// Get the network
	m.networksMutex.Lock()
	defer m.networksMutex.Unlock()
	
	network, exists := m.networks[networkID]
	if !exists {
		return fmt.Errorf("network %s not found", networkID)
	}
	
	// Check if the VM is already connected
	for _, id := range network.NetworkInfo.ConnectedVMs {
		if id == vmID {
			return fmt.Errorf("VM %s is already connected to network %s", vmID, networkID)
		}
	}
	
	// Connect the VM to the network
	// In a real implementation, this would configure the VM's network interfaces
	
	// Update the network info
	network.NetworkInfo.ConnectedVMs = append(network.NetworkInfo.ConnectedVMs, vmID)
	network.NetworkInfo.LastUpdated = time.Now()
	
	// Emit network updated event
	m.emitEvent(NetworkEvent{
		Type:      NetworkEventUpdated,
		Network:   *network,
		Timestamp: time.Now(),
		NodeID:    m.nodeID,
		Message:   fmt.Sprintf("VM %s connected to network %s", vmID, network.Name),
	})
	
	log.Printf("Connected VM %s to network %s", vmID, network.Name)
	
	return nil
}

// DisconnectVM disconnects a VM from a network
func (m *NetworkManager) DisconnectVM(ctx context.Context, networkID, vmID string) error {
	// Get the network
	m.networksMutex.Lock()
	defer m.networksMutex.Unlock()
	
	network, exists := m.networks[networkID]
	if !exists {
		return fmt.Errorf("network %s not found", networkID)
	}
	
	// Find the VM in the connected VMs
	found := false
	for i, id := range network.NetworkInfo.ConnectedVMs {
		if id == vmID {
			// Remove the VM from the list
			network.NetworkInfo.ConnectedVMs = append(
				network.NetworkInfo.ConnectedVMs[:i],
				network.NetworkInfo.ConnectedVMs[i+1:]...,
			)
			found = true
			break
		}
	}
	
	if !found {
		return fmt.Errorf("VM %s is not connected to network %s", vmID, networkID)
	}
	
	// Disconnect the VM from the network
	// In a real implementation, this would remove the VM's network interfaces
	
	// Update the network info
	network.NetworkInfo.LastUpdated = time.Now()
	
	// Emit network updated event
	m.emitEvent(NetworkEvent{
		Type:      NetworkEventUpdated,
		Network:   *network,
		Timestamp: time.Now(),
		NodeID:    m.nodeID,
		Message:   fmt.Sprintf("VM %s disconnected from network %s", vmID, network.Name),
	})
	
	log.Printf("Disconnected VM %s from network %s", vmID, network.Name)
	
	return nil
}

// Network creation implementations for different network types

func (m *NetworkManager) createBridgeNetwork(network *Network) error {
	// In a real implementation, this would create a bridge network using the host's network stack
	
	// Check if we can use the docker bridge driver as a helper
	if isDockerAvailable() {
		return createDockerNetwork(network, "bridge")
	}
	
	// Fall back to manual bridge creation
	// This is a simplified example - in a real system, you'd:
	// 1. Create a bridge interface
	// 2. Configure IP addressing
	// 3. Set up routing
	// 4. Configure iptables as needed
	
	bridgeName := fmt.Sprintf("br-%s", network.ID[:12])
	
	// Create the bridge
	cmd := exec.Command("ip", "link", "add", "name", bridgeName, "type", "bridge")
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("failed to create bridge interface: %w", err)
	}
	
	// Parse the network CIDR
	_, ipNet, err := net.ParseCIDR(network.IPAM.Subnet)
	if err != nil {
		return fmt.Errorf("invalid subnet %s: %w", network.IPAM.Subnet, err)
	}
	
	// Assign the gateway IP to the bridge
	gatewayIP := net.ParseIP(network.IPAM.Gateway)
	if gatewayIP == nil {
		return fmt.Errorf("invalid gateway IP %s", network.IPAM.Gateway)
	}
	
	cmd = exec.Command("ip", "addr", "add", 
		fmt.Sprintf("%s/%d", gatewayIP.String(), maskBits(ipNet.Mask)), 
		"dev", bridgeName)
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("failed to assign IP to bridge: %w", err)
	}
	
	// Bring up the bridge
	cmd = exec.Command("ip", "link", "set", "dev", bridgeName, "up")
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("failed to bring up bridge: %w", err)
	}
	
	// Store the bridge name in the network options
	if network.Options == nil {
		network.Options = make(map[string]string)
	}
	network.Options["bridge"] = bridgeName
	
	log.Printf("Created bridge network %s with bridge %s", network.Name, bridgeName)
	
	return nil
}

func (m *NetworkManager) createOverlayNetwork(network *Network) error {
	// In a real implementation, this would set up an overlay network using something like VXLAN
	
	// Check if we can use the docker overlay driver as a helper
	if isDockerAvailable() {
		return createDockerNetwork(network, "overlay")
	}
	
	// For now, just log that this would be implemented
	log.Printf("Overlay network creation would be implemented here: %s", network.Name)
	
	// Store a placeholder in the network options
	if network.Options == nil {
		network.Options = make(map[string]string)
	}
	network.Options["overlay_id"] = fmt.Sprintf("overlay-%s", network.ID[:12])
	
	return nil
}

func (m *NetworkManager) createMacvlanNetwork(network *Network) error {
	// In a real implementation, this would set up a macvlan network
	
	// Check if we can use the docker macvlan driver as a helper
	if isDockerAvailable() {
		return createDockerNetwork(network, "macvlan")
	}
	
	// For now, just log that this would be implemented
	log.Printf("Macvlan network creation would be implemented here: %s", network.Name)
	
	// Store a placeholder in the network options
	if network.Options == nil {
		network.Options = make(map[string]string)
	}
	network.Options["macvlan_id"] = fmt.Sprintf("macvlan-%s", network.ID[:12])
	
	return nil
}

// Network deletion implementations for different network types

func (m *NetworkManager) deleteBridgeNetwork(network *Network) error {
	// In a real implementation, this would remove a bridge network
	
	// Check if we used docker to create the network
	if isDockerAvailable() && network.Options != nil && network.Options["docker_network_id"] != "" {
		return deleteDockerNetwork(network.Options["docker_network_id"])
	}
	
	// Otherwise, manually delete the bridge
	if network.Options != nil && network.Options["bridge"] != "" {
		bridgeName := network.Options["bridge"]
		
		// Bring down the bridge
		cmd := exec.Command("ip", "link", "set", "dev", bridgeName, "down")
		if err := cmd.Run(); err != nil {
			return fmt.Errorf("failed to bring down bridge: %w", err)
		}
		
		// Delete the bridge
		cmd = exec.Command("ip", "link", "del", "dev", bridgeName)
		if err := cmd.Run(); err != nil {
			return fmt.Errorf("failed to delete bridge: %w", err)
		}
		
		log.Printf("Deleted bridge network %s with bridge %s", network.Name, bridgeName)
	}
	
	return nil
}

func (m *NetworkManager) deleteOverlayNetwork(network *Network) error {
	// In a real implementation, this would remove an overlay network
	
	// Check if we used docker to create the network
	if isDockerAvailable() && network.Options != nil && network.Options["docker_network_id"] != "" {
		return deleteDockerNetwork(network.Options["docker_network_id"])
	}
	
	// For now, just log that this would be implemented
	log.Printf("Overlay network deletion would be implemented here: %s", network.Name)
	
	return nil
}

func (m *NetworkManager) deleteMacvlanNetwork(network *Network) error {
	// In a real implementation, this would remove a macvlan network
	
	// Check if we used docker to create the network
	if isDockerAvailable() && network.Options != nil && network.Options["docker_network_id"] != "" {
		return deleteDockerNetwork(network.Options["docker_network_id"])
	}
	
	// For now, just log that this would be implemented
	log.Printf("Macvlan network deletion would be implemented here: %s", network.Name)
	
	return nil
}

// Helper functions

func (m *NetworkManager) emitEvent(event NetworkEvent) {
	m.eventMutex.RLock()
	defer m.eventMutex.RUnlock()
	
	for _, listener := range m.eventListeners {
		go func(l NetworkEventListener, e NetworkEvent) {
			defer func() {
				if r := recover(); r != nil {
					log.Printf("Recovered from panic in event listener: %v", r)
				}
			}()
			
			l(e)
		}(listener, event)
	}
}

func (m *NetworkManager) updateNetworks() {
	ticker := time.NewTicker(m.config.UpdateInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-m.ctx.Done():
			return
		case <-ticker.C:
			m.refreshNetworkStatus()
		}
	}
}

func (m *NetworkManager) refreshNetworkStatus() {
	m.networksMutex.Lock()
	defer m.networksMutex.Unlock()
	
	for _, network := range m.networks {
		// In a real implementation, this would check the actual status of the network
		// and update metrics
		
		network.NetworkInfo.LastUpdated = time.Now()
	}
}

func (m *NetworkManager) loadNetworks() error {
	// In a real implementation, this would load existing networks from the system
	
	// Check if we can use docker networks
	if isDockerAvailable() {
		return m.loadDockerNetworks()
	}
	
	return nil
}

func (m *NetworkManager) loadDockerNetworks() error {
	cmd := exec.Command("docker", "network", "ls", "--format", "{{.ID}}|{{.Name}}|{{.Driver}}")
	output, err := cmd.Output()
	if err != nil {
		return fmt.Errorf("failed to list docker networks: %w", err)
	}
	
	lines := strings.Split(string(output), "\n")
	for _, line := range lines {
		if line == "" {
			continue
		}
		
		parts := strings.Split(line, "|")
		if len(parts) != 3 {
			continue
		}
		
		dockerNetID := parts[0]
		name := parts[1]
		driver := parts[2]
		
		// Skip default networks
		if name == "bridge" || name == "host" || name == "none" {
			continue
		}
		
		// Get more details about the network
		cmd = exec.Command("docker", "network", "inspect", dockerNetID)
		inspectOutput, err := cmd.Output()
		if err != nil {
			log.Printf("Warning: Failed to inspect docker network %s: %v", dockerNetID, err)
			continue
		}
		
		var networkDetails []map[string]interface{}
		if err := json.Unmarshal(inspectOutput, &networkDetails); err != nil {
			log.Printf("Warning: Failed to parse docker network details: %v", err)
			continue
		}
		
		if len(networkDetails) == 0 {
			continue
		}
		
		details := networkDetails[0]
		
		// Convert driver to our network type
		var netType NetworkType
		switch driver {
		case "bridge":
			netType = NetworkTypeBridge
		case "overlay":
			netType = NetworkTypeOverlay
		case "macvlan":
			netType = NetworkTypeMacvlan
		default:
			// Skip unsupported network types
			continue
		}
		
		// Create a network object
		networkID := uuid.New().String()
		
		// Try to extract IPAM config
		ipamConfig := IPAMConfig{}
		if ipamDetails, ok := details["IPAM"].(map[string]interface{}); ok {
			if configs, ok := ipamDetails["Config"].([]interface{}); ok && len(configs) > 0 {
				if config, ok := configs[0].(map[string]interface{}); ok {
					if subnet, ok := config["Subnet"].(string); ok {
						ipamConfig.Subnet = subnet
					}
					if gateway, ok := config["Gateway"].(string); ok {
						ipamConfig.Gateway = gateway
					}
				}
			}
		}
		
		// Create the network object
		network := &Network{
			ID:        networkID,
			Name:      name,
			Type:      netType,
			IPAM:      ipamConfig,
			CreatedAt: time.Now(), // We don't know the real creation time
			NodeID:    m.nodeID,
			NetworkInfo: NetworkInfo{
				Active:      true,
				ConnectedVMs: []string{},
				LastUpdated: time.Now(),
			},
			Options: map[string]string{
				"docker_network_id": dockerNetID,
			},
		}
		
		// Store the network
		m.networks[networkID] = network
		m.networksByName[name] = networkID
		
		log.Printf("Loaded docker network %s (ID: %s) of type %s", name, networkID, netType)
	}
	
	return nil
}

// Helper to check if docker is available
func isDockerAvailable() bool {
	cmd := exec.Command("docker", "version")
	return cmd.Run() == nil
}

// Helper to create a docker network
func createDockerNetwork(network *Network, driver string) error {
	args := []string{
		"network", "create",
		"--driver", driver,
		"--subnet", network.IPAM.Subnet,
	}
	
	if network.IPAM.Gateway != "" {
		args = append(args, "--gateway", network.IPAM.Gateway)
	}
	
	if network.IPAM.IPRange != "" {
		args = append(args, "--ip-range", network.IPAM.IPRange)
	}
	
	if network.Internal {
		args = append(args, "--internal")
	}
	
	// Add labels
	for k, v := range network.Labels {
		args = append(args, "--label", fmt.Sprintf("%s=%s", k, v))
	}
	
	// Add the network name
	args = append(args, network.Name)
	
	// Create the network
	cmd := exec.Command("docker", args...)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed to create docker network: %w, output: %s", err, string(output))
	}
	
	// Get the docker network ID
	dockerNetID := strings.TrimSpace(string(output))
	
	// Store the docker network ID in the network options
	if network.Options == nil {
		network.Options = make(map[string]string)
	}
	network.Options["docker_network_id"] = dockerNetID
	
	log.Printf("Created docker network %s with ID %s", network.Name, dockerNetID)
	
	return nil
}

// Helper to delete a docker network
func deleteDockerNetwork(dockerNetID string) error {
	cmd := exec.Command("docker", "network", "rm", dockerNetID)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed to delete docker network: %w, output: %s", err, string(output))
	}
	
	log.Printf("Deleted docker network %s", dockerNetID)
	
	return nil
}

// Helper to compute netmask bits
func maskBits(mask net.IPMask) int {
	bits, _ := mask.Size()
	return bits
}
