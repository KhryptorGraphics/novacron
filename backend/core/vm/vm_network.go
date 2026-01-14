package vm

import (
	"context"
	"fmt"
	"log"
	"net"
	"os/exec"
	"strings"
	"sync"
	"time"
)

// NetworkType represents the type of network
type NetworkType string

const (
	// NetworkTypeBridge represents a bridge network
	NetworkTypeBridge NetworkType = "bridge"

	// NetworkTypeNAT represents a NAT network
	NetworkTypeNAT NetworkType = "nat"

	// NetworkTypeHost represents a host-only network
	NetworkTypeHost NetworkType = "host"

	// NetworkTypeIsolated represents an isolated network
	NetworkTypeIsolated NetworkType = "isolated"
)

// VMNetwork represents a VM network
type VMNetwork struct {
	ID        string            `json:"id"`
	Name      string            `json:"name"`
	Type      NetworkType       `json:"type"`
	Subnet    string            `json:"subnet"`
	Gateway   string            `json:"gateway"`
	DHCP      bool              `json:"dhcp"`
	DHCPRange string            `json:"dhcp_range,omitempty"`
	Bridge    string            `json:"bridge,omitempty"`
	MTU       int               `json:"mtu"`
	VLAN      int               `json:"vlan,omitempty"`
	CreatedAt time.Time         `json:"created_at"`
	UpdatedAt time.Time         `json:"updated_at"`
	Tags      []string          `json:"tags,omitempty"`
	Metadata  map[string]string `json:"metadata,omitempty"`
}

// VMNetworkInterface represents a VM network interface
type VMNetworkInterface struct {
	ID         string    `json:"id"`
	VMID       string    `json:"vm_id"`
	NetworkID  string    `json:"network_id"`
	MACAddress string    `json:"mac_address"`
	IPAddress  string    `json:"ip_address,omitempty"`
	Model      string    `json:"model"`
	MTU        int       `json:"mtu"`
	Index      int       `json:"index"`
	CreatedAt  time.Time `json:"created_at"`
	UpdatedAt  time.Time `json:"updated_at"`
}

// VMNetworkManager manages VM networks
type VMNetworkManager struct {
	networks        map[string]*VMNetwork
	networksMutex   sync.RWMutex
	interfaces      map[string][]*VMNetworkInterface
	interfacesMutex sync.RWMutex
	vmManager       *VMManager
}

// NewVMNetworkManager creates a new VM network manager
func NewVMNetworkManager(vmManager *VMManager) *VMNetworkManager {
	return &VMNetworkManager{
		networks:   make(map[string]*VMNetwork),
		interfaces: make(map[string][]*VMNetworkInterface),
		vmManager:  vmManager,
	}
}

// CreateNetwork creates a new VM network
func (m *VMNetworkManager) CreateNetwork(ctx context.Context, name string, networkType NetworkType, subnet string, gateway string, dhcp bool, dhcpRange string, bridge string, mtu int, vlan int, tags []string, metadata map[string]string) (*VMNetwork, error) {
	// Validate network type
	switch networkType {
	case NetworkTypeBridge, NetworkTypeNAT, NetworkTypeHost, NetworkTypeIsolated:
		// Valid network type
	default:
		return nil, fmt.Errorf("invalid network type: %s", networkType)
	}

	// Validate subnet
	_, ipNet, err := net.ParseCIDR(subnet)
	if err != nil {
		return nil, fmt.Errorf("invalid subnet: %w", err)
	}

	// Validate gateway
	if gateway != "" {
		gatewayIP := net.ParseIP(gateway)
		if gatewayIP == nil {
			return nil, fmt.Errorf("invalid gateway IP: %s", gateway)
		}

		// Check if gateway is in subnet
		if !ipNet.Contains(gatewayIP) {
			return nil, fmt.Errorf("gateway %s is not in subnet %s", gateway, subnet)
		}
	}

	// Validate DHCP range
	if dhcp && dhcpRange != "" {
		parts := strings.Split(dhcpRange, "-")
		if len(parts) != 2 {
			return nil, fmt.Errorf("invalid DHCP range format: %s", dhcpRange)
		}

		startIP := net.ParseIP(strings.TrimSpace(parts[0]))
		endIP := net.ParseIP(strings.TrimSpace(parts[1]))

		if startIP == nil || endIP == nil {
			return nil, fmt.Errorf("invalid DHCP range IPs: %s", dhcpRange)
		}

		// Check if IPs are in subnet
		if !ipNet.Contains(startIP) || !ipNet.Contains(endIP) {
			return nil, fmt.Errorf("DHCP range %s is not in subnet %s", dhcpRange, subnet)
		}
	}

	// Validate bridge
	if networkType == NetworkTypeBridge && bridge == "" {
		return nil, fmt.Errorf("bridge name is required for bridge networks")
	}

	// Validate MTU
	if mtu < 576 || mtu > 9000 {
		return nil, fmt.Errorf("invalid MTU: %d (must be between 576 and 9000)", mtu)
	}

	// Generate network ID
	networkID := fmt.Sprintf("net-%s", strings.ReplaceAll(name, " ", "-"))

	// Check if network already exists
	m.networksMutex.RLock()
	if _, exists := m.networks[networkID]; exists {
		m.networksMutex.RUnlock()
		return nil, fmt.Errorf("network with ID %s already exists", networkID)
	}
	m.networksMutex.RUnlock()

	// Create network
	network := &VMNetwork{
		ID:        networkID,
		Name:      name,
		Type:      networkType,
		Subnet:    subnet,
		Gateway:   gateway,
		DHCP:      dhcp,
		DHCPRange: dhcpRange,
		Bridge:    bridge,
		MTU:       mtu,
		VLAN:      vlan,
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
		Tags:      tags,
		Metadata:  metadata,
	}

	// Create network in the system
	if err := m.createNetworkInSystem(ctx, network); err != nil {
		return nil, fmt.Errorf("failed to create network in system: %w", err)
	}

	// Store network
	m.networksMutex.Lock()
	m.networks[networkID] = network
	m.networksMutex.Unlock()

	log.Printf("Created network %s (%s)", network.Name, network.ID)

	return network, nil
}

// GetNetwork returns a network by ID
func (m *VMNetworkManager) GetNetwork(networkID string) (*VMNetwork, error) {
	m.networksMutex.RLock()
	defer m.networksMutex.RUnlock()

	network, exists := m.networks[networkID]
	if !exists {
		return nil, fmt.Errorf("network %s not found", networkID)
	}

	return network, nil
}

// ListNetworks returns all networks
func (m *VMNetworkManager) ListNetworks() []*VMNetwork {
	m.networksMutex.RLock()
	defer m.networksMutex.RUnlock()

	networks := make([]*VMNetwork, 0, len(m.networks))
	for _, network := range m.networks {
		networks = append(networks, network)
	}

	return networks
}

// DeleteNetwork deletes a network
func (m *VMNetworkManager) DeleteNetwork(ctx context.Context, networkID string) error {
	// Get the network
	m.networksMutex.RLock()
	network, exists := m.networks[networkID]
	if !exists {
		m.networksMutex.RUnlock()
		return fmt.Errorf("network %s not found", networkID)
	}
	m.networksMutex.RUnlock()

	// Check if network is in use
	m.interfacesMutex.RLock()
	for vmID, interfaces := range m.interfaces {
		for _, iface := range interfaces {
			if iface.NetworkID == networkID {
				m.interfacesMutex.RUnlock()
				return fmt.Errorf("network %s is in use by VM %s", networkID, vmID)
			}
		}
	}
	m.interfacesMutex.RUnlock()

	// Delete network from the system
	if err := m.deleteNetworkFromSystem(ctx, network); err != nil {
		return fmt.Errorf("failed to delete network from system: %w", err)
	}

	// Remove network
	m.networksMutex.Lock()
	delete(m.networks, networkID)
	m.networksMutex.Unlock()

	log.Printf("Deleted network %s (%s)", network.Name, network.ID)

	return nil
}

// AttachNetworkInterface attaches a network interface to a VM
func (m *VMNetworkManager) AttachNetworkInterface(ctx context.Context, vmID, networkID, macAddress, ipAddress, model string, mtu int) (*VMNetworkInterface, error) {
	// Get the VM
	vm, err := m.vmManager.GetVM(vmID)
	if err != nil {
		return nil, fmt.Errorf("failed to get VM: %w", err)
	}

	// Get the network
	m.networksMutex.RLock()
	network, exists := m.networks[networkID]
	if !exists {
		m.networksMutex.RUnlock()
		return nil, fmt.Errorf("network %s not found", networkID)
	}
	m.networksMutex.RUnlock()

	// Validate MAC address
	if macAddress == "" {
		// Generate a random MAC address
		macAddress = generateMACAddress()
	} else {
		// Validate MAC address format
		_, err := net.ParseMAC(macAddress)
		if err != nil {
			return nil, fmt.Errorf("invalid MAC address: %w", err)
		}
	}

	// Validate IP address
	if ipAddress != "" {
		ip := net.ParseIP(ipAddress)
		if ip == nil {
			return nil, fmt.Errorf("invalid IP address: %s", ipAddress)
		}

		// Check if IP is in subnet
		_, ipNet, err := net.ParseCIDR(network.Subnet)
		if err != nil {
			return nil, fmt.Errorf("invalid subnet: %w", err)
		}

		if !ipNet.Contains(ip) {
			return nil, fmt.Errorf("IP address %s is not in subnet %s", ipAddress, network.Subnet)
		}
	}

	// Validate model
	if model == "" {
		model = "virtio"
	}

	// Validate MTU
	if mtu == 0 {
		mtu = network.MTU
	} else if mtu < 576 || mtu > 9000 {
		return nil, fmt.Errorf("invalid MTU: %d (must be between 576 and 9000)", mtu)
	}

	// Get existing interfaces for the VM
	m.interfacesMutex.RLock()
	interfaces, exists := m.interfaces[vmID]
	if !exists {
		interfaces = make([]*VMNetworkInterface, 0)
	}

	// Check if VM already has an interface on this network
	for _, iface := range interfaces {
		if iface.NetworkID == networkID {
			m.interfacesMutex.RUnlock()
			return nil, fmt.Errorf("VM %s already has an interface on network %s", vmID, networkID)
		}
	}
	m.interfacesMutex.RUnlock()

	// Generate interface ID
	interfaceID := fmt.Sprintf("%s-net%d", vmID, len(interfaces))

	// Create interface
	iface := &VMNetworkInterface{
		ID:         interfaceID,
		VMID:       vmID,
		NetworkID:  networkID,
		MACAddress: macAddress,
		IPAddress:  ipAddress,
		Model:      model,
		MTU:        mtu,
		Index:      len(interfaces),
		CreatedAt:  time.Now(),
		UpdatedAt:  time.Now(),
	}

	// Attach interface to VM
	if err := m.attachInterfaceToVM(ctx, vm, network, iface); err != nil {
		return nil, fmt.Errorf("failed to attach interface to VM: %w", err)
	}

	// Store interface
	m.interfacesMutex.Lock()
	m.interfaces[vmID] = append(interfaces, iface)
	m.interfacesMutex.Unlock()

	log.Printf("Attached network interface %s to VM %s on network %s", iface.ID, vmID, networkID)

	return iface, nil
}

// DetachNetworkInterface detaches a network interface from a VM
func (m *VMNetworkManager) DetachNetworkInterface(ctx context.Context, vmID, interfaceID string) error {
	// Get the VM
	vm, err := m.vmManager.GetVM(vmID)
	if err != nil {
		return fmt.Errorf("failed to get VM: %w", err)
	}

	// Get the interface
	m.interfacesMutex.RLock()
	interfaces, exists := m.interfaces[vmID]
	if !exists {
		m.interfacesMutex.RUnlock()
		return fmt.Errorf("VM %s has no network interfaces", vmID)
	}

	var iface *VMNetworkInterface
	var ifaceIndex int
	for i, intf := range interfaces {
		if intf.ID == interfaceID {
			iface = intf
			ifaceIndex = i
			break
		}
	}

	if iface == nil {
		m.interfacesMutex.RUnlock()
		return fmt.Errorf("interface %s not found for VM %s", interfaceID, vmID)
	}
	m.interfacesMutex.RUnlock()

	// Get the network
	m.networksMutex.RLock()
	network, exists := m.networks[iface.NetworkID]
	if !exists {
		m.networksMutex.RUnlock()
		return fmt.Errorf("network %s not found", iface.NetworkID)
	}
	m.networksMutex.RUnlock()

	// Detach interface from VM
	if err := m.detachInterfaceFromVM(ctx, vm, network, iface); err != nil {
		return fmt.Errorf("failed to detach interface from VM: %w", err)
	}

	// Remove interface
	m.interfacesMutex.Lock()
	m.interfaces[vmID] = append(interfaces[:ifaceIndex], interfaces[ifaceIndex+1:]...)
	m.interfacesMutex.Unlock()

	log.Printf("Detached network interface %s from VM %s", interfaceID, vmID)

	return nil
}

// ListNetworkInterfaces returns all network interfaces for a VM
func (m *VMNetworkManager) ListNetworkInterfaces(vmID string) ([]*VMNetworkInterface, error) {
	m.interfacesMutex.RLock()
	defer m.interfacesMutex.RUnlock()

	interfaces, exists := m.interfaces[vmID]
	if !exists {
		return make([]*VMNetworkInterface, 0), nil
	}

	// Create a copy of the interfaces
	result := make([]*VMNetworkInterface, len(interfaces))
	copy(result, interfaces)

	return result, nil
}

// GetNetworkInterface returns a network interface by ID
func (m *VMNetworkManager) GetNetworkInterface(vmID, interfaceID string) (*VMNetworkInterface, error) {
	m.interfacesMutex.RLock()
	defer m.interfacesMutex.RUnlock()

	interfaces, exists := m.interfaces[vmID]
	if !exists {
		return nil, fmt.Errorf("VM %s has no network interfaces", vmID)
	}

	for _, iface := range interfaces {
		if iface.ID == interfaceID {
			return iface, nil
		}
	}

	return nil, fmt.Errorf("interface %s not found for VM %s", interfaceID, vmID)
}

// UpdateNetworkInterface updates a network interface
func (m *VMNetworkManager) UpdateNetworkInterface(ctx context.Context, vmID, interfaceID string, ipAddress string, mtu int) (*VMNetworkInterface, error) {
	// Get the VM
	vm, err := m.vmManager.GetVM(vmID)
	if err != nil {
		return nil, fmt.Errorf("failed to get VM: %w", err)
	}

	// Get the interface
	m.interfacesMutex.Lock()
	defer m.interfacesMutex.Unlock()

	interfaces, exists := m.interfaces[vmID]
	if !exists {
		return nil, fmt.Errorf("VM %s has no network interfaces", vmID)
	}

	var iface *VMNetworkInterface
	for _, intf := range interfaces {
		if intf.ID == interfaceID {
			iface = intf
			break
		}
	}

	if iface == nil {
		return nil, fmt.Errorf("interface %s not found for VM %s", interfaceID, vmID)
	}

	// Get the network
	m.networksMutex.RLock()
	network, exists := m.networks[iface.NetworkID]
	if !exists {
		m.networksMutex.RUnlock()
		return nil, fmt.Errorf("network %s not found", iface.NetworkID)
	}
	m.networksMutex.RUnlock()

	// Validate IP address
	if ipAddress != "" && ipAddress != iface.IPAddress {
		ip := net.ParseIP(ipAddress)
		if ip == nil {
			return nil, fmt.Errorf("invalid IP address: %s", ipAddress)
		}

		// Check if IP is in subnet
		_, ipNet, err := net.ParseCIDR(network.Subnet)
		if err != nil {
			return nil, fmt.Errorf("invalid subnet: %w", err)
		}

		if !ipNet.Contains(ip) {
			return nil, fmt.Errorf("IP address %s is not in subnet %s", ipAddress, network.Subnet)
		}

		// Update IP address
		iface.IPAddress = ipAddress
	}

	// Validate MTU
	if mtu != 0 && mtu != iface.MTU {
		if mtu < 576 || mtu > 9000 {
			return nil, fmt.Errorf("invalid MTU: %d (must be between 576 and 9000)", mtu)
		}

		// Update MTU
		iface.MTU = mtu
	}

	// Update interface in the system
	if err := m.updateInterfaceInVM(ctx, vm, network, iface); err != nil {
		return nil, fmt.Errorf("failed to update interface in VM: %w", err)
	}

	// Update timestamp
	iface.UpdatedAt = time.Now()

	log.Printf("Updated network interface %s for VM %s", interfaceID, vmID)

	return iface, nil
}

// createNetworkInSystem creates a network in the system
func (m *VMNetworkManager) createNetworkInSystem(ctx context.Context, network *VMNetwork) error {
	// In a real implementation, this would create the network in the system
	// For example, creating a bridge, configuring DHCP, etc.

	// For simplicity, we'll just log the operation
	log.Printf("Creating network %s (%s) in system", network.Name, network.ID)

	// Simulate network creation based on type
	switch network.Type {
	case NetworkTypeBridge:
		// Check if bridge exists
		if network.Bridge != "" {
			cmd := exec.CommandContext(ctx, "ip", "link", "show", network.Bridge)
			if err := cmd.Run(); err != nil {
				// Bridge doesn't exist, create it
				log.Printf("Creating bridge %s", network.Bridge)
				// In a real implementation, this would create the bridge
			}
		}
	case NetworkTypeNAT:
		// In a real implementation, this would set up NAT
		log.Printf("Setting up NAT for network %s", network.ID)
	case NetworkTypeHost:
		// In a real implementation, this would set up a host-only network
		log.Printf("Setting up host-only network %s", network.ID)
	case NetworkTypeIsolated:
		// In a real implementation, this would set up an isolated network
		log.Printf("Setting up isolated network %s", network.ID)
	}

	// If DHCP is enabled, set up DHCP server
	if network.DHCP {
		log.Printf("Setting up DHCP server for network %s", network.ID)
		// In a real implementation, this would set up a DHCP server
	}

	return nil
}

// deleteNetworkFromSystem deletes a network from the system
func (m *VMNetworkManager) deleteNetworkFromSystem(ctx context.Context, network *VMNetwork) error {
	// In a real implementation, this would delete the network from the system
	// For example, removing a bridge, stopping DHCP, etc.

	// For simplicity, we'll just log the operation
	log.Printf("Deleting network %s (%s) from system", network.Name, network.ID)

	return nil
}

// attachInterfaceToVM attaches a network interface to a VM
func (m *VMNetworkManager) attachInterfaceToVM(ctx context.Context, vm *VM, network *VMNetwork, iface *VMNetworkInterface) error {
	// In a real implementation, this would attach the interface to the VM
	// For example, adding a network device to a QEMU/KVM VM

	// For simplicity, we'll just log the operation
	log.Printf("Attaching interface %s to VM %s on network %s", iface.ID, vm.ID(), network.ID)

	return nil
}

// detachInterfaceFromVM detaches a network interface from a VM
func (m *VMNetworkManager) detachInterfaceFromVM(ctx context.Context, vm *VM, network *VMNetwork, iface *VMNetworkInterface) error {
	// In a real implementation, this would detach the interface from the VM
	// For example, removing a network device from a QEMU/KVM VM

	// For simplicity, we'll just log the operation
	log.Printf("Detaching interface %s from VM %s on network %s", iface.ID, vm.ID(), network.ID)

	return nil
}

// updateInterfaceInVM updates a network interface in a VM
func (m *VMNetworkManager) updateInterfaceInVM(ctx context.Context, vm *VM, network *VMNetwork, iface *VMNetworkInterface) error {
	// In a real implementation, this would update the interface in the VM
	// For example, changing the IP address or MTU of a network device

	// For simplicity, we'll just log the operation
	log.Printf("Updating interface %s in VM %s on network %s", iface.ID, vm.ID(), network.ID)

	return nil
}

// generateMACAddress generates a random MAC address
func generateMACAddress() string {
	// Generate a random MAC address with the locally administered bit set
	// and the multicast bit cleared (unicast)
	mac := make([]byte, 6)
	mac[0] = 0x52 // Locally administered, unicast
	mac[1] = 0x54 // "T" for "Test"

	// Generate random bytes for the rest of the MAC address
	for i := 2; i < 6; i++ {
		mac[i] = byte(time.Now().UnixNano() >> uint((i-2)*8) & 0xff)
	}

	return fmt.Sprintf("%02x:%02x:%02x:%02x:%02x:%02x", mac[0], mac[1], mac[2], mac[3], mac[4], mac[5])
}
