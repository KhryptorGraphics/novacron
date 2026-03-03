package overlay

import (
	"context"
	"fmt"
	"log"
	"sync"
)

// OverlayType represents the type of network overlay
type OverlayType string

const (
	// VXLAN is a Layer 2 overlay over Layer 3 network
	VXLAN OverlayType = "vxlan"
	// GENEVE is a flexible overlay protocol with extensible options
	GENEVE OverlayType = "geneve"
	// GRE is a simple IP encapsulation protocol
	GRE OverlayType = "gre"
	// NVGRE is Network Virtualization using GRE
	NVGRE OverlayType = "nvgre"
	// MPLSoUDP is MPLS over UDP tunneling
	MPLSoUDP OverlayType = "mpls-over-udp"
	// VLAN is Virtual LAN (not a true overlay but included for completeness)
	VLAN OverlayType = "vlan"
)

// NetworkInterface represents a physical or virtual network interface
type NetworkInterface struct {
	// Name of the interface
	Name string
	// MAC address
	MACAddress string
	// MTU of the interface
	MTU int
	// Whether the interface is up
	IsUp bool
	// Assigned IP addresses with CIDR notation
	IPAddresses []string
}

// OverlayNetwork represents a virtual network created using an overlay technology
type OverlayNetwork struct {
	// Unique ID of the overlay network
	ID string
	// Human readable name
	Name string
	// Type of overlay
	Type OverlayType
	// CIDR of the network
	CIDR string
	// VNI (VXLAN Network Identifier) or equivalent identifier
	VNI uint32
	// MTU for the overlay
	MTU int
	// Whether the network is currently active
	Active bool
	// Network interfaces participating in this overlay
	Interfaces []string
	// Additional overlay-specific options
	Options map[string]string
}

// EndpointConfig represents the configuration of an endpoint in an overlay network
type EndpointConfig struct {
	// Overlay network ID
	NetworkID string
	// Endpoint name
	Name string
	// Endpoint MAC address
	MACAddress string
	// Endpoint IP address
	IPAddress string
	// Additional endpoint specific options
	Options map[string]string
}

// DriverCapabilities represents the capabilities of an overlay driver
type DriverCapabilities struct {
	// Types of overlays supported
	SupportedTypes []OverlayType
	// Maximum MTU supported
	MaxMTU int
	// Whether the driver supports layer 2 extensions
	SupportsL2Extension bool
	// Whether the driver supports network policies
	SupportsNetworkPolicies bool
	// Whether the driver supports quality of service
	SupportsQoS bool
	// Whether the driver supports service mesh
	SupportsServiceMesh bool
}

// OverlayDriver is the interface that overlay drivers must implement
type OverlayDriver interface {
	// Name returns the unique name of the driver
	Name() string

	// Initialize initializes the driver
	Initialize(ctx context.Context) error

	// Capabilities returns the capabilities of the driver
	Capabilities() DriverCapabilities

	// CreateNetwork creates a new overlay network
	CreateNetwork(ctx context.Context, network OverlayNetwork) error

	// DeleteNetwork deletes an overlay network
	DeleteNetwork(ctx context.Context, networkID string) error

	// UpdateNetwork updates an overlay network
	UpdateNetwork(ctx context.Context, network OverlayNetwork) error

	// GetNetwork returns information about an overlay network
	GetNetwork(ctx context.Context, networkID string) (OverlayNetwork, error)

	// ListNetworks returns a list of all overlay networks
	ListNetworks(ctx context.Context) ([]OverlayNetwork, error)

	// CreateEndpoint creates a new endpoint in an overlay network
	CreateEndpoint(ctx context.Context, endpoint EndpointConfig) error

	// DeleteEndpoint deletes an endpoint from an overlay network
	DeleteEndpoint(ctx context.Context, networkID, endpointName string) error

	// GetEndpoint returns information about an endpoint
	GetEndpoint(ctx context.Context, networkID, endpointName string) (EndpointConfig, error)

	// ListEndpoints returns a list of all endpoints in an overlay network
	ListEndpoints(ctx context.Context, networkID string) ([]EndpointConfig, error)

	// ApplyNetworkPolicy applies a network policy to an overlay network
	ApplyNetworkPolicy(ctx context.Context, networkID string, policy NetworkPolicy) error

	// RemoveNetworkPolicy removes a network policy from an overlay network
	RemoveNetworkPolicy(ctx context.Context, networkID, policyID string) error

	// Shutdown cleans up resources
	Shutdown(ctx context.Context) error
}

// NetworkPolicy represents a network policy for an overlay network
type NetworkPolicy struct {
	// Unique ID of the policy
	ID string
	// Human readable name
	Name string
	// Network ID this policy applies to
	NetworkID string
	// Priority - higher number = higher priority
	Priority int
	// List of rules in the policy
	Rules []PolicyRule
}

// PolicyRuleType represents the type of a policy rule
type PolicyRuleType string

const (
	// RuleAllow allows traffic matching the rule
	RuleAllow PolicyRuleType = "allow"
	// RuleDeny denies traffic matching the rule
	RuleDeny PolicyRuleType = "deny"
	// RuleRedirect redirects traffic to another destination
	RuleRedirect PolicyRuleType = "redirect"
	// RuleQoS applies quality of service settings
	RuleQoS PolicyRuleType = "qos"
	// RuleLimit limits traffic rate
	RuleLimit PolicyRuleType = "limit"
)

// PolicyRule represents a single rule in a network policy
type PolicyRule struct {
	// Type of rule
	Type PolicyRuleType
	// Source endpoint selector
	SourceSelector string
	// Destination endpoint selector
	DestinationSelector string
	// Protocol (tcp, udp, icmp, all)
	Protocol string
	// Source port range (for TCP/UDP)
	SourcePortRange string
	// Destination port range (for TCP/UDP)
	DestinationPortRange string
	// Action specific parameters
	ActionParams map[string]string
}

// OverlayManager manages network overlays and their drivers
type OverlayManager struct {
	// Map of driver name to driver instance
	drivers map[string]OverlayDriver
	// Map of network ID to the driver that manages it
	networkDrivers map[string]string
	// Mutex for protecting concurrent access
	mu sync.RWMutex
	// Is the manager initialized
	initialized bool
}

// NewOverlayManager creates a new overlay manager
func NewOverlayManager() *OverlayManager {
	return &OverlayManager{
		drivers:        make(map[string]OverlayDriver),
		networkDrivers: make(map[string]string),
		initialized:    false,
	}
}

// RegisterDriver registers an overlay driver with the manager
func (m *OverlayManager) RegisterDriver(driver OverlayDriver) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	driverName := driver.Name()
	if _, exists := m.drivers[driverName]; exists {
		return fmt.Errorf("driver with name %s already registered", driverName)
	}

	m.drivers[driverName] = driver
	return nil
}

// Initialize initializes the overlay manager and all registered drivers
func (m *OverlayManager) Initialize(ctx context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.initialized {
		return fmt.Errorf("overlay manager already initialized")
	}

	// Initialize all drivers
	for name, driver := range m.drivers {
		if err := driver.Initialize(ctx); err != nil {
			return fmt.Errorf("failed to initialize driver %s: %v", name, err)
		}
	}

	// Build the network to driver mapping
	for name, driver := range m.drivers {
		networks, err := driver.ListNetworks(ctx)
		if err != nil {
			log.Printf("Warning: failed to list networks for driver %s: %v", name, err)
			continue
		}

		for _, network := range networks {
			m.networkDrivers[network.ID] = name
		}
	}

	m.initialized = true
	return nil
}

// CreateNetwork creates a new overlay network using the specified driver
func (m *OverlayManager) CreateNetwork(ctx context.Context, network OverlayNetwork, driverName string) error {
	m.mu.RLock()
	driver, exists := m.drivers[driverName]
	m.mu.RUnlock()

	if !exists {
		return fmt.Errorf("driver %s not found", driverName)
	}

	if err := driver.CreateNetwork(ctx, network); err != nil {
		return err
	}

	m.mu.Lock()
	m.networkDrivers[network.ID] = driverName
	m.mu.Unlock()

	return nil
}

// DeleteNetwork deletes an overlay network
func (m *OverlayManager) DeleteNetwork(ctx context.Context, networkID string) error {
	m.mu.RLock()
	driverName, exists := m.networkDrivers[networkID]
	if !exists {
		m.mu.RUnlock()
		return fmt.Errorf("network %s not found", networkID)
	}

	driver, exists := m.drivers[driverName]
	m.mu.RUnlock()

	if !exists {
		return fmt.Errorf("driver %s not found for network %s", driverName, networkID)
	}

	if err := driver.DeleteNetwork(ctx, networkID); err != nil {
		return err
	}

	m.mu.Lock()
	delete(m.networkDrivers, networkID)
	m.mu.Unlock()

	return nil
}

// UpdateNetwork updates an overlay network
func (m *OverlayManager) UpdateNetwork(ctx context.Context, network OverlayNetwork) error {
	m.mu.RLock()
	driverName, exists := m.networkDrivers[network.ID]
	if !exists {
		m.mu.RUnlock()
		return fmt.Errorf("network %s not found", network.ID)
	}

	driver, exists := m.drivers[driverName]
	m.mu.RUnlock()

	if !exists {
		return fmt.Errorf("driver %s not found for network %s", driverName, network.ID)
	}

	return driver.UpdateNetwork(ctx, network)
}

// GetNetwork returns information about an overlay network
func (m *OverlayManager) GetNetwork(ctx context.Context, networkID string) (OverlayNetwork, error) {
	m.mu.RLock()
	driverName, exists := m.networkDrivers[networkID]
	if !exists {
		m.mu.RUnlock()
		return OverlayNetwork{}, fmt.Errorf("network %s not found", networkID)
	}

	driver, exists := m.drivers[driverName]
	m.mu.RUnlock()

	if !exists {
		return OverlayNetwork{}, fmt.Errorf("driver %s not found for network %s", driverName, networkID)
	}

	return driver.GetNetwork(ctx, networkID)
}

// ListNetworks returns a list of all overlay networks across all drivers
func (m *OverlayManager) ListNetworks(ctx context.Context) ([]OverlayNetwork, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	var allNetworks []OverlayNetwork
	for _, driver := range m.drivers {
		networks, err := driver.ListNetworks(ctx)
		if err != nil {
			return nil, fmt.Errorf("failed to list networks for driver %s: %v", driver.Name(), err)
		}
		allNetworks = append(allNetworks, networks...)
	}

	return allNetworks, nil
}

// CreateEndpoint creates a new endpoint in an overlay network
func (m *OverlayManager) CreateEndpoint(ctx context.Context, endpoint EndpointConfig) error {
	m.mu.RLock()
	driverName, exists := m.networkDrivers[endpoint.NetworkID]
	if !exists {
		m.mu.RUnlock()
		return fmt.Errorf("network %s not found", endpoint.NetworkID)
	}

	driver, exists := m.drivers[driverName]
	m.mu.RUnlock()

	if !exists {
		return fmt.Errorf("driver %s not found for network %s", driverName, endpoint.NetworkID)
	}

	return driver.CreateEndpoint(ctx, endpoint)
}

// DeleteEndpoint deletes an endpoint from an overlay network
func (m *OverlayManager) DeleteEndpoint(ctx context.Context, networkID, endpointName string) error {
	m.mu.RLock()
	driverName, exists := m.networkDrivers[networkID]
	if !exists {
		m.mu.RUnlock()
		return fmt.Errorf("network %s not found", networkID)
	}

	driver, exists := m.drivers[driverName]
	m.mu.RUnlock()

	if !exists {
		return fmt.Errorf("driver %s not found for network %s", driverName, networkID)
	}

	return driver.DeleteEndpoint(ctx, networkID, endpointName)
}

// GetEndpoint returns information about an endpoint
func (m *OverlayManager) GetEndpoint(ctx context.Context, networkID, endpointName string) (EndpointConfig, error) {
	m.mu.RLock()
	driverName, exists := m.networkDrivers[networkID]
	if !exists {
		m.mu.RUnlock()
		return EndpointConfig{}, fmt.Errorf("network %s not found", networkID)
	}

	driver, exists := m.drivers[driverName]
	m.mu.RUnlock()

	if !exists {
		return EndpointConfig{}, fmt.Errorf("driver %s not found for network %s", driverName, networkID)
	}

	return driver.GetEndpoint(ctx, networkID, endpointName)
}

// ListEndpoints returns a list of all endpoints in an overlay network
func (m *OverlayManager) ListEndpoints(ctx context.Context, networkID string) ([]EndpointConfig, error) {
	m.mu.RLock()
	driverName, exists := m.networkDrivers[networkID]
	if !exists {
		m.mu.RUnlock()
		return nil, fmt.Errorf("network %s not found", networkID)
	}

	driver, exists := m.drivers[driverName]
	m.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("driver %s not found for network %s", driverName, networkID)
	}

	return driver.ListEndpoints(ctx, networkID)
}

// ApplyNetworkPolicy applies a network policy to an overlay network
func (m *OverlayManager) ApplyNetworkPolicy(ctx context.Context, policy NetworkPolicy) error {
	m.mu.RLock()
	driverName, exists := m.networkDrivers[policy.NetworkID]
	if !exists {
		m.mu.RUnlock()
		return fmt.Errorf("network %s not found", policy.NetworkID)
	}

	driver, exists := m.drivers[driverName]
	m.mu.RUnlock()

	if !exists {
		return fmt.Errorf("driver %s not found for network %s", driverName, policy.NetworkID)
	}

	// Check if the driver supports network policies
	caps := driver.Capabilities()
	if !caps.SupportsNetworkPolicies {
		return fmt.Errorf("driver %s does not support network policies", driverName)
	}

	return driver.ApplyNetworkPolicy(ctx, policy.NetworkID, policy)
}

// RemoveNetworkPolicy removes a network policy from an overlay network
func (m *OverlayManager) RemoveNetworkPolicy(ctx context.Context, networkID, policyID string) error {
	m.mu.RLock()
	driverName, exists := m.networkDrivers[networkID]
	if !exists {
		m.mu.RUnlock()
		return fmt.Errorf("network %s not found", networkID)
	}

	driver, exists := m.drivers[driverName]
	m.mu.RUnlock()

	if !exists {
		return fmt.Errorf("driver %s not found for network %s", driverName, networkID)
	}

	// Check if the driver supports network policies
	caps := driver.Capabilities()
	if !caps.SupportsNetworkPolicies {
		return fmt.Errorf("driver %s does not support network policies", driverName)
	}

	return driver.RemoveNetworkPolicy(ctx, networkID, policyID)
}

// GetNetworkDriverCapabilities returns the capabilities of a driver
func (m *OverlayManager) GetNetworkDriverCapabilities(driverName string) (DriverCapabilities, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	driver, exists := m.drivers[driverName]
	if !exists {
		return DriverCapabilities{}, fmt.Errorf("driver %s not found", driverName)
	}

	return driver.Capabilities(), nil
}

// ListDrivers returns a list of all registered drivers
func (m *OverlayManager) ListDrivers() []string {
	m.mu.RLock()
	defer m.mu.RUnlock()

	var drivers []string
	for name := range m.drivers {
		drivers = append(drivers, name)
	}
	return drivers
}

// Shutdown shuts down the overlay manager and all drivers
func (m *OverlayManager) Shutdown(ctx context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if !m.initialized {
		return nil
	}

	var errs []error
	for name, driver := range m.drivers {
		if err := driver.Shutdown(ctx); err != nil {
			errs = append(errs, fmt.Errorf("failed to shutdown driver %s: %v", name, err))
		}
	}

	if len(errs) > 0 {
		return fmt.Errorf("errors shutting down drivers: %v", errs)
	}

	m.initialized = false
	return nil
}

// DriverFactory is a function that creates a new driver instance
type DriverFactory func() (OverlayDriver, error)

// DriverRegistry maintains a registry of driver factories
var DriverRegistry = make(map[string]DriverFactory)

// RegisterDriverFactory registers a driver factory
func RegisterDriverFactory(name string, factory DriverFactory) {
	DriverRegistry[name] = factory
}

// GetDriverFactory returns a driver factory by name
func GetDriverFactory(name string) (DriverFactory, bool) {
	factory, exists := DriverRegistry[name]
	return factory, exists
}
