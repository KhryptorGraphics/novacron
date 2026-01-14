package tenant

import (
	"context"
	"fmt"
	"log"
	"net"
	"sync"
	"time"

	"github.com/google/uuid"
)

// TenantNetworkType represents the type of network isolation
type TenantNetworkType string

const (
	// VXLANIsolation uses VXLAN overlays for tenant isolation
	VXLANIsolation TenantNetworkType = "vxlan"
	// GENEVEIsolation uses GENEVE overlays for tenant isolation
	GENEVEIsolation TenantNetworkType = "geneve"
	// VLANIsolation uses VLAN tagging for tenant isolation
	VLANIsolation TenantNetworkType = "vlan"
	// NamespaceIsolation uses Linux network namespaces
	NamespaceIsolation TenantNetworkType = "namespace"
	// VRFIsolation uses Virtual Routing and Forwarding
	VRFIsolation TenantNetworkType = "vrf"
)

// TenantStatus represents the operational status of a tenant
type TenantStatus string

const (
	TenantStatusActive      TenantStatus = "active"
	TenantStatusProvisioning TenantStatus = "provisioning"
	TenantStatusSuspended   TenantStatus = "suspended"
	TenantStatusTerminating TenantStatus = "terminating"
	TenantStatusTerminated  TenantStatus = "terminated"
	TenantStatusError       TenantStatus = "error"
)

// TenantQuotas defines resource quotas for a tenant
type TenantQuotas struct {
	MaxNetworks      int64   `json:"max_networks"`
	MaxSubnets       int64   `json:"max_subnets"`
	MaxPorts         int64   `json:"max_ports"`
	MaxFloatingIPs   int64   `json:"max_floating_ips"`
	MaxRouters       int64   `json:"max_routers"`
	MaxSecurityGroups int64  `json:"max_security_groups"`
	MaxLoadBalancers int64   `json:"max_load_balancers"`
	MaxBandwidthMbps int64   `json:"max_bandwidth_mbps"`
	MaxPolicies      int64   `json:"max_policies"`
	MaxVNIs          int64   `json:"max_vnis"`
}

// TenantResourceUsage tracks current resource usage for a tenant
type TenantResourceUsage struct {
	NetworksUsed        int64 `json:"networks_used"`
	SubnetsUsed         int64 `json:"subnets_used"`
	PortsUsed           int64 `json:"ports_used"`
	FloatingIPsUsed     int64 `json:"floating_ips_used"`
	RoutersUsed         int64 `json:"routers_used"`
	SecurityGroupsUsed  int64 `json:"security_groups_used"`
	LoadBalancersUsed   int64 `json:"load_balancers_used"`
	BandwidthUsedMbps   int64 `json:"bandwidth_used_mbps"`
	PoliciesUsed        int64 `json:"policies_used"`
	VNIsUsed            int64 `json:"vnis_used"`
	LastUpdated         time.Time `json:"last_updated"`
}

// TenantNetwork represents a network within a tenant
type TenantNetwork struct {
	ID                string            `json:"id"`
	Name              string            `json:"name"`
	TenantID          string            `json:"tenant_id"`
	Type              TenantNetworkType `json:"type"`
	VNI               uint32            `json:"vni"`
	VLANID            uint16            `json:"vlan_id,omitempty"`
	CIDR              string            `json:"cidr"`
	Gateway           string            `json:"gateway"`
	DNSServers        []string          `json:"dns_servers"`
	Subnets           []TenantSubnet    `json:"subnets"`
	SecurityGroups    []string          `json:"security_groups"`
	QoSPolicyID       string            `json:"qos_policy_id,omitempty"`
	RoutingTableID    string            `json:"routing_table_id,omitempty"`
	DHCPEnabled       bool              `json:"dhcp_enabled"`
	IPv6Enabled       bool              `json:"ipv6_enabled"`
	ExternalAccess    bool              `json:"external_access"`
	InterTenantAccess bool              `json:"inter_tenant_access"`
	Metadata          map[string]string `json:"metadata,omitempty"`
	Status            string            `json:"status"`
	CreatedAt         time.Time         `json:"created_at"`
	UpdatedAt         time.Time         `json:"updated_at"`
}

// TenantSubnet represents a subnet within a tenant network
type TenantSubnet struct {
	ID              string    `json:"id"`
	NetworkID       string    `json:"network_id"`
	Name            string    `json:"name"`
	CIDR            string    `json:"cidr"`
	Gateway         string    `json:"gateway"`
	AllocationPools []IPPool  `json:"allocation_pools"`
	DNSServers      []string  `json:"dns_servers"`
	HostRoutes      []Route   `json:"host_routes"`
	DHCPEnabled     bool      `json:"dhcp_enabled"`
	IPv6Enabled     bool      `json:"ipv6_enabled"`
	CreatedAt       time.Time `json:"created_at"`
	UpdatedAt       time.Time `json:"updated_at"`
}

// IPPool represents an IP allocation pool
type IPPool struct {
	Start string `json:"start"`
	End   string `json:"end"`
}

// Route represents a static route
type Route struct {
	Destination string `json:"destination"`
	NextHop     string `json:"next_hop"`
}

// TenantPort represents a network port within a tenant
type TenantPort struct {
	ID               string            `json:"id"`
	NetworkID        string            `json:"network_id"`
	SubnetID         string            `json:"subnet_id"`
	Name             string            `json:"name"`
	MACAddress       string            `json:"mac_address"`
	IPAddresses      []string          `json:"ip_addresses"`
	DeviceID         string            `json:"device_id,omitempty"`
	DeviceType       string            `json:"device_type,omitempty"`
	SecurityGroups   []string          `json:"security_groups"`
	PortSecurity     bool              `json:"port_security"`
	QoSPolicyID      string            `json:"qos_policy_id,omitempty"`
	FixedIPs         []FixedIP         `json:"fixed_ips"`
	AllowedAddressPairs []AddressPair `json:"allowed_address_pairs"`
	ExtraDHCPOpts    []DHCPOption      `json:"extra_dhcp_opts"`
	Metadata         map[string]string `json:"metadata,omitempty"`
	Status           string            `json:"status"`
	CreatedAt        time.Time         `json:"created_at"`
	UpdatedAt        time.Time         `json:"updated_at"`
}

// FixedIP represents a fixed IP assignment
type FixedIP struct {
	SubnetID  string `json:"subnet_id"`
	IPAddress string `json:"ip_address"`
}

// AddressPair represents an allowed address pair
type AddressPair struct {
	IPAddress  string `json:"ip_address"`
	MACAddress string `json:"mac_address"`
}

// DHCPOption represents a DHCP option
type DHCPOption struct {
	OptName  string `json:"opt_name"`
	OptValue string `json:"opt_value"`
}

// Tenant represents a network tenant with complete isolation
type Tenant struct {
	ID                string                 `json:"id"`
	Name              string                 `json:"name"`
	Description       string                 `json:"description"`
	Status            TenantStatus           `json:"status"`
	IsolationType     TenantNetworkType      `json:"isolation_type"`
	VNIRange          VNIRange              `json:"vni_range"`
	VLANRange         VLANRange             `json:"vlan_range,omitempty"`
	AddressSpace      []string              `json:"address_space"`
	Quotas            TenantQuotas          `json:"quotas"`
	ResourceUsage     TenantResourceUsage   `json:"resource_usage"`
	Networks          map[string]*TenantNetwork `json:"networks"`
	Ports             map[string]*TenantPort `json:"ports"`
	SecurityGroups    []string              `json:"security_groups"`
	QoSPolicies       []string              `json:"qos_policies"`
	RoutingTables     []string              `json:"routing_tables"`
	DefaultRouterID   string                `json:"default_router_id,omitempty"`
	ExternalNetworkID string                `json:"external_network_id,omitempty"`
	Tags              []string              `json:"tags,omitempty"`
	Metadata          map[string]string     `json:"metadata,omitempty"`
	CreatedAt         time.Time             `json:"created_at"`
	UpdatedAt         time.Time             `json:"updated_at"`
	CreatedBy         string                `json:"created_by"`
	ProjectID         string                `json:"project_id,omitempty"`
}

// VNIRange represents a range of VNI (VXLAN Network Identifier) values
type VNIRange struct {
	Start uint32 `json:"start"`
	End   uint32 `json:"end"`
	Used  []uint32 `json:"used"`
}

// VLANRange represents a range of VLAN ID values
type VLANRange struct {
	Start uint16 `json:"start"`
	End   uint16 `json:"end"`
	Used  []uint16 `json:"used"`
}

// TenantEvent represents tenant-related events
type TenantEvent struct {
	Type      string      `json:"type"`
	TenantID  string      `json:"tenant_id"`
	ResourceID string     `json:"resource_id,omitempty"`
	Data      interface{} `json:"data"`
	Timestamp time.Time   `json:"timestamp"`
	UserID    string      `json:"user_id,omitempty"`
}

// TenantEventListener is a callback for tenant events
type TenantEventListener func(event TenantEvent)

// TenantManagerConfig holds configuration for the tenant manager
type TenantManagerConfig struct {
	DefaultIsolationType  TenantNetworkType `json:"default_isolation_type"`
	VNIRangeStart        uint32            `json:"vni_range_start"`
	VNIRangeEnd          uint32            `json:"vni_range_end"`
	VLANRangeStart       uint16            `json:"vlan_range_start"`
	VLANRangeEnd         uint16            `json:"vlan_range_end"`
	DefaultAddressSpace  []string          `json:"default_address_space"`
	DefaultQuotas        TenantQuotas      `json:"default_quotas"`
	EnableResourceQuotas bool              `json:"enable_resource_quotas"`
	EnableMetrics        bool              `json:"enable_metrics"`
	MetricsInterval      time.Duration     `json:"metrics_interval"`
}

// DefaultTenantManagerConfig returns default configuration
func DefaultTenantManagerConfig() *TenantManagerConfig {
	return &TenantManagerConfig{
		DefaultIsolationType: VXLANIsolation,
		VNIRangeStart:       10000,
		VNIRangeEnd:         99999,
		VLANRangeStart:      100,
		VLANRangeEnd:        4000,
		DefaultAddressSpace: []string{"10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"},
		DefaultQuotas: TenantQuotas{
			MaxNetworks:       100,
			MaxSubnets:        500,
			MaxPorts:          2000,
			MaxFloatingIPs:    50,
			MaxRouters:        10,
			MaxSecurityGroups: 100,
			MaxLoadBalancers:  20,
			MaxBandwidthMbps:  1000,
			MaxPolicies:       200,
			MaxVNIs:           100,
		},
		EnableResourceQuotas: true,
		EnableMetrics:        true,
		MetricsInterval:      30 * time.Second,
	}
}

// TenantManager manages network tenants and their isolation
type TenantManager struct {
	config         *TenantManagerConfig
	tenants        map[string]*Tenant
	vniAllocator   *VNIAllocator
	vlanAllocator  *VLANAllocator
	eventListeners []TenantEventListener
	metrics        *TenantMetrics
	mutex          sync.RWMutex
	ctx            context.Context
	cancel         context.CancelFunc
	wg             sync.WaitGroup
}

// VNIAllocator manages VNI allocation across tenants
type VNIAllocator struct {
	rangeStart uint32
	rangeEnd   uint32
	allocated  map[uint32]string // VNI -> TenantID
	mutex      sync.Mutex
}

// VLANAllocator manages VLAN ID allocation across tenants  
type VLANAllocator struct {
	rangeStart uint16
	rangeEnd   uint16
	allocated  map[uint16]string // VLAN ID -> TenantID
	mutex      sync.Mutex
}

// TenantMetrics holds tenant management metrics
type TenantMetrics struct {
	TotalTenants        int64     `json:"total_tenants"`
	ActiveTenants       int64     `json:"active_tenants"`
	TotalNetworks       int64     `json:"total_networks"`
	TotalPorts          int64     `json:"total_ports"`
	VNIsAllocated       int64     `json:"vnis_allocated"`
	VLANsAllocated      int64     `json:"vlans_allocated"`
	QuotaViolations     int64     `json:"quota_violations"`
	ResourceUtilization float64   `json:"resource_utilization"`
	LastUpdated         time.Time `json:"last_updated"`
}

// NewTenantManager creates a new tenant manager
func NewTenantManager(config *TenantManagerConfig) *TenantManager {
	ctx, cancel := context.WithCancel(context.Background())
	
	tm := &TenantManager{
		config:         config,
		tenants:        make(map[string]*Tenant),
		eventListeners: make([]TenantEventListener, 0),
		ctx:            ctx,
		cancel:         cancel,
		metrics: &TenantMetrics{
			LastUpdated: time.Now(),
		},
	}
	
	// Initialize allocators
	tm.vniAllocator = &VNIAllocator{
		rangeStart: config.VNIRangeStart,
		rangeEnd:   config.VNIRangeEnd,
		allocated:  make(map[uint32]string),
	}
	
	tm.vlanAllocator = &VLANAllocator{
		rangeStart: config.VLANRangeStart,
		rangeEnd:   config.VLANRangeEnd,
		allocated:  make(map[uint16]string),
	}
	
	return tm
}

// Start starts the tenant manager
func (tm *TenantManager) Start() error {
	log.Println("Starting tenant manager")
	
	// Start metrics collection if enabled
	if tm.config.EnableMetrics {
		tm.wg.Add(1)
		go tm.collectMetrics()
	}
	
	tm.emitEvent(TenantEvent{
		Type:      "tenant_manager_started",
		Timestamp: time.Now(),
	})
	
	return nil
}

// Stop stops the tenant manager
func (tm *TenantManager) Stop() error {
	log.Println("Stopping tenant manager")
	
	tm.cancel()
	tm.wg.Wait()
	
	tm.emitEvent(TenantEvent{
		Type:      "tenant_manager_stopped",
		Timestamp: time.Now(),
	})
	
	return nil
}

// CreateTenant creates a new tenant with network isolation
func (tm *TenantManager) CreateTenant(ctx context.Context, name, description, createdBy string, isolationType TenantNetworkType) (*Tenant, error) {
	tm.mutex.Lock()
	defer tm.mutex.Unlock()
	
	// Generate tenant ID
	tenantID := uuid.New().String()
	
	// Check if tenant name already exists
	for _, tenant := range tm.tenants {
		if tenant.Name == name {
			return nil, fmt.Errorf("tenant with name '%s' already exists", name)
		}
	}
	
	// Use default isolation type if not specified
	if isolationType == "" {
		isolationType = tm.config.DefaultIsolationType
	}
	
	// Allocate VNI range for tenant
	vniRange, err := tm.allocateVNIRange(tenantID)
	if err != nil {
		return nil, fmt.Errorf("failed to allocate VNI range: %w", err)
	}
	
	// Allocate VLAN range if using VLAN isolation
	var vlanRange VLANRange
	if isolationType == VLANIsolation {
		vlanRange, err = tm.allocateVLANRange(tenantID)
		if err != nil {
			// Clean up VNI allocation
			tm.deallocateVNIRange(tenantID, vniRange)
			return nil, fmt.Errorf("failed to allocate VLAN range: %w", err)
		}
	}
	
	// Create tenant
	tenant := &Tenant{
		ID:              tenantID,
		Name:            name,
		Description:     description,
		Status:          TenantStatusProvisioning,
		IsolationType:   isolationType,
		VNIRange:        vniRange,
		VLANRange:       vlanRange,
		AddressSpace:    tm.config.DefaultAddressSpace,
		Quotas:          tm.config.DefaultQuotas,
		ResourceUsage:   TenantResourceUsage{LastUpdated: time.Now()},
		Networks:        make(map[string]*TenantNetwork),
		Ports:           make(map[string]*TenantPort),
		SecurityGroups:  make([]string, 0),
		QoSPolicies:     make([]string, 0),
		RoutingTables:   make([]string, 0),
		Tags:            make([]string, 0),
		Metadata:        make(map[string]string),
		CreatedAt:       time.Now(),
		UpdatedAt:       time.Now(),
		CreatedBy:       createdBy,
	}
	
	// Initialize tenant networking
	if err := tm.initializeTenantNetworking(ctx, tenant); err != nil {
		// Clean up allocations
		tm.deallocateVNIRange(tenantID, vniRange)
		if isolationType == VLANIsolation {
			tm.deallocateVLANRange(tenantID, vlanRange)
		}
		return nil, fmt.Errorf("failed to initialize tenant networking: %w", err)
	}
	
	tenant.Status = TenantStatusActive
	tenant.UpdatedAt = time.Now()
	
	// Store tenant
	tm.tenants[tenantID] = tenant
	
	// Update metrics
	tm.metrics.TotalTenants++
	tm.metrics.ActiveTenants++
	
	tm.emitEvent(TenantEvent{
		Type:      "tenant_created",
		TenantID:  tenantID,
		Data:      tenant,
		Timestamp: time.Now(),
		UserID:    createdBy,
	})
	
	log.Printf("Created tenant %s (%s) with %s isolation", name, tenantID, isolationType)
	return tenant, nil
}

// DeleteTenant deletes a tenant and all its resources
func (tm *TenantManager) DeleteTenant(ctx context.Context, tenantID string) error {
	tm.mutex.Lock()
	defer tm.mutex.Unlock()
	
	tenant, exists := tm.tenants[tenantID]
	if !exists {
		return fmt.Errorf("tenant %s not found", tenantID)
	}
	
	tenant.Status = TenantStatusTerminating
	tenant.UpdatedAt = time.Now()
	
	// Clean up tenant resources
	if err := tm.cleanupTenantResources(ctx, tenant); err != nil {
		tenant.Status = TenantStatusError
		return fmt.Errorf("failed to cleanup tenant resources: %w", err)
	}
	
	// Deallocate VNI and VLAN ranges
	tm.deallocateVNIRange(tenantID, tenant.VNIRange)
	if tenant.IsolationType == VLANIsolation {
		tm.deallocateVLANRange(tenantID, tenant.VLANRange)
	}
	
	tenant.Status = TenantStatusTerminated
	
	// Remove from tenants map
	delete(tm.tenants, tenantID)
	
	// Update metrics
	tm.metrics.TotalTenants--
	if tenant.Status == TenantStatusActive {
		tm.metrics.ActiveTenants--
	}
	
	tm.emitEvent(TenantEvent{
		Type:      "tenant_deleted",
		TenantID:  tenantID,
		Data:      tenant,
		Timestamp: time.Now(),
	})
	
	log.Printf("Deleted tenant %s (%s)", tenant.Name, tenantID)
	return nil
}

// GetTenant retrieves a tenant by ID
func (tm *TenantManager) GetTenant(tenantID string) (*Tenant, error) {
	tm.mutex.RLock()
	defer tm.mutex.RUnlock()
	
	tenant, exists := tm.tenants[tenantID]
	if !exists {
		return nil, fmt.Errorf("tenant %s not found", tenantID)
	}
	
	return tenant, nil
}

// ListTenants returns all tenants
func (tm *TenantManager) ListTenants() []*Tenant {
	tm.mutex.RLock()
	defer tm.mutex.RUnlock()
	
	tenants := make([]*Tenant, 0, len(tm.tenants))
	for _, tenant := range tm.tenants {
		tenants = append(tenants, tenant)
	}
	
	return tenants
}

// UpdateTenantQuotas updates resource quotas for a tenant
func (tm *TenantManager) UpdateTenantQuotas(tenantID string, quotas TenantQuotas) error {
	tm.mutex.Lock()
	defer tm.mutex.Unlock()
	
	tenant, exists := tm.tenants[tenantID]
	if !exists {
		return fmt.Errorf("tenant %s not found", tenantID)
	}
	
	// Validate quotas don't exceed current usage
	if err := tm.validateQuotas(tenant, quotas); err != nil {
		return fmt.Errorf("quota validation failed: %w", err)
	}
	
	tenant.Quotas = quotas
	tenant.UpdatedAt = time.Now()
	
	tm.emitEvent(TenantEvent{
		Type:       "tenant_quotas_updated",
		TenantID:   tenantID,
		Data:       quotas,
		Timestamp:  time.Now(),
	})
	
	return nil
}

// CreateTenantNetwork creates a network within a tenant
func (tm *TenantManager) CreateTenantNetwork(ctx context.Context, tenantID, name, cidr string, options map[string]interface{}) (*TenantNetwork, error) {
	tm.mutex.Lock()
	defer tm.mutex.Unlock()
	
	tenant, exists := tm.tenants[tenantID]
	if !exists {
		return nil, fmt.Errorf("tenant %s not found", tenantID)
	}
	
	// Check quotas
	if tm.config.EnableResourceQuotas {
		if tenant.ResourceUsage.NetworksUsed >= tenant.Quotas.MaxNetworks {
			return nil, fmt.Errorf("network quota exceeded for tenant %s", tenantID)
		}
	}
	
	// Validate CIDR
	_, ipNet, err := net.ParseCIDR(cidr)
	if err != nil {
		return nil, fmt.Errorf("invalid CIDR: %s", cidr)
	}
	
	// Check if CIDR conflicts with existing networks
	for _, network := range tenant.Networks {
		if tm.cidrOverlaps(cidr, network.CIDR) {
			return nil, fmt.Errorf("CIDR %s overlaps with existing network %s", cidr, network.CIDR)
		}
	}
	
	// Allocate VNI for the network
	vni, err := tm.allocateVNI(tenantID)
	if err != nil {
		return nil, fmt.Errorf("failed to allocate VNI: %w", err)
	}
	
	// Allocate VLAN ID if using VLAN isolation
	var vlanID uint16
	if tenant.IsolationType == VLANIsolation {
		vlanID, err = tm.allocateVLAN(tenantID)
		if err != nil {
			tm.deallocateVNI(tenantID, vni)
			return nil, fmt.Errorf("failed to allocate VLAN ID: %w", err)
		}
	}
	
	// Calculate gateway IP
	gateway := tm.calculateGatewayIP(ipNet)
	
	// Create network
	networkID := uuid.New().String()
	network := &TenantNetwork{
		ID:                networkID,
		Name:              name,
		TenantID:          tenantID,
		Type:              tenant.IsolationType,
		VNI:               vni,
		VLANID:            vlanID,
		CIDR:              cidr,
		Gateway:           gateway,
		DNSServers:        []string{"8.8.8.8", "8.8.4.4"},
		Subnets:           make([]TenantSubnet, 0),
		SecurityGroups:    make([]string, 0),
		DHCPEnabled:       true,
		IPv6Enabled:       false,
		ExternalAccess:    false,
		InterTenantAccess: false,
		Metadata:          make(map[string]string),
		Status:            "active",
		CreatedAt:         time.Now(),
		UpdatedAt:         time.Now(),
	}
	
	// Apply options
	tm.applyNetworkOptions(network, options)
	
	// Create the actual network infrastructure
	if err := tm.createNetworkInfrastructure(ctx, tenant, network); err != nil {
		tm.deallocateVNI(tenantID, vni)
		if tenant.IsolationType == VLANIsolation {
			tm.deallocateVLAN(tenantID, vlanID)
		}
		return nil, fmt.Errorf("failed to create network infrastructure: %w", err)
	}
	
	// Store network
	tenant.Networks[networkID] = network
	tenant.ResourceUsage.NetworksUsed++
	tenant.ResourceUsage.LastUpdated = time.Now()
	tenant.UpdatedAt = time.Now()
	
	tm.emitEvent(TenantEvent{
		Type:       "tenant_network_created",
		TenantID:   tenantID,
		ResourceID: networkID,
		Data:       network,
		Timestamp:  time.Now(),
	})
	
	log.Printf("Created network %s (%s) in tenant %s", name, networkID, tenantID)
	return network, nil
}

// Helper methods

func (tm *TenantManager) allocateVNIRange(tenantID string) (VNIRange, error) {
	tm.vniAllocator.mutex.Lock()
	defer tm.vniAllocator.mutex.Unlock()
	
	// Allocate a range of VNIs for this tenant (e.g., 100 VNIs)
	rangeSize := uint32(100)
	start := tm.vniAllocator.rangeStart
	
	// Find available range
	for start+rangeSize <= tm.vniAllocator.rangeEnd {
		available := true
		for i := start; i < start+rangeSize; i++ {
			if _, exists := tm.vniAllocator.allocated[i]; exists {
				available = false
				break
			}
		}
		
		if available {
			// Allocate the range
			for i := start; i < start+rangeSize; i++ {
				tm.vniAllocator.allocated[i] = tenantID
			}
			
			return VNIRange{
				Start: start,
				End:   start + rangeSize - 1,
				Used:  make([]uint32, 0),
			}, nil
		}
		
		start += rangeSize
	}
	
	return VNIRange{}, fmt.Errorf("no VNI range available")
}

func (tm *TenantManager) allocateVLANRange(tenantID string) (VLANRange, error) {
	tm.vlanAllocator.mutex.Lock()
	defer tm.vlanAllocator.mutex.Unlock()
	
	// Allocate a range of VLAN IDs for this tenant (e.g., 50 VLANs)
	rangeSize := uint16(50)
	start := tm.vlanAllocator.rangeStart
	
	// Find available range
	for start+rangeSize <= tm.vlanAllocator.rangeEnd {
		available := true
		for i := start; i < start+rangeSize; i++ {
			if _, exists := tm.vlanAllocator.allocated[i]; exists {
				available = false
				break
			}
		}
		
		if available {
			// Allocate the range
			for i := start; i < start+rangeSize; i++ {
				tm.vlanAllocator.allocated[i] = tenantID
			}
			
			return VLANRange{
				Start: start,
				End:   start + rangeSize - 1,
				Used:  make([]uint16, 0),
			}, nil
		}
		
		start += rangeSize
	}
	
	return VLANRange{}, fmt.Errorf("no VLAN range available")
}

func (tm *TenantManager) deallocateVNIRange(tenantID string, vniRange VNIRange) {
	tm.vniAllocator.mutex.Lock()
	defer tm.vniAllocator.mutex.Unlock()
	
	for i := vniRange.Start; i <= vniRange.End; i++ {
		if tm.vniAllocator.allocated[i] == tenantID {
			delete(tm.vniAllocator.allocated, i)
		}
	}
}

func (tm *TenantManager) deallocateVLANRange(tenantID string, vlanRange VLANRange) {
	tm.vlanAllocator.mutex.Lock()
	defer tm.vlanAllocator.mutex.Unlock()
	
	for i := vlanRange.Start; i <= vlanRange.End; i++ {
		if tm.vlanAllocator.allocated[i] == tenantID {
			delete(tm.vlanAllocator.allocated, i)
		}
	}
}

func (tm *TenantManager) allocateVNI(tenantID string) (uint32, error) {
	tenant, exists := tm.tenants[tenantID]
	if !exists {
		return 0, fmt.Errorf("tenant not found")
	}
	
	// Find available VNI in tenant's range
	for vni := tenant.VNIRange.Start; vni <= tenant.VNIRange.End; vni++ {
		used := false
		for _, usedVNI := range tenant.VNIRange.Used {
			if usedVNI == vni {
				used = true
				break
			}
		}
		
		if !used {
			tenant.VNIRange.Used = append(tenant.VNIRange.Used, vni)
			return vni, nil
		}
	}
	
	return 0, fmt.Errorf("no VNI available in tenant range")
}

func (tm *TenantManager) allocateVLAN(tenantID string) (uint16, error) {
	tenant, exists := tm.tenants[tenantID]
	if !exists {
		return 0, fmt.Errorf("tenant not found")
	}
	
	// Find available VLAN ID in tenant's range
	for vlanID := tenant.VLANRange.Start; vlanID <= tenant.VLANRange.End; vlanID++ {
		used := false
		for _, usedVLAN := range tenant.VLANRange.Used {
			if usedVLAN == vlanID {
				used = true
				break
			}
		}
		
		if !used {
			tenant.VLANRange.Used = append(tenant.VLANRange.Used, vlanID)
			return vlanID, nil
		}
	}
	
	return 0, fmt.Errorf("no VLAN ID available in tenant range")
}

func (tm *TenantManager) deallocateVNI(tenantID string, vni uint32) {
	tenant, exists := tm.tenants[tenantID]
	if !exists {
		return
	}
	
	// Remove VNI from used list
	for i, usedVNI := range tenant.VNIRange.Used {
		if usedVNI == vni {
			tenant.VNIRange.Used = append(tenant.VNIRange.Used[:i], tenant.VNIRange.Used[i+1:]...)
			break
		}
	}
}

func (tm *TenantManager) deallocateVLAN(tenantID string, vlanID uint16) {
	tenant, exists := tm.tenants[tenantID]
	if !exists {
		return
	}
	
	// Remove VLAN ID from used list
	for i, usedVLAN := range tenant.VLANRange.Used {
		if usedVLAN == vlanID {
			tenant.VLANRange.Used = append(tenant.VLANRange.Used[:i], tenant.VLANRange.Used[i+1:]...)
			break
		}
	}
}

func (tm *TenantManager) initializeTenantNetworking(ctx context.Context, tenant *Tenant) error {
	// This would initialize the actual networking infrastructure for the tenant
	// For now, we'll just simulate the initialization
	
	log.Printf("Initializing networking for tenant %s with %s isolation", 
		tenant.ID, tenant.IsolationType)
		
	// In a real implementation, this would:
	// 1. Create network namespaces if using namespace isolation
	// 2. Set up VXLAN/GENEVE tunnels if using overlay isolation
	// 3. Configure VLAN interfaces if using VLAN isolation
	// 4. Set up routing tables and rules
	// 5. Initialize security groups and firewall rules
	
	return nil
}

func (tm *TenantManager) cleanupTenantResources(ctx context.Context, tenant *Tenant) error {
	// Clean up all tenant resources
	log.Printf("Cleaning up resources for tenant %s", tenant.ID)
	
	// In a real implementation, this would:
	// 1. Delete all tenant networks and their infrastructure
	// 2. Clean up routing tables and rules
	// 3. Remove security groups and firewall rules
	// 4. Delete network namespaces or VLAN interfaces
	// 5. Clean up any remaining network state
	
	return nil
}

func (tm *TenantManager) validateQuotas(tenant *Tenant, quotas TenantQuotas) error {
	usage := tenant.ResourceUsage
	
	if usage.NetworksUsed > quotas.MaxNetworks {
		return fmt.Errorf("networks quota too low: current usage %d, requested quota %d", 
			usage.NetworksUsed, quotas.MaxNetworks)
	}
	
	if usage.PortsUsed > quotas.MaxPorts {
		return fmt.Errorf("ports quota too low: current usage %d, requested quota %d", 
			usage.PortsUsed, quotas.MaxPorts)
	}
	
	// Add more validation as needed
	
	return nil
}

func (tm *TenantManager) cidrOverlaps(cidr1, cidr2 string) bool {
	_, net1, err1 := net.ParseCIDR(cidr1)
	_, net2, err2 := net.ParseCIDR(cidr2)
	
	if err1 != nil || err2 != nil {
		return false
	}
	
	return net1.Contains(net2.IP) || net2.Contains(net1.IP)
}

func (tm *TenantManager) calculateGatewayIP(ipNet *net.IPNet) string {
	// Use the first available IP as gateway
	ip := ipNet.IP.To4()
	if ip == nil {
		// IPv6 - use the first IP
		return ipNet.IP.String()
	}
	
	// IPv4 - increment by 1
	ip[3]++
	return ip.String()
}

func (tm *TenantManager) applyNetworkOptions(network *TenantNetwork, options map[string]interface{}) {
	if dns, ok := options["dns_servers"].([]string); ok {
		network.DNSServers = dns
	}
	
	if dhcp, ok := options["dhcp_enabled"].(bool); ok {
		network.DHCPEnabled = dhcp
	}
	
	if ipv6, ok := options["ipv6_enabled"].(bool); ok {
		network.IPv6Enabled = ipv6
	}
	
	if external, ok := options["external_access"].(bool); ok {
		network.ExternalAccess = external
	}
	
	if inter, ok := options["inter_tenant_access"].(bool); ok {
		network.InterTenantAccess = inter
	}
	
	if meta, ok := options["metadata"].(map[string]string); ok {
		network.Metadata = meta
	}
}

func (tm *TenantManager) createNetworkInfrastructure(ctx context.Context, tenant *Tenant, network *TenantNetwork) error {
	// Create the actual network infrastructure
	log.Printf("Creating network infrastructure for network %s in tenant %s", 
		network.ID, tenant.ID)
	
	// In a real implementation, this would:
	// 1. Create OVS bridges and ports
	// 2. Set up VXLAN/GENEVE tunnels
	// 3. Configure flow rules for isolation
	// 4. Set up DHCP server if enabled
	// 5. Configure routing and NAT rules
	
	return nil
}

func (tm *TenantManager) collectMetrics() {
	defer tm.wg.Done()
	
	ticker := time.NewTicker(tm.config.MetricsInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-tm.ctx.Done():
			return
		case <-ticker.C:
			tm.updateMetrics()
		}
	}
}

func (tm *TenantManager) updateMetrics() {
	tm.mutex.RLock()
	defer tm.mutex.RUnlock()
	
	tm.metrics.TotalTenants = int64(len(tm.tenants))
	
	activeTenants := int64(0)
	totalNetworks := int64(0)
	totalPorts := int64(0)
	vniAllocated := int64(0)
	vlanAllocated := int64(0)
	
	for _, tenant := range tm.tenants {
		if tenant.Status == TenantStatusActive {
			activeTenants++
		}
		totalNetworks += int64(len(tenant.Networks))
		totalPorts += int64(len(tenant.Ports))
		vniAllocated += int64(len(tenant.VNIRange.Used))
		vlanAllocated += int64(len(tenant.VLANRange.Used))
	}
	
	tm.metrics.ActiveTenants = activeTenants
	tm.metrics.TotalNetworks = totalNetworks
	tm.metrics.TotalPorts = totalPorts
	tm.metrics.VNIsAllocated = vniAllocated
	tm.metrics.VLANsAllocated = vlanAllocated
	tm.metrics.LastUpdated = time.Now()
	
	// Calculate resource utilization
	totalVNIs := int64(tm.vniAllocator.rangeEnd - tm.vniAllocator.rangeStart + 1)
	if totalVNIs > 0 {
		tm.metrics.ResourceUtilization = float64(vniAllocated) / float64(totalVNIs)
	}
}

func (tm *TenantManager) emitEvent(event TenantEvent) {
	for _, listener := range tm.eventListeners {
		go func(l TenantEventListener, e TenantEvent) {
			defer func() {
				if r := recover(); r != nil {
					log.Printf("Event listener panic: %v", r)
				}
			}()
			l(e)
		}(listener, event)
	}
}

// GetMetrics returns current tenant metrics
func (tm *TenantManager) GetMetrics() *TenantMetrics {
	tm.mutex.RLock()
	defer tm.mutex.RUnlock()
	
	// Return a copy to avoid race conditions
	metrics := *tm.metrics
	return &metrics
}

// AddEventListener adds a tenant event listener
func (tm *TenantManager) AddEventListener(listener TenantEventListener) {
	tm.mutex.Lock()
	defer tm.mutex.Unlock()
	
	tm.eventListeners = append(tm.eventListeners, listener)
}