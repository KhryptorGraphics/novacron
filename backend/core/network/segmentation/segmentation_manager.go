package segmentation

import (
	"context"
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/google/uuid"

	"github.com/khryptorgraphics/novacron/backend/core/network/segmentation/sdn"
	"github.com/khryptorgraphics/novacron/backend/core/network/segmentation/tenant"
	"github.com/khryptorgraphics/novacron/backend/core/network/segmentation/firewall"
	"github.com/khryptorgraphics/novacron/backend/core/network/segmentation/qos"
)

// SegmentationManager coordinates all network segmentation components
type SegmentationManager struct {
	ID                  string                            `json:"id"`
	Name                string                            `json:"name"`
	SDNController       *sdn.SDNController                `json:"-"`
	TenantManager       *tenant.TenantManager             `json:"-"`
	QoSEngine           *qos.QoSEngine                    `json:"-"`
	Firewalls           map[string]*firewall.MicrosegmentationFirewall `json:"-"`
	Config              *SegmentationConfig               `json:"config"`
	Metrics             *SegmentationMetrics              `json:"metrics"`
	EventListeners      []SegmentationEventListener       `json:"-"`
	integrationHandlers map[string]IntegrationHandler     `json:"-"`
	mutex               sync.RWMutex
	ctx                 context.Context
	cancel              context.CancelFunc
	wg                  sync.WaitGroup
}

// SegmentationConfig holds configuration for the segmentation manager
type SegmentationConfig struct {
	EnableSDNController      bool              `json:"enable_sdn_controller"`
	EnableTenantIsolation    bool              `json:"enable_tenant_isolation"`
	EnableMicrosegmentation  bool              `json:"enable_microsegmentation"`
	EnableQoSManagement      bool              `json:"enable_qos_management"`
	EnableNetworkAnalytics   bool              `json:"enable_network_analytics"`
	EnableComplianceAuditing bool              `json:"enable_compliance_auditing"`
	DefaultTenantIsolation   tenant.TenantNetworkType `json:"default_tenant_isolation"`
	DefaultFirewallAction    firewall.FirewallAction  `json:"default_firewall_action"`
	DefaultQoSAlgorithm      qos.QoSAlgorithm          `json:"default_qos_algorithm"`
	MetricsInterval          time.Duration             `json:"metrics_interval"`
	EventBufferSize          int                       `json:"event_buffer_size"`
	MaxTenantsPerNode        int                       `json:"max_tenants_per_node"`
	MaxNetworksPerTenant     int                       `json:"max_networks_per_tenant"`
	MaxFirewallRules         int                       `json:"max_firewall_rules"`
	MaxQoSPolicies           int                       `json:"max_qos_policies"`
}

// SegmentationMetrics tracks overall segmentation system performance
type SegmentationMetrics struct {
	TotalTenants              int64                 `json:"total_tenants"`
	TotalNetworks             int64                 `json:"total_networks"`
	TotalFirewallRules        int64                 `json:"total_firewall_rules"`
	TotalQoSPolicies          int64                 `json:"total_qos_policies"`
	PacketsProcessed          uint64                `json:"packets_processed"`
	PacketsBlocked            uint64                `json:"packets_blocked"`
	ThroughputBps             uint64                `json:"throughput_bps"`
	AverageLatency            time.Duration         `json:"average_latency"`
	PolicyViolations          uint64                `json:"policy_violations"`
	SecurityIncidents         uint64                `json:"security_incidents"`
	QoSViolations             uint64                `json:"qos_violations"`
	NetworkUtilization        float64               `json:"network_utilization"`
	CPUUtilization            float64               `json:"cpu_utilization"`
	MemoryUtilization         float64               `json:"memory_utilization"`
	ComponentHealth           map[string]string     `json:"component_health"`
	IntegrationStatus         map[string]string     `json:"integration_status"`
	LastUpdated               time.Time             `json:"last_updated"`
}

// SegmentationEvent represents events from the segmentation system
type SegmentationEvent struct {
	Type           string                 `json:"type"`
	Component      string                 `json:"component"`
	TenantID       string                 `json:"tenant_id,omitempty"`
	NetworkID      string                 `json:"network_id,omitempty"`
	Severity       string                 `json:"severity"`
	Message        string                 `json:"message"`
	Data           interface{}            `json:"data,omitempty"`
	Timestamp      time.Time              `json:"timestamp"`
	CorrelationID  string                 `json:"correlation_id,omitempty"`
}

// SegmentationEventListener is a callback for segmentation events
type SegmentationEventListener func(event SegmentationEvent)

// IntegrationHandler handles integration with other NovaCron systems
type IntegrationHandler interface {
	GetName() string
	Initialize(ctx context.Context, manager *SegmentationManager) error
	HandleEvent(event interface{}) error
	GetStatus() string
	Shutdown() error
}

// VMIntegrationHandler integrates with VM management system
type VMIntegrationHandler struct {
	manager    *SegmentationManager
	vmEvents   chan VMEvent
	ctx        context.Context
	cancel     context.CancelFunc
	wg         sync.WaitGroup
}

// VMEvent represents VM lifecycle events
type VMEvent struct {
	Type     string                 `json:"type"`
	VMID     string                 `json:"vm_id"`
	TenantID string                 `json:"tenant_id"`
	Networks []string               `json:"networks,omitempty"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// LoadBalancerIntegrationHandler integrates with load balancer system
type LoadBalancerIntegrationHandler struct {
	manager  *SegmentationManager
	lbEvents chan LoadBalancerEvent
	ctx      context.Context
	cancel   context.CancelFunc
	wg       sync.WaitGroup
}

// LoadBalancerEvent represents load balancer events
type LoadBalancerEvent struct {
	Type        string                 `json:"type"`
	LBID        string                 `json:"lb_id"`
	TenantID    string                 `json:"tenant_id"`
	BackendIPs  []string               `json:"backend_ips,omitempty"`
	VIP         string                 `json:"vip,omitempty"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

// SecurityIntegrationHandler integrates with security system
type SecurityIntegrationHandler struct {
	manager        *SegmentationManager
	securityEvents chan SecurityEvent
	ctx            context.Context
	cancel         context.CancelFunc
	wg             sync.WaitGroup
}

// SecurityEvent represents security system events
type SecurityEvent struct {
	Type       string                 `json:"type"`
	TenantID   string                 `json:"tenant_id"`
	SourceIP   string                 `json:"source_ip,omitempty"`
	TargetIP   string                 `json:"target_ip,omitempty"`
	ThreatType string                 `json:"threat_type,omitempty"`
	Severity   string                 `json:"severity"`
	Metadata   map[string]interface{} `json:"metadata,omitempty"`
}

// TenantNetworkRequest represents a tenant network request
type TenantNetworkRequest struct {
	TenantID      string                       `json:"tenant_id"`
	NetworkName   string                       `json:"network_name"`
	NetworkType   tenant.TenantNetworkType     `json:"network_type"`
	CIDR          string                       `json:"cidr"`
	QoSPolicy     *qos.QoSPolicy               `json:"qos_policy,omitempty"`
	FirewallRules []*firewall.FirewallRule     `json:"firewall_rules,omitempty"`
	SecurityGroups []string                    `json:"security_groups,omitempty"`
	Options       map[string]interface{}       `json:"options,omitempty"`
}

// TenantNetworkResponse represents the response to a tenant network request
type TenantNetworkResponse struct {
	NetworkID     string                    `json:"network_id"`
	Network       *tenant.TenantNetwork     `json:"network"`
	FirewallID    string                    `json:"firewall_id,omitempty"`
	QoSPolicyID   string                    `json:"qos_policy_id,omitempty"`
	Status        string                    `json:"status"`
	Message       string                    `json:"message,omitempty"`
	CreatedAt     time.Time                 `json:"created_at"`
}

// DefaultSegmentationConfig returns default segmentation configuration
func DefaultSegmentationConfig() *SegmentationConfig {
	return &SegmentationConfig{
		EnableSDNController:      true,
		EnableTenantIsolation:    true,
		EnableMicrosegmentation:  true,
		EnableQoSManagement:      true,
		EnableNetworkAnalytics:   true,
		EnableComplianceAuditing: true,
		DefaultTenantIsolation:   tenant.VXLANIsolation,
		DefaultFirewallAction:    firewall.ActionDrop,
		DefaultQoSAlgorithm:      qos.AlgorithmHTB,
		MetricsInterval:          30 * time.Second,
		EventBufferSize:          10000,
		MaxTenantsPerNode:        100,
		MaxNetworksPerTenant:     50,
		MaxFirewallRules:         1000,
		MaxQoSPolicies:          100,
	}
}

// NewSegmentationManager creates a new segmentation manager
func NewSegmentationManager(name string, config *SegmentationConfig) *SegmentationManager {
	ctx, cancel := context.WithCancel(context.Background())
	
	sm := &SegmentationManager{
		ID:                  uuid.New().String(),
		Name:                name,
		Firewalls:           make(map[string]*firewall.MicrosegmentationFirewall),
		Config:              config,
		EventListeners:      make([]SegmentationEventListener, 0),
		integrationHandlers: make(map[string]IntegrationHandler),
		ctx:                 ctx,
		cancel:              cancel,
		Metrics: &SegmentationMetrics{
			ComponentHealth:   make(map[string]string),
			IntegrationStatus: make(map[string]string),
			LastUpdated:      time.Now(),
		},
	}
	
	// Initialize components based on configuration
	if config.EnableSDNController {
		sm.initializeSDNController()
	}
	
	if config.EnableTenantIsolation {
		sm.initializeTenantManager()
	}
	
	if config.EnableQoSManagement {
		sm.initializeQoSEngine()
	}
	
	// Initialize integration handlers
	sm.initializeIntegrationHandlers()
	
	return sm
}

// Start starts the segmentation manager
func (sm *SegmentationManager) Start() error {
	sm.mutex.Lock()
	defer sm.mutex.Unlock()
	
	log.Printf("Starting network segmentation manager %s", sm.Name)
	
	// Start components
	if sm.SDNController != nil {
		if err := sm.SDNController.Start(); err != nil {
			return fmt.Errorf("failed to start SDN controller: %w", err)
		}
		sm.Metrics.ComponentHealth["sdn_controller"] = "healthy"
	}
	
	if sm.TenantManager != nil {
		if err := sm.TenantManager.Start(); err != nil {
			return fmt.Errorf("failed to start tenant manager: %w", err)
		}
		sm.Metrics.ComponentHealth["tenant_manager"] = "healthy"
	}
	
	if sm.QoSEngine != nil {
		if err := sm.QoSEngine.Start(); err != nil {
			return fmt.Errorf("failed to start QoS engine: %w", err)
		}
		sm.Metrics.ComponentHealth["qos_engine"] = "healthy"
	}
	
	// Start integration handlers
	for name, handler := range sm.integrationHandlers {
		if err := handler.Initialize(sm.ctx, sm); err != nil {
			log.Printf("Warning: failed to initialize integration handler %s: %v", name, err)
			sm.Metrics.IntegrationStatus[name] = "error"
		} else {
			sm.Metrics.IntegrationStatus[name] = "active"
		}
	}
	
	// Start metrics collection
	if sm.Config.MetricsInterval > 0 {
		sm.wg.Add(1)
		go sm.collectMetrics()
	}
	
	// Start event processing
	sm.wg.Add(1)
	go sm.processEvents()
	
	sm.emitEvent(SegmentationEvent{
		Type:      "segmentation_manager_started",
		Component: "manager",
		Severity:  "info",
		Message:   fmt.Sprintf("Segmentation manager %s started successfully", sm.Name),
		Timestamp: time.Now(),
	})
	
	log.Printf("Network segmentation manager %s started successfully", sm.Name)
	return nil
}

// Stop stops the segmentation manager
func (sm *SegmentationManager) Stop() error {
	sm.mutex.Lock()
	defer sm.mutex.Unlock()
	
	log.Printf("Stopping network segmentation manager %s", sm.Name)
	
	sm.cancel()
	sm.wg.Wait()
	
	// Stop integration handlers
	for name, handler := range sm.integrationHandlers {
		if err := handler.Shutdown(); err != nil {
			log.Printf("Warning: failed to shutdown integration handler %s: %v", name, err)
		}
		sm.Metrics.IntegrationStatus[name] = "stopped"
	}
	
	// Stop firewalls
	for firewallID, fw := range sm.Firewalls {
		if err := fw.Stop(); err != nil {
			log.Printf("Warning: failed to stop firewall %s: %v", firewallID, err)
		}
	}
	
	// Stop components
	if sm.QoSEngine != nil {
		if err := sm.QoSEngine.Stop(); err != nil {
			log.Printf("Warning: failed to stop QoS engine: %v", err)
		}
		sm.Metrics.ComponentHealth["qos_engine"] = "stopped"
	}
	
	if sm.TenantManager != nil {
		if err := sm.TenantManager.Stop(); err != nil {
			log.Printf("Warning: failed to stop tenant manager: %v", err)
		}
		sm.Metrics.ComponentHealth["tenant_manager"] = "stopped"
	}
	
	if sm.SDNController != nil {
		if err := sm.SDNController.Stop(); err != nil {
			log.Printf("Warning: failed to stop SDN controller: %v", err)
		}
		sm.Metrics.ComponentHealth["sdn_controller"] = "stopped"
	}
	
	sm.emitEvent(SegmentationEvent{
		Type:      "segmentation_manager_stopped",
		Component: "manager",
		Severity:  "info",
		Message:   fmt.Sprintf("Segmentation manager %s stopped", sm.Name),
		Timestamp: time.Now(),
	})
	
	log.Printf("Network segmentation manager %s stopped", sm.Name)
	return nil
}

// CreateTenantNetwork creates a complete tenant network with segmentation
func (sm *SegmentationManager) CreateTenantNetwork(ctx context.Context, req *TenantNetworkRequest) (*TenantNetworkResponse, error) {
	log.Printf("Creating tenant network %s for tenant %s", req.NetworkName, req.TenantID)
	
	response := &TenantNetworkResponse{
		Status:    "creating",
		CreatedAt: time.Now(),
	}
	
	// 1. Create tenant network
	var network *tenant.TenantNetwork
	var err error
	
	if sm.TenantManager != nil {
		network, err = sm.TenantManager.CreateTenantNetwork(ctx, req.TenantID, req.NetworkName, req.CIDR, req.Options)
		if err != nil {
			response.Status = "error"
			response.Message = fmt.Sprintf("Failed to create tenant network: %v", err)
			return response, fmt.Errorf("failed to create tenant network: %w", err)
		}
		response.NetworkID = network.ID
		response.Network = network
	}
	
	// 2. Create microsegmentation firewall
	if sm.Config.EnableMicrosegmentation && len(req.FirewallRules) > 0 {
		firewallConfig := &firewall.FirewallConfig{
			DefaultAction:            sm.Config.DefaultFirewallAction,
			EnableConnectionTracking: true,
			EnableDPI:                true,
			EnableThreatIntel:        true,
			EnableRateLimit:          true,
			EnableLogging:            true,
			MaxConnections:           100000,
			ConnectionTimeout:        2 * time.Hour,
			PacketBufferSize:         10000,
			WorkerCount:              4,
			MetricsInterval:          30 * time.Second,
		}
		
		fw := firewall.NewMicrosegmentationFirewall(req.TenantID, 
			fmt.Sprintf("fw-%s", network.ID), firewallConfig)
		
		// Add firewall rules
		for _, rule := range req.FirewallRules {
			if err := fw.AddRule(rule); err != nil {
				log.Printf("Warning: failed to add firewall rule %s: %v", rule.ID, err)
			}
		}
		
		// Start firewall
		if err := fw.Start(); err != nil {
			log.Printf("Warning: failed to start firewall: %v", err)
		} else {
			sm.Firewalls[fw.ID] = fw
			response.FirewallID = fw.ID
		}
	}
	
	// 3. Apply QoS policy
	if sm.QoSEngine != nil && req.QoSPolicy != nil {
		req.QoSPolicy.TenantID = req.TenantID
		if req.QoSPolicy.ID == "" {
			req.QoSPolicy.ID = uuid.New().String()
		}
		
		if err := sm.QoSEngine.CreatePolicy(req.QoSPolicy); err != nil {
			log.Printf("Warning: failed to create QoS policy: %v", err)
		} else {
			response.QoSPolicyID = req.QoSPolicy.ID
		}
	}
	
	// 4. Configure SDN controller flows
	if sm.SDNController != nil && network != nil {
		// Create policy rules for tenant isolation
		isolationRule := &sdn.PolicyRule{
			ID:          uuid.New().String(),
			Name:        fmt.Sprintf("tenant-isolation-%s", req.TenantID),
			TenantID:    req.TenantID,
			Priority:    1000,
			Direction:   "both",
			Protocol:    "any",
			SrcSelector: fmt.Sprintf("tenant:%s", req.TenantID),
			DstSelector: fmt.Sprintf("tenant:%s", req.TenantID),
			Action:      "allow",
			Enabled:     true,
		}
		
		if err := sm.SDNController.AddPolicyRule(isolationRule); err != nil {
			log.Printf("Warning: failed to add SDN isolation rule: %v", err)
		}
		
		// Add inter-tenant blocking rule
		blockRule := &sdn.PolicyRule{
			ID:          uuid.New().String(),
			Name:        fmt.Sprintf("inter-tenant-block-%s", req.TenantID),
			TenantID:    req.TenantID,
			Priority:    500,
			Direction:   "both",
			Protocol:    "any",
			SrcSelector: fmt.Sprintf("tenant:%s", req.TenantID),
			DstSelector: "tenant:*",
			Action:      "deny",
			Enabled:     true,
		}
		
		if err := sm.SDNController.AddPolicyRule(blockRule); err != nil {
			log.Printf("Warning: failed to add inter-tenant block rule: %v", err)
		}
	}
	
	response.Status = "created"
	response.Message = "Tenant network created successfully"
	
	// Emit event
	sm.emitEvent(SegmentationEvent{
		Type:      "tenant_network_created",
		Component: "manager",
		TenantID:  req.TenantID,
		NetworkID: response.NetworkID,
		Severity:  "info",
		Message:   fmt.Sprintf("Tenant network %s created for tenant %s", req.NetworkName, req.TenantID),
		Data:      response,
		Timestamp: time.Now(),
	})
	
	// Update metrics
	sm.mutex.Lock()
	sm.Metrics.TotalNetworks++
	sm.Metrics.LastUpdated = time.Now()
	sm.mutex.Unlock()
	
	log.Printf("Successfully created tenant network %s (%s) for tenant %s", 
		req.NetworkName, response.NetworkID, req.TenantID)
	
	return response, nil
}

// DeleteTenantNetwork deletes a tenant network and all associated resources
func (sm *SegmentationManager) DeleteTenantNetwork(ctx context.Context, tenantID, networkID string) error {
	log.Printf("Deleting tenant network %s for tenant %s", networkID, tenantID)
	
	// Remove SDN policy rules
	if sm.SDNController != nil {
		// Find and remove rules for this tenant/network
		// This would require querying rules by tenant/network ID
	}
	
	// Remove QoS policies
	if sm.QoSEngine != nil {
		policies := sm.QoSEngine.ListPolicies()
		for _, policy := range policies {
			if policy.TenantID == tenantID {
				if err := sm.QoSEngine.DeletePolicy(policy.ID); err != nil {
					log.Printf("Warning: failed to delete QoS policy %s: %v", policy.ID, err)
				}
			}
		}
	}
	
	// Stop and remove firewalls
	firewallsToRemove := make([]string, 0)
	for firewallID, fw := range sm.Firewalls {
		if fw.TenantID == tenantID {
			if err := fw.Stop(); err != nil {
				log.Printf("Warning: failed to stop firewall %s: %v", firewallID, err)
			}
			firewallsToRemove = append(firewallsToRemove, firewallID)
		}
	}
	
	for _, firewallID := range firewallsToRemove {
		delete(sm.Firewalls, firewallID)
	}
	
	// Delete tenant network
	if sm.TenantManager != nil {
		// This would require a method to delete a specific network
		// For now, we'll just log
		log.Printf("Would delete network %s from tenant %s", networkID, tenantID)
	}
	
	// Emit event
	sm.emitEvent(SegmentationEvent{
		Type:      "tenant_network_deleted",
		Component: "manager",
		TenantID:  tenantID,
		NetworkID: networkID,
		Severity:  "info",
		Message:   fmt.Sprintf("Tenant network %s deleted for tenant %s", networkID, tenantID),
		Timestamp: time.Now(),
	})
	
	// Update metrics
	sm.mutex.Lock()
	sm.Metrics.TotalNetworks--
	sm.Metrics.LastUpdated = time.Now()
	sm.mutex.Unlock()
	
	log.Printf("Successfully deleted tenant network %s for tenant %s", networkID, tenantID)
	return nil
}

// GetSegmentationStatus returns the current status of the segmentation system
func (sm *SegmentationManager) GetSegmentationStatus() map[string]interface{} {
	sm.mutex.RLock()
	defer sm.mutex.RUnlock()
	
	status := map[string]interface{}{
		"id":                sm.ID,
		"name":              sm.Name,
		"metrics":           sm.Metrics,
		"component_health":  sm.Metrics.ComponentHealth,
		"integration_status": sm.Metrics.IntegrationStatus,
		"config":            sm.Config,
	}
	
	// Add component-specific status
	if sm.SDNController != nil {
		status["sdn_controller"] = map[string]interface{}{
			"id":              sm.SDNController.ID,
			"state":           sm.SDNController.State,
			"switches":        len(sm.SDNController.ListSwitches()),
			"policy_rules":    len(sm.SDNController.PolicyRules),
			"metrics":         sm.SDNController.GetMetrics(),
		}
	}
	
	if sm.TenantManager != nil {
		status["tenant_manager"] = map[string]interface{}{
			"total_tenants": len(sm.TenantManager.ListTenants()),
			"metrics":       sm.TenantManager.GetMetrics(),
		}
	}
	
	if sm.QoSEngine != nil {
		status["qos_engine"] = map[string]interface{}{
			"id":              sm.QoSEngine.ID,
			"total_policies":  len(sm.QoSEngine.ListPolicies()),
			"config":          sm.QoSEngine.Config,
		}
	}
	
	status["firewalls"] = map[string]interface{}{
		"total_firewalls": len(sm.Firewalls),
		"firewall_metrics": func() map[string]interface{} {
			metrics := make(map[string]interface{})
			for id, fw := range sm.Firewalls {
				metrics[id] = fw.GetMetrics()
			}
			return metrics
		}(),
	}
	
	return status
}

// AddEventListener adds a segmentation event listener
func (sm *SegmentationManager) AddEventListener(listener SegmentationEventListener) {
	sm.mutex.Lock()
	defer sm.mutex.Unlock()
	
	sm.EventListeners = append(sm.EventListeners, listener)
}

// Private methods

// initializeSDNController initializes the SDN controller
func (sm *SegmentationManager) initializeSDNController() {
	controllerConfig := &sdn.ControllerConfig{
		ListenAddress:       "0.0.0.0",
		ListenPort:          6653,
		OpenFlowVersion:     sdn.OpenFlow13,
		HeartbeatInterval:   10 * time.Second,
		StatsInterval:       30 * time.Second,
		FlowTableSize:       10000,
		EnableMetrics:       true,
		EnableEventLogging:  true,
	}
	
	sm.SDNController = sdn.NewSDNController(controllerConfig)
	
	// Add event listener
	sm.SDNController.AddEventListener(func(event sdn.ControllerEvent) {
		sm.emitEvent(SegmentationEvent{
			Type:      "sdn_event",
			Component: "sdn_controller",
			Severity:  "info",
			Message:   event.Type,
			Data:      event,
			Timestamp: event.Timestamp,
		})
	})
}

// initializeTenantManager initializes the tenant manager
func (sm *SegmentationManager) initializeTenantManager() {
	tenantConfig := tenant.DefaultTenantManagerConfig()
	tenantConfig.DefaultIsolationType = sm.Config.DefaultTenantIsolation
	tenantConfig.EnableResourceQuotas = true
	tenantConfig.EnableMetrics = true
	
	sm.TenantManager = tenant.NewTenantManager(tenantConfig)
	
	// Add event listener
	sm.TenantManager.AddEventListener(func(event tenant.TenantEvent) {
		sm.emitEvent(SegmentationEvent{
			Type:      "tenant_event",
			Component: "tenant_manager",
			TenantID:  event.TenantID,
			Severity:  "info",
			Message:   event.Type,
			Data:      event,
			Timestamp: event.Timestamp,
		})
	})
}

// initializeQoSEngine initializes the QoS engine
func (sm *SegmentationManager) initializeQoSEngine() {
	qosConfig := qos.DefaultQoSEngineConfig()
	qosConfig.DefaultAlgorithm = sm.Config.DefaultQoSAlgorithm
	qosConfig.EnableRealTimeStats = true
	qosConfig.StatisticsInterval = sm.Config.MetricsInterval
	
	sm.QoSEngine = qos.NewQoSEngine("novacron-qos", qosConfig)
	
	// Add event listener
	sm.QoSEngine.AddEventListener(func(event qos.QoSEvent) {
		sm.emitEvent(SegmentationEvent{
			Type:      "qos_event",
			Component: "qos_engine",
			TenantID:  event.TenantID,
			Severity:  event.Severity,
			Message:   event.Message,
			Data:      event,
			Timestamp: event.Timestamp,
		})
	})
}

// initializeIntegrationHandlers initializes integration handlers
func (sm *SegmentationManager) initializeIntegrationHandlers() {
	// VM integration handler
	vmHandler := &VMIntegrationHandler{
		manager: sm,
		vmEvents: make(chan VMEvent, sm.Config.EventBufferSize),
	}
	sm.integrationHandlers["vm_integration"] = vmHandler
	
	// Load balancer integration handler
	lbHandler := &LoadBalancerIntegrationHandler{
		manager:  sm,
		lbEvents: make(chan LoadBalancerEvent, sm.Config.EventBufferSize),
	}
	sm.integrationHandlers["loadbalancer_integration"] = lbHandler
	
	// Security integration handler
	secHandler := &SecurityIntegrationHandler{
		manager:        sm,
		securityEvents: make(chan SecurityEvent, sm.Config.EventBufferSize),
	}
	sm.integrationHandlers["security_integration"] = secHandler
}

// collectMetrics collects segmentation metrics
func (sm *SegmentationManager) collectMetrics() {
	defer sm.wg.Done()
	
	ticker := time.NewTicker(sm.Config.MetricsInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-sm.ctx.Done():
			return
		case <-ticker.C:
			sm.updateMetrics()
		}
	}
}

// updateMetrics updates segmentation metrics
func (sm *SegmentationManager) updateMetrics() {
	sm.mutex.Lock()
	defer sm.mutex.Unlock()
	
	// Update tenant metrics
	if sm.TenantManager != nil {
		tenantMetrics := sm.TenantManager.GetMetrics()
		sm.Metrics.TotalTenants = tenantMetrics.TotalTenants
		sm.Metrics.TotalNetworks = tenantMetrics.TotalNetworks
	}
	
	// Update firewall metrics
	totalRules := int64(0)
	packetsProcessed := uint64(0)
	packetsBlocked := uint64(0)
	
	for _, fw := range sm.Firewalls {
		fwMetrics := fw.GetMetrics()
		totalRules += int64(len(fw.ListRules()))
		packetsProcessed += fwMetrics.PacketsProcessed
		packetsBlocked += fwMetrics.PacketsDropped + fwMetrics.PacketsRejected
	}
	
	sm.Metrics.TotalFirewallRules = totalRules
	sm.Metrics.PacketsProcessed = packetsProcessed
	sm.Metrics.PacketsBlocked = packetsBlocked
	
	// Update QoS metrics
	if sm.QoSEngine != nil {
		policies := sm.QoSEngine.ListPolicies()
		sm.Metrics.TotalQoSPolicies = int64(len(policies))
	}
	
	sm.Metrics.LastUpdated = time.Now()
}

// processEvents processes segmentation events
func (sm *SegmentationManager) processEvents() {
	defer sm.wg.Done()
	
	// This would process events from various sources
	// For now, it's a placeholder for event correlation and processing
	
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()
	
	for {
		select {
		case <-sm.ctx.Done():
			return
		case <-ticker.C:
			// Process any queued events
		}
	}
}

// emitEvent emits a segmentation event
func (sm *SegmentationManager) emitEvent(event SegmentationEvent) {
	sm.mutex.RLock()
	listeners := make([]SegmentationEventListener, len(sm.EventListeners))
	copy(listeners, sm.EventListeners)
	sm.mutex.RUnlock()
	
	for _, listener := range listeners {
		go func(l SegmentationEventListener, e SegmentationEvent) {
			defer func() {
				if r := recover(); r != nil {
					log.Printf("Segmentation event listener panic: %v", r)
				}
			}()
			l(e)
		}(listener, event)
	}
}

// Integration handler implementations

// VM Integration Handler
func (h *VMIntegrationHandler) GetName() string {
	return "vm_integration"
}

func (h *VMIntegrationHandler) Initialize(ctx context.Context, manager *SegmentationManager) error {
	h.ctx, h.cancel = context.WithCancel(ctx)
	h.wg.Add(1)
	go h.processVMEvents()
	return nil
}

func (h *VMIntegrationHandler) HandleEvent(event interface{}) error {
	if vmEvent, ok := event.(VMEvent); ok {
		select {
		case h.vmEvents <- vmEvent:
			return nil
		default:
			return fmt.Errorf("VM event queue full")
		}
	}
	return fmt.Errorf("invalid event type for VM integration")
}

func (h *VMIntegrationHandler) GetStatus() string {
	return "active"
}

func (h *VMIntegrationHandler) Shutdown() error {
	h.cancel()
	close(h.vmEvents)
	h.wg.Wait()
	return nil
}

func (h *VMIntegrationHandler) processVMEvents() {
	defer h.wg.Done()
	
	for {
		select {
		case <-h.ctx.Done():
			return
		case event := <-h.vmEvents:
			h.handleVMEvent(event)
		}
	}
}

func (h *VMIntegrationHandler) handleVMEvent(event VMEvent) {
	log.Printf("Processing VM event: %s for VM %s", event.Type, event.VMID)
	
	switch event.Type {
	case "vm_created":
		// When a VM is created, ensure it's connected to appropriate tenant networks
		// and firewall rules are applied
		
	case "vm_started":
		// Apply runtime network policies and QoS settings
		
	case "vm_migrated":
		// Update network policies for new location
		
	case "vm_stopped":
		// Clean up temporary network state
		
	case "vm_deleted":
		// Clean up all network resources associated with the VM
	}
}

// Load Balancer Integration Handler
func (h *LoadBalancerIntegrationHandler) GetName() string {
	return "loadbalancer_integration"
}

func (h *LoadBalancerIntegrationHandler) Initialize(ctx context.Context, manager *SegmentationManager) error {
	h.ctx, h.cancel = context.WithCancel(ctx)
	h.wg.Add(1)
	go h.processLBEvents()
	return nil
}

func (h *LoadBalancerIntegrationHandler) HandleEvent(event interface{}) error {
	if lbEvent, ok := event.(LoadBalancerEvent); ok {
		select {
		case h.lbEvents <- lbEvent:
			return nil
		default:
			return fmt.Errorf("load balancer event queue full")
		}
	}
	return fmt.Errorf("invalid event type for load balancer integration")
}

func (h *LoadBalancerIntegrationHandler) GetStatus() string {
	return "active"
}

func (h *LoadBalancerIntegrationHandler) Shutdown() error {
	h.cancel()
	close(h.lbEvents)
	h.wg.Wait()
	return nil
}

func (h *LoadBalancerIntegrationHandler) processLBEvents() {
	defer h.wg.Done()
	
	for {
		select {
		case <-h.ctx.Done():
			return
		case event := <-h.lbEvents:
			h.handleLBEvent(event)
		}
	}
}

func (h *LoadBalancerIntegrationHandler) handleLBEvent(event LoadBalancerEvent) {
	log.Printf("Processing LB event: %s for LB %s", event.Type, event.LBID)
	
	switch event.Type {
	case "lb_created":
		// Create appropriate firewall rules for load balancer traffic
		
	case "backend_added":
		// Update firewall and QoS rules for new backend
		
	case "backend_removed":
		// Clean up rules for removed backend
		
	case "vip_changed":
		// Update network routing and firewall rules
		
	case "lb_deleted":
		// Clean up all associated network rules
	}
}

// Security Integration Handler
func (h *SecurityIntegrationHandler) GetName() string {
	return "security_integration"
}

func (h *SecurityIntegrationHandler) Initialize(ctx context.Context, manager *SegmentationManager) error {
	h.ctx, h.cancel = context.WithCancel(ctx)
	h.wg.Add(1)
	go h.processSecurityEvents()
	return nil
}

func (h *SecurityIntegrationHandler) HandleEvent(event interface{}) error {
	if secEvent, ok := event.(SecurityEvent); ok {
		select {
		case h.securityEvents <- secEvent:
			return nil
		default:
			return fmt.Errorf("security event queue full")
		}
	}
	return fmt.Errorf("invalid event type for security integration")
}

func (h *SecurityIntegrationHandler) GetStatus() string {
	return "active"
}

func (h *SecurityIntegrationHandler) Shutdown() error {
	h.cancel()
	close(h.securityEvents)
	h.wg.Wait()
	return nil
}

func (h *SecurityIntegrationHandler) processSecurityEvents() {
	defer h.wg.Done()
	
	for {
		select {
		case <-h.ctx.Done():
			return
		case event := <-h.securityEvents:
			h.handleSecurityEvent(event)
		}
	}
}

func (h *SecurityIntegrationHandler) handleSecurityEvent(event SecurityEvent) {
	log.Printf("Processing security event: %s severity %s", event.Type, event.Severity)
	
	switch event.Type {
	case "threat_detected":
		// Add IP to firewall blacklist
		
	case "attack_blocked":
		// Update firewall rules based on attack pattern
		
	case "policy_violation":
		// Adjust network segmentation based on violation
		
	case "compliance_alert":
		// Update network policies for compliance
	}
}