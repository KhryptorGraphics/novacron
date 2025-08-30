package nfv

import (
	"context"
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/google/uuid"
)

// NetworkFunction represents a virtualized network function
type NetworkFunction struct {
	ID              string                 `json:"id"`
	Name            string                 `json:"name"`
	Type            NFType                 `json:"type"`
	Version         string                 `json:"version"`
	Description     string                 `json:"description"`
	Image           NFImage                `json:"image"`
	Resources       ResourceRequirements   `json:"resources"`
	NetworkConfig   NetworkConfig          `json:"network_config"`
	Scaling         ScalingConfig          `json:"scaling"`
	HealthCheck     HealthCheckConfig      `json:"health_check"`
	Configuration   map[string]interface{} `json:"configuration"`
	Status          NFStatus               `json:"status"`
	CreatedAt       time.Time              `json:"created_at"`
	UpdatedAt       time.Time              `json:"updated_at"`
	Metadata        map[string]interface{} `json:"metadata,omitempty"`
}

// NFType represents the type of network function
type NFType string

const (
	NFTypeFirewall      NFType = "firewall"
	NFTypeLoadBalancer  NFType = "load_balancer"
	NFTypeRouter        NFType = "router"
	NFTypeSwitch        NFType = "switch"
	NFTypeVPN           NFType = "vpn"
	NFTypeIDS           NFType = "ids"
	NFTypeIPS           NFType = "ips"
	NFTypeDPI           NFType = "dpi"
	NFTypeNAT           NFType = "nat"
	NFTypeQoS           NFType = "qos"
	NFTypeWAN           NFType = "wan_optimizer"
	NFTypeProxy         NFType = "proxy"
	NFTypeDNS           NFType = "dns"
	NFTypeDHCP          NFType = "dhcp"
	NFTypeCustom        NFType = "custom"
)

// NFStatus represents the status of a network function
type NFStatus string

const (
	NFStatusDeploying   NFStatus = "deploying"
	NFStatusRunning     NFStatus = "running"
	NFStatusStopped     NFStatus = "stopped"
	NFStatusFailed      NFStatus = "failed"
	NFStatusScaling     NFStatus = "scaling"
	NFStatusUpdating    NFStatus = "updating"
	NFStatusMaintenance NFStatus = "maintenance"
)

// NFImage represents the container image for a network function
type NFImage struct {
	Repository  string            `json:"repository"`
	Tag         string            `json:"tag"`
	Digest      string            `json:"digest,omitempty"`
	PullPolicy  string            `json:"pull_policy"`
	Registry    string            `json:"registry,omitempty"`
	Credentials map[string]string `json:"credentials,omitempty"`
}

// ResourceRequirements defines resource requirements for a network function
type ResourceRequirements struct {
	CPU           CPURequirement    `json:"cpu"`
	Memory        MemoryRequirement `json:"memory"`
	Storage       StorageRequirement `json:"storage"`
	Network       NetworkRequirement `json:"network"`
	GPU           *GPURequirement   `json:"gpu,omitempty"`
	Accelerators  []AcceleratorReq  `json:"accelerators,omitempty"`
	Affinity      *AffinityConfig   `json:"affinity,omitempty"`
}

// CPURequirement defines CPU requirements
type CPURequirement struct {
	Requests      float64  `json:"requests"`      // CPU cores requested
	Limits        float64  `json:"limits"`        // CPU cores limit
	Architecture  string   `json:"architecture,omitempty"` // x86_64, arm64, etc.
	Features      []string `json:"features,omitempty"`     // AVX, SSE, etc.
}

// MemoryRequirement defines memory requirements
type MemoryRequirement struct {
	Requests    int64  `json:"requests"`    // Memory in MB
	Limits      int64  `json:"limits"`      // Memory in MB
	Type        string `json:"type,omitempty"` // DDR4, DDR5, etc.
	HugePages   bool   `json:"huge_pages"`
}

// StorageRequirement defines storage requirements
type StorageRequirement struct {
	Size        int64  `json:"size"`        // Storage in GB
	Type        string `json:"type"`        // SSD, NVMe, HDD
	IOPS        int    `json:"iops,omitempty"`
	Throughput  int    `json:"throughput,omitempty"` // MB/s
}

// NetworkRequirement defines network requirements
type NetworkRequirement struct {
	Bandwidth   int64    `json:"bandwidth"`   // Mbps
	Latency     int      `json:"latency"`     // ms
	PacketRate  int64    `json:"packet_rate"` // packets per second
	Protocols   []string `json:"protocols"`   // TCP, UDP, SCTP
	Features    []string `json:"features"`    // SR-IOV, DPDK
}

// GPURequirement defines GPU requirements
type GPURequirement struct {
	Count        int    `json:"count"`
	Type         string `json:"type"`         // NVIDIA, AMD
	Model        string `json:"model,omitempty"`
	Memory       int64  `json:"memory"`       // GB
	ComputeUnits int    `json:"compute_units,omitempty"`
}

// AcceleratorReq defines accelerator requirements
type AcceleratorReq struct {
	Type    string `json:"type"`    // FPGA, TPU, VPU
	Model   string `json:"model,omitempty"`
	Count   int    `json:"count"`
	Features []string `json:"features,omitempty"`
}

// AffinityConfig defines placement affinity/anti-affinity rules
type AffinityConfig struct {
	NodeAffinity     *NodeAffinity     `json:"node_affinity,omitempty"`
	PodAffinity      *PodAffinity      `json:"pod_affinity,omitempty"`
	PodAntiAffinity  *PodAffinity      `json:"pod_anti_affinity,omitempty"`
}

// NodeAffinity defines node affinity rules
type NodeAffinity struct {
	RequiredDuringSchedulingIgnoredDuringExecution  []NodeSelectorTerm `json:"required,omitempty"`
	PreferredDuringSchedulingIgnoredDuringExecution []NodePreference   `json:"preferred,omitempty"`
}

// PodAffinity defines pod affinity rules
type PodAffinity struct {
	RequiredDuringSchedulingIgnoredDuringExecution  []PodAffinityTerm `json:"required,omitempty"`
	PreferredDuringSchedulingIgnoredDuringExecution []WeightedPodAffinityTerm `json:"preferred,omitempty"`
}

// NodeSelectorTerm defines node selector terms
type NodeSelectorTerm struct {
	MatchExpressions []NodeSelectorRequirement `json:"match_expressions"`
}

// NodeSelectorRequirement defines node selector requirements
type NodeSelectorRequirement struct {
	Key      string   `json:"key"`
	Operator string   `json:"operator"`
	Values   []string `json:"values"`
}

// NodePreference defines weighted node preferences
type NodePreference struct {
	Weight     int32            `json:"weight"`
	Preference NodeSelectorTerm `json:"preference"`
}

// PodAffinityTerm defines pod affinity terms
type PodAffinityTerm struct {
	LabelSelector *LabelSelector `json:"label_selector"`
	TopologyKey   string         `json:"topology_key"`
}

// WeightedPodAffinityTerm defines weighted pod affinity terms
type WeightedPodAffinityTerm struct {
	Weight          int32           `json:"weight"`
	PodAffinityTerm PodAffinityTerm `json:"pod_affinity_term"`
}

// LabelSelector defines label selectors
type LabelSelector struct {
	MatchLabels      map[string]string `json:"match_labels,omitempty"`
	MatchExpressions []LabelRequirement `json:"match_expressions,omitempty"`
}

// LabelRequirement defines label requirements
type LabelRequirement struct {
	Key      string   `json:"key"`
	Operator string   `json:"operator"`
	Values   []string `json:"values"`
}

// NetworkConfig defines network configuration for the NF
type NetworkConfig struct {
	Interfaces []NetworkInterface `json:"interfaces"`
	Routes     []Route           `json:"routes,omitempty"`
	DNS        []string          `json:"dns,omitempty"`
	QoS        *QoSConfig        `json:"qos,omitempty"`
}

// NetworkInterface defines a network interface for the NF
type NetworkInterface struct {
	Name       string `json:"name"`
	Type       string `json:"type"`       // bridge, sriov, macvlan, etc.
	Network    string `json:"network"`    // Network ID/name
	IPAddress  string `json:"ip_address,omitempty"`
	MACAddress string `json:"mac_address,omitempty"`
	VLAN       int    `json:"vlan,omitempty"`
	Bandwidth  int64  `json:"bandwidth,omitempty"` // Mbps
}

// Route defines routing configuration
type Route struct {
	Destination string `json:"destination"`
	Gateway     string `json:"gateway"`
	Interface   string `json:"interface,omitempty"`
	Metric      int    `json:"metric,omitempty"`
}

// QoSConfig defines QoS configuration
type QoSConfig struct {
	Priority       int     `json:"priority"`
	MinBandwidth   int64   `json:"min_bandwidth"`  // Mbps
	MaxBandwidth   int64   `json:"max_bandwidth"`  // Mbps
	MaxLatency     int     `json:"max_latency"`    // ms
	MaxJitter      int     `json:"max_jitter"`     // ms
	MaxPacketLoss  float64 `json:"max_packet_loss"` // percentage
	DSCP           int     `json:"dscp,omitempty"`
	TrafficClass   string  `json:"traffic_class,omitempty"`
}

// ScalingConfig defines auto-scaling configuration
type ScalingConfig struct {
	Enabled         bool                     `json:"enabled"`
	MinReplicas     int                      `json:"min_replicas"`
	MaxReplicas     int                      `json:"max_replicas"`
	TargetMetrics   []ScalingMetric          `json:"target_metrics"`
	ScaleUpPolicy   ScalingPolicy            `json:"scale_up_policy"`
	ScaleDownPolicy ScalingPolicy            `json:"scale_down_policy"`
	Behavior        *ScalingBehavior         `json:"behavior,omitempty"`
}

// ScalingMetric defines metrics for auto-scaling
type ScalingMetric struct {
	Type          string  `json:"type"`          // CPU, Memory, Network, Custom
	TargetValue   float64 `json:"target_value"`  // Target utilization percentage
	MetricName    string  `json:"metric_name,omitempty"`
	MetricSelector map[string]string `json:"metric_selector,omitempty"`
}

// ScalingPolicy defines scaling policy
type ScalingPolicy struct {
	PeriodSeconds int `json:"period_seconds"`
	MaxChange     int `json:"max_change"` // Max instances to add/remove at once
}

// ScalingBehavior defines advanced scaling behavior
type ScalingBehavior struct {
	ScaleUp   *HPAScalingRules `json:"scale_up,omitempty"`
	ScaleDown *HPAScalingRules `json:"scale_down,omitempty"`
}

// HPAScalingRules defines HPA scaling rules
type HPAScalingRules struct {
	StabilizationWindowSeconds *int32                `json:"stabilization_window_seconds,omitempty"`
	SelectPolicy               *string               `json:"select_policy,omitempty"`
	Policies                   []HPAScalingPolicy    `json:"policies,omitempty"`
}

// HPAScalingPolicy defines individual HPA scaling policy
type HPAScalingPolicy struct {
	Type          string `json:"type"`
	Value         int32  `json:"value"`
	PeriodSeconds int32  `json:"period_seconds"`
}

// HealthCheckConfig defines health check configuration
type HealthCheckConfig struct {
	Enabled             bool          `json:"enabled"`
	InitialDelaySeconds int           `json:"initial_delay_seconds"`
	PeriodSeconds       int           `json:"period_seconds"`
	TimeoutSeconds      int           `json:"timeout_seconds"`
	FailureThreshold    int           `json:"failure_threshold"`
	SuccessThreshold    int           `json:"success_threshold"`
	HTTPGet             *HTTPGetCheck `json:"http_get,omitempty"`
	TCPSocket           *TCPCheck     `json:"tcp_socket,omitempty"`
	Exec                *ExecCheck    `json:"exec,omitempty"`
}

// HTTPGetCheck defines HTTP health check
type HTTPGetCheck struct {
	Path   string            `json:"path"`
	Port   int               `json:"port"`
	Scheme string            `json:"scheme"`
	Headers map[string]string `json:"headers,omitempty"`
}

// TCPCheck defines TCP health check
type TCPCheck struct {
	Port int `json:"port"`
}

// ExecCheck defines exec health check
type ExecCheck struct {
	Command []string `json:"command"`
}

// NFVServiceChain represents a service chain of network functions
type NFVServiceChain struct {
	ID            string                     `json:"id"`
	Name          string                     `json:"name"`
	Description   string                     `json:"description"`
	Functions     []ChainFunction            `json:"functions"`
	FlowRules     []ServiceChainFlowRule     `json:"flow_rules"`
	QoSProfile    *QoSConfig                 `json:"qos_profile,omitempty"`
	Status        ServiceChainStatus         `json:"status"`
	CreatedAt     time.Time                  `json:"created_at"`
	UpdatedAt     time.Time                  `json:"updated_at"`
	Metadata      map[string]interface{}     `json:"metadata,omitempty"`
}

// ChainFunction represents a function in a service chain
type ChainFunction struct {
	FunctionID string `json:"function_id"`
	Order      int    `json:"order"`
	Input      string `json:"input,omitempty"`
	Output     string `json:"output,omitempty"`
	Condition  string `json:"condition,omitempty"`
}

// ServiceChainFlowRule defines traffic flow rules for service chains
type ServiceChainFlowRule struct {
	ID          string            `json:"id"`
	Priority    int               `json:"priority"`
	Match       FlowMatch         `json:"match"`
	Actions     []string          `json:"actions"`
	Metadata    map[string]string `json:"metadata,omitempty"`
}

// FlowMatch defines flow matching criteria
type FlowMatch struct {
	SourceIP        string `json:"source_ip,omitempty"`
	DestinationIP   string `json:"destination_ip,omitempty"`
	SourcePort      int    `json:"source_port,omitempty"`
	DestinationPort int    `json:"destination_port,omitempty"`
	Protocol        string `json:"protocol,omitempty"`
	VLAN            int    `json:"vlan,omitempty"`
	DSCP            int    `json:"dscp,omitempty"`
}

// ServiceChainStatus represents the status of a service chain
type ServiceChainStatus string

const (
	ServiceChainStatusDeploying ServiceChainStatus = "deploying"
	ServiceChainStatusActive    ServiceChainStatus = "active"
	ServiceChainStatusFailed    ServiceChainStatus = "failed"
	ServiceChainStatusUpdating  ServiceChainStatus = "updating"
	ServiceChainStatusStopped   ServiceChainStatus = "stopped"
)

// NFVOrchestrator manages network function virtualization
type NFVOrchestrator struct {
	// State management
	functions      map[string]*NetworkFunction
	functionsMutex sync.RWMutex
	
	serviceChains      map[string]*NFVServiceChain
	serviceChainsMutex sync.RWMutex
	
	// Dependencies
	containerRuntime ContainerRuntime
	networkManager   NetworkManager
	resourceManager  ResourceManager
	
	// Configuration
	config NFVConfig
	
	// Context for cancellation
	ctx    context.Context
	cancel context.CancelFunc
	
	// Event handlers
	eventHandlers []NFVEventHandler
	eventMutex    sync.RWMutex
}

// NFVConfig holds configuration for the NFV orchestrator
type NFVConfig struct {
	DefaultRegistry        string        `json:"default_registry"`
	ImagePullTimeout       time.Duration `json:"image_pull_timeout"`
	HealthCheckInterval    time.Duration `json:"health_check_interval"`
	ScalingCheckInterval   time.Duration `json:"scaling_check_interval"`
	ResourceUpdateInterval time.Duration `json:"resource_update_interval"`
	MaxConcurrentDeploys   int           `json:"max_concurrent_deploys"`
	EnableAutoScaling      bool          `json:"enable_auto_scaling"`
	EnableHealthChecks     bool          `json:"enable_health_checks"`
	LogLevel               string        `json:"log_level"`
}

// DefaultNFVConfig returns default NFV configuration
func DefaultNFVConfig() NFVConfig {
	return NFVConfig{
		DefaultRegistry:        "registry.local",
		ImagePullTimeout:       5 * time.Minute,
		HealthCheckInterval:    30 * time.Second,
		ScalingCheckInterval:   60 * time.Second,
		ResourceUpdateInterval: 15 * time.Second,
		MaxConcurrentDeploys:   10,
		EnableAutoScaling:      true,
		EnableHealthChecks:     true,
		LogLevel:               "info",
	}
}

// ContainerRuntime interface for container operations
type ContainerRuntime interface {
	PullImage(ctx context.Context, image NFImage) error
	CreateContainer(ctx context.Context, nf *NetworkFunction) (string, error)
	StartContainer(ctx context.Context, containerID string) error
	StopContainer(ctx context.Context, containerID string) error
	DeleteContainer(ctx context.Context, containerID string) error
	GetContainerStatus(ctx context.Context, containerID string) (string, error)
	GetContainerLogs(ctx context.Context, containerID string) ([]string, error)
}

// NetworkManager interface for network operations
type NetworkManager interface {
	CreateNetwork(ctx context.Context, name string, config NetworkConfig) (string, error)
	DeleteNetwork(ctx context.Context, networkID string) error
	AttachToNetwork(ctx context.Context, containerID, networkID string, config NetworkInterface) error
	DetachFromNetwork(ctx context.Context, containerID, networkID string) error
	GetNetworkStatus(ctx context.Context, networkID string) (string, error)
}

// ResourceManager interface for resource management
type ResourceManager interface {
	AllocateResources(ctx context.Context, requirements ResourceRequirements) (string, error)
	DeallocateResources(ctx context.Context, allocationID string) error
	GetResourceUsage(ctx context.Context, allocationID string) (ResourceUsage, error)
	CheckResourceAvailability(ctx context.Context, requirements ResourceRequirements) (bool, error)
}

// ResourceUsage represents current resource usage
type ResourceUsage struct {
	CPU     float64 `json:"cpu_percent"`
	Memory  int64   `json:"memory_mb"`
	Storage int64   `json:"storage_gb"`
	Network NetworkUsage `json:"network"`
}

// NetworkUsage represents network usage metrics
type NetworkUsage struct {
	RxBytes     int64 `json:"rx_bytes"`
	TxBytes     int64 `json:"tx_bytes"`
	RxPackets   int64 `json:"rx_packets"`
	TxPackets   int64 `json:"tx_packets"`
	Bandwidth   int64 `json:"bandwidth_mbps"`
}

// NFVEventHandler handles NFV events
type NFVEventHandler interface {
	OnFunctionDeployed(function *NetworkFunction)
	OnFunctionStarted(function *NetworkFunction)
	OnFunctionStopped(function *NetworkFunction)
	OnFunctionFailed(function *NetworkFunction, err error)
	OnFunctionScaled(function *NetworkFunction, oldReplicas, newReplicas int)
	OnServiceChainCreated(chain *NFVServiceChain)
	OnServiceChainUpdated(chain *NFVServiceChain)
	OnServiceChainDeleted(chainID string)
}

// NewNFVOrchestrator creates a new NFV orchestrator
func NewNFVOrchestrator(
	config NFVConfig,
	containerRuntime ContainerRuntime,
	networkManager NetworkManager,
	resourceManager ResourceManager,
) *NFVOrchestrator {
	ctx, cancel := context.WithCancel(context.Background())
	
	return &NFVOrchestrator{
		functions:        make(map[string]*NetworkFunction),
		serviceChains:    make(map[string]*NFVServiceChain),
		containerRuntime: containerRuntime,
		networkManager:   networkManager,
		resourceManager:  resourceManager,
		config:           config,
		ctx:              ctx,
		cancel:           cancel,
	}
}

// Start starts the NFV orchestrator
func (o *NFVOrchestrator) Start() error {
	log.Println("Starting NFV Orchestrator")
	
	// Start management loops
	if o.config.EnableHealthChecks {
		go o.healthCheckLoop()
	}
	
	if o.config.EnableAutoScaling {
		go o.scalingLoop()
	}
	
	go o.resourceMonitoringLoop()
	
	log.Println("NFV Orchestrator started")
	return nil
}

// Stop stops the NFV orchestrator
func (o *NFVOrchestrator) Stop() error {
	log.Println("Stopping NFV Orchestrator")
	o.cancel()
	return nil
}

// RegisterEventHandler registers an NFV event handler
func (o *NFVOrchestrator) RegisterEventHandler(handler NFVEventHandler) {
	o.eventMutex.Lock()
	defer o.eventMutex.Unlock()
	
	o.eventHandlers = append(o.eventHandlers, handler)
}

// DeployNetworkFunction deploys a new network function
func (o *NFVOrchestrator) DeployNetworkFunction(ctx context.Context, nf *NetworkFunction) error {
	if nf.ID == "" {
		nf.ID = uuid.New().String()
	}
	
	nf.Status = NFStatusDeploying
	nf.CreatedAt = time.Now()
	nf.UpdatedAt = time.Now()
	
	// Store the function
	o.functionsMutex.Lock()
	o.functions[nf.ID] = nf
	o.functionsMutex.Unlock()
	
	// Deploy in background
	go o.deployFunction(ctx, nf)
	
	log.Printf("Deploying network function: %s (%s)", nf.Name, nf.ID)
	return nil
}

// deployFunction performs the actual deployment of a network function
func (o *NFVOrchestrator) deployFunction(ctx context.Context, nf *NetworkFunction) {
	// Check resource availability
	available, err := o.resourceManager.CheckResourceAvailability(ctx, nf.Resources)
	if err != nil {
		log.Printf("Failed to check resource availability for %s: %v", nf.ID, err)
		o.updateFunctionStatus(nf.ID, NFStatusFailed)
		o.notifyFunctionFailed(nf, err)
		return
	}
	
	if !available {
		log.Printf("Insufficient resources for function %s", nf.ID)
		o.updateFunctionStatus(nf.ID, NFStatusFailed)
		o.notifyFunctionFailed(nf, fmt.Errorf("insufficient resources"))
		return
	}
	
	// Allocate resources
	allocationID, err := o.resourceManager.AllocateResources(ctx, nf.Resources)
	if err != nil {
		log.Printf("Failed to allocate resources for %s: %v", nf.ID, err)
		o.updateFunctionStatus(nf.ID, NFStatusFailed)
		o.notifyFunctionFailed(nf, err)
		return
	}
	
	// Store allocation ID
	if nf.Metadata == nil {
		nf.Metadata = make(map[string]interface{})
	}
	nf.Metadata["resource_allocation_id"] = allocationID
	
	// Pull container image
	if err := o.containerRuntime.PullImage(ctx, nf.Image); err != nil {
		log.Printf("Failed to pull image for %s: %v", nf.ID, err)
		o.resourceManager.DeallocateResources(ctx, allocationID)
		o.updateFunctionStatus(nf.ID, NFStatusFailed)
		o.notifyFunctionFailed(nf, err)
		return
	}
	
	// Create container
	containerID, err := o.containerRuntime.CreateContainer(ctx, nf)
	if err != nil {
		log.Printf("Failed to create container for %s: %v", nf.ID, err)
		o.resourceManager.DeallocateResources(ctx, allocationID)
		o.updateFunctionStatus(nf.ID, NFStatusFailed)
		o.notifyFunctionFailed(nf, err)
		return
	}
	
	// Store container ID
	nf.Metadata["container_id"] = containerID
	
	// Configure networking
	if err := o.configureNetworking(ctx, nf, containerID); err != nil {
		log.Printf("Failed to configure networking for %s: %v", nf.ID, err)
		o.containerRuntime.DeleteContainer(ctx, containerID)
		o.resourceManager.DeallocateResources(ctx, allocationID)
		o.updateFunctionStatus(nf.ID, NFStatusFailed)
		o.notifyFunctionFailed(nf, err)
		return
	}
	
	// Start container
	if err := o.containerRuntime.StartContainer(ctx, containerID); err != nil {
		log.Printf("Failed to start container for %s: %v", nf.ID, err)
		o.containerRuntime.DeleteContainer(ctx, containerID)
		o.resourceManager.DeallocateResources(ctx, allocationID)
		o.updateFunctionStatus(nf.ID, NFStatusFailed)
		o.notifyFunctionFailed(nf, err)
		return
	}
	
	// Update status
	o.updateFunctionStatus(nf.ID, NFStatusRunning)
	o.notifyFunctionDeployed(nf)
	o.notifyFunctionStarted(nf)
	
	log.Printf("Successfully deployed network function: %s", nf.Name)
}

// configureNetworking configures networking for a network function
func (o *NFVOrchestrator) configureNetworking(ctx context.Context, nf *NetworkFunction, containerID string) error {
	for _, iface := range nf.NetworkConfig.Interfaces {
		// Attach container to network
		if err := o.networkManager.AttachToNetwork(ctx, containerID, iface.Network, iface); err != nil {
			return fmt.Errorf("failed to attach to network %s: %w", iface.Network, err)
		}
		
		log.Printf("Attached function %s to network %s via interface %s", nf.ID, iface.Network, iface.Name)
	}
	
	return nil
}

// StopNetworkFunction stops a network function
func (o *NFVOrchestrator) StopNetworkFunction(ctx context.Context, functionID string) error {
	o.functionsMutex.RLock()
	nf, exists := o.functions[functionID]
	o.functionsMutex.RUnlock()
	
	if !exists {
		return fmt.Errorf("network function not found: %s", functionID)
	}
	
	containerID, ok := nf.Metadata["container_id"].(string)
	if !ok {
		return fmt.Errorf("container ID not found for function %s", functionID)
	}
	
	// Stop container
	if err := o.containerRuntime.StopContainer(ctx, containerID); err != nil {
		return fmt.Errorf("failed to stop container: %w", err)
	}
	
	// Update status
	o.updateFunctionStatus(functionID, NFStatusStopped)
	o.notifyFunctionStopped(nf)
	
	log.Printf("Stopped network function: %s", nf.Name)
	return nil
}

// DeleteNetworkFunction deletes a network function
func (o *NFVOrchestrator) DeleteNetworkFunction(ctx context.Context, functionID string) error {
	o.functionsMutex.RLock()
	nf, exists := o.functions[functionID]
	o.functionsMutex.RUnlock()
	
	if !exists {
		return fmt.Errorf("network function not found: %s", functionID)
	}
	
	// Stop if running
	if nf.Status == NFStatusRunning {
		if err := o.StopNetworkFunction(ctx, functionID); err != nil {
			log.Printf("Warning: Failed to stop function before deletion: %v", err)
		}
	}
	
	containerID, ok := nf.Metadata["container_id"].(string)
	if ok {
		// Detach from networks
		for _, iface := range nf.NetworkConfig.Interfaces {
			if err := o.networkManager.DetachFromNetwork(ctx, containerID, iface.Network); err != nil {
				log.Printf("Warning: Failed to detach from network %s: %v", iface.Network, err)
			}
		}
		
		// Delete container
		if err := o.containerRuntime.DeleteContainer(ctx, containerID); err != nil {
			log.Printf("Warning: Failed to delete container: %v", err)
		}
	}
	
	// Deallocate resources
	if allocationID, ok := nf.Metadata["resource_allocation_id"].(string); ok {
		if err := o.resourceManager.DeallocateResources(ctx, allocationID); err != nil {
			log.Printf("Warning: Failed to deallocate resources: %v", err)
		}
	}
	
	// Remove from registry
	o.functionsMutex.Lock()
	delete(o.functions, functionID)
	o.functionsMutex.Unlock()
	
	log.Printf("Deleted network function: %s", nf.Name)
	return nil
}

// CreateServiceChain creates a new NFV service chain
func (o *NFVOrchestrator) CreateServiceChain(ctx context.Context, chain *NFVServiceChain) error {
	if chain.ID == "" {
		chain.ID = uuid.New().String()
	}
	
	chain.Status = ServiceChainStatusDeploying
	chain.CreatedAt = time.Now()
	chain.UpdatedAt = time.Now()
	
	// Validate service chain
	if err := o.validateServiceChain(chain); err != nil {
		return fmt.Errorf("service chain validation failed: %w", err)
	}
	
	// Store the service chain
	o.serviceChainsMutex.Lock()
	o.serviceChains[chain.ID] = chain
	o.serviceChainsMutex.Unlock()
	
	// Deploy in background
	go o.deployServiceChain(ctx, chain)
	
	log.Printf("Creating service chain: %s (%s)", chain.Name, chain.ID)
	return nil
}

// validateServiceChain validates a service chain configuration
func (o *NFVOrchestrator) validateServiceChain(chain *NFVServiceChain) error {
	if chain.Name == "" {
		return fmt.Errorf("service chain name cannot be empty")
	}
	
	if len(chain.Functions) == 0 {
		return fmt.Errorf("service chain must have at least one function")
	}
	
	// Validate that all functions exist
	o.functionsMutex.RLock()
	for _, chainFunc := range chain.Functions {
		if _, exists := o.functions[chainFunc.FunctionID]; !exists {
			o.functionsMutex.RUnlock()
			return fmt.Errorf("function not found: %s", chainFunc.FunctionID)
		}
	}
	o.functionsMutex.RUnlock()
	
	// Validate function ordering
	orders := make(map[int]bool)
	for _, chainFunc := range chain.Functions {
		if chainFunc.Order < 0 {
			return fmt.Errorf("function order must be non-negative")
		}
		if orders[chainFunc.Order] {
			return fmt.Errorf("duplicate function order: %d", chainFunc.Order)
		}
		orders[chainFunc.Order] = true
	}
	
	return nil
}

// deployServiceChain deploys a service chain
func (o *NFVOrchestrator) deployServiceChain(ctx context.Context, chain *NFVServiceChain) {
	// Implementation would configure network flows and policies
	// to route traffic through the chain of functions
	
	// For now, just mark as active
	o.updateServiceChainStatus(chain.ID, ServiceChainStatusActive)
	o.notifyServiceChainCreated(chain)
	
	log.Printf("Successfully deployed service chain: %s", chain.Name)
}

// ScaleNetworkFunction scales a network function
func (o *NFVOrchestrator) ScaleNetworkFunction(ctx context.Context, functionID string, replicas int) error {
	o.functionsMutex.RLock()
	nf, exists := o.functions[functionID]
	o.functionsMutex.RUnlock()
	
	if !exists {
		return fmt.Errorf("network function not found: %s", functionID)
	}
	
	if replicas < nf.Scaling.MinReplicas {
		return fmt.Errorf("replicas %d below minimum %d", replicas, nf.Scaling.MinReplicas)
	}
	
	if replicas > nf.Scaling.MaxReplicas {
		return fmt.Errorf("replicas %d above maximum %d", replicas, nf.Scaling.MaxReplicas)
	}
	
	// Get current replica count (simplified)
	currentReplicas := 1 // Would get from actual deployment
	
	if replicas == currentReplicas {
		return nil // No change needed
	}
	
	// Update status
	o.updateFunctionStatus(functionID, NFStatusScaling)
	
	// Perform scaling (implementation would create/destroy replicas)
	// For now, just notify
	o.notifyFunctionScaled(nf, currentReplicas, replicas)
	
	// Update status back to running
	o.updateFunctionStatus(functionID, NFStatusRunning)
	
	log.Printf("Scaled function %s from %d to %d replicas", nf.Name, currentReplicas, replicas)
	return nil
}

// healthCheckLoop performs periodic health checks on network functions
func (o *NFVOrchestrator) healthCheckLoop() {
	ticker := time.NewTicker(o.config.HealthCheckInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-o.ctx.Done():
			return
		case <-ticker.C:
			o.performHealthChecks()
		}
	}
}

// performHealthChecks performs health checks on all running functions
func (o *NFVOrchestrator) performHealthChecks() {
	o.functionsMutex.RLock()
	functions := make([]*NetworkFunction, 0, len(o.functions))
	for _, nf := range o.functions {
		if nf.Status == NFStatusRunning && nf.HealthCheck.Enabled {
			functions = append(functions, nf)
		}
	}
	o.functionsMutex.RUnlock()
	
	for _, nf := range functions {
		go o.performHealthCheck(nf)
	}
}

// performHealthCheck performs health check on a single function
func (o *NFVOrchestrator) performHealthCheck(nf *NetworkFunction) {
	containerID, ok := nf.Metadata["container_id"].(string)
	if !ok {
		return
	}
	
	// Get container status
	status, err := o.containerRuntime.GetContainerStatus(context.Background(), containerID)
	if err != nil {
		log.Printf("Health check failed for %s: %v", nf.Name, err)
		o.updateFunctionStatus(nf.ID, NFStatusFailed)
		o.notifyFunctionFailed(nf, err)
		return
	}
	
	if status != "running" {
		log.Printf("Function %s is not running (status: %s)", nf.Name, status)
		o.updateFunctionStatus(nf.ID, NFStatusFailed)
		o.notifyFunctionFailed(nf, fmt.Errorf("container not running"))
		return
	}
	
	// Perform application-level health checks based on configuration
	if nf.HealthCheck.HTTPGet != nil {
		// Would perform HTTP health check
	} else if nf.HealthCheck.TCPSocket != nil {
		// Would perform TCP health check
	} else if nf.HealthCheck.Exec != nil {
		// Would perform exec health check
	}
}

// scalingLoop performs periodic auto-scaling checks
func (o *NFVOrchestrator) scalingLoop() {
	ticker := time.NewTicker(o.config.ScalingCheckInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-o.ctx.Done():
			return
		case <-ticker.C:
			o.performScalingChecks()
		}
	}
}

// performScalingChecks performs auto-scaling checks on all functions
func (o *NFVOrchestrator) performScalingChecks() {
	o.functionsMutex.RLock()
	functions := make([]*NetworkFunction, 0, len(o.functions))
	for _, nf := range o.functions {
		if nf.Status == NFStatusRunning && nf.Scaling.Enabled {
			functions = append(functions, nf)
		}
	}
	o.functionsMutex.RUnlock()
	
	for _, nf := range functions {
		go o.checkScaling(nf)
	}
}

// checkScaling checks if a function needs to be scaled
func (o *NFVOrchestrator) checkScaling(nf *NetworkFunction) {
	allocationID, ok := nf.Metadata["resource_allocation_id"].(string)
	if !ok {
		return
	}
	
	// Get resource usage
	usage, err := o.resourceManager.GetResourceUsage(context.Background(), allocationID)
	if err != nil {
		log.Printf("Failed to get resource usage for %s: %v", nf.Name, err)
		return
	}
	
	// Check scaling metrics
	for _, metric := range nf.Scaling.TargetMetrics {
		var currentValue float64
		var shouldScale bool
		
		switch metric.Type {
		case "CPU":
			currentValue = usage.CPU
		case "Memory":
			currentValue = float64(usage.Memory) / float64(nf.Resources.Memory.Requests) * 100
		case "Network":
			currentValue = float64(usage.Network.Bandwidth)
		default:
			continue // Unknown metric type
		}
		
		// Simple scaling logic
		if currentValue > metric.TargetValue*1.2 { // Scale up threshold
			shouldScale = true
		} else if currentValue < metric.TargetValue*0.8 { // Scale down threshold
			shouldScale = true
		}
		
		if shouldScale {
			// Calculate new replica count (simplified)
			currentReplicas := 1 // Would get from actual deployment
			newReplicas := currentReplicas
			
			if currentValue > metric.TargetValue {
				newReplicas = currentReplicas + 1
			} else {
				newReplicas = currentReplicas - 1
			}
			
			// Scale the function
			if err := o.ScaleNetworkFunction(context.Background(), nf.ID, newReplicas); err != nil {
				log.Printf("Failed to scale function %s: %v", nf.Name, err)
			}
			break // Only scale based on first matching metric
		}
	}
}

// resourceMonitoringLoop monitors resource usage
func (o *NFVOrchestrator) resourceMonitoringLoop() {
	ticker := time.NewTicker(o.config.ResourceUpdateInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-o.ctx.Done():
			return
		case <-ticker.C:
			o.updateResourceMetrics()
		}
	}
}

// updateResourceMetrics updates resource metrics for all functions
func (o *NFVOrchestrator) updateResourceMetrics() {
	o.functionsMutex.RLock()
	functions := make([]*NetworkFunction, 0, len(o.functions))
	for _, nf := range o.functions {
		if nf.Status == NFStatusRunning {
			functions = append(functions, nf)
		}
	}
	o.functionsMutex.RUnlock()
	
	for _, nf := range functions {
		go o.updateFunctionMetrics(nf)
	}
}

// updateFunctionMetrics updates metrics for a single function
func (o *NFVOrchestrator) updateFunctionMetrics(nf *NetworkFunction) {
	allocationID, ok := nf.Metadata["resource_allocation_id"].(string)
	if !ok {
		return
	}
	
	usage, err := o.resourceManager.GetResourceUsage(context.Background(), allocationID)
	if err != nil {
		log.Printf("Failed to get resource usage for %s: %v", nf.Name, err)
		return
	}
	
	// Store metrics in function metadata
	if nf.Metadata == nil {
		nf.Metadata = make(map[string]interface{})
	}
	nf.Metadata["resource_usage"] = usage
	nf.Metadata["metrics_updated"] = time.Now()
}

// Helper methods for status updates

func (o *NFVOrchestrator) updateFunctionStatus(functionID string, status NFStatus) {
	o.functionsMutex.Lock()
	defer o.functionsMutex.Unlock()
	
	if nf, exists := o.functions[functionID]; exists {
		nf.Status = status
		nf.UpdatedAt = time.Now()
	}
}

func (o *NFVOrchestrator) updateServiceChainStatus(chainID string, status ServiceChainStatus) {
	o.serviceChainsMutex.Lock()
	defer o.serviceChainsMutex.Unlock()
	
	if chain, exists := o.serviceChains[chainID]; exists {
		chain.Status = status
		chain.UpdatedAt = time.Now()
	}
}

// Event notification methods

func (o *NFVOrchestrator) notifyFunctionDeployed(nf *NetworkFunction) {
	o.eventMutex.RLock()
	defer o.eventMutex.RUnlock()
	
	for _, handler := range o.eventHandlers {
		go handler.OnFunctionDeployed(nf)
	}
}

func (o *NFVOrchestrator) notifyFunctionStarted(nf *NetworkFunction) {
	o.eventMutex.RLock()
	defer o.eventMutex.RUnlock()
	
	for _, handler := range o.eventHandlers {
		go handler.OnFunctionStarted(nf)
	}
}

func (o *NFVOrchestrator) notifyFunctionStopped(nf *NetworkFunction) {
	o.eventMutex.RLock()
	defer o.eventMutex.RUnlock()
	
	for _, handler := range o.eventHandlers {
		go handler.OnFunctionStopped(nf)
	}
}

func (o *NFVOrchestrator) notifyFunctionFailed(nf *NetworkFunction, err error) {
	o.eventMutex.RLock()
	defer o.eventMutex.RUnlock()
	
	for _, handler := range o.eventHandlers {
		go handler.OnFunctionFailed(nf, err)
	}
}

func (o *NFVOrchestrator) notifyFunctionScaled(nf *NetworkFunction, oldReplicas, newReplicas int) {
	o.eventMutex.RLock()
	defer o.eventMutex.RUnlock()
	
	for _, handler := range o.eventHandlers {
		go handler.OnFunctionScaled(nf, oldReplicas, newReplicas)
	}
}

func (o *NFVOrchestrator) notifyServiceChainCreated(chain *NFVServiceChain) {
	o.eventMutex.RLock()
	defer o.eventMutex.RUnlock()
	
	for _, handler := range o.eventHandlers {
		go handler.OnServiceChainCreated(chain)
	}
}

func (o *NFVOrchestrator) notifyServiceChainUpdated(chain *NFVServiceChain) {
	o.eventMutex.RLock()
	defer o.eventMutex.RUnlock()
	
	for _, handler := range o.eventHandlers {
		go handler.OnServiceChainUpdated(chain)
	}
}

func (o *NFVOrchestrator) notifyServiceChainDeleted(chainID string) {
	o.eventMutex.RLock()
	defer o.eventMutex.RUnlock()
	
	for _, handler := range o.eventHandlers {
		go handler.OnServiceChainDeleted(chainID)
	}
}

// Public API methods

// GetNetworkFunction returns a network function by ID
func (o *NFVOrchestrator) GetNetworkFunction(functionID string) (*NetworkFunction, error) {
	o.functionsMutex.RLock()
	defer o.functionsMutex.RUnlock()
	
	nf, exists := o.functions[functionID]
	if !exists {
		return nil, fmt.Errorf("network function not found: %s", functionID)
	}
	
	return nf, nil
}

// ListNetworkFunctions returns all network functions
func (o *NFVOrchestrator) ListNetworkFunctions() []*NetworkFunction {
	o.functionsMutex.RLock()
	defer o.functionsMutex.RUnlock()
	
	functions := make([]*NetworkFunction, 0, len(o.functions))
	for _, nf := range o.functions {
		functions = append(functions, nf)
	}
	
	return functions
}

// GetServiceChain returns a service chain by ID
func (o *NFVOrchestrator) GetServiceChain(chainID string) (*NFVServiceChain, error) {
	o.serviceChainsMutex.RLock()
	defer o.serviceChainsMutex.RUnlock()
	
	chain, exists := o.serviceChains[chainID]
	if !exists {
		return nil, fmt.Errorf("service chain not found: %s", chainID)
	}
	
	return chain, nil
}

// ListServiceChains returns all service chains
func (o *NFVOrchestrator) ListServiceChains() []*NFVServiceChain {
	o.serviceChainsMutex.RLock()
	defer o.serviceChainsMutex.RUnlock()
	
	chains := make([]*NFVServiceChain, 0, len(o.serviceChains))
	for _, chain := range o.serviceChains {
		chains = append(chains, chain)
	}
	
	return chains
}