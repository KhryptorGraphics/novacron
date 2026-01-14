package multicloud

import (
	"context"
	"time"
)

// CloudProvider represents the interface that all cloud providers must implement
type CloudProvider interface {
	// Provider Information
	GetProviderType() CloudProviderType
	GetName() string
	GetRegions() []string
	GetAvailabilityZones(region string) []string

	// Authentication and Configuration
	Initialize(ctx context.Context, config CloudProviderConfig) error
	Validate(ctx context.Context) error
	GetCapabilities() []CloudCapability

	// VM Operations
	CreateVM(ctx context.Context, request *VMCreateRequest) (*VMInstance, error)
	GetVM(ctx context.Context, vmID string) (*VMInstance, error)
	ListVMs(ctx context.Context, filters map[string]string) ([]*VMInstance, error)
	UpdateVM(ctx context.Context, vmID string, updates *VMUpdateRequest) error
	DeleteVM(ctx context.Context, vmID string) error

	// VM Lifecycle
	StartVM(ctx context.Context, vmID string) error
	StopVM(ctx context.Context, vmID string) error
	RestartVM(ctx context.Context, vmID string) error
	SuspendVM(ctx context.Context, vmID string) error
	ResumeVM(ctx context.Context, vmID string) error

	// Migration Support
	ExportVM(ctx context.Context, vmID string, format VMExportFormat) (*VMExportData, error)
	ImportVM(ctx context.Context, data *VMExportData) (*VMInstance, error)
	SnapshotVM(ctx context.Context, vmID string, name string) (*VMSnapshot, error)
	RestoreSnapshot(ctx context.Context, snapshotID string) error

	// Resource Management
	GetResourceQuota(ctx context.Context) (*ResourceQuota, error)
	GetResourceUsage(ctx context.Context) (*ResourceUsage, error)
	GetPricing(ctx context.Context, resourceType string, region string) (*PricingInfo, error)

	// Networking
	CreateNetwork(ctx context.Context, request *NetworkCreateRequest) (*Network, error)
	GetNetwork(ctx context.Context, networkID string) (*Network, error)
	ListNetworks(ctx context.Context) ([]*Network, error)
	DeleteNetwork(ctx context.Context, networkID string) error

	// Storage
	CreateStorage(ctx context.Context, request *StorageCreateRequest) (*Storage, error)
	GetStorage(ctx context.Context, storageID string) (*Storage, error)
	ListStorage(ctx context.Context) ([]*Storage, error)
	DeleteStorage(ctx context.Context, storageID string) error

	// Monitoring and Health
	GetVMMetrics(ctx context.Context, vmID string, start, end time.Time) (*VMMetrics, error)
	GetProviderHealth(ctx context.Context) (*ProviderHealthStatus, error)
	
	// Cost Management
	GetCostEstimate(ctx context.Context, request *CostEstimateRequest) (*CostEstimate, error)
	GetBillingData(ctx context.Context, start, end time.Time) (*BillingData, error)

	// Compliance and Security
	GetComplianceStatus(ctx context.Context) (*ComplianceStatus, error)
	GetSecurityGroups(ctx context.Context) ([]*SecurityGroup, error)
	CreateSecurityGroup(ctx context.Context, request *SecurityGroupRequest) (*SecurityGroup, error)
}

// CloudProviderType represents different cloud provider types
type CloudProviderType string

const (
	ProviderAWS        CloudProviderType = "aws"
	ProviderAzure      CloudProviderType = "azure"
	ProviderGCP        CloudProviderType = "gcp"
	ProviderOnPremise  CloudProviderType = "on-premise"
	ProviderOpenStack  CloudProviderType = "openstack"
	ProviderVMware     CloudProviderType = "vmware"
	ProviderHyperV     CloudProviderType = "hyperv"
	ProviderKVM        CloudProviderType = "kvm"
)

// CloudCapability represents capabilities that a cloud provider supports
type CloudCapability string

const (
	CapabilityVMLiveMigration    CloudCapability = "vm_live_migration"
	CapabilityAutoScaling        CloudCapability = "auto_scaling"
	CapabilityLoadBalancing      CloudCapability = "load_balancing"
	CapabilityBlockStorage       CloudCapability = "block_storage"
	CapabilityObjectStorage      CloudCapability = "object_storage"
	CapabilityContainerRegistry  CloudCapability = "container_registry"
	CapabilityKubernetes         CloudCapability = "kubernetes"
	CapabilitySpotInstances      CloudCapability = "spot_instances"
	CapabilityReservedInstances  CloudCapability = "reserved_instances"
	CapabilityGPUCompute         CloudCapability = "gpu_compute"
	CapabilityHPCCompute         CloudCapability = "hpc_compute"
	CapabilityDatabaseServices   CloudCapability = "database_services"
	CapabilityMLServices         CloudCapability = "ml_services"
	CapabilityNetworkACLs        CloudCapability = "network_acls"
	CapabilityPrivateNetworking  CloudCapability = "private_networking"
	CapabilityVPNGateway         CloudCapability = "vpn_gateway"
	CapabilityDirectConnect      CloudCapability = "direct_connect"
	CapabilityIdentityManagement CloudCapability = "identity_management"
	CapabilityKeyManagement      CloudCapability = "key_management"
	CapabilityAuditLogging       CloudCapability = "audit_logging"
	CapabilityBackupServices     CloudCapability = "backup_services"
	CapabilityDisasterRecovery   CloudCapability = "disaster_recovery"
)

// CloudProviderConfig contains configuration for cloud providers
type CloudProviderConfig struct {
	Type          CloudProviderType      `json:"type"`
	Name          string                 `json:"name"`
	Credentials   map[string]string      `json:"credentials"`
	Regions       []string               `json:"regions"`
	DefaultRegion string                 `json:"default_region"`
	Endpoints     map[string]string      `json:"endpoints"`
	Options       map[string]interface{} `json:"options"`
}

// VMInstance represents a VM instance across any cloud provider
type VMInstance struct {
	ID               string                 `json:"id"`
	Name             string                 `json:"name"`
	Provider         CloudProviderType      `json:"provider"`
	Region           string                 `json:"region"`
	AvailabilityZone string                 `json:"availability_zone"`
	InstanceType     string                 `json:"instance_type"`
	State            VMState                `json:"state"`
	PublicIP         string                 `json:"public_ip,omitempty"`
	PrivateIP        string                 `json:"private_ip,omitempty"`
	ImageID          string                 `json:"image_id"`
	KeyPair          string                 `json:"key_pair,omitempty"`
	SecurityGroups   []string               `json:"security_groups"`
	Tags             map[string]string      `json:"tags"`
	CreatedAt        time.Time              `json:"created_at"`
	UpdatedAt        time.Time              `json:"updated_at"`
	Metadata         map[string]interface{} `json:"metadata"`
	
	// Resource specifications
	CPU              int    `json:"cpu"`
	Memory           int64  `json:"memory"` // in MB
	Storage          int64  `json:"storage"` // in GB
	NetworkBandwidth int64  `json:"network_bandwidth"` // in Mbps
	
	// Cost information
	HourlyRate       float64 `json:"hourly_rate"`
	MonthlyEstimate  float64 `json:"monthly_estimate"`
}

// VMState represents the state of a VM
type VMState string

const (
	VMStatePending     VMState = "pending"
	VMStateRunning     VMState = "running"
	VMStateStopping    VMState = "stopping"
	VMStateStopped     VMState = "stopped"
	VMStateTerminating VMState = "terminating"
	VMStateTerminated  VMState = "terminated"
	VMStateSuspended   VMState = "suspended"
	VMStateError       VMState = "error"
)

// VMCreateRequest represents a request to create a VM
type VMCreateRequest struct {
	Name             string            `json:"name"`
	InstanceType     string            `json:"instance_type"`
	ImageID          string            `json:"image_id"`
	Region           string            `json:"region"`
	AvailabilityZone string            `json:"availability_zone,omitempty"`
	KeyPair          string            `json:"key_pair,omitempty"`
	SecurityGroups   []string          `json:"security_groups"`
	UserData         string            `json:"user_data,omitempty"`
	Tags             map[string]string `json:"tags"`
	
	// Resource specifications
	CPU              int   `json:"cpu,omitempty"`
	Memory           int64 `json:"memory,omitempty"` // in MB
	Storage          int64 `json:"storage,omitempty"` // in GB
	NetworkBandwidth int64 `json:"network_bandwidth,omitempty"` // in Mbps
	
	// Advanced options
	SpotInstance       bool                   `json:"spot_instance"`
	MaxSpotPrice       float64                `json:"max_spot_price,omitempty"`
	PlacementGroup     string                 `json:"placement_group,omitempty"`
	DedicatedTenancy   bool                   `json:"dedicated_tenancy"`
	IamInstanceProfile string                 `json:"iam_instance_profile,omitempty"`
	MonitoringEnabled  bool                   `json:"monitoring_enabled"`
	EbsOptimized       bool                   `json:"ebs_optimized"`
	SriovNetSupport    bool                   `json:"sriov_net_support"`
	CustomOptions      map[string]interface{} `json:"custom_options,omitempty"`
}

// VMUpdateRequest represents a request to update a VM
type VMUpdateRequest struct {
	Name           *string           `json:"name,omitempty"`
	InstanceType   *string           `json:"instance_type,omitempty"`
	SecurityGroups []string          `json:"security_groups,omitempty"`
	Tags           map[string]string `json:"tags,omitempty"`
	
	// Resource updates
	CPU    *int   `json:"cpu,omitempty"`
	Memory *int64 `json:"memory,omitempty"`
}

// VMExportFormat represents different export formats
type VMExportFormat string

const (
	ExportFormatOVF    VMExportFormat = "ovf"
	ExportFormatOVA    VMExportFormat = "ova"
	ExportFormatVMDK   VMExportFormat = "vmdk"
	ExportFormatQCOW2  VMExportFormat = "qcow2"
	ExportFormatVHD    VMExportFormat = "vhd"
	ExportFormatVHDX   VMExportFormat = "vhdx"
	ExportFormatRAW    VMExportFormat = "raw"
	ExportFormatDocker VMExportFormat = "docker"
)

// VMExportData represents exported VM data
type VMExportData struct {
	Format      VMExportFormat         `json:"format"`
	Size        int64                  `json:"size"` // in bytes
	Checksum    string                 `json:"checksum"`
	DownloadURL string                 `json:"download_url,omitempty"`
	S3Location  string                 `json:"s3_location,omitempty"`
	Metadata    map[string]interface{} `json:"metadata"`
	ExpiresAt   time.Time              `json:"expires_at"`
}

// VMSnapshot represents a VM snapshot
type VMSnapshot struct {
	ID          string            `json:"id"`
	Name        string            `json:"name"`
	VMID        string            `json:"vm_id"`
	Description string            `json:"description"`
	Size        int64             `json:"size"` // in bytes
	State       string            `json:"state"`
	Progress    int               `json:"progress"` // 0-100
	Tags        map[string]string `json:"tags"`
	CreatedAt   time.Time         `json:"created_at"`
}

// ResourceQuota represents resource limits for a cloud provider
type ResourceQuota struct {
	MaxVMs              int   `json:"max_vms"`
	MaxCPU              int   `json:"max_cpu"`
	MaxMemory           int64 `json:"max_memory"` // in MB
	MaxStorage          int64 `json:"max_storage"` // in GB
	MaxNetworks         int   `json:"max_networks"`
	MaxSecurityGroups   int   `json:"max_security_groups"`
	MaxSnapshots        int   `json:"max_snapshots"`
	MaxFloatingIPs      int   `json:"max_floating_ips"`
	MaxLoadBalancers    int   `json:"max_load_balancers"`
}

// ResourceUsage represents current resource usage
type ResourceUsage struct {
	UsedVMs           int     `json:"used_vms"`
	UsedCPU           int     `json:"used_cpu"`
	UsedMemory        int64   `json:"used_memory"` // in MB
	UsedStorage       int64   `json:"used_storage"` // in GB
	UsedNetworks      int     `json:"used_networks"`
	UsedSecurityGroups int    `json:"used_security_groups"`
	UsedSnapshots     int     `json:"used_snapshots"`
	UsedFloatingIPs   int     `json:"used_floating_ips"`
	UsedLoadBalancers int     `json:"used_load_balancers"`
	TotalCost         float64 `json:"total_cost"` // monthly cost
}

// PricingInfo represents pricing information for resources
type PricingInfo struct {
	ResourceType    string                 `json:"resource_type"`
	Region          string                 `json:"region"`
	Currency        string                 `json:"currency"`
	PricePerHour    float64                `json:"price_per_hour,omitempty"`
	PricePerMonth   float64                `json:"price_per_month,omitempty"`
	PricePerUnit    float64                `json:"price_per_unit,omitempty"`
	Unit            string                 `json:"unit,omitempty"`
	TierPricing     []PricingTier          `json:"tier_pricing,omitempty"`
	ReservedPricing map[string]float64     `json:"reserved_pricing,omitempty"`
	SpotPricing     *SpotPricingInfo       `json:"spot_pricing,omitempty"`
	Metadata        map[string]interface{} `json:"metadata,omitempty"`
}

// PricingTier represents tiered pricing
type PricingTier struct {
	From  int     `json:"from"`
	To    int     `json:"to"`   // -1 for unlimited
	Price float64 `json:"price"`
}

// SpotPricingInfo represents spot instance pricing
type SpotPricingInfo struct {
	CurrentPrice float64   `json:"current_price"`
	AveragePrice float64   `json:"average_price"`
	MinPrice     float64   `json:"min_price"`
	MaxPrice     float64   `json:"max_price"`
	LastUpdated  time.Time `json:"last_updated"`
}

// Network represents a cloud network
type Network struct {
	ID          string            `json:"id"`
	Name        string            `json:"name"`
	CIDR        string            `json:"cidr"`
	Region      string            `json:"region"`
	State       string            `json:"state"`
	Subnets     []Subnet          `json:"subnets"`
	Tags        map[string]string `json:"tags"`
	CreatedAt   time.Time         `json:"created_at"`
}

// Subnet represents a network subnet
type Subnet struct {
	ID               string `json:"id"`
	Name             string `json:"name"`
	CIDR             string `json:"cidr"`
	AvailabilityZone string `json:"availability_zone"`
	State            string `json:"state"`
	AvailableIPs     int    `json:"available_ips"`
}

// NetworkCreateRequest represents a request to create a network
type NetworkCreateRequest struct {
	Name    string            `json:"name"`
	CIDR    string            `json:"cidr"`
	Region  string            `json:"region"`
	Subnets []SubnetRequest   `json:"subnets"`
	Tags    map[string]string `json:"tags"`
}

// SubnetRequest represents a request to create a subnet
type SubnetRequest struct {
	Name             string `json:"name"`
	CIDR             string `json:"cidr"`
	AvailabilityZone string `json:"availability_zone"`
}

// Storage represents cloud storage
type Storage struct {
	ID           string            `json:"id"`
	Name         string            `json:"name"`
	Type         StorageType       `json:"type"`
	Size         int64             `json:"size"` // in GB
	IOPS         int               `json:"iops,omitempty"`
	Throughput   int               `json:"throughput,omitempty"` // MB/s
	Encrypted    bool              `json:"encrypted"`
	Region       string            `json:"region"`
	AttachedTo   string            `json:"attached_to,omitempty"`
	State        string            `json:"state"`
	Tags         map[string]string `json:"tags"`
	CreatedAt    time.Time         `json:"created_at"`
}

// StorageType represents different storage types
type StorageType string

const (
	StorageTypeBlockSSD    StorageType = "block_ssd"
	StorageTypeBlockHDD    StorageType = "block_hdd"
	StorageTypeBlockNVMe   StorageType = "block_nvme"
	StorageTypeObject      StorageType = "object"
	StorageTypeFile        StorageType = "file"
	StorageTypeArchive     StorageType = "archive"
)

// StorageCreateRequest represents a request to create storage
type StorageCreateRequest struct {
	Name       string            `json:"name"`
	Type       StorageType       `json:"type"`
	Size       int64             `json:"size"` // in GB
	IOPS       int               `json:"iops,omitempty"`
	Throughput int               `json:"throughput,omitempty"`
	Encrypted  bool              `json:"encrypted"`
	Region     string            `json:"region"`
	Tags       map[string]string `json:"tags"`
}

// VMMetrics represents VM performance metrics
type VMMetrics struct {
	VMID      string                   `json:"vm_id"`
	StartTime time.Time                `json:"start_time"`
	EndTime   time.Time                `json:"end_time"`
	Metrics   map[string][]MetricPoint `json:"metrics"`
}

// MetricPoint represents a single metric data point
type MetricPoint struct {
	Timestamp time.Time `json:"timestamp"`
	Value     float64   `json:"value"`
}

// ProviderHealthStatus represents the health status of a cloud provider
type ProviderHealthStatus struct {
	Provider      CloudProviderType      `json:"provider"`
	Overall       HealthStatus           `json:"overall"`
	Services      map[string]HealthStatus `json:"services"`
	Regions       map[string]HealthStatus `json:"regions"`
	LastChecked   time.Time              `json:"last_checked"`
	Issues        []HealthIssue          `json:"issues,omitempty"`
}

// HealthStatus represents a health status
type HealthStatus string

const (
	HealthStatusHealthy   HealthStatus = "healthy"
	HealthStatusDegraded  HealthStatus = "degraded"
	HealthStatusUnhealthy HealthStatus = "unhealthy"
	HealthStatusUnknown   HealthStatus = "unknown"
)

// HealthIssue represents a health issue
type HealthIssue struct {
	ID          string    `json:"id"`
	Service     string    `json:"service"`
	Region      string    `json:"region,omitempty"`
	Severity    string    `json:"severity"`
	Description string    `json:"description"`
	StartTime   time.Time `json:"start_time"`
	EndTime     *time.Time `json:"end_time,omitempty"`
}

// CostEstimateRequest represents a request for cost estimation
type CostEstimateRequest struct {
	Resources        []ResourceEstimate `json:"resources"`
	Region           string             `json:"region"`
	Duration         string             `json:"duration"` // e.g., "1h", "24h", "30d"
	ReservedInstance bool               `json:"reserved_instance"`
	SpotInstance     bool               `json:"spot_instance"`
}

// ResourceEstimate represents a resource for cost estimation
type ResourceEstimate struct {
	Type     string                 `json:"type"` // vm, storage, network, etc.
	Config   map[string]interface{} `json:"config"`
	Quantity int                    `json:"quantity"`
}

// CostEstimate represents a cost estimation response
type CostEstimate struct {
	TotalCost    float64                `json:"total_cost"`
	Currency     string                 `json:"currency"`
	Duration     string                 `json:"duration"`
	Breakdown    []CostBreakdown        `json:"breakdown"`
	Assumptions  map[string]interface{} `json:"assumptions"`
	Confidence   float64                `json:"confidence"` // 0-1
	CreatedAt    time.Time              `json:"created_at"`
}

// CostBreakdown represents cost breakdown by resource type
type CostBreakdown struct {
	ResourceType string  `json:"resource_type"`
	Cost         float64 `json:"cost"`
	Quantity     int     `json:"quantity"`
	UnitCost     float64 `json:"unit_cost"`
}

// BillingData represents billing information
type BillingData struct {
	Provider     CloudProviderType      `json:"provider"`
	StartTime    time.Time              `json:"start_time"`
	EndTime      time.Time              `json:"end_time"`
	TotalCost    float64                `json:"total_cost"`
	Currency     string                 `json:"currency"`
	Resources    []ResourceBilling      `json:"resources"`
	Discounts    []BillingDiscount      `json:"discounts"`
	Taxes        []BillingTax           `json:"taxes"`
}

// ResourceBilling represents billing for a specific resource
type ResourceBilling struct {
	ResourceID   string  `json:"resource_id"`
	ResourceType string  `json:"resource_type"`
	Usage        float64 `json:"usage"`
	UnitCost     float64 `json:"unit_cost"`
	TotalCost    float64 `json:"total_cost"`
}

// BillingDiscount represents a billing discount
type BillingDiscount struct {
	Type        string  `json:"type"`
	Description string  `json:"description"`
	Amount      float64 `json:"amount"`
}

// BillingTax represents a billing tax
type BillingTax struct {
	Type        string  `json:"type"`
	Description string  `json:"description"`
	Rate        float64 `json:"rate"`
	Amount      float64 `json:"amount"`
}

// ComplianceStatus represents compliance status
type ComplianceStatus struct {
	Provider        CloudProviderType         `json:"provider"`
	OverallScore    float64                   `json:"overall_score"` // 0-100
	Compliances     []ComplianceFramework     `json:"compliances"`
	DataResidency   DataResidencyInfo         `json:"data_residency"`
	Certifications  []string                  `json:"certifications"`
	PolicyViolations []PolicyViolation        `json:"policy_violations"`
	LastAssessment  time.Time                 `json:"last_assessment"`
}

// ComplianceFramework represents a compliance framework
type ComplianceFramework struct {
	Name        string  `json:"name"`
	Version     string  `json:"version"`
	Score       float64 `json:"score"` // 0-100
	Status      string  `json:"status"` // compliant, non-compliant, partial
	Controls    int     `json:"controls"`
	Passed      int     `json:"passed"`
	Failed      int     `json:"failed"`
	NotApplicable int   `json:"not_applicable"`
}

// DataResidencyInfo represents data residency information
type DataResidencyInfo struct {
	PrimaryRegion    string   `json:"primary_region"`
	AllowedRegions   []string `json:"allowed_regions"`
	RestrictedRegions []string `json:"restricted_regions"`
	DataLocation     string   `json:"data_location"`
	CrossBorderTransfer bool  `json:"cross_border_transfer"`
}

// PolicyViolation represents a policy violation
type PolicyViolation struct {
	ID          string    `json:"id"`
	PolicyName  string    `json:"policy_name"`
	Resource    string    `json:"resource"`
	Severity    string    `json:"severity"`
	Description string    `json:"description"`
	DetectedAt  time.Time `json:"detected_at"`
	ResolvedAt  *time.Time `json:"resolved_at,omitempty"`
}

// SecurityGroup represents a security group
type SecurityGroup struct {
	ID          string              `json:"id"`
	Name        string              `json:"name"`
	Description string              `json:"description"`
	Rules       []SecurityGroupRule `json:"rules"`
	Tags        map[string]string   `json:"tags"`
	CreatedAt   time.Time           `json:"created_at"`
}

// SecurityGroupRule represents a security group rule
type SecurityGroupRule struct {
	Direction  string `json:"direction"` // ingress, egress
	Protocol   string `json:"protocol"`  // tcp, udp, icmp, all
	FromPort   int    `json:"from_port"`
	ToPort     int    `json:"to_port"`
	Source     string `json:"source"`     // CIDR or security group ID
	Action     string `json:"action"`     // allow, deny
}

// SecurityGroupRequest represents a request to create a security group
type SecurityGroupRequest struct {
	Name        string              `json:"name"`
	Description string              `json:"description"`
	Rules       []SecurityGroupRule `json:"rules"`
	Tags        map[string]string   `json:"tags"`
}