package abstraction

import (
	"context"
	"time"
)

// CloudProvider defines the unified interface for all cloud providers
type CloudProvider interface {
	// Provider Information
	GetProviderName() string
	GetProviderType() string
	GetRegion() string

	// VM Operations
	CreateVM(ctx context.Context, spec VMSpec) (*VM, error)
	DeleteVM(ctx context.Context, vmID string) error
	StartVM(ctx context.Context, vmID string) error
	StopVM(ctx context.Context, vmID string) error
	RestartVM(ctx context.Context, vmID string) error
	GetVM(ctx context.Context, vmID string) (*VM, error)
	ListVMs(ctx context.Context, filters map[string]string) ([]*VM, error)
	UpdateVM(ctx context.Context, vmID string, updates VMUpdate) error
	MigrateVM(ctx context.Context, vmID string, targetProvider string) (*MigrationJob, error)
	ResizeVM(ctx context.Context, vmID string, newSize VMSize) error

	// Networking Operations
	CreateVPC(ctx context.Context, spec VPCSpec) (*VPC, error)
	DeleteVPC(ctx context.Context, vpcID string) error
	GetVPC(ctx context.Context, vpcID string) (*VPC, error)
	ListVPCs(ctx context.Context) ([]*VPC, error)

	CreateSubnet(ctx context.Context, spec SubnetSpec) (*Subnet, error)
	DeleteSubnet(ctx context.Context, subnetID string) error
	GetSubnet(ctx context.Context, subnetID string) (*Subnet, error)

	CreateSecurityGroup(ctx context.Context, spec SecurityGroupSpec) (*SecurityGroup, error)
	DeleteSecurityGroup(ctx context.Context, sgID string) error
	UpdateSecurityGroup(ctx context.Context, sgID string, rules []SecurityRule) error

	AllocatePublicIP(ctx context.Context, vmID string) (string, error)
	ReleasePublicIP(ctx context.Context, ipAddress string) error

	// Storage Operations
	CreateVolume(ctx context.Context, spec VolumeSpec) (*Volume, error)
	DeleteVolume(ctx context.Context, volumeID string) error
	AttachVolume(ctx context.Context, volumeID string, vmID string) error
	DetachVolume(ctx context.Context, volumeID string, vmID string) error
	ResizeVolume(ctx context.Context, volumeID string, newSizeGB int) error

	CreateSnapshot(ctx context.Context, volumeID string, description string) (*Snapshot, error)
	DeleteSnapshot(ctx context.Context, snapshotID string) error
	RestoreSnapshot(ctx context.Context, snapshotID string) (*Volume, error)

	// Object Storage Operations
	CreateBucket(ctx context.Context, name string, region string) error
	DeleteBucket(ctx context.Context, name string) error
	UploadObject(ctx context.Context, bucket string, key string, data []byte) error
	DownloadObject(ctx context.Context, bucket string, key string) ([]byte, error)
	DeleteObject(ctx context.Context, bucket string, key string) error

	// Load Balancing Operations
	CreateLoadBalancer(ctx context.Context, spec LoadBalancerSpec) (*LoadBalancer, error)
	DeleteLoadBalancer(ctx context.Context, lbID string) error
	UpdateLoadBalancer(ctx context.Context, lbID string, targets []string) error

	// Cost Operations
	GetCost(ctx context.Context, timeRange TimeRange) (*CostReport, error)
	GetForecast(ctx context.Context, days int) (*CostForecast, error)
	GetResourceCost(ctx context.Context, resourceID string, timeRange TimeRange) (float64, error)

	// Monitoring Operations
	GetMetrics(ctx context.Context, resourceID string, metricName string, timeRange TimeRange) ([]MetricDataPoint, error)
	CreateAlert(ctx context.Context, spec AlertSpec) (*Alert, error)
	DeleteAlert(ctx context.Context, alertID string) error

	// Quota Operations
	GetQuotas(ctx context.Context) (*ResourceQuotas, error)
	GetUsage(ctx context.Context) (*ResourceUsage, error)

	// Health Check
	HealthCheck(ctx context.Context) error

	// Provider-specific features
	GetProviderSpecificFeatures() []string
	ExecuteProviderSpecificOperation(ctx context.Context, operation string, params map[string]interface{}) (interface{}, error)
}

// VMSpec defines the specification for creating a VM
type VMSpec struct {
	Name            string            `json:"name"`
	Size            VMSize            `json:"size"`
	Image           string            `json:"image"`
	VolumeSize      int               `json:"volume_size_gb"`
	VolumeType      string            `json:"volume_type"`
	NetworkID       string            `json:"network_id"`
	SubnetID        string            `json:"subnet_id"`
	SecurityGroups  []string          `json:"security_groups"`
	PublicIP        bool              `json:"public_ip"`
	SSHKeys         []string          `json:"ssh_keys"`
	UserData        string            `json:"user_data"`
	Tags            map[string]string `json:"tags"`
	AvailabilityZone string           `json:"availability_zone"`
	SpotInstance    bool              `json:"spot_instance"`
	MaxSpotPrice    float64           `json:"max_spot_price"`
}

// VMSize defines the size/type of a VM
type VMSize struct {
	CPUs     int    `json:"cpus"`
	MemoryGB int    `json:"memory_gb"`
	Type     string `json:"type"` // Provider-specific type
}

// VM represents a virtual machine
type VM struct {
	ID               string            `json:"id"`
	Name             string            `json:"name"`
	Provider         string            `json:"provider"`
	Region           string            `json:"region"`
	State            string            `json:"state"`
	Size             VMSize            `json:"size"`
	PublicIP         string            `json:"public_ip"`
	PrivateIP        string            `json:"private_ip"`
	NetworkID        string            `json:"network_id"`
	SubnetID         string            `json:"subnet_id"`
	SecurityGroups   []string          `json:"security_groups"`
	Volumes          []string          `json:"volumes"`
	Tags             map[string]string `json:"tags"`
	CreatedAt        time.Time         `json:"created_at"`
	LaunchedAt       time.Time         `json:"launched_at"`
	SpotInstance     bool              `json:"spot_instance"`
	AvailabilityZone string            `json:"availability_zone"`
}

// VMUpdate defines updates to apply to a VM
type VMUpdate struct {
	Name           *string            `json:"name,omitempty"`
	SecurityGroups *[]string          `json:"security_groups,omitempty"`
	Tags           *map[string]string `json:"tags,omitempty"`
}

// VPCSpec defines the specification for creating a VPC
type VPCSpec struct {
	Name      string            `json:"name"`
	CIDR      string            `json:"cidr"`
	Region    string            `json:"region"`
	EnableDNS bool              `json:"enable_dns"`
	Tags      map[string]string `json:"tags"`
}

// VPC represents a virtual private cloud
type VPC struct {
	ID        string            `json:"id"`
	Name      string            `json:"name"`
	CIDR      string            `json:"cidr"`
	Region    string            `json:"region"`
	Provider  string            `json:"provider"`
	State     string            `json:"state"`
	Subnets   []string          `json:"subnets"`
	Tags      map[string]string `json:"tags"`
	CreatedAt time.Time         `json:"created_at"`
}

// SubnetSpec defines the specification for creating a subnet
type SubnetSpec struct {
	VpcID            string            `json:"vpc_id"`
	Name             string            `json:"name"`
	CIDR             string            `json:"cidr"`
	AvailabilityZone string            `json:"availability_zone"`
	Public           bool              `json:"public"`
	Tags             map[string]string `json:"tags"`
}

// Subnet represents a subnet within a VPC
type Subnet struct {
	ID               string            `json:"id"`
	VpcID            string            `json:"vpc_id"`
	Name             string            `json:"name"`
	CIDR             string            `json:"cidr"`
	AvailabilityZone string            `json:"availability_zone"`
	Public           bool              `json:"public"`
	AvailableIPs     int               `json:"available_ips"`
	Tags             map[string]string `json:"tags"`
	CreatedAt        time.Time         `json:"created_at"`
}

// SecurityGroupSpec defines the specification for a security group
type SecurityGroupSpec struct {
	VpcID       string            `json:"vpc_id"`
	Name        string            `json:"name"`
	Description string            `json:"description"`
	Rules       []SecurityRule    `json:"rules"`
	Tags        map[string]string `json:"tags"`
}

// SecurityGroup represents a security group
type SecurityGroup struct {
	ID          string            `json:"id"`
	VpcID       string            `json:"vpc_id"`
	Name        string            `json:"name"`
	Description string            `json:"description"`
	Rules       []SecurityRule    `json:"rules"`
	Tags        map[string]string `json:"tags"`
	CreatedAt   time.Time         `json:"created_at"`
}

// SecurityRule defines a firewall rule
type SecurityRule struct {
	Direction  string `json:"direction"` // "ingress" or "egress"
	Protocol   string `json:"protocol"`  // "tcp", "udp", "icmp", "all"
	FromPort   int    `json:"from_port"`
	ToPort     int    `json:"to_port"`
	Source     string `json:"source"`      // CIDR or security group ID
	Destination string `json:"destination"` // For egress rules
	Description string `json:"description"`
}

// VolumeSpec defines the specification for creating a volume
type VolumeSpec struct {
	Name             string            `json:"name"`
	SizeGB           int               `json:"size_gb"`
	VolumeType       string            `json:"volume_type"` // Provider-specific
	IOPS             int               `json:"iops"`
	Throughput       int               `json:"throughput"`
	AvailabilityZone string            `json:"availability_zone"`
	Encrypted        bool              `json:"encrypted"`
	SnapshotID       string            `json:"snapshot_id,omitempty"`
	Tags             map[string]string `json:"tags"`
}

// Volume represents a storage volume
type Volume struct {
	ID               string            `json:"id"`
	Name             string            `json:"name"`
	SizeGB           int               `json:"size_gb"`
	VolumeType       string            `json:"volume_type"`
	State            string            `json:"state"`
	Attached         bool              `json:"attached"`
	AttachedTo       string            `json:"attached_to"`
	AvailabilityZone string            `json:"availability_zone"`
	Encrypted        bool              `json:"encrypted"`
	IOPS             int               `json:"iops"`
	Tags             map[string]string `json:"tags"`
	CreatedAt        time.Time         `json:"created_at"`
}

// Snapshot represents a volume snapshot
type Snapshot struct {
	ID          string            `json:"id"`
	VolumeID    string            `json:"volume_id"`
	Description string            `json:"description"`
	SizeGB      int               `json:"size_gb"`
	State       string            `json:"state"`
	Progress    int               `json:"progress"`
	Tags        map[string]string `json:"tags"`
	CreatedAt   time.Time         `json:"created_at"`
}

// LoadBalancerSpec defines the specification for a load balancer
type LoadBalancerSpec struct {
	Name           string            `json:"name"`
	Type           string            `json:"type"` // "application", "network"
	Scheme         string            `json:"scheme"` // "internet-facing", "internal"
	SubnetIDs      []string          `json:"subnet_ids"`
	SecurityGroups []string          `json:"security_groups"`
	Listeners      []LBListener      `json:"listeners"`
	HealthCheck    HealthCheckConfig `json:"health_check"`
	Tags           map[string]string `json:"tags"`
}

// LoadBalancer represents a load balancer
type LoadBalancer struct {
	ID             string            `json:"id"`
	Name           string            `json:"name"`
	Type           string            `json:"type"`
	DNSName        string            `json:"dns_name"`
	State          string            `json:"state"`
	SubnetIDs      []string          `json:"subnet_ids"`
	SecurityGroups []string          `json:"security_groups"`
	Targets        []string          `json:"targets"`
	Tags           map[string]string `json:"tags"`
	CreatedAt      time.Time         `json:"created_at"`
}

// LBListener defines a load balancer listener
type LBListener struct {
	Protocol string `json:"protocol"`
	Port     int    `json:"port"`
	TargetProtocol string `json:"target_protocol"`
	TargetPort     int    `json:"target_port"`
}

// HealthCheckConfig defines health check configuration
type HealthCheckConfig struct {
	Protocol           string        `json:"protocol"`
	Port               int           `json:"port"`
	Path               string        `json:"path"`
	Interval           time.Duration `json:"interval"`
	Timeout            time.Duration `json:"timeout"`
	HealthyThreshold   int           `json:"healthy_threshold"`
	UnhealthyThreshold int           `json:"unhealthy_threshold"`
}

// TimeRange defines a time range for queries
type TimeRange struct {
	Start time.Time `json:"start"`
	End   time.Time `json:"end"`
}

// CostReport represents a cost report
type CostReport struct {
	Provider     string             `json:"provider"`
	TotalCost    float64            `json:"total_cost"`
	Currency     string             `json:"currency"`
	TimeRange    TimeRange          `json:"time_range"`
	ByService    map[string]float64 `json:"by_service"`
	ByResource   map[string]float64 `json:"by_resource"`
	ByTag        map[string]float64 `json:"by_tag"`
	Recommendations []CostRecommendation `json:"recommendations"`
}

// CostForecast represents a cost forecast
type CostForecast struct {
	Provider      string             `json:"provider"`
	ForecastedCost float64           `json:"forecasted_cost"`
	Currency      string             `json:"currency"`
	Period        string             `json:"period"`
	Confidence    float64            `json:"confidence"`
	Breakdown     map[string]float64 `json:"breakdown"`
}

// CostRecommendation represents a cost optimization recommendation
type CostRecommendation struct {
	Type             string  `json:"type"`
	ResourceID       string  `json:"resource_id"`
	CurrentCost      float64 `json:"current_cost"`
	PotentialSavings float64 `json:"potential_savings"`
	Description      string  `json:"description"`
	Action           string  `json:"action"`
}

// MetricDataPoint represents a single metric data point
type MetricDataPoint struct {
	Timestamp time.Time `json:"timestamp"`
	Value     float64   `json:"value"`
	Unit      string    `json:"unit"`
}

// AlertSpec defines the specification for an alert
type AlertSpec struct {
	Name        string            `json:"name"`
	ResourceID  string            `json:"resource_id"`
	MetricName  string            `json:"metric_name"`
	Threshold   float64           `json:"threshold"`
	Comparison  string            `json:"comparison"` // "gt", "lt", "eq"
	Duration    time.Duration     `json:"duration"`
	Actions     []string          `json:"actions"`
	Tags        map[string]string `json:"tags"`
}

// Alert represents a monitoring alert
type Alert struct {
	ID         string            `json:"id"`
	Name       string            `json:"name"`
	ResourceID string            `json:"resource_id"`
	MetricName string            `json:"metric_name"`
	State      string            `json:"state"`
	Threshold  float64           `json:"threshold"`
	Tags       map[string]string `json:"tags"`
	CreatedAt  time.Time         `json:"created_at"`
	UpdatedAt  time.Time         `json:"updated_at"`
}

// ResourceQuotas represents resource quotas for a provider
type ResourceQuotas struct {
	MaxVMs           int `json:"max_vms"`
	MaxCPUs          int `json:"max_cpus"`
	MaxMemoryGB      int `json:"max_memory_gb"`
	MaxStorageGB     int `json:"max_storage_gb"`
	MaxNetworks      int `json:"max_networks"`
	MaxLoadBalancers int `json:"max_load_balancers"`
	MaxSnapshots     int `json:"max_snapshots"`
	MaxPublicIPs     int `json:"max_public_ips"`
}

// ResourceUsage represents current resource usage
type ResourceUsage struct {
	VMs           int `json:"vms"`
	CPUs          int `json:"cpus"`
	MemoryGB      int `json:"memory_gb"`
	StorageGB     int `json:"storage_gb"`
	Networks      int `json:"networks"`
	LoadBalancers int `json:"load_balancers"`
	Snapshots     int `json:"snapshots"`
	PublicIPs     int `json:"public_ips"`
}

// MigrationJob represents a VM migration job
type MigrationJob struct {
	ID             string        `json:"id"`
	VMID           string        `json:"vm_id"`
	SourceProvider string        `json:"source_provider"`
	TargetProvider string        `json:"target_provider"`
	State          string        `json:"state"`
	Progress       int           `json:"progress"`
	StartedAt      time.Time     `json:"started_at"`
	CompletedAt    *time.Time    `json:"completed_at,omitempty"`
	Error          string        `json:"error,omitempty"`
	EstimatedTime  time.Duration `json:"estimated_time"`
}
