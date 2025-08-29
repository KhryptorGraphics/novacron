// Package interfaces defines the core adapter interfaces for multi-cloud VM management
package interfaces

import (
	"context"
	"time"
)

// CloudAdapter defines the interface that all cloud adapters must implement
type CloudAdapter interface {
	// Metadata
	Name() string
	Version() string
	SupportedRegions() []string
	SupportedInstanceTypes() []string

	// Authentication and Configuration
	Configure(config CloudConfig) error
	ValidateCredentials(ctx context.Context) error

	// Instance Lifecycle Management
	CreateInstance(ctx context.Context, req *CreateInstanceRequest) (*Instance, error)
	GetInstance(ctx context.Context, instanceID string) (*Instance, error)
	ListInstances(ctx context.Context, filters *ListInstanceFilters) ([]*Instance, error)
	UpdateInstance(ctx context.Context, instanceID string, updates *UpdateInstanceRequest) (*Instance, error)
	DeleteInstance(ctx context.Context, instanceID string, force bool) error

	// Instance Operations
	StartInstance(ctx context.Context, instanceID string) error
	StopInstance(ctx context.Context, instanceID string, force bool) error
	RebootInstance(ctx context.Context, instanceID string) error

	// Monitoring and Metrics
	GetInstanceMetrics(ctx context.Context, instanceID string, opts *MetricsOptions) (*InstanceMetrics, error)
	GetInstanceLogs(ctx context.Context, instanceID string, opts *LogOptions) (*InstanceLogs, error)

	// Storage Management
	CreateVolume(ctx context.Context, req *CreateVolumeRequest) (*Volume, error)
	AttachVolume(ctx context.Context, volumeID, instanceID string) error
	DetachVolume(ctx context.Context, volumeID string) error
	DeleteVolume(ctx context.Context, volumeID string) error

	// Networking
	ListNetworks(ctx context.Context) ([]*Network, error)
	CreateSecurityGroup(ctx context.Context, req *CreateSecurityGroupRequest) (*SecurityGroup, error)
	DeleteSecurityGroup(ctx context.Context, groupID string) error

	// Resource Management
	GetQuotas(ctx context.Context) (*ResourceQuotas, error)
	GetCosts(ctx context.Context, opts *CostOptions) (*CostData, error)

	// Migration Support
	ExportInstance(ctx context.Context, instanceID string, opts *ExportOptions) (*ExportResult, error)
	ImportInstance(ctx context.Context, req *ImportInstanceRequest) (*Instance, error)

	// Health and Status
	HealthCheck(ctx context.Context) error
	GetStatus(ctx context.Context) (*AdapterStatus, error)
}

// CloudConfig represents cloud-specific configuration
type CloudConfig interface {
	GetProvider() string
	GetRegion() string
	GetCredentials() map[string]string
	GetSettings() map[string]interface{}
	Validate() error
}

// Instance represents a virtual machine instance
type Instance struct {
	ID               string                 `json:"id"`
	Name             string                 `json:"name"`
	Provider         string                 `json:"provider"`
	Region           string                 `json:"region"`
	Zone             string                 `json:"zone,omitempty"`
	InstanceType     string                 `json:"instance_type"`
	State            InstanceState          `json:"state"`
	PublicIP         string                 `json:"public_ip,omitempty"`
	PrivateIP        string                 `json:"private_ip,omitempty"`
	ImageID          string                 `json:"image_id"`
	KeyPairName      string                 `json:"key_pair_name,omitempty"`
	SecurityGroups   []string               `json:"security_groups,omitempty"`
	SubnetID         string                 `json:"subnet_id,omitempty"`
	VolumeIDs        []string               `json:"volume_ids,omitempty"`
	Tags             map[string]string      `json:"tags,omitempty"`
	Metadata         map[string]interface{} `json:"metadata,omitempty"`
	CreatedAt        time.Time              `json:"created_at"`
	LaunchedAt       *time.Time             `json:"launched_at,omitempty"`
	TerminatedAt     *time.Time             `json:"terminated_at,omitempty"`
	CostPerHour      float64                `json:"cost_per_hour,omitempty"`
	Architecture     string                 `json:"architecture,omitempty"`
	Platform         string                 `json:"platform,omitempty"`
	Hypervisor       string                 `json:"hypervisor,omitempty"`
	VirtualizationType string               `json:"virtualization_type,omitempty"`
}

// InstanceState represents the state of an instance
type InstanceState string

const (
	InstanceStatePending     InstanceState = "pending"
	InstanceStateRunning     InstanceState = "running"
	InstanceStateStopping    InstanceState = "stopping"
	InstanceStateStopped     InstanceState = "stopped"
	InstanceStateTerminating InstanceState = "terminating"
	InstanceStateTerminated  InstanceState = "terminated"
	InstanceStateRebooting   InstanceState = "rebooting"
	InstanceStateUnknown     InstanceState = "unknown"
)

// Volume represents a storage volume
type Volume struct {
	ID           string            `json:"id"`
	Name         string            `json:"name,omitempty"`
	SizeGB       int               `json:"size_gb"`
	VolumeType   string            `json:"volume_type"`
	State        VolumeState       `json:"state"`
	InstanceID   string            `json:"instance_id,omitempty"`
	Device       string            `json:"device,omitempty"`
	Encrypted    bool              `json:"encrypted"`
	IOPS         int               `json:"iops,omitempty"`
	Throughput   int               `json:"throughput,omitempty"`
	Tags         map[string]string `json:"tags,omitempty"`
	CreatedAt    time.Time         `json:"created_at"`
	CostPerMonth float64           `json:"cost_per_month,omitempty"`
}

// VolumeState represents the state of a volume
type VolumeState string

const (
	VolumeStateCreating  VolumeState = "creating"
	VolumeStateAvailable VolumeState = "available"
	VolumeStateInUse     VolumeState = "in-use"
	VolumeStateDeleting  VolumeState = "deleting"
	VolumeStateDeleted   VolumeState = "deleted"
	VolumeStateError     VolumeState = "error"
)

// Network represents a virtual network
type Network struct {
	ID        string            `json:"id"`
	Name      string            `json:"name"`
	CIDR      string            `json:"cidr"`
	Region    string            `json:"region"`
	Subnets   []*Subnet         `json:"subnets,omitempty"`
	Tags      map[string]string `json:"tags,omitempty"`
	CreatedAt time.Time         `json:"created_at"`
}

// Subnet represents a network subnet
type Subnet struct {
	ID                string            `json:"id"`
	Name              string            `json:"name"`
	CIDR              string            `json:"cidr"`
	Zone              string            `json:"zone"`
	NetworkID         string            `json:"network_id"`
	RouteTableID      string            `json:"route_table_id,omitempty"`
	IsPublic          bool              `json:"is_public"`
	AvailableIPs      int               `json:"available_ips"`
	Tags              map[string]string `json:"tags,omitempty"`
}

// SecurityGroup represents a security group/firewall rule set
type SecurityGroup struct {
	ID          string                 `json:"id"`
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	NetworkID   string                 `json:"network_id,omitempty"`
	Rules       []*SecurityGroupRule   `json:"rules"`
	Tags        map[string]string      `json:"tags,omitempty"`
}

// SecurityGroupRule represents a single security group rule
type SecurityGroupRule struct {
	ID          string `json:"id"`
	Direction   string `json:"direction"` // ingress/egress
	Protocol    string `json:"protocol"`  // tcp/udp/icmp/all
	FromPort    int    `json:"from_port,omitempty"`
	ToPort      int    `json:"to_port,omitempty"`
	CIDR        string `json:"cidr,omitempty"`
	SourceSGID  string `json:"source_sg_id,omitempty"`
	Description string `json:"description,omitempty"`
}

// Request and Response types

// CreateInstanceRequest represents a request to create an instance
type CreateInstanceRequest struct {
	Name             string            `json:"name"`
	ImageID          string            `json:"image_id"`
	InstanceType     string            `json:"instance_type"`
	KeyPairName      string            `json:"key_pair_name,omitempty"`
	SecurityGroupIDs []string          `json:"security_group_ids,omitempty"`
	SubnetID         string            `json:"subnet_id,omitempty"`
	UserData         string            `json:"user_data,omitempty"`
	Tags             map[string]string `json:"tags,omitempty"`
	MinCount         int               `json:"min_count,omitempty"`
	MaxCount         int               `json:"max_count,omitempty"`
	
	// Storage configuration
	RootVolumeSize int    `json:"root_volume_size,omitempty"`
	RootVolumeType string `json:"root_volume_type,omitempty"`
	
	// Advanced options
	IamInstanceProfile   string `json:"iam_instance_profile,omitempty"`
	PlacementGroup       string `json:"placement_group,omitempty"`
	Tenancy              string `json:"tenancy,omitempty"`
	MonitoringEnabled    bool   `json:"monitoring_enabled"`
	EbsOptimized         bool   `json:"ebs_optimized"`
	SriovNetSupport      bool   `json:"sriov_net_support"`
	InstanceInitiatedShutdownBehavior string `json:"instance_initiated_shutdown_behavior,omitempty"`
}

// UpdateInstanceRequest represents a request to update an instance
type UpdateInstanceRequest struct {
	Name         *string           `json:"name,omitempty"`
	InstanceType *string           `json:"instance_type,omitempty"`
	Tags         map[string]string `json:"tags,omitempty"`
	UserData     *string           `json:"user_data,omitempty"`
}

// ListInstanceFilters represents filters for listing instances
type ListInstanceFilters struct {
	States       []InstanceState   `json:"states,omitempty"`
	InstanceIDs  []string          `json:"instance_ids,omitempty"`
	Tags         map[string]string `json:"tags,omitempty"`
	Zone         string            `json:"zone,omitempty"`
	InstanceType string            `json:"instance_type,omitempty"`
	ImageID      string            `json:"image_id,omitempty"`
}

// CreateVolumeRequest represents a request to create a volume
type CreateVolumeRequest struct {
	Name       string            `json:"name,omitempty"`
	SizeGB     int               `json:"size_gb"`
	VolumeType string            `json:"volume_type"`
	Zone       string            `json:"zone"`
	Encrypted  bool              `json:"encrypted"`
	IOPS       int               `json:"iops,omitempty"`
	Throughput int               `json:"throughput,omitempty"`
	Tags       map[string]string `json:"tags,omitempty"`
	SnapshotID string            `json:"snapshot_id,omitempty"`
}

// CreateSecurityGroupRequest represents a request to create a security group
type CreateSecurityGroupRequest struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	NetworkID   string                 `json:"network_id,omitempty"`
	Rules       []*SecurityGroupRule   `json:"rules,omitempty"`
	Tags        map[string]string      `json:"tags,omitempty"`
}

// Metrics and Monitoring

// InstanceMetrics represents instance performance metrics
type InstanceMetrics struct {
	InstanceID     string                       `json:"instance_id"`
	Period         time.Duration                `json:"period"`
	StartTime      time.Time                    `json:"start_time"`
	EndTime        time.Time                    `json:"end_time"`
	CPUUtilization []*MetricDataPoint           `json:"cpu_utilization"`
	MemoryUsage    []*MetricDataPoint           `json:"memory_usage"`
	DiskReadOps    []*MetricDataPoint           `json:"disk_read_ops"`
	DiskWriteOps   []*MetricDataPoint           `json:"disk_write_ops"`
	NetworkIn      []*MetricDataPoint           `json:"network_in"`
	NetworkOut     []*MetricDataPoint           `json:"network_out"`
	CustomMetrics  map[string][]*MetricDataPoint `json:"custom_metrics,omitempty"`
}

// MetricDataPoint represents a single metric data point
type MetricDataPoint struct {
	Timestamp time.Time `json:"timestamp"`
	Value     float64   `json:"value"`
	Unit      string    `json:"unit"`
}

// MetricsOptions represents options for retrieving metrics
type MetricsOptions struct {
	StartTime time.Time     `json:"start_time"`
	EndTime   time.Time     `json:"end_time"`
	Period    time.Duration `json:"period"`
	Statistics []string     `json:"statistics,omitempty"` // Average, Sum, Maximum, etc.
}

// LogOptions represents options for retrieving logs
type LogOptions struct {
	StartTime time.Time `json:"start_time"`
	EndTime   time.Time `json:"end_time"`
	Filter    string    `json:"filter,omitempty"`
	MaxLines  int       `json:"max_lines,omitempty"`
}

// InstanceLogs represents instance logs
type InstanceLogs struct {
	InstanceID string       `json:"instance_id"`
	LogGroups  []*LogGroup  `json:"log_groups"`
}

// LogGroup represents a group of log streams
type LogGroup struct {
	Name    string       `json:"name"`
	Streams []*LogStream `json:"streams"`
}

// LogStream represents a log stream
type LogStream struct {
	Name    string      `json:"name"`
	Entries []*LogEntry `json:"entries"`
}

// LogEntry represents a single log entry
type LogEntry struct {
	Timestamp time.Time `json:"timestamp"`
	Message   string    `json:"message"`
	Level     string    `json:"level,omitempty"`
	Source    string    `json:"source,omitempty"`
}

// Resource Management

// ResourceQuotas represents resource quotas and limits
type ResourceQuotas struct {
	Provider  string                    `json:"provider"`
	Region    string                    `json:"region"`
	Quotas    map[string]*ResourceQuota `json:"quotas"`
	UpdatedAt time.Time                 `json:"updated_at"`
}

// ResourceQuota represents a single resource quota
type ResourceQuota struct {
	Name      string  `json:"name"`
	Used      int64   `json:"used"`
	Limit     int64   `json:"limit"`
	Available int64   `json:"available"`
	Unit      string  `json:"unit"`
	Percentage float64 `json:"percentage"`
}

// CostOptions represents options for cost retrieval
type CostOptions struct {
	StartTime    time.Time `json:"start_time"`
	EndTime      time.Time `json:"end_time"`
	Granularity  string    `json:"granularity"` // DAILY, MONTHLY, HOURLY
	GroupBy      []string  `json:"group_by,omitempty"`
	InstanceIDs  []string  `json:"instance_ids,omitempty"`
}

// CostData represents cost information
type CostData struct {
	TotalCost    float64                      `json:"total_cost"`
	Currency     string                       `json:"currency"`
	StartTime    time.Time                    `json:"start_time"`
	EndTime      time.Time                    `json:"end_time"`
	Granularity  string                       `json:"granularity"`
	CostBreakdown map[string]*CostBreakdown   `json:"cost_breakdown"`
	TimeSeries   []*CostDataPoint             `json:"time_series"`
}

// CostBreakdown represents cost breakdown by category
type CostBreakdown struct {
	Service     string  `json:"service"`
	Cost        float64 `json:"cost"`
	Usage       float64 `json:"usage"`
	Unit        string  `json:"unit"`
	Percentage  float64 `json:"percentage"`
}

// CostDataPoint represents a cost data point in time
type CostDataPoint struct {
	Timestamp time.Time `json:"timestamp"`
	Cost      float64   `json:"cost"`
	Usage     float64   `json:"usage,omitempty"`
}

// Migration Support

// ExportOptions represents options for exporting instances
type ExportOptions struct {
	Format          string            `json:"format"` // OVA, VMDK, VHD, etc.
	S3Bucket        string            `json:"s3_bucket,omitempty"`
	IncludeMetadata bool              `json:"include_metadata"`
	Compression     string            `json:"compression,omitempty"`
	Tags            map[string]string `json:"tags,omitempty"`
}

// ExportResult represents the result of an export operation
type ExportResult struct {
	ExportID     string            `json:"export_id"`
	Status       string            `json:"status"`
	Progress     float64           `json:"progress"`
	DownloadURL  string            `json:"download_url,omitempty"`
	Size         int64             `json:"size,omitempty"`
	Format       string            `json:"format"`
	CreatedAt    time.Time         `json:"created_at"`
	CompletedAt  *time.Time        `json:"completed_at,omitempty"`
	ErrorMessage string            `json:"error_message,omitempty"`
	Tags         map[string]string `json:"tags,omitempty"`
}

// ImportInstanceRequest represents a request to import an instance
type ImportInstanceRequest struct {
	Name          string            `json:"name"`
	ImageURL      string            `json:"image_url"`
	Format        string            `json:"format"`
	InstanceType  string            `json:"instance_type"`
	SubnetID      string            `json:"subnet_id,omitempty"`
	Tags          map[string]string `json:"tags,omitempty"`
	Description   string            `json:"description,omitempty"`
}

// AdapterStatus represents the status of a cloud adapter
type AdapterStatus struct {
	Name           string                 `json:"name"`
	Version        string                 `json:"version"`
	Provider       string                 `json:"provider"`
	Region         string                 `json:"region"`
	Status         string                 `json:"status"`
	LastHealthCheck time.Time             `json:"last_health_check"`
	Capabilities   []string               `json:"capabilities"`
	Limits         map[string]interface{} `json:"limits,omitempty"`
	Configuration  map[string]interface{} `json:"configuration,omitempty"`
	ErrorMessage   string                 `json:"error_message,omitempty"`
}

// AdapterFactory creates cloud adapters
type AdapterFactory interface {
	CreateAdapter(provider string, config CloudConfig) (CloudAdapter, error)
	SupportedProviders() []string
}