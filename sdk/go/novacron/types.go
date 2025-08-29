package novacron

import (
	"time"
)

// VM states
const (
	VMStateUnknown      = "unknown"
	VMStateCreated      = "created"
	VMStateCreating     = "creating"
	VMStateProvisioning = "provisioning"
	VMStateRunning      = "running"
	VMStateStopped      = "stopped"
	VMStatePaused       = "paused"
	VMStatePausing      = "pausing"
	VMStateResuming     = "resuming"
	VMStateRestarting   = "restarting"
	VMStateDeleting     = "deleting"
	VMStateMigrating    = "migrating"
	VMStateFailed       = "failed"
)

// Migration types
const (
	MigrationTypeCold = "cold"
	MigrationTypeWarm = "warm"
	MigrationTypeLive = "live"
)

// Migration statuses
const (
	MigrationStatusPending   = "pending"
	MigrationStatusRunning   = "running"
	MigrationStatusCompleted = "completed"
	MigrationStatusFailed    = "failed"
	MigrationStatusCancelled = "cancelled"
)

// VM represents a virtual machine
type VM struct {
	ID        string            `json:"id"`
	Name      string            `json:"name"`
	State     string            `json:"state"`
	NodeID    *string           `json:"node_id,omitempty"`
	OwnerID   *int              `json:"owner_id,omitempty"`
	TenantID  string            `json:"tenant_id"`
	Config    map[string]interface{} `json:"config,omitempty"`
	CreatedAt time.Time         `json:"created_at"`
	UpdatedAt time.Time         `json:"updated_at"`
}

// VMMetrics represents VM performance metrics
type VMMetrics struct {
	VMID         string    `json:"vm_id"`
	CPUUsage     float64   `json:"cpu_usage"`
	MemoryUsage  float64   `json:"memory_usage"`
	DiskUsage    float64   `json:"disk_usage,omitempty"`
	NetworkSent  int64     `json:"network_sent"`
	NetworkRecv  int64     `json:"network_recv"`
	IOPS         int       `json:"iops,omitempty"`
	LastUpdated  time.Time `json:"last_updated"`
}

// VMTemplate represents a VM template
type VMTemplate struct {
	ID          string                 `json:"id,omitempty"`
	Name        string                 `json:"name"`
	Description string                 `json:"description,omitempty"`
	Config      map[string]interface{} `json:"config"`
	Parameters  []TemplateParameter    `json:"parameters,omitempty"`
	CreatedAt   time.Time              `json:"created_at"`
	UpdatedAt   time.Time              `json:"updated_at"`
}

// TemplateParameter represents a template parameter
type TemplateParameter struct {
	Name         string      `json:"name"`
	Type         string      `json:"type"`
	Description  string      `json:"description,omitempty"`
	Required     bool        `json:"required,omitempty"`
	DefaultValue interface{} `json:"default_value,omitempty"`
}

// Migration represents a VM migration
type Migration struct {
	ID               string     `json:"id"`
	VMID             string     `json:"vm_id"`
	SourceNodeID     string     `json:"source_node_id"`
	TargetNodeID     string     `json:"target_node_id"`
	Type             string     `json:"type"`
	Status           string     `json:"status"`
	Progress         float64    `json:"progress"`
	BytesTotal       int64      `json:"bytes_total"`
	BytesTransferred int64      `json:"bytes_transferred"`
	StartedAt        *time.Time `json:"started_at,omitempty"`
	CompletedAt      *time.Time `json:"completed_at,omitempty"`
	ErrorMessage     *string    `json:"error_message,omitempty"`
	CreatedAt        time.Time  `json:"created_at"`
	UpdatedAt        time.Time  `json:"updated_at"`
}

// Node represents a cluster node
type Node struct {
	ID           string                 `json:"id"`
	Name         string                 `json:"name"`
	Address      string                 `json:"address"`
	Status       string                 `json:"status"`
	Capabilities map[string]interface{} `json:"capabilities,omitempty"`
	Resources    map[string]interface{} `json:"resources,omitempty"`
	Labels       map[string]string      `json:"labels,omitempty"`
	LastSeen     time.Time              `json:"last_seen"`
	CreatedAt    time.Time              `json:"created_at"`
	UpdatedAt    time.Time              `json:"updated_at"`
}

// NodeMetrics represents node performance metrics
type NodeMetrics struct {
	NodeID          string  `json:"node_id"`
	CPUUsage        float64 `json:"cpu_usage"`
	MemoryUsage     float64 `json:"memory_usage"`
	MemoryTotal     int64   `json:"memory_total"`
	MemoryAvailable int64   `json:"memory_available"`
	DiskUsage       float64 `json:"disk_usage"`
	DiskTotal       int64   `json:"disk_total"`
	DiskAvailable   int64   `json:"disk_available"`
	NetworkSent     int64   `json:"network_sent"`
	NetworkRecv     int64   `json:"network_recv"`
	LoadAverage1    float64 `json:"load_average_1"`
	LoadAverage5    float64 `json:"load_average_5"`
	LoadAverage15   float64 `json:"load_average_15"`
	Timestamp       time.Time `json:"timestamp"`
}

// SystemMetrics represents system-wide metrics
type SystemMetrics struct {
	TotalVMs       int     `json:"total_vms"`
	RunningVMs     int     `json:"running_vms"`
	TotalNodes     int     `json:"total_nodes"`
	OnlineNodes    int     `json:"online_nodes"`
	CPUUsage       float64 `json:"cpu_usage"`
	MemoryUsage    float64 `json:"memory_usage"`
	StorageUsage   float64 `json:"storage_usage"`
	NetworkSent    int64   `json:"network_sent"`
	NetworkRecv    int64   `json:"network_recv"`
	Timestamp      time.Time `json:"timestamp"`
}

// Request Types

// CreateVMRequest represents a request to create a VM
type CreateVMRequest struct {
	Name       string            `json:"name"`
	Command    string            `json:"command,omitempty"`
	Args       []string          `json:"args,omitempty"`
	CPUShares  int               `json:"cpu_shares"`
	MemoryMB   int               `json:"memory_mb"`
	DiskSizeGB int               `json:"disk_size_gb"`
	Tags       map[string]string `json:"tags,omitempty"`
	TenantID   string            `json:"tenant_id,omitempty"`
}

// UpdateVMRequest represents a request to update a VM
type UpdateVMRequest struct {
	Name       *string            `json:"name,omitempty"`
	CPUShares  *int               `json:"cpu_shares,omitempty"`
	MemoryMB   *int               `json:"memory_mb,omitempty"`
	DiskSizeGB *int               `json:"disk_size_gb,omitempty"`
	Tags       map[string]string  `json:"tags,omitempty"`
}

// MigrationRequest represents a request to migrate a VM
type MigrationRequest struct {
	TargetNodeID     string `json:"target_node_id"`
	Type             string `json:"type,omitempty"`
	Force            bool   `json:"force,omitempty"`
	BandwidthLimit   *int   `json:"bandwidth_limit,omitempty"`
	Compression      bool   `json:"compression,omitempty"`
}

// Options Types

// ListVMOptions represents options for listing VMs
type ListVMOptions struct {
	TenantID string
	State    string
	NodeID   string
}

// ListMigrationOptions represents options for listing migrations
type ListMigrationOptions struct {
	VMID   string
	Status string
}

// MetricsOptions represents options for retrieving metrics
type MetricsOptions struct {
	StartTime time.Time
	EndTime   time.Time
}

// SystemMetricsOptions represents options for retrieving system metrics
type SystemMetricsOptions struct {
	NodeID    string
	StartTime time.Time
	EndTime   time.Time
}

// Response Types

// HealthStatus represents the health status response
type HealthStatus struct {
	Status     string            `json:"status"`
	Version    string            `json:"version"`
	Uptime     string            `json:"uptime"`
	Components map[string]string `json:"components"`
}

// Version represents the version information response
type Version struct {
	Version   string `json:"version"`
	GitCommit string `json:"git_commit"`
	BuildTime string `json:"build_time"`
	GoVersion string `json:"go_version"`
}