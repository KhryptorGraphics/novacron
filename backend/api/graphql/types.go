package graphql

import (
	"time"
)

// GraphQL type definitions

// VM represents a virtual machine
type VM struct {
	ID        string    `json:"id"`
	Name      string    `json:"name"`
	State     string    `json:"state"`
	CPU       int       `json:"cpu"`
	Memory    int       `json:"memory"`
	Disk      int       `json:"disk"`
	Image     string    `json:"image"`
	Host      string    `json:"host"`
	IPAddress string    `json:"ipAddress,omitempty"`
	CreatedAt time.Time `json:"createdAt"`
	UpdatedAt time.Time `json:"updatedAt"`
	Metrics   *VMMetrics `json:"metrics,omitempty"`
	Snapshots []*Snapshot `json:"snapshots,omitempty"`
}

// VMMetrics represents VM performance metrics
type VMMetrics struct {
	CPUUsage    float64   `json:"cpuUsage"`
	MemoryUsage float64   `json:"memoryUsage"`
	DiskIO      float64   `json:"diskIO"`
	NetworkIO   float64   `json:"networkIO"`
	Timestamp   time.Time `json:"timestamp"`
}

// Snapshot represents a VM snapshot
type Snapshot struct {
	ID          string    `json:"id"`
	VMID        string    `json:"vmId"`
	Name        string    `json:"name"`
	Description string    `json:"description,omitempty"`
	Size        int       `json:"size"`
	CreatedAt   time.Time `json:"createdAt"`
}

// StorageVolume represents a storage volume
type StorageVolume struct {
	ID            string         `json:"id"`
	Name          string         `json:"name"`
	Size          int            `json:"size"`
	Tier          string         `json:"tier"`
	VMID          string         `json:"vmId,omitempty"`
	CreatedAt     time.Time      `json:"createdAt"`
	UpdatedAt     time.Time      `json:"updatedAt"`
	AccessPattern *AccessPattern `json:"accessPattern,omitempty"`
}

// AccessPattern represents volume access patterns
type AccessPattern struct {
	Temperature   string    `json:"temperature"`
	AccessRate    float64   `json:"accessRate"`
	LastAccessed  time.Time `json:"lastAccessed"`
	PredictedTier string    `json:"predictedTier"`
}

// StorageTierInfo represents storage tier information
type StorageTierInfo struct {
	Tier        string  `json:"tier"`
	Capacity    int     `json:"capacity"`
	Used        int     `json:"used"`
	Available   int     `json:"available"`
	VolumeCount int     `json:"volumeCount"`
	Performance float64 `json:"performance"`
	Cost        float64 `json:"cost"`
}

// Node represents a cluster node
type Node struct {
	ID       string       `json:"id"`
	Address  string       `json:"address"`
	State    string       `json:"state"`
	IsLeader bool         `json:"isLeader"`
	CPU      int          `json:"cpu"`
	Memory   int          `json:"memory"`
	Disk     int          `json:"disk"`
	VMCount  int          `json:"vmCount"`
	Metrics  *NodeMetrics `json:"metrics,omitempty"`
}

// NodeMetrics represents node performance metrics
type NodeMetrics struct {
	CPUUsage       float64   `json:"cpuUsage"`
	MemoryUsage    float64   `json:"memoryUsage"`
	DiskUsage      float64   `json:"diskUsage"`
	NetworkLatency float64   `json:"networkLatency"`
	RequestRate    float64   `json:"requestRate"`
	ErrorRate      float64   `json:"errorRate"`
	Timestamp      time.Time `json:"timestamp"`
}

// ClusterStatus represents cluster health status
type ClusterStatus struct {
	Healthy      bool       `json:"healthy"`
	TotalNodes   int        `json:"totalNodes"`
	HealthyNodes int        `json:"healthyNodes"`
	HasQuorum    bool       `json:"hasQuorum"`
	Leader       *Node      `json:"leader,omitempty"`
	LastElection *time.Time `json:"lastElection,omitempty"`
}

// Migration represents a VM migration
type Migration struct {
	ID               string     `json:"id"`
	VMID             string     `json:"vmId"`
	SourceHost       string     `json:"sourceHost"`
	TargetHost       string     `json:"targetHost"`
	Type             string     `json:"type"`
	Status           string     `json:"status"`
	Progress         float64    `json:"progress"`
	StartedAt        time.Time  `json:"startedAt"`
	CompletedAt      *time.Time `json:"completedAt,omitempty"`
	BytesTransferred int        `json:"bytesTransferred"`
	TotalBytes       int        `json:"totalBytes"`
}

// Alert represents a system alert
type Alert struct {
	ID             string     `json:"id"`
	Severity       string     `json:"severity"`
	Message        string     `json:"message"`
	Source         string     `json:"source"`
	Timestamp      time.Time  `json:"timestamp"`
	Acknowledged   bool       `json:"acknowledged"`
	AcknowledgedBy string     `json:"acknowledgedBy,omitempty"`
	AcknowledgedAt *time.Time `json:"acknowledgedAt,omitempty"`
}

// Event represents a system event
type Event struct {
	ID        string    `json:"id"`
	Type      string    `json:"type"`
	Message   string    `json:"message"`
	Source    string    `json:"source"`
	Timestamp time.Time `json:"timestamp"`
	Details   string    `json:"details,omitempty"`
}

// SystemMetrics represents system-wide metrics
type SystemMetrics struct {
	CPU       *CPUMetrics     `json:"cpu"`
	Memory    *MemoryMetrics  `json:"memory"`
	Disk      *DiskMetrics    `json:"disk"`
	Network   *NetworkMetrics `json:"network"`
	Timestamp time.Time       `json:"timestamp"`
}

// CPUMetrics represents CPU metrics
type CPUMetrics struct {
	Usage   float64   `json:"usage"`
	Cores   int       `json:"cores"`
	LoadAvg []float64 `json:"loadAvg"`
}

// MemoryMetrics represents memory metrics
type MemoryMetrics struct {
	Total     int `json:"total"`
	Used      int `json:"used"`
	Free      int `json:"free"`
	Cached    int `json:"cached"`
	Available int `json:"available"`
}

// DiskMetrics represents disk metrics
type DiskMetrics struct {
	Total int `json:"total"`
	Used  int `json:"used"`
	Free  int `json:"free"`
}

// NetworkMetrics represents network metrics
type NetworkMetrics struct {
	BytesIn    int `json:"bytesIn"`
	BytesOut   int `json:"bytesOut"`
	PacketsIn  int `json:"packetsIn"`
	PacketsOut int `json:"packetsOut"`
}

// Input types

// CreateVMInput represents input for creating a VM
type CreateVMInput struct {
	Name   string `json:"name"`
	CPU    int    `json:"cpu"`
	Memory int    `json:"memory"`
	Disk   int    `json:"disk"`
	Image  string `json:"image"`
}

// UpdateVMInput represents input for updating a VM
type UpdateVMInput struct {
	Name   *string `json:"name,omitempty"`
	CPU    *int    `json:"cpu,omitempty"`
	Memory *int    `json:"memory,omitempty"`
	Disk   *int    `json:"disk,omitempty"`
}

// MigrateVMInput represents input for migrating a VM
type MigrateVMInput struct {
	TargetHost string `json:"targetHost"`
	Live       bool   `json:"live"`
}

// CreateVolumeInput represents input for creating a volume
type CreateVolumeInput struct {
	Name string  `json:"name"`
	Size int     `json:"size"`
	Tier string  `json:"tier"`
	VMID *string `json:"vmId,omitempty"`
}

// CreateSnapshotInput represents input for creating a snapshot
type CreateSnapshotInput struct {
	Name        string  `json:"name"`
	Description *string `json:"description,omitempty"`
}

// TimeRangeInput represents a time range
type TimeRangeInput struct {
	From time.Time `json:"from"`
	To   time.Time `json:"to"`
}

// PaginationInput represents pagination parameters
type PaginationInput struct {
	Page     int `json:"page"`
	PageSize int `json:"pageSize"`
}