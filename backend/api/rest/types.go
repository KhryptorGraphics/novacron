package rest

import (
	"time"
	"github.com/khryptorgraphics/novacron/backend/core/vm"
)

// Request types

// CreateVMRequest represents a request to create a VM
type CreateVMRequest struct {
	Name   string `json:"name"`
	CPU    int    `json:"cpu"`
	Memory int64  `json:"memory"`
	Disk   int64  `json:"disk"`
	Image  string `json:"image"`
}

// UpdateVMRequest represents a request to update a VM
type UpdateVMRequest struct {
	Name   *string `json:"name,omitempty"`
	CPU    *int    `json:"cpu,omitempty"`
	Memory *int64  `json:"memory,omitempty"`
	Disk   *int64  `json:"disk,omitempty"`
}

func (r *UpdateVMRequest) toVMConfig() *vm.Config {
	config := &vm.Config{}
	if r.Name != nil {
		config.Name = *r.Name
	}
	if r.CPU != nil {
		config.CPU = *r.CPU
	}
	if r.Memory != nil {
		config.Memory = *r.Memory
	}
	if r.Disk != nil {
		config.Disk = *r.Disk
	}
	return config
}

// MigrateVMRequest represents a request to migrate a VM
type MigrateVMRequest struct {
	TargetHost string `json:"target_host"`
	Live       bool   `json:"live"`
}

// SnapshotVMRequest represents a request to snapshot a VM
type SnapshotVMRequest struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
}

// CreateVolumeRequest represents a request to create a volume
type CreateVolumeRequest struct {
	Name string `json:"name"`
	Size int64  `json:"size"`
	Tier string `json:"tier"`
}

// ChangeTierRequest represents a request to change storage tier
type ChangeTierRequest struct {
	NewTier string `json:"new_tier"`
}

// Response types

// Node represents a cluster node
type Node struct {
	ID      string `json:"id"`
	Address string `json:"address"`
	Status  string `json:"status"`
	CPU     int    `json:"cpu,omitempty"`
	Memory  int64  `json:"memory,omitempty"`
	Disk    int64  `json:"disk,omitempty"`
}

// ClusterHealth represents cluster health status
type ClusterHealth struct {
	Status       string    `json:"status"`
	TotalNodes   int       `json:"total_nodes"`
	HealthyNodes int       `json:"healthy_nodes"`
	HasQuorum    bool      `json:"has_quorum"`
	Leader       string    `json:"leader"`
	LastUpdated  time.Time `json:"last_updated"`
}

// SystemMetrics represents system-wide metrics
type SystemMetrics struct {
	CPU       CPUMetrics     `json:"cpu"`
	Memory    MemoryMetrics  `json:"memory"`
	Disk      DiskMetrics    `json:"disk"`
	Network   NetworkMetrics `json:"network"`
	Timestamp time.Time      `json:"timestamp"`
}

// CPUMetrics represents CPU metrics
type CPUMetrics struct {
	Usage   float64   `json:"usage"`
	Cores   int       `json:"cores"`
	LoadAvg []float64 `json:"load_avg"`
}

// MemoryMetrics represents memory metrics
type MemoryMetrics struct {
	Total     int64 `json:"total"`
	Used      int64 `json:"used"`
	Free      int64 `json:"free"`
	Cached    int64 `json:"cached"`
	Available int64 `json:"available"`
}

// DiskMetrics represents disk metrics
type DiskMetrics struct {
	Total int64 `json:"total"`
	Used  int64 `json:"used"`
	Free  int64 `json:"free"`
}

// NetworkMetrics represents network metrics
type NetworkMetrics struct {
	BytesIn    int64 `json:"bytes_in"`
	BytesOut   int64 `json:"bytes_out"`
	PacketsIn  int64 `json:"packets_in"`
	PacketsOut int64 `json:"packets_out"`
}

// Alert represents a system alert
type Alert struct {
	ID           string    `json:"id"`
	Severity     string    `json:"severity"`
	Message      string    `json:"message"`
	Source       string    `json:"source"`
	Timestamp    time.Time `json:"timestamp"`
	Acknowledged bool      `json:"acknowledged"`
}

// Event represents a system event
type Event struct {
	ID        string    `json:"id"`
	Type      string    `json:"type"`
	Message   string    `json:"message"`
	Source    string    `json:"source"`
	Timestamp time.Time `json:"timestamp"`
	Details   map[string]interface{} `json:"details,omitempty"`
}

// PaginatedResponse represents a paginated response
type PaginatedResponse struct {
	Data       interface{} `json:"data"`
	Total      int         `json:"total"`
	Page       int         `json:"page"`
	PageSize   int         `json:"page_size"`
	HasMore    bool        `json:"has_more"`
}