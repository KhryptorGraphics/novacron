package vm

import (
	"context"
	"time"
)

// Additional interfaces needed by the API handlers that aren't defined elsewhere

// MetricsCollector defines the interface for collecting VM metrics
type MetricsCollector interface {
	// GetVMMetrics returns metrics for a specific VM
	GetVMMetrics(ctx context.Context, vmID string) (*VMMetrics, error)
	
	// GetAggregatedMetrics returns aggregated metrics across all VMs
	GetAggregatedMetrics(ctx context.Context) (*AggregatedMetrics, error)
	
	// GetUtilization returns resource utilization statistics
	GetUtilization(ctx context.Context) (*UtilizationMetrics, error)
	
	// GetPerformanceMetrics returns performance metrics
	GetPerformanceMetrics(ctx context.Context, vmID string) (*PerformanceMetrics, error)
	
	// ExportMetrics exports metrics in Prometheus format
	ExportMetrics(ctx context.Context) ([]byte, error)
}

// HealthCheckerInterface defines the interface for health checking
type HealthCheckerInterface interface {
	// GetVMHealth returns health status for a specific VM
	GetVMHealth(ctx context.Context, vmID string) (*VMHealthStatus, error)
	
	// GetSystemHealth returns overall system health
	GetSystemHealth(ctx context.Context) (*SystemHealthStatus, error)
	
	// GetServicesHealth returns health of all services
	GetServicesHealth(ctx context.Context) (*ServicesHealthStatus, error)
	
	// LivenessProbe performs a liveness check
	LivenessProbe(ctx context.Context) (*ProbeResult, error)
	
	// ReadinessProbe performs a readiness check
	ReadinessProbe(ctx context.Context) (*ProbeResult, error)
}

// ClusterManager defines the interface for cluster management
type ClusterManager interface {
	// ListNodes returns all cluster nodes
	ListNodes(ctx context.Context) ([]*ClusterNode, error)
	
	// GetNode returns a specific cluster node
	GetNode(ctx context.Context, nodeID string) (*ClusterNode, error)
	
	// AddNode adds a node to the cluster
	AddNode(ctx context.Context, node *ClusterNode) error
	
	// RemoveNode removes a node from the cluster
	RemoveNode(ctx context.Context, nodeID string) error
	
	// DrainNode evacuates VMs from a node
	DrainNode(ctx context.Context, nodeID string) error
	
	// CordonNode prevents new VMs from being scheduled on the node
	CordonNode(ctx context.Context, nodeID string) error
	
	// UncordonNode allows VMs to be scheduled on the node again
	UncordonNode(ctx context.Context, nodeID string) error
	
	// GetClusterStatus returns overall cluster status
	GetClusterStatus(ctx context.Context) (*ClusterStatus, error)
	
	// RebalanceCluster rebalances VMs across nodes
	RebalanceCluster(ctx context.Context) error
}

// AggregatedMetrics represents aggregated metrics across multiple VMs
type AggregatedMetrics struct {
	TotalVMs          int                    `json:"total_vms"`
	RunningVMs        int                    `json:"running_vms"`
	TotalCPUUsage     float64               `json:"total_cpu_usage"`
	TotalMemoryUsage  int64                 `json:"total_memory_usage"`
	TotalNetworkSent  int64                 `json:"total_network_sent"`
	TotalNetworkRecv  int64                 `json:"total_network_recv"`
	PerNodeMetrics    map[string]*VMMetrics `json:"per_node_metrics"`
	LastUpdated       time.Time             `json:"last_updated"`
}

// UtilizationMetrics represents resource utilization
type UtilizationMetrics struct {
	CPUUtilization    float64   `json:"cpu_utilization"`
	MemoryUtilization float64   `json:"memory_utilization"`
	DiskUtilization   float64   `json:"disk_utilization"`
	NetworkUtilization float64  `json:"network_utilization"`
	Timestamp         time.Time `json:"timestamp"`
}

// PerformanceMetrics represents performance metrics
type PerformanceMetrics struct {
	VMID              string    `json:"vm_id"`
	ResponseTime      float64   `json:"response_time"`
	Throughput        float64   `json:"throughput"`
	IOPS              int64     `json:"iops"`
	Latency           float64   `json:"latency"`
	ErrorRate         float64   `json:"error_rate"`
	Timestamp         time.Time `json:"timestamp"`
}

// VMHealthStatus represents VM health status
type VMHealthStatus struct {
	VMID         string            `json:"vm_id"`
	Status       string            `json:"status"`
	Healthy      bool              `json:"healthy"`
	Checks       map[string]string `json:"checks"`
	LastCheck    time.Time         `json:"last_check"`
	Issues       []string          `json:"issues,omitempty"`
}

// SystemHealthStatus represents overall system health
type SystemHealthStatus struct {
	Status       string                       `json:"status"`
	Healthy      bool                         `json:"healthy"`
	Components   map[string]*ComponentHealth  `json:"components"`
	LastCheck    time.Time                    `json:"last_check"`
	Issues       []string                     `json:"issues,omitempty"`
}

// ServicesHealthStatus represents health of all services
type ServicesHealthStatus struct {
	Services    map[string]*ServiceHealth `json:"services"`
	LastCheck   time.Time                 `json:"last_check"`
}

// ComponentHealth represents health of a system component
type ComponentHealth struct {
	Name      string    `json:"name"`
	Status    string    `json:"status"`
	Healthy   bool      `json:"healthy"`
	Message   string    `json:"message,omitempty"`
	LastCheck time.Time `json:"last_check"`
}

// ServiceHealth represents health of a service
type ServiceHealth struct {
	Name      string    `json:"name"`
	Status    string    `json:"status"`
	Healthy   bool      `json:"healthy"`
	Version   string    `json:"version,omitempty"`
	Uptime    time.Duration `json:"uptime"`
	LastCheck time.Time `json:"last_check"`
}

// ProbeResult represents the result of a probe
type ProbeResult struct {
	Healthy   bool      `json:"healthy"`
	Status    string    `json:"status"`
	Message   string    `json:"message,omitempty"`
	Timestamp time.Time `json:"timestamp"`
}

// ClusterNode represents a node in the cluster
type ClusterNode struct {
	ID             string            `json:"id"`
	Name           string            `json:"name"`
	Address        string            `json:"address"`
	Status         string            `json:"status"`
	Roles          []string          `json:"roles"`
	Labels         map[string]string `json:"labels"`
	Annotations    map[string]string `json:"annotations"`
	Capacity       *NodeCapacity     `json:"capacity"`
	Allocatable    *NodeCapacity     `json:"allocatable"`
	Conditions     []*NodeCondition  `json:"conditions"`
	CreatedAt      time.Time         `json:"created_at"`
	LastHeartbeat  time.Time         `json:"last_heartbeat"`
	Cordoned       bool              `json:"cordoned"`
	Draining       bool              `json:"draining"`
}

// NodeCapacity represents the capacity of a node
type NodeCapacity struct {
	CPU      int64 `json:"cpu"`
	Memory   int64 `json:"memory"`
	Storage  int64 `json:"storage"`
	Pods     int   `json:"pods"`
}

// NodeCondition represents a condition of a node
type NodeCondition struct {
	Type               string    `json:"type"`
	Status             string    `json:"status"`
	LastHeartbeatTime  time.Time `json:"last_heartbeat_time"`
	LastTransitionTime time.Time `json:"last_transition_time"`
	Reason             string    `json:"reason"`
	Message            string    `json:"message"`
}

// ClusterStatus represents overall cluster status
type ClusterStatus struct {
	Status         string                   `json:"status"`
	TotalNodes     int                      `json:"total_nodes"`
	ReadyNodes     int                      `json:"ready_nodes"`
	NotReadyNodes  int                      `json:"not_ready_nodes"`
	TotalVMs       int                      `json:"total_vms"`
	RunningVMs     int                      `json:"running_vms"`
	Capacity       *NodeCapacity            `json:"capacity"`
	Allocated      *NodeCapacity            `json:"allocated"`
	Utilization    *UtilizationMetrics      `json:"utilization"`
	Nodes          []*ClusterNode           `json:"nodes"`
	LastUpdated    time.Time                `json:"last_updated"`
}