package edge

import (
	"time"
)

// EdgeNode represents an edge computing node
type EdgeNode struct {
	ID           string             `json:"id"`
	Name         string             `json:"name"`
	Type         EdgeType           `json:"type"`
	Location     GeoLocation        `json:"location"`
	Capabilities EdgeCapability     `json:"capabilities"`
	Resources    EdgeResources      `json:"resources"`
	Status       EdgeNodeStatus     `json:"status"`
	Latency      LatencyMetrics     `json:"latency"`
	Cost         CostMetrics        `json:"cost"`
	Network      NetworkInfo        `json:"network"`
	Metadata     map[string]string  `json:"metadata"`
	CreatedAt    time.Time          `json:"created_at"`
	UpdatedAt    time.Time          `json:"updated_at"`
	LastSeenAt   time.Time          `json:"last_seen_at"`
}

// GeoLocation represents geographic location
type GeoLocation struct {
	Latitude  float64 `json:"latitude"`
	Longitude float64 `json:"longitude"`
	Country   string  `json:"country"`
	Region    string  `json:"region"`
	City      string  `json:"city"`
	Datacenter string `json:"datacenter"`
}

// EdgeResources represents available resources on edge node
type EdgeResources struct {
	TotalCPUCores     int     `json:"total_cpu_cores"`
	UsedCPUCores      int     `json:"used_cpu_cores"`
	TotalMemoryMB     int64   `json:"total_memory_mb"`
	UsedMemoryMB      int64   `json:"used_memory_mb"`
	TotalStorageGB    int64   `json:"total_storage_gb"`
	UsedStorageGB     int64   `json:"used_storage_gb"`
	TotalBandwidthMbps int    `json:"total_bandwidth_mbps"`
	UsedBandwidthMbps  int    `json:"used_bandwidth_mbps"`
	GPUCount          int     `json:"gpu_count"`
	UtilizationPercent float64 `json:"utilization_percent"`
}

// EdgeNodeStatus represents the status of an edge node
type EdgeNodeStatus struct {
	State       EdgeNodeState `json:"state"`
	Health      HealthStatus  `json:"health"`
	ActiveVMs   int           `json:"active_vms"`
	LastCheck   time.Time     `json:"last_check"`
	ErrorCount  int           `json:"error_count"`
	UptimeSeconds int64       `json:"uptime_seconds"`
}

// EdgeNodeState represents the state of an edge node
type EdgeNodeState string

const (
	EdgeNodeStateOnline    EdgeNodeState = "online"
	EdgeNodeStateOffline   EdgeNodeState = "offline"
	EdgeNodeStateDegraded  EdgeNodeState = "degraded"
	EdgeNodeStateMaintenance EdgeNodeState = "maintenance"
)

// HealthStatus represents health status
type HealthStatus string

const (
	HealthStatusHealthy   HealthStatus = "healthy"
	HealthStatusUnhealthy HealthStatus = "unhealthy"
	HealthStatusUnknown   HealthStatus = "unknown"
)

// LatencyMetrics represents latency measurements
type LatencyMetrics struct {
	ToCloud      time.Duration `json:"to_cloud"`
	ToUser       time.Duration `json:"to_user"`
	RTTAvg       time.Duration `json:"rtt_avg"`
	RTTMin       time.Duration `json:"rtt_min"`
	RTTMax       time.Duration `json:"rtt_max"`
	JitterMs     float64       `json:"jitter_ms"`
	PacketLoss   float64       `json:"packet_loss"`
	MeasuredAt   time.Time     `json:"measured_at"`
}

// CostMetrics represents cost information
type CostMetrics struct {
	CostPerHour     float64 `json:"cost_per_hour"`
	CostPerGB       float64 `json:"cost_per_gb"`
	CostPerVM       float64 `json:"cost_per_vm"`
	Currency        string  `json:"currency"`
	BillingModel    string  `json:"billing_model"` // "prepaid", "postpaid", "free"
}

// NetworkInfo represents network configuration
type NetworkInfo struct {
	PublicIP      string   `json:"public_ip"`
	PrivateIP     string   `json:"private_ip"`
	VPNEndpoint   string   `json:"vpn_endpoint"`
	MeshPeers     []string `json:"mesh_peers"`
	CDNEndpoint   string   `json:"cdn_endpoint"`
	NetworkSliceID string  `json:"network_slice_id"`
}

// PlacementDecision represents an edge placement decision
type PlacementDecision struct {
	EdgeNodeID    string        `json:"edge_node_id"`
	Score         float64       `json:"score"`
	LatencyScore  float64       `json:"latency_score"`
	ResourceScore float64       `json:"resource_score"`
	CostScore     float64       `json:"cost_score"`
	ProximityScore float64      `json:"proximity_score"`
	Reason        string        `json:"reason"`
	EstimatedLatency time.Duration `json:"estimated_latency"`
	DecisionTime  time.Duration `json:"decision_time"`
	CreatedAt     time.Time     `json:"created_at"`
}

// PlacementRequest represents a request for edge placement
type PlacementRequest struct {
	VMID          string            `json:"vm_id"`
	UserLocation  *GeoLocation      `json:"user_location"`
	Requirements  PlacementRequirements `json:"requirements"`
	Constraints   PlacementConstraints  `json:"constraints"`
	Preferences   PlacementPreferences  `json:"preferences"`
}

// PlacementRequirements represents resource requirements
type PlacementRequirements struct {
	CPUCores      int   `json:"cpu_cores"`
	MemoryMB      int64 `json:"memory_mb"`
	StorageGB     int64 `json:"storage_gb"`
	BandwidthMbps int   `json:"bandwidth_mbps"`
	GPURequired   bool  `json:"gpu_required"`
	Architecture  string `json:"architecture"` // "x86_64", "arm64"
}

// PlacementConstraints represents hard constraints
type PlacementConstraints struct {
	MaxLatency       time.Duration `json:"max_latency"`
	RequiredRegions  []string      `json:"required_regions"`
	ExcludedRegions  []string      `json:"excluded_regions"`
	RequiredEdgeType EdgeType      `json:"required_edge_type"`
	DataResidency    string        `json:"data_residency"` // "EU", "US", etc.
}

// PlacementPreferences represents soft preferences
type PlacementPreferences struct {
	PreferredRegions []string `json:"preferred_regions"`
	PreferLowCost    bool     `json:"prefer_low_cost"`
	PreferLowLatency bool     `json:"prefer_low_latency"`
	GreenComputing   bool     `json:"green_computing"`
}

// MigrationRequest represents a migration request
type MigrationRequest struct {
	VMID           string        `json:"vm_id"`
	SourceNodeID   string        `json:"source_node_id"`
	TargetNodeID   string        `json:"target_node_id"`
	MigrationType  MigrationType `json:"migration_type"`
	MaxDowntime    time.Duration `json:"max_downtime"`
	Priority       int           `json:"priority"`
	Reason         string        `json:"reason"`
}

// MigrationType represents the type of migration
type MigrationType string

const (
	MigrationTypeLive        MigrationType = "live"
	MigrationTypeColdMigrate MigrationType = "cold"
	MigrationTypeSnapshot    MigrationType = "snapshot"
)

// MigrationStatus represents migration status
type MigrationStatus struct {
	VMID          string        `json:"vm_id"`
	State         MigrationState `json:"state"`
	Progress      float64       `json:"progress"`
	BytesCopied   int64         `json:"bytes_copied"`
	TotalBytes    int64         `json:"total_bytes"`
	DowntimeMs    int64         `json:"downtime_ms"`
	ElapsedTime   time.Duration `json:"elapsed_time"`
	EstimatedTime time.Duration `json:"estimated_time"`
	Error         string        `json:"error,omitempty"`
	StartedAt     time.Time     `json:"started_at"`
	CompletedAt   *time.Time    `json:"completed_at,omitempty"`
}

// MigrationState represents migration state
type MigrationState string

const (
	MigrationStatePending   MigrationState = "pending"
	MigrationStateRunning   MigrationState = "running"
	MigrationStateCompleted MigrationState = "completed"
	MigrationStateFailed    MigrationState = "failed"
	MigrationStateCancelled MigrationState = "cancelled"
)

// EdgeMetrics represents edge node metrics
type EdgeMetrics struct {
	NodeID            string        `json:"node_id"`
	CPUUtilization    float64       `json:"cpu_utilization"`
	MemoryUtilization float64       `json:"memory_utilization"`
	StorageUtilization float64      `json:"storage_utilization"`
	NetworkUtilization float64      `json:"network_utilization"`
	ActiveVMs         int           `json:"active_vms"`
	TotalRequests     int64         `json:"total_requests"`
	AvgLatencyMs      float64       `json:"avg_latency_ms"`
	P95LatencyMs      float64       `json:"p95_latency_ms"`
	ErrorRate         float64       `json:"error_rate"`
	Timestamp         time.Time     `json:"timestamp"`
}

// UserProximity represents user proximity to edge nodes
type UserProximity struct {
	UserLocation  GeoLocation   `json:"user_location"`
	NearestNodes  []EdgeNode    `json:"nearest_nodes"`
	DistancesKM   []float64     `json:"distances_km"`
	EstimatedLatencies []time.Duration `json:"estimated_latencies"`
	CalculatedAt  time.Time     `json:"calculated_at"`
}
