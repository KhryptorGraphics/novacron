package edge

import (
	"time"
)

// EdgeConfig defines configuration for edge computing integration
type EdgeConfig struct {
	// Discovery settings
	DiscoveryInterval    time.Duration `json:"discovery_interval" yaml:"discovery_interval"`
	DiscoveryTimeout     time.Duration `json:"discovery_timeout" yaml:"discovery_timeout"`

	// Placement optimization
	PlacementWeights     PlacementWeights `json:"placement_weights" yaml:"placement_weights"`
	MaxEdgeLatency       time.Duration    `json:"max_edge_latency" yaml:"max_edge_latency"`
	MinEdgeResources     ResourceReq      `json:"min_edge_resources" yaml:"min_edge_resources"`

	// Migration settings
	MigrationTimeout     time.Duration `json:"migration_timeout" yaml:"migration_timeout"`
	MigrationBandwidth   int64         `json:"migration_bandwidth" yaml:"migration_bandwidth"` // bytes/sec

	// Feature flags
	EnableMEC            bool `json:"enable_mec" yaml:"enable_mec"`
	EnableIoTGateway     bool `json:"enable_iot_gateway" yaml:"enable_iot_gateway"`
	EnableCDNEdge        bool `json:"enable_cdn_edge" yaml:"enable_cdn_edge"`
	EnableTelcoEdge      bool `json:"enable_telco_edge" yaml:"enable_telco_edge"`

	// Performance targets
	TargetProvisionTime  time.Duration `json:"target_provision_time" yaml:"target_provision_time"`  // <30s
	TargetMigrationTime  time.Duration `json:"target_migration_time" yaml:"target_migration_time"`  // <5s
	TargetPlacementTime  time.Duration `json:"target_placement_time" yaml:"target_placement_time"`  // <100ms

	// Resource management
	MaxEdgeVMsPerNode    int     `json:"max_edge_vms_per_node" yaml:"max_edge_vms_per_node"`
	EdgeResourceBuffer   float64 `json:"edge_resource_buffer" yaml:"edge_resource_buffer"` // 0.15 = 15%

	// Networking
	EdgeMeshEnabled      bool   `json:"edge_mesh_enabled" yaml:"edge_mesh_enabled"`
	VPNEnabled           bool   `json:"vpn_enabled" yaml:"vpn_enabled"`
	OfflineMode          bool   `json:"offline_mode" yaml:"offline_mode"`

	// Monitoring
	MetricsInterval      time.Duration `json:"metrics_interval" yaml:"metrics_interval"`
	HealthCheckInterval  time.Duration `json:"health_check_interval" yaml:"health_check_interval"`

	// Security
	RequireTLS           bool     `json:"require_tls" yaml:"require_tls"`
	AllowedRegions       []string `json:"allowed_regions" yaml:"allowed_regions"`
	DataResidencyRules   []string `json:"data_residency_rules" yaml:"data_residency_rules"`
}

// PlacementWeights defines weights for edge placement algorithm
type PlacementWeights struct {
	Latency    float64 `json:"latency" yaml:"latency"`       // 0.5
	Resources  float64 `json:"resources" yaml:"resources"`   // 0.3
	Cost       float64 `json:"cost" yaml:"cost"`             // 0.2
	Proximity  float64 `json:"proximity" yaml:"proximity"`   // Optional bonus
	Energy     float64 `json:"energy" yaml:"energy"`         // Green computing
}

// ResourceReq defines minimum resource requirements for edge nodes
type ResourceReq struct {
	MinCPUCores   int   `json:"min_cpu_cores" yaml:"min_cpu_cores"`
	MinMemoryMB   int64 `json:"min_memory_mb" yaml:"min_memory_mb"`
	MinStorageGB  int64 `json:"min_storage_gb" yaml:"min_storage_gb"`
	MinBandwidthMbps int `json:"min_bandwidth_mbps" yaml:"min_bandwidth_mbps"`
}

// DefaultEdgeConfig returns default configuration
func DefaultEdgeConfig() *EdgeConfig {
	return &EdgeConfig{
		DiscoveryInterval:    30 * time.Second,
		DiscoveryTimeout:     5 * time.Second,
		PlacementWeights: PlacementWeights{
			Latency:   0.5,
			Resources: 0.3,
			Cost:      0.2,
			Proximity: 0.0,
			Energy:    0.0,
		},
		MaxEdgeLatency: 100 * time.Millisecond,
		MinEdgeResources: ResourceReq{
			MinCPUCores:      2,
			MinMemoryMB:      2048,
			MinStorageGB:     20,
			MinBandwidthMbps: 100,
		},
		MigrationTimeout:     5 * time.Second,
		MigrationBandwidth:   1_000_000_000, // 1 Gbps
		EnableMEC:            true,
		EnableIoTGateway:     true,
		EnableCDNEdge:        true,
		EnableTelcoEdge:      false,
		TargetProvisionTime:  30 * time.Second,
		TargetMigrationTime:  5 * time.Second,
		TargetPlacementTime:  100 * time.Millisecond,
		MaxEdgeVMsPerNode:    10,
		EdgeResourceBuffer:   0.15,
		EdgeMeshEnabled:      true,
		VPNEnabled:           true,
		OfflineMode:          false,
		MetricsInterval:      10 * time.Second,
		HealthCheckInterval:  15 * time.Second,
		RequireTLS:           true,
		AllowedRegions:       []string{"*"},
		DataResidencyRules:   []string{},
	}
}

// Validate validates the configuration
func (c *EdgeConfig) Validate() error {
	if c.DiscoveryInterval < 1*time.Second {
		return ErrInvalidDiscoveryInterval
	}

	if c.MaxEdgeLatency < 1*time.Millisecond {
		return ErrInvalidMaxLatency
	}

	// Validate weights sum to approximately 1.0
	weightSum := c.PlacementWeights.Latency +
		c.PlacementWeights.Resources +
		c.PlacementWeights.Cost +
		c.PlacementWeights.Proximity +
		c.PlacementWeights.Energy

	if weightSum < 0.99 || weightSum > 1.01 {
		return ErrInvalidPlacementWeights
	}

	if c.MinEdgeResources.MinCPUCores < 1 {
		return ErrInvalidMinResources
	}

	if c.MigrationTimeout < 1*time.Second {
		return ErrInvalidMigrationTimeout
	}

	return nil
}

// EdgeType represents the type of edge deployment
type EdgeType string

const (
	EdgeType5GMEC      EdgeType = "5g_mec"
	EdgeTypeCDN        EdgeType = "cdn"
	EdgeTypeIoTGateway EdgeType = "iot_gateway"
	EdgeTypeTelco      EdgeType = "telco"
	EdgeTypeOnPremise  EdgeType = "on_premise"
)

// EdgeCapability represents capabilities of an edge node
type EdgeCapability struct {
	Type             EdgeType `json:"type"`
	SupportsGPU      bool     `json:"supports_gpu"`
	SupportsARM64    bool     `json:"supports_arm64"`
	SupportsX86      bool     `json:"supports_x86"`
	MaxVMs           int      `json:"max_vms"`
	HasLocalStorage  bool     `json:"has_local_storage"`
	NetworkSlicing   bool     `json:"network_slicing"`
	UltraLowLatency  bool     `json:"ultra_low_latency"`
}
