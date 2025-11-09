package federation

import (
	"fmt"
	"time"
)

// DWCPConfig contains DWCP-specific configuration for federation
type DWCPConfig struct {
	// Enable DWCP protocol for cross-cluster communication
	Enabled bool `json:"enabled" yaml:"enabled"`

	// HDE (Hierarchical Delta Encoding) settings
	HDEEnabled        bool          `json:"hde_enabled" yaml:"hde_enabled"`
	DictionarySize    int           `json:"dictionary_size" yaml:"dictionary_size"`
	BaselineInterval  time.Duration `json:"baseline_interval" yaml:"baseline_interval"`
	CompressionLevel  int           `json:"compression_level" yaml:"compression_level"`
	CompressionRatio  float64       `json:"compression_ratio" yaml:"compression_ratio"`

	// AMST (Adaptive Multi-Stream Transport) settings
	AMSTEnabled       bool          `json:"amst_enabled" yaml:"amst_enabled"`
	DataStreams       int           `json:"data_streams" yaml:"data_streams"`
	StreamBufferSize  int           `json:"stream_buffer_size" yaml:"stream_buffer_size"`

	// Network settings
	ConnectionTimeout time.Duration `json:"connection_timeout" yaml:"connection_timeout"`
	KeepAliveInterval time.Duration `json:"keepalive_interval" yaml:"keepalive_interval"`
	RetryInterval     time.Duration `json:"retry_interval" yaml:"retry_interval"`
	MaxRetries        int           `json:"max_retries" yaml:"max_retries"`

	// Bandwidth optimization
	BandwidthThreshold float64       `json:"bandwidth_threshold" yaml:"bandwidth_threshold"`
	AdaptiveMode       bool          `json:"adaptive_mode" yaml:"adaptive_mode"`
	LatencyThreshold   time.Duration `json:"latency_threshold" yaml:"latency_threshold"`

	// Partition tolerance
	PartitionTolerance bool          `json:"partition_tolerance" yaml:"partition_tolerance"`
	BufferSize         int           `json:"buffer_size" yaml:"buffer_size"`
	RecoveryTimeout    time.Duration `json:"recovery_timeout" yaml:"recovery_timeout"`
}

// FederationConfiguration contains complete federation configuration
type FederationConfiguration struct {
	// General federation settings
	Enabled           bool          `json:"enabled" yaml:"enabled"`
	ClusterID         string        `json:"cluster_id" yaml:"cluster_id"`
	Region            string        `json:"region" yaml:"region"`
	Datacenter        string        `json:"datacenter" yaml:"datacenter"`

	// DWCP protocol configuration
	DWCP              DWCPConfig    `json:"dwcp" yaml:"dwcp"`

	// State synchronization
	SyncInterval      time.Duration `json:"sync_interval" yaml:"sync_interval"`
	SyncBatchSize     int           `json:"sync_batch_size" yaml:"sync_batch_size"`
	EventualConsistency bool        `json:"eventual_consistency" yaml:"eventual_consistency"`

	// Consensus replication
	ConsensusTimeout  time.Duration `json:"consensus_timeout" yaml:"consensus_timeout"`
	LogBatchSize      int           `json:"log_batch_size" yaml:"log_batch_size"`
	LogCompression    bool          `json:"log_compression" yaml:"log_compression"`

	// Resource sharing
	ResourceSharing   bool          `json:"resource_sharing" yaml:"resource_sharing"`
	ShareThreshold    float64       `json:"share_threshold" yaml:"share_threshold"`
	BorrowingEnabled  bool          `json:"borrowing_enabled" yaml:"borrowing_enabled"`

	// Migration settings
	MigrationEnabled  bool          `json:"migration_enabled" yaml:"migration_enabled"`
	LiveMigration     bool          `json:"live_migration" yaml:"live_migration"`
	MigrationTimeout  time.Duration `json:"migration_timeout" yaml:"migration_timeout"`

	// Security
	TLSEnabled        bool          `json:"tls_enabled" yaml:"tls_enabled"`
	MutualTLS         bool          `json:"mutual_tls" yaml:"mutual_tls"`
	CertificatePath   string        `json:"certificate_path" yaml:"certificate_path"`
	PrivateKeyPath    string        `json:"private_key_path" yaml:"private_key_path"`
	CAPath            string        `json:"ca_path" yaml:"ca_path"`

	// Monitoring
	MetricsEnabled    bool          `json:"metrics_enabled" yaml:"metrics_enabled"`
	MetricsInterval   time.Duration `json:"metrics_interval" yaml:"metrics_interval"`
	TracingEnabled    bool          `json:"tracing_enabled" yaml:"tracing_enabled"`

	// Cluster endpoints
	ClusterEndpoints  map[string]ClusterEndpoint `json:"cluster_endpoints" yaml:"cluster_endpoints"`
}

// ClusterEndpoint defines connection details for a cluster
type ClusterEndpoint struct {
	ID                string        `json:"id" yaml:"id"`
	Address           string        `json:"address" yaml:"address"`
	Port              int           `json:"port" yaml:"port"`
	Region            string        `json:"region" yaml:"region"`
	Datacenter        string        `json:"datacenter" yaml:"datacenter"`
	Priority          int           `json:"priority" yaml:"priority"`
	Weight            int           `json:"weight" yaml:"weight"`
	MaxConnections    int           `json:"max_connections" yaml:"max_connections"`
	HealthCheckURL    string        `json:"health_check_url" yaml:"health_check_url"`
	HealthCheckInterval time.Duration `json:"health_check_interval" yaml:"health_check_interval"`
}

// DefaultFederationConfiguration returns default federation configuration
func DefaultFederationConfiguration() *FederationConfiguration {
	return &FederationConfiguration{
		Enabled:    true,
		ClusterID:  "novacron-primary",
		Region:     "us-east-1",
		Datacenter: "dc1",

		DWCP: DWCPConfig{
			Enabled:            true,
			HDEEnabled:         true,
			DictionarySize:     100 * 1024, // 100KB
			BaselineInterval:   5 * time.Minute,
			CompressionLevel:   6,
			CompressionRatio:   10.0,
			AMSTEnabled:        true,
			DataStreams:        4,
			StreamBufferSize:   64 * 1024, // 64KB
			ConnectionTimeout:  30 * time.Second,
			KeepAliveInterval:  30 * time.Second,
			RetryInterval:      5 * time.Second,
			MaxRetries:         10,
			BandwidthThreshold: 0.6,
			AdaptiveMode:       true,
			LatencyThreshold:   100 * time.Millisecond,
			PartitionTolerance: true,
			BufferSize:         10000,
			RecoveryTimeout:    5 * time.Minute,
		},

		SyncInterval:        10 * time.Second,
		SyncBatchSize:       100,
		EventualConsistency: false,

		ConsensusTimeout: 5 * time.Second,
		LogBatchSize:     100,
		LogCompression:   true,

		ResourceSharing:  true,
		ShareThreshold:   0.8,
		BorrowingEnabled: true,

		MigrationEnabled: true,
		LiveMigration:    true,
		MigrationTimeout: 10 * time.Minute,

		TLSEnabled:      true,
		MutualTLS:       true,
		CertificatePath: "/etc/novacron/certs/server.crt",
		PrivateKeyPath:  "/etc/novacron/certs/server.key",
		CAPath:          "/etc/novacron/certs/ca.crt",

		MetricsEnabled:  true,
		MetricsInterval: 30 * time.Second,
		TracingEnabled:  true,

		ClusterEndpoints: make(map[string]ClusterEndpoint),
	}
}

// ProductionDWCPConfig returns production-optimized DWCP configuration
func ProductionDWCPConfig() DWCPConfig {
	return DWCPConfig{
		Enabled:            true,
		HDEEnabled:         true,
		DictionarySize:     256 * 1024, // 256KB for better compression
		BaselineInterval:   3 * time.Minute, // More frequent baselines
		CompressionLevel:   7, // Higher compression
		CompressionRatio:   10.0,
		AMSTEnabled:        true,
		DataStreams:        8, // More parallel streams
		StreamBufferSize:   128 * 1024, // Larger buffers
		ConnectionTimeout:  20 * time.Second,
		KeepAliveInterval:  20 * time.Second,
		RetryInterval:      3 * time.Second,
		MaxRetries:         15,
		BandwidthThreshold: 0.5, // More aggressive optimization
		AdaptiveMode:       true,
		LatencyThreshold:   50 * time.Millisecond,
		PartitionTolerance: true,
		BufferSize:         50000, // Larger buffer for partitions
		RecoveryTimeout:    10 * time.Minute,
	}
}

// MultiRegionConfig returns configuration optimized for multi-region deployment
func MultiRegionConfig() *FederationConfiguration {
	config := DefaultFederationConfiguration()

	// Optimize for high-latency WAN links
	config.DWCP.CompressionLevel = 9 // Maximum compression
	config.DWCP.BaselineInterval = 2 * time.Minute // Frequent baselines
	config.DWCP.DataStreams = 16 // Maximum parallelism
	config.DWCP.BandwidthThreshold = 0.4 // Aggressive optimization
	config.DWCP.LatencyThreshold = 200 * time.Millisecond // Higher tolerance

	// Adjust sync settings for WAN
	config.SyncInterval = 30 * time.Second
	config.EventualConsistency = true // Allow eventual consistency
	config.ConsensusTimeout = 10 * time.Second // Longer timeout

	return config
}

// ValidateConfiguration validates the federation configuration
func (fc *FederationConfiguration) Validate() error {
	// Validate required fields
	if fc.ClusterID == "" {
		return fmt.Errorf("cluster ID is required")
	}

	// Validate DWCP settings
	if fc.DWCP.Enabled {
		if fc.DWCP.DictionarySize < 1024 {
			return fmt.Errorf("dictionary size must be at least 1KB")
		}
		if fc.DWCP.CompressionLevel < 1 || fc.DWCP.CompressionLevel > 9 {
			return fmt.Errorf("compression level must be between 1 and 9")
		}
		if fc.DWCP.DataStreams < 1 {
			return fmt.Errorf("must have at least 1 data stream")
		}
		if fc.DWCP.BandwidthThreshold < 0 || fc.DWCP.BandwidthThreshold > 1 {
			return fmt.Errorf("bandwidth threshold must be between 0 and 1")
		}
	}

	// Validate timeouts
	if fc.SyncInterval < time.Second {
		return fmt.Errorf("sync interval must be at least 1 second")
	}
	if fc.ConsensusTimeout < time.Second {
		return fmt.Errorf("consensus timeout must be at least 1 second")
	}

	// Validate endpoints
	for id, endpoint := range fc.ClusterEndpoints {
		if endpoint.Address == "" {
			return fmt.Errorf("cluster %s has no address", id)
		}
		if endpoint.Port <= 0 || endpoint.Port > 65535 {
			return fmt.Errorf("cluster %s has invalid port %d", id, endpoint.Port)
		}
	}

	return nil
}

// GetBandwidthSavings calculates estimated bandwidth savings with DWCP
func (fc *FederationConfiguration) GetBandwidthSavings() float64 {
	if !fc.DWCP.Enabled {
		return 0.0
	}

	// Base savings from compression
	compressionSavings := 1.0 - (1.0 / fc.DWCP.CompressionRatio)

	// Additional savings from delta encoding
	deltaSavings := 0.0
	if fc.DWCP.HDEEnabled {
		deltaSavings = 0.3 // Estimated 30% additional savings
	}

	// Additional savings from AMST multi-streaming
	streamSavings := 0.0
	if fc.DWCP.AMSTEnabled {
		streamSavings = 0.1 // Estimated 10% from efficient streaming
	}

	// Total savings (capped at realistic maximum)
	totalSavings := compressionSavings + deltaSavings + streamSavings
	if totalSavings > 0.9 {
		totalSavings = 0.9 // Cap at 90% reduction
	}

	return totalSavings * 100.0 // Return as percentage
}