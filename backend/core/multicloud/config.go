package multicloud

import (
	"time"
)

// MultiCloudConfig defines the configuration for multi-cloud operations
type MultiCloudConfig struct {
	Providers         []ProviderConfig   `json:"providers" yaml:"providers"`
	PreferredProvider string             `json:"preferred_provider" yaml:"preferred_provider"`
	EnableBursting    bool               `json:"enable_bursting" yaml:"enable_bursting"`
	BurstThresholds   BurstThresholds    `json:"burst_thresholds" yaml:"burst_thresholds"`
	EnableArbitrage   bool               `json:"enable_arbitrage" yaml:"enable_arbitrage"`
	CostOptimization  bool               `json:"cost_optimization" yaml:"cost_optimization"`
	DRProvider        string             `json:"dr_provider" yaml:"dr_provider"`
	NetworkConfig     NetworkConfig      `json:"network_config" yaml:"network_config"`
	MigrationConfig   MigrationConfig    `json:"migration_config" yaml:"migration_config"`
}

// ProviderConfig contains configuration for a specific cloud provider
type ProviderConfig struct {
	Name         string           `json:"name" yaml:"name"`
	Enabled      bool             `json:"enabled" yaml:"enabled"`
	Region       string           `json:"region" yaml:"region"`
	Credentials  CredentialConfig `json:"credentials" yaml:"credentials"`
	Quotas       ResourceQuotas   `json:"quotas" yaml:"quotas"`
	CostLimits   CostLimits       `json:"cost_limits" yaml:"cost_limits"`
	APIEndpoint  string           `json:"api_endpoint" yaml:"api_endpoint"`
	Timeout      time.Duration    `json:"timeout" yaml:"timeout"`
	RetryPolicy  RetryPolicy      `json:"retry_policy" yaml:"retry_policy"`
}

// CredentialConfig contains authentication credentials for a provider
type CredentialConfig struct {
	Type          string            `json:"type" yaml:"type"` // "access_key", "service_account", "managed_identity"
	AccessKey     string            `json:"access_key,omitempty" yaml:"access_key,omitempty"`
	SecretKey     string            `json:"secret_key,omitempty" yaml:"secret_key,omitempty"`
	ProjectID     string            `json:"project_id,omitempty" yaml:"project_id,omitempty"`
	ServiceAccount string           `json:"service_account,omitempty" yaml:"service_account,omitempty"`
	TenantID      string            `json:"tenant_id,omitempty" yaml:"tenant_id,omitempty"`
	ClientID      string            `json:"client_id,omitempty" yaml:"client_id,omitempty"`
	ClientSecret  string            `json:"client_secret,omitempty" yaml:"client_secret,omitempty"`
	KeyFile       string            `json:"key_file,omitempty" yaml:"key_file,omitempty"`
	Profile       string            `json:"profile,omitempty" yaml:"profile,omitempty"`
	ExtraParams   map[string]string `json:"extra_params,omitempty" yaml:"extra_params,omitempty"`
}

// ResourceQuotas defines resource limits for a provider
type ResourceQuotas struct {
	MaxVMs         int     `json:"max_vms" yaml:"max_vms"`
	MaxCPUs        int     `json:"max_cpus" yaml:"max_cpus"`
	MaxMemoryGB    int     `json:"max_memory_gb" yaml:"max_memory_gb"`
	MaxStorageGB   int     `json:"max_storage_gb" yaml:"max_storage_gb"`
	MaxNetworks    int     `json:"max_networks" yaml:"max_networks"`
	MaxLoadBalancers int   `json:"max_load_balancers" yaml:"max_load_balancers"`
	MaxSnapshots   int     `json:"max_snapshots" yaml:"max_snapshots"`
}

// CostLimits defines cost constraints for a provider
type CostLimits struct {
	DailyLimit   float64 `json:"daily_limit" yaml:"daily_limit"`
	MonthlyLimit float64 `json:"monthly_limit" yaml:"monthly_limit"`
	AlertThreshold float64 `json:"alert_threshold" yaml:"alert_threshold"`
}

// BurstThresholds defines when to trigger cloud bursting
type BurstThresholds struct {
	CPUThreshold       float64       `json:"cpu_threshold" yaml:"cpu_threshold"`
	MemoryThreshold    float64       `json:"memory_threshold" yaml:"memory_threshold"`
	QueueDepth         int           `json:"queue_depth" yaml:"queue_depth"`
	Duration           time.Duration `json:"duration" yaml:"duration"`
	ScaleBackThreshold float64       `json:"scale_back_threshold" yaml:"scale_back_threshold"`
	CooldownPeriod     time.Duration `json:"cooldown_period" yaml:"cooldown_period"`
}

// NetworkConfig defines cross-cloud networking configuration
type NetworkConfig struct {
	EnableVPN       bool              `json:"enable_vpn" yaml:"enable_vpn"`
	EnableSDWAN     bool              `json:"enable_sdwan" yaml:"enable_sdwan"`
	VPNType         string            `json:"vpn_type" yaml:"vpn_type"` // "ipsec", "wireguard", "openvpn"
	Bandwidth       int               `json:"bandwidth" yaml:"bandwidth"` // Mbps
	LatencyTarget   time.Duration     `json:"latency_target" yaml:"latency_target"`
	Encryption      string            `json:"encryption" yaml:"encryption"`
	Routes          []RouteConfig     `json:"routes" yaml:"routes"`
}

// RouteConfig defines a cross-cloud route
type RouteConfig struct {
	Source      string `json:"source" yaml:"source"`
	Destination string `json:"destination" yaml:"destination"`
	Gateway     string `json:"gateway" yaml:"gateway"`
	Metric      int    `json:"metric" yaml:"metric"`
}

// MigrationConfig defines migration settings
type MigrationConfig struct {
	EnableLiveMigration bool          `json:"enable_live_migration" yaml:"enable_live_migration"`
	ParallelMigrations  int           `json:"parallel_migrations" yaml:"parallel_migrations"`
	BandwidthLimit      int           `json:"bandwidth_limit" yaml:"bandwidth_limit"` // Mbps
	CompressionEnabled  bool          `json:"compression_enabled" yaml:"compression_enabled"`
	VerificationEnabled bool          `json:"verification_enabled" yaml:"verification_enabled"`
	RollbackEnabled     bool          `json:"rollback_enabled" yaml:"rollback_enabled"`
	Timeout             time.Duration `json:"timeout" yaml:"timeout"`
}

// RetryPolicy defines how to retry failed operations
type RetryPolicy struct {
	MaxRetries     int           `json:"max_retries" yaml:"max_retries"`
	InitialDelay   time.Duration `json:"initial_delay" yaml:"initial_delay"`
	MaxDelay       time.Duration `json:"max_delay" yaml:"max_delay"`
	BackoffFactor  float64       `json:"backoff_factor" yaml:"backoff_factor"`
	RetryableErrors []string      `json:"retryable_errors" yaml:"retryable_errors"`
}

// DefaultMultiCloudConfig returns a default configuration
func DefaultMultiCloudConfig() *MultiCloudConfig {
	return &MultiCloudConfig{
		PreferredProvider: "aws",
		EnableBursting:    true,
		BurstThresholds: BurstThresholds{
			CPUThreshold:       0.90,
			MemoryThreshold:    0.85,
			QueueDepth:         100,
			Duration:           5 * time.Minute,
			ScaleBackThreshold: 0.60,
			CooldownPeriod:     10 * time.Minute,
		},
		EnableArbitrage:  true,
		CostOptimization: true,
		NetworkConfig: NetworkConfig{
			EnableVPN:     true,
			EnableSDWAN:   false,
			VPNType:       "ipsec",
			Bandwidth:     1000, // 1 Gbps
			LatencyTarget: 50 * time.Millisecond,
			Encryption:    "aes256",
		},
		MigrationConfig: MigrationConfig{
			EnableLiveMigration: true,
			ParallelMigrations:  3,
			BandwidthLimit:      500, // 500 Mbps
			CompressionEnabled:  true,
			VerificationEnabled: true,
			RollbackEnabled:     true,
			Timeout:             30 * time.Minute,
		},
	}
}

// ProviderType represents supported cloud providers
type ProviderType string

const (
	ProviderAWS      ProviderType = "aws"
	ProviderGCP      ProviderType = "gcp"
	ProviderAzure    ProviderType = "azure"
	ProviderOracle   ProviderType = "oracle"
	ProviderAlibaba  ProviderType = "alibaba"
	ProviderIBM      ProviderType = "ibm"
	ProviderOnPremise ProviderType = "onpremise"
)

// VMState represents the state of a VM
type VMState string

const (
	VMStatePending    VMState = "pending"
	VMStateRunning    VMState = "running"
	VMStateStopped    VMState = "stopped"
	VMStateTerminated VMState = "terminated"
	VMStateMigrating  VMState = "migrating"
	VMStateError      VMState = "error"
)

// MigrationType represents the type of migration
type MigrationType string

const (
	MigrationTypeCold   MigrationType = "cold"
	MigrationTypeWarm   MigrationType = "warm"
	MigrationTypeLive   MigrationType = "live"
	MigrationTypeBatch  MigrationType = "batch"
)
