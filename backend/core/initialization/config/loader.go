// Package config provides configuration loading and validation for NovaCron initialization
package config

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"gopkg.in/yaml.v3"
)

// Config represents the complete NovaCron configuration
type Config struct {
	// Core system configuration
	System SystemConfig `yaml:"system" json:"system"`

	// DWCP configuration
	DWCP DWCPConfig `yaml:"dwcp" json:"dwcp"`

	// Network configuration
	Network NetworkConfig `yaml:"network" json:"network"`

	// Storage configuration
	Storage StorageConfig `yaml:"storage" json:"storage"`

	// Monitoring configuration
	Monitoring MonitoringConfig `yaml:"monitoring" json:"monitoring"`

	// Security configuration
	Security SecurityConfig `yaml:"security" json:"security"`
}

// SystemConfig defines core system settings
type SystemConfig struct {
	NodeID          string        `yaml:"node_id" json:"node_id"`
	DataDir         string        `yaml:"data_dir" json:"data_dir"`
	LogLevel        string        `yaml:"log_level" json:"log_level"`
	MaxConcurrency  int           `yaml:"max_concurrency" json:"max_concurrency"`
	HealthCheckPort int           `yaml:"health_check_port" json:"health_check_port"`
	ShutdownTimeout time.Duration `yaml:"shutdown_timeout" json:"shutdown_timeout"`
}

// DWCPConfig defines DWCP protocol settings
type DWCPConfig struct {
	// Mode detection
	EnableAutoDetection bool          `yaml:"enable_auto_detection" json:"enable_auto_detection"`
	ForcedMode          string        `yaml:"forced_mode" json:"forced_mode"` // "datacenter", "internet", "hybrid"
	DetectionInterval   time.Duration `yaml:"detection_interval" json:"detection_interval"`

	// Feature flags
	EnableV3Transport   bool `yaml:"enable_v3_transport" json:"enable_v3_transport"`
	EnableV3Compression bool `yaml:"enable_v3_compression" json:"enable_v3_compression"`
	EnableV3Prediction  bool `yaml:"enable_v3_prediction" json:"enable_v3_prediction"`
	EnableV3StateSync   bool `yaml:"enable_v3_state_sync" json:"enable_v3_state_sync"`
	EnableV3Consensus   bool `yaml:"enable_v3_consensus" json:"enable_v3_consensus"`
	EnableV3Placement   bool `yaml:"enable_v3_placement" json:"enable_v3_placement"`
	V3RolloutPercentage int  `yaml:"v3_rollout_percentage" json:"v3_rollout_percentage"`

	// Transport settings
	Transport TransportConfig `yaml:"transport" json:"transport"`

	// Compression settings
	Compression CompressionConfig `yaml:"compression" json:"compression"`

	// Consensus settings
	Consensus ConsensusConfig `yaml:"consensus" json:"consensus"`
}

// TransportConfig defines transport layer settings
type TransportConfig struct {
	// Multi-stream settings
	MinStreams      int           `yaml:"min_streams" json:"min_streams"`
	MaxStreams      int           `yaml:"max_streams" json:"max_streams"`
	StreamTimeout   time.Duration `yaml:"stream_timeout" json:"stream_timeout"`

	// RDMA settings (datacenter mode)
	EnableRDMA      bool   `yaml:"enable_rdma" json:"enable_rdma"`
	RDMADevice      string `yaml:"rdma_device" json:"rdma_device"`

	// Congestion control
	CongestionAlgo  string `yaml:"congestion_algo" json:"congestion_algo"`
}

// CompressionConfig defines compression settings
type CompressionConfig struct {
	Algorithm       string `yaml:"algorithm" json:"algorithm"` // "zstd", "lz4", "none"
	Level           int    `yaml:"level" json:"level"`         // 0-22 for zstd
	EnableDelta     bool   `yaml:"enable_delta" json:"enable_delta"`
	EnableMLModel   bool   `yaml:"enable_ml_model" json:"enable_ml_model"`
}

// ConsensusConfig defines consensus protocol settings
type ConsensusConfig struct {
	Protocol        string        `yaml:"protocol" json:"protocol"` // "raft", "pbft", "gossip"
	HeartbeatInterval time.Duration `yaml:"heartbeat_interval" json:"heartbeat_interval"`
	ElectionTimeout time.Duration `yaml:"election_timeout" json:"election_timeout"`
	ByzantineTolerance bool       `yaml:"byzantine_tolerance" json:"byzantine_tolerance"`
}

// NetworkConfig defines network settings
type NetworkConfig struct {
	ListenAddress   string        `yaml:"listen_address" json:"listen_address"`
	AdvertiseAddress string       `yaml:"advertise_address" json:"advertise_address"`
	BindPort        int           `yaml:"bind_port" json:"bind_port"`
	MaxConnections  int           `yaml:"max_connections" json:"max_connections"`
	ConnTimeout     time.Duration `yaml:"conn_timeout" json:"conn_timeout"`
	EnableTLS       bool          `yaml:"enable_tls" json:"enable_tls"`
	TLSCertPath     string        `yaml:"tls_cert_path" json:"tls_cert_path"`
	TLSKeyPath      string        `yaml:"tls_key_path" json:"tls_key_path"`
}

// StorageConfig defines storage settings
type StorageConfig struct {
	Backend         string `yaml:"backend" json:"backend"` // "sqlite", "postgres", "redis"
	ConnectionString string `yaml:"connection_string" json:"connection_string"`
	MaxConnections  int    `yaml:"max_connections" json:"max_connections"`
	EnableCache     bool   `yaml:"enable_cache" json:"enable_cache"`
	CacheTTL        time.Duration `yaml:"cache_ttl" json:"cache_ttl"`
}

// MonitoringConfig defines monitoring settings
type MonitoringConfig struct {
	EnableMetrics     bool   `yaml:"enable_metrics" json:"enable_metrics"`
	MetricsPort       int    `yaml:"metrics_port" json:"metrics_port"`
	EnableTracing     bool   `yaml:"enable_tracing" json:"enable_tracing"`
	TracingEndpoint   string `yaml:"tracing_endpoint" json:"tracing_endpoint"`
	EnableProfiling   bool   `yaml:"enable_profiling" json:"enable_profiling"`
	ProfilingPort     int    `yaml:"profiling_port" json:"profiling_port"`
}

// SecurityConfig defines security settings
type SecurityConfig struct {
	EnableAuth      bool     `yaml:"enable_auth" json:"enable_auth"`
	AuthMethod      string   `yaml:"auth_method" json:"auth_method"` // "mtls", "token", "none"
	TrustedNodes    []string `yaml:"trusted_nodes" json:"trusted_nodes"`
	EnableEncryption bool    `yaml:"enable_encryption" json:"enable_encryption"`
	EncryptionAlgo  string   `yaml:"encryption_algo" json:"encryption_algo"`
}

// Loader handles configuration loading from various sources
type Loader struct {
	configPath string
	config     *Config
}

// NewLoader creates a new configuration loader
func NewLoader(configPath string) *Loader {
	return &Loader{
		configPath: configPath,
	}
}

// Load loads configuration from file
func (l *Loader) Load() (*Config, error) {
	if l.configPath == "" {
		return nil, fmt.Errorf("config path not specified")
	}

	// Check if file exists
	if _, err := os.Stat(l.configPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("config file not found: %s", l.configPath)
	}

	// Read file
	data, err := os.ReadFile(l.configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	// Parse based on extension
	ext := filepath.Ext(l.configPath)
	var config Config

	switch ext {
	case ".yaml", ".yml":
		if err := yaml.Unmarshal(data, &config); err != nil {
			return nil, fmt.Errorf("failed to parse YAML config: %w", err)
		}
	case ".json":
		if err := json.Unmarshal(data, &config); err != nil {
			return nil, fmt.Errorf("failed to parse JSON config: %w", err)
		}
	default:
		return nil, fmt.Errorf("unsupported config format: %s (use .yaml or .json)", ext)
	}

	// Apply defaults
	if err := l.applyDefaults(&config); err != nil {
		return nil, fmt.Errorf("failed to apply defaults: %w", err)
	}

	// Validate configuration
	if err := l.validate(&config); err != nil {
		return nil, fmt.Errorf("config validation failed: %w", err)
	}

	l.config = &config
	return &config, nil
}

// applyDefaults applies default values to configuration
func (l *Loader) applyDefaults(config *Config) error {
	// System defaults
	if config.System.LogLevel == "" {
		config.System.LogLevel = "info"
	}
	if config.System.MaxConcurrency == 0 {
		config.System.MaxConcurrency = 1000
	}
	if config.System.HealthCheckPort == 0 {
		config.System.HealthCheckPort = 8080
	}
	if config.System.ShutdownTimeout == 0 {
		config.System.ShutdownTimeout = 30 * time.Second
	}

	// DWCP defaults
	if config.DWCP.DetectionInterval == 0 {
		config.DWCP.DetectionInterval = 30 * time.Second
	}
	if config.DWCP.V3RolloutPercentage < 0 || config.DWCP.V3RolloutPercentage > 100 {
		config.DWCP.V3RolloutPercentage = 0 // Default to disabled
	}

	// Transport defaults
	if config.DWCP.Transport.MinStreams == 0 {
		config.DWCP.Transport.MinStreams = 4
	}
	if config.DWCP.Transport.MaxStreams == 0 {
		config.DWCP.Transport.MaxStreams = 256
	}
	if config.DWCP.Transport.StreamTimeout == 0 {
		config.DWCP.Transport.StreamTimeout = 60 * time.Second
	}
	if config.DWCP.Transport.CongestionAlgo == "" {
		config.DWCP.Transport.CongestionAlgo = "cubic"
	}

	// Compression defaults
	if config.DWCP.Compression.Algorithm == "" {
		config.DWCP.Compression.Algorithm = "zstd"
	}
	if config.DWCP.Compression.Level == 0 {
		config.DWCP.Compression.Level = 3
	}

	// Consensus defaults
	if config.DWCP.Consensus.Protocol == "" {
		config.DWCP.Consensus.Protocol = "raft"
	}
	if config.DWCP.Consensus.HeartbeatInterval == 0 {
		config.DWCP.Consensus.HeartbeatInterval = 500 * time.Millisecond
	}
	if config.DWCP.Consensus.ElectionTimeout == 0 {
		config.DWCP.Consensus.ElectionTimeout = 5 * time.Second
	}

	// Network defaults
	if config.Network.ListenAddress == "" {
		config.Network.ListenAddress = "0.0.0.0"
	}
	if config.Network.BindPort == 0 {
		config.Network.BindPort = 9090
	}
	if config.Network.MaxConnections == 0 {
		config.Network.MaxConnections = 10000
	}
	if config.Network.ConnTimeout == 0 {
		config.Network.ConnTimeout = 30 * time.Second
	}

	// Storage defaults
	if config.Storage.Backend == "" {
		config.Storage.Backend = "sqlite"
	}
	if config.Storage.MaxConnections == 0 {
		config.Storage.MaxConnections = 100
	}
	if config.Storage.CacheTTL == 0 {
		config.Storage.CacheTTL = 5 * time.Minute
	}

	// Monitoring defaults
	if config.Monitoring.MetricsPort == 0 {
		config.Monitoring.MetricsPort = 9091
	}
	if config.Monitoring.ProfilingPort == 0 {
		config.Monitoring.ProfilingPort = 6060
	}

	// Security defaults
	if config.Security.AuthMethod == "" {
		config.Security.AuthMethod = "mtls"
	}
	if config.Security.EncryptionAlgo == "" {
		config.Security.EncryptionAlgo = "aes-256-gcm"
	}

	return nil
}

// validate validates the configuration
func (l *Loader) validate(config *Config) error {
	// System validation
	if config.System.NodeID == "" {
		return fmt.Errorf("system.node_id is required")
	}
	if config.System.DataDir == "" {
		return fmt.Errorf("system.data_dir is required")
	}

	validLogLevels := map[string]bool{"debug": true, "info": true, "warn": true, "error": true}
	if !validLogLevels[config.System.LogLevel] {
		return fmt.Errorf("invalid log_level: %s (must be debug, info, warn, or error)", config.System.LogLevel)
	}

	// DWCP validation
	if config.DWCP.ForcedMode != "" {
		validModes := map[string]bool{"datacenter": true, "internet": true, "hybrid": true}
		if !validModes[config.DWCP.ForcedMode] {
			return fmt.Errorf("invalid forced_mode: %s", config.DWCP.ForcedMode)
		}
	}

	if config.DWCP.V3RolloutPercentage < 0 || config.DWCP.V3RolloutPercentage > 100 {
		return fmt.Errorf("v3_rollout_percentage must be 0-100, got %d", config.DWCP.V3RolloutPercentage)
	}

	// Transport validation
	if config.DWCP.Transport.MinStreams > config.DWCP.Transport.MaxStreams {
		return fmt.Errorf("transport.min_streams (%d) cannot exceed max_streams (%d)",
			config.DWCP.Transport.MinStreams, config.DWCP.Transport.MaxStreams)
	}

	// Network validation
	if config.Network.BindPort < 1024 || config.Network.BindPort > 65535 {
		return fmt.Errorf("invalid bind_port: %d (must be 1024-65535)", config.Network.BindPort)
	}

	if config.Network.EnableTLS {
		if config.Network.TLSCertPath == "" || config.Network.TLSKeyPath == "" {
			return fmt.Errorf("tls_cert_path and tls_key_path required when enable_tls is true")
		}
	}

	// Storage validation
	validBackends := map[string]bool{"sqlite": true, "postgres": true, "redis": true}
	if !validBackends[config.Storage.Backend] {
		return fmt.Errorf("invalid storage backend: %s", config.Storage.Backend)
	}

	if config.Storage.ConnectionString == "" && config.Storage.Backend != "sqlite" {
		return fmt.Errorf("connection_string required for backend: %s", config.Storage.Backend)
	}

	return nil
}

// GetConfig returns the loaded configuration
func (l *Loader) GetConfig() *Config {
	return l.config
}

// LoadFromEnv loads configuration from environment variables (overrides)
func (l *Loader) LoadFromEnv(config *Config) error {
	// System overrides
	if nodeID := os.Getenv("NOVACRON_NODE_ID"); nodeID != "" {
		config.System.NodeID = nodeID
	}
	if dataDir := os.Getenv("NOVACRON_DATA_DIR"); dataDir != "" {
		config.System.DataDir = dataDir
	}
	if logLevel := os.Getenv("NOVACRON_LOG_LEVEL"); logLevel != "" {
		config.System.LogLevel = logLevel
	}

	// DWCP overrides
	if forcedMode := os.Getenv("NOVACRON_DWCP_MODE"); forcedMode != "" {
		config.DWCP.ForcedMode = forcedMode
	}

	// Network overrides
	if listenAddr := os.Getenv("NOVACRON_LISTEN_ADDRESS"); listenAddr != "" {
		config.Network.ListenAddress = listenAddr
	}

	return nil
}

// GenerateDefault generates a default configuration file
func GenerateDefault(path string) error {
	config := Config{
		System: SystemConfig{
			NodeID:          "novacron-node-1",
			DataDir:         "/var/lib/novacron",
			LogLevel:        "info",
			MaxConcurrency:  1000,
			HealthCheckPort: 8080,
			ShutdownTimeout: 30 * time.Second,
		},
		DWCP: DWCPConfig{
			EnableAutoDetection: true,
			ForcedMode:          "",
			DetectionInterval:   30 * time.Second,
			EnableV3Transport:   false,
			EnableV3Compression: false,
			EnableV3Prediction:  false,
			EnableV3StateSync:   false,
			EnableV3Consensus:   false,
			EnableV3Placement:   false,
			V3RolloutPercentage: 0,
			Transport: TransportConfig{
				MinStreams:     4,
				MaxStreams:     256,
				StreamTimeout:  60 * time.Second,
				EnableRDMA:     false,
				CongestionAlgo: "cubic",
			},
			Compression: CompressionConfig{
				Algorithm:    "zstd",
				Level:        3,
				EnableDelta:  true,
				EnableMLModel: false,
			},
			Consensus: ConsensusConfig{
				Protocol:           "raft",
				HeartbeatInterval:  500 * time.Millisecond,
				ElectionTimeout:    5 * time.Second,
				ByzantineTolerance: false,
			},
		},
		Network: NetworkConfig{
			ListenAddress:  "0.0.0.0",
			BindPort:       9090,
			MaxConnections: 10000,
			ConnTimeout:    30 * time.Second,
			EnableTLS:      true,
		},
		Storage: StorageConfig{
			Backend:        "sqlite",
			MaxConnections: 100,
			EnableCache:    true,
			CacheTTL:       5 * time.Minute,
		},
		Monitoring: MonitoringConfig{
			EnableMetrics:   true,
			MetricsPort:     9091,
			EnableTracing:   false,
			EnableProfiling: false,
		},
		Security: SecurityConfig{
			EnableAuth:       true,
			AuthMethod:       "mtls",
			EnableEncryption: true,
			EncryptionAlgo:   "aes-256-gcm",
		},
	}

	data, err := yaml.Marshal(&config)
	if err != nil {
		return fmt.Errorf("failed to marshal config: %w", err)
	}

	if err := os.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("failed to write config file: %w", err)
	}

	return nil
}
