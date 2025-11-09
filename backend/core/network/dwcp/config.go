package dwcp

import (
	"time"
)

// Config contains all DWCP configuration parameters
type Config struct {
	// General settings
	Enabled bool   `json:"enabled" yaml:"enabled"`
	Version string `json:"version" yaml:"version"`

	// Transport configuration
	Transport TransportConfig `json:"transport" yaml:"transport"`

	// Compression configuration
	Compression CompressionConfig `json:"compression" yaml:"compression"`

	// Prediction configuration
	Prediction PredictionConfig `json:"prediction" yaml:"prediction"`

	// Synchronization configuration
	Sync SyncConfig `json:"sync" yaml:"sync"`

	// Consensus configuration
	Consensus ConsensusConfig `json:"consensus" yaml:"consensus"`
}

// TransportConfig configures the multi-stream transport layer
type TransportConfig struct {
	// Multi-stream TCP settings
	MinStreams          int           `json:"min_streams" yaml:"min_streams"`
	MaxStreams          int           `json:"max_streams" yaml:"max_streams"`
	InitialStreams      int           `json:"initial_streams" yaml:"initial_streams"`
	StreamScalingFactor float64       `json:"stream_scaling_factor" yaml:"stream_scaling_factor"`

	// Congestion control
	CongestionAlgorithm string        `json:"congestion_algorithm" yaml:"congestion_algorithm"` // "bbr", "cubic", "reno"
	EnableECN           bool          `json:"enable_ecn" yaml:"enable_ecn"`

	// Buffer sizes
	SendBufferSize      int           `json:"send_buffer_size" yaml:"send_buffer_size"`
	RecvBufferSize      int           `json:"recv_buffer_size" yaml:"recv_buffer_size"`

	// Timeouts
	ConnectTimeout      time.Duration `json:"connect_timeout" yaml:"connect_timeout"`
	ReadTimeout         time.Duration `json:"read_timeout" yaml:"read_timeout"`
	WriteTimeout        time.Duration `json:"write_timeout" yaml:"write_timeout"`

	// RDMA settings (optional)
	EnableRDMA          bool          `json:"enable_rdma" yaml:"enable_rdma"`
	RDMADevice          string        `json:"rdma_device" yaml:"rdma_device"`
	RDMAPort            int           `json:"rdma_port" yaml:"rdma_port"`

	// Packet pacing
	EnablePacing        bool          `json:"enable_pacing" yaml:"enable_pacing"`
	PacingRate          int64         `json:"pacing_rate" yaml:"pacing_rate"` // bytes per second
}

// CompressionConfig configures the compression layer
type CompressionConfig struct {
	// Compression settings
	Enabled             bool             `json:"enabled" yaml:"enabled"`
	Algorithm           string           `json:"algorithm" yaml:"algorithm"` // "zstd", "lz4", "snappy"
	Level               CompressionLevel `json:"level" yaml:"level"`

	// Delta encoding settings
	EnableDeltaEncoding bool             `json:"enable_delta_encoding" yaml:"enable_delta_encoding"`
	BaselineInterval    time.Duration    `json:"baseline_interval" yaml:"baseline_interval"`
	MaxDeltaChain       int              `json:"max_delta_chain" yaml:"max_delta_chain"`

	// Phase 1: Advanced delta algorithms
	DeltaAlgorithm      string           `json:"delta_algorithm" yaml:"delta_algorithm"` // "xor", "rsync", "bsdiff", "auto"

	// Phase 1: Dictionary training
	EnableDictionary    bool             `json:"enable_dictionary" yaml:"enable_dictionary"`
	DictionaryUpdateInterval time.Duration `json:"dictionary_update_interval" yaml:"dictionary_update_interval"`

	// Adaptive compression
	EnableAdaptive      bool             `json:"enable_adaptive" yaml:"enable_adaptive"`
	AdaptiveThreshold   float64          `json:"adaptive_threshold" yaml:"adaptive_threshold"` // switch to higher level if ratio < threshold
	MinCompressionRatio float64          `json:"min_compression_ratio" yaml:"min_compression_ratio"`

	// Phase 1: Baseline synchronization
	EnableBaselineSync  bool             `json:"enable_baseline_sync" yaml:"enable_baseline_sync"`
	BaselineSyncInterval time.Duration   `json:"baseline_sync_interval" yaml:"baseline_sync_interval"`

	// Model pruning
	EnablePruning       bool             `json:"enable_pruning" yaml:"enable_pruning"`
	PruningInterval     time.Duration    `json:"pruning_interval" yaml:"pruning_interval"`
}

// PredictionConfig configures the bandwidth prediction engine
type PredictionConfig struct {
	Enabled            bool          `json:"enabled" yaml:"enabled"`
	ModelType          string        `json:"model_type" yaml:"model_type"` // "lstm", "arima", "prophet"
	PredictionHorizon  time.Duration `json:"prediction_horizon" yaml:"prediction_horizon"`
	UpdateInterval     time.Duration `json:"update_interval" yaml:"update_interval"`
	HistoryWindow      time.Duration `json:"history_window" yaml:"history_window"`
	ConfidenceLevel    float64       `json:"confidence_level" yaml:"confidence_level"`
}

// SyncConfig configures the state synchronization layer
type SyncConfig struct {
	Enabled            bool          `json:"enabled" yaml:"enabled"`
	SyncInterval       time.Duration `json:"sync_interval" yaml:"sync_interval"`
	MaxStaleness       time.Duration `json:"max_staleness" yaml:"max_staleness"`
	ConflictResolution string        `json:"conflict_resolution" yaml:"conflict_resolution"` // "lww", "mvcc", "crdt"
	EnableVersioning   bool          `json:"enable_versioning" yaml:"enable_versioning"`
}

// ConsensusConfig configures the adaptive consensus protocol
type ConsensusConfig struct {
	Enabled        bool          `json:"enabled" yaml:"enabled"`
	Algorithm      string        `json:"algorithm" yaml:"algorithm"` // "raft", "gossip", "byzantine"
	QuorumSize     int           `json:"quorum_size" yaml:"quorum_size"`
	ElectionTimeout time.Duration `json:"election_timeout" yaml:"election_timeout"`
	HeartbeatInterval time.Duration `json:"heartbeat_interval" yaml:"heartbeat_interval"`
	AdaptiveMode   bool          `json:"adaptive_mode" yaml:"adaptive_mode"`
}

// DefaultConfig returns a sensible default DWCP configuration
func DefaultConfig() *Config {
	return &Config{
		Enabled: false, // Disabled by default for backward compatibility
		Version: DWCPVersion,

		Transport: TransportConfig{
			MinStreams:          16,
			MaxStreams:          256,
			InitialStreams:      32,
			StreamScalingFactor: 1.5,
			CongestionAlgorithm: "bbr",
			EnableECN:           true,
			SendBufferSize:      16 * 1024 * 1024, // 16 MB
			RecvBufferSize:      16 * 1024 * 1024, // 16 MB
			ConnectTimeout:      30 * time.Second,
			ReadTimeout:         60 * time.Second,
			WriteTimeout:        60 * time.Second,
			EnableRDMA:          false,
			EnablePacing:        true,
			PacingRate:          1000 * 1024 * 1024, // 1 Gbps
		},

		Compression: CompressionConfig{
			Enabled:             true,
			Algorithm:           "zstd",
			Level:               CompressionLevelBalanced,
			EnableDeltaEncoding: true,
			BaselineInterval:    5 * time.Minute,
			MaxDeltaChain:       10,
			DeltaAlgorithm:      "auto", // Phase 1: Auto-select algorithm
			EnableDictionary:    true,   // Phase 1: Dictionary training
			DictionaryUpdateInterval: 24 * time.Hour,
			EnableAdaptive:      true,   // Phase 1: Adaptive compression
			AdaptiveThreshold:   15.0,   // Phase 1: Target 15x ratio
			MinCompressionRatio: 1.1,
			EnableBaselineSync:  false,  // Phase 1: Disabled until cluster configured
			BaselineSyncInterval: 5 * time.Second,
			EnablePruning:       true,
			PruningInterval:     10 * time.Minute,
		},

		Prediction: PredictionConfig{
			Enabled:           false, // Phase 2
			ModelType:         "lstm",
			PredictionHorizon: 5 * time.Minute,
			UpdateInterval:    30 * time.Second,
			HistoryWindow:     1 * time.Hour,
			ConfidenceLevel:   0.95,
		},

		Sync: SyncConfig{
			Enabled:            false, // Phase 3
			SyncInterval:       1 * time.Second,
			MaxStaleness:       5 * time.Second,
			ConflictResolution: "lww",
			EnableVersioning:   true,
		},

		Consensus: ConsensusConfig{
			Enabled:           false, // Phase 3
			Algorithm:         "raft",
			QuorumSize:        3,
			ElectionTimeout:   150 * time.Millisecond,
			HeartbeatInterval: 50 * time.Millisecond,
			AdaptiveMode:      false,
		},
	}
}

// ValidateConfig validates the DWCP configuration
func (c *Config) Validate() error {
	if !c.Enabled {
		return nil // Skip validation if disabled
	}

	// Validate transport configuration
	if c.Transport.MinStreams < 1 {
		return &DWCPError{Code: ErrCodeInvalidConfig, Message: "min_streams must be >= 1"}
	}
	if c.Transport.MaxStreams < c.Transport.MinStreams {
		return &DWCPError{Code: ErrCodeInvalidConfig, Message: "max_streams must be >= min_streams"}
	}
	if c.Transport.InitialStreams < c.Transport.MinStreams || c.Transport.InitialStreams > c.Transport.MaxStreams {
		return &DWCPError{Code: ErrCodeInvalidConfig, Message: "initial_streams must be between min_streams and max_streams"}
	}

	// Validate compression configuration
	if c.Compression.MaxDeltaChain < 1 {
		return &DWCPError{Code: ErrCodeInvalidConfig, Message: "max_delta_chain must be >= 1"}
	}

	return nil
}
