package dwcp

import (
	"time"
)

// CompressionLevel defines compression level constants
type CompressionLevel int

const (
	CompressionLevelNone     CompressionLevel = 0 // No compression
	CompressionLevelFast     CompressionLevel = 1 // Fast compression
	CompressionLevelBalanced CompressionLevel = 2 // Balanced compression
	CompressionLevelMax      CompressionLevel = 3 // Maximum compression
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
	MinStreams          int     `json:"min_streams" yaml:"min_streams"`
	MaxStreams          int     `json:"max_streams" yaml:"max_streams"`
	InitialStreams      int     `json:"initial_streams" yaml:"initial_streams"`
	StreamScalingFactor float64 `json:"stream_scaling_factor" yaml:"stream_scaling_factor"`

	// Congestion control
	CongestionAlgorithm string `json:"congestion_algorithm" yaml:"congestion_algorithm"` // "bbr", "cubic", "reno"
	EnableECN           bool   `json:"enable_ecn" yaml:"enable_ecn"`

	// Buffer sizes
	SendBufferSize int `json:"send_buffer_size" yaml:"send_buffer_size"`
	RecvBufferSize int `json:"recv_buffer_size" yaml:"recv_buffer_size"`

	// Timeouts
	ConnectTimeout time.Duration `json:"connect_timeout" yaml:"connect_timeout"`
	ReadTimeout    time.Duration `json:"read_timeout" yaml:"read_timeout"`
	WriteTimeout   time.Duration `json:"write_timeout" yaml:"write_timeout"`

	// RDMA settings (optional)
	EnableRDMA bool   `json:"enable_rdma" yaml:"enable_rdma"`
	RDMADevice string `json:"rdma_device" yaml:"rdma_device"`
	RDMAPort   int    `json:"rdma_port" yaml:"rdma_port"`

	// Packet pacing
	EnablePacing bool  `json:"enable_pacing" yaml:"enable_pacing"`
	PacingRate   int64 `json:"pacing_rate" yaml:"pacing_rate"` // bytes per second
}

// CompressionConfig configures the compression layer
type CompressionConfig struct {
	// Compression settings
	Enabled   bool             `json:"enabled" yaml:"enabled"`
	Algorithm string           `json:"algorithm" yaml:"algorithm"` // "zstd", "lz4", "snappy"
	Level     CompressionLevel `json:"level" yaml:"level"`

	// Delta encoding settings
	EnableDeltaEncoding bool          `json:"enable_delta_encoding" yaml:"enable_delta_encoding"`
	BaselineInterval    time.Duration `json:"baseline_interval" yaml:"baseline_interval"`
	MaxDeltaChain       int           `json:"max_delta_chain" yaml:"max_delta_chain"`

	// Phase 1: Advanced delta algorithms
	DeltaAlgorithm string `json:"delta_algorithm" yaml:"delta_algorithm"` // "xor", "rsync", "bsdiff", "auto"

	// Phase 1: Dictionary training
	EnableDictionary         bool          `json:"enable_dictionary" yaml:"enable_dictionary"`
	DictionaryUpdateInterval time.Duration `json:"dictionary_update_interval" yaml:"dictionary_update_interval"`

	// Adaptive compression
	EnableAdaptive      bool    `json:"enable_adaptive" yaml:"enable_adaptive"`
	AdaptiveThreshold   float64 `json:"adaptive_threshold" yaml:"adaptive_threshold"` // switch to higher level if ratio < threshold
	MinCompressionRatio float64 `json:"min_compression_ratio" yaml:"min_compression_ratio"`

	// Phase 1: Baseline synchronization
	EnableBaselineSync   bool          `json:"enable_baseline_sync" yaml:"enable_baseline_sync"`
	BaselineSyncInterval time.Duration `json:"baseline_sync_interval" yaml:"baseline_sync_interval"`

	// Model pruning
	EnablePruning   bool          `json:"enable_pruning" yaml:"enable_pruning"`
	PruningInterval time.Duration `json:"pruning_interval" yaml:"pruning_interval"`
}

// PredictionConfig configures the bandwidth prediction engine
type PredictionConfig struct {
	Enabled           bool          `json:"enabled" yaml:"enabled"`
	ModelType         string        `json:"model_type" yaml:"model_type"` // "lstm", "arima", "prophet"
	PredictionHorizon time.Duration `json:"prediction_horizon" yaml:"prediction_horizon"`
	UpdateInterval    time.Duration `json:"update_interval" yaml:"update_interval"`
	HistoryWindow     time.Duration `json:"history_window" yaml:"history_window"`
	ConfidenceLevel   float64       `json:"confidence_level" yaml:"confidence_level"`
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
	Enabled           bool          `json:"enabled" yaml:"enabled"`
	Algorithm         string        `json:"algorithm" yaml:"algorithm"` // "raft", "gossip", "byzantine"
	QuorumSize        int           `json:"quorum_size" yaml:"quorum_size"`
	ElectionTimeout   time.Duration `json:"election_timeout" yaml:"election_timeout"`
	HeartbeatInterval time.Duration `json:"heartbeat_interval" yaml:"heartbeat_interval"`
	AdaptiveMode      bool          `json:"adaptive_mode" yaml:"adaptive_mode"`
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
			Enabled:                  true,
			Algorithm:                "zstd",
			Level:                    CompressionLevelBalanced,
			EnableDeltaEncoding:      true,
			BaselineInterval:         5 * time.Minute,
			MaxDeltaChain:            10,
			DeltaAlgorithm:           "auto", // Phase 1: Auto-select algorithm
			EnableDictionary:         true,   // Phase 1: Dictionary training
			DictionaryUpdateInterval: 24 * time.Hour,
			EnableAdaptive:           true, // Phase 1: Adaptive compression
			AdaptiveThreshold:        15.0, // Phase 1: Target 15x ratio
			MinCompressionRatio:      1.1,
			EnableBaselineSync:       false, // Phase 1: Disabled until cluster configured
			BaselineSyncInterval:     5 * time.Second,
			EnablePruning:            true,
			PruningInterval:          10 * time.Minute,
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

// DeepCopy creates a deep copy of the Config on the heap
// This ensures all nested structures are properly copied and
// the returned pointer references heap-allocated memory
func (c *Config) DeepCopy() *Config {
	if c == nil {
		return nil
	}

	// Allocate new config on heap
	copy := &Config{
		Enabled: c.Enabled,
		Version: c.Version,

		// Deep copy Transport config
		Transport: TransportConfig{
			MinStreams:          c.Transport.MinStreams,
			MaxStreams:          c.Transport.MaxStreams,
			InitialStreams:      c.Transport.InitialStreams,
			StreamScalingFactor: c.Transport.StreamScalingFactor,
			CongestionAlgorithm: c.Transport.CongestionAlgorithm,
			EnableECN:           c.Transport.EnableECN,
			SendBufferSize:      c.Transport.SendBufferSize,
			RecvBufferSize:      c.Transport.RecvBufferSize,
			ConnectTimeout:      c.Transport.ConnectTimeout,
			ReadTimeout:         c.Transport.ReadTimeout,
			WriteTimeout:        c.Transport.WriteTimeout,
			EnableRDMA:          c.Transport.EnableRDMA,
			RDMADevice:          c.Transport.RDMADevice,
			RDMAPort:            c.Transport.RDMAPort,
			EnablePacing:        c.Transport.EnablePacing,
			PacingRate:          c.Transport.PacingRate,
		},

		// Deep copy Compression config
		Compression: CompressionConfig{
			Enabled:                  c.Compression.Enabled,
			Algorithm:                c.Compression.Algorithm,
			Level:                    c.Compression.Level,
			EnableDeltaEncoding:      c.Compression.EnableDeltaEncoding,
			BaselineInterval:         c.Compression.BaselineInterval,
			MaxDeltaChain:            c.Compression.MaxDeltaChain,
			DeltaAlgorithm:           c.Compression.DeltaAlgorithm,
			EnableDictionary:         c.Compression.EnableDictionary,
			DictionaryUpdateInterval: c.Compression.DictionaryUpdateInterval,
			EnableAdaptive:           c.Compression.EnableAdaptive,
			AdaptiveThreshold:        c.Compression.AdaptiveThreshold,
			MinCompressionRatio:      c.Compression.MinCompressionRatio,
			EnableBaselineSync:       c.Compression.EnableBaselineSync,
			BaselineSyncInterval:     c.Compression.BaselineSyncInterval,
			EnablePruning:            c.Compression.EnablePruning,
			PruningInterval:          c.Compression.PruningInterval,
		},

		// Deep copy Prediction config
		Prediction: PredictionConfig{
			Enabled:           c.Prediction.Enabled,
			ModelType:         c.Prediction.ModelType,
			PredictionHorizon: c.Prediction.PredictionHorizon,
			UpdateInterval:    c.Prediction.UpdateInterval,
			HistoryWindow:     c.Prediction.HistoryWindow,
			ConfidenceLevel:   c.Prediction.ConfidenceLevel,
		},

		// Deep copy Sync config
		Sync: SyncConfig{
			Enabled:            c.Sync.Enabled,
			SyncInterval:       c.Sync.SyncInterval,
			MaxStaleness:       c.Sync.MaxStaleness,
			ConflictResolution: c.Sync.ConflictResolution,
			EnableVersioning:   c.Sync.EnableVersioning,
		},

		// Deep copy Consensus config
		Consensus: ConsensusConfig{
			Enabled:           c.Consensus.Enabled,
			Algorithm:         c.Consensus.Algorithm,
			QuorumSize:        c.Consensus.QuorumSize,
			ElectionTimeout:   c.Consensus.ElectionTimeout,
			HeartbeatInterval: c.Consensus.HeartbeatInterval,
			AdaptiveMode:      c.Consensus.AdaptiveMode,
		},
	}

	return copy
}

// Validate validates the DWCP configuration
// This method ALWAYS validates all configuration fields regardless of the Enabled flag
// to prevent invalid configurations from being stored and causing issues later.
func (c *Config) Validate() error {
	if c == nil {
		return &DWCPError{Code: ErrCodeInvalidConfig, Message: "config cannot be nil"}
	}

	// Validate version
	if c.Version == "" {
		return &DWCPError{Code: ErrCodeInvalidConfig, Message: "version cannot be empty"}
	}

	// Validate transport configuration
	if err := c.validateTransport(); err != nil {
		return err
	}

	// Validate compression configuration
	if err := c.validateCompression(); err != nil {
		return err
	}

	// Validate prediction configuration
	if err := c.validatePrediction(); err != nil {
		return err
	}

	// Validate sync configuration
	if err := c.validateSync(); err != nil {
		return err
	}

	// Validate consensus configuration
	if err := c.validateConsensus(); err != nil {
		return err
	}

	return nil
}

// validateTransport validates transport configuration
func (c *Config) validateTransport() error {
	t := &c.Transport

	// Validate stream configuration
	if t.MinStreams < 1 {
		return &DWCPError{Code: ErrCodeInvalidConfig, Message: "transport.min_streams must be >= 1"}
	}
	if t.MinStreams > 1024 {
		return &DWCPError{Code: ErrCodeInvalidConfig, Message: "transport.min_streams must be <= 1024"}
	}
	if t.MaxStreams < t.MinStreams {
		return &DWCPError{Code: ErrCodeInvalidConfig, Message: "transport.max_streams must be >= min_streams"}
	}
	if t.MaxStreams > 4096 {
		return &DWCPError{Code: ErrCodeInvalidConfig, Message: "transport.max_streams must be <= 4096"}
	}
	if t.InitialStreams < t.MinStreams {
		return &DWCPError{Code: ErrCodeInvalidConfig, Message: "transport.initial_streams must be >= min_streams"}
	}
	if t.InitialStreams > t.MaxStreams {
		return &DWCPError{Code: ErrCodeInvalidConfig, Message: "transport.initial_streams must be <= max_streams"}
	}
	if t.StreamScalingFactor <= 1.0 || t.StreamScalingFactor > 10.0 {
		return &DWCPError{Code: ErrCodeInvalidConfig, Message: "transport.stream_scaling_factor must be between 1.0 and 10.0"}
	}

	// Validate congestion algorithm
	validAlgos := map[string]bool{"bbr": true, "cubic": true, "reno": true}
	if !validAlgos[t.CongestionAlgorithm] {
		return &DWCPError{Code: ErrCodeInvalidConfig, Message: "transport.congestion_algorithm must be one of: bbr, cubic, reno"}
	}

	// Validate buffer sizes
	if t.SendBufferSize < 1024 {
		return &DWCPError{Code: ErrCodeInvalidConfig, Message: "transport.send_buffer_size must be >= 1024 bytes"}
	}
	if t.SendBufferSize > 1024*1024*1024 { // 1 GB max
		return &DWCPError{Code: ErrCodeInvalidConfig, Message: "transport.send_buffer_size must be <= 1GB"}
	}
	if t.RecvBufferSize < 1024 {
		return &DWCPError{Code: ErrCodeInvalidConfig, Message: "transport.recv_buffer_size must be >= 1024 bytes"}
	}
	if t.RecvBufferSize > 1024*1024*1024 { // 1 GB max
		return &DWCPError{Code: ErrCodeInvalidConfig, Message: "transport.recv_buffer_size must be <= 1GB"}
	}

	// Validate timeouts
	if t.ConnectTimeout < time.Second {
		return &DWCPError{Code: ErrCodeInvalidConfig, Message: "transport.connect_timeout must be >= 1s"}
	}
	if t.ConnectTimeout > 5*time.Minute {
		return &DWCPError{Code: ErrCodeInvalidConfig, Message: "transport.connect_timeout must be <= 5m"}
	}
	if t.ReadTimeout < time.Second {
		return &DWCPError{Code: ErrCodeInvalidConfig, Message: "transport.read_timeout must be >= 1s"}
	}
	if t.ReadTimeout > 1*time.Hour {
		return &DWCPError{Code: ErrCodeInvalidConfig, Message: "transport.read_timeout must be <= 1h"}
	}
	if t.WriteTimeout < time.Second {
		return &DWCPError{Code: ErrCodeInvalidConfig, Message: "transport.write_timeout must be >= 1s"}
	}
	if t.WriteTimeout > 1*time.Hour {
		return &DWCPError{Code: ErrCodeInvalidConfig, Message: "transport.write_timeout must be <= 1h"}
	}

	// Validate RDMA configuration
	if t.EnableRDMA {
		if t.RDMADevice == "" {
			return &DWCPError{Code: ErrCodeInvalidConfig, Message: "transport.rdma_device must be specified when RDMA is enabled"}
		}
		if t.RDMAPort < 1 || t.RDMAPort > 65535 {
			return &DWCPError{Code: ErrCodeInvalidConfig, Message: "transport.rdma_port must be between 1 and 65535"}
		}
	}

	// Validate pacing configuration
	if t.EnablePacing {
		if t.PacingRate < 1024*1024 { // 1 Mbps min
			return &DWCPError{Code: ErrCodeInvalidConfig, Message: "transport.pacing_rate must be >= 1 Mbps"}
		}
		if t.PacingRate > 100*1024*1024*1024 { // 100 Gbps max
			return &DWCPError{Code: ErrCodeInvalidConfig, Message: "transport.pacing_rate must be <= 100 Gbps"}
		}
	}

	return nil
}

// validateCompression validates compression configuration
func (c *Config) validateCompression() error {
	comp := &c.Compression

	// Validate compression algorithm
	validAlgos := map[string]bool{"zstd": true, "lz4": true, "snappy": true}
	if comp.Algorithm != "" && !validAlgos[comp.Algorithm] {
		return &DWCPError{Code: ErrCodeInvalidConfig, Message: "compression.algorithm must be one of: zstd, lz4, snappy"}
	}

	// Validate compression level
	if comp.Level < CompressionLevelNone || comp.Level > CompressionLevelMax {
		return &DWCPError{Code: ErrCodeInvalidConfig, Message: "compression.level must be between 0 and 3"}
	}

	// Validate delta encoding
	if comp.EnableDeltaEncoding {
		if comp.BaselineInterval < time.Second {
			return &DWCPError{Code: ErrCodeInvalidConfig, Message: "compression.baseline_interval must be >= 1s"}
		}
		if comp.BaselineInterval > 24*time.Hour {
			return &DWCPError{Code: ErrCodeInvalidConfig, Message: "compression.baseline_interval must be <= 24h"}
		}
		if comp.MaxDeltaChain < 1 {
			return &DWCPError{Code: ErrCodeInvalidConfig, Message: "compression.max_delta_chain must be >= 1"}
		}
		if comp.MaxDeltaChain > 1000 {
			return &DWCPError{Code: ErrCodeInvalidConfig, Message: "compression.max_delta_chain must be <= 1000"}
		}
	}

	// Validate delta algorithm
	validDeltaAlgos := map[string]bool{"xor": true, "rsync": true, "bsdiff": true, "auto": true}
	if comp.DeltaAlgorithm != "" && !validDeltaAlgos[comp.DeltaAlgorithm] {
		return &DWCPError{Code: ErrCodeInvalidConfig, Message: "compression.delta_algorithm must be one of: xor, rsync, bsdiff, auto"}
	}

	// Validate dictionary training
	if comp.EnableDictionary {
		if comp.DictionaryUpdateInterval < 1*time.Minute {
			return &DWCPError{Code: ErrCodeInvalidConfig, Message: "compression.dictionary_update_interval must be >= 1m"}
		}
		if comp.DictionaryUpdateInterval > 7*24*time.Hour {
			return &DWCPError{Code: ErrCodeInvalidConfig, Message: "compression.dictionary_update_interval must be <= 7d"}
		}
	}

	// Validate adaptive compression
	if comp.EnableAdaptive {
		if comp.AdaptiveThreshold <= 1.0 {
			return &DWCPError{Code: ErrCodeInvalidConfig, Message: "compression.adaptive_threshold must be > 1.0"}
		}
		if comp.AdaptiveThreshold > 100.0 {
			return &DWCPError{Code: ErrCodeInvalidConfig, Message: "compression.adaptive_threshold must be <= 100.0"}
		}
		if comp.MinCompressionRatio < 1.0 {
			return &DWCPError{Code: ErrCodeInvalidConfig, Message: "compression.min_compression_ratio must be >= 1.0"}
		}
		if comp.MinCompressionRatio > comp.AdaptiveThreshold {
			return &DWCPError{Code: ErrCodeInvalidConfig, Message: "compression.min_compression_ratio must be <= adaptive_threshold"}
		}
	}

	// Validate baseline sync
	if comp.EnableBaselineSync {
		if comp.BaselineSyncInterval < 100*time.Millisecond {
			return &DWCPError{Code: ErrCodeInvalidConfig, Message: "compression.baseline_sync_interval must be >= 100ms"}
		}
		if comp.BaselineSyncInterval > 1*time.Minute {
			return &DWCPError{Code: ErrCodeInvalidConfig, Message: "compression.baseline_sync_interval must be <= 1m"}
		}
	}

	// Validate pruning
	if comp.EnablePruning {
		if comp.PruningInterval < 1*time.Minute {
			return &DWCPError{Code: ErrCodeInvalidConfig, Message: "compression.pruning_interval must be >= 1m"}
		}
		if comp.PruningInterval > 24*time.Hour {
			return &DWCPError{Code: ErrCodeInvalidConfig, Message: "compression.pruning_interval must be <= 24h"}
		}
	}

	return nil
}

// validatePrediction validates prediction configuration
// ALWAYS validates structure regardless of Enabled flag to prevent invalid configs
func (c *Config) validatePrediction() error {
	pred := &c.Prediction

	// Validate model type (always validate structure)
	validModels := map[string]bool{"lstm": true, "arima": true, "prophet": true}
	if pred.ModelType != "" && !validModels[pred.ModelType] {
		return &DWCPError{Code: ErrCodeInvalidConfig, Message: "prediction.model_type must be one of: lstm, arima, prophet"}
	}

	// Validate prediction horizon (always validate structure)
	if pred.PredictionHorizon != 0 && pred.PredictionHorizon < 1*time.Second {
		return &DWCPError{Code: ErrCodeInvalidConfig, Message: "prediction.prediction_horizon must be >= 1s"}
	}
	if pred.PredictionHorizon != 0 && pred.PredictionHorizon > 1*time.Hour {
		return &DWCPError{Code: ErrCodeInvalidConfig, Message: "prediction.prediction_horizon must be <= 1h"}
	}

	// Validate update interval (always validate structure)
	if pred.UpdateInterval != 0 && pred.UpdateInterval < 1*time.Second {
		return &DWCPError{Code: ErrCodeInvalidConfig, Message: "prediction.update_interval must be >= 1s"}
	}
	if pred.UpdateInterval != 0 && pred.PredictionHorizon != 0 && pred.UpdateInterval > pred.PredictionHorizon {
		return &DWCPError{Code: ErrCodeInvalidConfig, Message: "prediction.update_interval must be <= prediction_horizon"}
	}

	// Validate history window (always validate structure)
	if pred.HistoryWindow != 0 && pred.PredictionHorizon != 0 && pred.HistoryWindow < pred.PredictionHorizon {
		return &DWCPError{Code: ErrCodeInvalidConfig, Message: "prediction.history_window must be >= prediction_horizon"}
	}
	if pred.HistoryWindow != 0 && pred.HistoryWindow > 24*time.Hour {
		return &DWCPError{Code: ErrCodeInvalidConfig, Message: "prediction.history_window must be <= 24h"}
	}

	// Validate confidence level (always validate structure)
	if pred.ConfidenceLevel != 0.0 && (pred.ConfidenceLevel <= 0.0 || pred.ConfidenceLevel >= 1.0) {
		return &DWCPError{Code: ErrCodeInvalidConfig, Message: "prediction.confidence_level must be between 0.0 and 1.0"}
	}

	return nil
}

// validateSync validates sync configuration
// ALWAYS validates structure regardless of Enabled flag to prevent invalid configs
func (c *Config) validateSync() error {
	sync := &c.Sync

	// Validate sync interval (always validate structure)
	if sync.SyncInterval != 0 && sync.SyncInterval < 100*time.Millisecond {
		return &DWCPError{Code: ErrCodeInvalidConfig, Message: "sync.sync_interval must be >= 100ms"}
	}
	if sync.SyncInterval != 0 && sync.SyncInterval > 1*time.Minute {
		return &DWCPError{Code: ErrCodeInvalidConfig, Message: "sync.sync_interval must be <= 1m"}
	}

	// Validate max staleness (always validate structure)
	if sync.MaxStaleness != 0 && sync.SyncInterval != 0 && sync.MaxStaleness < sync.SyncInterval {
		return &DWCPError{Code: ErrCodeInvalidConfig, Message: "sync.max_staleness must be >= sync_interval"}
	}
	if sync.MaxStaleness != 0 && sync.MaxStaleness > 5*time.Minute {
		return &DWCPError{Code: ErrCodeInvalidConfig, Message: "sync.max_staleness must be <= 5m"}
	}

	// Validate conflict resolution (always validate structure)
	validResolution := map[string]bool{"lww": true, "mvcc": true, "crdt": true}
	if sync.ConflictResolution != "" && !validResolution[sync.ConflictResolution] {
		return &DWCPError{Code: ErrCodeInvalidConfig, Message: "sync.conflict_resolution must be one of: lww, mvcc, crdt"}
	}

	return nil
}

// validateConsensus validates consensus configuration
// ALWAYS validates structure regardless of Enabled flag to prevent invalid configs
func (c *Config) validateConsensus() error {
	cons := &c.Consensus

	// Validate algorithm (always validate structure)
	validAlgos := map[string]bool{"raft": true, "gossip": true, "byzantine": true}
	if cons.Algorithm != "" && !validAlgos[cons.Algorithm] {
		return &DWCPError{Code: ErrCodeInvalidConfig, Message: "consensus.algorithm must be one of: raft, gossip, byzantine"}
	}

	// Validate quorum size (always validate structure)
	if cons.QuorumSize != 0 && cons.QuorumSize < 1 {
		return &DWCPError{Code: ErrCodeInvalidConfig, Message: "consensus.quorum_size must be >= 1"}
	}
	if cons.QuorumSize > 1000 {
		return &DWCPError{Code: ErrCodeInvalidConfig, Message: "consensus.quorum_size must be <= 1000"}
	}

	// Validate election timeout (always validate structure)
	if cons.ElectionTimeout != 0 && cons.ElectionTimeout < 10*time.Millisecond {
		return &DWCPError{Code: ErrCodeInvalidConfig, Message: "consensus.election_timeout must be >= 10ms"}
	}
	if cons.ElectionTimeout != 0 && cons.ElectionTimeout > 10*time.Second {
		return &DWCPError{Code: ErrCodeInvalidConfig, Message: "consensus.election_timeout must be <= 10s"}
	}

	// Validate heartbeat interval (always validate structure)
	if cons.HeartbeatInterval != 0 && cons.HeartbeatInterval < 1*time.Millisecond {
		return &DWCPError{Code: ErrCodeInvalidConfig, Message: "consensus.heartbeat_interval must be >= 1ms"}
	}
	if cons.HeartbeatInterval != 0 && cons.ElectionTimeout != 0 && cons.HeartbeatInterval >= cons.ElectionTimeout {
		return &DWCPError{Code: ErrCodeInvalidConfig, Message: "consensus.heartbeat_interval must be < election_timeout"}
	}

	return nil
}
