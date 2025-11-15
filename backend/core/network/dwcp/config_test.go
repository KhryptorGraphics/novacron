package dwcp

import (
	"testing"
	"time"
)

// TestConfigDeepCopy verifies that DeepCopy creates a proper heap-allocated copy
func TestConfigDeepCopy(t *testing.T) {
	original := DefaultConfig()

	// Modify original values to ensure they're different from defaults
	original.Enabled = true
	original.Version = "test-version"
	original.Transport.MinStreams = 100
	original.Transport.CongestionAlgorithm = "cubic"
	original.Compression.Algorithm = "lz4"
	original.Compression.Level = CompressionLevelFast
	original.Prediction.ModelType = "arima"
	original.Sync.ConflictResolution = "mvcc"
	original.Consensus.Algorithm = "gossip"

	// Create deep copy
	copy := original.DeepCopy()

	// Verify copy is not nil
	if copy == nil {
		t.Fatal("DeepCopy returned nil")
	}

	// Verify copy has same values
	if copy.Enabled != original.Enabled {
		t.Errorf("Enabled mismatch: got %v, want %v", copy.Enabled, original.Enabled)
	}
	if copy.Version != original.Version {
		t.Errorf("Version mismatch: got %v, want %v", copy.Version, original.Version)
	}
	if copy.Transport.MinStreams != original.Transport.MinStreams {
		t.Errorf("Transport.MinStreams mismatch: got %v, want %v", copy.Transport.MinStreams, original.Transport.MinStreams)
	}
	if copy.Transport.CongestionAlgorithm != original.Transport.CongestionAlgorithm {
		t.Errorf("Transport.CongestionAlgorithm mismatch: got %v, want %v", copy.Transport.CongestionAlgorithm, original.Transport.CongestionAlgorithm)
	}
	if copy.Compression.Algorithm != original.Compression.Algorithm {
		t.Errorf("Compression.Algorithm mismatch: got %v, want %v", copy.Compression.Algorithm, original.Compression.Algorithm)
	}
	if copy.Compression.Level != original.Compression.Level {
		t.Errorf("Compression.Level mismatch: got %v, want %v", copy.Compression.Level, original.Compression.Level)
	}
	if copy.Prediction.ModelType != original.Prediction.ModelType {
		t.Errorf("Prediction.ModelType mismatch: got %v, want %v", copy.Prediction.ModelType, original.Prediction.ModelType)
	}
	if copy.Sync.ConflictResolution != original.Sync.ConflictResolution {
		t.Errorf("Sync.ConflictResolution mismatch: got %v, want %v", copy.Sync.ConflictResolution, original.Sync.ConflictResolution)
	}
	if copy.Consensus.Algorithm != original.Consensus.Algorithm {
		t.Errorf("Consensus.Algorithm mismatch: got %v, want %v", copy.Consensus.Algorithm, original.Consensus.Algorithm)
	}

	// Verify modifications to copy don't affect original (true independence)
	copy.Enabled = false
	copy.Version = "modified-version"
	copy.Transport.MinStreams = 200
	copy.Transport.CongestionAlgorithm = "reno"
	copy.Compression.Algorithm = "snappy"
	copy.Compression.Level = CompressionLevelMax
	copy.Prediction.ModelType = "prophet"
	copy.Sync.ConflictResolution = "crdt"
	copy.Consensus.Algorithm = "byzantine"

	// Original should remain unchanged
	if original.Enabled != true {
		t.Error("Modifying copy affected original.Enabled")
	}
	if original.Version != "test-version" {
		t.Error("Modifying copy affected original.Version")
	}
	if original.Transport.MinStreams != 100 {
		t.Error("Modifying copy affected original.Transport.MinStreams")
	}
	if original.Transport.CongestionAlgorithm != "cubic" {
		t.Error("Modifying copy affected original.Transport.CongestionAlgorithm")
	}
	if original.Compression.Algorithm != "lz4" {
		t.Error("Modifying copy affected original.Compression.Algorithm")
	}
	if original.Compression.Level != CompressionLevelFast {
		t.Error("Modifying copy affected original.Compression.Level")
	}
	if original.Prediction.ModelType != "arima" {
		t.Error("Modifying copy affected original.Prediction.ModelType")
	}
	if original.Sync.ConflictResolution != "mvcc" {
		t.Error("Modifying copy affected original.Sync.ConflictResolution")
	}
	if original.Consensus.Algorithm != "gossip" {
		t.Error("Modifying copy affected original.Consensus.Algorithm")
	}
}

// TestConfigDeepCopyNil verifies DeepCopy handles nil gracefully
func TestConfigDeepCopyNil(t *testing.T) {
	var nilConfig *Config
	copy := nilConfig.DeepCopy()
	if copy != nil {
		t.Error("DeepCopy of nil should return nil")
	}
}

// TestConfigDeepCopyAllFields verifies all fields are copied
func TestConfigDeepCopyAllFields(t *testing.T) {
	original := &Config{
		Enabled: true,
		Version: "1.2.3",
		Transport: TransportConfig{
			MinStreams:          10,
			MaxStreams:          100,
			InitialStreams:      20,
			StreamScalingFactor: 2.0,
			CongestionAlgorithm: "bbr",
			EnableECN:           true,
			SendBufferSize:      1024,
			RecvBufferSize:      2048,
			ConnectTimeout:      10 * time.Second,
			ReadTimeout:         20 * time.Second,
			WriteTimeout:        30 * time.Second,
			EnableRDMA:          true,
			RDMADevice:          "mlx5_0",
			RDMAPort:            1,
			EnablePacing:        true,
			PacingRate:          1000000,
		},
		Compression: CompressionConfig{
			Enabled:                  true,
			Algorithm:                "zstd",
			Level:                    CompressionLevelMax,
			EnableDeltaEncoding:      true,
			BaselineInterval:         5 * time.Minute,
			MaxDeltaChain:            15,
			DeltaAlgorithm:           "xor",
			EnableDictionary:         true,
			DictionaryUpdateInterval: 1 * time.Hour,
			EnableAdaptive:           true,
			AdaptiveThreshold:        10.0,
			MinCompressionRatio:      2.0,
			EnableBaselineSync:       true,
			BaselineSyncInterval:     10 * time.Second,
			EnablePruning:            true,
			PruningInterval:          15 * time.Minute,
		},
		Prediction: PredictionConfig{
			Enabled:           true,
			ModelType:         "lstm",
			PredictionHorizon: 10 * time.Minute,
			UpdateInterval:    1 * time.Minute,
			HistoryWindow:     2 * time.Hour,
			ConfidenceLevel:   0.99,
		},
		Sync: SyncConfig{
			Enabled:            true,
			SyncInterval:       2 * time.Second,
			MaxStaleness:       10 * time.Second,
			ConflictResolution: "crdt",
			EnableVersioning:   true,
		},
		Consensus: ConsensusConfig{
			Enabled:           true,
			Algorithm:         "raft",
			QuorumSize:        5,
			ElectionTimeout:   200 * time.Millisecond,
			HeartbeatInterval: 100 * time.Millisecond,
			AdaptiveMode:      true,
		},
	}

	copy := original.DeepCopy()

	// Verify Transport fields
	if copy.Transport.MinStreams != original.Transport.MinStreams {
		t.Errorf("Transport.MinStreams: got %v, want %v", copy.Transport.MinStreams, original.Transport.MinStreams)
	}
	if copy.Transport.MaxStreams != original.Transport.MaxStreams {
		t.Errorf("Transport.MaxStreams: got %v, want %v", copy.Transport.MaxStreams, original.Transport.MaxStreams)
	}
	if copy.Transport.InitialStreams != original.Transport.InitialStreams {
		t.Errorf("Transport.InitialStreams: got %v, want %v", copy.Transport.InitialStreams, original.Transport.InitialStreams)
	}
	if copy.Transport.StreamScalingFactor != original.Transport.StreamScalingFactor {
		t.Errorf("Transport.StreamScalingFactor: got %v, want %v", copy.Transport.StreamScalingFactor, original.Transport.StreamScalingFactor)
	}
	if copy.Transport.EnableECN != original.Transport.EnableECN {
		t.Errorf("Transport.EnableECN: got %v, want %v", copy.Transport.EnableECN, original.Transport.EnableECN)
	}
	if copy.Transport.SendBufferSize != original.Transport.SendBufferSize {
		t.Errorf("Transport.SendBufferSize: got %v, want %v", copy.Transport.SendBufferSize, original.Transport.SendBufferSize)
	}
	if copy.Transport.RecvBufferSize != original.Transport.RecvBufferSize {
		t.Errorf("Transport.RecvBufferSize: got %v, want %v", copy.Transport.RecvBufferSize, original.Transport.RecvBufferSize)
	}
	if copy.Transport.ConnectTimeout != original.Transport.ConnectTimeout {
		t.Errorf("Transport.ConnectTimeout: got %v, want %v", copy.Transport.ConnectTimeout, original.Transport.ConnectTimeout)
	}
	if copy.Transport.ReadTimeout != original.Transport.ReadTimeout {
		t.Errorf("Transport.ReadTimeout: got %v, want %v", copy.Transport.ReadTimeout, original.Transport.ReadTimeout)
	}
	if copy.Transport.WriteTimeout != original.Transport.WriteTimeout {
		t.Errorf("Transport.WriteTimeout: got %v, want %v", copy.Transport.WriteTimeout, original.Transport.WriteTimeout)
	}
	if copy.Transport.EnableRDMA != original.Transport.EnableRDMA {
		t.Errorf("Transport.EnableRDMA: got %v, want %v", copy.Transport.EnableRDMA, original.Transport.EnableRDMA)
	}
	if copy.Transport.RDMADevice != original.Transport.RDMADevice {
		t.Errorf("Transport.RDMADevice: got %v, want %v", copy.Transport.RDMADevice, original.Transport.RDMADevice)
	}
	if copy.Transport.RDMAPort != original.Transport.RDMAPort {
		t.Errorf("Transport.RDMAPort: got %v, want %v", copy.Transport.RDMAPort, original.Transport.RDMAPort)
	}
	if copy.Transport.EnablePacing != original.Transport.EnablePacing {
		t.Errorf("Transport.EnablePacing: got %v, want %v", copy.Transport.EnablePacing, original.Transport.EnablePacing)
	}
	if copy.Transport.PacingRate != original.Transport.PacingRate {
		t.Errorf("Transport.PacingRate: got %v, want %v", copy.Transport.PacingRate, original.Transport.PacingRate)
	}

	// Verify Compression fields
	if copy.Compression.Enabled != original.Compression.Enabled {
		t.Errorf("Compression.Enabled: got %v, want %v", copy.Compression.Enabled, original.Compression.Enabled)
	}
	if copy.Compression.MaxDeltaChain != original.Compression.MaxDeltaChain {
		t.Errorf("Compression.MaxDeltaChain: got %v, want %v", copy.Compression.MaxDeltaChain, original.Compression.MaxDeltaChain)
	}
	if copy.Compression.DeltaAlgorithm != original.Compression.DeltaAlgorithm {
		t.Errorf("Compression.DeltaAlgorithm: got %v, want %v", copy.Compression.DeltaAlgorithm, original.Compression.DeltaAlgorithm)
	}
	if copy.Compression.EnableDictionary != original.Compression.EnableDictionary {
		t.Errorf("Compression.EnableDictionary: got %v, want %v", copy.Compression.EnableDictionary, original.Compression.EnableDictionary)
	}
	if copy.Compression.DictionaryUpdateInterval != original.Compression.DictionaryUpdateInterval {
		t.Errorf("Compression.DictionaryUpdateInterval: got %v, want %v", copy.Compression.DictionaryUpdateInterval, original.Compression.DictionaryUpdateInterval)
	}
	if copy.Compression.AdaptiveThreshold != original.Compression.AdaptiveThreshold {
		t.Errorf("Compression.AdaptiveThreshold: got %v, want %v", copy.Compression.AdaptiveThreshold, original.Compression.AdaptiveThreshold)
	}
	if copy.Compression.MinCompressionRatio != original.Compression.MinCompressionRatio {
		t.Errorf("Compression.MinCompressionRatio: got %v, want %v", copy.Compression.MinCompressionRatio, original.Compression.MinCompressionRatio)
	}
	if copy.Compression.BaselineSyncInterval != original.Compression.BaselineSyncInterval {
		t.Errorf("Compression.BaselineSyncInterval: got %v, want %v", copy.Compression.BaselineSyncInterval, original.Compression.BaselineSyncInterval)
	}
	if copy.Compression.PruningInterval != original.Compression.PruningInterval {
		t.Errorf("Compression.PruningInterval: got %v, want %v", copy.Compression.PruningInterval, original.Compression.PruningInterval)
	}

	// Verify Prediction fields
	if copy.Prediction.PredictionHorizon != original.Prediction.PredictionHorizon {
		t.Errorf("Prediction.PredictionHorizon: got %v, want %v", copy.Prediction.PredictionHorizon, original.Prediction.PredictionHorizon)
	}
	if copy.Prediction.UpdateInterval != original.Prediction.UpdateInterval {
		t.Errorf("Prediction.UpdateInterval: got %v, want %v", copy.Prediction.UpdateInterval, original.Prediction.UpdateInterval)
	}
	if copy.Prediction.HistoryWindow != original.Prediction.HistoryWindow {
		t.Errorf("Prediction.HistoryWindow: got %v, want %v", copy.Prediction.HistoryWindow, original.Prediction.HistoryWindow)
	}
	if copy.Prediction.ConfidenceLevel != original.Prediction.ConfidenceLevel {
		t.Errorf("Prediction.ConfidenceLevel: got %v, want %v", copy.Prediction.ConfidenceLevel, original.Prediction.ConfidenceLevel)
	}

	// Verify Sync fields
	if copy.Sync.SyncInterval != original.Sync.SyncInterval {
		t.Errorf("Sync.SyncInterval: got %v, want %v", copy.Sync.SyncInterval, original.Sync.SyncInterval)
	}
	if copy.Sync.MaxStaleness != original.Sync.MaxStaleness {
		t.Errorf("Sync.MaxStaleness: got %v, want %v", copy.Sync.MaxStaleness, original.Sync.MaxStaleness)
	}
	if copy.Sync.EnableVersioning != original.Sync.EnableVersioning {
		t.Errorf("Sync.EnableVersioning: got %v, want %v", copy.Sync.EnableVersioning, original.Sync.EnableVersioning)
	}

	// Verify Consensus fields
	if copy.Consensus.QuorumSize != original.Consensus.QuorumSize {
		t.Errorf("Consensus.QuorumSize: got %v, want %v", copy.Consensus.QuorumSize, original.Consensus.QuorumSize)
	}
	if copy.Consensus.ElectionTimeout != original.Consensus.ElectionTimeout {
		t.Errorf("Consensus.ElectionTimeout: got %v, want %v", copy.Consensus.ElectionTimeout, original.Consensus.ElectionTimeout)
	}
	if copy.Consensus.HeartbeatInterval != original.Consensus.HeartbeatInterval {
		t.Errorf("Consensus.HeartbeatInterval: got %v, want %v", copy.Consensus.HeartbeatInterval, original.Consensus.HeartbeatInterval)
	}
	if copy.Consensus.AdaptiveMode != original.Consensus.AdaptiveMode {
		t.Errorf("Consensus.AdaptiveMode: got %v, want %v", copy.Consensus.AdaptiveMode, original.Consensus.AdaptiveMode)
	}
}

// BenchmarkConfigDeepCopy measures the performance of DeepCopy
func BenchmarkConfigDeepCopy(b *testing.B) {
	config := DefaultConfig()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = config.DeepCopy()
	}
}

// ============================================================================
// Configuration Validation Tests
// ============================================================================

// TestConfigValidate_NilConfig tests validation with nil config
func TestConfigValidate_NilConfig(t *testing.T) {
	var c *Config
	err := c.Validate()
	if err == nil {
		t.Fatal("expected error for nil config")
	}
	if err.Error() != "config cannot be nil" {
		t.Errorf("unexpected error message: %v", err)
	}
}

// TestConfigValidate_EmptyVersion tests validation with empty version
func TestConfigValidate_EmptyVersion(t *testing.T) {
	c := &Config{
		Version: "",
		Transport: TransportConfig{
			MinStreams:          1,
			MaxStreams:          10,
			InitialStreams:      5,
			StreamScalingFactor: 1.5,
			CongestionAlgorithm: "bbr",
			SendBufferSize:      1024,
			RecvBufferSize:      1024,
			ConnectTimeout:      time.Second,
			ReadTimeout:         time.Second,
			WriteTimeout:        time.Second,
		},
	}
	err := c.Validate()
	if err == nil {
		t.Fatal("expected error for empty version")
	}
	if err.Error() != "version cannot be empty" {
		t.Errorf("unexpected error message: %v", err)
	}
}

// TestConfigValidate_DefaultConfig tests that default config is valid
func TestConfigValidate_DefaultConfig(t *testing.T) {
	c := DefaultConfig()
	if err := c.Validate(); err != nil {
		t.Fatalf("default config should be valid: %v", err)
	}
}

// TestConfigValidate_DisabledConfig tests that validation runs even when disabled
func TestConfigValidate_DisabledConfig(t *testing.T) {
	c := DefaultConfig()
	c.Enabled = false

	// Set invalid transport config
	c.Transport.MinStreams = -1

	err := c.Validate()
	if err == nil {
		t.Fatal("expected validation to fail even when disabled")
	}
	if err.Error() != "transport.min_streams must be >= 1" {
		t.Errorf("unexpected error message: %v", err)
	}
}

// TestTransportValidation tests transport configuration validation
func TestTransportValidation(t *testing.T) {
	tests := []struct {
		name      string
		mutate    func(*Config)
		wantError string
	}{
		{
			name: "min_streams too low",
			mutate: func(c *Config) {
				c.Transport.MinStreams = 0
			},
			wantError: "transport.min_streams must be >= 1",
		},
		{
			name: "min_streams too high",
			mutate: func(c *Config) {
				c.Transport.MinStreams = 2000
			},
			wantError: "transport.min_streams must be <= 1024",
		},
		{
			name: "max_streams less than min_streams",
			mutate: func(c *Config) {
				c.Transport.MinStreams = 100
				c.Transport.MaxStreams = 50
			},
			wantError: "transport.max_streams must be >= min_streams",
		},
		{
			name: "max_streams too high",
			mutate: func(c *Config) {
				c.Transport.MaxStreams = 5000
			},
			wantError: "transport.max_streams must be <= 4096",
		},
		{
			name: "initial_streams less than min_streams",
			mutate: func(c *Config) {
				c.Transport.MinStreams = 50
				c.Transport.InitialStreams = 25
			},
			wantError: "transport.initial_streams must be >= min_streams",
		},
		{
			name: "initial_streams greater than max_streams",
			mutate: func(c *Config) {
				c.Transport.MaxStreams = 100
				c.Transport.InitialStreams = 150
			},
			wantError: "transport.initial_streams must be <= max_streams",
		},
		{
			name: "stream_scaling_factor too low",
			mutate: func(c *Config) {
				c.Transport.StreamScalingFactor = 1.0
			},
			wantError: "transport.stream_scaling_factor must be between 1.0 and 10.0",
		},
		{
			name: "stream_scaling_factor too high",
			mutate: func(c *Config) {
				c.Transport.StreamScalingFactor = 15.0
			},
			wantError: "transport.stream_scaling_factor must be between 1.0 and 10.0",
		},
		{
			name: "invalid congestion algorithm",
			mutate: func(c *Config) {
				c.Transport.CongestionAlgorithm = "invalid"
			},
			wantError: "transport.congestion_algorithm must be one of: bbr, cubic, reno",
		},
		{
			name: "send_buffer_size too small",
			mutate: func(c *Config) {
				c.Transport.SendBufferSize = 512
			},
			wantError: "transport.send_buffer_size must be >= 1024 bytes",
		},
		{
			name: "send_buffer_size too large",
			mutate: func(c *Config) {
				c.Transport.SendBufferSize = 2 * 1024 * 1024 * 1024 // 2 GB
			},
			wantError: "transport.send_buffer_size must be <= 1GB",
		},
		{
			name: "recv_buffer_size too small",
			mutate: func(c *Config) {
				c.Transport.RecvBufferSize = 512
			},
			wantError: "transport.recv_buffer_size must be >= 1024 bytes",
		},
		{
			name: "recv_buffer_size too large",
			mutate: func(c *Config) {
				c.Transport.RecvBufferSize = 2 * 1024 * 1024 * 1024 // 2 GB
			},
			wantError: "transport.recv_buffer_size must be <= 1GB",
		},
		{
			name: "connect_timeout too short",
			mutate: func(c *Config) {
				c.Transport.ConnectTimeout = 500 * time.Millisecond
			},
			wantError: "transport.connect_timeout must be >= 1s",
		},
		{
			name: "connect_timeout too long",
			mutate: func(c *Config) {
				c.Transport.ConnectTimeout = 10 * time.Minute
			},
			wantError: "transport.connect_timeout must be <= 5m",
		},
		{
			name: "read_timeout too short",
			mutate: func(c *Config) {
				c.Transport.ReadTimeout = 500 * time.Millisecond
			},
			wantError: "transport.read_timeout must be >= 1s",
		},
		{
			name: "read_timeout too long",
			mutate: func(c *Config) {
				c.Transport.ReadTimeout = 2 * time.Hour
			},
			wantError: "transport.read_timeout must be <= 1h",
		},
		{
			name: "write_timeout too short",
			mutate: func(c *Config) {
				c.Transport.WriteTimeout = 500 * time.Millisecond
			},
			wantError: "transport.write_timeout must be >= 1s",
		},
		{
			name: "write_timeout too long",
			mutate: func(c *Config) {
				c.Transport.WriteTimeout = 2 * time.Hour
			},
			wantError: "transport.write_timeout must be <= 1h",
		},
		{
			name: "rdma enabled without device",
			mutate: func(c *Config) {
				c.Transport.EnableRDMA = true
				c.Transport.RDMADevice = ""
			},
			wantError: "transport.rdma_device must be specified when RDMA is enabled",
		},
		{
			name: "rdma port invalid",
			mutate: func(c *Config) {
				c.Transport.EnableRDMA = true
				c.Transport.RDMADevice = "mlx5_0"
				c.Transport.RDMAPort = 0
			},
			wantError: "transport.rdma_port must be between 1 and 65535",
		},
		{
			name: "pacing rate too low",
			mutate: func(c *Config) {
				c.Transport.EnablePacing = true
				c.Transport.PacingRate = 512 * 1024 // 512 Kbps
			},
			wantError: "transport.pacing_rate must be >= 1 Mbps",
		},
		{
			name: "pacing rate too high",
			mutate: func(c *Config) {
				c.Transport.EnablePacing = true
				c.Transport.PacingRate = 200 * 1024 * 1024 * 1024 // 200 Gbps
			},
			wantError: "transport.pacing_rate must be <= 100 Gbps",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := DefaultConfig()
			tt.mutate(c)
			err := c.Validate()
			if err == nil {
				t.Fatal("expected validation error")
			}
			if err.Error() != tt.wantError {
				t.Errorf("expected error %q, got %q", tt.wantError, err.Error())
			}
		})
	}
}

// TestCompressionValidation tests compression configuration validation
func TestCompressionValidation(t *testing.T) {
	tests := []struct {
		name      string
		mutate    func(*Config)
		wantError string
	}{
		{
			name: "invalid compression algorithm",
			mutate: func(c *Config) {
				c.Compression.Algorithm = "invalid"
			},
			wantError: "compression.algorithm must be one of: zstd, lz4, snappy",
		},
		{
			name: "compression level too low",
			mutate: func(c *Config) {
				c.Compression.Level = CompressionLevel(-1)
			},
			wantError: "compression.level must be between 0 and 3",
		},
		{
			name: "compression level too high",
			mutate: func(c *Config) {
				c.Compression.Level = CompressionLevel(4)
			},
			wantError: "compression.level must be between 0 and 3",
		},
		{
			name: "baseline_interval too short",
			mutate: func(c *Config) {
				c.Compression.EnableDeltaEncoding = true
				c.Compression.BaselineInterval = 500 * time.Millisecond
			},
			wantError: "compression.baseline_interval must be >= 1s",
		},
		{
			name: "baseline_interval too long",
			mutate: func(c *Config) {
				c.Compression.EnableDeltaEncoding = true
				c.Compression.BaselineInterval = 25 * time.Hour
			},
			wantError: "compression.baseline_interval must be <= 24h",
		},
		{
			name: "max_delta_chain too low",
			mutate: func(c *Config) {
				c.Compression.EnableDeltaEncoding = true
				c.Compression.MaxDeltaChain = 0
			},
			wantError: "compression.max_delta_chain must be >= 1",
		},
		{
			name: "max_delta_chain too high",
			mutate: func(c *Config) {
				c.Compression.EnableDeltaEncoding = true
				c.Compression.MaxDeltaChain = 2000
			},
			wantError: "compression.max_delta_chain must be <= 1000",
		},
		{
			name: "invalid delta algorithm",
			mutate: func(c *Config) {
				c.Compression.DeltaAlgorithm = "invalid"
			},
			wantError: "compression.delta_algorithm must be one of: xor, rsync, bsdiff, auto",
		},
		{
			name: "dictionary_update_interval too short",
			mutate: func(c *Config) {
				c.Compression.EnableDictionary = true
				c.Compression.DictionaryUpdateInterval = 30 * time.Second
			},
			wantError: "compression.dictionary_update_interval must be >= 1m",
		},
		{
			name: "dictionary_update_interval too long",
			mutate: func(c *Config) {
				c.Compression.EnableDictionary = true
				c.Compression.DictionaryUpdateInterval = 8 * 24 * time.Hour
			},
			wantError: "compression.dictionary_update_interval must be <= 7d",
		},
		{
			name: "adaptive_threshold too low",
			mutate: func(c *Config) {
				c.Compression.EnableAdaptive = true
				c.Compression.AdaptiveThreshold = 0.5
			},
			wantError: "compression.adaptive_threshold must be > 1.0",
		},
		{
			name: "adaptive_threshold too high",
			mutate: func(c *Config) {
				c.Compression.EnableAdaptive = true
				c.Compression.AdaptiveThreshold = 150.0
			},
			wantError: "compression.adaptive_threshold must be <= 100.0",
		},
		{
			name: "min_compression_ratio too low",
			mutate: func(c *Config) {
				c.Compression.EnableAdaptive = true
				c.Compression.MinCompressionRatio = 0.5
			},
			wantError: "compression.min_compression_ratio must be >= 1.0",
		},
		{
			name: "min_compression_ratio exceeds adaptive_threshold",
			mutate: func(c *Config) {
				c.Compression.EnableAdaptive = true
				c.Compression.AdaptiveThreshold = 10.0
				c.Compression.MinCompressionRatio = 15.0
			},
			wantError: "compression.min_compression_ratio must be <= adaptive_threshold",
		},
		{
			name: "baseline_sync_interval too short",
			mutate: func(c *Config) {
				c.Compression.EnableBaselineSync = true
				c.Compression.BaselineSyncInterval = 50 * time.Millisecond
			},
			wantError: "compression.baseline_sync_interval must be >= 100ms",
		},
		{
			name: "baseline_sync_interval too long",
			mutate: func(c *Config) {
				c.Compression.EnableBaselineSync = true
				c.Compression.BaselineSyncInterval = 2 * time.Minute
			},
			wantError: "compression.baseline_sync_interval must be <= 1m",
		},
		{
			name: "pruning_interval too short",
			mutate: func(c *Config) {
				c.Compression.EnablePruning = true
				c.Compression.PruningInterval = 30 * time.Second
			},
			wantError: "compression.pruning_interval must be >= 1m",
		},
		{
			name: "pruning_interval too long",
			mutate: func(c *Config) {
				c.Compression.EnablePruning = true
				c.Compression.PruningInterval = 25 * time.Hour
			},
			wantError: "compression.pruning_interval must be <= 24h",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := DefaultConfig()
			tt.mutate(c)
			err := c.Validate()
			if err == nil {
				t.Fatal("expected validation error")
			}
			if err.Error() != tt.wantError {
				t.Errorf("expected error %q, got %q", tt.wantError, err.Error())
			}
		})
	}
}

// TestPredictionValidation tests prediction configuration validation
func TestPredictionValidation(t *testing.T) {
	tests := []struct {
		name      string
		mutate    func(*Config)
		wantError string
	}{
		{
			name: "invalid model type",
			mutate: func(c *Config) {
				c.Prediction.Enabled = true
				c.Prediction.ModelType = "invalid"
			},
			wantError: "prediction.model_type must be one of: lstm, arima, prophet",
		},
		{
			name: "prediction_horizon too short",
			mutate: func(c *Config) {
				c.Prediction.Enabled = true
				c.Prediction.PredictionHorizon = 500 * time.Millisecond
			},
			wantError: "prediction.prediction_horizon must be >= 1s",
		},
		{
			name: "prediction_horizon too long",
			mutate: func(c *Config) {
				c.Prediction.Enabled = true
				c.Prediction.PredictionHorizon = 2 * time.Hour
			},
			wantError: "prediction.prediction_horizon must be <= 1h",
		},
		{
			name: "update_interval too short",
			mutate: func(c *Config) {
				c.Prediction.Enabled = true
				c.Prediction.UpdateInterval = 500 * time.Millisecond
			},
			wantError: "prediction.update_interval must be >= 1s",
		},
		{
			name: "update_interval exceeds prediction_horizon",
			mutate: func(c *Config) {
				c.Prediction.Enabled = true
				c.Prediction.PredictionHorizon = 1 * time.Minute
				c.Prediction.UpdateInterval = 2 * time.Minute
			},
			wantError: "prediction.update_interval must be <= prediction_horizon",
		},
		{
			name: "history_window less than prediction_horizon",
			mutate: func(c *Config) {
				c.Prediction.Enabled = true
				c.Prediction.PredictionHorizon = 10 * time.Minute
				c.Prediction.HistoryWindow = 5 * time.Minute
			},
			wantError: "prediction.history_window must be >= prediction_horizon",
		},
		{
			name: "history_window too long",
			mutate: func(c *Config) {
				c.Prediction.Enabled = true
				c.Prediction.HistoryWindow = 25 * time.Hour
			},
			wantError: "prediction.history_window must be <= 24h",
		},
		{
			name: "confidence_level too low",
			mutate: func(c *Config) {
				c.Prediction.Enabled = true
				c.Prediction.ConfidenceLevel = 0.0
			},
			wantError: "prediction.confidence_level must be between 0.0 and 1.0",
		},
		{
			name: "confidence_level too high",
			mutate: func(c *Config) {
				c.Prediction.Enabled = true
				c.Prediction.ConfidenceLevel = 1.0
			},
			wantError: "prediction.confidence_level must be between 0.0 and 1.0",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := DefaultConfig()
			tt.mutate(c)
			err := c.Validate()
			if err == nil {
				t.Fatal("expected validation error")
			}
			if err.Error() != tt.wantError {
				t.Errorf("expected error %q, got %q", tt.wantError, err.Error())
			}
		})
	}
}

// TestSyncValidation tests sync configuration validation
func TestSyncValidation(t *testing.T) {
	tests := []struct {
		name      string
		mutate    func(*Config)
		wantError string
	}{
		{
			name: "sync_interval too short",
			mutate: func(c *Config) {
				c.Sync.Enabled = true
				c.Sync.SyncInterval = 50 * time.Millisecond
			},
			wantError: "sync.sync_interval must be >= 100ms",
		},
		{
			name: "sync_interval too long",
			mutate: func(c *Config) {
				c.Sync.Enabled = true
				c.Sync.SyncInterval = 2 * time.Minute
			},
			wantError: "sync.sync_interval must be <= 1m",
		},
		{
			name: "max_staleness less than sync_interval",
			mutate: func(c *Config) {
				c.Sync.Enabled = true
				c.Sync.SyncInterval = 5 * time.Second
				c.Sync.MaxStaleness = 3 * time.Second
			},
			wantError: "sync.max_staleness must be >= sync_interval",
		},
		{
			name: "max_staleness too long",
			mutate: func(c *Config) {
				c.Sync.Enabled = true
				c.Sync.MaxStaleness = 10 * time.Minute
			},
			wantError: "sync.max_staleness must be <= 5m",
		},
		{
			name: "invalid conflict resolution",
			mutate: func(c *Config) {
				c.Sync.Enabled = true
				c.Sync.ConflictResolution = "invalid"
			},
			wantError: "sync.conflict_resolution must be one of: lww, mvcc, crdt",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := DefaultConfig()
			tt.mutate(c)
			err := c.Validate()
			if err == nil {
				t.Fatal("expected validation error")
			}
			if err.Error() != tt.wantError {
				t.Errorf("expected error %q, got %q", tt.wantError, err.Error())
			}
		})
	}
}

// TestConsensusValidation tests consensus configuration validation
func TestConsensusValidation(t *testing.T) {
	tests := []struct {
		name      string
		mutate    func(*Config)
		wantError string
	}{
		{
			name: "invalid consensus algorithm",
			mutate: func(c *Config) {
				c.Consensus.Enabled = true
				c.Consensus.Algorithm = "invalid"
			},
			wantError: "consensus.algorithm must be one of: raft, gossip, byzantine",
		},
		{
			name: "quorum_size too low",
			mutate: func(c *Config) {
				c.Consensus.Enabled = true
				c.Consensus.QuorumSize = 0
			},
			wantError: "consensus.quorum_size must be >= 1",
		},
		{
			name: "quorum_size too high",
			mutate: func(c *Config) {
				c.Consensus.Enabled = true
				c.Consensus.QuorumSize = 2000
			},
			wantError: "consensus.quorum_size must be <= 1000",
		},
		{
			name: "election_timeout too short",
			mutate: func(c *Config) {
				c.Consensus.Enabled = true
				c.Consensus.ElectionTimeout = 5 * time.Millisecond
			},
			wantError: "consensus.election_timeout must be >= 10ms",
		},
		{
			name: "election_timeout too long",
			mutate: func(c *Config) {
				c.Consensus.Enabled = true
				c.Consensus.ElectionTimeout = 15 * time.Second
			},
			wantError: "consensus.election_timeout must be <= 10s",
		},
		{
			name: "heartbeat_interval too short",
			mutate: func(c *Config) {
				c.Consensus.Enabled = true
				c.Consensus.HeartbeatInterval = 500 * time.Microsecond
			},
			wantError: "consensus.heartbeat_interval must be >= 1ms",
		},
		{
			name: "heartbeat_interval exceeds election_timeout",
			mutate: func(c *Config) {
				c.Consensus.Enabled = true
				c.Consensus.ElectionTimeout = 100 * time.Millisecond
				c.Consensus.HeartbeatInterval = 150 * time.Millisecond
			},
			wantError: "consensus.heartbeat_interval must be < election_timeout",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := DefaultConfig()
			tt.mutate(c)
			err := c.Validate()
			if err == nil {
				t.Fatal("expected validation error")
			}
			if err.Error() != tt.wantError {
				t.Errorf("expected error %q, got %q", tt.wantError, err.Error())
			}
		})
	}
}

// TestValidConfig_AllComponentsEnabled tests a valid config with all components enabled
func TestValidConfig_AllComponentsEnabled(t *testing.T) {
	c := &Config{
		Enabled: true,
		Version: DWCPVersion,
		Transport: TransportConfig{
			MinStreams:          32,
			MaxStreams:          256,
			InitialStreams:      64,
			StreamScalingFactor: 2.0,
			CongestionAlgorithm: "bbr",
			EnableECN:           true,
			SendBufferSize:      8 * 1024 * 1024,
			RecvBufferSize:      8 * 1024 * 1024,
			ConnectTimeout:      10 * time.Second,
			ReadTimeout:         30 * time.Second,
			WriteTimeout:        30 * time.Second,
			EnableRDMA:          true,
			RDMADevice:          "mlx5_0",
			RDMAPort:            1,
			EnablePacing:        true,
			PacingRate:          10 * 1024 * 1024 * 1024, // 10 Gbps
		},
		Compression: CompressionConfig{
			Enabled:                  true,
			Algorithm:                "zstd",
			Level:                    CompressionLevelBalanced,
			EnableDeltaEncoding:      true,
			BaselineInterval:         1 * time.Minute,
			MaxDeltaChain:            50,
			DeltaAlgorithm:           "auto",
			EnableDictionary:         true,
			DictionaryUpdateInterval: 24 * time.Hour,
			EnableAdaptive:           true,
			AdaptiveThreshold:        10.0,
			MinCompressionRatio:      2.0,
			EnableBaselineSync:       true,
			BaselineSyncInterval:     5 * time.Second,
			EnablePruning:            true,
			PruningInterval:          5 * time.Minute,
		},
		Prediction: PredictionConfig{
			Enabled:           true,
			ModelType:         "lstm",
			PredictionHorizon: 5 * time.Minute,
			UpdateInterval:    30 * time.Second,
			HistoryWindow:     1 * time.Hour,
			ConfidenceLevel:   0.95,
		},
		Sync: SyncConfig{
			Enabled:            true,
			SyncInterval:       1 * time.Second,
			MaxStaleness:       5 * time.Second,
			ConflictResolution: "crdt",
			EnableVersioning:   true,
		},
		Consensus: ConsensusConfig{
			Enabled:           true,
			Algorithm:         "raft",
			QuorumSize:        5,
			ElectionTimeout:   200 * time.Millisecond,
			HeartbeatInterval: 50 * time.Millisecond,
			AdaptiveMode:      true,
		},
	}

	if err := c.Validate(); err != nil {
		t.Fatalf("valid config should pass validation: %v", err)
	}
}

// BenchmarkConfigValidate benchmarks the validation performance
func BenchmarkConfigValidate(b *testing.B) {
	c := DefaultConfig()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = c.Validate()
	}
}
