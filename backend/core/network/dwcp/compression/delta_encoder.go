package compression

import (
	"bytes"
	"fmt"
	"sync"
	"time"

	"github.com/klauspost/compress/zstd"
	"go.uber.org/zap"
)

// DeltaEncoder implements hierarchical delta encoding with baseline state management
// Phase 1: Production-ready with dictionary training and advanced algorithms
type DeltaEncoder struct {
	// Baseline management
	baselineStates  map[string]*BaselineState
	baselineMutex   sync.RWMutex
	baselineInterval time.Duration
	maxBaselineAge   time.Duration

	// Compression (with dictionary support)
	compressor   *zstd.Encoder
	decompressor *zstd.Decoder
	dictEncoder  *zstd.Encoder // Dictionary-based encoder
	dictDecoder  *zstd.Decoder // Dictionary-based decoder
	compressMutex sync.Mutex // Protects concurrent compress/decompress operations

	// Phase 1: Advanced features
	dictionaryTrainer *DictionaryTrainer
	deltaComputer     DeltaComputer
	adaptiveComp      *AdaptiveCompressor
	baselineSync      *BaselineSynchronizer
	metrics           *CompressionMetrics

	// Configuration
	config *DeltaEncodingConfig
	logger *zap.Logger

	// Legacy metrics (kept for backward compatibility)
	totalEncoded     uint64
	totalDecoded     uint64
	deltaHits        uint64
	baselineRefresh  uint64
	mu               sync.RWMutex
}

// BaselineState represents a baseline snapshot for delta computation
type BaselineState struct {
	Data      []byte
	Timestamp time.Time
	DeltaCount int // Number of deltas computed from this baseline
}

// DeltaEncodingConfig configuration for delta encoding
type DeltaEncodingConfig struct {
	Enabled            bool          `json:"enabled" yaml:"enabled"`
	BaselineInterval   time.Duration `json:"baseline_interval" yaml:"baseline_interval"`
	MaxBaselineAge     time.Duration `json:"max_baseline_age" yaml:"max_baseline_age"`
	MaxDeltaChain      int           `json:"max_delta_chain" yaml:"max_delta_chain"`
	CompressionLevel   int           `json:"compression_level" yaml:"compression_level"` // Zstandard: 0-9
	EnableDictionary   bool          `json:"enable_dictionary" yaml:"enable_dictionary"`

	// Phase 1: Advanced features
	DeltaAlgorithm     string        `json:"delta_algorithm" yaml:"delta_algorithm"`         // "xor", "rsync", "bsdiff", "auto"
	AdaptiveThreshold  float64       `json:"adaptive_threshold" yaml:"adaptive_threshold"`   // Auto-adjust compression if ratio < threshold
	MinCompressionRatio float64      `json:"min_compression_ratio" yaml:"min_compression_ratio"` // Skip compression if ratio < this
	EnableAdaptive     bool          `json:"enable_adaptive" yaml:"enable_adaptive"`         // Enable adaptive compression
	EnableBaselineSync bool          `json:"enable_baseline_sync" yaml:"enable_baseline_sync"` // Enable cluster sync
}

// DefaultDeltaEncodingConfig returns sensible defaults for Phase 1
func DefaultDeltaEncodingConfig() *DeltaEncodingConfig {
	return &DeltaEncodingConfig{
		Enabled:             true,
		BaselineInterval:    5 * time.Minute,
		MaxBaselineAge:      15 * time.Minute,
		MaxDeltaChain:       10,
		CompressionLevel:    3, // Balanced compression (Zstandard level 3)
		EnableDictionary:    true, // Phase 1: Dictionary training enabled
		DeltaAlgorithm:      "auto", // Phase 1: Auto-select algorithm
		AdaptiveThreshold:   3.0,
		MinCompressionRatio: 1.1,
		EnableAdaptive:      true,
		EnableBaselineSync:  false, // Disabled until cluster configured
	}
}

// NewDeltaEncoder creates a new delta encoder with Phase 1 features
func NewDeltaEncoder(config *DeltaEncodingConfig, logger *zap.Logger) (*DeltaEncoder, error) {
	if config == nil {
		config = DefaultDeltaEncodingConfig()
	}

	if logger == nil {
		logger, _ = zap.NewProduction()
	}

	// Create Zstandard encoder with specified compression level
	encoder, err := zstd.NewWriter(nil,
		zstd.WithEncoderLevel(zstd.EncoderLevelFromZstd(config.CompressionLevel)),
		zstd.WithEncoderConcurrency(4), // Use 4 concurrent goroutines
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create zstd encoder: %w", err)
	}

	// Create Zstandard decoder
	decoder, err := zstd.NewReader(nil,
		zstd.WithDecoderConcurrency(4),
	)
	if err != nil {
		encoder.Close()
		return nil, fmt.Errorf("failed to create zstd decoder: %w", err)
	}

	de := &DeltaEncoder{
		baselineStates:   make(map[string]*BaselineState),
		baselineInterval: config.BaselineInterval,
		maxBaselineAge:   config.MaxBaselineAge,
		compressor:       encoder,
		decompressor:     decoder,
		config:           config,
		logger:           logger,
		metrics:          NewCompressionMetrics(),
	}

	// Phase 1: Initialize dictionary trainer
	if config.EnableDictionary {
		dictConfig := DefaultDictionaryTrainingConfig()
		de.dictionaryTrainer, err = NewDictionaryTrainer(dictConfig, logger)
		if err != nil {
			logger.Warn("Failed to initialize dictionary trainer", zap.Error(err))
		}
	}

	// Phase 1: Initialize delta algorithm
	de.deltaComputer = DeltaAlgorithmFactory(DeltaAlgorithm(config.DeltaAlgorithm))

	// Phase 1: Initialize adaptive compressor
	if config.EnableAdaptive {
		adaptiveConfig := DefaultAdaptiveConfig()
		adaptiveConfig.MinCompressionRatio = config.MinCompressionRatio
		adaptiveConfig.TargetRatio = config.AdaptiveThreshold
		de.adaptiveComp = NewAdaptiveCompressor(adaptiveConfig, logger)
	}

	// Phase 1: Initialize baseline synchronizer
	if config.EnableBaselineSync {
		syncConfig := DefaultBaselineSyncConfig()
		syncConfig.Enabled = true
		de.baselineSync = NewBaselineSynchronizer(syncConfig, logger)
	}

	logger.Info("DeltaEncoder initialized with Phase 1 features",
		zap.Bool("dictionary", config.EnableDictionary),
		zap.String("delta_algorithm", config.DeltaAlgorithm),
		zap.Bool("adaptive", config.EnableAdaptive),
		zap.Bool("baseline_sync", config.EnableBaselineSync))

	return de, nil
}

// Encode applies delta encoding and compression to data with Phase 1 optimizations
// stateKey identifies the resource (e.g., "vm-123-memory")
func (de *DeltaEncoder) Encode(stateKey string, data []byte) (*EncodedData, error) {
	startTime := time.Now()

	if !de.config.Enabled {
		// If disabled, just compress without delta encoding
		return de.compressOnly(data)
	}

	// Phase 1: Add sample for dictionary training
	if de.dictionaryTrainer != nil {
		de.dictionaryTrainer.AddSample(de.extractResourceType(stateKey), data)
	}

	// Phase 1: Check if compression is worthwhile
	if de.adaptiveComp != nil && !de.adaptiveComp.ShouldCompress(len(data)) {
		de.metrics.RecordIncompressibleSkip()
		return &EncodedData{
			Data:           data,
			OriginalSize:   len(data),
			CompressedSize: len(data),
			IsDelta:        false,
			BaselineKey:    stateKey,
			Timestamp:      time.Now(),
		}, nil
	}

	de.baselineMutex.RLock()
	baseline, hasBaseline := de.baselineStates[stateKey]
	de.baselineMutex.RUnlock()

	// Check if we need to create or refresh baseline
	needsBaseline := !hasBaseline ||
		time.Since(baseline.Timestamp) > de.baselineInterval ||
		baseline.DeltaCount >= de.config.MaxDeltaChain

	if needsBaseline {
		result, err := de.createBaseline(stateKey, data)
		if err == nil && de.metrics != nil {
			duration := time.Since(startTime)
			de.metrics.RecordCompression(len(data), len(result.Data), duration, false, false)
		}
		return result, err
	}

	// Phase 1: Compute delta using advanced algorithm
	delta, err := de.deltaComputer.ComputeDelta(baseline.Data, data)
	if err != nil {
		return nil, fmt.Errorf("delta computation failed: %w", err)
	}

	// Phase 1: Record algorithm usage
	if de.metrics != nil {
		de.metrics.RecordDeltaAlgorithm(de.deltaComputer.Name())
	}

	// Phase 1: Compress with dictionary if available
	compressed, usedDict, err := de.compressWithDict(delta, stateKey)
	if err != nil {
		return nil, fmt.Errorf("delta compression failed: %w", err)
	}

	// Update baseline statistics
	de.baselineMutex.Lock()
	baseline.DeltaCount++
	de.baselineMutex.Unlock()

	de.mu.Lock()
	de.totalEncoded++
	de.deltaHits++
	de.mu.Unlock()

	// Phase 1: Record metrics
	duration := time.Since(startTime)
	if de.metrics != nil {
		de.metrics.RecordCompression(len(data), len(compressed), duration, true, usedDict)
	}

	// Phase 1: Record adaptive compression result
	if de.adaptiveComp != nil {
		ratio := float64(len(data)) / float64(len(compressed))
		de.adaptiveComp.RecordCompressionResult(ratio, duration)
	}

	return &EncodedData{
		Data:           compressed,
		OriginalSize:   len(data),
		CompressedSize: len(compressed),
		IsDelta:        true,
		BaselineKey:    stateKey,
		Timestamp:      time.Now(),
	}, nil
}

// Decode decompresses and applies delta reconstruction
func (de *DeltaEncoder) Decode(stateKey string, encoded *EncodedData) ([]byte, error) {
	if !de.config.Enabled || !encoded.IsDelta {
		// Just decompress
		return de.decompress(encoded.Data)
	}

	// Decompress delta
	delta, err := de.decompress(encoded.Data)
	if err != nil {
		return nil, fmt.Errorf("delta decompression failed: %w", err)
	}

	// Get baseline
	de.baselineMutex.RLock()
	baseline, hasBaseline := de.baselineStates[stateKey]
	de.baselineMutex.RUnlock()

	if !hasBaseline {
		return nil, fmt.Errorf("baseline not found for key: %s", stateKey)
	}

	// Reconstruct from baseline + delta
	reconstructed := de.applyDelta(baseline.Data, delta)

	de.mu.Lock()
	de.totalDecoded++
	de.mu.Unlock()

	return reconstructed, nil
}

// createBaseline creates a new baseline state
func (de *DeltaEncoder) createBaseline(stateKey string, data []byte) (*EncodedData, error) {
	// Compress the full state
	compressed, err := de.compress(data)
	if err != nil {
		return nil, fmt.Errorf("baseline compression failed: %w", err)
	}

	// Store baseline
	de.baselineMutex.Lock()
	de.baselineStates[stateKey] = &BaselineState{
		Data:       make([]byte, len(data)),
		Timestamp:  time.Now(),
		DeltaCount: 0,
	}
	copy(de.baselineStates[stateKey].Data, data)
	de.baselineMutex.Unlock()

	de.mu.Lock()
	de.totalEncoded++
	de.baselineRefresh++
	de.mu.Unlock()

	de.logger.Debug("Created baseline",
		zap.String("key", stateKey),
		zap.Int("size", len(data)),
		zap.Int("compressed", len(compressed)))

	return &EncodedData{
		Data:          compressed,
		OriginalSize:  len(data),
		CompressedSize: len(compressed),
		IsDelta:       false,
		BaselineKey:   stateKey,
		Timestamp:     time.Now(),
	}, nil
}

// computeDelta is now delegated to the delta computer (Phase 1)
func (de *DeltaEncoder) computeDelta(baseline, current []byte) []byte {
	delta, err := de.deltaComputer.ComputeDelta(baseline, current)
	if err != nil {
		// Fallback to XOR if advanced algorithm fails
		de.logger.Warn("Delta computation failed, falling back to XOR", zap.Error(err))
		xor := &XORDeltaComputer{}
		delta, _ = xor.ComputeDelta(baseline, current)
	}
	return delta
}

// applyDelta is now delegated to the delta computer (Phase 1)
func (de *DeltaEncoder) applyDelta(baseline, delta []byte) []byte {
	reconstructed, err := de.deltaComputer.ApplyDelta(baseline, delta)
	if err != nil {
		// Fallback to XOR if advanced algorithm fails
		de.logger.Warn("Delta application failed, falling back to XOR", zap.Error(err))
		xor := &XORDeltaComputer{}
		reconstructed, _ = xor.ApplyDelta(baseline, delta)
	}
	return reconstructed
}

// compress compresses data using Zstandard (thread-safe)
func (de *DeltaEncoder) compress(data []byte) ([]byte, error) {
	de.compressMutex.Lock()
	defer de.compressMutex.Unlock()

	var buf bytes.Buffer
	de.compressor.Reset(&buf)

	if _, err := de.compressor.Write(data); err != nil {
		return nil, err
	}

	if err := de.compressor.Close(); err != nil {
		return nil, err
	}

	return buf.Bytes(), nil
}

// decompress decompresses Zstandard-compressed data (thread-safe)
func (de *DeltaEncoder) decompress(data []byte) ([]byte, error) {
	de.compressMutex.Lock()
	defer de.compressMutex.Unlock()

	return de.decompressor.DecodeAll(data, nil)
}

// compressOnly compresses without delta encoding (fallback mode)
func (de *DeltaEncoder) compressOnly(data []byte) (*EncodedData, error) {
	compressed, err := de.compress(data)
	if err != nil {
		return nil, err
	}

	de.mu.Lock()
	de.totalEncoded++
	de.mu.Unlock()

	return &EncodedData{
		Data:          compressed,
		OriginalSize:  len(data),
		CompressedSize: len(compressed),
		IsDelta:       false,
		Timestamp:     time.Now(),
	}, nil
}

// PruneOldBaselines removes baselines that haven't been used recently
func (de *DeltaEncoder) PruneOldBaselines() int {
	de.baselineMutex.Lock()
	defer de.baselineMutex.Unlock()

	pruned := 0
	now := time.Now()

	for key, baseline := range de.baselineStates {
		if now.Sub(baseline.Timestamp) > de.maxBaselineAge {
			delete(de.baselineStates, key)
			pruned++
		}
	}

	if pruned > 0 {
		de.logger.Info("Pruned old baselines", zap.Int("count", pruned))
	}

	return pruned
}

// GetMetrics returns current encoding metrics
func (de *DeltaEncoder) GetMetrics() map[string]interface{} {
	de.mu.RLock()
	defer de.mu.RUnlock()

	de.baselineMutex.RLock()
	baselineCount := len(de.baselineStates)
	de.baselineMutex.RUnlock()

	var deltaHitRate float64
	if de.totalEncoded > 0 {
		deltaHitRate = float64(de.deltaHits) / float64(de.totalEncoded) * 100.0
	}

	return map[string]interface{}{
		"total_encoded":      de.totalEncoded,
		"total_decoded":      de.totalDecoded,
		"delta_hits":         de.deltaHits,
		"baseline_refreshes": de.baselineRefresh,
		"baseline_count":     baselineCount,
		"delta_hit_rate":     deltaHitRate,
		"compression_level":  de.config.CompressionLevel,
	}
}

// Close releases resources including Phase 1 components
func (de *DeltaEncoder) Close() error {
	de.compressor.Close()
	de.decompressor.Close()

	if de.dictEncoder != nil {
		de.dictEncoder.Close()
	}
	if de.dictDecoder != nil {
		de.dictDecoder.Close()
	}
	if de.dictionaryTrainer != nil {
		de.dictionaryTrainer.Close()
	}
	if de.baselineSync != nil {
		de.baselineSync.Close()
	}

	return nil
}

// Phase 1 Helper Methods

// extractResourceType extracts resource type from state key
// e.g., "vm-123-memory" -> "vm-memory"
func (de *DeltaEncoder) extractResourceType(stateKey string) string {
	// Simple extraction - production would parse more carefully
	if len(stateKey) > 3 && stateKey[:3] == "vm-" {
		// Find last dash
		lastDash := -1
		for i := len(stateKey) - 1; i >= 0; i-- {
			if stateKey[i] == '-' {
				lastDash = i
				break
			}
		}
		if lastDash > 0 && lastDash < len(stateKey)-1 {
			return "vm-" + stateKey[lastDash+1:]
		}
	}
	return "unknown"
}

// compressWithDict compresses data using dictionary if available
func (de *DeltaEncoder) compressWithDict(data []byte, stateKey string) ([]byte, bool, error) {
	resourceType := de.extractResourceType(stateKey)

	// Try dictionary compression first
	if de.dictionaryTrainer != nil {
		if dict, exists := de.dictionaryTrainer.GetDictionary(resourceType); exists {
			compressed, err := de.compressWithDictionary(data, dict)
			if err == nil {
				return compressed, true, nil
			}
			de.logger.Debug("Dictionary compression failed, falling back",
				zap.String("resource_type", resourceType),
				zap.Error(err))
		}
	}

	// Fallback to regular compression
	compressed, err := de.compress(data)
	return compressed, false, err
}

// compressWithDictionary compresses using a specific dictionary
func (de *DeltaEncoder) compressWithDictionary(data []byte, dict []byte) ([]byte, error) {
	de.compressMutex.Lock()
	defer de.compressMutex.Unlock()

	// Create or reuse dictionary encoder
	if de.dictEncoder == nil || de.dictDecoder == nil {
		var err error
		de.dictEncoder, err = zstd.NewWriter(nil,
			zstd.WithEncoderDict(dict),
			zstd.WithEncoderConcurrency(4),
		)
		if err != nil {
			return nil, err
		}

		de.dictDecoder, err = zstd.NewReader(nil,
			zstd.WithDecoderDicts(dict),
			zstd.WithDecoderConcurrency(4),
		)
		if err != nil {
			return nil, err
		}
	}

	var buf bytes.Buffer
	de.dictEncoder.Reset(&buf)

	if _, err := de.dictEncoder.Write(data); err != nil {
		return nil, err
	}

	if err := de.dictEncoder.Close(); err != nil {
		return nil, err
	}

	return buf.Bytes(), nil
}

// TrainDictionaries manually triggers dictionary training
func (de *DeltaEncoder) TrainDictionaries() error {
	if de.dictionaryTrainer == nil {
		return fmt.Errorf("dictionary training not enabled")
	}

	return de.dictionaryTrainer.TrainAllDictionaries()
}

// GetDetailedMetrics returns comprehensive Phase 1 metrics
func (de *DeltaEncoder) GetDetailedMetrics() map[string]interface{} {
	result := de.GetMetrics() // Get legacy metrics

	// Add Phase 1 metrics
	if de.metrics != nil {
		phase1Metrics := de.metrics.GetSnapshot()
		for k, v := range phase1Metrics {
			result[k] = v
		}
	}

	if de.dictionaryTrainer != nil {
		dictStats := de.dictionaryTrainer.GetStats()
		for k, v := range dictStats {
			result["dict_"+k] = v
		}
	}

	if de.adaptiveComp != nil {
		adaptiveStats := de.adaptiveComp.GetStats()
		for k, v := range adaptiveStats {
			result["adaptive_"+k] = v
		}
	}

	if de.baselineSync != nil {
		syncStats := de.baselineSync.GetStats()
		for k, v := range syncStats {
			result["sync_"+k] = v
		}
	}

	return result
}

// EncodedData represents compressed and optionally delta-encoded data
type EncodedData struct {
	Data          []byte
	OriginalSize  int
	CompressedSize int
	IsDelta       bool
	BaselineKey   string
	Timestamp     time.Time
}

// CompressionRatio returns the compression ratio achieved
func (ed *EncodedData) CompressionRatio() float64 {
	if ed.CompressedSize == 0 {
		return 0
	}
	return float64(ed.OriginalSize) / float64(ed.CompressedSize)
}
