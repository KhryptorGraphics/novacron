package encoding

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/klauspost/compress/zstd"
	"github.com/pierrec/lz4/v4"

	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/compression"
	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/upgrade"
)

// HDEv3 implements Hierarchical Delta Encoding v3 with:
// - ML-based compression selection
// - CRDT-based conflict-free sync
// - Mode-aware optimization (datacenter vs internet)
// - Enhanced delta encoding
type HDEv3 struct {
	// Configuration
	config *HDEv3Config

	// ML-based compression selector
	compSelector *CompressionSelector

	// CRDT integration for conflict-free sync
	crdtIntegration *CRDTIntegration

	// Delta encoding (reuse existing v1 delta encoder)
	deltaEncoder *compression.DeltaEncoder

	// Compression engines per algorithm
	zstdEncoder    *zstd.Encoder
	zstdMaxEncoder *zstd.Encoder
	zstdDecoder    *zstd.Decoder
	lz4Writer      *lz4.Writer
	lz4Reader      *lz4.Reader

	// Baseline management
	baselines   map[string]*BaselineV3
	baselineMu  sync.RWMutex

	// Performance metrics
	metrics *HDEv3Metrics

	// Synchronization
	mu     sync.RWMutex
	ctx    context.Context
	cancel context.CancelFunc
}

// HDEv3Config configuration for HDE v3
type HDEv3Config struct {
	NodeID string

	// Network mode
	NetworkMode upgrade.NetworkMode

	// Compression settings
	EnableMLCompression  bool
	EnableDeltaEncoding  bool
	EnableCRDT           bool

	// Delta encoding config (reuse from v1)
	DeltaConfig *compression.DeltaEncodingConfig

	// Baseline management
	BaselineRefreshInterval time.Duration
	MaxBaselines            int

	// Compression selector config
	SelectorConfig *SelectorConfig

	// Memory limits
	MaxMemoryUsage int64
}

// BaselineV3 represents a baseline with v3 enhancements
type BaselineV3 struct {
	ID        string
	Data      []byte
	Hash      []byte
	Timestamp time.Time
	Version   uint64
	Size      int64

	// CRDT metadata
	CRDTSynced   bool
	ConflictFree bool

	// Usage tracking
	UsageCount   int32
	LastAccessed time.Time
}

// HDEv3Metrics tracks performance metrics
type HDEv3Metrics struct {
	// Compression metrics
	TotalCompressed   atomic.Int64
	TotalDecompressed atomic.Int64
	BytesOriginal     atomic.Int64
	BytesCompressed   atomic.Int64

	// Delta metrics
	DeltaHits   atomic.Int64
	DeltaMisses atomic.Int64

	// CRDT metrics
	CRDTMerges    atomic.Int64
	CRDTConflicts atomic.Int64

	// Algorithm usage
	AlgorithmCounts map[CompressionAlgorithm]*atomic.Int64
	mu              sync.RWMutex
}

// DefaultHDEv3Config returns default configuration
func DefaultHDEv3Config(nodeID string) *HDEv3Config {
	return &HDEv3Config{
		NodeID:                  nodeID,
		NetworkMode:             upgrade.ModeHybrid,
		EnableMLCompression:     true,
		EnableDeltaEncoding:     true,
		EnableCRDT:              true,
		DeltaConfig:             compression.DefaultDeltaEncodingConfig(),
		BaselineRefreshInterval: 5 * time.Minute,
		MaxBaselines:            1000,
		SelectorConfig:          DefaultSelectorConfig(),
		MaxMemoryUsage:          2 * 1024 * 1024 * 1024, // 2GB
	}
}

// NewHDEv3 creates a new HDE v3 instance
func NewHDEv3(config *HDEv3Config) (*HDEv3, error) {
	if config == nil {
		config = DefaultHDEv3Config("default-node")
	}

	ctx, cancel := context.WithCancel(context.Background())

	hde := &HDEv3{
		config:     config,
		baselines:  make(map[string]*BaselineV3),
		metrics:    newHDEv3Metrics(),
		ctx:        ctx,
		cancel:     cancel,
	}

	// Initialize ML compression selector
	if config.EnableMLCompression {
		hde.compSelector = NewCompressionSelector(config.NetworkMode, config.SelectorConfig)
	}

	// Initialize CRDT integration
	if config.EnableCRDT {
		hde.crdtIntegration = NewCRDTIntegration(config.NodeID)
	}

	// Initialize delta encoder (reuse v1 implementation)
	if config.EnableDeltaEncoding {
		var err error
		hde.deltaEncoder, err = compression.NewDeltaEncoder(config.DeltaConfig, nil)
		if err != nil {
			return nil, fmt.Errorf("failed to create delta encoder: %w", err)
		}
	}

	// Initialize compression engines
	if err := hde.initCompressionEngines(); err != nil {
		return nil, fmt.Errorf("failed to initialize compression engines: %w", err)
	}

	// Start background tasks
	go hde.baselineCleanupLoop()

	return hde, nil
}

// initCompressionEngines initializes all compression engines
func (hde *HDEv3) initCompressionEngines() error {
	var err error

	// Zstd balanced encoder
	hde.zstdEncoder, err = zstd.NewWriter(nil,
		zstd.WithEncoderLevel(zstd.EncoderLevelFromZstd(3)),
		zstd.WithEncoderConcurrency(4),
	)
	if err != nil {
		return fmt.Errorf("failed to create zstd encoder: %w", err)
	}

	// Zstd max compression encoder
	hde.zstdMaxEncoder, err = zstd.NewWriter(nil,
		zstd.WithEncoderLevel(zstd.EncoderLevelFromZstd(19)),
		zstd.WithEncoderConcurrency(4),
	)
	if err != nil {
		return fmt.Errorf("failed to create zstd max encoder: %w", err)
	}

	// Zstd decoder
	hde.zstdDecoder, err = zstd.NewReader(nil,
		zstd.WithDecoderConcurrency(4),
	)
	if err != nil {
		return fmt.Errorf("failed to create zstd decoder: %w", err)
	}

	return nil
}

// Compress compresses data with ML-based algorithm selection
func (hde *HDEv3) Compress(vmID string, data []byte) (*CompressedDataV3, error) {
	if len(data) == 0 {
		return nil, errors.New("no data to compress")
	}

	startTime := time.Now()
	originalSize := len(data)
	hde.metrics.BytesOriginal.Add(int64(originalSize))

	// Step 1: Delta encoding (if enabled)
	var deltaData []byte
	var isDelta bool
	baselineKey := fmt.Sprintf("%s_state", vmID)

	if hde.config.EnableDeltaEncoding && hde.deltaEncoder != nil {
		encoded, err := hde.deltaEncoder.Encode(baselineKey, data)
		if err == nil && encoded.IsDelta {
			deltaData = encoded.Data
			isDelta = true
			hde.metrics.DeltaHits.Add(1)
		} else {
			deltaData = data
			hde.metrics.DeltaMisses.Add(1)
		}
	} else {
		deltaData = data
	}

	// Step 2: ML-based compression selection
	var algo CompressionAlgorithm
	if hde.config.EnableMLCompression && hde.compSelector != nil {
		algo = hde.compSelector.SelectCompression(deltaData, hde.config.NetworkMode)
	} else {
		// Fallback: mode-based selection
		algo = hde.selectByMode(len(deltaData))
	}

	// Step 3: Compress with selected algorithm
	compressed, err := hde.compressWithAlgorithm(deltaData, algo)
	if err != nil {
		return nil, fmt.Errorf("compression failed: %w", err)
	}

	duration := time.Since(startTime)

	// Step 4: Record performance for ML learning
	if hde.compSelector != nil {
		hde.compSelector.RecordPerformance(algo, len(deltaData), len(compressed), duration)
	}

	// Step 5: Update metrics
	hde.metrics.BytesCompressed.Add(int64(len(compressed)))
	hde.metrics.TotalCompressed.Add(1)
	hde.recordAlgorithmUse(algo)

	// Step 6: Register with CRDT if enabled
	if hde.config.EnableCRDT && hde.crdtIntegration != nil {
		hash := sha256.Sum256(data)
		hde.crdtIntegration.RegisterBaseline(baselineKey, hash[:], int64(originalSize))
	}

	// Create compressed data structure
	result := &CompressedDataV3{
		Data:               compressed,
		OriginalSize:       originalSize,
		CompressedSize:     len(compressed),
		IsDelta:            isDelta,
		Algorithm:          algo,
		BaselineKey:        baselineKey,
		Timestamp:          time.Now(),
		CompressionTime:    duration,
		NetworkMode:        hde.config.NetworkMode,
	}

	return result, nil
}

// Decompress decompresses data compressed with HDE v3
func (hde *HDEv3) Decompress(compressed *CompressedDataV3) ([]byte, error) {
	if compressed == nil || len(compressed.Data) == 0 {
		return nil, errors.New("no data to decompress")
	}

	// Step 1: Decompress with algorithm
	decompressed, err := hde.decompressWithAlgorithm(compressed.Data, compressed.Algorithm)
	if err != nil {
		return nil, fmt.Errorf("decompression failed: %w", err)
	}

	// Step 2: Apply delta reconstruction if needed
	if compressed.IsDelta && hde.deltaEncoder != nil {
		encodedData := &compression.EncodedData{
			Data:           decompressed,
			IsDelta:        true,
			BaselineKey:    compressed.BaselineKey,
		}
		reconstructed, err := hde.deltaEncoder.Decode(compressed.BaselineKey, encodedData)
		if err != nil {
			return nil, fmt.Errorf("delta reconstruction failed: %w", err)
		}
		decompressed = reconstructed
	}

	hde.metrics.TotalDecompressed.Add(1)

	return decompressed, nil
}

// compressWithAlgorithm compresses data with specified algorithm
func (hde *HDEv3) compressWithAlgorithm(data []byte, algo CompressionAlgorithm) ([]byte, error) {
	switch algo {
	case CompressionNone:
		return data, nil

	case CompressionLZ4:
		var buf bytes.Buffer
		writer := lz4.NewWriter(&buf)
		if _, err := writer.Write(data); err != nil {
			return nil, err
		}
		if err := writer.Close(); err != nil {
			return nil, err
		}
		return buf.Bytes(), nil

	case CompressionZstd:
		return hde.zstdEncoder.EncodeAll(data, nil), nil

	case CompressionZstdMax:
		return hde.zstdMaxEncoder.EncodeAll(data, nil), nil

	default:
		return hde.zstdEncoder.EncodeAll(data, nil), nil
	}
}

// decompressWithAlgorithm decompresses data with specified algorithm
func (hde *HDEv3) decompressWithAlgorithm(data []byte, algo CompressionAlgorithm) ([]byte, error) {
	switch algo {
	case CompressionNone:
		return data, nil

	case CompressionLZ4:
		reader := lz4.NewReader(bytes.NewReader(data))
		var buf bytes.Buffer
		if _, err := buf.ReadFrom(reader); err != nil {
			return nil, err
		}
		return buf.Bytes(), nil

	case CompressionZstd, CompressionZstdMax:
		return hde.zstdDecoder.DecodeAll(data, nil)

	default:
		return hde.zstdDecoder.DecodeAll(data, nil)
	}
}

// selectByMode selects algorithm based on network mode (fallback)
func (hde *HDEv3) selectByMode(dataSize int) CompressionAlgorithm {
	switch hde.config.NetworkMode {
	case upgrade.ModeDatacenter:
		if dataSize < 100*1024 {
			return CompressionLZ4
		}
		return CompressionZstd

	case upgrade.ModeInternet:
		return CompressionZstdMax

	default:
		return CompressionZstd
	}
}

// UpdateNetworkMode updates the network mode for adaptive compression
func (hde *HDEv3) UpdateNetworkMode(mode upgrade.NetworkMode) {
	hde.mu.Lock()
	defer hde.mu.Unlock()

	hde.config.NetworkMode = mode
	if hde.compSelector != nil {
		hde.compSelector.UpdateMode(mode)
	}
}

// MergeRemoteCRDT merges remote CRDT state for conflict-free synchronization
func (hde *HDEv3) MergeRemoteCRDT(remoteData []byte) error {
	if hde.crdtIntegration == nil {
		return errors.New("CRDT integration not enabled")
	}

	if err := hde.crdtIntegration.MergeRemoteState(remoteData); err != nil {
		return err
	}

	hde.metrics.CRDTMerges.Add(1)
	return nil
}

// ExportCRDTState exports CRDT state for synchronization
func (hde *HDEv3) ExportCRDTState() ([]byte, error) {
	if hde.crdtIntegration == nil {
		return nil, errors.New("CRDT integration not enabled")
	}

	return hde.crdtIntegration.ExportState()
}

// baselineCleanupLoop periodically cleans up old baselines
func (hde *HDEv3) baselineCleanupLoop() {
	ticker := time.NewTicker(hde.config.BaselineRefreshInterval)
	defer ticker.Stop()

	for {
		select {
		case <-hde.ctx.Done():
			return
		case <-ticker.C:
			hde.cleanupBaselines()
		}
	}
}

// cleanupBaselines removes old unused baselines
func (hde *HDEv3) cleanupBaselines() {
	hde.baselineMu.Lock()
	defer hde.baselineMu.Unlock()

	if len(hde.baselines) <= hde.config.MaxBaselines {
		return
	}

	// Remove oldest baselines
	cutoff := time.Now().Add(-hde.config.BaselineRefreshInterval * 2)
	for id, baseline := range hde.baselines {
		if baseline.Timestamp.Before(cutoff) && atomic.LoadInt32(&baseline.UsageCount) == 0 {
			delete(hde.baselines, id)
		}
	}

	// Cleanup delta encoder baselines
	if hde.deltaEncoder != nil {
		hde.deltaEncoder.PruneOldBaselines()
	}
}

// recordAlgorithmUse records algorithm usage for metrics
func (hde *HDEv3) recordAlgorithmUse(algo CompressionAlgorithm) {
	hde.metrics.mu.Lock()
	defer hde.metrics.mu.Unlock()

	if counter, exists := hde.metrics.AlgorithmCounts[algo]; exists {
		counter.Add(1)
	}
}

// GetMetrics returns comprehensive metrics
func (hde *HDEv3) GetMetrics() map[string]interface{} {
	metrics := make(map[string]interface{})

	// Basic metrics
	metrics["total_compressed"] = hde.metrics.TotalCompressed.Load()
	metrics["total_decompressed"] = hde.metrics.TotalDecompressed.Load()
	metrics["bytes_original"] = hde.metrics.BytesOriginal.Load()
	metrics["bytes_compressed"] = hde.metrics.BytesCompressed.Load()

	// Compression ratio
	original := hde.metrics.BytesOriginal.Load()
	compressed := hde.metrics.BytesCompressed.Load()
	if compressed > 0 {
		metrics["compression_ratio"] = float64(original) / float64(compressed)
		metrics["compression_percent"] = (1.0 - float64(compressed)/float64(original)) * 100.0
	}

	// Delta metrics
	deltaHits := hde.metrics.DeltaHits.Load()
	deltaMisses := hde.metrics.DeltaMisses.Load()
	if deltaHits+deltaMisses > 0 {
		metrics["delta_hit_rate"] = float64(deltaHits) / float64(deltaHits+deltaMisses) * 100.0
	}

	// CRDT metrics
	if hde.crdtIntegration != nil {
		crdtStats := hde.crdtIntegration.GetStats()
		for k, v := range crdtStats {
			metrics["crdt_"+k] = v
		}
	}

	// Compression selector metrics
	if hde.compSelector != nil {
		selectorStats := hde.compSelector.GetStats()
		for k, v := range selectorStats {
			metrics["selector_"+k] = v
		}
	}

	// Algorithm usage
	algoUsage := make(map[string]int64)
	hde.metrics.mu.RLock()
	for algo, counter := range hde.metrics.AlgorithmCounts {
		algoUsage[algo.String()] = counter.Load()
	}
	hde.metrics.mu.RUnlock()
	metrics["algorithm_usage"] = algoUsage

	return metrics
}

// Close releases all resources
func (hde *HDEv3) Close() error {
	hde.cancel()

	if hde.zstdEncoder != nil {
		hde.zstdEncoder.Close()
	}
	if hde.zstdMaxEncoder != nil {
		hde.zstdMaxEncoder.Close()
	}
	if hde.zstdDecoder != nil {
		hde.zstdDecoder.Close()
	}
	if hde.deltaEncoder != nil {
		hde.deltaEncoder.Close()
	}
	if hde.compSelector != nil {
		hde.compSelector.Close()
	}

	return nil
}

// CompressedDataV3 represents compressed data with v3 metadata
type CompressedDataV3 struct {
	Data            []byte
	OriginalSize    int
	CompressedSize  int
	IsDelta         bool
	Algorithm       CompressionAlgorithm
	BaselineKey     string
	Timestamp       time.Time
	CompressionTime time.Duration
	NetworkMode     upgrade.NetworkMode
}

// CompressionRatio returns the compression ratio
func (cd *CompressedDataV3) CompressionRatio() float64 {
	if cd.CompressedSize == 0 {
		return 0
	}
	return float64(cd.OriginalSize) / float64(cd.CompressedSize)
}

// Marshal serializes compressed data to bytes
func (cd *CompressedDataV3) Marshal() []byte {
	// Header: [version:1][isDelta:1][algo:1][mode:1][origSize:4][compSize:4][data...]
	header := make([]byte, 12)
	header[0] = 3 // Version 3
	if cd.IsDelta {
		header[1] = 1
	}
	header[2] = byte(cd.Algorithm)
	header[3] = byte(cd.NetworkMode)
	binary.BigEndian.PutUint32(header[4:8], uint32(cd.OriginalSize))
	binary.BigEndian.PutUint32(header[8:12], uint32(cd.CompressedSize))

	result := make([]byte, len(header)+len(cd.Data))
	copy(result, header)
	copy(result[len(header):], cd.Data)

	return result
}

// UnmarshalCompressedDataV3 deserializes compressed data from bytes
func UnmarshalCompressedDataV3(data []byte) (*CompressedDataV3, error) {
	if len(data) < 12 {
		return nil, errors.New("invalid compressed data: too short")
	}

	version := data[0]
	if version != 3 {
		return nil, fmt.Errorf("unsupported version: %d", version)
	}

	cd := &CompressedDataV3{
		IsDelta:        data[1] == 1,
		Algorithm:      CompressionAlgorithm(data[2]),
		NetworkMode:    upgrade.NetworkMode(data[3]),
		OriginalSize:   int(binary.BigEndian.Uint32(data[4:8])),
		CompressedSize: int(binary.BigEndian.Uint32(data[8:12])),
		Data:           data[12:],
		Timestamp:      time.Now(),
	}

	return cd, nil
}

// Helper: Create new metrics
func newHDEv3Metrics() *HDEv3Metrics {
	metrics := &HDEv3Metrics{
		AlgorithmCounts: make(map[CompressionAlgorithm]*atomic.Int64),
	}

	// Initialize algorithm counters
	for algo := CompressionNone; algo <= CompressionBrotli; algo++ {
		counter := &atomic.Int64{}
		metrics.AlgorithmCounts[algo] = counter
	}

	return metrics
}
