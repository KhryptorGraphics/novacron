// Package dwcp implements the Distributed WAN Communication Protocol for NovaCron
package dwcp

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
)

// HDE (Hierarchical Delta Encoding) provides multi-level delta compression
// optimized for VM memory and disk state transfers
type HDE struct {
	// Configuration
	config HDEConfig

	// Compression engines
	encoders map[CompressionLevel]*zstd.Encoder
	decoders map[CompressionLevel]*zstd.Decoder

	// Delta tracking
	baselines    map[string]*Baseline
	deltaTracker *DeltaTracker

	// Dictionary management
	dictionaries map[string][]byte
	dictMu       sync.RWMutex

	// Performance metrics
	compressionRatio atomic.Value // float64
	bytesCompressed  atomic.Int64
	bytesOriginal    atomic.Int64
	deltaHitRate     atomic.Value // float64

	// Synchronization
	mu     sync.RWMutex
	ctx    context.Context
	cancel context.CancelFunc
}

// HDEConfig contains configuration for Hierarchical Delta Encoding
type HDEConfig struct {
	// Compression levels by tier
	LocalLevel    int // Tier 1: Intra-cluster compression (default: 0 - fastest)
	RegionalLevel int // Tier 2: Inter-cluster compression (default: 3 - balanced)
	GlobalLevel   int // Tier 3: WAN compression (default: 9 - best compression)

	// Delta encoding settings
	EnableDelta     bool    // Enable delta encoding (default: true)
	BlockSize       int     // Block size for delta comparison (default: 4KB)
	MaxDeltaHistory int     // Maximum delta history entries (default: 100)
	DeltaThreshold  float64 // Minimum similarity for delta encoding (default: 0.7)

	// Dictionary settings
	EnableDictionary bool // Enable dictionary compression (default: true)
	DictSize         int  // Dictionary size in KB (default: 1024)
	TrainingSamples  int  // Number of samples for dictionary training (default: 100)

	// Memory management
	MaxMemoryUsage  int64         // Maximum memory for caching (default: 1GB)
	CleanupInterval time.Duration // Cleanup interval (default: 5 minutes)

	// Quantization for numerical data
	EnableQuantization bool // Enable quantization (default: true)
	QuantizationBits   int  // Bits for quantization (default: 16)
}

// HDECompressionLevel represents the compression level for different tiers.
// It is an alias of CompressionLevel so HDE compression tiers integrate cleanly
// with the global DWCP compression metrics.
type HDECompressionLevel = CompressionLevel

const (
	CompressionLocal    HDECompressionLevel = iota // Tier 1: Local/Intra-cluster
	CompressionRegional                            // Tier 2: Regional/Inter-cluster
	CompressionGlobal                              // Tier 3: Global/WAN
)

// Baseline represents a reference state for delta encoding
type Baseline struct {
	ID         string
	Data       []byte
	Hash       []byte
	Timestamp  time.Time
	UsageCount int32
	Size       int64
	Blocks     map[int][]byte // Block-level data for efficient delta
}

// DeltaTracker tracks delta changes between baselines
type DeltaTracker struct {
	deltas     map[string]*Delta
	history    []string
	maxHistory int
	mu         sync.RWMutex
}

// Delta represents the difference between two states
type Delta struct {
	BaselineID string
	TargetID   string
	Operations []DeltaOperation
	Size       int64
	Timestamp  time.Time
}

// DeltaOperation represents a single delta operation
type DeltaOperation struct {
	Type   DeltaOpType
	Offset int64
	Length int64
	Data   []byte
}

// DeltaOpType represents the type of delta operation
type DeltaOpType int

const (
	DeltaOpCopy   DeltaOpType = iota // Copy from baseline
	DeltaOpAdd                       // Add new data
	DeltaOpModify                    // Modify existing data
)

// NewHDE creates a new Hierarchical Delta Encoding instance
func NewHDE(config HDEConfig) (*HDE, error) {
	// Set defaults
	if config.BlockSize <= 0 {
		config.BlockSize = 4 * 1024 // 4KB
	}
	if config.MaxDeltaHistory <= 0 {
		config.MaxDeltaHistory = 100
	}
	if config.DeltaThreshold <= 0 || config.DeltaThreshold > 1 {
		config.DeltaThreshold = 0.7
	}
	if config.DictSize <= 0 {
		config.DictSize = 1024 // 1MB
	}
	if config.TrainingSamples <= 0 {
		config.TrainingSamples = 100
	}
	if config.MaxMemoryUsage <= 0 {
		config.MaxMemoryUsage = 1024 * 1024 * 1024 // 1GB
	}
	if config.CleanupInterval <= 0 {
		config.CleanupInterval = 5 * time.Minute
	}
	if config.QuantizationBits <= 0 {
		config.QuantizationBits = 16
	}

	// Validate compression levels
	if config.LocalLevel < 0 || config.LocalLevel > 22 {
		config.LocalLevel = 0
	}
	if config.RegionalLevel < 0 || config.RegionalLevel > 22 {
		config.RegionalLevel = 3
	}
	if config.GlobalLevel < 0 || config.GlobalLevel > 22 {
		config.GlobalLevel = 9
	}

	ctx, cancel := context.WithCancel(context.Background())

	hde := &HDE{
		config:       config,
		encoders:     make(map[CompressionLevel]*zstd.Encoder),
		decoders:     make(map[CompressionLevel]*zstd.Decoder),
		baselines:    make(map[string]*Baseline),
		dictionaries: make(map[string][]byte),
		ctx:          ctx,
		cancel:       cancel,
	}

	// Initialize compression ratio
	hde.compressionRatio.Store(float64(1.0))
	hde.deltaHitRate.Store(float64(0.0))

	// Create encoders for each compression level
	levels := map[CompressionLevel]int{
		CompressionLocal:    config.LocalLevel,
		CompressionRegional: config.RegionalLevel,
		CompressionGlobal:   config.GlobalLevel,
	}

	for level, compressionLevel := range levels {
		encoder, err := zstd.NewWriter(nil,
			zstd.WithEncoderLevel(zstd.EncoderLevelFromZstd(compressionLevel)),
			zstd.WithEncoderConcurrency(4),
		)
		if err != nil {
			return nil, fmt.Errorf("failed to create encoder for level %d: %w", level, err)
		}
		hde.encoders[level] = encoder

		decoder, err := zstd.NewReader(nil,
			zstd.WithDecoderConcurrency(4),
			zstd.WithDecoderMaxMemory(uint64(config.MaxMemoryUsage/4)),
		)
		if err != nil {
			return nil, fmt.Errorf("failed to create decoder for level %d: %w", level, err)
		}
		hde.decoders[level] = decoder
	}

	// Create delta tracker
	if config.EnableDelta {
		hde.deltaTracker = &DeltaTracker{
			deltas:     make(map[string]*Delta),
			history:    make([]string, 0, config.MaxDeltaHistory),
			maxHistory: config.MaxDeltaHistory,
		}
	}

	// Start cleanup routine
	go hde.cleanupLoop()

	return hde, nil
}

// CompressMemory compresses VM memory state with delta encoding
func (hde *HDE) CompressMemory(vmID string, memoryData []byte, tier CompressionLevel) ([]byte, error) {
	if len(memoryData) == 0 {
		return nil, errors.New("no memory data to compress")
	}

	// Update original size metric
	hde.bytesOriginal.Add(int64(len(memoryData)))

	// Check for existing baseline
	var deltaEncoded []byte
	if hde.config.EnableDelta {
		baseline, exists := hde.getBaseline(vmID + "_memory")
		if exists {
			// Compute delta
			delta, err := hde.computeDelta(baseline, memoryData)
			if err == nil && hde.isDeltaEfficient(delta, len(memoryData)) {
				// Encode delta
				deltaEncoded, err = hde.encodeDelta(delta)
				if err == nil {
					// Update delta hit rate
					hde.updateDeltaHitRate(true)
				}
			}
		}

		// Create/update baseline for next delta
		hde.createBaseline(vmID+"_memory", memoryData)
	}

	// Choose data to compress
	dataToCompress := memoryData
	if deltaEncoded != nil && len(deltaEncoded) < len(memoryData)/2 {
		// Use delta if it's significantly smaller
		dataToCompress = deltaEncoded
	} else {
		hde.updateDeltaHitRate(false)
	}

	// Apply quantization for numerical data if enabled
	if hde.config.EnableQuantization && tier == CompressionGlobal {
		dataToCompress = hde.quantize(dataToCompress)
	}

	// Compress with appropriate level
	compressed, err := hde.compress(dataToCompress, tier)
	if err != nil {
		return nil, fmt.Errorf("compression failed: %w", err)
	}

	// Update metrics
	hde.bytesCompressed.Add(int64(len(compressed)))
	hde.updateCompressionRatio()

	// Create packet with metadata
	packet := hde.createPacket(compressed, deltaEncoded != nil, tier)

	return packet, nil
}

// CompressDisk compresses VM disk blocks with delta encoding
func (hde *HDE) CompressDisk(vmID string, diskData []byte, blockID int, tier CompressionLevel) ([]byte, error) {
	if len(diskData) == 0 {
		return nil, errors.New("no disk data to compress")
	}

	// Update original size metric
	hde.bytesOriginal.Add(int64(len(diskData)))

	baselineKey := fmt.Sprintf("%s_disk_%d", vmID, blockID)

	// Check for existing baseline
	var deltaEncoded []byte
	if hde.config.EnableDelta {
		baseline, exists := hde.getBaseline(baselineKey)
		if exists {
			// Compute delta
			delta, err := hde.computeDelta(baseline, diskData)
			if err == nil && hde.isDeltaEfficient(delta, len(diskData)) {
				// Encode delta
				deltaEncoded, err = hde.encodeDelta(delta)
				if err == nil {
					hde.updateDeltaHitRate(true)
				}
			}
		}

		// Create/update baseline for next delta
		hde.createBaseline(baselineKey, diskData)
	}

	// Choose data to compress
	dataToCompress := diskData
	if deltaEncoded != nil && len(deltaEncoded) < len(diskData)/2 {
		dataToCompress = deltaEncoded
	} else {
		hde.updateDeltaHitRate(false)
	}

	// Use dictionary compression if available
	if hde.config.EnableDictionary {
		dict, exists := hde.getDictionary(vmID + "_disk")
		if exists {
			// Compress with dictionary
			compressed, err := hde.compressWithDict(dataToCompress, dict, tier)
			if err == nil {
				dataToCompress = compressed
			}
		}
	}

	// Compress with appropriate level
	compressed, err := hde.compress(dataToCompress, tier)
	if err != nil {
		return nil, fmt.Errorf("compression failed: %w", err)
	}

	// Update metrics
	hde.bytesCompressed.Add(int64(len(compressed)))
	hde.updateCompressionRatio()

	// Create packet with metadata
	packet := hde.createPacket(compressed, deltaEncoded != nil, tier)

	return packet, nil
}

// Decompress decompresses data compressed with HDE
func (hde *HDE) Decompress(data []byte) ([]byte, error) {
	if len(data) < 16 {
		return nil, errors.New("invalid compressed data: too short")
	}

	// Parse packet header
	isDelta := data[0] == 1
	tier := CompressionLevel(data[1])
	dataSize := binary.BigEndian.Uint64(data[8:16])

	// Currently we don't enforce dataSize, but keep parsing it for forward
	// compatibility with richer packet formats.
	_ = dataSize

	// Extract compressed data
	compressedData := data[16:]

	// Decompress
	decompressed, err := hde.decompress(compressedData, tier)
	if err != nil {
		return nil, fmt.Errorf("decompression failed: %w", err)
	}

	// If it's delta encoded, apply delta to baseline
	if isDelta {
		// This would need the baseline ID to be included in the packet
		// For now, return the decompressed delta
		// In production, we'd apply the delta to the baseline
	}

	return decompressed, nil
}

// compress compresses data with the specified compression level
func (hde *HDE) compress(data []byte, level HDECompressionLevel) ([]byte, error) {
	encoder, exists := hde.encoders[level]
	if !exists {
		return nil, fmt.Errorf("encoder not found for level %d", level)
	}

	return encoder.EncodeAll(data, nil), nil
}

// decompress decompresses data with the specified compression level
func (hde *HDE) decompress(data []byte, level HDECompressionLevel) ([]byte, error) {
	decoder, exists := hde.decoders[level]
	if !exists {
		return nil, fmt.Errorf("decoder not found for level %d", level)
	}

	return decoder.DecodeAll(data, nil)
}

// compressWithDict compresses data using a dictionary
func (hde *HDE) compressWithDict(data []byte, dict []byte, level HDECompressionLevel) ([]byte, error) {
	// Create encoder with dictionary
	encoder, err := zstd.NewWriter(nil,
		zstd.WithEncoderLevel(zstd.EncoderLevelFromZstd(hde.getCompressionLevel(level))),
		zstd.WithEncoderDict(dict),
	)
	if err != nil {
		return nil, err
	}
	defer encoder.Close()

	return encoder.EncodeAll(data, nil), nil
}

// computeDelta computes the delta between baseline and new data
func (hde *HDE) computeDelta(baseline *Baseline, newData []byte) (*Delta, error) {
	delta := &Delta{
		BaselineID: baseline.ID,
		TargetID:   generateID(newData),
		Operations: make([]DeltaOperation, 0),
		Timestamp:  time.Now(),
	}

	blockSize := hde.config.BlockSize
	numBlocks := (len(newData) + blockSize - 1) / blockSize

	for i := 0; i < numBlocks; i++ {
		start := i * blockSize
		end := start + blockSize
		if end > len(newData) {
			end = len(newData)
		}

		newBlock := newData[start:end]

		// Check if block exists in baseline
		if baselineBlock, exists := baseline.Blocks[i]; exists {
			if bytes.Equal(baselineBlock, newBlock) {
				// Block unchanged - copy from baseline
				delta.Operations = append(delta.Operations, DeltaOperation{
					Type:   DeltaOpCopy,
					Offset: int64(start),
					Length: int64(len(newBlock)),
				})
			} else {
				// Block modified
				delta.Operations = append(delta.Operations, DeltaOperation{
					Type:   DeltaOpModify,
					Offset: int64(start),
					Length: int64(len(newBlock)),
					Data:   newBlock,
				})
				delta.Size += int64(len(newBlock))
			}
		} else {
			// New block
			delta.Operations = append(delta.Operations, DeltaOperation{
				Type:   DeltaOpAdd,
				Offset: int64(start),
				Length: int64(len(newBlock)),
				Data:   newBlock,
			})
			delta.Size += int64(len(newBlock))
		}
	}

	return delta, nil
}

// encodeDelta encodes a delta into bytes
func (hde *HDE) encodeDelta(delta *Delta) ([]byte, error) {
	var buf bytes.Buffer

	// Write header
	buf.Write([]byte(delta.BaselineID))
	buf.Write([]byte{0}) // Separator

	// Write number of operations
	binary.Write(&buf, binary.BigEndian, uint32(len(delta.Operations)))

	// Write operations
	for _, op := range delta.Operations {
		buf.WriteByte(byte(op.Type))
		binary.Write(&buf, binary.BigEndian, op.Offset)
		binary.Write(&buf, binary.BigEndian, op.Length)

		if op.Type != DeltaOpCopy {
			binary.Write(&buf, binary.BigEndian, uint32(len(op.Data)))
			buf.Write(op.Data)
		}
	}

	return buf.Bytes(), nil
}

// isDeltaEfficient checks if delta encoding is efficient
func (hde *HDE) isDeltaEfficient(delta *Delta, originalSize int) bool {
	// Delta is efficient if it's less than threshold of original size
	return float64(delta.Size) < float64(originalSize)*hde.config.DeltaThreshold
}

// createBaseline creates or updates a baseline
func (hde *HDE) createBaseline(id string, data []byte) {
	hde.mu.Lock()
	defer hde.mu.Unlock()

	// Create blocks
	blockSize := hde.config.BlockSize
	numBlocks := (len(data) + blockSize - 1) / blockSize
	blocks := make(map[int][]byte)

	for i := 0; i < numBlocks; i++ {
		start := i * blockSize
		end := start + blockSize
		if end > len(data) {
			end = len(data)
		}
		blocks[i] = data[start:end]
	}

	baseline := &Baseline{
		ID:        id,
		Data:      data,
		Hash:      hashData(data),
		Timestamp: time.Now(),
		Size:      int64(len(data)),
		Blocks:    blocks,
	}

	hde.baselines[id] = baseline
}

// getBaseline retrieves a baseline
func (hde *HDE) getBaseline(id string) (*Baseline, bool) {
	hde.mu.RLock()
	defer hde.mu.RUnlock()

	baseline, exists := hde.baselines[id]
	if exists {
		atomic.AddInt32(&baseline.UsageCount, 1)
	}
	return baseline, exists
}

// getDictionary retrieves a dictionary
func (hde *HDE) getDictionary(id string) ([]byte, bool) {
	hde.dictMu.RLock()
	defer hde.dictMu.RUnlock()

	dict, exists := hde.dictionaries[id]
	return dict, exists
}

// TrainDictionary trains a compression dictionary from sample data
func (hde *HDE) TrainDictionary(id string, samples [][]byte) error {
	if len(samples) == 0 {
		return errors.New("no samples provided for dictionary training")
	}

	// Concatenate samples
	var totalSize int
	for _, sample := range samples {
		totalSize += len(sample)
	}

	trainingData := make([]byte, 0, totalSize)
	for _, sample := range samples {
		trainingData = append(trainingData, sample...)
	}

	// Train dictionary using zstd. We mirror the production dictionary trainer
	// in backend/core/network/dwcp/compression/dictionary_trainer.go.
	dict, err := zstd.BuildDict(zstd.BuildDictOptions{
		Contents: samples,
	})
	if err != nil {
		return fmt.Errorf("dictionary training failed: %w", err)
	}

	// Store dictionary
	hde.dictMu.Lock()
	hde.dictionaries[id] = dict
	hde.dictMu.Unlock()

	return nil
}

// quantize applies quantization to reduce precision of numerical data
func (hde *HDE) quantize(data []byte) []byte {
	// Simple quantization - reduce precision of floating point values
	// This is a simplified implementation
	// In production, would need more sophisticated quantization

	quantized := make([]byte, len(data))
	copy(quantized, data)

	// Apply quantization based on configured bits
	mask := byte(0xFF << (8 - hde.config.QuantizationBits/8))
	for i := range quantized {
		quantized[i] &= mask
	}

	return quantized
}

// createPacket creates a packet with metadata
func (hde *HDE) createPacket(compressed []byte, isDelta bool, tier CompressionLevel) []byte {
	packet := make([]byte, 16+len(compressed))

	// Header: [isDelta:1][tier:1][reserved:6][dataSize:8][compressed data]
	if isDelta {
		packet[0] = 1
	}
	packet[1] = byte(tier)
	binary.BigEndian.PutUint64(packet[8:16], uint64(len(compressed)))

	copy(packet[16:], compressed)

	return packet
}

// updateCompressionRatio updates the compression ratio metric
func (hde *HDE) updateCompressionRatio() {
	original := hde.bytesOriginal.Load()
	compressed := hde.bytesCompressed.Load()

	if compressed > 0 {
		ratio := float64(original) / float64(compressed)
		hde.compressionRatio.Store(ratio)
	}
}

// updateDeltaHitRate updates the delta hit rate metric
func (hde *HDE) updateDeltaHitRate(hit bool) {
	// Simple exponential moving average
	currentRate := hde.deltaHitRate.Load().(float64)
	if hit {
		newRate := currentRate*0.9 + 0.1
		hde.deltaHitRate.Store(newRate)
	} else {
		newRate := currentRate * 0.9
		hde.deltaHitRate.Store(newRate)
	}
}

// getCompressionLevel returns the zstd compression level for a tier
func (hde *HDE) getCompressionLevel(tier CompressionLevel) int {
	switch tier {
	case CompressionLocal:
		return hde.config.LocalLevel
	case CompressionRegional:
		return hde.config.RegionalLevel
	case CompressionGlobal:
		return hde.config.GlobalLevel
	default:
		return 3 // Default balanced
	}
}

// cleanupLoop periodically cleans up old baselines and deltas
func (hde *HDE) cleanupLoop() {
	ticker := time.NewTicker(hde.config.CleanupInterval)
	defer ticker.Stop()

	for {
		select {
		case <-hde.ctx.Done():
			return
		case <-ticker.C:
			hde.cleanup()
		}
	}
}

// cleanup removes old baselines and deltas
func (hde *HDE) cleanup() {
	hde.mu.Lock()
	defer hde.mu.Unlock()

	now := time.Now()
	cutoff := now.Add(-hde.config.CleanupInterval * 2)

	// Clean up old baselines
	for id, baseline := range hde.baselines {
		if baseline.Timestamp.Before(cutoff) && atomic.LoadInt32(&baseline.UsageCount) == 0 {
			delete(hde.baselines, id)
		}
	}

	// Clean up delta history
	if hde.deltaTracker != nil {
		hde.deltaTracker.cleanup(cutoff)
	}
}

// GetMetrics returns HDE metrics
func (hde *HDE) GetMetrics() map[string]interface{} {
	hde.mu.RLock()
	baselineCount := len(hde.baselines)
	hde.mu.RUnlock()

	hde.dictMu.RLock()
	dictCount := len(hde.dictionaries)
	hde.dictMu.RUnlock()

	return map[string]interface{}{
		"compression_ratio": hde.compressionRatio.Load(),
		"bytes_original":    hde.bytesOriginal.Load(),
		"bytes_compressed":  hde.bytesCompressed.Load(),
		"delta_hit_rate":    hde.deltaHitRate.Load(),
		"baseline_count":    baselineCount,
		"dictionary_count":  dictCount,
	}
}

// Close closes the HDE instance and releases resources
func (hde *HDE) Close() error {
	hde.cancel()

	// Close encoders
	for _, encoder := range hde.encoders {
		encoder.Close()
	}

	// Close decoders
	for _, decoder := range hde.decoders {
		decoder.Close()
	}

	return nil
}

// Helper functions

// hashData computes SHA256 hash of data
func hashData(data []byte) []byte {
	hash := sha256.Sum256(data)
	return hash[:]
}

// generateID generates a unique ID from data
func generateID(data []byte) string {
	hash := sha256.Sum256(data)
	return fmt.Sprintf("%x", hash[:8])
}

// cleanup removes old deltas from the tracker
func (dt *DeltaTracker) cleanup(cutoff time.Time) {
	dt.mu.Lock()
	defer dt.mu.Unlock()

	// Remove old deltas
	for id, delta := range dt.deltas {
		if delta.Timestamp.Before(cutoff) {
			delete(dt.deltas, id)
		}
	}

	// Trim history
	if len(dt.history) > dt.maxHistory {
		dt.history = dt.history[len(dt.history)-dt.maxHistory:]
	}
}

// Start initializes and starts the HDE compression layer
// Implements the Lifecycle interface
func (h *HDE) Start(ctx context.Context) error {
	h.mu.Lock()
	defer h.mu.Unlock()

	if h.ctx != nil {
		return fmt.Errorf("HDE already started")
	}

	// Create context for lifecycle management
	h.ctx, h.cancel = context.WithCancel(ctx)

	// Initialize compression engines if not already done
	if h.encoders == nil {
		h.encoders = make(map[CompressionLevel]*zstd.Encoder)
	}
	if h.decoders == nil {
		h.decoders = make(map[CompressionLevel]*zstd.Decoder)
	}
	if h.baselines == nil {
		h.baselines = make(map[string]*Baseline)
	}
	if h.dictionaries == nil {
		h.dictionaries = make(map[string][]byte)
	}

	// Initialize delta tracker if enabled
	if h.config.EnableDelta && h.deltaTracker == nil {
		h.deltaTracker = &DeltaTracker{
			deltas:     make(map[string]*Delta),
			history:    make([]string, 0, h.config.MaxDeltaHistory),
			maxHistory: h.config.MaxDeltaHistory,
		}
	}

	return nil
}

// Stop gracefully shuts down the HDE compression layer
// Implements the Lifecycle interface
func (h *HDE) Stop() error {
	h.mu.Lock()
	defer h.mu.Unlock()

	if h.cancel != nil {
		h.cancel()
	}

	// Close all encoders
	for _, encoder := range h.encoders {
		if encoder != nil {
			encoder.Close()
		}
	}

	// Close all decoders
	for _, decoder := range h.decoders {
		if decoder != nil {
			decoder.Close()
		}
	}

	// Clear maps to release memory
	h.encoders = nil
	h.decoders = nil
	h.baselines = nil
	h.dictionaries = nil

	h.ctx = nil
	h.cancel = nil

	return nil
}

// IsRunning returns true if the HDE compression layer is running
// Implements the Lifecycle interface
func (h *HDE) IsRunning() bool {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return h.ctx != nil
}
