package encoding

import (
	"context"
	"sync"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/upgrade"
)

// CompressionAlgorithm represents available compression algorithms
type CompressionAlgorithm int

const (
	CompressionNone CompressionAlgorithm = iota
	CompressionLZ4                       // Fast, moderate compression (datacenter mode)
	CompressionZstd                      // Balanced compression/speed
	CompressionZstdMax                   // Maximum compression (internet mode)
	CompressionBrotli                    // High compression for static data
)

// String returns the string representation of CompressionAlgorithm
func (ca CompressionAlgorithm) String() string {
	switch ca {
	case CompressionNone:
		return "none"
	case CompressionLZ4:
		return "lz4"
	case CompressionZstd:
		return "zstd"
	case CompressionZstdMax:
		return "zstd-max"
	case CompressionBrotli:
		return "brotli"
	default:
		return "unknown"
	}
}

// DataCharacteristics represents analyzed data characteristics
type DataCharacteristics struct {
	Size          int
	Entropy       float64 // 0.0 (uniform) to 1.0 (random)
	Compressible  bool
	RepeatPattern bool
	TextLike      bool
	BinaryLike    bool
}

// CompressionSelector provides ML-based compression algorithm selection
type CompressionSelector struct {
	mu sync.RWMutex

	// Current network mode
	mode upgrade.NetworkMode

	// Historical performance data
	algorithmStats map[CompressionAlgorithm]*AlgorithmStats

	// Configuration
	config *SelectorConfig

	// Adaptive learning
	learningEnabled bool
	ctx             context.Context
	cancel          context.CancelFunc
}

// AlgorithmStats tracks performance metrics for each algorithm
type AlgorithmStats struct {
	TotalBytes       int64
	CompressedBytes  int64
	TotalTime        time.Duration
	UseCount         int64
	AverageRatio     float64
	AverageSpeed     float64 // MB/s
	LastUsed         time.Time
}

// SelectorConfig configures the compression selector
type SelectorConfig struct {
	// Learning parameters
	LearningRate     float64
	AdaptiveEnabled  bool

	// Size thresholds
	SmallDataSize    int // < 1KB: minimal compression
	LargeDataSize    int // > 1MB: aggressive compression

	// Performance targets
	DatacenterTargetSpeed float64 // MB/s for datacenter mode
	InternetTargetRatio   float64 // Compression ratio for internet mode
}

// DefaultSelectorConfig returns default configuration
func DefaultSelectorConfig() *SelectorConfig {
	return &SelectorConfig{
		LearningRate:          0.1,
		AdaptiveEnabled:       true,
		SmallDataSize:         1024,       // 1KB
		LargeDataSize:         1024 * 1024, // 1MB
		DatacenterTargetSpeed: 1000.0,     // 1 GB/s
		InternetTargetRatio:   5.0,        // 5x compression
	}
}

// NewCompressionSelector creates a new ML-based compression selector
func NewCompressionSelector(mode upgrade.NetworkMode, config *SelectorConfig) *CompressionSelector {
	if config == nil {
		config = DefaultSelectorConfig()
	}

	ctx, cancel := context.WithCancel(context.Background())

	cs := &CompressionSelector{
		mode:            mode,
		algorithmStats:  make(map[CompressionAlgorithm]*AlgorithmStats),
		config:          config,
		learningEnabled: config.AdaptiveEnabled,
		ctx:             ctx,
		cancel:          cancel,
	}

	// Initialize stats for all algorithms
	for algo := CompressionNone; algo <= CompressionBrotli; algo++ {
		cs.algorithmStats[algo] = &AlgorithmStats{
			AverageRatio: 1.0,
			AverageSpeed: 100.0, // Default 100 MB/s
		}
	}

	return cs
}

// SelectCompression selects optimal compression based on data characteristics and network mode
func (cs *CompressionSelector) SelectCompression(data []byte, mode upgrade.NetworkMode) CompressionAlgorithm {
	cs.mu.RLock()
	defer cs.mu.RUnlock()

	// Update mode if changed
	if mode != cs.mode {
		cs.mode = mode
	}

	// Analyze data characteristics
	chars := cs.analyzeData(data)

	// Rule-based selection with mode awareness
	return cs.selectByMode(chars, mode)
}

// selectByMode selects algorithm based on network mode
func (cs *CompressionSelector) selectByMode(chars DataCharacteristics, mode upgrade.NetworkMode) CompressionAlgorithm {
	// Very small data: no compression overhead
	if chars.Size < cs.config.SmallDataSize {
		return CompressionNone
	}

	// Incompressible data: skip compression
	if !chars.Compressible || chars.Entropy > 0.9 {
		return CompressionNone
	}

	switch mode {
	case upgrade.ModeDatacenter:
		// Datacenter mode: Prioritize SPEED (CPU-efficient)
		// Low latency network, so compression time is critical
		if chars.Size < 100*1024 { // < 100KB
			return CompressionLZ4 // Fastest
		}
		return CompressionZstd // Balanced for larger data

	case upgrade.ModeInternet:
		// Internet mode: Prioritize BANDWIDTH (aggressive compression)
		// High latency network, so saving bytes is critical
		if chars.RepeatPattern || chars.TextLike {
			return CompressionZstdMax // Maximum compression for text
		}
		if chars.Size > cs.config.LargeDataSize {
			return CompressionZstdMax // Aggressive for large data
		}
		return CompressionZstd // Balanced compression

	case upgrade.ModeHybrid:
		// Hybrid mode: Adaptive selection based on learning
		if cs.learningEnabled {
			return cs.selectAdaptive(chars)
		}
		// Fallback to balanced
		return CompressionZstd

	default:
		return CompressionZstd
	}
}

// selectAdaptive uses ML-based selection with historical performance
func (cs *CompressionSelector) selectAdaptive(chars DataCharacteristics) CompressionAlgorithm {
	// Score each algorithm based on historical performance
	bestAlgo := CompressionZstd
	bestScore := 0.0

	candidates := []CompressionAlgorithm{
		CompressionLZ4,
		CompressionZstd,
		CompressionZstdMax,
	}

	for _, algo := range candidates {
		stats := cs.algorithmStats[algo]
		if stats.UseCount == 0 {
			continue
		}

		// Score = (compression_ratio * speed_factor)
		// Higher is better
		score := stats.AverageRatio * (stats.AverageSpeed / 100.0)

		// Adjust for data characteristics
		if chars.RepeatPattern && (algo == CompressionZstdMax) {
			score *= 1.5 // Favor max compression for repetitive data
		}
		if chars.Size < 10*1024 && algo == CompressionLZ4 {
			score *= 1.3 // Favor LZ4 for small data
		}

		if score > bestScore {
			bestScore = score
			bestAlgo = algo
		}
	}

	return bestAlgo
}

// analyzeData performs fast data analysis to determine characteristics
func (cs *CompressionSelector) analyzeData(data []byte) DataCharacteristics {
	chars := DataCharacteristics{
		Size: len(data),
	}

	if len(data) == 0 {
		return chars
	}

	// Sample-based entropy calculation (fast approximation)
	sampleSize := 256
	if len(data) < sampleSize {
		sampleSize = len(data)
	}

	// Simple entropy estimation using byte frequency
	freq := make([]int, 256)
	for i := 0; i < sampleSize; i++ {
		freq[data[i]]++
	}

	// Calculate Shannon entropy
	entropy := 0.0
	for _, count := range freq {
		if count > 0 {
			p := float64(count) / float64(sampleSize)
			entropy -= p * log2(p)
		}
	}
	chars.Entropy = entropy / 8.0 // Normalize to 0-1

	// Compressibility estimation
	chars.Compressible = chars.Entropy < 0.85

	// Detect repeat patterns (simple run-length check)
	repeatCount := 0
	for i := 1; i < sampleSize; i++ {
		if data[i] == data[i-1] {
			repeatCount++
		}
	}
	chars.RepeatPattern = float64(repeatCount)/float64(sampleSize) > 0.3

	// Detect text-like data (printable ASCII)
	printableCount := 0
	for i := 0; i < sampleSize; i++ {
		if data[i] >= 32 && data[i] <= 126 {
			printableCount++
		}
	}
	chars.TextLike = float64(printableCount)/float64(sampleSize) > 0.7
	chars.BinaryLike = !chars.TextLike

	return chars
}

// RecordPerformance records compression performance for learning
func (cs *CompressionSelector) RecordPerformance(
	algo CompressionAlgorithm,
	originalSize int,
	compressedSize int,
	duration time.Duration,
) {
	cs.mu.Lock()
	defer cs.mu.Unlock()

	stats := cs.algorithmStats[algo]
	if stats == nil {
		stats = &AlgorithmStats{}
		cs.algorithmStats[algo] = stats
	}

	// Update stats
	stats.TotalBytes += int64(originalSize)
	stats.CompressedBytes += int64(compressedSize)
	stats.TotalTime += duration
	stats.UseCount++
	stats.LastUsed = time.Now()

	// Calculate averages
	ratio := float64(originalSize) / float64(compressedSize)
	speed := float64(originalSize) / (1024 * 1024) / duration.Seconds() // MB/s

	// Exponential moving average
	alpha := cs.config.LearningRate
	stats.AverageRatio = alpha*ratio + (1-alpha)*stats.AverageRatio
	stats.AverageSpeed = alpha*speed + (1-alpha)*stats.AverageSpeed
}

// GetStats returns current algorithm statistics
func (cs *CompressionSelector) GetStats() map[string]interface{} {
	cs.mu.RLock()
	defer cs.mu.RUnlock()

	stats := make(map[string]interface{})
	stats["mode"] = cs.mode.String()
	stats["learning_enabled"] = cs.learningEnabled

	algoStats := make(map[string]interface{})
	for algo, stat := range cs.algorithmStats {
		if stat.UseCount > 0 {
			algoStats[algo.String()] = map[string]interface{}{
				"use_count":     stat.UseCount,
				"average_ratio": stat.AverageRatio,
				"average_speed": stat.AverageSpeed,
				"last_used":     stat.LastUsed,
			}
		}
	}
	stats["algorithms"] = algoStats

	return stats
}

// UpdateMode updates the network mode
func (cs *CompressionSelector) UpdateMode(mode upgrade.NetworkMode) {
	cs.mu.Lock()
	defer cs.mu.Unlock()
	cs.mode = mode
}

// Close releases resources
func (cs *CompressionSelector) Close() error {
	cs.cancel()
	return nil
}

// Helper functions

func log2(x float64) float64 {
	if x <= 0 {
		return 0
	}
	// Fast log2 approximation
	return 1.4426950408889634 * logApprox(x) // log2(x) = log(x) / log(2)
}

func logApprox(x float64) float64 {
	// Simple natural log approximation
	if x <= 0 {
		return 0
	}
	// Taylor series approximation around 1
	if x < 0.5 || x > 2.0 {
		// Use log properties for values outside [0.5, 2.0]
		exp := 0
		for x > 2.0 {
			x /= 2.0
			exp++
		}
		for x < 0.5 {
			x *= 2.0
			exp--
		}
		return float64(exp)*0.6931471805599453 + logApprox(x)
	}

	// Taylor series: ln(1+x) ≈ x - x²/2 + x³/3 - ...
	y := x - 1.0
	result := y
	term := y
	for i := 2; i < 10; i++ {
		term *= -y
		result += term / float64(i)
	}
	return result
}

// GetCompressionLevel returns zstd compression level for algorithm
func GetCompressionLevel(algo CompressionAlgorithm) int {
	switch algo {
	case CompressionNone:
		return 0
	case CompressionLZ4:
		return 1 // Fastest
	case CompressionZstd:
		return 3 // Balanced
	case CompressionZstdMax:
		return 19 // Maximum (70-85% reduction)
	case CompressionBrotli:
		return 11 // Maximum
	default:
		return 3
	}
}
