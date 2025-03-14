package vm

import (
	"context"
	"fmt"
	"io"
	"math"
	"sync"
	"time"

	"github.com/klauspost/compress/zstd"
)

// WANMigrationOptimizer provides optimizations for VM migrations over wide-area networks
// where bandwidth, latency, and connection stability may be limited
type WANMigrationOptimizer struct {
	// Configuration
	config WANMigrationConfig

	// Stats tracking
	stats WANMigrationStats

	// Context for cancellation
	ctx    context.Context
	cancel context.CancelFunc

	// Mutex for thread safety
	mu sync.RWMutex
}

// WANMigrationConfig contains configuration for WAN migration optimization
type WANMigrationConfig struct {
	// Compression settings
	CompressionLevel      int     // Compression level (0-9)
	AdaptiveCompression   bool    // Dynamically adjust compression based on CPU and bandwidth
	CompressibleThreshold float64 // Data compressibility threshold to bypass compression

	// Bandwidth management
	MaxBandwidthMbps       int           // Maximum bandwidth to use in Mbps
	DynamicBandwidth       bool          // Adjust bandwidth based on network conditions
	BandwidthProbeInterval time.Duration // Interval to probe available bandwidth

	// Transfer optimizations
	ChunkSizeKB  int // Size of chunks to transfer
	Parallelism  int // Number of parallel transfer streams
	RetryCount   int // Number of retries on transfer failure
	RetryDelayMs int // Delay between retries in milliseconds

	// Delta sync options
	EnableDeltaSync    bool   // Use delta synchronization for disk and memory
	DeltaHashAlgorithm string // Hash algorithm for delta sync ("xxhash", "sha256", etc.)
	DeltaBlockSizeKB   int    // Block size for delta detection

	// Advanced options
	EnablePreemptiveTransfer bool // Start transferring before VM is fully stopped in cold migrations
	EnableBackgroundTransfer bool // Transfer in background during VM operation
	QoSPriority              int  // QoS priority (0-7, higher = more priority)
}

// WANMigrationStats tracks statistics for WAN migration
type WANMigrationStats struct {
	StartTime               time.Time
	EndTime                 time.Time
	BytesTransferred        int64
	BytesBeforeCompression  int64
	BytesAfterCompression   int64
	TransfersAttempted      int
	TransfersSucceeded      int
	TransfersFailed         int
	RetryCount              int
	CompressionSavingsBytes int64
	DeltaSyncSavingsBytes   int64
	AverageBandwidthMbps    float64
	PeakBandwidthMbps       float64
	Latency                 time.Duration
	TotalDowntimeMs         int64
}

// NewWANMigrationOptimizer creates a new WAN migration optimizer with the specified configuration
func NewWANMigrationOptimizer(config WANMigrationConfig) *WANMigrationOptimizer {
	// Set reasonable defaults for any unspecified values
	if config.CompressionLevel == 0 {
		config.CompressionLevel = 3 // Default to medium compression
	}

	if config.ChunkSizeKB == 0 {
		config.ChunkSizeKB = 1024 // Default to 1MB chunks
	}

	if config.Parallelism == 0 {
		config.Parallelism = 4 // Default to 4 parallel streams
	}

	if config.RetryCount == 0 {
		config.RetryCount = 3 // Default to 3 retries
	}

	if config.RetryDelayMs == 0 {
		config.RetryDelayMs = 1000 // Default to 1 second delay
	}

	if config.DeltaBlockSizeKB == 0 {
		config.DeltaBlockSizeKB = 64 // Default to 64KB blocks
	}

	ctx, cancel := context.WithCancel(context.Background())

	return &WANMigrationOptimizer{
		config: config,
		stats: WANMigrationStats{
			StartTime: time.Now(),
		},
		ctx:    ctx,
		cancel: cancel,
	}
}

// DefaultWANMigrationConfig returns a default configuration for WAN migration
func DefaultWANMigrationConfig() WANMigrationConfig {
	return WANMigrationConfig{
		CompressionLevel:         3,
		AdaptiveCompression:      true,
		CompressibleThreshold:    0.95,
		MaxBandwidthMbps:         1000, // 1 Gbps
		DynamicBandwidth:         true,
		BandwidthProbeInterval:   30 * time.Second,
		ChunkSizeKB:              1024,
		Parallelism:              4,
		RetryCount:               3,
		RetryDelayMs:             1000,
		EnableDeltaSync:          true,
		DeltaHashAlgorithm:       "xxhash",
		DeltaBlockSizeKB:         64,
		EnablePreemptiveTransfer: true,
		EnableBackgroundTransfer: true,
		QoSPriority:              4,
	}
}

// OptimizeTransfer wraps the provided reader with optimization layers for WAN transfer
func (o *WANMigrationOptimizer) OptimizeTransfer(ctx context.Context, reader io.Reader) (io.Reader, error) {
	// Add compression if enabled
	if o.config.CompressionLevel > 0 {
		compReader, err := zstd.NewReader(reader)
		if err != nil {
			return nil, fmt.Errorf("failed to create compressed reader: %w", err)
		}

		// Return the compressed reader
		return compReader, nil
	}

	// If no optimizations are applied, return the original reader
	return reader, nil
}

// OptimizeWriter wraps the provided writer with optimization layers for WAN transfer
func (o *WANMigrationOptimizer) OptimizeWriter(ctx context.Context, writer io.Writer) (io.WriteCloser, error) {
	// Add compression if enabled
	if o.config.CompressionLevel > 0 {
		compressOpts := zstd.WithEncoderLevel(zstd.EncoderLevel(o.config.CompressionLevel))
		compWriter, err := zstd.NewWriter(writer, compressOpts)
		if err != nil {
			return nil, fmt.Errorf("failed to create compressed writer: %w", err)
		}

		// Return the compressed writer
		return compWriter, nil
	}

	// If no optimizations are applied, wrap the original writer to match the interface
	return &simpleWriteCloser{writer: writer}, nil
}

// simpleWriteCloser wraps an io.Writer to add the Close method
type simpleWriteCloser struct {
	writer io.Writer
}

func (w *simpleWriteCloser) Write(p []byte) (int, error) {
	return w.writer.Write(p)
}

func (w *simpleWriteCloser) Close() error {
	// No-op for the simple writer
	return nil
}

// GetStats returns the current migration statistics
func (o *WANMigrationOptimizer) GetStats() WANMigrationStats {
	o.mu.RLock()
	defer o.mu.RUnlock()

	// Calculate derived statistics
	stats := o.stats

	// If migration is still in progress, use current time for calculations
	if stats.EndTime.IsZero() {
		stats.EndTime = time.Now()
	}

	// Calculate compression ratio
	if stats.BytesBeforeCompression > 0 {
		stats.CompressionSavingsBytes = stats.BytesBeforeCompression - stats.BytesAfterCompression
	}

	// Calculate average bandwidth
	duration := stats.EndTime.Sub(stats.StartTime).Seconds()
	if duration > 0 && stats.BytesTransferred > 0 {
		// Convert bytes to bits and divide by seconds to get bps, then convert to Mbps
		stats.AverageBandwidthMbps = float64(stats.BytesTransferred*8) / (duration * 1000000)
	}

	return stats
}

// UpdateStats updates the migration statistics
func (o *WANMigrationOptimizer) UpdateStats(update func(*WANMigrationStats)) {
	o.mu.Lock()
	defer o.mu.Unlock()

	update(&o.stats)
}

// CompleteStats marks the migration as complete and finalizes statistics
func (o *WANMigrationOptimizer) CompleteStats() {
	o.mu.Lock()
	defer o.mu.Unlock()

	o.stats.EndTime = time.Now()
}

// Close releases resources used by the optimizer
func (o *WANMigrationOptimizer) Close() error {
	o.cancel()
	return nil
}

// EstimateTransferTime estimates the time required to transfer a given number of bytes
func (o *WANMigrationOptimizer) EstimateTransferTime(bytes int64) time.Duration {
	// Use configured bandwidth or a conservative default
	bandwidthMbps := float64(o.config.MaxBandwidthMbps)
	if bandwidthMbps <= 0 {
		bandwidthMbps = 100 // Assume 100 Mbps as a conservative default
	}

	// Estimate compression savings if enabled
	estimatedCompressedBytes := bytes
	if o.config.CompressionLevel > 0 {
		// Assume a conservative compression ratio based on level
		compressionRatio := 1.0
		switch {
		case o.config.CompressionLevel >= 9:
			compressionRatio = 3.0
		case o.config.CompressionLevel >= 6:
			compressionRatio = 2.5
		case o.config.CompressionLevel >= 3:
			compressionRatio = 2.0
		default:
			compressionRatio = 1.5
		}

		estimatedCompressedBytes = int64(float64(bytes) / compressionRatio)
	}

	// Estimate delta sync savings if enabled
	if o.config.EnableDeltaSync {
		// Assume delta sync saves approximately 30% for typical workloads
		estimatedCompressedBytes = int64(float64(estimatedCompressedBytes) * 0.7)
	}

	// Calculate transfer time: bytes * 8 (bits) / (bandwidth * 1,000,000 (Mbps to bps))
	transferTimeSeconds := (float64(estimatedCompressedBytes) * 8) / (bandwidthMbps * 1000000)

	// Add overhead for protocol, retries, etc. (add 10%)
	transferTimeSeconds *= 1.1

	// Convert to duration
	return time.Duration(transferTimeSeconds * float64(time.Second))
}

// TuneForNetwork adjusts optimization parameters based on network conditions
func (o *WANMigrationOptimizer) TuneForNetwork(bandwidth float64, latency time.Duration, packetLoss float64) {
	o.mu.Lock()
	defer o.mu.Unlock()

	// Adjust chunk size based on bandwidth-delay product (BDP)
	// BDP = bandwidth (bytes/sec) * latency (sec)
	bandwidthBytesPerSec := (bandwidth * 1000000) / 8
	bdpBytes := bandwidthBytesPerSec * latency.Seconds()

	// Set chunk size to a fraction of the BDP, capped between 64KB and 8MB
	optimalChunkSizeBytes := int(math.Max(65536, math.Min(bdpBytes/4, 8*1024*1024)))
	o.config.ChunkSizeKB = optimalChunkSizeBytes / 1024

	// Adjust compression level based on available bandwidth
	if o.config.AdaptiveCompression {
		if bandwidth < 10 { // Less than 10 Mbps
			// Use higher compression to save bandwidth
			o.config.CompressionLevel = 9
		} else if bandwidth < 50 { // 10-50 Mbps
			o.config.CompressionLevel = 6
		} else if bandwidth < 200 { // 50-200 Mbps
			o.config.CompressionLevel = 3
		} else { // > 200 Mbps
			// Use minimal compression for high bandwidth
			o.config.CompressionLevel = 1
		}
	}

	// Adjust parallelism based on bandwidth and packet loss
	if packetLoss > 0.01 { // > 1% packet loss
		// More streams for higher packet loss
		o.config.Parallelism = int(math.Min(8, math.Max(2, float64(o.config.Parallelism)+2)))
	} else {
		// Default parallelism for low packet loss
		o.config.Parallelism = 4
	}

	// Set retry parameters based on conditions
	if packetLoss > 0.05 { // > 5% packet loss
		o.config.RetryCount = 5
		o.config.RetryDelayMs = 2000
	} else if packetLoss > 0.01 { // > 1% packet loss
		o.config.RetryCount = 3
		o.config.RetryDelayMs = 1000
	} else {
		o.config.RetryCount = 2
		o.config.RetryDelayMs = 500
	}
}

// GetOptimizedMigrationOptions returns migration options optimized for WAN
func (o *WANMigrationOptimizer) GetOptimizedMigrationOptions(migrationType string) MigrationOptions {
	// Start with default options
	options := DefaultMigrationOptions()

	// Set migration type
	options.Type = migrationType

	// Apply optimization settings
	options.CompressionLevel = o.config.CompressionLevel

	// Convert bandwidth from Mbps to Bytes/sec
	if o.config.MaxBandwidthMbps > 0 {
		options.BandwidthLimit = int64((o.config.MaxBandwidthMbps * 1000000) / 8)
	}

	// Set priority
	options.Priority = o.config.QoSPriority

	return options
}
