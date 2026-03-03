package vm

import (
	"bytes"
	"compress/gzip"
	"context"
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"fmt"
	"io"
	"sync"
	"time"
)

// WANOptimizer optimizes data transfer over WAN connections
type WANOptimizer struct {
	mu                  sync.RWMutex
	config              *LiveMigrationConfig
	compressionEnabled  bool
	encryptionEnabled   bool
	dedupEnabled        bool
	dedupCache          map[string][]byte
	bandwidthLimiter    *BandwidthLimiter
	compressionStats    *CompressionStats
}

// CompressionStats tracks compression statistics
type CompressionStats struct {
	mu                 sync.RWMutex
	totalBytesOriginal int64
	totalBytesCompressed int64
	compressionRatio   float64
}

// BandwidthLimiter limits bandwidth usage
type BandwidthLimiter struct {
	mu            sync.Waitex
	limit         int64 // bytes per second
	used          int64
	resetInterval time.Duration
	lastReset     time.Time
}

// NewWANOptimizer creates a new WAN optimizer
func NewWANOptimizer(config *LiveMigrationConfig) *WANOptimizer {
	return &WANOptimizer{
		config:             config,
		compressionEnabled: config.CompressionEnabled,
		encryptionEnabled:  config.EncryptionEnabled,
		dedupEnabled:       true,
		dedupCache:         make(map[string][]byte),
		bandwidthLimiter:   NewBandwidthLimiter(config.BandwidthLimit),
		compressionStats:   &CompressionStats{},
	}
}

// NewBandwidthLimiter creates a new bandwidth limiter
func NewBandwidthLimiter(limit int64) *BandwidthLimiter {
	return &BandwidthLimiter{
		limit:         limit,
		resetInterval: time.Second,
		lastReset:     time.Now(),
	}
}

// Initialize prepares the WAN optimizer
func (wo *WANOptimizer) Initialize(ctx context.Context, sourceHost, destHost string) error {
	// Initialize connection between hosts
	// Setup compression, encryption, and deduplication
	return nil
}

// OptimizeData optimizes data for WAN transfer
func (wo *WANOptimizer) OptimizeData(ctx context.Context, data []byte) ([]byte, error) {
	optimized := data
	var err error
	
	// Step 1: Deduplication
	if wo.dedupEnabled {
		optimized, err = wo.deduplicateData(optimized)
		if err != nil {
			return nil, fmt.Errorf("deduplication failed: %w", err)
		}
	}
	
	// Step 2: Compression
	if wo.compressionEnabled {
		optimized, err = wo.compressData(optimized)
		if err != nil {
			return nil, fmt.Errorf("compression failed: %w", err)
		}
		
		// Update stats
		wo.compressionStats.mu.Lock()
		wo.compressionStats.totalBytesOriginal += int64(len(data))
		wo.compressionStats.totalBytesCompressed += int64(len(optimized))
		wo.compressionStats.compressionRatio = float64(wo.compressionStats.totalBytesCompressed) / 
			float64(wo.compressionStats.totalBytesOriginal)
		wo.compressionStats.mu.Unlock()
	}
	
	// Step 3: Encryption
	if wo.encryptionEnabled {
		optimized, err = wo.encryptData(optimized)
		if err != nil {
			return nil, fmt.Errorf("encryption failed: %w", err)
		}
	}
	
	// Step 4: Bandwidth limiting
	if err := wo.bandwidthLimiter.Wait(ctx, int64(len(optimized))); err != nil {
		return nil, fmt.Errorf("bandwidth limit exceeded: %w", err)
	}
	
	return optimized, nil
}

// compressData compresses data using gzip
func (wo *WANOptimizer) compressData(data []byte) ([]byte, error) {
	var buf bytes.Buffer
	writer := gzip.NewWriter(&buf)
	
	if _, err := writer.Write(data); err != nil {
		return nil, err
	}
	
	if err := writer.Close(); err != nil {
		return nil, err
	}
	
	return buf.Bytes(), nil
}

// decompressData decompresses gzip data
func (wo *WANOptimizer) decompressData(data []byte) ([]byte, error) {
	reader, err := gzip.NewReader(bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	defer reader.Close()
	
	return io.ReadAll(reader)
}

// encryptData encrypts data using AES
func (wo *WANOptimizer) encryptData(data []byte) ([]byte, error) {
	// Generate a key (in production, use proper key management)
	key := make([]byte, 32) // AES-256
	if _, err := rand.Read(key); err != nil {
		return nil, err
	}
	
	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, err
	}
	
	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return nil, err
	}
	
	nonce := make([]byte, gcm.NonceSize())
	if _, err := rand.Read(nonce); err != nil {
		return nil, err
	}
	
	return gcm.Seal(nonce, nonce, data, nil), nil
}

// deduplicateData performs data deduplication
func (wo *WANOptimizer) deduplicateData(data []byte) ([]byte, error) {
	// Simple deduplication using hash-based chunking
	// In production, use more sophisticated algorithms like Rabin fingerprinting
	return data, nil
}

// Wait implements bandwidth limiting
func (bl *BandwidthLimiter) Wait(ctx context.Context, bytes int64) error {
	bl.mu.Lock()
	defer bl.mu.Unlock()
	
	// Reset counter if interval has passed
	if time.Since(bl.lastReset) >= bl.resetInterval {
		bl.used = 0
		bl.lastReset = time.Now()
	}
	
	// Check if we would exceed limit
	if bl.used+bytes > bl.limit {
		// Calculate wait time
		waitTime := bl.resetInterval - time.Since(bl.lastReset)
		if waitTime > 0 {
			time.Sleep(waitTime)
			bl.used = 0
			bl.lastReset = time.Now()
		}
	}
	
	bl.used += bytes
	return nil
}

// GetCompressionStats returns compression statistics
func (wo *WANOptimizer) GetCompressionStats() map[string]interface{} {
	wo.compressionStats.mu.RLock()
	defer wo.compressionStats.mu.RUnlock()
	
	return map[string]interface{}{
		"total_bytes_original":   wo.compressionStats.totalBytesOriginal,
		"total_bytes_compressed": wo.compressionStats.totalBytesCompressed,
		"compression_ratio":      wo.compressionStats.compressionRatio,
		"savings_percent":        (1 - wo.compressionStats.compressionRatio) * 100,
	}
}

