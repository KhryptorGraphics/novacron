package migration

import (
	"bytes"
	"context"
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"net"
	"sync"
	"sync/atomic"
	"time"

	"github.com/klauspost/compress/zstd"
	"github.com/pierrec/lz4/v4"
	"golang.org/x/time/rate"
)

// CompressionType represents the compression algorithm to use
type CompressionType int

const (
	CompressionNone CompressionType = iota
	CompressionLZ4
	CompressionZSTD
	CompressionAdaptive // Automatically select based on data and network conditions
)

// QoSPriority represents Quality of Service priority levels
type QoSPriority int

const (
	QoSPriorityLow QoSPriority = iota
	QoSPriorityMedium
	QoSPriorityHigh
	QoSPriorityCritical
)

// WANOptimizer handles WAN optimization for VM migration
type WANOptimizer struct {
	// Compression
	compressionType  CompressionType
	compressionLevel int
	zstdEncoder      *zstd.Encoder
	zstdDecoder      *zstd.Decoder
	
	// Bandwidth management
	bandwidthLimit   int64 // bytes per second, 0 = unlimited
	rateLimiter      *rate.Limiter
	adaptiveLimiter  *AdaptiveBandwidthLimiter
	
	// QoS
	qosPriority      QoSPriority
	tcpOptimizer     *TCPOptimizer
	
	// Delta sync
	deltaTracker     *DeltaPageTracker
	pageCache        *PageCache
	
	// Encryption
	encryptionKey    []byte
	cipher           cipher.AEAD
	
	// Metrics
	bytesCompressed  atomic.Int64
	bytesTransferred atomic.Int64
	compressionRatio atomic.Value // float64
	transferRate     atomic.Int64 // bytes per second
	
	// Network conditions
	latency          atomic.Int64 // milliseconds
	packetLoss       atomic.Value // float64
	
	mu               sync.RWMutex
	ctx              context.Context
	cancel           context.CancelFunc
}

// WANOptimizerConfig contains configuration for WAN optimization
type WANOptimizerConfig struct {
	CompressionType  CompressionType
	CompressionLevel int // 1-9 for zstd, 1-16 for lz4
	BandwidthLimit   int64
	QoSPriority      QoSPriority
	EnableEncryption bool
	EncryptionKey    []byte
	EnableDeltaSync  bool
	PageCacheSize    int // MB
	TCPOptimization  bool
}

// NewWANOptimizer creates a new WAN optimizer
func NewWANOptimizer(config WANOptimizerConfig) (*WANOptimizer, error) {
	ctx, cancel := context.WithCancel(context.Background())
	
	w := &WANOptimizer{
		compressionType:  config.CompressionType,
		compressionLevel: config.CompressionLevel,
		bandwidthLimit:   config.BandwidthLimit,
		qosPriority:      config.QoSPriority,
		ctx:              ctx,
		cancel:           cancel,
	}
	
	// Initialize compression
	if err := w.initCompression(); err != nil {
		cancel()
		return nil, fmt.Errorf("failed to initialize compression: %w", err)
	}
	
	// Initialize bandwidth limiter
	if config.BandwidthLimit > 0 {
		w.rateLimiter = rate.NewLimiter(rate.Limit(config.BandwidthLimit), int(config.BandwidthLimit))
		w.adaptiveLimiter = NewAdaptiveBandwidthLimiter(config.BandwidthLimit)
	}
	
	// Initialize TCP optimizer
	if config.TCPOptimization {
		w.tcpOptimizer = NewTCPOptimizer()
	}
	
	// Initialize delta sync
	if config.EnableDeltaSync {
		w.deltaTracker = NewDeltaPageTracker()
		w.pageCache = NewPageCache(config.PageCacheSize * 1024 * 1024) // Convert MB to bytes
	}
	
	// Initialize encryption
	if config.EnableEncryption {
		if err := w.initEncryption(config.EncryptionKey); err != nil {
			cancel()
			return nil, fmt.Errorf("failed to initialize encryption: %w", err)
		}
	}
	
	// Start monitoring goroutine
	go w.monitorNetworkConditions()
	
	return w, nil
}

// initCompression initializes compression encoders/decoders
func (w *WANOptimizer) initCompression() error {
	switch w.compressionType {
	case CompressionZSTD:
		encoder, err := zstd.NewWriter(nil, 
			zstd.WithEncoderLevel(zstd.EncoderLevelFromZstd(w.compressionLevel)))
		if err != nil {
			return err
		}
		w.zstdEncoder = encoder
		
		decoder, err := zstd.NewReader(nil)
		if err != nil {
			return err
		}
		w.zstdDecoder = decoder
		
	case CompressionLZ4, CompressionAdaptive:
		// LZ4 doesn't require initialization
	}
	
	return nil
}

// initEncryption initializes AES-GCM encryption
func (w *WANOptimizer) initEncryption(key []byte) error {
	if len(key) == 0 {
		// Generate a random key
		key = make([]byte, 32)
		if _, err := rand.Read(key); err != nil {
			return err
		}
	}
	
	// Hash the key to ensure it's the right size
	hash := sha256.Sum256(key)
	w.encryptionKey = hash[:]
	
	block, err := aes.NewCipher(w.encryptionKey)
	if err != nil {
		return err
	}
	
	w.cipher, err = cipher.NewGCM(block)
	if err != nil {
		return err
	}
	
	return nil
}

// CompressData compresses data using the configured algorithm
func (w *WANOptimizer) CompressData(data []byte) ([]byte, error) {
	if w.compressionType == CompressionNone {
		return data, nil
	}
	
	// Adaptive compression: choose based on data size and network conditions
	if w.compressionType == CompressionAdaptive {
		return w.adaptiveCompress(data)
	}
	
	var compressed []byte
	var err error
	
	switch w.compressionType {
	case CompressionLZ4:
		compressed = make([]byte, lz4.CompressBlockBound(len(data)))
		n, err := lz4.CompressBlock(data, compressed, nil)
		if err != nil {
			return nil, err
		}
		compressed = compressed[:n]
		
	case CompressionZSTD:
		compressed = w.zstdEncoder.EncodeAll(data, nil)
	}
	
	// Update metrics
	w.bytesCompressed.Add(int64(len(compressed)))
	ratio := float64(len(compressed)) / float64(len(data))
	w.compressionRatio.Store(ratio)
	
	return compressed, err
}

// DecompressData decompresses data
func (w *WANOptimizer) DecompressData(compressed []byte, originalSize int) ([]byte, error) {
	if w.compressionType == CompressionNone {
		return compressed, nil
	}
	
	var decompressed []byte
	var err error
	
	// Read compression type from header if adaptive
	compressionType := w.compressionType
	if w.compressionType == CompressionAdaptive && len(compressed) > 0 {
		compressionType = CompressionType(compressed[0])
		compressed = compressed[1:]
	}
	
	switch compressionType {
	case CompressionLZ4:
		decompressed = make([]byte, originalSize)
		n, err := lz4.UncompressBlock(compressed, decompressed)
		if err != nil {
			return nil, err
		}
		decompressed = decompressed[:n]
		
	case CompressionZSTD:
		decompressed, err = w.zstdDecoder.DecodeAll(compressed, nil)
	}
	
	return decompressed, err
}

// adaptiveCompress selects the best compression algorithm based on conditions
func (w *WANOptimizer) adaptiveCompress(data []byte) ([]byte, error) {
	dataSize := len(data)
	latency := w.latency.Load()
	
	// For small data or low latency, use LZ4 for speed
	// For large data or high latency, use ZSTD for better compression
	var compressionType CompressionType
	if dataSize < 4096 || latency < 50 {
		compressionType = CompressionLZ4
	} else {
		compressionType = CompressionZSTD
	}
	
	// Compress with selected algorithm
	var compressed []byte
	switch compressionType {
	case CompressionLZ4:
		compressed = make([]byte, lz4.CompressBlockBound(dataSize))
		n, err := lz4.CompressBlock(data, compressed, nil)
		if err != nil {
			return nil, err
		}
		compressed = compressed[:n]
		
	case CompressionZSTD:
		compressed = w.zstdEncoder.EncodeAll(data, nil)
	}
	
	// Prepend compression type for decompression
	result := make([]byte, len(compressed)+1)
	result[0] = byte(compressionType)
	copy(result[1:], compressed)
	
	return result, nil
}

// TransferWithOptimization transfers data with WAN optimization
func (w *WANOptimizer) TransferWithOptimization(conn net.Conn, data []byte) error {
	// Apply delta sync if enabled
	if w.deltaTracker != nil {
		delta := w.deltaTracker.ComputeDelta(data)
		if delta != nil && len(delta) < len(data) {
			data = delta
		}
	}
	
	// Compress data
	compressed, err := w.CompressData(data)
	if err != nil {
		return fmt.Errorf("compression failed: %w", err)
	}
	
	// Encrypt if enabled
	if w.cipher != nil {
		compressed, err = w.encryptData(compressed)
		if err != nil {
			return fmt.Errorf("encryption failed: %w", err)
		}
	}
	
	// Apply bandwidth limiting
	if w.rateLimiter != nil {
		if err := w.rateLimiter.WaitN(w.ctx, len(compressed)); err != nil {
			return fmt.Errorf("rate limiting failed: %w", err)
		}
	}
	
	// Apply TCP optimizations
	if w.tcpOptimizer != nil {
		w.tcpOptimizer.OptimizeConnection(conn)
	}
	
	// Send data with length prefix
	header := make([]byte, 8)
	binary.BigEndian.PutUint64(header, uint64(len(compressed)))
	
	if _, err := conn.Write(header); err != nil {
		return fmt.Errorf("failed to write header: %w", err)
	}
	
	written, err := conn.Write(compressed)
	if err != nil {
		return fmt.Errorf("failed to write data: %w", err)
	}
	
	// Update metrics
	w.bytesTransferred.Add(int64(written))
	
	return nil
}

// encryptData encrypts data using AES-GCM
func (w *WANOptimizer) encryptData(data []byte) ([]byte, error) {
	nonce := make([]byte, w.cipher.NonceSize())
	if _, err := io.ReadFull(rand.Reader, nonce); err != nil {
		return nil, err
	}
	
	encrypted := w.cipher.Seal(nonce, nonce, data, nil)
	return encrypted, nil
}

// decryptData decrypts data using AES-GCM
func (w *WANOptimizer) decryptData(encrypted []byte) ([]byte, error) {
	if len(encrypted) < w.cipher.NonceSize() {
		return nil, errors.New("encrypted data too short")
	}
	
	nonce := encrypted[:w.cipher.NonceSize()]
	ciphertext := encrypted[w.cipher.NonceSize():]
	
	decrypted, err := w.cipher.Open(nil, nonce, ciphertext, nil)
	if err != nil {
		return nil, err
	}
	
	return decrypted, nil
}

// monitorNetworkConditions monitors network latency and packet loss
func (w *WANOptimizer) monitorNetworkConditions() {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()
	
	for {
		select {
		case <-w.ctx.Done():
			return
		case <-ticker.C:
			// This would normally measure actual network conditions
			// For now, we'll use placeholder values
			w.latency.Store(50) // 50ms latency
			w.packetLoss.Store(0.001) // 0.1% packet loss
			
			// Adjust bandwidth limiter based on conditions
			if w.adaptiveLimiter != nil {
				w.adaptiveLimiter.AdjustLimit(w.latency.Load(), w.packetLoss.Load().(float64))
			}
		}
	}
}

// GetMetrics returns current optimization metrics
func (w *WANOptimizer) GetMetrics() map[string]interface{} {
	ratio := 1.0
	if val := w.compressionRatio.Load(); val != nil {
		ratio = val.(float64)
	}
	
	packetLoss := 0.0
	if val := w.packetLoss.Load(); val != nil {
		packetLoss = val.(float64)
	}
	
	return map[string]interface{}{
		"bytes_compressed":  w.bytesCompressed.Load(),
		"bytes_transferred": w.bytesTransferred.Load(),
		"compression_ratio": ratio,
		"transfer_rate":     w.transferRate.Load(),
		"latency_ms":        w.latency.Load(),
		"packet_loss":       packetLoss,
	}
}

// SetBandwidthLimit updates the bandwidth limit
func (w *WANOptimizer) SetBandwidthLimit(bytesPerSecond int64) {
	w.mu.Lock()
	defer w.mu.Unlock()
	
	w.bandwidthLimit = bytesPerSecond
	if bytesPerSecond > 0 {
		w.rateLimiter = rate.NewLimiter(rate.Limit(bytesPerSecond), int(bytesPerSecond))
	} else {
		w.rateLimiter = nil
	}
}

// Close cleanly shuts down the WAN optimizer
func (w *WANOptimizer) Close() error {
	w.cancel()
	
	if w.zstdEncoder != nil {
		w.zstdEncoder.Close()
	}
	
	return nil
}

// AdaptiveBandwidthLimiter adjusts bandwidth based on network conditions
type AdaptiveBandwidthLimiter struct {
	baseBandwidth int64
	currentLimit  atomic.Int64
	mu            sync.RWMutex
}

// NewAdaptiveBandwidthLimiter creates a new adaptive bandwidth limiter
func NewAdaptiveBandwidthLimiter(baseBandwidth int64) *AdaptiveBandwidthLimiter {
	a := &AdaptiveBandwidthLimiter{
		baseBandwidth: baseBandwidth,
	}
	a.currentLimit.Store(baseBandwidth)
	return a
}

// AdjustLimit adjusts the bandwidth limit based on network conditions
func (a *AdaptiveBandwidthLimiter) AdjustLimit(latencyMs int64, packetLoss float64) {
	// Reduce bandwidth if latency is high or packet loss is significant
	adjustment := 1.0
	
	if latencyMs > 100 {
		adjustment *= 0.8
	} else if latencyMs > 200 {
		adjustment *= 0.6
	}
	
	if packetLoss > 0.01 {
		adjustment *= 0.7
	} else if packetLoss > 0.05 {
		adjustment *= 0.5
	}
	
	newLimit := int64(float64(a.baseBandwidth) * adjustment)
	a.currentLimit.Store(newLimit)
}

// GetCurrentLimit returns the current bandwidth limit
func (a *AdaptiveBandwidthLimiter) GetCurrentLimit() int64 {
	return a.currentLimit.Load()
}

// TCPOptimizer optimizes TCP connections for WAN transfers
type TCPOptimizer struct {
	bufferSize int
	noDelay    bool
	keepAlive  bool
}

// NewTCPOptimizer creates a new TCP optimizer
func NewTCPOptimizer() *TCPOptimizer {
	return &TCPOptimizer{
		bufferSize: 1024 * 1024, // 1MB buffer
		noDelay:    true,
		keepAlive:  true,
	}
}

// OptimizeConnection applies TCP optimizations to a connection
func (t *TCPOptimizer) OptimizeConnection(conn net.Conn) error {
	tcpConn, ok := conn.(*net.TCPConn)
	if !ok {
		return errors.New("not a TCP connection")
	}
	
	// Set TCP_NODELAY for low latency
	if err := tcpConn.SetNoDelay(t.noDelay); err != nil {
		return err
	}
	
	// Enable keep-alive
	if err := tcpConn.SetKeepAlive(t.keepAlive); err != nil {
		return err
	}
	
	// Set keep-alive period
	if err := tcpConn.SetKeepAlivePeriod(30 * time.Second); err != nil {
		return err
	}
	
	// Set buffer sizes
	if err := tcpConn.SetReadBuffer(t.bufferSize); err != nil {
		return err
	}
	if err := tcpConn.SetWriteBuffer(t.bufferSize); err != nil {
		return err
	}
	
	return nil
}

// DeltaPageTracker tracks memory page changes for delta synchronization
type DeltaPageTracker struct {
	pageSize     int
	pageHashes   map[uint64][]byte // page index -> hash
	dirtyPages   map[uint64]bool
	mu           sync.RWMutex
}

// NewDeltaPageTracker creates a new delta page tracker
func NewDeltaPageTracker() *DeltaPageTracker {
	return &DeltaPageTracker{
		pageSize:   4096, // 4KB pages
		pageHashes: make(map[uint64][]byte),
		dirtyPages: make(map[uint64]bool),
	}
}

// ComputeDelta computes the delta between current and previous state
func (d *DeltaPageTracker) ComputeDelta(data []byte) []byte {
	d.mu.Lock()
	defer d.mu.Unlock()
	
	var delta bytes.Buffer
	pageCount := (len(data) + d.pageSize - 1) / d.pageSize
	
	// Write header: number of pages
	binary.Write(&delta, binary.BigEndian, uint32(pageCount))
	
	for i := 0; i < pageCount; i++ {
		start := i * d.pageSize
		end := start + d.pageSize
		if end > len(data) {
			end = len(data)
		}
		
		page := data[start:end]
		hash := sha256.Sum256(page)
		pageIdx := uint64(i)
		
		// Check if page has changed
		if oldHash, exists := d.pageHashes[pageIdx]; exists {
			if bytes.Equal(oldHash[:], hash[:]) {
				// Page unchanged, write marker
				binary.Write(&delta, binary.BigEndian, uint32(0))
				continue
			}
		}
		
		// Page changed, write page data
		binary.Write(&delta, binary.BigEndian, uint32(len(page)))
		delta.Write(page)
		
		// Update hash
		d.pageHashes[pageIdx] = hash[:]
		d.dirtyPages[pageIdx] = true
	}
	
	return delta.Bytes()
}

// PageCache caches frequently accessed pages
type PageCache struct {
	cache    map[uint64][]byte
	maxSize  int
	currSize int
	mu       sync.RWMutex
}

// NewPageCache creates a new page cache
func NewPageCache(maxSize int) *PageCache {
	return &PageCache{
		cache:   make(map[uint64][]byte),
		maxSize: maxSize,
	}
}

// Get retrieves a page from cache
func (p *PageCache) Get(pageIdx uint64) ([]byte, bool) {
	p.mu.RLock()
	defer p.mu.RUnlock()
	
	data, exists := p.cache[pageIdx]
	return data, exists
}

// Put stores a page in cache
func (p *PageCache) Put(pageIdx uint64, data []byte) {
	p.mu.Lock()
	defer p.mu.Unlock()
	
	// Check if we need to evict
	if p.currSize+len(data) > p.maxSize {
		// Simple eviction: remove first item (could be improved with LRU)
		for idx, page := range p.cache {
			delete(p.cache, idx)
			p.currSize -= len(page)
			break
		}
	}
	
	p.cache[pageIdx] = data
	p.currSize += len(data)
}