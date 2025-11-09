// Package dwcp implements the Distributed WAN Communication Protocol for NovaCron
package dwcp

import (
	"context"
	"crypto/rand"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"net"
	"sync"
	"sync/atomic"
	"time"

	"golang.org/x/time/rate"
)

// AMST (Adaptive Multi-Stream Transport) provides high-performance WAN transfer
// using multiple parallel TCP streams with adaptive optimization
type AMST struct {
	// Configuration
	config AMSTConfig

	// Connection management
	streams       []*Stream
	streamPool    sync.Pool
	activeStreams atomic.Int32

	// Performance tracking
	bytesTransferred atomic.Int64
	transferRate     atomic.Int64 // bytes per second
	latency          atomic.Int64 // milliseconds
	packetLoss       atomic.Value // float64

	// Adaptive optimization
	optimizer       *StreamOptimizer
	chunkSize       atomic.Int32
	streamCount     atomic.Int32

	// Rate limiting
	rateLimiter     *rate.Limiter

	// Synchronization
	mu              sync.RWMutex
	ctx             context.Context
	cancel          context.CancelFunc

	// Error handling
	lastError       atomic.Value // error
	errorCount      atomic.Int32
	reconnectDelay  time.Duration
}

// AMSTConfig contains configuration for AMST
type AMSTConfig struct {
	// Connection parameters
	MinStreams      int           // Minimum number of parallel streams (default: 4)
	MaxStreams      int           // Maximum number of parallel streams (default: 256)
	InitialStreams  int           // Initial stream count (default: 16)

	// Performance tuning
	ChunkSize       int           // Initial chunk size in bytes (default: 64KB)
	MinChunkSize    int           // Minimum chunk size (default: 4KB)
	MaxChunkSize    int           // Maximum chunk size (default: 1MB)

	// Network optimization
	TCPNoDelay      bool          // Disable Nagle's algorithm (default: true)
	KeepAlive       bool          // Enable TCP keepalive (default: true)
	KeepAlivePeriod time.Duration // Keepalive period (default: 30s)

	// Bandwidth management
	BandwidthLimit  int64         // Maximum bandwidth in bytes/sec (0 = unlimited)
	BurstSize       int           // Token bucket burst size

	// Timeout settings
	ConnectTimeout  time.Duration // Connection timeout (default: 30s)
	ReadTimeout     time.Duration // Read timeout per chunk (default: 60s)
	WriteTimeout    time.Duration // Write timeout per chunk (default: 60s)

	// Adaptive optimization
	EnableAdaptive  bool          // Enable adaptive optimization (default: true)
	OptimizeInterval time.Duration // Optimization interval (default: 5s)

	// Reliability
	MaxRetries      int           // Maximum retry attempts (default: 3)
	RetryDelay      time.Duration // Delay between retries (default: 1s)
	EnablePacing    bool          // Enable packet pacing (default: true)
}

// Stream represents a single TCP stream in the multi-stream transport
type Stream struct {
	id              string
	conn            net.Conn
	amst            *AMST

	// Statistics
	bytesTransferred int64
	transferRate     int64
	errors           int32
	lastActive       time.Time

	// State
	active          bool
	mu              sync.Mutex
}

// StreamOptimizer dynamically optimizes stream parameters based on network conditions
type StreamOptimizer struct {
	amst            *AMST

	// Network measurements
	rttHistory      []time.Duration
	bandwidthHistory []int64
	lossHistory     []float64

	// Optimization state
	lastOptimization time.Time
	optimizationCount int

	mu              sync.Mutex
}

// NewAMST creates a new Adaptive Multi-Stream Transport instance
func NewAMST(config AMSTConfig) (*AMST, error) {
	// Validate and set defaults
	if config.MinStreams <= 0 {
		config.MinStreams = 4
	}
	if config.MaxStreams <= 0 {
		config.MaxStreams = 256
	}
	if config.InitialStreams <= 0 {
		config.InitialStreams = 16
	}
	if config.InitialStreams > config.MaxStreams {
		config.InitialStreams = config.MaxStreams
	}
	if config.InitialStreams < config.MinStreams {
		config.InitialStreams = config.MinStreams
	}

	if config.ChunkSize <= 0 {
		config.ChunkSize = 64 * 1024 // 64KB
	}
	if config.MinChunkSize <= 0 {
		config.MinChunkSize = 4 * 1024 // 4KB
	}
	if config.MaxChunkSize <= 0 {
		config.MaxChunkSize = 1024 * 1024 // 1MB
	}

	if config.ConnectTimeout <= 0 {
		config.ConnectTimeout = 30 * time.Second
	}
	if config.ReadTimeout <= 0 {
		config.ReadTimeout = 60 * time.Second
	}
	if config.WriteTimeout <= 0 {
		config.WriteTimeout = 60 * time.Second
	}

	if config.OptimizeInterval <= 0 {
		config.OptimizeInterval = 5 * time.Second
	}
	if config.MaxRetries <= 0 {
		config.MaxRetries = 3
	}
	if config.RetryDelay <= 0 {
		config.RetryDelay = time.Second
	}
	if config.KeepAlivePeriod <= 0 {
		config.KeepAlivePeriod = 30 * time.Second
	}

	ctx, cancel := context.WithCancel(context.Background())

	amst := &AMST{
		config:         config,
		streams:        make([]*Stream, 0, config.MaxStreams),
		reconnectDelay: config.RetryDelay,
		ctx:            ctx,
		cancel:         cancel,
	}

	// Initialize atomic values
	amst.chunkSize.Store(int32(config.ChunkSize))
	amst.streamCount.Store(int32(config.InitialStreams))
	amst.packetLoss.Store(float64(0))

	// Create rate limiter if bandwidth limit is set
	if config.BandwidthLimit > 0 {
		burstSize := config.BurstSize
		if burstSize <= 0 {
			burstSize = int(config.BandwidthLimit / 10) // 100ms worth of bandwidth
			if burstSize < config.MaxChunkSize {
				burstSize = config.MaxChunkSize
			}
		}
		amst.rateLimiter = rate.NewLimiter(rate.Limit(config.BandwidthLimit), burstSize)
	}

	// Create stream pool for efficient stream reuse
	amst.streamPool = sync.Pool{
		New: func() interface{} {
			return &Stream{
				amst: amst,
			}
		},
	}

	// Create optimizer if adaptive optimization is enabled
	if config.EnableAdaptive {
		amst.optimizer = &StreamOptimizer{
			amst:           amst,
			rttHistory:     make([]time.Duration, 0, 100),
			bandwidthHistory: make([]int64, 0, 100),
			lossHistory:    make([]float64, 0, 100),
		}

		// Start optimization loop
		go amst.optimizationLoop()
	}

	return amst, nil
}

// Connect establishes connections to the target host
func (amst *AMST) Connect(ctx context.Context, host string, port int) error {
	target := fmt.Sprintf("%s:%d", host, port)
	streamCount := int(amst.streamCount.Load())

	// Create initial streams
	var wg sync.WaitGroup
	errChan := make(chan error, streamCount)

	for i := 0; i < streamCount; i++ {
		wg.Add(1)
		go func(streamID int) {
			defer wg.Done()

			stream := amst.streamPool.Get().(*Stream)
			stream.id = fmt.Sprintf("%d-%d", time.Now().Unix(), streamID)
			stream.active = false

			// Create dialer with timeout
			dialer := &net.Dialer{
				Timeout:   amst.config.ConnectTimeout,
				KeepAlive: amst.config.KeepAlivePeriod,
			}

			// Connect with context
			conn, err := dialer.DialContext(ctx, "tcp", target)
			if err != nil {
				errChan <- fmt.Errorf("stream %d connection failed: %w", streamID, err)
				amst.streamPool.Put(stream)
				return
			}

			// Configure TCP options
			if tcpConn, ok := conn.(*net.TCPConn); ok {
				if amst.config.TCPNoDelay {
					tcpConn.SetNoDelay(true)
				}
				if amst.config.KeepAlive {
					tcpConn.SetKeepAlive(true)
					tcpConn.SetKeepAlivePeriod(amst.config.KeepAlivePeriod)
				}

				// Set buffer sizes for high-bandwidth transfers
				tcpConn.SetReadBuffer(4 * 1024 * 1024)  // 4MB
				tcpConn.SetWriteBuffer(4 * 1024 * 1024) // 4MB
			}

			stream.conn = conn
			stream.active = true
			stream.lastActive = time.Now()

			amst.mu.Lock()
			amst.streams = append(amst.streams, stream)
			amst.mu.Unlock()

			amst.activeStreams.Add(1)
		}(i)
	}

	wg.Wait()
	close(errChan)

	// Check for errors
	var errors []error
	for err := range errChan {
		errors = append(errors, err)
	}

	if len(errors) > 0 {
		// If some streams connected successfully, continue
		if amst.activeStreams.Load() > 0 {
			// Log errors but continue
			for _, err := range errors {
				fmt.Printf("AMST connection warning: %v\n", err)
			}
		} else {
			// All connections failed
			return fmt.Errorf("failed to establish any streams: %v", errors[0])
		}
	}

	return nil
}

// Transfer sends data using parallel streams with optimal chunking
func (amst *AMST) Transfer(ctx context.Context, data []byte, progressCallback func(int64)) error {
	if len(data) == 0 {
		return errors.New("no data to transfer")
	}

	totalSize := int64(len(data))
	chunkSize := int(amst.chunkSize.Load())

	// Calculate number of chunks
	numChunks := (len(data) + chunkSize - 1) / chunkSize

	// Create work channel
	type chunk struct {
		id     int
		offset int
		data   []byte
	}

	workChan := make(chan chunk, numChunks)

	// Populate work queue
	for i := 0; i < numChunks; i++ {
		start := i * chunkSize
		end := start + chunkSize
		if end > len(data) {
			end = len(data)
		}

		workChan <- chunk{
			id:     i,
			offset: start,
			data:   data[start:end],
		}
	}
	close(workChan)

	// Get active streams
	amst.mu.RLock()
	activeStreams := make([]*Stream, 0, len(amst.streams))
	for _, stream := range amst.streams {
		if stream.active {
			activeStreams = append(activeStreams, stream)
		}
	}
	amst.mu.RUnlock()

	if len(activeStreams) == 0 {
		return errors.New("no active streams available")
	}

	// Distribute work among streams
	var wg sync.WaitGroup
	errChan := make(chan error, len(activeStreams))
	transferredBytes := atomic.Int64{}

	for _, stream := range activeStreams {
		wg.Add(1)
		go func(s *Stream) {
			defer wg.Done()

			for chunk := range workChan {
				// Apply rate limiting if configured
				if amst.rateLimiter != nil {
					if err := amst.rateLimiter.WaitN(ctx, len(chunk.data)); err != nil {
						errChan <- fmt.Errorf("rate limiter error: %w", err)
						return
					}
				}

				// Create packet header
				header := make([]byte, 16)
				binary.BigEndian.PutUint32(header[0:4], uint32(chunk.id))
				binary.BigEndian.PutUint32(header[4:8], uint32(chunk.offset))
				binary.BigEndian.PutUint32(header[8:12], uint32(len(chunk.data)))
				binary.BigEndian.PutUint32(header[12:16], uint32(totalSize))

				// Set write timeout
				if amst.config.WriteTimeout > 0 {
					s.conn.SetWriteDeadline(time.Now().Add(amst.config.WriteTimeout))
				}

				// Send header
				if _, err := s.conn.Write(header); err != nil {
					s.active = false
					errChan <- fmt.Errorf("stream %s header write failed: %w", s.id, err)
					return
				}

				// Send data with optional pacing
				if amst.config.EnablePacing {
					// Send in smaller sub-chunks to avoid buffer bloat
					subChunkSize := 16 * 1024 // 16KB
					for offset := 0; offset < len(chunk.data); offset += subChunkSize {
						end := offset + subChunkSize
						if end > len(chunk.data) {
							end = len(chunk.data)
						}

						if _, err := s.conn.Write(chunk.data[offset:end]); err != nil {
							s.active = false
							errChan <- fmt.Errorf("stream %s data write failed: %w", s.id, err)
							return
						}

						// Small delay for pacing (adaptive based on bandwidth)
						if offset+subChunkSize < len(chunk.data) {
							time.Sleep(time.Microsecond * 100)
						}
					}
				} else {
					// Send entire chunk at once
					if _, err := s.conn.Write(chunk.data); err != nil {
						s.active = false
						errChan <- fmt.Errorf("stream %s data write failed: %w", s.id, err)
						return
					}
				}

				// Update statistics
				transferred := int64(len(chunk.data))
				s.bytesTransferred += transferred
				s.lastActive = time.Now()

				totalTransferred := transferredBytes.Add(transferred)
				amst.bytesTransferred.Add(transferred)

				// Callback for progress tracking
				if progressCallback != nil {
					progressCallback(totalTransferred)
				}
			}
		}(stream)
	}

	// Wait for all transfers to complete
	wg.Wait()
	close(errChan)

	// Check for errors
	for err := range errChan {
		if err != nil {
			return err
		}
	}

	// Verify all data was transferred
	if transferredBytes.Load() != totalSize {
		return fmt.Errorf("incomplete transfer: %d of %d bytes", transferredBytes.Load(), totalSize)
	}

	return nil
}

// Receive reads data from parallel streams and reconstructs the original data
func (amst *AMST) Receive(ctx context.Context, progressCallback func(int64)) ([]byte, error) {
	// Get active streams
	amst.mu.RLock()
	activeStreams := make([]*Stream, 0, len(amst.streams))
	for _, stream := range amst.streams {
		if stream.active {
			activeStreams = append(activeStreams, stream)
		}
	}
	amst.mu.RUnlock()

	if len(activeStreams) == 0 {
		return nil, errors.New("no active streams available")
	}

	// Chunk reassembly map
	chunks := make(map[int][]byte)
	var totalSize int64
	var mu sync.Mutex

	// Read from all streams concurrently
	var wg sync.WaitGroup
	errChan := make(chan error, len(activeStreams))
	receivedBytes := atomic.Int64{}

	for _, stream := range activeStreams {
		wg.Add(1)
		go func(s *Stream) {
			defer wg.Done()

			for {
				// Read header
				header := make([]byte, 16)

				if amst.config.ReadTimeout > 0 {
					s.conn.SetReadDeadline(time.Now().Add(amst.config.ReadTimeout))
				}

				if _, err := io.ReadFull(s.conn, header); err != nil {
					if err == io.EOF {
						// Normal completion
						return
					}
					s.active = false
					errChan <- fmt.Errorf("stream %s header read failed: %w", s.id, err)
					return
				}

				// Parse header
				chunkID := int(binary.BigEndian.Uint32(header[0:4]))
				offset := int(binary.BigEndian.Uint32(header[4:8]))
				chunkSize := int(binary.BigEndian.Uint32(header[8:12]))
				fileTotalSize := int64(binary.BigEndian.Uint32(header[12:16]))

				// Update total size if not set
				mu.Lock()
				if totalSize == 0 {
					totalSize = fileTotalSize
				}
				mu.Unlock()

				// Read chunk data
				chunkData := make([]byte, chunkSize)
				if _, err := io.ReadFull(s.conn, chunkData); err != nil {
					s.active = false
					errChan <- fmt.Errorf("stream %s data read failed: %w", s.id, err)
					return
				}

				// Store chunk
				mu.Lock()
				chunks[chunkID] = chunkData
				mu.Unlock()

				// Update statistics
				received := int64(chunkSize)
				s.bytesTransferred += received
				s.lastActive = time.Now()

				totalReceived := receivedBytes.Add(received)
				amst.bytesTransferred.Add(received)

				// Callback for progress tracking
				if progressCallback != nil {
					progressCallback(totalReceived)
				}

				// Check if we've received all data
				if totalReceived >= totalSize {
					return
				}

				// Check context cancellation
				select {
				case <-ctx.Done():
					errChan <- ctx.Err()
					return
				default:
				}
			}
		}(stream)
	}

	// Wait for all reads to complete
	wg.Wait()
	close(errChan)

	// Check for errors
	for err := range errChan {
		if err != nil && err != io.EOF {
			return nil, err
		}
	}

	// Reassemble data
	result := make([]byte, 0, totalSize)
	chunkID := 0
	for {
		chunk, exists := chunks[chunkID]
		if !exists {
			break
		}
		result = append(result, chunk...)
		chunkID++
	}

	if int64(len(result)) != totalSize {
		return nil, fmt.Errorf("incomplete receive: %d of %d bytes", len(result), totalSize)
	}

	return result, nil
}

// optimizationLoop continuously optimizes stream parameters
func (amst *AMST) optimizationLoop() {
	ticker := time.NewTicker(amst.config.OptimizeInterval)
	defer ticker.Stop()

	for {
		select {
		case <-amst.ctx.Done():
			return
		case <-ticker.C:
			amst.optimize()
		}
	}
}

// optimize adjusts stream count and chunk size based on network conditions
func (amst *AMST) optimize() {
	if amst.optimizer == nil {
		return
	}

	amst.optimizer.mu.Lock()
	defer amst.optimizer.mu.Unlock()

	// Calculate current transfer rate
	currentRate := amst.transferRate.Load()
	if currentRate == 0 {
		return // No data to optimize on
	}

	// Get current parameters
	currentStreams := int(amst.streamCount.Load())
	currentChunkSize := int(amst.chunkSize.Load())

	// Measure network conditions
	latency := amst.latency.Load()
	packetLoss := amst.packetLoss.Load().(float64)

	// Optimize stream count based on bandwidth-delay product
	optimalStreams := currentStreams
	if latency > 0 {
		// BDP = bandwidth * RTT
		bdp := currentRate * latency / 1000 // Convert ms to seconds
		// Optimal streams = BDP / chunk_size
		optimalStreams = int(bdp) / currentChunkSize

		// Apply bounds
		if optimalStreams < amst.config.MinStreams {
			optimalStreams = amst.config.MinStreams
		}
		if optimalStreams > amst.config.MaxStreams {
			optimalStreams = amst.config.MaxStreams
		}

		// Gradual adjustment (max 2x change per interval)
		if optimalStreams > currentStreams*2 {
			optimalStreams = currentStreams * 2
		} else if optimalStreams < currentStreams/2 {
			optimalStreams = currentStreams / 2
		}
	}

	// Optimize chunk size based on packet loss
	optimalChunkSize := currentChunkSize
	if packetLoss > 0.05 { // > 5% loss
		// Reduce chunk size for better granularity
		optimalChunkSize = currentChunkSize * 3 / 4
	} else if packetLoss < 0.01 { // < 1% loss
		// Increase chunk size for efficiency
		optimalChunkSize = currentChunkSize * 5 / 4
	}

	// Apply bounds
	if optimalChunkSize < amst.config.MinChunkSize {
		optimalChunkSize = amst.config.MinChunkSize
	}
	if optimalChunkSize > amst.config.MaxChunkSize {
		optimalChunkSize = amst.config.MaxChunkSize
	}

	// Update parameters if changed significantly (> 10% difference)
	if float64(abs(optimalStreams-currentStreams)) > float64(currentStreams)*0.1 {
		amst.streamCount.Store(int32(optimalStreams))
		fmt.Printf("AMST: Optimized stream count: %d -> %d\n", currentStreams, optimalStreams)
	}

	if float64(abs(optimalChunkSize-currentChunkSize)) > float64(currentChunkSize)*0.1 {
		amst.chunkSize.Store(int32(optimalChunkSize))
		fmt.Printf("AMST: Optimized chunk size: %d -> %d bytes\n", currentChunkSize, optimalChunkSize)
	}

	// Record history for future optimization
	amst.optimizer.bandwidthHistory = append(amst.optimizer.bandwidthHistory, currentRate)
	if len(amst.optimizer.bandwidthHistory) > 100 {
		amst.optimizer.bandwidthHistory = amst.optimizer.bandwidthHistory[1:]
	}

	if latency > 0 {
		amst.optimizer.rttHistory = append(amst.optimizer.rttHistory, time.Duration(latency)*time.Millisecond)
		if len(amst.optimizer.rttHistory) > 100 {
			amst.optimizer.rttHistory = amst.optimizer.rttHistory[1:]
		}
	}

	amst.optimizer.lossHistory = append(amst.optimizer.lossHistory, packetLoss)
	if len(amst.optimizer.lossHistory) > 100 {
		amst.optimizer.lossHistory = amst.optimizer.lossHistory[1:]
	}

	amst.optimizer.lastOptimization = time.Now()
	amst.optimizer.optimizationCount++
}

// UpdateMetrics updates network metrics for optimization
func (amst *AMST) UpdateMetrics(latencyMs int64, packetLoss float64, transferRate int64) {
	amst.latency.Store(latencyMs)
	amst.packetLoss.Store(packetLoss)
	amst.transferRate.Store(transferRate)
}

// GetMetrics returns current AMST metrics
func (amst *AMST) GetMetrics() map[string]interface{} {
	amst.mu.RLock()
	streamMetrics := make([]map[string]interface{}, 0, len(amst.streams))
	for _, stream := range amst.streams {
		if stream.active {
			streamMetrics = append(streamMetrics, map[string]interface{}{
				"id":                stream.id,
				"bytes_transferred": stream.bytesTransferred,
				"transfer_rate":     stream.transferRate,
				"errors":            stream.errors,
				"last_active":       stream.lastActive,
			})
		}
	}
	amst.mu.RUnlock()

	return map[string]interface{}{
		"active_streams":     amst.activeStreams.Load(),
		"total_streams":      amst.streamCount.Load(),
		"chunk_size":         amst.chunkSize.Load(),
		"bytes_transferred":  amst.bytesTransferred.Load(),
		"transfer_rate":      amst.transferRate.Load(),
		"latency_ms":         amst.latency.Load(),
		"packet_loss":        amst.packetLoss.Load(),
		"error_count":        amst.errorCount.Load(),
		"stream_metrics":     streamMetrics,
	}
}

// Close closes all streams and releases resources
func (amst *AMST) Close() error {
	amst.cancel()

	amst.mu.Lock()
	defer amst.mu.Unlock()

	for _, stream := range amst.streams {
		if stream.conn != nil {
			stream.conn.Close()
		}
	}

	amst.streams = nil
	amst.activeStreams.Store(0)

	return nil
}

// abs returns absolute value of integer
func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}