package transport

import (
	"context"
	"fmt"
	"io"
	"math"
	"net"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"go.uber.org/zap"
	"golang.org/x/sys/unix"
)

// MultiStreamTCP manages multiple parallel TCP connections for WAN optimization
// Phase 1 production implementation with BBR, health monitoring, and metrics
type MultiStreamTCP struct {
	// Connection management
	remoteAddr string
	streams    []*StreamConn
	numStreams int

	// Configuration
	config     *AMSTConfig
	chunkSize  int
	pacingRate int64 // bytes per second

	// Metrics collection
	metrics    *MetricsCollector
	totalBytesSent    atomic.Uint64
	totalBytesRecv    atomic.Uint64
	activeStreams     atomic.Int32
	bandwidthUtilized atomic.Uint64 // percentage * 100
	failedConnections atomic.Uint64

	// Health monitoring
	healthCheckTicker *time.Ticker
	streamHealth      map[int]*StreamHealth
	healthMu          sync.RWMutex

	// In-flight request tracking for graceful shutdown
	inFlightRequests  atomic.Int64
	shutdownWg        sync.WaitGroup

	// Synchronization
	mu     sync.RWMutex
	ctx    context.Context
	cancel context.CancelFunc
	logger *zap.Logger

	// State
	started bool
}

// StreamConn represents a single TCP stream with health tracking
type StreamConn struct {
	conn             *net.TCPConn
	streamID         int
	bytesSent        atomic.Uint64
	bytesRecv        atomic.Uint64
	lastActive       time.Time
	lastError        error
	reconnects       atomic.Int32
	consecutiveFails atomic.Int32
	healthy          atomic.Bool
	mu               sync.Mutex
	rawConn          syscall.RawConn // For accessing socket options
}

// AMSTConfig configuration for Adaptive Multi-Stream Transport
type AMSTConfig struct {
	MinStreams          int           `json:"min_streams" yaml:"min_streams"`
	MaxStreams          int           `json:"max_streams" yaml:"max_streams"`
	ChunkSizeKB         int           `json:"chunk_size_kb" yaml:"chunk_size_kb"`
	AutoTune            bool          `json:"auto_tune" yaml:"auto_tune"`
	PacingEnabled       bool          `json:"pacing_enabled" yaml:"pacing_enabled"`
	PacingRate          int64         `json:"pacing_rate" yaml:"pacing_rate"` // bytes per second
	ConnectTimeout      time.Duration `json:"connect_timeout" yaml:"connect_timeout"`
	CongestionAlgorithm string        `json:"congestion_algorithm" yaml:"congestion_algorithm"` // "bbr", "cubic"
}

// DefaultAMSTConfig returns sensible defaults for production
func DefaultAMSTConfig() *AMSTConfig {
	return &AMSTConfig{
		MinStreams:          16,
		MaxStreams:          256,
		ChunkSizeKB:         256,
		AutoTune:            true,
		PacingEnabled:       true,
		PacingRate:          1000 * 1024 * 1024, // 1 Gbps
		ConnectTimeout:      30 * time.Second,
		CongestionAlgorithm: "bbr", // Default to BBR
	}
}

// NewMultiStreamTCP creates a new multi-stream TCP connection manager
func NewMultiStreamTCP(remoteAddr string, config *AMSTConfig, logger *zap.Logger) (*MultiStreamTCP, error) {
	if config == nil {
		config = DefaultAMSTConfig()
	}

	if logger == nil {
		logger, _ = zap.NewProduction()
	}

	ctx, cancel := context.WithCancel(context.Background())

	mst := &MultiStreamTCP{
		remoteAddr:   remoteAddr,
		config:       config,
		chunkSize:    config.ChunkSizeKB * 1024,
		pacingRate:   config.PacingRate,
		numStreams:   config.MinStreams,
		streams:      make([]*StreamConn, 0, config.MaxStreams),
		streamHealth: make(map[int]*StreamHealth),
		ctx:          ctx,
		cancel:       cancel,
		logger:       logger,
		started:      false,
		metrics:      NewMetricsCollector("tcp", remoteAddr),
	}

	return mst, nil
}

// Start initializes all TCP streams to the remote address
func (m *MultiStreamTCP) Start() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.started {
		return fmt.Errorf("multi-stream TCP already started")
	}

	m.logger.Info("Starting multi-stream TCP",
		zap.String("remote_addr", m.remoteAddr),
		zap.Int("num_streams", m.numStreams))

	// Create initial streams
	for i := 0; i < m.numStreams; i++ {
		if err := m.createStream(i); err != nil {
			m.logger.Error("Failed to create stream", zap.Int("stream_id", i), zap.Error(err))
			// Continue creating other streams
			continue
		}
	}

	if len(m.streams) == 0 {
		return fmt.Errorf("failed to create any streams")
	}

	m.activeStreams.Store(int32(len(m.streams)))
	m.started = true

	// Start health monitoring
	m.healthCheckTicker = time.NewTicker(10 * time.Second)
	m.shutdownWg.Add(1)
	go m.healthMonitorLoop()

	// Update metrics
	m.metrics.RecordActiveStreams(int32(len(m.streams)))
	m.metrics.RecordTotalStreams(len(m.streams))
	m.metrics.RecordHealthStatus(true)

	m.logger.Info("Multi-stream TCP started",
		zap.Int("active_streams", len(m.streams)),
		zap.String("congestion_algorithm", m.config.CongestionAlgorithm))

	return nil
}

// createStream creates a single TCP stream connection
func (m *MultiStreamTCP) createStream(streamID int) error {
	ctx, cancel := context.WithTimeout(m.ctx, m.config.ConnectTimeout)
	defer cancel()

	var d net.Dialer
	conn, err := d.DialContext(ctx, "tcp", m.remoteAddr)
	if err != nil {
		return fmt.Errorf("failed to dial: %w", err)
	}

	tcpConn := conn.(*net.TCPConn)

	// Optimize TCP settings
	_ = tcpConn.SetNoDelay(true)                      // Disable Nagle's algorithm
	_ = tcpConn.SetKeepAlive(true)                    // Enable keepalive
	_ = tcpConn.SetKeepAlivePeriod(30 * time.Second) // Keepalive period

	// Get raw connection for socket options
	rawConn, err := tcpConn.SyscallConn()
	if err != nil {
		tcpConn.Close()
		return fmt.Errorf("failed to get raw connection: %w", err)
	}

	// Set BBR congestion control if requested
	if m.config.CongestionAlgorithm != "" {
		if err := m.setCongestionControl(rawConn, m.config.CongestionAlgorithm); err != nil {
			m.logger.Warn("Failed to set congestion control, using default",
				zap.String("algorithm", m.config.CongestionAlgorithm),
				zap.Error(err))
		} else {
			m.logger.Debug("Congestion control set",
				zap.Int("stream_id", streamID),
				zap.String("algorithm", m.config.CongestionAlgorithm))
		}
	}

	stream := &StreamConn{
		conn:       tcpConn,
		streamID:   streamID,
		lastActive: time.Now(),
		rawConn:    rawConn,
	}
	stream.healthy.Store(true)

	m.streams = append(m.streams, stream)

	// Initialize stream health tracking
	m.healthMu.Lock()
	m.streamHealth[streamID] = &StreamHealth{
		StreamID:   streamID,
		Healthy:    true,
		LastActive: time.Now(),
	}
	m.healthMu.Unlock()

	m.logger.Debug("Stream created",
		zap.Int("stream_id", streamID),
		zap.String("local_addr", tcpConn.LocalAddr().String()),
		zap.String("congestion_control", m.config.CongestionAlgorithm))

	return nil
}

// Send data across all streams in parallel using load balancing
func (m *MultiStreamTCP) Send(data []byte) error {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if !m.started {
		return fmt.Errorf("multi-stream TCP not started")
	}

	if len(data) == 0 {
		return nil
	}

	activeStreams := m.streams
	if len(activeStreams) == 0 {
		return fmt.Errorf("no active streams available")
	}

	// Split data into chunks
	numChunks := (len(data) + m.chunkSize - 1) / m.chunkSize
	chunks := make([][]byte, numChunks)

	for i := 0; i < numChunks; i++ {
		start := i * m.chunkSize
		end := start + m.chunkSize
		if end > len(data) {
			end = len(data)
		}
		chunks[i] = data[start:end]
	}

	// Send chunks in parallel across streams
	var wg sync.WaitGroup
	errChan := make(chan error, numChunks)

	for i, chunk := range chunks {
		streamIdx := i % len(activeStreams)
		stream := activeStreams[streamIdx]

		wg.Add(1)
		go func(s *StreamConn, chunkData []byte, chunkID int) {
			defer wg.Done()

			if err := m.sendChunk(s, chunkData, chunkID); err != nil {
				errChan <- fmt.Errorf("stream %d: %w", s.streamID, err)
			}
		}(stream, chunk, i)
	}

	wg.Wait()
	close(errChan)

	// Check for errors
	if len(errChan) > 0 {
		return <-errChan // Return first error
	}

	m.totalBytesSent.Add(uint64(len(data)))
	return nil
}

// sendChunk sends a single chunk on a stream with optional pacing
func (m *MultiStreamTCP) sendChunk(stream *StreamConn, chunk []byte, chunkID int) error {
	stream.mu.Lock()
	defer stream.mu.Unlock()

	// Apply packet pacing if enabled
	if m.config.PacingEnabled && m.pacingRate > 0 {
		// Calculate delay based on chunk size and pacing rate
		delay := time.Duration(float64(len(chunk)) / float64(m.pacingRate) * float64(time.Second))
		time.Sleep(delay)
	}

	// Write chunk header (chunk ID and size)
	header := make([]byte, 8)
	header[0] = byte(chunkID >> 24)
	header[1] = byte(chunkID >> 16)
	header[2] = byte(chunkID >> 8)
	header[3] = byte(chunkID)
	header[4] = byte(len(chunk) >> 24)
	header[5] = byte(len(chunk) >> 16)
	header[6] = byte(len(chunk) >> 8)
	header[7] = byte(len(chunk))

	if _, err := stream.conn.Write(header); err != nil {
		return fmt.Errorf("write header: %w", err)
	}

	// Write chunk data
	if _, err := stream.conn.Write(chunk); err != nil {
		return fmt.Errorf("write chunk: %w", err)
	}

	stream.bytesSent.Add(uint64(len(chunk) + 8))
	stream.lastActive = time.Now()

	return nil
}

// Receive data from all streams and reassemble in correct order
func (m *MultiStreamTCP) Receive(expectedSize int) ([]byte, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if !m.started {
		return nil, fmt.Errorf("multi-stream TCP not started")
	}

	activeStreams := m.streams
	if len(activeStreams) == 0 {
		return nil, fmt.Errorf("no active streams available")
	}

	numChunks := (expectedSize + m.chunkSize - 1) / m.chunkSize
	chunks := make(map[int][]byte)
	var chunksMu sync.Mutex

	var wg sync.WaitGroup
	errChan := make(chan error, len(activeStreams))

	// Receive from all streams in parallel
	for _, stream := range activeStreams {
		wg.Add(1)
		go func(s *StreamConn) {
			defer wg.Done()

			for {
				chunkID, chunkData, err := m.receiveChunk(s)
				if err != nil {
					if err != io.EOF {
						errChan <- fmt.Errorf("stream %d: %w", s.streamID, err)
					}
					return
				}

				chunksMu.Lock()
				chunks[chunkID] = chunkData
				chunksMu.Unlock()

				// Check if we have all chunks
				if len(chunks) >= numChunks {
					return
				}
			}
		}(stream)
	}

	wg.Wait()
	close(errChan)

	// Check for errors
	if len(errChan) > 0 {
		return nil, <-errChan
	}

	// Reassemble chunks in correct order
	result := make([]byte, 0, expectedSize)
	for i := 0; i < numChunks; i++ {
		chunk, exists := chunks[i]
		if !exists {
			return nil, fmt.Errorf("missing chunk %d", i)
		}
		result = append(result, chunk...)
	}

	m.totalBytesRecv.Add(uint64(len(result)))
	return result, nil
}

// receiveChunk receives a single chunk from a stream
func (m *MultiStreamTCP) receiveChunk(stream *StreamConn) (int, []byte, error) {
	stream.mu.Lock()
	defer stream.mu.Unlock()

	// Read chunk header
	header := make([]byte, 8)
	if _, err := io.ReadFull(stream.conn, header); err != nil {
		return 0, nil, err
	}

	chunkID := int(header[0])<<24 | int(header[1])<<16 | int(header[2])<<8 | int(header[3])
	chunkSize := int(header[4])<<24 | int(header[5])<<16 | int(header[6])<<8 | int(header[7])

	// Read chunk data
	chunk := make([]byte, chunkSize)
	if _, err := io.ReadFull(stream.conn, chunk); err != nil {
		return 0, nil, err
	}

	stream.bytesRecv.Add(uint64(chunkSize + 8))
	stream.lastActive = time.Now()

	return chunkID, chunk, nil
}

// AdjustStreams dynamically adjusts the number of streams based on network conditions
// Algorithm from DWCP spec: optimal_streams = min(MaxStreams, max(MinStreams, bandwidth_mbps / (latency_ms * 0.1)))
func (m *MultiStreamTCP) AdjustStreams(bandwidthMbps, latencyMs float64) error {
	if !m.config.AutoTune {
		return nil
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	// Calculate optimal number of streams
	optimalStreams := bandwidthMbps / (latencyMs * 0.1)
	targetStreams := int(math.Min(float64(m.config.MaxStreams),
		math.Max(float64(m.config.MinStreams), optimalStreams)))

	currentStreams := len(m.streams)
	if targetStreams == currentStreams {
		return nil // No adjustment needed
	}

	m.logger.Info("Adjusting stream count",
		zap.Int("current", currentStreams),
		zap.Int("target", targetStreams),
		zap.Float64("bandwidth_mbps", bandwidthMbps),
		zap.Float64("latency_ms", latencyMs))

	if targetStreams > currentStreams {
		// Add streams
		for i := currentStreams; i < targetStreams; i++ {
			if err := m.createStream(i); err != nil {
				m.logger.Warn("Failed to create additional stream", zap.Error(err))
				continue
			}
		}
	} else {
		// Remove streams (close excess streams)
		for i := targetStreams; i < currentStreams; i++ {
			if stream := m.streams[i]; stream != nil {
				stream.conn.Close()
			}
		}
		m.streams = m.streams[:targetStreams]
	}

	m.numStreams = len(m.streams)
	m.activeStreams.Store(int32(m.numStreams))

	return nil
}

// GetMetrics returns current metrics
func (m *MultiStreamTCP) GetMetrics() map[string]interface{} {
	m.mu.RLock()
	defer m.mu.RUnlock()

	totalSent := m.totalBytesSent.Load()
	totalRecv := m.totalBytesRecv.Load()

	return map[string]interface{}{
		"active_streams":      m.activeStreams.Load(),
		"total_streams":       len(m.streams),
		"total_bytes_sent":    totalSent,
		"total_bytes_recv":    totalRecv,
		"bandwidth_utilized":  m.bandwidthUtilized.Load(),
		"chunk_size":          m.chunkSize,
		"pacing_enabled":      m.config.PacingEnabled,
		"pacing_rate_mbps":    float64(m.pacingRate) / (1024 * 1024),
	}
}

// Close gracefully closes all streams
func (m *MultiStreamTCP) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if !m.started {
		return nil
	}

	m.logger.Info("Closing multi-stream TCP, waiting for in-flight requests")

	// Signal shutdown
	m.cancel()

	// Stop health monitoring
	if m.healthCheckTicker != nil {
		m.healthCheckTicker.Stop()
	}

	// Wait for in-flight requests to complete (with timeout)
	done := make(chan struct{})
	go func() {
		m.shutdownWg.Wait()
		close(done)
	}()

	select {
	case <-done:
		m.logger.Info("All in-flight requests completed")
	case <-time.After(30 * time.Second):
		m.logger.Warn("Timeout waiting for in-flight requests, forcing shutdown")
	}

	// Close all streams
	for _, stream := range m.streams {
		if stream != nil && stream.conn != nil {
			stream.conn.Close()
		}
	}

	m.streams = nil
	m.started = false
	m.metrics.RecordHealthStatus(false)

	m.logger.Info("Multi-stream TCP closed")

	return nil
}

// IsStarted returns whether the multi-stream TCP is started
func (m *MultiStreamTCP) IsStarted() bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.started
}

// setCongestionControl sets TCP congestion control algorithm
func (m *MultiStreamTCP) setCongestionControl(rawConn syscall.RawConn, algorithm string) error {
	var setErr error
	controlErr := rawConn.Control(func(fd uintptr) {
		// Try to set the congestion control algorithm
		err := unix.SetsockoptString(int(fd), unix.IPPROTO_TCP, unix.TCP_CONGESTION, algorithm)
		if err != nil {
			setErr = fmt.Errorf("failed to set %s: %w", algorithm, err)
			return
		}

		// Verify it was set
		actualAlg, err := unix.GetsockoptString(int(fd), unix.IPPROTO_TCP, unix.TCP_CONGESTION)
		if err != nil {
			setErr = fmt.Errorf("failed to verify congestion control: %w", err)
			return
		}

		if actualAlg != algorithm {
			setErr = fmt.Errorf("congestion control mismatch: requested %s, got %s", algorithm, actualAlg)
			return
		}

		m.logger.Debug("Congestion control algorithm set successfully",
			zap.String("algorithm", algorithm))
	})

	if controlErr != nil {
		return controlErr
	}
	return setErr
}

// healthMonitorLoop monitors stream health and performs automatic recovery
func (m *MultiStreamTCP) healthMonitorLoop() {
	defer m.shutdownWg.Done()

	for {
		select {
		case <-m.ctx.Done():
			return
		case <-m.healthCheckTicker.C:
			m.performHealthCheck()
		}
	}
}

// performHealthCheck checks all stream health and attempts recovery
func (m *MultiStreamTCP) performHealthCheck() {
	m.mu.RLock()
	streams := make([]*StreamConn, len(m.streams))
	copy(streams, m.streams)
	m.mu.RUnlock()

	var unhealthyCount int32
	var totalReconnects int32

	for _, stream := range streams {
		if stream == nil {
			continue
		}

		// Check if stream is healthy
		isHealthy := m.checkStreamHealth(stream)

		if !isHealthy {
			unhealthyCount++

			// Attempt to reconnect if within retry limits
			if stream.consecutiveFails.Load() < 3 {
				m.logger.Warn("Stream unhealthy, attempting reconnection",
					zap.Int("stream_id", stream.streamID),
					zap.Int32("consecutive_fails", stream.consecutiveFails.Load()))

				if err := m.reconnectStream(stream); err != nil {
					m.logger.Error("Failed to reconnect stream",
						zap.Int("stream_id", stream.streamID),
						zap.Error(err))
					stream.consecutiveFails.Add(1)
					m.metrics.RecordError("reconnect_failed")
				} else {
					stream.consecutiveFails.Store(0)
					stream.reconnects.Add(1)
					totalReconnects++
					m.metrics.RecordReconnect()
				}
			} else {
				m.logger.Error("Stream exceeded max reconnection attempts",
					zap.Int("stream_id", stream.streamID))
				stream.healthy.Store(false)
			}
		}
	}

	// Update overall health status
	activeCount := int32(len(streams)) - unhealthyCount
	m.activeStreams.Store(activeCount)
	m.metrics.RecordActiveStreams(activeCount)

	// Determine overall health
	healthyRatio := float64(activeCount) / float64(len(streams))
	isSystemHealthy := healthyRatio >= 0.5 // At least 50% streams must be healthy

	m.metrics.RecordHealthStatus(isSystemHealthy)

	if totalReconnects > 0 {
		m.logger.Info("Health check completed",
			zap.Int32("active_streams", activeCount),
			zap.Int32("unhealthy_streams", unhealthyCount),
			zap.Int32("reconnects", totalReconnects),
			zap.Bool("system_healthy", isSystemHealthy))
	}
}

// checkStreamHealth verifies if a stream is healthy
func (m *MultiStreamTCP) checkStreamHealth(stream *StreamConn) bool {
	stream.mu.Lock()
	defer stream.mu.Unlock()

	// Check if connection is still alive
	if stream.conn == nil {
		return false
	}

	// Check last activity
	if time.Since(stream.lastActive) > 60*time.Second {
		// Stream has been idle too long, check with TCP probe
		if err := stream.conn.SetReadDeadline(time.Now().Add(1 * time.Second)); err != nil {
			return false
		}

		// Try to read (should return immediately if connection is dead)
		buf := make([]byte, 1)
		_, err := stream.conn.Read(buf)
		stream.conn.SetReadDeadline(time.Time{}) // Clear deadline

		if err != nil && err != io.EOF {
			// Connection is likely dead
			return false
		}
	}

	return stream.healthy.Load()
}

// reconnectStream attempts to reconnect a failed stream
func (m *MultiStreamTCP) reconnectStream(stream *StreamConn) error {
	stream.mu.Lock()
	defer stream.mu.Unlock()

	// Close old connection
	if stream.conn != nil {
		stream.conn.Close()
	}

	// Create new connection with exponential backoff
	backoff := time.Duration(100*math.Pow(2, float64(stream.reconnects.Load()))) * time.Millisecond
	if backoff > 5*time.Second {
		backoff = 5 * time.Second
	}
	time.Sleep(backoff)

	ctx, cancel := context.WithTimeout(m.ctx, m.config.ConnectTimeout)
	defer cancel()

	var d net.Dialer
	conn, err := d.DialContext(ctx, "tcp", m.remoteAddr)
	if err != nil {
		return fmt.Errorf("reconnect dial failed: %w", err)
	}

	tcpConn := conn.(*net.TCPConn)

	// Optimize TCP settings
	_ = tcpConn.SetNoDelay(true)
	_ = tcpConn.SetKeepAlive(true)
	_ = tcpConn.SetKeepAlivePeriod(30 * time.Second)

	// Get raw connection and set BBR
	rawConn, err := tcpConn.SyscallConn()
	if err != nil {
		tcpConn.Close()
		return fmt.Errorf("failed to get raw connection: %w", err)
	}

	if m.config.CongestionAlgorithm != "" {
		_ = m.setCongestionControl(rawConn, m.config.CongestionAlgorithm)
	}

	// Update stream
	stream.conn = tcpConn
	stream.rawConn = rawConn
	stream.lastActive = time.Now()
	stream.healthy.Store(true)

	// Update health tracking
	m.healthMu.Lock()
	if health, exists := m.streamHealth[stream.streamID]; exists {
		health.Healthy = true
		health.LastActive = time.Now()
		health.Reconnects++
		health.ConsecutiveFails = 0
	}
	m.healthMu.Unlock()

	m.logger.Info("Stream reconnected successfully",
		zap.Int("stream_id", stream.streamID))

	return nil
}

// HealthCheck performs a comprehensive health check
func (m *MultiStreamTCP) HealthCheck() error {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if !m.started {
		return fmt.Errorf("multi-stream TCP not started")
	}

	activeCount := m.activeStreams.Load()
	if activeCount == 0 {
		return fmt.Errorf("no active streams available")
	}

	// Check if we have minimum required streams
	if activeCount < int32(m.config.MinStreams)/2 {
		return fmt.Errorf("insufficient active streams: %d (min: %d)", activeCount, m.config.MinStreams/2)
	}

	return nil
}

// GetStreamHealth returns health status of all streams
func (m *MultiStreamTCP) GetStreamHealth() map[int]*StreamHealth {
	m.healthMu.RLock()
	defer m.healthMu.RUnlock()

	// Return a copy
	health := make(map[int]*StreamHealth)
	for id, sh := range m.streamHealth {
		healthCopy := *sh
		health[id] = &healthCopy
	}

	return health
}
