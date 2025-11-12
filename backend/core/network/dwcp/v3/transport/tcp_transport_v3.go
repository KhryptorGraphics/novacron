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

// TCPTransportV3 provides internet-optimized TCP transport
// Features: BBR congestion control, adaptive streams (4-16), packet pacing
type TCPTransportV3 struct {
	// Configuration
	config *TCPTransportV3Config

	// Connection management
	remoteAddr     string
	streams        []*TCPStreamV3
	numStreams     int
	activeStreams  atomic.Int32

	// Congestion control
	congestionCtrl *CongestionController
	pacingEnabled  bool
	pacingRate     int64 // bytes per second

	// Metrics
	totalBytesSent atomic.Uint64
	totalBytesRecv atomic.Uint64
	avgRTT         atomic.Int64 // microseconds
	packetLoss     atomic.Value // float64

	// Adaptive tuning
	autoTune       bool
	chunkSize      int
	lastTuneTime   time.Time

	// Lifecycle
	ctx      context.Context
	cancel   context.CancelFunc
	mu       sync.RWMutex
	started  bool
	logger   *zap.Logger

	// Health monitoring
	healthTicker *time.Ticker
}

// TCPStreamV3 represents a single TCP stream with v3 optimizations
type TCPStreamV3 struct {
	conn            *net.TCPConn
	streamID        int
	bytesSent       atomic.Uint64
	bytesRecv       atomic.Uint64
	lastActive      time.Time
	healthy         atomic.Bool
	rawConn         syscall.RawConn
	rtt             atomic.Int64 // microseconds
	cwnd            atomic.Int32 // congestion window size
	mu              sync.Mutex
}

// TCPTransportV3Config configuration for internet-optimized TCP
type TCPTransportV3Config struct {
	RemoteAddr          string
	MinStreams          int           // 4 for internet
	MaxStreams          int           // 16 for internet
	ChunkSizeKB         int           // Adaptive chunk size
	AutoTune            bool          // Enable auto-tuning
	PacingEnabled       bool          // Enable packet pacing
	PacingRate          int64         // bytes per second
	CongestionAlgorithm string        // "bbr" or "cubic"
	ConnectTimeout      time.Duration
	ReadTimeout         time.Duration
	WriteTimeout        time.Duration
	KeepAlive           bool
	KeepAlivePeriod     time.Duration
}

// DefaultTCPTransportV3Config returns default configuration optimized for internet
func DefaultTCPTransportV3Config() *TCPTransportV3Config {
	return &TCPTransportV3Config{
		MinStreams:          4,
		MaxStreams:          16,
		ChunkSizeKB:         64, // Smaller chunks for internet
		AutoTune:            true,
		PacingEnabled:       true,
		PacingRate:          100 * 1024 * 1024, // 100 Mbps default
		CongestionAlgorithm: "bbr",
		ConnectTimeout:      30 * time.Second,
		ReadTimeout:         60 * time.Second,
		WriteTimeout:        60 * time.Second,
		KeepAlive:           true,
		KeepAlivePeriod:     30 * time.Second,
	}
}

// NewTCPTransportV3 creates a new internet-optimized TCP transport
func NewTCPTransportV3(config *TCPTransportV3Config, logger *zap.Logger) (*TCPTransportV3, error) {
	if config == nil {
		config = DefaultTCPTransportV3Config()
	}

	if logger == nil {
		logger, _ = zap.NewProduction()
	}

	ctx, cancel := context.WithCancel(context.Background())

	transport := &TCPTransportV3{
		config:        config,
		remoteAddr:    config.RemoteAddr,
		numStreams:    config.MinStreams,
		streams:       make([]*TCPStreamV3, 0, config.MaxStreams),
		pacingEnabled: config.PacingEnabled,
		pacingRate:    config.PacingRate,
		autoTune:      config.AutoTune,
		chunkSize:     config.ChunkSizeKB * 1024,
		ctx:           ctx,
		cancel:        cancel,
		logger:        logger,
		lastTuneTime:  time.Now(),
	}

	// Initialize packet loss to 0
	transport.packetLoss.Store(float64(0))

	// Create congestion controller
	if config.CongestionAlgorithm != "" {
		transport.congestionCtrl = NewCongestionController(
			config.CongestionAlgorithm,
			config.PacingRate,
			logger,
		)
	}

	return transport, nil
}

// Start initializes TCP streams
func (t *TCPTransportV3) Start(ctx context.Context) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.started {
		return fmt.Errorf("TCP transport v3 already started")
	}

	t.logger.Info("Starting TCP transport v3",
		zap.String("remote_addr", t.remoteAddr),
		zap.Int("num_streams", t.numStreams),
		zap.String("congestion_algorithm", t.config.CongestionAlgorithm))

	// Create initial streams
	for i := 0; i < t.numStreams; i++ {
		if err := t.createStream(ctx, i); err != nil {
			t.logger.Warn("Failed to create stream", zap.Int("stream_id", i), zap.Error(err))
			continue
		}
	}

	if len(t.streams) == 0 {
		return fmt.Errorf("failed to create any streams")
	}

	t.activeStreams.Store(int32(len(t.streams)))
	t.started = true

	// Start health monitoring
	t.healthTicker = time.NewTicker(10 * time.Second)
	go t.healthMonitorLoop()

	t.logger.Info("TCP transport v3 started",
		zap.Int("active_streams", len(t.streams)))

	return nil
}

// createStream creates a single TCP stream with v3 optimizations
func (t *TCPTransportV3) createStream(ctx context.Context, streamID int) error {
	connectCtx, cancel := context.WithTimeout(ctx, t.config.ConnectTimeout)
	defer cancel()

	var d net.Dialer
	conn, err := d.DialContext(connectCtx, "tcp", t.remoteAddr)
	if err != nil {
		return fmt.Errorf("dial failed: %w", err)
	}

	tcpConn := conn.(*net.TCPConn)

	// Optimize TCP settings for internet
	_ = tcpConn.SetNoDelay(true) // Disable Nagle
	if t.config.KeepAlive {
		_ = tcpConn.SetKeepAlive(true)
		_ = tcpConn.SetKeepAlivePeriod(t.config.KeepAlivePeriod)
	}

	// Set buffer sizes (smaller for internet)
	_ = tcpConn.SetReadBuffer(1024 * 1024)  // 1MB
	_ = tcpConn.SetWriteBuffer(1024 * 1024) // 1MB

	// Get raw connection for socket options
	rawConn, err := tcpConn.SyscallConn()
	if err != nil {
		tcpConn.Close()
		return fmt.Errorf("failed to get raw connection: %w", err)
	}

	// Set BBR or CUBIC congestion control
	if t.config.CongestionAlgorithm != "" {
		if err := t.setCongestionControl(rawConn, t.config.CongestionAlgorithm); err != nil {
			t.logger.Warn("Failed to set congestion control",
				zap.String("algorithm", t.config.CongestionAlgorithm),
				zap.Error(err))
		} else {
			t.logger.Debug("Congestion control set",
				zap.Int("stream_id", streamID),
				zap.String("algorithm", t.config.CongestionAlgorithm))
		}
	}

	stream := &TCPStreamV3{
		conn:       tcpConn,
		streamID:   streamID,
		lastActive: time.Now(),
		rawConn:    rawConn,
	}
	stream.healthy.Store(true)

	t.streams = append(t.streams, stream)

	t.logger.Debug("TCP stream v3 created",
		zap.Int("stream_id", streamID),
		zap.String("local_addr", tcpConn.LocalAddr().String()))

	return nil
}

// Send data with internet optimizations
func (t *TCPTransportV3) Send(ctx context.Context, data []byte) error {
	t.mu.RLock()
	defer t.mu.RUnlock()

	if !t.started {
		return fmt.Errorf("TCP transport v3 not started")
	}

	if len(data) == 0 {
		return fmt.Errorf("no data to send")
	}

	activeStreams := t.getHealthyStreams()
	if len(activeStreams) == 0 {
		return fmt.Errorf("no healthy streams available")
	}

	// Apply auto-tuning if enabled
	if t.autoTune && time.Since(t.lastTuneTime) > 5*time.Second {
		t.autoTuneParameters()
	}

	// Split data into chunks
	numChunks := (len(data) + t.chunkSize - 1) / t.chunkSize
	chunks := make([][]byte, numChunks)

	for i := 0; i < numChunks; i++ {
		start := i * t.chunkSize
		end := start + t.chunkSize
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
		go func(s *TCPStreamV3, chunkData []byte, chunkID int) {
			defer wg.Done()
			if err := t.sendChunk(ctx, s, chunkData, chunkID); err != nil {
				errChan <- fmt.Errorf("stream %d: %w", s.streamID, err)
			}
		}(stream, chunk, i)
	}

	wg.Wait()
	close(errChan)

	// Check for errors
	if len(errChan) > 0 {
		return <-errChan
	}

	t.totalBytesSent.Add(uint64(len(data)))
	return nil
}

// sendChunk sends a single chunk with pacing and congestion control
func (t *TCPTransportV3) sendChunk(ctx context.Context, stream *TCPStreamV3, chunk []byte, chunkID int) error {
	stream.mu.Lock()
	defer stream.mu.Unlock()

	// Apply congestion control
	if t.congestionCtrl != nil {
		delay := t.congestionCtrl.GetSendDelay(len(chunk))
		if delay > 0 {
			time.Sleep(delay)
		}
	}

	// Apply packet pacing
	if t.pacingEnabled && t.pacingRate > 0 {
		delay := time.Duration(float64(len(chunk)) / float64(t.pacingRate) * float64(time.Second))
		time.Sleep(delay)
	}

	// Set write timeout
	if t.config.WriteTimeout > 0 {
		stream.conn.SetWriteDeadline(time.Now().Add(t.config.WriteTimeout))
	}

	// Write chunk header (ID and size)
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
		stream.healthy.Store(false)
		return fmt.Errorf("write header: %w", err)
	}

	// Write chunk data
	if _, err := stream.conn.Write(chunk); err != nil {
		stream.healthy.Store(false)
		return fmt.Errorf("write chunk: %w", err)
	}

	stream.bytesSent.Add(uint64(len(chunk) + 8))
	stream.lastActive = time.Now()

	// Notify congestion controller of successful send
	if t.congestionCtrl != nil {
		t.congestionCtrl.OnPacketSent(len(chunk))
	}

	return nil
}

// Receive data from streams
func (t *TCPTransportV3) Receive(ctx context.Context, expectedSize int) ([]byte, error) {
	t.mu.RLock()
	defer t.mu.RUnlock()

	if !t.started {
		return nil, fmt.Errorf("TCP transport v3 not started")
	}

	activeStreams := t.getHealthyStreams()
	if len(activeStreams) == 0 {
		return nil, fmt.Errorf("no healthy streams available")
	}

	numChunks := (expectedSize + t.chunkSize - 1) / t.chunkSize
	chunks := make(map[int][]byte)
	var chunksMu sync.Mutex

	var wg sync.WaitGroup
	errChan := make(chan error, len(activeStreams))

	for _, stream := range activeStreams {
		wg.Add(1)
		go func(s *TCPStreamV3) {
			defer wg.Done()

			for {
				chunkID, chunkData, err := t.receiveChunk(s)
				if err != nil {
					if err != io.EOF {
						errChan <- fmt.Errorf("stream %d: %w", s.streamID, err)
					}
					return
				}

				chunksMu.Lock()
				chunks[chunkID] = chunkData
				chunksMu.Unlock()

				if len(chunks) >= numChunks {
					return
				}
			}
		}(stream)
	}

	wg.Wait()
	close(errChan)

	if len(errChan) > 0 {
		return nil, <-errChan
	}

	// Reassemble chunks
	result := make([]byte, 0, expectedSize)
	for i := 0; i < numChunks; i++ {
		chunk, exists := chunks[i]
		if !exists {
			return nil, fmt.Errorf("missing chunk %d", i)
		}
		result = append(result, chunk...)
	}

	t.totalBytesRecv.Add(uint64(len(result)))
	return result, nil
}

// receiveChunk receives a single chunk
func (t *TCPTransportV3) receiveChunk(stream *TCPStreamV3) (int, []byte, error) {
	stream.mu.Lock()
	defer stream.mu.Unlock()

	if t.config.ReadTimeout > 0 {
		stream.conn.SetReadDeadline(time.Now().Add(t.config.ReadTimeout))
	}

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

// AdjustStreams dynamically adjusts stream count for internet conditions
func (t *TCPTransportV3) AdjustStreams(bandwidthMbps, latencyMs float64) error {
	if !t.autoTune {
		return nil
	}

	t.mu.Lock()
	defer t.mu.Unlock()

	// For internet: optimal streams = bandwidth / (latency * 0.2)
	// More conservative than datacenter
	optimalStreams := bandwidthMbps / (latencyMs * 0.2)
	targetStreams := int(math.Min(float64(t.config.MaxStreams),
		math.Max(float64(t.config.MinStreams), optimalStreams)))

	currentStreams := len(t.streams)
	if targetStreams == currentStreams {
		return nil
	}

	t.logger.Info("Adjusting stream count for internet",
		zap.Int("current", currentStreams),
		zap.Int("target", targetStreams),
		zap.Float64("bandwidth_mbps", bandwidthMbps),
		zap.Float64("latency_ms", latencyMs))

	// Adjust streams gradually
	if targetStreams > currentStreams {
		// Add streams
		for i := currentStreams; i < targetStreams; i++ {
			if err := t.createStream(t.ctx, i); err != nil {
				t.logger.Warn("Failed to create additional stream", zap.Error(err))
				continue
			}
		}
	} else {
		// Remove streams
		for i := targetStreams; i < currentStreams; i++ {
			if stream := t.streams[i]; stream != nil {
				stream.conn.Close()
			}
		}
		t.streams = t.streams[:targetStreams]
	}

	t.numStreams = len(t.streams)
	t.activeStreams.Store(int32(t.numStreams))

	return nil
}

// autoTuneParameters automatically tunes chunk size and pacing based on conditions
func (t *TCPTransportV3) autoTuneParameters() {
	// Get average RTT
	avgRTT := time.Duration(t.avgRTT.Load()) * time.Microsecond
	packetLoss := t.packetLoss.Load().(float64)

	// Adjust chunk size based on packet loss
	if packetLoss > 0.05 { // >5% loss
		// Reduce chunk size for better recovery
		t.chunkSize = int(float64(t.chunkSize) * 0.8)
		if t.chunkSize < 16*1024 { // Min 16KB
			t.chunkSize = 16 * 1024
		}
	} else if packetLoss < 0.01 && avgRTT < 100*time.Millisecond {
		// Increase chunk size for efficiency
		t.chunkSize = int(float64(t.chunkSize) * 1.2)
		if t.chunkSize > 512*1024 { // Max 512KB for internet
			t.chunkSize = 512 * 1024
		}
	}

	t.lastTuneTime = time.Now()

	t.logger.Debug("Auto-tuned parameters",
		zap.Int("chunk_size_kb", t.chunkSize/1024),
		zap.Float64("packet_loss", packetLoss),
		zap.Duration("avg_rtt", avgRTT))
}

// getHealthyStreams returns list of healthy streams
func (t *TCPTransportV3) getHealthyStreams() []*TCPStreamV3 {
	healthy := make([]*TCPStreamV3, 0, len(t.streams))
	for _, stream := range t.streams {
		if stream.healthy.Load() {
			healthy = append(healthy, stream)
		}
	}
	return healthy
}

// setCongestionControl sets TCP congestion control algorithm
func (t *TCPTransportV3) setCongestionControl(rawConn syscall.RawConn, algorithm string) error {
	var setErr error
	controlErr := rawConn.Control(func(fd uintptr) {
		err := unix.SetsockoptString(int(fd), unix.IPPROTO_TCP, unix.TCP_CONGESTION, algorithm)
		if err != nil {
			setErr = fmt.Errorf("failed to set %s: %w", algorithm, err)
		}
	})

	if controlErr != nil {
		return controlErr
	}
	return setErr
}

// healthMonitorLoop monitors stream health
func (t *TCPTransportV3) healthMonitorLoop() {
	for {
		select {
		case <-t.ctx.Done():
			return
		case <-t.healthTicker.C:
			t.performHealthCheck()
		}
	}
}

// performHealthCheck checks all streams
func (t *TCPTransportV3) performHealthCheck() {
	t.mu.RLock()
	streams := make([]*TCPStreamV3, len(t.streams))
	copy(streams, t.streams)
	t.mu.RUnlock()

	var unhealthyCount int32
	for _, stream := range streams {
		if stream == nil {
			continue
		}

		// Check if stream is idle too long
		if time.Since(stream.lastActive) > 120*time.Second {
			stream.healthy.Store(false)
			unhealthyCount++
		}
	}

	activeCount := int32(len(streams)) - unhealthyCount
	t.activeStreams.Store(activeCount)
}

// GetMetrics returns transport metrics
func (t *TCPTransportV3) GetMetrics() TCPTransportV3Metrics {
	t.mu.RLock()
	defer t.mu.RUnlock()

	return TCPTransportV3Metrics{
		ActiveStreams:  int(t.activeStreams.Load()),
		TotalStreams:   len(t.streams),
		TotalBytesSent: t.totalBytesSent.Load(),
		TotalBytesRecv: t.totalBytesRecv.Load(),
		AvgRTT:         time.Duration(t.avgRTT.Load()) * time.Microsecond,
		PacketLoss:     t.packetLoss.Load().(float64),
		ChunkSize:      t.chunkSize,
		PacingRate:     t.pacingRate,
	}
}

// TCPTransportV3Metrics metrics for v3 transport
type TCPTransportV3Metrics struct {
	ActiveStreams  int
	TotalStreams   int
	TotalBytesSent uint64
	TotalBytesRecv uint64
	AvgRTT         time.Duration
	PacketLoss     float64
	ChunkSize      int
	PacingRate     int64
}

// Close gracefully shuts down transport
func (t *TCPTransportV3) Close() error {
	t.mu.Lock()
	defer t.mu.Unlock()

	if !t.started {
		return nil
	}

	t.logger.Info("Closing TCP transport v3")

	t.cancel()

	if t.healthTicker != nil {
		t.healthTicker.Stop()
	}

	// Close all streams
	for _, stream := range t.streams {
		if stream != nil && stream.conn != nil {
			stream.conn.Close()
		}
	}

	t.streams = nil
	t.started = false

	t.logger.Info("TCP transport v3 closed")
	return nil
}
