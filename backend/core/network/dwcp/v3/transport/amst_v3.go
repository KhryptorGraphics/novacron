package transport

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/transport"
	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/upgrade"
	"go.uber.org/zap"
)

// AMSTv3 provides hybrid datacenter + internet transport with mode detection
// Backward compatible with v1 RDMA for datacenter mode
// New internet-optimized TCP transport for internet mode
type AMSTv3 struct {
	// Configuration
	config *AMSTv3Config

	// Mode detection
	modeDetector *upgrade.ModeDetector
	currentMode  atomic.Value // upgrade.NetworkMode

	// Transport layers
	datacenterTransport *transport.RDMATransport    // v1 RDMA for datacenter
	internetTransport   *TCPTransportV3             // v3 TCP for internet
	congestionCtrl      *CongestionController       // BBR/CUBIC controller

	// Metrics and monitoring
	metrics           *transport.MetricsCollector
	totalBytesSent    atomic.Uint64
	totalBytesRecv    atomic.Uint64
	activeStreams     atomic.Int32
	modeTransitions   atomic.Uint64

	// Lifecycle management
	ctx       context.Context
	cancel    context.CancelFunc
	mu        sync.RWMutex
	started   bool
	logger    *zap.Logger

	// Adaptive optimization
	lastModeCheck     time.Time
	modeCheckInterval time.Duration
}

// AMSTv3Config configuration for hybrid transport
type AMSTv3Config struct {
	// Transport selection
	EnableDatacenter bool   // Enable datacenter mode (RDMA)
	EnableInternet   bool   // Enable internet mode (TCP)
	AutoMode         bool   // Automatically detect and switch modes

	// Datacenter settings (v1 compatibility)
	DatacenterStreams int    // 32-512 streams for datacenter
	RDMADevice        string // RDMA device name
	RDMAPort          int    // RDMA port

	// Internet settings (v3 features)
	InternetStreams int    // 4-16 streams for internet
	CongestionAlgorithm string // "bbr" or "cubic"
	PacingEnabled       bool   // Enable packet pacing
	PacingRate          int64  // bytes per second

	// Adaptive tuning
	AutoTune            bool          // Enable auto-tuning
	ChunkSizeKB         int           // Chunk size in KB
	ConnectTimeout      time.Duration // Connection timeout
	ModeCheckInterval   time.Duration // How often to check mode
	ModeSwitchThreshold float64       // Threshold for mode switching (0-1)

	// Common settings
	RemoteAddr string // Remote address
	MinStreams int    // Minimum streams
	MaxStreams int    // Maximum streams
}

// DefaultAMSTv3Config returns default configuration
func DefaultAMSTv3Config() *AMSTv3Config {
	return &AMSTv3Config{
		EnableDatacenter:    true,
		EnableInternet:      true,
		AutoMode:            true,
		DatacenterStreams:   64,  // High stream count for datacenter
		InternetStreams:     8,   // Low stream count for internet
		CongestionAlgorithm: "bbr",
		PacingEnabled:       true,
		PacingRate:          1000 * 1024 * 1024, // 1 Gbps default
		AutoTune:            true,
		ChunkSizeKB:         256,
		ConnectTimeout:      30 * time.Second,
		ModeCheckInterval:   5 * time.Second,
		ModeSwitchThreshold: 0.7, // Switch if 70% confident
		MinStreams:          4,
		MaxStreams:          512,
	}
}

// NewAMSTv3 creates a new hybrid AMST v3 transport
func NewAMSTv3(config *AMSTv3Config, detector *upgrade.ModeDetector, logger *zap.Logger) (*AMSTv3, error) {
	if config == nil {
		config = DefaultAMSTv3Config()
	}

	if detector == nil {
		detector = upgrade.NewModeDetector()
	}

	if logger == nil {
		logger, _ = zap.NewProduction()
	}

	ctx, cancel := context.WithCancel(context.Background())

	amst := &AMSTv3{
		config:            config,
		modeDetector:      detector,
		ctx:               ctx,
		cancel:            cancel,
		logger:            logger,
		modeCheckInterval: config.ModeCheckInterval,
		lastModeCheck:     time.Now(),
	}

	// Initialize with hybrid mode
	amst.currentMode.Store(upgrade.ModeHybrid)

	// Create metrics collector
	amst.metrics = transport.NewMetricsCollector("amst-v3", config.RemoteAddr)

	// Initialize datacenter transport (v1 RDMA)
	if config.EnableDatacenter {
		datacenterConfig := &transport.TransportConfig{
			EnableRDMA:          true,
			RDMADevice:          config.RDMADevice,
			RDMAPort:            config.RDMAPort,
			MinStreams:          config.DatacenterStreams,
			MaxStreams:          config.MaxStreams,
			ChunkSizeKB:         config.ChunkSizeKB,
			AutoTune:            config.AutoTune,
			RemoteAddr:          config.RemoteAddr,
			ConnectTimeout:      config.ConnectTimeout,
			CongestionAlgorithm: "cubic", // RDMA typically uses CUBIC
		}

		rdmaTransport, err := transport.NewRDMATransport(datacenterConfig, logger)
		if err != nil {
			logger.Warn("Failed to create RDMA transport, datacenter mode disabled",
				zap.Error(err))
		} else {
			amst.datacenterTransport = rdmaTransport
			logger.Info("Datacenter transport (RDMA) initialized")
		}
	}

	// Initialize internet transport (v3 TCP)
	if config.EnableInternet {
		internetConfig := &TCPTransportV3Config{
			MinStreams:          config.MinStreams,
			MaxStreams:          config.InternetStreams,
			ChunkSizeKB:         config.ChunkSizeKB,
			AutoTune:            config.AutoTune,
			PacingEnabled:       config.PacingEnabled,
			PacingRate:          config.PacingRate,
			ConnectTimeout:      config.ConnectTimeout,
			CongestionAlgorithm: config.CongestionAlgorithm,
			RemoteAddr:          config.RemoteAddr,
		}

		tcpTransport, err := NewTCPTransportV3(internetConfig, logger)
		if err != nil {
			logger.Error("Failed to create internet transport", zap.Error(err))
			return nil, fmt.Errorf("failed to create internet transport: %w", err)
		}
		amst.internetTransport = tcpTransport
		logger.Info("Internet transport (TCP v3) initialized")
	}

	// Create congestion controller
	if config.CongestionAlgorithm != "" {
		amst.congestionCtrl = NewCongestionController(
			config.CongestionAlgorithm,
			config.PacingRate,
			logger,
		)
	}

	return amst, nil
}

// Start initializes the transport layer
func (a *AMSTv3) Start(ctx context.Context, remoteAddr string) error {
	a.mu.Lock()
	defer a.mu.Unlock()

	if a.started {
		return fmt.Errorf("AMST v3 already started")
	}

	a.logger.Info("Starting AMST v3 hybrid transport",
		zap.String("remote_addr", remoteAddr),
		zap.Bool("auto_mode", a.config.AutoMode))

	// Detect initial mode
	initialMode := a.modeDetector.DetectMode(ctx)
	a.currentMode.Store(initialMode)

	a.logger.Info("Initial network mode detected",
		zap.String("mode", initialMode.String()))

	// Start appropriate transport based on mode
	switch initialMode {
	case upgrade.ModeDatacenter:
		if a.datacenterTransport != nil {
			if err := a.datacenterTransport.Start(); err != nil {
				a.logger.Warn("Failed to start datacenter transport, falling back to internet",
					zap.Error(err))
				a.currentMode.Store(upgrade.ModeInternet)
				initialMode = upgrade.ModeInternet
			} else {
				a.logger.Info("Datacenter transport (RDMA) started")
			}
		}
	case upgrade.ModeInternet:
		if a.internetTransport != nil {
			if err := a.internetTransport.Start(ctx); err != nil {
				return fmt.Errorf("failed to start internet transport: %w", err)
			}
			a.logger.Info("Internet transport (TCP v3) started")
		}
	case upgrade.ModeHybrid:
		// Start both transports for hybrid mode
		if a.datacenterTransport != nil {
			_ = a.datacenterTransport.Start()
		}
		if a.internetTransport != nil {
			_ = a.internetTransport.Start(ctx)
		}
		a.logger.Info("Hybrid mode: both transports started")
	}

	a.started = true
	a.metrics.RecordHealthStatus(true)

	// Start mode monitoring if auto mode is enabled
	if a.config.AutoMode {
		go a.modeMonitorLoop()
	}

	return nil
}

// SendData sends data using the appropriate transport based on detected mode
func (a *AMSTv3) SendData(ctx context.Context, data []byte) error {
	a.mu.RLock()
	defer a.mu.RUnlock()

	if !a.started {
		return fmt.Errorf("AMST v3 not started")
	}

	if len(data) == 0 {
		return fmt.Errorf("no data to send")
	}

	// Check if we need to re-detect mode
	if a.config.AutoMode && time.Since(a.lastModeCheck) > a.modeCheckInterval {
		a.checkAndSwitchMode(ctx)
	}

	mode := a.currentMode.Load().(upgrade.NetworkMode)

	startTime := time.Now()
	var err error

	switch mode {
	case upgrade.ModeDatacenter:
		err = a.sendViaDatacenter(data)
	case upgrade.ModeInternet:
		err = a.sendViaInternet(ctx, data)
	case upgrade.ModeHybrid:
		err = a.adaptiveSend(ctx, data)
	default:
		return fmt.Errorf("unknown network mode: %v", mode)
	}

	if err != nil {
		a.metrics.RecordError("send_failed")
		return err
	}

	// Record metrics
	a.totalBytesSent.Add(uint64(len(data)))
	a.metrics.RecordBytesSent(uint64(len(data)))
	a.metrics.RecordLatency("send", time.Since(startTime))

	return nil
}

// sendViaDatacenter sends via v1 RDMA transport
func (a *AMSTv3) sendViaDatacenter(data []byte) error {
	if a.datacenterTransport == nil {
		return fmt.Errorf("datacenter transport not available")
	}

	if !a.datacenterTransport.IsStarted() {
		return fmt.Errorf("datacenter transport not started")
	}

	return a.datacenterTransport.Send(data)
}

// sendViaInternet sends via v3 internet-optimized TCP
func (a *AMSTv3) sendViaInternet(ctx context.Context, data []byte) error {
	if a.internetTransport == nil {
		return fmt.Errorf("internet transport not available")
	}

	return a.internetTransport.Send(ctx, data)
}

// adaptiveSend intelligently selects transport based on data size and conditions
func (a *AMSTv3) adaptiveSend(ctx context.Context, data []byte) error {
	dataSize := len(data)

	// Decision logic:
	// - Small data (<1MB): prefer datacenter (low latency)
	// - Large data (>10MB): prefer internet (better for WAN)
	// - Medium data: use congestion controller decision

	if dataSize < 1024*1024 && a.datacenterTransport != nil {
		// Small data: use datacenter for low latency
		if err := a.sendViaDatacenter(data); err == nil {
			return nil
		}
		// Fallback to internet
		a.logger.Debug("Datacenter send failed, falling back to internet")
	}

	// Default to internet transport
	return a.sendViaInternet(ctx, data)
}

// TransferWithProgress sends data with progress callback (v1 compatibility)
func (a *AMSTv3) TransferWithProgress(ctx context.Context, data []byte, progressCallback func(int64, int64)) error {
	if len(data) == 0 {
		return fmt.Errorf("no data to transfer")
	}

	totalSize := int64(len(data))
	chunkSize := a.config.ChunkSizeKB * 1024
	transferred := int64(0)

	// Send in chunks with progress updates
	for offset := 0; offset < len(data); offset += chunkSize {
		end := offset + chunkSize
		if end > len(data) {
			end = len(data)
		}

		chunk := data[offset:end]
		if err := a.SendData(ctx, chunk); err != nil {
			return fmt.Errorf("chunk send failed at offset %d: %w", offset, err)
		}

		transferred += int64(len(chunk))
		if progressCallback != nil {
			progressCallback(transferred, totalSize)
		}
	}

	return nil
}

// AdjustStreams dynamically adjusts stream count based on network conditions
func (a *AMSTv3) AdjustStreams(bandwidthMbps, latencyMs float64) error {
	a.mu.RLock()
	defer a.mu.RUnlock()

	if !a.started {
		return fmt.Errorf("AMST v3 not started")
	}

	mode := a.currentMode.Load().(upgrade.NetworkMode)

	switch mode {
	case upgrade.ModeDatacenter:
		if a.datacenterTransport != nil {
			return a.datacenterTransport.AdjustStreams(bandwidthMbps, latencyMs)
		}
	case upgrade.ModeInternet:
		if a.internetTransport != nil {
			return a.internetTransport.AdjustStreams(bandwidthMbps, latencyMs)
		}
	case upgrade.ModeHybrid:
		// Adjust both transports
		if a.datacenterTransport != nil {
			_ = a.datacenterTransport.AdjustStreams(bandwidthMbps, latencyMs)
		}
		if a.internetTransport != nil {
			_ = a.internetTransport.AdjustStreams(bandwidthMbps, latencyMs)
		}
	}

	return nil
}

// modeMonitorLoop continuously monitors and adjusts network mode
func (a *AMSTv3) modeMonitorLoop() {
	ticker := time.NewTicker(a.modeCheckInterval)
	defer ticker.Stop()

	for {
		select {
		case <-a.ctx.Done():
			return
		case <-ticker.C:
			a.checkAndSwitchMode(context.Background())
		}
	}
}

// checkAndSwitchMode detects current mode and switches if necessary
func (a *AMSTv3) checkAndSwitchMode(ctx context.Context) {
	newMode := a.modeDetector.DetectMode(ctx)
	currentMode := a.currentMode.Load().(upgrade.NetworkMode)

	if newMode != currentMode {
		a.logger.Info("Network mode changed",
			zap.String("old_mode", currentMode.String()),
			zap.String("new_mode", newMode.String()))

		a.switchMode(ctx, newMode)
		a.modeTransitions.Add(1)
	}

	a.lastModeCheck = time.Now()
}

// switchMode switches to a different network mode
func (a *AMSTv3) switchMode(ctx context.Context, newMode upgrade.NetworkMode) {
	a.mu.Lock()
	defer a.mu.Unlock()

	oldMode := a.currentMode.Load().(upgrade.NetworkMode)

	// Stop old transport if exclusive mode
	switch oldMode {
	case upgrade.ModeDatacenter:
		// Keep RDMA running for potential quick switch back
	case upgrade.ModeInternet:
		// Keep TCP running
	}

	// Start new transport
	switch newMode {
	case upgrade.ModeDatacenter:
		if a.datacenterTransport != nil && !a.datacenterTransport.IsStarted() {
			_ = a.datacenterTransport.Start()
		}
	case upgrade.ModeInternet:
		if a.internetTransport != nil {
			_ = a.internetTransport.Start(ctx)
		}
	case upgrade.ModeHybrid:
		// Ensure both are running
		if a.datacenterTransport != nil && !a.datacenterTransport.IsStarted() {
			_ = a.datacenterTransport.Start()
		}
		if a.internetTransport != nil {
			_ = a.internetTransport.Start(ctx)
		}
	}

	a.currentMode.Store(newMode)
}

// GetCurrentMode returns the current network mode
func (a *AMSTv3) GetCurrentMode() upgrade.NetworkMode {
	return a.currentMode.Load().(upgrade.NetworkMode)
}

// GetMetrics returns comprehensive metrics
func (a *AMSTv3) GetMetrics() transport.TransportMetrics {
	a.mu.RLock()
	defer a.mu.RUnlock()

	baseMetrics := a.metrics.GetMetrics()
	mode := a.currentMode.Load().(upgrade.NetworkMode)

	// Augment with transport-specific metrics
	switch mode {
	case upgrade.ModeDatacenter:
		if a.datacenterTransport != nil {
			dcMetrics := a.datacenterTransport.GetMetrics()
			baseMetrics.ActiveStreams = dcMetrics.ActiveStreams
			baseMetrics.TotalStreams = dcMetrics.TotalStreams
			baseMetrics.TransportType = "rdma"
		}
	case upgrade.ModeInternet:
		if a.internetTransport != nil {
			intMetrics := a.internetTransport.GetMetrics()
			baseMetrics.ActiveStreams = int32(intMetrics.ActiveStreams)
			baseMetrics.TotalStreams = intMetrics.TotalStreams
			baseMetrics.TransportType = "tcp-v3"
		}
	case upgrade.ModeHybrid:
		// Combine metrics from both
		var totalActiveStreams int32
		var totalStreams int
		if a.datacenterTransport != nil {
			dcMetrics := a.datacenterTransport.GetMetrics()
			totalActiveStreams += dcMetrics.ActiveStreams
			totalStreams += dcMetrics.TotalStreams
		}
		if a.internetTransport != nil {
			intMetrics := a.internetTransport.GetMetrics()
			totalActiveStreams += int32(intMetrics.ActiveStreams)
			totalStreams += intMetrics.TotalStreams
		}
		baseMetrics.ActiveStreams = totalActiveStreams
		baseMetrics.TotalStreams = totalStreams
		baseMetrics.TransportType = "hybrid"
	}

	baseMetrics.Mode = mode.String()
	baseMetrics.TotalBytesSent = a.totalBytesSent.Load()
	baseMetrics.TotalBytesRecv = a.totalBytesRecv.Load()
	baseMetrics.CongestionControl = a.config.CongestionAlgorithm

	return baseMetrics
}

// Close gracefully shuts down all transports
func (a *AMSTv3) Close() error {
	a.mu.Lock()
	defer a.mu.Unlock()

	if !a.started {
		return nil
	}

	a.logger.Info("Closing AMST v3")

	a.cancel()

	// Close both transports
	if a.datacenterTransport != nil {
		if err := a.datacenterTransport.Close(); err != nil {
			a.logger.Error("Failed to close datacenter transport", zap.Error(err))
		}
	}

	if a.internetTransport != nil {
		if err := a.internetTransport.Close(); err != nil {
			a.logger.Error("Failed to close internet transport", zap.Error(err))
		}
	}

	a.started = false
	a.metrics.RecordHealthStatus(false)

	a.logger.Info("AMST v3 closed",
		zap.Uint64("total_bytes_sent", a.totalBytesSent.Load()),
		zap.Uint64("mode_transitions", a.modeTransitions.Load()))

	return nil
}
