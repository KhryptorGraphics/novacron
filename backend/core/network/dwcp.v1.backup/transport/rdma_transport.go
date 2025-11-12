package transport

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/transport/rdma"
	"go.uber.org/zap"
)

// RDMATransport provides RDMA-accelerated transport with TCP fallback
type RDMATransport struct {
	config  *TransportConfig
	logger  *zap.Logger
	metrics *MetricsCollector

	// RDMA state
	rdmaAvailable bool
	rdmaDevice    string
	rdmaPort      int
	rdmaManager   *rdma.RDMAManager

	// Fallback TCP transport
	tcpTransport *MultiStreamTCP

	// Lifecycle
	ctx     context.Context
	cancel  context.CancelFunc
	mu      sync.RWMutex
	started bool
}

// NewRDMATransport creates a new RDMA transport with TCP fallback
func NewRDMATransport(config *TransportConfig, logger *zap.Logger) (*RDMATransport, error) {
	if config == nil {
		tc := DefaultTransportConfig()
		config = tc
	}

	if logger == nil {
		logger, _ = zap.NewProduction()
	}

	ctx, cancel := context.WithCancel(context.Background())

	rt := &RDMATransport{
		config:        config,
		logger:        logger,
		rdmaDevice:    config.RDMADevice,
		rdmaPort:      config.RDMAPort,
		ctx:           ctx,
		cancel:        cancel,
		rdmaAvailable: false,
	}

	// Check RDMA availability
	if config.EnableRDMA {
		rt.rdmaAvailable = rt.checkRDMAAvailability()
	}

	// Determine transport type
	transportType := "tcp"
	if rt.rdmaAvailable {
		transportType = "rdma"
	}

	rt.metrics = NewMetricsCollector(transportType, config.RemoteAddr)

	// Always create TCP transport as fallback
	amstConfig := &AMSTConfig{
		MinStreams:     config.MinStreams,
		MaxStreams:     config.MaxStreams,
		ChunkSizeKB:    config.ChunkSizeKB,
		AutoTune:       config.AutoTune,
		PacingEnabled:  config.PacingEnabled,
		PacingRate:     config.PacingRate,
		ConnectTimeout: config.ConnectTimeout,
	}

	tcpTransport, err := NewMultiStreamTCP(config.RemoteAddr, amstConfig, logger)
	if err != nil {
		cancel()
		return nil, fmt.Errorf("failed to create TCP fallback: %w", err)
	}
	rt.tcpTransport = tcpTransport

	return rt, nil
}

// checkRDMAAvailability checks if RDMA is available on the system
func (rt *RDMATransport) checkRDMAAvailability() bool {
	rt.logger.Info("Checking RDMA availability",
		zap.String("device", rt.rdmaDevice),
		zap.Int("port", rt.rdmaPort))

	// Check if RDMA is available using libibverbs
	if !rdma.CheckAvailability() {
		rt.logger.Info("RDMA not available on system, falling back to TCP")
		return false
	}

	// Enumerate devices
	devices, err := rdma.GetDeviceList()
	if err != nil {
		rt.logger.Warn("Failed to enumerate RDMA devices", zap.Error(err))
		return false
	}

	if len(devices) == 0 {
		rt.logger.Info("No RDMA devices found, falling back to TCP")
		return false
	}

	// Log available devices
	rt.logger.Info("RDMA devices found", zap.Int("count", len(devices)))
	for i, dev := range devices {
		rt.logger.Debug("RDMA device",
			zap.Int("index", i),
			zap.String("name", dev.Name),
			zap.String("guid", dev.GUID),
			zap.Bool("supports_rc", dev.SupportsRC),
			zap.Bool("supports_rdma_write", dev.SupportsRDMAWrite))
	}

	// Check if specific device exists (if specified)
	if rt.rdmaDevice != "" {
		found := false
		for _, dev := range devices {
			if dev.Name == rt.rdmaDevice {
				found = true
				rt.logger.Info("Found requested RDMA device",
					zap.String("device", rt.rdmaDevice),
					zap.String("guid", dev.GUID))
				break
			}
		}
		if !found {
			rt.logger.Warn("Requested RDMA device not found, falling back to TCP",
				zap.String("device", rt.rdmaDevice))
			return false
		}
	}

	rt.logger.Info("RDMA available and ready")
	return true
}

// Start initializes the RDMA transport
func (rt *RDMATransport) Start() error {
	rt.mu.Lock()
	defer rt.mu.Unlock()

	if rt.started {
		return fmt.Errorf("RDMA transport already started")
	}

	if rt.rdmaAvailable {
		rt.logger.Info("Starting RDMA transport",
			zap.String("device", rt.rdmaDevice),
			zap.Int("port", rt.rdmaPort))

		// Create RDMA configuration
		rdmaConfig := &rdma.Config{
			DeviceName:      rt.rdmaDevice,
			Port:            rt.rdmaPort,
			GIDIndex:        0,
			MTU:             4096,
			MaxInlineData:   256,
			MaxSendWR:       1024,
			MaxRecvWR:       1024,
			MaxSGE:          16,
			QPType:          "RC",
			UseSRQ:          false,
			UseEventChannel: false, // Use polling for lowest latency
			SendBufferSize:  4 * 1024 * 1024,
			RecvBufferSize:  4 * 1024 * 1024,
		}

		// Initialize RDMA manager
		mgr, err := rdma.NewRDMAManager(rdmaConfig, rt.logger)
		if err != nil {
			rt.logger.Error("Failed to initialize RDMA manager, falling back to TCP",
				zap.Error(err))
			rt.rdmaAvailable = false
		} else {
			rt.rdmaManager = mgr
			rt.logger.Info("RDMA manager initialized successfully")

			// Note: Connection establishment (QP exchange) would happen
			// during the actual connection phase with peer
		}
	}

	// Start TCP transport (either as fallback or primary)
	if err := rt.tcpTransport.Start(); err != nil {
		return fmt.Errorf("failed to start TCP transport: %w", err)
	}

	rt.started = true
	rt.metrics.RecordHealthStatus(true)

	rt.logger.Info("RDMA transport started (using TCP)",
		zap.Bool("rdma_available", rt.rdmaAvailable))

	return nil
}

// Send data via RDMA or TCP
func (rt *RDMATransport) Send(data []byte) error {
	rt.mu.RLock()
	defer rt.mu.RUnlock()

	if !rt.started {
		return fmt.Errorf("RDMA transport not started")
	}

	startTime := time.Now()

	var err error
	if rt.rdmaAvailable && rt.rdmaManager != nil && rt.rdmaManager.IsConnected() {
		// Send via RDMA
		err = rt.rdmaManager.Send(data)
		if err != nil {
			rt.logger.Warn("RDMA send failed, falling back to TCP", zap.Error(err))
			rt.metrics.RecordError("rdma_send_failed")
			// Fall through to TCP
		}
	}

	// Use TCP transport if RDMA failed or not available
	if err != nil || !rt.rdmaAvailable || rt.rdmaManager == nil {
		err = rt.tcpTransport.Send(data)
	}

	// Record metrics
	if err != nil {
		rt.metrics.RecordError("send_failed")
		return err
	}

	rt.metrics.RecordBytesSent(uint64(len(data)))
	rt.metrics.RecordLatency("send", time.Since(startTime))

	return nil
}

// Receive data via RDMA or TCP
func (rt *RDMATransport) Receive(expectedSize int) ([]byte, error) {
	rt.mu.RLock()
	defer rt.mu.RUnlock()

	if !rt.started {
		return nil, fmt.Errorf("RDMA transport not started")
	}

	startTime := time.Now()

	var data []byte
	var err error

	if rt.rdmaAvailable && rt.rdmaManager != nil && rt.rdmaManager.IsConnected() {
		// Receive via RDMA
		data = make([]byte, expectedSize)
		var n int
		n, err = rt.rdmaManager.Receive(data)
		if err != nil {
			rt.logger.Warn("RDMA receive failed, falling back to TCP", zap.Error(err))
			rt.metrics.RecordError("rdma_recv_failed")
			data = nil
			// Fall through to TCP
		} else {
			data = data[:n]
		}
	}

	// Use TCP transport if RDMA failed or not available
	if err != nil || !rt.rdmaAvailable || rt.rdmaManager == nil {
		data, err = rt.tcpTransport.Receive(expectedSize)
	}

	// Record metrics
	if err != nil {
		rt.metrics.RecordError("receive_failed")
		return nil, err
	}

	rt.metrics.RecordBytesReceived(uint64(len(data)))
	rt.metrics.RecordLatency("receive", time.Since(startTime))

	return data, nil
}

// AdjustStreams dynamically adjusts stream count
func (rt *RDMATransport) AdjustStreams(bandwidthMbps, latencyMs float64) error {
	rt.mu.RLock()
	defer rt.mu.RUnlock()

	if !rt.started {
		return fmt.Errorf("RDMA transport not started")
	}

	// For RDMA, we would adjust QP count
	// For TCP fallback, use the existing implementation
	return rt.tcpTransport.AdjustStreams(bandwidthMbps, latencyMs)
}

// GetMetrics returns current transport metrics
func (rt *RDMATransport) GetMetrics() TransportMetrics {
	rt.mu.RLock()
	defer rt.mu.RUnlock()

	// Get base metrics
	metrics := rt.metrics.GetMetrics()

	// Augment with TCP transport metrics if using fallback
	if rt.tcpTransport != nil && rt.tcpTransport.IsStarted() {
		tcpMetrics := rt.tcpTransport.GetMetrics()

		metrics.ActiveStreams = tcpMetrics["active_streams"].(int32)
		metrics.TotalStreams = tcpMetrics["total_streams"].(int)

		if sent, ok := tcpMetrics["total_bytes_sent"].(uint64); ok {
			metrics.TotalBytesSent = sent
		}
		if recv, ok := tcpMetrics["total_bytes_recv"].(uint64); ok {
			metrics.TotalBytesRecv = recv
		}
	}

	// Set transport type
	if rt.rdmaAvailable {
		metrics.TransportType = "rdma"
	} else {
		metrics.TransportType = "tcp"
	}

	metrics.CongestionControl = rt.config.CongestionAlgorithm

	return metrics
}

// HealthCheck verifies transport health
func (rt *RDMATransport) HealthCheck() error {
	rt.mu.RLock()
	defer rt.mu.RUnlock()

	if !rt.started {
		return fmt.Errorf("RDMA transport not started")
	}

	startTime := time.Now()
	defer func() {
		rt.metrics.RecordHealthCheckDuration(time.Since(startTime))
	}()

	// Check RDMA health if available
	if rt.rdmaAvailable {
		// TODO: Check RDMA QP state, CQ depth, etc.
	}

	// Always check TCP transport health
	if rt.tcpTransport != nil && rt.tcpTransport.IsStarted() {
		tcpMetrics := rt.tcpTransport.GetMetrics()
		activeStreams := tcpMetrics["active_streams"].(int32)

		if activeStreams == 0 {
			rt.metrics.RecordHealthStatus(false)
			return fmt.Errorf("no active streams")
		}
	}

	rt.metrics.RecordHealthStatus(true)
	return nil
}

// IsStarted returns whether the transport is started
func (rt *RDMATransport) IsStarted() bool {
	rt.mu.RLock()
	defer rt.mu.RUnlock()
	return rt.started
}

// Close gracefully shuts down the transport
func (rt *RDMATransport) Close() error {
	rt.mu.Lock()
	defer rt.mu.Unlock()

	if !rt.started {
		return nil
	}

	rt.logger.Info("Closing RDMA transport")

	rt.cancel()

	// Close RDMA resources if available
	if rt.rdmaAvailable && rt.rdmaManager != nil {
		if err := rt.rdmaManager.Close(); err != nil {
			rt.logger.Error("Failed to close RDMA manager", zap.Error(err))
		}
		rt.rdmaManager = nil
	}

	// Close TCP transport
	if rt.tcpTransport != nil {
		if err := rt.tcpTransport.Close(); err != nil {
			rt.logger.Error("Failed to close TCP transport", zap.Error(err))
		}
	}

	rt.started = false
	rt.metrics.RecordHealthStatus(false)

	rt.logger.Info("RDMA transport closed")
	return nil
}

// SupportsRDMA returns whether RDMA is available
func (rt *RDMATransport) SupportsRDMA() bool {
	return rt.rdmaAvailable
}

// GetRDMADeviceInfo returns RDMA device information
func (rt *RDMATransport) GetRDMADeviceInfo() map[string]interface{} {
	rt.mu.RLock()
	defer rt.mu.RUnlock()

	info := map[string]interface{}{
		"rdma_available": rt.rdmaAvailable,
		"rdma_device":    rt.rdmaDevice,
		"rdma_port":      rt.rdmaPort,
		"using_fallback": !rt.rdmaAvailable,
	}

	// Add RDMA statistics if available
	if rt.rdmaAvailable && rt.rdmaManager != nil {
		rdmaStats := rt.rdmaManager.GetStats()
		info["rdma_stats"] = rdmaStats
		info["rdma_connected"] = rt.rdmaManager.IsConnected()

		// Add latency information
		if avgLatency, ok := rdmaStats["avg_send_latency_us"].(float64); ok {
			info["rdma_avg_latency_us"] = avgLatency
		}
	}

	return info
}
