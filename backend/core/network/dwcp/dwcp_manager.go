package dwcp

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/resilience"
	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/transport"
	"go.uber.org/zap"
)

// Manager is the main coordinator for all DWCP components
type Manager struct {
	config *Config
	logger *zap.Logger

	// Component interfaces
	transport   transport.Transport // AMST or RDMA transport (Phase 1)
	compression interface{}         // CompressionLayer interface (Phase 0-1)
	prediction  interface{}         // PredictionEngine interface (Phase 2)
	sync        interface{}         // SyncLayer interface (Phase 3)
	consensus   interface{}         // ConsensusLayer interface (Phase 3)

	// Resilience layer (Phase 2 Production Hardening)
	resilience *resilience.ResilienceManager

	// Metrics collection
	metrics      *DWCPMetrics
	metricsMutex sync.RWMutex

	// Lifecycle management
	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup

	// State
	enabled bool
	started bool
	mu      sync.RWMutex
}

// NewManager creates a new DWCP manager with the given configuration
func NewManager(config *Config, logger *zap.Logger) (*Manager, error) {
	if config == nil {
		config = DefaultConfig()
	}

	// Validate configuration
	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf("invalid configuration: %w", err)
	}

	if logger == nil {
		logger, _ = zap.NewProduction()
	}

	ctx, cancel := context.WithCancel(context.Background())

	m := &Manager{
		config:  config,
		logger:  logger,
		ctx:     ctx,
		cancel:  cancel,
		enabled: config.Enabled,
		started: false,
		metrics: &DWCPMetrics{
			Version: DWCPVersion,
			Enabled: config.Enabled,
		},
	}

	return m, nil
}

// Start initializes and starts all DWCP components
func (m *Manager) Start() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.started {
		return fmt.Errorf("DWCP manager already started")
	}

	if !m.enabled {
		m.logger.Info("DWCP is disabled, skipping initialization")
		return nil
	}

	m.logger.Info("Starting DWCP manager",
		zap.String("version", DWCPVersion),
		zap.Bool("enabled", m.enabled))

	// Initialize resilience layer (Phase 2 Production Hardening)
	if err := m.initializeResilience(); err != nil {
		return fmt.Errorf("failed to initialize resilience: %w", err)
	}

	// Initialize transport layer (Phase 1)
	if err := m.initializeTransport(); err != nil {
		return fmt.Errorf("failed to initialize transport: %w", err)
	}

	// TODO: Initialize compression layer (Phase 0-1)
	// m.compression = compression.New(...)

	// TODO: Initialize prediction engine (Phase 2)
	// if m.config.Prediction.Enabled {
	//     m.prediction = prediction.New(...)
	// }

	// TODO: Initialize sync layer (Phase 3)
	// if m.config.Sync.Enabled {
	//     m.sync = sync.New(...)
	// }

	// TODO: Initialize consensus layer (Phase 3)
	// if m.config.Consensus.Enabled {
	//     m.consensus = consensus.New(...)
	// }

	// Start metrics collection
	m.wg.Add(1)
	go m.metricsCollectionLoop()

	m.started = true
	m.logger.Info("DWCP manager started successfully")

	return nil
}

// Stop gracefully shuts down all DWCP components
func (m *Manager) Stop() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if !m.started {
		return nil
	}

	m.logger.Info("Stopping DWCP manager")

	// Cancel context to signal all goroutines
	m.cancel()

	// Wait for all goroutines to finish
	m.wg.Wait()

	// Shutdown components in reverse order
	// TODO: if m.consensus != nil { m.consensus.Stop() }
	// TODO: if m.sync != nil { m.sync.Stop() }
	// TODO: if m.prediction != nil { m.prediction.Stop() }
	// TODO: if m.compression != nil { m.compression.Stop() }

	// Shutdown transport layer
	if m.transport != nil {
		if err := m.transport.Close(); err != nil {
			m.logger.Error("Failed to close transport", zap.Error(err))
		}
	}

	// Shutdown resilience layer
	if m.resilience != nil {
		m.resilience.StopHealthMonitoring()
		m.logger.Info("Resilience layer stopped")
	}

	m.started = false
	m.logger.Info("DWCP manager stopped")

	return nil
}

// GetMetrics returns the current DWCP metrics
func (m *Manager) GetMetrics() *DWCPMetrics {
	m.metricsMutex.RLock()
	defer m.metricsMutex.RUnlock()

	// Return a copy to avoid race conditions
	metricsCopy := *m.metrics
	return &metricsCopy
}

// IsEnabled returns whether DWCP is enabled
func (m *Manager) IsEnabled() bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.enabled
}

// IsStarted returns whether DWCP manager has been started
func (m *Manager) IsStarted() bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.started
}

// GetConfig returns the current configuration
func (m *Manager) GetConfig() *Config {
	m.mu.RLock()
	defer m.mu.RUnlock()

	// Return a copy to prevent external modification
	configCopy := *m.config
	return &configCopy
}

// UpdateConfig updates the DWCP configuration (requires restart to take effect)
func (m *Manager) UpdateConfig(newConfig *Config) error {
	if err := newConfig.Validate(); err != nil {
		return fmt.Errorf("invalid configuration: %w", err)
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	if m.started {
		return fmt.Errorf("cannot update configuration while DWCP is running, call Stop() first")
	}

	m.config = newConfig
	m.enabled = newConfig.Enabled

	m.logger.Info("DWCP configuration updated",
		zap.Bool("enabled", newConfig.Enabled))

	return nil
}

// metricsCollectionLoop periodically updates DWCP metrics
func (m *Manager) metricsCollectionLoop() {
	defer m.wg.Done()

	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-m.ctx.Done():
			return
		case <-ticker.C:
			m.collectMetrics()
		}
	}
}

// collectMetrics gathers metrics from all DWCP components
func (m *Manager) collectMetrics() {
	m.metricsMutex.Lock()
	defer m.metricsMutex.Unlock()

	// Update basic status
	m.metrics.Enabled = m.enabled
	m.metrics.Version = DWCPVersion

	// TODO: Collect transport metrics (Phase 0-1)
	// if m.transport != nil {
	//     m.metrics.Transport = m.transport.GetMetrics()
	// }

	// TODO: Collect compression metrics (Phase 0-1)
	// if m.compression != nil {
	//     m.metrics.Compression = m.compression.GetMetrics()
	// }

	// TODO: Determine network tier (Phase 1)
	// m.metrics.Tier = m.detectNetworkTier()

	// TODO: Determine transport mode (Phase 1)
	// m.metrics.Mode = m.getTransportMode()
}

// initializeTransport initializes the AMST transport layer
func (m *Manager) initializeTransport() error {
	// TODO: Read transport config from m.config
	// For now, use defaults with environment-based overrides

	transportConfig := &transport.TransportConfig{
		RemoteAddr:          "", // Will be set per-connection
		ConnectTimeout:      30 * time.Second,
		MinStreams:          16,
		MaxStreams:          256,
		ChunkSizeKB:         256,
		AutoTune:            true,
		PacingEnabled:       true,
		PacingRate:          1000 * 1024 * 1024, // 1 Gbps
		EnableRDMA:          false,              // Disabled by default
		RDMADevice:          "mlx5_0",
		RDMAPort:            1,
		CongestionAlgorithm: "bbr",
		EnableRetries:       true,
		MaxRetries:          3,
		RetryBackoffMs:      100,
		HealthCheckInterval: 10 * time.Second,
	}

	// Create RDMA transport (with TCP fallback)
	rdmaTransport, err := transport.NewRDMATransport(transportConfig, m.logger)
	if err != nil {
		return fmt.Errorf("failed to create RDMA transport: %w", err)
	}

	m.transport = rdmaTransport

	m.logger.Info("Transport layer initialized",
		zap.Bool("rdma_enabled", transportConfig.EnableRDMA),
		zap.String("congestion_algorithm", transportConfig.CongestionAlgorithm))

	return nil
}

// GetTransport returns the transport layer for use by other components
func (m *Manager) GetTransport() transport.Transport {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.transport
}

// detectNetworkTier determines the current network tier based on metrics
func (m *Manager) detectNetworkTier() NetworkTier {
	if m.transport == nil {
		return NetworkTierTier2
	}

	metrics := m.transport.GetMetrics()
	latency := metrics.AverageLatencyMs

	// Determine tier based on latency
	if latency < 5 {
		return NetworkTierTier1 // Datacenter
	} else if latency < 50 {
		return NetworkTierTier2 // Metro
	} else if latency < 150 {
		return NetworkTierTier3 // Regional
	}
	return NetworkTierTier4 // Global
}

// getTransportMode returns the current transport mode
func (m *Manager) getTransportMode() TransportMode {
	if m.transport == nil {
		return TransportModeTCP
	}

	metrics := m.transport.GetMetrics()
	switch metrics.TransportType {
	case "rdma":
		return TransportModeRDMA
	case "tcp":
		return TransportModeTCP
	default:
		return TransportModeTCP
	}
}

// Health check interface for monitoring systems
func (m *Manager) HealthCheck() error {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if !m.enabled {
		return nil // Healthy if disabled
	}

	if !m.started {
		return fmt.Errorf("DWCP manager not started")
	}

	// Check transport health
	if m.transport != nil {
		if err := m.transport.HealthCheck(); err != nil {
			return fmt.Errorf("transport layer unhealthy: %w", err)
		}
	}

	// TODO: Check compression health (Phase 1)
	// if m.compression != nil && !m.compression.IsHealthy() {
	//     return fmt.Errorf("compression layer unhealthy")
	// }

	return nil
}
