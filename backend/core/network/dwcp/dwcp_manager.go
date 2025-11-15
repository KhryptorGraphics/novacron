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

	// Component interfaces (properly typed for type safety)
	transport   transport.Transport // AMST or RDMA transport (Phase 1)
	compression CompressionLayer    // HDE compression (Phase 0-1)
	prediction  PredictionEngine    // ML predictions (Phase 2)
	sync        SyncLayer           // State sync (Phase 3)
	consensus   ConsensusLayer      // Consensus (Phase 3)

	// Intelligent task partitioning (ITP)
	partitioner *TaskPartitioner

	// Resilience layer (Phase 2 Production Hardening)
	resilience     *resilience.ResilienceManager
	circuitBreaker *CircuitBreaker

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
		config:         config,
		logger:         logger,
		ctx:            ctx,
		cancel:         cancel,
		enabled:        config.Enabled,
		started:        false,
		circuitBreaker: NewCircuitBreaker(5, 30*time.Second), // 5 failures, 30s timeout
		metrics: &DWCPMetrics{
			Version: DWCPVersion,
			Enabled: config.Enabled,
		},
	}

	return m, nil
}

// Start initializes and starts all DWCP components in proper dependency order
// Implements Lifecycle interface with context-aware initialization
func (m *Manager) Start() error {
	// Create context for lifecycle management
	ctx := context.Background()
	return m.StartWithContext(ctx)
}

// StartWithContext initializes and starts all DWCP components with context
// Phase 0: Core Infrastructure → Phase 1: Intelligence → Phase 2: Coordination → Phase 3: Resilience
func (m *Manager) StartWithContext(ctx context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.started {
		return &DWCPError{Code: "ALREADY_STARTED", Message: "DWCP manager already started"}
	}

	if !m.enabled {
		m.logger.Info("DWCP is disabled, skipping initialization")
		return nil
	}

	m.logger.Info("Starting DWCP manager with lifecycle coordination",
		zap.String("version", DWCPVersion),
		zap.Bool("enabled", m.enabled))

	// Set context for lifecycle management
	m.ctx, m.cancel = context.WithCancel(ctx)

	// Phase 0: Core Infrastructure (dependencies first)
	if err := m.startPhase0Components(m.ctx); err != nil {
		m.cleanup()
		return &DWCPError{Code: "PHASE0_START_FAILED", Message: "Failed to start Phase 0 components", Cause: err}
	}

	// Phase 1: Intelligence Layer
	if err := m.startPhase1Components(m.ctx); err != nil {
		m.cleanup()
		return &DWCPError{Code: "PHASE1_START_FAILED", Message: "Failed to start Phase 1 components", Cause: err}
	}

	// Phase 2: Coordination Layer
	if err := m.startPhase2Components(m.ctx); err != nil {
		m.cleanup()
		return &DWCPError{Code: "PHASE2_START_FAILED", Message: "Failed to start Phase 2 components", Cause: err}
	}

	// Phase 3: Resilience Layer
	if err := m.startPhase3Components(m.ctx); err != nil {
		m.cleanup()
		return &DWCPError{Code: "PHASE3_START_FAILED", Message: "Failed to start Phase 3 components", Cause: err}
	}

	// Start management loops
	m.wg.Add(1)
	go m.metricsCollectionLoop()

	m.wg.Add(1)
	go m.healthMonitoringLoop()

	m.started = true
	m.logger.Info("DWCP manager started successfully with all phases")

	return nil
}

// startPhase0Components starts core infrastructure components (Transport, Compression)
func (m *Manager) startPhase0Components(ctx context.Context) error {
	m.logger.Info("Starting Phase 0: Core Infrastructure")

	// 1. Transport Layer (AMST) - Foundation for all communication
	if err := m.initializeTransport(); err != nil {
		return fmt.Errorf("failed to initialize transport: %w", err)
	}
	if m.transport != nil {
		if err := m.transport.Start(); err != nil {
			return fmt.Errorf("failed to start transport: %w", err)
		}
		m.logger.Info("Transport layer started successfully")
	}

	// 2. Compression Layer (HDE) - Data compression for transport
	if m.config.Compression.Enabled && m.compression != nil {
		if err := m.compression.Start(ctx); err != nil {
			return fmt.Errorf("failed to start compression layer: %w", err)
		}
		m.logger.Info("Compression layer started successfully")
	}

	return nil
}

// startPhase1Components starts intelligence layer components (Prediction, TaskPartitioner)
func (m *Manager) startPhase1Components(ctx context.Context) error {
	m.logger.Info("Starting Phase 1: Intelligence Layer")

	// 3. Prediction Engine (PBA) - Bandwidth prediction for optimization
	if m.config.Prediction.Enabled && m.prediction != nil {
		if err := m.prediction.Start(ctx); err != nil {
			return fmt.Errorf("failed to start prediction engine: %w", err)
		}
		m.logger.Info("Prediction engine started successfully")
	}

	// 4. Task Partitioner (ITP) - Intelligent task distribution
	if m.partitioner != nil {
		if err := m.partitioner.Start(ctx); err != nil {
			return fmt.Errorf("failed to start task partitioner: %w", err)
		}
		m.logger.Info("Task partitioner started successfully")
	}

	return nil
}

// startPhase2Components starts coordination layer components (Sync, Consensus)
func (m *Manager) startPhase2Components(ctx context.Context) error {
	m.logger.Info("Starting Phase 2: Coordination Layer")

	// 5. Sync Layer (ASS) - State synchronization
	if m.config.Sync.Enabled {
		// TODO: Start sync layer when CRDT/Raft implementation is complete
		m.logger.Info("Sync layer initialization deferred to Phase 3 state sync implementation")
	}

	// 6. Consensus Layer (ACP) - Distributed consensus
	if m.config.Consensus.Enabled {
		// TODO: Start consensus layer when ProBFT/Bullshark implementation is complete
		m.logger.Info("Consensus layer initialization deferred to Phase 3 consensus implementation")
	}

	return nil
}

// startPhase3Components starts resilience layer components (CircuitBreaker, Resilience)
func (m *Manager) startPhase3Components(ctx context.Context) error {
	m.logger.Info("Starting Phase 3: Resilience Layer")

	// Initialize resilience layer if not already done
	if m.resilience == nil {
		if err := m.initializeResilience(); err != nil {
			return fmt.Errorf("failed to initialize resilience layer: %w", err)
		}
	}

	// 7. Circuit Breaker - Failure protection
	if m.circuitBreaker != nil {
		m.logger.Info("Circuit breaker initialized successfully")
	}

	// 8. Resilience Manager - Overall resilience coordination
	if m.resilience != nil {
		m.logger.Info("Resilience manager started successfully")
	}

	return nil
}

// cleanup performs cleanup when startup fails
func (m *Manager) cleanup() {
	if m.cancel != nil {
		m.cancel()
	}
	// Additional cleanup logic can be added here
}

// Stop gracefully shuts down all DWCP components in reverse dependency order
// Phase 3 → Phase 2 → Phase 1 → Phase 0 (Resilience → Coordination → Intelligence → Core)
func (m *Manager) Stop() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if !m.started {
		return nil
	}

	m.logger.Info("Stopping DWCP manager with lifecycle coordination")

	// Cancel context to signal all goroutines
	if m.cancel != nil {
		m.cancel()
	}

	// Wait for all goroutines to finish
	m.wg.Wait()

	// Shutdown in reverse order of initialization (Phase 3 → Phase 0)
	m.stopPhase3Components()
	m.stopPhase2Components()
	m.stopPhase1Components()
	m.stopPhase0Components()

	m.started = false
	m.logger.Info("DWCP manager stopped successfully with all phases")

	return nil
}

// stopPhase3Components stops resilience layer components (reverse order)
func (m *Manager) stopPhase3Components() {
	m.logger.Info("Stopping Phase 3: Resilience Layer")

	// 8. Resilience Manager
	if m.resilience != nil {
		m.resilience.StopHealthMonitoring()
		m.logger.Info("Resilience manager stopped successfully")
	}

	// 7. Circuit Breaker
	if m.circuitBreaker != nil {
		m.logger.Info("Circuit breaker stopped successfully")
	}
}

// stopPhase2Components stops coordination layer components (reverse order)
func (m *Manager) stopPhase2Components() {
	m.logger.Info("Stopping Phase 2: Coordination Layer")

	// 6. Consensus Layer (ACP)
	if m.consensus != nil {
		if err := m.consensus.Stop(); err != nil {
			m.logger.Error("Failed to stop consensus layer", zap.Error(err))
		} else {
			m.logger.Info("Consensus layer stopped successfully")
		}
	}

	// 5. Sync Layer (ASS)
	if m.sync != nil {
		if err := m.sync.Stop(); err != nil {
			m.logger.Error("Failed to stop sync layer", zap.Error(err))
		} else {
			m.logger.Info("Sync layer stopped successfully")
		}
	}
}

// stopPhase1Components stops intelligence layer components (reverse order)
func (m *Manager) stopPhase1Components() {
	m.logger.Info("Stopping Phase 1: Intelligence Layer")

	// 4. Task Partitioner (ITP)
	if m.partitioner != nil {
		if err := m.partitioner.Stop(); err != nil {
			m.logger.Error("Failed to stop task partitioner", zap.Error(err))
		} else {
			m.logger.Info("Task partitioner stopped successfully")
		}
	}

	// 3. Prediction Engine (PBA)
	if m.prediction != nil {
		if err := m.prediction.Stop(); err != nil {
			m.logger.Error("Failed to stop prediction engine", zap.Error(err))
		} else {
			m.logger.Info("Prediction engine stopped successfully")
		}
	}
}

// stopPhase0Components stops core infrastructure components (reverse order)
func (m *Manager) stopPhase0Components() {
	m.logger.Info("Stopping Phase 0: Core Infrastructure")

	// 2. Compression Layer (HDE)
	if m.compression != nil {
		if err := m.compression.Stop(); err != nil {
			m.logger.Error("Failed to stop compression layer", zap.Error(err))
		} else {
			m.logger.Info("Compression layer stopped successfully")
		}
	}

	// 1. Transport Layer (AMST)
	if m.transport != nil {
		// Transport uses Close() method instead of Stop()
		if err := m.transport.Close(); err != nil {
			m.logger.Error("Failed to close transport", zap.Error(err))
		} else {
			m.logger.Info("Transport layer stopped successfully")
		}
	}
}

// GetMetrics returns the current DWCP metrics
func (m *Manager) GetMetrics() *DWCPMetrics {
	m.metricsMutex.RLock()
	defer m.metricsMutex.RUnlock()

	// Return a copy to avoid race conditions
	metricsCopy := *m.metrics
	return &metricsCopy
}

// IsRunning returns true if the DWCP manager is currently running
// Implements Lifecycle interface
func (m *Manager) IsRunning() bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.started && m.enabled
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

// GetConfig returns a deep copy of the current configuration
func (m *Manager) GetConfig() *Config {
	m.mu.RLock()
	defer m.mu.RUnlock()

	// Deep copy config to heap to ensure thread-safety and avoid stack escape
	return m.config.DeepCopy()
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
// Lock ordering: Always acquire m.mu before m.metricsMutex to prevent deadlocks
func (m *Manager) collectMetrics() {
	// Step 1: Acquire state lock first (consistent lock ordering prevents deadlocks)
	m.mu.RLock()

	// Step 2: Copy state values to local variables to minimize critical section
	// This allows us to release m.mu before acquiring metricsMutex
	enabled := m.enabled

	// Release state lock early to reduce contention
	m.mu.RUnlock()

	// Step 3: Now acquire metrics lock and update
	// Using local variables bridges the mutex boundary safely
	m.metricsMutex.Lock()
	defer m.metricsMutex.Unlock()

	// Update basic status using local copies (no race condition)
	m.metrics.Enabled = enabled
	m.metrics.Version = DWCPVersion

	// TODO: Collect transport metrics (Phase 0-1)
	// Safe to use local 'transport' variable here
	// if transport != nil {
	//     m.metrics.Transport = transport.GetMetrics()
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
	// For very high latency, treat as worst (global WAN) tier
	return NetworkTierTier3
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

// healthMonitoringLoop periodically checks component health and attempts recovery
func (m *Manager) healthMonitoringLoop() {
	defer m.wg.Done()

	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-m.ctx.Done():
			return
		case <-ticker.C:
			if err := m.checkComponentHealth(); err != nil {
				m.logger.Error("Component health check failed", zap.Error(err))
			}
		}
	}
}

// checkComponentHealth verifies all components are healthy and attempts recovery if needed
// RACE FIX: Collect recovery info while holding lock, spawn goroutines after releasing lock
func (m *Manager) checkComponentHealth() error {
	// Step 1: Collect health check results while holding read lock
	m.mu.RLock()

	if !m.enabled || !m.started {
		m.mu.RUnlock()
		return nil
	}

	// Collect recovery tasks to execute outside the lock (prevents race conditions)
	type recoveryTask struct {
		component string
		err       error
	}
	var recoveryTasks []recoveryTask

	// Check transport layer
	if m.transport != nil {
		if err := m.transport.HealthCheck(); err != nil {
			m.logger.Warn("Transport layer unhealthy", zap.Error(err))
			recoveryTasks = append(recoveryTasks, recoveryTask{"transport", err})
		}
	}

	// Check compression layer (Phase 1)
	if m.compression != nil {
		if !m.compression.IsHealthy() {
			m.logger.Warn("Compression layer unhealthy")
			recoveryTasks = append(recoveryTasks, recoveryTask{"compression", fmt.Errorf("compression layer health check failed")})
		}
	}

	// Check prediction engine (Phase 2)
	if m.prediction != nil {
		if !m.prediction.IsHealthy() {
			m.logger.Warn("Prediction engine unhealthy")
			recoveryTasks = append(recoveryTasks, recoveryTask{"prediction", fmt.Errorf("prediction engine health check failed")})
		}
	}

	// Check sync layer (Phase 3)
	if m.sync != nil {
		if !m.sync.IsHealthy() {
			m.logger.Warn("Sync layer unhealthy")
			recoveryTasks = append(recoveryTasks, recoveryTask{"sync", fmt.Errorf("sync layer health check failed")})
		}
	}

	// Check consensus layer (Phase 3)
	if m.consensus != nil {
		if !m.consensus.IsHealthy() {
			m.logger.Warn("Consensus layer unhealthy")
			recoveryTasks = append(recoveryTasks, recoveryTask{"consensus", fmt.Errorf("consensus layer health check failed")})
		}
	}

	// Step 2: Release lock before spawning goroutines (prevents race conditions)
	m.mu.RUnlock()

	// Step 3: Spawn recovery goroutines outside critical section
	for _, task := range recoveryTasks {
		go m.attemptComponentRecovery(task.component, task.err)
	}

	return nil
}

// attemptComponentRecovery tries to recover a failed component with exponential backoff
func (m *Manager) attemptComponentRecovery(componentName string, initialErr error) {
	m.logger.Info("Attempting component recovery",
		zap.String("component", componentName),
		zap.Error(initialErr))

	maxRetries := 3
	for i := 0; i < maxRetries; i++ {
		// Exponential backoff: 1s, 2s, 4s
		backoff := time.Duration(1<<uint(i)) * time.Second
		time.Sleep(backoff)

		m.logger.Info("Recovery attempt",
			zap.String("component", componentName),
			zap.Int("attempt", i+1),
			zap.Int("max_retries", maxRetries),
			zap.Duration("backoff", backoff))

		// Attempt component restart based on type
		var err error
		switch componentName {
		case "transport":
			// Reinitialize transport
			m.mu.Lock()
			if m.transport != nil {
				m.transport.Close()
			}
			err = m.initializeTransport()
			m.mu.Unlock()

		case "compression":
			m.mu.RLock()
			comp := m.compression
			m.mu.RUnlock()
			if comp != nil {
				comp.Stop()
				err = comp.Start(context.Background())
			}

		case "prediction":
			m.mu.RLock()
			pred := m.prediction
			m.mu.RUnlock()
			if pred != nil {
				pred.Stop()
				err = pred.Start(context.Background())
			}

		case "sync":
			m.mu.RLock()
			sync := m.sync
			m.mu.RUnlock()
			if sync != nil {
				sync.Stop()
				err = sync.Start(context.Background())
			}

		case "consensus":
			m.mu.RLock()
			cons := m.consensus
			m.mu.RUnlock()
			if cons != nil {
				cons.Stop()
				err = cons.Start(context.Background())
			}
		}

		if err == nil {
			m.logger.Info("Component recovered successfully",
				zap.String("component", componentName),
				zap.Int("attempts", i+1))
			return
		}

		m.logger.Warn("Recovery attempt failed",
			zap.String("component", componentName),
			zap.Int("attempt", i+1),
			zap.Error(err))
	}

	m.logger.Error("Component recovery failed after all attempts",
		zap.String("component", componentName),
		zap.Int("max_retries", maxRetries))
}

// SendWithCircuitBreaker sends data through transport with circuit breaker protection
func (m *Manager) SendWithCircuitBreaker(data []byte) error {
	if m.circuitBreaker == nil {
		return fmt.Errorf("circuit breaker not initialized")
	}

	return m.circuitBreaker.Call(func() error {
		if m.transport == nil {
			return fmt.Errorf("transport not initialized")
		}
		return m.transport.Send(data)
	})
}

// GetCircuitBreakerState returns the current circuit breaker state
func (m *Manager) GetCircuitBreakerState() string {
	if m.circuitBreaker == nil {
		return "not_initialized"
	}
	return m.circuitBreaker.GetState().String()
}
