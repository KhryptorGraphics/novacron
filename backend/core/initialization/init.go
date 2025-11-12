// Package initialization provides system initialization for NovaCron
package initialization

import (
	"context"
	"fmt"
	"time"

	"novacron/backend/core/initialization/config"
	"novacron/backend/core/initialization/di"
	"novacron/backend/core/initialization/logger"
	"novacron/backend/core/initialization/orchestrator"
	"novacron/backend/core/initialization/recovery"
)

// Initializer manages the complete system initialization process
type Initializer struct {
	config      *config.Config
	logger      *logger.Logger
	container   *di.Container
	orchestrator *orchestrator.Orchestrator
	recovery    *recovery.RecoveryManager
	metrics     *MetricsCollector
}

// MetricsCollector collects initialization metrics
type MetricsCollector struct {
	componentInitDurations map[string]time.Duration
	componentInitSuccess   map[string]bool
}

// NewMetricsCollector creates a new metrics collector
func NewMetricsCollector() *MetricsCollector {
	return &MetricsCollector{
		componentInitDurations: make(map[string]time.Duration),
		componentInitSuccess:   make(map[string]bool),
	}
}

// RecordComponentInit records component initialization metrics
func (m *MetricsCollector) RecordComponentInit(name string, duration time.Duration, success bool) {
	m.componentInitDurations[name] = duration
	m.componentInitSuccess[name] = success
}

// RecordComponentShutdown records component shutdown metrics
func (m *MetricsCollector) RecordComponentShutdown(name string, duration time.Duration, success bool) {
	// Implementation for shutdown metrics
}

// SetComponentStatus sets component status
func (m *MetricsCollector) SetComponentStatus(name string, status string) {
	// Implementation for status tracking
}

// GetMetrics returns collected metrics
func (m *MetricsCollector) GetMetrics() map[string]interface{} {
	return map[string]interface{}{
		"init_durations": m.componentInitDurations,
		"init_success":   m.componentInitSuccess,
	}
}

// NewInitializer creates a new system initializer
func NewInitializer(configPath string) (*Initializer, error) {
	// Load configuration
	loader := config.NewLoader(configPath)
	cfg, err := loader.Load()
	if err != nil {
		return nil, fmt.Errorf("failed to load config: %w", err)
	}

	// Apply environment overrides
	if err := loader.LoadFromEnv(cfg); err != nil {
		return nil, fmt.Errorf("failed to load env overrides: %w", err)
	}

	// Create logger
	log, err := logger.NewLogger(logger.Config{
		Level: cfg.System.LogLevel,
		File:  cfg.System.DataDir + "/novacron.log",
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create logger: %w", err)
	}

	// Create metrics collector
	metrics := NewMetricsCollector()

	// Create orchestrator
	orch := orchestrator.NewOrchestrator(log, metrics)

	// Create recovery manager
	recoveryPolicy := recovery.DefaultRecoveryPolicy()
	recoveryMgr := recovery.NewRecoveryManager(recoveryPolicy, log)

	// Create dependency injection container
	container := di.NewContainer()

	// Register core services
	if err := registerCoreServices(container, cfg, log, metrics, recoveryMgr); err != nil {
		return nil, fmt.Errorf("failed to register core services: %w", err)
	}

	return &Initializer{
		config:       cfg,
		logger:       log,
		container:    container,
		orchestrator: orch,
		recovery:     recoveryMgr,
		metrics:      metrics,
	}, nil
}

// Initialize performs system initialization
func (init *Initializer) Initialize(ctx context.Context) error {
	init.logger.Info("Starting NovaCron initialization",
		"node_id", init.config.System.NodeID,
		"version", "1.0.0",
	)

	startTime := time.Now()

	// Save initial checkpoint
	if err := init.recovery.SaveCheckpoint("init_start", map[string]interface{}{
		"timestamp": startTime,
		"config":    init.config,
	}); err != nil {
		return fmt.Errorf("failed to save checkpoint: %w", err)
	}

	// Register components
	if err := init.registerComponents(); err != nil {
		return fmt.Errorf("failed to register components: %w", err)
	}

	// Initialize with recovery
	err := init.recovery.WithRetry(ctx, "system-init", func() error {
		// Use parallel initialization for independent components
		return init.orchestrator.InitializeParallel(ctx, init.config.System.MaxConcurrency)
	})

	if err != nil {
		init.logger.Error("Initialization failed, attempting rollback", err)

		// Attempt rollback
		if rollbackErr := init.recovery.Rollback(ctx); rollbackErr != nil {
			init.logger.Error("Rollback failed", rollbackErr)
			return fmt.Errorf("initialization failed and rollback failed: %w (rollback: %v)", err, rollbackErr)
		}

		return fmt.Errorf("initialization failed (rollback successful): %w", err)
	}

	duration := time.Since(startTime)
	init.logger.Info("NovaCron initialization completed successfully", "duration", duration)

	// Save completion checkpoint
	if err := init.recovery.SaveCheckpoint("init_complete", map[string]interface{}{
		"timestamp": time.Now(),
		"duration":  duration,
	}); err != nil {
		init.logger.Warn("Failed to save completion checkpoint", "error", err.Error())
	}

	return nil
}

// Shutdown performs graceful system shutdown
func (init *Initializer) Shutdown(ctx context.Context) error {
	init.logger.Info("Starting graceful shutdown")

	shutdownCtx, cancel := context.WithTimeout(ctx, init.config.System.ShutdownTimeout)
	defer cancel()

	// Shutdown all components
	if err := init.orchestrator.Shutdown(shutdownCtx); err != nil {
		init.logger.Error("Shutdown encountered errors", err)
		return err
	}

	// Close logger
	if err := init.logger.Close(); err != nil {
		return fmt.Errorf("failed to close logger: %w", err)
	}

	return nil
}

// GetContainer returns the DI container
func (init *Initializer) GetContainer() *di.Container {
	return init.container
}

// GetConfig returns the configuration
func (init *Initializer) GetConfig() *config.Config {
	return init.config
}

// GetLogger returns the logger
func (init *Initializer) GetLogger() *logger.Logger {
	return init.logger
}

// HealthCheck performs system health check
func (init *Initializer) HealthCheck(ctx context.Context) error {
	return init.orchestrator.HealthCheck(ctx)
}

// GetStatus returns initialization status
func (init *Initializer) GetStatus() map[string]interface{} {
	componentStatus := init.orchestrator.GetStatus()
	metrics := init.metrics.GetMetrics()

	return map[string]interface{}{
		"components": componentStatus,
		"metrics":    metrics,
		"config":     init.config,
	}
}

// registerComponents registers all system components with the orchestrator
func (init *Initializer) registerComponents() error {
	// This will be extended with actual components in Phase 2
	// For now, register placeholder components

	init.logger.Info("Registering system components")

	// Components will be registered here based on the architecture design
	// Example component registration flow:
	// 1. Network layer components
	// 2. Storage layer components
	// 3. DWCP components (AMST, HDE, PBA, ASS, ITP, ACP)
	// 4. Application layer components

	return nil
}

// registerCoreServices registers core services in the DI container
func registerCoreServices(
	container *di.Container,
	cfg *config.Config,
	log *logger.Logger,
	metrics *MetricsCollector,
	recovery *recovery.RecoveryManager,
) error {
	// Register config
	if err := container.RegisterInstance("config", cfg); err != nil {
		return err
	}

	// Register logger
	if err := container.RegisterInstance("logger", log); err != nil {
		return err
	}

	// Register metrics
	if err := container.RegisterInstance("metrics", metrics); err != nil {
		return err
	}

	// Register recovery manager
	if err := container.RegisterInstance("recovery", recovery); err != nil {
		return err
	}

	return nil
}

// GenerateDefaultConfig generates a default configuration file
func GenerateDefaultConfig(path string) error {
	return config.GenerateDefault(path)
}
