package hybrid

import (
	"context"
	"fmt"
	"sync"
	"time"

	"go.uber.org/zap"
	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/upgrade"
)

// HybridManager coordinates hybrid architecture with automatic mode switching
type HybridManager struct {
	mu sync.RWMutex

	logger *zap.Logger

	// Core components
	orchestrator *HybridOrchestrator
	adapter      *ModeAwareAdapter

	// Configuration
	config *HybridConfig

	// State
	isInitialized bool
	isRunning     bool

	// Metrics
	metrics *HybridMetrics
}

// HybridMetrics tracks hybrid architecture metrics
type HybridMetrics struct {
	mu sync.RWMutex

	ModeChanges        int64
	DatacenterTime     time.Duration
	InternetTime       time.Duration
	HybridTime         time.Duration
	LastModeChange     time.Time
	TotalUptime        time.Duration
	FailedSwitches     int64
	SuccessfulSwitches int64
}

// NewHybridManager creates a new hybrid manager
func NewHybridManager(logger *zap.Logger, config *HybridConfig) *HybridManager {
	if config == nil {
		config = DefaultHybridConfig()
	}

	return &HybridManager{
		logger:       logger,
		config:       config,
		orchestrator: NewHybridOrchestrator(logger, config),
		adapter:      NewModeAwareAdapter(logger),
		metrics: &HybridMetrics{
			LastModeChange: time.Now(),
		},
	}
}

// Initialize initializes the hybrid manager
func (hm *HybridManager) Initialize(ctx context.Context) error {
	hm.mu.Lock()
	defer hm.mu.Unlock()

	if hm.isInitialized {
		return fmt.Errorf("hybrid manager already initialized")
	}

	// Validate component implementations
	if err := hm.adapter.ValidateComponentImplementations(); err != nil {
		return fmt.Errorf("component validation failed: %w", err)
	}

	// Register mode change callback
	hm.orchestrator.RegisterModeChangeCallback(hm.onModeChange)

	hm.isInitialized = true

	hm.logger.Info("Hybrid manager initialized",
		zap.Bool("auto_detect", hm.config.AutoDetect),
		zap.Duration("detection_interval", hm.config.DetectionInterval))

	return nil
}

// Start starts the hybrid manager
func (hm *HybridManager) Start(ctx context.Context) error {
	hm.mu.Lock()
	defer hm.mu.Unlock()

	if !hm.isInitialized {
		return fmt.Errorf("hybrid manager not initialized")
	}

	if hm.isRunning {
		return fmt.Errorf("hybrid manager already running")
	}

	if err := hm.orchestrator.Start(); err != nil {
		return fmt.Errorf("failed to start orchestrator: %w", err)
	}

	hm.isRunning = true

	hm.logger.Info("Hybrid manager started")

	return nil
}

// Stop stops the hybrid manager
func (hm *HybridManager) Stop(ctx context.Context) error {
	hm.mu.Lock()
	defer hm.mu.Unlock()

	if !hm.isRunning {
		return nil
	}

	if err := hm.orchestrator.Stop(); err != nil {
		return fmt.Errorf("failed to stop orchestrator: %w", err)
	}

	hm.isRunning = false

	hm.logger.Info("Hybrid manager stopped")

	return nil
}

// RegisterComponent registers a component with both v1 and v3 implementations
func (hm *HybridManager) RegisterComponent(name string, v1Impl, v3Impl interface{}) error {
	hm.mu.Lock()
	defer hm.mu.Unlock()

	return hm.adapter.RegisterComponent(name, v1Impl, v3Impl)
}

// GetComponent returns the active component implementation
func (hm *HybridManager) GetComponent(name string) (interface{}, error) {
	hm.mu.RLock()
	defer hm.mu.RUnlock()

	return hm.adapter.GetComponent(name)
}

// GetCurrentMode returns the current network mode
func (hm *HybridManager) GetCurrentMode() upgrade.NetworkMode {
	hm.mu.RLock()
	defer hm.mu.RUnlock()

	return hm.adapter.GetCurrentMode()
}

// onModeChange is called when mode changes
func (hm *HybridManager) onModeChange(oldMode, newMode upgrade.NetworkMode, reason string) error {
	hm.mu.Lock()
	defer hm.mu.Unlock()

	// Switch adapter to new mode
	if err := hm.adapter.SwitchMode(newMode); err != nil {
		hm.metrics.FailedSwitches++
		hm.logger.Error("Failed to switch mode",
			zap.String("old_mode", oldMode.String()),
			zap.String("new_mode", newMode.String()),
			zap.Error(err))
		return err
	}

	// Update metrics
	hm.metrics.ModeChanges++
	hm.metrics.SuccessfulSwitches++
	hm.metrics.LastModeChange = time.Now()

	hm.logger.Info("Mode change completed",
		zap.String("old_mode", oldMode.String()),
		zap.String("new_mode", newMode.String()),
		zap.String("reason", reason))

	return nil
}

// GetStats returns comprehensive statistics
func (hm *HybridManager) GetStats() map[string]interface{} {
	hm.mu.RLock()
	defer hm.mu.RUnlock()

	hm.metrics.mu.RLock()
	defer hm.metrics.mu.RUnlock()

	return map[string]interface{}{
		"initialized":         hm.isInitialized,
		"running":             hm.isRunning,
		"current_mode":        hm.adapter.GetCurrentMode().String(),
		"mode_changes":        hm.metrics.ModeChanges,
		"successful_switches": hm.metrics.SuccessfulSwitches,
		"failed_switches":     hm.metrics.FailedSwitches,
		"last_mode_change":    hm.metrics.LastModeChange,
		"components":          hm.adapter.ListComponents(),
	}
}

// ListComponents returns list of registered components
func (hm *HybridManager) ListComponents() []string {
	hm.mu.RLock()
	defer hm.mu.RUnlock()

	return hm.adapter.ListComponents()
}

