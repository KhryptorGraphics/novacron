package hybrid

import (
	"context"
	"fmt"
	"sync"
	"time"

	"go.uber.org/zap"
	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/upgrade"
)

// HybridOrchestrator manages automatic switching between datacenter and internet modes
type HybridOrchestrator struct {
	mu sync.RWMutex

	logger *zap.Logger

	// Current mode
	currentMode upgrade.NetworkMode

	// Mode detector
	modeDetector *upgrade.ModeDetector

	// Configuration
	config *HybridConfig

	// Callbacks for mode changes
	modeChangeCallbacks []ModeChangeCallback

	// Metrics
	modeChanges      int64
	lastModeChange   time.Time
	modeChangeReason string

	// Context for background operations
	ctx    context.Context
	cancel context.CancelFunc

	// Monitoring
	monitoringTicker *time.Ticker
	isRunning        bool
}

// HybridConfig configures hybrid mode behavior
type HybridConfig struct {
	// Enable hybrid mode
	Enabled bool

	// Enable automatic mode detection
	AutoDetect bool

	// Detection interval
	DetectionInterval time.Duration

	// Thresholds for mode switching
	DatacenterLatencyThreshold   time.Duration
	DatacenterBandwidthThreshold int64
	InternetLatencyThreshold     time.Duration
	InternetBandwidthThreshold   int64

	// Hysteresis to prevent flapping
	Hysteresis float64

	// Cooldown period between switches
	CooldownPeriod time.Duration

	// Graceful transition
	GracefulTransition bool
	DrainTimeout       time.Duration
}

// ModeChangeCallback is called when mode changes
type ModeChangeCallback func(oldMode, newMode upgrade.NetworkMode, reason string) error

// DefaultHybridConfig returns default configuration
func DefaultHybridConfig() *HybridConfig {
	return &HybridConfig{
		Enabled:                      true,
		AutoDetect:                   true,
		DetectionInterval:            10 * time.Second,
		DatacenterLatencyThreshold:   10 * time.Millisecond,
		DatacenterBandwidthThreshold: 1e9, // 1 Gbps
		InternetLatencyThreshold:     50 * time.Millisecond,
		InternetBandwidthThreshold:   1e9, // 1 Gbps
		Hysteresis:                   0.1,
		CooldownPeriod:               30 * time.Second,
		GracefulTransition:           true,
		DrainTimeout:                 10 * time.Second,
	}
}

// NewHybridOrchestrator creates a new hybrid orchestrator
func NewHybridOrchestrator(logger *zap.Logger, config *HybridConfig) *HybridOrchestrator {
	if config == nil {
		config = DefaultHybridConfig()
	}

	ctx, cancel := context.WithCancel(context.Background())

	return &HybridOrchestrator{
		logger:              logger,
		currentMode:         upgrade.ModeHybrid,
		modeDetector:        upgrade.NewModeDetector(),
		config:              config,
		modeChangeCallbacks: make([]ModeChangeCallback, 0),
		ctx:                 ctx,
		cancel:              cancel,
		lastModeChange:      time.Now(),
	}
}

// Start begins monitoring and automatic mode switching
func (ho *HybridOrchestrator) Start() error {
	ho.mu.Lock()
	defer ho.mu.Unlock()

	if ho.isRunning {
		return fmt.Errorf("hybrid orchestrator already running")
	}

	if !ho.config.Enabled || !ho.config.AutoDetect {
		ho.logger.Info("Hybrid mode disabled or auto-detection disabled")
		return nil
	}

	ho.isRunning = true
	ho.monitoringTicker = time.NewTicker(ho.config.DetectionInterval)

	go ho.monitoringLoop()

	ho.logger.Info("Hybrid orchestrator started",
		zap.Duration("detection_interval", ho.config.DetectionInterval),
		zap.Duration("cooldown_period", ho.config.CooldownPeriod))

	return nil
}

// Stop stops monitoring and mode switching
func (ho *HybridOrchestrator) Stop() error {
	ho.mu.Lock()
	defer ho.mu.Unlock()

	if !ho.isRunning {
		return nil
	}

	ho.isRunning = false
	ho.cancel()

	if ho.monitoringTicker != nil {
		ho.monitoringTicker.Stop()
	}

	ho.logger.Info("Hybrid orchestrator stopped")
	return nil
}

// monitoringLoop continuously monitors network conditions
func (ho *HybridOrchestrator) monitoringLoop() {
	for {
		select {
		case <-ho.ctx.Done():
			return
		case <-ho.monitoringTicker.C:
			ho.checkAndSwitchMode()
		}
	}
}

// checkAndSwitchMode detects current mode and switches if necessary
func (ho *HybridOrchestrator) checkAndSwitchMode() {
	ho.mu.Lock()
	defer ho.mu.Unlock()

	// Check cooldown period
	if time.Since(ho.lastModeChange) < ho.config.CooldownPeriod {
		return
	}

	newMode := ho.modeDetector.DetectMode(ho.ctx)

	if newMode != ho.currentMode {
		ho.switchMode(newMode, "automatic detection")
	}
}

// switchMode switches to a new mode
func (ho *HybridOrchestrator) switchMode(newMode upgrade.NetworkMode, reason string) {
	oldMode := ho.currentMode
	ho.currentMode = newMode
	ho.modeChanges++
	ho.lastModeChange = time.Now()
	ho.modeChangeReason = reason

	ho.logger.Info("Network mode switched",
		zap.String("old_mode", oldMode.String()),
		zap.String("new_mode", newMode.String()),
		zap.String("reason", reason))

	// Call registered callbacks
	for _, callback := range ho.modeChangeCallbacks {
		if err := callback(oldMode, newMode, reason); err != nil {
			ho.logger.Error("Mode change callback failed",
				zap.Error(err),
				zap.String("old_mode", oldMode.String()),
				zap.String("new_mode", newMode.String()))
		}
	}
}

// GetCurrentMode returns the current mode
func (ho *HybridOrchestrator) GetCurrentMode() upgrade.NetworkMode {
	ho.mu.RLock()
	defer ho.mu.RUnlock()
	return ho.currentMode
}

// RegisterModeChangeCallback registers a callback for mode changes
func (ho *HybridOrchestrator) RegisterModeChangeCallback(callback ModeChangeCallback) {
	ho.mu.Lock()
	defer ho.mu.Unlock()
	ho.modeChangeCallbacks = append(ho.modeChangeCallbacks, callback)
}

// GetStats returns statistics about mode switching
func (ho *HybridOrchestrator) GetStats() map[string]interface{} {
	ho.mu.RLock()
	defer ho.mu.RUnlock()

	return map[string]interface{}{
		"current_mode":      ho.currentMode.String(),
		"mode_changes":      ho.modeChanges,
		"last_mode_change":  ho.lastModeChange,
		"mode_change_reason": ho.modeChangeReason,
		"is_running":        ho.isRunning,
	}
}

