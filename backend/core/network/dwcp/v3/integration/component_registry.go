package integration

import (
	"context"
	"fmt"
	"sync"

	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/hybrid"
	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/upgrade"
	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/v3/encoding"
	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/v3/prediction"
	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/v3/transport"
	"go.uber.org/zap"
)

// ComponentRegistry manages all DWCP v3 components with hybrid mode support
type ComponentRegistry struct {
	mu sync.RWMutex

	logger *zap.Logger

	// Hybrid manager
	hybridManager *hybrid.HybridManager

	// v3 Components
	amst *transport.AMSTv3
	hde  *encoding.HDEv3
	pba  *prediction.PBAv3

	// Component states
	initialized bool
	started     bool

	// Configuration
	config *RegistryConfig
}

// RegistryConfig configuration for component registry
type RegistryConfig struct {
	NodeID string

	// Component configurations
	AMSTConfig *transport.AMSTv3Config
	HDEConfig  *encoding.HDEv3Config
	PBAConfig  *prediction.PBAv3Config

	// Hybrid configuration
	HybridConfig *hybrid.HybridConfig

	// Feature flags
	EnableAMST bool
	EnableHDE  bool
	EnablePBA  bool
}

// DefaultRegistryConfig returns default configuration
func DefaultRegistryConfig(nodeID string) *RegistryConfig {
	return &RegistryConfig{
		NodeID:       nodeID,
		AMSTConfig:   transport.DefaultAMSTv3Config(),
		HDEConfig:    encoding.DefaultHDEv3Config(nodeID),
		PBAConfig:    prediction.DefaultPBAv3Config(),
		HybridConfig: hybrid.DefaultHybridConfig(),
		EnableAMST:   true,
		EnableHDE:    true,
		EnablePBA:    true,
	}
}

// NewComponentRegistry creates a new component registry
func NewComponentRegistry(logger *zap.Logger, config *RegistryConfig) (*ComponentRegistry, error) {
	if config == nil {
		config = DefaultRegistryConfig("default-node")
	}

	return &ComponentRegistry{
		logger: logger,
		config: config,
	}, nil
}

// Initialize initializes all components
func (cr *ComponentRegistry) Initialize(ctx context.Context) error {
	cr.mu.Lock()
	defer cr.mu.Unlock()

	if cr.initialized {
		return fmt.Errorf("component registry already initialized")
	}

	// Create hybrid manager
	cr.hybridManager = hybrid.NewHybridManager(cr.logger, cr.config.HybridConfig)

	// Initialize hybrid manager
	if err := cr.hybridManager.Initialize(ctx); err != nil {
		return fmt.Errorf("failed to initialize hybrid manager: %w", err)
	}

	// Initialize components based on feature flags
	modeDetector := upgrade.NewModeDetector()

	if cr.config.EnableAMST {
		amst, err := transport.NewAMSTv3(cr.config.AMSTConfig, modeDetector, cr.logger)
		if err != nil {
			return fmt.Errorf("failed to create AMST v3: %w", err)
		}
		cr.amst = amst
	}

	if cr.config.EnableHDE {
		hde, err := encoding.NewHDEv3(cr.config.HDEConfig)
		if err != nil {
			return fmt.Errorf("failed to create HDE v3: %w", err)
		}
		cr.hde = hde
	}

	if cr.config.EnablePBA {
		pba, err := prediction.NewPBAv3(cr.config.PBAConfig)
		if err != nil {
			return fmt.Errorf("failed to create PBA v3: %w", err)
		}
		cr.pba = pba
	}

	cr.initialized = true

	cr.logger.Info("Component registry initialized",
		zap.String("node_id", cr.config.NodeID),
		zap.Bool("amst", cr.config.EnableAMST),
		zap.Bool("hde", cr.config.EnableHDE),
		zap.Bool("pba", cr.config.EnablePBA))

	return nil
}

// Start starts all components
func (cr *ComponentRegistry) Start(ctx context.Context) error {
	cr.mu.Lock()
	defer cr.mu.Unlock()

	if !cr.initialized {
		return fmt.Errorf("component registry not initialized")
	}

	if cr.started {
		return fmt.Errorf("component registry already started")
	}

	// Start hybrid manager
	if err := cr.hybridManager.Start(ctx); err != nil {
		return fmt.Errorf("failed to start hybrid manager: %w", err)
	}

	cr.started = true

	cr.logger.Info("Component registry started")

	return nil
}

// Stop stops all components
func (cr *ComponentRegistry) Stop(ctx context.Context) error {
	cr.mu.Lock()
	defer cr.mu.Unlock()

	if !cr.started {
		return nil
	}

	// Stop hybrid manager
	if err := cr.hybridManager.Stop(ctx); err != nil {
		return fmt.Errorf("failed to stop hybrid manager: %w", err)
	}

	cr.started = false
	cr.logger.Info("Component registry stopped")

	return nil
}

// GetAMST returns the AMST v3 component
func (cr *ComponentRegistry) GetAMST() *transport.AMSTv3 {
	cr.mu.RLock()
	defer cr.mu.RUnlock()
	return cr.amst
}

// GetHDE returns the HDE v3 component
func (cr *ComponentRegistry) GetHDE() *encoding.HDEv3 {
	cr.mu.RLock()
	defer cr.mu.RUnlock()
	return cr.hde
}

// GetPBA returns the PBA v3 component
func (cr *ComponentRegistry) GetPBA() *prediction.PBAv3 {
	cr.mu.RLock()
	defer cr.mu.RUnlock()
	return cr.pba
}

// GetCurrentMode returns the current network mode
func (cr *ComponentRegistry) GetCurrentMode() upgrade.NetworkMode {
	cr.mu.RLock()
	defer cr.mu.RUnlock()

	if cr.hybridManager == nil {
		return upgrade.ModeHybrid
	}

	return cr.hybridManager.GetCurrentMode()
}

// GetStats returns comprehensive statistics
func (cr *ComponentRegistry) GetStats() map[string]interface{} {
	cr.mu.RLock()
	defer cr.mu.RUnlock()

	stats := map[string]interface{}{
		"initialized": cr.initialized,
		"started":     cr.started,
		"node_id":     cr.config.NodeID,
		"components": map[string]bool{
			"amst": cr.amst != nil,
			"hde":  cr.hde != nil,
			"pba":  cr.pba != nil,
		},
	}

	if cr.hybridManager != nil {
		stats["hybrid"] = cr.hybridManager.GetStats()
	}

	return stats
}
