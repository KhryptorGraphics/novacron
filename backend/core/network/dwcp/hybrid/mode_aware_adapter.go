package hybrid

import (
	"fmt"
	"sync"

	"go.uber.org/zap"
	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/upgrade"
)

// ModeAwareAdapter adapts between v1 and v3 components based on network mode
type ModeAwareAdapter struct {
	mu sync.RWMutex

	logger *zap.Logger

	// Current mode
	currentMode upgrade.NetworkMode

	// Component implementations
	v1Components map[string]interface{}
	v3Components map[string]interface{}

	// Active components (selected based on mode)
	activeComponents map[string]interface{}

	// Component names
	componentNames []string
}

// NewModeAwareAdapter creates a new mode-aware adapter
func NewModeAwareAdapter(logger *zap.Logger) *ModeAwareAdapter {
	return &ModeAwareAdapter{
		logger:           logger,
		currentMode:      upgrade.ModeHybrid,
		v1Components:     make(map[string]interface{}),
		v3Components:     make(map[string]interface{}),
		activeComponents: make(map[string]interface{}),
		componentNames:   make([]string, 0),
	}
}

// RegisterComponent registers both v1 and v3 implementations of a component
func (maa *ModeAwareAdapter) RegisterComponent(name string, v1Impl, v3Impl interface{}) error {
	maa.mu.Lock()
	defer maa.mu.Unlock()

	if v1Impl == nil || v3Impl == nil {
		return fmt.Errorf("both v1 and v3 implementations required for component %s", name)
	}

	maa.v1Components[name] = v1Impl
	maa.v3Components[name] = v3Impl
	maa.componentNames = append(maa.componentNames, name)

	// Default to v1
	maa.activeComponents[name] = v1Impl

	maa.logger.Debug("Component registered",
		zap.String("component", name))

	return nil
}

// SwitchMode switches all components to the specified mode
func (maa *ModeAwareAdapter) SwitchMode(newMode upgrade.NetworkMode) error {
	maa.mu.Lock()
	defer maa.mu.Unlock()

	if newMode == maa.currentMode {
		return nil
	}

	oldMode := maa.currentMode
	maa.currentMode = newMode

	// Switch all components
	for _, name := range maa.componentNames {
		switch newMode {
		case upgrade.ModeDatacenter:
			// Datacenter mode: use v1 (optimized for low latency, high bandwidth)
			maa.activeComponents[name] = maa.v1Components[name]
		case upgrade.ModeInternet:
			// Internet mode: use v3 (optimized for high latency, Byzantine tolerance)
			maa.activeComponents[name] = maa.v3Components[name]
		case upgrade.ModeHybrid:
			// Hybrid mode: use v3 (adaptive)
			maa.activeComponents[name] = maa.v3Components[name]
		}
	}

	maa.logger.Info("Mode switched for all components",
		zap.String("old_mode", oldMode.String()),
		zap.String("new_mode", newMode.String()),
		zap.Int("components_switched", len(maa.componentNames)))

	return nil
}

// GetComponent returns the active implementation of a component
func (maa *ModeAwareAdapter) GetComponent(name string) (interface{}, error) {
	maa.mu.RLock()
	defer maa.mu.RUnlock()

	impl, exists := maa.activeComponents[name]
	if !exists {
		return nil, fmt.Errorf("component %s not registered", name)
	}

	return impl, nil
}

// GetCurrentMode returns the current mode
func (maa *ModeAwareAdapter) GetCurrentMode() upgrade.NetworkMode {
	maa.mu.RLock()
	defer maa.mu.RUnlock()
	return maa.currentMode
}

// GetComponentStats returns statistics about registered components
func (maa *ModeAwareAdapter) GetComponentStats() map[string]interface{} {
	maa.mu.RLock()
	defer maa.mu.RUnlock()

	stats := map[string]interface{}{
		"current_mode":      maa.currentMode.String(),
		"total_components":  len(maa.componentNames),
		"components":        maa.componentNames,
	}

	return stats
}

// ListComponents returns list of registered components
func (maa *ModeAwareAdapter) ListComponents() []string {
	maa.mu.RLock()
	defer maa.mu.RUnlock()

	components := make([]string, len(maa.componentNames))
	copy(components, maa.componentNames)
	return components
}

// ValidateComponentImplementations validates that all components have both v1 and v3 implementations
func (maa *ModeAwareAdapter) ValidateComponentImplementations() error {
	maa.mu.RLock()
	defer maa.mu.RUnlock()

	for _, name := range maa.componentNames {
		if _, hasV1 := maa.v1Components[name]; !hasV1 {
			return fmt.Errorf("component %s missing v1 implementation", name)
		}
		if _, hasV3 := maa.v3Components[name]; !hasV3 {
			return fmt.Errorf("component %s missing v3 implementation", name)
		}
	}

	maa.logger.Info("All component implementations validated",
		zap.Int("components", len(maa.componentNames)))

	return nil
}

