package hybrid

import (
	"context"
	"testing"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/upgrade"
	"go.uber.org/zap"
)

func TestHybridOrchestrator(t *testing.T) {
	logger := zap.NewNop()
	config := DefaultHybridConfig()
	config.DetectionInterval = 100 * time.Millisecond
	config.CooldownPeriod = 50 * time.Millisecond

	orchestrator := NewHybridOrchestrator(logger, config)

	// Test initial mode
	if orchestrator.GetCurrentMode() != upgrade.ModeHybrid {
		t.Errorf("Expected initial mode to be Hybrid, got %v", orchestrator.GetCurrentMode())
	}

	// Test start
	if err := orchestrator.Start(); err != nil {
		t.Fatalf("Failed to start orchestrator: %v", err)
	}

	// Test stop
	if err := orchestrator.Stop(); err != nil {
		t.Fatalf("Failed to stop orchestrator: %v", err)
	}
}

func TestModeAwareAdapter(t *testing.T) {
	logger := zap.NewNop()
	adapter := NewModeAwareAdapter(logger)

	// Mock implementations
	v1Impl := "v1-transport"
	v3Impl := "v3-transport"

	// Register component
	if err := adapter.RegisterComponent("transport", v1Impl, v3Impl); err != nil {
		t.Fatalf("Failed to register component: %v", err)
	}

	// Test get component (should be v1 by default)
	impl, err := adapter.GetComponent("transport")
	if err != nil {
		t.Fatalf("Failed to get component: %v", err)
	}
	if impl != v1Impl {
		t.Errorf("Expected v1 implementation, got %v", impl)
	}

	// Switch to internet mode (should use v3)
	if err := adapter.SwitchMode(upgrade.ModeInternet); err != nil {
		t.Fatalf("Failed to switch mode: %v", err)
	}

	impl, err = adapter.GetComponent("transport")
	if err != nil {
		t.Fatalf("Failed to get component: %v", err)
	}
	if impl != v3Impl {
		t.Errorf("Expected v3 implementation, got %v", impl)
	}

	// Switch to datacenter mode (should use v1)
	if err := adapter.SwitchMode(upgrade.ModeDatacenter); err != nil {
		t.Fatalf("Failed to switch mode: %v", err)
	}

	impl, err = adapter.GetComponent("transport")
	if err != nil {
		t.Fatalf("Failed to get component: %v", err)
	}
	if impl != v1Impl {
		t.Errorf("Expected v1 implementation, got %v", impl)
	}
}

func TestHybridManager(t *testing.T) {
	logger := zap.NewNop()
	config := DefaultHybridConfig()
	config.AutoDetect = false // Disable auto-detection for testing

	manager := NewHybridManager(logger, config)

	// Test initialize
	ctx := context.Background()
	if err := manager.Initialize(ctx); err != nil {
		t.Fatalf("Failed to initialize: %v", err)
	}

	// Register components
	if err := manager.RegisterComponent("transport", "v1-transport", "v3-transport"); err != nil {
		t.Fatalf("Failed to register component: %v", err)
	}

	// Test start
	if err := manager.Start(ctx); err != nil {
		t.Fatalf("Failed to start: %v", err)
	}

	// Test get component
	impl, err := manager.GetComponent("transport")
	if err != nil {
		t.Fatalf("Failed to get component: %v", err)
	}
	if impl != "v1-transport" {
		t.Errorf("Expected v1-transport, got %v", impl)
	}

	// Test get stats
	stats := manager.GetStats()
	if stats["initialized"] != true {
		t.Errorf("Expected initialized=true, got %v", stats["initialized"])
	}

	// Test stop
	if err := manager.Stop(ctx); err != nil {
		t.Fatalf("Failed to stop: %v", err)
	}
}

func TestModeChangeCallback(t *testing.T) {
	logger := zap.NewNop()
	config := DefaultHybridConfig()
	config.AutoDetect = false

	orchestrator := NewHybridOrchestrator(logger, config)

	callbackCalled := false
	var newMode upgrade.NetworkMode

	orchestrator.RegisterModeChangeCallback(func(old, new upgrade.NetworkMode, reason string) error {
		callbackCalled = true
		newMode = new
		return nil
	})

	// Manually trigger mode change
	orchestrator.mu.Lock()
	orchestrator.switchMode(upgrade.ModeInternet, "test")
	orchestrator.mu.Unlock()

	if !callbackCalled {
		t.Error("Expected callback to be called")
	}
	if newMode != upgrade.ModeInternet {
		t.Errorf("Expected new mode to be Internet, got %v", newMode)
	}
}

func TestComponentValidation(t *testing.T) {
	logger := zap.NewNop()
	adapter := NewModeAwareAdapter(logger)

	// Try to validate without registering components
	if err := adapter.ValidateComponentImplementations(); err != nil {
		t.Fatalf("Validation should succeed for empty adapter: %v", err)
	}

	// Register component
	adapter.RegisterComponent("transport", "v1", "v3")

	// Validate should succeed
	if err := adapter.ValidateComponentImplementations(); err != nil {
		t.Fatalf("Validation should succeed: %v", err)
	}
}
