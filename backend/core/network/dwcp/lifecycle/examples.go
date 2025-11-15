package lifecycle

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"go.uber.org/zap"
)

// ExampleTransportComponent shows how to implement ComponentLifecycle for a transport layer
type ExampleTransportComponent struct {
	*BaseLifecycle

	config *TransportConfig
	conn   interface{} // Simulated connection

	// Component-specific state
	running    bool
	runningMux sync.RWMutex
}

// TransportConfig is example configuration
type TransportConfig struct {
	Address string
	Port    int
	Timeout time.Duration
}

// NewExampleTransportComponent creates a new transport component
func NewExampleTransportComponent(logger *zap.Logger) *ExampleTransportComponent {
	base := NewBaseLifecycle("transport", logger)

	return &ExampleTransportComponent{
		BaseLifecycle: base,
	}
}

// Init initializes the transport component
func (t *ExampleTransportComponent) Init(ctx context.Context, config interface{}) error {
	// Validate and store configuration
	transportConfig, ok := config.(*TransportConfig)
	if !ok {
		return fmt.Errorf("invalid config type")
	}

	t.config = transportConfig

	// Perform initialization
	t.logger.Info("Initializing transport",
		zap.String("address", transportConfig.Address),
		zap.Int("port", transportConfig.Port))

	// Transition to initialized state
	return t.TransitionTo(StateInitialized)
}

// Start starts the transport component
func (t *ExampleTransportComponent) Start(ctx context.Context) error {
	// Transition to starting
	if err := t.TransitionTo(StateStarting); err != nil {
		return err
	}

	// Perform actual startup
	t.logger.Info("Starting transport connection")

	// Simulate connection establishment
	t.runningMux.Lock()
	t.conn = &struct{}{} // Simulated connection
	t.running = true
	t.runningMux.Unlock()

	// Transition to running
	return t.TransitionTo(StateRunning)
}

// Stop gracefully stops the transport component
func (t *ExampleTransportComponent) Stop(ctx context.Context) error {
	// Transition to stopping
	if err := t.TransitionTo(StateStopping); err != nil {
		return err
	}

	t.logger.Info("Stopping transport connection")

	// Drain in-flight operations
	t.drainOperations(ctx)

	// Close connection
	t.runningMux.Lock()
	t.conn = nil
	t.running = false
	t.runningMux.Unlock()

	// Transition to stopped
	return t.TransitionTo(StateStopped)
}

// Shutdown forcefully shuts down with timeout
func (t *ExampleTransportComponent) Shutdown(ctx context.Context, timeout time.Duration) error {
	shutdownCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	// Try graceful stop
	if err := t.Stop(shutdownCtx); err != nil {
		t.logger.Warn("Graceful stop failed, forcing shutdown", zap.Error(err))

		// Force close
		t.runningMux.Lock()
		t.conn = nil
		t.running = false
		t.runningMux.Unlock()

		return t.BaseLifecycle.Shutdown(ctx, timeout)
	}

	return nil
}

// HealthCheck performs transport health check
func (t *ExampleTransportComponent) HealthCheck(ctx context.Context) error {
	// Check base health
	if err := t.BaseLifecycle.HealthCheck(ctx); err != nil {
		return err
	}

	// Check transport-specific health
	t.runningMux.RLock()
	defer t.runningMux.RUnlock()

	if !t.running || t.conn == nil {
		return fmt.Errorf("transport not connected")
	}

	// Perform connectivity check
	// In real implementation, this would ping the remote endpoint
	return nil
}

// drainOperations waits for in-flight operations to complete
func (t *ExampleTransportComponent) drainOperations(ctx context.Context) {
	// Wait for in-flight operations with timeout
	deadline := time.After(t.BaseLifecycle.GetShutdownConfig().DrainTimeout)

	for {
		select {
		case <-ctx.Done():
			return
		case <-deadline:
			t.logger.Warn("Drain timeout reached, some operations may be incomplete")
			return
		default:
			// Check if operations are complete
			// In real implementation, check pending operation count
			if t.hasInflightOperations() {
				time.Sleep(10 * time.Millisecond)
				continue
			}
			return
		}
	}
}

func (t *ExampleTransportComponent) hasInflightOperations() bool {
	// Simulated check
	return false
}

// ExampleCompressionComponent shows a component with dependencies
type ExampleCompressionComponent struct {
	*BaseLifecycle

	transportComp *ExampleTransportComponent
	compressor    interface{} // Simulated compressor
}

// NewExampleCompressionComponent creates a compression component that depends on transport
func NewExampleCompressionComponent(transport *ExampleTransportComponent, logger *zap.Logger) *ExampleCompressionComponent {
	base := NewBaseLifecycle("compression", logger)
	base.SetDependencies([]string{"transport"})

	return &ExampleCompressionComponent{
		BaseLifecycle: base,
		transportComp: transport,
	}
}

// Init initializes the compression component
func (c *ExampleCompressionComponent) Init(ctx context.Context, config interface{}) error {
	c.logger.Info("Initializing compression engine")

	// Initialize compressor
	c.compressor = &struct{}{} // Simulated compressor

	return c.TransitionTo(StateInitialized)
}

// Start starts the compression component
func (c *ExampleCompressionComponent) Start(ctx context.Context) error {
	// Verify dependency is running
	if c.transportComp.GetState() != StateRunning {
		return fmt.Errorf("transport dependency not running")
	}

	if err := c.TransitionTo(StateStarting); err != nil {
		return err
	}

	c.logger.Info("Starting compression engine")

	// Start compression
	// In real implementation, start compression workers

	return c.TransitionTo(StateRunning)
}

// Stop stops the compression component
func (c *ExampleCompressionComponent) Stop(ctx context.Context) error {
	if err := c.TransitionTo(StateStopping); err != nil {
		return err
	}

	c.logger.Info("Stopping compression engine")

	// Stop compression workers
	// Flush pending compressions

	return c.TransitionTo(StateStopped)
}

// HealthCheck checks compression component health
func (c *ExampleCompressionComponent) HealthCheck(ctx context.Context) error {
	if err := c.BaseLifecycle.HealthCheck(ctx); err != nil {
		return err
	}

	if c.compressor == nil {
		return fmt.Errorf("compressor not initialized")
	}

	return nil
}

// ExampleRecoverableComponent shows recovery implementation
type ExampleRecoverableComponent struct {
	*BaseLifecycle
	failureCount atomic.Int32
}

// NewExampleRecoverableComponent creates a recoverable component
func NewExampleRecoverableComponent(logger *zap.Logger) *ExampleRecoverableComponent {
	return &ExampleRecoverableComponent{
		BaseLifecycle: NewBaseLifecycle("recoverable", logger),
	}
}

// Recover attempts to recover from failure
func (r *ExampleRecoverableComponent) Recover(ctx context.Context) error {
	r.logger.Info("Attempting recovery",
		zap.Int32("failure_count", r.failureCount.Load()))

	// Reset state
	if err := r.TransitionTo(StateInitialized); err != nil {
		return fmt.Errorf("failed to reset state: %w", err)
	}

	// Reinitialize
	if err := r.Init(ctx, nil); err != nil {
		r.failureCount.Add(1)
		return fmt.Errorf("reinitialization failed: %w", err)
	}

	// Restart
	if err := r.Start(ctx); err != nil {
		r.failureCount.Add(1)
		return fmt.Errorf("restart failed: %w", err)
	}

	r.logger.Info("Recovery successful")
	return nil
}

// GetRecoveryStrategy returns the recovery strategy
func (r *ExampleRecoverableComponent) GetRecoveryStrategy() RecoveryStrategy {
	return RecoveryStrategy{
		MaxRetries:         3,
		RetryBackoff:       1 * time.Second,
		ExponentialBackoff: true,
		MaxBackoff:         30 * time.Second,
		FailFast:           false,
	}
}

// ExampleUsage demonstrates how to use the lifecycle system
func ExampleUsage() {
	logger, _ := zap.NewProduction()

	// Create lifecycle manager
	manager := NewManager(DefaultManagerConfig(), logger)

	// Create components
	transport := NewExampleTransportComponent(logger)
	compression := NewExampleCompressionComponent(transport, logger)

	// Register components
	_ = manager.Register(transport)
	_ = manager.Register(compression)

	// Start all components (automatically handles dependencies)
	ctx := context.Background()
	if err := manager.StartAll(ctx); err != nil {
		logger.Fatal("Failed to start components", zap.Error(err))
	}

	// Components are now running...

	// Stop all components (automatically handles reverse dependencies)
	if err := manager.StopAll(ctx); err != nil {
		logger.Error("Failed to stop components", zap.Error(err))
	}
}
