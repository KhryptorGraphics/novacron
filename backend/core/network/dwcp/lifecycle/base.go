package lifecycle

import (
	"context"
	"fmt"
	"sync"
	"time"

	"go.uber.org/zap"
)

// BaseLifecycle provides common lifecycle implementation
// Components can embed this to get basic lifecycle functionality
type BaseLifecycle struct {
	name         string
	stateMachine *StateMachine
	dependencies []string
	logger       *zap.Logger

	// Timing metrics
	startTime      time.Time
	stopTime       time.Time
	startupMutex   sync.RWMutex
	shutdownConfig ShutdownConfig

	// Health tracking
	lastHealthCheck       time.Time
	lastHealthCheckResult error
	healthMutex           sync.RWMutex

	// Observer support
	observers      []Observer
	observersMutex sync.RWMutex
}

// NewBaseLifecycle creates a new base lifecycle
func NewBaseLifecycle(name string, logger *zap.Logger) *BaseLifecycle {
	if logger == nil {
		logger, _ = zap.NewProduction()
	}

	return &BaseLifecycle{
		name:           name,
		stateMachine:   NewStateMachine(name),
		dependencies:   make([]string, 0),
		logger:         logger,
		shutdownConfig: DefaultShutdownConfig(),
		observers:      make([]Observer, 0),
	}
}

// GetName returns component name
func (b *BaseLifecycle) GetName() string {
	return b.name
}

// GetState returns current lifecycle state
func (b *BaseLifecycle) GetState() State {
	return b.stateMachine.GetState()
}

// GetDependencies returns component dependencies
func (b *BaseLifecycle) GetDependencies() []string {
	return b.dependencies
}

// SetDependencies sets component dependencies
func (b *BaseLifecycle) SetDependencies(deps []string) {
	b.dependencies = deps
}

// TransitionTo transitions to new state with validation
func (b *BaseLifecycle) TransitionTo(newState State) error {
	oldState := b.stateMachine.GetState()

	if err := b.stateMachine.TransitionTo(newState); err != nil {
		b.logger.Error("State transition failed",
			zap.String("component", b.name),
			zap.String("from", oldState.String()),
			zap.String("to", newState.String()),
			zap.Error(err))
		return err
	}

	b.logger.Info("State transition",
		zap.String("component", b.name),
		zap.String("from", oldState.String()),
		zap.String("to", newState.String()))

	// Notify observers
	b.notifyStateChange(oldState, newState)

	return nil
}

// Init provides default initialization
// Components should override this with their specific logic
func (b *BaseLifecycle) Init(ctx context.Context, config interface{}) error {
	if err := b.TransitionTo(StateInitialized); err != nil {
		return fmt.Errorf("failed to transition to initialized: %w", err)
	}
	return nil
}

// Start provides default start
// Components should override this with their specific logic
func (b *BaseLifecycle) Start(ctx context.Context) error {
	if err := b.TransitionTo(StateStarting); err != nil {
		return fmt.Errorf("failed to transition to starting: %w", err)
	}

	b.startupMutex.Lock()
	b.startTime = time.Now()
	b.startupMutex.Unlock()

	if err := b.TransitionTo(StateRunning); err != nil {
		return fmt.Errorf("failed to transition to running: %w", err)
	}

	return nil
}

// Stop provides default stop
// Components should override this with their specific logic
func (b *BaseLifecycle) Stop(ctx context.Context) error {
	if err := b.TransitionTo(StateStopping); err != nil {
		return fmt.Errorf("failed to transition to stopping: %w", err)
	}

	b.startupMutex.Lock()
	b.stopTime = time.Now()
	b.startupMutex.Unlock()

	if err := b.TransitionTo(StateStopped); err != nil {
		return fmt.Errorf("failed to transition to stopped: %w", err)
	}

	return nil
}

// Shutdown provides default forceful shutdown with timeout
func (b *BaseLifecycle) Shutdown(ctx context.Context, timeout time.Duration) error {
	// Create timeout context
	shutdownCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	// Try graceful stop first
	stopErr := b.Stop(shutdownCtx)
	if stopErr == nil {
		return nil
	}

	// Graceful stop failed, force shutdown
	b.logger.Warn("Graceful shutdown failed, forcing",
		zap.String("component", b.name),
		zap.Error(stopErr))

	// Force state to stopped
	if err := b.stateMachine.TransitionTo(StateStopped); err != nil {
		// If we can't even force stop, transition to failed
		_ = b.stateMachine.TransitionTo(StateFailed)
		return fmt.Errorf("forced shutdown failed: %w", err)
	}

	return nil
}

// HealthCheck provides default health check
// Components should override with specific health logic
func (b *BaseLifecycle) HealthCheck(ctx context.Context) error {
	state := b.GetState()

	if !state.IsOperational() {
		err := fmt.Errorf("component not operational, state: %s", state.String())
		b.recordHealthCheck(err)
		return err
	}

	b.recordHealthCheck(nil)
	return nil
}

// recordHealthCheck records health check result
func (b *BaseLifecycle) recordHealthCheck(err error) {
	b.healthMutex.Lock()
	defer b.healthMutex.Unlock()

	b.lastHealthCheck = time.Now()
	b.lastHealthCheckResult = err

	if err != nil {
		b.notifyHealthCheckFailed(err)
	}
}

// GetLastHealthCheck returns last health check time and result
func (b *BaseLifecycle) GetLastHealthCheck() (time.Time, error) {
	b.healthMutex.RLock()
	defer b.healthMutex.RUnlock()
	return b.lastHealthCheck, b.lastHealthCheckResult
}

// GetUptime returns how long component has been running
func (b *BaseLifecycle) GetUptime() time.Duration {
	b.startupMutex.RLock()
	defer b.startupMutex.RUnlock()

	if b.startTime.IsZero() {
		return 0
	}

	if !b.stopTime.IsZero() {
		return b.stopTime.Sub(b.startTime)
	}

	return time.Since(b.startTime)
}

// GetStartupDuration returns how long startup took
func (b *BaseLifecycle) GetStartupDuration() time.Duration {
	b.startupMutex.RLock()
	defer b.startupMutex.RUnlock()

	if b.startTime.IsZero() {
		return 0
	}

	// If still starting or not yet running, return time so far
	state := b.GetState()
	if state == StateStarting || state == StateInitialized {
		return time.Since(b.startTime)
	}

	// If running, return time from start to running
	if state == StateRunning {
		// This is approximate since we don't track exact transition time
		return time.Since(b.startTime)
	}

	return 0
}

// GetMetrics returns component metrics
func (b *BaseLifecycle) GetMetrics() ComponentMetrics {
	b.healthMutex.RLock()
	lastHealthCheck := b.lastHealthCheck
	healthCheckSuccess := b.lastHealthCheckResult == nil
	b.healthMutex.RUnlock()

	stateMetrics := b.stateMachine.GetStateMetrics()

	return ComponentMetrics{
		ComponentName:       b.name,
		State:               b.GetState(),
		Uptime:              b.GetUptime(),
		StateTransitions:    int64(stateMetrics.TransitionCount),
		FailureCount:        int64(stateMetrics.FailedTransitions),
		LastHealthCheck:     lastHealthCheck,
		HealthCheckSuccess:  healthCheckSuccess,
		StartupDuration:     b.GetStartupDuration(),
		CustomMetrics:       make(map[string]interface{}),
	}
}

// SetShutdownConfig sets shutdown configuration
func (b *BaseLifecycle) SetShutdownConfig(config ShutdownConfig) {
	b.shutdownConfig = config
}

// GetShutdownConfig returns shutdown configuration
func (b *BaseLifecycle) GetShutdownConfig() ShutdownConfig {
	return b.shutdownConfig
}

// RegisterObserver registers an observer for lifecycle events
func (b *BaseLifecycle) RegisterObserver(observer Observer) {
	b.observersMutex.Lock()
	defer b.observersMutex.Unlock()
	b.observers = append(b.observers, observer)
	b.stateMachine.RegisterObserver(observer)
}

// UnregisterObserver removes an observer
func (b *BaseLifecycle) UnregisterObserver(observer Observer) {
	b.observersMutex.Lock()
	defer b.observersMutex.Unlock()

	for i, obs := range b.observers {
		if obs == observer {
			b.observers = append(b.observers[:i], b.observers[i+1:]...)
			break
		}
	}

	b.stateMachine.UnregisterObserver(observer)
}

// notifyStateChange notifies observers of state change
func (b *BaseLifecycle) notifyStateChange(oldState, newState State) {
	b.observersMutex.RLock()
	observers := make([]Observer, len(b.observers))
	copy(observers, b.observers)
	b.observersMutex.RUnlock()

	for _, observer := range observers {
		go observer.OnStateChange(b.name, oldState, newState)
	}
}

// notifyHealthCheckFailed notifies observers of health check failure
func (b *BaseLifecycle) notifyHealthCheckFailed(err error) {
	b.observersMutex.RLock()
	observers := make([]Observer, len(b.observers))
	copy(observers, b.observers)
	b.observersMutex.RUnlock()

	for _, observer := range observers {
		go observer.OnHealthCheckFailed(b.name, err)
	}
}

// WaitForState waits for component to reach desired state
func (b *BaseLifecycle) WaitForState(ctx context.Context, desiredState State, timeout time.Duration) error {
	return b.stateMachine.WaitForState(ctx, desiredState, timeout)
}

// IsHealthy returns true if last health check passed
func (b *BaseLifecycle) IsHealthy() bool {
	b.healthMutex.RLock()
	defer b.healthMutex.RUnlock()
	return b.lastHealthCheckResult == nil && b.GetState().IsOperational()
}

// GetStateMachine returns the underlying state machine
func (b *BaseLifecycle) GetStateMachine() *StateMachine {
	return b.stateMachine
}
