package lifecycle

import (
	"context"
	"time"
)

// ComponentLifecycle defines the standard lifecycle interface for all DWCP components
type ComponentLifecycle interface {
	// Init initializes the component with configuration
	// This method prepares the component but does not start it
	// Transitions: Uninitialized -> Initialized
	Init(ctx context.Context, config interface{}) error

	// Start starts the component and transitions to Running state
	// This method should be idempotent
	// Transitions: Initialized -> Starting -> Running
	Start(ctx context.Context) error

	// Stop gracefully stops the component
	// This method should wait for in-flight operations to complete
	// Transitions: Running -> Stopping -> Stopped
	Stop(ctx context.Context) error

	// Shutdown forcefully shuts down the component with timeout
	// This method is called when Stop() exceeds the timeout
	// Transitions: Any -> Stopped
	Shutdown(ctx context.Context, timeout time.Duration) error

	// HealthCheck verifies component health and returns error if unhealthy
	// This method should be fast (< 100ms for shallow checks)
	HealthCheck(ctx context.Context) error

	// GetState returns the current lifecycle state
	GetState() State

	// GetName returns the component name for logging and identification
	GetName() string

	// GetDependencies returns list of component names this component depends on
	// Dependencies are started before this component and stopped after
	GetDependencies() []string
}

// HealthLevel defines the depth of health checking
type HealthLevel int

const (
	// HealthLevelShallow performs basic health checks (< 100ms)
	HealthLevelShallow HealthLevel = iota

	// HealthLevelDeep performs comprehensive health checks (< 1s)
	HealthLevelDeep

	// HealthLevelFull performs exhaustive health checks (< 5s)
	HealthLevelFull
)

// HealthCheckable extends ComponentLifecycle with advanced health checking
type HealthCheckable interface {
	ComponentLifecycle

	// HealthCheckWithLevel performs health check at specified level
	HealthCheckWithLevel(ctx context.Context, level HealthLevel) error
}

// Recoverable extends ComponentLifecycle with recovery capabilities
type Recoverable interface {
	ComponentLifecycle

	// Recover attempts to recover from a failed state
	Recover(ctx context.Context) error

	// GetRecoveryStrategy returns the recovery strategy for this component
	GetRecoveryStrategy() RecoveryStrategy
}

// RecoveryStrategy defines how a component should be recovered
type RecoveryStrategy struct {
	// MaxRetries is the maximum number of recovery attempts
	MaxRetries int

	// RetryBackoff is the base backoff duration between retries
	RetryBackoff time.Duration

	// ExponentialBackoff enables exponential backoff
	ExponentialBackoff bool

	// MaxBackoff is the maximum backoff duration
	MaxBackoff time.Duration

	// FailFast causes component to fail without retry on certain errors
	FailFast bool
}

// DefaultRecoveryStrategy returns a sensible default recovery strategy
func DefaultRecoveryStrategy() RecoveryStrategy {
	return RecoveryStrategy{
		MaxRetries:         3,
		RetryBackoff:       1 * time.Second,
		ExponentialBackoff: true,
		MaxBackoff:         30 * time.Second,
		FailFast:           false,
	}
}

// Configurable extends ComponentLifecycle with dynamic configuration
type Configurable interface {
	ComponentLifecycle

	// UpdateConfig updates component configuration
	// Component must be stopped to update configuration
	UpdateConfig(ctx context.Context, config interface{}) error

	// GetConfig returns current configuration
	GetConfig() interface{}

	// ValidateConfig validates configuration without applying it
	ValidateConfig(config interface{}) error
}

// MetricsProvider extends ComponentLifecycle with metrics collection
type MetricsProvider interface {
	ComponentLifecycle

	// GetMetrics returns current component metrics
	GetMetrics() ComponentMetrics
}

// ComponentMetrics defines standard metrics for components
type ComponentMetrics struct {
	// ComponentName is the name of the component
	ComponentName string

	// State is the current lifecycle state
	State State

	// Uptime is how long component has been running
	Uptime time.Duration

	// StateTransitions is count of state transitions
	StateTransitions int64

	// FailureCount is count of failures
	FailureCount int64

	// RecoveryCount is count of successful recoveries
	RecoveryCount int64

	// LastHealthCheck is timestamp of last health check
	LastHealthCheck time.Time

	// HealthCheckSuccess indicates if last health check passed
	HealthCheckSuccess bool

	// HealthCheckDuration is duration of last health check
	HealthCheckDuration time.Duration

	// StartupDuration is how long startup took
	StartupDuration time.Duration

	// ShutdownDuration is how long shutdown took
	ShutdownDuration time.Duration

	// CustomMetrics are component-specific metrics
	CustomMetrics map[string]interface{}
}

// Observer watches component lifecycle events
type Observer interface {
	// OnStateChange is called when component state changes
	OnStateChange(component string, oldState, newState State)

	// OnHealthCheckFailed is called when health check fails
	OnHealthCheckFailed(component string, err error)

	// OnRecoveryStarted is called when recovery begins
	OnRecoveryStarted(component string)

	// OnRecoveryCompleted is called when recovery succeeds
	OnRecoveryCompleted(component string, duration time.Duration)

	// OnRecoveryFailed is called when recovery fails
	OnRecoveryFailed(component string, err error)
}

// Observable allows components to notify observers of lifecycle events
type Observable interface {
	// RegisterObserver adds an observer
	RegisterObserver(observer Observer)

	// UnregisterObserver removes an observer
	UnregisterObserver(observer Observer)

	// NotifyObservers notifies all registered observers
	NotifyObservers(event Event)
}

// Event represents a lifecycle event
type Event struct {
	// Type is the event type
	Type EventType

	// Component is the component name
	Component string

	// Timestamp is when event occurred
	Timestamp time.Time

	// OldState is the previous state (for state changes)
	OldState State

	// NewState is the new state (for state changes)
	NewState State

	// Error is any associated error
	Error error

	// Metadata contains additional event data
	Metadata map[string]interface{}
}

// EventType defines lifecycle event types
type EventType int

const (
	// EventStateChange indicates a state transition
	EventStateChange EventType = iota

	// EventHealthCheckFailed indicates health check failure
	EventHealthCheckFailed

	// EventRecoveryStarted indicates recovery beginning
	EventRecoveryStarted

	// EventRecoveryCompleted indicates successful recovery
	EventRecoveryCompleted

	// EventRecoveryFailed indicates failed recovery
	EventRecoveryFailed

	// EventConfigUpdated indicates configuration change
	EventConfigUpdated

	// EventShutdownTimeout indicates shutdown timeout exceeded
	EventShutdownTimeout
)

// String returns string representation of event type
func (e EventType) String() string {
	switch e {
	case EventStateChange:
		return "state_change"
	case EventHealthCheckFailed:
		return "health_check_failed"
	case EventRecoveryStarted:
		return "recovery_started"
	case EventRecoveryCompleted:
		return "recovery_completed"
	case EventRecoveryFailed:
		return "recovery_failed"
	case EventConfigUpdated:
		return "config_updated"
	case EventShutdownTimeout:
		return "shutdown_timeout"
	default:
		return "unknown"
	}
}

// ShutdownConfig defines shutdown behavior
type ShutdownConfig struct {
	// DrainTimeout is how long to wait for in-flight operations
	DrainTimeout time.Duration

	// CleanupTimeout is how long to wait for resource cleanup
	CleanupTimeout time.Duration

	// ForceTimeout is when to forcefully terminate
	ForceTimeout time.Duration

	// TotalTimeout is the overall shutdown timeout
	TotalTimeout time.Duration
}

// DefaultShutdownConfig returns default shutdown configuration
func DefaultShutdownConfig() ShutdownConfig {
	return ShutdownConfig{
		DrainTimeout:   30 * time.Second,
		CleanupTimeout: 10 * time.Second,
		ForceTimeout:   5 * time.Second,
		TotalTimeout:   45 * time.Second,
	}
}
