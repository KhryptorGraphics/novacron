// Package recovery provides error handling and recovery mechanisms for NovaCron
package recovery

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// RecoveryPolicy defines how to handle failures
type RecoveryPolicy struct {
	MaxRetries       int           // Maximum number of retry attempts
	RetryDelay       time.Duration // Delay between retries
	BackoffMultiplier float64       // Exponential backoff multiplier
	MaxBackoff       time.Duration // Maximum backoff duration
	EnableRollback   bool          // Enable automatic rollback on failure
	RollbackTimeout  time.Duration // Timeout for rollback operations
}

// DefaultRecoveryPolicy returns a sensible default policy
func DefaultRecoveryPolicy() RecoveryPolicy {
	return RecoveryPolicy{
		MaxRetries:       3,
		RetryDelay:       1 * time.Second,
		BackoffMultiplier: 2.0,
		MaxBackoff:       30 * time.Second,
		EnableRollback:   true,
		RollbackTimeout:  60 * time.Second,
	}
}

// RecoveryManager manages error recovery and rollback
type RecoveryManager struct {
	mu             sync.RWMutex
	policy         RecoveryPolicy
	checkpoints    map[string]*Checkpoint
	rollbackFuncs  []RollbackFunc
	logger         Logger
}

// Logger interface for logging
type Logger interface {
	Info(msg string, keysAndValues ...interface{})
	Error(msg string, err error, keysAndValues ...interface{})
	Debug(msg string, keysAndValues ...interface{})
	Warn(msg string, keysAndValues ...interface{})
}

// Checkpoint represents a recovery checkpoint
type Checkpoint struct {
	Name      string
	Timestamp time.Time
	State     interface{}
}

// RollbackFunc is a function that performs rollback
type RollbackFunc func(ctx context.Context) error

// NewRecoveryManager creates a new recovery manager
func NewRecoveryManager(policy RecoveryPolicy, logger Logger) *RecoveryManager {
	return &RecoveryManager{
		policy:        policy,
		checkpoints:   make(map[string]*Checkpoint),
		rollbackFuncs: make([]RollbackFunc, 0),
		logger:        logger,
	}
}

// WithRetry executes a function with retry logic
func (rm *RecoveryManager) WithRetry(ctx context.Context, name string, fn func() error) error {
	var lastErr error
	delay := rm.policy.RetryDelay

	for attempt := 0; attempt <= rm.policy.MaxRetries; attempt++ {
		if attempt > 0 {
			rm.logger.Info("Retrying operation",
				"name", name,
				"attempt", attempt,
				"max_retries", rm.policy.MaxRetries,
				"delay", delay,
			)

			select {
			case <-ctx.Done():
				return fmt.Errorf("operation cancelled: %w", ctx.Err())
			case <-time.After(delay):
			}

			// Calculate next backoff
			delay = time.Duration(float64(delay) * rm.policy.BackoffMultiplier)
			if delay > rm.policy.MaxBackoff {
				delay = rm.policy.MaxBackoff
			}
		}

		err := fn()
		if err == nil {
			if attempt > 0 {
				rm.logger.Info("Operation succeeded after retry", "name", name, "attempts", attempt+1)
			}
			return nil
		}

		lastErr = err
		rm.logger.Warn("Operation failed", "name", name, "attempt", attempt+1, "error", err.Error())
	}

	return fmt.Errorf("operation failed after %d attempts: %w", rm.policy.MaxRetries+1, lastErr)
}

// SaveCheckpoint saves a recovery checkpoint
func (rm *RecoveryManager) SaveCheckpoint(name string, state interface{}) error {
	rm.mu.Lock()
	defer rm.mu.Unlock()

	rm.checkpoints[name] = &Checkpoint{
		Name:      name,
		Timestamp: time.Now(),
		State:     state,
	}

	rm.logger.Debug("Checkpoint saved", "name", name)
	return nil
}

// GetCheckpoint retrieves a saved checkpoint
func (rm *RecoveryManager) GetCheckpoint(name string) (*Checkpoint, error) {
	rm.mu.RLock()
	defer rm.mu.RUnlock()

	checkpoint, exists := rm.checkpoints[name]
	if !exists {
		return nil, fmt.Errorf("checkpoint not found: %s", name)
	}

	return checkpoint, nil
}

// RegisterRollback registers a rollback function
func (rm *RecoveryManager) RegisterRollback(fn RollbackFunc) {
	rm.mu.Lock()
	defer rm.mu.Unlock()

	rm.rollbackFuncs = append(rm.rollbackFuncs, fn)
	rm.logger.Debug("Rollback function registered", "count", len(rm.rollbackFuncs))
}

// Rollback executes all registered rollback functions in reverse order
func (rm *RecoveryManager) Rollback(ctx context.Context) error {
	if !rm.policy.EnableRollback {
		rm.logger.Info("Rollback disabled by policy")
		return nil
	}

	rm.logger.Info("Starting rollback", "functions", len(rm.rollbackFuncs))

	// Create timeout context
	rollbackCtx, cancel := context.WithTimeout(ctx, rm.policy.RollbackTimeout)
	defer cancel()

	var errors []error

	// Execute rollback functions in reverse order
	rm.mu.RLock()
	funcs := make([]RollbackFunc, len(rm.rollbackFuncs))
	copy(funcs, rm.rollbackFuncs)
	rm.mu.RUnlock()

	for i := len(funcs) - 1; i >= 0; i-- {
		fn := funcs[i]

		rm.logger.Debug("Executing rollback function", "index", i)

		if err := fn(rollbackCtx); err != nil {
			rm.logger.Error("Rollback function failed", err, "index", i)
			errors = append(errors, fmt.Errorf("rollback %d: %w", i, err))
		}
	}

	if len(errors) > 0 {
		return fmt.Errorf("rollback errors: %v", errors)
	}

	rm.logger.Info("Rollback completed successfully")
	return nil
}

// ClearCheckpoints removes all saved checkpoints
func (rm *RecoveryManager) ClearCheckpoints() {
	rm.mu.Lock()
	defer rm.mu.Unlock()

	rm.checkpoints = make(map[string]*Checkpoint)
	rm.logger.Debug("All checkpoints cleared")
}

// ClearRollbackFuncs removes all registered rollback functions
func (rm *RecoveryManager) ClearRollbackFuncs() {
	rm.mu.Lock()
	defer rm.mu.Unlock()

	rm.rollbackFuncs = make([]RollbackFunc, 0)
	rm.logger.Debug("All rollback functions cleared")
}

// SafeExecute executes a function with panic recovery
func (rm *RecoveryManager) SafeExecute(name string, fn func() error) (err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("panic recovered in %s: %v", name, r)
			rm.logger.Error("Panic recovered", err, "name", name)
		}
	}()

	return fn()
}

// AttemptRecovery attempts to recover from a specific error
func (rm *RecoveryManager) AttemptRecovery(ctx context.Context, err error, recoveryFn func(context.Context) error) error {
	rm.logger.Info("Attempting recovery", "error", err.Error())

	recoveryCtx, cancel := context.WithTimeout(ctx, rm.policy.RollbackTimeout)
	defer cancel()

	if recoveryErr := recoveryFn(recoveryCtx); recoveryErr != nil {
		rm.logger.Error("Recovery failed", recoveryErr)
		return fmt.Errorf("recovery failed: %w (original error: %v)", recoveryErr, err)
	}

	rm.logger.Info("Recovery successful")
	return nil
}

// HealthCheck performs a health check with recovery
func (rm *RecoveryManager) HealthCheck(ctx context.Context, checkFn func(context.Context) error) error {
	return rm.WithRetry(ctx, "health-check", func() error {
		return checkFn(ctx)
	})
}

// WatchAndRecover watches for errors and attempts recovery
type WatchAndRecoverConfig struct {
	CheckInterval  time.Duration
	HealthCheckFn  func(context.Context) error
	RecoveryFn     func(context.Context, error) error
	MaxFailures    int
	FailureWindow  time.Duration
}

// WatchAndRecover continuously monitors health and attempts recovery on failure
func (rm *RecoveryManager) WatchAndRecover(ctx context.Context, config WatchAndRecoverConfig) error {
	ticker := time.NewTicker(config.CheckInterval)
	defer ticker.Stop()

	failureCount := 0
	firstFailure := time.Time{}

	for {
		select {
		case <-ctx.Done():
			rm.logger.Info("Health monitoring stopped")
			return ctx.Err()

		case <-ticker.C:
			err := config.HealthCheckFn(ctx)
			if err == nil {
				// Reset failure count on success
				if failureCount > 0 {
					rm.logger.Info("Health check recovered", "previous_failures", failureCount)
					failureCount = 0
				}
				continue
			}

			// Track failures
			now := time.Now()
			if firstFailure.IsZero() || now.Sub(firstFailure) > config.FailureWindow {
				firstFailure = now
				failureCount = 1
			} else {
				failureCount++
			}

			rm.logger.Warn("Health check failed",
				"error", err.Error(),
				"failure_count", failureCount,
				"max_failures", config.MaxFailures,
			)

			// Attempt recovery if threshold reached
			if failureCount >= config.MaxFailures {
				rm.logger.Error("Maximum failures reached, attempting recovery", err,
					"failures", failureCount,
				)

				if recoveryErr := config.RecoveryFn(ctx, err); recoveryErr != nil {
					rm.logger.Error("Recovery failed", recoveryErr)
					return fmt.Errorf("recovery failed after %d failures: %w", failureCount, recoveryErr)
				}

				rm.logger.Info("Recovery successful, resetting failure count")
				failureCount = 0
				firstFailure = time.Time{}
			}
		}
	}
}

// CircuitBreaker implements circuit breaker pattern
type CircuitBreaker struct {
	mu              sync.RWMutex
	failures        int
	lastFailure     time.Time
	state           CircuitState
	maxFailures     int
	resetTimeout    time.Duration
	halfOpenTimeout time.Duration
	logger          Logger
}

// CircuitState represents circuit breaker state
type CircuitState int

const (
	StateClosed CircuitState = iota
	StateOpen
	StateHalfOpen
)

// NewCircuitBreaker creates a new circuit breaker
func NewCircuitBreaker(maxFailures int, resetTimeout time.Duration, logger Logger) *CircuitBreaker {
	return &CircuitBreaker{
		maxFailures:     maxFailures,
		resetTimeout:    resetTimeout,
		halfOpenTimeout: resetTimeout / 2,
		state:           StateClosed,
		logger:          logger,
	}
}

// Execute executes a function through the circuit breaker
func (cb *CircuitBreaker) Execute(fn func() error) error {
	cb.mu.Lock()

	// Check if circuit should transition from open to half-open
	if cb.state == StateOpen && time.Since(cb.lastFailure) > cb.resetTimeout {
		cb.logger.Info("Circuit breaker transitioning to half-open")
		cb.state = StateHalfOpen
	}

	// Fail fast if circuit is open
	if cb.state == StateOpen {
		cb.mu.Unlock()
		return fmt.Errorf("circuit breaker is open")
	}

	cb.mu.Unlock()

	// Execute function
	err := fn()

	cb.mu.Lock()
	defer cb.mu.Unlock()

	if err != nil {
		cb.failures++
		cb.lastFailure = time.Now()

		if cb.failures >= cb.maxFailures {
			cb.logger.Warn("Circuit breaker opening", "failures", cb.failures)
			cb.state = StateOpen
		}

		return err
	}

	// Success - reset on success in half-open state
	if cb.state == StateHalfOpen {
		cb.logger.Info("Circuit breaker closing after successful half-open attempt")
		cb.failures = 0
		cb.state = StateClosed
	}

	return nil
}

// GetState returns current circuit breaker state
func (cb *CircuitBreaker) GetState() CircuitState {
	cb.mu.RLock()
	defer cb.mu.RUnlock()
	return cb.state
}

// Reset manually resets the circuit breaker
func (cb *CircuitBreaker) Reset() {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	cb.failures = 0
	cb.state = StateClosed
	cb.logger.Info("Circuit breaker manually reset")
}
