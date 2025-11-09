package resilience

import (
	"context"
	"errors"
	"sync"
	"time"

	"go.uber.org/zap"
)

var (
	// ErrTimeout is returned when an operation times out
	ErrTimeout = errors.New("operation timeout")
)

// TimeoutManager manages operation timeouts
type TimeoutManager struct {
	name           string
	defaultTimeout time.Duration
	timeouts       map[string]time.Duration
	logger         *zap.Logger
	mu             sync.RWMutex

	// Adaptive timeout tracking
	latencyHistory map[string][]time.Duration
	historySize    int
}

// NewTimeoutManager creates a new timeout manager
func NewTimeoutManager(name string, defaultTimeout time.Duration, logger *zap.Logger) *TimeoutManager {
	if logger == nil {
		logger = zap.NewNop()
	}

	return &TimeoutManager{
		name:           name,
		defaultTimeout: defaultTimeout,
		timeouts:       make(map[string]time.Duration),
		latencyHistory: make(map[string][]time.Duration),
		historySize:    100,
		logger:         logger,
	}
}

// ExecuteWithTimeout runs a function with a timeout
func (tm *TimeoutManager) ExecuteWithTimeout(name string, fn func(context.Context) error) error {
	timeout := tm.getTimeout(name)
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	return tm.executeWithContext(ctx, name, fn)
}

// ExecuteWithContext runs a function with a context
func (tm *TimeoutManager) ExecuteWithContext(ctx context.Context, name string, fn func(context.Context) error) error {
	timeout := tm.getTimeout(name)
	timeoutCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	return tm.executeWithContext(timeoutCtx, name, fn)
}

// executeWithContext executes function and tracks metrics
func (tm *TimeoutManager) executeWithContext(ctx context.Context, name string, fn func(context.Context) error) error {
	startTime := time.Now()
	errChan := make(chan error, 1)

	go func() {
		errChan <- fn(ctx)
	}()

	select {
	case err := <-errChan:
		latency := time.Since(startTime)
		tm.recordLatency(name, latency, err == nil)
		return err
	case <-ctx.Done():
		tm.logger.Debug("Operation timed out",
			zap.String("manager", tm.name),
			zap.String("operation", name),
			zap.Duration("timeout", tm.getTimeout(name)))
		return ErrTimeout
	}
}

// SetTimeout sets a specific timeout for an operation
func (tm *TimeoutManager) SetTimeout(name string, timeout time.Duration) {
	tm.mu.Lock()
	defer tm.mu.Unlock()
	tm.timeouts[name] = timeout
	tm.logger.Info("Timeout updated",
		zap.String("manager", tm.name),
		zap.String("operation", name),
		zap.Duration("timeout", timeout))
}

// GetTimeout gets the timeout for an operation
func (tm *TimeoutManager) GetTimeout(name string) time.Duration {
	return tm.getTimeout(name)
}

// getTimeout internal method to get timeout
func (tm *TimeoutManager) getTimeout(name string) time.Duration {
	tm.mu.RLock()
	defer tm.mu.RUnlock()

	if timeout, exists := tm.timeouts[name]; exists {
		return timeout
	}
	return tm.defaultTimeout
}

// AdaptTimeout adjusts timeout based on observed latency
func (tm *TimeoutManager) AdaptTimeout(name string, latency time.Duration, success bool) {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	current := tm.timeouts[name]
	if current == 0 {
		current = tm.defaultTimeout
	}

	// Simple adaptation: increase on failure, decrease on consistent success
	var newTimeout time.Duration
	if success {
		// On success, slowly reduce timeout (95% of current)
		newTimeout = time.Duration(float64(current) * 0.95)

		// But not less than 2x the observed latency
		minTimeout := latency * 2
		if newTimeout < minTimeout {
			newTimeout = minTimeout
		}
	} else {
		// On failure, increase timeout (110% of current)
		newTimeout = time.Duration(float64(current) * 1.10)
	}

	// Ensure reasonable bounds (10ms to 5 minutes)
	if newTimeout < 10*time.Millisecond {
		newTimeout = 10 * time.Millisecond
	} else if newTimeout > 5*time.Minute {
		newTimeout = 5 * time.Minute
	}

	tm.timeouts[name] = newTimeout
	tm.logger.Debug("Adapted timeout",
		zap.String("manager", tm.name),
		zap.String("operation", name),
		zap.Duration("oldTimeout", current),
		zap.Duration("newTimeout", newTimeout),
		zap.Bool("success", success))
}

// recordLatency records latency for adaptive timeout calculation
func (tm *TimeoutManager) recordLatency(name string, latency time.Duration, success bool) {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	history := tm.latencyHistory[name]
	history = append(history, latency)

	// Keep only recent history
	if len(history) > tm.historySize {
		history = history[len(history)-tm.historySize:]
	}

	tm.latencyHistory[name] = history

	// Adapt timeout if we have enough samples
	if len(history) >= 10 {
		tm.adaptTimeoutFromHistory(name, success)
	}
}

// adaptTimeoutFromHistory adapts timeout based on latency history
func (tm *TimeoutManager) adaptTimeoutFromHistory(name string, success bool) {
	history := tm.latencyHistory[name]
	if len(history) == 0 {
		return
	}

	// Calculate P95 latency
	p95 := tm.calculatePercentile(history, 95)

	// Set timeout to 2x P95 latency
	desiredTimeout := p95 * 2

	current := tm.timeouts[name]
	if current == 0 {
		current = tm.defaultTimeout
	}

	// Gradually move toward desired timeout
	var newTimeout time.Duration
	if desiredTimeout > current {
		// Increase timeout quickly on high latency
		newTimeout = current + (desiredTimeout-current)/2
	} else {
		// Decrease timeout slowly on low latency
		newTimeout = current - (current-desiredTimeout)/10
	}

	// Apply bounds
	if newTimeout < 10*time.Millisecond {
		newTimeout = 10 * time.Millisecond
	} else if newTimeout > 5*time.Minute {
		newTimeout = 5 * time.Minute
	}

	tm.timeouts[name] = newTimeout
}

// calculatePercentile calculates percentile from latency history
func (tm *TimeoutManager) calculatePercentile(latencies []time.Duration, percentile int) time.Duration {
	if len(latencies) == 0 {
		return 0
	}

	// Copy and sort
	sorted := make([]time.Duration, len(latencies))
	copy(sorted, latencies)

	// Simple bubble sort for small arrays
	for i := 0; i < len(sorted)-1; i++ {
		for j := 0; j < len(sorted)-i-1; j++ {
			if sorted[j] > sorted[j+1] {
				sorted[j], sorted[j+1] = sorted[j+1], sorted[j]
			}
		}
	}

	index := (len(sorted) * percentile) / 100
	if index >= len(sorted) {
		index = len(sorted) - 1
	}

	return sorted[index]
}

// GetMetrics returns timeout manager metrics
func (tm *TimeoutManager) GetMetrics() TimeoutMetrics {
	tm.mu.RLock()
	defer tm.mu.RUnlock()

	operations := make(map[string]OperationTimeout)
	for name, timeout := range tm.timeouts {
		avgLatency := time.Duration(0)
		if history := tm.latencyHistory[name]; len(history) > 0 {
			var sum time.Duration
			for _, l := range history {
				sum += l
			}
			avgLatency = sum / time.Duration(len(history))
		}

		operations[name] = OperationTimeout{
			Name:           name,
			CurrentTimeout: timeout,
			AverageLatency: avgLatency,
			SampleCount:    len(tm.latencyHistory[name]),
		}
	}

	return TimeoutMetrics{
		Name:           tm.name,
		DefaultTimeout: tm.defaultTimeout,
		Operations:     operations,
	}
}

// DeadlineManager manages deadlines for complex operations
type DeadlineManager struct {
	logger *zap.Logger
}

// NewDeadlineManager creates a new deadline manager
func NewDeadlineManager(logger *zap.Logger) *DeadlineManager {
	if logger == nil {
		logger = zap.NewNop()
	}

	return &DeadlineManager{
		logger: logger,
	}
}

// WithDeadline creates a context with deadline
func (dm *DeadlineManager) WithDeadline(parent context.Context, deadline time.Time) (context.Context, context.CancelFunc) {
	ctx, cancel := context.WithDeadline(parent, deadline)
	dm.logger.Debug("Deadline set",
		zap.Time("deadline", deadline),
		zap.Duration("timeUntil", time.Until(deadline)))
	return ctx, cancel
}

// WithDuration creates a context with duration-based deadline
func (dm *DeadlineManager) WithDuration(parent context.Context, duration time.Duration) (context.Context, context.CancelFunc) {
	ctx, cancel := context.WithTimeout(parent, duration)
	dm.logger.Debug("Duration deadline set",
		zap.Duration("duration", duration))
	return ctx, cancel
}

// CheckDeadline checks if context deadline has been exceeded
func (dm *DeadlineManager) CheckDeadline(ctx context.Context) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
		return nil
	}
}

// RemainingTime returns time remaining until deadline
func (dm *DeadlineManager) RemainingTime(ctx context.Context) time.Duration {
	deadline, ok := ctx.Deadline()
	if !ok {
		return 0
	}
	return time.Until(deadline)
}

// Metrics types

// TimeoutMetrics contains timeout manager metrics
type TimeoutMetrics struct {
	Name           string
	DefaultTimeout time.Duration
	Operations     map[string]OperationTimeout
}

// OperationTimeout contains timeout information for an operation
type OperationTimeout struct {
	Name           string
	CurrentTimeout time.Duration
	AverageLatency time.Duration
	SampleCount    int
}