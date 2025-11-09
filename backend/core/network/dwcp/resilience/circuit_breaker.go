package resilience

import (
	"errors"
	"sync"
	"time"

	"go.uber.org/zap"
)

var (
	// ErrCircuitBreakerOpen is returned when the circuit breaker is open
	ErrCircuitBreakerOpen = errors.New("circuit breaker is open")
)

// State represents the circuit breaker state
type State int

const (
	// StateClosed allows requests to pass through
	StateClosed State = iota
	// StateHalfOpen allows limited requests for testing
	StateHalfOpen
	// StateOpen blocks all requests
	StateOpen
)

// String returns the string representation of the state
func (s State) String() string {
	switch s {
	case StateClosed:
		return "closed"
	case StateHalfOpen:
		return "half-open"
	case StateOpen:
		return "open"
	default:
		return "unknown"
	}
}

// CircuitBreaker implements the circuit breaker pattern
type CircuitBreaker struct {
	name           string
	maxFailures    int
	timeout        time.Duration
	resetTimeout   time.Duration
	state          State
	failures       int
	lastFailTime   time.Time
	successCount   int
	halfOpenMax    int
	mu             sync.RWMutex
	logger         *zap.Logger

	// Metrics
	totalRequests  int64
	totalFailures  int64
	totalSuccesses int64
	stateChanges   int64
}

// NewCircuitBreaker creates a new circuit breaker
func NewCircuitBreaker(name string, maxFailures int, timeout, resetTimeout time.Duration, logger *zap.Logger) *CircuitBreaker {
	if logger == nil {
		logger = zap.NewNop()
	}

	return &CircuitBreaker{
		name:         name,
		maxFailures:  maxFailures,
		timeout:      timeout,
		resetTimeout: resetTimeout,
		state:        StateClosed,
		halfOpenMax:  3, // Allow 3 test requests in half-open state
		logger:       logger,
	}
}

// Execute runs the provided function with circuit breaker protection
func (cb *CircuitBreaker) Execute(fn func() error) error {
	cb.mu.Lock()
	cb.totalRequests++

	switch cb.state {
	case StateOpen:
		if time.Since(cb.lastFailTime) > cb.resetTimeout {
			// Transition to half-open state
			cb.transitionToHalfOpen()
		} else {
			cb.mu.Unlock()
			cb.logger.Debug("Circuit breaker rejecting request",
				zap.String("name", cb.name),
				zap.String("state", cb.state.String()))
			return ErrCircuitBreakerOpen
		}

	case StateHalfOpen:
		if cb.successCount >= cb.halfOpenMax {
			// Received enough successful requests, close the circuit
			cb.transitionToClosed()
		}
	}

	currentState := cb.state
	cb.mu.Unlock()

	// Execute the function with timeout
	type result struct {
		err error
	}

	done := make(chan result, 1)
	go func() {
		done <- result{err: fn()}
	}()

	var res result
	select {
	case res = <-done:
		// Function completed
	case <-time.After(cb.timeout):
		res.err = errors.New("circuit breaker timeout")
	}

	// Handle the result
	cb.mu.Lock()
	defer cb.mu.Unlock()

	if res.err != nil {
		cb.recordFailure()
		cb.totalFailures++

		if cb.failures >= cb.maxFailures && cb.state == StateClosed {
			cb.transitionToOpen()
		} else if cb.state == StateHalfOpen {
			// Failed in half-open state, reopen immediately
			cb.transitionToOpen()
		}

		return res.err
	}

	// Success
	cb.recordSuccess()
	cb.totalSuccesses++

	if currentState == StateHalfOpen {
		cb.successCount++
		if cb.successCount >= cb.halfOpenMax {
			cb.transitionToClosed()
		}
	}

	return nil
}

// GetState returns the current state of the circuit breaker
func (cb *CircuitBreaker) GetState() State {
	cb.mu.RLock()
	defer cb.mu.RUnlock()
	return cb.state
}

// GetMetrics returns circuit breaker metrics
func (cb *CircuitBreaker) GetMetrics() CircuitBreakerMetrics {
	cb.mu.RLock()
	defer cb.mu.RUnlock()

	successRate := float64(0)
	if cb.totalRequests > 0 {
		successRate = float64(cb.totalSuccesses) / float64(cb.totalRequests)
	}

	return CircuitBreakerMetrics{
		Name:           cb.name,
		State:          cb.state.String(),
		TotalRequests:  cb.totalRequests,
		TotalFailures:  cb.totalFailures,
		TotalSuccesses: cb.totalSuccesses,
		SuccessRate:    successRate,
		StateChanges:   cb.stateChanges,
		CurrentFailures: cb.failures,
	}
}

// Reset manually resets the circuit breaker
func (cb *CircuitBreaker) Reset() {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	cb.state = StateClosed
	cb.failures = 0
	cb.successCount = 0
	cb.lastFailTime = time.Time{}
	cb.stateChanges++

	cb.logger.Info("Circuit breaker manually reset",
		zap.String("name", cb.name))
}

// Private methods

func (cb *CircuitBreaker) transitionToOpen() {
	cb.state = StateOpen
	cb.lastFailTime = time.Now()
	cb.stateChanges++

	cb.logger.Warn("Circuit breaker opened",
		zap.String("name", cb.name),
		zap.Int("failures", cb.failures),
		zap.Time("lastFailure", cb.lastFailTime))
}

func (cb *CircuitBreaker) transitionToHalfOpen() {
	cb.state = StateHalfOpen
	cb.failures = 0
	cb.successCount = 0
	cb.stateChanges++

	cb.logger.Info("Circuit breaker transitioned to half-open",
		zap.String("name", cb.name))
}

func (cb *CircuitBreaker) transitionToClosed() {
	cb.state = StateClosed
	cb.failures = 0
	cb.successCount = 0
	cb.stateChanges++

	cb.logger.Info("Circuit breaker closed",
		zap.String("name", cb.name))
}

func (cb *CircuitBreaker) recordFailure() {
	cb.failures++
	cb.lastFailTime = time.Now()
}

func (cb *CircuitBreaker) recordSuccess() {
	// Reset consecutive failures on success in closed state
	if cb.state == StateClosed {
		cb.failures = 0
	}
}

// CircuitBreakerMetrics contains circuit breaker metrics
type CircuitBreakerMetrics struct {
	Name            string
	State           string
	TotalRequests   int64
	TotalFailures   int64
	TotalSuccesses  int64
	SuccessRate     float64
	StateChanges    int64
	CurrentFailures int
}