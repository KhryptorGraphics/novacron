package resilience

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"time"

	"go.uber.org/zap"
)

// RetryPolicy defines retry behavior
type RetryPolicy struct {
	name         string
	maxRetries   int
	initialDelay time.Duration
	maxDelay     time.Duration
	multiplier   float64
	jitter       bool
	logger       *zap.Logger

	// Retry conditions
	retryableErrors []error
	retryableCheck  func(error) bool

	// Metrics
	totalAttempts int64
	totalSuccess  int64
	totalFailures int64
}

// NewRetryPolicy creates a new retry policy
func NewRetryPolicy(name string, maxRetries int, initialDelay, maxDelay time.Duration, multiplier float64, jitter bool, logger *zap.Logger) *RetryPolicy {
	if logger == nil {
		logger = zap.NewNop()
	}

	return &RetryPolicy{
		name:         name,
		maxRetries:   maxRetries,
		initialDelay: initialDelay,
		maxDelay:     maxDelay,
		multiplier:   multiplier,
		jitter:       jitter,
		logger:       logger,
	}
}

// SetRetryableErrors sets specific errors that should trigger retry
func (rp *RetryPolicy) SetRetryableErrors(errors ...error) {
	rp.retryableErrors = errors
}

// SetRetryableCheck sets a custom function to determine if an error is retryable
func (rp *RetryPolicy) SetRetryableCheck(check func(error) bool) {
	rp.retryableCheck = check
}

// Execute runs a function with retry policy
func (rp *RetryPolicy) Execute(fn func() error) error {
	return rp.ExecuteWithContext(context.Background(), fn)
}

// ExecuteWithContext runs a function with retry policy and context
func (rp *RetryPolicy) ExecuteWithContext(ctx context.Context, fn func() error) error {
	var lastErr error
	delay := rp.initialDelay

	for attempt := 0; attempt <= rp.maxRetries; attempt++ {
		rp.totalAttempts++

		// Check context before attempt
		select {
		case <-ctx.Done():
			rp.totalFailures++
			return fmt.Errorf("retry cancelled: %w", ctx.Err())
		default:
		}

		// Execute function
		err := fn()
		if err == nil {
			rp.totalSuccess++
			if attempt > 0 {
				rp.logger.Info("Retry succeeded",
					zap.String("policy", rp.name),
					zap.Int("attempt", attempt+1))
			}
			return nil
		}

		lastErr = err

		// Check if this is the last attempt
		if attempt == rp.maxRetries {
			rp.totalFailures++
			break
		}

		// Check if error is retryable
		if !rp.isRetryable(err) {
			rp.totalFailures++
			rp.logger.Debug("Error is not retryable",
				zap.String("policy", rp.name),
				zap.Error(err))
			return err
		}

		// Calculate backoff delay
		actualDelay := rp.calculateDelay(delay, attempt)

		rp.logger.Debug("Retrying after delay",
			zap.String("policy", rp.name),
			zap.Int("attempt", attempt+1),
			zap.Duration("delay", actualDelay),
			zap.Error(err))

		// Wait with context
		timer := time.NewTimer(actualDelay)
		select {
		case <-timer.C:
		case <-ctx.Done():
			timer.Stop()
			rp.totalFailures++
			return fmt.Errorf("retry cancelled during backoff: %w", ctx.Err())
		}

		// Update delay for next iteration
		delay = time.Duration(float64(delay) * rp.multiplier)
		if delay > rp.maxDelay {
			delay = rp.maxDelay
		}
	}

	return fmt.Errorf("max retries (%d) exceeded: %w", rp.maxRetries, lastErr)
}

// calculateDelay calculates the actual delay with jitter if enabled
func (rp *RetryPolicy) calculateDelay(baseDelay time.Duration, attempt int) time.Duration {
	if !rp.jitter {
		return baseDelay
	}

	// Add random jitter (0-10% of base delay)
	jitterAmount := time.Duration(rand.Float64() * float64(baseDelay) * 0.1)
	return baseDelay + jitterAmount
}

// isRetryable determines if an error should trigger a retry
func (rp *RetryPolicy) isRetryable(err error) bool {
	// Check custom function first
	if rp.retryableCheck != nil {
		return rp.retryableCheck(err)
	}

	// Check specific error list
	for _, retryableErr := range rp.retryableErrors {
		if errors.Is(err, retryableErr) {
			return true
		}
	}

	// Default: retry all errors
	return true
}

// GetMetrics returns retry policy metrics
func (rp *RetryPolicy) GetMetrics() RetryPolicyMetrics {
	successRate := float64(0)
	if rp.totalAttempts > 0 {
		successRate = float64(rp.totalSuccess) / float64(rp.totalAttempts)
	}

	return RetryPolicyMetrics{
		Name:          rp.name,
		TotalAttempts: rp.totalAttempts,
		TotalSuccess:  rp.totalSuccess,
		TotalFailures: rp.totalFailures,
		SuccessRate:   successRate,
		MaxRetries:    rp.maxRetries,
	}
}

// ExponentialBackoff implements exponential backoff with retry
type ExponentialBackoff struct {
	name           string
	maxRetries     int
	baseDelay      time.Duration
	maxDelay       time.Duration
	jitterFactor   float64
	backoffFactor  float64
	logger         *zap.Logger
}

// NewExponentialBackoff creates a new exponential backoff policy
func NewExponentialBackoff(name string, maxRetries int, baseDelay, maxDelay time.Duration, logger *zap.Logger) *ExponentialBackoff {
	if logger == nil {
		logger = zap.NewNop()
	}

	return &ExponentialBackoff{
		name:          name,
		maxRetries:    maxRetries,
		baseDelay:     baseDelay,
		maxDelay:      maxDelay,
		jitterFactor:  0.1,
		backoffFactor: 2.0,
		logger:        logger,
	}
}

// Execute runs a function with exponential backoff
func (eb *ExponentialBackoff) Execute(fn func() error) error {
	return eb.ExecuteWithContext(context.Background(), fn)
}

// ExecuteWithContext runs a function with exponential backoff and context
func (eb *ExponentialBackoff) ExecuteWithContext(ctx context.Context, fn func() error) error {
	var lastErr error

	for attempt := 0; attempt <= eb.maxRetries; attempt++ {
		// Check context
		select {
		case <-ctx.Done():
			return fmt.Errorf("exponential backoff cancelled: %w", ctx.Err())
		default:
		}

		// Execute function
		err := fn()
		if err == nil {
			if attempt > 0 {
				eb.logger.Info("Exponential backoff succeeded",
					zap.String("name", eb.name),
					zap.Int("attempt", attempt+1))
			}
			return nil
		}

		lastErr = err

		// Check if this is the last attempt
		if attempt == eb.maxRetries {
			break
		}

		// Calculate exponential delay
		delay := eb.calculateExponentialDelay(attempt)

		eb.logger.Debug("Exponential backoff retry",
			zap.String("name", eb.name),
			zap.Int("attempt", attempt+1),
			zap.Duration("delay", delay),
			zap.Error(err))

		// Wait with context
		timer := time.NewTimer(delay)
		select {
		case <-timer.C:
		case <-ctx.Done():
			timer.Stop()
			return fmt.Errorf("exponential backoff cancelled during wait: %w", ctx.Err())
		}
	}

	return fmt.Errorf("exponential backoff max retries (%d) exceeded: %w", eb.maxRetries, lastErr)
}

// calculateExponentialDelay calculates delay with exponential backoff
func (eb *ExponentialBackoff) calculateExponentialDelay(attempt int) time.Duration {
	// Calculate base exponential delay
	baseMultiplier := math.Pow(eb.backoffFactor, float64(attempt))
	delay := time.Duration(float64(eb.baseDelay) * baseMultiplier)

	// Apply max delay cap
	if delay > eb.maxDelay {
		delay = eb.maxDelay
	}

	// Add jitter
	jitter := time.Duration(rand.Float64() * float64(delay) * eb.jitterFactor)
	return delay + jitter
}

// LinearBackoff implements linear backoff with retry
type LinearBackoff struct {
	name       string
	maxRetries int
	baseDelay  time.Duration
	increment  time.Duration
	maxDelay   time.Duration
	jitter     bool
	logger     *zap.Logger
}

// NewLinearBackoff creates a new linear backoff policy
func NewLinearBackoff(name string, maxRetries int, baseDelay, increment, maxDelay time.Duration, jitter bool, logger *zap.Logger) *LinearBackoff {
	if logger == nil {
		logger = zap.NewNop()
	}

	return &LinearBackoff{
		name:       name,
		maxRetries: maxRetries,
		baseDelay:  baseDelay,
		increment:  increment,
		maxDelay:   maxDelay,
		jitter:     jitter,
		logger:     logger,
	}
}

// Execute runs a function with linear backoff
func (lb *LinearBackoff) Execute(fn func() error) error {
	var lastErr error

	for attempt := 0; attempt <= lb.maxRetries; attempt++ {
		err := fn()
		if err == nil {
			if attempt > 0 {
				lb.logger.Info("Linear backoff succeeded",
					zap.String("name", lb.name),
					zap.Int("attempt", attempt+1))
			}
			return nil
		}

		lastErr = err

		if attempt == lb.maxRetries {
			break
		}

		// Calculate linear delay
		delay := lb.baseDelay + (time.Duration(attempt) * lb.increment)
		if delay > lb.maxDelay {
			delay = lb.maxDelay
		}

		// Add jitter if enabled
		if lb.jitter {
			jitter := time.Duration(rand.Float64() * float64(delay) * 0.1)
			delay += jitter
		}

		lb.logger.Debug("Linear backoff retry",
			zap.String("name", lb.name),
			zap.Int("attempt", attempt+1),
			zap.Duration("delay", delay))

		time.Sleep(delay)
	}

	return fmt.Errorf("linear backoff max retries (%d) exceeded: %w", lb.maxRetries, lastErr)
}

// FibonacciBackoff implements Fibonacci backoff with retry
type FibonacciBackoff struct {
	name       string
	maxRetries int
	baseDelay  time.Duration
	maxDelay   time.Duration
	jitter     bool
	logger     *zap.Logger
	fibCache   []int
}

// NewFibonacciBackoff creates a new Fibonacci backoff policy
func NewFibonacciBackoff(name string, maxRetries int, baseDelay, maxDelay time.Duration, jitter bool, logger *zap.Logger) *FibonacciBackoff {
	if logger == nil {
		logger = zap.NewNop()
	}

	fb := &FibonacciBackoff{
		name:       name,
		maxRetries: maxRetries,
		baseDelay:  baseDelay,
		maxDelay:   maxDelay,
		jitter:     jitter,
		logger:     logger,
		fibCache:   []int{0, 1},
	}

	// Pre-compute Fibonacci sequence
	for i := 2; i <= maxRetries+1; i++ {
		fb.fibCache = append(fb.fibCache, fb.fibCache[i-1]+fb.fibCache[i-2])
	}

	return fb
}

// Execute runs a function with Fibonacci backoff
func (fb *FibonacciBackoff) Execute(fn func() error) error {
	var lastErr error

	for attempt := 0; attempt <= fb.maxRetries; attempt++ {
		err := fn()
		if err == nil {
			if attempt > 0 {
				fb.logger.Info("Fibonacci backoff succeeded",
					zap.String("name", fb.name),
					zap.Int("attempt", attempt+1))
			}
			return nil
		}

		lastErr = err

		if attempt == fb.maxRetries {
			break
		}

		// Calculate Fibonacci delay
		fibMultiplier := fb.fibCache[min(attempt+1, len(fb.fibCache)-1)]
		delay := time.Duration(fibMultiplier) * fb.baseDelay

		if delay > fb.maxDelay {
			delay = fb.maxDelay
		}

		// Add jitter if enabled
		if fb.jitter {
			jitter := time.Duration(rand.Float64() * float64(delay) * 0.1)
			delay += jitter
		}

		fb.logger.Debug("Fibonacci backoff retry",
			zap.String("name", fb.name),
			zap.Int("attempt", attempt+1),
			zap.Duration("delay", delay))

		time.Sleep(delay)
	}

	return fmt.Errorf("fibonacci backoff max retries (%d) exceeded: %w", fb.maxRetries, lastErr)
}

// Helper function for min
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// Metrics types

// RetryPolicyMetrics contains retry policy metrics
type RetryPolicyMetrics struct {
	Name          string
	TotalAttempts int64
	TotalSuccess  int64
	TotalFailures int64
	SuccessRate   float64
	MaxRetries    int
}