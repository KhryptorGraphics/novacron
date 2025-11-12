package init

import (
	"context"
	"fmt"
	"time"
)

// WithRetry executes a function with retry logic
func WithRetry(ctx context.Context, policy RetryPolicy, fn func() error) error {
	var lastErr error
	delay := policy.Delay

	for attempt := 1; attempt <= policy.MaxAttempts; attempt++ {
		// Execute the function
		if err := fn(); err != nil {
			lastErr = err

			// If this is not the last attempt, wait and retry
			if attempt < policy.MaxAttempts {
				select {
				case <-time.After(delay):
					// Apply exponential backoff
					delay = time.Duration(float64(delay) * policy.Backoff)
				case <-ctx.Done():
					return ctx.Err()
				}
				continue
			}
		} else {
			// Success
			return nil
		}
	}

	return fmt.Errorf("max retries exceeded (%d attempts): %w", policy.MaxAttempts, lastErr)
}

// RetryableError wraps an error with retry information
type RetryableError struct {
	Err       error
	Retriable bool
	Delay     time.Duration
}

func (e *RetryableError) Error() string {
	return e.Err.Error()
}

func (e *RetryableError) Unwrap() error {
	return e.Err
}

// IsRetriable checks if an error is retriable
func IsRetriable(err error) bool {
	if err == nil {
		return false
	}

	if re, ok := err.(*RetryableError); ok {
		return re.Retriable
	}

	// Check for init errors
	if ie, ok := err.(*InitError); ok {
		return ie.Retriable
	}

	return false
}

// NewRetriableError creates a new retriable error
func NewRetriableError(err error, delay time.Duration) error {
	return &RetryableError{
		Err:       err,
		Retriable: true,
		Delay:     delay,
	}
}

// NewNonRetriableError creates a new non-retriable error
func NewNonRetriableError(err error) error {
	return &RetryableError{
		Err:       err,
		Retriable: false,
	}
}
