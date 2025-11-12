package resilience

import (
	"context"
	"errors"
	"sync"
	"time"

	"go.uber.org/zap"
	"golang.org/x/time/rate"
)

var (
	// ErrRateLimitExceeded is returned when rate limit is exceeded
	ErrRateLimitExceeded = errors.New("rate limit exceeded")
)

// RateLimiter provides basic rate limiting functionality
type RateLimiter struct {
	limiter   *rate.Limiter
	burst     int
	perSecond float64
	name      string
	logger    *zap.Logger

	// Metrics
	totalAllowed  int64
	totalRejected int64
	mu            sync.RWMutex
}

// NewRateLimiter creates a new rate limiter
func NewRateLimiter(name string, rps float64, burst int, logger *zap.Logger) *RateLimiter {
	if logger == nil {
		logger = zap.NewNop()
	}

	return &RateLimiter{
		limiter:   rate.NewLimiter(rate.Limit(rps), burst),
		perSecond: rps,
		burst:     burst,
		name:      name,
		logger:    logger,
	}
}

// Allow checks if a request is allowed
func (rl *RateLimiter) Allow() bool {
	allowed := rl.limiter.Allow()

	rl.mu.Lock()
	if allowed {
		rl.totalAllowed++
	} else {
		rl.totalRejected++
		rl.logger.Debug("Rate limit exceeded",
			zap.String("name", rl.name),
			zap.Float64("limit", rl.perSecond))
	}
	rl.mu.Unlock()

	return allowed
}

// Wait waits for permission to proceed
func (rl *RateLimiter) Wait(ctx context.Context) error {
	err := rl.limiter.Wait(ctx)

	rl.mu.Lock()
	if err == nil {
		rl.totalAllowed++
	} else {
		rl.totalRejected++
	}
	rl.mu.Unlock()

	return err
}

// Reserve reserves tokens for future use
func (rl *RateLimiter) Reserve() *rate.Reservation {
	return rl.limiter.Reserve()
}

// GetMetrics returns rate limiter metrics
func (rl *RateLimiter) GetMetrics() RateLimiterMetrics {
	rl.mu.RLock()
	defer rl.mu.RUnlock()

	return RateLimiterMetrics{
		Name:          rl.name,
		RatePerSecond: rl.perSecond,
		Burst:         rl.burst,
		TotalAllowed:  rl.totalAllowed,
		TotalRejected: rl.totalRejected,
	}
}

// AdaptiveRateLimiter adjusts rate based on system performance
type AdaptiveRateLimiter struct {
	limiter       *rate.Limiter
	targetLatency time.Duration
	latencyWindow []time.Duration
	windowSize    int
	minRate       float64
	maxRate       float64
	currentRate   float64
	name          string
	logger        *zap.Logger
	mu            sync.Mutex

	// Metrics
	adjustments   int64
	totalAllowed  int64
	totalRejected int64
}

// NewAdaptiveRateLimiter creates a new adaptive rate limiter
func NewAdaptiveRateLimiter(name string, initialRate, minRate, maxRate float64, targetLatency time.Duration, windowSize int, logger *zap.Logger) *AdaptiveRateLimiter {
	if logger == nil {
		logger = zap.NewNop()
	}

	return &AdaptiveRateLimiter{
		limiter:       rate.NewLimiter(rate.Limit(initialRate), int(initialRate)),
		targetLatency: targetLatency,
		latencyWindow: make([]time.Duration, 0, windowSize),
		windowSize:    windowSize,
		minRate:       minRate,
		maxRate:       maxRate,
		currentRate:   initialRate,
		name:          name,
		logger:        logger,
	}
}

// Allow checks if a request is allowed
func (arl *AdaptiveRateLimiter) Allow() bool {
	allowed := arl.limiter.Allow()

	arl.mu.Lock()
	if allowed {
		arl.totalAllowed++
	} else {
		arl.totalRejected++
	}
	arl.mu.Unlock()

	return allowed
}

// Wait waits for permission to proceed
func (arl *AdaptiveRateLimiter) Wait(ctx context.Context) error {
	err := arl.limiter.Wait(ctx)

	arl.mu.Lock()
	if err == nil {
		arl.totalAllowed++
	} else {
		arl.totalRejected++
	}
	arl.mu.Unlock()

	return err
}

// RecordLatency records a latency measurement and adjusts rate
func (arl *AdaptiveRateLimiter) RecordLatency(latency time.Duration) {
	arl.mu.Lock()
	defer arl.mu.Unlock()

	// Add to window
	arl.latencyWindow = append(arl.latencyWindow, latency)
	if len(arl.latencyWindow) > arl.windowSize {
		arl.latencyWindow = arl.latencyWindow[1:]
	}

	// Need enough samples before adjusting
	if len(arl.latencyWindow) < arl.windowSize/2 {
		return
	}

	avgLatency := arl.averageLatency()
	p95Latency := arl.percentileLatency(95)

	// Adjust rate based on P95 latency
	var newRate float64
	if p95Latency > arl.targetLatency*120/100 {
		// Latency too high, reduce rate by 10%
		newRate = arl.currentRate * 0.9
	} else if p95Latency < arl.targetLatency*80/100 && avgLatency < arl.targetLatency {
		// Latency good, increase rate by 5%
		newRate = arl.currentRate * 1.05
	} else {
		// No change
		return
	}

	// Apply bounds
	if newRate < arl.minRate {
		newRate = arl.minRate
	} else if newRate > arl.maxRate {
		newRate = arl.maxRate
	}

	if newRate != arl.currentRate {
		arl.currentRate = newRate
		arl.limiter.SetLimit(rate.Limit(newRate))
		arl.limiter.SetBurst(int(newRate))
		arl.adjustments++

		arl.logger.Info("Adaptive rate limiter adjusted",
			zap.String("name", arl.name),
			zap.Float64("newRate", newRate),
			zap.Duration("avgLatency", avgLatency),
			zap.Duration("p95Latency", p95Latency))
	}
}

// GetMetrics returns adaptive rate limiter metrics
func (arl *AdaptiveRateLimiter) GetMetrics() AdaptiveRateLimiterMetrics {
	arl.mu.Lock()
	defer arl.mu.Unlock()

	avgLatency := time.Duration(0)
	if len(arl.latencyWindow) > 0 {
		avgLatency = arl.averageLatency()
	}

	return AdaptiveRateLimiterMetrics{
		Name:           arl.name,
		CurrentRate:    arl.currentRate,
		MinRate:        arl.minRate,
		MaxRate:        arl.maxRate,
		TargetLatency:  arl.targetLatency,
		AverageLatency: avgLatency,
		Adjustments:    arl.adjustments,
		TotalAllowed:   arl.totalAllowed,
		TotalRejected:  arl.totalRejected,
	}
}

// Private methods

func (arl *AdaptiveRateLimiter) averageLatency() time.Duration {
	if len(arl.latencyWindow) == 0 {
		return 0
	}

	var sum time.Duration
	for _, l := range arl.latencyWindow {
		sum += l
	}
	return sum / time.Duration(len(arl.latencyWindow))
}

func (arl *AdaptiveRateLimiter) percentileLatency(percentile int) time.Duration {
	if len(arl.latencyWindow) == 0 {
		return 0
	}

	// Simple percentile calculation (not exact for small samples)
	sorted := make([]time.Duration, len(arl.latencyWindow))
	copy(sorted, arl.latencyWindow)

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

// TokenBucketRateLimiter implements token bucket algorithm
type TokenBucketRateLimiter struct {
	capacity     int
	tokens       int
	refillRate   int
	refillPeriod time.Duration
	lastRefill   time.Time
	name         string
	logger       *zap.Logger
	mu           sync.Mutex
}

// NewTokenBucketRateLimiter creates a new token bucket rate limiter
func NewTokenBucketRateLimiter(name string, capacity, refillRate int, refillPeriod time.Duration, logger *zap.Logger) *TokenBucketRateLimiter {
	if logger == nil {
		logger = zap.NewNop()
	}

	return &TokenBucketRateLimiter{
		capacity:     capacity,
		tokens:       capacity,
		refillRate:   refillRate,
		refillPeriod: refillPeriod,
		lastRefill:   time.Now(),
		name:         name,
		logger:       logger,
	}
}

// Allow checks if tokens are available
func (tb *TokenBucketRateLimiter) Allow(tokens int) bool {
	tb.mu.Lock()
	defer tb.mu.Unlock()

	tb.refill()

	if tb.tokens >= tokens {
		tb.tokens -= tokens
		return true
	}

	return false
}

// refill adds tokens based on elapsed time
func (tb *TokenBucketRateLimiter) refill() {
	now := time.Now()
	elapsed := now.Sub(tb.lastRefill)

	if elapsed >= tb.refillPeriod {
		periods := int(elapsed / tb.refillPeriod)
		tokensToAdd := periods * tb.refillRate

		tb.tokens += tokensToAdd
		if tb.tokens > tb.capacity {
			tb.tokens = tb.capacity
		}

		tb.lastRefill = now
	}
}

// Metrics types

// RateLimiterMetrics contains rate limiter metrics
type RateLimiterMetrics struct {
	Name          string
	RatePerSecond float64
	Burst         int
	TotalAllowed  int64
	TotalRejected int64
}

// AdaptiveRateLimiterMetrics contains adaptive rate limiter metrics
type AdaptiveRateLimiterMetrics struct {
	Name           string
	CurrentRate    float64
	MinRate        float64
	MaxRate        float64
	TargetLatency  time.Duration
	AverageLatency time.Duration
	Adjustments    int64
	TotalAllowed   int64
	TotalRejected  int64
}