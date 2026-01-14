package quotas

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// EnforcementEngine handles quota enforcement with performance optimization
type EnforcementEngine struct {
	// Enforcement cache for fast lookups
	cache       *EnforcementCache
	
	// Rate limiters for different resource types
	rateLimiters map[ResourceType]*RateLimiter
	
	// Circuit breakers for enforcement actions
	circuitBreakers map[EnforcementAction]*CircuitBreaker
	
	// Metrics for enforcement performance
	metrics *EnforcementMetrics
	
	// Configuration
	config *EnforcementConfig
	
	// Manager reference
	manager *Manager
	
	// Synchronization
	mu sync.RWMutex
}

// EnforcementCache provides fast quota lookup and enforcement decisions
type EnforcementCache struct {
	// Cached quota decisions
	decisions map[string]*CachedDecision
	
	// Cache TTL
	ttl time.Duration
	
	// Synchronization
	mu sync.RWMutex
	
	// Background cleanup
	ctx    context.Context
	cancel context.CancelFunc
}

// CachedDecision represents a cached enforcement decision
type CachedDecision struct {
	EntityID        string          `json:"entity_id"`
	ResourceType    ResourceType    `json:"resource_type"`
	Allowed         bool            `json:"allowed"`
	AvailableAmount int64           `json:"available_amount"`
	QuotaID         string          `json:"quota_id"`
	Timestamp       time.Time       `json:"timestamp"`
	ExpiresAt       time.Time       `json:"expires_at"`
}

// RateLimiter implements token bucket rate limiting for resource requests
type RateLimiter struct {
	resourceType ResourceType
	capacity     int64
	tokens       int64
	refillRate   int64 // tokens per second
	lastRefill   time.Time
	mu           sync.Mutex
}

// CircuitBreaker prevents cascading failures in enforcement
type CircuitBreaker struct {
	action       EnforcementAction
	state        CircuitState
	failureCount int64
	successCount int64
	lastFailure  time.Time
	config       *CircuitBreakerConfig
	mu           sync.Mutex
}

// CircuitState represents circuit breaker states
type CircuitState string

const (
	CircuitStateClosed   CircuitState = "closed"   // Normal operation
	CircuitStateOpen     CircuitState = "open"     // Failing, reject requests
	CircuitStateHalfOpen CircuitState = "half_open" // Testing recovery
)

// CircuitBreakerConfig holds circuit breaker configuration
type CircuitBreakerConfig struct {
	FailureThreshold int64         `json:"failure_threshold"`
	RecoveryTimeout  time.Duration `json:"recovery_timeout"`
	SuccessThreshold int64         `json:"success_threshold"`
}

// EnforcementConfig holds enforcement engine configuration
type EnforcementConfig struct {
	// Cache configuration
	CacheEnabled bool          `json:"cache_enabled"`
	CacheTTL     time.Duration `json:"cache_ttl"`
	
	// Rate limiting configuration
	RateLimitEnabled bool                             `json:"rate_limit_enabled"`
	RateLimits       map[ResourceType]*RateLimitConfig `json:"rate_limits"`
	
	// Circuit breaker configuration
	CircuitBreakerEnabled bool                                    `json:"circuit_breaker_enabled"`
	CircuitBreakerConfigs map[EnforcementAction]*CircuitBreakerConfig `json:"circuit_breaker_configs"`
	
	// Performance optimization
	BatchSize          int           `json:"batch_size"`
	MaxConcurrentChecks int          `json:"max_concurrent_checks"`
	CheckTimeout       time.Duration `json:"check_timeout"`
	
	// Metrics configuration
	MetricsEnabled bool `json:"metrics_enabled"`
}

// RateLimitConfig holds rate limiter configuration
type RateLimitConfig struct {
	Capacity   int64 `json:"capacity"`
	RefillRate int64 `json:"refill_rate"` // tokens per second
}

// EnforcementMetrics tracks enforcement performance
type EnforcementMetrics struct {
	// Request counts
	TotalRequests      int64 `json:"total_requests"`
	AllowedRequests    int64 `json:"allowed_requests"`
	DeniedRequests     int64 `json:"denied_requests"`
	CacheHits          int64 `json:"cache_hits"`
	CacheMisses        int64 `json:"cache_misses"`
	
	// Performance metrics
	AverageCheckTime   time.Duration `json:"average_check_time"`
	MaxCheckTime       time.Duration `json:"max_check_time"`
	
	// Error metrics
	EnforcementErrors  int64 `json:"enforcement_errors"`
	CircuitBreakerTrips int64 `json:"circuit_breaker_trips"`
	RateLimitHits      int64 `json:"rate_limit_hits"`
	
	// Synchronization
	mu sync.RWMutex
}

// DefaultEnforcementConfig returns a default enforcement configuration
func DefaultEnforcementConfig() *EnforcementConfig {
	return &EnforcementConfig{
		CacheEnabled: true,
		CacheTTL:     30 * time.Second,
		RateLimitEnabled: true,
		RateLimits: map[ResourceType]*RateLimitConfig{
			ResourceTypeCPU: {
				Capacity:   1000,
				RefillRate: 100,
			},
			ResourceTypeMemory: {
				Capacity:   1000,
				RefillRate: 100,
			},
			ResourceTypeStorage: {
				Capacity:   500,
				RefillRate: 50,
			},
		},
		CircuitBreakerEnabled: true,
		CircuitBreakerConfigs: map[EnforcementAction]*CircuitBreakerConfig{
			EnforcementActionDeny: {
				FailureThreshold: 5,
				RecoveryTimeout:  30 * time.Second,
				SuccessThreshold: 3,
			},
			EnforcementActionScale: {
				FailureThreshold: 3,
				RecoveryTimeout:  60 * time.Second,
				SuccessThreshold: 2,
			},
		},
		BatchSize:           100,
		MaxConcurrentChecks: 50,
		CheckTimeout:        5 * time.Second,
		MetricsEnabled:      true,
	}
}

// NewEnforcementEngine creates a new enforcement engine
func NewEnforcementEngine(manager *Manager, config *EnforcementConfig) *EnforcementEngine {
	if config == nil {
		config = DefaultEnforcementConfig()
	}

	engine := &EnforcementEngine{
		rateLimiters:    make(map[ResourceType]*RateLimiter),
		circuitBreakers: make(map[EnforcementAction]*CircuitBreaker),
		metrics:         &EnforcementMetrics{},
		config:          config,
		manager:         manager,
	}

	// Initialize cache if enabled
	if config.CacheEnabled {
		engine.cache = NewEnforcementCache(config.CacheTTL)
	}

	// Initialize rate limiters
	if config.RateLimitEnabled {
		for resourceType, rateLimitConfig := range config.RateLimits {
			engine.rateLimiters[resourceType] = &RateLimiter{
				resourceType: resourceType,
				capacity:     rateLimitConfig.Capacity,
				tokens:       rateLimitConfig.Capacity,
				refillRate:   rateLimitConfig.RefillRate,
				lastRefill:   time.Now(),
			}
		}
	}

	// Initialize circuit breakers
	if config.CircuitBreakerEnabled {
		for action, cbConfig := range config.CircuitBreakerConfigs {
			engine.circuitBreakers[action] = &CircuitBreaker{
				action: action,
				state:  CircuitStateClosed,
				config: cbConfig,
			}
		}
	}

	return engine
}

// NewEnforcementCache creates a new enforcement cache
func NewEnforcementCache(ttl time.Duration) *EnforcementCache {
	ctx, cancel := context.WithCancel(context.Background())
	
	cache := &EnforcementCache{
		decisions: make(map[string]*CachedDecision),
		ttl:       ttl,
		ctx:       ctx,
		cancel:    cancel,
	}

	// Start cleanup goroutine
	go cache.cleanupLoop()

	return cache
}

// EnforceQuota performs quota enforcement with optimization
func (e *EnforcementEngine) EnforceQuota(ctx context.Context, entityID string, resourceType ResourceType, amount int64) (*QuotaCheckResult, error) {
	startTime := time.Now()
	defer func() {
		duration := time.Since(startTime)
		e.updateMetrics(duration)
	}()

	e.metrics.mu.Lock()
	e.metrics.TotalRequests++
	e.metrics.mu.Unlock()

	// Check rate limiter first
	if e.config.RateLimitEnabled {
		if !e.checkRateLimit(resourceType) {
			e.metrics.mu.Lock()
			e.metrics.RateLimitHits++
			e.metrics.DeniedRequests++
			e.metrics.mu.Unlock()
			
			return &QuotaCheckResult{
				Allowed: false,
				Reason:  "rate limit exceeded",
			}, nil
		}
	}

	// Check cache if enabled
	if e.config.CacheEnabled {
		if result := e.checkCache(entityID, resourceType, amount); result != nil {
			e.metrics.mu.Lock()
			e.metrics.CacheHits++
			if result.Allowed {
				e.metrics.AllowedRequests++
			} else {
				e.metrics.DeniedRequests++
			}
			e.metrics.mu.Unlock()
			return result, nil
		}
		e.metrics.mu.Lock()
		e.metrics.CacheMisses++
		e.metrics.mu.Unlock()
	}

	// Perform actual quota check
	result, err := e.manager.CheckQuota(ctx, entityID, resourceType, amount)
	if err != nil {
		e.metrics.mu.Lock()
		e.metrics.EnforcementErrors++
		e.metrics.mu.Unlock()
		return nil, err
	}

	// Cache the result if enabled
	if e.config.CacheEnabled {
		e.cacheDecision(entityID, resourceType, result)
	}

	// Update metrics
	e.metrics.mu.Lock()
	if result.Allowed {
		e.metrics.AllowedRequests++
	} else {
		e.metrics.DeniedRequests++
	}
	e.metrics.mu.Unlock()

	return result, nil
}

// BatchEnforceQuota performs batch quota enforcement for better performance
func (e *EnforcementEngine) BatchEnforceQuota(ctx context.Context, requests []QuotaRequest) ([]QuotaCheckResult, error) {
	results := make([]QuotaCheckResult, len(requests))
	
	// Process in batches
	batchSize := e.config.BatchSize
	for i := 0; i < len(requests); i += batchSize {
		end := i + batchSize
		if end > len(requests) {
			end = len(requests)
		}

		batch := requests[i:end]
		batchResults, err := e.processBatch(ctx, batch)
		if err != nil {
			return nil, err
		}

		copy(results[i:end], batchResults)
	}

	return results, nil
}

// QuotaRequest represents a quota enforcement request
type QuotaRequest struct {
	EntityID     string       `json:"entity_id"`
	ResourceType ResourceType `json:"resource_type"`
	Amount       int64        `json:"amount"`
}

// processBatch processes a batch of quota requests
func (e *EnforcementEngine) processBatch(ctx context.Context, requests []QuotaRequest) ([]QuotaCheckResult, error) {
	results := make([]QuotaCheckResult, len(requests))
	
	// Use semaphore to limit concurrent checks
	semaphore := make(chan struct{}, e.config.MaxConcurrentChecks)
	
	// Process requests concurrently
	var wg sync.WaitGroup
	for i, request := range requests {
		wg.Add(1)
		go func(idx int, req QuotaRequest) {
			defer wg.Done()
			
			semaphore <- struct{}{} // Acquire
			defer func() { <-semaphore }() // Release
			
			result, err := e.EnforceQuota(ctx, req.EntityID, req.ResourceType, req.Amount)
			if err != nil {
				results[idx] = QuotaCheckResult{
					Allowed: false,
					Reason:  fmt.Sprintf("enforcement error: %v", err),
				}
			} else {
				results[idx] = *result
			}
		}(i, request)
	}
	
	wg.Wait()
	return results, nil
}

// Rate limiter methods

func (e *EnforcementEngine) checkRateLimit(resourceType ResourceType) bool {
	rateLimiter, exists := e.rateLimiters[resourceType]
	if !exists {
		return true // No rate limit configured
	}

	return rateLimiter.Allow()
}

func (r *RateLimiter) Allow() bool {
	r.mu.Lock()
	defer r.mu.Unlock()

	now := time.Now()
	elapsed := now.Sub(r.lastRefill).Seconds()
	
	// Refill tokens
	tokensToAdd := int64(elapsed * float64(r.refillRate))
	r.tokens = min(r.capacity, r.tokens+tokensToAdd)
	r.lastRefill = now

	if r.tokens > 0 {
		r.tokens--
		return true
	}

	return false
}

// Cache methods

func (e *EnforcementEngine) checkCache(entityID string, resourceType ResourceType, amount int64) *QuotaCheckResult {
	if e.cache == nil {
		return nil
	}

	decision := e.cache.Get(entityID, resourceType)
	if decision == nil || decision.ExpiresAt.Before(time.Now()) {
		return nil
	}

	// Check if cached decision is still valid for the requested amount
	if decision.Allowed && amount <= decision.AvailableAmount {
		return &QuotaCheckResult{
			Allowed:   true,
			Available: decision.AvailableAmount,
		}
	} else if !decision.Allowed {
		return &QuotaCheckResult{
			Allowed: false,
			Reason:  "cached denial",
		}
	}

	return nil
}

func (e *EnforcementEngine) cacheDecision(entityID string, resourceType ResourceType, result *QuotaCheckResult) {
	if e.cache == nil {
		return
	}

	decision := &CachedDecision{
		EntityID:        entityID,
		ResourceType:    resourceType,
		Allowed:         result.Allowed,
		AvailableAmount: result.Available,
		Timestamp:       time.Now(),
		ExpiresAt:       time.Now().Add(e.config.CacheTTL),
	}

	if result.Quota != nil {
		decision.QuotaID = result.Quota.ID
	}

	e.cache.Set(entityID, resourceType, decision)
}

func (c *EnforcementCache) Get(entityID string, resourceType ResourceType) *CachedDecision {
	c.mu.RLock()
	defer c.mu.RUnlock()

	key := fmt.Sprintf("%s:%s", entityID, resourceType)
	return c.decisions[key]
}

func (c *EnforcementCache) Set(entityID string, resourceType ResourceType, decision *CachedDecision) {
	c.mu.Lock()
	defer c.mu.Unlock()

	key := fmt.Sprintf("%s:%s", entityID, resourceType)
	c.decisions[key] = decision
}

func (c *EnforcementCache) cleanupLoop() {
	ticker := time.NewTicker(c.ttl / 2) // Cleanup twice per TTL period
	defer ticker.Stop()

	for {
		select {
		case <-c.ctx.Done():
			return
		case <-ticker.C:
			c.cleanup()
		}
	}
}

func (c *EnforcementCache) cleanup() {
	c.mu.Lock()
	defer c.mu.Unlock()

	now := time.Now()
	for key, decision := range c.decisions {
		if decision.ExpiresAt.Before(now) {
			delete(c.decisions, key)
		}
	}
}

// Circuit breaker methods

func (e *EnforcementEngine) ExecuteWithCircuitBreaker(action EnforcementAction, fn func() error) error {
	breaker, exists := e.circuitBreakers[action]
	if !exists {
		return fn() // No circuit breaker configured
	}

	return breaker.Execute(fn)
}

func (cb *CircuitBreaker) Execute(fn func() error) error {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	switch cb.state {
	case CircuitStateOpen:
		if time.Since(cb.lastFailure) < cb.config.RecoveryTimeout {
			return fmt.Errorf("circuit breaker is open for action: %s", cb.action)
		}
		cb.state = CircuitStateHalfOpen
		cb.successCount = 0
		cb.failureCount = 0

	case CircuitStateHalfOpen:
		// Limited requests allowed in half-open state
		if cb.successCount >= cb.config.SuccessThreshold {
			cb.state = CircuitStateClosed
			cb.failureCount = 0
		}
	}

	// Execute the function
	err := fn()

	if err != nil {
		cb.failureCount++
		cb.lastFailure = time.Now()

		if cb.failureCount >= cb.config.FailureThreshold {
			cb.state = CircuitStateOpen
		}

		return err
	}

	cb.successCount++
	if cb.state == CircuitStateClosed {
		cb.failureCount = 0 // Reset on success in closed state
	}

	return nil
}

// Metrics methods

func (e *EnforcementEngine) updateMetrics(duration time.Duration) {
	e.metrics.mu.Lock()
	defer e.metrics.mu.Unlock()

	// Update average check time (simple moving average)
	if e.metrics.AverageCheckTime == 0 {
		e.metrics.AverageCheckTime = duration
	} else {
		e.metrics.AverageCheckTime = (e.metrics.AverageCheckTime + duration) / 2
	}

	// Update max check time
	if duration > e.metrics.MaxCheckTime {
		e.metrics.MaxCheckTime = duration
	}
}

func (e *EnforcementEngine) GetMetrics() *EnforcementMetrics {
	e.metrics.mu.RLock()
	defer e.metrics.mu.RUnlock()

	// Return a copy
	metrics := *e.metrics
	return &metrics
}

func (e *EnforcementEngine) ResetMetrics() {
	e.metrics.mu.Lock()
	defer e.metrics.mu.Unlock()

	e.metrics.TotalRequests = 0
	e.metrics.AllowedRequests = 0
	e.metrics.DeniedRequests = 0
	e.metrics.CacheHits = 0
	e.metrics.CacheMisses = 0
	e.metrics.AverageCheckTime = 0
	e.metrics.MaxCheckTime = 0
	e.metrics.EnforcementErrors = 0
	e.metrics.CircuitBreakerTrips = 0
	e.metrics.RateLimitHits = 0
}

// Stop gracefully stops the enforcement engine
func (e *EnforcementEngine) Stop() {
	if e.cache != nil {
		e.cache.cancel()
	}
}

// Utility functions

func min(a, b int64) int64 {
	if a < b {
		return a
	}
	return b
}