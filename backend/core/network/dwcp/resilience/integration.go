package resilience

import (
	"context"
	"fmt"
	"time"

	"go.uber.org/zap"
)

// ResilienceManager integrates all resilience patterns
type ResilienceManager struct {
	name string

	// Resilience components
	circuitBreakers      map[string]*CircuitBreaker
	rateLimiters         map[string]*RateLimiter
	adaptiveRateLimiters map[string]*AdaptiveRateLimiter
	bulkheads            map[string]*Bulkhead
	retryPolicies        map[string]*RetryPolicy
	timeoutManager       *TimeoutManager
	healthChecker        *HealthChecker
	chaosMonkey          *ChaosMonkey
	degradationMgr       *DegradationManager
	errorBudgets         map[string]*ErrorBudget
	sloTracker           *SLOTracker
	latencyBudgets       map[string]*LatencyBudget

	logger *zap.Logger
}

// NewResilienceManager creates a comprehensive resilience manager
func NewResilienceManager(name string, logger *zap.Logger) *ResilienceManager {
	if logger == nil {
		logger = zap.NewNop()
	}

	return &ResilienceManager{
		name:                 name,
		circuitBreakers:      make(map[string]*CircuitBreaker),
		rateLimiters:         make(map[string]*RateLimiter),
		adaptiveRateLimiters: make(map[string]*AdaptiveRateLimiter),
		bulkheads:            make(map[string]*Bulkhead),
		retryPolicies:        make(map[string]*RetryPolicy),
		timeoutManager:       NewTimeoutManager(name+"-timeout", 30*time.Second, logger),
		healthChecker:        NewHealthChecker(name+"-health", 10*time.Second, 5*time.Second, logger),
		chaosMonkey:          NewChaosMonkey(name+"-chaos", 0.01, logger), // 1% default
		degradationMgr:       NewDegradationManager(name+"-degradation", logger),
		errorBudgets:         make(map[string]*ErrorBudget),
		sloTracker:           NewSLOTracker(name+"-slo", logger),
		latencyBudgets:       make(map[string]*LatencyBudget),
		logger:               logger,
	}
}

// Circuit Breaker operations

// RegisterCircuitBreaker registers a circuit breaker
func (rm *ResilienceManager) RegisterCircuitBreaker(name string, maxFailures int, timeout, resetTimeout time.Duration) {
	cb := NewCircuitBreaker(name, maxFailures, timeout, resetTimeout, rm.logger)
	rm.circuitBreakers[name] = cb
	rm.logger.Info("Circuit breaker registered",
		zap.String("manager", rm.name),
		zap.String("breaker", name))
}

// ExecuteWithCircuitBreaker executes with circuit breaker protection
func (rm *ResilienceManager) ExecuteWithCircuitBreaker(name string, fn func() error) error {
	cb, exists := rm.circuitBreakers[name]
	if !exists {
		return fmt.Errorf("circuit breaker %s not found", name)
	}
	return cb.Execute(fn)
}

// Rate Limiter operations

// RegisterRateLimiter registers a rate limiter
func (rm *ResilienceManager) RegisterRateLimiter(name string, rps float64, burst int) {
	rl := NewRateLimiter(name, rps, burst, rm.logger)
	rm.rateLimiters[name] = rl
	rm.logger.Info("Rate limiter registered",
		zap.String("manager", rm.name),
		zap.String("limiter", name))
}

// RegisterAdaptiveRateLimiter registers an adaptive rate limiter
func (rm *ResilienceManager) RegisterAdaptiveRateLimiter(name string, initialRate, minRate, maxRate float64, targetLatency time.Duration, windowSize int) {
	arl := NewAdaptiveRateLimiter(name, initialRate, minRate, maxRate, targetLatency, windowSize, rm.logger)
	rm.adaptiveRateLimiters[name] = arl
	rm.logger.Info("Adaptive rate limiter registered",
		zap.String("manager", rm.name),
		zap.String("limiter", name))
}

// CheckRateLimit checks if request is allowed by rate limiter
func (rm *ResilienceManager) CheckRateLimit(name string) bool {
	if rl, exists := rm.rateLimiters[name]; exists {
		return rl.Allow()
	}
	if arl, exists := rm.adaptiveRateLimiters[name]; exists {
		return arl.Allow()
	}
	return true
}

// Bulkhead operations

// RegisterBulkhead registers a bulkhead
func (rm *ResilienceManager) RegisterBulkhead(name string, maxConcurrent, maxQueueSize int, maxWaitTime time.Duration) {
	bh := NewBulkhead(name, maxConcurrent, maxQueueSize, maxWaitTime, rm.logger)
	rm.bulkheads[name] = bh
	rm.logger.Info("Bulkhead registered",
		zap.String("manager", rm.name),
		zap.String("bulkhead", name))
}

// ExecuteWithBulkhead executes with bulkhead protection
func (rm *ResilienceManager) ExecuteWithBulkhead(name string, fn func() error) error {
	bh, exists := rm.bulkheads[name]
	if !exists {
		return fmt.Errorf("bulkhead %s not found", name)
	}
	return bh.Execute(fn)
}

// Retry Policy operations

// RegisterRetryPolicy registers a retry policy
func (rm *ResilienceManager) RegisterRetryPolicy(name string, maxRetries int, initialDelay, maxDelay time.Duration, multiplier float64, jitter bool) {
	rp := NewRetryPolicy(name, maxRetries, initialDelay, maxDelay, multiplier, jitter, rm.logger)
	rm.retryPolicies[name] = rp
	rm.logger.Info("Retry policy registered",
		zap.String("manager", rm.name),
		zap.String("policy", name))
}

// ExecuteWithRetry executes with retry policy
func (rm *ResilienceManager) ExecuteWithRetry(name string, fn func() error) error {
	rp, exists := rm.retryPolicies[name]
	if !exists {
		return fmt.Errorf("retry policy %s not found", name)
	}
	return rp.Execute(fn)
}

// Health Check operations

// RegisterHealthCheck registers a health check
func (rm *ResilienceManager) RegisterHealthCheck(check HealthCheck) {
	rm.healthChecker.RegisterCheck(check)
}

// StartHealthMonitoring starts health monitoring
func (rm *ResilienceManager) StartHealthMonitoring() {
	rm.healthChecker.StartMonitoring()
}

// StopHealthMonitoring stops health monitoring
func (rm *ResilienceManager) StopHealthMonitoring() {
	rm.healthChecker.StopMonitoring()
}

// IsHealthy returns overall health status
func (rm *ResilienceManager) IsHealthy() bool {
	return rm.healthChecker.IsHealthy()
}

// Chaos Engineering operations

// EnableChaos enables chaos monkey
func (rm *ResilienceManager) EnableChaos() {
	rm.chaosMonkey.Enable()
}

// DisableChaos disables chaos monkey
func (rm *ResilienceManager) DisableChaos() {
	rm.chaosMonkey.Disable()
}

// RegisterFault registers a fault injector
func (rm *ResilienceManager) RegisterFault(fault FaultInjector) {
	rm.chaosMonkey.RegisterFault(fault)
}

// MaybeInjectFault potentially injects a fault
func (rm *ResilienceManager) MaybeInjectFault() error {
	return rm.chaosMonkey.MaybeInject()
}

// Degradation operations

// SetComponentDegradation sets degradation level for a component
func (rm *ResilienceManager) SetComponentDegradation(component string, level DegradationLevel) {
	rm.degradationMgr.SetComponentLevel(component, level)
}

// SetDegradationLevelChangeCallback registers a callback for overall degradation level changes.
// This allows external packages (like DWCP) to react to degradation transitions without
// accessing internal fields of the degradation manager.
func (rm *ResilienceManager) SetDegradationLevelChangeCallback(fn func(old, new DegradationLevel)) {
	rm.degradationMgr.SetLevelChangeCallback(fn)
}

// GetDegradationLevel returns current degradation level
func (rm *ResilienceManager) GetDegradationLevel() DegradationLevel {
	return rm.degradationMgr.GetLevel()
}

// ExecuteWithDegradation executes with degradation awareness
func (rm *ResilienceManager) ExecuteWithDegradation(normal, degraded, emergency func() error) error {
	return rm.degradationMgr.Execute(normal, degraded, emergency)
}

// Error Budget operations

// RegisterErrorBudget registers an error budget
func (rm *ResilienceManager) RegisterErrorBudget(name string, slo float64, windowDuration time.Duration) {
	eb := NewErrorBudget(name, slo, windowDuration, rm.logger)
	rm.errorBudgets[name] = eb
	rm.sloTracker.RegisterSLO(name, slo, windowDuration)
	rm.logger.Info("Error budget registered",
		zap.String("manager", rm.name),
		zap.String("budget", name),
		zap.Float64("slo", slo))
}

// RecordRequest records a request outcome for error budget
func (rm *ResilienceManager) RecordRequest(budgetName string, success bool) {
	if eb, exists := rm.errorBudgets[budgetName]; exists {
		eb.RecordRequest(success)
	}
	rm.sloTracker.RecordRequest(budgetName, success)
}

// IsBudgetExhausted checks if error budget is exhausted
func (rm *ResilienceManager) IsBudgetExhausted(name string) bool {
	if eb, exists := rm.errorBudgets[name]; exists {
		return eb.BudgetExhausted()
	}
	return false
}

// ForEachErrorBudget iterates over all registered error budgets.
// This provides controlled external access without exposing internal maps.
func (rm *ResilienceManager) ForEachErrorBudget(fn func(name string, budget *ErrorBudget)) {
	for name, eb := range rm.errorBudgets {
		fn(name, eb)
	}
}

// Latency Budget operations

// RegisterLatencyBudget registers a latency budget
func (rm *ResilienceManager) RegisterLatencyBudget(name string, targetLatency time.Duration, percentile float64, maxSamples int) {
	lb := NewLatencyBudget(name, targetLatency, percentile, maxSamples, rm.logger)
	rm.latencyBudgets[name] = lb
	rm.logger.Info("Latency budget registered",
		zap.String("manager", rm.name),
		zap.String("budget", name),
		zap.Duration("target", targetLatency))
}

// RecordLatency records latency for latency budget
func (rm *ResilienceManager) RecordLatency(budgetName string, latency time.Duration) {
	if lb, exists := rm.latencyBudgets[budgetName]; exists {
		lb.RecordLatency(latency)
	}

	// Also update adaptive rate limiters
	for _, arl := range rm.adaptiveRateLimiters {
		arl.RecordLatency(latency)
	}
}

// Comprehensive execution with all protections

// ExecuteWithAllProtections executes with full resilience stack
func (rm *ResilienceManager) ExecuteWithAllProtections(ctx context.Context, operationName string, fn func(context.Context) error) error {
	startTime := time.Now()

	// 1. Check chaos injection
	if err := rm.MaybeInjectFault(); err != nil {
		rm.logger.Debug("Chaos fault injected",
			zap.String("operation", operationName),
			zap.Error(err))
		return err
	}

	// 2. Check rate limit
	if !rm.CheckRateLimit(operationName) {
		rm.RecordRequest(operationName, false)
		return ErrRateLimitExceeded
	}

	// 3. Execute with all protections
	var execErr error
	err := rm.ExecuteWithBulkhead(operationName, func() error {
		return rm.ExecuteWithCircuitBreaker(operationName, func() error {
			return rm.ExecuteWithRetry(operationName, func() error {
				return rm.timeoutManager.ExecuteWithContext(ctx, operationName, fn)
			})
		})
	})

	// Record metrics
	latency := time.Since(startTime)
	success := err == nil

	rm.RecordRequest(operationName, success)
	rm.RecordLatency(operationName, latency)

	return execErr
}

// GetAllMetrics returns comprehensive metrics
func (rm *ResilienceManager) GetAllMetrics() ResilienceMetrics {
	metrics := ResilienceMetrics{
		Name:            rm.name,
		CircuitBreakers: make(map[string]CircuitBreakerMetrics),
		RateLimiters:    make(map[string]RateLimiterMetrics),
		Bulkheads:       make(map[string]BulkheadMetrics),
		RetryPolicies:   make(map[string]RetryPolicyMetrics),
		ErrorBudgets:    make(map[string]ErrorBudgetMetrics),
		LatencyBudgets:  make(map[string]LatencyBudgetMetrics),
		HealthChecker:   rm.healthChecker.GetMetrics(),
		ChaosMonkey:     rm.chaosMonkey.GetMetrics(),
		Degradation:     rm.degradationMgr.GetMetrics(),
		Timeout:         rm.timeoutManager.GetMetrics(),
	}

	for name, cb := range rm.circuitBreakers {
		metrics.CircuitBreakers[name] = cb.GetMetrics()
	}

	for name, rl := range rm.rateLimiters {
		metrics.RateLimiters[name] = rl.GetMetrics()
	}

	for name, bh := range rm.bulkheads {
		metrics.Bulkheads[name] = bh.GetMetrics()
	}

	for name, rp := range rm.retryPolicies {
		metrics.RetryPolicies[name] = rp.GetMetrics()
	}

	for name, eb := range rm.errorBudgets {
		metrics.ErrorBudgets[name] = eb.GetMetrics()
	}

	for name, lb := range rm.latencyBudgets {
		metrics.LatencyBudgets[name] = lb.GetMetrics()
	}

	return metrics
}

// Metrics type

// ResilienceMetrics contains all resilience metrics
type ResilienceMetrics struct {
	Name            string
	CircuitBreakers map[string]CircuitBreakerMetrics
	RateLimiters    map[string]RateLimiterMetrics
	Bulkheads       map[string]BulkheadMetrics
	RetryPolicies   map[string]RetryPolicyMetrics
	ErrorBudgets    map[string]ErrorBudgetMetrics
	LatencyBudgets  map[string]LatencyBudgetMetrics
	HealthChecker   HealthCheckerMetrics
	ChaosMonkey     ChaosMonkeyMetrics
	Degradation     DegradationMetrics
	Timeout         TimeoutMetrics
}
