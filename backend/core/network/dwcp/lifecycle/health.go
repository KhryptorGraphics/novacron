package lifecycle

import (
	"context"
	"sync"
	"time"

	"go.uber.org/zap"
)

// HealthMonitor monitors component health and triggers recovery
type HealthMonitor struct {
	manager        *Manager
	interval       time.Duration
	timeout        time.Duration
	logger         *zap.Logger
	ctx            context.Context
	cancel         context.CancelFunc
	wg             sync.WaitGroup
	healthStatus   map[string]error
	healthMutex    sync.RWMutex
	recoveryTracker *RecoveryTracker
}

// NewHealthMonitor creates a new health monitor
func NewHealthMonitor(manager *Manager, interval, timeout time.Duration, logger *zap.Logger) *HealthMonitor {
	return &HealthMonitor{
		manager:         manager,
		interval:        interval,
		timeout:         timeout,
		logger:          logger,
		healthStatus:    make(map[string]error),
		recoveryTracker: NewRecoveryTracker(logger),
	}
}

// Start begins health monitoring
func (h *HealthMonitor) Start(ctx context.Context) {
	h.ctx, h.cancel = context.WithCancel(ctx)

	h.wg.Add(1)
	go h.monitoringLoop()

	h.logger.Info("Health monitoring started",
		zap.Duration("interval", h.interval),
		zap.Duration("timeout", h.timeout))
}

// Stop stops health monitoring
func (h *HealthMonitor) Stop() {
	if h.cancel != nil {
		h.cancel()
	}
	h.wg.Wait()
	h.logger.Info("Health monitoring stopped")
}

// monitoringLoop periodically checks component health
func (h *HealthMonitor) monitoringLoop() {
	defer h.wg.Done()

	ticker := time.NewTicker(h.interval)
	defer ticker.Stop()

	for {
		select {
		case <-h.ctx.Done():
			return
		case <-ticker.C:
			h.checkAllComponents()
		}
	}
}

// checkAllComponents checks health of all components
func (h *HealthMonitor) checkAllComponents() {
	components := h.manager.GetAllComponents()

	var wg sync.WaitGroup
	for name, component := range components {
		wg.Add(1)
		go func(compName string, comp ComponentLifecycle) {
			defer wg.Done()
			h.checkComponent(compName, comp)
		}(name, component)
	}

	wg.Wait()
}

// checkComponent checks a single component's health
func (h *HealthMonitor) checkComponent(name string, component ComponentLifecycle) {
	// Create timeout context for health check
	ctx, cancel := context.WithTimeout(h.ctx, h.timeout)
	defer cancel()

	startTime := time.Now()
	err := component.HealthCheck(ctx)
	duration := time.Since(startTime)

	// Update health status
	h.healthMutex.Lock()
	h.healthStatus[name] = err
	h.healthMutex.Unlock()

	if err != nil {
		h.logger.Warn("Component health check failed",
			zap.String("component", name),
			zap.Error(err),
			zap.Duration("duration", duration))

		// Attempt recovery if component is recoverable
		if recoverable, ok := component.(Recoverable); ok {
			h.attemptRecovery(name, recoverable)
		}
	} else {
		h.logger.Debug("Component health check passed",
			zap.String("component", name),
			zap.Duration("duration", duration))
	}
}

// attemptRecovery attempts to recover a failed component
func (h *HealthMonitor) attemptRecovery(name string, component Recoverable) {
	// Check if already recovering
	if h.recoveryTracker.IsRecovering(name) {
		return
	}

	// Mark as recovering
	h.recoveryTracker.StartRecovery(name)

	// Spawn recovery goroutine
	go func() {
		defer h.recoveryTracker.EndRecovery(name)

		strategy := component.GetRecoveryStrategy()
		h.logger.Info("Starting component recovery",
			zap.String("component", name),
			zap.Int("max_retries", strategy.MaxRetries))

		startTime := time.Now()

		for attempt := 0; attempt < strategy.MaxRetries; attempt++ {
			// Calculate backoff
			backoff := h.calculateBackoff(strategy, attempt)

			if attempt > 0 {
				h.logger.Info("Recovery retry",
					zap.String("component", name),
					zap.Int("attempt", attempt+1),
					zap.Int("max_retries", strategy.MaxRetries),
					zap.Duration("backoff", backoff))

				select {
				case <-time.After(backoff):
				case <-h.ctx.Done():
					return
				}
			}

			// Attempt recovery
			ctx, cancel := context.WithTimeout(h.ctx, 30*time.Second)
			err := component.Recover(ctx)
			cancel()

			if err == nil {
				duration := time.Since(startTime)
				h.logger.Info("Component recovered successfully",
					zap.String("component", name),
					zap.Int("attempts", attempt+1),
					zap.Duration("duration", duration))

				h.recoveryTracker.RecordSuccess(name, duration)
				return
			}

			h.logger.Warn("Recovery attempt failed",
				zap.String("component", name),
				zap.Int("attempt", attempt+1),
				zap.Error(err))
		}

		// Recovery failed after all attempts
		h.logger.Error("Component recovery failed after all attempts",
			zap.String("component", name),
			zap.Int("max_retries", strategy.MaxRetries),
			zap.Duration("total_duration", time.Since(startTime)))

		h.recoveryTracker.RecordFailure(name)
	}()
}

// calculateBackoff calculates backoff duration for retry
func (h *HealthMonitor) calculateBackoff(strategy RecoveryStrategy, attempt int) time.Duration {
	backoff := strategy.RetryBackoff

	if strategy.ExponentialBackoff {
		backoff = strategy.RetryBackoff * time.Duration(1<<uint(attempt))
	}

	if backoff > strategy.MaxBackoff {
		backoff = strategy.MaxBackoff
	}

	return backoff
}

// GetHealthStatus returns current health status of all components
func (h *HealthMonitor) GetHealthStatus() map[string]error {
	h.healthMutex.RLock()
	defer h.healthMutex.RUnlock()

	status := make(map[string]error, len(h.healthStatus))
	for k, v := range h.healthStatus {
		status[k] = v
	}

	return status
}

// GetRecoveryMetrics returns recovery metrics
func (h *HealthMonitor) GetRecoveryMetrics() map[string]RecoveryMetrics {
	return h.recoveryTracker.GetMetrics()
}

// RecoveryTracker tracks component recovery attempts
type RecoveryTracker struct {
	recovering map[string]bool
	metrics    map[string]RecoveryMetrics
	mu         sync.RWMutex
	logger     *zap.Logger
}

// RecoveryMetrics tracks recovery statistics
type RecoveryMetrics struct {
	ComponentName       string
	TotalAttempts       int64
	SuccessfulRecovery  int64
	FailedRecovery      int64
	LastRecoveryTime    time.Time
	LastRecoverySuccess bool
	AverageRecoveryTime time.Duration
	CurrentlyRecovering bool
}

// NewRecoveryTracker creates a new recovery tracker
func NewRecoveryTracker(logger *zap.Logger) *RecoveryTracker {
	return &RecoveryTracker{
		recovering: make(map[string]bool),
		metrics:    make(map[string]RecoveryMetrics),
		logger:     logger,
	}
}

// IsRecovering checks if component is currently recovering
func (r *RecoveryTracker) IsRecovering(component string) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.recovering[component]
}

// StartRecovery marks component as recovering
func (r *RecoveryTracker) StartRecovery(component string) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.recovering[component] = true

	metrics := r.metrics[component]
	metrics.ComponentName = component
	metrics.TotalAttempts++
	metrics.CurrentlyRecovering = true
	r.metrics[component] = metrics

	// Notify observers
	r.logger.Info("Recovery started",
		zap.String("component", component))
}

// EndRecovery marks component as no longer recovering
func (r *RecoveryTracker) EndRecovery(component string) {
	r.mu.Lock()
	defer r.mu.Unlock()

	delete(r.recovering, component)

	metrics := r.metrics[component]
	metrics.CurrentlyRecovering = false
	r.metrics[component] = metrics
}

// RecordSuccess records successful recovery
func (r *RecoveryTracker) RecordSuccess(component string, duration time.Duration) {
	r.mu.Lock()
	defer r.mu.Unlock()

	metrics := r.metrics[component]
	metrics.SuccessfulRecovery++
	metrics.LastRecoveryTime = time.Now()
	metrics.LastRecoverySuccess = true

	// Update average recovery time
	if metrics.AverageRecoveryTime == 0 {
		metrics.AverageRecoveryTime = duration
	} else {
		metrics.AverageRecoveryTime = (metrics.AverageRecoveryTime + duration) / 2
	}

	r.metrics[component] = metrics
}

// RecordFailure records failed recovery
func (r *RecoveryTracker) RecordFailure(component string) {
	r.mu.Lock()
	defer r.mu.Unlock()

	metrics := r.metrics[component]
	metrics.FailedRecovery++
	metrics.LastRecoveryTime = time.Now()
	metrics.LastRecoverySuccess = false
	r.metrics[component] = metrics
}

// GetMetrics returns recovery metrics for all components
func (r *RecoveryTracker) GetMetrics() map[string]RecoveryMetrics {
	r.mu.RLock()
	defer r.mu.RUnlock()

	metrics := make(map[string]RecoveryMetrics, len(r.metrics))
	for k, v := range r.metrics {
		metrics[k] = v
	}

	return metrics
}

// GetComponentMetrics returns metrics for a specific component
func (r *RecoveryTracker) GetComponentMetrics(component string) RecoveryMetrics {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.metrics[component]
}

// Reset resets recovery tracking
func (r *RecoveryTracker) Reset() {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.recovering = make(map[string]bool)
	r.metrics = make(map[string]RecoveryMetrics)
}
