package resilience

import (
	"sync"
	"sync/atomic"
	"time"

	"go.uber.org/zap"
)

// ErrorBudget tracks SLO compliance and error budgets
type ErrorBudget struct {
	name           string
	slo            float64 // Service Level Objective (e.g., 99.9% = 0.999)
	totalRequests  int64
	failedRequests int64
	windowStart    time.Time
	windowDuration time.Duration
	logger         *zap.Logger
	mu             sync.RWMutex

	// Callbacks
	onBudgetExhausted  func()
	onBudgetRecovered  func()
	budgetExhausted    bool

	// Time-based tracking
	uptimeStart     time.Time
	downtimeTotal   time.Duration
	lastFailTime    time.Time
	lastSuccessTime time.Time
}

// NewErrorBudget creates a new error budget tracker
func NewErrorBudget(name string, slo float64, windowDuration time.Duration, logger *zap.Logger) *ErrorBudget {
	if logger == nil {
		logger = zap.NewNop()
	}

	now := time.Now()
	return &ErrorBudget{
		name:            name,
		slo:             slo,
		windowStart:     now,
		windowDuration:  windowDuration,
		logger:          logger,
		uptimeStart:     now,
		lastSuccessTime: now,
	}
}

// RecordRequest records a request outcome
func (eb *ErrorBudget) RecordRequest(success bool) {
	atomic.AddInt64(&eb.totalRequests, 1)

	eb.mu.Lock()
	defer eb.mu.Unlock()

	now := time.Now()

	if !success {
		atomic.AddInt64(&eb.failedRequests, 1)
		eb.lastFailTime = now

		// Track downtime
		if !eb.lastSuccessTime.IsZero() {
			eb.downtimeTotal += now.Sub(eb.lastSuccessTime)
		}
	} else {
		eb.lastSuccessTime = now
	}

	// Check if window needs to be reset
	if time.Since(eb.windowStart) > eb.windowDuration {
		eb.resetWindow()
	}

	// Check if budget exhausted
	wasExhausted := eb.budgetExhausted
	eb.budgetExhausted = eb.isExhausted()

	if !wasExhausted && eb.budgetExhausted {
		eb.logger.Error("Error budget exhausted",
			zap.String("name", eb.name),
			zap.Float64("slo", eb.slo),
			zap.Float64("actualSLO", eb.SuccessRate()))

		if eb.onBudgetExhausted != nil {
			go eb.onBudgetExhausted()
		}
	} else if wasExhausted && !eb.budgetExhausted {
		eb.logger.Info("Error budget recovered",
			zap.String("name", eb.name),
			zap.Float64("slo", eb.slo),
			zap.Float64("actualSLO", eb.SuccessRate()))

		if eb.onBudgetRecovered != nil {
			go eb.onBudgetRecovered()
		}
	}
}

// SuccessRate returns the current success rate
func (eb *ErrorBudget) SuccessRate() float64 {
	total := atomic.LoadInt64(&eb.totalRequests)
	if total == 0 {
		return 1.0
	}

	failed := atomic.LoadInt64(&eb.failedRequests)
	return 1.0 - float64(failed)/float64(total)
}

// BudgetExhausted returns true if error budget is exhausted
func (eb *ErrorBudget) BudgetExhausted() bool {
	eb.mu.RLock()
	defer eb.mu.RUnlock()
	return eb.budgetExhausted
}

// isExhausted checks if budget is exhausted (must be called with lock held)
func (eb *ErrorBudget) isExhausted() bool {
	return eb.SuccessRate() < eb.slo
}

// RemainingBudget returns percentage of error budget remaining
func (eb *ErrorBudget) RemainingBudget() float64 {
	total := atomic.LoadInt64(&eb.totalRequests)
	if total == 0 {
		return 1.0
	}

	failed := atomic.LoadInt64(&eb.failedRequests)
	allowedFailures := float64(total) * (1.0 - eb.slo)
	usedBudget := float64(failed) / allowedFailures

	remaining := 1.0 - usedBudget
	if remaining < 0 {
		remaining = 0
	}

	return remaining
}

// AllowedFailures returns number of failures allowed to stay within SLO
func (eb *ErrorBudget) AllowedFailures() int64 {
	total := atomic.LoadInt64(&eb.totalRequests)
	if total == 0 {
		return 0
	}

	allowedFailures := int64(float64(total) * (1.0 - eb.slo))
	failed := atomic.LoadInt64(&eb.failedRequests)

	remaining := allowedFailures - failed
	if remaining < 0 {
		remaining = 0
	}

	return remaining
}

// Availability returns availability percentage
func (eb *ErrorBudget) Availability() float64 {
	eb.mu.RLock()
	defer eb.mu.RUnlock()

	totalTime := time.Since(eb.uptimeStart)
	if totalTime == 0 {
		return 1.0
	}

	uptime := totalTime - eb.downtimeTotal
	return float64(uptime) / float64(totalTime)
}

// SetBudgetExhaustedCallback sets callback for budget exhaustion
func (eb *ErrorBudget) SetBudgetExhaustedCallback(fn func()) {
	eb.onBudgetExhausted = fn
}

// SetBudgetRecoveredCallback sets callback for budget recovery
func (eb *ErrorBudget) SetBudgetRecoveredCallback(fn func()) {
	eb.onBudgetRecovered = fn
}

// resetWindow resets the tracking window
func (eb *ErrorBudget) resetWindow() {
	now := time.Now()
	eb.windowStart = now
	atomic.StoreInt64(&eb.totalRequests, 0)
	atomic.StoreInt64(&eb.failedRequests, 0)

	eb.logger.Info("Error budget window reset",
		zap.String("name", eb.name),
		zap.Time("newWindowStart", now))
}

// Reset resets the error budget
func (eb *ErrorBudget) Reset() {
	eb.mu.Lock()
	defer eb.mu.Unlock()

	now := time.Now()
	eb.windowStart = now
	eb.uptimeStart = now
	eb.downtimeTotal = 0
	eb.budgetExhausted = false
	atomic.StoreInt64(&eb.totalRequests, 0)
	atomic.StoreInt64(&eb.failedRequests, 0)

	eb.logger.Info("Error budget reset",
		zap.String("name", eb.name))
}

// GetMetrics returns error budget metrics
func (eb *ErrorBudget) GetMetrics() ErrorBudgetMetrics {
	eb.mu.RLock()
	defer eb.mu.RUnlock()

	total := atomic.LoadInt64(&eb.totalRequests)
	failed := atomic.LoadInt64(&eb.failedRequests)

	return ErrorBudgetMetrics{
		Name:             eb.name,
		SLO:              eb.slo,
		TotalRequests:    total,
		FailedRequests:   failed,
		SuccessRate:      eb.SuccessRate(),
		RemainingBudget:  eb.RemainingBudget(),
		AllowedFailures:  eb.AllowedFailures(),
		BudgetExhausted:  eb.budgetExhausted,
		Availability:     eb.Availability(),
		WindowStart:      eb.windowStart,
		WindowDuration:   eb.windowDuration,
		TotalDowntime:    eb.downtimeTotal,
		LastFailTime:     eb.lastFailTime,
		LastSuccessTime:  eb.lastSuccessTime,
	}
}

// SLOTracker tracks multiple SLOs
type SLOTracker struct {
	name    string
	budgets map[string]*ErrorBudget
	logger  *zap.Logger
	mu      sync.RWMutex
}

// NewSLOTracker creates a new SLO tracker
func NewSLOTracker(name string, logger *zap.Logger) *SLOTracker {
	if logger == nil {
		logger = zap.NewNop()
	}

	return &SLOTracker{
		name:    name,
		budgets: make(map[string]*ErrorBudget),
		logger:  logger,
	}
}

// RegisterSLO registers an SLO to track
func (st *SLOTracker) RegisterSLO(name string, slo float64, windowDuration time.Duration) {
	st.mu.Lock()
	defer st.mu.Unlock()

	budget := NewErrorBudget(name, slo, windowDuration, st.logger)
	st.budgets[name] = budget

	st.logger.Info("SLO registered",
		zap.String("tracker", st.name),
		zap.String("slo", name),
		zap.Float64("target", slo))
}

// RecordRequest records a request for an SLO
func (st *SLOTracker) RecordRequest(name string, success bool) {
	st.mu.RLock()
	budget, exists := st.budgets[name]
	st.mu.RUnlock()

	if exists {
		budget.RecordRequest(success)
	}
}

// GetBudget returns an error budget by name
func (st *SLOTracker) GetBudget(name string) (*ErrorBudget, bool) {
	st.mu.RLock()
	defer st.mu.RUnlock()

	budget, exists := st.budgets[name]
	return budget, exists
}

// AllBudgetsHealthy returns true if all SLOs are being met
func (st *SLOTracker) AllBudgetsHealthy() bool {
	st.mu.RLock()
	defer st.mu.RUnlock()

	for _, budget := range st.budgets {
		if budget.BudgetExhausted() {
			return false
		}
	}

	return true
}

// GetAllMetrics returns metrics for all SLOs
func (st *SLOTracker) GetAllMetrics() map[string]ErrorBudgetMetrics {
	st.mu.RLock()
	defer st.mu.RUnlock()

	metrics := make(map[string]ErrorBudgetMetrics)
	for name, budget := range st.budgets {
		metrics[name] = budget.GetMetrics()
	}

	return metrics
}

// LatencyBudget tracks latency SLOs
type LatencyBudget struct {
	name            string
	targetLatency   time.Duration
	percentile      float64
	latencies       []time.Duration
	maxSamples      int
	logger          *zap.Logger
	mu              sync.RWMutex

	// Metrics
	totalRequests   int64
	exceedingRequests int64
}

// NewLatencyBudget creates a new latency budget tracker
func NewLatencyBudget(name string, targetLatency time.Duration, percentile float64, maxSamples int, logger *zap.Logger) *LatencyBudget {
	if logger == nil {
		logger = zap.NewNop()
	}

	return &LatencyBudget{
		name:          name,
		targetLatency: targetLatency,
		percentile:    percentile,
		maxSamples:    maxSamples,
		latencies:     make([]time.Duration, 0, maxSamples),
		logger:        logger,
	}
}

// RecordLatency records a latency measurement
func (lb *LatencyBudget) RecordLatency(latency time.Duration) {
	atomic.AddInt64(&lb.totalRequests, 1)

	if latency > lb.targetLatency {
		atomic.AddInt64(&lb.exceedingRequests, 1)
	}

	lb.mu.Lock()
	defer lb.mu.Unlock()

	lb.latencies = append(lb.latencies, latency)
	if len(lb.latencies) > lb.maxSamples {
		lb.latencies = lb.latencies[len(lb.latencies)-lb.maxSamples:]
	}
}

// PercentileLatency calculates percentile latency
func (lb *LatencyBudget) PercentileLatency() time.Duration {
	lb.mu.RLock()
	defer lb.mu.RUnlock()

	if len(lb.latencies) == 0 {
		return 0
	}

	// Copy and sort
	sorted := make([]time.Duration, len(lb.latencies))
	copy(sorted, lb.latencies)

	// Simple bubble sort
	for i := 0; i < len(sorted)-1; i++ {
		for j := 0; j < len(sorted)-i-1; j++ {
			if sorted[j] > sorted[j+1] {
				sorted[j], sorted[j+1] = sorted[j+1], sorted[j]
			}
		}
	}

	index := int(float64(len(sorted)) * lb.percentile / 100.0)
	if index >= len(sorted) {
		index = len(sorted) - 1
	}

	return sorted[index]
}

// BudgetExhausted returns true if latency budget is exhausted
func (lb *LatencyBudget) BudgetExhausted() bool {
	return lb.PercentileLatency() > lb.targetLatency
}

// ComplianceRate returns percentage of requests meeting latency target
func (lb *LatencyBudget) ComplianceRate() float64 {
	total := atomic.LoadInt64(&lb.totalRequests)
	if total == 0 {
		return 1.0
	}

	exceeding := atomic.LoadInt64(&lb.exceedingRequests)
	return 1.0 - float64(exceeding)/float64(total)
}

// GetMetrics returns latency budget metrics
func (lb *LatencyBudget) GetMetrics() LatencyBudgetMetrics {
	return LatencyBudgetMetrics{
		Name:              lb.name,
		TargetLatency:     lb.targetLatency,
		Percentile:        lb.percentile,
		CurrentLatency:    lb.PercentileLatency(),
		TotalRequests:     atomic.LoadInt64(&lb.totalRequests),
		ExceedingRequests: atomic.LoadInt64(&lb.exceedingRequests),
		ComplianceRate:    lb.ComplianceRate(),
		BudgetExhausted:   lb.BudgetExhausted(),
	}
}

// Metrics types

// ErrorBudgetMetrics contains error budget metrics
type ErrorBudgetMetrics struct {
	Name            string
	SLO             float64
	TotalRequests   int64
	FailedRequests  int64
	SuccessRate     float64
	RemainingBudget float64
	AllowedFailures int64
	BudgetExhausted bool
	Availability    float64
	WindowStart     time.Time
	WindowDuration  time.Duration
	TotalDowntime   time.Duration
	LastFailTime    time.Time
	LastSuccessTime time.Time
}

// LatencyBudgetMetrics contains latency budget metrics
type LatencyBudgetMetrics struct {
	Name              string
	TargetLatency     time.Duration
	Percentile        float64
	CurrentLatency    time.Duration
	TotalRequests     int64
	ExceedingRequests int64
	ComplianceRate    float64
	BudgetExhausted   bool
}