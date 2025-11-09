package resilience

import (
	"context"
	"fmt"
	"sync"
	"time"

	"go.uber.org/zap"
)

// HealthCheck defines a health check interface
type HealthCheck interface {
	Name() string
	Check(ctx context.Context) error
}

// HealthStatus represents the status of a health check
type HealthStatus struct {
	Name        string
	Healthy     bool
	LastCheck   time.Time
	LastSuccess time.Time
	LastFailure time.Time
	Error       string
	CheckCount  int64
	FailCount   int64
	SuccessRate float64
	Duration    time.Duration
}

// HealthChecker monitors system health
type HealthChecker struct {
	name         string
	checks       map[string]HealthCheck
	statuses     map[string]*HealthStatus
	interval     time.Duration
	timeout      time.Duration
	logger       *zap.Logger
	mu           sync.RWMutex
	stopCh       chan struct{}
	running      bool
	wg           sync.WaitGroup

	// Callbacks
	onHealthy   func(name string)
	onUnhealthy func(name string, err error)
}

// NewHealthChecker creates a new health checker
func NewHealthChecker(name string, interval, timeout time.Duration, logger *zap.Logger) *HealthChecker {
	if logger == nil {
		logger = zap.NewNop()
	}

	return &HealthChecker{
		name:     name,
		checks:   make(map[string]HealthCheck),
		statuses: make(map[string]*HealthStatus),
		interval: interval,
		timeout:  timeout,
		logger:   logger,
		stopCh:   make(chan struct{}),
	}
}

// RegisterCheck registers a health check
func (hc *HealthChecker) RegisterCheck(check HealthCheck) {
	hc.mu.Lock()
	defer hc.mu.Unlock()

	name := check.Name()
	hc.checks[name] = check
	hc.statuses[name] = &HealthStatus{
		Name:    name,
		Healthy: true,
	}

	hc.logger.Info("Health check registered",
		zap.String("checker", hc.name),
		zap.String("check", name))
}

// UnregisterCheck unregisters a health check
func (hc *HealthChecker) UnregisterCheck(name string) {
	hc.mu.Lock()
	defer hc.mu.Unlock()

	delete(hc.checks, name)
	delete(hc.statuses, name)

	hc.logger.Info("Health check unregistered",
		zap.String("checker", hc.name),
		zap.String("check", name))
}

// SetHealthyCallback sets callback for healthy status
func (hc *HealthChecker) SetHealthyCallback(fn func(name string)) {
	hc.onHealthy = fn
}

// SetUnhealthyCallback sets callback for unhealthy status
func (hc *HealthChecker) SetUnhealthyCallback(fn func(name string, err error)) {
	hc.onUnhealthy = fn
}

// StartMonitoring starts periodic health checks
func (hc *HealthChecker) StartMonitoring() {
	hc.mu.Lock()
	if hc.running {
		hc.mu.Unlock()
		return
	}
	hc.running = true
	hc.mu.Unlock()

	hc.wg.Add(1)
	go hc.monitorLoop()

	hc.logger.Info("Health monitoring started",
		zap.String("checker", hc.name),
		zap.Duration("interval", hc.interval))
}

// StopMonitoring stops health checks
func (hc *HealthChecker) StopMonitoring() {
	hc.mu.Lock()
	if !hc.running {
		hc.mu.Unlock()
		return
	}
	hc.running = false
	hc.mu.Unlock()

	close(hc.stopCh)
	hc.wg.Wait()

	hc.logger.Info("Health monitoring stopped",
		zap.String("checker", hc.name))
}

// monitorLoop runs periodic health checks
func (hc *HealthChecker) monitorLoop() {
	defer hc.wg.Done()

	ticker := time.NewTicker(hc.interval)
	defer ticker.Stop()

	// Run initial check immediately
	hc.runAllChecks()

	for {
		select {
		case <-ticker.C:
			hc.runAllChecks()
		case <-hc.stopCh:
			return
		}
	}
}

// runAllChecks executes all registered health checks
func (hc *HealthChecker) runAllChecks() {
	hc.mu.RLock()
	checks := make([]HealthCheck, 0, len(hc.checks))
	for _, check := range hc.checks {
		checks = append(checks, check)
	}
	hc.mu.RUnlock()

	var wg sync.WaitGroup
	for _, check := range checks {
		wg.Add(1)
		go func(c HealthCheck) {
			defer wg.Done()
			hc.runCheck(c)
		}(check)
	}

	wg.Wait()
}

// runCheck executes a single health check
func (hc *HealthChecker) runCheck(check HealthCheck) {
	ctx, cancel := context.WithTimeout(context.Background(), hc.timeout)
	defer cancel()

	name := check.Name()
	startTime := time.Now()

	err := check.Check(ctx)
	duration := time.Since(startTime)

	hc.mu.Lock()
	defer hc.mu.Unlock()

	status, exists := hc.statuses[name]
	if !exists {
		return
	}

	status.LastCheck = startTime
	status.Duration = duration
	status.CheckCount++

	previousHealth := status.Healthy

	if err != nil {
		status.Healthy = false
		status.LastFailure = startTime
		status.Error = err.Error()
		status.FailCount++

		hc.logger.Warn("Health check failed",
			zap.String("checker", hc.name),
			zap.String("check", name),
			zap.Error(err),
			zap.Duration("duration", duration))

		// Call unhealthy callback if status changed
		if previousHealth && hc.onUnhealthy != nil {
			go hc.onUnhealthy(name, err)
		}
	} else {
		status.Healthy = true
		status.LastSuccess = startTime
		status.Error = ""

		hc.logger.Debug("Health check passed",
			zap.String("checker", hc.name),
			zap.String("check", name),
			zap.Duration("duration", duration))

		// Call healthy callback if status changed
		if !previousHealth && hc.onHealthy != nil {
			go hc.onHealthy(name)
		}
	}

	// Update success rate
	if status.CheckCount > 0 {
		successCount := status.CheckCount - status.FailCount
		status.SuccessRate = float64(successCount) / float64(status.CheckCount)
	}
}

// CheckNow runs all health checks immediately
func (hc *HealthChecker) CheckNow() {
	hc.runAllChecks()
}

// GetStatus returns the status of a specific check
func (hc *HealthChecker) GetStatus(name string) (*HealthStatus, bool) {
	hc.mu.RLock()
	defer hc.mu.RUnlock()

	status, exists := hc.statuses[name]
	if !exists {
		return nil, false
	}

	// Return a copy
	statusCopy := *status
	return &statusCopy, true
}

// GetAllStatuses returns all health check statuses
func (hc *HealthChecker) GetAllStatuses() map[string]*HealthStatus {
	hc.mu.RLock()
	defer hc.mu.RUnlock()

	statuses := make(map[string]*HealthStatus, len(hc.statuses))
	for name, status := range hc.statuses {
		statusCopy := *status
		statuses[name] = &statusCopy
	}

	return statuses
}

// IsHealthy returns true if all checks are healthy
func (hc *HealthChecker) IsHealthy() bool {
	hc.mu.RLock()
	defer hc.mu.RUnlock()

	for _, status := range hc.statuses {
		if !status.Healthy {
			return false
		}
	}

	return true
}

// GetMetrics returns health checker metrics
func (hc *HealthChecker) GetMetrics() HealthCheckerMetrics {
	hc.mu.RLock()
	defer hc.mu.RUnlock()

	totalChecks := int64(0)
	totalFails := int64(0)
	healthyCount := 0
	unhealthyCount := 0

	for _, status := range hc.statuses {
		totalChecks += status.CheckCount
		totalFails += status.FailCount
		if status.Healthy {
			healthyCount++
		} else {
			unhealthyCount++
		}
	}

	overallSuccessRate := float64(0)
	if totalChecks > 0 {
		successCount := totalChecks - totalFails
		overallSuccessRate = float64(successCount) / float64(totalChecks)
	}

	return HealthCheckerMetrics{
		Name:               hc.name,
		TotalChecks:        len(hc.checks),
		HealthyChecks:      healthyCount,
		UnhealthyChecks:    unhealthyCount,
		TotalCheckRuns:     totalChecks,
		TotalFailures:      totalFails,
		OverallSuccessRate: overallSuccessRate,
		IsHealthy:          unhealthyCount == 0,
	}
}

// Common health check implementations

// PingHealthCheck checks if a service responds to ping
type PingHealthCheck struct {
	name     string
	pingFunc func(context.Context) error
}

// NewPingHealthCheck creates a ping-based health check
func NewPingHealthCheck(name string, pingFunc func(context.Context) error) *PingHealthCheck {
	return &PingHealthCheck{
		name:     name,
		pingFunc: pingFunc,
	}
}

// Name returns the check name
func (phc *PingHealthCheck) Name() string {
	return phc.name
}

// Check executes the ping check
func (phc *PingHealthCheck) Check(ctx context.Context) error {
	return phc.pingFunc(ctx)
}

// ThresholdHealthCheck checks if a metric is within threshold
type ThresholdHealthCheck struct {
	name        string
	metricFunc  func() float64
	minValue    float64
	maxValue    float64
	description string
}

// NewThresholdHealthCheck creates a threshold-based health check
func NewThresholdHealthCheck(name string, metricFunc func() float64, min, max float64) *ThresholdHealthCheck {
	return &ThresholdHealthCheck{
		name:       name,
		metricFunc: metricFunc,
		minValue:   min,
		maxValue:   max,
	}
}

// Name returns the check name
func (thc *ThresholdHealthCheck) Name() string {
	return thc.name
}

// Check executes the threshold check
func (thc *ThresholdHealthCheck) Check(ctx context.Context) error {
	value := thc.metricFunc()
	if value < thc.minValue {
		return fmt.Errorf("value %.2f below minimum threshold %.2f", value, thc.minValue)
	}
	if value > thc.maxValue {
		return fmt.Errorf("value %.2f exceeds maximum threshold %.2f", value, thc.maxValue)
	}
	return nil
}

// CompositeHealthCheck combines multiple checks
type CompositeHealthCheck struct {
	name   string
	checks []HealthCheck
}

// NewCompositeHealthCheck creates a composite health check
func NewCompositeHealthCheck(name string, checks ...HealthCheck) *CompositeHealthCheck {
	return &CompositeHealthCheck{
		name:   name,
		checks: checks,
	}
}

// Name returns the check name
func (chc *CompositeHealthCheck) Name() string {
	return chc.name
}

// Check executes all sub-checks
func (chc *CompositeHealthCheck) Check(ctx context.Context) error {
	for _, check := range chc.checks {
		if err := check.Check(ctx); err != nil {
			return fmt.Errorf("%s failed: %w", check.Name(), err)
		}
	}
	return nil
}

// Metrics types

// HealthCheckerMetrics contains health checker metrics
type HealthCheckerMetrics struct {
	Name               string
	TotalChecks        int
	HealthyChecks      int
	UnhealthyChecks    int
	TotalCheckRuns     int64
	TotalFailures      int64
	OverallSuccessRate float64
	IsHealthy          bool
}