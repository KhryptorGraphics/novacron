package conflict

import (
	"context"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	// Conflict detection metrics
	conflictDetectionRate = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "dwcp_conflict_detection_rate",
		Help: "Current conflict detection rate (conflicts/second)",
	})

	conflictResolutionSuccessRate = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "dwcp_conflict_resolution_success_rate",
		Help: "Conflict resolution success rate (0-1)",
	})

	manualInterventionRate = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "dwcp_manual_intervention_rate",
		Help: "Rate of conflicts requiring manual intervention",
	})

	// Performance metrics
	averageResolutionTime = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "dwcp_average_resolution_time_ms",
		Help: "Average conflict resolution time in milliseconds",
	})

	p99ResolutionTime = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "dwcp_p99_resolution_time_ms",
		Help: "99th percentile resolution time in milliseconds",
	})

	// Conflict backlog
	pendingConflictsCount = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "dwcp_pending_conflicts_count",
		Help: "Number of pending conflicts",
	})

	// Strategy usage
	strategyUsage = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "dwcp_strategy_usage_total",
		Help: "Total usage count per strategy",
	}, []string{"strategy"})

	// Resource hotspots
	resourceConflictCount = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "dwcp_resource_conflict_count",
		Help: "Conflict count per resource",
	}, []string{"resource_id"})

	// Data integrity
	dataLossEvents = promauto.NewCounter(prometheus.CounterOpts{
		Name: "dwcp_data_loss_events_total",
		Help: "Total number of data loss events",
	})

	invariantViolations = promauto.NewCounter(prometheus.CounterOpts{
		Name: "dwcp_invariant_violations_total",
		Help: "Total number of invariant violations detected",
	})
)

// MetricsCollector collects and reports conflict resolution metrics
type MetricsCollector struct {
	mu                   sync.RWMutex
	detector             *ConflictDetector
	policyManager        *PolicyManager
	auditLog             *AuditLog
	resolutionTimes      []float64
	successCount         int64
	failureCount         int64
	manualCount          int64
	totalCount           int64
	lastUpdateTime       time.Time
	collectionInterval   time.Duration
}

// NewMetricsCollector creates a new metrics collector
func NewMetricsCollector(detector *ConflictDetector, pm *PolicyManager, audit *AuditLog) *MetricsCollector {
	mc := &MetricsCollector{
		detector:           detector,
		policyManager:      pm,
		auditLog:           audit,
		resolutionTimes:    make([]float64, 0, 1000),
		lastUpdateTime:     time.Now(),
		collectionInterval: 10 * time.Second,
	}

	go mc.collectLoop()
	return mc
}

// collectLoop periodically collects and updates metrics
func (mc *MetricsCollector) collectLoop() {
	ticker := time.NewTicker(mc.collectionInterval)
	defer ticker.Stop()

	for range ticker.C {
		mc.collect()
	}
}

// collect collects current metrics
func (mc *MetricsCollector) collect() {
	mc.mu.Lock()
	defer mc.mu.Unlock()

	// Update conflict detection rate
	now := time.Now()
	duration := now.Sub(mc.lastUpdateTime).Seconds()
	if duration > 0 {
		rate := float64(mc.totalCount) / duration
		conflictDetectionRate.Set(rate)
	}

	// Update success rate
	if mc.totalCount > 0 {
		successRate := float64(mc.successCount) / float64(mc.totalCount)
		conflictResolutionSuccessRate.Set(successRate)

		manualRate := float64(mc.manualCount) / float64(mc.totalCount)
		manualInterventionRate.Set(manualRate)
	}

	// Update resolution times
	if len(mc.resolutionTimes) > 0 {
		avg := average(mc.resolutionTimes)
		averageResolutionTime.Set(avg)

		p99 := percentile(mc.resolutionTimes, 0.99)
		p99ResolutionTime.Set(p99)
	}

	// Update pending conflicts
	pending := mc.detector.GetPendingConflicts()
	pendingConflictsCount.Set(float64(len(pending)))

	// Reset counters
	mc.successCount = 0
	mc.failureCount = 0
	mc.manualCount = 0
	mc.totalCount = 0
	mc.resolutionTimes = mc.resolutionTimes[:0]
	mc.lastUpdateTime = now
}

// RecordResolution records a resolution attempt
func (mc *MetricsCollector) RecordResolution(strategy StrategyType, duration time.Duration, success bool, manual bool) {
	mc.mu.Lock()
	defer mc.mu.Unlock()

	mc.totalCount++
	if success {
		mc.successCount++
	} else {
		mc.failureCount++
	}

	if manual {
		mc.manualCount++
	}

	mc.resolutionTimes = append(mc.resolutionTimes, float64(duration.Milliseconds()))
	strategyUsage.WithLabelValues(strategy.String()).Inc()
}

// RecordResourceConflict records a conflict for a resource
func (mc *MetricsCollector) RecordResourceConflict(resourceID string) {
	resourceConflictCount.WithLabelValues(resourceID).Inc()
}

// RecordDataLoss records a data loss event
func (mc *MetricsCollector) RecordDataLoss() {
	dataLossEvents.Inc()
}

// RecordInvariantViolation records an invariant violation
func (mc *MetricsCollector) RecordInvariantViolation() {
	invariantViolations.Inc()
}

// GetMetricsSummary returns current metrics summary
func (mc *MetricsCollector) GetMetricsSummary() MetricsSummary {
	mc.mu.RLock()
	defer mc.mu.RUnlock()

	var avgResTime, p99ResTime float64
	if len(mc.resolutionTimes) > 0 {
		avgResTime = average(mc.resolutionTimes)
		p99ResTime = percentile(mc.resolutionTimes, 0.99)
	}

	successRate := 0.0
	if mc.totalCount > 0 {
		successRate = float64(mc.successCount) / float64(mc.totalCount)
	}

	return MetricsSummary{
		TotalConflicts:        mc.totalCount,
		SuccessfulResolutions: mc.successCount,
		FailedResolutions:     mc.failureCount,
		ManualInterventions:   mc.manualCount,
		SuccessRate:           successRate,
		AverageResolutionTime: avgResTime,
		P99ResolutionTime:     p99ResTime,
		PendingConflicts:      int64(len(mc.detector.GetPendingConflicts())),
	}
}

// MetricsSummary provides a summary of metrics
type MetricsSummary struct {
	TotalConflicts        int64
	SuccessfulResolutions int64
	FailedResolutions     int64
	ManualInterventions   int64
	SuccessRate           float64
	AverageResolutionTime float64
	P99ResolutionTime     float64
	PendingConflicts      int64
}

// PerformanceMonitor monitors performance against targets
type PerformanceMonitor struct {
	mu              sync.RWMutex
	targets         PerformanceTargets
	violations      []PerformanceViolation
	alertThreshold  int
}

// PerformanceTargets defines performance targets
type PerformanceTargets struct {
	MaxDetectionLatencyMS   float64
	MaxResolutionLatencyMS  float64
	MinSuccessRate          float64
	MaxManualInterventionRate float64
	MaxPendingConflicts     int
}

// DefaultPerformanceTargets returns default targets
func DefaultPerformanceTargets() PerformanceTargets {
	return PerformanceTargets{
		MaxDetectionLatencyMS:     1.0,
		MaxResolutionLatencyMS:    10.0,
		MinSuccessRate:            0.95,
		MaxManualInterventionRate: 0.05,
		MaxPendingConflicts:       100,
	}
}

// PerformanceViolation represents a target violation
type PerformanceViolation struct {
	Timestamp   time.Time
	Target      string
	Expected    float64
	Actual      float64
	Description string
}

// NewPerformanceMonitor creates a new performance monitor
func NewPerformanceMonitor(targets PerformanceTargets) *PerformanceMonitor {
	return &PerformanceMonitor{
		targets:        targets,
		violations:     make([]PerformanceViolation, 0),
		alertThreshold: 5,
	}
}

// CheckPerformance checks if performance meets targets
func (pm *PerformanceMonitor) CheckPerformance(summary MetricsSummary) []PerformanceViolation {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	violations := make([]PerformanceViolation, 0)

	// Check detection latency (not available in summary, would need separate tracking)

	// Check resolution latency
	if summary.AverageResolutionTime > pm.targets.MaxResolutionLatencyMS {
		violations = append(violations, PerformanceViolation{
			Timestamp:   time.Now(),
			Target:      "MaxResolutionLatency",
			Expected:    pm.targets.MaxResolutionLatencyMS,
			Actual:      summary.AverageResolutionTime,
			Description: "Average resolution time exceeds target",
		})
	}

	// Check success rate
	if summary.SuccessRate < pm.targets.MinSuccessRate {
		violations = append(violations, PerformanceViolation{
			Timestamp:   time.Now(),
			Target:      "MinSuccessRate",
			Expected:    pm.targets.MinSuccessRate,
			Actual:      summary.SuccessRate,
			Description: "Success rate below target",
		})
	}

	// Check manual intervention rate
	manualRate := 0.0
	if summary.TotalConflicts > 0 {
		manualRate = float64(summary.ManualInterventions) / float64(summary.TotalConflicts)
	}
	if manualRate > pm.targets.MaxManualInterventionRate {
		violations = append(violations, PerformanceViolation{
			Timestamp:   time.Now(),
			Target:      "MaxManualInterventionRate",
			Expected:    pm.targets.MaxManualInterventionRate,
			Actual:      manualRate,
			Description: "Manual intervention rate exceeds target",
		})
	}

	// Check pending conflicts
	if summary.PendingConflicts > int64(pm.targets.MaxPendingConflicts) {
		violations = append(violations, PerformanceViolation{
			Timestamp:   time.Now(),
			Target:      "MaxPendingConflicts",
			Expected:    float64(pm.targets.MaxPendingConflicts),
			Actual:      float64(summary.PendingConflicts),
			Description: "Too many pending conflicts",
		})
	}

	pm.violations = append(pm.violations, violations...)
	return violations
}

// GetViolations returns recent violations
func (pm *PerformanceMonitor) GetViolations() []PerformanceViolation {
	pm.mu.RLock()
	defer pm.mu.RUnlock()
	return pm.violations
}

// Helper functions

func average(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}

	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}

func percentile(values []float64, p float64) float64 {
	if len(values) == 0 {
		return 0
	}

	// Simple percentile calculation (should use proper algorithm in production)
	sorted := make([]float64, len(values))
	copy(sorted, values)

	// Simplified sort
	for i := 0; i < len(sorted); i++ {
		for j := i + 1; j < len(sorted); j++ {
			if sorted[i] > sorted[j] {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}

	index := int(float64(len(sorted)-1) * p)
	return sorted[index]
}

// Dashboard provides real-time conflict resolution dashboard
type Dashboard struct {
	mu              sync.RWMutex
	collector       *MetricsCollector
	perfMonitor     *PerformanceMonitor
	detector        *ConflictDetector
	auditLog        *AuditLog
	updateInterval  time.Duration
}

// NewDashboard creates a new dashboard
func NewDashboard(collector *MetricsCollector, perfMon *PerformanceMonitor, detector *ConflictDetector, audit *AuditLog) *Dashboard {
	return &Dashboard{
		collector:      collector,
		perfMonitor:    perfMon,
		detector:       detector,
		auditLog:       audit,
		updateInterval: 5 * time.Second,
	}
}

// GetDashboardData returns current dashboard data
func (d *Dashboard) GetDashboardData(ctx context.Context) *DashboardData {
	d.mu.RLock()
	defer d.mu.RUnlock()

	summary := d.collector.GetMetricsSummary()
	violations := d.perfMonitor.GetViolations()
	pending := d.detector.GetPendingConflicts()
	stats := d.auditLog.GetStatistics()

	return &DashboardData{
		Timestamp:            time.Now(),
		MetricsSummary:       summary,
		PerformanceViolations: violations,
		PendingConflicts:     pending,
		AuditStatistics:      stats,
		Health:               d.calculateHealth(summary, violations),
	}
}

// DashboardData contains dashboard information
type DashboardData struct {
	Timestamp             time.Time
	MetricsSummary        MetricsSummary
	PerformanceViolations []PerformanceViolation
	PendingConflicts      []*Conflict
	AuditStatistics       AuditStatistics
	Health                HealthStatus
}

// HealthStatus represents system health
type HealthStatus struct {
	Overall     string  // "healthy", "degraded", "critical"
	Score       float64 // 0-100
	Issues      []string
	Recommendations []string
}

// calculateHealth calculates overall system health
func (d *Dashboard) calculateHealth(summary MetricsSummary, violations []PerformanceViolation) HealthStatus {
	score := 100.0
	issues := make([]string, 0)
	recommendations := make([]string, 0)

	// Deduct for low success rate
	if summary.SuccessRate < 0.95 {
		score -= (0.95 - summary.SuccessRate) * 100
		issues = append(issues, "Low conflict resolution success rate")
		recommendations = append(recommendations, "Review resolution strategies and policies")
	}

	// Deduct for high pending conflicts
	if summary.PendingConflicts > 50 {
		score -= float64(summary.PendingConflicts-50) * 0.5
		issues = append(issues, "High number of pending conflicts")
		recommendations = append(recommendations, "Increase resolution capacity or review conflict sources")
	}

	// Deduct for performance violations
	score -= float64(len(violations)) * 5
	if len(violations) > 0 {
		issues = append(issues, "Performance targets not met")
		recommendations = append(recommendations, "Optimize resolution strategies or adjust targets")
	}

	// Determine overall status
	overall := "healthy"
	if score < 70 {
		overall = "degraded"
	}
	if score < 50 {
		overall = "critical"
	}

	if score < 0 {
		score = 0
	}

	return HealthStatus{
		Overall:         overall,
		Score:           score,
		Issues:          issues,
		Recommendations: recommendations,
	}
}
