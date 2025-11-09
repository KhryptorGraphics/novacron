package sla

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// SLA represents a Service Level Agreement
type SLA struct {
	ID                 string        `json:"id"`
	Name               string        `json:"name"`
	Description        string        `json:"description"`
	AvailabilityTarget float64       `json:"availability_target"` // 99.95%
	LatencyTarget      time.Duration `json:"latency_target"`      // P95 < 100ms
	ThroughputTarget   int64         `json:"throughput_target"`   // req/sec
	ErrorRateTarget    float64       `json:"error_rate_target"`   // <0.1%
	MeasurementWindow  time.Duration `json:"measurement_window"`  // 1 month
	ErrorBudget        float64       `json:"error_budget"`        // Calculated from availability
	CreatedAt          time.Time     `json:"created_at"`
	UpdatedAt          time.Time     `json:"updated_at"`
	Enabled            bool          `json:"enabled"`
}

// SLAMetrics represents actual SLA metrics
type SLAMetrics struct {
	SLAID                string        `json:"sla_id"`
	MeasurementStart     time.Time     `json:"measurement_start"`
	MeasurementEnd       time.Time     `json:"measurement_end"`
	Availability         float64       `json:"availability"`         // Actual availability %
	LatencyP95           time.Duration `json:"latency_p95"`          // Actual P95 latency
	LatencyP99           time.Duration `json:"latency_p99"`          // Actual P99 latency
	Throughput           int64         `json:"throughput"`           // Actual throughput
	ErrorRate            float64       `json:"error_rate"`           // Actual error rate
	TotalRequests        int64         `json:"total_requests"`
	SuccessfulRequests   int64         `json:"successful_requests"`
	FailedRequests       int64         `json:"failed_requests"`
	Uptime               time.Duration `json:"uptime"`
	Downtime             time.Duration `json:"downtime"`
	ErrorBudgetRemaining float64       `json:"error_budget_remaining"` // Percentage
}

// SLAViolation represents an SLA violation
type SLAViolation struct {
	ID           string        `json:"id"`
	SLAID        string        `json:"sla_id"`
	MetricType   string        `json:"metric_type"` // availability, latency, throughput, error-rate
	TargetValue  float64       `json:"target_value"`
	ActualValue  float64       `json:"actual_value"`
	Severity     string        `json:"severity"` // minor, major, critical
	DetectedAt   time.Time     `json:"detected_at"`
	ResolvedAt   *time.Time    `json:"resolved_at,omitempty"`
	Duration     time.Duration `json:"duration"`
	Impact       string        `json:"impact"`
	Notifications []string     `json:"notifications"` // Notification channels used
}

// SLAReport represents an SLA compliance report
type SLAReport struct {
	SLAID             string          `json:"sla_id"`
	ReportPeriod      string          `json:"report_period"` // daily, weekly, monthly
	GeneratedAt       time.Time       `json:"generated_at"`
	Metrics           *SLAMetrics     `json:"metrics"`
	ComplianceStatus  string          `json:"compliance_status"` // compliant, at-risk, violated
	Violations        []*SLAViolation `json:"violations"`
	Recommendations   []string        `json:"recommendations"`
	TrendAnalysis     *TrendAnalysis  `json:"trend_analysis"`
}

// TrendAnalysis represents trend analysis
type TrendAnalysis struct {
	AvailabilityTrend string `json:"availability_trend"` // improving, stable, degrading
	LatencyTrend      string `json:"latency_trend"`
	ThroughputTrend   string `json:"throughput_trend"`
	ErrorRateTrend    string `json:"error_rate_trend"`
}

// SLAManager manages SLAs and tracking
type SLAManager struct {
	mu                  sync.RWMutex
	slas                map[string]*SLA
	metricsCollector    *MetricsCollector
	violationDetector   *ViolationDetector
	reportGenerator     *ReportGenerator
	errorBudgetTracker  *ErrorBudgetTracker
	notificationManager *NotificationManager
	dashboardEnabled    bool
	metrics             *SLAManagerMetrics
}

// MetricsCollector collects SLA metrics
type MetricsCollector struct {
	mu              sync.RWMutex
	slaMetrics      map[string]*SLAMetrics // SLAID -> current metrics
	dataPoints      map[string][]DataPoint
	collectionInterval time.Duration
	running         bool
	stopCh          chan struct{}
}

// DataPoint represents a single metric data point
type DataPoint struct {
	Timestamp   time.Time     `json:"timestamp"`
	Latency     time.Duration `json:"latency"`
	Success     bool          `json:"success"`
	RequestSize int64         `json:"request_size"`
}

// ViolationDetector detects SLA violations
type ViolationDetector struct {
	mu         sync.RWMutex
	violations map[string][]*SLAViolation // SLAID -> violations
	autoNotify bool
}

// ReportGenerator generates SLA reports
type ReportGenerator struct {
	mu            sync.RWMutex
	reports       map[string][]*SLAReport // SLAID -> reports
	autoSchedule  bool
	scheduleFrequency time.Duration
}

// ErrorBudgetTracker tracks error budgets
type ErrorBudgetTracker struct {
	mu           sync.RWMutex
	budgets      map[string]*ErrorBudget // SLAID -> budget
}

type ErrorBudget struct {
	SLAID           string    `json:"sla_id"`
	TotalBudget     float64   `json:"total_budget"`     // Minutes allowed downtime
	ConsumedBudget  float64   `json:"consumed_budget"`  // Minutes consumed
	RemainingBudget float64   `json:"remaining_budget"` // Minutes remaining
	BudgetPercent   float64   `json:"budget_percent"`   // Percentage remaining
	LastUpdated     time.Time `json:"last_updated"`
}

// NotificationManager manages SLA notifications
type NotificationManager struct {
	mu       sync.RWMutex
	channels []string // email, slack, pagerduty
	enabled  bool
}

// SLAManagerMetrics tracks manager metrics
type SLAManagerMetrics struct {
	mu                  sync.RWMutex
	TotalSLAs           int64
	ActiveSLAs          int64
	TotalViolations     int64
	ReportsGenerated    int64
	NotificationsSent   int64
	AverageCompliance   float64
}

// NewSLAManager creates a new SLA manager
func NewSLAManager(dashboardEnabled bool, notificationChannels []string) *SLAManager {
	return &SLAManager{
		slas:                make(map[string]*SLA),
		metricsCollector:    newMetricsCollector(1 * time.Minute),
		violationDetector:   newViolationDetector(true),
		reportGenerator:     newReportGenerator(true, 24*time.Hour),
		errorBudgetTracker:  newErrorBudgetTracker(),
		notificationManager: newNotificationManager(notificationChannels),
		dashboardEnabled:    dashboardEnabled,
		metrics:             &SLAManagerMetrics{},
	}
}

func newMetricsCollector(interval time.Duration) *MetricsCollector {
	return &MetricsCollector{
		slaMetrics:         make(map[string]*SLAMetrics),
		dataPoints:         make(map[string][]DataPoint),
		collectionInterval: interval,
		stopCh:             make(chan struct{}),
	}
}

func newViolationDetector(autoNotify bool) *ViolationDetector {
	return &ViolationDetector{
		violations: make(map[string][]*SLAViolation),
		autoNotify: autoNotify,
	}
}

func newReportGenerator(autoSchedule bool, frequency time.Duration) *ReportGenerator {
	return &ReportGenerator{
		reports:           make(map[string][]*SLAReport),
		autoSchedule:      autoSchedule,
		scheduleFrequency: frequency,
	}
}

func newErrorBudgetTracker() *ErrorBudgetTracker {
	return &ErrorBudgetTracker{
		budgets: make(map[string]*ErrorBudget),
	}
}

func newNotificationManager(channels []string) *NotificationManager {
	return &NotificationManager{
		channels: channels,
		enabled:  len(channels) > 0,
	}
}

// CreateSLA creates a new SLA
func (sm *SLAManager) CreateSLA(ctx context.Context, sla *SLA) error {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	if sla.ID == "" {
		sla.ID = fmt.Sprintf("sla-%d", time.Now().UnixNano())
	}

	// Calculate error budget from availability target
	// Error budget = (1 - availability) * measurement window
	sla.ErrorBudget = (1 - sla.AvailabilityTarget) * float64(sla.MeasurementWindow.Minutes())

	sla.CreatedAt = time.Now()
	sla.UpdatedAt = time.Now()
	sla.Enabled = true

	sm.slas[sla.ID] = sla

	// Initialize error budget
	sm.errorBudgetTracker.initializeBudget(sla)

	sm.metrics.mu.Lock()
	sm.metrics.TotalSLAs++
	sm.metrics.ActiveSLAs++
	sm.metrics.mu.Unlock()

	return nil
}

// RecordMetric records a metric data point
func (sm *SLAManager) RecordMetric(ctx context.Context, slaID string, dataPoint DataPoint) error {
	sm.metricsCollector.mu.Lock()
	defer sm.metricsCollector.mu.Unlock()

	if _, exists := sm.metricsCollector.dataPoints[slaID]; !exists {
		sm.metricsCollector.dataPoints[slaID] = make([]DataPoint, 0)
	}

	dataPoint.Timestamp = time.Now()
	sm.metricsCollector.dataPoints[slaID] = append(sm.metricsCollector.dataPoints[slaID], dataPoint)

	// Update current metrics
	sm.updateCurrentMetrics(slaID)

	return nil
}

// updateCurrentMetrics updates current SLA metrics
func (sm *SLAManager) updateCurrentMetrics(slaID string) {
	sm.mu.RLock()
	sla, exists := sm.slas[slaID]
	sm.mu.RUnlock()

	if !exists {
		return
	}

	dataPoints := sm.metricsCollector.dataPoints[slaID]
	if len(dataPoints) == 0 {
		return
	}

	// Calculate metrics from data points
	metrics := &SLAMetrics{
		SLAID:            slaID,
		MeasurementStart: time.Now().Add(-sla.MeasurementWindow),
		MeasurementEnd:   time.Now(),
	}

	successCount := int64(0)
	failCount := int64(0)
	latencies := make([]time.Duration, 0)

	for _, dp := range dataPoints {
		if dp.Success {
			successCount++
		} else {
			failCount++
		}
		latencies = append(latencies, dp.Latency)
	}

	metrics.TotalRequests = successCount + failCount
	metrics.SuccessfulRequests = successCount
	metrics.FailedRequests = failCount

	if metrics.TotalRequests > 0 {
		metrics.Availability = float64(successCount) / float64(metrics.TotalRequests) * 100
		metrics.ErrorRate = float64(failCount) / float64(metrics.TotalRequests) * 100
	}

	// Calculate P95 and P99 latency
	if len(latencies) > 0 {
		metrics.LatencyP95 = calculatePercentile(latencies, 0.95)
		metrics.LatencyP99 = calculatePercentile(latencies, 0.99)
	}

	// Calculate throughput (requests per second)
	windowSeconds := sla.MeasurementWindow.Seconds()
	if windowSeconds > 0 {
		metrics.Throughput = int64(float64(metrics.TotalRequests) / windowSeconds)
	}

	// Calculate error budget remaining
	sm.errorBudgetTracker.mu.RLock()
	budget, exists := sm.errorBudgetTracker.budgets[slaID]
	sm.errorBudgetTracker.mu.RUnlock()

	if exists {
		metrics.ErrorBudgetRemaining = budget.BudgetPercent
	}

	sm.metricsCollector.slaMetrics[slaID] = metrics

	// Check for violations
	sm.checkSLACompliance(sla, metrics)
}

// calculatePercentile calculates percentile latency
func calculatePercentile(latencies []time.Duration, percentile float64) time.Duration {
	if len(latencies) == 0 {
		return 0
	}

	// Simple percentile calculation (in production, use proper algorithm)
	index := int(float64(len(latencies)) * percentile)
	if index >= len(latencies) {
		index = len(latencies) - 1
	}

	return latencies[index]
}

// checkSLACompliance checks if SLA is being met
func (sm *SLAManager) checkSLACompliance(sla *SLA, metrics *SLAMetrics) {
	violations := make([]*SLAViolation, 0)

	// Check availability
	if metrics.Availability < sla.AvailabilityTarget*100 {
		violation := &SLAViolation{
			ID:          fmt.Sprintf("violation-%d", time.Now().UnixNano()),
			SLAID:       sla.ID,
			MetricType:  "availability",
			TargetValue: sla.AvailabilityTarget * 100,
			ActualValue: metrics.Availability,
			Severity:    sm.calculateSeverity(sla.AvailabilityTarget*100, metrics.Availability),
			DetectedAt:  time.Now(),
			Impact:      fmt.Sprintf("Availability below target: %.2f%% vs %.2f%%", metrics.Availability, sla.AvailabilityTarget*100),
		}
		violations = append(violations, violation)
	}

	// Check latency
	if metrics.LatencyP95 > sla.LatencyTarget {
		violation := &SLAViolation{
			ID:          fmt.Sprintf("violation-%d", time.Now().UnixNano()),
			SLAID:       sla.ID,
			MetricType:  "latency",
			TargetValue: float64(sla.LatencyTarget.Milliseconds()),
			ActualValue: float64(metrics.LatencyP95.Milliseconds()),
			Severity:    "major",
			DetectedAt:  time.Now(),
			Impact:      fmt.Sprintf("Latency P95 above target: %v vs %v", metrics.LatencyP95, sla.LatencyTarget),
		}
		violations = append(violations, violation)
	}

	// Check throughput
	if metrics.Throughput < sla.ThroughputTarget {
		violation := &SLAViolation{
			ID:          fmt.Sprintf("violation-%d", time.Now().UnixNano()),
			SLAID:       sla.ID,
			MetricType:  "throughput",
			TargetValue: float64(sla.ThroughputTarget),
			ActualValue: float64(metrics.Throughput),
			Severity:    "minor",
			DetectedAt:  time.Now(),
			Impact:      fmt.Sprintf("Throughput below target: %d vs %d req/sec", metrics.Throughput, sla.ThroughputTarget),
		}
		violations = append(violations, violation)
	}

	// Check error rate
	if metrics.ErrorRate > sla.ErrorRateTarget*100 {
		violation := &SLAViolation{
			ID:          fmt.Sprintf("violation-%d", time.Now().UnixNano()),
			SLAID:       sla.ID,
			MetricType:  "error-rate",
			TargetValue: sla.ErrorRateTarget * 100,
			ActualValue: metrics.ErrorRate,
			Severity:    "critical",
			DetectedAt:  time.Now(),
			Impact:      fmt.Sprintf("Error rate above target: %.2f%% vs %.2f%%", metrics.ErrorRate, sla.ErrorRateTarget*100),
		}
		violations = append(violations, violation)
	}

	// Record violations
	if len(violations) > 0 {
		sm.violationDetector.mu.Lock()
		if _, exists := sm.violationDetector.violations[sla.ID]; !exists {
			sm.violationDetector.violations[sla.ID] = make([]*SLAViolation, 0)
		}
		sm.violationDetector.violations[sla.ID] = append(sm.violationDetector.violations[sla.ID], violations...)
		sm.violationDetector.mu.Unlock()

		sm.metrics.mu.Lock()
		sm.metrics.TotalViolations += int64(len(violations))
		sm.metrics.mu.Unlock()

		// Send notifications
		if sm.notificationManager.enabled {
			for _, violation := range violations {
				sm.notificationManager.sendNotification(violation)
			}
		}
	}
}

// calculateSeverity calculates violation severity
func (sm *SLAManager) calculateSeverity(target, actual float64) string {
	diff := target - actual

	if diff > 5 {
		return "critical"
	} else if diff > 2 {
		return "major"
	} else {
		return "minor"
	}
}

// GenerateReport generates an SLA report
func (sm *SLAManager) GenerateReport(ctx context.Context, slaID string, period string) (*SLAReport, error) {
	sm.mu.RLock()
	sla, exists := sm.slas[slaID]
	sm.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("SLA %s not found", slaID)
	}

	sm.metricsCollector.mu.RLock()
	metrics, metricsExist := sm.metricsCollector.slaMetrics[slaID]
	sm.metricsCollector.mu.RUnlock()

	if !metricsExist {
		metrics = &SLAMetrics{SLAID: slaID}
	}

	sm.violationDetector.mu.RLock()
	violations := sm.violationDetector.violations[slaID]
	sm.violationDetector.mu.RUnlock()

	report := &SLAReport{
		SLAID:         slaID,
		ReportPeriod:  period,
		GeneratedAt:   time.Now(),
		Metrics:       metrics,
		Violations:    violations,
		TrendAnalysis: sm.analyzeTrends(slaID),
	}

	// Determine compliance status
	if len(violations) == 0 {
		report.ComplianceStatus = "compliant"
	} else {
		criticalViolations := 0
		for _, v := range violations {
			if v.Severity == "critical" {
				criticalViolations++
			}
		}

		if criticalViolations > 0 {
			report.ComplianceStatus = "violated"
		} else {
			report.ComplianceStatus = "at-risk"
		}
	}

	// Generate recommendations
	report.Recommendations = sm.generateRecommendations(sla, metrics, violations)

	// Store report
	sm.reportGenerator.mu.Lock()
	if _, exists := sm.reportGenerator.reports[slaID]; !exists {
		sm.reportGenerator.reports[slaID] = make([]*SLAReport, 0)
	}
	sm.reportGenerator.reports[slaID] = append(sm.reportGenerator.reports[slaID], report)
	sm.reportGenerator.mu.Unlock()

	sm.metrics.mu.Lock()
	sm.metrics.ReportsGenerated++
	sm.metrics.mu.Unlock()

	return report, nil
}

// analyzeTrends analyzes SLA trends
func (sm *SLAManager) analyzeTrends(slaID string) *TrendAnalysis {
	// Simplified trend analysis
	return &TrendAnalysis{
		AvailabilityTrend: "stable",
		LatencyTrend:      "stable",
		ThroughputTrend:   "stable",
		ErrorRateTrend:    "improving",
	}
}

// generateRecommendations generates recommendations
func (sm *SLAManager) generateRecommendations(sla *SLA, metrics *SLAMetrics, violations []*SLAViolation) []string {
	recommendations := make([]string, 0)

	if metrics.Availability < sla.AvailabilityTarget*100 {
		recommendations = append(recommendations, "Investigate and resolve availability issues")
		recommendations = append(recommendations, "Consider implementing redundancy and failover mechanisms")
	}

	if metrics.LatencyP95 > sla.LatencyTarget {
		recommendations = append(recommendations, "Optimize application performance to reduce latency")
		recommendations = append(recommendations, "Review database query performance")
	}

	if metrics.ErrorRate > sla.ErrorRateTarget*100 {
		recommendations = append(recommendations, "Investigate root cause of errors")
		recommendations = append(recommendations, "Implement better error handling and retry logic")
	}

	if len(violations) > 5 {
		recommendations = append(recommendations, "Frequent violations detected - review SLA targets and infrastructure capacity")
	}

	return recommendations
}

// initializeBudget initializes error budget for SLA
func (ebt *ErrorBudgetTracker) initializeBudget(sla *SLA) {
	ebt.mu.Lock()
	defer ebt.mu.Unlock()

	budget := &ErrorBudget{
		SLAID:           sla.ID,
		TotalBudget:     sla.ErrorBudget,
		ConsumedBudget:  0,
		RemainingBudget: sla.ErrorBudget,
		BudgetPercent:   100.0,
		LastUpdated:     time.Now(),
	}

	ebt.budgets[sla.ID] = budget
}

// sendNotification sends violation notification
func (nm *NotificationManager) sendNotification(violation *SLAViolation) {
	nm.mu.Lock()
	defer nm.mu.Unlock()

	if !nm.enabled {
		return
	}

	// In production, this would send actual notifications
	for _, channel := range nm.channels {
		fmt.Printf("Sending notification to %s: %s violation detected\n", channel, violation.MetricType)
	}

	violation.Notifications = nm.channels
}

// GetMetrics returns SLA manager metrics
func (sm *SLAManager) GetMetrics() *SLAManagerMetrics {
	sm.metrics.mu.RLock()
	defer sm.metrics.mu.RUnlock()

	return &SLAManagerMetrics{
		TotalSLAs:         sm.metrics.TotalSLAs,
		ActiveSLAs:        sm.metrics.ActiveSLAs,
		TotalViolations:   sm.metrics.TotalViolations,
		ReportsGenerated:  sm.metrics.ReportsGenerated,
		NotificationsSent: sm.metrics.NotificationsSent,
		AverageCompliance: sm.metrics.AverageCompliance,
	}
}

// GetSLAMetrics returns current metrics for an SLA
func (sm *SLAManager) GetSLAMetrics(ctx context.Context, slaID string) (*SLAMetrics, error) {
	sm.metricsCollector.mu.RLock()
	defer sm.metricsCollector.mu.RUnlock()

	metrics, exists := sm.metricsCollector.slaMetrics[slaID]
	if !exists {
		return nil, fmt.Errorf("no metrics found for SLA %s", slaID)
	}

	return metrics, nil
}
