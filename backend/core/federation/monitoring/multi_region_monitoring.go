// Package monitoring implements multi-region monitoring for DWCP v3 federation
// Provides global metrics aggregation, cross-region SLA tracking, and distributed tracing
package monitoring

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// Global metrics for multi-region monitoring
var (
	slaCompliance = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "dwcp_federation_sla_compliance_percent",
			Help: "SLA compliance percentage per region",
		},
		[]string{"region", "sla_type"},
	)

	aggregatedMetrics = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "dwcp_federation_aggregated_metrics",
			Help: "Aggregated metrics across all regions",
		},
		[]string{"metric_name", "aggregation_type"},
	)

	healthScore = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "dwcp_federation_region_health_score",
			Help: "Overall health score for each region (0-100)",
		},
		[]string{"region"},
	)

	traceLatency = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "dwcp_federation_trace_latency_ms",
			Help:    "Distributed trace latency in milliseconds",
			Buckets: []float64{1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000},
		},
		[]string{"source_region", "target_region", "operation"},
	)

	incidentCount = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "dwcp_federation_incidents_total",
			Help: "Total number of incidents detected",
		},
		[]string{"region", "severity", "category"},
	)
)

// MultiRegionMonitor monitors federation health and performance
type MultiRegionMonitor struct {
	mu                 sync.RWMutex
	regions            map[string]*RegionMonitor
	slaDefinitions     map[string]*SLADefinition
	traceCollector     *TraceCollector
	metricsAggregator  *MetricsAggregator
	alertManager       *AlertManager
	dashboardExporter  *DashboardExporter
	stopCh             chan struct{}
	wg                 sync.WaitGroup
	monitoringInterval time.Duration
}

// RegionMonitor monitors a single region
type RegionMonitor struct {
	RegionID          string
	Metrics           *RegionMetrics
	SLAStatus         map[string]*SLAStatus
	HealthChecks      []*HealthCheck
	Incidents         []*Incident
	LastUpdate        time.Time
	CollectionEnabled bool
}

// RegionMetrics contains aggregated metrics for a region
type RegionMetrics struct {
	CPUUtilization     float64
	MemoryUtilization  float64
	StorageUtilization float64
	NetworkUtilization float64
	ActiveVMs          int64
	TotalRequests      int64
	SuccessfulRequests int64
	FailedRequests     int64
	AverageLatency     time.Duration
	P50Latency         time.Duration
	P95Latency         time.Duration
	P99Latency         time.Duration
	ErrorRate          float64
	Throughput         float64 // Requests per second
	Timestamp          time.Time
}

// SLADefinition defines service level objectives
type SLADefinition struct {
	SLAID              string
	Name               string
	Description        string
	MetricName         string
	TargetValue        float64
	ThresholdOperator  string // "less_than", "greater_than", "equals"
	MeasurementWindow  time.Duration
	ComplianceTarget   float64 // Percentage (e.g., 99.9%)
	Severity           string  // "critical", "high", "medium", "low"
	NotificationEmails []string
	Enabled            bool
}

// SLAStatus tracks SLA compliance status
type SLAStatus struct {
	SLAID             string
	CurrentValue      float64
	TargetValue       float64
	CompliancePercent float64
	Status            string // "compliant", "warning", "breach"
	LastBreach        time.Time
	BreachCount       int64
	ConsecutiveBreaches int
	LastCheck         time.Time
	History           []SLADataPoint
}

// SLADataPoint represents a single SLA measurement
type SLADataPoint struct {
	Timestamp  time.Time
	Value      float64
	Compliant  bool
}

// HealthCheck represents a health check result
type HealthCheck struct {
	CheckID      string
	CheckType    string // "http", "tcp", "ping", "custom"
	Target       string
	Status       string // "pass", "fail"
	ResponseTime time.Duration
	Message      string
	Timestamp    time.Time
}

// Incident represents a detected incident
type Incident struct {
	IncidentID   string
	RegionID     string
	Severity     string // "critical", "high", "medium", "low"
	Category     string // "availability", "performance", "security", "capacity"
	Title        string
	Description  string
	Status       string // "open", "acknowledged", "resolved"
	DetectedAt   time.Time
	AcknowledgedAt time.Time
	ResolvedAt   time.Time
	Impact       string
	RootCause    string
	Resolution   string
	Assignee     string
	Tags         []string
}

// TraceCollector collects distributed traces
type TraceCollector struct {
	mu      sync.RWMutex
	traces  map[string]*DistributedTrace
	enabled bool
}

// DistributedTrace represents a cross-region trace
type DistributedTrace struct {
	TraceID      string
	ParentSpanID string
	Spans        []*Span
	StartTime    time.Time
	EndTime      time.Time
	Duration     time.Duration
	Status       string
	Annotations  map[string]string
}

// Span represents a single span in a trace
type Span struct {
	SpanID       string
	ParentSpanID string
	Operation    string
	RegionID     string
	StartTime    time.Time
	EndTime      time.Time
	Duration     time.Duration
	Status       string
	Tags         map[string]string
	Logs         []SpanLog
}

// SpanLog represents a log entry within a span
type SpanLog struct {
	Timestamp time.Time
	Message   string
	Level     string
	Fields    map[string]interface{}
}

// MetricsAggregator aggregates metrics across regions
type MetricsAggregator struct {
	mu                sync.RWMutex
	globalMetrics     *GlobalMetrics
	aggregationRules  []*AggregationRule
	timeSeries        map[string][]*TimeSeriesPoint
}

// GlobalMetrics contains global aggregated metrics
type GlobalMetrics struct {
	TotalRegions       int
	HealthyRegions     int
	DegradedRegions    int
	UnhealthyRegions   int
	TotalVMs           int64
	TotalCPUCores      int64
	TotalMemoryGB      int64
	TotalStorageTB     int64
	GlobalLatency      time.Duration
	GlobalThroughput   float64
	GlobalErrorRate    float64
	AverageSLACompliance float64
	LastUpdate         time.Time
}

// AggregationRule defines how to aggregate metrics
type AggregationRule struct {
	RuleID         string
	MetricName     string
	AggregationType string // "sum", "avg", "min", "max", "count"
	SourceRegions  []string
	OutputName     string
	Enabled        bool
}

// TimeSeriesPoint represents a single time series data point
type TimeSeriesPoint struct {
	Timestamp time.Time
	Value     float64
	Labels    map[string]string
}

// AlertManager manages alerts and notifications
type AlertManager struct {
	mu             sync.RWMutex
	activeAlerts   map[string]*Alert
	alertRules     []*AlertRule
	notifiers      []Notifier
	silences       map[string]*Silence
}

// Alert represents an active alert
type Alert struct {
	AlertID      string
	RuleID       string
	RegionID     string
	Severity     string
	Title        string
	Description  string
	Status       string // "firing", "pending", "resolved"
	StartsAt     time.Time
	EndsAt       time.Time
	Labels       map[string]string
	Annotations  map[string]string
	SilenceID    string
}

// AlertRule defines alert conditions
type AlertRule struct {
	RuleID      string
	Name        string
	Expression  string
	Duration    time.Duration
	Severity    string
	Labels      map[string]string
	Annotations map[string]string
	Enabled     bool
}

// Notifier interface for alert notifications
type Notifier interface {
	Notify(alert *Alert) error
}

// Silence represents an alert silence
type Silence struct {
	SilenceID   string
	Matchers    map[string]string
	StartsAt    time.Time
	EndsAt      time.Time
	CreatedBy   string
	Comment     string
}

// DashboardExporter exports metrics to dashboard systems
type DashboardExporter struct {
	mu            sync.RWMutex
	exportTargets []ExportTarget
	enabled       bool
}

// ExportTarget defines an export destination
type ExportTarget struct {
	TargetID   string
	Type       string // "prometheus", "grafana", "datadog", "cloudwatch"
	Endpoint   string
	APIKey     string
	Enabled    bool
}

// NewMultiRegionMonitor creates a new multi-region monitor
func NewMultiRegionMonitor(cfg *MonitorConfig) (*MultiRegionMonitor, error) {
	if cfg == nil {
		return nil, fmt.Errorf("monitor config cannot be nil")
	}

	mrm := &MultiRegionMonitor{
		regions:            make(map[string]*RegionMonitor),
		slaDefinitions:     make(map[string]*SLADefinition),
		stopCh:             make(chan struct{}),
		monitoringInterval: cfg.MonitoringInterval,
		traceCollector: &TraceCollector{
			traces:  make(map[string]*DistributedTrace),
			enabled: cfg.EnableTracing,
		},
		metricsAggregator: &MetricsAggregator{
			globalMetrics: &GlobalMetrics{},
			timeSeries:    make(map[string][]*TimeSeriesPoint),
		},
		alertManager: &AlertManager{
			activeAlerts: make(map[string]*Alert),
			silences:     make(map[string]*Silence),
		},
		dashboardExporter: &DashboardExporter{
			enabled: cfg.EnableDashboardExport,
		},
	}

	// Initialize regions
	for _, regionID := range cfg.Regions {
		mrm.regions[regionID] = &RegionMonitor{
			RegionID:          regionID,
			Metrics:           &RegionMetrics{},
			SLAStatus:         make(map[string]*SLAStatus),
			CollectionEnabled: true,
		}
	}

	// Initialize SLA definitions
	for _, sla := range cfg.SLADefinitions {
		mrm.slaDefinitions[sla.SLAID] = sla
	}

	return mrm, nil
}

// MonitorConfig defines monitoring configuration
type MonitorConfig struct {
	Regions                []string
	MonitoringInterval     time.Duration
	SLADefinitions         []*SLADefinition
	EnableTracing          bool
	EnableDashboardExport  bool
	AlertNotifiers         []Notifier
}

// Start starts the monitoring system
func (mrm *MultiRegionMonitor) Start(ctx context.Context) error {
	// Start metrics collection
	mrm.wg.Add(1)
	go mrm.collectMetrics(ctx)

	// Start SLA monitoring
	mrm.wg.Add(1)
	go mrm.monitorSLAs(ctx)

	// Start health checks
	mrm.wg.Add(1)
	go mrm.performHealthChecks(ctx)

	// Start metrics aggregation
	mrm.wg.Add(1)
	go mrm.aggregateMetrics(ctx)

	// Start alert evaluation
	mrm.wg.Add(1)
	go mrm.evaluateAlerts(ctx)

	// Start dashboard export
	if mrm.dashboardExporter.enabled {
		mrm.wg.Add(1)
		go mrm.exportToDashboards(ctx)
	}

	return nil
}

// Stop stops the monitoring system
func (mrm *MultiRegionMonitor) Stop() error {
	close(mrm.stopCh)
	mrm.wg.Wait()
	return nil
}

// collectMetrics collects metrics from all regions
func (mrm *MultiRegionMonitor) collectMetrics(ctx context.Context) {
	defer mrm.wg.Done()

	ticker := time.NewTicker(mrm.monitoringInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-mrm.stopCh:
			return
		case <-ticker.C:
			mrm.collectRegionMetrics()
		}
	}
}

// collectRegionMetrics collects metrics from each region
func (mrm *MultiRegionMonitor) collectRegionMetrics() {
	mrm.mu.Lock()
	defer mrm.mu.Unlock()

	for _, monitor := range mrm.regions {
		if !monitor.CollectionEnabled {
			continue
		}

		// TODO: Implement actual metrics collection from regions
		// For now, simulate with dummy data
		monitor.Metrics = &RegionMetrics{
			CPUUtilization:     70.0,
			MemoryUtilization:  65.0,
			StorageUtilization: 55.0,
			NetworkUtilization: 40.0,
			ActiveVMs:          100,
			TotalRequests:      1000000,
			SuccessfulRequests: 999000,
			FailedRequests:     1000,
			AverageLatency:     25 * time.Millisecond,
			P99Latency:         95 * time.Millisecond,
			ErrorRate:          0.1,
			Throughput:         10000.0,
			Timestamp:          time.Now(),
		}

		monitor.LastUpdate = time.Now()
	}
}

// monitorSLAs monitors SLA compliance
func (mrm *MultiRegionMonitor) monitorSLAs(ctx context.Context) {
	defer mrm.wg.Done()

	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-mrm.stopCh:
			return
		case <-ticker.C:
			mrm.checkSLACompliance()
		}
	}
}

// checkSLACompliance checks SLA compliance for all regions
func (mrm *MultiRegionMonitor) checkSLACompliance() {
	mrm.mu.Lock()
	defer mrm.mu.Unlock()

	for slaID, sla := range mrm.slaDefinitions {
		if !sla.Enabled {
			continue
		}

		for regionID, monitor := range mrm.regions {
			status := mrm.evaluateSLA(sla, monitor.Metrics)

			// Update SLA status
			if monitor.SLAStatus[slaID] == nil {
				monitor.SLAStatus[slaID] = &SLAStatus{
					SLAID:       slaID,
					TargetValue: sla.TargetValue,
					History:     make([]SLADataPoint, 0, 100),
				}
			}

			slaStatus := monitor.SLAStatus[slaID]
			slaStatus.CurrentValue = status.Value
			slaStatus.CompliancePercent = status.CompliancePercent
			slaStatus.Status = status.Status
			slaStatus.LastCheck = time.Now()

			// Add to history
			slaStatus.History = append(slaStatus.History, SLADataPoint{
				Timestamp: time.Now(),
				Value:     status.Value,
				Compliant: status.Status == "compliant",
			})

			// Keep only last 1000 data points
			if len(slaStatus.History) > 1000 {
				slaStatus.History = slaStatus.History[1:]
			}

			// Update breach count
			if status.Status == "breach" {
				slaStatus.BreachCount++
				slaStatus.ConsecutiveBreaches++
				slaStatus.LastBreach = time.Now()

				// Create incident for SLA breach
				if slaStatus.ConsecutiveBreaches >= 3 {
					mrm.createSLABreachIncident(regionID, sla, slaStatus)
				}
			} else {
				slaStatus.ConsecutiveBreaches = 0
			}

			// Export metric
			slaCompliance.WithLabelValues(regionID, sla.Name).Set(status.CompliancePercent)
		}
	}
}

// SLAEvaluation represents SLA evaluation result
type SLAEvaluation struct {
	Value             float64
	CompliancePercent float64
	Status            string
}

// evaluateSLA evaluates SLA against metrics
func (mrm *MultiRegionMonitor) evaluateSLA(sla *SLADefinition, metrics *RegionMetrics) SLAEvaluation {
	var value float64
	var compliant bool

	// Extract relevant metric
	switch sla.MetricName {
	case "cpu_utilization":
		value = metrics.CPUUtilization
	case "memory_utilization":
		value = metrics.MemoryUtilization
	case "error_rate":
		value = metrics.ErrorRate
	case "average_latency":
		value = float64(metrics.AverageLatency.Milliseconds())
	case "p99_latency":
		value = float64(metrics.P99Latency.Milliseconds())
	default:
		value = 0
	}

	// Check compliance
	switch sla.ThresholdOperator {
	case "less_than":
		compliant = value < sla.TargetValue
	case "greater_than":
		compliant = value > sla.TargetValue
	case "equals":
		compliant = value == sla.TargetValue
	default:
		compliant = false
	}

	// Calculate compliance percentage (simplified)
	compliancePercent := 99.9 // Default high compliance
	if !compliant {
		compliancePercent = 90.0
	}

	status := "compliant"
	if compliancePercent < sla.ComplianceTarget {
		if compliancePercent < sla.ComplianceTarget-5 {
			status = "breach"
		} else {
			status = "warning"
		}
	}

	return SLAEvaluation{
		Value:             value,
		CompliancePercent: compliancePercent,
		Status:            status,
	}
}

// createSLABreachIncident creates an incident for SLA breach
func (mrm *MultiRegionMonitor) createSLABreachIncident(regionID string, sla *SLADefinition, status *SLAStatus) {
	incident := &Incident{
		IncidentID:  fmt.Sprintf("INC-%s-%s-%d", regionID, sla.SLAID, time.Now().Unix()),
		RegionID:    regionID,
		Severity:    sla.Severity,
		Category:    "performance",
		Title:       fmt.Sprintf("SLA Breach: %s", sla.Name),
		Description: fmt.Sprintf("SLA %s has been breached. Current: %.2f, Target: %.2f", sla.Name, status.CurrentValue, status.TargetValue),
		Status:      "open",
		DetectedAt:  time.Now(),
		Tags:        []string{"sla_breach", sla.Name},
	}

	monitor := mrm.regions[regionID]
	monitor.Incidents = append(monitor.Incidents, incident)

	incidentCount.WithLabelValues(regionID, incident.Severity, incident.Category).Inc()
}

// performHealthChecks performs health checks on all regions
func (mrm *MultiRegionMonitor) performHealthChecks(ctx context.Context) {
	defer mrm.wg.Done()

	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-mrm.stopCh:
			return
		case <-ticker.C:
			mrm.checkRegionHealth()
		}
	}
}

// checkRegionHealth checks health of all regions
func (mrm *MultiRegionMonitor) checkRegionHealth() {
	mrm.mu.Lock()
	defer mrm.mu.Unlock()

	for regionID, monitor := range mrm.regions {
		// Calculate health score
		score := mrm.calculateHealthScore(monitor)

		// Export metric
		healthScore.WithLabelValues(regionID).Set(score)

		// Create health check result
		status := "pass"
		if score < 70 {
			status = "fail"
		}

		healthCheck := &HealthCheck{
			CheckID:      fmt.Sprintf("HC-%s-%d", regionID, time.Now().Unix()),
			CheckType:    "composite",
			Target:       regionID,
			Status:       status,
			ResponseTime: 10 * time.Millisecond,
			Message:      fmt.Sprintf("Health score: %.2f", score),
			Timestamp:    time.Now(),
		}

		monitor.HealthChecks = append(monitor.HealthChecks, healthCheck)

		// Keep only last 100 health checks
		if len(monitor.HealthChecks) > 100 {
			monitor.HealthChecks = monitor.HealthChecks[1:]
		}
	}
}

// calculateHealthScore calculates overall health score for a region
func (mrm *MultiRegionMonitor) calculateHealthScore(monitor *RegionMonitor) float64 {
	if monitor.Metrics == nil {
		return 0
	}

	score := 100.0

	// Deduct for high utilization
	if monitor.Metrics.CPUUtilization > 90 {
		score -= 20
	} else if monitor.Metrics.CPUUtilization > 80 {
		score -= 10
	}

	if monitor.Metrics.MemoryUtilization > 90 {
		score -= 20
	} else if monitor.Metrics.MemoryUtilization > 80 {
		score -= 10
	}

	// Deduct for high error rate
	if monitor.Metrics.ErrorRate > 1.0 {
		score -= 30
	} else if monitor.Metrics.ErrorRate > 0.5 {
		score -= 15
	}

	// Deduct for high latency
	if monitor.Metrics.P99Latency > 100*time.Millisecond {
		score -= 20
	} else if monitor.Metrics.P99Latency > 50*time.Millisecond {
		score -= 10
	}

	return math.Max(0, score)
}

// aggregateMetrics aggregates metrics across all regions
func (mrm *MultiRegionMonitor) aggregateMetrics(ctx context.Context) {
	defer mrm.wg.Done()

	ticker := time.NewTicker(1 * time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-mrm.stopCh:
			return
		case <-ticker.C:
			mrm.computeGlobalMetrics()
		}
	}
}

// computeGlobalMetrics computes global aggregated metrics
func (mrm *MultiRegionMonitor) computeGlobalMetrics() {
	mrm.mu.RLock()
	defer mrm.mu.RUnlock()

	global := &GlobalMetrics{
		TotalRegions:   len(mrm.regions),
		LastUpdate:     time.Now(),
	}

	var totalLatency time.Duration
	var totalThroughput, totalErrorRate float64
	var slaComplianceSum float64
	slaCount := 0

	for _, monitor := range mrm.regions {
		// Count health status
		score := mrm.calculateHealthScore(monitor)
		if score >= 80 {
			global.HealthyRegions++
		} else if score >= 60 {
			global.DegradedRegions++
		} else {
			global.UnhealthyRegions++
		}

		// Aggregate metrics
		if monitor.Metrics != nil {
			global.TotalVMs += monitor.Metrics.ActiveVMs
			totalLatency += monitor.Metrics.AverageLatency
			totalThroughput += monitor.Metrics.Throughput
			totalErrorRate += monitor.Metrics.ErrorRate
		}

		// Aggregate SLA compliance
		for _, slaStatus := range monitor.SLAStatus {
			slaComplianceSum += slaStatus.CompliancePercent
			slaCount++
		}
	}

	// Calculate averages
	if len(mrm.regions) > 0 {
		global.GlobalLatency = totalLatency / time.Duration(len(mrm.regions))
		global.GlobalThroughput = totalThroughput
		global.GlobalErrorRate = totalErrorRate / float64(len(mrm.regions))
	}

	if slaCount > 0 {
		global.AverageSLACompliance = slaComplianceSum / float64(slaCount)
	}

	mrm.metricsAggregator.mu.Lock()
	mrm.metricsAggregator.globalMetrics = global
	mrm.metricsAggregator.mu.Unlock()

	// Export aggregated metrics
	aggregatedMetrics.WithLabelValues("total_regions", "count").Set(float64(global.TotalRegions))
	aggregatedMetrics.WithLabelValues("healthy_regions", "count").Set(float64(global.HealthyRegions))
	aggregatedMetrics.WithLabelValues("total_vms", "count").Set(float64(global.TotalVMs))
	aggregatedMetrics.WithLabelValues("global_latency_ms", "avg").Set(float64(global.GlobalLatency.Milliseconds()))
	aggregatedMetrics.WithLabelValues("global_throughput", "sum").Set(global.GlobalThroughput)
	aggregatedMetrics.WithLabelValues("sla_compliance", "avg").Set(global.AverageSLACompliance)
}

// evaluateAlerts evaluates alert rules
func (mrm *MultiRegionMonitor) evaluateAlerts(ctx context.Context) {
	defer mrm.wg.Done()

	ticker := time.NewTicker(15 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-mrm.stopCh:
			return
		case <-ticker.C:
			mrm.checkAlertRules()
		}
	}
}

// checkAlertRules checks all alert rules
func (mrm *MultiRegionMonitor) checkAlertRules() {
	// TODO: Implement alert rule evaluation
	// For now, this is a placeholder
}

// exportToDashboards exports metrics to dashboard systems
func (mrm *MultiRegionMonitor) exportToDashboards(ctx context.Context) {
	defer mrm.wg.Done()

	ticker := time.NewTicker(1 * time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-mrm.stopCh:
			return
		case <-ticker.C:
			mrm.performExport()
		}
	}
}

// performExport performs export to configured targets
func (mrm *MultiRegionMonitor) performExport() {
	// TODO: Implement actual export to dashboard systems
	// For now, this is a placeholder
}

// RecordTrace records a distributed trace
func (mrm *MultiRegionMonitor) RecordTrace(trace *DistributedTrace) error {
	if !mrm.traceCollector.enabled {
		return nil
	}

	mrm.traceCollector.mu.Lock()
	defer mrm.traceCollector.mu.Unlock()

	mrm.traceCollector.traces[trace.TraceID] = trace

	// Keep only last 10000 traces
	if len(mrm.traceCollector.traces) > 10000 {
		// Remove oldest traces
		// TODO: Implement proper LRU eviction
	}

	// Export trace latency metric
	if len(trace.Spans) >= 2 {
		sourceRegion := trace.Spans[0].RegionID
		targetRegion := trace.Spans[len(trace.Spans)-1].RegionID
		operation := trace.Spans[0].Operation

		traceLatency.WithLabelValues(sourceRegion, targetRegion, operation).
			Observe(float64(trace.Duration.Milliseconds()))
	}

	return nil
}

// GetGlobalMetrics returns global aggregated metrics
func (mrm *MultiRegionMonitor) GetGlobalMetrics() *GlobalMetrics {
	mrm.metricsAggregator.mu.RLock()
	defer mrm.metricsAggregator.mu.RUnlock()

	// Return a copy
	global := *mrm.metricsAggregator.globalMetrics
	return &global
}

// GetRegionMetrics returns metrics for a specific region
func (mrm *MultiRegionMonitor) GetRegionMetrics(regionID string) (*RegionMetrics, error) {
	mrm.mu.RLock()
	defer mrm.mu.RUnlock()

	monitor, exists := mrm.regions[regionID]
	if !exists {
		return nil, fmt.Errorf("region not found: %s", regionID)
	}

	// Return a copy
	metrics := *monitor.Metrics
	return &metrics, nil
}

// GetRegionIncidents returns incidents for a specific region
func (mrm *MultiRegionMonitor) GetRegionIncidents(regionID string) ([]*Incident, error) {
	mrm.mu.RLock()
	defer mrm.mu.RUnlock()

	monitor, exists := mrm.regions[regionID]
	if !exists {
		return nil, fmt.Errorf("region not found: %s", regionID)
	}

	return monitor.Incidents, nil
}
