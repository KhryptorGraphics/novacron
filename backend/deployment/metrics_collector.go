package deployment

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// DeploymentMetrics handles comprehensive deployment metrics tracking and monitoring
type DeploymentMetrics struct {
	config              *DeploymentMetricsConfig
	timeSeriesDB        *TimeSeriesDB
	alertManager        *AlertManager
	dashboardManager    *DashboardManager
	metricsAggregator   *MetricsAggregator
	anomalyDetector     *AnomalyDetector
	reportGenerator     *ReportGenerator
	
	// Synchronization
	mu                  sync.RWMutex
	activeCollectors    map[string]*MetricsCollector
	metricsSeries       map[string]*MetricsSeries
	alerts              map[string]*Alert
	
	// Prometheus metrics
	deploymentFrequency prometheus.CounterVec
	deploymentDuration  prometheus.HistogramVec
	deploymentSuccess   prometheus.CounterVec
	leadTime           prometheus.HistogramVec
	mttr               prometheus.HistogramVec
	changeFailureRate  prometheus.GaugeVec
	recoveryTime       prometheus.HistogramVec
}

// DeploymentMetricsConfig holds configuration for deployment metrics
type DeploymentMetricsConfig struct {
	TimeSeriesDB      *TimeSeriesDBConfig      `json:"timeseries_db"`
	Alerting          *AlertingConfig          `json:"alerting"`
	Dashboards        *DashboardConfig         `json:"dashboards"`
	Aggregation       *AggregationConfig       `json:"aggregation"`
	AnomalyDetection  *AnomalyDetectionConfig  `json:"anomaly_detection"`
	Reporting         *ReportingConfig         `json:"reporting"`
	
	// Collection settings
	MetricsInterval       time.Duration            `json:"metrics_interval"`
	RetentionPeriod       time.Duration            `json:"retention_period"`
	BatchSize            int                      `json:"batch_size"`
	BufferSize           int                      `json:"buffer_size"`
	
	// DORA metrics configuration
	DORAMetrics          *DORAMetricsConfig       `json:"dora_metrics"`
	SLOConfig            *SLOConfig               `json:"slo_config"`
	CustomMetrics        []*CustomMetricConfig    `json:"custom_metrics"`
}

// DORAMetricsConfig configures DORA (DevOps Research and Assessment) metrics
type DORAMetricsConfig struct {
	Enabled                   bool          `json:"enabled"`
	DeploymentFrequency       *FrequencyConfig `json:"deployment_frequency"`
	LeadTimeForChanges        *LeadTimeConfig  `json:"lead_time_for_changes"`
	ChangeFailureRate         *FailureRateConfig `json:"change_failure_rate"`
	MeanTimeToRecovery        *MTTRConfig      `json:"mean_time_to_recovery"`
}

// FrequencyConfig configures deployment frequency tracking
type FrequencyConfig struct {
	Enabled           bool              `json:"enabled"`
	TimeWindows       []time.Duration   `json:"time_windows"`
	Buckets          []string          `json:"buckets"`
	Targets          map[string]float64 `json:"targets"`
}

// LeadTimeConfig configures lead time tracking
type LeadTimeConfig struct {
	Enabled           bool              `json:"enabled"`
	StartEvent        string            `json:"start_event"`
	EndEvent          string            `json:"end_event"`
	Percentiles       []float64         `json:"percentiles"`
	Targets          map[string]time.Duration `json:"targets"`
}

// FailureRateConfig configures change failure rate tracking
type FailureRateConfig struct {
	Enabled           bool              `json:"enabled"`
	FailureEvents     []string          `json:"failure_events"`
	TimeWindow        time.Duration     `json:"time_window"`
	Threshold         float64           `json:"threshold"`
}

// MTTRConfig configures mean time to recovery tracking
type MTTRConfig struct {
	Enabled           bool              `json:"enabled"`
	IncidentStart     []string          `json:"incident_start"`
	IncidentEnd       []string          `json:"incident_end"`
	SeverityLevels    []string          `json:"severity_levels"`
	Targets          map[string]time.Duration `json:"targets"`
}

// SLOConfig configures Service Level Objectives
type SLOConfig struct {
	Enabled           bool              `json:"enabled"`
	SLOs              []*SLO            `json:"slos"`
	AlertingEnabled   bool              `json:"alerting_enabled"`
	BurnRateAlerts    *BurnRateConfig   `json:"burn_rate_alerts"`
}

// SLO represents a Service Level Objective
type SLO struct {
	Name              string            `json:"name"`
	Description       string            `json:"description"`
	Service           string            `json:"service"`
	SLI               *SLI              `json:"sli"`
	Target            float64           `json:"target"`
	TimeWindow        time.Duration     `json:"time_window"`
	ErrorBudget       *ErrorBudget      `json:"error_budget"`
	Labels            map[string]string `json:"labels"`
}

// SLI represents a Service Level Indicator
type SLI struct {
	Type              SLIType           `json:"type"`
	Query             string            `json:"query"`
	Threshold         float64           `json:"threshold"`
	GoodEvents        string            `json:"good_events"`
	TotalEvents       string            `json:"total_events"`
}

// SLIType represents the type of SLI
type SLIType string

const (
	SLIAvailability  SLIType = "availability"
	SLILatency       SLIType = "latency"
	SLIThroughput    SLIType = "throughput"
	SLIErrorRate     SLIType = "error_rate"
	SLICustom        SLIType = "custom"
)

// ErrorBudget represents an error budget
type ErrorBudget struct {
	Remaining         float64           `json:"remaining"`
	Consumed          float64           `json:"consumed"`
	Total             float64           `json:"total"`
	BurnRate          float64           `json:"burn_rate"`
	LastCalculated    time.Time         `json:"last_calculated"`
}

// BurnRateConfig configures burn rate alerting
type BurnRateConfig struct {
	FastBurn          *BurnRateThreshold `json:"fast_burn"`
	SlowBurn          *BurnRateThreshold `json:"slow_burn"`
}

// BurnRateThreshold defines a burn rate threshold
type BurnRateThreshold struct {
	Threshold         float64           `json:"threshold"`
	TimeWindow        time.Duration     `json:"time_window"`
	LookbackWindow    time.Duration     `json:"lookback_window"`
}

// CustomMetricConfig configures custom metrics
type CustomMetricConfig struct {
	Name              string            `json:"name"`
	Description       string            `json:"description"`
	Type              MetricType        `json:"type"`
	Query             string            `json:"query"`
	Labels            []string          `json:"labels"`
	Unit              string            `json:"unit"`
	Aggregation       AggregationType   `json:"aggregation"`
}

// MetricType represents the type of metric
type MetricType string

const (
	MetricTypeCounter   MetricType = "counter"
	MetricTypeGauge     MetricType = "gauge"
	MetricTypeHistogram MetricType = "histogram"
	MetricTypeSummary   MetricType = "summary"
)

// AggregationType represents the aggregation type
type AggregationType string

const (
	AggregationSum     AggregationType = "sum"
	AggregationAvg     AggregationType = "avg"
	AggregationMax     AggregationType = "max"
	AggregationMin     AggregationType = "min"
	AggregationCount   AggregationType = "count"
)

// MetricsCollector collects metrics from various sources
type MetricsCollector struct {
	ID                string                   `json:"id"`
	Name              string                   `json:"name"`
	Source            MetricsSource            `json:"source"`
	Status            CollectorStatus          `json:"status"`
	LastCollection    time.Time                `json:"last_collection"`
	MetricsCollected  int64                    `json:"metrics_collected"`
	ErrorCount        int64                    `json:"error_count"`
	
	// Configuration
	Config            *CollectorConfig         `json:"config"`
	
	// Context
	ctx               context.Context          `json:"-"`
	cancel            context.CancelFunc       `json:"-"`
	mu                sync.RWMutex             `json:"-"`
}

// MetricsSource represents the source of metrics
type MetricsSource string

const (
	SourcePrometheus    MetricsSource = "prometheus"
	SourceInfluxDB      MetricsSource = "influxdb"
	SourceCloudWatch    MetricsSource = "cloudwatch"
	SourceDatadog       MetricsSource = "datadog"
	SourceNewRelic      MetricsSource = "newrelic"
	SourceCustom        MetricsSource = "custom"
	SourceKubernetes    MetricsSource = "kubernetes"
	SourceGit           MetricsSource = "git"
	SourceCI            MetricsSource = "ci"
)

// CollectorStatus represents the status of a collector
type CollectorStatus string

const (
	CollectorActive    CollectorStatus = "active"
	CollectorIdle      CollectorStatus = "idle"
	CollectorError     CollectorStatus = "error"
	CollectorDisabled  CollectorStatus = "disabled"
)

// CollectorConfig holds configuration for a metrics collector
type CollectorConfig struct {
	Interval          time.Duration     `json:"interval"`
	Timeout           time.Duration     `json:"timeout"`
	Retries           int               `json:"retries"`
	Credentials       interface{}       `json:"credentials"`
	Filters           []string          `json:"filters"`
	Labels            map[string]string `json:"labels"`
	BatchSize         int               `json:"batch_size"`
}

// MetricsSeries represents a time series of metrics
type MetricsSeries struct {
	Name              string                   `json:"name"`
	Labels            map[string]string        `json:"labels"`
	Type              MetricType               `json:"type"`
	Unit              string                   `json:"unit"`
	Points            []*MetricsPoint          `json:"points"`
	Aggregations      map[string]float64       `json:"aggregations"`
	LastUpdated       time.Time                `json:"last_updated"`
	
	// Statistics
	Statistics        *SeriesStatistics        `json:"statistics"`
	
	// Retention
	RetentionPolicy   *RetentionPolicy         `json:"retention_policy"`
	
	// Synchronization
	mu                sync.RWMutex             `json:"-"`
}

// MetricsPoint represents a single point in a time series
type MetricsPoint struct {
	Timestamp         time.Time                `json:"timestamp"`
	Value             float64                  `json:"value"`
	Labels            map[string]string        `json:"labels,omitempty"`
}

// SeriesStatistics contains statistical information about a series
type SeriesStatistics struct {
	Count             int64                    `json:"count"`
	Sum               float64                  `json:"sum"`
	Mean              float64                  `json:"mean"`
	Min               float64                  `json:"min"`
	Max               float64                  `json:"max"`
	StdDev            float64                  `json:"std_dev"`
	Percentiles       map[string]float64       `json:"percentiles"`
	Trend             TrendDirection           `json:"trend"`
	LastCalculated    time.Time                `json:"last_calculated"`
}

// TrendDirection represents the direction of a trend
type TrendDirection string

const (
	TrendUp           TrendDirection = "up"
	TrendDown         TrendDirection = "down"
	TrendStable       TrendDirection = "stable"
	TrendUnknown      TrendDirection = "unknown"
)

// RetentionPolicy defines how long to retain metrics data
type RetentionPolicy struct {
	Duration          time.Duration            `json:"duration"`
	Resolution        time.Duration            `json:"resolution"`
	Aggregation       AggregationType          `json:"aggregation"`
}

// Alert represents a metrics-based alert
type Alert struct {
	ID                string                   `json:"id"`
	Name              string                   `json:"name"`
	Description       string                   `json:"description"`
	Severity          AlertSeverity            `json:"severity"`
	Status            AlertStatus              `json:"status"`
	Rule              *AlertRule               `json:"rule"`
	CreatedAt         time.Time                `json:"created_at"`
	UpdatedAt         time.Time                `json:"updated_at"`
	ResolvedAt        time.Time                `json:"resolved_at,omitempty"`
	
	// Alert details
	CurrentValue      float64                  `json:"current_value"`
	Threshold         float64                  `json:"threshold"`
	Labels            map[string]string        `json:"labels"`
	Annotations       map[string]string        `json:"annotations"`
	
	// Escalation
	EscalationLevel   int                      `json:"escalation_level"`
	NotificationsSent int                      `json:"notifications_sent"`
}

// AlertSeverity represents the severity of an alert
type AlertSeverity string

const (
	SeverityLow       AlertSeverity = "low"
	SeverityMedium    AlertSeverity = "medium"
	SeverityHigh      AlertSeverity = "high"
	SeverityCritical  AlertSeverity = "critical"
)

// AlertStatus represents the status of an alert
type AlertStatus string

const (
	AlertStatusFiring   AlertStatus = "firing"
	AlertStatusPending  AlertStatus = "pending"
	AlertStatusResolved AlertStatus = "resolved"
	AlertStatusSilenced AlertStatus = "silenced"
)

// AlertRule defines the conditions for an alert
type AlertRule struct {
	Expression        string                   `json:"expression"`
	Duration          time.Duration            `json:"duration"`
	Operator          string                   `json:"operator"`
	Threshold         float64                  `json:"threshold"`
	Labels            map[string]string        `json:"labels"`
	Annotations       map[string]string        `json:"annotations"`
}

// DeploymentEvent represents a deployment-related event for metrics
type DeploymentEvent struct {
	ID                string                   `json:"id"`
	Type              EventType                `json:"type"`
	Timestamp         time.Time                `json:"timestamp"`
	DeploymentID      string                   `json:"deployment_id"`
	Environment       string                   `json:"environment"`
	Version           string                   `json:"version"`
	Service           string                   `json:"service"`
	User              string                   `json:"user"`
	
	// Event details
	Status            string                   `json:"status"`
	Duration          time.Duration            `json:"duration,omitempty"`
	Error             string                   `json:"error,omitempty"`
	Metadata          map[string]interface{}   `json:"metadata"`
	
	// DORA metrics context
	CommitSHA         string                   `json:"commit_sha,omitempty"`
	CommitTimestamp   time.Time                `json:"commit_timestamp,omitempty"`
	PullRequestID     string                   `json:"pull_request_id,omitempty"`
	PipelineID        string                   `json:"pipeline_id,omitempty"`
}

// EventType represents the type of deployment event
type EventType string

const (
	EventDeploymentStarted    EventType = "deployment_started"
	EventDeploymentCompleted  EventType = "deployment_completed"
	EventDeploymentFailed     EventType = "deployment_failed"
	EventDeploymentRolledBack EventType = "deployment_rolled_back"
	EventIncidentStarted      EventType = "incident_started"
	EventIncidentResolved     EventType = "incident_resolved"
	EventCommitPushed         EventType = "commit_pushed"
	EventPRMerged            EventType = "pr_merged"
)

// NewDeploymentMetrics creates a new deployment metrics system
func NewDeploymentMetrics(config *DeploymentMetricsConfig) (*DeploymentMetrics, error) {
	if config == nil {
		return nil, fmt.Errorf("deployment metrics config cannot be nil")
	}

	dm := &DeploymentMetrics{
		config:           config,
		activeCollectors: make(map[string]*MetricsCollector),
		metricsSeries:    make(map[string]*MetricsSeries),
		alerts:          make(map[string]*Alert),
	}

	// Initialize components
	var err error

	dm.timeSeriesDB, err = NewTimeSeriesDB(config.TimeSeriesDB)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize time series DB: %w", err)
	}

	dm.alertManager, err = NewAlertManager(config.Alerting)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize alert manager: %w", err)
	}

	dm.dashboardManager, err = NewDashboardManager(config.Dashboards)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize dashboard manager: %w", err)
	}

	dm.metricsAggregator, err = NewMetricsAggregator(config.Aggregation)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize metrics aggregator: %w", err)
	}

	dm.anomalyDetector, err = NewAnomalyDetector(config.AnomalyDetection)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize anomaly detector: %w", err)
	}

	dm.reportGenerator, err = NewReportGenerator(config.Reporting)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize report generator: %w", err)
	}

	// Initialize Prometheus metrics
	dm.initializeMetrics()

	// Start background processes
	go dm.startMetricsCollection()
	go dm.startAlerting()
	go dm.startAnomalyDetection()

	return dm, nil
}

// initializeMetrics sets up Prometheus metrics
func (dm *DeploymentMetrics) initializeMetrics() {
	dm.deploymentFrequency = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "novacron_deployments_total",
		Help: "Total number of deployments",
	}, []string{"environment", "service", "status"})

	dm.deploymentDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "novacron_deployment_duration_seconds",
		Help:    "Duration of deployments in seconds",
		Buckets: prometheus.ExponentialBuckets(1, 2, 15),
	}, []string{"environment", "service"})

	dm.deploymentSuccess = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "novacron_deployment_success_total",
		Help: "Total number of successful deployments",
	}, []string{"environment", "service"})

	dm.leadTime = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "novacron_lead_time_seconds",
		Help:    "Lead time from commit to production in seconds",
		Buckets: []float64{300, 600, 1800, 3600, 7200, 14400, 28800, 86400, 172800, 604800},
	}, []string{"service", "environment"})

	dm.mttr = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "novacron_mttr_seconds",
		Help:    "Mean time to recovery in seconds",
		Buckets: []float64{60, 300, 900, 1800, 3600, 7200, 14400, 28800, 86400},
	}, []string{"service", "severity"})

	dm.changeFailureRate = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "novacron_change_failure_rate",
		Help: "Change failure rate as a percentage",
	}, []string{"service", "environment", "time_window"})

	dm.recoveryTime = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "novacron_recovery_time_seconds",
		Help:    "Time to recover from incidents in seconds",
		Buckets: []float64{60, 300, 900, 1800, 3600, 7200, 14400, 28800, 86400},
	}, []string{"service", "severity"})
}

// RecordDeploymentEvent records a deployment event for metrics
func (dm *DeploymentMetrics) RecordDeploymentEvent(event *DeploymentEvent) error {
	log.Printf("Recording deployment event: %s for deployment %s", event.Type, event.DeploymentID)

	// Record basic deployment metrics
	switch event.Type {
	case EventDeploymentStarted:
		dm.deploymentFrequency.WithLabelValues(event.Environment, event.Service, "started").Inc()
	case EventDeploymentCompleted:
		dm.deploymentFrequency.WithLabelValues(event.Environment, event.Service, "completed").Inc()
		dm.deploymentSuccess.WithLabelValues(event.Environment, event.Service).Inc()
		if event.Duration > 0 {
			dm.deploymentDuration.WithLabelValues(event.Environment, event.Service).Observe(event.Duration.Seconds())
		}
	case EventDeploymentFailed:
		dm.deploymentFrequency.WithLabelValues(event.Environment, event.Service, "failed").Inc()
	case EventDeploymentRolledBack:
		dm.deploymentFrequency.WithLabelValues(event.Environment, event.Service, "rolled_back").Inc()
	}

	// Calculate DORA metrics
	if dm.config.DORAMetrics.Enabled {
		dm.calculateDORAMetrics(event)
	}

	// Store event in time series DB
	if err := dm.timeSeriesDB.StoreEvent(event); err != nil {
		log.Printf("Warning: failed to store event in time series DB: %v", err)
	}

	// Check for SLO violations
	if dm.config.SLOConfig.Enabled {
		dm.checkSLOViolations(event)
	}

	// Trigger anomaly detection
	dm.anomalyDetector.ProcessEvent(event)

	return nil
}

// calculateDORAMetrics calculates DORA metrics from deployment events
func (dm *DeploymentMetrics) calculateDORAMetrics(event *DeploymentEvent) {
	switch event.Type {
	case EventDeploymentCompleted:
		// Lead Time for Changes
		if dm.config.DORAMetrics.LeadTimeForChanges.Enabled && !event.CommitTimestamp.IsZero() {
			leadTime := event.Timestamp.Sub(event.CommitTimestamp)
			dm.leadTime.WithLabelValues(event.Service, event.Environment).Observe(leadTime.Seconds())
		}

	case EventDeploymentFailed:
		// Change Failure Rate calculation is done periodically
		dm.updateChangeFailureRate(event)

	case EventIncidentResolved:
		// Mean Time to Recovery
		if dm.config.DORAMetrics.MeanTimeToRecovery.Enabled && event.Duration > 0 {
			severity := "unknown"
			if s, ok := event.Metadata["severity"].(string); ok {
				severity = s
			}
			dm.mttr.WithLabelValues(event.Service, severity).Observe(event.Duration.Seconds())
			dm.recoveryTime.WithLabelValues(event.Service, severity).Observe(event.Duration.Seconds())
		}
	}
}

// updateChangeFailureRate updates the change failure rate metric
func (dm *DeploymentMetrics) updateChangeFailureRate(event *DeploymentEvent) {
	// Query recent deployments to calculate failure rate
	timeWindow := dm.config.DORAMetrics.ChangeFailureRate.TimeWindow
	endTime := time.Now()
	startTime := endTime.Add(-timeWindow)

	totalDeployments, err := dm.timeSeriesDB.CountDeployments(event.Service, event.Environment, startTime, endTime)
	if err != nil {
		log.Printf("Warning: failed to count total deployments: %v", err)
		return
	}

	failedDeployments, err := dm.timeSeriesDB.CountFailedDeployments(event.Service, event.Environment, startTime, endTime)
	if err != nil {
		log.Printf("Warning: failed to count failed deployments: %v", err)
		return
	}

	if totalDeployments > 0 {
		failureRate := float64(failedDeployments) / float64(totalDeployments) * 100
		dm.changeFailureRate.WithLabelValues(event.Service, event.Environment, timeWindow.String()).Set(failureRate)
	}
}

// checkSLOViolations checks for SLO violations
func (dm *DeploymentMetrics) checkSLOViolations(event *DeploymentEvent) {
	for _, slo := range dm.config.SLOConfig.SLOs {
		if slo.Service != event.Service {
			continue
		}

		// Calculate current SLI value
		sliValue, err := dm.calculateSLI(slo.SLI)
		if err != nil {
			log.Printf("Warning: failed to calculate SLI for %s: %v", slo.Name, err)
			continue
		}

		// Check if SLO is violated
		if sliValue < slo.Target {
			alert := &Alert{
				ID:           fmt.Sprintf("slo-%s-%d", slo.Name, time.Now().UnixNano()),
				Name:         fmt.Sprintf("SLO Violation: %s", slo.Name),
				Description:  fmt.Sprintf("SLO %s is below target: %.2f%% < %.2f%%", slo.Name, sliValue*100, slo.Target*100),
				Severity:     SeverityHigh,
				Status:       AlertStatusFiring,
				CurrentValue: sliValue,
				Threshold:    slo.Target,
				CreatedAt:    time.Now(),
				UpdatedAt:    time.Now(),
				Labels:       map[string]string{"slo": slo.Name, "service": slo.Service},
			}

			dm.alertManager.TriggerAlert(alert)
		}

		// Update error budget
		dm.updateErrorBudget(slo, sliValue)
	}
}

// calculateSLI calculates the current value of an SLI
func (dm *DeploymentMetrics) calculateSLI(sli *SLI) (float64, error) {
	switch sli.Type {
	case SLIAvailability:
		return dm.timeSeriesDB.QueryAvailability(sli.Query)
	case SLILatency:
		return dm.timeSeriesDB.QueryLatency(sli.Query, sli.Threshold)
	case SLIErrorRate:
		return dm.timeSeriesDB.QueryErrorRate(sli.Query)
	case SLICustom:
		return dm.timeSeriesDB.QueryCustom(sli.Query)
	default:
		return 0, fmt.Errorf("unsupported SLI type: %s", sli.Type)
	}
}

// updateErrorBudget updates the error budget for an SLO
func (dm *DeploymentMetrics) updateErrorBudget(slo *SLO, currentSLI float64) {
	if slo.ErrorBudget == nil {
		slo.ErrorBudget = &ErrorBudget{}
	}

	// Calculate error budget consumption
	errorRate := 1.0 - currentSLI
	allowedErrorRate := 1.0 - slo.Target
	
	if allowedErrorRate > 0 {
		consumptionRate := errorRate / allowedErrorRate
		slo.ErrorBudget.BurnRate = consumptionRate
		
		// Update remaining budget
		timeElapsed := time.Since(slo.ErrorBudget.LastCalculated)
		if !slo.ErrorBudget.LastCalculated.IsZero() {
			timeRatio := float64(timeElapsed) / float64(slo.TimeWindow)
			budgetConsumed := consumptionRate * timeRatio
			slo.ErrorBudget.Consumed += budgetConsumed
			slo.ErrorBudget.Remaining = 1.0 - slo.ErrorBudget.Consumed
		}
	}
	
	slo.ErrorBudget.LastCalculated = time.Now()
}

// GenerateReport generates a deployment metrics report
func (dm *DeploymentMetrics) GenerateReport(ctx context.Context, request *ReportRequest) (*DeploymentReport, error) {
	log.Printf("Generating deployment metrics report for period %s to %s", 
		request.StartTime, request.EndTime)

	report := &DeploymentReport{
		Period:       &ReportPeriod{Start: request.StartTime, End: request.EndTime},
		GeneratedAt:  time.Now(),
		Services:     make([]*ServiceMetrics, 0),
		DORAMetrics:  &DORAReport{},
		SLOReport:    &SLOReport{},
		Incidents:    make([]*IncidentReport, 0),
		Trends:       &TrendAnalysis{},
	}

	// Generate DORA metrics report
	if dm.config.DORAMetrics.Enabled {
		doraReport, err := dm.generateDORAReport(ctx, request)
		if err != nil {
			log.Printf("Warning: failed to generate DORA report: %v", err)
		} else {
			report.DORAMetrics = doraReport
		}
	}

	// Generate SLO report
	if dm.config.SLOConfig.Enabled {
		sloReport, err := dm.generateSLOReport(ctx, request)
		if err != nil {
			log.Printf("Warning: failed to generate SLO report: %v", err)
		} else {
			report.SLOReport = sloReport
		}
	}

	// Generate service metrics
	serviceMetrics, err := dm.generateServiceMetrics(ctx, request)
	if err != nil {
		log.Printf("Warning: failed to generate service metrics: %v", err)
	} else {
		report.Services = serviceMetrics
	}

	// Generate trend analysis
	trends, err := dm.generateTrendAnalysis(ctx, request)
	if err != nil {
		log.Printf("Warning: failed to generate trend analysis: %v", err)
	} else {
		report.Trends = trends
	}

	return report, nil
}

// GetMetricsSeries retrieves a metrics series
func (dm *DeploymentMetrics) GetMetricsSeries(name string) (*MetricsSeries, error) {
	dm.mu.RLock()
	defer dm.mu.RUnlock()

	series, exists := dm.metricsSeries[name]
	if !exists {
		return nil, fmt.Errorf("metrics series %s not found", name)
	}

	return series, nil
}

// ListActiveAlerts lists all active alerts
func (dm *DeploymentMetrics) ListActiveAlerts() []*Alert {
	dm.mu.RLock()
	defer dm.mu.RUnlock()

	alerts := make([]*Alert, 0)
	for _, alert := range dm.alerts {
		if alert.Status == AlertStatusFiring || alert.Status == AlertStatusPending {
			alerts = append(alerts, alert)
		}
	}

	return alerts
}

// Background processes

func (dm *DeploymentMetrics) startMetricsCollection() {
	log.Printf("Starting metrics collection with interval %v", dm.config.MetricsInterval)

	ticker := time.NewTicker(dm.config.MetricsInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			dm.collectMetrics()
		}
	}
}

func (dm *DeploymentMetrics) startAlerting() {
	log.Printf("Starting alerting system")

	ticker := time.NewTicker(1 * time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			dm.processAlerts()
		}
	}
}

func (dm *DeploymentMetrics) startAnomalyDetection() {
	log.Printf("Starting anomaly detection")

	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			dm.detectAnomalies()
		}
	}
}

func (dm *DeploymentMetrics) collectMetrics() {
	log.Printf("Collecting metrics from all sources")

	dm.mu.RLock()
	collectors := make([]*MetricsCollector, 0, len(dm.activeCollectors))
	for _, collector := range dm.activeCollectors {
		collectors = append(collectors, collector)
	}
	dm.mu.RUnlock()

	// Collect from all active collectors
	for _, collector := range collectors {
		if collector.Status == CollectorActive {
			go dm.collectFromSource(collector)
		}
	}
}

func (dm *DeploymentMetrics) collectFromSource(collector *MetricsCollector) {
	log.Printf("Collecting metrics from source %s", collector.Source)

	// Implementation would collect metrics from the specific source
	// This is a simplified mock implementation
	
	collector.mu.Lock()
	collector.LastCollection = time.Now()
	collector.MetricsCollected++
	collector.mu.Unlock()
}

func (dm *DeploymentMetrics) processAlerts() {
	log.Printf("Processing alerts")

	dm.mu.RLock()
	alerts := make([]*Alert, 0, len(dm.alerts))
	for _, alert := range dm.alerts {
		if alert.Status == AlertStatusFiring {
			alerts = append(alerts, alert)
		}
	}
	dm.mu.RUnlock()

	// Process each active alert
	for _, alert := range alerts {
		dm.alertManager.ProcessAlert(alert)
	}
}

func (dm *DeploymentMetrics) detectAnomalies() {
	log.Printf("Running anomaly detection")

	dm.mu.RLock()
	series := make([]*MetricsSeries, 0, len(dm.metricsSeries))
	for _, s := range dm.metricsSeries {
		series = append(series, s)
	}
	dm.mu.RUnlock()

	// Run anomaly detection on each series
	for _, s := range series {
		anomalies := dm.anomalyDetector.DetectAnomalies(s)
		for _, anomaly := range anomalies {
			dm.handleAnomaly(anomaly)
		}
	}
}

func (dm *DeploymentMetrics) handleAnomaly(anomaly *Anomaly) {
	log.Printf("Detected anomaly in series %s: %s", anomaly.SeriesName, anomaly.Description)

	// Create alert for anomaly
	alert := &Alert{
		ID:           fmt.Sprintf("anomaly-%s-%d", anomaly.SeriesName, time.Now().UnixNano()),
		Name:         fmt.Sprintf("Anomaly Detected: %s", anomaly.SeriesName),
		Description:  anomaly.Description,
		Severity:     SeverityMedium,
		Status:       AlertStatusFiring,
		CreatedAt:    time.Now(),
		UpdatedAt:    time.Now(),
		Labels:       map[string]string{"type": "anomaly", "series": anomaly.SeriesName},
	}

	dm.alertManager.TriggerAlert(alert)
}

// Helper methods for report generation

func (dm *DeploymentMetrics) generateDORAReport(ctx context.Context, request *ReportRequest) (*DORAReport, error) {
	// Implementation would generate DORA metrics report
	return &DORAReport{
		DeploymentFrequency: &FrequencyReport{
			Daily:   5.2,
			Weekly:  36.4,
			Monthly: 158.3,
		},
		LeadTime: &LeadTimeReport{
			Average: 4 * time.Hour,
			P50:     2 * time.Hour,
			P90:     8 * time.Hour,
			P99:     24 * time.Hour,
		},
		ChangeFailureRate: &FailureRateReport{
			Rate:           2.3,
			TotalChanges:   152,
			FailedChanges:  3,
		},
		MTTR: &MTTRReport{
			Average: 45 * time.Minute,
			P50:     20 * time.Minute,
			P90:     2 * time.Hour,
			P99:     4 * time.Hour,
		},
	}, nil
}

func (dm *DeploymentMetrics) generateSLOReport(ctx context.Context, request *ReportRequest) (*SLOReport, error) {
	// Implementation would generate SLO report
	return &SLOReport{
		SLOs: []*SLOStatus{
			{
				Name:         "API Availability",
				CurrentSLI:   99.8,
				Target:       99.9,
				ErrorBudget:  &ErrorBudgetStatus{Remaining: 20.3, BurnRate: 0.8},
				Status:       "healthy",
			},
		},
	}, nil
}

func (dm *DeploymentMetrics) generateServiceMetrics(ctx context.Context, request *ReportRequest) ([]*ServiceMetrics, error) {
	// Implementation would generate service metrics
	return []*ServiceMetrics{
		{
			Name:            "api-service",
			Deployments:     23,
			SuccessRate:     95.7,
			AverageLeadTime: 2 * time.Hour,
			MTTR:            30 * time.Minute,
		},
	}, nil
}

func (dm *DeploymentMetrics) generateTrendAnalysis(ctx context.Context, request *ReportRequest) (*TrendAnalysis, error) {
	// Implementation would generate trend analysis
	return &TrendAnalysis{
		DeploymentFrequencyTrend: TrendUp,
		LeadTimeTrend:           TrendDown,
		FailureRateTrend:        TrendDown,
		MTTRTrend:              TrendStable,
	}, nil
}

// Mock implementations for referenced components and types

func NewTimeSeriesDB(config *TimeSeriesDBConfig) (*TimeSeriesDB, error) {
	return &TimeSeriesDB{}, nil
}

func NewAlertManager(config *AlertingConfig) (*AlertManager, error) {
	return &AlertManager{}, nil
}

func NewDashboardManager(config *DashboardConfig) (*DashboardManager, error) {
	return &DashboardManager{}, nil
}

func NewMetricsAggregator(config *AggregationConfig) (*MetricsAggregator, error) {
	return &MetricsAggregator{}, nil
}

func NewAnomalyDetector(config *AnomalyDetectionConfig) (*AnomalyDetector, error) {
	return &AnomalyDetector{}, nil
}

func NewReportGenerator(config *ReportingConfig) (*ReportGenerator, error) {
	return &ReportGenerator{}, nil
}

// Mock methods for TimeSeriesDB
func (tsdb *TimeSeriesDB) StoreEvent(event *DeploymentEvent) error {
	log.Printf("Storing event %s in time series DB", event.ID)
	return nil
}

func (tsdb *TimeSeriesDB) CountDeployments(service, environment string, start, end time.Time) (int, error) {
	return 10, nil // Mock value
}

func (tsdb *TimeSeriesDB) CountFailedDeployments(service, environment string, start, end time.Time) (int, error) {
	return 1, nil // Mock value
}

func (tsdb *TimeSeriesDB) QueryAvailability(query string) (float64, error) {
	return 0.998, nil // Mock 99.8% availability
}

func (tsdb *TimeSeriesDB) QueryLatency(query string, threshold float64) (float64, error) {
	return 0.95, nil // Mock 95% below threshold
}

func (tsdb *TimeSeriesDB) QueryErrorRate(query string) (float64, error) {
	return 0.002, nil // Mock 0.2% error rate
}

func (tsdb *TimeSeriesDB) QueryCustom(query string) (float64, error) {
	return 1.0, nil // Mock custom metric
}

// Mock methods for other components
func (am *AlertManager) TriggerAlert(alert *Alert) {
	log.Printf("Triggering alert: %s", alert.Name)
}

func (am *AlertManager) ProcessAlert(alert *Alert) {
	log.Printf("Processing alert: %s", alert.Name)
}

func (ad *AnomalyDetector) ProcessEvent(event *DeploymentEvent) {
	log.Printf("Processing event for anomaly detection: %s", event.ID)
}

func (ad *AnomalyDetector) DetectAnomalies(series *MetricsSeries) []*Anomaly {
	// Mock implementation - no anomalies detected
	return []*Anomaly{}
}

// Additional type definitions for the deployment metrics system

type TimeSeriesDBConfig struct{}
type AlertingConfig struct{}
type DashboardConfig struct{}
type AggregationConfig struct{}
type AnomalyDetectionConfig struct{}
type ReportingConfig struct{}

type TimeSeriesDB struct{}
type AlertManager struct{}
type DashboardManager struct{}
type MetricsAggregator struct{}
type AnomalyDetector struct{}
type ReportGenerator struct{}

type Anomaly struct {
	SeriesName  string `json:"series_name"`
	Description string `json:"description"`
	Severity    string `json:"severity"`
	Timestamp   time.Time `json:"timestamp"`
}

type ReportRequest struct {
	StartTime time.Time `json:"start_time"`
	EndTime   time.Time `json:"end_time"`
	Services  []string  `json:"services"`
}

type DeploymentReport struct {
	Period      *ReportPeriod    `json:"period"`
	GeneratedAt time.Time        `json:"generated_at"`
	Services    []*ServiceMetrics `json:"services"`
	DORAMetrics *DORAReport      `json:"dora_metrics"`
	SLOReport   *SLOReport       `json:"slo_report"`
	Incidents   []*IncidentReport `json:"incidents"`
	Trends      *TrendAnalysis   `json:"trends"`
}

type ReportPeriod struct {
	Start time.Time `json:"start"`
	End   time.Time `json:"end"`
}

type ServiceMetrics struct {
	Name            string        `json:"name"`
	Deployments     int           `json:"deployments"`
	SuccessRate     float64       `json:"success_rate"`
	AverageLeadTime time.Duration `json:"average_lead_time"`
	MTTR            time.Duration `json:"mttr"`
}

type DORAReport struct {
	DeploymentFrequency *FrequencyReport    `json:"deployment_frequency"`
	LeadTime           *LeadTimeReport     `json:"lead_time"`
	ChangeFailureRate  *FailureRateReport  `json:"change_failure_rate"`
	MTTR              *MTTRReport         `json:"mttr"`
}

type FrequencyReport struct {
	Daily   float64 `json:"daily"`
	Weekly  float64 `json:"weekly"`
	Monthly float64 `json:"monthly"`
}

type LeadTimeReport struct {
	Average time.Duration `json:"average"`
	P50     time.Duration `json:"p50"`
	P90     time.Duration `json:"p90"`
	P99     time.Duration `json:"p99"`
}

type FailureRateReport struct {
	Rate          float64 `json:"rate"`
	TotalChanges  int     `json:"total_changes"`
	FailedChanges int     `json:"failed_changes"`
}

type MTTRReport struct {
	Average time.Duration `json:"average"`
	P50     time.Duration `json:"p50"`
	P90     time.Duration `json:"p90"`
	P99     time.Duration `json:"p99"`
}

type SLOReport struct {
	SLOs []*SLOStatus `json:"slos"`
}

type SLOStatus struct {
	Name         string              `json:"name"`
	CurrentSLI   float64             `json:"current_sli"`
	Target       float64             `json:"target"`
	ErrorBudget  *ErrorBudgetStatus  `json:"error_budget"`
	Status       string              `json:"status"`
}

type ErrorBudgetStatus struct {
	Remaining float64 `json:"remaining"`
	BurnRate  float64 `json:"burn_rate"`
}

type IncidentReport struct {
	ID          string        `json:"id"`
	Service     string        `json:"service"`
	Severity    string        `json:"severity"`
	Duration    time.Duration `json:"duration"`
	Impact      string        `json:"impact"`
}

type TrendAnalysis struct {
	DeploymentFrequencyTrend TrendDirection `json:"deployment_frequency_trend"`
	LeadTimeTrend           TrendDirection `json:"lead_time_trend"`
	FailureRateTrend        TrendDirection `json:"failure_rate_trend"`
	MTTRTrend              TrendDirection `json:"mttr_trend"`
}