// Package observability provides real-time deployment tracking, change attribution,
// DORA metrics, Grafana dashboard integration, and alert routing for deployment monitoring.
package observability

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"
)

// DeploymentTracker tracks all deployment activities and metrics
type DeploymentTracker struct {
	mu                sync.RWMutex
	deployments       map[string]*TrackedDeployment
	metricsCollector  DORAMetricsCollector
	changeTracker     ChangeTracker
	dashboardExporter DashboardExporter
	alertRouter       AlertRouter
	eventStream       EventStream
	ctx               context.Context
	cancel            context.CancelFunc
	wg                sync.WaitGroup
}

// TrackedDeployment represents a tracked deployment
type TrackedDeployment struct {
	ID                string                `json:"id"`
	Name              string                `json:"name"`
	Version           string                `json:"version"`
	Environment       string                `json:"environment"`
	Status            DeploymentStatus      `json:"status"`
	StartTime         time.Time             `json:"start_time"`
	EndTime           *time.Time            `json:"end_time,omitempty"`
	Duration          time.Duration         `json:"duration"`
	Changes           []CodeChange          `json:"changes"`
	Metrics           *DeploymentMetrics    `json:"metrics"`
	Events            []DeploymentEvent     `json:"events"`
	Alerts            []DeploymentAlert     `json:"alerts"`
	LeadTime          time.Duration         `json:"lead_time"`
	RecoveryTime      *time.Duration        `json:"recovery_time,omitempty"`
	Failed            bool                  `json:"failed"`
	RolledBack        bool                  `json:"rolled_back"`
	ChangeFailureRate float64               `json:"change_failure_rate"`
}

// DeploymentStatus represents deployment status
type DeploymentStatus string

const (
	StatusDeploying  DeploymentStatus = "deploying"
	StatusValidating DeploymentStatus = "validating"
	StatusActive     DeploymentStatus = "active"
	StatusFailed     DeploymentStatus = "failed"
	StatusRolledBack DeploymentStatus = "rolled_back"
)

// CodeChange represents a code change in a deployment
type CodeChange struct {
	CommitHash    string    `json:"commit_hash"`
	Author        string    `json:"author"`
	Message       string    `json:"message"`
	Timestamp     time.Time `json:"timestamp"`
	FilesChanged  int       `json:"files_changed"`
	LinesAdded    int       `json:"lines_added"`
	LinesRemoved  int       `json:"lines_removed"`
	PullRequestID string    `json:"pull_request_id,omitempty"`
	Jira          string    `json:"jira,omitempty"`
}

// DeploymentMetrics contains deployment-specific metrics
type DeploymentMetrics struct {
	RequestCount      int64         `json:"request_count"`
	ErrorCount        int64         `json:"error_count"`
	ErrorRate         float64       `json:"error_rate"`
	LatencyP50        time.Duration `json:"latency_p50"`
	LatencyP95        time.Duration `json:"latency_p95"`
	LatencyP99        time.Duration `json:"latency_p99"`
	Throughput        float64       `json:"throughput"`
	CPU               float64       `json:"cpu_percent"`
	Memory            float64       `json:"memory_mb"`
	ActiveConnections int64         `json:"active_connections"`
	Timestamp         time.Time     `json:"timestamp"`
}

// DeploymentEvent represents an event during deployment
type DeploymentEvent struct {
	ID          string                 `json:"id"`
	Type        EventType              `json:"type"`
	Severity    EventSeverity          `json:"severity"`
	Message     string                 `json:"message"`
	Timestamp   time.Time              `json:"timestamp"`
	Source      string                 `json:"source"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

// EventType represents types of deployment events
type EventType string

const (
	EventTypeStart        EventType = "deployment_start"
	EventTypeProgress     EventType = "deployment_progress"
	EventTypeComplete     EventType = "deployment_complete"
	EventTypeFailed       EventType = "deployment_failed"
	EventTypeRollback     EventType = "deployment_rollback"
	EventTypeHealthCheck  EventType = "health_check"
	EventTypeMetricAlert  EventType = "metric_alert"
	EventTypeTrafficShift EventType = "traffic_shift"
)

// EventSeverity represents event severity
type EventSeverity string

const (
	SeverityInfo     EventSeverity = "info"
	SeverityWarning  EventSeverity = "warning"
	SeverityError    EventSeverity = "error"
	SeverityCritical EventSeverity = "critical"
)

// DeploymentAlert represents a deployment alert
type DeploymentAlert struct {
	ID            string        `json:"id"`
	Name          string        `json:"name"`
	Severity      EventSeverity `json:"severity"`
	Condition     string        `json:"condition"`
	Value         float64       `json:"value"`
	Threshold     float64       `json:"threshold"`
	Triggered     bool          `json:"triggered"`
	TriggeredAt   time.Time     `json:"triggered_at"`
	Resolved      bool          `json:"resolved"`
	ResolvedAt    *time.Time    `json:"resolved_at,omitempty"`
	NotificationsSent int        `json:"notifications_sent"`
}

// DORAMetrics contains DORA (DevOps Research and Assessment) metrics
type DORAMetrics struct {
	DeploymentFrequency   *DeploymentFrequency   `json:"deployment_frequency"`
	LeadTime              *LeadTimeMetrics       `json:"lead_time"`
	MeanTimeToRecover     *MTTRMetrics           `json:"mean_time_to_recover"`
	ChangeFailureRate     *ChangeFailureRate     `json:"change_failure_rate"`
	PerformanceTier       PerformanceTier        `json:"performance_tier"`
	MeasurementPeriod     time.Duration          `json:"measurement_period"`
	Timestamp             time.Time              `json:"timestamp"`
}

// DeploymentFrequency tracks deployment frequency
type DeploymentFrequency struct {
	TotalDeployments     int           `json:"total_deployments"`
	SuccessfulDeployments int          `json:"successful_deployments"`
	FailedDeployments    int           `json:"failed_deployments"`
	DeploymentsPerDay    float64       `json:"deployments_per_day"`
	DeploymentsPerWeek   float64       `json:"deployments_per_week"`
	AverageInterval      time.Duration `json:"average_interval"`
	Trend                string        `json:"trend"`
}

// LeadTimeMetrics tracks lead time for changes
type LeadTimeMetrics struct {
	AverageLeadTime  time.Duration `json:"average_lead_time"`
	MedianLeadTime   time.Duration `json:"median_lead_time"`
	P95LeadTime      time.Duration `json:"p95_lead_time"`
	MinLeadTime      time.Duration `json:"min_lead_time"`
	MaxLeadTime      time.Duration `json:"max_lead_time"`
	SampleCount      int           `json:"sample_count"`
}

// MTTRMetrics tracks Mean Time To Recover
type MTTRMetrics struct {
	AverageMTTR     time.Duration `json:"average_mttr"`
	MedianMTTR      time.Duration `json:"median_mttr"`
	P95MTTR         time.Duration `json:"p95_mttr"`
	MinMTTR         time.Duration `json:"min_mttr"`
	MaxMTTR         time.Duration `json:"max_mttr"`
	IncidentCount   int           `json:"incident_count"`
}

// ChangeFailureRate tracks change failure rate
type ChangeFailureRate struct {
	TotalChanges     int     `json:"total_changes"`
	FailedChanges    int     `json:"failed_changes"`
	FailureRate      float64 `json:"failure_rate"`
	RollbackCount    int     `json:"rollback_count"`
	HotfixCount      int     `json:"hotfix_count"`
}

// PerformanceTier represents DORA performance tier
type PerformanceTier string

const (
	TierElite    PerformanceTier = "elite"
	TierHigh     PerformanceTier = "high"
	TierMedium   PerformanceTier = "medium"
	TierLow      PerformanceTier = "low"
)

// DORAMetricsCollector collects DORA metrics
type DORAMetricsCollector interface {
	CollectMetrics(ctx context.Context, period time.Duration) (*DORAMetrics, error)
	CalculateDeploymentFrequency(deployments []*TrackedDeployment) *DeploymentFrequency
	CalculateLeadTime(deployments []*TrackedDeployment) *LeadTimeMetrics
	CalculateMTTR(deployments []*TrackedDeployment) *MTTRMetrics
	CalculateChangeFailureRate(deployments []*TrackedDeployment) *ChangeFailureRate
	DeterminePerformanceTier(metrics *DORAMetrics) PerformanceTier
}

// ChangeTracker tracks code changes and attributions
type ChangeTracker interface {
	TrackChanges(ctx context.Context, deploymentID string, fromCommit, toCommit string) ([]CodeChange, error)
	GetChangeAttribution(commitHash string) (*ChangeAttribution, error)
	LinkJiraIssue(commitHash string, jiraID string) error
}

// ChangeAttribution attributes changes to specific developers/teams
type ChangeAttribution struct {
	CommitHash   string    `json:"commit_hash"`
	Author       string    `json:"author"`
	AuthorEmail  string    `json:"author_email"`
	Team         string    `json:"team"`
	Component    string    `json:"component"`
	ImpactScore  float64   `json:"impact_score"`
	RiskScore    float64   `json:"risk_score"`
	Timestamp    time.Time `json:"timestamp"`
}

// DashboardExporter exports metrics to dashboards
type DashboardExporter interface {
	ExportToGrafana(ctx context.Context, metrics *DORAMetrics) error
	UpdateDashboard(ctx context.Context, deploymentID string, metrics *DeploymentMetrics) error
	CreateDeploymentAnnotation(ctx context.Context, deployment *TrackedDeployment) error
	GetDashboardURL(deploymentID string) string
}

// AlertRouter routes deployment alerts
type AlertRouter interface {
	RouteAlert(ctx context.Context, alert *DeploymentAlert) error
	ConfigureAlertRules(deploymentID string, rules []AlertRule) error
	GetActiveAlerts(deploymentID string) ([]DeploymentAlert, error)
	ResolveAlert(alertID string) error
}

// AlertRule defines an alert rule
type AlertRule struct {
	Name      string        `json:"name"`
	Metric    string        `json:"metric"`
	Condition string        `json:"condition"`
	Threshold float64       `json:"threshold"`
	Duration  time.Duration `json:"duration"`
	Severity  EventSeverity `json:"severity"`
	Actions   []string      `json:"actions"`
}

// EventStream streams deployment events in real-time
type EventStream interface {
	PublishEvent(ctx context.Context, event *DeploymentEvent) error
	SubscribeToEvents(ctx context.Context, deploymentID string) (<-chan *DeploymentEvent, error)
	GetEventHistory(deploymentID string, since time.Time) ([]DeploymentEvent, error)
}

// NewDeploymentTracker creates a new deployment tracker
func NewDeploymentTracker(
	metricsCollector DORAMetricsCollector,
	changeTracker ChangeTracker,
	dashboardExporter DashboardExporter,
	alertRouter AlertRouter,
	eventStream EventStream,
) *DeploymentTracker {
	ctx, cancel := context.WithCancel(context.Background())

	return &DeploymentTracker{
		deployments:       make(map[string]*TrackedDeployment),
		metricsCollector:  metricsCollector,
		changeTracker:     changeTracker,
		dashboardExporter: dashboardExporter,
		alertRouter:       alertRouter,
		eventStream:       eventStream,
		ctx:               ctx,
		cancel:            cancel,
	}
}

// TrackDeployment starts tracking a deployment
func (dt *DeploymentTracker) TrackDeployment(
	deploymentID string,
	name string,
	version string,
	environment string,
	fromCommit string,
	toCommit string,
) error {
	dt.mu.Lock()
	defer dt.mu.Unlock()

	// Track code changes
	changes, err := dt.changeTracker.TrackChanges(dt.ctx, deploymentID, fromCommit, toCommit)
	if err != nil {
		return fmt.Errorf("failed to track changes: %w", err)
	}

	// Calculate lead time (time from first commit to deployment)
	var leadTime time.Duration
	if len(changes) > 0 {
		earliestCommit := changes[0].Timestamp
		for _, change := range changes {
			if change.Timestamp.Before(earliestCommit) {
				earliestCommit = change.Timestamp
			}
		}
		leadTime = time.Since(earliestCommit)
	}

	deployment := &TrackedDeployment{
		ID:          deploymentID,
		Name:        name,
		Version:     version,
		Environment: environment,
		Status:      StatusDeploying,
		StartTime:   time.Now(),
		Changes:     changes,
		LeadTime:    leadTime,
		Events:      []DeploymentEvent{},
		Alerts:      []DeploymentAlert{},
	}

	dt.deployments[deploymentID] = deployment

	// Publish deployment start event
	event := &DeploymentEvent{
		ID:        fmt.Sprintf("%s-start", deploymentID),
		Type:      EventTypeStart,
		Severity:  SeverityInfo,
		Message:   fmt.Sprintf("Deployment %s started", name),
		Timestamp: time.Now(),
		Source:    "deployment_tracker",
	}
	_ = dt.eventStream.PublishEvent(dt.ctx, event)
	deployment.Events = append(deployment.Events, *event)

	// Create dashboard annotation
	_ = dt.dashboardExporter.CreateDeploymentAnnotation(dt.ctx, deployment)

	// Start monitoring
	dt.wg.Add(1)
	go dt.monitorDeployment(deploymentID)

	return nil
}

// UpdateDeploymentStatus updates deployment status
func (dt *DeploymentTracker) UpdateDeploymentStatus(deploymentID string, status DeploymentStatus) error {
	dt.mu.Lock()
	defer dt.mu.Unlock()

	deployment, exists := dt.deployments[deploymentID]
	if !exists {
		return fmt.Errorf("deployment %s not found", deploymentID)
	}

	deployment.Status = status

	// Publish status update event
	event := &DeploymentEvent{
		ID:        fmt.Sprintf("%s-status-%d", deploymentID, time.Now().Unix()),
		Type:      EventTypeProgress,
		Severity:  SeverityInfo,
		Message:   fmt.Sprintf("Deployment status: %s", status),
		Timestamp: time.Now(),
		Source:    "deployment_tracker",
	}
	_ = dt.eventStream.PublishEvent(dt.ctx, event)
	deployment.Events = append(deployment.Events, *event)

	return nil
}

// CompleteDeployment marks a deployment as complete
func (dt *DeploymentTracker) CompleteDeployment(deploymentID string, success bool) error {
	dt.mu.Lock()
	defer dt.mu.Unlock()

	deployment, exists := dt.deployments[deploymentID]
	if !exists {
		return fmt.Errorf("deployment %s not found", deploymentID)
	}

	now := time.Now()
	deployment.EndTime = &now
	deployment.Duration = now.Sub(deployment.StartTime)

	if success {
		deployment.Status = StatusActive
	} else {
		deployment.Status = StatusFailed
		deployment.Failed = true
	}

	// Publish completion event
	eventType := EventTypeComplete
	severity := SeverityInfo
	message := fmt.Sprintf("Deployment %s completed successfully", deployment.Name)

	if !success {
		eventType = EventTypeFailed
		severity = SeverityError
		message = fmt.Sprintf("Deployment %s failed", deployment.Name)
	}

	event := &DeploymentEvent{
		ID:        fmt.Sprintf("%s-complete", deploymentID),
		Type:      eventType,
		Severity:  severity,
		Message:   message,
		Timestamp: time.Now(),
		Source:    "deployment_tracker",
	}
	_ = dt.eventStream.PublishEvent(dt.ctx, event)
	deployment.Events = append(deployment.Events, *event)

	return nil
}

// RecordRollback records a deployment rollback
func (dt *DeploymentTracker) RecordRollback(deploymentID string, recoveryTime time.Duration) error {
	dt.mu.Lock()
	defer dt.mu.Unlock()

	deployment, exists := dt.deployments[deploymentID]
	if !exists {
		return fmt.Errorf("deployment %s not found", deploymentID)
	}

	deployment.Status = StatusRolledBack
	deployment.RolledBack = true
	deployment.RecoveryTime = &recoveryTime

	// Publish rollback event
	event := &DeploymentEvent{
		ID:        fmt.Sprintf("%s-rollback", deploymentID),
		Type:      EventTypeRollback,
		Severity:  SeverityWarning,
		Message:   fmt.Sprintf("Deployment %s rolled back (recovery time: %v)", deployment.Name, recoveryTime),
		Timestamp: time.Now(),
		Source:    "deployment_tracker",
	}
	_ = dt.eventStream.PublishEvent(dt.ctx, event)
	deployment.Events = append(deployment.Events, *event)

	return nil
}

// monitorDeployment monitors a deployment
func (dt *DeploymentTracker) monitorDeployment(deploymentID string) {
	defer dt.wg.Done()

	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			dt.mu.RLock()
			deployment, exists := dt.deployments[deploymentID]
			if !exists {
				dt.mu.RUnlock()
				return
			}

			// Stop monitoring if deployment is complete
			if deployment.Status == StatusActive ||
				deployment.Status == StatusFailed ||
				deployment.Status == StatusRolledBack {
				dt.mu.RUnlock()
				return
			}
			dt.mu.RUnlock()

			// Collect and update metrics
			// In real implementation, this would call actual metrics collection

		case <-dt.ctx.Done():
			return
		}
	}
}

// GetDORAMetrics retrieves DORA metrics for a time period
func (dt *DeploymentTracker) GetDORAMetrics(period time.Duration) (*DORAMetrics, error) {
	return dt.metricsCollector.CollectMetrics(dt.ctx, period)
}

// GetDeployment retrieves a tracked deployment
func (dt *DeploymentTracker) GetDeployment(deploymentID string) (*TrackedDeployment, error) {
	dt.mu.RLock()
	defer dt.mu.RUnlock()

	deployment, exists := dt.deployments[deploymentID]
	if !exists {
		return nil, fmt.Errorf("deployment %s not found", deploymentID)
	}

	return deployment, nil
}

// ListDeployments lists all tracked deployments
func (dt *DeploymentTracker) ListDeployments(since *time.Time) []*TrackedDeployment {
	dt.mu.RLock()
	defer dt.mu.RUnlock()

	deployments := make([]*TrackedDeployment, 0)
	for _, deployment := range dt.deployments {
		if since == nil || deployment.StartTime.After(*since) {
			deployments = append(deployments, deployment)
		}
	}

	return deployments
}

// Shutdown gracefully shuts down the tracker
func (dt *DeploymentTracker) Shutdown(ctx context.Context) error {
	dt.cancel()

	done := make(chan struct{})
	go func() {
		dt.wg.Wait()
		close(done)
	}()

	select {
	case <-done:
		return nil
	case <-ctx.Done():
		return fmt.Errorf("shutdown timeout exceeded")
	}
}

// CalculatePerformanceTier calculates DORA performance tier
func CalculatePerformanceTier(metrics *DORAMetrics) PerformanceTier {
	if metrics == nil {
		return TierLow
	}

	eliteCount := 0
	highCount := 0
	mediumCount := 0

	// Deployment Frequency criteria
	if metrics.DeploymentFrequency != nil {
		if metrics.DeploymentFrequency.DeploymentsPerDay >= 1.0 {
			eliteCount++
		} else if metrics.DeploymentFrequency.DeploymentsPerWeek >= 1.0 {
			highCount++
		} else if metrics.DeploymentFrequency.DeploymentsPerWeek >= 0.25 {
			mediumCount++
		}
	}

	// Lead Time criteria
	if metrics.LeadTime != nil {
		if metrics.LeadTime.MedianLeadTime < 24*time.Hour {
			eliteCount++
		} else if metrics.LeadTime.MedianLeadTime < 7*24*time.Hour {
			highCount++
		} else if metrics.LeadTime.MedianLeadTime < 30*24*time.Hour {
			mediumCount++
		}
	}

	// MTTR criteria
	if metrics.MeanTimeToRecover != nil {
		if metrics.MeanTimeToRecover.MedianMTTR < 1*time.Hour {
			eliteCount++
		} else if metrics.MeanTimeToRecover.MedianMTTR < 24*time.Hour {
			highCount++
		} else if metrics.MeanTimeToRecover.MedianMTTR < 7*24*time.Hour {
			mediumCount++
		}
	}

	// Change Failure Rate criteria
	if metrics.ChangeFailureRate != nil {
		if metrics.ChangeFailureRate.FailureRate < 0.15 {
			eliteCount++
		} else if metrics.ChangeFailureRate.FailureRate < 0.30 {
			highCount++
		} else if metrics.ChangeFailureRate.FailureRate < 0.45 {
			mediumCount++
		}
	}

	// Determine tier (need at least 3 metrics in a tier to qualify)
	if eliteCount >= 3 {
		return TierElite
	} else if highCount >= 3 || (eliteCount >= 2 && highCount >= 1) {
		return TierHigh
	} else if mediumCount >= 2 {
		return TierMedium
	}

	return TierLow
}

// MarshalJSON implements custom JSON marshaling
func (td *TrackedDeployment) MarshalJSON() ([]byte, error) {
	type Alias TrackedDeployment

	return json.Marshal(&struct {
		*Alias
		DurationString string `json:"duration_string"`
		LeadTimeString string `json:"lead_time_string"`
	}{
		Alias:          (*Alias)(td),
		DurationString: td.Duration.String(),
		LeadTimeString: td.LeadTime.String(),
	})
}
