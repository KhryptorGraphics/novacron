// Package canary provides progressive traffic routing and canary deployment capabilities
// for DWCP v3 production deployments with automatic rollback and A/B testing.
package canary

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sync"
	"time"
)

// CanaryStage represents a stage in the progressive rollout
type CanaryStage struct {
	Name            string        `json:"name"`
	TrafficPercent  float64       `json:"traffic_percent"`
	Duration        time.Duration `json:"duration"`
	HealthThreshold float64       `json:"health_threshold"`
	ErrorThreshold  float64       `json:"error_threshold"`
	LatencyP99Max   time.Duration `json:"latency_p99_max"`
}

// CanaryDeployment represents a canary deployment configuration
type CanaryDeployment struct {
	ID              string         `json:"id"`
	Name            string         `json:"name"`
	Namespace       string         `json:"namespace"`
	BaselineVersion string         `json:"baseline_version"`
	CanaryVersion   string         `json:"canary_version"`
	Stages          []CanaryStage  `json:"stages"`
	CurrentStage    int            `json:"current_stage"`
	Status          DeploymentStatus `json:"status"`
	StartTime       time.Time      `json:"start_time"`
	EndTime         *time.Time     `json:"end_time,omitempty"`
	Metrics         *DeploymentMetrics `json:"metrics"`
	ABTest          *ABTestConfig  `json:"ab_test,omitempty"`
	FeatureFlags    map[string]bool `json:"feature_flags"`
	GoldenSignals   *GoldenSignals `json:"golden_signals"`
}

// DeploymentStatus represents the status of a canary deployment
type DeploymentStatus string

const (
	StatusInitializing DeploymentStatus = "initializing"
	StatusProgressing  DeploymentStatus = "progressing"
	StatusValidating   DeploymentStatus = "validating"
	StatusRollingBack  DeploymentStatus = "rolling_back"
	StatusCompleted    DeploymentStatus = "completed"
	StatusFailed       DeploymentStatus = "failed"
	StatusPaused       DeploymentStatus = "paused"
)

// DeploymentMetrics tracks key metrics during deployment
type DeploymentMetrics struct {
	SuccessRate       float64           `json:"success_rate"`
	ErrorRate         float64           `json:"error_rate"`
	LatencyP50        time.Duration     `json:"latency_p50"`
	LatencyP95        time.Duration     `json:"latency_p95"`
	LatencyP99        time.Duration     `json:"latency_p99"`
	RequestsPerSecond float64           `json:"requests_per_second"`
	TotalRequests     int64             `json:"total_requests"`
	FailedRequests    int64             `json:"failed_requests"`
	SampleSize        int64             `json:"sample_size"`
	Timestamp         time.Time         `json:"timestamp"`
	CanaryMetrics     *ComponentMetrics `json:"canary_metrics"`
	BaselineMetrics   *ComponentMetrics `json:"baseline_metrics"`
}

// ComponentMetrics tracks metrics for a specific deployment component
type ComponentMetrics struct {
	RequestCount      int64         `json:"request_count"`
	ErrorCount        int64         `json:"error_count"`
	LatencySum        time.Duration `json:"latency_sum"`
	LatencyP50        time.Duration `json:"latency_p50"`
	LatencyP95        time.Duration `json:"latency_p95"`
	LatencyP99        time.Duration `json:"latency_p99"`
	MemoryUsageMB     float64       `json:"memory_usage_mb"`
	CPUUsagePercent   float64       `json:"cpu_usage_percent"`
	NetworkBytesIn    int64         `json:"network_bytes_in"`
	NetworkBytesOut   int64         `json:"network_bytes_out"`
}

// ABTestConfig configures A/B testing parameters
type ABTestConfig struct {
	Enabled                bool              `json:"enabled"`
	VariantA               string            `json:"variant_a"`
	VariantB               string            `json:"variant_b"`
	MetricName             string            `json:"metric_name"`
	SignificanceLevel      float64           `json:"significance_level"`
	MinimumSampleSize      int               `json:"minimum_sample_size"`
	ConfidenceInterval     float64           `json:"confidence_interval"`
	ObservedDifference     float64           `json:"observed_difference"`
	StatisticalSignificance bool             `json:"statistical_significance"`
	VariantMetrics         map[string]float64 `json:"variant_metrics"`
}

// GoldenSignals tracks the four golden signals of monitoring
type GoldenSignals struct {
	Latency     *LatencySignal     `json:"latency"`
	Traffic     *TrafficSignal     `json:"traffic"`
	Errors      *ErrorSignal       `json:"errors"`
	Saturation  *SaturationSignal  `json:"saturation"`
	Timestamp   time.Time          `json:"timestamp"`
	HealthScore float64            `json:"health_score"`
}

// LatencySignal tracks latency metrics
type LatencySignal struct {
	P50         time.Duration `json:"p50"`
	P95         time.Duration `json:"p95"`
	P99         time.Duration `json:"p99"`
	Mean        time.Duration `json:"mean"`
	Max         time.Duration `json:"max"`
	StdDev      time.Duration `json:"std_dev"`
	SampleCount int64         `json:"sample_count"`
}

// TrafficSignal tracks traffic patterns
type TrafficSignal struct {
	RequestsPerSecond float64 `json:"requests_per_second"`
	BytesPerSecond    float64 `json:"bytes_per_second"`
	ConnectionCount   int64   `json:"connection_count"`
	ActiveStreams     int64   `json:"active_streams"`
}

// ErrorSignal tracks error rates and types
type ErrorSignal struct {
	ErrorRate       float64           `json:"error_rate"`
	TotalErrors     int64             `json:"total_errors"`
	ErrorsByType    map[string]int64  `json:"errors_by_type"`
	CriticalErrors  int64             `json:"critical_errors"`
	RecoverableErrors int64           `json:"recoverable_errors"`
}

// SaturationSignal tracks resource utilization
type SaturationSignal struct {
	CPUUtilization    float64 `json:"cpu_utilization"`
	MemoryUtilization float64 `json:"memory_utilization"`
	DiskUtilization   float64 `json:"disk_utilization"`
	NetworkUtilization float64 `json:"network_utilization"`
	QueueDepth        int64   `json:"queue_depth"`
	ThreadPoolUsage   float64 `json:"thread_pool_usage"`
}

// CanaryController manages canary deployments with progressive rollout
type CanaryController struct {
	mu                sync.RWMutex
	deployments       map[string]*CanaryDeployment
	trafficRouter     TrafficRouter
	healthMonitor     HealthMonitor
	metricsCollector  MetricsCollector
	rollbackManager   RollbackManager
	abTestEngine      ABTestEngine
	featureFlagManager FeatureFlagManager
	signalsMonitor    GoldenSignalsMonitor
	ctx               context.Context
	cancel            context.CancelFunc
	wg                sync.WaitGroup
}

// TrafficRouter interface for routing traffic between baseline and canary
type TrafficRouter interface {
	RouteTraffic(ctx context.Context, deploymentID string, canaryPercent float64) error
	GetCurrentRouting(deploymentID string) (float64, error)
	ResetRouting(deploymentID string) error
}

// HealthMonitor interface for monitoring deployment health
type HealthMonitor interface {
	CheckHealth(ctx context.Context, deploymentID string, version string) (*HealthStatus, error)
	MonitorContinuous(ctx context.Context, deploymentID string) <-chan *HealthStatus
	GetHealthHistory(deploymentID string, duration time.Duration) ([]*HealthStatus, error)
}

// HealthStatus represents the health state of a deployment
type HealthStatus struct {
	DeploymentID    string    `json:"deployment_id"`
	Version         string    `json:"version"`
	Healthy         bool      `json:"healthy"`
	HealthScore     float64   `json:"health_score"`
	ChecksPassed    int       `json:"checks_passed"`
	ChecksFailed    int       `json:"checks_failed"`
	FailureReasons  []string  `json:"failure_reasons"`
	Timestamp       time.Time `json:"timestamp"`
	ResponseTime    time.Duration `json:"response_time"`
}

// MetricsCollector interface for collecting deployment metrics
type MetricsCollector interface {
	CollectMetrics(ctx context.Context, deploymentID string) (*DeploymentMetrics, error)
	CollectComponentMetrics(ctx context.Context, deploymentID string, version string) (*ComponentMetrics, error)
	CompareVersions(ctx context.Context, deploymentID string) (*MetricsComparison, error)
	RecordRequest(deploymentID string, version string, latency time.Duration, success bool)
}

// MetricsComparison compares metrics between baseline and canary
type MetricsComparison struct {
	DeploymentID       string        `json:"deployment_id"`
	Baseline           *ComponentMetrics `json:"baseline"`
	Canary             *ComponentMetrics `json:"canary"`
	LatencyDifference  time.Duration `json:"latency_difference"`
	ErrorRateDifference float64      `json:"error_rate_difference"`
	Significant        bool          `json:"significant"`
	Recommendation     string        `json:"recommendation"`
}

// RollbackManager interface for handling rollbacks
type RollbackManager interface {
	InitiateRollback(ctx context.Context, deploymentID string, reason string) error
	GetRollbackStatus(deploymentID string) (*RollbackStatus, error)
	CompleteRollback(deploymentID string) error
}

// RollbackStatus represents the status of a rollback operation
type RollbackStatus struct {
	DeploymentID string    `json:"deployment_id"`
	InProgress   bool      `json:"in_progress"`
	StartTime    time.Time `json:"start_time"`
	EndTime      *time.Time `json:"end_time,omitempty"`
	Reason       string    `json:"reason"`
	Success      bool      `json:"success"`
	Duration     time.Duration `json:"duration"`
}

// ABTestEngine interface for A/B testing
type ABTestEngine interface {
	StartTest(deploymentID string, config *ABTestConfig) error
	UpdateTest(deploymentID string, metrics map[string]float64) error
	EvaluateTest(deploymentID string) (*ABTestResult, error)
	StopTest(deploymentID string) error
}

// ABTestResult contains A/B test evaluation results
type ABTestResult struct {
	StatisticallySignificant bool    `json:"statistically_significant"`
	PValue                   float64 `json:"p_value"`
	ConfidenceInterval       [2]float64 `json:"confidence_interval"`
	Winner                   string  `json:"winner"`
	PercentImprovement       float64 `json:"percent_improvement"`
	SampleSizeSufficient     bool    `json:"sample_size_sufficient"`
	Recommendation           string  `json:"recommendation"`
}

// FeatureFlagManager interface for managing feature flags
type FeatureFlagManager interface {
	EnableFlag(deploymentID string, flag string) error
	DisableFlag(deploymentID string, flag string) error
	GetFlags(deploymentID string) (map[string]bool, error)
	EvaluateFlag(deploymentID string, flag string, context map[string]interface{}) (bool, error)
}

// GoldenSignalsMonitor interface for monitoring golden signals
type GoldenSignalsMonitor interface {
	CollectSignals(ctx context.Context, deploymentID string) (*GoldenSignals, error)
	CalculateHealthScore(signals *GoldenSignals) float64
	DetectAnomalies(signals *GoldenSignals, baseline *GoldenSignals) []Anomaly
}

// Anomaly represents a detected anomaly in golden signals
type Anomaly struct {
	Signal      string    `json:"signal"`
	Severity    string    `json:"severity"`
	Description string    `json:"description"`
	Value       float64   `json:"value"`
	Threshold   float64   `json:"threshold"`
	DetectedAt  time.Time `json:"detected_at"`
}

// NewCanaryController creates a new canary controller
func NewCanaryController(
	trafficRouter TrafficRouter,
	healthMonitor HealthMonitor,
	metricsCollector MetricsCollector,
	rollbackManager RollbackManager,
	abTestEngine ABTestEngine,
	featureFlagManager FeatureFlagManager,
	signalsMonitor GoldenSignalsMonitor,
) *CanaryController {
	ctx, cancel := context.WithCancel(context.Background())

	return &CanaryController{
		deployments:        make(map[string]*CanaryDeployment),
		trafficRouter:      trafficRouter,
		healthMonitor:      healthMonitor,
		metricsCollector:   metricsCollector,
		rollbackManager:    rollbackManager,
		abTestEngine:       abTestEngine,
		featureFlagManager: featureFlagManager,
		signalsMonitor:     signalsMonitor,
		ctx:                ctx,
		cancel:             cancel,
	}
}

// CreateDeployment creates a new canary deployment
func (cc *CanaryController) CreateDeployment(deployment *CanaryDeployment) error {
	cc.mu.Lock()
	defer cc.mu.Unlock()

	if deployment.ID == "" {
		return fmt.Errorf("deployment ID is required")
	}

	if _, exists := cc.deployments[deployment.ID]; exists {
		return fmt.Errorf("deployment %s already exists", deployment.ID)
	}

	// Set default stages if not provided
	if len(deployment.Stages) == 0 {
		deployment.Stages = cc.getDefaultStages()
	}

	// Initialize deployment
	deployment.Status = StatusInitializing
	deployment.CurrentStage = 0
	deployment.StartTime = time.Now()
	deployment.Metrics = &DeploymentMetrics{}
	deployment.GoldenSignals = &GoldenSignals{}

	cc.deployments[deployment.ID] = deployment

	// Start monitoring
	cc.wg.Add(1)
	go cc.monitorDeployment(deployment.ID)

	return nil
}

// getDefaultStages returns the default progressive rollout stages
func (cc *CanaryController) getDefaultStages() []CanaryStage {
	return []CanaryStage{
		{
			Name:            "Initial",
			TrafficPercent:  5.0,
			Duration:        5 * time.Minute,
			HealthThreshold: 0.99,
			ErrorThreshold:  0.01,
			LatencyP99Max:   500 * time.Millisecond,
		},
		{
			Name:            "Ramp-Up",
			TrafficPercent:  25.0,
			Duration:        10 * time.Minute,
			HealthThreshold: 0.99,
			ErrorThreshold:  0.01,
			LatencyP99Max:   500 * time.Millisecond,
		},
		{
			Name:            "Half",
			TrafficPercent:  50.0,
			Duration:        15 * time.Minute,
			HealthThreshold: 0.98,
			ErrorThreshold:  0.02,
			LatencyP99Max:   600 * time.Millisecond,
		},
		{
			Name:            "Majority",
			TrafficPercent:  75.0,
			Duration:        15 * time.Minute,
			HealthThreshold: 0.98,
			ErrorThreshold:  0.02,
			LatencyP99Max:   600 * time.Millisecond,
		},
		{
			Name:            "Full",
			TrafficPercent:  100.0,
			Duration:        10 * time.Minute,
			HealthThreshold: 0.98,
			ErrorThreshold:  0.02,
			LatencyP99Max:   600 * time.Millisecond,
		},
	}
}

// StartDeployment starts the canary deployment process
func (cc *CanaryController) StartDeployment(deploymentID string) error {
	cc.mu.Lock()
	deployment, exists := cc.deployments[deploymentID]
	if !exists {
		cc.mu.Unlock()
		return fmt.Errorf("deployment %s not found", deploymentID)
	}
	cc.mu.Unlock()

	// Start A/B test if configured
	if deployment.ABTest != nil && deployment.ABTest.Enabled {
		if err := cc.abTestEngine.StartTest(deploymentID, deployment.ABTest); err != nil {
			return fmt.Errorf("failed to start A/B test: %w", err)
		}
	}

	// Initialize feature flags
	if len(deployment.FeatureFlags) > 0 {
		for flag, enabled := range deployment.FeatureFlags {
			if enabled {
				if err := cc.featureFlagManager.EnableFlag(deploymentID, flag); err != nil {
					return fmt.Errorf("failed to enable feature flag %s: %w", flag, err)
				}
			}
		}
	}

	// Update status and start first stage
	cc.mu.Lock()
	deployment.Status = StatusProgressing
	cc.mu.Unlock()

	return cc.progressToNextStage(deploymentID)
}

// progressToNextStage progresses the deployment to the next stage
func (cc *CanaryController) progressToNextStage(deploymentID string) error {
	cc.mu.Lock()
	deployment, exists := cc.deployments[deploymentID]
	if !exists {
		cc.mu.Unlock()
		return fmt.Errorf("deployment %s not found", deploymentID)
	}

	if deployment.CurrentStage >= len(deployment.Stages) {
		cc.mu.Unlock()
		return cc.completeDeployment(deploymentID)
	}

	stage := deployment.Stages[deployment.CurrentStage]
	cc.mu.Unlock()

	// Route traffic to canary
	if err := cc.trafficRouter.RouteTraffic(cc.ctx, deploymentID, stage.TrafficPercent); err != nil {
		return fmt.Errorf("failed to route traffic: %w", err)
	}

	// Wait for stage duration while monitoring
	timer := time.NewTimer(stage.Duration)
	ticker := time.NewTicker(10 * time.Second)
	defer timer.Stop()
	defer ticker.Stop()

	for {
		select {
		case <-timer.C:
			// Stage complete, validate and progress
			if err := cc.validateStage(deploymentID); err != nil {
				return cc.initiateRollback(deploymentID, fmt.Sprintf("Stage validation failed: %v", err))
			}

			cc.mu.Lock()
			deployment.CurrentStage++
			cc.mu.Unlock()

			return cc.progressToNextStage(deploymentID)

		case <-ticker.C:
			// Continuous health check
			if err := cc.checkStageHealth(deploymentID); err != nil {
				return cc.initiateRollback(deploymentID, fmt.Sprintf("Health check failed: %v", err))
			}

		case <-cc.ctx.Done():
			return fmt.Errorf("deployment cancelled")
		}
	}
}

// validateStage validates the current deployment stage
func (cc *CanaryController) validateStage(deploymentID string) error {
	cc.mu.RLock()
	deployment, exists := cc.deployments[deploymentID]
	if !exists {
		cc.mu.RUnlock()
		return fmt.Errorf("deployment %s not found", deploymentID)
	}

	if deployment.CurrentStage >= len(deployment.Stages) {
		cc.mu.RUnlock()
		return nil
	}

	stage := deployment.Stages[deployment.CurrentStage]
	cc.mu.RUnlock()

	// Collect metrics
	metrics, err := cc.metricsCollector.CollectMetrics(cc.ctx, deploymentID)
	if err != nil {
		return fmt.Errorf("failed to collect metrics: %w", err)
	}

	// Collect golden signals
	signals, err := cc.signalsMonitor.CollectSignals(cc.ctx, deploymentID)
	if err != nil {
		return fmt.Errorf("failed to collect golden signals: %w", err)
	}

	// Calculate health score
	healthScore := cc.signalsMonitor.CalculateHealthScore(signals)

	// Validate against thresholds
	if metrics.ErrorRate > stage.ErrorThreshold {
		return fmt.Errorf("error rate %.4f exceeds threshold %.4f", metrics.ErrorRate, stage.ErrorThreshold)
	}

	if metrics.LatencyP99 > stage.LatencyP99Max {
		return fmt.Errorf("P99 latency %v exceeds max %v", metrics.LatencyP99, stage.LatencyP99Max)
	}

	if healthScore < stage.HealthThreshold {
		return fmt.Errorf("health score %.4f below threshold %.4f", healthScore, stage.HealthThreshold)
	}

	// Check golden signals for anomalies
	if deployment.GoldenSignals != nil {
		anomalies := cc.signalsMonitor.DetectAnomalies(signals, deployment.GoldenSignals)
		for _, anomaly := range anomalies {
			if anomaly.Severity == "critical" {
				return fmt.Errorf("critical anomaly detected in %s: %s", anomaly.Signal, anomaly.Description)
			}
		}
	}

	// Update deployment metrics and signals
	cc.mu.Lock()
	deployment.Metrics = metrics
	deployment.GoldenSignals = signals
	signals.HealthScore = healthScore
	cc.mu.Unlock()

	// Evaluate A/B test if running
	if deployment.ABTest != nil && deployment.ABTest.Enabled {
		result, err := cc.abTestEngine.EvaluateTest(deploymentID)
		if err != nil {
			return fmt.Errorf("failed to evaluate A/B test: %w", err)
		}

		if result.SampleSizeSufficient && !result.StatisticallySignificant {
			return fmt.Errorf("A/B test shows no significant improvement")
		}
	}

	return nil
}

// checkStageHealth performs continuous health monitoring during a stage
func (cc *CanaryController) checkStageHealth(deploymentID string) error {
	cc.mu.RLock()
	deployment, exists := cc.deployments[deploymentID]
	if !exists {
		cc.mu.RUnlock()
		return fmt.Errorf("deployment %s not found", deploymentID)
	}
	cc.mu.RUnlock()

	// Check canary health
	canaryHealth, err := cc.healthMonitor.CheckHealth(cc.ctx, deploymentID, deployment.CanaryVersion)
	if err != nil {
		return fmt.Errorf("failed to check canary health: %w", err)
	}

	if !canaryHealth.Healthy {
		return fmt.Errorf("canary unhealthy: %v", canaryHealth.FailureReasons)
	}

	// Check baseline health
	baselineHealth, err := cc.healthMonitor.CheckHealth(cc.ctx, deploymentID, deployment.BaselineVersion)
	if err != nil {
		return fmt.Errorf("failed to check baseline health: %w", err)
	}

	if !baselineHealth.Healthy {
		return fmt.Errorf("baseline unhealthy: %v", baselineHealth.FailureReasons)
	}

	// Compare metrics between versions
	comparison, err := cc.metricsCollector.CompareVersions(cc.ctx, deploymentID)
	if err != nil {
		return fmt.Errorf("failed to compare versions: %w", err)
	}

	// Check for significant degradation
	if comparison.ErrorRateDifference > 0.05 {
		return fmt.Errorf("canary error rate %.4f higher than baseline", comparison.ErrorRateDifference)
	}

	if comparison.LatencyDifference > 200*time.Millisecond {
		return fmt.Errorf("canary latency %v higher than baseline", comparison.LatencyDifference)
	}

	return nil
}

// initiateRollback initiates a rollback of the canary deployment
func (cc *CanaryController) initiateRollback(deploymentID string, reason string) error {
	cc.mu.Lock()
	deployment, exists := cc.deployments[deploymentID]
	if !exists {
		cc.mu.Unlock()
		return fmt.Errorf("deployment %s not found", deploymentID)
	}

	deployment.Status = StatusRollingBack
	cc.mu.Unlock()

	// Initiate rollback
	if err := cc.rollbackManager.InitiateRollback(cc.ctx, deploymentID, reason); err != nil {
		return fmt.Errorf("failed to initiate rollback: %w", err)
	}

	// Reset traffic to 100% baseline
	if err := cc.trafficRouter.ResetRouting(deploymentID); err != nil {
		return fmt.Errorf("failed to reset traffic routing: %w", err)
	}

	// Stop A/B test if running
	if deployment.ABTest != nil && deployment.ABTest.Enabled {
		_ = cc.abTestEngine.StopTest(deploymentID)
	}

	// Wait for rollback to complete
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			status, err := cc.rollbackManager.GetRollbackStatus(deploymentID)
			if err != nil {
				return fmt.Errorf("failed to get rollback status: %w", err)
			}

			if !status.InProgress {
				if status.Success {
					cc.mu.Lock()
					deployment.Status = StatusFailed
					now := time.Now()
					deployment.EndTime = &now
					cc.mu.Unlock()
					return nil
				}
				return fmt.Errorf("rollback failed")
			}

		case <-cc.ctx.Done():
			return fmt.Errorf("rollback cancelled")
		}
	}
}

// completeDeployment completes the canary deployment
func (cc *CanaryController) completeDeployment(deploymentID string) error {
	cc.mu.Lock()
	deployment, exists := cc.deployments[deploymentID]
	if !exists {
		cc.mu.Unlock()
		return fmt.Errorf("deployment %s not found", deploymentID)
	}

	deployment.Status = StatusCompleted
	now := time.Now()
	deployment.EndTime = &now
	cc.mu.Unlock()

	// Stop A/B test if running
	if deployment.ABTest != nil && deployment.ABTest.Enabled {
		result, _ := cc.abTestEngine.EvaluateTest(deploymentID)
		if result != nil {
			deployment.ABTest.StatisticalSignificance = result.StatisticallySignificant
			deployment.ABTest.ObservedDifference = result.PercentImprovement
		}
		_ = cc.abTestEngine.StopTest(deploymentID)
	}

	return nil
}

// monitorDeployment continuously monitors a deployment
func (cc *CanaryController) monitorDeployment(deploymentID string) {
	defer cc.wg.Done()

	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			cc.mu.RLock()
			deployment, exists := cc.deployments[deploymentID]
			if !exists {
				cc.mu.RUnlock()
				return
			}

			// Stop monitoring if deployment is complete or failed
			if deployment.Status == StatusCompleted || deployment.Status == StatusFailed {
				cc.mu.RUnlock()
				return
			}
			cc.mu.RUnlock()

			// Collect and update metrics
			metrics, err := cc.metricsCollector.CollectMetrics(cc.ctx, deploymentID)
			if err == nil {
				cc.mu.Lock()
				deployment.Metrics = metrics
				cc.mu.Unlock()
			}

			// Collect and update golden signals
			signals, err := cc.signalsMonitor.CollectSignals(cc.ctx, deploymentID)
			if err == nil {
				healthScore := cc.signalsMonitor.CalculateHealthScore(signals)
				signals.HealthScore = healthScore

				cc.mu.Lock()
				deployment.GoldenSignals = signals
				cc.mu.Unlock()
			}

		case <-cc.ctx.Done():
			return
		}
	}
}

// GetDeployment retrieves a deployment by ID
func (cc *CanaryController) GetDeployment(deploymentID string) (*CanaryDeployment, error) {
	cc.mu.RLock()
	defer cc.mu.RUnlock()

	deployment, exists := cc.deployments[deploymentID]
	if !exists {
		return nil, fmt.Errorf("deployment %s not found", deploymentID)
	}

	// Return a copy to prevent external modification
	deploymentCopy := *deployment
	return &deploymentCopy, nil
}

// ListDeployments lists all deployments
func (cc *CanaryController) ListDeployments() []*CanaryDeployment {
	cc.mu.RLock()
	defer cc.mu.RUnlock()

	deployments := make([]*CanaryDeployment, 0, len(cc.deployments))
	for _, deployment := range cc.deployments {
		deploymentCopy := *deployment
		deployments = append(deployments, &deploymentCopy)
	}

	return deployments
}

// PauseDeployment pauses a running deployment
func (cc *CanaryController) PauseDeployment(deploymentID string) error {
	cc.mu.Lock()
	defer cc.mu.Unlock()

	deployment, exists := cc.deployments[deploymentID]
	if !exists {
		return fmt.Errorf("deployment %s not found", deploymentID)
	}

	if deployment.Status != StatusProgressing {
		return fmt.Errorf("deployment is not in progressing state")
	}

	deployment.Status = StatusPaused
	return nil
}

// ResumeDeployment resumes a paused deployment
func (cc *CanaryController) ResumeDeployment(deploymentID string) error {
	cc.mu.Lock()
	deployment, exists := cc.deployments[deploymentID]
	if !exists {
		cc.mu.Unlock()
		return fmt.Errorf("deployment %s not found", deploymentID)
	}

	if deployment.Status != StatusPaused {
		cc.mu.Unlock()
		return fmt.Errorf("deployment is not paused")
	}

	deployment.Status = StatusProgressing
	cc.mu.Unlock()

	// Continue from current stage
	return cc.progressToNextStage(deploymentID)
}

// CancelDeployment cancels a deployment and rolls back
func (cc *CanaryController) CancelDeployment(deploymentID string) error {
	return cc.initiateRollback(deploymentID, "Deployment cancelled by user")
}

// GetDeploymentMetrics retrieves current metrics for a deployment
func (cc *CanaryController) GetDeploymentMetrics(deploymentID string) (*DeploymentMetrics, error) {
	cc.mu.RLock()
	defer cc.mu.RUnlock()

	deployment, exists := cc.deployments[deploymentID]
	if !exists {
		return nil, fmt.Errorf("deployment %s not found", deploymentID)
	}

	if deployment.Metrics == nil {
		return nil, fmt.Errorf("no metrics available yet")
	}

	metricsCopy := *deployment.Metrics
	return &metricsCopy, nil
}

// GetGoldenSignals retrieves current golden signals for a deployment
func (cc *CanaryController) GetGoldenSignals(deploymentID string) (*GoldenSignals, error) {
	cc.mu.RLock()
	defer cc.mu.RUnlock()

	deployment, exists := cc.deployments[deploymentID]
	if !exists {
		return nil, fmt.Errorf("deployment %s not found", deploymentID)
	}

	if deployment.GoldenSignals == nil {
		return nil, fmt.Errorf("no golden signals available yet")
	}

	signalsCopy := *deployment.GoldenSignals
	return &signalsCopy, nil
}

// Shutdown gracefully shuts down the canary controller
func (cc *CanaryController) Shutdown(ctx context.Context) error {
	cc.cancel()

	done := make(chan struct{})
	go func() {
		cc.wg.Wait()
		close(done)
	}()

	select {
	case <-done:
		return nil
	case <-ctx.Done():
		return fmt.Errorf("shutdown timeout exceeded")
	}
}

// CalculateHealthScore calculates overall health score from golden signals
func CalculateHealthScore(signals *GoldenSignals) float64 {
	if signals == nil {
		return 0.0
	}

	// Weight factors for each signal
	const (
		latencyWeight    = 0.25
		trafficWeight    = 0.20
		errorWeight      = 0.35
		saturationWeight = 0.20
	)

	// Calculate component scores (0-1 scale)
	latencyScore := calculateLatencyScore(signals.Latency)
	trafficScore := calculateTrafficScore(signals.Traffic)
	errorScore := 1.0 - signals.Errors.ErrorRate // Invert: lower error rate = higher score
	saturationScore := calculateSaturationScore(signals.Saturation)

	// Weighted average
	healthScore := (latencyScore * latencyWeight) +
		(trafficScore * trafficWeight) +
		(errorScore * errorWeight) +
		(saturationScore * saturationWeight)

	return math.Max(0.0, math.Min(1.0, healthScore))
}

// calculateLatencyScore calculates latency component score
func calculateLatencyScore(latency *LatencySignal) float64 {
	if latency == nil {
		return 0.0
	}

	// Target P99 latency: 500ms
	targetP99 := 500 * time.Millisecond

	if latency.P99 <= targetP99 {
		return 1.0
	}

	// Degrade score linearly up to 2x target
	ratio := float64(latency.P99) / float64(targetP99)
	score := 2.0 - ratio

	return math.Max(0.0, math.Min(1.0, score))
}

// calculateTrafficScore calculates traffic component score
func calculateTrafficScore(traffic *TrafficSignal) float64 {
	if traffic == nil {
		return 0.0
	}

	// Assume healthy if traffic is flowing
	if traffic.RequestsPerSecond > 0 && traffic.ConnectionCount > 0 {
		return 1.0
	}

	return 0.0
}

// calculateSaturationScore calculates saturation component score
func calculateSaturationScore(saturation *SaturationSignal) float64 {
	if saturation == nil {
		return 0.0
	}

	// Average utilization across resources
	avgUtilization := (saturation.CPUUtilization +
		saturation.MemoryUtilization +
		saturation.DiskUtilization +
		saturation.NetworkUtilization) / 4.0

	// Optimal utilization: 50-70%, degrade outside this range
	if avgUtilization >= 0.5 && avgUtilization <= 0.7 {
		return 1.0
	}

	if avgUtilization < 0.5 {
		// Under-utilized
		return avgUtilization / 0.5
	}

	// Over-utilized (>70%)
	if avgUtilization > 0.9 {
		return 0.0
	}

	return (1.0 - avgUtilization) / 0.3
}

// MarshalJSON implements custom JSON marshaling for CanaryDeployment
func (cd *CanaryDeployment) MarshalJSON() ([]byte, error) {
	type Alias CanaryDeployment

	return json.Marshal(&struct {
		*Alias
		Duration string `json:"duration,omitempty"`
	}{
		Alias: (*Alias)(cd),
		Duration: func() string {
			if cd.EndTime != nil {
				return cd.EndTime.Sub(cd.StartTime).String()
			}
			return time.Since(cd.StartTime).String()
		}(),
	})
}

// String returns a string representation of deployment status
func (ds DeploymentStatus) String() string {
	return string(ds)
}

// IsTerminal returns true if the deployment status is terminal
func (ds DeploymentStatus) IsTerminal() bool {
	return ds == StatusCompleted || ds == StatusFailed
}

// IsActive returns true if the deployment is actively processing
func (ds DeploymentStatus) IsActive() bool {
	return ds == StatusProgressing || ds == StatusValidating
}
