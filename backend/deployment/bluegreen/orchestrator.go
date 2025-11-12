// Package bluegreen provides zero-downtime deployment capabilities using blue-green
// deployment strategy with environment cloning, warm-up, and instant traffic switching.
package bluegreen

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"
)

// Environment represents a blue or green deployment environment
type Environment struct {
	Name              string                 `json:"name"`
	Color             EnvironmentColor       `json:"color"`
	Version           string                 `json:"version"`
	Status            EnvironmentStatus      `json:"status"`
	HealthScore       float64                `json:"health_score"`
	ReadinessChecks   []ReadinessCheck       `json:"readiness_checks"`
	WarmupStatus      *WarmupStatus          `json:"warmup_status,omitempty"`
	ResourceUsage     *ResourceUsage         `json:"resource_usage"`
	DatabaseMigration *MigrationStatus       `json:"database_migration,omitempty"`
	Replicas          int                    `json:"replicas"`
	CreatedAt         time.Time              `json:"created_at"`
	UpdatedAt         time.Time              `json:"updated_at"`
	Configuration     map[string]interface{} `json:"configuration"`
	Endpoints         []string               `json:"endpoints"`
}

// EnvironmentColor represents the color of an environment
type EnvironmentColor string

const (
	ColorBlue  EnvironmentColor = "blue"
	ColorGreen EnvironmentColor = "green"
)

// EnvironmentStatus represents the status of an environment
type EnvironmentStatus string

const (
	StatusProvisioning EnvironmentStatus = "provisioning"
	StatusWarmingUp    EnvironmentStatus = "warming_up"
	StatusReady        EnvironmentStatus = "ready"
	StatusActive       EnvironmentStatus = "active"
	StatusDraining     EnvironmentStatus = "draining"
	StatusTerminating  EnvironmentStatus = "terminating"
	StatusTerminated   EnvironmentStatus = "terminated"
	StatusFailed       EnvironmentStatus = "failed"
)

// ReadinessCheck represents a health/readiness check for an environment
type ReadinessCheck struct {
	Name        string        `json:"name"`
	Type        string        `json:"type"`
	Passed      bool          `json:"passed"`
	Message     string        `json:"message"`
	CheckedAt   time.Time     `json:"checked_at"`
	Duration    time.Duration `json:"duration"`
	Retries     int           `json:"retries"`
	MaxRetries  int           `json:"max_retries"`
}

// WarmupStatus tracks the warm-up process of an environment
type WarmupStatus struct {
	Started        bool          `json:"started"`
	Progress       float64       `json:"progress"`
	Phase          WarmupPhase   `json:"phase"`
	RequestsSent   int64         `json:"requests_sent"`
	SuccessRate    float64       `json:"success_rate"`
	AvgLatency     time.Duration `json:"avg_latency"`
	EstimatedTime  time.Duration `json:"estimated_time"`
	StartTime      time.Time     `json:"start_time"`
	EndTime        *time.Time    `json:"end_time,omitempty"`
}

// WarmupPhase represents a phase in the warm-up process
type WarmupPhase string

const (
	WarmupPhaseInit      WarmupPhase = "initialization"
	WarmupPhaseCache     WarmupPhase = "cache_warming"
	WarmupPhaseLoad      WarmupPhase = "load_testing"
	WarmupPhaseValidate  WarmupPhase = "validation"
	WarmupPhaseComplete  WarmupPhase = "complete"
)

// ResourceUsage tracks resource utilization of an environment
type ResourceUsage struct {
	CPUCores      float64   `json:"cpu_cores"`
	CPUPercent    float64   `json:"cpu_percent"`
	MemoryMB      float64   `json:"memory_mb"`
	MemoryPercent float64   `json:"memory_percent"`
	DiskGB        float64   `json:"disk_gb"`
	NetworkMbps   float64   `json:"network_mbps"`
	PodCount      int       `json:"pod_count"`
	Timestamp     time.Time `json:"timestamp"`
}

// MigrationStatus tracks database migration status
type MigrationStatus struct {
	Required       bool                   `json:"required"`
	Type           MigrationType          `json:"type"`
	Status         MigrationStatusType    `json:"status"`
	Version        string                 `json:"version"`
	Scripts        []MigrationScript      `json:"scripts"`
	Progress       float64                `json:"progress"`
	StartTime      time.Time              `json:"start_time"`
	EndTime        *time.Time             `json:"end_time,omitempty"`
	RollbackScript *MigrationScript       `json:"rollback_script,omitempty"`
	Errors         []string               `json:"errors,omitempty"`
}

// MigrationType represents the type of database migration
type MigrationType string

const (
	MigrationTypeForward  MigrationType = "forward"
	MigrationTypeBackward MigrationType = "backward"
	MigrationTypeBlueGreen MigrationType = "blue_green"
)

// MigrationStatusType represents migration execution status
type MigrationStatusType string

const (
	MigrationStatusPending   MigrationStatusType = "pending"
	MigrationStatusRunning   MigrationStatusType = "running"
	MigrationStatusCompleted MigrationStatusType = "completed"
	MigrationStatusFailed    MigrationStatusType = "failed"
	MigrationStatusRolledBack MigrationStatusType = "rolled_back"
)

// MigrationScript represents a database migration script
type MigrationScript struct {
	Name        string    `json:"name"`
	Version     string    `json:"version"`
	Content     string    `json:"content"`
	Checksum    string    `json:"checksum"`
	ExecutedAt  *time.Time `json:"executed_at,omitempty"`
	Duration    time.Duration `json:"duration"`
	Success     bool      `json:"success"`
	Error       string    `json:"error,omitempty"`
}

// BlueGreenDeployment represents a blue-green deployment
type BlueGreenDeployment struct {
	ID                string            `json:"id"`
	Name              string            `json:"name"`
	Namespace         string            `json:"namespace"`
	BlueEnvironment   *Environment      `json:"blue_environment"`
	GreenEnvironment  *Environment      `json:"green_environment"`
	ActiveEnvironment EnvironmentColor  `json:"active_environment"`
	TargetEnvironment EnvironmentColor  `json:"target_environment"`
	Status            DeploymentStatus  `json:"status"`
	TrafficSplit      *TrafficSplit     `json:"traffic_split"`
	SwitchStrategy    *SwitchStrategy   `json:"switch_strategy"`
	StartTime         time.Time         `json:"start_time"`
	EndTime           *time.Time        `json:"end_time,omitempty"`
	RollbackConfig    *RollbackConfig   `json:"rollback_config"`
}

// DeploymentStatus represents the status of a blue-green deployment
type DeploymentStatus string

const (
	DeploymentStatusInitializing DeploymentStatus = "initializing"
	DeploymentStatusProvisioning DeploymentStatus = "provisioning"
	DeploymentStatusWarmingUp    DeploymentStatus = "warming_up"
	DeploymentStatusValidating   DeploymentStatus = "validating"
	DeploymentStatusSwitching    DeploymentStatus = "switching"
	DeploymentStatusCompleted    DeploymentStatus = "completed"
	DeploymentStatusRollingBack  DeploymentStatus = "rolling_back"
	DeploymentStatusFailed       DeploymentStatus = "failed"
)

// TrafficSplit represents traffic distribution between environments
type TrafficSplit struct {
	Blue      float64   `json:"blue"`
	Green     float64   `json:"green"`
	UpdatedAt time.Time `json:"updated_at"`
}

// SwitchStrategy defines how traffic switching occurs
type SwitchStrategy struct {
	Type               SwitchType    `json:"type"`
	ValidationDuration time.Duration `json:"validation_duration"`
	DrainDuration      time.Duration `json:"drain_duration"`
	AutoRollback       bool          `json:"auto_rollback"`
	RollbackThreshold  *Threshold    `json:"rollback_threshold"`
}

// SwitchType represents the type of traffic switch
type SwitchType string

const (
	SwitchTypeInstant    SwitchType = "instant"
	SwitchTypeGradual    SwitchType = "gradual"
	SwitchTypeValidated  SwitchType = "validated"
)

// Threshold defines thresholds for auto-rollback
type Threshold struct {
	ErrorRatePercent  float64       `json:"error_rate_percent"`
	LatencyP99Max     time.Duration `json:"latency_p99_max"`
	HealthScoreMin    float64       `json:"health_score_min"`
}

// RollbackConfig configures rollback behavior
type RollbackConfig struct {
	Enabled           bool          `json:"enabled"`
	AutoRollback      bool          `json:"auto_rollback"`
	MaxDuration       time.Duration `json:"max_duration"`
	PreserveData      bool          `json:"preserve_data"`
	NotifyOnRollback  bool          `json:"notify_on_rollback"`
}

// BlueGreenOrchestrator orchestrates blue-green deployments
type BlueGreenOrchestrator struct {
	mu                  sync.RWMutex
	deployments         map[string]*BlueGreenDeployment
	environmentManager  EnvironmentManager
	trafficController   TrafficController
	migrationManager    MigrationManager
	healthChecker       HealthChecker
	warmupController    WarmupController
	ctx                 context.Context
	cancel              context.CancelFunc
	wg                  sync.WaitGroup
}

// EnvironmentManager manages environment lifecycle
type EnvironmentManager interface {
	ProvisionEnvironment(ctx context.Context, config *EnvironmentConfig) (*Environment, error)
	CloneEnvironment(ctx context.Context, source *Environment, targetVersion string) (*Environment, error)
	TerminateEnvironment(ctx context.Context, env *Environment) error
	GetEnvironmentStatus(ctx context.Context, envName string) (*Environment, error)
	ScaleEnvironment(ctx context.Context, envName string, replicas int) error
}

// EnvironmentConfig configures a new environment
type EnvironmentConfig struct {
	Name          string                 `json:"name"`
	Color         EnvironmentColor       `json:"color"`
	Version       string                 `json:"version"`
	Replicas      int                    `json:"replicas"`
	Resources     *ResourceRequirements  `json:"resources"`
	Configuration map[string]interface{} `json:"configuration"`
}

// ResourceRequirements specifies resource requirements
type ResourceRequirements struct {
	CPUCores    float64 `json:"cpu_cores"`
	MemoryMB    float64 `json:"memory_mb"`
	DiskGB      float64 `json:"disk_gb"`
	NetworkMbps float64 `json:"network_mbps"`
}

// TrafficController controls traffic routing
type TrafficController interface {
	SwitchTraffic(ctx context.Context, deploymentID string, target EnvironmentColor) error
	SplitTraffic(ctx context.Context, deploymentID string, split *TrafficSplit) error
	GetTrafficSplit(deploymentID string) (*TrafficSplit, error)
	DrainEnvironment(ctx context.Context, envName string, duration time.Duration) error
}

// MigrationManager manages database migrations
type MigrationManager interface {
	PlanMigration(ctx context.Context, fromVersion, toVersion string) (*MigrationPlan, error)
	ExecuteMigration(ctx context.Context, plan *MigrationPlan) (*MigrationStatus, error)
	RollbackMigration(ctx context.Context, status *MigrationStatus) error
	ValidateMigration(ctx context.Context, status *MigrationStatus) error
}

// MigrationPlan represents a planned database migration
type MigrationPlan struct {
	FromVersion    string              `json:"from_version"`
	ToVersion      string              `json:"to_version"`
	Type           MigrationType       `json:"type"`
	Scripts        []MigrationScript   `json:"scripts"`
	EstimatedTime  time.Duration       `json:"estimated_time"`
	RollbackScript *MigrationScript    `json:"rollback_script"`
	RequiresDowntime bool              `json:"requires_downtime"`
}

// HealthChecker performs health checks
type HealthChecker interface {
	CheckEnvironmentHealth(ctx context.Context, env *Environment) ([]ReadinessCheck, error)
	ContinuousHealthCheck(ctx context.Context, envName string) <-chan HealthCheckResult
	ValidateEnvironment(ctx context.Context, env *Environment) error
}

// HealthCheckResult represents a health check result
type HealthCheckResult struct {
	Environment string           `json:"environment"`
	Checks      []ReadinessCheck `json:"checks"`
	AllPassed   bool             `json:"all_passed"`
	Timestamp   time.Time        `json:"timestamp"`
}

// WarmupController manages environment warm-up
type WarmupController interface {
	StartWarmup(ctx context.Context, env *Environment, config *WarmupConfig) error
	GetWarmupStatus(envName string) (*WarmupStatus, error)
	WaitForWarmup(ctx context.Context, envName string, timeout time.Duration) error
}

// WarmupConfig configures environment warm-up
type WarmupConfig struct {
	CacheWarmup       bool          `json:"cache_warmup"`
	LoadTest          bool          `json:"load_test"`
	RequestsPerSecond int           `json:"requests_per_second"`
	Duration          time.Duration `json:"duration"`
	TargetEndpoints   []string      `json:"target_endpoints"`
}

// NewBlueGreenOrchestrator creates a new blue-green orchestrator
func NewBlueGreenOrchestrator(
	envManager EnvironmentManager,
	trafficController TrafficController,
	migrationManager MigrationManager,
	healthChecker HealthChecker,
	warmupController WarmupController,
) *BlueGreenOrchestrator {
	ctx, cancel := context.WithCancel(context.Background())

	return &BlueGreenOrchestrator{
		deployments:        make(map[string]*BlueGreenDeployment),
		environmentManager: envManager,
		trafficController:  trafficController,
		migrationManager:   migrationManager,
		healthChecker:      healthChecker,
		warmupController:   warmupController,
		ctx:                ctx,
		cancel:             cancel,
	}
}

// CreateDeployment creates a new blue-green deployment
func (bgo *BlueGreenOrchestrator) CreateDeployment(
	name, namespace, currentVersion, targetVersion string,
	config *DeploymentConfig,
) (*BlueGreenDeployment, error) {
	bgo.mu.Lock()
	defer bgo.mu.Unlock()

	deploymentID := fmt.Sprintf("%s-%d", name, time.Now().Unix())

	deployment := &BlueGreenDeployment{
		ID:                deploymentID,
		Name:              name,
		Namespace:         namespace,
		ActiveEnvironment: ColorBlue,
		TargetEnvironment: ColorGreen,
		Status:            DeploymentStatusInitializing,
		StartTime:         time.Now(),
		SwitchStrategy:    config.SwitchStrategy,
		RollbackConfig:    config.RollbackConfig,
	}

	// Create blue environment (current version)
	blueEnv := &Environment{
		Name:    fmt.Sprintf("%s-blue", name),
		Color:   ColorBlue,
		Version: currentVersion,
		Status:  StatusActive,
		Replicas: config.Replicas,
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}
	deployment.BlueEnvironment = blueEnv

	// Initialize traffic split (100% blue)
	deployment.TrafficSplit = &TrafficSplit{
		Blue:      100.0,
		Green:     0.0,
		UpdatedAt: time.Now(),
	}

	bgo.deployments[deploymentID] = deployment

	return deployment, nil
}

// DeploymentConfig configures a deployment
type DeploymentConfig struct {
	Replicas         int              `json:"replicas"`
	SwitchStrategy   *SwitchStrategy  `json:"switch_strategy"`
	RollbackConfig   *RollbackConfig  `json:"rollback_config"`
	WarmupConfig     *WarmupConfig    `json:"warmup_config"`
	MigrationEnabled bool             `json:"migration_enabled"`
}

// StartDeployment starts the blue-green deployment process
func (bgo *BlueGreenOrchestrator) StartDeployment(
	deploymentID string,
	targetVersion string,
	config *EnvironmentConfig,
) error {
	bgo.mu.Lock()
	deployment, exists := bgo.deployments[deploymentID]
	if !exists {
		bgo.mu.Unlock()
		return fmt.Errorf("deployment %s not found", deploymentID)
	}
	deployment.Status = DeploymentStatusProvisioning
	bgo.mu.Unlock()

	// Provision green environment
	greenEnv, err := bgo.provisionGreenEnvironment(deployment, targetVersion, config)
	if err != nil {
		return fmt.Errorf("failed to provision green environment: %w", err)
	}

	bgo.mu.Lock()
	deployment.GreenEnvironment = greenEnv
	bgo.mu.Unlock()

	// Execute database migration if needed
	if config.Configuration != nil && config.Configuration["migration_enabled"].(bool) {
		if err := bgo.executeMigration(deployment, targetVersion); err != nil {
			return fmt.Errorf("failed to execute migration: %w", err)
		}
	}

	// Warm up green environment
	if err := bgo.warmupEnvironment(deployment, greenEnv); err != nil {
		return fmt.Errorf("failed to warm up green environment: %w", err)
	}

	// Validate green environment
	if err := bgo.validateEnvironment(deployment, greenEnv); err != nil {
		return fmt.Errorf("green environment validation failed: %w", err)
	}

	// Switch traffic
	if err := bgo.switchTraffic(deployment); err != nil {
		return fmt.Errorf("failed to switch traffic: %w", err)
	}

	// Complete deployment
	return bgo.completeDeployment(deployment)
}

// provisionGreenEnvironment provisions the green environment
func (bgo *BlueGreenOrchestrator) provisionGreenEnvironment(
	deployment *BlueGreenDeployment,
	version string,
	config *EnvironmentConfig,
) (*Environment, error) {
	config.Name = fmt.Sprintf("%s-green", deployment.Name)
	config.Color = ColorGreen
	config.Version = version

	ctx, cancel := context.WithTimeout(bgo.ctx, 10*time.Minute)
	defer cancel()

	env, err := bgo.environmentManager.ProvisionEnvironment(ctx, config)
	if err != nil {
		return nil, err
	}

	env.Status = StatusProvisioning
	return env, nil
}

// executeMigration executes database migration
func (bgo *BlueGreenOrchestrator) executeMigration(
	deployment *BlueGreenDeployment,
	targetVersion string,
) error {
	ctx, cancel := context.WithTimeout(bgo.ctx, 30*time.Minute)
	defer cancel()

	currentVersion := deployment.BlueEnvironment.Version

	// Plan migration
	plan, err := bgo.migrationManager.PlanMigration(ctx, currentVersion, targetVersion)
	if err != nil {
		return fmt.Errorf("failed to plan migration: %w", err)
	}

	// Execute migration
	status, err := bgo.migrationManager.ExecuteMigration(ctx, plan)
	if err != nil {
		return fmt.Errorf("failed to execute migration: %w", err)
	}

	// Validate migration
	if err := bgo.migrationManager.ValidateMigration(ctx, status); err != nil {
		// Rollback migration
		_ = bgo.migrationManager.RollbackMigration(ctx, status)
		return fmt.Errorf("migration validation failed: %w", err)
	}

	bgo.mu.Lock()
	deployment.GreenEnvironment.DatabaseMigration = status
	bgo.mu.Unlock()

	return nil
}

// warmupEnvironment warms up the environment
func (bgo *BlueGreenOrchestrator) warmupEnvironment(
	deployment *BlueGreenDeployment,
	env *Environment,
) error {
	bgo.mu.Lock()
	deployment.Status = DeploymentStatusWarmingUp
	env.Status = StatusWarmingUp
	bgo.mu.Unlock()

	warmupConfig := &WarmupConfig{
		CacheWarmup:       true,
		LoadTest:          true,
		RequestsPerSecond: 100,
		Duration:          5 * time.Minute,
		TargetEndpoints:   env.Endpoints,
	}

	ctx, cancel := context.WithTimeout(bgo.ctx, 10*time.Minute)
	defer cancel()

	if err := bgo.warmupController.StartWarmup(ctx, env, warmupConfig); err != nil {
		return err
	}

	// Wait for warmup to complete
	if err := bgo.warmupController.WaitForWarmup(ctx, env.Name, 10*time.Minute); err != nil {
		return err
	}

	bgo.mu.Lock()
	env.Status = StatusReady
	bgo.mu.Unlock()

	return nil
}

// validateEnvironment validates the environment is ready
func (bgo *BlueGreenOrchestrator) validateEnvironment(
	deployment *BlueGreenDeployment,
	env *Environment,
) error {
	bgo.mu.Lock()
	deployment.Status = DeploymentStatusValidating
	bgo.mu.Unlock()

	ctx, cancel := context.WithTimeout(bgo.ctx, 5*time.Minute)
	defer cancel()

	// Perform health checks
	checks, err := bgo.healthChecker.CheckEnvironmentHealth(ctx, env)
	if err != nil {
		return err
	}

	// Verify all checks passed
	allPassed := true
	for _, check := range checks {
		if !check.Passed {
			allPassed = false
			break
		}
	}

	if !allPassed {
		return fmt.Errorf("environment validation failed: not all health checks passed")
	}

	bgo.mu.Lock()
	env.ReadinessChecks = checks
	env.HealthScore = 1.0
	bgo.mu.Unlock()

	return nil
}

// switchTraffic switches traffic from blue to green
func (bgo *BlueGreenOrchestrator) switchTraffic(deployment *BlueGreenDeployment) error {
	bgo.mu.Lock()
	deployment.Status = DeploymentStatusSwitching
	strategy := deployment.SwitchStrategy
	bgo.mu.Unlock()

	ctx, cancel := context.WithTimeout(bgo.ctx, 10*time.Minute)
	defer cancel()

	switch strategy.Type {
	case SwitchTypeInstant:
		return bgo.instantSwitch(ctx, deployment)
	case SwitchTypeGradual:
		return bgo.gradualSwitch(ctx, deployment)
	case SwitchTypeValidated:
		return bgo.validatedSwitch(ctx, deployment)
	default:
		return fmt.Errorf("unknown switch type: %s", strategy.Type)
	}
}

// instantSwitch performs instant traffic switch
func (bgo *BlueGreenOrchestrator) instantSwitch(
	ctx context.Context,
	deployment *BlueGreenDeployment,
) error {
	// Switch all traffic to green
	if err := bgo.trafficController.SwitchTraffic(ctx, deployment.ID, ColorGreen); err != nil {
		return err
	}

	bgo.mu.Lock()
	deployment.TrafficSplit = &TrafficSplit{
		Blue:      0.0,
		Green:     100.0,
		UpdatedAt: time.Now(),
	}
	deployment.ActiveEnvironment = ColorGreen
	deployment.GreenEnvironment.Status = StatusActive
	bgo.mu.Unlock()

	// Drain blue environment
	return bgo.drainBlueEnvironment(ctx, deployment)
}

// gradualSwitch performs gradual traffic switch
func (bgo *BlueGreenOrchestrator) gradualSwitch(
	ctx context.Context,
	deployment *BlueGreenDeployment,
) error {
	// Gradually shift traffic: 75% -> 50% -> 25% -> 0% blue
	splits := []TrafficSplit{
		{Blue: 75.0, Green: 25.0},
		{Blue: 50.0, Green: 50.0},
		{Blue: 25.0, Green: 75.0},
		{Blue: 0.0, Green: 100.0},
	}

	for _, split := range splits {
		if err := bgo.trafficController.SplitTraffic(ctx, deployment.ID, &split); err != nil {
			return err
		}

		bgo.mu.Lock()
		deployment.TrafficSplit = &split
		bgo.mu.Unlock()

		// Wait between splits
		time.Sleep(2 * time.Minute)

		// Validate health
		if err := bgo.healthChecker.ValidateEnvironment(ctx, deployment.GreenEnvironment); err != nil {
			// Rollback if health check fails
			return bgo.rollbackDeployment(ctx, deployment, "Health check failed during gradual switch")
		}
	}

	bgo.mu.Lock()
	deployment.ActiveEnvironment = ColorGreen
	deployment.GreenEnvironment.Status = StatusActive
	bgo.mu.Unlock()

	return bgo.drainBlueEnvironment(ctx, deployment)
}

// validatedSwitch performs validated traffic switch with auto-rollback
func (bgo *BlueGreenOrchestrator) validatedSwitch(
	ctx context.Context,
	deployment *BlueGreenDeployment,
) error {
	// Start with small traffic to green for validation
	validationSplit := &TrafficSplit{
		Blue:  90.0,
		Green: 10.0,
	}

	if err := bgo.trafficController.SplitTraffic(ctx, deployment.ID, validationSplit); err != nil {
		return err
	}

	bgo.mu.Lock()
	deployment.TrafficSplit = validationSplit
	bgo.mu.Unlock()

	// Monitor for validation duration
	validationCtx, cancel := context.WithTimeout(ctx, deployment.SwitchStrategy.ValidationDuration)
	defer cancel()

	healthChan := bgo.healthChecker.ContinuousHealthCheck(validationCtx, deployment.GreenEnvironment.Name)

	validationSuccess := true
	for {
		select {
		case result := <-healthChan:
			if !result.AllPassed {
				validationSuccess = false
				return bgo.rollbackDeployment(ctx, deployment, "Validation health checks failed")
			}

		case <-validationCtx.Done():
			// Validation period complete
			if validationSuccess {
				// Proceed with full switch
				return bgo.instantSwitch(ctx, deployment)
			}
			return fmt.Errorf("validation failed")
		}
	}
}

// drainBlueEnvironment drains the blue environment
func (bgo *BlueGreenOrchestrator) drainBlueEnvironment(
	ctx context.Context,
	deployment *BlueGreenDeployment,
) error {
	bgo.mu.Lock()
	deployment.BlueEnvironment.Status = StatusDraining
	drainDuration := deployment.SwitchStrategy.DrainDuration
	bgo.mu.Unlock()

	if drainDuration == 0 {
		drainDuration = 5 * time.Minute
	}

	return bgo.trafficController.DrainEnvironment(ctx, deployment.BlueEnvironment.Name, drainDuration)
}

// completeDeployment completes the deployment process
func (bgo *BlueGreenOrchestrator) completeDeployment(deployment *BlueGreenDeployment) error {
	bgo.mu.Lock()
	defer bgo.mu.Unlock()

	deployment.Status = DeploymentStatusCompleted
	now := time.Now()
	deployment.EndTime = &now

	return nil
}

// rollbackDeployment rolls back a deployment
func (bgo *BlueGreenOrchestrator) rollbackDeployment(
	ctx context.Context,
	deployment *BlueGreenDeployment,
	reason string,
) error {
	bgo.mu.Lock()
	deployment.Status = DeploymentStatusRollingBack
	bgo.mu.Unlock()

	// Switch traffic back to blue
	if err := bgo.trafficController.SwitchTraffic(ctx, deployment.ID, ColorBlue); err != nil {
		return fmt.Errorf("failed to switch traffic back to blue: %w", err)
	}

	bgo.mu.Lock()
	deployment.TrafficSplit = &TrafficSplit{
		Blue:      100.0,
		Green:     0.0,
		UpdatedAt: time.Now(),
	}
	deployment.ActiveEnvironment = ColorBlue
	bgo.mu.Unlock()

	// Rollback database migration if performed
	if deployment.GreenEnvironment.DatabaseMigration != nil {
		if err := bgo.migrationManager.RollbackMigration(ctx, deployment.GreenEnvironment.DatabaseMigration); err != nil {
			return fmt.Errorf("failed to rollback migration: %w", err)
		}
	}

	// Terminate green environment
	if err := bgo.environmentManager.TerminateEnvironment(ctx, deployment.GreenEnvironment); err != nil {
		return fmt.Errorf("failed to terminate green environment: %w", err)
	}

	bgo.mu.Lock()
	deployment.Status = DeploymentStatusFailed
	now := time.Now()
	deployment.EndTime = &now
	bgo.mu.Unlock()

	return nil
}

// GetDeployment retrieves a deployment by ID
func (bgo *BlueGreenOrchestrator) GetDeployment(deploymentID string) (*BlueGreenDeployment, error) {
	bgo.mu.RLock()
	defer bgo.mu.RUnlock()

	deployment, exists := bgo.deployments[deploymentID]
	if !exists {
		return nil, fmt.Errorf("deployment %s not found", deploymentID)
	}

	return deployment, nil
}

// ListDeployments lists all deployments
func (bgo *BlueGreenOrchestrator) ListDeployments() []*BlueGreenDeployment {
	bgo.mu.RLock()
	defer bgo.mu.RUnlock()

	deployments := make([]*BlueGreenDeployment, 0, len(bgo.deployments))
	for _, deployment := range bgo.deployments {
		deployments = append(deployments, deployment)
	}

	return deployments
}

// Shutdown gracefully shuts down the orchestrator
func (bgo *BlueGreenOrchestrator) Shutdown(ctx context.Context) error {
	bgo.cancel()

	done := make(chan struct{})
	go func() {
		bgo.wg.Wait()
		close(done)
	}()

	select {
	case <-done:
		return nil
	case <-ctx.Done():
		return fmt.Errorf("shutdown timeout exceeded")
	}
}

// MarshalJSON implements custom JSON marshaling
func (e *Environment) MarshalJSON() ([]byte, error) {
	type Alias Environment

	return json.Marshal(&struct {
		*Alias
		Age string `json:"age"`
	}{
		Alias: (*Alias)(e),
		Age:   time.Since(e.CreatedAt).String(),
	})
}
