package deployment

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"sync"
	"time"

	"github.com/gorilla/mux"
	"github.com/prometheus/client_golang/api"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// ZeroDowntimeDeployer manages zero-downtime deployments for NovaCron
type ZeroDowntimeDeployer struct {
	config                *DeploymentConfig
	blueGreenManager     *BlueGreenManager
	progressiveDelivery  *ProgressiveDelivery
	verificationService  *DeploymentVerification
	trafficManager      *TrafficManager
	rollbackManager     *RollbackManager
	gitOpsController    *GitOpsController
	metricsCollector    *DeploymentMetrics
	
	// Synchronization
	mu                  sync.RWMutex
	activeDeployments   map[string]*DeploymentSession
	
	// Monitoring
	deploymentGauge     prometheus.Gauge
	errorRate          prometheus.Counter
	deploymentDuration prometheus.Histogram
}

// DeploymentConfig holds configuration for zero-downtime deployments
type DeploymentConfig struct {
	Environment                string                `json:"environment"`
	BlueGreenConfig           *BlueGreenConfig      `json:"blue_green_config"`
	ProgressiveDeliveryConfig *ProgressiveConfig    `json:"progressive_delivery_config"`
	VerificationConfig        *VerificationConfig   `json:"verification_config"`
	TrafficConfig            *TrafficConfig        `json:"traffic_config"`
	RollbackConfig           *RollbackConfig       `json:"rollback_config"`
	GitOpsConfig             *GitOpsConfig         `json:"gitops_config"`
	MetricsConfig            *MetricsConfig        `json:"metrics_config"`
	
	// Global settings
	MaxDeploymentTime         time.Duration         `json:"max_deployment_time"`
	RollbackThreshold         float64               `json:"rollback_threshold"`
	HealthCheckInterval       time.Duration         `json:"health_check_interval"`
	DatabaseMigrationTimeout  time.Duration         `json:"db_migration_timeout"`
	SessionPreservationTime   time.Duration         `json:"session_preservation_time"`
}

// DeploymentSession represents an active deployment
type DeploymentSession struct {
	ID                 string                 `json:"id"`
	Version            string                 `json:"version"`
	Environment        string                 `json:"environment"`
	Status            DeploymentStatus       `json:"status"`
	StartTime         time.Time              `json:"start_time"`
	Stages            []DeploymentStage      `json:"stages"`
	CurrentStage      int                    `json:"current_stage"`
	Metrics           *SessionMetrics        `json:"metrics"`
	RollbackPlan      *RollbackPlan         `json:"rollback_plan"`
	CanaryAnalysis    *CanaryAnalysisResult `json:"canary_analysis,omitempty"`
	FeatureFlags      map[string]bool        `json:"feature_flags"`
	
	// Context
	ctx               context.Context        `json:"-"`
	cancel            context.CancelFunc     `json:"-"`
	mu                sync.RWMutex           `json:"-"`
}

// DeploymentStatus represents the current status of a deployment
type DeploymentStatus string

const (
	StatusPending      DeploymentStatus = "pending"
	StatusProvisioning DeploymentStatus = "provisioning"
	StatusDeploying    DeploymentStatus = "deploying"
	StatusVerifying    DeploymentStatus = "verifying"
	StatusCanary       DeploymentStatus = "canary"
	StatusRollingOut   DeploymentStatus = "rolling_out"
	StatusComplete     DeploymentStatus = "complete"
	StatusFailed       DeploymentStatus = "failed"
	StatusRollingBack  DeploymentStatus = "rolling_back"
	StatusRolledBack   DeploymentStatus = "rolled_back"
)

// DeploymentStage represents a stage in the deployment process
type DeploymentStage struct {
	Name          string            `json:"name"`
	Status        string            `json:"status"`
	StartTime     time.Time         `json:"start_time"`
	EndTime       time.Time         `json:"end_time"`
	Duration      time.Duration     `json:"duration"`
	Metrics       map[string]interface{} `json:"metrics"`
	Logs          []string          `json:"logs"`
	Checkpoints   []Checkpoint      `json:"checkpoints"`
}

// Checkpoint represents a verification point during deployment
type Checkpoint struct {
	Name      string    `json:"name"`
	Status    string    `json:"status"`
	Timestamp time.Time `json:"timestamp"`
	Result    interface{} `json:"result"`
	Error     string    `json:"error,omitempty"`
}

// NewZeroDowntimeDeployer creates a new zero-downtime deployment manager
func NewZeroDowntimeDeployer(config *DeploymentConfig) (*ZeroDowntimeDeployer, error) {
	if config == nil {
		return nil, fmt.Errorf("deployment config cannot be nil")
	}

	deployer := &ZeroDowntimeDeployer{
		config:            config,
		activeDeployments: make(map[string]*DeploymentSession),
	}

	// Initialize components
	var err error
	
	deployer.blueGreenManager, err = NewBlueGreenManager(config.BlueGreenConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize blue-green manager: %w", err)
	}

	deployer.progressiveDelivery, err = NewProgressiveDelivery(config.ProgressiveDeliveryConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize progressive delivery: %w", err)
	}

	deployer.verificationService, err = NewDeploymentVerification(config.VerificationConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize verification service: %w", err)
	}

	deployer.trafficManager, err = NewTrafficManager(config.TrafficConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize traffic manager: %w", err)
	}

	deployer.rollbackManager, err = NewRollbackManager(config.RollbackConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize rollback manager: %w", err)
	}

	deployer.gitOpsController, err = NewGitOpsController(config.GitOpsConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize GitOps controller: %w", err)
	}

	deployer.metricsCollector, err = NewDeploymentMetrics(config.MetricsConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize metrics collector: %w", err)
	}

	// Initialize Prometheus metrics
	deployer.initializeMetrics()

	return deployer, nil
}

// initializeMetrics sets up Prometheus metrics
func (zd *ZeroDowntimeDeployer) initializeMetrics() {
	zd.deploymentGauge = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "novacron_active_deployments",
		Help: "Number of active deployments",
	})

	zd.errorRate = promauto.NewCounter(prometheus.CounterOpts{
		Name: "novacron_deployment_errors_total",
		Help: "Total number of deployment errors",
	})

	zd.deploymentDuration = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "novacron_deployment_duration_seconds",
		Help:    "Duration of deployments in seconds",
		Buckets: prometheus.ExponentialBuckets(1, 2, 10),
	})
}

// Deploy initiates a zero-downtime deployment
func (zd *ZeroDowntimeDeployer) Deploy(ctx context.Context, req *DeploymentRequest) (*DeploymentSession, error) {
	zd.mu.Lock()
	defer zd.mu.Unlock()

	// Validate deployment request
	if err := zd.validateDeploymentRequest(req); err != nil {
		return nil, fmt.Errorf("invalid deployment request: %w", err)
	}

	// Create deployment session
	session := &DeploymentSession{
		ID:          generateDeploymentID(),
		Version:     req.Version,
		Environment: req.Environment,
		Status:      StatusPending,
		StartTime:   time.Now(),
		Stages:      make([]DeploymentStage, 0),
		Metrics:     NewSessionMetrics(),
		FeatureFlags: make(map[string]bool),
	}

	// Create context with timeout
	session.ctx, session.cancel = context.WithTimeout(ctx, zd.config.MaxDeploymentTime)

	// Generate rollback plan
	rollbackPlan, err := zd.rollbackManager.GenerateRollbackPlan(req)
	if err != nil {
		return nil, fmt.Errorf("failed to generate rollback plan: %w", err)
	}
	session.RollbackPlan = rollbackPlan

	// Store active deployment
	zd.activeDeployments[session.ID] = session

	// Start deployment goroutine
	go zd.executeDeployment(session, req)

	// Update metrics
	zd.deploymentGauge.Inc()

	return session, nil
}

// executeDeployment runs the complete deployment process
func (zd *ZeroDowntimeDeployer) executeDeployment(session *DeploymentSession, req *DeploymentRequest) {
	defer func() {
		if r := recover(); r != nil {
			log.Printf("Deployment panic recovered: %v", r)
			zd.handleDeploymentFailure(session, fmt.Errorf("deployment panic: %v", r))
		}
		
		// Cleanup
		zd.mu.Lock()
		delete(zd.activeDeployments, session.ID)
		zd.mu.Unlock()
		
		zd.deploymentGauge.Dec()
		zd.deploymentDuration.Observe(time.Since(session.StartTime).Seconds())
	}()

	// Execute deployment pipeline
	stages := []struct {
		name string
		fn   func(*DeploymentSession, *DeploymentRequest) error
	}{
		{"Environment Provisioning", zd.provisionEnvironment},
		{"Database Migration", zd.runDatabaseMigration},
		{"Application Deployment", zd.deployApplication},
		{"Verification", zd.verifyDeployment},
		{"Canary Analysis", zd.runCanaryAnalysis},
		{"Progressive Rollout", zd.executeProgressiveRollout},
		{"Traffic Switch", zd.switchTraffic},
		{"Final Verification", zd.finalVerification},
		{"Cleanup", zd.cleanupDeployment},
	}

	for i, stage := range stages {
		session.mu.Lock()
		session.CurrentStage = i
		session.Status = DeploymentStatus(fmt.Sprintf("executing_%s", stage.name))
		session.mu.Unlock()

		stageStart := time.Now()
		stageInfo := DeploymentStage{
			Name:      stage.name,
			Status:    "running",
			StartTime: stageStart,
			Metrics:   make(map[string]interface{}),
			Logs:      make([]string, 0),
			Checkpoints: make([]Checkpoint, 0),
		}

		// Execute stage
		if err := stage.fn(session, req); err != nil {
			stageInfo.Status = "failed"
			stageInfo.EndTime = time.Now()
			stageInfo.Duration = time.Since(stageStart)
			
			session.mu.Lock()
			session.Stages = append(session.Stages, stageInfo)
			session.mu.Unlock()
			
			zd.handleDeploymentFailure(session, err)
			return
		}

		// Stage completed successfully
		stageInfo.Status = "completed"
		stageInfo.EndTime = time.Now()
		stageInfo.Duration = time.Since(stageStart)
		
		session.mu.Lock()
		session.Stages = append(session.Stages, stageInfo)
		session.mu.Unlock()

		// Check for cancellation
		select {
		case <-session.ctx.Done():
			zd.handleDeploymentFailure(session, session.ctx.Err())
			return
		default:
		}
	}

	// Deployment completed successfully
	session.mu.Lock()
	session.Status = StatusComplete
	session.mu.Unlock()

	log.Printf("Deployment %s completed successfully", session.ID)
}

// provisionEnvironment sets up the target environment
func (zd *ZeroDowntimeDeployer) provisionEnvironment(session *DeploymentSession, req *DeploymentRequest) error {
	log.Printf("Provisioning environment for deployment %s", session.ID)

	// Determine target environment (blue or green)
	targetEnv, err := zd.blueGreenManager.DetermineTargetEnvironment(req.Environment)
	if err != nil {
		return fmt.Errorf("failed to determine target environment: %w", err)
	}

	// Provision infrastructure
	provisionReq := &ProvisionRequest{
		Environment: targetEnv,
		Version:     req.Version,
		Resources:   req.Resources,
		Config:      req.Config,
	}

	if err := zd.blueGreenManager.ProvisionEnvironment(session.ctx, provisionReq); err != nil {
		return fmt.Errorf("failed to provision environment: %w", err)
	}

	// Store target environment in session
	session.mu.Lock()
	session.Metrics.TargetEnvironment = targetEnv
	session.mu.Unlock()

	return nil
}

// runDatabaseMigration executes database migrations without downtime
func (zd *ZeroDowntimeDeployer) runDatabaseMigration(session *DeploymentSession, req *DeploymentRequest) error {
	log.Printf("Running database migration for deployment %s", session.ID)

	if req.DatabaseMigration == nil {
		log.Printf("No database migration required for deployment %s", session.ID)
		return nil
	}

	migrationCtx, cancel := context.WithTimeout(session.ctx, zd.config.DatabaseMigrationTimeout)
	defer cancel()

	// Execute backward-compatible migration
	migrationResult, err := zd.executeDatabaseMigration(migrationCtx, req.DatabaseMigration)
	if err != nil {
		return fmt.Errorf("database migration failed: %w", err)
	}

	// Store migration result
	session.mu.Lock()
	session.Metrics.MigrationResult = migrationResult
	session.mu.Unlock()

	return nil
}

// deployApplication deploys the new application version
func (zd *ZeroDowntimeDeployer) deployApplication(session *DeploymentSession, req *DeploymentRequest) error {
	log.Printf("Deploying application for deployment %s", session.ID)

	deployReq := &ApplicationDeployRequest{
		Version:     req.Version,
		Environment: session.Metrics.TargetEnvironment,
		Config:      req.Config,
		Artifacts:   req.Artifacts,
	}

	if err := zd.blueGreenManager.DeployApplication(session.ctx, deployReq); err != nil {
		return fmt.Errorf("application deployment failed: %w", err)
	}

	return nil
}

// verifyDeployment runs verification checks
func (zd *ZeroDowntimeDeployer) verifyDeployment(session *DeploymentSession, req *DeploymentRequest) error {
	log.Printf("Verifying deployment %s", session.ID)

	verificationReq := &VerificationRequest{
		Environment: session.Metrics.TargetEnvironment,
		Version:     req.Version,
		TestSuites:  req.VerificationTests,
	}

	result, err := zd.verificationService.RunVerification(session.ctx, verificationReq)
	if err != nil {
		return fmt.Errorf("deployment verification failed: %w", err)
	}

	// Store verification result
	session.mu.Lock()
	session.Metrics.VerificationResult = result
	session.mu.Unlock()

	return nil
}

// runCanaryAnalysis performs canary analysis
func (zd *ZeroDowntimeDeployer) runCanaryAnalysis(session *DeploymentSession, req *DeploymentRequest) error {
	log.Printf("Running canary analysis for deployment %s", session.ID)

	if !req.CanaryConfig.Enabled {
		log.Printf("Canary analysis disabled for deployment %s", session.ID)
		return nil
	}

	// Start canary traffic
	canaryReq := &CanaryRequest{
		TargetEnvironment: session.Metrics.TargetEnvironment,
		TrafficPercentage: req.CanaryConfig.InitialTrafficPercentage,
		Duration:         req.CanaryConfig.Duration,
		Metrics:          req.CanaryConfig.Metrics,
	}

	analysis, err := zd.progressiveDelivery.StartCanaryAnalysis(session.ctx, canaryReq)
	if err != nil {
		return fmt.Errorf("canary analysis failed: %w", err)
	}

	// Check if canary passed
	if !analysis.Passed {
		return fmt.Errorf("canary analysis failed: %s", analysis.FailureReason)
	}

	// Store canary analysis result
	session.mu.Lock()
	session.CanaryAnalysis = analysis
	session.mu.Unlock()

	return nil
}

// executeProgressiveRollout performs gradual traffic shifting
func (zd *ZeroDowntimeDeployer) executeProgressiveRollout(session *DeploymentSession, req *DeploymentRequest) error {
	log.Printf("Executing progressive rollout for deployment %s", session.ID)

	rolloutReq := &ProgressiveRolloutRequest{
		SourceEnvironment: zd.blueGreenManager.GetCurrentEnvironment(req.Environment),
		TargetEnvironment: session.Metrics.TargetEnvironment,
		Strategy:         req.RolloutStrategy,
		Steps:           req.RolloutSteps,
	}

	if err := zd.progressiveDelivery.ExecuteProgressiveRollout(session.ctx, rolloutReq); err != nil {
		return fmt.Errorf("progressive rollout failed: %w", err)
	}

	return nil
}

// switchTraffic performs the final traffic switch
func (zd *ZeroDowntimeDeployer) switchTraffic(session *DeploymentSession, req *DeploymentRequest) error {
	log.Printf("Switching traffic for deployment %s", session.ID)

	switchReq := &TrafficSwitchRequest{
		FromEnvironment: zd.blueGreenManager.GetCurrentEnvironment(req.Environment),
		ToEnvironment:   session.Metrics.TargetEnvironment,
		Strategy:       "instant", // Can be "gradual" for slower switch
		PreserveSession: true,
		DrainTimeout:   zd.config.SessionPreservationTime,
	}

	if err := zd.trafficManager.SwitchTraffic(session.ctx, switchReq); err != nil {
		return fmt.Errorf("traffic switch failed: %w", err)
	}

	// Update current environment
	zd.blueGreenManager.SetCurrentEnvironment(req.Environment, session.Metrics.TargetEnvironment)

	return nil
}

// finalVerification runs post-switch verification
func (zd *ZeroDowntimeDeployer) finalVerification(session *DeploymentSession, req *DeploymentRequest) error {
	log.Printf("Running final verification for deployment %s", session.ID)

	verificationReq := &VerificationRequest{
		Environment: session.Metrics.TargetEnvironment,
		Version:     req.Version,
		TestSuites:  req.PostDeploymentTests,
	}

	result, err := zd.verificationService.RunVerification(session.ctx, verificationReq)
	if err != nil {
		return fmt.Errorf("final verification failed: %w", err)
	}

	// Store final verification result
	session.mu.Lock()
	session.Metrics.FinalVerificationResult = result
	session.mu.Unlock()

	return nil
}

// cleanupDeployment cleans up resources
func (zd *ZeroDowntimeDeployer) cleanupDeployment(session *DeploymentSession, req *DeploymentRequest) error {
	log.Printf("Cleaning up deployment %s", session.ID)

	// Clean up old environment
	oldEnvironment := zd.blueGreenManager.GetOldEnvironment(req.Environment)
	if err := zd.blueGreenManager.CleanupEnvironment(session.ctx, oldEnvironment); err != nil {
		log.Printf("Warning: failed to cleanup old environment %s: %v", oldEnvironment, err)
	}

	// Update GitOps state
	if err := zd.gitOpsController.UpdateDeploymentState(session.ID, req.Version, "deployed"); err != nil {
		log.Printf("Warning: failed to update GitOps state: %v", err)
	}

	return nil
}

// handleDeploymentFailure handles deployment failures and triggers rollback
func (zd *ZeroDowntimeDeployer) handleDeploymentFailure(session *DeploymentSession, err error) {
	log.Printf("Deployment %s failed: %v", session.ID, err)

	session.mu.Lock()
	session.Status = StatusFailed
	session.mu.Unlock()

	zd.errorRate.Inc()

	// Trigger automatic rollback
	if err := zd.initiateRollback(session); err != nil {
		log.Printf("Rollback failed for deployment %s: %v", session.ID, err)
	}
}

// initiateRollback starts the rollback process
func (zd *ZeroDowntimeDeployer) initiateRollback(session *DeploymentSession) error {
	session.mu.Lock()
	session.Status = StatusRollingBack
	session.mu.Unlock()

	rollbackReq := &RollbackRequest{
		SessionID:    session.ID,
		RollbackPlan: session.RollbackPlan,
		Timeout:      10 * time.Second, // Sub-10 second rollback
	}

	if err := zd.rollbackManager.ExecuteRollback(session.ctx, rollbackReq); err != nil {
		return fmt.Errorf("rollback execution failed: %w", err)
	}

	session.mu.Lock()
	session.Status = StatusRolledBack
	session.mu.Unlock()

	return nil
}

// GetDeploymentStatus returns the current status of a deployment
func (zd *ZeroDowntimeDeployer) GetDeploymentStatus(deploymentID string) (*DeploymentSession, error) {
	zd.mu.RLock()
	defer zd.mu.RUnlock()

	session, exists := zd.activeDeployments[deploymentID]
	if !exists {
		return nil, fmt.Errorf("deployment %s not found", deploymentID)
	}

	return session, nil
}

// ListActiveDeployments returns all active deployments
func (zd *ZeroDowntimeDeployer) ListActiveDeployments() []*DeploymentSession {
	zd.mu.RLock()
	defer zd.mu.RUnlock()

	sessions := make([]*DeploymentSession, 0, len(zd.activeDeployments))
	for _, session := range zd.activeDeployments {
		sessions = append(sessions, session)
	}

	return sessions
}

// CancelDeployment cancels an active deployment
func (zd *ZeroDowntimeDeployer) CancelDeployment(deploymentID string) error {
	zd.mu.RLock()
	session, exists := zd.activeDeployments[deploymentID]
	zd.mu.RUnlock()

	if !exists {
		return fmt.Errorf("deployment %s not found", deploymentID)
	}

	session.cancel()
	return nil
}

// validateDeploymentRequest validates the deployment request
func (zd *ZeroDowntimeDeployer) validateDeploymentRequest(req *DeploymentRequest) error {
	if req.Version == "" {
		return fmt.Errorf("version cannot be empty")
	}

	if req.Environment == "" {
		return fmt.Errorf("environment cannot be empty")
	}

	if req.Artifacts == nil || len(req.Artifacts) == 0 {
		return fmt.Errorf("artifacts cannot be empty")
	}

	return nil
}

// executeDatabaseMigration executes database migration
func (zd *ZeroDowntimeDeployer) executeDatabaseMigration(ctx context.Context, migration *DatabaseMigration) (*MigrationResult, error) {
	// Implementation would depend on database type and migration framework
	// This is a placeholder for the actual migration logic
	
	result := &MigrationResult{
		Success:        true,
		MigrationsRun:  len(migration.Scripts),
		ExecutionTime:  time.Since(time.Now()),
		RollbackInfo:   migration.RollbackScripts,
	}

	return result, nil
}

// generateDeploymentID generates a unique deployment ID
func generateDeploymentID() string {
	return fmt.Sprintf("deploy-%d", time.Now().UnixNano())
}

// HTTP API endpoints for deployment management

// SetupDeploymentAPI sets up HTTP endpoints for deployment management
func (zd *ZeroDowntimeDeployer) SetupDeploymentAPI(router *mux.Router) {
	router.HandleFunc("/deploy", zd.handleDeploy).Methods("POST")
	router.HandleFunc("/deployments", zd.handleListDeployments).Methods("GET")
	router.HandleFunc("/deployments/{id}", zd.handleGetDeployment).Methods("GET")
	router.HandleFunc("/deployments/{id}/cancel", zd.handleCancelDeployment).Methods("POST")
	router.HandleFunc("/deployments/{id}/rollback", zd.handleRollback).Methods("POST")
	router.HandleFunc("/deployments/{id}/status", zd.handleDeploymentStatus).Methods("GET")
}

func (zd *ZeroDowntimeDeployer) handleDeploy(w http.ResponseWriter, r *http.Request) {
	var req DeploymentRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request body: %v", err), http.StatusBadRequest)
		return
	}

	session, err := zd.Deploy(r.Context(), &req)
	if err != nil {
		http.Error(w, fmt.Sprintf("Deployment failed: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(session)
}

func (zd *ZeroDowntimeDeployer) handleListDeployments(w http.ResponseWriter, r *http.Request) {
	deployments := zd.ListActiveDeployments()
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(deployments)
}

func (zd *ZeroDowntimeDeployer) handleGetDeployment(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	deploymentID := vars["id"]

	session, err := zd.GetDeploymentStatus(deploymentID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(session)
}

func (zd *ZeroDowntimeDeployer) handleCancelDeployment(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	deploymentID := vars["id"]

	if err := zd.CancelDeployment(deploymentID); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	w.WriteHeader(http.StatusOK)
	w.Write([]byte(`{"status": "cancelled"}`))
}

func (zd *ZeroDowntimeDeployer) handleRollback(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	deploymentID := vars["id"]

	session, err := zd.GetDeploymentStatus(deploymentID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}

	if err := zd.initiateRollback(session); err != nil {
		http.Error(w, fmt.Sprintf("Rollback failed: %v", err), http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusOK)
	w.Write([]byte(`{"status": "rollback_initiated"}`))
}

func (zd *ZeroDowntimeDeployer) handleDeploymentStatus(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	deploymentID := vars["id"]

	session, err := zd.GetDeploymentStatus(deploymentID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"id":            session.ID,
		"status":        session.Status,
		"current_stage": session.CurrentStage,
		"stages":        session.Stages,
		"metrics":       session.Metrics,
	})
}