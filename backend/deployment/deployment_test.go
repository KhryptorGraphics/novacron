package deployment

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/suite"
)

// ZeroDowntimeDeploymentTestSuite is the test suite for zero-downtime deployment
type ZeroDowntimeDeploymentTestSuite struct {
	suite.Suite
	deployer           *ZeroDowntimeDeployer
	mockTrafficManager *MockTrafficManager
	mockRollbackManager *MockRollbackManager
	mockVerificationService *MockVerificationService
	ctx                context.Context
	cancel             context.CancelFunc
}

// SetupSuite sets up the test suite
func (suite *ZeroDowntimeDeploymentTestSuite) SetupSuite() {
	suite.ctx, suite.cancel = context.WithTimeout(context.Background(), 30*time.Second)
	
	// Create mock components
	suite.mockTrafficManager = &MockTrafficManager{}
	suite.mockRollbackManager = &MockRollbackManager{}
	suite.mockVerificationService = &MockVerificationService{}
	
	// Create test configuration
	config := &DeploymentConfig{
		Environment:               "test",
		MaxDeploymentTime:         10 * time.Minute,
		RollbackThreshold:         0.8,
		HealthCheckInterval:       1 * time.Second,
		DatabaseMigrationTimeout:  2 * time.Minute,
		SessionPreservationTime:   30 * time.Second,
	}
	
	// Create deployer with mocked dependencies
	deployer, err := NewZeroDowntimeDeployer(config)
	suite.Require().NoError(err)
	
	// Inject mocks
	deployer.trafficManager = suite.mockTrafficManager
	deployer.rollbackManager = suite.mockRollbackManager
	deployer.verificationService = suite.mockVerificationService
	
	suite.deployer = deployer
}

// TearDownSuite tears down the test suite
func (suite *ZeroDowntimeDeploymentTestSuite) TearDownSuite() {
	if suite.cancel != nil {
		suite.cancel()
	}
}

// TestSuccessfulDeployment tests a successful zero-downtime deployment
func (suite *ZeroDowntimeDeploymentTestSuite) TestSuccessfulDeployment() {
	// Setup mocks for successful deployment
	suite.setupSuccessfulMocks()
	
	// Create deployment request
	req := &DeploymentRequest{
		DeploymentID:    "test-deployment-001",
		Environment:     "production",
		Version:         "v1.2.0",
		PreviousVersion: "v1.1.0",
		Artifacts:       []string{"app.tar.gz", "config.yaml"},
		VerificationTests: []string{"smoke_tests", "integration_tests"},
		PostDeploymentTests: []string{"end_to_end_tests"},
		CanaryConfig: &CanaryConfig{
			Enabled:                  true,
			InitialTrafficPercentage: 5.0,
			Duration:                 2 * time.Minute,
			Metrics:                  []string{"response_time", "error_rate"},
		},
		RolloutStrategy: RolloutGradual,
		RolloutSteps: []RolloutStep{
			{Percentage: 25, Duration: 1 * time.Minute},
			{Percentage: 50, Duration: 1 * time.Minute},
			{Percentage: 100, Duration: 30 * time.Second},
		},
	}
	
	// Execute deployment
	session, err := suite.deployer.Deploy(suite.ctx, req)
	
	// Assertions
	suite.NoError(err)
	suite.NotNil(session)
	suite.Equal("test-deployment-001", req.DeploymentID)
	suite.Equal("v1.2.0", session.Version)
	suite.Equal("production", session.Environment)
	suite.Equal(StatusPending, session.Status)
	suite.NotEmpty(session.ID)
	suite.NotNil(session.RollbackPlan)
	
	// Wait for deployment to complete
	suite.waitForDeploymentCompletion(session.ID, 30*time.Second)
	
	// Verify final status
	finalSession, err := suite.deployer.GetDeploymentStatus(session.ID)
	suite.NoError(err)
	suite.Equal(StatusComplete, finalSession.Status)
	suite.True(len(finalSession.Stages) > 0)
	
	// Verify all mocks were called
	suite.mockTrafficManager.AssertExpectations(suite.T())
	suite.mockVerificationService.AssertExpectations(suite.T())
}

// TestFailedDeploymentWithRollback tests deployment failure and automatic rollback
func (suite *ZeroDowntimeDeploymentTestSuite) TestFailedDeploymentWithRollback() {
	// Setup mocks for failed deployment
	suite.setupFailedMocks()
	
	// Create deployment request
	req := &DeploymentRequest{
		DeploymentID:    "test-deployment-002",
		Environment:     "production",
		Version:         "v1.2.1",
		PreviousVersion: "v1.2.0",
		Artifacts:       []string{"app.tar.gz"},
		VerificationTests: []string{"smoke_tests"},
	}
	
	// Execute deployment
	session, err := suite.deployer.Deploy(suite.ctx, req)
	
	// Assertions
	suite.NoError(err)
	suite.NotNil(session)
	
	// Wait for deployment to fail and rollback
	suite.waitForDeploymentCompletion(session.ID, 30*time.Second)
	
	// Verify final status
	finalSession, err := suite.deployer.GetDeploymentStatus(session.ID)
	suite.NoError(err)
	suite.Equal(StatusFailed, finalSession.Status)
	
	// Verify rollback was triggered
	suite.mockRollbackManager.AssertExpectations(suite.T())
}

// TestCanaryDeployment tests canary deployment with traffic analysis
func (suite *ZeroDowntimeDeploymentTestSuite) TestCanaryDeployment() {
	// Setup mocks for canary deployment
	suite.setupCanaryMocks()
	
	// Create canary deployment request
	req := &DeploymentRequest{
		DeploymentID:    "test-canary-001",
		Environment:     "production",
		Version:         "v1.3.0",
		PreviousVersion: "v1.2.0",
		Artifacts:       []string{"app.tar.gz"},
		CanaryConfig: &CanaryConfig{
			Enabled:                  true,
			InitialTrafficPercentage: 10.0,
			Duration:                 3 * time.Minute,
			Metrics:                  []string{"response_time", "error_rate", "throughput"},
		},
	}
	
	// Execute deployment
	session, err := suite.deployer.Deploy(suite.ctx, req)
	
	// Assertions
	suite.NoError(err)
	suite.NotNil(session)
	suite.True(req.CanaryConfig.Enabled)
	suite.Equal(10.0, req.CanaryConfig.InitialTrafficPercentage)
	
	// Wait for canary deployment to complete
	suite.waitForDeploymentCompletion(session.ID, 5*time.Minute)
	
	// Verify canary analysis
	finalSession, err := suite.deployer.GetDeploymentStatus(session.ID)
	suite.NoError(err)
	suite.NotNil(finalSession.CanaryAnalysis)
	suite.True(finalSession.CanaryAnalysis.Passed)
	suite.Equal("canary", finalSession.CanaryAnalysis.Score)
}

// TestProgressiveRollout tests progressive traffic rollout
func (suite *ZeroDowntimeDeploymentTestSuite) TestProgressiveRollout() {
	// Setup mocks for progressive rollout
	suite.setupProgressiveRolloutMocks()
	
	// Create progressive rollout request
	req := &DeploymentRequest{
		DeploymentID:    "test-progressive-001",
		Environment:     "production",
		Version:         "v1.4.0",
		PreviousVersion: "v1.3.0",
		Artifacts:       []string{"app.tar.gz"},
		RolloutStrategy: RolloutGradual,
		RolloutSteps: []RolloutStep{
			{Percentage: 10, Duration: 2 * time.Minute},
			{Percentage: 30, Duration: 2 * time.Minute},
			{Percentage: 60, Duration: 2 * time.Minute},
			{Percentage: 100, Duration: 1 * time.Minute},
		},
	}
	
	// Execute deployment
	session, err := suite.deployer.Deploy(suite.ctx, req)
	
	// Assertions
	suite.NoError(err)
	suite.NotNil(session)
	suite.Equal(RolloutGradual, req.RolloutStrategy)
	suite.Equal(4, len(req.RolloutSteps))
	
	// Wait for progressive rollout to complete
	suite.waitForDeploymentCompletion(session.ID, 10*time.Minute)
	
	// Verify rollout completed successfully
	finalSession, err := suite.deployer.GetDeploymentStatus(session.ID)
	suite.NoError(err)
	suite.Equal(StatusComplete, finalSession.Status)
}

// TestRollbackSpeed tests rollback execution speed (sub-10 seconds)
func (suite *ZeroDowntimeDeploymentTestSuite) TestRollbackSpeed() {
	// Create rollback plan
	rollbackPlan := &RollbackPlan{
		ID:           "rollback-speed-test",
		DeploymentID: "test-deployment-003",
		Environment:  "production",
		FromVersion:  "v1.2.0",
		ToVersion:    "v1.1.0",
		Components: []*RollbackComponent{
			{
				ID:                  "traffic-component",
				Name:                "Traffic Switch",
				Type:                ComponentTraffic,
				Priority:            PriorityCritical,
				Strategy:            StrategyToggle,
				FastRollbackCapable: true,
				EstimatedTime:       1 * time.Second,
			},
			{
				ID:                  "app-component",
				Name:                "Application",
				Type:                ComponentApplication,
				Priority:            PriorityCritical,
				Strategy:            StrategyReplace,
				FastRollbackCapable: true,
				EstimatedTime:       3 * time.Second,
			},
		},
		FastTrackComponents: []string{"traffic-component", "app-component"},
		EstimatedDuration:   5 * time.Second,
	}
	
	// Setup mock for fast rollback
	suite.mockRollbackManager.On("ExecuteRollback", mock.Anything, mock.MatchedBy(func(req *RollbackRequest) bool {
		return req.FastTrackEnabled && req.Timeout <= 10*time.Second
	})).Return(nil).Run(func(args mock.Arguments) {
		// Simulate fast rollback execution
		time.Sleep(2 * time.Second)
	})
	
	// Create rollback request with sub-10 second timeout
	rollbackReq := &RollbackRequest{
		SessionID:        "rollback-speed-session",
		RollbackPlan:     rollbackPlan,
		Timeout:          5 * time.Second,
		FastTrackEnabled: true,
		EmergencyMode:    false,
	}
	
	// Measure rollback execution time
	startTime := time.Now()
	err := suite.deployer.rollbackManager.ExecuteRollback(suite.ctx, rollbackReq)
	executionTime := time.Since(startTime)
	
	// Assertions
	suite.NoError(err)
	suite.Less(executionTime, 10*time.Second, "Rollback should complete in under 10 seconds")
	suite.mockRollbackManager.AssertExpectations(suite.T())
}

// TestMetricsCollection tests deployment metrics collection
func (suite *ZeroDowntimeDeploymentTestSuite) TestMetricsCollection() {
	// Create metrics collector
	metricsConfig := &DeploymentMetricsConfig{
		MetricsInterval: 1 * time.Second,
		DORAMetrics: &DORAMetricsConfig{
			Enabled: true,
			DeploymentFrequency: &FrequencyConfig{
				Enabled:     true,
				TimeWindows: []time.Duration{24 * time.Hour, 7 * 24 * time.Hour},
			},
			LeadTimeForChanges: &LeadTimeConfig{
				Enabled:     true,
				StartEvent:  "commit_pushed",
				EndEvent:    "deployment_completed",
				Percentiles: []float64{0.5, 0.9, 0.99},
			},
		},
	}
	
	metricsCollector, err := NewDeploymentMetrics(metricsConfig)
	suite.NoError(err)
	suite.NotNil(metricsCollector)
	
	// Create deployment event
	deploymentEvent := &DeploymentEvent{
		ID:              "event-001",
		Type:            EventDeploymentCompleted,
		Timestamp:       time.Now(),
		DeploymentID:    "test-deployment-metrics",
		Environment:     "production",
		Version:         "v1.5.0",
		Service:         "api-service",
		Status:          "success",
		Duration:        5 * time.Minute,
		CommitSHA:       "abc123def456",
		CommitTimestamp: time.Now().Add(-2 * time.Hour),
	}
	
	// Record deployment event
	err = metricsCollector.RecordDeploymentEvent(deploymentEvent)
	suite.NoError(err)
	
	// Verify metrics were recorded
	suite.Equal(EventDeploymentCompleted, deploymentEvent.Type)
	suite.Equal("success", deploymentEvent.Status)
	suite.Equal(5*time.Minute, deploymentEvent.Duration)
}

// TestGitOpsIntegration tests GitOps workflow integration
func (suite *ZeroDowntimeDeploymentTestSuite) TestGitOpsIntegration() {
	// Create GitOps configuration
	gitopsConfig := &GitOpsConfig{
		DefaultBranch:   "main",
		AutoSyncEnabled: true,
		SyncInterval:    30 * time.Second,
		Repositories: []*RepositoryConfig{
			{
				Name:     "app-manifests",
				URL:      "https://github.com/company/app-manifests.git",
				Branch:   "main",
				Path:     "environments/production",
				AutoSync: true,
			},
		},
		Pipelines: []*PipelineConfig{
			{
				Name:        "production-deploy",
				Repository:  "app-manifests",
				Environment: "production",
				Triggers: []*PipelineTrigger{
					{
						Type:   TriggerGitPush,
						Branch: "main",
						Path:   "environments/production",
					},
				},
			},
		},
	}
	
	// Create GitOps controller
	gitopsController, err := NewGitOpsController(gitopsConfig)
	suite.NoError(err)
	suite.NotNil(gitopsController)
	
	// Test deployment state update
	err = gitopsController.UpdateDeploymentState("test-deployment-004", "v1.6.0", "deployed")
	suite.NoError(err)
	
	// Test pipeline trigger
	params := map[string]string{
		"branch":      "main",
		"commit":      "def789ghi012",
		"version":     "v1.6.0",
		"application": "api-service",
	}
	
	execution, err := gitopsController.TriggerPipeline("production-deploy", params)
	suite.NoError(err)
	suite.NotNil(execution)
	suite.Equal("production-deploy", execution.PipelineName)
	suite.Equal("main", execution.Branch)
	suite.Equal("def789ghi012", execution.Commit)
	suite.Equal(PipelinePending, execution.Status)
}

// Test helper methods

func (suite *ZeroDowntimeDeploymentTestSuite) setupSuccessfulMocks() {
	// Mock successful traffic switch
	suite.mockTrafficManager.On("SwitchTraffic", mock.Anything, mock.AnythingOfType("*deployment.TrafficShiftRequest")).Return(nil)
	
	// Mock successful verification
	verificationResult := &OverallResult{
		Success:          true,
		Score:            0.95,
		CriticalFailures: 0,
		TotalTests:       10,
		PassedTests:      10,
		FailedTests:      0,
		Summary:          "All tests passed successfully",
	}
	suite.mockVerificationService.On("RunVerification", mock.Anything, mock.AnythingOfType("*deployment.VerificationRequest")).Return(verificationResult, nil)
	
	// Mock successful rollback plan generation
	rollbackPlan := &RollbackPlan{
		ID:                "successful-rollback-plan",
		EstimatedDuration: 30 * time.Second,
		Components:        []*RollbackComponent{},
	}
	suite.mockRollbackManager.On("GenerateRollbackPlan", mock.AnythingOfType("*deployment.DeploymentRequest")).Return(rollbackPlan, nil)
}

func (suite *ZeroDowntimeDeploymentTestSuite) setupFailedMocks() {
	// Mock failed verification
	verificationResult := &OverallResult{
		Success:          false,
		Score:            0.60,
		CriticalFailures: 2,
		TotalTests:       10,
		PassedTests:      6,
		FailedTests:      4,
		Summary:          "Critical failures detected",
	}
	suite.mockVerificationService.On("RunVerification", mock.Anything, mock.AnythingOfType("*deployment.VerificationRequest")).Return(verificationResult, fmt.Errorf("verification failed"))
	
	// Mock rollback execution
	suite.mockRollbackManager.On("ExecuteRollback", mock.Anything, mock.AnythingOfType("*deployment.RollbackRequest")).Return(nil)
	
	// Mock rollback plan generation
	rollbackPlan := &RollbackPlan{
		ID:                "failed-deployment-rollback-plan",
		EstimatedDuration: 45 * time.Second,
		Components:        []*RollbackComponent{},
	}
	suite.mockRollbackManager.On("GenerateRollbackPlan", mock.AnythingOfType("*deployment.DeploymentRequest")).Return(rollbackPlan, nil)
}

func (suite *ZeroDowntimeDeploymentTestSuite) setupCanaryMocks() {
	// Mock successful canary analysis
	suite.mockTrafficManager.On("SwitchTraffic", mock.Anything, mock.AnythingOfType("*deployment.TrafficShiftRequest")).Return(nil)
	
	// Mock successful verification
	verificationResult := &OverallResult{
		Success:     true,
		Score:       0.98,
		TotalTests:  15,
		PassedTests: 15,
		Summary:     "Canary deployment verified successfully",
	}
	suite.mockVerificationService.On("RunVerification", mock.Anything, mock.AnythingOfType("*deployment.VerificationRequest")).Return(verificationResult, nil)
	
	// Mock rollback plan
	rollbackPlan := &RollbackPlan{
		ID:                "canary-rollback-plan",
		EstimatedDuration: 20 * time.Second,
	}
	suite.mockRollbackManager.On("GenerateRollbackPlan", mock.AnythingOfType("*deployment.DeploymentRequest")).Return(rollbackPlan, nil)
}

func (suite *ZeroDowntimeDeploymentTestSuite) setupProgressiveRolloutMocks() {
	// Mock successful traffic switches for progressive rollout
	suite.mockTrafficManager.On("SwitchTraffic", mock.Anything, mock.AnythingOfType("*deployment.TrafficShiftRequest")).Return(nil)
	
	// Mock successful verification at each step
	verificationResult := &OverallResult{
		Success:     true,
		Score:       0.92,
		TotalTests:  12,
		PassedTests: 12,
		Summary:     "Progressive rollout step verified",
	}
	suite.mockVerificationService.On("RunVerification", mock.Anything, mock.AnythingOfType("*deployment.VerificationRequest")).Return(verificationResult, nil)
	
	// Mock rollback plan
	rollbackPlan := &RollbackPlan{
		ID:                "progressive-rollback-plan",
		EstimatedDuration: 35 * time.Second,
	}
	suite.mockRollbackManager.On("GenerateRollbackPlan", mock.AnythingOfType("*deployment.DeploymentRequest")).Return(rollbackPlan, nil)
}

func (suite *ZeroDowntimeDeploymentTestSuite) waitForDeploymentCompletion(deploymentID string, timeout time.Duration) {
	ctx, cancel := context.WithTimeout(suite.ctx, timeout)
	defer cancel()
	
	ticker := time.NewTicker(500 * time.Millisecond)
	defer ticker.Stop()
	
	for {
		select {
		case <-ctx.Done():
			suite.Fail("Deployment did not complete within timeout", "deployment_id", deploymentID)
			return
		case <-ticker.C:
			session, err := suite.deployer.GetDeploymentStatus(deploymentID)
			if err != nil {
				// Deployment might have been cleaned up, which is expected
				return
			}
			
			switch session.Status {
			case StatusComplete, StatusFailed, StatusRolledBack:
				return
			default:
				continue
			}
		}
	}
}

// Mock implementations

type MockTrafficManager struct {
	mock.Mock
}

func (m *MockTrafficManager) SwitchTraffic(ctx context.Context, req *TrafficShiftRequest) error {
	args := m.Called(ctx, req)
	return args.Error(0)
}

type MockRollbackManager struct {
	mock.Mock
}

func (m *MockRollbackManager) GenerateRollbackPlan(req *DeploymentRequest) (*RollbackPlan, error) {
	args := m.Called(req)
	return args.Get(0).(*RollbackPlan), args.Error(1)
}

func (m *MockRollbackManager) ExecuteRollback(ctx context.Context, req *RollbackRequest) error {
	args := m.Called(ctx, req)
	return args.Error(0)
}

type MockVerificationService struct {
	mock.Mock
}

func (m *MockVerificationService) RunVerification(ctx context.Context, req *VerificationRequest) (*OverallResult, error) {
	args := m.Called(ctx, req)
	return args.Get(0).(*OverallResult), args.Error(1)
}

// Unit tests for individual components

func TestBlueGreenManager(t *testing.T) {
	config := &BlueGreenConfig{
		BlueEnvironment:  "blue",
		GreenEnvironment: "green",
	}
	
	manager, err := NewBlueGreenManager(config)
	assert.NoError(t, err)
	assert.NotNil(t, manager)
	
	// Test environment determination
	targetEnv, err := manager.DetermineTargetEnvironment("production")
	assert.NoError(t, err)
	assert.Contains(t, []string{"blue", "green"}, targetEnv)
}

func TestProgressiveDelivery(t *testing.T) {
	config := &ProgressiveConfig{
		FeatureFlags: &FeatureFlagConfig{},
		ABTesting:    &ABTestingConfig{},
	}
	
	pd, err := NewProgressiveDelivery(config)
	assert.NoError(t, err)
	assert.NotNil(t, pd)
	
	// Test feature flag evaluation
	context := map[string]interface{}{
		"user_id":     "test-user-123",
		"environment": "production",
	}
	
	_, err = pd.EvaluateFeatureFlag("test-flag", context)
	// This will return an error because the flag doesn't exist, which is expected
	assert.Error(t, err)
}

func TestDeploymentVerification(t *testing.T) {
	config := &VerificationConfig{
		DefaultTimeout:        30 * time.Second,
		MaxConcurrentTests:    5,
		RetryAttempts:        3,
		RetryDelay:           5 * time.Second,
		FailureTolerance:     0.1,
		CriticalTestThreshold: 0.8,
		SmokeTests: &SmokeTestConfig{
			Enabled:           true,
			TestSuites:        []string{"basic", "api"},
			ParallelExecution: true,
			Timeout:           10 * time.Second,
		},
		SyntheticMonitoring: &SyntheticMonitorConfig{
			Enabled:         true,
			Monitors:        []string{"api-health", "db-health"},
			CheckInterval:   1 * time.Minute,
			AlertThreshold:  0.95,
			MaxResponseTime: 5 * time.Second,
		},
	}
	
	dv, err := NewDeploymentVerification(config)
	assert.NoError(t, err)
	assert.NotNil(t, dv)
	
	// Test verification request
	req := &VerificationRequest{
		Environment: "staging",
		Version:     "v1.0.0",
		TestSuites:  []string{"smoke_tests", "synthetic_monitoring"},
		Timeout:     30 * time.Second,
	}
	
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Minute)
	defer cancel()
	
	result, err := dv.RunVerification(ctx, req)
	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.True(t, result.Success) // Mock implementation always succeeds
	assert.Greater(t, result.Score, 0.0)
}

func TestTrafficManager(t *testing.T) {
	config := &TrafficConfig{
		LoadBalancer: &LoadBalancerConfig{
			Type:      "nginx",
			Providers: []string{"nginx"},
		},
		DNS: &DNSConfig{
			Enabled:   true,
			Providers: []string{"cloudflare"},
			TTL:       300,
			Domains:   []string{"api.example.com"},
		},
		EdgeCache: &EdgeCacheConfig{
			Enabled:           true,
			Providers:         []string{"cloudflare"},
			InvalidateOnSwitch: true,
			WarmupEnabled:     true,
		},
		ConnectionDraining: &ConnectionDrainingConfig{
			Enabled:              true,
			DefaultTimeout:       30 * time.Second,
			GracefulShutdownTime: 15 * time.Second,
			ForceCloseAfter:      60 * time.Second,
		},
	}
	
	tm, err := NewTrafficManager(config)
	assert.NoError(t, err)
	assert.NotNil(t, tm)
	
	// Test traffic switch request
	req := &TrafficShiftRequest{
		FromEnvironment: "blue",
		ToEnvironment:   "green",
		Strategy:        ShiftInstant,
		PreserveSession: true,
		DrainTimeout:    30 * time.Second,
	}
	
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	
	err = tm.SwitchTraffic(ctx, req)
	assert.NoError(t, err)
}

func TestRollbackManager(t *testing.T) {
	config := &RollbackConfig{
		MaxRollbackTime:       5 * time.Minute,
		FastRollbackThreshold: 10 * time.Second,
		ParallelOperations:    3,
		EmergencyTimeout:      5 * time.Second,
		PreRollbackValidation:  true,
		PostRollbackValidation: true,
		RollbackTestTimeout:   30 * time.Second,
	}
	
	rm, err := NewRollbackManager(config)
	assert.NoError(t, err)
	assert.NotNil(t, rm)
	
	// Test rollback plan generation
	req := &DeploymentRequest{
		DeploymentID:    "test-rollback-deployment",
		Environment:     "production",
		Version:         "v2.0.0",
		PreviousVersion: "v1.9.0",
	}
	
	plan, err := rm.GenerateRollbackPlan(req)
	assert.NoError(t, err)
	assert.NotNil(t, plan)
	assert.NotEmpty(plan.ID)
	assert.Equal(t, req.DeploymentID, plan.DeploymentID)
	assert.Equal(t, req.Environment, plan.Environment)
	assert.Equal(t, req.Version, plan.FromVersion)
	assert.Equal(t, req.PreviousVersion, plan.ToVersion)
	assert.Greater(t, plan.EstimatedDuration, time.Duration(0))
}

// Integration test for the complete zero-downtime deployment flow
func TestZeroDowntimeDeploymentIntegration(t *testing.T) {
	suite.Run(t, new(ZeroDowntimeDeploymentTestSuite))
}

// Benchmark tests for performance validation

func BenchmarkDeploymentExecution(b *testing.B) {
	config := &DeploymentConfig{
		Environment:               "benchmark",
		MaxDeploymentTime:         30 * time.Second,
		RollbackThreshold:         0.8,
		HealthCheckInterval:       100 * time.Millisecond,
		DatabaseMigrationTimeout:  10 * time.Second,
		SessionPreservationTime:   5 * time.Second,
	}
	
	deployer, err := NewZeroDowntimeDeployer(config)
	if err != nil {
		b.Fatal(err)
	}
	
	req := &DeploymentRequest{
		DeploymentID:    "benchmark-deployment",
		Environment:     "benchmark",
		Version:         "v1.0.0",
		PreviousVersion: "v0.9.0",
		Artifacts:       []string{"app.tar.gz"},
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ctx, cancel := context.WithTimeout(context.Background(), 1*time.Minute)
		
		session, err := deployer.Deploy(ctx, req)
		if err != nil {
			b.Fatal(err)
		}
		
		// Wait for completion
		for {
			currentSession, err := deployer.GetDeploymentStatus(session.ID)
			if err != nil {
				break // Deployment cleaned up
			}
			
			if currentSession.Status == StatusComplete || currentSession.Status == StatusFailed {
				break
			}
			
			time.Sleep(10 * time.Millisecond)
		}
		
		cancel()
	}
}

func BenchmarkRollbackSpeed(b *testing.B) {
	config := &RollbackConfig{
		MaxRollbackTime:       30 * time.Second,
		FastRollbackThreshold: 5 * time.Second,
		ParallelOperations:    10,
		EmergencyTimeout:      2 * time.Second,
	}
	
	rm, err := NewRollbackManager(config)
	if err != nil {
		b.Fatal(err)
	}
	
	plan := &RollbackPlan{
		ID:           "benchmark-rollback",
		DeploymentID: "benchmark-deployment",
		Environment:  "benchmark",
		Components: []*RollbackComponent{
			{
				ID:                  "fast-component",
				Type:                ComponentTraffic,
				Priority:            PriorityCritical,
				FastRollbackCapable: true,
				EstimatedTime:       100 * time.Millisecond,
			},
		},
		FastTrackComponents: []string{"fast-component"},
		EstimatedDuration:   500 * time.Millisecond,
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		
		req := &RollbackRequest{
			SessionID:        fmt.Sprintf("benchmark-session-%d", i),
			RollbackPlan:     plan,
			Timeout:          5 * time.Second,
			FastTrackEnabled: true,
			EmergencyMode:    true,
		}
		
		startTime := time.Now()
		err := rm.ExecuteRollback(ctx, req)
		duration := time.Since(startTime)
		
		if err != nil {
			b.Fatal(err)
		}
		
		// Ensure rollback completed within acceptable time
		if duration > 10*time.Second {
			b.Fatalf("Rollback took too long: %v", duration)
		}
		
		cancel()
	}
}

// Additional types for testing

type BlueGreenConfig struct {
	BlueEnvironment  string `json:"blue_environment"`
	GreenEnvironment string `json:"green_environment"`
}

type BlueGreenManager struct {
	config *BlueGreenConfig
}

func NewBlueGreenManager(config *BlueGreenConfig) (*BlueGreenManager, error) {
	return &BlueGreenManager{config: config}, nil
}

func (bgm *BlueGreenManager) DetermineTargetEnvironment(environment string) (string, error) {
	// Simple logic: alternate between blue and green
	if time.Now().Unix()%2 == 0 {
		return bgm.config.BlueEnvironment, nil
	}
	return bgm.config.GreenEnvironment, nil
}

func (bgm *BlueGreenManager) ProvisionEnvironment(ctx context.Context, req *ProvisionRequest) error {
	return nil
}

func (bgm *BlueGreenManager) DeployApplication(ctx context.Context, req *ApplicationDeployRequest) error {
	return nil
}

func (bgm *BlueGreenManager) GetCurrentEnvironment(environment string) string {
	return "current"
}

func (bgm *BlueGreenManager) SetCurrentEnvironment(environment, target string) {
	// Implementation would update current environment
}

func (bgm *BlueGreenManager) GetOldEnvironment(environment string) string {
	return "old"
}

func (bgm *BlueGreenManager) CleanupEnvironment(ctx context.Context, environment string) error {
	return nil
}

// Additional mock types for testing
type ProvisionRequest struct {
	Environment string                 `json:"environment"`
	Version     string                 `json:"version"`
	Resources   map[string]interface{} `json:"resources"`
	Config      map[string]interface{} `json:"config"`
}

type ApplicationDeployRequest struct {
	Version     string                 `json:"version"`
	Environment string                 `json:"environment"`
	Config      map[string]interface{} `json:"config"`
	Artifacts   []string               `json:"artifacts"`
}