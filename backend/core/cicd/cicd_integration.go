package cicd

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// PipelineStatus represents the status of a CI/CD pipeline
type PipelineStatus string

const (
	// StatusPending represents a pipeline that has been created but not started
	StatusPending PipelineStatus = "pending"
	// StatusRunning represents a pipeline that is currently running
	StatusRunning PipelineStatus = "running"
	// StatusSucceeded represents a pipeline that has completed successfully
	StatusSucceeded PipelineStatus = "succeeded"
	// StatusFailed represents a pipeline that has failed
	StatusFailed PipelineStatus = "failed"
	// StatusCanceled represents a pipeline that was canceled
	StatusCanceled PipelineStatus = "canceled"
)

// ProviderType represents the type of CI/CD provider
type ProviderType string

const (
	// ProviderJenkins represents the Jenkins CI/CD provider
	ProviderJenkins ProviderType = "jenkins"
	// ProviderGitHubActions represents the GitHub Actions CI/CD provider
	ProviderGitHubActions ProviderType = "github_actions"
	// ProviderGitLabCI represents the GitLab CI/CD provider
	ProviderGitLabCI ProviderType = "gitlab_ci"
	// ProviderCustom represents a custom CI/CD provider
	ProviderCustom ProviderType = "custom"
)

// ArtifactType represents the type of artifact
type ArtifactType string

const (
	// ArtifactBinary represents a binary artifact
	ArtifactBinary ArtifactType = "binary"
	// ArtifactContainer represents a container image
	ArtifactContainer ArtifactType = "container"
	// ArtifactPackage represents a package (e.g., npm, pip)
	ArtifactPackage ArtifactType = "package"
	// ArtifactArchive represents an archive (e.g., zip, tar)
	ArtifactArchive ArtifactType = "archive"
)

// PipelineArtifact represents an artifact produced by a pipeline
type PipelineArtifact struct {
	// ID is a unique identifier for this artifact
	ID string
	// Name of the artifact
	Name string
	// Type of artifact
	Type ArtifactType
	// Location where the artifact is stored
	Location string
	// Checksum of the artifact
	Checksum string
	// Size of the artifact in bytes
	Size int64
	// Metadata about the artifact
	Metadata map[string]string
	// Creation time
	CreatedAt time.Time
}

// PipelineStage represents a stage in a CI/CD pipeline
type PipelineStage struct {
	// Name of the stage
	Name string
	// Status of the stage
	Status PipelineStatus
	// Steps in the stage
	Steps []*PipelineStep
	// Start time
	StartTime time.Time
	// End time
	EndTime time.Time
	// Duration in seconds
	DurationSeconds int64
}

// PipelineStep represents a step in a pipeline stage
type PipelineStep struct {
	// Name of the step
	Name string
	// Status of the step
	Status PipelineStatus
	// Command executed in this step
	Command string
	// Output from the step
	Output string
	// Exit code of the command
	ExitCode int
	// Start time
	StartTime time.Time
	// End time
	EndTime time.Time
	// Duration in seconds
	DurationSeconds int64
}

// Pipeline represents a CI/CD pipeline
type Pipeline struct {
	// ID is a unique identifier for this pipeline
	ID string
	// Name of the pipeline
	Name string
	// Provider that runs this pipeline
	Provider ProviderType
	// Repository URL
	RepositoryURL string
	// Branch or tag
	Reference string
	// Commit SHA
	CommitSHA string
	// User who triggered the pipeline
	Triggered string
	// Status of the pipeline
	Status PipelineStatus
	// Stages in the pipeline
	Stages []*PipelineStage
	// Artifacts produced by the pipeline
	Artifacts []*PipelineArtifact
	// Start time
	StartTime time.Time
	// End time
	EndTime time.Time
	// Duration in seconds
	DurationSeconds int64
	// Environment variables for the pipeline
	Environment map[string]string
	// Webhook URL for notifications
	WebhookURL string
}

// DeploymentTarget represents a target environment for deployments
type DeploymentTarget struct {
	// ID is a unique identifier for this target
	ID string
	// Name of the target
	Name string
	// Type of environment (dev, staging, prod)
	Environment string
	// Host or hosts to deploy to
	Hosts []string
	// Credentials needed for deployment
	Credentials map[string]string
	// Configuration for the deployment
	Config map[string]string
}

// Deployment represents a deployment of a pipeline artifact
type Deployment struct {
	// ID is a unique identifier for this deployment
	ID string
	// Pipeline ID
	PipelineID string
	// Artifact ID
	ArtifactID string
	// Target ID
	TargetID string
	// Status of the deployment
	Status PipelineStatus
	// Version identifier for the deployment
	Version string
	// User who approved the deployment
	ApprovedBy string
	// Start time
	StartTime time.Time
	// End time
	EndTime time.Time
	// Duration in seconds
	DurationSeconds int64
	// Rollback information
	CanRollback bool
	// Previous deployment ID
	PreviousDeploymentID string
}

// CICDProvider defines the interface for CI/CD providers
type CICDProvider interface {
	// Name returns the name of the provider
	Name() string
	// Type returns the type of the provider
	Type() ProviderType
	// Initialize initializes the provider
	Initialize(ctx context.Context) error
	// CreatePipeline creates a new pipeline
	CreatePipeline(ctx context.Context, pipeline *Pipeline) error
	// TriggerPipeline triggers a pipeline
	TriggerPipeline(ctx context.Context, pipelineID string, params map[string]string) error
	// StopPipeline stops a running pipeline
	StopPipeline(ctx context.Context, pipelineID string) error
	// GetPipeline returns information about a pipeline
	GetPipeline(ctx context.Context, pipelineID string) (*Pipeline, error)
	// ListPipelines returns a list of pipelines
	ListPipelines(ctx context.Context, filter map[string]string) ([]*Pipeline, error)
	// GetArtifact returns information about an artifact
	GetArtifact(ctx context.Context, pipelineID, artifactID string) (*PipelineArtifact, error)
	// ListArtifacts returns a list of artifacts for a pipeline
	ListArtifacts(ctx context.Context, pipelineID string) ([]*PipelineArtifact, error)
	// DownloadArtifact downloads an artifact
	DownloadArtifact(ctx context.Context, pipelineID, artifactID, destination string) error
	// CreateDeployment creates a new deployment
	CreateDeployment(ctx context.Context, deployment *Deployment) error
	// ExecuteDeployment executes a deployment
	ExecuteDeployment(ctx context.Context, deploymentID string) error
	// RollbackDeployment rolls back a deployment
	RollbackDeployment(ctx context.Context, deploymentID string) error
	// GetDeployment returns information about a deployment
	GetDeployment(ctx context.Context, deploymentID string) (*Deployment, error)
	// ListDeployments returns a list of deployments
	ListDeployments(ctx context.Context, filter map[string]string) ([]*Deployment, error)
}

// CICDManager manages CI/CD providers and integrations
type CICDManager struct {
	// Mutex for protecting concurrent access
	mu sync.RWMutex
	// Map of provider name to provider
	providers map[string]CICDProvider
	// Map of deployment target ID to deployment target
	deploymentTargets map[string]*DeploymentTarget
	// Is the manager initialized
	initialized bool
}

// NewCICDManager creates a new CI/CD manager
func NewCICDManager() *CICDManager {
	return &CICDManager{
		providers:         make(map[string]CICDProvider),
		deploymentTargets: make(map[string]*DeploymentTarget),
		initialized:       false,
	}
}

// Initialize initializes the CI/CD manager
func (m *CICDManager) Initialize(ctx context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.initialized {
		return fmt.Errorf("CI/CD manager already initialized")
	}

	// Initialize all providers
	for name, provider := range m.providers {
		if err := provider.Initialize(ctx); err != nil {
			return fmt.Errorf("failed to initialize provider %s: %v", name, err)
		}
	}

	m.initialized = true
	return nil
}

// RegisterProvider registers a CI/CD provider with the manager
func (m *CICDManager) RegisterProvider(provider CICDProvider) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	name := provider.Name()
	if _, exists := m.providers[name]; exists {
		return fmt.Errorf("provider with name %s already registered", name)
	}

	m.providers[name] = provider
	return nil
}

// UnregisterProvider unregisters a CI/CD provider
func (m *CICDManager) UnregisterProvider(name string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if _, exists := m.providers[name]; !exists {
		return fmt.Errorf("provider %s not found", name)
	}

	delete(m.providers, name)
	return nil
}

// GetProvider returns a CI/CD provider by name
func (m *CICDManager) GetProvider(name string) (CICDProvider, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	provider, exists := m.providers[name]
	if !exists {
		return nil, fmt.Errorf("provider %s not found", name)
	}

	return provider, nil
}

// ListProviders returns a list of all registered CI/CD providers
func (m *CICDManager) ListProviders() []CICDProvider {
	m.mu.RLock()
	defer m.mu.RUnlock()

	providers := make([]CICDProvider, 0, len(m.providers))
	for _, provider := range m.providers {
		providers = append(providers, provider)
	}

	return providers
}

// CreatePipeline creates a new pipeline using the specified provider
func (m *CICDManager) CreatePipeline(ctx context.Context, providerName string, pipeline *Pipeline) error {
	m.mu.RLock()
	provider, exists := m.providers[providerName]
	m.mu.RUnlock()

	if !exists {
		return fmt.Errorf("provider %s not found", providerName)
	}

	return provider.CreatePipeline(ctx, pipeline)
}

// TriggerPipeline triggers a pipeline
func (m *CICDManager) TriggerPipeline(ctx context.Context, providerName, pipelineID string, params map[string]string) error {
	m.mu.RLock()
	provider, exists := m.providers[providerName]
	m.mu.RUnlock()

	if !exists {
		return fmt.Errorf("provider %s not found", providerName)
	}

	return provider.TriggerPipeline(ctx, pipelineID, params)
}

// StopPipeline stops a running pipeline
func (m *CICDManager) StopPipeline(ctx context.Context, providerName, pipelineID string) error {
	m.mu.RLock()
	provider, exists := m.providers[providerName]
	m.mu.RUnlock()

	if !exists {
		return fmt.Errorf("provider %s not found", providerName)
	}

	return provider.StopPipeline(ctx, pipelineID)
}

// GetPipeline returns information about a pipeline
func (m *CICDManager) GetPipeline(ctx context.Context, providerName, pipelineID string) (*Pipeline, error) {
	m.mu.RLock()
	provider, exists := m.providers[providerName]
	m.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("provider %s not found", providerName)
	}

	return provider.GetPipeline(ctx, pipelineID)
}

// ListPipelines returns a list of pipelines
func (m *CICDManager) ListPipelines(ctx context.Context, providerName string, filter map[string]string) ([]*Pipeline, error) {
	m.mu.RLock()
	provider, exists := m.providers[providerName]
	m.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("provider %s not found", providerName)
	}

	return provider.ListPipelines(ctx, filter)
}

// RegisterDeploymentTarget registers a deployment target
func (m *CICDManager) RegisterDeploymentTarget(target *DeploymentTarget) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if target.ID == "" {
		return fmt.Errorf("deployment target ID cannot be empty")
	}

	if _, exists := m.deploymentTargets[target.ID]; exists {
		return fmt.Errorf("deployment target with ID %s already exists", target.ID)
	}

	m.deploymentTargets[target.ID] = target
	return nil
}

// UnregisterDeploymentTarget unregisters a deployment target
func (m *CICDManager) UnregisterDeploymentTarget(targetID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if _, exists := m.deploymentTargets[targetID]; !exists {
		return fmt.Errorf("deployment target %s not found", targetID)
	}

	delete(m.deploymentTargets, targetID)
	return nil
}

// GetDeploymentTarget returns a deployment target by ID
func (m *CICDManager) GetDeploymentTarget(targetID string) (*DeploymentTarget, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	target, exists := m.deploymentTargets[targetID]
	if !exists {
		return nil, fmt.Errorf("deployment target %s not found", targetID)
	}

	return target, nil
}

// ListDeploymentTargets returns a list of all deployment targets
func (m *CICDManager) ListDeploymentTargets() []*DeploymentTarget {
	m.mu.RLock()
	defer m.mu.RUnlock()

	targets := make([]*DeploymentTarget, 0, len(m.deploymentTargets))
	for _, target := range m.deploymentTargets {
		targets = append(targets, target)
	}

	return targets
}

// CreateDeployment creates a new deployment
func (m *CICDManager) CreateDeployment(ctx context.Context, providerName string, deployment *Deployment) error {
	m.mu.RLock()
	provider, exists := m.providers[providerName]
	targetExists := false
	if deployment.TargetID != "" {
		_, targetExists = m.deploymentTargets[deployment.TargetID]
	}
	m.mu.RUnlock()

	if !exists {
		return fmt.Errorf("provider %s not found", providerName)
	}

	if deployment.TargetID != "" && !targetExists {
		return fmt.Errorf("deployment target %s not found", deployment.TargetID)
	}

	return provider.CreateDeployment(ctx, deployment)
}

// ExecuteDeployment executes a deployment
func (m *CICDManager) ExecuteDeployment(ctx context.Context, providerName, deploymentID string) error {
	m.mu.RLock()
	provider, exists := m.providers[providerName]
	m.mu.RUnlock()

	if !exists {
		return fmt.Errorf("provider %s not found", providerName)
	}

	return provider.ExecuteDeployment(ctx, deploymentID)
}

// RollbackDeployment rolls back a deployment
func (m *CICDManager) RollbackDeployment(ctx context.Context, providerName, deploymentID string) error {
	m.mu.RLock()
	provider, exists := m.providers[providerName]
	m.mu.RUnlock()

	if !exists {
		return fmt.Errorf("provider %s not found", providerName)
	}

	return provider.RollbackDeployment(ctx, deploymentID)
}

// GetDeployment returns information about a deployment
func (m *CICDManager) GetDeployment(ctx context.Context, providerName, deploymentID string) (*Deployment, error) {
	m.mu.RLock()
	provider, exists := m.providers[providerName]
	m.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("provider %s not found", providerName)
	}

	return provider.GetDeployment(ctx, deploymentID)
}

// ListDeployments returns a list of deployments
func (m *CICDManager) ListDeployments(ctx context.Context, providerName string, filter map[string]string) ([]*Deployment, error) {
	m.mu.RLock()
	provider, exists := m.providers[providerName]
	m.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("provider %s not found", providerName)
	}

	return provider.ListDeployments(ctx, filter)
}

// Shutdown shuts down the CI/CD manager and all providers
func (m *CICDManager) Shutdown(ctx context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if !m.initialized {
		return nil
	}

	// Nothing specific to clean up in this implementation
	m.initialized = false
	return nil
}
