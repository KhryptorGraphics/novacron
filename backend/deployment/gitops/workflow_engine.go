// Package gitops provides GitOps workflow integration with ArgoCD, git-based deployment
// approval, automated sync from git repository, drift detection, and multi-cluster management.
package gitops

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sync"
	"time"
)

// WorkflowEngine orchestrates GitOps-based deployments
type WorkflowEngine struct {
	mu                  sync.RWMutex
	workflows           map[string]*GitOpsWorkflow
	argocdClient        ArgoCDClient
	gitClient           GitClient
	approvalManager     ApprovalManager
	driftDetector       DriftDetector
	syncOrchestrator    SyncOrchestrator
	clusterManager      ClusterManager
	ctx                 context.Context
	cancel              context.CancelFunc
	wg                  sync.WaitGroup
}

// GitOpsWorkflow represents a GitOps deployment workflow
type GitOpsWorkflow struct {
	ID                string                `json:"id"`
	Name              string                `json:"name"`
	Repository        *GitRepository        `json:"repository"`
	TargetClusters    []TargetCluster       `json:"target_clusters"`
	Status            WorkflowStatus        `json:"status"`
	ApprovalRequired  bool                  `json:"approval_required"`
	Approvals         []Approval            `json:"approvals"`
	SyncPolicy        *SyncPolicy           `json:"sync_policy"`
	DriftStatus       *DriftStatus          `json:"drift_status,omitempty"`
	Applications      []Application         `json:"applications"`
	CreatedAt         time.Time             `json:"created_at"`
	UpdatedAt         time.Time             `json:"updated_at"`
	LastSyncTime      *time.Time            `json:"last_sync_time,omitempty"`
	NextSyncTime      *time.Time            `json:"next_sync_time,omitempty"`
}

// GitRepository represents a Git repository configuration
type GitRepository struct {
	URL           string            `json:"url"`
	Branch        string            `json:"branch"`
	Path          string            `json:"path"`
	Revision      string            `json:"revision"`
	Credentials   *GitCredentials   `json:"credentials,omitempty"`
	WebhookSecret string            `json:"webhook_secret,omitempty"`
	AutoSync      bool              `json:"auto_sync"`
}

// GitCredentials contains Git authentication credentials
type GitCredentials struct {
	Username      string `json:"username,omitempty"`
	Password      string `json:"password,omitempty"`
	SSHKey        string `json:"ssh_key,omitempty"`
	Token         string `json:"token,omitempty"`
}

// TargetCluster represents a deployment target cluster
type TargetCluster struct {
	Name          string            `json:"name"`
	Server        string            `json:"server"`
	Namespace     string            `json:"namespace"`
	Labels        map[string]string `json:"labels"`
	Status        ClusterStatus     `json:"status"`
	Version       string            `json:"version"`
	Healthy       bool              `json:"healthy"`
	LastSync      *time.Time        `json:"last_sync,omitempty"`
}

// ClusterStatus represents cluster status
type ClusterStatus string

const (
	ClusterStatusOnline  ClusterStatus = "online"
	ClusterStatusOffline ClusterStatus = "offline"
	ClusterStatusDegraded ClusterStatus = "degraded"
)

// WorkflowStatus represents workflow status
type WorkflowStatus string

const (
	WorkflowStatusPendingApproval WorkflowStatus = "pending_approval"
	WorkflowStatusApproved        WorkflowStatus = "approved"
	WorkflowStatusSyncing         WorkflowStatus = "syncing"
	WorkflowStatusSynced          WorkflowStatus = "synced"
	WorkflowStatusOutOfSync       WorkflowStatus = "out_of_sync"
	WorkflowStatusFailed          WorkflowStatus = "failed"
)

// Approval represents a deployment approval
type Approval struct {
	ID          string         `json:"id"`
	RequiredBy  string         `json:"required_by"`
	ApprovedBy  string         `json:"approved_by,omitempty"`
	Status      ApprovalStatus `json:"status"`
	RequestedAt time.Time      `json:"requested_at"`
	ApprovedAt  *time.Time     `json:"approved_at,omitempty"`
	Comment     string         `json:"comment,omitempty"`
	CommitHash  string         `json:"commit_hash"`
	Changes     []string       `json:"changes"`
}

// ApprovalStatus represents approval status
type ApprovalStatus string

const (
	ApprovalStatusPending  ApprovalStatus = "pending"
	ApprovalStatusApproved ApprovalStatus = "approved"
	ApprovalStatusRejected ApprovalStatus = "rejected"
	ApprovalStatusExpired  ApprovalStatus = "expired"
)

// SyncPolicy defines synchronization policy
type SyncPolicy struct {
	AutoSync          bool              `json:"auto_sync"`
	SelfHeal          bool              `json:"self_heal"`
	Prune             bool              `json:"prune"`
	AllowEmpty        bool              `json:"allow_empty"`
	SyncInterval      time.Duration     `json:"sync_interval"`
	RetryStrategy     *RetryStrategy    `json:"retry_strategy"`
	SyncOptions       []string          `json:"sync_options"`
}

// RetryStrategy defines retry strategy for failed syncs
type RetryStrategy struct {
	MaxRetries    int           `json:"max_retries"`
	RetryInterval time.Duration `json:"retry_interval"`
	Backoff       *BackoffConfig `json:"backoff,omitempty"`
}

// BackoffConfig configures exponential backoff
type BackoffConfig struct {
	InitialInterval time.Duration `json:"initial_interval"`
	MaxInterval     time.Duration `json:"max_interval"`
	Multiplier      float64       `json:"multiplier"`
}

// DriftStatus represents drift detection status
type DriftStatus struct {
	Detected      bool              `json:"detected"`
	LastCheck     time.Time         `json:"last_check"`
	DriftCount    int               `json:"drift_count"`
	Drifts        []ResourceDrift   `json:"drifts"`
	Severity      DriftSeverity     `json:"severity"`
	AutoReconcile bool              `json:"auto_reconcile"`
}

// ResourceDrift represents a detected drift
type ResourceDrift struct {
	Resource      string                 `json:"resource"`
	Kind          string                 `json:"kind"`
	Namespace     string                 `json:"namespace"`
	Expected      map[string]interface{} `json:"expected"`
	Actual        map[string]interface{} `json:"actual"`
	Differences   []string               `json:"differences"`
	DetectedAt    time.Time              `json:"detected_at"`
	Severity      DriftSeverity          `json:"severity"`
}

// DriftSeverity represents drift severity
type DriftSeverity string

const (
	DriftSeverityLow      DriftSeverity = "low"
	DriftSeverityMedium   DriftSeverity = "medium"
	DriftSeverityHigh     DriftSeverity = "high"
	DriftSeverityCritical DriftSeverity = "critical"
)

// Application represents an ArgoCD application
type Application struct {
	Name            string            `json:"name"`
	Namespace       string            `json:"namespace"`
	Project         string            `json:"project"`
	Source          *ApplicationSource `json:"source"`
	Destination     *ApplicationDestination `json:"destination"`
	SyncPolicy      *SyncPolicy       `json:"sync_policy"`
	Health          *HealthStatus     `json:"health"`
	SyncStatus      *SyncStatus       `json:"sync_status"`
	Resources       []Resource        `json:"resources"`
	CreatedAt       time.Time         `json:"created_at"`
}

// ApplicationSource defines application source
type ApplicationSource struct {
	RepoURL        string            `json:"repo_url"`
	Path           string            `json:"path"`
	TargetRevision string            `json:"target_revision"`
	Helm           *HelmSource       `json:"helm,omitempty"`
	Kustomize      *KustomizeSource  `json:"kustomize,omitempty"`
}

// HelmSource defines Helm-specific source
type HelmSource struct {
	ReleaseName string            `json:"release_name"`
	Values      string            `json:"values"`
	Parameters  map[string]string `json:"parameters"`
}

// KustomizeSource defines Kustomize-specific source
type KustomizeSource struct {
	NamePrefix string            `json:"name_prefix,omitempty"`
	NameSuffix string            `json:"name_suffix,omitempty"`
	Images     []string          `json:"images,omitempty"`
}

// ApplicationDestination defines application destination
type ApplicationDestination struct {
	Server    string `json:"server"`
	Namespace string `json:"namespace"`
}

// HealthStatus represents health status
type HealthStatus struct {
	Status  string    `json:"status"`
	Message string    `json:"message,omitempty"`
	UpdatedAt time.Time `json:"updated_at"`
}

// SyncStatus represents sync status
type SyncStatus struct {
	Status     string     `json:"status"`
	Revision   string     `json:"revision"`
	SyncedAt   *time.Time `json:"synced_at,omitempty"`
	ComparedTo ComparedTo `json:"compared_to"`
}

// ComparedTo contains comparison information
type ComparedTo struct {
	Source      ApplicationSource `json:"source"`
	Destination ApplicationDestination `json:"destination"`
}

// Resource represents a Kubernetes resource
type Resource struct {
	Group     string    `json:"group"`
	Kind      string    `json:"kind"`
	Namespace string    `json:"namespace"`
	Name      string    `json:"name"`
	Version   string    `json:"version"`
	Health    string    `json:"health"`
	Status    string    `json:"status"`
}

// ArgoCDClient interface for ArgoCD operations
type ArgoCDClient interface {
	CreateApplication(ctx context.Context, app *Application) error
	UpdateApplication(ctx context.Context, app *Application) error
	DeleteApplication(ctx context.Context, name string) error
	GetApplication(ctx context.Context, name string) (*Application, error)
	SyncApplication(ctx context.Context, name string, revision string) error
	GetApplicationHealth(ctx context.Context, name string) (*HealthStatus, error)
}

// GitClient interface for Git operations
type GitClient interface {
	Clone(ctx context.Context, repo *GitRepository) error
	Pull(ctx context.Context, repo *GitRepository) error
	GetLatestCommit(ctx context.Context, repo *GitRepository) (string, error)
	GetCommitDiff(ctx context.Context, repo *GitRepository, fromCommit, toCommit string) ([]string, error)
	ValidateManifests(ctx context.Context, path string) error
}

// ApprovalManager manages deployment approvals
type ApprovalManager interface {
	RequestApproval(ctx context.Context, workflowID string, changes []string) (*Approval, error)
	ApproveDeployment(ctx context.Context, approvalID string, approver string, comment string) error
	RejectDeployment(ctx context.Context, approvalID string, approver string, reason string) error
	GetPendingApprovals() ([]*Approval, error)
}

// DriftDetector detects configuration drift
type DriftDetector interface {
	DetectDrift(ctx context.Context, workflowID string) (*DriftStatus, error)
	ReconcileDrift(ctx context.Context, workflowID string, drift *ResourceDrift) error
	ContinuousMonitoring(ctx context.Context, workflowID string) <-chan *DriftStatus
}

// SyncOrchestrator orchestrates synchronization
type SyncOrchestrator interface {
	Sync(ctx context.Context, workflowID string) error
	GetSyncStatus(workflowID string) (*SyncResult, error)
	ScheduleSync(workflowID string, schedule time.Time) error
}

// SyncResult represents synchronization result
type SyncResult struct {
	WorkflowID    string        `json:"workflow_id"`
	Success       bool          `json:"success"`
	SyncedAt      time.Time     `json:"synced_at"`
	Duration      time.Duration `json:"duration"`
	Revision      string        `json:"revision"`
	ResourcesSync int           `json:"resources_synced"`
	Errors        []string      `json:"errors,omitempty"`
}

// ClusterManager manages multiple clusters
type ClusterManager interface {
	AddCluster(ctx context.Context, cluster *TargetCluster) error
	RemoveCluster(ctx context.Context, clusterName string) error
	GetClusterStatus(clusterName string) (*TargetCluster, error)
	ListClusters() ([]*TargetCluster, error)
	HealthCheck(ctx context.Context, clusterName string) (bool, error)
}

// NewWorkflowEngine creates a new GitOps workflow engine
func NewWorkflowEngine(
	argocdClient ArgoCDClient,
	gitClient GitClient,
	approvalManager ApprovalManager,
	driftDetector DriftDetector,
	syncOrchestrator SyncOrchestrator,
	clusterManager ClusterManager,
) *WorkflowEngine {
	ctx, cancel := context.WithCancel(context.Background())

	return &WorkflowEngine{
		workflows:        make(map[string]*GitOpsWorkflow),
		argocdClient:     argocdClient,
		gitClient:        gitClient,
		approvalManager:  approvalManager,
		driftDetector:    driftDetector,
		syncOrchestrator: syncOrchestrator,
		clusterManager:   clusterManager,
		ctx:              ctx,
		cancel:           cancel,
	}
}

// CreateWorkflow creates a new GitOps workflow
func (we *WorkflowEngine) CreateWorkflow(
	name string,
	repository *GitRepository,
	clusters []TargetCluster,
	syncPolicy *SyncPolicy,
	approvalRequired bool,
) (*GitOpsWorkflow, error) {
	we.mu.Lock()
	defer we.mu.Unlock()

	workflowID := generateWorkflowID(name)

	workflow := &GitOpsWorkflow{
		ID:               workflowID,
		Name:             name,
		Repository:       repository,
		TargetClusters:   clusters,
		Status:           WorkflowStatusPendingApproval,
		ApprovalRequired: approvalRequired,
		SyncPolicy:       syncPolicy,
		Applications:     []Application{},
		CreatedAt:        time.Now(),
		UpdatedAt:        time.Now(),
	}

	we.workflows[workflowID] = workflow

	// Start drift detection if enabled
	if syncPolicy.SelfHeal {
		we.wg.Add(1)
		go we.monitorDrift(workflowID)
	}

	// Schedule automatic sync if enabled
	if syncPolicy.AutoSync && syncPolicy.SyncInterval > 0 {
		we.wg.Add(1)
		go we.autoSync(workflowID)
	}

	return workflow, nil
}

// DeployWorkflow deploys a GitOps workflow
func (we *WorkflowEngine) DeployWorkflow(workflowID string) error {
	we.mu.RLock()
	workflow, exists := we.workflows[workflowID]
	if !exists {
		we.mu.RUnlock()
		return fmt.Errorf("workflow %s not found", workflowID)
	}
	we.mu.RUnlock()

	// Check if approval is required
	if workflow.ApprovalRequired {
		// Get latest commit
		latestCommit, err := we.gitClient.GetLatestCommit(we.ctx, workflow.Repository)
		if err != nil {
			return fmt.Errorf("failed to get latest commit: %w", err)
		}

		// Get changes
		var changes []string
		if workflow.Repository.Revision != "" {
			changes, err = we.gitClient.GetCommitDiff(we.ctx, workflow.Repository, workflow.Repository.Revision, latestCommit)
			if err != nil {
				return fmt.Errorf("failed to get commit diff: %w", err)
			}
		}

		// Request approval
		approval, err := we.approvalManager.RequestApproval(we.ctx, workflowID, changes)
		if err != nil {
			return fmt.Errorf("failed to request approval: %w", err)
		}

		we.mu.Lock()
		workflow.Approvals = append(workflow.Approvals, *approval)
		workflow.Status = WorkflowStatusPendingApproval
		we.mu.Unlock()

		return nil
	}

	// Deploy without approval
	return we.executeDeploy(workflowID)
}

// ApproveWorkflow approves a workflow deployment
func (we *WorkflowEngine) ApproveWorkflow(workflowID string, approver string, comment string) error {
	we.mu.RLock()
	workflow, exists := we.workflows[workflowID]
	if !exists {
		we.mu.RUnlock()
		return fmt.Errorf("workflow %s not found", workflowID)
	}

	if len(workflow.Approvals) == 0 {
		we.mu.RUnlock()
		return fmt.Errorf("no pending approvals for workflow %s", workflowID)
	}

	approval := &workflow.Approvals[len(workflow.Approvals)-1]
	we.mu.RUnlock()

	// Approve via approval manager
	if err := we.approvalManager.ApproveDeployment(we.ctx, approval.ID, approver, comment); err != nil {
		return err
	}

	// Update approval
	now := time.Now()
	approval.ApprovedBy = approver
	approval.ApprovedAt = &now
	approval.Status = ApprovalStatusApproved
	approval.Comment = comment

	we.mu.Lock()
	workflow.Status = WorkflowStatusApproved
	we.mu.Unlock()

	// Execute deployment
	return we.executeDeploy(workflowID)
}

// executeDeploy executes the deployment
func (we *WorkflowEngine) executeDeploy(workflowID string) error {
	we.mu.RLock()
	workflow, exists := we.workflows[workflowID]
	if !exists {
		we.mu.RUnlock()
		return fmt.Errorf("workflow %s not found", workflowID)
	}
	we.mu.RUnlock()

	we.mu.Lock()
	workflow.Status = WorkflowStatusSyncing
	we.mu.Unlock()

	// Validate manifests
	if err := we.gitClient.ValidateManifests(we.ctx, workflow.Repository.Path); err != nil {
		we.mu.Lock()
		workflow.Status = WorkflowStatusFailed
		we.mu.Unlock()
		return fmt.Errorf("manifest validation failed: %w", err)
	}

	// Sync to all target clusters
	for _, cluster := range workflow.TargetClusters {
		// Create ArgoCD application for cluster
		app := &Application{
			Name:      fmt.Sprintf("%s-%s", workflow.Name, cluster.Name),
			Namespace: cluster.Namespace,
			Project:   "default",
			Source: &ApplicationSource{
				RepoURL:        workflow.Repository.URL,
				Path:           workflow.Repository.Path,
				TargetRevision: workflow.Repository.Branch,
			},
			Destination: &ApplicationDestination{
				Server:    cluster.Server,
				Namespace: cluster.Namespace,
			},
			SyncPolicy: workflow.SyncPolicy,
			CreatedAt:  time.Now(),
		}

		if err := we.argocdClient.CreateApplication(we.ctx, app); err != nil {
			return fmt.Errorf("failed to create application for cluster %s: %w", cluster.Name, err)
		}

		// Sync application
		latestCommit, _ := we.gitClient.GetLatestCommit(we.ctx, workflow.Repository)
		if err := we.argocdClient.SyncApplication(we.ctx, app.Name, latestCommit); err != nil {
			return fmt.Errorf("failed to sync application for cluster %s: %w", cluster.Name, err)
		}

		we.mu.Lock()
		workflow.Applications = append(workflow.Applications, *app)
		we.mu.Unlock()
	}

	// Update status
	we.mu.Lock()
	workflow.Status = WorkflowStatusSynced
	now := time.Now()
	workflow.LastSyncTime = &now
	workflow.UpdatedAt = now
	we.mu.Unlock()

	return nil
}

// monitorDrift monitors for configuration drift
func (we *WorkflowEngine) monitorDrift(workflowID string) {
	defer we.wg.Done()

	driftChan := we.driftDetector.ContinuousMonitoring(we.ctx, workflowID)

	for {
		select {
		case driftStatus := <-driftChan:
			we.mu.Lock()
			workflow, exists := we.workflows[workflowID]
			if exists {
				workflow.DriftStatus = driftStatus

				// Auto-reconcile if enabled and drift detected
				if workflow.SyncPolicy.SelfHeal && driftStatus.Detected {
					for _, drift := range driftStatus.Drifts {
						_ = we.driftDetector.ReconcileDrift(we.ctx, workflowID, &drift)
					}
				}
			}
			we.mu.Unlock()

		case <-we.ctx.Done():
			return
		}
	}
}

// autoSync performs automatic synchronization
func (we *WorkflowEngine) autoSync(workflowID string) {
	defer we.wg.Done()

	we.mu.RLock()
	workflow, exists := we.workflows[workflowID]
	if !exists {
		we.mu.RUnlock()
		return
	}
	interval := workflow.SyncPolicy.SyncInterval
	we.mu.RUnlock()

	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			_ = we.syncOrchestrator.Sync(we.ctx, workflowID)

		case <-we.ctx.Done():
			return
		}
	}
}

// GetWorkflow retrieves a workflow
func (we *WorkflowEngine) GetWorkflow(workflowID string) (*GitOpsWorkflow, error) {
	we.mu.RLock()
	defer we.mu.RUnlock()

	workflow, exists := we.workflows[workflowID]
	if !exists {
		return nil, fmt.Errorf("workflow %s not found", workflowID)
	}

	return workflow, nil
}

// ListWorkflows lists all workflows
func (we *WorkflowEngine) ListWorkflows() []*GitOpsWorkflow {
	we.mu.RLock()
	defer we.mu.RUnlock()

	workflows := make([]*GitOpsWorkflow, 0, len(we.workflows))
	for _, workflow := range we.workflows {
		workflows = append(workflows, workflow)
	}

	return workflows
}

// Shutdown gracefully shuts down the engine
func (we *WorkflowEngine) Shutdown(ctx context.Context) error {
	we.cancel()

	done := make(chan struct{})
	go func() {
		we.wg.Wait()
		close(done)
	}()

	select {
	case <-done:
		return nil
	case <-ctx.Done():
		return fmt.Errorf("shutdown timeout exceeded")
	}
}

// generateWorkflowID generates a unique workflow ID
func generateWorkflowID(name string) string {
	data := fmt.Sprintf("%s:%d", name, time.Now().UnixNano())
	hash := sha256.Sum256([]byte(data))
	return hex.EncodeToString(hash[:])[:16]
}

// MarshalJSON implements custom JSON marshaling
func (gw *GitOpsWorkflow) MarshalJSON() ([]byte, error) {
	type Alias GitOpsWorkflow

	return json.Marshal(&struct {
		*Alias
		Age string `json:"age"`
	}{
		Alias: (*Alias)(gw),
		Age:   time.Since(gw.CreatedAt).String(),
	})
}
