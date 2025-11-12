// Package registry implements ML model registry with versioning and lifecycle management
package registry

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/google/uuid"
)

// ModelStage represents the deployment stage of a model
type ModelStage string

const (
	StageDevelopment ModelStage = "development"
	StageStaging     ModelStage = "staging"
	StageProduction  ModelStage = "production"
	StageArchived    ModelStage = "archived"
)

// ModelFramework represents ML framework types
type ModelFramework string

const (
	FrameworkTensorFlow ModelFramework = "tensorflow"
	FrameworkPyTorch    ModelFramework = "pytorch"
	FrameworkScikit     ModelFramework = "scikit-learn"
	FrameworkXGBoost    ModelFramework = "xgboost"
	FrameworkONNX       ModelFramework = "onnx"
	FrameworkCustom     ModelFramework = "custom"
)

// ModelMetadata contains comprehensive model information
type ModelMetadata struct {
	ID                string                 `json:"id"`
	Name              string                 `json:"name"`
	Version           string                 `json:"version"`
	Framework         ModelFramework         `json:"framework"`
	FrameworkVersion  string                 `json:"framework_version"`
	Stage             ModelStage             `json:"stage"`
	Description       string                 `json:"description"`
	Tags              []string               `json:"tags"`
	Author            string                 `json:"author"`
	CreatedAt         time.Time              `json:"created_at"`
	UpdatedAt         time.Time              `json:"updated_at"`

	// Training information
	TrainingDataset   string                 `json:"training_dataset"`
	TrainingStarted   time.Time              `json:"training_started"`
	TrainingCompleted time.Time              `json:"training_completed"`
	TrainingDuration  time.Duration          `json:"training_duration"`

	// Model artifacts
	ArtifactPath      string                 `json:"artifact_path"`
	ArtifactSize      int64                  `json:"artifact_size"`
	ArtifactChecksum  string                 `json:"artifact_checksum"`
	ConfigPath        string                 `json:"config_path"`

	// Performance metrics
	Metrics           map[string]float64     `json:"metrics"`
	Hyperparameters   map[string]interface{} `json:"hyperparameters"`

	// Deployment information
	DeploymentCount   int                    `json:"deployment_count"`
	LastDeployed      time.Time              `json:"last_deployed"`
	ApprovalStatus    string                 `json:"approval_status"`
	ApprovedBy        string                 `json:"approved_by"`
	ApprovedAt        time.Time              `json:"approved_at"`

	// Lineage
	ParentModelID     string                 `json:"parent_model_id"`
	DerivedFromDataset string                `json:"derived_from_dataset"`

	// Custom metadata
	CustomMetadata    map[string]interface{} `json:"custom_metadata"`
}

// ModelVersion represents a specific version of a model
type ModelVersion struct {
	Version    string        `json:"version"`
	ModelID    string        `json:"model_id"`
	Metadata   ModelMetadata `json:"metadata"`
	Changelog  string        `json:"changelog"`
	IsActive   bool          `json:"is_active"`
	CreatedAt  time.Time     `json:"created_at"`
}

// ModelComparison represents comparison between models
type ModelComparison struct {
	ModelA        string             `json:"model_a"`
	ModelB        string             `json:"model_b"`
	MetricsDiff   map[string]float64 `json:"metrics_diff"`
	Winner        string             `json:"winner"`
	ComparedAt    time.Time          `json:"compared_at"`
}

// ApprovalRequest represents a model approval workflow
type ApprovalRequest struct {
	ID            string    `json:"id"`
	ModelID       string    `json:"model_id"`
	RequestedBy   string    `json:"requested_by"`
	RequestedAt   time.Time `json:"requested_at"`
	TargetStage   ModelStage `json:"target_stage"`
	Justification string    `json:"justification"`
	Status        string    `json:"status"` // pending, approved, rejected
	ReviewedBy    string    `json:"reviewed_by"`
	ReviewedAt    time.Time `json:"reviewed_at"`
	Comments      []string  `json:"comments"`
}

// ModelRegistry manages ML model lifecycle
type ModelRegistry struct {
	mu             sync.RWMutex
	models         map[string]*ModelMetadata
	versions       map[string][]ModelVersion
	approvals      map[string]*ApprovalRequest
	comparisons    []ModelComparison
	storagePath    string
	metadataPath   string
	enableVersioning bool
}

// NewModelRegistry creates a new model registry
func NewModelRegistry(storagePath string) *ModelRegistry {
	return &ModelRegistry{
		models:         make(map[string]*ModelMetadata),
		versions:       make(map[string][]ModelVersion),
		approvals:      make(map[string]*ApprovalRequest),
		comparisons:    make([]ModelComparison, 0),
		storagePath:    storagePath,
		metadataPath:   filepath.Join(storagePath, "metadata"),
		enableVersioning: true,
	}
}

// RegisterModel registers a new model in the registry
func (mr *ModelRegistry) RegisterModel(ctx context.Context, metadata *ModelMetadata) error {
	mr.mu.Lock()
	defer mr.mu.Unlock()

	if metadata.ID == "" {
		metadata.ID = uuid.New().String()
	}

	if metadata.Version == "" {
		metadata.Version = "v1.0.0"
	}

	metadata.CreatedAt = time.Now()
	metadata.UpdatedAt = time.Now()
	metadata.Stage = StageDevelopment

	// Store model metadata
	mr.models[metadata.ID] = metadata

	// Create initial version
	if mr.enableVersioning {
		version := ModelVersion{
			Version:   metadata.Version,
			ModelID:   metadata.ID,
			Metadata:  *metadata,
			IsActive:  true,
			CreatedAt: time.Now(),
		}
		mr.versions[metadata.ID] = append(mr.versions[metadata.ID], version)
	}

	// Persist to disk
	if err := mr.saveMetadata(metadata); err != nil {
		return fmt.Errorf("failed to save metadata: %w", err)
	}

	return nil
}

// GetModel retrieves a model by ID
func (mr *ModelRegistry) GetModel(ctx context.Context, modelID string) (*ModelMetadata, error) {
	mr.mu.RLock()
	defer mr.mu.RUnlock()

	model, exists := mr.models[modelID]
	if !exists {
		return nil, fmt.Errorf("model not found: %s", modelID)
	}

	return model, nil
}

// UpdateModel updates model metadata
func (mr *ModelRegistry) UpdateModel(ctx context.Context, modelID string, updates map[string]interface{}) error {
	mr.mu.Lock()
	defer mr.mu.Unlock()

	model, exists := mr.models[modelID]
	if !exists {
		return fmt.Errorf("model not found: %s", modelID)
	}

	// Apply updates
	if desc, ok := updates["description"].(string); ok {
		model.Description = desc
	}
	if tags, ok := updates["tags"].([]string); ok {
		model.Tags = tags
	}
	if metrics, ok := updates["metrics"].(map[string]float64); ok {
		model.Metrics = metrics
	}

	model.UpdatedAt = time.Now()

	return mr.saveMetadata(model)
}

// CreateVersion creates a new version of a model
func (mr *ModelRegistry) CreateVersion(ctx context.Context, modelID, version, changelog string) error {
	mr.mu.Lock()
	defer mr.mu.Unlock()

	model, exists := mr.models[modelID]
	if !exists {
		return fmt.Errorf("model not found: %s", modelID)
	}

	// Deactivate previous versions
	for i := range mr.versions[modelID] {
		mr.versions[modelID][i].IsActive = false
	}

	// Create new version
	newVersion := ModelVersion{
		Version:   version,
		ModelID:   modelID,
		Metadata:  *model,
		Changelog: changelog,
		IsActive:  true,
		CreatedAt: time.Now(),
	}

	mr.versions[modelID] = append(mr.versions[modelID], newVersion)
	model.Version = version
	model.UpdatedAt = time.Now()

	return mr.saveMetadata(model)
}

// ListVersions returns all versions of a model
func (mr *ModelRegistry) ListVersions(ctx context.Context, modelID string) ([]ModelVersion, error) {
	mr.mu.RLock()
	defer mr.mu.RUnlock()

	versions, exists := mr.versions[modelID]
	if !exists {
		return nil, fmt.Errorf("no versions found for model: %s", modelID)
	}

	return versions, nil
}

// PromoteModel promotes a model to a new stage
func (mr *ModelRegistry) PromoteModel(ctx context.Context, modelID string, targetStage ModelStage) error {
	mr.mu.Lock()
	defer mr.mu.Unlock()

	model, exists := mr.models[modelID]
	if !exists {
		return fmt.Errorf("model not found: %s", modelID)
	}

	// Validate stage transition
	if !mr.isValidStageTransition(model.Stage, targetStage) {
		return fmt.Errorf("invalid stage transition: %s -> %s", model.Stage, targetStage)
	}

	model.Stage = targetStage
	model.UpdatedAt = time.Now()

	if targetStage == StageProduction {
		model.LastDeployed = time.Now()
		model.DeploymentCount++
	}

	return mr.saveMetadata(model)
}

// RequestApproval creates an approval request for stage promotion
func (mr *ModelRegistry) RequestApproval(ctx context.Context, modelID, requestedBy string, targetStage ModelStage, justification string) (string, error) {
	mr.mu.Lock()
	defer mr.mu.Unlock()

	model, exists := mr.models[modelID]
	if !exists {
		return "", fmt.Errorf("model not found: %s", modelID)
	}

	requestID := uuid.New().String()
	approval := &ApprovalRequest{
		ID:            requestID,
		ModelID:       modelID,
		RequestedBy:   requestedBy,
		RequestedAt:   time.Now(),
		TargetStage:   targetStage,
		Justification: justification,
		Status:        "pending",
		Comments:      make([]string, 0),
	}

	mr.approvals[requestID] = approval
	model.ApprovalStatus = "pending"

	return requestID, nil
}

// ApproveModel approves a model for promotion
func (mr *ModelRegistry) ApproveModel(ctx context.Context, approvalID, approvedBy string, comments string) error {
	mr.mu.Lock()
	defer mr.mu.Unlock()

	approval, exists := mr.approvals[approvalID]
	if !exists {
		return fmt.Errorf("approval request not found: %s", approvalID)
	}

	approval.Status = "approved"
	approval.ReviewedBy = approvedBy
	approval.ReviewedAt = time.Now()
	if comments != "" {
		approval.Comments = append(approval.Comments, comments)
	}

	// Update model
	model := mr.models[approval.ModelID]
	model.ApprovalStatus = "approved"
	model.ApprovedBy = approvedBy
	model.ApprovedAt = time.Now()

	// Auto-promote if approved
	return mr.promoteModelInternal(approval.ModelID, approval.TargetStage)
}

// CompareModels compares two models based on metrics
func (mr *ModelRegistry) CompareModels(ctx context.Context, modelA, modelB string) (*ModelComparison, error) {
	mr.mu.RLock()
	defer mr.mu.RUnlock()

	modelAMeta, exists := mr.models[modelA]
	if !exists {
		return nil, fmt.Errorf("model A not found: %s", modelA)
	}

	modelBMeta, exists := mr.models[modelB]
	if !exists {
		return nil, fmt.Errorf("model B not found: %s", modelB)
	}

	comparison := ModelComparison{
		ModelA:      modelA,
		ModelB:      modelB,
		MetricsDiff: make(map[string]float64),
		ComparedAt:  time.Now(),
	}

	// Calculate metric differences
	totalScoreA := 0.0
	totalScoreB := 0.0

	for metric, valueA := range modelAMeta.Metrics {
		if valueB, exists := modelBMeta.Metrics[metric]; exists {
			diff := valueA - valueB
			comparison.MetricsDiff[metric] = diff

			// Higher is better for most metrics
			totalScoreA += valueA
			totalScoreB += valueB
		}
	}

	if totalScoreA > totalScoreB {
		comparison.Winner = modelA
	} else if totalScoreB > totalScoreA {
		comparison.Winner = modelB
	} else {
		comparison.Winner = "tie"
	}

	mr.comparisons = append(mr.comparisons, comparison)

	return &comparison, nil
}

// StoreModelArtifact stores model artifact file
func (mr *ModelRegistry) StoreModelArtifact(ctx context.Context, modelID string, artifactPath string) error {
	mr.mu.Lock()
	defer mr.mu.Unlock()

	model, exists := mr.models[modelID]
	if !exists {
		return fmt.Errorf("model not found: %s", modelID)
	}

	// Calculate checksum
	checksum, err := mr.calculateChecksum(artifactPath)
	if err != nil {
		return fmt.Errorf("failed to calculate checksum: %w", err)
	}

	// Get file size
	info, err := os.Stat(artifactPath)
	if err != nil {
		return fmt.Errorf("failed to get file info: %w", err)
	}

	// Copy to registry storage
	destPath := filepath.Join(mr.storagePath, modelID, filepath.Base(artifactPath))
	if err := mr.copyFile(artifactPath, destPath); err != nil {
		return fmt.Errorf("failed to copy artifact: %w", err)
	}

	model.ArtifactPath = destPath
	model.ArtifactSize = info.Size()
	model.ArtifactChecksum = checksum
	model.UpdatedAt = time.Now()

	return mr.saveMetadata(model)
}

// ListModels returns all models with optional filtering
func (mr *ModelRegistry) ListModels(ctx context.Context, filter map[string]interface{}) ([]*ModelMetadata, error) {
	mr.mu.RLock()
	defer mr.mu.RUnlock()

	models := make([]*ModelMetadata, 0)

	for _, model := range mr.models {
		if mr.matchesFilter(model, filter) {
			models = append(models, model)
		}
	}

	return models, nil
}

// DeleteModel removes a model from the registry
func (mr *ModelRegistry) DeleteModel(ctx context.Context, modelID string) error {
	mr.mu.Lock()
	defer mr.mu.Unlock()

	model, exists := mr.models[modelID]
	if !exists {
		return fmt.Errorf("model not found: %s", modelID)
	}

	// Prevent deletion of production models
	if model.Stage == StageProduction {
		return fmt.Errorf("cannot delete production model: %s", modelID)
	}

	// Remove model artifacts
	if model.ArtifactPath != "" {
		os.RemoveAll(filepath.Dir(model.ArtifactPath))
	}

	// Remove from registry
	delete(mr.models, modelID)
	delete(mr.versions, modelID)

	return nil
}

// Internal helper methods

func (mr *ModelRegistry) isValidStageTransition(current, target ModelStage) bool {
	validTransitions := map[ModelStage][]ModelStage{
		StageDevelopment: {StageStaging, StageArchived},
		StageStaging:     {StageProduction, StageDevelopment, StageArchived},
		StageProduction:  {StageStaging, StageArchived},
		StageArchived:    {},
	}

	allowed, exists := validTransitions[current]
	if !exists {
		return false
	}

	for _, stage := range allowed {
		if stage == target {
			return true
		}
	}

	return false
}

func (mr *ModelRegistry) promoteModelInternal(modelID string, targetStage ModelStage) error {
	model := mr.models[modelID]
	model.Stage = targetStage
	model.UpdatedAt = time.Now()

	if targetStage == StageProduction {
		model.LastDeployed = time.Now()
		model.DeploymentCount++
	}

	return mr.saveMetadata(model)
}

func (mr *ModelRegistry) saveMetadata(metadata *ModelMetadata) error {
	os.MkdirAll(mr.metadataPath, 0755)

	filePath := filepath.Join(mr.metadataPath, metadata.ID+".json")
	data, err := json.MarshalIndent(metadata, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(filePath, data, 0644)
}

func (mr *ModelRegistry) calculateChecksum(filePath string) (string, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return "", err
	}
	defer file.Close()

	hash := sha256.New()
	if _, err := io.Copy(hash, file); err != nil {
		return "", err
	}

	return hex.EncodeToString(hash.Sum(nil)), nil
}

func (mr *ModelRegistry) copyFile(src, dst string) error {
	os.MkdirAll(filepath.Dir(dst), 0755)

	srcFile, err := os.Open(src)
	if err != nil {
		return err
	}
	defer srcFile.Close()

	dstFile, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer dstFile.Close()

	_, err = io.Copy(dstFile, srcFile)
	return err
}

func (mr *ModelRegistry) matchesFilter(model *ModelMetadata, filter map[string]interface{}) bool {
	if stage, ok := filter["stage"].(ModelStage); ok {
		if model.Stage != stage {
			return false
		}
	}

	if framework, ok := filter["framework"].(ModelFramework); ok {
		if model.Framework != framework {
			return false
		}
	}

	if name, ok := filter["name"].(string); ok {
		if model.Name != name {
			return false
		}
	}

	return true
}

// GetModelLineage returns the lineage tree of a model
func (mr *ModelRegistry) GetModelLineage(ctx context.Context, modelID string) ([]string, error) {
	mr.mu.RLock()
	defer mr.mu.RUnlock()

	lineage := make([]string, 0)
	current := modelID

	for current != "" {
		model, exists := mr.models[current]
		if !exists {
			break
		}

		lineage = append(lineage, current)
		current = model.ParentModelID
	}

	return lineage, nil
}

// GetApprovalRequests returns all pending approval requests
func (mr *ModelRegistry) GetApprovalRequests(ctx context.Context, status string) ([]*ApprovalRequest, error) {
	mr.mu.RLock()
	defer mr.mu.RUnlock()

	requests := make([]*ApprovalRequest, 0)

	for _, approval := range mr.approvals {
		if status == "" || approval.Status == status {
			requests = append(requests, approval)
		}
	}

	return requests, nil
}
