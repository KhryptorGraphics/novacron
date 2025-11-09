package registry

import (
	"fmt"
	"sync"
	"time"
)

// ModelRegistry manages ML model storage and versioning
type ModelRegistry struct {
	models     map[string]*ModelEntry
	versions   map[string][]*ModelVersion
	deployments map[string]*Deployment
	mu         sync.RWMutex
}

// ModelEntry represents a registered model
type ModelEntry struct {
	Name        string
	Description string
	CreatedAt   time.Time
	UpdatedAt   time.Time
	Tags        []string
	Owner       string
	Framework   string // "tensorflow", "pytorch", "onnx"
}

// ModelVersion represents a specific model version
type ModelVersion struct {
	ModelName    string
	Version      string
	Weights      interface{}
	Metrics      map[string]float64
	Hyperparams  map[string]interface{}
	CreatedAt    time.Time
	Status       string // "development", "staging", "production", "archived"
	ArtifactPath string
	Lineage      *ModelLineage
	Size         int64
}

// ModelLineage tracks model provenance
type ModelLineage struct {
	TrainingDataset string
	ParentModel     string
	ParentVersion   string
	TrainingJob     string
	GitCommit       string
	Environment     map[string]string
}

// Deployment represents a model deployment
type Deployment struct {
	ID            string
	ModelName     string
	ModelVersion  string
	Environment   string // "production", "staging", "development"
	Endpoint      string
	Status        string // "active", "inactive", "failed"
	DeployedAt    time.Time
	Traffic       float64 // Traffic percentage for A/B testing
	Metrics       *DeploymentMetrics
}

// DeploymentMetrics tracks deployment performance
type DeploymentMetrics struct {
	RequestCount    int64
	AvgLatency      float64
	P50Latency      float64
	P95Latency      float64
	P99Latency      float64
	ErrorRate       float64
	Throughput      float64
	LastUpdated     time.Time
}

// NewModelRegistry creates a new model registry
func NewModelRegistry() *ModelRegistry {
	return &ModelRegistry{
		models:      make(map[string]*ModelEntry),
		versions:    make(map[string][]*ModelVersion),
		deployments: make(map[string]*Deployment),
	}
}

// RegisterModel registers a new model
func (r *ModelRegistry) RegisterModel(name, description, framework, owner string, tags []string) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if _, exists := r.models[name]; exists {
		return fmt.Errorf("model %s already registered", name)
	}

	entry := &ModelEntry{
		Name:        name,
		Description: description,
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
		Tags:        tags,
		Owner:       owner,
		Framework:   framework,
	}

	r.models[name] = entry
	r.versions[name] = make([]*ModelVersion, 0)

	return nil
}

// AddVersion adds a new version of a model
func (r *ModelRegistry) AddVersion(modelName, version string, weights interface{}, metrics map[string]float64, hyperparams map[string]interface{}) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if _, exists := r.models[modelName]; !exists {
		return fmt.Errorf("model %s not registered", modelName)
	}

	// Check if version already exists
	for _, v := range r.versions[modelName] {
		if v.Version == version {
			return fmt.Errorf("version %s already exists for model %s", version, modelName)
		}
	}

	modelVersion := &ModelVersion{
		ModelName:   modelName,
		Version:     version,
		Weights:     weights,
		Metrics:     metrics,
		Hyperparams: hyperparams,
		CreatedAt:   time.Now(),
		Status:      "development",
	}

	r.versions[modelName] = append(r.versions[modelName], modelVersion)
	r.models[modelName].UpdatedAt = time.Now()

	return nil
}

// GetModel retrieves a model entry
func (r *ModelRegistry) GetModel(name string) (*ModelEntry, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	model, exists := r.models[name]
	if !exists {
		return nil, fmt.Errorf("model %s not found", name)
	}

	return model, nil
}

// GetVersion retrieves a specific model version
func (r *ModelRegistry) GetVersion(modelName, version string) (*ModelVersion, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	versions, exists := r.versions[modelName]
	if !exists {
		return nil, fmt.Errorf("model %s not found", modelName)
	}

	for _, v := range versions {
		if v.Version == version {
			return v, nil
		}
	}

	return nil, fmt.Errorf("version %s not found for model %s", version, modelName)
}

// GetLatestVersion retrieves the latest version of a model
func (r *ModelRegistry) GetLatestVersion(modelName string) (*ModelVersion, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	versions, exists := r.versions[modelName]
	if !exists {
		return nil, fmt.Errorf("model %s not found", modelName)
	}

	if len(versions) == 0 {
		return nil, fmt.Errorf("no versions found for model %s", modelName)
	}

	// Return most recently created version
	latest := versions[0]
	for _, v := range versions {
		if v.CreatedAt.After(latest.CreatedAt) {
			latest = v
		}
	}

	return latest, nil
}

// PromoteVersion promotes a version to a new status
func (r *ModelRegistry) PromoteVersion(modelName, version, newStatus string) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	versions, exists := r.versions[modelName]
	if !exists {
		return fmt.Errorf("model %s not found", modelName)
	}

	for _, v := range versions {
		if v.Version == version {
			v.Status = newStatus
			return nil
		}
	}

	return fmt.Errorf("version %s not found", version)
}

// Deploy deploys a model version
func (r *ModelRegistry) Deploy(modelName, version, environment, endpoint string, traffic float64) (*Deployment, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	// Verify model version exists
	versions, exists := r.versions[modelName]
	if !exists {
		return nil, fmt.Errorf("model %s not found", modelName)
	}

	versionExists := false
	for _, v := range versions {
		if v.Version == version {
			versionExists = true
			break
		}
	}

	if !versionExists {
		return nil, fmt.Errorf("version %s not found", version)
	}

	deploymentID := fmt.Sprintf("%s-%s-%s-%d", modelName, version, environment, time.Now().Unix())

	deployment := &Deployment{
		ID:           deploymentID,
		ModelName:    modelName,
		ModelVersion: version,
		Environment:  environment,
		Endpoint:     endpoint,
		Status:       "active",
		DeployedAt:   time.Now(),
		Traffic:      traffic,
		Metrics:      &DeploymentMetrics{
			LastUpdated: time.Now(),
		},
	}

	r.deployments[deploymentID] = deployment

	return deployment, nil
}

// GetDeployment retrieves a deployment
func (r *ModelRegistry) GetDeployment(deploymentID string) (*Deployment, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	deployment, exists := r.deployments[deploymentID]
	if !exists {
		return nil, fmt.Errorf("deployment %s not found", deploymentID)
	}

	return deployment, nil
}

// ListDeployments lists all deployments for a model
func (r *ModelRegistry) ListDeployments(modelName string) ([]*Deployment, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	deployments := make([]*Deployment, 0)
	for _, d := range r.deployments {
		if d.ModelName == modelName {
			deployments = append(deployments, d)
		}
	}

	return deployments, nil
}

// UpdateDeploymentMetrics updates deployment metrics
func (r *ModelRegistry) UpdateDeploymentMetrics(deploymentID string, metrics *DeploymentMetrics) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	deployment, exists := r.deployments[deploymentID]
	if !exists {
		return fmt.Errorf("deployment %s not found", deploymentID)
	}

	deployment.Metrics = metrics
	deployment.Metrics.LastUpdated = time.Now()

	return nil
}

// CompareVersions compares metrics between two model versions
func (r *ModelRegistry) CompareVersions(modelName, version1, version2 string) (map[string]float64, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	v1, err := r.getVersionLocked(modelName, version1)
	if err != nil {
		return nil, err
	}

	v2, err := r.getVersionLocked(modelName, version2)
	if err != nil {
		return nil, err
	}

	comparison := make(map[string]float64)
	for metric, val1 := range v1.Metrics {
		if val2, ok := v2.Metrics[metric]; ok {
			comparison[metric+"_diff"] = val1 - val2
			comparison[metric+"_ratio"] = val1 / val2
		}
	}

	return comparison, nil
}

// getVersionLocked retrieves version without lock (internal use)
func (r *ModelRegistry) getVersionLocked(modelName, version string) (*ModelVersion, error) {
	versions, exists := r.versions[modelName]
	if !exists {
		return nil, fmt.Errorf("model %s not found", modelName)
	}

	for _, v := range versions {
		if v.Version == version {
			return v, nil
		}
	}

	return nil, fmt.Errorf("version %s not found", version)
}

// ListModels lists all registered models
func (r *ModelRegistry) ListModels() []*ModelEntry {
	r.mu.RLock()
	defer r.mu.RUnlock()

	models := make([]*ModelEntry, 0, len(r.models))
	for _, m := range r.models {
		models = append(models, m)
	}

	return models
}

// SearchModels searches models by tags
func (r *ModelRegistry) SearchModels(tags []string) []*ModelEntry {
	r.mu.RLock()
	defer r.mu.RUnlock()

	results := make([]*ModelEntry, 0)
	for _, model := range r.models {
		if r.hasAllTags(model.Tags, tags) {
			results = append(results, model)
		}
	}

	return results
}

// hasAllTags checks if model has all required tags
func (r *ModelRegistry) hasAllTags(modelTags, requiredTags []string) bool {
	for _, required := range requiredTags {
		found := false
		for _, tag := range modelTags {
			if tag == required {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	return true
}

// DeleteVersion deletes a model version
func (r *ModelRegistry) DeleteVersion(modelName, version string) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	versions, exists := r.versions[modelName]
	if !exists {
		return fmt.Errorf("model %s not found", modelName)
	}

	for i, v := range versions {
		if v.Version == version {
			// Remove version from slice
			r.versions[modelName] = append(versions[:i], versions[i+1:]...)
			return nil
		}
	}

	return fmt.Errorf("version %s not found", version)
}

// SetLineage sets lineage information for a model version
func (r *ModelRegistry) SetLineage(modelName, version string, lineage *ModelLineage) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	versions, exists := r.versions[modelName]
	if !exists {
		return fmt.Errorf("model %s not found", modelName)
	}

	for _, v := range versions {
		if v.Version == version {
			v.Lineage = lineage
			return nil
		}
	}

	return fmt.Errorf("version %s not found", version)
}
