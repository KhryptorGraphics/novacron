// Package features implements a feature store for centralized feature management
package features

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/google/uuid"
)

// FeatureType represents the data type of a feature
type FeatureType string

const (
	FeatureTypeInt     FeatureType = "int"
	FeatureTypeFloat   FeatureType = "float"
	FeatureTypeString  FeatureType = "string"
	FeatureTypeBool    FeatureType = "bool"
	FeatureTypeArray   FeatureType = "array"
	FeatureTypeObject  FeatureType = "object"
)

// FeatureStatus represents feature lifecycle status
type FeatureStatus string

const (
	FeatureStatusDraft      FeatureStatus = "draft"
	FeatureStatusActive     FeatureStatus = "active"
	FeatureStatusDeprecated FeatureStatus = "deprecated"
	FeatureStatusArchived   FeatureStatus = "archived"
)

// Feature represents a feature definition
type Feature struct {
	ID            string                 `json:"id"`
	Name          string                 `json:"name"`
	Version       string                 `json:"version"`
	Description   string                 `json:"description"`
	FeatureType   FeatureType            `json:"feature_type"`
	Status        FeatureStatus          `json:"status"`
	CreatedAt     time.Time              `json:"created_at"`
	UpdatedAt     time.Time              `json:"updated_at"`

	// Feature definition
	EntityType    string                 `json:"entity_type"`  // e.g., "user", "transaction"
	EntityIDField string                 `json:"entity_id_field"`

	// Computation
	Source        string                 `json:"source"`        // Data source (table, stream, etc.)
	Transformation string                `json:"transformation"` // SQL, Python function, etc.
	Dependencies  []string               `json:"dependencies"`   // Other features this depends on

	// Metadata
	Owner         string                 `json:"owner"`
	Tags          []string               `json:"tags"`
	CustomMetadata map[string]interface{} `json:"custom_metadata"`

	// Monitoring
	MonitoringEnabled bool                `json:"monitoring_enabled"`
	DriftThreshold    float64             `json:"drift_threshold"`

	// Versioning
	ParentFeatureID   string              `json:"parent_feature_id"`
	VersionHistory    []string            `json:"version_history"`
}

// FeatureGroup represents a collection of related features
type FeatureGroup struct {
	ID            string    `json:"id"`
	Name          string    `json:"name"`
	Description   string    `json:"description"`
	EntityType    string    `json:"entity_type"`
	Features      []string  `json:"features"`  // Feature IDs
	CreatedAt     time.Time `json:"created_at"`
	UpdatedAt     time.Time `json:"updated_at"`
	Owner         string    `json:"owner"`
}

// FeatureValue represents a computed feature value
type FeatureValue struct {
	FeatureID    string      `json:"feature_id"`
	EntityID     string      `json:"entity_id"`
	Value        interface{} `json:"value"`
	Timestamp    time.Time   `json:"timestamp"`
	Version      string      `json:"version"`
}

// OnlineFeatureRequest represents a real-time feature request
type OnlineFeatureRequest struct {
	EntityID     string   `json:"entity_id"`
	FeatureNames []string `json:"feature_names"`
}

// OnlineFeatureResponse represents real-time feature response
type OnlineFeatureResponse struct {
	EntityID  string                 `json:"entity_id"`
	Features  map[string]interface{} `json:"features"`
	Timestamp time.Time              `json:"timestamp"`
	Latency   time.Duration          `json:"latency"`
}

// OfflineFeatureRequest represents a batch feature request
type OfflineFeatureRequest struct {
	EntityIDs    []string  `json:"entity_ids"`
	FeatureNames []string  `json:"feature_names"`
	StartTime    time.Time `json:"start_time"`
	EndTime      time.Time `json:"end_time"`
}

// FeatureStatistics represents feature statistics for monitoring
type FeatureStatistics struct {
	FeatureID    string    `json:"feature_id"`
	Timestamp    time.Time `json:"timestamp"`
	Count        int64     `json:"count"`
	NullCount    int64     `json:"null_count"`
	Mean         float64   `json:"mean"`
	StdDev       float64   `json:"std_dev"`
	Min          float64   `json:"min"`
	Max          float64   `json:"max"`
	Percentiles  map[string]float64 `json:"percentiles"`
}

// FeatureStore manages features for ML applications
type FeatureStore struct {
	mu                sync.RWMutex
	features          map[string]*Feature
	featureGroups     map[string]*FeatureGroup
	onlineCache       map[string]map[string]*FeatureValue  // featureID -> entityID -> value
	statistics        map[string]*FeatureStatistics
	enableMonitoring  bool
}

// NewFeatureStore creates a new feature store
func NewFeatureStore() *FeatureStore {
	return &FeatureStore{
		features:         make(map[string]*Feature),
		featureGroups:    make(map[string]*FeatureGroup),
		onlineCache:      make(map[string]map[string]*FeatureValue),
		statistics:       make(map[string]*FeatureStatistics),
		enableMonitoring: true,
	}
}

// RegisterFeature registers a new feature in the store
func (fs *FeatureStore) RegisterFeature(ctx context.Context, feature *Feature) error {
	fs.mu.Lock()
	defer fs.mu.Unlock()

	if feature.ID == "" {
		feature.ID = uuid.New().String()
	}

	if feature.Version == "" {
		feature.Version = "v1.0.0"
	}

	feature.CreatedAt = time.Now()
	feature.UpdatedAt = time.Now()
	feature.Status = FeatureStatusDraft

	fs.features[feature.ID] = feature

	// Initialize online cache for this feature
	fs.onlineCache[feature.ID] = make(map[string]*FeatureValue)

	return nil
}

// GetFeature retrieves a feature by ID
func (fs *FeatureStore) GetFeature(ctx context.Context, featureID string) (*Feature, error) {
	fs.mu.RLock()
	defer fs.mu.RUnlock()

	feature, exists := fs.features[featureID]
	if !exists {
		return nil, fmt.Errorf("feature not found: %s", featureID)
	}

	return feature, nil
}

// GetFeatureByName retrieves a feature by name
func (fs *FeatureStore) GetFeatureByName(ctx context.Context, name string) (*Feature, error) {
	fs.mu.RLock()
	defer fs.mu.RUnlock()

	for _, feature := range fs.features {
		if feature.Name == name && feature.Status == FeatureStatusActive {
			return feature, nil
		}
	}

	return nil, fmt.Errorf("active feature not found: %s", name)
}

// UpdateFeature updates an existing feature
func (fs *FeatureStore) UpdateFeature(ctx context.Context, featureID string, updates map[string]interface{}) error {
	fs.mu.Lock()
	defer fs.mu.Unlock()

	feature, exists := fs.features[featureID]
	if !exists {
		return fmt.Errorf("feature not found: %s", featureID)
	}

	// Apply updates
	if description, ok := updates["description"].(string); ok {
		feature.Description = description
	}
	if status, ok := updates["status"].(FeatureStatus); ok {
		feature.Status = status
	}
	if tags, ok := updates["tags"].([]string); ok {
		feature.Tags = tags
	}

	feature.UpdatedAt = time.Now()

	return nil
}

// CreateFeatureVersion creates a new version of an existing feature
func (fs *FeatureStore) CreateFeatureVersion(ctx context.Context, featureID, newVersion string) (string, error) {
	fs.mu.Lock()
	defer fs.mu.Unlock()

	parent, exists := fs.features[featureID]
	if !exists {
		return "", fmt.Errorf("feature not found: %s", featureID)
	}

	// Create new feature version
	newFeature := *parent
	newFeature.ID = uuid.New().String()
	newFeature.Version = newVersion
	newFeature.ParentFeatureID = featureID
	newFeature.CreatedAt = time.Now()
	newFeature.UpdatedAt = time.Now()
	newFeature.Status = FeatureStatusDraft

	// Update version history
	newFeature.VersionHistory = append(parent.VersionHistory, featureID)

	fs.features[newFeature.ID] = &newFeature

	return newFeature.ID, nil
}

// GetOnlineFeatures retrieves real-time feature values
func (fs *FeatureStore) GetOnlineFeatures(ctx context.Context, req *OnlineFeatureRequest) (*OnlineFeatureResponse, error) {
	startTime := time.Now()

	fs.mu.RLock()
	defer fs.mu.RUnlock()

	response := &OnlineFeatureResponse{
		EntityID:  req.EntityID,
		Features:  make(map[string]interface{}),
		Timestamp: time.Now(),
	}

	for _, featureName := range req.FeatureNames {
		// Find feature by name
		var feature *Feature
		for _, f := range fs.features {
			if f.Name == featureName && f.Status == FeatureStatusActive {
				feature = f
				break
			}
		}

		if feature == nil {
			continue
		}

		// Get from cache
		if entityCache, exists := fs.onlineCache[feature.ID]; exists {
			if value, exists := entityCache[req.EntityID]; exists {
				response.Features[featureName] = value.Value
			}
		}

		// If not in cache, compute (in real implementation, this would call the feature computation)
		if _, exists := response.Features[featureName]; !exists {
			// Placeholder: return nil for missing features
			response.Features[featureName] = nil
		}
	}

	response.Latency = time.Since(startTime)

	return response, nil
}

// GetOfflineFeatures retrieves historical feature values for batch processing
func (fs *FeatureStore) GetOfflineFeatures(ctx context.Context, req *OfflineFeatureRequest) ([]map[string]interface{}, error) {
	fs.mu.RLock()
	defer fs.mu.RUnlock()

	results := make([]map[string]interface{}, 0, len(req.EntityIDs))

	for _, entityID := range req.EntityIDs {
		entityFeatures := map[string]interface{}{
			"entity_id": entityID,
			"timestamp": time.Now(),
		}

		for _, featureName := range req.FeatureNames {
			// Find feature
			var feature *Feature
			for _, f := range fs.features {
				if f.Name == featureName {
					feature = f
					break
				}
			}

			if feature == nil {
				continue
			}

			// In real implementation, query offline store (data warehouse, data lake)
			// For now, use online cache as fallback
			if entityCache, exists := fs.onlineCache[feature.ID]; exists {
				if value, exists := entityCache[entityID]; exists {
					if value.Timestamp.After(req.StartTime) && value.Timestamp.Before(req.EndTime) {
						entityFeatures[featureName] = value.Value
					}
				}
			}
		}

		results = append(results, entityFeatures)
	}

	return results, nil
}

// WriteOnlineFeatures writes feature values to online store
func (fs *FeatureStore) WriteOnlineFeatures(ctx context.Context, values []FeatureValue) error {
	fs.mu.Lock()
	defer fs.mu.Unlock()

	for _, value := range values {
		if _, exists := fs.onlineCache[value.FeatureID]; !exists {
			fs.onlineCache[value.FeatureID] = make(map[string]*FeatureValue)
		}

		valueCopy := value
		fs.onlineCache[value.FeatureID][value.EntityID] = &valueCopy
	}

	return nil
}

// CreateFeatureGroup creates a group of related features
func (fs *FeatureStore) CreateFeatureGroup(ctx context.Context, group *FeatureGroup) error {
	fs.mu.Lock()
	defer fs.mu.Unlock()

	if group.ID == "" {
		group.ID = uuid.New().String()
	}

	group.CreatedAt = time.Now()
	group.UpdatedAt = time.Now()

	// Validate all features exist
	for _, featureID := range group.Features {
		if _, exists := fs.features[featureID]; !exists {
			return fmt.Errorf("feature not found in group: %s", featureID)
		}
	}

	fs.featureGroups[group.ID] = group

	return nil
}

// GetFeatureGroup retrieves a feature group
func (fs *FeatureStore) GetFeatureGroup(ctx context.Context, groupID string) (*FeatureGroup, error) {
	fs.mu.RLock()
	defer fs.mu.RUnlock()

	group, exists := fs.featureGroups[groupID]
	if !exists {
		return nil, fmt.Errorf("feature group not found: %s", groupID)
	}

	return group, nil
}

// ComputeFeatureStatistics computes statistics for feature monitoring
func (fs *FeatureStore) ComputeFeatureStatistics(ctx context.Context, featureID string) (*FeatureStatistics, error) {
	fs.mu.RLock()
	defer fs.mu.RUnlock()

	feature, exists := fs.features[featureID]
	if !exists {
		return nil, fmt.Errorf("feature not found: %s", featureID)
	}

	// Get all values for this feature
	entityCache, exists := fs.onlineCache[featureID]
	if !exists || len(entityCache) == 0 {
		return nil, fmt.Errorf("no data available for feature: %s", featureID)
	}

	// Collect numeric values
	values := make([]float64, 0, len(entityCache))
	nullCount := int64(0)

	for _, featureValue := range entityCache {
		if featureValue.Value == nil {
			nullCount++
			continue
		}

		// Convert to float64 for statistics
		switch v := featureValue.Value.(type) {
		case float64:
			values = append(values, v)
		case int:
			values = append(values, float64(v))
		case int64:
			values = append(values, float64(v))
		}
	}

	if len(values) == 0 {
		return nil, fmt.Errorf("no numeric values for feature: %s", featureID)
	}

	stats := &FeatureStatistics{
		FeatureID: featureID,
		Timestamp: time.Now(),
		Count:     int64(len(entityCache)),
		NullCount: nullCount,
	}

	// Calculate statistics
	stats.Mean = calculateMean(values)
	stats.StdDev = calculateStdDev(values, stats.Mean)
	stats.Min = findMin(values)
	stats.Max = findMax(values)
	stats.Percentiles = calculatePercentiles(values)

	fs.statistics[featureID] = stats

	return stats, nil
}

// GetFeatureStatistics retrieves cached statistics
func (fs *FeatureStore) GetFeatureStatistics(ctx context.Context, featureID string) (*FeatureStatistics, error) {
	fs.mu.RLock()
	defer fs.mu.RUnlock()

	stats, exists := fs.statistics[featureID]
	if !exists {
		return nil, fmt.Errorf("statistics not found for feature: %s", featureID)
	}

	return stats, nil
}

// ListFeatures returns all features with optional filtering
func (fs *FeatureStore) ListFeatures(ctx context.Context, filter map[string]interface{}) ([]*Feature, error) {
	fs.mu.RLock()
	defer fs.mu.RUnlock()

	features := make([]*Feature, 0)

	for _, feature := range fs.features {
		if matchesFilter(feature, filter) {
			features = append(features, feature)
		}
	}

	return features, nil
}

// DeleteFeature removes a feature from the store
func (fs *FeatureStore) DeleteFeature(ctx context.Context, featureID string) error {
	fs.mu.Lock()
	defer fs.mu.Unlock()

	feature, exists := fs.features[featureID]
	if !exists {
		return fmt.Errorf("feature not found: %s", featureID)
	}

	// Prevent deletion of active features
	if feature.Status == FeatureStatusActive {
		return fmt.Errorf("cannot delete active feature: %s", featureID)
	}

	// Remove feature and its cache
	delete(fs.features, featureID)
	delete(fs.onlineCache, featureID)
	delete(fs.statistics, featureID)

	return nil
}

// ExportFeatures exports features to JSON
func (fs *FeatureStore) ExportFeatures(ctx context.Context, featureIDs []string) (string, error) {
	fs.mu.RLock()
	defer fs.mu.RUnlock()

	exportData := make([]*Feature, 0, len(featureIDs))

	for _, featureID := range featureIDs {
		if feature, exists := fs.features[featureID]; exists {
			exportData = append(exportData, feature)
		}
	}

	jsonData, err := json.MarshalIndent(exportData, "", "  ")
	if err != nil {
		return "", fmt.Errorf("failed to marshal features: %w", err)
	}

	return string(jsonData), nil
}

// ImportFeatures imports features from JSON
func (fs *FeatureStore) ImportFeatures(ctx context.Context, jsonData string) error {
	fs.mu.Lock()
	defer fs.mu.Unlock()

	var features []*Feature
	if err := json.Unmarshal([]byte(jsonData), &features); err != nil {
		return fmt.Errorf("failed to unmarshal features: %w", err)
	}

	for _, feature := range features {
		fs.features[feature.ID] = feature
		fs.onlineCache[feature.ID] = make(map[string]*FeatureValue)
	}

	return nil
}

// Helper functions

func matchesFilter(feature *Feature, filter map[string]interface{}) bool {
	if status, ok := filter["status"].(FeatureStatus); ok {
		if feature.Status != status {
			return false
		}
	}

	if entityType, ok := filter["entity_type"].(string); ok {
		if feature.EntityType != entityType {
			return false
		}
	}

	if owner, ok := filter["owner"].(string); ok {
		if feature.Owner != owner {
			return false
		}
	}

	return true
}

func calculateMean(values []float64) float64 {
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}

func calculateStdDev(values []float64, mean float64) float64 {
	sumSquares := 0.0
	for _, v := range values {
		diff := v - mean
		sumSquares += diff * diff
	}
	variance := sumSquares / float64(len(values))
	return variance
}

func findMin(values []float64) float64 {
	min := values[0]
	for _, v := range values[1:] {
		if v < min {
			min = v
		}
	}
	return min
}

func findMax(values []float64) float64 {
	max := values[0]
	for _, v := range values[1:] {
		if v > max {
			max = v
		}
	}
	return max
}

func calculatePercentiles(values []float64) map[string]float64 {
	// Sort values (simplified implementation)
	// In production, use proper sorting algorithm

	n := len(values)
	return map[string]float64{
		"p25": values[n/4],
		"p50": values[n/2],
		"p75": values[3*n/4],
		"p90": values[9*n/10],
		"p95": values[95*n/100],
		"p99": values[99*n/100],
	}
}

// GetFeatureLineage returns the lineage of a feature
func (fs *FeatureStore) GetFeatureLineage(ctx context.Context, featureID string) ([]string, error) {
	fs.mu.RLock()
	defer fs.mu.RUnlock()

	feature, exists := fs.features[featureID]
	if !exists {
		return nil, fmt.Errorf("feature not found: %s", featureID)
	}

	lineage := make([]string, 0)
	lineage = append(lineage, featureID)

	// Traverse parent features
	current := feature
	for current.ParentFeatureID != "" {
		lineage = append(lineage, current.ParentFeatureID)
		parent, exists := fs.features[current.ParentFeatureID]
		if !exists {
			break
		}
		current = parent
	}

	return lineage, nil
}

// GetDependentFeatures returns features that depend on this feature
func (fs *FeatureStore) GetDependentFeatures(ctx context.Context, featureID string) ([]*Feature, error) {
	fs.mu.RLock()
	defer fs.mu.RUnlock()

	dependents := make([]*Feature, 0)

	for _, feature := range fs.features {
		for _, depID := range feature.Dependencies {
			if depID == featureID {
				dependents = append(dependents, feature)
				break
			}
		}
	}

	return dependents, nil
}
