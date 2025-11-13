// Package knowledge provides knowledge graph integration
package knowledge

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/cognitive"
)

// KnowledgeGraph manages infrastructure knowledge
type KnowledgeGraph struct {
	config     *cognitive.CognitiveConfig
	driver     GraphDriver
	entities   map[string]*cognitive.KnowledgeEntity
	relations  map[string]*cognitive.KnowledgeRelation
	lock       sync.RWMutex
	cache      *GraphCache
	metricsLock sync.RWMutex
	metrics    GraphMetrics
}

// GraphDriver interface for graph database
type GraphDriver interface {
	Connect(ctx context.Context, url, user, pass string) error
	Close() error
	CreateEntity(ctx context.Context, entity *cognitive.KnowledgeEntity) error
	CreateRelation(ctx context.Context, relation *cognitive.KnowledgeRelation) error
	Query(ctx context.Context, query string) ([]map[string]interface{}, error)
	GetEntity(ctx context.Context, id string) (*cognitive.KnowledgeEntity, error)
	GetRelations(ctx context.Context, entityID string) ([]*cognitive.KnowledgeRelation, error)
}

// GraphCache caches frequently accessed entities
type GraphCache struct {
	entities  map[string]*cognitive.KnowledgeEntity
	relations map[string][]*cognitive.KnowledgeRelation
	lock      sync.RWMutex
	ttl       time.Duration
}

// GraphMetrics tracks knowledge graph performance
type GraphMetrics struct {
	TotalEntities   int64
	TotalRelations  int64
	QueriesExecuted int64
	CacheHits       int64
	CacheMisses     int64
	AvgQueryLatency float64
}

// NewKnowledgeGraph creates a new knowledge graph
func NewKnowledgeGraph(config *cognitive.CognitiveConfig, driver GraphDriver) *KnowledgeGraph {
	return &KnowledgeGraph{
		config:    config,
		driver:    driver,
		entities:  make(map[string]*cognitive.KnowledgeEntity),
		relations: make(map[string]*cognitive.KnowledgeRelation),
		cache:     NewGraphCache(config.CacheTTL),
	}
}

// NewGraphCache creates a new graph cache
func NewGraphCache(ttl time.Duration) *GraphCache {
	return &GraphCache{
		entities:  make(map[string]*cognitive.KnowledgeEntity),
		relations: make(map[string][]*cognitive.KnowledgeRelation),
		ttl:       ttl,
	}
}

// Connect connects to the graph database
func (kg *KnowledgeGraph) Connect(ctx context.Context) error {
	return kg.driver.Connect(ctx, kg.config.KnowledgeGraphURL, kg.config.KnowledgeGraphUser, kg.config.KnowledgeGraphPass)
}

// AddEntity adds an entity to the knowledge graph
func (kg *KnowledgeGraph) AddEntity(ctx context.Context, entity *cognitive.KnowledgeEntity) error {
	kg.lock.Lock()
	defer kg.lock.Unlock()

	entity.CreatedAt = time.Now()
	entity.UpdatedAt = time.Now()

	if err := kg.driver.CreateEntity(ctx, entity); err != nil {
		return fmt.Errorf("failed to create entity: %w", err)
	}

	kg.entities[entity.ID] = entity
	kg.cache.AddEntity(entity)

	kg.metricsLock.Lock()
	kg.metrics.TotalEntities++
	kg.metricsLock.Unlock()

	return nil
}

// AddRelation adds a relationship between entities
func (kg *KnowledgeGraph) AddRelation(ctx context.Context, relation *cognitive.KnowledgeRelation) error {
	kg.lock.Lock()
	defer kg.lock.Unlock()

	relation.CreatedAt = time.Now()

	if err := kg.driver.CreateRelation(ctx, relation); err != nil {
		return fmt.Errorf("failed to create relation: %w", err)
	}

	kg.relations[relation.ID] = relation

	kg.metricsLock.Lock()
	kg.metrics.TotalRelations++
	kg.metricsLock.Unlock()

	return nil
}

// GetEntity retrieves an entity by ID
func (kg *KnowledgeGraph) GetEntity(ctx context.Context, id string) (*cognitive.KnowledgeEntity, error) {
	// Check cache first
	if entity := kg.cache.GetEntity(id); entity != nil {
		kg.recordCacheHit()
		return entity, nil
	}

	kg.recordCacheMiss()

	// Query database
	entity, err := kg.driver.GetEntity(ctx, id)
	if err != nil {
		return nil, fmt.Errorf("failed to get entity: %w", err)
	}

	kg.cache.AddEntity(entity)
	return entity, nil
}

// GetRelatedEntities finds entities related to the given entity
func (kg *KnowledgeGraph) GetRelatedEntities(ctx context.Context, entityID string, relationType string) ([]*cognitive.KnowledgeEntity, error) {
	// Get relations from cache or database
	relations, err := kg.driver.GetRelations(ctx, entityID)
	if err != nil {
		return nil, fmt.Errorf("failed to get relations: %w", err)
	}

	var relatedEntities []*cognitive.KnowledgeEntity
	for _, rel := range relations {
		if relationType != "" && rel.Type != relationType {
			continue
		}

		// Get the related entity
		relatedID := rel.To
		if rel.From == entityID {
			relatedID = rel.To
		} else {
			relatedID = rel.From
		}

		entity, err := kg.GetEntity(ctx, relatedID)
		if err != nil {
			continue // Skip if entity not found
		}

		relatedEntities = append(relatedEntities, entity)
	}

	return relatedEntities, nil
}

// QueryByType finds entities of a specific type
func (kg *KnowledgeGraph) QueryByType(ctx context.Context, entityType string) ([]*cognitive.KnowledgeEntity, error) {
	startTime := time.Now()

	query := fmt.Sprintf(`MATCH (n {type: "%s"}) RETURN n`, entityType)
	results, err := kg.driver.Query(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("query failed: %w", err)
	}

	var entities []*cognitive.KnowledgeEntity
	for _, result := range results {
		if entityData, ok := result["n"].(map[string]interface{}); ok {
			entity := kg.mapToEntity(entityData)
			entities = append(entities, entity)
		}
	}

	kg.recordQuery(time.Since(startTime))
	return entities, nil
}

// FindDependencies finds all dependencies of an entity
func (kg *KnowledgeGraph) FindDependencies(ctx context.Context, entityID string) ([]*cognitive.KnowledgeEntity, error) {
	return kg.GetRelatedEntities(ctx, entityID, "Depends-On")
}

// FindDependents finds all entities that depend on this entity
func (kg *KnowledgeGraph) FindDependents(ctx context.Context, entityID string) ([]*cognitive.KnowledgeEntity, error) {
	relations, err := kg.driver.GetRelations(ctx, entityID)
	if err != nil {
		return nil, err
	}

	var dependents []*cognitive.KnowledgeEntity
	for _, rel := range relations {
		if rel.Type == "Depends-On" && rel.To == entityID {
			entity, err := kg.GetEntity(ctx, rel.From)
			if err == nil {
				dependents = append(dependents, entity)
			}
		}
	}

	return dependents, nil
}

// AddBestPractice stores a best practice
func (kg *KnowledgeGraph) AddBestPractice(ctx context.Context, domain, title, description string) error {
	entity := &cognitive.KnowledgeEntity{
		ID:   fmt.Sprintf("bp-%d", time.Now().UnixNano()),
		Type: "BestPractice",
		Properties: map[string]interface{}{
			"domain":      domain,
			"title":       title,
			"description": description,
		},
	}

	return kg.AddEntity(ctx, entity)
}

// GetBestPractices retrieves best practices for a domain
func (kg *KnowledgeGraph) GetBestPractices(ctx context.Context, domain string) ([]*cognitive.KnowledgeEntity, error) {
	query := fmt.Sprintf(`MATCH (n {type: "BestPractice", domain: "%s"}) RETURN n`, domain)
	results, err := kg.driver.Query(ctx, query)
	if err != nil {
		return nil, err
	}

	var practices []*cognitive.KnowledgeEntity
	for _, result := range results {
		if entityData, ok := result["n"].(map[string]interface{}); ok {
			entity := kg.mapToEntity(entityData)
			practices = append(practices, entity)
		}
	}

	return practices, nil
}

// RecordIncident stores an incident for learning
func (kg *KnowledgeGraph) RecordIncident(ctx context.Context, title, description, rootCause, resolution string, affectedEntities []string) error {
	incident := &cognitive.KnowledgeEntity{
		ID:   fmt.Sprintf("incident-%d", time.Now().UnixNano()),
		Type: "Incident",
		Properties: map[string]interface{}{
			"title":       title,
			"description": description,
			"root_cause":  rootCause,
			"resolution":  resolution,
			"timestamp":   time.Now(),
		},
	}

	if err := kg.AddEntity(ctx, incident); err != nil {
		return err
	}

	// Link to affected entities
	for _, entityID := range affectedEntities {
		relation := &cognitive.KnowledgeRelation{
			ID:   fmt.Sprintf("rel-%d", time.Now().UnixNano()),
			Type: "AffectedBy",
			From: entityID,
			To:   incident.ID,
			Properties: map[string]interface{}{
				"timestamp": time.Now(),
			},
		}

		if err := kg.AddRelation(ctx, relation); err != nil {
			return err
		}
	}

	return nil
}

// GetIncidentHistory retrieves incident history for an entity
func (kg *KnowledgeGraph) GetIncidentHistory(ctx context.Context, entityID string) ([]*cognitive.KnowledgeEntity, error) {
	return kg.GetRelatedEntities(ctx, entityID, "AffectedBy")
}

// mapToEntity converts map to entity
func (kg *KnowledgeGraph) mapToEntity(data map[string]interface{}) *cognitive.KnowledgeEntity {
	entity := &cognitive.KnowledgeEntity{
		Properties: make(map[string]interface{}),
	}

	if id, ok := data["id"].(string); ok {
		entity.ID = id
	}
	if typ, ok := data["type"].(string); ok {
		entity.Type = typ
	}

	for k, v := range data {
		if k != "id" && k != "type" {
			entity.Properties[k] = v
		}
	}

	return entity
}

// recordCacheHit records a cache hit
func (kg *KnowledgeGraph) recordCacheHit() {
	kg.metricsLock.Lock()
	defer kg.metricsLock.Unlock()
	kg.metrics.CacheHits++
}

// recordCacheMiss records a cache miss
func (kg *KnowledgeGraph) recordCacheMiss() {
	kg.metricsLock.Lock()
	defer kg.metricsLock.Unlock()
	kg.metrics.CacheMisses++
}

// recordQuery records query execution
func (kg *KnowledgeGraph) recordQuery(latency time.Duration) {
	kg.metricsLock.Lock()
	defer kg.metricsLock.Unlock()

	kg.metrics.QueriesExecuted++
	alpha := 0.1
	kg.metrics.AvgQueryLatency = alpha*float64(latency.Milliseconds()) + (1-alpha)*kg.metrics.AvgQueryLatency
}

// GetMetrics returns knowledge graph metrics
func (kg *KnowledgeGraph) GetMetrics() GraphMetrics {
	kg.metricsLock.RLock()
	defer kg.metricsLock.RUnlock()
	return kg.metrics
}

// AddEntity adds entity to cache
func (gc *GraphCache) AddEntity(entity *cognitive.KnowledgeEntity) {
	gc.lock.Lock()
	defer gc.lock.Unlock()
	gc.entities[entity.ID] = entity
}

// GetEntity retrieves entity from cache
func (gc *GraphCache) GetEntity(id string) *cognitive.KnowledgeEntity {
	gc.lock.RLock()
	defer gc.lock.RUnlock()
	return gc.entities[id]
}

// MockGraphDriver is a mock implementation for testing
type MockGraphDriver struct {
	entities  map[string]*cognitive.KnowledgeEntity
	relations map[string]*cognitive.KnowledgeRelation
	lock      sync.RWMutex
}

// NewMockGraphDriver creates a mock driver
func NewMockGraphDriver() *MockGraphDriver {
	return &MockGraphDriver{
		entities:  make(map[string]*cognitive.KnowledgeEntity),
		relations: make(map[string]*cognitive.KnowledgeRelation),
	}
}

// Connect mock implementation
func (m *MockGraphDriver) Connect(ctx context.Context, url, user, pass string) error {
	return nil
}

// Close mock implementation
func (m *MockGraphDriver) Close() error {
	return nil
}

// CreateEntity mock implementation
func (m *MockGraphDriver) CreateEntity(ctx context.Context, entity *cognitive.KnowledgeEntity) error {
	m.lock.Lock()
	defer m.lock.Unlock()
	m.entities[entity.ID] = entity
	return nil
}

// CreateRelation mock implementation
func (m *MockGraphDriver) CreateRelation(ctx context.Context, relation *cognitive.KnowledgeRelation) error {
	m.lock.Lock()
	defer m.lock.Unlock()
	m.relations[relation.ID] = relation
	return nil
}

// Query mock implementation
func (m *MockGraphDriver) Query(ctx context.Context, query string) ([]map[string]interface{}, error) {
	return []map[string]interface{}{}, nil
}

// GetEntity mock implementation
func (m *MockGraphDriver) GetEntity(ctx context.Context, id string) (*cognitive.KnowledgeEntity, error) {
	m.lock.RLock()
	defer m.lock.RUnlock()

	entity, exists := m.entities[id]
	if !exists {
		return nil, fmt.Errorf("entity not found: %s", id)
	}
	return entity, nil
}

// GetRelations mock implementation
func (m *MockGraphDriver) GetRelations(ctx context.Context, entityID string) ([]*cognitive.KnowledgeRelation, error) {
	m.lock.RLock()
	defer m.lock.RUnlock()

	var relations []*cognitive.KnowledgeRelation
	for _, rel := range m.relations {
		if rel.From == entityID || rel.To == entityID {
			relations = append(relations, rel)
		}
	}
	return relations, nil
}
