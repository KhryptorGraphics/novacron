package features

import (
	"fmt"
	"sync"
	"time"
)

// FeatureStore manages feature computation and storage
type FeatureStore struct {
	features map[string]*Feature
	versions map[string][]*FeatureVersion
	mu       sync.RWMutex
}

type Feature struct {
	Name        string
	Description string
	DataType    string
	CreatedAt   time.Time
}

type FeatureVersion struct {
	Version   string
	Values    map[string]interface{}
	CreatedAt time.Time
}

func NewFeatureStore() *FeatureStore {
	return &FeatureStore{
		features: make(map[string]*Feature),
		versions: make(map[string][]*FeatureVersion),
	}
}

func (fs *FeatureStore) RegisterFeature(name, description, dataType string) error {
	fs.mu.Lock()
	defer fs.mu.Unlock()

	fs.features[name] = &Feature{
		Name:        name,
		Description: description,
		DataType:    dataType,
		CreatedAt:   time.Now(),
	}
	fs.versions[name] = make([]*FeatureVersion, 0)
	return nil
}

func (fs *FeatureStore) GetFeature(name string, keys []string) ([]interface{}, error) {
	fs.mu.RLock()
	defer fs.mu.RUnlock()

	versions, exists := fs.versions[name]
	if !exists || len(versions) == 0 {
		return nil, fmt.Errorf("feature %s not found", name)
	}

	latest := versions[len(versions)-1]
	values := make([]interface{}, len(keys))
	for i, key := range keys {
		values[i] = latest.Values[key]
	}
	return values, nil
}
