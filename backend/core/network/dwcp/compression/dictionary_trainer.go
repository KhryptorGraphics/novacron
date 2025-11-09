package compression

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/klauspost/compress/zstd"
	"go.uber.org/zap"
)

// DictionaryTrainer manages Zstandard dictionary training for improved compression
type DictionaryTrainer struct {
	// Dictionary storage
	dictionaries     map[string]*TrainedDictionary
	dictionariesPath string
	mu               sync.RWMutex

	// Training configuration
	config *DictionaryTrainingConfig
	logger *zap.Logger

	// Sample collection
	samples      map[string][][]byte // resourceType -> samples
	samplesMutex sync.Mutex
	maxSamples   int

	// Training scheduler
	lastTrained time.Time
	trainTicker *time.Ticker
	stopChan    chan struct{}
}

// TrainedDictionary represents a trained Zstandard dictionary
type TrainedDictionary struct {
	ResourceType string    `json:"resource_type"`
	Dictionary   []byte    `json:"dictionary"`
	SampleCount  int       `json:"sample_count"`
	TrainedAt    time.Time `json:"trained_at"`
	Version      int       `json:"version"`
}

// DictionaryTrainingConfig configuration for dictionary training
type DictionaryTrainingConfig struct {
	Enabled         bool          `json:"enabled" yaml:"enabled"`
	UpdateInterval  time.Duration `json:"update_interval" yaml:"update_interval"`
	MaxSamples      int           `json:"max_samples" yaml:"max_samples"`
	MinSampleSize   int           `json:"min_sample_size" yaml:"min_sample_size"`
	MaxDictSize     int           `json:"max_dict_size" yaml:"max_dict_size"`
	StoragePath     string        `json:"storage_path" yaml:"storage_path"`
}

// DefaultDictionaryTrainingConfig returns sensible defaults
func DefaultDictionaryTrainingConfig() *DictionaryTrainingConfig {
	return &DictionaryTrainingConfig{
		Enabled:        true,
		UpdateInterval: 24 * time.Hour,
		MaxSamples:     1000,
		MinSampleSize:  1024,       // 1 KB minimum
		MaxDictSize:    128 * 1024, // 128 KB max dictionary
		StoragePath:    "./compression/dictionaries",
	}
}

// NewDictionaryTrainer creates a new dictionary trainer
func NewDictionaryTrainer(config *DictionaryTrainingConfig, logger *zap.Logger) (*DictionaryTrainer, error) {
	if config == nil {
		config = DefaultDictionaryTrainingConfig()
	}

	if logger == nil {
		logger, _ = zap.NewProduction()
	}

	// Create storage directory
	if err := os.MkdirAll(config.StoragePath, 0755); err != nil {
		return nil, fmt.Errorf("failed to create dictionary storage: %w", err)
	}

	dt := &DictionaryTrainer{
		dictionaries:     make(map[string]*TrainedDictionary),
		dictionariesPath: config.StoragePath,
		config:           config,
		logger:           logger,
		samples:          make(map[string][][]byte),
		maxSamples:       config.MaxSamples,
		stopChan:         make(chan struct{}),
	}

	// Load existing dictionaries
	if err := dt.loadDictionaries(); err != nil {
		logger.Warn("Failed to load existing dictionaries", zap.Error(err))
	}

	// Start auto-training scheduler if enabled
	if config.Enabled && config.UpdateInterval > 0 {
		dt.startTrainingScheduler()
	}

	return dt, nil
}

// AddSample collects a sample for dictionary training
func (dt *DictionaryTrainer) AddSample(resourceType string, data []byte) {
	if !dt.config.Enabled {
		return
	}

	// Skip small samples
	if len(data) < dt.config.MinSampleSize {
		return
	}

	dt.samplesMutex.Lock()
	defer dt.samplesMutex.Unlock()

	if dt.samples[resourceType] == nil {
		dt.samples[resourceType] = make([][]byte, 0, dt.maxSamples)
	}

	// Add sample (make a copy to avoid data races)
	sample := make([]byte, len(data))
	copy(sample, data)
	dt.samples[resourceType] = append(dt.samples[resourceType], sample)

	// Limit sample count (FIFO)
	if len(dt.samples[resourceType]) > dt.maxSamples {
		dt.samples[resourceType] = dt.samples[resourceType][1:]
	}
}

// TrainDictionary trains a dictionary for a specific resource type
func (dt *DictionaryTrainer) TrainDictionary(resourceType string) error {
	dt.samplesMutex.Lock()
	samples, hasSamples := dt.samples[resourceType]
	dt.samplesMutex.Unlock()

	if !hasSamples || len(samples) == 0 {
		return fmt.Errorf("no samples available for resource type: %s", resourceType)
	}

	dt.logger.Info("Training dictionary",
		zap.String("resource_type", resourceType),
		zap.Int("sample_count", len(samples)))

	// Train dictionary using Zstandard
	dict, err := zstd.BuildDict(zstd.BuildDictOptions{
		Contents: samples,
	})

	if err != nil {
		return fmt.Errorf("dictionary training failed: %w", err)
	}

	// Store the trained dictionary
	trainedDict := &TrainedDictionary{
		ResourceType: resourceType,
		Dictionary:   dict,
		SampleCount:  len(samples),
		TrainedAt:    time.Now(),
		Version:      dt.getNextVersion(resourceType),
	}

	dt.mu.Lock()
	dt.dictionaries[resourceType] = trainedDict
	dt.mu.Unlock()

	// Persist to disk
	if err := dt.saveDictionary(trainedDict); err != nil {
		dt.logger.Warn("Failed to save dictionary", zap.Error(err))
	}

	dt.logger.Info("Dictionary trained successfully",
		zap.String("resource_type", resourceType),
		zap.Int("dict_size", len(dict)),
		zap.Int("samples", len(samples)))

	return nil
}

// GetDictionary retrieves a trained dictionary for a resource type
func (dt *DictionaryTrainer) GetDictionary(resourceType string) ([]byte, bool) {
	dt.mu.RLock()
	defer dt.mu.RUnlock()

	dict, exists := dt.dictionaries[resourceType]
	if !exists {
		return nil, false
	}

	return dict.Dictionary, true
}

// TrainAllDictionaries trains dictionaries for all resource types with samples
func (dt *DictionaryTrainer) TrainAllDictionaries() error {
	dt.samplesMutex.Lock()
	resourceTypes := make([]string, 0, len(dt.samples))
	for rt := range dt.samples {
		resourceTypes = append(resourceTypes, rt)
	}
	dt.samplesMutex.Unlock()

	var errors []error
	for _, rt := range resourceTypes {
		if err := dt.TrainDictionary(rt); err != nil {
			errors = append(errors, err)
			dt.logger.Error("Dictionary training failed",
				zap.String("resource_type", rt),
				zap.Error(err))
		}
	}

	if len(errors) > 0 {
		return fmt.Errorf("failed to train %d dictionaries", len(errors))
	}

	dt.lastTrained = time.Now()
	return nil
}

// saveDictionary persists a dictionary to disk
func (dt *DictionaryTrainer) saveDictionary(dict *TrainedDictionary) error {
	filename := filepath.Join(dt.dictionariesPath, fmt.Sprintf("%s_v%d.dict.json", dict.ResourceType, dict.Version))

	data, err := json.MarshalIndent(dict, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal dictionary: %w", err)
	}

	if err := os.WriteFile(filename, data, 0644); err != nil {
		return fmt.Errorf("failed to write dictionary file: %w", err)
	}

	return nil
}

// loadDictionaries loads all dictionaries from disk
func (dt *DictionaryTrainer) loadDictionaries() error {
	files, err := os.ReadDir(dt.dictionariesPath)
	if err != nil {
		return err
	}

	for _, file := range files {
		if filepath.Ext(file.Name()) != ".json" {
			continue
		}

		path := filepath.Join(dt.dictionariesPath, file.Name())
		data, err := os.ReadFile(path)
		if err != nil {
			dt.logger.Warn("Failed to read dictionary file",
				zap.String("file", file.Name()),
				zap.Error(err))
			continue
		}

		var dict TrainedDictionary
		if err := json.Unmarshal(data, &dict); err != nil {
			dt.logger.Warn("Failed to unmarshal dictionary",
				zap.String("file", file.Name()),
				zap.Error(err))
			continue
		}

		dt.mu.Lock()
		// Keep only the latest version for each resource type
		existing, exists := dt.dictionaries[dict.ResourceType]
		if !exists || dict.Version > existing.Version {
			dt.dictionaries[dict.ResourceType] = &dict
		}
		dt.mu.Unlock()

		dt.logger.Info("Loaded dictionary",
			zap.String("resource_type", dict.ResourceType),
			zap.Int("version", dict.Version),
			zap.Time("trained_at", dict.TrainedAt))
	}

	return nil
}

// getNextVersion returns the next version number for a resource type
func (dt *DictionaryTrainer) getNextVersion(resourceType string) int {
	dt.mu.RLock()
	defer dt.mu.RUnlock()

	if existing, exists := dt.dictionaries[resourceType]; exists {
		return existing.Version + 1
	}
	return 1
}

// startTrainingScheduler starts the automatic training scheduler
func (dt *DictionaryTrainer) startTrainingScheduler() {
	dt.trainTicker = time.NewTicker(dt.config.UpdateInterval)

	go func() {
		for {
			select {
			case <-dt.trainTicker.C:
				dt.logger.Info("Starting scheduled dictionary training")
				if err := dt.TrainAllDictionaries(); err != nil {
					dt.logger.Error("Scheduled training failed", zap.Error(err))
				}

			case <-dt.stopChan:
				dt.trainTicker.Stop()
				return
			}
		}
	}()

	dt.logger.Info("Dictionary training scheduler started",
		zap.Duration("interval", dt.config.UpdateInterval))
}

// GetStats returns dictionary statistics
func (dt *DictionaryTrainer) GetStats() map[string]interface{} {
	dt.mu.RLock()
	dictCount := len(dt.dictionaries)
	dt.mu.RUnlock()

	dt.samplesMutex.Lock()
	totalSamples := 0
	for _, samples := range dt.samples {
		totalSamples += len(samples)
	}
	resourceTypeCount := len(dt.samples)
	dt.samplesMutex.Unlock()

	return map[string]interface{}{
		"dictionary_count":    dictCount,
		"total_samples":       totalSamples,
		"resource_type_count": resourceTypeCount,
		"last_trained":        dt.lastTrained,
	}
}

// Close stops the training scheduler and releases resources
func (dt *DictionaryTrainer) Close() error {
	if dt.trainTicker != nil {
		close(dt.stopChan)
	}
	return nil
}
