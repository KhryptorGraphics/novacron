package cache

import (
	"encoding/json"
	"math"
	"os"
	"sort"
	"sync"
	"time"
)

// MLCacheReplacerImpl implements ML-based cache replacement
type MLCacheReplacerImpl struct {
	config *CacheConfig

	// Model weights and parameters
	weights     []float64
	bias        float64
	learningRate float64

	// Training data
	trainingData []*TrainingExample
	maxTrainingData int

	// Accuracy tracking
	predictions     int64
	correctPredictions int64

	// LSTM state for sequence prediction
	lstmHidden  []float64
	lstmCell    []float64

	mu sync.RWMutex
}

// TrainingExample represents a training example
type TrainingExample struct {
	Features []float64
	WasEvicted bool
	Timestamp time.Time
}

// NewMLCacheReplacer creates a new ML cache replacer
func NewMLCacheReplacer(config *CacheConfig) *MLCacheReplacerImpl {
	return &MLCacheReplacerImpl{
		config:          config,
		weights:         make([]float64, config.MLFeatureCount),
		learningRate:    config.MLLearningRate,
		trainingData:    make([]*TrainingExample, 0, 10000),
		maxTrainingData: 10000,
		lstmHidden:      make([]float64, 64),
		lstmCell:        make([]float64, 64),
	}
}

// PredictEvictionScore predicts the eviction score for an entry
func (ml *MLCacheReplacerImpl) PredictEvictionScore(entry *CacheEntry) float64 {
	ml.mu.RLock()
	defer ml.mu.RUnlock()

	if len(entry.Features) != len(ml.weights) {
		return 0.0
	}

	// Linear combination of features
	score := ml.bias
	for i, feature := range entry.Features {
		score += feature * ml.weights[i]
	}

	// Apply sigmoid activation
	return sigmoid(score)
}

// FindEvictionCandidates finds the best candidates for eviction
func (ml *MLCacheReplacerImpl) FindEvictionCandidates(tier CacheTier, count int) ([]*EvictionCandidate, error) {
	// This would need access to the cache tier's entries
	// For now, return empty slice
	return []*EvictionCandidate{}, nil
}

// Learn updates the model based on feedback
func (ml *MLCacheReplacerImpl) Learn(entry *CacheEntry, wasEvicted bool) error {
	ml.mu.Lock()
	defer ml.mu.Unlock()

	// Add to training data
	example := &TrainingExample{
		Features:   append([]float64{}, entry.Features...),
		WasEvicted: wasEvicted,
		Timestamp:  time.Now(),
	}

	ml.trainingData = append(ml.trainingData, example)

	// Limit training data size
	if len(ml.trainingData) > ml.maxTrainingData {
		ml.trainingData = ml.trainingData[len(ml.trainingData)-ml.maxTrainingData:]
	}

	// Online learning - update weights immediately
	if ml.config.EnableOnline {
		ml.updateWeights(example)
	}

	// Track prediction accuracy
	predicted := ml.predictWithWeights(example.Features, ml.weights, ml.bias)
	ml.predictions++

	predictedEvicted := predicted > 0.5
	if predictedEvicted == wasEvicted {
		ml.correctPredictions++
	}

	return nil
}

// updateWeights updates model weights using gradient descent
func (ml *MLCacheReplacerImpl) updateWeights(example *TrainingExample) {
	// Compute prediction
	predicted := ml.predictWithWeights(example.Features, ml.weights, ml.bias)

	// Compute error
	target := 0.0
	if example.WasEvicted {
		target = 1.0
	}
	error := target - predicted

	// Update weights using gradient descent
	for i := range ml.weights {
		gradient := error * example.Features[i]
		ml.weights[i] += ml.learningRate * gradient
	}

	// Update bias
	ml.bias += ml.learningRate * error
}

// predictWithWeights makes a prediction with given weights
func (ml *MLCacheReplacerImpl) predictWithWeights(features []float64, weights []float64, bias float64) float64 {
	score := bias
	for i, feature := range features {
		if i < len(weights) {
			score += feature * weights[i]
		}
	}
	return sigmoid(score)
}

// SaveModel saves the model to disk
func (ml *MLCacheReplacerImpl) SaveModel(path string) error {
	ml.mu.RLock()
	defer ml.mu.RUnlock()

	model := struct {
		Weights     []float64 `json:"weights"`
		Bias        float64   `json:"bias"`
		Predictions int64     `json:"predictions"`
		Correct     int64     `json:"correct"`
	}{
		Weights:     ml.weights,
		Bias:        ml.bias,
		Predictions: ml.predictions,
		Correct:     ml.correctPredictions,
	}

	data, err := json.Marshal(model)
	if err != nil {
		return err
	}

	return os.WriteFile(path, data, 0644)
}

// LoadModel loads the model from disk
func (ml *MLCacheReplacerImpl) LoadModel(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}

	var model struct {
		Weights     []float64 `json:"weights"`
		Bias        float64   `json:"bias"`
		Predictions int64     `json:"predictions"`
		Correct     int64     `json:"correct"`
	}

	if err := json.Unmarshal(data, &model); err != nil {
		return err
	}

	ml.mu.Lock()
	defer ml.mu.Unlock()

	ml.weights = model.Weights
	ml.bias = model.Bias
	ml.predictions = model.Predictions
	ml.correctPredictions = model.Correct

	return nil
}

// Accuracy returns the model's accuracy
func (ml *MLCacheReplacerImpl) Accuracy() float64 {
	ml.mu.RLock()
	defer ml.mu.RUnlock()

	if ml.predictions == 0 {
		return 0.0
	}

	return float64(ml.correctPredictions) / float64(ml.predictions)
}

// FindCandidatesInTier finds eviction candidates in a specific tier
func (ml *MLCacheReplacerImpl) FindCandidatesInTier(entries map[string]*CacheEntry, count int) []*EvictionCandidate {
	candidates := make([]*EvictionCandidate, 0, len(entries))

	for _, entry := range entries {
		score := ml.PredictEvictionScore(entry)

		candidates = append(candidates, &EvictionCandidate{
			Key:          entry.Key,
			Tier:         entry.Tier,
			Score:        score,
			Size:         entry.Size,
			LastAccessed: entry.LastAccessedAt,
			AccessCount:  entry.AccessCount,
			Features:     entry.Features,
		})
	}

	// Sort by score (higher score = more likely to evict)
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].Score > candidates[j].Score
	})

	// Return top N candidates
	if count < len(candidates) {
		return candidates[:count]
	}
	return candidates
}

// sigmoid activation function
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// BatchTrain performs batch training on accumulated examples
func (ml *MLCacheReplacerImpl) BatchTrain(epochs int) {
	ml.mu.Lock()
	defer ml.mu.Unlock()

	for epoch := 0; epoch < epochs; epoch++ {
		for _, example := range ml.trainingData {
			ml.updateWeights(example)
		}
	}
}
