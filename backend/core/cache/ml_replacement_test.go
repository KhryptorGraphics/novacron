package cache

import (
	"testing"
	"time"
)

func TestMLCacheReplacer_PredictEvictionScore(t *testing.T) {
	config := DefaultConfig()
	ml := NewMLCacheReplacer(config)

	entry := &CacheEntry{
		Key:            "test",
		Size:           1024,
		AccessCount:    10,
		LastAccessedAt: time.Now().Add(-1 * time.Hour),
		Features:       []float64{3600, 10, 1024, 0, 12, 0, 1.0, 0},
	}

	score := ml.PredictEvictionScore(entry)
	if score < 0 || score > 1 {
		t.Errorf("Score should be between 0 and 1, got %f", score)
	}
}

func TestMLCacheReplacer_Learn(t *testing.T) {
	config := DefaultConfig()
	ml := NewMLCacheReplacer(config)

	entry := &CacheEntry{
		Key:         "test",
		Size:        1024,
		AccessCount: 10,
		Features:    []float64{3600, 10, 1024, 0, 12, 0, 1.0, 0},
	}

	// Learn from eviction
	err := ml.Learn(entry, true)
	if err != nil {
		t.Fatalf("Learn failed: %v", err)
	}

	// Learn from non-eviction
	err = ml.Learn(entry, false)
	if err != nil {
		t.Fatalf("Learn failed: %v", err)
	}

	// Check accuracy tracking
	accuracy := ml.Accuracy()
	if accuracy < 0 || accuracy > 1 {
		t.Errorf("Accuracy should be between 0 and 1, got %f", accuracy)
	}
}

func TestMLCacheReplacer_SaveLoad(t *testing.T) {
	config := DefaultConfig()
	ml1 := NewMLCacheReplacer(config)

	// Train the model
	for i := 0; i < 100; i++ {
		entry := &CacheEntry{
			Key:         "test",
			Features:    []float64{float64(i), 10, 1024, 0, 12, 0, 1.0, 0},
		}
		ml1.Learn(entry, i%2 == 0)
	}

	// Save model
	path := "/tmp/test_ml_model.json"
	err := ml1.SaveModel(path)
	if err != nil {
		t.Fatalf("SaveModel failed: %v", err)
	}

	// Load into new instance
	ml2 := NewMLCacheReplacer(config)
	err = ml2.LoadModel(path)
	if err != nil {
		t.Fatalf("LoadModel failed: %v", err)
	}

	// Compare weights
	if len(ml1.weights) != len(ml2.weights) {
		t.Errorf("Weight count mismatch")
	}

	for i := range ml1.weights {
		if ml1.weights[i] != ml2.weights[i] {
			t.Errorf("Weight %d mismatch: %f != %f", i, ml1.weights[i], ml2.weights[i])
		}
	}
}

func TestMLCacheReplacer_FindCandidates(t *testing.T) {
	config := DefaultConfig()
	ml := NewMLCacheReplacer(config)

	// Create test entries
	entries := make(map[string]*CacheEntry)
	for i := 0; i < 10; i++ {
		key := string(rune('a' + i))
		entries[key] = &CacheEntry{
			Key:            key,
			Size:           1024,
			AccessCount:    int64(i),
			LastAccessedAt: time.Now().Add(-time.Duration(i) * time.Hour),
			Features:       []float64{float64(i * 3600), float64(i), 1024, 0, 12, 0, 1.0, 0},
			Tier:           L1,
		}
	}

	// Find candidates
	candidates := ml.FindCandidatesInTier(entries, 5)

	if len(candidates) != 5 {
		t.Errorf("Expected 5 candidates, got %d", len(candidates))
	}

	// Verify candidates are sorted by score
	for i := 1; i < len(candidates); i++ {
		if candidates[i-1].Score < candidates[i].Score {
			t.Errorf("Candidates not properly sorted by score")
		}
	}
}

func TestMLCacheReplacer_BatchTrain(t *testing.T) {
	config := DefaultConfig()
	ml := NewMLCacheReplacer(config)

	// Add training examples
	for i := 0; i < 50; i++ {
		entry := &CacheEntry{
			Features: []float64{float64(i), 10, 1024, 0, 12, 0, 1.0, 0},
		}
		ml.Learn(entry, i%2 == 0)
	}

	// Batch train
	ml.BatchTrain(10)

	// Model should have learned something
	if ml.Accuracy() == 0 {
		t.Errorf("Model should have non-zero accuracy after training")
	}
}

func BenchmarkMLReplacer_Predict(b *testing.B) {
	config := DefaultConfig()
	ml := NewMLCacheReplacer(config)

	entry := &CacheEntry{
		Features: []float64{3600, 10, 1024, 0, 12, 0, 1.0, 0},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ml.PredictEvictionScore(entry)
	}
}

func BenchmarkMLReplacer_Learn(b *testing.B) {
	config := DefaultConfig()
	ml := NewMLCacheReplacer(config)

	entry := &CacheEntry{
		Features: []float64{3600, 10, 1024, 0, 12, 0, 1.0, 0},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ml.Learn(entry, i%2 == 0)
	}
}
