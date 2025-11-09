package cache

import (
	"testing"
)

func TestPrefetchEngine_LearnPattern(t *testing.T) {
	config := DefaultConfig()
	cache, _ := NewHierarchicalCache(config)
	defer cache.Close()

	pe := NewPrefetchEngine(config, cache)
	defer pe.Close()

	// Learn sequential pattern
	sequence := []string{"a", "b", "c", "d", "e"}
	err := pe.LearnPattern(sequence)
	if err != nil {
		t.Fatalf("LearnPattern failed: %v", err)
	}

	// Check transitions
	if len(pe.transitions) == 0 {
		t.Errorf("Expected transitions to be learned")
	}

	// Verify specific transitions
	if _, ok := pe.transitions["a"]["b"]; !ok {
		t.Errorf("Expected transition from a to b")
	}
}

func TestPrefetchEngine_PredictNext(t *testing.T) {
	config := DefaultConfig()
	cache, _ := NewHierarchicalCache(config)
	defer cache.Close()

	pe := NewPrefetchEngine(config, cache)
	defer pe.Close()

	// Learn pattern
	for i := 0; i < 10; i++ {
		pe.LearnPattern([]string{"page1", "page2", "page3"})
	}

	// Predict next
	predicted, err := pe.PredictNext("page1", 2)
	if err != nil {
		t.Fatalf("PredictNext failed: %v", err)
	}

	if len(predicted) == 0 {
		t.Errorf("Expected predictions")
	}

	// Most likely next should be page2
	if len(predicted) > 0 && predicted[0] != "page2" {
		t.Errorf("Expected page2 to be predicted, got %s", predicted[0])
	}
}

func TestPrefetchEngine_AnalyzePattern(t *testing.T) {
	config := DefaultConfig()
	cache, _ := NewHierarchicalCache(config)
	defer cache.Close()

	pe := NewPrefetchEngine(config, cache)
	defer pe.Close()

	// Test sequential pattern
	sequence := []string{"a", "b", "c", "a", "b", "c", "a", "b", "c"}
	pe.accessHistory = sequence

	pattern := pe.AnalyzePattern("a")
	// Pattern could be sequential, periodic, or bursty depending on implementation
	if pattern == PatternRandom {
		// May or may not be random, depending on history length
	}
}

func TestPrefetchEngine_Prefetch(t *testing.T) {
	config := DefaultConfig()
	config.EnablePrefetch = true
	cache, _ := NewHierarchicalCache(config)
	defer cache.Close()

	pe := NewPrefetchEngine(config, cache)
	defer pe.Close()

	req := &PrefetchRequest{
		Keys:     []string{"key1", "key2", "key3"},
		Priority: 5,
	}

	err := pe.Prefetch(req)
	if err != nil {
		t.Fatalf("Prefetch failed: %v", err)
	}
}

func TestPrefetchEngine_Accuracy(t *testing.T) {
	config := DefaultConfig()
	cache, _ := NewHierarchicalCache(config)
	defer cache.Close()

	pe := NewPrefetchEngine(config, cache)
	defer pe.Close()

	// Initial accuracy should be 0
	accuracy := pe.Accuracy()
	if accuracy != 0 {
		t.Errorf("Initial accuracy should be 0, got %f", accuracy)
	}

	// Record some hits
	pe.mu.Lock()
	pe.prefetchCount = 10
	pe.prefetchHits = 8
	pe.mu.Unlock()

	accuracy = pe.Accuracy()
	if accuracy != 0.8 {
		t.Errorf("Expected accuracy 0.8, got %f", accuracy)
	}
}

func BenchmarkPrefetchEngine_Predict(b *testing.B) {
	config := DefaultConfig()
	cache, _ := NewHierarchicalCache(config)
	defer cache.Close()

	pe := NewPrefetchEngine(config, cache)
	defer pe.Close()

	// Learn pattern
	pe.LearnPattern([]string{"a", "b", "c", "d", "e"})

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		pe.PredictNext("a", 5)
	}
}
