package cache

import (
	"fmt"
	"testing"
	"time"
)

func TestHierarchicalCache_Basic(t *testing.T) {
	config := DefaultConfig()
	config.L1Size = 1024 * 1024 // 1MB
	config.L2Size = 5 * 1024 * 1024 // 5MB
	config.EnablePrefetch = false
	config.EnableDedup = false
	config.EnableCompression = false

	cache, err := NewHierarchicalCache(config)
	if err != nil {
		t.Fatalf("Failed to create cache: %v", err)
	}
	defer cache.Close()

	// Test Set and Get
	key := "test-key"
	value := []byte("test-value")

	err = cache.Set(key, value, 1*time.Hour)
	if err != nil {
		t.Fatalf("Failed to set value: %v", err)
	}

	retrieved, err := cache.Get(key)
	if err != nil {
		t.Fatalf("Failed to get value: %v", err)
	}

	if string(retrieved) != string(value) {
		t.Errorf("Expected %s, got %s", value, retrieved)
	}

	// Test cache hit
	stats := cache.Stats()
	if stats.HitRate == 0 {
		t.Errorf("Expected non-zero hit rate")
	}
}

func TestHierarchicalCache_Eviction(t *testing.T) {
	config := DefaultConfig()
	config.L1Size = 100 // 100 bytes
	config.L2Size = 0
	config.L3Size = 0
	config.EvictionPolicy = "lru"
	config.EnablePrefetch = false
	config.EnableDedup = false
	config.EnableCompression = false

	cache, err := NewHierarchicalCache(config)
	if err != nil {
		t.Fatalf("Failed to create cache: %v", err)
	}
	defer cache.Close()

	// Fill cache beyond capacity
	for i := 0; i < 10; i++ {
		key := fmt.Sprintf("key-%d", i)
		value := []byte(fmt.Sprintf("value-with-some-data-%d", i))
		cache.Set(key, value, 1*time.Hour)
	}

	// Older entries should be evicted
	_, err = cache.Get("key-0")
	if err != ErrNotFound {
		t.Errorf("Expected key-0 to be evicted")
	}

	// Recent entries should exist
	_, err = cache.Get("key-9")
	if err != nil {
		t.Errorf("Expected key-9 to exist: %v", err)
	}

	stats := cache.Stats()
	if stats.TotalEvictions == 0 {
		t.Errorf("Expected evictions to occur")
	}
}

func TestHierarchicalCache_Compression(t *testing.T) {
	config := DefaultConfig()
	config.L1Size = 1024 * 1024
	config.EnableCompression = true
	config.CompressionAlgo = "gzip"
	config.CompressionLevel = 6
	config.MinCompressionRatio = 1.1
	config.EnablePrefetch = false
	config.EnableDedup = false

	cache, err := NewHierarchicalCache(config)
	if err != nil {
		t.Fatalf("Failed to create cache: %v", err)
	}
	defer cache.Close()

	// Highly compressible data
	value := make([]byte, 1000)
	for i := range value {
		value[i] = 'A'
	}

	key := "compressed-key"
	err = cache.Set(key, value, 1*time.Hour)
	if err != nil {
		t.Fatalf("Failed to set value: %v", err)
	}

	retrieved, err := cache.Get(key)
	if err != nil {
		t.Fatalf("Failed to get value: %v", err)
	}

	if len(retrieved) != len(value) {
		t.Errorf("Expected %d bytes, got %d", len(value), len(retrieved))
	}

	// Verify data is correct
	for i := range retrieved {
		if retrieved[i] != 'A' {
			t.Errorf("Data corruption at byte %d", i)
			break
		}
	}
}

func TestHierarchicalCache_MultiTier(t *testing.T) {
	config := DefaultConfig()
	config.L1Size = 1024        // 1KB
	config.L2Size = 10 * 1024   // 10KB
	config.L3Size = 100 * 1024  // 100KB
	config.EnablePrefetch = false
	config.EnableDedup = false
	config.EnableCompression = false

	cache, err := NewHierarchicalCache(config)
	if err != nil {
		t.Fatalf("Failed to create cache: %v", err)
	}
	defer cache.Close()

	// Add to L1
	cache.Set("key1", []byte("value1"), 1*time.Hour)

	// Verify in L1
	if !cache.l1.Exists("key1") {
		t.Errorf("Expected key1 in L1")
	}

	// Fill L1 to trigger eviction to L2
	for i := 0; i < 100; i++ {
		key := fmt.Sprintf("filler-%d", i)
		cache.Set(key, []byte("some data here"), 1*time.Hour)
	}

	// Access key1 - should promote from L2 to L1
	cache.Get("key1")
}

func TestHierarchicalCache_Exists(t *testing.T) {
	config := DefaultConfig()
	config.L1Size = 1024 * 1024
	config.EnablePrefetch = false
	config.EnableDedup = false
	config.EnableCompression = false

	cache, err := NewHierarchicalCache(config)
	if err != nil {
		t.Fatalf("Failed to create cache: %v", err)
	}
	defer cache.Close()

	key := "test-key"
	if cache.Exists(key) {
		t.Errorf("Key should not exist")
	}

	cache.Set(key, []byte("value"), 1*time.Hour)

	if !cache.Exists(key) {
		t.Errorf("Key should exist")
	}

	cache.Delete(key)

	if cache.Exists(key) {
		t.Errorf("Key should not exist after deletion")
	}
}

func TestHierarchicalCache_Delete(t *testing.T) {
	config := DefaultConfig()
	config.L1Size = 1024 * 1024
	config.EnablePrefetch = false
	config.EnableDedup = false
	config.EnableCompression = false

	cache, err := NewHierarchicalCache(config)
	if err != nil {
		t.Fatalf("Failed to create cache: %v", err)
	}
	defer cache.Close()

	key := "test-key"
	cache.Set(key, []byte("value"), 1*time.Hour)

	err = cache.Delete(key)
	if err != nil {
		t.Fatalf("Failed to delete: %v", err)
	}

	_, err = cache.Get(key)
	if err != ErrNotFound {
		t.Errorf("Expected ErrNotFound after deletion")
	}
}

func TestHierarchicalCache_GetMulti(t *testing.T) {
	config := DefaultConfig()
	config.L1Size = 1024 * 1024
	config.EnablePrefetch = false
	config.EnableDedup = false
	config.EnableCompression = false

	cache, err := NewHierarchicalCache(config)
	if err != nil {
		t.Fatalf("Failed to create cache: %v", err)
	}
	defer cache.Close()

	// Set multiple keys
	cache.Set("key1", []byte("value1"), 1*time.Hour)
	cache.Set("key2", []byte("value2"), 1*time.Hour)
	cache.Set("key3", []byte("value3"), 1*time.Hour)

	// Get multiple
	result, err := cache.GetMulti([]string{"key1", "key2", "key3", "key4"})
	if err != nil {
		t.Fatalf("GetMulti failed: %v", err)
	}

	if len(result) != 3 {
		t.Errorf("Expected 3 results, got %d", len(result))
	}

	if string(result["key1"]) != "value1" {
		t.Errorf("Unexpected value for key1")
	}
}

func TestHierarchicalCache_SetMulti(t *testing.T) {
	config := DefaultConfig()
	config.L1Size = 1024 * 1024
	config.EnablePrefetch = false
	config.EnableDedup = false
	config.EnableCompression = false

	cache, err := NewHierarchicalCache(config)
	if err != nil {
		t.Fatalf("Failed to create cache: %v", err)
	}
	defer cache.Close()

	entries := map[string][]byte{
		"key1": []byte("value1"),
		"key2": []byte("value2"),
		"key3": []byte("value3"),
	}

	err = cache.SetMulti(entries, 1*time.Hour)
	if err != nil {
		t.Fatalf("SetMulti failed: %v", err)
	}

	// Verify all keys exist
	for key := range entries {
		if !cache.Exists(key) {
			t.Errorf("Key %s does not exist", key)
		}
	}
}

func TestHierarchicalCache_Stats(t *testing.T) {
	config := DefaultConfig()
	config.L1Size = 1024 * 1024
	config.EnablePrefetch = false
	config.EnableDedup = false
	config.EnableCompression = false

	cache, err := NewHierarchicalCache(config)
	if err != nil {
		t.Fatalf("Failed to create cache: %v", err)
	}
	defer cache.Close()

	// Generate some activity
	cache.Set("key1", []byte("value1"), 1*time.Hour)
	cache.Get("key1")  // Hit
	cache.Get("key2")  // Miss

	stats := cache.Stats()
	if stats.TotalAccesses != 2 {
		t.Errorf("Expected 2 accesses, got %d", stats.TotalAccesses)
	}

	if stats.TotalHits != 1 {
		t.Errorf("Expected 1 hit, got %d", stats.TotalHits)
	}

	if stats.TotalMisses != 1 {
		t.Errorf("Expected 1 miss, got %d", stats.TotalMisses)
	}
}

func BenchmarkCache_Set(b *testing.B) {
	config := DefaultConfig()
	config.L1Size = 100 * 1024 * 1024 // 100MB
	config.EnablePrefetch = false
	config.EnableDedup = false
	config.EnableCompression = false

	cache, _ := NewHierarchicalCache(config)
	defer cache.Close()

	value := make([]byte, 1024) // 1KB

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		key := fmt.Sprintf("key-%d", i)
		cache.Set(key, value, 1*time.Hour)
	}
}

func BenchmarkCache_Get(b *testing.B) {
	config := DefaultConfig()
	config.L1Size = 100 * 1024 * 1024
	config.EnablePrefetch = false
	config.EnableDedup = false
	config.EnableCompression = false

	cache, _ := NewHierarchicalCache(config)
	defer cache.Close()

	value := make([]byte, 1024)

	// Populate cache
	for i := 0; i < 1000; i++ {
		key := fmt.Sprintf("key-%d", i)
		cache.Set(key, value, 1*time.Hour)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		key := fmt.Sprintf("key-%d", i%1000)
		cache.Get(key)
	}
}

func BenchmarkCache_Compression(b *testing.B) {
	config := DefaultConfig()
	config.L1Size = 100 * 1024 * 1024
	config.EnableCompression = true
	config.CompressionAlgo = "gzip"
	config.EnablePrefetch = false
	config.EnableDedup = false

	cache, _ := NewHierarchicalCache(config)
	defer cache.Close()

	// Compressible data
	value := make([]byte, 10240)
	for i := range value {
		value[i] = byte(i % 256)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		key := fmt.Sprintf("key-%d", i)
		cache.Set(key, value, 1*time.Hour)
	}
}
