package cache

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/sirupsen/logrus"
)

// TestMultiTierCache tests the multi-tier cache functionality
func TestMultiTierCache(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.WarnLevel) // Reduce noise in tests

	// Create cache with memory-only configuration for testing
	config := &CacheConfig{
		L1Enabled:    true,
		L1MaxSize:    100,
		L1TTL:        1 * time.Minute,
		L1CleanupInt: 10 * time.Second,

		L2Enabled: false, // Disable Redis for unit tests
		L3Enabled: false, // Disable persistent cache for unit tests

		DefaultTTL:        5 * time.Minute,
		EnableMetrics:     true,
		MetricsInterval:   1 * time.Second, // Required for metrics collection
	}

	cache, err := NewMultiTierCache(config, logger)
	if err != nil {
		t.Fatalf("Failed to create cache: %v", err)
	}
	defer cache.Close()

	ctx := context.Background()

	// Test basic operations
	t.Run("BasicOperations", func(t *testing.T) {
		testBasicOperations(t, cache, ctx)
	})

	// Test TTL functionality
	t.Run("TTLFunctionality", func(t *testing.T) {
		testTTLFunctionality(t, cache, ctx)
	})

	// Test batch operations
	t.Run("BatchOperations", func(t *testing.T) {
		testBatchOperations(t, cache, ctx)
	})

	// Test metrics
	t.Run("Metrics", func(t *testing.T) {
		testMetrics(t, cache)
	})
}

func testBasicOperations(t *testing.T, cache Cache, ctx context.Context) {
	// Test SET and GET
	key := "test_key"
	value := []byte("test_value")
	ttl := 1 * time.Minute

	err := cache.Set(ctx, key, value, ttl)
	if err != nil {
		t.Fatalf("Failed to set value: %v", err)
	}

	retrievedValue, err := cache.Get(ctx, key)
	if err != nil {
		t.Fatalf("Failed to get value: %v", err)
	}

	if string(retrievedValue) != string(value) {
		t.Errorf("Expected %s, got %s", string(value), string(retrievedValue))
	}

	// Test EXISTS
	exists, err := cache.Exists(ctx, key)
	if err != nil {
		t.Fatalf("Failed to check existence: %v", err)
	}
	if !exists {
		t.Error("Key should exist")
	}

	// Test DELETE
	err = cache.Delete(ctx, key)
	if err != nil {
		t.Fatalf("Failed to delete key: %v", err)
	}

	// Verify deletion
	_, err = cache.Get(ctx, key)
	if err == nil {
		t.Error("Key should not exist after deletion")
	}
	if err != ErrCacheMiss {
		t.Errorf("Expected ErrCacheMiss, got %v", err)
	}
}

func testTTLFunctionality(t *testing.T, cache Cache, ctx context.Context) {
	key := "ttl_test_key"
	value := []byte("ttl_test_value")
	shortTTL := 100 * time.Millisecond

	// Set with short TTL
	err := cache.Set(ctx, key, value, shortTTL)
	if err != nil {
		t.Fatalf("Failed to set value with TTL: %v", err)
	}

	// Verify it exists initially
	_, err = cache.Get(ctx, key)
	if err != nil {
		t.Fatalf("Value should exist before TTL: %v", err)
	}

	// Wait for TTL to expire
	time.Sleep(150 * time.Millisecond)

	// Verify it's expired
	_, err = cache.Get(ctx, key)
	if err != ErrCacheMiss {
		t.Errorf("Value should be expired, got error: %v", err)
	}
}

func testBatchOperations(t *testing.T, cache Cache, ctx context.Context) {
	// Prepare batch data
	items := map[string]CacheItem{
		"batch_key_1": {Value: []byte("batch_value_1"), TTL: 1 * time.Minute},
		"batch_key_2": {Value: []byte("batch_value_2"), TTL: 1 * time.Minute},
		"batch_key_3": {Value: []byte("batch_value_3"), TTL: 1 * time.Minute},
	}

	// Test batch set
	err := cache.SetMulti(ctx, items)
	if err != nil {
		t.Fatalf("Failed to set multiple items: %v", err)
	}

	// Test batch get
	keys := []string{"batch_key_1", "batch_key_2", "batch_key_3", "non_existent_key"}
	results, err := cache.GetMulti(ctx, keys)
	if err != nil {
		t.Fatalf("Failed to get multiple items: %v", err)
	}

	// Verify results
	if len(results) != 3 {
		t.Errorf("Expected 3 results, got %d", len(results))
	}

	for key, expectedValue := range items {
		if actualValue, exists := results[key]; !exists {
			t.Errorf("Key %s should exist in results", key)
		} else if string(actualValue) != string(expectedValue.Value) {
			t.Errorf("Key %s: expected %s, got %s", key, string(expectedValue.Value), string(actualValue))
		}
	}

	// Test batch delete
	deleteKeys := []string{"batch_key_1", "batch_key_2"}
	err = cache.DeleteMulti(ctx, deleteKeys)
	if err != nil {
		t.Fatalf("Failed to delete multiple items: %v", err)
	}

	// Verify deletion
	for _, key := range deleteKeys {
		_, err := cache.Get(ctx, key)
		if err != ErrCacheMiss {
			t.Errorf("Key %s should be deleted", key)
		}
	}
}

func testMetrics(t *testing.T, cache Cache) {
	stats := cache.GetStats()

	// Basic metrics validation
	if stats.Sets == 0 {
		t.Error("Expected some SET operations to be recorded")
	}

	if stats.Hits == 0 {
		t.Error("Expected some cache hits to be recorded")
	}

	totalOps := stats.Hits + stats.Misses
	if totalOps == 0 {
		t.Error("Expected some operations to be recorded")
	}

	if stats.HitRate < 0 || stats.HitRate > 1 {
		t.Errorf("Hit rate should be between 0 and 1, got %f", stats.HitRate)
	}

	if stats.LastUpdated.IsZero() {
		t.Error("LastUpdated should be set")
	}
}

// BenchmarkCacheOperations benchmarks cache performance
func BenchmarkCacheOperations(b *testing.B) {
	logger := logrus.New()
	logger.SetLevel(logrus.ErrorLevel) // Minimize logging for benchmarks

	config := &CacheConfig{
		L1Enabled:     true,
		L1MaxSize:     10000,
		L1TTL:         5 * time.Minute,
		L1CleanupInt:  1 * time.Minute,
		L2Enabled:     false,
		L3Enabled:     false,
		DefaultTTL:    5 * time.Minute,
		EnableMetrics: false, // Disable for better benchmark performance
	}

	cache, err := NewMultiTierCache(config, logger)
	if err != nil {
		b.Fatalf("Failed to create cache: %v", err)
	}
	defer cache.Close()

	ctx := context.Background()
	value := []byte("benchmark_test_value_with_some_length_to_make_it_realistic")

	b.Run("Set", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			key := fmt.Sprintf("bench_set_key_%d", i)
			cache.Set(ctx, key, value, 5*time.Minute)
		}
	})

	// Pre-populate for Get benchmark
	for i := 0; i < 1000; i++ {
		key := fmt.Sprintf("bench_get_key_%d", i)
		cache.Set(ctx, key, value, 5*time.Minute)
	}

	b.Run("Get", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			key := fmt.Sprintf("bench_get_key_%d", i%1000)
			cache.Get(ctx, key)
		}
	})

	b.Run("GetMiss", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			key := fmt.Sprintf("nonexistent_key_%d", i)
			cache.Get(ctx, key)
		}
	})

	// Prepare batch data
	batchItems := make(map[string]CacheItem)
	for i := 0; i < 100; i++ {
		key := fmt.Sprintf("batch_key_%d", i)
		batchItems[key] = CacheItem{Value: value, TTL: 5 * time.Minute}
	}

	b.Run("BatchSet", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			cache.SetMulti(ctx, batchItems)
		}
	})

	batchKeys := make([]string, 100)
	for i := 0; i < 100; i++ {
		batchKeys[i] = fmt.Sprintf("batch_key_%d", i)
	}

	b.Run("BatchGet", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			cache.GetMulti(ctx, batchKeys)
		}
	})
}

// TestMemoryCache tests memory cache specifically
func TestMemoryCache(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.WarnLevel)

	config := &MemoryCacheConfig{
		MaxSize:       10,
		DefaultTTL:    1 * time.Minute,
		CleanupInt:    100 * time.Millisecond,
		EnableMetrics: true,
	}

	cache, err := NewMemoryCache(config, logger)
	if err != nil {
		t.Fatalf("Failed to create memory cache: %v", err)
	}
	defer cache.Close()

	ctx := context.Background()

	t.Run("LRUEviction", func(t *testing.T) {
		// Fill cache beyond capacity
		for i := 0; i < 12; i++ {
			key := fmt.Sprintf("eviction_key_%d", i)
			value := []byte(fmt.Sprintf("value_%d", i))
			err := cache.Set(ctx, key, value, 1*time.Minute)
			if err != nil {
				t.Fatalf("Failed to set key %s: %v", key, err)
			}
		}

		// First two keys should be evicted
		_, err := cache.Get(ctx, "eviction_key_0")
		if err != ErrCacheMiss {
			t.Error("First key should be evicted due to LRU")
		}

		_, err = cache.Get(ctx, "eviction_key_1")
		if err != ErrCacheMiss {
			t.Error("Second key should be evicted due to LRU")
		}

		// Last key should still exist
		_, err = cache.Get(ctx, "eviction_key_11")
		if err != nil {
			t.Errorf("Last key should exist: %v", err)
		}
	})

	t.Run("Expiration", func(t *testing.T) {
		key := "expiration_key"
		value := []byte("expiration_value")

		// Set with very short TTL
		err := cache.Set(ctx, key, value, 50*time.Millisecond)
		if err != nil {
			t.Fatalf("Failed to set expiring key: %v", err)
		}

		// Should exist initially
		_, err = cache.Get(ctx, key)
		if err != nil {
			t.Fatalf("Key should exist before expiration: %v", err)
		}

		// Wait for expiration and cleanup
		time.Sleep(200 * time.Millisecond)

		// Should be expired now
		_, err = cache.Get(ctx, key)
		if err != ErrCacheMiss {
			t.Errorf("Key should be expired: %v", err)
		}
	})
}