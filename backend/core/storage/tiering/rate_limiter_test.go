package tiering

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestRateLimiterBasic(t *testing.T) {
	// Create a rate limiter with 1MB/s and 2 concurrent operations
	rl := NewRateLimiter(1024*1024, 2)
	
	// Test acquiring permit
	ctx := context.Background()
	err := rl.AcquirePermit(ctx)
	if err != nil {
		t.Fatalf("Failed to acquire first permit: %v", err)
	}
	
	err = rl.AcquirePermit(ctx)
	if err != nil {
		t.Fatalf("Failed to acquire second permit: %v", err)
	}
	
	// Third permit should block (test with timeout)
	ctx2, cancel := context.WithTimeout(ctx, 50*time.Millisecond)
	defer cancel()
	
	err = rl.AcquirePermit(ctx2)
	if err == nil {
		t.Fatal("Expected timeout when acquiring third permit")
	}
	
	// Release a permit
	rl.ReleasePermit()
	
	// Now should be able to acquire
	err = rl.AcquirePermit(ctx)
	if err != nil {
		t.Fatalf("Failed to acquire permit after release: %v", err)
	}
}

func TestRateLimiterQuota(t *testing.T) {
	// Create a rate limiter with 100 bytes/s
	rl := NewRateLimiter(100, 5)
	ctx := context.Background()
	
	// First request should succeed immediately
	start := time.Now()
	err := rl.WaitForQuota(ctx, 50)
	if err != nil {
		t.Fatalf("Failed to get quota: %v", err)
	}
	elapsed := time.Since(start)
	if elapsed > 10*time.Millisecond {
		t.Errorf("First quota request took too long: %v", elapsed)
	}
	
	// Second request should also succeed (total 100 bytes)
	err = rl.WaitForQuota(ctx, 50)
	if err != nil {
		t.Fatalf("Failed to get second quota: %v", err)
	}
	
	// Third request should wait for window reset
	start = time.Now()
	err = rl.WaitForQuota(ctx, 50)
	if err != nil {
		t.Fatalf("Failed to get third quota: %v", err)
	}
	elapsed = time.Since(start)
	
	// Should have waited approximately 1 second for window reset
	if elapsed < 900*time.Millisecond || elapsed > 1100*time.Millisecond {
		t.Errorf("Expected to wait ~1s for window reset, got %v", elapsed)
	}
}

func TestMigrationRateLimiter(t *testing.T) {
	mrl := NewMigrationRateLimiter()
	ctx := context.Background()
	
	// Start a migration
	token, err := mrl.StartMigration(ctx, TierHot, TierWarm)
	if err != nil {
		t.Fatalf("Failed to start migration: %v", err)
	}
	
	// Transfer some bytes
	err = token.TransferBytes(ctx, 1024*1024) // 1MB
	if err != nil {
		t.Fatalf("Failed to transfer bytes: %v", err)
	}
	
	// Complete the migration
	token.Complete()
	
	// Check that bytes were recorded
	if token.bytesTransferred != 1024*1024 {
		t.Errorf("Expected 1MB transferred, got %d", token.bytesTransferred)
	}
}

func TestAdaptiveThrottling(t *testing.T) {
	config := RateLimiterConfig{
		GlobalMaxBytesPerSecond:  100 * 1024 * 1024, // 100 MB/s
		GlobalMaxConcurrent:      5,
		HotTierMaxBytesPerSecond: 50 * 1024 * 1024,
		WarmTierMaxBytesPerSecond: 30 * 1024 * 1024,
		ColdTierMaxBytesPerSecond: 20 * 1024 * 1024,
		EnableAdaptiveThrottling: true,
		MinBytesPerSecond:       1 * 1024 * 1024,
		MaxBytesPerSecond:       200 * 1024 * 1024,
		CPUThrottleThreshold:    80.0,
		MemoryThrottleThreshold: 85.0,
		NetworkThrottleThreshold: 90.0,
	}
	
	mrl := NewMigrationRateLimiterWithConfig(config)
	
	// Test throttling with high CPU usage
	systemLoad := SystemLoadInfo{
		CPUUsage:        90.0, // Above threshold
		MemoryUsage:     50.0,
		NetworkBandwidth: 50.0,
		DiskIOPS:        100.0,
	}
	
	originalRate := mrl.globalLimiter.maxBytesPerSecond
	mrl.AdaptiveThrottle(systemLoad)
	newRate := mrl.globalLimiter.maxBytesPerSecond
	
	if newRate >= originalRate {
		t.Error("Expected rate to decrease with high CPU usage")
	}
	
	// Test with low system load
	systemLoad = SystemLoadInfo{
		CPUUsage:        30.0,
		MemoryUsage:     40.0,
		NetworkBandwidth: 20.0,
		DiskIOPS:        50.0,
	}
	
	mrl.AdaptiveThrottle(systemLoad)
	finalRate := mrl.globalLimiter.maxBytesPerSecond
	
	if finalRate <= newRate {
		t.Error("Expected rate to increase with low system load")
	}
}

func TestPriorityMigration(t *testing.T) {
	mrl := NewMigrationRateLimiter()
	ctx := context.Background()
	
	// Get original rate
	originalRate := mrl.globalLimiter.maxBytesPerSecond
	
	// Start priority migration with 2x multiplier
	token, err := mrl.PriorityMigration(ctx, TierCold, TierHot, 2.0)
	if err != nil {
		t.Fatalf("Failed to start priority migration: %v", err)
	}
	defer token.Complete()
	
	// Rate should be back to original after priority migration starts
	currentRate := mrl.globalLimiter.maxBytesPerSecond
	if currentRate != originalRate {
		t.Errorf("Rate not restored after priority migration: expected %d, got %d", 
			originalRate, currentRate)
	}
}

func TestConcurrentMigrations(t *testing.T) {
	config := RateLimiterConfig{
		GlobalMaxBytesPerSecond:  10 * 1024 * 1024, // 10 MB/s
		GlobalMaxConcurrent:      3, // Only 3 concurrent migrations
		HotTierMaxBytesPerSecond: 5 * 1024 * 1024,
		WarmTierMaxBytesPerSecond: 3 * 1024 * 1024,
		ColdTierMaxBytesPerSecond: 2 * 1024 * 1024,
		EnableAdaptiveThrottling: false,
	}
	
	mrl := NewMigrationRateLimiterWithConfig(config)
	ctx := context.Background()
	
	// Start 3 migrations (should all succeed)
	tokens := make([]*MigrationToken, 3)
	for i := 0; i < 3; i++ {
		token, err := mrl.StartMigration(ctx, TierWarm, TierHot)
		if err != nil {
			t.Fatalf("Failed to start migration %d: %v", i, err)
		}
		tokens[i] = token
	}
	
	// 4th migration should block
	ctx2, cancel := context.WithTimeout(ctx, 50*time.Millisecond)
	defer cancel()
	
	_, err := mrl.StartMigration(ctx2, TierWarm, TierHot)
	if err == nil {
		t.Fatal("Expected 4th migration to fail due to concurrency limit")
	}
	
	// Complete one migration
	tokens[0].Complete()
	
	// Now 4th should succeed
	token4, err := mrl.StartMigration(ctx, TierWarm, TierHot)
	if err != nil {
		t.Fatalf("Failed to start 4th migration after release: %v", err)
	}
	
	// Clean up
	tokens[1].Complete()
	tokens[2].Complete()
	token4.Complete()
}

func TestRateLimiterMetrics(t *testing.T) {
	mrl := NewMigrationRateLimiter()
	ctx := context.Background()
	
	// Perform some migrations
	for i := 0; i < 5; i++ {
		token, err := mrl.StartMigration(ctx, TierHot, TierWarm)
		if err != nil {
			t.Fatalf("Failed to start migration: %v", err)
		}
		
		err = token.TransferBytes(ctx, 1024*1024) // 1MB each
		if err != nil {
			t.Fatalf("Failed to transfer bytes: %v", err)
		}
		
		token.Complete()
	}
	
	// Get metrics
	metrics := mrl.GetMetrics()
	
	bytesTransferred, ok := metrics["global_bytes_transferred"].(int64)
	if !ok {
		t.Fatal("global_bytes_transferred metric not found")
	}
	
	if bytesTransferred != 5*1024*1024 {
		t.Errorf("Expected 5MB transferred, got %d", bytesTransferred)
	}
	
	tierMetrics, ok := metrics["tiers"].(map[string]map[string]interface{})
	if !ok {
		t.Fatal("tier metrics not found")
	}
	
	hotMetrics, ok := tierMetrics["hot"]
	if !ok {
		t.Fatal("hot tier metrics not found")
	}
	
	migrations, ok := hotMetrics["migrations"].(int64)
	if !ok {
		t.Fatal("hot tier migrations count not found")
	}
	
	if migrations != 5 {
		t.Errorf("Expected 5 migrations, got %d", migrations)
	}
}

func TestRateLimiterStress(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping stress test in short mode")
	}
	
	config := RateLimiterConfig{
		GlobalMaxBytesPerSecond:  100 * 1024 * 1024, // 100 MB/s
		GlobalMaxConcurrent:      10,
		HotTierMaxBytesPerSecond: 50 * 1024 * 1024,
		WarmTierMaxBytesPerSecond: 30 * 1024 * 1024,
		ColdTierMaxBytesPerSecond: 20 * 1024 * 1024,
		EnableAdaptiveThrottling: true,
	}
	
	mrl := NewMigrationRateLimiterWithConfig(config)
	ctx := context.Background()
	
	// Track total bytes transferred
	var totalBytes int64
	var successfulMigrations int64
	
	// Run multiple concurrent migrations
	var wg sync.WaitGroup
	numWorkers := 20
	migrationsPerWorker := 10
	
	start := time.Now()
	
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			
			for j := 0; j < migrationsPerWorker; j++ {
				// Random tier selection
				sourceTier := TierLevel(workerID % 3)
				targetTier := TierLevel((workerID + 1) % 3)
				
				token, err := mrl.StartMigration(ctx, sourceTier, targetTier)
				if err != nil {
					continue // Skip if can't start
				}
				
				// Transfer random amount of data
				dataSize := int64((workerID*100 + j) * 1024) // Variable sizes
				err = token.TransferBytes(ctx, dataSize)
				if err == nil {
					atomic.AddInt64(&totalBytes, dataSize)
					atomic.AddInt64(&successfulMigrations, 1)
				}
				
				token.Complete()
				
				// Small delay between migrations
				time.Sleep(time.Duration(workerID) * time.Millisecond)
			}
		}(i)
	}
	
	wg.Wait()
	elapsed := time.Since(start)
	
	// Calculate effective rate
	effectiveRate := float64(totalBytes) / elapsed.Seconds()
	
	t.Logf("Stress test completed:")
	t.Logf("  Total bytes transferred: %d", totalBytes)
	t.Logf("  Successful migrations: %d", successfulMigrations)
	t.Logf("  Time elapsed: %v", elapsed)
	t.Logf("  Effective rate: %.2f MB/s", effectiveRate/(1024*1024))
	
	// Verify rate limiting worked (should not exceed global max significantly)
	maxExpectedRate := float64(config.GlobalMaxBytesPerSecond) * 1.2 // Allow 20% variance
	if effectiveRate > maxExpectedRate {
		t.Errorf("Rate limiting not effective: %.2f MB/s exceeds max %.2f MB/s", 
			effectiveRate/(1024*1024), maxExpectedRate/(1024*1024))
	}
	
	// Get final metrics
	metrics := mrl.GetMetrics()
	t.Logf("Final metrics: %+v", metrics)
}

// Benchmark tests

func BenchmarkRateLimiterAcquireRelease(b *testing.B) {
	rl := NewRateLimiter(100*1024*1024, 10)
	ctx := context.Background()
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rl.AcquirePermit(ctx)
		rl.ReleasePermit()
	}
}

func BenchmarkRateLimiterQuota(b *testing.B) {
	rl := NewRateLimiter(1024*1024*1024, 10) // 1GB/s (essentially unlimited)
	ctx := context.Background()
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rl.WaitForQuota(ctx, 1024) // 1KB transfers
	}
}

func BenchmarkMigrationRateLimiter(b *testing.B) {
	mrl := NewMigrationRateLimiter()
	ctx := context.Background()
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		token, _ := mrl.StartMigration(ctx, TierHot, TierWarm)
		token.TransferBytes(ctx, 1024)
		token.Complete()
	}
}

func BenchmarkConcurrentMigrationRateLimiter(b *testing.B) {
	config := RateLimiterConfig{
		GlobalMaxBytesPerSecond:  1024 * 1024 * 1024, // 1GB/s
		GlobalMaxConcurrent:      100,
		HotTierMaxBytesPerSecond: 500 * 1024 * 1024,
		WarmTierMaxBytesPerSecond: 300 * 1024 * 1024,
		ColdTierMaxBytesPerSecond: 200 * 1024 * 1024,
		EnableAdaptiveThrottling: false,
	}
	
	mrl := NewMigrationRateLimiterWithConfig(config)
	ctx := context.Background()
	
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			token, err := mrl.StartMigration(ctx, TierHot, TierWarm)
			if err == nil {
				token.TransferBytes(ctx, 1024)
				token.Complete()
			}
		}
	})
}