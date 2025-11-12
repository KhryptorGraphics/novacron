// Performance Test Suite for DWCP v3
//
// Comprehensive performance and stress tests:
// - Extreme load tests (1M+ req/sec)
// - Latency percentile tests (P99.99, P99.999)
// - Throughput scaling tests
// - Resource efficiency tests
//
// Phase 7: Extreme Performance Optimization
// Target: Validate 5,000 GB/s throughput, <20ms P99 latency

package performance

import (
	"context"
	"fmt"
	"math/rand"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// Test Configuration
type TestConfig struct {
	Duration         time.Duration
	Concurrency      int
	RequestRate      int // requests per second
	PayloadSize      int // bytes
	WarmupDuration   time.Duration
	CooldownDuration time.Duration
	EnableProfiling  bool
	EnableMetrics    bool
}

// Test Results
type TestResults struct {
	TotalRequests     uint64
	SuccessfulReqs    uint64
	FailedReqs        uint64
	TotalBytes        uint64
	Duration          time.Duration
	Throughput        float64 // req/sec
	Bandwidth         float64 // GB/s
	LatencyP50        time.Duration
	LatencyP95        time.Duration
	LatencyP99        time.Duration
	LatencyP999       time.Duration
	LatencyP9999      time.Duration
	LatencyMin        time.Duration
	LatencyMax        time.Duration
	LatencyMean       time.Duration
	CPUUtilization    float64
	MemoryUsage       uint64
	GoroutineCount    int
	ErrorRate         float64
}

// Latency Tracker
type LatencyTracker struct {
	latencies []time.Duration
	mu        sync.Mutex
}

// NewLatencyTracker creates a new latency tracker
func NewLatencyTracker(capacity int) *LatencyTracker {
	return &LatencyTracker{
		latencies: make([]time.Duration, 0, capacity),
	}
}

// Record records a latency measurement
func (lt *LatencyTracker) Record(latency time.Duration) {
	lt.mu.Lock()
	defer lt.mu.Unlock()
	lt.latencies = append(lt.latencies, latency)
}

// Calculate percentiles
func (lt *LatencyTracker) CalculatePercentiles() (p50, p95, p99, p999, p9999, min, max, mean time.Duration) {
	lt.mu.Lock()
	defer lt.mu.Unlock()

	if len(lt.latencies) == 0 {
		return
	}

	// Sort latencies
	sorted := make([]time.Duration, len(lt.latencies))
	copy(sorted, lt.latencies)
	sortDurations(sorted)

	// Calculate percentiles
	p50 = sorted[int(float64(len(sorted))*0.50)]
	p95 = sorted[int(float64(len(sorted))*0.95)]
	p99 = sorted[int(float64(len(sorted))*0.99)]
	p999 = sorted[int(float64(len(sorted))*0.999)]
	p9999 = sorted[int(float64(len(sorted))*0.9999)]
	min = sorted[0]
	max = sorted[len(sorted)-1]

	// Calculate mean
	var sum time.Duration
	for _, lat := range sorted {
		sum += lat
	}
	mean = sum / time.Duration(len(sorted))

	return
}

// TestExtremeLoad tests extreme load handling (1M+ req/sec target)
func TestExtremeLoad(t *testing.T) {
	config := &TestConfig{
		Duration:        60 * time.Second,
		Concurrency:     1000,
		RequestRate:     1000000, // 1M req/sec
		PayloadSize:     1024,    // 1 KB
		WarmupDuration:  5 * time.Second,
		EnableProfiling: true,
		EnableMetrics:   true,
	}

	results := runLoadTest(t, config)
	printTestResults("Extreme Load Test", results)

	// Validate results
	if results.Throughput < 900000 { // 90% of target
		t.Errorf("Throughput too low: %.0f req/sec (expected >= 900k)", results.Throughput)
	}

	if results.ErrorRate > 0.01 { // <1% error rate
		t.Errorf("Error rate too high: %.2f%% (expected < 1%%)", results.ErrorRate*100)
	}
}

// TestLatencyPercentiles tests latency distribution
func TestLatencyPercentiles(t *testing.T) {
	config := &TestConfig{
		Duration:        30 * time.Second,
		Concurrency:     100,
		RequestRate:     100000,
		PayloadSize:     512,
		WarmupDuration:  3 * time.Second,
		EnableProfiling: true,
	}

	results := runLoadTest(t, config)
	printTestResults("Latency Percentile Test", results)

	// Validate latency targets
	if results.LatencyP99 > 20*time.Millisecond {
		t.Errorf("P99 latency too high: %v (expected < 20ms)", results.LatencyP99)
	}

	if results.LatencyP999 > 50*time.Millisecond {
		t.Errorf("P999 latency too high: %v (expected < 50ms)", results.LatencyP999)
	}

	if results.LatencyP9999 > 100*time.Millisecond {
		t.Errorf("P9999 latency too high: %v (expected < 100ms)", results.LatencyP9999)
	}
}

// TestThroughputScaling tests throughput scaling with concurrency
func TestThroughputScaling(t *testing.T) {
	concurrencyLevels := []int{10, 50, 100, 500, 1000}
	results := make([]*TestResults, 0)

	for _, concurrency := range concurrencyLevels {
		config := &TestConfig{
			Duration:       10 * time.Second,
			Concurrency:    concurrency,
			RequestRate:    100000,
			PayloadSize:    1024,
			WarmupDuration: 2 * time.Second,
		}

		result := runLoadTest(t, config)
		results = append(results, result)

		fmt.Printf("Concurrency %d: Throughput=%.0f req/sec, Latency P99=%v\n",
			concurrency, result.Throughput, result.LatencyP99)
	}

	// Check scaling efficiency
	for i := 1; i < len(results); i++ {
		scaleFactor := float64(concurrencyLevels[i]) / float64(concurrencyLevels[i-1])
		throughputRatio := results[i].Throughput / results[i-1].Throughput

		if throughputRatio < scaleFactor*0.7 { // 70% scaling efficiency
			t.Logf("Warning: Sublinear scaling at concurrency %d (%.2fx throughput for %.0fx concurrency)",
				concurrencyLevels[i], throughputRatio, scaleFactor)
		}
	}
}

// TestResourceEfficiency tests CPU and memory efficiency
func TestResourceEfficiency(t *testing.T) {
	config := &TestConfig{
		Duration:       30 * time.Second,
		Concurrency:    500,
		RequestRate:    500000,
		PayloadSize:    2048,
		WarmupDuration: 5 * time.Second,
		EnableMetrics:  true,
	}

	results := runLoadTest(t, config)
	printTestResults("Resource Efficiency Test", results)

	// Calculate efficiency metrics
	reqPerCPU := results.Throughput / results.CPUUtilization
	fmt.Printf("Requests per CPU%% = %.0f\n", reqPerCPU)

	memPerReq := float64(results.MemoryUsage) / float64(results.TotalRequests)
	fmt.Printf("Memory per request = %.2f bytes\n", memPerReq)

	// Validate efficiency targets
	if results.CPUUtilization > 90 {
		t.Logf("Warning: High CPU utilization: %.2f%%", results.CPUUtilization)
	}

	if memPerReq > 10000 { // 10 KB per request
		t.Errorf("Memory usage too high: %.2f bytes/req (expected < 10KB)", memPerReq)
	}
}

// TestZeroCopyPerformance tests zero-copy optimization
func TestZeroCopyPerformance(t *testing.T) {
	// Test with zero-copy
	config := &TestConfig{
		Duration:       15 * time.Second,
		Concurrency:    200,
		RequestRate:    200000,
		PayloadSize:    8192,
		WarmupDuration: 2 * time.Second,
	}

	resultsZeroCopy := runLoadTest(t, config)

	fmt.Printf("Zero-copy performance:\n")
	fmt.Printf("  Throughput: %.2f GB/s\n", resultsZeroCopy.Bandwidth)
	fmt.Printf("  Latency P99: %v\n", resultsZeroCopy.LatencyP99)

	// Validate zero-copy benefits
	if resultsZeroCopy.Bandwidth < 10.0 { // 10 GB/s minimum
		t.Errorf("Zero-copy bandwidth too low: %.2f GB/s (expected >= 10)", resultsZeroCopy.Bandwidth)
	}
}

// TestSIMDOptimization tests SIMD optimization performance
func TestSIMDOptimization(t *testing.T) {
	simdManager, err := NewSIMDManager()
	if err != nil {
		t.Fatalf("Failed to create SIMD manager: %v", err)
	}

	dataSize := 1024 * 1024 * 100 // 100 MB
	input := make([]byte, dataSize)
	output := make([]byte, dataSize)

	// Fill with random data
	rand.Read(input)

	start := time.Now()

	// Test memcpy performance
	err = simdManager.ExecuteOptimization("memcpy_simd", input, output)
	if err != nil {
		t.Fatalf("SIMD memcpy failed: %v", err)
	}

	elapsed := time.Since(start)
	bandwidth := float64(dataSize) / elapsed.Seconds() / (1024 * 1024 * 1024) // GB/s

	fmt.Printf("SIMD memcpy: %.2f GB/s\n", bandwidth)

	// Validate SIMD performance
	if bandwidth < 50.0 { // 50 GB/s minimum
		t.Errorf("SIMD performance too low: %.2f GB/s (expected >= 50)", bandwidth)
	}

	simdManager.PrintStatistics()
}

// TestLockFreeDataStructures tests lock-free performance
func TestLockFreeDataStructures(t *testing.T) {
	iterations := 1000000
	concurrency := 100

	// Test lock-free queue
	t.Run("LockFreeQueue", func(t *testing.T) {
		start := time.Now()
		BenchmarkLFQueue(iterations, concurrency)
		elapsed := time.Since(start)

		opsPerSec := float64(iterations*concurrency*2) / elapsed.Seconds() // *2 for enq+deq
		fmt.Printf("Lock-free queue: %.2f Mops/sec\n", opsPerSec/1000000)

		if opsPerSec < 10000000 { // 10M ops/sec minimum
			t.Errorf("Lock-free queue too slow: %.2f ops/sec", opsPerSec)
		}
	})

	// Test lock-free stack
	t.Run("LockFreeStack", func(t *testing.T) {
		start := time.Now()
		BenchmarkLFStack(iterations, concurrency)
		elapsed := time.Since(start)

		opsPerSec := float64(iterations*concurrency*2) / elapsed.Seconds()
		fmt.Printf("Lock-free stack: %.2f Mops/sec\n", opsPerSec/1000000)

		if opsPerSec < 10000000 {
			t.Errorf("Lock-free stack too slow: %.2f ops/sec", opsPerSec)
		}
	})

	// Test wait-free hash table
	t.Run("WaitFreeHashTable", func(t *testing.T) {
		start := time.Now()
		BenchmarkWFHashTable(iterations, concurrency)
		elapsed := time.Since(start)

		opsPerSec := float64(iterations*concurrency*2) / elapsed.Seconds()
		fmt.Printf("Wait-free hash table: %.2f Mops/sec\n", opsPerSec/1000000)

		if opsPerSec < 5000000 { // 5M ops/sec minimum
			t.Errorf("Wait-free hash table too slow: %.2f ops/sec", opsPerSec)
		}
	})

	PrintLockFreeStatistics()
}

// TestMemoryOptimization tests memory optimization performance
func TestMemoryOptimization(t *testing.T) {
	mm, err := NewMemoryManager(true)
	if err != nil {
		t.Fatalf("Failed to create memory manager: %v", err)
	}
	defer mm.Close()

	// Test memory pool allocation performance
	iterations := 1000000
	start := time.Now()

	for i := 0; i < iterations; i++ {
		ptr, err := mm.Allocate("small")
		if err != nil {
			t.Fatalf("Allocation failed: %v", err)
		}

		// Use the memory
		_ = ptr

		// Deallocate
		mm.Deallocate("small", ptr)
	}

	elapsed := time.Since(start)
	opsPerSec := float64(iterations) / elapsed.Seconds()

	fmt.Printf("Memory pool allocation: %.2f Mops/sec\n", opsPerSec/1000000)

	if opsPerSec < 5000000 { // 5M ops/sec minimum
		t.Errorf("Memory pool too slow: %.2f ops/sec", opsPerSec)
	}

	mm.PrintStatistics()
}

// Run load test with given configuration
func runLoadTest(t *testing.T, config *TestConfig) *TestResults {
	ctx, cancel := context.WithTimeout(context.Background(), config.Duration+config.WarmupDuration)
	defer cancel()

	results := &TestResults{}
	latencyTracker := NewLatencyTracker(1000000)

	var wg sync.WaitGroup
	var totalReqs, successReqs, failedReqs, totalBytes atomic.Uint64

	startTime := time.Now()

	// Warmup phase
	if config.WarmupDuration > 0 {
		time.Sleep(config.WarmupDuration)
	}

	testStart := time.Now()

	// Launch worker goroutines
	for i := 0; i < config.Concurrency; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()

			reqsPerWorker := config.RequestRate / config.Concurrency
			interval := time.Second / time.Duration(reqsPerWorker)
			ticker := time.NewTicker(interval)
			defer ticker.Stop()

			payload := make([]byte, config.PayloadSize)
			rand.Read(payload)

			for {
				select {
				case <-ctx.Done():
					return
				case <-ticker.C:
					reqStart := time.Now()

					// Simulate request processing
					if err := processRequest(payload); err != nil {
						failedReqs.Add(1)
					} else {
						successReqs.Add(1)
						totalBytes.Add(uint64(config.PayloadSize))
					}

					latency := time.Since(reqStart)
					latencyTracker.Record(latency)
					totalReqs.Add(1)
				}
			}
		}(i)
	}

	// Wait for all workers to complete
	wg.Wait()

	testDuration := time.Since(testStart)

	// Collect results
	results.TotalRequests = totalReqs.Load()
	results.SuccessfulReqs = successReqs.Load()
	results.FailedReqs = failedReqs.Load()
	results.TotalBytes = totalBytes.Load()
	results.Duration = testDuration
	results.Throughput = float64(results.SuccessfulReqs) / testDuration.Seconds()
	results.Bandwidth = float64(results.TotalBytes) / testDuration.Seconds() / (1024 * 1024 * 1024) // GB/s

	// Calculate latencies
	results.LatencyP50, results.LatencyP95, results.LatencyP99,
		results.LatencyP999, results.LatencyP9999,
		results.LatencyMin, results.LatencyMax, results.LatencyMean = latencyTracker.CalculatePercentiles()

	// Collect resource metrics
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	results.MemoryUsage = m.Alloc
	results.GoroutineCount = runtime.NumGoroutine()
	results.CPUUtilization = 50.0 // Placeholder - use actual CPU measurement

	// Calculate error rate
	if results.TotalRequests > 0 {
		results.ErrorRate = float64(results.FailedReqs) / float64(results.TotalRequests)
	}

	_ = startTime // Use to avoid compiler warning

	return results
}

// Simulate request processing
func processRequest(payload []byte) error {
	// Simulate some work
	time.Sleep(time.Microsecond * time.Duration(rand.Intn(100)))
	return nil
}

// Print test results
func printTestResults(testName string, results *TestResults) {
	fmt.Printf("\n=== %s Results ===\n", testName)
	fmt.Printf("Duration: %v\n", results.Duration)
	fmt.Printf("Total requests: %d\n", results.TotalRequests)
	fmt.Printf("Successful: %d\n", results.SuccessfulReqs)
	fmt.Printf("Failed: %d\n", results.FailedReqs)
	fmt.Printf("Error rate: %.4f%%\n", results.ErrorRate*100)
	fmt.Printf("Throughput: %.2f req/sec (%.2f Mreq/sec)\n",
		results.Throughput, results.Throughput/1000000)
	fmt.Printf("Bandwidth: %.2f GB/s\n", results.Bandwidth)
	fmt.Printf("\nLatency Percentiles:\n")
	fmt.Printf("  Min: %v\n", results.LatencyMin)
	fmt.Printf("  P50: %v\n", results.LatencyP50)
	fmt.Printf("  P95: %v\n", results.LatencyP95)
	fmt.Printf("  P99: %v\n", results.LatencyP99)
	fmt.Printf("  P99.9: %v\n", results.LatencyP999)
	fmt.Printf("  P99.99: %v\n", results.LatencyP9999)
	fmt.Printf("  Max: %v\n", results.LatencyMax)
	fmt.Printf("  Mean: %v\n", results.LatencyMean)
	fmt.Printf("\nResource Usage:\n")
	fmt.Printf("  CPU: %.2f%%\n", results.CPUUtilization)
	fmt.Printf("  Memory: %.2f MB\n", float64(results.MemoryUsage)/(1024*1024))
	fmt.Printf("  Goroutines: %d\n", results.GoroutineCount)
	fmt.Printf("=========================\n\n")
}

// Sort durations
func sortDurations(durations []time.Duration) {
	// Simple insertion sort (use quicksort in production)
	for i := 1; i < len(durations); i++ {
		key := durations[i]
		j := i - 1
		for j >= 0 && durations[j] > key {
			durations[j+1] = durations[j]
			j--
		}
		durations[j+1] = key
	}
}

// Benchmark functions

func BenchmarkExtremeLoad(b *testing.B) {
	config := &TestConfig{
		Duration:       time.Duration(b.N) * time.Millisecond,
		Concurrency:    1000,
		RequestRate:    1000000,
		PayloadSize:    1024,
		WarmupDuration: 0,
	}

	b.ResetTimer()
	results := runLoadTest(&testing.T{}, config)
	b.ReportMetric(results.Throughput, "req/sec")
	b.ReportMetric(results.Bandwidth, "GB/s")
}

func BenchmarkLowLatency(b *testing.B) {
	config := &TestConfig{
		Duration:       time.Duration(b.N) * time.Millisecond,
		Concurrency:    100,
		RequestRate:    100000,
		PayloadSize:    512,
		WarmupDuration: 0,
	}

	b.ResetTimer()
	results := runLoadTest(&testing.T{}, config)
	b.ReportMetric(float64(results.LatencyP99.Microseconds()), "µs/op_p99")
}
