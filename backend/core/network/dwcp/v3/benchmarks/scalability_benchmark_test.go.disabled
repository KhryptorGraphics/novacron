package benchmarks

import (
	"fmt"
	"math"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// BenchmarkLinearScalability tests linear scalability from 1 to 1000 VMs
func BenchmarkLinearScalability(b *testing.B) {
	vmCounts := []int{1, 10, 50, 100, 250, 500, 1000}

	baselineResults := make(map[int]float64)

	for _, vmCount := range vmCounts {
		b.Run(fmt.Sprintf("%dVMs", vmCount), func(b *testing.B) {
			throughput := benchmarkScalability(b, vmCount)
			baselineResults[vmCount] = throughput
		})
	}

	// Calculate and report scalability metrics
	b.Run("ScalabilityAnalysis", func(b *testing.B) {
		if len(baselineResults) > 0 {
			linearityCoeff := calculateLinearityCoefficient(baselineResults)
			efficiencyRetention := calculateEfficiencyRetention(baselineResults)

			b.ReportMetric(linearityCoeff, "linearity_coeff")
			b.ReportMetric(efficiencyRetention, "efficiency_%")

			b.Logf("Scalability Analysis:")
			b.Logf("  Linearity Coefficient: %.4f (target: > 0.8)", linearityCoeff)
			b.Logf("  Efficiency Retention: %.2f%% (target: > 70%%)", efficiencyRetention)

			for count, throughput := range baselineResults {
				b.Logf("  %d VMs: %.2f ops/sec", count, throughput)
			}
		}
	})
}

func benchmarkScalability(b *testing.B, vmCount int) float64 {
	b.ReportAllocs()

	var totalOperations int64
	startTime := time.Now()

	b.ResetTimer()

	// Simulate workload for multiple iterations
	for i := 0; i < b.N; i++ {
		var wg sync.WaitGroup

		for j := 0; j < vmCount; j++ {
			wg.Add(1)
			go func(vmID int) {
				defer wg.Done()

				// Simulate typical VM operations
				// 1. Memory operation
				time.Sleep(100 * time.Microsecond)

				// 2. Network I/O
				time.Sleep(50 * time.Microsecond)

				// 3. State update
				time.Sleep(20 * time.Microsecond)

				atomic.AddInt64(&totalOperations, 1)
			}(j)
		}

		wg.Wait()
	}

	b.StopTimer()

	duration := time.Since(startTime)
	throughput := float64(totalOperations) / duration.Seconds()

	b.ReportMetric(throughput, "ops/sec")
	b.ReportMetric(float64(vmCount), "vm_count")

	return throughput
}

// BenchmarkResourceUsageUnderLoad tests resource usage at different scales
func BenchmarkResourceUsageUnderLoad(b *testing.B) {
	scenarios := []struct {
		name      string
		vmCount   int
		intensity string
	}{
		{"10VMs_Light", 10, "light"},
		{"50VMs_Light", 50, "light"},
		{"100VMs_Light", 100, "light"},
		{"10VMs_Heavy", 10, "heavy"},
		{"50VMs_Heavy", 50, "heavy"},
		{"100VMs_Heavy", 100, "heavy"},
	}

	for _, sc := range scenarios {
		b.Run(sc.name, func(b *testing.B) {
			benchmarkResourceUsage(b, sc.vmCount, sc.intensity)
		})
	}
}

func benchmarkResourceUsage(b *testing.B, vmCount int, intensity string) {
	b.ReportAllocs()

	var totalMemoryAllocated int64
	var totalGoroutines int64
	var peakMemory int64

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		currentMemory := int64(0)
		goroutineCount := int64(0)

		var wg sync.WaitGroup

		for j := 0; j < vmCount; j++ {
			wg.Add(1)
			goroutineCount++

			go func(vmID int) {
				defer wg.Done()

				// Allocate memory based on intensity
				var vmMemory int64
				if intensity == "light" {
					vmMemory = 1024 * 1024 // 1MB per VM
				} else {
					vmMemory = 10 * 1024 * 1024 // 10MB per VM
				}

				buffer := make([]byte, vmMemory)
				_ = buffer

				atomic.AddInt64(&currentMemory, vmMemory)
				atomic.AddInt64(&totalMemoryAllocated, vmMemory)

				// Simulate work
				if intensity == "light" {
					time.Sleep(100 * time.Microsecond)
				} else {
					time.Sleep(500 * time.Microsecond)
				}
			}(j)
		}

		wg.Wait()

		// Track peak memory
		if currentMemory > peakMemory {
			atomic.StoreInt64(&peakMemory, currentMemory)
		}

		atomic.AddInt64(&totalGoroutines, goroutineCount)
	}

	b.StopTimer()

	avgMemoryMB := float64(totalMemoryAllocated) / float64(b.N) / 1024 / 1024
	peakMemoryMB := float64(peakMemory) / 1024 / 1024
	avgGoroutines := float64(totalGoroutines) / float64(b.N)

	b.ReportMetric(avgMemoryMB, "avg_memory_MB")
	b.ReportMetric(peakMemoryMB, "peak_memory_MB")
	b.ReportMetric(avgGoroutines, "avg_goroutines")
}

// BenchmarkPerformanceDegradation tests performance degradation curve
func BenchmarkPerformanceDegradation(b *testing.B) {
	vmCounts := []int{10, 50, 100, 200, 500, 1000}

	degradationData := make(map[int]degradationMetrics)

	for _, vmCount := range vmCounts {
		b.Run(fmt.Sprintf("%dVMs", vmCount), func(b *testing.B) {
			metrics := benchmarkPerformanceDegradation(b, vmCount)
			degradationData[vmCount] = metrics
		})
	}

	// Analyze degradation
	b.Run("DegradationAnalysis", func(b *testing.B) {
		if len(degradationData) > 0 {
			baseline := degradationData[10]

			b.Logf("Performance Degradation Analysis:")
			b.Logf("  Baseline (10 VMs): %.2f ops/sec, %.2fms latency", baseline.throughput, baseline.latency)

			for _, count := range vmCounts {
				if count == 10 {
					continue
				}

				metrics := degradationData[count]
				throughputDegradation := (1 - metrics.throughput/baseline.throughput) * 100
				latencyIncrease := (metrics.latency / baseline.latency) * 100

				b.Logf("  %d VMs: %.2f%% throughput loss, %.0f%% latency increase",
					count, throughputDegradation, latencyIncrease)

				b.ReportMetric(throughputDegradation, fmt.Sprintf("degradation_%dvms_%%", count))
			}
		}
	})
}

type degradationMetrics struct {
	throughput float64
	latency    float64
	errorRate  float64
}

func benchmarkPerformanceDegradation(b *testing.B, vmCount int) degradationMetrics {
	b.ReportAllocs()

	var totalOperations int64
	var totalLatency time.Duration
	var errors int64

	startTime := time.Now()

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		var wg sync.WaitGroup

		for j := 0; j < vmCount; j++ {
			wg.Add(1)
			go func(vmID int) {
				defer wg.Done()

				opStart := time.Now()

				// Simulate operation with potential for errors under load
				if vmCount > 500 && vmID%100 == 0 {
					// Simulate occasional errors at high scale
					atomic.AddInt64(&errors, 1)
					return
				}

				// Simulate work that scales with load
				workTime := time.Duration(100+vmCount/10) * time.Microsecond
				time.Sleep(workTime)

				opLatency := time.Since(opStart)
				atomic.AddInt64(&totalOperations, 1)
				atomic.AddInt64((*int64)(&totalLatency), int64(opLatency))
			}(j)
		}

		wg.Wait()
	}

	b.StopTimer()

	duration := time.Since(startTime)
	throughput := float64(totalOperations) / duration.Seconds()
	avgLatency := float64(totalLatency.Milliseconds()) / float64(totalOperations)
	errorRate := float64(errors) / float64(totalOperations+errors) * 100

	b.ReportMetric(throughput, "ops/sec")
	b.ReportMetric(avgLatency, "latency_ms")
	b.ReportMetric(errorRate, "error_%")

	return degradationMetrics{
		throughput: throughput,
		latency:    avgLatency,
		errorRate:  errorRate,
	}
}

// BenchmarkConcurrencyScalability tests concurrency scaling
func BenchmarkConcurrencyScalability(b *testing.B) {
	concurrencyLevels := []int{1, 10, 50, 100, 500, 1000, 5000}

	for _, concurrency := range concurrencyLevels {
		b.Run(fmt.Sprintf("%dConcurrent", concurrency), func(b *testing.B) {
			benchmarkConcurrency(b, concurrency)
		})
	}
}

func benchmarkConcurrency(b *testing.B, concurrency int) {
	b.ReportAllocs()

	var totalOperations int64
	startTime := time.Now()

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		var wg sync.WaitGroup

		for j := 0; j < concurrency; j++ {
			wg.Add(1)
			go func() {
				defer wg.Done()

				// Simulate concurrent operation
				time.Sleep(100 * time.Microsecond)
				atomic.AddInt64(&totalOperations, 1)
			}()
		}

		wg.Wait()
	}

	b.StopTimer()

	duration := time.Since(startTime)
	throughput := float64(totalOperations) / duration.Seconds()

	b.ReportMetric(throughput, "ops/sec")
	b.ReportMetric(float64(concurrency), "concurrency")
}

// BenchmarkMemoryScalability tests memory scaling characteristics
func BenchmarkMemoryScalability(b *testing.B) {
	vmCounts := []int{10, 50, 100, 500, 1000}

	for _, vmCount := range vmCounts {
		b.Run(fmt.Sprintf("%dVMs", vmCount), func(b *testing.B) {
			benchmarkMemoryScalability(b, vmCount)
		})
	}
}

func benchmarkMemoryScalability(b *testing.B, vmCount int) {
	b.ReportAllocs()

	var totalAllocations int64

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		// Allocate memory structures for VMs
		vms := make([]vmState, vmCount)

		for j := range vms {
			vms[j] = vmState{
				id:          j,
				memory:      make([]byte, 1024*1024), // 1MB per VM
				state:       make(map[string]interface{}, 100),
				connections: make([]connection, 10),
			}

			atomic.AddInt64(&totalAllocations, 1)
		}

		// Simulate some work
		for j := range vms {
			vms[j].state["active"] = true
		}
	}

	b.StopTimer()

	memoryPerVM := float64(1024 + 100*16 + 10*32) / 1024 // Approximate KB per VM
	totalMemoryMB := memoryPerVM * float64(vmCount) / 1024

	b.ReportMetric(totalMemoryMB, "total_memory_MB")
	b.ReportMetric(memoryPerVM, "memory_per_vm_KB")
}

// BenchmarkNetworkScalability tests network connection scaling
func BenchmarkNetworkScalability(b *testing.B) {
	scenarios := []struct {
		name        string
		nodeCount   int
		connections int
	}{
		{"10Nodes_FullMesh", 10, 45},
		{"50Nodes_FullMesh", 50, 1225},
		{"100Nodes_Sparse", 100, 500},
		{"500Nodes_Sparse", 500, 2500},
		{"1000Nodes_Sparse", 1000, 5000},
	}

	for _, sc := range scenarios {
		b.Run(sc.name, func(b *testing.B) {
			benchmarkNetworkScalability(b, sc.nodeCount, sc.connections)
		})
	}
}

func benchmarkNetworkScalability(b *testing.B, nodeCount, connectionCount int) {
	b.ReportAllocs()

	var totalMessages int64

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		// Simulate message broadcasting
		var wg sync.WaitGroup

		for j := 0; j < connectionCount; j++ {
			wg.Add(1)
			go func() {
				defer wg.Done()

				// Simulate message send
				time.Sleep(10 * time.Microsecond)
				atomic.AddInt64(&totalMessages, 1)
			}()
		}

		wg.Wait()
	}

	b.StopTimer()

	messagesPerSecond := float64(totalMessages) / b.Elapsed().Seconds()

	b.ReportMetric(messagesPerSecond, "messages/sec")
	b.ReportMetric(float64(connectionCount), "connections")
}

// Helper types and functions

type vmState struct {
	id          int
	memory      []byte
	state       map[string]interface{}
	connections []connection
}

type connection struct {
	sourceID int
	destID   int
	latency  time.Duration
}

func calculateLinearityCoefficient(results map[int]float64) float64 {
	if len(results) < 2 {
		return 0
	}

	// Calculate linear regression coefficient
	var sumX, sumY, sumXY, sumX2 float64
	n := float64(len(results))

	for vmCount, throughput := range results {
		x := float64(vmCount)
		y := throughput

		sumX += x
		sumY += y
		sumXY += x * y
		sumX2 += x * x
	}

	// Calculate R^2 (coefficient of determination)
	meanX := sumX / n
	meanY := sumY / n

	var ssTotal, ssResidual float64
	for vmCount, throughput := range results {
		x := float64(vmCount)
		y := throughput

		// Calculate regression line: y = mx + b
		m := (n*sumXY - sumX*sumY) / (n*sumX2 - sumX*sumX)
		b := meanY - m*meanX

		predicted := m*x + b

		ssTotal += (y - meanY) * (y - meanY)
		ssResidual += (y - predicted) * (y - predicted)
	}

	r2 := 1 - (ssResidual / ssTotal)

	return math.Max(0, math.Min(1, r2))
}

func calculateEfficiencyRetention(results map[int]float64) float64 {
	if len(results) < 2 {
		return 0
	}

	// Get baseline (smallest VM count)
	minVMs := 1000000
	maxVMs := 0
	for vmCount := range results {
		if vmCount < minVMs {
			minVMs = vmCount
		}
		if vmCount > maxVMs {
			maxVMs = vmCount
		}
	}

	baselineThroughput := results[minVMs]
	maxThroughput := results[maxVMs]

	if baselineThroughput == 0 {
		return 0
	}

	// Calculate efficiency retention
	// Perfect scaling: throughput should grow linearly with VM count
	expectedThroughput := baselineThroughput * float64(maxVMs) / float64(minVMs)
	efficiency := (maxThroughput / expectedThroughput) * 100

	return math.Max(0, math.Min(100, efficiency))
}
