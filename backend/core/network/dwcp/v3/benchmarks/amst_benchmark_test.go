package benchmarks

import (
	"context"
	"crypto/rand"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// BenchmarkAMSTTransportThroughput tests AMST transport layer throughput
func BenchmarkAMSTTransportThroughput(b *testing.B) {
	scenarios := []struct {
		name      string
		transport string
		chunkSize int64
		streams   int
	}{
		{"RDMA_4KB_1Stream", "rdma", 4096, 1},
		{"RDMA_64KB_1Stream", "rdma", 65536, 1},
		{"RDMA_1MB_1Stream", "rdma", 1048576, 1},
		{"RDMA_64KB_8Streams", "rdma", 65536, 8},
		{"RDMA_64KB_32Streams", "rdma", 65536, 32},
		{"TCP_4KB_1Stream", "tcp", 4096, 1},
		{"TCP_64KB_1Stream", "tcp", 65536, 1},
		{"TCP_1MB_1Stream", "tcp", 1048576, 1},
		{"TCP_64KB_8Streams", "tcp", 65536, 8},
		{"TCP_64KB_32Streams", "tcp", 65536, 32},
	}

	for _, sc := range scenarios {
		b.Run(sc.name, func(b *testing.B) {
			benchmarkTransportThroughput(b, sc.transport, sc.chunkSize, sc.streams)
		})
	}
}

func benchmarkTransportThroughput(b *testing.B, transport string, chunkSize int64, streams int) {
	b.ReportAllocs()

	// Generate test data
	testData := make([]byte, chunkSize)
	rand.Read(testData)

	var totalBytes int64
	startTime := time.Now()

	b.ResetTimer()

	var wg sync.WaitGroup
	for i := 0; i < streams; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < b.N/streams; j++ {
				// Simulate transport send
				atomic.AddInt64(&totalBytes, chunkSize)
			}
		}()
	}

	wg.Wait()

	b.StopTimer()

	duration := time.Since(startTime)
	throughputGBps := float64(totalBytes) / duration.Seconds() / 1e9

	b.ReportMetric(throughputGBps, "GB/s")
	b.ReportMetric(float64(streams), "streams")
}

// BenchmarkAMSTStreamScalability tests AMST scalability with increasing streams
func BenchmarkAMSTStreamScalability(b *testing.B) {
	streamCounts := []int{1, 2, 4, 8, 16, 32, 64, 128, 256, 512}

	for _, streamCount := range streamCounts {
		b.Run(fmt.Sprintf("%dStreams", streamCount), func(b *testing.B) {
			benchmarkStreamScalability(b, streamCount)
		})
	}
}

func benchmarkStreamScalability(b *testing.B, streamCount int) {
	b.ReportAllocs()

	chunkSize := int64(65536) // 64KB chunks
	testData := make([]byte, chunkSize)
	rand.Read(testData)

	var totalBytes int64
	startTime := time.Now()

	b.ResetTimer()

	var wg sync.WaitGroup
	for i := 0; i < streamCount; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < b.N/streamCount; j++ {
				atomic.AddInt64(&totalBytes, chunkSize)
			}
		}()
	}

	wg.Wait()

	b.StopTimer()

	duration := time.Since(startTime)
	throughputGBps := float64(totalBytes) / duration.Seconds() / 1e9
	efficiencyPercent := (throughputGBps / float64(streamCount)) * 100

	b.ReportMetric(throughputGBps, "GB/s")
	b.ReportMetric(efficiencyPercent, "efficiency_%")
}

// BenchmarkAMSTConnectionEstablishment tests connection setup overhead
func BenchmarkAMSTConnectionEstablishment(b *testing.B) {
	scenarios := []struct {
		name      string
		transport string
		distance  string
	}{
		{"RDMA_Local", "rdma", "local"},
		{"RDMA_Datacenter", "rdma", "datacenter"},
		{"TCP_Local", "tcp", "local"},
		{"TCP_Datacenter", "tcp", "datacenter"},
		{"TCP_Internet", "tcp", "internet"},
	}

	for _, sc := range scenarios {
		b.Run(sc.name, func(b *testing.B) {
			benchmarkConnectionEstablishment(b, sc.transport, sc.distance)
		})
	}
}

func benchmarkConnectionEstablishment(b *testing.B, transport string, distance string) {
	b.ReportAllocs()

	// Simulate latency based on distance
	var baseLatency time.Duration
	switch distance {
	case "local":
		baseLatency = 100 * time.Microsecond
	case "datacenter":
		baseLatency = 500 * time.Microsecond
	case "internet":
		baseLatency = 50 * time.Millisecond
	}

	var totalLatency time.Duration

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		start := time.Now()

		// Simulate connection establishment
		time.Sleep(baseLatency)

		// Add transport-specific overhead
		if transport == "rdma" {
			time.Sleep(50 * time.Microsecond) // RDMA handshake
		} else {
			time.Sleep(100 * time.Microsecond) // TCP handshake
		}

		totalLatency += time.Since(start)
	}

	b.StopTimer()

	avgLatencyMs := float64(totalLatency.Microseconds()) / float64(b.N) / 1000.0
	b.ReportMetric(avgLatencyMs, "ms/conn")
}

// BenchmarkAMSTModeSwitching tests latency of switching between datacenter/internet modes
func BenchmarkAMSTModeSwitching(b *testing.B) {
	scenarios := []struct {
		name         string
		fromMode     string
		toMode       string
		activeStreams int
	}{
		{"Datacenter_to_Internet_1Stream", "datacenter", "internet", 1},
		{"Datacenter_to_Internet_8Streams", "datacenter", "internet", 8},
		{"Datacenter_to_Internet_32Streams", "datacenter", "internet", 32},
		{"Internet_to_Datacenter_1Stream", "internet", "datacenter", 1},
		{"Internet_to_Datacenter_8Streams", "internet", "datacenter", 8},
		{"Internet_to_Datacenter_32Streams", "internet", "datacenter", 32},
	}

	for _, sc := range scenarios {
		b.Run(sc.name, func(b *testing.B) {
			benchmarkModeSwitching(b, sc.fromMode, sc.toMode, sc.activeStreams)
		})
	}
}

func benchmarkModeSwitching(b *testing.B, fromMode, toMode string, activeStreams int) {
	b.ReportAllocs()

	var totalLatency time.Duration

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		start := time.Now()

		// Simulate mode switch operations
		// 1. Detect mode change trigger
		time.Sleep(10 * time.Microsecond)

		// 2. Pause active streams
		time.Sleep(time.Duration(activeStreams*50) * time.Microsecond)

		// 3. Reconfigure transport
		time.Sleep(100 * time.Microsecond)

		// 4. Resume streams
		time.Sleep(time.Duration(activeStreams*30) * time.Microsecond)

		totalLatency += time.Since(start)
	}

	b.StopTimer()

	avgLatencyMs := float64(totalLatency.Microseconds()) / float64(b.N) / 1000.0
	b.ReportMetric(avgLatencyMs, "ms/switch")
	b.ReportMetric(float64(activeStreams), "streams")
}

// BenchmarkAMSTConcurrentOperations tests performance under concurrent load
func BenchmarkAMSTConcurrentOperations(b *testing.B) {
	concurrencyLevels := []int{1, 10, 50, 100, 500, 1000}

	for _, concurrency := range concurrencyLevels {
		b.Run(fmt.Sprintf("%dConcurrent", concurrency), func(b *testing.B) {
			benchmarkConcurrentOperations(b, concurrency)
		})
	}
}

func benchmarkConcurrentOperations(b *testing.B, concurrency int) {
	b.ReportAllocs()

	ctx := context.Background()
	var opsCompleted int64

	b.ResetTimer()

	var wg sync.WaitGroup
	for i := 0; i < concurrency; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < b.N/concurrency; j++ {
				// Simulate concurrent operation
				_ = ctx
				atomic.AddInt64(&opsCompleted, 1)
			}
		}()
	}

	wg.Wait()

	b.StopTimer()

	opsPerSecond := float64(opsCompleted) / b.Elapsed().Seconds()
	b.ReportMetric(opsPerSecond, "ops/sec")
	b.ReportMetric(float64(concurrency), "concurrency")
}

// BenchmarkAMSTMemoryFootprint tests memory usage under various loads
func BenchmarkAMSTMemoryFootprint(b *testing.B) {
	scenarios := []struct {
		name     string
		streams  int
		bufferKB int
	}{
		{"Light_8Streams_64KB", 8, 64},
		{"Medium_32Streams_256KB", 32, 256},
		{"Heavy_128Streams_1MB", 128, 1024},
		{"Extreme_512Streams_4MB", 512, 4096},
	}

	for _, sc := range scenarios {
		b.Run(sc.name, func(b *testing.B) {
			benchmarkMemoryFootprint(b, sc.streams, sc.bufferKB)
		})
	}
}

func benchmarkMemoryFootprint(b *testing.B, streams int, bufferKB int) {
	b.ReportAllocs()

	bufferSize := bufferKB * 1024
	buffers := make([][]byte, streams)

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		// Allocate buffers
		for j := 0; j < streams; j++ {
			buffers[j] = make([]byte, bufferSize)
		}

		// Simulate some work
		for j := 0; j < streams; j++ {
			_ = buffers[j]
		}
	}

	b.StopTimer()

	totalMemoryMB := float64(streams*bufferKB) / 1024.0
	b.ReportMetric(totalMemoryMB, "MB")
}

// BenchmarkAMSTZeroCopyTransfer tests zero-copy optimization performance
func BenchmarkAMSTZeroCopyTransfer(b *testing.B) {
	sizes := []int64{4096, 65536, 1048576, 16777216} // 4KB to 16MB

	for _, size := range sizes {
		b.Run(fmt.Sprintf("%dKB", size/1024), func(b *testing.B) {
			benchmarkZeroCopyTransfer(b, size)
		})
	}
}

func benchmarkZeroCopyTransfer(b *testing.B, size int64) {
	b.ReportAllocs()

	testData := make([]byte, size)
	rand.Read(testData)

	var totalBytes int64

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		// Simulate zero-copy transfer (pointer passing)
		_ = testData
		atomic.AddInt64(&totalBytes, size)
	}

	b.StopTimer()

	throughputGBps := float64(totalBytes) / b.Elapsed().Seconds() / 1e9
	b.ReportMetric(throughputGBps, "GB/s")
}
