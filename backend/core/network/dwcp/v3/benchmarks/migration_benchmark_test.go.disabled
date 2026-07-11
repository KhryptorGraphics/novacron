package benchmarks

import (
	"fmt"
	"math/rand"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// BenchmarkVMMigrationDatacenter tests VM migration in datacenter mode
func BenchmarkVMMigrationDatacenter(b *testing.B) {
	scenarios := []struct {
		name      string
		vmSize    int64 // in MB
		bandwidth int64 // in Mbps
		streams   int
	}{
		{"1GB_10Gbps_8Streams", 1024, 10000, 8},
		{"2GB_10Gbps_8Streams", 2048, 10000, 8},
		{"4GB_10Gbps_16Streams", 4096, 10000, 16},
		{"8GB_25Gbps_32Streams", 8192, 25000, 32},
		{"16GB_25Gbps_32Streams", 16384, 25000, 32},
		{"32GB_40Gbps_64Streams", 32768, 40000, 64},
	}

	for _, sc := range scenarios {
		b.Run(sc.name, func(b *testing.B) {
			benchmarkDatacenterMigration(b, sc.vmSize, sc.bandwidth, sc.streams)
		})
	}
}

func benchmarkDatacenterMigration(b *testing.B, vmSizeMB, bandwidthMbps int64, streams int) {
	b.ReportAllocs()

	vmData := generateVMMemoryData(int(vmSizeMB * 1024 * 1024))

	var totalDowntime time.Duration
	var totalMigrationTime time.Duration
	var totalTransferred int64

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		start := time.Now()

		// Phase 1: Pre-copy iterations (iterative memory copying)
		dirtyPages := len(vmData)
		iterationCount := 0

		for dirtyPages > int(vmSizeMB*1024*10) && iterationCount < 10 { // Until < 10MB dirty
			iterationStart := time.Now()

			// Simulate transferring dirty pages
			transferTime := calculateTransferTime(int64(dirtyPages), bandwidthMbps, streams)
			time.Sleep(transferTime)

			// Simulate VM dirtying pages (10% per iteration)
			dirtyPages = dirtyPages / 10

			iterationCount++
			atomic.AddInt64(&totalTransferred, int64(dirtyPages))

			_ = time.Since(iterationStart)
		}

		// Phase 2: Stop-and-copy (downtime begins)
		downtimeStart := time.Now()

		// Stop VM
		time.Sleep(100 * time.Microsecond)

		// Transfer remaining dirty pages
		transferTime := calculateTransferTime(int64(dirtyPages), bandwidthMbps, streams)
		time.Sleep(transferTime)

		// Transfer device state
		time.Sleep(50 * time.Microsecond)

		// Start VM on destination
		time.Sleep(100 * time.Microsecond)

		downtime := time.Since(downtimeStart)
		totalDowntime += downtime

		migrationTime := time.Since(start)
		totalMigrationTime += migrationTime
	}

	b.StopTimer()

	avgDowntimeMs := float64(totalDowntime.Milliseconds()) / float64(b.N)
	avgMigrationTimeMs := float64(totalMigrationTime.Milliseconds()) / float64(b.N)
	avgThroughputGBps := float64(totalTransferred) / totalMigrationTime.Seconds() / 1e9

	b.ReportMetric(avgDowntimeMs, "downtime_ms")
	b.ReportMetric(avgMigrationTimeMs, "migration_ms")
	b.ReportMetric(avgThroughputGBps, "throughput_GB/s")

	// Validate downtime target: < 500ms
	if avgDowntimeMs >= 500 {
		b.Logf("WARNING: Downtime %.2fms exceeds 500ms target", avgDowntimeMs)
	}
}

// BenchmarkVMMigrationInternet tests VM migration in internet mode
func BenchmarkVMMigrationInternet(b *testing.B) {
	scenarios := []struct {
		name           string
		vmSize         int64 // in MB
		bandwidth      int64 // in Mbps
		compressionAlg string
	}{
		{"1GB_100Mbps_Snappy", 1024, 100, "snappy"},
		{"2GB_100Mbps_Snappy", 2048, 100, "snappy"},
		{"4GB_500Mbps_LZ4", 4096, 500, "lz4"},
		{"8GB_1000Mbps_LZ4", 8192, 1000, "lz4"},
		{"1GB_100Mbps_ZSTD", 1024, 100, "zstd"},
		{"4GB_500Mbps_ZSTD", 4096, 500, "zstd"},
	}

	for _, sc := range scenarios {
		b.Run(sc.name, func(b *testing.B) {
			benchmarkInternetMigration(b, sc.vmSize, sc.bandwidth, sc.compressionAlg)
		})
	}
}

func benchmarkInternetMigration(b *testing.B, vmSizeMB, bandwidthMbps int64, compressionAlg string) {
	b.ReportAllocs()

	vmData := generateVMMemoryData(int(vmSizeMB * 1024 * 1024))

	var totalMigrationTime time.Duration
	var totalOriginalSize int64
	var totalCompressedSize int64

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		start := time.Now()

		// Phase 1: Initial state transfer with compression
		compressedData, compressionTime := compressWithTiming(vmData, compressionAlg)
		transferTime := calculateTransferTime(int64(len(compressedData)), bandwidthMbps, 8)

		time.Sleep(compressionTime + transferTime)

		// Phase 2: Incremental dirty page transfers
		iterations := 5
		for iter := 0; iter < iterations; iter++ {
			// Simulate dirty pages (decreasing each iteration)
			dirtySize := len(vmData) / (10 * (iter + 1))
			dirtyData := vmData[:dirtySize]

			compressed, compTime := compressWithTiming(dirtyData, compressionAlg)
			transferTime = calculateTransferTime(int64(len(compressed)), bandwidthMbps, 4)

			time.Sleep(compTime + transferTime)

			atomic.AddInt64(&totalOriginalSize, int64(dirtySize))
			atomic.AddInt64(&totalCompressedSize, int64(len(compressed)))
		}

		// Phase 3: Final sync (downtime)
		finalDirtySize := len(vmData) / 100 // 1% dirty
		finalDirty := vmData[:finalDirtySize]

		compressed, compTime := compressWithTiming(finalDirty, compressionAlg)
		transferTime = calculateTransferTime(int64(len(compressed)), bandwidthMbps, 4)

		time.Sleep(compTime + transferTime)

		// Add VM restart time
		time.Sleep(500 * time.Millisecond)

		migrationTime := time.Since(start)
		totalMigrationTime += migrationTime

		atomic.AddInt64(&totalOriginalSize, int64(len(vmData))+int64(finalDirtySize))
		atomic.AddInt64(&totalCompressedSize, int64(len(compressedData))+int64(len(compressed)))
	}

	b.StopTimer()

	avgMigrationTimeSec := float64(totalMigrationTime.Seconds()) / float64(b.N)
	compressionRatio := (1 - float64(totalCompressedSize)/float64(totalOriginalSize)) * 100
	effectiveThroughputMBps := float64(totalOriginalSize) / totalMigrationTime.Seconds() / 1e6

	b.ReportMetric(avgMigrationTimeSec, "migration_sec")
	b.ReportMetric(compressionRatio, "compression_%")
	b.ReportMetric(effectiveThroughputMBps, "effective_MB/s")

	// Validate compression target: 75-85%
	if compressionRatio < 75 || compressionRatio > 85 {
		b.Logf("WARNING: Compression ratio %.2f%% outside 75-85%% target range", compressionRatio)
	}
}

// BenchmarkConcurrentMigrations tests scalability of concurrent migrations
func BenchmarkConcurrentMigrations(b *testing.B) {
	scenarios := []struct {
		name           string
		vmCount        int
		vmSizeMB       int64
		mode           string
	}{
		{"1VM_4GB_Datacenter", 1, 4096, "datacenter"},
		{"2VMs_4GB_Datacenter", 2, 4096, "datacenter"},
		{"5VMs_4GB_Datacenter", 5, 4096, "datacenter"},
		{"10VMs_4GB_Datacenter", 10, 4096, "datacenter"},
		{"1VM_2GB_Internet", 1, 2048, "internet"},
		{"2VMs_2GB_Internet", 2, 2048, "internet"},
		{"5VMs_2GB_Internet", 5, 2048, "internet"},
	}

	for _, sc := range scenarios {
		b.Run(sc.name, func(b *testing.B) {
			benchmarkConcurrentMigrations(b, sc.vmCount, sc.vmSizeMB, sc.mode)
		})
	}
}

func benchmarkConcurrentMigrations(b *testing.B, vmCount int, vmSizeMB int64, mode string) {
	b.ReportAllocs()

	var totalTime time.Duration
	var completedMigrations int64

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		start := time.Now()

		var wg sync.WaitGroup
		for j := 0; j < vmCount; j++ {
			wg.Add(1)
			go func(vmID int) {
				defer wg.Done()

				vmData := generateVMMemoryData(int(vmSizeMB * 1024 * 1024))

				if mode == "datacenter" {
					// Fast datacenter migration
					transferTime := calculateTransferTime(int64(len(vmData)), 10000, 16)
					time.Sleep(transferTime)
				} else {
					// Internet migration with compression
					compressed, compTime := compressWithTiming(vmData, "snappy")
					transferTime := calculateTransferTime(int64(len(compressed)), 100, 8)
					time.Sleep(compTime + transferTime)
				}

				atomic.AddInt64(&completedMigrations, 1)
			}(j)
		}

		wg.Wait()

		totalTime += time.Since(start)
	}

	b.StopTimer()

	avgTimePerBatchSec := float64(totalTime.Seconds()) / float64(b.N)
	migrationsPerSecond := float64(completedMigrations) / totalTime.Seconds()

	b.ReportMetric(avgTimePerBatchSec, "batch_time_sec")
	b.ReportMetric(migrationsPerSecond, "migrations/sec")
	b.ReportMetric(float64(vmCount), "concurrent_vms")
}

// BenchmarkMigrationDirtyPageTracking tests dirty page tracking performance
func BenchmarkMigrationDirtyPageTracking(b *testing.B) {
	scenarios := []struct {
		name       string
		vmSizeMB   int64
		dirtyRate  float64 // percentage of pages dirtied per iteration
	}{
		{"4GB_1%Dirty", 4096, 0.01},
		{"4GB_5%Dirty", 4096, 0.05},
		{"4GB_10%Dirty", 4096, 0.10},
		{"8GB_1%Dirty", 8192, 0.01},
		{"8GB_5%Dirty", 8192, 0.05},
		{"8GB_10%Dirty", 8192, 0.10},
	}

	for _, sc := range scenarios {
		b.Run(sc.name, func(b *testing.B) {
			benchmarkDirtyPageTracking(b, sc.vmSizeMB, sc.dirtyRate)
		})
	}
}

func benchmarkDirtyPageTracking(b *testing.B, vmSizeMB int64, dirtyRate float64) {
	b.ReportAllocs()

	pageSize := 4096 // 4KB pages
	totalPages := int(vmSizeMB * 1024 * 1024 / int64(pageSize))
	dirtyBitmap := make([]bool, totalPages)

	var trackingOperations int64

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		// Simulate marking pages as dirty
		dirtyCount := int(float64(totalPages) * dirtyRate)

		for j := 0; j < dirtyCount; j++ {
			pageIdx := rand.Intn(totalPages)
			dirtyBitmap[pageIdx] = true
			atomic.AddInt64(&trackingOperations, 1)
		}

		// Simulate scanning dirty bitmap
		dirtyPageCount := 0
		for _, dirty := range dirtyBitmap {
			if dirty {
				dirtyPageCount++
			}
		}

		// Clear dirty bitmap
		for j := range dirtyBitmap {
			dirtyBitmap[j] = false
		}
	}

	b.StopTimer()

	operationsPerSecond := float64(trackingOperations) / b.Elapsed().Seconds()

	b.ReportMetric(operationsPerSecond, "ops/sec")
	b.ReportMetric(float64(totalPages), "total_pages")
}

// BenchmarkMigrationCheckpointing tests checkpointing performance
func BenchmarkMigrationCheckpointing(b *testing.B) {
	vmSizes := []int64{1024, 2048, 4096, 8192} // MB

	for _, size := range vmSizes {
		b.Run(fmt.Sprintf("%dMB", size), func(b *testing.B) {
			benchmarkCheckpointing(b, size)
		})
	}
}

func benchmarkCheckpointing(b *testing.B, vmSizeMB int64) {
	b.ReportAllocs()

	vmData := generateVMMemoryData(int(vmSizeMB * 1024 * 1024))

	var totalCheckpointTime time.Duration
	var checkpointCount int64

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		start := time.Now()

		// Simulate creating checkpoint
		// 1. Pause VM
		time.Sleep(50 * time.Microsecond)

		// 2. Snapshot memory state
		checkpoint := make([]byte, len(vmData)/10) // Incremental checkpoint
		copy(checkpoint, vmData[:len(checkpoint)])

		// 3. Compress checkpoint
		_, compTime := compressWithTiming(checkpoint, "snappy")
		time.Sleep(compTime)

		// 4. Resume VM
		time.Sleep(50 * time.Microsecond)

		totalCheckpointTime += time.Since(start)
		atomic.AddInt64(&checkpointCount, 1)
	}

	b.StopTimer()

	avgCheckpointTimeMs := float64(totalCheckpointTime.Milliseconds()) / float64(b.N)
	checkpointsPerSecond := float64(checkpointCount) / b.Elapsed().Seconds()

	b.ReportMetric(avgCheckpointTimeMs, "ms/checkpoint")
	b.ReportMetric(checkpointsPerSecond, "checkpoints/sec")
}

// BenchmarkMigrationRollback tests rollback performance
func BenchmarkMigrationRollback(b *testing.B) {
	scenarios := []struct {
		name         string
		vmSizeMB     int64
		failureStage string
	}{
		{"4GB_EarlyFailure", 4096, "early"},
		{"4GB_MidFailure", 4096, "mid"},
		{"4GB_LateFailure", 4096, "late"},
		{"8GB_EarlyFailure", 8192, "early"},
		{"8GB_MidFailure", 8192, "mid"},
		{"8GB_LateFailure", 8192, "late"},
	}

	for _, sc := range scenarios {
		b.Run(sc.name, func(b *testing.B) {
			benchmarkMigrationRollback(b, sc.vmSizeMB, sc.failureStage)
		})
	}
}

func benchmarkMigrationRollback(b *testing.B, vmSizeMB int64, failureStage string) {
	b.ReportAllocs()

	vmData := generateVMMemoryData(int(vmSizeMB * 1024 * 1024))

	var totalRollbackTime time.Duration
	var rollbackCount int64

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		start := time.Now()

		// Simulate migration progress before failure
		switch failureStage {
		case "early":
			time.Sleep(100 * time.Millisecond)
		case "mid":
			time.Sleep(500 * time.Millisecond)
		case "late":
			time.Sleep(2 * time.Second)
		}

		// Simulate failure detection
		time.Sleep(50 * time.Millisecond)

		// Rollback operations
		rollbackStart := time.Now()

		// 1. Stop destination VM
		time.Sleep(100 * time.Microsecond)

		// 2. Cleanup transferred state
		time.Sleep(time.Duration(len(vmData)/10000) * time.Microsecond)

		// 3. Resume source VM
		time.Sleep(100 * time.Microsecond)

		rollbackTime := time.Since(rollbackStart)

		totalRollbackTime += rollbackTime
		atomic.AddInt64(&rollbackCount, 1)

		_ = time.Since(start)
	}

	b.StopTimer()

	avgRollbackTimeMs := float64(totalRollbackTime.Milliseconds()) / float64(b.N)
	rollbacksPerSecond := float64(rollbackCount) / b.Elapsed().Seconds()

	b.ReportMetric(avgRollbackTimeMs, "rollback_ms")
	b.ReportMetric(rollbacksPerSecond, "rollbacks/sec")
}

// Helper functions

func calculateTransferTime(dataSizeBytes, bandwidthMbps int64, streams int) time.Duration {
	// Calculate effective bandwidth with parallelism
	effectiveBandwidth := float64(bandwidthMbps) * (1 + 0.1*float64(streams-1)) // 10% boost per stream

	// Convert to bytes per second
	bytesPerSecond := effectiveBandwidth * 1000000 / 8

	// Calculate transfer time
	transferSeconds := float64(dataSizeBytes) / bytesPerSecond

	return time.Duration(transferSeconds * float64(time.Second))
}

func compressWithTiming(data []byte, algorithm string) ([]byte, time.Duration) {
	start := time.Now()

	compressed, _ := compressData(data, algorithm)

	duration := time.Since(start)

	// Simulate compression time if too fast
	minCompressionTime := time.Duration(len(data)/1000000) * time.Microsecond // 1 us per MB
	if duration < minCompressionTime {
		duration = minCompressionTime
	}

	return compressed, duration
}

// Note: generateVMMemoryData and compressData are defined in hde_benchmark_test.go
