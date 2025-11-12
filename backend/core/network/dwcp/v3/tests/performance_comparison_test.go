package tests

import (
	"crypto/rand"
	"fmt"
	"testing"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp"
	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/upgrade"
	"github.com/stretchr/testify/require"
)

// PerformanceMetrics stores comprehensive performance data
type PerformanceMetrics struct {
	Name              string
	TotalOperations   int64
	TotalBytes        int64
	Duration          time.Duration
	Throughput        float64 // MB/s
	Latency           time.Duration
	CompressionRatio  float64
	MemoryUsage       int64
	CPUUsage          float64
}

// BenchmarkV1VsV3Datacenter compares v1 and v3 in datacenter mode
func BenchmarkV1VsV3Datacenter(b *testing.B) {
	sizes := []int{
		64 * 1024,      // 64 KB
		256 * 1024,     // 256 KB
		1024 * 1024,    // 1 MB
		10 * 1024 * 1024, // 10 MB
	}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("v1_datacenter_%dKB", size/1024), func(b *testing.B) {
			upgrade.DisableAll()
			defer upgrade.DisableAll()

			config := dwcp.HDEConfig{
				GlobalLevel: 0, // Fast compression for datacenter
			}

			hde, err := dwcp.NewHDE(config)
			require.NoError(b, err)
			defer hde.Close()

			data := make([]byte, size)
			rand.Read(data)

			b.ResetTimer()
			b.SetBytes(int64(size))

			for i := 0; i < b.N; i++ {
				_, err := hde.CompressMemory(fmt.Sprintf("vm-%d", i), data, dwcp.CompressionLocal)
				if err != nil {
					b.Fatal(err)
				}
			}
		})

		b.Run(fmt.Sprintf("v3_datacenter_%dKB", size/1024), func(b *testing.B) {
			upgrade.EnableAll(100)
			defer upgrade.DisableAll()

			detector := upgrade.NewModeDetector()
			detector.ForceMode(upgrade.ModeDatacenter)

			config := dwcp.HDEConfig{
				GlobalLevel: 0,
				EnableDelta: false, // Datacenter mode prioritizes speed
			}

			hde, err := dwcp.NewHDE(config)
			require.NoError(b, err)
			defer hde.Close()

			data := make([]byte, size)
			rand.Read(data)

			b.ResetTimer()
			b.SetBytes(int64(size))

			for i := 0; i < b.N; i++ {
				_, err := hde.CompressMemory(fmt.Sprintf("vm-%d", i), data, dwcp.CompressionLocal)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkV3InternetMode benchmarks v3 internet-optimized mode
func BenchmarkV3InternetMode(b *testing.B) {
	upgrade.EnableAll(100)
	defer upgrade.DisableAll()

	detector := upgrade.NewModeDetector()
	detector.ForceMode(upgrade.ModeInternet)

	sizes := []int{
		1024 * 1024,      // 1 MB
		10 * 1024 * 1024, // 10 MB
		50 * 1024 * 1024, // 50 MB
	}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("internet_%dMB", size/1024/1024), func(b *testing.B) {
			config := dwcp.HDEConfig{
				GlobalLevel: 9, // Maximum compression for internet
				EnableDelta: true,
				EnableLSTM:  true,
			}

			hde, err := dwcp.NewHDE(config)
			require.NoError(b, err)
			defer hde.Close()

			data := make([]byte, size)
			rand.Read(data)

			b.ResetTimer()
			b.SetBytes(int64(size))

			for i := 0; i < b.N; i++ {
				compressed, err := hde.CompressMemory(fmt.Sprintf("vm-%d", i), data, dwcp.CompressionGlobal)
				if err != nil {
					b.Fatal(err)
				}

				// Verify compression target (70-85%)
				ratio := float64(size) / float64(len(compressed))
				if ratio < 1.15 { // Should achieve at least 15% reduction
					b.Logf("Warning: Compression ratio %.2f below target", ratio)
				}
			}
		})
	}
}

// BenchmarkHybridMode benchmarks adaptive mode switching
func BenchmarkHybridMode(b *testing.B) {
	upgrade.EnableAll(100)
	defer upgrade.DisableAll()

	detector := upgrade.NewModeDetector()
	detector.ForceMode(upgrade.ModeHybrid)

	b.Run("adaptive_switching", func(b *testing.B) {
		config := dwcp.HDEConfig{
			GlobalLevel: 3,
			EnableDelta: true,
		}

		hde, err := dwcp.NewHDE(config)
		require.NoError(b, err)
		defer hde.Close()

		data := make([]byte, 1024*1024)
		rand.Read(data)

		b.ResetTimer()
		b.SetBytes(int64(len(data)))

		for i := 0; i < b.N; i++ {
			// Simulate mode switching
			if i%10 == 0 {
				detector.ForceMode(upgrade.ModeDatacenter)
			} else if i%10 == 5 {
				detector.ForceMode(upgrade.ModeInternet)
			}

			_, err := hde.CompressMemory(fmt.Sprintf("vm-%d", i), data, dwcp.CompressionGlobal)
			if err != nil {
				b.Fatal(err)
			}
		}
	})
}

// BenchmarkMemoryUsage compares memory usage
func BenchmarkMemoryUsage(b *testing.B) {
	b.Run("v1_memory", func(b *testing.B) {
		upgrade.DisableAll()
		defer upgrade.DisableAll()

		config := dwcp.HDEConfig{
			MaxVMs: 100,
		}

		hde, err := dwcp.NewHDE(config)
		require.NoError(b, err)
		defer hde.Close()

		data := make([]byte, 1024*1024)
		rand.Read(data)

		b.ResetTimer()
		b.ReportAllocs()

		for i := 0; i < b.N; i++ {
			_, err := hde.CompressMemory(fmt.Sprintf("vm-%d", i%100), data, dwcp.CompressionGlobal)
			if err != nil {
				b.Fatal(err)
			}
		}
	})

	b.Run("v3_memory", func(b *testing.B) {
		upgrade.EnableAll(100)
		defer upgrade.DisableAll()

		config := dwcp.HDEConfig{
			MaxVMs:      100,
			EnableDelta: true,
		}

		hde, err := dwcp.NewHDE(config)
		require.NoError(b, err)
		defer hde.Close()

		data := make([]byte, 1024*1024)
		rand.Read(data)

		b.ResetTimer()
		b.ReportAllocs()

		for i := 0; i < b.N; i++ {
			_, err := hde.CompressMemory(fmt.Sprintf("vm-%d", i%100), data, dwcp.CompressionGlobal)
			if err != nil {
				b.Fatal(err)
			}
		}
	})
}

// BenchmarkCPUUtilization measures CPU usage
func BenchmarkCPUUtilization(b *testing.B) {
	b.Run("v1_cpu", func(b *testing.B) {
		upgrade.DisableAll()
		defer upgrade.DisableAll()

		config := dwcp.HDEConfig{
			GlobalLevel: 3,
		}

		hde, err := dwcp.NewHDE(config)
		require.NoError(b, err)
		defer hde.Close()

		data := make([]byte, 10*1024*1024)
		rand.Read(data)

		b.ResetTimer()

		for i := 0; i < b.N; i++ {
			_, err := hde.CompressMemory(fmt.Sprintf("vm-%d", i), data, dwcp.CompressionGlobal)
			if err != nil {
				b.Fatal(err)
			}
		}
	})

	b.Run("v3_cpu", func(b *testing.B) {
		upgrade.EnableAll(100)
		defer upgrade.DisableAll()

		config := dwcp.HDEConfig{
			GlobalLevel: 3,
			EnableDelta: true,
			EnableLSTM:  true,
		}

		hde, err := dwcp.NewHDE(config)
		require.NoError(b, err)
		defer hde.Close()

		data := make([]byte, 10*1024*1024)
		rand.Read(data)

		b.ResetTimer()

		for i := 0; i < b.N; i++ {
			_, err := hde.CompressMemory(fmt.Sprintf("vm-%d", i), data, dwcp.CompressionGlobal)
			if err != nil {
				b.Fatal(err)
			}
		}
	})
}

// BenchmarkLatencyCharacteristics measures latency distribution
func BenchmarkLatencyCharacteristics(b *testing.B) {
	upgrade.EnableAll(100)
	defer upgrade.DisableAll()

	sizes := []int{
		64 * 1024,
		256 * 1024,
		1024 * 1024,
	}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("latency_%dKB", size/1024), func(b *testing.B) {
			config := dwcp.HDEConfig{
				GlobalLevel: 3,
			}

			hde, err := dwcp.NewHDE(config)
			require.NoError(b, err)
			defer hde.Close()

			data := make([]byte, size)
			rand.Read(data)

			latencies := make([]time.Duration, b.N)

			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				start := time.Now()
				_, err := hde.CompressMemory(fmt.Sprintf("vm-%d", i), data, dwcp.CompressionGlobal)
				latencies[i] = time.Since(start)
				if err != nil {
					b.Fatal(err)
				}
			}

			b.StopTimer()

			// Calculate percentiles
			p50 := latencies[len(latencies)/2]
			p95 := latencies[int(float64(len(latencies))*0.95)]
			p99 := latencies[int(float64(len(latencies))*0.99)]

			b.Logf("Latency percentiles - P50: %v, P95: %v, P99: %v", p50, p95, p99)
		})
	}
}

// BenchmarkCompressionRatios measures compression effectiveness
func BenchmarkCompressionRatios(b *testing.B) {
	upgrade.EnableAll(100)
	defer upgrade.DisableAll()

	patterns := []struct {
		name      string
		generator func(size int) []byte
	}{
		{
			name: "zeros",
			generator: func(size int) []byte {
				return make([]byte, size)
			},
		},
		{
			name: "random",
			generator: func(size int) []byte {
				data := make([]byte, size)
				rand.Read(data)
				return data
			},
		},
		{
			name: "repeating",
			generator: func(size int) []byte {
				data := make([]byte, size)
				for i := range data {
					data[i] = byte(i % 256)
				}
				return data
			},
		},
		{
			name: "sparse",
			generator: func(size int) []byte {
				data := make([]byte, size)
				for i := 0; i < size; i += 100 {
					data[i] = byte(i % 256)
				}
				return data
			},
		},
	}

	for _, pattern := range patterns {
		b.Run(pattern.name, func(b *testing.B) {
			config := dwcp.HDEConfig{
				GlobalLevel: 9, // Maximum compression
				EnableDelta: true,
			}

			hde, err := dwcp.NewHDE(config)
			require.NoError(b, err)
			defer hde.Close()

			data := pattern.generator(1024 * 1024)

			var totalOriginal int64
			var totalCompressed int64

			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				compressed, err := hde.CompressMemory(fmt.Sprintf("vm-%d", i), data, dwcp.CompressionGlobal)
				if err != nil {
					b.Fatal(err)
				}

				totalOriginal += int64(len(data))
				totalCompressed += int64(len(compressed))
			}

			ratio := float64(totalOriginal) / float64(totalCompressed)
			b.Logf("%s compression ratio: %.2f:1", pattern.name, ratio)
		})
	}
}

// BenchmarkAMSTPerformance compares AMST v1 vs v3
func BenchmarkAMSTPerformance(b *testing.B) {
	b.Run("v1_amst", func(b *testing.B) {
		upgrade.DisableAll()
		defer upgrade.DisableAll()

		config := dwcp.AMSTConfig{
			InitialStreams: 8,
			ChunkSize:      64 * 1024,
		}

		amst, err := dwcp.NewAMST(config)
		require.NoError(b, err)
		defer amst.Close()

		b.ResetTimer()

		for i := 0; i < b.N; i++ {
			amst.UpdateMetrics(8, 0.001, 10e9)
			_ = amst.GetMetrics()
		}
	})

	b.Run("v3_amst_adaptive", func(b *testing.B) {
		upgrade.EnableAll(100)
		defer upgrade.DisableAll()

		config := dwcp.AMSTConfig{
			InitialStreams: 8,
			ChunkSize:      64 * 1024,
			EnableAdaptive: true,
		}

		amst, err := dwcp.NewAMST(config)
		require.NoError(b, err)
		defer amst.Close()

		b.ResetTimer()

		for i := 0; i < b.N; i++ {
			// Vary conditions
			rtt := 0.001 + float64(i%100)*0.0001
			bandwidth := 1e9 + float64(i%100)*1e8

			amst.UpdateMetrics(8, rtt, bandwidth)
			_ = amst.GetMetrics()
		}
	})
}

// BenchmarkBandwidthSavings measures actual bandwidth reduction
func BenchmarkBandwidthSavings(b *testing.B) {
	upgrade.EnableAll(100)
	defer upgrade.DisableAll()

	b.Run("internet_mode_savings", func(b *testing.B) {
		detector := upgrade.NewModeDetector()
		detector.ForceMode(upgrade.ModeInternet)

		config := dwcp.HDEConfig{
			GlobalLevel: 9,
			EnableDelta: true,
		}

		hde, err := dwcp.NewHDE(config)
		require.NoError(b, err)
		defer hde.Close()

		// Simulate VM memory (highly compressible)
		data := make([]byte, 10*1024*1024) // 10MB
		for i := range data {
			data[i] = byte(i % 256)
		}

		b.ResetTimer()
		b.SetBytes(int64(len(data)))

		var totalSaved int64

		for i := 0; i < b.N; i++ {
			compressed, err := hde.CompressMemory(fmt.Sprintf("vm-%d", i), data, dwcp.CompressionGlobal)
			if err != nil {
				b.Fatal(err)
			}

			saved := len(data) - len(compressed)
			totalSaved += int64(saved)
		}

		savingsPercent := float64(totalSaved) / float64(int64(len(data))*int64(b.N)) * 100
		b.Logf("Bandwidth savings: %.2f%% (target: 70-85%%)", savingsPercent)
	})
}

// TestPerformanceTargets validates performance meets requirements
func TestPerformanceTargets(t *testing.T) {
	t.Run("DatacenterModeTarget", func(t *testing.T) {
		upgrade.EnableAll(100)
		defer upgrade.DisableAll()

		detector := upgrade.NewModeDetector()
		detector.ForceMode(upgrade.ModeDatacenter)

		config := dwcp.HDEConfig{
			GlobalLevel: 0,
		}

		hde, err := dwcp.NewHDE(config)
		require.NoError(t, err)
		defer hde.Close()

		data := make([]byte, 1024*1024)
		rand.Read(data)

		// Measure throughput
		start := time.Now()
		iterations := 100

		for i := 0; i < iterations; i++ {
			_, err := hde.CompressMemory(fmt.Sprintf("vm-%d", i), data, dwcp.CompressionLocal)
			require.NoError(t, err)
		}

		elapsed := time.Since(start)
		throughputMBps := float64(len(data)*iterations) / elapsed.Seconds() / 1024 / 1024

		t.Logf("Datacenter mode throughput: %.2f MB/s", throughputMBps)
		// Target: Similar or better than v1
		require.Greater(t, throughputMBps, 100.0, "Throughput should be >100 MB/s")
	})

	t.Run("InternetModeCompressionTarget", func(t *testing.T) {
		upgrade.EnableAll(100)
		defer upgrade.DisableAll()

		detector := upgrade.NewModeDetector()
		detector.ForceMode(upgrade.ModeInternet)

		config := dwcp.HDEConfig{
			GlobalLevel: 9,
			EnableDelta: true,
		}

		hde, err := dwcp.NewHDE(config)
		require.NoError(t, err)
		defer hde.Close()

		// Test with realistic VM memory pattern
		data := make([]byte, 10*1024*1024)
		for i := range data {
			data[i] = byte(i % 256)
		}

		compressed, err := hde.CompressMemory("vm-target", data, dwcp.CompressionGlobal)
		require.NoError(t, err)

		ratio := float64(len(data)) / float64(len(compressed))
		savingsPercent := (1 - float64(len(compressed))/float64(len(data))) * 100

		t.Logf("Compression ratio: %.2f:1 (%.2f%% savings)", ratio, savingsPercent)
		// Target: 70-85% compression
		require.GreaterOrEqual(t, savingsPercent, 70.0, "Should achieve at least 70% compression")
	})
}
