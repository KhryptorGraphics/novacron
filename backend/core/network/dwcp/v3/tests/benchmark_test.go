package tests

import (
	"context"
	"crypto/rand"
	"fmt"
	"testing"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp"
	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/upgrade"
)

// BenchmarkAMSTv1VsV3 compares AMST v1 vs v3 performance
func BenchmarkAMSTv1VsV3(b *testing.B) {
	b.Run("v1_baseline", func(b *testing.B) {
		upgrade.DisableAll()
		defer upgrade.DisableAll()

		config := dwcp.AMSTConfig{
			InitialStreams: 8,
			ChunkSize:      64 * 1024,
		}

		amst, err := dwcp.NewAMST(config)
		if err != nil {
			b.Fatal(err)
		}
		defer amst.Close()

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			amst.UpdateMetrics(5, 0.001, 10e9)
			_ = amst.GetMetrics()
		}
	})

	b.Run("v3_enhanced", func(b *testing.B) {
		upgrade.EnableAll(100)
		defer upgrade.DisableAll()

		config := dwcp.AMSTConfig{
			InitialStreams: 8,
			ChunkSize:      64 * 1024,
			EnableAdaptive: true,
		}

		amst, err := dwcp.NewAMST(config)
		if err != nil {
			b.Fatal(err)
		}
		defer amst.Close()

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			amst.UpdateMetrics(5, 0.001, 10e9)
			_ = amst.GetMetrics()
		}
	})
}

// BenchmarkHDECompression benchmarks HDE compression performance
func BenchmarkHDECompression(b *testing.B) {
	upgrade.EnableAll(100)
	defer upgrade.DisableAll()

	sizes := []int{
		1024,          // 1 KB
		10 * 1024,     // 10 KB
		100 * 1024,    // 100 KB
		1024 * 1024,   // 1 MB
		10 * 1024 * 1024, // 10 MB
	}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("size_%d_bytes", size), func(b *testing.B) {
			config := dwcp.HDEConfig{
				GlobalLevel: 3,
				EnableDelta: true,
			}

			hde, err := dwcp.NewHDE(config)
			if err != nil {
				b.Fatal(err)
			}
			defer hde.Close()

			data := make([]byte, size)
			rand.Read(data)

			b.ResetTimer()
			b.SetBytes(int64(size))

			for i := 0; i < b.N; i++ {
				_, err := hde.CompressMemory("vm-bench", data, dwcp.CompressionGlobal)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkHDECompressionLevels benchmarks different compression levels
func BenchmarkHDECompressionLevels(b *testing.B) {
	upgrade.EnableAll(100)
	defer upgrade.DisableAll()

	levels := []struct {
		name  string
		tier  dwcp.CompressionLevel
	}{
		{"local_level_0", dwcp.CompressionLocal},
		{"regional_level_3", dwcp.CompressionRegional},
		{"global_level_9", dwcp.CompressionGlobal},
	}

	data := make([]byte, 1024*1024) // 1MB
	rand.Read(data)

	for _, level := range levels {
		b.Run(level.name, func(b *testing.B) {
			config := dwcp.HDEConfig{
				LocalLevel:    0,
				RegionalLevel: 3,
				GlobalLevel:   9,
			}

			hde, err := dwcp.NewHDE(config)
			if err != nil {
				b.Fatal(err)
			}
			defer hde.Close()

			b.ResetTimer()
			b.SetBytes(int64(len(data)))

			for i := 0; i < b.N; i++ {
				_, err := hde.CompressMemory("vm-bench", data, level.tier)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkHDEDeltaEncoding benchmarks delta encoding performance
func BenchmarkHDEDeltaEncoding(b *testing.B) {
	upgrade.EnableAll(100)
	defer upgrade.DisableAll()

	config := dwcp.HDEConfig{
		EnableDelta: true,
		BlockSize:   4 * 1024,
	}

	hde, err := dwcp.NewHDE(config)
	if err != nil {
		b.Fatal(err)
	}
	defer hde.Close()

	// Create baseline
	baseline := make([]byte, 100*1024)
	rand.Read(baseline)

	// First compression creates baseline
	_, err = hde.CompressMemory("vm-delta-bench", baseline, dwcp.CompressionGlobal)
	if err != nil {
		b.Fatal(err)
	}

	// Create modified version (1% change)
	modified := make([]byte, 100*1024)
	copy(modified, baseline)
	for i := 0; i < 1024; i++ {
		modified[i] = byte(i % 256)
	}

	b.ResetTimer()
	b.SetBytes(int64(len(modified)))

	for i := 0; i < b.N; i++ {
		_, err := hde.CompressMemory("vm-delta-bench", modified, dwcp.CompressionGlobal)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkModeDetection benchmarks mode detection performance
func BenchmarkModeDetection(b *testing.B) {
	upgrade.EnableAll(100)
	defer upgrade.DisableAll()

	detector := upgrade.NewModeDetector()
	ctx := context.Background()

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = detector.DetectMode(ctx)
	}
}

// BenchmarkModeSwitching benchmarks mode switching performance
func BenchmarkModeSwitching(b *testing.B) {
	upgrade.EnableAll(100)
	defer upgrade.DisableAll()

	detector := upgrade.NewModeDetector()

	modes := []upgrade.NetworkMode{
		upgrade.ModeDatacenter,
		upgrade.ModeInternet,
		upgrade.ModeHybrid,
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		mode := modes[i%len(modes)]
		detector.ForceMode(mode)
	}
}

// BenchmarkFeatureFlagEvaluation benchmarks feature flag evaluation
func BenchmarkFeatureFlagEvaluation(b *testing.B) {
	upgrade.EnableAll(50)
	defer upgrade.DisableAll()

	nodeIDs := make([]string, 1000)
	for i := range nodeIDs {
		nodeIDs[i] = fmt.Sprintf("node-%d", i)
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		nodeID := nodeIDs[i%len(nodeIDs)]
		_ = upgrade.ShouldUseV3(nodeID)
	}
}

// BenchmarkConcurrentCompression benchmarks concurrent compression
func BenchmarkConcurrentCompression(b *testing.B) {
	upgrade.EnableAll(100)
	defer upgrade.DisableAll()

	config := dwcp.HDEConfig{
		GlobalLevel: 3,
	}

	hde, err := dwcp.NewHDE(config)
	if err != nil {
		b.Fatal(err)
	}
	defer hde.Close()

	data := make([]byte, 100*1024)
	rand.Read(data)

	b.ResetTimer()
	b.SetBytes(int64(len(data)))

	b.RunParallel(func(pb *testing.PB) {
		vmID := fmt.Sprintf("vm-%d", time.Now().UnixNano())
		for pb.Next() {
			_, err := hde.CompressMemory(vmID, data, dwcp.CompressionGlobal)
			if err != nil {
				b.Fatal(err)
			}
		}
	})
}

// BenchmarkAMSTMetricsCollection benchmarks metrics collection
func BenchmarkAMSTMetricsCollection(b *testing.B) {
	upgrade.EnableAll(100)
	defer upgrade.DisableAll()

	config := dwcp.AMSTConfig{
		InitialStreams: 8,
	}

	amst, err := dwcp.NewAMST(config)
	if err != nil {
		b.Fatal(err)
	}
	defer amst.Close()

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = amst.GetMetrics()
	}
}

// BenchmarkHDEMetricsCollection benchmarks HDE metrics collection
func BenchmarkHDEMetricsCollection(b *testing.B) {
	upgrade.EnableAll(100)
	defer upgrade.DisableAll()

	config := dwcp.HDEConfig{}
	hde, err := dwcp.NewHDE(config)
	if err != nil {
		b.Fatal(err)
	}
	defer hde.Close()

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = hde.GetMetrics()
	}
}

// BenchmarkCompressionRatio benchmarks compression ratio calculation
func BenchmarkCompressionRatio(b *testing.B) {
	upgrade.EnableAll(100)
	defer upgrade.DisableAll()

	// Test different data patterns
	patterns := []struct {
		name string
		data []byte
	}{
		{
			name: "zeros",
			data: make([]byte, 1024*1024),
		},
		{
			name: "random",
			data: func() []byte {
				d := make([]byte, 1024*1024)
				rand.Read(d)
				return d
			}(),
		},
		{
			name: "pattern",
			data: func() []byte {
				d := make([]byte, 1024*1024)
				for i := range d {
					d[i] = byte(i % 256)
				}
				return d
			}(),
		},
	}

	for _, pattern := range patterns {
		b.Run(pattern.name, func(b *testing.B) {
			config := dwcp.HDEConfig{
				GlobalLevel: 3,
			}

			hde, err := dwcp.NewHDE(config)
			if err != nil {
				b.Fatal(err)
			}
			defer hde.Close()

			b.ResetTimer()
			b.SetBytes(int64(len(pattern.data)))

			for i := 0; i < b.N; i++ {
				compressed, err := hde.CompressMemory("vm-ratio", pattern.data, dwcp.CompressionGlobal)
				if err != nil {
					b.Fatal(err)
				}

				// Calculate ratio
				_ = float64(len(pattern.data)) / float64(len(compressed))
			}
		})
	}
}
