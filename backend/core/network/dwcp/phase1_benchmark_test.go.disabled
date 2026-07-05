package dwcp_test

import (
	"crypto/rand"
	"fmt"
	"testing"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/compression"
	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/transport"
	"go.uber.org/zap"
)

// BenchmarkAMSTThroughput measures throughput vs standard TCP
func BenchmarkAMSTThroughput(b *testing.B) {
	logger, _ := zap.NewDevelopment()

	benchmarks := []struct {
		name    string
		streams int
	}{
		{"SingleStream", 1},
		{"16Streams", 16},
		{"32Streams", 32},
		{"64Streams", 64},
		{"128Streams", 128},
	}

	dataSize := 10 * 1024 * 1024 // 10 MB per iteration
	testData := make([]byte, dataSize)
	rand.Read(testData)

	for _, bm := range benchmarks {
		b.Run(bm.name, func(b *testing.B) {
			listener, port := startBandwidthTrackingServer(b.(*testing.T), bm.streams)
			defer listener.Close()

			config := &transport.AMSTConfig{
				MinStreams:     bm.streams,
				MaxStreams:     bm.streams,
				InitialStreams: bm.streams,
				ChunkSizeKB:    512,
				AutoTune:       false,
				ConnectTimeout: 10 * time.Second,
			}

			mst, err := transport.NewMultiStreamTCP(
				fmt.Sprintf("localhost:%d", port),
				config,
				logger,
			)
			if err != nil {
				b.Fatalf("NewMultiStreamTCP failed: %v", err)
			}
			defer mst.Close()

			if err := mst.Start(); err != nil {
				b.Fatalf("Start failed: %v", err)
			}

			b.ResetTimer()
			b.SetBytes(int64(dataSize))

			for i := 0; i < b.N; i++ {
				if err := mst.Send(testData); err != nil {
					b.Fatalf("Send failed: %v", err)
				}
			}

			b.StopTimer()

			metrics := mst.GetMetrics()
			throughputMBps := metrics["throughput_mbps"].(float64)
			utilization := metrics["bandwidth_utilization"].(float64)

			b.ReportMetric(throughputMBps, "MB/s")
			b.ReportMetric(utilization*100, "utilization%")
		})
	}
}

// BenchmarkAMSTStreamScaling measures scalability with different stream counts
func BenchmarkAMSTStreamScaling(b *testing.B) {
	logger, _ := zap.NewDevelopment()

	streamCounts := []int{8, 16, 32, 64, 128, 256}
	dataSize := 5 * 1024 * 1024 // 5 MB
	testData := make([]byte, dataSize)
	rand.Read(testData)

	for _, streams := range streamCounts {
		b.Run(fmt.Sprintf("%dStreams", streams), func(b *testing.B) {
			listener, port := startBandwidthTrackingServer(b.(*testing.T), streams)
			defer listener.Close()

			config := &transport.AMSTConfig{
				MinStreams:     streams,
				MaxStreams:     streams,
				InitialStreams: streams,
				ChunkSizeKB:    256,
				AutoTune:       false,
				ConnectTimeout: 10 * time.Second,
			}

			mst, err := transport.NewMultiStreamTCP(
				fmt.Sprintf("localhost:%d", port),
				config,
				logger,
			)
			if err != nil {
				b.Skip("Failed to create multi-stream transport")
			}
			defer mst.Close()

			if err := mst.Start(); err != nil {
				b.Skip("Failed to start transport")
			}

			b.ResetTimer()
			b.SetBytes(int64(dataSize))

			for i := 0; i < b.N; i++ {
				mst.Send(testData)
			}
		})
	}
}

// BenchmarkHDECompression measures compression speed and ratio
func BenchmarkHDECompression(b *testing.B) {
	logger, _ := zap.NewDevelopment()

	benchmarks := []struct {
		name  string
		level int
	}{
		{"Fast", 0},
		{"Balanced", 3},
		{"Max", 9},
	}

	// Different data types
	dataTypes := []struct {
		name string
		data []byte
	}{
		{
			name: "VMMemory",
			data: generateCompressibleData(b.(*testing.T), 4*1024*1024, 0.7), // 70% compressible
		},
		{
			name: "ClusterState",
			data: generateCompressibleData(b.(*testing.T), 2*1024*1024, 0.8), // 80% compressible
		},
		{
			name: "Random",
			data: generateCompressibleData(b.(*testing.T), 4*1024*1024, 0.1), // 10% compressible
		},
	}

	for _, bm := range benchmarks {
		for _, dt := range dataTypes {
			testName := fmt.Sprintf("%s_%s", bm.name, dt.name)

			b.Run(testName, func(b *testing.B) {
				config := compression.DefaultDeltaEncodingConfig()
				config.CompressionLevel = bm.level

				encoder, err := compression.NewDeltaEncoder(config, logger)
				if err != nil {
					b.Fatalf("NewDeltaEncoder failed: %v", err)
				}
				defer encoder.Close()

				stateKey := "bench-" + testName

				b.ResetTimer()
				b.SetBytes(int64(len(dt.data)))

				var totalRatio float64
				for i := 0; i < b.N; i++ {
					encoded, err := encoder.Encode(stateKey, dt.data)
					if err != nil {
						b.Fatalf("Encode failed: %v", err)
					}
					totalRatio += encoded.CompressionRatio()
				}

				b.StopTimer()

				avgRatio := totalRatio / float64(b.N)
				b.ReportMetric(avgRatio, "ratio")
			})
		}
	}
}

// BenchmarkHDEDeltaEncoding measures delta encoding performance
func BenchmarkHDEDeltaEncoding(b *testing.B) {
	logger, _ := zap.NewDevelopment()

	config := compression.DefaultDeltaEncodingConfig()
	config.DeltaAlgorithm = "bsdiff"

	encoder, err := compression.NewDeltaEncoder(config, logger)
	if err != nil {
		b.Fatalf("NewDeltaEncoder failed: %v", err)
	}
	defer encoder.Close()

	// Create baseline
	baselineSize := 8 * 1024 * 1024 // 8 MB
	baseline := generateCompressibleData(b.(*testing.T), baselineSize, 0.6)

	stateKey := "bench-delta"

	// Encode baseline
	_, err = encoder.Encode(stateKey, baseline)
	if err != nil {
		b.Fatalf("Baseline encode failed: %v", err)
	}

	// Create modified versions with different change percentages
	changePercentages := []float64{0.01, 0.05, 0.10, 0.20}

	for _, changePercent := range changePercentages {
		testName := fmt.Sprintf("Change%.0fPercent", changePercent*100)

		b.Run(testName, func(b *testing.B) {
			modified := make([]byte, baselineSize)
			copy(modified, baseline)

			// Modify specified percentage
			changeSize := int(float64(baselineSize) * changePercent)
			for i := 0; i < changeSize; i++ {
				modified[i] = ^modified[i]
			}

			b.ResetTimer()
			b.SetBytes(int64(baselineSize))

			var totalRatio float64
			for i := 0; i < b.N; i++ {
				encoded, err := encoder.Encode(stateKey, modified)
				if err != nil {
					b.Fatalf("Delta encode failed: %v", err)
				}
				totalRatio += encoded.CompressionRatio()
			}

			b.StopTimer()

			avgRatio := totalRatio / float64(b.N)
			b.ReportMetric(avgRatio, "delta_ratio")
		})
	}
}

// BenchmarkMigrationSpeed measures 8GB VM migration time
func BenchmarkMigrationSpeed(b *testing.B) {
	logger, _ := zap.NewDevelopment()

	// Use smaller sample for benchmark (scale results for 8GB)
	sampleSize := 32 * 1024 * 1024 // 32 MB sample
	vmMemory := generateVMMemoryData(b.(*testing.T), sampleSize)

	listener, port := startBandwidthTrackingServer(b.(*testing.T), 128)
	defer listener.Close()

	// AMST setup
	amstConfig := &transport.AMSTConfig{
		MinStreams:          64,
		MaxStreams:          128,
		InitialStreams:      96,
		CongestionAlgorithm: "bbr",
		ChunkSizeKB:         512,
		AutoTune:            true,
		ConnectTimeout:      10 * time.Second,
	}

	mst, err := transport.NewMultiStreamTCP(fmt.Sprintf("localhost:%d", port), amstConfig, logger)
	if err != nil {
		b.Fatalf("NewMultiStreamTCP failed: %v", err)
	}
	defer mst.Close()

	if err := mst.Start(); err != nil {
		b.Fatalf("Start failed: %v", err)
	}

	// HDE setup
	hdeConfig := compression.DefaultDeltaEncodingConfig()
	hdeConfig.CompressionLevel = 6

	encoder, err := compression.NewDeltaEncoder(hdeConfig, logger)
	if err != nil {
		b.Fatalf("NewDeltaEncoder failed: %v", err)
	}
	defer encoder.Close()

	stateKey := "bench-migration"

	b.ResetTimer()
	b.SetBytes(int64(sampleSize))

	for i := 0; i < b.N; i++ {
		// Compress
		encoded, err := encoder.Encode(stateKey, vmMemory)
		if err != nil {
			b.Fatalf("Encode failed: %v", err)
		}

		// Transfer
		if err := mst.Send(encoded.Data); err != nil {
			b.Fatalf("Send failed: %v", err)
		}
	}

	b.StopTimer()

	// Report estimated 8GB time
	nsPerOp := b.Elapsed().Nanoseconds() / int64(b.N)
	scaleFactor := (8 * 1024) / 32 // 8GB / 32MB
	estimated8GBTime := time.Duration(nsPerOp * int64(scaleFactor))

	b.ReportMetric(estimated8GBTime.Seconds(), "est_8GB_sec")
}

// BenchmarkFederationSync measures state sync performance
func BenchmarkFederationSync(b *testing.B) {
	logger, _ := zap.NewDevelopment()

	stateSize := 4 * 1024 * 1024 // 4 MB cluster state
	clusterState := generateClusterStateData(b.(*testing.T), stateSize)

	listener, port := startBandwidthTrackingServer(b.(*testing.T), 64)
	defer listener.Close()

	// Transport setup
	amstConfig := &transport.AMSTConfig{
		MinStreams:     32,
		MaxStreams:     64,
		InitialStreams: 48,
		ChunkSizeKB:    256,
		AutoTune:       true,
		ConnectTimeout: 10 * time.Second,
	}

	mst, err := transport.NewMultiStreamTCP(fmt.Sprintf("localhost:%d", port), amstConfig, logger)
	if err != nil {
		b.Fatalf("NewMultiStreamTCP failed: %v", err)
	}
	defer mst.Close()

	if err := mst.Start(); err != nil {
		b.Fatalf("Start failed: %v", err)
	}

	// Compression setup
	hdeConfig := compression.DefaultDeltaEncodingConfig()
	hdeConfig.BaselineSyncEnabled = true

	encoder, err := compression.NewDeltaEncoder(hdeConfig, logger)
	if err != nil {
		b.Fatalf("NewDeltaEncoder failed: %v", err)
	}
	defer encoder.Close()

	stateKey := "bench-federation"

	// Encode baseline once
	_, err = encoder.Encode(stateKey, clusterState)
	if err != nil {
		b.Fatalf("Baseline encode failed: %v", err)
	}

	// Prepare modified state (2% change)
	modifiedState := make([]byte, stateSize)
	copy(modifiedState, clusterState)
	for i := 0; i < stateSize/50; i++ {
		modifiedState[i*50] = ^modifiedState[i*50]
	}

	b.ResetTimer()
	b.SetBytes(int64(stateSize))

	for i := 0; i < b.N; i++ {
		// Encode delta
		encoded, err := encoder.Encode(stateKey, modifiedState)
		if err != nil {
			b.Fatalf("Encode failed: %v", err)
		}

		// Transfer
		if err := mst.Send(encoded.Data); err != nil {
			b.Fatalf("Send failed: %v", err)
		}
	}
}

// BenchmarkConcurrentStreams measures scalability testing
func BenchmarkConcurrentStreams(b *testing.B) {
	logger, _ := zap.NewDevelopment()

	concurrencyLevels := []int{1, 4, 8, 16, 32}
	dataSize := 2 * 1024 * 1024 // 2 MB per operation

	for _, concurrency := range concurrencyLevels {
		b.Run(fmt.Sprintf("Concurrent%d", concurrency), func(b *testing.B) {
			listener, port := startBandwidthTrackingServer(b.(*testing.T), 128)
			defer listener.Close()

			config := &transport.AMSTConfig{
				MinStreams:     64,
				MaxStreams:     128,
				InitialStreams: 96,
				ChunkSizeKB:    256,
				AutoTune:       true,
				ConnectTimeout: 10 * time.Second,
			}

			mst, err := transport.NewMultiStreamTCP(
				fmt.Sprintf("localhost:%d", port),
				config,
				logger,
			)
			if err != nil {
				b.Skip("Failed to create transport")
			}
			defer mst.Close()

			if err := mst.Start(); err != nil {
				b.Skip("Failed to start transport")
			}

			// Prepare test data
			testData := make([]byte, dataSize)
			rand.Read(testData)

			b.ResetTimer()
			b.SetBytes(int64(dataSize * concurrency))

			b.RunParallel(func(pb *testing.PB) {
				for pb.Next() {
					mst.Send(testData)
				}
			})
		})
	}
}

// Helper functions for benchmarks

func generateCompressibleData(t testing.TB, size int, compressibility float64) []byte {
	data := make([]byte, size)

	compressibleSize := int(float64(size) * compressibility)
	randomSize := size - compressibleSize

	// Compressible part (repetitive pattern)
	pattern := []byte("BENCHMARK_PATTERN_DATA_")
	for i := 0; i < compressibleSize; i += len(pattern) {
		end := i + len(pattern)
		if end > compressibleSize {
			end = compressibleSize
		}
		copy(data[i:end], pattern[:end-i])
	}

	// Random part
	rand.Read(data[compressibleSize:])

	return data
}
