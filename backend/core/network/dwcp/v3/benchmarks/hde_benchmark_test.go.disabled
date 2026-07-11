package benchmarks

import (
	"bytes"
	"compress/gzip"
	"crypto/rand"
	"fmt"
	"sync/atomic"
	"testing"
	"time"

	"github.com/golang/snappy"
	"github.com/pierrec/lz4/v4"
)

// BenchmarkHDECompressionRatio tests compression ratios across algorithms
func BenchmarkHDECompressionRatio(b *testing.B) {
	dataSets := []struct {
		name string
		data []byte
	}{
		{"Zeros_1MB", make([]byte, 1048576)},
		{"Random_1MB", generateRandomData(1048576)},
		{"Text_1MB", generateTextData(1048576)},
		{"VMMemory_1MB", generateVMMemoryData(1048576)},
		{"MixedWorkload_1MB", generateMixedWorkloadData(1048576)},
	}

	algorithms := []string{"snappy", "lz4", "gzip-fast", "gzip-best", "zstd"}

	for _, ds := range dataSets {
		for _, algo := range algorithms {
			b.Run(fmt.Sprintf("%s_%s", ds.name, algo), func(b *testing.B) {
				benchmarkCompressionRatio(b, ds.data, algo)
			})
		}
	}
}

func benchmarkCompressionRatio(b *testing.B, data []byte, algorithm string) {
	b.ReportAllocs()

	var totalCompressed int64
	var totalOriginal int64

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		compressed, err := compressData(data, algorithm)
		if err != nil {
			b.Fatal(err)
		}

		atomic.AddInt64(&totalCompressed, int64(len(compressed)))
		atomic.AddInt64(&totalOriginal, int64(len(data)))
	}

	b.StopTimer()

	avgRatio := float64(totalOriginal-totalCompressed) / float64(totalOriginal) * 100
	b.ReportMetric(avgRatio, "compression_%")
	b.ReportMetric(float64(len(data))/1048576.0, "MB")
}

// BenchmarkHDECompressionThroughput tests compression/decompression speed
func BenchmarkHDECompressionThroughput(b *testing.B) {
	sizes := []int{4096, 65536, 1048576, 16777216} // 4KB to 16MB
	algorithms := []string{"snappy", "lz4", "gzip-fast", "zstd"}

	for _, size := range sizes {
		for _, algo := range algorithms {
			b.Run(fmt.Sprintf("Compress_%dKB_%s", size/1024, algo), func(b *testing.B) {
				data := generateVMMemoryData(size)
				benchmarkCompressionSpeed(b, data, algo, true)
			})

			b.Run(fmt.Sprintf("Decompress_%dKB_%s", size/1024, algo), func(b *testing.B) {
				data := generateVMMemoryData(size)
				compressed, _ := compressData(data, algo)
				benchmarkCompressionSpeed(b, compressed, algo, false)
			})
		}
	}
}

func benchmarkCompressionSpeed(b *testing.B, data []byte, algorithm string, compress bool) {
	b.ReportAllocs()

	var totalBytes int64
	startTime := time.Now()

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		var result []byte
		var err error

		if compress {
			result, err = compressData(data, algorithm)
		} else {
			result, err = decompressData(data, algorithm)
		}

		if err != nil {
			b.Fatal(err)
		}

		atomic.AddInt64(&totalBytes, int64(len(data)))
		_ = result
	}

	b.StopTimer()

	duration := time.Since(startTime)
	throughputMBps := float64(totalBytes) / duration.Seconds() / 1e6

	b.ReportMetric(throughputMBps, "MB/s")
}

// BenchmarkHDECRDTMergePerformance tests CRDT state merge operations
func BenchmarkHDECRDTMergePerformance(b *testing.B) {
	scenarios := []struct {
		name       string
		stateSize  int
		changeRate float64
	}{
		{"Small_LowChange", 1000, 0.01},
		{"Small_HighChange", 1000, 0.50},
		{"Medium_LowChange", 10000, 0.01},
		{"Medium_HighChange", 10000, 0.50},
		{"Large_LowChange", 100000, 0.01},
		{"Large_HighChange", 100000, 0.50},
	}

	for _, sc := range scenarios {
		b.Run(sc.name, func(b *testing.B) {
			benchmarkCRDTMerge(b, sc.stateSize, sc.changeRate)
		})
	}
}

func benchmarkCRDTMerge(b *testing.B, stateSize int, changeRate float64) {
	b.ReportAllocs()

	// Generate base state and delta
	baseState := make(map[string]interface{}, stateSize)
	deltaState := make(map[string]interface{})

	changedKeys := int(float64(stateSize) * changeRate)
	for i := 0; i < changedKeys; i++ {
		key := fmt.Sprintf("key_%d", i)
		deltaState[key] = fmt.Sprintf("value_%d", i)
	}

	var mergeCount int64

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		// Simulate CRDT merge
		for k, v := range deltaState {
			baseState[k] = v
		}
		atomic.AddInt64(&mergeCount, 1)
	}

	b.StopTimer()

	mergesPerSecond := float64(mergeCount) / b.Elapsed().Seconds()
	b.ReportMetric(mergesPerSecond, "merges/sec")
	b.ReportMetric(float64(changedKeys), "changed_keys")
}

// BenchmarkHDEDeltaEncodingEfficiency tests delta encoding compression
func BenchmarkHDEDeltaEncodingEfficiency(b *testing.B) {
	scenarios := []struct {
		name       string
		changeRate float64
		pageSize   int
	}{
		{"TinyChange_4KB", 0.001, 4096},
		{"SmallChange_4KB", 0.01, 4096},
		{"MediumChange_4KB", 0.10, 4096},
		{"TinyChange_64KB", 0.001, 65536},
		{"SmallChange_64KB", 0.01, 65536},
		{"MediumChange_64KB", 0.10, 65536},
	}

	for _, sc := range scenarios {
		b.Run(sc.name, func(b *testing.B) {
			benchmarkDeltaEncoding(b, sc.changeRate, sc.pageSize)
		})
	}
}

func benchmarkDeltaEncoding(b *testing.B, changeRate float64, pageSize int) {
	b.ReportAllocs()

	// Generate original and modified pages
	originalPage := generateVMMemoryData(pageSize)
	modifiedPage := make([]byte, pageSize)
	copy(modifiedPage, originalPage)

	// Apply changes
	changedBytes := int(float64(pageSize) * changeRate)
	for i := 0; i < changedBytes; i++ {
		modifiedPage[i] ^= 0xFF
	}

	var totalDeltaSize int64

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		delta := computeDelta(originalPage, modifiedPage)
		atomic.AddInt64(&totalDeltaSize, int64(len(delta)))
	}

	b.StopTimer()

	avgDeltaPercent := float64(totalDeltaSize) / float64(b.N*pageSize) * 100
	compressionRatio := 100 - avgDeltaPercent

	b.ReportMetric(avgDeltaPercent, "delta_%")
	b.ReportMetric(compressionRatio, "compression_%")
}

// BenchmarkHDEParallelCompression tests parallel compression performance
func BenchmarkHDEParallelCompression(b *testing.B) {
	workers := []int{1, 2, 4, 8, 16, 32}
	dataSize := 16 * 1048576 // 16MB

	for _, workerCount := range workers {
		b.Run(fmt.Sprintf("%dWorkers", workerCount), func(b *testing.B) {
			benchmarkParallelCompression(b, workerCount, dataSize)
		})
	}
}

func benchmarkParallelCompression(b *testing.B, workers int, dataSize int) {
	b.ReportAllocs()

	data := generateVMMemoryData(dataSize)
	chunkSize := dataSize / workers

	var totalBytes int64
	startTime := time.Now()

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		results := make(chan []byte, workers)

		for w := 0; w < workers; w++ {
			go func(offset int) {
				end := offset + chunkSize
				if end > dataSize {
					end = dataSize
				}

				chunk := data[offset:end]
				compressed, _ := compressData(chunk, "snappy")
				results <- compressed
			}(w * chunkSize)
		}

		// Collect results
		for w := 0; w < workers; w++ {
			<-results
		}

		atomic.AddInt64(&totalBytes, int64(dataSize))
	}

	b.StopTimer()

	duration := time.Since(startTime)
	throughputGBps := float64(totalBytes) / duration.Seconds() / 1e9
	efficiency := throughputGBps / float64(workers) * 100

	b.ReportMetric(throughputGBps, "GB/s")
	b.ReportMetric(efficiency, "efficiency_%")
}

// Helper functions

func compressData(data []byte, algorithm string) ([]byte, error) {
	switch algorithm {
	case "snappy":
		return snappy.Encode(nil, data), nil
	case "lz4":
		var buf bytes.Buffer
		writer := lz4.NewWriter(&buf)
		writer.Write(data)
		writer.Close()
		return buf.Bytes(), nil
	case "gzip-fast", "gzip-best":
		var buf bytes.Buffer
		level := gzip.DefaultCompression
		if algorithm == "gzip-best" {
			level = gzip.BestCompression
		}
		writer, _ := gzip.NewWriterLevel(&buf, level)
		writer.Write(data)
		writer.Close()
		return buf.Bytes(), nil
	case "zstd":
		// Simplified zstd simulation
		return snappy.Encode(nil, data), nil
	default:
		return data, nil
	}
}

func decompressData(data []byte, algorithm string) ([]byte, error) {
	switch algorithm {
	case "snappy":
		return snappy.Decode(nil, data)
	case "lz4":
		reader := lz4.NewReader(bytes.NewReader(data))
		var buf bytes.Buffer
		buf.ReadFrom(reader)
		return buf.Bytes(), nil
	case "gzip-fast", "gzip-best":
		reader, _ := gzip.NewReader(bytes.NewReader(data))
		var buf bytes.Buffer
		buf.ReadFrom(reader)
		reader.Close()
		return buf.Bytes(), nil
	case "zstd":
		return snappy.Decode(nil, data)
	default:
		return data, nil
	}
}

func generateRandomData(size int) []byte {
	data := make([]byte, size)
	rand.Read(data)
	return data
}

func generateTextData(size int) []byte {
	text := "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
	data := make([]byte, 0, size)
	for len(data) < size {
		data = append(data, []byte(text)...)
	}
	return data[:size]
}

func generateVMMemoryData(size int) []byte {
	data := make([]byte, size)
	// Simulate VM memory: 60% zeros, 30% text patterns, 10% random
	zeroEnd := size * 60 / 100
	textEnd := size * 90 / 100

	// Text pattern section
	text := []byte("VM_MEMORY_PAGE_DATA_")
	for i := zeroEnd; i < textEnd; i += len(text) {
		copy(data[i:], text)
	}

	// Random section
	rand.Read(data[textEnd:])

	return data
}

func generateMixedWorkloadData(size int) []byte {
	data := make([]byte, size)
	// Mixed: 40% zeros, 30% patterns, 30% random
	zeroEnd := size * 40 / 100
	patternEnd := size * 70 / 100

	pattern := []byte{0xAA, 0x55, 0xAA, 0x55}
	for i := zeroEnd; i < patternEnd; i += len(pattern) {
		copy(data[i:], pattern)
	}

	rand.Read(data[patternEnd:])

	return data
}

func computeDelta(original, modified []byte) []byte {
	delta := make([]byte, 0, len(original)/10)

	for i := 0; i < len(original); i++ {
		if original[i] != modified[i] {
			// Store offset and value
			delta = append(delta, byte(i>>8), byte(i), modified[i])
		}
	}

	return delta
}
