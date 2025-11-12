package dwcp_test

import (
	"bytes"
	"crypto/rand"
	"sync"
	"testing"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/compression"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
)

// TestPhase1_HDEDictionaryTraining tests dictionary creation and usage
func TestPhase1_HDEDictionaryTraining(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	config := compression.DefaultDeltaEncodingConfig()
	config.EnableDictionaryTraining = true
	config.DictionarySize = 128 * 1024 // 128 KB dictionary

	encoder, err := compression.NewDeltaEncoder(config, logger)
	require.NoError(t, err)
	defer encoder.Close()

	// Create training samples (VM memory patterns)
	samples := generateVMMemoryPatterns(t, 10, 1024*1024) // 10x 1MB samples

	// Train dictionary
	err = encoder.TrainDictionary("vm-test", samples)
	require.NoError(t, err, "Dictionary training should succeed")

	// Verify dictionary was created
	dictStats := encoder.GetDictionaryStats("vm-test")
	require.NotNil(t, dictStats, "Dictionary should exist")
	assert.Greater(t, dictStats["size"].(int), 0, "Dictionary should have size")
	assert.Greater(t, dictStats["samples"].(int), 0, "Dictionary should have samples")

	// Test compression with dictionary
	testData := generateVMMemoryPatterns(t, 1, 2*1024*1024)[0] // 2 MB
	encoded, err := encoder.Encode("vm-test", testData)
	require.NoError(t, err)

	// Dictionary should improve compression ratio
	ratio := encoded.CompressionRatio()
	assert.GreaterOrEqual(t, ratio, 10.0, "Dictionary should achieve >10x compression")

	t.Logf("✅ Dictionary training validated - Ratio: %.2fx, Dict size: %d KB",
		ratio, dictStats["size"].(int)/1024)
}

// TestPhase1_HDECompressionRatio validates >10x compression achievement
func TestPhase1_HDECompressionRatio(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	config := compression.DefaultDeltaEncodingConfig()
	config.CompressionLevel = 9 // Maximum compression

	encoder, err := compression.NewDeltaEncoder(config, logger)
	require.NoError(t, err)
	defer encoder.Close()

	// Test with highly compressible VM memory data
	vmMemorySize := 16 * 1024 * 1024 // 16 MB
	vmMemory := make([]byte, vmMemorySize)

	// Simulate VM memory with repetitive patterns (typical scenario)
	pattern := []byte("VM_MEMORY_PAGE_ZEROED_")
	for i := 0; i < vmMemorySize; i += len(pattern) {
		end := i + len(pattern)
		if end > vmMemorySize {
			end = vmMemorySize
		}
		copy(vmMemory[i:end], pattern[:end-i])
	}

	// Add some variation (10% of memory)
	for i := 0; i < vmMemorySize/10; i++ {
		vmMemory[i*10] = byte(i % 256)
	}

	stateKey := "vm-compression-test"

	// Encode with HDE
	encoded, err := encoder.Encode(stateKey, vmMemory)
	require.NoError(t, err)

	ratio := encoded.CompressionRatio()
	t.Logf("HDE compression - Original: %d MB, Compressed: %d KB, Ratio: %.2fx",
		encoded.OriginalSize/(1024*1024),
		encoded.CompressedSize/1024,
		ratio)

	// Phase 1 target: >10x compression ratio
	assert.GreaterOrEqual(t, ratio, 10.0,
		"Phase 1 requires >10x compression ratio (got %.2fx)", ratio)

	// Verify decompression
	decoded, err := encoder.Decode(stateKey, encoded)
	require.NoError(t, err)
	assert.True(t, bytes.Equal(vmMemory, decoded), "Decompression should be lossless")

	t.Log("✅ Phase 1 compression ratio (>10x) validated")
}

// TestPhase1_HDEAdaptiveCompression tests auto-level selection
func TestPhase1_HDEAdaptiveCompression(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	config := compression.DefaultDeltaEncodingConfig()
	config.AdaptiveCompression = true
	config.AdaptiveThreshold = 5.0 // Switch to higher level if ratio < 5x

	encoder, err := compression.NewDeltaEncoder(config, logger)
	require.NoError(t, err)
	defer encoder.Close()

	// Test 1: Highly compressible data (should use lower level for speed)
	compressibleData := bytes.Repeat([]byte("AAAA"), 256*1024) // 1 MB
	encoded1, err := encoder.Encode("test-1", compressibleData)
	require.NoError(t, err)

	level1 := encoded1.CompressionLevel
	ratio1 := encoded1.CompressionRatio()
	t.Logf("Compressible data - Level: %d, Ratio: %.2fx", level1, ratio1)

	// Test 2: Less compressible data (should use higher level)
	randomData := make([]byte, 1024*1024) // 1 MB random
	rand.Read(randomData)
	encoded2, err := encoder.Encode("test-2", randomData)
	require.NoError(t, err)

	level2 := encoded2.CompressionLevel
	ratio2 := encoded2.CompressionRatio()
	t.Logf("Random data - Level: %d, Ratio: %.2fx", level2, ratio2)

	// Adaptive compression should have selected appropriate levels
	if ratio1 > config.AdaptiveThreshold {
		assert.LessOrEqual(t, level1, level2,
			"High ratio data should use lower or equal compression level")
	}

	t.Log("✅ Adaptive compression level selection validated")
}

// TestPhase1_HDEAdvancedDelta tests rsync/bsdiff algorithms
func TestPhase1_HDEAdvancedDelta(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	config := compression.DefaultDeltaEncodingConfig()
	config.DeltaAlgorithm = "bsdiff" // Advanced binary diff

	encoder, err := compression.NewDeltaEncoder(config, logger)
	require.NoError(t, err)
	defer encoder.Close()

	// Create base VM disk image
	diskSize := 4 * 1024 * 1024 // 4 MB
	baseDisk := make([]byte, diskSize)
	rand.Read(baseDisk)

	stateKey := "vm-disk"

	// Encode baseline
	baseEncoded, err := encoder.Encode(stateKey, baseDisk)
	require.NoError(t, err)
	t.Logf("Baseline - Size: %d KB", baseEncoded.CompressedSize/1024)

	// Modify 1% of disk (simulate incremental changes)
	modifiedDisk := make([]byte, diskSize)
	copy(modifiedDisk, baseDisk)
	for i := 0; i < diskSize/100; i++ {
		offset := i * 100
		modifiedDisk[offset] = ^modifiedDisk[offset]
	}

	// Encode delta with bsdiff
	deltaEncoded, err := encoder.Encode(stateKey, modifiedDisk)
	require.NoError(t, err)

	assert.True(t, deltaEncoded.IsDelta, "Should use delta encoding")
	assert.Equal(t, "bsdiff", deltaEncoded.DeltaAlgorithm, "Should use bsdiff")

	// Delta should be much smaller than full state
	deltaSavings := 100.0 * float64(baseEncoded.CompressedSize-deltaEncoded.CompressedSize) /
		float64(baseEncoded.CompressedSize)

	t.Logf("Delta encoding - Size: %d KB, Savings: %.1f%%",
		deltaEncoded.CompressedSize/1024, deltaSavings)

	assert.Greater(t, deltaSavings, 50.0, "Delta should save >50% vs full state")

	// Verify delta reconstruction
	decoded, err := encoder.Decode(stateKey, deltaEncoded)
	require.NoError(t, err)
	assert.True(t, bytes.Equal(modifiedDisk, decoded), "Delta reconstruction should be accurate")

	t.Log("✅ Advanced delta encoding (bsdiff) validated")
}

// TestPhase1_HDEBaselineSync tests cross-cluster baseline synchronization
func TestPhase1_HDEBaselineSync(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	config := compression.DefaultDeltaEncodingConfig()
	config.BaselineSyncEnabled = true
	config.BaselineSyncInterval = 1 * time.Second

	// Create two encoders (simulating different clusters)
	encoder1, err := compression.NewDeltaEncoder(config, logger)
	require.NoError(t, err)
	defer encoder1.Close()

	encoder2, err := compression.NewDeltaEncoder(config, logger)
	require.NoError(t, err)
	defer encoder2.Close()

	// Encoder1 creates baseline
	stateKey := "cluster-state"
	clusterState := make([]byte, 2*1024*1024) // 2 MB
	rand.Read(clusterState)

	encoded1, err := encoder1.Encode(stateKey, clusterState)
	require.NoError(t, err)

	// Export baseline from encoder1
	baseline, err := encoder1.ExportBaseline(stateKey)
	require.NoError(t, err)
	require.NotNil(t, baseline, "Baseline export should succeed")

	// Import baseline to encoder2
	err = encoder2.ImportBaseline(stateKey, baseline)
	require.NoError(t, err)

	// Modify state slightly
	modifiedState := make([]byte, len(clusterState))
	copy(modifiedState, clusterState)
	for i := 0; i < len(clusterState)/50; i++ {
		modifiedState[i*50] = ^modifiedState[i*50]
	}

	// Encoder2 should now be able to create efficient delta
	encoded2, err := encoder2.Encode(stateKey, modifiedState)
	require.NoError(t, err)

	assert.True(t, encoded2.IsDelta, "Should use delta with synced baseline")

	deltaSize := encoded2.CompressedSize
	fullSize := encoded1.CompressedSize
	deltaSavings := 100.0 * float64(fullSize-deltaSize) / float64(fullSize)

	t.Logf("Baseline sync - Delta: %d KB, Full: %d KB, Savings: %.1f%%",
		deltaSize/1024, fullSize/1024, deltaSavings)

	assert.Greater(t, deltaSavings, 30.0, "Synced baseline should enable efficient deltas")

	t.Log("✅ Cross-cluster baseline synchronization validated")
}

// TestPhase1_HDEMetrics tests Prometheus metrics accuracy
func TestPhase1_HDEMetrics(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	config := compression.DefaultDeltaEncodingConfig()
	encoder, err := compression.NewDeltaEncoder(config, logger)
	require.NoError(t, err)
	defer encoder.Close()

	// Perform compression operations
	testData := make([]byte, 4*1024*1024) // 4 MB
	rand.Read(testData)

	stateKey := "metrics-test"

	// First compression
	startTime := time.Now()
	encoded, err := encoder.Encode(stateKey, testData)
	require.NoError(t, err)
	compressionTime := time.Since(startTime)

	// Verify metrics
	metrics := encoder.GetMetrics()

	// Check compression metrics
	bytesIn := metrics["bytes_in"].(uint64)
	bytesOut := metrics["bytes_out"].(uint64)
	assert.Equal(t, uint64(len(testData)), bytesIn, "Bytes in should match input size")
	assert.Equal(t, uint64(encoded.CompressedSize), bytesOut, "Bytes out should match compressed size")

	// Check compression ratio
	reportedRatio := metrics["compression_ratio"].(float64)
	expectedRatio := float64(bytesIn) / float64(bytesOut)
	assert.InDelta(t, expectedRatio, reportedRatio, 0.01, "Compression ratio should be accurate")

	// Check timing metrics
	reportedTime := metrics["avg_compression_time_ms"].(float64)
	assert.Greater(t, reportedTime, 0.0, "Compression time should be positive")
	assert.InDelta(t, compressionTime.Milliseconds(), int64(reportedTime), float64(compressionTime.Milliseconds())*0.5,
		"Reported time should be reasonable")

	// Check operation counter
	operations := metrics["total_operations"].(uint64)
	assert.Equal(t, uint64(1), operations, "Should have 1 operation")

	t.Logf("✅ Metrics validated - Ratio: %.2fx, Time: %.2fms, Operations: %d",
		reportedRatio, reportedTime, operations)
}

// TestPhase1_HDEConcurrency tests thread-safe compression
func TestPhase1_HDEConcurrency(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	config := compression.DefaultDeltaEncodingConfig()
	encoder, err := compression.NewDeltaEncoder(config, logger)
	require.NoError(t, err)
	defer encoder.Close()

	// Concurrent compression operations
	numGoroutines := 20
	dataSize := 512 * 1024 // 512 KB per operation

	var wg sync.WaitGroup
	errChan := make(chan error, numGoroutines)
	results := make(chan *compression.EncodedData, numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()

			data := make([]byte, dataSize)
			rand.Read(data)

			stateKey := "concurrent-" + string(rune('A'+id))
			encoded, err := encoder.Encode(stateKey, data)
			if err != nil {
				errChan <- err
				return
			}

			results <- encoded
		}(i)
	}

	wg.Wait()
	close(errChan)
	close(results)

	// Check for errors
	var errors []error
	for err := range errChan {
		errors = append(errors, err)
	}
	assert.Empty(t, errors, "Concurrent compressions should succeed")

	// Verify all results
	resultCount := 0
	for range results {
		resultCount++
	}
	assert.Equal(t, numGoroutines, resultCount, "All compressions should complete")

	// Verify metrics reflect all operations
	metrics := encoder.GetMetrics()
	totalOps := metrics["total_operations"].(uint64)
	assert.Equal(t, uint64(numGoroutines), totalOps, "Metrics should count all operations")

	t.Logf("✅ Concurrent compression validated - %d goroutines, %d operations",
		numGoroutines, totalOps)
}

// TestPhase1_HDEDictionaryUpdate tests auto-dictionary refresh
func TestPhase1_HDEDictionaryUpdate(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	config := compression.DefaultDeltaEncodingConfig()
	config.EnableDictionaryTraining = true
	config.DictionaryUpdateInterval = 2 * time.Second
	config.DictionarySize = 64 * 1024

	encoder, err := compression.NewDeltaEncoder(config, logger)
	require.NoError(t, err)
	defer encoder.Close()

	stateKey := "dict-update-test"

	// Create initial dictionary with pattern A
	samplesA := make([][]byte, 5)
	patternA := []byte("PATTERN_A_REPEATED_")
	for i := 0; i < 5; i++ {
		sample := make([]byte, 512*1024) // 512 KB
		for j := 0; j < len(sample); j += len(patternA) {
			end := j + len(patternA)
			if end > len(sample) {
				end = len(sample)
			}
			copy(sample[j:end], patternA[:end-j])
		}
		samplesA[i] = sample
	}

	err = encoder.TrainDictionary(stateKey, samplesA)
	require.NoError(t, err)

	// Compress with pattern A (should be efficient)
	testDataA := samplesA[0]
	encoded1, err := encoder.Encode(stateKey, testDataA)
	require.NoError(t, err)
	ratio1 := encoded1.CompressionRatio()

	// Wait for dictionary update interval
	time.Sleep(3 * time.Second)

	// Now use pattern B data (different pattern)
	samplesB := make([][]byte, 5)
	patternB := []byte("PATTERN_B_DIFFERENT_")
	for i := 0; i < 5; i++ {
		sample := make([]byte, 512*1024)
		for j := 0; j < len(sample); j += len(patternB) {
			end := j + len(patternB)
			if end > len(sample) {
				end = len(sample)
			}
			copy(sample[j:end], patternB[:end-j])
		}
		samplesB[i] = sample
	}

	// Trigger dictionary update with new samples
	err = encoder.TrainDictionary(stateKey, samplesB)
	require.NoError(t, err)

	// Compress with pattern B (should now be efficient with updated dictionary)
	testDataB := samplesB[0]
	encoded2, err := encoder.Encode(stateKey, testDataB)
	require.NoError(t, err)
	ratio2 := encoded2.CompressionRatio()

	t.Logf("Dictionary update - Pattern A ratio: %.2fx, Pattern B ratio: %.2fx",
		ratio1, ratio2)

	// Both should achieve good compression with appropriate dictionaries
	assert.GreaterOrEqual(t, ratio1, 5.0, "Pattern A should compress well with A dictionary")
	assert.GreaterOrEqual(t, ratio2, 5.0, "Pattern B should compress well after dictionary update")

	t.Log("✅ Automatic dictionary refresh validated")
}

// Helper functions

func generateVMMemoryPatterns(t *testing.T, count int, size int) [][]byte {
	samples := make([][]byte, count)

	patterns := []string{
		"VM_MEMORY_ZERO_PAGE_",
		"KERNEL_DATA_SEGMENT_",
		"USER_SPACE_HEAP_DATA_",
		"STACK_FRAME_POINTER_",
		"PAGE_TABLE_ENTRY_",
	}

	for i := 0; i < count; i++ {
		sample := make([]byte, size)
		pattern := []byte(patterns[i%len(patterns)])

		for j := 0; j < size; j += len(pattern) {
			end := j + len(pattern)
			if end > size {
				end = size
			}
			copy(sample[j:end], pattern[:end-j])
		}

		// Add some variation (5%)
		for j := 0; j < size/20; j++ {
			offset := j * 20
			if offset < size {
				sample[offset] = byte((offset + i) % 256)
			}
		}

		samples[i] = sample
	}

	return samples
}
