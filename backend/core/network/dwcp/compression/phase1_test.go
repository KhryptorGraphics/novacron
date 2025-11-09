package compression

import (
	"bytes"
	"crypto/rand"
	"fmt"
	"testing"
	"time"

	"go.uber.org/zap"
)

// TestDeltaEncoder_Phase1_DictionaryTraining tests dictionary training functionality
func TestDeltaEncoder_Phase1_DictionaryTraining(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	config := &DeltaEncodingConfig{
		Enabled:             true,
		BaselineInterval:    1 * time.Hour,
		MaxBaselineAge:      2 * time.Hour,
		MaxDeltaChain:       10,
		CompressionLevel:    3,
		EnableDictionary:    true,
		DeltaAlgorithm:      "xor",
		EnableAdaptive:      false,
		EnableBaselineSync:  false,
	}

	encoder, err := NewDeltaEncoder(config, logger)
	if err != nil {
		t.Fatalf("NewDeltaEncoder failed: %v", err)
	}
	defer encoder.Close()

	// Add samples for dictionary training
	stateKey := "vm-memory-001"
	pattern := []byte("VM_MEMORY_PAGE_DATA_PATTERN_REPEATING_")

	for i := 0; i < 50; i++ {
		data := make([]byte, 10*1024) // 10KB samples
		for j := 0; j < len(data); j += len(pattern) {
			copy(data[j:], pattern)
		}

		// Add sample
		encoder.dictionaryTrainer.AddSample("vm-memory", data)
	}

	// Train dictionary
	err = encoder.TrainDictionaries()
	if err != nil {
		t.Fatalf("Dictionary training failed: %v", err)
	}

	// Check dictionary exists
	dict, exists := encoder.dictionaryTrainer.GetDictionary("vm-memory")
	if !exists {
		t.Error("Dictionary should exist after training")
	}
	if len(dict) == 0 {
		t.Error("Dictionary should not be empty")
	}

	t.Logf("Trained dictionary size: %d bytes", len(dict))

	// Test compression with dictionary
	testData := make([]byte, 50*1024) // 50KB
	for i := 0; i < len(testData); i += len(pattern) {
		copy(testData[i:], pattern)
	}

	encoded, err := encoder.Encode(stateKey, testData)
	if err != nil {
		t.Fatalf("Encode with dictionary failed: %v", err)
	}

	ratio := encoded.CompressionRatio()
	t.Logf("Compression ratio with dictionary: %.2fx", ratio)

	// Dictionary should improve compression significantly
	if ratio < 15.0 {
		t.Errorf("Expected >15x compression ratio with dictionary, got %.2fx", ratio)
	}

	// Verify decoding
	decoded, err := encoder.Decode(stateKey, encoded)
	if err != nil {
		t.Fatalf("Decode failed: %v", err)
	}

	if !bytes.Equal(decoded, testData) {
		t.Error("Decoded data doesn't match original")
	}
}

// TestDeltaEncoder_Phase1_RSyncAlgorithm tests rsync-style delta encoding
func TestDeltaEncoder_Phase1_RSyncAlgorithm(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	config := &DeltaEncodingConfig{
		Enabled:             true,
		BaselineInterval:    1 * time.Hour,
		MaxBaselineAge:      2 * time.Hour,
		MaxDeltaChain:       10,
		CompressionLevel:    3,
		EnableDictionary:    false,
		DeltaAlgorithm:      "rsync",
		EnableAdaptive:      false,
		EnableBaselineSync:  false,
	}

	encoder, err := NewDeltaEncoder(config, logger)
	if err != nil {
		t.Fatalf("NewDeltaEncoder failed: %v", err)
	}
	defer encoder.Close()

	stateKey := "vm-rsync-test"

	// Create baseline with repeated blocks
	baseline := make([]byte, 64*1024) // 64KB
	blockPattern := []byte("BLOCK_DATA_PATTERN_0123456789")
	for i := 0; i < len(baseline); i += len(blockPattern) {
		copy(baseline[i:], blockPattern)
	}

	// First encode - creates baseline
	encoded1, err := encoder.Encode(stateKey, baseline)
	if err != nil {
		t.Fatalf("Baseline encode failed: %v", err)
	}

	// Modify some blocks (shift 25% of blocks)
	modified := make([]byte, len(baseline))
	copy(modified, baseline)
	for i := 0; i < len(modified)/4; i += len(blockPattern) {
		copy(modified[i:], []byte("MODIFIED_BLOCK_DIFFERENT_DATA_"))
	}

	// Second encode - should use rsync delta
	encoded2, err := encoder.Encode(stateKey, modified)
	if err != nil {
		t.Fatalf("Delta encode failed: %v", err)
	}

	if !encoded2.IsDelta {
		t.Error("Second encode should be delta")
	}

	// RSyncdelta should be efficient for block changes
	savings := 100.0 * float64(encoded1.CompressedSize-encoded2.CompressedSize) / float64(encoded1.CompressedSize)
	t.Logf("RSunc delta savings: %.1f%%", savings)

	// Decode and verify
	decoded, err := encoder.Decode(stateKey, encoded2)
	if err != nil {
		t.Fatalf("Delta decode failed: %v", err)
	}

	if !bytes.Equal(decoded, modified) {
		t.Error("RSunc delta reconstruction failed")
	}
}

// TestDeltaEncoder_Phase1_BSDiffAlgorithm tests bsdiff binary diff
func TestDeltaEncoder_Phase1_BSDiffAlgorithm(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	config := &DeltaEncodingConfig{
		Enabled:             true,
		BaselineInterval:    1 * time.Hour,
		MaxBaselineAge:      2 * time.Hour,
		MaxDeltaChain:       10,
		CompressionLevel:    3,
		EnableDictionary:    false,
		DeltaAlgorithm:      "bsdiff",
		EnableAdaptive:      false,
		EnableBaselineSync:  false,
	}

	encoder, err := NewDeltaEncoder(config, logger)
	if err != nil {
		t.Fatalf("NewDeltaEncoder failed: %v", err)
	}
	defer encoder.Close()

	stateKey := "vm-bsdiff-test"

	// Create baseline with some structure
	baseline := make([]byte, 32*1024) // 32KB
	rand.Read(baseline)

	// First encode
	encoded1, err := encoder.Encode(stateKey, baseline)
	if err != nil {
		t.Fatalf("Baseline encode failed: %v", err)
	}

	// Make small modifications (5%)
	modified := make([]byte, len(baseline))
	copy(modified, baseline)
	for i := 0; i < len(modified)/20; i++ {
		modified[i*20] = ^modified[i*20]
	}

	// Second encode with bsdiff
	encoded2, err := encoder.Encode(stateKey, modified)
	if err != nil {
		t.Fatalf("BSDiff encode failed: %v", err)
	}

	if !encoded2.IsDelta {
		t.Error("Second encode should be delta")
	}

	t.Logf("BSDiff: Baseline=%d bytes, Delta=%d bytes, Ratio=%.2fx",
		encoded1.CompressedSize, encoded2.CompressedSize,
		float64(encoded1.CompressedSize)/float64(encoded2.CompressedSize))

	// Decode and verify
	decoded, err := encoder.Decode(stateKey, encoded2)
	if err != nil {
		t.Fatalf("BSDiff decode failed: %v", err)
	}

	if !bytes.Equal(decoded, modified) {
		t.Error("BSDiff reconstruction failed")
	}
}

// TestDeltaEncoder_Phase1_AdaptiveAlgorithm tests auto-selection
func TestDeltaEncoder_Phase1_AdaptiveAlgorithm(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	config := &DeltaEncodingConfig{
		Enabled:             true,
		BaselineInterval:    1 * time.Hour,
		MaxBaselineAge:      2 * time.Hour,
		MaxDeltaChain:       10,
		CompressionLevel:    3,
		EnableDictionary:    false,
		DeltaAlgorithm:      "auto",
		EnableAdaptive:      false,
		EnableBaselineSync:  false,
	}

	encoder, err := NewDeltaEncoder(config, logger)
	if err != nil {
		t.Fatalf("NewDeltaEncoder failed: %v", err)
	}
	defer encoder.Close()

	// Test with small data (should use XOR)
	smallData := make([]byte, 5*1024) // 5KB
	rand.Read(smallData)

	encoded1, err := encoder.Encode("vm-small", smallData)
	if err != nil {
		t.Fatalf("Small data encode failed: %v", err)
	}

	// Test with medium data (should use rsync)
	mediumData := make([]byte, 100*1024) // 100KB
	rand.Read(mediumData)

	encoded2, err := encoder.Encode("vm-medium", mediumData)
	if err != nil {
		t.Fatalf("Medium data encode failed: %v", err)
	}

	// Test with large data (should use bsdiff)
	largeData := make([]byte, 2*1024*1024) // 2MB
	rand.Read(largeData)

	encoded3, err := encoder.Encode("vm-large", largeData)
	if err != nil {
		t.Fatalf("Large data encode failed: %v", err)
	}

	t.Logf("Auto algorithm selection: Small=%d, Medium=%d, Large=%d",
		encoded1.CompressedSize, encoded2.CompressedSize, encoded3.CompressedSize)
}

// TestDeltaEncoder_Phase1_AdaptiveCompression tests adaptive level adjustment
func TestDeltaEncoder_Phase1_AdaptiveCompression(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	config := &DeltaEncodingConfig{
		Enabled:             true,
		BaselineInterval:    1 * time.Hour,
		MaxBaselineAge:      2 * time.Hour,
		MaxDeltaChain:       10,
		CompressionLevel:    3,
		EnableDictionary:    false,
		DeltaAlgorithm:      "xor",
		EnableAdaptive:      true,
		AdaptiveThreshold:   10.0,
		MinCompressionRatio: 1.1,
		EnableBaselineSync:  false,
	}

	encoder, err := NewDeltaEncoder(config, logger)
	if err != nil {
		t.Fatalf("NewDeltaEncoder failed: %v", err)
	}
	defer encoder.Close()

	// Compress highly compressible data
	compressibleData := make([]byte, 100*1024)
	pattern := []byte("REPEATING_PATTERN_")
	for i := 0; i < len(compressibleData); i += len(pattern) {
		copy(compressibleData[i:], pattern)
	}

	for i := 0; i < 20; i++ {
		stateKey := fmt.Sprintf("vm-adaptive-%d", i)
		_, err := encoder.Encode(stateKey, compressibleData)
		if err != nil {
			t.Fatalf("Encode %d failed: %v", i, err)
		}
	}

	// Check adaptive stats
	stats := encoder.adaptiveComp.GetStats()
	t.Logf("Adaptive compression stats: %+v", stats)

	if stats["enabled"].(bool) != true {
		t.Error("Adaptive compression should be enabled")
	}
}

// TestDeltaEncoder_Phase1_CompressionRatio tests >15x target
func TestDeltaEncoder_Phase1_CompressionRatio(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	config := &DeltaEncodingConfig{
		Enabled:             true,
		BaselineInterval:    1 * time.Hour,
		MaxBaselineAge:      2 * time.Hour,
		MaxDeltaChain:       10,
		CompressionLevel:    9, // Max compression
		EnableDictionary:    true,
		DeltaAlgorithm:      "auto",
		EnableAdaptive:      true,
		AdaptiveThreshold:   15.0,
		MinCompressionRatio: 1.1,
		EnableBaselineSync:  false,
	}

	encoder, err := NewDeltaEncoder(config, logger)
	if err != nil {
		t.Fatalf("NewDeltaEncoder failed: %v", err)
	}
	defer encoder.Close()

	// Add dictionary samples
	stateKey := "vm-memory-ratio-test"
	pattern := []byte("VM_MEMORY_HIGHLY_REPETITIVE_PATTERN_DATA_")

	for i := 0; i < 100; i++ {
		sample := make([]byte, 20*1024)
		for j := 0; j < len(sample); j += len(pattern) {
			copy(sample[j:], pattern)
		}
		encoder.dictionaryTrainer.AddSample("vm-memory", sample)
	}

	// Train dictionary
	encoder.TrainDictionaries()

	// Test with highly compressible data
	testData := make([]byte, 1024*1024) // 1MB
	for i := 0; i < len(testData); i += len(pattern) {
		copy(testData[i:], pattern)
	}

	// First encode - baseline
	encoded, err := encoder.Encode(stateKey, testData)
	if err != nil {
		t.Fatalf("Encode failed: %v", err)
	}

	ratio := encoded.CompressionRatio()
	t.Logf("Phase 1 compression ratio: %.2fx (Original: %d bytes, Compressed: %d bytes)",
		ratio, encoded.OriginalSize, encoded.CompressedSize)

	// Phase 1 target: >15x compression ratio
	if ratio < 15.0 {
		t.Errorf("Compression ratio %.2fx is below Phase 1 target of 15x", ratio)
	}

	// Verify decoding
	decoded, err := encoder.Decode(stateKey, encoded)
	if err != nil {
		t.Fatalf("Decode failed: %v", err)
	}

	if !bytes.Equal(decoded, testData) {
		t.Error("Decoded data doesn't match original")
	}
}

// TestDeltaEncoder_Phase1_Metrics tests comprehensive metrics collection
func TestDeltaEncoder_Phase1_Metrics(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	config := DefaultDeltaEncodingConfig()
	encoder, err := NewDeltaEncoder(config, logger)
	if err != nil {
		t.Fatalf("NewDeltaEncoder failed: %v", err)
	}
	defer encoder.Close()

	// Perform some operations
	testData := make([]byte, 50*1024)
	rand.Read(testData)

	for i := 0; i < 10; i++ {
		stateKey := fmt.Sprintf("vm-metrics-%d", i)
		encoded, err := encoder.Encode(stateKey, testData)
		if err != nil {
			t.Fatalf("Encode failed: %v", err)
		}

		_, err = encoder.Decode(stateKey, encoded)
		if err != nil {
			t.Fatalf("Decode failed: %v", err)
		}
	}

	// Get detailed metrics
	metrics := encoder.GetDetailedMetrics()

	// Check required metrics
	requiredMetrics := []string{
		"compression_ratio",
		"delta_hit_rate",
		"total_operations",
		"bytes_original",
		"bytes_compressed",
	}

	for _, metric := range requiredMetrics {
		if _, exists := metrics[metric]; !exists {
			t.Errorf("Missing required metric: %s", metric)
		}
	}

	t.Logf("Phase 1 metrics: %+v", metrics)
}

// BenchmarkDeltaEncoder_Phase1_Dictionary benchmarks dictionary compression
func BenchmarkDeltaEncoder_Phase1_Dictionary(b *testing.B) {
	logger, _ := zap.NewProduction()

	config := &DeltaEncodingConfig{
		Enabled:             true,
		BaselineInterval:    1 * time.Hour,
		MaxBaselineAge:      2 * time.Hour,
		MaxDeltaChain:       10,
		CompressionLevel:    3,
		EnableDictionary:    true,
		DeltaAlgorithm:      "xor",
		EnableAdaptive:      false,
		EnableBaselineSync:  false,
	}

	encoder, _ := NewDeltaEncoder(config, logger)
	defer encoder.Close()

	// Train dictionary
	pattern := []byte("BENCHMARK_PATTERN_DATA_")
	for i := 0; i < 50; i++ {
		sample := make([]byte, 10*1024)
		for j := 0; j < len(sample); j += len(pattern) {
			copy(sample[j:], pattern)
		}
		encoder.dictionaryTrainer.AddSample("vm-memory", sample)
	}
	encoder.TrainDictionaries()

	// Benchmark data
	testData := make([]byte, 100*1024)
	for i := 0; i < len(testData); i += len(pattern) {
		copy(testData[i:], pattern)
	}

	b.ResetTimer()
	b.SetBytes(int64(len(testData)))

	for i := 0; i < b.N; i++ {
		stateKey := fmt.Sprintf("vm-bench-%d", i)
		_, err := encoder.Encode(stateKey, testData)
		if err != nil {
			b.Fatalf("Encode failed: %v", err)
		}
	}
}

// BenchmarkDeltaEncoder_Phase1_Adaptive benchmarks adaptive compression
func BenchmarkDeltaEncoder_Phase1_Adaptive(b *testing.B) {
	logger, _ := zap.NewProduction()

	config := &DeltaEncodingConfig{
		Enabled:             true,
		BaselineInterval:    1 * time.Hour,
		MaxBaselineAge:      2 * time.Hour,
		MaxDeltaChain:       10,
		CompressionLevel:    3,
		EnableDictionary:    false,
		DeltaAlgorithm:      "auto",
		EnableAdaptive:      true,
		AdaptiveThreshold:   10.0,
		MinCompressionRatio: 1.1,
		EnableBaselineSync:  false,
	}

	encoder, _ := NewDeltaEncoder(config, logger)
	defer encoder.Close()

	testData := make([]byte, 100*1024)
	rand.Read(testData)

	b.ResetTimer()
	b.SetBytes(int64(len(testData)))

	for i := 0; i < b.N; i++ {
		stateKey := fmt.Sprintf("vm-adaptive-%d", i)
		_, err := encoder.Encode(stateKey, testData)
		if err != nil {
			b.Fatalf("Encode failed: %v", err)
		}
	}
}
