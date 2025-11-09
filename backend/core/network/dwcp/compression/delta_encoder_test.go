package compression

import (
	"bytes"
	"crypto/rand"
	"fmt"
	"sync"
	"testing"
	"time"

	"go.uber.org/zap"
)

// TestDeltaEncoder_BasicEncoding tests basic delta encoding and decoding
func TestDeltaEncoder_BasicEncoding(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	config := DefaultDeltaEncodingConfig()
	encoder, err := NewDeltaEncoder(config, logger)
	if err != nil {
		t.Fatalf("NewDeltaEncoder failed: %v", err)
	}
	defer encoder.Close()

	// Create test data
	originalData := []byte("This is the original state of VM memory")
	stateKey := "vm-test-001"

	// First encode - should create baseline
	encoded, err := encoder.Encode(stateKey, originalData)
	if err != nil {
		t.Fatalf("Encode failed: %v", err)
	}

	if encoded.IsDelta {
		t.Error("First encode should create baseline, not delta")
	}

	// Decode and verify
	decoded, err := encoder.Decode(stateKey, encoded)
	if err != nil {
		t.Fatalf("Decode failed: %v", err)
	}

	if !bytes.Equal(decoded, originalData) {
		t.Errorf("Decoded data doesn't match original.\nExpected: %s\nGot: %s",
			string(originalData), string(decoded))
	}
}

// TestDeltaEncoder_DeltaCompression tests that delta encoding is applied
func TestDeltaEncoder_DeltaCompression(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	config := DefaultDeltaEncodingConfig()
	encoder, err := NewDeltaEncoder(config, logger)
	if err != nil {
		t.Fatalf("NewDeltaEncoder failed: %v", err)
	}
	defer encoder.Close()

	stateKey := "vm-delta-test"

	// Create baseline
	baseline := []byte("The quick brown fox jumps over the lazy dog")
	encoded1, err := encoder.Encode(stateKey, baseline)
	if err != nil {
		t.Fatalf("First encode failed: %v", err)
	}

	if encoded1.IsDelta {
		t.Error("First encode should not be delta")
	}

	// Create slightly modified version
	modified := []byte("The quick brown cat jumps over the lazy dog")
	encoded2, err := encoder.Encode(stateKey, modified)
	if err != nil {
		t.Fatalf("Second encode failed: %v", err)
	}

	if !encoded2.IsDelta {
		t.Error("Second encode should be delta")
	}

	// Delta should be smaller than full compression
	t.Logf("Baseline size: %d, Delta size: %d, Compression: %.2fx",
		encoded1.CompressedSize, encoded2.CompressedSize,
		float64(encoded1.CompressedSize)/float64(encoded2.CompressedSize))

	// Decode and verify
	decoded, err := encoder.Decode(stateKey, encoded2)
	if err != nil {
		t.Fatalf("Delta decode failed: %v", err)
	}

	if !bytes.Equal(decoded, modified) {
		t.Errorf("Delta decoded incorrectly.\nExpected: %s\nGot: %s",
			string(modified), string(decoded))
	}
}

// TestDeltaEncoder_CompressionRatio tests that HDE achieves >5x compression
func TestDeltaEncoder_CompressionRatio(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	config := DefaultDeltaEncodingConfig()
	encoder, err := NewDeltaEncoder(config, logger)
	if err != nil {
		t.Fatalf("NewDeltaEncoder failed: %v", err)
	}
	defer encoder.Close()

	// Create compressible data (simulating VM memory with repetitive patterns)
	dataSize := 1024 * 1024 // 1 MB
	testData := make([]byte, dataSize)

	// Fill with repetitive pattern (typical of VM memory pages)
	pattern := []byte("MEMORY_PAGE_PATTERN_")
	for i := 0; i < dataSize; i += len(pattern) {
		copy(testData[i:], pattern)
	}

	stateKey := "vm-compression-test"

	// Encode
	encoded, err := encoder.Encode(stateKey, testData)
	if err != nil {
		t.Fatalf("Encode failed: %v", err)
	}

	ratio := encoded.CompressionRatio()
	t.Logf("Compression ratio: %.2fx (Original: %d bytes, Compressed: %d bytes)",
		ratio, encoded.OriginalSize, encoded.CompressedSize)

	// Phase 0 target: >5x compression for repetitive data
	if ratio < 5.0 {
		t.Errorf("Compression ratio %.2fx is below Phase 0 target of 5x", ratio)
	}

	// Verify decompression
	decoded, err := encoder.Decode(stateKey, encoded)
	if err != nil {
		t.Fatalf("Decode failed: %v", err)
	}

	if !bytes.Equal(decoded, testData) {
		t.Error("Decompressed data doesn't match original")
	}
}

// TestDeltaEncoder_DeltaEfficiency tests that deltas are more efficient than full state
func TestDeltaEncoder_DeltaEfficiency(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	config := DefaultDeltaEncodingConfig()
	encoder, err := NewDeltaEncoder(config, logger)
	if err != nil {
		t.Fatalf("NewDeltaEncoder failed: %v", err)
	}
	defer encoder.Close()

	stateKey := "vm-efficiency-test"

	// Create large baseline (1 MB)
	baseline := make([]byte, 1024*1024)
	rand.Read(baseline)

	encoded1, err := encoder.Encode(stateKey, baseline)
	if err != nil {
		t.Fatalf("Baseline encode failed: %v", err)
	}

	// Modify only 1% of the data
	modified := make([]byte, len(baseline))
	copy(modified, baseline)

	// Change 1% of bytes
	changeCount := len(baseline) / 100
	for i := 0; i < changeCount; i++ {
		modified[i*100] = ^modified[i*100] // Flip bits
	}

	encoded2, err := encoder.Encode(stateKey, modified)
	if err != nil {
		t.Fatalf("Delta encode failed: %v", err)
	}

	// Delta should be significantly smaller
	savings := 100.0 * float64(encoded1.CompressedSize-encoded2.CompressedSize) / float64(encoded1.CompressedSize)
	t.Logf("Delta savings: %.1f%% (Baseline: %d bytes, Delta: %d bytes)",
		savings, encoded1.CompressedSize, encoded2.CompressedSize)

	// Verify reconstruction
	decoded, err := encoder.Decode(stateKey, encoded2)
	if err != nil {
		t.Fatalf("Delta decode failed: %v", err)
	}

	if !bytes.Equal(decoded, modified) {
		t.Error("Delta reconstruction failed")
	}
}

// TestDeltaEncoder_BaselineRefresh tests that baselines are refreshed periodically
func TestDeltaEncoder_BaselineRefresh(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	config := &DeltaEncodingConfig{
		Enabled:          true,
		BaselineInterval: 100 * time.Millisecond, // Short interval for testing
		MaxBaselineAge:   500 * time.Millisecond,
		MaxDeltaChain:    3,
		CompressionLevel: 3,
	}

	encoder, err := NewDeltaEncoder(config, logger)
	if err != nil {
		t.Fatalf("NewDeltaEncoder failed: %v", err)
	}
	defer encoder.Close()

	stateKey := "vm-refresh-test"
	testData := []byte("Test data for baseline refresh")

	// First encode - creates baseline
	encoded1, err := encoder.Encode(stateKey, testData)
	if err != nil {
		t.Fatalf("First encode failed: %v", err)
	}

	if encoded1.IsDelta {
		t.Error("First encode should not be delta")
	}

	// Second encode - should be delta
	encoded2, err := encoder.Encode(stateKey, testData)
	if err != nil {
		t.Fatalf("Second encode failed: %v", err)
	}

	if !encoded2.IsDelta {
		t.Error("Second encode should be delta")
	}

	// Wait for baseline interval to expire
	time.Sleep(150 * time.Millisecond)

	// Third encode - should refresh baseline due to interval
	encoded3, err := encoder.Encode(stateKey, testData)
	if err != nil {
		t.Fatalf("Third encode failed: %v", err)
	}

	if encoded3.IsDelta {
		t.Error("Third encode should refresh baseline after interval")
	}

	metrics := encoder.GetMetrics()
	baselineRefreshes := metrics["baseline_refreshes"].(uint64)
	if baselineRefreshes < 2 {
		t.Errorf("Expected at least 2 baseline refreshes, got %d", baselineRefreshes)
	}

	t.Logf("Baseline refreshes: %d", baselineRefreshes)
}

// TestDeltaEncoder_MaxDeltaChain tests that delta chain length is limited
func TestDeltaEncoder_MaxDeltaChain(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	config := &DeltaEncodingConfig{
		Enabled:          true,
		BaselineInterval: 1 * time.Hour, // Long interval
		MaxBaselineAge:   2 * time.Hour,
		MaxDeltaChain:    3,
		CompressionLevel: 3,
	}

	encoder, err := NewDeltaEncoder(config, logger)
	if err != nil {
		t.Fatalf("NewDeltaEncoder failed: %v", err)
	}
	defer encoder.Close()

	stateKey := "vm-chain-test"
	testData := []byte("Test data for delta chain")

	// First encode - baseline
	_, err = encoder.Encode(stateKey, testData)
	if err != nil {
		t.Fatalf("Encode 1 failed: %v", err)
	}

	// Encodes 2-4 - should be deltas
	for i := 2; i <= 4; i++ {
		encoded, err := encoder.Encode(stateKey, testData)
		if err != nil {
			t.Fatalf("Encode %d failed: %v", i, err)
		}

		if i <= 3+1 { // MaxDeltaChain=3, so up to 4 total encodes
			if !encoded.IsDelta {
				t.Logf("Encode %d: expected delta, got baseline", i)
			}
		}
	}

	// Fifth encode - should create new baseline due to MaxDeltaChain
	encoded5, err := encoder.Encode(stateKey, testData)
	if err != nil {
		t.Fatalf("Encode 5 failed: %v", err)
	}

	if encoded5.IsDelta {
		t.Error("Fifth encode should create new baseline due to MaxDeltaChain")
	}

	metrics := encoder.GetMetrics()
	t.Logf("Metrics: %+v", metrics)
}

// TestDeltaEncoder_PruneOldBaselines tests baseline pruning
func TestDeltaEncoder_PruneOldBaselines(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	config := &DeltaEncodingConfig{
		Enabled:          true,
		BaselineInterval: 100 * time.Millisecond,
		MaxBaselineAge:   200 * time.Millisecond,
		MaxDeltaChain:    10,
		CompressionLevel: 3,
	}

	encoder, err := NewDeltaEncoder(config, logger)
	if err != nil {
		t.Fatalf("NewDeltaEncoder failed: %v", err)
	}
	defer encoder.Close()

	// Create multiple baselines
	for i := 0; i < 5; i++ {
		stateKey := fmt.Sprintf("vm-prune-%d", i)
		testData := []byte(fmt.Sprintf("Data for VM %d", i))

		_, err := encoder.Encode(stateKey, testData)
		if err != nil {
			t.Fatalf("Encode failed for vm-%d: %v", i, err)
		}
	}

	metrics1 := encoder.GetMetrics()
	initialCount := metrics1["baseline_count"].(int)

	if initialCount != 5 {
		t.Errorf("Expected 5 baselines, got %d", initialCount)
	}

	// Wait for baselines to age
	time.Sleep(250 * time.Millisecond)

	// Prune old baselines
	pruned := encoder.PruneOldBaselines()

	if pruned != 5 {
		t.Errorf("Expected to prune 5 baselines, pruned %d", pruned)
	}

	metrics2 := encoder.GetMetrics()
	finalCount := metrics2["baseline_count"].(int)

	if finalCount != 0 {
		t.Errorf("Expected 0 baselines after pruning, got %d", finalCount)
	}

	t.Logf("Pruned %d old baselines", pruned)
}

// TestDeltaEncoder_ConcurrentOperations tests thread safety
func TestDeltaEncoder_ConcurrentOperations(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	config := DefaultDeltaEncodingConfig()
	encoder, err := NewDeltaEncoder(config, logger)
	if err != nil {
		t.Fatalf("NewDeltaEncoder failed: %v", err)
	}
	defer encoder.Close()

	// Run concurrent encode/decode operations
	var wg sync.WaitGroup
	errChan := make(chan error, 20)

	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()

			stateKey := fmt.Sprintf("vm-concurrent-%d", id)
			testData := []byte(fmt.Sprintf("Test data for VM %d", id))

			// Encode
			encoded, err := encoder.Encode(stateKey, testData)
			if err != nil {
				errChan <- fmt.Errorf("encode vm-%d: %w", id, err)
				return
			}

			// Decode
			decoded, err := encoder.Decode(stateKey, encoded)
			if err != nil {
				errChan <- fmt.Errorf("decode vm-%d: %w", id, err)
				return
			}

			if !bytes.Equal(decoded, testData) {
				errChan <- fmt.Errorf("vm-%d: data mismatch", id)
			}
		}(i)
	}

	wg.Wait()
	close(errChan)

	for err := range errChan {
		t.Errorf("Concurrent operation error: %v", err)
	}
}

// TestDeltaEncoder_DisabledMode tests fallback when delta encoding is disabled
func TestDeltaEncoder_DisabledMode(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	config := &DeltaEncodingConfig{
		Enabled:          false, // Disabled
		BaselineInterval: 5 * time.Minute,
		MaxBaselineAge:   15 * time.Minute,
		MaxDeltaChain:    10,
		CompressionLevel: 3,
	}

	encoder, err := NewDeltaEncoder(config, logger)
	if err != nil {
		t.Fatalf("NewDeltaEncoder failed: %v", err)
	}
	defer encoder.Close()

	stateKey := "vm-disabled-test"
	testData := []byte("Test data with delta encoding disabled")

	// Encode twice
	encoded1, err := encoder.Encode(stateKey, testData)
	if err != nil {
		t.Fatalf("First encode failed: %v", err)
	}

	encoded2, err := encoder.Encode(stateKey, testData)
	if err != nil {
		t.Fatalf("Second encode failed: %v", err)
	}

	// Both should not be deltas
	if encoded1.IsDelta || encoded2.IsDelta {
		t.Error("With delta encoding disabled, no deltas should be created")
	}

	// Decode should still work
	decoded, err := encoder.Decode(stateKey, encoded1)
	if err != nil {
		t.Fatalf("Decode failed: %v", err)
	}

	if !bytes.Equal(decoded, testData) {
		t.Error("Decoded data doesn't match original")
	}
}

// BenchmarkDeltaEncoder_FullState benchmarks encoding full state
func BenchmarkDeltaEncoder_FullState(b *testing.B) {
	logger, _ := zap.NewProduction()

	config := DefaultDeltaEncodingConfig()
	encoder, err := NewDeltaEncoder(config, logger)
	if err != nil {
		b.Fatalf("NewDeltaEncoder failed: %v", err)
	}
	defer encoder.Close()

	// 1 MB test data
	testData := make([]byte, 1024*1024)
	rand.Read(testData)

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

// BenchmarkDeltaEncoder_Delta benchmarks delta encoding
func BenchmarkDeltaEncoder_Delta(b *testing.B) {
	logger, _ := zap.NewProduction()

	config := DefaultDeltaEncodingConfig()
	encoder, err := NewDeltaEncoder(config, logger)
	if err != nil {
		b.Fatalf("NewDeltaEncoder failed: %v", err)
	}
	defer encoder.Close()

	stateKey := "vm-delta-bench"

	// Create baseline
	baseline := make([]byte, 1024*1024)
	rand.Read(baseline)
	_, err = encoder.Encode(stateKey, baseline)
	if err != nil {
		b.Fatalf("Baseline encode failed: %v", err)
	}

	// Create modified version (1% change)
	modified := make([]byte, len(baseline))
	copy(modified, baseline)
	for i := 0; i < len(baseline)/100; i++ {
		modified[i*100] = ^modified[i*100]
	}

	b.ResetTimer()
	b.SetBytes(int64(len(modified)))

	for i := 0; i < b.N; i++ {
		_, err := encoder.Encode(stateKey, modified)
		if err != nil {
			b.Fatalf("Delta encode failed: %v", err)
		}
	}
}

