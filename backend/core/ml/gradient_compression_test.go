package ml

import (
	"math"
	"testing"
)

func TestGradientCompressor_CompressGradient_None(t *testing.T) {
	config := DefaultGradientCompressionConfig()
	config.Algorithm = CompressionNone
	
	compressor := NewGradientCompressor(config)
	
	gradient := &GradientTensor{
		Shape:     []int{10},
		Data:      []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0},
		LayerName: "test_layer",
		Timestamp: 12345,
	}
	
	compressed, err := compressor.CompressGradient(gradient)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	
	if compressed.Algorithm != CompressionNone {
		t.Errorf("Expected algorithm %v, got %v", CompressionNone, compressed.Algorithm)
	}
	
	if compressed.CompressionRatio != 1.0 {
		t.Errorf("Expected compression ratio 1.0, got %f", compressed.CompressionRatio)
	}
	
	// Test decompression
	decompressed, err := compressor.DecompressGradient(compressed)
	if err != nil {
		t.Fatalf("Decompression failed: %v", err)
	}
	
	if len(decompressed.Data) != len(gradient.Data) {
		t.Errorf("Expected data length %d, got %d", len(gradient.Data), len(decompressed.Data))
	}
	
	for i, val := range gradient.Data {
		if math.Abs(float64(decompressed.Data[i]-val)) > 1e-6 {
			t.Errorf("Data mismatch at index %d: expected %f, got %f", i, val, decompressed.Data[i])
		}
	}
}

func TestGradientCompressor_CompressGradient_TopK(t *testing.T) {
	config := DefaultGradientCompressionConfig()
	config.Algorithm = CompressionTopK
	config.TopKRatio = 0.5 // Keep top 50%
	
	compressor := NewGradientCompressor(config)
	
	gradient := &GradientTensor{
		Shape:     []int{10},
		Data:      []float32{0.1, 5.0, 0.2, 8.0, 0.3, 2.0, 0.1, 9.0, 0.05, 1.0},
		LayerName: "test_layer",
		Timestamp: 12345,
	}
	
	compressed, err := compressor.CompressGradient(gradient)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	
	if compressed.Algorithm != CompressionTopK {
		t.Errorf("Expected algorithm %v, got %v", CompressionTopK, compressed.Algorithm)
	}
	
	expectedSparsity := 0.5 // Should keep 5 out of 10 elements
	if math.Abs(compressed.SparsityLevel-expectedSparsity) > 0.1 {
		t.Errorf("Expected sparsity ~%f, got %f", expectedSparsity, compressed.SparsityLevel)
	}
	
	// Test decompression
	decompressed, err := compressor.DecompressGradient(compressed)
	if err != nil {
		t.Fatalf("Decompression failed: %v", err)
	}
	
	if len(decompressed.Data) != len(gradient.Data) {
		t.Errorf("Expected data length %d, got %d", len(gradient.Data), len(decompressed.Data))
	}
	
	// Check that top-k elements are preserved
	nonZeroCount := 0
	for _, val := range decompressed.Data {
		if val != 0.0 {
			nonZeroCount++
		}
	}
	
	expectedNonZero := 5 // Top 50% of 10 elements
	if nonZeroCount != expectedNonZero {
		t.Errorf("Expected %d non-zero elements, got %d", expectedNonZero, nonZeroCount)
	}
}

func TestGradientCompressor_CompressGradient_RandomK(t *testing.T) {
	config := DefaultGradientCompressionConfig()
	config.Algorithm = CompressionRandomK
	config.TopKRatio = 0.3 // Keep random 30%
	
	compressor := NewGradientCompressor(config)
	
	gradient := &GradientTensor{
		Shape:     []int{20},
		Data:      make([]float32, 20),
		LayerName: "test_layer",
		Timestamp: 12345,
	}
	
	// Fill with test data
	for i := range gradient.Data {
		gradient.Data[i] = float32(i + 1)
	}
	
	compressed, err := compressor.CompressGradient(gradient)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	
	if compressed.Algorithm != CompressionRandomK {
		t.Errorf("Expected algorithm %v, got %v", CompressionRandomK, compressed.Algorithm)
	}
	
	expectedSparsity := 0.7 // Should remove 70% of elements
	if math.Abs(compressed.SparsityLevel-expectedSparsity) > 0.1 {
		t.Errorf("Expected sparsity ~%f, got %f", expectedSparsity, compressed.SparsityLevel)
	}
	
	// Test decompression
	decompressed, err := compressor.DecompressGradient(compressed)
	if err != nil {
		t.Fatalf("Decompression failed: %v", err)
	}
	
	nonZeroCount := 0
	for _, val := range decompressed.Data {
		if val != 0.0 {
			nonZeroCount++
		}
	}
	
	expectedNonZero := 6 // 30% of 20 elements
	if nonZeroCount != expectedNonZero {
		t.Errorf("Expected %d non-zero elements, got %d", expectedNonZero, nonZeroCount)
	}
}

func TestGradientCompressor_CompressGradient_Threshold(t *testing.T) {
	config := DefaultGradientCompressionConfig()
	config.Algorithm = CompressionThreshold
	config.SparsityThreshold = 2.0 // Only keep values with absolute value > 2.0
	
	compressor := NewGradientCompressor(config)
	
	gradient := &GradientTensor{
		Shape:     []int{10},
		Data:      []float32{0.5, 3.0, -1.5, 4.5, 1.0, -2.5, 0.8, 5.0, -0.2, 3.5},
		LayerName: "test_layer",
		Timestamp: 12345,
	}
	
	compressed, err := compressor.CompressGradient(gradient)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	
	if compressed.Algorithm != CompressionThreshold {
		t.Errorf("Expected algorithm %v, got %v", CompressionThreshold, compressed.Algorithm)
	}
	
	// Test decompression
	decompressed, err := compressor.DecompressGradient(compressed)
	if err != nil {
		t.Fatalf("Decompression failed: %v", err)
	}
	
	// Check that only values > threshold are preserved
	for i, originalVal := range gradient.Data {
		decompressedVal := decompressed.Data[i]
		
		if math.Abs(float64(originalVal)) > 2.0 {
			// Should be preserved
			if math.Abs(float64(decompressedVal-originalVal)) > 1e-6 {
				t.Errorf("Large value at index %d not preserved: expected %f, got %f", i, originalVal, decompressedVal)
			}
		} else {
			// Should be zeroed
			if decompressedVal != 0.0 {
				t.Errorf("Small value at index %d not zeroed: expected 0, got %f", i, decompressedVal)
			}
		}
	}
}

func TestGradientCompressor_CompressGradient_Quantization(t *testing.T) {
	config := DefaultGradientCompressionConfig()
	config.Algorithm = CompressionQuantization
	config.QuantizationBits = Bits8
	
	compressor := NewGradientCompressor(config)
	
	gradient := &GradientTensor{
		Shape:     []int{5},
		Data:      []float32{-2.5, -1.0, 0.0, 1.5, 3.0},
		LayerName: "test_layer",
		Timestamp: 12345,
	}
	
	compressed, err := compressor.CompressGradient(gradient)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	
	if compressed.Algorithm != CompressionQuantization {
		t.Errorf("Expected algorithm %v, got %v", CompressionQuantization, compressed.Algorithm)
	}
	
	if compressed.QuantizationLevel != Bits8 {
		t.Errorf("Expected quantization level %v, got %v", Bits8, compressed.QuantizationLevel)
	}
	
	expectedCompressionRatio := 0.25 // 8-bit vs 32-bit
	if math.Abs(compressed.CompressionRatio-expectedCompressionRatio) > 0.1 {
		t.Errorf("Expected compression ratio ~%f, got %f", expectedCompressionRatio, compressed.CompressionRatio)
	}
	
	// Test decompression
	decompressed, err := compressor.DecompressGradient(compressed)
	if err != nil {
		t.Fatalf("Decompression failed: %v", err)
	}
	
	if len(decompressed.Data) != len(gradient.Data) {
		t.Errorf("Expected data length %d, got %d", len(gradient.Data), len(decompressed.Data))
	}
	
	// Check that values are approximately preserved (quantization introduces some error)
	for i, originalVal := range gradient.Data {
		decompressedVal := decompressed.Data[i]
		// Allow for quantization error
		if math.Abs(float64(decompressedVal-originalVal)) > 0.5 {
			t.Errorf("Quantization error too large at index %d: expected ~%f, got %f", i, originalVal, decompressedVal)
		}
	}
}

func TestGradientCompressor_CompressGradient_Hybrid(t *testing.T) {
	config := DefaultGradientCompressionConfig()
	config.Algorithm = CompressionHybrid
	config.TopKRatio = 0.5
	config.QuantizationBits = Bits8
	
	compressor := NewGradientCompressor(config)
	
	gradient := &GradientTensor{
		Shape:     []int{10},
		Data:      []float32{0.1, 5.0, 0.2, 8.0, 0.3, 2.0, 0.1, 9.0, 0.05, 1.0},
		LayerName: "test_layer",
		Timestamp: 12345,
	}
	
	compressed, err := compressor.CompressGradient(gradient)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	
	if compressed.Algorithm != CompressionHybrid {
		t.Errorf("Expected algorithm %v, got %v", CompressionHybrid, compressed.Algorithm)
	}
	
	// Should have both sparsity and quantization
	if compressed.SparsityLevel <= 0 {
		t.Error("Expected non-zero sparsity level")
	}
	
	if compressed.QuantizationLevel != Bits8 {
		t.Errorf("Expected quantization level %v, got %v", Bits8, compressed.QuantizationLevel)
	}
	
	// Test decompression
	decompressed, err := compressor.DecompressGradient(compressed)
	if err != nil {
		t.Fatalf("Decompression failed: %v", err)
	}
	
	if len(decompressed.Data) != len(gradient.Data) {
		t.Errorf("Expected data length %d, got %d", len(gradient.Data), len(decompressed.Data))
	}
}

func TestGradientCompressor_ErrorMetrics(t *testing.T) {
	config := DefaultGradientCompressionConfig()
	config.Algorithm = CompressionQuantization
	config.QuantizationBits = Bits8
	config.ErrorBoundCompression = true
	
	compressor := NewGradientCompressor(config)
	
	gradient := &GradientTensor{
		Shape:     []int{100},
		Data:      make([]float32, 100),
		LayerName: "test_layer",
		Timestamp: 12345,
	}
	
	// Fill with random-like test data
	for i := range gradient.Data {
		gradient.Data[i] = float32(math.Sin(float64(i)) * 10.0)
	}
	
	compressed, err := compressor.CompressGradient(gradient)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	
	// Should have error metrics
	if compressed.ErrorMetrics == nil {
		t.Error("Expected error metrics to be calculated")
	} else {
		if compressed.ErrorMetrics.L1Error < 0 {
			t.Error("L1 error should be non-negative")
		}
		if compressed.ErrorMetrics.L2Error < 0 {
			t.Error("L2 error should be non-negative")
		}
		if compressed.ErrorMetrics.RelativeError < 0 {
			t.Error("Relative error should be non-negative")
		}
		if math.IsNaN(compressed.ErrorMetrics.SNR) {
			t.Error("SNR should not be NaN")
		}
	}
}

func TestGradientCompressor_UpdateConfig(t *testing.T) {
	config1 := DefaultGradientCompressionConfig()
	config1.Algorithm = CompressionTopK
	
	compressor := NewGradientCompressor(config1)
	
	// Get initial config
	initialConfig := compressor.GetConfig()
	if initialConfig.Algorithm != CompressionTopK {
		t.Errorf("Expected initial algorithm %v, got %v", CompressionTopK, initialConfig.Algorithm)
	}
	
	// Update config
	config2 := DefaultGradientCompressionConfig()
	config2.Algorithm = CompressionQuantization
	compressor.UpdateConfig(config2)
	
	// Get updated config
	updatedConfig := compressor.GetConfig()
	if updatedConfig.Algorithm != CompressionQuantization {
		t.Errorf("Expected updated algorithm %v, got %v", CompressionQuantization, updatedConfig.Algorithm)
	}
}

func TestGradientCompressor_GetCompressionStats(t *testing.T) {
	config := DefaultGradientCompressionConfig()
	config.Algorithm = CompressionTopK
	config.TopKRatio = 0.3
	
	compressor := NewGradientCompressor(config)
	
	gradient := &GradientTensor{
		Shape:     []int{20},
		Data:      make([]float32, 20),
		LayerName: "test_layer",
		Timestamp: 12345,
	}
	
	for i := range gradient.Data {
		gradient.Data[i] = float32(i + 1)
	}
	
	compressed, err := compressor.CompressGradient(gradient)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	
	stats := compressor.GetCompressionStats(compressed)
	
	// Check required stats
	if _, ok := stats["algorithm"]; !ok {
		t.Error("Stats should include algorithm")
	}
	if _, ok := stats["compression_ratio"]; !ok {
		t.Error("Stats should include compression_ratio")
	}
	if _, ok := stats["sparsity_level"]; !ok {
		t.Error("Stats should include sparsity_level")
	}
	if _, ok := stats["original_size"]; !ok {
		t.Error("Stats should include original_size")
	}
	if _, ok := stats["compressed_size"]; !ok {
		t.Error("Stats should include compressed_size")
	}
	
	// Verify values
	if stats["algorithm"] != CompressionTopK {
		t.Errorf("Expected algorithm %v in stats, got %v", CompressionTopK, stats["algorithm"])
	}
	
	originalSize := stats["original_size"].(int)
	expectedOriginalSize := 20 * 4 // 20 float32s
	if originalSize != expectedOriginalSize {
		t.Errorf("Expected original size %d, got %d", expectedOriginalSize, originalSize)
	}
}

func TestGradientCompressor_EstimateCompressionRatio(t *testing.T) {
	compressor := NewGradientCompressor(DefaultGradientCompressionConfig())
	
	testCases := []struct {
		algorithm GradientCompressionAlgorithm
		config    GradientCompressionConfig
		dataSize  int
		expected  float64
		tolerance float64
	}{
		{
			CompressionNone,
			GradientCompressionConfig{Algorithm: CompressionNone},
			1000,
			1.0,
			0.01,
		},
		{
			CompressionTopK,
			GradientCompressionConfig{Algorithm: CompressionTopK, TopKRatio: 0.1},
			1000,
			0.12, // 0.1 * 1.2 (overhead)
			0.05,
		},
		{
			CompressionQuantization,
			GradientCompressionConfig{Algorithm: CompressionQuantization, QuantizationBits: Bits8},
			1000,
			0.25,
			0.01,
		},
		{
			CompressionQuantization,
			GradientCompressionConfig{Algorithm: CompressionQuantization, QuantizationBits: Bits16},
			1000,
			0.5,
			0.01,
		},
	}
	
	for _, tc := range testCases {
		estimated := compressor.EstimateCompressionRatio(tc.dataSize, tc.config)
		
		if math.Abs(estimated-tc.expected) > tc.tolerance {
			t.Errorf("For algorithm %v, expected ratio ~%f, got %f", tc.algorithm, tc.expected, estimated)
		}
	}
}

func TestGradientCompressor_ConcurrentAccess(t *testing.T) {
	config := DefaultGradientCompressionConfig()
	compressor := NewGradientCompressor(config)
	
	gradient := &GradientTensor{
		Shape:     []int{100},
		Data:      make([]float32, 100),
		LayerName: "test_layer",
		Timestamp: 12345,
	}
	
	for i := range gradient.Data {
		gradient.Data[i] = float32(i)
	}
	
	// Test concurrent compression operations
	numGoroutines := 10
	done := make(chan bool, numGoroutines)
	
	for i := 0; i < numGoroutines; i++ {
		go func() {
			defer func() { done <- true }()
			
			_, err := compressor.CompressGradient(gradient)
			if err != nil {
				t.Errorf("Concurrent compression failed: %v", err)
			}
		}()
	}
	
	// Wait for all goroutines to complete
	for i := 0; i < numGoroutines; i++ {
		<-done
	}
}

func BenchmarkGradientCompressor_TopK(b *testing.B) {
	config := DefaultGradientCompressionConfig()
	config.Algorithm = CompressionTopK
	config.TopKRatio = 0.1
	
	compressor := NewGradientCompressor(config)
	
	gradient := &GradientTensor{
		Shape:     []int{10000},
		Data:      make([]float32, 10000),
		LayerName: "benchmark_layer",
		Timestamp: 12345,
	}
	
	for i := range gradient.Data {
		gradient.Data[i] = float32(i) * 0.001
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := compressor.CompressGradient(gradient)
		if err != nil {
			b.Fatalf("Compression failed: %v", err)
		}
	}
}

func BenchmarkGradientCompressor_Quantization(b *testing.B) {
	config := DefaultGradientCompressionConfig()
	config.Algorithm = CompressionQuantization
	config.QuantizationBits = Bits8
	
	compressor := NewGradientCompressor(config)
	
	gradient := &GradientTensor{
		Shape:     []int{10000},
		Data:      make([]float32, 10000),
		LayerName: "benchmark_layer",
		Timestamp: 12345,
	}
	
	for i := range gradient.Data {
		gradient.Data[i] = float32(math.Sin(float64(i))) * 5.0
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := compressor.CompressGradient(gradient)
		if err != nil {
			b.Fatalf("Compression failed: %v", err)
		}
	}
}

func BenchmarkGradientCompressor_Hybrid(b *testing.B) {
	config := DefaultGradientCompressionConfig()
	config.Algorithm = CompressionHybrid
	config.TopKRatio = 0.05
	config.QuantizationBits = Bits8
	
	compressor := NewGradientCompressor(config)
	
	gradient := &GradientTensor{
		Shape:     []int{10000},
		Data:      make([]float32, 10000),
		LayerName: "benchmark_layer",
		Timestamp: 12345,
	}
	
	for i := range gradient.Data {
		gradient.Data[i] = float32(i) * 0.0001
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := compressor.CompressGradient(gradient)
		if err != nil {
			b.Fatalf("Compression failed: %v", err)
		}
	}
}