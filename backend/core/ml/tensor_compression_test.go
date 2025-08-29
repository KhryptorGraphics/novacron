package ml

import (
	"math"
	"testing"
)

func TestTensorCompressor_CompressTensor_None(t *testing.T) {
	config := DefaultGradientCompressionConfig()
	compressor := NewTensorCompressor(config)
	
	tensor := &ModelTensor{
		Name:      "test_weight",
		Shape:     []int{4, 4},
		Data:      make([]float32, 16),
		Type:      TensorTypeWeight,
		LayerType: "linear",
	}
	
	// Fill with test data
	for i := range tensor.Data {
		tensor.Data[i] = float32(i + 1)
	}
	
	compressed, err := compressor.CompressTensor(tensor, TensorCompressionNone)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	
	if compressed.Algorithm != TensorCompressionNone {
		t.Errorf("Expected algorithm %v, got %v", TensorCompressionNone, compressed.Algorithm)
	}
	
	if compressed.CompressionRatio != 1.0 {
		t.Errorf("Expected compression ratio 1.0, got %f", compressed.CompressionRatio)
	}
	
	// Test decompression
	decompressed, err := compressor.DecompressTensor(compressed)
	if err != nil {
		t.Fatalf("Decompression failed: %v", err)
	}
	
	// Verify data integrity
	for i, originalVal := range tensor.Data {
		if math.Abs(float64(decompressed.Data[i]-originalVal)) > 1e-6 {
			t.Errorf("Data mismatch at index %d: expected %f, got %f", i, originalVal, decompressed.Data[i])
		}
	}
}

func TestTensorCompressor_CompressTensor_Pruning(t *testing.T) {
	config := DefaultGradientCompressionConfig()
	config.CompressionRatio = 0.3 // Keep only 30%
	compressor := NewTensorCompressor(config)
	
	tensor := &ModelTensor{
		Name:      "test_weight",
		Shape:     []int{5, 4},
		Data:      []float32{0.1, 5.0, 0.01, 3.0, 0.001, 8.0, 0.05, 2.0, 0.2, 7.0, 0.02, 4.0, 0.3, 6.0, 0.03, 1.0, 0.4, 9.0, 0.04, 0.5},
		Type:      TensorTypeWeight,
		LayerType: "linear",
	}
	
	compressed, err := compressor.CompressTensor(tensor, TensorCompressionPruning)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	
	if compressed.Algorithm != TensorCompressionPruning {
		t.Errorf("Expected algorithm %v, got %v", TensorCompressionPruning, compressed.Algorithm)
	}
	
	// Should achieve significant compression
	if compressed.CompressionRatio > 0.7 {
		t.Errorf("Expected significant compression, got ratio %f", compressed.CompressionRatio)
	}
	
	// Test decompression
	decompressed, err := compressor.DecompressTensor(compressed)
	if err != nil {
		t.Fatalf("Decompression failed: %v", err)
	}
	
	if len(decompressed.Data) != len(tensor.Data) {
		t.Errorf("Expected data length %d, got %d", len(tensor.Data), len(decompressed.Data))
	}
	
	// Count non-zero elements after decompression
	nonZeroCount := 0
	for _, val := range decompressed.Data {
		if val != 0.0 {
			nonZeroCount++
		}
	}
	
	expectedNonZero := int(float64(len(tensor.Data)) * config.CompressionRatio)
	if math.Abs(float64(nonZeroCount-expectedNonZero)) > float64(len(tensor.Data))*0.1 {
		t.Errorf("Expected ~%d non-zero elements, got %d", expectedNonZero, nonZeroCount)
	}
}

func TestTensorCompressor_CompressTensor_Quantization(t *testing.T) {
	config := DefaultGradientCompressionConfig()
	config.QuantizationBits = Bits8
	compressor := NewTensorCompressor(config)
	
	tensor := &ModelTensor{
		Name:      "test_weight",
		Shape:     []int{3, 3},
		Data:      []float32{-2.5, -1.0, 0.0, 1.5, 3.0, -0.5, 2.0, -1.5, 0.8},
		Type:      TensorTypeWeight,
		LayerType: "conv",
	}
	
	compressed, err := compressor.CompressTensor(tensor, TensorCompressionQuantization)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	
	if compressed.Algorithm != TensorCompressionQuantization {
		t.Errorf("Expected algorithm %v, got %v", TensorCompressionQuantization, compressed.Algorithm)
	}
	
	expectedCompressionRatio := 0.25 // 8-bit vs 32-bit
	if math.Abs(compressed.CompressionRatio-expectedCompressionRatio) > 0.05 {
		t.Errorf("Expected compression ratio ~%f, got %f", expectedCompressionRatio, compressed.CompressionRatio)
	}
	
	// Test decompression
	decompressed, err := compressor.DecompressTensor(compressed)
	if err != nil {
		t.Fatalf("Decompression failed: %v", err)
	}
	
	// Check quantization error is reasonable
	for i, originalVal := range tensor.Data {
		decompressedVal := decompressed.Data[i]
		// Allow for quantization error
		if math.Abs(float64(decompressedVal-originalVal)) > 0.2 {
			t.Errorf("Quantization error too large at index %d: expected ~%f, got %f", i, originalVal, decompressedVal)
		}
	}
}

func TestTensorCompressor_CompressTensor_SVD(t *testing.T) {
	config := DefaultGradientCompressionConfig()
	config.CompressionRatio = 0.5
	compressor := NewTensorCompressor(config)
	
	tensor := &ModelTensor{
		Name:      "test_weight",
		Shape:     []int{4, 4}, // Required for SVD
		Data:      make([]float32, 16),
		Type:      TensorTypeWeight,
		LayerType: "linear",
	}
	
	// Fill with structured test data (low-rank-ish)
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			tensor.Data[i*4+j] = float32((i + 1) * (j + 1))
		}
	}
	
	compressed, err := compressor.CompressTensor(tensor, TensorCompressionSVD)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	
	if compressed.Algorithm != TensorCompressionSVD {
		t.Errorf("Expected algorithm %v, got %v", TensorCompressionSVD, compressed.Algorithm)
	}
	
	if compressed.SVDComponents == nil {
		t.Error("Expected SVD components to be present")
	}
	
	// Test decompression
	decompressed, err := compressor.DecompressTensor(compressed)
	if err != nil {
		t.Fatalf("Decompression failed: %v", err)
	}
	
	if len(decompressed.Data) != len(tensor.Data) {
		t.Errorf("Expected data length %d, got %d", len(tensor.Data), len(decompressed.Data))
	}
}

func TestTensorCompressor_CompressTensor_KMeans(t *testing.T) {
	config := DefaultGradientCompressionConfig()
	config.QuantizationBits = Bits8 // 256 clusters
	compressor := NewTensorCompressor(config)
	
	tensor := &ModelTensor{
		Name:      "test_weight",
		Shape:     []int{10, 10},
		Data:      make([]float32, 100),
		Type:      TensorTypeWeight,
		LayerType: "linear",
	}
	
	// Fill with clusterable data
	for i := range tensor.Data {
		if i%3 == 0 {
			tensor.Data[i] = 1.0
		} else if i%3 == 1 {
			tensor.Data[i] = 0.0
		} else {
			tensor.Data[i] = -1.0
		}
	}
	
	compressed, err := compressor.CompressTensor(tensor, TensorCompressionKMeans)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	
	if compressed.Algorithm != TensorCompressionKMeans {
		t.Errorf("Expected algorithm %v, got %v", TensorCompressionKMeans, compressed.Algorithm)
	}
	
	if compressed.QuantizationClusters == nil {
		t.Error("Expected quantization clusters to be present")
	}
	
	// Test decompression
	decompressed, err := compressor.DecompressTensor(compressed)
	if err != nil {
		t.Fatalf("Decompression failed: %v", err)
	}
	
	// Check that values are clustered correctly
	uniqueValues := make(map[float32]bool)
	for _, val := range decompressed.Data {
		uniqueValues[val] = true
	}
	
	// Should have limited number of unique values due to clustering
	if len(uniqueValues) > 256 {
		t.Errorf("Expected at most 256 unique values, got %d", len(uniqueValues))
	}
}

func TestTensorCompressor_CompressTensor_Huffman(t *testing.T) {
	config := DefaultGradientCompressionConfig()
	compressor := NewTensorCompressor(config)
	
	tensor := &ModelTensor{
		Name:      "test_weight",
		Shape:     []int{8, 8},
		Data:      make([]float32, 64),
		Type:      TensorTypeWeight,
		LayerType: "linear",
	}
	
	// Fill with repetitive data (good for Huffman)
	for i := range tensor.Data {
		tensor.Data[i] = float32(i % 4) // Only 4 unique values
	}
	
	compressed, err := compressor.CompressTensor(tensor, TensorCompressionHuffman)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	
	if compressed.Algorithm != TensorCompressionHuffman {
		t.Errorf("Expected algorithm %v, got %v", TensorCompressionHuffman, compressed.Algorithm)
	}
	
	// Should achieve some compression due to repetitive data
	if compressed.CompressionRatio >= 1.0 {
		t.Error("Expected compression ratio < 1.0")
	}
	
	// Test decompression
	decompressed, err := compressor.DecompressTensor(compressed)
	if err != nil {
		t.Fatalf("Decompression failed: %v", err)
	}
	
	// Verify data integrity
	for i, originalVal := range tensor.Data {
		if math.Abs(float64(decompressed.Data[i]-originalVal)) > 1e-6 {
			t.Errorf("Data mismatch at index %d: expected %f, got %f", i, originalVal, decompressed.Data[i])
		}
	}
}

func TestTensorCompressor_GetTensorCompressionStats(t *testing.T) {
	config := DefaultGradientCompressionConfig()
	compressor := NewTensorCompressor(config)
	
	tensor := &ModelTensor{
		Name:      "test_weight",
		Shape:     []int{5, 5},
		Data:      make([]float32, 25),
		Type:      TensorTypeWeight,
		LayerType: "linear",
	}
	
	for i := range tensor.Data {
		tensor.Data[i] = float32(i)
	}
	
	compressed, err := compressor.CompressTensor(tensor, TensorCompressionQuantization)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	
	stats := compressor.GetTensorCompressionStats(compressed)
	
	// Check required stats
	requiredKeys := []string{
		"algorithm", "compression_ratio", "original_size_bytes",
		"compressed_size_bytes", "parameter_reduction", "tensor_type", "tensor_shape",
	}
	
	for _, key := range requiredKeys {
		if _, ok := stats[key]; !ok {
			t.Errorf("Stats should include %s", key)
		}
	}
	
	// Verify values
	if stats["algorithm"] != TensorCompressionQuantization {
		t.Errorf("Expected algorithm %v in stats, got %v", TensorCompressionQuantization, stats["algorithm"])
	}
	
	if stats["tensor_type"] != TensorTypeWeight {
		t.Errorf("Expected tensor type %v in stats, got %v", TensorTypeWeight, stats["tensor_type"])
	}
	
	originalSizeBytes := stats["original_size_bytes"].(int)
	expectedOriginalSize := 25 * 4 // 25 float32s
	if originalSizeBytes != expectedOriginalSize {
		t.Errorf("Expected original size %d bytes, got %d", expectedOriginalSize, originalSizeBytes)
	}
}

func TestTensorCompressor_NonSquareMatrix_SVD(t *testing.T) {
	config := DefaultGradientCompressionConfig()
	compressor := NewTensorCompressor(config)
	
	// Test non-square matrix
	tensor := &ModelTensor{
		Name:      "test_weight",
		Shape:     []int{3, 5}, // Non-square
		Data:      make([]float32, 15),
		Type:      TensorTypeWeight,
		LayerType: "linear",
	}
	
	for i := range tensor.Data {
		tensor.Data[i] = float32(i + 1)
	}
	
	compressed, err := compressor.CompressTensor(tensor, TensorCompressionSVD)
	if err != nil {
		t.Fatalf("Expected no error for non-square matrix, got %v", err)
	}
	
	// Test decompression
	decompressed, err := compressor.DecompressTensor(compressed)
	if err != nil {
		t.Fatalf("Decompression failed: %v", err)
	}
	
	if len(decompressed.Data) != len(tensor.Data) {
		t.Errorf("Expected data length %d, got %d", len(tensor.Data), len(decompressed.Data))
	}
}

func TestTensorCompressor_1D_Tensor(t *testing.T) {
	config := DefaultGradientCompressionConfig()
	compressor := NewTensorCompressor(config)
	
	// Test 1D tensor (bias)
	tensor := &ModelTensor{
		Name:      "test_bias",
		Shape:     []int{10},
		Data:      []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0},
		Type:      TensorTypeBias,
		LayerType: "linear",
	}
	
	// SVD should fallback to another method for 1D tensors
	compressed, err := compressor.CompressTensor(tensor, TensorCompressionSVD)
	if err != nil {
		t.Fatalf("Expected no error for 1D tensor, got %v", err)
	}
	
	// Should not use SVD algorithm
	if compressed.Algorithm == TensorCompressionSVD {
		t.Error("SVD should not be used for 1D tensors")
	}
}

func TestTensorCompressor_QuantizationBits(t *testing.T) {
	config := DefaultGradientCompressionConfig()
	compressor := NewTensorCompressor(config)
	
	tensor := &ModelTensor{
		Name:      "test_weight",
		Shape:     []int{4, 4},
		Data:      make([]float32, 16),
		Type:      TensorTypeWeight,
		LayerType: "linear",
	}
	
	for i := range tensor.Data {
		tensor.Data[i] = float32(i) - 8.0 // Center around zero
	}
	
	testCases := []struct {
		bits     QuantizationBits
		expected float64
	}{
		{Bits16, 0.5},
		{Bits8, 0.25},
		{Bits4, 0.125},
	}
	
	for _, tc := range testCases {
		config.QuantizationBits = tc.bits
		compressor.UpdateConfig(config)
		
		compressed, err := compressor.CompressTensor(tensor, TensorCompressionQuantization)
		if err != nil {
			t.Fatalf("Expected no error for %d bits, got %v", tc.bits, err)
		}
		
		if math.Abs(compressed.CompressionRatio-tc.expected) > 0.05 {
			t.Errorf("For %d bits, expected ratio ~%f, got %f", tc.bits, tc.expected, compressed.CompressionRatio)
		}
		
		// Test decompression
		decompressed, err := compressor.DecompressTensor(compressed)
		if err != nil {
			t.Fatalf("Decompression failed for %d bits: %v", tc.bits, err)
		}
		
		if len(decompressed.Data) != len(tensor.Data) {
			t.Errorf("Data length mismatch for %d bits", tc.bits)
		}
	}
}

func BenchmarkTensorCompressor_Pruning(b *testing.B) {
	config := DefaultGradientCompressionConfig()
	config.CompressionRatio = 0.1
	compressor := NewTensorCompressor(config)
	
	tensor := &ModelTensor{
		Name:      "benchmark_weight",
		Shape:     []int{100, 100},
		Data:      make([]float32, 10000),
		Type:      TensorTypeWeight,
		LayerType: "linear",
	}
	
	for i := range tensor.Data {
		tensor.Data[i] = float32(i) * 0.001
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := compressor.CompressTensor(tensor, TensorCompressionPruning)
		if err != nil {
			b.Fatalf("Compression failed: %v", err)
		}
	}
}

func BenchmarkTensorCompressor_Quantization(b *testing.B) {
	config := DefaultGradientCompressionConfig()
	config.QuantizationBits = Bits8
	compressor := NewTensorCompressor(config)
	
	tensor := &ModelTensor{
		Name:      "benchmark_weight",
		Shape:     []int{100, 100},
		Data:      make([]float32, 10000),
		Type:      TensorTypeWeight,
		LayerType: "linear",
	}
	
	for i := range tensor.Data {
		tensor.Data[i] = float32(math.Sin(float64(i))) * 10.0
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := compressor.CompressTensor(tensor, TensorCompressionQuantization)
		if err != nil {
			b.Fatalf("Compression failed: %v", err)
		}
	}
}

func BenchmarkTensorCompressor_KMeans(b *testing.B) {
	config := DefaultGradientCompressionConfig()
	config.QuantizationBits = Bits8 // 256 clusters
	compressor := NewTensorCompressor(config)
	
	tensor := &ModelTensor{
		Name:      "benchmark_weight",
		Shape:     []int{50, 50},
		Data:      make([]float32, 2500),
		Type:      TensorTypeWeight,
		LayerType: "linear",
	}
	
	for i := range tensor.Data {
		tensor.Data[i] = float32(i%10) * 0.1 // Create clusterable data
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := compressor.CompressTensor(tensor, TensorCompressionKMeans)
		if err != nil {
			b.Fatalf("Compression failed: %v", err)
		}
	}
}