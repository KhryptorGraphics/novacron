package ml

import (
	"math"
	"testing"
)

func TestSparsifier_SparsifyTensor_Magnitude(t *testing.T) {
	config := DefaultSparsificationConfig()
	config.Algorithm = SparsificationMagnitude
	config.SparsityRatio = 0.6 // Remove 60% of elements
	
	sparsifier := NewSparsifier(config)
	
	tensor := &GradientTensor{
		Shape: []int{10},
		Data:  []float32{0.1, 5.0, 0.05, 8.0, 0.2, 2.0, 0.01, 9.0, 0.3, 1.5},
	}
	
	sparse, metrics, err := sparsifier.SparsifyTensor(tensor, "test_layer")
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	
	if sparse.Algorithm != SparsificationMagnitude {
		t.Errorf("Expected algorithm %v, got %v", SparsificationMagnitude, sparse.Algorithm)
	}
	
	expectedSparsity := 0.6
	if math.Abs(sparse.SparsityLevel-expectedSparsity) > 0.1 {
		t.Errorf("Expected sparsity ~%f, got %f", expectedSparsity, sparse.SparsityLevel)
	}
	
	// Check metrics
	if metrics.OriginalSize != len(tensor.Data) {
		t.Errorf("Expected original size %d, got %d", len(tensor.Data), metrics.OriginalSize)
	}
	
	expectedNonZero := int(float64(len(tensor.Data)) * (1.0 - config.SparsityRatio))
	if math.Abs(float64(metrics.NonZeroCount-expectedNonZero)) > 1 {
		t.Errorf("Expected ~%d non-zero elements, got %d", expectedNonZero, metrics.NonZeroCount)
	}
	
	// Test densification
	densified, err := sparsifier.DensifyTensor(sparse)
	if err != nil {
		t.Fatalf("Densification failed: %v", err)
	}
	
	if len(densified.Data) != len(tensor.Data) {
		t.Errorf("Expected densified length %d, got %d", len(tensor.Data), len(densified.Data))
	}
	
	// Count non-zero elements in densified tensor
	nonZeroCount := 0
	for _, val := range densified.Data {
		if val != 0.0 {
			nonZeroCount++
		}
	}
	
	if nonZeroCount != len(sparse.Values) {
		t.Errorf("Expected %d non-zero elements after densification, got %d", len(sparse.Values), nonZeroCount)
	}
}

func TestSparsifier_SparsifyTensor_Random(t *testing.T) {
	config := DefaultSparsificationConfig()
	config.Algorithm = SparsificationRandom
	config.SparsityRatio = 0.7
	
	sparsifier := NewSparsifier(config)
	
	tensor := &GradientTensor{
		Shape: []int{20},
		Data:  make([]float32, 20),
	}
	
	for i := range tensor.Data {
		tensor.Data[i] = float32(i + 1)
	}
	
	sparse, metrics, err := sparsifier.SparsifyTensor(tensor, "test_layer")
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	
	if sparse.Algorithm != SparsificationRandom {
		t.Errorf("Expected algorithm %v, got %v", SparsificationRandom, sparse.Algorithm)
	}
	
	expectedSparsity := 0.7
	if math.Abs(sparse.SparsityLevel-expectedSparsity) > 0.1 {
		t.Errorf("Expected sparsity ~%f, got %f", expectedSparsity, sparse.SparsityLevel)
	}
	
	expectedNonZero := int(float64(len(tensor.Data)) * (1.0 - config.SparsityRatio))
	if math.Abs(float64(metrics.NonZeroCount-expectedNonZero)) > 1 {
		t.Errorf("Expected ~%d non-zero elements, got %d", expectedNonZero, metrics.NonZeroCount)
	}
}

func TestSparsifier_SparsifyTensor_Structured_Channel(t *testing.T) {
	config := DefaultSparsificationConfig()
	config.Algorithm = SparsificationStructured
	config.Granularity = GranularityChannel
	config.SparsityRatio = 0.5
	
	sparsifier := NewSparsifier(config)
	
	// Create 4D tensor (NCHW format): 1x4x2x2
	tensor := &GradientTensor{
		Shape: []int{1, 4, 2, 2}, // 1 batch, 4 channels, 2x2 spatial
		Data:  make([]float32, 16),
	}
	
	// Fill channels with different magnitudes
	for c := 0; c < 4; c++ {
		for i := 0; i < 4; i++ { // 2x2 = 4 elements per channel
			tensor.Data[c*4+i] = float32(c+1) * 0.5 // Channel 0: 0.5, Channel 1: 1.0, etc.
		}
	}
	
	sparse, _, err := sparsifier.SparsifyTensor(tensor, "test_layer")
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	
	if sparse.Algorithm != SparsificationStructured {
		t.Errorf("Expected algorithm %v, got %v", SparsificationStructured, sparse.Algorithm)
	}
	
	// Should have channel-level metadata
	if sparse.Metadata == nil {
		t.Error("Expected metadata for structured sparsification")
	} else {
		if granularity, ok := sparse.Metadata["granularity"]; !ok || granularity != "channel" {
			t.Error("Expected channel granularity in metadata")
		}
	}
	
	// Test densification
	densified, err := sparsifier.DensifyTensor(sparse)
	if err != nil {
		t.Fatalf("Densification failed: %v", err)
	}
	
	// Check that entire channels are either preserved or zeroed
	channelSize := 4
	for c := 0; c < 4; c++ {
		start := c * channelSize
		channelSum := float32(0)
		for i := 0; i < channelSize; i++ {
			if densified.Data[start+i] < 0 {
			channelSum += -densified.Data[start+i]
		} else {
			channelSum += densified.Data[start+i]
		}
		}
		// Channel should be either completely zero or completely preserved
		if channelSum != 0 && channelSum < 1.0 {
			t.Errorf("Channel %d partially preserved (sum=%f), expected all or nothing", c, channelSum)
		}
	}
}

func TestSparsifier_SparsifyTensor_Structured_Block(t *testing.T) {
	config := DefaultSparsificationConfig()
	config.Algorithm = SparsificationStructured
	config.Granularity = GranularityBlock
	config.BlockSize = []int{2, 2}
	config.SparsityRatio = 0.5
	
	sparsifier := NewSparsifier(config)
	
	// Create 2D tensor: 4x4
	tensor := &GradientTensor{
		Shape: []int{4, 4},
		Data:  make([]float32, 16),
	}
	
	// Fill with block-structured data
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			blockRow := i / 2
			blockCol := j / 2
			blockId := blockRow*2 + blockCol
			tensor.Data[i*4+j] = float32(blockId+1) * 0.25
		}
	}
	
	sparse, _, err := sparsifier.SparsifyTensor(tensor, "test_layer")
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	
	if sparse.Algorithm != SparsificationStructured {
		t.Errorf("Expected algorithm %v, got %v", SparsificationStructured, sparse.Algorithm)
	}
	
	// Should have block-level metadata
	if sparse.Metadata == nil {
		t.Error("Expected metadata for block sparsification")
	} else {
		if granularity, ok := sparse.Metadata["granularity"]; !ok || granularity != "block" {
			t.Error("Expected block granularity in metadata")
		}
	}
}

func TestSparsifier_SparsifyTensor_Gradient(t *testing.T) {
	config := DefaultSparsificationConfig()
	config.Algorithm = SparsificationGradient
	config.SparsityRatio = 0.6
	
	sparsifier := NewSparsifier(config)
	
	tensor := &GradientTensor{
		Shape: []int{8},
		Data:  []float32{0.1, 2.0, 0.05, 3.0, 0.2, 1.5, 0.01, 4.0},
	}
	
	// Run sparsification twice to build gradient history
	_, _, err := sparsifier.SparsifyTensor(tensor, "test_layer")
	if err != nil {
		t.Fatalf("First sparsification failed: %v", err)
	}
	
	sparse, metrics, err := sparsifier.SparsifyTensor(tensor, "test_layer")
	if err != nil {
		t.Fatalf("Second sparsification failed: %v", err)
	}
	
	if sparse.Algorithm != SparsificationGradient {
		t.Errorf("Expected algorithm %v, got %v", SparsificationGradient, sparse.Algorithm)
	}
	
	expectedSparsity := 0.6
	if math.Abs(sparse.SparsityLevel-expectedSparsity) > 0.1 {
		t.Errorf("Expected sparsity ~%f, got %f", expectedSparsity, sparse.SparsityLevel)
	}
	
	if metrics.SparsificationTimeMs <= 0 {
		t.Error("Expected positive sparsification time")
	}
}

func TestSparsifier_SparsifyTensor_Adaptive(t *testing.T) {
	config := DefaultSparsificationConfig()
	config.Algorithm = SparsificationAdaptive
	config.SparsityRatio = 0.8
	config.MinSparsity = 0.5
	config.MaxSparsity = 0.95
	
	sparsifier := NewSparsifier(config)
	
	// Test with different tensor shapes (should adapt differently)
	testCases := []struct {
		shape    []int
		expected string
	}{
		{[]int{10, 10}, "fully_connected"}, // 2D -> FC layer
		{[]int{3, 16, 5, 5}, "convolutional"}, // 4D -> Conv layer
		{[]int{100}, "other"}, // 1D -> Other
	}
	
	for _, tc := range testCases {
		tensor := &GradientTensor{
			Shape: tc.shape,
			Data:  make([]float32, 1),
		}
		
		// Calculate total size
		totalSize := 1
		for _, dim := range tc.shape {
			totalSize *= dim
		}
		tensor.Data = make([]float32, totalSize)
		
		for i := range tensor.Data {
			tensor.Data[i] = float32(i) * 0.01
		}
		
		sparse, _, err := sparsifier.SparsifyTensor(tensor, "adaptive_test")
		if err != nil {
			t.Fatalf("Adaptive sparsification failed for shape %v: %v", tc.shape, err)
		}
		
		if sparse.Algorithm != SparsificationAdaptive {
			t.Errorf("Expected algorithm %v, got %v", SparsificationAdaptive, sparse.Algorithm)
		}
		
		// Check that sparsity is within bounds
		if sparse.SparsityLevel < config.MinSparsity {
			t.Errorf("Sparsity %f below minimum %f for shape %v", sparse.SparsityLevel, config.MinSparsity, tc.shape)
		}
		
		if sparse.SparsityLevel > config.MaxSparsity {
			t.Errorf("Sparsity %f above maximum %f for shape %v", sparse.SparsityLevel, config.MaxSparsity, tc.shape)
		}
	}
}

func TestSparsifier_UpdateConfig(t *testing.T) {
	config1 := DefaultSparsificationConfig()
	config1.Algorithm = SparsificationMagnitude
	config1.SparsityRatio = 0.5
	
	sparsifier := NewSparsifier(config1)
	
	// Update configuration
	config2 := DefaultSparsificationConfig()
	config2.Algorithm = SparsificationRandom
	config2.SparsityRatio = 0.8
	
	sparsifier.UpdateConfig(config2)
	
	// Test that new config is used
	tensor := &GradientTensor{
		Shape: []int{10},
		Data:  make([]float32, 10),
	}
	
	for i := range tensor.Data {
		tensor.Data[i] = float32(i + 1)
	}
	
	sparse, _, err := sparsifier.SparsifyTensor(tensor, "test_layer")
	if err != nil {
		t.Fatalf("Sparsification with updated config failed: %v", err)
	}
	
	if sparse.Algorithm != SparsificationRandom {
		t.Errorf("Expected updated algorithm %v, got %v", SparsificationRandom, sparse.Algorithm)
	}
	
	expectedSparsity := 0.8
	if math.Abs(sparse.SparsityLevel-expectedSparsity) > 0.1 {
		t.Errorf("Expected updated sparsity ~%f, got %f", expectedSparsity, sparse.SparsityLevel)
	}
}

func TestSparsifier_GetSparsificationStats(t *testing.T) {
	config := DefaultSparsificationConfig()
	sparsifier := NewSparsifier(config)
	
	tensor := &GradientTensor{
		Shape: []int{20},
		Data:  make([]float32, 20),
	}
	
	for i := range tensor.Data {
		tensor.Data[i] = float32(i + 1)
	}
	
	sparse, metrics, err := sparsifier.SparsifyTensor(tensor, "test_layer")
	if err != nil {
		t.Fatalf("Sparsification failed: %v", err)
	}
	
	stats := sparsifier.GetSparsificationStats(sparse, metrics)
	
	// Check required stats
	requiredKeys := []string{
		"algorithm", "sparsity_level", "original_size", "non_zero_count",
		"compression_ratio", "flops_reduction", "sparsification_time_ms", "memory_saved_bytes",
	}
	
	for _, key := range requiredKeys {
		if _, ok := stats[key]; !ok {
			t.Errorf("Stats should include %s", key)
		}
	}
	
	// Verify values
	if stats["algorithm"] != sparse.Algorithm {
		t.Errorf("Expected algorithm %v in stats, got %v", sparse.Algorithm, stats["algorithm"])
	}
	
	if stats["original_size"] != metrics.OriginalSize {
		t.Errorf("Expected original size %d in stats, got %v", metrics.OriginalSize, stats["original_size"])
	}
}

func TestSparsifier_EstimateSparsificationBenefit(t *testing.T) {
	config := DefaultSparsificationConfig()
	config.SparsityRatio = 0.7
	sparsifier := NewSparsifier(config)
	
	tensorSize := 1000
	benefit := sparsifier.EstimateSparsificationBenefit(tensorSize, config)
	
	// Check required benefit metrics
	requiredKeys := []string{"memory_saving", "flops_reduction", "accuracy_impact", "compression_ratio"}
	
	for _, key := range requiredKeys {
		if _, ok := benefit[key]; !ok {
			t.Errorf("Benefit estimation should include %s", key)
		}
	}
	
	// Verify values are reasonable
	memorySaving := benefit["memory_saving"]
	if memorySaving != config.SparsityRatio {
		t.Errorf("Expected memory saving %f, got %f", config.SparsityRatio, memorySaving)
	}
	
	compressionRatio := benefit["compression_ratio"]
	expectedRatio := 1.0 - config.SparsityRatio
	if compressionRatio != expectedRatio {
		t.Errorf("Expected compression ratio %f, got %f", expectedRatio, compressionRatio)
	}
}

func TestSparsifier_CompareSparsificationMethods(t *testing.T) {
	config := DefaultSparsificationConfig()
	sparsifier := NewSparsifier(config)
	
	tensor := &GradientTensor{
		Shape: []int{50},
		Data:  make([]float32, 50),
	}
	
	for i := range tensor.Data {
		tensor.Data[i] = float32(i) * 0.1
	}
	
	results := sparsifier.CompareSparsificationMethods(tensor, "comparison_test")
	
	// Should have results for multiple algorithms
	expectedAlgorithms := []SparsificationAlgorithm{
		SparsificationMagnitude,
		SparsificationRandom,
		SparsificationStructured,
	}
	
	for _, alg := range expectedAlgorithms {
		if _, ok := results[alg]; !ok {
			t.Errorf("Expected results for algorithm %v", alg)
		} else {
			result := results[alg]
			if _, hasSparsity := result["sparsity_level"]; !hasSparsity {
				t.Errorf("Result for %v should include sparsity_level", alg)
			}
			if _, hasRatio := result["compression_ratio"]; !hasRatio {
				t.Errorf("Result for %v should include compression_ratio", alg)
			}
		}
	}
}

func TestSparsifier_EmptyTensor(t *testing.T) {
	config := DefaultSparsificationConfig()
	sparsifier := NewSparsifier(config)
	
	// Test with nil tensor
	_, _, err := sparsifier.SparsifyTensor(nil, "test_layer")
	if err == nil {
		t.Error("Expected error for nil tensor")
	}
	
	// Test with empty tensor
	emptyTensor := &GradientTensor{
		Shape: []int{0},
		Data:  []float32{},
	}
	
	_, _, err = sparsifier.SparsifyTensor(emptyTensor, "test_layer")
	if err == nil {
		t.Error("Expected error for empty tensor")
	}
}

func TestSparsifier_DensifyTensor_Invalid(t *testing.T) {
	config := DefaultSparsificationConfig()
	sparsifier := NewSparsifier(config)
	
	// Test with nil sparse data
	_, err := sparsifier.DensifyTensor(nil)
	if err == nil {
		t.Error("Expected error for nil sparse data")
	}
}

func TestSparsifier_ExtremeSparsity(t *testing.T) {
	config := DefaultSparsificationConfig()
	config.SparsityRatio = 0.99 // Very high sparsity
	sparsifier := NewSparsifier(config)
	
	tensor := &GradientTensor{
		Shape: []int{100},
		Data:  make([]float32, 100),
	}
	
	for i := range tensor.Data {
		tensor.Data[i] = float32(i + 1)
	}
	
	sparse, metrics, err := sparsifier.SparsifyTensor(tensor, "extreme_test")
	if err != nil {
		t.Fatalf("Extreme sparsification failed: %v", err)
	}
	
	// Should keep at least 1 element
	if metrics.NonZeroCount < 1 {
		t.Error("Should keep at least 1 element even with extreme sparsity")
	}
	
	// Test densification still works
	_, err = sparsifier.DensifyTensor(sparse)
	if err != nil {
		t.Fatalf("Densification of extremely sparse tensor failed: %v", err)
	}
}

func BenchmarkSparsifier_Magnitude(b *testing.B) {
	config := DefaultSparsificationConfig()
	config.Algorithm = SparsificationMagnitude
	config.SparsityRatio = 0.9
	
	sparsifier := NewSparsifier(config)
	
	tensor := &GradientTensor{
		Shape: []int{10000},
		Data:  make([]float32, 10000),
	}
	
	for i := range tensor.Data {
		tensor.Data[i] = float32(i) * 0.0001
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, err := sparsifier.SparsifyTensor(tensor, "benchmark_layer")
		if err != nil {
			b.Fatalf("Sparsification failed: %v", err)
		}
	}
}

func BenchmarkSparsifier_Random(b *testing.B) {
	config := DefaultSparsificationConfig()
	config.Algorithm = SparsificationRandom
	config.SparsityRatio = 0.9
	
	sparsifier := NewSparsifier(config)
	
	tensor := &GradientTensor{
		Shape: []int{10000},
		Data:  make([]float32, 10000),
	}
	
	for i := range tensor.Data {
		tensor.Data[i] = float32(i) * 0.0001
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, err := sparsifier.SparsifyTensor(tensor, "benchmark_layer")
		if err != nil {
			b.Fatalf("Sparsification failed: %v", err)
		}
	}
}

func BenchmarkSparsifier_Structured(b *testing.B) {
	config := DefaultSparsificationConfig()
	config.Algorithm = SparsificationStructured
	config.Granularity = GranularityChannel
	config.SparsityRatio = 0.5
	
	sparsifier := NewSparsifier(config)
	
	// 4D tensor for channel-wise sparsification
	tensor := &GradientTensor{
		Shape: []int{1, 64, 32, 32}, // Typical conv layer
		Data:  make([]float32, 1*64*32*32),
	}
	
	for i := range tensor.Data {
		tensor.Data[i] = float32(i%1000) * 0.001
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, err := sparsifier.SparsifyTensor(tensor, "benchmark_layer")
		if err != nil {
			b.Fatalf("Sparsification failed: %v", err)
		}
	}
}