package ml

import (
	"context"
	"fmt"
	"testing"
	"time"
)

func TestCompressionPipeline_CompressGradient(t *testing.T) {
	config := DefaultCompressionPipelineConfig()
	pipeline := NewCompressionPipeline(config)
	
	gradient := &GradientTensor{
		Shape:     []int{100},
		Data:      make([]float32, 100),
		LayerName: "test_layer",
		Timestamp: time.Now().Unix(),
	}
	
	for i := range gradient.Data {
		gradient.Data[i] = float32(i) * 0.01
	}
	
	ctx := context.Background()
	result, err := pipeline.CompressGradient(ctx, gradient)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	
	if !result.Success {
		t.Error("Expected successful compression")
	}
	
	if result.CompressionRatio >= 1.0 {
		t.Errorf("Expected compression ratio < 1.0, got %f", result.CompressionRatio)
	}
	
	if result.Method != "gradient_compression" {
		t.Errorf("Expected method 'gradient_compression', got %s", result.Method)
	}
	
	if result.ProcessingTimeMs <= 0 {
		t.Error("Expected positive processing time")
	}
}

func TestCompressionPipeline_CompressTensor(t *testing.T) {
	config := DefaultCompressionPipelineConfig()
	pipeline := NewCompressionPipeline(config)
	
	tensor := &ModelTensor{
		Name:      "test_weight",
		Shape:     []int{10, 10},
		Data:      make([]float32, 100),
		Type:      TensorTypeWeight,
		LayerType: "linear",
	}
	
	for i := range tensor.Data {
		tensor.Data[i] = float32(i) * 0.1
	}
	
	ctx := context.Background()
	result, err := pipeline.CompressTensor(ctx, tensor)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	
	if !result.Success {
		t.Error("Expected successful compression")
	}
	
	if result.CompressionRatio >= 1.0 {
		t.Errorf("Expected compression ratio < 1.0, got %f", result.CompressionRatio)
	}
	
	if result.Method != "tensor_compression" {
		t.Errorf("Expected method 'tensor_compression', got %s", result.Method)
	}
}

func TestCompressionPipeline_CompressBatch(t *testing.T) {
	config := DefaultCompressionPipelineConfig()
	config.MaxConcurrency = 2
	pipeline := NewCompressionPipeline(config)
	
	// Create batch of gradients
	gradients := make([]*GradientTensor, 3)
	for i := 0; i < 3; i++ {
		gradients[i] = &GradientTensor{
			Shape:     []int{50},
			Data:      make([]float32, 50),
			LayerName: fmt.Sprintf("layer_%d", i),
			Timestamp: time.Now().Unix(),
		}
		
		for j := range gradients[i].Data {
			gradients[i].Data[j] = float32(j+i*10) * 0.01
		}
	}
	
	// Create batch of tensors
	tensors := make([]*ModelTensor, 2)
	for i := 0; i < 2; i++ {
		tensors[i] = &ModelTensor{
			Name:      fmt.Sprintf("tensor_%d", i),
			Shape:     []int{8, 8},
			Data:      make([]float32, 64),
			Type:      TensorTypeWeight,
			LayerType: "linear",
		}
		
		for j := range tensors[i].Data {
			tensors[i].Data[j] = float32(j+i*20) * 0.05
		}
	}
	
	ctx := context.Background()
	results, err := pipeline.CompressBatch(ctx, gradients, tensors)
	if err != nil {
		t.Fatalf("Batch compression failed: %v", err)
	}
	
	expectedResults := len(gradients) + len(tensors)
	if len(results) != expectedResults {
		t.Errorf("Expected %d results, got %d", expectedResults, len(results))
	}
	
	// Check that all results are successful
	for i, result := range results {
		if result == nil {
			t.Errorf("Result %d is nil", i)
			continue
		}
		
		if !result.Success {
			t.Errorf("Result %d failed: %s", i, result.ErrorMessage)
		}
	}
}

func TestCompressionPipeline_ThresholdFiltering(t *testing.T) {
	config := DefaultCompressionPipelineConfig()
	config.CompressionThresholds.MinGradientSize = 1000 // High threshold
	pipeline := NewCompressionPipeline(config)
	
	// Small gradient below threshold
	smallGradient := &GradientTensor{
		Shape:     []int{10}, // 10 * 4 = 40 bytes, below 1000 byte threshold
		Data:      make([]float32, 10),
		LayerName: "small_layer",
		Timestamp: time.Now().Unix(),
	}
	
	ctx := context.Background()
	result, err := pipeline.CompressGradient(ctx, smallGradient)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	
	if result.Success {
		t.Error("Expected compression to be skipped due to size threshold")
	}
	
	if result.ErrorMessage != "gradient size below threshold" {
		t.Errorf("Expected threshold error message, got %s", result.ErrorMessage)
	}
}

func TestCompressionPipeline_AdaptiveCompression(t *testing.T) {
	config := DefaultCompressionPipelineConfig()
	config.QualitySettings.AdaptiveCompression = true
	pipeline := NewCompressionPipeline(config)
	
	testCases := []struct {
		tensorType TensorType
		layerType  string
		expected   TensorCompressionAlgorithm
	}{
		{TensorTypeWeight, "linear", TensorCompressionPruning},
		{TensorTypeWeight, "conv", TensorCompressionQuantization},
		{TensorTypeBias, "linear", TensorCompressionQuantization},
		{TensorTypeEmbedding, "embedding", TensorCompressionSVD},
		{TensorTypeAttentionWeight, "attention", TensorCompressionLowRank},
	}
	
	for _, tc := range testCases {
		tensor := &ModelTensor{
			Name:      "adaptive_test",
			Shape:     []int{16, 16},
			Data:      make([]float32, 256),
			Type:      tc.tensorType,
			LayerType: tc.layerType,
		}
		
		for i := range tensor.Data {
			tensor.Data[i] = float32(i) * 0.01
		}
		
		selected := pipeline.selectTensorCompressionAlgorithm(tensor)
		if selected != tc.expected {
			t.Errorf("For type %v layer %s, expected algorithm %v, got %v", 
				tc.tensorType, tc.layerType, tc.expected, selected)
		}
	}
}

func TestCompressionPipeline_QualitySettings(t *testing.T) {
	// High quality settings
	highQualityConfig := DefaultCompressionPipelineConfig()
	highQualityConfig.QualitySettings.QualityLevel = 0.9
	highQualityConfig.QualitySettings.AdaptiveCompression = true
	highQualityPipeline := NewCompressionPipeline(highQualityConfig)
	
	// Low quality settings  
	lowQualityConfig := DefaultCompressionPipelineConfig()
	lowQualityConfig.QualitySettings.QualityLevel = 0.3
	lowQualityConfig.QualitySettings.AdaptiveCompression = true
	lowQualityPipeline := NewCompressionPipeline(lowQualityConfig)
	
	tensor := &ModelTensor{
		Name:      "quality_test",
		Shape:     []int{10, 10},
		Data:      make([]float32, 100),
		Type:      TensorTypeWeight,
		LayerType: "linear",
	}
	
	for i := range tensor.Data {
		tensor.Data[i] = float32(i) * 0.01
	}
	
	// High quality should prefer gentler compression
	highQualityAlg := highQualityPipeline.selectTensorCompressionAlgorithm(tensor)
	
	// Low quality should prefer aggressive compression
	lowQualityAlg := lowQualityPipeline.selectTensorCompressionAlgorithm(tensor)
	
	// For weight tensors, high quality should prefer quantization over pruning
	if highQualityAlg == TensorCompressionPruning && lowQualityAlg == TensorCompressionQuantization {
		t.Error("Expected high quality to prefer gentler compression than low quality")
	}
}

func TestCompressionPipeline_Statistics(t *testing.T) {
	config := DefaultCompressionPipelineConfig()
	pipeline := NewCompressionPipeline(config)
	
	// Initial statistics should be empty
	initialStats := pipeline.GetStatistics()
	if initialStats.TotalOperations != 0 {
		t.Error("Expected zero initial operations")
	}
	
	// Compress some data
	gradient := &GradientTensor{
		Shape:     []int{100},
		Data:      make([]float32, 100),
		LayerName: "stats_test",
		Timestamp: time.Now().Unix(),
	}
	
	for i := range gradient.Data {
		gradient.Data[i] = float32(i) * 0.01
	}
	
	ctx := context.Background()
	_, err := pipeline.CompressGradient(ctx, gradient)
	if err != nil {
		t.Fatalf("Compression failed: %v", err)
	}
	
	// Check updated statistics
	updatedStats := pipeline.GetStatistics()
	if updatedStats.TotalOperations != 1 {
		t.Errorf("Expected 1 total operation, got %d", updatedStats.TotalOperations)
	}
	
	if updatedStats.GradientOperations != 1 {
		t.Errorf("Expected 1 gradient operation, got %d", updatedStats.GradientOperations)
	}
	
	if updatedStats.TotalBytesProcessed <= 0 {
		t.Error("Expected positive bytes processed")
	}
	
	if updatedStats.AverageCompressionRatio <= 0 || updatedStats.AverageCompressionRatio >= 1 {
		t.Errorf("Expected compression ratio between 0 and 1, got %f", updatedStats.AverageCompressionRatio)
	}
}

func TestCompressionPipeline_EstimateCompressionBenefit(t *testing.T) {
	config := DefaultCompressionPipelineConfig()
	pipeline := NewCompressionPipeline(config)
	
	gradientSizes := []int64{1024, 2048, 4096} // 3 gradients
	tensorSizes := []int64{8192, 16384}        // 2 tensors
	
	benefit := pipeline.EstimateCompressionBenefit(gradientSizes, tensorSizes)
	
	// Check required benefit fields
	requiredFields := []string{
		"total_original_size_mb", "estimated_compressed_size_mb", 
		"compression_ratio", "memory_savings_mb", "estimated_processing_ms",
		"gradient_items", "tensor_items",
	}
	
	for _, field := range requiredFields {
		if _, ok := benefit[field]; !ok {
			t.Errorf("Benefit estimation should include %s", field)
		}
	}
	
	// Verify counts
	if benefit["gradient_items"] != len(gradientSizes) {
		t.Errorf("Expected %d gradient items, got %v", len(gradientSizes), benefit["gradient_items"])
	}
	
	if benefit["tensor_items"] != len(tensorSizes) {
		t.Errorf("Expected %d tensor items, got %v", len(tensorSizes), benefit["tensor_items"])
	}
	
	// Verify compression ratio is reasonable
	ratio := benefit["compression_ratio"].(float64)
	if ratio <= 0 || ratio >= 1 {
		t.Errorf("Expected compression ratio between 0 and 1, got %f", ratio)
	}
}

func TestCompressionPipeline_UpdateConfig(t *testing.T) {
	initialConfig := DefaultCompressionPipelineConfig()
	initialConfig.EnableSparsification = false
	
	pipeline := NewCompressionPipeline(initialConfig)
	
	// Test initial configuration
	gradient := &GradientTensor{
		Shape:     []int{50},
		Data:      make([]float32, 50),
		LayerName: "config_test",
		Timestamp: time.Now().Unix(),
	}
	
	for i := range gradient.Data {
		gradient.Data[i] = float32(i) * 0.01
	}
	
	ctx := context.Background()
	result1, err := pipeline.CompressGradient(ctx, gradient)
	if err != nil {
		t.Fatalf("Initial compression failed: %v", err)
	}
	
	// Should not have sparsification applied
	if result1.Metadata["sparsification_applied"] == true {
		t.Error("Expected no sparsification with initial config")
	}
	
	// Update configuration to enable sparsification
	updatedConfig := initialConfig
	updatedConfig.EnableSparsification = true
	pipeline.UpdateConfig(updatedConfig)
	
	// Test with updated configuration
	_, err = pipeline.CompressGradient(ctx, gradient)
	if err != nil {
		t.Fatalf("Updated compression failed: %v", err)
	}
	
	// May have sparsification applied (depends on effectiveness)
	// At least configuration should be updated
	stats := pipeline.GetStatistics()
	if stats.TotalOperations < 2 {
		t.Error("Expected at least 2 operations after config update")
	}
}

func TestCompressionPipeline_OptimizeConfiguration(t *testing.T) {
	pipeline := NewCompressionPipeline(DefaultCompressionPipelineConfig())
	
	testProfiles := []struct {
		profile  WorkloadProfile
		expected map[string]interface{}
	}{
		{
			WorkloadProfile{
				ModelType:         "transformer",
				ModelSize:         1e9,
				MemoryConstrained: false,
				SpeedCritical:     false,
			},
			map[string]interface{}{
				"tensor_algorithm": TensorCompressionLowRank,
				"sparsification":   SparsificationStructured,
			},
		},
		{
			WorkloadProfile{
				ModelType:         "cnn",
				ModelSize:         1e6,
				MemoryConstrained: true,
				SpeedCritical:     false,
			},
			map[string]interface{}{
				"tensor_algorithm": TensorCompressionQuantization,
				"memory_aggressive": true,
			},
		},
	}
	
	for _, tc := range testProfiles {
		optimizedConfig := pipeline.OptimizeConfiguration(tc.profile)
		
		// Verify optimization based on profile
		if tc.profile.ModelType == "transformer" {
			if optimizedConfig.TensorConfig.Algorithm != TensorCompressionLowRank {
				t.Errorf("Expected low-rank for transformer, got %v", optimizedConfig.TensorConfig.Algorithm)
			}
		}
		
		if tc.profile.ModelType == "cnn" {
			if optimizedConfig.SparsificationConfig.Granularity != GranularityChannel {
				t.Errorf("Expected channel granularity for CNN, got %v", optimizedConfig.SparsificationConfig.Granularity)
			}
		}
		
		if tc.profile.MemoryConstrained {
			// Should have more aggressive compression
			if optimizedConfig.GradientConfig.CompressionRatio >= 0.1 {
				t.Error("Expected more aggressive gradient compression for memory-constrained profile")
			}
		}
		
		if tc.profile.SpeedCritical {
			// Should use faster algorithms
			if optimizedConfig.GradientConfig.Algorithm != CompressionTopK {
				t.Error("Expected TopK algorithm for speed-critical profile")
			}
			
			if optimizedConfig.MaxConcurrency < 8 {
				t.Error("Expected higher concurrency for speed-critical profile")
			}
		}
	}
}

func TestCompressionPipeline_GetCompressionRecommendations(t *testing.T) {
	pipeline := NewCompressionPipeline(DefaultCompressionPipelineConfig())
	
	testProfiles := []WorkloadProfile{
		{
			ModelType:         "transformer",
			ModelSize:         1e9,
			MemoryConstrained: false,
			SpeedCritical:     false,
			AccuracyTarget:    0.95,
		},
		{
			ModelType:         "cnn", 
			ModelSize:         1e6,
			MemoryConstrained: true,
			SpeedCritical:     true,
			AccuracyTarget:    0.90,
		},
	}
	
	for _, profile := range testProfiles {
		recommendations := pipeline.GetCompressionRecommendations(profile)
		
		// Check required recommendation fields
		requiredFields := []string{"explanation"}
		for _, field := range requiredFields {
			if _, ok := recommendations[field]; !ok {
				t.Errorf("Recommendations should include %s", field)
			}
		}
		
		// Check model-specific recommendations
		switch profile.ModelType {
		case "transformer":
			if recommendations["tensor_algorithm"] != "lowrank" {
				t.Error("Expected lowrank recommendation for transformer")
			}
		case "cnn":
			if recommendations["sparsification_granularity"] != "channel" {
				t.Error("Expected channel granularity recommendation for CNN")
			}
		}
		
		// Check constraint-based recommendations
		if profile.MemoryConstrained {
			if recommendations["priority"] != "memory_efficiency" {
				t.Error("Expected memory efficiency priority for constrained profile")
			}
		}
		
		if profile.SpeedCritical {
			if recommendations["parallel_processing"] != true {
				t.Error("Expected parallel processing recommendation for speed-critical profile")
			}
		}
		
		// Check size-based recommendations
		if profile.ModelSize > 1e9 {
			if recommendations["compression_aggressiveness"] != "high" {
				t.Error("Expected high aggressiveness for large models")
			}
		}
	}
}

func TestCompressionPipeline_ErrorHandling(t *testing.T) {
	config := DefaultCompressionPipelineConfig()
	pipeline := NewCompressionPipeline(config)
	
	ctx := context.Background()
	
	// Test nil gradient
	_, err := pipeline.CompressGradient(ctx, nil)
	if err == nil {
		t.Error("Expected error for nil gradient")
	}
	
	// Test nil tensor
	_, err = pipeline.CompressTensor(ctx, nil)
	if err == nil {
		t.Error("Expected error for nil tensor")
	}
	
	// Test empty batch
	_, err = pipeline.CompressBatch(ctx, nil, nil)
	if err == nil {
		t.Error("Expected error for empty batch")
	}
}

func BenchmarkCompressionPipeline_GradientCompression(b *testing.B) {
	config := DefaultCompressionPipelineConfig()
	pipeline := NewCompressionPipeline(config)
	
	gradient := &GradientTensor{
		Shape:     []int{1000},
		Data:      make([]float32, 1000),
		LayerName: "benchmark_layer",
		Timestamp: time.Now().Unix(),
	}
	
	for i := range gradient.Data {
		gradient.Data[i] = float32(i) * 0.001
	}
	
	ctx := context.Background()
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := pipeline.CompressGradient(ctx, gradient)
		if err != nil {
			b.Fatalf("Compression failed: %v", err)
		}
	}
}

func BenchmarkCompressionPipeline_TensorCompression(b *testing.B) {
	config := DefaultCompressionPipelineConfig()
	pipeline := NewCompressionPipeline(config)
	
	tensor := &ModelTensor{
		Name:      "benchmark_weight",
		Shape:     []int{100, 100},
		Data:      make([]float32, 10000),
		Type:      TensorTypeWeight,
		LayerType: "linear",
	}
	
	for i := range tensor.Data {
		tensor.Data[i] = float32(i) * 0.0001
	}
	
	ctx := context.Background()
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := pipeline.CompressTensor(ctx, tensor)
		if err != nil {
			b.Fatalf("Compression failed: %v", err)
		}
	}
}

func BenchmarkCompressionPipeline_BatchCompression(b *testing.B) {
	config := DefaultCompressionPipelineConfig()
	config.MaxConcurrency = 4
	pipeline := NewCompressionPipeline(config)
	
	// Create batch data
	gradients := make([]*GradientTensor, 10)
	for i := 0; i < 10; i++ {
		gradients[i] = &GradientTensor{
			Shape:     []int{500},
			Data:      make([]float32, 500),
			LayerName: fmt.Sprintf("layer_%d", i),
			Timestamp: time.Now().Unix(),
		}
		
		for j := range gradients[i].Data {
			gradients[i].Data[j] = float32(j+i*100) * 0.001
		}
	}
	
	ctx := context.Background()
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := pipeline.CompressBatch(ctx, gradients, nil)
		if err != nil {
			b.Fatalf("Batch compression failed: %v", err)
		}
	}
}