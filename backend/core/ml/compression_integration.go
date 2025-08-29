package ml

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"sync"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/storage/compression"
)

// CompressionPipeline manages the entire ML compression pipeline
type CompressionPipeline struct {
	gradientCompressor *GradientCompressor
	tensorCompressor   *TensorCompressor
	sparsifier         *Sparsifier
	storageCompressor  *compression.Compressor
	
	config CompressionPipelineConfig
	mutex  sync.RWMutex
	
	// Statistics tracking
	stats     *PipelineStatistics
	statsMux  sync.Mutex
}

// CompressionPipelineConfig holds configuration for the entire compression pipeline
type CompressionPipelineConfig struct {
	// EnableGradientCompression enables gradient compression
	EnableGradientCompression bool `json:"enable_gradient_compression"`
	
	// EnableTensorCompression enables model tensor compression
	EnableTensorCompression bool `json:"enable_tensor_compression"`
	
	// EnableSparsification enables sparsification algorithms
	EnableSparsification bool `json:"enable_sparsification"`
	
	// EnableStorageCompression enables storage-level compression
	EnableStorageCompression bool `json:"enable_storage_compression"`
	
	// GradientConfig configuration for gradient compression
	GradientConfig GradientCompressionConfig `json:"gradient_config"`
	
	// TensorConfig configuration for tensor compression
	TensorConfig TensorCompressionConfig `json:"tensor_config"`
	
	// SparsificationConfig configuration for sparsification
	SparsificationConfig SparsificationConfig `json:"sparsification_config"`
	
	// StorageConfig configuration for storage compression
	StorageConfig compression.CompressionConfig `json:"storage_config"`
	
	// CompressionThresholds size thresholds for applying compression
	CompressionThresholds CompressionThresholds `json:"compression_thresholds"`
	
	// QualitySettings quality vs compression trade-offs
	QualitySettings QualitySettings `json:"quality_settings"`
	
	// ParallelProcessing enables parallel compression processing
	ParallelProcessing bool `json:"parallel_processing"`
	
	// MaxConcurrency maximum concurrent compression operations
	MaxConcurrency int `json:"max_concurrency"`
}

// TensorCompressionConfig simplified tensor compression configuration
type TensorCompressionConfig struct {
	Algorithm        TensorCompressionAlgorithm `json:"algorithm"`
	CompressionRatio float64                    `json:"compression_ratio"`
	QuantizationBits QuantizationBits           `json:"quantization_bits"`
}

// CompressionThresholds defines size thresholds for applying different compression methods
type CompressionThresholds struct {
	// MinGradientSize minimum size in bytes to apply gradient compression
	MinGradientSize int64 `json:"min_gradient_size"`
	
	// MinTensorSize minimum size in bytes to apply tensor compression
	MinTensorSize int64 `json:"min_tensor_size"`
	
	// MinSparsificationSize minimum size to apply sparsification
	MinSparsificationSize int64 `json:"min_sparsification_size"`
	
	// MinStorageSize minimum size to apply storage compression
	MinStorageSize int64 `json:"min_storage_size"`
}

// QualitySettings defines quality vs compression trade-offs
type QualitySettings struct {
	// QualityLevel overall quality level (0.0-1.0)
	QualityLevel float64 `json:"quality_level"`
	
	// PreserveAccuracy prioritize accuracy over compression ratio
	PreserveAccuracy bool `json:"preserve_accuracy"`
	
	// MaxAccuracyLoss maximum acceptable accuracy loss
	MaxAccuracyLoss float64 `json:"max_accuracy_loss"`
	
	// AdaptiveCompression adjust compression based on importance
	AdaptiveCompression bool `json:"adaptive_compression"`
	
	// ErrorToleranceMode error tolerance for compression
	ErrorToleranceMode ErrorToleranceMode `json:"error_tolerance_mode"`
}

// ErrorToleranceMode defines error tolerance levels
type ErrorToleranceMode string

const (
	// ErrorToleranceStrict strict error bounds
	ErrorToleranceStrict ErrorToleranceMode = "strict"
	// ErrorToleranceBalanced balanced error tolerance
	ErrorToleranceBalanced ErrorToleranceMode = "balanced"
	// ErrorToleranceRelaxed relaxed error tolerance for maximum compression
	ErrorToleranceRelaxed ErrorToleranceMode = "relaxed"
)

// PipelineStatistics tracks compression pipeline statistics
type PipelineStatistics struct {
	// TotalOperations total compression operations
	TotalOperations int64 `json:"total_operations"`
	
	// GradientOperations gradient compression operations
	GradientOperations int64 `json:"gradient_operations"`
	
	// TensorOperations tensor compression operations
	TensorOperations int64 `json:"tensor_operations"`
	
	// SparsificationOperations sparsification operations
	SparsificationOperations int64 `json:"sparsification_operations"`
	
	// TotalBytesProcessed total bytes processed
	TotalBytesProcessed int64 `json:"total_bytes_processed"`
	
	// TotalBytesCompressed total bytes after compression
	TotalBytesCompressed int64 `json:"total_bytes_compressed"`
	
	// AverageCompressionRatio average compression ratio achieved
	AverageCompressionRatio float64 `json:"average_compression_ratio"`
	
	// TotalCompressionTime total time spent on compression
	TotalCompressionTimeMs int64 `json:"total_compression_time_ms"`
	
	// LastUpdated timestamp of last update
	LastUpdated time.Time `json:"last_updated"`
}

// CompressionResult contains the result of a compression operation
type CompressionResult struct {
	// Success whether compression was successful
	Success bool `json:"success"`
	
	// CompressedData the compressed data (could be gradient, tensor, or sparse)
	CompressedData interface{} `json:"compressed_data"`
	
	// CompressionRatio achieved compression ratio
	CompressionRatio float64 `json:"compression_ratio"`
	
	// Method compression method used
	Method string `json:"method"`
	
	// ProcessingTimeMs time taken for compression
	ProcessingTimeMs int64 `json:"processing_time_ms"`
	
	// ErrorMessage error message if compression failed
	ErrorMessage string `json:"error_message,omitempty"`
	
	// Metadata additional metadata about compression
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// NewCompressionPipeline creates a new compression pipeline
func NewCompressionPipeline(config CompressionPipelineConfig) *CompressionPipeline {
	pipeline := &CompressionPipeline{
		config: config,
		stats:  &PipelineStatistics{},
	}
	
	// Initialize compressors based on configuration
	if config.EnableGradientCompression {
		pipeline.gradientCompressor = NewGradientCompressor(config.GradientConfig)
	}
	
	if config.EnableTensorCompression {
		pipeline.tensorCompressor = NewTensorCompressor(config.GradientConfig) // Reuse config structure
	}
	
	if config.EnableSparsification {
		pipeline.sparsifier = NewSparsifier(config.SparsificationConfig)
	}
	
	if config.EnableStorageCompression {
		pipeline.storageCompressor = compression.NewCompressor(config.StorageConfig)
	}
	
	return pipeline
}

// UpdateConfig updates the pipeline configuration
func (cp *CompressionPipeline) UpdateConfig(config CompressionPipelineConfig) {
	cp.mutex.Lock()
	defer cp.mutex.Unlock()
	
	cp.config = config
	
	// Update component configurations
	if cp.gradientCompressor != nil && config.EnableGradientCompression {
		cp.gradientCompressor.UpdateConfig(config.GradientConfig)
	}
	
	if cp.tensorCompressor != nil && config.EnableTensorCompression {
		cp.tensorCompressor.UpdateConfig(config.GradientConfig)
	}
	
	if cp.sparsifier != nil && config.EnableSparsification {
		cp.sparsifier.UpdateConfig(config.SparsificationConfig)
	}
}

// CompressGradient compresses a gradient tensor through the pipeline
func (cp *CompressionPipeline) CompressGradient(ctx context.Context, gradient *GradientTensor) (*CompressionResult, error) {
	if gradient == nil {
		return nil, errors.New("gradient tensor is nil")
	}
	
	startTime := time.Now()
	result := &CompressionResult{
		Method:   "gradient_compression",
		Metadata: make(map[string]interface{}),
	}
	
	// Check size threshold
	gradientSize := int64(len(gradient.Data) * 4) // 4 bytes per float32
	if gradientSize < cp.config.CompressionThresholds.MinGradientSize {
		result.Success = false
		result.ErrorMessage = "gradient size below threshold"
		return result, nil
	}
	
	// Apply gradient compression
	var compressed interface{}
	var compressionRatio float64
	var err error
	
	if cp.config.EnableSparsification && cp.sparsifier != nil {
		// Apply sparsification first
		sparse, metrics, sparsErr := cp.sparsifier.SparsifyTensor(gradient, gradient.LayerName)
		if sparsErr == nil && sparse.SparsityLevel > 0.1 {
			// If significant sparsity achieved, use sparse representation
			compressed = sparse
			compressionRatio = metrics.CompressionRatio
			result.Metadata["sparsification_applied"] = true
			result.Metadata["sparsity_level"] = sparse.SparsityLevel
		}
	}
	
	// If not sparsified or sparsification not beneficial, use gradient compression
	if compressed == nil && cp.config.EnableGradientCompression && cp.gradientCompressor != nil {
		compressedGrad, gradErr := cp.gradientCompressor.CompressGradient(gradient)
		if gradErr == nil {
			compressed = compressedGrad
			compressionRatio = compressedGrad.CompressionRatio
			result.Metadata["gradient_compression_applied"] = true
			result.Metadata["algorithm"] = compressedGrad.Algorithm
		} else {
			err = gradErr
		}
	}
	
	// Apply storage-level compression if configured
	if compressed != nil && cp.config.EnableStorageCompression && cp.storageCompressor != nil {
		// Serialize compressed data
		serialized, serErr := json.Marshal(compressed)
		if serErr == nil {
			storageCompressed, storageErr := cp.storageCompressor.Compress(serialized)
			if storageErr == nil {
				// Update compression ratio
				originalSerializedSize := len(serialized)
				finalSize := len(storageCompressed)
				storageRatio := float64(finalSize) / float64(originalSerializedSize)
				compressionRatio *= storageRatio
				
				result.Metadata["storage_compression_applied"] = true
				result.Metadata["storage_compression_ratio"] = storageRatio
			}
		}
	}
	
	// Finalize result
	if compressed != nil {
		result.Success = true
		result.CompressedData = compressed
		result.CompressionRatio = compressionRatio
	} else {
		result.Success = false
		if err != nil {
			result.ErrorMessage = err.Error()
		} else {
			result.ErrorMessage = "no compression method applied"
		}
	}
	
	result.ProcessingTimeMs = time.Since(startTime).Milliseconds()
	
	// Update statistics
	cp.updateStatistics(result, gradientSize)
	
	return result, nil
}

// CompressTensor compresses a model tensor through the pipeline
func (cp *CompressionPipeline) CompressTensor(ctx context.Context, tensor *ModelTensor) (*CompressionResult, error) {
	if tensor == nil {
		return nil, errors.New("tensor is nil")
	}
	
	startTime := time.Now()
	result := &CompressionResult{
		Method:   "tensor_compression",
		Metadata: make(map[string]interface{}),
	}
	
	// Check size threshold
	tensorSize := int64(len(tensor.Data) * 4)
	if tensorSize < cp.config.CompressionThresholds.MinTensorSize {
		result.Success = false
		result.ErrorMessage = "tensor size below threshold"
		return result, nil
	}
	
	// Select compression algorithm based on tensor type and configuration
	algorithm := cp.selectTensorCompressionAlgorithm(tensor)
	
	var compressed interface{}
	var compressionRatio float64
	var err error
	
	// Apply tensor compression
	if cp.config.EnableTensorCompression && cp.tensorCompressor != nil {
		compressedTensor, tensorErr := cp.tensorCompressor.CompressTensor(tensor, algorithm)
		if tensorErr == nil {
			compressed = compressedTensor
			compressionRatio = compressedTensor.CompressionRatio
			result.Metadata["tensor_compression_applied"] = true
			result.Metadata["algorithm"] = compressedTensor.Algorithm
		} else {
			err = tensorErr
		}
	}
	
	// Apply storage-level compression
	if compressed != nil && cp.config.EnableStorageCompression && cp.storageCompressor != nil {
		serialized, serErr := json.Marshal(compressed)
		if serErr == nil {
			storageCompressed, storageErr := cp.storageCompressor.Compress(serialized)
			if storageErr == nil {
				storageRatio := float64(len(storageCompressed)) / float64(len(serialized))
				compressionRatio *= storageRatio
				result.Metadata["storage_compression_applied"] = true
			}
		}
	}
	
	// Finalize result
	if compressed != nil {
		result.Success = true
		result.CompressedData = compressed
		result.CompressionRatio = compressionRatio
	} else {
		result.Success = false
		if err != nil {
			result.ErrorMessage = err.Error()
		} else {
			result.ErrorMessage = "no compression method applied"
		}
	}
	
	result.ProcessingTimeMs = time.Since(startTime).Milliseconds()
	
	// Update statistics
	cp.updateStatistics(result, tensorSize)
	
	return result, nil
}

// CompressBatch compresses a batch of tensors in parallel
func (cp *CompressionPipeline) CompressBatch(ctx context.Context, gradients []*GradientTensor, tensors []*ModelTensor) ([]*CompressionResult, error) {
	totalItems := len(gradients) + len(tensors)
	if totalItems == 0 {
		return nil, errors.New("no items to compress")
	}
	
	results := make([]*CompressionResult, totalItems)
	errChan := make(chan error, totalItems)
	
	// Determine concurrency
	maxConcurrency := cp.config.MaxConcurrency
	if maxConcurrency <= 0 {
		maxConcurrency = 4 // Default
	}
	
	// Use semaphore to limit concurrency
	semaphore := make(chan struct{}, maxConcurrency)
	var wg sync.WaitGroup
	
	// Compress gradients
	for i, grad := range gradients {
		if grad == nil {
			continue
		}
		
		wg.Add(1)
		go func(idx int, gradient *GradientTensor) {
			defer wg.Done()
			semaphore <- struct{}{} // Acquire
			defer func() { <-semaphore }() // Release
			
			result, err := cp.CompressGradient(ctx, gradient)
			if err != nil {
				errChan <- err
				return
			}
			results[idx] = result
		}(i, grad)
	}
	
	// Compress tensors
	gradientCount := len(gradients)
	for i, tensor := range tensors {
		if tensor == nil {
			continue
		}
		
		wg.Add(1)
		go func(idx int, modelTensor *ModelTensor) {
			defer wg.Done()
			semaphore <- struct{}{} // Acquire
			defer func() { <-semaphore }() // Release
			
			result, err := cp.CompressTensor(ctx, modelTensor)
			if err != nil {
				errChan <- err
				return
			}
			results[gradientCount+idx] = result
		}(i, tensor)
	}
	
	// Wait for all operations to complete
	wg.Wait()
	close(errChan)
	
	// Check for errors
	var errors []error
	for err := range errChan {
		errors = append(errors, err)
	}
	
	if len(errors) > 0 {
		return results, fmt.Errorf("compression errors occurred: %v", errors)
	}
	
	return results, nil
}

// selectTensorCompressionAlgorithm selects the best compression algorithm for a tensor
func (cp *CompressionPipeline) selectTensorCompressionAlgorithm(tensor *ModelTensor) TensorCompressionAlgorithm {
	// Default algorithm from configuration
	algorithm := cp.config.TensorConfig.Algorithm
	
	// Adaptive selection based on tensor characteristics
	if cp.config.QualitySettings.AdaptiveCompression {
		switch tensor.Type {
		case TensorTypeWeight:
			// Weights benefit from pruning or quantization
			if len(tensor.Shape) == 2 {
				algorithm = TensorCompressionPruning // FC layers
			} else if len(tensor.Shape) == 4 {
				algorithm = TensorCompressionQuantization // Conv layers
			}
		case TensorTypeBias:
			// Biases are small, use quantization
			algorithm = TensorCompressionQuantization
		case TensorTypeEmbedding:
			// Embeddings benefit from SVD compression
			algorithm = TensorCompressionSVD
		case TensorTypeAttentionWeight:
			// Attention weights benefit from low-rank approximation
			algorithm = TensorCompressionLowRank
		default:
			// Use configured default
		}
		
		// Consider quality settings
		if cp.config.QualitySettings.QualityLevel > 0.8 {
			// High quality - use gentler compression
			if algorithm == TensorCompressionPruning {
				algorithm = TensorCompressionQuantization
			}
		} else if cp.config.QualitySettings.QualityLevel < 0.5 {
			// Low quality - use aggressive compression
			if algorithm == TensorCompressionQuantization {
				algorithm = TensorCompressionPruning
			}
		}
	}
	
	return algorithm
}

// updateStatistics updates pipeline statistics
func (cp *CompressionPipeline) updateStatistics(result *CompressionResult, originalSize int64) {
	cp.statsMux.Lock()
	defer cp.statsMux.Unlock()
	
	cp.stats.TotalOperations++
	cp.stats.TotalBytesProcessed += originalSize
	
	if result.Success {
		compressedSize := int64(float64(originalSize) * result.CompressionRatio)
		cp.stats.TotalBytesCompressed += compressedSize
		
		// Update average compression ratio
		totalCompressedData := cp.stats.TotalBytesCompressed
		totalOriginalData := cp.stats.TotalBytesProcessed
		if totalOriginalData > 0 {
			cp.stats.AverageCompressionRatio = float64(totalCompressedData) / float64(totalOriginalData)
		}
		
		// Update method-specific counters
		switch result.Method {
		case "gradient_compression":
			cp.stats.GradientOperations++
		case "tensor_compression":
			cp.stats.TensorOperations++
		}
		
		if result.Metadata["sparsification_applied"] == true {
			cp.stats.SparsificationOperations++
		}
	}
	
	cp.stats.TotalCompressionTimeMs += result.ProcessingTimeMs
	cp.stats.LastUpdated = time.Now()
}

// GetStatistics returns current pipeline statistics
func (cp *CompressionPipeline) GetStatistics() *PipelineStatistics {
	cp.statsMux.Lock()
	defer cp.statsMux.Unlock()
	
	// Return a copy to prevent data races
	stats := *cp.stats
	return &stats
}

// EstimateCompressionBenefit estimates the benefit of compressing given data
func (cp *CompressionPipeline) EstimateCompressionBenefit(gradientSizes []int64, tensorSizes []int64) map[string]interface{} {
	var totalOriginalSize int64
	var estimatedCompressedSize int64
	
	// Estimate gradient compression
	for _, size := range gradientSizes {
		totalOriginalSize += size
		if size >= cp.config.CompressionThresholds.MinGradientSize {
			// Use configured compression ratio
			ratio := cp.config.GradientConfig.CompressionRatio
			if cp.config.EnableSparsification {
				// Sparsification can achieve better ratios
				ratio *= 0.8
			}
			estimatedCompressedSize += int64(float64(size) * ratio)
		} else {
			estimatedCompressedSize += size // No compression
		}
	}
	
	// Estimate tensor compression
	for _, size := range tensorSizes {
		totalOriginalSize += size
		if size >= cp.config.CompressionThresholds.MinTensorSize {
			ratio := cp.config.TensorConfig.CompressionRatio
			estimatedCompressedSize += int64(float64(size) * ratio)
		} else {
			estimatedCompressedSize += size // No compression
		}
	}
	
	// Calculate benefits
	var compressionRatio, memorySavings, estimatedTimeMs float64
	
	if totalOriginalSize > 0 {
		compressionRatio = float64(estimatedCompressedSize) / float64(totalOriginalSize)
		memorySavings = float64(totalOriginalSize-estimatedCompressedSize) / (1024 * 1024) // MB
		
		// Estimate processing time (rough approximation: 1MB per 10ms)
		estimatedTimeMs = float64(totalOriginalSize) / (1024 * 1024) * 10
	}
	
	return map[string]interface{}{
		"total_original_size_mb":   float64(totalOriginalSize) / (1024 * 1024),
		"estimated_compressed_size_mb": float64(estimatedCompressedSize) / (1024 * 1024),
		"compression_ratio":        compressionRatio,
		"memory_savings_mb":        memorySavings,
		"estimated_processing_ms":  estimatedTimeMs,
		"gradient_items":          len(gradientSizes),
		"tensor_items":            len(tensorSizes),
	}
}

// DefaultCompressionPipelineConfig returns a default pipeline configuration
func DefaultCompressionPipelineConfig() CompressionPipelineConfig {
	return CompressionPipelineConfig{
		EnableGradientCompression: true,
		EnableTensorCompression:   true,
		EnableSparsification:      true,
		EnableStorageCompression:  false, // Often not needed with ML-specific compression
		
		GradientConfig:       DefaultGradientCompressionConfig(),
		SparsificationConfig: DefaultSparsificationConfig(),
		
		TensorConfig: TensorCompressionConfig{
			Algorithm:        TensorCompressionQuantization,
			CompressionRatio: 0.25, // 75% compression
			QuantizationBits: Bits8,
		},
		
		StorageConfig: compression.DefaultCompressionConfig(),
		
		CompressionThresholds: CompressionThresholds{
			MinGradientSize:       1024,      // 1KB
			MinTensorSize:         4096,      // 4KB
			MinSparsificationSize: 1024,      // 1KB
			MinStorageSize:        16384,     // 16KB
		},
		
		QualitySettings: QualitySettings{
			QualityLevel:        0.8,   // High quality
			PreserveAccuracy:    true,
			MaxAccuracyLoss:     0.05,  // 5% max accuracy loss
			AdaptiveCompression: true,
			ErrorToleranceMode:  ErrorToleranceBalanced,
		},
		
		ParallelProcessing: true,
		MaxConcurrency:     4,
	}
}

// OptimizeConfiguration optimizes the compression configuration for given workload characteristics
func (cp *CompressionPipeline) OptimizeConfiguration(workloadProfile WorkloadProfile) CompressionPipelineConfig {
	config := cp.config
	
	// Adjust based on workload characteristics
	switch workloadProfile.ModelType {
	case "transformer":
		// Transformers benefit from attention-specific optimizations
		config.TensorConfig.Algorithm = TensorCompressionLowRank
		config.SparsificationConfig.Algorithm = SparsificationStructured
		config.GradientConfig.Algorithm = CompressionTopK
		
	case "cnn":
		// CNNs benefit from channel-wise sparsification
		config.TensorConfig.Algorithm = TensorCompressionQuantization
		config.SparsificationConfig.Granularity = GranularityChannel
		config.GradientConfig.Algorithm = CompressionThreshold
		
	case "mlp":
		// MLPs can handle aggressive compression
		config.TensorConfig.Algorithm = TensorCompressionPruning
		config.SparsificationConfig.Algorithm = SparsificationMagnitude
		config.GradientConfig.CompressionRatio = 0.05 // Very aggressive
		
	default:
		// Use balanced settings
	}
	
	// Adjust for memory constraints
	if workloadProfile.MemoryConstrained {
		config.GradientConfig.CompressionRatio *= 0.5 // More aggressive
		config.TensorConfig.CompressionRatio *= 0.5
		config.SparsificationConfig.SparsityRatio = math.Min(0.95, config.SparsificationConfig.SparsityRatio*1.2)
	}
	
	// Adjust for speed requirements
	if workloadProfile.SpeedCritical {
		config.GradientConfig.Algorithm = CompressionTopK // Fast
		config.TensorConfig.Algorithm = TensorCompressionQuantization // Fast
		config.ParallelProcessing = true
		config.MaxConcurrency = 8 // Higher concurrency
	}
	
	return config
}

// WorkloadProfile describes characteristics of the ML workload
type WorkloadProfile struct {
	ModelType         string  `json:"model_type"`         // transformer, cnn, mlp, etc.
	ModelSize         int64   `json:"model_size"`         // Model size in parameters
	BatchSize         int     `json:"batch_size"`         // Training batch size
	MemoryConstrained bool    `json:"memory_constrained"` // Whether memory is limited
	SpeedCritical     bool    `json:"speed_critical"`     // Whether speed is critical
	AccuracyTarget    float64 `json:"accuracy_target"`    // Target accuracy to maintain
}

// GetCompressionRecommendations returns recommendations for optimal compression settings
func (cp *CompressionPipeline) GetCompressionRecommendations(profile WorkloadProfile) map[string]interface{} {
	recommendations := make(map[string]interface{})
	
	// Model-specific recommendations
	switch profile.ModelType {
	case "transformer":
		recommendations["gradient_algorithm"] = "topk"
		recommendations["tensor_algorithm"] = "lowrank"
		recommendations["sparsification_granularity"] = "structured"
		recommendations["explanation"] = "Transformers benefit from structured sparsification and low-rank approximation"
		
	case "cnn":
		recommendations["gradient_algorithm"] = "threshold"
		recommendations["tensor_algorithm"] = "quantization"
		recommendations["sparsification_granularity"] = "channel"
		recommendations["explanation"] = "CNNs work well with channel pruning and quantization"
		
	case "mlp":
		recommendations["gradient_algorithm"] = "topk"
		recommendations["tensor_algorithm"] = "pruning"
		recommendations["sparsification_granularity"] = "element"
		recommendations["explanation"] = "MLPs can handle aggressive element-wise pruning"
	}
	
	// Size-based recommendations
	if profile.ModelSize > 1e9 { // > 1B parameters
		recommendations["compression_aggressiveness"] = "high"
		recommendations["enable_sparsification"] = true
		recommendations["sparsity_target"] = 0.9
	} else if profile.ModelSize > 1e6 { // > 1M parameters
		recommendations["compression_aggressiveness"] = "medium"
		recommendations["sparsity_target"] = 0.7
	} else {
		recommendations["compression_aggressiveness"] = "low"
		recommendations["sparsity_target"] = 0.5
	}
	
	// Memory constraint recommendations
	if profile.MemoryConstrained {
		recommendations["priority"] = "memory_efficiency"
		recommendations["enable_all_methods"] = true
		recommendations["quality_trade_off"] = "favor_compression"
	} else {
		recommendations["priority"] = "quality_preservation"
		recommendations["quality_trade_off"] = "favor_accuracy"
	}
	
	// Speed requirement recommendations
	if profile.SpeedCritical {
		recommendations["parallel_processing"] = true
		recommendations["max_concurrency"] = 8
		recommendations["prefer_fast_algorithms"] = true
	}
	
	return recommendations
}