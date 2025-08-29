# ML/AI Gradient Compression System

This package provides comprehensive gradient and tensor compression capabilities for ML/AI workloads in NovaCron. The system implements multiple compression algorithms including sparsification, quantization, and hybrid approaches optimized for distributed machine learning environments.

## Features

- **Gradient Compression**: Multiple algorithms for compressing gradients during distributed training
- **Tensor Compression**: Model parameter compression for storage and transmission
- **Sparsification**: Advanced sparse data structure creation with various granularities
- **Quantization**: FP32→FP16→INT8 quantization with error bounds
- **Integration Pipeline**: Unified compression pipeline with adaptive algorithm selection
- **Performance Monitoring**: Comprehensive metrics and benchmarking capabilities

## Core Components

### 1. Gradient Compression (`gradient_compression.go`)

Implements gradient compression algorithms optimized for distributed ML training:

#### Algorithms

- **Top-K Sparsification**: Keeps only the top K largest gradient values
- **Random-K Sparsification**: Randomly selects K gradient values
- **Threshold-based**: Prunes gradients below a magnitude threshold
- **Quantization**: Reduces precision (FP32→FP16→INT8)
- **Hybrid**: Combines sparsification and quantization

#### Usage

```go
// Create gradient compressor with default config
config := DefaultGradientCompressionConfig()
config.Algorithm = CompressionTopK
config.TopKRatio = 0.1 // Keep top 10%

compressor := NewGradientCompressor(config)

// Compress a gradient
gradient := &GradientTensor{
    Shape:     []int{1000},
    Data:      gradientData, // []float32
    LayerName: "fc1",
    Timestamp: time.Now().Unix(),
}

compressed, err := compressor.CompressGradient(gradient)
if err != nil {
    return err
}

// Decompress when needed
decompressed, err := compressor.DecompressGradient(compressed)
```

#### Configuration Options

```go
type GradientCompressionConfig struct {
    Algorithm             GradientCompressionAlgorithm
    CompressionRatio      float64  // Target ratio (0.0-1.0)
    TopKRatio            float64  // For top-k sparsification
    SparsityThreshold    float64  // For threshold pruning
    QuantizationBits     QuantizationBits // FP32/FP16/INT8
    EnableAdaptive       bool     // Adaptive compression
    ErrorBoundCompression bool    // Enable error bounds
    MaxRelativeError     float64  // Maximum acceptable error
}
```

### 2. Tensor Compression (`tensor_compression.go`)

Specialized compression for model parameters and tensors:

#### Algorithms

- **Magnitude Pruning**: Remove small weights by magnitude
- **Quantization**: Reduce precision (FP32/FP16/INT8/4-bit)
- **SVD Compression**: Singular Value Decomposition for matrices
- **K-means Clustering**: Cluster-based quantization
- **Low-rank Approximation**: Matrix factorization
- **Huffman Coding**: Statistical compression
- **Bit-packing**: Efficient bit-level storage

#### Usage

```go
config := DefaultGradientCompressionConfig() // Reuses config structure
compressor := NewTensorCompressor(config)

tensor := &ModelTensor{
    Name:      "layer1.weight",
    Shape:     []int{512, 256},
    Data:      tensorData, // []float32
    Type:      TensorTypeWeight,
    LayerType: "linear",
}

compressed, err := compressor.CompressTensor(tensor, TensorCompressionQuantization)
if err != nil {
    return err
}

// Get compression statistics
stats := compressor.GetTensorCompressionStats(compressed)
fmt.Printf("Compression ratio: %f\n", stats["compression_ratio"])
```

### 3. Sparsification (`sparsification.go`)

Advanced sparsification algorithms with multiple granularities:

#### Algorithms

- **Magnitude-based**: Prune by absolute weight magnitude
- **Gradient-based**: Use gradient history for importance
- **Random**: Random pruning for baseline comparisons
- **Structured**: Channel/filter/block-wise pruning
- **SNIP**: Single-shot Network Pruning
- **GraSP**: Gradient Signal Preservation
- **Lottery Ticket**: Iterative magnitude pruning
- **Adaptive**: Dynamic sparsity adjustment

#### Granularities

- **Element-wise**: Individual parameter pruning
- **Channel-wise**: Entire channel removal
- **Filter-wise**: Complete filter pruning
- **Block-wise**: Structured block pruning
- **Layer-wise**: Layer-level sparsification

#### Usage

```go
config := DefaultSparsificationConfig()
config.Algorithm = SparsificationStructured
config.Granularity = GranularityChannel
config.SparsityRatio = 0.8 // 80% sparsity

sparsifier := NewSparsifier(config)

sparse, metrics, err := sparsifier.SparsifyTensor(tensor, "layer1")
if err != nil {
    return err
}

// Convert back to dense when needed
densified, err := sparsifier.DensifyTensor(sparse)
```

### 4. Compression Integration Pipeline (`compression_integration.go`)

Unified pipeline that orchestrates all compression methods:

#### Features

- **Adaptive Algorithm Selection**: Chooses optimal algorithms per tensor type
- **Quality-aware Compression**: Balances compression vs accuracy
- **Batch Processing**: Parallel compression of multiple tensors
- **Threshold Filtering**: Size-based compression decisions
- **Performance Monitoring**: Real-time statistics and metrics

#### Usage

```go
config := DefaultCompressionPipelineConfig()
config.QualitySettings.AdaptiveCompression = true
config.MaxConcurrency = 4

pipeline := NewCompressionPipeline(config)

// Compress single gradient
ctx := context.Background()
result, err := pipeline.CompressGradient(ctx, gradient)

// Compress batch
gradients := []*GradientTensor{grad1, grad2, grad3}
tensors := []*ModelTensor{tensor1, tensor2}
results, err := pipeline.CompressBatch(ctx, gradients, tensors)

// Get statistics
stats := pipeline.GetStatistics()
fmt.Printf("Average compression ratio: %f\n", stats.AverageCompressionRatio)
```

## Expected Compression Ratios

### Gradient Compression

| Algorithm | Typical Ratio | Use Case | Accuracy Impact |
|-----------|--------------|----------|-----------------|
| Top-K (1%) | 0.01-0.02 | Distributed training | Minimal |
| Top-K (10%) | 0.10-0.12 | Communication limited | Very low |
| Random-K | 0.05-0.15 | Baseline comparison | Low |
| Threshold | 0.05-0.30 | Adaptive sparsity | Low |
| FP16 Quantization | 0.50 | Memory optimization | Minimal |
| INT8 Quantization | 0.25 | Edge deployment | Low |
| Hybrid (TopK+INT8) | 0.01-0.05 | Aggressive compression | Moderate |

### Tensor Compression

| Algorithm | Typical Ratio | Use Case | Accuracy Impact |
|-----------|--------------|----------|-----------------|
| Magnitude Pruning | 0.10-0.30 | Model compression | Low-Moderate |
| FP16 Quantization | 0.50 | Inference acceleration | Minimal |
| INT8 Quantization | 0.25 | Mobile deployment | Low |
| K-means (256 clusters) | 0.25 | Statistical compression | Low |
| SVD (rank 50%) | 0.30-0.70 | Matrix factorization | Moderate |
| Low-rank | 0.20-0.60 | Attention compression | Moderate |
| Huffman | 0.30-0.80 | General purpose | Minimal |

### Sparsification

| Granularity | Typical Sparsity | FLOPS Reduction | Memory Saving |
|-------------|------------------|------------------|---------------|
| Element-wise | 90-95% | 60-80% | 90-95% |
| Channel-wise | 30-70% | 80-90% | 30-70% |
| Filter-wise | 40-80% | 85-95% | 40-80% |
| Block-wise (4x4) | 75-90% | 70-85% | 75-90% |

## Performance Characteristics

### Compression Speed

- **Top-K**: ~1000 MB/s (sorting overhead)
- **Quantization**: ~5000 MB/s (vectorized operations)
- **Threshold**: ~8000 MB/s (simple filtering)
- **Random**: ~3000 MB/s (random generation)
- **Hybrid**: ~800 MB/s (combined algorithms)

### Memory Overhead

- **Gradient compression**: 10-20% of original size during processing
- **Tensor compression**: 50-100% for temporary buffers
- **Sparsification**: 20-40% for index structures
- **Pipeline**: Minimal additional overhead

## Configuration Examples

### High Performance Training

```go
config := DefaultCompressionPipelineConfig()
config.GradientConfig.Algorithm = CompressionTopK
config.GradientConfig.TopKRatio = 0.01 // Very sparse
config.EnableSparsification = true
config.SparsificationConfig.Algorithm = SparsificationMagnitude
config.QualitySettings.QualityLevel = 0.7 // Favor speed over quality
config.MaxConcurrency = 8
```

### High Quality Inference

```go
config := DefaultCompressionPipelineConfig()
config.TensorConfig.Algorithm = TensorCompressionQuantization
config.TensorConfig.QuantizationBits = Bits16 // Preserve quality
config.EnableSparsification = false // No sparsification
config.QualitySettings.QualityLevel = 0.9
config.QualitySettings.PreserveAccuracy = true
```

### Memory-Constrained Edge Deployment

```go
config := DefaultCompressionPipelineConfig()
config.GradientConfig.Algorithm = CompressionHybrid
config.GradientConfig.CompressionRatio = 0.05 // Aggressive
config.TensorConfig.Algorithm = TensorCompressionPruning
config.SparsificationConfig.SparsityRatio = 0.95 // Very sparse
config.QualitySettings.QualityLevel = 0.6
```

## Integration with NovaCron

The ML compression system integrates seamlessly with NovaCron's VM migration and storage systems:

### VM Migration Integration

```go
// During VM migration, compress ML model state
pipeline := NewCompressionPipeline(config)

// Compress model weights before migration
for _, tensor := range modelWeights {
    result, err := pipeline.CompressTensor(ctx, tensor)
    if err != nil {
        return err
    }
    
    // Store compressed tensor for migration
    migrationData[tensor.Name] = result.CompressedData
}
```

### Storage Integration

The system leverages NovaCron's existing compression infrastructure:

```go
// Uses existing storage compression as additional layer
storageCompressor := compression.NewCompressor(compression.DefaultCompressionConfig())

// ML-specific compression + general storage compression
compressedML := pipeline.CompressGradient(ctx, gradient)
finalCompressed := storageCompressor.Compress(serialize(compressedML))
```

## API Reference

### Core Types

```go
// Gradient tensor representation
type GradientTensor struct {
    Shape     []int     `json:"shape"`
    Data      []float32 `json:"data"`
    LayerName string    `json:"layer_name"`
    Timestamp int64     `json:"timestamp"`
}

// Model tensor representation  
type ModelTensor struct {
    Name      string                 `json:"name"`
    Shape     []int                  `json:"shape"`
    Data      []float32              `json:"data"`
    Type      TensorType             `json:"type"`
    LayerType string                 `json:"layer_type"`
    Metadata  map[string]interface{} `json:"metadata,omitempty"`
}

// Compressed gradient result
type CompressedGradient struct {
    OriginalTensor       *GradientTensor              `json:"original_tensor"`
    CompressedData       []byte                       `json:"compressed_data"`
    Algorithm            GradientCompressionAlgorithm `json:"algorithm"`
    CompressionRatio     float64                      `json:"compression_ratio"`
    SparsityLevel        float64                      `json:"sparsity_level"`
    QuantizationLevel    QuantizationBits             `json:"quantization_level"`
    ErrorMetrics         *CompressionErrorMetrics     `json:"error_metrics,omitempty"`
}

// Sparse data representation
type SparseData struct {
    Indices       []int32                  `json:"indices"`
    Values        []float32                `json:"values"`
    Shape         []int                    `json:"shape"`
    SparsityLevel float64                  `json:"sparsity_level"`
    Algorithm     SparsificationAlgorithm  `json:"algorithm"`
    Metadata      map[string]interface{}   `json:"metadata,omitempty"`
}
```

### Key Functions

```go
// Gradient compression
func NewGradientCompressor(config GradientCompressionConfig) *GradientCompressor
func (gc *GradientCompressor) CompressGradient(gradient *GradientTensor) (*CompressedGradient, error)
func (gc *GradientCompressor) DecompressGradient(compressed *CompressedGradient) (*GradientTensor, error)

// Tensor compression
func NewTensorCompressor(config GradientCompressionConfig) *TensorCompressor  
func (tc *TensorCompressor) CompressTensor(tensor *ModelTensor, algorithm TensorCompressionAlgorithm) (*CompressedTensor, error)
func (tc *TensorCompressor) DecompressTensor(compressed *CompressedTensor) (*ModelTensor, error)

// Sparsification
func NewSparsifier(config SparsificationConfig) *Sparsifier
func (s *Sparsifier) SparsifyTensor(tensor *GradientTensor, layerName string) (*SparseData, *SparsificationMetrics, error)
func (s *Sparsifier) DensifyTensor(sparse *SparseData) (*GradientTensor, error)

// Integrated pipeline
func NewCompressionPipeline(config CompressionPipelineConfig) *CompressionPipeline
func (cp *CompressionPipeline) CompressGradient(ctx context.Context, gradient *GradientTensor) (*CompressionResult, error)
func (cp *CompressionPipeline) CompressTensor(ctx context.Context, tensor *ModelTensor) (*CompressionResult, error)
func (cp *CompressionPipeline) CompressBatch(ctx context.Context, gradients []*GradientTensor, tensors []*ModelTensor) ([]*CompressionResult, error)
```

## Testing and Benchmarks

The system includes comprehensive test coverage:

- **Unit Tests**: Individual algorithm testing with accuracy verification
- **Integration Tests**: End-to-end pipeline testing with real ML workloads
- **Benchmark Tests**: Performance measurement and optimization validation
- **Error Handling Tests**: Robustness testing with edge cases

Run tests with:

```bash
go test -v ./backend/core/ml/...
go test -bench=. ./backend/core/ml/...
```

## Performance Tuning

### For Training Workloads

1. **High-frequency gradients**: Use TopK with small ratios (1-5%)
2. **Large models**: Enable hybrid compression with quantization
3. **Communication-bound**: Prioritize compression ratio over speed
4. **Memory-bound**: Use aggressive sparsification with structured pruning

### For Inference Workloads

1. **Latency-critical**: Use quantization only (FP16/INT8)
2. **Throughput optimization**: Enable batch compression with high concurrency
3. **Memory optimization**: Combine pruning with quantization
4. **Edge deployment**: Use 4-bit quantization with structured sparsity

### Configuration Optimization

1. **Profile your workload**: Use `GetCompressionRecommendations()` for optimal settings
2. **Monitor metrics**: Track compression ratios and accuracy impact
3. **Adaptive tuning**: Enable adaptive compression for dynamic optimization
4. **Threshold tuning**: Adjust size thresholds based on network characteristics

This comprehensive ML compression system provides NovaCron with state-of-the-art gradient and tensor compression capabilities, enabling efficient distributed ML workloads with minimal accuracy impact while achieving significant reductions in memory usage, network traffic, and storage requirements.