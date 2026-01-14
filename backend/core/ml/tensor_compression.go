package ml

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"sort"
	"sync"

	"github.com/khryptorgraphics/novacron/backend/core/storage/compression"
)

// TensorCompressionAlgorithm defines compression algorithms for ML tensors
type TensorCompressionAlgorithm string

const (
	// TensorCompressionNone - no compression
	TensorCompressionNone TensorCompressionAlgorithm = "none"
	// TensorCompressionSVD - Singular Value Decomposition
	TensorCompressionSVD TensorCompressionAlgorithm = "svd"
	// TensorCompressionPruning - Weight pruning
	TensorCompressionPruning TensorCompressionAlgorithm = "pruning"
	// TensorCompressionQuantization - Parameter quantization
	TensorCompressionQuantization TensorCompressionAlgorithm = "quantization"
	// TensorCompressionHuffman - Huffman coding for weights
	TensorCompressionHuffman TensorCompressionAlgorithm = "huffman"
	// TensorCompressionKMeans - K-means clustering quantization
	TensorCompressionKMeans TensorCompressionAlgorithm = "kmeans"
	// TensorCompressionLowRank - Low-rank approximation
	TensorCompressionLowRank TensorCompressionAlgorithm = "lowrank"
	// TensorCompressionBitpacking - Bit-level packing
	TensorCompressionBitpacking TensorCompressionAlgorithm = "bitpacking"
)

// TensorType defines the type of tensor being compressed
type TensorType string

const (
	// TensorTypeWeight - Model weights/parameters
	TensorTypeWeight TensorType = "weight"
	// TensorTypeBias - Bias parameters
	TensorTypeBias TensorType = "bias"
	// TensorTypeActivation - Activation tensors
	TensorTypeActivation TensorType = "activation"
	// TensorTypeEmbedding - Embedding matrices
	TensorTypeEmbedding TensorType = "embedding"
	// TensorTypeConvKernel - Convolutional kernels
	TensorTypeConvKernel TensorType = "conv_kernel"
	// TensorTypeAttentionWeight - Attention weights
	TensorTypeAttentionWeight TensorType = "attention_weight"
)

// Note: TensorCompressionConfig is defined in compression_integration.go to avoid duplication

// ModelTensor represents a tensor from an ML model
type ModelTensor struct {
	// Name of the tensor (layer name + parameter type)
	Name string `json:"name"`
	
	// Shape of the tensor
	Shape []int `json:"shape"`
	
	// Data contains the tensor values
	Data []float32 `json:"data"`
	
	// Type of tensor (weight, bias, etc.)
	Type TensorType `json:"type"`
	
	// LayerType the layer this tensor belongs to (conv, linear, etc.)
	LayerType string `json:"layer_type"`
	
	// Metadata additional information about the tensor
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// CompressedTensor represents a compressed tensor
type CompressedTensor struct {
	// OriginalTensor reference to original tensor metadata
	OriginalTensor *ModelTensor `json:"original_tensor"`
	
	// CompressedData contains the compressed tensor data
	CompressedData []byte `json:"compressed_data"`
	
	// Algorithm used for compression
	Algorithm TensorCompressionAlgorithm `json:"algorithm"`
	
	// CompressionRatio achieved
	CompressionRatio float64 `json:"compression_ratio"`
	
	// CompressionMetrics contains detailed compression statistics
	CompressionMetrics *TensorCompressionMetrics `json:"compression_metrics"`
	
	// ReconstructionData additional data needed for reconstruction
	ReconstructionData map[string]interface{} `json:"reconstruction_data,omitempty"`
	
	// PrunedIndices for pruning-based compression
	PrunedIndices []int32 `json:"pruned_indices,omitempty"`
	
	// QuantizationClusters for k-means quantization
	QuantizationClusters []float32 `json:"quantization_clusters,omitempty"`
	
	// SVDComponents for SVD compression
	SVDComponents *SVDComponents `json:"svd_components,omitempty"`
}

// TensorCompressionMetrics tracks detailed compression statistics
type TensorCompressionMetrics struct {
	// OriginalSize in bytes
	OriginalSize int `json:"original_size"`
	
	// CompressedSize in bytes
	CompressedSize int `json:"compressed_size"`
	
	// CompressionRatio actual achieved ratio
	CompressionRatio float64 `json:"compression_ratio"`
	
	// ParameterReduction percentage of parameters reduced
	ParameterReduction float64 `json:"parameter_reduction"`
	
	// AccuracyLoss estimated accuracy loss from compression
	AccuracyLoss float64 `json:"accuracy_loss,omitempty"`
	
	// CompressionTime time taken to compress
	CompressionTimeMs int64 `json:"compression_time_ms"`
	
	// DecompressionTime time taken to decompress
	DecompressionTimeMs int64 `json:"decompression_time_ms"`
	
	// ErrorMetrics compression error statistics
	ErrorMetrics *CompressionErrorMetrics `json:"error_metrics,omitempty"`
}

// SVDComponents stores SVD decomposition components
type SVDComponents struct {
	// U matrix (left singular vectors)
	U []float32 `json:"u"`
	// S vector (singular values)
	S []float32 `json:"s"`
	// VT matrix (right singular vectors transposed)
	VT []float32 `json:"vt"`
	// UShape shape of U matrix
	UShape []int `json:"u_shape"`
	// VTShape shape of VT matrix
	VTShape []int `json:"vt_shape"`
	// Rank preserved rank
	Rank int `json:"rank"`
}

// TensorCompressor handles compression and decompression of ML tensors
type TensorCompressor struct {
	config GradientCompressionConfig // Reuse gradient compression config for consistency
	mutex  sync.RWMutex
}

// NewTensorCompressor creates a new tensor compressor
func NewTensorCompressor(config GradientCompressionConfig) *TensorCompressor {
	return &TensorCompressor{
		config: config,
	}
}

// UpdateConfig updates the compression configuration
func (tc *TensorCompressor) UpdateConfig(config GradientCompressionConfig) {
	tc.mutex.Lock()
	defer tc.mutex.Unlock()
	tc.config = config
}

// CompressTensor compresses a model tensor using the configured algorithm
func (tc *TensorCompressor) CompressTensor(tensor *ModelTensor, algorithm TensorCompressionAlgorithm) (*CompressedTensor, error) {
	if tensor == nil || len(tensor.Data) == 0 {
		return nil, errors.New("invalid tensor")
	}
	
	tc.mutex.RLock()
	config := tc.config
	tc.mutex.RUnlock()
	
	compressed := &CompressedTensor{
		OriginalTensor: tensor,
		Algorithm:      algorithm,
	}
	
	var err error
	
	switch algorithm {
	case TensorCompressionNone:
		compressed, err = tc.compressTensorNone(tensor)
	case TensorCompressionPruning:
		compressed, err = tc.compressTensorPruning(tensor, config)
	case TensorCompressionQuantization:
		compressed, err = tc.compressTensorQuantization(tensor, config)
	case TensorCompressionSVD:
		compressed, err = tc.compressTensorSVD(tensor, config)
	case TensorCompressionKMeans:
		compressed, err = tc.compressTensorKMeans(tensor, config)
	case TensorCompressionLowRank:
		compressed, err = tc.compressTensorLowRank(tensor, config)
	case TensorCompressionHuffman:
		compressed, err = tc.compressTensorHuffman(tensor)
	case TensorCompressionBitpacking:
		compressed, err = tc.compressTensorBitpacking(tensor, config)
	default:
		return nil, fmt.Errorf("unsupported tensor compression algorithm: %s", algorithm)
	}
	
	if err != nil {
		return nil, fmt.Errorf("tensor compression failed: %w", err)
	}
	
	// Calculate compression metrics
	originalSize := len(tensor.Data) * 4
	compressedSize := len(compressed.CompressedData)
	
	compressed.CompressionMetrics = &TensorCompressionMetrics{
		OriginalSize:       originalSize,
		CompressedSize:     compressedSize,
		CompressionRatio:   float64(compressedSize) / float64(originalSize),
		ParameterReduction: 1.0 - (float64(compressedSize)/float64(originalSize)),
	}
	
	return compressed, nil
}

// DecompressTensor decompresses a compressed tensor
func (tc *TensorCompressor) DecompressTensor(compressed *CompressedTensor) (*ModelTensor, error) {
	if compressed == nil || compressed.OriginalTensor == nil {
		return nil, errors.New("invalid compressed tensor")
	}
	
	var err error
	var data []float32
	
	switch compressed.Algorithm {
	case TensorCompressionNone:
		data, err = tc.decompressTensorNone(compressed)
	case TensorCompressionPruning:
		data, err = tc.decompressTensorPruning(compressed)
	case TensorCompressionQuantization:
		data, err = tc.decompressTensorQuantization(compressed)
	case TensorCompressionSVD:
		data, err = tc.decompressTensorSVD(compressed)
	case TensorCompressionKMeans:
		data, err = tc.decompressTensorKMeans(compressed)
	case TensorCompressionLowRank:
		data, err = tc.decompressTensorLowRank(compressed)
	case TensorCompressionHuffman:
		data, err = tc.decompressTensorHuffman(compressed)
	case TensorCompressionBitpacking:
		data, err = tc.decompressTensorBitpacking(compressed)
	default:
		return nil, fmt.Errorf("unsupported tensor decompression algorithm: %s", compressed.Algorithm)
	}
	
	if err != nil {
		return nil, fmt.Errorf("tensor decompression failed: %w", err)
	}
	
	result := &ModelTensor{
		Name:      compressed.OriginalTensor.Name,
		Shape:     compressed.OriginalTensor.Shape,
		Data:      data,
		Type:      compressed.OriginalTensor.Type,
		LayerType: compressed.OriginalTensor.LayerType,
		Metadata:  compressed.OriginalTensor.Metadata,
	}
	
	return result, nil
}

// compressTensorNone performs no compression
func (tc *TensorCompressor) compressTensorNone(tensor *ModelTensor) (*CompressedTensor, error) {
	buf := new(bytes.Buffer)
	err := binary.Write(buf, binary.LittleEndian, tensor.Data)
	if err != nil {
		return nil, err
	}
	
	return &CompressedTensor{
		OriginalTensor:   tensor,
		CompressedData:   buf.Bytes(),
		Algorithm:        TensorCompressionNone,
		CompressionRatio: 1.0,
	}, nil
}

// compressTensorPruning performs magnitude-based pruning
func (tc *TensorCompressor) compressTensorPruning(tensor *ModelTensor, config GradientCompressionConfig) (*CompressedTensor, error) {
	data := tensor.Data
	threshold := config.SparsityThreshold
	
	// Calculate magnitude threshold if not provided
	if threshold <= 0 {
		// Use percentile-based pruning
		sortedMagnitudes := make([]float64, len(data))
		for i, val := range data {
			sortedMagnitudes[i] = math.Abs(float64(val))
		}
		sort.Float64s(sortedMagnitudes)
		
		pruningRatio := 1.0 - config.CompressionRatio
		percentileIndex := int(pruningRatio * float64(len(sortedMagnitudes)))
		if percentileIndex >= len(sortedMagnitudes) {
			percentileIndex = len(sortedMagnitudes) - 1
		}
		threshold = sortedMagnitudes[percentileIndex]
	}
	
	// Apply pruning
	var prunedData []float32
	var prunedIndices []int32
	
	for i, val := range data {
		if math.Abs(float64(val)) > threshold {
			prunedData = append(prunedData, val)
			prunedIndices = append(prunedIndices, int32(i))
		}
	}
	
	// Compress pruned data
	dataBytes := new(bytes.Buffer)
	indicesBytes := new(bytes.Buffer)
	
	binary.Write(dataBytes, binary.LittleEndian, prunedData)
	binary.Write(indicesBytes, binary.LittleEndian, prunedIndices)
	
	compressor := compression.NewCompressor(compression.DefaultCompressionConfig())
	compressedData, _ := compressor.Compress(dataBytes.Bytes())
	compressedIndices, _ := compressor.Compress(indicesBytes.Bytes())
	
	// Combine compressed data
	combinedData := append(compressedData, compressedIndices...)
	
	compressionRatio := float64(len(combinedData)) / float64(len(data)*4)
	
	return &CompressedTensor{
		OriginalTensor:   tensor,
		CompressedData:   combinedData,
		Algorithm:        TensorCompressionPruning,
		CompressionRatio: compressionRatio,
		PrunedIndices:    prunedIndices,
		ReconstructionData: map[string]interface{}{
			"pruned_values": prunedData,
			"threshold":     threshold,
		},
	}, nil
}

// compressTensorQuantization performs parameter quantization
func (tc *TensorCompressor) compressTensorQuantization(tensor *ModelTensor, config GradientCompressionConfig) (*CompressedTensor, error) {
	data := tensor.Data
	bits := config.QuantizationBits
	
	if bits == Bits32 {
		return tc.compressTensorNone(tensor)
	}
	
	// Find min/max for quantization range
	minVal := float64(data[0])
	maxVal := float64(data[0])
	
	for _, val := range data {
		v := float64(val)
		if v < minVal {
			minVal = v
		}
		if v > maxVal {
			maxVal = v
		}
	}
	
	// Quantize based on bits
	var quantizedData []byte
	var scale float64
	var zeroPoint int32
	
	switch bits {
	case Bits16:
		quantizedData, scale, zeroPoint = tc.quantizeToFP16Tensor(data, minVal, maxVal)
	case Bits8:
		quantizedData, scale, zeroPoint = tc.quantizeToINT8Tensor(data, minVal, maxVal)
	case Bits4:
		quantizedData, scale, zeroPoint = tc.quantizeTo4BitTensor(data, minVal, maxVal)
	default:
		return nil, fmt.Errorf("unsupported quantization bits: %d", bits)
	}
	
	compressionRatio := float64(len(quantizedData)) / float64(len(data)*4)
	
	return &CompressedTensor{
		OriginalTensor:   tensor,
		CompressedData:   quantizedData,
		Algorithm:        TensorCompressionQuantization,
		CompressionRatio: compressionRatio,
		ReconstructionData: map[string]interface{}{
			"scale":      scale,
			"zero_point": zeroPoint,
			"min_val":    minVal,
			"max_val":    maxVal,
			"bits":       bits,
		},
	}, nil
}

// compressTensorSVD performs SVD-based compression
func (tc *TensorCompressor) compressTensorSVD(tensor *ModelTensor, config GradientCompressionConfig) (*CompressedTensor, error) {
	if len(tensor.Shape) != 2 {
		return nil, errors.New("SVD compression requires 2D tensor (matrix)")
	}
	
	rows := tensor.Shape[0]
	cols := tensor.Shape[1]
	data := tensor.Data
	
	// Reshape data into matrix format
	matrix := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		matrix[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			matrix[i][j] = float64(data[i*cols+j])
		}
	}
	
	// Perform simplified SVD (for demonstration - in practice, use proper SVD library)
	rank := int(float64(min(rows, cols)) * config.CompressionRatio)
	if rank <= 0 {
		rank = 1
	}
	
	// For this implementation, we'll use a simplified approach
	// In practice, you'd use a proper SVD library like gonum
	u, s, vt := tc.simplifiedSVD(matrix, rank)
	
	// Serialize SVD components
	svdComponents := &SVDComponents{
		U:       tc.flattenMatrix(u),
		S:       tc.convertFloat64ToFloat32(s),
		VT:      tc.flattenMatrix(vt),
		UShape:  []int{len(u), len(u[0])},
		VTShape: []int{len(vt), len(vt[0])},
		Rank:    rank,
	}
	
	// Compress SVD components
	buf := new(bytes.Buffer)
	binary.Write(buf, binary.LittleEndian, svdComponents.U)
	binary.Write(buf, binary.LittleEndian, svdComponents.S)
	binary.Write(buf, binary.LittleEndian, svdComponents.VT)
	
	compressor := compression.NewCompressor(compression.DefaultCompressionConfig())
	compressedData, _ := compressor.Compress(buf.Bytes())
	
	compressionRatio := float64(len(compressedData)) / float64(len(data)*4)
	
	return &CompressedTensor{
		OriginalTensor:   tensor,
		CompressedData:   compressedData,
		Algorithm:        TensorCompressionSVD,
		CompressionRatio: compressionRatio,
		SVDComponents:    svdComponents,
	}, nil
}

// compressTensorKMeans performs k-means clustering quantization
func (tc *TensorCompressor) compressTensorKMeans(tensor *ModelTensor, config GradientCompressionConfig) (*CompressedTensor, error) {
	data := tensor.Data
	k := 256 // Default 256 clusters for 8-bit equivalent
	
	if config.QuantizationBits == Bits4 {
		k = 16
	} else if config.QuantizationBits == Bits16 {
		k = 65536
	}
	
	// Simple k-means implementation (for demonstration)
	clusters, assignments := tc.simpleKMeans(data, k)
	
	// Encode assignments
	var assignmentBytes []byte
	if k <= 256 {
		// Use 8-bit assignments
		assignmentBytes = make([]byte, len(data))
		for i, assignment := range assignments {
			assignmentBytes[i] = byte(assignment)
		}
	} else {
		// Use 16-bit assignments
		buf := new(bytes.Buffer)
		for _, assignment := range assignments {
			binary.Write(buf, binary.LittleEndian, uint16(assignment))
		}
		assignmentBytes = buf.Bytes()
	}
	
	// Combine clusters and assignments
	clustersBytes := new(bytes.Buffer)
	binary.Write(clustersBytes, binary.LittleEndian, clusters)
	
	compressor := compression.NewCompressor(compression.DefaultCompressionConfig())
	compressedClusters, _ := compressor.Compress(clustersBytes.Bytes())
	compressedAssignments, _ := compressor.Compress(assignmentBytes)
	
	combinedData := append(compressedClusters, compressedAssignments...)
	compressionRatio := float64(len(combinedData)) / float64(len(data)*4)
	
	return &CompressedTensor{
		OriginalTensor:       tensor,
		CompressedData:       combinedData,
		Algorithm:            TensorCompressionKMeans,
		CompressionRatio:     compressionRatio,
		QuantizationClusters: clusters,
		ReconstructionData: map[string]interface{}{
			"assignments":  assignments,
			"num_clusters": k,
		},
	}, nil
}

// compressTensorLowRank performs low-rank approximation
func (tc *TensorCompressor) compressTensorLowRank(tensor *ModelTensor, config GradientCompressionConfig) (*CompressedTensor, error) {
	// Similar to SVD but with different rank selection strategy
	return tc.compressTensorSVD(tensor, config)
}

// compressTensorHuffman performs Huffman coding compression
func (tc *TensorCompressor) compressTensorHuffman(tensor *ModelTensor) (*CompressedTensor, error) {
	// For demonstration, use existing compression infrastructure
	buf := new(bytes.Buffer)
	binary.Write(buf, binary.LittleEndian, tensor.Data)
	
	// Use maximum compression settings
	compressorConfig := compression.DefaultCompressionConfig()
	compressorConfig.Level = compression.CompressionBest
	compressor := compression.NewCompressor(compressorConfig)
	
	compressedData, err := compressor.Compress(buf.Bytes())
	if err != nil {
		return nil, err
	}
	
	compressionRatio := float64(len(compressedData)) / float64(len(tensor.Data)*4)
	
	return &CompressedTensor{
		OriginalTensor:   tensor,
		CompressedData:   compressedData,
		Algorithm:        TensorCompressionHuffman,
		CompressionRatio: compressionRatio,
	}, nil
}

// compressTensorBitpacking performs bit-level packing
func (tc *TensorCompressor) compressTensorBitpacking(tensor *ModelTensor, config GradientCompressionConfig) (*CompressedTensor, error) {
	// Implement bit packing for reduced precision
	return tc.compressTensorQuantization(tensor, config)
}

// Decompression methods for tensors

func (tc *TensorCompressor) decompressTensorNone(compressed *CompressedTensor) ([]float32, error) {
	buf := bytes.NewReader(compressed.CompressedData)
	totalElements := len(compressed.OriginalTensor.Data)
	data := make([]float32, totalElements)
	
	err := binary.Read(buf, binary.LittleEndian, data)
	return data, err
}

func (tc *TensorCompressor) decompressTensorPruning(compressed *CompressedTensor) ([]float32, error) {
	totalElements := len(compressed.OriginalTensor.Data)
	result := make([]float32, totalElements)
	
	// Get pruned values from reconstruction data
	prunedValues, ok := compressed.ReconstructionData["pruned_values"].([]float32)
	if !ok {
		return nil, errors.New("missing pruned values in reconstruction data")
	}
	
	// Reconstruct from indices and values
	for i, idx := range compressed.PrunedIndices {
		if int(idx) < totalElements && i < len(prunedValues) {
			result[idx] = prunedValues[i]
		}
	}
	
	return result, nil
}

func (tc *TensorCompressor) decompressTensorQuantization(compressed *CompressedTensor) ([]float32, error) {
	reconstructionData := compressed.ReconstructionData
	bits := reconstructionData["bits"].(QuantizationBits)
	scale := reconstructionData["scale"].(float64)
	zeroPoint := reconstructionData["zero_point"].(int32)
	totalElements := len(compressed.OriginalTensor.Data)
	
	switch bits {
	case Bits16:
		return tc.dequantizeFromFP16Tensor(compressed.CompressedData, totalElements)
	case Bits8:
		return tc.dequantizeFromINT8Tensor(compressed.CompressedData, scale, zeroPoint, totalElements)
	case Bits4:
		return tc.dequantizeFrom4BitTensor(compressed.CompressedData, scale, zeroPoint, totalElements)
	default:
		return nil, fmt.Errorf("unsupported quantization bits: %d", bits)
	}
}

func (tc *TensorCompressor) decompressTensorSVD(compressed *CompressedTensor) ([]float32, error) {
	svd := compressed.SVDComponents
	if svd == nil {
		return nil, errors.New("missing SVD components")
	}
	
	// Reconstruct matrix from SVD components: A = U * S * VT
	rows := compressed.OriginalTensor.Shape[0]
	cols := compressed.OriginalTensor.Shape[1]
	
	// Convert back to matrices
	u := tc.reshapeMatrix(svd.U, svd.UShape[0], svd.UShape[1])
	vt := tc.reshapeMatrix(svd.VT, svd.VTShape[0], svd.VTShape[1])
	
	// Multiply U * S * VT
	result := make([]float32, rows*cols)
	
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			var sum float64
			for k := 0; k < svd.Rank; k++ {
				sum += float64(u[i][k]) * float64(svd.S[k]) * float64(vt[k][j])
			}
			result[i*cols+j] = float32(sum)
		}
	}
	
	return result, nil
}

func (tc *TensorCompressor) decompressTensorKMeans(compressed *CompressedTensor) ([]float32, error) {
	clusters := compressed.QuantizationClusters
	assignments, ok := compressed.ReconstructionData["assignments"].([]int)
	if !ok {
		return nil, errors.New("missing assignments in reconstruction data")
	}
	
	// Reconstruct data from cluster assignments
	result := make([]float32, len(assignments))
	for i, clusterIdx := range assignments {
		if clusterIdx >= 0 && clusterIdx < len(clusters) {
			result[i] = clusters[clusterIdx]
		}
	}
	
	return result, nil
}

func (tc *TensorCompressor) decompressTensorLowRank(compressed *CompressedTensor) ([]float32, error) {
	return tc.decompressTensorSVD(compressed)
}

func (tc *TensorCompressor) decompressTensorHuffman(compressed *CompressedTensor) ([]float32, error) {
	compressor := compression.NewCompressor(compression.DefaultCompressionConfig())
	decompressedData, err := compressor.Decompress(compressed.CompressedData, compression.CompressionGzip)
	if err != nil {
		return nil, err
	}
	
	buf := bytes.NewReader(decompressedData)
	totalElements := len(compressed.OriginalTensor.Data)
	data := make([]float32, totalElements)
	
	err = binary.Read(buf, binary.LittleEndian, data)
	return data, err
}

func (tc *TensorCompressor) decompressTensorBitpacking(compressed *CompressedTensor) ([]float32, error) {
	return tc.decompressTensorQuantization(compressed)
}

// Helper methods for tensor compression

func (tc *TensorCompressor) quantizeToFP16Tensor(data []float32, minVal, maxVal float64) ([]byte, float64, int32) {
	buf := new(bytes.Buffer)
	scale := 1.0
	zeroPoint := int32(0)
	
	for _, val := range data {
		bits := math.Float32bits(val)
		fp16Bits := uint16(bits >> 16)
		binary.Write(buf, binary.LittleEndian, fp16Bits)
	}
	
	return buf.Bytes(), scale, zeroPoint
}

func (tc *TensorCompressor) quantizeToINT8Tensor(data []float32, minVal, maxVal float64) ([]byte, float64, int32) {
	scale := (maxVal - minVal) / 255.0
	zeroPoint := int32(-minVal/scale)
	
	if zeroPoint < 0 {
		zeroPoint = 0
	}
	if zeroPoint > 255 {
		zeroPoint = 255
	}
	
	buf := new(bytes.Buffer)
	for _, val := range data {
		quantized := int32(float64(val)/scale) + zeroPoint
		if quantized < 0 {
			quantized = 0
		}
		if quantized > 255 {
			quantized = 255
		}
		buf.WriteByte(byte(quantized))
	}
	
	return buf.Bytes(), scale, zeroPoint
}

func (tc *TensorCompressor) quantizeTo4BitTensor(data []float32, minVal, maxVal float64) ([]byte, float64, int32) {
	scale := (maxVal - minVal) / 15.0
	zeroPoint := int32(-minVal/scale)
	
	if zeroPoint < 0 {
		zeroPoint = 0
	}
	if zeroPoint > 15 {
		zeroPoint = 15
	}
	
	buf := new(bytes.Buffer)
	for i := 0; i < len(data); i += 2 {
		val1 := int32(float64(data[i])/scale) + zeroPoint
		if val1 < 0 {
			val1 = 0
		}
		if val1 > 15 {
			val1 = 15
		}
		
		var val2 int32 = 0
		if i+1 < len(data) {
			val2 = int32(float64(data[i+1])/scale) + zeroPoint
			if val2 < 0 {
				val2 = 0
			}
			if val2 > 15 {
				val2 = 15
			}
		}
		
		packed := byte((val1 << 4) | val2)
		buf.WriteByte(packed)
	}
	
	return buf.Bytes(), scale, zeroPoint
}

func (tc *TensorCompressor) dequantizeFromFP16Tensor(data []byte, totalElements int) ([]float32, error) {
	result := make([]float32, totalElements)
	buf := bytes.NewReader(data)
	
	for i := 0; i < totalElements; i++ {
		var fp16Bits uint16
		err := binary.Read(buf, binary.LittleEndian, &fp16Bits)
		if err != nil {
			return nil, err
		}
		
		fp32Bits := uint32(fp16Bits) << 16
		result[i] = math.Float32frombits(fp32Bits)
	}
	
	return result, nil
}

func (tc *TensorCompressor) dequantizeFromINT8Tensor(data []byte, scale float64, zeroPoint int32, totalElements int) ([]float32, error) {
	result := make([]float32, totalElements)
	
	for i := 0; i < totalElements && i < len(data); i++ {
		quantized := int32(data[i])
		dequantized := float32((quantized - zeroPoint) * int32(scale))
		result[i] = dequantized
	}
	
	return result, nil
}

func (tc *TensorCompressor) dequantizeFrom4BitTensor(data []byte, scale float64, zeroPoint int32, totalElements int) ([]float32, error) {
	result := make([]float32, totalElements)
	
	for i := 0; i < totalElements; i += 2 {
		byteIdx := i / 2
		if byteIdx >= len(data) {
			break
		}
		
		packed := data[byteIdx]
		val1 := int32((packed >> 4) & 0x0F)
		val2 := int32(packed & 0x0F)
		
		result[i] = float32((val1 - zeroPoint) * int32(scale))
		if i+1 < totalElements {
			result[i+1] = float32((val2 - zeroPoint) * int32(scale))
		}
	}
	
	return result, nil
}

// Simplified SVD implementation (for demonstration)
func (tc *TensorCompressor) simplifiedSVD(matrix [][]float64, rank int) ([][]float64, []float64, [][]float64) {
	rows := len(matrix)
	cols := len(matrix[0])
	
	// For this example, return simplified matrices
	// In practice, use proper SVD from libraries like gonum
	u := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		u[i] = make([]float64, rank)
		for j := 0; j < rank; j++ {
			u[i][j] = 1.0 / float64(rank)
		}
	}
	
	s := make([]float64, rank)
	for i := 0; i < rank; i++ {
		s[i] = 1.0
	}
	
	vt := make([][]float64, rank)
	for i := 0; i < rank; i++ {
		vt[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			vt[i][j] = 1.0 / float64(rank)
		}
	}
	
	return u, s, vt
}

// Simple k-means implementation
func (tc *TensorCompressor) simpleKMeans(data []float32, k int) ([]float32, []int) {
	if k >= len(data) {
		k = len(data) / 2
	}
	if k <= 0 {
		k = 1
	}
	
	// Initialize clusters with random data points
	clusters := make([]float32, k)
	step := len(data) / k
	for i := 0; i < k; i++ {
		clusters[i] = data[i*step]
	}
	
	// Simple assignment based on nearest cluster
	assignments := make([]int, len(data))
	for i, val := range data {
		minDist := math.Abs(float64(val - clusters[0]))
		bestCluster := 0
		
		for j := 1; j < k; j++ {
			dist := math.Abs(float64(val - clusters[j]))
			if dist < minDist {
				minDist = dist
				bestCluster = j
			}
		}
		assignments[i] = bestCluster
	}
	
	return clusters, assignments
}

// Helper functions
func (tc *TensorCompressor) flattenMatrix(matrix [][]float64) []float32 {
	var result []float32
	for i := range matrix {
		for j := range matrix[i] {
			result = append(result, float32(matrix[i][j]))
		}
	}
	return result
}

func (tc *TensorCompressor) convertFloat64ToFloat32(input []float64) []float32 {
	result := make([]float32, len(input))
	for i, val := range input {
		result[i] = float32(val)
	}
	return result
}

func (tc *TensorCompressor) reshapeMatrix(data []float32, rows, cols int) [][]float32 {
	result := make([][]float32, rows)
	for i := 0; i < rows; i++ {
		result[i] = make([]float32, cols)
		for j := 0; j < cols; j++ {
			if i*cols+j < len(data) {
				result[i][j] = data[i*cols+j]
			}
		}
	}
	return result
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// GetTensorCompressionStats returns compression statistics for a tensor
func (tc *TensorCompressor) GetTensorCompressionStats(compressed *CompressedTensor) map[string]interface{} {
	stats := map[string]interface{}{
		"algorithm":           compressed.Algorithm,
		"compression_ratio":   compressed.CompressionRatio,
		"original_size_bytes": len(compressed.OriginalTensor.Data) * 4,
		"compressed_size_bytes": len(compressed.CompressedData),
		"parameter_reduction": 1.0 - compressed.CompressionRatio,
		"tensor_type":         compressed.OriginalTensor.Type,
		"tensor_shape":        compressed.OriginalTensor.Shape,
	}
	
	if compressed.CompressionMetrics != nil {
		stats["compression_time_ms"] = compressed.CompressionMetrics.CompressionTimeMs
		stats["decompression_time_ms"] = compressed.CompressionMetrics.DecompressionTimeMs
		if compressed.CompressionMetrics.ErrorMetrics != nil {
			stats["l1_error"] = compressed.CompressionMetrics.ErrorMetrics.L1Error
			stats["l2_error"] = compressed.CompressionMetrics.ErrorMetrics.L2Error
			stats["relative_error"] = compressed.CompressionMetrics.ErrorMetrics.RelativeError
			stats["snr_db"] = compressed.CompressionMetrics.ErrorMetrics.SNR
		}
	}
	
	return stats
}