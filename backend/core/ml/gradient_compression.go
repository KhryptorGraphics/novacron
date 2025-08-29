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

// GradientCompressionAlgorithm defines the compression algorithm for gradients
type GradientCompressionAlgorithm string

const (
	// CompressionNone - no compression
	CompressionNone GradientCompressionAlgorithm = "none"
	// CompressionTopK - top-k sparsification
	CompressionTopK GradientCompressionAlgorithm = "topk"
	// CompressionRandomK - random-k sparsification
	CompressionRandomK GradientCompressionAlgorithm = "randomk"
	// CompressionThreshold - threshold-based sparsification
	CompressionThreshold GradientCompressionAlgorithm = "threshold"
	// CompressionQuantization - gradient quantization
	CompressionQuantization GradientCompressionAlgorithm = "quantization"
	// CompressionHybrid - combination of sparsification and quantization
	CompressionHybrid GradientCompressionAlgorithm = "hybrid"
)

// QuantizationBits defines the bit precision for quantization
type QuantizationBits uint8

const (
	// Bits32 - FP32 (no quantization)
	Bits32 QuantizationBits = 32
	// Bits16 - FP16 quantization
	Bits16 QuantizationBits = 16
	// Bits8 - INT8 quantization
	Bits8 QuantizationBits = 8
	// Bits4 - 4-bit quantization (experimental)
	Bits4 QuantizationBits = 4
)

// GradientCompressionConfig holds configuration for gradient compression
type GradientCompressionConfig struct {
	// Algorithm specifies the compression algorithm to use
	Algorithm GradientCompressionAlgorithm `json:"algorithm"`
	
	// CompressionRatio target compression ratio (0.0-1.0)
	CompressionRatio float64 `json:"compression_ratio"`
	
	// TopKRatio for top-k sparsification (percentage of top gradients to keep)
	TopKRatio float64 `json:"topk_ratio"`
	
	// SparsityThreshold for threshold-based sparsification
	SparsityThreshold float64 `json:"sparsity_threshold"`
	
	// QuantizationBits for gradient quantization
	QuantizationBits QuantizationBits `json:"quantization_bits"`
	
	// EnableAdaptive enables adaptive compression based on gradient distribution
	EnableAdaptive bool `json:"enable_adaptive"`
	
	// MinCompressionRatio minimum compression ratio for adaptive mode
	MinCompressionRatio float64 `json:"min_compression_ratio"`
	
	// MaxCompressionRatio maximum compression ratio for adaptive mode
	MaxCompressionRatio float64 `json:"max_compression_ratio"`
	
	// ErrorBoundCompression enable error-bounded compression
	ErrorBoundCompression bool `json:"error_bound_compression"`
	
	// MaxRelativeError maximum relative error for error-bounded compression
	MaxRelativeError float64 `json:"max_relative_error"`
}

// DefaultGradientCompressionConfig returns default configuration for gradient compression
func DefaultGradientCompressionConfig() GradientCompressionConfig {
	return GradientCompressionConfig{
		Algorithm:             CompressionTopK,
		CompressionRatio:      0.1,  // 90% compression
		TopKRatio:            0.01,  // keep top 1% of gradients
		SparsityThreshold:    1e-5,  // threshold for sparse gradients
		QuantizationBits:     Bits16, // FP16 quantization
		EnableAdaptive:       true,
		MinCompressionRatio:  0.05,  // minimum 95% compression
		MaxCompressionRatio:  0.5,   // maximum 50% compression
		ErrorBoundCompression: false,
		MaxRelativeError:     0.01,  // 1% relative error bound
	}
}

// GradientTensor represents a gradient tensor with shape and data
type GradientTensor struct {
	// Shape of the tensor (dimensions)
	Shape []int `json:"shape"`
	
	// Data contains the gradient values as float32
	Data []float32 `json:"data"`
	
	// LayerName identifies the layer this gradient belongs to
	LayerName string `json:"layer_name"`
	
	// Timestamp when the gradient was computed
	Timestamp int64 `json:"timestamp"`
}

// CompressedGradient represents a compressed gradient with metadata
type CompressedGradient struct {
	// OriginalTensor reference to original tensor metadata
	OriginalTensor *GradientTensor `json:"original_tensor"`
	
	// CompressedData contains the compressed gradient data
	CompressedData []byte `json:"compressed_data"`
	
	// Algorithm used for compression
	Algorithm GradientCompressionAlgorithm `json:"algorithm"`
	
	// CompressionRatio achieved
	CompressionRatio float64 `json:"compression_ratio"`
	
	// SparsityLevel achieved (for sparse algorithms)
	SparsityLevel float64 `json:"sparsity_level"`
	
	// QuantizationLevel used (for quantization algorithms)
	QuantizationLevel QuantizationBits `json:"quantization_level"`
	
	// Indices for sparse gradients
	Indices []int32 `json:"indices,omitempty"`
	
	// Values for sparse gradients
	Values []float32 `json:"values,omitempty"`
	
	// QuantizationScale for dequantization
	QuantizationScale float64 `json:"quantization_scale,omitempty"`
	
	// QuantizationZeroPoint for asymmetric quantization
	QuantizationZeroPoint int32 `json:"quantization_zero_point,omitempty"`
	
	// ErrorMetrics contains compression error statistics
	ErrorMetrics *CompressionErrorMetrics `json:"error_metrics,omitempty"`
}

// CompressionErrorMetrics tracks compression error statistics
type CompressionErrorMetrics struct {
	// L1Error absolute error
	L1Error float64 `json:"l1_error"`
	
	// L2Error mean squared error
	L2Error float64 `json:"l2_error"`
	
	// RelativeError relative error
	RelativeError float64 `json:"relative_error"`
	
	// MaxError maximum absolute error
	MaxError float64 `json:"max_error"`
	
	// SNR signal-to-noise ratio
	SNR float64 `json:"snr"`
}

// GradientCompressor handles compression and decompression of gradients
type GradientCompressor struct {
	config GradientCompressionConfig
	mutex  sync.RWMutex
}

// NewGradientCompressor creates a new gradient compressor
func NewGradientCompressor(config GradientCompressionConfig) *GradientCompressor {
	return &GradientCompressor{
		config: config,
	}
}

// UpdateConfig updates the compression configuration
func (gc *GradientCompressor) UpdateConfig(config GradientCompressionConfig) {
	gc.mutex.Lock()
	defer gc.mutex.Unlock()
	gc.config = config
}

// GetConfig returns the current configuration
func (gc *GradientCompressor) GetConfig() GradientCompressionConfig {
	gc.mutex.RLock()
	defer gc.mutex.RUnlock()
	return gc.config
}

// CompressGradient compresses a gradient tensor using the configured algorithm
func (gc *GradientCompressor) CompressGradient(gradient *GradientTensor) (*CompressedGradient, error) {
	gc.mutex.RLock()
	config := gc.config
	gc.mutex.RUnlock()
	
	if gradient == nil || len(gradient.Data) == 0 {
		return nil, errors.New("invalid gradient tensor")
	}
	
	compressed := &CompressedGradient{
		OriginalTensor: gradient,
		Algorithm:      config.Algorithm,
	}
	
	var err error
	
	switch config.Algorithm {
	case CompressionNone:
		compressed, err = gc.compressNone(gradient)
	case CompressionTopK:
		compressed, err = gc.compressTopK(gradient, config)
	case CompressionRandomK:
		compressed, err = gc.compressRandomK(gradient, config)
	case CompressionThreshold:
		compressed, err = gc.compressThreshold(gradient, config)
	case CompressionQuantization:
		compressed, err = gc.compressQuantization(gradient, config)
	case CompressionHybrid:
		compressed, err = gc.compressHybrid(gradient, config)
	default:
		return nil, fmt.Errorf("unsupported compression algorithm: %s", config.Algorithm)
	}
	
	if err != nil {
		return nil, fmt.Errorf("compression failed: %w", err)
	}
	
	// Calculate error metrics
	if config.ErrorBoundCompression {
		decompressed, err := gc.DecompressGradient(compressed)
		if err == nil {
			compressed.ErrorMetrics = gc.calculateErrorMetrics(gradient.Data, decompressed.Data)
		}
	}
	
	return compressed, nil
}

// DecompressGradient decompresses a compressed gradient
func (gc *GradientCompressor) DecompressGradient(compressed *CompressedGradient) (*GradientTensor, error) {
	if compressed == nil || compressed.OriginalTensor == nil {
		return nil, errors.New("invalid compressed gradient")
	}
	
	var err error
	var data []float32
	
	switch compressed.Algorithm {
	case CompressionNone:
		data, err = gc.decompressNone(compressed)
	case CompressionTopK, CompressionRandomK, CompressionThreshold:
		data, err = gc.decompressSparse(compressed)
	case CompressionQuantization:
		data, err = gc.decompressQuantization(compressed)
	case CompressionHybrid:
		data, err = gc.decompressHybrid(compressed)
	default:
		return nil, fmt.Errorf("unsupported decompression algorithm: %s", compressed.Algorithm)
	}
	
	if err != nil {
		return nil, fmt.Errorf("decompression failed: %w", err)
	}
	
	result := &GradientTensor{
		Shape:     compressed.OriginalTensor.Shape,
		Data:      data,
		LayerName: compressed.OriginalTensor.LayerName,
		Timestamp: compressed.OriginalTensor.Timestamp,
	}
	
	return result, nil
}

// compressNone performs no compression (baseline)
func (gc *GradientCompressor) compressNone(gradient *GradientTensor) (*CompressedGradient, error) {
	// Convert float32 slice to bytes
	buf := new(bytes.Buffer)
	err := binary.Write(buf, binary.LittleEndian, gradient.Data)
	if err != nil {
		return nil, err
	}
	
	return &CompressedGradient{
		OriginalTensor:   gradient,
		CompressedData:   buf.Bytes(),
		Algorithm:        CompressionNone,
		CompressionRatio: 1.0,
		SparsityLevel:    0.0,
	}, nil
}

// compressTopK performs top-k sparsification
func (gc *GradientCompressor) compressTopK(gradient *GradientTensor, config GradientCompressionConfig) (*CompressedGradient, error) {
	data := gradient.Data
	k := int(float64(len(data)) * config.TopKRatio)
	
	if k <= 0 {
		k = 1
	}
	if k >= len(data) {
		return gc.compressNone(gradient)
	}
	
	// Create pairs of (absolute value, original index)
	type valueIndex struct {
		absValue float64
		index    int
		value    float32
	}
	
	pairs := make([]valueIndex, len(data))
	for i, val := range data {
		pairs[i] = valueIndex{
			absValue: math.Abs(float64(val)),
			index:    i,
			value:    val,
		}
	}
	
	// Sort by absolute value in descending order
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].absValue > pairs[j].absValue
	})
	
	// Take top-k elements
	topK := pairs[:k]
	
	// Sort by index for better compression
	sort.Slice(topK, func(i, j int) bool {
		return topK[i].index < topK[j].index
	})
	
	// Extract indices and values
	indices := make([]int32, k)
	values := make([]float32, k)
	
	for i, pair := range topK {
		indices[i] = int32(pair.index)
		values[i] = pair.value
	}
	
	// Compress indices and values using traditional compression
	indicesBytes := new(bytes.Buffer)
	valuesBytes := new(bytes.Buffer)
	
	binary.Write(indicesBytes, binary.LittleEndian, indices)
	binary.Write(valuesBytes, binary.LittleEndian, values)
	
	// Use existing compression infrastructure
	compressor := compression.NewCompressor(compression.DefaultCompressionConfig())
	compressedIndices, _ := compressor.Compress(indicesBytes.Bytes())
	compressedValues, _ := compressor.Compress(valuesBytes.Bytes())
	
	// Combine compressed data
	compressedData := append(compressedIndices, compressedValues...)
	
	originalSize := len(data) * 4
	compressedSize := len(compressedData)
	compressionRatio := float64(compressedSize) / float64(originalSize)
	sparsityLevel := 1.0 - float64(k)/float64(len(data))
	
	return &CompressedGradient{
		OriginalTensor:   gradient,
		CompressedData:   compressedData,
		Algorithm:        CompressionTopK,
		CompressionRatio: compressionRatio,
		SparsityLevel:    sparsityLevel,
		Indices:          indices,
		Values:           values,
	}, nil
}

// compressRandomK performs random-k sparsification
func (gc *GradientCompressor) compressRandomK(gradient *GradientTensor, config GradientCompressionConfig) (*CompressedGradient, error) {
	data := gradient.Data
	k := int(float64(len(data)) * config.TopKRatio)
	
	if k <= 0 {
		k = 1
	}
	if k >= len(data) {
		return gc.compressNone(gradient)
	}
	
	// Create random indices
	indices := make([]int, len(data))
	for i := range indices {
		indices[i] = i
	}
	
	// Simple Fisher-Yates shuffle for first k elements
	for i := 0; i < k; i++ {
		j := i + int(math.Floor(math.Mod(float64(i*7919+17), float64(len(data)-i))))
		indices[i], indices[j] = indices[j], indices[i]
	}
	
	// Sort selected indices
	selectedIndices := indices[:k]
	sort.Ints(selectedIndices)
	
	// Extract values
	indices32 := make([]int32, k)
	values := make([]float32, k)
	
	for i, idx := range selectedIndices {
		indices32[i] = int32(idx)
		values[i] = data[idx]
	}
	
	// Use same compression as top-k
	indicesBytes := new(bytes.Buffer)
	valuesBytes := new(bytes.Buffer)
	
	binary.Write(indicesBytes, binary.LittleEndian, indices32)
	binary.Write(valuesBytes, binary.LittleEndian, values)
	
	compressor := compression.NewCompressor(compression.DefaultCompressionConfig())
	compressedIndices, _ := compressor.Compress(indicesBytes.Bytes())
	compressedValues, _ := compressor.Compress(valuesBytes.Bytes())
	
	compressedData := append(compressedIndices, compressedValues...)
	
	originalSize := len(data) * 4
	compressedSize := len(compressedData)
	compressionRatio := float64(compressedSize) / float64(originalSize)
	sparsityLevel := 1.0 - float64(k)/float64(len(data))
	
	return &CompressedGradient{
		OriginalTensor:   gradient,
		CompressedData:   compressedData,
		Algorithm:        CompressionRandomK,
		CompressionRatio: compressionRatio,
		SparsityLevel:    sparsityLevel,
		Indices:          indices32,
		Values:           values,
	}, nil
}

// compressThreshold performs threshold-based sparsification
func (gc *GradientCompressor) compressThreshold(gradient *GradientTensor, config GradientCompressionConfig) (*CompressedGradient, error) {
	data := gradient.Data
	threshold := config.SparsityThreshold
	
	// Find elements above threshold
	var indices []int32
	var values []float32
	
	for i, val := range data {
		if math.Abs(float64(val)) > threshold {
			indices = append(indices, int32(i))
			values = append(values, val)
		}
	}
	
	if len(indices) == 0 {
		// All values below threshold, create minimal sparse representation
		indices = []int32{0}
		values = []float32{0.0}
	}
	
	// Compress using same method as top-k
	indicesBytes := new(bytes.Buffer)
	valuesBytes := new(bytes.Buffer)
	
	binary.Write(indicesBytes, binary.LittleEndian, indices)
	binary.Write(valuesBytes, binary.LittleEndian, values)
	
	compressor := compression.NewCompressor(compression.DefaultCompressionConfig())
	compressedIndices, _ := compressor.Compress(indicesBytes.Bytes())
	compressedValues, _ := compressor.Compress(valuesBytes.Bytes())
	
	compressedData := append(compressedIndices, compressedValues...)
	
	originalSize := len(data) * 4
	compressedSize := len(compressedData)
	compressionRatio := float64(compressedSize) / float64(originalSize)
	sparsityLevel := 1.0 - float64(len(indices))/float64(len(data))
	
	return &CompressedGradient{
		OriginalTensor:   gradient,
		CompressedData:   compressedData,
		Algorithm:        CompressionThreshold,
		CompressionRatio: compressionRatio,
		SparsityLevel:    sparsityLevel,
		Indices:          indices,
		Values:           values,
	}, nil
}

// compressQuantization performs gradient quantization
func (gc *GradientCompressor) compressQuantization(gradient *GradientTensor, config GradientCompressionConfig) (*CompressedGradient, error) {
	data := gradient.Data
	bits := config.QuantizationBits
	
	if bits == Bits32 {
		return gc.compressNone(gradient)
	}
	
	// Find min/max values for quantization
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
	
	// Calculate quantization parameters
	var quantizedData []byte
	var scale float64
	var zeroPoint int32
	
	switch bits {
	case Bits16:
		quantizedData, scale, zeroPoint = gc.quantizeToFP16(data, minVal, maxVal)
	case Bits8:
		quantizedData, scale, zeroPoint = gc.quantizeToINT8(data, minVal, maxVal)
	case Bits4:
		quantizedData, scale, zeroPoint = gc.quantizeTo4Bit(data, minVal, maxVal)
	default:
		return nil, fmt.Errorf("unsupported quantization bits: %d", bits)
	}
	
	originalSize := len(data) * 4
	compressedSize := len(quantizedData)
	compressionRatio := float64(compressedSize) / float64(originalSize)
	
	return &CompressedGradient{
		OriginalTensor:        gradient,
		CompressedData:        quantizedData,
		Algorithm:             CompressionQuantization,
		CompressionRatio:      compressionRatio,
		SparsityLevel:         0.0,
		QuantizationLevel:     bits,
		QuantizationScale:     scale,
		QuantizationZeroPoint: zeroPoint,
	}, nil
}

// compressHybrid combines sparsification and quantization
func (gc *GradientCompressor) compressHybrid(gradient *GradientTensor, config GradientCompressionConfig) (*CompressedGradient, error) {
	// First apply top-k sparsification
	sparseCompressed, err := gc.compressTopK(gradient, config)
	if err != nil {
		return nil, err
	}
	
	// Then quantize the selected values
	quantConfig := config
	quantConfig.Algorithm = CompressionQuantization
	
	// Create temporary tensor with sparse values
	sparseTensor := &GradientTensor{
		Shape:     gradient.Shape,
		Data:      sparseCompressed.Values,
		LayerName: gradient.LayerName,
		Timestamp: gradient.Timestamp,
	}
	
	quantCompressed, err := gc.compressQuantization(sparseTensor, quantConfig)
	if err != nil {
		return nil, err
	}
	
	// Combine compression results
	originalSize := len(gradient.Data) * 4
	hybridSize := len(quantCompressed.CompressedData) + len(sparseCompressed.Indices)*4
	compressionRatio := float64(hybridSize) / float64(originalSize)
	
	return &CompressedGradient{
		OriginalTensor:        gradient,
		CompressedData:        quantCompressed.CompressedData,
		Algorithm:             CompressionHybrid,
		CompressionRatio:      compressionRatio,
		SparsityLevel:         sparseCompressed.SparsityLevel,
		QuantizationLevel:     config.QuantizationBits,
		QuantizationScale:     quantCompressed.QuantizationScale,
		QuantizationZeroPoint: quantCompressed.QuantizationZeroPoint,
		Indices:               sparseCompressed.Indices,
		Values:                sparseCompressed.Values,
	}, nil
}

// Quantization helper methods

func (gc *GradientCompressor) quantizeToFP16(data []float32, minVal, maxVal float64) ([]byte, float64, int32) {
	// FP16 quantization - simple bit truncation
	buf := new(bytes.Buffer)
	scale := 1.0
	zeroPoint := int32(0)
	
	for _, val := range data {
		// Convert to FP16 by truncating mantissa
		bits := math.Float32bits(val)
		fp16Bits := uint16(bits >> 16) // Simple truncation
		binary.Write(buf, binary.LittleEndian, fp16Bits)
	}
	
	return buf.Bytes(), scale, zeroPoint
}

func (gc *GradientCompressor) quantizeToINT8(data []float32, minVal, maxVal float64) ([]byte, float64, int32) {
	// INT8 quantization with affine mapping
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

func (gc *GradientCompressor) quantizeTo4Bit(data []float32, minVal, maxVal float64) ([]byte, float64, int32) {
	// 4-bit quantization - pack 2 values per byte
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
		
		// Pack two 4-bit values into one byte
		packed := byte((val1 << 4) | val2)
		buf.WriteByte(packed)
	}
	
	return buf.Bytes(), scale, zeroPoint
}

// Decompression methods

func (gc *GradientCompressor) decompressNone(compressed *CompressedGradient) ([]float32, error) {
	buf := bytes.NewReader(compressed.CompressedData)
	totalElements := len(compressed.OriginalTensor.Data)
	data := make([]float32, totalElements)
	
	err := binary.Read(buf, binary.LittleEndian, data)
	return data, err
}

func (gc *GradientCompressor) decompressSparse(compressed *CompressedGradient) ([]float32, error) {
	totalElements := len(compressed.OriginalTensor.Data)
	result := make([]float32, totalElements)
	
	// Reconstruct from indices and values
	for i, idx := range compressed.Indices {
		if int(idx) < totalElements && i < len(compressed.Values) {
			result[idx] = compressed.Values[i]
		}
	}
	
	return result, nil
}

func (gc *GradientCompressor) decompressQuantization(compressed *CompressedGradient) ([]float32, error) {
	data := compressed.CompressedData
	bits := compressed.QuantizationLevel
	scale := compressed.QuantizationScale
	zeroPoint := compressed.QuantizationZeroPoint
	totalElements := len(compressed.OriginalTensor.Data)
	
	switch bits {
	case Bits16:
		return gc.dequantizeFromFP16(data, totalElements)
	case Bits8:
		return gc.dequantizeFromINT8(data, scale, zeroPoint, totalElements)
	case Bits4:
		return gc.dequantizeFrom4Bit(data, scale, zeroPoint, totalElements)
	default:
		return nil, fmt.Errorf("unsupported quantization bits: %d", bits)
	}
}

func (gc *GradientCompressor) decompressHybrid(compressed *CompressedGradient) ([]float32, error) {
	// First dequantize the values
	quantizedValues, err := gc.decompressQuantization(compressed)
	if err != nil {
		return nil, err
	}
	
	// Then reconstruct sparse tensor
	totalElements := len(compressed.OriginalTensor.Data)
	result := make([]float32, totalElements)
	
	// Map quantized values back to original positions
	for i, idx := range compressed.Indices {
		if int(idx) < totalElements && i < len(quantizedValues) {
			result[idx] = quantizedValues[i]
		}
	}
	
	return result, nil
}

// Dequantization helper methods

func (gc *GradientCompressor) dequantizeFromFP16(data []byte, totalElements int) ([]float32, error) {
	result := make([]float32, totalElements)
	buf := bytes.NewReader(data)
	
	for i := 0; i < totalElements; i++ {
		var fp16Bits uint16
		err := binary.Read(buf, binary.LittleEndian, &fp16Bits)
		if err != nil {
			return nil, err
		}
		
		// Convert back to FP32 by expanding mantissa
		fp32Bits := uint32(fp16Bits) << 16
		result[i] = math.Float32frombits(fp32Bits)
	}
	
	return result, nil
}

func (gc *GradientCompressor) dequantizeFromINT8(data []byte, scale float64, zeroPoint int32, totalElements int) ([]float32, error) {
	result := make([]float32, totalElements)
	
	for i := 0; i < totalElements && i < len(data); i++ {
		quantized := int32(data[i])
		dequantized := float32((quantized - zeroPoint) * int32(scale))
		result[i] = dequantized
	}
	
	return result, nil
}

func (gc *GradientCompressor) dequantizeFrom4Bit(data []byte, scale float64, zeroPoint int32, totalElements int) ([]float32, error) {
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

// calculateErrorMetrics computes error metrics between original and decompressed gradients
func (gc *GradientCompressor) calculateErrorMetrics(original, decompressed []float32) *CompressionErrorMetrics {
	if len(original) != len(decompressed) {
		return nil
	}
	
	n := float64(len(original))
	var l1Error, l2Error, maxError float64
	var originalNorm, noiseNorm float64
	
	for i := range original {
		diff := math.Abs(float64(original[i] - decompressed[i]))
		l1Error += diff
		l2Error += diff * diff
		
		if diff > maxError {
			maxError = diff
		}
		
		originalNorm += float64(original[i]) * float64(original[i])
		noiseNorm += diff * diff
	}
	
	l1Error /= n
	l2Error /= n
	l2Error = math.Sqrt(l2Error)
	
	originalRMS := math.Sqrt(originalNorm / n)
	relativeError := l2Error / originalRMS
	
	var snr float64
	if noiseNorm > 0 {
		snr = 10 * math.Log10(originalNorm/noiseNorm)
	} else {
		snr = math.Inf(1)
	}
	
	return &CompressionErrorMetrics{
		L1Error:       l1Error,
		L2Error:       l2Error,
		RelativeError: relativeError,
		MaxError:      maxError,
		SNR:           snr,
	}
}

// GetCompressionStats returns statistics about compression performance
func (gc *GradientCompressor) GetCompressionStats(compressed *CompressedGradient) map[string]interface{} {
	stats := map[string]interface{}{
		"algorithm":         compressed.Algorithm,
		"compression_ratio": compressed.CompressionRatio,
		"sparsity_level":    compressed.SparsityLevel,
		"quantization_bits": compressed.QuantizationLevel,
		"original_size":     len(compressed.OriginalTensor.Data) * 4,
		"compressed_size":   len(compressed.CompressedData),
	}
	
	if compressed.ErrorMetrics != nil {
		stats["l1_error"] = compressed.ErrorMetrics.L1Error
		stats["l2_error"] = compressed.ErrorMetrics.L2Error
		stats["relative_error"] = compressed.ErrorMetrics.RelativeError
		stats["max_error"] = compressed.ErrorMetrics.MaxError
		stats["snr_db"] = compressed.ErrorMetrics.SNR
	}
	
	return stats
}

// EstimateCompressionRatio estimates compression ratio for given configuration and data
func (gc *GradientCompressor) EstimateCompressionRatio(dataSize int, config GradientCompressionConfig) float64 {
	switch config.Algorithm {
	case CompressionNone:
		return 1.0
	case CompressionTopK, CompressionRandomK:
		// Sparsity compression: ratio depends on k and compression of indices/values
		sparsity := config.TopKRatio
		return sparsity * 1.2 // Account for index overhead and compression
	case CompressionThreshold:
		// Depends on data distribution, estimate conservatively
		return 0.3
	case CompressionQuantization:
		switch config.QuantizationBits {
		case Bits16:
			return 0.5
		case Bits8:
			return 0.25
		case Bits4:
			return 0.125
		default:
			return 1.0
		}
	case CompressionHybrid:
		// Combine sparsity and quantization
		sparsityRatio := config.TopKRatio * 1.2
		quantRatio := gc.EstimateCompressionRatio(dataSize, GradientCompressionConfig{
			Algorithm:        CompressionQuantization,
			QuantizationBits: config.QuantizationBits,
		})
		return sparsityRatio * quantRatio
	default:
		return 1.0
	}
}