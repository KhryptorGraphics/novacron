package compression

import (
	"fmt"
	"math"
	"math/rand"
)

// ModelCompressor provides model compression capabilities
type ModelCompressor struct {
	config *CompressionConfig
}

// CompressionConfig defines compression configuration
type CompressionConfig struct {
	Techniques        []string // "quantization", "pruning", "distillation", "low_rank"
	TargetCompression float64  // Target compression ratio (e.g., 5.0 for 5x)
	MaxAccuracyLoss   float64  // Max acceptable accuracy loss (e.g., 0.02 for 2%)
	QuantizationBits  int      // Bits for quantization (8, 16)
	PruningRatio      float64  // Ratio of weights to prune
	DistillationTemp  float64  // Temperature for knowledge distillation
}

// CompressionResult contains compression results
type CompressionResult struct {
	OriginalSize    int64
	CompressedSize  int64
	CompressionRatio float64
	AccuracyLoss    float64
	Technique       string
	Metrics         map[string]float64
}

// QuantizedModel represents a quantized neural network
type QuantizedModel struct {
	Weights        [][]int8
	Scales         []float64
	ZeroPoints     []int8
	QuantBits      int
	OriginalShape  []int
}

// PrunedModel represents a pruned neural network
type PrunedModel struct {
	Weights     [][]float64
	Mask        [][]bool
	PruningRatio float64
	Structured   bool
}

// DistilledModel represents a distilled model
type DistilledModel struct {
	StudentWeights [][]float64
	TeacherWeights [][]float64
	Temperature    float64
	Alpha          float64 // Distillation loss weight
}

// LowRankModel represents a low-rank factorized model
type LowRankModel struct {
	U    [][]float64
	V    [][]float64
	Rank int
}

// NewModelCompressor creates a new model compressor
func NewModelCompressor(config *CompressionConfig) *ModelCompressor {
	if config == nil {
		config = DefaultCompressionConfig()
	}
	return &ModelCompressor{config: config}
}

// DefaultCompressionConfig returns default compression configuration
func DefaultCompressionConfig() *CompressionConfig {
	return &CompressionConfig{
		Techniques:        []string{"quantization", "pruning"},
		TargetCompression: 5.0,
		MaxAccuracyLoss:   0.02,
		QuantizationBits:  8,
		PruningRatio:      0.5,
		DistillationTemp:  3.0,
	}
}

// Compress compresses a model using configured techniques
func (c *ModelCompressor) Compress(weights [][]float64) (*CompressionResult, [][]float64, error) {
	originalSize := c.calculateSize(weights)
	bestResult := &CompressionResult{
		OriginalSize:    originalSize,
		CompressedSize:  originalSize,
		CompressionRatio: 1.0,
	}
	bestWeights := weights

	for _, technique := range c.config.Techniques {
		var result *CompressionResult
		var compressed [][]float64
		var err error

		switch technique {
		case "quantization":
			result, compressed, err = c.quantize(weights)
		case "pruning":
			result, compressed, err = c.prune(weights)
		case "low_rank":
			result, compressed, err = c.lowRankFactorization(weights)
		default:
			continue
		}

		if err != nil {
			continue
		}

		// Check if this technique is better
		if result.CompressionRatio > bestResult.CompressionRatio &&
			result.AccuracyLoss <= c.config.MaxAccuracyLoss {
			bestResult = result
			bestWeights = compressed
		}
	}

	return bestResult, bestWeights, nil
}

// quantize performs weight quantization
func (c *ModelCompressor) quantize(weights [][]float64) (*CompressionResult, [][]float64, error) {
	if len(weights) == 0 || len(weights[0]) == 0 {
		return nil, nil, fmt.Errorf("empty weights")
	}

	// INT8 quantization
	quantBits := c.config.QuantizationBits
	qMin := -math.Pow(2, float64(quantBits-1))
	qMax := math.Pow(2, float64(quantBits-1)) - 1

	// Find min and max values
	minVal := weights[0][0]
	maxVal := weights[0][0]
	for i := range weights {
		for j := range weights[i] {
			if weights[i][j] < minVal {
				minVal = weights[i][j]
			}
			if weights[i][j] > maxVal {
				maxVal = weights[i][j]
			}
		}
	}

	// Calculate scale and zero point
	scale := (maxVal - minVal) / (qMax - qMin)
	zeroPoint := qMin - minVal/scale

	// Quantize
	quantized := make([][]int8, len(weights))
	for i := range weights {
		quantized[i] = make([]int8, len(weights[i]))
		for j := range weights[i] {
			qVal := weights[i][j]/scale + zeroPoint
			qVal = math.Max(qMin, math.Min(qMax, qVal))
			quantized[i][j] = int8(qVal)
		}
	}

	// Dequantize for accuracy estimation
	dequantized := make([][]float64, len(quantized))
	for i := range quantized {
		dequantized[i] = make([]float64, len(quantized[i]))
		for j := range quantized[i] {
			dequantized[i][j] = (float64(quantized[i][j]) - zeroPoint) * scale
		}
	}

	// Calculate compression ratio and accuracy loss
	originalSize := c.calculateSize(weights)
	compressedSize := int64(len(quantized) * len(quantized[0]) * quantBits / 8)
	compressionRatio := float64(originalSize) / float64(compressedSize)
	accuracyLoss := c.estimateAccuracyLoss(weights, dequantized)

	result := &CompressionResult{
		OriginalSize:    originalSize,
		CompressedSize:  compressedSize,
		CompressionRatio: compressionRatio,
		AccuracyLoss:    accuracyLoss,
		Technique:       fmt.Sprintf("quantization_int%d", quantBits),
		Metrics: map[string]float64{
			"scale":      scale,
			"zero_point": zeroPoint,
		},
	}

	return result, dequantized, nil
}

// prune performs weight pruning
func (c *ModelCompressor) prune(weights [][]float64) (*CompressionResult, [][]float64, error) {
	if len(weights) == 0 || len(weights[0]) == 0 {
		return nil, nil, fmt.Errorf("empty weights")
	}

	pruningRatio := c.config.PruningRatio

	// Magnitude-based pruning
	// Collect all weight magnitudes
	type weightInfo struct {
		i, j int
		mag  float64
	}

	allWeights := make([]weightInfo, 0)
	for i := range weights {
		for j := range weights[i] {
			allWeights = append(allWeights, weightInfo{
				i:   i,
				j:   j,
				mag: math.Abs(weights[i][j]),
			})
		}
	}

	// Sort by magnitude
	for i := 0; i < len(allWeights)-1; i++ {
		for j := i + 1; j < len(allWeights); j++ {
			if allWeights[j].mag < allWeights[i].mag {
				allWeights[i], allWeights[j] = allWeights[j], allWeights[i]
			}
		}
	}

	// Prune smallest weights
	numToPrune := int(float64(len(allWeights)) * pruningRatio)
	pruned := make([][]float64, len(weights))
	for i := range pruned {
		pruned[i] = make([]float64, len(weights[i]))
		copy(pruned[i], weights[i])
	}

	for i := 0; i < numToPrune; i++ {
		w := allWeights[i]
		pruned[w.i][w.j] = 0
	}

	// Calculate compression ratio (sparse representation)
	nonZero := len(allWeights) - numToPrune
	originalSize := c.calculateSize(weights)
	compressedSize := int64(nonZero * (8 + 4 + 4)) // value + row index + col index
	compressionRatio := float64(originalSize) / float64(compressedSize)
	accuracyLoss := c.estimateAccuracyLoss(weights, pruned)

	result := &CompressionResult{
		OriginalSize:    originalSize,
		CompressedSize:  compressedSize,
		CompressionRatio: compressionRatio,
		AccuracyLoss:    accuracyLoss,
		Technique:       "magnitude_pruning",
		Metrics: map[string]float64{
			"pruning_ratio": pruningRatio,
			"sparsity":      float64(numToPrune) / float64(len(allWeights)),
		},
	}

	return result, pruned, nil
}

// lowRankFactorization performs low-rank matrix factorization
func (c *ModelCompressor) lowRankFactorization(weights [][]float64) (*CompressionResult, [][]float64, error) {
	if len(weights) == 0 || len(weights[0]) == 0 {
		return nil, nil, fmt.Errorf("empty weights")
	}

	m := len(weights)
	n := len(weights[0])

	// Choose rank (compression target)
	targetRatio := c.config.TargetCompression
	rank := int(float64(m*n) / (targetRatio * float64(m+n)))
	if rank < 1 {
		rank = 1
	}
	if rank > min(m, n) {
		rank = min(m, n)
	}

	// Simplified SVD using power iteration
	U, V := c.simplifiedSVD(weights, rank)

	// Reconstruct
	reconstructed := make([][]float64, m)
	for i := range reconstructed {
		reconstructed[i] = make([]float64, n)
		for j := range reconstructed[i] {
			sum := 0.0
			for k := 0; k < rank; k++ {
				sum += U[i][k] * V[k][j]
			}
			reconstructed[i][j] = sum
		}
	}

	// Calculate compression
	originalSize := c.calculateSize(weights)
	compressedSize := int64((m*rank + rank*n) * 8) // U and V matrices
	compressionRatio := float64(originalSize) / float64(compressedSize)
	accuracyLoss := c.estimateAccuracyLoss(weights, reconstructed)

	result := &CompressionResult{
		OriginalSize:    originalSize,
		CompressedSize:  compressedSize,
		CompressionRatio: compressionRatio,
		AccuracyLoss:    accuracyLoss,
		Technique:       "low_rank_factorization",
		Metrics: map[string]float64{
			"rank": float64(rank),
		},
	}

	return result, reconstructed, nil
}

// simplifiedSVD performs simplified SVD using power iteration
func (c *ModelCompressor) simplifiedSVD(A [][]float64, rank int) ([][]float64, [][]float64) {
	m := len(A)
	n := len(A[0])

	U := make([][]float64, m)
	for i := range U {
		U[i] = make([]float64, rank)
		for j := range U[i] {
			U[i][j] = rand.NormFloat64()
		}
	}

	V := make([][]float64, rank)
	for i := range V {
		V[i] = make([]float64, n)
		for j := range V[i] {
			V[i][j] = rand.NormFloat64()
		}
	}

	// Power iteration
	iterations := 10
	for iter := 0; iter < iterations; iter++ {
		// Update U
		for i := 0; i < m; i++ {
			for k := 0; k < rank; k++ {
				sum := 0.0
				for j := 0; j < n; j++ {
					sum += A[i][j] * V[k][j]
				}
				U[i][k] = sum
			}
		}

		// Normalize U
		for k := 0; k < rank; k++ {
			norm := 0.0
			for i := 0; i < m; i++ {
				norm += U[i][k] * U[i][k]
			}
			norm = math.Sqrt(norm)
			if norm > 0 {
				for i := 0; i < m; i++ {
					U[i][k] /= norm
				}
			}
		}

		// Update V
		for k := 0; k < rank; k++ {
			for j := 0; j < n; j++ {
				sum := 0.0
				for i := 0; i < m; i++ {
					sum += A[i][j] * U[i][k]
				}
				V[k][j] = sum
			}
		}

		// Normalize V
		for k := 0; k < rank; k++ {
			norm := 0.0
			for j := 0; j < n; j++ {
				norm += V[k][j] * V[k][j]
			}
			norm = math.Sqrt(norm)
			if norm > 0 {
				for j := 0; j < n; j++ {
					V[k][j] /= norm
				}
			}
		}
	}

	return U, V
}

// KnowledgeDistillation performs knowledge distillation
func (c *ModelCompressor) KnowledgeDistillation(teacherWeights, studentWeights [][]float64, X [][]float64, y []float64) (*CompressionResult, [][]float64, error) {
	// Simplified knowledge distillation
	// In practice, implement full neural network training with distillation loss

	temperature := c.config.DistillationTemp
	alpha := 0.5 // Weight for distillation loss

	// Train student to match teacher's soft labels
	epochs := 100
	learningRate := 0.01

	student := make([][]float64, len(studentWeights))
	for i := range student {
		student[i] = make([]float64, len(studentWeights[i]))
		copy(student[i], studentWeights[i])
	}

	for epoch := 0; epoch < epochs; epoch++ {
		for i := range X {
			// Get teacher predictions (soft labels)
			teacherPred := c.predict(teacherWeights, X[i])
			teacherSoft := c.softmax(teacherPred, temperature)

			// Get student predictions
			studentPred := c.predict(student, X[i])
			studentSoft := c.softmax(studentPred, temperature)

			// Compute distillation loss
			distLoss := c.klDivergence(studentSoft, teacherSoft)

			// Compute hard label loss
			hardLoss := (studentPred[0] - y[i]) * (studentPred[0] - y[i])

			// Combined loss
			loss := alpha*distLoss + (1-alpha)*hardLoss

			// Gradient descent (simplified)
			for j := range student {
				for k := range student[j] {
					student[j][k] -= learningRate * loss * X[i][k%len(X[i])]
				}
			}
		}
	}

	originalSize := c.calculateSize(teacherWeights)
	compressedSize := c.calculateSize(student)
	compressionRatio := float64(originalSize) / float64(compressedSize)
	accuracyLoss := c.estimateAccuracyLoss(teacherWeights, student)

	result := &CompressionResult{
		OriginalSize:    originalSize,
		CompressedSize:  compressedSize,
		CompressionRatio: compressionRatio,
		AccuracyLoss:    accuracyLoss,
		Technique:       "knowledge_distillation",
		Metrics: map[string]float64{
			"temperature": temperature,
			"alpha":       alpha,
		},
	}

	return result, student, nil
}

// Helper functions

func (c *ModelCompressor) calculateSize(weights [][]float64) int64 {
	totalElements := 0
	for i := range weights {
		totalElements += len(weights[i])
	}
	return int64(totalElements * 8) // 8 bytes per float64
}

func (c *ModelCompressor) estimateAccuracyLoss(original, compressed [][]float64) float64 {
	// Simplified accuracy loss estimation using Frobenius norm
	sumSquaredDiff := 0.0
	sumSquaredOrig := 0.0

	for i := range original {
		for j := range original[i] {
			diff := original[i][j] - compressed[i][j]
			sumSquaredDiff += diff * diff
			sumSquaredOrig += original[i][j] * original[i][j]
		}
	}

	if sumSquaredOrig == 0 {
		return 0
	}

	// Normalize to [0, 1] range representing accuracy loss
	relativeLoss := math.Sqrt(sumSquaredDiff / sumSquaredOrig)
	return math.Min(relativeLoss, 1.0)
}

func (c *ModelCompressor) predict(weights [][]float64, x []float64) []float64 {
	// Simplified forward pass
	output := make([]float64, len(weights))
	for i := range weights {
		sum := 0.0
		for j := range weights[i] {
			if j < len(x) {
				sum += weights[i][j] * x[j]
			}
		}
		output[i] = sum
	}
	return output
}

func (c *ModelCompressor) softmax(x []float64, temperature float64) []float64 {
	result := make([]float64, len(x))
	sum := 0.0

	for i := range x {
		result[i] = math.Exp(x[i] / temperature)
		sum += result[i]
	}

	for i := range result {
		result[i] /= sum
	}

	return result
}

func (c *ModelCompressor) klDivergence(p, q []float64) float64 {
	kl := 0.0
	for i := range p {
		if p[i] > 0 && q[i] > 0 {
			kl += p[i] * math.Log(p[i]/q[i])
		}
	}
	return kl
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
