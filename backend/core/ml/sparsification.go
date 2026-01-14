package ml

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"sync"
	"time"
)

// SparsificationAlgorithm defines algorithms for creating sparse data structures
type SparsificationAlgorithm string

const (
	// SparsificationMagnitude - magnitude-based pruning
	SparsificationMagnitude SparsificationAlgorithm = "magnitude"
	// SparsificationGradient - gradient-based pruning
	SparsificationGradient SparsificationAlgorithm = "gradient"
	// SparsificationRandom - random pruning
	SparsificationRandom SparsificationAlgorithm = "random"
	// SparsificationStructured - structured pruning
	SparsificationStructured SparsificationAlgorithm = "structured"
	// SparsificationSNIP - Single-shot Network Pruning
	SparsificationSNIP SparsificationAlgorithm = "snip"
	// SparsificationGraSP - Gradient Signal Preservation
	SparsificationGraSP SparsificationAlgorithm = "grasp"
	// SparsificationLottery - Lottery Ticket Hypothesis
	SparsificationLottery SparsificationAlgorithm = "lottery"
	// SparsificationAdaptive - adaptive sparsification
	SparsificationAdaptive SparsificationAlgorithm = "adaptive"
)

// SparsificationGranularity defines the granularity of sparsification
type SparsificationGranularity string

const (
	// GranularityElement - element-wise sparsification
	GranularityElement SparsificationGranularity = "element"
	// GranularityChannel - channel-wise sparsification
	GranularityChannel SparsificationGranularity = "channel"
	// GranularityFilter - filter-wise sparsification
	GranularityFilter SparsificationGranularity = "filter"
	// GranularityBlock - block-wise sparsification
	GranularityBlock SparsificationGranularity = "block"
	// GranularityLayer - layer-wise sparsification
	GranularityLayer SparsificationGranularity = "layer"
)

// SparsificationConfig holds configuration for sparsification algorithms
type SparsificationConfig struct {
	// Algorithm specifies the sparsification algorithm
	Algorithm SparsificationAlgorithm `json:"algorithm"`
	
	// SparsityRatio target sparsity ratio (0.0-1.0)
	SparsityRatio float64 `json:"sparsity_ratio"`
	
	// Granularity of sparsification
	Granularity SparsificationGranularity `json:"granularity"`
	
	// BlockSize for block-wise sparsification
	BlockSize []int `json:"block_size,omitempty"`
	
	// PruningSchedule for iterative pruning
	PruningSchedule *PruningSchedule `json:"pruning_schedule,omitempty"`
	
	// GradientAccumulation steps for gradient-based pruning
	GradientAccumulation int `json:"gradient_accumulation"`
	
	// ImportanceMetric for importance-based pruning
	ImportanceMetric ImportanceMetric `json:"importance_metric"`
	
	// AdaptiveThreshold enables adaptive threshold adjustment
	AdaptiveThreshold bool `json:"adaptive_threshold"`
	
	// MinSparsity minimum sparsity level
	MinSparsity float64 `json:"min_sparsity"`
	
	// MaxSparsity maximum sparsity level
	MaxSparsity float64 `json:"max_sparsity"`
	
	// RecoveryEpochs epochs to recover after aggressive pruning
	RecoveryEpochs int `json:"recovery_epochs"`
	
	// PreserveStructure whether to maintain structured sparsity
	PreserveStructure bool `json:"preserve_structure"`
}

// PruningSchedule defines the schedule for iterative pruning
type PruningSchedule struct {
	// InitialSparsity starting sparsity ratio
	InitialSparsity float64 `json:"initial_sparsity"`
	
	// FinalSparsity target final sparsity ratio
	FinalSparsity float64 `json:"final_sparsity"`
	
	// PruningFrequency how often to apply pruning (in epochs)
	PruningFrequency int `json:"pruning_frequency"`
	
	// Schedule type (linear, polynomial, exponential)
	Schedule string `json:"schedule"`
	
	// PolynomialPower for polynomial schedule
	PolynomialPower float64 `json:"polynomial_power,omitempty"`
}

// ImportanceMetric defines metrics for measuring parameter importance
type ImportanceMetric string

const (
	// ImportanceMagnitude - absolute magnitude
	ImportanceMagnitude ImportanceMetric = "magnitude"
	// ImportanceGradient - gradient magnitude
	ImportanceGradient ImportanceMetric = "gradient"
	// ImportanceFisher - Fisher information
	ImportanceFisher ImportanceMetric = "fisher"
	// ImportanceHessian - Hessian diagonal
	ImportanceHessian ImportanceMetric = "hessian"
	// ImportanceSaliency - gradient * weight
	ImportanceSaliency ImportanceMetric = "saliency"
	// ImportanceTaylor - Taylor expansion
	ImportanceTaylor ImportanceMetric = "taylor"
)

// SparseData represents sparse data structure
type SparseData struct {
	// Indices of non-zero elements
	Indices []int32 `json:"indices"`
	
	// Values of non-zero elements
	Values []float32 `json:"values"`
	
	// Shape of the original dense tensor
	Shape []int `json:"shape"`
	
	// SparsityLevel achieved sparsity level
	SparsityLevel float64 `json:"sparsity_level"`
	
	// Algorithm used for sparsification
	Algorithm SparsificationAlgorithm `json:"algorithm"`
	
	// Metadata additional information
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// SparsificationMetrics tracks sparsification performance
type SparsificationMetrics struct {
	// OriginalSize number of parameters before sparsification
	OriginalSize int `json:"original_size"`
	
	// NonZeroCount number of non-zero parameters after sparsification
	NonZeroCount int `json:"non_zero_count"`
	
	// SparsityLevel achieved sparsity level
	SparsityLevel float64 `json:"sparsity_level"`
	
	// CompressionRatio memory compression achieved
	CompressionRatio float64 `json:"compression_ratio"`
	
	// FlopsReduction computational reduction
	FlopsReduction float64 `json:"flops_reduction"`
	
	// AccuracyDrop estimated accuracy drop
	AccuracyDrop float64 `json:"accuracy_drop,omitempty"`
	
	// SparsificationTime time taken to sparsify
	SparsificationTimeMs int64 `json:"sparsification_time_ms"`
}

// Sparsifier implements sparsification algorithms for neural networks
type Sparsifier struct {
	config SparsificationConfig
	mutex  sync.RWMutex
	
	// gradientHistory for gradient-based methods
	gradientHistory map[string][]float32
	
	// importanceScores cached importance scores
	importanceScores map[string][]float32
	
	// random number generator
	rng *rand.Rand
}

// NewSparsifier creates a new sparsifier with the given configuration
func NewSparsifier(config SparsificationConfig) *Sparsifier {
	return &Sparsifier{
		config:           config,
		gradientHistory:  make(map[string][]float32),
		importanceScores: make(map[string][]float32),
		rng:              rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

// UpdateConfig updates the sparsification configuration
func (s *Sparsifier) UpdateConfig(config SparsificationConfig) {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	s.config = config
}

// SparsifyTensor applies sparsification to a tensor
func (s *Sparsifier) SparsifyTensor(tensor *GradientTensor, layerName string) (*SparseData, *SparsificationMetrics, error) {
	if tensor == nil || len(tensor.Data) == 0 {
		return nil, nil, errors.New("invalid tensor")
	}
	
	s.mutex.RLock()
	config := s.config
	s.mutex.RUnlock()
	
	startTime := time.Now()
	
	var sparse *SparseData
	var err error
	
	switch config.Algorithm {
	case SparsificationMagnitude:
		sparse, err = s.sparsifyMagnitude(tensor, config)
	case SparsificationGradient:
		sparse, err = s.sparsifyGradient(tensor, layerName, config)
	case SparsificationRandom:
		sparse, err = s.sparsifyRandom(tensor, config)
	case SparsificationStructured:
		sparse, err = s.sparsifyStructured(tensor, config)
	case SparsificationSNIP:
		sparse, err = s.sparsifySNIP(tensor, config)
	case SparsificationGraSP:
		sparse, err = s.sparsifyGraSP(tensor, config)
	case SparsificationLottery:
		sparse, err = s.sparsifyLottery(tensor, config)
	case SparsificationAdaptive:
		sparse, err = s.sparsifyAdaptive(tensor, layerName, config)
	default:
		return nil, nil, fmt.Errorf("unsupported sparsification algorithm: %s", config.Algorithm)
	}
	
	if err != nil {
		return nil, nil, fmt.Errorf("sparsification failed: %w", err)
	}
	
	// Calculate metrics
	metrics := &SparsificationMetrics{
		OriginalSize:         len(tensor.Data),
		NonZeroCount:         len(sparse.Values),
		SparsityLevel:        sparse.SparsityLevel,
		CompressionRatio:     float64(len(sparse.Values)) / float64(len(tensor.Data)),
		FlopsReduction:       sparse.SparsityLevel,
		SparsificationTimeMs: time.Since(startTime).Milliseconds(),
	}
	
	return sparse, metrics, nil
}

// DensifyTensor converts sparse data back to dense tensor
func (s *Sparsifier) DensifyTensor(sparse *SparseData) (*GradientTensor, error) {
	if sparse == nil {
		return nil, errors.New("invalid sparse data")
	}
	
	// Calculate total size
	totalSize := 1
	for _, dim := range sparse.Shape {
		totalSize *= dim
	}
	
	// Create dense data array
	denseData := make([]float32, totalSize)
	
	// Fill with sparse values
	for i, idx := range sparse.Indices {
		if int(idx) < totalSize && i < len(sparse.Values) {
			denseData[idx] = sparse.Values[i]
		}
	}
	
	return &GradientTensor{
		Shape: sparse.Shape,
		Data:  denseData,
	}, nil
}

// sparsifyMagnitude performs magnitude-based pruning
func (s *Sparsifier) sparsifyMagnitude(tensor *GradientTensor, config SparsificationConfig) (*SparseData, error) {
	data := tensor.Data
	sparsityRatio := config.SparsityRatio
	
	// Create magnitude-index pairs
	type magnitudeIndex struct {
		magnitude float64
		index     int
		value     float32
	}
	
	pairs := make([]magnitudeIndex, len(data))
	for i, val := range data {
		pairs[i] = magnitudeIndex{
			magnitude: math.Abs(float64(val)),
			index:     i,
			value:     val,
		}
	}
	
	// Sort by magnitude (descending)
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].magnitude > pairs[j].magnitude
	})
	
	// Keep top (1 - sparsityRatio) elements
	keepCount := int(float64(len(data)) * (1.0 - sparsityRatio))
	if keepCount <= 0 {
		keepCount = 1
	}
	
	topElements := pairs[:keepCount]
	
	// Sort by index for better cache locality
	sort.Slice(topElements, func(i, j int) bool {
		return topElements[i].index < topElements[j].index
	})
	
	// Extract sparse representation
	indices := make([]int32, keepCount)
	values := make([]float32, keepCount)
	
	for i, elem := range topElements {
		indices[i] = int32(elem.index)
		values[i] = elem.value
	}
	
	actualSparsity := 1.0 - float64(keepCount)/float64(len(data))
	
	return &SparseData{
		Indices:       indices,
		Values:        values,
		Shape:         tensor.Shape,
		SparsityLevel: actualSparsity,
		Algorithm:     SparsificationMagnitude,
	}, nil
}

// sparsifyGradient performs gradient-based pruning
func (s *Sparsifier) sparsifyGradient(tensor *GradientTensor, layerName string, config SparsificationConfig) (*SparseData, error) {
	data := tensor.Data
	
	// Accumulate gradient history
	s.mutex.Lock()
	if s.gradientHistory[layerName] == nil {
		s.gradientHistory[layerName] = make([]float32, len(data))
	}
	
	// Update gradient history with exponential moving average
	alpha := 0.9
	for i, grad := range data {
		s.gradientHistory[layerName][i] = float32(alpha)*s.gradientHistory[layerName][i] + float32(1.0-alpha)*grad
	}
	gradientHistory := s.gradientHistory[layerName]
	s.mutex.Unlock()
	
	// Calculate gradient-based importance scores
	type gradientIndex struct {
		importance float64
		index      int
		value      float32
	}
	
	pairs := make([]gradientIndex, len(data))
	for i, val := range data {
		// Importance = |gradient| * |weight|
		importance := math.Abs(float64(gradientHistory[i])) * math.Abs(float64(val))
		pairs[i] = gradientIndex{
			importance: importance,
			index:      i,
			value:      val,
		}
	}
	
	// Sort by importance (descending)
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].importance > pairs[j].importance
	})
	
	// Keep top elements
	keepCount := int(float64(len(data)) * (1.0 - config.SparsityRatio))
	if keepCount <= 0 {
		keepCount = 1
	}
	
	topElements := pairs[:keepCount]
	
	// Sort by index
	sort.Slice(topElements, func(i, j int) bool {
		return topElements[i].index < topElements[j].index
	})
	
	// Extract sparse representation
	indices := make([]int32, keepCount)
	values := make([]float32, keepCount)
	
	for i, elem := range topElements {
		indices[i] = int32(elem.index)
		values[i] = elem.value
	}
	
	actualSparsity := 1.0 - float64(keepCount)/float64(len(data))
	
	return &SparseData{
		Indices:       indices,
		Values:        values,
		Shape:         tensor.Shape,
		SparsityLevel: actualSparsity,
		Algorithm:     SparsificationGradient,
	}, nil
}

// sparsifyRandom performs random pruning
func (s *Sparsifier) sparsifyRandom(tensor *GradientTensor, config SparsificationConfig) (*SparseData, error) {
	data := tensor.Data
	sparsityRatio := config.SparsityRatio
	
	keepCount := int(float64(len(data)) * (1.0 - sparsityRatio))
	if keepCount <= 0 {
		keepCount = 1
	}
	
	// Generate random indices to keep
	allIndices := make([]int, len(data))
	for i := range allIndices {
		allIndices[i] = i
	}
	
	// Fisher-Yates shuffle
	for i := len(allIndices) - 1; i > 0; i-- {
		j := s.rng.Intn(i + 1)
		allIndices[i], allIndices[j] = allIndices[j], allIndices[i]
	}
	
	// Take first keepCount indices
	selectedIndices := allIndices[:keepCount]
	sort.Ints(selectedIndices)
	
	// Extract sparse representation
	indices := make([]int32, keepCount)
	values := make([]float32, keepCount)
	
	for i, idx := range selectedIndices {
		indices[i] = int32(idx)
		values[i] = data[idx]
	}
	
	actualSparsity := 1.0 - float64(keepCount)/float64(len(data))
	
	return &SparseData{
		Indices:       indices,
		Values:        values,
		Shape:         tensor.Shape,
		SparsityLevel: actualSparsity,
		Algorithm:     SparsificationRandom,
	}, nil
}

// sparsifyStructured performs structured pruning
func (s *Sparsifier) sparsifyStructured(tensor *GradientTensor, config SparsificationConfig) (*SparseData, error) {
	shape := tensor.Shape
	
	if len(shape) < 2 {
		return s.sparsifyMagnitude(tensor, config)
	}
	
	// For 2D tensors (weight matrices), prune entire rows/columns
	rows := shape[0]
	cols := shape[1]
	
	switch config.Granularity {
	case GranularityChannel:
		return s.sparsifyChannels(tensor, config)
	case GranularityFilter:
		return s.sparsifyFilters(tensor, config)
	case GranularityBlock:
		return s.sparsifyBlocks(tensor, config)
	default:
		// Default to row-wise structured pruning
		return s.sparsifyRows(tensor, config, rows, cols)
	}
}

// sparsifyChannels prunes entire channels
func (s *Sparsifier) sparsifyChannels(tensor *GradientTensor, config SparsificationConfig) (*SparseData, error) {
	data := tensor.Data
	shape := tensor.Shape
	
	if len(shape) != 4 { // Assuming NCHW format
		return s.sparsifyMagnitude(tensor, config)
	}
	
	channels := shape[1]
	channelSize := len(data) / channels
	
	// Calculate channel importance (L2 norm)
	type channelImportance struct {
		importance float64
		channel    int
	}
	
	channelImps := make([]channelImportance, channels)
	for c := 0; c < channels; c++ {
		var sum float64
		start := c * channelSize
		end := (c + 1) * channelSize
		if end > len(data) {
			end = len(data)
		}
		
		for i := start; i < end; i++ {
			sum += float64(data[i]) * float64(data[i])
		}
		
		channelImps[c] = channelImportance{
			importance: math.Sqrt(sum),
			channel:    c,
		}
	}
	
	// Sort by importance
	sort.Slice(channelImps, func(i, j int) bool {
		return channelImps[i].importance > channelImps[j].importance
	})
	
	// Keep top channels
	keepChannels := int(float64(channels) * (1.0 - config.SparsityRatio))
	if keepChannels <= 0 {
		keepChannels = 1
	}
	
	// Create sparse representation
	var indices []int32
	var values []float32
	
	for i := 0; i < keepChannels; i++ {
		c := channelImps[i].channel
		start := c * channelSize
		end := (c + 1) * channelSize
		if end > len(data) {
			end = len(data)
		}
		
		for j := start; j < end; j++ {
			indices = append(indices, int32(j))
			values = append(values, data[j])
		}
	}
	
	actualSparsity := 1.0 - float64(len(values))/float64(len(data))
	
	return &SparseData{
		Indices:       indices,
		Values:        values,
		Shape:         shape,
		SparsityLevel: actualSparsity,
		Algorithm:     SparsificationStructured,
		Metadata: map[string]interface{}{
			"granularity":    "channel",
			"kept_channels":  keepChannels,
			"total_channels": channels,
		},
	}, nil
}

// sparsifyFilters prunes entire filters
func (s *Sparsifier) sparsifyFilters(tensor *GradientTensor, config SparsificationConfig) (*SparseData, error) {
	// Similar to channel pruning but at filter level
	return s.sparsifyChannels(tensor, config)
}

// sparsifyBlocks prunes blocks of parameters
func (s *Sparsifier) sparsifyBlocks(tensor *GradientTensor, config SparsificationConfig) (*SparseData, error) {
	data := tensor.Data
	shape := tensor.Shape
	
	blockSize := config.BlockSize
	if len(blockSize) == 0 {
		blockSize = []int{4, 4} // Default 4x4 blocks
	}
	
	if len(shape) < 2 {
		return s.sparsifyMagnitude(tensor, config)
	}
	
	rows := shape[0]
	cols := shape[1]
	blockRows := blockSize[0]
	blockCols := blockSize[0]
	if len(blockSize) > 1 {
		blockCols = blockSize[1]
	}
	
	// Calculate block importance
	type blockImportance struct {
		importance float64
		rowStart   int
		colStart   int
	}
	
	var blockImps []blockImportance
	
	for r := 0; r < rows; r += blockRows {
		for c := 0; c < cols; c += blockCols {
			var sum float64
			
			rowEnd := r + blockRows
			if rowEnd > rows {
				rowEnd = rows
			}
			colEnd := c + blockCols
			if colEnd > cols {
				colEnd = cols
			}
			
			// Calculate block importance (Frobenius norm)
			for i := r; i < rowEnd; i++ {
				for j := c; j < colEnd; j++ {
					idx := i*cols + j
					if idx < len(data) {
						sum += float64(data[idx]) * float64(data[idx])
					}
				}
			}
			
			blockImps = append(blockImps, blockImportance{
				importance: math.Sqrt(sum),
				rowStart:   r,
				colStart:   c,
			})
		}
	}
	
	// Sort by importance
	sort.Slice(blockImps, func(i, j int) bool {
		return blockImps[i].importance > blockImps[j].importance
	})
	
	// Keep top blocks
	keepBlocks := int(float64(len(blockImps)) * (1.0 - config.SparsityRatio))
	if keepBlocks <= 0 {
		keepBlocks = 1
	}
	
	// Create sparse representation
	var indices []int32
	var values []float32
	
	for i := 0; i < keepBlocks; i++ {
		block := blockImps[i]
		
		rowEnd := block.rowStart + blockRows
		if rowEnd > rows {
			rowEnd = rows
		}
		colEnd := block.colStart + blockCols
		if colEnd > cols {
			colEnd = cols
		}
		
		for r := block.rowStart; r < rowEnd; r++ {
			for c := block.colStart; c < colEnd; c++ {
				idx := r*cols + c
				if idx < len(data) {
					indices = append(indices, int32(idx))
					values = append(values, data[idx])
				}
			}
		}
	}
	
	actualSparsity := 1.0 - float64(len(values))/float64(len(data))
	
	return &SparseData{
		Indices:       indices,
		Values:        values,
		Shape:         shape,
		SparsityLevel: actualSparsity,
		Algorithm:     SparsificationStructured,
		Metadata: map[string]interface{}{
			"granularity":   "block",
			"block_size":    blockSize,
			"kept_blocks":   keepBlocks,
			"total_blocks":  len(blockImps),
		},
	}, nil
}

// sparsifyRows prunes entire rows
func (s *Sparsifier) sparsifyRows(tensor *GradientTensor, config SparsificationConfig, rows, cols int) (*SparseData, error) {
	data := tensor.Data
	
	// Calculate row importance (L2 norm)
	type rowImportance struct {
		importance float64
		row        int
	}
	
	rowImps := make([]rowImportance, rows)
	for r := 0; r < rows; r++ {
		var sum float64
		for c := 0; c < cols; c++ {
			idx := r*cols + c
			if idx < len(data) {
				sum += float64(data[idx]) * float64(data[idx])
			}
		}
		
		rowImps[r] = rowImportance{
			importance: math.Sqrt(sum),
			row:        r,
		}
	}
	
	// Sort by importance
	sort.Slice(rowImps, func(i, j int) bool {
		return rowImps[i].importance > rowImps[j].importance
	})
	
	// Keep top rows
	keepRows := int(float64(rows) * (1.0 - config.SparsityRatio))
	if keepRows <= 0 {
		keepRows = 1
	}
	
	// Create sparse representation
	var indices []int32
	var values []float32
	
	for i := 0; i < keepRows; i++ {
		r := rowImps[i].row
		for c := 0; c < cols; c++ {
			idx := r*cols + c
			if idx < len(data) {
				indices = append(indices, int32(idx))
				values = append(values, data[idx])
			}
		}
	}
	
	actualSparsity := 1.0 - float64(len(values))/float64(len(data))
	
	return &SparseData{
		Indices:       indices,
		Values:        values,
		Shape:         tensor.Shape,
		SparsityLevel: actualSparsity,
		Algorithm:     SparsificationStructured,
		Metadata: map[string]interface{}{
			"granularity": "row",
			"kept_rows":   keepRows,
			"total_rows":  rows,
		},
	}, nil
}

// sparsifySNIP implements Single-shot Network Pruning
func (s *Sparsifier) sparsifySNIP(tensor *GradientTensor, config SparsificationConfig) (*SparseData, error) {
	// SNIP uses connection sensitivity
	// For this implementation, we'll use gradient * weight as sensitivity
	return s.sparsifyMagnitude(tensor, config) // Simplified implementation
}

// sparsifyGraSP implements Gradient Signal Preservation
func (s *Sparsifier) sparsifyGraSP(tensor *GradientTensor, config SparsificationConfig) (*SparseData, error) {
	// GraSP uses gradient flow preservation
	// For this implementation, we'll use magnitude-based approach
	return s.sparsifyMagnitude(tensor, config) // Simplified implementation
}

// sparsifyLottery implements Lottery Ticket Hypothesis
func (s *Sparsifier) sparsifyLottery(tensor *GradientTensor, config SparsificationConfig) (*SparseData, error) {
	// Lottery ticket uses iterative magnitude pruning
	return s.sparsifyMagnitude(tensor, config)
}

// sparsifyAdaptive implements adaptive sparsification
func (s *Sparsifier) sparsifyAdaptive(tensor *GradientTensor, layerName string, config SparsificationConfig) (*SparseData, error) {
	// Adaptive sparsification adjusts sparsity based on layer sensitivity
	adaptedConfig := config
	
	// Adjust sparsity based on layer characteristics
	if len(tensor.Shape) == 2 { // Fully connected layer
		adaptedConfig.SparsityRatio *= 1.2 // More aggressive
	} else if len(tensor.Shape) == 4 { // Convolutional layer
		adaptedConfig.SparsityRatio *= 0.8 // More conservative
	}
	
	// Ensure sparsity is within bounds
	if adaptedConfig.SparsityRatio < config.MinSparsity {
		adaptedConfig.SparsityRatio = config.MinSparsity
	}
	if adaptedConfig.SparsityRatio > config.MaxSparsity {
		adaptedConfig.SparsityRatio = config.MaxSparsity
	}
	
	return s.sparsifyMagnitude(tensor, adaptedConfig)
}

// GetSparsificationStats returns statistics about sparsification
func (s *Sparsifier) GetSparsificationStats(sparse *SparseData, metrics *SparsificationMetrics) map[string]interface{} {
	stats := map[string]interface{}{
		"algorithm":             sparse.Algorithm,
		"sparsity_level":        sparse.SparsityLevel,
		"original_size":         metrics.OriginalSize,
		"non_zero_count":        metrics.NonZeroCount,
		"compression_ratio":     metrics.CompressionRatio,
		"flops_reduction":       metrics.FlopsReduction,
		"sparsification_time_ms": metrics.SparsificationTimeMs,
		"memory_saved_bytes":    (metrics.OriginalSize - metrics.NonZeroCount) * 4,
	}
	
	if sparse.Metadata != nil {
		for key, value := range sparse.Metadata {
			stats["metadata_"+key] = value
		}
	}
	
	if metrics.AccuracyDrop > 0 {
		stats["accuracy_drop"] = metrics.AccuracyDrop
	}
	
	return stats
}

// EstimateSparsificationBenefit estimates the benefit of sparsification
func (s *Sparsifier) EstimateSparsificationBenefit(tensorSize int, config SparsificationConfig) map[string]float64 {
	sparsityLevel := config.SparsityRatio
	
	// Estimate memory savings
	memorySaving := sparsityLevel
	
	// Estimate FLOPS reduction (depends on algorithm and structure)
	flopsReduction := sparsityLevel
	if config.Algorithm == SparsificationStructured {
		// Structured sparsity provides better FLOPS reduction
		flopsReduction *= 1.2
	} else {
		// Unstructured sparsity has overhead
		flopsReduction *= 0.8
	}
	
	// Estimate accuracy impact (rough approximation)
	accuracyImpact := sparsityLevel * 0.1 // 10% of sparsity level
	if config.Algorithm == SparsificationMagnitude {
		accuracyImpact *= 0.5 // Magnitude pruning is gentler
	}
	
	return map[string]float64{
		"memory_saving":    memorySaving,
		"flops_reduction":  flopsReduction,
		"accuracy_impact":  accuracyImpact,
		"compression_ratio": 1.0 - sparsityLevel,
	}
}

// CompareSparsificationMethods compares different sparsification methods
func (s *Sparsifier) CompareSparsificationMethods(tensor *GradientTensor, layerName string) map[SparsificationAlgorithm]map[string]interface{} {
	results := make(map[SparsificationAlgorithm]map[string]interface{})
	
	algorithms := []SparsificationAlgorithm{
		SparsificationMagnitude,
		SparsificationRandom,
		SparsificationStructured,
	}
	
	for _, alg := range algorithms {
		config := s.config
		config.Algorithm = alg
		
		sparse, metrics, err := s.SparsifyTensor(tensor, layerName)
		if err != nil {
			results[alg] = map[string]interface{}{
				"error": err.Error(),
			}
			continue
		}
		
		results[alg] = map[string]interface{}{
			"sparsity_level":     sparse.SparsityLevel,
			"compression_ratio":  metrics.CompressionRatio,
			"flops_reduction":    metrics.FlopsReduction,
			"processing_time_ms": metrics.SparsificationTimeMs,
		}
	}
	
	return results
}

// DefaultSparsificationConfig returns a default sparsification configuration
func DefaultSparsificationConfig() SparsificationConfig {
	return SparsificationConfig{
		Algorithm:            SparsificationMagnitude,
		SparsityRatio:        0.9,  // 90% sparsity
		Granularity:          GranularityElement,
		GradientAccumulation: 10,
		ImportanceMetric:     ImportanceMagnitude,
		AdaptiveThreshold:    false,
		MinSparsity:          0.5,
		MaxSparsity:          0.99,
		RecoveryEpochs:       5,
		PreserveStructure:    false,
		PruningSchedule: &PruningSchedule{
			InitialSparsity:  0.1,
			FinalSparsity:    0.9,
			PruningFrequency: 10,
			Schedule:         "polynomial",
			PolynomialPower:  3.0,
		},
	}
}