package monitoring

import (
	"context"
	"fmt"
	"math"
	"time"

	"go.uber.org/zap"
)

// IsolationForestModel implements anomaly detection using Isolation Forest
type IsolationForestModel struct {
	modelPath   string
	threshold   float64
	logger      *zap.Logger

	// Simplified implementation without ONNX for now
	// In production, use ONNX Runtime or similar
	trees       []*IsolationTree
	numTrees    int
	sampleSize  int
	contamination float64
}

// IsolationTree represents a single isolation tree
type IsolationTree struct {
	root *IsolationNode
	maxDepth int
}

// IsolationNode is a node in an isolation tree
type IsolationNode struct {
	splitFeature int
	splitValue   float64
	left         *IsolationNode
	right        *IsolationNode
	size         int
	depth        int
}

// NewIsolationForestModel creates a new Isolation Forest model
func NewIsolationForestModel(modelPath string, logger *zap.Logger) (*IsolationForestModel, error) {
	if logger == nil {
		logger = zap.NewNop()
	}

	return &IsolationForestModel{
		modelPath:     modelPath,
		threshold:     0.6, // Anomaly score threshold
		logger:        logger,
		numTrees:      100,
		sampleSize:    256,
		contamination: 0.01, // Expected 1% anomalies
		trees:         make([]*IsolationTree, 0),
	}, nil
}

// Detect detects anomalies using Isolation Forest
func (ifm *IsolationForestModel) Detect(ctx context.Context, metrics *MetricVector) (*Anomaly, error) {
	if len(ifm.trees) == 0 {
		return nil, fmt.Errorf("model not trained")
	}

	features := metrics.ToSlice()

	// Calculate anomaly score
	score := ifm.anomalyScore(features)

	// Determine if it's an anomaly
	isAnomaly := score > ifm.threshold

	if !isAnomaly {
		return nil, nil
	}

	// Calculate confidence
	confidence := math.Min(score/ifm.threshold, 1.0)

	// Find which metric is most anomalous
	metricName, expectedValue := ifm.findMostAnomalousMetric(features)
	actualValue := features[ifm.getMetricIndex(metricName)]
	deviation := math.Abs(actualValue - expectedValue)

	severity := calculateSeverity(confidence, deviation/expectedValue*100)

	return &Anomaly{
		Timestamp:   time.Now(),
		MetricName:  metricName,
		Value:       actualValue,
		Expected:    expectedValue,
		Deviation:   deviation,
		Severity:    severity,
		Confidence:  confidence,
		ModelType:   "isolation_forest",
		Description: fmt.Sprintf("Isolation Forest detected anomaly in %s (score: %.2f)", metricName, score),
		Context: map[string]interface{}{
			"score": score,
			"trees": len(ifm.trees),
		},
	}, nil
}

// Train trains the Isolation Forest (simplified implementation)
func (ifm *IsolationForestModel) Train(ctx context.Context, normalData []*MetricVector) error {
	if len(normalData) == 0 {
		return fmt.Errorf("no training data provided")
	}

	ifm.logger.Info("Training Isolation Forest",
		zap.Int("samples", len(normalData)),
		zap.Int("trees", ifm.numTrees))

	// Convert to feature matrix
	features := make([][]float64, len(normalData))
	for i, mv := range normalData {
		features[i] = mv.ToSlice()
	}

	// Build trees
	ifm.trees = make([]*IsolationTree, ifm.numTrees)
	for i := 0; i < ifm.numTrees; i++ {
		sample := ifm.sampleData(features)
		ifm.trees[i] = ifm.buildTree(sample, 0, int(math.Ceil(math.Log2(float64(ifm.sampleSize)))))
	}

	// Calculate threshold based on contamination
	scores := make([]float64, len(normalData))
	for i, mv := range normalData {
		scores[i] = ifm.anomalyScore(mv.ToSlice())
	}

	// Set threshold at contamination percentile
	ifm.threshold = ifm.percentile(scores, 1.0-ifm.contamination)

	ifm.logger.Info("Isolation Forest training completed",
		zap.Float64("threshold", ifm.threshold))

	return nil
}

// Name returns the detector name
func (ifm *IsolationForestModel) Name() string {
	return "isolation_forest"
}

// anomalyScore calculates the anomaly score for a sample
func (ifm *IsolationForestModel) anomalyScore(features []float64) float64 {
	avgPathLength := 0.0

	for _, tree := range ifm.trees {
		pathLength := ifm.pathLength(tree.root, features, 0)
		avgPathLength += pathLength
	}

	avgPathLength /= float64(len(ifm.trees))

	// Normalize by expected path length
	c := ifm.averagePathLength(ifm.sampleSize)
	score := math.Pow(2, -avgPathLength/c)

	return score
}

// pathLength calculates the path length for a sample in a tree
func (ifm *IsolationForestModel) pathLength(node *IsolationNode, features []float64, currentDepth int) float64 {
	if node.left == nil && node.right == nil {
		// External node
		return float64(currentDepth) + ifm.averagePathLength(node.size)
	}

	if features[node.splitFeature] < node.splitValue {
		return ifm.pathLength(node.left, features, currentDepth+1)
	}
	return ifm.pathLength(node.right, features, currentDepth+1)
}

// averagePathLength calculates the average path length for unsuccessful search in BST
func (ifm *IsolationForestModel) averagePathLength(n int) float64 {
	if n <= 1 {
		return 0
	}
	return 2.0*(math.Log(float64(n-1))+0.5772156649) - (2.0*float64(n-1))/float64(n)
}

// buildTree builds an isolation tree
func (ifm *IsolationForestModel) buildTree(data [][]float64, depth, maxDepth int) *IsolationTree {
	root := ifm.buildNode(data, depth, maxDepth)
	return &IsolationTree{
		root:     root,
		maxDepth: maxDepth,
	}
}

// buildNode recursively builds tree nodes
func (ifm *IsolationForestModel) buildNode(data [][]float64, depth, maxDepth int) *IsolationNode {
	node := &IsolationNode{
		size:  len(data),
		depth: depth,
	}

	// Termination conditions
	if depth >= maxDepth || len(data) <= 1 {
		return node
	}

	// Random split
	numFeatures := len(data[0])
	splitFeature := int(float64(numFeatures) * pseudoRandom())

	// Find min and max for the feature
	min, max := math.Inf(1), math.Inf(-1)
	for _, sample := range data {
		if sample[splitFeature] < min {
			min = sample[splitFeature]
		}
		if sample[splitFeature] > max {
			max = sample[splitFeature]
		}
	}

	if min == max {
		return node
	}

	// Random split value
	splitValue := min + (max-min)*pseudoRandom()

	node.splitFeature = splitFeature
	node.splitValue = splitValue

	// Partition data
	var leftData, rightData [][]float64
	for _, sample := range data {
		if sample[splitFeature] < splitValue {
			leftData = append(leftData, sample)
		} else {
			rightData = append(rightData, sample)
		}
	}

	if len(leftData) > 0 {
		node.left = ifm.buildNode(leftData, depth+1, maxDepth)
	}
	if len(rightData) > 0 {
		node.right = ifm.buildNode(rightData, depth+1, maxDepth)
	}

	return node
}

// sampleData randomly samples data
func (ifm *IsolationForestModel) sampleData(data [][]float64) [][]float64 {
	sampleSize := ifm.sampleSize
	if len(data) < sampleSize {
		sampleSize = len(data)
	}

	sample := make([][]float64, sampleSize)
	for i := 0; i < sampleSize; i++ {
		idx := int(float64(len(data)) * pseudoRandom())
		sample[i] = data[idx]
	}

	return sample
}

// findMostAnomalousMetric identifies which metric is most anomalous
func (ifm *IsolationForestModel) findMostAnomalousMetric(features []float64) (string, float64) {
	metricNames := []string{
		"bandwidth", "latency", "packet_loss", "jitter",
		"cpu_usage", "memory_usage", "error_rate",
	}

	maxDeviation := 0.0
	maxMetric := metricNames[0]
	expectedValue := 0.0

	// This is simplified - in production, track expected values from training data
	expectedValues := []float64{100.0, 10.0, 0.01, 1.0, 50.0, 60.0, 0.001}

	for i, value := range features {
		deviation := math.Abs(value - expectedValues[i])
		if deviation > maxDeviation {
			maxDeviation = deviation
			maxMetric = metricNames[i]
			expectedValue = expectedValues[i]
		}
	}

	return maxMetric, expectedValue
}

// getMetricIndex returns the index of a metric name
func (ifm *IsolationForestModel) getMetricIndex(metricName string) int {
	metrics := map[string]int{
		"bandwidth":    0,
		"latency":      1,
		"packet_loss":  2,
		"jitter":       3,
		"cpu_usage":    4,
		"memory_usage": 5,
		"error_rate":   6,
	}
	return metrics[metricName]
}

// percentile calculates the nth percentile
func (ifm *IsolationForestModel) percentile(values []float64, p float64) float64 {
	if len(values) == 0 {
		return 0
	}

	sorted := make([]float64, len(values))
	copy(sorted, values)

	// Simple sort
	for i := 0; i < len(sorted); i++ {
		for j := i + 1; j < len(sorted); j++ {
			if sorted[i] > sorted[j] {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}

	idx := int(float64(len(sorted)) * p)
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}

	return sorted[idx]
}

// Simple pseudo-random number generator (not cryptographically secure)
var randomState uint64 = uint64(time.Now().UnixNano())

func pseudoRandom() float64 {
	randomState = randomState*1103515245 + 12345
	return float64(randomState%1000000) / 1000000.0
}
