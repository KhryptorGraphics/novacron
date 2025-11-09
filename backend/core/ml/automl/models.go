package automl

import (
	"fmt"
	"math"
	"math/rand"
)

// RandomForestBuilder builds random forest models
type RandomForestBuilder struct{}

func (b *RandomForestBuilder) Build(params map[string]interface{}) Model {
	return &RandomForestModel{
		nTrees:     getIntParam(params, "n_trees", 100),
		maxDepth:   getIntParam(params, "max_depth", 10),
		minSamples: getIntParam(params, "min_samples", 2),
		trees:      make([]*DecisionTree, 0),
	}
}

func (b *RandomForestBuilder) DefaultParams() map[string]interface{} {
	return map[string]interface{}{
		"n_trees":     100,
		"max_depth":   10,
		"min_samples": 2,
	}
}

func (b *RandomForestBuilder) ParamSpace() map[string]ParamRange {
	return map[string]ParamRange{
		"n_trees": {
			Type: "int",
			Min:  10,
			Max:  500,
		},
		"max_depth": {
			Type: "int",
			Min:  3,
			Max:  20,
		},
		"min_samples": {
			Type: "int",
			Min:  1,
			Max:  10,
		},
	}
}

// RandomForestModel implements random forest
type RandomForestModel struct {
	nTrees     int
	maxDepth   int
	minSamples int
	trees      []*DecisionTree
}

func (m *RandomForestModel) Fit(X [][]float64, y []float64) error {
	m.trees = make([]*DecisionTree, m.nTrees)

	for i := 0; i < m.nTrees; i++ {
		// Bootstrap sample
		XSample, ySample := m.bootstrapSample(X, y)

		// Train tree
		tree := &DecisionTree{
			maxDepth:   m.maxDepth,
			minSamples: m.minSamples,
		}
		if err := tree.Fit(XSample, ySample); err != nil {
			return err
		}
		m.trees[i] = tree
	}

	return nil
}

func (m *RandomForestModel) Predict(X [][]float64) ([]float64, error) {
	predictions := make([]float64, len(X))

	for i, x := range X {
		sum := 0.0
		for _, tree := range m.trees {
			pred, _ := tree.PredictOne(x)
			sum += pred
		}
		predictions[i] = sum / float64(len(m.trees))
	}

	return predictions, nil
}

func (m *RandomForestModel) GetParams() map[string]interface{} {
	return map[string]interface{}{
		"n_trees":     m.nTrees,
		"max_depth":   m.maxDepth,
		"min_samples": m.minSamples,
	}
}

func (m *RandomForestModel) bootstrapSample(X [][]float64, y []float64) ([][]float64, []float64) {
	n := len(X)
	XSample := make([][]float64, n)
	ySample := make([]float64, n)

	for i := 0; i < n; i++ {
		idx := rand.Intn(n)
		XSample[i] = X[idx]
		ySample[i] = y[idx]
	}

	return XSample, ySample
}

// DecisionTree implements a decision tree
type DecisionTree struct {
	maxDepth   int
	minSamples int
	root       *TreeNode
}

type TreeNode struct {
	featureIdx int
	threshold  float64
	left       *TreeNode
	right      *TreeNode
	value      float64
	isLeaf     bool
}

func (t *DecisionTree) Fit(X [][]float64, y []float64) error {
	t.root = t.buildTree(X, y, 0)
	return nil
}

func (t *DecisionTree) buildTree(X [][]float64, y []float64, depth int) *TreeNode {
	// Check stopping criteria
	if depth >= t.maxDepth || len(X) <= t.minSamples {
		return &TreeNode{
			isLeaf: true,
			value:  mean(y),
		}
	}

	// Find best split
	bestFeature, bestThreshold, bestGain := t.findBestSplit(X, y)

	if bestGain <= 0 {
		return &TreeNode{
			isLeaf: true,
			value:  mean(y),
		}
	}

	// Split data
	XLeft, yLeft, XRight, yRight := t.splitData(X, y, bestFeature, bestThreshold)

	return &TreeNode{
		featureIdx: bestFeature,
		threshold:  bestThreshold,
		left:       t.buildTree(XLeft, yLeft, depth+1),
		right:      t.buildTree(XRight, yRight, depth+1),
		isLeaf:     false,
	}
}

func (t *DecisionTree) findBestSplit(X [][]float64, y []float64) (int, float64, float64) {
	bestFeature := 0
	bestThreshold := 0.0
	bestGain := 0.0

	if len(X) == 0 {
		return bestFeature, bestThreshold, bestGain
	}

	for feature := 0; feature < len(X[0]); feature++ {
		// Get unique values for threshold
		values := make([]float64, len(X))
		for i := range X {
			values[i] = X[i][feature]
		}

		// Try different thresholds
		for _, threshold := range values {
			gain := t.calculateGain(X, y, feature, threshold)
			if gain > bestGain {
				bestGain = gain
				bestFeature = feature
				bestThreshold = threshold
			}
		}
	}

	return bestFeature, bestThreshold, bestGain
}

func (t *DecisionTree) calculateGain(X [][]float64, y []float64, feature int, threshold float64) float64 {
	_, yLeft, _, yRight := t.splitData(X, y, feature, threshold)

	if len(yLeft) == 0 || len(yRight) == 0 {
		return 0
	}

	parentVar := variance(y)
	leftVar := variance(yLeft)
	rightVar := variance(yRight)

	n := float64(len(y))
	nLeft := float64(len(yLeft))
	nRight := float64(len(yRight))

	return parentVar - (nLeft/n)*leftVar - (nRight/n)*rightVar
}

func (t *DecisionTree) splitData(X [][]float64, y []float64, feature int, threshold float64) ([][]float64, []float64, [][]float64, []float64) {
	var XLeft, XRight [][]float64
	var yLeft, yRight []float64

	for i := range X {
		if X[i][feature] <= threshold {
			XLeft = append(XLeft, X[i])
			yLeft = append(yLeft, y[i])
		} else {
			XRight = append(XRight, X[i])
			yRight = append(yRight, y[i])
		}
	}

	return XLeft, yLeft, XRight, yRight
}

func (t *DecisionTree) PredictOne(x []float64) (float64, error) {
	node := t.root
	for !node.isLeaf {
		if x[node.featureIdx] <= node.threshold {
			node = node.left
		} else {
			node = node.right
		}
	}
	return node.value, nil
}

// XGBoostBuilder builds XGBoost models (simplified implementation)
type XGBoostBuilder struct{}

func (b *XGBoostBuilder) Build(params map[string]interface{}) Model {
	return &XGBoostModel{
		nEstimators:  getIntParam(params, "n_estimators", 100),
		learningRate: getFloatParam(params, "learning_rate", 0.1),
		maxDepth:     getIntParam(params, "max_depth", 6),
		trees:        make([]*DecisionTree, 0),
	}
}

func (b *XGBoostBuilder) DefaultParams() map[string]interface{} {
	return map[string]interface{}{
		"n_estimators":  100,
		"learning_rate": 0.1,
		"max_depth":     6,
	}
}

func (b *XGBoostBuilder) ParamSpace() map[string]ParamRange {
	return map[string]ParamRange{
		"n_estimators": {
			Type: "int",
			Min:  10,
			Max:  500,
		},
		"learning_rate": {
			Type:  "float",
			Min:   0.01,
			Max:   0.3,
			Scale: "log",
		},
		"max_depth": {
			Type: "int",
			Min:  3,
			Max:  12,
		},
	}
}

// XGBoostModel implements gradient boosting
type XGBoostModel struct {
	nEstimators  int
	learningRate float64
	maxDepth     int
	trees        []*DecisionTree
	basePred     float64
}

func (m *XGBoostModel) Fit(X [][]float64, y []float64) error {
	m.basePred = mean(y)
	residuals := make([]float64, len(y))
	for i := range y {
		residuals[i] = y[i] - m.basePred
	}

	m.trees = make([]*DecisionTree, m.nEstimators)

	for i := 0; i < m.nEstimators; i++ {
		tree := &DecisionTree{
			maxDepth:   m.maxDepth,
			minSamples: 2,
		}

		if err := tree.Fit(X, residuals); err != nil {
			return err
		}

		m.trees[i] = tree

		// Update residuals
		for j := range X {
			pred, _ := tree.PredictOne(X[j])
			residuals[j] -= m.learningRate * pred
		}
	}

	return nil
}

func (m *XGBoostModel) Predict(X [][]float64) ([]float64, error) {
	predictions := make([]float64, len(X))

	for i, x := range X {
		pred := m.basePred
		for _, tree := range m.trees {
			treePred, _ := tree.PredictOne(x)
			pred += m.learningRate * treePred
		}
		predictions[i] = pred
	}

	return predictions, nil
}

func (m *XGBoostModel) GetParams() map[string]interface{} {
	return map[string]interface{}{
		"n_estimators":  m.nEstimators,
		"learning_rate": m.learningRate,
		"max_depth":     m.maxDepth,
	}
}

// NeuralNetBuilder builds neural network models
type NeuralNetBuilder struct{}

func (b *NeuralNetBuilder) Build(params map[string]interface{}) Model {
	return &NeuralNetModel{
		hiddenSize:   getIntParam(params, "hidden_size", 64),
		learningRate: getFloatParam(params, "learning_rate", 0.01),
		epochs:       getIntParam(params, "epochs", 100),
	}
}

func (b *NeuralNetBuilder) DefaultParams() map[string]interface{} {
	return map[string]interface{}{
		"hidden_size":   64,
		"learning_rate": 0.01,
		"epochs":        100,
	}
}

func (b *NeuralNetBuilder) ParamSpace() map[string]ParamRange {
	return map[string]ParamRange{
		"hidden_size": {
			Type: "int",
			Min:  16,
			Max:  256,
		},
		"learning_rate": {
			Type:  "float",
			Min:   0.001,
			Max:   0.1,
			Scale: "log",
		},
		"epochs": {
			Type: "int",
			Min:  50,
			Max:  500,
		},
	}
}

// NeuralNetModel implements a simple neural network
type NeuralNetModel struct {
	hiddenSize   int
	learningRate float64
	epochs       int
	w1           [][]float64
	b1           []float64
	w2           []float64
	b2           float64
}

func (m *NeuralNetModel) Fit(X [][]float64, y []float64) error {
	if len(X) == 0 {
		return fmt.Errorf("empty dataset")
	}

	inputSize := len(X[0])

	// Initialize weights
	m.w1 = make([][]float64, inputSize)
	for i := range m.w1 {
		m.w1[i] = make([]float64, m.hiddenSize)
		for j := range m.w1[i] {
			m.w1[i][j] = rand.NormFloat64() * 0.1
		}
	}

	m.b1 = make([]float64, m.hiddenSize)
	m.w2 = make([]float64, m.hiddenSize)
	for i := range m.w2 {
		m.w2[i] = rand.NormFloat64() * 0.1
	}
	m.b2 = 0

	// Training loop
	for epoch := 0; epoch < m.epochs; epoch++ {
		for i := range X {
			m.trainStep(X[i], y[i])
		}
	}

	return nil
}

func (m *NeuralNetModel) trainStep(x []float64, y float64) {
	// Forward pass
	hidden := make([]float64, m.hiddenSize)
	for j := 0; j < m.hiddenSize; j++ {
		sum := m.b1[j]
		for i := range x {
			sum += x[i] * m.w1[i][j]
		}
		hidden[j] = relu(sum)
	}

	output := m.b2
	for j := range hidden {
		output += hidden[j] * m.w2[j]
	}

	// Backward pass
	dOutput := 2 * (output - y)

	for j := range m.w2 {
		m.w2[j] -= m.learningRate * dOutput * hidden[j]
	}
	m.b2 -= m.learningRate * dOutput

	for i := range x {
		for j := 0; j < m.hiddenSize; j++ {
			dHidden := dOutput * m.w2[j]
			if hidden[j] > 0 { // ReLU derivative
				m.w1[i][j] -= m.learningRate * dHidden * x[i]
				m.b1[j] -= m.learningRate * dHidden
			}
		}
	}
}

func (m *NeuralNetModel) Predict(X [][]float64) ([]float64, error) {
	predictions := make([]float64, len(X))

	for i, x := range X {
		hidden := make([]float64, m.hiddenSize)
		for j := 0; j < m.hiddenSize; j++ {
			sum := m.b1[j]
			for k := range x {
				sum += x[k] * m.w1[k][j]
			}
			hidden[j] = relu(sum)
		}

		output := m.b2
		for j := range hidden {
			output += hidden[j] * m.w2[j]
		}

		predictions[i] = output
	}

	return predictions, nil
}

func (m *NeuralNetModel) GetParams() map[string]interface{} {
	return map[string]interface{}{
		"hidden_size":   m.hiddenSize,
		"learning_rate": m.learningRate,
		"epochs":        m.epochs,
	}
}

// LinearModelBuilder builds linear regression models
type LinearModelBuilder struct{}

func (b *LinearModelBuilder) Build(params map[string]interface{}) Model {
	return &LinearModel{
		learningRate: getFloatParam(params, "learning_rate", 0.01),
		iterations:   getIntParam(params, "iterations", 1000),
	}
}

func (b *LinearModelBuilder) DefaultParams() map[string]interface{} {
	return map[string]interface{}{
		"learning_rate": 0.01,
		"iterations":    1000,
	}
}

func (b *LinearModelBuilder) ParamSpace() map[string]ParamRange {
	return map[string]ParamRange{
		"learning_rate": {
			Type:  "float",
			Min:   0.001,
			Max:   0.1,
			Scale: "log",
		},
		"iterations": {
			Type: "int",
			Min:  100,
			Max:  5000,
		},
	}
}

// LinearModel implements linear regression
type LinearModel struct {
	learningRate float64
	iterations   int
	weights      []float64
	bias         float64
}

func (m *LinearModel) Fit(X [][]float64, y []float64) error {
	if len(X) == 0 {
		return fmt.Errorf("empty dataset")
	}

	m.weights = make([]float64, len(X[0]))
	m.bias = 0

	for iter := 0; iter < m.iterations; iter++ {
		gradWeights := make([]float64, len(m.weights))
		gradBias := 0.0

		for i := range X {
			pred := m.predictOne(X[i])
			diff := pred - y[i]

			for j := range X[i] {
				gradWeights[j] += diff * X[i][j]
			}
			gradBias += diff
		}

		for j := range m.weights {
			m.weights[j] -= m.learningRate * gradWeights[j] / float64(len(X))
		}
		m.bias -= m.learningRate * gradBias / float64(len(X))
	}

	return nil
}

func (m *LinearModel) Predict(X [][]float64) ([]float64, error) {
	predictions := make([]float64, len(X))
	for i, x := range X {
		predictions[i] = m.predictOne(x)
	}
	return predictions, nil
}

func (m *LinearModel) predictOne(x []float64) float64 {
	pred := m.bias
	for i, w := range m.weights {
		pred += w * x[i]
	}
	return pred
}

func (m *LinearModel) GetParams() map[string]interface{} {
	return map[string]interface{}{
		"learning_rate": m.learningRate,
		"iterations":    m.iterations,
	}
}

// Helper functions

func getIntParam(params map[string]interface{}, key string, defaultVal int) int {
	if val, ok := params[key]; ok {
		if intVal, ok := val.(int); ok {
			return intVal
		}
	}
	return defaultVal
}

func getFloatParam(params map[string]interface{}, key string, defaultVal float64) float64 {
	if val, ok := params[key]; ok {
		if floatVal, ok := val.(float64); ok {
			return floatVal
		}
	}
	return defaultVal
}

func mean(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}

func variance(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	m := mean(values)
	sum := 0.0
	for _, v := range values {
		diff := v - m
		sum += diff * diff
	}
	return sum / float64(len(values))
}

func relu(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}
