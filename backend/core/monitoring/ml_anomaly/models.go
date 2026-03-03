package ml_anomaly

import (
	"fmt"
	"math"
	"sort"
	"time"

	"github.com/google/uuid"
)

// StatisticalModel implements statistical anomaly detection
type StatisticalModel struct {
	config     *DetectorConfig
	mean       float64
	stdDev     float64
	median     float64
	q1         float64
	q3         float64
	iqr        float64
	trained    bool
	parameters map[string]interface{}
}

// NewStatisticalModel creates a new statistical anomaly detection model
func NewStatisticalModel(config *DetectorConfig) *StatisticalModel {
	return &StatisticalModel{
		config:     config,
		parameters: make(map[string]interface{}),
	}
}

// Train trains the statistical model
func (m *StatisticalModel) Train(data []MetricDataPoint) error {
	if len(data) < 10 {
		return fmt.Errorf("insufficient data points for training: need at least 10, got %d", len(data))
	}

	values := make([]float64, len(data))
	for i, point := range data {
		values[i] = point.Value
	}

	// Calculate statistical measures
	m.mean = calculateMean(values)
	m.stdDev = calculateStdDev(values, m.mean)
	
	sortedValues := make([]float64, len(values))
	copy(sortedValues, values)
	sort.Float64s(sortedValues)
	
	m.median = calculateMedian(sortedValues)
	m.q1 = calculatePercentile(sortedValues, 0.25)
	m.q3 = calculatePercentile(sortedValues, 0.75)
	m.iqr = m.q3 - m.q1

	// Store parameters
	m.parameters = map[string]interface{}{
		"mean":   m.mean,
		"stddev": m.stdDev,
		"median": m.median,
		"q1":     m.q1,
		"q3":     m.q3,
		"iqr":    m.iqr,
	}

	m.trained = true
	return nil
}

// Detect detects anomalies using statistical methods
func (m *StatisticalModel) Detect(data []MetricDataPoint) ([]Anomaly, error) {
	if !m.trained {
		return nil, fmt.Errorf("model not trained")
	}

	var anomalies []Anomaly
	threshold := m.config.AnomalyThreshold

	for _, point := range data {
		// Z-score method
		zScore := math.Abs(point.Value-m.mean) / m.stdDev
		
		// IQR method
		iqrLowerBound := m.q1 - 1.5*m.iqr
		iqrUpperBound := m.q3 + 1.5*m.iqr
		iqrOutlier := point.Value < iqrLowerBound || point.Value > iqrUpperBound

		// Modified Z-score method
		mad := calculateMAD([]float64{point.Value}, m.median)
		modifiedZScore := 0.6745 * (point.Value - m.median) / mad

		isAnomalous := false
		confidence := 0.0
		method := ""

		// Determine if anomalous based on multiple methods
		if zScore > threshold {
			isAnomalous = true
			confidence = math.Min(zScore/threshold, 1.0)
			method = "z-score"
		} else if iqrOutlier {
			isAnomalous = true
			confidence = 0.8
			method = "iqr"
		} else if math.Abs(modifiedZScore) > threshold {
			isAnomalous = true
			confidence = math.Min(math.Abs(modifiedZScore)/threshold, 1.0)
			method = "modified-z-score"
		}

		if isAnomalous {
			severity := m.calculateSeverity(point.Value, m.mean, zScore)
			
			anomaly := Anomaly{
				ID:            uuid.New().String(),
				Timestamp:     point.Timestamp,
				Value:         point.Value,
				ExpectedValue: m.mean,
				Severity:      severity,
				Confidence:    confidence,
				Description:   fmt.Sprintf("Statistical anomaly detected using %s method (score: %.2f)", method, zScore),
				Labels:        point.Labels,
				Context: map[string]interface{}{
					"z_score":          zScore,
					"modified_z_score": modifiedZScore,
					"iqr_outlier":      iqrOutlier,
					"method":           method,
					"mean":             m.mean,
					"stddev":           m.stdDev,
				},
			}

			anomalies = append(anomalies, anomaly)
		}
	}

	return anomalies, nil
}

// Predict provides simple linear prediction
func (m *StatisticalModel) Predict(data []MetricDataPoint, horizon time.Duration) ([]Prediction, error) {
	if !m.trained || len(data) < 2 {
		return nil, fmt.Errorf("insufficient data for prediction")
	}

	// Simple linear trend calculation
	values := make([]float64, len(data))
	timestamps := make([]time.Time, len(data))
	
	for i, point := range data {
		values[i] = point.Value
		timestamps[i] = point.Timestamp
	}

	// Calculate linear regression
	slope, intercept := calculateLinearRegression(timestamps, values)
	
	var predictions []Prediction
	lastTime := timestamps[len(timestamps)-1]
	
	// Generate predictions
	steps := int(horizon.Minutes() / 5) // Predict every 5 minutes
	if steps < 1 {
		steps = 1
	}
	
	stepDuration := horizon / time.Duration(steps)
	
	for i := 1; i <= steps; i++ {
		predTime := lastTime.Add(time.Duration(i) * stepDuration)
		predValue := slope*float64(predTime.Unix()) + intercept
		
		// Calculate confidence bounds
		upperBound := predValue + 2*m.stdDev
		lowerBound := predValue - 2*m.stdDev
		
		prediction := Prediction{
			Timestamp:  predTime,
			Value:      predValue,
			Confidence: math.Max(0.1, 1.0-float64(i)*0.1), // Decreasing confidence over time
			UpperBound: upperBound,
			LowerBound: lowerBound,
			Context: map[string]interface{}{
				"slope":     slope,
				"intercept": intercept,
				"step":      i,
			},
		}
		
		predictions = append(predictions, prediction)
	}

	return predictions, nil
}

// GetName returns the model name
func (m *StatisticalModel) GetName() string {
	return "statistical"
}

// GetParameters returns model parameters
func (m *StatisticalModel) GetParameters() map[string]interface{} {
	return m.parameters
}

// SetParameters sets model parameters
func (m *StatisticalModel) SetParameters(params map[string]interface{}) error {
	m.parameters = params
	
	if mean, ok := params["mean"].(float64); ok {
		m.mean = mean
	}
	if stddev, ok := params["stddev"].(float64); ok {
		m.stdDev = stddev
	}
	if median, ok := params["median"].(float64); ok {
		m.median = median
	}
	if q1, ok := params["q1"].(float64); ok {
		m.q1 = q1
	}
	if q3, ok := params["q3"].(float64); ok {
		m.q3 = q3
	}
	if iqr, ok := params["iqr"].(float64); ok {
		m.iqr = iqr
	}
	
	m.trained = true
	return nil
}

// IsReady returns whether the model is ready for use
func (m *StatisticalModel) IsReady() bool {
	return m.trained
}

func (m *StatisticalModel) calculateSeverity(value, expected, zScore float64) AnomalySeverity {
	if zScore >= 4.0 {
		return SeverityCritical
	} else if zScore >= 3.0 {
		return SeverityHigh
	} else if zScore >= 2.0 {
		return SeverityMedium
	}
	return SeverityLow
}

// IsolationForestModel implements isolation forest anomaly detection
type IsolationForestModel struct {
	config     *DetectorConfig
	trees      []IsolationTree
	trained    bool
	parameters map[string]interface{}
}

// IsolationTree represents a single tree in the isolation forest
type IsolationTree struct {
	Root      *IsolationNode
	MaxDepth  int
	SampleSize int
}

// IsolationNode represents a node in an isolation tree
type IsolationNode struct {
	SplitAttribute string
	SplitValue     float64
	Left           *IsolationNode
	Right          *IsolationNode
	Size           int
	IsLeaf         bool
}

// NewIsolationForestModel creates a new isolation forest model
func NewIsolationForestModel(config *DetectorConfig) *IsolationForestModel {
	return &IsolationForestModel{
		config:     config,
		parameters: make(map[string]interface{}),
	}
}

// Train trains the isolation forest model
func (m *IsolationForestModel) Train(data []MetricDataPoint) error {
	if len(data) < 50 {
		return fmt.Errorf("insufficient data points for isolation forest: need at least 50, got %d", len(data))
	}

	// Convert to feature matrix
	features := m.extractFeatures(data)
	
	// Build isolation trees
	numTrees := 100
	sampleSize := int(math.Min(256, float64(len(features))))
	
	m.trees = make([]IsolationTree, numTrees)
	for i := 0; i < numTrees; i++ {
		// Sample data for this tree
		sample := m.sampleData(features, sampleSize)
		
		// Build tree
		maxDepth := int(math.Ceil(math.Log2(float64(sampleSize))))
		root := m.buildTree(sample, 0, maxDepth)
		
		m.trees[i] = IsolationTree{
			Root:      root,
			MaxDepth:  maxDepth,
			SampleSize: sampleSize,
		}
	}

	m.parameters = map[string]interface{}{
		"num_trees":   numTrees,
		"sample_size": sampleSize,
	}

	m.trained = true
	return nil
}

// Detect detects anomalies using isolation forest
func (m *IsolationForestModel) Detect(data []MetricDataPoint) ([]Anomaly, error) {
	if !m.trained {
		return nil, fmt.Errorf("model not trained")
	}

	var anomalies []Anomaly
	features := m.extractFeatures(data)

	for i, point := range data {
		if i >= len(features) {
			break
		}

		// Calculate anomaly score
		score := m.calculateAnomalyScore(features[i])
		
		// Threshold for anomaly (typically 0.6 or higher)
		if score > 0.6 {
			severity := m.calculateSeverityFromScore(score)
			
			anomaly := Anomaly{
				ID:            uuid.New().String(),
				Timestamp:     point.Timestamp,
				Value:         point.Value,
				ExpectedValue: calculateMean(m.extractValues(data)),
				Severity:      severity,
				Confidence:    score,
				Description:   fmt.Sprintf("Isolation forest anomaly detected (score: %.3f)", score),
				Labels:        point.Labels,
				Context: map[string]interface{}{
					"isolation_score": score,
					"features":       features[i],
				},
			}

			anomalies = append(anomalies, anomaly)
		}
	}

	return anomalies, nil
}

// Predict is not implemented for isolation forest
func (m *IsolationForestModel) Predict(data []MetricDataPoint, horizon time.Duration) ([]Prediction, error) {
	return nil, fmt.Errorf("prediction not supported by isolation forest model")
}

// GetName returns the model name
func (m *IsolationForestModel) GetName() string {
	return "isolation_forest"
}

// GetParameters returns model parameters
func (m *IsolationForestModel) GetParameters() map[string]interface{} {
	return m.parameters
}

// SetParameters sets model parameters
func (m *IsolationForestModel) SetParameters(params map[string]interface{}) error {
	m.parameters = params
	return nil
}

// IsReady returns whether the model is ready for use
func (m *IsolationForestModel) IsReady() bool {
	return m.trained
}

// Helper methods for IsolationForestModel

func (m *IsolationForestModel) extractFeatures(data []MetricDataPoint) []map[string]float64 {
	features := make([]map[string]float64, len(data))
	
	for i, point := range data {
		feature := map[string]float64{
			"value":     point.Value,
			"timestamp": float64(point.Timestamp.Unix()),
		}
		
		// Add rolling averages if we have enough data
		if i >= 5 {
			windowSum := 0.0
			for j := i - 4; j <= i; j++ {
				windowSum += data[j].Value
			}
			feature["rolling_avg_5"] = windowSum / 5.0
		}
		
		// Add hour of day as feature
		feature["hour"] = float64(point.Timestamp.Hour())
		feature["day_of_week"] = float64(point.Timestamp.Weekday())
		
		features[i] = feature
	}
	
	return features
}

func (m *IsolationForestModel) extractValues(data []MetricDataPoint) []float64 {
	values := make([]float64, len(data))
	for i, point := range data {
		values[i] = point.Value
	}
	return values
}

func (m *IsolationForestModel) sampleData(features []map[string]float64, size int) []map[string]float64 {
	if len(features) <= size {
		return features
	}
	
	// Simple random sampling
	sample := make([]map[string]float64, size)
	for i := 0; i < size; i++ {
		idx := i * len(features) / size
		sample[i] = features[idx]
	}
	
	return sample
}

func (m *IsolationForestModel) buildTree(data []map[string]float64, depth, maxDepth int) *IsolationNode {
	if depth >= maxDepth || len(data) <= 1 {
		return &IsolationNode{
			Size:   len(data),
			IsLeaf: true,
		}
	}
	
	// Randomly select attribute and split value
	if len(data) == 0 {
		return &IsolationNode{IsLeaf: true}
	}
	
	// Get all attributes
	var attributes []string
	for attr := range data[0] {
		attributes = append(attributes, attr)
	}
	
	if len(attributes) == 0 {
		return &IsolationNode{Size: len(data), IsLeaf: true}
	}
	
	// Random attribute selection
	attr := attributes[depth%len(attributes)]
	
	// Find min/max values for this attribute
	minVal := data[0][attr]
	maxVal := data[0][attr]
	for _, point := range data {
		if val, exists := point[attr]; exists {
			if val < minVal {
				minVal = val
			}
			if val > maxVal {
				maxVal = val
			}
		}
	}
	
	if minVal == maxVal {
		return &IsolationNode{Size: len(data), IsLeaf: true}
	}
	
	// Random split value
	splitValue := minVal + (maxVal-minVal)*0.5 // Simplified to midpoint
	
	// Split data
	var leftData, rightData []map[string]float64
	for _, point := range data {
		if val, exists := point[attr]; exists && val < splitValue {
			leftData = append(leftData, point)
		} else {
			rightData = append(rightData, point)
		}
	}
	
	// Recursively build subtrees
	left := m.buildTree(leftData, depth+1, maxDepth)
	right := m.buildTree(rightData, depth+1, maxDepth)
	
	return &IsolationNode{
		SplitAttribute: attr,
		SplitValue:     splitValue,
		Left:           left,
		Right:          right,
		Size:           len(data),
		IsLeaf:         false,
	}
}

func (m *IsolationForestModel) calculateAnomalyScore(features map[string]float64) float64 {
	totalPathLength := 0.0
	
	for _, tree := range m.trees {
		pathLength := m.getPathLength(features, tree.Root, 0)
		totalPathLength += pathLength
	}
	
	avgPathLength := totalPathLength / float64(len(m.trees))
	expectedPathLength := 2.0 * (math.Log(float64(m.trees[0].SampleSize-1)) + 0.5772156649) - (2.0 * float64(m.trees[0].SampleSize-1) / float64(m.trees[0].SampleSize))
	
	// Anomaly score
	score := math.Pow(2, -avgPathLength/expectedPathLength)
	return score
}

func (m *IsolationForestModel) getPathLength(features map[string]float64, node *IsolationNode, depth int) float64 {
	if node.IsLeaf {
		return float64(depth) + m.calculateC(node.Size)
	}
	
	if val, exists := features[node.SplitAttribute]; exists && val < node.SplitValue {
		return m.getPathLength(features, node.Left, depth+1)
	}
	
	return m.getPathLength(features, node.Right, depth+1)
}

func (m *IsolationForestModel) calculateC(n int) float64 {
	if n <= 1 {
		return 0
	}
	return 2.0*(math.Log(float64(n-1))+0.5772156649) - 2.0*float64(n-1)/float64(n)
}

func (m *IsolationForestModel) calculateSeverityFromScore(score float64) AnomalySeverity {
	if score >= 0.8 {
		return SeverityCritical
	} else if score >= 0.7 {
		return SeverityHigh
	} else if score >= 0.6 {
		return SeverityMedium
	}
	return SeverityLow
}

// Placeholder models for LSTM and Seasonal Decomposition
// These would require more complex implementations

// LSTMModel placeholder
type LSTMModel struct {
	config     *DetectorConfig
	trained    bool
	parameters map[string]interface{}
}

func NewLSTMModel(config *DetectorConfig) *LSTMModel {
	return &LSTMModel{
		config:     config,
		parameters: make(map[string]interface{}),
	}
}

func (m *LSTMModel) Train(data []MetricDataPoint) error {
	// LSTM training would be implemented here
	m.trained = true
	return nil
}

func (m *LSTMModel) Detect(data []MetricDataPoint) ([]Anomaly, error) {
	if !m.trained {
		return nil, fmt.Errorf("model not trained")
	}
	// LSTM anomaly detection would be implemented here
	return []Anomaly{}, nil
}

func (m *LSTMModel) Predict(data []MetricDataPoint, horizon time.Duration) ([]Prediction, error) {
	// LSTM prediction would be implemented here
	return []Prediction{}, nil
}

func (m *LSTMModel) GetName() string {
	return "lstm"
}

func (m *LSTMModel) GetParameters() map[string]interface{} {
	return m.parameters
}

func (m *LSTMModel) SetParameters(params map[string]interface{}) error {
	m.parameters = params
	return nil
}

func (m *LSTMModel) IsReady() bool {
	return m.trained
}

// SeasonalDecompositionModel placeholder
type SeasonalDecompositionModel struct {
	config     *DetectorConfig
	trained    bool
	parameters map[string]interface{}
}

func NewSeasonalDecompositionModel(config *DetectorConfig) *SeasonalDecompositionModel {
	return &SeasonalDecompositionModel{
		config:     config,
		parameters: make(map[string]interface{}),
	}
}

func (m *SeasonalDecompositionModel) Train(data []MetricDataPoint) error {
	// Seasonal decomposition training would be implemented here
	m.trained = true
	return nil
}

func (m *SeasonalDecompositionModel) Detect(data []MetricDataPoint) ([]Anomaly, error) {
	if !m.trained {
		return nil, fmt.Errorf("model not trained")
	}
	// Seasonal decomposition anomaly detection would be implemented here
	return []Anomaly{}, nil
}

func (m *SeasonalDecompositionModel) Predict(data []MetricDataPoint, horizon time.Duration) ([]Prediction, error) {
	// Seasonal decomposition prediction would be implemented here
	return []Prediction{}, nil
}

func (m *SeasonalDecompositionModel) GetName() string {
	return "seasonal_decomposition"
}

func (m *SeasonalDecompositionModel) GetParameters() map[string]interface{} {
	return m.parameters
}

func (m *SeasonalDecompositionModel) SetParameters(params map[string]interface{}) error {
	m.parameters = params
	return nil
}

func (m *SeasonalDecompositionModel) IsReady() bool {
	return m.trained
}

// Statistical utility functions

func calculateMean(values []float64) float64 {
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}

func calculateStdDev(values []float64, mean float64) float64 {
	sumSquares := 0.0
	for _, v := range values {
		diff := v - mean
		sumSquares += diff * diff
	}
	variance := sumSquares / float64(len(values))
	return math.Sqrt(variance)
}

func calculateMedian(sortedValues []float64) float64 {
	n := len(sortedValues)
	if n%2 == 0 {
		return (sortedValues[n/2-1] + sortedValues[n/2]) / 2.0
	}
	return sortedValues[n/2]
}

func calculatePercentile(sortedValues []float64, percentile float64) float64 {
	n := len(sortedValues)
	index := percentile * float64(n-1)
	
	if index == float64(int(index)) {
		return sortedValues[int(index)]
	}
	
	lower := int(math.Floor(index))
	upper := int(math.Ceil(index))
	weight := index - float64(lower)
	
	return sortedValues[lower]*(1-weight) + sortedValues[upper]*weight
}

func calculateMAD(values []float64, median float64) float64 {
	deviations := make([]float64, len(values))
	for i, v := range values {
		deviations[i] = math.Abs(v - median)
	}
	sort.Float64s(deviations)
	return calculateMedian(deviations)
}

func calculateLinearRegression(times []time.Time, values []float64) (slope, intercept float64) {
	n := float64(len(times))
	if n < 2 {
		return 0, values[0]
	}
	
	// Convert times to numeric values
	x := make([]float64, len(times))
	for i, t := range times {
		x[i] = float64(t.Unix())
	}
	
	// Calculate means
	meanX := calculateMean(x)
	meanY := calculateMean(values)
	
	// Calculate slope and intercept
	numerator := 0.0
	denominator := 0.0
	
	for i := 0; i < len(x); i++ {
		numerator += (x[i] - meanX) * (values[i] - meanY)
		denominator += (x[i] - meanX) * (x[i] - meanX)
	}
	
	if denominator == 0 {
		return 0, meanY
	}
	
	slope = numerator / denominator
	intercept = meanY - slope*meanX
	
	return slope, intercept
}