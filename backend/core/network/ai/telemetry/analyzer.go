package telemetry

import (
	"context"
	"fmt"
	"math"
	"sort"
	"sync"
	"time"
)

// AnomalyType represents different types of network anomalies
type AnomalyType int

const (
	AnomalyBandwidth AnomalyType = iota
	AnomalyLatency
	AnomalyPacketLoss
	AnomalyJitter
	AnomalyErrorRate
	AnomalyTrafficPattern
	AnomalyProtocolDistribution
)

// NetworkMetric represents a network metric data point
type NetworkMetric struct {
	Timestamp     time.Time
	Component     string
	MetricName    string
	Value         float64
	Tags          map[string]string
}

// Anomaly represents a detected anomaly
type Anomaly struct {
	ID            string
	Type          AnomalyType
	Component     string
	DetectedAt    time.Time
	Score         float64 // Anomaly score (0-1)
	Severity      string  // low, medium, high, critical
	MetricValue   float64
	ExpectedValue float64
	Deviation     float64 // Standard deviations from normal
	Description   string
}

// TelemetryAnalyzer performs AI-based network telemetry analysis
type TelemetryAnalyzer struct {
	mu sync.RWMutex

	// Anomaly detection models
	isolationForest *IsolationForest
	autoencoder     *Autoencoder
	statisticalDetector *StatisticalDetector

	// Baseline learning
	baseline        *BaselineModel
	baselineWindow  time.Duration
	baselineHistory map[string][]float64

	// Alert management
	alertThreshold  float64 // 3-sigma default
	alertCooldown   time.Duration
	lastAlerts      map[string]time.Time

	// Performance metrics
	anomaliesDetected int64
	falsePositives    int64
	detectionLatency  time.Duration
	analysisCount     int64

	// Historical data
	metricHistory   map[string]*CircularBuffer
	anomalyHistory  []Anomaly
	maxHistorySize  int
}

// IsolationForest for outlier detection
type IsolationForest struct {
	trees       []*IsolationTree
	numTrees    int
	sampleSize  int
	maxDepth    int
}

// IsolationTree represents a single tree in the forest
type IsolationTree struct {
	root *IsolationNode
}

// IsolationNode in isolation tree
type IsolationNode struct {
	isLeaf      bool
	left        *IsolationNode
	right       *IsolationNode
	feature     int
	splitValue  float64
	size        int
	depth       int
}

// Autoencoder for baseline learning
type Autoencoder struct {
	encoder     *NeuralLayer
	decoder     *NeuralLayer
	bottleneck  int
	inputSize   int
	threshold   float64
}

// NeuralLayer represents a layer in the autoencoder
type NeuralLayer struct {
	weights [][]float64
	bias    []float64
	activation string
}

// StatisticalDetector for statistical anomaly detection
type StatisticalDetector struct {
	mean           map[string]float64
	stdDev         map[string]float64
	movingAverage  map[string]*ExponentialMovingAverage
	seasonalModel  *SeasonalModel
}

// ExponentialMovingAverage for tracking trends
type ExponentialMovingAverage struct {
	value float64
	alpha float64
}

// SeasonalModel for detecting seasonal patterns
type SeasonalModel struct {
	hourlyPatterns  [24]float64
	dailyPatterns   [7]float64
	weeklyPatterns  [52]float64
	lastUpdate      time.Time
}

// BaselineModel learns normal behavior
type BaselineModel struct {
	profiles      map[string]*MetricProfile
	learningRate  float64
	adaptiveMode  bool
}

// MetricProfile represents normal behavior for a metric
type MetricProfile struct {
	Mean       float64
	StdDev     float64
	Min        float64
	Max        float64
	Percentiles map[int]float64 // 5, 25, 50, 75, 95
	UpdateCount int64
}

// CircularBuffer for efficient history management
type CircularBuffer struct {
	data     []float64
	capacity int
	head     int
	size     int
}

// NewTelemetryAnalyzer creates a new telemetry analyzer
func NewTelemetryAnalyzer() *TelemetryAnalyzer {
	return &TelemetryAnalyzer{
		alertThreshold:  3.0, // 3-sigma
		alertCooldown:   5 * time.Minute,
		lastAlerts:      make(map[string]time.Time),
		baselineWindow:  24 * time.Hour,
		baselineHistory: make(map[string][]float64),
		metricHistory:   make(map[string]*CircularBuffer),
		anomalyHistory:  make([]Anomaly, 0, 1000),
		maxHistorySize:  10000,
	}
}

// Initialize initializes the telemetry analyzer
func (t *TelemetryAnalyzer) Initialize(ctx context.Context) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	// Initialize Isolation Forest
	t.isolationForest = &IsolationForest{
		numTrees:   100,
		sampleSize: 256,
		maxDepth:   10,
	}
	t.isolationForest.build()

	// Initialize Autoencoder
	t.autoencoder = &Autoencoder{
		inputSize:  10, // Number of metrics
		bottleneck: 3,  // Compressed representation
		threshold:  0.1, // Reconstruction error threshold
	}
	t.autoencoder.initialize()

	// Initialize Statistical Detector
	t.statisticalDetector = &StatisticalDetector{
		mean:          make(map[string]float64),
		stdDev:        make(map[string]float64),
		movingAverage: make(map[string]*ExponentialMovingAverage),
		seasonalModel: &SeasonalModel{
			lastUpdate: time.Now(),
		},
	}

	// Initialize Baseline Model
	t.baseline = &BaselineModel{
		profiles:     make(map[string]*MetricProfile),
		learningRate: 0.01,
		adaptiveMode: true,
	}

	return nil
}

// AnalyzeMetrics analyzes network metrics for anomalies
func (t *TelemetryAnalyzer) AnalyzeMetrics(ctx context.Context, metrics []NetworkMetric) ([]Anomaly, error) {
	start := time.Now()
	defer func() {
		t.updateAnalysisMetrics(time.Since(start))
	}()

	t.mu.Lock()
	defer t.mu.Unlock()

	var anomalies []Anomaly

	// Update history
	for _, metric := range metrics {
		t.updateHistory(metric)
	}

	// Method 1: Isolation Forest
	if isoAnomalies := t.detectWithIsolationForest(metrics); len(isoAnomalies) > 0 {
		anomalies = append(anomalies, isoAnomalies...)
	}

	// Method 2: Autoencoder
	if autoAnomalies := t.detectWithAutoencoder(metrics); len(autoAnomalies) > 0 {
		anomalies = append(anomalies, autoAnomalies...)
	}

	// Method 3: Statistical Detection
	if statAnomalies := t.detectWithStatistics(metrics); len(statAnomalies) > 0 {
		anomalies = append(anomalies, statAnomalies...)
	}

	// Correlate and deduplicate anomalies
	anomalies = t.correlateAnomalies(anomalies)

	// Filter by alert threshold
	filteredAnomalies := t.filterAnomalies(anomalies)

	// Update baseline with normal data
	t.updateBaseline(metrics, filteredAnomalies)

	// Store anomaly history
	t.anomalyHistory = append(t.anomalyHistory, filteredAnomalies...)
	if len(t.anomalyHistory) > t.maxHistorySize {
		t.anomalyHistory = t.anomalyHistory[len(t.anomalyHistory)-t.maxHistorySize:]
	}

	// Update detection count
	t.anomaliesDetected += int64(len(filteredAnomalies))

	// Check if detection latency meets <1s requirement
	if time.Since(start) > time.Second {
		return filteredAnomalies, fmt.Errorf("detection took too long: %v", time.Since(start))
	}

	return filteredAnomalies, nil
}

// detectWithIsolationForest uses Isolation Forest for anomaly detection
func (t *TelemetryAnalyzer) detectWithIsolationForest(metrics []NetworkMetric) []Anomaly {
	var anomalies []Anomaly

	// Convert metrics to feature vectors
	features := t.metricsToFeatures(metrics)

	for i, feature := range features {
		score := t.isolationForest.anomalyScore(feature)

		if score > 0.7 { // High anomaly score
			anomaly := Anomaly{
				ID:          fmt.Sprintf("iso-%d", time.Now().UnixNano()),
				Type:        t.inferAnomalyType(metrics[i].MetricName),
				Component:   metrics[i].Component,
				DetectedAt:  time.Now(),
				Score:       score,
				Severity:    t.calculateSeverity(score),
				MetricValue: metrics[i].Value,
				Description: fmt.Sprintf("Isolation Forest detected anomaly in %s", metrics[i].MetricName),
			}
			anomalies = append(anomalies, anomaly)
		}
	}

	return anomalies
}

// detectWithAutoencoder uses autoencoder for anomaly detection
func (t *TelemetryAnalyzer) detectWithAutoencoder(metrics []NetworkMetric) []Anomaly {
	var anomalies []Anomaly

	// Group metrics by component
	componentMetrics := t.groupByComponent(metrics)

	for component, compMetrics := range componentMetrics {
		// Convert to input vector
		input := t.metricsToVector(compMetrics)

		// Get reconstruction error
		reconstructionError := t.autoencoder.getReconstructionError(input)

		if reconstructionError > t.autoencoder.threshold {
			anomaly := Anomaly{
				ID:          fmt.Sprintf("auto-%d", time.Now().UnixNano()),
				Type:        AnomalyTrafficPattern,
				Component:   component,
				DetectedAt:  time.Now(),
				Score:       reconstructionError / t.autoencoder.threshold,
				Severity:    t.calculateSeverity(reconstructionError / t.autoencoder.threshold),
				Description: "Autoencoder detected unusual pattern",
			}
			anomalies = append(anomalies, anomaly)
		}
	}

	return anomalies
}

// detectWithStatistics uses statistical methods for anomaly detection
func (t *TelemetryAnalyzer) detectWithStatistics(metrics []NetworkMetric) []Anomaly {
	var anomalies []Anomaly

	for _, metric := range metrics {
		key := fmt.Sprintf("%s:%s", metric.Component, metric.MetricName)

		// Update moving average
		if ema, exists := t.statisticalDetector.movingAverage[key]; exists {
			ema.update(metric.Value)

			// Check deviation from moving average
			deviation := math.Abs(metric.Value - ema.value)
			stdDev := t.statisticalDetector.stdDev[key]

			if stdDev > 0 && deviation > t.alertThreshold*stdDev {
				anomaly := Anomaly{
					ID:            fmt.Sprintf("stat-%d", time.Now().UnixNano()),
					Type:          t.inferAnomalyType(metric.MetricName),
					Component:     metric.Component,
					DetectedAt:    time.Now(),
					Score:         deviation / (t.alertThreshold * stdDev),
					Severity:      t.calculateSeverity(deviation / (t.alertThreshold * stdDev)),
					MetricValue:   metric.Value,
					ExpectedValue: ema.value,
					Deviation:     deviation / stdDev,
					Description:   fmt.Sprintf("%s deviates %.1f sigma from normal", metric.MetricName, deviation/stdDev),
				}

				// Check cooldown
				if t.shouldAlert(key) {
					anomalies = append(anomalies, anomaly)
					t.lastAlerts[key] = time.Now()
				}
			}
		} else {
			// Initialize moving average
			t.statisticalDetector.movingAverage[key] = &ExponentialMovingAverage{
				value: metric.Value,
				alpha: 0.1,
			}
		}

		// Update statistics
		t.updateStatistics(key, metric.Value)
	}

	return anomalies
}

// IsolationForest methods
func (f *IsolationForest) build() {
	f.trees = make([]*IsolationTree, f.numTrees)
	for i := 0; i < f.numTrees; i++ {
		f.trees[i] = &IsolationTree{}
		f.trees[i].build(f.generateSampleData(), 0, f.maxDepth)
	}
}

func (f *IsolationForest) anomalyScore(point []float64) float64 {
	pathLengths := make([]float64, f.numTrees)

	for i, tree := range f.trees {
		pathLengths[i] = tree.pathLength(point)
	}

	// Calculate average path length
	avgPathLength := 0.0
	for _, length := range pathLengths {
		avgPathLength += length
	}
	avgPathLength /= float64(f.numTrees)

	// Calculate anomaly score
	c := f.averagePathLength(f.sampleSize)
	score := math.Pow(2, -avgPathLength/c)

	return score
}

func (f *IsolationForest) averagePathLength(n int) float64 {
	if n <= 1 {
		return 0
	}
	if n == 2 {
		return 1
	}

	// Harmonic number approximation
	return 2 * (math.Log(float64(n-1)) + 0.5772156649) - 2*float64(n-1)/float64(n)
}

func (f *IsolationForest) generateSampleData() [][]float64 {
	// Generate synthetic sample data for training
	// In production, would use real historical data
	samples := make([][]float64, f.sampleSize)
	for i := range samples {
		samples[i] = make([]float64, 10) // 10 features
		for j := range samples[i] {
			samples[i][j] = randNormal() * 10
		}
	}
	return samples
}

func (tree *IsolationTree) build(data [][]float64, depth int, maxDepth int) {
	tree.root = tree.buildNode(data, depth, maxDepth)
}

func (tree *IsolationTree) buildNode(data [][]float64, depth int, maxDepth int) *IsolationNode {
	node := &IsolationNode{
		size:  len(data),
		depth: depth,
	}

	// Check stopping conditions
	if depth >= maxDepth || len(data) <= 1 {
		node.isLeaf = true
		return node
	}

	// Random feature and split value
	if len(data) > 0 && len(data[0]) > 0 {
		node.feature = randInt(len(data[0]))

		// Find min and max for the feature
		min, max := data[0][node.feature], data[0][node.feature]
		for _, point := range data[1:] {
			if point[node.feature] < min {
				min = point[node.feature]
			}
			if point[node.feature] > max {
				max = point[node.feature]
			}
		}

		if min == max {
			node.isLeaf = true
			return node
		}

		node.splitValue = min + randFloat()*(max-min)

		// Split data
		var leftData, rightData [][]float64
		for _, point := range data {
			if point[node.feature] <= node.splitValue {
				leftData = append(leftData, point)
			} else {
				rightData = append(rightData, point)
			}
		}

		// Recursively build children
		if len(leftData) > 0 {
			node.left = tree.buildNode(leftData, depth+1, maxDepth)
		}
		if len(rightData) > 0 {
			node.right = tree.buildNode(rightData, depth+1, maxDepth)
		}
	} else {
		node.isLeaf = true
	}

	return node
}

func (tree *IsolationTree) pathLength(point []float64) float64 {
	return tree.pathLengthRecursive(point, tree.root, 0)
}

func (tree *IsolationTree) pathLengthRecursive(point []float64, node *IsolationNode, currentDepth float64) float64 {
	if node == nil || node.isLeaf {
		return currentDepth + tree.c(node.size)
	}

	if point[node.feature] <= node.splitValue {
		return tree.pathLengthRecursive(point, node.left, currentDepth+1)
	}
	return tree.pathLengthRecursive(point, node.right, currentDepth+1)
}

func (tree *IsolationTree) c(size int) float64 {
	if size <= 1 {
		return 0
	}
	if size == 2 {
		return 1
	}
	return 2 * (math.Log(float64(size-1)) + 0.5772156649) - 2*float64(size-1)/float64(size)
}

// Autoencoder methods
func (a *Autoencoder) initialize() {
	// Initialize encoder
	a.encoder = &NeuralLayer{
		weights: initializeWeights(a.inputSize, a.bottleneck),
		bias:    make([]float64, a.bottleneck),
		activation: "relu",
	}

	// Initialize decoder
	a.decoder = &NeuralLayer{
		weights: initializeWeights(a.bottleneck, a.inputSize),
		bias:    make([]float64, a.inputSize),
		activation: "sigmoid",
	}
}

func (a *Autoencoder) getReconstructionError(input []float64) float64 {
	// Encode
	encoded := a.encoder.forward(input)

	// Decode
	reconstructed := a.decoder.forward(encoded)

	// Calculate MSE
	error := 0.0
	for i := range input {
		diff := input[i] - reconstructed[i]
		error += diff * diff
	}

	return math.Sqrt(error / float64(len(input)))
}

func (layer *NeuralLayer) forward(input []float64) []float64 {
	output := make([]float64, len(layer.bias))

	// Linear transformation
	for i := range output {
		sum := layer.bias[i]
		for j := range input {
			if j < len(layer.weights) && i < len(layer.weights[j]) {
				sum += input[j] * layer.weights[j][i]
			}
		}

		// Apply activation
		switch layer.activation {
		case "relu":
			output[i] = math.Max(0, sum)
		case "sigmoid":
			output[i] = 1.0 / (1.0 + math.Exp(-sum))
		default:
			output[i] = sum
		}
	}

	return output
}

// Helper methods
func (t *TelemetryAnalyzer) updateHistory(metric NetworkMetric) {
	key := fmt.Sprintf("%s:%s", metric.Component, metric.MetricName)

	if buffer, exists := t.metricHistory[key]; exists {
		buffer.add(metric.Value)
	} else {
		buffer := &CircularBuffer{
			capacity: 1000,
			data:     make([]float64, 1000),
		}
		buffer.add(metric.Value)
		t.metricHistory[key] = buffer
	}
}

func (t *TelemetryAnalyzer) metricsToFeatures(metrics []NetworkMetric) [][]float64 {
	features := make([][]float64, len(metrics))

	for i, metric := range metrics {
		features[i] = []float64{
			metric.Value,
			float64(metric.Timestamp.Unix()),
			// Add more features as needed
		}
	}

	return features
}

func (t *TelemetryAnalyzer) metricsToVector(metrics []NetworkMetric) []float64 {
	vector := make([]float64, t.autoencoder.inputSize)

	for i, metric := range metrics {
		if i < len(vector) {
			vector[i] = metric.Value / 100.0 // Normalize
		}
	}

	return vector
}

func (t *TelemetryAnalyzer) groupByComponent(metrics []NetworkMetric) map[string][]NetworkMetric {
	grouped := make(map[string][]NetworkMetric)

	for _, metric := range metrics {
		grouped[metric.Component] = append(grouped[metric.Component], metric)
	}

	return grouped
}

func (t *TelemetryAnalyzer) inferAnomalyType(metricName string) AnomalyType {
	switch metricName {
	case "bandwidth_util":
		return AnomalyBandwidth
	case "latency", "latency_ms":
		return AnomalyLatency
	case "packet_loss", "packet_loss_rate":
		return AnomalyPacketLoss
	case "jitter", "jitter_ms":
		return AnomalyJitter
	case "error_rate":
		return AnomalyErrorRate
	default:
		return AnomalyTrafficPattern
	}
}

func (t *TelemetryAnalyzer) calculateSeverity(score float64) string {
	if score > 0.9 {
		return "critical"
	} else if score > 0.7 {
		return "high"
	} else if score > 0.5 {
		return "medium"
	}
	return "low"
}

func (t *TelemetryAnalyzer) shouldAlert(key string) bool {
	if lastAlert, exists := t.lastAlerts[key]; exists {
		if time.Since(lastAlert) < t.alertCooldown {
			return false
		}
	}
	return true
}

func (t *TelemetryAnalyzer) updateStatistics(key string, value float64) {
	// Update mean and standard deviation
	if _, exists := t.statisticalDetector.mean[key]; !exists {
		t.statisticalDetector.mean[key] = value
		t.statisticalDetector.stdDev[key] = 0
	} else {
		// Welford's online algorithm
		oldMean := t.statisticalDetector.mean[key]
		t.statisticalDetector.mean[key] = oldMean + (value-oldMean)/100 // Simplified
		t.statisticalDetector.stdDev[key] = math.Sqrt(math.Pow(value-oldMean, 2) / 100)
	}
}

func (t *TelemetryAnalyzer) correlateAnomalies(anomalies []Anomaly) []Anomaly {
	// Remove duplicates and correlate related anomalies
	seen := make(map[string]bool)
	correlated := []Anomaly{}

	for _, anomaly := range anomalies {
		key := fmt.Sprintf("%s:%v", anomaly.Component, anomaly.Type)
		if !seen[key] {
			seen[key] = true
			correlated = append(correlated, anomaly)
		}
	}

	return correlated
}

func (t *TelemetryAnalyzer) filterAnomalies(anomalies []Anomaly) []Anomaly {
	filtered := []Anomaly{}

	for _, anomaly := range anomalies {
		if anomaly.Score > 0.5 { // Minimum score threshold
			filtered = append(filtered, anomaly)
		}
	}

	return filtered
}

func (t *TelemetryAnalyzer) updateBaseline(metrics []NetworkMetric, anomalies []Anomaly) {
	// Build set of anomalous metrics
	anomalousKeys := make(map[string]bool)
	for _, anomaly := range anomalies {
		key := fmt.Sprintf("%s:*", anomaly.Component)
		anomalousKeys[key] = true
	}

	// Update baseline with non-anomalous metrics
	for _, metric := range metrics {
		key := fmt.Sprintf("%s:%s", metric.Component, metric.MetricName)
		if !anomalousKeys[key] {
			t.baseline.update(key, metric.Value)
		}
	}
}

func (b *BaselineModel) update(key string, value float64) {
	profile, exists := b.profiles[key]
	if !exists {
		profile = &MetricProfile{
			Mean:        value,
			Min:         value,
			Max:         value,
			Percentiles: make(map[int]float64),
		}
		b.profiles[key] = profile
	}

	// Update profile with exponential moving average
	alpha := b.learningRate
	profile.Mean = profile.Mean*(1-alpha) + value*alpha

	if value < profile.Min {
		profile.Min = value
	}
	if value > profile.Max {
		profile.Max = value
	}

	profile.UpdateCount++
}

func (t *TelemetryAnalyzer) updateAnalysisMetrics(duration time.Duration) {
	// Exponential moving average
	alpha := 0.1
	t.detectionLatency = time.Duration(float64(t.detectionLatency)*(1-alpha) + float64(duration)*alpha)
	t.analysisCount++
}

// CircularBuffer methods
func (cb *CircularBuffer) add(value float64) {
	cb.data[cb.head] = value
	cb.head = (cb.head + 1) % cb.capacity
	if cb.size < cb.capacity {
		cb.size++
	}
}

func (cb *CircularBuffer) getValues() []float64 {
	values := make([]float64, cb.size)
	for i := 0; i < cb.size; i++ {
		idx := (cb.head - cb.size + i + cb.capacity) % cb.capacity
		values[i] = cb.data[idx]
	}
	return values
}

// ExponentialMovingAverage methods
func (ema *ExponentialMovingAverage) update(value float64) {
	ema.value = ema.alpha*value + (1-ema.alpha)*ema.value
}

// Helper functions
func initializeWeights(rows, cols int) [][]float64 {
	weights := make([][]float64, rows)
	scale := math.Sqrt(2.0 / float64(rows))

	for i := range weights {
		weights[i] = make([]float64, cols)
		for j := range weights[i] {
			weights[i][j] = randNormal() * scale
		}
	}

	return weights
}

func randNormal() float64 {
	// Box-Muller transform
	u1 := 1.0 - randFloat()
	u2 := 1.0 - randFloat()
	return math.Sqrt(-2.0*math.Log(u1)) * math.Cos(2.0*math.Pi*u2)
}

func randInt(max int) int {
	return int(randFloat() * float64(max))
}

func randFloat() float64 {
	return math.Float64frombits(0x3FF<<52|uint64(time.Now().UnixNano()&0xFFFFFFFFFFFFF)) - 1
}

// GetMetrics returns telemetry analyzer metrics
func (t *TelemetryAnalyzer) GetMetrics() map[string]interface{} {
	t.mu.RLock()
	defer t.mu.RUnlock()

	return map[string]interface{}{
		"anomalies_detected":   t.anomaliesDetected,
		"false_positives":      t.falsePositives,
		"detection_latency_ms": t.detectionLatency.Milliseconds(),
		"analysis_count":       t.analysisCount,
		"alert_threshold":      t.alertThreshold,
		"history_size":         len(t.anomalyHistory),
		"baseline_profiles":    len(t.baseline.profiles),
	}
}

// GetAnomalyHistory returns recent anomaly history
func (t *TelemetryAnalyzer) GetAnomalyHistory(limit int) []Anomaly {
	t.mu.RLock()
	defer t.mu.RUnlock()

	if limit > len(t.anomalyHistory) {
		limit = len(t.anomalyHistory)
	}

	// Return most recent anomalies
	start := len(t.anomalyHistory) - limit
	if start < 0 {
		start = 0
	}

	return t.anomalyHistory[start:]
}

// GetBaselineProfile returns the baseline profile for a metric
func (t *TelemetryAnalyzer) GetBaselineProfile(component, metric string) *MetricProfile {
	t.mu.RLock()
	defer t.mu.RUnlock()

	key := fmt.Sprintf("%s:%s", component, metric)
	return t.baseline.profiles[key]
}