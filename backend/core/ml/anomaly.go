// Package ml provides anomaly detection using Isolation Forest
package ml

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"sync"
	"time"

	"github.com/sirupsen/logrus"
)

// AnomalyDetector implements Isolation Forest for VM behavior anomaly detection
type AnomalyDetector struct {
	logger          *logrus.Logger
	forest          *IsolationForest
	thresholds      *AdaptiveThresholds
	alertManager    *AlertManager
	metricsBuffer   *MetricsRingBuffer
	anomalyScores   map[string]*AnomalyScore
	config          AnomalyConfig
	mu              sync.RWMutex
	learningEnabled bool
	modelVersion    string
}

// AnomalyConfig configures the anomaly detector
type AnomalyConfig struct {
	NumTrees           int           `json:"num_trees"`            // Number of trees in forest
	SampleSize         int           `json:"sample_size"`          // Subsample size
	ContaminationRate  float64       `json:"contamination_rate"`   // Expected anomaly rate
	AlertThreshold     float64       `json:"alert_threshold"`      // Score threshold for alerts
	LearningRate       float64       `json:"learning_rate"`        // Threshold adaptation rate
	WindowSize         time.Duration `json:"window_size"`          // Detection window
	UpdateInterval     time.Duration `json:"update_interval"`      // Model update frequency
	EnableAutoLearn    bool          `json:"enable_auto_learn"`    // Auto-adjust thresholds
	EnableClustering   bool          `json:"enable_clustering"`    // Group similar anomalies
}

// AnomalyScore represents anomaly detection results
type AnomalyScore struct {
	Timestamp       time.Time              `json:"timestamp"`
	VMId            string                 `json:"vm_id"`
	Score           float64                `json:"score"`            // 0-1, higher = more anomalous
	IsAnomaly       bool                   `json:"is_anomaly"`
	Severity        string                 `json:"severity"`         // low, medium, high, critical
	AnomalyType     string                 `json:"anomaly_type"`     // cpu_spike, memory_leak, etc.
	Features        map[string]float64     `json:"features"`
	Deviations      map[string]float64     `json:"deviations"`       // Feature deviations from normal
	ContextualInfo  map[string]interface{} `json:"contextual_info"`
	Confidence      float64                `json:"confidence"`
	FalsePositive   bool                   `json:"false_positive"`   // Marked by operator feedback
}

// IsolationForest implements the Isolation Forest algorithm
type IsolationForest struct {
	trees          []*IsolationTree
	numTrees       int
	sampleSize     int
	heightLimit    int
	anomalyScores  []float64
	trained        bool
	trainingData   [][]float64
}

// IsolationTree represents a single tree in the forest
type IsolationTree struct {
	root        *IsolationNode
	heightLimit int
	sampleSize  int
}

// IsolationNode represents a node in an isolation tree
type IsolationNode struct {
	left          *IsolationNode
	right         *IsolationNode
	featureIndex  int
	splitValue    float64
	size          int
	height        int
	isLeaf        bool
}

// AdaptiveThresholds manages self-learning threshold adjustment
type AdaptiveThresholds struct {
	thresholds      map[string]*ThresholdInfo
	historyWindow   time.Duration
	learningRate    float64
	mu              sync.RWMutex
}

// ThresholdInfo stores threshold information for a metric
type ThresholdInfo struct {
	Current      float64   `json:"current"`
	Historical   []float64 `json:"historical"`
	LastUpdated  time.Time `json:"last_updated"`
	Confidence   float64   `json:"confidence"`
	TruePositives int      `json:"true_positives"`
	FalsePositives int     `json:"false_positives"`
}

// AlertManager handles anomaly alerts
type AlertManager struct {
	alerts       []Alert
	subscribers  []AlertSubscriber
	alertHistory map[string][]Alert
	mu           sync.RWMutex
}

// Alert represents an anomaly alert
type Alert struct {
	ID           string       `json:"id"`
	Timestamp    time.Time    `json:"timestamp"`
	VMId         string       `json:"vm_id"`
	Severity     string       `json:"severity"`
	Type         string       `json:"type"`
	Message      string       `json:"message"`
	Score        float64      `json:"score"`
	Acknowledged bool         `json:"acknowledged"`
	ResolvedAt   *time.Time   `json:"resolved_at,omitempty"`
	Actions      []string     `json:"actions"`
}

// AlertSubscriber interface for alert notifications
type AlertSubscriber interface {
	OnAlert(alert Alert)
}

// MetricsRingBuffer stores recent metrics for analysis
type MetricsRingBuffer struct {
	data     []VMMetrics
	capacity int
	position int
	mu       sync.RWMutex
}

// VMMetrics represents VM performance metrics
type VMMetrics struct {
	Timestamp    time.Time          `json:"timestamp"`
	VMId         string             `json:"vm_id"`
	CPU          float64            `json:"cpu"`
	Memory       float64            `json:"memory"`
	DiskIO       float64            `json:"disk_io"`
	NetworkIO    float64            `json:"network_io"`
	Latency      float64            `json:"latency"`
	ErrorRate    float64            `json:"error_rate"`
	CustomMetrics map[string]float64 `json:"custom_metrics"`
}

// NewAnomalyDetector creates a new anomaly detector
func NewAnomalyDetector(logger *logrus.Logger, config AnomalyConfig) *AnomalyDetector {
	return &AnomalyDetector{
		logger:          logger,
		forest:          NewIsolationForest(config.NumTrees, config.SampleSize),
		thresholds:      NewAdaptiveThresholds(config.LearningRate),
		alertManager:    NewAlertManager(),
		metricsBuffer:   NewMetricsRingBuffer(10000),
		anomalyScores:   make(map[string]*AnomalyScore),
		config:          config,
		learningEnabled: config.EnableAutoLearn,
		modelVersion:    fmt.Sprintf("v%d", time.Now().Unix()),
	}
}

// NewIsolationForest creates a new Isolation Forest
func NewIsolationForest(numTrees, sampleSize int) *IsolationForest {
	heightLimit := int(math.Ceil(math.Log2(float64(sampleSize))))
	
	return &IsolationForest{
		trees:       make([]*IsolationTree, 0, numTrees),
		numTrees:    numTrees,
		sampleSize:  sampleSize,
		heightLimit: heightLimit,
	}
}

// NewAdaptiveThresholds creates adaptive threshold manager
func NewAdaptiveThresholds(learningRate float64) *AdaptiveThresholds {
	return &AdaptiveThresholds{
		thresholds:    make(map[string]*ThresholdInfo),
		historyWindow: 7 * 24 * time.Hour,
		learningRate:  learningRate,
	}
}

// NewAlertManager creates a new alert manager
func NewAlertManager() *AlertManager {
	return &AlertManager{
		alerts:       make([]Alert, 0),
		subscribers:  make([]AlertSubscriber, 0),
		alertHistory: make(map[string][]Alert),
	}
}

// NewMetricsRingBuffer creates a new ring buffer for metrics
func NewMetricsRingBuffer(capacity int) *MetricsRingBuffer {
	return &MetricsRingBuffer{
		data:     make([]VMMetrics, capacity),
		capacity: capacity,
		position: 0,
	}
}

// Start begins the anomaly detection service
func (ad *AnomalyDetector) Start(ctx context.Context) error {
	ad.logger.Info("Starting Anomaly Detection Service")
	
	// Initialize with historical data if available
	if err := ad.initializeModel(); err != nil {
		ad.logger.WithError(err).Warn("Failed to load historical model")
	}
	
	// Start detection loop
	go ad.detectionLoop(ctx)
	
	// Start model update loop
	go ad.modelUpdateLoop(ctx)
	
	// Start threshold adaptation loop
	if ad.config.EnableAutoLearn {
		go ad.thresholdAdaptationLoop(ctx)
	}
	
	// Start alert processing
	go ad.alertProcessingLoop(ctx)
	
	return nil
}

// DetectAnomaly performs real-time anomaly detection
func (ad *AnomalyDetector) DetectAnomaly(metrics VMMetrics) (*AnomalyScore, error) {
	startTime := time.Now()
	
	// Store metrics
	ad.metricsBuffer.Add(metrics)
	
	// Extract features
	features := ad.extractFeatures(metrics)
	
	// Calculate anomaly score using Isolation Forest
	score := ad.forest.AnomalyScore(features)
	
	// Determine if anomalous based on adaptive threshold
	threshold := ad.thresholds.GetThreshold("global")
	isAnomaly := score > threshold
	
	// Classify anomaly type and severity
	anomalyType := ad.classifyAnomalyType(features, score)
	severity := ad.calculateSeverity(score, threshold)
	
	// Calculate feature deviations
	deviations := ad.calculateDeviations(features)
	
	// Build anomaly score result
	result := &AnomalyScore{
		Timestamp:      metrics.Timestamp,
		VMId:           metrics.VMId,
		Score:          score,
		IsAnomaly:      isAnomaly,
		Severity:       severity,
		AnomalyType:    anomalyType,
		Features:       ad.featuresToMap(features),
		Deviations:     deviations,
		Confidence:     ad.calculateConfidence(score),
		FalsePositive:  false,
	}
	
	// Add contextual information
	result.ContextualInfo = ad.gatherContext(metrics, result)
	
	// Store result
	ad.mu.Lock()
	ad.anomalyScores[metrics.VMId] = result
	ad.mu.Unlock()
	
	// Generate alert if needed
	if isAnomaly && severity != "low" {
		ad.generateAlert(result)
	}
	
	detectionLatency := time.Since(startTime)
	if detectionLatency > 100*time.Millisecond {
		ad.logger.WithField("latency", detectionLatency).Warn("High anomaly detection latency")
	}
	
	return result, nil
}

// Train trains the Isolation Forest model
func (ad *AnomalyDetector) Train(trainingData []VMMetrics) error {
	ad.logger.Info("Training Isolation Forest model")
	
	// Convert metrics to feature vectors
	features := make([][]float64, len(trainingData))
	for i, metrics := range trainingData {
		features[i] = ad.extractFeatures(metrics)
	}
	
	// Train the forest
	ad.forest.Train(features)
	
	// Calculate initial thresholds
	scores := make([]float64, len(features))
	for i, feature := range features {
		scores[i] = ad.forest.AnomalyScore(feature)
	}
	
	// Set threshold based on contamination rate
	sort.Float64s(scores)
	thresholdIndex := int(float64(len(scores)) * (1 - ad.config.ContaminationRate))
	globalThreshold := scores[thresholdIndex]
	
	ad.thresholds.SetThreshold("global", globalThreshold, 0.8)
	
	ad.logger.WithFields(logrus.Fields{
		"samples":   len(trainingData),
		"threshold": globalThreshold,
		"trees":     ad.config.NumTrees,
	}).Info("Isolation Forest training completed")
	
	return nil
}

// IsolationForest Implementation

// Train builds the isolation forest
func (iforest *IsolationForest) Train(data [][]float64) {
	iforest.trainingData = data
	iforest.trees = make([]*IsolationTree, iforest.numTrees)
	
	for i := 0; i < iforest.numTrees; i++ {
		// Sample data
		sample := iforest.subsample(data)
		
		// Build tree
		tree := &IsolationTree{
			heightLimit: iforest.heightLimit,
			sampleSize:  len(sample),
		}
		tree.root = tree.build(sample, 0)
		iforest.trees[i] = tree
	}
	
	iforest.trained = true
}

// AnomalyScore calculates the anomaly score for a data point
func (iforest *IsolationForest) AnomalyScore(point []float64) float64 {
	if !iforest.trained {
		return 0.5 // Neutral score if not trained
	}
	
	totalPathLength := 0.0
	
	for _, tree := range iforest.trees {
		pathLength := tree.pathLength(point)
		totalPathLength += pathLength
	}
	
	avgPathLength := totalPathLength / float64(len(iforest.trees))
	
	// Calculate anomaly score
	n := float64(iforest.sampleSize)
	cn := 2*harmonicNumber(n-1) - (2*(n-1))/n
	
	score := math.Pow(2, -avgPathLength/cn)
	
	return score
}

func (iforest *IsolationForest) subsample(data [][]float64) [][]float64 {
	n := len(data)
	sampleSize := iforest.sampleSize
	if sampleSize > n {
		sampleSize = n
	}
	
	// Random sampling without replacement
	indices := rand.Perm(n)[:sampleSize]
	sample := make([][]float64, sampleSize)
	
	for i, idx := range indices {
		sample[i] = data[idx]
	}
	
	return sample
}

// IsolationTree Implementation

func (tree *IsolationTree) build(data [][]float64, height int) *IsolationNode {
	n := len(data)
	
	if height >= tree.heightLimit || n <= 1 {
		return &IsolationNode{
			size:   n,
			height: height,
			isLeaf: true,
		}
	}
	
	// Select random feature
	numFeatures := len(data[0])
	featureIndex := rand.Intn(numFeatures)
	
	// Get min and max values for the feature
	minVal, maxVal := data[0][featureIndex], data[0][featureIndex]
	for _, point := range data {
		if point[featureIndex] < minVal {
			minVal = point[featureIndex]
		}
		if point[featureIndex] > maxVal {
			maxVal = point[featureIndex]
		}
	}
	
	// Select random split value
	splitValue := minVal + rand.Float64()*(maxVal-minVal)
	
	// Split data
	var leftData, rightData [][]float64
	for _, point := range data {
		if point[featureIndex] < splitValue {
			leftData = append(leftData, point)
		} else {
			rightData = append(rightData, point)
		}
	}
	
	// Handle edge case where all points go to one side
	if len(leftData) == 0 || len(rightData) == 0 {
		return &IsolationNode{
			size:   n,
			height: height,
			isLeaf: true,
		}
	}
	
	return &IsolationNode{
		left:         tree.build(leftData, height+1),
		right:        tree.build(rightData, height+1),
		featureIndex: featureIndex,
		splitValue:   splitValue,
		size:         n,
		height:       height,
		isLeaf:       false,
	}
}

func (tree *IsolationTree) pathLength(point []float64) float64 {
	return tree.traverse(tree.root, point, 0)
}

func (tree *IsolationTree) traverse(node *IsolationNode, point []float64, currentHeight float64) float64 {
	if node.isLeaf {
		return currentHeight + c(float64(node.size))
	}
	
	if point[node.featureIndex] < node.splitValue {
		return tree.traverse(node.left, point, currentHeight+1)
	}
	return tree.traverse(node.right, point, currentHeight+1)
}

// Helper function for average path length adjustment
func c(n float64) float64 {
	if n <= 1 {
		return 0
	}
	if n == 2 {
		return 1
	}
	return 2*harmonicNumber(n-1) - (2*(n-1))/n
}

func harmonicNumber(n float64) float64 {
	return math.Log(n) + 0.5772156649 // Euler's constant
}

// Feature extraction and processing

func (ad *AnomalyDetector) extractFeatures(metrics VMMetrics) []float64 {
	features := []float64{
		metrics.CPU,
		metrics.Memory,
		metrics.DiskIO,
		metrics.NetworkIO,
		metrics.Latency,
		metrics.ErrorRate,
	}
	
	// Add temporal features
	hour := float64(metrics.Timestamp.Hour())
	dayOfWeek := float64(metrics.Timestamp.Weekday())
	features = append(features, hour/24.0, dayOfWeek/7.0)
	
	// Add statistical features from recent history
	recentMetrics := ad.metricsBuffer.GetRecent(metrics.VMId, 10)
	if len(recentMetrics) > 0 {
		features = append(features, ad.calculateStatisticalFeatures(recentMetrics)...)
	}
	
	return features
}

func (ad *AnomalyDetector) calculateStatisticalFeatures(metrics []VMMetrics) []float64 {
	// Calculate mean, std, trend for each metric
	cpuValues := make([]float64, len(metrics))
	memValues := make([]float64, len(metrics))
	
	for i, m := range metrics {
		cpuValues[i] = m.CPU
		memValues[i] = m.Memory
	}
	
	return []float64{
		mean(cpuValues),
		stdDev(cpuValues),
		trend(cpuValues),
		mean(memValues),
		stdDev(memValues),
		trend(memValues),
	}
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

func stdDev(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	m := mean(values)
	variance := 0.0
	for _, v := range values {
		variance += (v - m) * (v - m)
	}
	return math.Sqrt(variance / float64(len(values)))
}

func trend(values []float64) float64 {
	if len(values) < 2 {
		return 0
	}
	// Simple linear trend
	return values[len(values)-1] - values[0]
}

func (ad *AnomalyDetector) classifyAnomalyType(features []float64, score float64) string {
	// Analyze which features contribute most to anomaly
	if features[0] > 0.9 { // CPU
		return "cpu_spike"
	}
	if features[1] > 0.95 { // Memory
		return "memory_leak"
	}
	if features[2] > 0.8 && features[3] > 0.8 { // IO
		return "io_bottleneck"
	}
	if features[4] > 100 { // Latency
		return "high_latency"
	}
	if features[5] > 0.1 { // Error rate
		return "error_surge"
	}
	
	if score > 0.8 {
		return "complex_anomaly"
	}
	
	return "minor_deviation"
}

func (ad *AnomalyDetector) calculateSeverity(score, threshold float64) string {
	ratio := score / threshold
	
	if ratio < 1.2 {
		return "low"
	}
	if ratio < 1.5 {
		return "medium"
	}
	if ratio < 2.0 {
		return "high"
	}
	return "critical"
}

func (ad *AnomalyDetector) calculateDeviations(features []float64) map[string]float64 {
	// Calculate deviation from historical baseline
	deviations := make(map[string]float64)
	
	featureNames := []string{"cpu", "memory", "disk_io", "network_io", "latency", "error_rate"}
	
	for i, name := range featureNames {
		if i < len(features) {
			baseline := ad.thresholds.GetBaseline(name)
			if baseline > 0 {
				deviations[name] = (features[i] - baseline) / baseline
			}
		}
	}
	
	return deviations
}

func (ad *AnomalyDetector) calculateConfidence(score float64) float64 {
	// Confidence based on score clarity
	if score > 0.9 || score < 0.1 {
		return 0.95 // Very clear anomaly or normal
	}
	if score > 0.7 || score < 0.3 {
		return 0.8
	}
	return 0.6 // Borderline cases
}

func (ad *AnomalyDetector) featuresToMap(features []float64) map[string]float64 {
	featureMap := make(map[string]float64)
	featureNames := []string{"cpu", "memory", "disk_io", "network_io", "latency", "error_rate", "hour", "day_of_week"}
	
	for i, name := range featureNames {
		if i < len(features) {
			featureMap[name] = features[i]
		}
	}
	
	return featureMap
}

func (ad *AnomalyDetector) gatherContext(metrics VMMetrics, score *AnomalyScore) map[string]interface{} {
	context := make(map[string]interface{})
	
	// Add recent anomaly history
	recentAnomalies := ad.getRecentAnomalies(metrics.VMId, 1*time.Hour)
	context["recent_anomaly_count"] = len(recentAnomalies)
	
	// Add system state
	context["system_load"] = ad.getSystemLoad()
	context["active_vms"] = ad.getActiveVMCount()
	
	// Add temporal context
	context["is_peak_hour"] = metrics.Timestamp.Hour() >= 9 && metrics.Timestamp.Hour() <= 17
	context["is_weekend"] = metrics.Timestamp.Weekday() == time.Saturday || metrics.Timestamp.Weekday() == time.Sunday
	
	return context
}

// Adaptive Threshold Implementation

func (at *AdaptiveThresholds) GetThreshold(metric string) float64 {
	at.mu.RLock()
	defer at.mu.RUnlock()
	
	if info, exists := at.thresholds[metric]; exists {
		return info.Current
	}
	
	// Default threshold
	return 0.6
}

func (at *AdaptiveThresholds) SetThreshold(metric string, value float64, confidence float64) {
	at.mu.Lock()
	defer at.mu.Unlock()
	
	if info, exists := at.thresholds[metric]; exists {
		info.Historical = append(info.Historical, info.Current)
		if len(info.Historical) > 100 {
			info.Historical = info.Historical[1:]
		}
		info.Current = value
		info.Confidence = confidence
		info.LastUpdated = time.Now()
	} else {
		at.thresholds[metric] = &ThresholdInfo{
			Current:     value,
			Historical:  []float64{value},
			Confidence:  confidence,
			LastUpdated: time.Now(),
		}
	}
}

func (at *AdaptiveThresholds) GetBaseline(metric string) float64 {
	at.mu.RLock()
	defer at.mu.RUnlock()
	
	if info, exists := at.thresholds[metric]; exists && len(info.Historical) > 0 {
		return mean(info.Historical)
	}
	
	// Default baselines
	defaults := map[string]float64{
		"cpu":      50.0,
		"memory":   60.0,
		"disk_io":  40.0,
		"network_io": 30.0,
		"latency":  10.0,
		"error_rate": 0.01,
	}
	
	if val, exists := defaults[metric]; exists {
		return val
	}
	
	return 50.0
}

func (at *AdaptiveThresholds) UpdateWithFeedback(metric string, wasFalsePositive bool) {
	at.mu.Lock()
	defer at.mu.Unlock()
	
	if info, exists := at.thresholds[metric]; exists {
		if wasFalsePositive {
			info.FalsePositives++
			// Increase threshold to reduce false positives
			info.Current *= (1 + at.learningRate)
		} else {
			info.TruePositives++
			// Slightly decrease threshold if consistently correct
			if info.TruePositives > info.FalsePositives*2 {
				info.Current *= (1 - at.learningRate*0.5)
			}
		}
		
		// Update confidence
		total := info.TruePositives + info.FalsePositives
		if total > 0 {
			info.Confidence = float64(info.TruePositives) / float64(total)
		}
	}
}

// Alert Management

func (ad *AnomalyDetector) generateAlert(score *AnomalyScore) {
	alert := Alert{
		ID:        fmt.Sprintf("alert-%d", time.Now().UnixNano()),
		Timestamp: score.Timestamp,
		VMId:      score.VMId,
		Severity:  score.Severity,
		Type:      score.AnomalyType,
		Message:   fmt.Sprintf("Anomaly detected in VM %s: %s (score: %.3f)", score.VMId, score.AnomalyType, score.Score),
		Score:     score.Score,
		Actions:   ad.suggestActions(score),
	}
	
	ad.alertManager.AddAlert(alert)
}

func (ad *AnomalyDetector) suggestActions(score *AnomalyScore) []string {
	actions := []string{}
	
	switch score.AnomalyType {
	case "cpu_spike":
		actions = append(actions, "investigate_processes", "consider_scaling", "check_scheduled_tasks")
	case "memory_leak":
		actions = append(actions, "analyze_memory_dump", "restart_application", "check_memory_allocations")
	case "io_bottleneck":
		actions = append(actions, "check_disk_health", "optimize_io_operations", "consider_storage_upgrade")
	case "high_latency":
		actions = append(actions, "check_network_path", "analyze_slow_queries", "review_timeouts")
	case "error_surge":
		actions = append(actions, "check_logs", "review_recent_changes", "enable_debug_mode")
	default:
		actions = append(actions, "manual_investigation", "collect_diagnostics")
	}
	
	return actions
}

func (am *AlertManager) AddAlert(alert Alert) {
	am.mu.Lock()
	defer am.mu.Unlock()
	
	am.alerts = append(am.alerts, alert)
	
	// Store in history
	if _, exists := am.alertHistory[alert.VMId]; !exists {
		am.alertHistory[alert.VMId] = []Alert{}
	}
	am.alertHistory[alert.VMId] = append(am.alertHistory[alert.VMId], alert)
	
	// Notify subscribers
	for _, subscriber := range am.subscribers {
		go subscriber.OnAlert(alert)
	}
}

// MetricsRingBuffer Implementation

func (mrb *MetricsRingBuffer) Add(metrics VMMetrics) {
	mrb.mu.Lock()
	defer mrb.mu.Unlock()
	
	mrb.data[mrb.position] = metrics
	mrb.position = (mrb.position + 1) % mrb.capacity
}

func (mrb *MetricsRingBuffer) GetRecent(vmID string, count int) []VMMetrics {
	mrb.mu.RLock()
	defer mrb.mu.RUnlock()
	
	result := []VMMetrics{}
	
	// Start from most recent and work backwards
	for i := 0; i < count && i < mrb.capacity; i++ {
		idx := (mrb.position - 1 - i + mrb.capacity) % mrb.capacity
		if mrb.data[idx].VMId == vmID {
			result = append(result, mrb.data[idx])
		}
	}
	
	return result
}

// Loop implementations

func (ad *AnomalyDetector) detectionLoop(ctx context.Context) {
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()
	
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			ad.performBatchDetection()
		}
	}
}

func (ad *AnomalyDetector) modelUpdateLoop(ctx context.Context) {
	ticker := time.NewTicker(ad.config.UpdateInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			ad.updateModel()
		}
	}
}

func (ad *AnomalyDetector) thresholdAdaptationLoop(ctx context.Context) {
	ticker := time.NewTicker(1 * time.Hour)
	defer ticker.Stop()
	
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			ad.adaptThresholds()
		}
	}
}

func (ad *AnomalyDetector) alertProcessingLoop(ctx context.Context) {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()
	
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			ad.processAlerts()
		}
	}
}

func (ad *AnomalyDetector) performBatchDetection() {
	// Batch detection logic
	ad.logger.Debug("Performing batch anomaly detection")
}

func (ad *AnomalyDetector) updateModel() {
	// Model update logic
	ad.logger.Debug("Updating anomaly detection model")
}

func (ad *AnomalyDetector) adaptThresholds() {
	// Threshold adaptation logic
	ad.logger.Debug("Adapting detection thresholds")
}

func (ad *AnomalyDetector) processAlerts() {
	// Alert processing logic
	ad.logger.Debug("Processing anomaly alerts")
}

func (ad *AnomalyDetector) initializeModel() error {
	// Model initialization logic
	return nil
}

func (ad *AnomalyDetector) getRecentAnomalies(vmID string, duration time.Duration) []*AnomalyScore {
	ad.mu.RLock()
	defer ad.mu.RUnlock()
	
	cutoff := time.Now().Add(-duration)
	result := []*AnomalyScore{}
	
	for id, score := range ad.anomalyScores {
		if id == vmID && score.Timestamp.After(cutoff) && score.IsAnomaly {
			result = append(result, score)
		}
	}
	
	return result
}

func (ad *AnomalyDetector) getSystemLoad() float64 {
	// Simplified system load calculation
	return 0.65
}

func (ad *AnomalyDetector) getActiveVMCount() int {
	ad.mu.RLock()
	defer ad.mu.RUnlock()
	
	return len(ad.anomalyScores)
}

// MarkFalsePositive marks an anomaly as false positive for learning
func (ad *AnomalyDetector) MarkFalsePositive(vmID string, timestamp time.Time) {
	ad.mu.Lock()
	defer ad.mu.Unlock()
	
	if score, exists := ad.anomalyScores[vmID]; exists && score.Timestamp.Equal(timestamp) {
		score.FalsePositive = true
		
		// Update thresholds based on feedback
		ad.thresholds.UpdateWithFeedback("global", true)
		
		ad.logger.WithFields(logrus.Fields{
			"vm_id":     vmID,
			"timestamp": timestamp,
		}).Info("Anomaly marked as false positive for learning")
	}
}

// GetAnomalyScores returns current anomaly scores
func (ad *AnomalyDetector) GetAnomalyScores() map[string]*AnomalyScore {
	ad.mu.RLock()
	defer ad.mu.RUnlock()
	
	result := make(map[string]*AnomalyScore)
	for k, v := range ad.anomalyScores {
		result[k] = v
	}
	
	return result
}

// GetAlerts returns recent alerts
func (ad *AnomalyDetector) GetAlerts(limit int) []Alert {
	return ad.alertManager.GetRecentAlerts(limit)
}

func (am *AlertManager) GetRecentAlerts(limit int) []Alert {
	am.mu.RLock()
	defer am.mu.RUnlock()
	
	start := len(am.alerts) - limit
	if start < 0 {
		start = 0
	}
	
	return am.alerts[start:]
}