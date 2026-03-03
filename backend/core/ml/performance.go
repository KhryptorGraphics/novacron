// Package ml provides neural network-based performance prediction
package ml

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sync"
	"time"

	"github.com/sirupsen/logrus"
)

// PerformancePredictor implements neural network for workload performance prediction
type PerformancePredictor struct {
	logger              *logrus.Logger
	neuralNet           *NeuralNetwork
	patternRecognizer   *PatternRecognizer
	capacityPlanner     *CapacityPlanner
	migrationOptimizer  *MigrationOptimizer
	historicalData      *PerformanceHistory
	predictions         map[string]*PerformancePrediction
	config              PerformanceConfig
	metricsCollector    *MetricsCollector
	mu                  sync.RWMutex
}

// PerformanceConfig configures the performance predictor
type PerformanceConfig struct {
	ModelLayers        []int         `json:"model_layers"`        // Neural network architecture
	LearningRate       float64       `json:"learning_rate"`        // Training learning rate
	PredictionWindow   time.Duration `json:"prediction_window"`    // How far ahead to predict
	UpdateFrequency    time.Duration `json:"update_frequency"`     // Model update frequency
	AccuracyTarget     float64       `json:"accuracy_target"`      // Target accuracy (85%)
	LatencyTarget      time.Duration `json:"latency_target"`       // Target latency (250ms)
	EnableCapacityPlan bool          `json:"enable_capacity_plan"` // Enable capacity planning
	EnableMigrationOpt bool          `json:"enable_migration_opt"` // Enable migration optimization
}

// PerformancePrediction contains performance predictions
type PerformancePrediction struct {
	Timestamp            time.Time                `json:"timestamp"`
	VMId                 string                   `json:"vm_id"`
	WorkloadType         string                   `json:"workload_type"`
	PredictedThroughput  float64                  `json:"predicted_throughput"`
	PredictedLatency     time.Duration            `json:"predicted_latency"`
	PredictedIOPS        float64                  `json:"predicted_iops"`
	PredictedBandwidth   float64                  `json:"predicted_bandwidth"`
	PerformanceScore     float64                  `json:"performance_score"`
	Bottlenecks          []Bottleneck             `json:"bottlenecks"`
	Recommendations      []PerformanceRecommendation `json:"recommendations"`
	CapacityPlan         *CapacityPlan            `json:"capacity_plan,omitempty"`
	MigrationSuggestion  *MigrationSuggestion     `json:"migration_suggestion,omitempty"`
	Confidence           float64                  `json:"confidence"`
	ModelAccuracy        float64                  `json:"model_accuracy"`
}

// Bottleneck represents a performance bottleneck
type Bottleneck struct {
	Type        string    `json:"type"`        // cpu, memory, io, network
	Severity    string    `json:"severity"`    // low, medium, high, critical
	Impact      float64   `json:"impact"`      // Performance impact percentage
	TimeWindow  time.Time `json:"time_window"` // When bottleneck occurs
	Description string    `json:"description"`
}

// PerformanceRecommendation suggests performance improvements
type PerformanceRecommendation struct {
	Action          string        `json:"action"`
	Priority        string        `json:"priority"`
	ExpectedImprovement float64   `json:"expected_improvement"`
	Implementation  string        `json:"implementation"`
	Cost            float64       `json:"cost"`
	TimeToImplement time.Duration `json:"time_to_implement"`
}

// NeuralNetwork implements a feedforward neural network
type NeuralNetwork struct {
	layers      []Layer
	weights     [][][]float64
	biases      [][]float64
	activations [][]float64
	trained     bool
	accuracy    float64
	epochs      int
}

// Layer represents a neural network layer
type Layer struct {
	neurons    int
	activation string // relu, sigmoid, tanh, softmax
}

// PatternRecognizer identifies workload patterns
type PatternRecognizer struct {
	patterns        map[string]*WorkloadPattern
	patternHistory  []PatternOccurrence
	anomalyDetector *AnomalyDetector
	mu              sync.RWMutex
}

// WorkloadPattern represents a recognized workload pattern
type WorkloadPattern struct {
	ID              string                 `json:"id"`
	Name            string                 `json:"name"`
	Type            string                 `json:"type"` // periodic, burst, steady, growing
	Characteristics map[string]float64     `json:"characteristics"`
	Frequency       time.Duration          `json:"frequency"`
	Duration        time.Duration          `json:"duration"`
	ResourceProfile ResourceProfile        `json:"resource_profile"`
	LastSeen        time.Time              `json:"last_seen"`
	Occurrences     int                    `json:"occurrences"`
}

// ResourceProfile defines resource usage patterns
type ResourceProfile struct {
	CPUPattern      []float64 `json:"cpu_pattern"`
	MemoryPattern   []float64 `json:"memory_pattern"`
	IOPattern       []float64 `json:"io_pattern"`
	NetworkPattern  []float64 `json:"network_pattern"`
}

// PatternOccurrence records when a pattern was detected
type PatternOccurrence struct {
	PatternID  string    `json:"pattern_id"`
	Timestamp  time.Time `json:"timestamp"`
	VMId       string    `json:"vm_id"`
	Confidence float64   `json:"confidence"`
	Duration   time.Duration `json:"duration"`
}

// CapacityPlanner handles capacity planning
type CapacityPlanner struct {
	currentCapacity  *SystemCapacity
	projectedDemand  map[time.Time]*DemandProjection
	plans            []*CapacityPlan
	constraints      []CapacityConstraint
	mu               sync.RWMutex
}

// SystemCapacity represents current system capacity
type SystemCapacity struct {
	TotalCPU      int     `json:"total_cpu"`
	TotalMemory   int64   `json:"total_memory"`
	TotalStorage  int64   `json:"total_storage"`
	TotalNetwork  int64   `json:"total_network"`
	UsedCPU       int     `json:"used_cpu"`
	UsedMemory    int64   `json:"used_memory"`
	UsedStorage   int64   `json:"used_storage"`
	UsedNetwork   int64   `json:"used_network"`
	VMCount       int     `json:"vm_count"`
	NodeCount     int     `json:"node_count"`
}

// DemandProjection projects future resource demand
type DemandProjection struct {
	Timestamp      time.Time `json:"timestamp"`
	ProjectedCPU   int       `json:"projected_cpu"`
	ProjectedMemory int64    `json:"projected_memory"`
	ProjectedIOPS  int64     `json:"projected_iops"`
	Confidence     float64   `json:"confidence"`
	Scenario       string    `json:"scenario"` // best, likely, worst
}

// CapacityPlan represents a capacity planning recommendation
type CapacityPlan struct {
	ID              string           `json:"id"`
	CreatedAt       time.Time        `json:"created_at"`
	TimeHorizon     time.Duration    `json:"time_horizon"`
	RequiredChanges []CapacityChange `json:"required_changes"`
	EstimatedCost   float64          `json:"estimated_cost"`
	RiskLevel       string           `json:"risk_level"`
	Benefits        []string         `json:"benefits"`
}

// CapacityChange represents a required capacity change
type CapacityChange struct {
	Type       string    `json:"type"`       // add_node, upgrade_cpu, add_memory
	Magnitude  float64   `json:"magnitude"`  // Amount of change
	Timing     time.Time `json:"timing"`     // When to implement
	Priority   string    `json:"priority"`
	Justification string `json:"justification"`
}

// CapacityConstraint defines capacity planning constraints
type CapacityConstraint struct {
	Type      string  `json:"type"`      // budget, hardware, policy
	Limit     float64 `json:"limit"`
	Flexible  bool    `json:"flexible"`
	Description string `json:"description"`
}

// MigrationOptimizer optimizes VM migration timing
type MigrationOptimizer struct {
	migrationWindows []MigrationWindow
	costModel        *MigrationCostModel
	impactPredictor  *ImpactPredictor
	mu               sync.RWMutex
}

// MigrationWindow represents an optimal migration window
type MigrationWindow struct {
	Start           time.Time `json:"start"`
	End             time.Time `json:"end"`
	Score           float64   `json:"score"`
	ExpectedImpact  float64   `json:"expected_impact"`
	ResourceAvailable bool    `json:"resource_available"`
	RiskLevel       string    `json:"risk_level"`
}

// MigrationSuggestion suggests optimal migration timing
type MigrationSuggestion struct {
	VMId            string           `json:"vm_id"`
	SourceNode      string           `json:"source_node"`
	TargetNode      string           `json:"target_node"`
	OptimalWindow   MigrationWindow  `json:"optimal_window"`
	Reason          string           `json:"reason"`
	ExpectedBenefit float64          `json:"expected_benefit"`
	EstimatedTime   time.Duration    `json:"estimated_time"`
	Prerequisites   []string         `json:"prerequisites"`
}

// MigrationCostModel calculates migration costs
type MigrationCostModel struct {
	downtimeCost    float64
	performanceCost float64
	resourceCost    float64
	riskFactor      float64
}

// ImpactPredictor predicts migration impact
type ImpactPredictor struct {
	historicalImpacts []MigrationImpact
	model             *NeuralNetwork
}

// MigrationImpact records historical migration impacts
type MigrationImpact struct {
	Timestamp        time.Time     `json:"timestamp"`
	VMSize           int64         `json:"vm_size"`
	MigrationTime    time.Duration `json:"migration_time"`
	Downtime         time.Duration `json:"downtime"`
	PerformanceImpact float64      `json:"performance_impact"`
	Success          bool          `json:"success"`
}

// PerformanceHistory stores historical performance data
type PerformanceHistory struct {
	data      []PerformanceRecord
	capacity  int
	position  int
	mu        sync.RWMutex
}

// PerformanceRecord represents a historical performance record
type PerformanceRecord struct {
	Timestamp   time.Time `json:"timestamp"`
	VMId        string    `json:"vm_id"`
	Throughput  float64   `json:"throughput"`
	Latency     float64   `json:"latency"`
	IOPS        float64   `json:"iops"`
	Bandwidth   float64   `json:"bandwidth"`
	CPUUsage    float64   `json:"cpu_usage"`
	MemoryUsage float64   `json:"memory_usage"`
}

// MetricsCollector collects performance metrics
type MetricsCollector struct {
	collectors map[string]MetricCollector
	buffer     chan PerformanceRecord
	mu         sync.RWMutex
}

// MetricCollector interface for metric collection
type MetricCollector interface {
	Collect() (PerformanceRecord, error)
}

// NewPerformancePredictor creates a new performance predictor
func NewPerformancePredictor(logger *logrus.Logger, config PerformanceConfig) *PerformancePredictor {
	// Default neural network architecture if not specified
	if len(config.ModelLayers) == 0 {
		config.ModelLayers = []int{10, 128, 64, 32, 4} // Input, hidden layers, output
	}
	
	return &PerformancePredictor{
		logger:              logger,
		neuralNet:           NewNeuralNetwork(config.ModelLayers),
		patternRecognizer:   NewPatternRecognizer(),
		capacityPlanner:     NewCapacityPlanner(),
		migrationOptimizer:  NewMigrationOptimizer(),
		historicalData:      NewPerformanceHistory(10000),
		predictions:         make(map[string]*PerformancePrediction),
		config:              config,
		metricsCollector:    NewMetricsCollector(),
	}
}

// NewNeuralNetwork creates a new neural network
func NewNeuralNetwork(layers []int) *NeuralNetwork {
	nn := &NeuralNetwork{
		layers: make([]Layer, len(layers)),
		epochs: 0,
	}
	
	// Initialize layers
	for i, neurons := range layers {
		activation := "relu"
		if i == len(layers)-1 {
			activation = "sigmoid" // Output layer
		}
		nn.layers[i] = Layer{
			neurons:    neurons,
			activation: activation,
		}
	}
	
	// Initialize weights and biases
	nn.initializeWeights()
	
	return nn
}

// NewPatternRecognizer creates a pattern recognizer
func NewPatternRecognizer() *PatternRecognizer {
	return &PatternRecognizer{
		patterns:       make(map[string]*WorkloadPattern),
		patternHistory: make([]PatternOccurrence, 0),
	}
}

// NewCapacityPlanner creates a capacity planner
func NewCapacityPlanner() *CapacityPlanner {
	return &CapacityPlanner{
		currentCapacity: &SystemCapacity{},
		projectedDemand: make(map[time.Time]*DemandProjection),
		plans:           make([]*CapacityPlan, 0),
		constraints:     make([]CapacityConstraint, 0),
	}
}

// NewMigrationOptimizer creates a migration optimizer
func NewMigrationOptimizer() *MigrationOptimizer {
	return &MigrationOptimizer{
		migrationWindows: make([]MigrationWindow, 0),
		costModel:        NewMigrationCostModel(),
		impactPredictor:  NewImpactPredictor(),
	}
}

// NewMigrationCostModel creates a migration cost model
func NewMigrationCostModel() *MigrationCostModel {
	return &MigrationCostModel{
		downtimeCost:    100.0,  // Cost per minute of downtime
		performanceCost: 50.0,   // Cost per percent performance degradation
		resourceCost:    10.0,   // Cost per GB transferred
		riskFactor:      1.2,    // Risk multiplier
	}
}

// NewImpactPredictor creates an impact predictor
func NewImpactPredictor() *ImpactPredictor {
	return &ImpactPredictor{
		historicalImpacts: make([]MigrationImpact, 0),
		model:             NewNeuralNetwork([]int{6, 32, 16, 3}), // Specialized for impact prediction
	}
}

// NewPerformanceHistory creates a performance history buffer
func NewPerformanceHistory(capacity int) *PerformanceHistory {
	return &PerformanceHistory{
		data:     make([]PerformanceRecord, capacity),
		capacity: capacity,
		position: 0,
	}
}

// NewMetricsCollector creates a metrics collector
func NewMetricsCollector() *MetricsCollector {
	return &MetricsCollector{
		collectors: make(map[string]MetricCollector),
		buffer:     make(chan PerformanceRecord, 1000),
	}
}

// Start begins the performance prediction service
func (pp *PerformancePredictor) Start(ctx context.Context) error {
	pp.logger.Info("Starting Performance Prediction Service")
	
	// Load existing model if available
	if err := pp.loadModel(); err != nil {
		pp.logger.WithError(err).Warn("No existing model found, will train new one")
	}
	
	// Start metrics collection
	go pp.collectMetrics(ctx)
	
	// Start pattern recognition
	go pp.recognizePatterns(ctx)
	
	// Start prediction loop
	go pp.predictionLoop(ctx)
	
	// Start model training loop
	go pp.trainingLoop(ctx)
	
	// Start capacity planning if enabled
	if pp.config.EnableCapacityPlan {
		go pp.capacityPlanningLoop(ctx)
	}
	
	// Start migration optimization if enabled
	if pp.config.EnableMigrationOpt {
		go pp.migrationOptimizationLoop(ctx)
	}
	
	return nil
}

// PredictPerformance generates performance predictions
func (pp *PerformancePredictor) PredictPerformance(vmID string, workloadType string) (*PerformancePrediction, error) {
	startTime := time.Now()
	
	pp.mu.RLock()
	defer pp.mu.RUnlock()
	
	// Get historical data
	history := pp.historicalData.GetRecent(vmID, 100)
	if len(history) < 10 {
		return nil, fmt.Errorf("insufficient historical data for prediction")
	}
	
	// Extract features
	features := pp.extractFeatures(history, workloadType)
	
	// Run neural network prediction
	predictions := pp.neuralNet.Predict(features)
	
	// Identify patterns
	patterns := pp.patternRecognizer.IdentifyPatterns(history)
	
	// Detect bottlenecks
	bottlenecks := pp.detectBottlenecks(predictions, patterns)
	
	// Generate recommendations
	recommendations := pp.generateRecommendations(predictions, bottlenecks)
	
	// Create capacity plan if enabled
	var capacityPlan *CapacityPlan
	if pp.config.EnableCapacityPlan {
		capacityPlan = pp.capacityPlanner.CreatePlan(predictions, pp.config.PredictionWindow)
	}
	
	// Create migration suggestion if enabled
	var migrationSuggestion *MigrationSuggestion
	if pp.config.EnableMigrationOpt {
		migrationSuggestion = pp.migrationOptimizer.SuggestMigration(vmID, predictions)
	}
	
	// Calculate performance score
	performanceScore := pp.calculatePerformanceScore(predictions)
	
	// Calculate confidence
	confidence := pp.calculateConfidence(history, predictions)
	
	predictionLatency := time.Since(startTime)
	
	result := &PerformancePrediction{
		Timestamp:            time.Now(),
		VMId:                 vmID,
		WorkloadType:         workloadType,
		PredictedThroughput:  predictions[0] * 1000, // Convert to requests/sec
		PredictedLatency:     time.Duration(predictions[1] * float64(time.Millisecond)),
		PredictedIOPS:        predictions[2] * 10000,
		PredictedBandwidth:   predictions[3] * 1000, // Convert to Mbps
		PerformanceScore:     performanceScore,
		Bottlenecks:          bottlenecks,
		Recommendations:      recommendations,
		CapacityPlan:         capacityPlan,
		MigrationSuggestion:  migrationSuggestion,
		Confidence:           confidence,
		ModelAccuracy:        pp.neuralNet.accuracy,
	}
	
	// Store prediction
	pp.predictions[vmID] = result
	
	// Log if latency exceeds target
	if predictionLatency > pp.config.LatencyTarget {
		pp.logger.WithFields(logrus.Fields{
			"vm_id":   vmID,
			"latency": predictionLatency,
			"target":  pp.config.LatencyTarget,
		}).Warn("Prediction latency exceeded target")
	}
	
	return result, nil
}

// Neural Network Implementation

func (nn *NeuralNetwork) initializeWeights() {
	nn.weights = make([][][]float64, len(nn.layers)-1)
	nn.biases = make([][]float64, len(nn.layers)-1)
	
	for i := 0; i < len(nn.layers)-1; i++ {
		rows := nn.layers[i+1].neurons
		cols := nn.layers[i].neurons
		
		// Xavier initialization
		scale := math.Sqrt(2.0 / float64(rows+cols))
		
		nn.weights[i] = make([][]float64, rows)
		nn.biases[i] = make([]float64, rows)
		
		for j := 0; j < rows; j++ {
			nn.weights[i][j] = make([]float64, cols)
			for k := 0; k < cols; k++ {
				nn.weights[i][j][k] = (math.Sin(float64(j*cols+k)) - 0.5) * 2 * scale
			}
			nn.biases[i][j] = 0.01
		}
	}
}

// Predict performs forward propagation
func (nn *NeuralNetwork) Predict(input []float64) []float64 {
	nn.activations = make([][]float64, len(nn.layers))
	nn.activations[0] = input
	
	// Forward propagation
	for i := 0; i < len(nn.layers)-1; i++ {
		nn.activations[i+1] = nn.forwardLayer(i)
	}
	
	return nn.activations[len(nn.layers)-1]
}

func (nn *NeuralNetwork) forwardLayer(layerIndex int) []float64 {
	input := nn.activations[layerIndex]
	weights := nn.weights[layerIndex]
	biases := nn.biases[layerIndex]
	
	output := make([]float64, len(weights))
	
	for i := 0; i < len(weights); i++ {
		sum := biases[i]
		for j := 0; j < len(input); j++ {
			sum += input[j] * weights[i][j]
		}
		output[i] = nn.activate(sum, nn.layers[layerIndex+1].activation)
	}
	
	return output
}

func (nn *NeuralNetwork) activate(x float64, activation string) float64 {
	switch activation {
	case "relu":
		return math.Max(0, x)
	case "sigmoid":
		return 1.0 / (1.0 + math.Exp(-x))
	case "tanh":
		return math.Tanh(x)
	default:
		return x
	}
}

// Train trains the neural network
func (nn *NeuralNetwork) Train(data [][]float64, labels [][]float64, epochs int, learningRate float64) {
	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := 0.0
		correct := 0
		
		for i := 0; i < len(data); i++ {
			// Forward pass
			output := nn.Predict(data[i])
			
			// Calculate loss
			loss := 0.0
			for j := 0; j < len(output); j++ {
				diff := output[j] - labels[i][j]
				loss += diff * diff
				
				if math.Abs(diff) < 0.1 {
					correct++
				}
			}
			totalLoss += loss
			
			// Backward pass
			nn.backward(labels[i], learningRate)
		}
		
		// Update accuracy
		nn.accuracy = float64(correct) / float64(len(data)*len(labels[0]))
		nn.epochs = epoch + 1
		
		// Adaptive learning rate
		if epoch%50 == 0 && epoch > 0 {
			learningRate *= 0.95
		}
	}
	
	nn.trained = true
}

func (nn *NeuralNetwork) backward(target []float64, learningRate float64) {
	// Simplified backpropagation
	outputLayer := len(nn.layers) - 1
	output := nn.activations[outputLayer]
	
	// Calculate output error
	outputError := make([]float64, len(output))
	for i := 0; i < len(output); i++ {
		outputError[i] = output[i] - target[i]
	}
	
	// Backpropagate error and update weights
	errors := outputError
	for i := len(nn.layers) - 2; i >= 0; i-- {
		newErrors := make([]float64, nn.layers[i].neurons)
		
		// Update weights and biases
		for j := 0; j < len(nn.weights[i]); j++ {
			for k := 0; k < len(nn.weights[i][j]); k++ {
				nn.weights[i][j][k] -= learningRate * errors[j] * nn.activations[i][k]
				newErrors[k] += errors[j] * nn.weights[i][j][k]
			}
			nn.biases[i][j] -= learningRate * errors[j]
		}
		
		errors = newErrors
	}
}

// Pattern Recognition

func (pr *PatternRecognizer) IdentifyPatterns(history []PerformanceRecord) []*WorkloadPattern {
	pr.mu.RLock()
	defer pr.mu.RUnlock()
	
	detectedPatterns := []*WorkloadPattern{}
	
	// Analyze time series for patterns
	cpuPattern := pr.extractTimeSeries(history, "cpu")
	memPattern := pr.extractTimeSeries(history, "memory")
	
	// Check for periodic patterns
	if period := pr.detectPeriodicity(cpuPattern); period > 0 {
		pattern := &WorkloadPattern{
			ID:        fmt.Sprintf("periodic-%d", time.Now().Unix()),
			Name:      "Periodic Workload",
			Type:      "periodic",
			Frequency: time.Duration(period) * time.Second,
			ResourceProfile: ResourceProfile{
				CPUPattern:    cpuPattern,
				MemoryPattern: memPattern,
			},
			LastSeen:    time.Now(),
			Occurrences: 1,
		}
		detectedPatterns = append(detectedPatterns, pattern)
		pr.patterns[pattern.ID] = pattern
	}
	
	// Check for burst patterns
	if pr.detectBurst(cpuPattern) {
		pattern := &WorkloadPattern{
			ID:   fmt.Sprintf("burst-%d", time.Now().Unix()),
			Name: "Burst Workload",
			Type: "burst",
			ResourceProfile: ResourceProfile{
				CPUPattern:    cpuPattern,
				MemoryPattern: memPattern,
			},
			LastSeen:    time.Now(),
			Occurrences: 1,
		}
		detectedPatterns = append(detectedPatterns, pattern)
		pr.patterns[pattern.ID] = pattern
	}
	
	return detectedPatterns
}

func (pr *PatternRecognizer) extractTimeSeries(history []PerformanceRecord, metric string) []float64 {
	series := make([]float64, len(history))
	
	for i, record := range history {
		switch metric {
		case "cpu":
			series[i] = record.CPUUsage
		case "memory":
			series[i] = record.MemoryUsage
		case "iops":
			series[i] = record.IOPS
		case "bandwidth":
			series[i] = record.Bandwidth
		}
	}
	
	return series
}

func (pr *PatternRecognizer) detectPeriodicity(series []float64) int {
	// Simple autocorrelation to detect periodicity
	maxLag := len(series) / 2
	maxCorr := 0.0
	bestPeriod := 0
	
	for lag := 1; lag < maxLag; lag++ {
		corr := pr.autocorrelation(series, lag)
		if corr > maxCorr && corr > 0.7 { // Threshold for significant correlation
			maxCorr = corr
			bestPeriod = lag
		}
	}
	
	return bestPeriod
}

func (pr *PatternRecognizer) autocorrelation(series []float64, lag int) float64 {
	n := len(series) - lag
	if n <= 0 {
		return 0
	}
	
	mean := 0.0
	for _, v := range series {
		mean += v
	}
	mean /= float64(len(series))
	
	var num, den1, den2 float64
	for i := 0; i < n; i++ {
		num += (series[i] - mean) * (series[i+lag] - mean)
		den1 += (series[i] - mean) * (series[i] - mean)
		den2 += (series[i+lag] - mean) * (series[i+lag] - mean)
	}
	
	if den1 == 0 || den2 == 0 {
		return 0
	}
	
	return num / math.Sqrt(den1*den2)
}

func (pr *PatternRecognizer) detectBurst(series []float64) bool {
	// Detect sudden spikes in the series
	threshold := 2.0 // Standard deviations
	
	mean := 0.0
	for _, v := range series {
		mean += v
	}
	mean /= float64(len(series))
	
	variance := 0.0
	for _, v := range series {
		variance += (v - mean) * (v - mean)
	}
	stdDev := math.Sqrt(variance / float64(len(series)))
	
	burstCount := 0
	for _, v := range series {
		if v > mean+threshold*stdDev {
			burstCount++
		}
	}
	
	// If more than 10% of values are bursts
	return float64(burstCount)/float64(len(series)) > 0.1
}

// Helper methods

func (pp *PerformancePredictor) extractFeatures(history []PerformanceRecord, workloadType string) []float64 {
	features := make([]float64, 10)
	
	if len(history) == 0 {
		return features
	}
	
	// Calculate statistical features
	var sumThroughput, sumLatency, sumIOPS, sumBandwidth float64
	var sumCPU, sumMemory float64
	
	for _, record := range history {
		sumThroughput += record.Throughput
		sumLatency += record.Latency
		sumIOPS += record.IOPS
		sumBandwidth += record.Bandwidth
		sumCPU += record.CPUUsage
		sumMemory += record.MemoryUsage
	}
	
	n := float64(len(history))
	features[0] = sumThroughput / n
	features[1] = sumLatency / n
	features[2] = sumIOPS / n
	features[3] = sumBandwidth / n
	features[4] = sumCPU / n
	features[5] = sumMemory / n
	
	// Add workload type encoding
	workloadTypes := map[string]float64{
		"web":      0.2,
		"database": 0.4,
		"compute":  0.6,
		"storage":  0.8,
		"mixed":    0.5,
	}
	
	if val, exists := workloadTypes[workloadType]; exists {
		features[6] = val
	}
	
	// Add temporal features
	now := time.Now()
	features[7] = float64(now.Hour()) / 24.0
	features[8] = float64(now.Weekday()) / 7.0
	
	// Add trend
	if len(history) >= 2 {
		features[9] = history[len(history)-1].Throughput - history[0].Throughput
	}
	
	return features
}

func (pp *PerformancePredictor) detectBottlenecks(predictions []float64, patterns []*WorkloadPattern) []Bottleneck {
	bottlenecks := []Bottleneck{}
	
	// Check CPU bottleneck
	if predictions[0] < 0.3 { // Low throughput prediction
		bottlenecks = append(bottlenecks, Bottleneck{
			Type:        "cpu",
			Severity:    pp.calculateBottleneckSeverity(predictions[0]),
			Impact:      (1.0 - predictions[0]) * 100,
			TimeWindow:  time.Now().Add(pp.config.PredictionWindow),
			Description: "CPU resources limiting performance",
		})
	}
	
	// Check latency bottleneck
	if predictions[1] > 0.8 { // High latency prediction
		bottlenecks = append(bottlenecks, Bottleneck{
			Type:        "latency",
			Severity:    pp.calculateBottleneckSeverity(predictions[1]),
			Impact:      predictions[1] * 100,
			TimeWindow:  time.Now().Add(pp.config.PredictionWindow),
			Description: "High latency affecting response times",
		})
	}
	
	// Check IO bottleneck
	if predictions[2] < 0.4 { // Low IOPS prediction
		bottlenecks = append(bottlenecks, Bottleneck{
			Type:        "io",
			Severity:    pp.calculateBottleneckSeverity(predictions[2]),
			Impact:      (1.0 - predictions[2]) * 100,
			TimeWindow:  time.Now().Add(pp.config.PredictionWindow),
			Description: "I/O operations limiting performance",
		})
	}
	
	return bottlenecks
}

func (pp *PerformancePredictor) calculateBottleneckSeverity(value float64) string {
	if value < 0.2 || value > 0.9 {
		return "critical"
	}
	if value < 0.3 || value > 0.8 {
		return "high"
	}
	if value < 0.4 || value > 0.7 {
		return "medium"
	}
	return "low"
}

func (pp *PerformancePredictor) generateRecommendations(predictions []float64, bottlenecks []Bottleneck) []PerformanceRecommendation {
	recommendations := []PerformanceRecommendation{}
	
	for _, bottleneck := range bottlenecks {
		switch bottleneck.Type {
		case "cpu":
			recommendations = append(recommendations, PerformanceRecommendation{
				Action:              "scale_cpu",
				Priority:            bottleneck.Severity,
				ExpectedImprovement: bottleneck.Impact * 0.7,
				Implementation:      "Add 2 vCPUs to VM",
				Cost:                50.0,
				TimeToImplement:     5 * time.Minute,
			})
			
		case "memory":
			recommendations = append(recommendations, PerformanceRecommendation{
				Action:              "increase_memory",
				Priority:            bottleneck.Severity,
				ExpectedImprovement: bottleneck.Impact * 0.8,
				Implementation:      "Increase RAM by 4GB",
				Cost:                30.0,
				TimeToImplement:     10 * time.Minute,
			})
			
		case "io":
			recommendations = append(recommendations, PerformanceRecommendation{
				Action:              "optimize_storage",
				Priority:            bottleneck.Severity,
				ExpectedImprovement: bottleneck.Impact * 0.6,
				Implementation:      "Switch to SSD storage",
				Cost:                100.0,
				TimeToImplement:     30 * time.Minute,
			})
			
		case "latency":
			recommendations = append(recommendations, PerformanceRecommendation{
				Action:              "optimize_network",
				Priority:            bottleneck.Severity,
				ExpectedImprovement: bottleneck.Impact * 0.5,
				Implementation:      "Enable network acceleration",
				Cost:                20.0,
				TimeToImplement:     15 * time.Minute,
			})
		}
	}
	
	return recommendations
}

func (pp *PerformancePredictor) calculatePerformanceScore(predictions []float64) float64 {
	// Weighted combination of predictions
	throughputWeight := 0.3
	latencyWeight := 0.3
	iopsWeight := 0.2
	bandwidthWeight := 0.2
	
	// Normalize and invert latency (lower is better)
	normalizedLatency := 1.0 - math.Min(1.0, predictions[1])
	
	score := predictions[0]*throughputWeight +
		normalizedLatency*latencyWeight +
		predictions[2]*iopsWeight +
		predictions[3]*bandwidthWeight
	
	return math.Min(1.0, math.Max(0.0, score))
}

func (pp *PerformancePredictor) calculateConfidence(history []PerformanceRecord, predictions []float64) float64 {
	// Base confidence on model accuracy
	confidence := pp.neuralNet.accuracy
	
	// Adjust based on data quality
	if len(history) < 50 {
		confidence *= 0.8
	}
	
	// Adjust based on prediction extremity
	for _, pred := range predictions {
		if pred < 0.1 || pred > 0.9 {
			confidence *= 0.9
		}
	}
	
	return math.Max(0.1, math.Min(1.0, confidence))
}

// Loop implementations

func (pp *PerformancePredictor) collectMetrics(ctx context.Context) {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()
	
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			// Collect metrics from all collectors
			for _, collector := range pp.metricsCollector.collectors {
				if record, err := collector.Collect(); err == nil {
					pp.historicalData.Add(record)
				}
			}
		}
	}
}

func (pp *PerformancePredictor) recognizePatterns(ctx context.Context) {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()
	
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			pp.analyzePatterns()
		}
	}
}

func (pp *PerformancePredictor) predictionLoop(ctx context.Context) {
	ticker := time.NewTicker(1 * time.Minute)
	defer ticker.Stop()
	
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			pp.runPredictions()
		}
	}
}

func (pp *PerformancePredictor) trainingLoop(ctx context.Context) {
	ticker := time.NewTicker(pp.config.UpdateFrequency)
	defer ticker.Stop()
	
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			pp.trainModel()
		}
	}
}

func (pp *PerformancePredictor) capacityPlanningLoop(ctx context.Context) {
	ticker := time.NewTicker(1 * time.Hour)
	defer ticker.Stop()
	
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			pp.updateCapacityPlan()
		}
	}
}

func (pp *PerformancePredictor) migrationOptimizationLoop(ctx context.Context) {
	ticker := time.NewTicker(30 * time.Minute)
	defer ticker.Stop()
	
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			pp.optimizeMigrations()
		}
	}
}

func (pp *PerformancePredictor) analyzePatterns() {
	pp.logger.Debug("Analyzing workload patterns")
}

func (pp *PerformancePredictor) runPredictions() {
	pp.mu.Lock()
	defer pp.mu.Unlock()
	
	for vmID := range pp.predictions {
		_, err := pp.PredictPerformance(vmID, "mixed")
		if err != nil {
			pp.logger.WithError(err).WithField("vm_id", vmID).Error("Prediction failed")
		}
	}
}

func (pp *PerformancePredictor) trainModel() {
	pp.logger.Info("Training performance prediction model")
	
	// Get training data
	allHistory := pp.historicalData.GetAll()
	if len(allHistory) < 100 {
		pp.logger.Warn("Insufficient data for training")
		return
	}
	
	// Prepare training data
	data := make([][]float64, len(allHistory))
	labels := make([][]float64, len(allHistory))
	
	for i, record := range allHistory {
		data[i] = []float64{
			record.CPUUsage / 100.0,
			record.MemoryUsage / 100.0,
			float64(record.Timestamp.Hour()) / 24.0,
			float64(record.Timestamp.Weekday()) / 7.0,
		}
		
		labels[i] = []float64{
			record.Throughput / 1000.0,
			record.Latency / 1000.0,
			record.IOPS / 10000.0,
			record.Bandwidth / 1000.0,
		}
	}
	
	// Train model
	pp.neuralNet.Train(data, labels, 100, pp.config.LearningRate)
	
	pp.logger.WithField("accuracy", pp.neuralNet.accuracy).Info("Model training completed")
}

func (pp *PerformancePredictor) updateCapacityPlan() {
	pp.logger.Debug("Updating capacity plan")
}

func (pp *PerformancePredictor) optimizeMigrations() {
	pp.logger.Debug("Optimizing VM migrations")
}

func (pp *PerformancePredictor) loadModel() error {
	// Load model from file
	return nil
}

func (pp *PerformancePredictor) saveModel() error {
	// Save model to file
	return nil
}

// PerformanceHistory methods

func (ph *PerformanceHistory) Add(record PerformanceRecord) {
	ph.mu.Lock()
	defer ph.mu.Unlock()
	
	ph.data[ph.position] = record
	ph.position = (ph.position + 1) % ph.capacity
}

func (ph *PerformanceHistory) GetRecent(vmID string, count int) []PerformanceRecord {
	ph.mu.RLock()
	defer ph.mu.RUnlock()
	
	result := []PerformanceRecord{}
	
	for i := 0; i < count && i < ph.capacity; i++ {
		idx := (ph.position - 1 - i + ph.capacity) % ph.capacity
		if ph.data[idx].VMId == vmID {
			result = append(result, ph.data[idx])
		}
	}
	
	return result
}

func (ph *PerformanceHistory) GetAll() []PerformanceRecord {
	ph.mu.RLock()
	defer ph.mu.RUnlock()
	
	result := make([]PerformanceRecord, 0, ph.capacity)
	for _, record := range ph.data {
		if record.Timestamp.IsZero() {
			continue
		}
		result = append(result, record)
	}
	
	return result
}

// CapacityPlanner methods

func (cp *CapacityPlanner) CreatePlan(predictions []float64, horizon time.Duration) *CapacityPlan {
	cp.mu.Lock()
	defer cp.mu.Unlock()
	
	plan := &CapacityPlan{
		ID:          fmt.Sprintf("plan-%d", time.Now().Unix()),
		CreatedAt:   time.Now(),
		TimeHorizon: horizon,
		RiskLevel:   "medium",
	}
	
	// Analyze predictions and create capacity changes
	if predictions[0] > 0.8 { // High throughput predicted
		plan.RequiredChanges = append(plan.RequiredChanges, CapacityChange{
			Type:          "add_cpu",
			Magnitude:     2.0,
			Timing:        time.Now().Add(horizon / 2),
			Priority:      "high",
			Justification: "Predicted high throughput requires additional CPU",
		})
	}
	
	return plan
}

// MigrationOptimizer methods

func (mo *MigrationOptimizer) SuggestMigration(vmID string, predictions []float64) *MigrationSuggestion {
	mo.mu.RLock()
	defer mo.mu.RUnlock()
	
	// Find optimal migration window
	optimalWindow := mo.findOptimalWindow()
	
	return &MigrationSuggestion{
		VMId:            vmID,
		SourceNode:      "node1",
		TargetNode:      "node2",
		OptimalWindow:   optimalWindow,
		Reason:          "Performance optimization",
		ExpectedBenefit: 0.25,
		EstimatedTime:   5 * time.Minute,
		Prerequisites:   []string{"target_node_available", "network_capacity"},
	}
}

func (mo *MigrationOptimizer) findOptimalWindow() MigrationWindow {
	// Find the best migration window based on historical data
	now := time.Now()
	
	return MigrationWindow{
		Start:             now.Add(2 * time.Hour),
		End:               now.Add(3 * time.Hour),
		Score:             0.85,
		ExpectedImpact:    0.05,
		ResourceAvailable: true,
		RiskLevel:         "low",
	}
}

// GetPredictions returns all current predictions
func (pp *PerformancePredictor) GetPredictions() map[string]*PerformancePrediction {
	pp.mu.RLock()
	defer pp.mu.RUnlock()
	
	result := make(map[string]*PerformancePrediction)
	for k, v := range pp.predictions {
		result[k] = v
	}
	
	return result
}

// GetCapacityPlans returns current capacity plans
func (pp *PerformancePredictor) GetCapacityPlans() []*CapacityPlan {
	return pp.capacityPlanner.plans
}

// GetMigrationWindows returns optimal migration windows
func (pp *PerformancePredictor) GetMigrationWindows() []MigrationWindow {
	return pp.migrationOptimizer.migrationWindows
}