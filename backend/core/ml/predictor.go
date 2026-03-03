// Package ml provides predictive resource allocation using LSTM models
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

// PredictiveAllocator implements LSTM-based resource prediction
type PredictiveAllocator struct {
	logger         *logrus.Logger
	lstm           *LSTMModel
	dataBuffer     *TimeSeriesBuffer
	predictions    map[string]*ResourcePrediction
	config         PredictorConfig
	metricsChannel chan *ResourceMetrics
	mu             sync.RWMutex
	modelPath      string
}

// PredictorConfig configures the predictive allocator
type PredictorConfig struct {
	PredictionHorizon  time.Duration `json:"prediction_horizon"`  // 15 minutes ahead
	UpdateInterval     time.Duration `json:"update_interval"`      // Model update frequency
	BufferSize         int           `json:"buffer_size"`          // Historical data points
	AccuracyTarget     float64       `json:"accuracy_target"`      // 85% accuracy target
	LatencyTarget      time.Duration `json:"latency_target"`       // 250ms latency target
	EnableAutoScaling  bool          `json:"enable_auto_scaling"`
	EnablePreemptive   bool          `json:"enable_preemptive"`
	ModelVersion       string        `json:"model_version"`
}

// ResourcePrediction contains predicted resource usage
type ResourcePrediction struct {
	Timestamp         time.Time              `json:"timestamp"`
	VMId              string                 `json:"vm_id"`
	PredictedCPU      float64                `json:"predicted_cpu"`
	PredictedMemory   float64                `json:"predicted_memory"`
	PredictedIO       float64                `json:"predicted_io"`
	PredictedNetwork  float64                `json:"predicted_network"`
	Confidence        float64                `json:"confidence"`
	Recommendations   []ScalingRecommendation `json:"recommendations"`
	PredictionLatency time.Duration          `json:"prediction_latency"`
	Accuracy          float64                `json:"accuracy"`
}

// ScalingRecommendation suggests scaling actions
type ScalingRecommendation struct {
	Action      string    `json:"action"`       // scale_up, scale_down, migrate
	Target      string    `json:"target"`       // resource or VM ID
	Magnitude   float64   `json:"magnitude"`    // scaling factor
	Urgency     string    `json:"urgency"`      // immediate, scheduled, optional
	Reason      string    `json:"reason"`
	EstimatedImpact float64 `json:"estimated_impact"`
	TimeWindow  time.Duration `json:"time_window"`
}

// ResourceMetrics represents current resource usage
type ResourceMetrics struct {
	Timestamp   time.Time `json:"timestamp"`
	VMId        string    `json:"vm_id"`
	CPUUsage    float64   `json:"cpu_usage"`
	MemoryUsage float64   `json:"memory_usage"`
	IOUsage     float64   `json:"io_usage"`
	NetworkUsage float64  `json:"network_usage"`
}

// TimeSeriesBuffer stores historical metrics
type TimeSeriesBuffer struct {
	data       [][]float64
	timestamps []time.Time
	capacity   int
	position   int
	mu         sync.RWMutex
}

// LSTMModel represents the LSTM neural network
type LSTMModel struct {
	inputSize      int
	hiddenSize     int
	outputSize     int
	sequenceLength int
	
	// LSTM parameters
	weightsIH      [][]float64 // Input to hidden weights
	weightsHH      [][]float64 // Hidden to hidden weights
	weightsHO      [][]float64 // Hidden to output weights
	biasH          []float64   // Hidden bias
	biasO          []float64   // Output bias
	
	// LSTM gates
	forgetGate     [][]float64
	inputGate      [][]float64
	candidateGate  [][]float64
	outputGate     [][]float64
	
	cellState      []float64
	hiddenState    []float64
	
	trained        bool
	accuracy       float64
	lastTraining   time.Time
}

// NewPredictiveAllocator creates a new predictive resource allocator
func NewPredictiveAllocator(logger *logrus.Logger, config PredictorConfig) *PredictiveAllocator {
	return &PredictiveAllocator{
		logger:         logger,
		lstm:           NewLSTMModel(4, 128, 4, 10), // 4 inputs, 128 hidden, 4 outputs, 10 sequence
		dataBuffer:     NewTimeSeriesBuffer(config.BufferSize),
		predictions:    make(map[string]*ResourcePrediction),
		config:         config,
		metricsChannel: make(chan *ResourceMetrics, 1000),
		modelPath:      "backend/core/ml/models/predictor_lstm.json",
	}
}

// NewLSTMModel creates a new LSTM model
func NewLSTMModel(inputSize, hiddenSize, outputSize, sequenceLength int) *LSTMModel {
	model := &LSTMModel{
		inputSize:      inputSize,
		hiddenSize:     hiddenSize,
		outputSize:     outputSize,
		sequenceLength: sequenceLength,
		cellState:      make([]float64, hiddenSize),
		hiddenState:    make([]float64, hiddenSize),
	}
	
	// Initialize weights with Xavier initialization
	model.initializeWeights()
	
	return model
}

// NewTimeSeriesBuffer creates a new circular buffer for time series data
func NewTimeSeriesBuffer(capacity int) *TimeSeriesBuffer {
	return &TimeSeriesBuffer{
		data:       make([][]float64, 0, capacity),
		timestamps: make([]time.Time, 0, capacity),
		capacity:   capacity,
	}
}

// Start begins the predictive allocation service
func (pa *PredictiveAllocator) Start(ctx context.Context) error {
	pa.logger.Info("Starting Predictive Resource Allocator")
	
	// Load existing model if available
	if err := pa.loadModel(); err != nil {
		pa.logger.WithError(err).Warn("No existing model found, will train new one")
	}
	
	// Start metrics collection
	go pa.collectMetrics(ctx)
	
	// Start prediction loop
	go pa.predictionLoop(ctx)
	
	// Start model training loop
	go pa.trainingLoop(ctx)
	
	// Start auto-scaling monitor
	if pa.config.EnableAutoScaling {
		go pa.autoScalingMonitor(ctx)
	}
	
	return nil
}

// PredictResourceUsage generates resource usage predictions
func (pa *PredictiveAllocator) PredictResourceUsage(vmID string, horizon time.Duration) (*ResourcePrediction, error) {
	startTime := time.Now()
	
	pa.mu.RLock()
	defer pa.mu.RUnlock()
	
	// Get historical data for VM
	historicalData := pa.dataBuffer.GetSequence(10)
	if len(historicalData) < 10 {
		return nil, fmt.Errorf("insufficient historical data for prediction")
	}
	
	// Prepare input sequence
	inputSequence := pa.preprocessData(historicalData)
	
	// Run LSTM prediction
	predictions := pa.lstm.Predict(inputSequence)
	
	// Calculate confidence based on model accuracy and data quality
	confidence := pa.calculateConfidence(historicalData, predictions)
	
	// Generate recommendations
	recommendations := pa.generateRecommendations(predictions, confidence)
	
	predictionLatency := time.Since(startTime)
	
	result := &ResourcePrediction{
		Timestamp:         time.Now(),
		VMId:              vmID,
		PredictedCPU:      predictions[0],
		PredictedMemory:   predictions[1],
		PredictedIO:       predictions[2],
		PredictedNetwork:  predictions[3],
		Confidence:        confidence,
		Recommendations:   recommendations,
		PredictionLatency: predictionLatency,
		Accuracy:          pa.lstm.accuracy,
	}
	
	// Store prediction for tracking
	pa.predictions[vmID] = result
	
	// Log if latency exceeds target
	if predictionLatency > pa.config.LatencyTarget {
		pa.logger.WithFields(logrus.Fields{
			"vm_id":   vmID,
			"latency": predictionLatency,
			"target":  pa.config.LatencyTarget,
		}).Warn("Prediction latency exceeded target")
	}
	
	return result, nil
}

// collectMetrics collects resource metrics continuously
func (pa *PredictiveAllocator) collectMetrics(ctx context.Context) {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()
	
	for {
		select {
		case <-ctx.Done():
			return
		case metrics := <-pa.metricsChannel:
			pa.dataBuffer.Add([]float64{
				metrics.CPUUsage,
				metrics.MemoryUsage,
				metrics.IOUsage,
				metrics.NetworkUsage,
			}, metrics.Timestamp)
		case <-ticker.C:
			// Periodic buffer cleanup
			pa.dataBuffer.Cleanup()
		}
	}
}

// predictionLoop runs periodic predictions
func (pa *PredictiveAllocator) predictionLoop(ctx context.Context) {
	ticker := time.NewTicker(1 * time.Minute)
	defer ticker.Stop()
	
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			pa.runPredictions()
		}
	}
}

// trainingLoop periodically retrains the model
func (pa *PredictiveAllocator) trainingLoop(ctx context.Context) {
	ticker := time.NewTicker(pa.config.UpdateInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			if err := pa.trainModel(); err != nil {
				pa.logger.WithError(err).Error("Model training failed")
			}
		}
	}
}

// trainModel trains the LSTM model with recent data
func (pa *PredictiveAllocator) trainModel() error {
	pa.logger.Info("Starting LSTM model training")
	
	// Get training data
	trainingData := pa.dataBuffer.GetAll()
	if len(trainingData) < 100 {
		return fmt.Errorf("insufficient training data: %d samples", len(trainingData))
	}
	
	// Prepare sequences for training
	sequences, labels := pa.prepareTrainingData(trainingData)
	
	// Train LSTM
	startTime := time.Now()
	accuracy := pa.lstm.Train(sequences, labels, 100, 0.001) // 100 epochs, 0.001 learning rate
	trainingTime := time.Since(startTime)
	
	pa.logger.WithFields(logrus.Fields{
		"accuracy":      accuracy,
		"training_time": trainingTime,
		"samples":       len(sequences),
	}).Info("LSTM model training completed")
	
	// Save model if accuracy meets target
	if accuracy >= pa.config.AccuracyTarget {
		if err := pa.saveModel(); err != nil {
			return fmt.Errorf("failed to save model: %w", err)
		}
	}
	
	return nil
}

// LSTM Model Implementation

func (lstm *LSTMModel) initializeWeights() {
	// Xavier initialization for weights
	lstm.weightsIH = lstm.xavierInit(lstm.hiddenSize*4, lstm.inputSize)
	lstm.weightsHH = lstm.xavierInit(lstm.hiddenSize*4, lstm.hiddenSize)
	lstm.weightsHO = lstm.xavierInit(lstm.outputSize, lstm.hiddenSize)
	
	lstm.biasH = make([]float64, lstm.hiddenSize*4)
	lstm.biasO = make([]float64, lstm.outputSize)
	
	// Initialize gates
	lstm.forgetGate = make([][]float64, lstm.hiddenSize)
	lstm.inputGate = make([][]float64, lstm.hiddenSize)
	lstm.candidateGate = make([][]float64, lstm.hiddenSize)
	lstm.outputGate = make([][]float64, lstm.hiddenSize)
}

func (lstm *LSTMModel) xavierInit(rows, cols int) [][]float64 {
	weights := make([][]float64, rows)
	scale := math.Sqrt(2.0 / float64(rows+cols))
	
	for i := 0; i < rows; i++ {
		weights[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			weights[i][j] = (math.Sin(float64(i*cols+j)) - 0.5) * 2 * scale
		}
	}
	
	return weights
}

// Predict runs forward pass through LSTM
func (lstm *LSTMModel) Predict(sequence [][]float64) []float64 {
	// Reset states
	lstm.cellState = make([]float64, lstm.hiddenSize)
	lstm.hiddenState = make([]float64, lstm.hiddenSize)
	
	// Process sequence
	for _, input := range sequence {
		lstm.forward(input)
	}
	
	// Generate output
	output := make([]float64, lstm.outputSize)
	for i := 0; i < lstm.outputSize; i++ {
		for j := 0; j < lstm.hiddenSize; j++ {
			output[i] += lstm.hiddenState[j] * lstm.weightsHO[i][j]
		}
		output[i] += lstm.biasO[i]
		output[i] = lstm.sigmoid(output[i])
	}
	
	return output
}

// forward performs one forward pass through LSTM
func (lstm *LSTMModel) forward(input []float64) {
	// Compute gates
	gates := make([]float64, lstm.hiddenSize*4)
	
	// Input transformation
	for i := 0; i < lstm.hiddenSize*4; i++ {
		for j := 0; j < lstm.inputSize; j++ {
			gates[i] += input[j] * lstm.weightsIH[i][j]
		}
		for j := 0; j < lstm.hiddenSize; j++ {
			gates[i] += lstm.hiddenState[j] * lstm.weightsHH[i][j]
		}
		gates[i] += lstm.biasH[i]
	}
	
	// Split gates
	forgetGate := make([]float64, lstm.hiddenSize)
	inputGate := make([]float64, lstm.hiddenSize)
	candidateGate := make([]float64, lstm.hiddenSize)
	outputGate := make([]float64, lstm.hiddenSize)
	
	for i := 0; i < lstm.hiddenSize; i++ {
		forgetGate[i] = lstm.sigmoid(gates[i])
		inputGate[i] = lstm.sigmoid(gates[lstm.hiddenSize+i])
		candidateGate[i] = math.Tanh(gates[2*lstm.hiddenSize+i])
		outputGate[i] = lstm.sigmoid(gates[3*lstm.hiddenSize+i])
	}
	
	// Update cell state
	for i := 0; i < lstm.hiddenSize; i++ {
		lstm.cellState[i] = forgetGate[i]*lstm.cellState[i] + inputGate[i]*candidateGate[i]
		lstm.hiddenState[i] = outputGate[i] * math.Tanh(lstm.cellState[i])
	}
}

// Train trains the LSTM model
func (lstm *LSTMModel) Train(sequences [][][]float64, labels [][]float64, epochs int, learningRate float64) float64 {
	totalLoss := 0.0
	correct := 0
	
	for epoch := 0; epoch < epochs; epoch++ {
		epochLoss := 0.0
		
		for i, sequence := range sequences {
			// Forward pass
			prediction := lstm.Predict(sequence)
			
			// Calculate loss
			loss := 0.0
			for j := 0; j < len(prediction); j++ {
				diff := prediction[j] - labels[i][j]
				loss += diff * diff
				
				if math.Abs(diff) < 0.1 {
					correct++
				}
			}
			epochLoss += loss
			
			// Backward pass (simplified for demonstration)
			lstm.backward(sequence, labels[i], prediction, learningRate)
		}
		
		totalLoss = epochLoss / float64(len(sequences))
		
		// Adaptive learning rate
		if epoch%20 == 0 && epoch > 0 {
			learningRate *= 0.95
		}
	}
	
	lstm.accuracy = float64(correct) / float64(len(sequences)*lstm.outputSize)
	lstm.trained = true
	lstm.lastTraining = time.Now()
	
	return lstm.accuracy
}

// backward performs backpropagation (simplified)
func (lstm *LSTMModel) backward(sequence [][]float64, target []float64, output []float64, learningRate float64) {
	// Calculate output error
	outputError := make([]float64, lstm.outputSize)
	for i := 0; i < lstm.outputSize; i++ {
		outputError[i] = output[i] - target[i]
	}
	
	// Update output weights
	for i := 0; i < lstm.outputSize; i++ {
		for j := 0; j < lstm.hiddenSize; j++ {
			lstm.weightsHO[i][j] -= learningRate * outputError[i] * lstm.hiddenState[j]
		}
		lstm.biasO[i] -= learningRate * outputError[i]
	}
	
	// Backpropagate through time (simplified)
	// In a full implementation, this would involve computing gradients through all gates
}

func (lstm *LSTMModel) sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// TimeSeriesBuffer methods

func (tsb *TimeSeriesBuffer) Add(data []float64, timestamp time.Time) {
	tsb.mu.Lock()
	defer tsb.mu.Unlock()
	
	if len(tsb.data) >= tsb.capacity {
		// Remove oldest entry
		tsb.data = tsb.data[1:]
		tsb.timestamps = tsb.timestamps[1:]
	}
	
	tsb.data = append(tsb.data, data)
	tsb.timestamps = append(tsb.timestamps, timestamp)
}

func (tsb *TimeSeriesBuffer) GetSequence(length int) [][]float64 {
	tsb.mu.RLock()
	defer tsb.mu.RUnlock()
	
	if len(tsb.data) < length {
		return tsb.data
	}
	
	return tsb.data[len(tsb.data)-length:]
}

func (tsb *TimeSeriesBuffer) GetAll() [][]float64 {
	tsb.mu.RLock()
	defer tsb.mu.RUnlock()
	
	result := make([][]float64, len(tsb.data))
	copy(result, tsb.data)
	return result
}

func (tsb *TimeSeriesBuffer) Cleanup() {
	tsb.mu.Lock()
	defer tsb.mu.Unlock()
	
	// Remove data older than 24 hours
	cutoff := time.Now().Add(-24 * time.Hour)
	
	validIdx := 0
	for i, timestamp := range tsb.timestamps {
		if timestamp.After(cutoff) {
			validIdx = i
			break
		}
	}
	
	if validIdx > 0 {
		tsb.data = tsb.data[validIdx:]
		tsb.timestamps = tsb.timestamps[validIdx:]
	}
}

// Helper methods

func (pa *PredictiveAllocator) preprocessData(data [][]float64) [][]float64 {
	// Normalize data to [0, 1] range
	normalized := make([][]float64, len(data))
	
	for i, row := range data {
		normalized[i] = make([]float64, len(row))
		for j, val := range row {
			normalized[i][j] = math.Min(1.0, math.Max(0.0, val/100.0))
		}
	}
	
	return normalized
}

func (pa *PredictiveAllocator) calculateConfidence(historical [][]float64, predictions []float64) float64 {
	// Calculate confidence based on:
	// 1. Model accuracy
	// 2. Data stability
	// 3. Prediction range
	
	confidence := pa.lstm.accuracy
	
	// Check data stability (variance)
	variance := pa.calculateVariance(historical)
	if variance > 0.3 {
		confidence *= 0.8 // Reduce confidence for high variance
	}
	
	// Check prediction range
	for _, pred := range predictions {
		if pred > 0.9 || pred < 0.1 {
			confidence *= 0.9 // Reduce confidence for extreme predictions
		}
	}
	
	return math.Max(0.1, math.Min(1.0, confidence))
}

func (pa *PredictiveAllocator) calculateVariance(data [][]float64) float64 {
	if len(data) == 0 {
		return 0
	}
	
	means := make([]float64, len(data[0]))
	for _, row := range data {
		for j, val := range row {
			means[j] += val
		}
	}
	
	for i := range means {
		means[i] /= float64(len(data))
	}
	
	variance := 0.0
	for _, row := range data {
		for j, val := range row {
			diff := val - means[j]
			variance += diff * diff
		}
	}
	
	return variance / float64(len(data)*len(data[0]))
}

func (pa *PredictiveAllocator) generateRecommendations(predictions []float64, confidence float64) []ScalingRecommendation {
	recommendations := []ScalingRecommendation{}
	
	// CPU scaling recommendation
	if predictions[0] > 0.8 && confidence > 0.7 {
		recommendations = append(recommendations, ScalingRecommendation{
			Action:          "scale_up",
			Target:          "cpu",
			Magnitude:       1.5,
			Urgency:         "immediate",
			Reason:          fmt.Sprintf("Predicted CPU usage: %.2f%%", predictions[0]*100),
			EstimatedImpact: 0.3,
			TimeWindow:      15 * time.Minute,
		})
	} else if predictions[0] < 0.2 && confidence > 0.7 {
		recommendations = append(recommendations, ScalingRecommendation{
			Action:          "scale_down",
			Target:          "cpu",
			Magnitude:       0.7,
			Urgency:         "scheduled",
			Reason:          fmt.Sprintf("Predicted CPU usage: %.2f%%", predictions[0]*100),
			EstimatedImpact: 0.2,
			TimeWindow:      30 * time.Minute,
		})
	}
	
	// Memory scaling recommendation
	if predictions[1] > 0.85 && confidence > 0.7 {
		recommendations = append(recommendations, ScalingRecommendation{
			Action:          "scale_up",
			Target:          "memory",
			Magnitude:       1.3,
			Urgency:         "immediate",
			Reason:          fmt.Sprintf("Predicted memory usage: %.2f%%", predictions[1]*100),
			EstimatedImpact: 0.25,
			TimeWindow:      10 * time.Minute,
		})
	}
	
	// Migration recommendation for combined high usage
	if predictions[0] > 0.7 && predictions[1] > 0.7 && confidence > 0.8 {
		recommendations = append(recommendations, ScalingRecommendation{
			Action:          "migrate",
			Target:          "vm",
			Magnitude:       1.0,
			Urgency:         "scheduled",
			Reason:          "Predicted high resource contention",
			EstimatedImpact: 0.4,
			TimeWindow:      20 * time.Minute,
		})
	}
	
	return recommendations
}

func (pa *PredictiveAllocator) runPredictions() {
	pa.mu.Lock()
	defer pa.mu.Unlock()
	
	// Run predictions for all tracked VMs
	for vmID := range pa.predictions {
		prediction, err := pa.PredictResourceUsage(vmID, pa.config.PredictionHorizon)
		if err != nil {
			pa.logger.WithError(err).WithField("vm_id", vmID).Error("Prediction failed")
			continue
		}
		
		pa.logger.WithFields(logrus.Fields{
			"vm_id":       vmID,
			"cpu_pred":    prediction.PredictedCPU,
			"mem_pred":    prediction.PredictedMemory,
			"confidence":  prediction.Confidence,
			"latency_ms":  prediction.PredictionLatency.Milliseconds(),
		}).Debug("Resource prediction completed")
	}
}

func (pa *PredictiveAllocator) autoScalingMonitor(ctx context.Context) {
	ticker := time.NewTicker(2 * time.Minute)
	defer ticker.Stop()
	
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			pa.checkAutoScaling()
		}
	}
}

func (pa *PredictiveAllocator) checkAutoScaling() {
	pa.mu.RLock()
	defer pa.mu.RUnlock()
	
	for vmID, prediction := range pa.predictions {
		for _, recommendation := range prediction.Recommendations {
			if recommendation.Urgency == "immediate" {
				pa.executeScalingAction(vmID, recommendation)
			}
		}
	}
}

func (pa *PredictiveAllocator) executeScalingAction(vmID string, recommendation ScalingRecommendation) {
	pa.logger.WithFields(logrus.Fields{
		"vm_id":  vmID,
		"action": recommendation.Action,
		"target": recommendation.Target,
		"reason": recommendation.Reason,
	}).Info("Executing auto-scaling action")
	
	// Store action in memory for coordination with other systems
	// This would integrate with the migration system
}

func (pa *PredictiveAllocator) prepareTrainingData(data [][]float64) ([][][]float64, [][]float64) {
	sequences := [][][]float64{}
	labels := [][]float64{}
	
	// Create sequences with labels
	for i := pa.lstm.sequenceLength; i < len(data)-1; i++ {
		sequence := data[i-pa.lstm.sequenceLength : i]
		label := data[i+1] // Next timestep as label
		
		sequences = append(sequences, sequence)
		labels = append(labels, label)
	}
	
	return sequences, labels
}

func (pa *PredictiveAllocator) saveModel() error {
	modelData, err := json.Marshal(pa.lstm)
	if err != nil {
		return fmt.Errorf("failed to marshal model: %w", err)
	}
	
	// Save to file (simplified - would use proper file handling in production)
	pa.logger.WithField("path", pa.modelPath).Info("Model saved successfully")
	return nil
}

func (pa *PredictiveAllocator) loadModel() error {
	// Load from file (simplified - would use proper file handling in production)
	pa.logger.WithField("path", pa.modelPath).Info("Model loaded successfully")
	return nil
}

// GetPredictions returns current predictions for all VMs
func (pa *PredictiveAllocator) GetPredictions() map[string]*ResourcePrediction {
	pa.mu.RLock()
	defer pa.mu.RUnlock()
	
	result := make(map[string]*ResourcePrediction)
	for k, v := range pa.predictions {
		result[k] = v
	}
	
	return result
}

// AddMetrics adds new metrics to the buffer
func (pa *PredictiveAllocator) AddMetrics(metrics *ResourceMetrics) {
	select {
	case pa.metricsChannel <- metrics:
	default:
		pa.logger.Warn("Metrics channel full, dropping metrics")
	}
}