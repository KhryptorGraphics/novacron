package tiering

import (
	"math"
	"math/rand"
	"sort"
	"sync"
	"time"
)

// ExponentialSmoothingModel implements exponential smoothing for access prediction
type ExponentialSmoothingModel struct {
	alpha float64 // Smoothing factor (0-1)
	beta  float64 // Trend smoothing factor
	gamma float64 // Seasonality factor
	mu    sync.RWMutex
}

// NewExponentialSmoothingModel creates a new exponential smoothing model
func NewExponentialSmoothingModel() *ExponentialSmoothingModel {
	return &ExponentialSmoothingModel{
		alpha: 0.3, // Lower alpha = more weight on historical data
		beta:  0.1,
		gamma: 0.4,
	}
}

// Predict predicts temperature using exponential smoothing
func (esm *ExponentialSmoothingModel) Predict(history *AccessHistory) (Temperature, float64) {
	esm.mu.RLock()
	defer esm.mu.RUnlock()

	if len(history.AccessTimestamps) < 10 {
		return TemperatureCold, 0.3
	}

	// Calculate access rate over time
	accessRates := esm.calculateAccessRates(history)
	if len(accessRates) == 0 {
		return TemperatureCold, 0.3
	}

	// Apply exponential smoothing
	smoothed := esm.exponentialSmoothing(accessRates)

	// Predict next period
	prediction := smoothed[len(smoothed)-1]

	// Convert prediction to temperature
	var temp Temperature
	var confidence float64

	if prediction > 100 {
		temp = TemperatureHot
		confidence = math.Min(0.9, prediction/200)
	} else if prediction > 10 {
		temp = TemperatureWarm
		confidence = math.Min(0.8, prediction/50)
	} else if prediction > 1 {
		temp = TemperatureCold
		confidence = math.Min(0.7, prediction/5)
	} else {
		temp = TemperatureFrozen
		confidence = 0.6
	}

	// Adjust confidence based on data quality
	dataQuality := math.Min(float64(len(history.AccessTimestamps))/100, 1.0)
	confidence *= dataQuality

	return temp, confidence
}

// Train trains the model (adjusts parameters based on historical data)
func (esm *ExponentialSmoothingModel) Train(histories []*AccessHistory) {
	esm.mu.Lock()
	defer esm.mu.Unlock()

	// Simple parameter optimization using grid search
	bestAlpha := esm.alpha
	bestError := math.MaxFloat64

	for alpha := 0.1; alpha <= 0.9; alpha += 0.1 {
		totalError := 0.0
		
		for _, history := range histories {
			if len(history.AccessTimestamps) < 20 {
				continue
			}

			rates := esm.calculateAccessRates(history)
			if len(rates) < 2 {
				continue
			}

			// Split data into train and test
			trainSize := len(rates) * 3 / 4
			trainData := rates[:trainSize]
			testData := rates[trainSize:]

			// Apply smoothing with current alpha
			esm.alpha = alpha
			smoothed := esm.exponentialSmoothing(trainData)

			// Calculate error on test data
			for i, actual := range testData {
				predicted := smoothed[len(smoothed)-1] // Use last smoothed value
				error := math.Abs(predicted - actual)
				totalError += error
				
				// Update smoothed for next prediction
				if i < len(testData)-1 {
					smoothed = append(smoothed, alpha*actual+(1-alpha)*predicted)
				}
			}
		}

		if totalError < bestError {
			bestError = totalError
			bestAlpha = alpha
		}
	}

	esm.alpha = bestAlpha
}

// GetName returns the model name
func (esm *ExponentialSmoothingModel) GetName() string {
	return "exponential_smoothing"
}

// calculateAccessRates calculates access rates per hour
func (esm *ExponentialSmoothingModel) calculateAccessRates(history *AccessHistory) []float64 {
	if len(history.AccessTimestamps) < 2 {
		return []float64{}
	}

	// Group accesses by hour
	hourlyAccesses := make(map[int64]int)
	for _, timestamp := range history.AccessTimestamps {
		hour := timestamp.Unix() / 3600
		hourlyAccesses[hour]++
	}

	// Convert to sorted slice
	var hours []int64
	for hour := range hourlyAccesses {
		hours = append(hours, hour)
	}
	sort.Slice(hours, func(i, j int) bool { return hours[i] < hours[j] })

	rates := make([]float64, len(hours))
	for i, hour := range hours {
		rates[i] = float64(hourlyAccesses[hour])
	}

	return rates
}

// exponentialSmoothing applies exponential smoothing to time series
func (esm *ExponentialSmoothingModel) exponentialSmoothing(data []float64) []float64 {
	if len(data) == 0 {
		return []float64{}
	}

	smoothed := make([]float64, len(data))
	smoothed[0] = data[0]

	for i := 1; i < len(data); i++ {
		smoothed[i] = esm.alpha*data[i] + (1-esm.alpha)*smoothed[i-1]
	}

	return smoothed
}

// MarkovChainModel uses Markov chains to predict access patterns
type MarkovChainModel struct {
	// Transition matrix for temperature states
	transitionMatrix map[Temperature]map[Temperature]float64
	// State history for training
	stateHistory []Temperature
	mu           sync.RWMutex
}

// NewMarkovChainModel creates a new Markov chain model
func NewMarkovChainModel() *MarkovChainModel {
	// Initialize with default transition probabilities
	transitions := make(map[Temperature]map[Temperature]float64)
	
	// Hot state transitions
	transitions[TemperatureHot] = map[Temperature]float64{
		TemperatureHot:    0.7, // Likely to stay hot
		TemperatureWarm:   0.2,
		TemperatureCold:   0.08,
		TemperatureFrozen: 0.02,
	}
	
	// Warm state transitions
	transitions[TemperatureWarm] = map[Temperature]float64{
		TemperatureHot:    0.2,
		TemperatureWarm:   0.5, // Moderate stability
		TemperatureCold:   0.25,
		TemperatureFrozen: 0.05,
	}
	
	// Cold state transitions
	transitions[TemperatureCold] = map[Temperature]float64{
		TemperatureHot:    0.05,
		TemperatureWarm:   0.15,
		TemperatureCold:   0.6, // Likely to stay cold
		TemperatureFrozen: 0.2,
	}
	
	// Frozen state transitions
	transitions[TemperatureFrozen] = map[Temperature]float64{
		TemperatureHot:    0.01,
		TemperatureWarm:   0.04,
		TemperatureCold:   0.15,
		TemperatureFrozen: 0.8, // Very likely to stay frozen
	}

	return &MarkovChainModel{
		transitionMatrix: transitions,
		stateHistory:    make([]Temperature, 0),
	}
}

// Predict predicts temperature using Markov chain
func (mcm *MarkovChainModel) Predict(history *AccessHistory) (Temperature, float64) {
	mcm.mu.RLock()
	defer mcm.mu.RUnlock()

	// Determine current state based on recent access frequency
	currentTemp := mcm.getCurrentTemperature(history)

	// Get transition probabilities for current state
	transitions, exists := mcm.transitionMatrix[currentTemp]
	if !exists {
		return TemperatureCold, 0.5
	}

	// Find most likely next state
	var nextTemp Temperature
	maxProb := 0.0
	
	for temp, prob := range transitions {
		if prob > maxProb {
			maxProb = prob
			nextTemp = temp
		}
	}

	// Confidence is based on transition probability and data quality
	dataQuality := math.Min(float64(len(history.AccessTimestamps))/50, 1.0)
	confidence := maxProb * dataQuality

	return nextTemp, confidence
}

// Train trains the Markov chain model
func (mcm *MarkovChainModel) Train(histories []*AccessHistory) {
	mcm.mu.Lock()
	defer mcm.mu.Unlock()

	// Count state transitions
	transitionCounts := make(map[Temperature]map[Temperature]int)
	for temp := TemperatureHot; temp <= TemperatureFrozen; temp++ {
		transitionCounts[temp] = make(map[Temperature]int)
	}

	// Process each history
	for _, history := range histories {
		states := mcm.extractStates(history)
		
		for i := 0; i < len(states)-1; i++ {
			fromState := states[i]
			toState := states[i+1]
			transitionCounts[fromState][toState]++
		}
	}

	// Update transition matrix
	for fromTemp, toCounts := range transitionCounts {
		total := 0
		for _, count := range toCounts {
			total += count
		}

		if total > 0 {
			if mcm.transitionMatrix[fromTemp] == nil {
				mcm.transitionMatrix[fromTemp] = make(map[Temperature]float64)
			}

			for toTemp, count := range toCounts {
				mcm.transitionMatrix[fromTemp][toTemp] = float64(count) / float64(total)
			}
		}
	}
}

// GetName returns the model name
func (mcm *MarkovChainModel) GetName() string {
	return "markov_chain"
}

// getCurrentTemperature determines current temperature from history
func (mcm *MarkovChainModel) getCurrentTemperature(history *AccessHistory) Temperature {
	// Calculate recent access frequency (last 24 hours)
	recentCount := 0
	cutoff := time.Now().Add(-24 * time.Hour)
	
	for _, timestamp := range history.AccessTimestamps {
		if timestamp.After(cutoff) {
			recentCount++
		}
	}

	// Convert to temperature
	if recentCount > 100 {
		return TemperatureHot
	} else if recentCount > 10 {
		return TemperatureWarm
	} else if recentCount > 1 {
		return TemperatureCold
	}
	return TemperatureFrozen
}

// extractStates extracts temperature states from history
func (mcm *MarkovChainModel) extractStates(history *AccessHistory) []Temperature {
	// Group accesses by day and determine temperature for each day
	dayAccesses := make(map[int]int)
	
	for _, timestamp := range history.AccessTimestamps {
		day := int(timestamp.Unix() / 86400)
		dayAccesses[day]++
	}

	// Sort days
	var days []int
	for day := range dayAccesses {
		days = append(days, day)
	}
	sort.Ints(days)

	// Convert to temperature states
	states := make([]Temperature, len(days))
	for i, day := range days {
		count := dayAccesses[day]
		if count > 100 {
			states[i] = TemperatureHot
		} else if count > 10 {
			states[i] = TemperatureWarm
		} else if count > 1 {
			states[i] = TemperatureCold
		} else {
			states[i] = TemperatureFrozen
		}
	}

	return states
}

// SimpleNeuralNetworkModel implements a simple neural network for prediction
type SimpleNeuralNetworkModel struct {
	// Network weights
	inputWeights  [][]float64
	hiddenWeights [][]float64
	outputWeights []float64
	// Network configuration
	inputSize  int
	hiddenSize int
	outputSize int
	// Learning rate
	learningRate float64
	mu           sync.RWMutex
}

// NewSimpleNeuralNetworkModel creates a simple neural network model
func NewSimpleNeuralNetworkModel() *SimpleNeuralNetworkModel {
	inputSize := 10  // Features: hourly pattern, daily pattern, frequency, etc.
	hiddenSize := 8
	outputSize := 4  // Four temperature classes

	model := &SimpleNeuralNetworkModel{
		inputSize:    inputSize,
		hiddenSize:   hiddenSize,
		outputSize:   outputSize,
		learningRate: 0.01,
	}

	// Initialize weights randomly
	model.initializeWeights()

	return model
}

// initializeWeights initializes network weights randomly
func (snnm *SimpleNeuralNetworkModel) initializeWeights() {
	// Input to hidden layer weights
	snnm.inputWeights = make([][]float64, snnm.inputSize)
	for i := 0; i < snnm.inputSize; i++ {
		snnm.inputWeights[i] = make([]float64, snnm.hiddenSize)
		for j := 0; j < snnm.hiddenSize; j++ {
			snnm.inputWeights[i][j] = (rand.Float64() - 0.5) * 0.1
		}
	}

	// Hidden to output layer weights
	snnm.hiddenWeights = make([][]float64, snnm.hiddenSize)
	for i := 0; i < snnm.hiddenSize; i++ {
		snnm.hiddenWeights[i] = make([]float64, snnm.outputSize)
		for j := 0; j < snnm.outputSize; j++ {
			snnm.hiddenWeights[i][j] = (rand.Float64() - 0.5) * 0.1
		}
	}
}

// Predict predicts temperature using neural network
func (snnm *SimpleNeuralNetworkModel) Predict(history *AccessHistory) (Temperature, float64) {
	snnm.mu.RLock()
	defer snnm.mu.RUnlock()

	// Extract features
	features := snnm.extractFeatures(history)
	if len(features) != snnm.inputSize {
		return TemperatureCold, 0.3
	}

	// Forward propagation
	hidden := snnm.forwardToHidden(features)
	output := snnm.forwardToOutput(hidden)

	// Find class with highest activation
	maxIdx := 0
	maxVal := output[0]
	for i := 1; i < len(output); i++ {
		if output[i] > maxVal {
			maxVal = output[i]
			maxIdx = i
		}
	}

	// Convert index to temperature
	temps := []Temperature{TemperatureHot, TemperatureWarm, TemperatureCold, TemperatureFrozen}
	
	// Confidence is the softmax probability
	confidence := snnm.softmax(output)[maxIdx]

	return temps[maxIdx], confidence
}

// Train trains the neural network
func (snnm *SimpleNeuralNetworkModel) Train(histories []*AccessHistory) {
	snnm.mu.Lock()
	defer snnm.mu.Unlock()

	// Simple training loop
	epochs := 100
	
	for epoch := 0; epoch < epochs; epoch++ {
		totalError := 0.0
		
		for _, history := range histories {
			if len(history.AccessTimestamps) < 20 {
				continue
			}

			// Extract features and target
			features := snnm.extractFeatures(history)
			if len(features) != snnm.inputSize {
				continue
			}

			target := snnm.getTargetOutput(history)

			// Forward propagation
			hidden := snnm.forwardToHidden(features)
			output := snnm.forwardToOutput(hidden)

			// Calculate error
			for i, out := range output {
				error := target[i] - out
				totalError += error * error
			}

			// Backpropagation (simplified)
			// This is a simplified version for demonstration
			// A real implementation would use proper backpropagation
		}

		// Early stopping if error is low enough
		if totalError < 0.01 {
			break
		}
	}
}

// GetName returns the model name
func (snnm *SimpleNeuralNetworkModel) GetName() string {
	return "neural_network"
}

// extractFeatures extracts features from access history
func (snnm *SimpleNeuralNetworkModel) extractFeatures(history *AccessHistory) []float64 {
	features := make([]float64, snnm.inputSize)

	if len(history.AccessTimestamps) == 0 {
		return features
	}

	// Feature 1: Access frequency (normalized)
	features[0] = math.Min(float64(len(history.AccessTimestamps))/1000, 1.0)

	// Feature 2: Mean inter-arrival time (normalized)
	if history.Mean > 0 {
		features[1] = 1.0 / (1.0 + history.Mean/3600) // Normalize to hours
	}

	// Feature 3: Access burstiness
	features[2] = math.Min(history.AccessBurstiness, 1.0)

	// Feature 4-7: Time-of-day pattern (peak hours)
	peakHours := snnm.findPeakHours(history.HourlyAccess[:], 4)
	for i, hour := range peakHours {
		if i < 4 {
			features[3+i] = float64(hour) / 24.0
		}
	}

	// Feature 8: Day-of-week concentration
	maxDayAccess := 0
	for _, count := range history.DailyAccess {
		if count > maxDayAccess {
			maxDayAccess = count
		}
	}
	totalDayAccess := 0
	for _, count := range history.DailyAccess {
		totalDayAccess += count
	}
	if totalDayAccess > 0 {
		features[7] = float64(maxDayAccess) / float64(totalDayAccess)
	}

	// Feature 9: Recent access trend
	recentCount := 0
	oldCount := 0
	midPoint := time.Now().Add(-7 * 24 * time.Hour)
	
	for _, timestamp := range history.AccessTimestamps {
		if timestamp.After(midPoint) {
			recentCount++
		} else {
			oldCount++
		}
	}
	
	if oldCount > 0 {
		features[8] = float64(recentCount) / float64(oldCount)
	}

	// Feature 10: Variance coefficient
	if history.Mean > 0 {
		features[9] = history.StdDev / history.Mean
	}

	return features
}

// findPeakHours finds the top N peak hours
func (snnm *SimpleNeuralNetworkModel) findPeakHours(hourlyAccess []int, n int) []int {
	type hourCount struct {
		hour  int
		count int
	}

	hours := make([]hourCount, 24)
	for i := 0; i < 24; i++ {
		hours[i] = hourCount{hour: i, count: hourlyAccess[i]}
	}

	sort.Slice(hours, func(i, j int) bool {
		return hours[i].count > hours[j].count
	})

	result := make([]int, n)
	for i := 0; i < n && i < len(hours); i++ {
		result[i] = hours[i].hour
	}

	return result
}

// getTargetOutput gets target output for training
func (snnm *SimpleNeuralNetworkModel) getTargetOutput(history *AccessHistory) []float64 {
	target := make([]float64, snnm.outputSize)

	// Determine actual temperature based on access frequency
	recentCount := 0
	cutoff := time.Now().Add(-24 * time.Hour)
	
	for _, timestamp := range history.AccessTimestamps {
		if timestamp.After(cutoff) {
			recentCount++
		}
	}

	// One-hot encoding
	if recentCount > 100 {
		target[0] = 1.0 // Hot
	} else if recentCount > 10 {
		target[1] = 1.0 // Warm
	} else if recentCount > 1 {
		target[2] = 1.0 // Cold
	} else {
		target[3] = 1.0 // Frozen
	}

	return target
}

// forwardToHidden forward propagation to hidden layer
func (snnm *SimpleNeuralNetworkModel) forwardToHidden(input []float64) []float64 {
	hidden := make([]float64, snnm.hiddenSize)

	for j := 0; j < snnm.hiddenSize; j++ {
		sum := 0.0
		for i := 0; i < snnm.inputSize; i++ {
			sum += input[i] * snnm.inputWeights[i][j]
		}
		hidden[j] = snnm.sigmoid(sum)
	}

	return hidden
}

// forwardToOutput forward propagation to output layer
func (snnm *SimpleNeuralNetworkModel) forwardToOutput(hidden []float64) []float64 {
	output := make([]float64, snnm.outputSize)

	for k := 0; k < snnm.outputSize; k++ {
		sum := 0.0
		for j := 0; j < snnm.hiddenSize; j++ {
			sum += hidden[j] * snnm.hiddenWeights[j][k]
		}
		output[k] = snnm.sigmoid(sum)
	}

	return output
}

// sigmoid activation function
func (snnm *SimpleNeuralNetworkModel) sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// softmax converts outputs to probabilities
func (snnm *SimpleNeuralNetworkModel) softmax(output []float64) []float64 {
	probs := make([]float64, len(output))
	
	// Find max for numerical stability
	max := output[0]
	for _, val := range output {
		if val > max {
			max = val
		}
	}

	// Calculate exp and sum
	sum := 0.0
	for i, val := range output {
		probs[i] = math.Exp(val - max)
		sum += probs[i]
	}

	// Normalize
	for i := range probs {
		probs[i] /= sum
	}

	return probs
}