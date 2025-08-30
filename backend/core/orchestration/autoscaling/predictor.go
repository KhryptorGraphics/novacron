package autoscaling

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"time"
)

// ARIMAPredictor implements ARIMA (AutoRegressive Integrated Moving Average) prediction
type ARIMAPredictor struct {
	order     ARIMAOrder
	model     *ARIMAModel
	accuracy  float64
	modelInfo ModelInfo
}

// ARIMAOrder represents the ARIMA model order (p, d, q)
type ARIMAOrder struct {
	P int // AutoRegressive order
	D int // Differencing order
	Q int // Moving Average order
}

// ARIMAModel represents the trained ARIMA model
type ARIMAModel struct {
	AR         []float64 // AutoRegressive coefficients
	MA         []float64 // Moving Average coefficients
	Residuals  []float64 // Model residuals
	Mean       float64   // Series mean
	Variance   float64   // Series variance
	trained    bool
}

// NeuralNetworkPredictor implements a simple feedforward neural network
type NeuralNetworkPredictor struct {
	hiddenSize   int
	weights1     [][]float64 // Input to hidden layer weights
	weights2     []float64   // Hidden to output layer weights
	bias1        []float64   // Hidden layer bias
	bias2        float64     // Output layer bias
	accuracy     float64
	modelInfo    ModelInfo
	learningRate float64
}

// NewARIMAPredictor creates a new ARIMA predictor
func NewARIMAPredictor(order ARIMAOrder) *ARIMAPredictor {
	return &ARIMAPredictor{
		order: order,
		model: &ARIMAModel{
			AR: make([]float64, order.P),
			MA: make([]float64, order.Q),
		},
		modelInfo: ModelInfo{
			ModelType: "ARIMA",
			Version:   "1.0.0",
			Parameters: map[string]interface{}{
				"p": order.P,
				"d": order.D,
				"q": order.Q,
			},
		},
	}
}

// NewNeuralNetworkPredictor creates a new neural network predictor
func NewNeuralNetworkPredictor(hiddenSize int) *NeuralNetworkPredictor {
	inputSize := 5 // CPU, Memory, Network, Disk, Time features

	// Initialize weights with small random values
	weights1 := make([][]float64, inputSize)
	for i := range weights1 {
		weights1[i] = make([]float64, hiddenSize)
		for j := range weights1[i] {
			weights1[i][j] = (2*rand.Float64() - 1) * 0.1 // Random between -0.1 and 0.1
		}
	}

	weights2 := make([]float64, hiddenSize)
	for i := range weights2 {
		weights2[i] = (2*rand.Float64() - 1) * 0.1
	}

	bias1 := make([]float64, hiddenSize)
	for i := range bias1 {
		bias1[i] = 0.01
	}

	return &NeuralNetworkPredictor{
		hiddenSize:   hiddenSize,
		weights1:     weights1,
		weights2:     weights2,
		bias1:        bias1,
		bias2:        0.01,
		learningRate: 0.01,
		accuracy:     0.0, // Initialize with 0, will be set after training
		modelInfo: ModelInfo{
			ModelType: "NeuralNetwork",
			Version:   "1.0.0",
			Parameters: map[string]interface{}{
				"hidden_size":    hiddenSize,
				"input_size":     inputSize,
				"learning_rate":  0.01,
			},
		},
	}
}

// Train implements the Predictor interface for ARIMA
func (ap *ARIMAPredictor) Train(data []*MetricsData) error {
	if len(data) < 10 {
		return fmt.Errorf("insufficient data points for training, need at least 10, got %d", len(data))
	}

	// Sort data by timestamp
	sort.Slice(data, func(i, j int) bool {
		return data[i].Timestamp.Before(data[j].Timestamp)
	})

	// Extract CPU usage time series
	series := make([]float64, len(data))
	for i, d := range data {
		series[i] = d.CPUUsage
	}

	// Difference the series if needed
	diffSeries := ap.differencesSeries(series, ap.order.D)

	// Calculate mean and variance
	ap.model.Mean = ap.calculateMean(diffSeries)
	ap.model.Variance = ap.calculateVariance(diffSeries, ap.model.Mean)

	// Fit ARIMA model using least squares approximation
	if err := ap.fitARIMAModel(diffSeries); err != nil {
		return fmt.Errorf("failed to fit ARIMA model: %w", err)
	}

	ap.model.trained = true
	ap.modelInfo.TrainedAt = time.Now()
	ap.modelInfo.DataPoints = len(data)

	// Calculate accuracy (simplified as 1 - normalized RMSE)
	predictions, err := ap.predictSeries(diffSeries, len(diffSeries)/4)
	if err == nil {
		ap.accuracy = ap.calculateAccuracy(diffSeries[len(diffSeries)-len(predictions):], predictions)
	} else {
		ap.accuracy = 0.5 // Default accuracy if prediction fails
	}

	return nil
}

// Predict implements the Predictor interface for ARIMA
func (ap *ARIMAPredictor) Predict(current *MetricsData, horizonMinutes int) (*ResourcePrediction, error) {
	if !ap.model.trained {
		return nil, fmt.Errorf("model not trained")
	}

	// Simple prediction based on trend analysis
	// In a real implementation, this would use the full ARIMA model
	predictedCPU := current.CPUUsage
	confidence := ap.accuracy

	// Add some basic trend analysis
	trendDirection := TrendStable
	if current.CPUUsage > 0.7 {
		predictedCPU = math.Min(1.0, current.CPUUsage*1.1)
		trendDirection = TrendIncreasing
	} else if current.CPUUsage < 0.3 {
		predictedCPU = math.Max(0.0, current.CPUUsage*0.9)
		trendDirection = TrendDecreasing
	}

	// Calculate seasonal factor (simplified)
	hour := time.Now().Hour()
	seasonalFactor := 1.0
	if hour >= 9 && hour <= 17 {
		seasonalFactor = 1.2 // Business hours
	} else if hour >= 0 && hour <= 6 {
		seasonalFactor = 0.8 // Night time
	}

	predictedCPU *= seasonalFactor

	return &ResourcePrediction{
		TargetID:         current.TargetID,
		PredictionTime:   time.Now(),
		HorizonMinutes:   horizonMinutes,
		PredictedCPU:     predictedCPU,
		PredictedMemory:  current.MemoryUsage * (predictedCPU / math.Max(0.01, current.CPUUsage)),
		PredictedLoad:    predictedCPU * seasonalFactor,
		Confidence:       confidence,
		TrendDirection:   trendDirection,
		SeasonalFactor:   seasonalFactor,
		AnomalyScore:     ap.calculateAnomalyScore(current),
		Metadata: map[string]interface{}{
			"model_type": "ARIMA",
			"order":      ap.order,
		},
	}, nil
}

// Train implements the Predictor interface for Neural Network
func (nn *NeuralNetworkPredictor) Train(data []*MetricsData) error {
	if len(data) < 20 {
		return fmt.Errorf("insufficient data points for neural network training, need at least 20, got %d", len(data))
	}

	// Sort data by timestamp
	sort.Slice(data, func(i, j int) bool {
		return data[i].Timestamp.Before(data[j].Timestamp)
	})

	// Prepare training data
	inputs, targets := nn.prepareTrainingData(data)

	// Train the network using gradient descent
	epochs := 1000
	for epoch := 0; epoch < epochs; epoch++ {
		totalError := 0.0

		for i := range inputs {
			// Forward pass
			hidden := nn.forwardHidden(inputs[i])
			output := nn.forwardOutput(hidden)

			// Calculate error
			error := targets[i] - output
			totalError += error * error

			// Backward pass
			nn.backpropagate(inputs[i], hidden, output, error)
		}

		// Early stopping if error is small enough
		if totalError/float64(len(inputs)) < 0.001 {
			break
		}
	}

	nn.modelInfo.TrainedAt = time.Now()
	nn.modelInfo.DataPoints = len(data)

	// Calculate accuracy
	nn.accuracy = nn.calculateNeuralAccuracy(inputs, targets)

	return nil
}

// Predict implements the Predictor interface for Neural Network
func (nn *NeuralNetworkPredictor) Predict(current *MetricsData, horizonMinutes int) (*ResourcePrediction, error) {
	// Prepare input features
	input := nn.prepareInput(current, horizonMinutes)

	// Forward pass
	hidden := nn.forwardHidden(input)
	predictedCPU := nn.forwardOutput(hidden)

	// Ensure prediction is within valid range
	predictedCPU = math.Max(0.0, math.Min(1.0, predictedCPU))

	// Determine trend direction
	var trendDirection TrendDirection
	if predictedCPU > current.CPUUsage*1.05 {
		trendDirection = TrendIncreasing
	} else if predictedCPU < current.CPUUsage*0.95 {
		trendDirection = TrendDecreasing
	} else {
		trendDirection = TrendStable
	}

	// Calculate confidence based on network stability
	confidence := nn.accuracy

	return &ResourcePrediction{
		TargetID:         current.TargetID,
		PredictionTime:   time.Now(),
		HorizonMinutes:   horizonMinutes,
		PredictedCPU:     predictedCPU,
		PredictedMemory:  current.MemoryUsage * (predictedCPU / math.Max(0.01, current.CPUUsage)),
		PredictedLoad:    predictedCPU,
		Confidence:       confidence,
		TrendDirection:   trendDirection,
		SeasonalFactor:   1.0,
		AnomalyScore:     0.0, // Simplified
		Metadata: map[string]interface{}{
			"model_type":   "NeuralNetwork",
			"hidden_size":  nn.hiddenSize,
		},
	}, nil
}

// GetAccuracy returns the model accuracy
func (ap *ARIMAPredictor) GetAccuracy() float64 {
	return ap.accuracy
}

func (nn *NeuralNetworkPredictor) GetAccuracy() float64 {
	return nn.accuracy
}

// GetModelInfo returns model information
func (ap *ARIMAPredictor) GetModelInfo() ModelInfo {
	ap.modelInfo.Accuracy = ap.accuracy
	return ap.modelInfo
}

func (nn *NeuralNetworkPredictor) GetModelInfo() ModelInfo {
	nn.modelInfo.Accuracy = nn.accuracy
	return nn.modelInfo
}

// Helper methods for ARIMA

func (ap *ARIMAPredictor) differencesSeries(series []float64, order int) []float64 {
	result := make([]float64, len(series))
	copy(result, series)

	for d := 0; d < order; d++ {
		newResult := make([]float64, len(result)-1)
		for i := 1; i < len(result); i++ {
			newResult[i-1] = result[i] - result[i-1]
		}
		result = newResult
	}

	return result
}

func (ap *ARIMAPredictor) calculateMean(series []float64) float64 {
	if len(series) == 0 {
		return 0
	}

	sum := 0.0
	for _, v := range series {
		sum += v
	}
	return sum / float64(len(series))
}

func (ap *ARIMAPredictor) calculateVariance(series []float64, mean float64) float64 {
	if len(series) <= 1 {
		return 0
	}

	sum := 0.0
	for _, v := range series {
		sum += (v - mean) * (v - mean)
	}
	return sum / float64(len(series)-1)
}

func (ap *ARIMAPredictor) fitARIMAModel(series []float64) error {
	// Simplified ARIMA fitting using Yule-Walker equations
	// In practice, you would use maximum likelihood estimation

	if len(series) < ap.order.P+ap.order.Q {
		return fmt.Errorf("insufficient data for model order")
	}

	// Initialize AR coefficients with simple autoregression
	for i := 0; i < ap.order.P; i++ {
		if i+1 < len(series) {
			ap.model.AR[i] = 0.5 / float64(i+1) // Simple initialization
		}
	}

	// Initialize MA coefficients
	for i := 0; i < ap.order.Q; i++ {
		ap.model.MA[i] = 0.3 / float64(i+1) // Simple initialization
	}

	return nil
}

func (ap *ARIMAPredictor) predictSeries(series []float64, steps int) ([]float64, error) {
	if len(series) < ap.order.P {
		return nil, fmt.Errorf("insufficient data for prediction")
	}

	predictions := make([]float64, steps)
	extended := append([]float64(nil), series...)

	for i := 0; i < steps; i++ {
		prediction := 0.0

		// AR component
		for j := 0; j < ap.order.P && len(extended)-j-1 >= 0; j++ {
			prediction += ap.model.AR[j] * extended[len(extended)-j-1]
		}

		predictions[i] = prediction
		extended = append(extended, prediction)
	}

	return predictions, nil
}

func (ap *ARIMAPredictor) calculateAccuracy(actual, predicted []float64) float64 {
	if len(actual) != len(predicted) || len(actual) == 0 {
		return 0.0
	}

	mse := 0.0
	for i := range actual {
		diff := actual[i] - predicted[i]
		mse += diff * diff
	}
	mse /= float64(len(actual))

	// Convert MSE to accuracy (0-1 range)
	rmse := math.Sqrt(mse)
	accuracy := 1.0 / (1.0 + rmse)

	return math.Max(0.0, math.Min(1.0, accuracy))
}

func (ap *ARIMAPredictor) calculateAnomalyScore(current *MetricsData) float64 {
	// Simplified anomaly detection based on deviation from expected range
	expected := 0.5 // Simplified expected CPU usage
	deviation := math.Abs(current.CPUUsage - expected)
	anomalyScore := math.Min(1.0, deviation*2)

	return anomalyScore
}

// Helper methods for Neural Network

func (nn *NeuralNetworkPredictor) prepareTrainingData(data []*MetricsData) ([][]float64, []float64) {
	windowSize := 5
	inputs := make([][]float64, 0)
	targets := make([]float64, 0)

	for i := windowSize; i < len(data); i++ {
		input := nn.prepareInput(data[i-1], 30) // 30 minute horizon
		target := data[i].CPUUsage

		inputs = append(inputs, input)
		targets = append(targets, target)
	}

	return inputs, targets
}

func (nn *NeuralNetworkPredictor) prepareInput(current *MetricsData, horizonMinutes int) []float64 {
	// Normalize inputs to 0-1 range
	input := []float64{
		current.CPUUsage,
		current.MemoryUsage,
		current.NetworkIO / 1000.0,    // Normalize network IO
		current.DiskIO / 1000.0,       // Normalize disk IO
		float64(horizonMinutes) / 60.0, // Normalize horizon to hours
	}

	return input
}

func (nn *NeuralNetworkPredictor) forwardHidden(input []float64) []float64 {
	hidden := make([]float64, nn.hiddenSize)

	for i := 0; i < nn.hiddenSize; i++ {
		sum := nn.bias1[i]
		for j, x := range input {
			sum += nn.weights1[j][i] * x
		}
		hidden[i] = nn.sigmoid(sum)
	}

	return hidden
}

func (nn *NeuralNetworkPredictor) forwardOutput(hidden []float64) float64 {
	sum := nn.bias2
	for i, h := range hidden {
		sum += nn.weights2[i] * h
	}

	return nn.sigmoid(sum)
}

func (nn *NeuralNetworkPredictor) backpropagate(input, hidden []float64, output, error float64) {
	// Output layer gradient
	outputGrad := error * nn.sigmoidDerivative(output)

	// Update output weights and bias
	for i := range nn.weights2 {
		nn.weights2[i] += nn.learningRate * outputGrad * hidden[i]
	}
	nn.bias2 += nn.learningRate * outputGrad

	// Hidden layer gradients
	hiddenGrads := make([]float64, nn.hiddenSize)
	for i := 0; i < nn.hiddenSize; i++ {
		hiddenGrads[i] = outputGrad * nn.weights2[i] * nn.sigmoidDerivative(hidden[i])
	}

	// Update hidden layer weights and biases
	for i := 0; i < len(input); i++ {
		for j := 0; j < nn.hiddenSize; j++ {
			nn.weights1[i][j] += nn.learningRate * hiddenGrads[j] * input[i]
		}
	}

	for i := 0; i < nn.hiddenSize; i++ {
		nn.bias1[i] += nn.learningRate * hiddenGrads[i]
	}
}

func (nn *NeuralNetworkPredictor) sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func (nn *NeuralNetworkPredictor) sigmoidDerivative(x float64) float64 {
	return x * (1.0 - x)
}

func (nn *NeuralNetworkPredictor) calculateNeuralAccuracy(inputs [][]float64, targets []float64) float64 {
	correct := 0
	total := len(inputs)

	for i, input := range inputs {
		hidden := nn.forwardHidden(input)
		prediction := nn.forwardOutput(hidden)

		// Consider prediction correct if within 10% of target
		if math.Abs(prediction-targets[i]) < 0.1 {
			correct++
		}
	}

	return float64(correct) / float64(total)
}