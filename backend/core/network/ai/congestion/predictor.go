package congestion

import (
	"context"
	"fmt"
	"math"
	"sync"
	"time"
)

// TimeSeriesData represents network time series data
type TimeSeriesData struct {
	Timestamp           time.Time
	BandwidthUtil       float64 // Percentage 0-100
	PacketArrivalRate   float64 // Packets per second
	QueueDepth          int     // Number of packets in queue
	PacketDropRate      float64 // Drops per second
	Latency             float64 // Milliseconds
	TimeOfDay           int     // Hour 0-23
	DayOfWeek           int     // 0-6
	IsBusinessHour      bool
}

// CongestionPredictor uses LSTM for congestion forecasting
type CongestionPredictor struct {
	mu sync.RWMutex

	// LSTM parameters
	inputSize   int
	hiddenSize  int
	outputSize  int
	numLayers   int
	lookback    int // Time steps to look back
	horizon     int // Prediction horizon in seconds

	// LSTM components
	lstm         *LSTMNetwork
	scaler       *DataScaler
	threshold    float64 // Congestion threshold

	// Historical data
	history      []TimeSeriesData
	maxHistory   int
	lastUpdate   time.Time

	// Performance metrics
	predictions  int64
	accuracy     float64
	falsePositives int64
	falseNegatives int64

	// Proactive rerouting
	rerouteCallback func(prediction CongestionPrediction)
}

// LSTMNetwork represents an LSTM network for time series
type LSTMNetwork struct {
	// LSTM cell parameters
	inputWeights  [][]float64
	hiddenWeights [][]float64
	outputWeights [][]float64

	// Gate weights
	forgetGate [][]float64
	inputGate  [][]float64
	outputGate [][]float64
	cellGate   [][]float64

	// Cell state
	cellState   []float64
	hiddenState []float64
}

// DataScaler normalizes data for LSTM
type DataScaler struct {
	mean   []float64
	stdDev []float64
}

// CongestionPrediction represents a congestion forecast
type CongestionPrediction struct {
	LinkID            string
	PredictionTime    time.Time
	PredictedUtil     float64 // Predicted bandwidth utilization
	CongestionProb    float64 // Probability of congestion
	TimeUntilCongestion time.Duration
	Confidence        float64
	RecommendedAction string
}

// NewCongestionPredictor creates a new LSTM-based predictor
func NewCongestionPredictor() *CongestionPredictor {
	return &CongestionPredictor{
		inputSize:  10,  // Number of features
		hiddenSize: 64,  // LSTM hidden units
		outputSize: 1,   // Predict utilization
		numLayers:  2,   // Stacked LSTM layers
		lookback:   60,  // Look at last 60 seconds
		horizon:    60,  // Predict 1 minute ahead
		threshold:  80.0, // 80% utilization = congestion
		maxHistory: 3600, // Keep 1 hour of history
		history:    make([]TimeSeriesData, 0, 3600),
	}
}

// Initialize initializes the LSTM network
func (p *CongestionPredictor) Initialize() error {
	p.mu.Lock()
	defer p.mu.Unlock()

	// Initialize LSTM network
	p.lstm = p.createLSTM()

	// Initialize data scaler
	p.scaler = &DataScaler{
		mean:   make([]float64, p.inputSize),
		stdDev: make([]float64, p.inputSize),
	}

	// Pre-train on synthetic data if no history
	if len(p.history) == 0 {
		p.generateSyntheticData()
		p.pretrain()
	}

	return nil
}

// PredictCongestion predicts congestion for a link
func (p *CongestionPredictor) PredictCongestion(ctx context.Context, linkID string, currentData TimeSeriesData) (CongestionPrediction, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	startTime := time.Now()

	// Add current data to history
	p.history = append(p.history, currentData)
	if len(p.history) > p.maxHistory {
		p.history = p.history[1:]
	}

	// Check if we have enough history
	if len(p.history) < p.lookback {
		return CongestionPrediction{
			LinkID:         linkID,
			PredictionTime: time.Now(),
			Confidence:     0.0,
			RecommendedAction: "insufficient_data",
		}, fmt.Errorf("insufficient history: need %d, have %d", p.lookback, len(p.history))
	}

	// Prepare input sequence
	sequence := p.prepareSequence()

	// Normalize data
	normalizedSeq := p.scaler.transform(sequence)

	// LSTM forward pass
	prediction := p.lstm.predict(normalizedSeq)

	// Denormalize prediction
	predictedUtil := p.scaler.inverse(prediction)[0] * 100 // Convert to percentage

	// Calculate congestion probability
	congestionProb := p.calculateCongestionProbability(predictedUtil)

	// Calculate time until congestion
	timeUntilCongestion := p.estimateTimeUntilCongestion(currentData.BandwidthUtil, predictedUtil)

	// Determine recommended action
	action := p.determineAction(predictedUtil, congestionProb)

	// Update metrics
	p.predictions++

	pred := CongestionPrediction{
		LinkID:              linkID,
		PredictionTime:      time.Now(),
		PredictedUtil:       predictedUtil,
		CongestionProb:      congestionProb,
		TimeUntilCongestion: timeUntilCongestion,
		Confidence:          p.calculateConfidence(len(p.history)),
		RecommendedAction:   action,
	}

	// Trigger proactive rerouting if needed
	if congestionProb > 0.7 && p.rerouteCallback != nil {
		go p.rerouteCallback(pred)
	}

	// Check if prediction took less than 1 second
	if time.Since(startTime) > time.Second {
		return pred, fmt.Errorf("prediction took too long: %v", time.Since(startTime))
	}

	return pred, nil
}

// UpdateModel updates the LSTM model with new data
func (p *CongestionPredictor) UpdateModel(actual float64, predicted float64) {
	p.mu.Lock()
	defer p.mu.Unlock()

	// Calculate error
	error := math.Abs(actual - predicted)

	// Update accuracy (exponential moving average)
	alpha := 0.1
	accuracy := 1.0 - (error / 100.0) // Convert to accuracy
	p.accuracy = p.accuracy*(1-alpha) + accuracy*alpha

	// Update false positive/negative counts
	if predicted > p.threshold && actual <= p.threshold {
		p.falsePositives++
	} else if predicted <= p.threshold && actual > p.threshold {
		p.falseNegatives++
	}

	// Trigger retraining if accuracy drops
	if p.accuracy < 0.9 && len(p.history) >= 100 {
		go p.retrain()
	}
}

// createLSTM creates an LSTM network
func (p *CongestionPredictor) createLSTM() *LSTMNetwork {
	lstm := &LSTMNetwork{
		inputWeights:  p.initializeWeights(p.inputSize, p.hiddenSize),
		hiddenWeights: p.initializeWeights(p.hiddenSize, p.hiddenSize),
		outputWeights: p.initializeWeights(p.hiddenSize, p.outputSize),
		forgetGate:    p.initializeWeights(p.inputSize+p.hiddenSize, p.hiddenSize),
		inputGate:     p.initializeWeights(p.inputSize+p.hiddenSize, p.hiddenSize),
		outputGate:    p.initializeWeights(p.inputSize+p.hiddenSize, p.hiddenSize),
		cellGate:      p.initializeWeights(p.inputSize+p.hiddenSize, p.hiddenSize),
		cellState:     make([]float64, p.hiddenSize),
		hiddenState:   make([]float64, p.hiddenSize),
	}
	return lstm
}

// predict performs LSTM forward pass
func (lstm *LSTMNetwork) predict(sequence [][]float64) float64 {
	// Reset states for new sequence
	lstm.cellState = make([]float64, len(lstm.cellState))
	lstm.hiddenState = make([]float64, len(lstm.hiddenState))

	// Process sequence through LSTM
	for _, input := range sequence {
		// Concatenate input and hidden state
		concat := append(input, lstm.hiddenState...)

		// Forget gate
		forget := lstm.sigmoid(lstm.matmul(concat, lstm.forgetGate))

		// Input gate
		inputG := lstm.sigmoid(lstm.matmul(concat, lstm.inputGate))

		// Candidate values
		candidate := lstm.tanh(lstm.matmul(concat, lstm.cellGate))

		// Update cell state
		for i := range lstm.cellState {
			lstm.cellState[i] = lstm.cellState[i]*forget[i] + inputG[i]*candidate[i]
		}

		// Output gate
		outputG := lstm.sigmoid(lstm.matmul(concat, lstm.outputGate))

		// Update hidden state
		for i := range lstm.hiddenState {
			lstm.hiddenState[i] = outputG[i] * math.Tanh(lstm.cellState[i])
		}
	}

	// Final output layer
	output := lstm.matmul(lstm.hiddenState, lstm.outputWeights)
	return output[0]
}

// Helper functions for LSTM
func (lstm *LSTMNetwork) sigmoid(x []float64) []float64 {
	result := make([]float64, len(x))
	for i, v := range x {
		result[i] = 1.0 / (1.0 + math.Exp(-v))
	}
	return result
}

func (lstm *LSTMNetwork) tanh(x []float64) []float64 {
	result := make([]float64, len(x))
	for i, v := range x {
		result[i] = math.Tanh(v)
	}
	return result
}

func (lstm *LSTMNetwork) matmul(x []float64, w [][]float64) []float64 {
	result := make([]float64, len(w[0]))
	for i := 0; i < len(w[0]); i++ {
		sum := 0.0
		for j := 0; j < len(x) && j < len(w); j++ {
			sum += x[j] * w[j][i]
		}
		result[i] = sum
	}
	return result
}

// prepareSequence prepares input sequence for LSTM
func (p *CongestionPredictor) prepareSequence() [][]float64 {
	start := len(p.history) - p.lookback
	sequence := make([][]float64, p.lookback)

	for i := 0; i < p.lookback; i++ {
		data := p.history[start+i]
		features := []float64{
			data.BandwidthUtil,
			data.PacketArrivalRate,
			float64(data.QueueDepth),
			data.PacketDropRate,
			data.Latency,
			float64(data.TimeOfDay),
			float64(data.DayOfWeek),
			boolToFloat(data.IsBusinessHour),
			// Add derived features
			data.BandwidthUtil * data.PacketArrivalRate, // Interaction
			math.Sin(2 * math.Pi * float64(data.TimeOfDay) / 24), // Cyclic encoding
		}
		sequence[i] = features
	}

	return sequence
}

// calculateCongestionProbability calculates probability of congestion
func (p *CongestionPredictor) calculateCongestionProbability(predictedUtil float64) float64 {
	// Sigmoid function centered at threshold
	x := (predictedUtil - p.threshold) / 10.0
	return 1.0 / (1.0 + math.Exp(-x))
}

// estimateTimeUntilCongestion estimates time until congestion occurs
func (p *CongestionPredictor) estimateTimeUntilCongestion(currentUtil, predictedUtil float64) time.Duration {
	if predictedUtil <= p.threshold {
		return time.Duration(0) // No congestion predicted
	}

	// Linear interpolation
	rate := (predictedUtil - currentUtil) / float64(p.horizon)
	if rate <= 0 {
		return time.Duration(0)
	}

	timeToThreshold := (p.threshold - currentUtil) / rate
	return time.Duration(timeToThreshold) * time.Second
}

// determineAction determines recommended action based on prediction
func (p *CongestionPredictor) determineAction(predictedUtil, congestionProb float64) string {
	if congestionProb > 0.8 {
		return "immediate_reroute"
	} else if congestionProb > 0.6 {
		return "prepare_alternate_path"
	} else if congestionProb > 0.4 {
		return "monitor_closely"
	}
	return "no_action"
}

// initializeWeights initializes weight matrix
func (p *CongestionPredictor) initializeWeights(rows, cols int) [][]float64 {
	weights := make([][]float64, rows)
	scale := math.Sqrt(2.0 / float64(rows))

	for i := range weights {
		weights[i] = make([]float64, cols)
		for j := range weights[i] {
			weights[i][j] = randNorm() * scale
		}
	}

	return weights
}

// Data scaler methods
func (s *DataScaler) transform(data [][]float64) [][]float64 {
	normalized := make([][]float64, len(data))

	for i, row := range data {
		normalized[i] = make([]float64, len(row))
		for j, val := range row {
			if s.stdDev[j] > 0 {
				normalized[i][j] = (val - s.mean[j]) / s.stdDev[j]
			}
		}
	}

	return normalized
}

func (s *DataScaler) inverse(data []float64) []float64 {
	denormalized := make([]float64, len(data))

	for i, val := range data {
		if i < len(s.mean) && i < len(s.stdDev) {
			denormalized[i] = val*s.stdDev[i] + s.mean[i]
		}
	}

	return denormalized
}

// Training methods
func (p *CongestionPredictor) pretrain() {
	// Pretrain on synthetic patterns
	// Implementation depends on specific use case
}

func (p *CongestionPredictor) retrain() {
	// Retrain model with recent data
	// Implementation depends on training framework
}

func (p *CongestionPredictor) generateSyntheticData() {
	// Generate synthetic training data
	for i := 0; i < 1000; i++ {
		p.history = append(p.history, TimeSeriesData{
			Timestamp:         time.Now().Add(time.Duration(i) * time.Second),
			BandwidthUtil:     50 + 30*math.Sin(float64(i)/100),
			PacketArrivalRate: 1000 + 500*math.Cos(float64(i)/50),
			QueueDepth:        int(100 + 50*math.Sin(float64(i)/75)),
			PacketDropRate:    math.Max(0, 10*math.Sin(float64(i)/200)),
			Latency:           20 + 10*math.Cos(float64(i)/150),
			TimeOfDay:         (i / 3600) % 24,
			DayOfWeek:         (i / 86400) % 7,
			IsBusinessHour:    (i/3600)%24 >= 9 && (i/3600)%24 <= 17,
		})
	}
}

// Helper functions
func boolToFloat(b bool) float64 {
	if b {
		return 1.0
	}
	return 0.0
}

func randNorm() float64 {
	// Box-Muller transform for normal distribution
	u1 := 1.0 - math.SmallestNonzeroFloat64
	u2 := 1.0 - math.SmallestNonzeroFloat64
	return math.Sqrt(-2.0*math.Log(u1)) * math.Cos(2.0*math.Pi*u2)
}

func (p *CongestionPredictor) calculateConfidence(historySize int) float64 {
	// Confidence based on history size and accuracy
	historyConf := math.Min(1.0, float64(historySize)/float64(p.maxHistory))
	return historyConf * p.accuracy
}

// SetRerouteCallback sets the callback for proactive rerouting
func (p *CongestionPredictor) SetRerouteCallback(callback func(CongestionPrediction)) {
	p.rerouteCallback = callback
}

// GetMetrics returns predictor metrics
func (p *CongestionPredictor) GetMetrics() map[string]interface{} {
	p.mu.RLock()
	defer p.mu.RUnlock()

	return map[string]interface{}{
		"predictions":     p.predictions,
		"accuracy":        p.accuracy * 100, // Percentage
		"false_positives": p.falsePositives,
		"false_negatives": p.falseNegatives,
		"history_size":    len(p.history),
		"last_update":     p.lastUpdate,
	}
}