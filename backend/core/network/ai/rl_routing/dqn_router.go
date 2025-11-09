package rl_routing

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"
)

// State represents the network state for routing decision
type State struct {
	CurrentNode   string
	DestNode      string
	LinkLatencies map[string]float64 // Historical latencies
	LinkBandwidth map[string]float64 // Available bandwidth
	PacketPriority int
	QueueDepths   map[string]int
	TimeOfDay     int // Hour of day
}

// Action represents a routing decision (next hop)
type Action struct {
	NextHop string
	LinkID  string
}

// Experience for replay buffer
type Experience struct {
	State      State
	Action     Action
	Reward     float64
	NextState  State
	Done       bool
	Timestamp  time.Time
}

// DQNRouter implements Deep Q-Network for routing decisions
type DQNRouter struct {
	mu sync.RWMutex

	// Network parameters
	inputSize    int
	hiddenSize   int
	outputSize   int
	learningRate float64
	epsilon      float64 // Exploration rate
	gamma        float64 // Discount factor

	// DQN components
	qNetwork       *NeuralNetwork
	targetNetwork  *NeuralNetwork
	replayBuffer   []Experience
	bufferCapacity int
	batchSize      int
	updateFreq     int
	stepCount      int

	// Performance tracking
	avgDecisionTime time.Duration
	decisionCount   int64
	successRate     float64

	// Network topology
	topology map[string][]string // node -> neighbors
	links    map[string]*LinkInfo
}

// LinkInfo contains link characteristics
type LinkInfo struct {
	Source      string
	Destination string
	Bandwidth   float64
	Latency     float64
	PacketLoss  float64
	Utilization float64
}

// NeuralNetwork represents a simple feedforward network
type NeuralNetwork struct {
	weights1 [][]float64
	bias1    []float64
	weights2 [][]float64
	bias2    []float64
}

// NewDQNRouter creates a new DQN-based router
func NewDQNRouter(topology map[string][]string) *DQNRouter {
	return &DQNRouter{
		topology:       topology,
		links:          make(map[string]*LinkInfo),
		inputSize:      64,  // State features
		hiddenSize:     128, // Hidden layer size
		outputSize:     32,  // Max possible next hops
		learningRate:   0.001,
		epsilon:        0.1,
		gamma:          0.99,
		bufferCapacity: 10000,
		batchSize:      32,
		updateFreq:     100,
		replayBuffer:   make([]Experience, 0, 10000),
	}
}

// Initialize initializes the DQN networks
func (r *DQNRouter) Initialize() error {
	r.mu.Lock()
	defer r.mu.Unlock()

	// Initialize Q-network
	r.qNetwork = r.createNetwork()
	// Initialize target network with same weights
	r.targetNetwork = r.copyNetwork(r.qNetwork)

	return nil
}

// MakeRoutingDecision makes a routing decision using DQN
func (r *DQNRouter) MakeRoutingDecision(ctx context.Context, state State) (Action, error) {
	start := time.Now()
	defer func() {
		r.updateDecisionMetrics(time.Since(start))
	}()

	// Check context deadline for <1ms requirement
	deadline := time.Now().Add(900 * time.Microsecond) // Target: <1ms
	routeCtx, cancel := context.WithDeadline(ctx, deadline)
	defer cancel()

	// Epsilon-greedy exploration
	if rand.Float64() < r.epsilon {
		return r.exploreAction(state), nil
	}

	// Exploit: use Q-network for decision
	action, err := r.exploitAction(routeCtx, state)
	if err != nil {
		// Fallback to shortest path if DQN fails
		return r.fallbackRoute(state), nil
	}

	return action, nil
}

// exploreAction randomly selects an action for exploration
func (r *DQNRouter) exploreAction(state State) Action {
	r.mu.RLock()
	neighbors := r.topology[state.CurrentNode]
	r.mu.RUnlock()

	if len(neighbors) == 0 {
		return Action{NextHop: state.CurrentNode}
	}

	// Random selection
	nextHop := neighbors[rand.Intn(len(neighbors))]
	linkID := fmt.Sprintf("%s-%s", state.CurrentNode, nextHop)

	return Action{
		NextHop: nextHop,
		LinkID:  linkID,
	}
}

// exploitAction uses Q-network to select best action
func (r *DQNRouter) exploitAction(ctx context.Context, state State) (Action, error) {
	// Convert state to feature vector
	features := r.stateToFeatures(state)

	// Forward pass through Q-network
	qValues := r.qNetwork.forward(features)

	// Get valid actions (neighbors)
	r.mu.RLock()
	neighbors := r.topology[state.CurrentNode]
	r.mu.RUnlock()

	if len(neighbors) == 0 {
		return Action{}, fmt.Errorf("no neighbors found")
	}

	// Select action with highest Q-value
	bestAction := Action{NextHop: neighbors[0]}
	maxQ := qValues[0]

	for i, neighbor := range neighbors {
		if i < len(qValues) && qValues[i] > maxQ {
			maxQ = qValues[i]
			bestAction = Action{
				NextHop: neighbor,
				LinkID:  fmt.Sprintf("%s-%s", state.CurrentNode, neighbor),
			}
		}
	}

	return bestAction, nil
}

// Train trains the DQN with experience replay
func (r *DQNRouter) Train() {
	r.mu.Lock()
	defer r.mu.Unlock()

	if len(r.replayBuffer) < r.batchSize {
		return
	}

	// Sample batch from replay buffer
	batch := r.sampleBatch()

	// Prepare training data
	states := make([][]float64, r.batchSize)
	targets := make([][]float64, r.batchSize)

	for i, exp := range batch {
		states[i] = r.stateToFeatures(exp.State)

		// Calculate target Q-values
		currentQ := r.qNetwork.forward(states[i])
		targets[i] = currentQ

		if exp.Done {
			// Terminal state
			actionIndex := r.actionToIndex(exp.Action)
			targets[i][actionIndex] = exp.Reward
		} else {
			// Non-terminal: use target network for stability
			nextFeatures := r.stateToFeatures(exp.NextState)
			nextQ := r.targetNetwork.forward(nextFeatures)
			maxNextQ := r.maxFloat64(nextQ)

			actionIndex := r.actionToIndex(exp.Action)
			targets[i][actionIndex] = exp.Reward + r.gamma*maxNextQ
		}
	}

	// Update Q-network
	r.qNetwork.train(states, targets, r.learningRate)

	// Update step count
	r.stepCount++

	// Periodically update target network
	if r.stepCount%r.updateFreq == 0 {
		r.targetNetwork = r.copyNetwork(r.qNetwork)
	}
}

// AddExperience adds an experience to replay buffer
func (r *DQNRouter) AddExperience(exp Experience) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.replayBuffer = append(r.replayBuffer, exp)

	// Maintain buffer capacity
	if len(r.replayBuffer) > r.bufferCapacity {
		r.replayBuffer = r.replayBuffer[1:]
	}
}

// CalculateReward calculates reward for a routing decision
func (r *DQNRouter) CalculateReward(action Action, latency float64, packetLoss float64, bandwidth float64) float64 {
	// Reward function components:
	// - Negative latency (minimize)
	// - Negative packet loss (minimize)
	// - Bandwidth efficiency bonus (maximize utilization)

	latencyPenalty := -latency / 1000.0 // Convert to seconds
	lossPenalty := -packetLoss * 10.0   // Heavy penalty for loss
	bandwidthBonus := bandwidth / 1000.0 // Gbps bonus

	// Weighted sum
	reward := latencyPenalty*0.5 + lossPenalty*0.3 + bandwidthBonus*0.2

	return reward
}

// stateToFeatures converts state to feature vector
func (r *DQNRouter) stateToFeatures(state State) []float64 {
	features := make([]float64, r.inputSize)
	idx := 0

	// Node features (one-hot encoding simplified)
	features[idx] = float64(len(state.CurrentNode))
	idx++
	features[idx] = float64(len(state.DestNode))
	idx++

	// Link latencies (normalized)
	for _, latency := range state.LinkLatencies {
		if idx < len(features) {
			features[idx] = latency / 1000.0 // Normalize to seconds
			idx++
		}
	}

	// Link bandwidth (normalized)
	for _, bw := range state.LinkBandwidth {
		if idx < len(features) {
			features[idx] = bw / 10000.0 // Normalize to 10Gbps
			idx++
		}
	}

	// Packet priority
	if idx < len(features) {
		features[idx] = float64(state.PacketPriority) / 10.0
		idx++
	}

	// Queue depths
	for _, depth := range state.QueueDepths {
		if idx < len(features) {
			features[idx] = float64(depth) / 1000.0
			idx++
		}
	}

	// Time of day (cyclic encoding)
	if idx < len(features) {
		features[idx] = math.Sin(2 * math.Pi * float64(state.TimeOfDay) / 24.0)
		idx++
		if idx < len(features) {
			features[idx] = math.Cos(2 * math.Pi * float64(state.TimeOfDay) / 24.0)
		}
	}

	return features
}

// createNetwork creates a new neural network
func (r *DQNRouter) createNetwork() *NeuralNetwork {
	// Initialize weights with Xavier initialization
	nn := &NeuralNetwork{
		weights1: make([][]float64, r.inputSize),
		bias1:    make([]float64, r.hiddenSize),
		weights2: make([][]float64, r.hiddenSize),
		bias2:    make([]float64, r.outputSize),
	}

	// Initialize weights
	for i := 0; i < r.inputSize; i++ {
		nn.weights1[i] = make([]float64, r.hiddenSize)
		for j := 0; j < r.hiddenSize; j++ {
			nn.weights1[i][j] = rand.NormFloat64() * math.Sqrt(2.0/float64(r.inputSize))
		}
	}

	for i := 0; i < r.hiddenSize; i++ {
		nn.weights2[i] = make([]float64, r.outputSize)
		for j := 0; j < r.outputSize; j++ {
			nn.weights2[i][j] = rand.NormFloat64() * math.Sqrt(2.0/float64(r.hiddenSize))
		}
	}

	return nn
}

// forward performs forward pass through network
func (nn *NeuralNetwork) forward(input []float64) []float64 {
	// Hidden layer
	hidden := make([]float64, len(nn.bias1))
	for i := range hidden {
		sum := nn.bias1[i]
		for j, x := range input {
			if j < len(nn.weights1) && i < len(nn.weights1[j]) {
				sum += x * nn.weights1[j][i]
			}
		}
		hidden[i] = relu(sum)
	}

	// Output layer
	output := make([]float64, len(nn.bias2))
	for i := range output {
		sum := nn.bias2[i]
		for j, h := range hidden {
			if j < len(nn.weights2) && i < len(nn.weights2[j]) {
				sum += h * nn.weights2[j][i]
			}
		}
		output[i] = sum // Linear activation for Q-values
	}

	return output
}

// train updates network weights
func (nn *NeuralNetwork) train(inputs [][]float64, targets [][]float64, lr float64) {
	// Simplified training - in production, use proper backpropagation
	// This is a placeholder for demonstration
	for i := range inputs {
		output := nn.forward(inputs[i])

		// Calculate gradients and update weights
		for j := range output {
			error := targets[i][j] - output[j]
			// Update output layer weights
			for k := range nn.weights2 {
				if j < len(nn.weights2[k]) {
					nn.weights2[k][j] += lr * error * 0.01 // Simplified update
				}
			}
			nn.bias2[j] += lr * error * 0.01
		}
	}
}

// Helper functions
func (r *DQNRouter) copyNetwork(src *NeuralNetwork) *NeuralNetwork {
	dst := &NeuralNetwork{
		weights1: make([][]float64, len(src.weights1)),
		bias1:    make([]float64, len(src.bias1)),
		weights2: make([][]float64, len(src.weights2)),
		bias2:    make([]float64, len(src.bias2)),
	}

	for i := range src.weights1 {
		dst.weights1[i] = make([]float64, len(src.weights1[i]))
		copy(dst.weights1[i], src.weights1[i])
	}
	copy(dst.bias1, src.bias1)

	for i := range src.weights2 {
		dst.weights2[i] = make([]float64, len(src.weights2[i]))
		copy(dst.weights2[i], src.weights2[i])
	}
	copy(dst.bias2, src.bias2)

	return dst
}

func (r *DQNRouter) sampleBatch() []Experience {
	batch := make([]Experience, r.batchSize)
	indices := rand.Perm(len(r.replayBuffer))

	for i := 0; i < r.batchSize && i < len(indices); i++ {
		batch[i] = r.replayBuffer[indices[i]]
	}

	return batch
}

func (r *DQNRouter) actionToIndex(action Action) int {
	// Map action to index - simplified
	return len(action.NextHop) % r.outputSize
}

func (r *DQNRouter) fallbackRoute(state State) Action {
	// Simple shortest path fallback
	return Action{
		NextHop: state.DestNode,
		LinkID:  fmt.Sprintf("%s-%s", state.CurrentNode, state.DestNode),
	}
}

func (r *DQNRouter) updateDecisionMetrics(duration time.Duration) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.decisionCount++
	// Exponential moving average
	alpha := 0.1
	r.avgDecisionTime = time.Duration(float64(r.avgDecisionTime)*(1-alpha) + float64(duration)*alpha)
}

func (r *DQNRouter) maxFloat64(slice []float64) float64 {
	if len(slice) == 0 {
		return 0
	}
	max := slice[0]
	for _, v := range slice[1:] {
		if v > max {
			max = v
		}
	}
	return max
}

func relu(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

// GetMetrics returns routing metrics
func (r *DQNRouter) GetMetrics() map[string]interface{} {
	r.mu.RLock()
	defer r.mu.RUnlock()

	return map[string]interface{}{
		"avg_decision_time_us": r.avgDecisionTime.Microseconds(),
		"decision_count":       r.decisionCount,
		"success_rate":         r.successRate,
		"epsilon":              r.epsilon,
		"replay_buffer_size":   len(r.replayBuffer),
		"step_count":           r.stepCount,
	}
}