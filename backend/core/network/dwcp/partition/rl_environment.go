package partition

import (
	"math"
	"sync"
	"time"
)

// EnvironmentState represents the current state of the DWCP network for RL decision making
type EnvironmentState struct {
	// Stream metrics (4 streams × multiple features)
	StreamBandwidth   [4]float64 // Current bandwidth per stream (Mbps)
	StreamLatency     [4]float64 // Current latency per stream (ms)
	StreamCongestion  [4]float64 // Congestion level (0-1)
	StreamSuccessRate [4]float64 // Historical success rate (0-1)

	// Task characteristics
	TaskQueueDepth int     // Number of tasks waiting
	TaskSize       int     // Size of current task (bytes)
	TaskPriority   float64 // Priority level (0-1)

	// Temporal features
	TimeOfDay float64 // Hour as fraction (0-1)

	// Additional context
	CPUUtilization    float64
	MemoryUtilization float64
	NetworkLoad       float64

	mu sync.RWMutex
}

// NewEnvironmentState creates a new environment state with current metrics
func NewEnvironmentState() *EnvironmentState {
	return &EnvironmentState{
		StreamBandwidth:   [4]float64{100, 100, 100, 100}, // Default 100 Mbps
		StreamLatency:     [4]float64{10, 10, 10, 10},     // Default 10ms
		StreamCongestion:  [4]float64{0, 0, 0, 0},
		StreamSuccessRate: [4]float64{1, 1, 1, 1},
		TaskQueueDepth:    0,
		TaskSize:          0,
		TaskPriority:      0.5,
		TimeOfDay:         float64(time.Now().Hour()) / 24.0,
	}
}

// ToVector converts state to a float32 vector for neural network input
func (es *EnvironmentState) ToVector() []float32 {
	es.mu.RLock()
	defer es.mu.RUnlock()

	vector := make([]float32, 20)

	// Stream metrics (16 features)
	for i := 0; i < 4; i++ {
		vector[i] = float32(es.StreamBandwidth[i] / 1000.0)    // Normalize to 0-1 (assuming max 1Gbps)
		vector[4+i] = float32(es.StreamLatency[i] / 100.0)     // Normalize to 0-1 (assuming max 100ms)
		vector[8+i] = float32(es.StreamCongestion[i])
		vector[12+i] = float32(es.StreamSuccessRate[i])
	}

	// Task features (3 features)
	vector[16] = float32(math.Min(float64(es.TaskQueueDepth)/100.0, 1.0)) // Cap at 100
	vector[17] = float32(math.Min(float64(es.TaskSize)/1e9, 1.0))         // Normalize to GB
	vector[18] = float32(es.TaskPriority)

	// Temporal feature
	vector[19] = float32(es.TimeOfDay)

	return vector
}

// Action represents a partitioning decision
type Action int

const (
	// Single stream actions (0-3)
	ActionStream1 Action = iota
	ActionStream2
	ActionStream3
	ActionStream4

	// Two-stream split actions (4-9)
	ActionSplit12
	ActionSplit13
	ActionSplit14
	ActionSplit23
	ActionSplit24
	ActionSplit34

	// Three-stream split actions (10-13)
	ActionSplit123
	ActionSplit124
	ActionSplit134
	ActionSplit234

	// Four-stream split action (14)
	ActionSplitAll

	NumActions = 15
)

// RewardCalculator calculates reward for a given action and outcome
type RewardCalculator struct {
	// Reward weights
	AlphaThroughput   float64
	BetaLatency       float64
	GammaImbalance    float64
	DeltaCompletion   float64
	EpsilonRetransmit float64
}

// NewRewardCalculator creates a reward calculator with default weights
func NewRewardCalculator() *RewardCalculator {
	return &RewardCalculator{
		AlphaThroughput:   1.0,
		BetaLatency:       0.5,
		GammaImbalance:    0.3,
		DeltaCompletion:   2.0,
		EpsilonRetransmit: 1.0,
	}
}

// Calculate computes the reward for an action outcome
func (rc *RewardCalculator) Calculate(outcome *ActionOutcome) float64 {
	throughputImprovement := (outcome.ActualThroughput - outcome.BaselineThroughput) / outcome.BaselineThroughput
	latencyPenalty := math.Max(0, (outcome.ActualLatency-outcome.TargetLatency)/outcome.TargetLatency)
	imbalance := outcome.StreamImbalance

	reward := rc.AlphaThroughput*throughputImprovement -
		rc.BetaLatency*latencyPenalty -
		rc.GammaImbalance*imbalance

	if outcome.Completed {
		reward += rc.DeltaCompletion
	}

	if outcome.Retransmissions > 0 {
		reward -= rc.EpsilonRetransmit * float64(outcome.Retransmissions)
	}

	return reward
}

// ActionOutcome represents the result of executing an action
type ActionOutcome struct {
	ActualThroughput   float64
	BaselineThroughput float64
	ActualLatency      float64
	TargetLatency      float64
	StreamImbalance    float64
	Completed          bool
	Retransmissions    int
}

// Experience represents a single RL experience tuple
type Experience struct {
	State     []float32
	Action    Action
	Reward    float64
	NextState []float32
	Done      bool
	TDError   float64 // For prioritized replay
}

// ReplayBuffer stores experiences for training
type ReplayBuffer struct {
	buffer   []*Experience
	capacity int
	mu       sync.Mutex
}

// NewReplayBuffer creates a new replay buffer
func NewReplayBuffer(capacity int) *ReplayBuffer {
	return &ReplayBuffer{
		buffer:   make([]*Experience, 0, capacity),
		capacity: capacity,
	}
}

// Add adds an experience to the buffer
func (rb *ReplayBuffer) Add(exp *Experience) {
	rb.mu.Lock()
	defer rb.mu.Unlock()

	if len(rb.buffer) >= rb.capacity {
		rb.buffer = rb.buffer[1:] // Remove oldest
	}
	rb.buffer = append(rb.buffer, exp)
}

// Sample randomly samples experiences from the buffer
func (rb *ReplayBuffer) Sample(batchSize int) []*Experience {
	rb.mu.Lock()
	defer rb.mu.Unlock()

	if len(rb.buffer) < batchSize {
		return rb.buffer
	}

	// Simple random sampling (can be enhanced with prioritized replay)
	sampled := make([]*Experience, batchSize)
	indices := make(map[int]bool)

	for i := 0; i < batchSize; {
		idx := int(time.Now().UnixNano()) % len(rb.buffer)
		if !indices[idx] {
			sampled[i] = rb.buffer[idx]
			indices[idx] = true
			i++
		}
	}

	return sampled
}

// Size returns the current size of the buffer
func (rb *ReplayBuffer) Size() int {
	rb.mu.Lock()
	defer rb.mu.Unlock()
	return len(rb.buffer)
}

// EnvironmentSimulator simulates the network environment for training
type EnvironmentSimulator struct {
	baseLatency    float64
	baseBandwidth  float64
	congestionProb float64
	state          *EnvironmentState
	rewardCalc     *RewardCalculator
}

// NewEnvironmentSimulator creates a new environment simulator
func NewEnvironmentSimulator() *EnvironmentSimulator {
	return &EnvironmentSimulator{
		baseLatency:    10.0,  // 10ms base latency
		baseBandwidth:  100.0, // 100 Mbps base bandwidth
		congestionProb: 0.1,   // 10% chance of congestion
		state:          NewEnvironmentState(),
		rewardCalc:     NewRewardCalculator(),
	}
}

// Reset resets the environment to initial state
func (es *EnvironmentSimulator) Reset() *EnvironmentState {
	es.state = NewEnvironmentState()

	// Add some randomness to initial state
	for i := 0; i < 4; i++ {
		es.state.StreamBandwidth[i] = es.baseBandwidth * (0.8 + 0.4*randomFloat())
		es.state.StreamLatency[i] = es.baseLatency * (0.8 + 0.4*randomFloat())
		es.state.StreamCongestion[i] = randomFloat() * 0.3
		es.state.StreamSuccessRate[i] = 0.9 + 0.1*randomFloat()
	}

	es.state.TaskQueueDepth = int(randomFloat() * 50)
	es.state.TaskSize = int(randomFloat() * 1e9) // Up to 1GB
	es.state.TaskPriority = randomFloat()

	return es.state
}

// Step executes an action and returns the next state, reward, and done flag
func (es *EnvironmentSimulator) Step(action Action) (*EnvironmentState, float64, bool) {
	// Simulate action execution
	outcome := es.simulateAction(action)

	// Calculate reward
	reward := es.rewardCalc.Calculate(outcome)

	// Update state based on action outcome
	es.updateState(action, outcome)

	// Check if episode is done
	done := es.state.TaskQueueDepth == 0

	return es.state, reward, done
}

func (es *EnvironmentSimulator) simulateAction(action Action) *ActionOutcome {
	outcome := &ActionOutcome{
		BaselineThroughput: es.baseBandwidth,
		TargetLatency:      es.baseLatency,
	}

	// Simulate based on action type
	switch {
	case action <= ActionStream4:
		// Single stream
		streamIdx := int(action)
		outcome.ActualThroughput = es.state.StreamBandwidth[streamIdx] * es.state.StreamSuccessRate[streamIdx]
		outcome.ActualLatency = es.state.StreamLatency[streamIdx] * (1 + es.state.StreamCongestion[streamIdx])
		outcome.StreamImbalance = 0 // No imbalance for single stream

	case action <= ActionSplit34:
		// Two-stream split
		streams := getTwoStreamIndices(action)
		throughput := 0.0
		latency := 0.0
		for _, idx := range streams {
			throughput += es.state.StreamBandwidth[idx] * es.state.StreamSuccessRate[idx] * 0.5
			latency = math.Max(latency, es.state.StreamLatency[idx]*(1+es.state.StreamCongestion[idx]))
		}
		outcome.ActualThroughput = throughput
		outcome.ActualLatency = latency
		outcome.StreamImbalance = calculateImbalance(es.state, streams)

	case action <= ActionSplit234:
		// Three-stream split
		streams := getThreeStreamIndices(action)
		throughput := 0.0
		latency := 0.0
		for _, idx := range streams {
			throughput += es.state.StreamBandwidth[idx] * es.state.StreamSuccessRate[idx] * 0.33
			latency = math.Max(latency, es.state.StreamLatency[idx]*(1+es.state.StreamCongestion[idx]))
		}
		outcome.ActualThroughput = throughput
		outcome.ActualLatency = latency
		outcome.StreamImbalance = calculateImbalance(es.state, streams)

	case action == ActionSplitAll:
		// Four-stream split
		throughput := 0.0
		latency := 0.0
		for i := 0; i < 4; i++ {
			throughput += es.state.StreamBandwidth[i] * es.state.StreamSuccessRate[i] * 0.25
			latency = math.Max(latency, es.state.StreamLatency[i]*(1+es.state.StreamCongestion[i]))
		}
		outcome.ActualThroughput = throughput
		outcome.ActualLatency = latency
		outcome.StreamImbalance = calculateImbalance(es.state, []int{0, 1, 2, 3})
	}

	// Simulate completion and retransmissions
	outcome.Completed = randomFloat() > 0.1 // 90% success rate
	if !outcome.Completed {
		outcome.Retransmissions = int(randomFloat()*3) + 1
	}

	return outcome
}

func (es *EnvironmentSimulator) updateState(action Action, outcome *ActionOutcome) {
	// Update stream metrics based on usage
	usedStreams := getUsedStreams(action)

	for _, idx := range usedStreams {
		// Increase congestion on used streams
		es.state.StreamCongestion[idx] = math.Min(1.0, es.state.StreamCongestion[idx]+0.1)

		// Update success rate based on outcome
		if outcome.Completed {
			es.state.StreamSuccessRate[idx] = 0.95*es.state.StreamSuccessRate[idx] + 0.05
		} else {
			es.state.StreamSuccessRate[idx] = 0.95*es.state.StreamSuccessRate[idx] + 0.05*0.5
		}
	}

	// Decay congestion on unused streams
	for i := 0; i < 4; i++ {
		if !contains(usedStreams, i) {
			es.state.StreamCongestion[i] = math.Max(0, es.state.StreamCongestion[i]-0.05)
		}
	}

	// Update task queue
	if outcome.Completed && es.state.TaskQueueDepth > 0 {
		es.state.TaskQueueDepth--

		// Generate new task if queue not empty
		if es.state.TaskQueueDepth > 0 {
			es.state.TaskSize = int(randomFloat() * 1e9)
			es.state.TaskPriority = randomFloat()
		}
	}

	// Update time
	es.state.TimeOfDay = float64(time.Now().Hour()) / 24.0
}

// Helper functions

func randomFloat() float64 {
	return float64(time.Now().UnixNano()%1000) / 1000.0
}

func getTwoStreamIndices(action Action) []int {
	switch action {
	case ActionSplit12:
		return []int{0, 1}
	case ActionSplit13:
		return []int{0, 2}
	case ActionSplit14:
		return []int{0, 3}
	case ActionSplit23:
		return []int{1, 2}
	case ActionSplit24:
		return []int{1, 3}
	case ActionSplit34:
		return []int{2, 3}
	default:
		return []int{0, 1}
	}
}

func getThreeStreamIndices(action Action) []int {
	switch action {
	case ActionSplit123:
		return []int{0, 1, 2}
	case ActionSplit124:
		return []int{0, 1, 3}
	case ActionSplit134:
		return []int{0, 2, 3}
	case ActionSplit234:
		return []int{1, 2, 3}
	default:
		return []int{0, 1, 2}
	}
}

func getUsedStreams(action Action) []int {
	if action <= ActionStream4 {
		return []int{int(action)}
	} else if action <= ActionSplit34 {
		return getTwoStreamIndices(action)
	} else if action <= ActionSplit234 {
		return getThreeStreamIndices(action)
	} else {
		return []int{0, 1, 2, 3}
	}
}

func calculateImbalance(state *EnvironmentState, streams []int) float64 {
	if len(streams) <= 1 {
		return 0
	}

	var loads []float64
	for _, idx := range streams {
		load := state.StreamCongestion[idx] / state.StreamBandwidth[idx]
		loads = append(loads, load)
	}

	// Calculate standard deviation of loads
	mean := 0.0
	for _, l := range loads {
		mean += l
	}
	mean /= float64(len(loads))

	variance := 0.0
	for _, l := range loads {
		diff := l - mean
		variance += diff * diff
	}
	variance /= float64(len(loads))

	return math.Sqrt(variance)
}

func contains(slice []int, item int) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}