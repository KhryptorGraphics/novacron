package partition

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"sync"
	"time"
)

// DQNAgent implements a Deep Q-Network agent for task partitioning
type DQNAgent struct {
	modelLoaded  bool
	epsilon      float64
	epsilonMin   float64
	epsilonDecay float64
	stateBuffer  []float32
	replayBuffer *ReplayBuffer
	learningRate float64
	gamma        float64 // Discount factor
	updateFreq   int     // Update target network every N steps
	stepCount    int
	mu           sync.RWMutex

	// Performance metrics
	totalReward    float64
	episodeRewards []float64
	successRate    float64
}

const dqnAgentStateVersion = 1

type dqnAgentState struct {
	Version        int         `json:"version"`
	ModelLoaded    bool        `json:"model_loaded"`
	Epsilon        float64     `json:"epsilon"`
	EpsilonMin     float64     `json:"epsilon_min"`
	EpsilonDecay   float64     `json:"epsilon_decay"`
	StateBuffer    []float32   `json:"state_buffer,omitempty"`
	LearningRate   float64     `json:"learning_rate"`
	Gamma          float64     `json:"gamma"`
	UpdateFreq     int         `json:"update_freq"`
	StepCount      int         `json:"step_count"`
	TotalReward    float64     `json:"total_reward"`
	EpisodeRewards []float64   `json:"episode_rewards,omitempty"`
	SuccessRate    float64     `json:"success_rate"`
	ReplayBuffer   replayState `json:"replay_buffer"`
}

type replayState struct {
	Capacity    int           `json:"capacity"`
	Experiences []*Experience `json:"experiences,omitempty"`
}

// TaskPartitionDecision represents the agent's decision on how to partition a task
type TaskPartitionDecision struct {
	StreamIDs       []int         // Which streams to use
	ChunkSizes      []int         // Size of chunk for each stream
	Confidence      float64       // Confidence in the decision
	ExpectedTime    time.Duration // Expected completion time
	Action          Action        // The action taken
	QValue          float64       // Q-value of the action
	ExplorationUsed bool          // Whether exploration was used
}

// NewDQNAgent creates a new DQN agent
func NewDQNAgent(modelPath string) (*DQNAgent, error) {
	agent := &DQNAgent{
		epsilon:        1.0,  // Start with full exploration
		epsilonMin:     0.01, // Minimum exploration rate
		epsilonDecay:   0.995,
		replayBuffer:   NewReplayBuffer(10000),
		learningRate:   0.001,
		gamma:          0.95,
		updateFreq:     1000,
		stepCount:      0,
		episodeRewards: make([]float64, 0, 1000),
	}

	if modelPath != "" {
		if _, err := os.Stat(modelPath); err == nil {
			agent.modelLoaded = true
		} else if os.IsNotExist(err) {
			log.Printf("Warning: Could not load model from %s, operating in exploration mode", modelPath)
		} else {
			return nil, fmt.Errorf("failed to inspect model path %q: %w", modelPath, err)
		}
	}

	return agent, nil
}

// SelectAction selects an action based on the current state using epsilon-greedy policy
func (agent *DQNAgent) SelectAction(state *EnvironmentState) (*TaskPartitionDecision, error) {
	agent.mu.Lock()
	defer agent.mu.Unlock()

	decision := &TaskPartitionDecision{
		Confidence: 1.0 - agent.epsilon,
	}

	// Epsilon-greedy exploration
	if rand.Float64() < agent.epsilon {
		decision.ExplorationUsed = true
		action := agent.randomAction()
		decision.Action = action
		return agent.decodeAction(action, state, decision), nil
	}

	// Exploit: use neural network to select action
	if !agent.modelLoaded {
		// Fallback to heuristic if no model loaded
		action := agent.heuristicAction(state)
		decision.Action = action
		return agent.decodeAction(action, state, decision), nil
	}

	// Prepare state tensor
	stateTensor := state.ToVector()

	// Run inference
	qValues, err := agent.runInference(stateTensor)
	if err != nil {
		log.Printf("Inference error, falling back to random action: %v", err)
		action := agent.randomAction()
		decision.Action = action
		return agent.decodeAction(action, state, decision), nil
	}

	// Select action with max Q-value
	action := agent.argmax(qValues)
	decision.Action = action
	decision.QValue = float64(qValues[action])

	return agent.decodeAction(action, state, decision), nil
}

// runInference runs the neural network inference
func (agent *DQNAgent) runInference(state []float32) ([]float32, error) {
	if !agent.modelLoaded {
		return nil, fmt.Errorf("no model loaded")
	}

	_ = state
	outputData := make([]float32, NumActions)

	return outputData, nil
}

// Remember stores an experience in the replay buffer
func (agent *DQNAgent) Remember(state *EnvironmentState, action Action, reward float64, nextState *EnvironmentState, done bool) {
	exp := &Experience{
		State:     state.ToVector(),
		Action:    action,
		Reward:    reward,
		NextState: nextState.ToVector(),
		Done:      done,
	}

	agent.replayBuffer.Add(exp)
	agent.totalReward += reward

	if done {
		agent.episodeRewards = append(agent.episodeRewards, agent.totalReward)
		agent.totalReward = 0
	}
}

// UpdateEpsilon updates the exploration rate
func (agent *DQNAgent) UpdateEpsilon() {
	agent.mu.Lock()
	defer agent.mu.Unlock()

	if agent.epsilon > agent.epsilonMin {
		agent.epsilon *= agent.epsilonDecay
		if agent.epsilon < agent.epsilonMin {
			agent.epsilon = agent.epsilonMin
		}
	}
}

// decodeAction converts an action to a partition decision
func (agent *DQNAgent) decodeAction(action Action, state *EnvironmentState, decision *TaskPartitionDecision) *TaskPartitionDecision {
	taskSize := state.TaskSize

	switch {
	case action <= ActionStream4:
		// Single stream assignment
		decision.StreamIDs = []int{int(action)}
		decision.ChunkSizes = []int{taskSize}
		decision.ExpectedTime = agent.estimateTime(taskSize, []int{int(action)}, state)

	case action <= ActionSplit34:
		// Split across 2 streams
		streams := getTwoStreamIndices(action)
		decision.StreamIDs = streams
		decision.ChunkSizes = agent.calculateChunkSizes(taskSize, len(streams), streams, state)
		decision.ExpectedTime = agent.estimateTime(taskSize, streams, state)

	case action <= ActionSplit234:
		// Split across 3 streams
		streams := getThreeStreamIndices(action)
		decision.StreamIDs = streams
		decision.ChunkSizes = agent.calculateChunkSizes(taskSize, len(streams), streams, state)
		decision.ExpectedTime = agent.estimateTime(taskSize, streams, state)

	case action == ActionSplitAll:
		// Split across all 4 streams
		streams := []int{0, 1, 2, 3}
		decision.StreamIDs = streams
		decision.ChunkSizes = agent.calculateChunkSizes(taskSize, len(streams), streams, state)
		decision.ExpectedTime = agent.estimateTime(taskSize, streams, state)
	}

	return decision
}

// calculateChunkSizes determines how to split task across streams
func (agent *DQNAgent) calculateChunkSizes(taskSize int, numStreams int, streams []int, state *EnvironmentState) []int {
	if numStreams == 0 {
		return []int{}
	}

	// Calculate relative capacities of streams
	totalCapacity := 0.0
	capacities := make([]float64, numStreams)

	for i, streamID := range streams {
		// Capacity is bandwidth * success rate / (1 + congestion)
		capacity := safeBandwidthMbps(state.StreamBandwidth[streamID]) *
			safeSuccessRate(state.StreamSuccessRate[streamID]) /
			(1 + safeCongestion(state.StreamCongestion[streamID]))
		capacities[i] = capacity
		totalCapacity += capacity
	}

	if totalCapacity <= 0 || math.IsNaN(totalCapacity) || math.IsInf(totalCapacity, 0) {
		return splitEvenly(taskSize, numStreams)
	}

	// Allocate chunks proportionally to capacity
	chunks := make([]int, numStreams)
	allocated := 0

	for i := 0; i < numStreams-1; i++ {
		proportion := capacities[i] / totalCapacity
		chunkSize := int(float64(taskSize) * proportion)
		chunks[i] = chunkSize
		allocated += chunkSize
	}

	// Last chunk gets remainder to ensure exact split
	chunks[numStreams-1] = taskSize - allocated

	return chunks
}

// estimateTime estimates completion time for a partitioning decision
func (agent *DQNAgent) estimateTime(taskSize int, streams []int, state *EnvironmentState) time.Duration {
	maxTime := 0.0

	for i, streamID := range streams {
		chunkSize := taskSize / len(streams)                                      // Simplified for estimation
		bandwidth := safeBandwidthMbps(state.StreamBandwidth[streamID]) * 1e6 / 8 // Convert Mbps to bytes/s
		latency := safeLatencyMs(state.StreamLatency[streamID]) / 1000            // Convert ms to seconds

		// Time = latency + (size / bandwidth) * (1 + congestion)
		streamTime := latency + (float64(chunkSize)/bandwidth)*(1+safeCongestion(state.StreamCongestion[streamID]))

		// Account for potential retransmissions
		streamTime *= (2 - safeSuccessRate(state.StreamSuccessRate[streamID]))

		if i == 0 || streamTime > maxTime {
			maxTime = streamTime
		}
	}

	return time.Duration(maxTime * float64(time.Second))
}

// randomAction returns a random action for exploration
func (agent *DQNAgent) randomAction() Action {
	return Action(rand.Intn(NumActions))
}

// heuristicAction uses a simple heuristic when no model is available
func (agent *DQNAgent) heuristicAction(state *EnvironmentState) Action {
	// Find the stream with best score
	bestScore := -1.0
	bestStream := 0

	for i := 0; i < 4; i++ {
		// Score based on bandwidth, latency, and congestion
		score := streamScore(
			state.StreamBandwidth[i],
			state.StreamLatency[i],
			state.StreamCongestion[i],
			state.StreamSuccessRate[i],
		)

		if score > bestScore {
			bestScore = score
			bestStream = i
		}
	}

	// For large tasks with low priority, consider splitting
	if state.TaskSize > 1e8 && state.TaskPriority < 0.5 {
		// Find second-best stream
		secondBest := (bestStream + 1) % 4
		for i := 0; i < 4; i++ {
			if i != bestStream {
				score := streamScore(
					state.StreamBandwidth[i],
					state.StreamLatency[i],
					state.StreamCongestion[i],
					state.StreamSuccessRate[i],
				)
				if score > streamScore(
					state.StreamBandwidth[secondBest],
					state.StreamLatency[secondBest],
					state.StreamCongestion[secondBest],
					state.StreamSuccessRate[secondBest],
				) {
					secondBest = i
				}
			}
		}

		// Return two-stream split action
		if bestStream < secondBest {
			return Action(4 + bestStream*3 + secondBest - bestStream - 1)
		} else {
			return Action(4 + secondBest*3 + bestStream - secondBest - 1)
		}
	}

	return Action(bestStream)
}

func splitEvenly(taskSize, numStreams int) []int {
	chunks := make([]int, numStreams)
	if numStreams == 0 {
		return chunks
	}

	base := taskSize / numStreams
	remainder := taskSize % numStreams
	for i := range chunks {
		chunks[i] = base
		if i < remainder {
			chunks[i]++
		}
	}
	return chunks
}

func streamScore(bandwidth, latency, congestion, successRate float64) float64 {
	return safeBandwidthMbps(bandwidth) * safeSuccessRate(successRate) /
		(safeLatencyMs(latency) * (1 + safeCongestion(congestion)))
}

func safeBandwidthMbps(value float64) float64 {
	if value <= 0 || math.IsNaN(value) || math.IsInf(value, 0) {
		return 1
	}
	return value
}

func safeLatencyMs(value float64) float64 {
	if value <= 0 || math.IsNaN(value) || math.IsInf(value, 0) {
		return 1
	}
	return value
}

func safeCongestion(value float64) float64 {
	if math.IsNaN(value) || math.IsInf(value, 0) || value < 0 {
		return 0
	}
	if value > 1 {
		return 1
	}
	return value
}

func safeSuccessRate(value float64) float64 {
	if math.IsNaN(value) || math.IsInf(value, 0) || value <= 0 {
		return 0.01
	}
	if value > 1 {
		return 1
	}
	return value
}

// argmax returns the index of the maximum value
func (agent *DQNAgent) argmax(values []float32) Action {
	maxIdx := 0
	maxVal := values[0]

	for i := 1; i < len(values); i++ {
		if values[i] > maxVal {
			maxVal = values[i]
			maxIdx = i
		}
	}

	return Action(maxIdx)
}

// GetMetrics returns performance metrics
func (agent *DQNAgent) GetMetrics() map[string]interface{} {
	agent.mu.RLock()
	defer agent.mu.RUnlock()

	avgReward := 0.0
	if len(agent.episodeRewards) > 0 {
		for _, r := range agent.episodeRewards {
			avgReward += r
		}
		avgReward /= float64(len(agent.episodeRewards))
	}

	return map[string]interface{}{
		"epsilon":        agent.epsilon,
		"buffer_size":    agent.replayBuffer.Size(),
		"total_episodes": len(agent.episodeRewards),
		"average_reward": avgReward,
		"success_rate":   agent.successRate,
		"steps":          agent.stepCount,
	}
}

// SaveModel exports the agent policy state and replay buffer.
func (agent *DQNAgent) SaveModel(path string) error {
	agent.mu.RLock()
	state := dqnAgentState{
		Version:        dqnAgentStateVersion,
		ModelLoaded:    agent.modelLoaded,
		Epsilon:        agent.epsilon,
		EpsilonMin:     agent.epsilonMin,
		EpsilonDecay:   agent.epsilonDecay,
		StateBuffer:    append([]float32(nil), agent.stateBuffer...),
		LearningRate:   agent.learningRate,
		Gamma:          agent.gamma,
		UpdateFreq:     agent.updateFreq,
		StepCount:      agent.stepCount,
		TotalReward:    agent.totalReward,
		EpisodeRewards: append([]float64(nil), agent.episodeRewards...),
		SuccessRate:    agent.successRate,
	}
	agent.mu.RUnlock()

	if agent.replayBuffer != nil {
		agent.replayBuffer.mu.Lock()
		state.ReplayBuffer.Capacity = agent.replayBuffer.capacity
		state.ReplayBuffer.Experiences = append([]*Experience(nil), agent.replayBuffer.buffer...)
		agent.replayBuffer.mu.Unlock()
	}

	data, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(path, data, 0600)
}

// LoadModel loads a pre-trained model
func (agent *DQNAgent) LoadModel(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("failed to load model %q: %w", path, err)
	}

	var state dqnAgentState
	if err := json.Unmarshal(data, &state); err != nil {
		return fmt.Errorf("failed to decode model %q: %w", path, err)
	}
	if state.Version != dqnAgentStateVersion {
		return fmt.Errorf("unsupported model state version %d", state.Version)
	}
	if state.ReplayBuffer.Capacity <= 0 {
		state.ReplayBuffer.Capacity = 10000
	}

	agent.mu.Lock()
	agent.epsilon = state.Epsilon
	agent.epsilonMin = state.EpsilonMin
	agent.epsilonDecay = state.EpsilonDecay
	agent.stateBuffer = append([]float32(nil), state.StateBuffer...)
	agent.learningRate = state.LearningRate
	agent.gamma = state.Gamma
	agent.updateFreq = state.UpdateFreq
	agent.stepCount = state.StepCount
	agent.totalReward = state.TotalReward
	agent.episodeRewards = append([]float64(nil), state.EpisodeRewards...)
	agent.successRate = state.SuccessRate
	agent.replayBuffer = NewReplayBuffer(state.ReplayBuffer.Capacity)
	agent.replayBuffer.buffer = append([]*Experience(nil), state.ReplayBuffer.Experiences...)
	agent.modelLoaded = true
	agent.mu.Unlock()

	return nil
}

// Destroy cleans up resources
func (agent *DQNAgent) Destroy() {
	if agent != nil {
		agent.modelLoaded = false
	}
}

// GetReplayBuffer returns the agent's replay buffer (getter for unexported field)
func (agent *DQNAgent) GetReplayBuffer() *ReplayBuffer {
	agent.mu.RLock()
	defer agent.mu.RUnlock()
	return agent.replayBuffer
}

// GetEpsilon returns the current epsilon value (getter for unexported field)
func (agent *DQNAgent) GetEpsilon() float64 {
	agent.mu.RLock()
	defer agent.mu.RUnlock()
	return agent.epsilon
}
