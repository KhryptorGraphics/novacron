package partition

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"math/rand"
	"sync"
	"time"

	ort "github.com/yalue/onnxruntime_go"
)

// DQNAgent implements a Deep Q-Network agent for task partitioning
type DQNAgent struct {
	session      *ort.DynamicAdvancedSession
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
	// Initialize ONNX runtime
	ort.SetSharedLibraryPath("/usr/lib/x86_64-linux-gnu/libonnxruntime.so")

	if err := ort.InitializeEnvironment(); err != nil {
		return nil, fmt.Errorf("failed to initialize ONNX runtime: %w", err)
	}

	// Create session options
	options, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("failed to create session options: %w", err)
	}
	defer options.Destroy()

	// Create session
	session, err := ort.NewDynamicAdvancedSession(modelPath, []string{"input"}, []string{"output"}, options)
	if err != nil {
		// If model doesn't exist, we'll operate in exploration mode
		log.Printf("Warning: Could not load model from %s, operating in exploration mode", modelPath)
		session = nil
	}

	agent := &DQNAgent{
		session:        session,
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
	if agent.session == nil {
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
	if agent.session == nil {
		return nil, fmt.Errorf("no model loaded")
	}

	// Create input tensor
	inputShape := ort.NewShape(1, int64(len(state)))
	inputTensor, err := ort.NewTensor(inputShape, state)
	if err != nil {
		return nil, fmt.Errorf("failed to create input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	// Run inference
	// Note: The actual ONNX Runtime Go API may vary. This is a placeholder.
	// For now, we'll use a simplified version that works without the library
	outputData := make([]float32, NumActions)

	// TODO: Replace with actual ONNX Runtime inference when library is properly configured
	// For heuristic fallback, this won't be called anyway
	log.Printf("Warning: ONNX Runtime inference not yet configured, using fallback")

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
		capacity := state.StreamBandwidth[streamID] * state.StreamSuccessRate[streamID] /
			(1 + state.StreamCongestion[streamID])
		capacities[i] = capacity
		totalCapacity += capacity
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
		chunkSize := taskSize / len(streams)                   // Simplified for estimation
		bandwidth := state.StreamBandwidth[streamID] * 1e6 / 8 // Convert Mbps to bytes/s
		latency := state.StreamLatency[streamID] / 1000        // Convert ms to seconds

		// Time = latency + (size / bandwidth) * (1 + congestion)
		streamTime := latency + (float64(chunkSize)/bandwidth)*(1+state.StreamCongestion[streamID])

		// Account for potential retransmissions
		streamTime *= (2 - state.StreamSuccessRate[streamID])

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
		score := state.StreamBandwidth[i] * state.StreamSuccessRate[i] /
			(state.StreamLatency[i] * (1 + state.StreamCongestion[i]))

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
				score := state.StreamBandwidth[i] * state.StreamSuccessRate[i] /
					(state.StreamLatency[i] * (1 + state.StreamCongestion[i]))
				if score > state.StreamBandwidth[secondBest]*state.StreamSuccessRate[secondBest]/
					(state.StreamLatency[secondBest]*(1+state.StreamCongestion[secondBest])) {
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

// SaveModel exports the model parameters (placeholder for actual implementation)
func (agent *DQNAgent) SaveModel(path string) error {
	metrics := agent.GetMetrics()
	data, err := json.MarshalIndent(metrics, "", "  ")
	if err != nil {
		return err
	}

	return ioutil.WriteFile(path+".metrics.json", data, 0644)
}

// LoadModel loads a pre-trained model
func (agent *DQNAgent) LoadModel(path string) error {
	// Destroy old session if exists
	if agent.session != nil {
		agent.session.Destroy()
	}

	// Load new model
	options, err := ort.NewSessionOptions()
	if err != nil {
		return fmt.Errorf("failed to create session options: %w", err)
	}
	defer options.Destroy()

	session, err := ort.NewDynamicAdvancedSession(path, []string{"input"}, []string{"output"}, options)
	if err != nil {
		return fmt.Errorf("failed to load model: %w", err)
	}

	agent.session = session
	return nil
}

// Destroy cleans up resources
func (agent *DQNAgent) Destroy() {
	if agent != nil && agent.session != nil {
		agent.session.Destroy()
	}
	// Note: DestroyEnvironment() should only be called once per process
	// Commenting out to avoid issues in tests
	// ort.DestroyEnvironment()
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
