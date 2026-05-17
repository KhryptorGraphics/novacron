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

	ort "github.com/yalue/onnxruntime_go"
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

	inference qValueSession
	modelPath string
}

type qValueSession interface {
	Run(input []float32) ([]float32, error)
	Destroy() error
}

type onnxDQNSession struct {
	session      *ort.AdvancedSession
	inputTensor  *ort.Tensor[float32]
	outputTensor *ort.Tensor[float32]
}

var onnxEnvMu sync.Mutex

const dqnAgentStateVersion = 1

type dqnAgentState struct {
	Version        int         `json:"version"`
	ModelLoaded    bool        `json:"model_loaded"`
	ModelPath      string      `json:"model_path,omitempty"`
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
			session, err := newONNXDQNSession(modelPath)
			if err != nil {
				log.Printf("Warning: Could not load DQN model from %s, operating in heuristic-only mode: %v", modelPath, err)
			} else {
				agent.inference = session
				agent.modelLoaded = true
				agent.modelPath = modelPath
			}
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
	if state == nil {
		return nil, fmt.Errorf("state is nil")
	}

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
		log.Printf("Inference error, falling back to heuristic action: %v", err)
		action := agent.heuristicAction(state)
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
	if !agent.modelLoaded || agent.inference == nil {
		return nil, fmt.Errorf("no model loaded")
	}
	if len(state) != 20 {
		return nil, fmt.Errorf("invalid DQN state vector length %d", len(state))
	}

	qValues, err := agent.inference.Run(state)
	if err != nil {
		return nil, err
	}
	if len(qValues) != NumActions {
		return nil, fmt.Errorf("invalid DQN output length %d", len(qValues))
	}
	for i, value := range qValues {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return nil, fmt.Errorf("invalid DQN output at %d: %f", i, value)
		}
	}

	return qValues, nil
}

func newONNXDQNSession(modelPath string) (*onnxDQNSession, error) {
	if err := ensureONNXEnvironment(); err != nil {
		return nil, err
	}

	inputs, outputs, err := ort.GetInputOutputInfo(modelPath)
	if err != nil {
		return nil, fmt.Errorf("inspect ONNX model: %w", err)
	}
	if len(inputs) != 1 {
		return nil, fmt.Errorf("expected one ONNX input, got %d", len(inputs))
	}
	if len(outputs) != 1 {
		return nil, fmt.Errorf("expected one ONNX output, got %d", len(outputs))
	}
	if inputs[0].OrtValueType != ort.ONNXTypeTensor || inputs[0].DataType != ort.TensorElementDataTypeFloat {
		return nil, fmt.Errorf("expected float tensor input, got %s", inputs[0].String())
	}
	if outputs[0].OrtValueType != ort.ONNXTypeTensor || outputs[0].DataType != ort.TensorElementDataTypeFloat {
		return nil, fmt.Errorf("expected float tensor output, got %s", outputs[0].String())
	}
	if !dqnShapeAllows(inputs[0].Dimensions, 20) {
		return nil, fmt.Errorf("expected DQN input shape compatible with 20 features, got %s", inputs[0].Dimensions.String())
	}
	if !dqnShapeAllows(outputs[0].Dimensions, NumActions) {
		return nil, fmt.Errorf("expected DQN output shape compatible with %d actions, got %s", NumActions, outputs[0].Dimensions.String())
	}

	inputTensor, err := ort.NewTensor[float32](ort.Shape{1, 20}, make([]float32, 20))
	if err != nil {
		return nil, fmt.Errorf("create DQN input tensor: %w", err)
	}
	outputTensor, err := ort.NewTensor[float32](ort.Shape{1, int64(NumActions)}, make([]float32, NumActions))
	if err != nil {
		_ = inputTensor.Destroy()
		return nil, fmt.Errorf("create DQN output tensor: %w", err)
	}

	session, err := ort.NewAdvancedSession(
		modelPath,
		[]string{inputs[0].Name},
		[]string{outputs[0].Name},
		[]ort.Value{inputTensor},
		[]ort.Value{outputTensor},
		nil,
	)
	if err != nil {
		_ = inputTensor.Destroy()
		_ = outputTensor.Destroy()
		return nil, fmt.Errorf("create DQN ONNX session: %w", err)
	}

	return &onnxDQNSession{
		session:      session,
		inputTensor:  inputTensor,
		outputTensor: outputTensor,
	}, nil
}

func ensureONNXEnvironment() error {
	onnxEnvMu.Lock()
	defer onnxEnvMu.Unlock()

	if ort.IsInitialized() {
		return nil
	}
	if libraryPath := os.Getenv("ONNXRUNTIME_SHARED_LIBRARY_PATH"); libraryPath != "" {
		ort.SetSharedLibraryPath(libraryPath)
	}
	return ort.InitializeEnvironment()
}

func dqnShapeAllows(shape ort.Shape, width int) bool {
	if len(shape) == 0 {
		return false
	}
	last := shape[len(shape)-1]
	return last == int64(width)
}

func (session *onnxDQNSession) Run(input []float32) ([]float32, error) {
	copy(session.inputTensor.GetData(), input)
	session.outputTensor.ZeroContents()

	if err := session.session.Run(); err != nil {
		return nil, fmt.Errorf("run DQN ONNX session: %w", err)
	}

	output := make([]float32, NumActions)
	copy(output, session.outputTensor.GetData())
	return output, nil
}

func (session *onnxDQNSession) Destroy() error {
	var err error
	if session.session != nil {
		if e := session.session.Destroy(); e != nil {
			err = e
		}
		session.session = nil
	}
	if session.inputTensor != nil {
		if e := session.inputTensor.Destroy(); err == nil && e != nil {
			err = e
		}
		session.inputTensor = nil
	}
	if session.outputTensor != nil {
		if e := session.outputTensor.Destroy(); err == nil && e != nil {
			err = e
		}
		session.outputTensor = nil
	}
	return err
}

// Remember stores an experience in the replay buffer
func (agent *DQNAgent) Remember(state *EnvironmentState, action Action, reward float64, nextState *EnvironmentState, done bool) {
	if state == nil || nextState == nil {
		return
	}
	if action < 0 || action >= NumActions {
		return
	}
	if math.IsNaN(reward) || math.IsInf(reward, 0) {
		return
	}

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
	if decision == nil {
		decision = &TaskPartitionDecision{}
	}
	if state == nil || action < 0 || action >= NumActions {
		return decision
	}

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
	if taskSize <= 0 || !validStreamSelection(state, streams, numStreams) {
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

func validStreamSelection(state *EnvironmentState, streams []int, numStreams int) bool {
	if state == nil || numStreams <= 0 || len(streams) != numStreams {
		return false
	}
	for _, streamID := range streams {
		if streamID < 0 || streamID >= 4 {
			return false
		}
	}
	return true
}

// estimateTime estimates completion time for a partitioning decision
func (agent *DQNAgent) estimateTime(taskSize int, streams []int, state *EnvironmentState) time.Duration {
	if taskSize <= 0 || !validStreamSelection(state, streams, len(streams)) {
		return 0
	}

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
		ModelPath:      agent.modelPath,
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

	var session qValueSession
	modelLoaded := false
	if state.ModelLoaded && state.ModelPath != "" {
		loaded, err := newONNXDQNSession(state.ModelPath)
		if err != nil {
			log.Printf("Warning: Could not restore DQN model from %s, operating in heuristic-only mode: %v", state.ModelPath, err)
		} else {
			session = loaded
			modelLoaded = true
		}
	}

	agent.mu.Lock()
	if agent.inference != nil {
		_ = agent.inference.Destroy()
	}
	agent.epsilon = boundedFloat(state.Epsilon, 0, 1, agent.epsilon)
	agent.epsilonMin = boundedFloat(state.EpsilonMin, 0, 1, agent.epsilonMin)
	agent.epsilonDecay = boundedFloat(state.EpsilonDecay, 0, 1, agent.epsilonDecay)
	agent.stateBuffer = append([]float32(nil), state.StateBuffer...)
	agent.learningRate = positiveFloat(state.LearningRate, agent.learningRate)
	agent.gamma = boundedFloat(state.Gamma, 0, 1, agent.gamma)
	agent.updateFreq = positiveInt(state.UpdateFreq, agent.updateFreq)
	agent.stepCount = nonNegativeInt(state.StepCount, agent.stepCount)
	agent.totalReward = state.TotalReward
	agent.episodeRewards = append([]float64(nil), state.EpisodeRewards...)
	agent.successRate = boundedFloat(state.SuccessRate, 0, 1, agent.successRate)
	agent.replayBuffer = NewReplayBuffer(state.ReplayBuffer.Capacity)
	for _, exp := range state.ReplayBuffer.Experiences {
		if isValidReplayExperience(exp) {
			agent.replayBuffer.Add(cloneExperience(exp))
		}
	}
	agent.inference = session
	agent.modelLoaded = modelLoaded
	agent.modelPath = state.ModelPath
	agent.mu.Unlock()

	return nil
}

func isValidReplayExperience(exp *Experience) bool {
	if exp == nil || exp.State == nil || exp.NextState == nil {
		return false
	}
	if exp.Action < 0 || exp.Action >= NumActions {
		return false
	}
	return !math.IsNaN(exp.Reward) && !math.IsInf(exp.Reward, 0)
}

func boundedFloat(value, minValue, maxValue, fallback float64) float64 {
	if math.IsNaN(value) || math.IsInf(value, 0) || value < minValue || value > maxValue {
		return fallback
	}
	return value
}

func positiveFloat(value, fallback float64) float64 {
	if math.IsNaN(value) || math.IsInf(value, 0) || value <= 0 {
		return fallback
	}
	return value
}

func positiveInt(value, fallback int) int {
	if value <= 0 {
		return fallback
	}
	return value
}

func nonNegativeInt(value, fallback int) int {
	if value < 0 {
		return fallback
	}
	return value
}

func cloneExperience(exp *Experience) *Experience {
	return &Experience{
		State:     append([]float32(nil), exp.State...),
		Action:    exp.Action,
		Reward:    exp.Reward,
		NextState: append([]float32(nil), exp.NextState...),
		Done:      exp.Done,
		TDError:   exp.TDError,
	}
}

// Destroy cleans up resources
func (agent *DQNAgent) Destroy() {
	if agent != nil {
		agent.mu.Lock()
		defer agent.mu.Unlock()
		if agent.inference != nil {
			_ = agent.inference.Destroy()
			agent.inference = nil
		}
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
