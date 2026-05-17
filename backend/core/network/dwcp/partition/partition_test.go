package partition

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"
)

type fakeQValueSession struct {
	qValues   []float32
	err       error
	destroyed bool
}

func (session *fakeQValueSession) Run(_ []float32) ([]float32, error) {
	return append([]float32(nil), session.qValues...), session.err
}

func (session *fakeQValueSession) Destroy() error {
	session.destroyed = true
	return nil
}

func TestEnvironmentState(t *testing.T) {
	state := NewEnvironmentState()

	// Test state initialization
	if state.TaskQueueDepth != 0 {
		t.Errorf("Expected TaskQueueDepth 0, got %d", state.TaskQueueDepth)
	}

	// Test state vectorization
	vector := state.ToVector()
	if len(vector) != 20 {
		t.Errorf("Expected vector length 20, got %d", len(vector))
	}

	// Check normalization
	for i, v := range vector {
		if v < 0 || v > 1.5 {
			t.Errorf("Vector element %d out of expected range: %f", i, v)
		}
	}
}

func TestRewardCalculator(t *testing.T) {
	calc := NewRewardCalculator()

	tests := []struct {
		name     string
		outcome  *ActionOutcome
		expected float64 // Rough expected reward
	}{
		{
			name: "high_throughput_low_latency",
			outcome: &ActionOutcome{
				ActualThroughput:   150.0,
				BaselineThroughput: 100.0,
				ActualLatency:      8.0,
				TargetLatency:      10.0,
				StreamImbalance:    0.1,
				Completed:          true,
				Retransmissions:    0,
			},
			expected: 2.5, // Positive reward
		},
		{
			name: "low_throughput_high_latency",
			outcome: &ActionOutcome{
				ActualThroughput:   50.0,
				BaselineThroughput: 100.0,
				ActualLatency:      20.0,
				TargetLatency:      10.0,
				StreamImbalance:    0.3,
				Completed:          false,
				Retransmissions:    2,
			},
			expected: -2.0, // Negative reward
		},
		{
			name: "balanced_performance",
			outcome: &ActionOutcome{
				ActualThroughput:   100.0,
				BaselineThroughput: 100.0,
				ActualLatency:      10.0,
				TargetLatency:      10.0,
				StreamImbalance:    0.0,
				Completed:          true,
				Retransmissions:    0,
			},
			expected: 2.0, // Completion bonus
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reward := calc.Calculate(tt.outcome)

			if (tt.expected > 0 && reward < 0) || (tt.expected < 0 && reward > 0) {
				t.Errorf("Expected reward sign %v, got %v (reward: %f)", tt.expected > 0, reward > 0, reward)
			}
		})
	}
}

func TestReplayBuffer(t *testing.T) {
	buffer := NewReplayBuffer(100)

	// Test adding experiences
	for i := 0; i < 150; i++ {
		exp := &Experience{
			State:     make([]float32, 20),
			Action:    Action(i % 15),
			Reward:    float64(i),
			NextState: make([]float32, 20),
			Done:      i%10 == 0,
		}
		buffer.Add(exp)
	}

	// Check buffer size (should cap at 100)
	if buffer.Size() != 100 {
		t.Errorf("Expected buffer size 100, got %d", buffer.Size())
	}

	// Test sampling
	sample := buffer.Sample(32)
	if len(sample) != 32 {
		t.Errorf("Expected sample size 32, got %d", len(sample))
	}
}

func TestReplayBufferAddIgnoresInvalidInput(t *testing.T) {
	tests := []struct {
		name     string
		capacity int
		exp      *Experience
	}{
		{
			name:     "nil_experience",
			capacity: 1,
			exp:      nil,
		},
		{
			name:     "zero_capacity",
			capacity: 0,
			exp: &Experience{
				State:     make([]float32, 20),
				Action:    ActionStream1,
				Reward:    1,
				NextState: make([]float32, 20),
			},
		},
		{
			name:     "negative_capacity",
			capacity: -1,
			exp: &Experience{
				State:     make([]float32, 20),
				Action:    ActionStream1,
				Reward:    1,
				NextState: make([]float32, 20),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			buffer := NewReplayBuffer(tt.capacity)

			defer func() {
				if recovered := recover(); recovered != nil {
					t.Fatalf("invalid replay input should be ignored without panic: %v", recovered)
				}
			}()

			buffer.Add(tt.exp)

			if buffer.Size() != 0 {
				t.Fatalf("invalid replay input should not grow buffer: got %d", buffer.Size())
			}
		})
	}
}

func TestReplayBufferSampleHandlesInvalidBatchAndReturnsCopy(t *testing.T) {
	buffer := NewReplayBuffer(2)
	first := &Experience{
		State:     make([]float32, 20),
		Action:    ActionStream1,
		Reward:    1,
		NextState: make([]float32, 20),
	}
	buffer.Add(first)

	defer func() {
		if recovered := recover(); recovered != nil {
			t.Fatalf("invalid sample batch should not panic: %v", recovered)
		}
	}()

	if sample := buffer.Sample(-1); len(sample) != 0 {
		t.Fatalf("negative sample batch should return empty sample, got %d", len(sample))
	}

	sample := buffer.Sample(2)
	if len(sample) != 1 {
		t.Fatalf("oversized sample should return existing entries, got %d", len(sample))
	}
	sample[0] = &Experience{
		State:     make([]float32, 20),
		Action:    ActionStream2,
		Reward:    2,
		NextState: make([]float32, 20),
	}

	resampled := buffer.Sample(1)
	if len(resampled) != 1 {
		t.Fatalf("expected one resampled experience, got %d", len(resampled))
	}
	if resampled[0] != first {
		t.Fatal("sample should not expose internal replay buffer storage")
	}
}

func TestOnlineLearnerStopCancelsAutoUpdateLoop(t *testing.T) {
	agent, err := NewDQNAgent("")
	if err != nil {
		t.Fatalf("failed to create DQN agent: %v", err)
	}

	learner := NewOnlineLearner(agent, &OnlineLearnerConfig{
		UpdateFrequency:  time.Hour,
		MinExperiences:   1,
		TrainingScript:   "unused",
		ModelPath:        filepath.Join(t.TempDir(), "model"),
		EnableAutoUpdate: true,
	})

	learner.Stop()

	select {
	case <-learner.ctx.Done():
	case <-time.After(time.Second):
		t.Fatal("Stop did not cancel the online learner context")
	}
}

func TestOnlineLearnerStoppedRejectsModelWork(t *testing.T) {
	agent, err := NewDQNAgent("")
	if err != nil {
		t.Fatalf("failed to create DQN agent: %v", err)
	}

	learner := NewOnlineLearner(agent, &OnlineLearnerConfig{
		UpdateFrequency:  time.Hour,
		MinExperiences:   2,
		TrainingScript:   "unused",
		ModelPath:        filepath.Join(t.TempDir(), "model"),
		EnableAutoUpdate: false,
	})

	learner.Stop()

	if err := learner.ForceUpdate(); err == nil || err.Error() != "online learner stopped" {
		t.Fatalf("expected stopped ForceUpdate error, got %v", err)
	}

	results, err := learner.EvaluateModel(1)
	if err == nil || err.Error() != "online learner stopped" {
		t.Fatalf("expected stopped EvaluateModel error, got results=%v err=%v", results, err)
	}
}

func TestOnlineLearnerStoppedIgnoresCollectedExperience(t *testing.T) {
	agent, err := NewDQNAgent("")
	if err != nil {
		t.Fatalf("failed to create DQN agent: %v", err)
	}

	learner := NewOnlineLearner(agent, &OnlineLearnerConfig{
		UpdateFrequency:  time.Hour,
		MinExperiences:   2,
		TrainingScript:   "unused",
		ModelPath:        filepath.Join(t.TempDir(), "model"),
		EnableAutoUpdate: false,
	})

	learner.Stop()
	before := learner.GetStatus()

	learner.CollectExperience(
		NewEnvironmentState(),
		ActionStream1,
		1,
		NewEnvironmentState(),
		false,
	)

	after := learner.GetStatus()
	if after["experience_count"] != before["experience_count"] {
		t.Fatalf("stopped learner should not collect experience: before=%v after=%v", before["experience_count"], after["experience_count"])
	}
	if after["buffer_size"] != before["buffer_size"] {
		t.Fatalf("stopped learner should not grow replay buffer: before=%v after=%v", before["buffer_size"], after["buffer_size"])
	}
}

func TestOnlineLearnerCollectExperienceIgnoresInvalidAction(t *testing.T) {
	agent, err := NewDQNAgent("")
	if err != nil {
		t.Fatalf("failed to create DQN agent: %v", err)
	}

	learner := NewOnlineLearner(agent, &OnlineLearnerConfig{
		UpdateFrequency:  time.Hour,
		MinExperiences:   2,
		TrainingScript:   "unused",
		ModelPath:        filepath.Join(t.TempDir(), "model"),
		EnableAutoUpdate: false,
	})
	before := learner.GetStatus()

	learner.CollectExperience(
		NewEnvironmentState(),
		Action(NumActions),
		1,
		NewEnvironmentState(),
		false,
	)

	after := learner.GetStatus()
	if after["experience_count"] != before["experience_count"] {
		t.Fatalf("invalid action should not collect experience: before=%v after=%v", before["experience_count"], after["experience_count"])
	}
	if after["buffer_size"] != before["buffer_size"] {
		t.Fatalf("invalid action should not grow replay buffer: before=%v after=%v", before["buffer_size"], after["buffer_size"])
	}
}

func TestOnlineLearnerCollectExperienceIgnoresNilStates(t *testing.T) {
	tests := []struct {
		name      string
		state     *EnvironmentState
		nextState *EnvironmentState
	}{
		{
			name:      "nil_state",
			state:     nil,
			nextState: NewEnvironmentState(),
		},
		{
			name:      "nil_next_state",
			state:     NewEnvironmentState(),
			nextState: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			agent, err := NewDQNAgent("")
			if err != nil {
				t.Fatalf("failed to create DQN agent: %v", err)
			}

			learner := NewOnlineLearner(agent, &OnlineLearnerConfig{
				UpdateFrequency:  time.Hour,
				MinExperiences:   2,
				TrainingScript:   "unused",
				ModelPath:        filepath.Join(t.TempDir(), "model"),
				EnableAutoUpdate: false,
			})
			before := learner.GetStatus()

			defer func() {
				if recovered := recover(); recovered != nil {
					t.Fatalf("nil states should be ignored without panic: %v", recovered)
				}
			}()

			learner.CollectExperience(
				tt.state,
				ActionStream1,
				1,
				tt.nextState,
				false,
			)

			after := learner.GetStatus()
			if after["experience_count"] != before["experience_count"] {
				t.Fatalf("nil state should not collect experience: before=%v after=%v", before["experience_count"], after["experience_count"])
			}
			if after["buffer_size"] != before["buffer_size"] {
				t.Fatalf("nil state should not grow replay buffer: before=%v after=%v", before["buffer_size"], after["buffer_size"])
			}
		})
	}
}

func TestOnlineLearnerCollectExperienceIgnoresInvalidRewards(t *testing.T) {
	tests := []struct {
		name   string
		reward float64
	}{
		{name: "nan", reward: math.NaN()},
		{name: "positive_inf", reward: math.Inf(1)},
		{name: "negative_inf", reward: math.Inf(-1)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			agent, err := NewDQNAgent("")
			if err != nil {
				t.Fatalf("failed to create DQN agent: %v", err)
			}

			learner := NewOnlineLearner(agent, &OnlineLearnerConfig{
				UpdateFrequency:  time.Hour,
				MinExperiences:   2,
				TrainingScript:   "unused",
				ModelPath:        filepath.Join(t.TempDir(), "model"),
				EnableAutoUpdate: false,
			})
			before := learner.GetStatus()

			learner.CollectExperience(
				NewEnvironmentState(),
				ActionStream1,
				tt.reward,
				NewEnvironmentState(),
				false,
			)

			after := learner.GetStatus()
			if after["experience_count"] != before["experience_count"] {
				t.Fatalf("invalid reward should not collect experience: before=%v after=%v", before["experience_count"], after["experience_count"])
			}
			if after["buffer_size"] != before["buffer_size"] {
				t.Fatalf("invalid reward should not grow replay buffer: before=%v after=%v", before["buffer_size"], after["buffer_size"])
			}
			avgReward, ok := after["avg_reward"].(float64)
			if !ok {
				t.Fatalf("avg_reward has unexpected type %T", after["avg_reward"])
			}
			if math.IsNaN(avgReward) || math.IsInf(avgReward, 0) {
				t.Fatalf("invalid reward should not poison avg_reward: %v", avgReward)
			}
		})
	}
}

func TestOnlineLearnerEvaluateModelRejectsNonPositiveEpisodes(t *testing.T) {
	agent, err := NewDQNAgent("")
	if err != nil {
		t.Fatalf("failed to create DQN agent: %v", err)
	}

	learner := NewOnlineLearner(agent, &OnlineLearnerConfig{
		UpdateFrequency:  time.Hour,
		MinExperiences:   2,
		TrainingScript:   "unused",
		ModelPath:        filepath.Join(t.TempDir(), "model"),
		EnableAutoUpdate: false,
	})

	for _, episodes := range []int{0, -1} {
		results, err := learner.EvaluateModel(episodes)
		if err == nil || err.Error() != "episodes must be positive" {
			t.Fatalf("expected positive episodes error for %d, got results=%v err=%v", episodes, results, err)
		}
	}
}

func TestEnvironmentSimulator(t *testing.T) {
	sim := NewEnvironmentSimulator()

	// Test reset
	state := sim.Reset()
	if state == nil {
		t.Fatal("Reset returned nil state")
	}

	// Check state validity
	for i := 0; i < 4; i++ {
		if state.StreamBandwidth[i] <= 0 {
			t.Errorf("Invalid bandwidth for stream %d: %f", i, state.StreamBandwidth[i])
		}
		if state.StreamLatency[i] <= 0 {
			t.Errorf("Invalid latency for stream %d: %f", i, state.StreamLatency[i])
		}
	}

	// Test step
	action := ActionStream1
	nextState, reward, done := sim.Step(action)

	if nextState == nil {
		t.Error("Step returned nil next state")
	}

	if math.IsNaN(reward) || math.IsInf(reward, 0) {
		t.Errorf("Invalid reward: %f", reward)
	}

	if done && nextState.TaskQueueDepth != 0 {
		t.Error("Episode marked done but task queue not empty")
	}
}

func TestDQNAgentHeuristic(t *testing.T) {
	// Test without loading a model (uses heuristic)
	agent, err := NewDQNAgent("nonexistent_model.onnx")
	if err != nil {
		t.Skipf("ONNX Runtime not available, skipping: %v", err)
		return
	}
	defer agent.Destroy()

	state := NewEnvironmentState()
	state.StreamBandwidth = [4]float64{100, 80, 120, 90}
	state.StreamLatency = [4]float64{10, 15, 8, 12}
	state.StreamCongestion = [4]float64{0.1, 0.3, 0.05, 0.2}
	state.StreamSuccessRate = [4]float64{0.95, 0.90, 0.98, 0.92}
	state.TaskSize = 1e8

	decision, err := agent.SelectAction(state)
	if err != nil {
		t.Fatalf("SelectAction failed: %v", err)
	}

	if decision == nil {
		t.Fatal("Decision is nil")
	}

	if len(decision.StreamIDs) == 0 {
		t.Error("No streams selected")
	}

	if len(decision.ChunkSizes) != len(decision.StreamIDs) {
		t.Error("Chunk sizes don't match stream count")
	}

	totalSize := 0
	for _, size := range decision.ChunkSizes {
		totalSize += size
	}

	if totalSize != state.TaskSize {
		t.Errorf("Total chunk size %d doesn't match task size %d", totalSize, state.TaskSize)
	}
}

func TestDQNAgentFallsBackToHeuristicWhenInferenceUnavailable(t *testing.T) {
	agent, err := NewDQNAgent("")
	if err != nil {
		t.Fatalf("NewDQNAgent failed: %v", err)
	}
	defer agent.Destroy()

	agent.epsilon = 0
	agent.modelLoaded = true

	state := NewEnvironmentState()
	state.StreamBandwidth = [4]float64{10, 20, 30, 400}
	state.StreamLatency = [4]float64{100, 100, 100, 1}
	state.StreamCongestion = [4]float64{0.9, 0.9, 0.9, 0}
	state.StreamSuccessRate = [4]float64{0.5, 0.5, 0.5, 1}
	state.TaskSize = 1024
	state.TaskPriority = 0.9

	decision, err := agent.SelectAction(state)
	if err != nil {
		t.Fatalf("SelectAction failed: %v", err)
	}
	if decision.Action != ActionStream4 {
		t.Fatalf("Expected heuristic fallback to choose stream 4, got action %d", decision.Action)
	}
	if decision.ExplorationUsed {
		t.Fatal("Expected heuristic fallback without exploration")
	}
}

func TestDQNAgentUsesLoadedInferenceSession(t *testing.T) {
	agent, err := NewDQNAgent("")
	if err != nil {
		t.Fatalf("NewDQNAgent failed: %v", err)
	}
	defer agent.Destroy()

	qValues := make([]float32, NumActions)
	qValues[ActionStream4] = 10
	fakeSession := &fakeQValueSession{qValues: qValues}

	agent.epsilon = 0
	agent.modelLoaded = true
	agent.inference = fakeSession

	state := NewEnvironmentState()
	state.TaskSize = 1024

	decision, err := agent.SelectAction(state)
	if err != nil {
		t.Fatalf("SelectAction failed: %v", err)
	}
	if decision.Action != ActionStream4 {
		t.Fatalf("Expected model action stream 4, got %d", decision.Action)
	}
	if decision.QValue != 10 {
		t.Fatalf("Expected Q-value 10, got %f", decision.QValue)
	}
}

func TestDQNAgentDoesNotMarkInvalidONNXAsLoaded(t *testing.T) {
	modelPath := t.TempDir() + "/invalid.onnx"
	if err := os.WriteFile(modelPath, []byte("not an onnx model"), 0600); err != nil {
		t.Fatalf("WriteFile failed: %v", err)
	}

	agent, err := NewDQNAgent(modelPath)
	if err != nil {
		t.Fatalf("NewDQNAgent failed: %v", err)
	}
	defer agent.Destroy()

	if agent.modelLoaded {
		t.Fatal("Invalid ONNX artifact must not be marked as loaded")
	}
	if agent.inference != nil {
		t.Fatal("Invalid ONNX artifact must not install an inference session")
	}
}

func TestDQNAgentSaveLoadModel(t *testing.T) {
	agent, err := NewDQNAgent("")
	if err != nil {
		t.Fatalf("NewDQNAgent failed: %v", err)
	}
	defer agent.Destroy()

	agent.mu.Lock()
	agent.modelLoaded = true
	agent.epsilon = 0.42
	agent.epsilonMin = 0.02
	agent.epsilonDecay = 0.9
	agent.stateBuffer = []float32{0.1, 0.2, 0.3}
	agent.learningRate = 0.005
	agent.gamma = 0.8
	agent.updateFreq = 64
	agent.stepCount = 17
	agent.totalReward = 12.5
	agent.episodeRewards = []float64{1.5, 2.5}
	agent.successRate = 0.75
	agent.mu.Unlock()

	agent.replayBuffer.Add(&Experience{
		State:     []float32{1, 2},
		Action:    ActionStream2,
		Reward:    3.5,
		NextState: []float32{4, 5},
		Done:      true,
		TDError:   0.25,
	})

	modelPath := t.TempDir() + "/dqn-agent.json"
	if err := agent.SaveModel(modelPath); err != nil {
		t.Fatalf("SaveModel failed: %v", err)
	}

	loaded, err := NewDQNAgent("")
	if err != nil {
		t.Fatalf("NewDQNAgent failed: %v", err)
	}
	defer loaded.Destroy()

	if err := loaded.LoadModel(modelPath); err != nil {
		t.Fatalf("LoadModel failed: %v", err)
	}

	if loaded.modelLoaded {
		t.Fatal("Serialized policy state without an ONNX path must not mark inference as loaded")
	}
	if loaded.epsilon != agent.epsilon || loaded.gamma != agent.gamma || loaded.stepCount != agent.stepCount {
		t.Fatalf("Loaded scalar state mismatch: got epsilon=%v gamma=%v steps=%v", loaded.epsilon, loaded.gamma, loaded.stepCount)
	}
	if len(loaded.episodeRewards) != len(agent.episodeRewards) || loaded.episodeRewards[1] != agent.episodeRewards[1] {
		t.Fatal("Loaded episode rewards mismatch")
	}
	if loaded.replayBuffer.Size() != 1 {
		t.Fatalf("Expected one replay experience, got %d", loaded.replayBuffer.Size())
	}
}

func TestDQNAgentLoadModelFiltersInvalidReplayExperiences(t *testing.T) {
	modelState := dqnAgentState{
		Version: dqnAgentStateVersion,
		ReplayBuffer: replayState{
			Capacity: 5,
			Experiences: []*Experience{
				nil,
				{
					State:     nil,
					Action:    ActionStream1,
					Reward:    1,
					NextState: []float32{1, 2},
				},
				{
					State:     []float32{1, 2},
					Action:    Action(NumActions),
					Reward:    1,
					NextState: []float32{3, 4},
				},
				{
					State:     []float32{1, 2},
					Action:    ActionStream1,
					Reward:    1,
					NextState: nil,
				},
				{
					State:     []float32{1, 2},
					Action:    ActionStream2,
					Reward:    2,
					NextState: []float32{3, 4},
				},
			},
		},
	}

	data, err := json.Marshal(modelState)
	if err != nil {
		t.Fatalf("Marshal failed: %v", err)
	}

	modelPath := filepath.Join(t.TempDir(), "dqn-agent.json")
	if err := os.WriteFile(modelPath, data, 0600); err != nil {
		t.Fatalf("WriteFile failed: %v", err)
	}

	agent, err := NewDQNAgent("")
	if err != nil {
		t.Fatalf("NewDQNAgent failed: %v", err)
	}
	defer agent.Destroy()

	if err := agent.LoadModel(modelPath); err != nil {
		t.Fatalf("LoadModel failed: %v", err)
	}

	if agent.replayBuffer.Size() != 1 {
		t.Fatalf("Expected only valid replay experience to load, got %d", agent.replayBuffer.Size())
	}
	loaded := agent.replayBuffer.Sample(1)
	if len(loaded) != 1 || loaded[0].Action != ActionStream2 || loaded[0].Reward != 2 {
		t.Fatalf("Loaded replay experience mismatch: %#v", loaded)
	}
}

func TestActionDecoding(t *testing.T) {
	agent, err := NewDQNAgent("nonexistent.onnx")
	if err != nil {
		t.Skipf("ONNX Runtime not available, skipping: %v", err)
		return
	}
	defer agent.Destroy()

	state := NewEnvironmentState()
	state.TaskSize = 1000000

	tests := []struct {
		action          Action
		expectedStreams int
	}{
		{ActionStream1, 1},
		{ActionStream2, 1},
		{ActionStream3, 1},
		{ActionStream4, 1},
		{ActionSplit12, 2},
		{ActionSplit13, 2},
		{ActionSplit34, 2},
		{ActionSplit123, 3},
		{ActionSplit234, 3},
		{ActionSplitAll, 4},
	}

	for _, tt := range tests {
		t.Run(tt.action.String(), func(t *testing.T) {
			decision := &TaskPartitionDecision{}
			result := agent.decodeAction(tt.action, state, decision)

			if len(result.StreamIDs) != tt.expectedStreams {
				t.Errorf("Expected %d streams, got %d", tt.expectedStreams, len(result.StreamIDs))
			}

			if len(result.ChunkSizes) != tt.expectedStreams {
				t.Errorf("Expected %d chunk sizes, got %d", tt.expectedStreams, len(result.ChunkSizes))
			}
		})
	}
}

func TestChunkSizeCalculation(t *testing.T) {
	agent, err := NewDQNAgent("nonexistent.onnx")
	if err != nil {
		t.Skipf("ONNX Runtime not available, skipping: %v", err)
		return
	}
	defer agent.Destroy()

	state := NewEnvironmentState()
	state.StreamBandwidth = [4]float64{100, 200, 150, 50}
	state.StreamSuccessRate = [4]float64{0.9, 0.95, 0.92, 0.85}
	state.StreamCongestion = [4]float64{0.1, 0.05, 0.15, 0.3}

	taskSize := 1000000
	streams := []int{0, 1, 2}

	chunks := agent.calculateChunkSizes(taskSize, len(streams), streams, state)

	// Verify total equals task size
	total := 0
	for _, chunk := range chunks {
		total += chunk
	}

	if total != taskSize {
		t.Errorf("Total chunks %d != task size %d", total, taskSize)
	}

	// Verify proportional allocation (stream 1 should get more than stream 0)
	if chunks[1] <= chunks[0] {
		t.Error("Expected stream with higher bandwidth to get larger chunk")
	}
}

func TestTimeEstimation(t *testing.T) {
	agent, err := NewDQNAgent("nonexistent.onnx")
	if err != nil {
		t.Skipf("ONNX Runtime not available, skipping: %v", err)
		return
	}
	defer agent.Destroy()

	state := NewEnvironmentState()
	state.StreamBandwidth = [4]float64{100, 100, 100, 100} // 100 Mbps
	state.StreamLatency = [4]float64{10, 10, 10, 10}       // 10ms
	state.StreamCongestion = [4]float64{0, 0, 0, 0}
	state.StreamSuccessRate = [4]float64{1, 1, 1, 1}

	taskSize := 100000000 // 100 MB

	// Single stream
	time1 := agent.estimateTime(taskSize, []int{0}, state)

	// Two streams (should be faster)
	time2 := agent.estimateTime(taskSize, []int{0, 1}, state)

	// Four streams (should be fastest)
	time4 := agent.estimateTime(taskSize, []int{0, 1, 2, 3}, state)

	if time4 >= time2 || time2 >= time1 {
		t.Error("Expected faster completion with more streams")
	}

	if time1 <= 0 || time2 <= 0 || time4 <= 0 {
		t.Error("Invalid time estimates")
	}
}

func TestDQNAgentHandlesInvalidTelemetry(t *testing.T) {
	agent, err := NewDQNAgent("")
	if err != nil {
		t.Fatalf("NewDQNAgent failed: %v", err)
	}
	defer agent.Destroy()

	state := NewEnvironmentState()
	state.TaskSize = 1024
	state.TaskPriority = 0.1
	state.StreamBandwidth = [4]float64{0, -10, math.NaN(), math.Inf(1)}
	state.StreamLatency = [4]float64{0, -5, math.NaN(), math.Inf(1)}
	state.StreamCongestion = [4]float64{-1, 2, math.NaN(), math.Inf(1)}
	state.StreamSuccessRate = [4]float64{0, -0.2, math.NaN(), math.Inf(1)}

	streams := []int{0, 1, 2, 3}
	chunks := agent.calculateChunkSizes(state.TaskSize, len(streams), streams, state)
	if len(chunks) != len(streams) {
		t.Fatalf("Expected %d chunks, got %d", len(streams), len(chunks))
	}

	total := 0
	for i, chunk := range chunks {
		if chunk <= 0 {
			t.Fatalf("Expected positive chunk %d, got %d", i, chunk)
		}
		total += chunk
	}
	if total != state.TaskSize {
		t.Fatalf("Total chunks %d != task size %d", total, state.TaskSize)
	}

	estimated := agent.estimateTime(state.TaskSize, streams, state)
	if estimated <= 0 || estimated > time.Hour {
		t.Fatalf("Expected bounded positive estimate, got %s", estimated)
	}

	action := agent.heuristicAction(state)
	if action < 0 || action >= Action(NumActions) {
		t.Fatalf("Invalid heuristic action: %d", action)
	}
}

func TestExplorationExploitation(t *testing.T) {
	agent, err := NewDQNAgent("nonexistent.onnx")
	if err != nil {
		t.Skipf("ONNX Runtime not available, skipping: %v", err)
		return
	}
	defer agent.Destroy()

	state := NewEnvironmentState()

	// High epsilon - should mostly explore
	agent.epsilon = 1.0
	explorationCount := 0

	for i := 0; i < 100; i++ {
		decision, _ := agent.SelectAction(state)
		if decision.ExplorationUsed {
			explorationCount++
		}
	}

	if explorationCount < 90 {
		t.Errorf("Expected >90%% exploration with epsilon=1.0, got %d%%", explorationCount)
	}

	// Low epsilon - should mostly exploit
	agent.epsilon = 0.0
	explorationCount = 0

	for i := 0; i < 100; i++ {
		decision, _ := agent.SelectAction(state)
		if decision.ExplorationUsed {
			explorationCount++
		}
	}

	if explorationCount > 5 {
		t.Errorf("Expected <5%% exploration with epsilon=0.0, got %d%%", explorationCount)
	}
}

func TestMemoryStorage(t *testing.T) {
	agent, err := NewDQNAgent("nonexistent.onnx")
	if err != nil {
		t.Skipf("ONNX Runtime not available, skipping: %v", err)
		return
	}
	defer agent.Destroy()

	state := NewEnvironmentState()
	nextState := NewEnvironmentState()
	nextState.TaskQueueDepth = 5

	// Store experiences
	for i := 0; i < 50; i++ {
		agent.Remember(state, Action(i%15), float64(i), nextState, false)
	}

	// Final experience with done=true
	agent.Remember(state, ActionStream1, 100.0, nextState, true)

	if agent.replayBuffer.Size() != 51 {
		t.Errorf("Expected 51 experiences, got %d", agent.replayBuffer.Size())
	}

	if len(agent.episodeRewards) != 1 {
		t.Errorf("Expected 1 episode reward recorded, got %d", len(agent.episodeRewards))
	}
}

func TestDQNAgentRememberIgnoresInvalidExperience(t *testing.T) {
	tests := []struct {
		name      string
		state     *EnvironmentState
		action    Action
		reward    float64
		nextState *EnvironmentState
	}{
		{
			name:      "nil_state",
			state:     nil,
			action:    ActionStream1,
			reward:    1,
			nextState: NewEnvironmentState(),
		},
		{
			name:      "nil_next_state",
			state:     NewEnvironmentState(),
			action:    ActionStream1,
			reward:    1,
			nextState: nil,
		},
		{
			name:      "invalid_action",
			state:     NewEnvironmentState(),
			action:    Action(NumActions),
			reward:    1,
			nextState: NewEnvironmentState(),
		},
		{
			name:      "nan_reward",
			state:     NewEnvironmentState(),
			action:    ActionStream1,
			reward:    math.NaN(),
			nextState: NewEnvironmentState(),
		},
		{
			name:      "infinite_reward",
			state:     NewEnvironmentState(),
			action:    ActionStream1,
			reward:    math.Inf(1),
			nextState: NewEnvironmentState(),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			agent, err := NewDQNAgent("nonexistent.onnx")
			if err != nil {
				t.Skipf("ONNX Runtime not available, skipping: %v", err)
				return
			}
			defer agent.Destroy()

			beforeSize := agent.replayBuffer.Size()
			beforeReward := agent.totalReward

			defer func() {
				if recovered := recover(); recovered != nil {
					t.Fatalf("invalid experience should be ignored without panic: %v", recovered)
				}
			}()

			agent.Remember(tt.state, tt.action, tt.reward, tt.nextState, false)

			if agent.replayBuffer.Size() != beforeSize {
				t.Fatalf("invalid experience should not grow replay buffer: before=%d after=%d", beforeSize, agent.replayBuffer.Size())
			}
			if agent.totalReward != beforeReward {
				t.Fatalf("invalid experience should not change total reward: before=%v after=%v", beforeReward, agent.totalReward)
			}
		})
	}
}

func TestEpsilonDecay(t *testing.T) {
	agent, err := NewDQNAgent("nonexistent.onnx")
	if err != nil {
		t.Skipf("ONNX Runtime not available, skipping: %v", err)
		return
	}
	defer agent.Destroy()

	initialEpsilon := agent.epsilon

	// Update epsilon multiple times
	for i := 0; i < 100; i++ {
		agent.UpdateEpsilon()
	}

	if agent.epsilon >= initialEpsilon {
		t.Error("Epsilon should decay over time")
	}

	if agent.epsilon < agent.epsilonMin {
		t.Errorf("Epsilon %f below minimum %f", agent.epsilon, agent.epsilonMin)
	}

	// Continue decaying - should not go below min
	for i := 0; i < 10000; i++ {
		agent.UpdateEpsilon()
	}

	if agent.epsilon != agent.epsilonMin {
		t.Errorf("Expected epsilon to reach minimum %f, got %f", agent.epsilonMin, agent.epsilon)
	}
}

func BenchmarkStateVectorization(b *testing.B) {
	state := NewEnvironmentState()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = state.ToVector()
	}
}

func BenchmarkSelectAction(b *testing.B) {
	agent, err := NewDQNAgent("nonexistent.onnx")
	if err != nil {
		b.Skipf("ONNX Runtime not available, skipping: %v", err)
		return
	}
	defer agent.Destroy()

	state := NewEnvironmentState()
	agent.epsilon = 0 // Disable exploration for consistent benchmarking

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = agent.SelectAction(state)
	}
}

func BenchmarkRewardCalculation(b *testing.B) {
	calc := NewRewardCalculator()
	outcome := &ActionOutcome{
		ActualThroughput:   120.0,
		BaselineThroughput: 100.0,
		ActualLatency:      12.0,
		TargetLatency:      10.0,
		StreamImbalance:    0.2,
		Completed:          true,
		Retransmissions:    0,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = calc.Calculate(outcome)
	}
}

func BenchmarkEnvironmentStep(b *testing.B) {
	sim := NewEnvironmentSimulator()
	sim.Reset()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, _ = sim.Step(Action(i % 15))
	}
}

// Helper method for Action string representation (for testing)
func (a Action) String() string {
	names := []string{
		"Stream1", "Stream2", "Stream3", "Stream4",
		"Split12", "Split13", "Split14", "Split23", "Split24", "Split34",
		"Split123", "Split124", "Split134", "Split234",
		"SplitAll",
	}
	if int(a) < len(names) {
		return names[a]
	}
	return "Unknown"
}
