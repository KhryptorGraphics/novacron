package partition

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"sync"
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

func TestEnvironmentStateToVectorSanitizesInvalidValues(t *testing.T) {
	state := NewEnvironmentState()
	state.StreamBandwidth = [4]float64{math.NaN(), math.Inf(1), -10, 2000}
	state.StreamLatency = [4]float64{math.NaN(), math.Inf(1), -5, 250}
	state.StreamCongestion = [4]float64{math.NaN(), math.Inf(1), -1, 2}
	state.StreamSuccessRate = [4]float64{math.NaN(), math.Inf(1), -0.1, 2}
	state.TaskQueueDepth = -10
	state.TaskSize = -1024
	state.TaskPriority = math.Inf(1)
	state.TimeOfDay = math.NaN()

	vector := state.ToVector()
	if len(vector) != 20 {
		t.Fatalf("Expected vector length 20, got %d", len(vector))
	}
	for i, value := range vector {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			t.Fatalf("Vector element %d is non-finite: %f", i, value)
		}
		if value < 0 || value > 1 {
			t.Fatalf("Vector element %d is out of normalized range: %f", i, value)
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

func TestRewardCalculatorHandlesInvalidOutcome(t *testing.T) {
	calc := NewRewardCalculator()

	tests := []struct {
		name    string
		outcome *ActionOutcome
	}{
		{
			name:    "nil_outcome",
			outcome: nil,
		},
		{
			name: "zero_baseline_and_target",
			outcome: &ActionOutcome{
				ActualThroughput:   100,
				BaselineThroughput: 0,
				ActualLatency:      10,
				TargetLatency:      0,
				StreamImbalance:    math.NaN(),
				Completed:          true,
				Retransmissions:    -1,
			},
		},
		{
			name: "non_finite_values",
			outcome: &ActionOutcome{
				ActualThroughput:   math.Inf(1),
				BaselineThroughput: math.NaN(),
				ActualLatency:      math.Inf(-1),
				TargetLatency:      math.NaN(),
				StreamImbalance:    math.Inf(1),
				Retransmissions:    1,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if recovered := recover(); recovered != nil {
					t.Fatalf("invalid reward outcome should not panic: %v", recovered)
				}
			}()

			reward := calc.Calculate(tt.outcome)
			if math.IsNaN(reward) || math.IsInf(reward, 0) {
				t.Fatalf("invalid reward outcome should produce finite reward, got %v", reward)
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

func TestNewOnlineLearnerRejectsNilAgent(t *testing.T) {
	defer func() {
		if recovered := recover(); recovered != nil {
			t.Fatalf("nil agent should be rejected without panic: %v", recovered)
		}
	}()

	learner := NewOnlineLearner(nil, &OnlineLearnerConfig{EnableAutoUpdate: false})
	if learner != nil {
		t.Fatalf("nil agent should not create learner: %#v", learner)
	}
}

func TestNewOnlineLearnerInitializesMissingReplayBuffer(t *testing.T) {
	agent := &DQNAgent{}
	learner := NewOnlineLearner(agent, &OnlineLearnerConfig{
		UpdateFrequency:  time.Hour,
		MinExperiences:   1,
		TrainingScript:   "unused",
		ModelPath:        filepath.Join(t.TempDir(), "model"),
		EnableAutoUpdate: false,
	})
	if learner == nil {
		t.Fatal("zero-value agent should create learner with initialized replay buffer")
	}
	defer learner.Stop()

	defer func() {
		if recovered := recover(); recovered != nil {
			t.Fatalf("zero-value agent learner should collect without panic: %v", recovered)
		}
	}()

	learner.CollectExperience(NewEnvironmentState(), ActionStream1, 1, NewEnvironmentState(), false)

	status := learner.GetStatus()
	if status["buffer_size"] != 1 {
		t.Fatalf("expected initialized replay buffer to collect one experience, got status=%v", status)
	}
}

func TestNewOnlineLearnerNormalizesInvalidConfig(t *testing.T) {
	agent, err := NewDQNAgent("")
	if err != nil {
		t.Fatalf("failed to create DQN agent: %v", err)
	}

	learner := NewOnlineLearner(agent, &OnlineLearnerConfig{
		UpdateFrequency:  -time.Hour,
		MinExperiences:   -10,
		TrainingScript:   "",
		ModelPath:        "",
		EnableAutoUpdate: false,
	})
	if learner == nil {
		t.Fatal("invalid config should be normalized, not reject learner")
	}
	defer learner.Stop()

	status := learner.GetStatus()
	if status["min_experiences"] != 1000 {
		t.Fatalf("invalid min experiences should fall back to 1000, got %v", status["min_experiences"])
	}
	if learner.updateFrequency != 24*time.Hour {
		t.Fatalf("invalid update frequency should fall back to 24h, got %s", learner.updateFrequency)
	}
	if learner.trainingScript != "training/train_dqn.py" {
		t.Fatalf("empty training script should fall back to default, got %q", learner.trainingScript)
	}
	if learner.modelPath != "models/dqn_online" {
		t.Fatalf("empty model path should fall back to default, got %q", learner.modelPath)
	}

	if err := learner.ForceUpdate(); err == nil {
		t.Fatal("normalized min experiences should prevent force update with empty buffer")
	}
}

func TestOnlineLearnerExportExperiencesSkipsInvalidSamples(t *testing.T) {
	agent, err := NewDQNAgent("")
	if err != nil {
		t.Fatalf("failed to create DQN agent: %v", err)
	}

	learner := NewOnlineLearner(agent, &OnlineLearnerConfig{
		UpdateFrequency:  time.Hour,
		MinExperiences:   1,
		TrainingScript:   "unused",
		ModelPath:        filepath.Join(t.TempDir(), "model"),
		EnableAutoUpdate: false,
	})
	if learner == nil {
		t.Fatal("failed to create learner")
	}
	defer learner.Stop()

	learner.replayBuffer.Add(&Experience{
		State:     []float32{1, 2},
		Action:    ActionStream1,
		Reward:    1,
		NextState: []float32{2, 3},
		TDError:   0.25,
	})
	learner.replayBuffer.Add(&Experience{
		State:     nil,
		Action:    ActionStream1,
		Reward:    2,
		NextState: []float32{2, 3},
	})
	learner.replayBuffer.Add(&Experience{
		State:     []float32{1, 2},
		Action:    Action(NumActions),
		Reward:    2,
		NextState: []float32{2, 3},
	})
	learner.replayBuffer.Add(&Experience{
		State:     []float32{1, 2},
		Action:    ActionStream1,
		Reward:    math.NaN(),
		NextState: []float32{2, 3},
	})
	learner.replayBuffer.Add(&Experience{
		State:     []float32{1, 2},
		Action:    ActionStream1,
		Reward:    2,
		NextState: []float32{2, 3},
		TDError:   math.Inf(1),
	})

	exportPath := filepath.Join(t.TempDir(), "experiences.json")
	if err := learner.exportExperiences(exportPath); err != nil {
		t.Fatalf("export should skip invalid samples without failing: %v", err)
	}

	data, err := os.ReadFile(exportPath)
	if err != nil {
		t.Fatalf("failed to read exported experiences: %v", err)
	}

	var exported []struct {
		State     []float32 `json:"state"`
		Action    int       `json:"action"`
		Reward    float64   `json:"reward"`
		NextState []float32 `json:"next_state"`
		TDError   float64   `json:"td_error"`
	}
	if err := json.Unmarshal(data, &exported); err != nil {
		t.Fatalf("exported experiences should be valid JSON: %v", err)
	}
	if len(exported) != 1 {
		t.Fatalf("expected only the valid experience to be exported, got %d: %s", len(exported), data)
	}
	if exported[0].Action != int(ActionStream1) || exported[0].Reward != 1 || exported[0].TDError != 0.25 {
		t.Fatalf("unexpected exported experience: %+v", exported[0])
	}
}

func TestOnlineLearnerExportExperiencesCreatesParentDirectory(t *testing.T) {
	agent, err := NewDQNAgent("")
	if err != nil {
		t.Fatalf("failed to create DQN agent: %v", err)
	}

	learner := NewOnlineLearner(agent, &OnlineLearnerConfig{
		UpdateFrequency:  time.Hour,
		MinExperiences:   1,
		TrainingScript:   "unused",
		ModelPath:        filepath.Join(t.TempDir(), "model"),
		EnableAutoUpdate: false,
	})
	if learner == nil {
		t.Fatal("failed to create learner")
	}
	defer learner.Stop()

	learner.replayBuffer.Add(&Experience{
		State:     []float32{1},
		Action:    ActionStream1,
		Reward:    1,
		NextState: []float32{2},
	})

	exportPath := filepath.Join(t.TempDir(), "nested", "training", "experiences.json")
	if err := learner.exportExperiences(exportPath); err != nil {
		t.Fatalf("export should create parent directories: %v", err)
	}
	if _, err := os.Stat(exportPath); err != nil {
		t.Fatalf("exported file missing: %v", err)
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

func TestOnlineLearnerStoppedRejectsQueuedUpdate(t *testing.T) {
	agent, err := NewDQNAgent("")
	if err != nil {
		t.Fatalf("failed to create DQN agent: %v", err)
	}

	modelPath := filepath.Join(t.TempDir(), "nested", "model")
	learner := NewOnlineLearner(agent, &OnlineLearnerConfig{
		UpdateFrequency:  time.Hour,
		MinExperiences:   1,
		TrainingScript:   "unused",
		ModelPath:        modelPath,
		EnableAutoUpdate: false,
	})
	if learner == nil {
		t.Fatal("failed to create learner")
	}

	learner.replayBuffer.Add(&Experience{
		State:     []float32{1},
		Action:    ActionStream1,
		Reward:    1,
		NextState: []float32{2},
	})

	learner.Stop()
	learner.triggerUpdate()

	exportPath := modelPath + "_experiences.json"
	if _, err := os.Stat(exportPath); !os.IsNotExist(err) {
		t.Fatalf("stopped queued update should not export experiences, stat err=%v", err)
	}
	status := learner.GetStatus()
	if status["is_training"] != false {
		t.Fatalf("stopped queued update should not leave training active: %v", status)
	}
	if status["update_count"] != 0 {
		t.Fatalf("stopped queued update should not increment updates: %v", status)
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

func TestEnvironmentSimulatorStepIgnoresInvalidAction(t *testing.T) {
	tests := []struct {
		name     string
		action   Action
		nilState bool
		wantDone bool
	}{
		{name: "negative_action", action: Action(-1)},
		{name: "too_large_action", action: Action(NumActions)},
		{name: "nil_state", action: Action(-1), nilState: true, wantDone: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sim := NewEnvironmentSimulator()
			if tt.nilState {
				sim.state = nil
			} else {
				sim.state = NewEnvironmentState()
				sim.state.TaskQueueDepth = 3
				sim.state.TimeOfDay = 0.25
			}

			var beforeBandwidth, beforeLatency, beforeCongestion, beforeSuccessRate [4]float64
			var beforeQueueDepth, beforeTaskSize int
			var beforeTaskPriority, beforeTimeOfDay float64
			if sim.state != nil {
				beforeBandwidth = sim.state.StreamBandwidth
				beforeLatency = sim.state.StreamLatency
				beforeCongestion = sim.state.StreamCongestion
				beforeSuccessRate = sim.state.StreamSuccessRate
				beforeQueueDepth = sim.state.TaskQueueDepth
				beforeTaskSize = sim.state.TaskSize
				beforeTaskPriority = sim.state.TaskPriority
				beforeTimeOfDay = sim.state.TimeOfDay
			}

			defer func() {
				if recovered := recover(); recovered != nil {
					t.Fatalf("invalid simulator action should not panic: %v", recovered)
				}
			}()

			nextState, reward, done := sim.Step(tt.action)

			if nextState != sim.state {
				t.Fatal("invalid simulator action should return the current state")
			}
			if reward != 0 {
				t.Fatalf("invalid simulator action should return zero reward, got %f", reward)
			}
			if done != tt.wantDone {
				t.Fatalf("invalid simulator action done mismatch: got %v want %v", done, tt.wantDone)
			}
			if sim.state == nil {
				return
			}
			if sim.state.StreamBandwidth != beforeBandwidth ||
				sim.state.StreamLatency != beforeLatency ||
				sim.state.StreamCongestion != beforeCongestion ||
				sim.state.StreamSuccessRate != beforeSuccessRate ||
				sim.state.TaskQueueDepth != beforeQueueDepth ||
				sim.state.TaskSize != beforeTaskSize ||
				sim.state.TaskPriority != beforeTaskPriority ||
				sim.state.TimeOfDay != beforeTimeOfDay {
				t.Fatal("invalid simulator action should not mutate environment state")
			}
		})
	}
}

func TestEnvironmentSimulatorStepInitializesZeroValueSimulator(t *testing.T) {
	var sim EnvironmentSimulator

	defer func() {
		if recovered := recover(); recovered != nil {
			t.Fatalf("zero-value simulator should not panic on valid action: %v", recovered)
		}
	}()

	nextState, reward, done := sim.Step(ActionStream1)

	if nextState == nil {
		t.Fatal("zero-value simulator should initialize state")
	}
	if sim.rewardCalc == nil {
		t.Fatal("zero-value simulator should initialize reward calculator")
	}
	if math.IsNaN(reward) || math.IsInf(reward, 0) {
		t.Fatalf("zero-value simulator should return finite reward, got %f", reward)
	}
	if done != (nextState.TaskQueueDepth == 0) {
		t.Fatalf("done mismatch for zero-value simulator: done=%v queue=%d", done, nextState.TaskQueueDepth)
	}
}

func TestEnvironmentSimulatorStepSanitizesInvalidMutableState(t *testing.T) {
	sim := NewEnvironmentSimulator()
	sim.state.StreamBandwidth = [4]float64{math.NaN(), math.Inf(1), -100, 0}
	sim.state.StreamLatency = [4]float64{math.NaN(), math.Inf(-1), -5, 0}
	sim.state.StreamCongestion = [4]float64{math.NaN(), math.Inf(1), -0.5, 2}
	sim.state.StreamSuccessRate = [4]float64{math.NaN(), math.Inf(-1), -0.25, 2}

	defer func() {
		if recovered := recover(); recovered != nil {
			t.Fatalf("invalid mutable state should not panic during step: %v", recovered)
		}
	}()

	nextState, reward, _ := sim.Step(ActionSplitAll)

	if math.IsNaN(reward) || math.IsInf(reward, 0) {
		t.Fatalf("invalid mutable state should produce finite reward, got %f", reward)
	}
	for i := 0; i < 4; i++ {
		if math.IsNaN(nextState.StreamBandwidth[i]) || math.IsInf(nextState.StreamBandwidth[i], 0) ||
			nextState.StreamBandwidth[i] <= 0 {
			t.Fatalf("stream %d bandwidth should be finite and positive, got %f", i, nextState.StreamBandwidth[i])
		}
		if math.IsNaN(nextState.StreamLatency[i]) || math.IsInf(nextState.StreamLatency[i], 0) ||
			nextState.StreamLatency[i] <= 0 {
			t.Fatalf("stream %d latency should be finite and positive, got %f", i, nextState.StreamLatency[i])
		}
		if math.IsNaN(nextState.StreamCongestion[i]) || math.IsInf(nextState.StreamCongestion[i], 0) ||
			nextState.StreamCongestion[i] < 0 || nextState.StreamCongestion[i] > 1 {
			t.Fatalf("stream %d congestion should be finite and normalized, got %f", i, nextState.StreamCongestion[i])
		}
		if math.IsNaN(nextState.StreamSuccessRate[i]) || math.IsInf(nextState.StreamSuccessRate[i], 0) ||
			nextState.StreamSuccessRate[i] < 0 || nextState.StreamSuccessRate[i] > 1 {
			t.Fatalf("stream %d success rate should be finite and normalized, got %f", i, nextState.StreamSuccessRate[i])
		}
	}
}

func TestCalculateImbalanceHandlesInvalidInput(t *testing.T) {
	tests := []struct {
		name    string
		state   *EnvironmentState
		streams []int
	}{
		{
			name:    "nil_state",
			state:   nil,
			streams: []int{0, 1},
		},
		{
			name:    "invalid_stream_indices",
			state:   NewEnvironmentState(),
			streams: []int{-1, 4},
		},
		{
			name:  "invalid_bandwidth_values",
			state: &EnvironmentState{StreamBandwidth: [4]float64{0, math.NaN(), math.Inf(1), -1}},
			streams: []int{
				0,
				1,
				2,
				3,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if recovered := recover(); recovered != nil {
					t.Fatalf("invalid imbalance input should not panic: %v", recovered)
				}
			}()

			imbalance := calculateImbalance(tt.state, tt.streams)
			if math.IsNaN(imbalance) || math.IsInf(imbalance, 0) {
				t.Fatalf("invalid imbalance input should produce finite result, got %f", imbalance)
			}
			if imbalance < 0 {
				t.Fatalf("imbalance should not be negative, got %f", imbalance)
			}
		})
	}
}

func TestStreamIndexHelpersIgnoreInvalidActions(t *testing.T) {
	tests := []struct {
		name    string
		streams []int
	}{
		{name: "two_stream_helper", streams: getTwoStreamIndices(ActionStream1)},
		{name: "three_stream_helper", streams: getThreeStreamIndices(ActionSplit12)},
		{name: "used_streams_negative", streams: getUsedStreams(Action(-1))},
		{name: "used_streams_too_large", streams: getUsedStreams(Action(NumActions))},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if len(tt.streams) != 0 {
				t.Fatalf("invalid action should not map to streams, got %v", tt.streams)
			}
		})
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

func TestDQNAgentSelectActionRejectsNilState(t *testing.T) {
	agent, err := NewDQNAgent("")
	if err != nil {
		t.Fatalf("NewDQNAgent failed: %v", err)
	}
	defer agent.Destroy()

	defer func() {
		if recovered := recover(); recovered != nil {
			t.Fatalf("nil state should return an error without panic: %v", recovered)
		}
	}()

	decision, err := agent.SelectAction(nil)
	if err == nil {
		t.Fatalf("expected nil state error, got decision=%v", decision)
	}
	if decision != nil {
		t.Fatalf("nil state should not produce a decision: %#v", decision)
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

func TestDQNAgentSaveModelCreatesParentDirectory(t *testing.T) {
	agent, err := NewDQNAgent("")
	if err != nil {
		t.Fatalf("NewDQNAgent failed: %v", err)
	}
	defer agent.Destroy()

	modelPath := filepath.Join(t.TempDir(), "nested", "state", "dqn-agent.json")
	if err := agent.SaveModel(modelPath); err != nil {
		t.Fatalf("SaveModel should create parent directories: %v", err)
	}
	if _, err := os.Stat(modelPath); err != nil {
		t.Fatalf("saved model file missing: %v", err)
	}
}

func TestDQNAgentSaveModelSkipsInvalidReplayExperiences(t *testing.T) {
	agent, err := NewDQNAgent("")
	if err != nil {
		t.Fatalf("NewDQNAgent failed: %v", err)
	}
	defer agent.Destroy()

	agent.replayBuffer.Add(&Experience{
		State:     []float32{1},
		Action:    ActionStream1,
		Reward:    1,
		NextState: []float32{2},
		TDError:   0.25,
	})
	agent.replayBuffer.Add(&Experience{
		State:     []float32{float32(math.NaN())},
		Action:    ActionStream1,
		Reward:    2,
		NextState: []float32{3},
	})
	agent.replayBuffer.Add(&Experience{
		State:     []float32{1},
		Action:    ActionStream1,
		Reward:    3,
		NextState: []float32{2},
		TDError:   math.Inf(1),
	})

	modelPath := filepath.Join(t.TempDir(), "dqn-agent.json")
	if err := agent.SaveModel(modelPath); err != nil {
		t.Fatalf("SaveModel should skip invalid replay samples: %v", err)
	}

	loaded, err := NewDQNAgent("")
	if err != nil {
		t.Fatalf("NewDQNAgent failed: %v", err)
	}
	defer loaded.Destroy()

	if err := loaded.LoadModel(modelPath); err != nil {
		t.Fatalf("LoadModel failed: %v", err)
	}
	if loaded.replayBuffer.Size() != 1 {
		t.Fatalf("expected only valid replay experience to persist, got %d", loaded.replayBuffer.Size())
	}
}

func TestDQNAgentSaveModelSanitizesNonFiniteState(t *testing.T) {
	agent, err := NewDQNAgent("")
	if err != nil {
		t.Fatalf("NewDQNAgent failed: %v", err)
	}
	defer agent.Destroy()

	agent.mu.Lock()
	agent.epsilon = math.NaN()
	agent.epsilonMin = math.Inf(1)
	agent.epsilonDecay = math.Inf(-1)
	agent.stateBuffer = []float32{1, float32(math.NaN()), 2, float32(math.Inf(1))}
	agent.learningRate = math.NaN()
	agent.gamma = math.Inf(1)
	agent.totalReward = math.NaN()
	agent.episodeRewards = []float64{1, math.NaN(), math.Inf(1), 3}
	agent.successRate = math.Inf(-1)
	agent.mu.Unlock()

	modelPath := filepath.Join(t.TempDir(), "dqn-agent.json")
	if err := agent.SaveModel(modelPath); err != nil {
		t.Fatalf("SaveModel should sanitize non-finite state: %v", err)
	}

	loaded, err := NewDQNAgent("")
	if err != nil {
		t.Fatalf("NewDQNAgent failed: %v", err)
	}
	defer loaded.Destroy()

	if err := loaded.LoadModel(modelPath); err != nil {
		t.Fatalf("LoadModel failed: %v", err)
	}

	metrics := loaded.GetMetrics()
	avgReward, ok := metrics["average_reward"].(float64)
	if !ok {
		t.Fatalf("average_reward has unexpected type %T", metrics["average_reward"])
	}
	if avgReward != 2 {
		t.Fatalf("expected persisted finite episode rewards only, got average %v", avgReward)
	}
	for i, value := range loaded.stateBuffer {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			t.Fatalf("loaded state buffer index %d should be finite, got %v", i, value)
		}
	}
	if math.IsNaN(loaded.totalReward) || math.IsInf(loaded.totalReward, 0) {
		t.Fatalf("loaded total reward should be finite, got %v", loaded.totalReward)
	}
}

func TestDQNAgentSaveModelSanitizesInvalidIntegerState(t *testing.T) {
	agent, err := NewDQNAgent("")
	if err != nil {
		t.Fatalf("NewDQNAgent failed: %v", err)
	}
	defer agent.Destroy()

	agent.mu.Lock()
	agent.updateFreq = -10
	agent.stepCount = -7
	agent.mu.Unlock()

	modelPath := filepath.Join(t.TempDir(), "dqn-agent.json")
	if err := agent.SaveModel(modelPath); err != nil {
		t.Fatalf("SaveModel failed: %v", err)
	}

	data, err := os.ReadFile(modelPath)
	if err != nil {
		t.Fatalf("ReadFile failed: %v", err)
	}

	var state dqnAgentState
	if err := json.Unmarshal(data, &state); err != nil {
		t.Fatalf("Unmarshal failed: %v", err)
	}
	if state.UpdateFreq != 1000 {
		t.Fatalf("expected invalid update frequency to persist as default 1000, got %d", state.UpdateFreq)
	}
	if state.StepCount != 0 {
		t.Fatalf("expected invalid step count to persist as 0, got %d", state.StepCount)
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

func TestDQNAgentLoadModelSanitizesInvalidScalarState(t *testing.T) {
	modelState := dqnAgentState{
		Version:        dqnAgentStateVersion,
		Epsilon:        2,
		EpsilonMin:     -0.1,
		EpsilonDecay:   -0.5,
		LearningRate:   -1,
		Gamma:          1.5,
		UpdateFreq:     -10,
		StepCount:      -7,
		TotalReward:    -5,
		EpisodeRewards: []float64{1, 3},
		SuccessRate:    2,
		ReplayBuffer: replayState{
			Capacity: 1,
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

	if agent.epsilon != 1 {
		t.Fatalf("invalid epsilon should fall back to default 1, got %v", agent.epsilon)
	}
	if agent.epsilonMin != 0.01 {
		t.Fatalf("invalid epsilonMin should fall back to default 0.01, got %v", agent.epsilonMin)
	}
	if agent.epsilonDecay != 0.995 {
		t.Fatalf("invalid epsilonDecay should fall back to default 0.995, got %v", agent.epsilonDecay)
	}
	if agent.learningRate != 0.001 {
		t.Fatalf("invalid learningRate should fall back to default 0.001, got %v", agent.learningRate)
	}
	if agent.gamma != 0.95 {
		t.Fatalf("invalid gamma should fall back to default 0.95, got %v", agent.gamma)
	}
	if agent.updateFreq != 1000 {
		t.Fatalf("invalid updateFreq should fall back to default 1000, got %v", agent.updateFreq)
	}
	if agent.stepCount != 0 {
		t.Fatalf("invalid stepCount should fall back to default 0, got %v", agent.stepCount)
	}
	if agent.totalReward != -5 {
		t.Fatalf("finite totalReward should be preserved, got %v", agent.totalReward)
	}
	if agent.successRate != 0 {
		t.Fatalf("invalid successRate should fall back to default 0, got %v", agent.successRate)
	}

	metrics := agent.GetMetrics()
	avgReward, ok := metrics["average_reward"].(float64)
	if !ok {
		t.Fatalf("average_reward has unexpected type %T", metrics["average_reward"])
	}
	if avgReward != 2 {
		t.Fatalf("expected valid episode rewards to remain average 2, got %v", avgReward)
	}
}

func TestDQNAgentGetMetricsIgnoresNonFiniteEpisodeRewards(t *testing.T) {
	agent, err := NewDQNAgent("")
	if err != nil {
		t.Fatalf("NewDQNAgent failed: %v", err)
	}
	defer agent.Destroy()

	agent.mu.Lock()
	agent.episodeRewards = []float64{1, math.NaN(), math.Inf(1), 3}
	agent.mu.Unlock()

	metrics := agent.GetMetrics()
	avgReward, ok := metrics["average_reward"].(float64)
	if !ok {
		t.Fatalf("average_reward has unexpected type %T", metrics["average_reward"])
	}
	if math.IsNaN(avgReward) || math.IsInf(avgReward, 0) {
		t.Fatalf("average_reward should remain finite, got %v", avgReward)
	}
	if avgReward != 2 {
		t.Fatalf("expected average reward from finite samples only, got %v", avgReward)
	}
}

func TestDQNAgentGetMetricsSanitizesInvalidScalarState(t *testing.T) {
	agent, err := NewDQNAgent("")
	if err != nil {
		t.Fatalf("NewDQNAgent failed: %v", err)
	}
	defer agent.Destroy()

	agent.mu.Lock()
	agent.epsilon = math.NaN()
	agent.successRate = math.Inf(1)
	agent.stepCount = -7
	agent.mu.Unlock()

	metrics := agent.GetMetrics()
	epsilon, ok := metrics["epsilon"].(float64)
	if !ok {
		t.Fatalf("epsilon has unexpected type %T", metrics["epsilon"])
	}
	if math.IsNaN(epsilon) || math.IsInf(epsilon, 0) || epsilon < 0 || epsilon > 1 {
		t.Fatalf("epsilon metric should be finite and bounded, got %v", epsilon)
	}

	successRate, ok := metrics["success_rate"].(float64)
	if !ok {
		t.Fatalf("success_rate has unexpected type %T", metrics["success_rate"])
	}
	if math.IsNaN(successRate) || math.IsInf(successRate, 0) || successRate < 0 || successRate > 1 {
		t.Fatalf("success_rate metric should be finite and bounded, got %v", successRate)
	}

	steps, ok := metrics["steps"].(int)
	if !ok {
		t.Fatalf("steps has unexpected type %T", metrics["steps"])
	}
	if steps < 0 {
		t.Fatalf("steps metric should be non-negative, got %d", steps)
	}
}

func TestEvaluationThroughputIgnoresInvalidInputs(t *testing.T) {
	tests := []struct {
		name         string
		taskSize     int
		expectedTime time.Duration
	}{
		{name: "zero_task_size", taskSize: 0, expectedTime: time.Second},
		{name: "negative_task_size", taskSize: -1, expectedTime: time.Second},
		{name: "zero_expected_time", taskSize: 100, expectedTime: 0},
		{name: "negative_expected_time", taskSize: 100, expectedTime: -time.Second},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			throughput := evaluationThroughput(tt.taskSize, tt.expectedTime)
			if throughput != 0 {
				t.Fatalf("invalid evaluation throughput input should return 0, got %v", throughput)
			}
			if math.IsNaN(throughput) || math.IsInf(throughput, 0) {
				t.Fatalf("invalid evaluation throughput input should remain finite, got %v", throughput)
			}
		})
	}
}

func TestEvaluationResultsCalculateStatsUsesStandardDeviation(t *testing.T) {
	results := &EvaluationResults{
		Episodes:    2,
		Rewards:     []float64{2, 4},
		Throughputs: []float64{10, 14},
		Latencies:   []float64{1, 5},
	}

	results.calculateStats()

	if results.MeanReward != 3 {
		t.Fatalf("expected mean reward 3, got %v", results.MeanReward)
	}
	if results.StdReward != 1 {
		t.Fatalf("expected reward stddev 1, got %v", results.StdReward)
	}
	if results.StdThroughput != 2 {
		t.Fatalf("expected throughput stddev 2, got %v", results.StdThroughput)
	}
	if results.StdLatency != 2 {
		t.Fatalf("expected latency stddev 2, got %v", results.StdLatency)
	}
	if results.SuccessRate != 1 {
		t.Fatalf("expected success rate 1, got %v", results.SuccessRate)
	}
}

func TestEvaluationResultsCalculateStatsHandlesEmptySamples(t *testing.T) {
	results := &EvaluationResults{}

	results.calculateStats()

	stats := []struct {
		name  string
		value float64
	}{
		{name: "mean_reward", value: results.MeanReward},
		{name: "std_reward", value: results.StdReward},
		{name: "mean_throughput", value: results.MeanThroughput},
		{name: "std_throughput", value: results.StdThroughput},
		{name: "mean_latency", value: results.MeanLatency},
		{name: "std_latency", value: results.StdLatency},
		{name: "success_rate", value: results.SuccessRate},
	}
	for _, stat := range stats {
		if math.IsNaN(stat.value) || math.IsInf(stat.value, 0) {
			t.Fatalf("%s should remain finite for empty samples, got %v", stat.name, stat.value)
		}
		if stat.value != 0 {
			t.Fatalf("%s should default to 0 for empty samples, got %v", stat.name, stat.value)
		}
	}
}

func TestEvaluationResultsCalculateStatsIgnoresNonFiniteSamples(t *testing.T) {
	results := &EvaluationResults{
		Rewards:     []float64{2, math.NaN(), math.Inf(1), 4},
		Throughputs: []float64{10, math.Inf(-1), 14},
		Latencies:   []float64{1, math.NaN(), 5},
	}

	results.calculateStats()

	stats := []struct {
		name  string
		value float64
	}{
		{name: "mean_reward", value: results.MeanReward},
		{name: "std_reward", value: results.StdReward},
		{name: "mean_throughput", value: results.MeanThroughput},
		{name: "std_throughput", value: results.StdThroughput},
		{name: "mean_latency", value: results.MeanLatency},
		{name: "std_latency", value: results.StdLatency},
		{name: "success_rate", value: results.SuccessRate},
	}
	for _, stat := range stats {
		if math.IsNaN(stat.value) || math.IsInf(stat.value, 0) {
			t.Fatalf("%s should remain finite with non-finite samples, got %v", stat.name, stat.value)
		}
	}
	if results.MeanReward != 3 || results.StdReward != 1 {
		t.Fatalf("expected reward stats from finite samples only, mean=%v std=%v", results.MeanReward, results.StdReward)
	}
	if results.MeanThroughput != 12 || results.StdThroughput != 2 {
		t.Fatalf("expected throughput stats from finite samples only, mean=%v std=%v", results.MeanThroughput, results.StdThroughput)
	}
	if results.MeanLatency != 3 || results.StdLatency != 2 {
		t.Fatalf("expected latency stats from finite samples only, mean=%v std=%v", results.MeanLatency, results.StdLatency)
	}
	if results.SuccessRate != 1 {
		t.Fatalf("expected success rate from finite rewards only, got %v", results.SuccessRate)
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

func TestActionDecodingIgnoresInvalidInput(t *testing.T) {
	agent, err := NewDQNAgent("nonexistent.onnx")
	if err != nil {
		t.Skipf("ONNX Runtime not available, skipping: %v", err)
		return
	}
	defer agent.Destroy()

	tests := []struct {
		name        string
		action      Action
		state       *EnvironmentState
		decision    *TaskPartitionDecision
		wantStreams int
	}{
		{name: "nil_state", action: ActionStream1, state: nil, decision: &TaskPartitionDecision{}},
		{name: "nil_decision", action: ActionStream1, state: NewEnvironmentState(), decision: nil, wantStreams: 1},
		{name: "negative_action", action: Action(-1), state: NewEnvironmentState(), decision: &TaskPartitionDecision{}},
		{name: "too_large_action", action: Action(NumActions), state: NewEnvironmentState(), decision: &TaskPartitionDecision{}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if recovered := recover(); recovered != nil {
					t.Fatalf("invalid decode input should not panic: %v", recovered)
				}
			}()

			result := agent.decodeAction(tt.action, tt.state, tt.decision)
			if result == nil {
				t.Fatal("invalid decode input should return a decision object")
			}
			if len(result.StreamIDs) != tt.wantStreams || len(result.ChunkSizes) != tt.wantStreams {
				t.Fatalf("decode result stream count mismatch: got streams=%v chunks=%v want=%d", result.StreamIDs, result.ChunkSizes, tt.wantStreams)
			}
			if tt.wantStreams == 0 && result.ExpectedTime != 0 {
				t.Fatalf("invalid decode input should leave expected time empty, got %s", result.ExpectedTime)
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

func TestChunkSizeCalculationIgnoresInvalidInput(t *testing.T) {
	agent, err := NewDQNAgent("nonexistent.onnx")
	if err != nil {
		t.Skipf("ONNX Runtime not available, skipping: %v", err)
		return
	}
	defer agent.Destroy()

	tests := []struct {
		name       string
		taskSize   int
		numStreams int
		streams    []int
		state      *EnvironmentState
	}{
		{name: "nil_state", taskSize: 100, numStreams: 1, streams: []int{0}, state: nil},
		{name: "negative_stream", taskSize: 100, numStreams: 1, streams: []int{-1}, state: NewEnvironmentState()},
		{name: "too_large_stream", taskSize: 100, numStreams: 1, streams: []int{4}, state: NewEnvironmentState()},
		{name: "mismatched_stream_count", taskSize: 100, numStreams: 2, streams: []int{0}, state: NewEnvironmentState()},
		{name: "zero_task_size", taskSize: 0, numStreams: 1, streams: []int{0}, state: NewEnvironmentState()},
		{name: "negative_task_size", taskSize: -100, numStreams: 1, streams: []int{0}, state: NewEnvironmentState()},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if recovered := recover(); recovered != nil {
					t.Fatalf("invalid chunk-size input should not panic: %v", recovered)
				}
			}()

			chunks := agent.calculateChunkSizes(tt.taskSize, tt.numStreams, tt.streams, tt.state)
			if len(chunks) != 0 {
				t.Fatalf("invalid chunk-size input should return no chunks, got %v", chunks)
			}
		})
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

func TestTimeEstimationIgnoresInvalidInput(t *testing.T) {
	agent, err := NewDQNAgent("nonexistent.onnx")
	if err != nil {
		t.Skipf("ONNX Runtime not available, skipping: %v", err)
		return
	}
	defer agent.Destroy()

	tests := []struct {
		name    string
		streams []int
		state   *EnvironmentState
	}{
		{name: "nil_state", streams: []int{0}, state: nil},
		{name: "empty_streams", streams: nil, state: NewEnvironmentState()},
		{name: "negative_stream", streams: []int{-1}, state: NewEnvironmentState()},
		{name: "too_large_stream", streams: []int{4}, state: NewEnvironmentState()},
		{name: "zero_task_size", streams: []int{0}, state: NewEnvironmentState()},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if recovered := recover(); recovered != nil {
					t.Fatalf("invalid time-estimation input should not panic: %v", recovered)
				}
			}()

			taskSize := 100
			if tt.name == "zero_task_size" {
				taskSize = 0
			}
			estimated := agent.estimateTime(taskSize, tt.streams, tt.state)
			if estimated != 0 {
				t.Fatalf("invalid time-estimation input should return zero duration, got %s", estimated)
			}
		})
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

func TestDQNAgentRememberConcurrentMetricsAccess(t *testing.T) {
	agent, err := NewDQNAgent("")
	if err != nil {
		t.Fatalf("NewDQNAgent failed: %v", err)
	}
	defer agent.Destroy()

	state := NewEnvironmentState()
	nextState := NewEnvironmentState()

	var wg sync.WaitGroup
	for i := 0; i < 16; i++ {
		wg.Add(2)
		go func() {
			defer wg.Done()
			for j := 0; j < 250; j++ {
				agent.Remember(state, ActionStream1, 1, nextState, j%10 == 0)
			}
		}()
		go func() {
			defer wg.Done()
			for j := 0; j < 250; j++ {
				_ = agent.GetMetrics()
			}
		}()
	}

	wg.Wait()
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
