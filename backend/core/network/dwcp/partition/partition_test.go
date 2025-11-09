package partition

import (
	"math"
	"testing"
)

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
		action         Action
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