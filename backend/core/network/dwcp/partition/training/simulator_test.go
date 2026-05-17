package main

import (
	"math"
	"testing"

	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/partition"
)

func TestSimulateEpisodeRejectsInvalidInputs(t *testing.T) {
	tests := []struct {
		name     string
		agent    *partition.DQNAgent
		maxSteps int
	}{
		{name: "nil_agent", agent: nil, maxSteps: 1},
		{name: "zero_steps", agent: newTrainingTestAgent(t), maxSteps: 0},
		{name: "negative_steps", agent: newTrainingTestAgent(t), maxSteps: -1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if recovered := recover(); recovered != nil {
					t.Fatalf("invalid episode input should return an error without panic: %v", recovered)
				}
			}()
			if tt.agent != nil {
				defer tt.agent.Destroy()
			}

			metrics, err := NewNetworkSimulator().SimulateEpisode(tt.agent, tt.maxSteps)
			if err == nil {
				t.Fatalf("expected invalid input error, got metrics=%#v", metrics)
			}
			if metrics != nil {
				t.Fatalf("invalid input should not return metrics: %#v", metrics)
			}
		})
	}
}

func TestRunTrainingRejectsInvalidInputs(t *testing.T) {
	tests := []struct {
		name     string
		agent    *partition.DQNAgent
		episodes int
		maxSteps int
	}{
		{name: "nil_agent", agent: nil, episodes: 1, maxSteps: 1},
		{name: "zero_episodes", agent: newTrainingTestAgent(t), episodes: 0, maxSteps: 1},
		{name: "negative_episodes", agent: newTrainingTestAgent(t), episodes: -1, maxSteps: 1},
		{name: "zero_steps", agent: newTrainingTestAgent(t), episodes: 1, maxSteps: 0},
		{name: "negative_steps", agent: newTrainingTestAgent(t), episodes: 1, maxSteps: -1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if recovered := recover(); recovered != nil {
					t.Fatalf("invalid training input should return an error without panic: %v", recovered)
				}
			}()
			if tt.agent != nil {
				defer tt.agent.Destroy()
			}

			err := NewNetworkSimulator().RunTraining(tt.agent, tt.episodes, tt.maxSteps)
			if err == nil {
				t.Fatal("expected invalid training input error")
			}
		})
	}
}

func newTrainingTestAgent(t *testing.T) *partition.DQNAgent {
	t.Helper()

	agent, err := partition.NewDQNAgent("")
	if err != nil {
		t.Fatalf("NewDQNAgent failed: %v", err)
	}
	return agent
}

func TestTrainingSimulatorIgnoresMalformedDecisions(t *testing.T) {
	sim := NewNetworkSimulator()
	state := partition.NewEnvironmentState()
	state.TaskSize = 100
	state.StreamBandwidth = [4]float64{100, 100, 100, 100}
	state.StreamLatency = [4]float64{10, 10, 10, 10}
	state.StreamSuccessRate = [4]float64{1, 1, 1, 1}
	invalidTelemetryState := partition.NewEnvironmentState()
	invalidTelemetryState.TaskSize = 100
	invalidTelemetryState.StreamBandwidth = [4]float64{math.NaN(), math.Inf(1), -1, 0}
	invalidTelemetryState.StreamLatency = [4]float64{math.NaN(), math.Inf(1), -1, 0}
	invalidTelemetryState.StreamCongestion = [4]float64{math.NaN(), math.Inf(1), -1, 2}
	invalidTelemetryState.StreamSuccessRate = [4]float64{math.NaN(), math.Inf(1), -1, 2}

	tests := []struct {
		name     string
		state    *partition.EnvironmentState
		decision *partition.TaskPartitionDecision
	}{
		{name: "nil_state", state: nil, decision: &partition.TaskPartitionDecision{StreamIDs: []int{0}, ChunkSizes: []int{100}}},
		{name: "nil_decision", state: state, decision: nil},
		{name: "negative_stream", state: state, decision: &partition.TaskPartitionDecision{StreamIDs: []int{-1}, ChunkSizes: []int{100}}},
		{name: "too_large_stream", state: state, decision: &partition.TaskPartitionDecision{StreamIDs: []int{4}, ChunkSizes: []int{100}}},
		{name: "missing_chunk", state: state, decision: &partition.TaskPartitionDecision{StreamIDs: []int{0}, ChunkSizes: nil}},
		{name: "zero_task_size", state: partition.NewEnvironmentState(), decision: &partition.TaskPartitionDecision{StreamIDs: []int{0}, ChunkSizes: []int{100}}},
		{name: "invalid_telemetry", state: invalidTelemetryState, decision: &partition.TaskPartitionDecision{StreamIDs: []int{0, 1, 2, 3}, ChunkSizes: []int{25, 25, 25, 25}}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if recovered := recover(); recovered != nil {
					t.Fatalf("malformed decision should be ignored without panic: %v", recovered)
				}
			}()

			outcome := sim.calculateOutcome(tt.decision, tt.state)
			if outcome == nil {
				t.Fatal("expected fallback outcome, got nil")
			}
			assertFiniteOutcome(t, outcome)

			nextState := sim.updateState(tt.state, tt.decision, outcome)
			if nextState == nil {
				t.Fatal("expected fallback next state, got nil")
			}
		})
	}
}

func assertFiniteOutcome(t *testing.T, outcome *partition.ActionOutcome) {
	t.Helper()

	values := map[string]float64{
		"actual_throughput":   outcome.ActualThroughput,
		"baseline_throughput": outcome.BaselineThroughput,
		"actual_latency":      outcome.ActualLatency,
		"target_latency":      outcome.TargetLatency,
		"stream_imbalance":    outcome.StreamImbalance,
	}
	for name, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			t.Fatalf("%s should be finite, got %v", name, value)
		}
	}
}
