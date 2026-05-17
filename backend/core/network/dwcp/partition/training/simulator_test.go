package main

import (
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

func newTrainingTestAgent(t *testing.T) *partition.DQNAgent {
	t.Helper()

	agent, err := partition.NewDQNAgent("")
	if err != nil {
		t.Fatalf("NewDQNAgent failed: %v", err)
	}
	return agent
}
