package nas

import (
	"context"
	"testing"
	"time"
)

func TestNASEngine(t *testing.T) {
	tests := []struct {
		name       string
		config     *NASConfig
		wantError  bool
	}{
		{
			name: "random_search",
			config: &NASConfig{
				SearchAlgorithm: "random",
				MaxTrials:       10,
				ParallelTrials:  2,
				TargetMetric:    "accuracy",
				MetricGoal:      "maximize",
				LatencyBudget:   10.0,
			},
			wantError: false,
		},
		{
			name: "bayesian_search",
			config: &NASConfig{
				SearchAlgorithm: "bayesian",
				MaxTrials:       15,
				ParallelTrials:  3,
				TargetMetric:    "accuracy",
				MetricGoal:      "maximize",
			},
			wantError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			engine := NewNASEngine(tt.config)

			// Create dummy dataset
			trainData := Dataset{
				X: make([][]float64, 100),
				Y: make([]float64, 100),
			}
			for i := range trainData.X {
				trainData.X[i] = make([]float64, 10)
			}

			valData := Dataset{
				X: make([][]float64, 20),
				Y: make([]float64, 20),
			}
			for i := range valData.X {
				valData.X[i] = make([]float64, 10)
			}

			ctx, cancel := context.WithTimeout(context.Background(), 1*time.Minute)
			defer cancel()

			bestArch, err := engine.Search(ctx, trainData, valData)

			if tt.wantError && err == nil {
				t.Error("expected error but got none")
			}

			if !tt.wantError && err != nil {
				t.Errorf("unexpected error: %v", err)
			}

			if bestArch != nil {
				if bestArch.Metrics["accuracy"] < 0.5 {
					t.Errorf("accuracy too low: %f", bestArch.Metrics["accuracy"])
				}

				if bestArch.Metrics["latency"] > tt.config.LatencyBudget {
					t.Errorf("latency exceeds budget: %f > %f",
						bestArch.Metrics["latency"], tt.config.LatencyBudget)
				}

				t.Logf("Best architecture: %d layers, Accuracy: %.4f, Latency: %.2fms",
					len(bestArch.Layers), bestArch.Metrics["accuracy"], bestArch.Metrics["latency"])
			}
		})
	}
}

func TestSearchController(t *testing.T) {
	space := DefaultSearchSpace()
	controller := NewSearchController("random")

	// Test random search
	candidate := controller.randomSearch(space)
	if len(candidate.Layers) == 0 {
		t.Error("no layers generated")
	}

	// Test with history
	history := []CandidateArchitecture{
		{
			Layers:   []Layer{{Type: "conv"}},
			Status:   "completed",
			Metrics:  map[string]float64{"accuracy": 0.85},
		},
	}

	candidate = controller.GenerateCandidate(space, history)
	if len(candidate.Layers) == 0 {
		t.Error("no layers generated with history")
	}

	t.Logf("Generated architecture with %d layers", len(candidate.Layers))
}

func BenchmarkNAS(b *testing.B) {
	config := &NASConfig{
		SearchAlgorithm: "random",
		MaxTrials:       5,
		ParallelTrials:  2,
	}

	engine := NewNASEngine(config)

	trainData := Dataset{X: make([][]float64, 50), Y: make([]float64, 50)}
	for i := range trainData.X {
		trainData.X[i] = make([]float64, 10)
	}

	valData := Dataset{X: make([][]float64, 10), Y: make([]float64, 10)}
	for i := range valData.X {
		valData.X[i] = make([]float64, 10)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ctx := context.Background()
		_, _ = engine.Search(ctx, trainData, valData)
	}
}
