package hpo

import (
	"context"
	"math"
	"testing"
	"time"
)

func TestHPOOptimizer(t *testing.T) {
	space := map[string]ParamDef{
		"learning_rate": {
			Type:  "float",
			Min:   0.001,
			Max:   0.1,
			Scale: "log",
		},
		"num_layers": {
			Type: "int",
			Min:  2,
			Max:  10,
		},
		"activation": {
			Type:   "categorical",
			Values: []interface{}{"relu", "tanh", "sigmoid"},
		},
	}

	// Objective function: minimize quadratic
	objective := func(params map[string]interface{}) (map[string]float64, error) {
		lr := params["learning_rate"].(float64)
		numLayers := params["num_layers"].(int)

		// Simulate loss: prefer lr=0.01 and num_layers=5
		loss := math.Pow(lr-0.01, 2) + math.Pow(float64(numLayers-5), 2)

		return map[string]float64{
			"loss":     loss,
			"accuracy": 1.0 / (1.0 + loss),
		}, nil
	}

	tests := []struct {
		name      string
		algorithm string
		maxTrials int
		wantError bool
	}{
		{
			name:      "bayesian_optimization",
			algorithm: "bayesian",
			maxTrials: 20,
			wantError: false,
		},
		{
			name:      "random_search",
			algorithm: "random",
			maxTrials: 15,
			wantError: false,
		},
		{
			name:      "grid_search",
			algorithm: "grid",
			maxTrials: 50,
			wantError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := &HPOConfig{
				Algorithm:           tt.algorithm,
				MaxTrials:           tt.maxTrials,
				ParallelTrials:      2,
				MetricGoal:          "minimize",
				EarlyStoppingRounds: 5,
				TimeoutPerTrial:     5 * time.Second,
			}

			optimizer := NewHPOOptimizer(config, space, objective)

			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()

			bestTrial, err := optimizer.Optimize(ctx, "loss")

			if tt.wantError && err == nil {
				t.Error("expected error but got none")
			}

			if !tt.wantError && err != nil {
				t.Errorf("unexpected error: %v", err)
			}

			if bestTrial != nil {
				t.Logf("Best trial: Loss=%.4f, Params=%+v",
					bestTrial.Metrics["loss"], bestTrial.Params)

				if bestTrial.Metrics["loss"] > 100 {
					t.Error("optimization did not converge")
				}
			}

			trials := optimizer.GetTrials()
			completedCount := 0
			for _, trial := range trials {
				if trial.Status == "completed" {
					completedCount++
				}
			}

			t.Logf("Completed trials: %d/%d", completedCount, len(trials))
		})
	}
}

func TestGaussianProcess(t *testing.T) {
	kernel := RBFKernel(1.0)
	gp := NewGaussianProcess(kernel, 0.01)

	// Fit some data points
	X := [][]float64{
		{0.0}, {1.0}, {2.0}, {3.0},
	}
	y := []float64{0.0, 1.0, 4.0, 9.0} // y = x^2

	for i := range X {
		gp.Fit(X[i], y[i])
	}

	// Predict
	testX := []float64{1.5}
	mean, variance := gp.Predict(testX)

	t.Logf("GP Prediction: mean=%.2f, variance=%.2f", mean, variance)

	if variance < 0 {
		t.Error("variance should be non-negative")
	}
}

func TestHyperband(t *testing.T) {
	space := map[string]ParamDef{
		"learning_rate": {
			Type:  "float",
			Min:   0.001,
			Max:   0.1,
			Scale: "log",
		},
	}

	objective := func(params map[string]interface{}) (map[string]float64, error) {
		lr := params["learning_rate"].(float64)
		epochs := 1
		if e, ok := params["epochs"]; ok {
			epochs = e.(int)
		}

		// Simulate loss decreasing with more epochs
		loss := 1.0 / float64(epochs) * math.Abs(lr-0.01)

		return map[string]float64{"loss": loss}, nil
	}

	config := &HPOConfig{
		Algorithm:      "hyperband",
		MaxTrials:      50,
		ParallelTrials: 1,
		MetricGoal:     "minimize",
	}

	optimizer := NewHPOOptimizer(config, space, objective)

	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Minute)
	defer cancel()

	bestTrial, err := optimizer.Optimize(ctx, "loss")
	if err != nil {
		t.Fatalf("optimization failed: %v", err)
	}

	t.Logf("Hyperband best trial: Loss=%.4f", bestTrial.Metrics["loss"])
}

func BenchmarkBayesianOptimization(b *testing.B) {
	space := map[string]ParamDef{
		"x": {Type: "float", Min: -5, Max: 5},
		"y": {Type: "float", Min: -5, Max: 5},
	}

	objective := func(params map[string]interface{}) (map[string]float64, error) {
		x := params["x"].(float64)
		y := params["y"].(float64)
		loss := x*x + y*y
		return map[string]float64{"loss": loss}, nil
	}

	config := &HPOConfig{
		Algorithm:      "bayesian",
		MaxTrials:      10,
		ParallelTrials: 1,
		MetricGoal:     "minimize",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		optimizer := NewHPOOptimizer(config, space, objective)
		ctx := context.Background()
		_, _ = optimizer.Optimize(ctx, "loss")
	}
}
