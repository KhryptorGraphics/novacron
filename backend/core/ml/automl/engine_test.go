package automl

import (
	"context"
	"testing"
	"time"
)

func TestAutoMLEngine(t *testing.T) {
	tests := []struct {
		name          string
		config        *AutoMLConfig
		dataSize      int
		expectedError bool
	}{
		{
			name: "successful_automl_training",
			config: &AutoMLConfig{
				MaxTrials:          10,
				TimeoutPerTrial:    30 * time.Second,
				TargetMetric:       "accuracy",
				MetricGoal:         "maximize",
				ValidationSplit:    0.2,
				CVFolds:            3,
				EarlyStoppingRounds: 5,
				ParallelTrials:     2,
				AutoFeatureEng:     true,
				ModelTypes:         []string{"random_forest", "xgboost"},
			},
			dataSize:      100,
			expectedError: false,
		},
		{
			name: "quick_convergence",
			config: &AutoMLConfig{
				MaxTrials:          5,
				TimeoutPerTrial:    10 * time.Second,
				TargetMetric:       "accuracy",
				MetricGoal:         "maximize",
				ParallelTrials:     1,
			},
			dataSize:      50,
			expectedError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			engine := NewAutoMLEngine(tt.config)

			// Generate synthetic data
			X, y := generateSyntheticData(tt.dataSize, 5)
			featureNames := []string{"f1", "f2", "f3", "f4", "f5"}

			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
			defer cancel()

			bestModel, err := engine.Fit(ctx, X, y, featureNames)

			if tt.expectedError && err == nil {
				t.Errorf("expected error but got none")
			}

			if !tt.expectedError && err != nil {
				t.Errorf("unexpected error: %v", err)
			}

			if bestModel != nil {
				if bestModel.Metrics["accuracy"] < 0.5 {
					t.Errorf("accuracy too low: %f", bestModel.Metrics["accuracy"])
				}

				t.Logf("Best model: %s, Accuracy: %.4f, Train time: %v",
					bestModel.ModelType, bestModel.Metrics["accuracy"], bestModel.TrainTime)
			}

			// Verify experiments were run
			experiments := engine.GetExperiments()
			if len(experiments) == 0 {
				t.Error("no experiments were run")
			}

			completedCount := 0
			for _, exp := range experiments {
				if exp.Status == "completed" {
					completedCount++
				}
			}

			if completedCount == 0 {
				t.Error("no completed experiments")
			}

			t.Logf("Completed experiments: %d/%d", completedCount, len(experiments))
		})
	}
}

func TestFeatureEngineering(t *testing.T) {
	fe := NewFeatureEngineer()

	X := [][]float64{
		{1.0, 2.0, 3.0},
		{2.0, 3.0, 4.0},
		{3.0, 4.0, 5.0},
	}
	featureNames := []string{"a", "b", "c"}

	// Test polynomial features
	XPoly, namesPoly := fe.addPolynomialFeatures(X, featureNames, 2)
	if len(XPoly[0]) != 6 { // Original 3 + squared 3
		t.Errorf("expected 6 features, got %d", len(XPoly[0]))
	}
	if len(namesPoly) != 6 {
		t.Errorf("expected 6 feature names, got %d", len(namesPoly))
	}

	// Test interaction features
	XInt, namesInt := fe.addInteractionFeatures(X, featureNames)
	if len(XInt[0]) <= len(X[0]) {
		t.Error("interaction features not added")
	}

	// Test normalization
	XNorm := fe.normalizeFeatures(X)
	for i := range XNorm {
		for j := range XNorm[i] {
			if XNorm[i][j] < 0 || XNorm[i][j] > 1 {
				t.Errorf("normalized value out of range: %f", XNorm[i][j])
			}
		}
	}

	t.Logf("Feature engineering test passed")
}

func TestModelEvaluator(t *testing.T) {
	config := DefaultAutoMLConfig()
	evaluator := NewModelEvaluator(config)

	// Create simple linear model
	model := &LinearModel{
		weights: []float64{0.5, 0.3, 0.2},
		bias:    0.1,
	}

	X := [][]float64{
		{1.0, 2.0, 3.0},
		{2.0, 3.0, 4.0},
		{3.0, 4.0, 5.0},
	}
	y := []float64{2.0, 3.0, 4.0}

	metrics, err := evaluator.Evaluate(model, X, y)
	if err != nil {
		t.Fatalf("evaluation failed: %v", err)
	}

	requiredMetrics := []string{"accuracy", "mse", "rmse", "mae", "r2"}
	for _, metric := range requiredMetrics {
		if _, ok := metrics[metric]; !ok {
			t.Errorf("missing metric: %s", metric)
		}
	}

	t.Logf("Metrics: %+v", metrics)
}

func generateSyntheticData(n, m int) ([][]float64, []float64) {
	X := make([][]float64, n)
	y := make([]float64, n)

	for i := 0; i < n; i++ {
		X[i] = make([]float64, m)
		for j := 0; j < m; j++ {
			X[i][j] = float64(i*j%10) / 10.0
		}
		// y = sum of features with some pattern
		y[i] = 0.0
		for j := 0; j < m; j++ {
			y[i] += X[i][j] * 0.5
		}
		if y[i] > 2.5 {
			y[i] = 1.0
		} else {
			y[i] = 0.0
		}
	}

	return X, y
}

func BenchmarkAutoML(b *testing.B) {
	config := &AutoMLConfig{
		MaxTrials:       5,
		ParallelTrials:  2,
		TargetMetric:    "accuracy",
		MetricGoal:      "maximize",
		AutoFeatureEng:  false,
	}

	engine := NewAutoMLEngine(config)
	X, y := generateSyntheticData(50, 5)
	featureNames := []string{"f1", "f2", "f3", "f4", "f5"}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ctx := context.Background()
		_, _ = engine.Fit(ctx, X, y, featureNames)
	}
}
