package scaling

import (
	"testing"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/zeroops"
)

func TestAutonomousScaler(t *testing.T) {
	config := zeroops.DefaultZeroOpsConfig()
	scaler := NewAutonomousScaler(config)

	err := scaler.Start()
	if err != nil {
		t.Fatalf("Failed to start scaler: %v", err)
	}

	time.Sleep(100 * time.Millisecond)

	err = scaler.Stop()
	if err != nil {
		t.Fatalf("Failed to stop scaler: %v", err)
	}
}

func TestPredictiveScaling(t *testing.T) {
	config := zeroops.DefaultZeroOpsConfig()
	scaler := NewAutonomousScaler(config)

	prediction := scaler.predictor.PredictWorkload(15 * time.Minute)

	if prediction.Accuracy < config.ScalingConfig.MinPredictionAccuracy {
		t.Errorf("Prediction accuracy %.2f below threshold %.2f",
			prediction.Accuracy, config.ScalingConfig.MinPredictionAccuracy)
	}
}

func TestMultiDimensionalScaling(t *testing.T) {
	config := zeroops.DefaultZeroOpsConfig()
	scaler := NewAutonomousScaler(config)

	prediction := &WorkloadPrediction{
		Timestamp: time.Now(),
		Duration:  15 * time.Minute,
		Accuracy:  0.95,
		Metrics: map[string]float64{
			"cpu":     0.85,
			"memory":  0.70,
			"network": 0.60,
			"storage": 0.55,
			"gpu":     0.80,
		},
	}

	dimensions := scaler.analyzeDimensions(prediction)

	if !dimensions.RequiresScaleUp() {
		t.Error("Expected scale up required with high utilization")
	}

	maxUtil := dimensions.MaxUtilization()
	if maxUtil != 0.85 {
		t.Errorf("Expected max utilization 0.85, got %.2f", maxUtil)
	}
}

func TestScaleToZero(t *testing.T) {
	config := zeroops.DefaultZeroOpsConfig()
	scaler := NewAutonomousScaler(config)

	workload := &Workload{
		ID:   "test-workload",
		Type: "batch",
	}

	result := scaler.scaleEngine.ScaleToZero(workload)
	if !result.Success {
		t.Error("Failed to scale to zero")
	}
}

func TestScaleFromZero(t *testing.T) {
	config := zeroops.DefaultZeroOpsConfig()
	scaler := NewAutonomousScaler(config)

	workload := &Workload{
		ID:   "test-workload",
		Type: "api",
	}

	start := time.Now()
	result := scaler.scaleEngine.ScaleFromZero(workload)
	duration := time.Since(start)

	maxTime := time.Duration(config.ScalingConfig.ScaleFromZeroMaxSeconds) * time.Second
	if duration > maxTime {
		t.Errorf("Scale from zero took %v, exceeds target %v", duration, maxTime)
	}

	if !result.Success {
		t.Error("Failed to scale from zero")
	}
}

func TestScalingDecision(t *testing.T) {
	config := zeroops.DefaultZeroOpsConfig()
	scaler := NewAutonomousScaler(config)

	prediction := &WorkloadPrediction{
		Timestamp: time.Now(),
		Duration:  15 * time.Minute,
		Accuracy:  0.92,
		Metrics: map[string]float64{
			"cpu":     0.75,
			"memory":  0.65,
			"network": 0.60,
		},
	}

	decision := scaler.makeScalingDecision(prediction)

	if decision.ShouldScale && decision.Direction == "up" {
		if decision.OptimizedAmount <= 0 {
			t.Error("Expected positive scale amount")
		}
	}
}

func TestCostPerformanceOptimization(t *testing.T) {
	config := zeroops.DefaultZeroOpsConfig()
	scaler := NewAutonomousScaler(config)

	// Test cost-performance tradeoff
	costWeight := config.ScalingConfig.CostOptimizationWeight
	perfWeight := config.ScalingConfig.PerformanceWeight

	if costWeight+perfWeight != 1.0 {
		t.Errorf("Weights should sum to 1.0, got %.2f", costWeight+perfWeight)
	}
}
