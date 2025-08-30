package autoscaling

import (
	"testing"
	"time"

	"github.com/sirupsen/logrus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/khryptorgraphics/novacron/backend/core/orchestration/events"
)

func TestDefaultAutoScaler(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.ErrorLevel) // Reduce noise in tests
	
	eventBus := events.NewNATSEventBus(logger)
	autoScaler := NewDefaultAutoScaler(logger, eventBus)

	t.Run("StartAndStopMonitoring", func(t *testing.T) {
		err := autoScaler.StartMonitoring()
		require.NoError(t, err)
		
		// Try to start again - should fail
		err = autoScaler.StartMonitoring()
		assert.Error(t, err)
		
		err = autoScaler.StopMonitoring()
		require.NoError(t, err)
		
		// Try to stop again - should fail
		err = autoScaler.StopMonitoring()
		assert.Error(t, err)
	})

	t.Run("AddRemoveTargets", func(t *testing.T) {
		target := &AutoScalerTarget{
			ID:      "test-target",
			Type:    "vm",
			Enabled: true,
			Thresholds: &ScalingThresholds{
				CPUScaleUpThreshold:      0.8,
				CPUScaleDownThreshold:    0.2,
				MemoryScaleUpThreshold:   0.8,
				MemoryScaleDownThreshold: 0.2,
				MinReplicas:              1,
				MaxReplicas:              10,
				CooldownPeriod:           5 * time.Minute,
				PredictionWeight:         0.3,
			},
		}

		err := autoScaler.AddTarget(target)
		require.NoError(t, err)

		targets := autoScaler.GetTargets()
		assert.Len(t, targets, 1)
		assert.Equal(t, "test-target", targets["test-target"].ID)

		err = autoScaler.RemoveTarget("test-target")
		require.NoError(t, err)

		targets = autoScaler.GetTargets()
		assert.Len(t, targets, 0)
	})

	t.Run("GetScalingDecisionWithoutTarget", func(t *testing.T) {
		_, err := autoScaler.GetScalingDecision("non-existent")
		assert.Error(t, err)
	})

	t.Run("GetScalingDecisionWithDisabledTarget", func(t *testing.T) {
		target := &AutoScalerTarget{
			ID:      "disabled-target",
			Type:    "vm",
			Enabled: false,
		}

		err := autoScaler.AddTarget(target)
		require.NoError(t, err)

		decision, err := autoScaler.GetScalingDecision("disabled-target")
		require.NoError(t, err)
		assert.Equal(t, ScalingActionNoAction, decision.Action)
		assert.Contains(t, decision.Reason, "disabled")

		autoScaler.RemoveTarget("disabled-target")
	})

	t.Run("GetStatus", func(t *testing.T) {
		status := autoScaler.GetStatus()
		assert.NotNil(t, status)
		assert.False(t, status.Running)
		assert.Equal(t, 0, status.TargetsCount)
	})
}

func TestMetricsCollector(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.ErrorLevel)
	
	collector := NewDefaultMetricsCollector(logger)

	t.Run("CollectMetrics", func(t *testing.T) {
		metrics, err := collector.CollectMetrics()
		require.NoError(t, err)
		assert.NotNil(t, metrics)
		assert.NotEmpty(t, metrics.TargetID)
		assert.True(t, metrics.CPUUsage >= 0 && metrics.CPUUsage <= 1)
		assert.True(t, metrics.MemoryUsage >= 0 && metrics.MemoryUsage <= 1)
	})

	t.Run("GetHistoricalMetrics", func(t *testing.T) {
		// Collect some metrics first
		collector.CollectMetrics()
		time.Sleep(10 * time.Millisecond)
		collector.CollectMetrics()

		end := time.Now()
		start := end.Add(-1 * time.Hour)
		
		metrics, err := collector.GetHistoricalMetrics(start, end)
		require.NoError(t, err)
		assert.True(t, len(metrics) >= 1)
	})

	t.Run("Subscribe", func(t *testing.T) {
		called := false
		handler := MetricsHandlerFunc(func(metrics *MetricsData) error {
			called = true
			assert.NotNil(t, metrics)
			return nil
		})

		err := collector.Subscribe(handler)
		require.NoError(t, err)

		// Collect metrics to trigger handler
		collector.CollectMetrics()
		
		// Give handler time to be called
		time.Sleep(10 * time.Millisecond)
		assert.True(t, called)
	})

	t.Run("StartStopCollection", func(t *testing.T) {
		err := collector.StartCollection()
		require.NoError(t, err)

		// Give collection time to run
		time.Sleep(50 * time.Millisecond)

		err = collector.StopCollection()
		require.NoError(t, err)
	})
}

func TestScalingDecisionEngine(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.ErrorLevel)
	
	engine := NewDefaultScalingDecisionEngine(logger)

	t.Run("SetGetThresholds", func(t *testing.T) {
		thresholds := &ScalingThresholds{
			CPUScaleUpThreshold:      0.8,
			CPUScaleDownThreshold:    0.2,
			MemoryScaleUpThreshold:   0.8,
			MemoryScaleDownThreshold: 0.2,
			MinReplicas:              1,
			MaxReplicas:              5,
			CooldownPeriod:           2 * time.Minute,
			PredictionWeight:         0.3,
		}

		err := engine.SetThresholds(thresholds)
		require.NoError(t, err)

		retrievedThresholds := engine.GetThresholds()
		assert.Equal(t, thresholds.CPUScaleUpThreshold, retrievedThresholds.CPUScaleUpThreshold)
		assert.Equal(t, thresholds.MaxReplicas, retrievedThresholds.MaxReplicas)
	})

	t.Run("MakeDecisionScaleUp", func(t *testing.T) {
		prediction := &ResourcePrediction{
			TargetID:         "test-target",
			PredictedCPU:     0.9,
			PredictedMemory:  0.8,
			Confidence:       0.8,
			TrendDirection:   TrendIncreasing,
		}

		current := &MetricsData{
			TargetID:    "test-target",
			CPUUsage:    0.85,
			MemoryUsage: 0.75,
			ActiveVMs:   3,
		}

		decision, err := engine.MakeDecision(prediction, current)
		require.NoError(t, err)
		assert.Equal(t, "test-target", decision.TargetID)
		assert.Equal(t, ScalingActionScaleUp, decision.Action)
		assert.True(t, decision.TargetScale > decision.CurrentScale)
	})

	t.Run("MakeDecisionScaleDown", func(t *testing.T) {
		// Create a fresh engine to avoid state interference
		freshEngine := NewDefaultScalingDecisionEngine(logger)
		
		// Set thresholds that will allow scale down
		thresholds := &ScalingThresholds{
			CPUScaleUpThreshold:      0.8,
			CPUScaleDownThreshold:    0.2,  // Set higher than test data
			MemoryScaleUpThreshold:   0.8,
			MemoryScaleDownThreshold: 0.2,  // Set higher than test data
			MinReplicas:              1,
			MaxReplicas:              10,
			CooldownPeriod:           2 * time.Minute,
			PredictionWeight:         0.3,
		}
		err := freshEngine.SetThresholds(thresholds)
		require.NoError(t, err)
		
		prediction := &ResourcePrediction{
			TargetID:         "test-target",
			PredictedCPU:     0.1,
			PredictedMemory:  0.1,
			Confidence:       0.8,
			TrendDirection:   TrendDecreasing,
		}

		current := &MetricsData{
			TargetID:    "test-target",
			CPUUsage:    0.1,  // Below both CPU and Memory thresholds
			MemoryUsage: 0.1,  // Below both thresholds
			ActiveVMs:   5,
		}

		decision, err := freshEngine.MakeDecision(prediction, current)
		require.NoError(t, err)
		assert.Equal(t, ScalingActionScaleDown, decision.Action)
		assert.True(t, decision.TargetScale < decision.CurrentScale)
	})

	t.Run("MakeDecisionNoAction", func(t *testing.T) {
		prediction := &ResourcePrediction{
			TargetID:         "test-target",
			PredictedCPU:     0.5,
			PredictedMemory:  0.4,
			Confidence:       0.8,
			TrendDirection:   TrendStable,
		}

		current := &MetricsData{
			TargetID:    "test-target",
			CPUUsage:    0.45,
			MemoryUsage: 0.5,
			ActiveVMs:   3,
		}

		decision, err := engine.MakeDecision(prediction, current)
		require.NoError(t, err)
		assert.Equal(t, ScalingActionNoAction, decision.Action)
		assert.Equal(t, decision.CurrentScale, decision.TargetScale)
	})

	t.Run("CooldownPeriod", func(t *testing.T) {
		// Set short cooldown for testing
		thresholds := engine.GetThresholds()
		thresholds.CooldownPeriod = 100 * time.Millisecond
		engine.SetThresholds(thresholds)

		prediction := &ResourcePrediction{
			TargetID:         "cooldown-target",
			PredictedCPU:     0.9,
			PredictedMemory:  0.8,
			Confidence:       0.8,
			TrendDirection:   TrendIncreasing,
		}

		current := &MetricsData{
			TargetID:    "cooldown-target",
			CPUUsage:    0.85,
			MemoryUsage: 0.75,
			ActiveVMs:   3,
		}

		// First decision should trigger scaling
		decision1, err := engine.MakeDecision(prediction, current)
		require.NoError(t, err)
		assert.Equal(t, ScalingActionScaleUp, decision1.Action)

		// Second decision immediately after should be in cooldown
		decision2, err := engine.MakeDecision(prediction, current)
		require.NoError(t, err)
		assert.Equal(t, ScalingActionNoAction, decision2.Action)
		assert.Contains(t, decision2.Reason, "cooldown")

		// Wait for cooldown to expire
		time.Sleep(150 * time.Millisecond)

		// Third decision should work again
		decision3, err := engine.MakeDecision(prediction, current)
		require.NoError(t, err)
		assert.Equal(t, ScalingActionScaleUp, decision3.Action)
	})

	t.Run("GetScaleState", func(t *testing.T) {
		// Trigger a scaling decision first
		prediction := &ResourcePrediction{
			TargetID:         "state-target",
			PredictedCPU:     0.9,
			Confidence:       0.8,
		}

		current := &MetricsData{
			TargetID:    "state-target",
			CPUUsage:    0.85,
			ActiveVMs:   2,
		}

		engine.MakeDecision(prediction, current)

		state, exists := engine.GetScaleState("state-target")
		assert.True(t, exists)
		assert.NotNil(t, state)
		assert.True(t, len(state.ScaleHistory) > 0)
	})
}

func TestARIMAPredictor(t *testing.T) {
	predictor := NewARIMAPredictor(ARIMAOrder{P: 2, D: 1, Q: 1})

	// Generate test data
	testData := generateTestMetrics(20)

	t.Run("Train", func(t *testing.T) {
		err := predictor.Train(testData)
		require.NoError(t, err)
		assert.True(t, predictor.GetAccuracy() >= 0)
	})

	t.Run("TrainInsufficientData", func(t *testing.T) {
		shortData := generateTestMetrics(5)
		err := predictor.Train(shortData)
		assert.Error(t, err)
	})

	t.Run("Predict", func(t *testing.T) {
		predictor.Train(testData)
		
		current := testData[len(testData)-1]
		prediction, err := predictor.Predict(current, 30)
		require.NoError(t, err)
		
		assert.Equal(t, current.TargetID, prediction.TargetID)
		assert.Equal(t, 30, prediction.HorizonMinutes)
		assert.True(t, prediction.PredictedCPU >= 0)
		assert.True(t, prediction.Confidence >= 0)
	})

	t.Run("PredictWithoutTraining", func(t *testing.T) {
		freshPredictor := NewARIMAPredictor(ARIMAOrder{P: 1, D: 1, Q: 1})
		current := testData[0]
		
		_, err := freshPredictor.Predict(current, 30)
		assert.Error(t, err)
	})

	t.Run("GetModelInfo", func(t *testing.T) {
		predictor.Train(testData)
		info := predictor.GetModelInfo()
		
		assert.Equal(t, "ARIMA", info.ModelType)
		assert.NotZero(t, info.DataPoints)
		assert.True(t, info.Accuracy >= 0 && info.Accuracy <= 1)
	})
}

func TestNeuralNetworkPredictor(t *testing.T) {
	predictor := NewNeuralNetworkPredictor(5)

	// Generate test data
	testData := generateTestMetrics(30)

	t.Run("Train", func(t *testing.T) {
		err := predictor.Train(testData)
		require.NoError(t, err)
		assert.True(t, predictor.GetAccuracy() >= 0)
	})

	t.Run("TrainInsufficientData", func(t *testing.T) {
		shortData := generateTestMetrics(10)
		err := predictor.Train(shortData)
		assert.Error(t, err)
	})

	t.Run("Predict", func(t *testing.T) {
		predictor.Train(testData)
		
		current := testData[len(testData)-1]
		prediction, err := predictor.Predict(current, 30)
		require.NoError(t, err)
		
		assert.Equal(t, current.TargetID, prediction.TargetID)
		assert.Equal(t, 30, prediction.HorizonMinutes)
		assert.True(t, prediction.PredictedCPU >= 0)
		assert.True(t, prediction.Confidence >= 0)
	})

	t.Run("GetModelInfo", func(t *testing.T) {
		predictor.Train(testData)
		info := predictor.GetModelInfo()
		
		assert.Equal(t, "NeuralNetwork", info.ModelType)
		assert.NotZero(t, info.DataPoints)
		assert.True(t, info.Accuracy >= 0 && info.Accuracy <= 1)
	})
}

// Helper functions

func generateTestMetrics(count int) []*MetricsData {
	metrics := make([]*MetricsData, count)
	baseTime := time.Now().Add(-time.Duration(count) * time.Minute)
	
	for i := 0; i < count; i++ {
		// Generate somewhat realistic CPU usage pattern
		cpuUsage := 0.3 + 0.4*float64(i%10)/10.0 // Oscillating pattern
		if i > count/2 {
			cpuUsage += 0.2 // Trend upward
		}
		
		metrics[i] = &MetricsData{
			Timestamp:    baseTime.Add(time.Duration(i) * time.Minute),
			TargetID:     "test-target",
			TargetType:   "vm",
			CPUUsage:     cpuUsage,
			MemoryUsage:  cpuUsage * 0.8, // Memory correlates with CPU
			NetworkIO:    10.0 + float64(i%5),
			DiskIO:       100.0 + float64(i%10)*10,
			ActiveVMs:    3 + i%3,
		}
	}
	
	return metrics
}