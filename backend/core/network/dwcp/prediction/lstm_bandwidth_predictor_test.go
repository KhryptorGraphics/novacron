package prediction

import (
	"math"
	"testing"
	"time"
)

func TestLSTMPredictorFallsBackToHistoryForecast(t *testing.T) {
	predictor, err := NewLSTMPredictor("")
	if err != nil {
		t.Fatalf("NewLSTMPredictor() error = %v", err)
	}

	history := make([]NetworkSample, 10)
	for i := range history {
		history[i] = NetworkSample{
			Timestamp:     time.Now().Add(time.Duration(i) * time.Second),
			BandwidthMbps: 100 + float64(i*10),
			LatencyMs:     20 + float64(i),
			PacketLoss:    0.01,
			JitterMs:      2,
			TimeOfDay:     12,
			DayOfWeek:     2,
		}
	}

	prediction, err := predictor.Predict(history)
	if err != nil {
		t.Fatalf("Predict() error = %v", err)
	}

	assertFloatNear(t, prediction.PredictedBandwidthMbps, 154, 0.001)
	assertFloatNear(t, prediction.PredictedLatencyMs, 25.4, 0.001)
	assertFloatNear(t, prediction.PredictedPacketLoss, 0.01, 0.001)
	assertFloatNear(t, prediction.PredictedJitterMs, 2, 0.001)
	if prediction.Confidence > 0.5 {
		t.Fatalf("Confidence = %f, want no-model confidence capped at 0.5", prediction.Confidence)
	}
}

func assertFloatNear(t *testing.T, got, want, tolerance float64) {
	t.Helper()
	if math.Abs(got-want) > tolerance {
		t.Fatalf("got %f, want %f +/- %f", got, want, tolerance)
	}
}
