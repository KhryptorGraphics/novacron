package prediction

import (
	"math"
	"testing"
	"time"

	baseprediction "github.com/khryptorgraphics/novacron/backend/core/network/dwcp/prediction"
)

func TestLSTMPredictorV3FallsBackToHistoryForecast(t *testing.T) {
	predictor, err := NewLSTMPredictorV3("")
	if err != nil {
		t.Fatalf("NewLSTMPredictorV3() error = %v", err)
	}

	history := make([]baseprediction.NetworkSample, 60)
	for i := range history {
		history[i] = baseprediction.NetworkSample{
			Timestamp:     time.Now().Add(time.Duration(i) * time.Second),
			BandwidthMbps: 400 + float64(i*10),
			LatencyMs:     80 + float64(i),
			PacketLoss:    0.02,
			JitterMs:      8,
			TimeOfDay:     13,
			DayOfWeek:     3,
		}
	}

	prediction, err := predictor.Predict(history)
	if err != nil {
		t.Fatalf("Predict() error = %v", err)
	}

	assertFloatNear(t, prediction.PredictedBandwidthMbps, 704.833333, 0.001)
	assertFloatNear(t, prediction.PredictedLatencyMs, 110.483333, 0.001)
	assertFloatNear(t, prediction.PredictedPacketLoss, 0.02, 0.001)
	assertFloatNear(t, prediction.PredictedJitterMs, 8, 0.001)
	if prediction.Confidence != 0.70 {
		t.Fatalf("Confidence = %f, want 0.70 base confidence", prediction.Confidence)
	}
}

func assertFloatNear(t *testing.T, got, want, tolerance float64) {
	t.Helper()
	if math.Abs(got-want) > tolerance {
		t.Fatalf("got %f, want %f +/- %f", got, want, tolerance)
	}
}
