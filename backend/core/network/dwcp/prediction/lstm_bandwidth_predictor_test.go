package prediction

import (
	"fmt"
	"math"
	"testing"
	"time"
)

type fakeBandwidthInferenceSession struct {
	output    []float32
	err       error
	inputSeen []float32
	destroyed bool
}

func (session *fakeBandwidthInferenceSession) Run(input []float32) ([]float32, error) {
	session.inputSeen = append([]float32(nil), input...)
	if session.err != nil {
		return nil, session.err
	}
	return append([]float32(nil), session.output...), nil
}

func (session *fakeBandwidthInferenceSession) Destroy() error {
	session.destroyed = true
	return nil
}

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

func TestLSTMPredictorUsesLoadedInferenceOutput(t *testing.T) {
	predictor, err := NewLSTMPredictor("")
	if err != nil {
		t.Fatalf("NewLSTMPredictor() error = %v", err)
	}
	fakeSession := &fakeBandwidthInferenceSession{
		output: []float32{0.42, 0.18, 0.03, 0.24},
	}
	predictor.inference = fakeSession
	predictor.modelLoaded = true

	history := makeTestNetworkHistory()
	prediction, err := predictor.Predict(history)
	if err != nil {
		t.Fatalf("Predict() error = %v", err)
	}

	assertFloatNear(t, prediction.PredictedBandwidthMbps, 420, 0.001)
	assertFloatNear(t, prediction.PredictedLatencyMs, 18, 0.001)
	assertFloatNear(t, prediction.PredictedPacketLoss, 0.03, 0.001)
	assertFloatNear(t, prediction.PredictedJitterMs, 12, 0.001)

	if len(fakeSession.inputSeen) != predictor.sequenceLength*predictor.featureCount {
		t.Fatalf("input length = %d, want %d", len(fakeSession.inputSeen), predictor.sequenceLength*predictor.featureCount)
	}
	assertFloatNear(t, float64(fakeSession.inputSeen[0]), history[0].BandwidthMbps/1000, 0.001)
	assertFloatNear(t, float64(fakeSession.inputSeen[1]), history[0].LatencyMs/100, 0.001)
}

func TestLSTMPredictorFallsBackWhenInferenceUnavailable(t *testing.T) {
	predictor, err := NewLSTMPredictor("")
	if err != nil {
		t.Fatalf("NewLSTMPredictor() error = %v", err)
	}
	predictor.modelLoaded = true

	prediction, err := predictor.Predict(makeTestNetworkHistory())
	if err != nil {
		t.Fatalf("Predict() error = %v", err)
	}

	assertFloatNear(t, prediction.PredictedBandwidthMbps, 154, 0.001)
	if prediction.Confidence > 0.5 {
		t.Fatalf("Confidence = %f, want fallback confidence capped at 0.5", prediction.Confidence)
	}
}

func TestLSTMPredictorRejectsInvalidInferenceOutput(t *testing.T) {
	predictor, err := NewLSTMPredictor("")
	if err != nil {
		t.Fatalf("NewLSTMPredictor() error = %v", err)
	}
	predictor.inference = &fakeBandwidthInferenceSession{
		output: []float32{0.42, float32(math.NaN()), 0.03, 0.24},
	}
	predictor.modelLoaded = true

	_, err = predictor.Predict(makeTestNetworkHistory())
	if err == nil {
		t.Fatal("Predict() error = nil, want invalid output error")
	}
}

func TestLSTMPredictorCloseDestroysInferenceSession(t *testing.T) {
	predictor, err := NewLSTMPredictor("")
	if err != nil {
		t.Fatalf("NewLSTMPredictor() error = %v", err)
	}
	fakeSession := &fakeBandwidthInferenceSession{}
	predictor.inference = fakeSession
	predictor.modelLoaded = true

	predictor.Close()

	if !fakeSession.destroyed {
		t.Fatal("Close() did not destroy inference session")
	}
	if predictor.modelLoaded {
		t.Fatal("Close() left modelLoaded = true")
	}
}

func TestLSTMPredictorPredictCanRunWhileModelVersionChanges(t *testing.T) {
	predictor, err := NewLSTMPredictor("")
	if err != nil {
		t.Fatalf("NewLSTMPredictor() error = %v", err)
	}
	history := makeTestNetworkHistory()

	done := make(chan struct{})
	go func() {
		defer close(done)
		for i := 0; i < 100; i++ {
			if _, err := predictor.Predict(history); err != nil {
				t.Errorf("Predict() error = %v", err)
				return
			}
		}
	}()

	for i := 0; i < 100; i++ {
		predictor.mu.Lock()
		predictor.modelVersion = fmt.Sprintf("v%d", i)
		predictor.mu.Unlock()
	}
	<-done
}

func TestLSTMPredictorUpdateActualIgnoresInvalidTelemetry(t *testing.T) {
	predictor, err := NewLSTMPredictor("")
	if err != nil {
		t.Fatalf("NewLSTMPredictor() error = %v", err)
	}

	history := makeTestNetworkHistory()
	prediction, err := predictor.Predict(history)
	if err != nil {
		t.Fatalf("Predict() error = %v", err)
	}

	predictor.UpdateActual(prediction.PredictionTime.Add(time.Second), NetworkSample{
		Timestamp:     prediction.PredictionTime.Add(time.Second),
		BandwidthMbps: 0,
		LatencyMs:     0,
	})

	metrics := predictor.GetMetrics()
	if math.IsNaN(metrics.AvgPredictionError) || math.IsInf(metrics.AvgPredictionError, 0) {
		t.Fatalf("AvgPredictionError = %f, want finite value", metrics.AvgPredictionError)
	}
	if math.IsNaN(metrics.Accuracy) || math.IsInf(metrics.Accuracy, 0) {
		t.Fatalf("Accuracy = %f, want finite value", metrics.Accuracy)
	}
	assertFloatNear(t, metrics.AvgPredictionError, 0, 0.001)
	assertFloatNear(t, metrics.Accuracy, 1, 0.001)
}

func makeTestNetworkHistory() []NetworkSample {
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
	return history
}

func assertFloatNear(t *testing.T, got, want, tolerance float64) {
	t.Helper()
	if math.Abs(got-want) > tolerance {
		t.Fatalf("got %f, want %f +/- %f", got, want, tolerance)
	}
}
