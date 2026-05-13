package monitoring

import (
	"testing"

	"go.uber.org/zap"
)

func TestDetectAnomalyReportsExpectedAndDeviation(t *testing.T) {
	detector, err := NewAnomalyDetector(&DetectorConfig{}, zap.NewNop())
	if err != nil {
		t.Fatalf("NewAnomalyDetector() error = %v", err)
	}

	result := detector.DetectAnomaly("latency", 750)
	if !result.IsAnomaly {
		t.Fatal("DetectAnomaly() did not flag high latency")
	}
	if result.Expected != 500 {
		t.Fatalf("Expected = %f, want 500", result.Expected)
	}
	if result.Deviation != 250 {
		t.Fatalf("Deviation = %f, want 250", result.Deviation)
	}
}

func TestDetectAnomalyUsesValueAsExpectedInsideRange(t *testing.T) {
	detector, err := NewAnomalyDetector(&DetectorConfig{}, zap.NewNop())
	if err != nil {
		t.Fatalf("NewAnomalyDetector() error = %v", err)
	}

	result := detector.DetectAnomaly("packet_loss", 2.5)
	if result.IsAnomaly {
		t.Fatal("DetectAnomaly() flagged in-range packet loss")
	}
	if result.Expected != 2.5 {
		t.Fatalf("Expected = %f, want 2.5", result.Expected)
	}
	if result.Deviation != 0 {
		t.Fatalf("Deviation = %f, want 0", result.Deviation)
	}
}
