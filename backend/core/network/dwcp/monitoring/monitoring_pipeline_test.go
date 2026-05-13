package monitoring

import (
	"testing"
	"time"

	"go.uber.org/zap"
)

func TestMonitoringPipelineStoresRecentAnomalies(t *testing.T) {
	detector, err := NewAnomalyDetector(DefaultDetectorConfig(), zap.NewNop())
	if err != nil {
		t.Fatalf("NewAnomalyDetector failed: %v", err)
	}
	pipeline := NewMonitoringPipeline(detector, nil, time.Second, zap.NewNop())

	old := &Anomaly{
		Timestamp:  time.Now().Add(-2 * time.Hour),
		MetricName: "latency",
		Severity:   SeverityWarning,
		ModelType:  "test",
		Context:    map[string]interface{}{"source": "old"},
	}
	recent := &Anomaly{
		Timestamp:  time.Now(),
		MetricName: "packet_loss",
		Severity:   SeverityCritical,
		ModelType:  "test",
		Context:    map[string]interface{}{"source": "recent"},
	}

	pipeline.storeAnomaly(old)
	pipeline.storeAnomaly(recent)

	anomalies := pipeline.GetRecentAnomalies(time.Hour)
	if len(anomalies) != 1 {
		t.Fatalf("recent anomaly count = %d, want 1", len(anomalies))
	}
	if anomalies[0].MetricName != "packet_loss" {
		t.Fatalf("recent anomaly metric = %s, want packet_loss", anomalies[0].MetricName)
	}

	anomalies[0].Context["source"] = "mutated"
	again := pipeline.GetRecentAnomalies(time.Hour)
	if again[0].Context["source"] != "recent" {
		t.Fatalf("stored anomaly context was mutated through returned copy: %v", again[0].Context)
	}
}

func TestMonitoringPipelineAnomalyHistoryBound(t *testing.T) {
	detector, err := NewAnomalyDetector(DefaultDetectorConfig(), zap.NewNop())
	if err != nil {
		t.Fatalf("NewAnomalyDetector failed: %v", err)
	}
	pipeline := NewMonitoringPipeline(detector, nil, time.Second, zap.NewNop())
	pipeline.maxAnomalies = 2

	for i := 0; i < 3; i++ {
		pipeline.storeAnomaly(&Anomaly{
			Timestamp:  time.Now().Add(time.Duration(i) * time.Second),
			MetricName: "metric",
			Severity:   SeverityInfo,
			ModelType:  "test",
			Value:      float64(i),
		})
	}

	anomalies := pipeline.GetRecentAnomalies(0)
	if len(anomalies) != 2 {
		t.Fatalf("history count = %d, want 2", len(anomalies))
	}
	if anomalies[0].Value != 1 || anomalies[1].Value != 2 {
		t.Fatalf("history did not retain newest anomalies: %+v", anomalies)
	}
}
