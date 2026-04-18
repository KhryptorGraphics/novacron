package monitoring

import (
	"testing"
	"time"
)

type stubLifecycleCollector struct {
	started int
	stopped int
}

func (s *stubLifecycleCollector) Start() error {
	s.started++
	return nil
}

func (s *stubLifecycleCollector) Stop() error {
	s.stopped++
	return nil
}

func TestCollectorManagerHandlesLifecycleCollectors(t *testing.T) {
	manager := NewCollectorManager()
	collector := &stubLifecycleCollector{}

	manager.AddCollector(collector)

	if err := manager.StartAll(); err != nil {
		t.Fatalf("StartAll returned error: %v", err)
	}
	if err := manager.StopAll(); err != nil {
		t.Fatalf("StopAll returned error: %v", err)
	}

	if collector.started != 1 {
		t.Fatalf("expected collector to start once, got %d", collector.started)
	}
	if collector.stopped != 1 {
		t.Fatalf("expected collector to stop once, got %d", collector.stopped)
	}
	if got := manager.GetCollectors(); len(got) != 0 {
		t.Fatalf("expected lifecycle-only collectors to be skipped from GetCollectors, got %d", len(got))
	}
}

func TestSystemCollectorCollectRegistersCurrentValues(t *testing.T) {
	registry := NewMetricRegistry()
	collector := NewSystemCollector(registry, time.Second)

	if err := collector.registerMetrics(); err != nil {
		t.Fatalf("registerMetrics returned error: %v", err)
	}

	batches, err := collector.Collect()
	if err != nil {
		t.Fatalf("Collect returned error: %v", err)
	}

	if len(batches) != 1 {
		t.Fatalf("expected one batch, got %d", len(batches))
	}
	if batches[0].Size() != len(collector.metrics) {
		t.Fatalf("expected %d metrics in batch, got %d", len(collector.metrics), batches[0].Size())
	}

	series, err := registry.GetMetric("system.cpu.count")
	if err != nil {
		t.Fatalf("GetMetric returned error: %v", err)
	}

	lastValue := series.GetLastValue()
	if lastValue == nil {
		t.Fatal("expected cpu count series to have a latest value")
	}
	if lastValue.Value < 1 {
		t.Fatalf("expected cpu count to be positive, got %f", lastValue.Value)
	}
}

func TestMetricHistoryManagerUsesMetricSeriesCompatibility(t *testing.T) {
	registry := NewMetricRegistry()
	base := time.Now().Add(-2 * time.Hour)

	for i, value := range []float64{1, 2, 3} {
		metric := NewMetric("vm.count", MetricTypeGauge, value, map[string]string{"node": "n1"})
		metric.Timestamp = base.Add(time.Duration(i) * time.Hour)
		registry.Register(metric)
	}

	history := NewMetricHistoryManager(registry, 24*time.Hour, time.Hour)

	values, err := history.GetHistoricalValues("vm.count", base.Add(-time.Minute), time.Now().Add(time.Minute))
	if err != nil {
		t.Fatalf("GetHistoricalValues returned error: %v", err)
	}
	if len(values) != 3 {
		t.Fatalf("expected three values, got %d", len(values))
	}

	slope, err := history.AnalyzeMetricTrend("vm.count", 3*time.Hour)
	if err != nil {
		t.Fatalf("AnalyzeMetricTrend returned error: %v", err)
	}
	if slope <= 0 {
		t.Fatalf("expected positive slope, got %f", slope)
	}

	predicted, err := history.PredictMetricValue("vm.count", time.Now().Add(time.Hour))
	if err != nil {
		t.Fatalf("PredictMetricValue returned error: %v", err)
	}
	if predicted <= 3 {
		t.Fatalf("expected predicted value above latest sample, got %f", predicted)
	}
}
