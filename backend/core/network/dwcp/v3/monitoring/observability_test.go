package monitoring

import (
	"testing"
	"time"
)

func TestCollectProfileDataUsesRuntimeAndProcMetrics(t *testing.T) {
	oi := &ObservabilityIntegration{profiler: &PerformanceProfiler{}}

	oi.collectProfileData()
	time.Sleep(10 * time.Millisecond)
	oi.collectProfileData()

	data := oi.GetProfilingData()
	if _, ok := data["avg_cpu_percent"].(float64); !ok {
		t.Fatalf("avg_cpu_percent missing or wrong type: %#v", data["avg_cpu_percent"])
	}
	if _, ok := data["avg_memory_mb"].(uint64); !ok {
		t.Fatalf("avg_memory_mb missing or wrong type: %#v", data["avg_memory_mb"])
	}
	if _, ok := data["io_read_bytes"].(uint64); !ok {
		t.Fatalf("io_read_bytes missing or wrong type: %#v", data["io_read_bytes"])
	}
	if _, ok := data["io_write_bytes"].(uint64); !ok {
		t.Fatalf("io_write_bytes missing or wrong type: %#v", data["io_write_bytes"])
	}
}

func TestLogCollectorSearchMatchesMessageComponentAndFields(t *testing.T) {
	collector := newLogCollector(10)
	collector.Add(&StructuredLog{
		Timestamp: time.Now(),
		Level:     "info",
		Component: "transport",
		Message:   "mode switched",
		Fields:    map[string]interface{}{"node": "edge-a"},
	})
	collector.Add(&StructuredLog{
		Timestamp: time.Now(),
		Level:     "warn",
		Component: "compression",
		Message:   "ratio degraded",
		Fields:    map[string]interface{}{"node": "edge-b"},
	})

	if got := collector.Search("mode", "", "", time.Time{}, 10); len(got) != 1 || got[0].Component != "transport" {
		t.Fatalf("message search returned unexpected results: %+v", got)
	}
	if got := collector.Search("compression", "", "", time.Time{}, 10); len(got) != 1 || got[0].Component != "compression" {
		t.Fatalf("component search returned unexpected results: %+v", got)
	}
	if got := collector.Search("edge-b", "", "", time.Time{}, 10); len(got) != 1 || got[0].Component != "compression" {
		t.Fatalf("field search returned unexpected results: %+v", got)
	}
}
