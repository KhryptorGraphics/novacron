package dwcp_test

import (
	"context"
	"crypto/rand"
	"testing"
	"time"
)

// TestMigrationAdapter_GetMetrics_AvgThroughputVsBaseline proves
// GetMetrics() exposes avg_throughput_vs_reference_ratio (not the old,
// misleadingly-named average_speedup) and that the value tracks real
// throughput against the documented fixed reference point (20 MB/s for
// memory) - not a DWCP-vs-standard comparison. The exported map key
// says "reference" (avoiding collision with the unrelated "baseline_count"
// key in the same map, which counts stored VM memory/disk snapshots -
// see novacron-y45) while the underlying Go field/function keep the
// "Baseline" name (avgThroughputVsBaseline, updateAvgThroughputVsBaseline)
// since they don't sit next to that collision internally. Regression
// coverage for the novacron-chz fix.
func TestMigrationAdapter_GetMetrics_AvgThroughputVsBaseline(t *testing.T) {
	// enableDWCP=true: the throughput-vs-reference calculation
	// (migration_adapter.go, alongside compression) only runs on the
	// DWCP-enabled path today - migrateMemoryStandard is a separate
	// function that never calls updateAvgThroughputVsBaseline.
	adapter, memCh, _ := newLoopbackMigrationAdapter(t, true)

	metrics := adapter.GetMetrics()
	if _, ok := metrics["avg_throughput_vs_reference_ratio"]; !ok {
		t.Fatal("GetMetrics() missing avg_throughput_vs_reference_ratio key")
	}
	if _, stillPresent := metrics["average_speedup"]; stillPresent {
		t.Error("GetMetrics() still exposes the old, misleading 'average_speedup' key")
	}

	initial, ok := metrics["avg_throughput_vs_reference_ratio"].(float64)
	if !ok {
		t.Fatalf("avg_throughput_vs_reference_ratio is not a float64: %T", metrics["avg_throughput_vs_reference_ratio"])
	}
	if initial != 1.0 {
		t.Errorf("expected initial avg_throughput_vs_reference_ratio of 1.0, got %v", initial)
	}

	data := make([]byte, 4*1024*1024)
	rand.Read(data)
	if err := adapter.MigrateVMMemory(context.Background(), "vm-metrics-test", data, "127.0.0.1", nil); err != nil {
		t.Fatalf("MigrateVMMemory failed: %v", err)
	}
	waitForMemory(t, memCh, 30*time.Second)

	after := adapter.GetMetrics()["avg_throughput_vs_reference_ratio"].(float64)
	if after == initial {
		t.Error("avg_throughput_vs_reference_ratio did not update after a completed migration")
	}
	// Loopback throughput vastly exceeds the 20 MB/s reference point, so
	// the ratio should be well above 1.0 - proving it moved in a
	// sensible direction (real throughput / fixed reference), not just
	// that it changed at all.
	if after <= 1.0 {
		t.Errorf("expected avg_throughput_vs_reference_ratio > 1.0 after a fast loopback migration, got %v", after)
	}
}
