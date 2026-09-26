package main

import "testing"

// TestHostMetricsAreRealNotFabricated proves the /monitoring/metrics source no
// longer returns the old hardcoded placeholder values or invented narrative, and
// that the four current* keys are ALWAYS present (so unguarded client
// dereferences like analytics/page.tsx `currentNetworkUsage.toFixed(1)` don't
// crash). Discriminator: the prior handler returned a static map with
// "cpuAnalysis"/"timeLabels" and a constant currentMemoryUsage:72.1.
func TestHostMetricsAreRealNotFabricated(t *testing.T) {
	m := hostMetrics(".")

	// No fabricated narrative/trend fields.
	for _, k := range []string{"cpuAnalysis", "memoryAnalysis", "timeLabels", "cpuChangePercentage", "memoryChangePercentage"} {
		if _, ok := m[k]; ok {
			t.Fatalf("hostMetrics still returns fabricated field %q", k)
		}
	}

	// The four current* keys must ALWAYS be present (value may be nil when a
	// metric is unmeasurable, but the key must exist for client rendering).
	for _, k := range []string{"currentCpuUsage", "currentMemoryUsage", "currentDiskUsage", "currentNetworkUsage"} {
		if _, ok := m[k]; !ok {
			t.Fatalf("hostMetrics must always emit key %q (clients dereference it unguarded)", k)
		}
	}

	// When present as a real measurement, values must be physically plausible.
	if v, ok := m["currentMemoryUsage"].(float64); ok {
		if v <= 0 || v > 100 {
			t.Fatalf("currentMemoryUsage %v out of physical range (0,100]", v)
		}
	}
	if v, ok := m["currentDiskUsage"].(float64); ok {
		if v < 0 || v > 100 {
			t.Fatalf("currentDiskUsage %v out of physical range [0,100]", v)
		}
	}
	if v, ok := m["currentCpuUsage"].(float64); ok {
		if v < 0 || v > 100 {
			t.Fatalf("currentCpuUsage %v out of physical range [0,100]", v)
		}
	}
	if v, ok := m["currentNetworkUsage"].(float64); ok {
		if v < 0 {
			t.Fatalf("currentNetworkUsage %v must be non-negative MB/s", v)
		}
	}
}

func TestMetricsDataFromHostUsesMeasuredFractions(t *testing.T) {
	data, err := metricsDataFromHost(map[string]interface{}{
		"currentCpuUsage":     80.0,
		"currentMemoryUsage":  40.0,
		"currentNetworkUsage": 1.25,
		"currentDiskUsage":    70.0,
	})
	if err != nil {
		t.Fatal(err)
	}
	if data.CPUUsage != 0.8 || data.MemoryUsage != 0.4 {
		t.Fatalf("fractions = %v %v", data.CPUUsage, data.MemoryUsage)
	}
	if data.NetworkIO != 1.25 || data.DiskIO != 0 {
		t.Fatalf("io = net %v disk %v", data.NetworkIO, data.DiskIO)
	}
	if data.CustomMetrics["disk_usage"] != 0.7 {
		t.Fatalf("disk usage = %v", data.CustomMetrics["disk_usage"])
	}

	if _, err := metricsDataFromHost(map[string]interface{}{
		"currentCpuUsage":    nil,
		"currentMemoryUsage": 40.0,
	}); err == nil {
		t.Fatal("missing cpu was published")
	}
}
