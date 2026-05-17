package commands

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestTopCommandFetchesMonitoringSummary(t *testing.T) {
	withTempHome(t)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/monitoring/metrics" {
			t.Fatalf("expected monitoring metrics path, got %s", r.URL.Path)
		}
		_ = json.NewEncoder(w).Encode(monitoringSummary{
			CurrentCPUUsage:         42.5,
			CurrentMemoryUsage:      67.25,
			CurrentDiskUsage:        18,
			CurrentNetworkUsage:     9.5,
			CPUChangePercentage:     1.2,
			MemoryChangePercentage:  -0.4,
			DiskChangePercentage:    0,
			NetworkChangePercentage: 3.75,
			CPUAnalysis:             "CPU usage is healthy",
			MemoryAnalysis:          "Memory pressure is moderate",
		})
	}))
	defer server.Close()
	addCurrentTestCluster(t, server.URL)

	var output bytes.Buffer
	cmd := NewTopCommand()
	cmd.SetOut(&output)
	if err := cmd.Execute(); err != nil {
		t.Fatalf("top command failed: %v", err)
	}

	for _, expected := range []string{
		"RESOURCE", "USAGE", "CHANGE", "CPU", "42.50%", "Memory", "67.25%",
		"Disk", "18.00%", "Network", "9.50%", "CPU usage is healthy", "Memory pressure is moderate",
	} {
		if !strings.Contains(output.String(), expected) {
			t.Fatalf("expected top output to contain %q, got:\n%s", expected, output.String())
		}
	}
}

func TestTopCommandRejectsUnsupportedResource(t *testing.T) {
	cmd := NewTopCommand()
	cmd.SetArgs([]string{"vm"})

	if err := cmd.Execute(); err == nil || !strings.Contains(err.Error(), "unsupported resource") {
		t.Fatalf("expected unsupported resource error, got %v", err)
	}
}
