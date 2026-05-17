package commands

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestMonitorMetricsFetchesVMMetrics(t *testing.T) {
	withTempHome(t)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v1/vms/vm-42/metrics" {
			t.Fatalf("expected VM metrics path, got %s", r.URL.Path)
		}
		_ = json.NewEncoder(w).Encode(vmMetricsResponse{
			ID:          "vm-42",
			CPUUsage:    32.5,
			MemoryUsage: 61.25,
		})
	}))
	defer server.Close()
	addCurrentTestCluster(t, server.URL)

	var output bytes.Buffer
	cmd := NewMonitorCommand()
	cmd.SetOut(&output)
	cmd.SetArgs([]string{"metrics", "vm-42"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("monitor metrics failed: %v", err)
	}

	for _, expected := range []string{"ID", "vm-42", "CPU USAGE", "32.50%", "MEMORY USAGE", "61.25%"} {
		if !strings.Contains(output.String(), expected) {
			t.Fatalf("expected monitor metrics output to contain %q, got:\n%s", expected, output.String())
		}
	}
}

func TestMonitorVMSFetchesMonitoringVMs(t *testing.T) {
	withTempHome(t)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/monitoring/vms" {
			t.Fatalf("expected monitoring VMs path, got %s", r.URL.Path)
		}
		_ = json.NewEncoder(w).Encode([]monitoringVM{
			{
				VMID:        "vm-42",
				Name:        "web-1",
				Status:      "running",
				CPUUsage:    32.5,
				MemoryUsage: 61.25,
				DiskUsage:   10,
				NetworkRx:   1024,
				NetworkTx:   2048,
				IOPS:        100,
			},
		})
	}))
	defer server.Close()
	addCurrentTestCluster(t, server.URL)

	var output bytes.Buffer
	cmd := NewMonitorCommand()
	cmd.SetOut(&output)
	cmd.SetArgs([]string{"vms"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("monitor vms failed: %v", err)
	}

	for _, expected := range []string{"VM ID", "NAME", "STATUS", "CPU", "MEMORY", "vm-42", "web-1", "running"} {
		if !strings.Contains(output.String(), expected) {
			t.Fatalf("expected monitor vms output to contain %q, got:\n%s", expected, output.String())
		}
	}
}
