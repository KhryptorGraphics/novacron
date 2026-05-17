package commands

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/novacron/cli/pkg/config"
)

func TestNodeListFetchesClusterNodes(t *testing.T) {
	withTempHome(t)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/cluster/nodes" {
			t.Fatalf("expected node list path, got %s", r.URL.Path)
		}
		_ = json.NewEncoder(w).Encode([]clusterNode{
			{
				ID:                 "node-1",
				Status:             "available",
				CPU:                16,
				UsedCPU:            4,
				Memory:             32768,
				UsedMemoryMB:       8192,
				VMCount:            3,
				CPUUsagePercent:    25,
				MemoryUsagePercent: 25,
				Schedulable:        true,
			},
		})
	}))
	defer server.Close()
	addCurrentTestCluster(t, server.URL)

	var output bytes.Buffer
	cmd := NewNodeCommand()
	cmd.SetOut(&output)
	cmd.SetArgs([]string{"list"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("node list failed: %v", err)
	}

	for _, expected := range []string{"NAME", "STATUS", "CPU", "MEMORY", "VMS", "node-1", "available", "16"} {
		if !strings.Contains(output.String(), expected) {
			t.Fatalf("expected node list to contain %q, got:\n%s", expected, output.String())
		}
	}
}

func TestNodeGetFetchesNodeDetails(t *testing.T) {
	withTempHome(t)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/cluster/nodes/node-1" {
			t.Fatalf("expected node get path, got %s", r.URL.Path)
		}
		_ = json.NewEncoder(w).Encode(clusterNode{
			ID:           "node-1",
			Status:       "available",
			CPU:          16,
			UsedCPU:      4,
			Memory:       32768,
			UsedMemoryMB: 8192,
			Disk:         1000,
			UsedDiskGB:   100,
			VMCount:      3,
			Schedulable:  true,
			Labels:       map[string]string{"zone": "edge-a"},
		})
	}))
	defer server.Close()
	addCurrentTestCluster(t, server.URL)

	var output bytes.Buffer
	cmd := NewNodeCommand()
	cmd.SetOut(&output)
	cmd.SetArgs([]string{"get", "node-1"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("node get failed: %v", err)
	}

	for _, expected := range []string{"id: node-1", "status: available", "cpu: 16", "zone: edge-a"} {
		if !strings.Contains(output.String(), expected) {
			t.Fatalf("expected node details to contain %q, got:\n%s", expected, output.String())
		}
	}
}

func addCurrentTestCluster(t *testing.T, serverURL string) {
	t.Helper()

	manager, err := config.NewManager("")
	if err != nil {
		t.Fatalf("new config manager: %v", err)
	}
	if err := manager.AddCluster(config.Cluster{
		Name:      "prod",
		Server:    serverURL,
		Namespace: "default",
	}); err != nil {
		t.Fatalf("add cluster: %v", err)
	}
}
