package commands

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/novacron/cli/pkg/config"
)

func TestClusterListPrintsConfiguredClusters(t *testing.T) {
	withTempHome(t)

	manager, err := config.NewManager("")
	if err != nil {
		t.Fatalf("new config manager: %v", err)
	}
	if err := manager.AddCluster(config.Cluster{Name: "prod", Server: "https://prod.example.com", Namespace: "production"}); err != nil {
		t.Fatalf("add prod: %v", err)
	}
	if err := manager.AddCluster(config.Cluster{Name: "stage", Server: "https://stage.example.com", Namespace: "default"}); err != nil {
		t.Fatalf("add stage: %v", err)
	}

	var output bytes.Buffer
	cmd := NewClusterCommand()
	cmd.SetOut(&output)
	cmd.SetArgs([]string{"list"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("cluster list failed: %v", err)
	}

	for _, expected := range []string{"CURRENT", "NAME", "SERVER", "prod", "stage", "*"} {
		if !strings.Contains(output.String(), expected) {
			t.Fatalf("expected cluster list to contain %q, got:\n%s", expected, output.String())
		}
	}
}

func TestClusterInfoFetchesClusterHealth(t *testing.T) {
	withTempHome(t)

	now := time.Now().UTC().Truncate(time.Second)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/cluster/health" {
			t.Fatalf("expected health path, got %s", r.URL.Path)
		}
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"status":        "healthy",
			"total_nodes":   3,
			"healthy_nodes": 3,
			"has_quorum":    true,
			"leader":        "node-1",
			"last_updated":  now,
		})
	}))
	defer server.Close()

	manager, err := config.NewManager("")
	if err != nil {
		t.Fatalf("new config manager: %v", err)
	}
	if err := manager.AddCluster(config.Cluster{Name: "prod", Server: server.URL, Namespace: "default"}); err != nil {
		t.Fatalf("add cluster: %v", err)
	}

	var output bytes.Buffer
	cmd := NewClusterCommand()
	cmd.SetOut(&output)
	cmd.SetArgs([]string{"info"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("cluster info failed: %v", err)
	}

	for _, expected := range []string{"STATUS", "healthy", "TOTAL NODES", "3", "LEADER", "node-1"} {
		if !strings.Contains(output.String(), expected) {
			t.Fatalf("expected cluster info to contain %q, got:\n%s", expected, output.String())
		}
	}
}
