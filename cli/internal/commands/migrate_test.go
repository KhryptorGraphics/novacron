package commands

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestMigratePostsLiveMigrationRequest(t *testing.T) {
	withTempHome(t)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Fatalf("expected POST, got %s", r.Method)
		}
		if r.URL.Path != "/migration/live/vm-1" {
			t.Fatalf("expected live migration path, got %s", r.URL.Path)
		}
		var req liveMigrationRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode live migration request: %v", err)
		}
		if req.TargetNode != "node-b" || req.MaxDowntime != 250 || req.Bandwidth != 1024 || req.Priority != "high" {
			t.Fatalf("unexpected live migration request: %#v", req)
		}
		w.WriteHeader(http.StatusAccepted)
		_ = json.NewEncoder(w).Encode(migrationResponse{
			MigrationID: "migration-1",
			Status:      "queued",
			VMID:        "vm-1",
			TargetNode:  req.TargetNode,
			Priority:    req.Priority,
			CreatedAt:   "2026-05-17T01:15:00Z",
		})
	}))
	defer server.Close()
	addCurrentTestCluster(t, server.URL)

	var output bytes.Buffer
	cmd := NewMigrateCommand()
	cmd.SetOut(&output)
	cmd.SetArgs([]string{"vm-1", "--target-node", "node-b", "--max-downtime", "250", "--bandwidth", "1024", "--priority", "high"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("migrate failed: %v", err)
	}

	for _, expected := range []string{"migration_id: migration-1", "status: queued", "vm_id: vm-1", "target_node: node-b", "priority: high"} {
		if !strings.Contains(output.String(), expected) {
			t.Fatalf("expected migrate output to contain %q, got:\n%s", expected, output.String())
		}
	}
}

func TestMigrateRequiresTargetNode(t *testing.T) {
	cmd := NewMigrateCommand()
	cmd.SetArgs([]string{"vm-1"})

	if err := cmd.Execute(); err == nil || !strings.Contains(err.Error(), "target-node is required") {
		t.Fatalf("expected target-node validation error, got %v", err)
	}
}

func TestMigrateRejectsNegativeTuningValues(t *testing.T) {
	cmd := NewMigrateCommand()
	cmd.SetArgs([]string{"vm-1", "--target-node", "node-b", "--bandwidth", "-1"})

	if err := cmd.Execute(); err == nil || !strings.Contains(err.Error(), "bandwidth must be non-negative") {
		t.Fatalf("expected bandwidth validation error, got %v", err)
	}
}
