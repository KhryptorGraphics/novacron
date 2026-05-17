package commands

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestRolloutRollbackPostsMigrationJobRollback(t *testing.T) {
	withTempHome(t)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Fatalf("expected POST, got %s", r.Method)
		}
		if r.URL.Path != "/migration/jobs/job-1/rollback" {
			t.Fatalf("expected rollback path, got %s", r.URL.Path)
		}
		w.WriteHeader(http.StatusAccepted)
		_ = json.NewEncoder(w).Encode(rolloutRollbackResponse{
			JobID:      "job-1",
			RollbackID: "rollback-1",
			Status:     "queued",
			CreatedAt:  "2026-05-17T01:38:00Z",
			Condition:  "operator_requested",
		})
	}))
	defer server.Close()
	addCurrentTestCluster(t, server.URL)

	var output bytes.Buffer
	cmd := NewRolloutCommand()
	cmd.SetOut(&output)
	cmd.SetArgs([]string{"rollback", "job-1"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("rollout rollback failed: %v", err)
	}

	for _, expected := range []string{"job_id: job-1", "rollback_id: rollback-1", "status: queued", "condition: operator_requested"} {
		if !strings.Contains(output.String(), expected) {
			t.Fatalf("expected rollout output to contain %q, got:\n%s", expected, output.String())
		}
	}
}

func TestRolloutRollbackRequiresJobID(t *testing.T) {
	cmd := NewRolloutCommand()
	cmd.SetArgs([]string{"rollback"})

	if err := cmd.Execute(); err == nil {
		t.Fatal("expected missing job id error")
	}
}

func TestRolloutRejectsUnsupportedSubcommand(t *testing.T) {
	cmd := NewRolloutCommand()
	cmd.SetArgs([]string{"status", "job-1"})

	if err := cmd.Execute(); err == nil || !strings.Contains(err.Error(), "unknown command") {
		t.Fatalf("expected unsupported rollout subcommand error, got %v", err)
	}
}
