package commands

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestSnapshotCreatePostsCanonicalRequest(t *testing.T) {
	withTempHome(t)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Fatalf("expected POST, got %s", r.Method)
		}
		if r.URL.Path != "/api/v1/vms/vm-1/snapshot" {
			t.Fatalf("expected snapshot path, got %s", r.URL.Path)
		}
		var req createSnapshotRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode snapshot request: %v", err)
		}
		if req.Name != "before-upgrade" || req.Description != "pre upgrade" || !req.Memory || !req.Quiesce {
			t.Fatalf("unexpected snapshot request: %#v", req)
		}
		w.WriteHeader(http.StatusAccepted)
		_ = json.NewEncoder(w).Encode(snapshotResponse{
			SnapshotID: "snap-1",
			Status:     "creating",
			Message:    "Snapshot creation initiated",
		})
	}))
	defer server.Close()
	addCurrentTestCluster(t, server.URL)

	var output bytes.Buffer
	cmd := NewSnapshotCommand()
	cmd.SetOut(&output)
	cmd.SetArgs([]string{"create", "vm-1", "before-upgrade", "--description", "pre upgrade", "--memory", "--quiesce"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("snapshot create failed: %v", err)
	}

	for _, expected := range []string{"snapshot_id: snap-1", "status: creating", "message: Snapshot creation initiated"} {
		if !strings.Contains(output.String(), expected) {
			t.Fatalf("expected snapshot output to contain %q, got:\n%s", expected, output.String())
		}
	}
}

func TestSnapshotCreateRequiresName(t *testing.T) {
	cmd := NewSnapshotCommand()
	cmd.SetArgs([]string{"create", "vm-1"})

	if err := cmd.Execute(); err == nil {
		t.Fatal("expected missing snapshot name error")
	}
}

func TestSnapshotRejectsUnsupportedSubcommand(t *testing.T) {
	cmd := NewSnapshotCommand()
	cmd.SetArgs([]string{"delete", "snap-1"})

	if err := cmd.Execute(); err == nil || !strings.Contains(err.Error(), "unknown command") {
		t.Fatalf("expected unsupported snapshot subcommand error, got %v", err)
	}
}
