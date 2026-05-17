package commands

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestApplyVMManifestPostsCreateRequest(t *testing.T) {
	withTempHome(t)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Fatalf("expected POST, got %s", r.Method)
		}
		if r.URL.Path != "/api/v1/vms" {
			t.Fatalf("expected VM create path, got %s", r.URL.Path)
		}
		var req createVMRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode VM apply request: %v", err)
		}
		if req.Name != "web-1" || req.NodeID != "node-a" || req.CPUShares != 2 || req.MemoryMB != 4096 {
			t.Fatalf("unexpected VM apply request: %#v", req)
		}
		w.WriteHeader(http.StatusCreated)
		_ = json.NewEncoder(w).Encode(coreVM{ID: "vm-1", Name: req.Name, Status: "creating", NodeID: req.NodeID})
	}))
	defer server.Close()
	addCurrentTestCluster(t, server.URL)

	manifest := writeApplyManifest(t, `
kind: VM
metadata:
  name: web-1
spec:
  node_id: node-a
  cpu_shares: 2
  memory_mb: 4096
`)

	var output bytes.Buffer
	cmd := NewApplyCommand()
	cmd.SetOut(&output)
	cmd.SetArgs([]string{"-f", manifest})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("apply vm failed: %v", err)
	}

	for _, expected := range []string{"id: vm-1", "name: web-1", "status: creating", "node_id: node-a"} {
		if !strings.Contains(output.String(), expected) {
			t.Fatalf("expected apply output to contain %q, got:\n%s", expected, output.String())
		}
	}
}

func TestApplyNetworkManifestPostsCreateRequest(t *testing.T) {
	withTempHome(t)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Fatalf("expected POST, got %s", r.Method)
		}
		if r.URL.Path != "/api/v1/networks" {
			t.Fatalf("expected network create path, got %s", r.URL.Path)
		}
		var req createNetworkRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode network apply request: %v", err)
		}
		if req.Name != "public" || req.Type != "bridged" || req.Subnet != "10.10.0.0/24" || req.Gateway != "10.10.0.1" {
			t.Fatalf("unexpected network apply request: %#v", req)
		}
		w.WriteHeader(http.StatusCreated)
		_ = json.NewEncoder(w).Encode(coreNetwork{ID: "net-1", Name: req.Name, Type: req.Type, Subnet: req.Subnet, Gateway: req.Gateway, Status: "active"})
	}))
	defer server.Close()
	addCurrentTestCluster(t, server.URL)

	manifest := writeApplyManifest(t, `
kind: Network
metadata:
  name: public
spec:
  type: bridged
  subnet: 10.10.0.0/24
  gateway: 10.10.0.1
`)

	var output bytes.Buffer
	cmd := NewApplyCommand()
	cmd.SetOut(&output)
	cmd.SetArgs([]string{"--file", manifest})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("apply network failed: %v", err)
	}
	if !strings.Contains(output.String(), "id: net-1") || !strings.Contains(output.String(), "subnet: 10.10.0.0/24") {
		t.Fatalf("expected apply network output, got:\n%s", output.String())
	}
}

func TestApplyRequiresFile(t *testing.T) {
	cmd := NewApplyCommand()

	if err := cmd.Execute(); err == nil || !strings.Contains(err.Error(), "file is required") {
		t.Fatalf("expected file validation error, got %v", err)
	}
}

func writeApplyManifest(t *testing.T, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "resource.yaml")
	if err := os.WriteFile(path, []byte(strings.TrimSpace(content)+"\n"), 0o600); err != nil {
		t.Fatalf("write manifest: %v", err)
	}
	return path
}
