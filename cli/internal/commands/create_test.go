package commands

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestCreateVMPostsCanonicalCreateRequest(t *testing.T) {
	withTempHome(t)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v1/vms" {
			t.Fatalf("expected VM create path, got %s", r.URL.Path)
		}
		if r.Method != http.MethodPost {
			t.Fatalf("expected POST, got %s", r.Method)
		}
		var req createVMRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode create VM request: %v", err)
		}
		if req.Name != "web-1" || req.NodeID != "node-a" || req.CPUShares != 4 || req.MemoryMB != 8192 {
			t.Fatalf("unexpected VM create request: %#v", req)
		}
		w.WriteHeader(http.StatusCreated)
		_ = json.NewEncoder(w).Encode(coreVM{
			ID: "vm-1", Name: req.Name, State: "creating", Status: "creating", NodeID: req.NodeID, TenantID: "default",
		})
	}))
	defer server.Close()
	addCurrentTestCluster(t, server.URL)

	var output bytes.Buffer
	cmd := NewCreateCommand()
	cmd.SetOut(&output)
	cmd.SetArgs([]string{"vm", "web-1", "--node", "node-a", "--cpu", "4", "--memory", "8192"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("create vm failed: %v", err)
	}

	for _, expected := range []string{"id: vm-1", "name: web-1", "status: creating", "node_id: node-a"} {
		if !strings.Contains(output.String(), expected) {
			t.Fatalf("expected create vm output to contain %q, got:\n%s", expected, output.String())
		}
	}
}

func TestCreateNetworkPostsCanonicalCreateRequest(t *testing.T) {
	withTempHome(t)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v1/networks" {
			t.Fatalf("expected network create path, got %s", r.URL.Path)
		}
		if r.Method != http.MethodPost {
			t.Fatalf("expected POST, got %s", r.Method)
		}
		var req createNetworkRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode create network request: %v", err)
		}
		if req.Name != "public" || req.Type != "bridged" || req.Subnet != "10.0.0.0/24" || req.Gateway != "10.0.0.1" {
			t.Fatalf("unexpected network create request: %#v", req)
		}
		w.WriteHeader(http.StatusCreated)
		_ = json.NewEncoder(w).Encode(coreNetwork{
			ID: "net-1", Name: req.Name, Type: req.Type, Subnet: req.Subnet, Gateway: req.Gateway, Status: "active",
		})
	}))
	defer server.Close()
	addCurrentTestCluster(t, server.URL)

	var output bytes.Buffer
	cmd := NewCreateCommand()
	cmd.SetOut(&output)
	cmd.SetArgs([]string{"network", "public", "--type", "bridged", "--subnet", "10.0.0.0/24", "--gateway", "10.0.0.1"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("create network failed: %v", err)
	}

	for _, expected := range []string{"id: net-1", "name: public", "type: bridged", "subnet: 10.0.0.0/24"} {
		if !strings.Contains(output.String(), expected) {
			t.Fatalf("expected create network output to contain %q, got:\n%s", expected, output.String())
		}
	}
}

func TestCreateNetworkRequiresSubnet(t *testing.T) {
	cmd := NewCreateCommand()
	cmd.SetArgs([]string{"network", "public"})

	if err := cmd.Execute(); err == nil || !strings.Contains(err.Error(), "subnet is required") {
		t.Fatalf("expected subnet validation error, got %v", err)
	}
}
