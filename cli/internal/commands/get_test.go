package commands

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestGetVMsFetchesCanonicalVMList(t *testing.T) {
	withTempHome(t)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v1/vms" {
			t.Fatalf("expected VM list path, got %s", r.URL.Path)
		}
		_ = json.NewEncoder(w).Encode([]coreVM{
			{ID: "vm-1", Name: "web-1", State: "running", Status: "running", NodeID: "node-a", TenantID: "default"},
		})
	}))
	defer server.Close()
	addCurrentTestCluster(t, server.URL)

	var output bytes.Buffer
	cmd := NewGetCommand()
	cmd.SetOut(&output)
	cmd.SetArgs([]string{"vms"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("get vms failed: %v", err)
	}

	for _, expected := range []string{"ID", "NAME", "STATUS", "NODE", "TENANT", "vm-1", "web-1", "running"} {
		if !strings.Contains(output.String(), expected) {
			t.Fatalf("expected get vms output to contain %q, got:\n%s", expected, output.String())
		}
	}
}

func TestGetVMFetchesCanonicalVMDetail(t *testing.T) {
	withTempHome(t)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v1/vms/vm-1" {
			t.Fatalf("expected VM detail path, got %s", r.URL.Path)
		}
		_ = json.NewEncoder(w).Encode(coreVM{
			ID: "vm-1", Name: "web-1", State: "running", Status: "running", NodeID: "node-a", TenantID: "default",
		})
	}))
	defer server.Close()
	addCurrentTestCluster(t, server.URL)

	var output bytes.Buffer
	cmd := NewGetCommand()
	cmd.SetOut(&output)
	cmd.SetArgs([]string{"vm", "vm-1"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("get vm failed: %v", err)
	}

	for _, expected := range []string{"id: vm-1", "name: web-1", "status: running", "node_id: node-a"} {
		if !strings.Contains(output.String(), expected) {
			t.Fatalf("expected get vm output to contain %q, got:\n%s", expected, output.String())
		}
	}
}

func TestGetNetworksFetchesCanonicalNetworkList(t *testing.T) {
	withTempHome(t)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v1/networks" {
			t.Fatalf("expected network list path, got %s", r.URL.Path)
		}
		_ = json.NewEncoder(w).Encode([]coreNetwork{
			{ID: "net-1", Name: "public", Type: "bridged", Subnet: "10.0.0.0/24", Gateway: "10.0.0.1", Status: "active"},
		})
	}))
	defer server.Close()
	addCurrentTestCluster(t, server.URL)

	var output bytes.Buffer
	cmd := NewGetCommand()
	cmd.SetOut(&output)
	cmd.SetArgs([]string{"networks"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("get networks failed: %v", err)
	}

	for _, expected := range []string{"ID", "NAME", "TYPE", "SUBNET", "STATUS", "net-1", "public", "bridged"} {
		if !strings.Contains(output.String(), expected) {
			t.Fatalf("expected get networks output to contain %q, got:\n%s", expected, output.String())
		}
	}
}

func TestGetCommandRejectsUnsupportedResource(t *testing.T) {
	cmd := NewGetCommand()
	cmd.SetArgs([]string{"pods"})

	if err := cmd.Execute(); err == nil || !strings.Contains(err.Error(), "unsupported resource") {
		t.Fatalf("expected unsupported resource error, got %v", err)
	}
}
