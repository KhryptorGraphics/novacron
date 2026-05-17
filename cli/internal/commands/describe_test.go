package commands

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestDescribeVMFetchesCanonicalVMDetail(t *testing.T) {
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
	cmd := NewDescribeCommand()
	cmd.SetOut(&output)
	cmd.SetArgs([]string{"vm", "vm-1"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("describe vm failed: %v", err)
	}

	for _, expected := range []string{"Name:", "web-1", "ID:", "vm-1", "Status:", "running", "Node:", "node-a"} {
		if !strings.Contains(output.String(), expected) {
			t.Fatalf("expected describe vm output to contain %q, got:\n%s", expected, output.String())
		}
	}
}

func TestDescribeNetworkFetchesCanonicalNetworkDetail(t *testing.T) {
	withTempHome(t)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v1/networks/net-1" {
			t.Fatalf("expected network detail path, got %s", r.URL.Path)
		}
		_ = json.NewEncoder(w).Encode(coreNetwork{
			ID: "net-1", Name: "public", Type: "bridged", Subnet: "10.0.0.0/24", Gateway: "10.0.0.1", Status: "active",
		})
	}))
	defer server.Close()
	addCurrentTestCluster(t, server.URL)

	var output bytes.Buffer
	cmd := NewDescribeCommand()
	cmd.SetOut(&output)
	cmd.SetArgs([]string{"network", "net-1"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("describe network failed: %v", err)
	}

	for _, expected := range []string{"Name:", "public", "ID:", "net-1", "Type:", "bridged", "Subnet:", "10.0.0.0/24"} {
		if !strings.Contains(output.String(), expected) {
			t.Fatalf("expected describe network output to contain %q, got:\n%s", expected, output.String())
		}
	}
}

func TestDescribeCommandRejectsUnsupportedResource(t *testing.T) {
	cmd := NewDescribeCommand()
	cmd.SetArgs([]string{"pod", "pod-1"})

	if err := cmd.Execute(); err == nil || !strings.Contains(err.Error(), "unsupported resource") {
		t.Fatalf("expected unsupported resource error, got %v", err)
	}
}
