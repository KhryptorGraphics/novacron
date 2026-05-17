package commands

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestDeleteVMRequiresForce(t *testing.T) {
	cmd := NewDeleteCommand()
	cmd.SetArgs([]string{"vm", "vm-1"})

	if err := cmd.Execute(); err == nil || !strings.Contains(err.Error(), "requires --force") {
		t.Fatalf("expected force validation error, got %v", err)
	}
}

func TestDeleteVMCallsCanonicalDeleteRoute(t *testing.T) {
	withTempHome(t)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodDelete {
			t.Fatalf("expected DELETE, got %s", r.Method)
		}
		if r.URL.Path != "/api/v1/vms/vm-1" {
			t.Fatalf("expected VM delete path, got %s", r.URL.Path)
		}
		_ = json.NewEncoder(w).Encode(deleteResponse{ID: "vm-1", Status: "deleted"})
	}))
	defer server.Close()
	addCurrentTestCluster(t, server.URL)

	var output bytes.Buffer
	cmd := NewDeleteCommand()
	cmd.SetOut(&output)
	cmd.SetArgs([]string{"vm", "vm-1", "--force"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("delete vm failed: %v", err)
	}

	for _, expected := range []string{"id: vm-1", "status: deleted"} {
		if !strings.Contains(output.String(), expected) {
			t.Fatalf("expected delete vm output to contain %q, got:\n%s", expected, output.String())
		}
	}
}

func TestDeleteNetworkCallsCanonicalDeleteRoute(t *testing.T) {
	withTempHome(t)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodDelete {
			t.Fatalf("expected DELETE, got %s", r.Method)
		}
		if r.URL.Path != "/api/v1/networks/net-1" {
			t.Fatalf("expected network delete path, got %s", r.URL.Path)
		}
		_ = json.NewEncoder(w).Encode(deleteResponse{ID: "net-1", Status: "deleted"})
	}))
	defer server.Close()
	addCurrentTestCluster(t, server.URL)

	var output bytes.Buffer
	cmd := NewDeleteCommand()
	cmd.SetOut(&output)
	cmd.SetArgs([]string{"network", "net-1", "--force"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("delete network failed: %v", err)
	}
	if !strings.Contains(output.String(), "id: net-1") {
		t.Fatalf("expected delete network output, got:\n%s", output.String())
	}
}

func TestDeleteInterfaceCallsCanonicalDeleteRoute(t *testing.T) {
	withTempHome(t)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodDelete {
			t.Fatalf("expected DELETE, got %s", r.Method)
		}
		if r.URL.Path != "/api/v1/vms/vm-1/interfaces/iface-1" {
			t.Fatalf("expected interface delete path, got %s", r.URL.Path)
		}
		_ = json.NewEncoder(w).Encode(deleteResponse{ID: "iface-1", VMID: "vm-1", Status: "detached"})
	}))
	defer server.Close()
	addCurrentTestCluster(t, server.URL)

	var output bytes.Buffer
	cmd := NewDeleteCommand()
	cmd.SetOut(&output)
	cmd.SetArgs([]string{"interface", "iface-1", "--vm", "vm-1", "--force"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("delete interface failed: %v", err)
	}
	if !strings.Contains(output.String(), "status: detached") {
		t.Fatalf("expected delete interface output, got:\n%s", output.String())
	}
}
