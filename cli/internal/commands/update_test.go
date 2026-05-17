package commands

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestUpdateInterfaceCallsCanonicalPutRoute(t *testing.T) {
	withTempHome(t)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPut {
			t.Fatalf("expected PUT, got %s", r.Method)
		}
		if r.URL.Path != "/api/v1/vms/vm-1/interfaces/iface-1" {
			t.Fatalf("expected interface update path, got %s", r.URL.Path)
		}
		var req updateVMInterfaceRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode update interface request: %v", err)
		}
		if req.NetworkID != "net-1" || req.Name != "eth1" || req.IPAddress != "10.0.0.10" || req.Status != "active" {
			t.Fatalf("unexpected interface update request: %#v", req)
		}
		_ = json.NewEncoder(w).Encode(coreVMInterface{
			ID: "iface-1", VMID: "vm-1", NetworkID: req.NetworkID, Name: req.Name, IPAddress: req.IPAddress, Status: req.Status,
		})
	}))
	defer server.Close()
	addCurrentTestCluster(t, server.URL)

	var output bytes.Buffer
	cmd := NewUpdateCommand()
	cmd.SetOut(&output)
	cmd.SetArgs([]string{"interface", "iface-1", "--vm", "vm-1", "--network", "net-1", "--name", "eth1", "--ip", "10.0.0.10", "--status", "active"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("update interface failed: %v", err)
	}

	for _, expected := range []string{"id: iface-1", "vm_id: vm-1", "network_id: net-1", "name: eth1", "status: active"} {
		if !strings.Contains(output.String(), expected) {
			t.Fatalf("expected update interface output to contain %q, got:\n%s", expected, output.String())
		}
	}
}

func TestUpdateInterfaceRequiresVM(t *testing.T) {
	cmd := NewUpdateCommand()
	cmd.SetArgs([]string{"interface", "iface-1", "--status", "active"})

	if err := cmd.Execute(); err == nil || !strings.Contains(err.Error(), "vm is required") {
		t.Fatalf("expected vm validation error, got %v", err)
	}
}

func TestUpdateInterfaceRequiresUpdateField(t *testing.T) {
	cmd := NewUpdateCommand()
	cmd.SetArgs([]string{"interface", "iface-1", "--vm", "vm-1"})

	if err := cmd.Execute(); err == nil || !strings.Contains(err.Error(), "at least one update field is required") {
		t.Fatalf("expected update field validation error, got %v", err)
	}
}

func TestUpdateRejectsUnsupportedResource(t *testing.T) {
	cmd := NewUpdateCommand()
	cmd.SetArgs([]string{"vm", "vm-1", "--status", "running"})

	if err := cmd.Execute(); err == nil || !strings.Contains(err.Error(), "unsupported resource") {
		t.Fatalf("expected unsupported resource error, got %v", err)
	}
}
