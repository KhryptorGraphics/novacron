package commands

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/novacron/cli/pkg/api"
	"github.com/novacron/cli/pkg/service"
)

func TestParseVMManifestYAML(t *testing.T) {
	manifest := []byte(`
name: edge-vm
namespace: production
metadata:
  labels:
    app: fabric
spec:
  running: true
  template:
    spec:
      resources:
        cpu: 8
        memory: 32Gi
        disk: 200Gi
      image:
        source: ubuntu-24.04
      networks:
        - name: default
          type: bridge
          ipv4:
            method: dhcp
      userData: "#cloud-config"
`)

	vm, err := parseVMManifest(manifest)
	if err != nil {
		t.Fatalf("parseVMManifest returned error: %v", err)
	}

	if vm.Name != "edge-vm" || vm.Namespace != "production" {
		t.Fatalf("unexpected VM identity: %#v", vm)
	}
	if vm.Metadata.Labels["app"] != "fabric" {
		t.Fatalf("expected metadata labels to parse, got %#v", vm.Metadata.Labels)
	}
	spec := vm.Spec.Template.Spec
	if spec.Resources.CPU != 8 || spec.Resources.Memory != "32Gi" || spec.Image.Source != "ubuntu-24.04" {
		t.Fatalf("unexpected VM spec: %#v", spec)
	}
	if len(spec.Networks) != 1 || spec.Networks[0].IPv4 == nil || spec.Networks[0].IPv4.Method != "dhcp" {
		t.Fatalf("unexpected network config: %#v", spec.Networks)
	}
}

func TestParseVMManifestRejectsMissingName(t *testing.T) {
	_, err := parseVMManifest([]byte(`{"spec":{"running":true}}`))
	if err == nil || !strings.Contains(err.Error(), "name is required") {
		t.Fatalf("expected missing name error, got %v", err)
	}
}

func TestWaitForVMPollsUntilTargetPhase(t *testing.T) {
	originalInterval := waitPollInterval
	waitPollInterval = time.Millisecond
	defer func() { waitPollInterval = originalInterval }()

	var calls atomic.Int32
	vmService, closeServer := testVMService(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			t.Fatalf("unexpected method: %s", r.Method)
		}

		phase := "Pending"
		if calls.Add(1) >= 2 {
			phase = "Running"
		}
		_ = json.NewEncoder(w).Encode(api.VirtualMachine{
			Name: "vm-a",
			Status: api.VMStatus{
				Phase: phase,
			},
		})
	})
	defer closeServer()

	if err := waitForVM(vmService, "default", "vm-a", "Running", time.Second); err != nil {
		t.Fatalf("waitForVM returned error: %v", err)
	}
	if calls.Load() < 2 {
		t.Fatalf("expected polling until target phase, got %d calls", calls.Load())
	}
}

func TestWaitForVMDeletedTreatsNotFoundAsDeleted(t *testing.T) {
	originalInterval := waitPollInterval
	waitPollInterval = time.Millisecond
	defer func() { waitPollInterval = originalInterval }()

	var calls atomic.Int32
	vmService, closeServer := testVMService(t, func(w http.ResponseWriter, r *http.Request) {
		if calls.Add(1) == 1 {
			_ = json.NewEncoder(w).Encode(api.VirtualMachine{Name: "vm-a"})
			return
		}

		w.WriteHeader(http.StatusNotFound)
		_ = json.NewEncoder(w).Encode(api.ErrorResponse{
			Code:    "not_found",
			Message: "VM not found",
		})
	})
	defer closeServer()

	if err := waitForVMDeleted(vmService, "default", "vm-a", time.Second); err != nil {
		t.Fatalf("waitForVMDeleted returned error: %v", err)
	}
	if calls.Load() < 2 {
		t.Fatalf("expected polling until API reports not found, got %d calls", calls.Load())
	}
}

func TestPrintVMTableIncludesOptionalColumns(t *testing.T) {
	now := time.Now().Add(-2 * time.Minute)
	output := captureStdout(t, func() {
		err := printVMTable([]api.VirtualMachine{
			{
				Name:      "vm-a",
				Namespace: "prod",
				Status: api.VMStatus{
					Phase:       "Running",
					NodeName:    "node-1",
					IPAddresses: []string{"10.0.0.5"},
				},
				CreatedAt: now,
			},
		}, true, true)
		if err != nil {
			t.Fatalf("printVMTable returned error: %v", err)
		}
	})

	for _, expected := range []string{"NAME", "NAMESPACE", "STATUS", "NODE", "IP", "vm-a", "prod", "Running", "node-1", "10.0.0.5"} {
		if !strings.Contains(output, expected) {
			t.Fatalf("expected table output to contain %q, got:\n%s", expected, output)
		}
	}
}

func testVMService(t *testing.T, handler http.HandlerFunc) (*service.VMService, func()) {
	t.Helper()

	server := httptest.NewServer(handler)
	client, err := api.NewClient(server.URL)
	if err != nil {
		server.Close()
		t.Fatalf("failed to create API client: %v", err)
	}

	return service.NewVMService(client), server.Close
}

func captureStdout(t *testing.T, fn func()) string {
	t.Helper()

	oldStdout := os.Stdout
	reader, writer, err := os.Pipe()
	if err != nil {
		t.Fatalf("failed to create pipe: %v", err)
	}
	os.Stdout = writer

	fn()

	if err := writer.Close(); err != nil {
		t.Fatalf("failed to close writer: %v", err)
	}
	os.Stdout = oldStdout

	data, err := io.ReadAll(reader)
	if err != nil {
		t.Fatalf("failed to read stdout: %v", err)
	}
	return string(data)
}

func TestWaitForVMTimeoutIncludesLastPhase(t *testing.T) {
	originalInterval := waitPollInterval
	waitPollInterval = time.Millisecond
	defer func() { waitPollInterval = originalInterval }()

	vmService, closeServer := testVMService(t, func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(api.VirtualMachine{
			Name:   "vm-a",
			Status: api.VMStatus{Phase: "Pending"},
		})
	})
	defer closeServer()

	err := waitForVM(vmService, "default", "vm-a", "Running", 5*time.Millisecond)
	if err == nil || !strings.Contains(err.Error(), "last phase Pending") {
		t.Fatalf("expected timeout with last phase, got %v", err)
	}
}

func TestWaitForVMMigrationPollsUntilTargetNode(t *testing.T) {
	originalInterval := waitPollInterval
	waitPollInterval = time.Millisecond
	defer func() { waitPollInterval = originalInterval }()

	var calls atomic.Int32
	vmService, closeServer := testVMService(t, func(w http.ResponseWriter, r *http.Request) {
		state := "running"
		nodeName := "node-a"
		if calls.Add(1) >= 2 {
			state = "completed"
			nodeName = "node-b"
		}
		_ = json.NewEncoder(w).Encode(api.VirtualMachine{
			Name: "vm-a",
			Status: api.VMStatus{
				Phase:    "Running",
				NodeName: nodeName,
				Migration: &api.MigrationStatus{
					State:      state,
					TargetNode: "node-b",
					Progress:   100,
				},
			},
		})
	})
	defer closeServer()

	if err := waitForVMMigration(vmService, "default", "vm-a", "node-b", time.Second); err != nil {
		t.Fatalf("waitForVMMigration returned error: %v", err)
	}
	if calls.Load() < 2 {
		t.Fatalf("expected polling until migration completion, got %d calls", calls.Load())
	}
}

func TestWaitForVMMigrationFailsOnTerminalFailure(t *testing.T) {
	originalInterval := waitPollInterval
	waitPollInterval = time.Millisecond
	defer func() { waitPollInterval = originalInterval }()

	vmService, closeServer := testVMService(t, func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(api.VirtualMachine{
			Name: "vm-a",
			Status: api.VMStatus{
				NodeName: "node-a",
				Migration: &api.MigrationStatus{
					State:      "failed",
					TargetNode: "node-b",
					Progress:   60,
				},
			},
		})
	})
	defer closeServer()

	err := waitForVMMigration(vmService, "default", "vm-a", "node-b", time.Second)
	if err == nil || !strings.Contains(err.Error(), "migration failed") {
		t.Fatalf("expected migration failure, got %v", err)
	}
}
