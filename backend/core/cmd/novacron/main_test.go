package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/khryptorgraphics/novacron/backend/core/hypervisor"
	"github.com/khryptorgraphics/novacron/backend/core/vm"
)

func TestLoadConfigMissingUsesDefaults(t *testing.T) {
	t.Parallel()

	dataDir := t.TempDir()

	config, err := loadConfig(filepath.Join(dataDir, "missing.yaml"), "node-a", dataDir)
	if err != nil {
		t.Fatalf("loadConfig returned error for missing file: %v", err)
	}

	if got, want := config.Storage.BasePath, filepath.Join(dataDir, "storage"); got != want {
		t.Fatalf("storage base path = %q, want %q", got, want)
	}
	if got, want := config.Hypervisor.ID, "node-a"; got != want {
		t.Fatalf("hypervisor id = %q, want %q", got, want)
	}
	if got, want := config.Hypervisor.Name, "node-a"; got != want {
		t.Fatalf("hypervisor name = %q, want %q", got, want)
	}
	if got, want := config.Hypervisor.Role, hypervisor.RoleWorker; got != want {
		t.Fatalf("hypervisor role = %q, want %q", got, want)
	}
	if got, want := config.VMManager.DefaultDriver, vm.VMTypeKVM; got != want {
		t.Fatalf("vm manager default driver = %q, want %q", got, want)
	}
}

func TestLoadConfigRejectsInvalidYAML(t *testing.T) {
	t.Parallel()

	dataDir := t.TempDir()
	configPath := filepath.Join(dataDir, "config.yaml")
	if err := os.WriteFile(configPath, []byte("storage: ["), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	if _, err := loadConfig(configPath, "node-a", dataDir); err == nil {
		t.Fatal("loadConfig succeeded for invalid YAML, want error")
	}
}

func TestLoadConfigAppliesOverrides(t *testing.T) {
	t.Parallel()

	dataDir := t.TempDir()
	configPath := filepath.Join(dataDir, "config.yaml")
	configYAML := `
storage:
  base_path: /tmp/novacron-storage
hypervisor:
  name: edge-alpha
  role: master
vm_manager:
  default_driver: containerd
scheduler:
  minimum_node_count: 3
`
	if err := os.WriteFile(configPath, []byte(configYAML), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	config, err := loadConfig(configPath, "node-a", dataDir)
	if err != nil {
		t.Fatalf("loadConfig returned error: %v", err)
	}

	if got, want := config.Storage.BasePath, "/tmp/novacron-storage"; got != want {
		t.Fatalf("storage base path = %q, want %q", got, want)
	}
	if got, want := config.Hypervisor.Name, "edge-alpha"; got != want {
		t.Fatalf("hypervisor name = %q, want %q", got, want)
	}
	if got, want := config.Hypervisor.Role, hypervisor.RoleMaster; got != want {
		t.Fatalf("hypervisor role = %q, want %q", got, want)
	}
	if got, want := config.VMManager.DefaultDriver, vm.VMTypeContainerd; got != want {
		t.Fatalf("vm manager default driver = %q, want %q", got, want)
	}
	if got, want := config.Scheduler.MinimumNodeCount, 3; got != want {
		t.Fatalf("scheduler minimum node count = %d, want %d", got, want)
	}
}

func TestEnsureNonStubKVMRuntimeRejectsCoreStub(t *testing.T) {
	t.Parallel()

	manager, err := vm.NewVMManager(vm.DefaultVMManagerConfig())
	if err != nil {
		t.Fatalf("NewVMManager returned error: %v", err)
	}
	defer manager.Stop()

	err = ensureNonStubKVMRuntime(manager)
	if err == nil {
		t.Fatal("ensureNonStubKVMRuntime succeeded, want stub-driver error")
	}
	if !strings.Contains(err.Error(), "CoreStubDriver") {
		t.Fatalf("ensureNonStubKVMRuntime error = %q, want mention of CoreStubDriver", err)
	}
}
