package main

import (
	"os"
	"os/exec"
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
  tenant_quota:
    default:
      max_vms: 3
      max_cpu_units: 8
      max_memory_mb: 4096
    overrides:
      tenant-blue:
        max_vms: 2
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
	if got, want := config.VMManager.TenantQuota.Default.MaxVMs, 3; got != want {
		t.Fatalf("vm manager default tenant max_vms = %d, want %d", got, want)
	}
	if got, want := config.VMManager.TenantQuota.Default.MaxCPUUnits, 8; got != want {
		t.Fatalf("vm manager default tenant max_cpu_units = %d, want %d", got, want)
	}
	if got, want := config.VMManager.TenantQuota.Default.MaxMemoryMB, int64(4096); got != want {
		t.Fatalf("vm manager default tenant max_memory_mb = %d, want %d", got, want)
	}
	if got, want := config.VMManager.TenantQuota.Overrides["tenant-blue"].MaxVMs, 2; got != want {
		t.Fatalf("tenant-blue max_vms = %d, want %d", got, want)
	}
	if got, want := config.Scheduler.MinimumNodeCount, 3; got != want {
		t.Fatalf("scheduler minimum node count = %d, want %d", got, want)
	}
}

func TestEnsureNonStubKVMRuntimeUsesRealKVMWhenAvailable(t *testing.T) {
	t.Parallel()

	manager, err := vm.NewVMManager(newTestVMManagerConfig(t, "qemu-system-x86_64"))
	if err != nil {
		t.Fatalf("NewVMManager returned error: %v", err)
	}
	defer manager.Stop()

	if err := ensureNonStubKVMRuntime(manager); err != nil {
		t.Fatalf("ensureNonStubKVMRuntime returned error: %v", err)
	}
}

func TestEnsureNonStubKVMRuntimeRejectsExplicitStubFallback(t *testing.T) {
	t.Setenv("NOVACRON_ALLOW_STUB_KVM", "1")

	manager, err := vm.NewVMManager(newTestVMManagerConfig(t, "missing-qemu-for-stub-test"))
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

func TestRegisterLocalSchedulerNodeRegistersInventory(t *testing.T) {
	t.Parallel()

	manager, err := vm.NewVMManager(newTestVMManagerConfig(t, "qemu-system-x86_64"))
	if err != nil {
		t.Fatalf("NewVMManager returned error: %v", err)
	}
	defer manager.Stop()

	if err := registerLocalSchedulerNode(manager, "test-node", t.TempDir()); err != nil {
		t.Fatalf("registerLocalSchedulerNode returned error: %v", err)
	}

	nodes := manager.ListSchedulerNodes()
	if len(nodes) != 1 {
		t.Fatalf("expected one scheduler node, got %d", len(nodes))
	}

	node := nodes[0]
	if node.NodeID != "test-node" {
		t.Fatalf("scheduler node id = %q, want %q", node.NodeID, "test-node")
	}
	if node.Status != "available" {
		t.Fatalf("scheduler node status = %q, want %q", node.Status, "available")
	}
	if node.TotalCPU < 1 {
		t.Fatalf("scheduler total cpu = %d, want >= 1", node.TotalCPU)
	}
	if node.TotalMemoryMB < 1 {
		t.Fatalf("scheduler total memory = %d, want >= 1", node.TotalMemoryMB)
	}
	if node.TotalDiskGB < 1 {
		t.Fatalf("scheduler total disk = %d, want >= 1", node.TotalDiskGB)
	}

	if err := registerLocalSchedulerNode(manager, "test-node", t.TempDir()); err != nil {
		t.Fatalf("registerLocalSchedulerNode update returned error: %v", err)
	}
	if len(manager.ListSchedulerNodes()) != 1 {
		t.Fatalf("expected scheduler node update to keep a single entry, got %d", len(manager.ListSchedulerNodes()))
	}
}

func newTestVMManagerConfig(t *testing.T, qemuPath string) vm.VMManagerConfig {
	t.Helper()

	if qemuPath != "" {
		if qemuPath == "qemu-system-x86_64" {
			if _, err := exec.LookPath(qemuPath); err != nil {
				t.Skipf("qemu-system-x86_64 not available in PATH: %v", err)
			}
		}
	}

	config := vm.DefaultVMManagerConfig()
	config.Drivers[vm.VMTypeKVM] = vm.VMDriverConfigManager{
		Enabled: true,
		Config: map[string]interface{}{
			"node_id":   "test-node",
			"qemu_path": qemuPath,
			"vm_path":   filepath.Join(t.TempDir(), "vms"),
		},
	}

	return config
}
