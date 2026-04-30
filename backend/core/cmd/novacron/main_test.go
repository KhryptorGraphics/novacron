package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
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

func TestLoadConfigSupportsSharedRuntimeManifest(t *testing.T) {
	t.Parallel()

	configPath := filepath.Join(t.TempDir(), "config.yaml")
	manifestYAML := `
system:
  node_id: manifest-node
  data_dir: /tmp/manifest-data
runtime:
  version: v1alpha1
  auth_mode: disabled
  enabled_services:
    - api
    - vm
    - scheduler
    - storage
    - network
    - hypervisor
  discovery_seeds:
    - id: seed-a
      address: 10.0.0.20:8090
      tags:
        - trusted
monitoring:
  enable_metrics: true
security:
  enable_auth: true
  enable_encryption: true
`
	if err := os.WriteFile(configPath, []byte(manifestYAML), 0o644); err != nil {
		t.Fatalf("write manifest: %v", err)
	}

	config, err := loadConfig(configPath, "fallback-node", "/tmp/fallback-data")
	if err != nil {
		t.Fatalf("loadConfig returned error: %v", err)
	}

	if got, want := config.Hypervisor.ID, "manifest-node"; got != want {
		t.Fatalf("hypervisor id = %q, want %q", got, want)
	}
	if got, want := config.Storage.BasePath, "/tmp/manifest-data/storage"; got != want {
		t.Fatalf("storage base path = %q, want %q", got, want)
	}
	if config.Auth.Enabled {
		t.Fatal("expected runtime auth to be disabled when auth_mode is disabled")
	}
	if !config.Storage.Encryption {
		t.Fatal("expected storage encryption to follow manifest security settings")
	}
	if !config.Network.BandwidthMonitoringEnabled {
		t.Fatal("expected network bandwidth monitoring to follow manifest monitoring settings")
	}
	if got, want := config.Services.Version, "v1alpha1"; got != want {
		t.Fatalf("services manifest version = %q, want %q", got, want)
	}
	if got, want := config.Services.AuthMode, "disabled"; got != want {
		t.Fatalf("services auth mode = %q, want %q", got, want)
	}
	if !serviceEnabled(config.Services.EnabledServices, "scheduler") {
		t.Fatalf("expected services enabled list to contain scheduler, got %#v", config.Services.EnabledServices)
	}
	if len(config.Services.DiscoverySeeds) != 1 {
		t.Fatalf("expected one discovery seed, got %#v", config.Services.DiscoverySeeds)
	}
	if got, want := config.Services.DiscoverySeeds[0].Address, "10.0.0.20:8090"; got != want {
		t.Fatalf("discovery seed address = %q, want %q", got, want)
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
auth:
  enabled: true
  frontend_url: http://localhost:3000
  default_tenant_id: tenant-auth
  default_cluster_id: cluster-auth
  oauth:
    github:
      client_id: github-client
      client_secret: github-secret
      redirect_url: http://localhost:8090/api/auth/oauth/github/callback
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
	if got, want := config.Auth.Enabled, true; got != want {
		t.Fatalf("auth enabled = %t, want %t", got, want)
	}
	if got, want := config.Auth.FrontendURL, "http://localhost:3000"; got != want {
		t.Fatalf("auth frontend url = %q, want %q", got, want)
	}
	if got, want := config.Auth.DefaultTenantID, "tenant-auth"; got != want {
		t.Fatalf("auth default tenant id = %q, want %q", got, want)
	}
	if got, want := config.Auth.DefaultClusterID, "cluster-auth"; got != want {
		t.Fatalf("auth default cluster id = %q, want %q", got, want)
	}
	if got, want := config.Auth.OAuth.GitHub.ClientID, "github-client"; got != want {
		t.Fatalf("auth github client id = %q, want %q", got, want)
	}
	if got, want := config.Auth.OAuth.GitHub.ClientSecret, "github-secret"; got != want {
		t.Fatalf("auth github client secret = %q, want %q", got, want)
	}
	if got, want := config.Auth.OAuth.GitHub.RedirectURL, "http://localhost:8090/api/auth/oauth/github/callback"; got != want {
		t.Fatalf("auth github redirect url = %q, want %q", got, want)
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

func TestInitializeAPIReportsDisabledServicesWhenVMServiceIsOff(t *testing.T) {
	t.Parallel()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	config := runtimeConfig{
		Auth: runtimeAuthConfig{Enabled: false},
		Services: runtimeManifestSummary{
			Version:           "v1alpha1",
			DeploymentProfile: "single-node",
			DiscoveryMode:     "disabled",
			FederationMode:    "disabled",
			MigrationMode:     "disabled",
			AuthMode:          "disabled",
			EnabledServices:   []string{"api"},
		},
	}

	apiServer, err := initializeAPI(ctx, config, "127.0.0.1:0", nil, nil, nil, nil, nil, nil)
	if err != nil {
		t.Fatalf("initializeAPI returned error: %v", err)
	}
	defer apiServer.Shutdown(context.Background())

	report := getJSONResponse[runtimeServiceReport](t, apiServer, "/internal/runtime/v1/services")
	if got, want := report.Status, "healthy"; got != want {
		t.Fatalf("service report status = %q, want %q", got, want)
	}
	if got, want := report.Manifest.Version, "v1alpha1"; got != want {
		t.Fatalf("manifest version = %q, want %q", got, want)
	}
	if got, want := report.Manifest.DeploymentProfile, "single-node"; got != want {
		t.Fatalf("deployment profile = %q, want %q", got, want)
	}

	serviceStates := make(map[string]runtimeServiceStatus, len(report.Services))
	for _, service := range report.Services {
		serviceStates[service.Name] = service
	}

	if got := serviceStates["api"].State; got != runtimeServiceStateRunning {
		t.Fatalf("api state = %q, want %q", got, runtimeServiceStateRunning)
	}
	if got := serviceStates["vm"].State; got != runtimeServiceStateDisabled {
		t.Fatalf("vm state = %q, want %q", got, runtimeServiceStateDisabled)
	}
	if got := serviceStates["vm"].Reason; got != "disabled by runtime manifest" {
		t.Fatalf("vm reason = %q, want disabled by runtime manifest", got)
	}
	if got := serviceStates["auth"].State; got != runtimeServiceStateDisabled {
		t.Fatalf("auth state = %q, want %q", got, runtimeServiceStateDisabled)
	}
	if got := serviceStates["network"].State; got != runtimeServiceStateDisabled {
		t.Fatalf("network state = %q, want %q", got, runtimeServiceStateDisabled)
	}
	if got := serviceStates["scheduler"].State; got != runtimeServiceStateDisabled {
		t.Fatalf("scheduler state = %q, want %q", got, runtimeServiceStateDisabled)
	}
	if len(report.DisabledServices) == 0 {
		t.Fatal("expected disabled services to be reported")
	}

	nodes := getJSONResponse[[]runtimeNode](t, apiServer, "/api/cluster/nodes")
	if len(nodes) != 0 {
		t.Fatalf("expected no cluster nodes when vm service is disabled, got %d", len(nodes))
	}
}

func TestInitializeAPIExposesClusterLocalEndpoints(t *testing.T) {
	t.Parallel()

	manager, err := vm.NewVMManager(newTestVMManagerConfig(t, "qemu-system-x86_64"))
	if err != nil {
		t.Fatalf("NewVMManager returned error: %v", err)
	}
	defer manager.Stop()

	if err := registerLocalSchedulerNode(manager, "test-node", t.TempDir()); err != nil {
		t.Fatalf("registerLocalSchedulerNode returned error: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	apiServer, err := initializeAPI(ctx, runtimeConfig{}, "127.0.0.1:0", manager, nil, nil, nil, nil, nil)
	if err != nil {
		t.Fatalf("initializeAPI returned error: %v", err)
	}
	defer apiServer.Shutdown(context.Background())

	healthzResponse := getJSONResponse[map[string]string](t, apiServer, "/healthz")
	if got, want := healthzResponse["status"], "ok"; got != want {
		t.Fatalf("healthz status = %q, want %q", got, want)
	}

	nodes := getJSONResponse[[]runtimeNode](t, apiServer, "/api/cluster/nodes")
	if len(nodes) != 1 {
		t.Fatalf("expected one cluster node, got %d", len(nodes))
	}

	node := nodes[0]
	if got, want := node.ID, "test-node"; got != want {
		t.Fatalf("node id = %q, want %q", got, want)
	}
	if !node.Schedulable {
		t.Fatal("expected cluster node to be schedulable")
	}
	if node.CPU < 1 {
		t.Fatalf("node cpu = %d, want >= 1", node.CPU)
	}

	selectedNode := getJSONResponse[runtimeNode](t, apiServer, "/api/cluster/nodes/test-node")
	if got, want := selectedNode.ID, "test-node"; got != want {
		t.Fatalf("selected node id = %q, want %q", got, want)
	}

	health := getJSONResponse[runtimeClusterHealth](t, apiServer, "/api/cluster/health")
	if got, want := health.Status, "healthy"; got != want {
		t.Fatalf("cluster health status = %q, want %q", got, want)
	}
	if got, want := health.TotalNodes, 1; got != want {
		t.Fatalf("cluster total nodes = %d, want %d", got, want)
	}
	if got, want := health.Leader, "test-node"; got != want {
		t.Fatalf("cluster leader = %q, want %q", got, want)
	}

	leader := getJSONResponse[map[string]string](t, apiServer, "/api/cluster/leader")
	if got, want := leader["id"], "test-node"; got != want {
		t.Fatalf("leader id = %q, want %q", got, want)
	}
	if got, want := leader["scope"], "cluster-local"; got != want {
		t.Fatalf("leader scope = %q, want %q", got, want)
	}
}

func TestInitializeAPIExposesInternalRuntimeMonitoringMetrics(t *testing.T) {
	t.Parallel()

	manager, err := vm.NewVMManager(newTestVMManagerConfig(t, "qemu-system-x86_64"))
	if err != nil {
		t.Fatalf("NewVMManager returned error: %v", err)
	}
	defer manager.Stop()

	if err := registerLocalSchedulerNode(manager, "test-node", t.TempDir()); err != nil {
		t.Fatalf("registerLocalSchedulerNode returned error: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	apiServer, err := initializeAPI(ctx, runtimeConfig{}, "127.0.0.1:0", manager, nil, nil, nil, nil, nil)
	if err != nil {
		t.Fatalf("initializeAPI returned error: %v", err)
	}
	defer apiServer.Shutdown(context.Background())

	url := fmt.Sprintf("http://%s/internal/runtime/v1/monitoring/metrics", apiServer.address)
	response, err := http.Get(url)
	if err != nil {
		t.Fatalf("GET %s returned error: %v", url, err)
	}
	defer response.Body.Close()

	if got, want := response.StatusCode, http.StatusOK; got != want {
		body, _ := io.ReadAll(response.Body)
		t.Fatalf("GET %s status = %d, want %d: %s", url, got, want, strings.TrimSpace(string(body)))
	}

	body, err := io.ReadAll(response.Body)
	if err != nil {
		t.Fatalf("read %s response: %v", url, err)
	}

	var payload map[string]json.RawMessage
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatalf("decode %s payload map: %v", url, err)
	}

	wantKeys := []string{
		"cpuAnalysis",
		"cpuChangePercentage",
		"currentCpuUsage",
		"currentDiskUsage",
		"currentMemoryUsage",
		"currentNetworkUsage",
		"diskChangePercentage",
		"memoryAnalysis",
		"memoryChangePercentage",
		"networkChangePercentage",
		"timeLabels",
	}
	gotKeys := make([]string, 0, len(payload))
	for key := range payload {
		gotKeys = append(gotKeys, key)
	}
	sort.Strings(gotKeys)
	if len(gotKeys) != len(wantKeys) {
		t.Fatalf("monitoring payload key count = %d, want %d (%v)", len(gotKeys), len(wantKeys), gotKeys)
	}
	for i := range wantKeys {
		if gotKeys[i] != wantKeys[i] {
			t.Fatalf("monitoring payload keys = %v, want %v", gotKeys, wantKeys)
		}
	}

	var metrics runtimeMonitoringSummary
	if err := json.Unmarshal(body, &metrics); err != nil {
		t.Fatalf("decode %s metrics response: %v", url, err)
	}

	if metrics.CurrentMemoryUsage <= 0 {
		t.Fatalf("currentMemoryUsage = %f, want > 0 from registered scheduler node", metrics.CurrentMemoryUsage)
	}
	if metrics.CurrentNetworkUsage != 0 {
		t.Fatalf("currentNetworkUsage = %f, want deterministic unavailable fallback of 0", metrics.CurrentNetworkUsage)
	}
	if metrics.NetworkChangePercentage != 0 {
		t.Fatalf("networkChangePercentage = %f, want deterministic unavailable fallback of 0", metrics.NetworkChangePercentage)
	}
	if len(metrics.TimeLabels) != 5 {
		t.Fatalf("timeLabels length = %d, want 5", len(metrics.TimeLabels))
	}
	if metrics.CpuAnalysis == "" {
		t.Fatal("cpuAnalysis is empty, want deterministic analysis text")
	}
	if metrics.MemoryAnalysis == "" {
		t.Fatal("memoryAnalysis is empty, want deterministic analysis text")
	}
}

func getJSONResponse[T any](t *testing.T, apiServer *APIServer, path string) T {
	t.Helper()

	var zero T
	url := fmt.Sprintf("http://%s%s", apiServer.address, path)
	response, err := http.Get(url)
	if err != nil {
		t.Fatalf("GET %s returned error: %v", url, err)
	}
	defer response.Body.Close()

	if response.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(response.Body)
		t.Fatalf("GET %s status = %d, want %d: %s", url, response.StatusCode, http.StatusOK, strings.TrimSpace(string(body)))
	}

	if err := json.NewDecoder(response.Body).Decode(&zero); err != nil {
		t.Fatalf("decode %s response: %v", url, err)
	}

	return zero
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
