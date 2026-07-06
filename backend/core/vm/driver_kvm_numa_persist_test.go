package vm

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
)

// This file exercises CEILING 1: ConfigureNUMA persisting the topology into
// config.json (Config.Tags["numa.topology"]) so it survives a driver restart
// between ConfigureNUMA and Start. The pure test reloads the persisted config the
// way adopt/reload does (in-memory NUMA field intentionally nil) and inspects the
// launch args; the real-boot test additionally boots the reloaded VM under qemu.

// reloadVMFromConfig rebuilds a KVMVMInfo from a persisted config.json the way
// adoptRunningVMs does — Config is rehydrated from disk but the in-memory NUMA
// field is left nil (that is exactly the state a restart produces).
func reloadVMFromConfig(t *testing.T, vmID, vmDir string) *KVMVMInfo {
	t.Helper()
	info := &KVMVMInfo{
		ID: vmID, State: StateCreated,
		DiskPath:    filepath.Join(vmDir, "disk.qcow2"),
		ConfigPath:  filepath.Join(vmDir, "config.json"),
		MonitorPath: filepath.Join(vmDir, "monitor.sock"),
		VNCPort:     5900,
	}
	data, err := os.ReadFile(info.ConfigPath)
	if err != nil {
		t.Fatalf("read persisted config.json: %v", err)
	}
	if err := json.Unmarshal(data, &info.Config); err != nil {
		t.Fatalf("reload config: %v", err)
	}
	return info
}

// TestConfigureNUMAPersistsAcrossRestart proves CEILING 1 with no qemu: after
// ConfigureNUMA, a FRESH driver that reloads the VM from config.json (in-memory
// NUMA lost) still emits the 2-node -numa topology at launch.
//
// Discriminating by construction: the reloaded VM's in-memory NUMA field is nil
// (asserted), so if buildQEMUArgs read only that field (the pre-fix behaviour) it
// would emit zero -numa. Two -numa nodes appear only because effectiveNUMA falls
// back to the persisted Config.Tags["numa.topology"].
func TestConfigureNUMAPersistsAcrossRestart(t *testing.T) {
	ctx := context.Background()
	vmBase := t.TempDir()
	const vmID = "numa-persist-vm"
	vmDir := filepath.Join(vmBase, vmID)
	if err := os.MkdirAll(vmDir, 0755); err != nil {
		t.Fatalf("mkdir vmDir: %v", err)
	}

	// Driver 1: register a created VM (no qemu-img needed — we do not Create a real
	// disk) and configure NUMA. ConfigureNUMA must JSON-encode into config.json.
	d1 := &KVMDriverEnhanced{
		qemuBinaryPath: "/usr/bin/qemu-system-x86_64", // "pc" machine; never executed here
		vmBasePath:     vmBase,
		vms:            map[string]*KVMVMInfo{},
	}
	d1.vms[vmID] = &KVMVMInfo{
		ID: vmID, State: StateCreated,
		Config:      VMConfig{ID: vmID, Name: vmID, MemoryMB: 512, CPUShares: 2},
		DiskPath:    filepath.Join(vmDir, "disk.qcow2"),
		ConfigPath:  filepath.Join(vmDir, "config.json"),
		MonitorPath: filepath.Join(vmDir, "monitor.sock"),
		VNCPort:     5900,
	}

	topo := &NUMATopology{Mode: "strict", Nodes: []NUMANode{
		{ID: 0, CPUs: "0", MemoryMB: 256},
		{ID: 1, CPUs: "1", MemoryMB: 256},
	}}
	if err := d1.ConfigureNUMA(ctx, vmID, topo); err != nil {
		t.Fatalf("ConfigureNUMA: %v", err)
	}

	// The topology must now be on disk, not just in memory.
	data, err := os.ReadFile(filepath.Join(vmDir, "config.json"))
	if err != nil {
		t.Fatalf("read config.json after ConfigureNUMA: %v", err)
	}
	if !strings.Contains(string(data), numaTopologyTag) {
		t.Fatalf("config.json missing %q tag; topology was not persisted:\n%s", numaTopologyTag, data)
	}

	// Simulate a driver restart: fresh driver, reload the VM from config.json.
	d2 := &KVMDriverEnhanced{qemuBinaryPath: d1.qemuBinaryPath, vmBasePath: vmBase, vms: map[string]*KVMVMInfo{}}
	info2 := reloadVMFromConfig(t, vmID, vmDir)
	d2.vms[vmID] = info2

	// Discriminating precondition: the in-memory NUMA field was lost by the restart.
	if info2.NUMA != nil {
		t.Fatalf("reloaded VM unexpectedly has in-memory NUMA set; test would not discriminate")
	}

	args := d2.buildQEMUArgs(info2)
	if n := countArg(args, "-numa"); n != 2 { // countArg defined in driver_kvm_iothread_test.go
		t.Fatalf("restarted driver did not re-apply NUMA from config.json: got %d -numa nodes, want 2\nargs=%v", n, args)
	}
	t.Logf("PASS: 2-node NUMA topology survived a simulated driver restart via config.json and was re-emitted at launch")
}

// TestNUMAPersistRealBootAfterReload is the end-to-end variant: create a real VM,
// ConfigureNUMA, then boot it from a FRESH driver that reloaded the VM from
// config.json (in-memory NUMA lost), and assert (independent QMP `info numa`) the
// guest actually booted with the 2-node topology. SKIPS cleanly without qemu.
func TestNUMAPersistRealBootAfterReload(t *testing.T) {
	qemuBin, cirros := advOpsSkip(t) // defined in driver_kvm_advops_test.go

	base := t.TempDir()
	vmBase := filepath.Join(base, "vms")
	const vmID = "numa-persist-boot-vm"
	const memMB = 512

	// Driver 1: create a real VM and persist a 2-node NUMA topology.
	drv1, err := newKVMDriverEnhanced(qemuBin, vmBase)
	if err != nil {
		t.Skipf("skip: KVM driver init failed: %v", err)
	}
	d1 := drv1.(*KVMDriverEnhanced)
	ctx := context.Background()
	if _, err := d1.Create(ctx, VMConfig{
		ID: vmID, Name: vmID, Type: VMTypeKVM,
		MemoryMB: memMB, CPUShares: 2, Image: cirros, // 2 present vCPUs, one per node
	}); err != nil {
		t.Fatalf("create VM: %v", err)
	}
	topo := &NUMATopology{Mode: "strict", Nodes: []NUMANode{
		{ID: 0, CPUs: "0", MemoryMB: memMB / 2},
		{ID: 1, CPUs: "1", MemoryMB: memMB / 2},
	}}
	if err := d1.ConfigureNUMA(ctx, vmID, topo); err != nil {
		t.Fatalf("ConfigureNUMA: %v", err)
	}

	// Driver 2 = "after restart". Its adoptRunningVMs will NOT pick up this
	// created-but-not-running VM (no pidfile), so reload it from config.json the
	// way adopt does — Config rehydrated, in-memory NUMA nil.
	drv2, err := newKVMDriverEnhanced(qemuBin, vmBase)
	if err != nil {
		t.Skipf("skip: KVM driver re-init failed: %v", err)
	}
	d2 := drv2.(*KVMDriverEnhanced)
	info2 := reloadVMFromConfig(t, vmID, filepath.Join(vmBase, vmID))
	if info2.NUMA != nil {
		t.Fatalf("reloaded VM unexpectedly has in-memory NUMA set; test would not discriminate")
	}
	d2.vms[vmID] = info2

	if err := d2.Start(ctx, vmID); err != nil {
		t.Fatalf("start reloaded VM (invalid/absent -numa would boot 0 nodes or die): %v", err)
	}
	defer func() {
		_ = d2.Stop(context.Background(), vmID)
		if p := d2.vms[vmID]; p != nil && p.PID > 0 && syscall.Kill(p.PID, 0) == nil {
			_ = syscall.Kill(p.PID, syscall.SIGKILL)
		}
	}()

	sock := filepath.Join(vmBase, vmID, "qmp.sock")
	raw, err := advOpsQMP(t, sock, "human-monitor-command", map[string]interface{}{"command-line": "info numa"})
	if err != nil {
		t.Fatalf("info numa via QMP: %v", err)
	}
	var out string
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatalf("parse info numa %q: %v", string(raw), err)
	}
	if !strings.Contains(out, "2 nodes") || !strings.Contains(out, "node 1 cpus: 1") {
		t.Fatalf("reloaded VM did not boot with the persisted 2-node topology:\n%s", out)
	}
	t.Logf("PASS: VM booted from a fresh driver (in-memory NUMA lost) still got its 2-node topology from config.json:\n%s",
		strings.TrimSpace(out))
}
