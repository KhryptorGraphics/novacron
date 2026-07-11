package vm

import (
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"testing"
	"time"
)

// This file exercises the advanced VM ops added to the KVM driver: CPU hotplug,
// memory hotplug and NUMA. Each test boots a REAL qemu guest (TCG on this arm64
// box, no /dev/kvm needed) and asserts the effect over an INDEPENDENT QMP
// connection, so the assertion does not depend on the code under test. All tests
// SKIP cleanly (never fail) when qemu / qemu-img / a cirros image are absent,
// matching the sibling migration / cpu-pinning / hotplug tests.

// advOpsSkip runs the shared skip-guards and returns the qemu binary + cirros
// image path, or skips the test.
func advOpsSkip(t *testing.T) (string, string) {
	t.Helper()
	qemuBin := defaultQEMUBinary("") // arch-aware: qemu-system-x86_64 on amd64, -aarch64 on arm64
	if _, err := exec.LookPath(qemuBin); err != nil {
		t.Skipf("skip: %s not installed", qemuBin)
	}
	if _, err := exec.LookPath("qemu-img"); err != nil {
		t.Skip("skip: qemu-img not installed")
	}
	cirros := findCirrosImage() // arch-aware, defined in driver_kvm_migrate_test.go
	if cirros == "" {
		t.Skip("skip: cirros image not found in known locations")
	}
	return qemuBin, cirros
}

// advOpsQMP opens a fresh QMP connection (independent of the driver's own) and
// runs one command, returning its return payload or error.
func advOpsQMP(t *testing.T, sock, cmd string, args map[string]interface{}) (json.RawMessage, error) {
	t.Helper()
	q, err := qmpDial(sock, 10*time.Second)
	if err != nil {
		t.Fatalf("dial QMP %s: %v", sock, err)
	}
	defer q.Close()
	return q.execute(cmd, args)
}

// advOpsCPUCount returns the guest's present vCPU count from query-cpus-fast.
func advOpsCPUCount(t *testing.T, sock string) int {
	t.Helper()
	raw, err := advOpsQMP(t, sock, "query-cpus-fast", nil)
	if err != nil {
		t.Fatalf("query-cpus-fast: %v", err)
	}
	var cpus []json.RawMessage
	if err := json.Unmarshal(raw, &cpus); err != nil {
		t.Fatalf("parse query-cpus-fast %q: %v", string(raw), err)
	}
	return len(cpus)
}

// advOpsMemDeviceIDs returns the set of memory-device ids from query-memory-devices.
func advOpsMemDeviceIDs(t *testing.T, sock string) map[string]bool {
	t.Helper()
	raw, err := advOpsQMP(t, sock, "query-memory-devices", nil)
	if err != nil {
		t.Fatalf("query-memory-devices: %v", err)
	}
	var devs []struct {
		Data struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.Unmarshal(raw, &devs); err != nil {
		t.Fatalf("parse query-memory-devices %q: %v", string(raw), err)
	}
	out := map[string]bool{}
	for _, dv := range devs {
		if dv.Data.ID != "" {
			out[dv.Data.ID] = true
		}
	}
	return out
}

// advOpsHasAArch64UEFI reports whether the aarch64 UEFI code image (needed for
// the virt machine's acpi-ged, i.e. memory hotplug) is installed.
func advOpsHasAArch64UEFI() bool {
	for _, p := range edk2CodePaths { // package var in driver_kvm_enhanced.go
		if _, err := os.Stat(p); err == nil {
			return true
		}
	}
	return false
}

// argValue returns the token following flag in an arg list, or "" if absent.
func argValue(args []string, flag string) string {
	for i := 0; i < len(args)-1; i++ {
		if args[i] == flag {
			return args[i+1]
		}
	}
	return ""
}

// TestBuildQEMUArgsHotplugOptIn proves the CRITICAL SAFETY property with no qemu:
// a default VM (no headroom tags, no NUMA) emits the SAME -smp/-m as before and
// no -numa, while the opt-in tags/topology are the ONLY thing that changes them.
// This is what keeps the Gate-1 boot / Gate-2 migration args byte-for-byte stable.
func TestBuildQEMUArgsHotplugOptIn(t *testing.T) {
	// x86 machine ("pc") avoids the aarch64 UEFI/pcie side effects; the -smp/-m
	// opt-in logic is arch-independent. buildQEMUArgs execs nothing, just builds args.
	d := &KVMDriverEnhanced{qemuBinaryPath: "/usr/bin/qemu-system-x86_64", vmBasePath: t.TempDir()}
	tmp := t.TempDir()
	mk := func(cfg VMConfig, numa *NUMATopology) []string {
		return d.buildQEMUArgs(&KVMVMInfo{
			ID: "u", DiskPath: filepath.Join(tmp, "disk.qcow2"),
			MonitorPath: filepath.Join(tmp, "mon.sock"), VNCPort: 5900,
			Config: cfg, NUMA: numa,
		})
	}

	// 1. Default VM: plain counts, no headroom syntax, no NUMA.
	def := mk(VMConfig{MemoryMB: 512, CPUShares: 2}, nil)
	if got := argValue(def, "-smp"); got != "2" {
		t.Fatalf("default -smp changed: got %q, want %q", got, "2")
	}
	if got := argValue(def, "-m"); got != "512" {
		t.Fatalf("default -m changed: got %q, want %q", got, "512")
	}
	for _, a := range def {
		if strings.Contains(a, "maxcpus") || strings.Contains(a, "maxmem") || strings.Contains(a, "slots=") {
			t.Fatalf("default VM unexpectedly got hotplug headroom in %q", a)
		}
		if a == "-numa" {
			t.Fatalf("default VM unexpectedly got -numa")
		}
	}

	// 2. Opt-in headroom via the TYPED Config fields flips ONLY -smp/-m.
	hp := mk(VMConfig{MemoryMB: 512, CPUShares: 1, MaxVCPUs: 4, MaxMemoryMB: 2048, MemSlots: 2}, nil)
	if got := argValue(hp, "-smp"); got != "1,maxcpus=4" {
		t.Fatalf("opt-in -smp: got %q, want %q", got, "1,maxcpus=4")
	}
	if got := argValue(hp, "-m"); got != "512,slots=2,maxmem=2048M" {
		t.Fatalf("opt-in -m: got %q, want %q", got, "512,slots=2,maxmem=2048M")
	}

	// 3. A MaxVCPUs <= cpus (no real headroom) must NOT alter -smp.
	noop := mk(VMConfig{MemoryMB: 512, CPUShares: 2, MaxVCPUs: 2}, nil)
	if got := argValue(noop, "-smp"); got != "2" {
		t.Fatalf("MaxVCPUs<=cpus should not change -smp: got %q", got)
	}

	// 4. NUMA topology via the TYPED Config.NUMA field adds -numa/-object.
	nu := mk(VMConfig{MemoryMB: 512, CPUShares: 2, NUMA: &NUMATopology{Nodes: []NUMANode{
		{ID: 0, CPUs: "0", MemoryMB: 256}, {ID: 1, CPUs: "1", MemoryMB: 256},
	}}}, nil)
	if n := strings.Count(strings.Join(nu, " "), "-numa "); n != 2 {
		t.Fatalf("expected 2 -numa nodes from Config.NUMA, got %d in %v", n, nu)
	}

	// 5. Back-compat: the deprecated string tags still work when the typed field
	// is unset (a caller this driver does not own — e.g. driver_kvm_iothread_test —
	// still sets Tags["iothreads"], so the fallback must stay).
	tagOnly := mk(VMConfig{MemoryMB: 512, CPUShares: 1, Tags: map[string]string{
		"hotplug.maxvcpus": "4", "hotplug.maxmem_mb": "2048", "hotplug.mem_slots": "2",
	}}, nil)
	if got := argValue(tagOnly, "-smp"); got != "1,maxcpus=4" {
		t.Fatalf("tag fallback -smp: got %q, want %q", got, "1,maxcpus=4")
	}
	if got := argValue(tagOnly, "-m"); got != "512,slots=2,maxmem=2048M" {
		t.Fatalf("tag fallback -m: got %q, want %q", got, "512,slots=2,maxmem=2048M")
	}

	// 6. The typed field takes PRECEDENCE over a conflicting legacy tag.
	both := mk(VMConfig{MemoryMB: 512, CPUShares: 1, MaxVCPUs: 8,
		Tags: map[string]string{"hotplug.maxvcpus": "4"}}, nil)
	if got := argValue(both, "-smp"); got != "1,maxcpus=8" {
		t.Fatalf("typed field should win over tag: got %q, want %q", got, "1,maxcpus=8")
	}

	t.Logf("PASS: default -smp/-m unchanged; typed headroom+NUMA opt-in, tags still a fallback, typed wins over tag")
}

// TestCPUHotplugRealQMP boots a VM with opt-in cpu-hotplug headroom (1 present
// vCPU, maxcpus=2 via the "hotplug.maxvcpus" tag), hot-plugs one vCPU, and
// asserts (independent QMP) that query-cpus-fast reports one MORE present vCPU.
//
// Discriminating by construction: query-cpus-fast lists only present vCPUs, so a
// bare object without the device_add in hotPlugCPU would leave the count at 1 --
// only device_add makes the new vCPU present.
//
// NOTE: some machines cannot hot-plug vCPUs at all -- notably aarch64 "virt" on
// QEMU 8.2, where query-hotpluggable-cpus errors "machine does not support
// hot-plugging CPUs". This test detects that and SKIPS cleanly; the device_add
// path is exercised where the machine supports it (x86 "pc"/"q35").
func TestCPUHotplugRealQMP(t *testing.T) {
	qemuBin, cirros := advOpsSkip(t)

	base := t.TempDir()
	vmBase := filepath.Join(base, "vms")
	drv, err := newKVMDriverEnhanced(qemuBin, vmBase)
	if err != nil {
		t.Skipf("skip: KVM driver init failed: %v", err)
	}
	d := drv.(*KVMDriverEnhanced)
	ctx := context.Background()
	const vmID = "cpuhotplug-vm"

	if _, err := d.Create(ctx, VMConfig{
		ID: vmID, Name: vmID, Type: VMTypeKVM,
		MemoryMB: 512, CPUShares: 1, Image: cirros,
		MaxVCPUs: 2, // opt-in headroom (typed field)
	}); err != nil {
		t.Fatalf("create VM: %v", err)
	}
	if err := d.Start(ctx, vmID); err != nil {
		t.Fatalf("start VM: %v", err)
	}
	defer func() {
		_ = d.Stop(context.Background(), vmID)
		// Locked read: raw d.vms[vmID] access here would race monitorVM's
		// locked PID/State writes on process exit (novacron -race gate).
		if info, err := d.GetInfo(context.Background(), vmID); err == nil && info.PID > 0 && syscall.Kill(info.PID, 0) == nil {
			_ = syscall.Kill(info.PID, syscall.SIGKILL)
		}
	}()

	sock := filepath.Join(vmBase, vmID, "qmp.sock")

	// Skip cleanly on machines that cannot hot-plug vCPUs (e.g. aarch64 "virt").
	if _, err := advOpsQMP(t, sock, "query-hotpluggable-cpus", nil); err != nil {
		t.Skipf("skip: machine does not support vCPU hot-plug: %v", err)
	}

	if n := advOpsCPUCount(t, sock); n != 1 {
		t.Fatalf("precondition: expected 1 present vCPU before hot-plug, got %d", n)
	}

	if err := d.HotPlugDevice(ctx, vmID, &DeviceConfig{Type: "cpu", Name: "cpu-hot-1"}); err != nil {
		t.Fatalf("HotPlugDevice cpu: %v", err)
	}

	if n := advOpsCPUCount(t, sock); n != 2 {
		t.Fatalf("expected 2 present vCPUs after hot-plug, got %d", n)
	}
	t.Logf("PASS: present vCPU count increased 1 -> 2 after HotPlugDevice cpu")
}

// TestMemHotplugRealQMP boots a VM with opt-in memory-hotplug headroom
// ("hotplug.maxmem_mb"/"hotplug.mem_slots" tags -> -m 512,slots=2,maxmem=1024M),
// hot-plugs a 128MiB DIMM, and asserts (independent QMP) that query-memory-devices
// lists it.
//
// Discriminating by construction: query-memory-devices is empty until a pc-dimm
// device_add lands; a bare object-add memory-backend-ram does NOT appear. So if
// the device_add in hotPlugMemory were removed, the presence assertion fails.
func TestMemHotplugRealQMP(t *testing.T) {
	qemuBin, cirros := advOpsSkip(t)

	// aarch64 "virt" memory hotplug needs the acpi-ged, which the machine only
	// creates when UEFI firmware is loaded (buildQEMUArgs loads it via ensureUEFI).
	// Skip if the firmware is absent -- the DIMM device_add would otherwise fail
	// "memory hotplug is not enabled: missing acpi-ged device".
	if runtime.GOARCH != "amd64" && !advOpsHasAArch64UEFI() {
		// x86 "pc" does DIMM hotplug via SeaBIOS/ACPI without UEFI; only the
		// aarch64 "virt" acpi-ged path needs the edk2 firmware.
		t.Skip("skip: aarch64 UEFI firmware not found; memory hotplug needs acpi-ged")
	}

	base := t.TempDir()
	vmBase := filepath.Join(base, "vms")
	drv, err := newKVMDriverEnhanced(qemuBin, vmBase)
	if err != nil {
		t.Skipf("skip: KVM driver init failed: %v", err)
	}
	d := drv.(*KVMDriverEnhanced)
	ctx := context.Background()
	const vmID = "memhotplug-vm"

	if _, err := d.Create(ctx, VMConfig{
		ID: vmID, Name: vmID, Type: VMTypeKVM,
		MemoryMB: 512, CPUShares: 1, Image: cirros,
		MaxMemoryMB: 1024, MemSlots: 2, // opt-in headroom (typed fields)
	}); err != nil {
		t.Fatalf("create VM: %v", err)
	}
	if err := d.Start(ctx, vmID); err != nil {
		t.Fatalf("start VM: %v", err)
	}
	defer func() {
		_ = d.Stop(context.Background(), vmID)
		// Locked read: raw d.vms[vmID] access here would race monitorVM's
		// locked PID/State writes on process exit (novacron -race gate).
		if info, err := d.GetInfo(context.Background(), vmID); err == nil && info.PID > 0 && syscall.Kill(info.PID, 0) == nil {
			_ = syscall.Kill(info.PID, syscall.SIGKILL)
		}
	}()

	sock := filepath.Join(vmBase, vmID, "qmp.sock")
	const devName = "dimm0"

	if advOpsMemDeviceIDs(t, sock)[devName] {
		t.Fatalf("precondition: DIMM %q unexpectedly present before hot-plug", devName)
	}

	if err := d.HotPlugDevice(ctx, vmID, &DeviceConfig{
		Type: "memory", Name: devName,
		Parameters: map[string]interface{}{"size_mb": 128},
	}); err != nil {
		t.Fatalf("HotPlugDevice memory: %v", err)
	}

	if !advOpsMemDeviceIDs(t, sock)[devName] {
		t.Fatalf("hot-plugged DIMM %q not present in query-memory-devices", devName)
	}
	t.Logf("PASS: DIMM %q visible in query-memory-devices after HotPlugDevice memory", devName)
}

// TestNUMATwoNodeBoot creates a VM, stores a 2-node NUMA topology via
// ConfigureNUMA (before Start), boots it, and asserts (independent QMP) that the
// running machine reports a 2-node topology via `info numa`. It also asserts the
// honest live-reconfig guard: ConfigureNUMA on the now-running VM must error.
//
// Discriminating by construction: a non-NUMA VM reports "0 nodes" (verified), so
// asserting "2 nodes" + "node 1 cpus: 1" fails if numaArgs emitted nothing. An
// inconsistent topology (node mem not summing to -m) makes qemu die on arrival,
// so a passing boot also proves the emitted -numa/-object args are valid.
func TestNUMATwoNodeBoot(t *testing.T) {
	qemuBin, cirros := advOpsSkip(t)

	base := t.TempDir()
	vmBase := filepath.Join(base, "vms")
	drv, err := newKVMDriverEnhanced(qemuBin, vmBase)
	if err != nil {
		t.Skipf("skip: KVM driver init failed: %v", err)
	}
	d := drv.(*KVMDriverEnhanced)
	ctx := context.Background()
	const vmID = "numa-vm"
	const memMB = 512

	if _, err := d.Create(ctx, VMConfig{
		ID: vmID, Name: vmID, Type: VMTypeKVM,
		MemoryMB: memMB, CPUShares: 2, Image: cirros, // 2 present vCPUs, one per node
	}); err != nil {
		t.Fatalf("create VM: %v", err)
	}

	// vcpu0 + 256MiB -> node0, vcpu1 + 256MiB -> node1 (node mem sums to MemoryMB).
	topo := &NUMATopology{
		Mode: "strict",
		Nodes: []NUMANode{
			{ID: 0, CPUs: "0", MemoryMB: memMB / 2},
			{ID: 1, CPUs: "1", MemoryMB: memMB / 2},
		},
	}
	if err := d.ConfigureNUMA(ctx, vmID, topo); err != nil {
		t.Fatalf("ConfigureNUMA before start: %v", err)
	}

	if err := d.Start(ctx, vmID); err != nil {
		t.Fatalf("start VM with NUMA (invalid -numa would die on arrival): %v", err)
	}
	defer func() {
		_ = d.Stop(context.Background(), vmID)
		// Locked read: raw d.vms[vmID] access here would race monitorVM's
		// locked PID/State writes on process exit (novacron -race gate).
		if info, err := d.GetInfo(context.Background(), vmID); err == nil && info.PID > 0 && syscall.Kill(info.PID, 0) == nil {
			_ = syscall.Kill(info.PID, syscall.SIGKILL)
		}
	}()

	if st, err := d.GetStatus(ctx, vmID); err != nil || st != StateRunning {
		t.Fatalf("VM not running after NUMA start: state=%v err=%v", st, err)
	}

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
		t.Fatalf("info numa did not reflect a 2-node topology:\n%s", out)
	}
	t.Logf("PASS: guest booted with 2-node NUMA topology:\n%s", strings.TrimSpace(out))

	// Honest guard: NUMA cannot be re-topologized on a live VM.
	if err := d.ConfigureNUMA(ctx, vmID, topo); err == nil {
		t.Fatalf("expected ConfigureNUMA to reject a running VM, got nil")
	} else {
		t.Logf("PASS: ConfigureNUMA on a running VM correctly errored: %v", err)
	}
}
