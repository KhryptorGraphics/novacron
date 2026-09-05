package vm

import (
	"context"
	"encoding/json"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"

	"golang.org/x/sys/unix"
)

// This file exercises CEILING 2: buildQEMUArgs emitting `-object iothread` (opt-in
// via Config.Tags["iothreads"]=N) so ConfigureCPUPinning's iothread branch is no
// longer a no-op. TestBuildQEMUArgsIOThreadOptIn is a pure arg-builder test (no
// qemu) proving the default disk device is byte-for-byte unchanged; the real-QMP
// test boots a guest and asserts the pin actually moved the iothread's affinity.

// countArg counts args exactly equal to want (shared with the NUMA-persist test).
func countArg(args []string, want string) int {
	n := 0
	for _, a := range args {
		if a == want {
			n++
		}
	}
	return n
}

// diskDeviceArg returns the primary-disk -device value (the arg beginning with
// "virtio-blk-pci,drive=migdisk"), or "" if absent.
func diskDeviceArg(args []string) string {
	for _, a := range args {
		if strings.HasPrefix(a, "virtio-blk-pci,drive="+kvmMigDiskNode) {
			return a
		}
	}
	return ""
}

// countIOThreadObjs counts the `-object iothread,id=iothreadN` values in args.
func countIOThreadObjs(args []string) int {
	n := 0
	for _, a := range args {
		if strings.HasPrefix(a, "iothread,id=iothread") {
			n++
		}
	}
	return n
}

// TestBuildQEMUArgsIOThreadOptIn proves the CRITICAL SAFETY property with no qemu:
// a default VM (no "iothreads" tag) emits NO iothread object and keeps the primary
// disk -device byte-for-byte identical, while the opt-in tag is the ONLY thing that
// adds iothreads + moves the disk onto iothread0. This keeps migration cutover
// (which never sets the tag) unaffected.
func TestBuildQEMUArgsIOThreadOptIn(t *testing.T) {
	// x86 "pc" machine avoids aarch64 UEFI/pcie side effects; the iothread opt-in
	// logic is arch-independent. buildQEMUArgs execs nothing, just builds args.
	d := &KVMDriverEnhanced{qemuBinaryPath: "/usr/bin/qemu-system-x86_64", vmBasePath: t.TempDir()}
	tmp := t.TempDir()
	mk := func(cfg VMConfig) []string {
		return d.buildQEMUArgs(&KVMVMInfo{
			ID: "u", DiskPath: filepath.Join(tmp, "disk.qcow2"),
			MonitorPath: filepath.Join(tmp, "mon.sock"), VNCPort: 5900,
			Config: cfg,
		})
	}

	// 1. Default VM: no iothread object, and the primary disk device is the EXACT
	// literal it has always been (migration cutover depends on this byte-for-byte).
	def := mk(VMConfig{MemoryMB: 512, CPUShares: 2})
	if n := countIOThreadObjs(def); n != 0 {
		t.Fatalf("default VM unexpectedly got %d iothread objects", n)
	}
	const wantDefaultDisk = "virtio-blk-pci,drive=" + kvmMigDiskNode
	if got := diskDeviceArg(def); got != wantDefaultDisk {
		t.Fatalf("default disk -device changed: got %q, want %q (migration topology must stay byte-for-byte)", got, wantDefaultDisk)
	}
	for _, a := range def {
		if strings.Contains(a, "iothread") {
			t.Fatalf("default VM unexpectedly mentions iothread in arg %q", a)
		}
	}

	// 2. Opt-in iothreads=2: two iothread objects AND the primary disk runs on
	// iothread0. The rest of the disk device string is unchanged.
	hp := mk(VMConfig{MemoryMB: 512, CPUShares: 2, Tags: map[string]string{"iothreads": "2"}})
	if n := countIOThreadObjs(hp); n != 2 {
		t.Fatalf("iothreads=2: got %d iothread objects, want 2", n)
	}
	if got, want := diskDeviceArg(hp), wantDefaultDisk+",iothread=iothread0"; got != want {
		t.Fatalf("opt-in disk -device: got %q, want %q", got, want)
	}

	// 3. iothreads=0 (explicit) is still a no-op: no object, disk unchanged.
	zero := mk(VMConfig{MemoryMB: 512, CPUShares: 2, Tags: map[string]string{"iothreads": "0"}})
	if n := countIOThreadObjs(zero); n != 0 {
		t.Fatalf("iothreads=0: got %d iothread objects, want 0", n)
	}
	if got := diskDeviceArg(zero); got != wantDefaultDisk {
		t.Fatalf("iothreads=0 disk -device changed: got %q, want %q", got, wantDefaultDisk)
	}
	t.Logf("PASS: default disk device unchanged + no iothreads; iothreads=N is strictly opt-in")
}

// TestIOThreadPinningRealQMP boots a VM with opt-in iothreads=1 (so buildQEMUArgs
// emits `-object iothread,id=iothread0` and puts the primary disk on it), then pins
// iothread0 via ConfigureCPUPinning and asserts (independent QMP + SchedGetaffinity)
// that the iothread's host thread affinity was narrowed to exactly one CPU.
//
// Discriminating by construction: if buildQEMUArgs did NOT emit the iothread object
// (the CEILING-2 bug), query-iothreads returns [] — the test fails at the explicit
// "no iothreads present" guard. With the object, the thread starts allowed on >=2
// CPUs (precondition) and ends allowed on exactly the target CPU. If pinIOThreads /
// SchedSetaffinity were removed, the post-mask would still equal the multi-CPU
// pre-mask and the final assertion would fail. SKIPS cleanly when qemu/qemu-img/a
// cirros image are unavailable, matching the sibling advanced-ops tests.
func TestIOThreadPinningRealQMP(t *testing.T) {
	qemuBin, cirros := advOpsSkip(t) // defined in driver_kvm_advops_test.go

	base := t.TempDir()
	vmBase := filepath.Join(base, "vms")
	drv, err := newKVMDriverEnhanced(qemuBin, vmBase, 3*time.Second)
	if err != nil {
		t.Skipf("skip: KVM driver init failed: %v", err)
	}
	d := drv.(*KVMDriverEnhanced)
	ctx := context.Background()
	const vmID = "iothread-pin-vm"

	if _, err := d.Create(ctx, VMConfig{
		ID: vmID, Name: vmID, Type: VMTypeKVM,
		MemoryMB: 512, CPUShares: 2, Image: cirros,
		Tags: map[string]string{"iothreads": "1"}, // opt-in: -object iothread,id=iothread0
	}); err != nil {
		t.Fatalf("create VM: %v", err)
	}
	if err := d.Start(ctx, vmID); err != nil {
		t.Fatalf("start VM (virtio-blk on iothread0): %v", err)
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

	// Independently resolve iothread0's host thread-id over QMP (this existing at
	// all is the proof CEILING 2 is closed), so the affinity read-back does not
	// depend on the code under test.
	raw, err := advOpsQMP(t, sock, "query-iothreads", nil)
	if err != nil {
		t.Fatalf("query-iothreads: %v", err)
	}
	var iothreads []struct {
		ID       string `json:"id"`
		ThreadID int    `json:"thread-id"`
	}
	if err := json.Unmarshal(raw, &iothreads); err != nil {
		t.Fatalf("parse query-iothreads %q: %v", string(raw), err)
	}
	if len(iothreads) == 0 {
		t.Fatalf("no iothreads present: buildQEMUArgs did not emit -object iothread (CEILING 2 not closed)")
	}
	tid := -1
	for _, it := range iothreads {
		if it.ID == "iothread0" {
			tid = it.ThreadID
		}
	}
	if tid <= 0 {
		t.Fatalf("no host thread-id for iothread0 in %q", string(raw))
	}

	// Precondition: the iothread starts allowed on >=2 host CPUs. Pick a real,
	// currently-allowed target so the pin is valid even under a constrained cpuset.
	var pre unix.CPUSet
	if err := unix.SchedGetaffinity(tid, &pre); err != nil {
		t.Fatalf("read pre-pin affinity of iothread0 (tid %d): %v", tid, err)
	}
	var allowed []int
	for c := 0; c < 1024; c++ {
		if pre.IsSet(c) {
			allowed = append(allowed, c)
		}
	}
	if len(allowed) < 2 {
		t.Skipf("need >=2 allowed host CPUs to discriminate a single-CPU pin, got %d", len(allowed))
	}
	targetCPU := allowed[len(allowed)-1]

	if err := d.ConfigureCPUPinning(ctx, vmID, &CPUPinningConfig{
		IOThreads: []IOThreadPinning{{IOThread: 0, CPUSet: strconv.Itoa(targetCPU)}},
	}); err != nil {
		t.Fatalf("ConfigureCPUPinning iothread: %v", err)
	}

	// Assertion: iothread0's thread is now allowed on EXACTLY targetCPU.
	var got unix.CPUSet
	if err := unix.SchedGetaffinity(tid, &got); err != nil {
		t.Fatalf("read post-pin affinity of iothread0 (tid %d): %v", tid, err)
	}
	if got.Count() != 1 || !got.IsSet(targetCPU) {
		t.Fatalf("iothread0 (tid %d) affinity not pinned to cpu %d: pre-count=%d post-count=%d isset=%v",
			tid, targetCPU, pre.Count(), got.Count(), got.IsSet(targetCPU))
	}
	t.Logf("PASS: iothread0 (tid %d) affinity narrowed from %d cpus to exactly cpu %d via ConfigureCPUPinning",
		tid, pre.Count(), targetCPU)
}
