package vm

import (
	"context"
	"encoding/json"
	"os/exec"
	"path/filepath"
	"strconv"
	"syscall"
	"testing"
	"time"

	"golang.org/x/sys/unix"
)

// TestConfigureCPUPinningRealVCPU boots a real qemu guest and asserts that
// ConfigureCPUPinning actually changes a vCPU thread's host CPU affinity via
// sched_setaffinity(2). It is discriminating by construction: the vCPU thread
// starts allowed on >=2 host CPUs (asserted as a precondition), and after
// pinning vcpu 0 to a single target CPU its affinity mask must be EXACTLY that
// one CPU. If the SchedSetaffinity call were removed, the post-mask would still
// equal the multi-CPU pre-mask and the final assertion would fail.
//
// Works under TCG (no /dev/kvm needed): qemu creates vCPU threads at machine
// init, before the guest boots, so query-cpus-fast reports a valid thread-id
// immediately after Start. SKIPS (never fails) when qemu / qemu-img / a cirros
// image are unavailable, matching the sibling migration tests' convention.
func TestConfigureCPUPinningRealVCPU(t *testing.T) {
	qemuBin := defaultQEMUBinary("") // arch-aware: qemu-system-x86_64 on amd64, -aarch64 on arm64

	if _, err := exec.LookPath(qemuBin); err != nil {
		t.Skipf("skip: %s not installed", qemuBin)
	}
	if _, err := exec.LookPath("qemu-img"); err != nil {
		t.Skip("skip: qemu-img not installed")
	}
	cirros := findCirrosImage() // defined in driver_kvm_migrate_test.go
	if cirros == "" {
		t.Skip("skip: cirros aarch64 image not found in known locations")
	}

	base := t.TempDir()
	vmBase := filepath.Join(base, "vms")
	drv, err := newKVMDriverEnhanced(qemuBin, vmBase, 3*time.Second)
	if err != nil {
		t.Skipf("skip: KVM driver init failed: %v", err)
	}
	d := drv.(*KVMDriverEnhanced)

	ctx := context.Background()
	const vmID = "cpupin-vm"

	// CPUShares:2 -> -smp 2, so the guest has vcpu 0 and 1.
	if _, err := d.Create(ctx, VMConfig{
		ID: vmID, Name: vmID, Type: VMTypeKVM,
		MemoryMB: 512, CPUShares: 2,
		Image: cirros,
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

	// Independently resolve vcpu 0's host thread-id over QMP (reusing the driver's
	// own QMP client), so the affinity read-back does not depend on the code under
	// test having mapped it correctly.
	sock := filepath.Join(vmBase, vmID, "qmp.sock")
	q, err := qmpDial(sock, 10*time.Second)
	if err != nil {
		t.Fatalf("dial QMP %s: %v", sock, err)
	}
	raw, err := q.execute("query-cpus-fast", nil)
	if err != nil {
		q.Close()
		t.Fatalf("query-cpus-fast: %v", err)
	}
	q.Close()
	var cpus []struct {
		CPUIndex int `json:"cpu-index"`
		ThreadID int `json:"thread-id"`
	}
	if err := json.Unmarshal(raw, &cpus); err != nil {
		t.Fatalf("parse query-cpus-fast %q: %v", string(raw), err)
	}
	tid := -1
	for _, c := range cpus {
		if c.CPUIndex == 0 {
			tid = c.ThreadID
		}
	}
	if tid <= 0 {
		t.Fatalf("no host thread-id for vcpu 0 in %q", string(raw))
	}

	// Precondition: the vCPU thread starts allowed on >=2 host CPUs. Pick a real,
	// currently-allowed target so the pin is valid even under a constrained cpuset.
	var pre unix.CPUSet
	if err := unix.SchedGetaffinity(tid, &pre); err != nil {
		t.Fatalf("read pre-pin affinity of tid %d: %v", tid, err)
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
	targetCPU := allowed[len(allowed)-1] // a real allowed CPU that is not the only one

	if err := d.ConfigureCPUPinning(ctx, vmID, &CPUPinningConfig{
		VCPUs: []VCPUPinning{{VCPU: 0, CPUSet: strconv.Itoa(targetCPU)}},
	}); err != nil {
		t.Fatalf("ConfigureCPUPinning: %v", err)
	}

	// Assertion: vcpu 0's thread is now allowed on EXACTLY targetCPU.
	var got unix.CPUSet
	if err := unix.SchedGetaffinity(tid, &got); err != nil {
		t.Fatalf("read post-pin affinity of tid %d: %v", tid, err)
	}
	if got.Count() != 1 || !got.IsSet(targetCPU) {
		t.Fatalf("vcpu 0 (tid %d) affinity not pinned to cpu %d: pre-count=%d post-count=%d isset=%v",
			tid, targetCPU, pre.Count(), got.Count(), got.IsSet(targetCPU))
	}
	t.Logf("PASS: vcpu 0 (tid %d) affinity narrowed from %d cpus to exactly cpu %d",
		tid, pre.Count(), targetCPU)
}
