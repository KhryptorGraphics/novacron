package vm

import (
	"context"
	"os/exec"
	"syscall"
	"testing"
	"time"
)

// TestManagerRestartReadoptsRunningVM is a real single-node chaos experiment
// substituting for a multi-node "kill one node's api-server process, assert
// the survivor keeps serving" scenario: no second node is available in this
// sandbox (see STATUS.md's two-node microcluster notes -- that setup lives on
// a separate physical LAN host, not reachable as part of this sandbox), so
// per the task's own fallback this exercises the single-node substitute the
// task names verbatim: "kill the VM manager mid-operation and assert clean
// recovery on restart".
//
// It creates and starts a REAL qemu-backed VM (mid-boot -- the "operation" in
// flight), then discards the driver object entirely and constructs a fresh one
// against the same vmBasePath, exactly as a restarted api-server process would
// (state lives on disk + the qemu -pidfile, not in the driver's memory). The
// real qemu OS process is never touched by this -- only the Go-level manager
// state is thrown away and rebuilt, the same way a process crash+restart would
// lose in-memory state but leave the already-forked qemu child running.
// ponytail: this discards the Go object rather than SIGKILLing a separate
// api-server OS process, since the test binary itself is the only "manager"
// process here; the on-disk recovery mechanism it exercises
// (adoptRunningVMs, real -pidfile + /proc/<pid>/cmdline verification) is
// exactly what a real process crash+restart relies on, so the substitution is
// faithful to the code path, not just the vocabulary.
func TestManagerRestartReadoptsRunningVM(t *testing.T) {
	qemuBin := defaultQEMUBinary("")
	if _, err := exec.LookPath(qemuBin); err != nil {
		t.Skipf("skip: %s not installed", qemuBin)
	}
	if _, err := exec.LookPath("qemu-img"); err != nil {
		t.Skip("skip: qemu-img not installed")
	}
	cirros := findCirrosImage()
	if cirros == "" {
		t.Skip("skip: cirros image not found in known locations")
	}

	vmBase := t.TempDir()
	ctx := context.Background()
	const vmID = "restart-survivor"

	// --- "Node 1": create + start a real VM, then vanish mid-boot. ---
	drv1, err := newKVMDriverEnhanced(qemuBin, vmBase)
	if err != nil {
		t.Skipf("skip: KVM driver init failed: %v", err)
	}
	d1 := drv1.(*KVMDriverEnhanced)

	if _, err := d1.Create(ctx, VMConfig{
		ID: vmID, Name: vmID, Type: VMTypeKVM,
		MemoryMB: 512, CPUShares: 2,
		Image: cirros,
	}); err != nil {
		t.Fatalf("create VM: %v", err)
	}
	if err := d1.Start(ctx, vmID); err != nil {
		t.Fatalf("start VM: %v", err)
	}
	info1, err := d1.GetInfo(ctx, vmID)
	if err != nil {
		t.Fatalf("get VM info: %v", err)
	}
	originalPID := info1.PID
	if originalPID <= 0 {
		t.Fatalf("started VM has no PID: %+v", info1)
	}
	defer func() {
		if syscall.Kill(originalPID, 0) == nil {
			_ = syscall.Kill(originalPID, syscall.SIGKILL)
		}
	}()

	// Chaos: the manager disappears here -- mid-boot, no graceful Stop, no
	// cleanup. drv1/d1 is simply dropped; the qemu child process (already
	// forked, has its own -pidfile) keeps running on its own, unsupervised.
	drv1 = nil
	d1 = nil
	_ = drv1
	_ = d1

	// Give the abandoned qemu a moment to keep booting on its own, same as a
	// real crash where the guest doesn't care that its manager is gone.
	time.Sleep(1 * time.Second)

	// --- "Node 1" restarts: a fresh driver instance re-adopts on construction. ---
	drv2, err := newKVMDriverEnhanced(qemuBin, vmBase)
	if err != nil {
		t.Fatalf("restarted driver init failed: %v", err)
	}
	d2 := drv2.(*KVMDriverEnhanced)

	info2, err := d2.GetInfo(ctx, vmID)
	if err != nil {
		t.Fatalf("restarted manager lost track of %s entirely: %v", vmID, err)
	}
	if info2.State != StateRunning {
		t.Fatalf("restarted manager sees %s as %v, want running", vmID, info2.State)
	}
	if info2.PID != originalPID {
		t.Fatalf("restarted manager adopted PID %d, want the original %d (a differing PID means it lost the real process and is tracking something else, or nothing)", info2.PID, originalPID)
	}
	if syscall.Kill(originalPID, 0) != nil {
		t.Fatalf("original qemu PID %d is dead -- the abandoned VM did not survive the manager outage", originalPID)
	}

	t.Logf("PASS: manager restart re-adopted VM %s (PID %d) as running with zero interruption to the guest", vmID, originalPID)
}
