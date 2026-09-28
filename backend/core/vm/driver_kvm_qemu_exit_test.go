package vm

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"syscall"
	"testing"
	"time"
)

// newQEMUExitTestDriver builds a driver on a real qemu-system under a fresh
// temp dir, or skips when qemu / qemu-img are unavailable. Runs under TCG when
// /dev/kvm is not accessible (no guest image is booted; qemu only has to stay
// up with its QMP socket open).
func newQEMUExitTestDriver(t *testing.T, qmpStartupTimeout time.Duration) *KVMDriverEnhanced {
	t.Helper()
	qemuBin, _ := findQemuAndCirros()
	if qemuBin == "" {
		t.Skip("skip: no qemu-system for this arch")
	}
	if _, err := exec.LookPath("qemu-img"); err != nil {
		t.Skip("skip: qemu-img not installed")
	}
	drv, err := newKVMDriverEnhanced(qemuBin, filepath.Join(t.TempDir(), "vms"), qmpStartupTimeout)
	if err != nil {
		t.Skipf("skip: KVM driver init failed: %v", err)
	}
	return drv.(*KVMDriverEnhanced)
}

// createQEMUExitTestVM creates vmID and, BEFORE it is started, registers a
// cleanup that stops it -- so even a failed Start or a failed assertion never
// leaves a qemu running into the t.TempDir removal (novacron-c0p). Stop after
// the test's own Delete reports "not found", which is fine; a qemu that is
// still alive after the cleanup Stop is a leak and fails the test.
func createQEMUExitTestVM(t *testing.T, d *KVMDriverEnhanced, vmID string) {
	t.Helper()
	if _, err := d.Create(context.Background(), VMConfig{
		ID: vmID, Name: vmID, Type: VMTypeKVM, MemoryMB: 128, CPUShares: 1,
	}); err != nil {
		t.Fatalf("create %s: %v", vmID, err)
	}
	t.Cleanup(func() {
		_ = d.Stop(context.Background(), vmID)
		if pid := d.findQEMUProcessByVMID(vmID); pid > 0 {
			_ = syscall.Kill(pid, syscall.SIGKILL)
			t.Errorf("qemu PID %d for %s still running after cleanup Stop", pid, vmID)
		}
	})
}

func startedQEMUPID(t *testing.T, d *KVMDriverEnhanced, vmID string) int {
	t.Helper()
	if err := d.Start(context.Background(), vmID); err != nil {
		t.Fatalf("start %s: %v", vmID, err)
	}
	info, err := d.GetInfo(context.Background(), vmID)
	if err != nil || info.PID <= 0 {
		t.Fatalf("started VM %s has no qemu PID: info=%+v err=%v", vmID, info, err)
	}
	return info.PID
}

// deleteQEMUExitTestVM checks Delete's observable cleanup: the VM is forgotten and
// its directory is gone.
func deleteQEMUExitTestVM(t *testing.T, d *KVMDriverEnhanced, vmID string) {
	t.Helper()
	if err := d.Delete(context.Background(), vmID); err != nil {
		t.Fatalf("Delete after qemu exited: %v", err)
	}
	if _, err := d.GetStatus(context.Background(), vmID); err == nil {
		t.Fatalf("VM %s still tracked after Delete", vmID)
	}
	if _, err := os.Stat(filepath.Join(d.vmBasePath, vmID)); !os.IsNotExist(err) {
		t.Fatalf("VM dir still present after Delete: stat err=%v", err)
	}
}

func requireQEMUExitTestStatus(t *testing.T, d *KVMDriverEnhanced, vmID string, want State) {
	t.Helper()
	if st, err := d.GetStatus(context.Background(), vmID); err != nil || st != want {
		t.Fatalf("status of %s = %q (err %v), want %q", vmID, st, err, want)
	}
}

// TestKVMQEMUExitStopDelete reproduces novacron-3pw: qemu dies on its own
// (SIGKILLed from outside the driver, as by the OOM killer). The driver must
// observe the exit, and Stop/Delete must succeed afterwards -- they used to
// fail "VM ... is not running", leaving the VM undeletable. Each case covers a
// distinct ordering against the monitorVM reaper recording the exit.
func TestKVMQEMUExitStopDelete(t *testing.T) {
	d := newQEMUExitTestDriver(t, 30*time.Second)
	ctx := context.Background()

	t.Run("exit observed then Stop and Delete", func(t *testing.T) {
		const vmID = "qexit-observed"
		createQEMUExitTestVM(t, d, vmID)
		pid := startedQEMUPID(t, d, vmID)
		if err := syscall.Kill(pid, syscall.SIGKILL); err != nil {
			t.Fatalf("kill qemu %d: %v", pid, err)
		}

		// No driver call in between: the reaper alone must record the crash.
		deadline := time.Now().Add(15 * time.Second)
		for {
			st, err := d.GetStatus(ctx, vmID)
			if err != nil {
				t.Fatalf("GetStatus: %v", err)
			}
			if st != StateRunning {
				break
			}
			if time.Now().After(deadline) {
				t.Fatalf("driver still reports %s running 15s after its qemu %d was killed", vmID, pid)
			}
			time.Sleep(20 * time.Millisecond)
		}
		requireQEMUExitTestStatus(t, d, vmID, StateFailed)
		if info, err := d.GetInfo(ctx, vmID); err != nil || info.PID != 0 {
			t.Fatalf("exited VM still carries a PID: info=%+v err=%v", info, err)
		}

		if err := d.Stop(ctx, vmID); err != nil {
			t.Fatalf("Stop after qemu exited on its own: %v", err)
		}
		requireQEMUExitTestStatus(t, d, vmID, StateStopped)
		if err := d.Stop(ctx, vmID); err != nil {
			t.Fatalf("second Stop is not idempotent: %v", err)
		}
		deleteQEMUExitTestVM(t, d, vmID)
	})

	t.Run("Stop racing the reaper", func(t *testing.T) {
		const vmID = "qexit-stoprace"
		createQEMUExitTestVM(t, d, vmID)
		pid := startedQEMUPID(t, d, vmID)
		if err := syscall.Kill(pid, syscall.SIGKILL); err != nil {
			t.Fatalf("kill qemu %d: %v", pid, err)
		}
		if err := d.Stop(ctx, vmID); err != nil {
			t.Fatalf("Stop right after qemu was killed: %v", err)
		}
		requireQEMUExitTestStatus(t, d, vmID, StateStopped)
		deleteQEMUExitTestVM(t, d, vmID)
	})

	t.Run("Delete without Stop", func(t *testing.T) {
		const vmID = "qexit-delete"
		createQEMUExitTestVM(t, d, vmID)
		pid := startedQEMUPID(t, d, vmID)
		if err := syscall.Kill(pid, syscall.SIGKILL); err != nil {
			t.Fatalf("kill qemu %d: %v", pid, err)
		}
		deleteQEMUExitTestVM(t, d, vmID)
	})
}

// TestKVMFailedLaunchLeavesNoQEMU covers the other way the driver and qemu
// diverged: a launch whose liveness check fails (qemu slow to open QMP on a
// loaded host, forced here with a 1ns startup timeout) returned an error and
// reported StateFailed while that qemu kept running -- Stop then refused it as
// "not running", so it outlived the test into the t.TempDir removal
// ("directory not empty", novacron-c0p). The failed Start must leave no qemu,
// and Stop/Delete must then succeed.
func TestKVMFailedLaunchLeavesNoQEMU(t *testing.T) {
	d := newQEMUExitTestDriver(t, time.Nanosecond)
	ctx := context.Background()
	const vmID = "qexit-abandoned"
	createQEMUExitTestVM(t, d, vmID)

	if err := d.Start(ctx, vmID); err == nil {
		t.Fatal("Start succeeded despite a 1ns QMP startup timeout")
	}
	requireQEMUExitTestStatus(t, d, vmID, StateFailed)
	if pid := d.findQEMUProcessByVMID(vmID); pid > 0 {
		t.Fatalf("failed Start left qemu PID %d running", pid)
	}
	if err := d.Stop(ctx, vmID); err != nil {
		t.Fatalf("Stop after a failed launch: %v", err)
	}
	requireQEMUExitTestStatus(t, d, vmID, StateStopped)
	deleteQEMUExitTestVM(t, d, vmID)
}
