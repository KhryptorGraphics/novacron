package vm

import (
	"context"
	"encoding/json"
	"fmt"
	"os/exec"
	"path/filepath"
	"syscall"
	"testing"
	"time"
)

// TestLiveMigrationRollbackOnDestFailure proves migrateWithStats' failure/rollback
// path: when the destination qemu dies mid-migration, the source migration fails
// and the driver must (1) leave the SOURCE guest running (migrate_cancel + resume),
// and (2) abort the half-started destination so it is not left orphaned.
//
// The source migration is throttled to 100 KB/s so the dest-kill (fired ~1.5s in)
// reliably lands mid-transfer -- precopy cannot converge that fast, so the source
// stays in migration until the dest dies. SKIPs when qemu / iso tool / cirros are
// absent, like the sibling cutover test; runs under TCG without /dev/kvm.
func TestLiveMigrationRollbackOnDestFailure(t *testing.T) {
	qemuBin := defaultQEMUBinary("")
	if _, err := exec.LookPath(qemuBin); err != nil {
		t.Skipf("skip: %s not installed", qemuBin)
	}
	if _, err := exec.LookPath("qemu-img"); err != nil {
		t.Skip("skip: qemu-img not installed")
	}
	isoTool := firstInPath("genisoimage", "mkisofs", "xorriso")
	if isoTool == "" {
		t.Skip("skip: no ISO tool (genisoimage/mkisofs/xorriso) to build the cloud-init seed")
	}
	cirros := findCirrosImage()
	if cirros == "" {
		t.Skip("skip: cirros image not found in known locations")
	}

	base := t.TempDir()
	vmBase := filepath.Join(base, "vms")
	drv, err := newKVMDriverEnhanced(qemuBin, vmBase, 3*time.Second)
	if err != nil {
		t.Skipf("skip: KVM driver init failed: %v", err)
	}
	d := drv.(*KVMDriverEnhanced)

	seed, err := buildCounterSeedISO(filepath.Join(base, "seed"), isoTool)
	if err != nil {
		t.Skipf("skip: could not build cloud-init seed ISO: %v", err)
	}

	ctx := context.Background()
	const srcID, destID = "rb-src", "rb-dst"

	if _, err := d.Create(ctx, VMConfig{
		ID: srcID, Name: srcID, Type: VMTypeKVM,
		MemoryMB: 512, CPUShares: 2,
		Image: cirros, CloudInitISO: seed,
	}); err != nil {
		t.Fatalf("create source VM: %v", err)
	}
	if err := d.StartMigrationSource(ctx, srcID); err != nil {
		t.Fatalf("start migration source: %v", err)
	}
	srcInfo, err := d.GetInfo(ctx, srcID)
	if err != nil {
		t.Fatalf("get source VM info: %v", err)
	}
	srcPID := srcInfo.PID

	defer func() {
		_ = d.Stop(context.Background(), destID)
		if info, err := d.GetInfo(context.Background(), destID); err == nil && info.PID > 0 {
			_ = syscall.Kill(info.PID, syscall.SIGKILL)
		}
		if syscall.Kill(srcPID, 0) == nil {
			_ = syscall.Kill(srcPID, syscall.SIGKILL)
		}
	}()

	// Wait for the source guest to boot (emit the counter).
	srcConsole := filepath.Join(vmBase, srcID, "console.log")
	if _, ok := waitForMigTick(srcConsole, 0, 150*time.Second); !ok {
		t.Fatalf("source guest never booted; stderr: %s",
			tailFile(filepath.Join(vmBase, srcID, "qemu-stderr.log")))
	}

	// Stand up the destination waiting for the incoming stream.
	port := freeTCPPort()
	uri := fmt.Sprintf("tcp:127.0.0.1:%d", port)
	destDir := filepath.Join(base, "dst")
	if _, err := d.StartIncoming(ctx, srcID, destID, destDir, uri); err != nil {
		t.Fatalf("start incoming destination: %v", err)
	}
	destInfo, err := d.GetInfo(ctx, destID)
	if err != nil {
		t.Fatalf("get dest VM info: %v", err)
	}
	destPID := destInfo.PID

	// Throttle the source migration so the dest-kill lands mid-transfer.
	srcQMP := filepath.Join(vmBase, srcID, "qmp.sock")
	if q, e := qmpDial(srcQMP, 5*time.Second); e == nil {
		if _, e2 := q.execute("migrate-set-parameters", map[string]interface{}{"max-bandwidth": 100 * 1024}); e2 != nil {
			q.Close()
			t.Fatalf("throttle source migration: %v", e2)
		}
		q.Close()
	} else {
		t.Fatalf("dial source qmp to throttle: %v", e)
	}

	// Kill the destination shortly after the migration starts.
	go func() {
		time.Sleep(1500 * time.Millisecond)
		_ = syscall.Kill(destPID, syscall.SIGKILL)
	}()

	// The migration must fail; migrateWithStats runs the rollback.
	_, _, mErr := d.migrateWithStats(ctx, srcID, uri, map[string]string{"dest_vm_id": destID})
	if mErr == nil {
		t.Fatalf("migration unexpectedly succeeded; expected failure from killed dest")
	}
	t.Logf("migration failed as expected: %v", mErr)

	// (1) Source guest must have recovered to running.
	if !waitSourceRunning(srcQMP, 15*time.Second) {
		t.Fatalf("source did not recover to 'running' after the failed migration")
	}
	if syscall.Kill(srcPID, 0) != nil {
		t.Fatalf("source qemu PID %d died on the failed migration (should have recovered)", srcPID)
	}

	// (2) Destination must be gone: process dead and no longer tracked by the driver.
	if !waitProcessGone(destPID, 10*time.Second) {
		t.Fatalf("dest qemu PID %d still alive after rollback (orphan)", destPID)
	}
	d.vmLock.RLock()
	_, stillTracked := d.vms[destID]
	d.vmLock.RUnlock()
	if stillTracked {
		t.Fatalf("dest %s still tracked in the driver after rollback (leak)", destID)
	}

	t.Logf("PASS: source recovered to running, dest aborted with no orphan")
}

// waitSourceRunning dials a source QMP socket and returns true once query-status
// reports the guest running, or false on timeout.
func waitSourceRunning(qmpSock string, timeout time.Duration) bool {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if q, err := qmpDial(qmpSock, 2*time.Second); err == nil {
			raw, e := q.execute("query-status", nil)
			q.Close()
			if e == nil {
				var st struct {
					Running bool   `json:"running"`
					Status  string `json:"status"`
				}
				if json.Unmarshal(raw, &st) == nil && (st.Running || st.Status == "running") {
					return true
				}
			}
		}
		time.Sleep(300 * time.Millisecond)
	}
	return false
}
