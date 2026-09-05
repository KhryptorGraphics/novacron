package vm

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"
)

// TestCreateRollsBackOnDiskExhaustion is a real resource-exhaustion chaos
// experiment: it points the driver's VM storage at a genuinely tiny (4 MiB)
// tmpfs filesystem, then creates a VM from the real cirros base image (~24 MiB
// on disk). qemu-img convert must actually run out of space partway through
// (a real ENOSPC from the kernel, not a simulated/injected error), and Create
// must roll back cleanly: no orphaned VM directory on the tmpfs and no entry
// left in the driver's in-memory map. This exercises the same rollback path
// documented in STATUS.md ("VM creation rolls back atomically") and grepped
// from driver_kvm_enhanced.go's Create(): a deferred os.RemoveAll(vmDir) that
// only fires when the create never reaches the "created = true" line.
//
// Needs passwordless `sudo mount`/`umount` of tmpfs (no root capability inside
// a plain `go test` process), so it SKIPs cleanly wherever that is not
// available, like the sibling real-qemu tests skip on missing qemu/cirros.
func TestCreateRollsBackOnDiskExhaustion(t *testing.T) {
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
	if err := exec.Command("sudo", "-n", "true").Run(); err != nil {
		t.Skip("skip: passwordless `sudo` not available to mount a size-limited tmpfs")
	}

	base := t.TempDir()
	vmBase := filepath.Join(base, "vms")
	if err := os.MkdirAll(vmBase, 0755); err != nil {
		t.Fatalf("mkdir vmBase: %v", err)
	}

	// A real, tiny filesystem: 4 MiB, well under the ~24 MiB cirros image that
	// createDiskImage's `qemu-img convert` must fully write as a fresh qcow2.
	mount := exec.Command("sudo", "-n", "mount", "-t", "tmpfs", "-o", "size=4m,mode=0777", "tmpfs", vmBase)
	if out, err := mount.CombinedOutput(); err != nil {
		t.Skipf("skip: could not mount size-limited tmpfs: %v (%s)", err, string(out))
	}
	defer func() {
		if out, err := exec.Command("sudo", "-n", "umount", vmBase).CombinedOutput(); err != nil {
			// Lazy-unmount fallback so a stuck mount doesn't wedge the test host.
			exec.Command("sudo", "-n", "umount", "-l", vmBase).Run()
			t.Logf("umount %s: %v (%s)", vmBase, err, string(out))
		}
	}()

	drv, err := newKVMDriverEnhanced(qemuBin, vmBase, 3*time.Second)
	if err != nil {
		t.Skipf("skip: KVM driver init failed: %v", err)
	}
	d := drv.(*KVMDriverEnhanced)

	const vmID = "disk-exhaustion-victim"
	ctx := context.Background()
	_, createErr := d.Create(ctx, VMConfig{
		ID: vmID, Name: vmID, Type: VMTypeKVM,
		MemoryMB: 512, CPUShares: 1,
		Image: cirros,
	})

	// (1) The create must fail -- a real ENOSPC from qemu-img convert hitting
	// the 4 MiB tmpfs ceiling while copying the ~24 MiB base image.
	if createErr == nil {
		t.Fatalf("Create unexpectedly succeeded on a 4 MiB filesystem with a ~24 MiB base image")
	}
	t.Logf("Create failed as expected under disk exhaustion: %v", createErr)

	// (2) No orphaned VM directory left behind on the tmpfs.
	vmDir := filepath.Join(vmBase, vmID)
	if _, statErr := os.Stat(vmDir); !os.IsNotExist(statErr) {
		t.Fatalf("VM directory %s still exists after failed create (rollback leak); stat err: %v", vmDir, statErr)
	}

	// (3) No entry left in the driver's manager map.
	d.vmLock.RLock()
	_, stillTracked := d.vms[vmID]
	d.vmLock.RUnlock()
	if stillTracked {
		t.Fatalf("VM %s still tracked in the driver after a failed create (leak)", vmID)
	}

	t.Logf("PASS: disk-exhausted create failed cleanly with no orphaned dir and no manager entry")
}
