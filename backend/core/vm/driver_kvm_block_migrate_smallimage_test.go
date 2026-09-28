package vm

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"
)

// TestH71SmallRawImageRootBlockNode reproduces novacron-h71: create a VM from
// a tiny (8 MiB) raw image, start it, then attempt a block migration. If the
// bead's suspected root cause is real, drive-mirror fails with
// "Need a root block node".
func TestH71SmallRawImageRootBlockNode(t *testing.T) {
	qemuBin, _ := findQemuAndCirros()
	if qemuBin == "" {
		t.Skip("skip: no qemu-system for this arch")
	}
	if _, err := exec.LookPath("qemu-img"); err != nil {
		t.Skip("skip: qemu-img not installed")
	}

	base := t.TempDir()
	vmBase := filepath.Join(base, "vms")
	drv, err := newKVMDriverEnhanced(qemuBin, vmBase, 3*time.Second)
	if err != nil {
		t.Skipf("skip: KVM driver init failed: %v", err)
	}
	d := drv.(*KVMDriverEnhanced)

	// Build an 8 MiB raw image, exactly matching h71's repro.
	rawImg := filepath.Join(base, "tiny.raw")
	if out, err := exec.Command("qemu-img", "create", "-f", "raw", rawImg, "8M").CombinedOutput(); err != nil {
		t.Fatalf("create raw image: %v: %s", err, out)
	}

	ctx := context.Background()
	const srcID = "h71-src"
	if _, err := d.Create(ctx, VMConfig{
		ID: srcID, Name: srcID, Type: VMTypeKVM,
		MemoryMB: 128, CPUShares: 1, // DiskSizeGB: 0 -- no resize, stays 8 MiB
		Image: rawImg,
	}); err != nil {
		t.Fatalf("create source VM: %v", err)
	}
	// Registered before Start and after t.TempDir, so it runs first (LIFO)
	// even when Start or a later step fails: Stop returns only once the qemu
	// is reaped and its exit fully recorded, so nothing still writes under
	// base when t.TempDir removes it ("directory not empty", novacron-c0p).
	stopAtCleanup(t, d, srcID)
	if err := d.Start(ctx, srcID); err != nil {
		t.Fatalf("start source: %v", err)
	}

	srcInfo, err := d.GetInfo(ctx, srcID)
	if err != nil {
		t.Fatalf("get source VM info: %v", err)
	}
	srcDisk := srcInfo.RootFS
	t.Logf("source disk: %s", srcDisk)
	if out, err := exec.Command("qemu-img", "info", srcDisk).CombinedOutput(); err == nil {
		t.Logf("qemu-img info:\n%s", out)
	}

	virtBytes, err := sourceDiskVirtualSize(ctx, srcDisk)
	if err != nil {
		t.Fatalf("source disk virtual size: %v", err)
	}
	t.Logf("virtual size: %d", virtBytes)

	port := freeTCPPort()
	ramURI := fmt.Sprintf("tcp:127.0.0.1:%d", port)
	destDir := filepath.Join(base, "dst")
	const destID = "h71-dst"
	_, nbdURI, err := d.StartIncomingBlock(ctx, destID, destDir, ramURI, "127.0.0.1", virtBytes, d.vms[srcID].Config)
	if err != nil {
		t.Fatalf("start block-migration dest: %v", err)
	}
	stopAtCleanup(t, d, destID)

	downtimeMs, totalMs, err := d.migrateBlockWithStats(ctx, srcID, ramURI, nbdURI, nil)
	if err != nil {
		// Dump both stderr logs before failing so we can see the exact QEMU state.
		srcErr, _ := os.ReadFile(filepath.Join(vmBase, srcID, "qemu-stderr.log"))
		dstErr, _ := os.ReadFile(filepath.Join(destDir, "qemu-stderr.log"))
		t.Fatalf("block migrate FAILED (this is the h71 repro if it says 'Need a root block node'): %v\n--- src stderr ---\n%s\n--- dst stderr ---\n%s", err, srcErr, dstErr)
	}
	t.Logf("block migrate completed downtime=%dms total=%dms", downtimeMs, totalMs)
}

// stopAtCleanup stops vmID when the test ends, reporting a failed Stop -- a
// qemu it could not stop would outlive the test into its t.TempDir removal.
func stopAtCleanup(t *testing.T, d *KVMDriverEnhanced, vmID string) {
	t.Helper()
	t.Cleanup(func() {
		if err := d.Stop(context.Background(), vmID); err != nil {
			t.Errorf("cleanup: stop %s: %v", vmID, err)
		}
	})
}
