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
	if err := d.Start(ctx, srcID); err != nil {
		t.Fatalf("start source: %v", err)
	}
	defer func() { _ = d.Stop(context.Background(), srcID) }()

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
	defer func() { _ = d.Stop(context.Background(), destID) }()

	downtimeMs, totalMs, err := d.migrateBlockWithStats(ctx, srcID, ramURI, nbdURI, nil)
	if err != nil {
		// Dump both stderr logs before failing so we can see the exact QEMU state.
		srcErr, _ := os.ReadFile(filepath.Join(vmBase, srcID, "qemu-stderr.log"))
		dstErr, _ := os.ReadFile(filepath.Join(destDir, "qemu-stderr.log"))
		t.Fatalf("block migrate FAILED (this is the h71 repro if it says 'Need a root block node'): %v\n--- src stderr ---\n%s\n--- dst stderr ---\n%s", err, srcErr, dstErr)
	}
	t.Logf("block migrate completed downtime=%dms total=%dms", downtimeMs, totalMs)
}
