package vm

import (
	"context"
	"fmt"
	"os/exec"
	"path/filepath"
	"testing"
	"time"
)

// TestSV9RoundTripBlockMigration reproduces novacron-sv9: migrate a VM
// A -> B -> A (block migration_type) and confirms the second (B->A) leg
// succeeds, even though node A previously hosted this exact VM ID.
func TestSV9RoundTripBlockMigration(t *testing.T) {
	qemuBin, _ := findQemuAndCirros()
	if qemuBin == "" {
		t.Skip("skip: no qemu-system for this arch")
	}
	if _, err := exec.LookPath("qemu-img"); err != nil {
		t.Skip("skip: qemu-img not installed")
	}

	base := t.TempDir()
	// Two separate driver instances = two separate nodes with separate
	// vmBase directories, exactly like two separate api-server processes.
	nodeABase := filepath.Join(base, "node-a", "vms")
	nodeBBase := filepath.Join(base, "node-b", "vms")
	drvA, err := newKVMDriverEnhanced(qemuBin, nodeABase, 3*time.Second)
	if err != nil {
		t.Skipf("skip: KVM driver init failed: %v", err)
	}
	drvB, err := newKVMDriverEnhanced(qemuBin, nodeBBase, 3*time.Second)
	if err != nil {
		t.Skipf("skip: KVM driver init failed: %v", err)
	}
	dA := drvA.(*KVMDriverEnhanced)
	dB := drvB.(*KVMDriverEnhanced)
	const vmID = "sv9-vm"
	// Belt-and-suspenders: whichever node ends up holding the VM (or both, if
	// a leg fails partway), make sure nothing outlives the test. Stop on an
	// already-stopped/absent VM is a safe no-op.
	defer func() {
		_ = dA.Stop(context.Background(), vmID)
		_ = dB.Stop(context.Background(), vmID)
	}()

	rawImg := filepath.Join(base, "tiny.raw")
	if out, err := exec.Command("qemu-img", "create", "-f", "raw", rawImg, "8M").CombinedOutput(); err != nil {
		t.Fatalf("create raw image: %v: %s", err, out)
	}

	ctx := context.Background()

	// --- create + start on node A (the VM's original residency) ---
	if _, err := dA.Create(ctx, VMConfig{
		ID: vmID, Name: vmID, Type: VMTypeKVM,
		MemoryMB: 128, CPUShares: 1, Image: rawImg,
	}); err != nil {
		t.Fatalf("create on node A: %v", err)
	}
	if err := dA.Start(ctx, vmID); err != nil {
		t.Fatalf("start on node A: %v", err)
	}
	srcInfoA, err := dA.GetInfo(ctx, vmID)
	if err != nil {
		t.Fatalf("get info on node A: %v", err)
	}
	t.Logf("leg 1 (A->B): node A source disk %s", srcInfoA.RootFS)

	// --- leg 1: A -> B ---
	virtBytes, err := sourceDiskVirtualSize(ctx, srcInfoA.RootFS)
	if err != nil {
		t.Fatalf("source disk virtual size: %v", err)
	}
	portAB := freeTCPPort()
	ramURIAB := fmt.Sprintf("tcp:127.0.0.1:%d", portAB)
	destDirB := filepath.Join(nodeBBase, vmID)
	_, nbdURIAB, err := dB.StartIncomingBlock(ctx, vmID, destDirB, ramURIAB, "127.0.0.1", virtBytes, dA.vms[vmID].Config)
	if err != nil {
		t.Fatalf("leg 1: start block-migration dest on B: %v", err)
	}
	if _, _, err := dA.migrateBlockWithStats(ctx, vmID, ramURIAB, nbdURIAB, nil); err != nil {
		t.Fatalf("leg 1 (A->B) block migrate FAILED: %v", err)
	}
	_ = dB.FinishIncomingBlock(ctx, vmID)
	t.Logf("leg 1 (A->B) complete")

	// Node A's driver map still has a stale entry for vmID at this point
	// (forgetVM only cleans the VMManager layer, not the KVMDriverEnhanced
	// map) -- this is the exact condition sv9 suspects. Confirm it:
	if _, stillTracked := dA.vms[vmID]; stillTracked {
		t.Logf("confirmed: node A's driver-level vms map still has a stale entry for %s after migrate-away", vmID)
	}

	// --- leg 2: B -> A (the VM returns to its original node) ---
	destInfoB, err := dB.GetInfo(ctx, vmID)
	if err != nil {
		t.Fatalf("get info on node B: %v", err)
	}
	virtBytes2, err := sourceDiskVirtualSize(ctx, destInfoB.RootFS)
	if err != nil {
		t.Fatalf("dest disk virtual size: %v", err)
	}
	portBA := freeTCPPort()
	ramURIBA := fmt.Sprintf("tcp:127.0.0.1:%d", portBA)
	destDirA := filepath.Join(nodeABase, vmID) // SAME directory as the VM's original residency
	_, nbdURIBA, err := dA.StartIncomingBlock(ctx, vmID, destDirA, ramURIBA, "127.0.0.1", virtBytes2, dB.vms[vmID].Config)
	if err != nil {
		t.Fatalf("leg 2 (B->A) start block-migration dest FAILED (this is the sv9 repro): %v", err)
	}
	if _, _, err := dB.migrateBlockWithStats(ctx, vmID, ramURIBA, nbdURIBA, nil); err != nil {
		t.Fatalf("leg 2 (B->A) block migrate FAILED (this is the sv9 repro): %v", err)
	}
	_ = dA.FinishIncomingBlock(ctx, vmID)
	t.Logf("PASS: A->B->A round trip complete, VM %s is back on node A", vmID)
}
