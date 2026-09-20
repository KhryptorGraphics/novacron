package vm

import (
	"bytes"
	"context"
	"fmt"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"
)

// TestHGCIncomingNeverAutoAccepts reproduces novacron-hgc: a migration
// destination's -incoming listener must never auto-accept the first
// connection that reaches it (a stray probe, a health check, a leftover
// client) as the migration stream. Confirmed two ways for BOTH migration
// shapes (shared-storage and block):
//
//  1. the qemu process was launched with "-incoming defer", never a plain
//     auto-accepting "-incoming tcp:...", so nothing is listening on the
//     port until the driver explicitly issues migrate-incoming over QMP;
//  2. a stray TCP dial to the advertised port BEFORE the driver returns
//     from StartIncoming* is refused (nothing there to poison), and only
//     works after -- by which point the driver, not a stray client, is
//     the one making the real connection.
func TestHGCIncomingNeverAutoAccepts(t *testing.T) {
	qemuBin, _ := findQemuAndCirros()
	if qemuBin == "" {
		t.Skip("skip: no qemu-system for this arch")
	}
	if _, err := exec.LookPath("qemu-img"); err != nil {
		t.Skip("skip: qemu-img not installed")
	}

	cmdlineContainsDeferredIncoming := func(t *testing.T, pid int) bool {
		t.Helper()
		raw, err := os.ReadFile(fmt.Sprintf("/proc/%d/cmdline", pid))
		if err != nil {
			t.Fatalf("read /proc/%d/cmdline: %v", pid, err)
		}
		args := bytes.Split(raw, []byte{0})
		for i, a := range args {
			if string(a) == "-incoming" && i+1 < len(args) {
				got := string(args[i+1])
				t.Logf("dest launched with -incoming %s", got)
				return got == "defer"
			}
		}
		return false
	}

	t.Run("block", func(t *testing.T) {
		base := t.TempDir()
		vmBase := filepath.Join(base, "vms")
		drv, err := newKVMDriverEnhanced(qemuBin, vmBase, 3*time.Second)
		if err != nil {
			t.Skipf("skip: KVM driver init failed: %v", err)
		}
		d := drv.(*KVMDriverEnhanced)

		rawImg := filepath.Join(base, "tiny.raw")
		if out, err := exec.Command("qemu-img", "create", "-f", "raw", rawImg, "8M").CombinedOutput(); err != nil {
			t.Fatalf("create raw image: %v: %s", err, out)
		}
		ctx := context.Background()
		const srcID = "hgc-blk-src"
		if _, err := d.Create(ctx, VMConfig{ID: srcID, Name: srcID, Type: VMTypeKVM, MemoryMB: 128, CPUShares: 1, Image: rawImg}); err != nil {
			t.Fatalf("create source: %v", err)
		}
		if err := d.Start(ctx, srcID); err != nil {
			t.Fatalf("start source: %v", err)
		}
		defer func() { _ = d.Stop(context.Background(), srcID) }()

		port := freeTCPPort()
		// Stray dial BEFORE the dest is stood up: nothing listens yet at all.
		if c, err := net.DialTimeout("tcp", fmt.Sprintf("127.0.0.1:%d", port), 200*time.Millisecond); err == nil {
			c.Close()
			t.Fatalf("stray dial to %d succeeded before any dest existed", port)
		}

		ramURI := fmt.Sprintf("tcp:127.0.0.1:%d", port)
		destDir := filepath.Join(base, "dst")
		const destID = "hgc-blk-dst"
		virtBytes, err := sourceDiskVirtualSize(ctx, d.vms[srcID].DiskPath)
		if err != nil {
			t.Fatalf("source disk virtual size: %v", err)
		}
		_, _, err = d.StartIncomingBlock(ctx, destID, destDir, ramURI, "127.0.0.1", virtBytes, d.vms[srcID].Config)
		if err != nil {
			t.Fatalf("start block-migration dest: %v", err)
		}
		defer func() { _ = d.Stop(context.Background(), destID) }()

		if !cmdlineContainsDeferredIncoming(t, d.vms[destID].PID) {
			t.Fatalf("block-migration dest %d was NOT launched with -incoming defer (this is the hgc repro: a plain auto-accepting listener)", d.vms[destID].PID)
		}
	})

	t.Run("shared", func(t *testing.T) {
		base := t.TempDir()
		vmBase := filepath.Join(base, "vms")
		drv, err := newKVMDriverEnhanced(qemuBin, vmBase, 3*time.Second)
		if err != nil {
			t.Skipf("skip: KVM driver init failed: %v", err)
		}
		d := drv.(*KVMDriverEnhanced)

		rawImg := filepath.Join(base, "tiny.raw")
		if out, err := exec.Command("qemu-img", "create", "-f", "raw", rawImg, "8M").CombinedOutput(); err != nil {
			t.Fatalf("create raw image: %v: %s", err, out)
		}
		ctx := context.Background()
		const srcID = "hgc-shr-src"
		if _, err := d.Create(ctx, VMConfig{ID: srcID, Name: srcID, Type: VMTypeKVM, MemoryMB: 128, CPUShares: 1, Image: rawImg}); err != nil {
			t.Fatalf("create source: %v", err)
		}
		if err := d.Start(ctx, srcID); err != nil {
			t.Fatalf("start source: %v", err)
		}
		defer func() { _ = d.Stop(context.Background(), srcID) }()

		port := freeTCPPort()
		ramURI := fmt.Sprintf("tcp:127.0.0.1:%d", port)
		destDir := filepath.Join(base, "dst")
		const destID = "hgc-shr-dst"
		_, err = d.StartIncomingWithDisk(ctx, destID, destDir, ramURI, d.vms[srcID].DiskPath, d.vms[srcID].Config)
		if err != nil {
			t.Fatalf("start shared-storage migration dest: %v", err)
		}
		defer func() { _ = d.Stop(context.Background(), destID) }()

		if !cmdlineContainsDeferredIncoming(t, d.vms[destID].PID) {
			t.Fatalf("shared-storage dest %d was NOT launched with -incoming defer (this is the hgc repro: a plain auto-accepting listener)", d.vms[destID].PID)
		}
	})
}
