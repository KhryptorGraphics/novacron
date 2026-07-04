package vm

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"testing"
	"time"
)

// TestBlockMigrationLocalhostCutover proves NON-SHARED-STORAGE (block) live
// migration end to end: the destination has its OWN empty disk (separate dir) and
// receives the source's disk purely over an NBD drive-mirror while the guest runs,
// then RAM cuts over. It asserts the facts that distinguish a real block cutover
// from both a cold restart AND a shared-storage cheat:
//
//  1. query-migrate reached "completed" (migrateBlockWithStats returns nil),
//  2. the destination guest continues the console counter with NO fresh-boot
//     marker (live RAM cutover, not a reboot from an empty disk),
//  3. the source qemu process exits,
//  4. a sentinel the GUEST wrote to its disk before cutover is present on the
//     DESTINATION's own disk afterwards -- i.e. the guest's disk state actually
//     transferred over the mirror (this is what a guestless proof cannot see).
//
// The dest disk lives in a separate directory and is created empty by the driver,
// so if the block mirror did not work the dest could not boot the guest at all.
// SKIPS (never fails) when qemu / an ISO tool / a cirros image are absent. Runs
// under TCG (arm64, this box = the -blockdev arm64 boot gate) or KVM (x86, .53).
func TestBlockMigrationLocalhostCutover(t *testing.T) {
	qemuBin, cirros := findQemuAndCirros()
	if qemuBin == "" {
		t.Skipf("skip: no qemu-system for GOARCH=%s", runtime.GOARCH)
	}
	if _, err := exec.LookPath("qemu-img"); err != nil {
		t.Skip("skip: qemu-img not installed")
	}
	isoTool := firstInPath("genisoimage", "mkisofs", "xorriso")
	if isoTool == "" {
		t.Skip("skip: no ISO tool to build the cloud-init seed")
	}
	if cirros == "" {
		t.Skipf("skip: no cirros image for %s found", runtime.GOARCH)
	}

	base := t.TempDir()
	vmBase := filepath.Join(base, "vms")
	drv, err := newKVMDriverEnhanced(qemuBin, vmBase)
	if err != nil {
		t.Skipf("skip: KVM driver init failed: %v", err)
	}
	d := drv.(*KVMDriverEnhanced)

	// The guest writes this sentinel to a raw offset in the UNUSED tail of the
	// (1 GiB-resized) disk -- past the ~112 MiB cirros filesystem, so no fs
	// corruption -- then announces SENTINEL_WRITTEN. offset must fit the disk.
	const sentinel = "BLOCKMIG_SENTINEL_v1"
	const sentinelOff = int64(900 * 1024 * 1024)
	seed, err := buildSentinelSeedISO(filepath.Join(base, "seed"), isoTool, sentinel, sentinelOff)
	if err != nil {
		t.Skipf("skip: could not build cloud-init seed ISO: %v", err)
	}

	ctx := context.Background()
	const srcID, destID = "blk-src", "blk-dst"

	if _, err := d.Create(ctx, VMConfig{
		ID: srcID, Name: srcID, Type: VMTypeKVM,
		MemoryMB: 512, CPUShares: 2, DiskSizeGB: 1, // resize -> unused tail for the sentinel
		Image: cirros, CloudInitISO: seed,
	}); err != nil {
		t.Fatalf("create source VM: %v", err)
	}
	if err := d.Start(ctx, srcID); err != nil { // normal start -- block migration must work on an ordinary VM
		t.Fatalf("start source: %v", err)
	}
	srcPID := d.vms[srcID].PID
	srcDisk := d.vms[srcID].DiskPath

	defer func() { // belt-and-suspenders: never leak qemu
		_ = d.Stop(context.Background(), destID)
		if p := d.vms[destID]; p != nil && p.PID > 0 {
			_ = syscall.Kill(p.PID, syscall.SIGKILL)
		}
		if syscall.Kill(srcPID, 0) == nil {
			_ = syscall.Kill(srcPID, syscall.SIGKILL)
		}
	}()

	// Boot + counter, then wait for the guest to write & sync its disk sentinel.
	srcConsole := filepath.Join(vmBase, srcID, "console.log")
	srcPre, ok := waitForMigTick(srcConsole, 0, 180*time.Second)
	if !ok {
		t.Fatalf("source guest never emitted MIGTICK (arm64 -blockdev boot gate FAILED); console: %s\nstderr: %s",
			srcConsole, tailFile(filepath.Join(vmBase, srcID, "qemu-stderr.log")))
	}
	if !waitForConsoleMarker(srcConsole, "SENTINEL_WRITTEN", 60*time.Second) {
		t.Fatalf("source guest never confirmed disk sentinel write; console: %s", srcConsole)
	}

	// Destination gets its OWN empty disk sized to the source's virtual size.
	virtBytes, err := sourceDiskVirtualSize(ctx, srcDisk)
	if err != nil {
		t.Fatalf("source disk virtual size: %v", err)
	}
	port := freeTCPPort()
	ramURI := fmt.Sprintf("tcp:127.0.0.1:%d", port)
	destDir := filepath.Join(base, "dst")
	_, nbdURI, err := d.StartIncomingBlock(ctx, destID, destDir, ramURI, "127.0.0.1", virtBytes, d.vms[srcID].Config)
	if err != nil {
		t.Fatalf("start block-migration dest: %v", err)
	}
	// The dest must persist config.json, else a dest-node restart cannot re-adopt
	// the migrated VM into the manager and its qemu is orphaned. Guards the
	// saveVMConfig call in the incoming path from being silently dropped.
	if _, statErr := os.Stat(filepath.Join(destDir, "config.json")); statErr != nil {
		t.Fatalf("dest config.json not persisted -- migrated VM would not re-adopt on a dest restart: %v", statErr)
	}
	destPID := d.vms[destID].PID // capture before Stop zeroes it

	downtimeMs, totalMs, err := d.migrateBlockWithStats(ctx, srcID, ramURI, nbdURI)
	if err != nil {
		t.Fatalf("block migrate did not complete: %v", err)
	}
	t.Logf("block migrate completed downtime=%dms total=%dms nbd=%s", downtimeMs, totalMs, nbdURI)
	_ = d.FinishIncomingBlock(ctx, destID)

	// (3) source qemu exits.
	if !waitProcessGone(srcPID, 20*time.Second) {
		t.Fatalf("source qemu PID %d still alive after block migration", srcPID)
	}

	// (2) destination console: counter resumes, no fresh-boot banner.
	destConsole := filepath.Join(destDir, "console.log")
	destTick, ok := waitForMigTick(destConsole, srcPre, 40*time.Second)
	if !ok {
		t.Fatalf("destination never emitted MIGTICK after cutover; console: %s", destConsole)
	}
	if destTick < srcPre {
		t.Fatalf("counter went backwards after cutover: dest=%d < src_pre=%d", destTick, srcPre)
	}
	// Cold boot vs live cutover: use EARLY kernel-boot markers, not late
	// login-stage ones. The kernel prints these once at t=0 of a real boot (on
	// the source), so they can never appear on a live-migrated dest console; late
	// markers like "login:"/"cirros-cloud.net" can straddle the cutover under slow
	// TCG (the guest's boot finishing after migration) and falsely trip.
	if data, _ := os.ReadFile(destConsole); data != nil {
		for _, marker := range []string{"Linux version", "Kernel command line"} {
			if strings.Contains(string(data), marker) {
				t.Fatalf("destination console shows kernel-boot marker %q: cold boot, not a live cutover", marker)
			}
		}
	}

	// (4) the guest's disk write is present on the DEST's OWN disk. Stop the dest
	// (release its qcow2), flatten to raw, and read the sentinel offset. If the
	// mirror had not transferred the guest's disk state this fails.
	destDisk := filepath.Join(destDir, "disk.qcow2")
	_ = d.Stop(context.Background(), destID)
	if !waitProcessGone(destPID, 15*time.Second) {
		t.Fatalf("destination qemu did not stop; cannot read its disk offline")
	}
	got, err := readDiskAt(ctx, destDisk, sentinelOff, int64(len(sentinel)))
	if err != nil {
		t.Fatalf("read dest disk sentinel: %v", err)
	}
	if string(got) != sentinel {
		t.Fatalf("dest disk missing the guest's write: got %q want %q -- disk did NOT transfer over the mirror", got, sentinel)
	}
	t.Logf("PASS: block cutover src=%d -> dest=%d, no reboot, source exited, guest disk sentinel present on dest's own disk", srcPre, destTick)
}

// findQemuAndCirros returns the qemu-system binary and a cirros image path for the
// current GOARCH, or ("","") to skip. Checks the cache dirs used across this env.
func findQemuAndCirros() (string, string) {
	arch := "aarch64"
	imgArch := "aarch64"
	if runtime.GOARCH == "amd64" {
		arch, imgArch = "x86_64", "x86_64"
	}
	qemuBin := "qemu-system-" + arch
	if _, err := exec.LookPath(qemuBin); err != nil {
		return "", ""
	}
	home, _ := os.UserHomeDir()
	for _, dir := range []string{
		filepath.Join(home, "novacron-run/images"),
		filepath.Join(home, "novacron-e2e/images"),
		"/var/lib/novacron/images", "/tmp/novacron-images",
	} {
		for _, name := range []string{
			fmt.Sprintf("cirros-0.6.2-%s-disk.img", imgArch),
			fmt.Sprintf("cirros-%s-disk.img", imgArch),
		} {
			c := filepath.Join(dir, name)
			if fi, err := os.Stat(c); err == nil && fi.Size() > 0 {
				return qemuBin, c
			}
		}
	}
	return qemuBin, ""
}

// buildSentinelSeedISO writes a NoCloud seed that (a) emits the MIGTICK counter to
// the console and (b) once, after boot settles, writes `sentinel` to raw byte
// offset `off` on /dev/vda, syncs, and prints SENTINEL_WRITTEN.
func buildSentinelSeedISO(dir, tool, sentinel string, off int64) (string, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	userData := fmt.Sprintf("#!/bin/sh\n"+
		"(while true; do echo \"MIGTICK $(date +%%s)\"; sleep 1; done) > /dev/console 2>&1 &\n"+
		"sleep 8\n"+
		"printf '%s' | dd of=/dev/vda bs=1 seek=%d conv=notrunc 2>/dev/null\n"+
		"sync\n"+
		"echo SENTINEL_WRITTEN > /dev/console 2>&1\n", sentinel, off)
	if err := os.WriteFile(filepath.Join(dir, "user-data"), []byte(userData), 0o644); err != nil {
		return "", err
	}
	if err := os.WriteFile(filepath.Join(dir, "meta-data"),
		[]byte(`{"instance-id": "iid-blk01", "local-hostname": "cirros-blk"}`), 0o644); err != nil {
		return "", err
	}
	iso := filepath.Join(dir, "seed.iso")
	var cmd *exec.Cmd
	switch tool {
	case "xorriso":
		cmd = exec.Command("xorriso", "-as", "genisoimage", "-output", iso,
			"-volid", "cidata", "-joliet", "-rock",
			filepath.Join(dir, "user-data"), filepath.Join(dir, "meta-data"))
	default:
		cmd = exec.Command(tool, "-output", iso, "-volid", "cidata", "-joliet", "-rock",
			filepath.Join(dir, "user-data"), filepath.Join(dir, "meta-data"))
	}
	if out, err := cmd.CombinedOutput(); err != nil {
		return "", fmt.Errorf("%s: %w: %s", tool, err, string(out))
	}
	return iso, nil
}

// waitForConsoleMarker polls a console log until it contains marker.
func waitForConsoleMarker(consolePath, marker string, timeout time.Duration) bool {
	deadline := time.Now().Add(timeout)
	for {
		if data, err := os.ReadFile(consolePath); err == nil && strings.Contains(string(data), marker) {
			return true
		}
		if time.Now().After(deadline) {
			return false
		}
		time.Sleep(500 * time.Millisecond)
	}
}

// readDiskAt flattens a qcow2 to raw and reads n bytes at virtual offset off,
// so a test can check guest-visible disk content without qemu-io.
func readDiskAt(ctx context.Context, qcowPath string, off, n int64) ([]byte, error) {
	flat := qcowPath + ".flat.raw"
	defer os.Remove(flat)
	if out, err := exec.CommandContext(ctx, "qemu-img", "convert", "-O", "raw", qcowPath, flat).CombinedOutput(); err != nil {
		return nil, fmt.Errorf("qemu-img convert: %w: %s", err, string(out))
	}
	f, err := os.Open(flat)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	buf := make([]byte, n)
	if _, err := f.ReadAt(buf, off); err != nil {
		return nil, err
	}
	return buf, nil
}
