package vm

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// TestMigrationCPUHintsFallsBackToLiveProcess reproduces novacron-z59: a
// source VM started BEFORE launch.json existed (or whose file was lost) must
// still report accurate accel/CPU hints for a migration, recovered directly
// from its own running qemu process instead of leaving the destination to
// silently guess its own default (the exact cross-accel mismatch the parent
// fix -- persistLaunchHints/MigrationCPUHints -- was built to prevent).
func TestMigrationCPUHintsFallsBackToLiveProcess(t *testing.T) {
	qemuBin, _ := findQemuAndCirros()
	if qemuBin == "" {
		t.Skip("skip: no qemu-system for this arch")
	}

	base := t.TempDir()
	drv, err := newKVMDriverEnhanced(qemuBin, filepath.Join(base, "vms"), 3*time.Second)
	if err != nil {
		t.Skipf("skip: KVM driver init failed: %v", err)
	}
	d := drv.(*KVMDriverEnhanced)

	ctx := context.Background()
	const vmID = "z59-vm"
	if _, err := d.Create(ctx, VMConfig{ID: vmID, Name: vmID, Type: VMTypeKVM, MemoryMB: 128}); err != nil {
		t.Fatalf("create: %v", err)
	}
	if err := d.Start(ctx, vmID); err != nil {
		t.Fatalf("start: %v", err)
	}
	defer func() { _ = d.Stop(context.Background(), vmID) }()

	// launch.json was written by this same Start -- capture what it says the
	// real answer is, then delete it to simulate a pre-novacron-z59-parent-fix
	// VM (started before the hints file existed).
	wantHints := d.MigrationCPUHints(vmID)
	if len(wantHints) == 0 || wantHints["accel"] == "" || wantHints["cpu_model"] == "" {
		t.Fatalf("precondition failed: launch.json hints are empty: %+v", wantHints)
	}
	launchJSON := filepath.Join(d.runtimeDir(d.vms[vmID]), "launch.json")
	if err := os.Remove(launchJSON); err != nil {
		t.Fatalf("remove launch.json: %v", err)
	}
	if _, err := os.Stat(launchJSON); err == nil {
		t.Fatal("launch.json still present after removal")
	}

	// The fallback must now recover the SAME accel/cpu_model directly from
	// the live process, with no launch.json to read.
	gotHints := d.MigrationCPUHints(vmID)
	if gotHints["accel"] != wantHints["accel"] || gotHints["cpu_model"] != wantHints["cpu_model"] {
		t.Fatalf("fallback hints = %+v, want %+v (recovered from the live process's own /proc/<pid>/cmdline)", gotHints, wantHints)
	}
	t.Logf("PASS: recovered accel=%s cpu_model=%s from the live process with no launch.json present", gotHints["accel"], gotHints["cpu_model"])
}

// TestMigrationCPUHintsNilWhenProcessGone proves the fallback does not
// invent an answer once the source process is actually gone (stopped VM,
// or one that never started) -- MigrationCPUHints must return an empty map,
// not stale or fabricated data.
func TestMigrationCPUHintsNilWhenProcessGone(t *testing.T) {
	qemuBin, _ := findQemuAndCirros()
	if qemuBin == "" {
		t.Skip("skip: no qemu-system for this arch")
	}
	base := t.TempDir()
	drv, err := newKVMDriverEnhanced(qemuBin, filepath.Join(base, "vms"), 3*time.Second)
	if err != nil {
		t.Skipf("skip: KVM driver init failed: %v", err)
	}
	d := drv.(*KVMDriverEnhanced)

	ctx := context.Background()
	const vmID = "z59-never-started"
	if _, err := d.Create(ctx, VMConfig{ID: vmID, Name: vmID, Type: VMTypeKVM, MemoryMB: 128}); err != nil {
		t.Fatalf("create: %v", err)
	}
	// Never started: no launch.json, no PID, no live process.
	hints := d.MigrationCPUHints(vmID)
	if len(hints) != 0 {
		t.Fatalf("MigrationCPUHints for a never-started VM = %+v, want empty", hints)
	}
}
