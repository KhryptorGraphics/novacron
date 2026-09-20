package main

import (
	"database/sql"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"syscall"
	"testing"

	"github.com/google/uuid"

	core_vm "github.com/khryptorgraphics/novacron/backend/core/vm"
)

// TestReconcileOrphanedMigrationDestsAdoptsUnregisteredLiveVM reproduces
// novacron-05h's residual gap: a migration destination whose
// registerMigratedDest goroutine died with a prior api-server process before
// it ran has a live qemu process and a config.json, but NO vms row at all --
// reconcileVMState's "SELECT id, state FROM vms" query never even sees it,
// so it stays invisible to /api/vms and unreachable by control ops forever.
// reconcileOrphanedMigrationDests must find and re-register it at boot.
//
// Does not need a real qemu binary: pidFileAlive only requires the recorded
// pid's /proc/<pid>/cmdline to contain "qemu" and the vm id, so a lightweight
// stand-in process with a crafted argv exercises the exact same check a real
// qemu process launched by buildQEMUArgs would satisfy (its own args always
// contain both, e.g. -pidfile <vmBase>/<id>/qemu.pid).
func TestReconcileOrphanedMigrationDestsAdoptsUnregisteredLiveVM(t *testing.T) {
	admin, err := sql.Open("postgres", pgAdminDSN())
	if err != nil {
		t.Skipf("skip: cannot open postgres admin connection: %v", err)
	}
	if err := admin.Ping(); err != nil {
		admin.Close()
		t.Skipf("skip: postgres not reachable: %v", err)
	}

	const dbName = "novacron_reconcile_orphan_test"
	_, _ = admin.Exec(`DROP DATABASE IF EXISTS "` + dbName + `"`)
	if _, err := admin.Exec(`CREATE DATABASE "` + dbName + `"`); err != nil {
		admin.Close()
		t.Fatalf("create database: %v", err)
	}
	applyMigrations(t, dbDSN(dbName), dbName)

	sqlDB, err := sql.Open("postgres", dbDSN(dbName))
	if err != nil {
		admin.Close()
		t.Fatalf("open db: %v", err)
	}
	t.Cleanup(func() {
		sqlDB.Close()
		_, _ = admin.Exec(`DROP DATABASE IF EXISTS "` + dbName + `"`)
		admin.Close()
	})

	base := t.TempDir()
	vmBase := filepath.Join(base, "vms")
	vmID := uuid.NewString()
	vmDir := filepath.Join(vmBase, vmID)
	if err := os.MkdirAll(vmDir, 0755); err != nil {
		t.Fatalf("mkdir vmDir: %v", err)
	}

	cfg := core_vm.VMConfig{ID: vmID, Name: "orphaned migration dest", Type: core_vm.VMTypeKVM, MemoryMB: 256, VCPUs: 1}
	cfgJSON, err := json.Marshal(cfg)
	if err != nil {
		t.Fatalf("marshal config: %v", err)
	}
	if err := os.WriteFile(filepath.Join(vmDir, "config.json"), cfgJSON, 0644); err != nil {
		t.Fatalf("write config.json: %v", err)
	}

	// A stand-in "qemu" process: its own argv contains "qemu" and the vm id,
	// which is all pidFileAlive checks (see backend/cmd/api-server/main.go).
	// Setpgid + killing the negated pid cleans up "sleep" too: sh -c does not
	// exec-replace itself here (the trailing "# comment" makes it a compound
	// command), so sleep runs as sh's child, not sh's replacement.
	proc := exec.Command("sh", "-c", "sleep 300 # qemu "+vmID)
	proc.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	if err := proc.Start(); err != nil {
		t.Fatalf("start stand-in process: %v", err)
	}
	defer func() { _ = syscall.Kill(-proc.Process.Pid, syscall.SIGKILL) }()
	if err := os.WriteFile(filepath.Join(vmDir, "qemu.pid"), []byte(strconv.Itoa(proc.Process.Pid)), 0644); err != nil {
		t.Fatalf("write qemu.pid: %v", err)
	}

	// Ground truth: NO vms row exists for this id (simulates registerMigratedDest
	// never having run -- its goroutine died with the prior process).
	var count int
	if err := sqlDB.QueryRow(`SELECT COUNT(*) FROM vms WHERE id = $1`, vmID).Scan(&count); err != nil {
		t.Fatalf("precondition query: %v", err)
	}
	if count != 0 {
		t.Fatalf("precondition failed: vms row already exists for %s", vmID)
	}

	manager := newStubVMManager(t)
	defer manager.Stop()

	reconcileOrphanedMigrationDests(sqlDB, vmBase, manager)

	var nodeID sql.NullString
	var state string
	if err := sqlDB.QueryRow(`SELECT node_id, state FROM vms WHERE id = $1`, vmID).Scan(&nodeID, &state); err != nil {
		t.Fatalf("reconcile did not create a vms row for the orphaned dest: %v", err)
	}
	if state != "running" {
		t.Fatalf("reconciled row state = %q, want %q", state, "running")
	}
	if !nodeID.Valid || nodeID.String == "" {
		t.Fatal("reconciled row has no node_id")
	}
	if _, err := manager.GetVM(vmID); err != nil {
		t.Fatalf("reconcile did not adopt the VM into the manager: %v", err)
	}
	t.Logf("PASS: orphaned dest %s adopted into the manager and registered in the DB (node_id=%s, state=%s)", vmID, nodeID.String, state)
}

// TestReconcileOrphanedMigrationDestsSkipsDeadProcesses proves the reconcile
// does not fabricate a DB row for a directory whose process is actually
// gone (a genuinely abandoned half-created directory, not novacron-05h's
// scenario) -- pidFileAlive correctly reports false and the directory is
// left alone.
func TestReconcileOrphanedMigrationDestsSkipsDeadProcesses(t *testing.T) {
	admin, err := sql.Open("postgres", pgAdminDSN())
	if err != nil {
		t.Skipf("skip: cannot open postgres admin connection: %v", err)
	}
	if err := admin.Ping(); err != nil {
		admin.Close()
		t.Skipf("skip: postgres not reachable: %v", err)
	}

	const dbName = "novacron_reconcile_orphan_dead_test"
	_, _ = admin.Exec(`DROP DATABASE IF EXISTS "` + dbName + `"`)
	if _, err := admin.Exec(`CREATE DATABASE "` + dbName + `"`); err != nil {
		admin.Close()
		t.Fatalf("create database: %v", err)
	}
	applyMigrations(t, dbDSN(dbName), dbName)

	sqlDB, err := sql.Open("postgres", dbDSN(dbName))
	if err != nil {
		admin.Close()
		t.Fatalf("open db: %v", err)
	}
	t.Cleanup(func() {
		sqlDB.Close()
		_, _ = admin.Exec(`DROP DATABASE IF EXISTS "` + dbName + `"`)
		admin.Close()
	})

	base := t.TempDir()
	vmBase := filepath.Join(base, "vms")
	vmID := uuid.NewString()
	vmDir := filepath.Join(vmBase, vmID)
	if err := os.MkdirAll(vmDir, 0755); err != nil {
		t.Fatalf("mkdir vmDir: %v", err)
	}
	cfg := core_vm.VMConfig{ID: vmID, Name: "abandoned", Type: core_vm.VMTypeKVM, MemoryMB: 256}
	cfgJSON, _ := json.Marshal(cfg)
	_ = os.WriteFile(filepath.Join(vmDir, "config.json"), cfgJSON, 0644)
	// An implausible pid: no process, alive or otherwise, should ever have it.
	_ = os.WriteFile(filepath.Join(vmDir, "qemu.pid"), []byte("2000000000"), 0644)

	manager := newStubVMManager(t)
	defer manager.Stop()

	reconcileOrphanedMigrationDests(sqlDB, vmBase, manager)

	var count int
	if err := sqlDB.QueryRow(`SELECT COUNT(*) FROM vms WHERE id = $1`, vmID).Scan(&count); err != nil {
		t.Fatalf("query: %v", err)
	}
	if count != 0 {
		t.Fatalf("reconcile created a vms row for a directory with no live process: count=%d", count)
	}
}
