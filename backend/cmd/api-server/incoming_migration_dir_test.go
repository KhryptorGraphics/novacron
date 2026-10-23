package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	core_vm "github.com/khryptorgraphics/novacron/backend/core/vm"
)

// The source node keeps <vmBase>/<id> after a successful migration, so a
// bounce-back finds a stale directory there. These cases pin when it is safe to
// replace it and when it belongs to something live.
func TestPrepareIncomingMigrationDir(t *testing.T) {
	const disk = "disk.qcow2"

	newDest := func(t *testing.T) string {
		t.Helper()
		return filepath.Join(t.TempDir(), "vm-1")
	}
	seedStaleDir := func(t *testing.T, dest string) {
		t.Helper()
		if err := os.MkdirAll(dest, 0o700); err != nil {
			t.Fatalf("seed dir: %v", err)
		}
		if err := os.WriteFile(filepath.Join(dest, disk), []byte("stale"), 0o600); err != nil {
			t.Fatalf("seed disk: %v", err)
		}
	}

	t.Run("fresh destination is created and marked", func(t *testing.T) {
		dest := newDest(t)
		if err := prepareIncomingMigrationDir(dest, false); err != nil {
			t.Fatalf("fresh dir: %v", err)
		}
		if !incomingMigrationMarkerExists(dest) {
			t.Fatal("fresh destination must carry the in-flight marker")
		}
	})

	t.Run("stale directory from a VM that left is replaced", func(t *testing.T) {
		dest := newDest(t)
		seedStaleDir(t, dest)
		if err := prepareIncomingMigrationDir(dest, false); err != nil {
			t.Fatalf("stale dir must be adoptable: %v", err)
		}
		if !incomingMigrationMarkerExists(dest) {
			t.Fatal("adopted destination must carry the in-flight marker")
		}
		if _, err := os.Lstat(filepath.Join(dest, disk)); !os.IsNotExist(err) {
			t.Fatalf("the stale disk should have been removed (stat err = %v)", err)
		}
	})

	t.Run("in-flight migration is a conflict and is not disturbed", func(t *testing.T) {
		dest := newDest(t)
		seedStaleDir(t, dest)
		if err := os.WriteFile(filepath.Join(dest, incomingMigrationMarker), nil, 0o600); err != nil {
			t.Fatalf("seed marker: %v", err)
		}
		err := prepareIncomingMigrationDir(dest, false)
		if err == nil || !strings.Contains(err.Error(), "already in progress") {
			t.Fatalf("err = %v, want an in-progress conflict", err)
		}
		if _, err := os.Lstat(filepath.Join(dest, disk)); err != nil {
			t.Fatalf("a concurrent migration's directory must not be touched: %v", err)
		}
	})

	t.Run("a VM that still exists here keeps its disk", func(t *testing.T) {
		dest := newDest(t)
		seedStaleDir(t, dest)
		err := prepareIncomingMigrationDir(dest, true)
		if err == nil || !strings.Contains(err.Error(), "already exists") {
			t.Fatalf("err = %v, want an already-exists conflict", err)
		}
		if _, err := os.Lstat(filepath.Join(dest, disk)); err != nil {
			t.Fatalf("a live VM's disk must not be deleted: %v", err)
		}
	})
}

func TestIncomingVMIsLiveOnlyWhenTheDriverRunsIt(t *testing.T) {
	vmCfg := core_vm.DefaultVMManagerConfig()
	vmCfg.Drivers[core_vm.VMTypeProcess] = core_vm.VMDriverConfigManager{
		Enabled: true, Config: map[string]interface{}{"base_path": t.TempDir()},
	}
	vmManager, err := core_vm.NewVMManager(vmCfg)
	if err != nil {
		t.Fatalf("new vm manager: %v", err)
	}
	defer vmManager.Stop()

	create := func(name string, args ...string) core_vm.CreateVMRequest {
		return core_vm.CreateVMRequest{
			Name:                  name,
			AllowMissingOwnership: true,
			Spec: core_vm.VMConfig{
				Type: core_vm.VMTypeProcess, Command: "/bin/sleep", Args: args, TenantID: "default",
			},
		}
	}
	stopped, err := vmManager.CreateVM(t.Context(), create("stopped-vm", "0"))
	if err != nil {
		t.Fatalf("create stopped vm: %v", err)
	}
	if incomingVMIsLive(vmManager, stopped.ID()) {
		t.Fatal("a created-but-not-started VM must not count as live")
	}

	running, err := vmManager.CreateVM(t.Context(), create("running-vm", "30"))
	if err != nil {
		t.Fatalf("create running vm: %v", err)
	}
	if err := vmManager.StartVM(t.Context(), running.ID()); err != nil {
		t.Fatalf("start vm: %v", err)
	}
	if !incomingVMIsLive(vmManager, running.ID()) {
		t.Fatal("a running VM must block replacing its directory")
	}
	if incomingVMIsLive(vmManager, "never-existed") {
		t.Fatal("an unknown VM must not count as live")
	}
}
