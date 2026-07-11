package vm

import (
	"context"
	"os/exec"
	"testing"
)

// TestProcessDriverLifecycle exercises the real create/start/stop/delete
// lifecycle of the process driver against an actual OS process.
func TestProcessDriverLifecycle(t *testing.T) {
	if _, err := exec.LookPath("sleep"); err != nil {
		t.Skip("sleep binary not available; skipping process lifecycle test")
	}

	ctx := context.Background()
	d, err := NewProcessDriver(map[string]interface{}{
		"node_id":   "test-node",
		"base_path": t.TempDir(),
	})
	if err != nil {
		t.Fatalf("NewProcessDriver: %v", err)
	}

	id, err := d.Create(ctx, VMConfig{Name: "sleeper", Command: "sleep", Args: []string{"30"}})
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	if id == "" {
		t.Fatal("Create returned an empty id")
	}

	// Created but not started -> stopped.
	if st, err := d.GetStatus(ctx, id); err != nil || st != StateStopped {
		t.Fatalf("status after create = %v (err %v), want %v", st, err, StateStopped)
	}

	if err := d.Start(ctx, id); err != nil {
		t.Fatalf("Start: %v", err)
	}
	if st, err := d.GetStatus(ctx, id); err != nil || st != StateRunning {
		t.Fatalf("status after start = %v (err %v), want %v", st, err, StateRunning)
	}

	info, err := d.GetInfo(ctx, id)
	if err != nil {
		t.Fatalf("GetInfo: %v", err)
	}
	if info.State != StateRunning || info.PID <= 0 {
		t.Fatalf("GetInfo = %+v, want running with pid > 0", info)
	}
	if info.Name != "sleeper" {
		t.Fatalf("GetInfo.Name = %q, want %q", info.Name, "sleeper")
	}

	vms, err := d.ListVMs(ctx)
	if err != nil {
		t.Fatalf("ListVMs: %v", err)
	}
	found := false
	for _, v := range vms {
		if v.ID == id {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("ListVMs did not include created VM %s", id)
	}

	if err := d.Stop(ctx, id); err != nil {
		t.Fatalf("Stop: %v", err)
	}
	if st, err := d.GetStatus(ctx, id); err != nil || st != StateStopped {
		t.Fatalf("status after stop = %v (err %v), want %v", st, err, StateStopped)
	}

	if err := d.Delete(ctx, id); err != nil {
		t.Fatalf("Delete: %v", err)
	}
	if _, err := d.GetStatus(ctx, id); err == nil {
		t.Fatal("status after delete: expected a not-found error, got nil")
	}
}

// TestProcessDriverCreateRequiresCommand verifies a process VM cannot be created
// without something to run.
func TestProcessDriverCreateRequiresCommand(t *testing.T) {
	d, err := NewProcessDriver(map[string]interface{}{"base_path": t.TempDir()})
	if err != nil {
		t.Fatalf("NewProcessDriver: %v", err)
	}
	if _, err := d.Create(context.Background(), VMConfig{Name: "nocmd"}); err == nil {
		t.Fatal("Create with no command: expected an error, got nil")
	}
}

// TestProcessFactoriesUnified proves the two driver factories no longer diverge:
// both the live-API path (NewProcessDriver, used by NewVMDriverFactory) and the
// init-time path (VMManager.createDriverForType) now build the same real driver
// instead of two different "not implemented" stubs.
func TestProcessFactoriesUnified(t *testing.T) {
	d1, err := NewProcessDriver(map[string]interface{}{"base_path": t.TempDir()})
	if err != nil || d1 == nil {
		t.Fatalf("NewProcessDriver: driver=%v err=%v; want a real driver", d1, err)
	}

	m := &VMManager{}
	d2, err := m.createDriverForType(VMTypeProcess, map[string]interface{}{"base_path": t.TempDir()})
	if err != nil || d2 == nil {
		t.Fatalf("createDriverForType(process): driver=%v err=%v; want a real driver", d2, err)
	}

	if _, ok := d1.(*ProcessDriver); !ok {
		t.Fatalf("NewProcessDriver returned %T, want *ProcessDriver", d1)
	}
	if _, ok := d2.(*ProcessDriver); !ok {
		t.Fatalf("createDriverForType returned %T, want *ProcessDriver", d2)
	}
}
