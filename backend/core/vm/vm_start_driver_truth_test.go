package vm

import (
	"context"
	"errors"
	"testing"
)

// These tests pin startVM's contract against the driver: a cached
// StateRunning is only trusted when the driver confirms it, because the
// manager's maintenance loop never refreshes the cache.

func newStartTruthManager(t *testing.T) *VMManager {
	t.Helper()
	mgr, err := NewVMManager(VMManagerConfig{})
	if err != nil {
		t.Fatalf("NewVMManager: %v", err)
	}
	return mgr
}

func newStartTruthVM(t *testing.T, id string, state State) *VM {
	t.Helper()
	vm, err := NewVM(VMConfig{
		ID:      id,
		Name:    id,
		Type:    VMTypeProcess,
		Command: "/bin/true",
	})
	if err != nil {
		t.Fatalf("NewVM: %v", err)
	}
	vm.SetState(state)
	return vm
}

// A cached Running VM whose process has exited must be started for real.
func TestStartTruthCachedRunningDriverStoppedStartsVM(t *testing.T) {
	const id = "vm-start-truth-stale"
	driver := newFakeCrashDriver()
	driver.flip(id, StateStopped)
	vm := newStartTruthVM(t, id, StateRunning)
	mgr := newStartTruthManager(t)

	resp, err := mgr.startVM(context.Background(), vm, driver)
	if err != nil {
		t.Fatalf("startVM: %v", err)
	}
	if !resp.Success {
		t.Fatalf("startVM success = false, error = %q", resp.ErrorMessage)
	}
	if starts, _, _ := driver.callCounts(id); starts != 1 {
		t.Fatalf("driver.Start calls = %d, want 1", starts)
	}
	if got := vm.State(); got != StateRunning {
		t.Fatalf("vm.State() = %s, want %s", got, StateRunning)
	}
}

// A cached Running VM the driver also reports running is already up: startVM
// is idempotent and does not restart it.
func TestStartTruthCachedRunningDriverRunningIsIdempotent(t *testing.T) {
	const id = "vm-start-truth-live"
	driver := newFakeCrashDriver()
	driver.flip(id, StateRunning)
	vm := newStartTruthVM(t, id, StateRunning)
	mgr := newStartTruthManager(t)

	resp, err := mgr.startVM(context.Background(), vm, driver)
	if err != nil {
		t.Fatalf("startVM: %v", err)
	}
	if !resp.Success {
		t.Fatalf("startVM success = false, error = %q", resp.ErrorMessage)
	}
	if starts, _, _ := driver.callCounts(id); starts != 0 {
		t.Fatalf("driver.Start calls = %d, want 0 (idempotent)", starts)
	}
	if got := vm.State(); got != StateRunning {
		t.Fatalf("vm.State() = %s, want %s", got, StateRunning)
	}
}

// If the driver cannot be asked, the cache is unverifiable: start anyway
// rather than claim a success that may not hold.
func TestStartTruthCachedRunningStatusErrorFallsThroughToStart(t *testing.T) {
	const id = "vm-start-truth-unreachable"
	driver := newFakeCrashDriver()
	driver.statusErr[id] = errors.New("driver unreachable")
	vm := newStartTruthVM(t, id, StateRunning)
	mgr := newStartTruthManager(t)

	resp, err := mgr.startVM(context.Background(), vm, driver)
	if err != nil {
		t.Fatalf("startVM: %v", err)
	}
	if !resp.Success {
		t.Fatalf("startVM success = false, error = %q", resp.ErrorMessage)
	}
	if starts, _, _ := driver.callCounts(id); starts != 1 {
		t.Fatalf("driver.Start calls = %d, want 1", starts)
	}
	if got := vm.State(); got != StateRunning {
		t.Fatalf("vm.State() = %s, want %s", got, StateRunning)
	}
}

// The ordinary path must keep working: a cached Stopped VM starts once.
func TestStartTruthCachedStoppedStartsVM(t *testing.T) {
	const id = "vm-start-truth-stopped"
	driver := newFakeCrashDriver()
	driver.flip(id, StateStopped)
	vm := newStartTruthVM(t, id, StateStopped)
	mgr := newStartTruthManager(t)

	resp, err := mgr.startVM(context.Background(), vm, driver)
	if err != nil {
		t.Fatalf("startVM: %v", err)
	}
	if !resp.Success {
		t.Fatalf("startVM success = false, error = %q", resp.ErrorMessage)
	}
	if starts, _, _ := driver.callCounts(id); starts != 1 {
		t.Fatalf("driver.Start calls = %d, want 1", starts)
	}
	if got := vm.State(); got != StateRunning {
		t.Fatalf("vm.State() = %s, want %s", got, StateRunning)
	}
}
