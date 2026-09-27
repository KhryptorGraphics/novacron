package main

import (
	"context"
	"testing"
	"time"

	core_vm "github.com/khryptorgraphics/novacron/backend/core/vm"
)

func newTestVMManagerWithVM(t *testing.T, vmID string, state core_vm.State) *core_vm.VMManager {
	t.Helper()
	mgr, err := core_vm.NewVMManager(core_vm.VMManagerConfig{})
	if err != nil {
		t.Fatalf("NewVMManager: %v", err)
	}
	vm, err := core_vm.NewVM(core_vm.VMConfig{
		ID:         vmID,
		Name:       vmID,
		Type:       core_vm.VMTypeProcess,
		Command:    "/bin/true",
		CPUShares:  1024,
		MemoryMB:   512,
		DiskSizeGB: 4,
	})
	if err != nil {
		t.Fatalf("NewVM: %v", err)
	}
	mgr.AddVM(vm)
	vm.SetState(state)
	return mgr
}

func TestVMHealthSourceMissingVMIsError(t *testing.T) {
	mgr, err := core_vm.NewVMManager(core_vm.VMManagerConfig{})
	if err != nil {
		t.Fatalf("NewVMManager: %v", err)
	}
	source := vmHealthSource(mgr, nil)
	sample, err := source("does-not-exist")
	if err == nil {
		t.Fatal("expected an error for a missing VM, got nil")
	}
	if sample != nil {
		t.Fatalf("expected no sample for a missing VM, got %+v", sample)
	}
}

func TestVMHealthSourceRunningVMIsHealthy(t *testing.T) {
	mgr := newTestVMManagerWithVM(t, "vm-running", core_vm.StateRunning)
	source := vmHealthSource(mgr, nil)

	sample, err := source("vm-running")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if sample == nil || !sample.Healthy {
		t.Fatalf("expected a healthy sample, got %+v", sample)
	}
}

func TestVMHealthSourceFailedUntrackedIsUnhealthy(t *testing.T) {
	mgr := newTestVMManagerWithVM(t, "vm-failed", core_vm.StateFailed)
	// No supervisor at all: it will never act on this crash.
	source := vmHealthSource(mgr, nil)

	sample, err := source("vm-failed")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if sample == nil || sample.Healthy {
		t.Fatalf("expected an unhealthy sample for an untracked failed VM, got %+v", sample)
	}
}

func TestVMHealthSourceFailedButSupervisorWatchingIsNilSample(t *testing.T) {
	mgr := newTestVMManagerWithVM(t, "vm-failed-watched", core_vm.StateFailed)
	sup := core_vm.NewRestartSupervisor(mgr, nil, time.Hour) // long tick: no background loop ever started
	sup.RecordStart(context.Background(), "vm-failed-watched")
	source := vmHealthSource(mgr, sup)

	sample, err := source("vm-failed-watched")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if sample != nil {
		t.Fatalf("expected no sample while the supervisor is still watching/backing off, got %+v", sample)
	}
}

func TestVMHealthSourceRecordStoppedIsNilSample(t *testing.T) {
	mgr := newTestVMManagerWithVM(t, "vm-stopped", core_vm.StateFailed)
	sup := core_vm.NewRestartSupervisor(mgr, nil, time.Hour)
	ctx := context.Background()
	sup.RecordStart(ctx, "vm-stopped")
	sup.RecordStop(ctx, "vm-stopped")
	source := vmHealthSource(mgr, sup)

	sample, err := source("vm-stopped")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if sample != nil {
		t.Fatalf("expected no sample for a RecordStop'd VM, got %+v", sample)
	}
}

func TestVMHealthSourcePolicyNoIsNilSample(t *testing.T) {
	mgr := newTestVMManagerWithVM(t, "vm-policy-no", core_vm.StateFailed)
	sup := core_vm.NewRestartSupervisor(mgr, nil, time.Hour)
	ctx := context.Background()
	sup.SetRestartPolicy("vm-policy-no", core_vm.RestartPolicyNo)
	sup.RecordStart(ctx, "vm-policy-no")
	source := vmHealthSource(mgr, sup)

	sample, err := source("vm-policy-no")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if sample != nil {
		t.Fatalf("expected no sample for a policy=no VM, got %+v", sample)
	}
}

func TestVMHealthSourceOtherStatesAreNilSample(t *testing.T) {
	for _, state := range []core_vm.State{
		core_vm.StateStopped, core_vm.StatePaused, core_vm.StateRestarting,
		core_vm.StateMigrating, core_vm.StateCreating, core_vm.StateDeleting, core_vm.StateUnknown,
	} {
		t.Run(string(state), func(t *testing.T) {
			mgr := newTestVMManagerWithVM(t, "vm-other", state)
			source := vmHealthSource(mgr, nil)
			sample, err := source("vm-other")
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if sample != nil {
				t.Fatalf("expected no sample for state %q, got %+v", state, sample)
			}
		})
	}
}

func TestVMMigrationTargetSelectorExcludesSelfAndPicksBestFit(t *testing.T) {
	mgr := newTestVMManagerWithVM(t, "vm-migrate", core_vm.StateRunning)
	self := selfNodeID()

	selector := &vmMigrationTargetSelector{
		vmManager: mgr,
		candidates: func(excludeNode string) []drainCandidate {
			if excludeNode != self {
				t.Fatalf("expected candidates to be asked to exclude self-node %q, got %q", self, excludeNode)
			}
			// Deliberately include a too-small candidate first: the selector
			// must skip it and pick the one with enough capacity.
			return []drainCandidate{
				{NodeID: "too-small", LinkBps: 1e9, MemFreeMB: 1},
				{NodeID: "best-fit", LinkBps: 1e6, MemFreeMB: 4096},
			}
		},
	}

	target, err := selector.SelectTarget(context.Background(), "vm-migrate")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if target != "best-fit" {
		t.Fatalf("expected best-fit, got %q", target)
	}
}

func TestVMMigrationTargetSelectorNoCapacityErrors(t *testing.T) {
	mgr := newTestVMManagerWithVM(t, "vm-migrate2", core_vm.StateRunning)
	selector := &vmMigrationTargetSelector{
		vmManager: mgr,
		candidates: func(excludeNode string) []drainCandidate {
			return []drainCandidate{{NodeID: "too-small", MemFreeMB: 1}}
		},
	}

	if _, err := selector.SelectTarget(context.Background(), "vm-migrate2"); err == nil {
		t.Fatal("expected an error when no peer has capacity")
	}
}

func TestVMMigrationTargetSelectorMissingVMErrors(t *testing.T) {
	mgr, err := core_vm.NewVMManager(core_vm.VMManagerConfig{})
	if err != nil {
		t.Fatalf("NewVMManager: %v", err)
	}
	selector := &vmMigrationTargetSelector{
		vmManager:  mgr,
		candidates: func(excludeNode string) []drainCandidate { return nil },
	}
	if _, err := selector.SelectTarget(context.Background(), "does-not-exist"); err == nil {
		t.Fatal("expected an error for a missing VM")
	}
}
