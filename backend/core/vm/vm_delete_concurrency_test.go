package vm

import (
	"context"
	"fmt"
	"sync"
	"testing"
)

// strictDeleteDriver models a real driver (like KVMDriverEnhanced) whose Stop and
// Delete error on a second call: Stop on an already-stopped VM returns "not
// running" and Delete on an already-gone VM returns "not found". A concurrent
// deleteVM that let more than one caller reach the driver would surface those as
// spurious failures -- so this driver makes the idempotency claim observable.
type strictDeleteDriver struct {
	CoreStubDriver
	mu      sync.Mutex
	stopped bool
	deleted bool
}

func (d *strictDeleteDriver) Stop(ctx context.Context, vmID string) error {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.stopped {
		return fmt.Errorf("VM %s is not running", vmID)
	}
	d.stopped = true
	return nil
}

func (d *strictDeleteDriver) Delete(ctx context.Context, vmID string) error {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.deleted {
		return fmt.Errorf("VM %s not found", vmID)
	}
	d.deleted = true
	return nil
}

// TestDeleteVM_ConcurrentIdempotent proves deleteVM is idempotent under
// concurrent deletes of the same running VM: exactly one caller performs the
// stop+delete+accounting, every caller returns success (no spurious "not
// running"/"not found"), the CPU/memory accounting is released exactly once, and
// the VM ends up removed from the manager.
func TestDeleteVM_ConcurrentIdempotent(t *testing.T) {
	m, err := NewVMManager(VMManagerConfig{
		DefaultDriver: VMTypeKVM,
		Drivers: map[VMType]VMDriverConfigManager{
			VMTypeKVM: {Enabled: true, Config: map[string]interface{}{}},
		},
		Scheduler: VMSchedulerConfig{Type: "default", Config: map[string]interface{}{}},
	})
	if err != nil {
		t.Fatalf("NewVMManager: %v", err)
	}
	defer m.Shutdown()

	cfg := VMConfig{ID: "del-conc", Name: "dc", Type: VMTypeKVM, CPUShares: 1000, MemoryMB: 512}
	vm, err := NewVM(cfg)
	if err != nil {
		t.Fatalf("NewVM: %v", err)
	}
	vm.mutex.Lock()
	vm.state = StateRunning // exercise the stop-before-delete path
	vm.mutex.Unlock()
	m.vmsMutex.Lock()
	m.vms[cfg.ID] = vm
	m.vmsMutex.Unlock()

	// Seed a second VM's worth of reservation so a double release drops below the
	// baseline (rather than being masked by the negative-value clamp).
	cpuDelta := cpuAllocationForConfig(cfg)
	memDelta := int64(cfg.MemoryMB)
	m.resourceMutex.Lock()
	m.allocatedCPU = cpuDelta * 2
	m.allocatedMemoryMB = memDelta * 2
	m.resourceMutex.Unlock()

	drv := &strictDeleteDriver{}
	const n = 8
	var wg sync.WaitGroup
	results := make([]*VMOperationResponse, n)
	errs := make([]error, n)
	start := make(chan struct{})
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			<-start
			results[i], errs[i] = m.deleteVM(context.Background(), vm, drv)
		}(i)
	}
	close(start)
	wg.Wait()

	for i := 0; i < n; i++ {
		if errs[i] != nil {
			t.Fatalf("concurrent delete %d errored (should be idempotent): %v", i, errs[i])
		}
		if results[i] == nil || !results[i].Success {
			t.Fatalf("concurrent delete %d not successful: %+v", i, results[i])
		}
	}

	m.resourceMutex.Lock()
	gotCPU, gotMem := m.allocatedCPU, m.allocatedMemoryMB
	m.resourceMutex.Unlock()
	if gotCPU != cpuDelta {
		t.Fatalf("allocatedCPU=%d, want %d -- accounting released not exactly once", gotCPU, cpuDelta)
	}
	if gotMem != memDelta {
		t.Fatalf("allocatedMemoryMB=%d, want %d -- accounting released not exactly once", gotMem, memDelta)
	}

	m.vmsMutex.Lock()
	_, still := m.vms[cfg.ID]
	m.vmsMutex.Unlock()
	if still {
		t.Fatalf("VM %s still tracked in manager after delete", cfg.ID)
	}
}
