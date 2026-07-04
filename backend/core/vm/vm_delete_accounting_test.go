package vm

import (
	"context"
	"testing"
)

// TestDeleteVM_AccountingReleasedExactlyOnce proves the wave-2 manager fix: a
// repeated deleteVM (as happens with a driver whose Delete is idempotent, or
// two concurrent deletes racing past the m.vms lookup) must release the global
// CPU/memory accounting EXACTLY once. CoreStubDriver.Delete returns nil
// unconditionally, so without the claim-and-remove guard the second call
// double-subtracts the counters. A baseline of one extra VM's reservation is
// seeded so a double release drops below baseline rather than being hidden by
// the negative-value clamp.
func TestDeleteVM_AccountingReleasedExactlyOnce(t *testing.T) {
	config := VMManagerConfig{
		DefaultDriver: VMTypeKVM,
		Drivers: map[VMType]VMDriverConfigManager{
			VMTypeKVM: {Enabled: true, Config: map[string]interface{}{}},
		},
		Scheduler: VMSchedulerConfig{Type: "default", Config: map[string]interface{}{}},
	}
	m, err := NewVMManager(config)
	if err != nil {
		t.Fatalf("NewVMManager: %v", err)
	}
	defer m.Shutdown()

	cfg := VMConfig{ID: "acct-test", Name: "acct", Type: VMTypeKVM, CPUShares: 2000, MemoryMB: 1024}
	vm, err := NewVM(cfg)
	if err != nil {
		t.Fatalf("NewVM: %v", err)
	}
	vm.mutex.Lock()
	vm.state = StateCreated // not running -> deleteVM skips the stop path
	vm.mutex.Unlock()

	cpuDelta := cpuAllocationForConfig(cfg)
	memDelta := int64(cfg.MemoryMB)

	m.vmsMutex.Lock()
	m.vms[cfg.ID] = vm
	m.vmsMutex.Unlock()
	m.resourceMutex.Lock()
	m.allocatedCPU = cpuDelta * 2
	m.allocatedMemoryMB = memDelta * 2
	m.resourceMutex.Unlock()

	stub := &CoreStubDriver{} // Delete() returns nil -> the 2nd call is not gated

	if _, err := m.deleteVM(context.Background(), vm, stub); err != nil {
		t.Fatalf("first deleteVM: %v", err)
	}
	_, _ = m.deleteVM(context.Background(), vm, stub) // 2nd must be a no-op for accounting

	m.resourceMutex.Lock()
	gotCPU, gotMem := m.allocatedCPU, m.allocatedMemoryMB
	m.resourceMutex.Unlock()

	if gotCPU != cpuDelta {
		t.Fatalf("allocatedCPU=%d, want %d -- accounting released more than once", gotCPU, cpuDelta)
	}
	if gotMem != memDelta {
		t.Fatalf("allocatedMemoryMB=%d, want %d -- accounting released more than once", gotMem, memDelta)
	}
}
