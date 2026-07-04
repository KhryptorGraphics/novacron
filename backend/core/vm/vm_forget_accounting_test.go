package vm

import "testing"

// TestForgetVM_RetiresAndReleasesAccounting proves the source-node retirement on
// cross-node migration: forgetVM removes the VM from the manager AND releases this
// node's global CPU/memory reservation exactly once (a migrated-away VM no longer
// consumes source capacity). A baseline of one extra VM's reservation is seeded so
// a missing release (counter stays high) or a double release (drops below
// baseline) is visible rather than hidden by the zero clamp. A second forgetVM
// must be a no-op -- the claim-and-remove guard gates accounting on removal.
func TestForgetVM_RetiresAndReleasesAccounting(t *testing.T) {
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

	cfg := VMConfig{ID: "forget-test", Name: "forget", Type: VMTypeKVM, CPUShares: 2000, MemoryMB: 1024}
	vm, err := NewVM(cfg)
	if err != nil {
		t.Fatalf("NewVM: %v", err)
	}
	cpuDelta := cpuAllocationForConfig(cfg)
	memDelta := int64(cfg.MemoryMB)

	m.vmsMutex.Lock()
	m.vms[cfg.ID] = vm
	m.vmsMutex.Unlock()
	m.resourceMutex.Lock()
	m.allocatedCPU = cpuDelta * 2 // baseline: one other VM's reservation
	m.allocatedMemoryMB = memDelta * 2
	m.resourceMutex.Unlock()

	m.forgetVM(cfg.ID)
	m.forgetVM(cfg.ID) // idempotent: must not double-release

	m.vmsMutex.RLock()
	_, stillTracked := m.vms[cfg.ID]
	m.vmsMutex.RUnlock()
	if stillTracked {
		t.Fatalf("VM still tracked after forgetVM -- source not retired")
	}

	m.resourceMutex.Lock()
	gotCPU, gotMem := m.allocatedCPU, m.allocatedMemoryMB
	m.resourceMutex.Unlock()
	if gotCPU != cpuDelta {
		t.Fatalf("allocatedCPU=%d, want %d -- source accounting not released exactly once", gotCPU, cpuDelta)
	}
	if gotMem != memDelta {
		t.Fatalf("allocatedMemoryMB=%d, want %d -- source accounting not released exactly once", gotMem, memDelta)
	}
}

// TestRegisterMigrationPeer_Resolves proves peer address resolution is decoupled
// from the scheduler: a registered peer resolves by node id, an unknown id yields
// "", and (critically) registering a peer does NOT add a scheduler node -- which
// would activate CanAdmitVM and break local VM creation.
func TestRegisterMigrationPeer_Resolves(t *testing.T) {
	config := VMManagerConfig{
		DefaultDriver: VMTypeKVM,
		Drivers:       map[VMType]VMDriverConfigManager{VMTypeKVM: {Enabled: true, Config: map[string]interface{}{}}},
		Scheduler:     VMSchedulerConfig{Type: "default", Config: map[string]interface{}{}},
	}
	m, err := NewVMManager(config)
	if err != nil {
		t.Fatalf("NewVMManager: %v", err)
	}
	defer m.Shutdown()

	m.RegisterMigrationPeer("node2", "10.0.0.2:9090")
	if got := m.migrationPeerAddr("node2"); got != "10.0.0.2:9090" {
		t.Fatalf("migrationPeerAddr(node2)=%q, want 10.0.0.2:9090", got)
	}
	if got := m.migrationPeerAddr("unknown"); got != "" {
		t.Fatalf("migrationPeerAddr(unknown)=%q, want empty", got)
	}
	if nodes := m.ListSchedulerNodes(); len(nodes) != 0 {
		t.Fatalf("registering a migration peer leaked %d scheduler node(s) -- would activate admission control", len(nodes))
	}
}
