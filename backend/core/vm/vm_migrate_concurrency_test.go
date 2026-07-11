package vm

import (
	"context"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// blockingMigrateDriver is a mock VMDriver whose Migrate blocks until released,
// so two concurrent migrateVM calls on the same VM can be observed racing. It
// embeds CoreStubDriver for the rest of the interface and reports migration
// support (the stub does not) so migrateVM proceeds past its capability check.
type blockingMigrateDriver struct {
	CoreStubDriver
	entered chan struct{} // buffered(1): signaled once the winner is inside Migrate
	release chan struct{} // closed by the test to let the winner's Migrate return
	calls   int32
}

func (d *blockingMigrateDriver) SupportsMigrate() bool { return true }

func (d *blockingMigrateDriver) Migrate(ctx context.Context, vmID, target string, params map[string]string) error {
	atomic.AddInt32(&d.calls, 1)
	select {
	case d.entered <- struct{}{}:
	default:
	}
	<-d.release
	return nil
}

// TestMigrateVM_ConcurrentSameVMRejected proves the same-VM migration
// concurrency guard: while one migration of a VM is in flight (StateMigrating),
// a second concurrent migrateVM on the SAME VM must be cleanly rejected --
// Success=false, no error, and driver.Migrate never entered a second time (which
// would be a competing QMP session) -- with no corrupted state.
func TestMigrateVM_ConcurrentSameVMRejected(t *testing.T) {
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

	cfg := VMConfig{ID: "mig-conc", Name: "mc", Type: VMTypeKVM, CPUShares: 1000, MemoryMB: 512}
	vm, err := NewVM(cfg)
	if err != nil {
		t.Fatalf("NewVM: %v", err)
	}
	vm.mutex.Lock()
	vm.state = StateRunning
	vm.mutex.Unlock()
	m.vmsMutex.Lock()
	m.vms[cfg.ID] = vm
	m.vmsMutex.Unlock()

	drv := &blockingMigrateDriver{entered: make(chan struct{}, 1), release: make(chan struct{})}
	// Explicit uri skips resolveMigrationURI (no real target node needed).
	params := map[string]string{"target_node": "node-b", "uri": "tcp:127.0.0.1:59999"}

	var wg sync.WaitGroup
	var firstResp *VMOperationResponse
	var firstErr error

	// Start the first (winning) migration and wait until it is inside driver.Migrate,
	// i.e. the VM is claimed as StateMigrating.
	wg.Add(1)
	go func() {
		defer wg.Done()
		firstResp, firstErr = m.migrateVM(context.Background(), vm, drv, params)
	}()
	select {
	case <-drv.entered:
	case <-time.After(5 * time.Second):
		close(drv.release)
		t.Fatal("first migration never reached driver.Migrate")
	}

	// Second concurrent migration on the same VM: must be rejected immediately.
	secondResp, secondErr := m.migrateVM(context.Background(), vm, drv, params)

	// Let the first migration complete, then join.
	close(drv.release)
	wg.Wait()

	// The rejected (second) call: clean rejection, not a system error.
	if secondErr != nil {
		t.Fatalf("second migrate returned error, want clean rejection: %v", secondErr)
	}
	if secondResp == nil || secondResp.Success {
		t.Fatalf("second concurrent migrate must be rejected (Success=false), got %+v", secondResp)
	}
	if !strings.Contains(strings.ToLower(secondResp.ErrorMessage), "migrat") {
		t.Fatalf("rejection message %q should say migration is already in progress", secondResp.ErrorMessage)
	}

	// The first call: succeeds.
	if firstErr != nil {
		t.Fatalf("first migrate errored: %v", firstErr)
	}
	if firstResp == nil || !firstResp.Success {
		t.Fatalf("first migrate should succeed, got %+v", firstResp)
	}

	// driver.Migrate must have run exactly once -- the second was rejected before
	// ever reaching the driver (no competing QMP session).
	if got := atomic.LoadInt32(&drv.calls); got != 1 {
		t.Fatalf("driver.Migrate called %d times, want 1 (second must be rejected before the driver)", got)
	}
}
