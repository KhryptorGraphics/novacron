package vm

import (
	"sync"
	"testing"
)

// TestVMDriverFactory_ConcurrentAccess_Race reproduces novacron-t72: the
// factory closure returned by NewVMDriverFactory caches initialized drivers
// in a plain map with no synchronization. VMManager.getDriver invokes that
// closure on every CreateVM/StartVM/StopVM/DeleteVM/RestartVM/PauseVM/
// ResumeVM request with no manager-side lock either, so concurrent HTTP
// requests race on the cache map. In production that trips Go's runtime
// "fatal error: concurrent map writes/reads" -- an unrecoverable process
// crash that the api-server's panic-recovery middleware cannot catch.
//
// This test drives many goroutines through the same factory instance,
// hitting both the cache-miss (first init, map write) and cache-hit
// (cached read) paths concurrently for the same VM type. It also asserts
// the pre-existing lazy-init/single-instance contract still holds: every
// caller must observe an error-free result and the exact same cached
// *ContainerDriver instance.
//
// Run with the race detector: `go test -race -run TestVMDriverFactory_ConcurrentAccess_Race ./vm/`.
// Without a lock guarding the cache map, this reliably reports a DATA RACE.
// With the fix, it passes cleanly.
func TestVMDriverFactory_ConcurrentAccess_Race(t *testing.T) {
	config := DefaultVMDriverConfig("race-test-node")
	factory := NewVMDriverFactory(config)

	// VMTypeContainer's driver constructor (NewContainerDriver) always
	// succeeds without touching any external binary/daemon (docker, qemu,
	// containerd), so every goroutine here exercises the real cache
	// read/write path deterministically instead of racing against
	// environment-dependent driver-init failures.
	vmConfig := VMConfig{
		ID:   "race-vm",
		Name: "race-vm",
		Tags: map[string]string{"vm_type": string(VMTypeContainer)},
	}

	const goroutines = 200

	var (
		wg        sync.WaitGroup
		resultsMu sync.Mutex
		results   = make([]VMDriver, 0, goroutines)
	)

	wg.Add(goroutines)
	for range goroutines {
		go func() {
			defer wg.Done()
			driver, err := factory(vmConfig)
			if err != nil {
				t.Errorf("factory() error = %v", err)
				return
			}
			if driver == nil {
				t.Error("factory() returned nil driver with nil error")
				return
			}
			resultsMu.Lock()
			results = append(results, driver)
			resultsMu.Unlock()
		}()
	}
	wg.Wait()

	if len(results) != goroutines {
		t.Fatalf("expected %d successful results, got %d", goroutines, len(results))
	}

	// Preserve existing behavior: the factory must lazily init exactly once
	// per VM type and hand back the SAME cached driver instance every time.
	first := results[0]
	for i, d := range results[1:] {
		if d != first {
			t.Fatalf("result[%d] driver instance differs from result[0]; caching contract broken", i+1)
		}
	}
}

// TestVMDriverFactory_ConcurrentMixedTypes_Race exercises concurrent
// cache-miss writes for several distinct VM types at once, which stresses
// the map-write side of the race independently of the cache-hit path above.
func TestVMDriverFactory_ConcurrentMixedTypes_Race(t *testing.T) {
	config := DefaultVMDriverConfig("race-test-node-mixed")
	factory := NewVMDriverFactory(config)

	// Mix a type that always succeeds (container) with one that always
	// fails fast without external deps (process is "not yet implemented").
	// Both paths read-then-maybe-write the shared cache map, so failures
	// must not be allowed to skip the lock either.
	types := []VMType{VMTypeContainer, VMTypeProcess}

	const goroutines = 100
	var wg sync.WaitGroup
	wg.Add(goroutines)
	for i := range goroutines {
		vmType := types[i%len(types)]
		go func(vmType VMType) {
			defer wg.Done()
			vmConfig := VMConfig{
				ID:   "race-vm-mixed",
				Name: "race-vm-mixed",
				Tags: map[string]string{"vm_type": string(vmType)},
			}
			// Errors are expected for VMTypeProcess; only concurrent cache
			// map access under -race matters here.
			_, _ = factory(vmConfig)
		}(vmType)
	}
	wg.Wait()
}
