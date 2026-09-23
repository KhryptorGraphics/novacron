package vm

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"
)

// fakeCrashDriver is a fake compute-driver harness for the restart
// supervisor: tests flip process state (crash injection), fail Start/Create
// on demand, and count lifecycle calls. Unrelated VMDriver behavior comes
// from the shared MockHypervisor.
type fakeCrashDriver struct {
	*MockHypervisor

	mu          sync.Mutex
	status      map[string]VMState
	statusErr   map[string]error
	starts      map[string]int
	stops       map[string]int
	creates     map[string]int
	createdCfg  map[string]VMConfig
	failStart   bool
	failCreate  bool
	unknownLeft map[string]int // next N Start calls fail with "not found"
}

func newFakeCrashDriver() *fakeCrashDriver {
	return &fakeCrashDriver{
		MockHypervisor: NewMockHypervisor("test-node", "fake"),
		status:         make(map[string]VMState),
		statusErr:      make(map[string]error),
		starts:         make(map[string]int),
		stops:          make(map[string]int),
		creates:        make(map[string]int),
		createdCfg:     make(map[string]VMConfig),
		unknownLeft:    make(map[string]int),
	}
}

func (f *fakeCrashDriver) Create(ctx context.Context, config VMConfig) (string, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.creates[config.ID]++
	f.createdCfg[config.ID] = config
	if f.failCreate {
		return "", fmt.Errorf("create failed")
	}
	f.status[config.ID] = StateStopped
	return config.ID, nil
}

func (f *fakeCrashDriver) Start(ctx context.Context, vmID string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.starts[vmID]++
	if f.unknownLeft[vmID] > 0 {
		f.unknownLeft[vmID]--
		return fmt.Errorf("VM %s not found", vmID)
	}
	if f.failStart {
		return fmt.Errorf("start failed")
	}
	f.status[vmID] = StateRunning
	return nil
}

func (f *fakeCrashDriver) Stop(ctx context.Context, vmID string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.stops[vmID]++
	if _, ok := f.status[vmID]; !ok {
		return fmt.Errorf("VM %s not running", vmID)
	}
	f.status[vmID] = StateStopped
	return nil
}

func (f *fakeCrashDriver) GetStatus(ctx context.Context, vmID string) (VMState, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if err, ok := f.statusErr[vmID]; ok {
		return StateUnknown, err
	}
	return f.status[vmID], nil
}

func (f *fakeCrashDriver) flip(vmID string, st VMState) {
	f.mu.Lock()
	f.status[vmID] = st
	f.mu.Unlock()
}

func (f *fakeCrashDriver) callCounts(vmID string) (starts, stops, creates int) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.starts[vmID], f.stops[vmID], f.creates[vmID]
}

// fakeClock drives the supervisor's backoff timeline deterministically.
type fakeClock struct {
	mu sync.Mutex
	t  time.Time
}

func newFakeClock() *fakeClock {
	return &fakeClock{t: time.Date(2026, 9, 22, 12, 0, 0, 0, time.UTC)}
}

func (c *fakeClock) now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.t
}

func (c *fakeClock) advance(d time.Duration) {
	c.mu.Lock()
	c.t = c.t.Add(d)
	c.mu.Unlock()
}

// newSupervisedVM builds a manager with one running VM backed by fake, plus a
// supervisor on a 5ms real tick driven by clock.
func newSupervisedVM(t *testing.T, id string, fake *fakeCrashDriver, clock *fakeClock) (*VMManager, *RestartSupervisor) {
	t.Helper()
	mgr, err := NewVMManager(VMManagerConfig{})
	if err != nil {
		t.Fatalf("NewVMManager: %v", err)
	}
	mgr.driverFactory = func(config VMConfig) (VMDriver, error) { return fake, nil }

	vm, err := NewVM(VMConfig{
		ID:         id,
		Name:       id,
		Type:       VMTypeProcess,
		Command:    "/bin/true",
		CPUShares:  1024,
		MemoryMB:   512,
		DiskSizeGB: 4,
	})
	if err != nil {
		t.Fatalf("NewVM: %v", err)
	}
	mgr.AddVM(vm)
	vm.SetState(StateRunning)
	fake.flip(id, StateRunning)

	sup := NewRestartSupervisor(mgr, nil, 5*time.Millisecond)
	sup.now = clock.now
	sup.stableAfter = time.Minute
	return mgr, sup
}

func waitFor(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(2 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s", what)
}

func waitForState(t *testing.T, sup *RestartSupervisor, vmID, state string) {
	t.Helper()
	waitFor(t, "state="+state, func() bool {
		st, ok := sup.Inspect(context.Background(), vmID)
		return ok && st.State == state
	})
}

func waitForStarts(t *testing.T, fake *fakeCrashDriver, vmID string, n int) {
	t.Helper()
	waitFor(t, fmt.Sprintf("starts>=%d", n), func() bool {
		s, _, _ := fake.callCounts(vmID)
		return s >= n
	})
}

func TestRestartSupervisorRestartsCrashedVM(t *testing.T) {
	ctx := context.Background()
	fake := newFakeCrashDriver()
	clock := newFakeClock()
	mgr, sup := newSupervisedVM(t, "vm-crash", fake, clock)
	_ = mgr

	sup.RecordStart(ctx, "vm-crash")
	if err := sup.Start(ctx); err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer sup.Stop()

	// Healthy VM: no restarts even after many ticks.
	clock.advance(10 * time.Second)
	time.Sleep(30 * time.Millisecond)
	if n, _, _ := fake.callCounts("vm-crash"); n != 0 {
		t.Fatalf("healthy VM restarted %d times", n)
	}

	// Crash the process; fib backoff attempt 1 is 2s of supervisor time.
	fake.flip("vm-crash", StateStopped)
	waitForState(t, sup, "vm-crash", restartStateBackingOff)
	clock.advance(3 * time.Second)
	waitForStarts(t, fake, "vm-crash", 1)

	status, ok := sup.Inspect(ctx, "vm-crash")
	if !ok {
		t.Fatal("Inspect did not find supervised VM")
	}
	if status.Attempts != 1 {
		t.Fatalf("attempts = %d, want 1", status.Attempts)
	}
	if status.State != restartStateWatching {
		t.Fatalf("state = %q, want %q after successful restart", status.State, restartStateWatching)
	}
	if status.Policy != RestartPolicyOnFailure {
		t.Fatalf("policy = %q, want default %q", status.Policy, RestartPolicyOnFailure)
	}
	if fake.status["vm-crash"] != StateRunning {
		t.Fatalf("fake driver state = %q, want running", fake.status["vm-crash"])
	}
}

func TestRestartSupervisorPolicyNoNeverRestarts(t *testing.T) {
	ctx := context.Background()
	fake := newFakeCrashDriver()
	clock := newFakeClock()
	_, sup := newSupervisedVM(t, "vm-no", fake, clock)

	sup.SetRestartPolicy("vm-no", RestartPolicyNo)
	sup.RecordStart(ctx, "vm-no")
	if err := sup.Start(ctx); err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer sup.Stop()

	fake.flip("vm-no", StateStopped)
	clock.advance(300 * time.Second)
	time.Sleep(50 * time.Millisecond)

	if n, _, _ := fake.callCounts("vm-no"); n != 0 {
		t.Fatalf("policy=no VM restarted %d times", n)
	}
	status, _ := sup.Inspect(ctx, "vm-no")
	if status.Attempts != 0 || status.State != restartStateWatching {
		t.Fatalf("policy=no: attempts=%d state=%q, want 0/%s",
			status.Attempts, status.State, restartStateWatching)
	}
}

func TestRestartSupervisorEnvDefaultPolicy(t *testing.T) {
	t.Setenv(RestartEnvVar, "no")
	ctx := context.Background()
	fake := newFakeCrashDriver()
	clock := newFakeClock()
	_, sup := newSupervisedVM(t, "vm-env", fake, clock)

	sup.RecordStart(ctx, "vm-env")
	if err := sup.Start(ctx); err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer sup.Stop()

	fake.flip("vm-env", StateStopped)
	clock.advance(300 * time.Second)
	time.Sleep(50 * time.Millisecond)

	if n, _, _ := fake.callCounts("vm-env"); n != 0 {
		t.Fatalf("env policy=no: restarted %d times", n)
	}
	status, _ := sup.Inspect(ctx, "vm-env")
	if status.Policy != RestartPolicyNo {
		t.Fatalf("policy = %q, want env %q", status.Policy, RestartPolicyNo)
	}
}

func TestRestartSupervisorRecreatesVMWithSameConfig(t *testing.T) {
	ctx := context.Background()
	fake := newFakeCrashDriver()
	clock := newFakeClock()
	mgr, sup := newSupervisedVM(t, "vm-recreate", fake, clock)
	vm, err := mgr.GetVM("vm-recreate")
	if err != nil {
		t.Fatalf("GetVM: %v", err)
	}
	want := vm.Config()

	// First Start after the crash fails "not found": supervisor must recreate
	// with the original config and start it.
	fake.mu.Lock()
	fake.unknownLeft["vm-recreate"] = 1
	fake.mu.Unlock()

	sup.RecordStart(ctx, "vm-recreate")
	if err := sup.Start(ctx); err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer sup.Stop()

	fake.flip("vm-recreate", StateStopped)
	waitForState(t, sup, "vm-recreate", restartStateBackingOff)
	clock.advance(3 * time.Second)
	waitFor(t, "recreate path", func() bool {
		_, _, c := fake.callCounts("vm-recreate")
		return c >= 1
	})

	fake.mu.Lock()
	got := fake.createdCfg["vm-recreate"]
	fake.mu.Unlock()
	if got.ID != want.ID || got.Name != want.Name || got.CPUShares != want.CPUShares || got.MemoryMB != want.MemoryMB {
		t.Fatalf("recreate used wrong config: got %+v want %+v", got, want)
	}
	status, _ := sup.Inspect(ctx, "vm-recreate")
	if status.Attempts != 1 || status.State != restartStateWatching {
		t.Fatalf("after recreate: attempts=%d state=%q", status.Attempts, status.State)
	}
	if fake.status["vm-recreate"] != StateRunning {
		t.Fatalf("fake state = %q, want running", fake.status["vm-recreate"])
	}
}

func TestRestartSupervisorPermanentFailureAfterMaxAttempts(t *testing.T) {
	ctx := context.Background()
	fake := newFakeCrashDriver()
	fake.failStart = true
	fake.failCreate = true
	clock := newFakeClock()
	_, sup := newSupervisedVM(t, "vm-doomed", fake, clock)

	sup.RecordStart(ctx, "vm-doomed")
	if err := sup.Start(ctx); err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer sup.Stop()

	fake.flip("vm-doomed", StateStopped)
	waitForState(t, sup, "vm-doomed", restartStateBackingOff)
	// Each attempt becomes due after its fib backoff; walk all 8 deterministically.
	for attempt := 1; attempt <= restartMaxAttempts; attempt++ {
		clock.advance(60 * time.Second) // comfortably past every remaining backoff
		waitForStarts(t, fake, "vm-doomed", attempt)
	}
	waitForState(t, sup, "vm-doomed", restartStatePermanentFailure)

	status, _ := sup.Inspect(ctx, "vm-doomed")
	if status.Attempts != restartMaxAttempts {
		t.Fatalf("attempts = %d, want %d", status.Attempts, restartMaxAttempts)
	}
	if status.LastError == "" {
		t.Fatal("permanent failure carried no last error")
	}

	// Once permanent-failure, further observation changes nothing.
	nBefore, _, _ := fake.callCounts("vm-doomed")
	clock.advance(600 * time.Second)
	time.Sleep(50 * time.Millisecond)
	if n, _, _ := fake.callCounts("vm-doomed"); n != nBefore {
		t.Fatalf("permanent-failure VM restarted again: %d -> %d", nBefore, n)
	}
}

func TestRestartSupervisorRecordStopSuppressesAlways(t *testing.T) {
	ctx := context.Background()
	fake := newFakeCrashDriver()
	clock := newFakeClock()
	_, sup := newSupervisedVM(t, "vm-stopped", fake, clock)

	sup.SetRestartPolicy("vm-stopped", RestartPolicyAlways)
	sup.RecordStart(ctx, "vm-stopped")
	if err := sup.Start(ctx); err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer sup.Stop()

	// Intentional stop: even policy "always" must not restart it.
	fake.flip("vm-stopped", StateStopped)
	sup.RecordStop(ctx, "vm-stopped")
	clock.advance(300 * time.Second)
	time.Sleep(50 * time.Millisecond)

	if n, _, _ := fake.callCounts("vm-stopped"); n != 0 {
		t.Fatalf("RecordStop'd VM restarted %d times under policy=always", n)
	}
	status, _ := sup.Inspect(ctx, "vm-stopped")
	if status.State != restartStateStopped {
		t.Fatalf("state = %q, want %q", status.State, restartStateStopped)
	}

	// A fresh RecordStart re-arms supervision.
	sup.RecordStart(ctx, "vm-stopped")
	fake.flip("vm-stopped", StateStopped)
	waitForState(t, sup, "vm-stopped", restartStateBackingOff)
	clock.advance(3 * time.Second)
	waitForStarts(t, fake, "vm-stopped", 1)
}

func TestRestartSupervisorLifecycleIdempotentAndRaceSafe(t *testing.T) {
	ctx := context.Background()
	fake := newFakeCrashDriver()
	clock := newFakeClock()
	_, sup := newSupervisedVM(t, "vm-life", fake, clock)

	if err := sup.Start(ctx); err != nil {
		t.Fatalf("Start: %v", err)
	}
	if err := sup.Start(ctx); err != nil {
		t.Fatalf("second Start: %v", err)
	}

	// Concurrent lifecycle churn and churn of records from "callers".
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			id := fmt.Sprintf("vm-life-%d", i)
			for j := 0; j < 50; j++ {
				sup.RecordStart(ctx, id)
				sup.SetRestartPolicy(id, RestartPolicyAlways)
				sup.Inspect(ctx, id)
				_ = sup.Statuses(ctx)
				sup.RecordStop(ctx, id)
			}
		}(i)
	}
	wg.Wait()

	sup.Stop()
	sup.Stop() // second Stop is a no-op

	if _, ok := sup.Inspect(ctx, "never-seen"); ok {
		t.Fatal("Inspect found a VM that was never supervised")
	}
}

func TestRestartSupervisorBackoffSequence(t *testing.T) {
	want := []time.Duration{2, 2, 4, 6, 10, 16, 26, 42}
	for i, w := range want {
		if got := restartBackoff(i + 1); got != w*time.Second {
			t.Fatalf("backoff(%d) = %v, want %v", i+1, got, w*time.Second)
		}
	}
}
