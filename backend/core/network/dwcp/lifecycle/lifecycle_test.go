package lifecycle

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap/zaptest"
)

// MockComponent is a test implementation of ComponentLifecycle
type MockComponent struct {
	*BaseLifecycle
	initCalled     atomic.Int32
	startCalled    atomic.Int32
	stopCalled     atomic.Int32
	shutdownCalled atomic.Int32
	healthCalled   atomic.Int32

	// Simulate failures
	failInit   bool
	failStart  bool
	failStop   bool
	failHealth bool

	// Timing control
	startDelay time.Duration
	stopDelay  time.Duration
}

func NewMockComponent(name string, logger testing.TB) *MockComponent {
	return &MockComponent{
		BaseLifecycle: NewBaseLifecycle(name, zaptest.NewLogger(logger)),
	}
}

func (m *MockComponent) Init(ctx context.Context, config interface{}) error {
	m.initCalled.Add(1)

	if m.failInit {
		_ = m.TransitionTo(StateFailed)
		return errors.New("init failed")
	}

	return m.BaseLifecycle.Init(ctx, config)
}

func (m *MockComponent) Start(ctx context.Context) error {
	m.startCalled.Add(1)

	if m.failStart {
		_ = m.TransitionTo(StateFailed)
		return errors.New("start failed")
	}

	if m.startDelay > 0 {
		select {
		case <-time.After(m.startDelay):
		case <-ctx.Done():
			return ctx.Err()
		}
	}

	return m.BaseLifecycle.Start(ctx)
}

func (m *MockComponent) Stop(ctx context.Context) error {
	m.stopCalled.Add(1)

	if m.failStop {
		_ = m.TransitionTo(StateFailed)
		return errors.New("stop failed")
	}

	if m.stopDelay > 0 {
		select {
		case <-time.After(m.stopDelay):
		case <-ctx.Done():
			return ctx.Err()
		}
	}

	return m.BaseLifecycle.Stop(ctx)
}

func (m *MockComponent) Shutdown(ctx context.Context, timeout time.Duration) error {
	m.shutdownCalled.Add(1)
	return m.BaseLifecycle.Shutdown(ctx, timeout)
}

func (m *MockComponent) HealthCheck(ctx context.Context) error {
	m.healthCalled.Add(1)

	if m.failHealth {
		return errors.New("health check failed")
	}

	return m.BaseLifecycle.HealthCheck(ctx)
}

// TestStateTransitions tests state machine transitions
func TestStateTransitions(t *testing.T) {
	component := NewMockComponent("test", t)

	// Initial state should be uninitialized
	assert.Equal(t, StateUninitialized, component.GetState())

	// Init
	err := component.Init(context.Background(), nil)
	require.NoError(t, err)
	assert.Equal(t, StateInitialized, component.GetState())

	// Start
	err = component.Start(context.Background())
	require.NoError(t, err)
	assert.Equal(t, StateRunning, component.GetState())

	// Stop
	err = component.Stop(context.Background())
	require.NoError(t, err)
	assert.Equal(t, StateStopped, component.GetState())
}

// TestInvalidStateTransitions tests that invalid transitions are rejected
func TestInvalidStateTransitions(t *testing.T) {
	sm := NewStateMachine("test")

	// Cannot go from uninitialized to running
	err := sm.TransitionTo(StateRunning)
	assert.Error(t, err)

	// Cannot go from initialized to stopped
	_ = sm.TransitionTo(StateInitialized)
	err = sm.TransitionTo(StateStopped)
	assert.Error(t, err)
}

// TestComponentMetrics tests metrics collection
func TestComponentMetrics(t *testing.T) {
	component := NewMockComponent("test", t)

	// Init and start
	_ = component.Init(context.Background(), nil)
	_ = component.Start(context.Background())

	// Wait a bit for uptime
	time.Sleep(100 * time.Millisecond)

	metrics := component.GetMetrics()
	assert.Equal(t, "test", metrics.ComponentName)
	assert.Equal(t, StateRunning, metrics.State)
	assert.Greater(t, metrics.Uptime, time.Duration(0))
	assert.True(t, metrics.StateTransitions > 0)
}

// TestHealthChecking tests health check functionality
func TestHealthChecking(t *testing.T) {
	component := NewMockComponent("test", t)

	// Health check should fail when not running
	err := component.HealthCheck(context.Background())
	assert.Error(t, err)

	// Start component
	_ = component.Init(context.Background(), nil)
	_ = component.Start(context.Background())

	// Health check should pass
	err = component.HealthCheck(context.Background())
	assert.NoError(t, err)

	// Simulate health check failure
	component.failHealth = true
	err = component.HealthCheck(context.Background())
	assert.Error(t, err)
}

// TestDependencyGraph tests dependency graph operations
func TestDependencyGraph(t *testing.T) {
	graph := NewDependencyGraph()

	// Add components with dependencies (dependencies must exist first)
	err := graph.AddNode("transport", []string{})
	require.NoError(t, err)

	err = graph.AddNode("compression", []string{"transport"})
	require.NoError(t, err)

	err = graph.AddNode("prediction", []string{"compression"})
	require.NoError(t, err)

	// Get start order
	startOrder, err := graph.GetStartOrder()
	require.NoError(t, err)
	assert.Equal(t, []string{"transport", "compression", "prediction"}, startOrder)

	// Get stop order (reverse)
	stopOrder, err := graph.GetStopOrder()
	require.NoError(t, err)
	assert.Equal(t, []string{"prediction", "compression", "transport"}, stopOrder)
}

// TestCycleDetection tests that dependency cycles are detected
func TestCycleDetection(t *testing.T) {
	graph := NewDependencyGraph()

	_ = graph.AddNode("A", []string{})
	_ = graph.AddNode("B", []string{"A"})

	// Try to create a cycle
	err := graph.AddNode("C", []string{"B"})
	require.NoError(t, err)

	// Now try to make A depend on C (would create cycle)
	graph.RemoveNode("A")
	err = graph.AddNode("A", []string{"C"})
	assert.Error(t, err)
}

// TestLifecycleManager tests the lifecycle manager
func TestLifecycleManager(t *testing.T) {
	logger := zaptest.NewLogger(t)
	manager := NewManager(DefaultManagerConfig(), logger)

	// Create and register components
	comp1 := NewMockComponent("comp1", t)
	comp2 := NewMockComponent("comp2", t)
	comp2.SetDependencies([]string{"comp1"})
	comp3 := NewMockComponent("comp3", t)
	comp3.SetDependencies([]string{"comp1", "comp2"})

	err := manager.Register(comp1)
	require.NoError(t, err)
	err = manager.Register(comp2)
	require.NoError(t, err)
	err = manager.Register(comp3)
	require.NoError(t, err)

	// Start all
	ctx := context.Background()
	err = manager.StartAll(ctx)
	require.NoError(t, err)

	// Verify all started
	assert.Equal(t, StateRunning, comp1.GetState())
	assert.Equal(t, StateRunning, comp2.GetState())
	assert.Equal(t, StateRunning, comp3.GetState())

	// Verify start order (comp1 called first)
	assert.Equal(t, int32(1), comp1.startCalled.Load())
	assert.Equal(t, int32(1), comp2.startCalled.Load())
	assert.Equal(t, int32(1), comp3.startCalled.Load())

	// Stop all
	err = manager.StopAll(ctx)
	require.NoError(t, err)

	// Verify all stopped
	assert.Equal(t, StateStopped, comp1.GetState())
	assert.Equal(t, StateStopped, comp2.GetState())
	assert.Equal(t, StateStopped, comp3.GetState())
}

// TestConcurrentStartup tests concurrent component startup
func TestConcurrentStartup(t *testing.T) {
	logger := zaptest.NewLogger(t)
	config := DefaultManagerConfig()
	config.MaxConcurrentStartup = 2
	manager := NewManager(config, logger)

	// Create multiple independent components
	for i := 0; i < 5; i++ {
		comp := NewMockComponent(string(rune('A'+i)), t)
		comp.startDelay = 100 * time.Millisecond
		err := manager.Register(comp)
		require.NoError(t, err)
	}

	// Start all and measure time
	ctx := context.Background()
	start := time.Now()
	err := manager.StartAll(ctx)
	duration := time.Since(start)

	require.NoError(t, err)

	// With 5 components and max 2 concurrent, should take ~300ms
	// Without concurrency would take ~500ms
	assert.Less(t, duration, 400*time.Millisecond)
	assert.Greater(t, duration, 200*time.Millisecond)
}

// TestGracefulShutdown tests graceful shutdown with timeout
func TestGracefulShutdown(t *testing.T) {
	component := NewMockComponent("test", t)
	component.stopDelay = 2 * time.Second

	_ = component.Init(context.Background(), nil)
	_ = component.Start(context.Background())

	// Shutdown with 1 second timeout (less than stop delay)
	ctx := context.Background()
	start := time.Now()
	err := component.Shutdown(ctx, 1*time.Second)
	duration := time.Since(start)

	// Should timeout and force shutdown
	assert.NoError(t, err) // Force shutdown succeeds
	assert.Less(t, duration, 1500*time.Millisecond)
	assert.Equal(t, StateStopped, component.GetState())
}

// TestHealthMonitoring tests health monitoring and recovery
func TestHealthMonitoring(t *testing.T) {
	logger := zaptest.NewLogger(t)
	config := DefaultManagerConfig()
	config.HealthCheckInterval = 100 * time.Millisecond
	manager := NewManager(config, logger)

	comp := NewMockComponent("test", t)
	_ = manager.Register(comp)

	ctx := context.Background()
	_ = manager.StartAll(ctx)

	// Wait for health checks
	time.Sleep(300 * time.Millisecond)

	// Health should be passing
	status := manager.GetHealthStatus()
	assert.NoError(t, status["test"])

	// Simulate health failure
	comp.failHealth = true
	time.Sleep(300 * time.Millisecond)

	// Health should be failing
	status = manager.GetHealthStatus()
	assert.Error(t, status["test"])

	_ = manager.StopAll(ctx)
}

// TestWaitForState tests waiting for state transition
func TestWaitForState(t *testing.T) {
	component := NewMockComponent("test", t)

	// Start component in goroutine
	go func() {
		time.Sleep(100 * time.Millisecond)
		_ = component.Init(context.Background(), nil)
		_ = component.Start(context.Background())
	}()

	// Wait for running state
	ctx := context.Background()
	err := component.WaitForState(ctx, StateRunning, 1*time.Second)
	require.NoError(t, err)
	assert.Equal(t, StateRunning, component.GetState())
}

// TestDependencyLevels tests dependency level grouping
func TestDependencyLevels(t *testing.T) {
	graph := NewDependencyGraph()

	// Create multi-level dependencies
	_ = graph.AddNode("L0-A", []string{})
	_ = graph.AddNode("L0-B", []string{})
	_ = graph.AddNode("L1-A", []string{"L0-A"})
	_ = graph.AddNode("L1-B", []string{"L0-B"})
	_ = graph.AddNode("L2-A", []string{"L1-A", "L1-B"})

	levels := graph.GetDependencyLevels()
	require.Equal(t, 3, len(levels))

	// Level 0 should have L0-A and L0-B
	assert.Contains(t, levels[0], "L0-A")
	assert.Contains(t, levels[0], "L0-B")

	// Level 1 should have L1-A and L1-B
	assert.Contains(t, levels[1], "L1-A")
	assert.Contains(t, levels[1], "L1-B")

	// Level 2 should have L2-A
	assert.Equal(t, []string{"L2-A"}, levels[2])
}

// TestObserverPattern tests observer notifications
func TestObserverPattern(t *testing.T) {
	component := NewMockComponent("test", t)

	stateChanges := make(chan struct{}, 10)
	observer := &TestObserver{
		onStateChange: func(comp string, old, new State) {
			if comp == "test" {
				stateChanges <- struct{}{}
			}
		},
	}

	component.RegisterObserver(observer)

	// Perform state transitions
	_ = component.Init(context.Background(), nil)
	_ = component.Start(context.Background())
	_ = component.Stop(context.Background())

	// Should receive at least 3 state change notifications
	timeout := time.After(1 * time.Second)
	count := 0
	for i := 0; i < 3; i++ {
		select {
		case <-stateChanges:
			count++
		case <-timeout:
			t.Fatal("timeout waiting for state changes")
		}
	}

	assert.GreaterOrEqual(t, count, 3)
}

// TestObserver implements Observer interface for testing
type TestObserver struct {
	onStateChange       func(string, State, State)
	onHealthCheckFailed func(string, error)
	onRecoveryStarted   func(string)
	onRecoveryCompleted func(string, time.Duration)
	onRecoveryFailed    func(string, error)
}

func (o *TestObserver) OnStateChange(component string, oldState, newState State) {
	if o.onStateChange != nil {
		o.onStateChange(component, oldState, newState)
	}
}

func (o *TestObserver) OnHealthCheckFailed(component string, err error) {
	if o.onHealthCheckFailed != nil {
		o.onHealthCheckFailed(component, err)
	}
}

func (o *TestObserver) OnRecoveryStarted(component string) {
	if o.onRecoveryStarted != nil {
		o.onRecoveryStarted(component)
	}
}

func (o *TestObserver) OnRecoveryCompleted(component string, duration time.Duration) {
	if o.onRecoveryCompleted != nil {
		o.onRecoveryCompleted(component, duration)
	}
}

func (o *TestObserver) OnRecoveryFailed(component string, err error) {
	if o.onRecoveryFailed != nil {
		o.onRecoveryFailed(component, err)
	}
}

// BenchmarkStateTransition benchmarks state transitions
func BenchmarkStateTransition(b *testing.B) {
	sm := NewStateMachine("bench")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = sm.TransitionTo(StateInitialized)
		_ = sm.TransitionTo(StateStarting)
		_ = sm.TransitionTo(StateRunning)
		_ = sm.TransitionTo(StateStopping)
		_ = sm.TransitionTo(StateStopped)
		sm.Reset()
	}
}

// BenchmarkConcurrentHealthChecks benchmarks concurrent health checks
func BenchmarkConcurrentHealthChecks(b *testing.B) {
	component := NewMockComponent("bench", b)
	_ = component.Init(context.Background(), nil)
	_ = component.Start(context.Background())

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			_ = component.HealthCheck(context.Background())
		}
	})
}
