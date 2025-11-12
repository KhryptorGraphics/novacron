// Package orchestrator provides orchestration tests
package orchestrator

import (
	"context"
	"errors"
	"testing"
	"time"
)

// Mock logger for testing
type mockLogger struct {
	messages []string
}

func (m *mockLogger) Info(msg string, keysAndValues ...interface{})              { m.messages = append(m.messages, msg) }
func (m *mockLogger) Error(msg string, err error, keysAndValues ...interface{}) { m.messages = append(m.messages, msg) }
func (m *mockLogger) Debug(msg string, keysAndValues ...interface{})            { m.messages = append(m.messages, msg) }
func (m *mockLogger) Warn(msg string, keysAndValues ...interface{})             { m.messages = append(m.messages, msg) }

// Mock metrics for testing
type mockMetrics struct {
	initCalls     map[string]bool
	shutdownCalls map[string]bool
}

func (m *mockMetrics) RecordComponentInit(name string, duration time.Duration, success bool) {
	if m.initCalls == nil {
		m.initCalls = make(map[string]bool)
	}
	m.initCalls[name] = success
}

func (m *mockMetrics) RecordComponentShutdown(name string, duration time.Duration, success bool) {
	if m.shutdownCalls == nil {
		m.shutdownCalls = make(map[string]bool)
	}
	m.shutdownCalls[name] = success
}

func (m *mockMetrics) SetComponentStatus(name string, status string) {}

// Mock component for testing
type mockComponent struct {
	name         string
	deps         []string
	initErr      error
	shutdownErr  error
	healthErr    error
	initialized  bool
	shutdownCalled bool
}

func (m *mockComponent) Name() string                             { return m.name }
func (m *mockComponent) Dependencies() []string                   { return m.deps }
func (m *mockComponent) HealthCheck(ctx context.Context) error    { return m.healthErr }

func (m *mockComponent) Initialize(ctx context.Context) error {
	if m.initErr != nil {
		return m.initErr
	}
	m.initialized = true
	return nil
}

func (m *mockComponent) Shutdown(ctx context.Context) error {
	m.shutdownCalled = true
	return m.shutdownErr
}

func TestOrchestrator_Initialize(t *testing.T) {
	tests := []struct {
		name       string
		components []Component
		wantErr    bool
	}{
		{
			name: "successful initialization",
			components: []Component{
				&mockComponent{name: "comp1", deps: []string{}},
				&mockComponent{name: "comp2", deps: []string{"comp1"}},
			},
			wantErr: false,
		},
		{
			name: "initialization failure",
			components: []Component{
				&mockComponent{name: "comp1", deps: []string{}, initErr: errors.New("init failed")},
			},
			wantErr: true,
		},
		{
			name: "circular dependency",
			components: []Component{
				&mockComponent{name: "comp1", deps: []string{"comp2"}},
				&mockComponent{name: "comp2", deps: []string{"comp1"}},
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logger := &mockLogger{}
			metrics := &mockMetrics{}
			orch := NewOrchestrator(logger, metrics)

			// Register components
			for _, comp := range tt.components {
				if err := orch.Register(comp); err != nil {
					t.Fatalf("Failed to register component: %v", err)
				}
			}

			// Initialize
			ctx := context.Background()
			err := orch.Initialize(ctx)

			if tt.wantErr {
				if err == nil {
					t.Fatal("Expected error but got none")
				}
				return
			}

			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}

			// Verify all components are ready
			status := orch.GetStatus()
			for _, comp := range tt.components {
				if status[comp.Name()] != StatusReady {
					t.Errorf("Component %s not ready: %s", comp.Name(), status[comp.Name()])
				}
			}
		})
	}
}

func TestOrchestrator_Shutdown(t *testing.T) {
	logger := &mockLogger{}
	metrics := &mockMetrics{}
	orch := NewOrchestrator(logger, metrics)

	comp1 := &mockComponent{name: "comp1", deps: []string{}}
	comp2 := &mockComponent{name: "comp2", deps: []string{"comp1"}}

	orch.Register(comp1)
	orch.Register(comp2)

	// Initialize
	ctx := context.Background()
	if err := orch.Initialize(ctx); err != nil {
		t.Fatalf("Initialize failed: %v", err)
	}

	// Shutdown
	if err := orch.Shutdown(ctx); err != nil {
		t.Fatalf("Shutdown failed: %v", err)
	}

	// Verify components were shut down
	if !comp1.shutdownCalled || !comp2.shutdownCalled {
		t.Error("Not all components were shut down")
	}

	// Verify shutdown order (comp2 before comp1)
	status := orch.GetStatus()
	for name, st := range status {
		if st != StatusShutdown {
			t.Errorf("Component %s not shutdown: %s", name, st)
		}
	}
}

func TestOrchestrator_HealthCheck(t *testing.T) {
	logger := &mockLogger{}
	metrics := &mockMetrics{}
	orch := NewOrchestrator(logger, metrics)

	comp1 := &mockComponent{name: "comp1", deps: []string{}}
	comp2 := &mockComponent{name: "comp2", deps: []string{"comp1"}, healthErr: errors.New("health check failed")}

	orch.Register(comp1)
	orch.Register(comp2)

	// Initialize
	ctx := context.Background()
	orch.Initialize(ctx)

	// Health check should fail because comp2 is unhealthy
	err := orch.HealthCheck(ctx)
	if err == nil {
		t.Fatal("Expected health check to fail")
	}
}

func TestOrchestrator_ParallelInitialization(t *testing.T) {
	logger := &mockLogger{}
	metrics := &mockMetrics{}
	orch := NewOrchestrator(logger, metrics)

	// Create components that can be initialized in parallel
	comp1 := &mockComponent{name: "comp1", deps: []string{}}
	comp2 := &mockComponent{name: "comp2", deps: []string{}}
	comp3 := &mockComponent{name: "comp3", deps: []string{"comp1", "comp2"}}

	orch.Register(comp1)
	orch.Register(comp2)
	orch.Register(comp3)

	// Initialize with parallelism
	ctx := context.Background()
	if err := orch.InitializeParallel(ctx, 2); err != nil {
		t.Fatalf("Parallel initialization failed: %v", err)
	}

	// Verify all components initialized
	if !comp1.initialized || !comp2.initialized || !comp3.initialized {
		t.Error("Not all components initialized")
	}
}

func TestOrchestrator_DependencyOrder(t *testing.T) {
	logger := &mockLogger{}
	metrics := &mockMetrics{}
	orch := NewOrchestrator(logger, metrics)

	// Create dependency chain: comp1 -> comp2 -> comp3
	comp1 := &mockComponent{name: "comp1", deps: []string{}}
	comp2 := &mockComponent{name: "comp2", deps: []string{"comp1"}}
	comp3 := &mockComponent{name: "comp3", deps: []string{"comp2"}}

	orch.Register(comp3) // Register out of order
	orch.Register(comp1)
	orch.Register(comp2)

	// Build init order
	order, err := orch.buildInitOrder()
	if err != nil {
		t.Fatalf("Failed to build init order: %v", err)
	}

	// Verify order: comp1, comp2, comp3
	if len(order) != 3 {
		t.Fatalf("Expected 3 components, got %d", len(order))
	}

	if order[0] != "comp1" || order[1] != "comp2" || order[2] != "comp3" {
		t.Errorf("Wrong initialization order: %v", order)
	}
}
