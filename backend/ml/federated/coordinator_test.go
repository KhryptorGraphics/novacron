package federated

import (
	"testing"
	"time"
)

// mockTopologyOptimizer is a test double satisfying the TopologyOptimizer
// interface. It records whether its methods were invoked so the test can
// prove the coordinator actually dispatches through the interface value
// stored in FederatedCoordinator.topology.
//
// This is a regression test for a pointer-vs-interface bug: the
// `topology` field was declared as `*TopologyOptimizer` (a pointer to an
// interface) instead of `TopologyOptimizer` (the interface itself). That
// mistake does not compile once anything tries to assign an interface
// value to the field or call a method through it, so this test would fail
// to build (not just fail at runtime) if the bug were reintroduced.
type mockTopologyOptimizer struct {
	optimizeCalled  bool
	optimizeClients []*Client
	updateCalled    chan int // receives clientID when UpdateClientPerformance runs
}

func (m *mockTopologyOptimizer) OptimizeTopology(roundNumber int, budgetConstraint float64) ([]*Client, error) {
	m.optimizeCalled = true
	return m.optimizeClients, nil
}

func (m *mockTopologyOptimizer) UpdateClientPerformance(clientID int, quality, reliability float64) error {
	m.updateCalled <- clientID
	return nil
}

func (m *mockTopologyOptimizer) GetTopologyStats() (map[string]interface{}, error) {
	return map[string]interface{}{"mock": true}, nil
}

func TestSetTopologyOptimizer_DispatchesThroughInterface(t *testing.T) {
	coord := NewFederatedCoordinator(0.9, 10, "fedavg")

	client := &Client{ID: 1, DataSize: 100, ComputeCapacity: 1.0, Reliability: 0.5}
	if err := coord.RegisterClient(client); err != nil {
		t.Fatalf("RegisterClient failed: %v", err)
	}

	mock := &mockTopologyOptimizer{optimizeClients: []*Client{client}, updateCalled: make(chan int, 1)}

	// SetTopologyOptimizer accepts a TopologyOptimizer interface value.
	// If the coordinator's `topology` field were `*TopologyOptimizer`
	// (pointer-to-interface), this line would fail to compile.
	coord.SetTopologyOptimizer(mock)

	selected, err := coord.selectClients(1)
	if err != nil {
		t.Fatalf("selectClients returned error: %v", err)
	}
	if !mock.optimizeCalled {
		t.Fatal("expected OptimizeTopology to be invoked through the interface, but it was not called")
	}
	if len(selected) != 1 || selected[0].ID != 1 {
		t.Fatalf("expected the mock's selected clients to be returned, got %+v", selected)
	}

	// updateClientMetrics also dispatches through c.topology and must
	// compile/run the same way. UpdateClientPerformance is launched via
	// `go`, so synchronize on the channel it sends to rather than racing
	// on a shared bool.
	update := &ModelUpdate{ClientID: 1, Accuracy: 0.8, DataSize: 100}
	coord.updateClientMetrics([]*ModelUpdate{update})

	select {
	case gotClientID := <-mock.updateCalled:
		if gotClientID != 1 {
			t.Fatalf("expected UpdateClientPerformance called with clientID 1, got %d", gotClientID)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("expected UpdateClientPerformance to be invoked through the interface, but it was not called")
	}
}

func TestSelectClients_FallsBackWithoutOptimizer(t *testing.T) {
	coord := NewFederatedCoordinator(0.9, 10, "fedavg")
	client := &Client{ID: 2, DataSize: 50, ComputeCapacity: 1.0}
	if err := coord.RegisterClient(client); err != nil {
		t.Fatalf("RegisterClient failed: %v", err)
	}

	selected, err := coord.selectClients(1)
	if err != nil {
		t.Fatalf("selectClients returned error: %v", err)
	}
	if len(selected) != 1 {
		t.Fatalf("expected random fallback to select the single registered client, got %d", len(selected))
	}
}
