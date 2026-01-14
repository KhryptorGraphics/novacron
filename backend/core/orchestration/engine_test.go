package orchestration

import (
	"context"
	"testing"
	"time"

	"github.com/sirupsen/logrus"
	"github.com/khryptorgraphics/novacron/backend/core/orchestration/events"
)

func TestHandleNodeFailurePublishesHealingAndInvokesEvacuation(t *testing.T) {
	logger := logrus.New()
	e := NewDefaultOrchestrationEngine(logger)

	// Track publish calls
	published := make(chan *events.OrchestrationEvent, 1)
	// Replace eventBus with a stub
	e.eventBus = &stubEventBus{publishFn: func(ctx context.Context, ev *events.OrchestrationEvent) error {
		published <- ev
		return nil
	}}

	// Track evacuation calls
	called := make(chan string, 1)
	stubEvac := &stubEvacuationHandler{fn: func(ctx context.Context, nodeID string) error {
		called <- nodeID
		return nil
	}}
	e.SetEvacuationHandler(stubEvac)

	ev := &events.OrchestrationEvent{
		Type:      events.EventTypeNodeFailure,
		Timestamp: time.Now(),
		Data:      map[string]interface{}{"node_id": "node-x"},
	}
	if err := e.handleNodeFailure(context.Background(), ev); err != nil {
		t.Fatalf("handleNodeFailure returned error: %v", err)
	}

	// Verify healing event was published
	select {
	case out := <-published:
		if out.Type != events.EventTypeHealingTriggered {
			t.Fatalf("expected healing.triggered, got %v", out.Type)
		}
		if out.Data["node_id"].(string) != "node-x" {
			t.Fatalf("expected node_id=node-x, got %v", out.Data["node_id"])}
	case <-time.After(1 * time.Second):
		t.Fatal("no event published")
	}

	// Verify evacuation invoked
	select {
	case nid := <-called:
		if nid != "node-x" { t.Fatalf("expected node-x, got %s", nid) }
	case <-time.After(1 * time.Second):
		t.Fatal("evacuation was not invoked")
	}
}

func TestHandleNodeMetricsUpdatesState(t *testing.T) {
	logger := logrus.New()
	e := NewDefaultOrchestrationEngine(logger)
	metricsEv := &events.OrchestrationEvent{
		Type:      events.EventTypeNodeMetrics,
		Timestamp: time.Now(),
		Data: map[string]interface{}{
			"node_id": "node-m",
			"cpu_utilization": 77.5,
			"memory_utilization": 61.0,
			"disk_utilization": 40.0,
			"network_utilization": 22.0,
			"active_vms": 3,
			"healthy": true,
		},
	}
	if err := e.handleNodeMetrics(context.Background(), metricsEv); err != nil {
		t.Fatalf("handleNodeMetrics returned error: %v", err)
	}
	if _, ok := e.metrics["nodes.node-m.cpu_utilization"]; !ok {
		t.Fatal("cpu metric not recorded")
	}
	e.mu.RLock()
	st := e.nodeStatuses["node-m"]
	e.mu.RUnlock()
	if !st.Healthy {
		t.Fatal("expected node to be marked healthy")
	}
}

type stubEventBus struct{ publishFn func(ctx context.Context, ev *events.OrchestrationEvent) error }

func (s *stubEventBus) Connect(ctx context.Context, config events.EventBusConfig) error { return nil }
func (s *stubEventBus) Disconnect() error { return nil }
func (s *stubEventBus) Publish(ctx context.Context, event *events.OrchestrationEvent) error { return s.publishFn(ctx, event) }
func (s *stubEventBus) Subscribe(ctx context.Context, eventTypes []events.EventType, handler events.EventHandler) (*events.Subscription, error) {
	return nil, nil
}
func (s *stubEventBus) SubscribeToAll(ctx context.Context, handler events.EventHandler) (*events.Subscription, error) { return nil, nil }
func (s *stubEventBus) GetHealth() events.HealthStatus { return events.HealthStatus{} }
func (s *stubEventBus) GetMetrics() events.EventBusMetrics { return events.EventBusMetrics{} }

type stubEvacuationHandler struct{ fn func(ctx context.Context, nodeID string) error }

func (s *stubEvacuationHandler) EvacuateNode(ctx context.Context, nodeID string) error { return s.fn(ctx, nodeID) }

