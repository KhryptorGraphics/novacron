package orchestration

import (
	"context"
	"testing"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/orchestration/autoscaling"
	"github.com/khryptorgraphics/novacron/backend/core/orchestration/events"
	"github.com/sirupsen/logrus"
	"github.com/stretchr/testify/require"
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

// TestSetEventBusMakesNodeFailureSubscriptionLive proves the gap SetEventBus
// closes: before it existed, Start subscribed handleNodeFailure/etc to the
// engine's own internal NoopEventBus, so a NodeFailure event published on the
// bus shared with autoscaler/healing/policy never reached this engine at all.
// With SetEventBus, Start's subscription is against the SAME bus, so a
// published event is actually delivered.
func TestSetEventBusMakesNodeFailureSubscriptionLive(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.ErrorLevel)
	e := NewDefaultOrchestrationEngine(logger)

	sharedBus := events.NewInProcessEventBus(logger)
	e.SetEventBus(sharedBus)

	require.NoError(t, e.Start(context.Background()))
	defer e.Stop(context.Background())

	require.NoError(t, sharedBus.Publish(context.Background(), &events.OrchestrationEvent{
		Type:      events.EventTypeNodeFailure,
		Timestamp: time.Now(),
		Data:      map[string]interface{}{"node_id": "node-shared"},
	}))

	require.Eventually(t, func() bool {
		statuses := e.GetNodeStatuses()
		st, ok := statuses["node-shared"]
		return ok && !st.Healthy
	}, time.Second, 5*time.Millisecond, "handleNodeFailure should have marked node-shared unhealthy via the shared bus subscription")
}

// TestAutoscalerScalingTriggeredUpdatesEngineStatus runs the path api-server
// wires: an actionable autoscaler decision publishes scaling.triggered on the
// shared bus and the engine's scaling subscription records it in
// GetStatus().Metrics; the follow-up no_action decision (target in cooldown)
// triggers nothing.
func TestAutoscalerScalingTriggeredUpdatesEngineStatus(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.ErrorLevel)
	ctx := context.Background()
	bus := events.NewInProcessEventBus(logger)
	e := NewDefaultOrchestrationEngine(logger)
	e.SetEventBus(bus)
	require.NoError(t, e.Start(ctx))
	defer e.Stop(ctx)

	scaler := autoscaling.NewDefaultAutoScaler(logger, bus)
	require.NoError(t, scaler.SetMetricsSource(func() (*autoscaling.MetricsData, error) {
		return &autoscaling.MetricsData{
			Timestamp: time.Now(), TargetID: "web", TargetType: "vm",
			CPUUsage: 0.95, MemoryUsage: 0.5, ActiveVMs: 2,
		}, nil
	}))
	require.NoError(t, scaler.AddTarget(&autoscaling.AutoScalerTarget{ID: "web", Type: "vm", Enabled: true}))

	decision, err := scaler.GetScalingDecision("web")
	require.NoError(t, err)
	require.Equal(t, autoscaling.ScalingActionScaleUp, decision.Action)

	require.Eventually(t, func() bool {
		m := e.GetStatus().Metrics
		return m["scaling.web.action"] == "scale_up" &&
			m["scaling.web.current_scale"] == 2 &&
			m["scaling.web.target_scale"] == decision.TargetScale
	}, time.Second, 5*time.Millisecond, "engine never recorded the triggered scale-up")

	cooled, err := scaler.GetScalingDecision("web")
	require.NoError(t, err)
	require.Equal(t, autoscaling.ScalingActionNoAction, cooled.Action)

	// Barrier: the engine's subscription drains in publish order, so once this
	// later event is handled, any trigger from the no_action decision has been.
	require.NoError(t, bus.Publish(ctx, &events.OrchestrationEvent{
		Type: events.EventTypeNodeRecovered, Timestamp: time.Now(),
		Data: map[string]interface{}{"node_id": "barrier"},
	}))
	require.Eventually(t, func() bool {
		_, ok := e.GetNodeStatuses()["barrier"]
		return ok
	}, time.Second, 5*time.Millisecond)
	require.Equal(t, uint64(1), e.GetStatus().Metrics["scaling_triggered_total"])
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

