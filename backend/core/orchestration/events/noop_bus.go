//go:build !experimental

package events

import (
	"context"
	"time"
)

// NoopEventBus is a core-mode event bus that satisfies EventBus without external deps
// It does not deliver messages; Publish/Subscribe succeed for local development
// Metrics/health are kept minimal.
type NoopEventBus struct{
	metrics EventBusMetrics
	health  HealthStatus
}

func NewNoopEventBus() *NoopEventBus {
	return &NoopEventBus{
		metrics: EventBusMetrics{},
		health: HealthStatus{Connected: true, Status: "noop"},
	}
}

func (n *NoopEventBus) Connect(ctx context.Context, config EventBusConfig) error { return nil }
func (n *NoopEventBus) Disconnect() error { return nil }
func (n *NoopEventBus) Publish(ctx context.Context, event *OrchestrationEvent) error {
	n.metrics.EventsPublished++
	n.metrics.LastEventTime = time.Now()
	return nil
}
func (n *NoopEventBus) Subscribe(ctx context.Context, eventTypes []EventType, handler EventHandler) (*Subscription, error) {
	// No delivery; return a dummy subscription
	return &Subscription{ID: "noop", Topic: "all", Handler: handler, Active: true, CreatedAt: time.Now()}, nil
}
func (n *NoopEventBus) SubscribeToAll(ctx context.Context, handler EventHandler) (*Subscription, error) {
	return &Subscription{ID: "noop-all", Topic: "all", Handler: handler, Active: true, CreatedAt: time.Now()}, nil
}
func (n *NoopEventBus) GetHealth() HealthStatus { return n.health }
func (n *NoopEventBus) GetMetrics() EventBusMetrics { return n.metrics }

