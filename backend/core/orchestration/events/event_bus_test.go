package events

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/sirupsen/logrus"
	
)

func TestNATSEventBus_PublishSubscribe(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.DebugLevel)
	
	bus := NewNATSEventBus(logger)
	
	// Test without connection should fail
	ctx := context.Background()
	event := &OrchestrationEvent{
		Type:      EventTypeVMCreated,
		Source:    "test",
		Timestamp: time.Now(),
		Data:      map[string]interface{}{"test": "data"},
		Priority:  PriorityNormal,
	}
	
	err := bus.Publish(ctx, event)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "not connected")
}

func TestNATSEventBus_HealthCheck(t *testing.T) {
	logger := logrus.New()
	bus := NewNATSEventBus(logger)
	
	health := bus.GetHealth()
	assert.False(t, health.Connected)
	assert.Equal(t, "disconnected", health.Status)
}

func TestNATSEventBus_Metrics(t *testing.T) {
	logger := logrus.New()
	bus := NewNATSEventBus(logger)
	
	metrics := bus.GetMetrics()
	assert.Equal(t, uint64(0), metrics.EventsPublished)
	assert.Equal(t, uint64(0), metrics.EventsReceived)
	assert.Equal(t, 0, metrics.SubscriptionCount)
}

func TestEventHandlerFunc(t *testing.T) {
	called := false
	handler := NewEventHandlerFunc("test-handler", "Test Handler", func(ctx context.Context, event *OrchestrationEvent) error {
		called = true
		return nil
	})
	
	assert.Equal(t, "test-handler", handler.GetHandlerID())
	
	event := &OrchestrationEvent{
		Type:   EventTypeVMCreated,
		Source: "test",
	}
	
	err := handler.HandleEvent(context.Background(), event)
	assert.NoError(t, err)
	assert.True(t, called)
}

func TestCompositeEventHandler(t *testing.T) {
	logger := logrus.New()
	composite := NewCompositeEventHandler("composite", logger)
	
	// Add handlers for different event types
	vmHandler := NewEventHandlerFunc("vm-handler", "VM Handler", func(ctx context.Context, event *OrchestrationEvent) error {
		return nil
	})
	
	nodeHandler := NewEventHandlerFunc("node-handler", "Node Handler", func(ctx context.Context, event *OrchestrationEvent) error {
		return nil
	})
	
	composite.AddHandler([]EventType{EventTypeVMCreated}, vmHandler)
	composite.AddHandler([]EventType{EventTypeNodeFailure}, nodeHandler)
	
	// Test VM event handling
	vmEvent := &OrchestrationEvent{
		Type:   EventTypeVMCreated,
		Source: "test",
	}
	
	err := composite.HandleEvent(context.Background(), vmEvent)
	assert.NoError(t, err)
	
	// Test node event handling
	nodeEvent := &OrchestrationEvent{
		Type:   EventTypeNodeFailure,
		Source: "test",
	}
	
	err = composite.HandleEvent(context.Background(), nodeEvent)
	assert.NoError(t, err)
	
	// Test unknown event type (should not error but do nothing)
	unknownEvent := &OrchestrationEvent{
		Type:   EventTypeScalingTriggered,
		Source: "test",
	}
	
	err = composite.HandleEvent(context.Background(), unknownEvent)
	assert.NoError(t, err)
}

func TestFilteredEventHandler(t *testing.T) {
	logger := logrus.New()
	
	handled := false
	baseHandler := NewEventHandlerFunc("base", "Base", func(ctx context.Context, event *OrchestrationEvent) error {
		handled = true
		return nil
	})
	
	// Filter by source
	filter := EventFilter{
		Sources: []string{"allowed-source"},
	}
	
	filteredHandler := NewFilteredEventHandler("filtered", filter, baseHandler, logger)
	
	// Event from allowed source should be handled
	allowedEvent := &OrchestrationEvent{
		Type:   EventTypeVMCreated,
		Source: "allowed-source",
	}
	
	err := filteredHandler.HandleEvent(context.Background(), allowedEvent)
	assert.NoError(t, err)
	assert.True(t, handled)
	
	// Reset flag
	handled = false
	
	// Event from disallowed source should be filtered out
	disallowedEvent := &OrchestrationEvent{
		Type:   EventTypeVMCreated,
		Source: "disallowed-source",
	}
	
	err = filteredHandler.HandleEvent(context.Background(), disallowedEvent)
	assert.NoError(t, err)
	assert.False(t, handled)
}

func TestFilteredEventHandler_Priority(t *testing.T) {
	logger := logrus.New()
	
	handled := false
	baseHandler := NewEventHandlerFunc("base", "Base", func(ctx context.Context, event *OrchestrationEvent) error {
		handled = true
		return nil
	})
	
	// Filter by priority
	filter := EventFilter{
		Priorities: []EventPriority{PriorityCritical},
	}
	
	filteredHandler := NewFilteredEventHandler("filtered", filter, baseHandler, logger)
	
	// Critical event should be handled
	criticalEvent := &OrchestrationEvent{
		Type:     EventTypeNodeFailure,
		Source:   "test",
		Priority: PriorityCritical,
	}
	
	err := filteredHandler.HandleEvent(context.Background(), criticalEvent)
	assert.NoError(t, err)
	assert.True(t, handled)
	
	// Reset flag
	handled = false
	
	// Normal priority event should be filtered out
	normalEvent := &OrchestrationEvent{
		Type:     EventTypeVMCreated,
		Source:   "test",
		Priority: PriorityNormal,
	}
	
	err = filteredHandler.HandleEvent(context.Background(), normalEvent)
	assert.NoError(t, err)
	assert.False(t, handled)
}

func TestRetryEventHandler(t *testing.T) {
	logger := logrus.New()
	
	attempts := 0
	failingHandler := NewEventHandlerFunc("failing", "Failing", func(ctx context.Context, event *OrchestrationEvent) error {
		attempts++
		if attempts < 3 {
			return assert.AnError
		}
		return nil
	})
	
	retryHandler := NewRetryEventHandler("retry", failingHandler, 3, logger)
	
	event := &OrchestrationEvent{
		Type:   EventTypeVMCreated,
		Source: "test",
	}
	
	err := retryHandler.HandleEvent(context.Background(), event)
	assert.NoError(t, err)
	assert.Equal(t, 3, attempts)
}

func TestRetryEventHandler_MaxRetriesExceeded(t *testing.T) {
	logger := logrus.New()
	
	alwaysFailingHandler := NewEventHandlerFunc("failing", "Failing", func(ctx context.Context, event *OrchestrationEvent) error {
		return assert.AnError
	})
	
	retryHandler := NewRetryEventHandler("retry", alwaysFailingHandler, 2, logger)
	
	event := &OrchestrationEvent{
		Type:   EventTypeVMCreated,
		Source: "test",
	}
	
	err := retryHandler.HandleEvent(context.Background(), event)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed after 3 attempts")
}

func TestAsyncEventHandler(t *testing.T) {
	logger := logrus.New()
	
	processedEvents := make(chan string, 10)
	baseHandler := NewEventHandlerFunc("base", "Base", func(ctx context.Context, event *OrchestrationEvent) error {
		processedEvents <- event.ID
		return nil
	})
	
	asyncHandler := NewAsyncEventHandler("async", baseHandler, 2, 5, logger)
	defer asyncHandler.Shutdown()
	
	// Send multiple events
	events := []*OrchestrationEvent{
		{ID: "1", Type: EventTypeVMCreated, Source: "test"},
		{ID: "2", Type: EventTypeVMStarted, Source: "test"},
		{ID: "3", Type: EventTypeVMStopped, Source: "test"},
	}
	
	for _, event := range events {
		err := asyncHandler.HandleEvent(context.Background(), event)
		assert.NoError(t, err)
	}
	
	// Collect processed event IDs
	var processedIDs []string
	timeout := time.NewTimer(5 * time.Second)
	defer timeout.Stop()
	
	for i := 0; i < len(events); i++ {
		select {
		case id := <-processedEvents:
			processedIDs = append(processedIDs, id)
		case <-timeout.C:
			t.Fatal("Timeout waiting for events to be processed")
		}
	}
	
	assert.Len(t, processedIDs, 3)
	assert.Contains(t, processedIDs, "1")
	assert.Contains(t, processedIDs, "2")
	assert.Contains(t, processedIDs, "3")
}