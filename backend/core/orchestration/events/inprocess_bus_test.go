package events

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/sirupsen/logrus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type funcEventHandler struct {
	id string
	fn func(ctx context.Context, event *OrchestrationEvent) error
}

func (h *funcEventHandler) HandleEvent(ctx context.Context, event *OrchestrationEvent) error {
	return h.fn(ctx, event)
}
func (h *funcEventHandler) GetHandlerID() string { return h.id }

func newTestLogger() *logrus.Logger {
	logger := logrus.New()
	logger.SetLevel(logrus.ErrorLevel)
	return logger
}

func TestInProcessEventBus_PublishFansOutToAllTypeSubscribers(t *testing.T) {
	bus := NewInProcessEventBus(newTestLogger())
	ctx := context.Background()

	var mu sync.Mutex
	got := map[string]int{}
	done := make(chan struct{}, 2)

	for _, id := range []string{"a", "b"} {
		id := id
		_, err := bus.Subscribe(ctx, []EventType{EventTypeVMCreated}, &funcEventHandler{
			id: id,
			fn: func(_ context.Context, event *OrchestrationEvent) error {
				mu.Lock()
				got[id]++
				mu.Unlock()
				done <- struct{}{}
				return nil
			},
		})
		require.NoError(t, err)
	}

	require.NoError(t, bus.Publish(ctx, &OrchestrationEvent{Type: EventTypeVMCreated, Target: "vm-1"}))

	for i := 0; i < 2; i++ {
		select {
		case <-done:
		case <-time.After(time.Second):
			t.Fatal("timed out waiting for subscriber delivery")
		}
	}

	mu.Lock()
	defer mu.Unlock()
	assert.Equal(t, 1, got["a"])
	assert.Equal(t, 1, got["b"])
}

func TestInProcessEventBus_SubscribeToAllReceivesEveryType(t *testing.T) {
	bus := NewInProcessEventBus(newTestLogger())
	ctx := context.Background()

	received := make(chan EventType, 2)
	_, err := bus.SubscribeToAll(ctx, &funcEventHandler{
		id: "all",
		fn: func(_ context.Context, event *OrchestrationEvent) error {
			received <- event.Type
			return nil
		},
	})
	require.NoError(t, err)

	require.NoError(t, bus.Publish(ctx, &OrchestrationEvent{Type: EventTypeVMCreated}))
	require.NoError(t, bus.Publish(ctx, &OrchestrationEvent{Type: EventTypeNodeFailure}))

	seen := map[EventType]bool{}
	for i := 0; i < 2; i++ {
		select {
		case et := <-received:
			seen[et] = true
		case <-time.After(time.Second):
			t.Fatal("timed out waiting for SubscribeToAll delivery")
		}
	}
	assert.True(t, seen[EventTypeVMCreated])
	assert.True(t, seen[EventTypeNodeFailure])
}

func TestInProcessEventBus_Unsubscribe(t *testing.T) {
	bus := NewInProcessEventBus(newTestLogger())
	ctx := context.Background()

	var count int32
	var mu sync.Mutex
	ack := make(chan struct{}, 4)
	sub, err := bus.Subscribe(ctx, []EventType{EventTypeVMCreated}, &funcEventHandler{
		id: "x",
		fn: func(_ context.Context, event *OrchestrationEvent) error {
			mu.Lock()
			count++
			mu.Unlock()
			ack <- struct{}{}
			return nil
		},
	})
	require.NoError(t, err)

	require.NoError(t, bus.Publish(ctx, &OrchestrationEvent{Type: EventTypeVMCreated}))
	select {
	case <-ack:
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for first delivery")
	}

	require.NoError(t, bus.Unsubscribe(sub.ID))
	assert.Equal(t, 0, bus.GetMetrics().SubscriptionCount)

	// Program-order + mutex guarantee: by the time Unsubscribe returned above,
	// the registry no longer contains this subscription, so this Publish
	// cannot reach it — no timing dependency needed.
	require.NoError(t, bus.Publish(ctx, &OrchestrationEvent{Type: EventTypeVMCreated}))
	select {
	case <-ack:
		t.Fatal("unsubscribed handler should not have received a second event")
	case <-time.After(100 * time.Millisecond):
	}

	mu.Lock()
	defer mu.Unlock()
	assert.Equal(t, int32(1), count)
}

func TestInProcessEventBus_SlowSubscriberNeverBlocksPublisherOrOthers(t *testing.T) {
	bus := NewInProcessEventBus(newTestLogger())
	ctx := context.Background()

	block := make(chan struct{})
	slowStarted := make(chan struct{}, defaultSubscriberQueueSize+10)
	_, err := bus.Subscribe(ctx, []EventType{EventTypeVMCreated}, &funcEventHandler{
		id: "slow",
		fn: func(_ context.Context, event *OrchestrationEvent) error {
			slowStarted <- struct{}{}
			<-block
			return nil
		},
	})
	require.NoError(t, err)

	fastDone := make(chan struct{}, 1)
	_, err = bus.Subscribe(ctx, []EventType{EventTypeVMCreated}, &funcEventHandler{
		id: "fast",
		fn: func(_ context.Context, event *OrchestrationEvent) error {
			select {
			case fastDone <- struct{}{}:
			default:
			}
			return nil
		},
	})
	require.NoError(t, err)

	require.NoError(t, bus.Publish(ctx, &OrchestrationEvent{Type: EventTypeVMCreated}))
	select {
	case <-fastDone:
	case <-time.After(200 * time.Millisecond):
		t.Fatal("fast subscriber should receive its event promptly even though the slow one never drains")
	}
	// Let the slow handler actually start consuming its one in-flight event.
	select {
	case <-slowStarted:
	case <-time.After(time.Second):
		t.Fatal("slow subscriber never started")
	}

	start := time.Now()
	for i := 0; i < defaultSubscriberQueueSize+10; i++ {
		require.NoError(t, bus.Publish(ctx, &OrchestrationEvent{Type: EventTypeVMCreated}))
	}
	elapsed := time.Since(start)
	assert.Less(t, elapsed, 2*time.Second, "Publish calls must never block on a stalled subscriber")

	close(block)
	assert.Greater(t, bus.GetMetrics().EventsDropped, uint64(0))
}

func TestInProcessEventBus_PanicInHandlerIsRecoveredAndIsolated(t *testing.T) {
	bus := NewInProcessEventBus(newTestLogger())
	ctx := context.Background()

	okDone := make(chan struct{}, 1)
	_, err := bus.Subscribe(ctx, []EventType{EventTypeVMCreated}, &funcEventHandler{
		id: "panics",
		fn: func(_ context.Context, event *OrchestrationEvent) error {
			panic("boom")
		},
	})
	require.NoError(t, err)
	_, err = bus.Subscribe(ctx, []EventType{EventTypeVMCreated}, &funcEventHandler{
		id: "ok",
		fn: func(_ context.Context, event *OrchestrationEvent) error {
			select {
			case okDone <- struct{}{}:
			default:
			}
			return nil
		},
	})
	require.NoError(t, err)

	require.NoError(t, bus.Publish(ctx, &OrchestrationEvent{Type: EventTypeVMCreated}))

	select {
	case <-okDone:
	case <-time.After(time.Second):
		t.Fatal("non-panicking subscriber should still receive the event")
	}

	require.Eventually(t, func() bool {
		return bus.GetMetrics().EventsFailed >= 1
	}, time.Second, 5*time.Millisecond)
}

func TestInProcessEventBus_ConnectDisconnect(t *testing.T) {
	bus := NewInProcessEventBus(newTestLogger())
	ctx := context.Background()

	require.NoError(t, bus.Connect(ctx, EventBusConfig{}))
	assert.True(t, bus.GetHealth().Connected)

	received := make(chan struct{}, 1)
	_, err := bus.SubscribeToAll(ctx, &funcEventHandler{
		id: "x",
		fn: func(_ context.Context, event *OrchestrationEvent) error {
			select {
			case received <- struct{}{}:
			default:
			}
			return nil
		},
	})
	require.NoError(t, err)

	require.NoError(t, bus.Disconnect())
	assert.False(t, bus.GetHealth().Connected)

	require.NoError(t, bus.Publish(ctx, &OrchestrationEvent{Type: EventTypeVMCreated}))
	select {
	case <-received:
		t.Fatal("a previously-active subscriber must not receive events after Disconnect")
	case <-time.After(100 * time.Millisecond):
	}
}
