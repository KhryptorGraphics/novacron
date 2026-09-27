package events

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
	"github.com/sirupsen/logrus"
)

// defaultSubscriberQueueSize bounds the per-subscriber delivery channel. A
// subscriber that cannot keep up drops events instead of blocking Publish or
// any other subscriber (see trySend).
const defaultSubscriberQueueSize = 256

// InProcessEventBus is an in-memory EventBus: no external broker, real
// asynchronous delivery. Publish fans out to every SubscribeToAll handler and
// every Subscribe handler registered for the event's type. Delivery to each
// subscriber goes through its own bounded queue drained by a dedicated
// goroutine, so a slow or panicking handler can never block Publish or any
// other subscriber.
type InProcessEventBus struct {
	mu            sync.RWMutex // guards registry + is-safe-to-close-channel ordering (see below)
	logger        *logrus.Logger
	subscriptions map[string]*inProcessSubscription
	allSubs       map[string]*inProcessSubscription
	typeSubs      map[EventType]map[string]*inProcessSubscription
	connectedAt   time.Time
	connected     bool

	published     atomic.Uint64
	processed     atomic.Uint64
	failed        atomic.Uint64
	dropped       atomic.Uint64
	lastEventTime atomic.Pointer[time.Time]
}

type inProcessSubscription struct {
	sub     *Subscription
	handler EventHandler
	queue   chan *OrchestrationEvent
	cancel  context.CancelFunc
}

// NewInProcessEventBus builds a ready-to-use in-process bus. Connect is not
// required before Publish/Subscribe (unlike NATSEventBus) but is supported so
// callers written against the EventBus interface behave the same way.
func NewInProcessEventBus(logger *logrus.Logger) *InProcessEventBus {
	if logger == nil {
		logger = logrus.New()
	}
	return &InProcessEventBus{
		logger:        logger,
		subscriptions: make(map[string]*inProcessSubscription),
		allSubs:       make(map[string]*inProcessSubscription),
		typeSubs:      make(map[EventType]map[string]*inProcessSubscription),
	}
}

// Connect marks the bus connected. There is no external transport to dial.
func (b *InProcessEventBus) Connect(ctx context.Context, config EventBusConfig) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.connectedAt = time.Now()
	b.connected = true
	return nil
}

// Disconnect cancels every subscription's drain goroutine and clears the
// registry. Publish calls after Disconnect are accepted (matching Noop/NATS
// "publish never fails locally") but reach no subscriber.
func (b *InProcessEventBus) Disconnect() error {
	b.mu.Lock()
	defer b.mu.Unlock()
	for id, s := range b.subscriptions {
		s.cancel()
		delete(b.subscriptions, id)
	}
	b.allSubs = make(map[string]*inProcessSubscription)
	b.typeSubs = make(map[EventType]map[string]*inProcessSubscription)
	b.connected = false
	return nil
}

// Publish fans the event out to every matching subscriber's queue.
func (b *InProcessEventBus) Publish(ctx context.Context, event *OrchestrationEvent) error {
	if event == nil {
		return fmt.Errorf("orchestration event cannot be nil")
	}
	b.published.Add(1)
	now := time.Now()
	b.lastEventTime.Store(&now)

	b.mu.RLock()
	defer b.mu.RUnlock() // held for the whole send loop: Unsubscribe/Disconnect (Lock) cannot
	// remove+let-go-of a subscription concurrently with a send to it — this is
	// what makes it safe to never close() a subscriber queue: by the time a
	// subscription is removed from the maps, no in-flight Publish holding the
	// RLock can still be iterating it, and no future Publish will find it.
	for _, s := range b.allSubs {
		b.trySend(s, event)
	}
	for _, s := range b.typeSubs[event.Type] {
		b.trySend(s, event)
	}
	return nil
}

func (b *InProcessEventBus) trySend(s *inProcessSubscription, event *OrchestrationEvent) {
	select {
	case s.queue <- event:
	default:
		b.dropped.Add(1)
		b.logger.WithFields(logrus.Fields{
			"subscription_id": s.sub.ID,
			"event_type":      event.Type,
		}).Warn("event bus: subscriber queue full; dropping event")
	}
}

// Subscribe registers handler for the given event types.
func (b *InProcessEventBus) Subscribe(ctx context.Context, eventTypes []EventType, handler EventHandler) (*Subscription, error) {
	if handler == nil {
		return nil, fmt.Errorf("event handler cannot be nil")
	}
	topic := "custom"
	if len(eventTypes) > 0 {
		topic = string(eventTypes[0])
		for _, t := range eventTypes[1:] {
			topic += "," + string(t)
		}
	}
	sub := &Subscription{
		ID:        uuid.New().String(),
		Topic:     topic,
		Handler:   handler,
		Active:    true,
		CreatedAt: time.Now(),
	}
	subCtx, cancel := context.WithCancel(context.Background())
	is := &inProcessSubscription{
		sub:     sub,
		handler: handler,
		queue:   make(chan *OrchestrationEvent, defaultSubscriberQueueSize),
		cancel:  cancel,
	}

	b.mu.Lock()
	b.subscriptions[sub.ID] = is
	for _, t := range eventTypes {
		if b.typeSubs[t] == nil {
			b.typeSubs[t] = make(map[string]*inProcessSubscription)
		}
		b.typeSubs[t][sub.ID] = is
	}
	b.mu.Unlock()

	go b.runSubscription(subCtx, is)
	return sub, nil
}

// SubscribeToAll registers handler for every event type.
func (b *InProcessEventBus) SubscribeToAll(ctx context.Context, handler EventHandler) (*Subscription, error) {
	if handler == nil {
		return nil, fmt.Errorf("event handler cannot be nil")
	}
	sub := &Subscription{
		ID:        uuid.New().String(),
		Topic:     "all",
		Handler:   handler,
		Active:    true,
		CreatedAt: time.Now(),
	}
	subCtx, cancel := context.WithCancel(context.Background())
	is := &inProcessSubscription{
		sub:     sub,
		handler: handler,
		queue:   make(chan *OrchestrationEvent, defaultSubscriberQueueSize),
		cancel:  cancel,
	}

	b.mu.Lock()
	b.subscriptions[sub.ID] = is
	b.allSubs[sub.ID] = is
	b.mu.Unlock()

	go b.runSubscription(subCtx, is)
	return sub, nil
}

func (b *InProcessEventBus) runSubscription(ctx context.Context, is *inProcessSubscription) {
	for {
		select {
		case <-ctx.Done():
			return
		case event := <-is.queue:
			b.dispatch(ctx, is, event)
		}
	}
}

func (b *InProcessEventBus) dispatch(ctx context.Context, is *inProcessSubscription, event *OrchestrationEvent) {
	defer func() {
		if r := recover(); r != nil {
			b.failed.Add(1)
			b.logger.WithFields(logrus.Fields{
				"subscription_id": is.sub.ID,
				"panic":           r,
			}).Error("event bus: handler panicked")
		}
	}()
	if err := is.handler.HandleEvent(ctx, event); err != nil {
		b.failed.Add(1)
		b.logger.WithError(err).WithField("subscription_id", is.sub.ID).Warn("event bus: handler error")
		return
	}
	b.processed.Add(1)
}

// Unsubscribe mirrors NATSEventBus.Unsubscribe (not part of the EventBus
// interface). Lock() here happens-after any in-flight Publish's RLock, so by
// the time we delete from the maps and return, no Publish can ever enqueue to
// this subscription's channel again — cancel() alone is enough to stop the
// drain goroutine; the channel is never closed (avoids a send-on-closed race
// entirely).
func (b *InProcessEventBus) Unsubscribe(subscriptionID string) error {
	b.mu.Lock()
	is, ok := b.subscriptions[subscriptionID]
	if !ok {
		b.mu.Unlock()
		return fmt.Errorf("subscription %s not found", subscriptionID)
	}
	delete(b.subscriptions, subscriptionID)
	delete(b.allSubs, subscriptionID)
	for et, m := range b.typeSubs {
		delete(m, subscriptionID)
		if len(m) == 0 {
			delete(b.typeSubs, et)
		}
	}
	b.mu.Unlock()

	is.sub.Active = false
	is.cancel()
	return nil
}

// GetHealth reports the bus's connected state.
func (b *InProcessEventBus) GetHealth() HealthStatus {
	b.mu.RLock()
	defer b.mu.RUnlock()
	status := "disconnected"
	if b.connected {
		status = "connected"
	}
	return HealthStatus{Connected: b.connected, Status: status, LastPing: time.Now()}
}

// GetMetrics reports live counters.
func (b *InProcessEventBus) GetMetrics() EventBusMetrics {
	b.mu.RLock()
	n := len(b.subscriptions)
	connectedAt := b.connectedAt
	b.mu.RUnlock()

	m := EventBusMetrics{
		EventsPublished:   b.published.Load(),
		EventsProcessed:   b.processed.Load(),
		EventsFailed:      b.failed.Load(),
		EventsDropped:     b.dropped.Load(),
		SubscriptionCount: n,
	}
	if !connectedAt.IsZero() {
		m.ConnectionUptime = time.Since(connectedAt)
	}
	if p := b.lastEventTime.Load(); p != nil {
		m.LastEventTime = *p
	}
	return m
}
