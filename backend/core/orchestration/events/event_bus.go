package events

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/nats-io/nats.go"
	"github.com/google/uuid"
	"github.com/sirupsen/logrus"
	
)

// EventBus defines the interface for event bus operations
type EventBus interface {
	// Connect establishes connection to the event bus
	Connect(ctx context.Context, config EventBusConfig) error
	
	// Disconnect closes the connection to the event bus
	Disconnect() error
	
	// Publish publishes an event to the bus
	Publish(ctx context.Context, event *OrchestrationEvent) error
	
	// Subscribe subscribes to events of specific types
	Subscribe(ctx context.Context, eventTypes []EventType, handler EventHandler) (*Subscription, error)
	
	// SubscribeToAll subscribes to all events
	SubscribeToAll(ctx context.Context, handler EventHandler) (*Subscription, error)
	
	// GetHealth returns the health status of the event bus
	GetHealth() HealthStatus
	
	// GetMetrics returns metrics about the event bus
	GetMetrics() EventBusMetrics
}

// EventHandler defines the interface for event handlers
type EventHandler interface {
	HandleEvent(ctx context.Context, event *OrchestrationEvent) error
	GetHandlerID() string
}

// EventBusConfig contains configuration for the event bus
type EventBusConfig struct {
	URL             string        `json:"url"`
	ClusterID       string        `json:"cluster_id"`
	ClientID        string        `json:"client_id"`
	MaxReconnects   int           `json:"max_reconnects"`
	ReconnectWait   time.Duration `json:"reconnect_wait"`
	PingInterval    time.Duration `json:"ping_interval"`
	MaxPingWait     time.Duration `json:"max_ping_wait"`
	BufferSize      int           `json:"buffer_size"`
	Durability      bool          `json:"durability"`
	QueueGroup      string        `json:"queue_group"`
}

// Subscription represents an active subscription
type Subscription struct {
	ID        string
	Topic     string
	Handler   EventHandler
	Active    bool
	CreatedAt time.Time
	natsSubscription *nats.Subscription
}

// HealthStatus represents the health of the event bus
type HealthStatus struct {
	Connected    bool      `json:"connected"`
	LastError    string    `json:"last_error,omitempty"`
	LastPing     time.Time `json:"last_ping"`
	Reconnects   int       `json:"reconnects"`
	Status       string    `json:"status"`
}

// EventBusMetrics contains metrics about event bus operations
type EventBusMetrics struct {
	EventsPublished   uint64 `json:"events_published"`
	EventsReceived    uint64 `json:"events_received"`
	EventsProcessed   uint64 `json:"events_processed"`
	EventsFailed      uint64 `json:"events_failed"`
	SubscriptionCount int    `json:"subscription_count"`
	ConnectionUptime  time.Duration `json:"connection_uptime"`
	LastEventTime     time.Time `json:"last_event_time"`
}

// NATSEventBus implements EventBus using NATS
type NATSEventBus struct {
	mu             sync.RWMutex
	conn           *nats.Conn
	config         EventBusConfig
	subscriptions  map[string]*Subscription
	metrics        EventBusMetrics
	health         HealthStatus
	logger         *logrus.Logger
	connectedAt    time.Time
}

// NewNATSEventBus creates a new NATS-based event bus
func NewNATSEventBus(logger *logrus.Logger) *NATSEventBus {
	return &NATSEventBus{
		subscriptions: make(map[string]*Subscription),
		logger:        logger,
		health: HealthStatus{
			Connected: false,
			Status:    "disconnected",
		},
	}
}

// Connect establishes connection to NATS
func (n *NATSEventBus) Connect(ctx context.Context, config EventBusConfig) error {
	n.mu.Lock()
	defer n.mu.Unlock()

	n.config = config
	
	opts := []nats.Option{
		nats.Name(config.ClientID),
		nats.MaxReconnects(config.MaxReconnects),
		nats.ReconnectWait(config.ReconnectWait),
		nats.PingInterval(config.PingInterval),
		nats.MaxPingsOutstanding(3),
	}

	// Add connection handlers
	opts = append(opts, nats.ConnectHandler(func(nc *nats.Conn) {
		n.logger.Info("Connected to NATS")
		n.mu.Lock()
		n.health.Connected = true
		n.health.Status = "connected"
		n.health.LastPing = time.Now()
		n.connectedAt = time.Now()
		n.mu.Unlock()
	}))

	opts = append(opts, nats.DisconnectErrHandler(func(nc *nats.Conn, err error) {
		n.logger.WithError(err).Warn("Disconnected from NATS")
		n.mu.Lock()
		n.health.Connected = false
		n.health.Status = "disconnected"
		if err != nil {
			n.health.LastError = err.Error()
		}
		n.mu.Unlock()
	}))

	opts = append(opts, nats.ReconnectHandler(func(nc *nats.Conn) {
		n.logger.Info("Reconnected to NATS")
		n.mu.Lock()
		n.health.Connected = true
		n.health.Status = "connected"
		n.health.Reconnects++
		n.mu.Unlock()
	}))

	conn, err := nats.Connect(config.URL, opts...)
	if err != nil {
		n.health.LastError = err.Error()
		return fmt.Errorf("failed to connect to NATS: %w", err)
	}

	n.conn = conn
	n.connectedAt = time.Now()
	return nil
}

// Disconnect closes the connection to NATS
func (n *NATSEventBus) Disconnect() error {
	n.mu.Lock()
	defer n.mu.Unlock()

	if n.conn != nil {
		// Unsubscribe all active subscriptions
		for _, sub := range n.subscriptions {
			if sub.natsSubscription != nil {
				sub.natsSubscription.Unsubscribe()
			}
		}
		
		n.conn.Close()
		n.conn = nil
	}

	n.health.Connected = false
	n.health.Status = "disconnected"
	return nil
}

// Publish publishes an event to the bus
func (n *NATSEventBus) Publish(ctx context.Context, event *OrchestrationEvent) error {
	n.mu.RLock()
	conn := n.conn
	n.mu.RUnlock()

	if conn == nil {
		return fmt.Errorf("not connected to NATS")
	}

	// Generate event ID if not provided
	if event.ID == "" {
		event.ID = uuid.New().String()
	}

	// Set timestamp if not provided
	if event.Timestamp.IsZero() {
		event.Timestamp = time.Now()
	}

	// Serialize event
	data, err := json.Marshal(event)
	if err != nil {
		return fmt.Errorf("failed to marshal event: %w", err)
	}

	// Create subject based on event type
	subject := fmt.Sprintf("orchestration.events.%s", string(event.Type))

	// Add headers for metadata
	msg := &nats.Msg{
		Subject: subject,
		Data:    data,
		Header:  make(nats.Header),
	}

	msg.Header.Add("event-id", event.ID)
	msg.Header.Add("event-type", string(event.Type))
	msg.Header.Add("event-source", event.Source)
	msg.Header.Add("priority", fmt.Sprintf("%d", event.Priority))

	if event.TTL > 0 {
		msg.Header.Add("ttl", event.TTL.String())
	}

	// Publish with timeout
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
		if err := conn.PublishMsg(msg); err != nil {
			n.mu.Lock()
			n.metrics.EventsFailed++
			n.mu.Unlock()
			return fmt.Errorf("failed to publish event: %w", err)
		}
	}

	// Update metrics
	n.mu.Lock()
	n.metrics.EventsPublished++
	n.metrics.LastEventTime = time.Now()
	n.mu.Unlock()

	n.logger.WithFields(logrus.Fields{
		"event_id":   event.ID,
		"event_type": event.Type,
		"source":     event.Source,
		"priority":   event.Priority,
	}).Debug("Event published")

	return nil
}

// Subscribe subscribes to events of specific types
func (n *NATSEventBus) Subscribe(ctx context.Context, eventTypes []EventType, handler EventHandler) (*Subscription, error) {
	n.mu.Lock()
	defer n.mu.Unlock()

	if n.conn == nil {
		return nil, fmt.Errorf("not connected to NATS")
	}

	subscription := &Subscription{
		ID:        uuid.New().String(),
		Handler:   handler,
		Active:    true,
		CreatedAt: time.Now(),
	}

	// Create subject pattern for multiple event types
	var subjects []string
	for _, eventType := range eventTypes {
		subjects = append(subjects, fmt.Sprintf("orchestration.events.%s", string(eventType)))
	}

	// For simplicity, subscribe to each type separately
	// In production, consider using subject wildcards or streaming
	var natsSubscriptions []*nats.Subscription
	
	for _, subject := range subjects {
		natsSub, err := n.conn.QueueSubscribe(subject, n.config.QueueGroup, func(msg *nats.Msg) {
			n.handleNATSMessage(msg, handler)
		})
		
		if err != nil {
			// Clean up any successful subscriptions
			for _, sub := range natsSubscriptions {
				sub.Unsubscribe()
			}
			return nil, fmt.Errorf("failed to subscribe to %s: %w", subject, err)
		}
		
		natsSubscriptions = append(natsSubscriptions, natsSub)
	}

	// Store the first subscription for reference (simplified)
	if len(natsSubscriptions) > 0 {
		subscription.natsSubscription = natsSubscriptions[0]
		subscription.Topic = subjects[0] // Store first subject
	}

	n.subscriptions[subscription.ID] = subscription
	n.metrics.SubscriptionCount = len(n.subscriptions)

	n.logger.WithFields(logrus.Fields{
		"subscription_id": subscription.ID,
		"handler_id":      handler.GetHandlerID(),
		"event_types":     eventTypes,
	}).Info("Subscribed to events")

	return subscription, nil
}

// SubscribeToAll subscribes to all events
func (n *NATSEventBus) SubscribeToAll(ctx context.Context, handler EventHandler) (*Subscription, error) {
	n.mu.Lock()
	defer n.mu.Unlock()

	if n.conn == nil {
		return nil, fmt.Errorf("not connected to NATS")
	}

	subscription := &Subscription{
		ID:        uuid.New().String(),
		Topic:     "orchestration.events.*",
		Handler:   handler,
		Active:    true,
		CreatedAt: time.Now(),
	}

	natsSub, err := n.conn.QueueSubscribe("orchestration.events.*", n.config.QueueGroup, func(msg *nats.Msg) {
		n.handleNATSMessage(msg, handler)
	})

	if err != nil {
		return nil, fmt.Errorf("failed to subscribe to all events: %w", err)
	}

	subscription.natsSubscription = natsSub
	n.subscriptions[subscription.ID] = subscription
	n.metrics.SubscriptionCount = len(n.subscriptions)

	n.logger.WithFields(logrus.Fields{
		"subscription_id": subscription.ID,
		"handler_id":      handler.GetHandlerID(),
	}).Info("Subscribed to all events")

	return subscription, nil
}

// handleNATSMessage processes incoming NATS messages
func (n *NATSEventBus) handleNATSMessage(msg *nats.Msg, handler EventHandler) {
	// Update metrics
	n.mu.Lock()
	n.metrics.EventsReceived++
	n.mu.Unlock()

	// Deserialize event
	var event OrchestrationEvent
	if err := json.Unmarshal(msg.Data, &event); err != nil {
		n.logger.WithError(err).Error("Failed to unmarshal event")
		n.mu.Lock()
		n.metrics.EventsFailed++
		n.mu.Unlock()
		return
	}

	// Create context with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Handle the event
	if err := handler.HandleEvent(ctx, &event); err != nil {
		n.logger.WithError(err).WithFields(logrus.Fields{
			"event_id":   event.ID,
			"event_type": event.Type,
			"handler_id": handler.GetHandlerID(),
		}).Error("Failed to handle event")
		
		n.mu.Lock()
		n.metrics.EventsFailed++
		n.mu.Unlock()
		return
	}

	// Update success metrics
	n.mu.Lock()
	n.metrics.EventsProcessed++
	n.mu.Unlock()
}

// GetHealth returns the health status of the event bus
func (n *NATSEventBus) GetHealth() HealthStatus {
	n.mu.RLock()
	defer n.mu.RUnlock()

	health := n.health
	if n.conn != nil && n.conn.IsConnected() {
		health.Connected = true
		health.Status = "connected"
		health.LastPing = time.Now()
	}

	return health
}

// GetMetrics returns metrics about the event bus
func (n *NATSEventBus) GetMetrics() EventBusMetrics {
	n.mu.RLock()
	defer n.mu.RUnlock()

	metrics := n.metrics
	if !n.connectedAt.IsZero() {
		metrics.ConnectionUptime = time.Since(n.connectedAt)
	}

	return metrics
}

// Unsubscribe removes a subscription
func (n *NATSEventBus) Unsubscribe(subscriptionID string) error {
	n.mu.Lock()
	defer n.mu.Unlock()

	subscription, exists := n.subscriptions[subscriptionID]
	if !exists {
		return fmt.Errorf("subscription %s not found", subscriptionID)
	}

	if subscription.natsSubscription != nil {
		if err := subscription.natsSubscription.Unsubscribe(); err != nil {
			return fmt.Errorf("failed to unsubscribe: %w", err)
		}
	}

	subscription.Active = false
	delete(n.subscriptions, subscriptionID)
	n.metrics.SubscriptionCount = len(n.subscriptions)

	n.logger.WithField("subscription_id", subscriptionID).Info("Unsubscribed from events")
	return nil
}