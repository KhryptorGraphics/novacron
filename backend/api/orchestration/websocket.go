package orchestration

import (
	"context"
	"encoding/json"
	"net/http"
	"time"

	"github.com/gorilla/websocket"
	"github.com/sirupsen/logrus"
	"github.com/khryptorgraphics/novacron/backend/core/orchestration/events"
)

// WebSocketManager manages WebSocket connections for real-time orchestration events
type WebSocketManager struct {
	logger        *logrus.Logger
	eventBus      events.EventBus
	upgrader      websocket.Upgrader
	clients       map[*WebSocketClient]bool
	clientsMutex  sync.RWMutex
	register      chan *WebSocketClient
	unregister    chan *WebSocketClient
	broadcast     chan []byte
	ctx           context.Context
	cancel        context.CancelFunc
}

import "sync"

// WebSocketClient represents a connected WebSocket client
type WebSocketClient struct {
	id         string
	conn       *websocket.Conn
	send       chan []byte
	manager    *WebSocketManager
	filters    *EventFilters
	lastPing   time.Time
}

// EventFilters defines filters for WebSocket event subscriptions
type EventFilters struct {
	EventTypes []string `json:"event_types,omitempty"`
	Sources    []string `json:"sources,omitempty"`
	Targets    []string `json:"targets,omitempty"`
	Priorities []int    `json:"priorities,omitempty"`
}

// WebSocketMessage represents a message sent over WebSocket
type WebSocketMessage struct {
	Type      string      `json:"type"`
	Data      interface{} `json:"data,omitempty"`
	Timestamp time.Time   `json:"timestamp"`
	Error     string      `json:"error,omitempty"`
}

// NewWebSocketManager creates a new WebSocket manager
func NewWebSocketManager(logger *logrus.Logger, eventBus events.EventBus) *WebSocketManager {
	ctx, cancel := context.WithCancel(context.Background())
	
	return &WebSocketManager{
		logger:     logger,
		eventBus:   eventBus,
		upgrader: websocket.Upgrader{
			CheckOrigin: func(r *http.Request) bool {
				// In production, implement proper origin checking
				return true
			},
			ReadBufferSize:  1024,
			WriteBufferSize: 1024,
		},
		clients:    make(map[*WebSocketClient]bool),
		register:   make(chan *WebSocketClient),
		unregister: make(chan *WebSocketClient),
		broadcast:  make(chan []byte),
		ctx:        ctx,
		cancel:     cancel,
	}
}

// Start starts the WebSocket manager
func (wsm *WebSocketManager) Start() error {
	wsm.logger.Info("Starting WebSocket manager")

	// Subscribe to orchestration events
	eventHandler := events.NewEventHandlerFunc("websocket-manager", "WebSocket Event Handler", wsm.handleOrchestrationEvent)
	
	_, err := wsm.eventBus.SubscribeToAll(wsm.ctx, eventHandler)
	if err != nil {
		return fmt.Errorf("failed to subscribe to events: %w", err)
	}

	// Start the hub
	go wsm.run()

	wsm.logger.Info("WebSocket manager started")
	return nil
}

import "fmt"

// Stop stops the WebSocket manager
func (wsm *WebSocketManager) Stop() error {
	wsm.logger.Info("Stopping WebSocket manager")
	wsm.cancel()
	
	// Close all client connections
	wsm.clientsMutex.Lock()
	for client := range wsm.clients {
		client.conn.Close()
	}
	wsm.clientsMutex.Unlock()
	
	wsm.logger.Info("WebSocket manager stopped")
	return nil
}

// HandleWebSocket handles WebSocket upgrade requests
func (wsm *WebSocketManager) HandleWebSocket(w http.ResponseWriter, r *http.Request) {
	conn, err := wsm.upgrader.Upgrade(w, r, nil)
	if err != nil {
		wsm.logger.WithError(err).Error("WebSocket upgrade failed")
		return
	}

	client := &WebSocketClient{
		id:       fmt.Sprintf("client-%d", time.Now().UnixNano()),
		conn:     conn,
		send:     make(chan []byte, 256),
		manager:  wsm,
		filters:  &EventFilters{},
		lastPing: time.Now(),
	}

	wsm.register <- client

	// Start goroutines for this client
	go client.writePump()
	go client.readPump()

	wsm.logger.WithField("client_id", client.id).Info("WebSocket client connected")
}

// GetStats returns WebSocket statistics
func (wsm *WebSocketManager) GetStats() map[string]interface{} {
	wsm.clientsMutex.RLock()
	defer wsm.clientsMutex.RUnlock()
	
	return map[string]interface{}{
		"connected_clients": len(wsm.clients),
		"timestamp":        time.Now(),
	}
}

// Private methods

func (wsm *WebSocketManager) run() {
	ticker := time.NewTicker(30 * time.Second) // Ping clients every 30 seconds
	defer ticker.Stop()

	for {
		select {
		case <-wsm.ctx.Done():
			return

		case client := <-wsm.register:
			wsm.clientsMutex.Lock()
			wsm.clients[client] = true
			wsm.clientsMutex.Unlock()
			
			// Send welcome message
			welcome := WebSocketMessage{
				Type:      "connected",
				Data:      map[string]string{"client_id": client.id},
				Timestamp: time.Now(),
			}
			client.sendMessage(welcome)

		case client := <-wsm.unregister:
			wsm.clientsMutex.Lock()
			if _, ok := wsm.clients[client]; ok {
				delete(wsm.clients, client)
				close(client.send)
			}
			wsm.clientsMutex.Unlock()
			
			wsm.logger.WithField("client_id", client.id).Info("WebSocket client disconnected")

		case message := <-wsm.broadcast:
			wsm.clientsMutex.RLock()
			for client := range wsm.clients {
				select {
				case client.send <- message:
				default:
					delete(wsm.clients, client)
					close(client.send)
				}
			}
			wsm.clientsMutex.RUnlock()

		case <-ticker.C:
			// Send ping to all clients
			wsm.pingClients()
		}
	}
}

func (wsm *WebSocketManager) handleOrchestrationEvent(ctx context.Context, event *events.OrchestrationEvent) error {
	message := WebSocketMessage{
		Type:      "orchestration_event",
		Data:      event,
		Timestamp: time.Now(),
	}

	messageBytes, err := json.Marshal(message)
	if err != nil {
		wsm.logger.WithError(err).Error("Failed to marshal WebSocket message")
		return err
	}

	// Send to filtered clients
	wsm.clientsMutex.RLock()
	for client := range wsm.clients {
		if client.shouldReceiveEvent(event) {
			select {
			case client.send <- messageBytes:
			default:
				// Client buffer is full, close connection
				delete(wsm.clients, client)
				close(client.send)
			}
		}
	}
	wsm.clientsMutex.RUnlock()

	return nil
}

func (wsm *WebSocketManager) pingClients() {
	pingMessage := WebSocketMessage{
		Type:      "ping",
		Timestamp: time.Now(),
	}

	messageBytes, err := json.Marshal(pingMessage)
	if err != nil {
		wsm.logger.WithError(err).Error("Failed to marshal ping message")
		return
	}

	wsm.clientsMutex.RLock()
	staleClients := make([]*WebSocketClient, 0)
	
	for client := range wsm.clients {
		// Check for stale connections (no pong response in 2 minutes)
		if time.Since(client.lastPing) > 2*time.Minute {
			staleClients = append(staleClients, client)
			continue
		}

		select {
		case client.send <- messageBytes:
		default:
			staleClients = append(staleClients, client)
		}
	}
	wsm.clientsMutex.RUnlock()

	// Clean up stale clients
	for _, client := range staleClients {
		wsm.unregister <- client
	}
}

// WebSocketClient methods

func (c *WebSocketClient) readPump() {
	defer func() {
		c.manager.unregister <- c
		c.conn.Close()
	}()

	c.conn.SetReadLimit(512)
	c.conn.SetReadDeadline(time.Now().Add(60 * time.Second))
	c.conn.SetPongHandler(func(string) error {
		c.lastPing = time.Now()
		c.conn.SetReadDeadline(time.Now().Add(60 * time.Second))
		return nil
	})

	for {
		_, messageBytes, err := c.conn.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				c.manager.logger.WithError(err).Error("WebSocket error")
			}
			break
		}

		// Handle client message
		if err := c.handleMessage(messageBytes); err != nil {
			c.manager.logger.WithError(err).WithField("client_id", c.id).Error("Failed to handle client message")
		}
	}
}

func (c *WebSocketClient) writePump() {
	ticker := time.NewTicker(54 * time.Second)
	defer func() {
		ticker.Stop()
		c.conn.Close()
	}()

	for {
		select {
		case message, ok := <-c.send:
			c.conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
			if !ok {
				c.conn.WriteMessage(websocket.CloseMessage, []byte{})
				return
			}

			w, err := c.conn.NextWriter(websocket.TextMessage)
			if err != nil {
				return
			}
			w.Write(message)

			// Add queued messages to current message
			n := len(c.send)
			for i := 0; i < n; i++ {
				w.Write([]byte{'\n'})
				w.Write(<-c.send)
			}

			if err := w.Close(); err != nil {
				return
			}

		case <-ticker.C:
			c.conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
			if err := c.conn.WriteMessage(websocket.PingMessage, nil); err != nil {
				return
			}
		}
	}
}

func (c *WebSocketClient) handleMessage(messageBytes []byte) error {
	var message map[string]interface{}
	if err := json.Unmarshal(messageBytes, &message); err != nil {
		return fmt.Errorf("invalid JSON message: %w", err)
	}

	messageType, ok := message["type"].(string)
	if !ok {
		return fmt.Errorf("missing message type")
	}

	switch messageType {
	case "subscribe":
		return c.handleSubscribe(message)
	case "unsubscribe":
		return c.handleUnsubscribe(message)
	case "pong":
		c.lastPing = time.Now()
		return nil
	default:
		return fmt.Errorf("unknown message type: %s", messageType)
	}
}

func (c *WebSocketClient) handleSubscribe(message map[string]interface{}) error {
	filtersData, ok := message["filters"]
	if !ok {
		// Subscribe to all events
		c.filters = &EventFilters{}
		return nil
	}

	filtersBytes, err := json.Marshal(filtersData)
	if err != nil {
		return fmt.Errorf("invalid filters format: %w", err)
	}

	var filters EventFilters
	if err := json.Unmarshal(filtersBytes, &filters); err != nil {
		return fmt.Errorf("failed to parse filters: %w", err)
	}

	c.filters = &filters
	
	response := WebSocketMessage{
		Type:      "subscribed",
		Data:      map[string]interface{}{"filters": filters},
		Timestamp: time.Now(),
	}
	
	c.sendMessage(response)
	return nil
}

func (c *WebSocketClient) handleUnsubscribe(message map[string]interface{}) error {
	c.filters = &EventFilters{}
	
	response := WebSocketMessage{
		Type:      "unsubscribed",
		Timestamp: time.Now(),
	}
	
	c.sendMessage(response)
	return nil
}

func (c *WebSocketClient) sendMessage(message WebSocketMessage) {
	messageBytes, err := json.Marshal(message)
	if err != nil {
		c.manager.logger.WithError(err).Error("Failed to marshal message")
		return
	}

	select {
	case c.send <- messageBytes:
	default:
		// Client buffer is full, close connection
		close(c.send)
	}
}

func (c *WebSocketClient) shouldReceiveEvent(event *events.OrchestrationEvent) bool {
	filters := c.filters

	// Check event types filter
	if len(filters.EventTypes) > 0 {
		found := false
		for _, eventType := range filters.EventTypes {
			if string(event.Type) == eventType {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}

	// Check sources filter
	if len(filters.Sources) > 0 {
		found := false
		for _, source := range filters.Sources {
			if event.Source == source {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}

	// Check targets filter
	if len(filters.Targets) > 0 {
		found := false
		for _, target := range filters.Targets {
			if event.Target == target {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}

	// Check priorities filter
	if len(filters.Priorities) > 0 {
		found := false
		for _, priority := range filters.Priorities {
			if int(event.Priority) == priority {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}

	return true
}