//go:build !experimental

package orchestration

import (
	"context"
	"encoding/json"
	"net/http"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"
	"github.com/sirupsen/logrus"
	events "github.com/khryptorgraphics/novacron/backend/core/orchestration/events"
	auth "github.com/khryptorgraphics/novacron/backend/core/auth"
)

// WebSocketSecurityConfig defines security settings for WebSocket connections
type WebSocketSecurityConfig struct {
	// AllowedOrigins defines allowed origins for WebSocket connections
	AllowedOrigins []string
	// RequireAuthentication requires JWT token authentication
	RequireAuthentication bool
	// RateLimitConnections per minute per IP
	RateLimitConnections int
	// RateLimitMessages per minute per connection
	RateLimitMessages int
	// MaxConnections maximum concurrent connections
	MaxConnections int
	// RequirePermissions required permissions for WebSocket access
	RequirePermissions []string
}

// DefaultWebSocketSecurityConfig returns secure default configuration
func DefaultWebSocketSecurityConfig() WebSocketSecurityConfig {
	return WebSocketSecurityConfig{
		AllowedOrigins: []string{
			"http://localhost:3000",
			"https://localhost:3000",
			"http://127.0.0.1:3000",
			"https://127.0.0.1:3000",
		},
		RequireAuthentication: true,
		RateLimitConnections: 60,  // 60 connections per minute per IP
		RateLimitMessages: 300,    // 300 messages per minute per connection
		MaxConnections: 1000,
		RequirePermissions: []string{"system:read"},
	}
}

// RateLimitTracker tracks rate limits for connections and messages
type RateLimitTracker struct {
	// connectionLimits tracks connection attempts per IP
	connectionLimits map[string]*RateLimitEntry
	// messageLimits tracks message rates per client
	messageLimits map[string]*RateLimitEntry
	mutex sync.RWMutex
}

// RateLimitEntry tracks rate limiting data
type RateLimitEntry struct {
	count     int
	window    time.Time
	lastReset time.Time
}

// WebSocketManager manages WebSocket connections for real-time orchestration events
type WebSocketManager struct {
	logger          *logrus.Logger
	eventBus        events.EventBus
	jwtService      *auth.JWTService
	authService     auth.AuthService
	securityConfig  WebSocketSecurityConfig
	rateLimiter     *RateLimitTracker

	upgrader        websocket.Upgrader
	clients         map[*WebSocketClient]bool
	clientsMutex    sync.RWMutex
	register        chan *WebSocketClient
	unregister      chan *WebSocketClient
	broadcast       chan []byte
	ctx             context.Context
	cancel          context.CancelFunc
}


// WebSocketClient represents a connected WebSocket client
type WebSocketClient struct {
	id            string
	conn          *websocket.Conn
	send          chan []byte
	manager       *WebSocketManager
	filters       *EventFilters
	lastPing      time.Time
	// Security context
	userID        string
	tenantID      string
	sessionID     string
	clientIP      string
	userAgent     string
	permissions   []string
	authenticated bool
	connectedAt   time.Time
	lastActivity  time.Time
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

// NewWebSocketManager creates a new WebSocket manager with security features
func NewWebSocketManager(
	logger *logrus.Logger,
	eventBus events.EventBus,
	jwtService *auth.JWTService,
	authService auth.AuthService,
	securityConfig WebSocketSecurityConfig,
) *WebSocketManager {
	ctx, cancel := context.WithCancel(context.Background())

	if len(securityConfig.AllowedOrigins) == 0 {
		securityConfig = DefaultWebSocketSecurityConfig()
	}

	return &WebSocketManager{
		logger:         logger,
		eventBus:       eventBus,
		jwtService:     jwtService,
		authService:    authService,
		securityConfig: securityConfig,
		rateLimiter: &RateLimitTracker{
			connectionLimits: make(map[string]*RateLimitEntry),
			messageLimits:    make(map[string]*RateLimitEntry),
		},
		upgrader: websocket.Upgrader{
			CheckOrigin: func(r *http.Request) bool {
				return checkOrigin(r, securityConfig.AllowedOrigins)
			},
			ReadBufferSize:  2048,
			WriteBufferSize: 2048,
			Error: func(w http.ResponseWriter, r *http.Request, status int, reason error) {
				logger.WithFields(logrus.Fields{
					"status": status,
					"reason": reason.Error(),
					"origin": r.Header.Get("Origin"),
					"ip":     getClientIP(r),
				}).Error("WebSocket upgrade failed")
				w.WriteHeader(status)
				w.Write([]byte("WebSocket upgrade failed"))
			},
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
	wsm.logger.Info("Starting WebSocket manager with security features")

	// Subscribe to orchestration events
	eventHandler := events.NewEventHandlerFunc("websocket-manager", "WebSocket Event Handler", wsm.handleOrchestrationEvent)

	_, err := wsm.eventBus.SubscribeToAll(wsm.ctx, eventHandler)
	if err != nil {
		return fmt.Errorf("failed to subscribe to events: %w", err)
	}

	// Start the hub
	go wsm.run()

	// Start rate limit cleanup
	wsm.startRateLimitCleanup()

	wsm.logger.WithFields(logrus.Fields{
		"max_connections":        wsm.securityConfig.MaxConnections,
		"require_authentication": wsm.securityConfig.RequireAuthentication,
		"allowed_origins":        len(wsm.securityConfig.AllowedOrigins),
	}).Info("WebSocket manager started with security")
	return nil
}


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

// HandleWebSocket handles WebSocket upgrade requests with security
func (wsm *WebSocketManager) HandleWebSocket(w http.ResponseWriter, r *http.Request) {
	clientIP := getClientIP(r)
	userAgent := r.UserAgent()
	origin := r.Header.Get("Origin")

	// Check connection rate limit
	if !wsm.checkConnectionRateLimit(clientIP) {
		wsm.logger.WithFields(logrus.Fields{
			"ip":     clientIP,
			"origin": origin,
		}).Warn("WebSocket connection rate limited")
		http.Error(w, "Too many connection attempts", http.StatusTooManyRequests)
		return
	}

	// Check maximum connections
	wsm.clientsMutex.RLock()
	connectionCount := len(wsm.clients)
	wsm.clientsMutex.RUnlock()

	if connectionCount >= wsm.securityConfig.MaxConnections {
		wsm.logger.WithField("count", connectionCount).Warn("Maximum WebSocket connections reached")
		http.Error(w, "Maximum connections exceeded", http.StatusServiceUnavailable)
		return
	}

	// Extract and validate authentication if required
	var userID, tenantID, sessionID string
	var permissions []string
	authenticated := false

	if wsm.securityConfig.RequireAuthentication {
		token := extractAuthToken(r)
		if token == "" {
			wsm.logger.WithField("ip", clientIP).Warn("WebSocket connection missing authentication token")
			http.Error(w, "Authentication required", http.StatusUnauthorized)
			return
		}

		// Validate JWT token
		claims, err := wsm.jwtService.ValidateToken(token)
		if err != nil {
			wsm.logger.WithFields(logrus.Fields{
				"error": err.Error(),
				"ip":    clientIP,
			}).Warn("WebSocket authentication failed")
			http.Error(w, "Invalid authentication token", http.StatusUnauthorized)
			return
		}

		// Validate session
		session, err := wsm.authService.ValidateSession(claims.SessionID, token)
		if err != nil {
			wsm.logger.WithFields(logrus.Fields{
				"error": err.Error(),
				"user":  claims.UserID,
				"ip":    clientIP,
			}).Warn("WebSocket session validation failed")
			http.Error(w, "Invalid session", http.StatusUnauthorized)
			return
		}

		// Check required permissions
		for _, requiredPerm := range wsm.securityConfig.RequirePermissions {
			parts := strings.Split(requiredPerm, ":")
			if len(parts) != 2 {
				continue
			}
			resource, action := parts[0], parts[1]
			hasPermission, err := wsm.authService.HasPermissionInTenant(
				claims.UserID, claims.TenantID, resource, action)
			if err != nil || !hasPermission {
				wsm.logger.WithFields(logrus.Fields{
					"user":       claims.UserID,
					"permission": requiredPerm,
					"ip":         clientIP,
				}).Warn("WebSocket insufficient permissions")
				http.Error(w, "Insufficient permissions", http.StatusForbidden)
				return
			}
		}

		userID = claims.UserID
		tenantID = claims.TenantID
		sessionID = session.ID
		permissions = claims.Permissions
		authenticated = true
	}

	// Upgrade to WebSocket
	conn, err := wsm.upgrader.Upgrade(w, r, nil)
	if err != nil {
		wsm.logger.WithError(err).Error("WebSocket upgrade failed")
		return
	}

	now := time.Now()
	client := &WebSocketClient{
		id:            fmt.Sprintf("client-%d", now.UnixNano()),
		conn:          conn,
		send:          make(chan []byte, 256),
		manager:       wsm,
		filters:       &EventFilters{},
		lastPing:      now,
		userID:        userID,
		tenantID:      tenantID,
		sessionID:     sessionID,
		clientIP:      clientIP,
		userAgent:     userAgent,
		permissions:   permissions,
		authenticated: authenticated,
		connectedAt:   now,
		lastActivity:  now,
	}

	wsm.register <- client

	// Start goroutines for this client
	go client.writePump()
	go client.readPump()

	wsm.logger.WithFields(logrus.Fields{
		"client_id":     client.id,
		"user_id":       userID,
		"tenant_id":     tenantID,
		"authenticated": authenticated,
		"ip":            clientIP,
	}).Info("WebSocket client connected")
}

// GetStats returns WebSocket statistics with security metrics
func (wsm *WebSocketManager) GetStats() map[string]interface{} {
	wsm.clientsMutex.RLock()
	connectedClients := len(wsm.clients)
	authenticatedClients := 0
	for client := range wsm.clients {
		if client.authenticated {
			authenticatedClients++
		}
	}
	wsm.clientsMutex.RUnlock()

	wsm.rateLimiter.mutex.RLock()
	connectionLimitEntries := len(wsm.rateLimiter.connectionLimits)
	messageLimitEntries := len(wsm.rateLimiter.messageLimits)
	wsm.rateLimiter.mutex.RUnlock()

	return map[string]interface{}{
		"connected_clients":      connectedClients,
		"authenticated_clients":  authenticatedClients,
		"unauthenticated_clients": connectedClients - authenticatedClients,
		"max_connections":       wsm.securityConfig.MaxConnections,
		"connection_rate_limits": connectionLimitEntries,
		"message_rate_limits":    messageLimitEntries,
		"timestamp":             time.Now(),
		"security_config": map[string]interface{}{
			"require_authentication": wsm.securityConfig.RequireAuthentication,
			"rate_limit_connections": wsm.securityConfig.RateLimitConnections,
			"rate_limit_messages":    wsm.securityConfig.RateLimitMessages,
			"allowed_origins":        wsm.securityConfig.AllowedOrigins,
		},
	}
}

// Private methods

func (wsm *WebSocketManager) run() {
	ticker := time.NewTicker(30 * time.Second) // Ping clients every 30 seconds
	cleanupTicker := time.NewTicker(5 * time.Minute) // Cleanup stale connections
	defer ticker.Stop()
	defer cleanupTicker.Stop()

	for {
		select {
		case <-wsm.ctx.Done():
			return

		case client := <-wsm.register:
			wsm.clientsMutex.Lock()
			wsm.clients[client] = true
			wsm.clientsMutex.Unlock()

			// Send welcome message with security info
			welcome := WebSocketMessage{
				Type:      "connected",
				Data: map[string]interface{}{
					"client_id":     client.id,
					"authenticated": client.authenticated,
					"user_id":       client.userID,
					"tenant_id":     client.tenantID,
					"server_time":   time.Now(),
				},
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

			// Clean up message rate limits for this client
			wsm.rateLimiter.mutex.Lock()
			delete(wsm.rateLimiter.messageLimits, client.id)
			wsm.rateLimiter.mutex.Unlock()

			wsm.logger.WithFields(logrus.Fields{
				"client_id": client.id,
				"user_id":   client.userID,
				"duration":  time.Since(client.connectedAt),
			}).Info("WebSocket client disconnected")

		case message := <-wsm.broadcast:
			wsm.clientsMutex.RLock()
			for client := range wsm.clients {
				// Security: only send to authenticated clients if auth is required
				if wsm.securityConfig.RequireAuthentication && !client.authenticated {
					continue
				}

				select {
				case client.send <- message:
				default:
					delete(wsm.clients, client)
					close(client.send)
				}
			}
			wsm.clientsMutex.RUnlock()

		case <-ticker.C:
			// Send ping to all clients and check for stale connections
			wsm.pingClients()

		case <-cleanupTicker.C:
			// Clean up stale connections and rate limits
			wsm.cleanupStaleConnections()
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

	// Send to authorized and filtered clients only
	wsm.clientsMutex.RLock()
	for client := range wsm.clients {
		// Security check: only authenticated clients receive events
		if wsm.securityConfig.RequireAuthentication && !client.authenticated {
			continue
		}

		// Check if client has permission for this event type
		if !client.hasPermissionForEvent(event) {
			continue
		}

		// Check event filters
		if client.shouldReceiveEvent(event) {
			select {
			case client.send <- messageBytes:
			default:
				// Client buffer is full, close connection
				delete(wsm.clients, client)
				close(client.send)
				wsm.logger.WithField("client_id", client.id).Warn("Client buffer full, closing connection")
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

	// Set security limits
	c.conn.SetReadLimit(8192) // 8KB max message size
	c.conn.SetReadDeadline(time.Now().Add(60 * time.Second))
	c.conn.SetPongHandler(func(string) error {
		c.lastPing = time.Now()
		c.lastActivity = time.Now()
		c.conn.SetReadDeadline(time.Now().Add(60 * time.Second))
		return nil
	})

	for {
		_, messageBytes, err := c.conn.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				c.manager.logger.WithFields(logrus.Fields{
					"error":     err.Error(),
					"client_id": c.id,
					"user_id":   c.userID,
					"ip":        c.clientIP,
				}).Error("WebSocket unexpected close")
			}
			break
		}

		// Handle client message with error logging
		if err := c.handleMessage(messageBytes); err != nil {
			c.manager.logger.WithFields(logrus.Fields{
				"error":      err.Error(),
				"client_id":  c.id,
				"user_id":    c.userID,
				"ip":         c.clientIP,
				"message_size": len(messageBytes),
			}).Error("Failed to handle WebSocket message")
			
			// Close connection on repeated errors
			break
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
	// Check message rate limit
	if !c.manager.checkMessageRateLimit(c.id) {
		c.manager.logger.WithField("client_id", c.id).Warn("WebSocket message rate limited")
		return fmt.Errorf("message rate limit exceeded")
	}

	// Update activity timestamp
	c.lastActivity = time.Now()

	// Validate message size (prevent DoS)
	if len(messageBytes) > 8192 { // 8KB limit
		c.manager.logger.WithFields(logrus.Fields{
			"client_id": c.id,
			"size":      len(messageBytes),
		}).Warn("WebSocket message too large")
		return fmt.Errorf("message too large")
	}

	var message map[string]interface{}
	if err := json.Unmarshal(messageBytes, &message); err != nil {
		return fmt.Errorf("invalid JSON message: %w", err)
	}

	messageType, ok := message["type"].(string)
	if !ok {
		return fmt.Errorf("missing message type")
	}

	// Log suspicious activity
	if messageType != "pong" && messageType != "subscribe" && messageType != "unsubscribe" {
		c.manager.logger.WithFields(logrus.Fields{
			"client_id":    c.id,
			"message_type": messageType,
			"user_id":      c.userID,
			"ip":           c.clientIP,
		}).Warn("WebSocket suspicious message type")
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