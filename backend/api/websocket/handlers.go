package websocket

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/mux"
	"github.com/gorilla/websocket"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/sirupsen/logrus"

	"github.com/khryptorgraphics/novacron/backend/core/vm"
)

var (
	activeConnections = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "novacron_websocket_connections_active",
		Help: "Number of active WebSocket connections",
	}, []string{"endpoint", "client_type"})

	messagesSent = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "novacron_websocket_messages_sent_total",
		Help: "Total number of WebSocket messages sent",
	}, []string{"endpoint", "message_type"})

	messagesReceived = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "novacron_websocket_messages_received_total",
		Help: "Total number of WebSocket messages received",
	}, []string{"endpoint", "message_type"})

	connectionDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "novacron_websocket_connection_duration_seconds",
		Help:    "Duration of WebSocket connections",
		Buckets: prometheus.ExponentialBuckets(1, 2, 10),
	}, []string{"endpoint"})
)

// AllowedOrigins configures the browser Origins permitted to open a
// cross-origin WebSocket connection (mirrors cfg.CORS.AllowedOrigins /
// buildCORSHandler's "*" convention). Passed as a NewWebSocketHandler dep.
type AllowedOrigins []string

// MetricsProvider returns one host-metrics sample, or nil when unavailable.
// Passed as a NewWebSocketHandler dep.
type MetricsProvider func() map[string]interface{}

// MetricsInterval overrides the default metrics broadcast cadence. Passed as
// a NewWebSocketHandler dep.
type MetricsInterval time.Duration

// WebSocketHandler manages WebSocket connections for real-time features
type WebSocketHandler struct {
	vmManager      vmLookup
	consoleManager consoleService
	logger         *logrus.Logger

	upgrader websocket.Upgrader

	allowedOrigins  []string
	metricsProvider MetricsProvider
	metricsInterval time.Duration // sampling cadence for the shared cache below

	// metricsCache holds the one shared sample every metrics client's own
	// pump reads at its own interval (per Resolution (a): per-client
	// interval/sources semantics are kept; only alerts/logs use broadcast
	// fan-out).
	metricsCacheMu sync.RWMutex
	metricsCache   map[string]interface{}

	// Connection pools
	consoleClients map[string][]*WebSocketClient
	metricsClients []*WebSocketClient
	alertClients   []*WebSocketClient
	logClients     map[string][]*WebSocketClient

	clientsMutex sync.RWMutex

	// Broadcasting channels (alerts/logs only — metrics uses the per-client
	// paced metricsWritePump + shared cache below instead; see Resolution (a)).
	alertsBroadcast chan AlertMessage
	logsBroadcast   chan LogMessage

	ctx    context.Context
	cancel context.CancelFunc
}

type vmLookup interface {
	GetVM(vmID string) (*vm.VM, error)
}

type consoleService interface {
	CreateConsoleSession(ctx context.Context, vmID string) (string, error)
	SendInput(ctx context.Context, sessionID string, input string) error
	StreamOutput(ctx context.Context, sessionID string, output chan<- string)
}

// WebSocketClient represents a connected WebSocket client
type WebSocketClient struct {
	ID           string
	Connection   *websocket.Conn
	ClientType   string
	Filters      map[string]interface{} // access ONLY via getFilters/setFilters below (read/written across pump + broadcast goroutines)
	LastActivity time.Time
	ConnectedAt  time.Time
	UserID       string
	Roles        []string

	filtersMu sync.RWMutex
	send      chan []byte
	ctx       context.Context
	cancel    context.CancelFunc
}

func (c *WebSocketClient) getFilters() map[string]interface{} {
	c.filtersMu.RLock()
	defer c.filtersMu.RUnlock()
	return c.Filters
}

func (c *WebSocketClient) setFilters(f map[string]interface{}) {
	c.filtersMu.Lock()
	defer c.filtersMu.Unlock()
	c.Filters = f
}

// ConsoleMessage represents console output message
type ConsoleMessage struct {
	Type      string    `json:"type"`
	VMID      string    `json:"vm_id"`
	SessionID string    `json:"session_id"`
	Data      string    `json:"data"`
	Timestamp time.Time `json:"timestamp"`
}

// MetricsMessage represents real-time metrics message
type MetricsMessage struct {
	Type      string                 `json:"type"`
	Source    string                 `json:"source"`
	Metrics   map[string]interface{} `json:"metrics"`
	Timestamp time.Time              `json:"timestamp"`
	Labels    map[string]string      `json:"labels,omitempty"`
}

// AlertMessage represents an alert notification message. Shape matches
// frontend/src/lib/ws/useAdminWebSocket.ts's AdminWebSocketMessage contract:
// {type, data, timestamp}. Type is one of useAdminWebSocket's union; Data
// carries the type-specific payload it reads (e.g. data.severity/data.title
// for "security_alert", data.id/data.status/data.previous_status for
// "vm_status").
type AlertMessage struct {
	Type      string                 `json:"type"`
	Data      map[string]interface{} `json:"data"`
	Timestamp time.Time              `json:"timestamp"`
}

// VMStatusMessage is PublishVMStatus's typed input.
type VMStatusMessage struct {
	ID             string `json:"id"`
	Name           string `json:"name,omitempty"`
	Status         string `json:"status"`
	PreviousStatus string `json:"previous_status,omitempty"`
}

// LogMessage represents log streaming message
type LogMessage struct {
	Type      string                 `json:"type"`
	Source    string                 `json:"source"`
	Level     string                 `json:"level"`
	Message   string                 `json:"message"`
	Timestamp time.Time              `json:"timestamp"`
	Component string                 `json:"component,omitempty"`
	VMID      string                 `json:"vm_id,omitempty"`
	Labels    map[string]string      `json:"labels,omitempty"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
}

// WebSocketMessage represents a generic WebSocket message
type WebSocketMessage struct {
	Type      string                 `json:"type"`
	Action    string                 `json:"action,omitempty"`
	Data      interface{}            `json:"data,omitempty"`
	Filters   map[string]interface{} `json:"filters,omitempty"`
	Timestamp time.Time              `json:"timestamp"`
}

// NewWebSocketHandler creates a new WebSocket handler
func NewWebSocketHandler(vmManager vmLookup, consoleManager consoleService, deps ...interface{}) *WebSocketHandler {
	ctx, cancel := context.WithCancel(context.Background())
	var logger *logrus.Logger
	var allowedOrigins []string
	var metricsProvider MetricsProvider
	metricsInterval := 5 * time.Second
	for _, dependency := range deps {
		switch candidate := dependency.(type) {
		case *logrus.Logger:
			if candidate != nil {
				logger = candidate
			}
		case AllowedOrigins:
			allowedOrigins = []string(candidate)
		case MetricsProvider:
			if candidate != nil {
				metricsProvider = candidate
			}
		case MetricsInterval:
			if candidate > 0 {
				metricsInterval = time.Duration(candidate)
			}
		}
	}
	if logger == nil {
		logger = logrus.New()
	}

	handler := &WebSocketHandler{
		vmManager:      vmManager,
		consoleManager: consoleManager,
		logger:         logger,

		allowedOrigins:  allowedOrigins,
		metricsProvider: metricsProvider,
		metricsInterval: metricsInterval,

		upgrader: websocket.Upgrader{
			ReadBufferSize:  1024,
			WriteBufferSize: 1024,
			// "bearer" lets a browser WebSocket handshake (which cannot set an
			// Authorization header) authenticate via
			// `Sec-WebSocket-Protocol: bearer, <token>` instead; the caller's
			// auth middleware validates the token and this upgrader echoes the
			// subprotocol back so the browser accepts the handshake.
			Subprotocols: []string{"bearer"},
		},

		consoleClients: make(map[string][]*WebSocketClient),
		metricsClients: make([]*WebSocketClient, 0),
		alertClients:   make([]*WebSocketClient, 0),
		logClients:     make(map[string][]*WebSocketClient),

		alertsBroadcast: make(chan AlertMessage, 100),
		logsBroadcast:    make(chan LogMessage, 100),

		ctx:    ctx,
		cancel: cancel,
	}
	handler.upgrader.CheckOrigin = handler.checkOrigin

	// Start background workers
	go handler.broadcastWorker()
	go handler.cleanupWorker()
	go handler.metricsProducerLoop()

	return handler
}

// checkOrigin rejects cross-site WebSocket hijacking (CSWSH): requests
// without an Origin header (non-browser clients) and same-origin requests are
// allowed; everything else must appear in the configured allowedOrigins list
// (or that list must contain "*", matching buildCORSHandler's convention).
func (h *WebSocketHandler) checkOrigin(r *http.Request) bool {
	origin := r.Header.Get("Origin")
	if origin == "" {
		return true
	}
	o, err := url.Parse(origin)
	if err != nil || o.Host == "" {
		return false
	}
	if strings.EqualFold(o.Host, r.Host) {
		return true
	}
	for _, allowed := range h.allowedOrigins {
		allowed = strings.TrimSpace(allowed)
		if allowed == "*" || strings.EqualFold(allowed, origin) {
			return true
		}
		if a, err := url.Parse(allowed); err == nil && a.Host != "" &&
			strings.EqualFold(a.Scheme, o.Scheme) && strings.EqualFold(a.Host, o.Host) {
			return true
		}
	}
	return false
}

// RegisterWebSocketRoutes registers WebSocket API routes
func (h *WebSocketHandler) RegisterWebSocketRoutes(router *mux.Router, require func(string, http.HandlerFunc) http.Handler) {
	for _, prefix := range []string{"/api/ws", "/ws"} {
		wsRouter := router.PathPrefix(prefix).Subrouter()

		// Console WebSocket (operator+)
		wsRouter.Handle("/console/{vmId}", require("operator", h.HandleConsoleWebSocket)).Methods("GET")

		// Metrics streaming (viewer+)
		wsRouter.Handle("/metrics", require("viewer", h.HandleMetricsWebSocket)).Methods("GET")

		// Alert notifications (viewer+)
		wsRouter.Handle("/alerts", require("viewer", h.HandleAlertsWebSocket)).Methods("GET")

		// Log streaming (admin+)
		wsRouter.Handle("/logs", require("admin", h.HandleLogsWebSocket)).Methods("GET")
		wsRouter.Handle("/logs/{source}", require("admin", h.HandleSourceLogsWebSocket)).Methods("GET")
	}
}

// HandleConsoleWebSocket handles /ws/console/{vmId}
// @Summary VM console WebSocket
// @Description Connect to VM console via WebSocket for real-time terminal access
// @Tags WebSocket
// @Param vmId path string true "VM ID"
// @Param session query string false "Console session ID"
// @Success 101 "Switching Protocols"
// @Failure 400 "Bad Request"
// @Failure 401 "Unauthorized"
// @Failure 404 "VM not found"
// @Failure 500 "Internal Server Error"
// @Router /ws/console/{vmId} [get]
func (h *WebSocketHandler) HandleConsoleWebSocket(w http.ResponseWriter, r *http.Request) {
	if h.vmManager == nil || h.consoleManager == nil {
		http.Error(w, "VM console websocket is not supported by the canonical backend", http.StatusNotImplemented)
		return
	}

	timer := prometheus.NewTimer(connectionDuration.WithLabelValues("console"))
	defer timer.ObserveDuration()

	vars := mux.Vars(r)
	vmID := vars["vmId"]

	if vmID == "" {
		http.Error(w, "VM ID is required", http.StatusBadRequest)
		return
	}

	// Validate VM exists and is accessible
	vmInfo, err := h.vmManager.GetVM(vmID)
	if err != nil {
		if err == vm.ErrVMNotFound {
			http.Error(w, "VM not found", http.StatusNotFound)
			return
		}
		h.logger.WithError(err).Error("Failed to get VM info for console")
		http.Error(w, "Internal server error", http.StatusInternalServerError)
		return
	}

	// Check VM is running
	if vmInfo.State() != vm.StateRunning {
		http.Error(w, "VM must be running for console access", http.StatusConflict)
		return
	}

	// Upgrade connection
	conn, err := h.upgrader.Upgrade(w, r, nil)
	if err != nil {
		h.logger.WithError(err).Error("Failed to upgrade WebSocket connection")
		return
	}

	// Get or create console session
	sessionID := r.URL.Query().Get("session")
	if sessionID == "" {
		sessionID, err = h.consoleManager.CreateConsoleSession(r.Context(), vmID)
		if err != nil {
			h.logger.WithError(err).Error("Failed to create console session")
			conn.Close()
			return
		}
	}

	// Create client
	clientCtx, clientCancel := context.WithCancel(h.ctx)
	client := &WebSocketClient{
		ID:           h.generateClientID(),
		Connection:   conn,
		ClientType:   "console",
		Filters:      map[string]interface{}{"vm_id": vmID, "session_id": sessionID},
		LastActivity: time.Now(),
		ConnectedAt:  time.Now(),
		UserID:       h.getUserIDFromRequest(r),
		Roles:        h.getUserRolesFromRequest(r),
		send:         make(chan []byte, 256),
		ctx:          clientCtx,
		cancel:       clientCancel,
	}

	// Add to client pool
	h.addConsoleClient(vmID, client)
	activeConnections.WithLabelValues("console", "vm").Inc()

	h.logger.WithFields(logrus.Fields{
		"client_id":  client.ID,
		"vm_id":      vmID,
		"session_id": sessionID,
		"user_id":    client.UserID,
	}).Info("Console WebSocket client connected")

	// Start client handlers
	go h.consoleWritePump(client, vmID, sessionID)
	go h.consoleReadPump(client, vmID, sessionID)
}

// HandleMetricsWebSocket handles /ws/metrics
// @Summary Metrics streaming WebSocket
// @Description Stream real-time system and VM metrics
// @Tags WebSocket
// @Param sources query string false "Comma-separated metric sources"
// @Param interval query int false "Update interval in seconds" default(5)
// @Success 101 "Switching Protocols"
// @Failure 400 "Bad Request"
// @Failure 401 "Unauthorized"
// @Failure 500 "Internal Server Error"
// @Router /ws/metrics [get]
func (h *WebSocketHandler) HandleMetricsWebSocket(w http.ResponseWriter, r *http.Request) {
	timer := prometheus.NewTimer(connectionDuration.WithLabelValues("metrics"))
	defer timer.ObserveDuration()

	// Parse filters
	sources := h.parseCommaSeparated(r.URL.Query().Get("sources"))
	interval := h.parseIntWithDefault(r.URL.Query().Get("interval"), 5)

	if interval < 1 || interval > 300 {
		http.Error(w, "Interval must be between 1 and 300 seconds", http.StatusBadRequest)
		return
	}

	// Upgrade connection
	conn, err := h.upgrader.Upgrade(w, r, nil)
	if err != nil {
		h.logger.WithError(err).Error("Failed to upgrade WebSocket connection")
		return
	}

	// Create client
	clientCtx, clientCancel := context.WithCancel(h.ctx)
	client := &WebSocketClient{
		ID:           h.generateClientID(),
		Connection:   conn,
		ClientType:   "metrics",
		Filters:      map[string]interface{}{"sources": sources, "interval": interval},
		LastActivity: time.Now(),
		ConnectedAt:  time.Now(),
		UserID:       h.getUserIDFromRequest(r),
		Roles:        h.getUserRolesFromRequest(r),
		send:         make(chan []byte, 256),
		ctx:          clientCtx,
		cancel:       clientCancel,
	}

	// Add to client pool
	h.addMetricsClient(client)
	activeConnections.WithLabelValues("metrics", "system").Inc()

	h.logger.WithFields(logrus.Fields{
		"client_id": client.ID,
		"sources":   sources,
		"interval":  interval,
		"user_id":   client.UserID,
	}).Info("Metrics WebSocket client connected")

	// Start client handlers
	go h.metricsWritePump(client)
	go h.metricsReadPump(client)
}

// HandleAlertsWebSocket handles /ws/alerts
// @Summary Alert notifications WebSocket
// @Description Stream real-time alert notifications
// @Tags WebSocket
// @Param severity query string false "Filter by severity (comma-separated)"
// @Param sources query string false "Filter by sources (comma-separated)"
// @Success 101 "Switching Protocols"
// @Failure 400 "Bad Request"
// @Failure 401 "Unauthorized"
// @Failure 500 "Internal Server Error"
// @Router /ws/alerts [get]
func (h *WebSocketHandler) HandleAlertsWebSocket(w http.ResponseWriter, r *http.Request) {
	timer := prometheus.NewTimer(connectionDuration.WithLabelValues("alerts"))
	defer timer.ObserveDuration()

	// Parse filters
	severities := h.parseCommaSeparated(r.URL.Query().Get("severity"))
	sources := h.parseCommaSeparated(r.URL.Query().Get("sources"))

	// Upgrade connection
	conn, err := h.upgrader.Upgrade(w, r, nil)
	if err != nil {
		h.logger.WithError(err).Error("Failed to upgrade WebSocket connection")
		return
	}

	// Create client
	clientCtx, clientCancel := context.WithCancel(h.ctx)
	client := &WebSocketClient{
		ID:           h.generateClientID(),
		Connection:   conn,
		ClientType:   "alerts",
		Filters:      map[string]interface{}{"severities": severities, "sources": sources},
		LastActivity: time.Now(),
		ConnectedAt:  time.Now(),
		UserID:       h.getUserIDFromRequest(r),
		Roles:        h.getUserRolesFromRequest(r),
		send:         make(chan []byte, 256),
		ctx:          clientCtx,
		cancel:       clientCancel,
	}

	// Add to client pool
	h.addAlertClient(client)
	activeConnections.WithLabelValues("alerts", "notification").Inc()

	h.logger.WithFields(logrus.Fields{
		"client_id":  client.ID,
		"severities": severities,
		"sources":    sources,
		"user_id":    client.UserID,
	}).Info("Alerts WebSocket client connected")

	// Start client handlers
	go h.genericWritePump(client, "alerts")
	go h.alertsReadPump(client)
}

// HandleLogsWebSocket handles /ws/logs
// @Summary Log streaming WebSocket
// @Description Stream real-time log entries from all sources
// @Tags WebSocket
// @Param level query string false "Filter by log level (comma-separated)"
// @Param components query string false "Filter by components (comma-separated)"
// @Success 101 "Switching Protocols"
// @Failure 400 "Bad Request"
// @Failure 401 "Unauthorized"
// @Failure 500 "Internal Server Error"
// @Router /ws/logs [get]
func (h *WebSocketHandler) HandleLogsWebSocket(w http.ResponseWriter, r *http.Request) {
	h.handleLogsWebSocket(w, r, "all")
}

// HandleSourceLogsWebSocket handles /ws/logs/{source}
// @Summary Source-specific log streaming WebSocket
// @Description Stream real-time log entries from a specific source
// @Tags WebSocket
// @Param source path string true "Log source (vm, system, audit)"
// @Param level query string false "Filter by log level (comma-separated)"
// @Param vm_id query string false "Filter by VM ID (for vm source)"
// @Success 101 "Switching Protocols"
// @Failure 400 "Bad Request"
// @Failure 401 "Unauthorized"
// @Failure 500 "Internal Server Error"
// @Router /ws/logs/{source} [get]
func (h *WebSocketHandler) HandleSourceLogsWebSocket(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	source := vars["source"]

	if source == "" {
		http.Error(w, "Source is required", http.StatusBadRequest)
		return
	}

	validSources := map[string]bool{"vm": true, "system": true, "audit": true}
	if !validSources[source] {
		http.Error(w, "Invalid source. Must be one of: vm, system, audit", http.StatusBadRequest)
		return
	}

	h.handleLogsWebSocket(w, r, source)
}

// Shutdown gracefully shuts down the WebSocket handler
func (h *WebSocketHandler) Shutdown() {
	h.cancel()

	h.clientsMutex.Lock()
	defer h.clientsMutex.Unlock()

	// Close all console clients
	for vmID, clients := range h.consoleClients {
		for _, client := range clients {
			client.cancel()
			client.Connection.Close()
		}
		delete(h.consoleClients, vmID)
	}

	// Close all metrics clients
	for _, client := range h.metricsClients {
		client.cancel()
		client.Connection.Close()
	}
	h.metricsClients = nil

	// Close all alert clients
	for _, client := range h.alertClients {
		client.cancel()
		client.Connection.Close()
	}
	h.alertClients = nil

	// Close all log clients
	for source, clients := range h.logClients {
		for _, client := range clients {
			client.cancel()
			client.Connection.Close()
		}
		delete(h.logClients, source)
	}

	h.logger.Info("WebSocket handler shutdown complete")
}

// Internal methods

func (h *WebSocketHandler) handleLogsWebSocket(w http.ResponseWriter, r *http.Request, source string) {
	timer := prometheus.NewTimer(connectionDuration.WithLabelValues("logs"))
	defer timer.ObserveDuration()

	// Parse filters
	levels := h.parseCommaSeparated(r.URL.Query().Get("level"))
	components := h.parseCommaSeparated(r.URL.Query().Get("components"))
	vmID := r.URL.Query().Get("vm_id")

	// Upgrade connection
	conn, err := h.upgrader.Upgrade(w, r, nil)
	if err != nil {
		h.logger.WithError(err).Error("Failed to upgrade WebSocket connection")
		return
	}

	// Create client
	clientCtx, clientCancel := context.WithCancel(h.ctx)
	client := &WebSocketClient{
		ID:           h.generateClientID(),
		Connection:   conn,
		ClientType:   "logs",
		Filters:      map[string]interface{}{"source": source, "levels": levels, "components": components, "vm_id": vmID},
		LastActivity: time.Now(),
		ConnectedAt:  time.Now(),
		UserID:       h.getUserIDFromRequest(r),
		Roles:        h.getUserRolesFromRequest(r),
		send:         make(chan []byte, 256),
		ctx:          clientCtx,
		cancel:       clientCancel,
	}

	// Add to client pool
	h.addLogClient(source, client)
	activeConnections.WithLabelValues("logs", source).Inc()

	h.logger.WithFields(logrus.Fields{
		"client_id":  client.ID,
		"source":     source,
		"levels":     levels,
		"components": components,
		"vm_id":      vmID,
		"user_id":    client.UserID,
	}).Info("Logs WebSocket client connected")

	// Start client handlers
	go h.genericWritePump(client, "logs")
	go h.logsReadPump(client, source)
}

func (h *WebSocketHandler) addConsoleClient(vmID string, client *WebSocketClient) {
	h.clientsMutex.Lock()
	defer h.clientsMutex.Unlock()

	if h.consoleClients[vmID] == nil {
		h.consoleClients[vmID] = make([]*WebSocketClient, 0)
	}
	h.consoleClients[vmID] = append(h.consoleClients[vmID], client)
}

func (h *WebSocketHandler) addMetricsClient(client *WebSocketClient) {
	h.clientsMutex.Lock()
	defer h.clientsMutex.Unlock()
	h.metricsClients = append(h.metricsClients, client)
}

func (h *WebSocketHandler) addAlertClient(client *WebSocketClient) {
	h.clientsMutex.Lock()
	defer h.clientsMutex.Unlock()
	h.alertClients = append(h.alertClients, client)
}

func (h *WebSocketHandler) addLogClient(source string, client *WebSocketClient) {
	h.clientsMutex.Lock()
	defer h.clientsMutex.Unlock()

	if h.logClients[source] == nil {
		h.logClients[source] = make([]*WebSocketClient, 0)
	}
	h.logClients[source] = append(h.logClients[source], client)
}

func (h *WebSocketHandler) removeClient(client *WebSocketClient) {
	h.clientsMutex.Lock()
	defer h.clientsMutex.Unlock()

	switch client.ClientType {
	case "console":
		vmID, ok := client.getFilters()["vm_id"].(string)
		if ok {
			clients := h.consoleClients[vmID]
			for i, c := range clients {
				if c.ID == client.ID {
					h.consoleClients[vmID] = append(clients[:i], clients[i+1:]...)
					break
				}
			}
			if len(h.consoleClients[vmID]) == 0 {
				delete(h.consoleClients, vmID)
			}
		}
		activeConnections.WithLabelValues("console", "vm").Dec()

	case "metrics":
		for i, c := range h.metricsClients {
			if c.ID == client.ID {
				h.metricsClients = append(h.metricsClients[:i], h.metricsClients[i+1:]...)
				break
			}
		}
		activeConnections.WithLabelValues("metrics", "system").Dec()

	case "alerts":
		for i, c := range h.alertClients {
			if c.ID == client.ID {
				h.alertClients = append(h.alertClients[:i], h.alertClients[i+1:]...)
				break
			}
		}
		activeConnections.WithLabelValues("alerts", "notification").Dec()

	case "logs":
		source, ok := client.getFilters()["source"].(string)
		if ok {
			clients := h.logClients[source]
			for i, c := range clients {
				if c.ID == client.ID {
					h.logClients[source] = append(clients[:i], clients[i+1:]...)
					break
				}
			}
			if len(h.logClients[source]) == 0 {
				delete(h.logClients, source)
			}
			activeConnections.WithLabelValues("logs", source).Dec()
		}
	}
}

// Pump methods for different WebSocket types
func (h *WebSocketHandler) consoleReadPump(client *WebSocketClient, vmID, sessionID string) {
	defer func() {
		client.cancel()
		client.Connection.Close()
		h.removeClient(client)
		h.logger.WithField("client_id", client.ID).Info("Console WebSocket client disconnected")
	}()

	client.Connection.SetReadLimit(512)
	client.Connection.SetReadDeadline(time.Now().Add(60 * time.Second))
	client.Connection.SetPongHandler(func(string) error {
		client.Connection.SetReadDeadline(time.Now().Add(60 * time.Second))
		return nil
	})

	for {
		select {
		case <-client.ctx.Done():
			return
		default:
			_, message, err := client.Connection.ReadMessage()
			if err != nil {
				if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
					h.logger.WithError(err).Error("Console WebSocket read error")
				}
				return
			}

			// Send input to console session
			if err := h.consoleManager.SendInput(client.ctx, sessionID, string(message)); err != nil {
				h.logger.WithError(err).Error("Failed to send console input")
			}

			messagesReceived.WithLabelValues("console", "input").Inc()
			client.LastActivity = time.Now()
		}
	}
}

func (h *WebSocketHandler) consoleWritePump(client *WebSocketClient, vmID, sessionID string) {
	ticker := time.NewTicker(54 * time.Second)
	defer func() {
		ticker.Stop()
		client.Connection.Close()
	}()

	// Start console output streaming
	outputChan := make(chan string, 100)
	go h.consoleManager.StreamOutput(client.ctx, sessionID, outputChan)

	for {
		select {
		case <-client.ctx.Done():
			return
		case output := <-outputChan:
			msg := ConsoleMessage{
				Type:      "output",
				VMID:      vmID,
				SessionID: sessionID,
				Data:      output,
				Timestamp: time.Now(),
			}

			data, _ := json.Marshal(msg)
			select {
			case client.send <- data:
				messagesSent.WithLabelValues("console", "output").Inc()
			default:
				close(client.send)
				return
			}

		case message := <-client.send:
			client.Connection.SetWriteDeadline(time.Now().Add(10 * time.Second))
			if err := client.Connection.WriteMessage(websocket.TextMessage, message); err != nil {
				return
			}

		case <-ticker.C:
			client.Connection.SetWriteDeadline(time.Now().Add(10 * time.Second))
			if err := client.Connection.WriteMessage(websocket.PingMessage, nil); err != nil {
				return
			}
		}
	}
}

func (h *WebSocketHandler) metricsReadPump(client *WebSocketClient) {
	defer func() {
		client.cancel()
		client.Connection.Close()
		h.removeClient(client)
		h.logger.WithField("client_id", client.ID).Info("Metrics WebSocket client disconnected")
	}()

	client.Connection.SetReadLimit(512)
	client.Connection.SetReadDeadline(time.Now().Add(60 * time.Second))
	client.Connection.SetPongHandler(func(string) error {
		client.Connection.SetReadDeadline(time.Now().Add(60 * time.Second))
		return nil
	})

	for {
		select {
		case <-client.ctx.Done():
			return
		default:
			_, message, err := client.Connection.ReadMessage()
			if err != nil {
				if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
					h.logger.WithError(err).Error("Metrics WebSocket read error")
				}
				return
			}

			// Handle control messages (filter updates, etc.)
			var controlMsg WebSocketMessage
			if err := json.Unmarshal(message, &controlMsg); err == nil {
				if controlMsg.Type == "update_filters" {
				client.setFilters(controlMsg.Filters)
				}
			}

			messagesReceived.WithLabelValues("metrics", "control").Inc()
			client.LastActivity = time.Now()
		}
	}
}

func (h *WebSocketHandler) alertsReadPump(client *WebSocketClient) {
	defer func() {
		client.cancel()
		client.Connection.Close()
		h.removeClient(client)
		h.logger.WithField("client_id", client.ID).Info("Alerts WebSocket client disconnected")
	}()

	client.Connection.SetReadLimit(512)
	client.Connection.SetReadDeadline(time.Now().Add(60 * time.Second))
	client.Connection.SetPongHandler(func(string) error {
		client.Connection.SetReadDeadline(time.Now().Add(60 * time.Second))
		return nil
	})

	for {
		select {
		case <-client.ctx.Done():
			return
		default:
			_, message, err := client.Connection.ReadMessage()
			if err != nil {
				if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
					h.logger.WithError(err).Error("Alerts WebSocket read error")
				}
				return
			}

			// Handle control messages
			var controlMsg WebSocketMessage
			if err := json.Unmarshal(message, &controlMsg); err == nil {
				if controlMsg.Type == "update_filters" {
					client.setFilters(controlMsg.Filters)
				}
			}

			messagesReceived.WithLabelValues("alerts", "control").Inc()
			client.LastActivity = time.Now()
		}
	}
}

func (h *WebSocketHandler) logsReadPump(client *WebSocketClient, source string) {
	defer func() {
		client.cancel()
		client.Connection.Close()
		h.removeClient(client)
		h.logger.WithField("client_id", client.ID).Info("Logs WebSocket client disconnected")
	}()

	client.Connection.SetReadLimit(512)
	client.Connection.SetReadDeadline(time.Now().Add(60 * time.Second))
	client.Connection.SetPongHandler(func(string) error {
		client.Connection.SetReadDeadline(time.Now().Add(60 * time.Second))
		return nil
	})

	for {
		select {
		case <-client.ctx.Done():
			return
		default:
			_, message, err := client.Connection.ReadMessage()
			if err != nil {
				if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
					h.logger.WithError(err).Error("Logs WebSocket read error")
				}
				return
			}

			// Handle control messages
			var controlMsg WebSocketMessage
			if err := json.Unmarshal(message, &controlMsg); err == nil {
				if controlMsg.Type == "update_filters" {
					client.setFilters(controlMsg.Filters)
				}
			}

			messagesReceived.WithLabelValues("logs", "control").Inc()
			client.LastActivity = time.Now()
		}
	}
}


// genericWritePump is the shared write loop for metrics/alerts/logs clients:
// drain client.send (fed exclusively by the fan-out in broadcastTo*Clients,
// itself fed exclusively by broadcastWorker) and ping on an idle timer. The
// per-channel broadcast reads that used to compete with this loop for
// h.metricsBroadcast/h.alertsBroadcast/h.logsBroadcast are gone — those
// channels now have exactly one consumer (broadcastWorker).
func (h *WebSocketHandler) genericWritePump(client *WebSocketClient, endpoint string) {
	ticker := time.NewTicker(54 * time.Second)
	defer func() {
		ticker.Stop()
		client.Connection.Close()
	}()
	for {
		select {
		case <-client.ctx.Done():
			return
		case message := <-client.send:
			client.Connection.SetWriteDeadline(time.Now().Add(10 * time.Second))
			if err := client.Connection.WriteMessage(websocket.TextMessage, message); err != nil {
				return
			}
		case <-ticker.C:
			client.Connection.SetWriteDeadline(time.Now().Add(10 * time.Second))
			if err := client.Connection.WriteMessage(websocket.PingMessage, nil); err != nil {
				return
			}
		}
	}
}
func (h *WebSocketHandler) broadcastWorker() {
	for {
		select {
		case <-h.ctx.Done():
			return
		case alert := <-h.alertsBroadcast:
			h.broadcastToAlertClients(alert)
		case logMsg := <-h.logsBroadcast:
			h.broadcastToLogClients(logMsg)
		}
	}
}

func (h *WebSocketHandler) cleanupWorker() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-h.ctx.Done():
			return
		case <-ticker.C:
			h.cleanupInactiveClients()
		}
	}
}

func (h *WebSocketHandler) cleanupInactiveClients() {
	cutoff := time.Now().Add(-5 * time.Minute)

	h.clientsMutex.Lock()
	defer h.clientsMutex.Unlock()

	// Clean up console clients
	for vmID, clients := range h.consoleClients {
		activeClients := make([]*WebSocketClient, 0, len(clients))
		for _, client := range clients {
			if client.LastActivity.After(cutoff) {
				activeClients = append(activeClients, client)
			} else {
				client.cancel()
				client.Connection.Close()
				activeConnections.WithLabelValues("console", "vm").Dec()
			}
		}
		if len(activeClients) == 0 {
			delete(h.consoleClients, vmID)
		} else {
			h.consoleClients[vmID] = activeClients
		}
	}

	// Similar cleanup for other client types...
}

// Helper functions
func (h *WebSocketHandler) generateClientID() string {
	return fmt.Sprintf("client-%d", time.Now().UnixNano())
}

func (h *WebSocketHandler) getUserIDFromRequest(r *http.Request) string {
	if userID, ok := r.Context().Value("user_id").(string); ok && strings.TrimSpace(userID) != "" {
		return strings.TrimSpace(userID)
	}
	return ""
}

func (h *WebSocketHandler) getUserRolesFromRequest(r *http.Request) []string {
	roleSet := make(map[string]struct{})
	if roles, ok := r.Context().Value("roles").([]string); ok {
		for _, role := range roles {
			trimmed := strings.TrimSpace(role)
			if trimmed != "" {
				roleSet[trimmed] = struct{}{}
			}
		}
	}
	if role, ok := r.Context().Value("role").(string); ok && strings.TrimSpace(role) != "" {
		roleSet[strings.TrimSpace(role)] = struct{}{}
	}

	if len(roleSet) == 0 {
		return []string{"viewer"}
	}

	roles := make([]string, 0, len(roleSet))
	for role := range roleSet {
		roles = append(roles, role)
	}
	slices.Sort(roles)
	return roles
}

func (h *WebSocketHandler) parseCommaSeparated(str string) []string {
	if str == "" {
		return []string{}
	}
	return strings.Split(str, ",")
}

func (h *WebSocketHandler) parseIntWithDefault(str string, defaultVal int) int {
	if val, err := strconv.Atoi(str); err == nil {
		return val
	}
	return defaultVal
}

// sampleMetrics reads the injected provider. A nil provider, or one that
// returns nil, yields status "no_data" — never fabricated zeros.
func (h *WebSocketHandler) sampleMetrics() map[string]interface{} {
	if h.metricsProvider == nil {
		return map[string]interface{}{"timestamp": time.Now().UnixMilli(), "status": "no_data"}
	}
	sample := h.metricsProvider()
	if sample == nil {
		return map[string]interface{}{"timestamp": time.Now().UnixMilli(), "status": "no_data"}
	}
	return sample
}

// currentMetrics returns a shallow copy of the shared cached sample (never
// mutated by callers, so a copy avoids any cross-client aliasing).
func (h *WebSocketHandler) currentMetrics() map[string]interface{} {
	h.metricsCacheMu.RLock()
	defer h.metricsCacheMu.RUnlock()
	if h.metricsCache == nil {
		return map[string]interface{}{"timestamp": time.Now().UnixMilli(), "status": "no_data"}
	}
	out := make(map[string]interface{}, len(h.metricsCache))
	for k, v := range h.metricsCache {
		out[k] = v
	}
	return out
}

// filterMetrics narrows sample to the requested sources when any are given.
// Host-level snapshots aren't naturally source-keyed, so an unmatched source
// name is simply absent rather than an error; if NONE of the requested
// sources match anything in the sample, the unfiltered sample is returned
// rather than silently emptying the response.
func filterMetrics(sample map[string]interface{}, sources []string) map[string]interface{} {
	if len(sources) == 0 {
		return sample
	}
	out := map[string]interface{}{}
	if ts, ok := sample["timestamp"]; ok {
		out["timestamp"] = ts
	}
	if st, ok := sample["status"]; ok {
		out["status"] = st
	}
	matched := false
	for _, s := range sources {
		if v, ok := sample[s]; ok {
			out[s] = v
			matched = true
		}
	}
	if !matched {
		return sample
	}
	return out
}

// metricsProducerLoop keeps the shared metricsCache fresh. Each connected
// metrics client's own metricsWritePump reads it independently, at its own
// configured interval, filtered to its own requested sources (Resolution
// (a): per-client interval/sources semantics are kept; only alerts/logs use
// broadcast fan-out).
func (h *WebSocketHandler) metricsProducerLoop() {
	ticker := time.NewTicker(h.metricsInterval)
	defer ticker.Stop()
	for {
		select {
		case <-h.ctx.Done():
			return
		case <-ticker.C:
			sample := h.sampleMetrics()
			h.metricsCacheMu.Lock()
			h.metricsCache = sample
			h.metricsCacheMu.Unlock()
		}
	}
}

// metricsWritePump is the metrics endpoint's own per-client write loop:
// unlike alerts/logs (broadcast fan-out via genericWritePump), each metrics
// client reads the shared cache at its OWN configured interval (1-300s,
// default 5) and applies its OWN requested source filter.
func (h *WebSocketHandler) metricsWritePump(client *WebSocketClient) {
	interval := 5
	filters := client.getFilters()
	sources, _ := filters["sources"].([]string)
	if iv, ok := filters["interval"].(int); ok && iv >= 1 && iv <= 300 {
		interval = iv
	}
	metricsTicker := time.NewTicker(time.Duration(interval) * time.Second)
	pingTicker := time.NewTicker(54 * time.Second)
	defer func() {
		metricsTicker.Stop()
		pingTicker.Stop()
		client.Connection.Close()
	}()

	for {
		select {
		case <-client.ctx.Done():
			return
		case <-metricsTicker.C:
			msg := MetricsMessage{Type: "metric", Source: "system", Metrics: filterMetrics(h.currentMetrics(), sources), Timestamp: time.Now()}
			data, err := json.Marshal(msg)
			if err != nil {
				h.logger.WithError(err).Error("marshal metrics message")
				continue
			}
			select {
			case client.send <- data:
				messagesSent.WithLabelValues("metrics", "update").Inc()
			default:
				h.logger.WithField("client_id", client.ID).Warn("client send queue full; disconnecting slow client")
				client.cancel()
				return
			}
		case message := <-client.send:
			client.Connection.SetWriteDeadline(time.Now().Add(10 * time.Second))
			if err := client.Connection.WriteMessage(websocket.TextMessage, message); err != nil {
				return
			}
		case <-pingTicker.C:
			client.Connection.SetWriteDeadline(time.Now().Add(10 * time.Second))
			if err := client.Connection.WriteMessage(websocket.PingMessage, nil); err != nil {
				return
			}
		}
	}
}

func (h *WebSocketHandler) matchesAlertFilters(alert AlertMessage, filters map[string]interface{}) bool {
	severity, _ := alert.Data["severity"].(string)
	severities, ok := filters["severities"].([]string)
	if ok && len(severities) > 0 {
		found := false
		for _, sev := range severities {
			if sev == severity {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}

	source, _ := alert.Data["source"].(string)
	sources, ok := filters["sources"].([]string)
	if ok && len(sources) > 0 {
		found := false
		for _, src := range sources {
			if src == source {
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

func (h *WebSocketHandler) matchesLogFilters(logMsg LogMessage, source string, filters map[string]interface{}) bool {
	// Check source filter
	if source != "all" && logMsg.Source != source {
		return false
	}

	// Check level filters
	levels, ok := filters["levels"].([]string)
	if ok && len(levels) > 0 {
		found := false
		for _, level := range levels {
			if level == logMsg.Level {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}

	// Check component filters
	components, ok := filters["components"].([]string)
	if ok && len(components) > 0 {
		found := false
		for _, comp := range components {
			if comp == logMsg.Component {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}

	// Check VM ID filter
	vmID, ok := filters["vm_id"].(string)
	if ok && vmID != "" && logMsg.VMID != vmID {
		return false
	}

	return true
}

// deliver enqueues data on the client's bounded send buffer. A full buffer
// means the client is too slow to keep up: it is disconnected rather than
// left to block the broadcaster or any other client.
func (h *WebSocketHandler) deliver(c *WebSocketClient, data []byte, endpoint string) {
	select {
	case c.send <- data:
		messagesSent.WithLabelValues(endpoint, "broadcast").Inc()
	default:
		h.logger.WithField("client_id", c.ID).Warn("client send queue full; disconnecting slow client")
		c.cancel()
		// Unblocks the read pump's pending ReadMessage promptly; its deferred
		// cleanup calls removeClient — matches consoleReadPump's pattern.
		c.Connection.Close()
	}
}


// PublishAlert queues an alert for broadcast.
func (h *WebSocketHandler) PublishAlert(msg AlertMessage) {
	select {
	case h.alertsBroadcast <- msg:
	default:
		h.logger.Warn("alerts broadcast channel full; dropping")
	}
}

// PublishLog queues a log entry for broadcast.
func (h *WebSocketHandler) PublishLog(msg LogMessage) {
	select {
	case h.logsBroadcast <- msg:
	default:
		h.logger.Warn("logs broadcast channel full; dropping")
	}
}

// PublishVMStatus is a typed convenience wrapper around PublishAlert for
// vm_status events (frontend/src/lib/ws/useAdminWebSocket.ts's "vm_status"
// case reads data.id/data.status/data.previous_status).
func (h *WebSocketHandler) PublishVMStatus(msg VMStatusMessage) {
	h.PublishAlert(AlertMessage{Type: "vm_status", Timestamp: time.Now(), Data: map[string]interface{}{
		"id": msg.ID, "name": msg.Name, "status": msg.Status, "previous_status": msg.PreviousStatus,
	}})
}

func (h *WebSocketHandler) broadcastToAlertClients(alert AlertMessage) {
	h.clientsMutex.RLock()
	clients := append([]*WebSocketClient(nil), h.alertClients...)
	h.clientsMutex.RUnlock()
	for _, c := range clients {
		if !h.matchesAlertFilters(alert, c.getFilters()) {
			continue
		}
		data, err := json.Marshal(alert)
		if err != nil {
			h.logger.WithError(err).Error("marshal alert message")
			return
		}
		h.deliver(c, data, "alerts")
	}
}

func (h *WebSocketHandler) broadcastToLogClients(logMsg LogMessage) {
	h.clientsMutex.RLock()
	seen := make(map[string]bool)
	var clients []*WebSocketClient
	for _, pool := range h.logClients {
		for _, c := range pool {
			if !seen[c.ID] {
				seen[c.ID] = true
				clients = append(clients, c)
			}
		}
	}
	h.clientsMutex.RUnlock()
	for _, c := range clients {
		filters := c.getFilters()
		source, _ := filters["source"].(string)
		if !h.matchesLogFilters(logMsg, source, filters) {
			continue
		}
		data, err := json.Marshal(logMsg)
		if err != nil {
			h.logger.WithError(err).Error("marshal log message")
			return
		}
		h.deliver(c, data, "logs")
	}
}
