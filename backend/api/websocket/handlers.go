package websocket

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"time"

	"github.com/gorilla/mux"
	"github.com/gorilla/websocket"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/sirupsen/logrus"
	
	"github.com/khryptorgraphics/novacron/backend/core/vm"
	"github.com/khryptorgraphics/novacron/backend/core/monitoring"
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

// WebSocketHandler manages WebSocket connections for real-time features
type WebSocketHandler struct {
	vmManager       *vm.VMManager
	consoleManager  *vm.ConsoleManager
	metricRegistry  *monitoring.MetricRegistry
	alertManager    *monitoring.AlertManager
	logger          *logrus.Logger

	upgrader        websocket.Upgrader
	
	// Connection pools
	consoleClients  map[string][]*WebSocketClient
	metricsClients  []*WebSocketClient
	alertClients    []*WebSocketClient
	logClients      map[string][]*WebSocketClient
	
	clientsMutex    sync.RWMutex
	
	// Broadcasting channels
	metricsBroadcast chan MetricsMessage
	alertsBroadcast  chan AlertMessage
	logsBroadcast    chan LogMessage
	
	ctx             context.Context
	cancel          context.CancelFunc
}

// WebSocketClient represents a connected WebSocket client
type WebSocketClient struct {
	ID           string
	Connection   *websocket.Conn
	ClientType   string
	Filters      map[string]interface{}
	LastActivity time.Time
	ConnectedAt  time.Time
	UserID       string
	Roles        []string
	
	send         chan []byte
	ctx          context.Context
	cancel       context.CancelFunc
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
	Type        string                 `json:"type"`
	Source      string                 `json:"source"`
	Metrics     map[string]interface{} `json:"metrics"`
	Timestamp   time.Time              `json:"timestamp"`
	Labels      map[string]string      `json:"labels,omitempty"`
}

// AlertMessage represents alert notification message
type AlertMessage struct {
	Type        string                 `json:"type"`
	AlertID     string                 `json:"alert_id"`
	Severity    string                 `json:"severity"`
	Title       string                 `json:"title"`
	Description string                 `json:"description"`
	Source      string                 `json:"source"`
	Timestamp   time.Time              `json:"timestamp"`
	Labels      map[string]string      `json:"labels,omitempty"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

// LogMessage represents log streaming message
type LogMessage struct {
	Type        string                 `json:"type"`
	Source      string                 `json:"source"`
	Level       string                 `json:"level"`
	Message     string                 `json:"message"`
	Timestamp   time.Time              `json:"timestamp"`
	Component   string                 `json:"component,omitempty"`
	VMID        string                 `json:"vm_id,omitempty"`
	Labels      map[string]string      `json:"labels,omitempty"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
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
func NewWebSocketHandler(vmManager *vm.VMManager, consoleManager *vm.ConsoleManager, metricRegistry *monitoring.MetricRegistry, alertManager *monitoring.AlertManager, logger *logrus.Logger) *WebSocketHandler {
	ctx, cancel := context.WithCancel(context.Background())
	
	handler := &WebSocketHandler{
		vmManager:      vmManager,
		consoleManager: consoleManager,
		metricRegistry: metricRegistry,
		alertManager:   alertManager,
		logger:         logger,
		
		upgrader: websocket.Upgrader{
			ReadBufferSize:  1024,
			WriteBufferSize: 1024,
			CheckOrigin: func(r *http.Request) bool {
				// In production, implement proper origin checking
				return true
			},
		},
		
		consoleClients: make(map[string][]*WebSocketClient),
		metricsClients: make([]*WebSocketClient, 0),
		alertClients:   make([]*WebSocketClient, 0),
		logClients:     make(map[string][]*WebSocketClient),
		
		metricsBroadcast: make(chan MetricsMessage, 100),
		alertsBroadcast:  make(chan AlertMessage, 100),
		logsBroadcast:    make(chan LogMessage, 100),
		
		ctx:    ctx,
		cancel: cancel,
	}
	
	// Start background workers
	go handler.broadcastWorker()
	go handler.cleanupWorker()
	
	return handler
}

// RegisterWebSocketRoutes registers WebSocket API routes
func (h *WebSocketHandler) RegisterWebSocketRoutes(router *mux.Router, require func(string, http.HandlerFunc) http.Handler) {
	wsRouter := router.PathPrefix("/ws").Subrouter()

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
	timer := prometheus.NewTimer(connectionDuration.WithLabelValues("console"))
	defer timer.ObserveDuration()

	vars := mux.Vars(r)
	vmID := vars["vmId"]

	if vmID == "" {
		http.Error(w, "VM ID is required", http.StatusBadRequest)
		return
	}

	// Validate VM exists and is accessible
	vmInfo, err := h.vmManager.GetVM(r.Context(), vmID)
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
	if vmInfo.Status != "running" {
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
		"client_id":   client.ID,
		"severities":  severities,
		"sources":     sources,
		"user_id":     client.UserID,
	}).Info("Alerts WebSocket client connected")

	// Start client handlers
	go h.alertsWritePump(client)
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
	go h.logsWritePump(client, source)
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
		vmID, ok := client.Filters["vm_id"].(string)
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
		source, ok := client.Filters["source"].(string)
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
					client.Filters = controlMsg.Filters
				}
			}

			messagesReceived.WithLabelValues("metrics", "control").Inc()
			client.LastActivity = time.Now()
		}
	}
}

func (h *WebSocketHandler) metricsWritePump(client *WebSocketClient) {
	ticker := time.NewTicker(54 * time.Second)
	interval := client.Filters["interval"].(int)
	metricsTicker := time.NewTicker(time.Duration(interval) * time.Second)
	
	defer func() {
		ticker.Stop()
		metricsTicker.Stop()
		client.Connection.Close()
	}()

	for {
		select {
		case <-client.ctx.Done():
			return
			
		case <-metricsTicker.C:
			// Collect and send metrics
			sources, _ := client.Filters["sources"].([]string)
			metrics := h.collectMetrics(sources)
			
			msg := MetricsMessage{
				Type:      "metrics_update",
				Source:    "system",
				Metrics:   metrics,
				Timestamp: time.Now(),
			}
			
			data, _ := json.Marshal(msg)
			select {
			case client.send <- data:
				messagesSent.WithLabelValues("metrics", "update").Inc()
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
					client.Filters = controlMsg.Filters
				}
			}

			messagesReceived.WithLabelValues("alerts", "control").Inc()
			client.LastActivity = time.Now()
		}
	}
}

func (h *WebSocketHandler) alertsWritePump(client *WebSocketClient) {
	ticker := time.NewTicker(54 * time.Second)
	defer func() {
		ticker.Stop()
		client.Connection.Close()
	}()

	for {
		select {
		case <-client.ctx.Done():
			return
			
		case alert := <-h.alertsBroadcast:
			// Apply filters
			if h.matchesAlertFilters(alert, client.Filters) {
				data, _ := json.Marshal(alert)
				select {
				case client.send <- data:
					messagesSent.WithLabelValues("alerts", "notification").Inc()
				default:
					close(client.send)
					return
				}
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
					client.Filters = controlMsg.Filters
				}
			}

			messagesReceived.WithLabelValues("logs", "control").Inc()
			client.LastActivity = time.Now()
		}
	}
}

func (h *WebSocketHandler) logsWritePump(client *WebSocketClient, source string) {
	ticker := time.NewTicker(54 * time.Second)
	defer func() {
		ticker.Stop()
		client.Connection.Close()
	}()

	for {
		select {
		case <-client.ctx.Done():
			return
			
		case logMsg := <-h.logsBroadcast:
			// Apply filters
			if h.matchesLogFilters(logMsg, source, client.Filters) {
				data, _ := json.Marshal(logMsg)
				select {
				case client.send <- data:
					messagesSent.WithLabelValues("logs", "entry").Inc()
				default:
					close(client.send)
					return
				}
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

func (h *WebSocketHandler) broadcastWorker() {
	for {
		select {
		case <-h.ctx.Done():
			return
		case metric := <-h.metricsBroadcast:
			h.broadcastToMetricsClients(metric)
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
	// Extract from JWT token or session
	return "user-id" // placeholder
}

func (h *WebSocketHandler) getUserRolesFromRequest(r *http.Request) []string {
	// Extract from JWT token or session
	return []string{"viewer"} // placeholder
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

func (h *WebSocketHandler) collectMetrics(sources []string) map[string]interface{} {
	result := make(map[string]interface{})

	// If no specific sources requested, collect common metrics
	if len(sources) == 0 {
		sources = []string{"cpu_usage", "memory_usage", "disk_usage", "network_io", "bandwidth", "qos"}
	}

	// Collect metrics from metric registry
	if h.metricRegistry != nil {
		for _, source := range sources {
			series, err := h.metricRegistry.GetMetrics(source)
			if err == nil && len(series) > 0 {
				// Get the latest value from the first series
				if len(series[0].Metrics) > 0 {
					latestMetric := series[0].Metrics[len(series[0].Metrics)-1]
					result[source] = latestMetric.Value
				}
			}
		}
	}

	// Add current timestamp
	result["timestamp"] = time.Now().UnixMilli()

	// If no real metrics found, return some system metrics
	if len(result) <= 1 {
		// Fall back to basic system info if registry is empty
		result["cpu_usage"] = 0.0
		result["memory_usage"] = 0.0
		result["disk_usage"] = 0.0
		result["network_io"] = 0.0
		result["status"] = "no_data"
	}

	return result
}

func (h *WebSocketHandler) matchesAlertFilters(alert AlertMessage, filters map[string]interface{}) bool {
	severities, ok := filters["severities"].([]string)
	if ok && len(severities) > 0 {
		found := false
		for _, sev := range severities {
			if sev == alert.Severity {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}

	sources, ok := filters["sources"].([]string)
	if ok && len(sources) > 0 {
		found := false
		for _, src := range sources {
			if src == alert.Source {
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

func (h *WebSocketHandler) broadcastToMetricsClients(metric MetricsMessage) {
	// Implementation would broadcast to all metrics clients
}

func (h *WebSocketHandler) broadcastToAlertClients(alert AlertMessage) {
	// Implementation would broadcast to all alert clients
}

func (h *WebSocketHandler) broadcastToLogClients(logMsg LogMessage) {
	// Implementation would broadcast to all log clients
}