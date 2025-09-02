package federation

import (
	"encoding/json"
	"net/http"
	"time"

	"github.com/gorilla/mux"
	"github.com/gorilla/websocket"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	
	"github.com/khryptorgraphics/novacron/backend/core/federation"
	"github.com/khryptorgraphics/novacron/backend/pkg/middleware"
)

var (
	apiRequestDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "novacron_federation_api_duration_seconds",
		Help:    "Duration of federation API requests",
		Buckets: prometheus.DefBuckets,
	}, []string{"method", "endpoint"})

	apiRequestCount = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "novacron_federation_api_requests_total",
		Help: "Total number of federation API requests",
	}, []string{"method", "endpoint", "status"})

	wsConnections = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "novacron_federation_websocket_connections",
		Help: "Current number of WebSocket connections",
	})
)

// Handler handles federation API requests
type Handler struct {
	manager   federation.FederationManager
	upgrader  websocket.Upgrader
	wsClients map[*websocket.Conn]bool
	broadcast chan interface{}
}

// NewHandler creates a new federation API handler
func NewHandler(manager federation.FederationManager) *Handler {
	return &Handler{
		manager: manager,
		upgrader: websocket.Upgrader{
			CheckOrigin: func(r *http.Request) bool {
				// In production, implement proper origin checking
				return true
			},
		},
		wsClients: make(map[*websocket.Conn]bool),
		broadcast: make(chan interface{}, 100),
	}
}

// RegisterRoutes registers federation API routes
func (h *Handler) RegisterRoutes(router *mux.Router) {
	// Apply authentication middleware
	api := router.PathPrefix("/api/v1/federation").Subrouter()
	api.Use(middleware.Authenticate)
	api.Use(h.metricsMiddleware)

	// Federation management
	api.HandleFunc("/status", h.GetStatus).Methods("GET")
	api.HandleFunc("/join", h.JoinFederation).Methods("POST")
	api.HandleFunc("/leave", h.LeaveFederation).Methods("POST")

	// Node management
	api.HandleFunc("/nodes", h.ListNodes).Methods("GET")
	api.HandleFunc("/nodes/{nodeID}", h.GetNode).Methods("GET")
	api.HandleFunc("/nodes/{nodeID}/health", h.GetNodeHealth).Methods("GET")

	// Resource management
	api.HandleFunc("/resources", h.GetResources).Methods("GET")
	api.HandleFunc("/resources/request", h.RequestResources).Methods("POST")
	api.HandleFunc("/resources/{allocationID}", h.ReleaseResources).Methods("DELETE")
	api.HandleFunc("/allocations", h.ListAllocations).Methods("GET")

	// Monitoring
	api.HandleFunc("/metrics", h.GetMetrics).Methods("GET")
	api.HandleFunc("/health", h.GetHealth).Methods("GET")

	// WebSocket for real-time updates
	api.HandleFunc("/ws", h.WebSocketHandler)

	// Inter-cluster communication
	api.HandleFunc("/cluster/sync", h.ClusterSync).Methods("POST")
	api.HandleFunc("/cluster/replicate", h.ClusterReplicate).Methods("POST")
}

// GetStatus returns the federation status
func (h *Handler) GetStatus(w http.ResponseWriter, r *http.Request) {
	timer := prometheus.NewTimer(apiRequestDuration.WithLabelValues("GET", "/status"))
	defer timer.ObserveDuration()

	ctx := r.Context()

	// Get nodes
	nodes, err := h.manager.GetNodes(ctx)
	if err != nil {
		h.sendError(w, "Failed to get nodes", http.StatusInternalServerError)
		apiRequestCount.WithLabelValues("GET", "/status", "error").Inc()
		return
	}

	// Get leader
	leader, _ := h.manager.GetLeader(ctx)

	// Build status response
	status := map[string]interface{}{
		"is_leader":    h.manager.IsLeader(),
		"node_count":   len(nodes),
		"leader":       leader,
		"nodes":        nodes,
		"timestamp":    time.Now(),
	}

	h.sendJSON(w, status, http.StatusOK)
	apiRequestCount.WithLabelValues("GET", "/status", "success").Inc()
}

// JoinFederation handles federation join requests
func (h *Handler) JoinFederation(w http.ResponseWriter, r *http.Request) {
	timer := prometheus.NewTimer(apiRequestDuration.WithLabelValues("POST", "/join"))
	defer timer.ObserveDuration()

	var req struct {
		JoinAddresses []string `json:"join_addresses"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		h.sendError(w, "Invalid request body", http.StatusBadRequest)
		apiRequestCount.WithLabelValues("POST", "/join", "error").Inc()
		return
	}

	ctx := r.Context()
	if err := h.manager.JoinFederation(ctx, req.JoinAddresses); err != nil {
		h.sendError(w, err.Error(), http.StatusInternalServerError)
		apiRequestCount.WithLabelValues("POST", "/join", "error").Inc()
		return
	}

	h.sendJSON(w, map[string]string{"status": "joined"}, http.StatusOK)
	apiRequestCount.WithLabelValues("POST", "/join", "success").Inc()

	// Broadcast join event
	h.broadcast <- map[string]interface{}{
		"event": "node_joined",
		"timestamp": time.Now(),
	}
}

// LeaveFederation handles federation leave requests
func (h *Handler) LeaveFederation(w http.ResponseWriter, r *http.Request) {
	timer := prometheus.NewTimer(apiRequestDuration.WithLabelValues("POST", "/leave"))
	defer timer.ObserveDuration()

	ctx := r.Context()
	if err := h.manager.LeaveFederation(ctx); err != nil {
		h.sendError(w, err.Error(), http.StatusInternalServerError)
		apiRequestCount.WithLabelValues("POST", "/leave", "error").Inc()
		return
	}

	h.sendJSON(w, map[string]string{"status": "left"}, http.StatusOK)
	apiRequestCount.WithLabelValues("POST", "/leave", "success").Inc()

	// Broadcast leave event
	h.broadcast <- map[string]interface{}{
		"event": "node_left",
		"timestamp": time.Now(),
	}
}

// ListNodes returns all nodes in the federation
func (h *Handler) ListNodes(w http.ResponseWriter, r *http.Request) {
	timer := prometheus.NewTimer(apiRequestDuration.WithLabelValues("GET", "/nodes"))
	defer timer.ObserveDuration()

	ctx := r.Context()
	nodes, err := h.manager.GetNodes(ctx)
	if err != nil {
		h.sendError(w, "Failed to get nodes", http.StatusInternalServerError)
		apiRequestCount.WithLabelValues("GET", "/nodes", "error").Inc()
		return
	}

	h.sendJSON(w, nodes, http.StatusOK)
	apiRequestCount.WithLabelValues("GET", "/nodes", "success").Inc()
}

// GetNode returns information about a specific node
func (h *Handler) GetNode(w http.ResponseWriter, r *http.Request) {
	timer := prometheus.NewTimer(apiRequestDuration.WithLabelValues("GET", "/nodes/{nodeID}"))
	defer timer.ObserveDuration()

	vars := mux.Vars(r)
	nodeID := vars["nodeID"]

	ctx := r.Context()
	node, err := h.manager.GetNode(ctx, nodeID)
	if err != nil {
		h.sendError(w, "Node not found", http.StatusNotFound)
		apiRequestCount.WithLabelValues("GET", "/nodes/{nodeID}", "error").Inc()
		return
	}

	h.sendJSON(w, node, http.StatusOK)
	apiRequestCount.WithLabelValues("GET", "/nodes/{nodeID}", "success").Inc()
}

// GetNodeHealth returns health information for a node
func (h *Handler) GetNodeHealth(w http.ResponseWriter, r *http.Request) {
	timer := prometheus.NewTimer(apiRequestDuration.WithLabelValues("GET", "/nodes/{nodeID}/health"))
	defer timer.ObserveDuration()

	vars := mux.Vars(r)
	nodeID := vars["nodeID"]

	ctx := r.Context()
	node, err := h.manager.GetNode(ctx, nodeID)
	if err != nil {
		h.sendError(w, "Node not found", http.StatusNotFound)
		apiRequestCount.WithLabelValues("GET", "/nodes/{nodeID}/health", "error").Inc()
		return
	}

	// Get health status
	health, err := h.manager.GetHealth(ctx)
	if err != nil {
		h.sendError(w, "Failed to get health status", http.StatusInternalServerError)
		apiRequestCount.WithLabelValues("GET", "/nodes/{nodeID}/health", "error").Inc()
		return
	}

	response := map[string]interface{}{
		"node":   node,
		"health": health,
	}

	h.sendJSON(w, response, http.StatusOK)
	apiRequestCount.WithLabelValues("GET", "/nodes/{nodeID}/health", "success").Inc()
}

// GetResources returns available resources in the federation
func (h *Handler) GetResources(w http.ResponseWriter, r *http.Request) {
	timer := prometheus.NewTimer(apiRequestDuration.WithLabelValues("GET", "/resources"))
	defer timer.ObserveDuration()

	ctx := r.Context()

	// Get all nodes
	nodes, err := h.manager.GetNodes(ctx)
	if err != nil {
		h.sendError(w, "Failed to get nodes", http.StatusInternalServerError)
		apiRequestCount.WithLabelValues("GET", "/resources", "error").Inc()
		return
	}

	// Aggregate resources
	totalResources := &federation.ResourceInventory{}
	for _, node := range nodes {
		totalResources.TotalCPU += node.Capabilities.Resources.TotalCPU
		totalResources.UsedCPU += node.Capabilities.Resources.UsedCPU
		totalResources.TotalMemory += node.Capabilities.Resources.TotalMemory
		totalResources.UsedMemory += node.Capabilities.Resources.UsedMemory
		totalResources.TotalStorage += node.Capabilities.Resources.TotalStorage
		totalResources.UsedStorage += node.Capabilities.Resources.UsedStorage
		totalResources.VMs += node.Capabilities.Resources.VMs
		totalResources.Containers += node.Capabilities.Resources.Containers
		totalResources.NetworkPools += node.Capabilities.Resources.NetworkPools
	}

	response := map[string]interface{}{
		"total_resources": totalResources,
		"node_count":      len(nodes),
		"timestamp":       time.Now(),
	}

	h.sendJSON(w, response, http.StatusOK)
	apiRequestCount.WithLabelValues("GET", "/resources", "success").Inc()
}

// RequestResources handles resource allocation requests
func (h *Handler) RequestResources(w http.ResponseWriter, r *http.Request) {
	timer := prometheus.NewTimer(apiRequestDuration.WithLabelValues("POST", "/resources/request"))
	defer timer.ObserveDuration()

	var req federation.ResourceRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		h.sendError(w, "Invalid request body", http.StatusBadRequest)
		apiRequestCount.WithLabelValues("POST", "/resources/request", "error").Inc()
		return
	}

	// Set default duration if not specified
	if req.Duration == 0 {
		req.Duration = 1 * time.Hour
	}

	ctx := r.Context()
	allocation, err := h.manager.RequestResources(ctx, &req)
	if err != nil {
		h.sendError(w, err.Error(), http.StatusInternalServerError)
		apiRequestCount.WithLabelValues("POST", "/resources/request", "error").Inc()
		return
	}

	h.sendJSON(w, allocation, http.StatusOK)
	apiRequestCount.WithLabelValues("POST", "/resources/request", "success").Inc()

	// Broadcast allocation event
	h.broadcast <- map[string]interface{}{
		"event":      "resource_allocated",
		"allocation": allocation,
		"timestamp":  time.Now(),
	}
}

// ReleaseResources handles resource release requests
func (h *Handler) ReleaseResources(w http.ResponseWriter, r *http.Request) {
	timer := prometheus.NewTimer(apiRequestDuration.WithLabelValues("DELETE", "/resources/{allocationID}"))
	defer timer.ObserveDuration()

	vars := mux.Vars(r)
	allocationID := vars["allocationID"]

	ctx := r.Context()
	if err := h.manager.ReleaseResources(ctx, allocationID); err != nil {
		h.sendError(w, err.Error(), http.StatusInternalServerError)
		apiRequestCount.WithLabelValues("DELETE", "/resources/{allocationID}", "error").Inc()
		return
	}

	h.sendJSON(w, map[string]string{"status": "released"}, http.StatusOK)
	apiRequestCount.WithLabelValues("DELETE", "/resources/{allocationID}", "success").Inc()

	// Broadcast release event
	h.broadcast <- map[string]interface{}{
		"event":         "resource_released",
		"allocation_id": allocationID,
		"timestamp":     time.Now(),
	}
}

// ListAllocations returns all resource allocations
func (h *Handler) ListAllocations(w http.ResponseWriter, r *http.Request) {
	timer := prometheus.NewTimer(apiRequestDuration.WithLabelValues("GET", "/allocations"))
	defer timer.ObserveDuration()

	// This would typically call a method on the manager to get allocations
	// For now, return empty list
	allocations := []federation.ResourceAllocation{}

	h.sendJSON(w, allocations, http.StatusOK)
	apiRequestCount.WithLabelValues("GET", "/allocations", "success").Inc()
}

// GetMetrics returns federation metrics
func (h *Handler) GetMetrics(w http.ResponseWriter, r *http.Request) {
	timer := prometheus.NewTimer(apiRequestDuration.WithLabelValues("GET", "/metrics"))
	defer timer.ObserveDuration()

	ctx := r.Context()

	// Get nodes
	nodes, _ := h.manager.GetNodes(ctx)

	// Calculate metrics
	metrics := map[string]interface{}{
		"node_count":       len(nodes),
		"is_leader":        h.manager.IsLeader(),
		"ws_connections":   len(h.wsClients),
		"timestamp":        time.Now(),
	}

	h.sendJSON(w, metrics, http.StatusOK)
	apiRequestCount.WithLabelValues("GET", "/metrics", "success").Inc()
}

// GetHealth returns the health status of the federation
func (h *Handler) GetHealth(w http.ResponseWriter, r *http.Request) {
	timer := prometheus.NewTimer(apiRequestDuration.WithLabelValues("GET", "/health"))
	defer timer.ObserveDuration()

	ctx := r.Context()
	health, err := h.manager.GetHealth(ctx)
	if err != nil {
		h.sendError(w, "Failed to get health", http.StatusInternalServerError)
		apiRequestCount.WithLabelValues("GET", "/health", "error").Inc()
		return
	}

	h.sendJSON(w, health, http.StatusOK)
	apiRequestCount.WithLabelValues("GET", "/health", "success").Inc()
}

// WebSocketHandler handles WebSocket connections for real-time updates
func (h *Handler) WebSocketHandler(w http.ResponseWriter, r *http.Request) {
	conn, err := h.upgrader.Upgrade(w, r, nil)
	if err != nil {
		h.sendError(w, "Failed to upgrade connection", http.StatusBadRequest)
		return
	}
	defer conn.Close()

	// Register client
	h.wsClients[conn] = true
	wsConnections.Inc()
	defer func() {
		delete(h.wsClients, conn)
		wsConnections.Dec()
	}()

	// Send initial status
	ctx := r.Context()
	nodes, _ := h.manager.GetNodes(ctx)
	initialStatus := map[string]interface{}{
		"event":      "connected",
		"node_count": len(nodes),
		"is_leader":  h.manager.IsLeader(),
		"timestamp":  time.Now(),
	}
	conn.WriteJSON(initialStatus)

	// Handle incoming messages
	go h.handleWebSocketMessages(conn)

	// Send broadcasts
	for {
		select {
		case message := <-h.broadcast:
			for client := range h.wsClients {
				if err := client.WriteJSON(message); err != nil {
					client.Close()
					delete(h.wsClients, client)
				}
			}
		case <-r.Context().Done():
			return
		}
	}
}

func (h *Handler) handleWebSocketMessages(conn *websocket.Conn) {
	for {
		var msg map[string]interface{}
		err := conn.ReadJSON(&msg)
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				// Log error
			}
			break
		}

		// Handle different message types
		if msgType, ok := msg["type"].(string); ok {
			switch msgType {
			case "ping":
				conn.WriteJSON(map[string]interface{}{
					"type":      "pong",
					"timestamp": time.Now(),
				})
			case "subscribe":
				// Handle subscription to specific events
			case "unsubscribe":
				// Handle unsubscription
			}
		}
	}
}

// ClusterSync handles cluster synchronization requests
func (h *Handler) ClusterSync(w http.ResponseWriter, r *http.Request) {
	timer := prometheus.NewTimer(apiRequestDuration.WithLabelValues("POST", "/cluster/sync"))
	defer timer.ObserveDuration()

	var req struct {
		NodeID    string                    `json:"node_id"`
		ClusterID string                    `json:"cluster_id"`
		State     map[string]interface{}    `json:"state"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		h.sendError(w, "Invalid request body", http.StatusBadRequest)
		apiRequestCount.WithLabelValues("POST", "/cluster/sync", "error").Inc()
		return
	}

	// Process synchronization
	// In production, would synchronize state with cluster

	h.sendJSON(w, map[string]string{"status": "synchronized"}, http.StatusOK)
	apiRequestCount.WithLabelValues("POST", "/cluster/sync", "success").Inc()
}

// ClusterReplicate handles cluster replication requests
func (h *Handler) ClusterReplicate(w http.ResponseWriter, r *http.Request) {
	timer := prometheus.NewTimer(apiRequestDuration.WithLabelValues("POST", "/cluster/replicate"))
	defer timer.ObserveDuration()

	var req struct {
		SourceNodeID string   `json:"source_node_id"`
		TargetNodeID string   `json:"target_node_id"`
		Resources    []string `json:"resources"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		h.sendError(w, "Invalid request body", http.StatusBadRequest)
		apiRequestCount.WithLabelValues("POST", "/cluster/replicate", "error").Inc()
		return
	}

	// Process replication
	// In production, would replicate resources between nodes

	h.sendJSON(w, map[string]string{"status": "replicated"}, http.StatusOK)
	apiRequestCount.WithLabelValues("POST", "/cluster/replicate", "success").Inc()
}

// Helper methods

func (h *Handler) sendJSON(w http.ResponseWriter, data interface{}, status int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(data)
}

func (h *Handler) sendError(w http.ResponseWriter, message string, status int) {
	h.sendJSON(w, map[string]string{"error": message}, status)
}

func (h *Handler) metricsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		
		// Call the next handler
		next.ServeHTTP(w, r)
		
		// Record request duration
		duration := time.Since(start)
		apiRequestDuration.WithLabelValues(r.Method, r.URL.Path).Observe(duration.Seconds())
	})
}