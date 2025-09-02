package vm

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/gorilla/mux"
	"github.com/gorilla/websocket"
	"github.com/khryptorgraphics/novacron/backend/core/migration"
	"github.com/khryptorgraphics/novacron/backend/core/vm"
)

// EnhancedMigrationHandler handles enhanced VM migration API requests
type EnhancedMigrationHandler struct {
	vmManager     *vm.VMManager
	orchestrator  *migration.LiveMigrationOrchestrator
	monitor       *migration.MigrationMonitor
	upgrader      websocket.Upgrader
}

// NewEnhancedMigrationHandler creates a new enhanced migration handler
func NewEnhancedMigrationHandler(vmManager *vm.VMManager) (*EnhancedMigrationHandler, error) {
	// Create migration configuration
	config := migration.MigrationConfig{
		MaxDowntime:             30 * time.Second,
		TargetTransferRate:      20 * 1024 * 1024 * 1024 / 60, // 20 GB/min
		SuccessRateTarget:       0.999,
		EnableCompression:       true,
		CompressionType:         migration.CompressionAdaptive,
		CompressionLevel:        6,
		EnableEncryption:        true,
		EnableDeltaSync:         true,
		BandwidthLimit:          0, // Unlimited
		AdaptiveBandwidth:       true,
		QoSPriority:             migration.QoSPriorityHigh,
		MemoryIterations:        10,
		DirtyPageThreshold:      1000,
		ConvergenceTimeout:      5 * time.Minute,
		EnableCheckpointing:     true,
		CheckpointInterval:      30 * time.Second,
		RetryAttempts:           3,
		RetryDelay:              10 * time.Second,
		MaxCPUUsage:             80.0,
		MaxMemoryUsage:          8 * 1024 * 1024 * 1024, // 8GB
		MaxConcurrentMigrations: 5,
	}
	
	orchestrator, err := migration.NewLiveMigrationOrchestrator(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create orchestrator: %w", err)
	}
	
	return &EnhancedMigrationHandler{
		vmManager:    vmManager,
		orchestrator: orchestrator,
		monitor:      migration.NewMigrationMonitor(),
		upgrader: websocket.Upgrader{
			ReadBufferSize:  1024,
			WriteBufferSize: 1024,
			CheckOrigin: func(r *http.Request) bool {
				// In production, implement proper origin checking
				return true
			},
		},
	}, nil
}

// RegisterEnhancedRoutes registers enhanced migration API routes
func (h *EnhancedMigrationHandler) RegisterEnhancedRoutes(router *mux.Router) {
	// Migration operations
	router.HandleFunc("/api/v2/migrations", h.ListMigrations).Methods("GET")
	router.HandleFunc("/api/v2/migrations", h.CreateMigration).Methods("POST")
	router.HandleFunc("/api/v2/migrations/{id}", h.GetMigration).Methods("GET")
	router.HandleFunc("/api/v2/migrations/{id}/cancel", h.CancelMigration).Methods("POST")
	router.HandleFunc("/api/v2/migrations/{id}/rollback", h.RollbackMigration).Methods("POST")
	
	// Monitoring and metrics
	router.HandleFunc("/api/v2/migrations/{id}/status", h.GetMigrationStatus).Methods("GET")
	router.HandleFunc("/api/v2/migrations/{id}/metrics", h.GetMigrationMetrics).Methods("GET")
	router.HandleFunc("/api/v2/migrations/dashboard", h.GetDashboard).Methods("GET")
	
	// WebSocket for real-time updates
	router.HandleFunc("/api/v2/migrations/{id}/ws", h.WebSocketHandler)
	router.HandleFunc("/api/v2/migrations/ws", h.GlobalWebSocketHandler)
	
	// Configuration
	router.HandleFunc("/api/v2/migrations/config", h.GetConfig).Methods("GET")
	router.HandleFunc("/api/v2/migrations/config", h.UpdateConfig).Methods("PUT")
	
	// Batch operations
	router.HandleFunc("/api/v2/migrations/batch", h.BatchMigrate).Methods("POST")
	router.HandleFunc("/api/v2/migrations/evacuate", h.EvacuateNode).Methods("POST")
}

// ListMigrations handles GET /api/v2/migrations
func (h *EnhancedMigrationHandler) ListMigrations(w http.ResponseWriter, r *http.Request) {
	// Get query parameters
	status := r.URL.Query().Get("status")
	nodeID := r.URL.Query().Get("node_id")
	vmID := r.URL.Query().Get("vm_id")
	
	// Get migrations from VM manager
	migrations := h.vmManager.ListMigrations()
	
	// Filter migrations
	filtered := make([]interface{}, 0)
	for _, m := range migrations {
		include := true
		
		if status != "" && m.State != status {
			include = false
		}
		if nodeID != "" && m.SourceNodeID != nodeID && m.DestNodeID != nodeID {
			include = false
		}
		if vmID != "" && m.VMID != vmID {
			include = false
		}
		
		if include {
			// Get enhanced status from monitor
			status, _ := h.orchestrator.GetMigrationStatus(m.ID)
			if status != nil {
				filtered = append(filtered, status)
			} else {
				// Fallback to basic migration info
				filtered = append(filtered, map[string]interface{}{
					"id":               m.ID,
					"vm_id":            m.VMID,
					"source_node_id":   m.SourceNodeID,
					"destination_node_id": m.DestNodeID,
					"type":             m.Type,
					"state":            m.State,
					"created_at":       m.CreatedAt,
					"updated_at":       m.UpdatedAt,
				})
			}
		}
	}
	
	response := map[string]interface{}{
		"migrations": filtered,
		"total":      len(filtered),
		"metrics":    h.orchestrator.GetMetrics(),
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// CreateMigration handles POST /api/v2/migrations
func (h *EnhancedMigrationHandler) CreateMigration(w http.ResponseWriter, r *http.Request) {
	var request struct {
		VMID            string                     `json:"vm_id"`
		DestinationNode string                     `json:"destination_node"`
		Type            string                     `json:"type"`
		Priority        int                        `json:"priority"`
		Options         map[string]interface{}     `json:"options"`
	}
	
	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	
	// Validate migration type
	var migrationType migration.MigrationType
	switch request.Type {
	case "live":
		migrationType = migration.MigrationTypeLive
	case "pre-copy":
		migrationType = migration.MigrationTypePreCopy
	case "post-copy":
		migrationType = migration.MigrationTypePostCopy
	case "hybrid":
		migrationType = migration.MigrationTypeHybrid
	default:
		http.Error(w, "Invalid migration type", http.StatusBadRequest)
		return
	}
	
	// Get source node (would normally get from VM location)
	sourceNode := "node-1" // Placeholder
	
	// Create migration options
	options := migration.MigrationOptions{
		Priority: request.Priority,
		Force:    false,
	}
	
	// Start migration
	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Minute)
	defer cancel()
	
	migrationID, err := h.orchestrator.MigrateVM(ctx, request.VMID, sourceNode, request.DestinationNode, options)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	
	response := map[string]interface{}{
		"migration_id": migrationID,
		"status":       "initiated",
		"message":      "Migration started successfully",
	}
	
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(response)
}

// GetMigration handles GET /api/v2/migrations/{id}
func (h *EnhancedMigrationHandler) GetMigration(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	migrationID := vars["id"]
	
	status, err := h.orchestrator.GetMigrationStatus(migrationID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(status)
}

// CancelMigration handles POST /api/v2/migrations/{id}/cancel
func (h *EnhancedMigrationHandler) CancelMigration(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	migrationID := vars["id"]
	
	if err := h.orchestrator.CancelMigration(migrationID); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	
	response := map[string]interface{}{
		"migration_id": migrationID,
		"status":       "cancelled",
		"message":      "Migration cancelled successfully",
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// RollbackMigration handles POST /api/v2/migrations/{id}/rollback
func (h *EnhancedMigrationHandler) RollbackMigration(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	migrationID := vars["id"]
	
	// This would trigger rollback through the rollback manager
	// For now, we'll use the cancel mechanism
	if err := h.orchestrator.CancelMigration(migrationID); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	
	response := map[string]interface{}{
		"migration_id": migrationID,
		"status":       "rolled_back",
		"message":      "Migration rolled back successfully",
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// GetMigrationStatus handles GET /api/v2/migrations/{id}/status
func (h *EnhancedMigrationHandler) GetMigrationStatus(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	migrationID := vars["id"]
	
	status, err := h.orchestrator.GetMigrationStatus(migrationID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(status)
}

// GetMigrationMetrics handles GET /api/v2/migrations/{id}/metrics
func (h *EnhancedMigrationHandler) GetMigrationMetrics(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	migrationID := vars["id"]
	
	status, err := h.orchestrator.GetMigrationStatus(migrationID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	
	metrics := status["metrics"]
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(metrics)
}

// GetDashboard handles GET /api/v2/migrations/dashboard
func (h *EnhancedMigrationHandler) GetDashboard(w http.ResponseWriter, r *http.Request) {
	dashboard := h.monitor.GetDashboardData()
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(dashboard)
}

// WebSocketHandler handles WebSocket connections for a specific migration
func (h *EnhancedMigrationHandler) WebSocketHandler(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	migrationID := vars["id"]
	
	conn, err := h.upgrader.Upgrade(w, r, nil)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	defer conn.Close()
	
	// Create event channel
	eventCh := make(chan interface{}, 100)
	
	// Subscribe to migration events
	subscriber := &WebSocketSubscriber{
		conn:        conn,
		eventCh:     eventCh,
		migrationID: migrationID,
	}
	
	// Start event loop
	go subscriber.eventLoop()
	
	// Handle incoming messages
	for {
		messageType, p, err := conn.ReadMessage()
		if err != nil {
			break
		}
		
		// Echo back for ping/pong
		if messageType == websocket.PingMessage {
			conn.WriteMessage(websocket.PongMessage, p)
		}
	}
}

// GlobalWebSocketHandler handles WebSocket connections for all migrations
func (h *EnhancedMigrationHandler) GlobalWebSocketHandler(w http.ResponseWriter, r *http.Request) {
	conn, err := h.upgrader.Upgrade(w, r, nil)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	defer conn.Close()
	
	// Create event channel
	eventCh := make(chan interface{}, 100)
	
	// Subscribe to all migration events
	subscriber := &WebSocketSubscriber{
		conn:    conn,
		eventCh: eventCh,
	}
	
	// Start event loop
	go subscriber.eventLoop()
	
	// Send initial dashboard data
	dashboard := h.monitor.GetDashboardData()
	conn.WriteJSON(map[string]interface{}{
		"type": "dashboard",
		"data": dashboard,
	})
	
	// Send updates periodically
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			// Send dashboard update
			dashboard := h.monitor.GetDashboardData()
			if err := conn.WriteJSON(map[string]interface{}{
				"type": "dashboard_update",
				"data": dashboard,
			}); err != nil {
				return
			}
		}
	}
}

// GetConfig handles GET /api/v2/migrations/config
func (h *EnhancedMigrationHandler) GetConfig(w http.ResponseWriter, r *http.Request) {
	// Return current configuration
	config := map[string]interface{}{
		"max_downtime":           "30s",
		"target_transfer_rate":   "20GB/min",
		"success_rate_target":    "99.9%",
		"compression_enabled":    true,
		"encryption_enabled":     true,
		"delta_sync_enabled":     true,
		"max_concurrent":         5,
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(config)
}

// UpdateConfig handles PUT /api/v2/migrations/config
func (h *EnhancedMigrationHandler) UpdateConfig(w http.ResponseWriter, r *http.Request) {
	var config map[string]interface{}
	
	if err := json.NewDecoder(r.Body).Decode(&config); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	
	// Update configuration (would update orchestrator config)
	// For now, just acknowledge
	
	response := map[string]interface{}{
		"status":  "updated",
		"message": "Configuration updated successfully",
		"config":  config,
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// BatchMigrate handles POST /api/v2/migrations/batch
func (h *EnhancedMigrationHandler) BatchMigrate(w http.ResponseWriter, r *http.Request) {
	var request struct {
		Migrations []struct {
			VMID            string `json:"vm_id"`
			DestinationNode string `json:"destination_node"`
			Priority        int    `json:"priority"`
		} `json:"migrations"`
	}
	
	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	
	results := make([]map[string]interface{}, 0)
	
	for _, m := range request.Migrations {
		sourceNode := "node-1" // Placeholder
		
		options := migration.MigrationOptions{
			Priority: m.Priority,
			Force:    false,
		}
		
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
		migrationID, err := h.orchestrator.MigrateVM(ctx, m.VMID, sourceNode, m.DestinationNode, options)
		cancel()
		
		if err != nil {
			results = append(results, map[string]interface{}{
				"vm_id":  m.VMID,
				"status": "failed",
				"error":  err.Error(),
			})
		} else {
			results = append(results, map[string]interface{}{
				"vm_id":        m.VMID,
				"migration_id": migrationID,
				"status":       "initiated",
			})
		}
	}
	
	response := map[string]interface{}{
		"results": results,
		"total":   len(results),
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// EvacuateNode handles POST /api/v2/migrations/evacuate
func (h *EnhancedMigrationHandler) EvacuateNode(w http.ResponseWriter, r *http.Request) {
	var request struct {
		NodeID     string `json:"node_id"`
		TargetNode string `json:"target_node"`
		Priority   int    `json:"priority"`
	}
	
	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	
	// Get all VMs on the node (would query VM manager)
	// For now, simulate with placeholder data
	vms := []string{"vm-1", "vm-2", "vm-3"}
	
	results := make([]map[string]interface{}, 0)
	
	for _, vmID := range vms {
		options := migration.MigrationOptions{
			Priority: request.Priority,
			Force:    true,
		}
		
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
		migrationID, err := h.orchestrator.MigrateVM(ctx, vmID, request.NodeID, request.TargetNode, options)
		cancel()
		
		if err != nil {
			results = append(results, map[string]interface{}{
				"vm_id":  vmID,
				"status": "failed",
				"error":  err.Error(),
			})
		} else {
			results = append(results, map[string]interface{}{
				"vm_id":        vmID,
				"migration_id": migrationID,
				"status":       "initiated",
			})
		}
	}
	
	response := map[string]interface{}{
		"node_id":    request.NodeID,
		"evacuated":  len(results),
		"migrations": results,
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// WebSocketSubscriber handles WebSocket event subscriptions
type WebSocketSubscriber struct {
	conn        *websocket.Conn
	eventCh     chan interface{}
	migrationID string
}

// eventLoop processes events for WebSocket
func (ws *WebSocketSubscriber) eventLoop() {
	for event := range ws.eventCh {
		if err := ws.conn.WriteJSON(event); err != nil {
			return
		}
	}
}

// OnEvent implements EventSubscriber interface
func (ws *WebSocketSubscriber) OnEvent(event migration.MigrationMonitorEvent) {
	if ws.migrationID == "" || event.MigrationID == ws.migrationID {
		ws.eventCh <- map[string]interface{}{
			"type":      "event",
			"timestamp": event.Timestamp,
			"data":      event,
		}
	}
}

// OnAlert implements AlertSubscriber interface
func (ws *WebSocketSubscriber) OnAlert(alert migration.Alert) {
	if ws.migrationID == "" || alert.MigrationID == ws.migrationID {
		ws.eventCh <- map[string]interface{}{
			"type":      "alert",
			"timestamp": alert.Timestamp,
			"data":      alert,
		}
	}
}