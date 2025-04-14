package vm

import (
	"context"
	"encoding/json"
	"net/http"
	"time"

	"github.com/gorilla/mux"
	"github.com/khryptorgraphics/novacron/backend/core/vm"
)

// MigrationHandler handles VM migration API requests
type MigrationHandler struct {
	vmManager *vm.VMManager
}

// NewMigrationHandler creates a new VM migration API handler
func NewMigrationHandler(vmManager *vm.VMManager) *MigrationHandler {
	return &MigrationHandler{
		vmManager: vmManager,
	}
}

// RegisterRoutes registers VM migration API routes
func (h *MigrationHandler) RegisterRoutes(router *mux.Router) {
	router.HandleFunc("/migrations", h.ListMigrations).Methods("GET")
	router.HandleFunc("/migrations", h.CreateMigration).Methods("POST")
	router.HandleFunc("/migrations/{id}", h.GetMigration).Methods("GET")
	router.HandleFunc("/migrations/{id}/cancel", h.CancelMigration).Methods("POST")
}

// ListMigrations handles GET /migrations
func (h *MigrationHandler) ListMigrations(w http.ResponseWriter, r *http.Request) {
	// Get migrations
	migrations := h.vmManager.ListMigrations()
	
	// Convert to response format
	response := make([]map[string]interface{}, 0, len(migrations))
	for _, migration := range migrations {
		response = append(response, map[string]interface{}{
			"id":             migration.ID,
			"vm_id":          migration.VMID,
			"source_node_id": migration.SourceNodeID,
			"dest_node_id":   migration.DestNodeID,
			"state":          migration.State,
			"type":           migration.Type,
			"created_at":     migration.CreatedAt,
			"updated_at":     migration.UpdatedAt,
			"completed_at":   migration.CompletedAt,
			"error":          migration.Error,
		})
	}
	
	// Write response
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// CreateMigration handles POST /migrations
func (h *MigrationHandler) CreateMigration(w http.ResponseWriter, r *http.Request) {
	// Parse request
	var request struct {
		VMID       string `json:"vm_id"`
		DestNodeID string `json:"dest_node_id"`
		Type       string `json:"type"`
	}
	
	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	
	// Validate migration type
	var migrationType vm.MigrationType
	switch request.Type {
	case "offline":
		migrationType = vm.MigrationTypeOffline
	case "suspend":
		migrationType = vm.MigrationTypeSuspend
	case "live":
		migrationType = vm.MigrationTypeLive
	default:
		http.Error(w, "Invalid migration type. Must be 'offline', 'suspend', or 'live'.", http.StatusBadRequest)
		return
	}
	
	// Create migration
	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()
	
	migration, err := h.vmManager.MigrateVM(ctx, request.VMID, request.DestNodeID, migrationType)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	
	// Write response
	response := map[string]interface{}{
		"id":             migration.ID,
		"vm_id":          migration.VMID,
		"source_node_id": migration.SourceNodeID,
		"dest_node_id":   migration.DestNodeID,
		"state":          migration.State,
		"type":           migration.Type,
		"created_at":     migration.CreatedAt,
		"updated_at":     migration.UpdatedAt,
	}
	
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(response)
}

// GetMigration handles GET /migrations/{id}
func (h *MigrationHandler) GetMigration(w http.ResponseWriter, r *http.Request) {
	// Get migration ID from URL
	vars := mux.Vars(r)
	migrationID := vars["id"]
	
	// Get migration
	migration, err := h.vmManager.GetMigration(migrationID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	
	// Write response
	response := map[string]interface{}{
		"id":             migration.ID,
		"vm_id":          migration.VMID,
		"source_node_id": migration.SourceNodeID,
		"dest_node_id":   migration.DestNodeID,
		"state":          migration.State,
		"type":           migration.Type,
		"created_at":     migration.CreatedAt,
		"updated_at":     migration.UpdatedAt,
		"completed_at":   migration.CompletedAt,
		"error":          migration.Error,
		"progress":       migration.Progress,
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// CancelMigration handles POST /migrations/{id}/cancel
func (h *MigrationHandler) CancelMigration(w http.ResponseWriter, r *http.Request) {
	// Get migration ID from URL
	vars := mux.Vars(r)
	migrationID := vars["id"]
	
	// Cancel migration
	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()
	
	if err := h.vmManager.CancelMigration(ctx, migrationID); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	
	// Get updated migration
	migration, err := h.vmManager.GetMigration(migrationID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	
	// Write response
	response := map[string]interface{}{
		"id":             migration.ID,
		"vm_id":          migration.VMID,
		"source_node_id": migration.SourceNodeID,
		"dest_node_id":   migration.DestNodeID,
		"state":          migration.State,
		"type":           migration.Type,
		"created_at":     migration.CreatedAt,
		"updated_at":     migration.UpdatedAt,
		"completed_at":   migration.CompletedAt,
		"error":          migration.Error,
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}
