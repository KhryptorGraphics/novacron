package vm

import (
	"context"
	"encoding/json"
	"net/http"
	"time"

	"github.com/gorilla/mux"
	"github.com/khryptorgraphics/novacron/backend/core/vm"
)

// SnapshotHandler handles VM snapshot API requests
type SnapshotHandler struct {
	snapshotManager *vm.VMSnapshotManager
}

// NewSnapshotHandler creates a new VM snapshot API handler
func NewSnapshotHandler(snapshotManager *vm.VMSnapshotManager) *SnapshotHandler {
	return &SnapshotHandler{
		snapshotManager: snapshotManager,
	}
}

// RegisterRoutes registers VM snapshot API routes
func (h *SnapshotHandler) RegisterRoutes(router *mux.Router) {
	router.HandleFunc("/vms/{vm_id}/snapshots", h.ListSnapshots).Methods("GET")
	router.HandleFunc("/vms/{vm_id}/snapshots", h.CreateSnapshot).Methods("POST")
	router.HandleFunc("/vms/{vm_id}/snapshots/{id}", h.GetSnapshot).Methods("GET")
	router.HandleFunc("/vms/{vm_id}/snapshots/{id}", h.DeleteSnapshot).Methods("DELETE")
	router.HandleFunc("/vms/{vm_id}/snapshots/{id}/restore", h.RestoreSnapshot).Methods("POST")
}

// ListSnapshots handles GET /vms/{vm_id}/snapshots
func (h *SnapshotHandler) ListSnapshots(w http.ResponseWriter, r *http.Request) {
	// Get VM ID from URL
	vars := mux.Vars(r)
	vmID := vars["vm_id"]
	
	// Get snapshots
	snapshots := h.snapshotManager.ListSnapshotsForVM(vmID)
	
	// Convert to response format
	response := make([]map[string]interface{}, 0, len(snapshots))
	for _, snapshot := range snapshots {
		response = append(response, map[string]interface{}{
			"id":          snapshot.ID,
			"vm_id":       snapshot.VMID,
			"name":        snapshot.Name,
			"description": snapshot.Description,
			"type":        snapshot.Type,
			"status":      snapshot.Status,
			"created_at":  snapshot.CreatedAt,
			"updated_at":  snapshot.UpdatedAt,
			"size":        snapshot.Size,
			"parent_id":   snapshot.ParentID,
			"tags":        snapshot.Tags,
			"metadata":    snapshot.Metadata,
		})
	}
	
	// Write response
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// CreateSnapshot handles POST /vms/{vm_id}/snapshots
func (h *SnapshotHandler) CreateSnapshot(w http.ResponseWriter, r *http.Request) {
	// Get VM ID from URL
	vars := mux.Vars(r)
	vmID := vars["vm_id"]
	
	// Parse request
	var request struct {
		Name        string            `json:"name"`
		Description string            `json:"description"`
		Type        string            `json:"type"`
		Tags        []string          `json:"tags"`
		Metadata    map[string]string `json:"metadata"`
	}
	
	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	
	// Validate snapshot type
	var snapshotType vm.SnapshotType
	switch request.Type {
	case "full":
		snapshotType = vm.SnapshotTypeFull
	case "incremental":
		snapshotType = vm.SnapshotTypeIncremental
	case "differential":
		snapshotType = vm.SnapshotTypeDifferential
	default:
		snapshotType = vm.SnapshotTypeFull
	}
	
	// Create snapshot
	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()
	
	snapshot, err := h.snapshotManager.CreateSnapshot(ctx, vmID, request.Name, request.Description, snapshotType, request.Tags, request.Metadata)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	
	// Write response
	response := map[string]interface{}{
		"id":          snapshot.ID,
		"vm_id":       snapshot.VMID,
		"name":        snapshot.Name,
		"description": snapshot.Description,
		"type":        snapshot.Type,
		"status":      snapshot.Status,
		"created_at":  snapshot.CreatedAt,
		"updated_at":  snapshot.UpdatedAt,
		"tags":        snapshot.Tags,
		"metadata":    snapshot.Metadata,
	}
	
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(response)
}

// GetSnapshot handles GET /vms/{vm_id}/snapshots/{id}
func (h *SnapshotHandler) GetSnapshot(w http.ResponseWriter, r *http.Request) {
	// Get snapshot ID from URL
	vars := mux.Vars(r)
	snapshotID := vars["id"]
	
	// Get snapshot
	snapshot, err := h.snapshotManager.GetSnapshot(snapshotID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	
	// Write response
	response := map[string]interface{}{
		"id":          snapshot.ID,
		"vm_id":       snapshot.VMID,
		"name":        snapshot.Name,
		"description": snapshot.Description,
		"type":        snapshot.Type,
		"status":      snapshot.Status,
		"created_at":  snapshot.CreatedAt,
		"updated_at":  snapshot.UpdatedAt,
		"size":        snapshot.Size,
		"parent_id":   snapshot.ParentID,
		"tags":        snapshot.Tags,
		"metadata":    snapshot.Metadata,
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// DeleteSnapshot handles DELETE /vms/{vm_id}/snapshots/{id}
func (h *SnapshotHandler) DeleteSnapshot(w http.ResponseWriter, r *http.Request) {
	// Get snapshot ID from URL
	vars := mux.Vars(r)
	snapshotID := vars["id"]
	
	// Delete snapshot
	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()
	
	if err := h.snapshotManager.DeleteSnapshot(ctx, snapshotID); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	
	// Write response
	w.WriteHeader(http.StatusNoContent)
}

// RestoreSnapshot handles POST /vms/{vm_id}/snapshots/{id}/restore
func (h *SnapshotHandler) RestoreSnapshot(w http.ResponseWriter, r *http.Request) {
	// Get snapshot ID from URL
	vars := mux.Vars(r)
	snapshotID := vars["id"]
	
	// Restore snapshot
	ctx, cancel := context.WithTimeout(r.Context(), 60*time.Second)
	defer cancel()
	
	if err := h.snapshotManager.RestoreSnapshot(ctx, snapshotID); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	
	// Write response
	w.WriteHeader(http.StatusNoContent)
}
