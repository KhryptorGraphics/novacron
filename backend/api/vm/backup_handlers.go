package vm

import (
	"context"
	"encoding/json"
	"net/http"
	"strconv"
	"time"

	"github.com/gorilla/mux"
	"github.com/khryptorgraphics/novacron/backend/core/vm"
)

// BackupHandler handles VM backup API requests
type BackupHandler struct {
	backupManager *vm.VMBackupManager
}

// NewBackupHandler creates a new VM backup API handler
func NewBackupHandler(backupManager *vm.VMBackupManager) *BackupHandler {
	return &BackupHandler{
		backupManager: backupManager,
	}
}

// RegisterRoutes registers VM backup API routes
func (h *BackupHandler) RegisterRoutes(router *mux.Router) {
	router.HandleFunc("/vms/{vm_id}/backups", h.ListBackups).Methods("GET")
	router.HandleFunc("/vms/{vm_id}/backups", h.CreateBackup).Methods("POST")
	router.HandleFunc("/vms/{vm_id}/backups/{id}", h.GetBackup).Methods("GET")
	router.HandleFunc("/vms/{vm_id}/backups/{id}", h.DeleteBackup).Methods("DELETE")
	router.HandleFunc("/vms/{vm_id}/backups/{id}/restore", h.RestoreBackup).Methods("POST")
}

// ListBackups handles GET /vms/{vm_id}/backups
func (h *BackupHandler) ListBackups(w http.ResponseWriter, r *http.Request) {
	// Get VM ID from URL
	vars := mux.Vars(r)
	vmID := vars["vm_id"]
	
	// Get backups
	backups := h.backupManager.ListBackupsForVM(vmID)
	
	// Convert to response format
	response := make([]map[string]interface{}, 0, len(backups))
	for _, backup := range backups {
		response = append(response, map[string]interface{}{
			"id":          backup.ID,
			"vm_id":       backup.VMID,
			"name":        backup.Name,
			"description": backup.Description,
			"type":        backup.Type,
			"status":      backup.Status,
			"created_at":  backup.CreatedAt,
			"updated_at":  backup.UpdatedAt,
			"size":        backup.Size,
			"parent_id":   backup.ParentID,
			"tags":        backup.Tags,
			"metadata":    backup.Metadata,
			"expires_at":  backup.ExpiresAt,
		})
	}
	
	// Write response
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// CreateBackup handles POST /vms/{vm_id}/backups
func (h *BackupHandler) CreateBackup(w http.ResponseWriter, r *http.Request) {
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
		ExpiresIn   int               `json:"expires_in_days"`
	}
	
	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	
	// Validate backup type
	var backupType vm.BackupType
	switch request.Type {
	case "full":
		backupType = vm.BackupTypeFull
	case "incremental":
		backupType = vm.BackupTypeIncremental
	case "differential":
		backupType = vm.BackupTypeDifferential
	default:
		backupType = vm.BackupTypeFull
	}
	
	// Set expiration time if provided
	var expiresIn *time.Duration
	if request.ExpiresIn > 0 {
		duration := time.Duration(request.ExpiresIn) * 24 * time.Hour
		expiresIn = &duration
	}
	
	// Create backup
	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()
	
	backup, err := h.backupManager.CreateBackup(ctx, vmID, request.Name, request.Description, backupType, request.Tags, request.Metadata, expiresIn)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	
	// Write response
	response := map[string]interface{}{
		"id":          backup.ID,
		"vm_id":       backup.VMID,
		"name":        backup.Name,
		"description": backup.Description,
		"type":        backup.Type,
		"status":      backup.Status,
		"created_at":  backup.CreatedAt,
		"updated_at":  backup.UpdatedAt,
		"tags":        backup.Tags,
		"metadata":    backup.Metadata,
		"expires_at":  backup.ExpiresAt,
	}
	
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(response)
}

// GetBackup handles GET /vms/{vm_id}/backups/{id}
func (h *BackupHandler) GetBackup(w http.ResponseWriter, r *http.Request) {
	// Get backup ID from URL
	vars := mux.Vars(r)
	backupID := vars["id"]
	
	// Get backup
	backup, err := h.backupManager.GetBackup(backupID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	
	// Write response
	response := map[string]interface{}{
		"id":          backup.ID,
		"vm_id":       backup.VMID,
		"name":        backup.Name,
		"description": backup.Description,
		"type":        backup.Type,
		"status":      backup.Status,
		"created_at":  backup.CreatedAt,
		"updated_at":  backup.UpdatedAt,
		"size":        backup.Size,
		"parent_id":   backup.ParentID,
		"tags":        backup.Tags,
		"metadata":    backup.Metadata,
		"expires_at":  backup.ExpiresAt,
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// DeleteBackup handles DELETE /vms/{vm_id}/backups/{id}
func (h *BackupHandler) DeleteBackup(w http.ResponseWriter, r *http.Request) {
	// Get backup ID from URL
	vars := mux.Vars(r)
	backupID := vars["id"]
	
	// Delete backup
	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()
	
	if err := h.backupManager.DeleteBackup(ctx, backupID); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	
	// Write response
	w.WriteHeader(http.StatusNoContent)
}

// RestoreBackup handles POST /vms/{vm_id}/backups/{id}/restore
func (h *BackupHandler) RestoreBackup(w http.ResponseWriter, r *http.Request) {
	// Get backup ID from URL
	vars := mux.Vars(r)
	backupID := vars["id"]
	
	// Restore backup
	ctx, cancel := context.WithTimeout(r.Context(), 60*time.Second)
	defer cancel()
	
	if err := h.backupManager.RestoreBackup(ctx, backupID); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	
	// Write response
	w.WriteHeader(http.StatusNoContent)
}
