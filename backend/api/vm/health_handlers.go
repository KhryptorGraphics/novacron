package vm

import (
	"context"
	"encoding/json"
	"net/http"
	"time"

	"github.com/gorilla/mux"
	"github.com/khryptorgraphics/novacron/backend/core/vm"
)

// HealthHandler handles VM health API requests
type HealthHandler struct {
	healthManager *vm.VMHealthManager
}

// NewHealthHandler creates a new VM health API handler
func NewHealthHandler(healthManager *vm.VMHealthManager) *HealthHandler {
	return &HealthHandler{
		healthManager: healthManager,
	}
}

// RegisterRoutes registers VM health API routes
func (h *HealthHandler) RegisterRoutes(router *mux.Router) {
	router.HandleFunc("/vms/{vm_id}/health", h.GetVMHealth).Methods("GET")
	router.HandleFunc("/vms/health", h.GetAllVMHealth).Methods("GET")
	router.HandleFunc("/vms/{vm_id}/health/check", h.CheckVMHealth).Methods("POST")
}

// GetVMHealth handles GET /vms/{vm_id}/health
func (h *HealthHandler) GetVMHealth(w http.ResponseWriter, r *http.Request) {
	// Get VM ID from URL
	vars := mux.Vars(r)
	vmID := vars["vm_id"]
	
	// Get VM health
	health, err := h.healthManager.GetVMHealth(vmID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	
	// Write response
	response := map[string]interface{}{
		"vm_id":        health.VMID,
		"status":       health.Status,
		"last_checked": health.LastChecked,
		"checks":       health.Checks,
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// GetAllVMHealth handles GET /vms/health
func (h *HealthHandler) GetAllVMHealth(w http.ResponseWriter, r *http.Request) {
	// Get all VM health
	healthMap := h.healthManager.GetAllVMHealth()
	
	// Convert to response format
	response := make(map[string]map[string]interface{})
	for vmID, health := range healthMap {
		response[vmID] = map[string]interface{}{
			"vm_id":        health.VMID,
			"status":       health.Status,
			"last_checked": health.LastChecked,
			"checks":       health.Checks,
		}
	}
	
	// Write response
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// CheckVMHealth handles POST /vms/{vm_id}/health/check
func (h *HealthHandler) CheckVMHealth(w http.ResponseWriter, r *http.Request) {
	// Get VM ID from URL
	vars := mux.Vars(r)
	vmID := vars["vm_id"]
	
	// Check VM health
	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()
	
	health, err := h.healthManager.CheckVMHealth(ctx, vmID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	
	// Write response
	response := map[string]interface{}{
		"vm_id":        health.VMID,
		"status":       health.Status,
		"last_checked": health.LastChecked,
		"checks":       health.Checks,
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}
