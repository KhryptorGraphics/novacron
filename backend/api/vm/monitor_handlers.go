//go:build experimental

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

// MonitorHandler handles VM monitoring API requests
type MonitorHandler struct {
	vmMonitor *vm.VMMonitor
}

// NewMonitorHandler creates a new VM monitoring API handler
func NewMonitorHandler(vmMonitor *vm.VMMonitor) *MonitorHandler {
	return &MonitorHandler{
		vmMonitor: vmMonitor,
	}
}

// RegisterRoutes registers VM monitoring API routes
func (h *MonitorHandler) RegisterRoutes(router *mux.Router) {
	router.HandleFunc("/vms/{vm_id}/alerts", h.GetVMAlerts).Methods("GET")
	router.HandleFunc("/vms/{vm_id}/alerts/active", h.GetActiveVMAlerts).Methods("GET")
	router.HandleFunc("/vms/{vm_id}/alerts/{alert_id}/resolve", h.ResolveVMAlert).Methods("POST")
}

// GetVMAlerts handles GET /vms/{vm_id}/alerts
func (h *MonitorHandler) GetVMAlerts(w http.ResponseWriter, r *http.Request) {
	// Get VM ID from URL
	vars := mux.Vars(r)
	vmID := vars["vm_id"]

	// Get alerts
	alerts := h.vmMonitor.GetAlerts(vmID)

	// Convert to response format
	response := make([]map[string]interface{}, 0, len(alerts))
	for _, alert := range alerts {
		alertResponse := map[string]interface{}{
			"id":        alert.ID,
			"vm_id":     alert.VMID,
			"level":     alert.Level,
			"message":   alert.Message,
			"timestamp": alert.Timestamp,
			"resolved":  alert.Resolved,
		}

		if alert.ResolvedAt != nil {
			alertResponse["resolved_at"] = alert.ResolvedAt
		}

		response = append(response, alertResponse)
	}

	// Write response
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// GetActiveVMAlerts handles GET /vms/{vm_id}/alerts/active
func (h *MonitorHandler) GetActiveVMAlerts(w http.ResponseWriter, r *http.Request) {
	// Get VM ID from URL
	vars := mux.Vars(r)
	vmID := vars["vm_id"]

	// Get active alerts
	alerts := h.vmMonitor.GetActiveAlerts(vmID)

	// Convert to response format
	response := make([]map[string]interface{}, 0, len(alerts))
	for _, alert := range alerts {
		response = append(response, map[string]interface{}{
			"id":        alert.ID,
			"vm_id":     alert.VMID,
			"level":     alert.Level,
			"message":   alert.Message,
			"timestamp": alert.Timestamp,
		})
	}

	// Write response
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// ResolveVMAlert handles POST /vms/{vm_id}/alerts/{alert_id}/resolve
func (h *MonitorHandler) ResolveVMAlert(w http.ResponseWriter, r *http.Request) {
	// Get VM ID and alert ID from URL
	vars := mux.Vars(r)
	vmID := vars["vm_id"]
	alertID := vars["alert_id"]

	// Resolve alert
	if err := h.vmMonitor.ResolveAlert(vmID, alertID); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Write response
	w.WriteHeader(http.StatusNoContent)
}
