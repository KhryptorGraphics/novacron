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

// MetricsHandler handles VM metrics API requests
type MetricsHandler struct {
	metricsCollector *vm.VMMetricsCollector
}

// NewMetricsHandler creates a new VM metrics API handler
func NewMetricsHandler(metricsCollector *vm.VMMetricsCollector) *MetricsHandler {
	return &MetricsHandler{
		metricsCollector: metricsCollector,
	}
}

// RegisterRoutes registers VM metrics API routes
func (h *MetricsHandler) RegisterRoutes(router *mux.Router) {
	router.HandleFunc("/vms/{vm_id}/metrics", h.GetVMMetrics).Methods("GET")
	router.HandleFunc("/vms/{vm_id}/metrics/{metric_name}", h.GetVMMetric).Methods("GET")
	router.HandleFunc("/vms/{vm_id}/metrics/export", h.ExportVMMetrics).Methods("POST")
}

// GetVMMetrics handles GET /vms/{vm_id}/metrics
func (h *MetricsHandler) GetVMMetrics(w http.ResponseWriter, r *http.Request) {
	// Get VM ID from URL
	vars := mux.Vars(r)
	vmID := vars["vm_id"]

	// Get metrics
	metrics := h.metricsCollector.GetMetrics(vmID)

	// Convert to response format
	response := make([]map[string]interface{}, 0, len(metrics))
	for _, metric := range metrics {
		response = append(response, map[string]interface{}{
			"vm_id": metric.VMID,
			"type":  metric.Type,
			"name":  metric.Name,
			"unit":  metric.Unit,
			"values": metric.Values,
		})
	}

	// Write response
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// GetVMMetric handles GET /vms/{vm_id}/metrics/{metric_name}
func (h *MetricsHandler) GetVMMetric(w http.ResponseWriter, r *http.Request) {
	// Get VM ID and metric name from URL
	vars := mux.Vars(r)
	vmID := vars["vm_id"]
	metricName := vars["metric_name"]

	// Get metric
	metric, err := h.metricsCollector.GetMetric(vmID, metricName)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}

	// Parse query parameters
	limit := 0
	if limitStr := r.URL.Query().Get("limit"); limitStr != "" {
		limit, _ = strconv.Atoi(limitStr)
	}

	// Limit values if requested
	values := metric.Values
	if limit > 0 && limit < len(values) {
		values = values[len(values)-limit:]
	}

	// Write response
	response := map[string]interface{}{
		"vm_id":  metric.VMID,
		"type":   metric.Type,
		"name":   metric.Name,
		"unit":   metric.Unit,
		"values": values,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// ExportVMMetrics handles POST /vms/{vm_id}/metrics/export
func (h *MetricsHandler) ExportVMMetrics(w http.ResponseWriter, r *http.Request) {
	// Get VM ID from URL
	vars := mux.Vars(r)
	vmID := vars["vm_id"]

	// Export metrics
	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()

	if err := h.metricsCollector.ExportMetrics(ctx); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Write response
	response := map[string]interface{}{
		"vm_id":    vmID,
		"exported": true,
		"timestamp": time.Now(),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}
