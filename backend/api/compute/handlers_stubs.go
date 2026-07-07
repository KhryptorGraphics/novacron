package api

import (
	"encoding/json"
	"net/http"
)

// Stub handlers for routes registered in RegisterRoutes that have no
// implementation yet. Each returns 501 Not Implemented so the API surface
// stays honest instead of silently disappearing.

func notImplemented(w http.ResponseWriter, feature string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusNotImplemented)
	_ = json.NewEncoder(w).Encode(map[string]interface{}{
		"error": map[string]string{
			"code":    "not_implemented",
			"message": feature + " is not implemented yet",
		},
	})
}

// GetJobHistory returns execution history for a job.
func (h *ComputeAPIHandler) GetJobHistory(w http.ResponseWriter, r *http.Request) {
	notImplemented(w, "job history")
}

// BatchOperations performs batch job operations.
func (h *ComputeAPIHandler) BatchOperations(w http.ResponseWriter, r *http.Request) {
	notImplemented(w, "batch operations")
}

// ListNodes lists compute nodes.
func (h *ComputeAPIHandler) ListNodes(w http.ResponseWriter, r *http.Request) {
	notImplemented(w, "node listing")
}

// GetNode returns details for a compute node.
func (h *ComputeAPIHandler) GetNode(w http.ResponseWriter, r *http.Request) {
	notImplemented(w, "node details")
}

// GetNodeJobs lists jobs scheduled on a node.
func (h *ComputeAPIHandler) GetNodeJobs(w http.ResponseWriter, r *http.Request) {
	notImplemented(w, "node job listing")
}

// GetSchedulingPolicy returns the active scheduling policy.
func (h *ComputeAPIHandler) GetSchedulingPolicy(w http.ResponseWriter, r *http.Request) {
	notImplemented(w, "scheduling policy retrieval")
}

// SetSchedulingPolicy updates the active scheduling policy.
func (h *ComputeAPIHandler) SetSchedulingPolicy(w http.ResponseWriter, r *http.Request) {
	notImplemented(w, "scheduling policy update")
}

// CapacityPlanning returns capacity planning analysis.
func (h *ComputeAPIHandler) CapacityPlanning(w http.ResponseWriter, r *http.Request) {
	notImplemented(w, "capacity planning")
}

// ResourceForecast returns resource usage forecasts.
func (h *ComputeAPIHandler) ResourceForecast(w http.ResponseWriter, r *http.Request) {
	notImplemented(w, "resource forecasting")
}

// ReserveResources reserves resources for future use.
func (h *ComputeAPIHandler) ReserveResources(w http.ResponseWriter, r *http.Request) {
	notImplemented(w, "resource reservation")
}

// ReleaseResources releases previously reserved resources.
func (h *ComputeAPIHandler) ReleaseResources(w http.ResponseWriter, r *http.Request) {
	notImplemented(w, "resource release")
}
