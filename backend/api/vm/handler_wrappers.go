package vm

import (
	"context"
	"net/http"
	"time"

	"github.com/gorilla/mux"
	"github.com/khryptorgraphics/novacron/backend/core/vm"
)

// VMHandler wraps the existing Handler for router compatibility
type VMHandler struct {
	*Handler
}

// NewVMHandler creates a new VMHandler
func NewVMHandler(vmManager *vm.VMManager) *VMHandler {
	return &VMHandler{
		Handler: NewHandler(vmManager),
	}
}

// Add missing methods that the router expects but Handler doesn't have
func (h *VMHandler) SuspendVM(w http.ResponseWriter, r *http.Request) {
	// Use PauseVM as a substitute for SuspendVM
	h.PauseVM(w, r)
}

func (h *VMHandler) GetConsole(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	vmID := vars["vm_id"]

	// Simplified console access response
	response := map[string]interface{}{
		"console_url": "/console/" + vmID,
		"protocol":    "websocket",
		"message":     "Console access available",
	}

	writeJSON(w, http.StatusOK, response)
}

func (h *VMHandler) ResizeVM(w http.ResponseWriter, r *http.Request) {
	writeError(w, http.StatusNotImplemented, "NOT_IMPLEMENTED", "VM resize not implemented")
}

func (h *VMHandler) AttachVolume(w http.ResponseWriter, r *http.Request) {
	writeError(w, http.StatusNotImplemented, "NOT_IMPLEMENTED", "Volume attachment not implemented")
}

func (h *VMHandler) DetachVolume(w http.ResponseWriter, r *http.Request) {
	writeError(w, http.StatusNotImplemented, "NOT_IMPLEMENTED", "Volume detachment not implemented")
}

func (h *VMHandler) AttachNetwork(w http.ResponseWriter, r *http.Request) {
	writeError(w, http.StatusNotImplemented, "NOT_IMPLEMENTED", "Network attachment not implemented")
}

func (h *VMHandler) DetachNetwork(w http.ResponseWriter, r *http.Request) {
	writeError(w, http.StatusNotImplemented, "NOT_IMPLEMENTED", "Network detachment not implemented")
}

// MigrationHandler handles migration operations
type MigrationHandler struct {
	*MigrationHandler_Impl // Use existing implementation if available
	vmManager              *vm.VMManager
}

// MigrationHandler_Impl is the internal implementation
type MigrationHandler_Impl struct {
	vmManager *vm.VMManager
}

// NewMigrationHandler creates a new MigrationHandler
func NewMigrationHandler(migrationManager vm.MigrationManager) *MigrationHandler {
	// For now, create a basic implementation
	// This would normally use the migrationManager parameter
	return &MigrationHandler{
		MigrationHandler_Impl: &MigrationHandler_Impl{},
	}
}

func (h *MigrationHandler) InitiateMigration(w http.ResponseWriter, r *http.Request) {
	writeError(w, http.StatusNotImplemented, "NOT_IMPLEMENTED", "Migration not implemented")
}

func (h *MigrationHandler) GetMigrationStatus(w http.ResponseWriter, r *http.Request) {
	writeError(w, http.StatusNotImplemented, "NOT_IMPLEMENTED", "Migration status not implemented")
}

func (h *MigrationHandler) ListMigrations(w http.ResponseWriter, r *http.Request) {
	response := map[string]interface{}{
		"migrations": []interface{}{},
		"total":      0,
	}
	writeJSON(w, http.StatusOK, response)
}

func (h *MigrationHandler) CancelMigration(w http.ResponseWriter, r *http.Request) {
	writeError(w, http.StatusNotImplemented, "NOT_IMPLEMENTED", "Migration cancellation not implemented")
}

func (h *MigrationHandler) PauseMigration(w http.ResponseWriter, r *http.Request) {
	writeError(w, http.StatusNotImplemented, "NOT_IMPLEMENTED", "Migration pause not implemented")
}

func (h *MigrationHandler) ResumeMigration(w http.ResponseWriter, r *http.Request) {
	writeError(w, http.StatusNotImplemented, "NOT_IMPLEMENTED", "Migration resume not implemented")
}

func (h *MigrationHandler) GetVMMigrationHistory(w http.ResponseWriter, r *http.Request) {
	response := map[string]interface{}{
		"migrations": []interface{}{},
		"total":      0,
	}
	writeJSON(w, http.StatusOK, response)
}

// MetricsHandler handles metrics operations
type MetricsHandler struct {
	metricsCollector vm.MetricsCollector
}

// NewMetricsHandler creates a new MetricsHandler
func NewMetricsHandler(metricsCollector vm.MetricsCollector) *MetricsHandler {
	return &MetricsHandler{
		metricsCollector: metricsCollector,
	}
}

func (h *MetricsHandler) GetVMMetrics(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	vmID := vars["vm_id"]

	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
	defer cancel()

	if h.metricsCollector != nil {
		metrics, err := h.metricsCollector.GetVMMetrics(ctx, vmID)
		if err != nil {
			writeError(w, http.StatusInternalServerError, "METRICS_ERROR", err.Error())
			return
		}
		writeJSON(w, http.StatusOK, metrics)
		return
	}

	// Fallback response
	response := map[string]interface{}{
		"vm_id":        vmID,
		"cpu_usage":    0.0,
		"memory_usage": 0,
		"last_updated": time.Now(),
	}
	writeJSON(w, http.StatusOK, response)
}

func (h *MetricsHandler) GetAggregatedMetrics(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
	defer cancel()

	if h.metricsCollector != nil {
		metrics, err := h.metricsCollector.GetAggregatedMetrics(ctx)
		if err != nil {
			writeError(w, http.StatusInternalServerError, "METRICS_ERROR", err.Error())
			return
		}
		writeJSON(w, http.StatusOK, metrics)
		return
	}

	// Fallback response
	response := map[string]interface{}{
		"total_vms":      0,
		"running_vms":    0,
		"total_cpu":      0.0,
		"total_memory":   0,
		"last_updated":   time.Now(),
	}
	writeJSON(w, http.StatusOK, response)
}

func (h *MetricsHandler) GetUtilization(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
	defer cancel()

	if h.metricsCollector != nil {
		metrics, err := h.metricsCollector.GetUtilization(ctx)
		if err != nil {
			writeError(w, http.StatusInternalServerError, "METRICS_ERROR", err.Error())
			return
		}
		writeJSON(w, http.StatusOK, metrics)
		return
	}

	// Fallback response
	response := map[string]interface{}{
		"cpu_utilization":    0.0,
		"memory_utilization": 0.0,
		"disk_utilization":   0.0,
		"timestamp":          time.Now(),
	}
	writeJSON(w, http.StatusOK, response)
}

func (h *MetricsHandler) GetPerformanceMetrics(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	vmID := vars["vm_id"]

	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
	defer cancel()

	if h.metricsCollector != nil {
		metrics, err := h.metricsCollector.GetPerformanceMetrics(ctx, vmID)
		if err != nil {
			writeError(w, http.StatusInternalServerError, "METRICS_ERROR", err.Error())
			return
		}
		writeJSON(w, http.StatusOK, metrics)
		return
	}

	// Fallback response
	response := map[string]interface{}{
		"vm_id":         vmID,
		"response_time": 0.0,
		"throughput":    0.0,
		"latency":       0.0,
		"timestamp":     time.Now(),
	}
	writeJSON(w, http.StatusOK, response)
}

func (h *MetricsHandler) ExportMetrics(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
	defer cancel()

	if h.metricsCollector != nil {
		data, err := h.metricsCollector.ExportMetrics(ctx)
		if err != nil {
			writeError(w, http.StatusInternalServerError, "EXPORT_ERROR", err.Error())
			return
		}
		w.Header().Set("Content-Type", "text/plain")
		w.WriteHeader(http.StatusOK)
		w.Write(data)
		return
	}

	// Fallback Prometheus format
	w.Header().Set("Content-Type", "text/plain")
	w.WriteHeader(http.StatusOK)
	w.Write([]byte("# No metrics available\n"))
}

// HealthHandler handles health check operations
type HealthHandler struct {
	healthChecker vm.HealthCheckerInterface
}

// NewHealthHandler creates a new HealthHandler
func NewHealthHandler(healthChecker vm.HealthCheckerInterface) *HealthHandler {
	return &HealthHandler{
		healthChecker: healthChecker,
	}
}

func (h *HealthHandler) GetVMHealth(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	vmID := vars["vm_id"]

	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()

	if h.healthChecker != nil {
		health, err := h.healthChecker.GetVMHealth(ctx, vmID)
		if err != nil {
			writeError(w, http.StatusInternalServerError, "HEALTH_ERROR", err.Error())
			return
		}
		writeJSON(w, http.StatusOK, health)
		return
	}

	// Fallback response
	response := map[string]interface{}{
		"vm_id":      vmID,
		"status":     "unknown",
		"healthy":    false,
		"last_check": time.Now(),
		"message":    "Health checker not available",
	}
	writeJSON(w, http.StatusOK, response)
}

func (h *HealthHandler) GetSystemHealth(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()

	if h.healthChecker != nil {
		health, err := h.healthChecker.GetSystemHealth(ctx)
		if err != nil {
			writeError(w, http.StatusInternalServerError, "HEALTH_ERROR", err.Error())
			return
		}
		writeJSON(w, http.StatusOK, health)
		return
	}

	// Fallback response
	response := map[string]interface{}{
		"status":     "unknown",
		"healthy":    false,
		"last_check": time.Now(),
		"message":    "Health checker not available",
	}
	writeJSON(w, http.StatusOK, response)
}

func (h *HealthHandler) GetServicesHealth(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()

	if h.healthChecker != nil {
		health, err := h.healthChecker.GetServicesHealth(ctx)
		if err != nil {
			writeError(w, http.StatusInternalServerError, "HEALTH_ERROR", err.Error())
			return
		}
		writeJSON(w, http.StatusOK, health)
		return
	}

	// Fallback response
	response := map[string]interface{}{
		"services":   map[string]interface{}{},
		"last_check": time.Now(),
		"message":    "Health checker not available",
	}
	writeJSON(w, http.StatusOK, response)
}

func (h *HealthHandler) LivenessProbe(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()

	if h.healthChecker != nil {
		result, err := h.healthChecker.LivenessProbe(ctx)
		if err != nil {
			writeError(w, http.StatusServiceUnavailable, "PROBE_ERROR", err.Error())
			return
		}
		if !result.Healthy {
			writeJSON(w, http.StatusServiceUnavailable, result)
			return
		}
		writeJSON(w, http.StatusOK, result)
		return
	}

	// Fallback response
	response := map[string]interface{}{
		"healthy":   true,
		"status":    "ok",
		"timestamp": time.Now(),
	}
	writeJSON(w, http.StatusOK, response)
}

func (h *HealthHandler) ReadinessProbe(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()

	if h.healthChecker != nil {
		result, err := h.healthChecker.ReadinessProbe(ctx)
		if err != nil {
			writeError(w, http.StatusServiceUnavailable, "PROBE_ERROR", err.Error())
			return
		}
		if !result.Healthy {
			writeJSON(w, http.StatusServiceUnavailable, result)
			return
		}
		writeJSON(w, http.StatusOK, result)
		return
	}

	// Fallback response
	response := map[string]interface{}{
		"healthy":   true,
		"status":    "ready",
		"timestamp": time.Now(),
	}
	writeJSON(w, http.StatusOK, response)
}

// ClusterHandler handles cluster operations
type ClusterHandler struct {
	clusterManager vm.ClusterManager
}

// NewClusterHandler creates a new ClusterHandler
func NewClusterHandler(clusterManager vm.ClusterManager) *ClusterHandler {
	return &ClusterHandler{
		clusterManager: clusterManager,
	}
}

func (h *ClusterHandler) ListNodes(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
	defer cancel()

	if h.clusterManager != nil {
		nodes, err := h.clusterManager.ListNodes(ctx)
		if err != nil {
			writeError(w, http.StatusInternalServerError, "CLUSTER_ERROR", err.Error())
			return
		}
		writeJSON(w, http.StatusOK, map[string]interface{}{
			"nodes": nodes,
			"total": len(nodes),
		})
		return
	}

	// Fallback response
	response := map[string]interface{}{
		"nodes": []interface{}{},
		"total": 0,
	}
	writeJSON(w, http.StatusOK, response)
}

func (h *ClusterHandler) GetNode(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	nodeID := vars["node_id"]

	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
	defer cancel()

	if h.clusterManager != nil {
		node, err := h.clusterManager.GetNode(ctx, nodeID)
		if err != nil {
			writeError(w, http.StatusNotFound, "NODE_NOT_FOUND", err.Error())
			return
		}
		writeJSON(w, http.StatusOK, node)
		return
	}

	writeError(w, http.StatusNotFound, "NODE_NOT_FOUND", "Cluster manager not available")
}

func (h *ClusterHandler) AddNode(w http.ResponseWriter, r *http.Request) {
	writeError(w, http.StatusNotImplemented, "NOT_IMPLEMENTED", "Add node not implemented")
}

func (h *ClusterHandler) RemoveNode(w http.ResponseWriter, r *http.Request) {
	writeError(w, http.StatusNotImplemented, "NOT_IMPLEMENTED", "Remove node not implemented")
}

func (h *ClusterHandler) DrainNode(w http.ResponseWriter, r *http.Request) {
	writeError(w, http.StatusNotImplemented, "NOT_IMPLEMENTED", "Drain node not implemented")
}

func (h *ClusterHandler) CordonNode(w http.ResponseWriter, r *http.Request) {
	writeError(w, http.StatusNotImplemented, "NOT_IMPLEMENTED", "Cordon node not implemented")
}

func (h *ClusterHandler) UncordonNode(w http.ResponseWriter, r *http.Request) {
	writeError(w, http.StatusNotImplemented, "NOT_IMPLEMENTED", "Uncordon node not implemented")
}

func (h *ClusterHandler) GetClusterStatus(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
	defer cancel()

	if h.clusterManager != nil {
		status, err := h.clusterManager.GetClusterStatus(ctx)
		if err != nil {
			writeError(w, http.StatusInternalServerError, "CLUSTER_ERROR", err.Error())
			return
		}
		writeJSON(w, http.StatusOK, status)
		return
	}

	// Fallback response
	response := map[string]interface{}{
		"status":       "unknown",
		"total_nodes":  0,
		"ready_nodes":  0,
		"total_vms":    0,
		"running_vms":  0,
		"last_updated": time.Now(),
	}
	writeJSON(w, http.StatusOK, response)
}

func (h *ClusterHandler) RebalanceCluster(w http.ResponseWriter, r *http.Request) {
	writeError(w, http.StatusNotImplemented, "NOT_IMPLEMENTED", "Cluster rebalance not implemented")
}