//go:build experimental

package vm


import (
	"context"
	"encoding/json"
	"net/http"
	"time"

	"github.com/gorilla/mux"
	"github.com/khryptorgraphics/novacron/backend/core/vm"
)

// SchedulerHandler handles VM scheduler API requests
type SchedulerHandler struct {
	scheduler *vm.VMScheduler
	vmManager *vm.VMManager
}

// NewSchedulerHandler creates a new VM scheduler API handler
func NewSchedulerHandler(scheduler *vm.VMScheduler, vmManager *vm.VMManager) *SchedulerHandler {
	return &SchedulerHandler{
		scheduler: scheduler,
		vmManager: vmManager,
	}
}

// RegisterRoutes registers VM scheduler API routes
func (h *SchedulerHandler) RegisterRoutes(router *mux.Router) {
	router.HandleFunc("/scheduler/nodes", h.ListNodes).Methods("GET")
	router.HandleFunc("/scheduler/nodes", h.RegisterNode).Methods("POST")
	router.HandleFunc("/scheduler/nodes/{id}", h.GetNode).Methods("GET")
	router.HandleFunc("/scheduler/nodes/{id}", h.UpdateNode).Methods("PUT")
	router.HandleFunc("/scheduler/nodes/{id}", h.UnregisterNode).Methods("DELETE")
	router.HandleFunc("/scheduler/vms/{vm_id}/schedule", h.ScheduleVM).Methods("POST")
}

// ListNodes handles GET /scheduler/nodes
func (h *SchedulerHandler) ListNodes(w http.ResponseWriter, r *http.Request) {
	// Get nodes
	nodes := h.scheduler.ListNodes()

	// Convert to response format
	response := make([]map[string]interface{}, 0, len(nodes))
	for _, node := range nodes {
		response = append(response, map[string]interface{}{
			"node_id":              node.NodeID,
			"total_cpu":            node.TotalCPU,
			"used_cpu":             node.UsedCPU,
			"total_memory_mb":      node.TotalMemoryMB,
			"used_memory_mb":       node.UsedMemoryMB,
			"total_disk_gb":        node.TotalDiskGB,
			"used_disk_gb":         node.UsedDiskGB,
			"cpu_usage_percent":    node.CPUUsagePercent,
			"memory_usage_percent": node.MemoryUsagePercent,
			"disk_usage_percent":   node.DiskUsagePercent,
			"vm_count":             node.VMCount,
			"status":               node.Status,
			"labels":               node.Labels,
		})
	}

	// Write response
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// RegisterNode handles POST /scheduler/nodes
func (h *SchedulerHandler) RegisterNode(w http.ResponseWriter, r *http.Request) {
	// Parse request
	var request struct {
		NodeID            string            `json:"node_id"`
		TotalCPU          int               `json:"total_cpu"`
		TotalMemoryMB     int               `json:"total_memory_mb"`
		TotalDiskGB       int               `json:"total_disk_gb"`
		Status            string            `json:"status"`
		Labels            map[string]string `json:"labels"`
	}

	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Create node info
	nodeInfo := &vm.NodeResourceInfo{
		NodeID:            request.NodeID,
		TotalCPU:          request.TotalCPU,
		UsedCPU:           0,
		TotalMemoryMB:     request.TotalMemoryMB,
		UsedMemoryMB:      0,
		TotalDiskGB:       request.TotalDiskGB,
		UsedDiskGB:        0,
		CPUUsagePercent:   0,
		MemoryUsagePercent: 0,
		DiskUsagePercent:   0,
		VMCount:           0,
		Status:            request.Status,
		Labels:            request.Labels,
	}

	// Register node
	if err := h.scheduler.RegisterNode(nodeInfo); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Write response
	response := map[string]interface{}{
		"node_id":              nodeInfo.NodeID,
		"total_cpu":            nodeInfo.TotalCPU,
		"used_cpu":             nodeInfo.UsedCPU,
		"total_memory_mb":      nodeInfo.TotalMemoryMB,
		"used_memory_mb":       nodeInfo.UsedMemoryMB,
		"total_disk_gb":        nodeInfo.TotalDiskGB,
		"used_disk_gb":         nodeInfo.UsedDiskGB,
		"cpu_usage_percent":    nodeInfo.CPUUsagePercent,
		"memory_usage_percent": nodeInfo.MemoryUsagePercent,
		"disk_usage_percent":   nodeInfo.DiskUsagePercent,
		"vm_count":             nodeInfo.VMCount,
		"status":               nodeInfo.Status,
		"labels":               nodeInfo.Labels,
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(response)
}

// GetNode handles GET /scheduler/nodes/{id}
func (h *SchedulerHandler) GetNode(w http.ResponseWriter, r *http.Request) {
	// Get node ID from URL
	vars := mux.Vars(r)
	nodeID := vars["id"]

	// Get node
	node, err := h.scheduler.GetNode(nodeID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}

	// Write response
	response := map[string]interface{}{
		"node_id":              node.NodeID,
		"total_cpu":            node.TotalCPU,
		"used_cpu":             node.UsedCPU,
		"total_memory_mb":      node.TotalMemoryMB,
		"used_memory_mb":       node.UsedMemoryMB,
		"total_disk_gb":        node.TotalDiskGB,
		"used_disk_gb":         node.UsedDiskGB,
		"cpu_usage_percent":    node.CPUUsagePercent,
		"memory_usage_percent": node.MemoryUsagePercent,
		"disk_usage_percent":   node.DiskUsagePercent,
		"vm_count":             node.VMCount,
		"status":               node.Status,
		"labels":               node.Labels,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// UpdateNode handles PUT /scheduler/nodes/{id}
func (h *SchedulerHandler) UpdateNode(w http.ResponseWriter, r *http.Request) {
	// Get node ID from URL
	vars := mux.Vars(r)
	nodeID := vars["id"]

	// Parse request
	var request struct {
		TotalCPU          int               `json:"total_cpu"`
		TotalMemoryMB     int               `json:"total_memory_mb"`
		TotalDiskGB       int               `json:"total_disk_gb"`
		Status            string            `json:"status"`
		Labels            map[string]string `json:"labels"`
	}

	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Get current node info
	node, err := h.scheduler.GetNode(nodeID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}

	// Update node info
	node.TotalCPU = request.TotalCPU
	node.TotalMemoryMB = request.TotalMemoryMB
	node.TotalDiskGB = request.TotalDiskGB
	node.Status = request.Status
	node.Labels = request.Labels

	// Update usage percentages
	node.CPUUsagePercent = float64(node.UsedCPU) / float64(node.TotalCPU) * 100
	node.MemoryUsagePercent = float64(node.UsedMemoryMB) / float64(node.TotalMemoryMB) * 100
	node.DiskUsagePercent = float64(node.UsedDiskGB) / float64(node.TotalDiskGB) * 100

	// Update node
	if err := h.scheduler.UpdateNode(node); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Write response
	response := map[string]interface{}{
		"node_id":              node.NodeID,
		"total_cpu":            node.TotalCPU,
		"used_cpu":             node.UsedCPU,
		"total_memory_mb":      node.TotalMemoryMB,
		"used_memory_mb":       node.UsedMemoryMB,
		"total_disk_gb":        node.TotalDiskGB,
		"used_disk_gb":         node.UsedDiskGB,
		"cpu_usage_percent":    node.CPUUsagePercent,
		"memory_usage_percent": node.MemoryUsagePercent,
		"disk_usage_percent":   node.DiskUsagePercent,
		"vm_count":             node.VMCount,
		"status":               node.Status,
		"labels":               node.Labels,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// UnregisterNode handles DELETE /scheduler/nodes/{id}
func (h *SchedulerHandler) UnregisterNode(w http.ResponseWriter, r *http.Request) {
	// Get node ID from URL
	vars := mux.Vars(r)
	nodeID := vars["id"]

	// Unregister node
	if err := h.scheduler.UnregisterNode(nodeID); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Write response
	w.WriteHeader(http.StatusNoContent)
}

// ScheduleVM handles POST /scheduler/vms/{vm_id}/schedule
func (h *SchedulerHandler) ScheduleVM(w http.ResponseWriter, r *http.Request) {
	// Get VM ID from URL
	vars := mux.Vars(r)
	vmID := vars["vm_id"]

	// Get VM
	vm, err := h.vmManager.GetVM(vmID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}

	// Schedule VM
	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()

	nodeID, err := h.scheduler.ScheduleVM(ctx, vm)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Reserve resources
	if err := h.scheduler.ReserveResources(nodeID, vm); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Write response
	response := map[string]interface{}{
		"vm_id":   vmID,
		"node_id": nodeID,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}
