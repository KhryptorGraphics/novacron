package vm

import (
	"context"
	"encoding/json"
	"net/http"
	"time"

	"github.com/gorilla/mux"
	"github.com/khryptorgraphics/novacron/backend/core/vm"
)

// Handler handles VM API requests
type Handler struct {
	vmManager *vm.VMManager
}

// NewHandler creates a new VM API handler
func NewHandler(vmManager *vm.VMManager) *Handler {
	return &Handler{
		vmManager: vmManager,
	}
}

// RegisterRoutes registers VM API routes
func (h *Handler) RegisterRoutes(router *mux.Router) {
	router.HandleFunc("/vms", h.ListVMs).Methods("GET")
	router.HandleFunc("/vms", h.CreateVM).Methods("POST")
	router.HandleFunc("/vms/{id}", h.GetVM).Methods("GET")
	router.HandleFunc("/vms/{id}", h.UpdateVM).Methods("PUT")
	router.HandleFunc("/vms/{id}", h.DeleteVM).Methods("DELETE")
	router.HandleFunc("/vms/{id}/start", h.StartVM).Methods("POST")
	router.HandleFunc("/vms/{id}/stop", h.StopVM).Methods("POST")
	router.HandleFunc("/vms/{id}/restart", h.RestartVM).Methods("POST")
	router.HandleFunc("/vms/{id}/pause", h.PauseVM).Methods("POST")
	router.HandleFunc("/vms/{id}/resume", h.ResumeVM).Methods("POST")
	router.HandleFunc("/vms/{id}/metrics", h.GetVMMetrics).Methods("GET")
}

// ListVMs handles GET /vms
func (h *Handler) ListVMs(w http.ResponseWriter, r *http.Request) {
	// Get VMs
	vms := h.vmManager.ListVMs()
	
	// Convert to response format
	response := make([]map[string]interface{}, 0, len(vms))
	for _, vm := range vms {
		response = append(response, map[string]interface{}{
			"id":         vm.ID(),
			"name":       vm.Name(),
			"state":      vm.State(),
			"node_id":    vm.GetNodeID(),
			"created_at": vm.GetCreatedAt(),
			"updated_at": vm.GetUpdatedAt(),
		})
	}
	
	// Write response
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// CreateVM handles POST /vms
func (h *Handler) CreateVM(w http.ResponseWriter, r *http.Request) {
	// Parse request
	var request struct {
		Name       string            `json:"name"`
		Command    string            `json:"command"`
		Args       []string          `json:"args"`
		CPUShares  int               `json:"cpu_shares"`
		MemoryMB   int               `json:"memory_mb"`
		DiskSizeGB int               `json:"disk_size_gb"`
		Tags       map[string]string `json:"tags"`
	}
	
	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	
	// Create VM config
	config := vm.VMConfig{
		Name:       request.Name,
		Command:    request.Command,
		Args:       request.Args,
		CPUShares:  request.CPUShares,
		MemoryMB:   request.MemoryMB,
		DiskSizeGB: request.DiskSizeGB,
		Tags:       request.Tags,
	}
	
	// Create VM
	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()
	
	newVM, err := h.vmManager.CreateVM(ctx, config)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	
	// Write response
	response := map[string]interface{}{
		"id":         newVM.ID(),
		"name":       newVM.Name(),
		"state":      newVM.State(),
		"node_id":    newVM.GetNodeID(),
		"created_at": newVM.GetCreatedAt(),
		"updated_at": newVM.GetUpdatedAt(),
	}
	
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(response)
}

// GetVM handles GET /vms/{id}
func (h *Handler) GetVM(w http.ResponseWriter, r *http.Request) {
	// Get VM ID from URL
	vars := mux.Vars(r)
	vmID := vars["id"]
	
	// Get VM
	vm, err := h.vmManager.GetVM(vmID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	
	// Write response
	response := map[string]interface{}{
		"id":         vm.ID(),
		"name":       vm.Name(),
		"state":      vm.State(),
		"node_id":    vm.GetNodeID(),
		"created_at": vm.GetCreatedAt(),
		"updated_at": vm.GetUpdatedAt(),
		"config": map[string]interface{}{
			"command":     vm.GetCommand(),
			"args":        vm.GetArgs(),
			"cpu_shares":  vm.GetCPUShares(),
			"memory_mb":   vm.GetMemoryMB(),
			"disk_size_gb": vm.GetDiskSizeGB(),
			"tags":        vm.GetTags(),
		},
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// UpdateVM handles PUT /vms/{id}
func (h *Handler) UpdateVM(w http.ResponseWriter, r *http.Request) {
	// Get VM ID from URL
	vars := mux.Vars(r)
	vmID := vars["id"]
	
	// Parse request
	var request struct {
		Name       string            `json:"name"`
		CPUShares  int               `json:"cpu_shares"`
		MemoryMB   int               `json:"memory_mb"`
		DiskSizeGB int               `json:"disk_size_gb"`
		Tags       map[string]string `json:"tags"`
	}
	
	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	
	// Get VM
	vm, err := h.vmManager.GetVM(vmID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	
	// Update VM
	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()
	
	// Update name if provided
	if request.Name != "" {
		vm.SetName(request.Name)
	}
	
	// Update CPU shares if provided
	if request.CPUShares > 0 {
		vm.SetCPUShares(request.CPUShares)
	}
	
	// Update memory if provided
	if request.MemoryMB > 0 {
		vm.SetMemoryMB(request.MemoryMB)
	}
	
	// Update disk size if provided
	if request.DiskSizeGB > 0 {
		vm.SetDiskSizeGB(request.DiskSizeGB)
	}
	
	// Update tags if provided
	if request.Tags != nil {
		vm.SetTags(request.Tags)
	}
	
	// Write response
	response := map[string]interface{}{
		"id":         vm.ID(),
		"name":       vm.Name(),
		"state":      vm.State(),
		"node_id":    vm.GetNodeID(),
		"created_at": vm.GetCreatedAt(),
		"updated_at": vm.GetUpdatedAt(),
		"config": map[string]interface{}{
			"command":     vm.GetCommand(),
			"args":        vm.GetArgs(),
			"cpu_shares":  vm.GetCPUShares(),
			"memory_mb":   vm.GetMemoryMB(),
			"disk_size_gb": vm.GetDiskSizeGB(),
			"tags":        vm.GetTags(),
		},
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// DeleteVM handles DELETE /vms/{id}
func (h *Handler) DeleteVM(w http.ResponseWriter, r *http.Request) {
	// Get VM ID from URL
	vars := mux.Vars(r)
	vmID := vars["id"]
	
	// Delete VM
	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()
	
	if err := h.vmManager.DeleteVM(ctx, vmID); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	
	// Write response
	w.WriteHeader(http.StatusNoContent)
}

// StartVM handles POST /vms/{id}/start
func (h *Handler) StartVM(w http.ResponseWriter, r *http.Request) {
	// Get VM ID from URL
	vars := mux.Vars(r)
	vmID := vars["id"]
	
	// Start VM
	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()
	
	if err := h.vmManager.StartVM(ctx, vmID); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	
	// Get updated VM
	vm, err := h.vmManager.GetVM(vmID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	
	// Write response
	response := map[string]interface{}{
		"id":         vm.ID(),
		"name":       vm.Name(),
		"state":      vm.State(),
		"node_id":    vm.GetNodeID(),
		"created_at": vm.GetCreatedAt(),
		"updated_at": vm.GetUpdatedAt(),
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// StopVM handles POST /vms/{id}/stop
func (h *Handler) StopVM(w http.ResponseWriter, r *http.Request) {
	// Get VM ID from URL
	vars := mux.Vars(r)
	vmID := vars["id"]
	
	// Stop VM
	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()
	
	if err := h.vmManager.StopVM(ctx, vmID); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	
	// Get updated VM
	vm, err := h.vmManager.GetVM(vmID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	
	// Write response
	response := map[string]interface{}{
		"id":         vm.ID(),
		"name":       vm.Name(),
		"state":      vm.State(),
		"node_id":    vm.GetNodeID(),
		"created_at": vm.GetCreatedAt(),
		"updated_at": vm.GetUpdatedAt(),
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// RestartVM handles POST /vms/{id}/restart
func (h *Handler) RestartVM(w http.ResponseWriter, r *http.Request) {
	// Get VM ID from URL
	vars := mux.Vars(r)
	vmID := vars["id"]
	
	// Restart VM
	ctx, cancel := context.WithTimeout(r.Context(), 60*time.Second)
	defer cancel()
	
	if err := h.vmManager.RestartVM(ctx, vmID); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	
	// Get updated VM
	vm, err := h.vmManager.GetVM(vmID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	
	// Write response
	response := map[string]interface{}{
		"id":         vm.ID(),
		"name":       vm.Name(),
		"state":      vm.State(),
		"node_id":    vm.GetNodeID(),
		"created_at": vm.GetCreatedAt(),
		"updated_at": vm.GetUpdatedAt(),
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// PauseVM handles POST /vms/{id}/pause
func (h *Handler) PauseVM(w http.ResponseWriter, r *http.Request) {
	// Get VM ID from URL
	vars := mux.Vars(r)
	vmID := vars["id"]
	
	// Pause VM
	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()
	
	if err := h.vmManager.PauseVM(ctx, vmID); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	
	// Get updated VM
	vm, err := h.vmManager.GetVM(vmID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	
	// Write response
	response := map[string]interface{}{
		"id":         vm.ID(),
		"name":       vm.Name(),
		"state":      vm.State(),
		"node_id":    vm.GetNodeID(),
		"created_at": vm.GetCreatedAt(),
		"updated_at": vm.GetUpdatedAt(),
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// ResumeVM handles POST /vms/{id}/resume
func (h *Handler) ResumeVM(w http.ResponseWriter, r *http.Request) {
	// Get VM ID from URL
	vars := mux.Vars(r)
	vmID := vars["id"]
	
	// Resume VM
	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()
	
	if err := h.vmManager.ResumeVM(ctx, vmID); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	
	// Get updated VM
	vm, err := h.vmManager.GetVM(vmID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	
	// Write response
	response := map[string]interface{}{
		"id":         vm.ID(),
		"name":       vm.Name(),
		"state":      vm.State(),
		"node_id":    vm.GetNodeID(),
		"created_at": vm.GetCreatedAt(),
		"updated_at": vm.GetUpdatedAt(),
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// GetVMMetrics handles GET /vms/{id}/metrics
func (h *Handler) GetVMMetrics(w http.ResponseWriter, r *http.Request) {
	// Get VM ID from URL
	vars := mux.Vars(r)
	vmID := vars["id"]
	
	// Get VM
	vm, err := h.vmManager.GetVM(vmID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	
	// Get VM stats
	stats := vm.GetStats()
	
	// Write response
	response := map[string]interface{}{
		"vm_id":       vm.ID(),
		"cpu_usage":   stats.CPUUsage,
		"memory_usage": stats.MemoryUsage,
		"network_sent": stats.NetworkSent,
		"network_recv": stats.NetworkRecv,
		"last_updated": stats.LastUpdated,
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}
