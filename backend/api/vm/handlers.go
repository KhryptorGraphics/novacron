package vm

import (
	"context"
	"encoding/json"
	"net/http"
	"time"
	"sort"
	"strconv"
	"strings"


	"github.com/gorilla/mux"
	"github.com/khryptorgraphics/novacron/backend/core/vm"
)

const jsonCT = "application/json; charset=utf-8"

func writeJSON(w http.ResponseWriter, status int, v interface{}) {
	w.Header().Set("Content-Type", jsonCT)
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, code, msg string) {
	w.Header().Set("Content-Type", jsonCT)
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]interface{}{
		"error": map[string]string{"code": code, "message": msg},
	})
}

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

	// Collect into slice for processing
	type item struct{ id string; vm *vm.VM }
	items := make([]item, 0, len(vms))
	for _, m := range vms { items = append(items, item{id: m.ID(), vm: m}) }

	// Parse and validate query params
	q := r.URL.Query()
	page := 1; pageSize := 20; sortBy := "createdAt"; sortDir := "asc"
	if v := q.Get("page"); v != "" { if n, err := strconv.Atoi(v); err != nil || n < 1 { writeError(w, http.StatusBadRequest, "invalid_argument", "invalid page"); return } else { page = n } }
	if v := q.Get("pageSize"); v != "" { if n, err := strconv.Atoi(v); err != nil || n < 1 || n > 100 { writeError(w, http.StatusBadRequest, "invalid_argument", "invalid pageSize"); return } else { pageSize = n } }
	if v := q.Get("sortBy"); v != "" { switch v { case "name","createdAt","state": sortBy = v; default: writeError(w, http.StatusBadRequest, "invalid_argument", "invalid sortBy"); return } }
	if v := q.Get("sortDir"); v != "" { switch v { case "asc","desc": sortDir = v; default: writeError(w, http.StatusBadRequest, "invalid_argument", "invalid sortDir"); return } }
	stateFilter := strings.ToLower(q.Get("state"))
	nodeIDFilter := q.Get("nodeId")
	query := strings.ToLower(q.Get("q"))

	// Filter
	filtered := items[:0]
	for _, it := range items {
		vm := it.vm
		if stateFilter != "" && strings.ToLower(string(vm.State())) != stateFilter { continue }
		if nodeIDFilter != "" && vm.GetNodeID() != nodeIDFilter { continue }
		if query != "" {
			name := strings.ToLower(vm.Name())
			id := strings.ToLower(vm.ID())
			if !strings.Contains(name, query) && !strings.Contains(id, query) { continue }
		}
		filtered = append(filtered, it)
	}

	// Sort
	sort.SliceStable(filtered, func(i, j int) bool {
		vi, vj := filtered[i].vm, filtered[j].vm
		less := false
		switch sortBy {
		case "name":
			if vi.Name() == vj.Name() { less = filtered[i].id < filtered[j].id } else { less = vi.Name() < vj.Name() }
		case "state":
			if vi.State() == vj.State() { less = filtered[i].id < filtered[j].id } else { less = string(vi.State()) < string(vj.State()) }
		default: // createdAt
			ci, cj := vi.GetCreatedAt(), vj.GetCreatedAt()
			if ci.Equal(cj) { less = filtered[i].id < filtered[j].id } else { less = ci.Before(cj) }
		}
		if sortDir == "asc" { return less }
		return !less
	})

	// Paginate
	total := len(filtered)
	start := (page-1)*pageSize
	if start > total { start = total }
	end := start + pageSize
	if end > total { end = total }
	paged := filtered[start:end]

	// Project for response
	response := make([]map[string]interface{}, 0, len(paged))
	for _, it := range paged {
		vm := it.vm
		response = append(response, map[string]interface{}{
			"id":         vm.ID(),
			"name":       vm.Name(),
			"state":      vm.State(),
			"node_id":    vm.GetNodeID(),
			"created_at": vm.GetCreatedAt(),
			"updated_at": vm.GetUpdatedAt(),
		})
	}

	// Set pagination header
	totalPages := (total + pageSize - 1) / pageSize
	pagination := map[string]interface{}{
		"page": page, "pageSize": pageSize, "total": total, "totalPages": totalPages,
		"sortBy": sortBy, "sortDir": sortDir,
	}
	pjson, _ := json.Marshal(pagination)
	w.Header().Set("X-Pagination", string(pjson))

	// Write response
	w.Header().Set("Content-Type", jsonCT)
	json.NewEncoder(w).Encode(response)
}

// CreateVM handles POST /vms
func (h *Handler) CreateVM(w http.ResponseWriter, r *http.Request) {
	// Parse request (minimal validation; allow empty command in core tests)
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
		writeError(w, http.StatusBadRequest, "invalid_argument", "invalid JSON payload")
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

	// Create VM request
	createRequest := vm.CreateVMRequest{
		Name: request.Name,
		Spec: config,
		Tags: request.Tags,
	}

	// Create VM
	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()

	newVM, err := h.vmManager.CreateVM(ctx, createRequest)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "internal", err.Error())
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

	w.Header().Set("Content-Type", jsonCT)
	w.Header().Set("Location", "/api/v1/vms/"+newVM.ID())
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
		writeError(w, http.StatusNotFound, "not_found", "VM not found")
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

	w.Header().Set("Content-Type", jsonCT)
	json.NewEncoder(w).Encode(response)
}

// UpdateVM handles PUT /vms/{id}
func (h *Handler) UpdateVM(w http.ResponseWriter, r *http.Request) {
	// Get VM ID from URL
	vars := mux.Vars(r)
	vmID := vars["id"]

	// Get VM
	vm, err := h.vmManager.GetVM(vmID)
	if err != nil {
		writeError(w, http.StatusNotFound, "not_found", "VM not found")
		return
	}

	// PATCH semantics: allow only name and tags; reject others if present
	var raw map[string]json.RawMessage
	if err := json.NewDecoder(r.Body).Decode(&raw); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_argument", "invalid JSON payload")
		return
	}
	if len(raw) == 0 { writeError(w, http.StatusBadRequest, "invalid_argument", "no supported fields to update"); return }
	unsupported := make([]string, 0)
	for k := range raw { if k != "name" && k != "tags" { unsupported = append(unsupported, k) } }
	if len(unsupported) > 0 {
		writeError(w, http.StatusBadRequest, "invalid_argument", "unsupported fields: "+strings.Join(unsupported, ", "))
		return
	}
	// Apply allowed updates
	if v, ok := raw["name"]; ok {
		var name string
		if err := json.Unmarshal(v, &name); err != nil { writeError(w, http.StatusBadRequest, "invalid_argument", "name must be a string"); return }
		name = strings.TrimSpace(name)
		if name == "" { writeError(w, http.StatusBadRequest, "invalid_argument", "name cannot be empty"); return }
		vm.SetName(name)
	}
	if v, ok := raw["tags"]; ok {
		var tags map[string]string
		if err := json.Unmarshal(v, &tags); err != nil { writeError(w, http.StatusBadRequest, "invalid_argument", "tags must be an object of string:string"); return }
		vm.SetTags(tags)
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

	w.Header().Set("Content-Type", jsonCT)
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
		writeError(w, http.StatusInternalServerError, "internal", err.Error())
		return
	}

	// Write response (consistent envelope; choose 200 OK here)
	w.Header().Set("Content-Type", jsonCT)
	json.NewEncoder(w).Encode(map[string]interface{}{"id": vmID})
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
		writeError(w, http.StatusInternalServerError, "internal", err.Error())
		return
	}

	// Get updated VM
	vm, err := h.vmManager.GetVM(vmID)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "internal", err.Error())
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

	w.Header().Set("Content-Type", jsonCT)
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
		writeError(w, http.StatusInternalServerError, "internal", err.Error())
		return
	}

	// Get updated VM
	vm, err := h.vmManager.GetVM(vmID)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "internal", err.Error())
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

	w.Header().Set("Content-Type", jsonCT)
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
		writeError(w, http.StatusInternalServerError, "internal", err.Error())
		return
	}

	// Get updated VM
	vm, err := h.vmManager.GetVM(vmID)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "internal", err.Error())
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

	w.Header().Set("Content-Type", jsonCT)
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
		writeError(w, http.StatusInternalServerError, "internal", err.Error())
		return
	}

	// Get updated VM
	vm, err := h.vmManager.GetVM(vmID)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "internal", err.Error())
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

	w.Header().Set("Content-Type", jsonCT)
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
		writeError(w, http.StatusInternalServerError, "internal", err.Error())
		return
	}

	// Get updated VM
	vm, err := h.vmManager.GetVM(vmID)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "internal", err.Error())
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

	w.Header().Set("Content-Type", jsonCT)
	json.NewEncoder(w).Encode(response)
}
