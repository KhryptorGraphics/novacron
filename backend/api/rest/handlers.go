package rest

import (
	"context"
	"encoding/json"
	"net/http"
	"strconv"
	"time"

	"github.com/gorilla/mux"
	"github.com/khryptorgraphics/novacron/backend/core/storage/tiering"
	"github.com/khryptorgraphics/novacron/backend/core/vm"
)

// APIHandler handles REST API requests
type APIHandler struct {
	vmManager      *vm.VMManager
	storageManager *tiering.TierManager
}

// NewAPIHandler creates a new API handler
func NewAPIHandler(vmManager *vm.VMManager, storageManager *tiering.TierManager) *APIHandler {
	return &APIHandler{
		vmManager:      vmManager,
		storageManager: storageManager,
	}
}

// RegisterRoutes registers all API routes
func (h *APIHandler) RegisterRoutes(router *mux.Router) {
	// VM endpoints
	router.HandleFunc("/api/vms", h.ListVMs).Methods("GET")
	router.HandleFunc("/api/vms", h.CreateVM).Methods("POST")
	router.HandleFunc("/api/vms/{id}", h.GetVM).Methods("GET")
	router.HandleFunc("/api/vms/{id}", h.UpdateVM).Methods("PUT")
	router.HandleFunc("/api/vms/{id}", h.DeleteVM).Methods("DELETE")
	router.HandleFunc("/api/vms/{id}/start", h.StartVM).Methods("POST")
	router.HandleFunc("/api/vms/{id}/stop", h.StopVM).Methods("POST")
	router.HandleFunc("/api/vms/{id}/restart", h.RestartVM).Methods("POST")
	router.HandleFunc("/api/vms/{id}/migrate", h.MigrateVM).Methods("POST")
	router.HandleFunc("/api/vms/{id}/snapshot", h.SnapshotVM).Methods("POST")
	router.HandleFunc("/api/vms/{id}/metrics", h.GetVMMetrics).Methods("GET")
	
	// Storage endpoints
	router.HandleFunc("/api/storage/volumes", h.ListVolumes).Methods("GET")
	router.HandleFunc("/api/storage/volumes", h.CreateVolume).Methods("POST")
	router.HandleFunc("/api/storage/volumes/{id}", h.GetVolume).Methods("GET")
	router.HandleFunc("/api/storage/volumes/{id}", h.DeleteVolume).Methods("DELETE")
	router.HandleFunc("/api/storage/volumes/{id}/tier", h.ChangeVolumeTier).Methods("PUT")
	router.HandleFunc("/api/storage/tiers", h.ListStorageTiers).Methods("GET")
	router.HandleFunc("/api/storage/metrics", h.GetStorageMetrics).Methods("GET")
	
	// Cluster endpoints
	router.HandleFunc("/api/cluster/nodes", h.ListNodes).Methods("GET")
	router.HandleFunc("/api/cluster/nodes/{id}", h.GetNode).Methods("GET")
	router.HandleFunc("/api/cluster/health", h.GetClusterHealth).Methods("GET")
	router.HandleFunc("/api/cluster/leader", h.GetLeader).Methods("GET")
	
	// Monitoring endpoints
	router.HandleFunc("/api/monitoring/metrics", h.GetSystemMetrics).Methods("GET")
	router.HandleFunc("/api/monitoring/alerts", h.GetAlerts).Methods("GET")
	router.HandleFunc("/api/monitoring/events", h.GetEvents).Methods("GET")
}

// VM Handlers

// ListVMs returns all VMs
func (h *APIHandler) ListVMs(w http.ResponseWriter, r *http.Request) {
	vmsMap := h.vmManager.ListVMs()
	
	// Convert map to slice for JSON response
	vms := make([]*vm.VM, 0, len(vmsMap))
	for _, vmInstance := range vmsMap {
		vms = append(vms, vmInstance)
	}
	
	respondJSON(w, vms)
}

// CreateVM creates a new VM
func (h *APIHandler) CreateVM(w http.ResponseWriter, r *http.Request) {
	var req CreateVMRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	
	// Convert API request to VM package request format
	createReq := vm.CreateVMRequest{
		Name: req.Name,
		Spec: vm.VMConfig{
			Name:       req.Name,
			CPUShares:  req.CPU,
			MemoryMB:   int(req.Memory),
			DiskSizeGB: int(req.Disk),
			Command:    "/bin/bash", // default command
		},
		Tags: make(map[string]string),
	}
	
	ctx := context.Background()
	vmInstance, err := h.vmManager.CreateVM(ctx, createReq)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	
	respondJSON(w, vmInstance)
}

// GetVM returns a specific VM
func (h *APIHandler) GetVM(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	vmID := vars["id"]
	
	vm, err := h.vmManager.GetVM(vmID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	
	respondJSON(w, vm)
}

// UpdateVM updates a VM configuration
func (h *APIHandler) UpdateVM(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	vmID := vars["id"]
	
	var req UpdateVMRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	
	// For now, return a placeholder response as UpdateVM is not implemented
	// TODO: Implement UpdateVM method on VMManager
	respondJSON(w, map[string]string{"status": "update not implemented", "vm_id": vmID})
}

// DeleteVM deletes a VM
func (h *APIHandler) DeleteVM(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	vmID := vars["id"]
	
	ctx := context.Background()
	if err := h.vmManager.DeleteVM(ctx, vmID); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	
	w.WriteHeader(http.StatusNoContent)
}

// StartVM starts a VM
func (h *APIHandler) StartVM(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	vmID := vars["id"]
	
	ctx := context.Background()
	if err := h.vmManager.StartVM(ctx, vmID); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	
	respondJSON(w, map[string]string{"status": "started"})
}

// StopVM stops a VM
func (h *APIHandler) StopVM(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	vmID := vars["id"]
	
	ctx := context.Background()
	if err := h.vmManager.StopVM(ctx, vmID); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	
	respondJSON(w, map[string]string{"status": "stopped"})
}

// RestartVM restarts a VM
func (h *APIHandler) RestartVM(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	vmID := vars["id"]
	
	ctx := context.Background()
	if err := h.vmManager.RestartVM(ctx, vmID); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	
	respondJSON(w, map[string]string{"status": "restarted"})
}

// MigrateVM migrates a VM to another host
func (h *APIHandler) MigrateVM(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	vmID := vars["id"]
	
	var req MigrateVMRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	
	// TODO: Implement MigrateVM method on VMManager
	respondJSON(w, map[string]string{"status": "migration not implemented", "vm_id": vmID, "target": req.TargetHost})
}

// SnapshotVM creates a snapshot of a VM
func (h *APIHandler) SnapshotVM(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	vmID := vars["id"]
	
	var req SnapshotVMRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	
	// TODO: Implement CreateSnapshot method on VMManager
	respondJSON(w, map[string]string{"status": "snapshot not implemented", "vm_id": vmID, "name": req.Name})
}

// GetVMMetrics returns metrics for a VM
func (h *APIHandler) GetVMMetrics(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	vmID := vars["id"]
	
	// Parse time range from query params (currently not used by VM manager method)
	// from := r.URL.Query().Get("from")
	// to := r.URL.Query().Get("to")
	
	ctx := context.Background()
	metrics, err := h.vmManager.GetVMMetrics(ctx, vmID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	
	respondJSON(w, metrics)
}

// Storage Handlers

// ListVolumes returns all storage volumes
func (h *APIHandler) ListVolumes(w http.ResponseWriter, r *http.Request) {
	// TODO: Implement volume listing through TierManager
	volumes := []map[string]interface{}{}
	respondJSON(w, volumes)
}

// CreateVolume creates a new storage volume
func (h *APIHandler) CreateVolume(w http.ResponseWriter, r *http.Request) {
	var req CreateVolumeRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	
	// TODO: Implement volume creation through TierManager
	volume := map[string]interface{}{
		"name": req.Name,
		"size": req.Size,
		"tier": req.Tier,
		"status": "created",
	}
	
	respondJSON(w, volume)
}

// GetVolume returns a specific volume
func (h *APIHandler) GetVolume(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	volumeID := vars["id"]
	
	// TODO: Implement volume retrieval through TierManager
	volume := map[string]interface{}{
		"id": volumeID,
		"name": "sample-volume",
		"status": "available",
	}
	
	respondJSON(w, volume)
}

// DeleteVolume deletes a volume
func (h *APIHandler) DeleteVolume(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	volumeID := vars["id"]
	
	// TODO: Implement volume deletion through TierManager
	_ = volumeID // placeholder usage
	
	w.WriteHeader(http.StatusNoContent)
}

// ChangeVolumeTier changes the storage tier of a volume
func (h *APIHandler) ChangeVolumeTier(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	volumeID := vars["id"]
	
	var req ChangeTierRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	
	// TODO: Implement tier migration through TierManager
	_ = volumeID // placeholder usage
	
	respondJSON(w, map[string]string{"status": "migrated", "new_tier": req.NewTier})
}

// ListStorageTiers returns all storage tiers
func (h *APIHandler) ListStorageTiers(w http.ResponseWriter, r *http.Request) {
	// Use the available GetTierStats method
	stats := h.storageManager.GetTierStats()
	respondJSON(w, stats)
}

// GetStorageMetrics returns storage metrics
func (h *APIHandler) GetStorageMetrics(w http.ResponseWriter, r *http.Request) {
	// Use the available GetTierStats method for metrics
	metrics := h.storageManager.GetTierStats()
	respondJSON(w, metrics)
}

// Cluster Handlers

// ListNodes returns all cluster nodes
func (h *APIHandler) ListNodes(w http.ResponseWriter, r *http.Request) {
	// Implementation would fetch from cluster membership
	nodes := []Node{
		{ID: "node-1", Address: "192.168.1.100", Status: "healthy"},
		{ID: "node-2", Address: "192.168.1.101", Status: "healthy"},
		{ID: "node-3", Address: "192.168.1.102", Status: "degraded"},
	}
	respondJSON(w, nodes)
}

// GetNode returns a specific node
func (h *APIHandler) GetNode(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	nodeID := vars["id"]
	
	// Implementation would fetch from cluster membership
	node := Node{
		ID:      nodeID,
		Address: "192.168.1.100",
		Status:  "healthy",
		CPU:     8,
		Memory:  32768,
		Disk:    1048576,
	}
	
	respondJSON(w, node)
}

// GetClusterHealth returns cluster health status
func (h *APIHandler) GetClusterHealth(w http.ResponseWriter, r *http.Request) {
	health := ClusterHealth{
		Status:       "healthy",
		TotalNodes:   3,
		HealthyNodes: 2,
		HasQuorum:    true,
		Leader:       "node-1",
		LastUpdated:  time.Now(),
	}
	
	respondJSON(w, health)
}

// GetLeader returns the current cluster leader
func (h *APIHandler) GetLeader(w http.ResponseWriter, r *http.Request) {
	leader := map[string]string{
		"id":      "node-1",
		"address": "192.168.1.100",
		"term":    "5",
	}
	
	respondJSON(w, leader)
}

// Monitoring Handlers

// GetSystemMetrics returns system-wide metrics
func (h *APIHandler) GetSystemMetrics(w http.ResponseWriter, r *http.Request) {
	// Parse metric type from query params
	metricType := r.URL.Query().Get("type")
	if metricType == "" {
		metricType = "all"
	}
	
	metrics := SystemMetrics{
		CPU: CPUMetrics{
			Usage:      65.5,
			Cores:      32,
			LoadAvg:    []float64{2.5, 2.8, 3.1},
		},
		Memory: MemoryMetrics{
			Total:     131072,
			Used:      85000,
			Free:      46072,
			Cached:    24000,
			Available: 70072,
		},
		Disk: DiskMetrics{
			Total: 4194304,
			Used:  2097152,
			Free:  2097152,
		},
		Network: NetworkMetrics{
			BytesIn:  1048576000,
			BytesOut: 524288000,
			PacketsIn: 1000000,
			PacketsOut: 500000,
		},
		Timestamp: time.Now(),
	}
	
	respondJSON(w, metrics)
}

// GetAlerts returns active alerts
func (h *APIHandler) GetAlerts(w http.ResponseWriter, r *http.Request) {
	// Parse severity from query params
	severity := r.URL.Query().Get("severity")
	
	alerts := []Alert{
		{
			ID:          "alert-1",
			Severity:    "warning",
			Message:     "High memory usage on node-3",
			Source:      "node-3",
			Timestamp:   time.Now().Add(-10 * time.Minute),
			Acknowledged: false,
		},
		{
			ID:          "alert-2",
			Severity:    "info",
			Message:     "Scheduled maintenance window approaching",
			Source:      "system",
			Timestamp:   time.Now().Add(-1 * time.Hour),
			Acknowledged: true,
		},
	}
	
	// Filter by severity if specified
	if severity != "" {
		filtered := []Alert{}
		for _, alert := range alerts {
			if alert.Severity == severity {
				filtered = append(filtered, alert)
			}
		}
		alerts = filtered
	}
	
	respondJSON(w, alerts)
}

// GetEvents returns system events
func (h *APIHandler) GetEvents(w http.ResponseWriter, r *http.Request) {
	// Parse limit from query params
	limitStr := r.URL.Query().Get("limit")
	limit := 100
	if limitStr != "" {
		if l, err := strconv.Atoi(limitStr); err == nil {
			limit = l
		}
	}
	
	events := []Event{
		{
			ID:        "event-1",
			Type:      "vm.created",
			Message:   "VM web-server-1 created",
			Source:    "api",
			Timestamp: time.Now().Add(-5 * time.Minute),
		},
		{
			ID:        "event-2",
			Type:      "vm.migrated",
			Message:   "VM database-1 migrated from node-1 to node-2",
			Source:    "scheduler",
			Timestamp: time.Now().Add(-15 * time.Minute),
		},
		{
			ID:        "event-3",
			Type:      "storage.tier_changed",
			Message:   "Volume data-vol-1 moved from hot to warm tier",
			Source:    "storage",
			Timestamp: time.Now().Add(-30 * time.Minute),
		},
	}
	
	// Limit results
	if len(events) > limit {
		events = events[:limit]
	}
	
	respondJSON(w, events)
}

// Helper function to send JSON responses
func respondJSON(w http.ResponseWriter, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(data)
}