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

// StorageHandler handles VM storage API requests
type StorageHandler struct {
	storageManager *vm.VMStorageManager
}

// NewStorageHandler creates a new VM storage API handler
func NewStorageHandler(storageManager *vm.VMStorageManager) *StorageHandler {
	return &StorageHandler{
		storageManager: storageManager,
	}
}

// RegisterRoutes registers VM storage API routes
func (h *StorageHandler) RegisterRoutes(router *mux.Router) {
	router.HandleFunc("/storage/pools", h.ListStoragePools).Methods("GET")
	router.HandleFunc("/storage/pools", h.CreateStoragePool).Methods("POST")
	router.HandleFunc("/storage/pools/{id}", h.GetStoragePool).Methods("GET")
	router.HandleFunc("/storage/pools/{id}", h.DeleteStoragePool).Methods("DELETE")
	router.HandleFunc("/storage/volumes", h.ListStorageVolumes).Methods("GET")
	router.HandleFunc("/storage/volumes", h.CreateStorageVolume).Methods("POST")
	router.HandleFunc("/storage/volumes/{id}", h.GetStorageVolume).Methods("GET")
	router.HandleFunc("/storage/volumes/{id}", h.DeleteStorageVolume).Methods("DELETE")
	router.HandleFunc("/storage/volumes/{id}/resize", h.ResizeStorageVolume).Methods("POST")
	router.HandleFunc("/storage/volumes/{id}/clone", h.CloneStorageVolume).Methods("POST")
}

// ListStoragePools handles GET /storage/pools
func (h *StorageHandler) ListStoragePools(w http.ResponseWriter, r *http.Request) {
	// Get pools
	pools := h.storageManager.ListStoragePools()

	// Convert to response format
	response := make([]map[string]interface{}, 0, len(pools))
	for _, pool := range pools {
		response = append(response, map[string]interface{}{
			"id":          pool.ID,
			"name":        pool.Name,
			"type":        pool.Type,
			"path":        pool.Path,
			"total_space": pool.TotalSpace,
			"used_space":  pool.UsedSpace,
			"created_at":  pool.CreatedAt,
			"updated_at":  pool.UpdatedAt,
			"tags":        pool.Tags,
			"metadata":    pool.Metadata,
		})
	}

	// Write response
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// CreateStoragePool handles POST /storage/pools
func (h *StorageHandler) CreateStoragePool(w http.ResponseWriter, r *http.Request) {
	// Parse request
	var request struct {
		Name       string            `json:"name"`
		Type       string            `json:"type"`
		Path       string            `json:"path"`
		Tags       []string          `json:"tags"`
		Metadata   map[string]string `json:"metadata"`
	}

	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Validate storage type
	var storageType vm.StorageType
	switch request.Type {
	case "local":
		storageType = vm.StorageTypeLocal
	case "nfs":
		storageType = vm.StorageTypeNFS
	case "ceph":
		storageType = vm.StorageTypeCeph
	case "iscsi":
		storageType = vm.StorageTypeISCSI
	default:
		http.Error(w, "Invalid storage type. Must be 'local', 'nfs', 'ceph', or 'iscsi'.", http.StatusBadRequest)
		return
	}

	// Create pool
	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()

	pool, err := h.storageManager.CreateStoragePool(ctx, request.Name, storageType, request.Path, request.Tags, request.Metadata)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Write response
	response := map[string]interface{}{
		"id":          pool.ID,
		"name":        pool.Name,
		"type":        pool.Type,
		"path":        pool.Path,
		"total_space": pool.TotalSpace,
		"used_space":  pool.UsedSpace,
		"created_at":  pool.CreatedAt,
		"updated_at":  pool.UpdatedAt,
		"tags":        pool.Tags,
		"metadata":    pool.Metadata,
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(response)
}

// GetStoragePool handles GET /storage/pools/{id}
func (h *StorageHandler) GetStoragePool(w http.ResponseWriter, r *http.Request) {
	// Get pool ID from URL
	vars := mux.Vars(r)
	poolID := vars["id"]

	// Get pool
	pool, err := h.storageManager.GetStoragePool(poolID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}

	// Write response
	response := map[string]interface{}{
		"id":          pool.ID,
		"name":        pool.Name,
		"type":        pool.Type,
		"path":        pool.Path,
		"total_space": pool.TotalSpace,
		"used_space":  pool.UsedSpace,
		"created_at":  pool.CreatedAt,
		"updated_at":  pool.UpdatedAt,
		"tags":        pool.Tags,
		"metadata":    pool.Metadata,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// DeleteStoragePool handles DELETE /storage/pools/{id}
func (h *StorageHandler) DeleteStoragePool(w http.ResponseWriter, r *http.Request) {
	// Get pool ID from URL
	vars := mux.Vars(r)
	poolID := vars["id"]

	// Delete pool
	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()

	if err := h.storageManager.DeleteStoragePool(ctx, poolID); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Write response
	w.WriteHeader(http.StatusNoContent)
}

// ListStorageVolumes handles GET /storage/volumes
func (h *StorageHandler) ListStorageVolumes(w http.ResponseWriter, r *http.Request) {
	// Get pool ID from query parameters
	poolID := r.URL.Query().Get("pool_id")

	// Get volumes
	var volumes []*vm.StorageVolume
	if poolID != "" {
		volumes = h.storageManager.ListStorageVolumesInPool(poolID)
	} else {
		volumes = h.storageManager.ListStorageVolumes()
	}

	// Convert to response format
	response := make([]map[string]interface{}, 0, len(volumes))
	for _, volume := range volumes {
		response = append(response, map[string]interface{}{
			"id":         volume.ID,
			"name":       volume.Name,
			"pool_id":    volume.PoolID,
			"format":     volume.Format,
			"capacity":   volume.Capacity,
			"allocation": volume.Allocation,
			"path":       volume.Path,
			"created_at": volume.CreatedAt,
			"updated_at": volume.UpdatedAt,
			"tags":       volume.Tags,
			"metadata":   volume.Metadata,
		})
	}

	// Write response
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// CreateStorageVolume handles POST /storage/volumes
func (h *StorageHandler) CreateStorageVolume(w http.ResponseWriter, r *http.Request) {
	// Parse request
	var request struct {
		Name       string            `json:"name"`
		PoolID     string            `json:"pool_id"`
		Format     string            `json:"format"`
		Capacity   int64             `json:"capacity"`
		Tags       []string          `json:"tags"`
		Metadata   map[string]string `json:"metadata"`
	}

	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Validate format
	var format vm.StorageFormat
	switch request.Format {
	case "raw":
		format = vm.StorageFormatRaw
	case "qcow2":
		format = vm.StorageFormatQCOW2
	case "vmdk":
		format = vm.StorageFormatVMDK
	case "vhd":
		format = vm.StorageFormatVHD
	default:
		http.Error(w, "Invalid storage format. Must be 'raw', 'qcow2', 'vmdk', or 'vhd'.", http.StatusBadRequest)
		return
	}

	// Create volume
	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()

	volume, err := h.storageManager.CreateStorageVolume(ctx, request.Name, request.PoolID, format, request.Capacity, request.Tags, request.Metadata)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Write response
	response := map[string]interface{}{
		"id":         volume.ID,
		"name":       volume.Name,
		"pool_id":    volume.PoolID,
		"format":     volume.Format,
		"capacity":   volume.Capacity,
		"allocation": volume.Allocation,
		"path":       volume.Path,
		"created_at": volume.CreatedAt,
		"updated_at": volume.UpdatedAt,
		"tags":       volume.Tags,
		"metadata":   volume.Metadata,
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(response)
}

// GetStorageVolume handles GET /storage/volumes/{id}
func (h *StorageHandler) GetStorageVolume(w http.ResponseWriter, r *http.Request) {
	// Get volume ID from URL
	vars := mux.Vars(r)
	volumeID := vars["id"]

	// Get volume
	volume, err := h.storageManager.GetStorageVolume(volumeID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}

	// Write response
	response := map[string]interface{}{
		"id":         volume.ID,
		"name":       volume.Name,
		"pool_id":    volume.PoolID,
		"format":     volume.Format,
		"capacity":   volume.Capacity,
		"allocation": volume.Allocation,
		"path":       volume.Path,
		"created_at": volume.CreatedAt,
		"updated_at": volume.UpdatedAt,
		"tags":       volume.Tags,
		"metadata":   volume.Metadata,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// DeleteStorageVolume handles DELETE /storage/volumes/{id}
func (h *StorageHandler) DeleteStorageVolume(w http.ResponseWriter, r *http.Request) {
	// Get volume ID from URL
	vars := mux.Vars(r)
	volumeID := vars["id"]

	// Delete volume
	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()

	if err := h.storageManager.DeleteStorageVolume(ctx, volumeID); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Write response
	w.WriteHeader(http.StatusNoContent)
}

// ResizeStorageVolume handles POST /storage/volumes/{id}/resize
func (h *StorageHandler) ResizeStorageVolume(w http.ResponseWriter, r *http.Request) {
	// Get volume ID from URL
	vars := mux.Vars(r)
	volumeID := vars["id"]

	// Parse request
	var request struct {
		NewCapacity int64 `json:"new_capacity"`
	}

	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Resize volume
	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()

	volume, err := h.storageManager.ResizeStorageVolume(ctx, volumeID, request.NewCapacity)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Write response
	response := map[string]interface{}{
		"id":         volume.ID,
		"name":       volume.Name,
		"pool_id":    volume.PoolID,
		"format":     volume.Format,
		"capacity":   volume.Capacity,
		"allocation": volume.Allocation,
		"path":       volume.Path,
		"created_at": volume.CreatedAt,
		"updated_at": volume.UpdatedAt,
		"tags":       volume.Tags,
		"metadata":   volume.Metadata,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// CloneStorageVolume handles POST /storage/volumes/{id}/clone
func (h *StorageHandler) CloneStorageVolume(w http.ResponseWriter, r *http.Request) {
	// Get volume ID from URL
	vars := mux.Vars(r)
	volumeID := vars["id"]

	// Parse request
	var request struct {
		Name     string            `json:"name"`
		Tags     []string          `json:"tags"`
		Metadata map[string]string `json:"metadata"`
	}

	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Clone volume
	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()

	volume, err := h.storageManager.CloneStorageVolume(ctx, volumeID, request.Name, request.Tags, request.Metadata)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Write response
	response := map[string]interface{}{
		"id":         volume.ID,
		"name":       volume.Name,
		"pool_id":    volume.PoolID,
		"format":     volume.Format,
		"capacity":   volume.Capacity,
		"allocation": volume.Allocation,
		"path":       volume.Path,
		"created_at": volume.CreatedAt,
		"updated_at": volume.UpdatedAt,
		"tags":       volume.Tags,
		"metadata":   volume.Metadata,
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(response)
}
