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

// ClusterHandler handles VM cluster API requests
type ClusterHandler struct {
	clusterManager *vm.VMClusterManager
}

// NewClusterHandler creates a new VM cluster API handler
func NewClusterHandler(clusterManager *vm.VMClusterManager) *ClusterHandler {
	return &ClusterHandler{
		clusterManager: clusterManager,
	}
}

// RegisterRoutes registers VM cluster API routes
func (h *ClusterHandler) RegisterRoutes(router *mux.Router) {
	router.HandleFunc("/clusters", h.ListClusters).Methods("GET")
	router.HandleFunc("/clusters", h.CreateCluster).Methods("POST")
	router.HandleFunc("/clusters/{id}", h.GetCluster).Methods("GET")
	router.HandleFunc("/clusters/{id}", h.DeleteCluster).Methods("DELETE")
	router.HandleFunc("/clusters/{id}/members", h.AddClusterMember).Methods("POST")
	router.HandleFunc("/clusters/{id}/members/{vm_id}", h.RemoveClusterMember).Methods("DELETE")
	router.HandleFunc("/clusters/{id}/start", h.StartCluster).Methods("POST")
	router.HandleFunc("/clusters/{id}/stop", h.StopCluster).Methods("POST")
}

// ListClusters handles GET /clusters
func (h *ClusterHandler) ListClusters(w http.ResponseWriter, r *http.Request) {
	// Get clusters
	clusters := h.clusterManager.ListClusters()

	// Convert to response format
	response := make([]map[string]interface{}, 0, len(clusters))
	for _, cluster := range clusters {
		response = append(response, map[string]interface{}{
			"id":          cluster.ID,
			"name":        cluster.Name,
			"description": cluster.Description,
			"state":       cluster.State,
			"created_at":  cluster.CreatedAt,
			"updated_at":  cluster.UpdatedAt,
			"member_count": len(cluster.Members),
			"tags":        cluster.Tags,
			"metadata":    cluster.Metadata,
		})
	}

	// Write response
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// CreateCluster handles POST /clusters
func (h *ClusterHandler) CreateCluster(w http.ResponseWriter, r *http.Request) {
	// Parse request
	var request struct {
		Name        string            `json:"name"`
		Description string            `json:"description"`
		MasterCount int               `json:"master_count"`
		WorkerCount int               `json:"worker_count"`
		VMConfig    vm.VMConfig       `json:"vm_config"`
		Tags        []string          `json:"tags"`
		Metadata    map[string]string `json:"metadata"`
	}

	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Create cluster
	ctx, cancel := context.WithTimeout(r.Context(), 60*time.Second)
	defer cancel()

	cluster, err := h.clusterManager.CreateCluster(ctx, request.Name, request.Description, request.MasterCount, request.WorkerCount, request.VMConfig, request.Tags, request.Metadata)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Write response
	response := map[string]interface{}{
		"id":          cluster.ID,
		"name":        cluster.Name,
		"description": cluster.Description,
		"state":       cluster.State,
		"created_at":  cluster.CreatedAt,
		"updated_at":  cluster.UpdatedAt,
		"members":     cluster.Members,
		"tags":        cluster.Tags,
		"metadata":    cluster.Metadata,
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(response)
}

// GetCluster handles GET /clusters/{id}
func (h *ClusterHandler) GetCluster(w http.ResponseWriter, r *http.Request) {
	// Get cluster ID from URL
	vars := mux.Vars(r)
	clusterID := vars["id"]

	// Get cluster
	cluster, err := h.clusterManager.GetCluster(clusterID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}

	// Write response
	response := map[string]interface{}{
		"id":          cluster.ID,
		"name":        cluster.Name,
		"description": cluster.Description,
		"state":       cluster.State,
		"created_at":  cluster.CreatedAt,
		"updated_at":  cluster.UpdatedAt,
		"members":     cluster.Members,
		"tags":        cluster.Tags,
		"metadata":    cluster.Metadata,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// DeleteCluster handles DELETE /clusters/{id}
func (h *ClusterHandler) DeleteCluster(w http.ResponseWriter, r *http.Request) {
	// Get cluster ID from URL
	vars := mux.Vars(r)
	clusterID := vars["id"]

	// Delete cluster
	ctx, cancel := context.WithTimeout(r.Context(), 60*time.Second)
	defer cancel()

	if err := h.clusterManager.DeleteCluster(ctx, clusterID); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Write response
	w.WriteHeader(http.StatusNoContent)
}

// AddClusterMember handles POST /clusters/{id}/members
func (h *ClusterHandler) AddClusterMember(w http.ResponseWriter, r *http.Request) {
	// Get cluster ID from URL
	vars := mux.Vars(r)
	clusterID := vars["id"]

	// Parse request
	var request struct {
		Role     string      `json:"role"`
		VMConfig vm.VMConfig `json:"vm_config"`
	}

	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Validate role
	var role vm.ClusterRole
	switch request.Role {
	case "master":
		role = vm.ClusterRoleMaster
	case "worker":
		role = vm.ClusterRoleWorker
	case "storage":
		role = vm.ClusterRoleStorage
	default:
		http.Error(w, "Invalid role. Must be 'master', 'worker', or 'storage'.", http.StatusBadRequest)
		return
	}

	// Add member
	ctx, cancel := context.WithTimeout(r.Context(), 60*time.Second)
	defer cancel()

	member, err := h.clusterManager.AddClusterMember(ctx, clusterID, role, request.VMConfig)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Write response
	response := map[string]interface{}{
		"vm_id":     member.VMID,
		"role":      member.Role,
		"joined_at": member.JoinedAt,
		"status":    member.Status,
		"node_id":   member.NodeID,
		"ip_address": member.IPAddress,
		"hostname":  member.Hostname,
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(response)
}

// RemoveClusterMember handles DELETE /clusters/{id}/members/{vm_id}
func (h *ClusterHandler) RemoveClusterMember(w http.ResponseWriter, r *http.Request) {
	// Get cluster ID and VM ID from URL
	vars := mux.Vars(r)
	clusterID := vars["id"]
	vmID := vars["vm_id"]

	// Remove member
	ctx, cancel := context.WithTimeout(r.Context(), 60*time.Second)
	defer cancel()

	if err := h.clusterManager.RemoveClusterMember(ctx, clusterID, vmID); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Write response
	w.WriteHeader(http.StatusNoContent)
}

// StartCluster handles POST /clusters/{id}/start
func (h *ClusterHandler) StartCluster(w http.ResponseWriter, r *http.Request) {
	// Get cluster ID from URL
	vars := mux.Vars(r)
	clusterID := vars["id"]

	// Start cluster
	ctx, cancel := context.WithTimeout(r.Context(), 60*time.Second)
	defer cancel()

	if err := h.clusterManager.StartCluster(ctx, clusterID); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Get updated cluster
	cluster, err := h.clusterManager.GetCluster(clusterID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Write response
	response := map[string]interface{}{
		"id":          cluster.ID,
		"name":        cluster.Name,
		"description": cluster.Description,
		"state":       cluster.State,
		"created_at":  cluster.CreatedAt,
		"updated_at":  cluster.UpdatedAt,
		"members":     cluster.Members,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// StopCluster handles POST /clusters/{id}/stop
func (h *ClusterHandler) StopCluster(w http.ResponseWriter, r *http.Request) {
	// Get cluster ID from URL
	vars := mux.Vars(r)
	clusterID := vars["id"]

	// Stop cluster
	ctx, cancel := context.WithTimeout(r.Context(), 60*time.Second)
	defer cancel()

	if err := h.clusterManager.StopCluster(ctx, clusterID); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Get updated cluster
	cluster, err := h.clusterManager.GetCluster(clusterID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Write response
	response := map[string]interface{}{
		"id":          cluster.ID,
		"name":        cluster.Name,
		"description": cluster.Description,
		"state":       cluster.State,
		"created_at":  cluster.CreatedAt,
		"updated_at":  cluster.UpdatedAt,
		"members":     cluster.Members,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}
