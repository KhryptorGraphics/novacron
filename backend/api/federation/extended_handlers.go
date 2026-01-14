package federation

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"time"

	"github.com/gorilla/mux"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/khryptorgraphics/novacron/backend/core/federation"
)

var (
	federationOperationDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "novacron_federation_operation_duration_seconds",
		Help:    "Duration of federation operations",
		Buckets: prometheus.DefBuckets,
	}, []string{"operation"})

	federationOperationCount = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "novacron_federation_operations_total",
		Help: "Total number of federation operations",
	}, []string{"operation", "status"})

	activeNodesGauge = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "novacron_federation_active_nodes",
		Help: "Number of active federation nodes",
	}, []string{"cluster"})
)

// ExtendedHandler handles advanced federation API requests
type ExtendedHandler struct {
	manager         federation.FederationManager
	nodeManager     federation.NodeManager
	clusterManager  federation.ClusterManager
	healthChecker   federation.HealthChecker
}

// NewExtendedHandler creates a new extended federation API handler
func NewExtendedHandler(manager federation.FederationManager, nodeManager federation.NodeManager, clusterManager federation.ClusterManager, healthChecker federation.HealthChecker) *ExtendedHandler {
	return &ExtendedHandler{
		manager:         manager,
		nodeManager:     nodeManager,
		clusterManager:  clusterManager,
		healthChecker:   healthChecker,
	}
}

// FederationNode represents a federation node
type FederationNode struct {
	ID          string                 `json:"id"`
	Name        string                 `json:"name"`
	Address     string                 `json:"address"`
	Port        int                    `json:"port"`
	Status      string                 `json:"status"`
	Role        string                 `json:"role"`        // leader, follower, candidate
	Region      string                 `json:"region"`
	Zone        string                 `json:"zone"`
	Capacity    NodeCapacity           `json:"capacity"`
	Usage       NodeUsage              `json:"usage"`
	LastSeen    time.Time              `json:"last_seen"`
	JoinedAt    time.Time              `json:"joined_at"`
	Version     string                 `json:"version"`
	Tags        map[string]string      `json:"tags"`
	Metadata    map[string]interface{} `json:"metadata"`
}

// NodeCapacity represents node resource capacity
type NodeCapacity struct {
	CPU        int   `json:"cpu"`        // cores
	Memory     int64 `json:"memory"`     // bytes
	Storage    int64 `json:"storage"`    // bytes
	Network    int64 `json:"network"`    // bytes/sec
	MaxVMs     int   `json:"max_vms"`
}

// NodeUsage represents current node resource usage
type NodeUsage struct {
	CPU        float64 `json:"cpu"`         // percentage
	Memory     float64 `json:"memory"`      // percentage
	Storage    float64 `json:"storage"`     // percentage
	Network    float64 `json:"network"`     // percentage
	ActiveVMs  int     `json:"active_vms"`
	TotalVMs   int     `json:"total_vms"`
}

// JoinRequest represents federation join request
type JoinRequest struct {
	NodeName    string            `json:"node_name" validate:"required,min=1,max=64"`
	Address     string            `json:"address" validate:"required,ip"`
	Port        int               `json:"port" validate:"required,min=1,max=65535"`
	Region      string            `json:"region,omitempty"`
	Zone        string            `json:"zone,omitempty"`
	Capacity    NodeCapacity      `json:"capacity" validate:"required"`
	Tags        map[string]string `json:"tags,omitempty"`
	AuthToken   string            `json:"auth_token" validate:"required"`
}

// LeaveRequest represents federation leave request
type LeaveRequest struct {
	NodeID    string `json:"node_id,omitempty"`
	Force     bool   `json:"force,omitempty"`
	Graceful  bool   `json:"graceful,omitempty"`
	Timeout   int    `json:"timeout,omitempty"` // seconds
}

// FederationHealthStatus represents overall federation health
type FederationHealthStatus struct {
	Status          string                    `json:"status"`         // healthy, degraded, unhealthy
	TotalNodes      int                       `json:"total_nodes"`
	HealthyNodes    int                       `json:"healthy_nodes"`
	UnhealthyNodes  int                       `json:"unhealthy_nodes"`
	LeaderNode      *FederationNode           `json:"leader_node,omitempty"`
	ClusterInfo     ClusterInfo               `json:"cluster_info"`
	NetworkLatency  map[string]time.Duration  `json:"network_latency"`
	LastElection    *time.Time                `json:"last_election,omitempty"`
	Issues          []HealthIssue             `json:"issues,omitempty"`
	Recommendations []string                  `json:"recommendations,omitempty"`
}

// ClusterInfo represents cluster-level information
type ClusterInfo struct {
	ClusterID       string    `json:"cluster_id"`
	Version         string    `json:"version"`
	CreatedAt       time.Time `json:"created_at"`
	TotalCapacity   NodeCapacity `json:"total_capacity"`
	TotalUsage      NodeUsage    `json:"total_usage"`
	ReplicationMode string    `json:"replication_mode"`
	ConsensusAlgo   string    `json:"consensus_algorithm"`
}

// HealthIssue represents a federation health issue
type HealthIssue struct {
	Type        string    `json:"type"`        // network, consensus, resource, security
	Severity    string    `json:"severity"`    // low, medium, high, critical
	Description string    `json:"description"`
	NodeID      string    `json:"node_id,omitempty"`
	Timestamp   time.Time `json:"timestamp"`
	Resolution  string    `json:"resolution,omitempty"`
}

// RegisterExtendedRoutes registers extended federation API routes
func (h *ExtendedHandler) RegisterExtendedRoutes(router *mux.Router, require func(string, http.HandlerFunc) http.Handler) {
	federationRouter := router.PathPrefix("/api/v1/federation").Subrouter()

	// Node management (viewer+ for list, operator+ for join/leave)
	federationRouter.Handle("/nodes", require("viewer", h.ListNodes)).Methods("GET")
	federationRouter.Handle("/nodes/{id}", require("viewer", h.GetNode)).Methods("GET")
	federationRouter.Handle("/join", require("operator", h.JoinFederation)).Methods("POST")
	federationRouter.Handle("/leave", require("operator", h.LeaveFederation)).Methods("POST")
	
	// Health monitoring (viewer+)
	federationRouter.Handle("/health", require("viewer", h.GetFederationHealth)).Methods("GET")
	federationRouter.Handle("/health/detailed", require("admin", h.GetDetailedHealth)).Methods("GET")
	
	// Cluster management (admin only)
	federationRouter.Handle("/cluster/info", require("viewer", h.GetClusterInfo)).Methods("GET")
	federationRouter.Handle("/cluster/rebalance", require("admin", h.RebalanceCluster)).Methods("POST")
	federationRouter.Handle("/cluster/maintenance", require("admin", h.ToggleMaintenanceMode)).Methods("POST")
}

// ListNodes handles GET /api/v1/federation/nodes
// @Summary List federation nodes
// @Description List all nodes in the federation with their status and capacity
// @Tags Federation
// @Produce json
// @Param page query int false "Page number" default(1)
// @Param limit query int false "Items per page" default(20)
// @Param status query string false "Filter by status" Enums(active,inactive,joining,leaving)
// @Param region query string false "Filter by region"
// @Param role query string false "Filter by role" Enums(leader,follower,candidate)
// @Success 200 {object} map[string]interface{}
// @Failure 400 {object} map[string]interface{} "Invalid request"
// @Failure 401 {object} map[string]interface{} "Unauthorized"
// @Failure 500 {object} map[string]interface{} "Internal server error"
// @Router /api/v1/federation/nodes [get]
func (h *ExtendedHandler) ListNodes(w http.ResponseWriter, r *http.Request) {
	timer := prometheus.NewTimer(federationOperationDuration.WithLabelValues("list_nodes"))
	defer timer.ObserveDuration()

	// Parse query parameters
	page := 1
	limit := 20

	if pageStr := r.URL.Query().Get("page"); pageStr != "" {
		if p, err := strconv.Atoi(pageStr); err == nil && p > 0 {
			page = p
		}
	}

	if limitStr := r.URL.Query().Get("limit"); limitStr != "" {
		if l, err := strconv.Atoi(limitStr); err == nil && l > 0 && l <= 100 {
			limit = l
		}
	}

	filters := federation.NodeFilters{
		Status: r.URL.Query().Get("status"),
		Region: r.URL.Query().Get("region"),
		Role:   r.URL.Query().Get("role"),
	}

	// Get nodes from federation manager
	nodes, total, err := h.nodeManager.ListNodes(r.Context(), federation.ListNodesRequest{
		Page:    page,
		Limit:   limit,
		Filters: filters,
	})
	if err != nil {
		federationOperationCount.WithLabelValues("list_nodes", "error").Inc()
		writeError(w, http.StatusInternalServerError, "LIST_ERROR", "Failed to list federation nodes")
		return
	}

	// Convert to API response format
	apiNodes := make([]FederationNode, len(nodes))
	for i, node := range nodes {
		apiNodes[i] = h.convertToAPINode(node)
	}

	// Update metrics
	activeCount := 0
	for _, node := range apiNodes {
		if node.Status == "active" {
			activeCount++
		}
	}
	activeNodesGauge.WithLabelValues("default").Set(float64(activeCount))

	federationOperationCount.WithLabelValues("list_nodes", "success").Inc()

	response := map[string]interface{}{
		"nodes": apiNodes,
		"pagination": map[string]interface{}{
			"page":        page,
			"limit":       limit,
			"total":       total,
			"total_pages": (total + limit - 1) / limit,
		},
		"summary": map[string]interface{}{
			"total_nodes":   total,
			"active_nodes":  activeCount,
			"regions":       h.getUniqueRegions(apiNodes),
		},
	}

	writeJSON(w, http.StatusOK, response)
}

// GetNode handles GET /api/v1/federation/nodes/{id}
// @Summary Get federation node details
// @Description Get detailed information about a specific federation node
// @Tags Federation
// @Produce json
// @Param id path string true "Node ID"
// @Success 200 {object} FederationNode
// @Failure 400 {object} map[string]interface{} "Invalid request"
// @Failure 401 {object} map[string]interface{} "Unauthorized"
// @Failure 404 {object} map[string]interface{} "Node not found"
// @Failure 500 {object} map[string]interface{} "Internal server error"
// @Router /api/v1/federation/nodes/{id} [get]
func (h *ExtendedHandler) GetNode(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	nodeID := vars["id"]

	if nodeID == "" {
		writeError(w, http.StatusBadRequest, "MISSING_NODE_ID", "Node ID is required")
		return
	}

	node, err := h.nodeManager.GetNode(r.Context(), nodeID)
	if err != nil {
		if err == federation.ErrNodeNotFound {
			writeError(w, http.StatusNotFound, "NODE_NOT_FOUND", "Federation node not found")
			return
		}
		writeError(w, http.StatusInternalServerError, "INTERNAL_ERROR", "Failed to retrieve node information")
		return
	}

	apiNode := h.convertToAPINode(node)
	writeJSON(w, http.StatusOK, apiNode)
}

// JoinFederation handles POST /api/v1/federation/join
// @Summary Join federation cluster
// @Description Join the federation cluster as a new node
// @Tags Federation
// @Accept json
// @Produce json
// @Param request body JoinRequest true "Join request"
// @Success 202 {object} map[string]interface{}
// @Failure 400 {object} map[string]interface{} "Invalid request"
// @Failure 401 {object} map[string]interface{} "Unauthorized"
// @Failure 409 {object} map[string]interface{} "Node already exists"
// @Failure 500 {object} map[string]interface{} "Internal server error"
// @Router /api/v1/federation/join [post]
func (h *ExtendedHandler) JoinFederation(w http.ResponseWriter, r *http.Request) {
	timer := prometheus.NewTimer(federationOperationDuration.WithLabelValues("join"))
	defer timer.ObserveDuration()

	var req JoinRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		federationOperationCount.WithLabelValues("join", "error").Inc()
		writeError(w, http.StatusBadRequest, "INVALID_JSON", "Invalid JSON format")
		return
	}

	// Validate request
	if err := h.validateJoinRequest(req); err != nil {
		federationOperationCount.WithLabelValues("join", "error").Inc()
		writeError(w, http.StatusBadRequest, "VALIDATION_ERROR", err.Error())
		return
	}

	// Check if node already exists
	if exists, _ := h.nodeManager.NodeExists(r.Context(), req.NodeName); exists {
		federationOperationCount.WithLabelValues("join", "conflict").Inc()
		writeError(w, http.StatusConflict, "NODE_EXISTS", "A node with this name already exists")
		return
	}

	// Validate authentication token
	if !h.validateAuthToken(req.AuthToken) {
		federationOperationCount.WithLabelValues("join", "unauthorized").Inc()
		writeError(w, http.StatusUnauthorized, "INVALID_TOKEN", "Invalid authentication token")
		return
	}

	// Initiate join process
	joinRequest := federation.JoinRequest{
		NodeName:  req.NodeName,
		Address:   req.Address,
		Port:      req.Port,
		Region:    req.Region,
		Zone:      req.Zone,
		Capacity:  h.convertCapacity(req.Capacity),
		Tags:      req.Tags,
		AuthToken: req.AuthToken,
	}

	nodeID, err := h.clusterManager.JoinCluster(r.Context(), joinRequest)
	if err != nil {
		federationOperationCount.WithLabelValues("join", "error").Inc()
		writeError(w, http.StatusInternalServerError, "JOIN_ERROR", "Failed to join federation cluster")
		return
	}

	federationOperationCount.WithLabelValues("join", "success").Inc()

	writeJSON(w, http.StatusAccepted, map[string]interface{}{
		"node_id":    nodeID,
		"node_name":  req.NodeName,
		"status":     "joining",
		"message":    "Federation join process initiated",
		"started_at": time.Now(),
	})
}

// LeaveFederation handles POST /api/v1/federation/leave
// @Summary Leave federation cluster
// @Description Leave the federation cluster gracefully or forcefully
// @Tags Federation
// @Accept json
// @Produce json
// @Param request body LeaveRequest true "Leave request"
// @Success 202 {object} map[string]interface{}
// @Failure 400 {object} map[string]interface{} "Invalid request"
// @Failure 401 {object} map[string]interface{} "Unauthorized"
// @Failure 404 {object} map[string]interface{} "Node not found"
// @Failure 500 {object} map[string]interface{} "Internal server error"
// @Router /api/v1/federation/leave [post]
func (h *ExtendedHandler) LeaveFederation(w http.ResponseWriter, r *http.Request) {
	timer := prometheus.NewTimer(federationOperationDuration.WithLabelValues("leave"))
	defer timer.ObserveDuration()

	var req LeaveRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		federationOperationCount.WithLabelValues("leave", "error").Inc()
		writeError(w, http.StatusBadRequest, "INVALID_JSON", "Invalid JSON format")
		return
	}

	// Default to current node if not specified
	if req.NodeID == "" {
		req.NodeID = h.getCurrentNodeID()
	}

	// Default timeout
	if req.Timeout == 0 {
		req.Timeout = 300 // 5 minutes
	}

	// Validate node exists
	node, err := h.nodeManager.GetNode(r.Context(), req.NodeID)
	if err != nil {
		if err == federation.ErrNodeNotFound {
			federationOperationCount.WithLabelValues("leave", "not_found").Inc()
			writeError(w, http.StatusNotFound, "NODE_NOT_FOUND", "Node not found")
			return
		}
		federationOperationCount.WithLabelValues("leave", "error").Inc()
		writeError(w, http.StatusInternalServerError, "INTERNAL_ERROR", "Failed to retrieve node information")
		return
	}

	// Check if this is the last node or leader
	if node.Role == "leader" && !req.Force {
		// Check cluster size
		nodes, _, _ := h.nodeManager.ListNodes(r.Context(), federation.ListNodesRequest{})
		if len(nodes) > 1 {
			federationOperationCount.WithLabelValues("leave", "error").Inc()
			writeError(w, http.StatusConflict, "LEADER_CANNOT_LEAVE", "Leader node cannot leave without force flag when other nodes exist")
			return
		}
	}

	// Initiate leave process
	leaveRequest := federation.LeaveRequest{
		NodeID:   req.NodeID,
		Force:    req.Force,
		Graceful: req.Graceful,
		Timeout:  time.Duration(req.Timeout) * time.Second,
	}

	if err := h.clusterManager.LeaveCluster(r.Context(), leaveRequest); err != nil {
		federationOperationCount.WithLabelValues("leave", "error").Inc()
		writeError(w, http.StatusInternalServerError, "LEAVE_ERROR", "Failed to leave federation cluster")
		return
	}

	federationOperationCount.WithLabelValues("leave", "success").Inc()

	writeJSON(w, http.StatusAccepted, map[string]interface{}{
		"node_id":    req.NodeID,
		"node_name":  node.Name,
		"status":     "leaving",
		"graceful":   req.Graceful,
		"timeout":    req.Timeout,
		"message":    "Federation leave process initiated",
		"started_at": time.Now(),
	})
}

// GetFederationHealth handles GET /api/v1/federation/health
// @Summary Get federation health status
// @Description Get overall federation cluster health information
// @Tags Federation
// @Produce json
// @Success 200 {object} FederationHealthStatus
// @Failure 401 {object} map[string]interface{} "Unauthorized"
// @Failure 500 {object} map[string]interface{} "Internal server error"
// @Router /api/v1/federation/health [get]
func (h *ExtendedHandler) GetFederationHealth(w http.ResponseWriter, r *http.Request) {
	health, err := h.healthChecker.GetClusterHealth(r.Context())
	if err != nil {
		writeError(w, http.StatusInternalServerError, "HEALTH_ERROR", "Failed to retrieve federation health")
		return
	}

	apiHealth := h.convertToAPIHealth(health)
	writeJSON(w, http.StatusOK, apiHealth)
}

// GetDetailedHealth handles GET /api/v1/federation/health/detailed
// @Summary Get detailed federation health status
// @Description Get comprehensive federation health information including diagnostics
// @Tags Federation
// @Produce json
// @Success 200 {object} map[string]interface{}
// @Failure 401 {object} map[string]interface{} "Unauthorized"
// @Failure 500 {object} map[string]interface{} "Internal server error"
// @Router /api/v1/federation/health/detailed [get]
func (h *ExtendedHandler) GetDetailedHealth(w http.ResponseWriter, r *http.Request) {
	detailedHealth, err := h.healthChecker.GetDetailedHealth(r.Context())
	if err != nil {
		writeError(w, http.StatusInternalServerError, "HEALTH_ERROR", "Failed to retrieve detailed health information")
		return
	}

	writeJSON(w, http.StatusOK, detailedHealth)
}

// GetClusterInfo handles GET /api/v1/federation/cluster/info
// @Summary Get cluster information
// @Description Get general information about the federation cluster
// @Tags Federation
// @Produce json
// @Success 200 {object} ClusterInfo
// @Failure 401 {object} map[string]interface{} "Unauthorized"
// @Failure 500 {object} map[string]interface{} "Internal server error"
// @Router /api/v1/federation/cluster/info [get]
func (h *ExtendedHandler) GetClusterInfo(w http.ResponseWriter, r *http.Request) {
	info, err := h.clusterManager.GetClusterInfo(r.Context())
	if err != nil {
		writeError(w, http.StatusInternalServerError, "INFO_ERROR", "Failed to retrieve cluster information")
		return
	}

	apiInfo := h.convertToAPIClusterInfo(info)
	writeJSON(w, http.StatusOK, apiInfo)
}

// RebalanceCluster handles POST /api/v1/federation/cluster/rebalance
// @Summary Rebalance federation cluster
// @Description Trigger cluster rebalancing to optimize resource distribution
// @Tags Federation
// @Produce json
// @Success 202 {object} map[string]interface{}
// @Failure 401 {object} map[string]interface{} "Unauthorized"
// @Failure 500 {object} map[string]interface{} "Internal server error"
// @Router /api/v1/federation/cluster/rebalance [post]
func (h *ExtendedHandler) RebalanceCluster(w http.ResponseWriter, r *http.Request) {
	rebalanceID, err := h.clusterManager.TriggerRebalance(r.Context())
	if err != nil {
		writeError(w, http.StatusInternalServerError, "REBALANCE_ERROR", "Failed to trigger cluster rebalancing")
		return
	}

	writeJSON(w, http.StatusAccepted, map[string]interface{}{
		"rebalance_id": rebalanceID,
		"status":       "started",
		"message":      "Cluster rebalancing initiated",
		"started_at":   time.Now(),
	})
}

// ToggleMaintenanceMode handles POST /api/v1/federation/cluster/maintenance
// @Summary Toggle maintenance mode
// @Description Enable or disable cluster maintenance mode
// @Tags Federation
// @Accept json
// @Produce json
// @Param request body map[string]interface{} true "Maintenance mode request"
// @Success 200 {object} map[string]interface{}
// @Failure 400 {object} map[string]interface{} "Invalid request"
// @Failure 401 {object} map[string]interface{} "Unauthorized"
// @Failure 500 {object} map[string]interface{} "Internal server error"
// @Router /api/v1/federation/cluster/maintenance [post]
func (h *ExtendedHandler) ToggleMaintenanceMode(w http.ResponseWriter, r *http.Request) {
	var req map[string]interface{}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "INVALID_JSON", "Invalid JSON format")
		return
	}

	enabled, ok := req["enabled"].(bool)
	if !ok {
		writeError(w, http.StatusBadRequest, "MISSING_ENABLED", "enabled field is required")
		return
	}

	reason := ""
	if r, ok := req["reason"].(string); ok {
		reason = r
	}

	if err := h.clusterManager.SetMaintenanceMode(r.Context(), enabled, reason); err != nil {
		writeError(w, http.StatusInternalServerError, "MAINTENANCE_ERROR", "Failed to toggle maintenance mode")
		return
	}

	status := "disabled"
	if enabled {
		status = "enabled"
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"maintenance_mode": status,
		"reason":          reason,
		"timestamp":       time.Now(),
	})
}

// Helper functions

func (h *ExtendedHandler) convertToAPINode(node federation.Node) FederationNode {
	return FederationNode{
		ID:       node.ID,
		Name:     node.Name,
		Address:  node.Address,
		Port:     node.Port,
		Status:   node.Status,
		Role:     node.Role,
		Region:   node.Region,
		Zone:     node.Zone,
		Capacity: NodeCapacity{
			CPU:        node.Capacity.CPU,
			Memory:     node.Capacity.Memory,
			Storage:    node.Capacity.Storage,
			Network:    node.Capacity.Network,
			MaxVMs:     node.Capacity.MaxVMs,
		},
		Usage: NodeUsage{
			CPU:       node.Usage.CPU,
			Memory:    node.Usage.Memory,
			Storage:   node.Usage.Storage,
			Network:   node.Usage.Network,
			ActiveVMs: node.Usage.ActiveVMs,
			TotalVMs:  node.Usage.TotalVMs,
		},
		LastSeen:  node.LastSeen,
		JoinedAt:  node.JoinedAt,
		Version:   node.Version,
		Tags:      node.Tags,
		Metadata:  node.Metadata,
	}
}

func (h *ExtendedHandler) convertToAPIHealth(health federation.ClusterHealth) FederationHealthStatus {
	apiNodes := make([]FederationNode, len(health.Nodes))
	for i, node := range health.Nodes {
		apiNodes[i] = h.convertToAPINode(node)
	}

	var leaderNode *FederationNode
	for _, node := range apiNodes {
		if node.Role == "leader" {
			leaderNode = &node
			break
		}
	}

	return FederationHealthStatus{
		Status:         health.Status,
		TotalNodes:     health.TotalNodes,
		HealthyNodes:   health.HealthyNodes,
		UnhealthyNodes: health.UnhealthyNodes,
		LeaderNode:     leaderNode,
		ClusterInfo:    h.convertToAPIClusterInfo(health.ClusterInfo),
		NetworkLatency: health.NetworkLatency,
		LastElection:   health.LastElection,
		Issues:         h.convertHealthIssues(health.Issues),
		Recommendations: health.Recommendations,
	}
}

func (h *ExtendedHandler) convertToAPIClusterInfo(info federation.ClusterInfo) ClusterInfo {
	return ClusterInfo{
		ClusterID:       info.ClusterID,
		Version:         info.Version,
		CreatedAt:       info.CreatedAt,
		TotalCapacity:   NodeCapacity{
			CPU:     info.TotalCapacity.CPU,
			Memory:  info.TotalCapacity.Memory,
			Storage: info.TotalCapacity.Storage,
			Network: info.TotalCapacity.Network,
			MaxVMs:  info.TotalCapacity.MaxVMs,
		},
		TotalUsage: NodeUsage{
			CPU:       info.TotalUsage.CPU,
			Memory:    info.TotalUsage.Memory,
			Storage:   info.TotalUsage.Storage,
			Network:   info.TotalUsage.Network,
			ActiveVMs: info.TotalUsage.ActiveVMs,
			TotalVMs:  info.TotalUsage.TotalVMs,
		},
		ReplicationMode: info.ReplicationMode,
		ConsensusAlgo:   info.ConsensusAlgorithm,
	}
}

func (h *ExtendedHandler) convertHealthIssues(issues []federation.HealthIssue) []HealthIssue {
	apiIssues := make([]HealthIssue, len(issues))
	for i, issue := range issues {
		apiIssues[i] = HealthIssue{
			Type:        issue.Type,
			Severity:    issue.Severity,
			Description: issue.Description,
			NodeID:      issue.NodeID,
			Timestamp:   issue.Timestamp,
			Resolution:  issue.Resolution,
		}
	}
	return apiIssues
}

func (h *ExtendedHandler) validateJoinRequest(req JoinRequest) error {
	if req.NodeName == "" {
		return fmt.Errorf("node name is required")
	}
	if req.Address == "" {
		return fmt.Errorf("address is required")
	}
	if req.Port <= 0 || req.Port > 65535 {
		return fmt.Errorf("port must be between 1 and 65535")
	}
	if req.AuthToken == "" {
		return fmt.Errorf("authentication token is required")
	}
	if req.Capacity.CPU <= 0 {
		return fmt.Errorf("CPU capacity must be positive")
	}
	if req.Capacity.Memory <= 0 {
		return fmt.Errorf("memory capacity must be positive")
	}
	return nil
}

func (h *ExtendedHandler) validateAuthToken(token string) bool {
	// Implementation would validate the authentication token
	// This is a placeholder
	return len(token) > 10
}

func (h *ExtendedHandler) convertCapacity(capacity NodeCapacity) federation.NodeCapacity {
	return federation.NodeCapacity{
		CPU:     capacity.CPU,
		Memory:  capacity.Memory,
		Storage: capacity.Storage,
		Network: capacity.Network,
		MaxVMs:  capacity.MaxVMs,
	}
}

func (h *ExtendedHandler) getCurrentNodeID() string {
	// Implementation would return the current node's ID
	return "current-node-id"
}

func (h *ExtendedHandler) getUniqueRegions(nodes []FederationNode) []string {
	regionMap := make(map[string]bool)
	for _, node := range nodes {
		if node.Region != "" {
			regionMap[node.Region] = true
		}
	}
	
	regions := make([]string, 0, len(regionMap))
	for region := range regionMap {
		regions = append(regions, region)
	}
	return regions
}