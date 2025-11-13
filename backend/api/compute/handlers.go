package api

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strconv"
	"time"

	"github.com/gorilla/mux"
	compcore "github.com/khryptorgraphics/novacron/backend/core/compute"
	sched "github.com/khryptorgraphics/novacron/backend/core/scheduler"
)

// ComputeAPIHandler handles HTTP requests for compute job management
type ComputeAPIHandler struct {
	jobManager   *compcore.ComputeJobManager
	loadBalancer *compcore.ComputeJobLoadBalancer
	scheduler    *sched.Scheduler
	ctx          context.Context
}

// NewComputeAPIHandler creates a new compute API handler
func NewComputeAPIHandler(jobManager *compcore.ComputeJobManager, loadBalancer *compcore.ComputeJobLoadBalancer, scheduler *sched.Scheduler) *ComputeAPIHandler {
	return &ComputeAPIHandler{
		jobManager:   jobManager,
		loadBalancer: loadBalancer,
		scheduler:    scheduler,
		ctx:          context.Background(),
	}
}

// RegisterRoutes registers all compute API routes
func (h *ComputeAPIHandler) RegisterRoutes(router *mux.Router) {
	// Job management endpoints - aligned with planned paths
	router.HandleFunc("/api/v1/jobs", h.CreateJob).Methods("POST")
	router.HandleFunc("/api/v1/jobs", h.ListJobs).Methods("GET")
	router.HandleFunc("/api/v1/jobs/{id}", h.GetJob).Methods("GET")
	router.HandleFunc("/api/v1/jobs/{id}", h.UpdateJob).Methods("PUT")
	router.HandleFunc("/api/v1/jobs/{id}", h.DeleteJob).Methods("DELETE")
	router.HandleFunc("/api/v1/jobs/{id}/start", h.StartJob).Methods("POST")
	router.HandleFunc("/api/v1/jobs/{id}/stop", h.StopJob).Methods("POST")
	router.HandleFunc("/api/v1/jobs/{id}/pause", h.PauseJob).Methods("POST")
	router.HandleFunc("/api/v1/jobs/{id}/resume", h.ResumeJob).Methods("POST")
	router.HandleFunc("/api/v1/jobs/{id}/cancel", h.CancelJob).Methods("POST")
	router.HandleFunc("/api/v1/jobs/{id}/logs", h.GetJobLogs).Methods("GET")
	router.HandleFunc("/api/v1/jobs/{id}/status", h.GetJobStatus).Methods("GET")
	router.HandleFunc("/api/v1/jobs/{id}/metrics", h.GetJobMetrics).Methods("GET")

	// Queue management endpoints
	router.HandleFunc("/api/v1/compute/queues", h.ListQueues).Methods("GET")
	router.HandleFunc("/api/v1/compute/queues/{name}", h.GetQueue).Methods("GET")
	router.HandleFunc("/api/v1/compute/queues/{name}/jobs", h.GetQueueJobs).Methods("GET")
	router.HandleFunc("/api/v1/compute/queues/{name}/pause", h.PauseQueue).Methods("POST")
	router.HandleFunc("/api/v1/compute/queues/{name}/resume", h.ResumeQueue).Methods("POST")
	router.HandleFunc("/api/v1/compute/queues/{name}/drain", h.DrainQueue).Methods("POST")

	// Load balancing endpoints
	router.HandleFunc("/api/v1/compute/loadbalancer/status", h.GetLoadBalancerStatus).Methods("GET")
	router.HandleFunc("/api/v1/compute/loadbalancer/algorithms", h.ListAlgorithms).Methods("GET")
	router.HandleFunc("/api/v1/compute/loadbalancer/algorithm", h.SetAlgorithm).Methods("PUT")
	router.HandleFunc("/api/v1/compute/loadbalancer/metrics", h.GetLoadBalancerMetrics).Methods("GET")

	// Resource management endpoints - aligned with planned paths
	router.HandleFunc("/api/v1/resources/clusters", h.ListClusters).Methods("GET")
	router.HandleFunc("/api/v1/resources/clusters/{id}", h.GetCluster).Methods("GET")
	router.HandleFunc("/api/v1/resources/clusters/{id}/resources", h.GetClusterResources).Methods("GET")
	router.HandleFunc("/api/v1/resources/clusters/{id}/utilization", h.GetClusterUtilization).Methods("GET")
	router.HandleFunc("/api/v1/resources/global", h.GetGlobalResources).Methods("GET")
	router.HandleFunc("/api/v1/resources/allocate", h.AllocateResources).Methods("POST")

	// VM migration endpoints - aligned with planned paths
	router.HandleFunc("/api/v1/vms/{id}/migrate-cross-cluster", h.MigrateCrossCluster).Methods("POST")

	// Memory management endpoints - aligned with planned paths
	router.HandleFunc("/api/v1/memory/pools", h.ListMemoryPools).Methods("GET")
	router.HandleFunc("/api/v1/memory/pools", h.CreateMemoryPool).Methods("POST")
	router.HandleFunc("/api/v1/memory/pools/{id}", h.GetMemoryPool).Methods("GET")
	router.HandleFunc("/api/v1/memory/allocate", h.AllocateMemory).Methods("POST")
	router.HandleFunc("/api/v1/memory/release/{id}", h.ReleaseMemory).Methods("POST")

	// Performance and optimization endpoints
	router.HandleFunc("/api/v1/compute/performance/metrics", h.GetPerformanceMetrics).Methods("GET")
	router.HandleFunc("/api/v1/compute/performance/optimize", h.OptimizePerformance).Methods("POST")
	router.HandleFunc("/api/v1/compute/performance/benchmark", h.RunBenchmark).Methods("POST")

	// Health and monitoring endpoints
	router.HandleFunc("/api/v1/compute/health", h.HealthCheck).Methods("GET")
	router.HandleFunc("/api/v1/compute/stats", h.GetStats).Methods("GET")

	// Additional REST endpoints for planned specification compliance
	router.HandleFunc("/api/v1/compute/jobs/{id}/history", h.GetJobHistory).Methods("GET")
	router.HandleFunc("/api/v1/compute/jobs/batch", h.BatchOperations).Methods("POST")
	router.HandleFunc("/api/v1/compute/nodes", h.ListNodes).Methods("GET")
	router.HandleFunc("/api/v1/compute/nodes/{id}", h.GetNode).Methods("GET")
	router.HandleFunc("/api/v1/compute/nodes/{id}/jobs", h.GetNodeJobs).Methods("GET")
	router.HandleFunc("/api/v1/compute/scheduler/policy", h.GetSchedulingPolicy).Methods("GET")
	router.HandleFunc("/api/v1/compute/scheduler/policy", h.SetSchedulingPolicy).Methods("PUT")

	// Resource planning and capacity endpoints
	router.HandleFunc("/api/v1/compute/capacity/plan", h.CapacityPlanning).Methods("POST")
	router.HandleFunc("/api/v1/compute/resources/forecast", h.ResourceForecast).Methods("GET")
	router.HandleFunc("/api/v1/compute/resources/reserve", h.ReserveResources).Methods("POST")
	router.HandleFunc("/api/v1/compute/resources/release", h.ReleaseResources).Methods("POST")
}

// Request/Response types

type CreateJobRequest struct {
	Name         string               `json:"name"`
	Description  string               `json:"description"`
	JobType      string               `json:"job_type"`
	Priority     int                  `json:"priority"`
	QueueName    string               `json:"queue_name"`
	Command      []string             `json:"command"`
	Environment  map[string]string    `json:"environment"`
	Resources    ResourceRequirements `json:"resources"`
	Constraints  []JobConstraint      `json:"constraints"`
	Dependencies []string             `json:"dependencies"`
	Tags         map[string]string    `json:"tags"`
	Timeout      int                  `json:"timeout_seconds"`
}

type ResourceRequirements struct {
	CPUCores    float64 `json:"cpu_cores"`
	MemoryGB    float64 `json:"memory_gb"`
	DiskGB      float64 `json:"disk_gb"`
	GPUs        int     `json:"gpus"`
	NetworkMbps float64 `json:"network_mbps"`
}

type JobConstraint struct {
	Type     string      `json:"type"`
	Operator string      `json:"operator"`
	Value    interface{} `json:"value"`
}

type JobResponse struct {
	ID            string                 `json:"id"`
	Name          string                 `json:"name"`
	Description   string                 `json:"description"`
	JobType       string                 `json:"job_type"`
	Status        string                 `json:"status"`
	Priority      int                    `json:"priority"`
	QueueName     string                 `json:"queue_name"`
	ClusterID     string                 `json:"cluster_id"`
	NodeID        string                 `json:"node_id"`
	Resources     ResourceRequirements   `json:"resources"`
	CreatedAt     time.Time              `json:"created_at"`
	StartedAt     *time.Time             `json:"started_at"`
	CompletedAt   *time.Time             `json:"completed_at"`
	ExecutionTime *int64                 `json:"execution_time_ms"`
	Tags          map[string]string      `json:"tags"`
	Metadata      map[string]interface{} `json:"metadata"`
	ErrorMessage  string                 `json:"error_message,omitempty"`
}

type AllocateResourcesRequest struct {
	Constraints        []ResourceConstraint `json:"constraints"`
	NetworkConstraints *NetworkConstraint   `json:"network_constraints,omitempty"`
	Priority           int                  `json:"priority"`
	TimeoutSeconds     int                  `json:"timeout_seconds"`
	ClusterHint        string               `json:"cluster_hint,omitempty"`
	Strategy           string               `json:"strategy,omitempty"`
}

type ResourceConstraint struct {
	Type      string  `json:"type"`
	MinAmount float64 `json:"min_amount"`
	MaxAmount float64 `json:"max_amount"`
}

type NetworkConstraint struct {
	MinBandwidthBps    uint64  `json:"min_bandwidth_bps"`
	MaxLatencyMs       float64 `json:"max_latency_ms"`
	RequiredTopology   string  `json:"required_topology"`
	MinConnections     int     `json:"min_connections"`
	BandwidthGuarantee bool    `json:"bandwidth_guarantee"`
}

// Job Management Handlers

func (h *ComputeAPIHandler) CreateJob(w http.ResponseWriter, r *http.Request) {
	var req CreateJobRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request body: %v", err), http.StatusBadRequest)
		return
	}

	// Convert API request to internal job structure
	job := &compcore.ComputeJob{
		Name:        req.Name,
		Description: req.Description,
		Type:        compcore.JobType(req.JobType),
		Priority:    req.Priority,
		QueueName:   req.QueueName,
		Command:     req.Command,
		Environment: req.Environment,
		Resources: compcore.ResourceRequirements{
			CPUCores:    req.Resources.CPUCores,
			MemoryGB:    req.Resources.MemoryGB,
			DiskGB:      req.Resources.DiskGB,
			GPUs:        req.Resources.GPUs,
			NetworkMbps: req.Resources.NetworkMbps,
		},
		Tags:      req.Tags,
		Timeout:   time.Duration(req.Timeout) * time.Second,
		CreatedAt: time.Now(),
		Status:    compcore.JobStatusPending,
	}

	// Convert constraints
	for _, constraint := range req.Constraints {
		job.Constraints = append(job.Constraints, compcore.JobConstraint{
			Type:     constraint.Type,
			Operator: constraint.Operator,
			Value:    constraint.Value,
		})
	}

	// Submit job to job manager
	jobID, err := h.jobManager.SubmitJob(h.ctx, job)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to submit job: %v", err), http.StatusInternalServerError)
		return
	}

	job.ID = jobID

	// Convert to response format
	response := convertJobToResponse(job)

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(response)

	log.Printf("Created compute job %s: %s", jobID, req.Name)
}

func (h *ComputeAPIHandler) ListJobs(w http.ResponseWriter, r *http.Request) {
	// Parse query parameters
	queueName := r.URL.Query().Get("queue")
	status := r.URL.Query().Get("status")
	limit := 100
	if l := r.URL.Query().Get("limit"); l != "" {
		if parsed, err := strconv.Atoi(l); err == nil && parsed > 0 {
			limit = parsed
		}
	}

	offset := 0
	if o := r.URL.Query().Get("offset"); o != "" {
		if parsed, err := strconv.Atoi(o); err == nil && parsed >= 0 {
			offset = parsed
		}
	}

	// Get jobs from job manager using individual parameters
	jobs, err := h.jobManager.ListJobsWithParams(h.ctx, queueName, status, limit, offset)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to list jobs: %v", err), http.StatusInternalServerError)
		return
	}

	// Convert to response format
	var responses []JobResponse
	for _, job := range jobs {
		responses = append(responses, convertJobToResponse(job))
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"jobs":   responses,
		"total":  len(responses),
		"limit":  limit,
		"offset": offset,
	})
}

func (h *ComputeAPIHandler) GetJob(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	jobID := vars["id"]

	job, err := h.jobManager.GetJob(h.ctx, jobID)
	if err != nil {
		http.Error(w, fmt.Sprintf("Job not found: %v", err), http.StatusNotFound)
		return
	}

	response := convertJobToResponse(job)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (h *ComputeAPIHandler) UpdateJob(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	jobID := vars["id"]

	var updateReq map[string]interface{}
	if err := json.NewDecoder(r.Body).Decode(&updateReq); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request body: %v", err), http.StatusBadRequest)
		return
	}

	err := h.jobManager.UpdateJob(h.ctx, jobID, updateReq)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to update job: %v", err), http.StatusInternalServerError)
		return
	}

	job, err := h.jobManager.GetJob(h.ctx, jobID)
	if err != nil {
		http.Error(w, fmt.Sprintf("Job not found after update: %v", err), http.StatusNotFound)
		return
	}

	response := convertJobToResponse(job)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (h *ComputeAPIHandler) DeleteJob(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	jobID := vars["id"]

	err := h.jobManager.DeleteJob(h.ctx, jobID)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to delete job: %v", err), http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusNoContent)
	log.Printf("Deleted compute job %s", jobID)
}

func (h *ComputeAPIHandler) StartJob(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	jobID := vars["id"]

	err := h.jobManager.StartJob(h.ctx, jobID)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to start job: %v", err), http.StatusInternalServerError)
		return
	}

	job, err := h.jobManager.GetJob(h.ctx, jobID)
	if err != nil {
		http.Error(w, fmt.Sprintf("Job not found: %v", err), http.StatusNotFound)
		return
	}

	response := convertJobToResponse(job)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (h *ComputeAPIHandler) StopJob(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	jobID := vars["id"]

	err := h.jobManager.StopJob(h.ctx, jobID)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to stop job: %v", err), http.StatusInternalServerError)
		return
	}

	job, err := h.jobManager.GetJob(h.ctx, jobID)
	if err != nil {
		http.Error(w, fmt.Sprintf("Job not found: %v", err), http.StatusNotFound)
		return
	}

	response := convertJobToResponse(job)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (h *ComputeAPIHandler) PauseJob(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	jobID := vars["id"]

	err := h.jobManager.PauseJob(h.ctx, jobID)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to pause job: %v", err), http.StatusInternalServerError)
		return
	}

	job, err := h.jobManager.GetJob(h.ctx, jobID)
	if err != nil {
		http.Error(w, fmt.Sprintf("Job not found: %v", err), http.StatusNotFound)
		return
	}

	response := convertJobToResponse(job)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (h *ComputeAPIHandler) ResumeJob(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	jobID := vars["id"]

	err := h.jobManager.ResumeJob(h.ctx, jobID)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to resume job: %v", err), http.StatusInternalServerError)
		return
	}

	job, err := h.jobManager.GetJob(h.ctx, jobID)
	if err != nil {
		http.Error(w, fmt.Sprintf("Job not found: %v", err), http.StatusNotFound)
		return
	}

	response := convertJobToResponse(job)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (h *ComputeAPIHandler) CancelJob(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	jobID := vars["id"]

	err := h.jobManager.CancelJob(h.ctx, jobID)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to cancel job: %v", err), http.StatusInternalServerError)
		return
	}

	job, err := h.jobManager.GetJob(h.ctx, jobID)
	if err != nil {
		http.Error(w, fmt.Sprintf("Job not found: %v", err), http.StatusNotFound)
		return
	}

	response := convertJobToResponse(job)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (h *ComputeAPIHandler) GetJobLogs(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	jobID := vars["id"]

	lines := 1000
	if l := r.URL.Query().Get("lines"); l != "" {
		if parsed, err := strconv.Atoi(l); err == nil && parsed > 0 {
			lines = parsed
		}
	}

	logs, err := h.jobManager.GetJobLogs(h.ctx, jobID, lines)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to get job logs: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"job_id": jobID,
		"logs":   logs,
		"lines":  len(logs),
	})
}

func (h *ComputeAPIHandler) GetJobStatus(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	jobID := vars["id"]

	status, err := h.jobManager.GetJobStatus(h.ctx, jobID)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to get job status: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"job_id": jobID,
		"status": status,
	})
}

func (h *ComputeAPIHandler) GetJobMetrics(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	jobID := vars["id"]

	metrics, err := h.jobManager.GetJobMetrics(h.ctx, jobID)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to get job metrics: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"job_id":  jobID,
		"metrics": metrics,
	})
}

// Queue Management Handlers

func (h *ComputeAPIHandler) ListQueues(w http.ResponseWriter, r *http.Request) {
	queues, err := h.jobManager.ListQueues(h.ctx)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to list queues: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"queues": queues,
		"total":  len(queues),
	})
}

func (h *ComputeAPIHandler) GetQueue(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	queueName := vars["name"]

	queue, err := h.jobManager.GetQueue(h.ctx, queueName)
	if err != nil {
		http.Error(w, fmt.Sprintf("Queue not found: %v", err), http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(queue)
}

func (h *ComputeAPIHandler) GetQueueJobs(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	queueName := vars["name"]

	jobs, err := h.jobManager.GetQueueJobs(h.ctx, queueName)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to get queue jobs: %v", err), http.StatusInternalServerError)
		return
	}

	var responses []JobResponse
	for _, job := range jobs {
		responses = append(responses, convertJobToResponse(job))
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"queue": queueName,
		"jobs":  responses,
		"total": len(responses),
	})
}

func (h *ComputeAPIHandler) PauseQueue(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	queueName := vars["name"]

	err := h.jobManager.PauseQueue(h.ctx, queueName)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to pause queue: %v", err), http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"queue":  queueName,
		"status": "paused",
	})
}

func (h *ComputeAPIHandler) ResumeQueue(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	queueName := vars["name"]

	err := h.jobManager.ResumeQueue(h.ctx, queueName)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to resume queue: %v", err), http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"queue":  queueName,
		"status": "active",
	})
}

func (h *ComputeAPIHandler) DrainQueue(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	queueName := vars["name"]

	err := h.jobManager.DrainQueue(h.ctx, queueName)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to drain queue: %v", err), http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"queue":  queueName,
		"status": "draining",
	})
}

// Load Balancer Handlers

func (h *ComputeAPIHandler) GetLoadBalancerStatus(w http.ResponseWriter, r *http.Request) {
	status := h.loadBalancer.GetStatus()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(status)
}

func (h *ComputeAPIHandler) ListAlgorithms(w http.ResponseWriter, r *http.Request) {
	algorithms := h.loadBalancer.GetAvailableAlgorithms()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"algorithms": algorithms,
		"total":      len(algorithms),
	})
}

func (h *ComputeAPIHandler) SetAlgorithm(w http.ResponseWriter, r *http.Request) {
	var req map[string]string
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request body: %v", err), http.StatusBadRequest)
		return
	}

	algorithm, exists := req["algorithm"]
	if !exists {
		http.Error(w, "Missing algorithm field", http.StatusBadRequest)
		return
	}

	err := h.loadBalancer.SetAlgorithm(compcore.LoadBalancingAlgorithm(algorithm))
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to set algorithm: %v", err), http.StatusBadRequest)
		return
	}

	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"algorithm": algorithm,
		"status":    "updated",
	})
}

func (h *ComputeAPIHandler) GetLoadBalancerMetrics(w http.ResponseWriter, r *http.Request) {
	metrics := h.loadBalancer.GetMetrics()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(metrics)
}

// Resource Management Handlers

func (h *ComputeAPIHandler) ListClusters(w http.ResponseWriter, r *http.Request) {
	inventory, err := h.scheduler.GetGlobalResourceInventory()
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to get clusters: %v", err), http.StatusInternalServerError)
		return
	}

	var clusters []map[string]interface{}
	for clusterID, cluster := range inventory {
		clusters = append(clusters, map[string]interface{}{
			"id":             clusterID,
			"location":       cluster.Location,
			"is_healthy":     cluster.IsHealthy,
			"priority":       cluster.Priority,
			"network_cost":   cluster.NetworkCost,
			"last_heartbeat": cluster.LastHeartbeat,
			"capabilities":   cluster.Capabilities,
			"node_count":     len(cluster.Nodes),
		})
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"clusters": clusters,
		"total":    len(clusters),
	})
}

func (h *ComputeAPIHandler) GetCluster(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	clusterID := vars["id"]

	inventory, err := h.scheduler.GetGlobalResourceInventory()
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to get cluster: %v", err), http.StatusInternalServerError)
		return
	}

	cluster, exists := inventory[clusterID]
	if !exists {
		http.Error(w, "Cluster not found", http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(cluster)
}

func (h *ComputeAPIHandler) GetClusterResources(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	clusterID := vars["id"]

	inventory, err := h.scheduler.GetGlobalResourceInventory()
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to get cluster resources: %v", err), http.StatusInternalServerError)
		return
	}

	cluster, exists := inventory[clusterID]
	if !exists {
		http.Error(w, "Cluster not found", http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"cluster_id": clusterID,
		"resources":  cluster.Resources,
		"nodes":      cluster.Nodes,
	})
}

func (h *ComputeAPIHandler) GetClusterUtilization(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	clusterID := vars["id"]

	utilization, err := h.scheduler.GetGlobalResourceUtilization()
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to get utilization: %v", err), http.StatusInternalServerError)
		return
	}

	clusterUtil, exists := utilization[clusterID]
	if !exists {
		http.Error(w, "Cluster not found", http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"cluster_id":  clusterID,
		"utilization": clusterUtil,
	})
}

func (h *ComputeAPIHandler) GetGlobalResources(w http.ResponseWriter, r *http.Request) {
	inventory, err := h.scheduler.GetGlobalResourceInventory()
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to get global resources: %v", err), http.StatusInternalServerError)
		return
	}

	utilization, err := h.scheduler.GetGlobalResourceUtilization()
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to get utilization: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"inventory":   inventory,
		"utilization": utilization,
	})
}

func (h *ComputeAPIHandler) AllocateResources(w http.ResponseWriter, r *http.Request) {
	var req AllocateResourcesRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request body: %v", err), http.StatusBadRequest)
		return
	}

	// Convert API constraints to scheduler constraints
	var constraints []sched.ResourceConstraint
	for _, constraint := range req.Constraints {
		constraints = append(constraints, sched.ResourceConstraint{
			Type:      sched.ResourceType(constraint.Type),
			MinAmount: constraint.MinAmount,
			MaxAmount: constraint.MaxAmount,
		})
	}

	// Convert network constraints if present
	var networkConstraints *sched.NetworkConstraint
	if req.NetworkConstraints != nil {
		networkConstraints = &sched.NetworkConstraint{
			MinBandwidthBps:    req.NetworkConstraints.MinBandwidthBps,
			MaxLatencyMs:       req.NetworkConstraints.MaxLatencyMs,
			RequiredTopology:   req.NetworkConstraints.RequiredTopology,
			MinConnections:     req.NetworkConstraints.MinConnections,
			BandwidthGuarantee: req.NetworkConstraints.BandwidthGuarantee,
		}
	}

	// Create resource request
	requestID, err := h.scheduler.RequestResourcesWithNetworkConstraints(
		constraints,
		networkConstraints,
		req.Priority,
		time.Duration(req.TimeoutSeconds)*time.Second,
	)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to request resources: %v", err), http.StatusInternalServerError)
		return
	}

	// Try global allocation if strategy is specified
	if req.Strategy != "" {
		// For now, use basic global allocation
		resourceRequest := &sched.ResourceRequest{
			ID:                 requestID,
			Constraints:        constraints,
			NetworkConstraints: networkConstraints,
			Priority:           req.Priority,
			Timeout:            time.Duration(req.TimeoutSeconds) * time.Second,
			CreatedAt:          time.Now(),
		}

		allocation, err := h.scheduler.ScheduleGlobalResourceAllocation(resourceRequest)
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to allocate resources globally: %v", err), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)
		json.NewEncoder(w).Encode(map[string]interface{}{
			"request_id": requestID,
			"allocation": allocation,
			"strategy":   req.Strategy,
		})
		return
	}

	// Regular local allocation
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"request_id": requestID,
		"status":     "pending",
	})
}

// Performance and Monitoring Handlers

func (h *ComputeAPIHandler) GetPerformanceMetrics(w http.ResponseWriter, r *http.Request) {
	metrics, err := h.jobManager.GetPerformanceMetrics(h.ctx)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to get performance metrics: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(metrics)
}

func (h *ComputeAPIHandler) OptimizePerformance(w http.ResponseWriter, r *http.Request) {
	err := h.jobManager.OptimizePerformance(h.ctx)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to optimize performance: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":    "optimization completed",
		"timestamp": time.Now(),
	})
}

func (h *ComputeAPIHandler) RunBenchmark(w http.ResponseWriter, r *http.Request) {
	result, err := h.jobManager.RunBenchmark(h.ctx)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to run benchmark: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(result)
}

func (h *ComputeAPIHandler) HealthCheck(w http.ResponseWriter, r *http.Request) {
	health := map[string]interface{}{
		"status":        "healthy",
		"timestamp":     time.Now(),
		"job_manager":   h.jobManager.IsHealthy(h.ctx),
		"load_balancer": h.loadBalancer.IsHealthy(),
		"scheduler":     true, // scheduler doesn't have IsHealthy method yet
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(health)
}

func (h *ComputeAPIHandler) GetStats(w http.ResponseWriter, r *http.Request) {
	stats := map[string]interface{}{
		"job_manager_stats": func() interface{} {
			stats, err := h.jobManager.GetStatistics(h.ctx)
			if err != nil {
				return map[string]interface{}{"error": err.Error()}
			}
			return stats
		}(),
		"load_balancer": h.loadBalancer.GetStatistics(),
		"scheduler": map[string]interface{}{
			"nodes":       h.scheduler.GetNodesStatus(),
			"requests":    h.scheduler.GetPendingRequests(),
			"allocations": h.scheduler.GetActiveAllocations(),
			"tasks":       h.scheduler.GetTasks(),
		},
		"timestamp": time.Now(),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(stats)
}

// Helper functions

func convertJobToResponse(job *compcore.ComputeJob) JobResponse {
	response := JobResponse{
		ID:          job.ID,
		Name:        job.Name,
		Description: job.Description,
		JobType:     string(job.Type),
		Status:      string(job.Status),
		Priority:    int(job.Priority),
		QueueName:   job.QueueName,
		Resources: ResourceRequirements{
			CPUCores:    job.Resources.CPUCores,
			MemoryGB:    job.Resources.MemoryGB,
			DiskGB:      job.Resources.StorageGB,
			GPUs:        job.Resources.GPUCount,
			NetworkMbps: job.Resources.NetworkMbps,
		},
		CreatedAt: job.SubmittedAt,
		Tags:      job.Tags,
		Metadata:  job.Metadata,
	}

	// Handle cluster placement - use first placement if available
	if len(job.ClusterPlacements) > 0 {
		response.ClusterID = job.ClusterPlacements[0].ClusterID
		if len(job.ClusterPlacements[0].NodeIDs) > 0 {
			response.NodeID = job.ClusterPlacements[0].NodeIDs[0]
		}
	}

	// Handle time fields with proper pointer handling
	if job.StartedAt != nil {
		response.StartedAt = job.StartedAt
	}

	if job.CompletedAt != nil {
		response.CompletedAt = job.CompletedAt
		if job.StartedAt != nil {
			execTime := job.CompletedAt.Sub(*job.StartedAt).Milliseconds()
			response.ExecutionTime = &execTime
		}
	}

	if job.ErrorMessage != "" {
		response.ErrorMessage = job.ErrorMessage
	}

	return response
}

// Migration and Memory Management Handlers

func (h *ComputeAPIHandler) MigrateCrossCluster(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	vmID := vars["id"]

	var req map[string]interface{}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request body: %v", err), http.StatusBadRequest)
		return
	}

	// DEFERRED: Cross-cluster migration requires coordination protocol implementation
	// This would involve cluster membership, consensus, and state transfer mechanisms
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"vm_id":     vmID,
		"status":    "not_implemented",
		"message":   "Cross-cluster migration requires distributed coordination",
		"timestamp": time.Now(),
	})
}

func (h *ComputeAPIHandler) ListMemoryPools(w http.ResponseWriter, r *http.Request) {
	// DEFERRED: Memory pool listing requires memory manager integration
	// Would query available memory pools from hypervisor and memory allocator
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"pools":  []interface{}{},
		"total":  0,
		"status": "not_implemented",
	})
}

func (h *ComputeAPIHandler) CreateMemoryPool(w http.ResponseWriter, r *http.Request) {
	var req map[string]interface{}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request body: %v", err), http.StatusBadRequest)
		return
	}

	// DEFERRED: Memory pool creation requires memory manager API
	// Would create dedicated memory pool via hypervisor or NUMA allocator
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusNotImplemented)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"error":     "Memory pool creation not implemented",
		"status":    "not_implemented",
		"timestamp": time.Now(),
	})
}

func (h *ComputeAPIHandler) GetMemoryPool(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	poolID := vars["id"]

	// DEFERRED: Memory pool retrieval requires memory manager API
	// Would fetch pool details from hypervisor memory statistics
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusNotImplemented)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"pool_id": poolID,
		"error":   "Memory pool retrieval not implemented",
		"status":  "not_implemented",
	})
}

func (h *ComputeAPIHandler) AllocateMemory(w http.ResponseWriter, r *http.Request) {
	var req map[string]interface{}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request body: %v", err), http.StatusBadRequest)
		return
	}

	// DEFERRED: Memory allocation requires memory manager integration
	// Would allocate memory from pool using hypervisor allocation API
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusNotImplemented)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"error":     "Memory allocation not implemented",
		"status":    "not_implemented",
		"timestamp": time.Now(),
	})
}

func (h *ComputeAPIHandler) ReleaseMemory(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	allocationID := vars["id"]

	// DEFERRED: Memory release requires memory manager integration
	// Would release allocated memory back to pool via hypervisor API
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusNotImplemented)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"allocation_id": allocationID,
		"error":         "Memory release not implemented",
		"status":        "not_implemented",
		"timestamp":     time.Now(),
	})
}
