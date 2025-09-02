package vm

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/gorilla/mux"
	"github.com/khryptorgraphics/novacron/backend/core/vm"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	bulkOperationDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "novacron_vm_bulk_operation_duration_seconds",
		Help:    "Duration of VM bulk operations",
		Buckets: prometheus.DefBuckets,
	}, []string{"operation"})

	bulkOperationCount = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "novacron_vm_bulk_operations_total",
		Help: "Total number of VM bulk operations",
	}, []string{"operation", "status"})

	consoleAccessCount = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "novacron_vm_console_access_total",
		Help: "Total number of VM console access requests",
	}, []string{"vm_id", "status"})
)

// ExtendedHandler handles advanced VM API requests
type ExtendedHandler struct {
	vmManager       *vm.VMManager
	snapshotManager *vm.VMSnapshotManager
	consoleManager  *vm.ConsoleManager
}

// NewExtendedHandler creates a new extended VM API handler
func NewExtendedHandler(vmManager *vm.VMManager, snapshotManager *vm.VMSnapshotManager, consoleManager *vm.ConsoleManager) *ExtendedHandler {
	return &ExtendedHandler{
		vmManager:       vmManager,
		snapshotManager: snapshotManager,
		consoleManager:  consoleManager,
	}
}

// BulkCreateRequest represents bulk VM creation request
type BulkCreateRequest struct {
	VMs []VMCreateSpec `json:"vms" validate:"required,dive"`
}

// VMCreateSpec represents VM creation specification
type VMCreateSpec struct {
	Name        string            `json:"name" validate:"required,min=1,max=64"`
	Template    string            `json:"template" validate:"required"`
	CPU         int               `json:"cpu" validate:"required,min=1,max=64"`
	Memory      int               `json:"memory" validate:"required,min=512,max=131072"`
	Disk        int               `json:"disk" validate:"required,min=10,max=10240"`
	Network     string            `json:"network,omitempty"`
	Tags        map[string]string `json:"tags,omitempty"`
	Environment map[string]string `json:"environment,omitempty"`
}

// BulkDeleteRequest represents bulk VM deletion request
type BulkDeleteRequest struct {
	VMIDs []string `json:"vm_ids" validate:"required,dive,uuid"`
	Force bool     `json:"force,omitempty"`
}

// ConsoleAccessResponse represents console access response
type ConsoleAccessResponse struct {
	ConsoleURL   string    `json:"console_url"`
	SessionID    string    `json:"session_id"`
	ExpiresAt    time.Time `json:"expires_at"`
	Protocol     string    `json:"protocol"`
	Instructions string    `json:"instructions,omitempty"`
}

// SnapshotRequest represents snapshot creation request
type SnapshotRequest struct {
	Name        string `json:"name" validate:"required,min=1,max=64"`
	Description string `json:"description,omitempty,max=255"`
	Memory      bool   `json:"memory,omitempty"`
	Quiesce     bool   `json:"quiesce,omitempty"`
}

// CloneRequest represents VM clone request
type CloneRequest struct {
	Name        string            `json:"name" validate:"required,min=1,max=64"`
	Description string            `json:"description,omitempty,max=255"`
	LinkedClone bool              `json:"linked_clone,omitempty"`
	PowerOn     bool              `json:"power_on,omitempty"`
	Network     string            `json:"network,omitempty"`
	Tags        map[string]string `json:"tags,omitempty"`
}

// BulkOperationResponse represents bulk operation response
type BulkOperationResponse struct {
	OperationID string                   `json:"operation_id"`
	Status      string                   `json:"status"`
	Total       int                      `json:"total"`
	Successful  int                      `json:"successful"`
	Failed      int                      `json:"failed"`
	Results     []BulkOperationResult    `json:"results"`
	Errors      []string                 `json:"errors,omitempty"`
	StartedAt   time.Time                `json:"started_at"`
	CompletedAt *time.Time               `json:"completed_at,omitempty"`
}

// BulkOperationResult represents individual operation result
type BulkOperationResult struct {
	ID      string `json:"id,omitempty"`
	Name    string `json:"name"`
	Status  string `json:"status"`
	Error   string `json:"error,omitempty"`
	Details map[string]interface{} `json:"details,omitempty"`
}

// RegisterExtendedRoutes registers extended VM API routes
func (h *ExtendedHandler) RegisterExtendedRoutes(router *mux.Router, require func(string, http.HandlerFunc) http.Handler) {
	vmRouter := router.PathPrefix("/api/v1").Subrouter()

	// Bulk operations (admin only)
	vmRouter.Handle("/vms/bulk-create", require("admin", h.BulkCreateVMs)).Methods("POST")
	vmRouter.Handle("/vms/bulk-delete", require("admin", h.BulkDeleteVMs)).Methods("POST")

	// Console access (operator+)
	vmRouter.Handle("/vms/{id}/console", require("operator", h.GetConsoleAccess)).Methods("GET")

	// Snapshot operations (operator+)
	vmRouter.Handle("/vms/{id}/snapshot", require("operator", h.CreateSnapshot)).Methods("POST")
	vmRouter.Handle("/vms/{id}/snapshots", require("viewer", h.ListSnapshots)).Methods("GET")

	// Clone operations (operator+)
	vmRouter.Handle("/vms/{id}/clone", require("operator", h.CloneVM)).Methods("POST")
}

// BulkCreateVMs handles POST /api/v1/vms/bulk-create
// @Summary Bulk create VMs
// @Description Create multiple VMs in a single operation with validation
// @Tags VMs
// @Accept json
// @Produce json
// @Param request body BulkCreateRequest true "Bulk create request"
// @Success 202 {object} BulkOperationResponse
// @Failure 400 {object} map[string]interface{} "Invalid request"
// @Failure 401 {object} map[string]interface{} "Unauthorized"
// @Failure 403 {object} map[string]interface{} "Forbidden"
// @Failure 500 {object} map[string]interface{} "Internal server error"
// @Router /api/v1/vms/bulk-create [post]
func (h *ExtendedHandler) BulkCreateVMs(w http.ResponseWriter, r *http.Request) {
	timer := prometheus.NewTimer(bulkOperationDuration.WithLabelValues("create"))
	defer timer.ObserveDuration()

	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Minute)
	defer cancel()

	var req BulkCreateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		bulkOperationCount.WithLabelValues("create", "error").Inc()
		writeError(w, http.StatusBadRequest, "INVALID_JSON", "Invalid JSON format")
		return
	}

	// Validate request
	if len(req.VMs) == 0 {
		bulkOperationCount.WithLabelValues("create", "error").Inc()
		writeError(w, http.StatusBadRequest, "EMPTY_REQUEST", "No VMs specified for creation")
		return
	}

	if len(req.VMs) > 50 {
		bulkOperationCount.WithLabelValues("create", "error").Inc()
		writeError(w, http.StatusBadRequest, "TOO_MANY_VMS", "Maximum 50 VMs can be created in bulk")
		return
	}

	// Validate each VM spec
	for i, vmSpec := range req.VMs {
		if err := h.validateVMSpec(vmSpec); err != nil {
			bulkOperationCount.WithLabelValues("create", "error").Inc()
			writeError(w, http.StatusBadRequest, "INVALID_VM_SPEC", 
				fmt.Sprintf("VM %d validation failed: %s", i+1, err.Error()))
			return
		}
	}

	// Start bulk operation
	operationID := h.generateOperationID()
	response := &BulkOperationResponse{
		OperationID: operationID,
		Status:      "in_progress",
		Total:       len(req.VMs),
		StartedAt:   time.Now(),
		Results:     make([]BulkOperationResult, 0, len(req.VMs)),
	}

	// Process VMs asynchronously
	go h.processBulkCreate(ctx, operationID, req.VMs, response)

	bulkOperationCount.WithLabelValues("create", "success").Inc()
	writeJSON(w, http.StatusAccepted, response)
}

// BulkDeleteVMs handles POST /api/v1/vms/bulk-delete
// @Summary Bulk delete VMs
// @Description Delete multiple VMs in a single operation with optional force
// @Tags VMs
// @Accept json
// @Produce json
// @Param request body BulkDeleteRequest true "Bulk delete request"
// @Success 202 {object} BulkOperationResponse
// @Failure 400 {object} map[string]interface{} "Invalid request"
// @Failure 401 {object} map[string]interface{} "Unauthorized"
// @Failure 403 {object} map[string]interface{} "Forbidden"
// @Failure 500 {object} map[string]interface{} "Internal server error"
// @Router /api/v1/vms/bulk-delete [post]
func (h *ExtendedHandler) BulkDeleteVMs(w http.ResponseWriter, r *http.Request) {
	timer := prometheus.NewTimer(bulkOperationDuration.WithLabelValues("delete"))
	defer timer.ObserveDuration()

	ctx, cancel := context.WithTimeout(r.Context(), 15*time.Minute)
	defer cancel()

	var req BulkDeleteRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		bulkOperationCount.WithLabelValues("delete", "error").Inc()
		writeError(w, http.StatusBadRequest, "INVALID_JSON", "Invalid JSON format")
		return
	}

	// Validate request
	if len(req.VMIDs) == 0 {
		bulkOperationCount.WithLabelValues("delete", "error").Inc()
		writeError(w, http.StatusBadRequest, "EMPTY_REQUEST", "No VM IDs specified for deletion")
		return
	}

	if len(req.VMIDs) > 50 {
		bulkOperationCount.WithLabelValues("delete", "error").Inc()
		writeError(w, http.StatusBadRequest, "TOO_MANY_VMS", "Maximum 50 VMs can be deleted in bulk")
		return
	}

	// Start bulk operation
	operationID := h.generateOperationID()
	response := &BulkOperationResponse{
		OperationID: operationID,
		Status:      "in_progress",
		Total:       len(req.VMIDs),
		StartedAt:   time.Now(),
		Results:     make([]BulkOperationResult, 0, len(req.VMIDs)),
	}

	// Process deletions asynchronously
	go h.processBulkDelete(ctx, operationID, req.VMIDs, req.Force, response)

	bulkOperationCount.WithLabelValues("delete", "success").Inc()
	writeJSON(w, http.StatusAccepted, response)
}

// GetConsoleAccess handles GET /api/v1/vms/{id}/console
// @Summary Get VM console access
// @Description Generate secure console access URL for VM
// @Tags VMs
// @Produce json
// @Param id path string true "VM ID"
// @Success 200 {object} ConsoleAccessResponse
// @Failure 400 {object} map[string]interface{} "Invalid request"
// @Failure 401 {object} map[string]interface{} "Unauthorized"
// @Failure 404 {object} map[string]interface{} "VM not found"
// @Failure 500 {object} map[string]interface{} "Internal server error"
// @Router /api/v1/vms/{id}/console [get]
func (h *ExtendedHandler) GetConsoleAccess(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	vmID := vars["id"]

	// Validate VM ID
	if vmID == "" {
		consoleAccessCount.WithLabelValues("", "error").Inc()
		writeError(w, http.StatusBadRequest, "MISSING_VM_ID", "VM ID is required")
		return
	}

	// Check if VM exists and is accessible
	vmInfo, err := h.vmManager.GetVM(r.Context(), vmID)
	if err != nil {
		if err == vm.ErrVMNotFound {
			consoleAccessCount.WithLabelValues(vmID, "not_found").Inc()
			writeError(w, http.StatusNotFound, "VM_NOT_FOUND", "VM not found")
			return
		}
		consoleAccessCount.WithLabelValues(vmID, "error").Inc()
		writeError(w, http.StatusInternalServerError, "INTERNAL_ERROR", "Failed to retrieve VM information")
		return
	}

	// Check VM state
	if vmInfo.Status != "running" {
		consoleAccessCount.WithLabelValues(vmID, "unavailable").Inc()
		writeError(w, http.StatusConflict, "VM_NOT_RUNNING", "VM must be running to access console")
		return
	}

	// Generate console access
	sessionID, err := h.consoleManager.CreateConsoleSession(r.Context(), vmID)
	if err != nil {
		consoleAccessCount.WithLabelValues(vmID, "error").Inc()
		writeError(w, http.StatusInternalServerError, "CONSOLE_ERROR", "Failed to create console session")
		return
	}

	// Generate secure console URL
	baseURL := h.getBaseURL(r)
	consoleURL := fmt.Sprintf("%s/console/%s?session=%s", baseURL, vmID, sessionID)

	response := ConsoleAccessResponse{
		ConsoleURL:   consoleURL,
		SessionID:    sessionID,
		ExpiresAt:    time.Now().Add(4 * time.Hour),
		Protocol:     "websocket",
		Instructions: "Connect to the WebSocket endpoint to access VM console",
	}

	consoleAccessCount.WithLabelValues(vmID, "success").Inc()
	writeJSON(w, http.StatusOK, response)
}

// CreateSnapshot handles POST /api/v1/vms/{id}/snapshot
// @Summary Create VM snapshot
// @Description Create a snapshot of the VM with optional memory capture
// @Tags VMs
// @Accept json
// @Produce json
// @Param id path string true "VM ID"
// @Param request body SnapshotRequest true "Snapshot request"
// @Success 202 {object} map[string]interface{}
// @Failure 400 {object} map[string]interface{} "Invalid request"
// @Failure 401 {object} map[string]interface{} "Unauthorized"
// @Failure 404 {object} map[string]interface{} "VM not found"
// @Failure 500 {object} map[string]interface{} "Internal server error"
// @Router /api/v1/vms/{id}/snapshot [post]
func (h *ExtendedHandler) CreateSnapshot(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	vmID := vars["id"]

	var req SnapshotRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "INVALID_JSON", "Invalid JSON format")
		return
	}

	// Validate request
	if req.Name == "" {
		writeError(w, http.StatusBadRequest, "MISSING_NAME", "Snapshot name is required")
		return
	}

	// Check if VM exists
	_, err := h.vmManager.GetVM(r.Context(), vmID)
	if err != nil {
		if err == vm.ErrVMNotFound {
			writeError(w, http.StatusNotFound, "VM_NOT_FOUND", "VM not found")
			return
		}
		writeError(w, http.StatusInternalServerError, "INTERNAL_ERROR", "Failed to retrieve VM information")
		return
	}

	// Create snapshot asynchronously
	snapshotID, err := h.snapshotManager.CreateSnapshot(r.Context(), vmID, vm.SnapshotSpec{
		Name:        req.Name,
		Description: req.Description,
		Memory:      req.Memory,
		Quiesce:     req.Quiesce,
	})
	if err != nil {
		writeError(w, http.StatusInternalServerError, "SNAPSHOT_ERROR", "Failed to create snapshot")
		return
	}

	writeJSON(w, http.StatusAccepted, map[string]interface{}{
		"snapshot_id": snapshotID,
		"status":      "creating",
		"message":     "Snapshot creation initiated",
	})
}

// ListSnapshots handles GET /api/v1/vms/{id}/snapshots
// @Summary List VM snapshots
// @Description List all snapshots for a VM with pagination support
// @Tags VMs
// @Produce json
// @Param id path string true "VM ID"
// @Param page query int false "Page number" default(1)
// @Param limit query int false "Items per page" default(20)
// @Success 200 {object} map[string]interface{}
// @Failure 400 {object} map[string]interface{} "Invalid request"
// @Failure 401 {object} map[string]interface{} "Unauthorized"
// @Failure 404 {object} map[string]interface{} "VM not found"
// @Failure 500 {object} map[string]interface{} "Internal server error"
// @Router /api/v1/vms/{id}/snapshots [get]
func (h *ExtendedHandler) ListSnapshots(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	vmID := vars["id"]

	// Parse pagination parameters
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

	// Check if VM exists
	_, err := h.vmManager.GetVM(r.Context(), vmID)
	if err != nil {
		if err == vm.ErrVMNotFound {
			writeError(w, http.StatusNotFound, "VM_NOT_FOUND", "VM not found")
			return
		}
		writeError(w, http.StatusInternalServerError, "INTERNAL_ERROR", "Failed to retrieve VM information")
		return
	}

	// Get snapshots with pagination
	snapshots, total, err := h.snapshotManager.ListSnapshotsPaginated(r.Context(), vmID, page, limit)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "SNAPSHOT_ERROR", "Failed to retrieve snapshots")
		return
	}

	response := map[string]interface{}{
		"snapshots": snapshots,
		"pagination": map[string]interface{}{
			"page":       page,
			"limit":      limit,
			"total":      total,
			"total_pages": (total + limit - 1) / limit,
		},
	}

	writeJSON(w, http.StatusOK, response)
}

// CloneVM handles POST /api/v1/vms/{id}/clone
// @Summary Clone VM
// @Description Create a clone of the VM with new configuration
// @Tags VMs
// @Accept json
// @Produce json
// @Param id path string true "VM ID"
// @Param request body CloneRequest true "Clone request"
// @Success 202 {object} map[string]interface{}
// @Failure 400 {object} map[string]interface{} "Invalid request"
// @Failure 401 {object} map[string]interface{} "Unauthorized"
// @Failure 404 {object} map[string]interface{} "VM not found"
// @Failure 500 {object} map[string]interface{} "Internal server error"
// @Router /api/v1/vms/{id}/clone [post]
func (h *ExtendedHandler) CloneVM(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	sourceVMID := vars["id"]

	var req CloneRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "INVALID_JSON", "Invalid JSON format")
		return
	}

	// Validate request
	if req.Name == "" {
		writeError(w, http.StatusBadRequest, "MISSING_NAME", "Clone name is required")
		return
	}

	// Check if source VM exists
	sourceVM, err := h.vmManager.GetVM(r.Context(), sourceVMID)
	if err != nil {
		if err == vm.ErrVMNotFound {
			writeError(w, http.StatusNotFound, "VM_NOT_FOUND", "Source VM not found")
			return
		}
		writeError(w, http.StatusInternalServerError, "INTERNAL_ERROR", "Failed to retrieve source VM information")
		return
	}

	// Check if name is already in use
	if exists, _ := h.vmManager.VMExists(r.Context(), req.Name); exists {
		writeError(w, http.StatusConflict, "NAME_IN_USE", "VM name already exists")
		return
	}

	// Create clone asynchronously
	cloneID, err := h.vmManager.CloneVM(r.Context(), sourceVMID, vm.CloneSpec{
		Name:        req.Name,
		Description: req.Description,
		LinkedClone: req.LinkedClone,
		PowerOn:     req.PowerOn,
		Network:     req.Network,
		Tags:        req.Tags,
	})
	if err != nil {
		writeError(w, http.StatusInternalServerError, "CLONE_ERROR", "Failed to initiate VM clone")
		return
	}

	writeJSON(w, http.StatusAccepted, map[string]interface{}{
		"clone_id":    cloneID,
		"name":        req.Name,
		"source_vm":   sourceVM.Name,
		"status":      "cloning",
		"message":     "VM clone initiated",
		"linked_clone": req.LinkedClone,
	})
}

// Helper functions

func (h *ExtendedHandler) validateVMSpec(spec VMCreateSpec) error {
	if spec.Name == "" {
		return fmt.Errorf("name is required")
	}
	if spec.Template == "" {
		return fmt.Errorf("template is required")
	}
	if spec.CPU < 1 || spec.CPU > 64 {
		return fmt.Errorf("CPU must be between 1 and 64")
	}
	if spec.Memory < 512 || spec.Memory > 131072 {
		return fmt.Errorf("memory must be between 512MB and 128GB")
	}
	if spec.Disk < 10 || spec.Disk > 10240 {
		return fmt.Errorf("disk must be between 10GB and 10TB")
	}
	return nil
}

func (h *ExtendedHandler) generateOperationID() string {
	return fmt.Sprintf("bulk-%d", time.Now().Unix())
}

func (h *ExtendedHandler) getBaseURL(r *http.Request) string {
	scheme := "http"
	if r.TLS != nil {
		scheme = "https"
	}
	return fmt.Sprintf("%s://%s", scheme, r.Host)
}

func (h *ExtendedHandler) processBulkCreate(ctx context.Context, operationID string, vmSpecs []VMCreateSpec, response *BulkOperationResponse) {
	// Implementation would process VMs in parallel with proper error handling
	// This is a placeholder for the actual bulk creation logic
}

func (h *ExtendedHandler) processBulkDelete(ctx context.Context, operationID string, vmIDs []string, force bool, response *BulkOperationResponse) {
	// Implementation would process deletions in parallel with proper error handling
	// This is a placeholder for the actual bulk deletion logic
}