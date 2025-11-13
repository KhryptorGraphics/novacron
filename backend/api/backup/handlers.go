package backup

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/gorilla/mux"
	"github.com/khryptorgraphics/novacron/backend/core/backup"
)

// BackupAPIServer provides REST endpoints for backup operations
type BackupAPIServer struct {
	backupManager   *backup.BackupManager
	retentionManager *backup.RetentionManager
	restoreManager  *backup.RestoreManager
	router          *mux.Router
}

// BackupCreateRequest represents a backup creation request
type BackupCreateRequest struct {
	VMID        string            `json:"vm_id"`
	VMPath      string            `json:"vm_path"`
	BackupType  string            `json:"backup_type"`
	PolicyID    string            `json:"policy_id,omitempty"`
	Metadata    map[string]string `json:"metadata,omitempty"`
}

// BackupCreateResponse represents a backup creation response
type BackupCreateResponse struct {
	BackupID    string            `json:"backup_id"`
	VMID        string            `json:"vm_id"`
	Type        string            `json:"type"`
	Status      string            `json:"status"`
	CreatedAt   time.Time         `json:"created_at"`
	Metadata    map[string]string `json:"metadata,omitempty"`
}

// BackupListResponse represents a backup list response
type BackupListResponse struct {
	Backups []BackupInfo `json:"backups"`
	Total   int          `json:"total"`
}

// BackupInfo represents backup information in API responses
type BackupInfo struct {
	ID              string            `json:"id"`
	VMID            string            `json:"vm_id"`
	Type            string            `json:"type"`
	Size            int64             `json:"size"`
	CompressedSize  int64             `json:"compressed_size"`
	CreatedAt       time.Time         `json:"created_at"`
	ParentID        string            `json:"parent_id,omitempty"`
	BlockCount      int64             `json:"block_count"`
	ChangedBlocks   int64             `json:"changed_blocks"`
	CompressionRatio float64          `json:"compression_ratio"`
	Metadata        map[string]string `json:"metadata,omitempty"`
}

// BackupVerificationResponse represents a backup verification response for the backup API
type BackupVerificationResponse struct {
	BackupID         string                 `json:"backup_id"`
	Status           string                 `json:"status"`
	CheckedItems     int                    `json:"checked_items"`
	ErrorsFound      []string               `json:"errors_found"`
	VerificationTime time.Time              `json:"verification_time"`
	Details          map[string]interface{} `json:"details,omitempty"`
}

// RestoreCreateRequest represents a restore request
type RestoreCreateRequest struct {
	BackupID        string            `json:"backup_id"`
	TargetPath      string            `json:"target_path"`
	RestoreType     string            `json:"restore_type,omitempty"`
	PointInTime     *time.Time        `json:"point_in_time,omitempty"`
	SelectiveFiles  []string          `json:"selective_files,omitempty"`
	VerifyRestore   bool              `json:"verify_restore"`
	OverwriteExisting bool            `json:"overwrite_existing"`
	Metadata        map[string]string `json:"metadata,omitempty"`
}

// RestoreStatusResponse represents a restore status response
type RestoreStatusResponse struct {
	ID             string    `json:"id"`
	BackupID       string    `json:"backup_id"`
	VMID           string    `json:"vm_id"`
	Status         string    `json:"status"`
	Progress       int       `json:"progress"`
	StartedAt      time.Time `json:"started_at"`
	CompletedAt    *time.Time `json:"completed_at,omitempty"`
	TotalBytes     int64     `json:"total_bytes"`
	RestoredBytes  int64     `json:"restored_bytes"`
	Error          string    `json:"error,omitempty"`
}

// RetentionPolicyRequest represents a retention policy request
type RetentionPolicyRequest struct {
	Name        string                    `json:"name"`
	Description string                    `json:"description"`
	Rules       *backup.RetentionRules    `json:"rules"`
	GFSConfig   *backup.GFSConfig         `json:"gfs_config,omitempty"`
	Enabled     bool                      `json:"enabled"`
	Metadata    map[string]string         `json:"metadata,omitempty"`
}

// NewBackupAPIServer creates a new backup API server
func NewBackupAPIServer(
	backupManager *backup.BackupManager,
	retentionManager *backup.RetentionManager,
	restoreManager *backup.RestoreManager,
) *BackupAPIServer {
	server := &BackupAPIServer{
		backupManager:    backupManager,
		retentionManager: retentionManager,
		restoreManager:   restoreManager,
		router:           mux.NewRouter(),
	}
	
	server.setupRoutes()
	return server
}

// setupRoutes sets up the API routes
func (s *BackupAPIServer) setupRoutes() {
	api := s.router.PathPrefix("/api/v1/backup").Subrouter()
	
	// Backup operations
	api.HandleFunc("/backups", s.createBackup).Methods("POST")
	api.HandleFunc("/backups", s.listBackups).Methods("GET")
	api.HandleFunc("/backups/{backup_id}", s.getBackup).Methods("GET")
	api.HandleFunc("/backups/{backup_id}", s.deleteBackup).Methods("DELETE")
	api.HandleFunc("/backups/{backup_id}/verify", s.verifyBackup).Methods("POST")
	
	// VM-specific backup operations
	api.HandleFunc("/vms/{vm_id}/backups", s.listVMBackups).Methods("GET")
	api.HandleFunc("/vms/{vm_id}/backups", s.createVMBackup).Methods("POST")
	api.HandleFunc("/vms/{vm_id}/cbt/init", s.initializeCBT).Methods("POST")
	api.HandleFunc("/vms/{vm_id}/cbt/stats", s.getCBTStats).Methods("GET")
	
	// Restore operations
	api.HandleFunc("/restore", s.createRestore).Methods("POST")
	api.HandleFunc("/restore/{restore_id}", s.getRestoreStatus).Methods("GET")
	api.HandleFunc("/restore/{restore_id}", s.cancelRestore).Methods("DELETE")
	api.HandleFunc("/restore/operations", s.listRestoreOperations).Methods("GET")
	api.HandleFunc("/restore/point-in-time", s.restorePointInTime).Methods("POST")
	api.HandleFunc("/restore/{restore_id}/verify", s.verifyRestore).Methods("POST")
	
	// Retention operations
	api.HandleFunc("/retention/policies", s.createRetentionPolicy).Methods("POST")
	api.HandleFunc("/retention/policies", s.listRetentionPolicies).Methods("GET")
	api.HandleFunc("/retention/policies/{policy_id}", s.getRetentionPolicy).Methods("GET")
	api.HandleFunc("/retention/policies/{policy_id}", s.updateRetentionPolicy).Methods("PUT")
	api.HandleFunc("/retention/policies/{policy_id}", s.deleteRetentionPolicy).Methods("DELETE")
	api.HandleFunc("/retention/apply", s.applyRetention).Methods("POST")
	api.HandleFunc("/retention/jobs", s.listRetentionJobs).Methods("GET")
	api.HandleFunc("/retention/jobs/{job_id}", s.getRetentionJob).Methods("GET")
	
	// Recovery testing
	api.HandleFunc("/recovery/test", s.testRecovery).Methods("POST")
	
	// Monitoring and statistics
	api.HandleFunc("/stats", s.getBackupStats).Methods("GET")
	api.HandleFunc("/health", s.getHealthStatus).Methods("GET")
	api.HandleFunc("/dedup/stats", s.getDedupStats).Methods("GET")
}

// Router returns the API router
func (s *BackupAPIServer) Router() *mux.Router {
	return s.router
}

// Backup operation handlers

func (s *BackupAPIServer) createBackup(w http.ResponseWriter, r *http.Request) {
	var req BackupCreateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid JSON: %v", err), http.StatusBadRequest)
		return
	}
	
	// Validate request
	if req.VMID == "" {
		http.Error(w, "VM ID is required", http.StatusBadRequest)
		return
	}
	if req.VMPath == "" {
		http.Error(w, "VM path is required", http.StatusBadRequest)
		return
	}
	if req.BackupType == "" {
		req.BackupType = string(backup.IncrementalBackup)
	}
	
	// Parse backup type
	var backupType backup.BackupType
	switch strings.ToLower(req.BackupType) {
	case "full":
		backupType = backup.FullBackup
	case "incremental":
		backupType = backup.IncrementalBackup
	case "differential":
		backupType = backup.DifferentialBackup
	default:
		http.Error(w, fmt.Sprintf("Invalid backup type: %s", req.BackupType), http.StatusBadRequest)
		return
	}
	
	// Ensure VM ID is in metadata
	if req.Metadata == nil {
		req.Metadata = make(map[string]string)
	}
	req.Metadata["vm_id"] = req.VMID
	
	// Create backup job first
	job := &backup.BackupJob{
		ID:      fmt.Sprintf("job-%d", time.Now().Unix()),
		Name:    fmt.Sprintf("Backup for %s", req.VMID),
		Type:    backupType,
		Targets: []*backup.BackupTarget{{ID: req.VMID, ResourceID: req.VMID, Type: "vm"}},
		Storage: &backup.StorageConfig{Type: backup.LocalStorage},
		Enabled: true,
		TenantID: "default",
		Metadata: req.Metadata,  // Pass metadata including vm_id
	}
	
	// Register the job first
	if err := s.backupManager.CreateBackupJob(job); err != nil {
		http.Error(w, fmt.Sprintf("Failed to register backup job: %v", err), http.StatusInternalServerError)
		return
	}
	
	// Now run the registered job
	backupResult, err := s.backupManager.RunBackupJob(r.Context(), job.ID)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to create backup: %v", err), http.StatusInternalServerError)
		return
	}
	
	// Prepare response
	response := BackupCreateResponse{
		BackupID:  backupResult.ID,
		VMID:      req.VMID,
		Type:      string(backupResult.Type),
		Status:    string(backupResult.State),
		CreatedAt: backupResult.StartedAt,
		Metadata:  backupResult.Metadata,
	}
	
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(response)
}

func (s *BackupAPIServer) listBackups(w http.ResponseWriter, r *http.Request) {
	// Parse query parameters with pagination and filtering
	vmID := r.URL.Query().Get("vm_id")
	backupType := r.URL.Query().Get("type")
	state := r.URL.Query().Get("state")
	jobID := r.URL.Query().Get("job_id")
	startDateStr := r.URL.Query().Get("start_date")
	endDateStr := r.URL.Query().Get("end_date")
	limitStr := r.URL.Query().Get("limit")
	offsetStr := r.URL.Query().Get("offset")
	
	// Default pagination values
	limit := 50
	offset := 0
	
	// Parse pagination parameters
	if limitStr != "" {
		if parsedLimit, err := strconv.Atoi(limitStr); err == nil && parsedLimit > 0 {
			limit = parsedLimit
			if limit > 100 {
				limit = 100 // Max limit
			}
		}
	}
	if offsetStr != "" {
		if parsedOffset, err := strconv.Atoi(offsetStr); err == nil && parsedOffset >= 0 {
			offset = parsedOffset
		}
	}
	
	// Parse date filters
	var startDate, endDate time.Time
	var hasStartDate, hasEndDate bool
	if startDateStr != "" {
		if parsed, err := time.Parse(time.RFC3339, startDateStr); err == nil {
			startDate = parsed
			hasStartDate = true
		}
	}
	if endDateStr != "" {
		if parsed, err := time.Parse(time.RFC3339, endDateStr); err == nil {
			endDate = parsed
			hasEndDate = true
		}
	}
	
	// Check if any filters are provided
	hasFilters := vmID != "" || backupType != "" || state != "" || jobID != "" || hasStartDate || hasEndDate
	
	var backupList []backup.BackupInfo
	var err error
	
	if !hasFilters {
		// No filters provided, use ListAllBackups for efficiency
		backupList, err = s.backupManager.ListAllBackups(r.Context())
	} else {
		// Filters provided, use filtered listing
		filter := backup.BackupFilter{
			VMID:  vmID,
			Type:  backupType,
			State: state,
			JobID: jobID,
		}
		if hasStartDate {
			filter.StartDate = startDate
		}
		if hasEndDate {
			filter.EndDate = endDate
		}
		
		// Use manager-side filtering for filtered requests
		filteredBackups, err := s.backupManager.ListBackupsFiltered(r.Context(), filter)
		if err == nil {
			// Convert from []*Backup to []BackupInfo
			for _, b := range filteredBackups {
				vmID := b.VMID
				if vmID == "" && b.Metadata != nil {
					vmID = b.Metadata["vm_id"]
				}
				
				backupInfo := backup.BackupInfo{
					ID:        b.ID,
					JobID:     b.JobID,
					VMID:      vmID,
					Type:      b.Type,
					State:     b.State,
					Size:      b.Size,
					StartedAt: b.StartedAt,
					TenantID:  b.TenantID,
					ParentID:  b.ParentID,
					Metadata:  b.Metadata,
				}
				if !b.CompletedAt.IsZero() {
					backupInfo.CompletedAt = &b.CompletedAt
				}
				backupList = append(backupList, backupInfo)
			}
		}
	}
	
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to list backups: %v", err), http.StatusInternalServerError)
		return
	}
	
	// Convert to API response format
	var allBackups []BackupInfo
	for _, backupInfo := range backupList {
		compressionRatio := float64(1.0)
		compressedSize := backupInfo.Size // Placeholder - actual compression would be stored separately
		
		// Use VMID from backup info, or from metadata if available
		resultVMID := backupInfo.VMID
		if resultVMID == "" && backupInfo.Metadata != nil {
			resultVMID = backupInfo.Metadata["vm_id"]
		}
		
		apiBackupInfo := BackupInfo{
			ID:               backupInfo.ID,
			VMID:             resultVMID,
			Type:             string(backupInfo.Type),
			Size:             backupInfo.Size,
			CompressedSize:   compressedSize,
			CreatedAt:        backupInfo.StartedAt,
			ParentID:         backupInfo.ParentID,
			BlockCount:       0, // Not available in basic BackupInfo
			ChangedBlocks:    0, // Not available in basic BackupInfo
			CompressionRatio: compressionRatio,
			Metadata:         backupInfo.Metadata,
		}
		
		allBackups = append(allBackups, apiBackupInfo)
	}
	
	// Apply pagination
	total := len(allBackups)
	if offset >= total {
		allBackups = []BackupInfo{}
	} else {
		end := offset + limit
		if end > total {
			end = total
		}
		allBackups = allBackups[offset:end]
	}
	
	// Include pagination metadata in response
	response := struct {
		Backups []BackupInfo `json:"backups"`
		Total   int          `json:"total"`
		Limit   int          `json:"limit"`
		Offset  int          `json:"offset"`
	}{
		Backups: allBackups,
		Total:   total,
		Limit:   limit,
		Offset:  offset,
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (s *BackupAPIServer) getBackup(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	backupID := vars["backup_id"]
	
	// Get backup manifest to retrieve proper VM ID
	manifest, err := s.backupManager.GetBackupManifest(backupID)
	if err != nil {
		if errors.Is(err, backup.ErrBackupNotFound) {
			http.Error(w, fmt.Sprintf("Backup not found: %v", err), http.StatusNotFound)
		} else {
			http.Error(w, fmt.Sprintf("Failed to get backup manifest: %v", err), http.StatusInternalServerError)
		}
		return
	}
	
	backup, err := s.backupManager.GetBackup(backupID)
	if err != nil {
		if errors.Is(err, backup.ErrBackupNotFound) {
			http.Error(w, fmt.Sprintf("Backup not found: %v", err), http.StatusNotFound)
		} else {
			http.Error(w, fmt.Sprintf("Failed to get backup: %v", err), http.StatusInternalServerError)
		}
		return
	}
	
	compressionRatio := float64(1.0)
	compressedSize := backup.Size // Placeholder
	
	backupInfo := BackupInfo{
		ID:               backup.ID,
		VMID:             manifest.VMID, // Use VMID from manifest, not JobID
		Type:             string(backup.Type),
		Size:             backup.Size,
		CompressedSize:   compressedSize,
		CreatedAt:        backup.StartedAt,
		ParentID:         backup.ParentID,
		BlockCount:       0, // Not available in basic Backup
		ChangedBlocks:    0, // Not available in basic Backup
		CompressionRatio: compressionRatio,
		Metadata:         backup.Metadata,
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(backupInfo)
}

func (s *BackupAPIServer) deleteBackup(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	backupID := vars["backup_id"]
	
	// Delete backup through backup manager
	err := s.backupManager.DeleteBackup(r.Context(), backupID)
	if err != nil {
		// Use typed error checking
		if errors.Is(err, backup.ErrBackupNotFound) {
			http.Error(w, fmt.Sprintf("Backup not found: %v", err), http.StatusNotFound)
		} else {
			http.Error(w, fmt.Sprintf("Failed to delete backup: %v", err), http.StatusInternalServerError)
		}
		return
	}
	
	w.WriteHeader(http.StatusNoContent)
}

func (s *BackupAPIServer) verifyBackup(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	backupID := vars["backup_id"]
	
	// Verify backup through backup manager using VerifyBackup
	result, err := s.backupManager.VerifyBackup(r.Context(), backupID)
	if err != nil {
		// Use typed error checking
		if errors.Is(err, backup.ErrBackupNotFound) {
			http.Error(w, fmt.Sprintf("Backup not found: %v", err), http.StatusNotFound)
		} else {
			http.Error(w, fmt.Sprintf("Failed to verify backup: %v", err), http.StatusInternalServerError)
		}
		return
	}
	
	// Convert to typed response
	response := BackupVerificationResponse{
		BackupID:         result.BackupID,
		Status:           result.Status,
		CheckedItems:     result.CheckedItems,
		ErrorsFound:      result.ErrorsFound,
		VerificationTime: result.VerificationTime,
		Details:          result.Details,
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (s *BackupAPIServer) listVMBackups(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	vmID := vars["vm_id"]
	
	// Redirect to general list backups with VM ID filter
	r.URL.RawQuery = fmt.Sprintf("vm_id=%s", vmID)
	s.listBackups(w, r)
}

func (s *BackupAPIServer) createVMBackup(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	vmID := vars["vm_id"]
	
	var req BackupCreateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid JSON: %v", err), http.StatusBadRequest)
		return
	}
	
	// Override VM ID from URL
	req.VMID = vmID
	
	// Re-encode and call createBackup
	r.Body = io.NopCloser(jsonReader(req))
	s.createBackup(w, r)
}

func (s *BackupAPIServer) initializeCBT(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	vmID := vars["vm_id"]
	
	// Parse request body for VM size
	var req struct {
		VMSize int64 `json:"vm_size"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid JSON: %v", err), http.StatusBadRequest)
		return
	}
	
	if req.VMSize <= 0 {
		http.Error(w, "Valid VM size is required", http.StatusBadRequest)
		return
	}
	
	// Initialize CBT
	tracker, err := s.backupManager.InitializeCBT(vmID, req.VMSize)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to initialize CBT: %v", err), http.StatusInternalServerError)
		return
	}
	
	response := map[string]interface{}{
		"vm_id":        tracker.VMID(),
		"total_blocks": tracker.TotalBlocks(),
		"block_size":   tracker.BlockSize(),
		"initialized":  true,
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (s *BackupAPIServer) getCBTStats(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	vmID := vars["vm_id"]
	
	stats, err := s.backupManager.GetCBTStats(vmID)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to get CBT stats: %v", err), http.StatusNotFound)
		return
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(stats)
}

// Restore operation handlers

func (s *BackupAPIServer) createRestore(w http.ResponseWriter, r *http.Request) {
	var req RestoreCreateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid JSON: %v", err), http.StatusBadRequest)
		return
	}
	
	// Validate request
	if req.BackupID == "" {
		http.Error(w, "Backup ID is required", http.StatusBadRequest)
		return
	}
	if req.TargetPath == "" {
		http.Error(w, "Target path is required", http.StatusBadRequest)
		return
	}
	
	// Get backup manifest to determine VM ID
	manifest, err := s.backupManager.GetBackupManifest(req.BackupID)
	if err != nil {
		http.Error(w, fmt.Sprintf("Backup not found: %v", err), http.StatusNotFound)
		return
	}
	
	// Create restore request
	restoreReq := &backup.RestoreRequest{
		VMID:           manifest.VMID,
		BackupID:       req.BackupID,
		RestoreType:    req.RestoreType,
		TargetPath:     req.TargetPath,
		SelectiveFiles: req.SelectiveFiles,
		Options: backup.RestoreOptions{
			VerifyRestore:       req.VerifyRestore,
			OverwriteExisting:   req.OverwriteExisting,
			EnableDecompression: true,
			CreateTargetDir:     true,
		},
		Metadata: req.Metadata,
	}
	
	if req.PointInTime != nil {
		restoreReq.PointInTime = *req.PointInTime
	}
	
	// Create restore operation
	operation, err := s.restoreManager.CreateRestoreOperation(restoreReq)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to create restore operation: %v", err), http.StatusInternalServerError)
		return
	}
	
	response := RestoreStatusResponse{
		ID:            operation.ID,
		BackupID:      operation.BackupID,
		VMID:          operation.VMID,
		Status:        operation.Status,
		Progress:      operation.Progress,
		StartedAt:     operation.StartedAt,
		TotalBytes:    operation.TotalBytes,
		RestoredBytes: operation.RestoredBytes,
	}
	
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(response)
}

func (s *BackupAPIServer) getRestoreStatus(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	restoreID := vars["restore_id"]
	
	operation, err := s.restoreManager.GetRestoreOperation(restoreID)
	if err != nil {
		http.Error(w, fmt.Sprintf("Restore operation not found: %v", err), http.StatusNotFound)
		return
	}
	
	response := RestoreStatusResponse{
		ID:            operation.ID,
		BackupID:      operation.BackupID,
		VMID:          operation.VMID,
		Status:        operation.Status,
		Progress:      operation.Progress,
		StartedAt:     operation.StartedAt,
		TotalBytes:    operation.TotalBytes,
		RestoredBytes: operation.RestoredBytes,
		Error:         operation.Error,
	}
	
	if !operation.CompletedAt.IsZero() {
		response.CompletedAt = &operation.CompletedAt
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (s *BackupAPIServer) cancelRestore(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	restoreID := vars["restore_id"]
	
	err := s.restoreManager.CancelRestoreOperation(restoreID)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to cancel restore: %v", err), http.StatusInternalServerError)
		return
	}
	
	w.WriteHeader(http.StatusNoContent)
}

func (s *BackupAPIServer) listRestoreOperations(w http.ResponseWriter, r *http.Request) {
	operations := s.restoreManager.ListRestoreOperations()
	
	var responses []RestoreStatusResponse
	for _, op := range operations {
		response := RestoreStatusResponse{
			ID:            op.ID,
			BackupID:      op.BackupID,
			VMID:          op.VMID,
			Status:        op.Status,
			Progress:      op.Progress,
			StartedAt:     op.StartedAt,
			TotalBytes:    op.TotalBytes,
			RestoredBytes: op.RestoredBytes,
			Error:         op.Error,
		}
		
		if !op.CompletedAt.IsZero() {
			response.CompletedAt = &op.CompletedAt
		}
		
		responses = append(responses, response)
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"operations": responses,
		"total":      len(responses),
	})
}

func (s *BackupAPIServer) restorePointInTime(w http.ResponseWriter, r *http.Request) {
	var req struct {
		VMID        string    `json:"vm_id"`
		PointInTime time.Time `json:"point_in_time"`
		TargetPath  string    `json:"target_path"`
	}
	
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid JSON: %v", err), http.StatusBadRequest)
		return
	}
	
	if req.VMID == "" {
		http.Error(w, "VM ID is required", http.StatusBadRequest)
		return
	}
	if req.TargetPath == "" {
		http.Error(w, "Target path is required", http.StatusBadRequest)
		return
	}
	if req.PointInTime.IsZero() {
		http.Error(w, "Point in time is required", http.StatusBadRequest)
		return
	}
	
	operation, err := s.restoreManager.RestoreFromPointInTime(req.VMID, req.PointInTime, req.TargetPath)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to create point-in-time restore: %v", err), http.StatusInternalServerError)
		return
	}
	
	response := RestoreStatusResponse{
		ID:            operation.ID,
		BackupID:      operation.BackupID,
		VMID:          operation.VMID,
		Status:        operation.Status,
		Progress:      operation.Progress,
		StartedAt:     operation.StartedAt,
		TotalBytes:    operation.TotalBytes,
		RestoredBytes: operation.RestoredBytes,
	}
	
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(response)
}

func (s *BackupAPIServer) verifyRestore(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	restoreID := vars["restore_id"]
	
	result, err := s.restoreManager.ValidateRestore(restoreID)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to verify restore: %v", err), http.StatusInternalServerError)
		return
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(result)
}

// Retention policy handlers

func (s *BackupAPIServer) createRetentionPolicy(w http.ResponseWriter, r *http.Request) {
	var req RetentionPolicyRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid JSON: %v", err), http.StatusBadRequest)
		return
	}
	
	if req.Name == "" {
		http.Error(w, "Policy name is required", http.StatusBadRequest)
		return
	}
	
	policy := &backup.RetentionPolicy{
		ID:          fmt.Sprintf("policy-%d", time.Now().Unix()),
		Name:        req.Name,
		Description: req.Description,
		Rules:       req.Rules,
		GFSConfig:   req.GFSConfig,
		Enabled:     req.Enabled,
		Metadata:    req.Metadata,
	}
	
	if err := s.retentionManager.CreatePolicy(policy); err != nil {
		http.Error(w, fmt.Sprintf("Failed to create policy: %v", err), http.StatusInternalServerError)
		return
	}
	
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(policy)
}

func (s *BackupAPIServer) listRetentionPolicies(w http.ResponseWriter, r *http.Request) {
	policies := s.retentionManager.ListPolicies()
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"policies": policies,
		"total":    len(policies),
	})
}

func (s *BackupAPIServer) getRetentionPolicy(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	policyID := vars["policy_id"]
	
	policy, err := s.retentionManager.GetPolicy(policyID)
	if err != nil {
		http.Error(w, fmt.Sprintf("Policy not found: %v", err), http.StatusNotFound)
		return
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(policy)
}

func (s *BackupAPIServer) updateRetentionPolicy(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	policyID := vars["policy_id"]
	
	var req RetentionPolicyRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid JSON: %v", err), http.StatusBadRequest)
		return
	}
	
	policy := &backup.RetentionPolicy{
		ID:          policyID,
		Name:        req.Name,
		Description: req.Description,
		Rules:       req.Rules,
		GFSConfig:   req.GFSConfig,
		Enabled:     req.Enabled,
		Metadata:    req.Metadata,
	}
	
	if err := s.retentionManager.UpdatePolicy(policy); err != nil {
		http.Error(w, fmt.Sprintf("Failed to update policy: %v", err), http.StatusInternalServerError)
		return
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(policy)
}

func (s *BackupAPIServer) deleteRetentionPolicy(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	policyID := vars["policy_id"]
	
	if err := s.retentionManager.DeletePolicy(policyID); err != nil {
		http.Error(w, fmt.Sprintf("Failed to delete policy: %v", err), http.StatusInternalServerError)
		return
	}
	
	w.WriteHeader(http.StatusNoContent)
}

func (s *BackupAPIServer) applyRetention(w http.ResponseWriter, r *http.Request) {
	var req struct {
		VMID     string `json:"vm_id"`
		PolicyID string `json:"policy_id"`
	}
	
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid JSON: %v", err), http.StatusBadRequest)
		return
	}
	
	if req.VMID == "" {
		http.Error(w, "VM ID is required", http.StatusBadRequest)
		return
	}
	
	job, err := s.retentionManager.ApplyRetention(req.VMID, req.PolicyID)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to apply retention: %v", err), http.StatusInternalServerError)
		return
	}
	
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(job)
}

func (s *BackupAPIServer) listRetentionJobs(w http.ResponseWriter, r *http.Request) {
	jobs := s.retentionManager.ListRetentionJobs()
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"jobs":  jobs,
		"total": len(jobs),
	})
}

func (s *BackupAPIServer) getRetentionJob(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	jobID := vars["job_id"]
	
	job, err := s.retentionManager.GetRetentionJob(jobID)
	if err != nil {
		http.Error(w, fmt.Sprintf("Job not found: %v", err), http.StatusNotFound)
		return
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(job)
}

// Recovery testing handlers

func (s *BackupAPIServer) testRecovery(w http.ResponseWriter, r *http.Request) {
	var req struct {
		BackupID string `json:"backup_id"`
		TestType string `json:"test_type"`
	}
	
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid JSON: %v", err), http.StatusBadRequest)
		return
	}
	
	if req.BackupID == "" {
		http.Error(w, "Backup ID is required", http.StatusBadRequest)
		return
	}
	if req.TestType == "" {
		req.TestType = "basic"
	}
	
	result, err := s.restoreManager.TestRecovery(req.BackupID, req.TestType)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to test recovery: %v", err), http.StatusInternalServerError)
		return
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(result)
}

// Monitoring and statistics handlers

func (s *BackupAPIServer) getBackupStats(w http.ResponseWriter, r *http.Request) {
	// Return placeholder statistics until comprehensive metrics are implemented
	// In production, this would aggregate data from backup manager, retention policies, and storage
	stats := map[string]interface{}{
		"total_backups":   0,
		"total_size":      0,
		"compression_ratio": 0.0,
		"deduplication_ratio": 0.0,
		"backup_success_rate": 0.0,
		"average_backup_time": "0s",
		"status": "placeholder", // Indicates this is not yet fully implemented
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(stats); err != nil {
		http.Error(w, fmt.Sprintf("Failed to encode statistics: %v", err), http.StatusInternalServerError)
		return
	}
}

func (s *BackupAPIServer) getHealthStatus(w http.ResponseWriter, r *http.Request) {
	health := map[string]interface{}{
		"status": "healthy",
		"components": map[string]string{
			"backup_manager":    "healthy",
			"retention_manager": "healthy",
			"restore_manager":   "healthy",
		},
		"timestamp": time.Now(),
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(health)
}

func (s *BackupAPIServer) getDedupStats(w http.ResponseWriter, r *http.Request) {
	// Return placeholder deduplication statistics
	// In production, this would query the actual deduplication engine
	stats := map[string]interface{}{
		"total_bytes":        0,
		"unique_bytes":       0,
		"deduplicated_bytes": 0,
		"compression_ratio":  1.0,
		"chunk_count":        0,
		"unique_chunks":      0,
		"last_updated":       time.Now(),
		"status": "placeholder", // Indicates this is not yet connected to dedup engine
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(stats)
}

// Helper functions

func jsonReader(v interface{}) io.Reader {
	data, _ := json.Marshal(v)
	return strings.NewReader(string(data))
}

// Additional helper methods would be added for CBTTracker interface implementation