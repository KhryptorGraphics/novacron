package backup

import (
	"context"
	"encoding/json"
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
	backupManager   *backup.IncrementalBackupManager
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
	backupManager *backup.IncrementalBackupManager,
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
	
	// Create backup
	manifest, err := s.backupManager.CreateIncrementalBackup(r.Context(), req.VMID, req.VMPath, backupType)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to create backup: %v", err), http.StatusInternalServerError)
		return
	}
	
	// Prepare response
	response := BackupCreateResponse{
		BackupID:  manifest.BackupID,
		VMID:      manifest.VMID,
		Type:      string(manifest.Type),
		Status:    "completed",
		CreatedAt: manifest.CreatedAt,
		Metadata:  manifest.Metadata,
	}
	
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(response)
}

func (s *BackupAPIServer) listBackups(w http.ResponseWriter, r *http.Request) {
	// Parse query parameters
	vmID := r.URL.Query().Get("vm_id")
	
	var allBackups []BackupInfo
	
	if vmID != "" {
		// List backups for specific VM
		backupIDs, err := s.backupManager.ListBackups(vmID)
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to list backups: %v", err), http.StatusInternalServerError)
			return
		}
		
		for _, backupID := range backupIDs {
			manifest, err := s.backupManager.GetBackupManifest(backupID)
			if err != nil {
				continue // Skip invalid backups
			}
			
			compressionRatio := float64(1.0)
			if manifest.Size > 0 {
				compressionRatio = float64(manifest.CompressedSize) / float64(manifest.Size)
			}
			
			backupInfo := BackupInfo{
				ID:               manifest.BackupID,
				VMID:             manifest.VMID,
				Type:             string(manifest.Type),
				Size:             manifest.Size,
				CompressedSize:   manifest.CompressedSize,
				CreatedAt:        manifest.CreatedAt,
				ParentID:         manifest.ParentID,
				BlockCount:       manifest.BlockCount,
				ChangedBlocks:    manifest.ChangedBlocks,
				CompressionRatio: compressionRatio,
				Metadata:         manifest.Metadata,
			}
			
			allBackups = append(allBackups, backupInfo)
		}
	} else {
		// TODO: Implement listing all backups across all VMs
		http.Error(w, "Listing all backups not yet implemented", http.StatusNotImplemented)
		return
	}
	
	response := BackupListResponse{
		Backups: allBackups,
		Total:   len(allBackups),
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (s *BackupAPIServer) getBackup(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	backupID := vars["backup_id"]
	
	manifest, err := s.backupManager.GetBackupManifest(backupID)
	if err != nil {
		http.Error(w, fmt.Sprintf("Backup not found: %v", err), http.StatusNotFound)
		return
	}
	
	compressionRatio := float64(1.0)
	if manifest.Size > 0 {
		compressionRatio = float64(manifest.CompressedSize) / float64(manifest.Size)
	}
	
	backupInfo := BackupInfo{
		ID:               manifest.BackupID,
		VMID:             manifest.VMID,
		Type:             string(manifest.Type),
		Size:             manifest.Size,
		CompressedSize:   manifest.CompressedSize,
		CreatedAt:        manifest.CreatedAt,
		ParentID:         manifest.ParentID,
		BlockCount:       manifest.BlockCount,
		ChangedBlocks:    manifest.ChangedBlocks,
		CompressionRatio: compressionRatio,
		Metadata:         manifest.Metadata,
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(backupInfo)
}

func (s *BackupAPIServer) deleteBackup(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	backupID := vars["backup_id"]
	
	// TODO: Implement backup deletion
	http.Error(w, "Backup deletion not yet implemented", http.StatusNotImplemented)
}

func (s *BackupAPIServer) verifyBackup(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	backupID := vars["backup_id"]
	
	// TODO: Implement backup verification
	http.Error(w, "Backup verification not yet implemented", http.StatusNotImplemented)
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
	r.Body = jsonReader(req)
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
	// TODO: Implement comprehensive backup statistics
	stats := map[string]interface{}{
		"total_backups":   0,
		"total_size":      0,
		"compression_ratio": 0.0,
		"deduplication_ratio": 0.0,
		"backup_success_rate": 0.0,
		"average_backup_time": "0s",
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(stats)
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
	// Get deduplication statistics
	// TODO: Get actual stats from deduplication engine
	stats := map[string]interface{}{
		"total_bytes":        0,
		"unique_bytes":       0,
		"deduplicated_bytes": 0,
		"compression_ratio":  1.0,
		"chunk_count":        0,
		"unique_chunks":      0,
		"last_updated":       time.Now(),
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