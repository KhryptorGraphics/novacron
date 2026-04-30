package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	dr "github.com/khryptorgraphics/novacron/backend/core/dr"
	"github.com/khryptorgraphics/novacron/backend/core/vm"
)

type runtimeDRRuntime struct {
	orchestrator *dr.Orchestrator
	api          *dr.DRAPI
	started      bool
	vmResolver   runtimeDRVMResolver
	backupPath   string
	backupMu     sync.RWMutex
	backups      map[string]runtimeDRBackupRecord
	restoreMu    sync.RWMutex
	restores     map[string]runtimeDRRestoreRecord
}

type runtimeDRVMResolver interface {
	GetVM(vmID string) (*vm.VM, error)
}

type runtimeDRBackupRecord struct {
	BackupID   string            `json:"backup_id"`
	VMID       string            `json:"vm_id,omitempty"`
	Type       string            `json:"type"`
	Status     string            `json:"status"`
	Location   string            `json:"location,omitempty"`
	SizeBytes  int64             `json:"size_bytes,omitempty"`
	CreatedAt  time.Time         `json:"created_at"`
	VerifiedAt time.Time         `json:"verified_at,omitempty"`
	UpdatedAt  time.Time         `json:"updated_at"`
	Metadata   map[string]string `json:"metadata,omitempty"`
}

type runtimeDRBackupRequest struct {
	BackupID  string            `json:"backup_id"`
	VMID      string            `json:"vm_id,omitempty"`
	Type      string            `json:"type,omitempty"`
	Status    string            `json:"status,omitempty"`
	Location  string            `json:"location,omitempty"`
	SizeBytes int64             `json:"size_bytes,omitempty"`
	Metadata  map[string]string `json:"metadata,omitempty"`
}

type runtimeDRRestoreRequest struct {
	BackupID     string   `json:"backup_id"`
	VMID         string   `json:"vm_id"`
	TargetRegion string   `json:"target_region,omitempty"`
	Selective    []string `json:"selective,omitempty"`
}

type runtimeDRRestoreRecord struct {
	RestoreID    string    `json:"restore_id"`
	BackupID     string    `json:"backup_id"`
	VMID         string    `json:"vm_id"`
	TargetType   string    `json:"target_type"`
	TargetRegion string    `json:"target_region,omitempty"`
	VMState      string    `json:"vm_state"`
	Status       string    `json:"status"`
	CreatedAt    time.Time `json:"created_at"`
	UpdatedAt    time.Time `json:"updated_at"`
	Error        string    `json:"error,omitempty"`
}

type runtimeDRStatusResponse struct {
	Enabled          bool              `json:"enabled"`
	State            string            `json:"state"`
	PrimaryRegion    string            `json:"primary_region"`
	SecondaryRegions []string          `json:"secondary_regions"`
	ActiveFailovers  int               `json:"active_failovers"`
	LastFailover     time.Time         `json:"last_failover,omitempty"`
	LastBackup       time.Time         `json:"last_backup,omitempty"`
	HealthScore      float64           `json:"health_score"`
	RTOSeconds       int64             `json:"rto_seconds"`
	RPOSeconds       int64             `json:"rpo_seconds"`
	BackupCount      int64             `json:"backup_count"`
	RestoreCount     int64             `json:"restore_count"`
	BackupMetrics    dr.BackupMetrics  `json:"backup_metrics"`
	RestoreMetrics   dr.RestoreMetrics `json:"restore_metrics"`
}

func initializeRuntimeDR(config runtimeConfig, vmResolver runtimeDRVMResolver) (*runtimeDRRuntime, error) {
	if !runtimeServiceEnabled(config, "backup") {
		return nil, nil
	}

	drConfig := runtimeDRConfigFromRuntime(config)
	orchestrator, err := dr.NewOrchestrator(drConfig)
	if err != nil {
		return nil, fmt.Errorf("create DR orchestrator: %w", err)
	}
	if err := orchestrator.Start(); err != nil {
		return nil, fmt.Errorf("start DR orchestrator: %w", err)
	}

	runtime := &runtimeDRRuntime{
		orchestrator: orchestrator,
		api:          dr.NewDRAPI(orchestrator),
		started:      true,
		vmResolver:   vmResolver,
		backupPath:   runtimeDRBackupMetadataPath(config),
		backups:      make(map[string]runtimeDRBackupRecord),
		restores:     make(map[string]runtimeDRRestoreRecord),
	}
	if err := runtime.loadBackups(); err != nil {
		_ = runtime.Stop()
		return nil, err
	}
	return runtime, nil
}

func runtimeDRConfigFromRuntime(config runtimeConfig) *dr.DRConfig {
	drConfig := dr.DefaultDRConfig()
	policy := runtimeMobilityPolicyFromConfig(config)

	if policy.Backup.RPOSeconds > 0 {
		drConfig.RPO = time.Duration(policy.Backup.RPOSeconds) * time.Second
	}
	if policy.Recovery.RTOSeconds > 0 {
		drConfig.RTO = time.Duration(policy.Recovery.RTOSeconds) * time.Second
	}

	drConfig.BackupSchedule.FullBackup = ""
	drConfig.BackupSchedule.IncrementalBackup = ""
	drConfig.BackupSchedule.TransactionLog = false
	drConfig.TestingEnabled = false
	drConfig.ChaosEnabled = false

	backupRoot := filepath.Join(config.Storage.BasePath, "backups")
	drConfig.BackupLocations = []dr.BackupLocation{
		{
			ID:       "runtime-local",
			Type:     "local",
			Region:   drConfig.PrimaryRegion,
			Endpoint: backupRoot,
			Priority: 1,
			Metadata: map[string]string{
				"deployment_profile": config.Services.DeploymentProfile,
				"storage_base_path":  config.Storage.BasePath,
			},
		},
	}

	return drConfig
}

func runtimeDRBackupMetadataPath(config runtimeConfig) string {
	return filepath.Join(config.Storage.BasePath, "backups", "metadata.json")
}

func (runtime *runtimeDRRuntime) loadBackups() error {
	if runtime == nil || runtime.backupPath == "" {
		return nil
	}

	data, err := os.ReadFile(runtime.backupPath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return fmt.Errorf("read backup metadata: %w", err)
	}

	var records []runtimeDRBackupRecord
	if err := json.Unmarshal(data, &records); err != nil {
		return fmt.Errorf("parse backup metadata: %w", err)
	}

	runtime.backupMu.Lock()
	defer runtime.backupMu.Unlock()
	for _, record := range records {
		if strings.TrimSpace(record.BackupID) == "" {
			continue
		}
		runtime.backups[record.BackupID] = record
	}
	return nil
}

func (runtime *runtimeDRRuntime) saveBackupsLocked() error {
	if runtime == nil || runtime.backupPath == "" {
		return nil
	}

	records := runtime.listBackupsLocked()
	data, err := json.MarshalIndent(records, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal backup metadata: %w", err)
	}

	if err := os.MkdirAll(filepath.Dir(runtime.backupPath), 0o755); err != nil {
		return fmt.Errorf("create backup metadata directory: %w", err)
	}
	tempPath := runtime.backupPath + ".tmp"
	if err := os.WriteFile(tempPath, data, 0o600); err != nil {
		return fmt.Errorf("write backup metadata: %w", err)
	}
	if err := os.Rename(tempPath, runtime.backupPath); err != nil {
		return fmt.Errorf("replace backup metadata: %w", err)
	}
	return nil
}

func (runtime *runtimeDRRuntime) listBackupsLocked() []runtimeDRBackupRecord {
	records := make([]runtimeDRBackupRecord, 0, len(runtime.backups))
	for _, record := range runtime.backups {
		records = append(records, record)
	}
	sort.Slice(records, func(i, j int) bool {
		return records[i].UpdatedAt.After(records[j].UpdatedAt)
	})
	return records
}

func (runtime *runtimeDRRuntime) ListBackups() ([]runtimeDRBackupRecord, error) {
	if runtime == nil {
		return nil, fmt.Errorf("DR runtime is not initialized")
	}

	runtime.backupMu.RLock()
	defer runtime.backupMu.RUnlock()
	return runtime.listBackupsLocked(), nil
}

func (runtime *runtimeDRRuntime) RegisterBackup(request runtimeDRBackupRequest) (runtimeDRBackupRecord, error) {
	if runtime == nil {
		return runtimeDRBackupRecord{}, fmt.Errorf("DR runtime is not initialized")
	}

	backupID := strings.TrimSpace(request.BackupID)
	if backupID == "" {
		return runtimeDRBackupRecord{}, fmt.Errorf("backup_id is required")
	}

	now := time.Now().UTC()
	record := runtimeDRBackupRecord{
		BackupID:   backupID,
		VMID:       strings.TrimSpace(request.VMID),
		Type:       strings.TrimSpace(request.Type),
		Status:     strings.TrimSpace(request.Status),
		Location:   strings.TrimSpace(request.Location),
		SizeBytes:  request.SizeBytes,
		CreatedAt:  now,
		VerifiedAt: now,
		UpdatedAt:  now,
		Metadata:   request.Metadata,
	}
	if record.Type == "" {
		record.Type = "manual"
	}
	if record.Status == "" {
		record.Status = "verified"
	}

	runtime.backupMu.Lock()
	defer runtime.backupMu.Unlock()
	if existing, ok := runtime.backups[backupID]; ok {
		record.CreatedAt = existing.CreatedAt
	}
	runtime.backups[backupID] = record
	if err := runtime.saveBackupsLocked(); err != nil {
		return runtimeDRBackupRecord{}, err
	}
	return record, nil
}

func (runtime *runtimeDRRuntime) StartRestore(request runtimeDRRestoreRequest) (runtimeDRRestoreRecord, error) {
	if runtime == nil || runtime.api == nil {
		return runtimeDRRestoreRecord{}, fmt.Errorf("DR runtime is not initialized")
	}
	if runtime.vmResolver == nil {
		return runtimeDRRestoreRecord{}, fmt.Errorf("VM runtime is not initialized")
	}

	backupID := strings.TrimSpace(request.BackupID)
	if backupID == "" {
		return runtimeDRRestoreRecord{}, fmt.Errorf("backup_id is required")
	}
	vmID := strings.TrimSpace(request.VMID)
	if vmID == "" {
		return runtimeDRRestoreRecord{}, fmt.Errorf("vm_id is required")
	}

	runtime.backupMu.RLock()
	backupRecord, ok := runtime.backups[backupID]
	runtime.backupMu.RUnlock()
	if !ok {
		return runtimeDRRestoreRecord{}, fmt.Errorf("backup_id %q is not registered", backupID)
	}
	if !strings.EqualFold(backupRecord.Status, "verified") {
		return runtimeDRRestoreRecord{}, fmt.Errorf("backup_id %q is not verified", backupID)
	}
	if backupRecord.VMID != "" && backupRecord.VMID != vmID {
		return runtimeDRRestoreRecord{}, fmt.Errorf("backup_id %q belongs to vm_id %q", backupID, backupRecord.VMID)
	}

	vmInstance, err := runtime.vmResolver.GetVM(vmID)
	if err != nil {
		return runtimeDRRestoreRecord{}, fmt.Errorf("validate restore target VM: %w", err)
	}

	target := dr.RestoreTarget{
		Type:         "vm",
		TargetID:     vmID,
		TargetRegion: strings.TrimSpace(request.TargetRegion),
		Selective:    append([]string(nil), request.Selective...),
	}
	restoreID, err := runtime.api.RestoreFromBackupWithID(backupID, target)
	if err != nil {
		return runtimeDRRestoreRecord{}, err
	}

	now := time.Now().UTC()
	record := runtimeDRRestoreRecord{
		RestoreID:    restoreID,
		BackupID:     backupID,
		VMID:         vmID,
		TargetType:   target.Type,
		TargetRegion: target.TargetRegion,
		VMState:      string(vmInstance.State()),
		Status:       "running",
		CreatedAt:    now,
		UpdatedAt:    now,
	}

	runtime.restoreMu.Lock()
	runtime.restores[restoreID] = record
	runtime.restoreMu.Unlock()
	return record, nil
}

func (runtime *runtimeDRRuntime) ListRestores() ([]runtimeDRRestoreRecord, error) {
	if runtime == nil {
		return nil, fmt.Errorf("DR runtime is not initialized")
	}

	runtime.restoreMu.RLock()
	defer runtime.restoreMu.RUnlock()
	records := make([]runtimeDRRestoreRecord, 0, len(runtime.restores))
	for _, record := range runtime.restores {
		records = append(records, record)
	}
	sort.Slice(records, func(i, j int) bool {
		return records[i].UpdatedAt.After(records[j].UpdatedAt)
	})
	return records, nil
}

func (runtime *runtimeDRRuntime) Stop() error {
	if runtime == nil || runtime.orchestrator == nil || !runtime.started {
		return nil
	}
	runtime.started = false
	return runtime.orchestrator.Stop()
}

func (runtime *runtimeDRRuntime) Status() (runtimeDRStatusResponse, error) {
	if runtime == nil || runtime.api == nil {
		return runtimeDRStatusResponse{Enabled: false, State: runtimeServiceStateUnavailable}, fmt.Errorf("DR runtime is not initialized")
	}

	status, err := runtime.api.GetDRStatus()
	if err != nil {
		return runtimeDRStatusResponse{Enabled: true, State: runtimeServiceStateUnavailable}, err
	}

	return runtimeDRStatusResponse{
		Enabled:          true,
		State:            status.State.String(),
		PrimaryRegion:    status.PrimaryRegion,
		SecondaryRegions: append([]string(nil), status.SecondaryRegions...),
		ActiveFailovers:  status.ActiveFailovers,
		LastFailover:     status.LastFailover,
		LastBackup:       status.LastBackup,
		HealthScore:      status.HealthScore,
		RTOSeconds:       int64(status.RTO / time.Second),
		RPOSeconds:       int64(status.RPO / time.Second),
		BackupCount:      runtime.backupCount(status.BackupCount),
		RestoreCount:     runtime.restoreCount(status.RestoreCount),
		BackupMetrics:    runtime.api.GetBackupMetrics(),
		RestoreMetrics:   runtime.api.GetRestoreMetrics(),
	}, nil
}

func (runtime *runtimeDRRuntime) backupCount(defaultCount int64) int64 {
	if runtime == nil {
		return defaultCount
	}
	runtime.backupMu.RLock()
	defer runtime.backupMu.RUnlock()
	if len(runtime.backups) == 0 {
		return defaultCount
	}
	return int64(len(runtime.backups))
}

func (runtime *runtimeDRRuntime) restoreCount(defaultCount int64) int64 {
	if runtime == nil {
		return defaultCount
	}
	runtime.restoreMu.RLock()
	defer runtime.restoreMu.RUnlock()
	if len(runtime.restores) == 0 {
		return defaultCount
	}
	return int64(len(runtime.restores))
}

func runtimeGetDRStatusHandler(runtimeDR *runtimeDRRuntime) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		status, err := runtimeDR.Status()
		if err != nil {
			respondRuntimeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": err.Error()})
			return
		}
		respondRuntimeJSON(w, http.StatusOK, status)
	}
}

func runtimeListDRBackupsHandler(runtimeDR *runtimeDRRuntime) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		records, err := runtimeDR.ListBackups()
		if err != nil {
			respondRuntimeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": err.Error()})
			return
		}
		respondRuntimeJSON(w, http.StatusOK, records)
	}
}

func runtimeListDRRestoresHandler(runtimeDR *runtimeDRRuntime) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		records, err := runtimeDR.ListRestores()
		if err != nil {
			respondRuntimeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": err.Error()})
			return
		}
		respondRuntimeJSON(w, http.StatusOK, records)
	}
}

func runtimeStartDRRestoreHandler(runtimeDR *runtimeDRRuntime) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var request runtimeDRRestoreRequest
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			respondRuntimeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid request body"})
			return
		}
		record, err := runtimeDR.StartRestore(request)
		if err != nil {
			status := http.StatusBadRequest
			if strings.Contains(err.Error(), "not initialized") {
				status = http.StatusServiceUnavailable
			} else if strings.Contains(err.Error(), "not found") || strings.Contains(err.Error(), "not registered") {
				status = http.StatusNotFound
			}
			respondRuntimeJSON(w, status, map[string]string{"error": err.Error()})
			return
		}
		respondRuntimeJSON(w, http.StatusAccepted, record)
	}
}

func runtimeRegisterDRBackupHandler(runtimeDR *runtimeDRRuntime) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var request runtimeDRBackupRequest
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			respondRuntimeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid request body"})
			return
		}
		record, err := runtimeDR.RegisterBackup(request)
		if err != nil {
			respondRuntimeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
			return
		}
		respondRuntimeJSON(w, http.StatusAccepted, record)
	}
}
