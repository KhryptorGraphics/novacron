package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/vm"
)

type runtimeMobilityPolicyResponse struct {
	Mode   string                   `json:"mode"`
	Policy vm.MigrationBackupPolicy `json:"policy"`
}

type runtimeColdMigrationRequest struct {
	VMID             string `json:"vm_id"`
	TargetNodeID     string `json:"target_node_id"`
	TargetStorageDir string `json:"target_storage_dir"`
	BackupVerified   bool   `json:"backup_verified"`
	BackupID         string `json:"backup_id,omitempty"`
}

type runtimeCheckpointRestoreRequest struct {
	VMID             string `json:"vm_id"`
	TargetNodeID     string `json:"target_node_id"`
	TargetStorageDir string `json:"target_storage_dir"`
	CheckpointID     string `json:"checkpoint_id,omitempty"`
	BackupVerified   bool   `json:"backup_verified"`
	BackupID         string `json:"backup_id,omitempty"`
}

type runtimeVerifyBackupRequest struct {
	BackupID string `json:"backup_id"`
	VMID     string `json:"vm_id,omitempty"`
}

type runtimeVerifiedBackupResponse struct {
	BackupID   string    `json:"backup_id"`
	VMID       string    `json:"vm_id,omitempty"`
	VerifiedAt time.Time `json:"verified_at"`
}

type runtimeColdMigrationResponse struct {
	ID                string             `json:"id"`
	VMID              string             `json:"vm_id"`
	SourceNodeID      string             `json:"source_node_id"`
	DestinationNodeID string             `json:"destination_node_id"`
	Type              vm.MigrationType   `json:"type"`
	Status            vm.MigrationStatus `json:"status"`
	Progress          float64            `json:"progress"`
	Error             string             `json:"error,omitempty"`
	Policy            string             `json:"policy"`
	BackupVerified    bool               `json:"backup_verified"`
	BackupID          string             `json:"backup_id,omitempty"`
	CheckpointID      string             `json:"checkpoint_id,omitempty"`
	RollbackAttempted bool               `json:"rollback_attempted"`
	RollbackSucceeded bool               `json:"rollback_succeeded"`
	RollbackError     string             `json:"rollback_error,omitempty"`
}

func runtimeMobilityPolicyFromConfig(config runtimeConfig) vm.MigrationBackupPolicy {
	policy := vm.DefaultMigrationBackupPolicy(config.Services.MigrationMode)
	policy.Metadata = map[string]string{
		"deployment_profile": config.Services.DeploymentProfile,
		"migration_mode":     config.Services.MigrationMode,
		"storage_base_path":  config.Storage.BasePath,
	}
	return policy.Normalize()
}

func runtimeGetMobilityPolicyHandler(config runtimeConfig) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		policy := runtimeMobilityPolicyFromConfig(config)
		respondRuntimeJSON(w, http.StatusOK, runtimeMobilityPolicyResponse{
			Mode:   vm.NormalizeMigrationMode(config.Services.MigrationMode),
			Policy: policy,
		})
	}
}

func runtimeListMobilityOperationsHandler(config runtimeConfig, migrationManager *vm.VMMigrationManager) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		if migrationManager == nil {
			respondRuntimeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "migration runtime is not initialized"})
			return
		}
		policy := runtimeMobilityPolicyFromConfig(config)
		operations := migrationManager.ListMigrations()
		response := make([]runtimeColdMigrationResponse, 0, len(operations))
		for _, operation := range operations {
			response = append(response, runtimeColdMigrationResponseFromMigration(
				operation,
				policy,
				runtimeMigrationOptionEnabled(operation.Options, vm.MigrationOptionBackupVerified),
				operation.Options[vm.MigrationOptionBackupID],
				operation.Options[vm.MigrationOptionCheckpointID],
			))
		}
		respondRuntimeJSON(w, http.StatusOK, response)
	}
}

func runtimeListVerifiedBackupsHandler(migrationManager *vm.VMMigrationManager) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		if migrationManager == nil {
			respondRuntimeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "migration runtime is not initialized"})
			return
		}
		backups := migrationManager.ListVerifiedBackups()
		response := make([]runtimeVerifiedBackupResponse, 0, len(backups))
		for _, backup := range backups {
			response = append(response, runtimeVerifiedBackupResponse{
				BackupID:   backup.ID,
				VMID:       backup.VMID,
				VerifiedAt: backup.VerifiedAt,
			})
		}
		respondRuntimeJSON(w, http.StatusOK, response)
	}
}

func runtimeVerifyBackupHandler(migrationManager *vm.VMMigrationManager) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if migrationManager == nil {
			respondRuntimeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "migration runtime is not initialized"})
			return
		}
		var request runtimeVerifyBackupRequest
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			respondRuntimeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid backup verification request"})
			return
		}
		backup, err := migrationManager.RecordVerifiedBackup(request.BackupID, request.VMID, time.Now().UTC())
		if err != nil {
			respondRuntimeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
			return
		}
		respondRuntimeJSON(w, http.StatusAccepted, runtimeVerifiedBackupResponse{
			BackupID:   backup.ID,
			VMID:       backup.VMID,
			VerifiedAt: backup.VerifiedAt,
		})
	}
}

func runtimeStartColdMigrationHandler(config runtimeConfig, migrationManager *vm.VMMigrationManager) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if migrationManager == nil {
			respondRuntimeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "migration runtime is not initialized"})
			return
		}

		var request runtimeColdMigrationRequest
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			respondRuntimeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid cold migration request"})
			return
		}
		request.VMID = strings.TrimSpace(request.VMID)
		request.TargetNodeID = strings.TrimSpace(request.TargetNodeID)
		request.TargetStorageDir = strings.TrimSpace(request.TargetStorageDir)
		request.BackupID = strings.TrimSpace(request.BackupID)
		if request.VMID == "" || request.TargetNodeID == "" || request.TargetStorageDir == "" {
			respondRuntimeJSON(w, http.StatusBadRequest, map[string]string{"error": "vm_id, target_node_id, and target_storage_dir are required"})
			return
		}

		backupVerified := migrationManager.BackupVerified(request.BackupID)
		options := map[string]string{
			vm.MigrationOptionBackupVerified: fmt.Sprintf("%t", backupVerified),
		}
		if request.BackupID != "" {
			options[vm.MigrationOptionBackupID] = request.BackupID
		}
		migration := &vm.VMMigration{
			ID:                fmt.Sprintf("cold-%s-%d", request.VMID, time.Now().UTC().UnixNano()),
			VMID:              request.VMID,
			SourceNodeID:      runtimeDiscoveryNodeID(config),
			DestinationNodeID: request.TargetNodeID,
			Type:              vm.MigrationTypeCold,
			Status:            vm.MigrationStatusPending,
			CreatedAt:         time.Now().UTC(),
			UpdatedAt:         time.Now().UTC(),
			Options:           options,
		}
		policy := runtimeMobilityPolicyFromConfig(config)
		destinationManager := vm.NewVMMigrationManager(request.TargetNodeID, request.TargetStorageDir)
		if err := migrationManager.ExecuteMigrationWithPolicy(r.Context(), migration, destinationManager, policy); err != nil {
			status := http.StatusConflict
			if strings.Contains(err.Error(), "requires a verified recent backup") {
				status = http.StatusPreconditionFailed
			}
			respondRuntimeJSON(w, status, runtimeColdMigrationResponseFromMigration(migration, policy, backupVerified, request.BackupID, ""))
			return
		}
		respondRuntimeJSON(w, http.StatusAccepted, runtimeColdMigrationResponseFromMigration(migration, policy, backupVerified, request.BackupID, ""))
	}
}

func runtimeStartCheckpointRestoreHandler(config runtimeConfig, migrationManager *vm.VMMigrationManager) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if migrationManager == nil {
			respondRuntimeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "migration runtime is not initialized"})
			return
		}

		var request runtimeCheckpointRestoreRequest
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			respondRuntimeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid checkpoint restore request"})
			return
		}
		request.VMID = strings.TrimSpace(request.VMID)
		request.TargetNodeID = strings.TrimSpace(request.TargetNodeID)
		request.TargetStorageDir = strings.TrimSpace(request.TargetStorageDir)
		request.CheckpointID = strings.TrimSpace(request.CheckpointID)
		request.BackupID = strings.TrimSpace(request.BackupID)
		if request.VMID == "" || request.TargetNodeID == "" || request.TargetStorageDir == "" {
			respondRuntimeJSON(w, http.StatusBadRequest, map[string]string{"error": "vm_id, target_node_id, and target_storage_dir are required"})
			return
		}

		backupVerified := migrationManager.BackupVerified(request.BackupID)
		options := map[string]string{
			vm.MigrationOptionBackupVerified: fmt.Sprintf("%t", backupVerified),
		}
		if request.BackupID != "" {
			options[vm.MigrationOptionBackupID] = request.BackupID
		}
		if request.CheckpointID != "" {
			options[vm.MigrationOptionCheckpointID] = request.CheckpointID
		}
		now := time.Now().UTC()
		migration := &vm.VMMigration{
			ID:                fmt.Sprintf("checkpoint-%s-%d", request.VMID, now.UnixNano()),
			VMID:              request.VMID,
			SourceNodeID:      runtimeDiscoveryNodeID(config),
			DestinationNodeID: request.TargetNodeID,
			Type:              vm.MigrationType(vm.MigrationModeCheckpoint),
			Status:            vm.MigrationStatusPending,
			CreatedAt:         now,
			UpdatedAt:         now,
			Options:           options,
		}
		policy := runtimeMobilityPolicyFromConfig(config)
		destinationManager := vm.NewVMMigrationManager(request.TargetNodeID, request.TargetStorageDir)
		if err := migrationManager.ExecuteCheckpointRestoreWithPolicy(r.Context(), migration, destinationManager, policy); err != nil {
			status := http.StatusConflict
			if strings.Contains(err.Error(), "requires a verified recent backup") {
				status = http.StatusPreconditionFailed
			}
			respondRuntimeJSON(w, status, runtimeColdMigrationResponseFromMigration(migration, policy, backupVerified, request.BackupID, request.CheckpointID))
			return
		}
		respondRuntimeJSON(w, http.StatusAccepted, runtimeColdMigrationResponseFromMigration(migration, policy, backupVerified, request.BackupID, request.CheckpointID))
	}
}

func runtimeColdMigrationResponseFromMigration(migration *vm.VMMigration, policy vm.MigrationBackupPolicy, backupVerified bool, backupID string, checkpointID string) runtimeColdMigrationResponse {
	response := runtimeColdMigrationResponse{
		Policy:         policy.DefaultMigrationMode,
		BackupVerified: backupVerified,
		BackupID:       backupID,
		CheckpointID:   checkpointID,
	}
	if migration == nil {
		return response
	}
	response.ID = migration.ID
	response.VMID = migration.VMID
	response.SourceNodeID = migration.SourceNodeID
	response.DestinationNodeID = migration.DestinationNodeID
	response.Type = migration.Type
	response.Status = migration.Status
	response.Progress = migration.Progress
	response.Error = migration.Error
	response.RollbackAttempted = migration.RollbackAttempted
	response.RollbackSucceeded = migration.RollbackSucceeded
	response.RollbackError = migration.RollbackError
	return response
}

func runtimeMigrationOptionEnabled(options map[string]string, key string) bool {
	switch strings.TrimSpace(strings.ToLower(options[key])) {
	case "1", "true", "yes", "verified":
		return true
	default:
		return false
	}
}
