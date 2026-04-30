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

const (
	runtimeMobilityFailureDomainOption  = "failure_domain"
	runtimeMobilityFailureDomainNode    = "node_loss"
	runtimeMobilityFailureDomainReplica = "replica_loss"
)

type runtimeMobilityRecoveryResponse struct {
	State                 string        `json:"state"`
	OperationCount        int           `json:"operation_count"`
	CompletedCount        int           `json:"completed_count"`
	FailedCount           int           `json:"failed_count"`
	RolledBackCount       int           `json:"rolled_back_count"`
	NodeLossCount         int           `json:"node_loss_count"`
	ReplicaLossCount      int           `json:"replica_loss_count"`
	VerifiedBackupCount   int           `json:"verified_backup_count"`
	LatestBackupID        string        `json:"latest_backup_id,omitempty"`
	LatestBackupAge       time.Duration `json:"latest_backup_age"`
	BackupRPOSeconds      int64         `json:"backup_rpo_seconds"`
	BackupWithinRPO       bool          `json:"backup_within_rpo"`
	RTOSeconds            int64         `json:"rto_seconds"`
	LastOperationStatus   string        `json:"last_operation_status,omitempty"`
	LastOperationError    string        `json:"last_operation_error,omitempty"`
	LastRollbackAttempted bool          `json:"last_rollback_attempted"`
	LastRollbackSucceeded bool          `json:"last_rollback_succeeded"`
	LastRollbackError     string        `json:"last_rollback_error,omitempty"`
	RecoveryActionsNeeded []string      `json:"recovery_actions_needed,omitempty"`
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

func runtimeGetMobilityRecoveryHandler(config runtimeConfig, migrationManager *vm.VMMigrationManager) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		if migrationManager == nil {
			respondRuntimeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "migration runtime is not initialized"})
			return
		}
		respondRuntimeJSON(w, http.StatusOK, runtimeMobilityRecoveryFromState(runtimeMobilityPolicyFromConfig(config), migrationManager.ListMigrations(), migrationManager.ListVerifiedBackups(), time.Now().UTC()))
	}
}

func runtimeMobilityRecoveryFromState(policy vm.MigrationBackupPolicy, operations []*vm.VMMigration, backups []vm.VerifiedBackup, now time.Time) runtimeMobilityRecoveryResponse {
	policy = policy.Normalize()
	response := runtimeMobilityRecoveryResponse{
		State:               "healthy",
		OperationCount:      len(operations),
		VerifiedBackupCount: len(backups),
		BackupRPOSeconds:    policy.Backup.RPOSeconds,
		BackupWithinRPO:     !policy.Backup.RequireRecentBackup,
		RTOSeconds:          policy.Recovery.RTOSeconds,
	}
	var latestOperation *vm.VMMigration
	for _, operation := range operations {
		if operation == nil {
			continue
		}
		switch operation.Status {
		case vm.MigrationStatusCompleted:
			response.CompletedCount++
		case vm.MigrationStatusFailed:
			response.FailedCount++
		case vm.MigrationStatusRolledBack:
			response.RolledBackCount++
		}
		if latestOperation == nil || operation.UpdatedAt.After(latestOperation.UpdatedAt) {
			latestOperation = operation
		}
		switch strings.TrimSpace(operation.Options[runtimeMobilityFailureDomainOption]) {
		case runtimeMobilityFailureDomainNode:
			response.NodeLossCount++
		case runtimeMobilityFailureDomainReplica:
			response.ReplicaLossCount++
		}
	}
	if latestOperation != nil {
		response.LastOperationStatus = string(latestOperation.Status)
		response.LastOperationError = latestOperation.Error
		response.LastRollbackAttempted = latestOperation.RollbackAttempted
		response.LastRollbackSucceeded = latestOperation.RollbackSucceeded
		response.LastRollbackError = latestOperation.RollbackError
	}

	var latestBackup *vm.VerifiedBackup
	for i := range backups {
		if latestBackup == nil || backups[i].VerifiedAt.After(latestBackup.VerifiedAt) {
			latestBackup = &backups[i]
		}
	}
	if latestBackup != nil {
		response.LatestBackupID = latestBackup.ID
		response.LatestBackupAge = now.Sub(latestBackup.VerifiedAt)
		if response.LatestBackupAge < 0 {
			response.LatestBackupAge = 0
		}
		if policy.Backup.RequireRecentBackup {
			rpo := time.Duration(policy.Backup.RPOSeconds) * time.Second
			response.BackupWithinRPO = policy.Backup.RPOSeconds <= 0 || response.LatestBackupAge <= rpo
		}
	}

	if policy.Backup.RequireRecentBackup && !response.BackupWithinRPO {
		response.State = "degraded"
		response.RecoveryActionsNeeded = append(response.RecoveryActionsNeeded, "verify_recent_backup")
	}
	if response.FailedCount > 0 {
		response.State = "failed"
		response.RecoveryActionsNeeded = append(response.RecoveryActionsNeeded, "inspect_failed_mobility_operation")
	}
	if response.RolledBackCount > 0 && response.State != "failed" {
		response.State = "recovering"
		response.RecoveryActionsNeeded = append(response.RecoveryActionsNeeded, "review_rolled_back_mobility_operation")
	}
	if response.NodeLossCount > 0 {
		response.State = "failed"
		response.RecoveryActionsNeeded = append(response.RecoveryActionsNeeded, "restore_workload_on_surviving_node")
	}
	if response.ReplicaLossCount > 0 && response.State != "failed" {
		response.State = "degraded"
		response.RecoveryActionsNeeded = append(response.RecoveryActionsNeeded, "reseed_storage_replica")
	}
	if response.LastRollbackAttempted && !response.LastRollbackSucceeded {
		response.State = "failed"
		response.RecoveryActionsNeeded = append(response.RecoveryActionsNeeded, "repair_failed_rollback")
	}
	return response
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
