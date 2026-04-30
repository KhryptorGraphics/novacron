package main

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/vm"
)

func TestRuntimeMobilityPolicyEndpointGatesLiveMigration(t *testing.T) {
	t.Parallel()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	config := defaultRuntimeConfig("node-a", t.TempDir())
	config.Services.MigrationMode = vm.MigrationModeLive
	apiServer, err := initializeAPI(ctx, config, "127.0.0.1:0", nil, nil, nil, nil, nil, nil)
	if err != nil {
		t.Fatalf("initializeAPI returned error: %v", err)
	}
	defer apiServer.Shutdown(context.Background())

	response := getJSONResponse[runtimeMobilityPolicyResponse](t, apiServer, "/internal/runtime/v1/mobility/policy")
	if got, want := response.Mode, vm.MigrationModeLive; got != want {
		t.Fatalf("mobility mode = %q, want %q", got, want)
	}
	if response.Policy.LiveMigrationGate.Enabled {
		t.Fatal("live migration gate should remain disabled by default")
	}
	if got, want := response.Policy.DefaultMigrationMode, vm.MigrationModeCheckpoint; got != want {
		t.Fatalf("policy default mode = %q, want gated fallback %q", got, want)
	}
	if containsRuntimeMobilityString(response.Policy.AllowedMigrationModes, vm.MigrationModeLive) {
		t.Fatalf("live migration should not be in allowed modes without gate: %#v", response.Policy.AllowedMigrationModes)
	}
	if !containsRuntimeMobilityString(response.Policy.AllowedMigrationModes, vm.MigrationModeCheckpoint) {
		t.Fatalf("checkpoint should be allowed before live migration: %#v", response.Policy.AllowedMigrationModes)
	}
}

func TestRuntimeColdMigrationEndpointExecutesPolicyVerifiedMigration(t *testing.T) {
	t.Parallel()

	sourceDir := t.TempDir()
	targetDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(sourceDir, "vm-1.state"), []byte("vm-state"), 0o644); err != nil {
		t.Fatalf("write source state: %v", err)
	}
	config := defaultRuntimeConfig("node-a", t.TempDir())
	config.Services.MigrationMode = vm.MigrationModeCold
	migrationManager := vm.NewVMMigrationManager("node-a", sourceDir)
	router := newRuntimeRouter(config, nil, migrationManager, nil, nil, nil, nil, nil, nil, nil, nil)
	verifyRuntimeBackup(t, router, "backup-1", "vm-1")

	payload := runtimeColdMigrationRequest{
		VMID:             "vm-1",
		TargetNodeID:     "node-b",
		TargetStorageDir: targetDir,
		BackupVerified:   true,
		BackupID:         "backup-1",
	}
	response := postRuntimeMobilityJSON[runtimeColdMigrationResponse](t, router, "/internal/runtime/v1/mobility/cold-migrations", payload, http.StatusAccepted)
	if got, want := response.Status, vm.MigrationStatusCompleted; got != want {
		t.Fatalf("migration status = %q, want %q", got, want)
	}
	if got, want := response.Progress, float64(100); got != want {
		t.Fatalf("migration progress = %f, want %f", got, want)
	}
	if _, err := os.Stat(filepath.Join(targetDir, "vm-1.state")); err != nil {
		t.Fatalf("expected target state file to be copied: %v", err)
	}
}

func TestRuntimeColdMigrationEndpointRequiresVerifiedBackup(t *testing.T) {
	t.Parallel()

	sourceDir := t.TempDir()
	targetDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(sourceDir, "vm-1.state"), []byte("vm-state"), 0o644); err != nil {
		t.Fatalf("write source state: %v", err)
	}
	config := defaultRuntimeConfig("node-a", t.TempDir())
	config.Services.MigrationMode = vm.MigrationModeCold
	migrationManager := vm.NewVMMigrationManager("node-a", sourceDir)
	router := newRuntimeRouter(config, nil, migrationManager, nil, nil, nil, nil, nil, nil, nil, nil)

	payload := runtimeColdMigrationRequest{
		VMID:             "vm-1",
		TargetNodeID:     "node-b",
		TargetStorageDir: targetDir,
	}
	response := postRuntimeMobilityJSON[runtimeColdMigrationResponse](t, router, "/internal/runtime/v1/mobility/cold-migrations", payload, http.StatusPreconditionFailed)
	if got, want := response.Status, vm.MigrationStatusPending; got != want {
		t.Fatalf("migration status = %q, want %q", got, want)
	}
	if response.Error != "" {
		t.Fatalf("precondition failure should not mutate migration error, got %q", response.Error)
	}
}

func TestRuntimeMobilityOperationsEndpointReportsHistory(t *testing.T) {
	t.Parallel()

	sourceDir := t.TempDir()
	targetDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(sourceDir, "vm-1.state"), []byte("vm-state"), 0o644); err != nil {
		t.Fatalf("write source state: %v", err)
	}
	config := defaultRuntimeConfig("node-a", t.TempDir())
	config.Services.MigrationMode = vm.MigrationModeCold
	migrationManager := vm.NewVMMigrationManager("node-a", sourceDir)
	router := newRuntimeRouter(config, nil, migrationManager, nil, nil, nil, nil, nil, nil, nil, nil)
	verifyRuntimeBackup(t, router, "backup-1", "vm-1")

	payload := runtimeColdMigrationRequest{
		VMID:             "vm-1",
		TargetNodeID:     "node-b",
		TargetStorageDir: targetDir,
		BackupVerified:   true,
		BackupID:         "backup-1",
	}
	postRuntimeMobilityJSON[runtimeColdMigrationResponse](t, router, "/internal/runtime/v1/mobility/cold-migrations", payload, http.StatusAccepted)

	operations := getRuntimeMobilityJSON[[]runtimeColdMigrationResponse](t, router, "/internal/runtime/v1/mobility/operations", http.StatusOK)
	if got, want := len(operations), 1; got != want {
		t.Fatalf("operation count = %d, want %d", got, want)
	}
	operation := operations[0]
	if got, want := operation.VMID, "vm-1"; got != want {
		t.Fatalf("operation vm id = %q, want %q", got, want)
	}
	if got, want := operation.Status, vm.MigrationStatusCompleted; got != want {
		t.Fatalf("operation status = %q, want %q", got, want)
	}
	if !operation.BackupVerified {
		t.Fatal("operation should preserve backup verification state")
	}
	if got, want := operation.BackupID, "backup-1"; got != want {
		t.Fatalf("operation backup id = %q, want %q", got, want)
	}
}

func TestRuntimeColdMigrationEndpointIgnoresUnregisteredBackupFlag(t *testing.T) {
	t.Parallel()

	sourceDir := t.TempDir()
	targetDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(sourceDir, "vm-1.state"), []byte("vm-state"), 0o644); err != nil {
		t.Fatalf("write source state: %v", err)
	}
	config := defaultRuntimeConfig("node-a", t.TempDir())
	config.Services.MigrationMode = vm.MigrationModeCold
	migrationManager := vm.NewVMMigrationManager("node-a", sourceDir)
	router := newRuntimeRouter(config, nil, migrationManager, nil, nil, nil, nil, nil, nil, nil, nil)

	payload := runtimeColdMigrationRequest{
		VMID:             "vm-1",
		TargetNodeID:     "node-b",
		TargetStorageDir: targetDir,
		BackupVerified:   true,
		BackupID:         "unregistered-backup",
	}
	response := postRuntimeMobilityJSON[runtimeColdMigrationResponse](t, router, "/internal/runtime/v1/mobility/cold-migrations", payload, http.StatusPreconditionFailed)
	if response.BackupVerified {
		t.Fatal("runtime should not trust request-supplied backup_verified without registry entry")
	}
	if got, want := response.Status, vm.MigrationStatusPending; got != want {
		t.Fatalf("migration status = %q, want %q", got, want)
	}
}

func TestRuntimeMobilityRecoveryEndpointReportsHealthyState(t *testing.T) {
	t.Parallel()

	sourceDir := t.TempDir()
	targetDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(sourceDir, "vm-1.state"), []byte("vm-state"), 0o644); err != nil {
		t.Fatalf("write source state: %v", err)
	}
	config := defaultRuntimeConfig("node-a", t.TempDir())
	config.Services.MigrationMode = vm.MigrationModeCold
	migrationManager := vm.NewVMMigrationManager("node-a", sourceDir)
	router := newRuntimeRouter(config, nil, migrationManager, nil, nil, nil, nil, nil, nil, nil, nil)
	verifyRuntimeBackup(t, router, "backup-1", "vm-1")

	payload := runtimeColdMigrationRequest{
		VMID:             "vm-1",
		TargetNodeID:     "node-b",
		TargetStorageDir: targetDir,
		BackupID:         "backup-1",
	}
	postRuntimeMobilityJSON[runtimeColdMigrationResponse](t, router, "/internal/runtime/v1/mobility/cold-migrations", payload, http.StatusAccepted)

	response := getRuntimeMobilityJSON[runtimeMobilityRecoveryResponse](t, router, "/internal/runtime/v1/mobility/recovery", http.StatusOK)
	if got, want := response.State, "healthy"; got != want {
		t.Fatalf("recovery state = %q, want %q", got, want)
	}
	if got, want := response.CompletedCount, 1; got != want {
		t.Fatalf("completed count = %d, want %d", got, want)
	}
	if !response.BackupWithinRPO {
		t.Fatal("registered recent backup should be within RPO")
	}
	if got, want := response.LatestBackupID, "backup-1"; got != want {
		t.Fatalf("latest backup id = %q, want %q", got, want)
	}
}

func TestRuntimeMobilityRecoveryFromStateReportsStaleBackup(t *testing.T) {
	t.Parallel()

	policy := vm.DefaultMigrationBackupPolicy(vm.MigrationModeCold)
	now := time.Date(2026, 4, 30, 12, 0, 0, 0, time.UTC)
	response := runtimeMobilityRecoveryFromState(policy, nil, []vm.VerifiedBackup{{
		ID:         "backup-stale",
		VMID:       "vm-1",
		VerifiedAt: now.Add(-time.Duration(policy.Backup.RPOSeconds+1) * time.Second),
	}}, now)
	if got, want := response.State, "degraded"; got != want {
		t.Fatalf("recovery state = %q, want %q", got, want)
	}
	if response.BackupWithinRPO {
		t.Fatal("stale backup should not be within RPO")
	}
	if !containsRuntimeMobilityString(response.RecoveryActionsNeeded, "verify_recent_backup") {
		t.Fatalf("recovery actions = %#v, want verify_recent_backup", response.RecoveryActionsNeeded)
	}
}

func TestRuntimeMobilityRecoveryFromStateReportsNodeLoss(t *testing.T) {
	t.Parallel()

	policy := vm.DefaultMigrationBackupPolicy(vm.MigrationModeCold)
	now := time.Date(2026, 4, 30, 12, 0, 0, 0, time.UTC)
	response := runtimeMobilityRecoveryFromState(policy, []*vm.VMMigration{{
		ID:        "node-loss-1",
		VMID:      "vm-1",
		Status:    vm.MigrationStatusFailed,
		Error:     "source node lost during migration",
		UpdatedAt: now.Add(-time.Minute),
		Options: map[string]string{
			runtimeMobilityFailureDomainOption: runtimeMobilityFailureDomainNode,
		},
	}}, []vm.VerifiedBackup{{
		ID:         "backup-node-loss",
		VMID:       "vm-1",
		VerifiedAt: now.Add(-time.Minute),
	}}, now)
	if got, want := response.State, "failed"; got != want {
		t.Fatalf("recovery state = %q, want %q", got, want)
	}
	if got, want := response.NodeLossCount, 1; got != want {
		t.Fatalf("node loss count = %d, want %d", got, want)
	}
	if !containsRuntimeMobilityString(response.RecoveryActionsNeeded, "restore_workload_on_surviving_node") {
		t.Fatalf("recovery actions = %#v, want restore_workload_on_surviving_node", response.RecoveryActionsNeeded)
	}
}

func TestRuntimeMobilityRecoveryFromStateReportsReplicaLoss(t *testing.T) {
	t.Parallel()

	policy := vm.DefaultMigrationBackupPolicy(vm.MigrationModeCold)
	now := time.Date(2026, 4, 30, 12, 0, 0, 0, time.UTC)
	response := runtimeMobilityRecoveryFromState(policy, []*vm.VMMigration{{
		ID:        "replica-loss-1",
		VMID:      "vm-1",
		Status:    vm.MigrationStatusCompleted,
		UpdatedAt: now.Add(-time.Minute),
		Options: map[string]string{
			runtimeMobilityFailureDomainOption: runtimeMobilityFailureDomainReplica,
		},
	}}, []vm.VerifiedBackup{{
		ID:         "backup-replica-loss",
		VMID:       "vm-1",
		VerifiedAt: now.Add(-time.Minute),
	}}, now)
	if got, want := response.State, "degraded"; got != want {
		t.Fatalf("recovery state = %q, want %q", got, want)
	}
	if got, want := response.ReplicaLossCount, 1; got != want {
		t.Fatalf("replica loss count = %d, want %d", got, want)
	}
	if !containsRuntimeMobilityString(response.RecoveryActionsNeeded, "reseed_storage_replica") {
		t.Fatalf("recovery actions = %#v, want reseed_storage_replica", response.RecoveryActionsNeeded)
	}
}

func TestRuntimeMobilityRecoveryEndpointReportsRolledBackOperation(t *testing.T) {
	t.Parallel()

	sourceDir := t.TempDir()
	targetDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(sourceDir, "vm-restore.checkpoint"), []byte("checkpoint-state"), 0o644); err != nil {
		t.Fatalf("write source checkpoint: %v", err)
	}
	config := defaultRuntimeConfig("node-a", t.TempDir())
	config.Services.MigrationMode = vm.MigrationModeCheckpoint
	migrationManager := vm.NewVMMigrationManager("node-a", sourceDir)
	router := newRuntimeRouter(config, nil, migrationManager, nil, nil, nil, nil, nil, nil, nil, nil)
	verifyRuntimeBackup(t, router, "backup-restore-1", "vm-restore")

	payload := runtimeCheckpointRestoreRequest{
		VMID:             "vm-restore",
		TargetNodeID:     "node-b",
		TargetStorageDir: targetDir,
		BackupID:         "backup-restore-1",
	}
	body, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("marshal payload: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	req := httptest.NewRequest(http.MethodPost, "/internal/runtime/v1/mobility/checkpoint-restores", bytes.NewReader(body)).WithContext(ctx)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusConflict {
		t.Fatalf("POST checkpoint restore status = %d, want %d: %s", rec.Code, http.StatusConflict, rec.Body.String())
	}

	response := getRuntimeMobilityJSON[runtimeMobilityRecoveryResponse](t, router, "/internal/runtime/v1/mobility/recovery", http.StatusOK)
	if got, want := response.State, "recovering"; got != want {
		t.Fatalf("recovery state = %q, want %q", got, want)
	}
	if got, want := response.RolledBackCount, 1; got != want {
		t.Fatalf("rolled back count = %d, want %d", got, want)
	}
	if got, want := response.LastOperationError, context.Canceled.Error(); got != want {
		t.Fatalf("last operation error = %q, want %q", got, want)
	}
	if !containsRuntimeMobilityString(response.RecoveryActionsNeeded, "review_rolled_back_mobility_operation") {
		t.Fatalf("recovery actions = %#v, want review_rolled_back_mobility_operation", response.RecoveryActionsNeeded)
	}
}

func TestRuntimeCheckpointRestoreEndpointExecutesPolicyVerifiedRestore(t *testing.T) {
	t.Parallel()

	sourceDir := t.TempDir()
	targetDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(sourceDir, "checkpoint-1.checkpoint"), []byte("checkpoint-state"), 0o644); err != nil {
		t.Fatalf("write source checkpoint: %v", err)
	}
	config := defaultRuntimeConfig("node-a", t.TempDir())
	config.Services.MigrationMode = vm.MigrationModeCheckpoint
	migrationManager := vm.NewVMMigrationManager("node-a", sourceDir)
	router := newRuntimeRouter(config, nil, migrationManager, nil, nil, nil, nil, nil, nil, nil, nil)
	verifyRuntimeBackup(t, router, "backup-restore-1", "vm-restore")

	payload := runtimeCheckpointRestoreRequest{
		VMID:             "vm-restore",
		TargetNodeID:     "node-b",
		TargetStorageDir: targetDir,
		CheckpointID:     "checkpoint-1",
		BackupVerified:   true,
		BackupID:         "backup-restore-1",
	}
	response := postRuntimeMobilityJSON[runtimeColdMigrationResponse](t, router, "/internal/runtime/v1/mobility/checkpoint-restores", payload, http.StatusAccepted)
	if got, want := response.Type, vm.MigrationType(vm.MigrationModeCheckpoint); got != want {
		t.Fatalf("restore type = %q, want %q", got, want)
	}
	if got, want := response.Status, vm.MigrationStatusCompleted; got != want {
		t.Fatalf("restore status = %q, want %q", got, want)
	}
	if got, want := response.CheckpointID, "checkpoint-1"; got != want {
		t.Fatalf("checkpoint id = %q, want %q", got, want)
	}
	restored, err := os.ReadFile(filepath.Join(targetDir, "vm-restore.state"))
	if err != nil {
		t.Fatalf("expected target state file to be restored: %v", err)
	}
	if got, want := string(restored), "checkpoint-state"; got != want {
		t.Fatalf("restored state = %q, want %q", got, want)
	}
}

func TestRuntimeCheckpointRestoreEndpointRequiresCheckpointPolicy(t *testing.T) {
	t.Parallel()

	sourceDir := t.TempDir()
	targetDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(sourceDir, "vm-restore.checkpoint"), []byte("checkpoint-state"), 0o644); err != nil {
		t.Fatalf("write source checkpoint: %v", err)
	}
	config := defaultRuntimeConfig("node-a", t.TempDir())
	config.Services.MigrationMode = vm.MigrationModeCold
	migrationManager := vm.NewVMMigrationManager("node-a", sourceDir)
	router := newRuntimeRouter(config, nil, migrationManager, nil, nil, nil, nil, nil, nil, nil, nil)
	verifyRuntimeBackup(t, router, "backup-restore-1", "vm-restore")

	payload := runtimeCheckpointRestoreRequest{
		VMID:             "vm-restore",
		TargetNodeID:     "node-b",
		TargetStorageDir: targetDir,
		BackupVerified:   true,
		BackupID:         "backup-restore-1",
	}
	response := postRuntimeMobilityJSON[runtimeColdMigrationResponse](t, router, "/internal/runtime/v1/mobility/checkpoint-restores", payload, http.StatusConflict)
	if got, want := response.Status, vm.MigrationStatusPending; got != want {
		t.Fatalf("restore status = %q, want %q", got, want)
	}
	if _, err := os.Stat(filepath.Join(targetDir, "vm-restore.state")); !os.IsNotExist(err) {
		t.Fatalf("checkpoint restore should not create target state when policy rejects it, stat err: %v", err)
	}
}

func TestRuntimeCheckpointRestoreEndpointReportsInterruptedRollback(t *testing.T) {
	t.Parallel()

	sourceDir := t.TempDir()
	targetDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(sourceDir, "vm-restore.checkpoint"), []byte("checkpoint-state"), 0o644); err != nil {
		t.Fatalf("write source checkpoint: %v", err)
	}
	config := defaultRuntimeConfig("node-a", t.TempDir())
	config.Services.MigrationMode = vm.MigrationModeCheckpoint
	migrationManager := vm.NewVMMigrationManager("node-a", sourceDir)
	router := newRuntimeRouter(config, nil, migrationManager, nil, nil, nil, nil, nil, nil, nil, nil)
	verifyRuntimeBackup(t, router, "backup-restore-1", "vm-restore")

	payload := runtimeCheckpointRestoreRequest{
		VMID:             "vm-restore",
		TargetNodeID:     "node-b",
		TargetStorageDir: targetDir,
		BackupVerified:   true,
		BackupID:         "backup-restore-1",
	}
	body, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("marshal payload: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	req := httptest.NewRequest(http.MethodPost, "/internal/runtime/v1/mobility/checkpoint-restores", bytes.NewReader(body)).WithContext(ctx)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusConflict {
		t.Fatalf("POST checkpoint restore status = %d, want %d: %s", rec.Code, http.StatusConflict, rec.Body.String())
	}

	var response runtimeColdMigrationResponse
	if err := json.NewDecoder(rec.Body).Decode(&response); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if got, want := response.Status, vm.MigrationStatusRolledBack; got != want {
		t.Fatalf("restore status = %q, want %q", got, want)
	}
	if !response.RollbackAttempted {
		t.Fatal("rollback should be reported as attempted")
	}
	if !response.RollbackSucceeded {
		t.Fatal("rollback should be reported as successful")
	}
	if response.Error != context.Canceled.Error() {
		t.Fatalf("restore error = %q, want %q", response.Error, context.Canceled.Error())
	}
}

func postRuntimeMobilityJSON[T any](t *testing.T, router http.Handler, path string, payload interface{}, expectedStatus int) T {
	t.Helper()

	var zero T
	body, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("marshal payload: %v", err)
	}
	req := httptest.NewRequest(http.MethodPost, path, bytes.NewReader(body))
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != expectedStatus {
		t.Fatalf("POST %s status = %d, want %d: %s", path, rec.Code, expectedStatus, rec.Body.String())
	}
	if err := json.NewDecoder(rec.Body).Decode(&zero); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	return zero
}

func verifyRuntimeBackup(t *testing.T, router http.Handler, backupID string, vmID string) runtimeVerifiedBackupResponse {
	t.Helper()
	return postRuntimeMobilityJSON[runtimeVerifiedBackupResponse](t, router, "/internal/runtime/v1/mobility/backups/verified", runtimeVerifyBackupRequest{
		BackupID: backupID,
		VMID:     vmID,
	}, http.StatusAccepted)
}

func getRuntimeMobilityJSON[T any](t *testing.T, router http.Handler, path string, expectedStatus int) T {
	t.Helper()

	var zero T
	req := httptest.NewRequest(http.MethodGet, path, nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != expectedStatus {
		t.Fatalf("GET %s status = %d, want %d: %s", path, rec.Code, expectedStatus, rec.Body.String())
	}
	if err := json.NewDecoder(rec.Body).Decode(&zero); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	return zero
}

func containsRuntimeMobilityString(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}
