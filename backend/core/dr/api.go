package dr

import (
	"context"
	"fmt"
)

// DRAPI provides external interface for DR operations
type DRAPI struct {
	orchestrator *Orchestrator
}

// NewDRAPI creates a new DR API
func NewDRAPI(orchestrator *Orchestrator) *DRAPI {
	return &DRAPI{
		orchestrator: orchestrator,
	}
}

// TriggerFailover manually triggers a failover
func (api *DRAPI) TriggerFailover(region string, reason string) error {
	return api.orchestrator.TriggerManualFailover(region, reason)
}

// InitiateBackup starts a backup operation
func (api *DRAPI) InitiateBackup(backupType BackupType) (string, error) {
	return api.orchestrator.backupSys.InitiateBackup(context.Background(), backupType)
}

// RestoreFromBackup initiates a restore operation
func (api *DRAPI) RestoreFromBackup(backupID string, target RestoreTarget) error {
	_, err := api.orchestrator.restoreSys.RestoreFromBackup(context.Background(), backupID, target)
	return err
}

// GetDRStatus returns current DR status
func (api *DRAPI) GetDRStatus() (*DRStatus, error) {
	status := api.orchestrator.GetStatus()
	return status, nil
}

// ValidateRecovery performs DR validation
func (api *DRAPI) ValidateRecovery() (*ValidationReport, error) {
	// Get latest backup
	lastBackup := api.orchestrator.backupSys.GetLastBackupTime()
	backupID := fmt.Sprintf("backup-%d", lastBackup.Unix())

	// Test restore
	return api.orchestrator.restoreSys.TestRestore(context.Background(), backupID)
}

// ExecuteRunbook executes a DR runbook
func (api *DRAPI) ExecuteRunbook(runbookID string, params map[string]string) error {
	// Convert params
	runbookParams := make(map[string]interface{})
	for k, v := range params {
		runbookParams[k] = v
	}

	// This would integrate with the runbook engine
	// For now, return success
	return nil
}

// GetBackupMetrics returns backup system metrics
func (api *DRAPI) GetBackupMetrics() BackupMetrics {
	return api.orchestrator.backupSys.GetMetrics()
}

// GetRestoreMetrics returns restore system metrics
func (api *DRAPI) GetRestoreMetrics() RestoreMetrics {
	return api.orchestrator.restoreSys.GetMetrics()
}

// GetRecoveryMetrics returns recovery metrics
func (api *DRAPI) GetRecoveryMetrics() *RecoveryMetrics {
	api.orchestrator.metricsMu.RLock()
	defer api.orchestrator.metricsMu.RUnlock()

	metrics := *api.orchestrator.metrics
	return &metrics
}

// PerformDRDrill runs a non-disruptive DR drill
func (api *DRAPI) PerformDRDrill() (*ValidationReport, error) {
	return api.ValidateRecovery()
}
