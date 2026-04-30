package main

import (
	"fmt"
	"net/http"
	"path/filepath"
	"time"

	dr "github.com/khryptorgraphics/novacron/backend/core/dr"
)

type runtimeDRRuntime struct {
	orchestrator *dr.Orchestrator
	api          *dr.DRAPI
	started      bool
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

func initializeRuntimeDR(config runtimeConfig) (*runtimeDRRuntime, error) {
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

	return &runtimeDRRuntime{
		orchestrator: orchestrator,
		api:          dr.NewDRAPI(orchestrator),
		started:      true,
	}, nil
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
		BackupCount:      status.BackupCount,
		RestoreCount:     status.RestoreCount,
		BackupMetrics:    runtime.api.GetBackupMetrics(),
		RestoreMetrics:   runtime.api.GetRestoreMetrics(),
	}, nil
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
