package main

import (
	"net/http"

	"github.com/khryptorgraphics/novacron/backend/core/vm"
)

type runtimeMobilityPolicyResponse struct {
	Mode   string                   `json:"mode"`
	Policy vm.MigrationBackupPolicy `json:"policy"`
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
