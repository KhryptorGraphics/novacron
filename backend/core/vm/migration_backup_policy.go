package vm

import (
	"fmt"
	"sort"
	"strings"
)

const (
	MigrationModeDisabled   = "disabled"
	MigrationModeCold       = "cold"
	MigrationModeCheckpoint = "checkpoint"
	MigrationModeLive       = "live"
)

// MigrationBackupPolicy is the canonical VM mobility and recovery contract
// shared by migration, backup/DR, placement, and runtime API surfaces.
type MigrationBackupPolicy struct {
	AllowedMigrationModes []string                    `yaml:"allowed_migration_modes" json:"allowed_migration_modes"`
	DefaultMigrationMode  string                      `yaml:"default_migration_mode" json:"default_migration_mode"`
	CheckpointTarget      string                      `yaml:"checkpoint_target,omitempty" json:"checkpoint_target,omitempty"`
	Replication           VMMobilityReplicationPolicy `yaml:"replication" json:"replication"`
	Backup                VMBackupPolicy              `yaml:"backup" json:"backup"`
	Rollback              VMRollbackPolicy            `yaml:"rollback" json:"rollback"`
	Recovery              VMRecoveryPolicy            `yaml:"recovery" json:"recovery"`
	LiveMigrationGate     VMLiveMigrationGate         `yaml:"live_migration_gate" json:"live_migration_gate"`
	Metadata              map[string]string           `yaml:"metadata,omitempty" json:"metadata,omitempty"`
}

type VMBackupPolicy struct {
	Enabled             bool   `yaml:"enabled" json:"enabled"`
	RequireRecentBackup bool   `yaml:"require_recent_backup" json:"require_recent_backup"`
	BackupClass         string `yaml:"backup_class,omitempty" json:"backup_class,omitempty"`
	Retention           string `yaml:"retention,omitempty" json:"retention,omitempty"`
	RPOSeconds          int64  `yaml:"rpo_seconds,omitempty" json:"rpo_seconds,omitempty"`
}

type VMMobilityReplicationPolicy struct {
	Enabled bool   `yaml:"enabled" json:"enabled"`
	Factor  int    `yaml:"factor" json:"factor"`
	Mode    string `yaml:"mode,omitempty" json:"mode,omitempty"`
}

type VMRollbackPolicy struct {
	Enabled    bool     `yaml:"enabled" json:"enabled"`
	Conditions []string `yaml:"conditions,omitempty" json:"conditions,omitempty"`
}

type VMRecoveryPolicy struct {
	CheckpointRestore bool  `yaml:"checkpoint_restore" json:"checkpoint_restore"`
	BackupRestore     bool  `yaml:"backup_restore" json:"backup_restore"`
	RTOSeconds        int64 `yaml:"rto_seconds,omitempty" json:"rto_seconds,omitempty"`
}

type VMLiveMigrationGate struct {
	Enabled          bool     `yaml:"enabled" json:"enabled"`
	Reason           string   `yaml:"reason,omitempty" json:"reason,omitempty"`
	RequiredFeatures []string `yaml:"required_features,omitempty" json:"required_features,omitempty"`
}

// DefaultMigrationBackupPolicy returns the conservative Phase 3 baseline:
// cold migration first, checkpoint restore second, and live migration gated.
func DefaultMigrationBackupPolicy(mode string) MigrationBackupPolicy {
	defaultMode := NormalizeMigrationMode(mode)
	if defaultMode == "" {
		defaultMode = MigrationModeDisabled
	}
	effectiveDefaultMode := defaultMode
	if defaultMode == MigrationModeLive {
		effectiveDefaultMode = MigrationModeCheckpoint
	}

	policy := MigrationBackupPolicy{
		AllowedMigrationModes: []string{MigrationModeCold},
		DefaultMigrationMode:  effectiveDefaultMode,
		CheckpointTarget:      "local",
		Replication: VMMobilityReplicationPolicy{
			Enabled: defaultMode != MigrationModeDisabled,
			Factor:  1,
			Mode:    "async",
		},
		Backup: VMBackupPolicy{
			Enabled:             defaultMode != MigrationModeDisabled,
			RequireRecentBackup: defaultMode != MigrationModeDisabled,
			BackupClass:         "local",
			Retention:           "24h",
			RPOSeconds:          300,
		},
		Rollback: VMRollbackPolicy{
			Enabled: true,
			Conditions: []string{
				"destination_activation_failed",
				"verification_failed",
				"transfer_interrupted",
			},
		},
		Recovery: VMRecoveryPolicy{
			CheckpointRestore: defaultMode == MigrationModeCheckpoint || defaultMode == MigrationModeLive,
			BackupRestore:     defaultMode != MigrationModeDisabled,
			RTOSeconds:        900,
		},
		LiveMigrationGate: VMLiveMigrationGate{
			Enabled: false,
			Reason:  "live migration requires explicit latency, storage, and driver feature gates",
			RequiredFeatures: []string{
				"shared-or-replicated-storage",
				"bounded-rtt",
				"driver-live-migration",
			},
		},
	}

	switch defaultMode {
	case MigrationModeDisabled:
		policy.AllowedMigrationModes = []string{}
		policy.Backup.RequireRecentBackup = false
		policy.Rollback.Enabled = false
	case MigrationModeCheckpoint:
		policy.AllowedMigrationModes = []string{MigrationModeCold, MigrationModeCheckpoint}
	case MigrationModeLive:
		policy.AllowedMigrationModes = []string{MigrationModeCold, MigrationModeCheckpoint}
	}

	return policy.Normalize()
}

func (p MigrationBackupPolicy) Normalize() MigrationBackupPolicy {
	p.DefaultMigrationMode = NormalizeMigrationMode(p.DefaultMigrationMode)
	if p.DefaultMigrationMode == "" {
		p.DefaultMigrationMode = MigrationModeDisabled
	}
	p.AllowedMigrationModes = normalizeMigrationModeList(p.AllowedMigrationModes)
	if p.CheckpointTarget = strings.TrimSpace(p.CheckpointTarget); p.CheckpointTarget == "" && p.DefaultMigrationMode != MigrationModeDisabled {
		p.CheckpointTarget = "local"
	}
	p.Replication.Mode = strings.TrimSpace(strings.ToLower(p.Replication.Mode))
	if p.Replication.Mode == "" && p.Replication.Enabled {
		p.Replication.Mode = "async"
	}
	p.Backup.BackupClass = strings.TrimSpace(p.Backup.BackupClass)
	p.Backup.Retention = strings.TrimSpace(p.Backup.Retention)
	p.Rollback.Conditions = normalizePolicyStringList(p.Rollback.Conditions)
	p.LiveMigrationGate.RequiredFeatures = normalizePolicyStringList(p.LiveMigrationGate.RequiredFeatures)
	return p
}

func (p MigrationBackupPolicy) Validate() error {
	p = p.Normalize()
	if p.DefaultMigrationMode != MigrationModeDisabled && !containsPolicyValue(p.AllowedMigrationModes, p.DefaultMigrationMode) {
		return fmt.Errorf("default migration mode %q is not allowed by policy", p.DefaultMigrationMode)
	}
	if p.Replication.Factor < 0 {
		return fmt.Errorf("replication factor cannot be negative")
	}
	if p.Backup.RPOSeconds < 0 {
		return fmt.Errorf("backup rpo_seconds cannot be negative")
	}
	if p.Recovery.RTOSeconds < 0 {
		return fmt.Errorf("recovery rto_seconds cannot be negative")
	}
	if containsPolicyValue(p.AllowedMigrationModes, MigrationModeLive) && !p.LiveMigrationGate.Enabled {
		return fmt.Errorf("live migration cannot be allowed while live_migration_gate.enabled is false")
	}
	return nil
}

func (p MigrationBackupPolicy) AllowsMigrationMode(mode string) bool {
	return containsPolicyValue(p.Normalize().AllowedMigrationModes, NormalizeMigrationMode(mode))
}

func NormalizeMigrationMode(mode string) string {
	switch strings.TrimSpace(strings.ToLower(mode)) {
	case "", MigrationModeDisabled:
		return MigrationModeDisabled
	case MigrationModeCold:
		return MigrationModeCold
	case MigrationModeCheckpoint:
		return MigrationModeCheckpoint
	case MigrationModeLive:
		return MigrationModeLive
	default:
		return strings.TrimSpace(strings.ToLower(mode))
	}
}

func normalizeMigrationModeList(values []string) []string {
	normalized := make([]string, 0, len(values))
	for _, value := range values {
		mode := NormalizeMigrationMode(value)
		switch mode {
		case "", MigrationModeDisabled:
			continue
		case MigrationModeCold, MigrationModeCheckpoint, MigrationModeLive:
			if !containsPolicyValue(normalized, mode) {
				normalized = append(normalized, mode)
			}
		}
	}
	sort.Strings(normalized)
	return normalized
}

func normalizePolicyStringList(values []string) []string {
	normalized := make([]string, 0, len(values))
	for _, value := range values {
		item := strings.TrimSpace(strings.ToLower(value))
		if item == "" || containsPolicyValue(normalized, item) {
			continue
		}
		normalized = append(normalized, item)
	}
	sort.Strings(normalized)
	return normalized
}

func containsPolicyValue(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}
