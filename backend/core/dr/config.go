package dr

import (
	"encoding/json"
	"fmt"
	"os"
	"time"
)

// DRConfig contains all disaster recovery configuration
type DRConfig struct {
	// Backup configuration
	BackupSchedule  BackupSchedule  `json:"backup_schedule"`
	RetentionPolicy RetentionPolicy `json:"retention_policy"`
	BackupLocations []BackupLocation `json:"backup_locations"`

	// Failover configuration
	FailoverPolicy FailoverPolicy `json:"failover_policy"`

	// RTO/RPO targets
	RTO time.Duration `json:"rto"` // 30 minutes default
	RPO time.Duration `json:"rpo"` // 5 minutes default

	// Automation settings
	AutoFailover      bool     `json:"auto_failover"`
	RequireApproval   bool     `json:"require_approval"`
	NotificationTargets []string `json:"notification_targets"`

	// Multi-region settings
	PrimaryRegion     string   `json:"primary_region"`
	SecondaryRegions  []string `json:"secondary_regions"`
	MinActiveRegions  int      `json:"min_active_regions"`

	// Health monitoring
	HealthChecks []HealthCheck `json:"health_checks"`

	// Chaos engineering
	ChaosEnabled      bool          `json:"chaos_enabled"`
	ChaosSchedule     string        `json:"chaos_schedule"` // Cron expression
	ChaosBlastRadius  int           `json:"chaos_blast_radius"`

	// Security
	EncryptionEnabled bool   `json:"encryption_enabled"`
	EncryptionKeyID   string `json:"encryption_key_id"`

	// Compliance
	ComplianceMode    string `json:"compliance_mode"`
	AuditLogRetention int    `json:"audit_log_retention_days"`

	// Testing
	DRTestSchedule    string `json:"dr_test_schedule"` // Cron expression
	TestingEnabled    bool   `json:"testing_enabled"`
}

// DefaultDRConfig returns production-ready default configuration
func DefaultDRConfig() *DRConfig {
	return &DRConfig{
		BackupSchedule: BackupSchedule{
			FullBackup:        "0 0 * * *",      // Daily at midnight
			IncrementalBackup: "0 * * * *",      // Hourly
			TransactionLog:    true,             // Continuous
			SnapshotInterval:  1 * time.Hour,
		},
		RetentionPolicy: RetentionPolicy{
			HourlyRetentionDays:  7,
			DailyRetentionDays:   30,
			WeeklyRetentionDays:  90,
			MonthlyRetentionDays: 365,
			YearlyRetentionYears: 7,
			LegalHoldEnabled:     false,
			ComplianceMode:       "SOC2",
		},
		FailoverPolicy: FailoverPolicy{
			AutomaticTriggers: []TriggerCondition{
				{
					MetricName:    "region_health_score",
					Threshold:     0.3,
					Duration:      2 * time.Minute,
					Operator:      "lt",
					RequireQuorum: true,
				},
				{
					MetricName:    "error_rate",
					Threshold:     0.5,
					Duration:      1 * time.Minute,
					Operator:      "gt",
					RequireQuorum: true,
				},
			},
			MinHealthyRegions:   2,
			QuorumRequirement:   2,
			MaxFailoverAttempts: 3,
			RollbackOnFailure:   true,
			ApprovalRequired:    false,
			NotifyBeforeFailover: true,
		},
		RTO:               30 * time.Minute,
		RPO:               5 * time.Minute,
		AutoFailover:      true,
		RequireApproval:   false,
		MinActiveRegions:  2,
		HealthChecks: []HealthCheck{
			{
				Level:              1,
				Name:               "process_liveness",
				Endpoint:           "/healthz",
				Interval:           10 * time.Second,
				Timeout:            5 * time.Second,
				HealthyThreshold:   2,
				UnhealthyThreshold: 3,
				ExpectedStatus:     200,
			},
			{
				Level:              2,
				Name:               "service_readiness",
				Endpoint:           "/ready",
				Interval:           30 * time.Second,
				Timeout:            10 * time.Second,
				HealthyThreshold:   2,
				UnhealthyThreshold: 2,
				ExpectedStatus:     200,
			},
			{
				Level:              3,
				Name:               "regional_capacity",
				Endpoint:           "/metrics/capacity",
				Interval:           1 * time.Minute,
				Timeout:            15 * time.Second,
				HealthyThreshold:   1,
				UnhealthyThreshold: 2,
				ExpectedStatus:     200,
			},
			{
				Level:              4,
				Name:               "global_health",
				Endpoint:           "/metrics/global",
				Interval:           5 * time.Minute,
				Timeout:            30 * time.Second,
				HealthyThreshold:   1,
				UnhealthyThreshold: 1,
				ExpectedStatus:     200,
			},
		},
		ChaosEnabled:      false, // Disabled by default, enable in staging
		ChaosSchedule:     "0 2 * * 0", // Sunday 2 AM
		ChaosBlastRadius:  5,
		EncryptionEnabled: true,
		ComplianceMode:    "SOC2",
		AuditLogRetention: 365,
		DRTestSchedule:    "0 3 1 * *", // First day of month at 3 AM
		TestingEnabled:    true,
	}
}

// LoadDRConfig loads configuration from file
func LoadDRConfig(configPath string) (*DRConfig, error) {
	data, err := os.ReadFile(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	var config DRConfig
	if err := json.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("failed to parse config: %w", err)
	}

	return &config, nil
}

// SaveDRConfig saves configuration to file
func (c *DRConfig) Save(configPath string) error {
	data, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal config: %w", err)
	}

	if err := os.WriteFile(configPath, data, 0600); err != nil {
		return fmt.Errorf("failed to write config file: %w", err)
	}

	return nil
}

// Validate validates the DR configuration
func (c *DRConfig) Validate() error {
	if c.RTO <= 0 {
		return fmt.Errorf("RTO must be positive")
	}
	if c.RPO <= 0 {
		return fmt.Errorf("RPO must be positive")
	}
	if c.RPO > c.RTO {
		return fmt.Errorf("RPO cannot be greater than RTO")
	}
	if c.MinActiveRegions < 1 {
		return fmt.Errorf("min_active_regions must be at least 1")
	}
	if len(c.SecondaryRegions) < c.MinActiveRegions-1 {
		return fmt.Errorf("not enough secondary regions configured")
	}
	if c.FailoverPolicy.QuorumRequirement < 1 {
		return fmt.Errorf("quorum requirement must be at least 1")
	}
	if len(c.BackupLocations) == 0 {
		return fmt.Errorf("at least one backup location must be configured")
	}
	return nil
}

// GetBackupLocationByPriority returns backup locations sorted by priority
func (c *DRConfig) GetBackupLocationByPriority() []BackupLocation {
	// Make a copy to avoid modifying the original
	locations := make([]BackupLocation, len(c.BackupLocations))
	copy(locations, c.BackupLocations)

	// Simple insertion sort by priority
	for i := 1; i < len(locations); i++ {
		key := locations[i]
		j := i - 1
		for j >= 0 && locations[j].Priority > key.Priority {
			locations[j+1] = locations[j]
			j--
		}
		locations[j+1] = key
	}

	return locations
}
