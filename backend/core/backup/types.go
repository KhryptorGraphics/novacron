package backup

import (
	"errors"
	"time"
)

// BackupFilter defines filtering criteria for listing backups
type BackupFilter struct {
	TenantID  string
	VMID      string
	Type      string
	State     string
	StartDate time.Time
	EndDate   time.Time
	JobID     string
}

// Common severity levels used across the backup system
const (
	SeverityCritical = "critical"
	SeverityHigh     = "high"
	SeverityMedium   = "medium"
	SeverityLow      = "low"
	SeverityInfo     = "info"
)

// HealthStatus represents the health status of backup components
type HealthStatus string

const (
	HealthStatusHealthy   HealthStatus = "healthy"
	HealthStatusDegraded  HealthStatus = "degraded"
	HealthStatusUnhealthy HealthStatus = "unhealthy"
	HealthStatusUnknown   HealthStatus = "unknown"
)

// BackupType defines the type of backup operation
type BackupType string

const (
	FullBackup         BackupType = "full"
	IncrementalBackup  BackupType = "incremental"
	DifferentialBackup BackupType = "differential"
	SnapshotBackup     BackupType = "snapshot"
)

// BackupStatus represents the status of a backup operation
type BackupStatus string

const (
	BackupStatusPending    BackupStatus = "pending"
	BackupStatusRunning    BackupStatus = "running"
	BackupStatusCompleted  BackupStatus = "completed"
	BackupStatusFailed     BackupStatus = "failed"
	BackupStatusCancelled  BackupStatus = "cancelled"
	BackupStatusRetrying   BackupStatus = "retrying"
)

// SystemReplicationStatus represents the status of system-wide replication
type SystemReplicationStatus string

const (
	SystemReplicationStatusActive    SystemReplicationStatus = "active"
	SystemReplicationStatusSyncing   SystemReplicationStatus = "syncing"
	SystemReplicationStatusFailed    SystemReplicationStatus = "failed"
	SystemReplicationStatusPaused    SystemReplicationStatus = "paused"
	SystemReplicationStatusStopped   SystemReplicationStatus = "stopped"
)

// SystemRecoveryStatus represents the status of system-wide disaster recovery operations
type SystemRecoveryStatus string

const (
	SystemRecoveryStatusReady      SystemRecoveryStatus = "ready"
	SystemRecoveryStatusInProgress SystemRecoveryStatus = "in_progress"
	SystemRecoveryStatusCompleted  SystemRecoveryStatus = "completed"
	SystemRecoveryStatusFailed     SystemRecoveryStatus = "failed"
	SystemRecoveryStatusTesting    SystemRecoveryStatus = "testing"
)

// EncryptionType defines the type of encryption used
type EncryptionType string

const (
	EncryptionTypeNone   EncryptionType = "none"
	EncryptionTypeAES256 EncryptionType = "aes256"
	EncryptionTypeRSA    EncryptionType = "rsa"
)

// CompressionType defines the type of compression used
type CompressionType string

const (
	CompressionTypeNone CompressionType = "none"
	CompressionTypeGzip CompressionType = "gzip"
	CompressionTypeLZ4  CompressionType = "lz4"
	CompressionTypeZstd CompressionType = "zstd"
)

// CloudProvider defines supported cloud storage providers
type CloudProvider string

const (
	CloudProviderAWS   CloudProvider = "aws"
	CloudProviderAzure CloudProvider = "azure"
	CloudProviderGCP   CloudProvider = "gcp"
	CloudProviderLocal CloudProvider = "local"
)

// ScheduleFrequency defines backup schedule frequencies
type ScheduleFrequency string

const (
	ScheduleFrequencyHourly  ScheduleFrequency = "hourly"
	ScheduleFrequencyDaily   ScheduleFrequency = "daily"
	ScheduleFrequencyWeekly  ScheduleFrequency = "weekly"
	ScheduleFrequencyMonthly ScheduleFrequency = "monthly"
	ScheduleFrequencyCustom  ScheduleFrequency = "custom"
)

// Common time-based constants
const (
	DefaultBackupTimeout    = 30 * time.Minute
	DefaultRPOThreshold     = 1 * time.Hour
	DefaultRTOThreshold     = 4 * time.Hour
	DefaultRetentionPeriod  = 30 * 24 * time.Hour // 30 days
	DefaultReplicationLag   = 5 * time.Minute
	DefaultHealthCheckInterval = 1 * time.Minute
	MaxIncrementals         = 10 // Maximum incremental backups before full backup
)

// Error definitions - sentinel errors for typed error checking
var (
	ErrInvalidBackupType      = errors.New("invalid backup type")
	ErrBackupNotFound         = errors.New("backup not found")
	ErrInsufficientStorage    = errors.New("insufficient storage space")
	ErrEncryptionFailed       = errors.New("encryption failed")
	ErrCompressionFailed      = errors.New("compression failed")
	ErrReplicationFailed      = errors.New("replication failed")
	ErrRecoveryFailed         = errors.New("recovery failed")
	ErrVerificationFailed     = errors.New("verification failed")
	ErrUnauthorized           = errors.New("unauthorized access")
	ErrConfigurationInvalid   = errors.New("invalid configuration")
	ErrNetworkError           = errors.New("network error")
	ErrStorageError           = errors.New("storage error")
)

// ResourceUsageMetrics represents system resource usage
type ResourceUsageMetrics struct {
	CPUUsage     float64   `json:"cpu_usage"`
	MemoryUsage  int64     `json:"memory_usage"`
	DiskUsage    int64     `json:"disk_usage"`
	NetworkIO    int64     `json:"network_io"`
	Timestamp    time.Time `json:"timestamp"`
}

// VerificationResult represents the result of a backup verification
type VerificationResult struct {
	BackupID         string                 `json:"backup_id"`
	Status           string                 `json:"status"`
	CheckedItems     int                    `json:"checked_items"`
	ErrorsFound      []string               `json:"errors_found"`
	VerificationTime time.Time              `json:"verification_time"`
	Details          map[string]interface{} `json:"details,omitempty"`
}