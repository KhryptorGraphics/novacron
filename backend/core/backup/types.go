package backup

import (
	"time"
)

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

// Error definitions
const (
	ErrInvalidBackupType      = "invalid backup type"
	ErrBackupNotFound         = "backup not found"
	ErrInsufficientStorage    = "insufficient storage space"
	ErrEncryptionFailed       = "encryption failed"
	ErrCompressionFailed      = "compression failed"
	ErrReplicationFailed      = "replication failed"
	ErrRecoveryFailed         = "recovery failed"
	ErrVerificationFailed     = "verification failed"
	ErrUnauthorized           = "unauthorized access"
	ErrConfigurationInvalid   = "invalid configuration"
	ErrNetworkError           = "network error"
	ErrStorageError           = "storage error"
)

// ResourceUsageMetrics represents system resource usage
type ResourceUsageMetrics struct {
	CPUUsage     float64   `json:"cpu_usage"`
	MemoryUsage  int64     `json:"memory_usage"`
	DiskUsage    int64     `json:"disk_usage"`
	NetworkIO    int64     `json:"network_io"`
	Timestamp    time.Time `json:"timestamp"`
}