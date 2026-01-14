// Package shared contains common interfaces and types used across core packages
// This package breaks import cycles by providing shared abstractions
package shared

import (
	"context"
	"time"
)

// FederationManager defines the interface for managing cluster federation
// This interface is used by both vm and federation packages
type FederationManager interface {
	// RegisterCluster registers a new cluster in the federation
	RegisterCluster(ctx context.Context, clusterID string, endpoint string) error

	// UnregisterCluster removes a cluster from the federation
	UnregisterCluster(ctx context.Context, clusterID string) error

	// GetCluster retrieves cluster information
	GetCluster(ctx context.Context, clusterID string) (ClusterInfo, error)

	// ListClusters returns all registered clusters
	ListClusters(ctx context.Context) ([]ClusterInfo, error)

	// SendMessage sends a message to another cluster
	SendMessage(ctx context.Context, message interface{}) error

	// GetHealth returns the health status of the federation
	GetHealth(ctx context.Context) error
}

// ClusterInfo contains information about a federated cluster
type ClusterInfo struct {
	ID           string
	Endpoint     string
	LastSeen     time.Time
	IsHealthy    bool
	Capabilities []string
	Metadata     map[string]string
}

// DistributedStateCoordinator defines the interface for coordinating distributed VM state
type DistributedStateCoordinator interface {
	// SyncState synchronizes VM state across clusters
	SyncState(ctx context.Context, vmID string) error

	// GetState retrieves the current state of a VM
	GetState(ctx context.Context, vmID string) (interface{}, error)

	// UpdateState updates the state of a VM
	UpdateState(ctx context.Context, vmID string, state interface{}) error
}

// VMState represents the basic VM state information
type VMState struct {
	ID          string
	ClusterID   string
	Status      string
	LastUpdated time.Time
	Metadata    map[string]interface{}
}

// ConsensusLog represents a consensus log entry for replication
type ConsensusLog struct {
	Term  uint64
	Index uint64
	Type  LogType
	Data  []byte
}

// LogType represents the type of consensus log entry
type LogType string

const (
	LogTypeCommand   LogType = "command"
	LogTypeNoOp      LogType = "noop"
	LogTypeBarrier   LogType = "barrier"
	LogTypeConfig    LogType = "config"
)

// FederationManagerInterface extends FederationManager with additional methods
// This is used by the backup system to avoid import cycles
type FederationManagerInterface interface {
	// GetLocalClusterID returns the ID of the local cluster
	GetLocalClusterID() string

	// ListClusters returns all registered clusters (simplified version without context)
	ListClusters() []ClusterInfo
}

// AuthInfo contains authentication information for a cluster
type AuthInfo struct {
	AuthToken string
	Username  string
	Password  string
}

// ClusterInfoExtended extends ClusterInfo with authentication info
type ClusterInfoExtended struct {
	ClusterInfo
	AuthInfo *AuthInfo
}

// BackupManagerInterface defines the interface for backup operations
// This interface breaks the import cycle between core/backup and core/federation
type BackupManagerInterface interface {
	// CreateBackup creates a new backup
	CreateBackup(ctx context.Context, req *BackupRequest) (*BackupResult, error)

	// VerifyBackup verifies the integrity of a backup
	VerifyBackup(ctx context.Context, req *VerificationRequest) (*VerificationResult, error)

	// CreateSnapshot creates a VM snapshot
	CreateSnapshot(ctx context.Context, req *SnapshotRequest) (*SnapshotResult, error)
}

// BackupRequest represents a backup request
type BackupRequest struct {
	VMID          string
	Type          BackupType
	BaseBackupID  string
	Compression   bool
	Encryption    bool
	RetentionDays int
	Priority      Priority
}

// BackupResult represents the result of a backup operation
type BackupResult struct {
	BackupID string
	Path     string
	Size     int64
	Duration time.Duration
}

// VerificationRequest represents a backup verification request
type VerificationRequest struct {
	BackupID string
	Type     VerificationType
}

// VerificationResult represents the result of a verification operation
type VerificationResult struct {
	Valid    bool
	Message  string
	Checksum string
}

// SnapshotRequest represents a snapshot request
type SnapshotRequest struct {
	VMID        string
	Name        string
	Description string
	Memory      bool
}

// SnapshotResult represents the result of a snapshot operation
type SnapshotResult struct {
	SnapshotID string
	Path       string
	Size       int64
}

// BackupType represents the type of backup
type BackupType string

const (
	BackupTypeFull        BackupType = "full"
	BackupTypeIncremental BackupType = "incremental"
	BackupTypeDifferential BackupType = "differential"
)

// Priority represents operation priority
type Priority string

const (
	PriorityLow    Priority = "low"
	PriorityMedium Priority = "medium"
	PriorityHigh   Priority = "high"
)

// VerificationType represents the type of verification
type VerificationType string

const (
	VerificationTypeChecksum VerificationType = "checksum"
	VerificationTypeFull     VerificationType = "full"
)

// ReplicationSystemInterface defines the interface for replication operations
type ReplicationSystemInterface interface {
	// StartReplication starts a replication job
	StartReplication(ctx context.Context, resourceID string, config *ReplicationConfig) (string, error)

	// GetReplicationStatus gets the status of a replication job
	GetReplicationStatus(ctx context.Context, replicationID string) (*ReplicationStatus, error)

	// ReplicateBackup replicates a backup to a remote location
	ReplicateBackup(ctx context.Context, task *ReplicationTask) error
}

// ReplicationConfig represents replication configuration
type ReplicationConfig struct {
	SourceNode    string
	TargetNodes   []string
	Mode          ReplicationMode
	Interval      time.Duration
	RetryAttempts int
}

// ReplicationMode represents the replication mode
type ReplicationMode string

const (
	ReplicationModeSync  ReplicationMode = "sync"
	ReplicationModeAsync ReplicationMode = "async"
)

// ReplicationStatus represents the status of a replication job
type ReplicationStatus struct {
	Status   ReplicationStatusType
	Progress float64
	Error    string
}

// ReplicationStatusType represents the replication status
type ReplicationStatusType string

const (
	ReplicationStatusPending   ReplicationStatusType = "pending"
	ReplicationStatusRunning   ReplicationStatusType = "running"
	ReplicationStatusCompleted ReplicationStatusType = "completed"
	ReplicationStatusFailed    ReplicationStatusType = "failed"
)

// ReplicationTask represents a replication task
type ReplicationTask struct {
	BackupID    string
	SourcePath  string
	TargetNode  string
	TargetPath  string
	Compression bool
	Encryption  bool
}

// Logger interface for logging operations
type Logger interface {
	Info(msg string, keysAndValues ...interface{})
	Error(msg string, keysAndValues ...interface{})
	Warn(msg string, keysAndValues ...interface{})
	Debug(msg string, keysAndValues ...interface{})
}
