package shared

import (
	"context"
	"time"
)

// FederationManagerInterface defines the interface for federation management
type FederationManagerInterface interface {
	IsLeader() bool
	GetNodes(ctx context.Context) ([]FederationNode, error)
	GetLeader(ctx context.Context) (*FederationNode, error)
	RequestResources(ctx context.Context, req *ResourceRequest) (*ResourceAllocation, error)
	ReleaseResources(ctx context.Context, allocationID string) error
	ListClusters() []ClusterInfo
	GetLocalClusterID() string
	GetNode(ctx context.Context, nodeID string) (*FederationNode, error)
	GetHealth(ctx context.Context) (*HealthStatus, error)
	JoinFederation(ctx context.Context, joinAddresses []string) error
	LeaveFederation(ctx context.Context) error
}

// FederationNode represents a node in the federation
type FederationNode struct {
	ID           string            `json:"id"`
	Address      string            `json:"address"`
	Status       string            `json:"status"`
	LastSeen     time.Time         `json:"last_seen"`
	Capabilities NodeCapabilities  `json:"capabilities"`
	Metadata     map[string]string `json:"metadata"`
}

// NodeCapabilities represents the capabilities of a federation node
type NodeCapabilities struct {
	Resources    ResourceInventory `json:"resources"`
	Features     []string          `json:"features"`
	Version      string            `json:"version"`
	Architecture string            `json:"architecture"`
}

// ResourceInventory tracks resource usage
type ResourceInventory struct {
	TotalCPU      int     `json:"total_cpu"`
	UsedCPU       int     `json:"used_cpu"`
	TotalMemory   int64   `json:"total_memory"`
	UsedMemory    int64   `json:"used_memory"`
	TotalStorage  int64   `json:"total_storage"`
	UsedStorage   int64   `json:"used_storage"`
	VMs           int     `json:"vms"`
	Containers    int     `json:"containers"`
	NetworkPools  int     `json:"network_pools"`
}

// ResourceRequest represents a request for resources
type ResourceRequest struct {
	ID           string        `json:"id"`
	ResourceType string        `json:"resource_type"`
	CPUCores     int           `json:"cpu_cores"`
	MemoryGB     int           `json:"memory_gb"`
	StorageGB    int           `json:"storage_gb"`
	Duration     time.Duration `json:"duration"`
	Priority     int           `json:"priority"`
	Requirements map[string]interface{} `json:"requirements"`
}

// ResourceAllocation represents an allocated resource
type ResourceAllocation struct {
	ID           string                 `json:"id"`
	RequestID    string                 `json:"request_id"`
	NodeID       string                 `json:"node_id"`
	Resources    AllocatedResources     `json:"resources"`
	Status       string                 `json:"status"`
	AllocatedAt  time.Time              `json:"allocated_at"`
	ExpiresAt    time.Time              `json:"expires_at"`
	Metadata     map[string]interface{} `json:"metadata"`
}

// AllocatedResources represents the actually allocated resources
type AllocatedResources struct {
	CPUCores  int   `json:"cpu_cores"`
	MemoryGB  int   `json:"memory_gb"`
	StorageGB int   `json:"storage_gb"`
	NetworkID string `json:"network_id,omitempty"`
}

// ClusterInfo represents information about a cluster
type ClusterInfo struct {
	ID       string                 `json:"id"`
	Name     string                 `json:"name"`
	Endpoint string                 `json:"endpoint"`
	Status   string                 `json:"status"`
	AuthInfo *AuthInfo              `json:"auth_info,omitempty"`
	Metadata map[string]interface{} `json:"metadata"`
}

// AuthInfo represents authentication information
type AuthInfo struct {
	AuthToken string `json:"auth_token"`
	Username  string `json:"username"`
	Password  string `json:"password"`
}

// HealthStatus represents health status information
type HealthStatus struct {
	Status    string            `json:"status"`
	Message   string            `json:"message"`
	Timestamp time.Time         `json:"timestamp"`
	Details   map[string]interface{} `json:"details"`
}

// VMManagerInterface defines the interface for VM management
type VMManagerInterface interface {
	CreateVM(ctx context.Context, config VMConfig) (*VM, error)
	DeleteVM(ctx context.Context, vmID string) error
	StartVM(ctx context.Context, vmID string) error
	StopVM(ctx context.Context, vmID string) error
	RestartVM(ctx context.Context, vmID string) error
	GetVM(ctx context.Context, vmID string) (*VM, error)
	ListVMs(ctx context.Context) ([]*VM, error)
	ListMigrations(ctx context.Context) ([]*Migration, error)
	MigrateVM(ctx context.Context, vmID, targetNodeID string) error
}

// VM represents a virtual machine
type VM struct {
	ID       string            `json:"id"`
	Name     string            `json:"name"`
	Status   string            `json:"status"`
	CPU      int               `json:"cpu"`
	Memory   int64             `json:"memory"`
	Storage  int64             `json:"storage"`
	Networks []NetworkConfig   `json:"networks"`
	Metadata map[string]string `json:"metadata"`
}

// VMConfig represents VM configuration
type VMConfig struct {
	Name      string            `json:"name"`
	CPU       int               `json:"cpu"`
	Memory    int64             `json:"memory"`
	Storage   int64             `json:"storage"`
	Networks  []NetworkConfig   `json:"networks"`
	Image     string            `json:"image"`
	Metadata  map[string]string `json:"metadata"`
}

// NetworkConfig represents network configuration
type NetworkConfig struct {
	Name     string `json:"name"`
	Type     string `json:"type"`
	Address  string `json:"address"`
	Gateway  string `json:"gateway"`
	Netmask  string `json:"netmask"`
}

// Migration represents a VM migration
type Migration struct {
	ID         string    `json:"id"`
	VMID       string    `json:"vm_id"`
	SourceNode string    `json:"source_node"`
	TargetNode string    `json:"target_node"`
	Status     string    `json:"status"`
	Progress   float64   `json:"progress"`
	StartedAt  time.Time `json:"started_at"`
	CompletedAt *time.Time `json:"completed_at,omitempty"`
}

// Logger interface for logging
type Logger interface {
	Info(msg string, args ...interface{})
	Error(msg string, args ...interface{})
	Warn(msg string, args ...interface{})
	Debug(msg string, args ...interface{})
}

// BackupManagerInterface defines the interface for backup management
type BackupManagerInterface interface {
	CreateBackup(ctx context.Context, req *BackupRequest) (*BackupResult, error)
	VerifyBackup(ctx context.Context, req *VerificationRequest) (*VerificationResult, error)
	CreateSnapshot(ctx context.Context, req *SnapshotRequest) (*Snapshot, error)
}

// ReplicationSystemInterface defines the interface for replication
type ReplicationSystemInterface interface {
	StartReplication(ctx context.Context, resourceID string, config *ReplicationConfig) (string, error)
	GetReplicationStatus(ctx context.Context, replicationID string) (*ReplicationStatus, error)
	ReplicateBackup(ctx context.Context, task *ReplicationTask) error
}

// BackupRequest represents a backup request
type BackupRequest struct {
	VMID         string      `json:"vm_id"`
	Type         BackupType  `json:"type"`
	Compression  bool        `json:"compression"`
	Encryption   bool        `json:"encryption"`
	RetentionDays int        `json:"retention_days"`
	Priority     BackupPriority `json:"priority"`
	BaseBackupID string      `json:"base_backup_id,omitempty"`
}

// BackupResult represents the result of a backup operation
type BackupResult struct {
	BackupID string `json:"backup_id"`
	Path     string `json:"path"`
	Size     int64  `json:"size"`
}

// VerificationRequest represents a backup verification request
type VerificationRequest struct {
	BackupID string           `json:"backup_id"`
	Type     VerificationType `json:"type"`
}

// VerificationResult represents the result of backup verification
type VerificationResult struct {
	Valid   bool   `json:"valid"`
	Message string `json:"message"`
}

// SnapshotRequest represents a snapshot request
type SnapshotRequest struct {
	VMID        string `json:"vm_id"`
	Name        string `json:"name"`
	Description string `json:"description"`
	Memory      bool   `json:"memory"`
}

// Snapshot represents a VM snapshot
type Snapshot struct {
	ID          string    `json:"id"`
	VMID        string    `json:"vm_id"`
	Name        string    `json:"name"`
	Description string    `json:"description"`
	CreatedAt   time.Time `json:"created_at"`
}

// ReplicationConfig represents replication configuration
type ReplicationConfig struct {
	SourceNode    string              `json:"source_node"`
	TargetNodes   []string            `json:"target_nodes"`
	Mode          ReplicationMode     `json:"mode"`
	Interval      time.Duration       `json:"interval"`
	RetryAttempts int                 `json:"retry_attempts"`
}

// ReplicationStatus represents replication status
type ReplicationStatus struct {
	Status   ReplicationStatusType `json:"status"`
	Progress float64               `json:"progress"`
	Error    string                `json:"error,omitempty"`
}

// ReplicationTask represents a replication task
type ReplicationTask struct {
	BackupID     string `json:"backup_id"`
	SourcePath   string `json:"source_path"`
	TargetNode   string `json:"target_node"`
	TargetPath   string `json:"target_path"`
	Compression  bool   `json:"compression"`
	Encryption   bool   `json:"encryption"`
}

// Enums and constants
type BackupType string

const (
	BackupTypeFull        BackupType = "full"
	BackupTypeIncremental BackupType = "incremental"
	BackupTypeDifferential BackupType = "differential"
)

type BackupPriority string

const (
	PriorityLow    BackupPriority = "low"
	PriorityMedium BackupPriority = "medium"
	PriorityHigh   BackupPriority = "high"
)

type VerificationType string

const (
	VerificationTypeChecksum VerificationType = "checksum"
	VerificationTypeIntegrity VerificationType = "integrity"
)

type ReplicationMode string

const (
	ReplicationModeSync  ReplicationMode = "sync"
	ReplicationModeAsync ReplicationMode = "async"
)

type ReplicationStatusType string

const (
	ReplicationStatusPending   ReplicationStatusType = "pending"
	ReplicationStatusRunning   ReplicationStatusType = "running"
	ReplicationStatusCompleted ReplicationStatusType = "completed"
	ReplicationStatusFailed    ReplicationStatusType = "failed"
)