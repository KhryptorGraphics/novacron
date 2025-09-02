// Package shared contains common types used across multiple packages
// to avoid circular dependencies between federation, api, and backup modules.
package shared

import (
	"context"
	"time"
)

// VMIdentifier represents a unique identifier for a VM across the federation
type VMIdentifier struct {
	ID        string `json:"id"`
	ClusterID string `json:"cluster_id"`
	NodeID    string `json:"node_id"`
}

// ResourceMetrics contains resource usage metrics
type ResourceMetrics struct {
	CPUUsage    float64 `json:"cpu_usage"`
	MemoryUsage float64 `json:"memory_usage"`
	DiskUsage   float64 `json:"disk_usage"`
	NetworkIO   float64 `json:"network_io"`
	Timestamp   time.Time `json:"timestamp"`
}

// BackupMetadata contains backup-related metadata
type BackupMetadata struct {
	BackupID     string    `json:"backup_id"`
	VMIdentifier VMIdentifier `json:"vm_identifier"`
	CreatedAt    time.Time `json:"created_at"`
	Size         int64     `json:"size"`
	Type         string    `json:"type"`
	Location     string    `json:"location"`
}

// FederationNode represents a node in the federation
type FederationNode struct {
	ID          string            `json:"id"`
	Name        string            `json:"name"`
	ClusterID   string            `json:"cluster_id"`
	Address     string            `json:"address"`
	State       string            `json:"state"`
	Role        string            `json:"role"`
	Metadata    map[string]string `json:"metadata"`
	LastSeen    time.Time         `json:"last_seen"`
	JoinedAt    time.Time         `json:"joined_at"`
}

// APIRequest represents a generic API request structure
type APIRequest struct {
	ID        string                 `json:"id"`
	Method    string                 `json:"method"`
	Path      string                 `json:"path"`
	Headers   map[string]string      `json:"headers"`
	Body      interface{}            `json:"body"`
	Timestamp time.Time              `json:"timestamp"`
}

// APIResponse represents a generic API response structure
type APIResponse struct {
	ID        string                 `json:"id"`
	Status    int                    `json:"status"`
	Headers   map[string]string      `json:"headers"`
	Body      interface{}            `json:"body"`
	Error     string                 `json:"error,omitempty"`
	Timestamp time.Time              `json:"timestamp"`
}

// OperationContext provides context for operations across modules
type OperationContext struct {
	Ctx           context.Context
	RequestID     string
	UserID        string
	FederationID  string
	TraceID       string
	SpanID        string
}

// HealthStatus represents the health status of a component
type HealthStatus struct {
	Component   string    `json:"component"`
	Status      string    `json:"status"` // healthy, degraded, unhealthy
	Message     string    `json:"message,omitempty"`
	LastChecked time.Time `json:"last_checked"`
	Metrics     *ResourceMetrics `json:"metrics,omitempty"`
}

// EventType represents types of events in the system
type EventType string

const (
	EventTypeVMCreated    EventType = "vm.created"
	EventTypeVMDeleted    EventType = "vm.deleted"
	EventTypeVMStarted    EventType = "vm.started"
	EventTypeVMStopped    EventType = "vm.stopped"
	EventTypeBackupStarted EventType = "backup.started"
	EventTypeBackupCompleted EventType = "backup.completed"
	EventTypeNodeJoined   EventType = "node.joined"
	EventTypeNodeLeft     EventType = "node.left"
)

// Event represents a system event
type Event struct {
	ID        string                 `json:"id"`
	Type      EventType              `json:"type"`
	Source    string                 `json:"source"`
	Data      map[string]interface{} `json:"data"`
	Timestamp time.Time              `json:"timestamp"`
}

// Permission represents an access permission
type Permission struct {
	Resource string `json:"resource"`
	Action   string `json:"action"`
	Effect   string `json:"effect"` // allow or deny
}

// User represents a system user
type User struct {
	ID          string       `json:"id"`
	Username    string       `json:"username"`
	Email       string       `json:"email"`
	Role        string       `json:"role"`
	Permissions []Permission `json:"permissions"`
	CreatedAt   time.Time    `json:"created_at"`
	UpdatedAt   time.Time    `json:"updated_at"`
}

// CloudProvider represents a cloud provider configuration
type CloudProvider struct {
	ID       string            `json:"id"`
	Name     string            `json:"name"`
	Type     string            `json:"type"` // aws, azure, gcp, onprem
	Region   string            `json:"region"`
	Endpoint string            `json:"endpoint"`
	Config   map[string]string `json:"config"`
	Active   bool              `json:"active"`
}

// StorageBackend represents a storage backend configuration
type StorageBackend struct {
	ID       string            `json:"id"`
	Name     string            `json:"name"`
	Type     string            `json:"type"` // s3, azure-blob, gcs, nfs
	Endpoint string            `json:"endpoint"`
	Config   map[string]string `json:"config"`
	Active   bool              `json:"active"`
}

// JobStatus represents the status of an async job
type JobStatus string

const (
	JobStatusPending   JobStatus = "pending"
	JobStatusRunning   JobStatus = "running"
	JobStatusCompleted JobStatus = "completed"
	JobStatusFailed    JobStatus = "failed"
	JobStatusCancelled JobStatus = "cancelled"
)

// Job represents an asynchronous job
type Job struct {
	ID          string                 `json:"id"`
	Type        string                 `json:"type"`
	Status      JobStatus              `json:"status"`
	Progress    int                    `json:"progress"` // 0-100
	StartedAt   time.Time              `json:"started_at"`
	CompletedAt *time.Time             `json:"completed_at,omitempty"`
	Result      map[string]interface{} `json:"result,omitempty"`
	Error       string                 `json:"error,omitempty"`
}