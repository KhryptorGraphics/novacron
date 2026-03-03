package backup

import (
	"context"
	"time"
)

// Minimal interfaces for NovaCron integration - stubs to allow backup system to build independently

// AuthService interface for authentication integration
type AuthService interface {
	ValidateToken(ctx context.Context, token string) (*AuthContext, error)
	GetUserPermissions(ctx context.Context, userID string) ([]string, error)
	CheckPermission(ctx context.Context, userID, resource, action string) error
}

// AuthContext represents authentication context
type AuthContext struct {
	UserID   string
	TenantID string
	Roles    []string
	Scopes   []string
}

// StorageService interface for storage integration
type StorageService interface {
	GetStorageInfo(ctx context.Context, path string) (*StorageInfo, error)
	CreateSnapshot(ctx context.Context, volumeID string) (*StorageSnapshot, error)
	DeleteSnapshot(ctx context.Context, snapshotID string) error
	GetAvailableSpace(ctx context.Context, path string) (int64, error)
}

// StorageInfo represents storage information
type StorageInfo struct {
	Path           string
	TotalSpace     int64
	AvailableSpace int64
	UsedSpace      int64
	FileSystem     string
}

// StorageSnapshot represents a storage snapshot
type StorageSnapshot struct {
	ID        string
	VolumeID  string
	Path      string
	Size      int64
	CreatedAt time.Time
}

// MonitoringSystem interface for monitoring integration
type MonitoringSystem interface {
	RecordMetric(ctx context.Context, name string, value float64, tags map[string]string) error
	RecordEvent(ctx context.Context, event *MonitoringEvent) error
	CreateAlert(ctx context.Context, alert *Alert) error
}

// MonitoringEvent represents a monitoring event
type MonitoringEvent struct {
	ID        string
	Type      string
	Source    string
	Message   string
	Severity  string
	Timestamp time.Time
	Data      map[string]interface{}
}

// Alert represents a monitoring alert
type Alert struct {
	ID          string
	Name        string
	Description string
	Severity    string
	Condition   string
	Actions     []string
	CreatedAt   time.Time
}

// VM interface for VM integration
type VM interface {
	GetID() string
	GetName() string
	GetState() string
	GetSize() int64
	Pause(ctx context.Context) error
	Resume(ctx context.Context) error
	GetBlockDevices(ctx context.Context) ([]BlockDevice, error)
}

// BlockDevice represents a VM block device
type BlockDevice struct {
	Name   string
	Path   string
	Size   int64
	Driver string
}

// VMManager interface for VM management integration
type VMManager interface {
	GetVM(ctx context.Context, vmID string) (VM, error)
	ListVMs(ctx context.Context) ([]VM, error)
	CreateSnapshot(ctx context.Context, vmID string, name string) error
	DeleteSnapshot(ctx context.Context, vmID, snapshotID string) error
}