package storage

import (
	"context"
	"errors"
	"fmt"
	"io"
	"sync"
	"time"
)

// Common error definitions
var (
	ErrVolumeNotFound    = errors.New("volume not found")
	ErrVolumeExists      = errors.New("volume already exists")
	ErrVolumeInUse       = errors.New("volume is in use")
	ErrInvalidVolumeName = errors.New("invalid volume name")
	ErrStorageFull       = errors.New("storage is full")
	ErrInvalidOperation  = errors.New("invalid operation")
	ErrPermissionDenied  = errors.New("permission denied")
	ErrNotImplemented    = errors.New("not implemented")
)

// VolumeType represents the type of volume
type VolumeType string

const (
	// VolumeTypeBlock represents a block device volume
	VolumeTypeBlock VolumeType = "block"

	// VolumeTypeFile represents a file-based volume
	VolumeTypeFile VolumeType = "file"

	// VolumeTypeObject represents an object storage volume
	VolumeTypeObject VolumeType = "object"

	// VolumeTypeDistributed represents a distributed filesystem volume
	VolumeTypeDistributed VolumeType = "distributed"

	// VolumeTypeEphemeral represents an ephemeral volume that is lost when the VM is stopped
	VolumeTypeEphemeral VolumeType = "ephemeral"
)

// VolumeState represents the state of a volume
type VolumeState string

const (
	// VolumeStateCreating indicates the volume is being created
	VolumeStateCreating VolumeState = "creating"

	// VolumeStateAvailable indicates the volume is available for use
	VolumeStateAvailable VolumeState = "available"

	// VolumeStateAttaching indicates the volume is being attached to a VM
	VolumeStateAttaching VolumeState = "attaching"

	// VolumeStateAttached indicates the volume is attached to a VM
	VolumeStateAttached VolumeState = "attached"

	// VolumeStateDetaching indicates the volume is being detached from a VM
	VolumeStateDetaching VolumeState = "detaching"

	// VolumeStateDeleting indicates the volume is being deleted
	VolumeStateDeleting VolumeState = "deleting"

	// VolumeStateFailed indicates the volume is in a failed state
	VolumeStateFailed VolumeState = "failed"
)

// VolumeInfo contains information about a volume
type VolumeInfo struct {
	// Unique identifier for the volume
	ID string `json:"id"`

	// Human-readable name for the volume
	Name string `json:"name"`

	// Type of volume
	Type VolumeType `json:"type"`

	// Current state of the volume
	State VolumeState `json:"state"`

	// Size of the volume in bytes
	Size int64 `json:"size"`

	// ID of the VM this volume is attached to (if any)
	AttachedToVM string `json:"attached_to_vm,omitempty"`

	// When the volume was created
	CreatedAt time.Time `json:"created_at"`

	// When the volume was last modified
	UpdatedAt time.Time `json:"updated_at"`

	// Custom metadata for the volume
	Metadata map[string]string `json:"metadata,omitempty"`

	// Whether the volume is bootable
	Bootable bool `json:"bootable"`

	// Whether the volume is encrypted
	Encrypted bool `json:"encrypted"`

	// Replication factor for the volume
	ReplicationFactor int `json:"replication_factor"`

	// Nodes this volume is stored on
	NodeIDs []string `json:"node_ids,omitempty"`

	// Performance metrics (IOPS, throughput)
	IOPSRead  int `json:"iops_read"`
	IOPSWrite int `json:"iops_write"`
	MBpsRead  int `json:"mbps_read"`
	MBpsWrite int `json:"mbps_write"`
}

// VolumeCreateOptions contains options for creating a volume
type VolumeCreateOptions struct {
	// Name of the volume
	Name string `json:"name"`

	// Type of volume
	Type VolumeType `json:"type"`

	// Size of the volume in bytes
	Size int64 `json:"size"`

	// Source for the volume (optional)
	Source string `json:"source,omitempty"`

	// Source type ("snapshot", "volume", "image", "empty")
	SourceType string `json:"source_type,omitempty"`

	// Whether the volume is bootable
	Bootable bool `json:"bootable"`

	// Whether the volume should be encrypted
	Encrypted bool `json:"encrypted"`

	// Encryption key (optional)
	EncryptionKey string `json:"encryption_key,omitempty"`

	// Desired replication factor
	ReplicationFactor int `json:"replication_factor"`

	// Specific nodes to store the volume on (optional)
	NodeIDs []string `json:"node_ids,omitempty"`

	// Custom metadata for the volume
	Metadata map[string]string `json:"metadata,omitempty"`
}

// VolumeAttachOptions contains options for attaching a volume
type VolumeAttachOptions struct {
	// ID of the VM to attach to
	VMID string `json:"vm_id"`

	// Device name (optional)
	Device string `json:"device,omitempty"`

	// Read-only attachment
	ReadOnly bool `json:"read_only"`
}

// VolumeDetachOptions contains options for detaching a volume
type VolumeDetachOptions struct {
	// Force detachment even if the VM is not responding
	Force bool `json:"force"`
}

// VolumeResizeOptions contains options for resizing a volume
type VolumeResizeOptions struct {
	// New size in bytes
	NewSize int64 `json:"new_size"`
}

// VolumeEventType represents the type of volume event
type VolumeEventType string

const (
	// VolumeEventCreated indicates the volume was created
	VolumeEventCreated VolumeEventType = "created"

	// VolumeEventDeleted indicates the volume was deleted
	VolumeEventDeleted VolumeEventType = "deleted"

	// VolumeEventAttached indicates the volume was attached to a VM
	VolumeEventAttached VolumeEventType = "attached"

	// VolumeEventDetached indicates the volume was detached from a VM
	VolumeEventDetached VolumeEventType = "detached"

	// VolumeEventResized indicates the volume was resized
	VolumeEventResized VolumeEventType = "resized"

	// VolumeEventStateChanged indicates the volume state has changed
	VolumeEventStateChanged VolumeEventType = "state_changed"
)

// VolumeEvent represents an event related to a volume
type VolumeEvent struct {
	// Type of event
	Type VolumeEventType `json:"type"`

	// ID of the volume
	VolumeID string `json:"volume_id"`

	// Name of the volume
	VolumeName string `json:"volume_name"`

	// Additional data for the event
	Data interface{} `json:"data,omitempty"`

	// Time of the event
	Timestamp time.Time `json:"timestamp"`
}

// VolumeEventListener is a function that handles volume events
type VolumeEventListener func(event VolumeEvent)

// StorageConfig contains configuration for the storage service
type StorageConfig struct {
	// Root directory for storage
	RootDir string `json:"root_dir"`

	// Maximum storage capacity in bytes (0 = unlimited)
	MaxCapacity int64 `json:"max_capacity"`

	// Default volume type
	DefaultVolumeType VolumeType `json:"default_volume_type"`

	// Default replication factor
	DefaultReplicationFactor int `json:"default_replication_factor"`

	// Whether to enable encryption by default
	DefaultEncryption bool `json:"default_encryption"`

	// Storage driver to use
	Driver string `json:"driver"`

	// Driver-specific options
	DriverOptions map[string]string `json:"driver_options,omitempty"`
}

// DefaultStorageConfig returns a default configuration
func DefaultStorageConfig() StorageConfig {
	return StorageConfig{
		RootDir:                  "/var/lib/novacron/storage",
		MaxCapacity:              0, // Unlimited
		DefaultVolumeType:        VolumeTypeFile,
		DefaultReplicationFactor: 1,
		DefaultEncryption:        false,
		Driver:                   "local",
		DriverOptions:            make(map[string]string),
	}
}

// StorageService defines the interface for storage operations
type StorageService interface {
	// Start starts the storage service
	Start() error

	// Stop stops the storage service
	Stop() error

	// CreateVolume creates a new volume
	CreateVolume(ctx context.Context, opts VolumeCreateOptions) (*VolumeInfo, error)

	// DeleteVolume deletes a volume
	DeleteVolume(ctx context.Context, volumeID string) error

	// GetVolume returns information about a volume
	GetVolume(ctx context.Context, volumeID string) (*VolumeInfo, error)

	// ListVolumes lists all volumes
	ListVolumes(ctx context.Context) ([]VolumeInfo, error)

	// AttachVolume attaches a volume to a VM
	AttachVolume(ctx context.Context, volumeID string, opts VolumeAttachOptions) error

	// DetachVolume detaches a volume from a VM
	DetachVolume(ctx context.Context, volumeID string, opts VolumeDetachOptions) error

	// ResizeVolume resizes a volume
	ResizeVolume(ctx context.Context, volumeID string, opts VolumeResizeOptions) error

	// OpenVolume opens a volume for reading/writing
	OpenVolume(ctx context.Context, volumeID string) (io.ReadWriteCloser, error)

	// GetVolumeStats returns statistics for a volume
	GetVolumeStats(ctx context.Context, volumeID string) (map[string]interface{}, error)

	// AddVolumeEventListener adds a listener for volume events
	AddVolumeEventListener(listener VolumeEventListener)

	// RemoveVolumeEventListener removes a listener for volume events
	RemoveVolumeEventListener(listener VolumeEventListener)
}

// BaseStorageService provides a base implementation of StorageService
type BaseStorageService struct {
	config    StorageConfig
	volumes   map[string]VolumeInfo
	listeners []VolumeEventListener
	mu        sync.RWMutex
	ctx       context.Context
	cancel    context.CancelFunc
	running   bool
}

// NewBaseStorageService creates a new base storage service
func NewBaseStorageService(config StorageConfig) *BaseStorageService {
	ctx, cancel := context.WithCancel(context.Background())
	return &BaseStorageService{
		config:    config,
		volumes:   make(map[string]VolumeInfo),
		listeners: make([]VolumeEventListener, 0),
		ctx:       ctx,
		cancel:    cancel,
		running:   false,
	}
}

// Start starts the storage service
func (s *BaseStorageService) Start() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.running {
		return fmt.Errorf("storage service already running")
	}

	s.running = true
	return nil
}

// Stop stops the storage service
func (s *BaseStorageService) Stop() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if !s.running {
		return nil
	}

	s.cancel()
	s.running = false
	return nil
}

// CreateVolume creates a new volume
func (s *BaseStorageService) CreateVolume(ctx context.Context, opts VolumeCreateOptions) (*VolumeInfo, error) {
	return nil, ErrNotImplemented
}

// DeleteVolume deletes a volume
func (s *BaseStorageService) DeleteVolume(ctx context.Context, volumeID string) error {
	return ErrNotImplemented
}

// GetVolume returns information about a volume
func (s *BaseStorageService) GetVolume(ctx context.Context, volumeID string) (*VolumeInfo, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	volume, exists := s.volumes[volumeID]
	if !exists {
		return nil, ErrVolumeNotFound
	}

	// Return a copy to avoid race conditions
	volumeCopy := volume
	return &volumeCopy, nil
}

// ListVolumes lists all volumes
func (s *BaseStorageService) ListVolumes(ctx context.Context) ([]VolumeInfo, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	volumes := make([]VolumeInfo, 0, len(s.volumes))
	for _, volume := range s.volumes {
		volumes = append(volumes, volume)
	}

	return volumes, nil
}

// AttachVolume attaches a volume to a VM
func (s *BaseStorageService) AttachVolume(ctx context.Context, volumeID string, opts VolumeAttachOptions) error {
	return ErrNotImplemented
}

// DetachVolume detaches a volume from a VM
func (s *BaseStorageService) DetachVolume(ctx context.Context, volumeID string, opts VolumeDetachOptions) error {
	return ErrNotImplemented
}

// ResizeVolume resizes a volume
func (s *BaseStorageService) ResizeVolume(ctx context.Context, volumeID string, opts VolumeResizeOptions) error {
	return ErrNotImplemented
}

// OpenVolume opens a volume for reading/writing
func (s *BaseStorageService) OpenVolume(ctx context.Context, volumeID string) (io.ReadWriteCloser, error) {
	return nil, ErrNotImplemented
}

// GetVolumeStats returns statistics for a volume
func (s *BaseStorageService) GetVolumeStats(ctx context.Context, volumeID string) (map[string]interface{}, error) {
	return nil, ErrNotImplemented
}

// AddVolumeEventListener adds a listener for volume events
func (s *BaseStorageService) AddVolumeEventListener(listener VolumeEventListener) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.listeners = append(s.listeners, listener)
}

// RemoveVolumeEventListener removes a listener for volume events
func (s *BaseStorageService) RemoveVolumeEventListener(listener VolumeEventListener) {
	s.mu.Lock()
	defer s.mu.Unlock()

	for i, l := range s.listeners {
		if fmt.Sprintf("%p", l) == fmt.Sprintf("%p", listener) {
			s.listeners = append(s.listeners[:i], s.listeners[i+1:]...)
			return
		}
	}
}

// NotifyEvent notifies all listeners of an event
func (s *BaseStorageService) NotifyEvent(event VolumeEvent) {
	s.mu.RLock()
	listeners := make([]VolumeEventListener, len(s.listeners))
	copy(listeners, s.listeners)
	s.mu.RUnlock()

	for _, listener := range listeners {
		go listener(event)
	}
}

// IsRunning returns whether the service is running
func (s *BaseStorageService) IsRunning() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.running
}
