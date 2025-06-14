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

	// VolumeTypeLocal is a local volume (on the host filesystem)
	VolumeTypeLocal VolumeType = "local"
	
	// VolumeTypeNFS is an NFS mounted volume
	VolumeTypeNFS VolumeType = "nfs"
	
	// VolumeTypeCeph is a Ceph volume
	VolumeTypeCeph VolumeType = "ceph"
)

// VolumeFormat defines the filesystem format
type VolumeFormat string

const (
	// VolumeFormatExt4 is the ext4 filesystem
	VolumeFormatExt4 VolumeFormat = "ext4"
	
	// VolumeFormatXFS is the XFS filesystem
	VolumeFormatXFS VolumeFormat = "xfs"
	
	// VolumeFormatRAW is a raw block device (no filesystem)
	VolumeFormatRAW VolumeFormat = "raw"
)

// VolumeStatus represents the status of a volume
type VolumeStatus string

const (
	VolumeStatusCreating  VolumeStatus = "creating"
	VolumeStatusAvailable VolumeStatus = "available"
	VolumeStatusInUse     VolumeStatus = "in_use"
	VolumeStatusDeleting  VolumeStatus = "deleting"
	VolumeStatusError     VolumeStatus = "error"
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

	// Additional fields for storage manager compatibility
	Format   VolumeFormat `json:"format,omitempty"`   // Filesystem format
	SizeMB   int          `json:"size_mb,omitempty"`  // Size in MB for backward compatibility
	Path     string       `json:"path,omitempty"`     // Path to the volume file
	Status   VolumeStatus `json:"status,omitempty"`   // Volume status for storage manager
	Labels   map[string]string `json:"labels,omitempty"` // Volume labels
	Snapshots []VolumeSnapshot `json:"snapshots,omitempty"` // Volume snapshots
	LastUpdated time.Time `json:"last_updated,omitempty"` // Last update time
	Available bool `json:"available,omitempty"` // Whether volume is available
	UsedMB int `json:"used_mb,omitempty"` // Used space in MB
}

// VolumeSnapshot represents a volume snapshot
type VolumeSnapshot struct {
	ID         string            `json:"id"`
	VolumeID   string            `json:"volume_id"`
	Name       string            `json:"name"`
	Description string           `json:"description,omitempty"`
	CreatedAt  time.Time         `json:"created_at"`
	Size       int64             `json:"size"`
	Metadata   map[string]string `json:"metadata,omitempty"`
}

// VolumeCreateOptions contains options for creating a volume
type VolumeCreateOptions struct {
	// Name of the volume
	Name string `json:"name"`

	// Type of volume
	Type VolumeType `json:"type"`

	// Size of the volume in bytes
	Size int64 `json:"size"`

	// Format of the volume (for filesystem volumes)
	Format VolumeFormat `json:"format,omitempty"`

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

	// VolumeEventSnapshoted is emitted when a volume snapshot is created
	VolumeEventSnapshoted VolumeEventType = "snapshoted"
	
	// VolumeEventError is emitted on volume errors
	VolumeEventError VolumeEventType = "error"
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
	s.mu.Lock()
	defer s.mu.Unlock()

	// Validate options
	if opts.Name == "" {
		return nil, ErrInvalidVolumeName
	}

	// Generate unique volume ID
	volumeID := generateVolumeID()

	// Create volume info
	volume := VolumeInfo{
		ID:                volumeID,
		Name:              opts.Name,
		Type:              opts.Type,
		State:             VolumeStateCreating,
		Size:              opts.Size,
		CreatedAt:         time.Now(),
		UpdatedAt:         time.Now(),
		Metadata:          opts.Metadata,
		Bootable:          opts.Bootable,
		Encrypted:         opts.Encrypted,
		ReplicationFactor: opts.ReplicationFactor,
		NodeIDs:           opts.NodeIDs,
	}

	// If no type specified, use default
	if volume.Type == "" {
		volume.Type = s.config.DefaultVolumeType
	}

	// If no replication factor specified, use default
	if volume.ReplicationFactor <= 0 {
		volume.ReplicationFactor = s.config.DefaultReplicationFactor
	}

	// Store volume in memory
	s.volumes[volumeID] = volume

	// Create the actual volume asynchronously
	go func() {
		// Simulate volume creation
		time.Sleep(2 * time.Second)

		s.mu.Lock()
		if vol, exists := s.volumes[volumeID]; exists {
			vol.State = VolumeStateAvailable
			vol.UpdatedAt = time.Now()
			s.volumes[volumeID] = vol
		}
		s.mu.Unlock()

		// Notify listeners
		s.NotifyEvent(VolumeEvent{
			Type:       VolumeEventCreated,
			VolumeID:   volumeID,
			VolumeName: opts.Name,
			Data:       volume,
			Timestamp:  time.Now(),
		})
	}()

	return &volume, nil
}

// DeleteVolume deletes a volume
func (s *BaseStorageService) DeleteVolume(ctx context.Context, volumeID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	volume, exists := s.volumes[volumeID]
	if !exists {
		return ErrVolumeNotFound
	}

	// Check if volume is in use
	if volume.AttachedToVM != "" {
		return ErrVolumeInUse
	}

	// Check if volume is already being deleted
	if volume.State == VolumeStateDeleting {
		return ErrInvalidOperation
	}

	// Update state to deleting
	volume.State = VolumeStateDeleting
	volume.UpdatedAt = time.Now()
	s.volumes[volumeID] = volume

	// Delete the volume asynchronously
	go func() {
		// Simulate volume deletion
		time.Sleep(1 * time.Second)

		s.mu.Lock()
		delete(s.volumes, volumeID)
		s.mu.Unlock()

		// Notify listeners
		s.NotifyEvent(VolumeEvent{
			Type:       VolumeEventDeleted,
			VolumeID:   volumeID,
			VolumeName: volume.Name,
			Timestamp:  time.Now(),
		})
	}()

	return nil
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
	s.mu.Lock()
	defer s.mu.Unlock()

	volume, exists := s.volumes[volumeID]
	if !exists {
		return ErrVolumeNotFound
	}

	// Check if volume is available
	if volume.State != VolumeStateAvailable {
		return ErrInvalidOperation
	}

	// Check if already attached
	if volume.AttachedToVM != "" {
		return ErrVolumeInUse
	}

	// Update state to attaching
	volume.State = VolumeStateAttaching
	volume.UpdatedAt = time.Now()
	s.volumes[volumeID] = volume

	// Attach the volume asynchronously
	go func() {
		// Simulate attachment
		time.Sleep(500 * time.Millisecond)

		s.mu.Lock()
		if vol, exists := s.volumes[volumeID]; exists {
			vol.State = VolumeStateAttached
			vol.AttachedToVM = opts.VMID
			vol.UpdatedAt = time.Now()
			s.volumes[volumeID] = vol
		}
		s.mu.Unlock()

		// Notify listeners
		s.NotifyEvent(VolumeEvent{
			Type:       VolumeEventAttached,
			VolumeID:   volumeID,
			VolumeName: volume.Name,
			Data:       opts,
			Timestamp:  time.Now(),
		})
	}()

	return nil
}

// DetachVolume detaches a volume from a VM
func (s *BaseStorageService) DetachVolume(ctx context.Context, volumeID string, opts VolumeDetachOptions) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	volume, exists := s.volumes[volumeID]
	if !exists {
		return ErrVolumeNotFound
	}

	// Check if volume is attached
	if volume.State != VolumeStateAttached {
		return ErrInvalidOperation
	}

	// Update state to detaching
	volume.State = VolumeStateDetaching
	volume.UpdatedAt = time.Now()
	s.volumes[volumeID] = volume

	// Detach the volume asynchronously
	go func() {
		// Simulate detachment
		time.Sleep(500 * time.Millisecond)

		s.mu.Lock()
		if vol, exists := s.volumes[volumeID]; exists {
			vol.State = VolumeStateAvailable
			vol.AttachedToVM = ""
			vol.UpdatedAt = time.Now()
			s.volumes[volumeID] = vol
		}
		s.mu.Unlock()

		// Notify listeners
		s.NotifyEvent(VolumeEvent{
			Type:       VolumeEventDetached,
			VolumeID:   volumeID,
			VolumeName: volume.Name,
			Data:       opts,
			Timestamp:  time.Now(),
		})
	}()

	return nil
}

// ResizeVolume resizes a volume
func (s *BaseStorageService) ResizeVolume(ctx context.Context, volumeID string, opts VolumeResizeOptions) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	volume, exists := s.volumes[volumeID]
	if !exists {
		return ErrVolumeNotFound
	}

	// Check if volume is available or attached (can resize attached volumes in some cases)
	if volume.State != VolumeStateAvailable && volume.State != VolumeStateAttached {
		return ErrInvalidOperation
	}

	// Check if new size is larger
	if opts.NewSize <= volume.Size {
		return fmt.Errorf("new size must be larger than current size")
	}

	// Update volume size
	oldSize := volume.Size
	volume.Size = opts.NewSize
	volume.UpdatedAt = time.Now()
	s.volumes[volumeID] = volume

	// Notify listeners
	s.NotifyEvent(VolumeEvent{
		Type:       VolumeEventResized,
		VolumeID:   volumeID,
		VolumeName: volume.Name,
		Data: map[string]interface{}{
			"old_size": oldSize,
			"new_size": opts.NewSize,
		},
		Timestamp: time.Now(),
	})

	return nil
}

// OpenVolume opens a volume for reading/writing
func (s *BaseStorageService) OpenVolume(ctx context.Context, volumeID string) (io.ReadWriteCloser, error) {
	s.mu.RLock()
	volume, exists := s.volumes[volumeID]
	s.mu.RUnlock()

	if !exists {
		return nil, ErrVolumeNotFound
	}

	// Check if volume is available or attached
	if volume.State != VolumeStateAvailable && volume.State != VolumeStateAttached {
		return nil, ErrInvalidOperation
	}

	// In a real implementation, this would open the actual storage backend
	// For now, return a simple in-memory buffer
	return &volumeHandle{
		volumeID: volumeID,
		service:  s,
		buffer:   make([]byte, volume.Size),
	}, nil
}

// GetVolumeStats returns statistics for a volume
func (s *BaseStorageService) GetVolumeStats(ctx context.Context, volumeID string) (map[string]interface{}, error) {
	s.mu.RLock()
	volume, exists := s.volumes[volumeID]
	s.mu.RUnlock()

	if !exists {
		return nil, ErrVolumeNotFound
	}

	// Return volume statistics
	stats := map[string]interface{}{
		"id":                 volume.ID,
		"name":               volume.Name,
		"type":               volume.Type,
		"state":              volume.State,
		"size_bytes":         volume.Size,
		"attached_to":        volume.AttachedToVM,
		"created_at":         volume.CreatedAt,
		"updated_at":         volume.UpdatedAt,
		"encrypted":          volume.Encrypted,
		"replication_factor": volume.ReplicationFactor,
		"node_count":         len(volume.NodeIDs),
		"iops_read":          volume.IOPSRead,
		"iops_write":         volume.IOPSWrite,
		"mbps_read":          volume.MBpsRead,
		"mbps_write":         volume.MBpsWrite,
	}

	return stats, nil
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

// volumeHandle implements io.ReadWriteCloser for volume access
type volumeHandle struct {
	volumeID string
	service  *BaseStorageService
	buffer   []byte
	offset   int64
}

func (v *volumeHandle) Read(p []byte) (n int, err error) {
	if v.offset >= int64(len(v.buffer)) {
		return 0, io.EOF
	}
	n = copy(p, v.buffer[v.offset:])
	v.offset += int64(n)
	return n, nil
}

func (v *volumeHandle) Write(p []byte) (n int, err error) {
	if v.offset >= int64(len(v.buffer)) {
		return 0, io.ErrShortWrite
	}
	n = copy(v.buffer[v.offset:], p)
	v.offset += int64(n)
	return n, nil
}

func (v *volumeHandle) Close() error {
	// In a real implementation, this would flush data to disk
	return nil
}

// generateVolumeID generates a unique volume ID
func generateVolumeID() string {
	// In a real implementation, this would use a proper UUID library
	return fmt.Sprintf("vol-%d-%d", time.Now().Unix(), time.Now().Nanosecond())
}
