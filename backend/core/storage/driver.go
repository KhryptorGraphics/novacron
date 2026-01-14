package storage

import (
	"context"
	"fmt"
	"sync"
)

// StorageDriver defines the interface that all storage drivers must implement
type StorageDriver interface {
	// Initialize initializes the storage driver
	Initialize() error

	// Shutdown shuts down the storage driver
	Shutdown() error

	// CreateVolume creates a new volume
	CreateVolume(ctx context.Context, volumeID string, sizeBytes int64) error

	// DeleteVolume deletes a volume
	DeleteVolume(ctx context.Context, volumeID string) error

	// AttachVolume attaches a volume to a node
	AttachVolume(ctx context.Context, volumeID, nodeID string) error

	// DetachVolume detaches a volume from a node
	DetachVolume(ctx context.Context, volumeID, nodeID string) error

	// ReadVolume reads data from a volume
	ReadVolume(ctx context.Context, volumeID string, offset int64, size int) ([]byte, error)

	// WriteVolume writes data to a volume
	WriteVolume(ctx context.Context, volumeID string, offset int64, data []byte) error

	// GetVolumeInfo returns information about a volume
	GetVolumeInfo(ctx context.Context, volumeID string) (*VolumeInfo, error)

	// ListVolumes lists all volumes
	ListVolumes(ctx context.Context) ([]string, error)

	// GetCapabilities returns the capabilities of the driver
	GetCapabilities() DriverCapabilities

	// CreateSnapshot creates a snapshot of a volume
	CreateSnapshot(ctx context.Context, volumeID, snapshotID string) error

	// DeleteSnapshot deletes a snapshot
	DeleteSnapshot(ctx context.Context, volumeID, snapshotID string) error

	// RestoreSnapshot restores a volume from a snapshot
	RestoreSnapshot(ctx context.Context, volumeID, snapshotID string) error
}

// DriverCapabilities describes what a storage driver can do
type DriverCapabilities struct {
	// Whether the driver supports snapshots
	SupportsSnapshots bool

	// Whether the driver supports replication
	SupportsReplication bool

	// Whether the driver supports encryption
	SupportsEncryption bool

	// Whether the driver supports compression
	SupportsCompression bool

	// Whether the driver supports deduplication
	SupportsDeduplication bool

	// Maximum volume size in bytes (0 = unlimited)
	MaxVolumeSize int64

	// Minimum volume size in bytes
	MinVolumeSize int64
}

// DriverFactory is a function that creates a storage driver
type DriverFactory func(config map[string]interface{}) (StorageDriver, error)

// driverRegistry holds registered storage drivers
var driverRegistry = struct {
	sync.RWMutex
	drivers map[string]DriverFactory
}{
	drivers: make(map[string]DriverFactory),
}

// RegisterDriver registers a storage driver factory
func RegisterDriver(name string, factory DriverFactory) {
	driverRegistry.Lock()
	defer driverRegistry.Unlock()

	if factory == nil {
		panic("storage: RegisterDriver factory is nil")
	}
	if _, exists := driverRegistry.drivers[name]; exists {
		panic(fmt.Sprintf("storage: RegisterDriver called twice for driver %s", name))
	}

	driverRegistry.drivers[name] = factory
}

// CreateDriver creates a storage driver by name
func CreateDriver(name string, config map[string]interface{}) (StorageDriver, error) {
	driverRegistry.RLock()
	factory, exists := driverRegistry.drivers[name]
	driverRegistry.RUnlock()

	if !exists {
		return nil, fmt.Errorf("storage: unknown driver %q (did you forget to register it?)", name)
	}

	return factory(config)
}

// ListDrivers returns a list of registered driver names
func ListDrivers() []string {
	driverRegistry.RLock()
	defer driverRegistry.RUnlock()

	names := make([]string, 0, len(driverRegistry.drivers))
	for name := range driverRegistry.drivers {
		names = append(names, name)
	}
	return names
}

// GetDriverFactory returns the factory function for a driver
func GetDriverFactory(name string) (DriverFactory, bool) {
	driverRegistry.RLock()
	defer driverRegistry.RUnlock()

	factory, exists := driverRegistry.drivers[name]
	return factory, exists
}