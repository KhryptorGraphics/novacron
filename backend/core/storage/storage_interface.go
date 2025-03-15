package storage

import (
	"context"
	"io"
)

// StorageDriver defines the interface that all storage drivers must implement
// to be compatible with the NovaCron storage subsystem.
type StorageDriver interface {
	// Name returns the unique name of the driver
	Name() string

	// Initialize initializes the driver with its configuration
	Initialize() error

	// Shutdown cleans up resources and shuts down the driver
	Shutdown() error

	// Volume Management

	// CreateVolume creates a new volume with the given name and size in GB
	CreateVolume(ctx context.Context, name string, sizeGB int) error

	// DeleteVolume removes the specified volume
	DeleteVolume(ctx context.Context, name string) error

	// ResizeVolume changes the size of an existing volume
	ResizeVolume(ctx context.Context, name string, newSizeGB int) error

	// GetVolumeInfo returns information about a volume
	GetVolumeInfo(ctx context.Context, name string) (map[string]interface{}, error)

	// ListVolumes returns a list of all volumes
	ListVolumes(ctx context.Context) ([]string, error)

	// CloneVolume creates a copy of a volume
	CloneVolume(ctx context.Context, sourceName, destName string) error

	// Snapshot Management

	// CreateSnapshot creates a point-in-time snapshot of a volume
	CreateSnapshot(ctx context.Context, volumeName, snapshotName string) error

	// DeleteSnapshot removes a snapshot
	DeleteSnapshot(ctx context.Context, volumeName, snapshotName string) error

	// ListSnapshots lists all snapshots for a volume
	ListSnapshots(ctx context.Context, volumeName string) ([]string, error)

	// Data Operations

	// WriteVolumeData writes data to a volume at the specified offset
	// Returns the number of bytes written
	WriteVolumeData(ctx context.Context, volumeName string, offset int64, data io.Reader) (int64, error)

	// ReadVolumeData reads data from a volume at the specified offset for the specified length
	// If length is 0, reads until the end of the volume
	ReadVolumeData(ctx context.Context, volumeName string, offset int64, length int64) (io.ReadCloser, error)

	// Metrics

	// GetMetrics returns performance and usage metrics for the storage system
	GetMetrics(ctx context.Context) (map[string]interface{}, error)
}

// StorageDriverFactory defines a function that creates a new storage driver instance
type StorageDriverFactory func(config map[string]interface{}) (StorageDriver, error)

// RegisteredDrivers holds all registered storage driver factories
var RegisteredDrivers = make(map[string]StorageDriverFactory)

// RegisterDriver registers a storage driver factory with the system
func RegisterDriver(name string, factory StorageDriverFactory) {
	RegisteredDrivers[name] = factory
}

// GetDriver returns a storage driver factory by name
func GetDriver(name string) (StorageDriverFactory, bool) {
	factory, exists := RegisteredDrivers[name]
	return factory, exists
}

// ListDrivers returns a list of all registered driver names
func ListDrivers() []string {
	drivers := make([]string, 0, len(RegisteredDrivers))
	for name := range RegisteredDrivers {
		drivers = append(drivers, name)
	}
	return drivers
}
