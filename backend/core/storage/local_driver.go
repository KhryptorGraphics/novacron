package storage

import (
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// LocalStorageDriver implements the StorageDriver interface for local storage
type LocalStorageDriver struct {
	// Configuration
	rootDir string

	// Initialized state
	initialized bool

	// Mutex for thread safety
	mu sync.RWMutex

	// Volume cache
	volumeCache map[string]*VolumeInfo
}

// NewLocalStorageDriver creates a new local storage driver
func NewLocalStorageDriver() *LocalStorageDriver {
	return &LocalStorageDriver{
		rootDir:     "/var/lib/novacron/volumes",
		initialized: false,
		volumeCache: make(map[string]*VolumeInfo),
	}
}

// Initialize initializes the local storage driver
func (d *LocalStorageDriver) Initialize() error {
	d.mu.Lock()
	defer d.mu.Unlock()

	if d.initialized {
		return fmt.Errorf("driver already initialized")
	}

	// Create root directory if it doesn't exist
	if err := os.MkdirAll(d.rootDir, 0755); err != nil {
		return fmt.Errorf("failed to create root directory: %v", err)
	}

	// Load existing volumes
	if err := d.loadExistingVolumes(); err != nil {
		return fmt.Errorf("failed to load existing volumes: %v", err)
	}

	d.initialized = true
	return nil
}

// Shutdown shuts down the local storage driver
func (d *LocalStorageDriver) Shutdown() error {
	d.mu.Lock()
	defer d.mu.Unlock()

	d.initialized = false
	d.volumeCache = make(map[string]*VolumeInfo)
	return nil
}

// CreateVolume creates a new volume
func (d *LocalStorageDriver) CreateVolume(ctx context.Context, volumeID string, sizeBytes int64) error {
	d.mu.Lock()
	defer d.mu.Unlock()

	if !d.initialized {
		return fmt.Errorf("driver not initialized")
	}

	volumePath := filepath.Join(d.rootDir, volumeID+".img")

	// Check if volume already exists
	if _, err := os.Stat(volumePath); err == nil {
		return fmt.Errorf("volume %s already exists", volumeID)
	}

	// Create the volume file using qemu-img for better compatibility
	sizeMB := sizeBytes / (1024 * 1024)
	if sizeMB < 1 {
		sizeMB = 1
	}

	cmd := exec.CommandContext(ctx, "qemu-img", "create", "-f", "qcow2", volumePath, fmt.Sprintf("%dM", sizeMB))
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed to create volume file: %v (output: %s)", err, string(output))
	}

	// Create volume info and cache it
	volumeInfo := &VolumeInfo{
		ID:                volumeID,
		Name:              volumeID,
		Type:              VolumeTypeLocal,
		State:             VolumeStateAvailable,
		Size:              sizeBytes,
		Path:              volumePath,
		CreatedAt:         time.Now(),
		UpdatedAt:         time.Now(),
		Metadata:          make(map[string]string),
		Bootable:          false,
		Encrypted:         false,
		ReplicationFactor: 1, // Local storage has no replication
	}

	d.volumeCache[volumeID] = volumeInfo
	return nil
}

// DeleteVolume deletes a volume
func (d *LocalStorageDriver) DeleteVolume(ctx context.Context, volumeID string) error {
	d.mu.Lock()
	defer d.mu.Unlock()

	if !d.initialized {
		return fmt.Errorf("driver not initialized")
	}

	volumePath := filepath.Join(d.rootDir, volumeID+".img")

	// Remove the volume file
	if err := os.Remove(volumePath); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("failed to delete volume file: %v", err)
	}

	// Remove from cache
	delete(d.volumeCache, volumeID)
	return nil
}

// AttachVolume attaches a volume to a node (for local storage, this is mostly a no-op)
func (d *LocalStorageDriver) AttachVolume(ctx context.Context, volumeID, nodeID string) error {
	d.mu.Lock()
	defer d.mu.Unlock()

	if !d.initialized {
		return fmt.Errorf("driver not initialized")
	}

	// For local storage, we just update the cache
	if vol, exists := d.volumeCache[volumeID]; exists {
		vol.State = VolumeStateAttached
		vol.AttachedToVM = nodeID
		vol.UpdatedAt = time.Now()
	}

	return nil
}

// DetachVolume detaches a volume from a node
func (d *LocalStorageDriver) DetachVolume(ctx context.Context, volumeID, nodeID string) error {
	d.mu.Lock()
	defer d.mu.Unlock()

	if !d.initialized {
		return fmt.Errorf("driver not initialized")
	}

	// For local storage, we just update the cache
	if vol, exists := d.volumeCache[volumeID]; exists {
		vol.State = VolumeStateAvailable
		vol.AttachedToVM = ""
		vol.UpdatedAt = time.Now()
	}

	return nil
}

// ReadVolume reads data from a volume
func (d *LocalStorageDriver) ReadVolume(ctx context.Context, volumeID string, offset int64, size int) ([]byte, error) {
	d.mu.RLock()
	defer d.mu.RUnlock()

	if !d.initialized {
		return nil, fmt.Errorf("driver not initialized")
	}

	volumePath := filepath.Join(d.rootDir, volumeID+".img")

	file, err := os.Open(volumePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open volume file: %v", err)
	}
	defer file.Close()

	// Seek to offset
	if _, err := file.Seek(offset, io.SeekStart); err != nil {
		return nil, fmt.Errorf("failed to seek to offset: %v", err)
	}

	// Read data
	data := make([]byte, size)
	n, err := file.Read(data)
	if err != nil && err != io.EOF {
		return nil, fmt.Errorf("failed to read volume data: %v", err)
	}

	return data[:n], nil
}

// WriteVolume writes data to a volume
func (d *LocalStorageDriver) WriteVolume(ctx context.Context, volumeID string, offset int64, data []byte) error {
	d.mu.Lock()
	defer d.mu.Unlock()

	if !d.initialized {
		return fmt.Errorf("driver not initialized")
	}

	volumePath := filepath.Join(d.rootDir, volumeID+".img")

	file, err := os.OpenFile(volumePath, os.O_WRONLY, 0644)
	if err != nil {
		return fmt.Errorf("failed to open volume file for writing: %v", err)
	}
	defer file.Close()

	// Seek to offset
	if _, err := file.Seek(offset, io.SeekStart); err != nil {
		return fmt.Errorf("failed to seek to offset: %v", err)
	}

	// Write data
	if _, err := file.Write(data); err != nil {
		return fmt.Errorf("failed to write volume data: %v", err)
	}

	return nil
}

// GetVolumeInfo returns information about a volume
func (d *LocalStorageDriver) GetVolumeInfo(ctx context.Context, volumeID string) (*VolumeInfo, error) {
	d.mu.RLock()
	defer d.mu.RUnlock()

	if !d.initialized {
		return nil, fmt.Errorf("driver not initialized")
	}

	volume, exists := d.volumeCache[volumeID]
	if !exists {
		return nil, fmt.Errorf("volume %s not found", volumeID)
	}

	// Return a copy
	volumeCopy := *volume
	return &volumeCopy, nil
}

// ListVolumes lists all volumes
func (d *LocalStorageDriver) ListVolumes(ctx context.Context) ([]string, error) {
	d.mu.RLock()
	defer d.mu.RUnlock()

	if !d.initialized {
		return nil, fmt.Errorf("driver not initialized")
	}

	volumes := make([]string, 0, len(d.volumeCache))
	for volumeID := range d.volumeCache {
		volumes = append(volumes, volumeID)
	}

	return volumes, nil
}

// GetCapabilities returns the capabilities of the local storage driver
func (d *LocalStorageDriver) GetCapabilities() DriverCapabilities {
	return DriverCapabilities{
		SupportsSnapshots:     true,  // qemu-img supports snapshots
		SupportsReplication:   false, // Local storage doesn't support replication
		SupportsEncryption:    false, // Not implemented yet
		SupportsCompression:   true,  // qcow2 supports compression
		SupportsDeduplication: false, // Not supported
		MaxVolumeSize:         0,     // Unlimited (within filesystem limits)
		MinVolumeSize:         1024,  // 1KB minimum
	}
}

// CreateSnapshot creates a snapshot of a volume
func (d *LocalStorageDriver) CreateSnapshot(ctx context.Context, volumeID, snapshotID string) error {
	d.mu.Lock()
	defer d.mu.Unlock()

	if !d.initialized {
		return fmt.Errorf("driver not initialized")
	}

	volumePath := filepath.Join(d.rootDir, volumeID+".img")

	// Create snapshot using qemu-img
	cmd := exec.CommandContext(ctx, "qemu-img", "snapshot", "-c", snapshotID, volumePath)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed to create snapshot: %v (output: %s)", err, string(output))
	}

	return nil
}

// DeleteSnapshot deletes a snapshot
func (d *LocalStorageDriver) DeleteSnapshot(ctx context.Context, volumeID, snapshotID string) error {
	d.mu.Lock()
	defer d.mu.Unlock()

	if !d.initialized {
		return fmt.Errorf("driver not initialized")
	}

	volumePath := filepath.Join(d.rootDir, volumeID+".img")

	// Delete snapshot using qemu-img
	cmd := exec.CommandContext(ctx, "qemu-img", "snapshot", "-d", snapshotID, volumePath)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed to delete snapshot: %v (output: %s)", err, string(output))
	}

	return nil
}

// RestoreSnapshot restores a volume from a snapshot
func (d *LocalStorageDriver) RestoreSnapshot(ctx context.Context, volumeID, snapshotID string) error {
	d.mu.Lock()
	defer d.mu.Unlock()

	if !d.initialized {
		return fmt.Errorf("driver not initialized")
	}

	volumePath := filepath.Join(d.rootDir, volumeID+".img")

	// Restore snapshot using qemu-img
	cmd := exec.CommandContext(ctx, "qemu-img", "snapshot", "-a", snapshotID, volumePath)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed to restore snapshot: %v (output: %s)", err, string(output))
	}

	return nil
}

// loadExistingVolumes loads existing volume files from disk
func (d *LocalStorageDriver) loadExistingVolumes() error {
	// Scan the root directory for volume files
	files, err := os.ReadDir(d.rootDir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil // Directory doesn't exist yet, that's OK
		}
		return fmt.Errorf("failed to read volume directory: %v", err)
	}

	for _, file := range files {
		if file.IsDir() {
			continue
		}

		// Check if this is a volume file (ends with .img)
		if !strings.HasSuffix(file.Name(), ".img") {
			continue
		}

		// Extract volume ID (filename without .img extension)
		volumeID := strings.TrimSuffix(file.Name(), ".img")
		volumePath := filepath.Join(d.rootDir, file.Name())

		// Get file info
		info, err := os.Stat(volumePath)
		if err != nil {
			continue
		}

		// Create volume info
		volumeInfo := &VolumeInfo{
			ID:                volumeID,
			Name:              volumeID,
			Type:              VolumeTypeLocal,
			State:             VolumeStateAvailable,
			Size:              info.Size(),
			Path:              volumePath,
			CreatedAt:         info.ModTime(), // Use file mod time as creation time
			UpdatedAt:         info.ModTime(),
			Metadata:          make(map[string]string),
			Bootable:          false,
			Encrypted:         false,
			ReplicationFactor: 1,
		}

		d.volumeCache[volumeID] = volumeInfo
	}

	return nil
}