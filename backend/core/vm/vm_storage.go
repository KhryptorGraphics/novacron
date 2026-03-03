package vm

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
)

// StorageType represents the type of storage
type StorageType string

const (
	// StorageTypeLocal represents local storage
	StorageTypeLocal StorageType = "local"

	// StorageTypeNFS represents NFS storage
	StorageTypeNFS StorageType = "nfs"

	// StorageTypeCeph represents Ceph storage
	StorageTypeCeph StorageType = "ceph"

	// StorageTypeISCSI represents iSCSI storage
	StorageTypeISCSI StorageType = "iscsi"
)

// StorageFormat represents the format of storage
type StorageFormat string

const (
	// StorageFormatRaw represents raw storage format
	StorageFormatRaw StorageFormat = "raw"

	// StorageFormatQCOW2 represents QCOW2 storage format
	StorageFormatQCOW2 StorageFormat = "qcow2"

	// StorageFormatVMDK represents VMDK storage format
	StorageFormatVMDK StorageFormat = "vmdk"

	// StorageFormatVHD represents VHD storage format
	StorageFormatVHD StorageFormat = "vhd"
)

// StoragePool represents a storage pool
type StoragePool struct {
	ID         string            `json:"id"`
	Name       string            `json:"name"`
	Type       StorageType       `json:"type"`
	Path       string            `json:"path"`
	TotalSpace int64             `json:"total_space"`
	UsedSpace  int64             `json:"used_space"`
	CreatedAt  time.Time         `json:"created_at"`
	UpdatedAt  time.Time         `json:"updated_at"`
	Tags       []string          `json:"tags,omitempty"`
	Metadata   map[string]string `json:"metadata,omitempty"`
}

// StorageVolume represents a storage volume
type StorageVolume struct {
	ID         string            `json:"id"`
	Name       string            `json:"name"`
	PoolID     string            `json:"pool_id"`
	Format     StorageFormat     `json:"format"`
	Capacity   int64             `json:"capacity"`
	Allocation int64             `json:"allocation"`
	Path       string            `json:"path"`
	CreatedAt  time.Time         `json:"created_at"`
	UpdatedAt  time.Time         `json:"updated_at"`
	Tags       []string          `json:"tags,omitempty"`
	Metadata   map[string]string `json:"metadata,omitempty"`
}

// VMStorageManager manages VM storage
type VMStorageManager struct {
	pools        map[string]*StoragePool
	poolsMutex   sync.RWMutex
	volumes      map[string]*StorageVolume
	volumesMutex sync.RWMutex
	vmManager    *VMManager
}

// NewVMStorageManager creates a new VM storage manager
func NewVMStorageManager(vmManager *VMManager) *VMStorageManager {
	return &VMStorageManager{
		pools:     make(map[string]*StoragePool),
		volumes:   make(map[string]*StorageVolume),
		vmManager: vmManager,
	}
}

// CreateStoragePool creates a new storage pool
func (m *VMStorageManager) CreateStoragePool(ctx context.Context, name string, storageType StorageType, path string, tags []string, metadata map[string]string) (*StoragePool, error) {
	// Validate storage type
	switch storageType {
	case StorageTypeLocal, StorageTypeNFS, StorageTypeCeph, StorageTypeISCSI:
		// Valid storage type
	default:
		return nil, fmt.Errorf("invalid storage type: %s", storageType)
	}

	// Validate path
	if path == "" {
		return nil, fmt.Errorf("path is required")
	}

	// For local storage, check if path exists
	if storageType == StorageTypeLocal {
		if _, err := os.Stat(path); os.IsNotExist(err) {
			// Create directory
			if err := os.MkdirAll(path, 0755); err != nil {
				return nil, fmt.Errorf("failed to create storage pool directory: %w", err)
			}
		}
	}

	// Generate pool ID
	poolID := fmt.Sprintf("pool-%s", strings.ReplaceAll(name, " ", "-"))

	// Check if pool already exists
	m.poolsMutex.RLock()
	if _, exists := m.pools[poolID]; exists {
		m.poolsMutex.RUnlock()
		return nil, fmt.Errorf("storage pool with ID %s already exists", poolID)
	}
	m.poolsMutex.RUnlock()

	// Get storage space information
	var totalSpace, usedSpace int64
	if storageType == StorageTypeLocal {
		// Get disk usage information
		cmd := exec.CommandContext(ctx, "df", "--output=size,used", path)
		output, err := cmd.Output()
		if err == nil {
			lines := strings.Split(string(output), "\n")
			if len(lines) >= 2 {
				fields := strings.Fields(lines[1])
				if len(fields) >= 2 {
					totalSpace, _ = strconv.ParseInt(fields[0], 10, 64)
					usedSpace, _ = strconv.ParseInt(fields[1], 10, 64)
					// Convert from KB to bytes
					totalSpace *= 1024
					usedSpace *= 1024
				}
			}
		}
	}

	// Create pool
	pool := &StoragePool{
		ID:         poolID,
		Name:       name,
		Type:       storageType,
		Path:       path,
		TotalSpace: totalSpace,
		UsedSpace:  usedSpace,
		CreatedAt:  time.Now(),
		UpdatedAt:  time.Now(),
		Tags:       tags,
		Metadata:   metadata,
	}

	// Create pool in the system
	if err := m.createPoolInSystem(ctx, pool); err != nil {
		return nil, fmt.Errorf("failed to create storage pool in system: %w", err)
	}

	// Store pool
	m.poolsMutex.Lock()
	m.pools[poolID] = pool
	m.poolsMutex.Unlock()

	log.Printf("Created storage pool %s (%s)", pool.Name, pool.ID)

	return pool, nil
}

// GetStoragePool returns a storage pool by ID
func (m *VMStorageManager) GetStoragePool(poolID string) (*StoragePool, error) {
	m.poolsMutex.RLock()
	defer m.poolsMutex.RUnlock()

	pool, exists := m.pools[poolID]
	if !exists {
		return nil, fmt.Errorf("storage pool %s not found", poolID)
	}

	return pool, nil
}

// ListStoragePools returns all storage pools
func (m *VMStorageManager) ListStoragePools() []*StoragePool {
	m.poolsMutex.RLock()
	defer m.poolsMutex.RUnlock()

	pools := make([]*StoragePool, 0, len(m.pools))
	for _, pool := range m.pools {
		pools = append(pools, pool)
	}

	return pools
}

// DeleteStoragePool deletes a storage pool
func (m *VMStorageManager) DeleteStoragePool(ctx context.Context, poolID string) error {
	// Get the pool
	m.poolsMutex.RLock()
	pool, exists := m.pools[poolID]
	if !exists {
		m.poolsMutex.RUnlock()
		return fmt.Errorf("storage pool %s not found", poolID)
	}
	m.poolsMutex.RUnlock()

	// Check if pool has volumes
	m.volumesMutex.RLock()
	for _, volume := range m.volumes {
		if volume.PoolID == poolID {
			m.volumesMutex.RUnlock()
			return fmt.Errorf("storage pool %s has volumes", poolID)
		}
	}
	m.volumesMutex.RUnlock()

	// Delete pool from the system
	if err := m.deletePoolFromSystem(ctx, pool); err != nil {
		return fmt.Errorf("failed to delete storage pool from system: %w", err)
	}

	// Remove pool
	m.poolsMutex.Lock()
	delete(m.pools, poolID)
	m.poolsMutex.Unlock()

	log.Printf("Deleted storage pool %s (%s)", pool.Name, pool.ID)

	return nil
}

// CreateStorageVolume creates a new storage volume
func (m *VMStorageManager) CreateStorageVolume(ctx context.Context, name string, poolID string, format StorageFormat, capacity int64, tags []string, metadata map[string]string) (*StorageVolume, error) {
	// Get the pool
	m.poolsMutex.RLock()
	pool, exists := m.pools[poolID]
	if !exists {
		m.poolsMutex.RUnlock()
		return nil, fmt.Errorf("storage pool %s not found", poolID)
	}
	m.poolsMutex.RUnlock()

	// Validate format
	switch format {
	case StorageFormatRaw, StorageFormatQCOW2, StorageFormatVMDK, StorageFormatVHD:
		// Valid format
	default:
		return nil, fmt.Errorf("invalid storage format: %s", format)
	}

	// Validate capacity
	if capacity <= 0 {
		return nil, fmt.Errorf("capacity must be greater than 0")
	}

	// Check if there's enough space in the pool
	if pool.TotalSpace > 0 && pool.UsedSpace+capacity > pool.TotalSpace {
		return nil, fmt.Errorf("not enough space in storage pool %s", poolID)
	}

	// Generate volume ID
	volumeID := uuid.New().String()

	// Generate volume path
	var volumePath string
	switch pool.Type {
	case StorageTypeLocal:
		volumePath = filepath.Join(pool.Path, volumeID+"."+string(format))
	case StorageTypeNFS:
		volumePath = filepath.Join(pool.Path, volumeID+"."+string(format))
	case StorageTypeCeph:
		volumePath = fmt.Sprintf("rbd:%s/%s", pool.Name, volumeID)
	case StorageTypeISCSI:
		volumePath = fmt.Sprintf("iscsi:%s/%s", pool.Name, volumeID)
	}

	// Create volume
	volume := &StorageVolume{
		ID:         volumeID,
		Name:       name,
		PoolID:     poolID,
		Format:     format,
		Capacity:   capacity,
		Allocation: 0,
		Path:       volumePath,
		CreatedAt:  time.Now(),
		UpdatedAt:  time.Now(),
		Tags:       tags,
		Metadata:   metadata,
	}

	// Create volume in the system
	if err := m.createVolumeInSystem(ctx, pool, volume); err != nil {
		return nil, fmt.Errorf("failed to create storage volume in system: %w", err)
	}

	// Store volume
	m.volumesMutex.Lock()
	m.volumes[volumeID] = volume
	m.volumesMutex.Unlock()

	// Update pool used space
	m.poolsMutex.Lock()
	pool.UsedSpace += capacity
	pool.UpdatedAt = time.Now()
	m.poolsMutex.Unlock()

	log.Printf("Created storage volume %s (%s) in pool %s", volume.Name, volume.ID, pool.Name)

	return volume, nil
}

// GetStorageVolume returns a storage volume by ID
func (m *VMStorageManager) GetStorageVolume(volumeID string) (*StorageVolume, error) {
	m.volumesMutex.RLock()
	defer m.volumesMutex.RUnlock()

	volume, exists := m.volumes[volumeID]
	if !exists {
		return nil, fmt.Errorf("storage volume %s not found", volumeID)
	}

	return volume, nil
}

// ListStorageVolumes returns all storage volumes
func (m *VMStorageManager) ListStorageVolumes() []*StorageVolume {
	m.volumesMutex.RLock()
	defer m.volumesMutex.RUnlock()

	volumes := make([]*StorageVolume, 0, len(m.volumes))
	for _, volume := range m.volumes {
		volumes = append(volumes, volume)
	}

	return volumes
}

// ListStorageVolumesInPool returns all storage volumes in a pool
func (m *VMStorageManager) ListStorageVolumesInPool(poolID string) []*StorageVolume {
	m.volumesMutex.RLock()
	defer m.volumesMutex.RUnlock()

	volumes := make([]*StorageVolume, 0)
	for _, volume := range m.volumes {
		if volume.PoolID == poolID {
			volumes = append(volumes, volume)
		}
	}

	return volumes
}

// DeleteStorageVolume deletes a storage volume
func (m *VMStorageManager) DeleteStorageVolume(ctx context.Context, volumeID string) error {
	// Get the volume
	m.volumesMutex.RLock()
	volume, exists := m.volumes[volumeID]
	if !exists {
		m.volumesMutex.RUnlock()
		return fmt.Errorf("storage volume %s not found", volumeID)
	}
	m.volumesMutex.RUnlock()

	// Get the pool
	m.poolsMutex.RLock()
	pool, exists := m.pools[volume.PoolID]
	if !exists {
		m.poolsMutex.RUnlock()
		return fmt.Errorf("storage pool %s not found", volume.PoolID)
	}
	m.poolsMutex.RUnlock()

	// Delete volume from the system
	if err := m.deleteVolumeFromSystem(ctx, pool, volume); err != nil {
		return fmt.Errorf("failed to delete storage volume from system: %w", err)
	}

	// Remove volume
	m.volumesMutex.Lock()
	delete(m.volumes, volumeID)
	m.volumesMutex.Unlock()

	// Update pool used space
	m.poolsMutex.Lock()
	pool.UsedSpace -= volume.Capacity
	if pool.UsedSpace < 0 {
		pool.UsedSpace = 0
	}
	pool.UpdatedAt = time.Now()
	m.poolsMutex.Unlock()

	log.Printf("Deleted storage volume %s (%s) from pool %s", volume.Name, volume.ID, pool.Name)

	return nil
}

// ResizeStorageVolume resizes a storage volume
func (m *VMStorageManager) ResizeStorageVolume(ctx context.Context, volumeID string, newCapacity int64) (*StorageVolume, error) {
	// Get the volume
	m.volumesMutex.RLock()
	volume, exists := m.volumes[volumeID]
	if !exists {
		m.volumesMutex.RUnlock()
		return nil, fmt.Errorf("storage volume %s not found", volumeID)
	}
	m.volumesMutex.RUnlock()

	// Get the pool
	m.poolsMutex.RLock()
	pool, exists := m.pools[volume.PoolID]
	if !exists {
		m.poolsMutex.RUnlock()
		return nil, fmt.Errorf("storage pool %s not found", volume.PoolID)
	}
	m.poolsMutex.RUnlock()

	// Validate new capacity
	if newCapacity <= 0 {
		return nil, fmt.Errorf("new capacity must be greater than 0")
	}

	// Check if new capacity is smaller than current allocation
	if newCapacity < volume.Allocation {
		return nil, fmt.Errorf("new capacity cannot be smaller than current allocation")
	}

	// Check if there's enough space in the pool for the increase
	capacityDiff := newCapacity - volume.Capacity
	if capacityDiff > 0 {
		m.poolsMutex.RLock()
		if pool.TotalSpace > 0 && pool.UsedSpace+capacityDiff > pool.TotalSpace {
			m.poolsMutex.RUnlock()
			return nil, fmt.Errorf("not enough space in storage pool %s", volume.PoolID)
		}
		m.poolsMutex.RUnlock()
	}

	// Resize volume in the system
	if err := m.resizeVolumeInSystem(ctx, pool, volume, newCapacity); err != nil {
		return nil, fmt.Errorf("failed to resize storage volume in system: %w", err)
	}

	// Update volume
	m.volumesMutex.Lock()
	oldCapacity := volume.Capacity
	volume.Capacity = newCapacity
	volume.UpdatedAt = time.Now()
	m.volumesMutex.Unlock()

	// Update pool used space
	m.poolsMutex.Lock()
	pool.UsedSpace += (newCapacity - oldCapacity)
	pool.UpdatedAt = time.Now()
	m.poolsMutex.Unlock()

	log.Printf("Resized storage volume %s (%s) from %d to %d bytes", volume.Name, volume.ID, oldCapacity, newCapacity)

	return volume, nil
}

// CloneStorageVolume clones a storage volume
func (m *VMStorageManager) CloneStorageVolume(ctx context.Context, sourceVolumeID, name string, tags []string, metadata map[string]string) (*StorageVolume, error) {
	// Get the source volume
	m.volumesMutex.RLock()
	sourceVolume, exists := m.volumes[sourceVolumeID]
	if !exists {
		m.volumesMutex.RUnlock()
		return nil, fmt.Errorf("source storage volume %s not found", sourceVolumeID)
	}
	m.volumesMutex.RUnlock()

	// Get the pool
	m.poolsMutex.RLock()
	pool, exists := m.pools[sourceVolume.PoolID]
	if !exists {
		m.poolsMutex.RUnlock()
		return nil, fmt.Errorf("storage pool %s not found", sourceVolume.PoolID)
	}
	m.poolsMutex.RUnlock()

	// Check if there's enough space in the pool
	if pool.TotalSpace > 0 && pool.UsedSpace+sourceVolume.Capacity > pool.TotalSpace {
		return nil, fmt.Errorf("not enough space in storage pool %s", sourceVolume.PoolID)
	}

	// Generate volume ID
	volumeID := uuid.New().String()

	// Generate volume path
	var volumePath string
	switch pool.Type {
	case StorageTypeLocal:
		volumePath = filepath.Join(pool.Path, volumeID+"."+string(sourceVolume.Format))
	case StorageTypeNFS:
		volumePath = filepath.Join(pool.Path, volumeID+"."+string(sourceVolume.Format))
	case StorageTypeCeph:
		volumePath = fmt.Sprintf("rbd:%s/%s", pool.Name, volumeID)
	case StorageTypeISCSI:
		volumePath = fmt.Sprintf("iscsi:%s/%s", pool.Name, volumeID)
	}

	// Create volume
	volume := &StorageVolume{
		ID:         volumeID,
		Name:       name,
		PoolID:     sourceVolume.PoolID,
		Format:     sourceVolume.Format,
		Capacity:   sourceVolume.Capacity,
		Allocation: sourceVolume.Allocation,
		Path:       volumePath,
		CreatedAt:  time.Now(),
		UpdatedAt:  time.Now(),
		Tags:       tags,
		Metadata:   metadata,
	}

	// Clone volume in the system
	if err := m.cloneVolumeInSystem(ctx, pool, sourceVolume, volume); err != nil {
		return nil, fmt.Errorf("failed to clone storage volume in system: %w", err)
	}

	// Store volume
	m.volumesMutex.Lock()
	m.volumes[volumeID] = volume
	m.volumesMutex.Unlock()

	// Update pool used space
	m.poolsMutex.Lock()
	pool.UsedSpace += volume.Capacity
	pool.UpdatedAt = time.Now()
	m.poolsMutex.Unlock()

	log.Printf("Cloned storage volume %s (%s) to %s (%s)", sourceVolume.Name, sourceVolume.ID, volume.Name, volume.ID)

	return volume, nil
}

// createPoolInSystem creates a storage pool in the system
func (m *VMStorageManager) createPoolInSystem(ctx context.Context, pool *StoragePool) error {
	// In a real implementation, this would create the storage pool in the system
	// For example, creating a directory, setting up NFS, etc.

	// For simplicity, we'll just log the operation
	log.Printf("Creating storage pool %s (%s) in system", pool.Name, pool.ID)

	// For local storage, create the directory if it doesn't exist
	if pool.Type == StorageTypeLocal {
		if _, err := os.Stat(pool.Path); os.IsNotExist(err) {
			if err := os.MkdirAll(pool.Path, 0755); err != nil {
				return fmt.Errorf("failed to create storage pool directory: %w", err)
			}
		}
	}

	return nil
}

// deletePoolFromSystem deletes a storage pool from the system
func (m *VMStorageManager) deletePoolFromSystem(ctx context.Context, pool *StoragePool) error {
	// In a real implementation, this would delete the storage pool from the system
	// For example, removing a directory, unmounting NFS, etc.

	// For simplicity, we'll just log the operation
	log.Printf("Deleting storage pool %s (%s) from system", pool.Name, pool.ID)

	return nil
}

// createVolumeInSystem creates a storage volume in the system
func (m *VMStorageManager) createVolumeInSystem(ctx context.Context, pool *StoragePool, volume *StorageVolume) error {
	// In a real implementation, this would create the storage volume in the system
	// For example, creating a file, setting up RBD, etc.

	// For simplicity, we'll just log the operation
	log.Printf("Creating storage volume %s (%s) in system", volume.Name, volume.ID)

	// For local storage, create the volume file
	if pool.Type == StorageTypeLocal {
		// Create the volume file based on format
		switch volume.Format {
		case StorageFormatRaw:
			// Create a raw file
			cmd := exec.CommandContext(ctx, "dd", "if=/dev/zero", "of="+volume.Path, "bs=1M", "count="+strconv.FormatInt(volume.Capacity/(1024*1024), 10))
			if err := cmd.Run(); err != nil {
				return fmt.Errorf("failed to create raw volume file: %w", err)
			}
		case StorageFormatQCOW2:
			// Create a QCOW2 file
			cmd := exec.CommandContext(ctx, "qemu-img", "create", "-f", "qcow2", volume.Path, strconv.FormatInt(volume.Capacity, 10))
			if err := cmd.Run(); err != nil {
				return fmt.Errorf("failed to create QCOW2 volume file: %w", err)
			}
		case StorageFormatVMDK:
			// Create a VMDK file
			cmd := exec.CommandContext(ctx, "qemu-img", "create", "-f", "vmdk", volume.Path, strconv.FormatInt(volume.Capacity, 10))
			if err := cmd.Run(); err != nil {
				return fmt.Errorf("failed to create VMDK volume file: %w", err)
			}
		case StorageFormatVHD:
			// Create a VHD file
			cmd := exec.CommandContext(ctx, "qemu-img", "create", "-f", "vpc", volume.Path, strconv.FormatInt(volume.Capacity, 10))
			if err := cmd.Run(); err != nil {
				return fmt.Errorf("failed to create VHD volume file: %w", err)
			}
		}
	}

	return nil
}

// deleteVolumeFromSystem deletes a storage volume from the system
func (m *VMStorageManager) deleteVolumeFromSystem(ctx context.Context, pool *StoragePool, volume *StorageVolume) error {
	// In a real implementation, this would delete the storage volume from the system
	// For example, removing a file, deleting RBD, etc.

	// For simplicity, we'll just log the operation
	log.Printf("Deleting storage volume %s (%s) from system", volume.Name, volume.ID)

	// For local storage, delete the volume file
	if pool.Type == StorageTypeLocal {
		if err := os.Remove(volume.Path); err != nil {
			return fmt.Errorf("failed to delete volume file: %w", err)
		}
	}

	return nil
}

// resizeVolumeInSystem resizes a storage volume in the system
func (m *VMStorageManager) resizeVolumeInSystem(ctx context.Context, pool *StoragePool, volume *StorageVolume, newCapacity int64) error {
	// In a real implementation, this would resize the storage volume in the system
	// For example, resizing a file, resizing RBD, etc.

	// For simplicity, we'll just log the operation
	log.Printf("Resizing storage volume %s (%s) in system", volume.Name, volume.ID)

	// For local storage, resize the volume file
	if pool.Type == StorageTypeLocal {
		// Resize the volume file based on format
		switch volume.Format {
		case StorageFormatRaw:
			// Resize a raw file
			cmd := exec.CommandContext(ctx, "truncate", "-s", strconv.FormatInt(newCapacity, 10), volume.Path)
			if err := cmd.Run(); err != nil {
				return fmt.Errorf("failed to resize raw volume file: %w", err)
			}
		case StorageFormatQCOW2, StorageFormatVMDK, StorageFormatVHD:
			// Resize a QCOW2/VMDK/VHD file
			cmd := exec.CommandContext(ctx, "qemu-img", "resize", volume.Path, strconv.FormatInt(newCapacity, 10))
			if err := cmd.Run(); err != nil {
				return fmt.Errorf("failed to resize volume file: %w", err)
			}
		}
	}

	return nil
}

// cloneVolumeInSystem clones a storage volume in the system
func (m *VMStorageManager) cloneVolumeInSystem(ctx context.Context, pool *StoragePool, sourceVolume, targetVolume *StorageVolume) error {
	// In a real implementation, this would clone the storage volume in the system
	// For example, copying a file, cloning RBD, etc.

	// For simplicity, we'll just log the operation
	log.Printf("Cloning storage volume %s (%s) to %s (%s) in system", sourceVolume.Name, sourceVolume.ID, targetVolume.Name, targetVolume.ID)

	// For local storage, clone the volume file
	if pool.Type == StorageTypeLocal {
		// Clone the volume file based on format
		switch sourceVolume.Format {
		case StorageFormatRaw:
			// Clone a raw file
			cmd := exec.CommandContext(ctx, "cp", sourceVolume.Path, targetVolume.Path)
			if err := cmd.Run(); err != nil {
				return fmt.Errorf("failed to clone raw volume file: %w", err)
			}
		case StorageFormatQCOW2, StorageFormatVMDK, StorageFormatVHD:
			// Clone a QCOW2/VMDK/VHD file
			cmd := exec.CommandContext(ctx, "qemu-img", "convert", "-O", string(sourceVolume.Format), sourceVolume.Path, targetVolume.Path)
			if err := cmd.Run(); err != nil {
				return fmt.Errorf("failed to clone volume file: %w", err)
			}
		}
	}

	return nil
}
