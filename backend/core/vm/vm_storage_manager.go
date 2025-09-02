package vm

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/storage"
)

// VMStorageManager manages storage operations for virtual machines
type VMStorageManager struct {
	// Integrated storage manager
	storageManager *storage.IntegratedStorageManager

	// VM-specific volume mappings
	vmVolumes map[string][]string // VM ID -> Volume IDs
	volumeVMs map[string]string   // Volume ID -> VM ID

	// Mutex for thread safety
	mu sync.RWMutex

	// Context for operations
	ctx context.Context
}

// StorageType represents the type of storage
type StorageType string

const (
	StorageTypeLocal StorageType = "local"
	StorageTypeNFS   StorageType = "nfs"
	StorageTypeCeph  StorageType = "ceph"
	StorageTypeISCSI StorageType = "iscsi"
)

// StorageFormat represents the format of storage
type StorageFormat string

const (
	StorageFormatRaw   StorageFormat = "raw"
	StorageFormatQCOW2 StorageFormat = "qcow2"
	StorageFormatVMDK  StorageFormat = "vmdk"
	StorageFormatVHD   StorageFormat = "vhd"
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
	Tags       []string          `json:"tags"`
	Metadata   map[string]string `json:"metadata"`
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
	Tags       []string          `json:"tags"`
	Metadata   map[string]string `json:"metadata"`
}

// NewVMStorageManager creates a new VM storage manager
func NewVMStorageManager(ctx context.Context) *VMStorageManager {
	// Create integrated storage manager
	config := storage.DefaultStorageConfig()
	storageManager := storage.NewIntegratedStorageManager(config)

	return &VMStorageManager{
		storageManager: storageManager,
		vmVolumes:     make(map[string][]string),
		volumeVMs:     make(map[string]string),
		ctx:           ctx,
	}
}

// Start starts the VM storage manager
func (vsm *VMStorageManager) Start() error {
	return vsm.storageManager.Start()
}

// Stop stops the VM storage manager
func (vsm *VMStorageManager) Stop() error {
	return vsm.storageManager.Stop()
}

// CreateStoragePool creates a new storage pool
func (vsm *VMStorageManager) CreateStoragePool(ctx context.Context, name string, storageType StorageType, path string, tags []string, metadata map[string]string) (*StoragePool, error) {
	// For now, create a simple pool representation
	// In a real implementation, this would create actual storage pools
	pool := &StoragePool{
		ID:         fmt.Sprintf("pool-%d", time.Now().Unix()),
		Name:       name,
		Type:       storageType,
		Path:       path,
		TotalSpace: 1000 * 1024 * 1024 * 1024, // 1TB default
		UsedSpace:  0,
		CreatedAt:  time.Now(),
		UpdatedAt:  time.Now(),
		Tags:       tags,
		Metadata:   metadata,
	}

	return pool, nil
}

// ListStoragePools lists all storage pools
func (vsm *VMStorageManager) ListStoragePools() []*StoragePool {
	// Return mock pools for now
	return []*StoragePool{
		{
			ID:         "pool-local",
			Name:       "Local Storage",
			Type:       StorageTypeLocal,
			Path:       "/var/lib/novacron/volumes",
			TotalSpace: 1000 * 1024 * 1024 * 1024,
			UsedSpace:  100 * 1024 * 1024 * 1024,
			CreatedAt:  time.Now().Add(-24 * time.Hour),
			UpdatedAt:  time.Now(),
			Tags:       []string{"default", "local"},
			Metadata:   map[string]string{"type": "local"},
		},
		{
			ID:         "pool-ceph",
			Name:       "Ceph Storage",
			Type:       StorageTypeCeph,
			Path:       "rbd",
			TotalSpace: 10000 * 1024 * 1024 * 1024,
			UsedSpace:  2000 * 1024 * 1024 * 1024,
			CreatedAt:  time.Now().Add(-48 * time.Hour),
			UpdatedAt:  time.Now(),
			Tags:       []string{"distributed", "ceph"},
			Metadata:   map[string]string{"type": "ceph", "pool": "novacron"},
		},
	}
}

// GetStoragePool gets a storage pool by ID
func (vsm *VMStorageManager) GetStoragePool(poolID string) (*StoragePool, error) {
	pools := vsm.ListStoragePools()
	for _, pool := range pools {
		if pool.ID == poolID {
			return pool, nil
		}
	}
	return nil, fmt.Errorf("storage pool %s not found", poolID)
}

// DeleteStoragePool deletes a storage pool
func (vsm *VMStorageManager) DeleteStoragePool(ctx context.Context, poolID string) error {
	// For now, just validate that the pool exists
	_, err := vsm.GetStoragePool(poolID)
	return err
}

// CreateStorageVolume creates a new storage volume
func (vsm *VMStorageManager) CreateStorageVolume(ctx context.Context, name string, poolID string, format StorageFormat, capacity int64, tags []string, metadata map[string]string) (*StorageVolume, error) {
	// Convert to storage manager format
	volumeType := storage.VolumeTypeLocal
	if poolID == "pool-ceph" {
		volumeType = storage.VolumeTypeCeph
	}

	opts := storage.VolumeCreateOptions{
		Name:              name,
		Type:              volumeType,
		Size:              capacity,
		Format:            storage.VolumeFormat(format),
		Bootable:          false,
		Encrypted:         false,
		ReplicationFactor: 1,
		Metadata:          metadata,
	}

	// Create volume using integrated storage manager
	volumeInfo, err := vsm.storageManager.CreateVolume(ctx, opts)
	if err != nil {
		return nil, fmt.Errorf("failed to create volume: %v", err)
	}

	// Convert back to VM storage format
	volume := &StorageVolume{
		ID:         volumeInfo.ID,
		Name:       volumeInfo.Name,
		PoolID:     poolID,
		Format:     format,
		Capacity:   volumeInfo.Size,
		Allocation: volumeInfo.Size / 2, // Assume 50% allocation
		Path:       volumeInfo.Path,
		CreatedAt:  volumeInfo.CreatedAt,
		UpdatedAt:  volumeInfo.UpdatedAt,
		Tags:       tags,
		Metadata:   metadata,
	}

	return volume, nil
}

// ListStorageVolumes lists all storage volumes
func (vsm *VMStorageManager) ListStorageVolumes() []*StorageVolume {
	volumes, err := vsm.storageManager.ListVolumes(vsm.ctx)
	if err != nil {
		return []*StorageVolume{}
	}

	result := make([]*StorageVolume, 0, len(volumes))
	for _, vol := range volumes {
		storageVol := &StorageVolume{
			ID:         vol.ID,
			Name:       vol.Name,
			PoolID:     "pool-local", // Default pool
			Format:     StorageFormat(vol.Format),
			Capacity:   vol.Size,
			Allocation: vol.Size / 2,
			Path:       vol.Path,
			CreatedAt:  vol.CreatedAt,
			UpdatedAt:  vol.UpdatedAt,
			Tags:       []string{},
			Metadata:   vol.Metadata,
		}

		// Determine pool based on volume type
		if vol.Type == storage.VolumeTypeCeph {
			storageVol.PoolID = "pool-ceph"
		}

		result = append(result, storageVol)
	}

	return result
}

// ListStorageVolumesInPool lists storage volumes in a specific pool
func (vsm *VMStorageManager) ListStorageVolumesInPool(poolID string) []*StorageVolume {
	allVolumes := vsm.ListStorageVolumes()
	result := make([]*StorageVolume, 0)

	for _, vol := range allVolumes {
		if vol.PoolID == poolID {
			result = append(result, vol)
		}
	}

	return result
}

// GetStorageVolume gets a storage volume by ID
func (vsm *VMStorageManager) GetStorageVolume(volumeID string) (*StorageVolume, error) {
	volumeInfo, err := vsm.storageManager.GetVolume(vsm.ctx, volumeID)
	if err != nil {
		return nil, err
	}

	volume := &StorageVolume{
		ID:         volumeInfo.ID,
		Name:       volumeInfo.Name,
		PoolID:     "pool-local",
		Format:     StorageFormat(volumeInfo.Format),
		Capacity:   volumeInfo.Size,
		Allocation: volumeInfo.Size / 2,
		Path:       volumeInfo.Path,
		CreatedAt:  volumeInfo.CreatedAt,
		UpdatedAt:  volumeInfo.UpdatedAt,
		Tags:       []string{},
		Metadata:   volumeInfo.Metadata,
	}

	if volumeInfo.Type == storage.VolumeTypeCeph {
		volume.PoolID = "pool-ceph"
	}

	return volume, nil
}

// DeleteStorageVolume deletes a storage volume
func (vsm *VMStorageManager) DeleteStorageVolume(ctx context.Context, volumeID string) error {
	vsm.mu.Lock()
	defer vsm.mu.Unlock()

	// Check if volume is attached to any VM
	if vmID, attached := vsm.volumeVMs[volumeID]; attached {
		return fmt.Errorf("volume %s is attached to VM %s", volumeID, vmID)
	}

	return vsm.storageManager.DeleteVolume(ctx, volumeID)
}

// ResizeStorageVolume resizes a storage volume
func (vsm *VMStorageManager) ResizeStorageVolume(ctx context.Context, volumeID string, newCapacity int64) (*StorageVolume, error) {
	// For now, just return the existing volume with updated capacity
	// In a real implementation, this would resize the actual storage
	volume, err := vsm.GetStorageVolume(volumeID)
	if err != nil {
		return nil, err
	}

	volume.Capacity = newCapacity
	volume.UpdatedAt = time.Now()

	return volume, nil
}

// CloneStorageVolume clones a storage volume
func (vsm *VMStorageManager) CloneStorageVolume(ctx context.Context, volumeID string, name string, tags []string, metadata map[string]string) (*StorageVolume, error) {
	// Get source volume
	sourceVolume, err := vsm.GetStorageVolume(volumeID)
	if err != nil {
		return nil, fmt.Errorf("source volume not found: %v", err)
	}

	// Create new volume with same properties
	return vsm.CreateStorageVolume(ctx, name, sourceVolume.PoolID, sourceVolume.Format, sourceVolume.Capacity, tags, metadata)
}

// AttachVolumeToVM attaches a storage volume to a VM
func (vsm *VMStorageManager) AttachVolumeToVM(ctx context.Context, volumeID string, vmID string, device string, readOnly bool) error {
	vsm.mu.Lock()
	defer vsm.mu.Unlock()

	// Check if volume is already attached
	if attachedVM, attached := vsm.volumeVMs[volumeID]; attached {
		if attachedVM == vmID {
			return nil // Already attached to this VM
		}
		return fmt.Errorf("volume %s is already attached to VM %s", volumeID, attachedVM)
	}

	// Attach volume using storage manager
	opts := storage.VolumeAttachOptions{
		VMID:     vmID,
		Device:   device,
		ReadOnly: readOnly,
	}

	if err := vsm.storageManager.AttachVolume(ctx, volumeID, opts); err != nil {
		return fmt.Errorf("failed to attach volume: %v", err)
	}

	// Update mappings
	if vsm.vmVolumes[vmID] == nil {
		vsm.vmVolumes[vmID] = make([]string, 0)
	}
	vsm.vmVolumes[vmID] = append(vsm.vmVolumes[vmID], volumeID)
	vsm.volumeVMs[volumeID] = vmID

	return nil
}

// DetachVolumeFromVM detaches a storage volume from a VM
func (vsm *VMStorageManager) DetachVolumeFromVM(ctx context.Context, volumeID string, vmID string, force bool) error {
	vsm.mu.Lock()
	defer vsm.mu.Unlock()

	// Check if volume is attached to this VM
	attachedVM, attached := vsm.volumeVMs[volumeID]
	if !attached || attachedVM != vmID {
		return fmt.Errorf("volume %s is not attached to VM %s", volumeID, vmID)
	}

	// Detach volume using storage manager
	opts := storage.VolumeDetachOptions{
		Force: force,
	}

	if err := vsm.storageManager.DetachVolume(ctx, volumeID, opts); err != nil {
		return fmt.Errorf("failed to detach volume: %v", err)
	}

	// Update mappings
	if volumes, exists := vsm.vmVolumes[vmID]; exists {
		// Remove volume from VM's volume list
		for i, vol := range volumes {
			if vol == volumeID {
				vsm.vmVolumes[vmID] = append(volumes[:i], volumes[i+1:]...)
				break
			}
		}
	}
	delete(vsm.volumeVMs, volumeID)

	return nil
}

// GetVMVolumes gets all volumes attached to a VM
func (vsm *VMStorageManager) GetVMVolumes(vmID string) ([]*StorageVolume, error) {
	vsm.mu.RLock()
	defer vsm.mu.RUnlock()

	volumeIDs, exists := vsm.vmVolumes[vmID]
	if !exists {
		return []*StorageVolume{}, nil
	}

	volumes := make([]*StorageVolume, 0, len(volumeIDs))
	for _, volumeID := range volumeIDs {
		volume, err := vsm.GetStorageVolume(volumeID)
		if err != nil {
			continue // Skip volumes that can't be found
		}
		volumes = append(volumes, volume)
	}

	return volumes, nil
}

// GetStorageMetrics gets storage metrics
func (vsm *VMStorageManager) GetStorageMetrics(ctx context.Context) (*storage.StorageMetrics, error) {
	return vsm.storageManager.GetStorageMetrics(ctx)
}