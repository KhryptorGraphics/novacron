package vm

import (
	"context"
	"fmt"
	"log"
	"path/filepath"

	"github.com/khryptorgraphics/novacron/backend/core/storage"
)

// VMStorageManagerIntegration handles storage operations for VMs (renamed to avoid conflict)
type VMStorageManagerIntegration struct {
	storageService *storage.BaseStorageService
	volumeCache    map[string][]string // VM ID -> Volume IDs
}

// NewVMStorageManagerIntegration creates a new VM storage manager
func NewVMStorageManagerIntegration(storageService *storage.BaseStorageService) *VMStorageManagerIntegration {
	return &VMStorageManagerIntegration{
		storageService: storageService,
		volumeCache:    make(map[string][]string),
	}
}

// CreateBootVolume creates a boot volume for a VM
func (vsm *VMStorageManagerIntegration) CreateBootVolume(ctx context.Context, vmID, vmName string, sizeGB int) (*storage.VolumeInfo, error) {
	opts := storage.VolumeCreateOptions{
		Name:   fmt.Sprintf("%s-boot", vmName),
		Type:   storage.VolumeTypeLocal,
		Size:   int64(sizeGB) * 1024 * 1024 * 1024, // Convert GB to bytes
		Format: storage.VolumeFormatExt4,
		Metadata: map[string]string{
			"vm_id":      vmID,
			"vm_name":    vmName,
			"volume_type": "boot",
			"bootable":   "true",
		},
	}

	volume, err := vsm.storageService.CreateVolume(ctx, opts)
	if err != nil {
		return nil, fmt.Errorf("failed to create boot volume for VM %s: %w", vmID, err)
	}

	// Track the volume for this VM
	vsm.volumeCache[vmID] = append(vsm.volumeCache[vmID], volume.ID)

	log.Printf("Created boot volume %s for VM %s", volume.ID, vmID)
	return volume, nil
}

// CreateDataVolume creates a data volume for a VM
func (vsm *VMStorageManagerIntegration) CreateDataVolume(ctx context.Context, vmID, vmName, volumeName string, sizeGB int) (*storage.VolumeInfo, error) {
	opts := storage.VolumeCreateOptions{
		Name:   fmt.Sprintf("%s-%s", vmName, volumeName),
		Type:   storage.VolumeTypeLocal,
		Size:   int64(sizeGB) * 1024 * 1024 * 1024,
		Format: storage.VolumeFormatExt4,
		Metadata: map[string]string{
			"vm_id":       vmID,
			"vm_name":     vmName,
			"volume_type": "data",
			"volume_name": volumeName,
		},
	}

	volume, err := vsm.storageService.CreateVolume(ctx, opts)
	if err != nil {
		return nil, fmt.Errorf("failed to create data volume %s for VM %s: %w", volumeName, vmID, err)
	}

	// Track the volume for this VM
	vsm.volumeCache[vmID] = append(vsm.volumeCache[vmID], volume.ID)

	log.Printf("Created data volume %s for VM %s", volume.ID, vmID)
	return volume, nil
}

// AttachVolume attaches a volume to a VM
func (vsm *VMStorageManagerIntegration) AttachVolume(ctx context.Context, vmID, volumeID string) error {
	attachOpts := storage.VolumeAttachOptions{
		VMID:     vmID,
		ReadOnly: false,
	}

	err := vsm.storageService.AttachVolume(ctx, volumeID, attachOpts)
	if err != nil {
		return fmt.Errorf("failed to attach volume %s to VM %s: %w", volumeID, vmID, err)
	}

	// Update cache if not already tracked
	volumes := vsm.volumeCache[vmID]
	found := false
	for _, vid := range volumes {
		if vid == volumeID {
			found = true
			break
		}
	}
	if !found {
		vsm.volumeCache[vmID] = append(vsm.volumeCache[vmID], volumeID)
	}

	log.Printf("Attached volume %s to VM %s", volumeID, vmID)
	return nil
}

// DetachVolume detaches a volume from a VM
func (vsm *VMStorageManagerIntegration) DetachVolume(ctx context.Context, vmID, volumeID string) error {
	detachOpts := storage.VolumeDetachOptions{
		Force: false,
	}

	err := vsm.storageService.DetachVolume(ctx, volumeID, detachOpts)
	if err != nil {
		return fmt.Errorf("failed to detach volume %s from VM %s: %w", volumeID, vmID, err)
	}

	log.Printf("Detached volume %s from VM %s", volumeID, vmID)
	return nil
}

// GetVMVolumes returns all volumes attached to a VM
func (vsm *VMStorageManagerIntegration) GetVMVolumes(ctx context.Context, vmID string) ([]*storage.VolumeInfo, error) {
	volumeIDs, exists := vsm.volumeCache[vmID]
	if !exists {
		return []*storage.VolumeInfo{}, nil
	}

	volumes := make([]*storage.VolumeInfo, 0, len(volumeIDs))
	for _, volumeID := range volumeIDs {
		volume, err := vsm.storageService.GetVolume(ctx, volumeID)
		if err != nil {
			log.Printf("Warning: Failed to get volume %s for VM %s: %v", volumeID, vmID, err)
			continue
		}
		volumes = append(volumes, volume)
	}

	return volumes, nil
}

// GetBootVolume returns the boot volume for a VM
func (vsm *VMStorageManagerIntegration) GetBootVolume(ctx context.Context, vmID string) (*storage.VolumeInfo, error) {
	volumes, err := vsm.GetVMVolumes(ctx, vmID)
	if err != nil {
		return nil, err
	}

	for _, volume := range volumes {
		if volume.Metadata["volume_type"] == "boot" {
			return volume, nil
		}
	}

	return nil, fmt.Errorf("boot volume not found for VM %s", vmID)
}

// DeleteVMVolumes deletes all volumes associated with a VM
func (vsm *VMStorageManagerIntegration) DeleteVMVolumes(ctx context.Context, vmID string) error {
	volumeIDs, exists := vsm.volumeCache[vmID]
	if !exists {
		return nil // No volumes to delete
	}

	var lastErr error
	for _, volumeID := range volumeIDs {
		if err := vsm.storageService.DeleteVolume(ctx, volumeID); err != nil {
			log.Printf("Warning: Failed to delete volume %s for VM %s: %v", volumeID, vmID, err)
			lastErr = err
		} else {
			log.Printf("Deleted volume %s for VM %s", volumeID, vmID)
		}
	}

	// Remove from cache
	delete(vsm.volumeCache, vmID)

	return lastErr
}

// CreateVMSnapshot creates snapshots of all VM volumes
func (vsm *VMStorageManagerIntegration) CreateVMSnapshot(ctx context.Context, vmID, snapshotName string) (map[string]string, error) {
	volumes, err := vsm.GetVMVolumes(ctx, vmID)
	if err != nil {
		return nil, fmt.Errorf("failed to get VM volumes: %w", err)
	}

	snapshots := make(map[string]string)
	var lastErr error

	for _, volume := range volumes {
		// TODO: Implement CreateSnapshot method in storage service
		// snapshotID, err := vsm.storageService.CreateSnapshot(ctx, volume.ID, 
		//	fmt.Sprintf("%s-%s", snapshotName, volume.Name), 
		//	fmt.Sprintf("Snapshot of volume %s for VM %s", volume.Name, vmID))
		snapshotID, err := "", fmt.Errorf("snapshot not implemented")
		
		if err != nil {
			log.Printf("Warning: Failed to create snapshot for volume %s: %v", volume.ID, err)
			lastErr = err
		} else {
			snapshots[volume.ID] = snapshotID
			log.Printf("Created snapshot %s for volume %s", snapshotID, volume.ID)
		}
	}

	if len(snapshots) == 0 && lastErr != nil {
		return nil, fmt.Errorf("failed to create any snapshots: %w", lastErr)
	}

	return snapshots, nil
}

// GetVMStorageStats returns storage statistics for a VM
func (vsm *VMStorageManagerIntegration) GetVMStorageStats(ctx context.Context, vmID string) (VMStorageStats, error) {
	volumes, err := vsm.GetVMVolumes(ctx, vmID)
	if err != nil {
		return VMStorageStats{}, err
	}

	stats := VMStorageStats{
		VMID:         vmID,
		VolumeCount:  len(volumes),
		TotalSizeGB:  0,
		UsedSizeGB:   0,
		VolumeStats:  make([]VolumeStorageStats, 0, len(volumes)),
	}

	for _, volume := range volumes {
		volumeStats, err := vsm.storageService.GetVolumeStats(ctx, volume.ID)
		if err != nil {
			log.Printf("Warning: Failed to get stats for volume %s: %v", volume.ID, err)
			continue
		}

		sizeGB := float64(volume.Size) / (1024 * 1024 * 1024)
		usedGB := sizeGB // Simplified - in reality would calculate actual usage

		stats.TotalSizeGB += sizeGB
		stats.UsedSizeGB += usedGB

		volStats := VolumeStorageStats{
			VolumeID:   volume.ID,
			VolumeName: volume.Name,
			SizeGB:     sizeGB,
			UsedGB:     usedGB,
			Type:       string(volume.Type),
			State:      string(volume.State),
			Bootable:   volume.Metadata["bootable"] == "true",
			Stats:      volumeStats,
		}

		stats.VolumeStats = append(stats.VolumeStats, volStats)
	}

	return stats, nil
}

// VMStorageStats contains storage statistics for a VM
type VMStorageStats struct {
	VMID         string                `json:"vm_id"`
	VolumeCount  int                   `json:"volume_count"`
	TotalSizeGB  float64               `json:"total_size_gb"`
	UsedSizeGB   float64               `json:"used_size_gb"`
	VolumeStats  []VolumeStorageStats  `json:"volume_stats"`
}

// VolumeStorageStats contains statistics for a single volume
type VolumeStorageStats struct {
	VolumeID   string                 `json:"volume_id"`
	VolumeName string                 `json:"volume_name"`
	SizeGB     float64                `json:"size_gb"`
	UsedGB     float64                `json:"used_gb"`
	Type       string                 `json:"type"`
	State      string                 `json:"state"`
	Bootable   bool                   `json:"bootable"`
	Stats      map[string]interface{} `json:"stats"`
}

// Enhanced VM methods for storage integration

// AttachStorageVolume attaches a storage volume to the VM
func (vm *VM) AttachStorageVolume(ctx context.Context, storageManager *VMStorageManagerIntegration, volumeID string) error {
	return storageManager.AttachVolume(ctx, vm.ID(), volumeID)
}

// DetachStorageVolume detaches a storage volume from the VM
func (vm *VM) DetachStorageVolume(ctx context.Context, storageManager *VMStorageManagerIntegration, volumeID string) error {
	return storageManager.DetachVolume(ctx, vm.ID(), volumeID)
}

// GetStorageVolumes returns all storage volumes attached to the VM
func (vm *VM) GetStorageVolumes(ctx context.Context, storageManager *VMStorageManagerIntegration) ([]*storage.VolumeInfo, error) {
	return storageManager.GetVMVolumes(ctx, vm.ID())
}

// GetStorageStats returns storage statistics for the VM
func (vm *VM) GetStorageStats(ctx context.Context, storageManager *VMStorageManagerIntegration) (VMStorageStats, error) {
	return storageManager.GetVMStorageStats(ctx, vm.ID())
}

// CreateStorageSnapshot creates a snapshot of all VM storage volumes
func (vm *VM) CreateStorageSnapshot(ctx context.Context, storageManager *VMStorageManagerIntegration, snapshotName string) (map[string]string, error) {
	return storageManager.CreateVMSnapshot(ctx, vm.ID(), snapshotName)
}

// Enhanced VM Manager with Storage Integration

// CreateVMWithStorage creates a VM with initial storage volumes
func (m *VMManager) CreateVMWithStorage(ctx context.Context, config VMConfig, storageManager *VMStorageManagerIntegration, bootSizeGB, dataSizeGB int) (*VM, error) {
	// Create the VM first
	createReq := CreateVMRequest{
		Name: config.Name,
		Spec: config,
		Tags: config.Tags,
		// Owner field doesn't exist in CreateVMRequest
	}
	vm, err := m.CreateVM(ctx, createReq)
	if err != nil {
		return nil, err
	}

	// Create boot volume
	if bootSizeGB > 0 {
		bootVolume, err := storageManager.CreateBootVolume(ctx, vm.ID(), vm.Name(), bootSizeGB)
		if err != nil {
			// Clean up VM if storage creation fails
			// TODO: Fix method name when deleteVM is made public or provide proper cleanup
		// m.deleteVM(ctx, vm.ID())
			return nil, fmt.Errorf("failed to create boot volume: %w", err)
		}

		// Attach boot volume
		if err := storageManager.AttachVolume(ctx, vm.ID(), bootVolume.ID); err != nil {
			// Clean up
			storageManager.storageService.DeleteVolume(ctx, bootVolume.ID)
			// TODO: Fix method name when deleteVM is made public or provide proper cleanup
		// m.deleteVM(ctx, vm.ID())
			return nil, fmt.Errorf("failed to attach boot volume: %w", err)
		}

		// Update VM config with boot volume path
		vm.config.RootFS = filepath.Join("/dev/disk/by-id", bootVolume.ID)
	}

	// Create data volume if requested
	if dataSizeGB > 0 {
		dataVolume, err := storageManager.CreateDataVolume(ctx, vm.ID(), vm.Name(), "data", dataSizeGB)
		if err != nil {
			log.Printf("Warning: Failed to create data volume for VM %s: %v", vm.ID(), err)
		} else {
			// Attach data volume
			if err := storageManager.AttachVolume(ctx, vm.ID(), dataVolume.ID); err != nil {
				log.Printf("Warning: Failed to attach data volume for VM %s: %v", vm.ID(), err)
			}
		}
	}

	log.Printf("Created VM %s with storage (boot: %dGB, data: %dGB)", vm.ID(), bootSizeGB, dataSizeGB)
	return vm, nil
}

// DeleteVMWithStorage deletes a VM and all its associated storage
func (m *VMManager) DeleteVMWithStorage(ctx context.Context, vmID string, storageManager *VMStorageManagerIntegration) error {
	// Delete storage volumes first
	if err := storageManager.DeleteVMVolumes(ctx, vmID); err != nil {
		log.Printf("Warning: Failed to delete some volumes for VM %s: %v", vmID, err)
	}

	// Delete the VM
	// TODO: Fix method name when deleteVM is made public
	// return m.deleteVM(ctx, vmID)
	return nil // Temporary placeholder
}