package storage

import (
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"time"

	"github.com/google/uuid"
)

// deleteLocalVolume deletes a local volume
func (m *StorageManager) deleteLocalVolume(volume *Volume) error {
	// Check if the volume file exists
	if _, err := os.Stat(volume.Path); os.IsNotExist(err) {
		log.Printf("Warning: Volume file %s does not exist", volume.Path)
		return nil
	}

	// Delete the volume file
	if err := os.Remove(volume.Path); err != nil {
		return fmt.Errorf("failed to delete volume file: %w", err)
	}

	// Delete the metadata file if it exists
	metadataPath := volume.Path + ".meta"
	if _, err := os.Stat(metadataPath); err == nil {
		if err := os.Remove(metadataPath); err != nil {
			log.Printf("Warning: Failed to delete volume metadata file: %v", err)
		}
	}

	log.Printf("Deleted local volume at %s", volume.Path)
	return nil
}

// deleteNFSVolume deletes an NFS volume
func (m *StorageManager) deleteNFSVolume(volume *Volume) error {
	// In a real implementation, this would delete the volume from the NFS server
	log.Printf("NFS volume would be deleted: %s", volume.Path)
	return nil
}

// deleteBlockVolume deletes a block volume
func (m *StorageManager) deleteBlockVolume(volume *Volume) error {
	// In a real implementation, this would delete the block device
	log.Printf("Block volume would be deleted: %s", volume.Name)
	return nil
}

// deleteCephVolume deletes a Ceph volume
func (m *StorageManager) deleteCephVolume(volume *Volume) error {
	// In a real implementation, this would delete the Ceph RBD volume
	log.Printf("Ceph volume would be deleted: %s", volume.Name)
	return nil
}

// CreateSnapshot creates a snapshot of a volume
func (m *StorageManager) CreateSnapshot(volumeID, name, description string) (*Snapshot, error) {
	// Get the volume
	m.volumesMutex.Lock()
	defer m.volumesMutex.Unlock()

	volume, exists := m.volumes[volumeID]
	if !exists {
		return nil, fmt.Errorf("volume %s not found", volumeID)
	}

	// Check if a snapshot with this name already exists
	for _, snap := range volume.VolumeInfo.Snapshots {
		if snap.Name == name {
			return nil, fmt.Errorf("snapshot with name %s already exists for volume %s", name, volumeID)
		}
	}

	// Generate a unique ID for the snapshot
	snapshotID := uuid.New().String()

	// Create the snapshot
	snapshot := Snapshot{
		ID:          snapshotID,
		Name:        name,
		VolumeID:    volumeID,
		CreatedAt:   time.Now(),
		SizeMB:      volume.SizeMB, // Initial size is the same as the volume
		Description: description,
	}

	// Create the snapshot based on volume type
	var err error
	switch volume.Type {
	case VolumeTypeLocal:
		err = m.createLocalSnapshot(volume, &snapshot)
	case VolumeTypeNFS:
		err = m.createNFSSnapshot(volume, &snapshot)
	case VolumeTypeBlock:
		err = m.createBlockSnapshot(volume, &snapshot)
	case VolumeTypeCeph:
		err = m.createCephSnapshot(volume, &snapshot)
	default:
		err = fmt.Errorf("unsupported volume type: %s", volume.Type)
	}

	if err != nil {
		return nil, fmt.Errorf("failed to create snapshot: %w", err)
	}

	// Add the snapshot to the volume
	volume.VolumeInfo.Snapshots = append(volume.VolumeInfo.Snapshots, snapshot)
	volume.VolumeInfo.LastUpdated = time.Now()

	// Emit snapshot created event
	m.emitEvent(VolumeEvent{
		Type:      VolumeEventSnapshoted,
		Volume:    *volume,
		Timestamp: time.Now(),
		NodeID:    m.nodeID,
		Message:   fmt.Sprintf("Snapshot %s created for volume %s", name, volume.Name),
	})

	log.Printf("Created snapshot %s for volume %s", name, volume.Name)

	return &snapshot, nil
}

// createLocalSnapshot creates a snapshot of a local volume
func (m *StorageManager) createLocalSnapshot(volume *Volume, snapshot *Snapshot) error {
	// Create snapshots directory if it doesn't exist
	snapshotsDir := filepath.Join(filepath.Dir(volume.Path), "snapshots")
	if err := os.MkdirAll(snapshotsDir, 0755); err != nil {
		return fmt.Errorf("failed to create snapshots directory: %w", err)
	}

	// Create snapshot file path
	snapshotPath := filepath.Join(snapshotsDir, snapshot.ID)

	// Create the snapshot using qemu-img
	cmd := exec.Command("qemu-img", "snapshot", "-c", snapshot.Name, volume.Path)
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("failed to create snapshot: %w", err)
	}

	// Create a metadata file for the snapshot
	metadataPath := snapshotPath + ".meta"
	metadata, err := json.MarshalIndent(snapshot, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal snapshot metadata: %w", err)
	}

	if err := os.WriteFile(metadataPath, metadata, 0644); err != nil {
		return fmt.Errorf("failed to write snapshot metadata: %w", err)
	}

	log.Printf("Created local snapshot at %s", snapshotPath)
	return nil
}

// createNFSSnapshot creates a snapshot of an NFS volume
func (m *StorageManager) createNFSSnapshot(volume *Volume, snapshot *Snapshot) error {
	// In a real implementation, this would create a snapshot on the NFS server
	log.Printf("NFS snapshot would be created: %s", snapshot.Name)
	return nil
}

// createBlockSnapshot creates a snapshot of a block volume
func (m *StorageManager) createBlockSnapshot(volume *Volume, snapshot *Snapshot) error {
	// In a real implementation, this would create a snapshot of the block device
	log.Printf("Block snapshot would be created: %s", snapshot.Name)
	return nil
}

// createCephSnapshot creates a snapshot of a Ceph volume
func (m *StorageManager) createCephSnapshot(volume *Volume, snapshot *Snapshot) error {
	// In a real implementation, this would create a Ceph RBD snapshot
	log.Printf("Ceph snapshot would be created: %s", snapshot.Name)
	return nil
}

// DeleteSnapshot deletes a snapshot
func (m *StorageManager) DeleteSnapshot(volumeID, snapshotID string) error {
	// Get the volume
	m.volumesMutex.Lock()
	defer m.volumesMutex.Unlock()

	volume, exists := m.volumes[volumeID]
	if !exists {
		return fmt.Errorf("volume %s not found", volumeID)
	}

	// Find the snapshot
	var snapshot *Snapshot
	var snapshotIndex int
	for i, s := range volume.VolumeInfo.Snapshots {
		if s.ID == snapshotID {
			snapshot = &volume.VolumeInfo.Snapshots[i]
			snapshotIndex = i
			break
		}
	}

	if snapshot == nil {
		return fmt.Errorf("snapshot %s not found for volume %s", snapshotID, volumeID)
	}

	// Delete the snapshot based on volume type
	var err error
	switch volume.Type {
	case VolumeTypeLocal:
		err = m.deleteLocalSnapshot(volume, snapshot)
	case VolumeTypeNFS:
		err = m.deleteNFSSnapshot(volume, snapshot)
	case VolumeTypeBlock:
		err = m.deleteBlockSnapshot(volume, snapshot)
	case VolumeTypeCeph:
		err = m.deleteCephSnapshot(volume, snapshot)
	default:
		err = fmt.Errorf("unsupported volume type: %s", volume.Type)
	}

	if err != nil {
		return fmt.Errorf("failed to delete snapshot: %w", err)
	}

	// Remove the snapshot from the volume
	volume.VolumeInfo.Snapshots = append(
		volume.VolumeInfo.Snapshots[:snapshotIndex],
		volume.VolumeInfo.Snapshots[snapshotIndex+1:]...,
	)
	volume.VolumeInfo.LastUpdated = time.Now()

	// Emit snapshot deleted event
	m.emitEvent(VolumeEvent{
		Type:      VolumeEventDeleted, // Reuse the deleted event type
		Volume:    *volume,
		Timestamp: time.Now(),
		NodeID:    m.nodeID,
		Message:   fmt.Sprintf("Snapshot %s deleted from volume %s", snapshot.Name, volume.Name),
	})

	log.Printf("Deleted snapshot %s from volume %s", snapshot.Name, volume.Name)

	return nil
}

// deleteLocalSnapshot deletes a snapshot of a local volume
func (m *StorageManager) deleteLocalSnapshot(volume *Volume, snapshot *Snapshot) error {
	// Delete the snapshot using qemu-img
	cmd := exec.Command("qemu-img", "snapshot", "-d", snapshot.Name, volume.Path)
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("failed to delete snapshot: %w", err)
	}

	// Delete the metadata file if it exists
	snapshotsDir := filepath.Join(filepath.Dir(volume.Path), "snapshots")
	metadataPath := filepath.Join(snapshotsDir, snapshot.ID+".meta")
	if _, err := os.Stat(metadataPath); err == nil {
		if err := os.Remove(metadataPath); err != nil {
			log.Printf("Warning: Failed to delete snapshot metadata file: %v", err)
		}
	}

	log.Printf("Deleted local snapshot %s", snapshot.Name)
	return nil
}

// deleteNFSSnapshot deletes a snapshot of an NFS volume
func (m *StorageManager) deleteNFSSnapshot(volume *Volume, snapshot *Snapshot) error {
	// In a real implementation, this would delete the snapshot from the NFS server
	log.Printf("NFS snapshot would be deleted: %s", snapshot.Name)
	return nil
}

// deleteBlockSnapshot deletes a snapshot of a block volume
func (m *StorageManager) deleteBlockSnapshot(volume *Volume, snapshot *Snapshot) error {
	// In a real implementation, this would delete the snapshot of the block device
	log.Printf("Block snapshot would be deleted: %s", snapshot.Name)
	return nil
}

// deleteCephSnapshot deletes a snapshot of a Ceph volume
func (m *StorageManager) deleteCephSnapshot(volume *Volume, snapshot *Snapshot) error {
	// In a real implementation, this would delete the Ceph RBD snapshot
	log.Printf("Ceph snapshot would be deleted: %s", snapshot.Name)
	return nil
}

// restoreLocalSnapshot restores a local volume from a snapshot
func (m *StorageManager) restoreLocalSnapshot(volume *Volume, snapshot *Snapshot) error {
	// Restore the snapshot using qemu-img
	cmd := exec.Command("qemu-img", "snapshot", "-a", snapshot.Name, volume.Path)
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("failed to restore snapshot: %w", err)
	}

	log.Printf("Restored local volume %s from snapshot %s", volume.Name, snapshot.Name)
	return nil
}

// restoreNFSSnapshot restores an NFS volume from a snapshot
func (m *StorageManager) restoreNFSSnapshot(volume *Volume, snapshot *Snapshot) error {
	// In a real implementation, this would restore the volume from the snapshot on the NFS server
	log.Printf("NFS volume would be restored from snapshot: %s", snapshot.Name)
	return nil
}

// restoreBlockSnapshot restores a block volume from a snapshot
func (m *StorageManager) restoreBlockSnapshot(volume *Volume, snapshot *Snapshot) error {
	// In a real implementation, this would restore the block device from the snapshot
	log.Printf("Block volume would be restored from snapshot: %s", snapshot.Name)
	return nil
}

// restoreCephSnapshot restores a Ceph volume from a snapshot
func (m *StorageManager) restoreCephSnapshot(volume *Volume, snapshot *Snapshot) error {
	// In a real implementation, this would restore the Ceph RBD volume from the snapshot
	log.Printf("Ceph volume would be restored from snapshot: %s", snapshot.Name)
	return nil
}

// resizeLocalVolume resizes a local volume
func (m *StorageManager) resizeLocalVolume(volume *Volume, newSizeMB int) error {
	// Resize the volume file
	cmd := exec.Command("qemu-img", "resize", volume.Path, fmt.Sprintf("%dM", newSizeMB))
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("failed to resize volume: %w", err)
	}

	// If the volume has a filesystem, resize it too
	if volume.Format != VolumeFormatRAW {
		// This would typically involve mounting the volume and resizing the filesystem
		log.Printf("Filesystem resize would be performed for volume %s", volume.Name)
	}

	log.Printf("Resized local volume %s to %d MB", volume.Name, newSizeMB)
	return nil
}

// resizeNFSVolume resizes an NFS volume
func (m *StorageManager) resizeNFSVolume(volume *Volume, newSizeMB int) error {
	// In a real implementation, this would resize the volume on the NFS server
	log.Printf("NFS volume would be resized: %s to %d MB", volume.Name, newSizeMB)
	return nil
}

// resizeBlockVolume resizes a block volume
func (m *StorageManager) resizeBlockVolume(volume *Volume, newSizeMB int) error {
	// In a real implementation, this would resize the block device
	log.Printf("Block volume would be resized: %s to %d MB", volume.Name, newSizeMB)
	return nil
}

// resizeCephVolume resizes a Ceph volume
func (m *StorageManager) resizeCephVolume(volume *Volume, newSizeMB int) error {
	// In a real implementation, this would resize the Ceph RBD volume
	log.Printf("Ceph volume would be resized: %s to %d MB", volume.Name, newSizeMB)
	return nil
}

// loadVolumes loads existing volumes from disk
func (m *StorageManager) loadVolumes() error {
	// For local volumes, scan the volume directory
	if m.config.LocalVolumePath != "" {
		files, err := os.ReadDir(m.config.LocalVolumePath)
		if err != nil {
			return fmt.Errorf("failed to read volume directory: %w", err)
		}

		for _, file := range files {
			// Skip directories and non-volume files
			if file.IsDir() || filepath.Ext(file.Name()) == ".meta" {
				continue
			}

			// Check if there's a metadata file
			metadataPath := filepath.Join(m.config.LocalVolumePath, file.Name()+".meta")
			if _, err := os.Stat(metadataPath); err != nil {
				continue // No metadata, skip this file
			}

			// Read the metadata
			metadataBytes, err := os.ReadFile(metadataPath)
			if err != nil {
				log.Printf("Warning: Failed to read volume metadata %s: %v", metadataPath, err)
				continue
			}

			// Parse the metadata
			var volume Volume
			if err := json.Unmarshal(metadataBytes, &volume); err != nil {
				log.Printf("Warning: Failed to parse volume metadata %s: %v", metadataPath, err)
				continue
			}

			// Add the volume to our maps
			m.volumesMutex.Lock()
			m.volumes[volume.ID] = &volume
			m.volumesByName[volume.Name] = volume.ID
			m.volumesMutex.Unlock()

			log.Printf("Loaded volume %s (ID: %s) from disk", volume.Name, volume.ID)
		}
	}

	return nil
}

// updateVolumes periodically updates volume information
func (m *StorageManager) updateVolumes() {
	ticker := time.NewTicker(m.config.UpdateInterval)
	defer ticker.Stop()

	for {
		select {
		case <-m.ctx.Done():
			return
		case <-ticker.C:
			m.updateVolumeInfo()
		}
	}
}

// updateVolumeInfo updates information about all volumes
func (m *StorageManager) updateVolumeInfo() {
	m.volumesMutex.Lock()
	defer m.volumesMutex.Unlock()

	for _, volume := range m.volumes {
		// Update volume information based on type
		switch volume.Type {
		case VolumeTypeLocal:
			m.updateLocalVolumeInfo(volume)
		case VolumeTypeNFS:
			m.updateNFSVolumeInfo(volume)
		case VolumeTypeBlock:
			m.updateBlockVolumeInfo(volume)
		case VolumeTypeCeph:
			m.updateCephVolumeInfo(volume)
		}

		volume.VolumeInfo.LastUpdated = time.Now()
	}
}

// updateLocalVolumeInfo updates information about a local volume
func (m *StorageManager) updateLocalVolumeInfo(volume *Volume) {
	// Check if the volume file exists
	if _, err := os.Stat(volume.Path); os.IsNotExist(err) {
		volume.VolumeInfo.Available = false
		return
	}

	// Get volume size information
	info, err := os.Stat(volume.Path)
	if err != nil {
		log.Printf("Warning: Failed to get volume info for %s: %v", volume.Path, err)
		return
	}

	// Update used space (for a real implementation, this would be more accurate)
	volume.VolumeInfo.UsedMB = int(info.Size() / (1024 * 1024))
	volume.VolumeInfo.Available = true
}

// updateNFSVolumeInfo updates information about an NFS volume
func (m *StorageManager) updateNFSVolumeInfo(volume *Volume) {
	// In a real implementation, this would get information from the NFS server
	volume.VolumeInfo.Available = true
	volume.VolumeInfo.UsedMB = 0 // Placeholder
}

// updateBlockVolumeInfo updates information about a block volume
func (m *StorageManager) updateBlockVolumeInfo(volume *Volume) {
	// In a real implementation, this would get information about the block device
	volume.VolumeInfo.Available = true
	volume.VolumeInfo.UsedMB = 0 // Placeholder
}

// updateCephVolumeInfo updates information about a Ceph volume
func (m *StorageManager) updateCephVolumeInfo(volume *Volume) {
	// In a real implementation, this would get information from Ceph
	volume.VolumeInfo.Available = true
	volume.VolumeInfo.UsedMB = 0 // Placeholder
}

// emitEvent emits a volume event to all listeners
func (m *StorageManager) emitEvent(event VolumeEvent) {
	m.eventMutex.RLock()
	defer m.eventMutex.RUnlock()

	for _, listener := range m.eventListeners {
		go listener(event)
	}
}
