package storage

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
)



// StorageManager manages storage volumes and operations
type StorageManager struct {
	basePath    string
	volumes     map[string]*VolumeInfo
	mutex       sync.RWMutex
	compression bool
	encryption  bool
	dedup       bool
}

// StorageManagerConfig contains configuration for the storage manager
type StorageManagerConfig struct {
	BasePath    string `json:"base_path"`
	Compression bool   `json:"compression"`
	Encryption  bool   `json:"encryption"`
	Dedup       bool   `json:"deduplication"`
}

// NewStorageManager creates a new storage manager instance
func NewStorageManager(config StorageManagerConfig) (*StorageManager, error) {
	if config.BasePath == "" {
		config.BasePath = "/var/lib/novacron/storage"
	}

	// Ensure base path exists
	if err := os.MkdirAll(config.BasePath, 0755); err != nil {
		return nil, fmt.Errorf("failed to create storage base path: %w", err)
	}

	manager := &StorageManager{
		basePath:    config.BasePath,
		volumes:     make(map[string]*VolumeInfo),
		compression: config.Compression,
		encryption:  config.Encryption,
		dedup:       config.Dedup,
	}

	// Load existing volumes
	if err := manager.loadExistingVolumes(); err != nil {
		log.Printf("Warning: Failed to load existing volumes: %v", err)
	}

	log.Printf("Storage manager initialized at %s", config.BasePath)
	return manager, nil
}

// CreateVolume creates a new storage volume
func (sm *StorageManager) CreateVolume(ctx context.Context, opts VolumeCreateOptions) (*VolumeInfo, error) {
	sm.mutex.Lock()
	defer sm.mutex.Unlock()

	// Check if volume with same name already exists
	for _, vol := range sm.volumes {
		if vol.Name == opts.Name {
			return nil, fmt.Errorf("volume with name %s already exists", opts.Name)
		}
	}

	volumeID := uuid.New().String()
	volumePath := filepath.Join(sm.basePath, fmt.Sprintf("%s.%s", volumeID, opts.Format))

	volume := &VolumeInfo{
		ID:        volumeID,
		Name:      opts.Name,
		Type:      opts.Type,
		Format:    opts.Format,
		SizeMB:    int(opts.Size / (1024 * 1024)), // Convert bytes to MB
		Size:      opts.Size,
		Path:      volumePath,
		Status:    VolumeStatusCreating,
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
		Labels:    opts.Metadata, // Use metadata as labels for now
		Metadata:  make(map[string]string),
	}

	// Store in memory first
	sm.volumes[volumeID] = volume

	// Create the actual volume file
	go func() {
		if err := sm.createVolumeFile(volume); err != nil {
			sm.mutex.Lock()
			volume.Status = VolumeStatusError
			volume.Metadata["error"] = err.Error()
			sm.mutex.Unlock()
			log.Printf("Failed to create volume file for %s: %v", volume.Name, err)
		} else {
			sm.mutex.Lock()
			volume.Status = VolumeStatusAvailable
			volume.UpdatedAt = time.Now()
			sm.mutex.Unlock()
			log.Printf("Successfully created volume %s", volume.Name)
		}
		
		// Save volume metadata
		sm.saveVolumeMetadata(volume)
	}()

	return volume, nil
}

// DeleteVolume deletes a storage volume
func (sm *StorageManager) DeleteVolume(ctx context.Context, volumeID string) error {
	sm.mutex.Lock()
	defer sm.mutex.Unlock()

	volume, exists := sm.volumes[volumeID]
	if !exists {
		return fmt.Errorf("volume %s not found", volumeID)
	}

	if volume.Status == VolumeStatusInUse {
		return fmt.Errorf("cannot delete volume %s: currently in use", volumeID)
	}

	// Mark as deleting
	volume.Status = VolumeStatusDeleting
	volume.UpdatedAt = time.Now()

	// Delete the actual file
	go func() {
		if err := os.Remove(volume.Path); err != nil {
			log.Printf("Failed to delete volume file %s: %v", volume.Path, err)
		}

		// Remove metadata file
		metadataPath := volume.Path + ".meta"
		if err := os.Remove(metadataPath); err != nil {
			log.Printf("Failed to delete volume metadata %s: %v", metadataPath, err)
		}

		sm.mutex.Lock()
		delete(sm.volumes, volumeID)
		sm.mutex.Unlock()

		log.Printf("Successfully deleted volume %s", volume.Name)
	}()

	return nil
}

// GetVolume retrieves a volume by ID
func (sm *StorageManager) GetVolume(ctx context.Context, volumeID string) (*VolumeInfo, error) {
	sm.mutex.RLock()
	defer sm.mutex.RUnlock()

	volume, exists := sm.volumes[volumeID]
	if !exists {
		return nil, fmt.Errorf("volume %s not found", volumeID)
	}

	// Return a copy to prevent external modification
	volumeCopy := *volume
	return &volumeCopy, nil
}

// ListVolumes lists all volumes
func (sm *StorageManager) ListVolumes(ctx context.Context) ([]*VolumeInfo, error) {
	sm.mutex.RLock()
	defer sm.mutex.RUnlock()

	volumes := make([]*VolumeInfo, 0, len(sm.volumes))
	for _, volume := range sm.volumes {
		volumeCopy := *volume
		volumes = append(volumes, &volumeCopy)
	}

	return volumes, nil
}

// AttachVolume attaches a volume to a VM
func (sm *StorageManager) AttachVolume(ctx context.Context, volumeID, vmID string) error {
	sm.mutex.Lock()
	defer sm.mutex.Unlock()

	volume, exists := sm.volumes[volumeID]
	if !exists {
		return fmt.Errorf("volume %s not found", volumeID)
	}

	if volume.Status != VolumeStatusAvailable {
		return fmt.Errorf("volume %s is not available for attachment", volumeID)
	}

	volume.Status = VolumeStatusInUse
	volume.AttachedToVM = vmID
	volume.UpdatedAt = time.Now()

	// Save updated metadata
	sm.saveVolumeMetadata(volume)

	log.Printf("Attached volume %s to VM %s", volume.Name, vmID)
	return nil
}

// DetachVolume detaches a volume from a VM
func (sm *StorageManager) DetachVolume(ctx context.Context, volumeID string) error {
	sm.mutex.Lock()
	defer sm.mutex.Unlock()

	volume, exists := sm.volumes[volumeID]
	if !exists {
		return fmt.Errorf("volume %s not found", volumeID)
	}

	if volume.Status != VolumeStatusInUse {
		return fmt.Errorf("volume %s is not attached", volumeID)
	}

	vmID := volume.AttachedToVM
	volume.Status = VolumeStatusAvailable
	volume.AttachedToVM = ""
	volume.UpdatedAt = time.Now()

	// Save updated metadata
	sm.saveVolumeMetadata(volume)

	log.Printf("Detached volume %s from VM %s", volume.Name, vmID)
	return nil
}

// ResizeVolume resizes a volume
func (sm *StorageManager) ResizeVolume(ctx context.Context, volumeID string, newSizeMB int) error {
	sm.mutex.Lock()
	defer sm.mutex.Unlock()

	volume, exists := sm.volumes[volumeID]
	if !exists {
		return fmt.Errorf("volume %s not found", volumeID)
	}

	if newSizeMB <= volume.SizeMB {
		return fmt.Errorf("new size must be larger than current size")
	}

	oldSize := volume.SizeMB
	volume.SizeMB = newSizeMB
	volume.UpdatedAt = time.Now()

	// Resize the actual file
	go func() {
		if err := sm.resizeVolumeFile(volume, newSizeMB); err != nil {
			sm.mutex.Lock()
			volume.SizeMB = oldSize // Revert on error
			volume.Metadata["resize_error"] = err.Error()
			sm.mutex.Unlock()
			log.Printf("Failed to resize volume %s: %v", volume.Name, err)
		} else {
			log.Printf("Successfully resized volume %s from %dMB to %dMB", volume.Name, oldSize, newSizeMB)
		}
		
		sm.saveVolumeMetadata(volume)
	}()

	return nil
}

// GetVolumeMetrics returns metrics for a volume
func (sm *StorageManager) GetVolumeMetrics(ctx context.Context, volumeID string) (*VolumeMetrics, error) {
	sm.mutex.RLock()
	volume, exists := sm.volumes[volumeID]
	sm.mutex.RUnlock()

	if !exists {
		return nil, fmt.Errorf("volume %s not found", volumeID)
	}

	// Get file stats
	stat, err := os.Stat(volume.Path)
	if err != nil {
		return nil, fmt.Errorf("failed to get volume file stats: %w", err)
	}

	metrics := &VolumeMetrics{
		VolumeID:     volumeID,
		SizeBytes:    stat.Size(),
		UsedBytes:    stat.Size(), // For simplicity, assume fully used
		IOPS:         0,           // Would need monitoring to calculate
		Throughput:   0,           // Would need monitoring to calculate
		LastAccessed: stat.ModTime(),
	}

	return metrics, nil
}

// VolumeMetrics contains performance metrics for a volume
type VolumeMetrics struct {
	VolumeID     string    `json:"volume_id"`
	SizeBytes    int64     `json:"size_bytes"`
	UsedBytes    int64     `json:"used_bytes"`
	IOPS         float64   `json:"iops"`
	Throughput   float64   `json:"throughput_mbps"`
	LastAccessed time.Time `json:"last_accessed"`
}

// --- Private helper methods ---

// createVolumeFile creates the actual volume file
func (sm *StorageManager) createVolumeFile(volume *VolumeInfo) error {
	switch volume.Format {
	case VolumeFormatExt4, VolumeFormatXFS:
		return sm.createFilesystemVolume(volume)
	case VolumeFormatRAW:
		return sm.createRawVolume(volume)
	default:
		return fmt.Errorf("unsupported volume format: %s", volume.Format)
	}
}

// createFilesystemVolume creates a volume with a filesystem
func (sm *StorageManager) createFilesystemVolume(volume *VolumeInfo) error {
	// Create a raw file first
	if err := sm.createRawVolume(volume); err != nil {
		return err
	}

	// Format with filesystem
	var cmd *exec.Cmd
	switch volume.Format {
	case VolumeFormatExt4:
		cmd = exec.Command("mkfs.ext4", "-F", volume.Path)
	case VolumeFormatXFS:
		cmd = exec.Command("mkfs.xfs", "-f", volume.Path)
	default:
		return fmt.Errorf("unsupported filesystem format: %s", volume.Format)
	}

	if err := cmd.Run(); err != nil {
		os.Remove(volume.Path) // Clean up on error
		return fmt.Errorf("failed to format volume: %w", err)
	}

	return nil
}

// createRawVolume creates a raw volume file
func (sm *StorageManager) createRawVolume(volume *VolumeInfo) error {
	file, err := os.Create(volume.Path)
	if err != nil {
		return fmt.Errorf("failed to create volume file: %w", err)
	}
	defer file.Close()

	// Allocate space
	sizeBytes := int64(volume.SizeMB) * 1024 * 1024
	if err := file.Truncate(sizeBytes); err != nil {
		os.Remove(volume.Path) // Clean up on error
		return fmt.Errorf("failed to allocate volume space: %w", err)
	}

	return nil
}

// resizeVolumeFile resizes the volume file
func (sm *StorageManager) resizeVolumeFile(volume *VolumeInfo, newSizeMB int) error {
	file, err := os.OpenFile(volume.Path, os.O_WRONLY, 0644)
	if err != nil {
		return fmt.Errorf("failed to open volume file: %w", err)
	}
	defer file.Close()

	newSizeBytes := int64(newSizeMB) * 1024 * 1024
	if err := file.Truncate(newSizeBytes); err != nil {
		return fmt.Errorf("failed to resize volume file: %w", err)
	}

	return nil
}

// saveVolumeMetadata saves volume metadata to disk
func (sm *StorageManager) saveVolumeMetadata(volume *VolumeInfo) error {
	metadataPath := volume.Path + ".meta"
	
	data, err := json.Marshal(volume)
	if err != nil {
		return fmt.Errorf("failed to marshal volume metadata: %w", err)
	}

	if err := os.WriteFile(metadataPath, data, 0644); err != nil {
		return fmt.Errorf("failed to write volume metadata: %w", err)
	}

	return nil
}

// loadExistingVolumes loads volumes from disk
func (sm *StorageManager) loadExistingVolumes() error {
	entries, err := os.ReadDir(sm.basePath)
	if err != nil {
		return fmt.Errorf("failed to read storage directory: %w", err)
	}

	for _, entry := range entries {
		if !entry.IsDir() && strings.HasSuffix(entry.Name(), ".meta") {
			if err := sm.loadVolumeFromMetadata(entry.Name()); err != nil {
				log.Printf("Failed to load volume from %s: %v", entry.Name(), err)
			}
		}
	}

	log.Printf("Loaded %d existing volumes", len(sm.volumes))
	return nil
}

// loadVolumeFromMetadata loads a single volume from its metadata file
func (sm *StorageManager) loadVolumeFromMetadata(metadataFile string) error {
	metadataPath := filepath.Join(sm.basePath, metadataFile)
	
	data, err := os.ReadFile(metadataPath)
	if err != nil {
		return fmt.Errorf("failed to read metadata file: %w", err)
	}

	var volume VolumeInfo
	if err := json.Unmarshal(data, &volume); err != nil {
		return fmt.Errorf("failed to unmarshal volume metadata: %w", err)
	}

	// Verify volume file still exists
	if _, err := os.Stat(volume.Path); err != nil {
		log.Printf("Volume file %s no longer exists, skipping", volume.Path)
		return nil
	}

	sm.volumes[volume.ID] = &volume
	return nil
}
