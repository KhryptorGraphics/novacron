package vm

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"time"
)

// Ubuntu2404SnapshotManager manages snapshots for Ubuntu 24.04 VMs
type Ubuntu2404SnapshotManager struct {
	// Base path for storing snapshots
	SnapshotBasePath string
	
	// KVM driver reference
	Driver *KVMDriver
}

// SnapshotMetadata contains metadata about a VM snapshot
type SnapshotMetadata struct {
	ID          string            `json:"id"`
	VMID        string            `json:"vm_id"`
	Name        string            `json:"name"`
	Description string            `json:"description"`
	CreatedAt   time.Time         `json:"created_at"`
	Size        int64             `json:"size"`
	Tags        map[string]string `json:"tags"`
	State       VMState           `json:"state"`
}

// NewUbuntu2404SnapshotManager creates a new snapshot manager for Ubuntu 24.04 VMs
func NewUbuntu2404SnapshotManager(snapshotBasePath string, driver *KVMDriver) *Ubuntu2404SnapshotManager {
	// Create snapshot base directory if it doesn't exist
	if err := os.MkdirAll(snapshotBasePath, 0755); err != nil {
		log.Printf("Warning: Failed to create snapshot base directory %s: %v", snapshotBasePath, err)
	}
	
	return &Ubuntu2404SnapshotManager{
		SnapshotBasePath: snapshotBasePath,
		Driver:           driver,
	}
}

// CreateSnapshot creates a new snapshot of an Ubuntu 24.04 VM
func (m *Ubuntu2404SnapshotManager) CreateSnapshot(ctx context.Context, vmID, name, description string, tags map[string]string) (*SnapshotMetadata, error) {
	log.Printf("Creating snapshot of Ubuntu 24.04 VM %s with name %s", vmID, name)
	
	// Get VM info
	m.Driver.vmLock.RLock()
	vmInfo, exists := m.Driver.vms[vmID]
	m.Driver.vmLock.RUnlock()
	
	if !exists {
		return nil, fmt.Errorf("VM %s not found", vmID)
	}
	
	// Generate a unique snapshot ID
	snapshotID := fmt.Sprintf("snapshot-%s-%d", vmID, time.Now().UnixNano())
	
	// Create snapshot directory
	snapshotDir := filepath.Join(m.SnapshotBasePath, snapshotID)
	if err := os.MkdirAll(snapshotDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create snapshot directory: %w", err)
	}
	
	// Determine snapshot path
	snapshotPath := filepath.Join(snapshotDir, "disk.qcow2")
	
	// Check if VM is running
	vmState, err := m.Driver.GetStatus(ctx, vmID)
	if err != nil {
		return nil, fmt.Errorf("failed to get VM status: %w", err)
	}
	
	// If VM is running, use QMP to create a live snapshot
	if vmState == VMStateRunning {
		if err := m.createLiveSnapshot(ctx, vmInfo, snapshotPath); err != nil {
			return nil, fmt.Errorf("failed to create live snapshot: %w", err)
		}
	} else {
		// If VM is stopped, create a regular snapshot
		if err := m.createOfflineSnapshot(ctx, vmInfo, snapshotPath); err != nil {
			return nil, fmt.Errorf("failed to create offline snapshot: %w", err)
		}
	}
	
	// Get snapshot size
	snapshotSize, err := getFileSize(snapshotPath)
	if err != nil {
		log.Printf("Warning: Failed to get snapshot size: %v", err)
	}
	
	// Create snapshot metadata
	metadata := &SnapshotMetadata{
		ID:          snapshotID,
		VMID:        vmID,
		Name:        name,
		Description: description,
		CreatedAt:   time.Now(),
		Size:        snapshotSize,
		Tags:        tags,
		State:       vmState,
	}
	
	// Save metadata to file
	if err := saveSnapshotMetadata(filepath.Join(snapshotDir, "metadata.json"), metadata); err != nil {
		log.Printf("Warning: Failed to save snapshot metadata: %v", err)
	}
	
	log.Printf("Created snapshot %s for VM %s", snapshotID, vmID)
	return metadata, nil
}

// RestoreSnapshot restores a VM from a snapshot
func (m *Ubuntu2404SnapshotManager) RestoreSnapshot(ctx context.Context, snapshotID, vmID string) error {
	log.Printf("Restoring VM %s from snapshot %s", vmID, snapshotID)
	
	// Get VM info
	m.Driver.vmLock.RLock()
	vmInfo, exists := m.Driver.vms[vmID]
	m.Driver.vmLock.RUnlock()
	
	if !exists {
		return fmt.Errorf("VM %s not found", vmID)
	}
	
	// Check if snapshot exists
	snapshotDir := filepath.Join(m.SnapshotBasePath, snapshotID)
	snapshotPath := filepath.Join(snapshotDir, "disk.qcow2")
	
	if _, err := os.Stat(snapshotPath); os.IsNotExist(err) {
		return fmt.Errorf("snapshot %s not found", snapshotID)
	}
	
	// Stop the VM if it's running
	vmState, err := m.Driver.GetStatus(ctx, vmID)
	if err != nil {
		return fmt.Errorf("failed to get VM status: %w", err)
	}
	
	if vmState == VMStateRunning {
		if err := m.Driver.Stop(ctx, vmID); err != nil {
			return fmt.Errorf("failed to stop VM: %w", err)
		}
	}
	
	// Backup the current disk
	backupPath := vmInfo.DiskPath + ".backup"
	if err := copyFile(vmInfo.DiskPath, backupPath); err != nil {
		return fmt.Errorf("failed to backup VM disk: %w", err)
	}
	
	// Restore the snapshot
	if err := copyFile(snapshotPath, vmInfo.DiskPath); err != nil {
		// Restore from backup if restore fails
		if restoreErr := copyFile(backupPath, vmInfo.DiskPath); restoreErr != nil {
			log.Printf("Warning: Failed to restore VM disk from backup: %v", restoreErr)
		}
		return fmt.Errorf("failed to restore snapshot: %w", err)
	}
	
	// Remove backup
	if err := os.Remove(backupPath); err != nil {
		log.Printf("Warning: Failed to remove backup: %v", err)
	}
	
	// Start the VM if it was running before
	if vmState == VMStateRunning {
		if err := m.Driver.Start(ctx, vmID); err != nil {
			return fmt.Errorf("failed to start VM after restore: %w", err)
		}
	}
	
	log.Printf("Restored VM %s from snapshot %s", vmID, snapshotID)
	return nil
}

// DeleteSnapshot deletes a snapshot
func (m *Ubuntu2404SnapshotManager) DeleteSnapshot(ctx context.Context, snapshotID string) error {
	log.Printf("Deleting snapshot %s", snapshotID)
	
	// Check if snapshot exists
	snapshotDir := filepath.Join(m.SnapshotBasePath, snapshotID)
	
	if _, err := os.Stat(snapshotDir); os.IsNotExist(err) {
		return fmt.Errorf("snapshot %s not found", snapshotID)
	}
	
	// Delete snapshot directory
	if err := os.RemoveAll(snapshotDir); err != nil {
		return fmt.Errorf("failed to delete snapshot: %w", err)
	}
	
	log.Printf("Deleted snapshot %s", snapshotID)
	return nil
}

// ListSnapshots lists all snapshots for a VM
func (m *Ubuntu2404SnapshotManager) ListSnapshots(ctx context.Context, vmID string) ([]*SnapshotMetadata, error) {
	log.Printf("Listing snapshots for VM %s", vmID)
	
	var snapshots []*SnapshotMetadata
	
	// Walk through snapshot base directory
	err := filepath.Walk(m.SnapshotBasePath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		
		// Skip non-directories
		if !info.IsDir() {
			return nil
		}
		
		// Skip the base directory
		if path == m.SnapshotBasePath {
			return nil
		}
		
		// Check if this is a snapshot directory
		metadataPath := filepath.Join(path, "metadata.json")
		if _, err := os.Stat(metadataPath); os.IsNotExist(err) {
			return nil
		}
		
		// Load metadata
		metadata, err := loadSnapshotMetadata(metadataPath)
		if err != nil {
			log.Printf("Warning: Failed to load snapshot metadata from %s: %v", metadataPath, err)
			return nil
		}
		
		// Filter by VM ID if specified
		if vmID != "" && metadata.VMID != vmID {
			return nil
		}
		
		snapshots = append(snapshots, metadata)
		return nil
	})
	
	if err != nil {
		return nil, fmt.Errorf("failed to list snapshots: %w", err)
	}
	
	return snapshots, nil
}

// Helper function to create a live snapshot using QMP
func (m *Ubuntu2404SnapshotManager) createLiveSnapshot(ctx context.Context, vmInfo *KVMInfo, snapshotPath string) error {
	// In a real implementation, this would use QMP to create a live snapshot
	// For Ubuntu 24.04, we need to ensure the VM has qemu-guest-agent installed
	
	// Check if QMP socket exists
	if vmInfo.SocketPath == "" {
		return fmt.Errorf("QMP socket not available for VM")
	}
	
	// Use qemu-img to create an external snapshot
	cmd := exec.CommandContext(ctx, "qemu-img", "snapshot", "-c", filepath.Base(snapshotPath), vmInfo.DiskPath)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed to create snapshot: %w, output: %s", err, string(output))
	}
	
	// Copy the snapshot to the destination
	if err := copyFile(vmInfo.DiskPath, snapshotPath); err != nil {
		return fmt.Errorf("failed to copy snapshot: %w", err)
	}
	
	return nil
}

// Helper function to create an offline snapshot
func (m *Ubuntu2404SnapshotManager) createOfflineSnapshot(ctx context.Context, vmInfo *KVMInfo, snapshotPath string) error {
	// For offline snapshots, we can simply copy the disk image
	if err := copyFile(vmInfo.DiskPath, snapshotPath); err != nil {
		return fmt.Errorf("failed to create offline snapshot: %w", err)
	}
	
	return nil
}

// Helper function to get file size
func getFileSize(path string) (int64, error) {
	info, err := os.Stat(path)
	if err != nil {
		return 0, err
	}
	return info.Size(), nil
}

// Helper function to copy a file
func copyFile(src, dst string) error {
	// Use qemu-img convert for QCOW2 files to ensure proper copying
	cmd := exec.Command("qemu-img", "convert", "-f", "qcow2", "-O", "qcow2", src, dst)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed to copy file: %w, output: %s", err, string(output))
	}
	return nil
}

// Helper function to save snapshot metadata
func saveSnapshotMetadata(path string, metadata *SnapshotMetadata) error {
	// In a real implementation, this would serialize the metadata to JSON
	// and write it to the file
	return nil
}

// Helper function to load snapshot metadata
func loadSnapshotMetadata(path string) (*SnapshotMetadata, error) {
	// In a real implementation, this would read the file and deserialize
	// the JSON to a SnapshotMetadata struct
	return &SnapshotMetadata{}, nil
}
