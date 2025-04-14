package vm

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/google/uuid"
)

// BackupType represents the type of backup
type BackupType string

const (
	// BackupTypeFull represents a full backup
	BackupTypeFull BackupType = "full"
	
	// BackupTypeIncremental represents an incremental backup
	BackupTypeIncremental BackupType = "incremental"
	
	// BackupTypeDifferential represents a differential backup
	BackupTypeDifferential BackupType = "differential"
)

// BackupStatus represents the status of a backup
type BackupStatus string

const (
	// BackupStatusCreating indicates the backup is being created
	BackupStatusCreating BackupStatus = "creating"
	
	// BackupStatusCompleted indicates the backup completed successfully
	BackupStatusCompleted BackupStatus = "completed"
	
	// BackupStatusFailed indicates the backup failed
	BackupStatusFailed BackupStatus = "failed"
	
	// BackupStatusRestoring indicates the backup is being restored
	BackupStatusRestoring BackupStatus = "restoring"
	
	// BackupStatusDeleting indicates the backup is being deleted
	BackupStatusDeleting BackupStatus = "deleting"
	
	// BackupStatusDeleted indicates the backup has been deleted
	BackupStatusDeleted BackupStatus = "deleted"
)

// VMBackup represents a VM backup
type VMBackup struct {
	ID          string       `json:"id"`
	VMID        string       `json:"vm_id"`
	Name        string       `json:"name"`
	Description string       `json:"description"`
	Type        BackupType   `json:"type"`
	Status      BackupStatus `json:"status"`
	CreatedAt   time.Time    `json:"created_at"`
	UpdatedAt   time.Time    `json:"updated_at"`
	Size        int64        `json:"size"`
	ParentID    string       `json:"parent_id,omitempty"`
	Tags        []string     `json:"tags,omitempty"`
	Metadata    map[string]string `json:"metadata,omitempty"`
	StoragePath string       `json:"storage_path"`
	ExpiresAt   *time.Time   `json:"expires_at,omitempty"`
}

// BackupStorageProvider is an interface for backup storage providers
type BackupStorageProvider interface {
	// Store stores a backup
	Store(ctx context.Context, vmID, backupID string, data []byte) (string, error)
	
	// Retrieve retrieves a backup
	Retrieve(ctx context.Context, storagePath string) ([]byte, error)
	
	// Delete deletes a backup
	Delete(ctx context.Context, storagePath string) error
	
	// List lists all backups for a VM
	List(ctx context.Context, vmID string) ([]string, error)
}

// LocalBackupStorageProvider is a local backup storage provider
type LocalBackupStorageProvider struct {
	storageDir string
}

// NewLocalBackupStorageProvider creates a new local backup storage provider
func NewLocalBackupStorageProvider(storageDir string) *LocalBackupStorageProvider {
	return &LocalBackupStorageProvider{
		storageDir: storageDir,
	}
}

// Store stores a backup
func (p *LocalBackupStorageProvider) Store(ctx context.Context, vmID, backupID string, data []byte) (string, error) {
	// Create backup directory
	backupDir := filepath.Join(p.storageDir, "backups", vmID)
	if err := os.MkdirAll(backupDir, 0755); err != nil {
		return "", fmt.Errorf("failed to create backup directory: %w", err)
	}
	
	// Create backup file
	backupFile := filepath.Join(backupDir, backupID+".backup")
	if err := os.WriteFile(backupFile, data, 0644); err != nil {
		return "", fmt.Errorf("failed to write backup file: %w", err)
	}
	
	return backupFile, nil
}

// Retrieve retrieves a backup
func (p *LocalBackupStorageProvider) Retrieve(ctx context.Context, storagePath string) ([]byte, error) {
	return os.ReadFile(storagePath)
}

// Delete deletes a backup
func (p *LocalBackupStorageProvider) Delete(ctx context.Context, storagePath string) error {
	return os.Remove(storagePath)
}

// List lists all backups for a VM
func (p *LocalBackupStorageProvider) List(ctx context.Context, vmID string) ([]string, error) {
	backupDir := filepath.Join(p.storageDir, "backups", vmID)
	
	// Check if directory exists
	if _, err := os.Stat(backupDir); os.IsNotExist(err) {
		return []string{}, nil
	}
	
	// List files in directory
	files, err := os.ReadDir(backupDir)
	if err != nil {
		return nil, fmt.Errorf("failed to list backup directory: %w", err)
	}
	
	// Filter backup files
	backups := make([]string, 0, len(files))
	for _, file := range files {
		if !file.IsDir() && filepath.Ext(file.Name()) == ".backup" {
			backups = append(backups, filepath.Join(backupDir, file.Name()))
		}
	}
	
	return backups, nil
}

// VMBackupManager manages VM backups
type VMBackupManager struct {
	backups       map[string]*VMBackup
	backupsMutex  sync.RWMutex
	vmManager     *VMManager
	storageProvider BackupStorageProvider
}

// NewVMBackupManager creates a new VM backup manager
func NewVMBackupManager(vmManager *VMManager, storageProvider BackupStorageProvider) *VMBackupManager {
	return &VMBackupManager{
		backups:        make(map[string]*VMBackup),
		vmManager:      vmManager,
		storageProvider: storageProvider,
	}
}

// CreateBackup creates a new VM backup
func (m *VMBackupManager) CreateBackup(ctx context.Context, vmID, name, description string, backupType BackupType, tags []string, metadata map[string]string, expiresIn *time.Duration) (*VMBackup, error) {
	// Get the VM
	vm, err := m.vmManager.GetVM(vmID)
	if err != nil {
		return nil, fmt.Errorf("failed to get VM: %w", err)
	}
	
	// Generate a unique ID for the backup
	backupID := uuid.New().String()
	
	// Set expiration time if provided
	var expiresAt *time.Time
	if expiresIn != nil {
		t := time.Now().Add(*expiresIn)
		expiresAt = &t
	}
	
	// Create the backup
	backup := &VMBackup{
		ID:          backupID,
		VMID:        vmID,
		Name:        name,
		Description: description,
		Type:        backupType,
		Status:      BackupStatusCreating,
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
		Tags:        tags,
		Metadata:    metadata,
		ExpiresAt:   expiresAt,
	}
	
	// Store the backup
	m.backupsMutex.Lock()
	m.backups[backupID] = backup
	m.backupsMutex.Unlock()
	
	// Create the backup in a goroutine
	go func() {
		err := m.createBackupFiles(ctx, vm, backup)
		
		m.backupsMutex.Lock()
		defer m.backupsMutex.Unlock()
		
		if err != nil {
			backup.Status = BackupStatusFailed
			backup.UpdatedAt = time.Now()
			backup.Metadata["error"] = err.Error()
			log.Printf("Failed to create backup %s for VM %s: %v", backupID, vmID, err)
			return
		}
		
		backup.Status = BackupStatusCompleted
		backup.UpdatedAt = time.Now()
		log.Printf("Created backup %s for VM %s", backupID, vmID)
	}()
	
	return backup, nil
}

// GetBackup returns a backup by ID
func (m *VMBackupManager) GetBackup(backupID string) (*VMBackup, error) {
	m.backupsMutex.RLock()
	defer m.backupsMutex.RUnlock()
	
	backup, exists := m.backups[backupID]
	if !exists {
		return nil, fmt.Errorf("backup %s not found", backupID)
	}
	
	return backup, nil
}

// ListBackups returns all backups
func (m *VMBackupManager) ListBackups() []*VMBackup {
	m.backupsMutex.RLock()
	defer m.backupsMutex.RUnlock()
	
	backups := make([]*VMBackup, 0, len(m.backups))
	for _, backup := range m.backups {
		backups = append(backups, backup)
	}
	
	return backups
}

// ListBackupsForVM returns all backups for a VM
func (m *VMBackupManager) ListBackupsForVM(vmID string) []*VMBackup {
	m.backupsMutex.RLock()
	defer m.backupsMutex.RUnlock()
	
	backups := make([]*VMBackup, 0)
	for _, backup := range m.backups {
		if backup.VMID == vmID {
			backups = append(backups, backup)
		}
	}
	
	return backups
}

// DeleteBackup deletes a backup
func (m *VMBackupManager) DeleteBackup(ctx context.Context, backupID string) error {
	// Get the backup
	m.backupsMutex.Lock()
	backup, exists := m.backups[backupID]
	if !exists {
		m.backupsMutex.Unlock()
		return fmt.Errorf("backup %s not found", backupID)
	}
	
	// Update backup status
	backup.Status = BackupStatusDeleting
	backup.UpdatedAt = time.Now()
	m.backupsMutex.Unlock()
	
	// Delete the backup files
	err := m.storageProvider.Delete(ctx, backup.StoragePath)
	
	m.backupsMutex.Lock()
	defer m.backupsMutex.Unlock()
	
	if err != nil {
		backup.Status = BackupStatusFailed
		backup.UpdatedAt = time.Now()
		backup.Metadata["error"] = err.Error()
		return fmt.Errorf("failed to delete backup files: %w", err)
	}
	
	// Remove the backup from the manager
	delete(m.backups, backupID)
	
	return nil
}

// RestoreBackup restores a VM from a backup
func (m *VMBackupManager) RestoreBackup(ctx context.Context, backupID string) error {
	// Get the backup
	m.backupsMutex.RLock()
	backup, exists := m.backups[backupID]
	if !exists {
		m.backupsMutex.RUnlock()
		return fmt.Errorf("backup %s not found", backupID)
	}
	
	// Update backup status
	backup.Status = BackupStatusRestoring
	backup.UpdatedAt = time.Now()
	m.backupsMutex.RUnlock()
	
	// Get the VM
	vm, err := m.vmManager.GetVM(backup.VMID)
	if err != nil {
		m.backupsMutex.Lock()
		backup.Status = BackupStatusFailed
		backup.UpdatedAt = time.Now()
		backup.Metadata["error"] = err.Error()
		m.backupsMutex.Unlock()
		return fmt.Errorf("failed to get VM: %w", err)
	}
	
	// Check if the VM is running
	if vm.State() == StateRunning {
		m.backupsMutex.Lock()
		backup.Status = BackupStatusFailed
		backup.UpdatedAt = time.Now()
		backup.Metadata["error"] = "VM must be stopped before restoring a backup"
		m.backupsMutex.Unlock()
		return fmt.Errorf("VM must be stopped before restoring a backup")
	}
	
	// Restore the backup
	err = m.restoreBackupFiles(ctx, vm, backup)
	
	m.backupsMutex.Lock()
	defer m.backupsMutex.Unlock()
	
	if err != nil {
		backup.Status = BackupStatusFailed
		backup.UpdatedAt = time.Now()
		backup.Metadata["error"] = err.Error()
		return fmt.Errorf("failed to restore backup: %w", err)
	}
	
	backup.Status = BackupStatusCompleted
	backup.UpdatedAt = time.Now()
	
	return nil
}

// createBackupFiles creates the backup files
func (m *VMBackupManager) createBackupFiles(ctx context.Context, vm *VM, backup *VMBackup) error {
	// In a real implementation, this would:
	// 1. Create a backup of the VM's disk
	// 2. Create a backup of the VM's configuration
	// 3. Store the backup in the storage provider
	
	// For simplicity, we'll just create a placeholder backup
	backupData := map[string]interface{}{
		"vm_id":      vm.ID(),
		"vm_name":    vm.Name(),
		"vm_state":   vm.State(),
		"backup_id":  backup.ID,
		"created_at": backup.CreatedAt,
		"type":       backup.Type,
		"metadata":   backup.Metadata,
	}
	
	// Marshal backup data
	backupJSON, err := json.Marshal(backupData)
	if err != nil {
		return fmt.Errorf("failed to marshal backup data: %w", err)
	}
	
	// Store backup in storage provider
	storagePath, err := m.storageProvider.Store(ctx, vm.ID(), backup.ID, backupJSON)
	if err != nil {
		return fmt.Errorf("failed to store backup: %w", err)
	}
	
	// Update backup
	backup.StoragePath = storagePath
	backup.Size = int64(len(backupJSON))
	
	return nil
}

// restoreBackupFiles restores a VM from a backup
func (m *VMBackupManager) restoreBackupFiles(ctx context.Context, vm *VM, backup *VMBackup) error {
	// In a real implementation, this would:
	// 1. Retrieve the backup from the storage provider
	// 2. Restore the VM's disk from the backup
	// 3. Restore the VM's configuration from the backup
	
	// For simplicity, we'll just log the restore operation
	log.Printf("Restoring VM %s from backup %s", vm.ID(), backup.ID)
	
	// Retrieve backup from storage provider
	_, err := m.storageProvider.Retrieve(ctx, backup.StoragePath)
	if err != nil {
		return fmt.Errorf("failed to retrieve backup: %w", err)
	}
	
	// In a real implementation, we would restore the VM's disk and configuration
	
	return nil
}

// CleanupExpiredBackups deletes expired backups
func (m *VMBackupManager) CleanupExpiredBackups(ctx context.Context) error {
	m.backupsMutex.Lock()
	defer m.backupsMutex.Unlock()
	
	now := time.Now()
	expiredBackups := make([]string, 0)
	
	// Find expired backups
	for id, backup := range m.backups {
		if backup.ExpiresAt != nil && now.After(*backup.ExpiresAt) {
			expiredBackups = append(expiredBackups, id)
		}
	}
	
	// Delete expired backups
	for _, id := range expiredBackups {
		backup := m.backups[id]
		
		// Delete backup files
		if err := m.storageProvider.Delete(ctx, backup.StoragePath); err != nil {
			log.Printf("Warning: Failed to delete expired backup %s: %v", id, err)
			continue
		}
		
		// Remove backup from manager
		delete(m.backups, id)
		
		log.Printf("Deleted expired backup %s for VM %s", id, backup.VMID)
	}
	
	return nil
}
