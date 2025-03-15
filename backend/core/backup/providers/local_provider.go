package providers

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/backup"
)

// LocalStorageProvider provides backup storage on the local filesystem
type LocalStorageProvider struct {
	// ID is the unique identifier for this provider
	id string

	// Name is a human-readable name for this provider
	name string

	// BaseDir is the base directory where backups are stored
	baseDir string

	// Mutex protects concurrent access
	mutex sync.RWMutex

	// metadata stores backup metadata
	metadata map[string]*backup.Backup
}

// NewLocalStorageProvider creates a new local storage provider
func NewLocalStorageProvider(id, name, baseDir string) (*LocalStorageProvider, error) {
	// Create base directory if it doesn't exist
	if err := os.MkdirAll(baseDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create base directory: %w", err)
	}

	// Create metadata directory
	metadataDir := filepath.Join(baseDir, "metadata")
	if err := os.MkdirAll(metadataDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create metadata directory: %w", err)
	}

	// Create data directory
	dataDir := filepath.Join(baseDir, "data")
	if err := os.MkdirAll(dataDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create data directory: %w", err)
	}

	provider := &LocalStorageProvider{
		id:       id,
		name:     name,
		baseDir:  baseDir,
		metadata: make(map[string]*backup.Backup),
	}

	// Load existing metadata
	if err := provider.loadMetadata(); err != nil {
		return nil, fmt.Errorf("failed to load metadata: %w", err)
	}

	return provider, nil
}

// ID returns the provider ID
func (p *LocalStorageProvider) ID() string {
	return p.id
}

// Name returns the provider name
func (p *LocalStorageProvider) Name() string {
	return p.name
}

// Type returns the type of storage this provider supports
func (p *LocalStorageProvider) Type() backup.StorageType {
	return backup.LocalStorage
}

// CreateBackup creates a backup
func (p *LocalStorageProvider) CreateBackup(ctx context.Context, job *backup.BackupJob) (*backup.Backup, error) {
	p.mutex.Lock()
	defer p.mutex.Unlock()

	// Create a new backup
	backupID := fmt.Sprintf("backup-%d", time.Now().UnixNano())
	backupDir := filepath.Join(p.baseDir, "data", backupID)
	if err := os.MkdirAll(backupDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create backup directory: %w", err)
	}

	// Create backup metadata
	b := &backup.Backup{
		ID:              backupID,
		JobID:           job.ID,
		Type:            job.Type,
		State:           backup.BackupInProgress,
		StartedAt:       time.Now(),
		StorageLocation: backupDir,
		TenantID:        job.TenantID,
		TargetMetadata:  make(map[string]map[string]string),
	}

	// Create a metadata file for each target
	for _, target := range job.Targets {
		targetDir := filepath.Join(backupDir, target.ID)
		if err := os.MkdirAll(targetDir, 0755); err != nil {
			return nil, fmt.Errorf("failed to create target directory: %w", err)
		}

		// Save target metadata
		targetMetadata := map[string]string{
			"name":        target.Name,
			"type":        target.Type,
			"resource_id": target.ResourceID,
			"status":      "pending",
		}

		// If there are additional metadata, add them
		if target.Metadata != nil {
			for k, v := range target.Metadata {
				targetMetadata[k] = v
			}
		}

		// Add to backup metadata
		b.TargetMetadata[target.ID] = targetMetadata

		// Write target metadata to disk
		targetMetadataFile := filepath.Join(targetDir, "metadata.json")
		targetMetadataBytes, err := json.MarshalIndent(targetMetadata, "", "  ")
		if err != nil {
			return nil, fmt.Errorf("failed to marshal target metadata: %w", err)
		}
		if err := ioutil.WriteFile(targetMetadataFile, targetMetadataBytes, 0644); err != nil {
			return nil, fmt.Errorf("failed to write target metadata: %w", err)
		}
	}

	// Save backup metadata
	if err := p.saveBackupMetadata(b); err != nil {
		return nil, fmt.Errorf("failed to save backup metadata: %w", err)
	}

	// Add to in-memory cache
	p.metadata[backupID] = b

	// For the sake of the implementation, we'll assume the backup is complete immediately
	// In a real implementation, this would happen asynchronously as data is copied
	b.State = backup.BackupCompleted
	b.CompletedAt = time.Now()
	b.Size = p.calculateBackupSize(backupDir)

	// Update target metadata to show completion
	for targetID := range b.TargetMetadata {
		b.TargetMetadata[targetID]["status"] = "completed"
	}

	// Save updated metadata
	if err := p.saveBackupMetadata(b); err != nil {
		return nil, fmt.Errorf("failed to save updated backup metadata: %w", err)
	}

	return b, nil
}

// DeleteBackup deletes a backup
func (p *LocalStorageProvider) DeleteBackup(ctx context.Context, backupID string) error {
	p.mutex.Lock()
	defer p.mutex.Unlock()

	// Check if backup exists
	b, exists := p.metadata[backupID]
	if !exists {
		return fmt.Errorf("backup with ID %s does not exist", backupID)
	}

	// Delete backup directory
	backupDir := b.StorageLocation
	if err := os.RemoveAll(backupDir); err != nil {
		return fmt.Errorf("failed to delete backup directory: %w", err)
	}

	// Delete metadata file
	metadataFile := filepath.Join(p.baseDir, "metadata", fmt.Sprintf("%s.json", backupID))
	if err := os.Remove(metadataFile); err != nil {
		return fmt.Errorf("failed to delete backup metadata: %w", err)
	}

	// Remove from in-memory cache
	delete(p.metadata, backupID)

	return nil
}

// RestoreBackup restores a backup
func (p *LocalStorageProvider) RestoreBackup(ctx context.Context, job *backup.RestoreJob) error {
	p.mutex.RLock()
	defer p.mutex.RUnlock()

	// Check if backup exists
	b, exists := p.metadata[job.BackupID]
	if !exists {
		return fmt.Errorf("backup with ID %s does not exist", job.BackupID)
	}

	// Verify backup is complete
	if b.State != backup.BackupCompleted {
		return fmt.Errorf("backup with ID %s is not in completed state", job.BackupID)
	}

	// In a real implementation, we would restore the backup based on the restore targets
	// For simplicity, we'll just update the restore job status
	for _, target := range job.Targets {
		// Check if source target exists in the backup
		if _, exists := b.TargetMetadata[target.SourceID]; !exists {
			return fmt.Errorf("source target %s does not exist in backup", target.SourceID)
		}

		// Update target state
		target.State = backup.RestoreCompleted
	}

	return nil
}

// ListBackups lists backups
func (p *LocalStorageProvider) ListBackups(ctx context.Context, filter map[string]interface{}) ([]*backup.Backup, error) {
	p.mutex.RLock()
	defer p.mutex.RUnlock()

	// Apply filters (if any)
	results := make([]*backup.Backup, 0, len(p.metadata))
	for _, b := range p.metadata {
		include := true

		// Apply tenant filter
		if tenantID, ok := filter["tenant_id"].(string); ok && tenantID != "" {
			if b.TenantID != tenantID {
				include = false
			}
		}

		// Apply job filter
		if jobID, ok := filter["job_id"].(string); ok && jobID != "" {
			if b.JobID != jobID {
				include = false
			}
		}

		// Apply state filter
		if state, ok := filter["state"].(backup.BackupState); ok && state != "" {
			if b.State != state {
				include = false
			}
		}

		// Apply type filter
		if backupType, ok := filter["type"].(backup.BackupType); ok && backupType != "" {
			if b.Type != backupType {
				include = false
			}
		}

		// Include if all filters passed
		if include {
			results = append(results, b)
		}
	}

	return results, nil
}

// GetBackup gets a backup by ID
func (p *LocalStorageProvider) GetBackup(ctx context.Context, backupID string) (*backup.Backup, error) {
	p.mutex.RLock()
	defer p.mutex.RUnlock()

	// Check if backup exists
	b, exists := p.metadata[backupID]
	if !exists {
		return nil, fmt.Errorf("backup with ID %s does not exist", backupID)
	}

	return b, nil
}

// ValidateBackup validates a backup
func (p *LocalStorageProvider) ValidateBackup(ctx context.Context, backupID string) error {
	p.mutex.RLock()
	defer p.mutex.RUnlock()

	// Check if backup exists
	b, exists := p.metadata[backupID]
	if !exists {
		return fmt.Errorf("backup with ID %s does not exist", backupID)
	}

	// Verify backup directory exists
	backupDir := b.StorageLocation
	if _, err := os.Stat(backupDir); os.IsNotExist(err) {
		return fmt.Errorf("backup directory %s does not exist", backupDir)
	}

	// Check each target
	for targetID := range b.TargetMetadata {
		targetDir := filepath.Join(backupDir, targetID)
		if _, err := os.Stat(targetDir); os.IsNotExist(err) {
			return fmt.Errorf("target directory %s does not exist", targetDir)
		}

		// Check metadata file
		targetMetadataFile := filepath.Join(targetDir, "metadata.json")
		if _, err := os.Stat(targetMetadataFile); os.IsNotExist(err) {
			return fmt.Errorf("target metadata file %s does not exist", targetMetadataFile)
		}
	}

	return nil
}

// saveBackupMetadata saves backup metadata to disk
func (p *LocalStorageProvider) saveBackupMetadata(b *backup.Backup) error {
	// Create metadata file
	metadataFile := filepath.Join(p.baseDir, "metadata", fmt.Sprintf("%s.json", b.ID))
	metadataBytes, err := json.MarshalIndent(b, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal backup metadata: %w", err)
	}
	if err := ioutil.WriteFile(metadataFile, metadataBytes, 0644); err != nil {
		return fmt.Errorf("failed to write backup metadata: %w", err)
	}

	return nil
}

// loadMetadata loads existing backup metadata from disk
func (p *LocalStorageProvider) loadMetadata() error {
	// Get metadata directory
	metadataDir := filepath.Join(p.baseDir, "metadata")

	// Read metadata files
	files, err := ioutil.ReadDir(metadataDir)
	if err != nil {
		if os.IsNotExist(err) {
			// Directory doesn't exist, nothing to load
			return nil
		}
		return fmt.Errorf("failed to read metadata directory: %w", err)
	}

	// Process each file
	for _, file := range files {
		if file.IsDir() || !strings.HasSuffix(file.Name(), ".json") {
			continue
		}

		// Read metadata file
		metadataFile := filepath.Join(metadataDir, file.Name())
		metadataBytes, err := ioutil.ReadFile(metadataFile)
		if err != nil {
			return fmt.Errorf("failed to read metadata file %s: %w", metadataFile, err)
		}

		// Unmarshal metadata
		var b backup.Backup
		if err := json.Unmarshal(metadataBytes, &b); err != nil {
			return fmt.Errorf("failed to unmarshal metadata from %s: %w", metadataFile, err)
		}

		// Add to in-memory cache
		p.metadata[b.ID] = &b
	}

	return nil
}

// calculateBackupSize calculates the size of a backup
func (p *LocalStorageProvider) calculateBackupSize(backupDir string) int64 {
	var size int64
	filepath.Walk(backupDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return nil
		}
		if !info.IsDir() {
			size += info.Size()
		}
		return nil
	})
	return size
}

// BackupSession represents an active backup operation
type BackupSession struct {
	// provider is the parent provider
	provider *LocalStorageProvider

	// backup is the backup being created
	backup *backup.Backup

	// targetDir is the current target directory
	targetDir string

	// currentTarget is the current target being backed up
	currentTarget string
}

// NewBackupSession creates a new backup session
func (p *LocalStorageProvider) NewBackupSession(b *backup.Backup, targetID string) (*BackupSession, error) {
	// Create target directory
	targetDir := filepath.Join(b.StorageLocation, targetID)
	if err := os.MkdirAll(targetDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create target directory: %w", err)
	}

	return &BackupSession{
		provider:      p,
		backup:        b,
		targetDir:     targetDir,
		currentTarget: targetID,
	}, nil
}

// WriteFile writes a file to the backup
func (s *BackupSession) WriteFile(filename string, data []byte) error {
	// Create subdirectories if needed
	filePath := filepath.Join(s.targetDir, filename)
	if err := os.MkdirAll(filepath.Dir(filePath), 0755); err != nil {
		return fmt.Errorf("failed to create directories for %s: %w", filePath, err)
	}

	// Write file
	if err := ioutil.WriteFile(filePath, data, 0644); err != nil {
		return fmt.Errorf("failed to write file %s: %w", filePath, err)
	}

	return nil
}

// CopyFile copies a file from a source path to the backup
func (s *BackupSession) CopyFile(sourcePath, destFilename string) error {
	// Open source file
	sourceFile, err := os.Open(sourcePath)
	if err != nil {
		return fmt.Errorf("failed to open source file %s: %w", sourcePath, err)
	}
	defer sourceFile.Close()

	// Create subdirectories if needed
	destPath := filepath.Join(s.targetDir, destFilename)
	if err := os.MkdirAll(filepath.Dir(destPath), 0755); err != nil {
		return fmt.Errorf("failed to create directories for %s: %w", destPath, err)
	}

	// Create destination file
	destFile, err := os.Create(destPath)
	if err != nil {
		return fmt.Errorf("failed to create destination file %s: %w", destPath, err)
	}
	defer destFile.Close()

	// Copy file contents
	_, err = io.Copy(destFile, sourceFile)
	if err != nil {
		return fmt.Errorf("failed to copy file from %s to %s: %w", sourcePath, destPath, err)
	}

	return nil
}

// Complete marks the backup session as complete
func (s *BackupSession) Complete() error {
	// Update target metadata
	s.provider.mutex.Lock()
	defer s.provider.mutex.Unlock()

	if targetMetadata, exists := s.backup.TargetMetadata[s.currentTarget]; exists {
		targetMetadata["status"] = "completed"
	}

	// Check if all targets are completed
	allCompleted := true
	for _, targetMetadata := range s.backup.TargetMetadata {
		if targetMetadata["status"] != "completed" {
			allCompleted = false
			break
		}
	}

	// If all targets are complete, mark the backup as complete
	if allCompleted {
		s.backup.State = backup.BackupCompleted
		s.backup.CompletedAt = time.Now()
		s.backup.Size = s.provider.calculateBackupSize(s.backup.StorageLocation)
	}

	// Save backup metadata
	if err := s.provider.saveBackupMetadata(s.backup); err != nil {
		return fmt.Errorf("failed to save backup metadata: %w", err)
	}

	return nil
}
