package providers

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io/ioutil"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/backup"
)

// IncrementalStorageProvider wraps an existing provider and adds incremental backup support
type IncrementalStorageProvider struct {
	// baseProvider is the underlying provider to wrap
	baseProvider backup.BackupProvider

	// checksumCache stores checksums for files
	checksumCache map[string]string

	// fullBackupInterval is the interval between full backups
	fullBackupInterval time.Duration

	// lastFullBackup is the timestamp of the last full backup
	lastFullBackup time.Time
}

// NewIncrementalStorageProvider creates a new incremental storage provider
func NewIncrementalStorageProvider(baseProvider backup.BackupProvider, fullBackupInterval time.Duration) (*IncrementalStorageProvider, error) {
	return &IncrementalStorageProvider{
		baseProvider:       baseProvider,
		checksumCache:      make(map[string]string),
		fullBackupInterval: fullBackupInterval,
	}, nil
}

// ID returns the provider ID
func (p *IncrementalStorageProvider) ID() string {
	return fmt.Sprintf("incremental-%s", p.baseProvider.ID())
}

// Name returns the provider name
func (p *IncrementalStorageProvider) Name() string {
	return fmt.Sprintf("Incremental %s", p.baseProvider.Name())
}

// Type returns the type of storage this provider supports
func (p *IncrementalStorageProvider) Type() backup.StorageType {
	return p.baseProvider.Type()
}

// CreateBackup creates an incremental backup
func (p *IncrementalStorageProvider) CreateBackup(ctx context.Context, job *backup.BackupJob) (*backup.Backup, error) {
	// Determine if we need a full backup
	needFullBackup := p.needsFullBackup()

	// Set the backup type based on whether we need a full backup
	originalType := job.Type
	if needFullBackup {
		job.Type = backup.FullBackup
	} else if job.Type != backup.IncrementalBackup {
		job.Type = backup.IncrementalBackup
	}

	// Create the backup with the base provider
	b, err := p.baseProvider.CreateBackup(ctx, job)
	if err != nil {
		// Restore original backup type
		job.Type = originalType
		return nil, err
	}

	// Add incremental metadata
	if b.Metadata == nil {
		b.Metadata = make(map[string]string)
	}
	b.Metadata["incremental"] = "true"
	b.Metadata["incremental_type"] = string(job.Type)

	// If this was a full backup, update the last full backup timestamp
	if job.Type == backup.FullBackup {
		p.lastFullBackup = time.Now()
	}

	// Restore original backup type
	job.Type = originalType

	return b, nil
}

// needsFullBackup determines if a full backup is needed
func (p *IncrementalStorageProvider) needsFullBackup() bool {
	// If no full backup has been done, we need one
	if p.lastFullBackup.IsZero() {
		return true
	}

	// If the interval has passed, we need a full backup
	return time.Since(p.lastFullBackup) >= p.fullBackupInterval
}

// DeleteBackup deletes a backup
func (p *IncrementalStorageProvider) DeleteBackup(ctx context.Context, backupID string) error {
	return p.baseProvider.DeleteBackup(ctx, backupID)
}

// RestoreBackup restores a backup
func (p *IncrementalStorageProvider) RestoreBackup(ctx context.Context, job *backup.RestoreJob) error {
	// Get the backup
	b, err := p.GetBackup(ctx, job.BackupID)
	if err != nil {
		return err
	}

	// For incremental backups, we need to locate the parent backup chain
	if b.Type == backup.IncrementalBackup && b.ParentID != "" {
		// In a real implementation, we would restore the parent backup first
		// and then apply the incremental changes
		// For simplicity, we'll just restore the incremental backup directly
	}

	return p.baseProvider.RestoreBackup(ctx, job)
}

// ListBackups lists backups
func (p *IncrementalStorageProvider) ListBackups(ctx context.Context, filter map[string]interface{}) ([]*backup.Backup, error) {
	return p.baseProvider.ListBackups(ctx, filter)
}

// GetBackup gets a backup by ID
func (p *IncrementalStorageProvider) GetBackup(ctx context.Context, backupID string) (*backup.Backup, error) {
	return p.baseProvider.GetBackup(ctx, backupID)
}

// ValidateBackup validates a backup
func (p *IncrementalStorageProvider) ValidateBackup(ctx context.Context, backupID string) error {
	return p.baseProvider.ValidateBackup(ctx, backupID)
}

// IncrementalBackupSession represents an active incremental backup operation
type IncrementalBackupSession struct {
	// baseSession is the underlying session
	baseSession *BackupSession

	// provider is the incremental provider
	provider *IncrementalStorageProvider

	// parent is the parent backup (for incremental backups)
	parent *backup.Backup

	// changedFiles tracks which files have changed
	changedFiles map[string]bool
}

// NewIncrementalBackupSession creates a new incremental backup session
func (p *IncrementalStorageProvider) NewIncrementalBackupSession(b *backup.Backup, targetID string, parent *backup.Backup) (*IncrementalBackupSession, error) {
	// Get the base provider
	baseProvider, ok := p.baseProvider.(*LocalStorageProvider)
	if !ok {
		return nil, fmt.Errorf("base provider must be LocalStorageProvider")
	}

	// Create a base session
	baseSession, err := baseProvider.NewBackupSession(b, targetID)
	if err != nil {
		return nil, err
	}

	return &IncrementalBackupSession{
		baseSession:  baseSession,
		provider:     p,
		parent:       parent,
		changedFiles: make(map[string]bool),
	}, nil
}

// WriteFile writes a file to the backup if it has changed
func (s *IncrementalBackupSession) WriteFile(filename string, data []byte) error {
	// Calculate checksum
	checksum := calculateChecksum(data)

	// Check if file has changed
	hasChanged := true
	if s.parent != nil && s.parent.Type == backup.IncrementalBackup {
		// Check if file exists in the parent backup and has the same checksum
		parentChecksum, ok := s.provider.checksumCache[fmt.Sprintf("%s:%s", s.parent.ID, filename)]
		if ok && parentChecksum == checksum {
			hasChanged = false
		}
	}

	// If the file has changed, write it
	if hasChanged {
		// Write the file
		if err := s.baseSession.WriteFile(filename, data); err != nil {
			return err
		}

		// Mark file as changed
		s.changedFiles[filename] = true

		// Update checksum cache
		s.provider.checksumCache[fmt.Sprintf("%s:%s", s.baseSession.backup.ID, filename)] = checksum
	}

	return nil
}

// CopyFile copies a file from a source path to the backup if it has changed
func (s *IncrementalBackupSession) CopyFile(sourcePath, destFilename string) error {
	// Read the source file
	data, err := ioutil.ReadFile(sourcePath)
	if err != nil {
		return fmt.Errorf("failed to read source file: %w", err)
	}

	// Write the file
	return s.WriteFile(destFilename, data)
}

// Complete marks the backup session as complete
func (s *IncrementalBackupSession) Complete() error {
	// Add a manifest of changed files
	changedFilesList := make([]string, 0, len(s.changedFiles))
	for filename := range s.changedFiles {
		changedFilesList = append(changedFilesList, filename)
	}

	// Write the manifest file
	if err := s.baseSession.WriteFile("__manifest.txt", []byte(fmt.Sprintf("%d\n", len(changedFilesList)))); err != nil {
		return err
	}

	return s.baseSession.Complete()
}

// calculateChecksum calculates a SHA-256 checksum for data
func calculateChecksum(data []byte) string {
	h := sha256.New()
	h.Write(data)
	return hex.EncodeToString(h.Sum(nil))
}
