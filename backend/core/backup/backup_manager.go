package backup

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"
)

// BackupType is defined in types.go

// BackupState represents the state of a backup
type BackupState string

const (
	// BackupPending indicates the backup is pending
	BackupPending BackupState = "pending"

	// BackupInProgress indicates the backup is in progress
	BackupInProgress BackupState = "in_progress"

	// BackupCompleted indicates the backup completed successfully
	BackupCompleted BackupState = "completed"

	// BackupFailed indicates the backup failed
	BackupFailed BackupState = "failed"

	// BackupCancelled indicates the backup was cancelled
	BackupCancelled BackupState = "cancelled"
)

// RestoreState represents the state of a restore operation
type RestoreState string

const (
	// RestorePending indicates the restore is pending
	RestorePending RestoreState = "pending"

	// RestoreInProgress indicates the restore is in progress
	RestoreInProgress RestoreState = "in_progress"

	// RestoreCompleted indicates the restore completed successfully
	RestoreCompleted RestoreState = "completed"

	// RestoreFailed indicates the restore failed
	RestoreFailed RestoreState = "failed"

	// RestoreCancelled indicates the restore was cancelled
	RestoreCancelled RestoreState = "cancelled"
)

// StorageType represents the type of backup storage
type StorageType string

const (
	// LocalStorage represents local storage
	LocalStorage StorageType = "local"

	// ObjectStorage represents object storage (e.g., S3)
	ObjectStorage StorageType = "object"

	// BlockStorage represents block storage (e.g., EBS)
	BlockStorage StorageType = "block"

	// FileStorage represents file storage (e.g., NFS)
	FileStorage StorageType = "file"
)

// BackupTarget represents a target for backup
type BackupTarget struct {
	// ID is the unique identifier of the target
	ID string `json:"id"`

	// Name is the human-readable name of the target
	Name string `json:"name"`

	// Type is the type of target (e.g., VM, volume, database)
	Type string `json:"type"`

	// ResourceID is the ID of the resource to back up
	ResourceID string `json:"resource_id"`

	// TenantID is the ID of the tenant this target belongs to
	TenantID string `json:"tenant_id"`

	// Metadata is additional metadata for the target
	Metadata map[string]string `json:"metadata,omitempty"`
}

// BackupJob represents a backup job
type BackupJob struct {
	// ID is the unique identifier of the job
	ID string `json:"id"`

	// Name is the human-readable name of the job
	Name string `json:"name"`

	// Description is a description of the job
	Description string `json:"description"`

	// Type is the type of backup
	Type BackupType `json:"type"`

	// Targets are the targets to back up
	Targets []*BackupTarget `json:"targets"`

	// Storage is the storage configuration
	Storage *StorageConfig `json:"storage"`

	// Schedule is the schedule for the job
	Schedule *Schedule `json:"schedule"`

	// Retention is the retention policy
	Retention *RetentionPolicy `json:"retention"`

	// Enabled indicates if the job is enabled
	Enabled bool `json:"enabled"`

	// TenantID is the ID of the tenant this job belongs to
	TenantID string `json:"tenant_id"`

	// CreatedAt is when the job was created
	CreatedAt time.Time `json:"created_at"`

	// UpdatedAt is when the job was last updated
	UpdatedAt time.Time `json:"updated_at"`

	// LastRunAt is when the job was last run
	LastRunAt time.Time `json:"last_run_at,omitempty"`

	// LastRunStatus is the status of the last run
	LastRunStatus BackupState `json:"last_run_status,omitempty"`

	// LastSuccessfulRunAt is when the job last completed successfully
	LastSuccessfulRunAt time.Time `json:"last_successful_run_at,omitempty"`

	// NextRunAt is when the job will next run
	NextRunAt time.Time `json:"next_run_at,omitempty"`

	// Metadata is additional metadata for the job
	Metadata map[string]string `json:"metadata,omitempty"`
}

// StorageConfig represents storage configuration for backups
type StorageConfig struct {
	// Type is the type of storage
	Type StorageType `json:"type"`

	// Config is the configuration for the storage
	Config map[string]interface{} `json:"config"`

	// Encryption specifies if backups should be encrypted
	Encryption bool `json:"encryption"`

	// EncryptionKeyID is the ID of the encryption key to use
	EncryptionKeyID string `json:"encryption_key_id,omitempty"`

	// Compression specifies if backups should be compressed
	Compression bool `json:"compression"`

	// CompressionLevel is the compression level
	CompressionLevel int `json:"compression_level"`
}

// Schedule represents a schedule for backup jobs
type Schedule struct {
	// Type is the type of schedule (e.g., cron, interval)
	Type string `json:"type"`

	// Expression is the schedule expression
	Expression string `json:"expression"`

	// StartTime is when the schedule should start
	StartTime time.Time `json:"start_time,omitempty"`

	// EndTime is when the schedule should end
	EndTime time.Time `json:"end_time,omitempty"`

	// TimeZone is the time zone for the schedule
	TimeZone string `json:"time_zone,omitempty"`
}

// RetentionPolicy represents a backup retention policy
type RetentionPolicy struct {
	// KeepLast specifies the number of backups to keep
	KeepLast int `json:"keep_last"`

	// KeepDaily specifies the number of daily backups to keep
	KeepDaily int `json:"keep_daily"`

	// KeepWeekly specifies the number of weekly backups to keep
	KeepWeekly int `json:"keep_weekly"`

	// KeepMonthly specifies the number of monthly backups to keep
	KeepMonthly int `json:"keep_monthly"`

	// KeepYearly specifies the number of yearly backups to keep
	KeepYearly int `json:"keep_yearly"`

	// RetentionDuration is the duration to keep backups
	RetentionDuration time.Duration `json:"retention_duration"`
}

// Backup represents a single backup
type Backup struct {
	// ID is the unique identifier of the backup
	ID string `json:"id"`

	// JobID is the ID of the job that created this backup
	JobID string `json:"job_id"`

	// VMID is the ID of the VM that was backed up
	VMID string `json:"vm_id"`

	// Type is the type of backup
	Type BackupType `json:"type"`

	// State is the state of the backup
	State BackupState `json:"state"`

	// ParentID is the ID of the parent backup for incremental backups
	ParentID string `json:"parent_id,omitempty"`

	// StartedAt is when the backup started
	StartedAt time.Time `json:"started_at"`

	// CompletedAt is when the backup completed
	CompletedAt time.Time `json:"completed_at,omitempty"`

	// Size is the size of the backup in bytes
	Size int64 `json:"size"`

	// StorageLocation is where the backup is stored
	StorageLocation string `json:"storage_location"`

	// Metadata is additional metadata for the backup
	Metadata map[string]string `json:"metadata,omitempty"`

	// TenantID is the ID of the tenant this backup belongs to
	TenantID string `json:"tenant_id"`

	// TargetMetadata contains metadata for each target
	TargetMetadata map[string]map[string]string `json:"target_metadata,omitempty"`

	// Error is the error message if the backup failed
	Error string `json:"error,omitempty"`
}

// RestoreJob represents a restore job
type RestoreJob struct {
	// ID is the unique identifier of the job
	ID string `json:"id"`

	// Name is the human-readable name of the job
	Name string `json:"name"`

	// BackupID is the ID of the backup to restore from
	BackupID string `json:"backup_id"`

	// Targets are the targets to restore
	Targets []*RestoreTarget `json:"targets"`

	// State is the state of the restore
	State RestoreState `json:"state"`

	// Options are the restore options
	Options *RestoreOptions `json:"options"`

	// StartedAt is when the restore started
	StartedAt time.Time `json:"started_at"`

	// CompletedAt is when the restore completed
	CompletedAt time.Time `json:"completed_at,omitempty"`

	// TenantID is the ID of the tenant this restore belongs to
	TenantID string `json:"tenant_id"`

	// Error is the error message if the restore failed
	Error string `json:"error,omitempty"`
}

// RestoreTarget represents a target for restore
type RestoreTarget struct {
	// SourceID is the ID of the source in the backup
	SourceID string `json:"source_id"`

	// DestinationID is the ID of the destination to restore to
	DestinationID string `json:"destination_id"`

	// Type is the type of target (e.g., VM, volume, database)
	Type string `json:"type"`

	// State is the state of the restore for this target
	State RestoreState `json:"state"`

	// Options are target-specific restore options
	Options map[string]interface{} `json:"options,omitempty"`
}

// RestoreOptions represents options for a restore job
type RestoreOptions struct {
	// PointInTime is the point in time to restore to
	PointInTime time.Time `json:"point_in_time,omitempty"`

	// OverwriteExisting specifies if existing resources should be overwritten
	OverwriteExisting bool `json:"overwrite_existing"`

	// RestorePermissions specifies if permissions should be restored
	RestorePermissions bool `json:"restore_permissions"`

	// ValidateBeforeRestore specifies if the backup should be validated before restoring
	ValidateBeforeRestore bool `json:"validate_before_restore"`

	// TestRestore specifies if this is a test restore
	TestRestore bool `json:"test_restore"`
}

// BackupProvider defines the interface for backup providers
type BackupProvider interface {
	// ID returns the provider ID
	ID() string

	// Name returns the provider name
	Name() string

	// Type returns the type of storage this provider supports
	Type() StorageType

	// CreateBackup creates a backup
	CreateBackup(ctx context.Context, job *BackupJob) (*Backup, error)

	// DeleteBackup deletes a backup
	DeleteBackup(ctx context.Context, backupID string) error

	// RestoreBackup restores a backup
	RestoreBackup(ctx context.Context, job *RestoreJob) error

	// ListBackups lists backups
	ListBackups(ctx context.Context, filter map[string]interface{}) ([]*Backup, error)

	// GetBackup gets a backup by ID
	GetBackup(ctx context.Context, backupID string) (*Backup, error)

	// ValidateBackup validates a backup
	ValidateBackup(ctx context.Context, backupID string) error
}

// BackupManager manages backup and restore operations
type BackupManager struct {
	// providers is a map of provider ID to provider
	providers map[string]BackupProvider

	// jobs is a map of job ID to job
	jobs map[string]*BackupJob

	// backups is a map of backup ID to backup
	backups map[string]*Backup

	// restores is a map of restore job ID to restore job
	restores map[string]*RestoreJob

	// tenantJobs is a map of tenant ID to job IDs
	tenantJobs map[string][]string

	// tenantBackups is a map of tenant ID to backup IDs
	tenantBackups map[string][]string

	// tenantRestores is a map of tenant ID to restore IDs
	tenantRestores map[string][]string

	// scheduler is the scheduler for backup jobs
	scheduler *BackupScheduler

	// mutex protects the maps
	mutex sync.RWMutex
}

// NewBackupManager creates a new backup manager
func NewBackupManager() *BackupManager {
	manager := &BackupManager{
		providers:      make(map[string]BackupProvider),
		jobs:           make(map[string]*BackupJob),
		backups:        make(map[string]*Backup),
		restores:       make(map[string]*RestoreJob),
		tenantJobs:     make(map[string][]string),
		tenantBackups:  make(map[string][]string),
		tenantRestores: make(map[string][]string),
	}

	// Create scheduler
	manager.scheduler = NewBackupScheduler(manager)

	return manager
}

// Start starts the backup manager
func (m *BackupManager) Start() error {
	// Start scheduler
	return m.scheduler.Start()
}

// Stop stops the backup manager
func (m *BackupManager) Stop() error {
	// Stop scheduler
	return m.scheduler.Stop()
}

// RegisterProvider registers a backup provider
func (m *BackupManager) RegisterProvider(provider BackupProvider) error {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	// Check if provider already exists
	if _, exists := m.providers[provider.ID()]; exists {
		return fmt.Errorf("provider with ID %s already exists", provider.ID())
	}

	// Add provider
	m.providers[provider.ID()] = provider

	return nil
}

// GetProvider gets a provider by ID
func (m *BackupManager) GetProvider(providerID string) (BackupProvider, error) {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	// Check if provider exists
	provider, exists := m.providers[providerID]
	if !exists {
		return nil, fmt.Errorf("provider with ID %s does not exist", providerID)
	}

	return provider, nil
}

// CreateBackupJob creates a backup job
func (m *BackupManager) CreateBackupJob(job *BackupJob) error {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	// Check if job ID already exists
	if _, exists := m.jobs[job.ID]; exists {
		return fmt.Errorf("job with ID %s already exists", job.ID)
	}

	// Set created and updated times
	now := time.Now()
	job.CreatedAt = now
	job.UpdatedAt = now

	// Add job
	m.jobs[job.ID] = job

	// Add to tenant jobs
	m.tenantJobs[job.TenantID] = append(m.tenantJobs[job.TenantID], job.ID)

	// Schedule job if enabled
	if job.Enabled && job.Schedule != nil {
		m.scheduler.ScheduleJob(job)
	}

	return nil
}

// UpdateBackupJob updates a backup job
func (m *BackupManager) UpdateBackupJob(job *BackupJob) error {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	// Check if job exists
	existingJob, exists := m.jobs[job.ID]
	if !exists {
		return fmt.Errorf("job with ID %s does not exist", job.ID)
	}

	// Check tenant ID
	if existingJob.TenantID != job.TenantID {
		return errors.New("cannot change tenant ID of a backup job")
	}

	// Update timestamps
	job.CreatedAt = existingJob.CreatedAt
	job.UpdatedAt = time.Now()

	// Update job
	m.jobs[job.ID] = job

	// Reschedule if needed
	if job.Enabled && job.Schedule != nil {
		m.scheduler.ScheduleJob(job)
	} else {
		m.scheduler.UnscheduleJob(job.ID)
	}

	return nil
}

// DeleteBackupJob deletes a backup job
func (m *BackupManager) DeleteBackupJob(jobID string) error {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	// Check if job exists
	job, exists := m.jobs[jobID]
	if !exists {
		return fmt.Errorf("job with ID %s does not exist", jobID)
	}

	// Unschedule job
	m.scheduler.UnscheduleJob(jobID)

	// Remove from tenant jobs
	tenantJobs := m.tenantJobs[job.TenantID]
	for i, id := range tenantJobs {
		if id == jobID {
			m.tenantJobs[job.TenantID] = append(tenantJobs[:i], tenantJobs[i+1:]...)
			break
		}
	}

	// Remove job
	delete(m.jobs, jobID)

	return nil
}

// GetBackupJob gets a backup job by ID
func (m *BackupManager) GetBackupJob(jobID string) (*BackupJob, error) {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	// Check if job exists
	job, exists := m.jobs[jobID]
	if !exists {
		return nil, fmt.Errorf("job with ID %s: %w", jobID, ErrBackupNotFound)
	}

	return job, nil
}

// ListBackupJobs lists backup jobs
func (m *BackupManager) ListBackupJobs(tenantID string) ([]*BackupJob, error) {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	// If tenant ID is provided, list only jobs for that tenant
	if tenantID != "" {
		jobIDs := m.tenantJobs[tenantID]
		jobs := make([]*BackupJob, 0, len(jobIDs))
		for _, jobID := range jobIDs {
			if job, exists := m.jobs[jobID]; exists {
				jobs = append(jobs, job)
			}
		}
		return jobs, nil
	}

	// List all jobs
	jobs := make([]*BackupJob, 0, len(m.jobs))
	for _, job := range m.jobs {
		jobs = append(jobs, job)
	}

	return jobs, nil
}

// RunBackupJob runs a backup job
func (m *BackupManager) RunBackupJob(ctx context.Context, jobID string) (*Backup, error) {
	// Get job
	job, err := m.GetBackupJob(jobID)
	if err != nil {
		return nil, err
	}

	// Check if job is enabled
	if !job.Enabled {
		return nil, errors.New("job is not enabled")
	}

	// Get storage config
	storage := job.Storage
	if storage == nil {
		return nil, errors.New("job has no storage configuration")
	}

	// Get provider for storage type
	var provider BackupProvider
	for _, p := range m.providers {
		if p.Type() == storage.Type {
			provider = p
			break
		}
	}
	if provider == nil {
		return nil, fmt.Errorf("no provider found for storage type %s", storage.Type)
	}

	// Update job
	m.mutex.Lock()
	job.LastRunAt = time.Now()
	job.LastRunStatus = BackupInProgress
	m.jobs[job.ID] = job
	m.mutex.Unlock()

	// Create backup
	backup, err := provider.CreateBackup(ctx, job)
	if err != nil {
		// Update job with failure
		m.mutex.Lock()
		job.LastRunStatus = BackupFailed
		m.jobs[job.ID] = job
		m.mutex.Unlock()

		return nil, err
	}

	// Ensure VMID is populated from job targets or metadata
	if backup.VMID == "" {
		// Try to get VMID from job targets
		if len(job.Targets) > 0 && job.Targets[0].ResourceID != "" {
			backup.VMID = job.Targets[0].ResourceID
		} else if backup.Metadata != nil && backup.Metadata["vm_id"] != "" {
			backup.VMID = backup.Metadata["vm_id"]
		}
	}

	// Update job with success
	m.mutex.Lock()
	job.LastRunStatus = BackupCompleted
	job.LastSuccessfulRunAt = time.Now()
	// Update next run time based on schedule
	if job.Schedule != nil {
		// This is a simplified calculation - a real implementation would be more complex
		job.NextRunAt = calculateNextRunTime(job.Schedule)
	}
	m.jobs[job.ID] = job

	// Add backup to maps
	m.backups[backup.ID] = backup
	m.tenantBackups[backup.TenantID] = append(m.tenantBackups[backup.TenantID], backup.ID)
	m.mutex.Unlock()

	// Apply retention policy
	m.applyRetentionPolicy(job)

	return backup, nil
}

// GetBackup gets a backup by ID
func (m *BackupManager) GetBackup(backupID string) (*Backup, error) {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	// Check if backup exists
	backup, exists := m.backups[backupID]
	if !exists {
		return nil, fmt.Errorf("backup with ID %s: %w", backupID, ErrBackupNotFound)
	}

	return backup, nil
}

// ListBackups lists backups
func (m *BackupManager) ListBackups(tenantID string, jobID string) ([]*Backup, error) {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	var backupIDs []string
	if tenantID != "" {
		// List backups for tenant
		backupIDs = m.tenantBackups[tenantID]
	} else {
		// List all backups
		backupIDs = make([]string, 0, len(m.backups))
		for id := range m.backups {
			backupIDs = append(backupIDs, id)
		}
	}

	backups := make([]*Backup, 0, len(backupIDs))
	for _, id := range backupIDs {
		backup, exists := m.backups[id]
		if exists && (jobID == "" || backup.JobID == jobID) {
			backups = append(backups, backup)
		}
	}

	return backups, nil
}

// ListBackupsFiltered lists backups with filtering support
func (m *BackupManager) ListBackupsFiltered(ctx context.Context, filter BackupFilter) ([]*Backup, error) {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	backups := make([]*Backup, 0)
	
	// Start with tenant filtering if specified
	var candidateBackups []*Backup
	if filter.TenantID != "" {
		backupIDs := m.tenantBackups[filter.TenantID]
		for _, id := range backupIDs {
			if backup := m.backups[id]; backup != nil {
				candidateBackups = append(candidateBackups, backup)
			}
		}
	} else {
		// Get all backups
		for _, backup := range m.backups {
			candidateBackups = append(candidateBackups, backup)
		}
	}
	
	// Apply filters
	for _, backup := range candidateBackups {
		// Filter by type
		if filter.Type != "" && string(backup.Type) != filter.Type {
			continue
		}
		
		// Filter by VMID
		if filter.VMID != "" && backup.VMID != filter.VMID {
			continue
		}
		
		// Filter by date range
		if !filter.StartDate.IsZero() && backup.CreatedAt.Before(filter.StartDate) {
			continue
		}
		if !filter.EndDate.IsZero() && backup.CreatedAt.After(filter.EndDate) {
			continue
		}
		
		// Filter by state
		if filter.State != "" && string(backup.State) != filter.State {
			continue
		}
		
		// Filter by JobID
		if filter.JobID != "" && backup.JobID != filter.JobID {
			continue
		}
		
		backups = append(backups, backup)
	}
	
	return backups, nil
}

// DeleteBackup deletes a backup
func (m *BackupManager) DeleteBackup(ctx context.Context, backupID string) error {
	// Get backup
	backup, err := m.GetBackup(backupID)
	if err != nil {
		return err
	}

	// Get provider for backup
	var provider BackupProvider
	m.mutex.RLock()
	for _, p := range m.providers {
		if _, err := p.GetBackup(ctx, backupID); err == nil {
			provider = p
			break
		}
	}
	m.mutex.RUnlock()

	if provider == nil {
		return errors.New("no provider found for backup")
	}

	// Capture original children for rollback
	m.mutex.RLock()
	originalChildren := make(map[string]string) // childID -> originalParentID
	for id, b := range m.backups {
		if b.ParentID == backupID {
			originalChildren[id] = backupID
		}
	}
	m.mutex.RUnlock()

	// Delete backup from provider first
	if err := provider.DeleteBackup(ctx, backupID); err != nil {
		// Provider delete failed, no rollback needed as we haven't changed state
		return fmt.Errorf("failed to delete backup data: %w", err)
	}

	// Provider delete succeeded, now update backup chains
	if err := m.updateBackupChains(ctx, backupID); err != nil {
		// Chain update failed, attempt to restore original parent IDs
		m.mutex.Lock()
		for childID, originalParentID := range originalChildren {
			if child, exists := m.backups[childID]; exists {
				child.ParentID = originalParentID
			}
		}
		m.mutex.Unlock()
		return fmt.Errorf("failed to update backup chains (rolled back): %w", err)
	}

	// Remove backup from maps
	m.mutex.Lock()
	delete(m.backups, backupID)
	tenantBackups := m.tenantBackups[backup.TenantID]
	for i, id := range tenantBackups {
		if id == backupID {
			m.tenantBackups[backup.TenantID] = append(tenantBackups[:i], tenantBackups[i+1:]...)
			break
		}
	}
	m.mutex.Unlock()

	return nil
}

// CreateRestoreJob creates a restore job
func (m *BackupManager) CreateRestoreJob(ctx context.Context, job *RestoreJob) error {
	// Verify backup exists
	_, err := m.GetBackup(job.BackupID)
	if err != nil {
		return err
	}

	// Get provider for backup
	var provider BackupProvider
	m.mutex.RLock()
	for _, p := range m.providers {
		if _, err := p.GetBackup(ctx, job.BackupID); err == nil {
			provider = p
			break
		}
	}
	m.mutex.RUnlock()

	if provider == nil {
		return errors.New("no provider found for backup")
	}

	// Set job state and start time
	job.State = RestoreInProgress
	job.StartedAt = time.Now()

	// Add job to maps
	m.mutex.Lock()
	m.restores[job.ID] = job
	m.tenantRestores[job.TenantID] = append(m.tenantRestores[job.TenantID], job.ID)
	m.mutex.Unlock()

	// Start restore
	go func() {
		err := provider.RestoreBackup(ctx, job)
		m.mutex.Lock()
		defer m.mutex.Unlock()

		// Update job
		job.CompletedAt = time.Now()
		if err != nil {
			job.State = RestoreFailed
			job.Error = err.Error()
		} else {
			job.State = RestoreCompleted
		}
		m.restores[job.ID] = job
	}()

	return nil
}

// GetRestoreJob gets a restore job by ID
func (m *BackupManager) GetRestoreJob(jobID string) (*RestoreJob, error) {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	// Check if job exists
	job, exists := m.restores[jobID]
	if !exists {
		return nil, fmt.Errorf("restore job with ID %s: %w", jobID, ErrBackupNotFound)
	}

	return job, nil
}

// ListRestoreJobs lists restore jobs
func (m *BackupManager) ListRestoreJobs(tenantID string) ([]*RestoreJob, error) {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	// If tenant ID is provided, list only jobs for that tenant
	if tenantID != "" {
		jobIDs := m.tenantRestores[tenantID]
		jobs := make([]*RestoreJob, 0, len(jobIDs))
		for _, jobID := range jobIDs {
			if job, exists := m.restores[jobID]; exists {
				jobs = append(jobs, job)
			}
		}
		return jobs, nil
	}

	// List all jobs
	jobs := make([]*RestoreJob, 0, len(m.restores))
	for _, job := range m.restores {
		jobs = append(jobs, job)
	}

	return jobs, nil
}

// ValidateBackup validates a backup
func (m *BackupManager) ValidateBackup(ctx context.Context, backupID string) error {
	// Verify backup exists
	_, err := m.GetBackup(backupID)
	if err != nil {
		return err
	}

	// Get provider for backup
	var provider BackupProvider
	m.mutex.RLock()
	for _, p := range m.providers {
		if _, err := p.GetBackup(ctx, backupID); err == nil {
			provider = p
			break
		}
	}
	m.mutex.RUnlock()

	if provider == nil {
		return errors.New("no provider found for backup")
	}

	// Validate backup
	return provider.ValidateBackup(ctx, backupID)
}

// applyRetentionPolicy applies the retention policy for a job
func (m *BackupManager) applyRetentionPolicy(job *BackupJob) {
	// If no retention policy, nothing to do
	if job.Retention == nil {
		return
	}

	// Get backups for job
	backups, err := m.ListBackups("", job.ID)
	if err != nil {
		// Log error
		return
	}

	// Sort backups by time
	// In a real implementation, we would use a more sophisticated algorithm
	// to apply retention policies based on daily, weekly, monthly, etc.
	// as well as handling dependencies between incremental backups

	// For now, just implement a simple keep-last policy
	if job.Retention.KeepLast > 0 && len(backups) > job.Retention.KeepLast {
		// Determine which backups to delete
		// Skip the most recent KeepLast backups
		for i := 0; i < len(backups)-job.Retention.KeepLast; i++ {
			// Delete backup
			m.DeleteBackup(context.Background(), backups[i].ID)
		}
	}
}

// VerifyBackup verifies the integrity of a backup
func (m *BackupManager) VerifyBackup(ctx context.Context, backupID string) (*VerificationResult, error) {
	// Verify backup exists
	backup, err := m.GetBackup(backupID)
	if err != nil {
		return nil, err
	}

	// Get provider for backup
	var provider BackupProvider
	m.mutex.RLock()
	for _, p := range m.providers {
		if _, err := p.GetBackup(ctx, backupID); err == nil {
			provider = p
			break
		}
	}
	m.mutex.RUnlock()

	if provider == nil {
		return nil, errors.New("no provider found for backup")
	}

	result := &VerificationResult{
		BackupID:         backupID,
		Status:           "valid",
		CheckedItems:     0,
		ErrorsFound:      make([]string, 0),
		VerificationTime: time.Now(),
		Details:          make(map[string]interface{}),
	}

	// Validate backup through provider
	err = provider.ValidateBackup(ctx, backupID)
	if err != nil {
		result.Status = "corrupted"
		result.ErrorsFound = append(result.ErrorsFound, err.Error())
		result.Details["validation_error"] = err.Error()
	} else {
		result.CheckedItems = 1 // Basic validation count
		result.Details["validation_passed"] = true
	}

	// Verify backup chain consistency for incremental backups
	if backup.ParentID != "" {
		if err := m.verifyBackupChain(ctx, backupID); err != nil {
			result.Status = "incomplete"
			result.ErrorsFound = append(result.ErrorsFound, fmt.Sprintf("Chain verification failed: %v", err))
			result.Details["chain_error"] = err.Error()
		} else {
			result.CheckedItems++
			result.Details["chain_verified"] = true
		}
	}

	return result, nil
}

// ListAllBackups returns all backups across all VMs
func (m *BackupManager) ListAllBackups(ctx context.Context) ([]BackupInfo, error) {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	// Directly scan m.backups map to get all backups
	backups := make([]BackupInfo, 0, len(m.backups))
	for _, backup := range m.backups {
			// Use VMID field directly, fall back to metadata if not set
			vmID := backup.VMID
			if vmID == "" && backup.Metadata != nil {
				vmID = backup.Metadata["vm_id"]
			}
			
			backupInfo := BackupInfo{
				ID:        backup.ID,
				JobID:     backup.JobID,
				VMID:      vmID,  // Use actual VM ID field, not JobID
				Type:      backup.Type,
				State:     backup.State,
				Size:      backup.Size,
				StartedAt: backup.StartedAt,
				TenantID:  backup.TenantID,
				ParentID:  backup.ParentID,
				Metadata:  backup.Metadata,
			}
			if !backup.CompletedAt.IsZero() {
				backupInfo.CompletedAt = &backup.CompletedAt
			}
		backups = append(backups, backupInfo)
	}

	return backups, nil
}

// Helper methods

// hasChildBackups checks if a backup has children in an incremental chain
func (m *BackupManager) hasChildBackups(backupID string) bool {
	for _, backup := range m.backups {
		if backup.ParentID == backupID {
			return true
		}
	}
	return false
}

// updateBackupChains handles chain updates when a backup is deleted
func (m *BackupManager) updateBackupChains(ctx context.Context, deletedBackupID string) error {
	// Find child backups that reference the deleted backup
	childBackups := make([]*Backup, 0)
	for _, backup := range m.backups {
		if backup.ParentID == deletedBackupID {
			childBackups = append(childBackups, backup)
		}
	}

	// Update child backups to reference the deleted backup's parent
	deletedBackup := m.backups[deletedBackupID]
	for _, child := range childBackups {
		child.ParentID = deletedBackup.ParentID
	}

	return nil
}

// verifyBackupChain verifies the consistency of an incremental backup chain
func (m *BackupManager) verifyBackupChain(ctx context.Context, backupID string) error {
	backup := m.backups[backupID]
	if backup.ParentID == "" {
		return nil // No parent to verify
	}

	// Check if parent exists
	_, exists := m.backups[backup.ParentID]
	if !exists {
		return fmt.Errorf("parent backup %s not found", backup.ParentID)
	}

	// Recursively verify parent chain
	return m.verifyBackupChain(ctx, backup.ParentID)
}

// BackupInfo represents backup information for API responses
type BackupInfo struct {
	ID          string            `json:"id"`
	JobID       string            `json:"job_id"`
	VMID        string            `json:"vm_id"`     // Add proper VM ID field
	Type        BackupType        `json:"type"`
	State       BackupState       `json:"state"`
	Size        int64             `json:"size"`
	StartedAt   time.Time         `json:"started_at"`
	CompletedAt *time.Time        `json:"completed_at,omitempty"`
	TenantID    string            `json:"tenant_id"`
	ParentID    string            `json:"parent_id,omitempty"`
	Metadata    map[string]string `json:"metadata,omitempty"`
}

// calculateNextRunTime calculates the next run time for a schedule
func calculateNextRunTime(schedule *Schedule) time.Time {
	// This is a simplified implementation
	// A real implementation would need to handle cron expressions, intervals, etc.
	return time.Now().Add(24 * time.Hour)
}

// GetBackupManifest retrieves the manifest for a backup
// This is a stub that returns a basic manifest constructed from backup metadata
func (m *BackupManager) GetBackupManifest(backupID string) (*BackupManifest, error) {
	backup, err := m.GetBackup(backupID)
	if err != nil {
		return nil, err
	}
	
	// Extract VM ID from metadata
	vmID := ""
	if backup.Metadata != nil {
		vmID = backup.Metadata["vm_id"]
	}
	
	// Create a basic manifest from backup data
	manifest := &BackupManifest{
		BackupID:     backup.ID,
		VMID:         vmID,
		Type:         backup.Type,
		ParentID:     backup.ParentID,
		Size:         backup.Size,
		CompressedSize: backup.Size, // Placeholder - actual compression would be stored
		CreatedAt:    backup.StartedAt,
		Metadata:     backup.Metadata,
	}
	
	return manifest, nil
}

// InitializeCBT initializes Changed Block Tracking for a VM
// This is a stub implementation
func (m *BackupManager) InitializeCBT(vmID string, vmSize int64) (*CBTTracker, error) {
	if vmID == "" {
		return nil, fmt.Errorf("VM ID is required")
	}
	if vmSize <= 0 {
		return nil, fmt.Errorf("VM size must be positive")
	}
	
	// Create a basic CBT tracker (stub)
	tracker := &CBTTracker{
		vmID:        vmID,
		totalBlocks: vmSize / CBTBlockSize,
		blockSize:   CBTBlockSize,
		createdAt:   time.Now(),
		updatedAt:   time.Now(),
		blocks:      make(map[int64]*BlockInfo),
	}
	
	return tracker, nil
}

// GetCBTStats retrieves CBT statistics for a VM
// This is a stub implementation
func (m *BackupManager) GetCBTStats(vmID string) (map[string]interface{}, error) {
	if vmID == "" {
		return nil, fmt.Errorf("VM ID is required")
	}
	
	// Return placeholder stats
	stats := map[string]interface{}{
		"vm_id":         vmID,
		"total_blocks":  0,
		"changed_blocks": 0,
		"block_size":    CBTBlockSize,
		"last_backup":   nil,
		"initialized":   false,
	}
	
	return stats, nil
}

// Test helper methods

// AddBackupForTest adds a backup for testing purposes
func (m *BackupManager) AddBackupForTest(backup *Backup) {
	m.mutex.Lock()
	defer m.mutex.Unlock()
	m.backups[backup.ID] = backup
	m.tenantBackups[backup.TenantID] = append(m.tenantBackups[backup.TenantID], backup.ID)
}

// BackupScheduler schedules and executes backup jobs
type BackupScheduler struct {
	// manager is the backup manager
	manager *BackupManager

	// jobSchedules maps job IDs to their next scheduled run time
	jobSchedules map[string]time.Time

	// mutex protects jobSchedules
	mutex sync.RWMutex

	// stopChan is used to stop the scheduler
	stopChan chan struct{}

	// wg is used to wait for the scheduler to stop
	wg sync.WaitGroup
}

// NewBackupScheduler creates a new backup scheduler
func NewBackupScheduler(manager *BackupManager) *BackupScheduler {
	return &BackupScheduler{
		manager:      manager,
		jobSchedules: make(map[string]time.Time),
		stopChan:     make(chan struct{}),
	}
}

// Start starts the scheduler
func (s *BackupScheduler) Start() error {
	s.wg.Add(1)
	go s.run()
	return nil
}

// Stop stops the scheduler
func (s *BackupScheduler) Stop() error {
	close(s.stopChan)
	s.wg.Wait()
	return nil
}

// ScheduleJob schedules a job
func (s *BackupScheduler) ScheduleJob(job *BackupJob) {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	// Calculate next run time
	nextRun := calculateNextRunTime(job.Schedule)
	s.jobSchedules[job.ID] = nextRun

	// Update job
	job.NextRunAt = nextRun
}

// UnscheduleJob unschedules a job
func (s *BackupScheduler) UnscheduleJob(jobID string) {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	delete(s.jobSchedules, jobID)
}

// run is the main scheduler loop
func (s *BackupScheduler) run() {
	defer s.wg.Done()

	ticker := time.NewTicker(1 * time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-s.stopChan:
			return
		case <-ticker.C:
			s.checkAndRunDueJobs()
		}
	}
}

// checkAndRunDueJobs checks for jobs due to run and executes them
func (s *BackupScheduler) checkAndRunDueJobs() {
	now := time.Now()

	// Get jobs due to run
	dueJobs := make([]string, 0)
	s.mutex.Lock()
	for jobID, nextRun := range s.jobSchedules {
		if nextRun.Before(now) || nextRun.Equal(now) {
			dueJobs = append(dueJobs, jobID)
		}
	}
	s.mutex.Unlock()

	// Run due jobs
	for _, jobID := range dueJobs {
		job, err := s.manager.GetBackupJob(jobID)
		if err != nil {
			// Log error and continue
			continue
		}

		// Skip if job is not enabled
		if !job.Enabled {
			continue
		}

		// Run job in a goroutine
		go func(j *BackupJob) {
			_, err := s.manager.RunBackupJob(context.Background(), j.ID)
			if err != nil {
				// Log error
			}
		}(job)
	}
}
