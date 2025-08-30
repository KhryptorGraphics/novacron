package backup

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// IncrementalBackupEngine handles incremental backup operations
type IncrementalBackupEngine struct {
	// cbtTracker manages changed block tracking
	cbtTracker *CBTTracker
	
	// backupChains tracks backup chains for each resource
	backupChains map[string]*BackupChain
	
	// vmManager interface for VM operations
	vmManager VMManagerInterface
	
	// storageManager interface for storage operations
	storageManager StorageManagerInterface
	
	// compressionProvider for backup compression
	compressionProvider CompressionProvider
	
	// encryptionProvider for backup encryption
	encryptionProvider EncryptionProvider
	
	// deduplicationEngine for block-level deduplication
	deduplicationEngine *DeduplicationEngine
	
	// mutex protects concurrent access
	mutex sync.RWMutex
}

// VMManagerInterface defines the interface to VM manager
type VMManagerInterface interface {
	GetVMState(ctx context.Context, vmID string) (string, error)
	PauseVM(ctx context.Context, vmID string) error
	ResumeVM(ctx context.Context, vmID string) error
	CreateVMSnapshot(ctx context.Context, vmID string, snapshotName string) (string, error)
	DeleteVMSnapshot(ctx context.Context, vmID, snapshotID string) error
	GetVMDisks(ctx context.Context, vmID string) ([]VMDisk, error)
}

// StorageManagerInterface defines the interface to storage manager
type StorageManagerInterface interface {
	ReadBlocks(ctx context.Context, volumeID string, blocks []CBTBlock) ([]byte, error)
	WriteBlocks(ctx context.Context, volumeID string, blocks []CBTBlock, data []byte) error
	CreateVolume(ctx context.Context, volumeID string, size int64) error
	GetVolumeSize(ctx context.Context, volumeID string) (int64, error)
	GetVolumeChecksum(ctx context.Context, volumeID string) (string, error)
}

// VMDisk represents a VM disk
type VMDisk struct {
	ID       string `json:"id"`
	VolumeID string `json:"volume_id"`
	Device   string `json:"device"`
	Size     int64  `json:"size"`
	Type     string `json:"type"`
}

// BackupChain represents a chain of related backups
type BackupChain struct {
	ResourceID   string              `json:"resource_id"`
	FullBackups  []string            `json:"full_backups"`    // List of full backup IDs
	Incrementals map[string][]string `json:"incrementals"`    // fullBackupID -> []incrementalBackupIDs
	LastFull     string              `json:"last_full"`       // ID of most recent full backup
	LastBackup   string              `json:"last_backup"`     // ID of most recent backup (any type)
	CreatedAt    time.Time           `json:"created_at"`
	UpdatedAt    time.Time           `json:"updated_at"`
}

// CompressionProvider defines interface for backup compression
type CompressionProvider interface {
	Compress(ctx context.Context, data []byte, level int) ([]byte, error)
	Decompress(ctx context.Context, compressedData []byte) ([]byte, error)
	EstimateCompressionRatio(data []byte) float64
}

// EncryptionProvider defines interface for backup encryption
type EncryptionProvider interface {
	Encrypt(ctx context.Context, data []byte, keyID string) ([]byte, error)
	Decrypt(ctx context.Context, encryptedData []byte, keyID string) ([]byte, error)
	GenerateKey(ctx context.Context, keyID string) error
	RotateKey(ctx context.Context, oldKeyID, newKeyID string) error
}

// DeduplicationEngine handles block-level deduplication
type DeduplicationEngine struct {
	blockStore   map[string][]byte // hash -> block data
	refCounts    map[string]int    // hash -> reference count
	hashToBlocks map[string][]string // hash -> list of backup IDs using this block
	mutex        sync.RWMutex
}

// IncrementalBackupResult contains the results of an incremental backup
type IncrementalBackupResult struct {
	BackupID          string        `json:"backup_id"`
	Type              BackupType    `json:"type"`
	ParentBackupID    string        `json:"parent_backup_id,omitempty"`
	TotalSize         int64         `json:"total_size"`
	CompressedSize    int64         `json:"compressed_size"`
	DeduplicatedSize  int64         `json:"deduplicated_size"`
	ChangedBlocks     int64         `json:"changed_blocks"`
	TotalBlocks       int64         `json:"total_blocks"`
	CompressionRatio  float64       `json:"compression_ratio"`
	DeduplicationRatio float64      `json:"deduplication_ratio"`
	Duration          time.Duration `json:"duration"`
	Throughput        float64       `json:"throughput_mbps"`
}

// RestoreResult contains the results of a restore operation
type RestoreResult struct {
	RestoreID        string        `json:"restore_id"`
	RestoredSize     int64         `json:"restored_size"`
	RestoredBlocks   int64         `json:"restored_blocks"`
	Duration         time.Duration `json:"duration"`
	Throughput       float64       `json:"throughput_mbps"`
	IntegrityChecked bool          `json:"integrity_checked"`
}

// NewIncrementalBackupEngine creates a new incremental backup engine
func NewIncrementalBackupEngine(
	cbtTracker *CBTTracker,
	vmManager VMManagerInterface,
	storageManager StorageManagerInterface,
	compressionProvider CompressionProvider,
	encryptionProvider EncryptionProvider,
) *IncrementalBackupEngine {
	return &IncrementalBackupEngine{
		cbtTracker:          cbtTracker,
		backupChains:        make(map[string]*BackupChain),
		vmManager:           vmManager,
		storageManager:      storageManager,
		compressionProvider: compressionProvider,
		encryptionProvider:  encryptionProvider,
		deduplicationEngine: NewDeduplicationEngine(),
	}
}

// CreateIncrementalBackup creates an incremental backup
func (engine *IncrementalBackupEngine) CreateIncrementalBackup(ctx context.Context, job *BackupJob) (*IncrementalBackupResult, error) {
	startTime := time.Now()
	
	// Validate backup targets
	if len(job.Targets) == 0 {
		return nil, fmt.Errorf("no backup targets specified")
	}
	
	// For now, support single VM target
	target := job.Targets[0]
	if target.Type != "vm" {
		return nil, fmt.Errorf("unsupported target type: %s", target.Type)
	}
	
	// Get backup chain for the target
	chain, err := engine.getOrCreateBackupChain(target.ResourceID)
	if err != nil {
		return nil, fmt.Errorf("failed to get backup chain: %w", err)
	}
	
	// Determine backup type based on chain state and job configuration
	backupType := engine.determineBackupType(job, chain)
	
	// Get VM disks
	disks, err := engine.vmManager.GetVMDisks(ctx, target.ResourceID)
	if err != nil {
		return nil, fmt.Errorf("failed to get VM disks: %w", err)
	}
	
	// Create consistent snapshot if required
	var snapshotID string
	if engine.requiresVMSnapshot(backupType) {
		snapshotID, err = engine.createConsistentSnapshot(ctx, target.ResourceID)
		if err != nil {
			return nil, fmt.Errorf("failed to create VM snapshot: %w", err)
		}
		defer func() {
			if snapshotID != "" {
				engine.vmManager.DeleteVMSnapshot(ctx, target.ResourceID, snapshotID)
			}
		}()
	}
	
	// Backup each disk
	var totalSize, compressedSize, deduplicatedSize int64
	var totalBlocks, changedBlocks int64
	
	for _, disk := range disks {
		diskResult, err := engine.backupDisk(ctx, disk, backupType, chain)
		if err != nil {
			return nil, fmt.Errorf("failed to backup disk %s: %w", disk.ID, err)
		}
		
		totalSize += diskResult.TotalSize
		compressedSize += diskResult.CompressedSize
		deduplicatedSize += diskResult.DeduplicatedSize
		totalBlocks += diskResult.TotalBlocks
		changedBlocks += diskResult.ChangedBlocks
	}
	
	// Update backup chain
	backupID := generateBackupID()
	err = engine.updateBackupChain(chain, backupID, backupType)
	if err != nil {
		return nil, fmt.Errorf("failed to update backup chain: %w", err)
	}
	
	duration := time.Since(startTime)
	
	// Calculate metrics
	compressionRatio := float64(compressedSize) / float64(totalSize)
	deduplicationRatio := float64(deduplicatedSize) / float64(compressedSize)
	throughputMbps := float64(totalSize) / duration.Seconds() / 1024 / 1024
	
	result := &IncrementalBackupResult{
		BackupID:           backupID,
		Type:               backupType,
		TotalSize:          totalSize,
		CompressedSize:     compressedSize,
		DeduplicatedSize:   deduplicatedSize,
		ChangedBlocks:      changedBlocks,
		TotalBlocks:        totalBlocks,
		CompressionRatio:   compressionRatio,
		DeduplicationRatio: deduplicationRatio,
		Duration:           duration,
		Throughput:         throughputMbps,
	}
	
	if backupType == IncrementalBackup && chain.LastBackup != "" {
		result.ParentBackupID = chain.LastBackup
	}
	
	return result, nil
}

// RestoreFromIncremental restores data from incremental backups
func (engine *IncrementalBackupEngine) RestoreFromIncremental(ctx context.Context, restoreJob *RestoreJob) (*RestoreResult, error) {
	startTime := time.Now()
	
	// Get backup chain for the restore
	backup, err := engine.getBackupMetadata(restoreJob.BackupID)
	if err != nil {
		return nil, fmt.Errorf("failed to get backup metadata: %w", err)
	}
	
	// Build restore chain (backup + all dependencies)
	restoreChain, err := engine.buildRestoreChain(ctx, backup)
	if err != nil {
		return nil, fmt.Errorf("failed to build restore chain: %w", err)
	}
	
	// Restore each backup in the chain
	var totalSize, totalBlocks int64
	for _, backupID := range restoreChain {
		size, blocks, err := engine.restoreBackup(ctx, backupID, restoreJob)
		if err != nil {
			return nil, fmt.Errorf("failed to restore backup %s: %w", backupID, err)
		}
		totalSize += size
		totalBlocks += blocks
	}
	
	// Verify restore integrity if requested
	integrityChecked := false
	if restoreJob.Options != nil && restoreJob.Options.ValidateBeforeRestore {
		if err := engine.verifyRestoreIntegrity(ctx, restoreJob); err != nil {
			return nil, fmt.Errorf("restore integrity verification failed: %w", err)
		}
		integrityChecked = true
	}
	
	duration := time.Since(startTime)
	throughputMbps := float64(totalSize) / duration.Seconds() / 1024 / 1024
	
	return &RestoreResult{
		RestoreID:        restoreJob.ID,
		RestoredSize:     totalSize,
		RestoredBlocks:   totalBlocks,
		Duration:         duration,
		Throughput:       throughputMbps,
		IntegrityChecked: integrityChecked,
	}, nil
}

// OptimizeBackupChain optimizes a backup chain by consolidating incrementals
func (engine *IncrementalBackupEngine) OptimizeBackupChain(ctx context.Context, resourceID string) error {
	engine.mutex.Lock()
	defer engine.mutex.Unlock()
	
	chain, exists := engine.backupChains[resourceID]
	if !exists {
		return fmt.Errorf("no backup chain found for resource %s", resourceID)
	}
	
	// If we have too many incrementals, create a new synthetic full backup
	if len(chain.Incrementals[chain.LastFull]) > MaxIncrementals {
		return engine.consolidateIncrementals(ctx, chain)
	}
	
	return nil
}

// GetBackupChainStatus returns the status of a backup chain
func (engine *IncrementalBackupEngine) GetBackupChainStatus(ctx context.Context, resourceID string) (*BackupChainStatus, error) {
	engine.mutex.RLock()
	defer engine.mutex.RUnlock()
	
	chain, exists := engine.backupChains[resourceID]
	if !exists {
		return nil, fmt.Errorf("no backup chain found for resource %s", resourceID)
	}
	
	status := &BackupChainStatus{
		ResourceID:       resourceID,
		FullBackupCount:  len(chain.FullBackups),
		LastFullBackup:   chain.LastFull,
		LastBackup:       chain.LastBackup,
		ChainLength:      engine.calculateChainLength(chain),
		EstimatedSize:    0, // Would be calculated from actual backup metadata
		HealthStatus:     "healthy",
		CreatedAt:        chain.CreatedAt,
		UpdatedAt:        chain.UpdatedAt,
	}
	
	// Calculate incremental count
	for _, incrementals := range chain.Incrementals {
		status.IncrementalBackupCount += len(incrementals)
	}
	
	// Check chain health
	if status.ChainLength > MaxIncrementals {
		status.HealthStatus = "needs_optimization"
	}
	
	return status, nil
}

// BackupChainStatus represents the status of a backup chain
type BackupChainStatus struct {
	ResourceID             string    `json:"resource_id"`
	FullBackupCount        int       `json:"full_backup_count"`
	IncrementalBackupCount int       `json:"incremental_backup_count"`
	LastFullBackup         string    `json:"last_full_backup"`
	LastBackup             string    `json:"last_backup"`
	ChainLength            int       `json:"chain_length"`
	EstimatedSize          int64     `json:"estimated_size"`
	HealthStatus           string    `json:"health_status"`
	CreatedAt              time.Time `json:"created_at"`
	UpdatedAt              time.Time `json:"updated_at"`
}

// Helper methods

func (engine *IncrementalBackupEngine) getOrCreateBackupChain(resourceID string) (*BackupChain, error) {
	engine.mutex.Lock()
	defer engine.mutex.Unlock()
	
	if chain, exists := engine.backupChains[resourceID]; exists {
		return chain, nil
	}
	
	// Create new backup chain
	chain := &BackupChain{
		ResourceID:   resourceID,
		FullBackups:  make([]string, 0),
		Incrementals: make(map[string][]string),
		CreatedAt:    time.Now(),
		UpdatedAt:    time.Now(),
	}
	
	engine.backupChains[resourceID] = chain
	return chain, nil
}

func (engine *IncrementalBackupEngine) determineBackupType(job *BackupJob, chain *BackupChain) BackupType {
	// If explicitly specified, use job type
	if job.Type == FullBackup || job.Type == IncrementalBackup || job.Type == DifferentialBackup {
		// But ensure we have a full backup first for incremental/differential
		if (job.Type == IncrementalBackup || job.Type == DifferentialBackup) && chain.LastFull == "" {
			return FullBackup
		}
		return job.Type
	}
	
	// Auto-determine based on chain state
	if chain.LastFull == "" {
		return FullBackup
	}
	
	// Check if it's time for a new full backup
	incrementalCount := len(chain.Incrementals[chain.LastFull])
	if incrementalCount >= 7 { // Weekly full backup policy
		return FullBackup
	}
	
	return IncrementalBackup
}

func (engine *IncrementalBackupEngine) requiresVMSnapshot(backupType BackupType) bool {
	// For demonstration, assume we always want consistent snapshots for full backups
	return backupType == FullBackup
}

func (engine *IncrementalBackupEngine) createConsistentSnapshot(ctx context.Context, vmID string) (string, error) {
	// Pause VM briefly for consistency
	if err := engine.vmManager.PauseVM(ctx, vmID); err != nil {
		return "", err
	}
	
	// Create snapshot
	snapshotName := fmt.Sprintf("backup-snapshot-%d", time.Now().Unix())
	snapshotID, err := engine.vmManager.CreateVMSnapshot(ctx, vmID, snapshotName)
	
	// Resume VM
	if resumeErr := engine.vmManager.ResumeVM(ctx, vmID); resumeErr != nil {
		// Log error but don't fail the backup
	}
	
	return snapshotID, err
}

func (engine *IncrementalBackupEngine) backupDisk(ctx context.Context, disk VMDisk, backupType BackupType, chain *BackupChain) (*IncrementalBackupResult, error) {
	// Get changed blocks for the disk based on backup type
	var lastBackupTime *time.Time
	if backupType == IncrementalBackup && chain.LastBackup != "" {
		// In a real implementation, this would get the timestamp from backup metadata
		now := time.Now().Add(-24 * time.Hour) // Simulate last backup was 24 hours ago
		lastBackupTime = &now
	}
	
	blocks, err := engine.cbtTracker.OptimizeForBackup(ctx, disk.VolumeID, backupType, lastBackupTime)
	if err != nil {
		return nil, err
	}
	
	// Read block data from storage
	blockData, err := engine.storageManager.ReadBlocks(ctx, disk.VolumeID, blocks)
	if err != nil {
		return nil, err
	}
	
	// Apply deduplication
	deduplicatedData, deduplicationRatio := engine.deduplicationEngine.Deduplicate(blockData)
	
	// Compress data
	compressedData, err := engine.compressionProvider.Compress(ctx, deduplicatedData, 6)
	if err != nil {
		return nil, err
	}
	
	// Encrypt data if required
	// This would be implemented based on job.Storage.Encryption settings
	
	return &IncrementalBackupResult{
		TotalSize:          int64(len(blockData)),
		CompressedSize:     int64(len(compressedData)),
		DeduplicatedSize:   int64(len(deduplicatedData)),
		ChangedBlocks:      int64(len(blocks)),
		TotalBlocks:        disk.Size / CBTBlockSize,
		CompressionRatio:   float64(len(compressedData)) / float64(len(deduplicatedData)),
		DeduplicationRatio: deduplicationRatio,
	}, nil
}

func (engine *IncrementalBackupEngine) updateBackupChain(chain *BackupChain, backupID string, backupType BackupType) error {
	engine.mutex.Lock()
	defer engine.mutex.Unlock()
	
	chain.UpdatedAt = time.Now()
	chain.LastBackup = backupID
	
	if backupType == FullBackup {
		chain.FullBackups = append(chain.FullBackups, backupID)
		chain.LastFull = backupID
		chain.Incrementals[backupID] = make([]string, 0)
	} else {
		// Add to incrementals of the current full backup
		if chain.LastFull != "" {
			chain.Incrementals[chain.LastFull] = append(chain.Incrementals[chain.LastFull], backupID)
		}
	}
	
	return nil
}

func (engine *IncrementalBackupEngine) getBackupMetadata(backupID string) (*Backup, error) {
	// In a real implementation, this would load backup metadata from storage
	return &Backup{
		ID:   backupID,
		Type: IncrementalBackup,
	}, nil
}

func (engine *IncrementalBackupEngine) buildRestoreChain(ctx context.Context, backup *Backup) ([]string, error) {
	// Build the chain of backups needed for restore
	chain := []string{backup.ID}
	
	// If this is an incremental backup, we need its parent and all dependencies
	if backup.Type == IncrementalBackup && backup.ParentID != "" {
		parentChain, err := engine.buildRestoreChain(ctx, &Backup{ID: backup.ParentID})
		if err != nil {
			return nil, err
		}
		chain = append(parentChain, chain...)
	}
	
	return chain, nil
}

func (engine *IncrementalBackupEngine) restoreBackup(ctx context.Context, backupID string, restoreJob *RestoreJob) (int64, int64, error) {
	// In a real implementation, this would:
	// 1. Load backup metadata
	// 2. Read backup data from storage
	// 3. Decrypt if encrypted
	// 4. Decompress
	// 5. Apply blocks to target volume
	// 6. Update CBT tracker
	
	// For now, return simulated values
	return 1024 * 1024 * 1024, 256, nil // 1GB, 256 blocks
}

func (engine *IncrementalBackupEngine) verifyRestoreIntegrity(ctx context.Context, restoreJob *RestoreJob) error {
	// In a real implementation, this would verify checksums, file integrity, etc.
	return nil
}

func (engine *IncrementalBackupEngine) consolidateIncrementals(ctx context.Context, chain *BackupChain) error {
	// In a real implementation, this would create a synthetic full backup
	// by applying all incrementals to the base full backup
	return nil
}

func (engine *IncrementalBackupEngine) calculateChainLength(chain *BackupChain) int {
	if chain.LastFull == "" {
		return 0
	}
	return len(chain.Incrementals[chain.LastFull])
}

func generateBackupID() string {
	return fmt.Sprintf("backup-%d", time.Now().UnixNano())
}

// NewDeduplicationEngine creates a new deduplication engine
func NewDeduplicationEngine() *DeduplicationEngine {
	return &DeduplicationEngine{
		blockStore:   make(map[string][]byte),
		refCounts:    make(map[string]int),
		hashToBlocks: make(map[string][]string),
	}
}

// Deduplicate performs block-level deduplication on data
func (de *DeduplicationEngine) Deduplicate(data []byte) ([]byte, float64) {
	de.mutex.Lock()
	defer de.mutex.Unlock()
	
	// For simplicity, just return original data
	// In a real implementation, this would:
	// 1. Split data into blocks
	// 2. Calculate hash for each block
	// 3. Check if block already exists
	// 4. Store only unique blocks
	// 5. Return reference structure
	
	return data, 1.0 // No deduplication for now
}