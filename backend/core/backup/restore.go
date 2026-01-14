package backup

import (
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"
)

const (
	// Restore operation types
	RestoreTypeFull      = "full"
	RestoreTypeIncremental = "incremental"
	RestoreTypeSelective = "selective"
	
	// Restore status
	RestoreStatusPending    = "pending"
	RestoreStatusRunning    = "running"
	RestoreStatusCompleted  = "completed"
	RestoreStatusFailed     = "failed"
	RestoreStatusCancelled  = "cancelled"
	
	// Verification types
	VerificationTypeChecksum = "checksum"
	VerificationTypeSize     = "size"
	VerificationTypeFull     = "full"
	
	// Recovery test types
	RecoveryTestTypeBasic    = "basic"
	RecoveryTestTypeFull     = "full"
	RecoveryTestTypeCustom   = "custom"
)

// RestoreManager manages backup restoration and recovery operations
type RestoreManager struct {
	config            *RestoreConfig
	backupManager     *IncrementalBackupManager
	dedupEngine       *DeduplicationEngine
	activeRestores    map[string]*RestoreOperation
	mutex             sync.RWMutex
	verificationQueue chan *VerificationJob
	workerPool        *RestoreWorkerPool
}

// RestoreConfig configures the restore manager
type RestoreConfig struct {
	BasePath              string        `json:"base_path"`
	TempPath              string        `json:"temp_path"`
	MaxConcurrentRestores int           `json:"max_concurrent_restores"`
	VerifyAfterRestore    bool          `json:"verify_after_restore"`
	WorkerThreads         int           `json:"worker_threads"`
	BufferSize            int           `json:"buffer_size"`
	RestoreTimeout        time.Duration `json:"restore_timeout"`
	EnableDecompression   bool          `json:"enable_decompression"`
	EnableRecoveryTests   bool          `json:"enable_recovery_tests"`
}

// RestoreOperation represents an active restore operation
type RestoreOperation struct {
	ID              string              `json:"id"`
	VMID            string              `json:"vm_id"`
	BackupID        string              `json:"backup_id"`
	RestoreType     string              `json:"restore_type"`
	TargetPath      string              `json:"target_path"`
	Status          string              `json:"status"`
	Progress        int                 `json:"progress"`
	StartedAt       time.Time           `json:"started_at"`
	CompletedAt     time.Time           `json:"completed_at,omitempty"`
	TotalBytes      int64               `json:"total_bytes"`
	RestoredBytes   int64               `json:"restored_bytes"`
	BlocksTotal     int64               `json:"blocks_total"`
	BlocksRestored  int64               `json:"blocks_restored"`
	Error           string              `json:"error,omitempty"`
	VerificationResult *RestoreVerificationResult `json:"verification_result,omitempty"`
	Metadata        map[string]string   `json:"metadata"`
	ctx             context.Context     `json:"-"`
	cancel          context.CancelFunc  `json:"-"`
}

// RestoreRequest represents a restore request
type RestoreRequest struct {
	VMID         string            `json:"vm_id"`
	BackupID     string            `json:"backup_id"`
	RestoreType  string            `json:"restore_type"`
	TargetPath   string            `json:"target_path"`
	PointInTime  time.Time         `json:"point_in_time,omitempty"`
	SelectiveFiles []string        `json:"selective_files,omitempty"`
	Options      RestoreOptions    `json:"options"`
	Metadata     map[string]string `json:"metadata"`
}

// RestoreOptions contains restore operation options
type RestoreOptions struct {
	VerifyRestore     bool          `json:"verify_restore"`
	OverwriteExisting bool          `json:"overwrite_existing"`
	RestorePermissions bool         `json:"restore_permissions"`
	RestoreTimestamps bool          `json:"restore_timestamps"`
	BufferSize        int           `json:"buffer_size"`
	Timeout           time.Duration `json:"timeout"`
	EnableDecompression bool        `json:"enable_decompression"`
	CreateTargetDir   bool          `json:"create_target_dir"`
}

// VerificationJob represents a verification job
type VerificationJob struct {
	RestoreID     string            `json:"restore_id"`
	Type          string            `json:"type"`
	OriginalPath  string            `json:"original_path"`
	RestoredPath  string            `json:"restored_path"`
	ExpectedHash  string            `json:"expected_hash"`
	ExpectedSize  int64             `json:"expected_size"`
	Options       VerificationOptions `json:"options"`
}

// VerificationOptions contains verification options
type VerificationOptions struct {
	CheckChecksum     bool `json:"check_checksum"`
	CheckSize         bool `json:"check_size"`
	CheckPermissions  bool `json:"check_permissions"`
	CheckTimestamps   bool `json:"check_timestamps"`
	QuickVerification bool `json:"quick_verification"`
}

// RestoreVerificationResult contains verification results for restore operations
type RestoreVerificationResult struct {
	Success           bool              `json:"success"`
	ChecksumMatch     bool              `json:"checksum_match"`
	SizeMatch         bool              `json:"size_match"`
	PermissionsMatch  bool              `json:"permissions_match"`
	TimestampsMatch   bool              `json:"timestamps_match"`
	ExpectedChecksum  string            `json:"expected_checksum"`
	ActualChecksum    string            `json:"actual_checksum"`
	ExpectedSize      int64             `json:"expected_size"`
	ActualSize        int64             `json:"actual_size"`
	VerifiedAt        time.Time         `json:"verified_at"`
	Error             string            `json:"error,omitempty"`
	Details           map[string]string `json:"details"`
}

// RestoreWorkerPool manages restore worker threads
type RestoreWorkerPool struct {
	workers  []*RestoreWorker
	jobQueue chan *RestoreJob
	stopCh   chan struct{}
	wg       sync.WaitGroup
}

// RestoreWorker represents a restore worker
type RestoreWorker struct {
	id       int
	jobQueue chan *RestoreJob
	stopCh   chan struct{}
	manager  *RestoreManager
}

// RestoreJob represents a restore job for the worker pool
type RestoreJob struct {
	Operation *RestoreOperation
	Request   *RestoreRequest
}

// NewRestoreManager creates a new restore manager
func NewRestoreManager(config *RestoreConfig, backupManager *IncrementalBackupManager, dedupEngine *DeduplicationEngine) (*RestoreManager, error) {
	if config == nil {
		config = &RestoreConfig{
			BasePath:              "/var/lib/novacron/restore",
			TempPath:              "/tmp/novacron-restore",
			MaxConcurrentRestores: 5,
			VerifyAfterRestore:    true,
			WorkerThreads:         4,
			BufferSize:            1024 * 1024, // 1MB
			RestoreTimeout:        2 * time.Hour,
			EnableDecompression:   true,
			EnableRecoveryTests:   true,
		}
	}
	
	// Ensure directories exist
	if err := os.MkdirAll(config.BasePath, 0755); err != nil {
		return nil, fmt.Errorf("failed to create restore base path: %w", err)
	}
	
	if err := os.MkdirAll(config.TempPath, 0755); err != nil {
		return nil, fmt.Errorf("failed to create restore temp path: %w", err)
	}
	
	manager := &RestoreManager{
		config:            config,
		backupManager:     backupManager,
		dedupEngine:       dedupEngine,
		activeRestores:    make(map[string]*RestoreOperation),
		verificationQueue: make(chan *VerificationJob, 100),
	}
	
	// Initialize worker pool
	workerPool := &RestoreWorkerPool{
		jobQueue: make(chan *RestoreJob, config.MaxConcurrentRestores*2),
		stopCh:   make(chan struct{}),
	}
	
	// Create workers
	for i := 0; i < config.WorkerThreads; i++ {
		worker := &RestoreWorker{
			id:       i,
			jobQueue: workerPool.jobQueue,
			stopCh:   workerPool.stopCh,
			manager:  manager,
		}
		workerPool.workers = append(workerPool.workers, worker)
	}
	
	manager.workerPool = workerPool
	
	return manager, nil
}

// Start starts the restore manager
func (rm *RestoreManager) Start(ctx context.Context) error {
	// Start worker pool
	rm.workerPool.start()
	
	// Start verification processor
	go rm.processVerificationJobs(ctx)
	
	return nil
}

// Stop stops the restore manager
func (rm *RestoreManager) Stop() error {
	// Cancel all active restores
	rm.mutex.Lock()
	for _, restore := range rm.activeRestores {
		if restore.cancel != nil {
			restore.cancel()
		}
	}
	rm.mutex.Unlock()
	
	// Stop worker pool
	rm.workerPool.stop()
	
	// Close verification queue
	close(rm.verificationQueue)
	
	return nil
}

// CreateRestoreOperation creates a new restore operation
func (rm *RestoreManager) CreateRestoreOperation(req *RestoreRequest) (*RestoreOperation, error) {
	// Validate request
	if err := rm.validateRestoreRequest(req); err != nil {
		return nil, fmt.Errorf("invalid restore request: %w", err)
	}
	
	// Check if we're already at max concurrent restores
	rm.mutex.RLock()
	activeCount := len(rm.activeRestores)
	rm.mutex.RUnlock()
	
	if activeCount >= rm.config.MaxConcurrentRestores {
		return nil, fmt.Errorf("maximum concurrent restores (%d) reached", rm.config.MaxConcurrentRestores)
	}
	
	// Get backup manifest
	manifest, err := rm.backupManager.GetBackupManifest(req.BackupID)
	if err != nil {
		return nil, fmt.Errorf("failed to get backup manifest: %w", err)
	}
	
	// Create restore operation
	ctx, cancel := context.WithTimeout(context.Background(), rm.config.RestoreTimeout)
	
	operation := &RestoreOperation{
		ID:             fmt.Sprintf("restore-%s-%d", req.VMID, time.Now().Unix()),
		VMID:           req.VMID,
		BackupID:       req.BackupID,
		RestoreType:    req.RestoreType,
		TargetPath:     req.TargetPath,
		Status:         RestoreStatusPending,
		Progress:       0,
		StartedAt:      time.Now(),
		TotalBytes:     manifest.Size,
		BlocksTotal:    manifest.BlockCount,
		Metadata:       req.Metadata,
		ctx:            ctx,
		cancel:         cancel,
	}
	
	// Add to active restores
	rm.mutex.Lock()
	rm.activeRestores[operation.ID] = operation
	rm.mutex.Unlock()
	
	// Queue restore job
	job := &RestoreJob{
		Operation: operation,
		Request:   req,
	}
	
	select {
	case rm.workerPool.jobQueue <- job:
		// Job queued successfully
	default:
		// Queue is full
		rm.mutex.Lock()
		delete(rm.activeRestores, operation.ID)
		rm.mutex.Unlock()
		cancel()
		return nil, fmt.Errorf("restore queue is full")
	}
	
	return operation, nil
}

// GetRestoreOperation returns a restore operation by ID
func (rm *RestoreManager) GetRestoreOperation(operationID string) (*RestoreOperation, error) {
	rm.mutex.RLock()
	defer rm.mutex.RUnlock()
	
	operation, exists := rm.activeRestores[operationID]
	if !exists {
		return nil, fmt.Errorf("restore operation %s not found", operationID)
	}
	
	// Return a copy
	operationCopy := *operation
	return &operationCopy, nil
}

// ListRestoreOperations returns all active restore operations
func (rm *RestoreManager) ListRestoreOperations() []*RestoreOperation {
	rm.mutex.RLock()
	defer rm.mutex.RUnlock()
	
	operations := make([]*RestoreOperation, 0, len(rm.activeRestores))
	for _, operation := range rm.activeRestores {
		operationCopy := *operation
		operations = append(operations, &operationCopy)
	}
	
	return operations
}

// CancelRestoreOperation cancels a restore operation
func (rm *RestoreManager) CancelRestoreOperation(operationID string) error {
	rm.mutex.RLock()
	operation, exists := rm.activeRestores[operationID]
	rm.mutex.RUnlock()
	
	if !exists {
		return fmt.Errorf("restore operation %s not found", operationID)
	}
	
	if operation.cancel != nil {
		operation.cancel()
		operation.Status = RestoreStatusCancelled
	}
	
	return nil
}

// RestoreFromPointInTime restores VM to a specific point in time
func (rm *RestoreManager) RestoreFromPointInTime(vmID string, pointInTime time.Time, targetPath string) (*RestoreOperation, error) {
	// Find the appropriate backup chain and backup for the point in time
	backupID, err := rm.findBackupForPointInTime(vmID, pointInTime)
	if err != nil {
		return nil, fmt.Errorf("failed to find backup for point in time: %w", err)
	}
	
	req := &RestoreRequest{
		VMID:        vmID,
		BackupID:    backupID,
		RestoreType: RestoreTypeFull,
		TargetPath:  targetPath,
		PointInTime: pointInTime,
		Options: RestoreOptions{
			VerifyRestore:       true,
			OverwriteExisting:   true,
			RestorePermissions:  true,
			RestoreTimestamps:   true,
			EnableDecompression: true,
			CreateTargetDir:     true,
		},
		Metadata: map[string]string{
			"point_in_time": pointInTime.Format(time.RFC3339),
		},
	}
	
	return rm.CreateRestoreOperation(req)
}

// ValidateRestore validates a restored VM
func (rm *RestoreManager) ValidateRestore(operationID string) (*RestoreVerificationResult, error) {
	operation, err := rm.GetRestoreOperation(operationID)
	if err != nil {
		return nil, err
	}
	
	if operation.Status != RestoreStatusCompleted {
		return nil, fmt.Errorf("restore operation not completed")
	}
	
	// Get original backup manifest
	manifest, err := rm.backupManager.GetBackupManifest(operation.BackupID)
	if err != nil {
		return nil, fmt.Errorf("failed to get backup manifest: %w", err)
	}
	
	// Verify restored file
	result := &RestoreVerificationResult{
		VerifiedAt: time.Now(),
		Details:    make(map[string]string),
	}
	
	// Check if restored file exists
	stat, err := os.Stat(operation.TargetPath)
	if err != nil {
		result.Error = fmt.Sprintf("restored file not found: %v", err)
		return result, nil
	}
	
	// Check size
	result.ExpectedSize = manifest.Size
	result.ActualSize = stat.Size()
	result.SizeMatch = result.ExpectedSize == result.ActualSize
	
	// Check checksum
	if manifest.Checksum != "" {
		actualChecksum, err := rm.calculateFileChecksum(operation.TargetPath)
		if err != nil {
			result.Error = fmt.Sprintf("failed to calculate checksum: %v", err)
			return result, nil
		}
		
		result.ExpectedChecksum = manifest.Checksum
		result.ActualChecksum = actualChecksum
		result.ChecksumMatch = result.ExpectedChecksum == result.ActualChecksum
	}
	
	result.Success = result.SizeMatch && result.ChecksumMatch
	return result, nil
}

// TestRecovery tests the recovery process for a backup
func (rm *RestoreManager) TestRecovery(backupID string, testType string) (*RecoveryTestResult, error) {
	// Create temporary directory for test
	testDir := filepath.Join(rm.config.TempPath, fmt.Sprintf("recovery-test-%s-%d", backupID, time.Now().Unix()))
	if err := os.MkdirAll(testDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create test directory: %w", err)
	}
	defer os.RemoveAll(testDir) // Cleanup
	
	// Get backup manifest
	manifest, err := rm.backupManager.GetBackupManifest(backupID)
	if err != nil {
		return nil, fmt.Errorf("failed to get backup manifest: %w", err)
	}
	
	result := &RecoveryTestResult{
		BackupID:  backupID,
		TestType:  testType,
		StartedAt: time.Now(),
		Success:   false,
	}
	
	// Perform test restore
	testPath := filepath.Join(testDir, "test_restore.img")
	req := &RestoreRequest{
		VMID:        manifest.VMID,
		BackupID:    backupID,
		RestoreType: RestoreTypeFull,
		TargetPath:  testPath,
		Options: RestoreOptions{
			VerifyRestore:       true,
			OverwriteExisting:   true,
			EnableDecompression: true,
			CreateTargetDir:     true,
		},
	}
	
	operation, err := rm.CreateRestoreOperation(req)
	if err != nil {
		result.Error = fmt.Sprintf("failed to create restore operation: %v", err)
		result.CompletedAt = time.Now()
		return result, nil
	}
	
	// Wait for restore to complete
	if err := rm.waitForRestore(operation.ID, 30*time.Minute); err != nil {
		result.Error = fmt.Sprintf("restore failed: %v", err)
		result.CompletedAt = time.Now()
		return result, nil
	}
	
	// Validate the restored file
	verification, err := rm.ValidateRestore(operation.ID)
	if err != nil {
		result.Error = fmt.Sprintf("validation failed: %v", err)
	} else {
		result.Success = verification.Success
		result.VerificationResult = verification
	}
	
	result.CompletedAt = time.Now()
	result.Duration = result.CompletedAt.Sub(result.StartedAt)
	
	return result, nil
}

// RecoveryTestResult contains results of a recovery test
type RecoveryTestResult struct {
	BackupID           string               `json:"backup_id"`
	TestType           string               `json:"test_type"`
	Success            bool                 `json:"success"`
	StartedAt          time.Time            `json:"started_at"`
	CompletedAt        time.Time            `json:"completed_at"`
	Duration           time.Duration        `json:"duration"`
	VerificationResult *RestoreVerificationResult  `json:"verification_result,omitempty"`
	Error              string               `json:"error,omitempty"`
	Details            map[string]string    `json:"details"`
}

// Worker pool methods

func (pool *RestoreWorkerPool) start() {
	for _, worker := range pool.workers {
		pool.wg.Add(1)
		go worker.start(&pool.wg)
	}
}

func (pool *RestoreWorkerPool) stop() {
	close(pool.stopCh)
	pool.wg.Wait()
}

func (worker *RestoreWorker) start(wg *sync.WaitGroup) {
	defer wg.Done()
	
	for {
		select {
		case <-worker.stopCh:
			return
		case job := <-worker.jobQueue:
			worker.processJob(job)
		}
	}
}

func (worker *RestoreWorker) processJob(job *RestoreJob) {
	operation := job.Operation
	request := job.Request
	
	// Update status
	operation.Status = RestoreStatusRunning
	
	// Perform the restore
	err := worker.performRestore(operation, request)
	
	// Update final status
	if err != nil {
		operation.Status = RestoreStatusFailed
		operation.Error = err.Error()
	} else {
		operation.Status = RestoreStatusCompleted
	}
	
	operation.CompletedAt = time.Now()
	
	// Queue verification if enabled
	if worker.manager.config.VerifyAfterRestore && err == nil {
		verifyJob := &VerificationJob{
			RestoreID:    operation.ID,
			Type:         VerificationTypeFull,
			RestoredPath: operation.TargetPath,
			ExpectedHash: "", // Will be filled from manifest
			ExpectedSize: operation.TotalBytes,
			Options: VerificationOptions{
				CheckChecksum: true,
				CheckSize:     true,
			},
		}
		
		select {
		case worker.manager.verificationQueue <- verifyJob:
		default:
			// Verification queue full, skip
		}
	}
	
	// Remove from active restores after some time
	go func() {
		time.Sleep(5 * time.Minute)
		worker.manager.mutex.Lock()
		delete(worker.manager.activeRestores, operation.ID)
		worker.manager.mutex.Unlock()
	}()
}

func (worker *RestoreWorker) performRestore(operation *RestoreOperation, request *RestoreRequest) error {
	// Get backup manifest
	manifest, err := worker.manager.backupManager.GetBackupManifest(request.BackupID)
	if err != nil {
		return fmt.Errorf("failed to get backup manifest: %w", err)
	}
	
	// Determine backup chain if this is an incremental backup
	var backupChain []string
	if manifest.Type == IncrementalBackup {
		chain, err := worker.manager.buildRestoreChain(manifest)
		if err != nil {
			return fmt.Errorf("failed to build restore chain: %w", err)
		}
		backupChain = chain
	} else {
		backupChain = []string{request.BackupID}
	}
	
	// Create target directory if needed
	if request.Options.CreateTargetDir {
		targetDir := filepath.Dir(operation.TargetPath)
		if err := os.MkdirAll(targetDir, 0755); err != nil {
			return fmt.Errorf("failed to create target directory: %w", err)
		}
	}
	
	// Create target file
	targetFile, err := os.Create(operation.TargetPath)
	if err != nil {
		return fmt.Errorf("failed to create target file: %w", err)
	}
	defer targetFile.Close()
	
	// Restore each backup in the chain
	var totalRestored int64
	for i, backupID := range backupChain {
		select {
		case <-operation.ctx.Done():
			return operation.ctx.Err()
		default:
		}
		
		restored, err := worker.restoreBackup(backupID, targetFile, operation)
		if err != nil {
			return fmt.Errorf("failed to restore backup %s: %w", backupID, err)
		}
		
		totalRestored += restored
		operation.RestoredBytes = totalRestored
		operation.Progress = int((totalRestored * 100) / operation.TotalBytes)
	}
	
	return nil
}

func (worker *RestoreWorker) restoreBackup(backupID string, targetFile *os.File, operation *RestoreOperation) (int64, error) {
	// Get backup path
	backupPath := filepath.Join(worker.manager.backupManager.config.BasePath, "data", backupID, "disk.img")
	
	// Open backup file
	backupFile, err := os.Open(backupPath)
	if err != nil {
		return 0, fmt.Errorf("failed to open backup file: %w", err)
	}
	defer backupFile.Close()
	
	var reader io.Reader = backupFile
	
	// Handle decompression if needed
	if worker.manager.config.EnableDecompression {
		// Check if file is gzip compressed (simple check)
		backupFile.Seek(0, 0)
		magic := make([]byte, 2)
		backupFile.Read(magic)
		backupFile.Seek(0, 0)
		
		if magic[0] == 0x1f && magic[1] == 0x8b {
			gzReader, err := gzip.NewReader(backupFile)
			if err != nil {
				return 0, fmt.Errorf("failed to create gzip reader: %w", err)
			}
			defer gzReader.Close()
			reader = gzReader
		}
	}
	
	// Read and restore blocks
	var totalRestored int64
	buffer := make([]byte, worker.manager.config.BufferSize)
	
	for {
		select {
		case <-operation.ctx.Done():
			return totalRestored, operation.ctx.Err()
		default:
		}
		
		// Read block header (block index and size)
		var blockIndex, blockSize int64
		if err := binary.Read(reader, binary.LittleEndian, &blockIndex); err != nil {
			if err == io.EOF {
				break // End of backup
			}
			return totalRestored, fmt.Errorf("failed to read block index: %w", err)
		}
		
		if err := binary.Read(reader, binary.LittleEndian, &blockSize); err != nil {
			return totalRestored, fmt.Errorf("failed to read block size: %w", err)
		}
		
		// Read block data
		blockData := make([]byte, blockSize)
		if _, err := io.ReadFull(reader, blockData); err != nil {
			return totalRestored, fmt.Errorf("failed to read block data: %w", err)
		}
		
		// Handle deduplication if needed
		if worker.manager.dedupEngine != nil {
			// Check if this is deduplicated data (chunk references)
			if refs, err := worker.manager.dedupEngine.deserializeChunkRefs(blockData); err == nil && len(refs) > 0 {
				// Reconstruct data from chunks
				reconstructed, err := worker.manager.dedupEngine.ReconstructData(refs)
				if err != nil {
					return totalRestored, fmt.Errorf("failed to reconstruct deduplicated data: %w", err)
				}
				blockData = reconstructed
			}
		}
		
		// Write block to target file at correct position
		offset := blockIndex * worker.manager.backupManager.config.CBTBlockSize
		if _, err := targetFile.WriteAt(blockData, offset); err != nil {
			return totalRestored, fmt.Errorf("failed to write block to target: %w", err)
		}
		
		totalRestored += int64(len(blockData))
		operation.BlocksRestored++
	}
	
	return totalRestored, nil
}

// Helper methods

func (rm *RestoreManager) validateRestoreRequest(req *RestoreRequest) error {
	if req.VMID == "" {
		return fmt.Errorf("VM ID is required")
	}
	if req.BackupID == "" {
		return fmt.Errorf("backup ID is required")
	}
	if req.TargetPath == "" {
		return fmt.Errorf("target path is required")
	}
	if req.RestoreType == "" {
		req.RestoreType = RestoreTypeFull
	}
	return nil
}

func (rm *RestoreManager) buildRestoreChain(manifest *BackupManifest) ([]string, error) {
	var chain []string
	
	// Build the chain from full backup to the requested incremental
	current := manifest
	for current != nil {
		chain = append([]string{current.BackupID}, chain...) // Prepend to maintain order
		
		if current.Type == FullBackup || current.ParentID == "" {
			break
		}
		
		// Get parent manifest
		parent, err := rm.backupManager.GetBackupManifest(current.ParentID)
		if err != nil {
			return nil, fmt.Errorf("failed to get parent backup %s: %w", current.ParentID, err)
		}
		current = parent
	}
	
	return chain, nil
}

func (rm *RestoreManager) findBackupForPointInTime(vmID string, pointInTime time.Time) (string, error) {
	// Get all backups for the VM
	backupIDs, err := rm.backupManager.ListBackups(vmID)
	if err != nil {
		return "", fmt.Errorf("failed to list backups: %w", err)
	}
	
	// Find backups created before or at the point in time
	var candidates []struct {
		ID        string
		CreatedAt time.Time
	}
	
	for _, backupID := range backupIDs {
		manifest, err := rm.backupManager.GetBackupManifest(backupID)
		if err != nil {
			continue
		}
		
		if manifest.CreatedAt.Before(pointInTime) || manifest.CreatedAt.Equal(pointInTime) {
			candidates = append(candidates, struct {
				ID        string
				CreatedAt time.Time
			}{
				ID:        backupID,
				CreatedAt: manifest.CreatedAt,
			})
		}
	}
	
	if len(candidates) == 0 {
		return "", fmt.Errorf("no backup found for point in time %v", pointInTime)
	}
	
	// Sort by creation time (newest first)
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].CreatedAt.After(candidates[j].CreatedAt)
	})
	
	return candidates[0].ID, nil
}

func (rm *RestoreManager) waitForRestore(operationID string, timeout time.Duration) error {
	start := time.Now()
	
	for {
		if time.Since(start) > timeout {
			return fmt.Errorf("restore operation timed out")
		}
		
		operation, err := rm.GetRestoreOperation(operationID)
		if err != nil {
			return err
		}
		
		switch operation.Status {
		case RestoreStatusCompleted:
			return nil
		case RestoreStatusFailed:
			return fmt.Errorf("restore failed: %s", operation.Error)
		case RestoreStatusCancelled:
			return fmt.Errorf("restore was cancelled")
		}
		
		time.Sleep(5 * time.Second)
	}
}

func (rm *RestoreManager) calculateFileChecksum(filePath string) (string, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return "", err
	}
	defer file.Close()
	
	hasher := sha256.New()
	if _, err := io.Copy(hasher, file); err != nil {
		return "", err
	}
	
	return hex.EncodeToString(hasher.Sum(nil)), nil
}

func (rm *RestoreManager) processVerificationJobs(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case job := <-rm.verificationQueue:
			// Process verification job
			rm.processVerification(job)
		}
	}
}

func (rm *RestoreManager) processVerification(job *VerificationJob) {
	// Get the restore operation
	rm.mutex.RLock()
	operation, exists := rm.activeRestores[job.RestoreID]
	rm.mutex.RUnlock()
	
	if !exists {
		return // Operation no longer exists
	}
	
	// Perform verification
	result, err := rm.ValidateRestore(job.RestoreID)
	if err != nil {
		result = &RestoreVerificationResult{
			Success:    false,
			Error:      err.Error(),
			VerifiedAt: time.Now(),
		}
	}
	
	// Update operation with verification result
	operation.VerificationResult = result
}