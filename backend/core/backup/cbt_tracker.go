package backup

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sync"
	"time"
)

// CBTBlockSize represents the size of blocks tracked by CBT (4KB default)
const CBTBlockSize = 4096

// CBTTracker manages Changed Block Tracking for incremental backups
type CBTTracker struct {
	// blockMap tracks the hash of each block for a resource
	blockMap map[string]map[int64]string // resourceID -> blockIndex -> hash
	
	// changeLog tracks when blocks were last changed
	changeLog map[string]map[int64]time.Time // resourceID -> blockIndex -> changeTime
	
	// baselineTimestamp tracks when the baseline was established
	baselineTimestamp map[string]time.Time // resourceID -> timestamp
	
	// mutex protects concurrent access
	mutex sync.RWMutex
	
	// storage for persistent CBT data
	storage CBTStorage
}

// CBTBlock represents a tracked block
type CBTBlock struct {
	ResourceID string    `json:"resource_id"`
	BlockIndex int64     `json:"block_index"`
	Hash       string    `json:"hash"`
	Size       int64     `json:"size"`
	Changed    bool      `json:"changed"`
	Timestamp  time.Time `json:"timestamp"`
}

// CBTSnapshot represents a point-in-time view of tracked blocks
type CBTSnapshot struct {
	ID         string               `json:"id"`
	ResourceID string               `json:"resource_id"`
	Timestamp  time.Time           `json:"timestamp"`
	TotalSize  int64               `json:"total_size"`
	BlockCount int64               `json:"block_count"`
	Blocks     map[int64]CBTBlock  `json:"blocks"`
}

// CBTDelta represents the changes between two snapshots
type CBTDelta struct {
	SourceSnapshotID string      `json:"source_snapshot_id"`
	TargetSnapshotID string      `json:"target_snapshot_id"`
	ChangedBlocks    []CBTBlock  `json:"changed_blocks"`
	DeletedBlocks    []int64     `json:"deleted_blocks"`
	NewBlocks        []CBTBlock  `json:"new_blocks"`
	TotalChangedSize int64       `json:"total_changed_size"`
	ChangeRatio      float64     `json:"change_ratio"`
}

// CBTStorage defines the interface for persistent CBT data storage
type CBTStorage interface {
	SaveSnapshot(ctx context.Context, snapshot *CBTSnapshot) error
	LoadSnapshot(ctx context.Context, snapshotID string) (*CBTSnapshot, error)
	ListSnapshots(ctx context.Context, resourceID string) ([]*CBTSnapshot, error)
	DeleteSnapshot(ctx context.Context, snapshotID string) error
	SaveDelta(ctx context.Context, delta *CBTDelta) error
	LoadDelta(ctx context.Context, sourceID, targetID string) (*CBTDelta, error)
}

// NewCBTTracker creates a new Changed Block Tracker
func NewCBTTracker(storage CBTStorage) *CBTTracker {
	return &CBTTracker{
		blockMap:          make(map[string]map[int64]string),
		changeLog:         make(map[string]map[int64]time.Time),
		baselineTimestamp: make(map[string]time.Time),
		storage:           storage,
	}
}

// InitializeResource initializes CBT tracking for a resource
func (cbt *CBTTracker) InitializeResource(ctx context.Context, resourceID string, resourceSize int64) error {
	cbt.mutex.Lock()
	defer cbt.mutex.Unlock()
	
	// Initialize maps for the resource
	cbt.blockMap[resourceID] = make(map[int64]string)
	cbt.changeLog[resourceID] = make(map[int64]time.Time)
	cbt.baselineTimestamp[resourceID] = time.Now()
	
	// Create initial baseline snapshot
	blockCount := (resourceSize + CBTBlockSize - 1) / CBTBlockSize
	snapshot := &CBTSnapshot{
		ID:         generateCBTSnapshotID(resourceID),
		ResourceID: resourceID,
		Timestamp:  time.Now(),
		TotalSize:  resourceSize,
		BlockCount: blockCount,
		Blocks:     make(map[int64]CBTBlock),
	}
	
	// Initialize all blocks as baseline (empty hash indicates baseline)
	for i := int64(0); i < blockCount; i++ {
		block := CBTBlock{
			ResourceID: resourceID,
			BlockIndex: i,
			Hash:       "", // Empty hash for baseline
			Size:       min(CBTBlockSize, resourceSize-i*CBTBlockSize),
			Changed:    false,
			Timestamp:  snapshot.Timestamp,
		}
		snapshot.Blocks[i] = block
		cbt.blockMap[resourceID][i] = ""
		cbt.changeLog[resourceID][i] = snapshot.Timestamp
	}
	
	return cbt.storage.SaveSnapshot(ctx, snapshot)
}

// UpdateBlock updates a block and marks it as changed
func (cbt *CBTTracker) UpdateBlock(ctx context.Context, resourceID string, blockIndex int64, blockData []byte) error {
	cbt.mutex.Lock()
	defer cbt.mutex.Unlock()
	
	// Calculate new block hash
	hash := sha256.Sum256(blockData)
	hashStr := hex.EncodeToString(hash[:])
	
	// Check if resource is tracked
	if _, exists := cbt.blockMap[resourceID]; !exists {
		return fmt.Errorf("resource %s is not being tracked", resourceID)
	}
	
	// Get current hash
	currentHash, exists := cbt.blockMap[resourceID][blockIndex]
	if !exists {
		// New block
		cbt.blockMap[resourceID][blockIndex] = hashStr
		cbt.changeLog[resourceID][blockIndex] = time.Now()
		return nil
	}
	
	// Check if block actually changed
	if currentHash != hashStr {
		cbt.blockMap[resourceID][blockIndex] = hashStr
		cbt.changeLog[resourceID][blockIndex] = time.Now()
	}
	
	return nil
}

// GetChangedBlocks returns blocks that changed since the given timestamp
func (cbt *CBTTracker) GetChangedBlocks(ctx context.Context, resourceID string, since time.Time) ([]CBTBlock, error) {
	cbt.mutex.RLock()
	defer cbt.mutex.RUnlock()
	
	// Check if resource is tracked
	changeLog, exists := cbt.changeLog[resourceID]
	if !exists {
		return nil, fmt.Errorf("resource %s is not being tracked", resourceID)
	}
	
	var changedBlocks []CBTBlock
	for blockIndex, changeTime := range changeLog {
		if changeTime.After(since) {
			hash := cbt.blockMap[resourceID][blockIndex]
			block := CBTBlock{
				ResourceID: resourceID,
				BlockIndex: blockIndex,
				Hash:       hash,
				Size:       CBTBlockSize, // Simplified - real implementation would track actual size
				Changed:    true,
				Timestamp:  changeTime,
			}
			changedBlocks = append(changedBlocks, block)
		}
	}
	
	return changedBlocks, nil
}

// CreateSnapshot creates a point-in-time snapshot of tracked blocks
func (cbt *CBTTracker) CreateSnapshot(ctx context.Context, resourceID string) (*CBTSnapshot, error) {
	cbt.mutex.RLock()
	defer cbt.mutex.RUnlock()
	
	// Check if resource is tracked
	blockMap, exists := cbt.blockMap[resourceID]
	if !exists {
		return nil, fmt.Errorf("resource %s is not being tracked", resourceID)
	}
	
	changeLog := cbt.changeLog[resourceID]
	
	// Create snapshot
	snapshot := &CBTSnapshot{
		ID:         generateCBTSnapshotID(resourceID),
		ResourceID: resourceID,
		Timestamp:  time.Now(),
		BlockCount: int64(len(blockMap)),
		Blocks:     make(map[int64]CBTBlock),
	}
	
	var totalSize int64
	for blockIndex, hash := range blockMap {
		changeTime := changeLog[blockIndex]
		block := CBTBlock{
			ResourceID: resourceID,
			BlockIndex: blockIndex,
			Hash:       hash,
			Size:       CBTBlockSize,
			Changed:    false, // Will be determined during delta calculation
			Timestamp:  changeTime,
		}
		snapshot.Blocks[blockIndex] = block
		totalSize += block.Size
	}
	
	snapshot.TotalSize = totalSize
	
	// Save snapshot to storage
	if err := cbt.storage.SaveSnapshot(ctx, snapshot); err != nil {
		return nil, err
	}
	
	return snapshot, nil
}

// CalculateDelta calculates the difference between two snapshots
func (cbt *CBTTracker) CalculateDelta(ctx context.Context, sourceSnapshotID, targetSnapshotID string) (*CBTDelta, error) {
	// Load snapshots
	sourceSnapshot, err := cbt.storage.LoadSnapshot(ctx, sourceSnapshotID)
	if err != nil {
		return nil, fmt.Errorf("failed to load source snapshot: %w", err)
	}
	
	targetSnapshot, err := cbt.storage.LoadSnapshot(ctx, targetSnapshotID)
	if err != nil {
		return nil, fmt.Errorf("failed to load target snapshot: %w", err)
	}
	
	// Ensure snapshots are for the same resource
	if sourceSnapshot.ResourceID != targetSnapshot.ResourceID {
		return nil, fmt.Errorf("snapshots are for different resources")
	}
	
	delta := &CBTDelta{
		SourceSnapshotID: sourceSnapshotID,
		TargetSnapshotID: targetSnapshotID,
		ChangedBlocks:    make([]CBTBlock, 0),
		DeletedBlocks:    make([]int64, 0),
		NewBlocks:        make([]CBTBlock, 0),
	}
	
	var totalChangedSize int64
	
	// Find changed and new blocks
	for blockIndex, targetBlock := range targetSnapshot.Blocks {
		if sourceBlock, exists := sourceSnapshot.Blocks[blockIndex]; exists {
			// Block exists in both snapshots - check if changed
			if sourceBlock.Hash != targetBlock.Hash {
				targetBlock.Changed = true
				delta.ChangedBlocks = append(delta.ChangedBlocks, targetBlock)
				totalChangedSize += targetBlock.Size
			}
		} else {
			// Block is new in target snapshot
			targetBlock.Changed = true
			delta.NewBlocks = append(delta.NewBlocks, targetBlock)
			totalChangedSize += targetBlock.Size
		}
	}
	
	// Find deleted blocks
	for blockIndex := range sourceSnapshot.Blocks {
		if _, exists := targetSnapshot.Blocks[blockIndex]; !exists {
			delta.DeletedBlocks = append(delta.DeletedBlocks, blockIndex)
		}
	}
	
	delta.TotalChangedSize = totalChangedSize
	
	// Calculate change ratio
	if targetSnapshot.TotalSize > 0 {
		delta.ChangeRatio = float64(totalChangedSize) / float64(targetSnapshot.TotalSize)
	}
	
	// Save delta to storage
	if err := cbt.storage.SaveDelta(ctx, delta); err != nil {
		return nil, err
	}
	
	return delta, nil
}

// OptimizeForBackup returns the optimal blocks to backup based on CBT data
func (cbt *CBTTracker) OptimizeForBackup(ctx context.Context, resourceID string, backupType BackupType, lastBackupTime *time.Time) ([]CBTBlock, error) {
	switch backupType {
	case FullBackup:
		// For full backups, return all blocks
		return cbt.getAllBlocks(ctx, resourceID)
	case IncrementalBackup:
		// For incremental backups, return blocks changed since last backup
		if lastBackupTime == nil {
			return nil, fmt.Errorf("incremental backup requires last backup timestamp")
		}
		return cbt.GetChangedBlocks(ctx, resourceID, *lastBackupTime)
	case DifferentialBackup:
		// For differential backups, return blocks changed since baseline
		cbt.mutex.RLock()
		baseline := cbt.baselineTimestamp[resourceID]
		cbt.mutex.RUnlock()
		return cbt.GetChangedBlocks(ctx, resourceID, baseline)
	default:
		return nil, fmt.Errorf("unsupported backup type: %s", backupType)
	}
}

// GetChangeRate returns the change rate for a resource (blocks changed per hour)
func (cbt *CBTTracker) GetChangeRate(ctx context.Context, resourceID string, duration time.Duration) (float64, error) {
	cbt.mutex.RLock()
	defer cbt.mutex.RUnlock()
	
	changeLog, exists := cbt.changeLog[resourceID]
	if !exists {
		return 0, fmt.Errorf("resource %s is not being tracked", resourceID)
	}
	
	// Count changes in the specified duration
	since := time.Now().Add(-duration)
	var changeCount int
	for _, changeTime := range changeLog {
		if changeTime.After(since) {
			changeCount++
		}
	}
	
	// Calculate rate per hour
	hours := duration.Hours()
	if hours > 0 {
		return float64(changeCount) / hours, nil
	}
	
	return 0, nil
}

// EstimateBackupSize estimates the size of a backup based on CBT data
func (cbt *CBTTracker) EstimateBackupSize(ctx context.Context, resourceID string, backupType BackupType, lastBackupTime *time.Time) (int64, error) {
	blocks, err := cbt.OptimizeForBackup(ctx, resourceID, backupType, lastBackupTime)
	if err != nil {
		return 0, err
	}
	
	var totalSize int64
	for _, block := range blocks {
		totalSize += block.Size
	}
	
	return totalSize, nil
}

// Cleanup removes old CBT data for a resource
func (cbt *CBTTracker) Cleanup(ctx context.Context, resourceID string) error {
	cbt.mutex.Lock()
	defer cbt.mutex.Unlock()
	
	// Remove from memory
	delete(cbt.blockMap, resourceID)
	delete(cbt.changeLog, resourceID)
	delete(cbt.baselineTimestamp, resourceID)
	
	// Clean up snapshots in storage
	snapshots, err := cbt.storage.ListSnapshots(ctx, resourceID)
	if err != nil {
		return err
	}
	
	for _, snapshot := range snapshots {
		if err := cbt.storage.DeleteSnapshot(ctx, snapshot.ID); err != nil {
			// Log error but continue cleanup
			continue
		}
	}
	
	return nil
}

// getAllBlocks returns all blocks for a resource
func (cbt *CBTTracker) getAllBlocks(ctx context.Context, resourceID string) ([]CBTBlock, error) {
	cbt.mutex.RLock()
	defer cbt.mutex.RUnlock()
	
	blockMap, exists := cbt.blockMap[resourceID]
	if !exists {
		return nil, fmt.Errorf("resource %s is not being tracked", resourceID)
	}
	
	changeLog := cbt.changeLog[resourceID]
	
	var allBlocks []CBTBlock
	for blockIndex, hash := range blockMap {
		changeTime := changeLog[blockIndex]
		block := CBTBlock{
			ResourceID: resourceID,
			BlockIndex: blockIndex,
			Hash:       hash,
			Size:       CBTBlockSize,
			Changed:    true, // Mark as changed for full backup
			Timestamp:  changeTime,
		}
		allBlocks = append(allBlocks, block)
	}
	
	return allBlocks, nil
}

// generateCBTSnapshotID generates a unique snapshot ID
func generateCBTSnapshotID(resourceID string) string {
	hash := sha256.Sum256([]byte(fmt.Sprintf("%s-%d", resourceID, time.Now().UnixNano())))
	return fmt.Sprintf("cbt-snapshot-%s", hex.EncodeToString(hash[:8]))
}

// min returns the minimum of two int64 values
func min(a, b int64) int64 {
	if a < b {
		return a
	}
	return b
}

// LocalCBTStorage implements CBTStorage interface using local file system
type LocalCBTStorage struct {
	basePath string
	mutex    sync.RWMutex
}

// NewLocalCBTStorage creates a new local CBT storage
func NewLocalCBTStorage(basePath string) *LocalCBTStorage {
	return &LocalCBTStorage{
		basePath: basePath,
	}
}

// SaveSnapshot saves a snapshot to local storage
func (s *LocalCBTStorage) SaveSnapshot(ctx context.Context, snapshot *CBTSnapshot) error {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	
	data, err := json.Marshal(snapshot)
	if err != nil {
		return err
	}
	
	// In a real implementation, this would write to a file
	// For now, just validate the operation
	if len(data) == 0 {
		return fmt.Errorf("failed to serialize snapshot")
	}
	
	return nil
}

// LoadSnapshot loads a snapshot from local storage
func (s *LocalCBTStorage) LoadSnapshot(ctx context.Context, snapshotID string) (*CBTSnapshot, error) {
	s.mutex.RLock()
	defer s.mutex.RUnlock()
	
	// In a real implementation, this would read from a file
	// For now, return a minimal snapshot for testing
	return &CBTSnapshot{
		ID:        snapshotID,
		Timestamp: time.Now(),
		Blocks:    make(map[int64]CBTBlock),
	}, nil
}

// ListSnapshots lists all snapshots for a resource
func (s *LocalCBTStorage) ListSnapshots(ctx context.Context, resourceID string) ([]*CBTSnapshot, error) {
	s.mutex.RLock()
	defer s.mutex.RUnlock()
	
	// Return empty list for now
	return []*CBTSnapshot{}, nil
}

// DeleteSnapshot deletes a snapshot
func (s *LocalCBTStorage) DeleteSnapshot(ctx context.Context, snapshotID string) error {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	
	// In a real implementation, this would delete the file
	return nil
}

// SaveDelta saves a delta to local storage
func (s *LocalCBTStorage) SaveDelta(ctx context.Context, delta *CBTDelta) error {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	
	data, err := json.Marshal(delta)
	if err != nil {
		return err
	}
	
	// In a real implementation, this would write to a file
	if len(data) == 0 {
		return fmt.Errorf("failed to serialize delta")
	}
	
	return nil
}

// LoadDelta loads a delta from local storage
func (s *LocalCBTStorage) LoadDelta(ctx context.Context, sourceID, targetID string) (*CBTDelta, error) {
	s.mutex.RLock()
	defer s.mutex.RUnlock()
	
	// In a real implementation, this would read from a file
	return &CBTDelta{
		SourceSnapshotID: sourceID,
		TargetSnapshotID: targetID,
		ChangedBlocks:    make([]CBTBlock, 0),
		DeletedBlocks:    make([]int64, 0),
		NewBlocks:        make([]CBTBlock, 0),
	}, nil
}