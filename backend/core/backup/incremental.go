package backup

import (
	"bufio"
	"bytes"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/klauspost/compress/zstd"
)

const (
	// CBT block size for change tracking (4KB)
	CBTBlockSize = 4096
	
	// Maximum number of incremental backups before forcing a full backup
	MaxIncrementals = 10
	
	// CBT metadata file extension
	CBTMetaExt = ".cbt"
	
	// Backup manifest file name
	BackupManifestFile = "backup_manifest.json"
)

// CBTTracker implements Changed Block Tracking for incremental backups
type CBTTracker struct {
	vmID     string
	basePath string
	mutex    sync.RWMutex
	
	// Block change tracking data
	blocks      map[int64]*BlockInfo
	totalBlocks int64
	blockSize   int64
	
	// Metadata
	version    int64
	generation int64
	createdAt  time.Time
	updatedAt  time.Time
}

// VMID returns the VM ID for this tracker
func (t *CBTTracker) VMID() string {
	t.mutex.RLock()
	defer t.mutex.RUnlock()
	return t.vmID
}

// TotalBlocks returns the total number of blocks
func (t *CBTTracker) TotalBlocks() int64 {
	t.mutex.RLock()
	defer t.mutex.RUnlock()
	return t.totalBlocks
}

// BlockSize returns the block size
func (t *CBTTracker) BlockSize() int64 {
	t.mutex.RLock()
	defer t.mutex.RUnlock()
	return t.blockSize
}

// BlockInfo contains information about a tracked block
type BlockInfo struct {
	Index     int64     `json:"index"`
	Checksum  string    `json:"checksum"`
	Changed   bool      `json:"changed"`
	LastWrite time.Time `json:"last_write"`
	Size      int64     `json:"size"`
}

// CBTMetadata contains CBT tracking metadata
type CBTMetadata struct {
	VMID        string               `json:"vm_id"`
	Version     int64                `json:"version"`
	Generation  int64                `json:"generation"`
	TotalBlocks int64                `json:"total_blocks"`
	BlockSize   int64                `json:"block_size"`
	CreatedAt   time.Time            `json:"created_at"`
	UpdatedAt   time.Time            `json:"updated_at"`
	Blocks      map[int64]*BlockInfo `json:"blocks"`
}

// IncrementalBackupManager manages incremental backups with CBT
type IncrementalBackupManager struct {
	config     *IncrementalConfig
	trackers   map[string]*CBTTracker
	mutex      sync.RWMutex
	compressor *zstd.Encoder
	
	// Deduplication engine
	dedupEngine *DeduplicationEngine
	
	// Backup chain management
	chains map[string]*BackupChain
}

// IncrementalConfig configures the incremental backup system
type IncrementalConfig struct {
	BasePath          string        `json:"base_path"`
	EnableCompression bool          `json:"enable_compression"`
	EnableDedup       bool          `json:"enable_dedup"`
	CompressionLevel  int           `json:"compression_level"`
	MaxIncrementals   int           `json:"max_incrementals"`
	CBTBlockSize      int64         `json:"cbt_block_size"`
	WorkerThreads     int           `json:"worker_threads"`
	BufferSize        int           `json:"buffer_size"`
}

// BackupChain tracks a chain of incremental backups
type BackupChain struct {
	VMID         string            `json:"vm_id"`
	FullBackupID string            `json:"full_backup_id"`
	Incrementals []string          `json:"incrementals"`
	CreatedAt    time.Time         `json:"created_at"`
	UpdatedAt    time.Time         `json:"updated_at"`
	Metadata     map[string]string `json:"metadata"`
}

// BackupManifest describes the contents of a backup
type BackupManifest struct {
	BackupID     string            `json:"backup_id"`
	VMID         string            `json:"vm_id"`
	Type         BackupType        `json:"type"`
	ParentID     string            `json:"parent_id,omitempty"`
	Size         int64             `json:"size"`
	CompressedSize int64           `json:"compressed_size"`
	BlockCount   int64             `json:"block_count"`
	ChangedBlocks int64            `json:"changed_blocks"`
	CreatedAt    time.Time         `json:"created_at"`
	Files        []BackupFile      `json:"files"`
	CBTVersion   int64             `json:"cbt_version"`
	Checksum     string            `json:"checksum"`
	Metadata     map[string]string `json:"metadata"`
}

// BackupFile represents a file in the backup
type BackupFile struct {
	Path         string    `json:"path"`
	Size         int64     `json:"size"`
	Checksum     string    `json:"checksum"`
	Compressed   bool      `json:"compressed"`
	BlockRange   [2]int64  `json:"block_range"` // Start and end block indices
	CreatedAt    time.Time `json:"created_at"`
}

// NewIncrementalBackupManager creates a new incremental backup manager
func NewIncrementalBackupManager(config *IncrementalConfig) (*IncrementalBackupManager, error) {
	if config == nil {
		config = &IncrementalConfig{
			BasePath:          "/var/lib/novacron/backups",
			EnableCompression: true,
			EnableDedup:       true,
			CompressionLevel:  3,
			MaxIncrementals:   MaxIncrementals,
			CBTBlockSize:      CBTBlockSize,
			WorkerThreads:     4,
			BufferSize:        1024 * 1024, // 1MB
		}
	}
	
	// Ensure base path exists
	if err := os.MkdirAll(config.BasePath, 0755); err != nil {
		return nil, fmt.Errorf("failed to create backup base path: %w", err)
	}
	
	// Initialize zstd encoder
	encoder, err := zstd.NewWriter(nil, zstd.WithEncoderLevel(zstd.EncoderLevelFromZstd(config.CompressionLevel)))
	if err != nil {
		return nil, fmt.Errorf("failed to initialize compressor: %w", err)
	}
	
	// Initialize deduplication engine
	dedupEngine, err := NewDeduplicationEngine(filepath.Join(config.BasePath, "dedup"))
	if err != nil {
		return nil, fmt.Errorf("failed to initialize deduplication engine: %w", err)
	}
	
	manager := &IncrementalBackupManager{
		config:      config,
		trackers:    make(map[string]*CBTTracker),
		compressor:  encoder,
		dedupEngine: dedupEngine,
		chains:      make(map[string]*BackupChain),
	}
	
	// Load existing backup chains
	if err := manager.loadBackupChains(); err != nil {
		return nil, fmt.Errorf("failed to load backup chains: %w", err)
	}
	
	return manager, nil
}

// InitializeCBT initializes CBT tracking for a VM
func (m *IncrementalBackupManager) InitializeCBT(vmID string, vmSize int64) (*CBTTracker, error) {
	m.mutex.Lock()
	defer m.mutex.Unlock()
	
	// Check if tracker already exists
	if tracker, exists := m.trackers[vmID]; exists {
		return tracker, nil
	}
	
	totalBlocks := (vmSize + m.config.CBTBlockSize - 1) / m.config.CBTBlockSize
	
	tracker := &CBTTracker{
		vmID:        vmID,
		basePath:    filepath.Join(m.config.BasePath, "cbt", vmID),
		blocks:      make(map[int64]*BlockInfo),
		totalBlocks: totalBlocks,
		blockSize:   m.config.CBTBlockSize,
		version:     1,
		generation:  1,
		createdAt:   time.Now(),
		updatedAt:   time.Now(),
	}
	
	// Ensure CBT directory exists
	if err := os.MkdirAll(tracker.basePath, 0755); err != nil {
		return nil, fmt.Errorf("failed to create CBT directory: %w", err)
	}
	
	// Initialize all blocks
	for i := int64(0); i < totalBlocks; i++ {
		tracker.blocks[i] = &BlockInfo{
			Index:     i,
			Changed:   true, // Initially all blocks are considered changed
			LastWrite: tracker.createdAt,
			Size:      m.config.CBTBlockSize,
		}
	}
	
	// Save CBT metadata
	if err := tracker.saveCBTMetadata(); err != nil {
		return nil, fmt.Errorf("failed to save CBT metadata: %w", err)
	}
	
	m.trackers[vmID] = tracker
	return tracker, nil
}

// TrackBlockChange tracks changes to a specific block
func (m *IncrementalBackupManager) TrackBlockChange(vmID string, blockIndex int64, data []byte) error {
	m.mutex.RLock()
	tracker, exists := m.trackers[vmID]
	m.mutex.RUnlock()
	
	if !exists {
		return fmt.Errorf("CBT tracker not found for VM %s", vmID)
	}
	
	tracker.mutex.Lock()
	defer tracker.mutex.Unlock()
	
	// Calculate checksum
	checksum := calculateBlockChecksum(data)
	
	block, exists := tracker.blocks[blockIndex]
	if !exists {
		block = &BlockInfo{
			Index: blockIndex,
			Size:  int64(len(data)),
		}
		tracker.blocks[blockIndex] = block
	}
	
	// Check if block actually changed
	if block.Checksum != checksum {
		block.Checksum = checksum
		block.Changed = true
		block.LastWrite = time.Now()
		block.Size = int64(len(data))
		
		tracker.updatedAt = time.Now()
		tracker.generation++
	}
	
	return nil
}

// CreateIncrementalBackup creates an incremental backup
func (m *IncrementalBackupManager) CreateIncrementalBackup(ctx context.Context, vmID string, vmPath string, backupType BackupType) (*BackupManifest, error) {
	m.mutex.RLock()
	tracker, exists := m.trackers[vmID]
	m.mutex.RUnlock()
	
	if !exists {
		return nil, fmt.Errorf("CBT tracker not found for VM %s", vmID)
	}
	
	// Determine if we need a full backup
	chain, exists := m.chains[vmID]
	needFullBackup := !exists || len(chain.Incrementals) >= m.config.MaxIncrementals || backupType == FullBackup
	
	if needFullBackup {
		backupType = FullBackup
	} else {
		backupType = IncrementalBackup
	}
	
	backupID := fmt.Sprintf("%s-%s-%d", vmID, backupType, time.Now().Unix())
	backupPath := filepath.Join(m.config.BasePath, "data", backupID)
	
	if err := os.MkdirAll(backupPath, 0755); err != nil {
		return nil, fmt.Errorf("failed to create backup directory: %w", err)
	}
	
	manifest := &BackupManifest{
		BackupID:     backupID,
		VMID:         vmID,
		Type:         backupType,
		CreatedAt:    time.Now(),
		Files:        make([]BackupFile, 0),
		CBTVersion:   tracker.version,
		Metadata:     make(map[string]string),
	}
	
	if chain != nil && backupType == IncrementalBackup {
		if len(chain.Incrementals) > 0 {
			manifest.ParentID = chain.Incrementals[len(chain.Incrementals)-1]
		} else {
			manifest.ParentID = chain.FullBackupID
		}
	}
	
	// Perform the backup
	if err := m.performBackup(ctx, tracker, vmPath, backupPath, manifest); err != nil {
		os.RemoveAll(backupPath) // Cleanup on failure
		return nil, fmt.Errorf("backup failed: %w", err)
	}
	
	// Save manifest
	manifestPath := filepath.Join(backupPath, BackupManifestFile)
	if err := saveJSON(manifestPath, manifest); err != nil {
		return nil, fmt.Errorf("failed to save manifest: %w", err)
	}
	
	// Update backup chain
	if err := m.updateBackupChain(vmID, backupID, backupType); err != nil {
		return nil, fmt.Errorf("failed to update backup chain: %w", err)
	}
	
	// Reset CBT changed flags after successful backup
	if err := tracker.resetChangedBlocks(); err != nil {
		return nil, fmt.Errorf("failed to reset CBT: %w", err)
	}
	
	return manifest, nil
}

// performBackup performs the actual backup operation
func (m *IncrementalBackupManager) performBackup(ctx context.Context, tracker *CBTTracker, vmPath, backupPath string, manifest *BackupManifest) error {
	tracker.mutex.RLock()
	defer tracker.mutex.RUnlock()
	
	vmFile, err := os.Open(vmPath)
	if err != nil {
		return fmt.Errorf("failed to open VM file: %w", err)
	}
	defer vmFile.Close()
	
	backupFile := filepath.Join(backupPath, "disk.img")
	output, err := os.Create(backupFile)
	if err != nil {
		return fmt.Errorf("failed to create backup file: %w", err)
	}
	defer output.Close()
	
	var writer io.Writer = output
	var compressedWriter io.WriteCloser
	
	if m.config.EnableCompression {
		compressedWriter = gzip.NewWriter(output)
		writer = compressedWriter
		defer compressedWriter.Close()
	}
	
	buffer := make([]byte, m.config.CBTBlockSize)
	var totalSize int64
	var compressedSize int64
	var blockCount int64
	var changedBlocks int64
	
	// Process blocks
	for blockIndex := int64(0); blockIndex < tracker.totalBlocks; blockIndex++ {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
		
		block, exists := tracker.blocks[blockIndex]
		
		// For full backups, include all blocks
		// For incremental backups, only include changed blocks
		includeBlock := manifest.Type == FullBackup || (exists && block.Changed)
		
		if includeBlock {
			// Read block from VM file
			offset := blockIndex * tracker.blockSize
			n, err := vmFile.ReadAt(buffer[:tracker.blockSize], offset)
			if err != nil && err != io.EOF {
				return fmt.Errorf("failed to read block %d: %w", blockIndex, err)
			}
			
			blockData := buffer[:n]
			
			// Apply deduplication if enabled
			if m.config.EnableDedup {
				dedupData, err := m.dedupEngine.ProcessBlock(blockData)
				if err != nil {
					return fmt.Errorf("deduplication failed for block %d: %w", blockIndex, err)
				}
				blockData = dedupData
			}
			
			// Write block header (block index and size)
			if err := binary.Write(writer, binary.LittleEndian, blockIndex); err != nil {
				return fmt.Errorf("failed to write block header: %w", err)
			}
			if err := binary.Write(writer, binary.LittleEndian, int64(len(blockData))); err != nil {
				return fmt.Errorf("failed to write block size: %w", err)
			}
			
			// Write block data
			if _, err := writer.Write(blockData); err != nil {
				return fmt.Errorf("failed to write block data: %w", err)
			}
			
			totalSize += int64(n)
			changedBlocks++
		}
		
		blockCount++
	}
	
	// Get compressed size
	if compressedWriter != nil {
		compressedWriter.Close()
		if stat, err := output.Stat(); err == nil {
			compressedSize = stat.Size()
		}
	} else {
		compressedSize = totalSize
	}
	
	// Update manifest
	manifest.Size = totalSize
	manifest.CompressedSize = compressedSize
	manifest.BlockCount = blockCount
	manifest.ChangedBlocks = changedBlocks
	manifest.Checksum = calculateFileChecksum(backupFile)
	
	manifest.Files = append(manifest.Files, BackupFile{
		Path:       "disk.img",
		Size:       totalSize,
		Checksum:   manifest.Checksum,
		Compressed: m.config.EnableCompression,
		CreatedAt:  time.Now(),
	})
	
	return nil
}

// calculateBlockChecksum calculates SHA-256 checksum for block data
func calculateBlockChecksum(data []byte) string {
	hash := sha256.Sum256(data)
	return hex.EncodeToString(hash[:])
}

// calculateFileChecksum calculates SHA-256 checksum for a file
func calculateFileChecksum(filePath string) string {
	file, err := os.Open(filePath)
	if err != nil {
		return ""
	}
	defer file.Close()
	
	hasher := sha256.New()
	if _, err := io.Copy(hasher, file); err != nil {
		return ""
	}
	
	return hex.EncodeToString(hasher.Sum(nil))
}

// saveCBTMetadata saves CBT metadata to disk
func (t *CBTTracker) saveCBTMetadata() error {
	t.mutex.Lock()
	defer t.mutex.Unlock()
	
	metadata := &CBTMetadata{
		VMID:        t.vmID,
		Version:     t.version,
		Generation:  t.generation,
		TotalBlocks: t.totalBlocks,
		BlockSize:   t.blockSize,
		CreatedAt:   t.createdAt,
		UpdatedAt:   t.updatedAt,
		Blocks:      t.blocks,
	}
	
	metaPath := filepath.Join(t.basePath, "metadata"+CBTMetaExt)
	return saveJSON(metaPath, metadata)
}

// resetChangedBlocks resets all changed flags after successful backup
func (t *CBTTracker) resetChangedBlocks() error {
	t.mutex.Lock()
	defer t.mutex.Unlock()
	
	for _, block := range t.blocks {
		block.Changed = false
	}
	
	t.version++
	t.updatedAt = time.Now()
	
	return t.saveCBTMetadata()
}

// updateBackupChain updates the backup chain for a VM
func (m *IncrementalBackupManager) updateBackupChain(vmID, backupID string, backupType BackupType) error {
	m.mutex.Lock()
	defer m.mutex.Unlock()
	
	chain, exists := m.chains[vmID]
	if !exists || backupType == FullBackup {
		// Create new chain
		chain = &BackupChain{
			VMID:         vmID,
			FullBackupID: backupID,
			Incrementals: make([]string, 0),
			CreatedAt:    time.Now(),
			UpdatedAt:    time.Now(),
			Metadata:     make(map[string]string),
		}
		m.chains[vmID] = chain
	} else {
		// Add to existing chain
		chain.Incrementals = append(chain.Incrementals, backupID)
		chain.UpdatedAt = time.Now()
	}
	
	// Save chain to disk
	chainPath := filepath.Join(m.config.BasePath, "chains", vmID+".json")
	if err := os.MkdirAll(filepath.Dir(chainPath), 0755); err != nil {
		return fmt.Errorf("failed to create chains directory: %w", err)
	}
	
	return saveJSON(chainPath, chain)
}

// loadBackupChains loads existing backup chains from disk
func (m *IncrementalBackupManager) loadBackupChains() error {
	chainsDir := filepath.Join(m.config.BasePath, "chains")
	if _, err := os.Stat(chainsDir); os.IsNotExist(err) {
		return nil // No chains to load
	}
	
	files, err := filepath.Glob(filepath.Join(chainsDir, "*.json"))
	if err != nil {
		return fmt.Errorf("failed to glob chain files: %w", err)
	}
	
	for _, file := range files {
		var chain BackupChain
		if err := loadJSON(file, &chain); err != nil {
			continue // Skip invalid chain files
		}
		m.chains[chain.VMID] = &chain
	}
	
	return nil
}

// GetBackupChain returns the backup chain for a VM
func (m *IncrementalBackupManager) GetBackupChain(vmID string) (*BackupChain, error) {
	m.mutex.RLock()
	defer m.mutex.RUnlock()
	
	chain, exists := m.chains[vmID]
	if !exists {
		return nil, fmt.Errorf("backup chain not found for VM %s", vmID)
	}
	
	// Return a copy
	chainCopy := *chain
	return &chainCopy, nil
}

// ListBackups returns all backups for a VM
func (m *IncrementalBackupManager) ListBackups(vmID string) ([]string, error) {
	chain, err := m.GetBackupChain(vmID)
	if err != nil {
		return nil, err
	}
	
	backups := make([]string, 0, len(chain.Incrementals)+1)
	backups = append(backups, chain.FullBackupID)
	backups = append(backups, chain.Incrementals...)
	
	return backups, nil
}

// GetBackupManifest loads a backup manifest
func (m *IncrementalBackupManager) GetBackupManifest(backupID string) (*BackupManifest, error) {
	manifestPath := filepath.Join(m.config.BasePath, "data", backupID, BackupManifestFile)
	
	var manifest BackupManifest
	if err := loadJSON(manifestPath, &manifest); err != nil {
		return nil, fmt.Errorf("failed to load manifest for backup %s: %w", backupID, err)
	}
	
	return &manifest, nil
}

// GetCBTStats returns CBT statistics for a VM
func (m *IncrementalBackupManager) GetCBTStats(vmID string) (map[string]interface{}, error) {
	m.mutex.RLock()
	tracker, exists := m.trackers[vmID]
	m.mutex.RUnlock()
	
	if !exists {
		return nil, fmt.Errorf("CBT tracker not found for VM %s", vmID)
	}
	
	tracker.mutex.RLock()
	defer tracker.mutex.RUnlock()
	
	changedBlocks := int64(0)
	for _, block := range tracker.blocks {
		if block.Changed {
			changedBlocks++
		}
	}
	
	stats := map[string]interface{}{
		"vm_id":         tracker.vmID,
		"version":       tracker.version,
		"generation":    tracker.generation,
		"total_blocks":  tracker.totalBlocks,
		"changed_blocks": changedBlocks,
		"block_size":    tracker.blockSize,
		"created_at":    tracker.createdAt,
		"updated_at":    tracker.updatedAt,
	}
	
	return stats, nil
}

// Utility functions
func saveJSON(path string, data interface{}) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()
	
	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	return encoder.Encode(data)
}

func loadJSON(path string, data interface{}) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()
	
	return json.NewDecoder(file).Decode(data)
}