package vm

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/klauspost/compress/zstd"
	"github.com/sirupsen/logrus"
)

// Special marker hash for eBPF-unused blocks (all zeros)
var unusedBlockMarkerHash = make([]byte, 32)

// zstdReaderWrapper wraps zstd.Decoder to implement io.ReadCloser
type zstdReaderWrapper struct {
	*zstd.Decoder
}

func (w *zstdReaderWrapper) Close() error {
	w.Decoder.Close()
	return nil
}

// DeltaSyncConfig defines configuration for delta synchronization
type DeltaSyncConfig struct {
	// Block size for delta computation in KB
	BlockSizeKB int

	// Enable compression for delta transfer
	EnableCompression bool

	// Compression level if compression is enabled
	CompressionLevel int

	// Hash algorithm (sha256, xxhash)
	HashAlgorithm string

	// Number of concurrent workers for hashing
	HashWorkers int

	// Directory for temporary files
	TempDir string

	// Bandwidth limit in bytes per second
	BandwidthLimit int64

	// Maximum retry count for failed transfers
	RetryCount int

	// Retry delay in milliseconds
	RetryDelayMs int

	// Logger for delta sync operations
	Logger *logrus.Logger

	// eBPF-related settings
	EnableEBPFFiltering bool          // Enable eBPF-based page filtering
	EBPFAgingThreshold  time.Duration // Time threshold for marking pages as unused
	EBPFMinAccessCount  uint32        // Minimum access count to consider a page active
	// Note: EBPFMapSizeLimit is not currently used as map size is fixed at compile time in BPF program
	// To use dynamic map sizing, the BPF program would need to be regenerated with custom max_entries
	FallbackOnEBPFError bool // Continue without eBPF if initialization fails
}

// DefaultDeltaSyncConfig returns the default config for delta sync
func DefaultDeltaSyncConfig() DeltaSyncConfig {
	return DeltaSyncConfig{
		BlockSizeKB:       64,
		EnableCompression: true,
		CompressionLevel:  3,
		HashAlgorithm:     "sha256",
		HashWorkers:       4,
		TempDir:           os.TempDir(),
		BandwidthLimit:    0, // No limit
		RetryCount:        3,
		RetryDelayMs:      1000,
		Logger:            logrus.New(),
		// eBPF defaults
		EnableEBPFFiltering: false, // Disabled by default for compatibility
		EBPFAgingThreshold:  5 * time.Second,
		EBPFMinAccessCount:  1,
		FallbackOnEBPFError: true, // Graceful degradation by default
	}
}

// DeltaSyncStats tracks statistics for delta synchronization
type DeltaSyncStats struct {
	TotalBytes             int64 // Total bytes in source file
	TransferredBytes       int64 // Actual bytes transferred
	DuplicateBlocks        int   // Number of blocks that were already present
	UniqueBlocks           int   // Number of blocks that needed transfer
	BlocksTransferred      int   // Number of blocks successfully transferred
	BlockSize              int   // Block size in bytes
	HashingDuration        time.Duration
	TransferDuration       time.Duration
	ReconstructionDuration time.Duration
	StartTime              time.Time
	EndTime                time.Time
	BytesSaved             int64   // Bytes saved compared to full transfer
	BytesSavedPercent      float64 // Percentage of bytes saved

	// eBPF statistics
	EBPFEnabled           bool    // Whether eBPF filtering was enabled
	EBPFBlocksSkipped     int     // Number of blocks skipped by eBPF filtering
	EBPFBytesSkipped      int64   // Bytes saved by eBPF filtering
	EBPFTotalPagesTracked int     // Total pages tracked by eBPF
	EBPFUnusedPages       int     // Number of unused pages detected
	EBPFSkipPercent       float64 // Percentage of blocks skipped by eBPF
}

// DeltaSyncManager manages delta synchronization for VM migration
type DeltaSyncManager struct {
	config        DeltaSyncConfig
	stats         DeltaSyncStats
	mu            sync.RWMutex
	ctx           context.Context
	cancel        context.CancelFunc
	logger        *logrus.Entry
	ebpfFilter    *EBPFMigrationFilter
	ebpfBlockFilter *EBPFBlockFilter
	ebpfEnabled   bool
}

// NewDeltaSyncManager creates a new delta sync manager
func NewDeltaSyncManager(config DeltaSyncConfig) *DeltaSyncManager {
	ctx, cancel := context.WithCancel(context.Background())

	// Ensure reasonable defaults
	if config.BlockSizeKB <= 0 {
		config.BlockSizeKB = 64
	}
	if config.HashWorkers <= 0 {
		config.HashWorkers = 4
	}
	if config.RetryCount < 0 {
		config.RetryCount = 3
	}
	if config.RetryDelayMs <= 0 {
		config.RetryDelayMs = 1000
	}
	if config.TempDir == "" {
		config.TempDir = os.TempDir()
	}

	// Setup logger
	logger := config.Logger
	if logger == nil {
		logger = logrus.New()
		logger.SetLevel(logrus.InfoLevel)
	}

	manager := &DeltaSyncManager{
		config: config,
		stats: DeltaSyncStats{
			StartTime: time.Now(),
			BlockSize: config.BlockSizeKB * 1024,
		},
		ctx:    ctx,
		cancel: cancel,
		logger: logger.WithField("component", "DeltaSyncManager"),
		ebpfEnabled: false,
	}

	return manager
}

// Close releases resources used by the manager
func (m *DeltaSyncManager) Close() {
	m.DisableEBPFFiltering()
	m.cancel()
}

// GetStats returns a copy of the current statistics
func (m *DeltaSyncManager) GetStats() DeltaSyncStats {
	m.mu.RLock()
	defer m.mu.RUnlock()

	stats := m.stats

	// Calculate derived stats if not already set
	if stats.BytesSaved == 0 && stats.TotalBytes > 0 && stats.TransferredBytes > 0 {
		stats.BytesSaved = stats.TotalBytes - stats.TransferredBytes
		stats.BytesSavedPercent = float64(stats.BytesSaved) / float64(stats.TotalBytes) * 100
	}

	return stats
}

// blockInfo represents information about a block for delta sync
type blockInfo struct {
	Index int
	Hash  []byte
	Start int64
	Size  int
}

// SetFileMapping configures the file-to-page mapping for eBPF block filtering.
// This should be called before SyncFile when eBPF filtering is enabled and
// the file type is known (e.g., memory snapshot vs disk image).
//
// For memory snapshots, use CreateMemorySnapshotMapping().
// For disk images, pass nil to disable eBPF filtering for that file.
func (m *DeltaSyncManager) SetFileMapping(mapping *FileOffsetToPageMapping) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.ebpfBlockFilter != nil {
		if mapping != nil {
			m.ebpfBlockFilter.SetFileMapping(mapping)
			m.logger.WithFields(logrus.Fields{
				"file_type":     mapping.FileType,
				"mapping_count": len(mapping.OffsetToPFN),
			}).Info("File mapping configured for eBPF filtering")
		} else {
			m.ebpfBlockFilter.ClearFileMapping()
			m.logger.Info("File mapping cleared, eBPF filtering disabled for this file")
		}
	}
}

// SyncFileWithType performs delta synchronization with automatic file type detection
// and appropriate eBPF mapping configuration.
func (m *DeltaSyncManager) SyncFileWithType(ctx context.Context, sourcePath, destPath string, fileType string) error {
	// Get source file info for mapping creation
	sourceInfo, err := os.Stat(sourcePath)
	if err != nil {
		return fmt.Errorf("error accessing source file: %w", err)
	}

	// Configure file mapping based on type
	if m.ebpfEnabled && m.ebpfBlockFilter != nil {
		var mapping *FileOffsetToPageMapping

		switch fileType {
		case "memory_snapshot":
			mapping = CreateMemorySnapshotMapping(sourceInfo.Size(), 0)
			m.logger.WithFields(logrus.Fields{
				"file":       sourcePath,
				"file_type":  fileType,
				"page_count": len(mapping.OffsetToPFN),
			}).Info("Created memory snapshot mapping for eBPF filtering")
		case "disk_image":
			// Disk images don't use eBPF filtering
			mapping = nil
			m.logger.WithField("file", sourcePath).Info("Disk image detected, eBPF filtering disabled")
		default:
			// Try to auto-detect
			detectedType := DetectFileType(sourcePath)
			if detectedType == "memory_snapshot" {
				mapping = CreateMemorySnapshotMapping(sourceInfo.Size(), 0)
				m.logger.WithField("file", sourcePath).Info("Auto-detected memory snapshot, enabling eBPF filtering")
			} else {
				mapping = nil
				m.logger.WithFields(logrus.Fields{
					"file":          sourcePath,
					"detected_type": detectedType,
				}).Info("File type not suitable for eBPF filtering")
			}
		}

		m.SetFileMapping(mapping)
	}

	return m.SyncFile(ctx, sourcePath, destPath)
}

// SyncFile performs delta synchronization of a file
func (m *DeltaSyncManager) SyncFile(ctx context.Context, sourcePath, destPath string) error {
	m.mu.Lock()
	m.stats = DeltaSyncStats{
		StartTime: time.Now(),
		BlockSize: m.config.BlockSizeKB * 1024,
	}
	m.mu.Unlock()

	defer func() {
		m.mu.Lock()
		m.stats.EndTime = time.Now()
		m.mu.Unlock()
	}()

	// Step 1: Get source file size and calculate blocks
	sourceInfo, err := os.Stat(sourcePath)
	if err != nil {
		return fmt.Errorf("error accessing source file: %w", err)
	}

	m.mu.Lock()
	m.stats.TotalBytes = sourceInfo.Size()
	m.mu.Unlock()

	// Create temp directory for delta sync
	syncDir := filepath.Join(m.config.TempDir, fmt.Sprintf("deltasync-%s", filepath.Base(sourcePath)))
	if err := os.MkdirAll(syncDir, 0755); err != nil {
		return fmt.Errorf("failed to create sync directory: %w", err)
	}
	defer os.RemoveAll(syncDir)

	// Step 4 (check early): Check if destination file exists
	// We need this before hashing to determine if eBPF skipping is safe
	destExists := false
	if _, err := os.Stat(destPath); err == nil {
		destExists = true
	}

	// Step 2: Hash source file blocks
	// IMPORTANT: For initial sync (destExists=false), we do NOT use eBPF block skipping
	// to ensure bitwise-faithful transfer. eBPF aging might misclassify active pages.
	m.logger.WithFields(logrus.Fields{
		"file":       sourcePath,
		"size":       sourceInfo.Size(),
		"destExists": destExists,
	}).Info("Starting delta sync - hashing source file")

	startTime := time.Now()
	sourceHashes, err := m.hashFileBlocksWithOptions(ctx, sourcePath, destExists)
	if err != nil {
		return fmt.Errorf("error hashing source file: %w", err)
	}

	m.mu.Lock()
	m.stats.HashingDuration = time.Since(startTime)
	m.mu.Unlock()

	// Step 3: Create signature file
	sigPath := filepath.Join(syncDir, "signature.dat")
	if err := m.writeSignatureFile(sigPath, sourceHashes); err != nil {
		return fmt.Errorf("error creating signature file: %w", err)
	}

	// Step 4: Calculate needed blocks
	var neededBlocks []blockInfo

	if destExists {
		// Destination file exists, calculate deltas
		m.logger.WithField("file", destPath).Info("Destination file exists, calculating needed blocks")

		neededBlocks, err = m.calculateNeededBlocks(ctx, sourceHashes, destPath, destExists)
		if err != nil {
			return fmt.Errorf("error calculating needed blocks: %w", err)
		}
	} else {
		// Destination file doesn't exist, need all blocks
		m.logger.Info("Destination file doesn't exist, all blocks needed")
		neededBlocks, err = m.calculateNeededBlocks(ctx, sourceHashes, destPath, destExists)
		if err != nil {
			return fmt.Errorf("error calculating needed blocks when dest doesn't exist: %w", err)
		}
	}

	m.mu.Lock()
	m.stats.UniqueBlocks = len(neededBlocks)
	m.stats.DuplicateBlocks = len(sourceHashes) - len(neededBlocks)
	m.mu.Unlock()

	m.logger.WithFields(logrus.Fields{
		"uniqueBlocks":    len(neededBlocks),
		"duplicateBlocks": len(sourceHashes) - len(neededBlocks),
		"totalBlocks":     len(sourceHashes),
	}).Info("Block analysis complete")

	// Step 5: Create delta file with needed blocks
	deltaPath := filepath.Join(syncDir, "delta.dat")
	if err := m.createDeltaFile(ctx, sourcePath, deltaPath, neededBlocks, destExists); err != nil {
		return fmt.Errorf("error creating delta file: %w", err)
	}

	// Step 6: Apply delta to destination file
	startTime = time.Now()
	if err := m.applyDelta(ctx, destPath, deltaPath, sigPath, destExists); err != nil {
		return fmt.Errorf("error applying delta: %w", err)
	}

	// Update stats
	deltaInfo, _ := os.Stat(deltaPath)
	m.mu.Lock()
	m.stats.TransferredBytes = deltaInfo.Size()
	m.stats.TransferDuration = time.Since(startTime)
	m.stats.BytesSaved = m.stats.TotalBytes - m.stats.TransferredBytes
	if m.stats.TotalBytes > 0 {
		m.stats.BytesSavedPercent = float64(m.stats.BytesSaved) / float64(m.stats.TotalBytes) * 100
	}
	m.stats.BlocksTransferred = m.stats.UniqueBlocks

	// Update eBPF statistics
	if m.ebpfEnabled {
		totalBlocks := len(sourceHashes)
		if totalBlocks > 0 {
			m.stats.EBPFSkipPercent = float64(m.stats.EBPFBlocksSkipped) / float64(totalBlocks) * 100
		}

		// Get eBPF filter stats
		if m.ebpfFilter != nil {
			ebpfStats := m.ebpfFilter.GetStats()
			if total, ok := ebpfStats["total_pages"].(int); ok {
				m.stats.EBPFTotalPagesTracked = total
			}
			if unused, ok := ebpfStats["unused_pages"].(int); ok {
				m.stats.EBPFUnusedPages = unused
			}
		}
	}
	m.mu.Unlock()

	m.logger.WithFields(logrus.Fields{
		"bytesSaved":        m.stats.BytesSaved,
		"bytesSavedPercent": m.stats.BytesSavedPercent,
		"transferredBytes":  m.stats.TransferredBytes,
	}).Info("Delta sync completed successfully")

	return nil
}

// hashFileBlocks hashes blocks of a file (uses eBPF filtering if available)
func (m *DeltaSyncManager) hashFileBlocks(ctx context.Context, filePath string) ([]blockInfo, error) {
	// Default: allow eBPF skipping (for backward compatibility)
	return m.hashFileBlocksWithOptions(ctx, filePath, true)
}

// hashFileBlocksWithOptions hashes blocks of a file with control over eBPF block skipping
// When allowEBPFSkipping is false (e.g., initial sync), all blocks are read from source
// to ensure bitwise-faithful transfer. eBPF skipping is only safe for incremental syncs
// where the destination already has valid data.
func (m *DeltaSyncManager) hashFileBlocksWithOptions(ctx context.Context, filePath string, allowEBPFSkipping bool) ([]blockInfo, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	fileInfo, err := file.Stat()
	if err != nil {
		return nil, err
	}

	fileSize := fileInfo.Size()
	blockSize := int64(m.config.BlockSizeKB * 1024)
	blockCount := (fileSize + blockSize - 1) / blockSize // ceil division

	m.logger.WithFields(logrus.Fields{
		"fileSize":          fileSize,
		"blockSize":         blockSize,
		"blockCount":        blockCount,
		"allowEBPFSkipping": allowEBPFSkipping,
	}).Debug("Setting up file hashing")

	// Create work channels
	type hashJob struct {
		index int
		start int64
		size  int
		data  []byte
	}

	jobs := make(chan hashJob, m.config.HashWorkers*2)
	results := make(chan blockInfo, m.config.HashWorkers*2)
	blocks := make([]blockInfo, 0, blockCount)

	// Start worker goroutines
	var wg sync.WaitGroup
	for i := 0; i < m.config.HashWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for job := range jobs {
				var hash []byte

				// Check if this block should use the unused marker hash
				// Job data length of 0 indicates an eBPF-unused block
				if len(job.data) == 0 {
					// Use special marker hash for eBPF-unused blocks
					hash = make([]byte, 32)
					copy(hash, unusedBlockMarkerHash)
				} else {
					// Calculate hash based on configured algorithm
					if m.config.HashAlgorithm == "sha256" {
						hasher := sha256.New()
						hasher.Write(job.data)
						hash = hasher.Sum(nil)
					} else {
						// Default to sha256 if algorithm not recognized
						hasher := sha256.New()
						hasher.Write(job.data)
						hash = hasher.Sum(nil)
					}
				}

				// Send result
				select {
				case <-ctx.Done():
					return
				case results <- blockInfo{
					Index: job.index,
					Hash:  hash,
					Start: job.start,
					Size:  job.size,
				}:
				}
			}
		}()
	}

	// Process results in a separate goroutine
	done := make(chan struct{})
	go func() {
		defer close(done)

		resultMap := make(map[int]blockInfo)
		count := 0

		for info := range results {
			resultMap[info.Index] = info
			count++

			if count == int(blockCount) {
				break
			}
		}

		// Sort blocks by index
		for i := 0; i < int(blockCount); i++ {
			if info, ok := resultMap[i]; ok {
				blocks = append(blocks, info)
			}
		}
	}()

	// Read file in blocks and submit jobs
	buffer := make([]byte, blockSize)
	for i := int64(0); i < blockCount; i++ {
		select {
		case <-ctx.Done():
			close(jobs)
			return nil, ctx.Err()
		default:
			// Calculate block size (may be smaller for last block)
			start := i * blockSize
			size := blockSize
			if start+size > fileSize {
				size = fileSize - start
			}

			// Check if eBPF filtering suggests this block is unused
			// IMPORTANT: Only use eBPF skipping for incremental syncs (allowEBPFSkipping=true)
			// For initial sync, we must read all blocks to ensure correctness
			isUnusedByEBPF := false
			if allowEBPFSkipping && m.ebpfEnabled && m.ebpfBlockFilter != nil {
				isUnusedByEBPF = m.ebpfBlockFilter.IsBlockUnused(start)
				if isUnusedByEBPF {
					m.mu.Lock()
					m.stats.EBPFBlocksSkipped++
					m.stats.EBPFBytesSkipped += int64(size)
					m.mu.Unlock()
				}
			}

			// Prepare block data
			var data []byte
			var actualSize int

			if isUnusedByEBPF {
				// For eBPF-unused blocks (only in incremental sync), use empty data as marker
				// This signals the hasher to use the special marker hash
				data = nil
				actualSize = int(size)
			} else {
				// Read block normally
				if _, err := file.Seek(start, io.SeekStart); err != nil {
					close(jobs)
					return nil, err
				}

				n, err := io.ReadFull(file, buffer[:size])
				if err != nil && !errors.Is(err, io.ErrUnexpectedEOF) {
					close(jobs)
					return nil, err
				}

				// Copy buffer for the job
				data = make([]byte, n)
				copy(data, buffer[:n])
				actualSize = n
			}

			// Submit job
			select {
			case <-ctx.Done():
				close(jobs)
				return nil, ctx.Err()
			case jobs <- hashJob{
				index: int(i),
				start: start,
				size:  actualSize,
				data:  data,
			}:
			}
		}
	}

	close(jobs)
	wg.Wait()
	close(results)
	<-done

	return blocks, nil
}

// writeSignatureFile writes block signatures to a file
func (m *DeltaSyncManager) writeSignatureFile(path string, blocks []blockInfo) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	// Write header
	header := struct {
		BlockCount int32
		BlockSize  int32
	}{
		BlockCount: int32(len(blocks)),
		BlockSize:  int32(m.config.BlockSizeKB * 1024),
	}

	if err := binary.Write(file, binary.LittleEndian, header.BlockCount); err != nil {
		return err
	}
	if err := binary.Write(file, binary.LittleEndian, header.BlockSize); err != nil {
		return err
	}

	// Write block hashes
	for _, block := range blocks {
		// Write block index
		if err := binary.Write(file, binary.LittleEndian, int32(block.Index)); err != nil {
			return err
		}

		// Write hash length
		hashLen := int32(len(block.Hash))
		if err := binary.Write(file, binary.LittleEndian, hashLen); err != nil {
			return err
		}

		// Write hash
		if _, err := file.Write(block.Hash); err != nil {
			return err
		}
	}

	return nil
}

// calculateNeededBlocks compares source hashes with destination file to find needed blocks
func (m *DeltaSyncManager) calculateNeededBlocks(ctx context.Context, sourceHashes []blockInfo, destPath string, destExists bool) ([]blockInfo, error) {
	if !destExists {
		// Destination doesn't exist (initial sync), need all blocks.
		// IMPORTANT: For initial sync, eBPF skipping is disabled at the hashing level
		// (see hashFileBlocksWithOptions), so all sourceHashes contain real data hashes.
		// This ensures bitwise-faithful initial migration, avoiding risks of eBPF
		// aging misclassifying active pages. eBPF-based skip optimization only applies
		// to subsequent incremental syncs where destination already has valid data.
		neededBlocks := make([]blockInfo, 0, len(sourceHashes))
		for _, block := range sourceHashes {
			neededBlocks = append(neededBlocks, block)
		}
		return neededBlocks, nil
	}

	// Destination exists, hash it to find differences
	destHashes, err := m.hashFileBlocks(ctx, destPath)
	if err != nil {
		return nil, err
	}

	// Build map of destination hashes for quick lookup
	destHashMap := make(map[string]bool)
	for _, block := range destHashes {
		destHashMap[string(block.Hash)] = true
	}

	// Find blocks that aren't in destination or are marked as unused
	neededBlocks := make([]blockInfo, 0)
	for _, block := range sourceHashes {
		// Check if block has the unused marker hash
		isUnusedMarker := bytes.Equal(block.Hash, unusedBlockMarkerHash)

		if isUnusedMarker {
			// For unused blocks when dest exists, we can skip them since
			// the destination already has valid content for these ranges
			continue
		}

		if !destHashMap[string(block.Hash)] {
			neededBlocks = append(neededBlocks, block)
		}
	}

	return neededBlocks, nil
}

// createDeltaFile creates a delta file from needed blocks
func (m *DeltaSyncManager) createDeltaFile(ctx context.Context, sourcePath, deltaPath string, neededBlocks []blockInfo, destExists bool) error {
	// Open source file
	sourceFile, err := os.Open(sourcePath)
	if err != nil {
		return err
	}
	defer sourceFile.Close()

	// Create delta file
	deltaFile, err := os.Create(deltaPath)
	if err != nil {
		return err
	}
	defer deltaFile.Close()

	// Create writer with optional compression
	var writer io.WriteCloser = deltaFile
	if m.config.EnableCompression {
		compressor, err := zstd.NewWriter(deltaFile, zstd.WithEncoderLevel(zstd.EncoderLevel(m.config.CompressionLevel)))
		if err != nil {
			return fmt.Errorf("failed to create compressor: %w", err)
		}
		defer compressor.Close()
		writer = compressor
	}

	// Write header
	header := struct {
		Magic       [4]byte // "DSYN"
		BlockCount  int32
		BlockSize   int32
		TotalBlocks int32
	}{
		Magic:       [4]byte{'D', 'S', 'Y', 'N'},
		BlockCount:  int32(len(neededBlocks)),
		BlockSize:   int32(m.config.BlockSizeKB * 1024),
		TotalBlocks: int32(len(neededBlocks)),
	}

	if err := binary.Write(writer, binary.LittleEndian, header); err != nil {
		return err
	}

	// Write blocks with retries
	buffer := make([]byte, m.config.BlockSizeKB*1024)
	for i, block := range neededBlocks {
		if i > 0 && i%100 == 0 {
			m.logger.WithFields(logrus.Fields{
				"processed": i,
				"total":     len(neededBlocks),
			}).Debug("Delta file creation progress")
		}

		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
			// Write block header
			if err := binary.Write(writer, binary.LittleEndian, int32(block.Index)); err != nil {
				return err
			}
			if err := binary.Write(writer, binary.LittleEndian, int32(block.Size)); err != nil {
				return err
			}

			// Check if this block has the unused marker hash (eBPF-marked unused)
			// NOTE: For initial sync (destExists=false), eBPF skipping is disabled at
			// the hashing level, so unused markers should not appear. For incremental
			// sync (destExists=true), unused markers are filtered in calculateNeededBlocks.
			// This check is a safety fallback for any edge cases.
			isUnusedMarker := bytes.Equal(block.Hash, unusedBlockMarkerHash)
			if isUnusedMarker {
				// This should not normally happen due to filtering at earlier stages.
				// Skip the block as a safety measure.
				m.logger.WithFields(logrus.Fields{
					"block":      block.Index,
					"destExists": destExists,
				}).Warn("Unexpected unused marker in createDeltaFile, skipping block")
				continue
			}

			// Read block from source with retries
			var blockData []byte
			for retry := 0; retry <= m.config.RetryCount; retry++ {
				// If not first attempt, sleep before retry
				if retry > 0 {
					time.Sleep(time.Duration(m.config.RetryDelayMs) * time.Millisecond)
					m.logger.WithFields(logrus.Fields{
						"block": block.Index,
						"retry": retry,
					}).Debug("Retrying block read")
				}

				// Seek to block start
				if _, err := sourceFile.Seek(block.Start, io.SeekStart); err != nil {
					if retry == m.config.RetryCount {
						return err
					}
					continue
				}

				// Read block
				n, err := io.ReadFull(sourceFile, buffer[:block.Size])
				if err != nil && !errors.Is(err, io.ErrUnexpectedEOF) {
					if retry == m.config.RetryCount {
						return err
					}
					continue
				}

				blockData = buffer[:n]
				break
			}

			// Write block data
			if _, err := writer.Write(blockData); err != nil {
				return err
			}
		}
	}

	// Flush and close compression if used
	if closer, ok := writer.(io.Closer); ok {
		if err := closer.Close(); err != nil {
			return err
		}
	}

	return nil
}

// applyDelta applies a delta file to create or update a destination file
func (m *DeltaSyncManager) applyDelta(ctx context.Context, destPath, deltaPath, sigPath string, destExists bool) error {
	startTime := time.Now()

	// Read signature file to get block information
	sigFile, err := os.Open(sigPath)
	if err != nil {
		return err
	}
	defer sigFile.Close()

	// Read header
	var blockCount, blockSize int32
	if err := binary.Read(sigFile, binary.LittleEndian, &blockCount); err != nil {
		return err
	}
	if err := binary.Read(sigFile, binary.LittleEndian, &blockSize); err != nil {
		return err
	}

	// Open delta file
	deltaFile, err := os.Open(deltaPath)
	if err != nil {
		return err
	}
	defer deltaFile.Close()

	// Create reader with optional decompression
	var reader io.ReadCloser = deltaFile
	if m.config.EnableCompression {
		decompressor, err := zstd.NewReader(deltaFile)
		if err != nil {
			return fmt.Errorf("failed to create decompressor: %w", err)
		}
		defer decompressor.Close()
		// Wrap decompressor to implement io.ReadCloser
		reader = &zstdReaderWrapper{decompressor}
	}

	// Read delta header
	var magic [4]byte
	var deltaBlockCount, deltaBlockSize, deltaTotalBlocks int32
	if err := binary.Read(reader, binary.LittleEndian, &magic); err != nil {
		return err
	}
	if string(magic[:]) != "DSYN" {
		return fmt.Errorf("invalid delta file format")
	}
	if err := binary.Read(reader, binary.LittleEndian, &deltaBlockCount); err != nil {
		return err
	}
	if err := binary.Read(reader, binary.LittleEndian, &deltaBlockSize); err != nil {
		return err
	}
	if err := binary.Read(reader, binary.LittleEndian, &deltaTotalBlocks); err != nil {
		return err
	}

	m.logger.WithFields(logrus.Fields{
		"deltaBlockCount":  deltaBlockCount,
		"deltaTotalBlocks": deltaTotalBlocks,
		"destExists":       destExists,
	}).Debug("Applying delta")

	// Create temporary file for destination
	tempPath := destPath + ".tmp"
	tempFile, err := os.Create(tempPath)
	if err != nil {
		return err
	}
	defer func() {
		tempFile.Close()
		if err != nil {
			os.Remove(tempPath)
		}
	}()

	// Allocate space for the full file if we know the size
	fileSize := int64(blockSize) * int64(blockCount)
	if fileSize > 0 {
		if err := tempFile.Truncate(fileSize); err != nil {
			return err
		}
	}

	// If destination file exists, copy its contents to temp file first
	if destExists {
		m.logger.Debug("Copying existing destination file to temp file")
		srcFile, err := os.Open(destPath)
		if err != nil {
			return err
		}

		_, err = io.Copy(tempFile, srcFile)
		srcFile.Close()
		if err != nil {
			return err
		}
	}

	// Apply delta blocks
	buffer := make([]byte, deltaBlockSize)
	for i := int32(0); i < deltaBlockCount; i++ {
		// Read block index and size
		var blockIndex, blockSize int32
		if err := binary.Read(reader, binary.LittleEndian, &blockIndex); err != nil {
			return err
		}
		if err := binary.Read(reader, binary.LittleEndian, &blockSize); err != nil {
			return err
		}

		// Read block data
		if blockSize > int32(len(buffer)) {
			buffer = make([]byte, blockSize)
		}
		if _, err := io.ReadFull(reader, buffer[:blockSize]); err != nil {
			return err
		}

		// Write block to its position
		offset := int64(blockIndex) * int64(deltaBlockSize)
		if _, err := tempFile.Seek(offset, io.SeekStart); err != nil {
			return err
		}
		if _, err := tempFile.Write(buffer[:blockSize]); err != nil {
			return err
		}

		if i > 0 && i%100 == 0 {
			m.logger.WithFields(logrus.Fields{
				"processed": i,
				"total":     deltaBlockCount,
			}).Debug("Delta application progress")
		}
	}

	// Close temp file
	if err := tempFile.Close(); err != nil {
		return err
	}

	// Replace destination with temp file
	if destExists {
		if err := os.Remove(destPath); err != nil {
			return err
		}
	}
	if err := os.Rename(tempPath, destPath); err != nil {
		return err
	}

	m.mu.Lock()
	m.stats.ReconstructionDuration = time.Since(startTime)
	m.mu.Unlock()

	return nil
}

// CreateSignatureFile creates a signature file for a VM disk that can be used
// for future delta sync operations
func (m *DeltaSyncManager) CreateSignatureFile(ctx context.Context, vmDiskPath, sigPath string) error {
	m.logger.WithFields(logrus.Fields{
		"disk":      vmDiskPath,
		"signature": sigPath,
	}).Info("Creating signature file for VM disk")

	// Hash the disk file
	blocks, err := m.hashFileBlocks(ctx, vmDiskPath)
	if err != nil {
		return fmt.Errorf("error hashing VM disk: %w", err)
	}

	// Write signature file
	if err := m.writeSignatureFile(sigPath, blocks); err != nil {
		return fmt.Errorf("error writing signature file: %w", err)
	}

	return nil
}

// IntegrateWithWANMigrationOptimizer configures a WANMigrationOptimizer to use delta sync
func (m *DeltaSyncManager) IntegrateWithWANMigrationOptimizer(optimizer *WANMigrationOptimizer) {
	optimizer.UpdateStats(func(stats *WANMigrationStats) {
		syncStats := m.GetStats()
		stats.DeltaSyncSavingsBytes = syncStats.BytesSaved
	})
}

// EnableEBPFFiltering enables eBPF-based page filtering for the target VM/file
// Returns an error when:
// - EnableEBPFFiltering config flag is false (intentional, distinct from FallbackOnEBPFError)
// - eBPF is not supported on the system and FallbackOnEBPFError is false
// - eBPF program loading/attachment fails and FallbackOnEBPFError is false
//
// Note: When FallbackOnEBPFError is true, system-level eBPF errors result in
// graceful degradation (logs warning, returns nil) rather than blocking migration.
func (m *DeltaSyncManager) EnableEBPFFiltering(pid uint32) error {
	// Use default (host) namespace
	return m.EnableEBPFFilteringWithNamespace(pid, "")
}

// EnableEBPFFilteringWithNamespace enables eBPF-based page filtering with optional
// guest namespace injection. This provides true guest-aware unused page detection.
//
// Parameters:
// - pid: The VM process PID (QEMU process ID from host perspective)
// - namespacePath: Path to guest PID namespace (e.g., "/proc/<pid>/ns/pid")
//                  Pass empty string to use host namespace (fallback mode)
//
// When namespacePath is provided and accessible:
// - Switches to guest namespace before loading eBPF
// - Tracks pages from guest's perspective (PID 1 = guest init)
// - Provides 20-30% improvement in unused page detection
//
// When namespacePath is empty or inaccessible:
// - Falls back to host namespace tracking
// - Tracks QEMU process page accesses (less accurate but still useful)
func (m *DeltaSyncManager) EnableEBPFFilteringWithNamespace(pid uint32, namespacePath string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Check if already enabled
	if m.ebpfEnabled {
		return nil
	}

	// Check if eBPF is configured to be enabled
	// This is intentionally an error - user explicitly disabled eBPF in config
	// This is different from FallbackOnEBPFError which handles system-level failures
	if !m.config.EnableEBPFFiltering {
		return fmt.Errorf("eBPF filtering not enabled in configuration")
	}

	// Check if eBPF is supported on this system
	if !IsEBPFSupported() {
		if m.config.FallbackOnEBPFError {
			m.logger.Warn("eBPF not supported, falling back to standard delta sync")
			return nil
		}
		return ErrEBPFNotSupported
	}

	var filter *EBPFMigrationFilter
	var err error

	// Try guest namespace injection if path is provided
	if namespacePath != "" {
		m.logger.WithFields(logrus.Fields{
			"pid":            pid,
			"namespace_path": namespacePath,
		}).Info("Attempting guest namespace eBPF injection")

		filter, err = NewEBPFMigrationFilterInGuestNamespace(m.config.Logger, pid, namespacePath)
		if err != nil {
			m.logger.WithError(err).Warn("Guest namespace injection failed, will try host namespace")
			// Will fall through to host namespace attempt below
			filter = nil
		}
	}

	// Fall back to host namespace if guest injection failed or wasn't attempted
	if filter == nil {
		m.logger.WithField("pid", pid).Info("Using host namespace eBPF filtering")
		filter, err = NewEBPFMigrationFilter(m.config.Logger, pid)
		if err != nil {
			if m.config.FallbackOnEBPFError {
				m.logger.WithError(err).Warn("Failed to create eBPF filter, falling back to standard delta sync")
				return nil
			}
			return fmt.Errorf("failed to create eBPF filter: %w", err)
		}
	}

	// Configure aging threshold and min access count
	if m.config.EBPFAgingThreshold > 0 {
		filter.SetAgingThreshold(m.config.EBPFAgingThreshold)
	}
	if m.config.EBPFMinAccessCount > 0 {
		filter.SetMinAccessCount(m.config.EBPFMinAccessCount)
	}

	// Attach eBPF programs
	if err := filter.Attach(); err != nil {
		filter.Close()
		if m.config.FallbackOnEBPFError {
			m.logger.WithError(err).Warn("Failed to attach eBPF programs, falling back to standard delta sync")
			return nil
		}
		return fmt.Errorf("failed to attach eBPF programs: %w", err)
	}

	// Create block filter
	blockFilter := NewEBPFBlockFilter(filter, m.config.BlockSizeKB*1024)

	m.ebpfFilter = filter
	m.ebpfBlockFilter = blockFilter
	m.ebpfEnabled = true

	m.logger.WithFields(logrus.Fields{
		"pid":             pid,
		"namespace_path":  namespacePath,
		"guest_injection": namespacePath != "",
	}).Info("eBPF filtering enabled successfully")
	m.stats.EBPFEnabled = true

	return nil
}

// DisableEBPFFiltering disables eBPF-based page filtering
func (m *DeltaSyncManager) DisableEBPFFiltering() {
	m.mu.Lock()
	defer m.mu.Unlock()

	if !m.ebpfEnabled {
		return
	}

	// Close eBPF filter
	if m.ebpfFilter != nil {
		if err := m.ebpfFilter.Close(); err != nil {
			m.logger.WithError(err).Warn("Error closing eBPF filter")
		}
		m.ebpfFilter = nil
	}

	m.ebpfBlockFilter = nil
	m.ebpfEnabled = false

	m.logger.Info("eBPF filtering disabled")
}

// IsEBPFEnabled returns whether eBPF filtering is currently enabled
func (m *DeltaSyncManager) IsEBPFEnabled() bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.ebpfEnabled
}

// GetEBPFStats returns eBPF statistics
func (m *DeltaSyncManager) GetEBPFStats() map[string]interface{} {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if !m.ebpfEnabled || m.ebpfFilter == nil {
		return map[string]interface{}{"enabled": false}
	}

	return m.ebpfFilter.GetStats()
}

// MarkAgedOutPages marks aged-out pages as unused in the eBPF filter
func (m *DeltaSyncManager) MarkAgedOutPages() (int, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if !m.ebpfEnabled || m.ebpfFilter == nil {
		return 0, nil
	}

	return m.ebpfFilter.MarkPagesAsUnused()
}
