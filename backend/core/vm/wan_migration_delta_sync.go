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
}

// DeltaSyncManager manages delta synchronization for VM migration
type DeltaSyncManager struct {
	config DeltaSyncConfig
	stats  DeltaSyncStats
	mu     sync.RWMutex
	ctx    context.Context
	cancel context.CancelFunc
	logger *logrus.Entry
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

	return &DeltaSyncManager{
		config: config,
		stats: DeltaSyncStats{
			StartTime: time.Now(),
			BlockSize: config.BlockSizeKB * 1024,
		},
		ctx:    ctx,
		cancel: cancel,
		logger: logger.WithField("component", "DeltaSyncManager"),
	}
}

// Close releases resources used by the manager
func (m *DeltaSyncManager) Close() {
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

	// Step 2: Hash source file blocks
	m.logger.WithFields(logrus.Fields{
		"file": sourcePath,
		"size": sourceInfo.Size(),
	}).Info("Starting delta sync - hashing source file")

	startTime := time.Now()
	sourceHashes, err := m.hashFileBlocks(ctx, sourcePath)
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

	// Step 4: Check existing destination file if exists
	var neededBlocks []blockInfo
	destExists := false

	if _, err := os.Stat(destPath); err == nil {
		destExists = true
		// Destination file exists, calculate deltas
		m.logger.WithField("file", destPath).Info("Destination file exists, calculating needed blocks")

		neededBlocks, err = m.calculateNeededBlocks(ctx, sourceHashes, destPath)
		if err != nil {
			return fmt.Errorf("error calculating needed blocks: %w", err)
		}
	} else {
		// Destination file doesn't exist, need all blocks
		m.logger.Info("Destination file doesn't exist, all blocks needed")
		neededBlocks = make([]blockInfo, len(sourceHashes))
		copy(neededBlocks, sourceHashes)
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
	if err := m.createDeltaFile(ctx, sourcePath, deltaPath, neededBlocks); err != nil {
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
	m.mu.Unlock()

	m.logger.WithFields(logrus.Fields{
		"bytesSaved":        m.stats.BytesSaved,
		"bytesSavedPercent": m.stats.BytesSavedPercent,
		"transferredBytes":  m.stats.TransferredBytes,
	}).Info("Delta sync completed successfully")

	return nil
}

// hashFileBlocks hashes blocks of a file
func (m *DeltaSyncManager) hashFileBlocks(ctx context.Context, filePath string) ([]blockInfo, error) {
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
		"fileSize":   fileSize,
		"blockSize":  blockSize,
		"blockCount": blockCount,
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
				// Calculate hash based on configured algorithm
				var hash []byte
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

			// Read block
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
			data := make([]byte, n)
			copy(data, buffer[:n])

			// Submit job
			select {
			case <-ctx.Done():
				close(jobs)
				return nil, ctx.Err()
			case jobs <- hashJob{
				index: int(i),
				start: start,
				size:  n,
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
func (m *DeltaSyncManager) calculateNeededBlocks(ctx context.Context, sourceHashes []blockInfo, destPath string) ([]blockInfo, error) {
	// Hash destination file
	destHashes, err := m.hashFileBlocks(ctx, destPath)
	if err != nil {
		return nil, err
	}

	// Build map of destination hashes for quick lookup
	destHashMap := make(map[string]bool)
	for _, block := range destHashes {
		destHashMap[string(block.Hash)] = true
	}

	// Find blocks that aren't in destination
	neededBlocks := make([]blockInfo, 0)
	for _, block := range sourceHashes {
		if !destHashMap[string(block.Hash)] {
			neededBlocks = append(neededBlocks, block)
		}
	}

	return neededBlocks, nil
}

// createDeltaFile creates a delta file from needed blocks
func (m *DeltaSyncManager) createDeltaFile(ctx context.Context, sourcePath, deltaPath string, neededBlocks []blockInfo) error {
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
