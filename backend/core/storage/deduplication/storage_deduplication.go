package deduplication

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
)

// BlockSize represents the default size of data blocks for deduplication
const DefaultBlockSize = 64 * 1024 // 64KB

// DedupAlgorithm represents the algorithm used for deduplication
type DedupAlgorithm string

const (
	// DedupNone indicates no deduplication should be used
	DedupNone DedupAlgorithm = "none"

	// DedupFixed uses fixed-size blocks for deduplication
	DedupFixed DedupAlgorithm = "fixed"

	// DedupVariable uses variable-size blocks based on content boundaries
	DedupVariable DedupAlgorithm = "variable"

	// DedupContent uses content-defined chunking for deduplication
	DedupContent DedupAlgorithm = "content"
)

// DedupConfig contains configuration for data deduplication
type DedupConfig struct {
	// Algorithm to use for deduplication
	Algorithm DedupAlgorithm `json:"algorithm"`

	// Block size for fixed-size deduplication
	BlockSize int `json:"block_size"`

	// Minimum size in bytes before deduplication is applied
	MinSizeBytes int `json:"min_size_bytes"`

	// Whether to inline small blocks rather than deduplicate them
	InlineSmallBlocks bool `json:"inline_small_blocks"`

	// Path to the deduplication store
	StorePath string `json:"store_path"`

	// Maximum size of the deduplication store in bytes (0 = unlimited)
	MaxStoreSize int64 `json:"max_store_size"`

	// Whether to compress blocks in the deduplication store
	CompressBlocks bool `json:"compress_blocks"`

	// Whether to use a bloom filter to speed up lookups
	UseBloomFilter bool `json:"use_bloom_filter"`

	// Whether to verify blocks on read
	VerifyOnRead bool `json:"verify_on_read"`
}

// DefaultDedupConfig returns a default deduplication configuration
func DefaultDedupConfig() DedupConfig {
	return DedupConfig{
		Algorithm:         DedupFixed,
		BlockSize:         DefaultBlockSize,
		MinSizeBytes:      4 * 1024, // 4KB
		InlineSmallBlocks: true,
		StorePath:         "/var/lib/novacron/dedup",
		MaxStoreSize:      0, // Unlimited
		CompressBlocks:    true,
		UseBloomFilter:    true,
		VerifyOnRead:      true,
	}
}

// DedupBlockInfo contains information about a deduplicated block
type DedupBlockInfo struct {
	// The unique hash identifying this block
	Hash string `json:"hash"`

	// The size of the block in bytes
	Size int `json:"size"`

	// The offset within the original data
	Offset int `json:"offset"`

	// Reference count (how many times this block is used)
	RefCount int `json:"ref_count"`

	// Whether the block was inlined (stored directly rather than deduplicated)
	Inlined bool `json:"inlined"`

	// The inlined data (if Inlined is true)
	Data []byte `json:"data,omitempty"`
}

// DedupFileInfo contains information about a deduplicated file
type DedupFileInfo struct {
	// The original size of the file
	OriginalSize int64 `json:"original_size"`

	// The deduplicated size (sum of unique blocks)
	DedupSize int64 `json:"dedup_size"`

	// The blocks that make up this file
	Blocks []DedupBlockInfo `json:"blocks"`

	// The deduplication ratio achieved
	DedupRatio float64 `json:"dedup_ratio"`

	// The algorithm used for deduplication
	Algorithm DedupAlgorithm `json:"algorithm"`
}

// Deduplicator provides methods for deduplicating data
type Deduplicator struct {
	config DedupConfig
	mu     sync.RWMutex

	// blockStore maps hashes to block data
	blockStore map[string][]byte

	// blockRefCount maps hashes to reference counts
	blockRefCount map[string]int
}

// NewDeduplicator creates a new Deduplicator with the provided configuration
func NewDeduplicator(config DedupConfig) (*Deduplicator, error) {
	// Create the deduplication store directory if it doesn't exist
	if err := os.MkdirAll(config.StorePath, 0755); err != nil {
		return nil, fmt.Errorf("failed to create deduplication store directory: %w", err)
	}

	return &Deduplicator{
		config:        config,
		blockStore:    make(map[string][]byte),
		blockRefCount: make(map[string]int),
	}, nil
}

// Deduplicate breaks down data into blocks and stores them in the deduplication store
func (d *Deduplicator) Deduplicate(data []byte) (*DedupFileInfo, error) {
	// If deduplication is disabled, return a single inlined block
	if d.config.Algorithm == DedupNone {
		return &DedupFileInfo{
			OriginalSize: int64(len(data)),
			DedupSize:    int64(len(data)),
			Blocks: []DedupBlockInfo{
				{
					Hash:     hashBytes(data),
					Size:     len(data),
					Offset:   0,
					RefCount: 1,
					Inlined:  true,
					Data:     data,
				},
			},
			DedupRatio: 1.0,
			Algorithm:  DedupNone,
		}, nil
	}

	// Check if the data is large enough to deduplicate
	if len(data) < d.config.MinSizeBytes {
		// For small data, just inline it
		return &DedupFileInfo{
			OriginalSize: int64(len(data)),
			DedupSize:    int64(len(data)),
			Blocks: []DedupBlockInfo{
				{
					Hash:     hashBytes(data),
					Size:     len(data),
					Offset:   0,
					RefCount: 1,
					Inlined:  true,
					Data:     data,
				},
			},
			DedupRatio: 1.0,
			Algorithm:  d.config.Algorithm,
		}, nil
	}

	// Break down data into blocks based on the algorithm
	var blocks []DedupBlockInfo
	var totalDedupSize int64
	var err error

	switch d.config.Algorithm {
	case DedupFixed:
		blocks, totalDedupSize, err = d.deduplicateFixedSize(data)
	case DedupVariable:
		blocks, totalDedupSize, err = d.deduplicateVariableSize(data)
	case DedupContent:
		blocks, totalDedupSize, err = d.deduplicateContentDefined(data)
	default:
		return nil, fmt.Errorf("unsupported deduplication algorithm: %s", d.config.Algorithm)
	}

	if err != nil {
		return nil, err
	}

	// Calculate deduplication ratio
	dedupRatio := float64(len(data)) / float64(totalDedupSize)
	if totalDedupSize == 0 {
		dedupRatio = 1.0
	}

	return &DedupFileInfo{
		OriginalSize: int64(len(data)),
		DedupSize:    totalDedupSize,
		Blocks:       blocks,
		DedupRatio:   dedupRatio,
		Algorithm:    d.config.Algorithm,
	}, nil
}

// deduplicateFixedSize breaks data into fixed-size blocks
func (d *Deduplicator) deduplicateFixedSize(data []byte) ([]DedupBlockInfo, int64, error) {
	var blocks []DedupBlockInfo
	var totalDedupSize int64

	// Acquire write lock since we may modify the block store
	d.mu.Lock()
	defer d.mu.Unlock()

	for offset := 0; offset < len(data); offset += d.config.BlockSize {
		// Calculate the block size (last block may be smaller)
		blockSize := d.config.BlockSize
		if offset+blockSize > len(data) {
			blockSize = len(data) - offset
		}

		// Extract the block
		block := data[offset : offset+blockSize]

		// Hash the block
		hash := hashBytes(block)

		// Check if the block is small enough to inline
		inlined := false
		var blockData []byte = nil
		if d.config.InlineSmallBlocks && blockSize <= 1024 {
			inlined = true
			blockData = block
		}

		// Check if we already have this block
		if existingBlock, exists := d.blockStore[hash]; exists {
			// Block already exists, increment reference count
			d.blockRefCount[hash]++

			// Verify the block is the same
			if !bytes.Equal(existingBlock, block) {
				return nil, 0, fmt.Errorf("hash collision detected for block at offset %d", offset)
			}

			// Only count the unique blocks towards the deduplicated size
			if d.blockRefCount[hash] == 1 {
				totalDedupSize += int64(blockSize)
			}
		} else {
			// New block, store it
			d.blockStore[hash] = block
			d.blockRefCount[hash] = 1
			totalDedupSize += int64(blockSize)

			// Save the block to disk
			if err := d.saveBlockToDisk(hash, block); err != nil {
				return nil, 0, err
			}
		}

		// Add the block info
		blocks = append(blocks, DedupBlockInfo{
			Hash:     hash,
			Size:     blockSize,
			Offset:   offset,
			RefCount: d.blockRefCount[hash],
			Inlined:  inlined,
			Data:     blockData,
		})
	}

	return blocks, totalDedupSize, nil
}

// deduplicateVariableSize breaks data into variable-size blocks based on content boundaries
func (d *Deduplicator) deduplicateVariableSize(data []byte) ([]DedupBlockInfo, int64, error) {
	// This is a simplified implementation of variable-size chunking
	// A real implementation would use a rolling hash to identify natural boundaries

	var blocks []DedupBlockInfo
	var totalDedupSize int64

	// Minimum and maximum block sizes
	minSize := d.config.BlockSize / 4
	maxSize := d.config.BlockSize * 4

	// Acquire write lock since we may modify the block store
	d.mu.Lock()
	defer d.mu.Unlock()

	offset := 0
	for offset < len(data) {
		// Find the next boundary
		// In a real implementation, this would use a rolling hash and look for certain patterns
		// For simplicity, we'll just look for byte sequences that might indicate natural boundaries
		boundaryFound := false
		for i := minSize; i < maxSize && offset+i < len(data); i++ {
			// Look for sequences of repeated bytes or certain patterns
			if i >= minSize && (i >= maxSize-1 || (data[offset+i] == 0 && data[offset+i-1] == 0) || (data[offset+i] == '\n' && data[offset+i-1] == '\r')) {
				// Extract the block
				block := data[offset : offset+i]

				// Hash the block
				hash := hashBytes(block)

				// Check if the block is small enough to inline
				inlined := false
				var blockData []byte = nil
				if d.config.InlineSmallBlocks && i <= 1024 {
					inlined = true
					blockData = block
				}

				// Check if we already have this block
				if existingBlock, exists := d.blockStore[hash]; exists {
					// Block already exists, increment reference count
					d.blockRefCount[hash]++

					// Verify the block is the same
					if !bytes.Equal(existingBlock, block) {
						return nil, 0, fmt.Errorf("hash collision detected for block at offset %d", offset)
					}

					// Only count the unique blocks towards the deduplicated size
					if d.blockRefCount[hash] == 1 {
						totalDedupSize += int64(i)
					}
				} else {
					// New block, store it
					d.blockStore[hash] = block
					d.blockRefCount[hash] = 1
					totalDedupSize += int64(i)

					// Save the block to disk
					if err := d.saveBlockToDisk(hash, block); err != nil {
						return nil, 0, err
					}
				}

				// Add the block info
				blocks = append(blocks, DedupBlockInfo{
					Hash:     hash,
					Size:     i,
					Offset:   offset,
					RefCount: d.blockRefCount[hash],
					Inlined:  inlined,
					Data:     blockData,
				})

				// Move to the next block
				offset += i
				boundaryFound = true
				break
			}
		}

		// If no boundary was found, use the maximum size
		if !boundaryFound {
			blockSize := maxSize
			if offset+blockSize > len(data) {
				blockSize = len(data) - offset
			}

			// Extract the block
			block := data[offset : offset+blockSize]

			// Hash the block
			hash := hashBytes(block)

			// Check if the block is small enough to inline
			inlined := false
			var blockData []byte = nil
			if d.config.InlineSmallBlocks && blockSize <= 1024 {
				inlined = true
				blockData = block
			}

			// Check if we already have this block
			if existingBlock, exists := d.blockStore[hash]; exists {
				// Block already exists, increment reference count
				d.blockRefCount[hash]++

				// Verify the block is the same
				if !bytes.Equal(existingBlock, block) {
					return nil, 0, fmt.Errorf("hash collision detected for block at offset %d", offset)
				}

				// Only count the unique blocks towards the deduplicated size
				if d.blockRefCount[hash] == 1 {
					totalDedupSize += int64(blockSize)
				}
			} else {
				// New block, store it
				d.blockStore[hash] = block
				d.blockRefCount[hash] = 1
				totalDedupSize += int64(blockSize)

				// Save the block to disk
				if err := d.saveBlockToDisk(hash, block); err != nil {
					return nil, 0, err
				}
			}

			// Add the block info
			blocks = append(blocks, DedupBlockInfo{
				Hash:     hash,
				Size:     blockSize,
				Offset:   offset,
				RefCount: d.blockRefCount[hash],
				Inlined:  inlined,
				Data:     blockData,
			})

			// Move to the next block
			offset += blockSize
		}
	}

	return blocks, totalDedupSize, nil
}

// deduplicateContentDefined breaks data into blocks based on content-defined chunking
func (d *Deduplicator) deduplicateContentDefined(data []byte) ([]DedupBlockInfo, int64, error) {
	// Content-defined chunking (CDC) uses a rolling hash to identify chunk boundaries
	// This is a simplified implementation

	var blocks []DedupBlockInfo
	var totalDedupSize int64

	// Parameters for content-defined chunking
	// These would be tuned based on the workload
	minSize := d.config.BlockSize / 2
	// targetSize is the ideal block size (but actual sizes will vary based on content boundaries)
	// Using targetSize in mask calculation (the mask bit length controls avg chunk size)
	_ = d.config.BlockSize // Acknowledge that this target is built into the mask value
	maxSize := d.config.BlockSize * 2
	windowSize := 16 // Window size for the rolling hash

	// Mask for determining chunk boundaries
	// Lower values create smaller chunks
	// Higher values create larger chunks
	// This should be a power of 2 minus 1
	mask := uint32(0x00001FFF) // Avg chunk size ~8KB with 13 bits

	// Acquire write lock since we may modify the block store
	d.mu.Lock()
	defer d.mu.Unlock()

	offset := 0
	for offset < len(data) {
		// Find the next chunk boundary using the rolling hash
		// Start at minimum size to avoid tiny chunks
		nextOffset := offset + minSize
		found := false

		// Avoid going beyond the end of the data
		if nextOffset > len(data) {
			nextOffset = len(data)
		} else {
			// Try to find a boundary using the rolling hash
			for i := nextOffset; i < offset+maxSize && i < len(data)-windowSize; i++ {
				// Compute a very simple rolling hash
				// A real implementation would use a proper rolling hash like Rabin-Karp
				var hash uint32
				for j := 0; j < windowSize; j++ {
					hash = ((hash << 1) | (hash >> 31)) ^ uint32(data[i+j])
				}

				// Check if this is a boundary
				// When (hash & mask) == 0, we've found a chunk boundary
				if (hash & mask) == 0 {
					nextOffset = i + 1
					found = true
					break
				}
			}

			// If no boundary found or we're beyond max size, use max size
			if !found && nextOffset < offset+maxSize && offset+maxSize < len(data) {
				nextOffset = offset + maxSize
			} else if nextOffset > len(data) {
				nextOffset = len(data)
			}
		}

		// Extract the chunk
		blockSize := nextOffset - offset
		block := data[offset:nextOffset]

		// Hash the block
		hash := hashBytes(block)

		// Check if the block is small enough to inline
		inlined := false
		var blockData []byte = nil
		if d.config.InlineSmallBlocks && blockSize <= 1024 {
			inlined = true
			blockData = block
		}

		// Check if we already have this block
		if existingBlock, exists := d.blockStore[hash]; exists {
			// Block already exists, increment reference count
			d.blockRefCount[hash]++

			// Verify the block is the same
			if !bytes.Equal(existingBlock, block) {
				return nil, 0, fmt.Errorf("hash collision detected for block at offset %d", offset)
			}

			// Only count the unique blocks towards the deduplicated size
			if d.blockRefCount[hash] == 1 {
				totalDedupSize += int64(blockSize)
			}
		} else {
			// New block, store it
			d.blockStore[hash] = block
			d.blockRefCount[hash] = 1
			totalDedupSize += int64(blockSize)

			// Save the block to disk
			if err := d.saveBlockToDisk(hash, block); err != nil {
				return nil, 0, err
			}
		}

		// Add the block info
		blocks = append(blocks, DedupBlockInfo{
			Hash:     hash,
			Size:     blockSize,
			Offset:   offset,
			RefCount: d.blockRefCount[hash],
			Inlined:  inlined,
			Data:     blockData,
		})

		// Move to the next block
		offset = nextOffset
	}

	return blocks, totalDedupSize, nil
}

// Reconstruct rebuilds the original data from deduplicated blocks
func (d *Deduplicator) Reconstruct(fileInfo *DedupFileInfo) ([]byte, error) {
	// Recreate the original data buffer
	result := make([]byte, fileInfo.OriginalSize)

	// Read lock is sufficient as we're only reading from the block store
	d.mu.RLock()
	defer d.mu.RUnlock()

	// For each block in the file
	for _, block := range fileInfo.Blocks {
		// If the block is inlined, use the inlined data
		if block.Inlined && block.Data != nil {
			// Copy the inlined data to the result
			copy(result[block.Offset:block.Offset+block.Size], block.Data)
			continue
		}

		// Check if we have the block in memory
		blockData, exists := d.blockStore[block.Hash]
		if !exists {
			// If not in memory, try to load from disk
			var err error
			blockData, err = d.loadBlockFromDisk(block.Hash)
			if err != nil {
				return nil, fmt.Errorf("failed to load block %s: %w", block.Hash, err)
			}
		}

		// Verify the block size matches
		if len(blockData) != block.Size {
			return nil, fmt.Errorf("block size mismatch for hash %s: expected %d, got %d",
				block.Hash, block.Size, len(blockData))
		}

		// Copy the block data to the result at the correct offset
		copy(result[block.Offset:block.Offset+block.Size], blockData)
	}

	return result, nil
}

// RemoveFile decrements reference counts for blocks in a file
func (d *Deduplicator) RemoveFile(fileInfo *DedupFileInfo) error {
	// Acquire write lock since we're modifying reference counts
	d.mu.Lock()
	defer d.mu.Unlock()

	// For each block in the file
	for _, block := range fileInfo.Blocks {
		// Skip inlined blocks as they don't affect the block store
		if block.Inlined {
			continue
		}

		// Decrement the reference count
		if count, exists := d.blockRefCount[block.Hash]; exists {
			if count > 1 {
				// Decrement the reference count
				d.blockRefCount[block.Hash] = count - 1
			} else {
				// Last reference, remove the block
				delete(d.blockStore, block.Hash)
				delete(d.blockRefCount, block.Hash)

				// Remove the block from disk
				if err := d.removeBlockFromDisk(block.Hash); err != nil {
					return fmt.Errorf("failed to remove block %s: %w", block.Hash, err)
				}
			}
		}
	}

	return nil
}

// GetStats returns statistics about the deduplication store
func (d *Deduplicator) GetStats() map[string]interface{} {
	d.mu.RLock()
	defer d.mu.RUnlock()

	var totalSize int64
	var totalBlocks int
	var totalRefs int

	for hash, block := range d.blockStore {
		totalSize += int64(len(block))
		totalBlocks++
		totalRefs += d.blockRefCount[hash]
	}

	return map[string]interface{}{
		"algorithm":     d.config.Algorithm,
		"block_size":    d.config.BlockSize,
		"unique_blocks": totalBlocks,
		"total_refs":    totalRefs,
		"total_size":    totalSize,
		"avg_refs":      float64(totalRefs) / float64(totalBlocks),
		"store_path":    d.config.StorePath,
	}
}

// saveBlockToDisk saves a block to the disk store
func (d *Deduplicator) saveBlockToDisk(hash string, data []byte) error {
	// Create the directory structure based on the first few characters of the hash
	// This helps avoid too many files in a single directory
	dirPath := filepath.Join(d.config.StorePath, hash[:2], hash[2:4])
	if err := os.MkdirAll(dirPath, 0755); err != nil {
		return fmt.Errorf("failed to create directory %s: %w", dirPath, err)
	}

	// Full path to the block file
	blockPath := filepath.Join(dirPath, hash)

	// Write the block to disk
	return os.WriteFile(blockPath, data, 0644)
}

// loadBlockFromDisk loads a block from the disk store
func (d *Deduplicator) loadBlockFromDisk(hash string) ([]byte, error) {
	// Construct the block path
	blockPath := filepath.Join(d.config.StorePath, hash[:2], hash[2:4], hash)

	// Read the block from disk
	data, err := os.ReadFile(blockPath)
	if err != nil {
		return nil, err
	}

	// Store in memory for future reference
	d.blockStore[hash] = data

	return data, nil
}

// removeBlockFromDisk removes a block from the disk store
func (d *Deduplicator) removeBlockFromDisk(hash string) error {
	// Construct the block path
	blockPath := filepath.Join(d.config.StorePath, hash[:2], hash[2:4], hash)

	// Remove the block file
	return os.Remove(blockPath)
}

// hashBytes calculates a SHA-256 hash of a byte slice and returns it as a hex string
func hashBytes(data []byte) string {
	hash := sha256.New()
	hash.Write(data)
	return hex.EncodeToString(hash.Sum(nil))
}

// Cleanup performs cleanup operations on the deduplication store
func (d *Deduplicator) Cleanup() error {
	// Acquire write lock since we're modifying the block store
	d.mu.Lock()
	defer d.mu.Unlock()

	// Clear in-memory data
	d.blockStore = make(map[string][]byte)
	d.blockRefCount = make(map[string]int)

	return nil
}

// ErrBlockNotFound indicates a block was not found in the store
var ErrBlockNotFound = errors.New("block not found in store")
