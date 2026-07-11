package deduplication

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"math/bits"
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
		// For small data, store it as a single block. Whether that block is
		// inlined (data carried in the DedupBlockInfo itself) or persisted to
		// the block store must still respect InlineSmallBlocks — this path
		// used to always inline regardless of config, which both ignored the
		// setting and, since Deduplicate never wrote the block to disk either
		// way, left RemoveFile's disk cleanup with a block hash that had
		// nothing to remove.
		hash := hashBytes(data)
		block := DedupBlockInfo{
			Hash:     hash,
			Size:     len(data),
			Offset:   0,
			RefCount: 1,
			Inlined:  d.config.InlineSmallBlocks,
		}
		if d.config.InlineSmallBlocks {
			block.Data = data
		} else {
			d.mu.Lock()
			if _, exists := d.blockStore[hash]; exists {
				d.blockRefCount[hash]++
			} else {
				d.blockStore[hash] = data
				d.blockRefCount[hash] = 1
				if err := d.saveBlockToDisk(hash, data); err != nil {
					d.mu.Unlock()
					return nil, err
				}
			}
			block.RefCount = d.blockRefCount[hash]
			d.mu.Unlock()
		}

		return &DedupFileInfo{
			OriginalSize: int64(len(data)),
			DedupSize:    int64(len(data)),
			Blocks:       []DedupBlockInfo{block},
			DedupRatio:   1.0,
			Algorithm:    d.config.Algorithm,
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

		// Check if the block is small enough to inline. "Small" is relative
		// to the configured block size: a full-size block (blockSize ==
		// d.config.BlockSize) is a normal block, not a small fragment, even
		// when BlockSize itself is small (e.g. BlockSize: 1024 in tests).
		// Only a genuinely undersized block — the last, partial chunk of a
		// file — is a candidate for inlining. Without the BlockSize
		// comparison, small-BlockSize configs would inline every full block,
		// routing them around the block store entirely and defeating
		// deduplication.
		inlined := d.config.InlineSmallBlocks && blockSize < d.config.BlockSize && blockSize <= 1024

		var blockData []byte
		var refCount int
		if inlined {
			// Inlined blocks carry their data directly in DedupBlockInfo.Data
			// and must NOT also be written to the shared block store/disk:
			// RemoveFile skips inlined blocks when cleaning up, so anything
			// persisted here would never be reclaimed. They also aren't
			// deduplicated against the store, so each occurrence counts
			// fully towards the deduplicated size.
			blockData = block
			refCount = 1
			totalDedupSize += int64(blockSize)
		} else if existingBlock, exists := d.blockStore[hash]; exists {
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
			refCount = d.blockRefCount[hash]
		} else {
			// New block, store it
			d.blockStore[hash] = block
			d.blockRefCount[hash] = 1
			totalDedupSize += int64(blockSize)

			// Save the block to disk
			if err := d.saveBlockToDisk(hash, block); err != nil {
				return nil, 0, err
			}
			refCount = 1
		}

		// Add the block info
		blocks = append(blocks, DedupBlockInfo{
			Hash:     hash,
			Size:     blockSize,
			Offset:   offset,
			RefCount: refCount,
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

				// Check if the block is small enough to inline. "Small" is
				// relative to this algorithm's own minSize floor, not a bare
				// constant — see deduplicateFixedSize/deduplicateContentDefined
				// for why a bare "<=1024" check inlines (and thus silently
				// skips deduplicating) every chunk in small-BlockSize configs.
				inlined := d.config.InlineSmallBlocks && i < minSize && i <= 1024

				var blockData []byte
				var refCount int
				if inlined {
					// Inlined blocks carry their data directly in
					// DedupBlockInfo.Data and must NOT also be written to the
					// shared block store/disk — see deduplicateFixedSize.
					blockData = block
					refCount = 1
					totalDedupSize += int64(i)
				} else if existingBlock, exists := d.blockStore[hash]; exists {
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
					refCount = d.blockRefCount[hash]
				} else {
					// New block, store it
					d.blockStore[hash] = block
					d.blockRefCount[hash] = 1
					totalDedupSize += int64(i)

					// Save the block to disk
					if err := d.saveBlockToDisk(hash, block); err != nil {
						return nil, 0, err
					}
					refCount = 1
				}

				// Add the block info
				blocks = append(blocks, DedupBlockInfo{
					Hash:     hash,
					Size:     i,
					Offset:   offset,
					RefCount: refCount,
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

			// Check if the block is small enough to inline. "Small" is
			// relative to minSize, not a bare constant — see the
			// boundary-found branch above for the full explanation.
			inlined := d.config.InlineSmallBlocks && blockSize < minSize && blockSize <= 1024

			var blockData []byte
			var refCount int
			if inlined {
				// Inlined blocks carry their data directly in
				// DedupBlockInfo.Data and must NOT also be written to the
				// shared block store/disk — see deduplicateFixedSize.
				blockData = block
				refCount = 1
				totalDedupSize += int64(blockSize)
			} else if existingBlock, exists := d.blockStore[hash]; exists {
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
				refCount = d.blockRefCount[hash]
			} else {
				// New block, store it
				d.blockStore[hash] = block
				d.blockRefCount[hash] = 1
				totalDedupSize += int64(blockSize)

				// Save the block to disk
				if err := d.saveBlockToDisk(hash, block); err != nil {
					return nil, 0, err
				}
				refCount = 1
			}

			// Add the block info
			blocks = append(blocks, DedupBlockInfo{
				Hash:     hash,
				Size:     blockSize,
				Offset:   offset,
				RefCount: refCount,
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
	windowSize := 16 // Window size for the rolling hash

	// The average chunk size the mask targets is a fraction of BlockSize, not
	// BlockSize itself. Boundary detection only works if the [minSize,
	// maxSize) scan actually turns up a matching window before giving up and
	// falling back to a forced maxSize cut; with the average pinned to the
	// full BlockSize, that window was too coarse relative to a single
	// BlockSize-sized chunk, so a repeated block's boundary routinely wasn't
	// re-found once its surrounding content (and thus the scan's starting
	// offset within it) changed — duplicates went undetected even with a
	// correctly-sized mask. A finer target gives the scan many more chances
	// to hit a genuine content boundary, which is what makes duplicate
	// detection actually reliable.
	target := d.config.BlockSize / 8
	if target < windowSize*2 {
		target = windowSize * 2
	}
	minSize := target / 2
	maxSize := d.config.BlockSize
	if maxSize < target*2 {
		maxSize = target * 2
	}

	// Mask for determining chunk boundaries. A boundary is declared once
	// roughly 1-in-2^popcount(mask) window hashes match, so the average chunk
	// size is ~2^popcount(mask); the mask must therefore be derived from the
	// target size above, not hardcoded. It used to be a fixed 13-bit mask
	// (~8KB average chunks) regardless of config.
	targetBits := bits.Len(uint(target))
	if targetBits > 0 {
		targetBits--
	}
	mask := uint32(1)<<uint(targetBits) - 1

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

		// Check if the block is small enough to inline. As in
		// deduplicateFixedSize, "small" must be relative to this algorithm's
		// own sizing, not a bare constant — but here that means smaller than
		// minSize (the floor content-defined chunking targets), not smaller
		// than BlockSize: maxSize is capped at BlockSize, so nearly every
		// chunk is already < BlockSize by construction, and comparing against
		// BlockSize would inline (and thus never deduplicate) almost
		// everything.
		inlined := d.config.InlineSmallBlocks && blockSize < minSize && blockSize <= 1024

		var blockData []byte
		if inlined {
			// Inlined blocks carry their data directly in DedupBlockInfo.Data
			// and must NOT also be written to the shared block store/disk —
			// see deduplicateFixedSize for the full explanation.
			blockData = block
			totalDedupSize += int64(blockSize)
		} else if existingBlock, exists := d.blockStore[hash]; exists {
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

		// Add the block info. Inlined blocks never touch blockRefCount (they
		// bypass the shared store), so their ref count is always 1 — a map
		// lookup here would silently read back the zero value instead.
		refCount := 1
		if !inlined {
			refCount = d.blockRefCount[hash]
		}
		blocks = append(blocks, DedupBlockInfo{
			Hash:     hash,
			Size:     blockSize,
			Offset:   offset,
			RefCount: refCount,
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

		// Decrement the reference count. If this deduplicator instance never
		// called Deduplicate for this hash (e.g. it only loaded the block via
		// Reconstruct after a restart, so loadBlockFromDisk populated
		// blockStore but not blockRefCount), fall back to the count recorded
		// in the file's own metadata at the time it was deduplicated — the
		// best available signal for how many references currently exist.
		// Without this fallback, RemoveFile silently no-ops for every block
		// a fresh instance didn't itself create, leaking them on disk forever.
		count, exists := d.blockRefCount[block.Hash]
		if !exists {
			count = block.RefCount
		}

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
