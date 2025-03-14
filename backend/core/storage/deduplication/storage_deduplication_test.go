package deduplication

import (
	"bytes"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
)

func TestDeduplication(t *testing.T) {
	// Create a temporary directory for the test
	tempDir, err := os.MkdirTemp("", "deduplication-test-*")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// Create deduplicator with the temp directory
	config := DefaultDedupConfig()
	config.StorePath = tempDir
	config.MinSizeBytes = 1 // Set low to ensure all test data is processed

	deduplicator, err := NewDeduplicator(config)
	if err != nil {
		t.Fatalf("Failed to create deduplicator: %v", err)
	}

	// Generate test data with some repetitive patterns
	data := generateTestData(1024 * 10) // 10KB data with repetition

	// Test for each algorithm
	algorithms := []DedupAlgorithm{
		DedupNone,
		DedupFixed,
		DedupVariable,
		DedupContent,
	}

	for _, algorithm := range algorithms {
		t.Run(string(algorithm), func(t *testing.T) {
			// Update the algorithm
			config.Algorithm = algorithm
			deduplicator, err = NewDeduplicator(config)
			if err != nil {
				t.Fatalf("Failed to create deduplicator: %v", err)
			}

			// Deduplicate the data
			fileInfo, err := deduplicator.Deduplicate(data)
			if err != nil {
				t.Fatalf("Failed to deduplicate data: %v", err)
			}

			// Check the file info
			if fileInfo.OriginalSize != int64(len(data)) {
				t.Errorf("Expected original size %d, got %d", len(data), fileInfo.OriginalSize)
			}

			if fileInfo.Algorithm != algorithm {
				t.Errorf("Expected algorithm %s, got %s", algorithm, fileInfo.Algorithm)
			}

			// For DedupNone, verify there's only one block that spans the entire data
			if algorithm == DedupNone {
				if len(fileInfo.Blocks) != 1 {
					t.Errorf("Expected 1 block for DedupNone, got %d", len(fileInfo.Blocks))
				}
				if !fileInfo.Blocks[0].Inlined {
					t.Errorf("Expected block to be inlined for DedupNone")
				}
				if fileInfo.Blocks[0].Size != len(data) {
					t.Errorf("Expected block size %d, got %d", len(data), fileInfo.Blocks[0].Size)
				}
			} else {
				// For other algorithms, verify we have multiple blocks
				if len(fileInfo.Blocks) <= 1 {
					t.Errorf("Expected multiple blocks, got %d", len(fileInfo.Blocks))
				}

				// Check block consistency
				totalSize := 0
				for _, block := range fileInfo.Blocks {
					if block.Size <= 0 {
						t.Errorf("Invalid block size: %d", block.Size)
					}
					if block.Offset < 0 || block.Offset+block.Size > len(data) {
						t.Errorf("Invalid block offset or size: offset=%d, size=%d", block.Offset, block.Size)
					}
					totalSize += block.Size
				}

				// Check that blocks cover the entire data
				if totalSize != len(data) {
					t.Errorf("Blocks don't cover entire data: total=%d, data=%d", totalSize, len(data))
				}
			}

			// Reconstruct the data
			reconstructed, err := deduplicator.Reconstruct(fileInfo)
			if err != nil {
				t.Fatalf("Failed to reconstruct data: %v", err)
			}

			// Verify the reconstructed data matches the original
			if !bytes.Equal(reconstructed, data) {
				t.Errorf("Reconstructed data doesn't match original")
			}

			// Check deduplication ratio (even no-op deduplication should report a ratio)
			if fileInfo.DedupRatio <= 0 {
				t.Errorf("Expected positive deduplication ratio, got %f", fileInfo.DedupRatio)
			}

			// Remove the file and clean up
			err = deduplicator.RemoveFile(fileInfo)
			if err != nil {
				t.Fatalf("Failed to remove file: %v", err)
			}
		})
	}
}

func TestDeduplicationWithDuplicateData(t *testing.T) {
	// Create a temporary directory for the test
	tempDir, err := os.MkdirTemp("", "deduplication-test-*")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// Create deduplicator with the temp directory
	config := DefaultDedupConfig()
	config.StorePath = tempDir
	config.Algorithm = DedupFixed
	config.BlockSize = 1024 // 1KB blocks

	deduplicator, err := NewDeduplicator(config)
	if err != nil {
		t.Fatalf("Failed to create deduplicator: %v", err)
	}

	// Generate test data with high duplication
	dataBlock := bytes.Repeat([]byte("ABCDEFGH"), 128) // 1KB block
	data := bytes.Repeat(dataBlock, 10)                // 10KB data, all identical blocks

	// Deduplicate the data
	fileInfo, err := deduplicator.Deduplicate(data)
	if err != nil {
		t.Fatalf("Failed to deduplicate data: %v", err)
	}

	// Check deduplication statistics
	if fileInfo.OriginalSize != int64(len(data)) {
		t.Errorf("Expected original size %d, got %d", len(data), fileInfo.OriginalSize)
	}

	// With perfect duplication, DedupSize should be much smaller than OriginalSize
	// Fixed-size deduplication with identical blocks should achieve high ratio
	if fileInfo.DedupRatio < 5.0 {
		t.Errorf("Expected high deduplication ratio for duplicate data, got %f", fileInfo.DedupRatio)
	}

	// Should have many blocks, but only one unique block (all have same hash)
	uniqueHashes := make(map[string]bool)
	for _, block := range fileInfo.Blocks {
		uniqueHashes[block.Hash] = true
	}

	// Should be exactly 1 unique hash for purely duplicate data
	if len(uniqueHashes) != 1 {
		t.Errorf("Expected 1 unique block hash, got %d", len(uniqueHashes))
	}

	// Get stats and verify
	stats := deduplicator.GetStats()
	uniqueBlocks := stats["unique_blocks"].(int)
	totalRefs := stats["total_refs"].(int)

	// Should be 1 unique block and 10 references
	if uniqueBlocks != 1 {
		t.Errorf("Expected 1 unique block, got %d", uniqueBlocks)
	}
	if totalRefs != 10 {
		t.Errorf("Expected 10 total references, got %d", totalRefs)
	}

	// Reconstruct and verify
	reconstructed, err := deduplicator.Reconstruct(fileInfo)
	if err != nil {
		t.Fatalf("Failed to reconstruct data: %v", err)
	}
	if !bytes.Equal(reconstructed, data) {
		t.Errorf("Reconstructed data doesn't match original")
	}
}

func TestDeduplicationWithMixedData(t *testing.T) {
	// Create a temporary directory for the test
	tempDir, err := os.MkdirTemp("", "deduplication-test-*")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// Create deduplicator with the temp directory
	config := DefaultDedupConfig()
	config.StorePath = tempDir
	config.Algorithm = DedupContent
	config.BlockSize = 1024 // 1KB target size

	deduplicator, err := NewDeduplicator(config)
	if err != nil {
		t.Fatalf("Failed to create deduplicator: %v", err)
	}

	// Generate mixed data - some duplicated, some unique
	repeatedBlock := bytes.Repeat([]byte("REPEATED"), 128)  // 1KB
	uniqueBlock1 := generateRandomData(1024)                // 1KB
	uniqueBlock2 := generateRandomData(1024)                // 1KB
	textBlock := bytes.Repeat([]byte("This is text. "), 64) // 960B

	// Create data with a mix of repeated and unique content
	data := bytes.Buffer{}
	data.Write(repeatedBlock)
	data.Write(uniqueBlock1)
	data.Write(repeatedBlock)
	data.Write(textBlock)
	data.Write(uniqueBlock2)
	data.Write(repeatedBlock)
	data.Write(repeatedBlock)
	data.Write(uniqueBlock1)
	data.Write(textBlock)
	data.Write(repeatedBlock)

	// Deduplicate the data
	fileInfo, err := deduplicator.Deduplicate(data.Bytes())
	if err != nil {
		t.Fatalf("Failed to deduplicate data: %v", err)
	}

	// Check for a reasonable deduplication ratio with mixed data
	// Not as high as with pure duplicates, but should still be > 1.0
	if fileInfo.DedupRatio <= 1.0 || fileInfo.DedupRatio > 5.0 {
		t.Errorf("Unexpected deduplication ratio for mixed data: %f", fileInfo.DedupRatio)
	}

	// Reconstruct and verify
	reconstructed, err := deduplicator.Reconstruct(fileInfo)
	if err != nil {
		t.Fatalf("Failed to reconstruct data: %v", err)
	}
	if !bytes.Equal(reconstructed, data.Bytes()) {
		t.Errorf("Reconstructed data doesn't match original")
	}
}

func TestInliningSmallBlocks(t *testing.T) {
	// Create a temporary directory for the test
	tempDir, err := os.MkdirTemp("", "deduplication-test-*")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// Create deduplicator with the temp directory
	config := DefaultDedupConfig()
	config.StorePath = tempDir
	config.Algorithm = DedupFixed
	config.BlockSize = 4096    // 4KB blocks
	config.MinSizeBytes = 1024 // Process files >= 1KB

	// Test both with and without inlining
	testCases := []struct {
		name              string
		inlineSmallBlocks bool
		expectedInlined   bool
	}{
		{
			name:              "With Inlining",
			inlineSmallBlocks: true,
			expectedInlined:   true,
		},
		{
			name:              "Without Inlining",
			inlineSmallBlocks: false,
			expectedInlined:   false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			config.InlineSmallBlocks = tc.inlineSmallBlocks
			deduplicator, err := NewDeduplicator(config)
			if err != nil {
				t.Fatalf("Failed to create deduplicator: %v", err)
			}

			// Create a small data block (smaller than default inlining threshold)
			smallData := []byte("This is a small piece of data that should be inlined")

			// Deduplicate the data
			fileInfo, err := deduplicator.Deduplicate(smallData)
			if err != nil {
				t.Fatalf("Failed to deduplicate data: %v", err)
			}

			// Check if the block was inlined
			if len(fileInfo.Blocks) != 1 {
				t.Fatalf("Expected 1 block, got %d", len(fileInfo.Blocks))
			}

			if fileInfo.Blocks[0].Inlined != tc.expectedInlined {
				t.Errorf("Inlining was %v, expected %v", fileInfo.Blocks[0].Inlined, tc.expectedInlined)
			}

			// If inlined, data should be present
			if tc.expectedInlined && fileInfo.Blocks[0].Data == nil {
				t.Errorf("Block was inlined but data is nil")
			}

			// If not inlined, data should be nil
			if !tc.expectedInlined && fileInfo.Blocks[0].Data != nil {
				t.Errorf("Block was not inlined but data is not nil")
			}

			// Reconstruct and verify
			reconstructed, err := deduplicator.Reconstruct(fileInfo)
			if err != nil {
				t.Fatalf("Failed to reconstruct data: %v", err)
			}
			if !bytes.Equal(reconstructed, smallData) {
				t.Errorf("Reconstructed data doesn't match original")
			}
		})
	}
}

func TestDeduplicationStore(t *testing.T) {
	// Create a temporary directory for the test
	tempDir, err := os.MkdirTemp("", "deduplication-store-test-*")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// Create deduplicator with the temp directory
	config := DefaultDedupConfig()
	config.StorePath = tempDir
	config.Algorithm = DedupFixed
	config.BlockSize = 1024 // 1KB blocks

	deduplicator, err := NewDeduplicator(config)
	if err != nil {
		t.Fatalf("Failed to create deduplicator: %v", err)
	}

	// Generate test data
	data := generateTestData(1024 * 5) // 5KB

	// Deduplicate the data
	fileInfo, err := deduplicator.Deduplicate(data)
	if err != nil {
		t.Fatalf("Failed to deduplicate data: %v", err)
	}

	// Verify blocks were saved to disk
	blocksFound := 0
	err = filepath.Walk(tempDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() {
			blocksFound++
		}
		return nil
	})
	if err != nil {
		t.Fatalf("Failed to walk directory: %v", err)
	}

	// Count unique blocks
	uniqueHashes := make(map[string]bool)
	for _, block := range fileInfo.Blocks {
		if !block.Inlined {
			uniqueHashes[block.Hash] = true
		}
	}

	// Verify the number of blocks on disk matches the number of unique blocks
	if blocksFound != len(uniqueHashes) {
		t.Errorf("Expected %d blocks on disk, found %d", len(uniqueHashes), blocksFound)
	}

	// Clean memory but keep data on disk
	deduplicator.Cleanup()

	// Create a new deduplicator instance to test loading from disk
	newDeduplicator, err := NewDeduplicator(config)
	if err != nil {
		t.Fatalf("Failed to create new deduplicator: %v", err)
	}

	// Reconstruct data from new instance (will require loading from disk)
	reconstructed, err := newDeduplicator.Reconstruct(fileInfo)
	if err != nil {
		t.Fatalf("Failed to reconstruct data: %v", err)
	}

	// Verify the reconstructed data matches the original
	if !bytes.Equal(reconstructed, data) {
		t.Errorf("Reconstructed data from disk doesn't match original")
	}

	// Remove the file and clean up
	err = newDeduplicator.RemoveFile(fileInfo)
	if err != nil {
		t.Fatalf("Failed to remove file: %v", err)
	}

	// Verify blocks were removed from disk
	blocksRemaining := 0
	err = filepath.Walk(tempDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() {
			blocksRemaining++
		}
		return nil
	})
	if err != nil {
		t.Fatalf("Failed to walk directory: %v", err)
	}

	// Should be 0 blocks remaining
	if blocksRemaining != 0 {
		t.Errorf("Expected 0 blocks after removal, found %d", blocksRemaining)
	}
}

// Helper functions to generate test data

// generateTestData creates data with some repetition to test deduplication
func generateTestData(size int) []byte {
	// Create some repeated blocks of data
	repeatedBlock1 := bytes.Repeat([]byte("REPEATED-BLOCK-1-"), 64)
	repeatedBlock2 := bytes.Repeat([]byte("REPEATED-BLOCK-2-"), 64)

	// Create some random data
	randomBlock1 := generateRandomData(1024)
	randomBlock2 := generateRandomData(1024)

	// Create a buffer to hold all the data
	buffer := bytes.Buffer{}

	// Fill the buffer to the requested size
	for buffer.Len() < size {
		// Mix repeated and random data
		switch rand.Intn(4) {
		case 0:
			buffer.Write(repeatedBlock1)
		case 1:
			buffer.Write(repeatedBlock2)
		case 2:
			buffer.Write(randomBlock1)
		case 3:
			buffer.Write(randomBlock2)
		}
	}

	// Trim to exact size
	return buffer.Bytes()[:size]
}

// generateRandomData creates random data of the specified size
func generateRandomData(size int) []byte {
	data := make([]byte, size)
	rand.Read(data)
	return data
}
