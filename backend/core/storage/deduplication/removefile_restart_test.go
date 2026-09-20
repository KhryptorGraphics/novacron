package deduplication

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

// TestRemoveFileAfterRestartRepeatedBlock is a regression test for a census
// flake (2026-09-20): when a file contains a block whose hash occurs three
// or more times, RemoveFile on a FRESH deduplicator instance (one that never
// called Deduplicate — e.g. after a restart, where blocks are only loaded
// via Reconstruct) removes the block from disk more than once and fails
// with ENOENT on the second removal.
//
// Pre-fix behavior: RemoveFile falls back to each block entry's per-
// occurrence RefCount metadata when the fresh instance has no map entry.
// For occurrences [1, 2, 3] of one hash: occurrence 1 -> metadata count 1 ->
// removes from disk; occurrence 2 -> map miss -> count 2 -> map becomes 1;
// occurrence 3 -> map hit count 1 -> treated as last reference -> second
// disk removal -> ENOENT. The census flake fired only when generateTestData
// (unseeded math/rand, auto-seeded since Go 1.20) happened to emit a block
// three or more times; this test builds the data by hand so it exercises
// the path on every run.
//
// Both the same-instance path and the restart path must leave zero block
// files on disk: an ENOENT-swallowing "fix" that leaked files would fail the
// same-instance assertion below.
func TestRemoveFileAfterRestartRepeatedBlock(t *testing.T) {
	tempDir, err := os.MkdirTemp("", "dedup-restart-repeat-*")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	config := DefaultDedupConfig()
	config.StorePath = tempDir
	config.Algorithm = DedupFixed
	config.BlockSize = 1024
	config.MinSizeBytes = 1  // route 3 KiB through the fixed-size path
	config.InlineSmallBlocks = false // force every block through the store

	deduplicator, err := NewDeduplicator(config)
	if err != nil {
		t.Fatalf("Failed to create deduplicator: %v", err)
	}

	// One 1 KiB block content, repeated three times in one file.
	block := bytes.Repeat([]byte("REPEATED-BLOCK-"), 72)[:1024]
	data := bytes.Repeat(block, 3)

	fileInfo, err := deduplicator.Deduplicate(data)
	if err != nil {
		t.Fatalf("Failed to deduplicate data: %v", err)
	}
	if len(fileInfo.Blocks) != 3 {
		t.Fatalf("Expected 3 block entries, got %d", len(fileInfo.Blocks))
	}
	hash := fileInfo.Blocks[0].Hash
	for _, b := range fileInfo.Blocks {
		if b.Hash != hash {
			t.Fatalf("Expected all blocks to share one hash, got %s and %s", hash, b.Hash)
		}
	}

	// Same-instance removal must succeed and leave the store empty.
	if err := deduplicator.RemoveFile(fileInfo); err != nil {
		t.Fatalf("RemoveFile (same instance) failed: %v", err)
	}
	if n := countFiles(t, tempDir); n != 0 {
		t.Errorf("Same-instance removal left %d block files, want 0", n)
	}

	// Rededuplicate onto disk, then simulate a restart: memory cleared,
	// blocks only on disk, loaded lazily via Reconstruct.
	deduplicator.Cleanup()
	fresh, err := NewDeduplicator(config)
	if err != nil {
		t.Fatalf("Failed to create fresh deduplicator: %v", err)
	}
	fileInfo2, err := fresh.Deduplicate(data)
	if err != nil {
		t.Fatalf("Failed to rededuplicate data: %v", err)
	}
	fresh.Cleanup() // drop memory state, keep disk state
	if _, err := fresh.Reconstruct(fileInfo2); err != nil {
		t.Fatalf("Failed to reconstruct: %v", err)
	}

	// The pre-fix failure point: the third occurrence triggers a second
	// disk removal and RemoveFile returns ENOENT.
	if err := fresh.RemoveFile(fileInfo2); err != nil {
		t.Fatalf("RemoveFile after restart failed for repeated block: %v", err)
	}
	if n := countFiles(t, tempDir); n != 0 {
		t.Errorf("Restart removal left %d block files, want 0", n)
	}
}

// countFiles returns the number of regular files under dir.
func countFiles(t *testing.T, dir string) int {
	t.Helper()
	count := 0
	err := filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() {
			count++
		}
		return nil
	})
	if err != nil {
		t.Fatalf("Failed to walk directory: %v", err)
	}
	return count
}
