package vm_test

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/sirupsen/logrus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	vm "novacron/backend/core/vm"
)

func TestDeltaSync_FileCreation(t *testing.T) {
	tmpDir := t.TempDir()

	// Create a test source file
	sourcePath := filepath.Join(tmpDir, "source.img")
	sourceData := make([]byte, 10*1024*1024) // 10MB
	for i := range sourceData {
		sourceData[i] = byte(i % 256)
	}

	err := os.WriteFile(sourcePath, sourceData, 0644)
	require.NoError(t, err)

	// Create delta sync manager
	config := vm.DefaultDeltaSyncConfig()
	config.TempDir = tmpDir
	config.BlockSizeKB = 64

	manager := vm.NewDeltaSyncManager(config)
	require.NotNil(t, manager)
	defer manager.Close()

	// Test signature file creation
	sigPath := filepath.Join(tmpDir, "signature.dat")
	ctx := context.Background()

	err = manager.CreateSignatureFile(ctx, sourcePath, sigPath)
	assert.NoError(t, err, "Should create signature file")
	assert.FileExists(t, sigPath, "Signature file should exist")

	// Verify signature file size
	sigInfo, err := os.Stat(sigPath)
	require.NoError(t, err)
	assert.Greater(t, sigInfo.Size(), int64(0), "Signature file should not be empty")

	t.Logf("Signature file size: %d bytes", sigInfo.Size())
}

func TestDeltaSync_FullTransfer(t *testing.T) {
	tmpDir := t.TempDir()

	// Create source file
	sourcePath := filepath.Join(tmpDir, "source.img")
	sourceData := make([]byte, 5*1024*1024) // 5MB
	for i := range sourceData {
		sourceData[i] = byte(i % 256)
	}
	err := os.WriteFile(sourcePath, sourceData, 0644)
	require.NoError(t, err)

	// Destination path (doesn't exist yet)
	destPath := filepath.Join(tmpDir, "dest.img")

	// Create manager
	config := vm.DefaultDeltaSyncConfig()
	config.TempDir = tmpDir
	config.BlockSizeKB = 64
	config.EnableCompression = true

	manager := vm.NewDeltaSyncManager(config)
	defer manager.Close()

	// Perform sync
	ctx := context.Background()
	err = manager.SyncFile(ctx, sourcePath, destPath)
	assert.NoError(t, err, "Sync should succeed")

	// Verify destination file exists and has correct content
	assert.FileExists(t, destPath)

	destData, err := os.ReadFile(destPath)
	require.NoError(t, err)
	assert.Equal(t, sourceData, destData, "Destination should match source")

	// Check stats
	stats := manager.GetStats()
	assert.Equal(t, int64(len(sourceData)), stats.TotalBytes)
	assert.Greater(t, stats.TransferredBytes, int64(0))

	t.Logf("Stats: Total=%d, Transferred=%d, Saved=%d (%.2f%%)",
		stats.TotalBytes, stats.TransferredBytes, stats.BytesSaved, stats.BytesSavedPercent)
}

func TestDeltaSync_IncrementalUpdate(t *testing.T) {
	tmpDir := t.TempDir()

	// Create initial source and destination
	sourcePath := filepath.Join(tmpDir, "source.img")
	destPath := filepath.Join(tmpDir, "dest.img")

	initialData := make([]byte, 10*1024*1024) // 10MB
	for i := range initialData {
		initialData[i] = byte(i % 256)
	}

	err := os.WriteFile(sourcePath, initialData, 0644)
	require.NoError(t, err)

	err = os.WriteFile(destPath, initialData, 0644)
	require.NoError(t, err)

	// Modify source slightly (change 1MB in the middle)
	modifiedData := make([]byte, len(initialData))
	copy(modifiedData, initialData)

	modifyStart := 5 * 1024 * 1024
	modifyEnd := 6 * 1024 * 1024
	for i := modifyStart; i < modifyEnd; i++ {
		modifiedData[i] = byte((i + 128) % 256)
	}

	err = os.WriteFile(sourcePath, modifiedData, 0644)
	require.NoError(t, err)

	// Perform delta sync
	config := vm.DefaultDeltaSyncConfig()
	config.TempDir = tmpDir
	config.BlockSizeKB = 64

	manager := vm.NewDeltaSyncManager(config)
	defer manager.Close()

	ctx := context.Background()
	err = manager.SyncFile(ctx, sourcePath, destPath)
	assert.NoError(t, err, "Delta sync should succeed")

	// Verify result
	destData, err := os.ReadFile(destPath)
	require.NoError(t, err)
	assert.Equal(t, modifiedData, destData, "Destination should match modified source")

	// Check that we saved significant bandwidth
	stats := manager.GetStats()
	assert.Greater(t, stats.DuplicateBlocks, 0, "Should have duplicate blocks")
	assert.Greater(t, stats.BytesSavedPercent, 50.0, "Should save >50% bandwidth")

	t.Logf("Incremental sync stats:")
	t.Logf("  Total blocks: %d", stats.DuplicateBlocks+stats.UniqueBlocks)
	t.Logf("  Duplicate blocks: %d", stats.DuplicateBlocks)
	t.Logf("  Unique blocks: %d", stats.UniqueBlocks)
	t.Logf("  Bytes saved: %.2f%%", stats.BytesSavedPercent)
}

func TestDeltaSync_WithEBPF(t *testing.T) {
	if !vm.IsEBPFSupported() {
		t.Skip("eBPF not supported")
	}

	tmpDir := t.TempDir()

	// Create test file
	sourcePath := filepath.Join(tmpDir, "source.img")
	sourceData := make([]byte, 10*1024*1024) // 10MB
	for i := range sourceData {
		sourceData[i] = byte(i % 256)
	}
	err := os.WriteFile(sourcePath, sourceData, 0644)
	require.NoError(t, err)

	destPath := filepath.Join(tmpDir, "dest.img")

	// Create manager with eBPF enabled
	config := vm.DefaultDeltaSyncConfig()
	config.TempDir = tmpDir
	config.BlockSizeKB = 64
	config.EnableEBPFFiltering = true
	config.FallbackOnEBPFError = true

	manager := vm.NewDeltaSyncManager(config)
	defer manager.Close()

	// Try to enable eBPF (may fail if not root, which is OK)
	pid := uint32(os.Getpid())
	err = manager.EnableEBPFFiltering(pid)
	if err != nil {
		t.Logf("eBPF not enabled (may require root): %v", err)
	}

	ebpfEnabled := manager.IsEBPFEnabled()
	t.Logf("eBPF enabled: %v", ebpfEnabled)

	// Perform sync
	ctx := context.Background()
	err = manager.SyncFile(ctx, sourcePath, destPath)
	assert.NoError(t, err, "Sync should succeed even if eBPF failed")

	// Check stats
	stats := manager.GetStats()
	t.Logf("Delta sync stats with eBPF:")
	t.Logf("  eBPF enabled: %v", stats.EBPFEnabled)
	t.Logf("  Total bytes: %d", stats.TotalBytes)
	t.Logf("  Transferred: %d", stats.TransferredBytes)

	if stats.EBPFEnabled {
		t.Logf("  eBPF blocks skipped: %d", stats.EBPFBlocksSkipped)
		t.Logf("  eBPF bytes skipped: %d", stats.EBPFBytesSkipped)
		t.Logf("  eBPF skip percent: %.2f%%", stats.EBPFSkipPercent)
	}
}

func TestDeltaSync_LargeFile(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping large file test in short mode")
	}

	tmpDir := t.TempDir()

	// Create large test file (100MB)
	sourcePath := filepath.Join(tmpDir, "large.img")
	sourceSize := 100 * 1024 * 1024

	// Create file with pattern
	file, err := os.Create(sourcePath)
	require.NoError(t, err)

	buffer := make([]byte, 1024*1024) // 1MB buffer
	for i := 0; i < sourceSize/(1024*1024); i++ {
		for j := range buffer {
			buffer[j] = byte((i + j) % 256)
		}
		_, err = file.Write(buffer)
		require.NoError(t, err)
	}
	file.Close()

	destPath := filepath.Join(tmpDir, "large_dest.img")

	// Sync with default config
	config := vm.DefaultDeltaSyncConfig()
	config.TempDir = tmpDir
	config.BlockSizeKB = 128 // Larger blocks for large files
	config.HashWorkers = 8   // More workers

	manager := vm.NewDeltaSyncManager(config)
	defer manager.Close()

	ctx := context.Background()
	start := time.Now()

	err = manager.SyncFile(ctx, sourcePath, destPath)
	assert.NoError(t, err)

	duration := time.Since(start)

	stats := manager.GetStats()
	throughputMBps := float64(stats.TotalBytes) / duration.Seconds() / (1024 * 1024)

	t.Logf("Large file sync completed:")
	t.Logf("  Size: %d MB", sourceSize/(1024*1024))
	t.Logf("  Duration: %v", duration)
	t.Logf("  Throughput: %.2f MB/s", throughputMBps)
	t.Logf("  Hashing duration: %v", stats.HashingDuration)
}

func TestDeltaSync_Compression(t *testing.T) {
	tmpDir := t.TempDir()

	// Create compressible data (lots of zeros)
	sourcePath := filepath.Join(tmpDir, "compressible.img")
	sourceData := make([]byte, 10*1024*1024) // 10MB of mostly zeros
	for i := 0; i < len(sourceData); i += 1000 {
		sourceData[i] = byte(i % 256) // Sparse data
	}
	err := os.WriteFile(sourcePath, sourceData, 0644)
	require.NoError(t, err)

	destPath := filepath.Join(tmpDir, "dest.img")

	// Test with compression enabled
	config := vm.DefaultDeltaSyncConfig()
	config.TempDir = tmpDir
	config.EnableCompression = true
	config.CompressionLevel = 6

	manager := vm.NewDeltaSyncManager(config)
	defer manager.Close()

	ctx := context.Background()
	err = manager.SyncFile(ctx, sourcePath, destPath)
	assert.NoError(t, err)

	stats := manager.GetStats()
	compressionRatio := float64(stats.TotalBytes) / float64(stats.TransferredBytes)

	t.Logf("Compression results:")
	t.Logf("  Original size: %d", stats.TotalBytes)
	t.Logf("  Transferred: %d", stats.TransferredBytes)
	t.Logf("  Compression ratio: %.2fx", compressionRatio)

	assert.Greater(t, compressionRatio, 1.5, "Should achieve reasonable compression")
}
