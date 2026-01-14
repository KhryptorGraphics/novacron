package vm_test

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	vm "novacron/backend/core/vm"
)

func BenchmarkDeltaSync_HashingOnly(b *testing.B) {
	tmpDir := b.TempDir()

	// Create test file
	testPath := filepath.Join(tmpDir, "test.img")
	testData := make([]byte, 100*1024*1024) // 100MB
	for i := range testData {
		testData[i] = byte(i % 256)
	}
	os.WriteFile(testPath, testData, 0644)

	config := vm.DefaultDeltaSyncConfig()
	config.TempDir = tmpDir
	config.BlockSizeKB = 64

	b.ResetTimer()
	b.SetBytes(int64(len(testData)))

	for i := 0; i < b.N; i++ {
		manager := vm.NewDeltaSyncManager(config)
		sigPath := filepath.Join(tmpDir, "sig.dat")
		manager.CreateSignatureFile(context.Background(), testPath, sigPath)
		manager.Close()
		os.Remove(sigPath)
	}
}

func BenchmarkDeltaSync_FullSync(b *testing.B) {
	tmpDir := b.TempDir()

	sourcePath := filepath.Join(tmpDir, "source.img")
	sourceData := make([]byte, 50*1024*1024) // 50MB
	for i := range sourceData {
		sourceData[i] = byte(i % 256)
	}
	os.WriteFile(sourcePath, sourceData, 0644)

	config := vm.DefaultDeltaSyncConfig()
	config.TempDir = tmpDir
	config.BlockSizeKB = 64

	b.ResetTimer()
	b.SetBytes(int64(len(sourceData)))

	for i := 0; i < b.N; i++ {
		destPath := filepath.Join(tmpDir, "dest.img")
		manager := vm.NewDeltaSyncManager(config)
		manager.SyncFile(context.Background(), sourcePath, destPath)
		manager.Close()
		os.Remove(destPath)
	}
}

func BenchmarkDeltaSync_IncrementalSync(b *testing.B) {
	tmpDir := b.TempDir()

	// Create initial data
	baseData := make([]byte, 50*1024*1024) // 50MB
	for i := range baseData {
		baseData[i] = byte(i % 256)
	}

	sourcePath := filepath.Join(tmpDir, "source.img")
	destPath := filepath.Join(tmpDir, "dest.img")

	// Write initial identical files
	os.WriteFile(sourcePath, baseData, 0644)
	os.WriteFile(destPath, baseData, 0644)

	// Modify source slightly (10%)
	modifiedData := make([]byte, len(baseData))
	copy(modifiedData, baseData)
	modifySize := len(baseData) / 10
	for i := 0; i < modifySize; i++ {
		modifiedData[i] = byte((i + 128) % 256)
	}
	os.WriteFile(sourcePath, modifiedData, 0644)

	config := vm.DefaultDeltaSyncConfig()
	config.TempDir = tmpDir
	config.BlockSizeKB = 64

	b.ResetTimer()
	b.SetBytes(int64(len(baseData)))

	for i := 0; i < b.N; i++ {
		// Reset dest to base
		os.WriteFile(destPath, baseData, 0644)

		manager := vm.NewDeltaSyncManager(config)
		manager.SyncFile(context.Background(), sourcePath, destPath)
		manager.Close()
	}
}

func BenchmarkDeltaSync_WithCompression(b *testing.B) {
	tmpDir := b.TempDir()

	// Create compressible data
	sourceData := make([]byte, 50*1024*1024) // 50MB
	for i := 0; i < len(sourceData); i += 100 {
		sourceData[i] = byte(i % 256)
	}

	sourcePath := filepath.Join(tmpDir, "source.img")
	os.WriteFile(sourcePath, sourceData, 0644)

	config := vm.DefaultDeltaSyncConfig()
	config.TempDir = tmpDir
	config.EnableCompression = true
	config.CompressionLevel = 3

	b.ResetTimer()
	b.SetBytes(int64(len(sourceData)))

	for i := 0; i < b.N; i++ {
		destPath := filepath.Join(tmpDir, "dest.img")
		manager := vm.NewDeltaSyncManager(config)
		manager.SyncFile(context.Background(), sourcePath, destPath)
		manager.Close()
		os.Remove(destPath)
	}
}

func BenchmarkDeltaSync_DifferentBlockSizes(b *testing.B) {
	tmpDir := b.TempDir()

	sourceData := make([]byte, 50*1024*1024) // 50MB
	for i := range sourceData {
		sourceData[i] = byte(i % 256)
	}

	sourcePath := filepath.Join(tmpDir, "source.img")
	os.WriteFile(sourcePath, sourceData, 0644)

	blockSizes := []int{32, 64, 128, 256, 512}

	for _, blockSize := range blockSizes {
		b.Run(blockSize+"KB", func(b *testing.B) {
			config := vm.DefaultDeltaSyncConfig()
			config.TempDir = tmpDir
			config.BlockSizeKB = blockSize

			b.ResetTimer()
			b.SetBytes(int64(len(sourceData)))

			for i := 0; i < b.N; i++ {
				destPath := filepath.Join(tmpDir, "dest.img")
				manager := vm.NewDeltaSyncManager(config)
				manager.SyncFile(context.Background(), sourcePath, destPath)
				manager.Close()
				os.Remove(destPath)
			}
		})
	}
}

func BenchmarkDeltaSync_ParallelHashing(b *testing.B) {
	tmpDir := b.TempDir()

	sourceData := make([]byte, 100*1024*1024) // 100MB
	for i := range sourceData {
		sourceData[i] = byte(i % 256)
	}

	sourcePath := filepath.Join(tmpDir, "source.img")
	os.WriteFile(sourcePath, sourceData, 0644)

	workers := []int{1, 2, 4, 8, 16}

	for _, numWorkers := range workers {
		b.Run(numWorkers+"Workers", func(b *testing.B) {
			config := vm.DefaultDeltaSyncConfig()
			config.TempDir = tmpDir
			config.HashWorkers = numWorkers

			b.ResetTimer()
			b.SetBytes(int64(len(sourceData)))

			for i := 0; i < b.N; i++ {
				manager := vm.NewDeltaSyncManager(config)
				sigPath := filepath.Join(tmpDir, "sig.dat")
				manager.CreateSignatureFile(context.Background(), sourcePath, sigPath)
				manager.Close()
				os.Remove(sigPath)
			}
		})
	}
}

func BenchmarkEBPF_PageTracking(b *testing.B) {
	if !vm.IsEBPFSupported() {
		b.Skip("eBPF not supported")
	}

	logger := createTestLogger()
	pid := uint32(os.Getpid())

	filter, err := vm.NewEBPFMigrationFilter(logger, pid)
	if err != nil {
		b.Skipf("Cannot create eBPF filter: %v", err)
	}
	defer filter.Close()

	err = filter.Attach()
	if err != nil {
		b.Skipf("Cannot attach eBPF: %v", err)
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		pfn := uint64(i % 10000)
		_ = filter.IsPageUnused(pfn)
	}
}

func BenchmarkEBPF_BlockFiltering(b *testing.B) {
	if !vm.IsEBPFSupported() {
		b.Skip("eBPF not supported")
	}

	logger := createTestLogger()
	pid := uint32(os.Getpid())

	filter, err := vm.NewEBPFMigrationFilter(logger, pid)
	if err != nil {
		b.Skipf("Cannot create eBPF filter: %v", err)
	}
	defer filter.Close()

	err = filter.Attach()
	if err != nil {
		b.Skipf("Cannot attach eBPF: %v", err)
	}

	blockSize := 64 * 1024
	blockFilter := vm.NewEBPFBlockFilter(filter, blockSize)

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		blockOffset := int64((i % 1000) * blockSize)
		_ = blockFilter.IsBlockUnused(blockOffset)
	}
}

func BenchmarkDeltaSync_WithEBPF(b *testing.B) {
	if !vm.IsEBPFSupported() {
		b.Skip("eBPF not supported")
	}

	tmpDir := b.TempDir()

	sourceData := make([]byte, 50*1024*1024) // 50MB
	for i := range sourceData {
		sourceData[i] = byte(i % 256)
	}

	sourcePath := filepath.Join(tmpDir, "source.img")
	os.WriteFile(sourcePath, sourceData, 0644)

	config := vm.DefaultDeltaSyncConfig()
	config.TempDir = tmpDir
	config.EnableEBPFFiltering = true
	config.FallbackOnEBPFError = true

	b.ResetTimer()
	b.SetBytes(int64(len(sourceData)))

	for i := 0; i < b.N; i++ {
		destPath := filepath.Join(tmpDir, "dest.img")
		manager := vm.NewDeltaSyncManager(config)

		// Try to enable eBPF (may fail if not root)
		manager.EnableEBPFFiltering(uint32(os.Getpid()))

		manager.SyncFile(context.Background(), sourcePath, destPath)
		manager.Close()
		os.Remove(destPath)
	}
}
