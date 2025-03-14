package main

import (
	"context"
	"log"
	"os"
	"path/filepath"
	"time"

	"github.com/google/uuid"
	"github.com/khryptorgraphics/novacron/backend/core/vm"
)

// Example demonstrating WAN migration optimization
func main() {
	log.Println("Starting WAN Migration Optimization Example")

	// Create temporary directories for source and destination files
	tmpDir, err := os.MkdirTemp("", "wan-migration-test")
	if err != nil {
		log.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	// Create source and destination directories
	sourceDir := filepath.Join(tmpDir, "source")
	destDir := filepath.Join(tmpDir, "dest")

	for _, dir := range []string{sourceDir, destDir} {
		if err := os.MkdirAll(dir, 0755); err != nil {
			log.Fatalf("Failed to create directory %s: %v", dir, err)
		}
	}

	// Create a test disk file
	sourceDiskPath := filepath.Join(sourceDir, "disk.img")
	createTestFile(sourceDiskPath, 100*1024*1024) // 100MB test file

	destDiskPath := filepath.Join(destDir, "disk.img")

	// Create WAN migration optimizer
	config := vm.DefaultWANMigrationConfig()
	config.CompressionLevel = 6   // Higher compression for WAN
	config.EnableDeltaSync = true // Enable delta sync
	config.ChunkSizeKB = 256      // 256KB chunks
	config.Parallelism = 4        // 4 parallel transfers
	config.MaxBandwidthMbps = 100 // 100 Mbps limit
	config.QoSPriority = 5        // Higher priority for migration traffic

	optimizer := vm.NewWANMigrationOptimizer(config)

	// Log starting info
	log.Printf("Starting migration with WAN optimization")
	log.Printf("Source disk: %s (%d bytes)", sourceDiskPath, fileSize(sourceDiskPath))
	log.Printf("Destination: %s", destDiskPath)
	log.Printf("Configuration: Compression=%d, DeltaSync=%v, Parallelism=%d, BandwidthLimit=%d Mbps",
		config.CompressionLevel, config.EnableDeltaSync, config.Parallelism, config.MaxBandwidthMbps)

	// Simulate network conditions for WAN
	networkConditions := simulateNetworkConditions()
	log.Printf("Network conditions: Bandwidth=%.1f Mbps, Latency=%v, Packet loss=%.2f%%",
		networkConditions.bandwidthMbps, networkConditions.latency, networkConditions.packetLossPct)

	// Tune the optimizer for the network conditions
	optimizer.TuneForNetwork(
		networkConditions.bandwidthMbps,
		networkConditions.latency,
		networkConditions.packetLossPct/100.0, // Convert percent to fraction
	)

	// Create a delta sync manager
	deltaSyncConfig := vm.DefaultDeltaSyncConfig()
	deltaSyncConfig.CompressionLevel = config.CompressionLevel
	deltaSyncConfig.BlockSizeKB = 64
	deltaSyncManager := vm.NewDeltaSyncManager(deltaSyncConfig)
	defer deltaSyncManager.Close()

	// Integrate with the WAN migration optimizer
	deltaSyncManager.IntegrateWithWANMigrationOptimizer(optimizer)

	// Perform delta sync
	ctx := context.Background()
	startTime := time.Now()

	log.Println("Starting delta synchronization of VM disk")
	err = deltaSyncManager.SyncFile(ctx, sourceDiskPath, destDiskPath)
	if err != nil {
		log.Fatalf("Failed to sync file: %v", err)
	}

	duration := time.Since(startTime)

	// Get stats
	syncStats := deltaSyncManager.GetStats()
	optimizerStats := optimizer.GetStats()

	// Display results
	log.Printf("Migration completed in %v", duration)
	log.Printf("Total bytes: %d", syncStats.TotalBytes)
	log.Printf("Transferred bytes: %d (%.1f%%)",
		syncStats.TransferredBytes,
		float64(syncStats.TransferredBytes)/float64(syncStats.TotalBytes)*100)
	log.Printf("Bytes saved with delta sync: %d (%.1f%%)",
		syncStats.BytesSaved,
		syncStats.BytesSavedPercent)
	log.Printf("Average bandwidth: %.2f Mbps", optimizerStats.AverageBandwidthMbps)
	log.Printf("Total downtime: %dms", optimizerStats.TotalDowntimeMs)

	log.Printf("Migration verification: %v", verifyMigration(sourceDiskPath, destDiskPath))
}

// networkCondition represents simulated WAN network conditions
type networkCondition struct {
	bandwidthMbps float64
	latency       time.Duration
	packetLossPct float64
}

// simulateNetworkConditions returns simulated network conditions for a WAN link
func simulateNetworkConditions() networkCondition {
	// Simulate a typical WAN connection
	return networkCondition{
		bandwidthMbps: 50 + float64(time.Now().UnixNano()%50),                                         // 50-100 Mbps
		latency:       50*time.Millisecond + time.Duration(time.Now().UnixNano()%50)*time.Millisecond, // 50-100ms
		packetLossPct: 0.1 + float64(time.Now().UnixNano()%10)/100.0,                                  // 0.1-0.2%
	}
}

// createTestFile creates a test file of the specified size with random data
func createTestFile(path string, size int64) {
	file, err := os.Create(path)
	if err != nil {
		log.Fatalf("Failed to create test file: %v", err)
	}
	defer file.Close()

	// Generate a unique pattern so we don't just have zeros
	// (which would compress too easily and not be representative)
	data := make([]byte, 1024)
	for i := 0; i < 1024; i++ {
		data[i] = byte(uuid.New().ID() & 0xFF)
	}

	// Write the pattern repeatedly until we reach the desired size
	remaining := size
	for remaining > 0 {
		writeSize := int64(len(data))
		if remaining < writeSize {
			writeSize = remaining
		}

		_, err := file.Write(data[:writeSize])
		if err != nil {
			log.Fatalf("Failed to write to test file: %v", err)
		}

		remaining -= writeSize
	}
}

// fileSize returns the size of a file in bytes
func fileSize(path string) int64 {
	info, err := os.Stat(path)
	if err != nil {
		log.Fatalf("Failed to get file info: %v", err)
		return 0
	}
	return info.Size()
}

// verifyMigration verifies that the source and destination files are identical
func verifyMigration(sourcePath, destPath string) bool {
	sourceHash, err := vm.CalculateFileChecksum(sourcePath)
	if err != nil {
		log.Printf("Failed to calculate source checksum: %v", err)
		return false
	}

	destHash, err := vm.CalculateFileChecksum(destPath)
	if err != nil {
		log.Printf("Failed to calculate destination checksum: %v", err)
		return false
	}

	return sourceHash == destHash
}
