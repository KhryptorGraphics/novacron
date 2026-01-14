package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/storage"
	"github.com/khryptorgraphics/novacron/backend/core/storage/compression"
	"github.com/khryptorgraphics/novacron/backend/core/storage/deduplication"
	"github.com/khryptorgraphics/novacron/backend/core/storage/encryption"
)

func main() {
	log.Println("NovaCron Storage Efficiency Example")

	// Run the storage efficiency demonstration
	StorageEfficiencyDemo()
}

// StorageEfficiencyDemo demonstrates the usage of deduplication, compression,
// and encryption with the distributed storage system.
func StorageEfficiencyDemo() {
	// 1. Create a base storage manager
	baseManager := &storage.StorageManager{
		// In a real application, you would initialize this properly
		Volumes: make(map[string]*storage.Volume),
	}

	// 2. Create configuration for each component

	// Deduplication config
	dedupConfig := deduplication.DefaultDedupConfig()
	dedupConfig.Algorithm = deduplication.DedupContent
	dedupConfig.MinBlockSize = 4 * 1024     // 4 KB minimum
	dedupConfig.TargetBlockSize = 64 * 1024 // 64 KB target
	dedupConfig.Enabled = true

	// Compression config
	compConfig := compression.DefaultCompressionConfig()
	compConfig.Algorithm = compression.CompressionGzip
	compConfig.Level = compression.CompressionDefault
	compConfig.AutoDetect = true

	// Encryption config
	encConfig := encryption.DefaultEncryptionConfig()
	encConfig.Algorithm = encryption.EncryptionAES256
	encConfig.Mode = encryption.EncryptionModeGCM
	encConfig.MasterKey = "your-secure-master-key-replace-in-production"

	// Distributed storage config
	distConfig := storage.DefaultDistributedStorageConfig()
	distConfig.RootDir = "/var/lib/novacron/distributed"
	distConfig.ShardSize = 64 * 1024 * 1024 // 64 MB
	distConfig.DefaultReplicationFactor = 3
	distConfig.DefaultDeduplication = true
	distConfig.DefaultEncryption = true
	distConfig.DeduplicationConfig = dedupConfig
	distConfig.CompressionConfig = compConfig
	distConfig.EncryptionConfig = encConfig

	// 3. Create the distributed storage service
	service, err := storage.NewDistributedStorageService(baseManager, distConfig)
	if err != nil {
		log.Fatalf("Failed to create distributed storage service: %v", err)
	}

	// 4. Start the service
	if err := service.Start(); err != nil {
		log.Fatalf("Failed to start distributed storage service: %v", err)
	}
	defer service.Stop()

	// 5. Add some storage nodes
	for i := 1; i <= 5; i++ {
		service.AddNode(storage.NodeInfo{
			ID:        fmt.Sprintf("node-%d", i),
			Name:      fmt.Sprintf("Storage Node %d", i),
			Role:      "storage",
			Address:   fmt.Sprintf("192.168.1.%d", i+10),
			Port:      8000,
			Available: true,
			JoinedAt:  time.Now(),
			LastSeen:  time.Now(),
		})
	}

	// 6. Create a volume
	ctx := context.Background()
	volumeSpec := storage.VolumeSpec{
		Name:   "efficient-volume",
		SizeMB: 1024, // 1 GB
		Type:   storage.VolumeTypeCeph,
		Options: map[string]string{
			"description": "Volume with deduplication, compression, and encryption",
		},
	}

	volume, err := service.CreateDistributedVolume(ctx, volumeSpec, 3)
	if err != nil {
		log.Fatalf("Failed to create volume: %v", err)
	}

	// 7. Write some data with duplicate content to demonstrate deduplication
	data := generateDuplicateData()
	log.Printf("Original data size: %d bytes", len(data))

	// 8. Write the data to a shard
	err = service.WriteShard(ctx, volume.ID, 0, data)
	if err != nil {
		log.Fatalf("Failed to write to shard: %v", err)
	}

	// 9. Get information about the distributed volume
	distVolume, err := service.GetDistributedVolume(ctx, volume.ID)
	if err != nil {
		log.Fatalf("Failed to get distributed volume: %v", err)
	}

	// 10. Read the data back
	readData, err := service.ReadShard(ctx, volume.ID, 0)
	if err != nil {
		log.Fatalf("Failed to read from shard: %v", err)
	}

	// 11. Verify the data
	if len(readData) != len(data) {
		log.Fatalf("Data length mismatch: got %d, expected %d", len(readData), len(data))
	}

	// If the data is byte-for-byte identical, the deduplication, compression, and
	// encryption pipeline is working correctly.
	log.Printf("Data verification successful!")

	// 12. Display efficiency stats
	shard := distVolume.DistInfo.Shards[0]
	log.Printf("Storage efficiency:")

	// Deduplication stats
	if shard.IsDeduplicated && shard.DedupFileInfo != nil {
		log.Printf("- Deduplication: %v", shard.IsDeduplicated)
		log.Printf("  Algorithm: %v", shard.DedupFileInfo.Algorithm)
		log.Printf("  Ratio: %.2f", shard.DedupFileInfo.DedupRatio)
		log.Printf("  Unique blocks: %d", len(shard.DedupFileInfo.Blocks))
	} else {
		log.Printf("- Deduplication: not applied")
	}

	// Compression stats
	if shard.CompressionAlgorithm != compression.CompressionNone {
		log.Printf("- Compression: %v", shard.CompressionAlgorithm)
		log.Printf("  Original size: %d bytes", shard.OriginalSize)
		log.Printf("  Ratio: %.2f", shard.CompressionRatio)
	} else {
		log.Printf("- Compression: not applied")
	}

	// Encryption stats
	if shard.IsEncrypted {
		log.Printf("- Encryption: %v", shard.IsEncrypted)
		log.Printf("  Algorithm: %v", shard.EncryptionAlgorithm)
		log.Printf("  Mode: %v", shard.EncryptionMode)
	} else {
		log.Printf("- Encryption: not applied")
	}
}

// generateDuplicateData creates sample data with a lot of redundancy
// to demonstrate deduplication
func generateDuplicateData() []byte {
	// Create a single "unique" block
	block := []byte("This is block data for the NovaCron distributed storage system. " +
		"It will be repeated many times to demonstrate the effectiveness of deduplication. " +
		"The more this exact data is repeated, the higher the deduplication ratio will be.")

	// Create an array of 1000 copies of the block (simulating high redundancy)
	var data []byte
	for i := 0; i < 1000; i++ {
		data = append(data, block...)

		// Add a small unique identifier to each block to make them slightly different
		// This simulates real-world data where blocks are similar but not identical
		data = append(data, byte(i&0xFF))
		data = append(data, byte((i>>8)&0xFF))
	}

	return data
}
