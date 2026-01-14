package storage

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/storage/compression"
	"github.com/khryptorgraphics/novacron/backend/core/storage/encryption"
)

func TestDistributedStorageService_CreateDistributedVolume(t *testing.T) {
	// Create base storage manager
	baseConfig := StorageManagerConfig{
		BasePath:    "/tmp/test-storage",
		Compression: false,
		Encryption:  false,
		Dedup:       false,
	}
	baseManager, err := NewStorageManager(baseConfig)
	if err != nil {
		t.Fatalf("Failed to create base manager: %v", err)
	}

	// Create distributed storage config
	distConfig := DefaultDistributedStorageConfig()
	distConfig.RootDir = "/tmp/test-distributed"

	// Create distributed storage service
	distService, err := NewDistributedStorageService(baseManager, distConfig)
	if err != nil {
		t.Fatalf("Failed to create distributed service: %v", err)
	}

	err = distService.Start()
	if err != nil {
		t.Fatalf("Failed to start distributed service: %v", err)
	}
	defer distService.Stop()

	// Add some nodes
	for i := 0; i < 3; i++ {
		node := NodeInfo{
			ID:        fmt.Sprintf("node-%d", i),
			Name:      fmt.Sprintf("test-node-%d", i),
			Address:   fmt.Sprintf("192.168.1.%d", i+10),
			Port:      8090,
			Available: true,
			JoinedAt:  time.Now(),
			LastSeen:  time.Now(),
		}
		distService.AddNode(node)
	}

	// Create a distributed volume
	opts := VolumeCreateOptions{
		Name:   "test-dist-volume",
		Type:   VolumeTypeDistributed,
		Size: 100 * 1024 * 1024, // 100MB in bytes
	}

	ctx := context.Background()
	volume, err := distService.CreateDistributedVolume(ctx, opts, 2) // Replication factor 2
	if err != nil {
		t.Fatalf("Failed to create distributed volume: %v", err)
	}

	if volume.Name != opts.Name {
		t.Errorf("Expected volume name %s, got %s", opts.Name, volume.Name)
	}

	// Verify distributed volume exists
	distVolume, err := distService.GetDistributedVolume(ctx, volume.ID)
	if err != nil {
		t.Fatalf("Failed to get distributed volume: %v", err)
	}

	if distVolume.DistInfo.ShardCount <= 0 {
		t.Error("Expected positive shard count")
	}

	if distVolume.DistInfo.NodeCount != 3 {
		t.Errorf("Expected node count 3, got %d", distVolume.DistInfo.NodeCount)
	}
}

func TestDistributedStorageService_ReadWriteShard(t *testing.T) {
	// Create base storage manager
	baseConfig := StorageManagerConfig{
		BasePath:    "/tmp/test-storage",
		Compression: false,
		Encryption:  false,
		Dedup:       false,
	}
	baseManager, err := NewStorageManager(baseConfig)
	if err != nil {
		t.Fatalf("Failed to create base manager: %v", err)
	}

	// Create distributed storage config
	distConfig := DefaultDistributedStorageConfig()
	distConfig.RootDir = "/tmp/test-distributed"

	// Create distributed storage service
	distService, err := NewDistributedStorageService(baseManager, distConfig)
	if err != nil {
		t.Fatalf("Failed to create distributed service: %v", err)
	}

	err = distService.Start()
	if err != nil {
		t.Fatalf("Failed to start distributed service: %v", err)
	}
	defer distService.Stop()

	// Add nodes
	for i := 0; i < 2; i++ {
		node := NodeInfo{
			ID:        fmt.Sprintf("node-%d", i),
			Name:      fmt.Sprintf("test-node-%d", i),
			Address:   fmt.Sprintf("192.168.1.%d", i+10),
			Port:      8090,
			Available: true,
			JoinedAt:  time.Now(),
			LastSeen:  time.Now(),
		}
		distService.AddNode(node)
	}

	// Create a distributed volume
	opts := VolumeCreateOptions{
		Name:   "test-rw-volume",
		Type:   VolumeTypeDistributed,
		Size: 10, // 10MB
	}

	ctx := context.Background()
	volume, err := distService.CreateDistributedVolume(ctx, opts, 1)
	if err != nil {
		t.Fatalf("Failed to create distributed volume: %v", err)
	}

	// Test write/read shard
	testData := []byte("Hello, distributed world!")
	shardIndex := 0

	err = distService.WriteShard(ctx, volume.ID, shardIndex, testData)
	if err != nil {
		t.Fatalf("Failed to write shard: %v", err)
	}

	readData, err := distService.ReadShard(ctx, volume.ID, shardIndex)
	if err != nil {
		t.Fatalf("Failed to read shard: %v", err)
	}

	if string(readData) != string(testData) {
		t.Errorf("Expected to read %s, got %s", string(testData), string(readData))
	}
}

func TestDistributedStorageService_WithCompression(t *testing.T) {
	// Create base storage manager
	baseConfig := StorageManagerConfig{
		BasePath:    "/tmp/test-storage",
		Compression: true,
		Encryption:  false,
		Dedup:       false,
	}
	baseManager, err := NewStorageManager(baseConfig)
	if err != nil {
		t.Fatalf("Failed to create base manager: %v", err)
	}

	// Create distributed storage config with compression
	distConfig := DefaultDistributedStorageConfig()
	distConfig.RootDir = "/tmp/test-distributed-comp"
	distConfig.CompressionConfig = compression.CompressionConfig{
		Algorithm: compression.CompressionGzip,
		Level:     6,
		MinSizeBytes: 1024,
		MaxSizeBytes: 1024 * 1024,
		AutoDetect: true,
	}

	// Create distributed storage service
	distService, err := NewDistributedStorageService(baseManager, distConfig)
	if err != nil {
		t.Fatalf("Failed to create distributed service: %v", err)
	}

	err = distService.Start()
	if err != nil {
		t.Fatalf("Failed to start distributed service: %v", err)
	}
	defer distService.Stop()

	// Add nodes
	for i := 0; i < 2; i++ {
		node := NodeInfo{
			ID:        fmt.Sprintf("node-%d", i),
			Name:      fmt.Sprintf("test-node-%d", i),
			Address:   fmt.Sprintf("192.168.1.%d", i+10),
			Port:      8090,
			Available: true,
			JoinedAt:  time.Now(),
			LastSeen:  time.Now(),
		}
		distService.AddNode(node)
	}

	// Create a distributed volume
	opts := VolumeCreateOptions{
		Name:   "test-comp-volume",
		Type:   VolumeTypeDistributed,
		Size: 10,
	}

	ctx := context.Background()
	volume, err := distService.CreateDistributedVolume(ctx, opts, 1)
	if err != nil {
		t.Fatalf("Failed to create distributed volume: %v", err)
	}

	// Test with compressible data
	testData := make([]byte, 1024)
	for i := range testData {
		testData[i] = byte('A') // Highly compressible
	}

	shardIndex := 0
	err = distService.WriteShard(ctx, volume.ID, shardIndex, testData)
	if err != nil {
		t.Fatalf("Failed to write compressed shard: %v", err)
	}

	readData, err := distService.ReadShard(ctx, volume.ID, shardIndex)
	if err != nil {
		t.Fatalf("Failed to read compressed shard: %v", err)
	}

	if len(readData) != len(testData) {
		t.Errorf("Expected decompressed data length %d, got %d", len(testData), len(readData))
	}

	// Verify data integrity
	for i, b := range readData {
		if b != testData[i] {
			t.Errorf("Data mismatch at byte %d: expected %d, got %d", i, testData[i], b)
			break
		}
	}
}

func TestDistributedStorageService_WithEncryption(t *testing.T) {
	// Create base storage manager
	baseConfig := StorageManagerConfig{
		BasePath:    "/tmp/test-storage",
		Compression: false,
		Encryption:  true,
		Dedup:       false,
	}
	baseManager, err := NewStorageManager(baseConfig)
	if err != nil {
		t.Fatalf("Failed to create base manager: %v", err)
	}

	// Create distributed storage config with encryption
	distConfig := DefaultDistributedStorageConfig()
	distConfig.RootDir = "/tmp/test-distributed-enc"
	distConfig.DefaultEncryption = true
	distConfig.EncryptionConfig = encryption.EncryptionConfig{
		Algorithm: encryption.EncryptionAES256,
		Mode:      encryption.EncryptionModeGCM,
		MasterKey: "test-master-key-32-bytes-long!!!",
		SaltPrefix: "novacron",
		Authenticate: true,
		MinSizeBytes: 1024,
	}

	// Create distributed storage service
	distService, err := NewDistributedStorageService(baseManager, distConfig)
	if err != nil {
		t.Fatalf("Failed to create distributed service: %v", err)
	}

	err = distService.Start()
	if err != nil {
		t.Fatalf("Failed to start distributed service: %v", err)
	}
	defer distService.Stop()

	// Add nodes
	for i := 0; i < 2; i++ {
		node := NodeInfo{
			ID:        fmt.Sprintf("node-%d", i),
			Name:      fmt.Sprintf("test-node-%d", i),
			Address:   fmt.Sprintf("192.168.1.%d", i+10),
			Port:      8090,
			Available: true,
			JoinedAt:  time.Now(),
			LastSeen:  time.Now(),
		}
		distService.AddNode(node)
	}

	// Create a distributed volume
	opts := VolumeCreateOptions{
		Name:   "test-enc-volume",
		Type:   VolumeTypeDistributed,
		Size: 10,
	}

	ctx := context.Background()
	volume, err := distService.CreateDistributedVolume(ctx, opts, 1)
	if err != nil {
		t.Fatalf("Failed to create distributed volume: %v", err)
	}

	// Test with sensitive data
	testData := []byte("This is sensitive data that should be encrypted!")

	shardIndex := 0
	err = distService.WriteShard(ctx, volume.ID, shardIndex, testData)
	if err != nil {
		t.Fatalf("Failed to write encrypted shard: %v", err)
	}

	readData, err := distService.ReadShard(ctx, volume.ID, shardIndex)
	if err != nil {
		t.Fatalf("Failed to read encrypted shard: %v", err)
	}

	if string(readData) != string(testData) {
		t.Errorf("Expected decrypted data %s, got %s", string(testData), string(readData))
	}
}

func TestDistributedStorageService_RepairVolume(t *testing.T) {
	// Create base storage manager
	baseConfig := StorageManagerConfig{
		BasePath:    "/tmp/test-storage",
		Compression: false,
		Encryption:  false,
		Dedup:       false,
	}
	baseManager, err := NewStorageManager(baseConfig)
	if err != nil {
		t.Fatalf("Failed to create base manager: %v", err)
	}

	// Create distributed storage config
	distConfig := DefaultDistributedStorageConfig()
	distConfig.RootDir = "/tmp/test-distributed-repair"

	// Create distributed storage service
	distService, err := NewDistributedStorageService(baseManager, distConfig)
	if err != nil {
		t.Fatalf("Failed to create distributed service: %v", err)
	}

	err = distService.Start()
	if err != nil {
		t.Fatalf("Failed to start distributed service: %v", err)
	}
	defer distService.Stop()

	// Add nodes
	for i := 0; i < 4; i++ {
		node := NodeInfo{
			ID:        fmt.Sprintf("node-%d", i),
			Name:      fmt.Sprintf("test-node-%d", i),
			Address:   fmt.Sprintf("192.168.1.%d", i+10),
			Port:      8090,
			Available: true,
			JoinedAt:  time.Now(),
			LastSeen:  time.Now(),
		}
		distService.AddNode(node)
	}

	// Create a distributed volume
	opts := VolumeCreateOptions{
		Name:   "test-repair-volume",
		Type:   VolumeTypeDistributed,
		Size: 10,
	}

	ctx := context.Background()
	volume, err := distService.CreateDistributedVolume(ctx, opts, 2) // Replication factor 2
	if err != nil {
		t.Fatalf("Failed to create distributed volume: %v", err)
	}

	// Get the distributed volume and mark some shards as needing healing
	distVolume, err := distService.GetDistributedVolume(ctx, volume.ID)
	if err != nil {
		t.Fatalf("Failed to get distributed volume: %v", err)
	}

	// Simulate shard failure
	if len(distVolume.DistInfo.Shards) > 0 {
		distVolume.DistInfo.Shards[0].NeedsHealing = true
	}

	// Test repair
	err = distService.RepairVolume(ctx, volume.ID)
	if err != nil {
		t.Fatalf("Failed to repair volume: %v", err)
	}

	// Verify health percentage improved
	repairedVolume, err := distService.GetDistributedVolume(ctx, volume.ID)
	if err != nil {
		t.Fatalf("Failed to get repaired volume: %v", err)
	}

	if repairedVolume.DistInfo.HealthPercentage < 90.0 {
		t.Errorf("Expected health percentage >= 90%%, got %.2f", repairedVolume.DistInfo.HealthPercentage)
	}
}

func TestDistributedStorageService_RebalanceVolume(t *testing.T) {
	// Create base storage manager
	baseConfig := StorageManagerConfig{
		BasePath:    "/tmp/test-storage",
		Compression: false,
		Encryption:  false,
		Dedup:       false,
	}
	baseManager, err := NewStorageManager(baseConfig)
	if err != nil {
		t.Fatalf("Failed to create base manager: %v", err)
	}

	// Create distributed storage config
	distConfig := DefaultDistributedStorageConfig()
	distConfig.RootDir = "/tmp/test-distributed-rebalance"

	// Create distributed storage service
	distService, err := NewDistributedStorageService(baseManager, distConfig)
	if err != nil {
		t.Fatalf("Failed to create distributed service: %v", err)
	}

	err = distService.Start()
	if err != nil {
		t.Fatalf("Failed to start distributed service: %v", err)
	}
	defer distService.Stop()

	// Add initial nodes
	for i := 0; i < 2; i++ {
		node := NodeInfo{
			ID:        fmt.Sprintf("node-%d", i),
			Name:      fmt.Sprintf("test-node-%d", i),
			Address:   fmt.Sprintf("192.168.1.%d", i+10),
			Port:      8090,
			Available: true,
			JoinedAt:  time.Now(),
			LastSeen:  time.Now(),
		}
		distService.AddNode(node)
	}

	// Create a distributed volume
	opts := VolumeCreateOptions{
		Name:   "test-rebalance-volume",
		Type:   VolumeTypeDistributed,
		Size: 10,
	}

	ctx := context.Background()
	volume, err := distService.CreateDistributedVolume(ctx, opts, 2)
	if err != nil {
		t.Fatalf("Failed to create distributed volume: %v", err)
	}

	// Add more nodes
	for i := 2; i < 4; i++ {
		node := NodeInfo{
			ID:        fmt.Sprintf("node-%d", i),
			Name:      fmt.Sprintf("test-node-%d", i),
			Address:   fmt.Sprintf("192.168.1.%d", i+10),
			Port:      8090,
			Available: true,
			JoinedAt:  time.Now(),
			LastSeen:  time.Now(),
		}
		distService.AddNode(node)
	}

	// Test rebalance
	err = distService.RebalanceVolume(ctx, volume.ID)
	if err != nil {
		t.Fatalf("Failed to rebalance volume: %v", err)
	}

	// Verify rebalancing updated the volume
	rebalancedVolume, err := distService.GetDistributedVolume(ctx, volume.ID)
	if err != nil {
		t.Fatalf("Failed to get rebalanced volume: %v", err)
	}

	if rebalancedVolume.DistInfo.NodeCount != 4 {
		t.Errorf("Expected node count 4 after rebalance, got %d", rebalancedVolume.DistInfo.NodeCount)
	}
}

func TestDistributedStorageService_NodeManagement(t *testing.T) {
	// Create base storage manager
	baseConfig := StorageManagerConfig{
		BasePath:    "/tmp/test-storage",
		Compression: false,
		Encryption:  false,
		Dedup:       false,
	}
	baseManager, err := NewStorageManager(baseConfig)
	if err != nil {
		t.Fatalf("Failed to create base manager: %v", err)
	}

	// Create distributed storage config
	distConfig := DefaultDistributedStorageConfig()
	distConfig.RootDir = "/tmp/test-distributed-nodes"

	// Create distributed storage service
	distService, err := NewDistributedStorageService(baseManager, distConfig)
	if err != nil {
		t.Fatalf("Failed to create distributed service: %v", err)
	}

	err = distService.Start()
	if err != nil {
		t.Fatalf("Failed to start distributed service: %v", err)
	}
	defer distService.Stop()

	// Test adding nodes
	for i := 0; i < 3; i++ {
		node := NodeInfo{
			ID:        fmt.Sprintf("node-%d", i),
			Name:      fmt.Sprintf("test-node-%d", i),
			Address:   fmt.Sprintf("192.168.1.%d", i+10),
			Port:      8090,
			Available: true,
			JoinedAt:  time.Now(),
			LastSeen:  time.Now(),
		}
		distService.AddNode(node)
	}

	// Test getting available nodes
	availableNodes := distService.GetAvailableNodes()
	if len(availableNodes) != 3 {
		t.Errorf("Expected 3 available nodes, got %d", len(availableNodes))
	}

	// Test removing a node
	distService.RemoveNode("node-1")
	availableNodes = distService.GetAvailableNodes()
	if len(availableNodes) != 2 {
		t.Errorf("Expected 2 available nodes after removal, got %d", len(availableNodes))
	}

	// Verify the correct node was removed
	for _, node := range availableNodes {
		if node.ID == "node-1" {
			t.Error("Node-1 should have been removed")
		}
	}
}

// Benchmark tests for distributed storage
func BenchmarkDistributedStorageService_WriteShard(b *testing.B) {
	// Create base storage manager
	baseConfig := StorageManagerConfig{
		BasePath:    "/tmp/bench-storage",
		Compression: false,
		Encryption:  false,
		Dedup:       false,
	}
	baseManager, _ := NewStorageManager(baseConfig)

	// Create distributed storage config
	distConfig := DefaultDistributedStorageConfig()
	distConfig.RootDir = "/tmp/bench-distributed"

	// Create distributed storage service
	distService, _ := NewDistributedStorageService(baseManager, distConfig)
	distService.Start()
	defer distService.Stop()

	// Add nodes
	for i := 0; i < 2; i++ {
		node := NodeInfo{
			ID:        fmt.Sprintf("node-%d", i),
			Available: true,
			JoinedAt:  time.Now(),
			LastSeen:  time.Now(),
		}
		distService.AddNode(node)
	}

	// Create volume
	opts := VolumeCreateOptions{Name: "bench-volume", Type: VolumeTypeDistributed, Size: 100}
	ctx := context.Background()
	volume, _ := distService.CreateDistributedVolume(ctx, opts, 1)

	// Test data
	testData := make([]byte, 1024) // 1KB
	for i := range testData {
		testData[i] = byte(i % 256)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		err := distService.WriteShard(ctx, volume.ID, 0, testData)
		if err != nil {
			b.Fatalf("Failed to write shard: %v", err)
		}
	}
}

func BenchmarkDistributedStorageService_ReadShard(b *testing.B) {
	// Create base storage manager
	baseConfig := StorageManagerConfig{
		BasePath:    "/tmp/bench-storage",
		Compression: false,
		Encryption:  false,
		Dedup:       false,
	}
	baseManager, _ := NewStorageManager(baseConfig)

	// Create distributed storage config
	distConfig := DefaultDistributedStorageConfig()
	distConfig.RootDir = "/tmp/bench-distributed"

	// Create distributed storage service
	distService, _ := NewDistributedStorageService(baseManager, distConfig)
	distService.Start()
	defer distService.Stop()

	// Add nodes
	for i := 0; i < 2; i++ {
		node := NodeInfo{
			ID:        fmt.Sprintf("node-%d", i),
			Available: true,
			JoinedAt:  time.Now(),
			LastSeen:  time.Now(),
		}
		distService.AddNode(node)
	}

	// Create volume and write test data
	opts := VolumeCreateOptions{Name: "bench-volume", Type: VolumeTypeDistributed, Size: 100}
	ctx := context.Background()
	volume, _ := distService.CreateDistributedVolume(ctx, opts, 1)

	testData := make([]byte, 1024)
	for i := range testData {
		testData[i] = byte(i % 256)
	}
	distService.WriteShard(ctx, volume.ID, 0, testData)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := distService.ReadShard(ctx, volume.ID, 0)
		if err != nil {
			b.Fatalf("Failed to read shard: %v", err)
		}
	}
}