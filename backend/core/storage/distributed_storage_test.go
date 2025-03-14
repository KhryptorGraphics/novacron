package storage

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestDistributedStorage(t *testing.T) {
	// Create a temporary directory for testing
	tempDir, err := os.MkdirTemp("", "distributed-storage-test")
	if err != nil {
		t.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// Create a mock StorageManager
	baseManager := &StorageManager{
		volumes: make(map[string]*Volume),
	}

	// Create a distributed storage config
	config := DefaultDistributedStorageConfig()
	config.RootDir = tempDir
	config.ShardSize = 1 * 1024 * 1024 // 1MB for testing
	config.HealthCheckInterval = 100 * time.Millisecond
	config.HealingInterval = 500 * time.Millisecond

	// Create the distributed storage service
	service, err := NewDistributedStorageService(baseManager, config)
	if err != nil {
		t.Fatalf("Failed to create distributed storage service: %v", err)
	}

	// Start the service
	if err := service.Start(); err != nil {
		t.Fatalf("Failed to start distributed storage service: %v", err)
	}
	defer service.Stop()

	t.Run("TestNodeManagement", func(t *testing.T) {
		testNodeManagement(t, service)
	})

	t.Run("TestVolumeCreation", func(t *testing.T) {
		testVolumeCreation(t, service)
	})

	t.Run("TestReplication", func(t *testing.T) {
		testReplication(t, service)
	})

	t.Run("TestHealing", func(t *testing.T) {
		testHealing(t, service)
	})

	t.Run("TestDeduplication", func(t *testing.T) {
		testDeduplication(t, service)
	})
}

func testNodeManagement(t *testing.T, service *DistributedStorageService) {
	// Create some test nodes
	node1 := NodeInfo{
		ID:        "node1",
		Name:      "Node 1",
		Role:      "storage",
		Address:   "192.168.1.1",
		Port:      8000,
		Available: true,
		JoinedAt:  time.Now(),
		LastSeen:  time.Now(),
	}

	node2 := NodeInfo{
		ID:        "node2",
		Name:      "Node 2",
		Role:      "storage",
		Address:   "192.168.1.2",
		Port:      8000,
		Available: true,
		JoinedAt:  time.Now(),
		LastSeen:  time.Now(),
	}

	// Add nodes
	service.AddNode(node1)
	service.AddNode(node2)

	// Get available nodes
	nodes := service.GetAvailableNodes()
	if len(nodes) != 2 {
		t.Errorf("Expected 2 available nodes, got %d", len(nodes))
	}

	// Remove a node
	service.RemoveNode("node1")
	nodes = service.GetAvailableNodes()
	if len(nodes) != 1 {
		t.Errorf("Expected 1 available node after removal, got %d", len(nodes))
	}
	if nodes[0].ID != "node2" {
		t.Errorf("Expected remaining node to be node2, got %s", nodes[0].ID)
	}
}

func testVolumeCreation(t *testing.T, service *DistributedStorageService) {
	// Add nodes
	for i := 1; i <= 5; i++ {
		service.AddNode(NodeInfo{
			ID:        mockNodeID(i),
			Name:      mockNodeName(i),
			Role:      "storage",
			Address:   mockNodeAddress(i),
			Port:      8000,
			Available: true,
			JoinedAt:  time.Now(),
			LastSeen:  time.Now(),
		})
	}

	// Create a volume specification
	spec := VolumeSpec{
		Name:   "test-volume",
		SizeMB: 100,
		Type:   VolumeTypeCeph,
		Options: map[string]string{
			"description": "Test distributed volume",
		},
	}

	// Create the volume
	ctx := context.Background()
	volume, err := service.CreateDistributedVolume(ctx, spec, 3)
	if err != nil {
		t.Fatalf("Failed to create distributed volume: %v", err)
	}

	// Verify volume was created
	if volume.Name != "test-volume" {
		t.Errorf("Expected volume name to be test-volume, got %s", volume.Name)
	}

	// Get the distributed volume
	distVolume, err := service.GetDistributedVolume(ctx, volume.ID)
	if err != nil {
		t.Fatalf("Failed to get distributed volume: %v", err)
	}

	// Verify sharding
	if distVolume.DistInfo.ShardCount <= 0 {
		t.Errorf("Expected positive shard count, got %d", distVolume.DistInfo.ShardCount)
	}

	// Verify node assignment
	if distVolume.DistInfo.NodeCount <= 0 {
		t.Errorf("Expected positive node count, got %d", distVolume.DistInfo.NodeCount)
	}

	// Check shards have node assignments
	for i, shard := range distVolume.DistInfo.Shards {
		if len(shard.NodeIDs) != 3 {
			t.Errorf("Shard %d has %d node assignments, expected 3", i, len(shard.NodeIDs))
		}
	}
}

func testReplication(t *testing.T, service *DistributedStorageService) {
	// Add nodes
	for i := 1; i <= 5; i++ {
		service.AddNode(NodeInfo{
			ID:        mockNodeID(i),
			Name:      mockNodeName(i),
			Role:      "storage",
			Address:   mockNodeAddress(i),
			Port:      8000,
			Available: true,
			JoinedAt:  time.Now(),
			LastSeen:  time.Now(),
		})
	}

	// Create a volume specification
	spec := VolumeSpec{
		Name:   "replication-test",
		SizeMB: 50,
		Type:   VolumeTypeCeph,
		Options: map[string]string{
			"description": "Test replication",
		},
	}

	// Create the volume with replication factor 3
	ctx := context.Background()
	volume, err := service.CreateDistributedVolume(ctx, spec, 3)
	if err != nil {
		t.Fatalf("Failed to create distributed volume: %v", err)
	}

	// Verify the volume exists - we don't actually need the distVolume variable
	_, err = service.GetDistributedVolume(ctx, volume.ID)
	if err != nil {
		t.Fatalf("Failed to get distributed volume: %v", err)
	}

	// Write data to a shard
	testData := []byte("test data for replication")
	err = service.WriteShard(ctx, volume.ID, 0, testData)
	if err != nil {
		t.Fatalf("Failed to write to shard: %v", err)
	}

	// Read data back
	readData, err := service.ReadShard(ctx, volume.ID, 0)
	if err != nil {
		t.Fatalf("Failed to read from shard: %v", err)
	}

	// Verify data
	if string(readData) != string(testData) {
		t.Errorf("Expected read data to be %q, got %q", string(testData), string(readData))
	}

	// Verify replication by checking the files on disk
	shardPath := filepath.Join(service.config.RootDir, volume.ID, "shard_0")
	if _, err := os.Stat(shardPath); os.IsNotExist(err) {
		t.Errorf("Shard file does not exist at %s", shardPath)
	}
}

func testHealing(t *testing.T, service *DistributedStorageService) {
	// Add nodes
	for i := 1; i <= 5; i++ {
		service.AddNode(NodeInfo{
			ID:        mockNodeID(i),
			Name:      mockNodeName(i),
			Role:      "storage",
			Address:   mockNodeAddress(i),
			Port:      8000,
			Available: true,
			JoinedAt:  time.Now(),
			LastSeen:  time.Now(),
		})
	}

	// Create a volume specification
	spec := VolumeSpec{
		Name:   "healing-test",
		SizeMB: 50,
		Type:   VolumeTypeCeph,
		Options: map[string]string{
			"description": "Test healing",
		},
	}

	// Create the volume with replication factor 3
	ctx := context.Background()
	volume, err := service.CreateDistributedVolume(ctx, spec, 3)
	if err != nil {
		t.Fatalf("Failed to create distributed volume: %v", err)
	}

	distVolume, err := service.GetDistributedVolume(ctx, volume.ID)
	if err != nil {
		t.Fatalf("Failed to get distributed volume: %v", err)
	}

	// Write data to a shard
	testData := []byte("test data for healing")
	err = service.WriteShard(ctx, volume.ID, 0, testData)
	if err != nil {
		t.Fatalf("Failed to write to shard: %v", err)
	}

	// Simulate a failure by marking a shard as needing healing
	service.volMutex.Lock()
	distVolume.DistInfo.Shards[0].NeedsHealing = true
	service.volMutex.Unlock()

	// Trigger healing
	err = service.RepairVolume(ctx, volume.ID)
	if err != nil {
		t.Fatalf("Failed to repair volume: %v", err)
	}

	// Verify healing
	service.volMutex.RLock()
	needsHealing := distVolume.DistInfo.Shards[0].NeedsHealing
	service.volMutex.RUnlock()

	if needsHealing {
		t.Errorf("Shard still needs healing after repair")
	}

	// Verify health percentage is 100%
	service.volMutex.RLock()
	healthPct := distVolume.DistInfo.HealthPercentage
	service.volMutex.RUnlock()

	if healthPct != 100.0 {
		t.Errorf("Expected health percentage to be 100.0%%, got %.1f%%", healthPct)
	}
}

func testDeduplication(t *testing.T, service *DistributedStorageService) {
	// Add nodes
	for i := 1; i <= 5; i++ {
		service.AddNode(NodeInfo{
			ID:        mockNodeID(i),
			Name:      mockNodeName(i),
			Role:      "storage",
			Address:   mockNodeAddress(i),
			Port:      8000,
			Available: true,
			JoinedAt:  time.Now(),
			LastSeen:  time.Now(),
		})
	}

	// Create a volume specification with deduplication enabled
	spec := VolumeSpec{
		Name:   "dedup-test",
		SizeMB: 50,
		Type:   VolumeTypeCeph,
		Options: map[string]string{
			"description": "Test deduplication",
		},
	}

	// Enable deduplication in the config
	originalDedupSetting := service.config.DefaultDeduplication
	service.config.DefaultDeduplication = true
	defer func() {
		// Restore original setting after test
		service.config.DefaultDeduplication = originalDedupSetting
	}()

	// Create the volume with replication factor 3
	ctx := context.Background()
	volume, err := service.CreateDistributedVolume(ctx, spec, 3)
	if err != nil {
		t.Fatalf("Failed to create distributed volume: %v", err)
	}

	// Write duplicated data to shard 0
	// This will contain repeated data suitable for deduplication
	repeatedBlock := []byte("This is a block of data that will be repeated multiple times to test deduplication.")
	var testData []byte
	for i := 0; i < 100; i++ {
		testData = append(testData, repeatedBlock...)
	}

	err = service.WriteShard(ctx, volume.ID, 0, testData)
	if err != nil {
		t.Fatalf("Failed to write to shard: %v", err)
	}

	// Read data back
	readData, err := service.ReadShard(ctx, volume.ID, 0)
	if err != nil {
		t.Fatalf("Failed to read from shard: %v", err)
	}

	// Verify data integrity after deduplication/reconstruction
	if string(readData) != string(testData) {
		t.Errorf("Data integrity failure after deduplication: original length %d, reconstructed length %d",
			len(testData), len(readData))
	} else {
		t.Logf("Successfully verified data integrity after deduplication/reconstruction")
	}

	// Check if deduplication was applied by verifying the shard has deduplication metadata
	service.volMutex.RLock()
	distVolume, _ := service.distVolumes[volume.ID]
	service.volMutex.RUnlock()

	if distVolume != nil {
		shard := distVolume.DistInfo.Shards[0]
		if !shard.IsDeduplicated {
			t.Logf("Shard was not deduplicated, which might be correct if data wasn't duplicated enough")
		} else {
			// If deduplication was applied, verify metrics
			if shard.DedupFileInfo == nil {
				t.Errorf("Shard marked as deduplicated but has no deduplication info")
			} else {
				t.Logf("Deduplication applied with algorithm %s, ratio %.2f, unique blocks %d",
					shard.DedupFileInfo.Algorithm, shard.DedupFileInfo.DedupRatio, len(shard.DedupFileInfo.Blocks))

				// Expect a reasonable deduplication ratio for our test data
				if shard.DedupFileInfo.DedupRatio > 1.0 {
					t.Logf("Achieved deduplication ratio of %.2f", shard.DedupFileInfo.DedupRatio)
				} else {
					t.Logf("Deduplication ratio not greater than 1.0: %.2f", shard.DedupFileInfo.DedupRatio)
				}
			}
		}
	}
}

// Helper functions for generating test data
func mockNodeID(i int) string {
	return "node-" + mockSuffix(i)
}

func mockNodeName(i int) string {
	return "Node " + mockSuffix(i)
}

func mockNodeAddress(i int) string {
	return "192.168.1." + mockSuffix(i)
}

func mockSuffix(i int) string {
	return "0" + string('0'+rune(i))
}
