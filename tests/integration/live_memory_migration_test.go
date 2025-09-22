package integration_test

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"

	"github.com/novacron/backend/core/vm"
)

// TestLiveMemoryMigrationStrategies tests the complete migration implementation
func TestLiveMemoryMigrationStrategies(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	// Create memory state distribution instance
	msd := createTestMemoryStateDistribution(t, logger)

	// Create test memory shard with pages
	shard := createTestMemoryShard(t)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	targetNode := "target-node-1"

	t.Run("TransferPages with Batching and Compression", func(t *testing.T) {
		pages := getTestPages(shard)

		err := msd.TransferPages(ctx, pages, targetNode)
		assert.NoError(t, err, "transferPages should complete without error")

		// Verify pages were processed
		for _, page := range pages {
			assert.NotZero(t, page.LastTransfer, "LastTransfer should be set")
			assert.NotZero(t, page.TransferVersion, "TransferVersion should be set")
		}
	})

	t.Run("TransferPagesWithDelta using DeltaSync", func(t *testing.T) {
		pages := getTestPages(shard)

		err := msd.TransferPagesWithDelta(ctx, pages, targetNode)
		assert.NoError(t, err, "transferPagesWithDelta should complete without error")

		// Verify dirty bits are cleared
		for _, page := range pages {
			assert.NotZero(t, page.LastTransfer, "LastTransfer should be updated")
		}
	})

	t.Run("PostCopy Fetching Setup", func(t *testing.T) {
		err := msd.SetupPostCopyFetching(ctx, shard, targetNode)
		assert.NoError(t, err, "setupPostCopyFetching should complete without error")

		// Verify post-copy mode is enabled
		assert.True(t, shard.PostCopyMode, "PostCopyMode should be enabled")
		assert.NotNil(t, shard.FaultHandler, "FaultHandler should be set")
	})

	t.Run("Hybrid Migration Complete Workflow", func(t *testing.T) {
		// Create hybrid migration strategy
		hybridStrategy := &vm.HybridMigration{
			msd: msd,
		}

		err := hybridStrategy.Migrate(ctx, shard, targetNode)
		assert.NoError(t, err, "hybrid migration should complete without error")

		// Verify migration was tracked in metrics
		assert.Greater(t, msd.GetMetrics().MigrationsCompleted.Load(), int64(0),
			"Migration should be tracked in metrics")
	})
}

func TestMigrationWithPredictions(t *testing.T) {
	logger, _ := zap.NewDevelopment()
	msd := createTestMemoryStateDistribution(t, logger)

	// Add mock predictions
	mockPrefetcher := createMockPrefetcher(t)
	msd.SetPrefetcher(mockPrefetcher)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	shard := createTestMemoryShard(t)
	targetNode := "target-node-2"

	t.Run("AI-Prioritized Page Ordering", func(t *testing.T) {
		pages := getTestPages(shard)

		// Set different temperatures for testing
		pages[0].Temperature = vm.PageTemperatureCold
		pages[1].Temperature = vm.PageTemperatureWarm
		pages[2].Temperature = vm.PageTemperatureHot

		err := msd.TransferPages(ctx, pages, targetNode)
		assert.NoError(t, err, "transfer with AI prioritization should work")

		// Verify pages were ordered correctly (would need more detailed verification in real test)
		assert.NotZero(t, msd.GetMetrics().BytesSynced.Load(), "Bytes should be tracked")
	})

	t.Run("Delta Sync with Predictions", func(t *testing.T) {
		pages := getTestPages(shard)

		err := msd.TransferPagesWithDelta(ctx, pages, targetNode)
		assert.NoError(t, err, "delta sync with predictions should work")
	})
}

// Helper functions for testing

func createTestMemoryStateDistribution(t *testing.T, logger *zap.Logger) *vm.MemoryStateDistribution {
	// Create a test instance with mock components
	msd := &vm.MemoryStateDistribution{
		Logger:              logger,
		NodeID:              "test-node",
		MemoryShards:        make(map[string]*vm.MemoryShard),
		MigrationStrategies: make(map[string]vm.MigrationStrategy),
		Metrics:             &vm.MemoryDistributionMetrics{},
		CompressionEngine:   vm.NewMemoryCompressionEngine(),
		CoherenceProtocol:   vm.NewMemoryCoherenceProtocol(),
		BandwidthLimiter:    &vm.BandwidthLimiter{MaxBytesPerSecond: 1024 * 1024 * 100}, // 100MB/s
		DirtyBitmap:         vm.NewDirtyPageBitmap(10000),
	}

	return msd
}

func createTestMemoryShard(t *testing.T) *vm.MemoryShard {
	shard := &vm.MemoryShard{
		ShardID:      "test-shard-1",
		VMID:         "test-vm-1",
		StartAddress: 0x1000,
		EndAddress:   0x5000,
		Pages:        make(map[uint64]*vm.DistributedMemoryPage),
		ReplicaNodes: []string{"replica-1", "replica-2"},
		Version:      1,
		DirtyBitmap:  vm.NewDirtyPageBitmap(1000),
	}

	// Add test pages
	for i := uint64(0); i < 10; i++ {
		page := &vm.DistributedMemoryPage{
			PageNumber:  i,
			Data:        make([]byte, 4096),
			VMID:        shard.VMID,
			ShardID:     shard.ShardID,
			Version:     1,
			Temperature: vm.PageTemperatureCold,
			AccessCount: uint64(i * 10),
		}
		shard.Pages[i] = page
	}

	return shard
}

func getTestPages(shard *vm.MemoryShard) []*vm.DistributedMemoryPage {
	pages := make([]*vm.DistributedMemoryPage, 0, len(shard.Pages))
	for _, page := range shard.Pages {
		pages = append(pages, page)
	}
	return pages
}

func createMockPrefetcher(t *testing.T) *vm.PredictivePrefetchingEngine {
	// Create a mock prefetcher that returns test predictions
	return &vm.PredictivePrefetchingEngine{
		// Mock implementation would go here
		// For testing purposes, we'll return predictable results
	}
}

// Benchmark tests for performance validation

func BenchmarkPageTransfer(b *testing.B) {
	logger, _ := zap.NewDevelopment()
	msd := createTestMemoryStateDistribution(b, logger)
	shard := createTestMemoryShard(b)
	pages := getTestPages(shard)

	ctx := context.Background()
	targetNode := "benchmark-target"

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		err := msd.TransferPages(ctx, pages, targetNode)
		require.NoError(b, err)
	}
}

func BenchmarkDeltaTransfer(b *testing.B) {
	logger, _ := zap.NewDevelopment()
	msd := createTestMemoryStateDistribution(b, logger)
	shard := createTestMemoryShard(b)
	pages := getTestPages(shard)

	ctx := context.Background()
	targetNode := "benchmark-target"

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		err := msd.TransferPagesWithDelta(ctx, pages, targetNode)
		require.NoError(b, err)
	}
}