package integration

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"novacron/backend/core/vm"
)

// TestDistributedStateIntegration tests the complete distributed state management system
func TestDistributedStateIntegration(t *testing.T) {
	ctx := context.Background()

	// Test VM state sharding
	t.Run("VMStateSharding", func(t *testing.T) {
		testVMStateSharding(t, ctx)
	})

	// Test memory state distribution
	t.Run("MemoryStateDistribution", func(t *testing.T) {
		testMemoryStateDistribution(t, ctx)
	})

	// Test distributed state coordinator
	t.Run("DistributedStateCoordinator", func(t *testing.T) {
		testDistributedStateCoordinator(t, ctx)
	})

	// Test predictive prefetching integration
	t.Run("PredictivePrefetchingIntegration", func(t *testing.T) {
		testPredictivePrefetchingIntegration(t, ctx)
	})

	// Test end-to-end VM migration
	t.Run("EndToEndVMMigration", func(t *testing.T) {
		testEndToEndVMMigration(t, ctx)
	})

	// Test state consistency across nodes
	t.Run("StateConsistencyAcrossNodes", func(t *testing.T) {
		testStateConsistencyAcrossNodes(t, ctx)
	})

	// Test conflict resolution
	t.Run("ConflictResolution", func(t *testing.T) {
		testConflictResolution(t, ctx)
	})

	// Test performance metrics
	t.Run("PerformanceMetrics", func(t *testing.T) {
		testPerformanceMetrics(t, ctx)
	})
}

func testVMStateSharding(t *testing.T, ctx context.Context) {
	// Create VM state sharding manager
	shardingManager := vm.NewVMStateShardingManager([]string{"node1", "node2", "node3"})
	require.NotNil(t, shardingManager)

	// Test VM shard allocation
	vmID := "test-vm-1"
	shardID, err := shardingManager.AllocateShards(ctx, vmID, 2) // 2 replicas
	require.NoError(t, err)
	assert.NotEmpty(t, shardID)

	// Test state storage and retrieval
	testState := &vm.VMState{
		ID:       vmID,
		State:    vm.StateRunning,
		NodeID:   "node1",
		UpdatedAt: time.Now(),
	}

	err = shardingManager.UpdateVMState(ctx, vmID, testState)
	require.NoError(t, err)

	retrievedState, err := shardingManager.GetVMState(ctx, vmID)
	require.NoError(t, err)
	assert.Equal(t, testState.ID, retrievedState.ID)
	assert.Equal(t, testState.State, retrievedState.State)

	// Test shard rebalancing
	newNodes := []string{"node1", "node2", "node3", "node4"}
	err = shardingManager.RebalanceShards(ctx, newNodes)
	require.NoError(t, err)

	// Verify VM state is still accessible after rebalancing
	retrievedState, err = shardingManager.GetVMState(ctx, vmID)
	require.NoError(t, err)
	assert.Equal(t, testState.ID, retrievedState.ID)

	// Test failover scenario
	err = shardingManager.HandleNodeFailure(ctx, "node1")
	require.NoError(t, err)

	// Verify state is still accessible after node failure
	retrievedState, err = shardingManager.GetVMState(ctx, vmID)
	require.NoError(t, err)
	assert.Equal(t, testState.ID, retrievedState.ID)
}

func testMemoryStateDistribution(t *testing.T, ctx context.Context) {
	// Create memory state distribution manager
	memDistManager := vm.NewMemoryStateDistribution([]string{"node1", "node2", "node3"})
	require.NotNil(t, memDistManager)

	vmID := "test-vm-memory"

	// Test memory state distribution
	memoryState := &vm.MemoryState{
		VMID:         vmID,
		TotalPages:   1000,
		PageSize:     4096,
		DirtyPages:   []string{"page_1", "page_2", "page_3"},
		LastUpdated:  time.Now(),
	}

	err := memDistManager.DistributeMemoryState(ctx, vmID, memoryState)
	require.NoError(t, err)

	// Test memory delta synchronization
	deltaState := &vm.MemoryDelta{
		VMID:           vmID,
		ModifiedPages:  []string{"page_1", "page_4"},
		DeltaTimestamp: time.Now(),
		CompressionRatio: 0.7,
	}

	err = memDistManager.SyncMemoryDelta(ctx, vmID, deltaState)
	require.NoError(t, err)

	// Test live memory migration
	migrationRequest := &vm.LiveMigrationRequest{
		VMID:         vmID,
		SourceNode:   "node1",
		TargetNode:   "node2",
		Strategy:     vm.MigrationStrategyHybrid,
		MaxDowntime:  time.Millisecond * 100,
	}

	migrationResult, err := memDistManager.MigrateLiveMemory(ctx, migrationRequest)
	require.NoError(t, err)
	assert.NotNil(t, migrationResult)
	assert.True(t, migrationResult.Success)
	assert.LessOrEqual(t, migrationResult.DowntimeMs, float64(100))

	// Test memory compression and deduplication
	compressionStats, err := memDistManager.GetCompressionStats(ctx, vmID)
	require.NoError(t, err)
	assert.NotNil(t, compressionStats)
	assert.Greater(t, compressionStats.CompressionRatio, 0.0)
}

func testDistributedStateCoordinator(t *testing.T, ctx context.Context) {
	// Create distributed state coordinator
	coordinator := vm.NewDistributedStateCoordinator([]string{"node1", "node2", "node3"})
	require.NotNil(t, coordinator)

	vmID := "test-vm-coordinator"

	// Test VM state migration coordination
	migrationRequest := &vm.StateMigrationRequest{
		VMID:         vmID,
		SourceNode:   "node1",
		TargetNode:   "node2",
		IncludeMemory: true,
		IncludeState:  true,
		Priority:     vm.MigrationPriorityHigh,
	}

	migrationResult, err := coordinator.MigrateVMState(ctx, migrationRequest)
	require.NoError(t, err)
	assert.NotNil(t, migrationResult)
	assert.True(t, migrationResult.Success)

	// Test conflict resolution
	conflict := &vm.StateConflict{
		ConflictID:    "conflict-1",
		ConflictType:  "version_mismatch",
		SourceNode:    "node1",
		TargetNode:    "node2",
		ConflictField: "vm_state",
		SourceValue:   vm.StateRunning,
		TargetValue:   vm.StatePaused,
		DetectedAt:    time.Now(),
		Severity:      vm.ConflictSeverityMedium,
	}

	resolution, err := coordinator.ResolveConflict(ctx, conflict)
	require.NoError(t, err)
	assert.NotNil(t, resolution)
	assert.NotEmpty(t, resolution.Resolution)

	// Test global state optimization
	optimizationRequest := &vm.OptimizationRequest{
		Scope:      vm.OptimizationScopeCluster,
		Objectives: []vm.OptimizationObjective{
			vm.OptimizeLatency,
			vm.OptimizeBandwidth,
			vm.OptimizeMemoryUsage,
		},
		MaxDuration: time.Minute * 10,
	}

	optimizationResult, err := coordinator.OptimizeStatePlacement(ctx, optimizationRequest)
	require.NoError(t, err)
	assert.NotNil(t, optimizationResult)
	assert.Greater(t, optimizationResult.ImprovementScore, 0.0)

	// Test distributed transaction management
	transaction := &vm.DistributedTransaction{
		TransactionID: "tx-1",
		Operations: []vm.TransactionOperation{
			{Type: vm.OpUpdateState, VMID: vmID, Data: map[string]interface{}{"state": vm.StateRunning}},
			{Type: vm.OpUpdateMemory, VMID: vmID, Data: map[string]interface{}{"memory_mb": 2048}},
		},
		IsolationLevel: vm.IsolationSerializable,
		Timeout:        time.Second * 30,
	}

	txResult, err := coordinator.ExecuteTransaction(ctx, transaction)
	require.NoError(t, err)
	assert.NotNil(t, txResult)
	assert.True(t, txResult.Committed)
}

func testPredictivePrefetchingIntegration(t *testing.T, ctx context.Context) {
	// Create predictive prefetching with enhanced integration
	prefetcher := vm.NewPredictivePrefetching(vm.DefaultPredictivePrefetchingConfig())
	require.NotNil(t, prefetcher)

	vmID := "test-vm-prefetch"

	// Test migration integration
	migrationData := &vm.MigrationData{
		VMID:           vmID,
		SourceNode:     "node1",
		TargetNode:     "node2",
		MemorySize:     2048 * 1024 * 1024, // 2GB
		ExpectedPages:  []string{"page_1", "page_2", "page_3", "page_4", "page_5"},
		AccessPatterns: map[string]float64{
			"page_1": 0.9, "page_2": 0.7, "page_3": 0.5, "page_4": 0.3, "page_5": 0.1,
		},
	}

	predictions, err := prefetcher.PredictMigrationPages(ctx, migrationData)
	require.NoError(t, err)
	assert.NotEmpty(t, predictions)
	assert.Greater(t, len(predictions), 0)

	// Verify predictions are ordered by confidence
	for i := 1; i < len(predictions); i++ {
		assert.GreaterOrEqual(t, predictions[i-1].Confidence, predictions[i].Confidence)
	}

	// Test cross-node coordination
	coordinationRequest := &vm.CrossNodeRequest{
		SourceNode:    "node1",
		TargetNodes:   []string{"node2", "node3"},
		VMID:          vmID,
		CoordinationType: vm.CoordinationPrefetch,
		Priority:      vm.PriorityHigh,
	}

	coordinationResult, err := prefetcher.CoordinateAcrossNodes(ctx, coordinationRequest)
	require.NoError(t, err)
	assert.NotNil(t, coordinationResult)
	assert.True(t, coordinationResult.Success)

	// Test memory state prefetching
	prefetchRequest := &vm.MemoryPrefetchRequest{
		VMID:         vmID,
		TargetNode:   "node2",
		PredictedPages: predictions,
		PrefetchSize: 1024 * 1024, // 1MB
		Deadline:     time.Now().Add(time.Second * 30),
	}

	prefetchResult, err := prefetcher.PrefetchMemoryState(ctx, prefetchRequest)
	require.NoError(t, err)
	assert.NotNil(t, prefetchResult)
	assert.Greater(t, prefetchResult.PrefetchedBytes, int64(0))

	// Test real-time prediction API
	predictionRequest := &vm.RealTimePredictionRequest{
		VMID:        vmID,
		CurrentPage: "page_current",
		Context:     "migration_active",
		WindowSize:  10,
	}

	realtimePrediction, err := prefetcher.GetRealTimePrediction(ctx, predictionRequest)
	require.NoError(t, err)
	assert.NotNil(t, realtimePrediction)
	assert.Greater(t, realtimePrediction.Confidence, 0.0)
	assert.LessOrEqual(t, realtimePrediction.LatencyMs, float64(vm.TARGET_PREDICTION_LATENCY_MS))
}

func testEndToEndVMMigration(t *testing.T, ctx context.Context) {
	// Test complete end-to-end VM migration using all distributed components
	sourceNode := "node1"
	targetNode := "node2"
	vmID := "test-vm-e2e"

	// Initialize components
	shardingManager := vm.NewVMStateShardingManager([]string{sourceNode, targetNode, "node3"})
	memDistManager := vm.NewMemoryStateDistribution([]string{sourceNode, targetNode, "node3"})
	coordinator := vm.NewDistributedStateCoordinator([]string{sourceNode, targetNode, "node3"})
	prefetcher := vm.NewPredictivePrefetching(vm.DefaultPredictivePrefetchingConfig())

	// Step 1: Prepare VM state on source node
	vmState := &vm.VMState{
		ID:       vmID,
		State:    vm.StateRunning,
		NodeID:   sourceNode,
		MemoryMB: 2048,
		UpdatedAt: time.Now(),
	}

	err := shardingManager.UpdateVMState(ctx, vmID, vmState)
	require.NoError(t, err)

	// Step 2: Predict and prefetch memory pages
	migrationData := &vm.MigrationData{
		VMID:       vmID,
		SourceNode: sourceNode,
		TargetNode: targetNode,
		MemorySize: 2048 * 1024 * 1024,
		ExpectedPages: []string{"page_1", "page_2", "page_3", "page_4", "page_5"},
		AccessPatterns: map[string]float64{
			"page_1": 0.9, "page_2": 0.7, "page_3": 0.5, "page_4": 0.3, "page_5": 0.1,
		},
	}

	predictions, err := prefetcher.PredictMigrationPages(ctx, migrationData)
	require.NoError(t, err)

	prefetchRequest := &vm.MemoryPrefetchRequest{
		VMID:         vmID,
		TargetNode:   targetNode,
		PredictedPages: predictions,
		PrefetchSize: 1024 * 1024,
		Deadline:     time.Now().Add(time.Second * 30),
	}

	prefetchResult, err := prefetcher.PrefetchMemoryState(ctx, prefetchRequest)
	require.NoError(t, err)
	assert.Greater(t, prefetchResult.PrefetchedBytes, int64(0))

	// Step 3: Execute live memory migration
	liveMigrationRequest := &vm.LiveMigrationRequest{
		VMID:       vmID,
		SourceNode: sourceNode,
		TargetNode: targetNode,
		Strategy:   vm.MigrationStrategyHybrid,
		MaxDowntime: time.Millisecond * 100,
	}

	migrationResult, err := memDistManager.MigrateLiveMemory(ctx, liveMigrationRequest)
	require.NoError(t, err)
	assert.True(t, migrationResult.Success)

	// Step 4: Coordinate state migration
	stateMigrationRequest := &vm.StateMigrationRequest{
		VMID:         vmID,
		SourceNode:   sourceNode,
		TargetNode:   targetNode,
		IncludeMemory: true,
		IncludeState:  true,
		Priority:     vm.MigrationPriorityHigh,
	}

	stateMigrationResult, err := coordinator.MigrateVMState(ctx, stateMigrationRequest)
	require.NoError(t, err)
	assert.True(t, stateMigrationResult.Success)

	// Step 5: Verify VM state on target node
	retrievedState, err := shardingManager.GetVMState(ctx, vmID)
	require.NoError(t, err)
	assert.Equal(t, vmID, retrievedState.ID)
	assert.Equal(t, targetNode, retrievedState.NodeID)

	// Step 6: Verify total migration time is within acceptable limits
	totalMigrationTime := stateMigrationResult.CompletedAt.Sub(stateMigrationResult.StartedAt)
	assert.Less(t, totalMigrationTime, time.Second*10) // Should complete within 10 seconds
}

func testStateConsistencyAcrossNodes(t *testing.T, ctx context.Context) {
	nodes := []string{"node1", "node2", "node3"}
	vmID := "test-vm-consistency"

	// Initialize state management across nodes
	shardingManager := vm.NewVMStateShardingManager(nodes)
	coordinator := vm.NewDistributedStateCoordinator(nodes)

	// Create initial VM state
	initialState := &vm.VMState{
		ID:       vmID,
		State:    vm.StateRunning,
		NodeID:   "node1",
		MemoryMB: 1024,
		UpdatedAt: time.Now(),
	}

	err := shardingManager.UpdateVMState(ctx, vmID, initialState)
	require.NoError(t, err)

	// Simulate concurrent updates from multiple nodes
	updateChan := make(chan error, len(nodes))

	for i, node := range nodes {
		go func(nodeID string, iteration int) {
			updatedState := &vm.VMState{
				ID:       vmID,
				State:    vm.StateRunning,
				NodeID:   nodeID,
				MemoryMB: 1024 + iteration*256, // Different memory values
				UpdatedAt: time.Now(),
			}
			updateChan <- shardingManager.UpdateVMState(ctx, vmID, updatedState)
		}(node, i)
	}

	// Wait for all updates to complete
	for i := 0; i < len(nodes); i++ {
		err := <-updateChan
		require.NoError(t, err)
	}

	// Verify eventual consistency
	time.Sleep(time.Second * 2) // Allow time for consistency protocols

	finalState, err := shardingManager.GetVMState(ctx, vmID)
	require.NoError(t, err)
	assert.Equal(t, vmID, finalState.ID)
	assert.Equal(t, vm.StateRunning, finalState.State)

	// Test consistency validation
	consistencyReport, err := coordinator.ValidateConsistency(ctx, vmID)
	require.NoError(t, err)
	assert.NotNil(t, consistencyReport)
	assert.True(t, consistencyReport.IsConsistent)
}

func testConflictResolution(t *testing.T, ctx context.Context) {
	coordinator := vm.NewDistributedStateCoordinator([]string{"node1", "node2", "node3"})

	// Create a state conflict scenario
	conflict := &vm.StateConflict{
		ConflictID:    "conflict-test-1",
		ConflictType:  "concurrent_update",
		SourceNode:    "node1",
		TargetNode:    "node2",
		ConflictField: "memory_mb",
		SourceValue:   1024,
		TargetValue:   2048,
		DetectedAt:    time.Now(),
		Severity:      vm.ConflictSeverityHigh,
		Metadata: map[string]interface{}{
			"vm_id": "test-vm-conflict",
			"operation": "update_memory",
		},
	}

	// Test automatic conflict resolution
	resolution, err := coordinator.ResolveConflict(ctx, conflict)
	require.NoError(t, err)
	assert.NotNil(t, resolution)
	assert.NotEmpty(t, resolution.Resolution)
	assert.NotNil(t, resolution.ResolvedValue)

	// Test conflict resolution with voting
	conflict.Severity = vm.ConflictSeverityCritical
	resolution, err = coordinator.ResolveConflict(ctx, conflict)
	require.NoError(t, err)
	assert.NotNil(t, resolution)

	// Test conflict resolution timeout
	conflict.ConflictID = "conflict-timeout-test"
	ctx, cancel := context.WithTimeout(ctx, time.Millisecond*100)
	defer cancel()

	resolution, err = coordinator.ResolveConflict(ctx, conflict)
	// Should either succeed quickly or timeout gracefully
	if err != nil {
		assert.Contains(t, err.Error(), "timeout")
	} else {
		assert.NotNil(t, resolution)
	}
}

func testPerformanceMetrics(t *testing.T, ctx context.Context) {
	vmID := "test-vm-metrics"
	nodes := []string{"node1", "node2", "node3"}

	// Initialize components
	shardingManager := vm.NewVMStateShardingManager(nodes)
	memDistManager := vm.NewMemoryStateDistribution(nodes)
	coordinator := vm.NewDistributedStateCoordinator(nodes)

	// Perform operations to generate metrics
	vmState := &vm.VMState{
		ID:       vmID,
		State:    vm.StateRunning,
		NodeID:   "node1",
		MemoryMB: 2048,
		UpdatedAt: time.Now(),
	}

	err := shardingManager.UpdateVMState(ctx, vmID, vmState)
	require.NoError(t, err)

	// Test memory distribution metrics
	memoryState := &vm.MemoryState{
		VMID:        vmID,
		TotalPages:  1000,
		PageSize:    4096,
		DirtyPages:  []string{"page_1", "page_2", "page_3"},
		LastUpdated: time.Now(),
	}

	err = memDistManager.DistributeMemoryState(ctx, vmID, memoryState)
	require.NoError(t, err)

	// Collect performance metrics
	metrics, err := coordinator.GetPerformanceMetrics(ctx, vmID)
	require.NoError(t, err)
	assert.NotNil(t, metrics)

	// Verify state access metrics
	assert.NotNil(t, metrics.StateAccess)
	assert.GreaterOrEqual(t, metrics.StateAccess.ThroughputOpsPerSec, 0.0)

	// Verify memory distribution metrics
	assert.NotNil(t, metrics.MemoryDistribution)
	assert.GreaterOrEqual(t, metrics.MemoryDistribution.CompressionRatio, 0.0)

	// Verify network performance metrics
	assert.NotNil(t, metrics.NetworkPerformance)
	assert.GreaterOrEqual(t, metrics.NetworkPerformance.ThroughputMbps, 0.0)

	// Verify consistency metrics
	assert.NotNil(t, metrics.ConsistencyMetrics)
	assert.GreaterOrEqual(t, metrics.ConsistencyMetrics.SyncSuccessRate, 0.0)

	// Test metrics aggregation across cluster
	clusterMetrics, err := coordinator.GetClusterMetrics(ctx)
	require.NoError(t, err)
	assert.NotNil(t, clusterMetrics)
	assert.Greater(t, len(clusterMetrics), 0)

	// Test performance optimization recommendations
	recommendations, err := coordinator.GetOptimizationRecommendations(ctx, vmID)
	require.NoError(t, err)
	assert.NotNil(t, recommendations)
}

// Benchmark tests for performance validation
func BenchmarkDistributedStateOperations(b *testing.B) {
	ctx := context.Background()
	shardingManager := vm.NewVMStateShardingManager([]string{"node1", "node2", "node3"})

	b.Run("StateUpdate", func(b *testing.B) {
		vmState := &vm.VMState{
			ID:       "bench-vm",
			State:    vm.StateRunning,
			NodeID:   "node1",
			MemoryMB: 1024,
			UpdatedAt: time.Now(),
		}

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			vmState.UpdatedAt = time.Now()
			err := shardingManager.UpdateVMState(ctx, fmt.Sprintf("bench-vm-%d", i), vmState)
			if err != nil {
				b.Fatal(err)
			}
		}
	})

	b.Run("StateRetrieval", func(b *testing.B) {
		// Pre-populate with test data
		vmState := &vm.VMState{
			ID:       "bench-vm-read",
			State:    vm.StateRunning,
			NodeID:   "node1",
			MemoryMB: 1024,
			UpdatedAt: time.Now(),
		}
		shardingManager.UpdateVMState(ctx, "bench-vm-read", vmState)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, err := shardingManager.GetVMState(ctx, "bench-vm-read")
			if err != nil {
				b.Fatal(err)
			}
		}
	})
}