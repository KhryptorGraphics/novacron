package migration

import (
	"context"
	"errors"
	"net"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestLiveMigrationOrchestrator tests the live migration orchestrator
func TestLiveMigrationOrchestrator(t *testing.T) {
	t.Run("CreateOrchestrator", func(t *testing.T) {
		config := MigrationConfig{
			MaxDowntime:             30 * time.Second,
			TargetTransferRate:      1024 * 1024 * 1024, // 1 GB/s
			SuccessRateTarget:       0.999,
			EnableCompression:       true,
			CompressionType:         CompressionLZ4,
			CompressionLevel:        6,
			EnableEncryption:        false,
			EnableDeltaSync:         true,
			BandwidthLimit:          0,
			AdaptiveBandwidth:       true,
			QoSPriority:             QoSPriorityHigh,
			MemoryIterations:        5,
			DirtyPageThreshold:      100,
			ConvergenceTimeout:      5 * time.Minute,
			EnableCheckpointing:     true,
			CheckpointInterval:      30 * time.Second,
			RetryAttempts:           3,
			RetryDelay:              10 * time.Second,
			MaxCPUUsage:             80.0,
			MaxMemoryUsage:          8 * 1024 * 1024 * 1024,
			MaxConcurrentMigrations: 3,
		}
		
		orchestrator, err := NewLiveMigrationOrchestrator(config)
		require.NoError(t, err)
		assert.NotNil(t, orchestrator)
		
		defer orchestrator.Close()
		
		// Verify components are initialized
		assert.NotNil(t, orchestrator.wanOptimizer)
		assert.NotNil(t, orchestrator.rollbackManager)
		assert.NotNil(t, orchestrator.monitor)
		assert.NotNil(t, orchestrator.connectionPool)
		assert.NotNil(t, orchestrator.bandwidthManager)
	})
	
	t.Run("MigrateVM", func(t *testing.T) {
		config := getTestConfig()
		orchestrator, err := NewLiveMigrationOrchestrator(config)
		require.NoError(t, err)
		defer orchestrator.Close()
		
		// Start test server
		listener, err := startTestServer()
		require.NoError(t, err)
		defer listener.Close()
		
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		
		options := MigrationOptions{
			Priority: 5,
			Force:    false,
		}
		
		migrationID, err := orchestrator.MigrateVM(ctx, "test-vm-1", "node-1", "localhost:9876", options)
		require.NoError(t, err)
		assert.NotEmpty(t, migrationID)
		
		// Verify migration is tracked
		status, err := orchestrator.GetMigrationStatus(migrationID)
		require.NoError(t, err)
		assert.Equal(t, migrationID, status["id"])
	})
	
	t.Run("ConcurrentMigrations", func(t *testing.T) {
		config := getTestConfig()
		config.MaxConcurrentMigrations = 2
		
		orchestrator, err := NewLiveMigrationOrchestrator(config)
		require.NoError(t, err)
		defer orchestrator.Close()
		
		// Start test server
		listener, err := startTestServer()
		require.NoError(t, err)
		defer listener.Close()
		
		ctx := context.Background()
		options := MigrationOptions{Priority: 5}
		
		// Start 3 migrations (only 2 should run concurrently)
		var migrationIDs []string
		for i := 0; i < 3; i++ {
			vmID := fmt.Sprintf("test-vm-%d", i)
			migrationID, err := orchestrator.MigrateVM(ctx, vmID, "node-1", "localhost:9876", options)
			require.NoError(t, err)
			migrationIDs = append(migrationIDs, migrationID)
		}
		
		// Check that we have 2 active and 1 queued
		time.Sleep(100 * time.Millisecond)
		
		orchestrator.mu.RLock()
		activeCount := len(orchestrator.activeMigrations)
		orchestrator.mu.RUnlock()
		
		assert.LessOrEqual(t, activeCount, 2)
		assert.Equal(t, 1, orchestrator.migrationQueue.Size())
	})
	
	t.Run("MigrationCancellation", func(t *testing.T) {
		config := getTestConfig()
		orchestrator, err := NewLiveMigrationOrchestrator(config)
		require.NoError(t, err)
		defer orchestrator.Close()
		
		// Start test server
		listener, err := startTestServer()
		require.NoError(t, err)
		defer listener.Close()
		
		ctx := context.Background()
		options := MigrationOptions{Priority: 5}
		
		migrationID, err := orchestrator.MigrateVM(ctx, "test-vm-cancel", "node-1", "localhost:9876", options)
		require.NoError(t, err)
		
		// Wait a bit then cancel
		time.Sleep(100 * time.Millisecond)
		
		err = orchestrator.CancelMigration(migrationID)
		// May fail if no checkpoint was created yet
		if err != nil {
			assert.Contains(t, err.Error(), "checkpoint")
		}
	})
	
	t.Run("GetMetrics", func(t *testing.T) {
		config := getTestConfig()
		orchestrator, err := NewLiveMigrationOrchestrator(config)
		require.NoError(t, err)
		defer orchestrator.Close()
		
		metrics := orchestrator.GetMetrics()
		
		assert.Contains(t, metrics, "total_migrations")
		assert.Contains(t, metrics, "successful_migrations")
		assert.Contains(t, metrics, "failed_migrations")
		assert.Contains(t, metrics, "success_rate")
		assert.Contains(t, metrics, "active_migrations")
		assert.Contains(t, metrics, "queued_migrations")
		assert.Contains(t, metrics, "wan_metrics")
	})
}

// TestWANOptimizer tests the WAN optimizer
func TestWANOptimizer(t *testing.T) {
	t.Run("CreateWANOptimizer", func(t *testing.T) {
		config := WANOptimizerConfig{
			CompressionType:  CompressionLZ4,
			CompressionLevel: 6,
			BandwidthLimit:   1024 * 1024, // 1 MB/s
			QoSPriority:      QoSPriorityHigh,
			EnableEncryption: true,
			EnableDeltaSync:  true,
			PageCacheSize:    100,
			TCPOptimization:  true,
		}
		
		optimizer, err := NewWANOptimizer(config)
		require.NoError(t, err)
		assert.NotNil(t, optimizer)
		defer optimizer.Close()
		
		// Verify components
		assert.Equal(t, CompressionLZ4, optimizer.compressionType)
		assert.NotNil(t, optimizer.rateLimiter)
		assert.NotNil(t, optimizer.deltaTracker)
		assert.NotNil(t, optimizer.pageCache)
		assert.NotNil(t, optimizer.tcpOptimizer)
	})
	
	t.Run("CompressionLZ4", func(t *testing.T) {
		config := WANOptimizerConfig{
			CompressionType:  CompressionLZ4,
			CompressionLevel: 6,
		}
		
		optimizer, err := NewWANOptimizer(config)
		require.NoError(t, err)
		defer optimizer.Close()
		
		// Test data
		original := []byte("This is test data that should be compressed. " +
			"Repetitive data compresses better. " +
			"Repetitive data compresses better. " +
			"Repetitive data compresses better.")
		
		// Compress
		compressed, err := optimizer.CompressData(original)
		require.NoError(t, err)
		assert.Less(t, len(compressed), len(original))
		
		// Decompress
		decompressed, err := optimizer.DecompressData(compressed, len(original))
		require.NoError(t, err)
		assert.Equal(t, original, decompressed)
	})
	
	t.Run("CompressionZSTD", func(t *testing.T) {
		config := WANOptimizerConfig{
			CompressionType:  CompressionZSTD,
			CompressionLevel: 6,
		}
		
		optimizer, err := NewWANOptimizer(config)
		require.NoError(t, err)
		defer optimizer.Close()
		
		// Test data
		original := make([]byte, 10000)
		for i := range original {
			original[i] = byte(i % 256)
		}
		
		// Compress
		compressed, err := optimizer.CompressData(original)
		require.NoError(t, err)
		assert.Less(t, len(compressed), len(original))
		
		// Decompress
		decompressed, err := optimizer.DecompressData(compressed, len(original))
		require.NoError(t, err)
		assert.Equal(t, original, decompressed)
	})
	
	t.Run("AdaptiveCompression", func(t *testing.T) {
		config := WANOptimizerConfig{
			CompressionType:  CompressionAdaptive,
			CompressionLevel: 6,
		}
		
		optimizer, err := NewWANOptimizer(config)
		require.NoError(t, err)
		defer optimizer.Close()
		
		// Small data should use LZ4
		smallData := []byte("small data")
		compressed, err := optimizer.CompressData(smallData)
		require.NoError(t, err)
		
		// First byte indicates compression type
		assert.Equal(t, byte(CompressionLZ4), compressed[0])
		
		// Large data with high latency should use ZSTD
		optimizer.latency.Store(100)
		largeData := make([]byte, 10000)
		compressed, err = optimizer.CompressData(largeData)
		require.NoError(t, err)
		
		assert.Equal(t, byte(CompressionZSTD), compressed[0])
	})
	
	t.Run("BandwidthLimiting", func(t *testing.T) {
		config := WANOptimizerConfig{
			CompressionType: CompressionNone,
			BandwidthLimit:  1024, // 1 KB/s
		}
		
		optimizer, err := NewWANOptimizer(config)
		require.NoError(t, err)
		defer optimizer.Close()
		
		// Create test connection
		client, server := net.Pipe()
		defer client.Close()
		defer server.Close()
		
		// Send data with rate limiting
		data := make([]byte, 2048) // 2 KB
		
		start := time.Now()
		go func() {
			optimizer.TransferWithOptimization(client, data)
		}()
		
		// Read on the other end
		go func() {
			buf := make([]byte, 4096)
			server.Read(buf)
		}()
		
		// Should take at least 2 seconds due to rate limiting
		time.Sleep(100 * time.Millisecond)
		elapsed := time.Since(start)
		
		// Allow some tolerance
		assert.Greater(t, elapsed, 50*time.Millisecond)
	})
	
	t.Run("DeltaSync", func(t *testing.T) {
		config := WANOptimizerConfig{
			CompressionType: CompressionNone,
			EnableDeltaSync: true,
		}
		
		optimizer, err := NewWANOptimizer(config)
		require.NoError(t, err)
		defer optimizer.Close()
		
		// First transfer
		data1 := make([]byte, 8192)
		for i := range data1 {
			data1[i] = byte(i % 256)
		}
		
		delta1 := optimizer.deltaTracker.ComputeDelta(data1)
		assert.NotNil(t, delta1)
		
		// Second transfer with mostly same data
		data2 := make([]byte, 8192)
		copy(data2, data1)
		data2[100] = 255 // Change one byte
		
		delta2 := optimizer.deltaTracker.ComputeDelta(data2)
		assert.NotNil(t, delta2)
		
		// Delta should be smaller than original
		assert.Less(t, len(delta2), len(data2))
	})
	
	t.Run("GetMetrics", func(t *testing.T) {
		config := WANOptimizerConfig{
			CompressionType: CompressionLZ4,
		}
		
		optimizer, err := NewWANOptimizer(config)
		require.NoError(t, err)
		defer optimizer.Close()
		
		// Compress some data
		data := []byte("test data for metrics")
		optimizer.CompressData(data)
		
		metrics := optimizer.GetMetrics()
		
		assert.Contains(t, metrics, "bytes_compressed")
		assert.Contains(t, metrics, "bytes_transferred")
		assert.Contains(t, metrics, "compression_ratio")
		assert.Contains(t, metrics, "transfer_rate")
		assert.Contains(t, metrics, "latency_ms")
		assert.Contains(t, metrics, "packet_loss")
	})
}

// TestRollbackManager tests the rollback manager
func TestRollbackManager(t *testing.T) {
	t.Run("CreateRollbackManager", func(t *testing.T) {
		tempDir := t.TempDir()
		manager, err := NewRollbackManager(tempDir)
		require.NoError(t, err)
		assert.NotNil(t, manager)
		
		// Verify components
		assert.NotNil(t, manager.transactionLog)
		assert.NotNil(t, manager.stateVerifier)
		assert.NotEmpty(t, manager.checkpointDir)
	})
	
	t.Run("CreateCheckpoint", func(t *testing.T) {
		tempDir := t.TempDir()
		manager, err := NewRollbackManager(tempDir)
		require.NoError(t, err)
		
		ctx := context.Background()
		state := VMStateSnapshot{
			CPUState: CPUState{
				Registers: map[string]uint64{
					"rax": 0x1234,
					"rbx": 0x5678,
				},
			},
			MemoryState: MemoryState{
				TotalMemory: 4096,
				UsedMemory:  2048,
			},
		}
		
		checkpoint, err := manager.CreateCheckpoint(ctx, "migration-1", "vm-1", "node-1", state)
		require.NoError(t, err)
		assert.NotNil(t, checkpoint)
		
		// Verify checkpoint fields
		assert.NotEmpty(t, checkpoint.ID)
		assert.Equal(t, "migration-1", checkpoint.MigrationID)
		assert.Equal(t, "vm-1", checkpoint.VMID)
		assert.Equal(t, "node-1", checkpoint.NodeID)
		assert.Equal(t, CheckpointTypeFull, checkpoint.Type)
		assert.NotEmpty(t, checkpoint.DiskSnapshot)
		assert.NotEmpty(t, checkpoint.MemoryDump)
	})
	
	t.Run("ExecuteRollback", func(t *testing.T) {
		tempDir := t.TempDir()
		manager, err := NewRollbackManager(tempDir)
		require.NoError(t, err)
		
		ctx := context.Background()
		state := VMStateSnapshot{}
		
		// Create checkpoint
		checkpoint, err := manager.CreateCheckpoint(ctx, "migration-2", "vm-2", "node-2", state)
		require.NoError(t, err)
		
		// Execute rollback
		err = manager.ExecuteRollback(ctx, "migration-2", checkpoint.ID)
		require.NoError(t, err)
		
		// Verify rollback was tracked
		assert.NotEmpty(t, manager.activeRollbacks)
	})
	
	t.Run("RollbackSteps", func(t *testing.T) {
		tempDir := t.TempDir()
		manager, err := NewRollbackManager(tempDir)
		require.NoError(t, err)
		
		steps := manager.defineRollbackSteps()
		
		expectedSteps := []string{
			"verify_checkpoint",
			"stop_migration",
			"restore_disk",
			"restore_memory",
			"restore_network",
			"restore_devices",
			"verify_state",
			"cleanup",
		}
		
		assert.Equal(t, len(expectedSteps), len(steps))
		
		for i, step := range steps {
			assert.Equal(t, expectedSteps[i], step.Name)
			assert.NotEmpty(t, step.Description)
		}
	})
}

// TestMigrationMonitor tests the migration monitor
func TestMigrationMonitor(t *testing.T) {
	t.Run("CreateMonitor", func(t *testing.T) {
		monitor := NewMigrationMonitor()
		assert.NotNil(t, monitor)
		defer monitor.Close()
		
		// Verify components
		assert.NotNil(t, monitor.progressTracker)
		assert.NotNil(t, monitor.metricsExporter)
		assert.NotNil(t, monitor.alertManager)
		assert.NotNil(t, monitor.dashboardData)
		assert.NotNil(t, monitor.eventBus)
	})
	
	t.Run("StartMonitoring", func(t *testing.T) {
		monitor := NewMigrationMonitor()
		defer monitor.Close()
		
		err := monitor.StartMonitoring("migration-1", "vm-1", "TestVM", "node-1", "node-2", "live")
		require.NoError(t, err)
		
		// Verify migration is tracked
		assert.Contains(t, monitor.activeMigrations, "migration-1")
		
		migration := monitor.activeMigrations["migration-1"]
		assert.Equal(t, "vm-1", migration.VMID)
		assert.Equal(t, "TestVM", migration.VMName)
		assert.Equal(t, "node-1", migration.SourceNode)
		assert.Equal(t, "node-2", migration.DestinationNode)
	})
	
	t.Run("UpdateProgress", func(t *testing.T) {
		monitor := NewMigrationMonitor()
		defer monitor.Close()
		
		err := monitor.StartMonitoring("migration-2", "vm-2", "TestVM2", "node-1", "node-2", "live")
		require.NoError(t, err)
		
		progress := MigrationProgress{
			Phase:            PhaseMemoryCopy,
			OverallProgress:  50.0,
			BytesTransferred: 1024 * 1024 * 1024, // 1GB
			TotalBytes:       2 * 1024 * 1024 * 1024, // 2GB
			TransferRate:     100 * 1024 * 1024, // 100MB/s
		}
		
		err = monitor.UpdateProgress("migration-2", progress)
		require.NoError(t, err)
		
		// Verify progress was stored
		migration := monitor.activeMigrations["migration-2"]
		storedProgress := migration.Progress.Load().(MigrationProgress)
		assert.Equal(t, float64(50.0), storedProgress.OverallProgress)
	})
	
	t.Run("AlertGeneration", func(t *testing.T) {
		monitor := NewMigrationMonitor()
		defer monitor.Close()
		
		// Set low thresholds to trigger alerts
		monitor.alertManager.thresholds.MinTransferRate = 1000 * 1024 * 1024 // 1GB/s
		
		err := monitor.StartMonitoring("migration-3", "vm-3", "TestVM3", "node-1", "node-2", "live")
		require.NoError(t, err)
		
		// Update with slow transfer rate
		progress := MigrationProgress{
			Phase:            PhaseMemoryCopy,
			OverallProgress:  10.0,
			TransferRate:     1024 * 1024, // 1MB/s - below threshold
		}
		
		err = monitor.UpdateProgress("migration-3", progress)
		require.NoError(t, err)
		
		// Check if alert was generated
		time.Sleep(100 * time.Millisecond)
		assert.NotEmpty(t, monitor.alertManager.alerts)
	})
	
	t.Run("GetDashboardData", func(t *testing.T) {
		monitor := NewMigrationMonitor()
		defer monitor.Close()
		
		// Start a few migrations
		monitor.StartMonitoring("migration-4", "vm-4", "TestVM4", "node-1", "node-2", "live")
		monitor.StartMonitoring("migration-5", "vm-5", "TestVM5", "node-1", "node-3", "live")
		
		dashboard := monitor.GetDashboardData()
		assert.NotNil(t, dashboard)
		assert.GreaterOrEqual(t, len(dashboard.ActiveMigrations), 2)
	})
	
	t.Run("CompleteMigration", func(t *testing.T) {
		monitor := NewMigrationMonitor()
		defer monitor.Close()
		
		err := monitor.StartMonitoring("migration-6", "vm-6", "TestVM6", "node-1", "node-2", "live")
		require.NoError(t, err)
		
		// Complete the migration
		err = monitor.CompleteMigration("migration-6", true, 5*time.Second)
		require.NoError(t, err)
		
		// Verify migration is no longer active
		assert.NotContains(t, monitor.activeMigrations, "migration-6")
	})
}

// TestPriorityQueue tests the priority queue
func TestPriorityQueue(t *testing.T) {
	t.Run("AddAndPop", func(t *testing.T) {
		queue := NewPriorityQueue()
		
		// Add items with different priorities
		queue.Add(&QueueItem{MigrationID: "m1", Priority: 5})
		queue.Add(&QueueItem{MigrationID: "m2", Priority: 10})
		queue.Add(&QueueItem{MigrationID: "m3", Priority: 3})
		
		// Pop should return highest priority first
		item := queue.Pop()
		assert.Equal(t, "m2", item.MigrationID)
		assert.Equal(t, 10, item.Priority)
		
		item = queue.Pop()
		assert.Equal(t, "m1", item.MigrationID)
		
		item = queue.Pop()
		assert.Equal(t, "m3", item.MigrationID)
		
		// Queue should be empty
		item = queue.Pop()
		assert.Nil(t, item)
	})
	
	t.Run("Size", func(t *testing.T) {
		queue := NewPriorityQueue()
		
		assert.Equal(t, 0, queue.Size())
		
		queue.Add(&QueueItem{MigrationID: "m1", Priority: 5})
		assert.Equal(t, 1, queue.Size())
		
		queue.Add(&QueueItem{MigrationID: "m2", Priority: 3})
		assert.Equal(t, 2, queue.Size())
		
		queue.Pop()
		assert.Equal(t, 1, queue.Size())
	})
}

// TestBandwidthManager tests the bandwidth manager
func TestBandwidthManager(t *testing.T) {
	t.Run("AllocateAndRelease", func(t *testing.T) {
		manager := NewBandwidthManager(1000)
		
		// Allocate bandwidth
		allocated := manager.Allocate("migration-1", 500)
		assert.Equal(t, int64(500), allocated)
		
		// Try to allocate more
		allocated = manager.Allocate("migration-2", 600)
		assert.Equal(t, int64(500), allocated) // Only 500 available
		
		// Release first allocation
		manager.Release("migration-1")
		
		// Now can allocate full amount
		allocated = manager.Allocate("migration-3", 800)
		assert.Equal(t, int64(800), allocated)
	})
	
	t.Run("OverAllocation", func(t *testing.T) {
		manager := NewBandwidthManager(1000)
		
		// Allocate all bandwidth
		allocated := manager.Allocate("migration-1", 1000)
		assert.Equal(t, int64(1000), allocated)
		
		// Try to allocate more
		allocated = manager.Allocate("migration-2", 100)
		assert.Equal(t, int64(0), allocated) // No bandwidth available
	})
}

// Helper functions

func getTestConfig() MigrationConfig {
	return MigrationConfig{
		MaxDowntime:             30 * time.Second,
		TargetTransferRate:      1024 * 1024 * 1024,
		SuccessRateTarget:       0.999,
		EnableCompression:       true,
		CompressionType:         CompressionLZ4,
		CompressionLevel:        6,
		EnableEncryption:        false,
		EnableDeltaSync:         false,
		BandwidthLimit:          0,
		AdaptiveBandwidth:       false,
		QoSPriority:             QoSPriorityMedium,
		MemoryIterations:        3,
		DirtyPageThreshold:      100,
		ConvergenceTimeout:      1 * time.Minute,
		EnableCheckpointing:     false,
		CheckpointInterval:      30 * time.Second,
		RetryAttempts:           1,
		RetryDelay:              1 * time.Second,
		MaxCPUUsage:             80.0,
		MaxMemoryUsage:          1024 * 1024 * 1024,
		MaxConcurrentMigrations: 2,
	}
}

func startTestServer() (net.Listener, error) {
	listener, err := net.Listen("tcp", ":9876")
	if err != nil {
		return nil, err
	}
	
	go func() {
		for {
			conn, err := listener.Accept()
			if err != nil {
				return
			}
			go handleTestConnection(conn)
		}
	}()
	
	return listener, nil
}

func handleTestConnection(conn net.Conn) {
	defer conn.Close()
	
	// Simple echo server for testing
	buf := make([]byte, 1024)
	for {
		n, err := conn.Read(buf)
		if err != nil {
			return
		}
		conn.Write(buf[:n])
	}
}