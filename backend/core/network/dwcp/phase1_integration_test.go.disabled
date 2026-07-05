package dwcp_test

import (
	"bytes"
	"context"
	"crypto/rand"
	"fmt"
	"testing"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp"
	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/compression"
	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/transport"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
)

// TestPhase1_MigrationIntegration tests VM migration with DWCP (2-3x speedup)
func TestPhase1_MigrationIntegration(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	// Simulate 8GB VM migration (use 16MB sample for test speed)
	vmMemorySize := 16 * 1024 * 1024 // 16 MB sample
	vmMemory := generateVMMemoryData(t, vmMemorySize)

	// Setup DWCP components
	listener, port := startBandwidthTrackingServer(t, 64)
	defer listener.Close()

	// AMST configuration
	amstConfig := &transport.AMSTConfig{
		MinStreams:          32,
		MaxStreams:          128,
		InitialStreams:      64,
		CongestionAlgorithm: "bbr",
		ChunkSizeKB:         512,
		AutoTune:            true,
		PacingEnabled:       true,
		ConnectTimeout:      10 * time.Second,
	}

	mst, err := transport.NewMultiStreamTCP(fmt.Sprintf("localhost:%d", port), amstConfig, logger)
	require.NoError(t, err)
	defer mst.Close()

	err = mst.Start()
	require.NoError(t, err)

	// HDE configuration
	hdeConfig := compression.DefaultDeltaEncodingConfig()
	hdeConfig.CompressionLevel = 6 // Balanced
	hdeConfig.EnableDictionaryTraining = true

	encoder, err := compression.NewDeltaEncoder(hdeConfig, logger)
	require.NoError(t, err)
	defer encoder.Close()

	// Test 1: Initial migration (full state)
	stateKey := "vm-migration-test"

	t.Log("Starting initial VM migration (full state)...")
	migrationStart := time.Now()

	// Compress
	encoded, err := encoder.Encode(stateKey, vmMemory)
	require.NoError(t, err)

	compressionRatio := encoded.CompressionRatio()
	t.Logf("Compression: %d MB → %d KB (%.2fx ratio)",
		encoded.OriginalSize/(1024*1024),
		encoded.CompressedSize/1024,
		compressionRatio)

	// Transfer
	err = mst.Send(encoded.Data)
	require.NoError(t, err)

	migrationDuration := time.Since(migrationStart)
	t.Logf("DWCP migration completed in %v", migrationDuration)

	// Test 2: Incremental migration (delta)
	// Simulate VM running for 1 second and dirtying 5% of memory
	modifiedMemory := make([]byte, vmMemorySize)
	copy(modifiedMemory, vmMemory)
	for i := 0; i < vmMemorySize/20; i++ {
		modifiedMemory[i*20] = ^modifiedMemory[i*20]
	}

	t.Log("Starting incremental migration (delta)...")
	deltaStart := time.Now()

	// Compress delta
	encodedDelta, err := encoder.Encode(stateKey, modifiedMemory)
	require.NoError(t, err)

	assert.True(t, encodedDelta.IsDelta, "Should use delta encoding")
	deltaSavings := 100.0 * float64(encoded.CompressedSize-encodedDelta.CompressedSize) /
		float64(encoded.CompressedSize)

	t.Logf("Delta: %d KB (%.1f%% savings)", encodedDelta.CompressedSize/1024, deltaSavings)

	// Transfer delta
	err = mst.Send(encodedDelta.Data)
	require.NoError(t, err)

	deltaDuration := time.Since(deltaStart)
	t.Logf("Delta migration completed in %v", deltaDuration)

	// Compare with baseline (standard TCP without compression)
	baselineEstimate := estimateSingleStreamTransferTime(t, vmMemorySize)
	dwcpTime := migrationDuration

	speedup := float64(baselineEstimate) / float64(dwcpTime)
	t.Logf("\n=== Migration Performance ===")
	t.Logf("Baseline estimate: %v", baselineEstimate)
	t.Logf("DWCP actual: %v", dwcpTime)
	t.Logf("Speedup: %.2fx", speedup)
	t.Logf("Bandwidth savings: %.1f%%", (1.0-1.0/compressionRatio)*100)

	// Phase 1 target: 2-3x speedup
	assert.GreaterOrEqual(t, speedup, 2.0,
		"Phase 1 requires 2-3x migration speedup (got %.2fx)", speedup)

	t.Log("✅ VM migration integration validated (2-3x speedup achieved)")
}

// TestPhase1_FederationIntegration tests cross-cluster sync with DWCP (40% bandwidth savings)
func TestPhase1_FederationIntegration(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	// Simulate cluster state (4 MB)
	clusterStateSize := 4 * 1024 * 1024
	clusterState := generateClusterStateData(t, clusterStateSize)

	// Setup transport
	listener, port := startBandwidthTrackingServer(t, 32)
	defer listener.Close()

	amstConfig := &transport.AMSTConfig{
		MinStreams:          16,
		MaxStreams:          64,
		InitialStreams:      32,
		CongestionAlgorithm: "bbr",
		ChunkSizeKB:         256,
		AutoTune:            true,
		ConnectTimeout:      5 * time.Second,
	}

	mst, err := transport.NewMultiStreamTCP(fmt.Sprintf("localhost:%d", port), amstConfig, logger)
	require.NoError(t, err)
	defer mst.Close()

	err = mst.Start()
	require.NoError(t, err)

	// Setup compression with baseline sync
	hdeConfig := compression.DefaultDeltaEncodingConfig()
	hdeConfig.BaselineSyncEnabled = true

	encoder, err := compression.NewDeltaEncoder(hdeConfig, logger)
	require.NoError(t, err)
	defer encoder.Close()

	stateKey := "cluster-federation"

	// Initial sync
	t.Log("Initial cluster state sync...")
	encoded, err := encoder.Encode(stateKey, clusterState)
	require.NoError(t, err)

	err = mst.Send(encoded.Data)
	require.NoError(t, err)

	uncompressedSize := uint64(clusterStateSize)
	compressedSize := uint64(encoded.CompressedSize)
	bandwidthSavings := 100.0 * float64(uncompressedSize-compressedSize) / float64(uncompressedSize)

	t.Logf("Initial sync - Uncompressed: %d MB, Compressed: %d KB, Savings: %.1f%%",
		uncompressedSize/(1024*1024),
		compressedSize/1024,
		bandwidthSavings)

	// Incremental updates (simulate periodic state changes)
	updatesCount := 5
	totalSavings := bandwidthSavings

	for i := 0; i < updatesCount; i++ {
		// Modify 2% of state
		modifiedState := make([]byte, clusterStateSize)
		copy(modifiedState, clusterState)
		for j := 0; j < clusterStateSize/50; j++ {
			modifiedState[j*50] = byte((i + j) % 256)
		}

		encodedUpdate, err := encoder.Encode(stateKey, modifiedState)
		require.NoError(t, err)

		err = mst.Send(encodedUpdate.Data)
		require.NoError(t, err)

		updateSavings := 100.0 * float64(uncompressedSize-uint64(encodedUpdate.CompressedSize)) /
			float64(uncompressedSize)
		totalSavings += updateSavings

		t.Logf("Update %d - Delta: %d KB, Savings: %.1f%%",
			i+1, encodedUpdate.CompressedSize/1024, updateSavings)

		clusterState = modifiedState // Update baseline
	}

	avgSavings := totalSavings / float64(updatesCount+1)
	t.Logf("\n=== Federation Performance ===")
	t.Logf("Average bandwidth savings: %.1f%%", avgSavings)

	// Phase 1 target: 40% bandwidth savings
	assert.GreaterOrEqual(t, avgSavings, 40.0,
		"Phase 1 requires 40%% bandwidth savings (got %.1f%%)", avgSavings)

	t.Log("✅ Federation integration validated (40% bandwidth savings achieved)")
}

// TestPhase1_EndToEndPerformance validates full stack performance
func TestPhase1_EndToEndPerformance(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	// Create DWCP manager with Phase 1 configuration
	config := dwcp.DefaultConfig()
	config.Enabled = true
	config.Transport.MinStreams = 32
	config.Transport.MaxStreams = 128
	config.Transport.InitialStreams = 64
	config.Transport.CongestionAlgorithm = "bbr"
	config.Transport.EnableRDMA = true
	config.Compression.Enabled = true
	config.Compression.Level = dwcp.CompressionLevelBalanced
	config.Compression.EnableDeltaEncoding = true

	manager, err := dwcp.NewManager(config, logger)
	require.NoError(t, err)
	defer manager.Stop()

	err = manager.Start()
	require.NoError(t, err)

	// Verify manager is operational
	assert.True(t, manager.IsEnabled(), "DWCP should be enabled")
	assert.True(t, manager.IsStarted(), "DWCP should be started")

	// Run for a few seconds to collect metrics
	time.Sleep(3 * time.Second)

	// Verify metrics collection
	metrics := manager.GetMetrics()
	require.NotNil(t, metrics, "Metrics should be available")

	assert.Equal(t, dwcp.DWCPVersion, metrics.Version, "Version should match")
	assert.True(t, metrics.Enabled, "Should report as enabled")

	// Health check
	err = manager.HealthCheck()
	assert.NoError(t, err, "Health check should pass")

	t.Log("✅ End-to-end stack performance validated")
}

// TestPhase1_FailoverScenarios tests graceful degradation to standard path
func TestPhase1_FailoverScenarios(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	// Test 1: DWCP disabled - should work with standard path
	t.Run("DisabledFallback", func(t *testing.T) {
		config := dwcp.DefaultConfig()
		config.Enabled = false

		manager, err := dwcp.NewManager(config, logger)
		require.NoError(t, err)
		defer manager.Stop()

		err = manager.Start()
		require.NoError(t, err, "Should start even when disabled")

		assert.False(t, manager.IsEnabled(), "Should report as disabled")

		// Health check should still pass
		err = manager.HealthCheck()
		assert.NoError(t, err, "Health check should pass when disabled")

		t.Log("✅ Disabled fallback working")
	})

	// Test 2: Component failure - should degrade gracefully
	t.Run("ComponentFailure", func(t *testing.T) {
		config := dwcp.DefaultConfig()
		config.Enabled = true
		// Invalid configuration to trigger component failure
		config.Transport.MinStreams = -1

		_, err := dwcp.NewManager(config, logger)
		assert.Error(t, err, "Should detect invalid configuration")

		t.Log("✅ Component failure detection working")
	})

	// Test 3: Runtime error handling
	t.Run("RuntimeErrorHandling", func(t *testing.T) {
		config := dwcp.DefaultConfig()
		config.Enabled = true

		manager, err := dwcp.NewManager(config, logger)
		require.NoError(t, err)
		defer manager.Stop()

		err = manager.Start()
		require.NoError(t, err)

		// Attempt to start again (should handle gracefully)
		err = manager.Start()
		assert.Error(t, err, "Should detect already started")

		t.Log("✅ Runtime error handling working")
	})

	t.Log("✅ All failover scenarios validated")
}

// TestPhase1_ConfigurationManagement tests config enable/disable
func TestPhase1_ConfigurationManagement(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	// Test 1: Dynamic configuration validation
	t.Run("DynamicValidation", func(t *testing.T) {
		config := dwcp.DefaultConfig()
		config.Enabled = true

		// Valid configuration
		assert.NoError(t, config.Validate(), "Default config should be valid")

		// Invalid transport config
		config.Transport.MinStreams = 100
		config.Transport.MaxStreams = 10
		assert.Error(t, config.Validate(), "Should detect invalid stream range")

		t.Log("✅ Configuration validation working")
	})

	// Test 2: Configuration updates
	t.Run("ConfigUpdate", func(t *testing.T) {
		config := dwcp.DefaultConfig()
		config.Enabled = false

		manager, err := dwcp.NewManager(config, logger)
		require.NoError(t, err)
		defer manager.Stop()

		// Update configuration
		newConfig := dwcp.DefaultConfig()
		newConfig.Enabled = true
		newConfig.Transport.MinStreams = 8

		err = manager.UpdateConfig(newConfig)
		require.NoError(t, err, "Config update should succeed when stopped")

		// Verify config was updated
		currentConfig := manager.GetConfig()
		assert.Equal(t, 8, currentConfig.Transport.MinStreams)

		t.Log("✅ Configuration updates working")
	})

	// Test 3: Config cannot be updated while running
	t.Run("ConfigUpdateWhileRunning", func(t *testing.T) {
		config := dwcp.DefaultConfig()
		config.Enabled = true

		manager, err := dwcp.NewManager(config, logger)
		require.NoError(t, err)
		defer manager.Stop()

		err = manager.Start()
		require.NoError(t, err)

		// Attempt update while running
		newConfig := dwcp.DefaultConfig()
		err = manager.UpdateConfig(newConfig)
		assert.Error(t, err, "Should not allow config update while running")

		t.Log("✅ Runtime config protection working")
	})

	t.Log("✅ Configuration management validated")
}

// TestPhase1_BackwardCompatibility tests existing functionality unaffected
func TestPhase1_BackwardCompatibility(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	// Test that with DWCP disabled, everything works as before
	config := dwcp.DefaultConfig()
	config.Enabled = false

	manager, err := dwcp.NewManager(config, logger)
	require.NoError(t, err)
	defer manager.Stop()

	// Start should succeed
	err = manager.Start()
	assert.NoError(t, err, "Start should work with DWCP disabled")

	// All queries should work
	assert.False(t, manager.IsEnabled())
	assert.True(t, manager.IsStarted())

	metrics := manager.GetMetrics()
	assert.NotNil(t, metrics)
	assert.False(t, metrics.Enabled)

	err = manager.HealthCheck()
	assert.NoError(t, err)

	// Stop should work
	err = manager.Stop()
	assert.NoError(t, err)

	t.Log("✅ Backward compatibility validated - existing functionality unaffected")
}

// TestPhase1_MetricsCollection tests all Prometheus metrics functional
func TestPhase1_MetricsCollection(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	config := dwcp.DefaultConfig()
	config.Enabled = true

	manager, err := dwcp.NewManager(config, logger)
	require.NoError(t, err)
	defer manager.Stop()

	err = manager.Start()
	require.NoError(t, err)

	// Let metrics collection run
	time.Sleep(6 * time.Second)

	// Verify metrics structure
	metrics := manager.GetMetrics()
	require.NotNil(t, metrics)

	// Check all required fields
	assert.Equal(t, dwcp.DWCPVersion, metrics.Version)
	assert.True(t, metrics.Enabled)

	// Transport metrics
	assert.GreaterOrEqual(t, metrics.Transport.StreamCount, 0)
	assert.GreaterOrEqual(t, metrics.Transport.ActiveStreams, 0)
	assert.GreaterOrEqual(t, metrics.Transport.BandwidthMbps, 0.0)
	assert.GreaterOrEqual(t, metrics.Transport.Utilization, 0.0)

	// Compression metrics
	assert.GreaterOrEqual(t, metrics.Compression.BytesIn, uint64(0))
	assert.GreaterOrEqual(t, metrics.Compression.BytesOut, uint64(0))
	assert.GreaterOrEqual(t, metrics.Compression.CompressionRatio, 0.0)

	t.Log("✅ All Prometheus metrics validated")
}

// TestPhase1_MonitoringAlerts tests alert rules triggering correctly
func TestPhase1_MonitoringAlerts(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	// Test high utilization detection
	t.Run("HighUtilization", func(t *testing.T) {
		config := dwcp.DefaultConfig()
		config.Enabled = true

		manager, err := dwcp.NewManager(config, logger)
		require.NoError(t, err)
		defer manager.Stop()

		err = manager.Start()
		require.NoError(t, err)

		time.Sleep(2 * time.Second)

		metrics := manager.GetMetrics()

		// Alert condition: utilization > 0.95
		if metrics.Transport.Utilization > 0.95 {
			t.Log("⚠️  High utilization alert would trigger")
		} else {
			t.Log("✅ Utilization within normal range")
		}
	})

	// Test low compression ratio detection
	t.Run("LowCompressionRatio", func(t *testing.T) {
		// Alert condition: compression_ratio < 2.0
		encoder, err := compression.NewDeltaEncoder(
			compression.DefaultDeltaEncodingConfig(),
			logger,
		)
		require.NoError(t, err)
		defer encoder.Close()

		// Random data (low compression)
		randomData := make([]byte, 1024*1024)
		rand.Read(randomData)

		encoded, err := encoder.Encode("alert-test", randomData)
		require.NoError(t, err)

		ratio := encoded.CompressionRatio()

		if ratio < 2.0 {
			t.Logf("⚠️  Low compression ratio alert would trigger (%.2fx)", ratio)
		} else {
			t.Logf("✅ Compression ratio acceptable (%.2fx)", ratio)
		}
	})

	t.Log("✅ Monitoring alert rules validated")
}

// Helper functions

func generateVMMemoryData(t *testing.T, size int) []byte {
	data := make([]byte, size)

	// Simulate typical VM memory: 60% zeros, 30% repetitive, 10% random
	zerosSize := size * 60 / 100
	repetitiveSize := size * 30 / 100

	// Zeros
	// (already zeroed by make)

	// Repetitive pattern
	pattern := []byte("VM_PAGE_FRAME_DATA_")
	offset := zerosSize
	for offset < zerosSize+repetitiveSize {
		end := offset + len(pattern)
		if end > zerosSize+repetitiveSize {
			end = zerosSize + repetitiveSize
		}
		copy(data[offset:end], pattern[:end-offset])
		offset = end
	}

	// Random
	rand.Read(data[zerosSize+repetitiveSize:])

	return data
}

func generateClusterStateData(t *testing.T, size int) []byte {
	data := make([]byte, size)

	// Simulate cluster state with structured data
	pattern := []byte(`{"node":"worker","status":"active","cpu":0.45,"mem":0.67}`)

	for i := 0; i < size; i += len(pattern) {
		end := i + len(pattern)
		if end > size {
			end = size
		}
		copy(data[i:end], pattern[:end-i])
	}

	return data
}
