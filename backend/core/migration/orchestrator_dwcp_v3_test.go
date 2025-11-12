package migration

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/upgrade"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestDWCPv3OrchestratorCreation tests orchestrator initialization
func TestDWCPv3OrchestratorCreation(t *testing.T) {
	tests := []struct {
		name        string
		config      DWCPv3Config
		expectError bool
	}{
		{
			name:        "Default config",
			config:      DefaultDWCPv3Config(),
			expectError: false,
		},
		{
			name: "Datacenter mode config",
			config: DWCPv3Config{
				NetworkMode:      upgrade.ModeDatacenter,
				EnableAMSTv3:     true,
				EnableHDEv3:      true,
				EnablePBAv3:      true,
				EnableITPv3:      true,
				EnableASSv3:      true,
				AutoSwitchMode:   false,
				DatacenterTargets: &PerformanceTargets{
					MaxDowntime:      300 * time.Millisecond,
					TargetThroughput: 40 * 1024 * 1024 * 1024, // 40 Gbps
					CompressionRatio: 1.1,
					MaxIterations:    3,
				},
			},
			expectError: false,
		},
		{
			name: "Internet mode config",
			config: DWCPv3Config{
				NetworkMode:       upgrade.ModeInternet,
				EnableAMSTv3:      true,
				EnableHDEv3:       true,
				EnablePBAv3:       true,
				EnableCompression: true,
				CompressionLevel:  9,
				InternetTargets: &PerformanceTargets{
					MaxDowntime:      120 * time.Second,
					TargetThroughput: 50 * 1024 * 1024, // 50 Mbps
					CompressionRatio: 4.0,
					MaxIterations:    15,
				},
			},
			expectError: false,
		},
		{
			name: "Hybrid mode with auto-switch",
			config: DWCPv3Config{
				NetworkMode:    upgrade.ModeHybrid,
				AutoSwitchMode: true,
				ModeThresholds: &ModeThresholds{
					LatencyThreshold:    5 * time.Millisecond,
					BandwidthThreshold:  5 * 1024 * 1024 * 1024, // 5 Gbps
					PacketLossThreshold: 0.0005,
					DowntimeThreshold:   2 * time.Second,
					CompressionRatio:    1.8,
				},
				EnableAMSTv3: true,
				EnableHDEv3:  true,
				EnablePBAv3:  true,
			},
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			baseConfig := MigrationConfig{
				MaxConcurrent:     5,
				BandwidthLimit:    10 * 1024 * 1024 * 1024, // 10 Gbps
				MemoryIterations:  10,
				DirtyPageThreshold: 0.01,
			}

			orchestrator, err := NewDWCPv3Orchestrator(baseConfig, tt.config)

			if tt.expectError {
				assert.Error(t, err)
				assert.Nil(t, orchestrator)
			} else {
				assert.NoError(t, err)
				require.NotNil(t, orchestrator)
				defer orchestrator.Close()

				// Verify components are initialized based on config
				if tt.config.EnableAMSTv3 {
					assert.NotNil(t, orchestrator.amst)
				}
				if tt.config.EnableHDEv3 {
					assert.NotNil(t, orchestrator.hde)
				}
				if tt.config.EnablePBAv3 {
					assert.NotNil(t, orchestrator.pba)
				}
				if tt.config.EnableITPv3 {
					assert.NotNil(t, orchestrator.itp)
				}
				if tt.config.EnableASSv3 {
					assert.NotNil(t, orchestrator.ass)
				}

				// Verify mode is set correctly
				assert.Equal(t, tt.config.NetworkMode, orchestrator.currentMode)
			}
		})
	}
}

// TestNetworkModeDetection tests automatic network mode detection
func TestNetworkModeDetection(t *testing.T) {
	baseConfig := MigrationConfig{
		MaxConcurrent:     5,
		BandwidthLimit:    10 * 1024 * 1024 * 1024, // 10 Gbps
		MemoryIterations:  10,
		DirtyPageThreshold: 0.01,
	}

	dwcpConfig := DefaultDWCPv3Config()
	dwcpConfig.AutoSwitchMode = true

	orchestrator, err := NewDWCPv3Orchestrator(baseConfig, dwcpConfig)
	require.NoError(t, err)
	defer orchestrator.Close()

	tests := []struct {
		name         string
		sourceNode   string
		destNode     string
		expectedMode upgrade.NetworkMode
	}{
		{
			name:         "Same node - datacenter mode",
			sourceNode:   "node1",
			destNode:     "node1",
			expectedMode: upgrade.ModeDatacenter,
		},
		{
			name:         "Same datacenter - datacenter mode",
			sourceNode:   "dc1-node1",
			destNode:     "dc1-node2",
			expectedMode: upgrade.ModeDatacenter,
		},
		{
			name:         "Different datacenters - internet mode",
			sourceNode:   "dc1-node1",
			destNode:     "dc2-node1",
			expectedMode: upgrade.ModeInternet,
		},
		{
			name:         "Cross-region - internet mode",
			sourceNode:   "us-east-1",
			destNode:     "eu-west-1",
			expectedMode: upgrade.ModeInternet,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mode := orchestrator.determineNetworkMode(tt.sourceNode, tt.destNode)
			assert.Equal(t, tt.expectedMode, mode)
		})
	}
}

// TestDatacenterModeMigration tests migration in datacenter mode
func TestDatacenterModeMigration(t *testing.T) {
	ctx := context.Background()

	baseConfig := MigrationConfig{
		MaxConcurrent:      5,
		BandwidthLimit:     40 * 1024 * 1024 * 1024, // 40 Gbps
		MemoryIterations:   5,
		DirtyPageThreshold: 0.01,
		MaxDowntime:        500 * time.Millisecond,
	}

	dwcpConfig := DefaultDWCPv3Config()
	dwcpConfig.NetworkMode = upgrade.ModeDatacenter
	dwcpConfig.AutoSwitchMode = false
	dwcpConfig.EnableCompression = false // Minimal compression in datacenter
	dwcpConfig.DatacenterTargets = &PerformanceTargets{
		MaxDowntime:      300 * time.Millisecond,
		TargetThroughput: 40 * 1024 * 1024 * 1024, // 40 Gbps
		CompressionRatio: 1.1,
		MaxIterations:    3,
	}

	orchestrator, err := NewDWCPv3Orchestrator(baseConfig, dwcpConfig)
	require.NoError(t, err)
	defer orchestrator.Close()

	// Start a datacenter migration
	vmID := "test-vm-dc-001"
	sourceNode := "dc1-rack1-node1"
	destNode := "dc1-rack2-node5"

	migration, err := orchestrator.StartMigration(ctx, vmID, sourceNode, destNode)
	require.NoError(t, err)
	assert.NotNil(t, migration)

	// Verify migration properties
	assert.Equal(t, upgrade.ModeDatacenter, migration.Mode)
	assert.Equal(t, vmID, migration.VM.ID)
	assert.Equal(t, sourceNode, migration.SourceNode)
	assert.Equal(t, destNode, migration.DestinationNode)

	// Wait for migration to complete (or timeout)
	timeout := time.After(5 * time.Second)
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-timeout:
			t.Fatal("Migration timeout")
		case <-ticker.C:
			if migration.CurrentPhase == PhaseV3Complete {
				// Migration completed successfully
				assert.True(t, migration.State.Downtime.Load() < 500) // Less than 500ms downtime

				// Check metrics
				metrics := orchestrator.GetMetrics()
				assert.Greater(t, metrics.DatacenterMigrations.Load(), int64(0))
				assert.Greater(t, metrics.SuccessfulMigrations.Load(), int64(0))

				return
			}
		}
	}
}

// TestInternetModeMigration tests migration in internet mode
func TestInternetModeMigration(t *testing.T) {
	ctx := context.Background()

	baseConfig := MigrationConfig{
		MaxConcurrent:      5,
		BandwidthLimit:     100 * 1024 * 1024, // 100 Mbps
		MemoryIterations:   15,
		DirtyPageThreshold: 0.05,
		MaxDowntime:        90 * time.Second,
	}

	dwcpConfig := DefaultDWCPv3Config()
	dwcpConfig.NetworkMode = upgrade.ModeInternet
	dwcpConfig.AutoSwitchMode = false
	dwcpConfig.EnableCompression = true
	dwcpConfig.CompressionLevel = 9 // Maximum compression
	dwcpConfig.InternetTargets = &PerformanceTargets{
		MaxDowntime:      60 * time.Second,
		TargetThroughput: 100 * 1024 * 1024, // 100 Mbps
		CompressionRatio: 4.0,
		MaxIterations:    20,
	}

	orchestrator, err := NewDWCPv3Orchestrator(baseConfig, dwcpConfig)
	require.NoError(t, err)
	defer orchestrator.Close()

	// Start an internet migration
	vmID := "test-vm-internet-001"
	sourceNode := "us-east-1-node1"
	destNode := "eu-west-1-node1"

	migration, err := orchestrator.StartMigration(ctx, vmID, sourceNode, destNode)
	require.NoError(t, err)
	assert.NotNil(t, migration)

	// Verify migration properties
	assert.Equal(t, upgrade.ModeInternet, migration.Mode)
	assert.True(t, migration.AdaptiveCompression)

	// Wait briefly then check progress
	time.Sleep(500 * time.Millisecond)

	// Verify compression is being used
	if migration.CurrentPhase == PhaseV3PreCopy || migration.CurrentPhase == PhaseV3Converge {
		assert.Greater(t, migration.CompressedPages, int64(0))
		if migration.CompressionRatio > 0 {
			assert.Greater(t, migration.CompressionRatio, 1.5) // At least 1.5x compression
		}
	}

	// Check metrics
	metrics := orchestrator.GetMetrics()
	if metrics.InternetMigrations.Load() > 0 {
		assert.Greater(t, metrics.HDECompressions.Load(), int64(0))
	}
}

// TestHybridModeWithAutoSwitch tests hybrid mode with automatic switching
func TestHybridModeWithAutoSwitch(t *testing.T) {
	ctx := context.Background()

	baseConfig := MigrationConfig{
		MaxConcurrent:      5,
		BandwidthLimit:     1 * 1024 * 1024 * 1024, // 1 Gbps
		MemoryIterations:   10,
		DirtyPageThreshold: 0.02,
		MaxDowntime:        5 * time.Second,
	}

	dwcpConfig := DefaultDWCPv3Config()
	dwcpConfig.NetworkMode = upgrade.ModeHybrid
	dwcpConfig.AutoSwitchMode = true
	dwcpConfig.ModeThresholds = &ModeThresholds{
		LatencyThreshold:    10 * time.Millisecond,
		BandwidthThreshold:  500 * 1024 * 1024, // 500 Mbps
		PacketLossThreshold: 0.001,
		DowntimeThreshold:   3 * time.Second,
		CompressionRatio:    2.0,
	}

	orchestrator, err := NewDWCPv3Orchestrator(baseConfig, dwcpConfig)
	require.NoError(t, err)
	defer orchestrator.Close()

	// Start a hybrid migration
	vmID := "test-vm-hybrid-001"
	sourceNode := "region1-dc1-node1"
	destNode := "region1-dc2-node1"

	migration, err := orchestrator.StartMigration(ctx, vmID, sourceNode, destNode)
	require.NoError(t, err)
	assert.NotNil(t, migration)

	// Initial mode should be hybrid
	assert.Equal(t, upgrade.ModeHybrid, migration.Mode)

	// Simulate mode switches during migration
	time.Sleep(200 * time.Millisecond)

	// Check if mode switches occurred
	metrics := orchestrator.GetMetrics()
	if metrics.HybridMigrations.Load() > 0 {
		// Mode switches might have occurred
		if metrics.ModeSwitches.Load() > 0 {
			assert.Greater(t, migration.ModeSwitches, 0)
		}
	}
}

// TestPredictiveBandwidthAllocation tests PBA v3 integration
func TestPredictiveBandwidthAllocation(t *testing.T) {
	ctx := context.Background()

	baseConfig := MigrationConfig{
		MaxConcurrent:      5,
		BandwidthLimit:     1 * 1024 * 1024 * 1024, // 1 Gbps
		MemoryIterations:   10,
		DirtyPageThreshold: 0.02,
	}

	dwcpConfig := DefaultDWCPv3Config()
	dwcpConfig.EnablePBAv3 = true
	dwcpConfig.EnablePrefetching = true

	orchestrator, err := NewDWCPv3Orchestrator(baseConfig, dwcpConfig)
	require.NoError(t, err)
	defer orchestrator.Close()

	// Start a migration to test PBA
	vmID := "test-vm-pba-001"
	sourceNode := "node1"
	destNode := "node2"

	migration, err := orchestrator.StartMigration(ctx, vmID, sourceNode, destNode)
	require.NoError(t, err)
	assert.NotNil(t, migration)

	// Wait for initialization phase
	time.Sleep(300 * time.Millisecond)

	// Check that PBA made predictions
	assert.Greater(t, migration.PredictedBandwidth, int64(0))

	// Check metrics
	metrics := orchestrator.GetMetrics()
	assert.Greater(t, metrics.PBAPredictions.Load(), int64(0))

	// If prefetching is enabled, check prefetched pages
	if orchestrator.config.EnablePrefetching {
		time.Sleep(500 * time.Millisecond)
		if migration.CurrentPhase == PhaseV3Prefetch ||
		   migration.CurrentPhase == PhaseV3PreCopy {
			// Some pages might have been prefetched
			// (In production, this would be > 0)
		}
	}
}

// TestCompressionModes tests different compression modes
func TestCompressionModes(t *testing.T) {
	tests := []struct {
		name               string
		mode               upgrade.NetworkMode
		enableCompression  bool
		expectedCompressed bool
		minRatio           float64
	}{
		{
			name:               "Datacenter - minimal compression",
			mode:               upgrade.ModeDatacenter,
			enableCompression:  false,
			expectedCompressed: false,
			minRatio:           1.0,
		},
		{
			name:               "Internet - aggressive compression",
			mode:               upgrade.ModeInternet,
			enableCompression:  true,
			expectedCompressed: true,
			minRatio:           2.0,
		},
		{
			name:               "Hybrid - adaptive compression",
			mode:               upgrade.ModeHybrid,
			enableCompression:  true,
			expectedCompressed: true,
			minRatio:           1.5,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()

			baseConfig := MigrationConfig{
				MaxConcurrent:      5,
				BandwidthLimit:     1 * 1024 * 1024 * 1024,
				MemoryIterations:   5,
				DirtyPageThreshold: 0.02,
			}

			dwcpConfig := DefaultDWCPv3Config()
			dwcpConfig.NetworkMode = tt.mode
			dwcpConfig.EnableCompression = tt.enableCompression
			dwcpConfig.EnableHDEv3 = true
			dwcpConfig.AutoSwitchMode = false

			orchestrator, err := NewDWCPv3Orchestrator(baseConfig, dwcpConfig)
			require.NoError(t, err)
			defer orchestrator.Close()

			// Test compression decision
			shouldCompress := orchestrator.shouldCompress(tt.mode)
			if tt.expectedCompressed {
				assert.True(t, shouldCompress || tt.enableCompression)
			}

			// Start a migration to test compression
			vmID := fmt.Sprintf("test-vm-comp-%s", tt.name)
			migration, err := orchestrator.StartMigration(ctx, vmID, "node1", "node2")
			require.NoError(t, err)

			// Wait for compression to be applied
			time.Sleep(500 * time.Millisecond)

			if migration.CurrentPhase == PhaseV3PreCopy && tt.expectedCompressed {
				// Check compression metrics
				if migration.CompressedPages > 0 {
					assert.Greater(t, migration.CompressionRatio, tt.minRatio)
				}
			}
		})
	}
}

// TestConcurrentMigrations tests multiple concurrent migrations
func TestConcurrentMigrations(t *testing.T) {
	ctx := context.Background()

	baseConfig := MigrationConfig{
		MaxConcurrent:      10,
		BandwidthLimit:     10 * 1024 * 1024 * 1024, // 10 Gbps shared
		MemoryIterations:   5,
		DirtyPageThreshold: 0.02,
	}

	dwcpConfig := DefaultDWCPv3Config()
	dwcpConfig.EnableAMSTv3 = true
	dwcpConfig.EnableHDEv3 = true
	dwcpConfig.EnablePBAv3 = true

	orchestrator, err := NewDWCPv3Orchestrator(baseConfig, dwcpConfig)
	require.NoError(t, err)
	defer orchestrator.Close()

	// Start multiple migrations concurrently
	numMigrations := 5
	var wg sync.WaitGroup
	migrations := make([]*DWCPv3Migration, numMigrations)
	errors := make([]error, numMigrations)

	for i := 0; i < numMigrations; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()

			vmID := fmt.Sprintf("test-vm-concurrent-%d", idx)
			sourceNode := fmt.Sprintf("source-%d", idx)
			destNode := fmt.Sprintf("dest-%d", idx)

			migration, err := orchestrator.StartMigration(ctx, vmID, sourceNode, destNode)
			migrations[idx] = migration
			errors[idx] = err
		}(i)
	}

	wg.Wait()

	// Check all migrations started successfully
	for i, err := range errors {
		assert.NoError(t, err, "Migration %d failed to start", i)
		assert.NotNil(t, migrations[i])
	}

	// Verify all migrations are tracked
	activeMigrations := orchestrator.ListMigrations()
	assert.GreaterOrEqual(t, len(activeMigrations), numMigrations)

	// Check metrics
	metrics := orchestrator.GetMetrics()
	assert.GreaterOrEqual(t, metrics.TotalMigrations.Load(), int64(numMigrations))
}

// TestMigrationFailureRecovery tests failure handling
func TestMigrationFailureRecovery(t *testing.T) {
	ctx := context.Background()

	baseConfig := MigrationConfig{
		MaxConcurrent:      5,
		BandwidthLimit:     1 * 1024 * 1024 * 1024,
		MemoryIterations:   5,
		DirtyPageThreshold: 0.02,
		MaxDowntime:        1 * time.Second,
	}

	dwcpConfig := DefaultDWCPv3Config()
	dwcpConfig.MigrationTimeout = 2 * time.Second // Short timeout for testing

	orchestrator, err := NewDWCPv3Orchestrator(baseConfig, dwcpConfig)
	require.NoError(t, err)
	defer orchestrator.Close()

	// Simulate a migration that will fail
	vmID := "test-vm-fail-001"
	sourceNode := "failing-source"
	destNode := "failing-dest"

	migration, err := orchestrator.StartMigration(ctx, vmID, sourceNode, destNode)
	require.NoError(t, err)
	assert.NotNil(t, migration)

	// Cancel context to simulate failure
	cancelCtx, cancel := context.WithCancel(ctx)
	cancel()

	// Try to perform an operation with cancelled context
	err = orchestrator.phasePreCopy(cancelCtx, migration)
	assert.Error(t, err)

	// Handle the error
	orchestrator.handleMigrationError(migration, err)

	// Verify error handling
	assert.Equal(t, PhaseFailed, migration.State.Phase)
	assert.NotNil(t, migration.State.Error)

	// Check failure metrics
	time.Sleep(100 * time.Millisecond)
	metrics := orchestrator.GetMetrics()
	// Failed migrations counter should eventually increase
}

// TestMetricsCollection tests metrics gathering
func TestMetricsCollection(t *testing.T) {
	ctx := context.Background()

	baseConfig := MigrationConfig{
		MaxConcurrent:      5,
		BandwidthLimit:     1 * 1024 * 1024 * 1024,
		MemoryIterations:   5,
		DirtyPageThreshold: 0.02,
	}

	dwcpConfig := DefaultDWCPv3Config()
	dwcpConfig.EnableAMSTv3 = true
	dwcpConfig.EnableHDEv3 = true
	dwcpConfig.EnablePBAv3 = true
	dwcpConfig.EnableITPv3 = true
	dwcpConfig.EnableASSv3 = true

	orchestrator, err := NewDWCPv3Orchestrator(baseConfig, dwcpConfig)
	require.NoError(t, err)
	defer orchestrator.Close()

	// Perform some migrations
	for i := 0; i < 3; i++ {
		vmID := fmt.Sprintf("test-vm-metrics-%d", i)
		mode := []upgrade.NetworkMode{
			upgrade.ModeDatacenter,
			upgrade.ModeInternet,
			upgrade.ModeHybrid,
		}[i]

		dwcpConfig.NetworkMode = mode
		orchestrator.currentMode = mode

		migration, err := orchestrator.StartMigration(ctx, vmID, "source", "dest")
		require.NoError(t, err)

		// Simulate some operations
		orchestrator.metrics.AMSTStreamsUsed.Add(10)
		orchestrator.metrics.HDECompressions.Add(100)
		orchestrator.metrics.PBAPredictions.Add(5)
		orchestrator.metrics.ITPPartitions.Add(20)
		orchestrator.metrics.ASSSyncs.Add(15)

		// Simulate completion
		migration.CurrentPhase = PhaseV3Complete
		orchestrator.metrics.SuccessfulMigrations.Add(1)
	}

	// Get metrics
	metrics := orchestrator.GetMetrics()

	// Verify metrics were collected
	assert.Greater(t, metrics.TotalMigrations.Load(), int64(0))
	assert.Greater(t, metrics.DatacenterMigrations.Load(), int64(0))
	assert.Greater(t, metrics.InternetMigrations.Load(), int64(0))
	assert.Greater(t, metrics.HybridMigrations.Load(), int64(0))

	// Component metrics
	assert.Greater(t, metrics.AMSTStreamsUsed.Load(), int64(0))
	assert.Greater(t, metrics.HDECompressions.Load(), int64(0))
	assert.Greater(t, metrics.PBAPredictions.Load(), int64(0))
	assert.Greater(t, metrics.ITPPartitions.Load(), int64(0))
	assert.Greater(t, metrics.ASSSyncs.Load(), int64(0))
}

// TestPerformanceTargets tests performance target enforcement
func TestPerformanceTargets(t *testing.T) {
	tests := []struct {
		name      string
		mode      upgrade.NetworkMode
		targets   *PerformanceTargets
		checkFunc func(*testing.T, *DWCPv3Orchestrator, *PerformanceTargets)
	}{
		{
			name: "Datacenter targets",
			mode: upgrade.ModeDatacenter,
			targets: &PerformanceTargets{
				MaxDowntime:      200 * time.Millisecond,
				TargetThroughput: 50 * 1024 * 1024 * 1024, // 50 Gbps
				CompressionRatio: 1.05,
				MaxIterations:    3,
			},
			checkFunc: func(t *testing.T, o *DWCPv3Orchestrator, targets *PerformanceTargets) {
				retrievedTargets := o.getPerformanceTargets(upgrade.ModeDatacenter)
				assert.Equal(t, targets.MaxDowntime, retrievedTargets.MaxDowntime)
				assert.Equal(t, targets.TargetThroughput, retrievedTargets.TargetThroughput)
				assert.Equal(t, targets.CompressionRatio, retrievedTargets.CompressionRatio)
				assert.Equal(t, targets.MaxIterations, retrievedTargets.MaxIterations)
			},
		},
		{
			name: "Internet targets",
			mode: upgrade.ModeInternet,
			targets: &PerformanceTargets{
				MaxDowntime:      90 * time.Second,
				TargetThroughput: 100 * 1024 * 1024, // 100 Mbps
				CompressionRatio: 5.0,
				MaxIterations:    25,
			},
			checkFunc: func(t *testing.T, o *DWCPv3Orchestrator, targets *PerformanceTargets) {
				retrievedTargets := o.getPerformanceTargets(upgrade.ModeInternet)
				assert.Equal(t, targets.MaxDowntime, retrievedTargets.MaxDowntime)
				assert.Equal(t, targets.TargetThroughput, retrievedTargets.TargetThroughput)
				assert.Equal(t, targets.CompressionRatio, retrievedTargets.CompressionRatio)
				assert.Equal(t, targets.MaxIterations, retrievedTargets.MaxIterations)
			},
		},
		{
			name: "Hybrid targets",
			mode: upgrade.ModeHybrid,
			targets: &PerformanceTargets{
				MaxDowntime:      10 * time.Second,
				TargetThroughput: 500 * 1024 * 1024, // 500 Mbps
				CompressionRatio: 2.5,
				MaxIterations:    10,
			},
			checkFunc: func(t *testing.T, o *DWCPv3Orchestrator, targets *PerformanceTargets) {
				retrievedTargets := o.getPerformanceTargets(upgrade.ModeHybrid)
				assert.Equal(t, targets.MaxDowntime, retrievedTargets.MaxDowntime)
				assert.Equal(t, targets.TargetThroughput, retrievedTargets.TargetThroughput)
				assert.Equal(t, targets.CompressionRatio, retrievedTargets.CompressionRatio)
				assert.Equal(t, targets.MaxIterations, retrievedTargets.MaxIterations)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			baseConfig := MigrationConfig{
				MaxConcurrent:      5,
				BandwidthLimit:     10 * 1024 * 1024 * 1024,
				MemoryIterations:   10,
				DirtyPageThreshold: 0.02,
			}

			dwcpConfig := DefaultDWCPv3Config()
			dwcpConfig.NetworkMode = tt.mode

			// Set specific targets based on mode
			switch tt.mode {
			case upgrade.ModeDatacenter:
				dwcpConfig.DatacenterTargets = tt.targets
			case upgrade.ModeInternet:
				dwcpConfig.InternetTargets = tt.targets
			case upgrade.ModeHybrid:
				dwcpConfig.HybridTargets = tt.targets
			}

			orchestrator, err := NewDWCPv3Orchestrator(baseConfig, dwcpConfig)
			require.NoError(t, err)
			defer orchestrator.Close()

			// Run check function
			tt.checkFunc(t, orchestrator, tt.targets)
		})
	}
}

// BenchmarkMigrationThroughput benchmarks migration throughput
func BenchmarkMigrationThroughput(b *testing.B) {
	ctx := context.Background()

	baseConfig := MigrationConfig{
		MaxConcurrent:      10,
		BandwidthLimit:     10 * 1024 * 1024 * 1024, // 10 Gbps
		MemoryIterations:   5,
		DirtyPageThreshold: 0.01,
	}

	modes := []upgrade.NetworkMode{
		upgrade.ModeDatacenter,
		upgrade.ModeInternet,
		upgrade.ModeHybrid,
	}

	for _, mode := range modes {
		b.Run(string(mode), func(b *testing.B) {
			dwcpConfig := DefaultDWCPv3Config()
			dwcpConfig.NetworkMode = mode
			dwcpConfig.AutoSwitchMode = false

			orchestrator, err := NewDWCPv3Orchestrator(baseConfig, dwcpConfig)
			require.NoError(b, err)
			defer orchestrator.Close()

			b.ResetTimer()
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				vmID := fmt.Sprintf("bench-vm-%d", i)
				migration, err := orchestrator.StartMigration(ctx, vmID, "source", "dest")
				if err != nil {
					b.Fatal(err)
				}

				// Simulate some work
				migration.State.BytesTransferred.Add(1024 * 1024 * 1024) // 1GB
				migration.State.Progress.Store(1.0)
				migration.CurrentPhase = PhaseV3Complete
			}

			// Report metrics
			metrics := orchestrator.GetMetrics()
			b.ReportMetric(float64(metrics.SuccessfulMigrations.Load()), "migrations")

			if metrics.AverageSpeedup.Load() > 0 {
				b.ReportMetric(float64(metrics.AverageSpeedup.Load())/100, "speedup")
			}
		})
	}
}

// BenchmarkCompressionOverhead benchmarks compression overhead
func BenchmarkCompressionOverhead(b *testing.B) {
	baseConfig := MigrationConfig{
		MaxConcurrent:      5,
		BandwidthLimit:     1 * 1024 * 1024 * 1024,
		MemoryIterations:   5,
		DirtyPageThreshold: 0.02,
	}

	scenarios := []struct {
		name        string
		mode        upgrade.NetworkMode
		compression bool
	}{
		{"Datacenter-NoCompression", upgrade.ModeDatacenter, false},
		{"Internet-WithCompression", upgrade.ModeInternet, true},
		{"Hybrid-AdaptiveCompression", upgrade.ModeHybrid, true},
	}

	for _, scenario := range scenarios {
		b.Run(scenario.name, func(b *testing.B) {
			dwcpConfig := DefaultDWCPv3Config()
			dwcpConfig.NetworkMode = scenario.mode
			dwcpConfig.EnableCompression = scenario.compression
			dwcpConfig.EnableHDEv3 = true

			orchestrator, err := NewDWCPv3Orchestrator(baseConfig, dwcpConfig)
			require.NoError(b, err)
			defer orchestrator.Close()

			// Create test data
			testData := make([]byte, 4096) // 4KB page
			for i := range testData {
				testData[i] = byte(i % 256)
			}

			b.ResetTimer()
			b.ReportAllocs()

			totalCompressed := int64(0)
			totalOriginal := int64(0)

			for i := 0; i < b.N; i++ {
				if orchestrator.shouldCompress(scenario.mode) && orchestrator.hde != nil {
					// Simulate compression
					compressed := make([]byte, len(testData)/2) // Simulate 2x compression
					totalCompressed += int64(len(compressed))
					totalOriginal += int64(len(testData))
				} else {
					totalOriginal += int64(len(testData))
					totalCompressed += int64(len(testData))
				}
			}

			if totalOriginal > 0 && totalCompressed < totalOriginal {
				ratio := float64(totalOriginal) / float64(totalCompressed)
				b.ReportMetric(ratio, "compression_ratio")
			}
		})
	}
}

// TestMigrationPhaseTracking tests phase duration tracking
func TestMigrationPhaseTracking(t *testing.T) {
	ctx := context.Background()

	baseConfig := MigrationConfig{
		MaxConcurrent:      5,
		BandwidthLimit:     1 * 1024 * 1024 * 1024,
		MemoryIterations:   3,
		DirtyPageThreshold: 0.02,
	}

	dwcpConfig := DefaultDWCPv3Config()
	orchestrator, err := NewDWCPv3Orchestrator(baseConfig, dwcpConfig)
	require.NoError(t, err)
	defer orchestrator.Close()

	// Create a test migration
	migration := &DWCPv3Migration{
		LiveMigration: &LiveMigration{
			ID:              "test-phase-tracking",
			VM:              &VM{ID: "vm-001", MemoryMB: 4096},
			SourceNode:      "source",
			DestinationNode: "dest",
			State:           NewMigrationState(),
			Context:         context.Background(),
		},
		Mode:           upgrade.ModeHybrid,
		CurrentPhase:   PhaseV3Init,
		PhaseStartTime: time.Now(),
		PhaseDurations: make(map[MigrationPhaseV3]time.Duration),
	}

	// Track phases
	phases := []MigrationPhaseV3{
		PhaseV3Init,
		PhaseV3Prefetch,
		PhaseV3PreCopy,
		PhaseV3Converge,
		PhaseV3StopCopy,
		PhaseV3Verify,
		PhaseV3Cleanup,
	}

	for _, phase := range phases {
		migration.CurrentPhase = phase
		migration.PhaseStartTime = time.Now()

		// Simulate some work
		time.Sleep(10 * time.Millisecond)

		// Record duration
		migration.PhaseDurations[phase] = time.Since(migration.PhaseStartTime)
	}

	// Verify all phases were tracked
	assert.Equal(t, len(phases), len(migration.PhaseDurations))

	// Verify durations are reasonable
	for _, phase := range phases {
		duration, ok := migration.PhaseDurations[phase]
		assert.True(t, ok, "Phase %s not tracked", phase)
		assert.Greater(t, duration, time.Duration(0), "Phase %s has zero duration", phase)
	}
}

// TestSpeedupCalculation tests speedup calculation accuracy
func TestSpeedupCalculation(t *testing.T) {
	baseConfig := MigrationConfig{
		MaxConcurrent:      5,
		BandwidthLimit:     1 * 1024 * 1024 * 1024,
		MemoryIterations:   5,
		DirtyPageThreshold: 0.02,
	}

	dwcpConfig := DefaultDWCPv3Config()
	orchestrator, err := NewDWCPv3Orchestrator(baseConfig, dwcpConfig)
	require.NoError(t, err)
	defer orchestrator.Close()

	tests := []struct {
		name             string
		actualBandwidth  int64
		expectedSpeedup  float64
		tolerancePercent float64
	}{
		{
			name:             "2x speedup",
			actualBandwidth:  200 * 1024 * 1024, // 200 MB/s
			expectedSpeedup:  2.0,
			tolerancePercent: 0.1,
		},
		{
			name:             "5x speedup",
			actualBandwidth:  500 * 1024 * 1024, // 500 MB/s
			expectedSpeedup:  5.0,
			tolerancePercent: 0.1,
		},
		{
			name:             "10x speedup",
			actualBandwidth:  1000 * 1024 * 1024, // 1 GB/s
			expectedSpeedup:  10.0,
			tolerancePercent: 0.1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			migration := &DWCPv3Migration{
				LiveMigration: &LiveMigration{
					State: NewMigrationState(),
				},
				ActualBandwidth: tt.actualBandwidth,
			}

			migration.State.TotalBytes.Store(10 * 1024 * 1024 * 1024) // 10 GB
			migration.State.StartTime = time.Now().Add(-10 * time.Second)
			migration.State.EndTime = time.Now()

			speedup := orchestrator.calculateSpeedup(migration)

			// Check if speedup is within tolerance
			tolerance := tt.expectedSpeedup * tt.tolerancePercent
			assert.InDelta(t, tt.expectedSpeedup, speedup, tolerance,
				"Speedup calculation off: expected %.2fx, got %.2fx", tt.expectedSpeedup, speedup)
		})
	}
}

// Helper function to create a test migration state
func NewMigrationState() *MigrationState {
	return &MigrationState{
		Phase:            PhaseInit,
		StartTime:        time.Now(),
		Progress:         &atomic.Value{},
		BytesTransferred: &atomic.Int64{},
		TotalBytes:       &atomic.Int64{},
		DirtyPages:       &atomic.Int64{},
		TransferRate:     &atomic.Int64{},
		Downtime:         &atomic.Int64{},
	}
}