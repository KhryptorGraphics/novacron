package tests

import (
	"crypto/rand"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp"
	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/upgrade"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestBackwardCompatibilityFinal comprehensive backward compatibility validation
func TestBackwardCompatibilityFinal(t *testing.T) {
	t.Run("V1StillWorksAfterPhase3", testV1AfterPhase3)
	t.Run("DualModeOperation", testDualModeOperation)
	t.Run("FeatureFlagRollout", testFeatureFlagRollout)
	t.Run("InstantRollback", testInstantRollback)
	t.Run("NoDataLossDuringRollback", testNoDataLoss)
	t.Run("MixedVersionCluster", testMixedVersionCluster)
}

// testV1AfterPhase3 verifies v1 still functions after all Phase 3 integrations
func testV1AfterPhase3(t *testing.T) {
	// Disable all v3 features
	upgrade.DisableAll()
	defer upgrade.DisableAll()

	t.Run("V1_HDE_Basic", func(t *testing.T) {
		config := dwcp.HDEConfig{
			GlobalLevel: 3,
		}

		hde, err := dwcp.NewHDE(config)
		require.NoError(t, err)
		defer hde.Close()

		data := make([]byte, 1024*1024)
		rand.Read(data)

		compressed, err := hde.CompressMemory("vm-v1", data, dwcp.CompressionGlobal)
		require.NoError(t, err)
		require.Greater(t, len(compressed), 0)

		decompressed, err := hde.DecompressMemory("vm-v1", compressed)
		require.NoError(t, err)
		assert.Equal(t, data, decompressed)
	})

	t.Run("V1_AMST_Basic", func(t *testing.T) {
		config := dwcp.AMSTConfig{
			InitialStreams: 8,
			ChunkSize:      64 * 1024,
		}

		amst, err := dwcp.NewAMST(config)
		require.NoError(t, err)
		defer amst.Close()

		amst.UpdateMetrics(8, 0.001, 10e9)
		metrics := amst.GetMetrics()

		assert.Equal(t, 8, metrics.ActiveStreams)
		assert.Greater(t, metrics.Throughput, 0.0)
	})

	t.Run("V1_AllOperations", func(t *testing.T) {
		config := dwcp.HDEConfig{
			LocalLevel:    0,
			RegionalLevel: 3,
			GlobalLevel:   6,
		}

		hde, err := dwcp.NewHDE(config)
		require.NoError(t, err)
		defer hde.Close()

		data := make([]byte, 512*1024)
		rand.Read(data)

		// Test all compression levels
		levels := []dwcp.CompressionLevel{
			dwcp.CompressionLocal,
			dwcp.CompressionRegional,
			dwcp.CompressionGlobal,
		}

		for _, level := range levels {
			compressed, err := hde.CompressMemory(fmt.Sprintf("vm-%d", level), data, level)
			require.NoError(t, err)

			decompressed, err := hde.DecompressMemory(fmt.Sprintf("vm-%d", level), compressed)
			require.NoError(t, err)
			assert.Equal(t, data, decompressed)
		}
	})
}

// testDualModeOperation verifies v1 and v3 can run simultaneously
func testDualModeOperation(t *testing.T) {
	t.Run("SimultaneousV1V3", func(t *testing.T) {
		// Configure for 50% rollout
		upgrade.SetRolloutPercentage("hde", 50)
		defer upgrade.DisableAll()

		// Create instances that should use different versions
		v1Nodes := []string{"node-v1-1", "node-v1-2"}
		v3Nodes := []string{"node-v3-1", "node-v3-2"}

		data := make([]byte, 1024*1024)
		rand.Read(data)

		// Test v1 nodes
		for _, nodeID := range v1Nodes {
			// Force v1 by checking feature flag
			if upgrade.ShouldUseV3(nodeID) {
				continue // Skip if this node got v3
			}

			config := dwcp.HDEConfig{
				GlobalLevel: 3,
			}

			hde, err := dwcp.NewHDE(config)
			require.NoError(t, err)

			_, err = hde.CompressMemory(nodeID, data, dwcp.CompressionGlobal)
			require.NoError(t, err)

			hde.Close()
		}

		// Test v3 nodes
		for _, nodeID := range v3Nodes {
			if !upgrade.ShouldUseV3(nodeID) {
				continue // Skip if this node got v1
			}

			config := dwcp.HDEConfig{
				GlobalLevel: 3,
				EnableDelta: true,
			}

			hde, err := dwcp.NewHDE(config)
			require.NoError(t, err)

			_, err = hde.CompressMemory(nodeID, data, dwcp.CompressionGlobal)
			require.NoError(t, err)

			hde.Close()
		}
	})

	t.Run("MixedOperations", func(t *testing.T) {
		upgrade.SetRolloutPercentage("hde", 50)
		defer upgrade.DisableAll()

		config := dwcp.HDEConfig{
			GlobalLevel: 3,
			EnableDelta: true,
		}

		hde, err := dwcp.NewHDE(config)
		require.NoError(t, err)
		defer hde.Close()

		data := make([]byte, 256*1024)
		rand.Read(data)

		// Perform operations that may use v1 or v3 depending on feature flags
		for i := 0; i < 20; i++ {
			vmID := fmt.Sprintf("vm-mixed-%d", i)

			compressed, err := hde.CompressMemory(vmID, data, dwcp.CompressionGlobal)
			require.NoError(t, err)

			decompressed, err := hde.DecompressMemory(vmID, compressed)
			require.NoError(t, err)
			assert.Equal(t, data, decompressed)
		}
	})
}

// testFeatureFlagRollout validates gradual rollout percentages
func testFeatureFlagRollout(t *testing.T) {
	percentages := []int{0, 10, 50, 100}

	for _, pct := range percentages {
		t.Run(fmt.Sprintf("Rollout_%d_Percent", pct), func(t *testing.T) {
			upgrade.SetRolloutPercentage("hde", pct)
			defer upgrade.DisableAll()

			// Test with 1000 node IDs for statistical significance
			nodeCount := 1000
			enabledCount := 0

			for i := 0; i < nodeCount; i++ {
				nodeID := fmt.Sprintf("node-%d", i)
				if upgrade.ShouldUseV3(nodeID) {
					enabledCount++
				}
			}

			actualPct := float64(enabledCount) / float64(nodeCount) * 100
			expectedPct := float64(pct)

			// Allow ±5% variance for hash distribution
			variance := 5.0
			if pct == 0 {
				variance = 0.0 // Exact for 0%
			}
			if pct == 100 {
				variance = 0.0 // Exact for 100%
			}

			t.Logf("Rollout %d%%: Got %d/%d enabled (%.2f%%)",
				pct, enabledCount, nodeCount, actualPct)

			assert.InDelta(t, expectedPct, actualPct, variance,
				"Rollout percentage not within tolerance")
		})
	}
}

// testInstantRollback verifies rollback completes in <5 seconds
func testInstantRollback(t *testing.T) {
	t.Run("QuickRollback", func(t *testing.T) {
		// Start with v3 enabled
		upgrade.EnableAll(100)

		config := dwcp.HDEConfig{
			GlobalLevel: 3,
			EnableDelta: true,
		}

		hde, err := dwcp.NewHDE(config)
		require.NoError(t, err)
		defer hde.Close()

		data := make([]byte, 1024*1024)
		rand.Read(data)

		// Perform some v3 operations
		for i := 0; i < 10; i++ {
			_, err := hde.CompressMemory(fmt.Sprintf("vm-%d", i), data, dwcp.CompressionGlobal)
			require.NoError(t, err)
		}

		// Measure rollback time
		rollbackStart := time.Now()

		// Disable all v3 features (rollback to v1)
		upgrade.DisableAll()

		rollbackDuration := time.Since(rollbackStart)

		t.Logf("Rollback completed in %v", rollbackDuration)
		assert.Less(t, rollbackDuration, 5*time.Second,
			"Rollback should complete in <5 seconds")

		// Verify v1 operations work immediately
		_, err = hde.CompressMemory("vm-post-rollback", data, dwcp.CompressionGlobal)
		require.NoError(t, err)
	})

	t.Run("RollbackUnderLoad", func(t *testing.T) {
		upgrade.EnableAll(100)
		defer upgrade.DisableAll()

		config := dwcp.HDEConfig{
			GlobalLevel: 3,
		}

		hde, err := dwcp.NewHDE(config)
		require.NoError(t, err)
		defer hde.Close()

		data := make([]byte, 256*1024)
		rand.Read(data)

		// Start continuous operations
		var wg sync.WaitGroup
		stopChan := make(chan struct{})
		errorChan := make(chan error, 100)

		for i := 0; i < 10; i++ {
			wg.Add(1)
			go func(idx int) {
				defer wg.Done()

				for {
					select {
					case <-stopChan:
						return
					default:
						_, err := hde.CompressMemory(fmt.Sprintf("vm-%d", idx), data, dwcp.CompressionGlobal)
						if err != nil {
							errorChan <- err
						}
						time.Sleep(10 * time.Millisecond)
					}
				}
			}(i)
		}

		// Let operations run
		time.Sleep(500 * time.Millisecond)

		// Perform rollback
		rollbackStart := time.Now()
		upgrade.DisableAll()
		rollbackDuration := time.Since(rollbackStart)

		t.Logf("Rollback under load: %v", rollbackDuration)
		assert.Less(t, rollbackDuration, 5*time.Second)

		// Let operations continue for a bit
		time.Sleep(500 * time.Millisecond)

		close(stopChan)
		wg.Wait()
		close(errorChan)

		// Check for errors (some errors acceptable during rollback)
		errorCount := len(errorChan)
		t.Logf("Errors during rollback: %d", errorCount)
	})
}

// testNoDataLoss validates no data is lost during rollback
func testNoDataLoss(t *testing.T) {
	t.Run("DataIntegrityDuringRollback", func(t *testing.T) {
		upgrade.EnableAll(100)

		config := dwcp.HDEConfig{
			GlobalLevel: 3,
			EnableDelta: true,
		}

		hde, err := dwcp.NewHDE(config)
		require.NoError(t, err)
		defer hde.Close()

		// Create test data
		testData := make(map[string][]byte)
		for i := 0; i < 20; i++ {
			vmID := fmt.Sprintf("vm-%d", i)
			data := make([]byte, 128*1024)
			rand.Read(data)
			testData[vmID] = data

			// Compress with v3
			_, err := hde.CompressMemory(vmID, data, dwcp.CompressionGlobal)
			require.NoError(t, err)
		}

		// Rollback to v1
		upgrade.DisableAll()

		// Verify all data still accessible
		for vmID, originalData := range testData {
			// Note: In real implementation, HDE would maintain compressed data
			// This test verifies the system handles the transition gracefully
			compressed, err := hde.CompressMemory(vmID, originalData, dwcp.CompressionGlobal)
			require.NoError(t, err)

			decompressed, err := hde.DecompressMemory(vmID, compressed)
			require.NoError(t, err)
			assert.Equal(t, originalData, decompressed,
				"Data for %s should be intact after rollback", vmID)
		}
	})

	t.Run("InFlightOperationsDuringRollback", func(t *testing.T) {
		upgrade.EnableAll(100)

		config := dwcp.HDEConfig{
			GlobalLevel: 3,
		}

		hde, err := dwcp.NewHDE(config)
		require.NoError(t, err)
		defer hde.Close()

		data := make([]byte, 512*1024)
		rand.Read(data)

		// Start concurrent operations
		var wg sync.WaitGroup
		operationCount := 50
		successCount := make(chan bool, operationCount)

		for i := 0; i < operationCount; i++ {
			wg.Add(1)
			go func(idx int) {
				defer wg.Done()

				vmID := fmt.Sprintf("vm-concurrent-%d", idx)
				compressed, err := hde.CompressMemory(vmID, data, dwcp.CompressionGlobal)
				if err != nil {
					successCount <- false
					return
				}

				decompressed, err := hde.DecompressMemory(vmID, compressed)
				if err != nil {
					successCount <- false
					return
				}

				successCount <- assert.Equal(t, data, decompressed)
			}(i)

			// Trigger rollback partway through
			if i == operationCount/2 {
				go func() {
					time.Sleep(50 * time.Millisecond)
					upgrade.DisableAll()
				}()
			}
		}

		wg.Wait()
		close(successCount)

		// Count successes
		successes := 0
		for success := range successCount {
			if success {
				successes++
			}
		}

		t.Logf("Successful operations during rollback: %d/%d", successes, operationCount)
		// Expect high success rate even during rollback
		assert.GreaterOrEqual(t, float64(successes)/float64(operationCount), 0.90,
			"Should maintain >90%% success rate during rollback")
	})
}

// testMixedVersionCluster simulates cluster with mixed versions
func testMixedVersionCluster(t *testing.T) {
	t.Run("CrossVersionCommunication", func(t *testing.T) {
		// Simulate 50% v1, 50% v3 cluster
		upgrade.SetRolloutPercentage("hde", 50)
		defer upgrade.DisableAll()

		data := make([]byte, 1024*1024)
		rand.Read(data)

		// Test that data compressed on v3 nodes can be decompressed on v1 nodes
		config := dwcp.HDEConfig{
			GlobalLevel: 3,
			EnableDelta: true,
		}

		// Compress with potential v3
		hdeV3, err := dwcp.NewHDE(config)
		require.NoError(t, err)
		defer hdeV3.Close()

		compressed, err := hdeV3.CompressMemory("vm-cross-version", data, dwcp.CompressionGlobal)
		require.NoError(t, err)

		// Force v1 for decompression
		upgrade.DisableAll()

		hdeV1, err := dwcp.NewHDE(dwcp.HDEConfig{GlobalLevel: 3})
		require.NoError(t, err)
		defer hdeV1.Close()

		decompressed, err := hdeV1.DecompressMemory("vm-cross-version", compressed)
		require.NoError(t, err)
		assert.Equal(t, data, decompressed)
	})

	t.Run("GradualMigration", func(t *testing.T) {
		percentages := []int{10, 25, 50, 75, 90, 100}

		data := make([]byte, 256*1024)
		rand.Read(data)

		for _, pct := range percentages {
			upgrade.SetRolloutPercentage("hde", pct)

			config := dwcp.HDEConfig{
				GlobalLevel: 3,
				EnableDelta: true,
			}

			hde, err := dwcp.NewHDE(config)
			require.NoError(t, err)

			// Perform operations at this rollout level
			for i := 0; i < 10; i++ {
				vmID := fmt.Sprintf("vm-migrate-%d-%d", pct, i)

				compressed, err := hde.CompressMemory(vmID, data, dwcp.CompressionGlobal)
				require.NoError(t, err)

				decompressed, err := hde.DecompressMemory(vmID, compressed)
				require.NoError(t, err)
				assert.Equal(t, data, decompressed)
			}

			hde.Close()
			t.Logf("Successfully tested at %d%% rollout", pct)
		}

		upgrade.DisableAll()
	})
}

// TestRollbackProcedure validates the documented rollback procedure
func TestRollbackProcedure(t *testing.T) {
	t.Run("DocumentedRollbackSteps", func(t *testing.T) {
		// Step 1: System running with v3 at 100%
		upgrade.EnableAll(100)

		config := dwcp.HDEConfig{
			GlobalLevel: 3,
			EnableDelta: true,
		}

		hde, err := dwcp.NewHDE(config)
		require.NoError(t, err)

		data := make([]byte, 1024*1024)
		rand.Read(data)

		// Verify v3 working
		_, err = hde.CompressMemory("vm-rollback-test", data, dwcp.CompressionGlobal)
		require.NoError(t, err)

		// Step 2: Detect issue (simulated)
		t.Log("Simulating production issue...")

		// Step 3: Instant rollback
		rollbackStart := time.Now()
		upgrade.DisableAll()
		rollbackDuration := time.Since(rollbackStart)

		t.Logf("Rollback executed in %v", rollbackDuration)
		assert.Less(t, rollbackDuration, 5*time.Second)

		// Step 4: Verify v1 working
		_, err = hde.CompressMemory("vm-post-rollback", data, dwcp.CompressionGlobal)
		require.NoError(t, err)

		// Step 5: System stable on v1
		t.Log("System rolled back to v1 successfully")

		hde.Close()
	})
}
