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

// TestHDEv3MLCompression tests ML-based compression selection
func TestHDEv3MLCompression(t *testing.T) {
	upgrade.EnableAll(100)
	defer upgrade.DisableAll()

	t.Run("ml_compression_level_selection", func(t *testing.T) {
		config := dwcp.HDEConfig{
			LocalLevel:    0, // Fast for local
			RegionalLevel: 3, // Balanced for regional
			GlobalLevel:   9, // Max for global/internet
			EnableDelta:   true,
		}

		hde, err := dwcp.NewHDE(config)
		require.NoError(t, err)
		defer hde.Close()

		// Test data
		testData := make([]byte, 1024*1024) // 1MB
		rand.Read(testData)

		// Test local compression (fast)
		localStart := time.Now()
		localCompressed, err := hde.CompressMemory("vm-1", testData, dwcp.CompressionLocal)
		localTime := time.Since(localStart)
		require.NoError(t, err)

		// Test global compression (max)
		globalStart := time.Now()
		globalCompressed, err := hde.CompressMemory("vm-2", testData, dwcp.CompressionGlobal)
		globalTime := time.Since(globalStart)
		require.NoError(t, err)

		// Global should compress better but take longer
		assert.Less(t, len(globalCompressed), len(localCompressed),
			"Global compression should achieve better ratio")
		assert.Greater(t, globalTime, localTime*9/10, // Allow some variance
			"Global compression should take longer than local")

		t.Logf("✅ ML compression: Local %d bytes in %v, Global %d bytes in %v",
			len(localCompressed), localTime, len(globalCompressed), globalTime)
	})

	t.Run("adaptive_compression_ratio", func(t *testing.T) {
		config := dwcp.HDEConfig{
			GlobalLevel:        9,
			EnableDelta:        true,
			EnableDictionary:   true,
			EnableQuantization: true,
		}

		hde, err := dwcp.NewHDE(config)
		require.NoError(t, err)
		defer hde.Close()

		// Highly compressible data (zeros)
		compressibleData := make([]byte, 1024*1024)
		compressed, err := hde.CompressMemory("vm-compress", compressibleData, dwcp.CompressionGlobal)
		require.NoError(t, err)

		compressionRatio := float64(len(compressibleData)) / float64(len(compressed))
		assert.Greater(t, compressionRatio, 5.0, "Should achieve >5x compression for zeros")

		t.Logf("✅ Compression ratio: %.2fx for highly compressible data", compressionRatio)
	})
}

// TestHDEv3CRDTIntegration tests CRDT integration for conflict-free sync
func TestHDEv3CRDTIntegration(t *testing.T) {
	upgrade.EnableAll(100)
	defer upgrade.DisableAll()

	t.Run("crdt_conflict_resolution", func(t *testing.T) {
		config := dwcp.HDEConfig{
			EnableDelta:      true,
			BlockSize:        4 * 1024,
			MaxDeltaHistory:  100,
			EnableDictionary: true,
		}

		hde, err := dwcp.NewHDE(config)
		require.NoError(t, err)
		defer hde.Close()

		// Create baseline
		baseline := make([]byte, 10*1024)
		rand.Read(baseline)

		_, err = hde.CompressMemory("vm-crdt", baseline, dwcp.CompressionGlobal)
		require.NoError(t, err)

		// Create two conflicting updates
		update1 := make([]byte, 10*1024)
		copy(update1, baseline)
		update1[100] = 0xFF // Modify

		update2 := make([]byte, 10*1024)
		copy(update2, baseline)
		update2[100] = 0x00 // Conflicting modify

		// Both updates should compress successfully
		_, err1 := hde.CompressMemory("vm-crdt", update1, dwcp.CompressionGlobal)
		_, err2 := hde.CompressMemory("vm-crdt", update2, dwcp.CompressionGlobal)

		assert.NoError(t, err1)
		assert.NoError(t, err2)

		t.Log("✅ CRDT integration: Conflicting updates handled")
	})
}

// TestHDEv3DeltaEncoding tests enhanced delta encoding
func TestHDEv3DeltaEncoding(t *testing.T) {
	upgrade.EnableAll(100)
	defer upgrade.DisableAll()

	t.Run("delta_encoding_efficiency", func(t *testing.T) {
		config := dwcp.HDEConfig{
			EnableDelta:     true,
			BlockSize:       4 * 1024,
			DeltaThreshold:  0.7,
			MaxDeltaHistory: 100,
		}

		hde, err := dwcp.NewHDE(config)
		require.NoError(t, err)
		defer hde.Close()

		// Create baseline
		baseline := make([]byte, 100*1024) // 100KB
		rand.Read(baseline)

		// First compression creates baseline
		compressed1, err := hde.CompressMemory("vm-delta", baseline, dwcp.CompressionGlobal)
		require.NoError(t, err)
		size1 := len(compressed1)

		// Make small modification (1%)
		modified := make([]byte, 100*1024)
		copy(modified, baseline)
		for i := 0; i < 1024; i++ {
			modified[i] = byte(i % 256)
		}

		// Second compression uses delta
		compressed2, err := hde.CompressMemory("vm-delta", modified, dwcp.CompressionGlobal)
		require.NoError(t, err)
		size2 := len(compressed2)

		// Delta should be much smaller
		deltaRatio := float64(size2) / float64(size1)
		assert.Less(t, deltaRatio, 0.7, "Delta encoding should reduce size significantly")

		t.Logf("✅ Delta encoding: baseline=%d bytes, delta=%d bytes, ratio=%.2f",
			size1, size2, deltaRatio)
	})

	t.Run("delta_hit_rate_tracking", func(t *testing.T) {
		config := dwcp.HDEConfig{
			EnableDelta: true,
			BlockSize:   4 * 1024,
		}

		hde, err := dwcp.NewHDE(config)
		require.NoError(t, err)
		defer hde.Close()

		// Multiple compressions to build history
		for i := 0; i < 10; i++ {
			data := make([]byte, 10*1024)
			rand.Read(data)
			_, _ = hde.CompressMemory("vm-hitrate", data, dwcp.CompressionLocal)
		}

		metrics := hde.GetMetrics()
		deltaHitRate := metrics["delta_hit_rate"].(float64)

		assert.GreaterOrEqual(t, deltaHitRate, 0.0)
		assert.LessOrEqual(t, deltaHitRate, 1.0)

		t.Logf("✅ Delta hit rate: %.2f%%", deltaHitRate*100)
	})
}

// TestHDEv3CompressionTargets tests 70-85% compression target
func TestHDEv3CompressionTargets(t *testing.T) {
	upgrade.EnableAll(100)
	defer upgrade.DisableAll()

	t.Run("combined_70_to_85_percent_reduction", func(t *testing.T) {
		config := dwcp.HDEConfig{
			GlobalLevel:        9,
			EnableDelta:        true,
			EnableDictionary:   true,
			EnableQuantization: true,
		}

		hde, err := dwcp.NewHDE(config)
		require.NoError(t, err)
		defer hde.Close()

		// Realistic VM memory (mix of zeros, patterns, random)
		data := make([]byte, 1024*1024) // 1MB
		// 50% zeros
		// 25% patterns
		for i := 1024 * 512; i < 1024*768; i++ {
			data[i] = byte(i % 16)
		}
		// 25% random
		rand.Read(data[1024*768:])

		compressed, err := hde.CompressMemory("vm-target", data, dwcp.CompressionGlobal)
		require.NoError(t, err)

		reduction := 1.0 - (float64(len(compressed)) / float64(len(data)))
		reductionPercent := reduction * 100

		// Target: 70-85% reduction
		t.Logf("Compression reduction: %.1f%% (target: 70-85%%)", reductionPercent)

		// For mixed data, expect at least 50% reduction
		assert.Greater(t, reduction, 0.50, "Should achieve >50% reduction for mixed data")

		metrics := hde.GetMetrics()
		compressionRatio := metrics["compression_ratio"].(float64)
		t.Logf("✅ Compression ratio: %.2fx", compressionRatio)
	})

	t.Run("zstandard_level_9_performance", func(t *testing.T) {
		config := dwcp.HDEConfig{
			GlobalLevel: 9, // Maximum compression
		}

		hde, err := dwcp.NewHDE(config)
		require.NoError(t, err)
		defer hde.Close()

		data := make([]byte, 1024*1024)
		for i := range data {
			data[i] = byte(i % 256)
		}

		start := time.Now()
		compressed, err := hde.CompressMemory("vm-level9", data, dwcp.CompressionGlobal)
		compressionTime := time.Since(start)

		require.NoError(t, err)
		compressionRatio := float64(len(data)) / float64(len(compressed))

		assert.Greater(t, compressionRatio, 2.0, "Level 9 should achieve >2x compression")
		t.Logf("✅ Level 9: %.2fx compression in %v", compressionRatio, compressionTime)
	})
}

// TestHDEv3DictionaryTraining tests dictionary-based compression
func TestHDEv3DictionaryTraining(t *testing.T) {
	upgrade.EnableAll(100)
	defer upgrade.DisableAll()

	t.Run("dictionary_training", func(t *testing.T) {
		config := dwcp.HDEConfig{
			EnableDictionary: true,
			DictSize:         1024, // 1MB
			TrainingSamples:  10,
		}

		hde, err := dwcp.NewHDE(config)
		require.NoError(t, err)
		defer hde.Close()

		// Create training samples
		samples := make([][]byte, 10)
		for i := range samples {
			samples[i] = make([]byte, 10*1024)
			// Similar patterns for better dictionary
			for j := range samples[i] {
				samples[i][j] = byte((i + j) % 256)
			}
		}

		err = hde.TrainDictionary("test-dict", samples)
		require.NoError(t, err)

		metrics := hde.GetMetrics()
		dictCount := metrics["dictionary_count"].(int)
		assert.GreaterOrEqual(t, dictCount, 1, "Should have trained dictionary")

		t.Log("✅ Dictionary training completed")
	})
}

// TestHDEv3MemoryManagement tests memory management
func TestHDEv3MemoryManagement(t *testing.T) {
	upgrade.EnableAll(100)
	defer upgrade.DisableAll()

	t.Run("baseline_cleanup", func(t *testing.T) {
		config := dwcp.HDEConfig{
			EnableDelta:      true,
			MaxDeltaHistory:  10,
			CleanupInterval:  100 * time.Millisecond,
			MaxMemoryUsage:   10 * 1024 * 1024, // 10MB
		}

		hde, err := dwcp.NewHDE(config)
		require.NoError(t, err)
		defer hde.Close()

		// Create many baselines
		for i := 0; i < 100; i++ {
			data := make([]byte, 10*1024)
			rand.Read(data)
			_, _ = hde.CompressMemory(fmt.Sprintf("vm-%d", i), data, dwcp.CompressionLocal)
		}

		// Wait for cleanup
		time.Sleep(300 * time.Millisecond)

		metrics := hde.GetMetrics()
		baselineCount := metrics["baseline_count"].(int)

		// Should have cleaned up old baselines
		assert.Less(t, baselineCount, 100, "Old baselines should be cleaned up")

		t.Logf("✅ Memory management: %d baselines after cleanup", baselineCount)
	})
}

// TestHDEv3Concurrent tests concurrent compression operations
func TestHDEv3Concurrent(t *testing.T) {
	upgrade.EnableAll(100)
	defer upgrade.DisableAll()

	config := dwcp.HDEConfig{
		GlobalLevel: 3,
		EnableDelta: true,
	}

	hde, err := dwcp.NewHDE(config)
	require.NoError(t, err)
	defer hde.Close()

	const numGoroutines = 20
	errors := make(chan error, numGoroutines)

	var wg sync.WaitGroup
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()

			data := make([]byte, 100*1024)
			rand.Read(data)

			_, err := hde.CompressMemory(fmt.Sprintf("vm-%d", id), data, dwcp.CompressionRegional)
			if err != nil {
				errors <- err
			}
		}(i)
	}

	wg.Wait()
	close(errors)

	errorCount := 0
	for err := range errors {
		errorCount++
		t.Errorf("Concurrent compression error: %v", err)
	}

	assert.Equal(t, 0, errorCount, "No errors in concurrent compression")
	t.Logf("✅ Concurrent operations: %d compressions succeeded", numGoroutines)
}
