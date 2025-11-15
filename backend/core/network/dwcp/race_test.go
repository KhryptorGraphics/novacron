package dwcp_test

import (
	"sync"
	"testing"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap/zaptest"
)

// TestRaceConditionFix verifies the P0 race condition fix in collectMetrics
// Run with: go test -race -run TestRaceConditionFix
func TestRaceConditionFix(t *testing.T) {
	config := dwcp.DefaultConfig()
	config.Enabled = true
	logger := zaptest.NewLogger(t)

	manager, err := dwcp.NewManager(config, logger)
	require.NoError(t, err)
	require.NotNil(t, manager)

	err = manager.Start()
	require.NoError(t, err)
	defer func() {
		err := manager.Stop()
		assert.NoError(t, err)
	}()

	// Simulate high concurrency scenario
	var wg sync.WaitGroup
	iterations := 200
	duration := 3 * time.Second
	stopTime := time.Now().Add(duration)

	// Concurrent readers
	for i := 0; i < iterations; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for time.Now().Before(stopTime) {
				metrics := manager.GetMetrics()
				assert.NotNil(t, metrics)
				assert.Equal(t, dwcp.DWCPVersion, metrics.Version)
				time.Sleep(time.Millisecond)
			}
		}(i)
	}

	// Concurrent state checkers
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for time.Now().Before(stopTime) {
				_ = manager.IsEnabled()
				_ = manager.IsStarted()
				time.Sleep(2 * time.Millisecond)
			}
		}(i)
	}

	// Concurrent config readers
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for time.Now().Before(stopTime) {
				config := manager.GetConfig()
				assert.NotNil(t, config)
				time.Sleep(2 * time.Millisecond)
			}
		}(i)
	}

	wg.Wait()
}

// TestMetricsCollectionStress tests metrics collection under stress
// This test verifies that collectMetrics() doesn't cause data races
func TestMetricsCollectionStress(t *testing.T) {
	config := dwcp.DefaultConfig()
	config.Enabled = true
	logger := zaptest.NewLogger(t)

	manager, err := dwcp.NewManager(config, logger)
	require.NoError(t, err)

	err = manager.Start()
	require.NoError(t, err)
	defer manager.Stop()

	// Let metrics collection run for several cycles
	time.Sleep(12 * time.Second)

	// Verify metrics are being collected correctly
	metrics := manager.GetMetrics()
	assert.NotNil(t, metrics)
	assert.Equal(t, dwcp.DWCPVersion, metrics.Version)
	assert.True(t, metrics.Enabled)
}

// BenchmarkGetMetrics benchmarks metrics retrieval performance
func BenchmarkGetMetrics(b *testing.B) {
	config := dwcp.DefaultConfig()
	config.Enabled = true
	logger := zaptest.NewLogger(b)

	manager, err := dwcp.NewManager(config, logger)
	require.NoError(b, err)

	err = manager.Start()
	require.NoError(b, err)
	defer manager.Stop()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = manager.GetMetrics()
	}
}

// BenchmarkGetMetricsParallel benchmarks parallel metrics retrieval
func BenchmarkGetMetricsParallel(b *testing.B) {
	config := dwcp.DefaultConfig()
	config.Enabled = true
	logger := zaptest.NewLogger(b)

	manager, err := dwcp.NewManager(config, logger)
	require.NoError(b, err)

	err = manager.Start()
	require.NoError(b, err)
	defer manager.Stop()

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			_ = manager.GetMetrics()
		}
	})
}

// BenchmarkConcurrentOperations benchmarks mixed concurrent operations
func BenchmarkConcurrentOperations(b *testing.B) {
	config := dwcp.DefaultConfig()
	config.Enabled = true
	logger := zaptest.NewLogger(b)

	manager, err := dwcp.NewManager(config, logger)
	require.NoError(b, err)

	err = manager.Start()
	require.NoError(b, err)
	defer manager.Stop()

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			// Mix of operations that compete for locks
			_ = manager.GetMetrics()
			_ = manager.GetConfig()
			_ = manager.IsEnabled()
			_ = manager.IsStarted()
		}
	})
}
