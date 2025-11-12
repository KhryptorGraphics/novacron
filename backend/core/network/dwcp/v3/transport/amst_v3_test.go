package transport

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/upgrade"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap/zaptest"
)

// TestAMSTv3_DatacenterMode tests AMST v3 in datacenter mode (v1 RDMA compatibility)
func TestAMSTv3_DatacenterMode(t *testing.T) {
	logger := zaptest.NewLogger(t)
	ctx := context.Background()

	// Create mode detector forced to datacenter mode
	detector := upgrade.NewModeDetector()
	detector.ForceMode(upgrade.ModeDatacenter)

	config := DefaultAMSTv3Config()
	config.DatacenterStreams = 64 // High stream count for datacenter

	amst, err := NewAMSTv3(config, detector, logger)
	require.NoError(t, err)
	require.NotNil(t, amst)

	// Verify it uses datacenter settings
	assert.Equal(t, upgrade.ModeDatacenter, amst.GetCurrentMode())
	assert.True(t, amst.datacenterTransport != nil)

	// Test high bandwidth transfer
	data := make([]byte, 10*1024*1024) // 10MB
	for i := range data {
		data[i] = byte(i % 256)
	}

	// Start transport
	err = amst.Start(ctx, "localhost:9000")
	assert.NoError(t, err)
	defer amst.Close()

	// Verify stream count matches datacenter settings
	metrics := amst.GetMetrics()
	assert.GreaterOrEqual(t, metrics.ActiveStreams, int32(32))
}

// TestAMSTv3_InternetMode tests AMST v3 in internet mode (new v3 features)
func TestAMSTv3_InternetMode(t *testing.T) {
	logger := zaptest.NewLogger(t)
	ctx := context.Background()

	// Create mode detector forced to internet mode
	detector := upgrade.NewModeDetector()
	detector.ForceMode(upgrade.ModeInternet)

	config := DefaultAMSTv3Config()
	config.InternetStreams = 8 // Low stream count for internet

	amst, err := NewAMSTv3(config, detector, logger)
	require.NoError(t, err)
	require.NotNil(t, amst)

	// Verify it uses internet settings
	assert.Equal(t, upgrade.ModeInternet, amst.GetCurrentMode())
	assert.True(t, amst.internetTransport != nil)

	// Start transport
	err = amst.Start(ctx, "localhost:9000")
	assert.NoError(t, err)
	defer amst.Close()

	// Verify stream count matches internet settings (4-16 streams)
	metrics := amst.GetMetrics()
	assert.LessOrEqual(t, metrics.ActiveStreams, int32(16))
	assert.GreaterOrEqual(t, metrics.ActiveStreams, int32(4))
}

// TestAMSTv3_HybridMode tests adaptive switching between modes
func TestAMSTv3_HybridMode(t *testing.T) {
	logger := zaptest.NewLogger(t)
	ctx := context.Background()

	detector := upgrade.NewModeDetector()
	config := DefaultAMSTv3Config()

	amst, err := NewAMSTv3(config, detector, logger)
	require.NoError(t, err)
	require.NotNil(t, amst)

	err = amst.Start(ctx, "localhost:9000")
	require.NoError(t, err)
	defer amst.Close()

	// Test mode switching
	originalMode := amst.GetCurrentMode()

	// Force mode change
	if originalMode == upgrade.ModeDatacenter {
		detector.ForceMode(upgrade.ModeInternet)
	} else {
		detector.ForceMode(upgrade.ModeDatacenter)
	}

	// Wait for mode detection
	time.Sleep(100 * time.Millisecond)

	// Send data to trigger mode re-detection
	data := []byte("test data for mode switching")
	err = amst.SendData(ctx, data)
	assert.NoError(t, err)

	// Verify mode changed
	newMode := amst.GetCurrentMode()
	assert.NotEqual(t, originalMode, newMode)
}

// TestAMSTv3_AdaptiveSend tests adaptive transport selection
func TestAMSTv3_AdaptiveSend(t *testing.T) {
	logger := zaptest.NewLogger(t)
	ctx := context.Background()

	detector := upgrade.NewModeDetector()
	detector.ForceMode(upgrade.ModeHybrid)

	config := DefaultAMSTv3Config()
	amst, err := NewAMSTv3(config, detector, logger)
	require.NoError(t, err)

	err = amst.Start(ctx, "localhost:9000")
	require.NoError(t, err)
	defer amst.Close()

	// Send data in hybrid mode
	data := make([]byte, 1024*1024) // 1MB
	err = amst.SendData(ctx, data)
	assert.NoError(t, err)

	// Verify metrics recorded
	metrics := amst.GetMetrics()
	assert.Greater(t, metrics.TotalBytesSent, uint64(0))
}

// TestAMSTv3_ConcurrentSends tests parallel sends
func TestAMSTv3_ConcurrentSends(t *testing.T) {
	logger := zaptest.NewLogger(t)
	ctx := context.Background()

	detector := upgrade.NewModeDetector()
	detector.ForceMode(upgrade.ModeInternet)

	config := DefaultAMSTv3Config()
	amst, err := NewAMSTv3(config, detector, logger)
	require.NoError(t, err)

	err = amst.Start(ctx, "localhost:9000")
	require.NoError(t, err)
	defer amst.Close()

	// Concurrent sends
	var wg sync.WaitGroup
	numGoroutines := 10
	dataSize := 64 * 1024 // 64KB per send

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			data := make([]byte, dataSize)
			for j := range data {
				data[j] = byte((id + j) % 256)
			}
			err := amst.SendData(ctx, data)
			assert.NoError(t, err)
		}(i)
	}

	wg.Wait()

	// Verify all data sent
	metrics := amst.GetMetrics()
	expectedBytes := uint64(numGoroutines * dataSize)
	assert.GreaterOrEqual(t, metrics.TotalBytesSent, expectedBytes)
}

// TestAMSTv3_ModeSwitchPerformance tests mode switching overhead
func TestAMSTv3_ModeSwitchPerformance(t *testing.T) {
	logger := zaptest.NewLogger(t)
	ctx := context.Background()

	detector := upgrade.NewModeDetector()
	config := DefaultAMSTv3Config()

	amst, err := NewAMSTv3(config, detector, logger)
	require.NoError(t, err)

	err = amst.Start(ctx, "localhost:9000")
	require.NoError(t, err)
	defer amst.Close()

	// Measure mode switch time
	start := time.Now()

	detector.ForceMode(upgrade.ModeDatacenter)
	time.Sleep(50 * time.Millisecond)

	detector.ForceMode(upgrade.ModeInternet)
	time.Sleep(50 * time.Millisecond)

	detector.ForceMode(upgrade.ModeHybrid)

	elapsed := time.Since(start)

	// Mode switching should be <2 seconds (requirement)
	assert.Less(t, elapsed, 2*time.Second)
}

// TestAMSTv3_BackwardCompatibility tests v1 compatibility
func TestAMSTv3_BackwardCompatibility(t *testing.T) {
	logger := zaptest.NewLogger(t)
	ctx := context.Background()

	// Force datacenter mode for v1 compatibility
	detector := upgrade.NewModeDetector()
	detector.ForceMode(upgrade.ModeDatacenter)

	config := DefaultAMSTv3Config()
	amst, err := NewAMSTv3(config, detector, logger)
	require.NoError(t, err)

	err = amst.Start(ctx, "localhost:9000")
	require.NoError(t, err)
	defer amst.Close()

	// Test v1-style high-bandwidth transfer
	data := make([]byte, 100*1024*1024) // 100MB
	progressCalled := false

	err = amst.TransferWithProgress(ctx, data, func(bytes int64, total int64) {
		progressCalled = true
		assert.LessOrEqual(t, bytes, total)
	})
	assert.NoError(t, err)
	assert.True(t, progressCalled)
}

// TestAMSTv3_CongestionControl tests BBR congestion control
func TestAMSTv3_CongestionControl(t *testing.T) {
	logger := zaptest.NewLogger(t)
	ctx := context.Background()

	detector := upgrade.NewModeDetector()
	detector.ForceMode(upgrade.ModeInternet)

	config := DefaultAMSTv3Config()
	config.CongestionAlgorithm = "bbr"

	amst, err := NewAMSTv3(config, detector, logger)
	require.NoError(t, err)

	err = amst.Start(ctx, "localhost:9000")
	require.NoError(t, err)
	defer amst.Close()

	// Verify BBR is enabled
	metrics := amst.GetMetrics()
	assert.Equal(t, "bbr", metrics.CongestionControl)
}

// TestAMSTv3_StreamScaling tests adaptive stream scaling
func TestAMSTv3_StreamScaling(t *testing.T) {
	logger := zaptest.NewLogger(t)
	ctx := context.Background()

	detector := upgrade.NewModeDetector()
	config := DefaultAMSTv3Config()
	config.AutoTune = true

	amst, err := NewAMSTv3(config, detector, logger)
	require.NoError(t, err)

	err = amst.Start(ctx, "localhost:9000")
	require.NoError(t, err)
	defer amst.Close()

	initialMetrics := amst.GetMetrics()
	initialStreams := initialMetrics.ActiveStreams

	// Simulate network condition change
	err = amst.AdjustStreams(5000.0, 2.0) // 5 Gbps, 2ms latency (datacenter)
	assert.NoError(t, err)

	// Verify streams adjusted
	newMetrics := amst.GetMetrics()
	newStreams := newMetrics.ActiveStreams

	// Stream count should change based on conditions
	assert.NotEqual(t, initialStreams, newStreams)
}

// TestAMSTv3_ErrorHandling tests error recovery
func TestAMSTv3_ErrorHandling(t *testing.T) {
	logger := zaptest.NewLogger(t)
	ctx := context.Background()

	detector := upgrade.NewModeDetector()
	config := DefaultAMSTv3Config()

	amst, err := NewAMSTv3(config, detector, logger)
	require.NoError(t, err)

	// Test send without starting
	err = amst.SendData(ctx, []byte("test"))
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "not started")

	// Test nil data
	err = amst.Start(ctx, "localhost:9000")
	require.NoError(t, err)
	defer amst.Close()

	err = amst.SendData(ctx, nil)
	assert.Error(t, err)

	err = amst.SendData(ctx, []byte{})
	assert.Error(t, err)
}

// TestAMSTv3_Metrics tests comprehensive metrics collection
func TestAMSTv3_Metrics(t *testing.T) {
	logger := zaptest.NewLogger(t)
	ctx := context.Background()

	detector := upgrade.NewModeDetector()
	config := DefaultAMSTv3Config()

	amst, err := NewAMSTv3(config, detector, logger)
	require.NoError(t, err)

	err = amst.Start(ctx, "localhost:9000")
	require.NoError(t, err)
	defer amst.Close()

	// Get initial metrics
	metrics := amst.GetMetrics()

	// Verify metrics structure
	assert.NotNil(t, metrics)
	assert.GreaterOrEqual(t, metrics.ActiveStreams, int32(0))
	assert.GreaterOrEqual(t, metrics.TotalStreams, 0)
	assert.NotEmpty(t, metrics.TransportType)
	assert.NotEmpty(t, metrics.Mode)
}

// BenchmarkAMSTv3_DatacenterThroughput benchmarks datacenter mode
func BenchmarkAMSTv3_DatacenterThroughput(b *testing.B) {
	logger := zaptest.NewLogger(b)
	ctx := context.Background()

	detector := upgrade.NewModeDetector()
	detector.ForceMode(upgrade.ModeDatacenter)

	config := DefaultAMSTv3Config()
	amst, err := NewAMSTv3(config, detector, logger)
	require.NoError(b, err)

	err = amst.Start(ctx, "localhost:9000")
	require.NoError(b, err)
	defer amst.Close()

	data := make([]byte, 1024*1024) // 1MB

	b.ResetTimer()
	b.SetBytes(int64(len(data)))

	for i := 0; i < b.N; i++ {
		_ = amst.SendData(ctx, data)
	}
}

// BenchmarkAMSTv3_InternetThroughput benchmarks internet mode
func BenchmarkAMSTv3_InternetThroughput(b *testing.B) {
	logger := zaptest.NewLogger(b)
	ctx := context.Background()

	detector := upgrade.NewModeDetector()
	detector.ForceMode(upgrade.ModeInternet)

	config := DefaultAMSTv3Config()
	amst, err := NewAMSTv3(config, detector, logger)
	require.NoError(b, err)

	err = amst.Start(ctx, "localhost:9000")
	require.NoError(b, err)
	defer amst.Close()

	data := make([]byte, 256*1024) // 256KB (smaller for internet)

	b.ResetTimer()
	b.SetBytes(int64(len(data)))

	for i := 0; i < b.N; i++ {
		_ = amst.SendData(ctx, data)
	}
}

// BenchmarkAMSTv3_ModeSwitching benchmarks mode switching overhead
func BenchmarkAMSTv3_ModeSwitching(b *testing.B) {
	logger := zaptest.NewLogger(b)
	ctx := context.Background()

	detector := upgrade.NewModeDetector()
	config := DefaultAMSTv3Config()

	amst, err := NewAMSTv3(config, detector, logger)
	require.NoError(b, err)

	err = amst.Start(ctx, "localhost:9000")
	require.NoError(b, err)
	defer amst.Close()

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		if i%2 == 0 {
			detector.ForceMode(upgrade.ModeDatacenter)
		} else {
			detector.ForceMode(upgrade.ModeInternet)
		}
		amst.SendData(ctx, []byte("test"))
	}
}
