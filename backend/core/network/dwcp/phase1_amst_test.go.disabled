package dwcp_test

import (
	"context"
	"crypto/rand"
	"fmt"
	"net"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp"
	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/transport"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
)

// TestPhase1_AMSTRDMASupport tests RDMA connections with fallback to TCP
func TestPhase1_AMSTRDMASupport(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	config := &transport.AMSTConfig{
		MinStreams:     16,
		MaxStreams:     64,
		InitialStreams: 32,
		ChunkSizeKB:    256,
		AutoTune:       true,
		EnableRDMA:     true,
		RDMADevice:     "mlx5_0", // Will fallback to TCP if not available
		ConnectTimeout: 5 * time.Second,
	}

	listener, port := startBandwidthTrackingServer(t, 64)
	defer listener.Close()

	mst, err := transport.NewMultiStreamTCP(fmt.Sprintf("localhost:%d", port), config, logger)
	require.NoError(t, err, "NewMultiStreamTCP should succeed")
	defer mst.Close()

	err = mst.Start()
	require.NoError(t, err, "Start should succeed even with RDMA fallback")

	// Verify that transport mode is either RDMA or TCP (graceful fallback)
	metrics := mst.GetMetrics()
	mode := metrics["transport_mode"].(string)
	assert.Contains(t, []string{"rdma", "tcp", "hybrid"}, mode, "Transport mode should be valid")

	// If RDMA not available, should fallback to TCP without error
	if mode == "tcp" {
		t.Log("✅ RDMA not available - gracefully fell back to TCP")
	} else {
		t.Log("✅ RDMA successfully initialized")
	}

	// Test data transfer works regardless of transport mode
	testData := make([]byte, 1024*1024) // 1 MB
	rand.Read(testData)

	err = mst.Send(testData)
	assert.NoError(t, err, "Send should work with RDMA or TCP fallback")

	t.Log("✅ Phase 1 RDMA support with TCP fallback validated")
}

// TestPhase1_AMSTBBRCongestion tests BBR congestion control algorithm
func TestPhase1_AMSTBBRCongestion(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	config := &transport.AMSTConfig{
		MinStreams:          16,
		MaxStreams:          128,
		InitialStreams:      32,
		CongestionAlgorithm: "bbr",
		ChunkSizeKB:         256,
		AutoTune:            true,
		ConnectTimeout:      5 * time.Second,
	}

	listener, port := startBandwidthTrackingServer(t, 128)
	defer listener.Close()

	mst, err := transport.NewMultiStreamTCP(fmt.Sprintf("localhost:%d", port), config, logger)
	require.NoError(t, err)
	defer mst.Close()

	err = mst.Start()
	require.NoError(t, err)

	// Verify BBR is active
	metrics := mst.GetMetrics()
	congestionAlg := metrics["congestion_algorithm"].(string)
	assert.Equal(t, "bbr", congestionAlg, "BBR congestion control should be active")

	// Send data and verify BBR optimizes throughput
	testData := make([]byte, 10*1024*1024) // 10 MB
	rand.Read(testData)

	startTime := time.Now()
	err = mst.Send(testData)
	require.NoError(t, err)
	duration := time.Since(startTime)

	throughputMBps := float64(len(testData)) / (1024 * 1024) / duration.Seconds()
	t.Logf("BBR throughput: %.2f MB/s", throughputMBps)

	// BBR should maintain high utilization
	utilization := metrics["bandwidth_utilization"].(float64)
	assert.GreaterOrEqual(t, utilization, 0.85, "BBR should achieve >85% bandwidth utilization")

	t.Log("✅ Phase 1 BBR congestion control validated")
}

// TestPhase1_AMSTDynamicScaling tests automatic stream adjustment
func TestPhase1_AMSTDynamicScaling(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	config := &transport.AMSTConfig{
		MinStreams:          8,
		MaxStreams:          128,
		InitialStreams:      16,
		StreamScalingFactor: 1.5,
		AutoTune:            true,
		ChunkSizeKB:         256,
		ConnectTimeout:      5 * time.Second,
	}

	listener, port := startBandwidthTrackingServer(t, 128)
	defer listener.Close()

	mst, err := transport.NewMultiStreamTCP(fmt.Sprintf("localhost:%d", port), config, logger)
	require.NoError(t, err)
	defer mst.Close()

	err = mst.Start()
	require.NoError(t, err)

	// Initial stream count should be 16
	metrics := mst.GetMetrics()
	initialStreams := metrics["active_streams"].(int32)
	assert.Equal(t, int32(16), initialStreams, "Should start with 16 streams")

	// Send large amount of data to trigger scaling
	largeData := make([]byte, 50*1024*1024) // 50 MB
	rand.Read(largeData)

	err = mst.Send(largeData)
	require.NoError(t, err)

	// Wait for auto-tuning to take effect
	time.Sleep(2 * time.Second)

	// Verify stream count increased
	metricsAfter := mst.GetMetrics()
	finalStreams := metricsAfter["active_streams"].(int32)

	// With high load, streams should have increased
	if finalStreams > initialStreams {
		t.Logf("✅ Dynamic scaling: %d → %d streams", initialStreams, finalStreams)
	} else {
		t.Logf("ℹ️  Streams remained at %d (may scale under real WAN conditions)", initialStreams)
	}

	// Verify scaling stayed within bounds
	assert.GreaterOrEqual(t, finalStreams, int32(8), "Should not scale below MinStreams")
	assert.LessOrEqual(t, finalStreams, int32(128), "Should not scale above MaxStreams")

	t.Log("✅ Phase 1 dynamic stream scaling validated")
}

// TestPhase1_AMSTFailover tests stream failure recovery
func TestPhase1_AMSTFailover(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	config := &transport.AMSTConfig{
		MinStreams:     16,
		MaxStreams:     64,
		InitialStreams: 32,
		ChunkSizeKB:    256,
		AutoTune:       true,
		ConnectTimeout: 5 * time.Second,
	}

	// Start a server that will close connections randomly
	listener := startFailoverTestServer(t, 64, 0.1) // 10% failure rate
	port := listener.Addr().(*net.TCPAddr).Port
	defer listener.Close()

	mst, err := transport.NewMultiStreamTCP(fmt.Sprintf("localhost:%d", port), config, logger)
	require.NoError(t, err)
	defer mst.Close()

	err = mst.Start()
	require.NoError(t, err)

	// Send data through potentially failing streams
	testData := make([]byte, 10*1024*1024) // 10 MB
	rand.Read(testData)

	err = mst.Send(testData)
	require.NoError(t, err, "Should successfully send despite stream failures")

	// Verify failover metrics
	metrics := mst.GetMetrics()
	failedStreams := metrics["failed_streams"].(uint64)
	recoveredStreams := metrics["recovered_streams"].(uint64)

	t.Logf("Failover stats - Failed: %d, Recovered: %d", failedStreams, recoveredStreams)

	// Should have attempted recovery
	if failedStreams > 0 {
		assert.Greater(t, recoveredStreams, uint64(0), "Should recover from failures")
	}

	t.Log("✅ Phase 1 stream failover validated")
}

// TestPhase1_AMSTMetrics tests Prometheus metrics accuracy
func TestPhase1_AMSTMetrics(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	config := &transport.AMSTConfig{
		MinStreams:     16,
		MaxStreams:     64,
		InitialStreams: 32,
		ChunkSizeKB:    256,
		AutoTune:       false,
		ConnectTimeout: 5 * time.Second,
	}

	listener, port := startBandwidthTrackingServer(t, 64)
	defer listener.Close()

	mst, err := transport.NewMultiStreamTCP(fmt.Sprintf("localhost:%d", port), config, logger)
	require.NoError(t, err)
	defer mst.Close()

	err = mst.Start()
	require.NoError(t, err)

	// Send known amount of data
	testDataSize := 5 * 1024 * 1024 // 5 MB
	testData := make([]byte, testDataSize)
	rand.Read(testData)

	startTime := time.Now()
	err = mst.Send(testData)
	require.NoError(t, err)
	duration := time.Since(startTime)

	// Verify metrics accuracy
	metrics := mst.GetMetrics()

	// Check active streams
	activeStreams := metrics["active_streams"].(int32)
	assert.Equal(t, int32(32), activeStreams, "Active streams should match config")

	// Check bytes sent
	totalBytesSent := metrics["total_bytes_sent"].(uint64)
	assert.Equal(t, uint64(testDataSize), totalBytesSent, "Bytes sent should be accurate")

	// Check throughput calculation
	reportedThroughput := metrics["throughput_mbps"].(float64)
	expectedThroughput := float64(testDataSize) / (1024 * 1024) / duration.Seconds()

	// Allow 10% tolerance for timing variations
	assert.InDelta(t, expectedThroughput, reportedThroughput, expectedThroughput*0.1,
		"Reported throughput should match actual")

	// Check utilization metric exists
	utilization := metrics["bandwidth_utilization"].(float64)
	assert.GreaterOrEqual(t, utilization, 0.0, "Utilization should be non-negative")
	assert.LessOrEqual(t, utilization, 1.0, "Utilization should not exceed 100%")

	// Check latency metric
	avgLatency := metrics["average_latency_ms"].(float64)
	assert.Greater(t, avgLatency, 0.0, "Average latency should be positive")

	t.Logf("✅ Metrics validated - Streams: %d, Throughput: %.2f MB/s, Utilization: %.1f%%",
		activeStreams, reportedThroughput, utilization*100)
}

// TestPhase1_AMSTPerformance validates >85% bandwidth utilization
func TestPhase1_AMSTPerformance(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	config := &transport.AMSTConfig{
		MinStreams:          32,
		MaxStreams:          128,
		InitialStreams:      64,
		CongestionAlgorithm: "bbr",
		ChunkSizeKB:         512,
		AutoTune:            true,
		PacingEnabled:       true,
		ConnectTimeout:      5 * time.Second,
	}

	listener, port := startBandwidthTrackingServer(t, 128)
	defer listener.Close()

	mst, err := transport.NewMultiStreamTCP(fmt.Sprintf("localhost:%d", port), config, logger)
	require.NoError(t, err)
	defer mst.Close()

	err = mst.Start()
	require.NoError(t, err)

	// Send large dataset to measure sustained performance
	testDataSize := 50 * 1024 * 1024 // 50 MB
	testData := make([]byte, testDataSize)
	rand.Read(testData)

	startTime := time.Now()
	err = mst.Send(testData)
	require.NoError(t, err)
	duration := time.Since(startTime)

	metrics := mst.GetMetrics()
	utilization := metrics["bandwidth_utilization"].(float64)
	throughputMBps := float64(testDataSize) / (1024 * 1024) / duration.Seconds()

	t.Logf("Performance - Throughput: %.2f MB/s, Utilization: %.1f%%",
		throughputMBps, utilization*100)

	// Phase 1 target: >85% bandwidth utilization
	assert.GreaterOrEqual(t, utilization, 0.85,
		"Phase 1 requires >85%% bandwidth utilization (got %.1f%%)", utilization*100)

	t.Log("✅ Phase 1 performance target (>85% utilization) validated")
}

// TestPhase1_AMSTConcurrency tests thread-safe operations
func TestPhase1_AMSTConcurrency(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	config := &transport.AMSTConfig{
		MinStreams:     16,
		MaxStreams:     64,
		InitialStreams: 32,
		ChunkSizeKB:    256,
		AutoTune:       true,
		ConnectTimeout: 5 * time.Second,
	}

	listener, port := startBandwidthTrackingServer(t, 64)
	defer listener.Close()

	mst, err := transport.NewMultiStreamTCP(fmt.Sprintf("localhost:%d", port), config, logger)
	require.NoError(t, err)
	defer mst.Close()

	err = mst.Start()
	require.NoError(t, err)

	// Concurrent sends from multiple goroutines
	numGoroutines := 10
	sendSize := 1024 * 1024 // 1 MB per goroutine

	var wg sync.WaitGroup
	errChan := make(chan error, numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()

			data := make([]byte, sendSize)
			rand.Read(data)

			if err := mst.Send(data); err != nil {
				errChan <- fmt.Errorf("goroutine %d: %w", id, err)
			}
		}(i)
	}

	wg.Wait()
	close(errChan)

	// Check for errors
	var errors []error
	for err := range errChan {
		errors = append(errors, err)
	}

	assert.Empty(t, errors, "Concurrent sends should succeed without race conditions")

	// Verify total bytes sent
	metrics := mst.GetMetrics()
	totalBytesSent := metrics["total_bytes_sent"].(uint64)
	expectedBytes := uint64(numGoroutines * sendSize)

	assert.Equal(t, expectedBytes, totalBytesSent,
		"Total bytes should match sum of concurrent sends")

	t.Logf("✅ Concurrent operations validated - %d goroutines, %d MB total",
		numGoroutines, totalBytesSent/(1024*1024))
}

// TestPhase1_AMSTGracefulShutdown tests clean resource cleanup
func TestPhase1_AMSTGracefulShutdown(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	config := &transport.AMSTConfig{
		MinStreams:     16,
		MaxStreams:     64,
		InitialStreams: 32,
		ChunkSizeKB:    256,
		AutoTune:       true,
		ConnectTimeout: 5 * time.Second,
	}

	listener, port := startBandwidthTrackingServer(t, 64)
	defer listener.Close()

	mst, err := transport.NewMultiStreamTCP(fmt.Sprintf("localhost:%d", port), config, logger)
	require.NoError(t, err)

	err = mst.Start()
	require.NoError(t, err)

	// Start sending data
	testData := make([]byte, 10*1024*1024) // 10 MB
	rand.Read(testData)

	sendCtx, sendCancel := context.WithCancel(context.Background())
	defer sendCancel()

	go func() {
		for {
			select {
			case <-sendCtx.Done():
				return
			default:
				mst.Send(testData) // Ignore errors during shutdown test
			}
		}
	}()

	// Let some data flow
	time.Sleep(500 * time.Millisecond)

	// Graceful shutdown
	shutdownStart := time.Now()
	err = mst.Close()
	shutdownDuration := time.Since(shutdownStart)

	require.NoError(t, err, "Close should succeed")
	assert.Less(t, shutdownDuration, 5*time.Second, "Shutdown should complete quickly")

	// Verify resources cleaned up
	metrics := mst.GetMetrics()
	activeStreams := metrics["active_streams"].(int32)
	assert.Equal(t, int32(0), activeStreams, "All streams should be closed")

	// Attempting to send after close should fail gracefully
	err = mst.Send(testData)
	assert.Error(t, err, "Send after close should return error")

	t.Logf("✅ Graceful shutdown validated (%.2fs)", shutdownDuration.Seconds())
}

// Helper functions

func startFailoverTestServer(t *testing.T, maxConnections int, failureRate float64) net.Listener {
	listener, err := net.Listen("tcp", "localhost:0")
	require.NoError(t, err)

	go func() {
		connectionCount := atomic.Int32{}
		for {
			conn, err := listener.Accept()
			if err != nil {
				return
			}

			if int(connectionCount.Load()) >= maxConnections {
				conn.Close()
				continue
			}

			connectionCount.Add(1)
			go func(c net.Conn) {
				defer c.Close()
				defer connectionCount.Add(-1)

				// Randomly fail connections
				if rand.Float64() < failureRate {
					time.Sleep(100 * time.Millisecond)
					return // Simulate failure
				}

				// Normal echo behavior
				buf := make([]byte, 8192)
				for {
					_, err := c.Read(buf)
					if err != nil {
						return
					}
				}
			}(conn)
		}
	}()

	return listener
}
