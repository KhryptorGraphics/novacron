package transport

import (
	"fmt"
	"runtime"
	"testing"
	"time"

	"go.uber.org/zap"
	"golang.org/x/sys/unix"
)

// isLinux returns true if running on Linux
func isLinux() bool {
	return runtime.GOOS == "linux"
}

// TestBBRCongestionControl tests BBR congestion control setting
func TestBBRCongestionControl(t *testing.T) {
	// Skip if not on Linux
	if !isLinux() {
		t.Skip("BBR congestion control only available on Linux")
	}

	logger, _ := zap.NewDevelopment()

	listener, port := startTestServer(t, 4)
	defer listener.Close()

	config := &AMSTConfig{
		MinStreams:          4,
		MaxStreams:          16,
		ChunkSizeKB:         256,
		AutoTune:            false,
		PacingEnabled:       false,
		ConnectTimeout:      5 * time.Second,
		CongestionAlgorithm: "bbr",
	}

	mst, err := NewMultiStreamTCP(fmt.Sprintf("localhost:%d", port), config, logger)
	if err != nil {
		t.Fatalf("NewMultiStreamTCP failed: %v", err)
	}
	defer mst.Close()

	if err := mst.Start(); err != nil {
		t.Fatalf("Start failed: %v", err)
	}

	// Verify BBR is set on at least one stream
	mst.mu.RLock()
	if len(mst.streams) == 0 {
		mst.mu.RUnlock()
		t.Fatal("No streams created")
	}

	stream := mst.streams[0]
	mst.mu.RUnlock()

	// Try to verify congestion control (best effort)
	var actualAlg string
	stream.rawConn.Control(func(fd uintptr) {
		alg, err := unix.GetsockoptString(int(fd), unix.IPPROTO_TCP, unix.TCP_CONGESTION)
		if err == nil {
			actualAlg = alg
		}
	})

	if actualAlg == "bbr" {
		t.Logf("BBR successfully set on stream")
	} else {
		t.Logf("BBR not available, using default: %s", actualAlg)
	}
}

// TestHealthMonitoring tests stream health monitoring
func TestHealthMonitoring(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	listener, port := startTestServer(t, 8)
	defer listener.Close()

	config := &AMSTConfig{
		MinStreams:     8,
		MaxStreams:     16,
		ChunkSizeKB:    128,
		AutoTune:       false,
		PacingEnabled:  false,
		ConnectTimeout: 5 * time.Second,
	}

	mst, err := NewMultiStreamTCP(fmt.Sprintf("localhost:%d", port), config, logger)
	if err != nil {
		t.Fatalf("NewMultiStreamTCP failed: %v", err)
	}
	defer mst.Close()

	if err := mst.Start(); err != nil {
		t.Fatalf("Start failed: %v", err)
	}

	// Wait for health check to run
	time.Sleep(1 * time.Second)

	// Check stream health
	health := mst.GetStreamHealth()
	if len(health) == 0 {
		t.Error("No stream health information available")
	}

	healthyCount := 0
	for _, sh := range health {
		if sh.Healthy {
			healthyCount++
		}
	}

	t.Logf("Healthy streams: %d/%d", healthyCount, len(health))

	if healthyCount == 0 {
		t.Error("No healthy streams found")
	}
}

// TestMetricsCollection tests Prometheus metrics
func TestMetricsCollection(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	listener, port := startEchoServer(t, 8)
	defer listener.Close()

	config := &AMSTConfig{
		MinStreams:     8,
		MaxStreams:     16,
		ChunkSizeKB:    128,
		AutoTune:       false,
		PacingEnabled:  false,
		ConnectTimeout: 5 * time.Second,
	}

	mst, err := NewMultiStreamTCP(fmt.Sprintf("localhost:%d", port), config, logger)
	if err != nil {
		t.Fatalf("NewMultiStreamTCP failed: %v", err)
	}
	defer mst.Close()

	if err := mst.Start(); err != nil {
		t.Fatalf("Start failed: %v", err)
	}

	// Send some data to generate metrics
	testData := make([]byte, 1024*1024) // 1 MB
	if err := mst.Send(testData); err != nil {
		t.Fatalf("Send failed: %v", err)
	}

	// Check metrics
	if mst.metrics == nil {
		t.Fatal("Metrics collector not initialized")
	}

	metrics := mst.metrics.GetMetrics()
	if metrics.TotalBytesSent != uint64(len(testData)) {
		t.Errorf("Expected %d bytes sent, got %d", len(testData), metrics.TotalBytesSent)
	}

	if !metrics.Healthy {
		t.Error("System should be healthy")
	}

	t.Logf("Metrics: %+v", metrics)
}

// TestGracefulShutdown tests graceful shutdown with in-flight requests
func TestGracefulShutdown(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	listener, port := startEchoServer(t, 8)
	defer listener.Close()

	config := &AMSTConfig{
		MinStreams:     8,
		MaxStreams:     16,
		ChunkSizeKB:    128,
		AutoTune:       false,
		PacingEnabled:  false,
		ConnectTimeout: 5 * time.Second,
	}

	mst, err := NewMultiStreamTCP(fmt.Sprintf("localhost:%d", port), config, logger)
	if err != nil {
		t.Fatalf("NewMultiStreamTCP failed: %v", err)
	}

	if err := mst.Start(); err != nil {
		t.Fatalf("Start failed: %v", err)
	}

	// Start some background sends
	done := make(chan bool)
	go func() {
		for i := 0; i < 5; i++ {
			testData := make([]byte, 512*1024)
			mst.Send(testData)
			time.Sleep(100 * time.Millisecond)
		}
		done <- true
	}()

	// Wait a bit then close
	time.Sleep(200 * time.Millisecond)

	startClose := time.Now()
	if err := mst.Close(); err != nil {
		t.Fatalf("Close failed: %v", err)
	}
	closeDuration := time.Since(startClose)

	// Wait for background goroutine
	<-done

	t.Logf("Graceful shutdown took %v", closeDuration)

	if closeDuration > 35*time.Second {
		t.Error("Shutdown took longer than expected timeout")
	}
}

// TestRDMATransport tests RDMA transport with TCP fallback
func TestRDMATransport(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	listener, port := startTestServer(t, 8)
	defer listener.Close()

	config := &TransportConfig{
		RemoteAddr:          fmt.Sprintf("localhost:%d", port),
		ConnectTimeout:      5 * time.Second,
		MinStreams:          8,
		MaxStreams:          16,
		ChunkSizeKB:         256,
		AutoTune:            false,
		PacingEnabled:       false,
		EnableRDMA:          true,
		RDMADevice:          "mlx5_0",
		RDMAPort:            1,
		CongestionAlgorithm: "bbr",
	}

	rdma, err := NewRDMATransport(config, logger)
	if err != nil {
		t.Fatalf("NewRDMATransport failed: %v", err)
	}
	defer rdma.Close()

	if err := rdma.Start(); err != nil {
		t.Fatalf("Start failed: %v", err)
	}

	// Check if RDMA is available (likely not in test environment)
	if !rdma.SupportsRDMA() {
		t.Log("RDMA not available, using TCP fallback")
	}

	// Test basic send
	testData := make([]byte, 1024*1024)
	if err := rdma.Send(testData); err != nil {
		t.Fatalf("Send failed: %v", err)
	}

	// Check metrics
	metrics := rdma.GetMetrics()
	t.Logf("Transport type: %s", metrics.TransportType)
	t.Logf("Bytes sent: %d", metrics.TotalBytesSent)

	// Should be using TCP fallback
	if metrics.TransportType != "tcp" {
		t.Errorf("Expected TCP fallback, got %s", metrics.TransportType)
	}
}

// TestHealthCheck tests the health check functionality
func TestHealthCheck(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	listener, port := startTestServer(t, 16)
	defer listener.Close()

	config := &AMSTConfig{
		MinStreams:     16,
		MaxStreams:     32,
		ChunkSizeKB:    256,
		AutoTune:       false,
		PacingEnabled:  false,
		ConnectTimeout: 5 * time.Second,
	}

	mst, err := NewMultiStreamTCP(fmt.Sprintf("localhost:%d", port), config, logger)
	if err != nil {
		t.Fatalf("NewMultiStreamTCP failed: %v", err)
	}
	defer mst.Close()

	if err := mst.Start(); err != nil {
		t.Fatalf("Start failed: %v", err)
	}

	// Health check should pass
	if err := mst.HealthCheck(); err != nil {
		t.Errorf("Health check failed: %v", err)
	}

	// Close and verify health check fails
	mst.Close()

	if err := mst.HealthCheck(); err == nil {
		t.Error("Health check should fail after close")
	}
}

// TestStreamReconnection tests automatic stream reconnection
func TestStreamReconnection(t *testing.T) {
	t.Skip("Manual test - requires controlled connection drops")

	logger, _ := zap.NewDevelopment()

	listener, port := startTestServer(t, 8)
	defer listener.Close()

	config := &AMSTConfig{
		MinStreams:     8,
		MaxStreams:     16,
		ChunkSizeKB:    128,
		AutoTune:       false,
		PacingEnabled:  false,
		ConnectTimeout: 5 * time.Second,
	}

	mst, err := NewMultiStreamTCP(fmt.Sprintf("localhost:%d", port), config, logger)
	if err != nil {
		t.Fatalf("NewMultiStreamTCP failed: %v", err)
	}
	defer mst.Close()

	if err := mst.Start(); err != nil {
		t.Fatalf("Start failed: %v", err)
	}

	initialStreams := mst.activeStreams.Load()
	t.Logf("Initial active streams: %d", initialStreams)

	// TODO: Simulate connection drops and verify reconnection
	// This would require a more sophisticated test server

	time.Sleep(15 * time.Second) // Wait for health check cycles

	finalStreams := mst.activeStreams.Load()
	t.Logf("Final active streams: %d", finalStreams)
}

// BenchmarkBBRThroughput benchmarks throughput with BBR
func BenchmarkBBRThroughput(b *testing.B) {
	logger, _ := zap.NewProduction()

	listener, port := startEchoServer(nil, 32)
	defer listener.Close()

	config := &AMSTConfig{
		MinStreams:          32,
		MaxStreams:          32,
		ChunkSizeKB:         256,
		AutoTune:            false,
		PacingEnabled:       false,
		ConnectTimeout:      5 * time.Second,
		CongestionAlgorithm: "bbr",
	}

	mst, err := NewMultiStreamTCP(fmt.Sprintf("localhost:%d", port), config, logger)
	if err != nil {
		b.Fatalf("NewMultiStreamTCP failed: %v", err)
	}
	defer mst.Close()

	if err := mst.Start(); err != nil {
		b.Fatalf("Start failed: %v", err)
	}

	testData := make([]byte, 1024*1024)

	b.ResetTimer()
	b.SetBytes(int64(len(testData)))

	for i := 0; i < b.N; i++ {
		if err := mst.Send(testData); err != nil {
			b.Fatalf("Send failed: %v", err)
		}
	}

	b.StopTimer()

	metrics := mst.metrics.GetMetrics()
	b.Logf("Total bytes sent: %d", metrics.TotalBytesSent)
	b.Logf("Throughput: %.2f Mbps", metrics.ThroughputMbps)
}
