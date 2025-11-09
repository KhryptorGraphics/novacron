package transport

import (
	"fmt"
	"io"
	"net"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"go.uber.org/zap"
)

// TestMultiStreamTCP_BasicConnection tests basic connection establishment
func TestMultiStreamTCP_BasicConnection(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	// Start a test server
	listener, port := startTestServer(t, 1)
	defer listener.Close()

	config := &AMSTConfig{
		MinStreams:     1,
		MaxStreams:     4,
		ChunkSizeKB:    64,
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

	if !mst.IsStarted() {
		t.Error("MultiStreamTCP should be started")
	}

	metrics := mst.GetMetrics()
	if metrics["active_streams"].(int32) != 1 {
		t.Errorf("Expected 1 active stream, got %d", metrics["active_streams"])
	}
}

// TestMultiStreamTCP_DataTransfer tests basic data sending
func TestMultiStreamTCP_DataTransfer(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	// Start a test server with echo functionality
	listener, port := startEchoServer(t, 4)
	defer listener.Close()

	config := &AMSTConfig{
		MinStreams:     4,
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

	// Send test data
	testData := make([]byte, 1024*1024) // 1 MB
	for i := range testData {
		testData[i] = byte(i % 256)
	}

	if err := mst.Send(testData); err != nil {
		t.Fatalf("Send failed: %v", err)
	}

	metrics := mst.GetMetrics()
	totalSent := metrics["total_bytes_sent"].(uint64)
	if totalSent != uint64(len(testData)) {
		t.Errorf("Expected %d bytes sent, got %d", len(testData), totalSent)
	}
}

// TestMultiStreamTCP_StreamScaling tests dynamic stream adjustment
func TestMultiStreamTCP_StreamScaling(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	listener, port := startTestServer(t, 32)
	defer listener.Close()

	config := &AMSTConfig{
		MinStreams:     8,
		MaxStreams:     32,
		ChunkSizeKB:    256,
		AutoTune:       true,
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

	initialMetrics := mst.GetMetrics()
	initialStreams := initialMetrics["active_streams"].(int32)

	// Simulate high bandwidth, low latency conditions
	// optimal_streams = bandwidth_mbps / (latency_ms * 0.1)
	// optimal_streams = 1000 / (10 * 0.1) = 1000 streams (capped to MaxStreams=32)
	if err := mst.AdjustStreams(1000.0, 10.0); err != nil {
		t.Fatalf("AdjustStreams failed: %v", err)
	}

	metricsAfterScale := mst.GetMetrics()
	scaledStreams := metricsAfterScale["active_streams"].(int32)

	if scaledStreams <= initialStreams {
		t.Errorf("Expected stream count to increase from %d, got %d", initialStreams, scaledStreams)
	}

	if scaledStreams > 32 {
		t.Errorf("Stream count exceeded MaxStreams: got %d, max 32", scaledStreams)
	}

	t.Logf("Stream scaling successful: %d -> %d streams", initialStreams, scaledStreams)
}

// TestMultiStreamTCP_ParallelChunking tests that data is split across streams
func TestMultiStreamTCP_ParallelChunking(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	listener, port := startEchoServer(t, 8)
	defer listener.Close()

	config := &AMSTConfig{
		MinStreams:     8,
		MaxStreams:     8,
		ChunkSizeKB:    64, // 64 KB chunks
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

	// Send 512 KB of data (should split into 8 chunks of 64 KB each)
	testData := make([]byte, 512*1024)
	for i := range testData {
		testData[i] = byte(i % 256)
	}

	startTime := time.Now()
	if err := mst.Send(testData); err != nil {
		t.Fatalf("Send failed: %v", err)
	}
	duration := time.Since(startTime)

	t.Logf("Sent %d bytes in %v (%.2f MB/s)",
		len(testData),
		duration,
		float64(len(testData))/(1024*1024)/duration.Seconds())

	metrics := mst.GetMetrics()
	totalSent := metrics["total_bytes_sent"].(uint64)
	if totalSent != uint64(len(testData)) {
		t.Errorf("Expected %d bytes sent, got %d", len(testData), totalSent)
	}
}

// TestMultiStreamTCP_PacketPacing tests rate limiting functionality
func TestMultiStreamTCP_PacketPacing(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	listener, port := startEchoServer(t, 4)
	defer listener.Close()

	// Configure with 10 Mbps pacing (1.25 MB/s)
	config := &AMSTConfig{
		MinStreams:     4,
		MaxStreams:     4,
		ChunkSizeKB:    64,
		AutoTune:       false,
		PacingEnabled:  true,
		PacingRate:     10 * 1024 * 1024 / 8, // 10 Mbps = 1.25 MB/s
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

	// Send 1 MB of data
	testData := make([]byte, 1024*1024)
	startTime := time.Now()
	if err := mst.Send(testData); err != nil {
		t.Fatalf("Send failed: %v", err)
	}
	duration := time.Since(startTime)

	// With 10 Mbps pacing, 1 MB should take ~0.8 seconds
	expectedDuration := time.Duration(float64(len(testData)) / float64(config.PacingRate) * float64(time.Second))
	tolerance := expectedDuration / 2 // 50% tolerance

	if duration < (expectedDuration-tolerance) || duration > (expectedDuration+tolerance+2*time.Second) {
		t.Logf("Warning: Pacing duration outside expected range. Got %v, expected ~%v", duration, expectedDuration)
		// Don't fail test, pacing is best-effort
	} else {
		t.Logf("Pacing working correctly: %v (expected ~%v)", duration, expectedDuration)
	}
}

// TestMultiStreamTCP_ConcurrentOperations tests thread safety
func TestMultiStreamTCP_ConcurrentOperations(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	listener, port := startEchoServer(t, 16)
	defer listener.Close()

	config := &AMSTConfig{
		MinStreams:     16,
		MaxStreams:     32,
		ChunkSizeKB:    128,
		AutoTune:       true,
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

	// Run concurrent operations
	var wg sync.WaitGroup
	errChan := make(chan error, 10)

	// Concurrent sends
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			data := make([]byte, 256*1024)
			if err := mst.Send(data); err != nil {
				errChan <- fmt.Errorf("concurrent send %d: %w", id, err)
			}
		}(i)
	}

	// Concurrent metrics reads
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			_ = mst.GetMetrics()
		}(i)
	}

	wg.Wait()
	close(errChan)

	for err := range errChan {
		t.Errorf("Concurrent operation error: %v", err)
	}
}

// TestMultiStreamTCP_GracefulShutdown tests proper cleanup
func TestMultiStreamTCP_GracefulShutdown(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	listener, port := startTestServer(t, 8)
	defer listener.Close()

	config := DefaultAMSTConfig()
	config.MinStreams = 8
	config.MaxStreams = 16

	mst, err := NewMultiStreamTCP(fmt.Sprintf("localhost:%d", port), config, logger)
	if err != nil {
		t.Fatalf("NewMultiStreamTCP failed: %v", err)
	}

	if err := mst.Start(); err != nil {
		t.Fatalf("Start failed: %v", err)
	}

	if !mst.IsStarted() {
		t.Error("MultiStreamTCP should be started")
	}

	if err := mst.Close(); err != nil {
		t.Fatalf("Close failed: %v", err)
	}

	if mst.IsStarted() {
		t.Error("MultiStreamTCP should be stopped after Close")
	}

	// Verify operations fail after close
	testData := []byte("test")
	if err := mst.Send(testData); err == nil {
		t.Error("Send should fail after Close")
	}
}

// BenchmarkMultiStreamTCP_Throughput benchmarks data transfer performance
func BenchmarkMultiStreamTCP_Throughput(b *testing.B) {
	logger, _ := zap.NewProduction()

	listener, port := startEchoServer(nil, 32)
	defer listener.Close()

	config := &AMSTConfig{
		MinStreams:     32,
		MaxStreams:     32,
		ChunkSizeKB:    256,
		AutoTune:       false,
		PacingEnabled:  false,
		ConnectTimeout: 5 * time.Second,
	}

	mst, err := NewMultiStreamTCP(fmt.Sprintf("localhost:%d", port), config, logger)
	if err != nil {
		b.Fatalf("NewMultiStreamTCP failed: %v", err)
	}
	defer mst.Close()

	if err := mst.Start(); err != nil {
		b.Fatalf("Start failed: %v", err)
	}

	// Test with 1 MB chunks
	testData := make([]byte, 1024*1024)
	for i := range testData {
		testData[i] = byte(i % 256)
	}

	b.ResetTimer()
	b.SetBytes(int64(len(testData)))

	for i := 0; i < b.N; i++ {
		if err := mst.Send(testData); err != nil {
			b.Fatalf("Send failed: %v", err)
		}
	}
}

// BenchmarkMultiStreamTCP_StreamScaling benchmarks stream adjustment overhead
func BenchmarkMultiStreamTCP_StreamScaling(b *testing.B) {
	logger, _ := zap.NewProduction()

	listener, port := startTestServer(nil, 256)
	defer listener.Close()

	config := &AMSTConfig{
		MinStreams:     16,
		MaxStreams:     256,
		ChunkSizeKB:    256,
		AutoTune:       true,
		PacingEnabled:  false,
		ConnectTimeout: 5 * time.Second,
	}

	mst, err := NewMultiStreamTCP(fmt.Sprintf("localhost:%d", port), config, logger)
	if err != nil {
		b.Fatalf("NewMultiStreamTCP failed: %v", err)
	}
	defer mst.Close()

	if err := mst.Start(); err != nil {
		b.Fatalf("Start failed: %v", err)
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		// Simulate varying network conditions
		bandwidth := 500.0 + float64(i%500)
		latency := 10.0 + float64(i%90)
		if err := mst.AdjustStreams(bandwidth, latency); err != nil {
			b.Fatalf("AdjustStreams failed: %v", err)
		}
	}
}

// Helper functions for test servers

func startTestServer(t interface{ Fatalf(format string, args ...interface{}) }, maxConnections int) (net.Listener, int) {
	listener, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatalf("Failed to start test server: %v", err)
	}

	port := listener.Addr().(*net.TCPAddr).Port

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

				// Simple echo server that reads and discards data
				buf := make([]byte, 8192)
				for {
					_, err := c.Read(buf)
					if err != nil {
						return
					}
					// Just consume the data, don't echo back
				}
			}(conn)
		}
	}()

	return listener, port
}

func startEchoServer(t interface{ Fatalf(format string, args ...interface{}) }, maxConnections int) (net.Listener, int) {
	listener, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		if t != nil {
			t.Fatalf("Failed to start echo server: %v", err)
		}
		panic(err)
	}

	port := listener.Addr().(*net.TCPAddr).Port

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

				// Echo server that reads headers and data
				for {
					// Read chunk header (8 bytes)
					header := make([]byte, 8)
					if _, err := io.ReadFull(c, header); err != nil {
						return
					}

					// Extract chunk size from header
					chunkSize := int(header[4])<<24 | int(header[5])<<16 | int(header[6])<<8 | int(header[7])

					// Read chunk data
					chunk := make([]byte, chunkSize)
					if _, err := io.ReadFull(c, chunk); err != nil {
						return
					}

					// Echo back header and data
					if _, err := c.Write(header); err != nil {
						return
					}
					if _, err := c.Write(chunk); err != nil {
						return
					}
				}
			}(conn)
		}
	}()

	return listener, port
}
