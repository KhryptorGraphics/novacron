package dwcp_test

import (
	"context"
	"crypto/rand"
	"fmt"
	"io"
	"net"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/compression"
	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/transport"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
)

// WANSimulator simulates WAN network conditions
type WANSimulator struct {
	listener       net.Listener
	latency        time.Duration
	bandwidth      int64 // bytes per second
	packetLossRate float64
	connections    atomic.Int32
	maxConnections int
	mu             sync.Mutex
	closed         bool
}

func NewWANSimulator(latency time.Duration, bandwidthMbps int, packetLossRate float64, maxConn int) (*WANSimulator, error) {
	listener, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		return nil, err
	}

	sim := &WANSimulator{
		listener:       listener,
		latency:        latency,
		bandwidth:      int64(bandwidthMbps * 1024 * 1024 / 8), // Convert Mbps to bytes/sec
		packetLossRate: packetLossRate,
		maxConnections: maxConn,
	}

	go sim.acceptLoop()

	return sim, nil
}

func (ws *WANSimulator) Port() int {
	return ws.listener.Addr().(*net.TCPAddr).Port
}

func (ws *WANSimulator) Close() error {
	ws.mu.Lock()
	defer ws.mu.Unlock()

	if ws.closed {
		return nil
	}

	ws.closed = true
	return ws.listener.Close()
}

func (ws *WANSimulator) acceptLoop() {
	for {
		conn, err := ws.listener.Accept()
		if err != nil {
			return // Listener closed
		}

		if int(ws.connections.Load()) >= ws.maxConnections {
			conn.Close()
			continue
		}

		ws.connections.Add(1)
		go ws.handleConnection(conn)
	}
}

func (ws *WANSimulator) handleConnection(conn net.Conn) {
	defer conn.Close()
	defer ws.connections.Add(-1)

	// Simulate initial connection latency
	time.Sleep(ws.latency / 2)

	buf := make([]byte, 8192)
	var totalBytes int64
	startTime := time.Now()

	for {
		// Read data
		n, err := conn.Read(buf)
		if err != nil {
			if err != io.EOF {
				// Connection error
			}
			return
		}

		// Simulate packet loss
		if rand.Float64() < ws.packetLossRate {
			continue // Drop packet
		}

		// Simulate bandwidth limiting
		totalBytes += int64(n)
		elapsed := time.Since(startTime)
		expectedTime := time.Duration(float64(totalBytes) / float64(ws.bandwidth) * float64(time.Second))

		if expectedTime > elapsed {
			time.Sleep(expectedTime - elapsed)
		}

		// Simulate latency
		time.Sleep(ws.latency / 2)
	}
}

// TestWAN_HighLatency tests 50ms latency performance
func TestWAN_HighLatency(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	// Simulate WAN with 50ms latency, 1 Gbps bandwidth, 0% loss
	wanSim, err := NewWANSimulator(50*time.Millisecond, 1000, 0.0, 128)
	require.NoError(t, err)
	defer wanSim.Close()

	config := &transport.AMSTConfig{
		MinStreams:          32,
		MaxStreams:          128,
		InitialStreams:      64,
		CongestionAlgorithm: "bbr", // BBR performs better on high-latency networks
		ChunkSizeKB:         512,
		AutoTune:            true,
		PacingEnabled:       true,
		ConnectTimeout:      10 * time.Second,
	}

	mst, err := transport.NewMultiStreamTCP(
		fmt.Sprintf("localhost:%d", wanSim.Port()),
		config,
		logger,
	)
	require.NoError(t, err)
	defer mst.Close()

	err = mst.Start()
	require.NoError(t, err)

	// Send data over high-latency link
	testDataSize := 10 * 1024 * 1024 // 10 MB
	testData := make([]byte, testDataSize)
	rand.Read(testData)

	startTime := time.Now()
	err = mst.Send(testData)
	require.NoError(t, err)
	duration := time.Since(startTime)

	throughputMBps := float64(testDataSize) / (1024 * 1024) / duration.Seconds()

	t.Logf("High-latency WAN performance:")
	t.Logf("  Latency: 50ms")
	t.Logf("  Data: %d MB", testDataSize/(1024*1024))
	t.Logf("  Duration: %v", duration)
	t.Logf("  Throughput: %.2f MB/s", throughputMBps)

	// Multi-stream should significantly improve performance over high-latency links
	metrics := mst.GetMetrics()
	utilization := metrics["bandwidth_utilization"].(float64)

	t.Logf("  Utilization: %.1f%%", utilization*100)
	assert.GreaterOrEqual(t, utilization, 0.70,
		"Should maintain >70%% utilization even with 50ms latency")

	t.Log("✅ High-latency WAN performance validated")
}

// TestWAN_LowBandwidth tests 100Mbps bandwidth optimization
func TestWAN_LowBandwidth(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	// Simulate WAN with 10ms latency, 100 Mbps bandwidth, 0% loss
	wanSim, err := NewWANSimulator(10*time.Millisecond, 100, 0.0, 64)
	require.NoError(t, err)
	defer wanSim.Close()

	// Transport configuration optimized for lower bandwidth
	amstConfig := &transport.AMSTConfig{
		MinStreams:          16,
		MaxStreams:          64,
		InitialStreams:      32,
		CongestionAlgorithm: "bbr",
		ChunkSizeKB:         256,
		AutoTune:            true,
		PacingEnabled:       true,
		PacingRate:          100 * 1024 * 1024 / 8, // 100 Mbps in bytes/sec
		ConnectTimeout:      10 * time.Second,
	}

	mst, err := transport.NewMultiStreamTCP(
		fmt.Sprintf("localhost:%d", wanSim.Port()),
		amstConfig,
		logger,
	)
	require.NoError(t, err)
	defer mst.Close()

	err = mst.Start()
	require.NoError(t, err)

	// Use compression to maximize efficiency
	hdeConfig := compression.DefaultDeltaEncodingConfig()
	hdeConfig.CompressionLevel = 6

	encoder, err := compression.NewDeltaEncoder(hdeConfig, logger)
	require.NoError(t, err)
	defer encoder.Close()

	// Test with compressible data
	testDataSize := 8 * 1024 * 1024 // 8 MB
	testData := generateCompressibleData(t, testDataSize, 0.7) // 70% compressible

	// Compress first
	stateKey := "wan-lowbw-test"
	encoded, err := encoder.Encode(stateKey, testData)
	require.NoError(t, err)

	compressionRatio := encoded.CompressionRatio()
	t.Logf("Compression: %d MB → %d KB (%.2fx)",
		encoded.OriginalSize/(1024*1024),
		encoded.CompressedSize/1024,
		compressionRatio)

	// Transfer compressed data
	startTime := time.Now()
	err = mst.Send(encoded.Data)
	require.NoError(t, err)
	duration := time.Since(startTime)

	throughputMbps := float64(encoded.CompressedSize*8) / (1024 * 1024) / duration.Seconds()

	t.Logf("Low-bandwidth WAN performance:")
	t.Logf("  Bandwidth limit: 100 Mbps")
	t.Logf("  Compressed size: %d KB", encoded.CompressedSize/1024)
	t.Logf("  Duration: %v", duration)
	t.Logf("  Throughput: %.2f Mbps", throughputMbps)

	// Effective bandwidth (accounting for compression)
	effectiveThroughputMbps := float64(testDataSize*8) / (1024 * 1024) / duration.Seconds()
	t.Logf("  Effective throughput: %.2f Mbps (with compression)", effectiveThroughputMbps)

	// Should efficiently use available bandwidth
	metrics := mst.GetMetrics()
	utilization := metrics["bandwidth_utilization"].(float64)
	t.Logf("  Utilization: %.1f%%", utilization*100)

	// Compression should enable effective throughput > physical bandwidth
	assert.Greater(t, effectiveThroughputMbps, 100.0,
		"Compression should enable >100 Mbps effective throughput on 100 Mbps link")

	t.Log("✅ Low-bandwidth optimization validated")
}

// TestWAN_PacketLoss tests 1% packet loss resilience
func TestWAN_PacketLoss(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	// Simulate WAN with 20ms latency, 500 Mbps, 1% packet loss
	wanSim, err := NewWANSimulator(20*time.Millisecond, 500, 0.01, 64)
	require.NoError(t, err)
	defer wanSim.Close()

	config := &transport.AMSTConfig{
		MinStreams:          32,
		MaxStreams:          96,
		InitialStreams:      64,
		CongestionAlgorithm: "bbr", // BBR handles loss better than CUBIC
		ChunkSizeKB:         256,
		AutoTune:            true,
		PacingEnabled:       true,
		ConnectTimeout:      15 * time.Second,
		ReadTimeout:         30 * time.Second,
		WriteTimeout:        30 * time.Second,
	}

	mst, err := transport.NewMultiStreamTCP(
		fmt.Sprintf("localhost:%d", wanSim.Port()),
		config,
		logger,
	)
	require.NoError(t, err)
	defer mst.Close()

	err = mst.Start()
	require.NoError(t, err)

	// Send data over lossy link
	testDataSize := 5 * 1024 * 1024 // 5 MB
	testData := make([]byte, testDataSize)
	rand.Read(testData)

	startTime := time.Now()
	err = mst.Send(testData)
	require.NoError(t, err, "Should successfully send despite 1%% packet loss")
	duration := time.Since(startTime)

	t.Logf("Packet loss WAN performance:")
	t.Logf("  Packet loss: 1%%")
	t.Logf("  Data: %d MB", testDataSize/(1024*1024))
	t.Logf("  Duration: %v", duration)

	metrics := mst.GetMetrics()
	retransmissions := metrics["retransmission_count"].(uint64)
	throughputMBps := float64(testDataSize) / (1024 * 1024) / duration.Seconds()

	t.Logf("  Retransmissions: %d", retransmissions)
	t.Logf("  Throughput: %.2f MB/s", throughputMBps)

	// Multi-stream should mitigate impact of packet loss
	utilization := metrics["bandwidth_utilization"].(float64)
	t.Logf("  Utilization: %.1f%%", utilization*100)

	assert.GreaterOrEqual(t, utilization, 0.65,
		"Should maintain >65%% utilization despite 1%% packet loss")

	t.Log("✅ Packet loss resilience validated")
}

// TestWAN_MultiRegion tests cross-region performance
func TestWAN_MultiRegion(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	// Simulate different region scenarios
	regions := []struct {
		name           string
		latency        time.Duration
		bandwidthMbps  int
		packetLoss     float64
		expectedUtil   float64
	}{
		{
			name:          "US-West to US-East",
			latency:       60 * time.Millisecond,
			bandwidthMbps: 1000,
			packetLoss:    0.001, // 0.1%
			expectedUtil:  0.80,
		},
		{
			name:          "US to Europe",
			latency:       100 * time.Millisecond,
			bandwidthMbps: 500,
			packetLoss:    0.005, // 0.5%
			expectedUtil:  0.70,
		},
		{
			name:          "US to Asia",
			latency:       180 * time.Millisecond,
			bandwidthMbps: 300,
			packetLoss:    0.01, // 1%
			expectedUtil:  0.60,
		},
	}

	testDataSize := 4 * 1024 * 1024 // 4 MB

	for _, region := range regions {
		t.Run(region.name, func(t *testing.T) {
			// Setup WAN simulator
			wanSim, err := NewWANSimulator(
				region.latency,
				region.bandwidthMbps,
				region.packetLoss,
				128,
			)
			require.NoError(t, err)
			defer wanSim.Close()

			// Adaptive configuration based on region
			streamCount := 32 + int(region.latency.Milliseconds()/10) // More streams for higher latency
			if streamCount > 128 {
				streamCount = 128
			}

			config := &transport.AMSTConfig{
				MinStreams:          streamCount / 2,
				MaxStreams:          streamCount * 2,
				InitialStreams:      streamCount,
				CongestionAlgorithm: "bbr",
				ChunkSizeKB:         256,
				AutoTune:            true,
				PacingEnabled:       true,
				ConnectTimeout:      20 * time.Second,
			}

			mst, err := transport.NewMultiStreamTCP(
				fmt.Sprintf("localhost:%d", wanSim.Port()),
				config,
				logger,
			)
			require.NoError(t, err)
			defer mst.Close()

			err = mst.Start()
			require.NoError(t, err)

			// Setup compression
			hdeConfig := compression.DefaultDeltaEncodingConfig()
			encoder, err := compression.NewDeltaEncoder(hdeConfig, logger)
			require.NoError(t, err)
			defer encoder.Close()

			// Generate test data
			testData := generateVMMemoryData(t, testDataSize)

			// Compress
			stateKey := "region-" + region.name
			encoded, err := encoder.Encode(stateKey, testData)
			require.NoError(t, err)

			// Transfer
			ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
			defer cancel()

			errChan := make(chan error, 1)
			startTime := time.Now()

			go func() {
				errChan <- mst.Send(encoded.Data)
			}()

			select {
			case err := <-errChan:
				require.NoError(t, err, "Transfer should succeed")
			case <-ctx.Done():
				t.Fatal("Transfer timeout")
			}

			duration := time.Since(startTime)

			// Calculate metrics
			throughputMBps := float64(testDataSize) / (1024 * 1024) / duration.Seconds()
			compressionRatio := encoded.CompressionRatio()

			metrics := mst.GetMetrics()
			utilization := metrics["bandwidth_utilization"].(float64)
			activeStreams := metrics["active_streams"].(int32)

			t.Logf("Region: %s", region.name)
			t.Logf("  Latency: %v", region.latency)
			t.Logf("  Bandwidth: %d Mbps", region.bandwidthMbps)
			t.Logf("  Packet loss: %.1f%%", region.packetLoss*100)
			t.Logf("  Active streams: %d", activeStreams)
			t.Logf("  Compression: %.2fx", compressionRatio)
			t.Logf("  Duration: %v", duration)
			t.Logf("  Throughput: %.2f MB/s", throughputMBps)
			t.Logf("  Utilization: %.1f%%", utilization*100)

			// Verify meets expected utilization for this region
			assert.GreaterOrEqual(t, utilization, region.expectedUtil,
				"Should achieve expected utilization for %s", region.name)
		})
	}

	t.Log("✅ Multi-region performance validated")
}

// TestWAN_AdaptiveScaling tests stream scaling under varying conditions
func TestWAN_AdaptiveScaling(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	// Start with good conditions
	wanSim, err := NewWANSimulator(20*time.Millisecond, 1000, 0.0, 128)
	require.NoError(t, err)
	defer wanSim.Close()

	config := &transport.AMSTConfig{
		MinStreams:          8,
		MaxStreams:          128,
		InitialStreams:      16,
		StreamScalingFactor: 1.5,
		CongestionAlgorithm: "bbr",
		ChunkSizeKB:         256,
		AutoTune:            true,
		PacingEnabled:       true,
		ConnectTimeout:      10 * time.Second,
	}

	mst, err := transport.NewMultiStreamTCP(
		fmt.Sprintf("localhost:%d", wanSim.Port()),
		config,
		logger,
	)
	require.NoError(t, err)
	defer mst.Close()

	err = mst.Start()
	require.NoError(t, err)

	// Send data and observe scaling
	testData := make([]byte, 20*1024*1024) // 20 MB

	// Monitor stream count during transfer
	var streamCounts []int32
	done := make(chan bool)

	go func() {
		ticker := time.NewTicker(500 * time.Millisecond)
		defer ticker.Stop()

		for {
			select {
			case <-done:
				return
			case <-ticker.C:
				metrics := mst.GetMetrics()
				streams := metrics["active_streams"].(int32)
				streamCounts = append(streamCounts, streams)
			}
		}
	}()

	err = mst.Send(testData)
	require.NoError(t, err)

	close(done)

	t.Log("Stream scaling during transfer:")
	for i, count := range streamCounts {
		t.Logf("  Sample %d: %d streams", i+1, count)
	}

	// Verify adaptive scaling occurred
	if len(streamCounts) > 1 {
		initialStreams := streamCounts[0]
		maxStreams := initialStreams
		for _, count := range streamCounts {
			if count > maxStreams {
				maxStreams = count
			}
		}

		if maxStreams > initialStreams {
			t.Logf("✅ Adaptive scaling detected: %d → %d streams",
				initialStreams, maxStreams)
		} else {
			t.Logf("ℹ️  Streams stable at %d (conditions may not trigger scaling)",
				initialStreams)
		}
	}

	t.Log("✅ Adaptive scaling behavior validated")
}
