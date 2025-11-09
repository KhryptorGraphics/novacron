package rdma

import (
	"testing"
	"time"

	"go.uber.org/zap"
)

// TestCheckAvailability tests RDMA availability detection
func TestCheckAvailability(t *testing.T) {
	available := CheckAvailability()
	t.Logf("RDMA available: %v", available)

	if !available {
		t.Skip("RDMA not available on this system, skipping hardware tests")
	}
}

// TestGetDeviceList tests device enumeration
func TestGetDeviceList(t *testing.T) {
	if !CheckAvailability() {
		t.Skip("RDMA not available")
	}

	devices, err := GetDeviceList()
	if err != nil {
		t.Fatalf("Failed to get device list: %v", err)
	}

	if len(devices) == 0 {
		t.Skip("No RDMA devices found")
	}

	t.Logf("Found %d RDMA device(s)", len(devices))
	for i, dev := range devices {
		t.Logf("Device %d: %s (GUID: %s)", i, dev.Name, dev.GUID)
		t.Logf("  Supports RC: %v", dev.SupportsRC)
		t.Logf("  Supports UD: %v", dev.SupportsUD)
		t.Logf("  Supports RDMA Write: %v", dev.SupportsRDMAWrite)
		t.Logf("  Supports RDMA Read: %v", dev.SupportsRDMARead)
		t.Logf("  Supports Atomic: %v", dev.SupportsAtomic)
		t.Logf("  Max QP: %d", dev.MaxQP)
		t.Logf("  Max CQ: %d", dev.MaxCQ)
	}
}

// TestRDMAInitialization tests RDMA context initialization
func TestRDMAInitialization(t *testing.T) {
	if !CheckAvailability() {
		t.Skip("RDMA not available")
	}

	logger, _ := zap.NewDevelopment()
	config := DefaultConfig()

	ctx, err := Initialize(config.DeviceName, config.Port, config.UseEventChannel)
	if err != nil {
		t.Fatalf("Failed to initialize RDMA: %v", err)
	}
	defer ctx.Close()

	if !ctx.IsConnected() {
		t.Log("RDMA context created but not connected (expected)")
	}

	// Get connection info
	info, err := ctx.GetConnInfo()
	if err != nil {
		t.Fatalf("Failed to get connection info: %v", err)
	}

	t.Logf("Local connection info:")
	t.Logf("  LID: %d", info.LID)
	t.Logf("  QP Number: %d", info.QPNum)
	t.Logf("  PSN: %d", info.PSN)
	t.Logf("  GID: %x", info.GID)

	logger.Info("RDMA initialization test passed")
}

// TestRDMAManager tests the high-level RDMA manager
func TestRDMAManager(t *testing.T) {
	if !CheckAvailability() {
		t.Skip("RDMA not available")
	}

	logger, _ := zap.NewDevelopment()
	config := DefaultConfig()

	mgr, err := NewRDMAManager(config, logger)
	if err != nil {
		t.Fatalf("Failed to create RDMA manager: %v", err)
	}
	defer mgr.Close()

	// Test getting local connection info
	localInfo := mgr.GetLocalConnInfo()
	t.Logf("Manager local info: QP=%d, LID=%d", localInfo.QPNum, localInfo.LID)

	// Test statistics
	stats := mgr.GetStats()
	t.Logf("Initial stats: %+v", stats)

	if !mgr.IsConnected() {
		t.Log("Manager not connected (expected without peer)")
	}
}

// TestMemoryRegistration tests memory registration
func TestMemoryRegistration(t *testing.T) {
	if !CheckAvailability() {
		t.Skip("RDMA not available")
	}

	ctx, err := Initialize("", 1, false)
	if err != nil {
		t.Fatalf("Failed to initialize RDMA: %v", err)
	}
	defer ctx.Close()

	// Test registering memory
	buffer := make([]byte, 1024*1024) // 1MB
	err = ctx.RegisterMemory(buffer)
	if err != nil {
		t.Fatalf("Failed to register memory: %v", err)
	}

	// Get buffer info
	addr := ctx.GetBufferAddr()
	rkey := ctx.GetRKey()

	t.Logf("Registered memory: addr=0x%x, rkey=0x%x", addr, rkey)

	if addr == 0 || rkey == 0 {
		t.Error("Invalid buffer address or rkey")
	}

	// Test unregistering memory
	err = ctx.UnregisterMemory()
	if err != nil {
		t.Fatalf("Failed to unregister memory: %v", err)
	}
}

// TestConnInfoJSON tests connection info serialization
func TestConnInfoJSON(t *testing.T) {
	info := ConnInfo{
		LID:   123,
		QPNum: 456,
		PSN:   789,
		GID:   [16]byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
	}

	jsonStr, err := ExchangeConnInfoJSON(info)
	if err != nil {
		t.Fatalf("Failed to marshal conn info: %v", err)
	}

	t.Logf("Serialized conn info: %s", jsonStr)

	parsedInfo, err := ParseConnInfoJSON(jsonStr)
	if err != nil {
		t.Fatalf("Failed to parse conn info: %v", err)
	}

	if parsedInfo.LID != info.LID || parsedInfo.QPNum != info.QPNum || parsedInfo.PSN != info.PSN {
		t.Error("Parsed conn info does not match original")
	}

	t.Log("Connection info JSON serialization test passed")
}

// TestRDMAManagerStats tests statistics tracking
func TestRDMAManagerStats(t *testing.T) {
	if !CheckAvailability() {
		t.Skip("RDMA not available")
	}

	logger, _ := zap.NewDevelopment()
	config := DefaultConfig()

	mgr, err := NewRDMAManager(config, logger)
	if err != nil {
		t.Fatalf("Failed to create RDMA manager: %v", err)
	}
	defer mgr.Close()

	// Get initial stats
	stats := mgr.GetStats()
	t.Logf("Initial statistics:")
	for key, value := range stats {
		t.Logf("  %s: %v", key, value)
	}

	// Verify all expected fields are present
	expectedFields := []string{
		"send_operations", "recv_operations", "write_operations", "read_operations",
		"send_completions", "recv_completions", "send_errors", "recv_errors",
		"bytes_sent", "bytes_received",
		"avg_send_latency_ns", "avg_recv_latency_ns",
		"min_send_latency_ns", "max_send_latency_ns",
		"avg_send_latency_us", "avg_recv_latency_us",
	}

	for _, field := range expectedFields {
		if _, ok := stats[field]; !ok {
			t.Errorf("Missing expected stat field: %s", field)
		}
	}
}

// BenchmarkRDMAInitialization benchmarks RDMA initialization
func BenchmarkRDMAInitialization(b *testing.B) {
	if !CheckAvailability() {
		b.Skip("RDMA not available")
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ctx, err := Initialize("", 1, false)
		if err != nil {
			b.Fatalf("Failed to initialize: %v", err)
		}
		ctx.Close()
	}
}

// BenchmarkMemoryRegistration benchmarks memory registration
func BenchmarkMemoryRegistration(b *testing.B) {
	if !CheckAvailability() {
		b.Skip("RDMA not available")
	}

	ctx, err := Initialize("", 1, false)
	if err != nil {
		b.Fatalf("Failed to initialize: %v", err)
	}
	defer ctx.Close()

	buffer := make([]byte, 1024*1024) // 1MB

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := ctx.RegisterMemory(buffer); err != nil {
			b.Fatalf("Failed to register: %v", err)
		}
		ctx.UnregisterMemory()
	}
}

// MockRDMATest tests fallback behavior when RDMA is not available
func TestMockRDMAFallback(t *testing.T) {
	// This test always runs, even without RDMA hardware
	available := CheckAvailability()
	t.Logf("RDMA availability check returned: %v", available)

	if !available {
		t.Log("RDMA not available - TCP fallback will be used (expected behavior)")
	} else {
		t.Log("RDMA available - hardware acceleration will be used")
	}

	// Test that we handle the case gracefully
	_, err := GetDeviceList()
	if !available {
		// Should return empty list, not error
		if err != nil {
			t.Logf("Warning: GetDeviceList returned error when RDMA not available: %v", err)
		}
	}
}

// TestRDMALatencyTracking tests latency measurement
func TestRDMALatencyTracking(t *testing.T) {
	if !CheckAvailability() {
		t.Skip("RDMA not available")
	}

	logger, _ := zap.NewDevelopment()
	config := DefaultConfig()

	mgr, err := NewRDMAManager(config, logger)
	if err != nil {
		t.Fatalf("Failed to create RDMA manager: %v", err)
	}
	defer mgr.Close()

	// Simulate latency update
	mgr.updateSendLatency(500) // 500ns
	mgr.updateSendLatency(600)
	mgr.updateSendLatency(700)

	stats := mgr.GetStats()
	avgLatencyNs := stats["avg_send_latency_ns"].(uint64)
	minLatencyNs := stats["min_send_latency_ns"].(uint64)
	maxLatencyNs := stats["max_send_latency_ns"].(uint64)

	t.Logf("Latency stats: avg=%dns, min=%dns, max=%dns", avgLatencyNs, minLatencyNs, maxLatencyNs)

	if minLatencyNs > maxLatencyNs {
		t.Error("Min latency should be <= max latency")
	}

	if avgLatencyNs < minLatencyNs || avgLatencyNs > maxLatencyNs {
		t.Error("Average latency should be between min and max")
	}
}

// TestConcurrentRDMAOperations tests concurrent RDMA manager access
func TestConcurrentRDMAOperations(t *testing.T) {
	if !CheckAvailability() {
		t.Skip("RDMA not available")
	}

	logger, _ := zap.NewDevelopment()
	config := DefaultConfig()

	mgr, err := NewRDMAManager(config, logger)
	if err != nil {
		t.Fatalf("Failed to create RDMA manager: %v", err)
	}
	defer mgr.Close()

	// Spawn multiple goroutines accessing stats concurrently
	done := make(chan bool)
	for i := 0; i < 10; i++ {
		go func() {
			for j := 0; j < 100; j++ {
				_ = mgr.GetStats()
				_ = mgr.IsConnected()
				_ = mgr.GetLocalConnInfo()
				time.Sleep(time.Microsecond)
			}
			done <- true
		}()
	}

	// Wait for all goroutines
	for i := 0; i < 10; i++ {
		<-done
	}

	t.Log("Concurrent access test passed")
}
