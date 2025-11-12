// Package metrics provides examples for instrumenting DWCP components
package metrics_test

import (
	"fmt"
	"time"

	"novacron/backend/core/network/dwcp/metrics"
)

// ExampleInitializeMetrics shows how to initialize the metrics system
func ExampleInitializeMetrics() {
	// Initialize metrics with cluster name, node name, and port
	err := metrics.InitializeMetrics("production-cluster", "node-01", 9090)
	if err != nil {
		panic(err)
	}
	defer metrics.ShutdownMetrics()

	// Set version information (call once at startup)
	metrics.SetVersionInfo("production-cluster", "1.0.0", "abc123def456", "2025-11-08")

	// Metrics are now being collected and exposed at :9090/metrics
	fmt.Println("Metrics initialized")
	// Output: Metrics initialized
}

// ExampleAMSTMetricsWrapper shows how to instrument AMST streams
func ExampleAMSTMetricsWrapper() {
	// Initialize metrics first
	metrics.InitializeMetrics("cluster-1", "node-01", 9090)
	defer metrics.ShutdownMetrics()

	// Create AMST metrics wrapper
	amstMetrics := metrics.NewAMSTMetricsWrapper()

	// Start a new stream
	streamID := "stream-12345"
	amstMetrics.OnStreamStart(streamID)

	// Simulate data transfer
	for i := 0; i < 10; i++ {
		bytesSent := int64(1024 * 1024) // 1 MB
		bytesReceived := int64(512 * 1024) // 512 KB
		amstMetrics.OnStreamData(streamID, bytesSent, bytesReceived)
		time.Sleep(100 * time.Millisecond)
	}

	// Update bandwidth utilization
	usedBandwidth := int64(100 * 1024 * 1024) // 100 MB/s
	availableBandwidth := int64(1024 * 1024 * 1024) // 1 GB/s
	amstMetrics.OnBandwidthUpdate(usedBandwidth, availableBandwidth)

	// End the stream
	amstMetrics.OnStreamEnd(streamID)

	fmt.Println("AMST stream metrics recorded")
	// Output: AMST stream metrics recorded
}

// ExampleAMSTMetricsWrapper_withErrors shows how to record stream errors
func ExampleAMSTMetricsWrapper_withErrors() {
	metrics.InitializeMetrics("cluster-1", "node-01", 9090)
	defer metrics.ShutdownMetrics()

	amstMetrics := metrics.NewAMSTMetricsWrapper()
	streamID := "stream-error-test"

	amstMetrics.OnStreamStart(streamID)

	// Simulate various error types
	amstMetrics.OnStreamError(streamID, "connection_timeout")
	amstMetrics.OnStreamError(streamID, "network_unreachable")
	amstMetrics.OnStreamError(streamID, "stream_reset")

	amstMetrics.OnStreamEnd(streamID)

	fmt.Println("Stream errors recorded")
	// Output: Stream errors recorded
}

// ExampleHDEMetricsWrapper shows how to instrument HDE compression
func ExampleHDEMetricsWrapper() {
	metrics.InitializeMetrics("cluster-1", "node-01", 9090)
	defer metrics.ShutdownMetrics()

	// Create HDE metrics wrapper
	hdeMetrics := metrics.NewHDEMetricsWrapper()

	// Record successful compression with delta encoding
	originalSize := int64(10 * 1024 * 1024) // 10 MB
	compressedSize := int64(1 * 1024 * 1024) // 1 MB (10x compression)
	deltaHit := true // Delta encoding was used

	hdeMetrics.OnCompressionComplete("vm_memory", originalSize, compressedSize, deltaHit)

	// Record successful decompression
	hdeMetrics.OnDecompressionComplete(true)

	// Update baseline count
	hdeMetrics.OnBaselineUpdate("vm_state", 42)

	// Update dictionary efficiency (75% improvement)
	hdeMetrics.OnDictionaryUpdate(75.0)

	fmt.Println("HDE compression metrics recorded")
	// Output: HDE compression metrics recorded
}

// ExampleHDEMetricsWrapper_differentDataTypes shows compression for different data types
func ExampleHDEMetricsWrapper_differentDataTypes() {
	metrics.InitializeMetrics("cluster-1", "node-01", 9090)
	defer metrics.ShutdownMetrics()

	hdeMetrics := metrics.NewHDEMetricsWrapper()

	// VM Memory compression (highly compressible)
	hdeMetrics.OnCompressionComplete("vm_memory", 100*1024*1024, 10*1024*1024, true)

	// Disk state compression (moderately compressible)
	hdeMetrics.OnCompressionComplete("disk_state", 500*1024*1024, 100*1024*1024, true)

	// Network state compression (less compressible)
	hdeMetrics.OnCompressionComplete("network_state", 10*1024*1024, 5*1024*1024, false)

	fmt.Println("Multiple data type compressions recorded")
	// Output: Multiple data type compressions recorded
}

// ExampleMigrationMetricsWrapper shows how to track VM migrations
func ExampleMigrationMetricsWrapper() {
	metrics.InitializeMetrics("cluster-1", "node-01", 9090)
	defer metrics.ShutdownMetrics()

	// Create migration metrics wrapper
	migrationMetrics := metrics.NewMigrationMetricsWrapper()

	// Start migration tracking
	migrationID := "migration-vm-1234"
	migrationMetrics.OnMigrationStart(migrationID)

	// Simulate migration work
	time.Sleep(2 * time.Second)

	// Complete migration
	destNode := "node-02"
	dwcpEnabled := true
	migrationMetrics.OnMigrationComplete(migrationID, destNode, dwcpEnabled)

	// Record speedup factor (DWCP was 8.5x faster)
	migrationMetrics.OnSpeedupCalculated("large_vm", 8.5)

	fmt.Println("Migration metrics recorded")
	// Output: Migration metrics recorded
}

// ExampleMigrationMetricsWrapper_comparison shows DWCP vs standard migration
func ExampleMigrationMetricsWrapper_comparison() {
	metrics.InitializeMetrics("cluster-1", "node-01", 9090)
	defer metrics.ShutdownMetrics()

	migrationMetrics := metrics.NewMigrationMetricsWrapper()

	// Track standard migration (DWCP disabled)
	standardMigrationID := "migration-standard-001"
	migrationMetrics.OnMigrationStart(standardMigrationID)
	time.Sleep(10 * time.Second) // Slower migration
	migrationMetrics.OnMigrationComplete(standardMigrationID, "node-02", false)

	// Track DWCP-enabled migration
	dwcpMigrationID := "migration-dwcp-001"
	migrationMetrics.OnMigrationStart(dwcpMigrationID)
	time.Sleep(1 * time.Second) // Much faster with DWCP
	migrationMetrics.OnMigrationComplete(dwcpMigrationID, "node-02", true)

	// Record the speedup (10x faster)
	migrationMetrics.OnSpeedupCalculated("standard_vm", 10.0)

	fmt.Println("Migration comparison recorded")
	// Output: Migration comparison recorded
}

// ExampleSystemMetricsWrapper shows how to track system health
func ExampleSystemMetricsWrapper() {
	metrics.InitializeMetrics("cluster-1", "node-01", 9090)
	defer metrics.ShutdownMetrics()

	// Create system metrics wrapper
	systemMetrics := metrics.NewSystemMetricsWrapper()

	// Report component health
	systemMetrics.OnComponentHealthChange("amst", metrics.HealthHealthy)
	systemMetrics.OnComponentHealthChange("hde", metrics.HealthHealthy)
	systemMetrics.OnComponentHealthChange("dwcp_manager", metrics.HealthHealthy)

	// Enable features
	systemMetrics.OnFeatureToggle("amst_enabled", true)
	systemMetrics.OnFeatureToggle("hde_enabled", true)
	systemMetrics.OnFeatureToggle("auto_tuning", true)

	fmt.Println("System metrics recorded")
	// Output: System metrics recorded
}

// ExampleSystemMetricsWrapper_degradedState shows handling degraded components
func ExampleSystemMetricsWrapper_degradedState() {
	metrics.InitializeMetrics("cluster-1", "node-01", 9090)
	defer metrics.ShutdownMetrics()

	systemMetrics := metrics.NewSystemMetricsWrapper()

	// AMST component is healthy
	systemMetrics.OnComponentHealthChange("amst", metrics.HealthHealthy)

	// HDE component is degraded (running but not optimal)
	systemMetrics.OnComponentHealthChange("hde", metrics.HealthDegraded)

	// DWCP manager is down
	systemMetrics.OnComponentHealthChange("dwcp_manager", metrics.HealthDown)

	fmt.Println("Component health states recorded")
	// Output: Component health states recorded
}

// ExampleCollector_multipleOperations shows comprehensive metrics collection
func ExampleCollector_multipleOperations() {
	metrics.InitializeMetrics("cluster-1", "node-01", 9090)
	defer metrics.ShutdownMetrics()

	collector := metrics.GetCollector()

	// Start multiple streams
	for i := 0; i < 5; i++ {
		streamID := fmt.Sprintf("stream-%d", i)
		collector.StartStream(streamID)
		collector.RecordStreamData(streamID, 1024*1024, 512*1024)
	}

	// Record compression operations
	collector.RecordCompressionOperation("vm_memory", 100*1024*1024, 10*1024*1024, true)
	collector.RecordCompressionOperation("disk_state", 200*1024*1024, 30*1024*1024, true)

	// Update system metrics
	collector.UpdateBandwidthUtilization(500*1024*1024, 1024*1024*1024)
	collector.UpdateBaselineCount("vm_state", 15)
	collector.UpdateDictionaryEfficiency(82.5)

	// Record migration
	migrationID := "migration-001"
	collector.StartMigration(migrationID)
	time.Sleep(500 * time.Millisecond)
	collector.EndMigration(migrationID, "node-02", true)

	// End streams
	for i := 0; i < 5; i++ {
		streamID := fmt.Sprintf("stream-%d", i)
		collector.EndStream(streamID)
	}

	fmt.Printf("Active streams: %d\n", collector.GetActiveStreamCount())
	fmt.Printf("Delta hit rate: %.2f%%\n", collector.GetDeltaHitRate())

	// Output:
	// Active streams: 0
	// Delta hit rate: 100.00%
}

// ExampleRecordStreamMetrics shows convenience function usage
func ExampleRecordStreamMetrics() {
	metrics.InitializeMetrics("cluster-1", "node-01", 9090)
	defer metrics.ShutdownMetrics()

	// Record successful stream data transfer
	metrics.RecordStreamMetrics("stream-001", 1024*1024, 512*1024, "")

	// Record stream with error
	metrics.RecordStreamMetrics("stream-002", 0, 0, "connection_timeout")

	fmt.Println("Stream metrics recorded using convenience function")
	// Output: Stream metrics recorded using convenience function
}

// ExampleRecordCompressionMetrics shows compression convenience function
func ExampleRecordCompressionMetrics() {
	metrics.InitializeMetrics("cluster-1", "node-01", 9090)
	defer metrics.ShutdownMetrics()

	// Record compression with delta hit
	metrics.RecordCompressionMetrics("vm_memory", 50*1024*1024, 5*1024*1024, true)

	// Record compression without delta hit (baseline)
	metrics.RecordCompressionMetrics("vm_memory", 50*1024*1024, 10*1024*1024, false)

	fmt.Println("Compression metrics recorded using convenience function")
	// Output: Compression metrics recorded using convenience function
}

// ExampleRecordComponentHealth shows health convenience function
func ExampleRecordComponentHealth() {
	metrics.InitializeMetrics("cluster-1", "node-01", 9090)
	defer metrics.ShutdownMetrics()

	// Record healthy component
	metrics.RecordComponentHealth("amst", true)

	// Record unhealthy component
	metrics.RecordComponentHealth("hde", false)

	fmt.Println("Component health recorded using convenience function")
	// Output: Component health recorded using convenience function
}
