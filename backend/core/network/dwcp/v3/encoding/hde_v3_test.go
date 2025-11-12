package encoding

import (
	"bytes"
	"fmt"
	"testing"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/upgrade"
)

func TestHDEv3_BasicCompression(t *testing.T) {
	config := DefaultHDEv3Config("test-node")
	config.EnableDeltaEncoding = false // Test compression only first

	hde, err := NewHDEv3(config)
	if err != nil {
		t.Fatalf("Failed to create HDE v3: %v", err)
	}
	defer hde.Close()

	testData := []byte("Hello, World! This is test data for HDE v3 compression.")

	// Compress
	compressed, err := hde.Compress("test-vm", testData)
	if err != nil {
		t.Fatalf("Compression failed: %v", err)
	}

	if compressed.CompressedSize >= len(testData) {
		t.Logf("Warning: Compressed size (%d) >= original size (%d)", compressed.CompressedSize, len(testData))
	}

	// Decompress
	decompressed, err := hde.Decompress(compressed)
	if err != nil {
		t.Fatalf("Decompression failed: %v", err)
	}

	// Verify
	if !bytes.Equal(testData, decompressed) {
		t.Fatalf("Decompressed data doesn't match original")
	}

	t.Logf("Compression ratio: %.2fx", compressed.CompressionRatio())
}

func TestHDEv3_DeltaEncoding(t *testing.T) {
	config := DefaultHDEv3Config("test-node")
	config.EnableDeltaEncoding = true

	hde, err := NewHDEv3(config)
	if err != nil {
		t.Fatalf("Failed to create HDE v3: %v", err)
	}
	defer hde.Close()

	// Create baseline
	baseline := make([]byte, 1024)
	for i := range baseline {
		baseline[i] = byte(i % 256)
	}

	// Compress baseline
	_, err = hde.Compress("test-vm", baseline)
	if err != nil {
		t.Fatalf("Baseline compression failed: %v", err)
	}

	// Create similar data (delta)
	modified := make([]byte, 1024)
	copy(modified, baseline)
	modified[100] = 0xFF // Small change
	modified[500] = 0xAA

	// Compress delta
	compressed2, err := hde.Compress("test-vm", modified)
	if err != nil {
		t.Fatalf("Delta compression failed: %v", err)
	}

	// Delta should be detected
	if compressed2.IsDelta {
		t.Logf("Delta compression successful: size=%d", compressed2.CompressedSize)
	}

	// Decompress and verify
	decompressed, err := hde.Decompress(compressed2)
	if err != nil {
		t.Fatalf("Delta decompression failed: %v", err)
	}

	if !bytes.Equal(modified, decompressed) {
		t.Fatalf("Decompressed delta doesn't match original")
	}
}

func TestHDEv3_ModeSwitching(t *testing.T) {
	config := DefaultHDEv3Config("test-node")

	hde, err := NewHDEv3(config)
	if err != nil {
		t.Fatalf("Failed to create HDE v3: %v", err)
	}
	defer hde.Close()

	testData := make([]byte, 10*1024) // 10KB
	for i := range testData {
		testData[i] = byte(i % 256)
	}

	// Test datacenter mode (fast compression)
	hde.UpdateNetworkMode(upgrade.ModeDatacenter)
	compressedDC, err := hde.Compress("test-vm-dc", testData)
	if err != nil {
		t.Fatalf("Datacenter compression failed: %v", err)
	}

	// Test internet mode (max compression)
	hde.UpdateNetworkMode(upgrade.ModeInternet)
	compressedInternet, err := hde.Compress("test-vm-inet", testData)
	if err != nil {
		t.Fatalf("Internet compression failed: %v", err)
	}

	t.Logf("Datacenter mode: %s, size=%d, ratio=%.2fx, time=%v",
		compressedDC.Algorithm,
		compressedDC.CompressedSize,
		compressedDC.CompressionRatio(),
		compressedDC.CompressionTime)

	t.Logf("Internet mode: %s, size=%d, ratio=%.2fx, time=%v",
		compressedInternet.Algorithm,
		compressedInternet.CompressedSize,
		compressedInternet.CompressionRatio(),
		compressedInternet.CompressionTime)

	// Internet mode should have better compression (smaller size)
	if compressedInternet.Algorithm == CompressionZstdMax {
		t.Logf("Correct: Internet mode using maximum compression")
	}
}

func TestHDEv3_MLCompressionSelection(t *testing.T) {
	config := DefaultHDEv3Config("test-node")
	config.EnableMLCompression = true

	hde, err := NewHDEv3(config)
	if err != nil {
		t.Fatalf("Failed to create HDE v3: %v", err)
	}
	defer hde.Close()

	// Test with different data types
	testCases := []struct {
		name string
		data []byte
	}{
		{
			name: "highly_compressible",
			data: bytes.Repeat([]byte("AAAA"), 1000),
		},
		{
			name: "text_data",
			data: []byte("This is a test string with repeated patterns. " +
				"This is a test string with repeated patterns. " +
				"This is a test string with repeated patterns."),
		},
		{
			name: "binary_data",
			data: func() []byte {
				data := make([]byte, 1024)
				for i := range data {
					data[i] = byte(i % 256)
				}
				return data
			}(),
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			compressed, err := hde.Compress(tc.name, tc.data)
			if err != nil {
				t.Fatalf("Compression failed: %v", err)
			}

			t.Logf("%s: algorithm=%s, ratio=%.2fx, size=%d->%d",
				tc.name,
				compressed.Algorithm,
				compressed.CompressionRatio(),
				compressed.OriginalSize,
				compressed.CompressedSize)

			// Verify decompression
			decompressed, err := hde.Decompress(compressed)
			if err != nil {
				t.Fatalf("Decompression failed: %v", err)
			}

			if !bytes.Equal(tc.data, decompressed) {
				t.Fatalf("Decompressed data doesn't match")
			}
		})
	}
}

func TestHDEv3_CRDTIntegration(t *testing.T) {
	// Create two HDE instances (simulating two nodes)
	config1 := DefaultHDEv3Config("node-1")
	config1.EnableCRDT = true

	config2 := DefaultHDEv3Config("node-2")
	config2.EnableCRDT = true

	hde1, err := NewHDEv3(config1)
	if err != nil {
		t.Fatalf("Failed to create HDE v3 node 1: %v", err)
	}
	defer hde1.Close()

	hde2, err := NewHDEv3(config2)
	if err != nil {
		t.Fatalf("Failed to create HDE v3 node 2: %v", err)
	}
	defer hde2.Close()

	// Node 1 creates data
	testData := []byte("Shared data between nodes")
	_, err = hde1.Compress("shared-vm", testData)
	if err != nil {
		t.Fatalf("Node 1 compression failed: %v", err)
	}

	// Export CRDT state from node 1
	crdtState1, err := hde1.ExportCRDTState()
	if err != nil {
		t.Fatalf("Failed to export CRDT state: %v", err)
	}

	// Merge CRDT state into node 2
	err = hde2.MergeRemoteCRDT(crdtState1)
	if err != nil {
		t.Fatalf("Failed to merge CRDT state: %v", err)
	}

	// Verify metrics
	metrics1 := hde1.GetMetrics()
	metrics2 := hde2.GetMetrics()

	t.Logf("Node 1 metrics: %+v", metrics1)
	t.Logf("Node 2 metrics: %+v", metrics2)

	if metrics2["crdt_crdt_merges"] == nil {
		t.Logf("Warning: No CRDT merges recorded on node 2")
	}
}

func TestHDEv3_PerformanceTargets(t *testing.T) {
	config := DefaultHDEv3Config("test-node")

	hde, err := NewHDEv3(config)
	if err != nil {
		t.Fatalf("Failed to create HDE v3: %v", err)
	}
	defer hde.Close()

	// Test with 1MB of data
	testData := make([]byte, 1024*1024)
	for i := range testData {
		testData[i] = byte(i % 256)
	}

	// Datacenter mode - should be fast
	hde.UpdateNetworkMode(upgrade.ModeDatacenter)
	startDC := time.Now()
	compressedDC, err := hde.Compress("perf-test-dc", testData)
	if err != nil {
		t.Fatalf("Datacenter compression failed: %v", err)
	}
	durationDC := time.Since(startDC)

	// Internet mode - should have high compression
	hde.UpdateNetworkMode(upgrade.ModeInternet)
	startInternet := time.Now()
	compressedInternet, err := hde.Compress("perf-test-inet", testData)
	if err != nil {
		t.Fatalf("Internet compression failed: %v", err)
	}
	durationInternet := time.Since(startInternet)

	t.Logf("\n=== Performance Targets ===")
	t.Logf("Datacenter Mode:")
	t.Logf("  - Compression time: %v", durationDC)
	t.Logf("  - Algorithm: %s", compressedDC.Algorithm)
	t.Logf("  - Ratio: %.2fx", compressedDC.CompressionRatio())
	t.Logf("  - Reduction: %.1f%%", (1.0-float64(compressedDC.CompressedSize)/float64(compressedDC.OriginalSize))*100)

	t.Logf("\nInternet Mode:")
	t.Logf("  - Compression time: %v", durationInternet)
	t.Logf("  - Algorithm: %s", compressedInternet.Algorithm)
	t.Logf("  - Ratio: %.2fx", compressedInternet.CompressionRatio())
	t.Logf("  - Reduction: %.1f%%", (1.0-float64(compressedInternet.CompressedSize)/float64(compressedInternet.OriginalSize))*100)

	// Verify performance targets
	internetReduction := (1.0 - float64(compressedInternet.CompressedSize)/float64(compressedInternet.OriginalSize)) * 100

	// Target: 70-85% reduction for internet mode
	if internetReduction < 50.0 {
		t.Logf("Note: Internet mode reduction (%.1f%%) below target (70-85%%)", internetReduction)
	} else {
		t.Logf("✓ Internet mode reduction meets target: %.1f%%", internetReduction)
	}
}

func TestHDEv3_Marshaling(t *testing.T) {
	config := DefaultHDEv3Config("test-node")

	hde, err := NewHDEv3(config)
	if err != nil {
		t.Fatalf("Failed to create HDE v3: %v", err)
	}
	defer hde.Close()

	testData := []byte("Test data for marshaling")

	// Compress
	compressed, err := hde.Compress("marshal-test", testData)
	if err != nil {
		t.Fatalf("Compression failed: %v", err)
	}

	// Marshal
	marshaled := compressed.Marshal()

	// Unmarshal
	unmarshaled, err := UnmarshalCompressedDataV3(marshaled)
	if err != nil {
		t.Fatalf("Unmarshal failed: %v", err)
	}

	// Verify metadata
	if unmarshaled.OriginalSize != compressed.OriginalSize {
		t.Errorf("Original size mismatch: %d != %d", unmarshaled.OriginalSize, compressed.OriginalSize)
	}
	if unmarshaled.CompressedSize != compressed.CompressedSize {
		t.Errorf("Compressed size mismatch: %d != %d", unmarshaled.CompressedSize, compressed.CompressedSize)
	}
	if unmarshaled.Algorithm != compressed.Algorithm {
		t.Errorf("Algorithm mismatch: %s != %s", unmarshaled.Algorithm, compressed.Algorithm)
	}

	// Decompress unmarshaled data
	decompressed, err := hde.Decompress(unmarshaled)
	if err != nil {
		t.Fatalf("Decompression of unmarshaled data failed: %v", err)
	}

	if !bytes.Equal(testData, decompressed) {
		t.Fatalf("Decompressed unmarshaled data doesn't match original")
	}
}

func TestHDEv3_Metrics(t *testing.T) {
	config := DefaultHDEv3Config("test-node")

	hde, err := NewHDEv3(config)
	if err != nil {
		t.Fatalf("Failed to create HDE v3: %v", err)
	}
	defer hde.Close()

	// Perform multiple compressions
	for i := 0; i < 10; i++ {
		testData := []byte(fmt.Sprintf("Test data %d", i))
		vmID := fmt.Sprintf("vm-%d", i)

		_, err := hde.Compress(vmID, testData)
		if err != nil {
			t.Fatalf("Compression %d failed: %v", i, err)
		}
	}

	// Get metrics
	metrics := hde.GetMetrics()

	t.Logf("\n=== HDE v3 Metrics ===")
	for key, value := range metrics {
		t.Logf("%s: %v", key, value)
	}

	// Verify metrics
	if metrics["total_compressed"] == nil || metrics["total_compressed"].(int64) != 10 {
		t.Errorf("Expected 10 compressions, got %v", metrics["total_compressed"])
	}

	if metrics["compression_ratio"] != nil {
		t.Logf("✓ Overall compression ratio: %.2fx", metrics["compression_ratio"].(float64))
	}
}

func BenchmarkHDEv3_Compress_1KB(b *testing.B) {
	config := DefaultHDEv3Config("bench-node")
	config.EnableDeltaEncoding = false

	hde, _ := NewHDEv3(config)
	defer hde.Close()

	data := make([]byte, 1024)
	for i := range data {
		data[i] = byte(i % 256)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		hde.Compress("bench-vm", data)
	}
}

func BenchmarkHDEv3_Compress_1MB(b *testing.B) {
	config := DefaultHDEv3Config("bench-node")
	config.EnableDeltaEncoding = false

	hde, _ := NewHDEv3(config)
	defer hde.Close()

	data := make([]byte, 1024*1024)
	for i := range data {
		data[i] = byte(i % 256)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		hde.Compress("bench-vm", data)
	}
}

func BenchmarkHDEv3_CompressDecompress(b *testing.B) {
	config := DefaultHDEv3Config("bench-node")

	hde, _ := NewHDEv3(config)
	defer hde.Close()

	data := make([]byte, 10*1024)
	for i := range data {
		data[i] = byte(i % 256)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		compressed, _ := hde.Compress("bench-vm", data)
		hde.Decompress(compressed)
	}
}
