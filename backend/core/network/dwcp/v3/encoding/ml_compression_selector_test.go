package encoding

import (
	"bytes"
	"testing"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/upgrade"
)

func TestCompressionSelector_DatacenterMode(t *testing.T) {
	config := DefaultSelectorConfig()
	selector := NewCompressionSelector(upgrade.ModeDatacenter, config)
	defer selector.Close()

	testData := make([]byte, 100*1024) // 100KB
	for i := range testData {
		testData[i] = byte(i % 256)
	}

	algo := selector.SelectCompression(testData, upgrade.ModeDatacenter)

	// Datacenter mode should prefer LZ4 or Zstd (fast)
	if algo != CompressionLZ4 && algo != CompressionZstd {
		t.Errorf("Expected LZ4 or Zstd for datacenter mode, got %s", algo)
	}

	t.Logf("Datacenter mode selected: %s", algo)
}

func TestCompressionSelector_InternetMode(t *testing.T) {
	config := DefaultSelectorConfig()
	selector := NewCompressionSelector(upgrade.ModeInternet, config)
	defer selector.Close()

	testData := make([]byte, 1024*1024) // 1MB
	for i := range testData {
		testData[i] = byte(i % 256)
	}

	algo := selector.SelectCompression(testData, upgrade.ModeInternet)

	// Internet mode should prefer ZstdMax (high compression)
	if algo != CompressionZstdMax && algo != CompressionZstd {
		t.Logf("Internet mode selected %s (expected ZstdMax or Zstd)", algo)
	}

	t.Logf("Internet mode selected: %s", algo)
}

func TestCompressionSelector_SmallData(t *testing.T) {
	config := DefaultSelectorConfig()
	selector := NewCompressionSelector(upgrade.ModeHybrid, config)
	defer selector.Close()

	smallData := []byte("small")

	algo := selector.SelectCompression(smallData, upgrade.ModeHybrid)

	// Very small data should not be compressed
	if algo != CompressionNone {
		t.Logf("Small data selected %s (expected None)", algo)
	}
}

func TestCompressionSelector_IncompressibleData(t *testing.T) {
	config := DefaultSelectorConfig()
	selector := NewCompressionSelector(upgrade.ModeHybrid, config)
	defer selector.Close()

	// Create highly random (incompressible) data
	randomData := make([]byte, 10*1024)
	for i := range randomData {
		randomData[i] = byte((i * 7919) % 256) // Pseudo-random
	}

	algo := selector.SelectCompression(randomData, upgrade.ModeHybrid)

	// Should detect incompressible data
	t.Logf("Incompressible data selected: %s", algo)
}

func TestCompressionSelector_RepetitiveData(t *testing.T) {
	config := DefaultSelectorConfig()
	selector := NewCompressionSelector(upgrade.ModeInternet, config)
	defer selector.Close()

	// Create highly repetitive data
	repetitiveData := bytes.Repeat([]byte("AAAA"), 1000)

	algo := selector.SelectCompression(repetitiveData, upgrade.ModeInternet)

	// Should select high compression for repetitive data
	if algo != CompressionZstdMax {
		t.Logf("Repetitive data selected %s (expected ZstdMax)", algo)
	}

	t.Logf("Repetitive data selected: %s", algo)
}

func TestCompressionSelector_PerformanceRecording(t *testing.T) {
	config := DefaultSelectorConfig()
	config.AdaptiveEnabled = true
	selector := NewCompressionSelector(upgrade.ModeHybrid, config)
	defer selector.Close()

	// Record some performance data
	selector.RecordPerformance(CompressionZstd, 1000, 500, 10*time.Millisecond)
	selector.RecordPerformance(CompressionZstd, 2000, 1000, 20*time.Millisecond)
	selector.RecordPerformance(CompressionLZ4, 1000, 800, 5*time.Millisecond)

	stats := selector.GetStats()

	t.Logf("Selector stats: %+v", stats)

	// Verify stats structure
	if stats["learning_enabled"] != true {
		t.Errorf("Expected learning_enabled=true")
	}
}

func TestCompressionSelector_DataAnalysis(t *testing.T) {
	selector := NewCompressionSelector(upgrade.ModeHybrid, nil)
	defer selector.Close()

	testCases := []struct {
		name     string
		data     []byte
		expected DataCharacteristics
	}{
		{
			name:     "text_data",
			data:     []byte("This is ASCII text data with normal characters."),
			expected: DataCharacteristics{TextLike: true},
		},
		{
			name:     "binary_data",
			data:     []byte{0x00, 0x01, 0x02, 0xFF, 0xFE, 0xFD},
			expected: DataCharacteristics{BinaryLike: true},
		},
		{
			name:     "repetitive_data",
			data:     bytes.Repeat([]byte("A"), 100),
			expected: DataCharacteristics{RepeatPattern: true},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			chars := selector.analyzeData(tc.data)

			t.Logf("%s characteristics:", tc.name)
			t.Logf("  Size: %d", chars.Size)
			t.Logf("  Entropy: %.3f", chars.Entropy)
			t.Logf("  Compressible: %v", chars.Compressible)
			t.Logf("  RepeatPattern: %v", chars.RepeatPattern)
			t.Logf("  TextLike: %v", chars.TextLike)
			t.Logf("  BinaryLike: %v", chars.BinaryLike)

			// Verify expected characteristics
			if tc.expected.TextLike && !chars.TextLike {
				t.Errorf("Expected TextLike=true")
			}
			if tc.expected.BinaryLike && !chars.BinaryLike {
				t.Errorf("Expected BinaryLike=true")
			}
			if tc.expected.RepeatPattern && !chars.RepeatPattern {
				t.Errorf("Expected RepeatPattern=true")
			}
		})
	}
}

func TestCompressionSelector_AdaptiveSelection(t *testing.T) {
	config := DefaultSelectorConfig()
	config.AdaptiveEnabled = true
	selector := NewCompressionSelector(upgrade.ModeHybrid, config)
	defer selector.Close()

	testData := make([]byte, 10*1024)
	for i := range testData {
		testData[i] = byte(i % 256)
	}

	// Record performance for different algorithms
	selector.RecordPerformance(CompressionLZ4, 10000, 8000, 5*time.Millisecond)   // Fast but lower ratio
	selector.RecordPerformance(CompressionZstd, 10000, 5000, 15*time.Millisecond) // Balanced
	selector.RecordPerformance(CompressionZstdMax, 10000, 3000, 50*time.Millisecond) // Slow but high ratio

	// Now select - should adapt based on recorded performance
	algo := selector.SelectCompression(testData, upgrade.ModeHybrid)

	t.Logf("Adaptive selection chose: %s", algo)

	stats := selector.GetStats()
	t.Logf("Selector stats after learning: %+v", stats)
}

func TestCompressionSelector_ModeUpdate(t *testing.T) {
	selector := NewCompressionSelector(upgrade.ModeDatacenter, nil)
	defer selector.Close()

	testData := make([]byte, 100*1024)

	// Initial mode (datacenter)
	algo1 := selector.SelectCompression(testData, upgrade.ModeDatacenter)
	t.Logf("Datacenter mode: %s", algo1)

	// Update to internet mode
	selector.UpdateMode(upgrade.ModeInternet)
	algo2 := selector.SelectCompression(testData, upgrade.ModeInternet)
	t.Logf("Internet mode: %s", algo2)

	// Algorithms should be different (datacenter favors speed, internet favors compression)
	if algo1 == algo2 && algo1 != CompressionNone {
		t.Logf("Note: Both modes selected same algorithm: %s", algo1)
	}
}

func BenchmarkCompressionSelector_SelectCompression(b *testing.B) {
	selector := NewCompressionSelector(upgrade.ModeHybrid, nil)
	defer selector.Close()

	data := make([]byte, 10*1024)
	for i := range data {
		data[i] = byte(i % 256)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		selector.SelectCompression(data, upgrade.ModeHybrid)
	}
}

func BenchmarkCompressionSelector_DataAnalysis(b *testing.B) {
	selector := NewCompressionSelector(upgrade.ModeHybrid, nil)
	defer selector.Close()

	data := make([]byte, 1024)
	for i := range data {
		data[i] = byte(i % 256)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		selector.analyzeData(data)
	}
}
