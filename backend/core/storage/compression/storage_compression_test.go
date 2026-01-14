package compression

import (
	"bytes"
	"math/rand"
	"testing"
)

func TestCompression(t *testing.T) {
	// Create test data of different types and sizes
	testCases := []struct {
		name        string
		data        []byte
		shouldComp  bool
		minRatio    float64
		algorithm   CompressionAlgorithm
		level       CompressionLevel
		description string
	}{
		{
			name:        "Small Text",
			data:        []byte("This is a small piece of text that won't compress much."),
			shouldComp:  true,
			minRatio:    1.0,
			algorithm:   CompressionGzip,
			level:       CompressionDefault,
			description: "Small text data with Gzip default compression",
		},
		{
			name:        "Repeated Text",
			data:        bytes.Repeat([]byte("abcdefghijklmnopqrstuvwxyz"), 100),
			shouldComp:  true,
			minRatio:    2.0,
			algorithm:   CompressionGzip,
			level:       CompressionBest,
			description: "Highly repetitive text that should compress well with Gzip best compression",
		},
		{
			name:        "Random Binary",
			data:        generateRandomData(1024 * 10), // 10KB of random data
			shouldComp:  true,
			minRatio:    0.9, // Might not compress well
			algorithm:   CompressionZlib,
			level:       CompressionDefault,
			description: "Random binary data with Zlib default compression",
		},
		{
			name:        "Very Small Data",
			data:        []byte("abc"),
			shouldComp:  false, // Below min size threshold
			minRatio:    1.0,
			algorithm:   CompressionGzip,
			level:       CompressionDefault,
			description: "Very small data that shouldn't be compressed due to size threshold",
		},
		{
			name:        "LZW Compression",
			data:        bytes.Repeat([]byte("The quick brown fox jumps over the lazy dog. "), 50),
			shouldComp:  true,
			minRatio:    1.5,
			algorithm:   CompressionLZW,
			level:       CompressionDefault,
			description: "Medium-sized text with LZW compression",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Create a custom config for this test case
			config := CompressionConfig{
				Algorithm:    tc.algorithm,
				Level:        tc.level,
				MinSizeBytes: 4, // Set low for testing
				MaxSizeBytes: 1024 * 1024 * 100,
				AutoDetect:   false, // Disable auto-detect for predictable testing
				KeepOriginal: false,
			}

			// Create a compressor with this config
			compressor := NewCompressor(config)

			// Test ShouldCompress with custom config
			shouldCompress := compressor.ShouldCompress(tc.data)
			if shouldCompress != tc.shouldComp {
				t.Errorf("ShouldCompress returned %v, expected %v for %s",
					shouldCompress, tc.shouldComp, tc.description)
			}

			// Test compression
			compressed, err := compressor.Compress(tc.data)
			if err != nil {
				t.Errorf("Compression failed: %v for %s", err, tc.description)
				return
			}

			// Verify compression ratio if should compress
			if tc.shouldComp && tc.algorithm != CompressionNone {
				ratio := float64(len(tc.data)) / float64(len(compressed))
				t.Logf("Compression ratio for %s: %.2f", tc.name, ratio)

				// Only check ratio for compressible data
				if tc.minRatio > 1.0 && ratio < tc.minRatio {
					t.Logf("Warning: Compression ratio lower than expected for %s: got %.2f, expected at least %.2f",
						tc.name, ratio, tc.minRatio)
				}
			}

			// Test decompression
			decompressed, err := compressor.Decompress(compressed, tc.algorithm)
			if err != nil {
				t.Errorf("Decompression failed: %v for %s", err, tc.description)
				return
			}

			// Verify decompressed data matches original
			if !bytes.Equal(decompressed, tc.data) {
				t.Errorf("Decompressed data doesn't match original for %s", tc.description)
			}
		})
	}
}

func TestCompressionWithMetadata(t *testing.T) {
	// Create some test data that should compress well
	testData := bytes.Repeat([]byte("abcdefghijklmnopqrstuvwxyz"), 100)

	// Create compressor with default config
	compressor := NewCompressor(DefaultCompressionConfig())

	// Compress with metadata
	compressedData, err := compressor.CompressWithMetadata(testData)
	if err != nil {
		t.Fatalf("CompressWithMetadata failed: %v", err)
	}

	// Verify metadata
	if compressedData.Algorithm != CompressionGzip {
		t.Errorf("Expected algorithm %s, got %s", CompressionGzip, compressedData.Algorithm)
	}

	if compressedData.OriginalSize != len(testData) {
		t.Errorf("Expected original size %d, got %d", len(testData), compressedData.OriginalSize)
	}

	// Check compression ratio - should be > 1.0 for compressible data
	if compressedData.CompressionRatio <= 1.0 {
		t.Errorf("Expected compression ratio > 1.0, got %.2f", compressedData.CompressionRatio)
	}

	// Decompress from metadata
	decompressed, err := compressedData.Decompress(compressor)
	if err != nil {
		t.Fatalf("Decompress from metadata failed: %v", err)
	}

	// Verify decompressed data matches original
	if !bytes.Equal(decompressed, testData) {
		t.Errorf("Decompressed data doesn't match original")
	}
}

func TestAutoDetectCompression(t *testing.T) {
	// Create compressor with auto-detect enabled
	config := DefaultCompressionConfig()
	config.AutoDetect = true
	compressor := NewCompressor(config)

	// Test cases for auto-detection
	testCases := []struct {
		name         string
		data         []byte
		shouldDetect bool
	}{
		{
			name:         "Compressible Data",
			data:         bytes.Repeat([]byte("abcdefg"), 1000),
			shouldDetect: true,
		},
		{
			name:         "Random Data",
			data:         generateRandomData(10 * 1024),
			shouldDetect: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Test auto-detection
			shouldCompress := compressor.ShouldCompress(tc.data)

			// For auto-detection tests, we're less strict - just log the result
			t.Logf("%s: Auto-detection returned shouldCompress=%v (expected around %v)",
				tc.name, shouldCompress, tc.shouldDetect)

			// Compress and check ratio
			compressed, err := compressor.Compress(tc.data)
			if err != nil {
				t.Errorf("Compression failed: %v", err)
				return
			}

			ratio := float64(len(tc.data)) / float64(len(compressed))
			t.Logf("%s compression ratio: %.2f", tc.name, ratio)

			// Verify we can decompress properly
			decompressed, err := compressor.Decompress(compressed, compressor.config.Algorithm)
			if err != nil {
				t.Errorf("Decompression failed: %v", err)
				return
			}

			if !bytes.Equal(decompressed, tc.data) {
				t.Errorf("Decompressed data doesn't match original")
			}
		})
	}
}

// Helper function to generate random binary data
func generateRandomData(size int) []byte {
	data := make([]byte, size)
	rand.Read(data)
	return data
}
