package compression

import (
	"bytes"
	"compress/gzip"
	"compress/lzw"
	"compress/zlib"
	"io"
	"io/ioutil"
)

// CompressionAlgorithm represents the algorithm used for compressing data
type CompressionAlgorithm string

const (
	// CompressionNone indicates no compression should be used
	CompressionNone CompressionAlgorithm = "none"

	// CompressionGzip uses the gzip compression algorithm
	CompressionGzip CompressionAlgorithm = "gzip"

	// CompressionZlib uses the zlib compression algorithm
	CompressionZlib CompressionAlgorithm = "zlib"

	// CompressionLZW uses the LZW compression algorithm
	CompressionLZW CompressionAlgorithm = "lzw"
)

// CompressionLevel defines the level of compression to use
type CompressionLevel int

const (
	// CompressionDefault uses the default compression level of the algorithm
	CompressionDefault CompressionLevel = 0

	// CompressionFastest optimizes for speed at the expense of compression ratio
	CompressionFastest CompressionLevel = 1

	// CompressionBest optimizes for compression ratio at the expense of speed
	CompressionBest CompressionLevel = 9
)

// CompressionConfig contains configuration for data compression
type CompressionConfig struct {
	// Algorithm to use for compression
	Algorithm CompressionAlgorithm `json:"algorithm"`

	// Compression level to use
	Level CompressionLevel `json:"level"`

	// Minimum size in bytes for data to be compressed
	MinSizeBytes int `json:"min_size_bytes"`

	// Maximum size in bytes for data to be compressed
	MaxSizeBytes int `json:"max_size_bytes"`

	// Whether to auto-detect if compression would be beneficial
	AutoDetect bool `json:"auto_detect"`

	// Whether to maintain the original data alongside the compressed data
	KeepOriginal bool `json:"keep_original"`
}

// DefaultCompressionConfig returns a default compression configuration
func DefaultCompressionConfig() CompressionConfig {
	return CompressionConfig{
		Algorithm:    CompressionGzip,
		Level:        CompressionDefault,
		MinSizeBytes: 4 * 1024,         // 4KB
		MaxSizeBytes: 1024 * 1024 * 32, // 32MB
		AutoDetect:   true,
		KeepOriginal: false,
	}
}

// Compressor provides methods for compressing and decompressing data
type Compressor struct {
	config CompressionConfig
}

// NewCompressor creates a new Compressor with the provided configuration
func NewCompressor(config CompressionConfig) *Compressor {
	return &Compressor{
		config: config,
	}
}

// ShouldCompress determines if data should be compressed based on config and content
func (c *Compressor) ShouldCompress(data []byte) bool {
	// Check size constraints
	size := len(data)
	if size < c.config.MinSizeBytes || (c.config.MaxSizeBytes > 0 && size > c.config.MaxSizeBytes) {
		return false
	}

	// If auto-detect is enabled, do a quick compression test on a sample
	if c.config.AutoDetect && size > 1024 {
		// Take a sample of the data (first 1KB)
		sampleSize := 1024
		if size < sampleSize {
			sampleSize = size
		}
		sample := data[:sampleSize]

		// Try compressing the sample
		compressed, err := c.Compress(sample)
		if err != nil {
			return false
		}

		// If compressed size is at least 10% smaller, compression is worthwhile
		return len(compressed) < int(float64(sampleSize)*0.9)
	}

	// Default to compressing if it meets size requirements
	return true
}

// Compress compresses the provided data using the configured algorithm and level
func (c *Compressor) Compress(data []byte) ([]byte, error) {
	if c.config.Algorithm == CompressionNone {
		return data, nil
	}

	var buf bytes.Buffer
	var compressor io.WriteCloser
	var err error

	// Create the appropriate compressor
	switch c.config.Algorithm {
	case CompressionGzip:
		compressor, err = gzip.NewWriterLevel(&buf, int(c.config.Level))
	case CompressionZlib:
		compressor, err = zlib.NewWriterLevel(&buf, int(c.config.Level))
	case CompressionLZW:
		compressor = lzw.NewWriter(&buf, lzw.MSB, 8)
	default:
		// Default to gzip
		compressor, err = gzip.NewWriterLevel(&buf, int(c.config.Level))
	}

	if err != nil {
		return nil, err
	}

	// Write data to the compressor
	_, err = compressor.Write(data)
	if err != nil {
		compressor.Close()
		return nil, err
	}

	// Close the compressor to flush any remaining data
	err = compressor.Close()
	if err != nil {
		return nil, err
	}

	return buf.Bytes(), nil
}

// Decompress decompresses the provided data using the specified algorithm
func (c *Compressor) Decompress(data []byte, algorithm CompressionAlgorithm) ([]byte, error) {
	if algorithm == CompressionNone {
		return data, nil
	}

	var decompressor io.ReadCloser
	var err error

	// Create the appropriate decompressor
	buf := bytes.NewBuffer(data)
	switch algorithm {
	case CompressionGzip:
		decompressor, err = gzip.NewReader(buf)
	case CompressionZlib:
		decompressor, err = zlib.NewReader(buf)
	case CompressionLZW:
		decompressor = lzw.NewReader(buf, lzw.MSB, 8)
	default:
		// Default to gzip
		decompressor, err = gzip.NewReader(buf)
	}

	if err != nil {
		return nil, err
	}
	defer decompressor.Close()

	// Read decompressed data
	result, err := ioutil.ReadAll(decompressor)
	if err != nil {
		return nil, err
	}

	return result, nil
}

// CompressedData represents data that has been compressed
type CompressedData struct {
	// The compressed data bytes
	Data []byte `json:"data"`

	// The algorithm used for compression
	Algorithm CompressionAlgorithm `json:"algorithm"`

	// The original size of the data before compression
	OriginalSize int `json:"original_size"`

	// The compression ratio (original size / compressed size)
	CompressionRatio float64 `json:"compression_ratio"`
}

// CompressWithMetadata compresses data and returns metadata about the compression
func (c *Compressor) CompressWithMetadata(data []byte) (*CompressedData, error) {
	// If shouldn't compress, return original with none algorithm
	if !c.ShouldCompress(data) {
		return &CompressedData{
			Data:             data,
			Algorithm:        CompressionNone,
			OriginalSize:     len(data),
			CompressionRatio: 1.0,
		}, nil
	}

	// Compress the data
	compressed, err := c.Compress(data)
	if err != nil {
		return nil, err
	}

	// Calculate compression ratio
	originalSize := len(data)
	compressedSize := len(compressed)
	ratio := float64(originalSize) / float64(compressedSize)

	return &CompressedData{
		Data:             compressed,
		Algorithm:        c.config.Algorithm,
		OriginalSize:     originalSize,
		CompressionRatio: ratio,
	}, nil
}

// Decompress decompresses the CompressedData and returns the original data
func (cd *CompressedData) Decompress(c *Compressor) ([]byte, error) {
	return c.Decompress(cd.Data, cd.Algorithm)
}
