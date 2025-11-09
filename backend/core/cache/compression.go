package cache

import (
	"bytes"
	"compress/gzip"
	"io"
)

// CompressionEngine handles compression and decompression
type CompressionEngine struct {
	config *CacheConfig
}

// NewCompressionEngine creates a new compression engine
func NewCompressionEngine(config *CacheConfig) *CompressionEngine {
	return &CompressionEngine{
		config: config,
	}
}

// Compress compresses data
func (ce *CompressionEngine) Compress(data []byte) ([]byte, float64, error) {
	if len(data) == 0 {
		return data, 1.0, nil
	}

	var buf bytes.Buffer

	switch ce.config.CompressionAlgo {
	case "gzip", "":
		w, err := gzip.NewWriterLevel(&buf, ce.config.CompressionLevel)
		if err != nil {
			return nil, 0, err
		}

		if _, err := w.Write(data); err != nil {
			return nil, 0, err
		}

		if err := w.Close(); err != nil {
			return nil, 0, err
		}
	default:
		// Fallback to gzip
		w, err := gzip.NewWriterLevel(&buf, ce.config.CompressionLevel)
		if err != nil {
			return nil, 0, err
		}

		if _, err := w.Write(data); err != nil {
			return nil, 0, err
		}

		if err := w.Close(); err != nil {
			return nil, 0, err
		}
	}

	compressed := buf.Bytes()
	ratio := float64(len(data)) / float64(len(compressed))

	return compressed, ratio, nil
}

// Decompress decompresses data
func (ce *CompressionEngine) Decompress(data []byte) ([]byte, error) {
	if len(data) == 0 {
		return data, nil
	}

	buf := bytes.NewReader(data)

	switch ce.config.CompressionAlgo {
	case "gzip", "":
		r, err := gzip.NewReader(buf)
		if err != nil {
			return nil, err
		}
		defer r.Close()

		return io.ReadAll(r)
	default:
		// Fallback to gzip
		r, err := gzip.NewReader(buf)
		if err != nil {
			return nil, err
		}
		defer r.Close()

		return io.ReadAll(r)
	}
}
